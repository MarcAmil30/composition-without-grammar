#!/usr/bin/env python3
"""Where did the fine-tuning loss actually go?

The training curve reports ONE number averaged over every token of a record

    <species><delim>CDS<delim>5'UTR<delim>3'UTR        e.g.  +#ATG...+$TTATG...

and `DataCollatorForLanguageModeling(mlm=False)` applies no label mask, so the
coding sequence -- which is far more predictable than regulatory DNA and, at a
median 858 nt against a 200 bp upstream, far longer -- dominates that average.
A falling curve therefore says little about the upstream.

This script splits the same teacher-forced cross-entropy by region:

    species token | delimiters | CDS | 5'UTR body | 5'UTR last 20 bp (SD) | 3'UTR

and reports, per region, the mean CE and the share of TOTAL loss it contributes.
The comparison that matters is the 5'UTR body (what the paper is about) against
the CDS (what the curve is mostly measuring).

Run under the same container/venv as the retrieval jobs; see the .sbatch beside
this file. Writes loss_decomposition.csv.
"""
from __future__ import annotations

# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
WORK_ROOT = _os.environ.get("WORK_ROOT", _os.path.expanduser("~"))
# -------------------------------------------------------------------

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

WORK = f"{WORK_ROOT}"
DEFAULT_CKPT = os.environ.get(
    "PROK_CKPT",
    f"{WORK}/finetuned_evo_lr0.0001_bs2__maxsteps80000"
    "_PROK_combined_species_cds_up_down_v3_TRAIN.csv_patience_2"
    "/final_model_after_max_steps")
DEFAULT_TEST = (f"{WORK}/Proking_repo/data/finetune_sequences_data_small_scale"
                "/PROK_combined_species_cds_up_down_v4_TEST.csv")
MAX_LENGTH = 2500          # as used in training for cds_up_down records
SD_WINDOW = 20             # last 20 bp of the 5'UTR
DELIMS = set("#$*")
LN2 = float(np.log(2.0))


def regions(seq: str) -> np.ndarray:
    """Label every character of a record. Byte-level tokenizer, so 1 char = 1 token.

    Layout is <species><delim>payload<delim>payload...  The species token is the
    first character; '#' opens the CDS, '$' the 5'UTR, '*' the 3'UTR. Records
    that lack a CDS simply start '+$'.
    """
    lab = np.empty(len(seq), dtype=object)
    lab[:] = "other"
    if not seq:
        return lab
    lab[0] = "species"
    cur = "other"
    utr5_span = []
    for i, ch in enumerate(seq):
        if i == 0:
            continue
        if ch in DELIMS:
            lab[i] = "delim"
            cur = {"#": "cds", "$": "utr5", "*": "utr3"}[ch]
            continue
        if ch == seq[0]:              # species token repeated before a delimiter
            lab[i] = "species"
            continue
        lab[i] = cur
        if cur == "utr5":
            utr5_span.append(i)
    # the last SD_WINDOW bases of the 5'UTR abut the start codon
    for i in utr5_span[-SD_WINDOW:]:
        lab[i] = "utr5_sd"
    for i in utr5_span[:-SD_WINDOW]:
        lab[i] = "utr5_body"
    return lab


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--test", default=DEFAULT_TEST)
    ap.add_argument("--limit", type=int, default=2000, help="0 = all records")
    ap.add_argument("--out", default="loss_decomposition.csv")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.ckpt, trust_remote_code=True)
    tok.pad_token = "X"
    model = AutoModelForCausalLM.from_pretrained(
        a.ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()

    df = pd.read_csv(a.test)
    if a.limit:
        df = df.head(a.limit)
    print(f"{len(df)} records from {os.path.basename(a.test)}", flush=True)

    # one token per character is what makes the region labels line up
    probe = df.training_sequence.iloc[0][:64]
    assert len(tok(probe)["input_ids"]) == len(probe), (
        "tokenizer is not one-token-per-character; region labels would be "
        f"misaligned ({len(tok(probe)['input_ids'])} ids for {len(probe)} chars)")

    tot_ce: dict[str, float] = {}
    tot_n: dict[str, int] = {}
    for k, row in enumerate(df.itertuples()):
        seq = str(row.training_sequence)[:MAX_LENGTH]
        ids = torch.tensor(tok(seq)["input_ids"], device="cuda")[None]
        logits = model(ids).logits.float()
        # causal shift: position i predicts token i+1
        ce = F.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="none")
        lab = regions(seq)[1:]                       # align with the shifted targets
        ce = ce.cpu().numpy()[:len(lab)]
        for r in np.unique(lab):
            m = lab == r
            tot_ce[r] = tot_ce.get(r, 0.0) + float(ce[m].sum())
            tot_n[r] = tot_n.get(r, 0) + int(m.sum())
        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{len(df)}", flush=True)

    grand_ce = sum(tot_ce.values())
    grand_n = sum(tot_n.values())
    rows = []
    for r in sorted(tot_n, key=lambda x: -tot_n[x]):
        rows.append(dict(region=r, n_tokens=tot_n[r],
                         tokens_pct=100 * tot_n[r] / grand_n,
                         mean_ce_nats=tot_ce[r] / tot_n[r],
                         mean_ce_bits=tot_ce[r] / tot_n[r] / LN2,
                         loss_share_pct=100 * tot_ce[r] / grand_ce))
    out = pd.DataFrame(rows)
    out.loc[len(out)] = dict(region="ALL", n_tokens=grand_n, tokens_pct=100.0,
                             mean_ce_nats=grand_ce / grand_n,
                             mean_ce_bits=grand_ce / grand_n / LN2,
                             loss_share_pct=100.0)
    out.to_csv(a.out, index=False)
    print()
    print(out.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print(f"\nwrote {a.out}")
    print("\nRead the ALL row against the training curve to confirm the units, "
          "then compare utr5_body with cds: the curve is an average over both, "
          "weighted by n_tokens.")


if __name__ == "__main__":
    main()
