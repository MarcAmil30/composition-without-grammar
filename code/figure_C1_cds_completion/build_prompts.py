#!/usr/bin/env python3
"""Build codon-aligned CDS midpoint prompts for the Figure 2 E. coli loci."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASES = set("ACGT")


def stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loci", type=Path, required=True)
    ap.add_argument("--flanks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260820)
    args = ap.parse_args()

    loci = pd.read_csv(args.loci, usecols=["locus_tag", "gene"])
    flanks = pd.read_csv(args.flanks, usecols=["accession", "locus_tag", "cds_seq"], dtype=str)
    joined = loci.merge(flanks, on="locus_tag", how="left", validate="one_to_one")
    if joined["cds_seq"].isna().any():
        missing = joined.loc[joined["cds_seq"].isna(), "locus_tag"].tolist()
        raise RuntimeError(f"missing CDS for {len(missing)} loci: {missing[:10]}")

    kept, excluded = [], []
    for row in joined.itertuples(index=False):
        cds = str(row.cds_seq).upper()
        reason = None
        if not cds or set(cds) - BASES:
            reason = "non_acgt"
        elif len(cds) % 3:
            reason = "length_not_multiple_of_three"
        cut = 3 * ((len(cds) // 2) // 3)
        if reason is None and len(cds) - cut < args.length:
            reason = "less_than_40bp_after_midpoint"
        if reason:
            excluded.append({"locus_tag": row.locus_tag, "gene": row.gene,
                             "cds_len": len(cds), "reason": reason})
            continue

        native = cds[:cut]
        shuffled = np.array(list(native))
        np.random.default_rng(stable_seed(row.locus_tag, args.seed)).shuffle(shuffled)
        shuffled = "".join(shuffled.tolist())
        assert len(native) % 3 == 0 and sorted(native) == sorted(shuffled)
        kept.append({
            "row_index": len(kept), "locus_tag": row.locus_tag, "gene": row.gene,
            "accession": row.accession, "cds_len": len(cds), "cut": cut,
            "prompt_native": native, "prompt_shuffled": shuffled,
            "true_next": cds[cut:cut + args.length],
        })

    out = pd.DataFrame(kept)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, compression="gzip")
    manifest = {
        "source_loci": str(args.loci), "source_cds": str(args.flanks),
        "n_input_loci": len(loci), "n_eligible": len(out),
        "n_excluded": len(excluded), "excluded": excluded,
        "continuation_bp": args.length, "shuffle_seed": args.seed,
        "cut_rule": "3 * floor((CDS_length / 2) / 3)",
        "shuffle": "deterministic mononucleotide permutation of each native prompt",
    }
    args.out.with_suffix("").with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
