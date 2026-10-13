#!/usr/bin/env python3
"""Generate 40-bp CDS completions and recover A/C/G/T probabilities."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASES = "ACGT"
EPS = 1e-12


def load_models(data_root: Path):
    sys.path.insert(0, str(data_root / "code/generation"))
    import models  # type: ignore
    return models


def base_ids(adapter) -> list[int]:
    if adapter.tok is None:
        return [ord(base) for base in BASES]
    out = []
    for base in BASES:
        ids = adapter.tok(base, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise RuntimeError(f"{base} is not one token: {ids}")
        out.append(ids[0])
    return out


def encode(adapter, sequences: list[str]):
    import torch
    if adapter.tok is None:
        tok = adapter.model.tokenizer
        return torch.tensor([tok.tokenize(s) for s in sequences], dtype=torch.long,
                            device=adapter.device)
    return adapter.tok(sequences, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to(adapter.device)


def forward_logits(adapter, ids):
    if adapter.tok is None:
        output = adapter.model.forward(ids)
        while isinstance(output, (tuple, list)):
            output = output[0]
        return output.float()
    return adapter.model(ids).logits.float()


def probabilities(adapter, prompt: str, continuations: list[str], ids_acgt: list[int], batch: int):
    import torch
    result = np.full((len(continuations), 40, 4), np.nan, dtype=np.float32)
    p_off = np.full((len(continuations), 40), np.nan, dtype=np.float32)
    for begin in range(0, len(continuations), batch):
        seqs = continuations[begin:begin + batch]
        full = [prompt + s for s in seqs]
        encoded = encode(adapter, full)
        with torch.inference_mode():
            logits = forward_logits(adapter, encoded)
        for local, seq in enumerate(seqs):
            selected = logits[local, len(prompt) - 1:len(prompt) - 1 + len(seq)]
            full_p = torch.softmax(selected, dim=-1)
            p4 = full_p[:, ids_acgt].cpu().numpy()
            result[begin + local, :len(seq)] = p4
            p_off[begin + local, :len(seq)] = 1.0 - p4.sum(axis=1)
    return result, p_off


def atomic_npz(path: Path, **arrays) -> None:
    temp = Path(str(path) + ".partial.npz")
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["evo1", "evo2", "hyenadna"], required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--stop", type=int, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--prob-batch", type=int, default=4)
    args = ap.parse_args()

    models = load_models(args.data_root)
    exp = args.data_root / "benchmark/cds_completion_probabilities"
    prompts = pd.read_csv(exp / "prompts/cds_midpoint_prompts.csv.gz", keep_default_na=False)
    prompts = prompts.iloc[args.start:min(args.stop, len(prompts))]
    stem = f"genes_{args.start:04d}_{min(args.stop, len(prompts) + args.start):04d}"
    out_dir = exp / "shards" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / f"{stem}.npz"
    out_meta = out_dir / f"{stem}.csv.gz"
    out_manifest = out_dir / f"{stem}.manifest.json"
    if out_npz.exists() and out_meta.exists() and out_manifest.exists():
        print(f"[skip] complete shard {out_npz}")
        return

    cfg = models.SamplingConfig(n_samples=20, temperature=1.0, top_k=0, top_p=1.0,
                                greedy=True, seed=0, batch_size=16)
    adapter = models.build(args.model)
    # Long CDS prompts make Evo 1's StripedHyena FFT/cache path much larger
    # than the short Figure 2 prompts. One sequence per forward stays below a
    # GH200's memory limit; this changes throughput, not the sampling law.
    if args.model == "evo1":
        adapter.spec.seqs_per_forward = 1
        args.prob_batch = 1
    adapter.load()
    ids_acgt = base_ids(adapter)

    all_p, all_off, metadata = [], [], []
    teacher_p, teacher_off, teacher_meta = [], [], []
    invalid_generated = []
    for done, row in enumerate(prompts.itertuples(index=False), 1):
        for condition in ("native", "shuffled"):
            prompt = getattr(row, f"prompt_{condition}")
            generated = adapter.continue_([prompt], 40, cfg)[0]
            raw = [generated["greedy"]] + generated["samples"]
            raw_decode = ["greedy"] + ["sample"] * len(generated["samples"])
            raw_idx = [-1] + list(range(len(generated["samples"])))
            valid = [i for i, s in enumerate(raw) if len(s) == 40 and not (set(s) - set(BASES))]
            for i, s in enumerate(raw):
                if i not in valid:
                    invalid_generated.append({
                        "locus_tag": row.locus_tag, "condition": condition,
                        "decode": raw_decode[i], "sample_idx": raw_idx[i],
                        "length": len(s), "non_acgt": sorted(set(s) - set(BASES)),
                    })
            if not valid:
                raise RuntimeError(f"no valid continuations for {row.locus_tag}/{condition}")
            continuations = [raw[i] for i in valid]
            decode = [raw_decode[i] for i in valid]
            sample_idx = [raw_idx[i] for i in valid]
            p4, poff = probabilities(adapter, prompt, continuations, ids_acgt, args.prob_batch)
            all_p.append(p4); all_off.append(poff)
            for d, si, seq in zip(decode, sample_idx, continuations):
                metadata.append({"locus_tag": row.locus_tag, "gene": row.gene,
                                 "condition": condition, "decode": d, "sample_idx": si,
                                 "prompt_len": len(prompt), "generation": seq})

            tp4, tpoff = probabilities(adapter, prompt, [row.true_next], ids_acgt, 1)
            teacher_p.append(tp4[0]); teacher_off.append(tpoff[0])
            teacher_meta.append({"locus_tag": row.locus_tag, "gene": row.gene,
                                 "condition": condition, "prompt_len": len(prompt),
                                 "true_next": row.true_next})
        print(f"[{args.model}] {done}/{len(prompts)}", flush=True)

    p_acgt = np.concatenate(all_p, axis=0)
    p_off = np.concatenate(all_off, axis=0)
    q = p_acgt / np.clip(p_acgt.sum(axis=-1, keepdims=True), EPS, None)
    entropy = -(q * np.log2(np.clip(q, EPS, None))).sum(axis=-1)
    atomic_npz(out_npz, p_acgt=p_acgt, p_off=p_off, entropy_bits=entropy,
               teacher_p_acgt=np.stack(teacher_p), teacher_p_off=np.stack(teacher_off),
               bases=np.array(list(BASES)))
    pd.DataFrame(metadata).to_csv(out_meta, index=False, compression="gzip")
    pd.DataFrame(teacher_meta).to_csv(out_dir / f"{stem}.teacher.csv.gz", index=False,
                                      compression="gzip")
    manifest = {"model": args.model, "checkpoint": models.REGISTRY[args.model].checkpoint,
                "start": args.start, "stop": args.stop, "n_genes": len(prompts),
                "sampling": cfg.as_dict(), "continuation_bp": 40,
                "n_invalid_generations_excluded": len(invalid_generated),
                "invalid_generations": invalid_generated,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID")}
    out_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
