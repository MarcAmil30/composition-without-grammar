#!/usr/bin/env python3
"""Collapse raw CDS-completion shards into the profile cache used for plotting."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aggregate_and_plot import BASES, MODELS, gene_profiles, load_model, teacher_profiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="directory containing prompts/, shards/, and results/")
    parser.add_argument("--expected-shards", type=int, default=27)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    prompts = pd.read_csv(args.root / "prompts" / "cds_midpoint_prompts.csv.gz")
    true = np.stack([
        np.array([[float(base == candidate) for candidate in BASES] for base in seq])
        for seq in prompts.true_next
    ])
    arrays: dict[str, np.ndarray] = {"real": true.mean(axis=0)}

    for model, _ in MODELS:
        meta, probabilities, teacher_meta, teacher_probabilities = load_model(
            args.root, model, args.expected_shards
        )
        for condition, profile in gene_profiles(meta, probabilities).items():
            arrays[f"{model}_{condition}"] = profile
        for condition, profile in teacher_profiles(
            teacher_meta, teacher_probabilities
        ).items():
            arrays[f"{model}_teacher_{condition}"] = profile

    out = args.out or args.root / "results" / "cds_completion_profiles.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    print(f"wrote {out} ({len(arrays)} arrays; {len(prompts)} genes)")


if __name__ == "__main__":
    main()
