#!/usr/bin/env python3
"""Validate, merge and score completed Figure 2d generation blocks."""
from __future__ import annotations

# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
ICBINB_ROOT = _os.environ.get("ICBINB_ROOT", str(_Path(__file__).resolve().parents[3]))
# -------------------------------------------------------------------

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASES = "ACGT"
BASE_INDEX = {base: index for index, base in enumerate(BASES)}


def scan(sequence: str, matrix: np.ndarray) -> tuple[int | None, float]:
    width = matrix.shape[0]
    best_position, best_score = None, -np.inf
    for position in range(len(sequence) - width + 1):
        window = sequence[position:position + width]
        if any(base not in BASE_INDEX for base in window):
            continue
        score = float(sum(matrix[i, BASE_INDEX[base]] for i, base in enumerate(window)))
        if score > best_score:
            best_position, best_score = position, score
    return best_position, best_score


def shuffled(sequence: str, seed: int) -> str:
    chars = list(sequence)
    np.random.default_rng(seed).shuffle(chars)
    return "".join(chars)


def bootstrap_gene(values: pd.Series, seed: int = 0, repeats: int = 4000) -> tuple[float, float]:
    array = values.to_numpy(float)
    rng = np.random.default_rng(seed)
    means = np.empty(repeats)
    for index in range(repeats):
        means[index] = rng.choice(array, size=len(array), replace=True).mean()
    return tuple(np.quantile(means, [0.025, 0.975]) * 100)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", required=True, type=int)
    parser.add_argument("--expected-genes", type=int)
    parser.add_argument(
        "--max-genes", type=int,
        help="score only rows with gene_index below this fixed subset boundary",
    )
    parser.add_argument("--data-root", type=Path, default=Path(
        f"{ICBINB_ROOT}/data"))
    args = parser.parse_args()

    exp = args.data_root / "benchmark/fig2d_long_context"
    block_dir = exp / "generations" / args.model / f"ctx{args.context}"
    files = sorted(block_dir.glob("genes_*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"no completed blocks in {block_dir}")
    frames = [pd.read_csv(path, keep_default_na=False) for path in files]
    generations = pd.concat(frames, ignore_index=True)
    if args.max_genes is not None:
        generations = generations.loc[generations.gene_index < args.max_genes].copy()
        if generations.empty:
            raise RuntimeError(f"no generations below --max-genes {args.max_genes}")
    key = ["locus_tag", "decode", "sample_idx"]
    duplicates = generations.duplicated(key).sum()
    if duplicates:
        raise RuntimeError(f"{duplicates} duplicate generation keys")
    n_genes = generations.locus_tag.nunique()
    if args.expected_genes and n_genes != args.expected_genes:
        raise RuntimeError(f"expected {args.expected_genes} genes, found {n_genes}")

    truth = pd.read_csv(
        args.data_root / "benchmark/upstream_only/prompts_681_leaked.csv"
    ).set_index("locus_tag")
    pwm_dir = args.data_root / "benchmark/upstream_only/pwms"
    matrices = {
        sigma: pd.read_csv(pwm_dir / f"pwm_m10_{sigma}.csv", index_col=0)[list(BASES)].to_numpy()
        for sigma in ("sigma24", "sigma32", "sigma38", "sigma54", "sigma70")
    }
    matrices["pooled"] = pd.read_csv(
        pwm_dir / "pwm_m10_pooled.csv", index_col=0
    )[list(BASES)].to_numpy()

    scored = []
    for index, row in enumerate(generations.itertuples(index=False)):
        annotation = truth.loc[row.locus_tag]
        matrix = matrices.get(str(annotation.sigma), matrices["pooled"])
        true_position = int(annotation.m10_offset_in_target)
        position, score = scan(row.generation, matrix)
        random_position, random_score = scan(shuffled(row.generation, index), matrix)
        error = np.nan if position is None else position - true_position
        random_error = np.nan if random_position is None else random_position - true_position
        true_hexamer = str(annotation.upstream_seq)[
            int(annotation.m10_start):int(annotation.m10_end)
        ]
        generated_hexamer = row.generation[true_position:true_position + len(true_hexamer)]
        hamming = (sum(a != b for a, b in zip(generated_hexamer, true_hexamer))
                   if len(generated_hexamer) == len(true_hexamer) else np.nan)
        scored.append({
            **row._asdict(),
            "true_m10_position": true_position,
            "predicted_m10_position": position,
            "position_error": error,
            "abs_position_error": abs(error) if np.isfinite(error) else np.nan,
            "pwm_score": score,
            "hamming_at_true": hamming,
            "within_0": bool(np.isfinite(error) and abs(error) <= 0),
            "within_1": bool(np.isfinite(error) and abs(error) <= 1),
            "within_2": bool(np.isfinite(error) and abs(error) <= 2),
            "within_3": bool(np.isfinite(error) and abs(error) <= 3),
            "within_5": bool(np.isfinite(error) and abs(error) <= 5),
            "shuffle_position_error": random_error,
            "shuffle_pwm_score": random_score,
            "shuffle_within_2": bool(np.isfinite(random_error) and abs(random_error) <= 2),
        })
    scores = pd.DataFrame(scored)

    score_dir = exp / "scores" / args.model
    summary_dir = exp / "summaries"
    score_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    score_path = score_dir / f"ctx{args.context}_per_generation.csv.gz"
    scores.to_csv(score_path, index=False, compression="gzip")

    per_gene = scores.groupby("locus_tag", as_index=True).agg(
        within_0=("within_0", "mean"), within_1=("within_1", "mean"),
        within_2=("within_2", "mean"), within_3=("within_3", "mean"),
        within_5=("within_5", "mean"), shuffle_within_2=("shuffle_within_2", "mean"),
        mean_abs_error=("abs_position_error", "mean"), mean_pwm_score=("pwm_score", "mean"),
        mean_hamming=("hamming_at_true", "mean"),
    )
    per_gene_path = score_dir / f"ctx{args.context}_per_gene.csv"
    per_gene.to_csv(per_gene_path)
    low, high = bootstrap_gene(per_gene.within_2)
    shuffle_low, shuffle_high = bootstrap_gene(per_gene.shuffle_within_2, seed=1)
    summary = {
        "model": args.model,
        "context_bp": args.context,
        "n_genes": n_genes,
        "n_generations": len(scores),
        "within_2_pct": float(scores.within_2.mean() * 100),
        "within_2_gene_bootstrap_95": [float(low), float(high)],
        "shuffle_within_2_pct": float(scores.shuffle_within_2.mean() * 100),
        "shuffle_gene_bootstrap_95": [float(shuffle_low), float(shuffle_high)],
        "exact_pct": float(scores.within_0.mean() * 100),
        "mean_abs_position_error": float(scores.abs_position_error.mean()),
        "mean_hamming_at_true": float(scores.hamming_at_true.mean()),
        "score_file": str(score_path),
        "score_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
    }
    summary_path = summary_dir / f"{args.model}_ctx{args.context}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
