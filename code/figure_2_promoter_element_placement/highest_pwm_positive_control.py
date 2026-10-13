#!/usr/bin/env python3
"""Positive control for the highest-PWM-window base-matching metric.

`highest_pwm_base_matches.py` scores a generation by scanning it with the
element PWM, keeping the highest-scoring six-base window, and counting how many
of those bases match the gene's annotated element. The matching ceiling for that
metric is not 6/6: run the identical scan on the *real* continuation the model
was asked to write and the scan sometimes prefers a window other than the
annotated element. This script measures that ceiling.

For every gene we scan its real 45 bp promoter continuation (or the real
Shine--Dalgarno continuation), take the best-scoring window, and compare its six
bases with the annotated proximal box / SD core of the same gene.

Outputs one row per element with the mean match, a 2,000-replicate gene bootstrap
interval, the median, and how often the scan recovers the annotated element.

    python highest_pwm_positive_control.py \
        --prompts data/prompts_681_leaked.csv \
        --pwm-dir data/pwms \
        --output results/upstream_only/fig2x2/highest_pwm_positive_control.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASES = "ACGT"
BASE_INDEX = {base: index for index, base in enumerate(BASES)}
WIDTH = 6


def load_pwms(directory: Path) -> dict[str, np.ndarray]:
    return {
        path.stem.removeprefix("pwm_"): pd.read_csv(
            path, index_col=0
        )[list(BASES)].to_numpy()
        for path in sorted(directory.glob("pwm_*.csv"))
    }


def select_pwm(task: str, sigma: str, pwms: dict[str, np.ndarray]) -> np.ndarray:
    if task == "generate_sd":
        return pwms["sd_present"] if "sd_present" in pwms else pwms["sd_all"]
    return pwms.get(f"m10_{sigma}", pwms["m10_pooled"])


def best_window(sequence: str, pwm: np.ndarray) -> str:
    best_score = -np.inf
    best = ""
    for start in range(len(sequence) - len(pwm) + 1):
        window = sequence[start : start + len(pwm)]
        if any(base not in BASE_INDEX for base in window):
            continue
        score = sum(pwm[position, BASE_INDEX[base]]
                    for position, base in enumerate(window))
        if score > best_score:
            best_score = score
            best = window
    return best


def matches(left: str, right: str) -> int:
    return sum(a == b for a, b in zip(left, right))


def bootstrap(values: np.ndarray, seed: int, repeats: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def analyse(prompts: pd.DataFrame, pwms: dict[str, np.ndarray],
            repeats: int) -> pd.DataFrame:
    tasks = (("generate_promoter", "target_promoter", "m10_end"),
             ("generate_sd", "target_sd", "sd_end"))
    output = []
    for seed, (task, target_column, end_column) in enumerate(tasks):
        rows = prompts
        if task == "generate_promoter":
            rows = rows[rows.sigma == "sigma70"]
        values = []
        for row in rows.itertuples(index=False):
            end = int(getattr(row, end_column))
            native = row.upstream_seq[max(0, end - WIDTH) : end]
            hit = best_window(getattr(row, target_column) or "",
                              select_pwm(task, str(row.sigma), pwms))
            if len(native) == WIDTH and len(hit) == WIDTH:
                values.append(matches(hit, native))
        values = np.asarray(values, dtype=float)
        low, high = bootstrap(values, seed, repeats)
        output.append({
            "task": task,
            "kind": "real",
            "mean_matches": values.mean(),
            "ci_low": low,
            "ci_high": high,
            "median_matches": float(np.median(values)),
            "n_exact": int((values == WIDTH).sum()),
            "n_genes": len(values),
        })
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--pwm-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()

    results = analyse(
        pd.read_csv(args.prompts, keep_default_na=False),
        load_pwms(args.pwm_dir),
        args.bootstrap_repeats,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
    print(results.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
