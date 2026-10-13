#!/usr/bin/env python3
"""Compare the best PWM window in each generation with the native element."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

BASES = "ACGT"
BASE_INDEX = {base: index for index, base in enumerate(BASES)}
MODELS = ["evo1", "evo1_131k", "evo2", "hyenadna", "promogen2"]
TASKS = ["generate_promoter", "generate_sd"]
BACKGROUNDS = ["leaked", "clean"]
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


def shuffle(sequence: str, key: str) -> str:
    """Composition-preserving shuffle with a stable seed for each generation."""
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    return "".join(rng.permutation(list(sequence)).tolist())


def matches(left: str, right: str) -> int:
    return sum(a == b for a, b in zip(left, right))


def bootstrap(values: np.ndarray, seed: int, repeats: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def analyse(bench: Path, repeats: int) -> pd.DataFrame:
    pwms = load_pwms(bench / "pwms")
    truth = pd.read_csv(
        bench / "prompts_681_leaked.csv", keep_default_na=False
    ).set_index("locus_tag")
    records = []

    for task in TASKS:
        for background in BACKGROUNDS:
            for model in MODELS:
                path = bench / "generations" / model / f"{task}__{background}.csv"
                generations = pd.read_csv(path, keep_default_na=False)
                for row_number, row in enumerate(generations.itertuples(index=False)):
                    annotation = truth.loc[row.locus_tag]
                    if task == "generate_promoter" and annotation["sigma"] != "sigma70":
                        continue

                    sequence = row.generation or ""
                    pwm = select_pwm(task, str(annotation["sigma"]), pwms)
                    end = int(annotation[
                        "m10_end" if task == "generate_promoter" else "sd_end"
                    ])
                    native = annotation["upstream_seq"][max(0, end - WIDTH) : end]
                    key = "|".join(map(str, (
                        model, task, background, row.locus_tag,
                        row.decode, row.sample_idx, row_number,
                    )))
                    generated_hit = best_window(sequence, pwm)
                    shuffled_hit = best_window(shuffle(sequence, key), pwm)
                    if all(len(item) == WIDTH
                           for item in (native, generated_hit, shuffled_hit)):
                        records.append({
                            "task": task,
                            "background": background,
                            "model": model,
                            "locus_tag": row.locus_tag,
                            "generated": matches(generated_hit, native),
                            "shuffled": matches(shuffled_hit, native),
                        })

    return pd.DataFrame(records)


def summarise(per_generation: pd.DataFrame, repeats: int) -> pd.DataFrame:
    output = []
    seed = 0
    for task in TASKS:
        for background in BACKGROUNDS:
            for model in MODELS:
                subset = per_generation[
                    (per_generation.task == task)
                    & (per_generation.background == background)
                    & (per_generation.model == model)
                ]
                for kind in ("generated", "shuffled"):
                    values = subset.groupby("locus_tag")[kind].mean().to_numpy()
                    low, high = bootstrap(values, seed, repeats)
                    seed += 1
                    output.append({
                        "task": task,
                        "background": background,
                        "model": model,
                        "kind": kind,
                        "mean_matches": values.mean(),
                        "ci_low": low,
                        "ci_high": high,
                        "n_genes": len(values),
                        "n_generations": len(subset),
                    })
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()

    # The full per-generation table is kept in memory only; the saved output is
    # the gene-aggregated table used for reporting.
    results = summarise(analyse(args.bench_dir, args.bootstrap_repeats),
                        args.bootstrap_repeats)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
    else:
        print(results.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
