#!/usr/bin/env python3
"""Aggregate raw probability blocks into a plotting-source table."""
from __future__ import annotations

# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
ICBINB_ROOT = _os.environ.get("ICBINB_ROOT", str(_Path(__file__).resolve().parents[3]))
# -------------------------------------------------------------------

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", required=True, type=int)
    parser.add_argument("--data-root", type=Path, default=Path(
        f"{ICBINB_ROOT}/data"))
    args = parser.parse_args()
    exp = args.data_root / "benchmark/fig2d_long_context"
    source_dir = exp / "probabilities" / args.model / f"ctx{args.context}"
    files = sorted(source_dir.glob("genes_*.npz"))
    if not files:
        raise FileNotFoundError(source_dir)
    p4, off, entropy = [], [], []
    for path in files:
        data = np.load(path)
        p4.append(data["p_acgt"])
        off.append(data["p_off"])
        entropy.append(data["entropy_bits"])
    p4_array = np.concatenate(p4)
    off_array = np.concatenate(off)
    entropy_array = np.concatenate(entropy)
    n = np.isfinite(entropy_array).sum(axis=0)
    table = pd.DataFrame({
        "model": args.model,
        "context_bp": args.context,
        "position": np.arange(p4_array.shape[1]),
        "n": n,
        "p_A": np.nanmean(p4_array[:, :, 0], axis=0),
        "p_C": np.nanmean(p4_array[:, :, 1], axis=0),
        "p_G": np.nanmean(p4_array[:, :, 2], axis=0),
        "p_T": np.nanmean(p4_array[:, :, 3], axis=0),
        "p_off": np.nanmean(off_array, axis=0),
        "entropy_bits": np.nanmean(entropy_array, axis=0),
    })
    table["p_AT"] = table.p_A + table.p_T
    table["p_GC"] = table.p_C + table.p_G
    output_dir = exp / "summaries/probabilities"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.model}_ctx{args.context}.csv"
    table.to_csv(output, index=False)
    print(f"wrote {output} from {len(files)} blocks and {len(p4_array)} generations")


if __name__ == "__main__":
    main()
