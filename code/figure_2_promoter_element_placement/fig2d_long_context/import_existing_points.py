#!/usr/bin/env python3
"""Import the already completed 200-1000 bp Figure 2d points."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.source)
    data = data[(~data.is_inter.astype(bool)) & data.model.isin(["evo1", "evo2", "hyenadna"])]
    args.out.mkdir(parents=True, exist_ok=True)
    for row in data.itertuples(index=False):
        payload = {
            "model": row.model,
            "context_bp": int(row.x),
            "n_genes": 680,
            "n_generations": 14280,
            "within_2_pct": float(row.exact),
            "within_2_gene_bootstrap_95": [float(row.lo), float(row.hi)],
            "imported_from_completed_figure2d": str(args.source),
        }
        path = args.out / f"{row.model}_ctx{int(row.x)}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(path)


if __name__ == "__main__":
    main()
