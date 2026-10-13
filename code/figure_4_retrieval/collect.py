#!/usr/bin/env python3
"""Gather every finished retrieval result into one table (the paper table).

  python3 collect.py            print + write results/RESULTS.md and results.csv
"""
import glob
import json
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RES = Path(os.environ.get("ICBINB_RETRIEVAL_RESULTS", REPO_ROOT / "data" / "retrieval" / "results"))

rows = []
for f in sorted(glob.glob(str(RES / "*" / "*_summary.json"))):
    if "_smoke" in os.path.basename(f):
        continue                      # pilot runs (n=120) are never reported
    d = json.load(open(f))
    g = d.get("guard_true_vs_decoy", {})
    v = d.get("variance_decomposition", {})
    rows.append({
        "model": d.get("model", os.path.basename(os.path.dirname(f))),
        "scorer": d.get("scorer", ""),
        "variant": d["name"],
        "norm": d.get("normalization", "-"),
        "n": d["n_genes"],
        "top1_%": round(d["top1"] * 100, 2),
        "ci_lo": round(d["top1_ci95"][0] * 100, 2),
        "ci_hi": round(d["top1_ci95"][1] * 100, 2),
        "top5_%": round(d["top5"] * 100, 2),
        "median_rank": d["median_rank"],
        "mrr": round(d["mrr"], 4),
        "fold_chance": round(d["fold_over_chance"], 1),
        "true>decoy_%": round(g.get("frac_true_higher", float("nan")) * 100, 1),
        "guard_p": f"{g.get('wilcoxon_p_greater', float('nan')):.1e}",
        "interaction_%": round(v.get("interaction_frac", float("nan")) * 100, 1),
    })

if not rows:
    print("no results yet")
    raise SystemExit

df = pd.DataFrame(rows).sort_values(["model", "scorer", "variant"])
df.to_csv(RES / "results.csv", index=False)

head = df[df.variant.str.contains("pmi$|embed_cos_centered", regex=True)]
with open(RES / "RESULTS.md", "w") as fh:
    fh.write("# Task 4 results (chance: top-1 1%, median rank 50)\n\n")
    fh.write("## Headline (PMI for scoring, centered cosine for embedding)\n\n")
    fh.write(head.to_markdown(index=False) + "\n\n## All variants\n\n")
    fh.write(df.to_markdown(index=False) + "\n")

pd.set_option("display.width", 200, "display.max_columns", 50)
print("HEADLINE\n", head.to_string(index=False))
print(f"\nwrote {RES}/RESULTS.md and results.csv ({len(df)} rows)")
