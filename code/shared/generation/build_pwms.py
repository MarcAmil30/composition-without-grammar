#!/usr/bin/env python3
"""
Build the scoring matrices for the upstream-only benchmark.

Everything is derived from the annotated real sequences in prompts_681_leaked.csv
-- nothing is assumed from a literature consensus.

  -10 box   one PWM per sigma factor. RegulonDB box widths are not uniform, so
            the window is anchored at the box's 3' end and a 6 bp core is taken
            (matching the convention used for the paper's logo figure).
  -35 box   same, anchored at the 5' end. Built for completeness; the benchmark
            scores the -10.
  SD        one PWM, anchored on the SD 3' end. Anchoring on the SD rather than
            on the ATG matters: the motif slides (spacing 3-12 bp), so an
            ATG-aligned matrix smears to 0.43 bits where an SD-anchored one
            reaches 1.04.

Output: <out>/pwm_<element>[_<sigma>].csv  -- log2 odds vs upstream background,
one row per position, columns A/C/G/T. Plus pwm_manifest.json.

    python build_pwms.py [bench_dir] [out_dir]
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASES = list("ACGT")
CORE = 6            # scoring-core width, in bp
PSEUDO = 0.25       # Laplace pseudocount per base, per position
MIN_N = 25          # a sigma factor below this gets folded into the pooled PWM


def counts_to_logodds(seqs, bg):
    """Column-wise log2( p(base) / background ) with a pseudocount."""
    w = len(seqs[0])
    mat = np.zeros((w, 4))
    for j in range(w):
        c = Counter(s[j] for s in seqs)
        tot = len(seqs) + 4 * PSEUDO
        for i, b in enumerate(BASES):
            mat[j, i] = np.log2(((c.get(b, 0) + PSEUDO) / tot) / bg[b])
    return pd.DataFrame(mat, columns=BASES, index=range(w))


def info_content(seqs):
    """Total information of the alignment, in bits vs uniform."""
    w = len(seqs[0])
    tot = 0.0
    for j in range(w):
        c = Counter(s[j] for s in seqs)
        p = np.array([c.get(b, 0) for b in BASES], float)
        p /= p.sum()
        h = -np.sum(p[p > 0] * np.log2(p[p > 0]))
        tot += 2.0 - h
    return tot


def main(bench, out):
    os.makedirs(out, exist_ok=True)
    df = pd.read_csv(os.path.join(bench, "prompts_681_leaked.csv"))
    df["sigma"] = df["sigma"].fillna("unknown").astype(str)

    allseq = "".join(df.upstream_seq)
    bg = {b: allseq.count(b) / len(allseq) for b in BASES}
    print("upstream background:", {k: round(v, 4) for k, v in bg.items()})

    manifest = {"background": bg, "core_width": CORE, "pseudocount": PSEUDO,
                "n_genes": len(df), "matrices": {}}

    def emit(name, seqs, note):
        if len(seqs) < 5:
            print(f"  {name}: only {len(seqs)} sequences, skipped")
            return
        m = counts_to_logodds(seqs, bg)
        path = os.path.join(out, f"pwm_{name}.csv")
        m.to_csv(path, index_label="pos")
        cons = "".join(max(BASES, key=lambda b: m.loc[j, b]) for j in range(len(m)))
        ic = info_content(seqs)
        manifest["matrices"][name] = {"n": len(seqs), "width": len(m),
                                      "consensus": cons, "bits": round(ic, 3),
                                      "note": note, "file": os.path.basename(path)}
        print(f"  {name:22s} n={len(seqs):4d}  {cons}  {ic:5.2f} bits")

    # ---- -10 box: anchor 3' end, take the last CORE bp ----
    print("\n-10 box (3'-anchored):")
    by_sigma = defaultdict(list)
    pooled10 = []
    for r in df.itertuples():
        s = r.upstream_seq[max(0, r.m10_end - CORE):r.m10_end]
        if len(s) != CORE or set(s) - set(BASES):
            continue
        by_sigma[r.sigma].append(s)
        pooled10.append(s)
    emit("m10_pooled", pooled10, "all sigma factors pooled; fallback matrix")
    for sg, seqs in sorted(by_sigma.items(), key=lambda kv: -len(kv[1])):
        if len(seqs) >= MIN_N:
            emit(f"m10_{sg}", seqs, f"{sg} only")
        else:
            print(f"  m10_{str(sg):12s} n={len(seqs):4d}  below MIN_N={MIN_N}, use pooled")

    # ---- -35 box: anchor 5' end ----
    print("\n-35 box (5'-anchored):")
    by_sigma35 = defaultdict(list)
    pooled35 = []
    for r in df.itertuples():
        s = r.upstream_seq[r.m35_start:r.m35_start + CORE]
        if len(s) != CORE or set(s) - set(BASES):
            continue
        by_sigma35[r.sigma].append(s)
        pooled35.append(s)
    emit("m35_pooled", pooled35, "all sigma factors pooled")
    for sg, seqs in sorted(by_sigma35.items(), key=lambda kv: -len(kv[1])):
        if len(seqs) >= MIN_N:
            emit(f"m35_{sg}", seqs, f"{sg} only")

    # ---- SD: anchor 3' end of the annotated span ----
    print("\nShine-Dalgarno (3'-anchored):")
    sd_all, sd_present = [], []
    for r in df.itertuples():
        s = r.upstream_seq[max(0, r.sd_end - CORE):r.sd_end]
        if len(s) != CORE or set(s) - set(BASES):
            continue
        sd_all.append(s)
        if r.sd_present == 1:
            sd_present.append(s)
    emit("sd_all", sd_all, "all 681 genes at their best-hit position")
    emit("sd_present", sd_present,
         "sd_present=1 only -- the sharper matrix, use this for scoring")

    json.dump(manifest, open(os.path.join(out, "pwm_manifest.json"), "w"), indent=2)
    print(f"\nwrote {len(manifest['matrices'])} matrices -> {out}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    bench = sys.argv[1] if len(sys.argv) > 1 else os.fspath(repo_root / "data" / "upstream_only")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(bench, "pwms")
    main(bench, out)
