#!/usr/bin/env python3
"""
Score the generations for the upstream-only benchmark.

Reads generations/<model>/<task>__<background>.csv, scores every row against the
gene's own annotation, and writes one long per-generation table plus an
aggregated summary. No model is loaded; this is pure CPU and can be re-run freely
as the metrics evolve.

Task 1 -- generate_promoter (45 bp continuation from the end of the -35)
    pos_error     | argmax-PWM position minus the real -10 offset
    seq_mismatch  | Hamming distance of that hexamer to the gene's real -10 (0-6)
    spacer        | the argmax position itself = the -35 -> -10 distance
    pwm_score     | log2-odds of the hit, with the real box's score as a ceiling

Task 2 -- generate_sd (40 bp continuation from L-20)
    pos_error     | argmax SD-PWM position minus the real SD offset
    seq_mismatch  | Hamming to the gene's real SD hexamer (0-6)
    codon_found   | did it emit this gene's own start codon, and at what offset
                  | (truth = 20). EOS position is used for models that stop early.
    own_spacing   | its own SD -> its own start codon distance (real mode 6-7)
    pwm_score     | as above

A "random" reference row is produced alongside every real one by scoring a
composition-preserving shuffle of the same generation. It is not a competing
model -- it is what the argmax does when the grammar is destroyed but the base
composition is not, and it is what makes pwm_score comparable across models that
generate at different GC.

    python score.py [--gen-dir ...] [--out ...] [--jobs N]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

BASES = "ACGT"
IDX = {b: i for i, b in enumerate(BASES)}
CORE = 6

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCH = os.fspath(os.environ.get("BENCH_DIR", REPO_ROOT / "data" / "upstream_only"))


# ---------------------------------------------------------------------
# PWM scanning
# ---------------------------------------------------------------------
def load_pwms(pwm_dir):
    mats = {}
    for p in glob.glob(os.path.join(pwm_dir, "pwm_*.csv")):
        name = os.path.basename(p)[4:-4]
        mats[name] = pd.read_csv(p, index_col=0)[list(BASES)].to_numpy()
    return mats


def scan(seq, mat):
    """Best-scoring window. Returns (position, score, hexamer) or (None, nan, '')."""
    w = mat.shape[0]
    if len(seq) < w:
        return None, float("nan"), ""
    best_p, best_s = None, -1e18
    for p in range(len(seq) - w + 1):
        sub = seq[p:p + w]
        if any(c not in IDX for c in sub):
            continue
        s = sum(mat[j, IDX[c]] for j, c in enumerate(sub))
        if s > best_s:
            best_p, best_s = p, s
    if best_p is None:
        return None, float("nan"), ""
    return best_p, best_s, seq[best_p:best_p + w]


def score_fixed(seq, mat):
    """Score a specific hexamer (used for the real-sequence ceiling)."""
    if len(seq) != mat.shape[0] or any(c not in IDX for c in seq):
        return float("nan")
    return sum(mat[j, IDX[c]] for j, c in enumerate(seq))


def hamming(a, b):
    if len(a) != len(b) or not a:
        return None
    return sum(x != y for x, y in zip(a, b))


def shuffled(seq, rng):
    ch = list(seq)
    rng.shuffle(ch)
    return "".join(ch)


# ---------------------------------------------------------------------
def pwm_for(task, sigma, mats):
    if task == "generate_sd":
        return mats.get("sd_present", mats.get("sd_all"))
    return mats.get(f"m10_{sigma}", mats["m10_pooled"])


def score_file(path, truth, mats, rng):
    gen = pd.read_csv(path, keep_default_na=False)
    if gen.empty:
        return pd.DataFrame()
    task = gen.task.iloc[0]
    rows = []

    for r in gen.itertuples():
        t = truth.loc[r.locus_tag]
        mat = pwm_for(task, t["sigma"], mats)
        g = r.generation or ""

        if task == "generate_promoter":
            true_off = int(t["m10_offset_in_target"])
            true_hex = t["upstream_seq"][max(0, int(t["m10_end"]) - CORE):int(t["m10_end"])]
        else:
            true_off = int(t["sd_start"]) - (int(t["L"]) - 20)
            true_hex = t["upstream_seq"][max(0, int(t["sd_end"]) - CORE):int(t["sd_end"])]

        p, s, hexa = scan(g, mat)
        ps, ss, _ = scan(shuffled(g, rng), mat)

        row = dict(
            locus_tag=r.locus_tag, model=r.model, task=task,
            background=r.background, decode=r.decode, sample_idx=r.sample_idx,
            sigma=t["sigma"], gen_len=len(g),
            gc=(g.count("G") + g.count("C")) / len(g) if g else float("nan"),
            hit_pos=p, hit_score=s, hit_hex=hexa,
            true_pos=true_off, true_hex=true_hex,
            true_score=score_fixed(true_hex, mat),
            shuf_score=ss, shuf_pos=ps,
            pos_error=(abs(p - true_off) if p is not None else None),
            seq_mismatch=hamming(hexa, true_hex),
        )

        if task == "generate_promoter":
            row["spacer"] = p
        else:
            codon = t["start_codon"]
            row["start_codon"] = codon
            # the gene's own start codon, searched from the expected offset outward
            found = None
            if codon and len(codon) == 3:
                cands = [i for i in range(len(g) - 2) if g[i:i + 3] == codon]
                if cands:
                    found = min(cands, key=lambda i: (abs(i - 20), i))
            row["codon_pos"] = found
            row["codon_found"] = int(found is not None)
            row["codon_pos_error"] = abs(found - 20) if found is not None else None
            row["stopped_early"] = int(len(g) < 40)
            row["own_spacing"] = (found - (p + CORE)) if (found is not None and p is not None) else None

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
def summarise(df):
    """One row per (model, task, background)."""
    out = []
    for (m, t, b), g in df.groupby(["model", "task", "background"], dropna=False):
        smp = g[g.decode == "sample"]
        d = dict(model=m, task=t, background=b,
                 n_generations=len(g), n_genes=g.locus_tag.nunique(),
                 mean_gen_len=g.gen_len.mean(), mean_gc=g.gc.mean(),
                 pos_error_median=smp.pos_error.median(),
                 pos_error_mean=smp.pos_error.mean(),
                 frac_pos_exact=(smp.pos_error == 0).mean(),
                 frac_pos_within2=(smp.pos_error <= 2).mean(),
                 seq_mismatch_median=smp.seq_mismatch.median(),
                 frac_seq_within1=(smp.seq_mismatch <= 1).mean(),
                 hit_score_mean=smp.hit_score.mean(),
                 shuf_score_mean=smp.shuf_score.mean(),
                 true_score_mean=smp.true_score.mean())
        d["score_gap_vs_shuffle"] = d["hit_score_mean"] - d["shuf_score_mean"]
        d["score_frac_of_ceiling"] = d["hit_score_mean"] / d["true_score_mean"] \
            if d["true_score_mean"] else float("nan")
        if t == "generate_sd":
            d.update(codon_found_rate=smp.codon_found.mean(),
                     codon_pos_error_median=smp.codon_pos_error.median(),
                     stopped_early_rate=smp.stopped_early.mean(),
                     own_spacing_median=smp.own_spacing.median())
        else:
            d["spacer_median"] = smp.spacer.median()
        out.append(d)
    return pd.DataFrame(out).sort_values(["task", "model", "background"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", default=os.path.join(BENCH, "generations"))
    ap.add_argument("--pwm-dir", default=os.path.join(BENCH, "pwms"))
    ap.add_argument("--out", default=os.path.join(BENCH, "scores"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)

    truth = pd.read_csv(os.path.join(BENCH, "prompts_681_leaked.csv"))
    truth["sigma"] = truth["sigma"].fillna("unknown").astype(str)
    truth["start_codon"] = truth["start_codon"].fillna("")
    truth = truth.set_index("locus_tag")

    mats = load_pwms(a.pwm_dir)
    print(f"loaded {len(mats)} PWMs: {sorted(mats)}")

    files = sorted(glob.glob(os.path.join(a.gen_dir, "*", "*__*.csv")))
    files = [f for f in files if "_smoke" not in f and "manifest" not in f]
    if not files:
        print(f"no generation files under {a.gen_dir}")
        return
    print(f"scoring {len(files)} shards")

    parts = []
    for f in files:
        d = score_file(f, truth, mats, rng)
        if d.empty:
            continue
        parts.append(d)
        print(f"  {os.path.relpath(f, a.gen_dir):55s} {len(d):6d} rows")

    allrows = pd.concat(parts, ignore_index=True)
    allrows.to_csv(os.path.join(a.out, "per_generation.csv.gz"),
                   index=False, compression="gzip")

    summ = summarise(allrows)
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)

    # per-gene aggregation, for the gene-specificity regression
    keys = ["model", "task", "background", "locus_tag"]
    pergene = (allrows[allrows.decode == "sample"]
               .groupby(keys, dropna=False)
               .agg(pos_error=("pos_error", "median"),
                    seq_mismatch=("seq_mismatch", "median"),
                    hit_score=("hit_score", "mean"),
                    true_score=("true_score", "first"),
                    shuf_score=("shuf_score", "mean"),
                    sigma=("sigma", "first"))
               .reset_index())
    pergene.to_csv(os.path.join(a.out, "per_gene.csv.gz"),
                   index=False, compression="gzip")

    json.dump({"n_shards": len(files), "n_rows": len(allrows),
               "pwms": sorted(mats), "seed": a.seed},
              open(os.path.join(a.out, "scoring_manifest.json"), "w"), indent=2)

    pd.set_option("display.width", 200, "display.max_columns", 40)
    print("\n" + summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote -> {a.out}")


if __name__ == "__main__":
    main()
