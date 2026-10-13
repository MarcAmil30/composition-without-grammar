"""
Singleton-pair robustness version of step 4: attach the third-position clock reading.

Protein identity is a bad clock -- a protein held still by selection stays similar while its
DNA keeps mutating at silent sites. So we measure the clock at THIRD CODON POSITIONS (a dS
proxy) and ask whether upstream similarity still tracks protein identity once the clock is
held fixed.

  P3  third-codon-position identity over the blastp-aligned region  -> time since divergence
  PID protein identity                                              -> functional constraint

The test: within a narrow band of P3 (same amount of elapsed divergence), do pairs with
conserved proteins have more similar upstreams than pairs with drifted proteins?

  yes -> function contributes something beyond inheritance
  no  -> everything in Arm A was the molecular clock

Needs `query_vs_members_aln.tsv` (blastp with qseq/sseq) and the CDS nucleotide sequences of
both sides.

    python grammar_step4.py

Outputs: results/upstream_grammar/pairs_cds_singleton_pairs.parquet,
         cds_conditioning_singleton_pairs.json
"""
# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
CL_ROOT = _os.environ.get("CL_ROOT", "../Contrastive_Learning_NeurIPS2026")
# -------------------------------------------------------------------

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

PROJ = f"{CL_ROOT}"
OUT = f"{PROJ}/results/upstream_grammar"
GFLANKS = f"{PROJ}/data/gamma_genus/gamma_genus_flanks.csv"
RFLANKS = f"{PROJ}/data/ref2/ref2_flanks.csv"
ECOLI_ACC = "GCF_000005845.2"

# ------------------------------------------------------------------ scored pairs
sc = pd.read_parquet(f"{OUT}/ortho_scored_singleton_pairs.parquet")
frames = pd.read_parquet(f"{OUT}/ortho_frames_singleton_pairs.parquet",
                         columns=["as_row", "qseqid", "member_row", "member_locus"])
sc = sc.merge(frames.drop_duplicates(["as_row", "member_row"]), on=["as_row", "member_row"],
              how="left")
print(f"scored pairs: {len(sc):,}", flush=True)

# ------------------------------------------------------------------ E. coli query CDS
q = pd.read_parquet(f"{PROJ}/results/step0/analysis_set.parquet",
                    columns=["locus_tag", "species"]).reset_index()
q = q[q["species"] == "Escherichia coli"].rename(columns={"index": "as_row"})
as2locus = dict(zip(q["as_row"], q["locus_tag"]))

ref = pd.read_csv(RFLANKS, usecols=["accession", "locus_tag", "cds_seq"],
                  dtype={"locus_tag": "string", "cds_seq": "string"})
ref = ref[ref["accession"] == ECOLI_ACC]
locus2cds = dict(zip(ref["locus_tag"], ref["cds_seq"]))
qcds = {a: locus2cds.get(l) for a, l in as2locus.items()}
print(f"E. coli CDS available for {sum(v is not None for v in qcds.values())} queries", flush=True)

# ------------------------------------------------------------------ member CDS
need = set(sc["member_row"].dropna().astype(int))
need_arr = np.sort(np.fromiter(need, dtype=np.int64))
mcds = {}
start = 0
for ci, ch in enumerate(pd.read_csv(GFLANKS, chunksize=200_000, usecols=["cds_seq"],
                                    dtype={"cds_seq": "string"})):
    idx = np.arange(start, start + len(ch))
    start += len(ch)
    want = np.intersect1d(idx, need_arr, assume_unique=True)
    if want.size:
        sub = ch.iloc[want - (start - len(ch))]
        for ri, cs in zip(want, sub["cds_seq"]):
            mcds[int(ri)] = cs
    if ci % 10 == 0:
        print(f"  chunk {ci:4d} rows={start:>10,} kept={len(mcds):>7,}", flush=True)
print(f"member CDS pulled: {len(mcds):,}", flush=True)

# ------------------------------------------------------------------ blastp alignments
cols = ["qseqid", "sseqid", "pident", "length", "qstart", "qend", "sstart", "send",
        "bitscore", "qseq", "sseq"]
aln = pd.read_csv(f"{OUT}/query_vs_members_aln.tsv", sep="\t", names=cols)
aln["member_row"] = aln["sseqid"].astype(str).str.split("|").str[0].astype(int)
aln = (aln.sort_values("bitscore", ascending=False)
          .groupby(["qseqid", "member_row"], as_index=False).first())
print(f"alignments: {len(aln):,}", flush=True)


def third_pos_identity(qseq_aln, sseq_aln, qstart, sstart, qcds_seq, scds_seq):
    """Identity at third codon positions over the aligned, ungapped columns."""
    if not isinstance(qcds_seq, str) or not isinstance(scds_seq, str):
        return np.nan, np.nan
    qi, si = qstart - 1, sstart - 1              # 0-based residue index
    same3 = tot3 = same_nt = tot_nt = 0
    for a, b in zip(qseq_aln, sseq_aln):
        if a != "-" and b != "-":
            qo, so = qi * 3, si * 3
            if qo + 3 <= len(qcds_seq) and so + 3 <= len(scds_seq):
                ca, cb = qcds_seq[qo:qo + 3], scds_seq[so:so + 3]
                same3 += ca[2] == cb[2]
                tot3 += 1
                same_nt += sum(x == y for x, y in zip(ca, cb))
                tot_nt += 3
        if a != "-":
            qi += 1
        if b != "-":
            si += 1
    if tot3 < 30:
        return np.nan, np.nan
    return 100.0 * same3 / tot3, 100.0 * same_nt / tot_nt


rows = []
for i, r in enumerate(aln.itertuples()):
    as_row = int(str(r.qseqid).split("|")[0])
    p3, pnt = third_pos_identity(r.qseq, r.sseq, r.qstart, r.sstart,
                                 qcds.get(as_row), mcds.get(r.member_row))
    rows.append((as_row, r.member_row, r.pident, p3, pnt, r.length))
    if i % 5000 == 0:
        print(f"  aln {i:,}/{len(aln):,}", flush=True)
cds = pd.DataFrame(rows, columns=["as_row", "member_row", "pident_prot", "p3", "pnt",
                                  "aln_len"])
df = sc.merge(cds, on=["as_row", "member_row"], how="inner")
df = df[np.isfinite(df["p3"]) & np.isfinite(df["pident_prot"])]
df.to_parquet(f"{OUT}/pairs_cds_singleton_pairs.parquet", index=False)
print(f"pairs with a clock reading: {len(df):,}", flush=True)

# ------------------------------------------------------------------ the test
METRIC = "d2star_k4_test"
CTRL = "d2star_k4_c1"
df["excess"] = df[METRIC] - df[CTRL]

# bins of the clock; inside a bin, elapsed divergence is (approximately) held fixed
bins = [(0, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 101)]
res = {"metric": METRIC, "n_pairs": int(len(df)),
       "spearman_p3_vs_excess_overall": None, "bins": []}
m = np.isfinite(df["excess"])
rho_all, p_all = spearmanr(df.loc[m, "p3"], df.loc[m, "excess"])
res["spearman_p3_vs_excess_overall"] = [float(rho_all), float(p_all)]

for lo, hi in bins:
    b = df[(df["p3"] >= lo) & (df["p3"] < hi) & np.isfinite(df["excess"])]
    if len(b) < 50:
        continue
    rho, p = spearmanr(b["pident_prot"], b["excess"])
    # conserved-protein vs drifted-protein halves at the SAME clock reading
    med = b["pident_prot"].median()
    hi_p = b.loc[b["pident_prot"] > med, "excess"]
    lo_p = b.loc[b["pident_prot"] <= med, "excess"]
    u, pu = mannwhitneyu(hi_p, lo_p, alternative="greater")
    res["bins"].append(dict(
        p3_lo=lo, p3_hi=hi, n=int(len(b)),
        median_p3=float(b["p3"].median()),
        median_pident=float(med),
        spearman_pident_vs_excess=float(rho), spearman_p=float(p),
        median_excess_conserved=float(hi_p.median()),
        median_excess_drifted=float(lo_p.median()),
        mannwhitney_p=float(pu)))

with open(f"{OUT}/cds_conditioning_singleton_pairs.json", "w") as f:
    json.dump(res, f, indent=2)

print(f"\noverall Spearman(P3, excess) = {rho_all:+.3f}  p={p_all:.2e}")
print(f"{'P3 bin':<10}{'n':>7}{'medP3':>8}{'rho(PID)':>10}{'p':>11}"
      f"{'exc_cons':>10}{'exc_drift':>10}{'MW p':>11}")
for b in res["bins"]:
    print(f"{b['p3_lo']:>3}-{b['p3_hi']:<6}{b['n']:>7}{b['median_p3']:>8.1f}"
          f"{b['spearman_pident_vs_excess']:>+10.3f}{b['spearman_p']:>11.2e}"
          f"{b['median_excess_conserved']:>10.4f}{b['median_excess_drifted']:>10.4f}"
          f"{b['mannwhitney_p']:>11.2e}")
print("GRAMMAR_STEP4_OK", flush=True)
