"""
Singleton-pair robustness version of step 3: score every pair and run the statistics.

For each (E. coli query, target genome) pair we score the query's 200 bp upstream against
  test  the orthologue's upstream in that genome
  C1    a length-matched non-family gene's upstream in the SAME genome
  C4    a random downstream flank in the same genome  ("any DNA from here" floor)
  C3    a dinucleotide-preserving shuffle of the test upstream

Two metrics at k = 3, 4, 5:
  pearson  r between raw k-mer count vectors -- assumption-free, composition-sensitive
  d2star   normalised D2*, k-mer counts divided by their expectation under an order-(k-2)
           Markov background fitted to each sequence, so composition is handled inside the
           statistic (Reinert/Waterman family of alignment-free statistics)

Statistics
  1  test vs C1, paired Wilcoxon, per taxonomic stratum          -- pipeline positive control
  2  Spearman(similarity, protein identity) WITHIN each stratum  -- the actual test: does
     upstream similarity track protein similarity at matched ancestry?

    python grammar_step3.py

Outputs: results/upstream_grammar/ortho_scored_singleton_pairs.parquet,
         grammar_summary_singleton_pairs.json
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
from scipy.stats import wilcoxon, spearmanr

PROJ = f"{CL_ROOT}"
OUT = f"{PROJ}/results/upstream_grammar"
KS = [3, 4, 5]
SEED = 20260813
RANKS = ["same_genus", "same_family", "same_order", "same_class"]

B2I = {"A": 0, "C": 1, "G": 2, "T": 3}


def encode(seq):
    a = np.full(len(seq), -1, np.int8)
    for b, i in B2I.items():
        a[np.frombuffer(seq.encode(), dtype="S1") == b.encode()] = i
    return a


def kmer_counts(code, k):
    """Counts of all 4**k k-mers; windows containing a non-ACGT base are dropped."""
    n = len(code) - k + 1
    if n <= 0:
        return np.zeros(4 ** k, np.float64)
    idx = np.zeros(n, np.int64)
    ok = np.ones(n, bool)
    for j in range(k):
        c = code[j:j + n]
        ok &= c >= 0
        idx = idx * 4 + np.where(c >= 0, c, 0)
    return np.bincount(idx[ok], minlength=4 ** k).astype(np.float64)


def expected_markov(code, k):
    """E[count of w] under the order-(k-2) Markov chain fitted to this sequence:
       E(w) = N(prefix_{k-1}) * N(suffix_{k-1}) / N(middle_{k-2}) ."""
    ck = kmer_counts(code, k)
    c1 = kmer_counts(code, k - 1)
    c2 = kmer_counts(code, k - 2) if k >= 3 else None
    w = np.arange(4 ** k)
    pre = w // 4                      # first k-1 bases
    suf = w % (4 ** (k - 1))          # last k-1 bases
    mid = (w // 4) % (4 ** (k - 2)) if k >= 3 else None
    if c2 is None:
        e = c1[pre] * c1[suf] / max(ck.sum(), 1)
    else:
        denom = np.where(c2[mid] > 0, c2[mid], np.nan)
        e = c1[pre] * c1[suf] / denom
    return ck, np.nan_to_num(e, nan=0.0)


def pearson(a, b):
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def d2star(ca, ea, cb, eb):
    """Normalised D2*: correlation of (count - expected)/sqrt(expected)."""
    m = (ea > 0) & (eb > 0)
    if m.sum() < 10:
        return np.nan
    x = (ca[m] - ea[m]) / np.sqrt(ea[m])
    y = (cb[m] - eb[m]) / np.sqrt(eb[m])
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else np.nan


def dinuc_shuffle(seq, rng):
    if len(seq) < 3:
        return seq
    graph = {}
    for a, b in zip(seq[:-1], seq[1:]):
        graph.setdefault(a, []).append(b)
    last = seq[-1]
    for _ in range(20):
        g = {k: v[:] for k, v in graph.items()}
        for k in g:
            rng.shuffle(g[k])
        out, cur, ok = [seq[0]], seq[0], True
        for _ in range(sum(len(v) for v in g.values())):
            if not g.get(cur):
                ok = False
                break
            nxt = g[cur].pop()
            out.append(nxt)
            cur = nxt
        if ok and len(out) == len(seq) and out[-1] == last:
            return "".join(out)
    return "".join(out)


# ------------------------------------------------------------------ data
df = pd.read_parquet(f"{OUT}/ortho_frames_singleton_pairs.parquet")
q = pd.read_parquet(f"{PROJ}/results/step0/analysis_set.parquet",
                    columns=["upstream_seq", "species", "locus_tag"])
q = q[q["species"] == "Escherichia coli"]
qup = {int(i): s[-200:] for i, s in zip(q.index, q["upstream_seq"]) if isinstance(s, str)}
df = df[df["as_row"].isin(qup)].reset_index(drop=True)
print(f"pairs: {len(df):,} | queries: {df['as_row'].nunique()} | genomes: {df['accession'].nunique()}",
      flush=True)

# per-pair protein identity from the targeted blastp (step 3a)
pid_path = f"{OUT}/query_vs_members.tsv"
if os.path.exists(pid_path):
    c = ["qseqid", "sseqid", "pident", "length", "qlen", "slen", "bitscore", "evalue", "qcovs"]
    pid = pd.read_csv(pid_path, sep="\t", names=c)
    pid["member_row"] = pid["sseqid"].astype(str).str.split("|").str[0].astype(int)
    pid = (pid.sort_values("bitscore", ascending=False)
              .groupby(["qseqid", "member_row"], as_index=False).first())
    df = df.merge(pid[["qseqid", "member_row", "pident", "qcovs"]],
                  on=["qseqid", "member_row"], how="left")
    print(f"protein identity attached for {df['pident'].notna().sum():,} pairs", flush=True)
else:
    df["pident"] = np.nan
    print("WARNING: no query_vs_members.tsv -- protein identity unavailable", flush=True)

# ------------------------------------------------------------------ score
rng = np.random.default_rng(SEED)
qcache = {}
rows = []
for i, r in enumerate(df.itertuples()):
    key = r.as_row
    if key not in qcache:
        code = encode(qup[key])
        qcache[key] = {k: expected_markov(code, k) for k in KS}
    rec = dict(as_row=r.as_row, accession=r.accession, member_row=r.member_row,
               taxon_rank=r.taxon_rank, pident=r.pident)
    seqs = {"test": r.test_up, "c1": r.c1_up, "c4": r.c4_up,
            "c3": dinuc_shuffle(r.test_up, rng)}
    for name, s in seqs.items():
        if not isinstance(s, str) or len(s) < 50:
            for k in KS:
                rec[f"pearson_k{k}_{name}"] = np.nan
                rec[f"d2star_k{k}_{name}"] = np.nan
            continue
        code = encode(s)
        for k in KS:
            ca, ea = qcache[key][k]
            cb, eb = expected_markov(code, k)
            rec[f"pearson_k{k}_{name}"] = pearson(ca, cb)
            rec[f"d2star_k{k}_{name}"] = d2star(ca, ea, cb, eb)
    rows.append(rec)
    if i % 5000 == 0:
        print(f"  {i:,}/{len(df):,}", flush=True)

sc = pd.DataFrame(rows)
sc.to_parquet(f"{OUT}/ortho_scored_singleton_pairs.parquet", index=False)
print(f"wrote {OUT}/ortho_scored_singleton_pairs.parquet", flush=True)

# ------------------------------------------------------------------ statistics
summary = {"n_pairs": int(len(sc)), "n_queries": int(sc["as_row"].nunique()),
           "n_genomes": int(sc["accession"].nunique()), "strata": {}}
for rank in RANKS:
    s = sc[sc["taxon_rank"] == rank]
    if len(s) < 20:
        continue
    d = {"n": int(len(s)), "n_genomes": int(s["accession"].nunique()), "metrics": {}}
    for metric in ("pearson", "d2star"):
        for k in KS:
            t = s[f"{metric}_k{k}_test"].to_numpy()
            c1 = s[f"{metric}_k{k}_c1"].to_numpy()
            c3 = s[f"{metric}_k{k}_c3"].to_numpy()
            c4 = s[f"{metric}_k{k}_c4"].to_numpy()
            m = np.isfinite(t) & np.isfinite(c1)
            if m.sum() < 20:
                continue
            stat, p = wilcoxon(t[m], c1[m], alternative="greater")
            entry = dict(n=int(m.sum()),
                         median_test=float(np.nanmedian(t[m])),
                         median_c1=float(np.nanmedian(c1[m])),
                         median_c3=float(np.nanmedian(c3[m])),
                         median_c4=float(np.nanmedian(c4[m])),
                         delta_test_minus_c1=float(np.nanmedian(t[m] - c1[m])),
                         wilcoxon_p=float(p))
            # the actual test: does similarity track protein identity at matched ancestry?
            pi = s["pident"].to_numpy()
            mm = m & np.isfinite(pi)
            if mm.sum() >= 30:
                rho, pp = spearmanr(pi[mm], t[mm])
                entry["spearman_pident_vs_test"] = float(rho)
                entry["spearman_p"] = float(pp)
                rho_c, _ = spearmanr(pi[mm], c1[mm])
                entry["spearman_pident_vs_c1"] = float(rho_c)
            d["metrics"][f"{metric}_k{k}"] = entry
    summary["strata"][rank] = d

with open(f"{OUT}/grammar_summary_singleton_pairs.json", "w") as f:
    json.dump(summary, f, indent=2)

for rank, d in summary["strata"].items():
    e = d["metrics"].get("d2star_k4")
    if e:
        print(f"{rank:<13} n={d['n']:>6}  D2*k4 test={e['median_test']:.3f} "
              f"C1={e['median_c1']:.3f} C3={e['median_c3']:.3f}  "
              f"delta={e['delta_test_minus_c1']:+.3f}  p={e['wilcoxon_p']:.2e}  "
              f"rho(pident)={e.get('spearman_pident_vs_test', float('nan')):+.3f}")
print("GRAMMAR_STEP3_OK", flush=True)
