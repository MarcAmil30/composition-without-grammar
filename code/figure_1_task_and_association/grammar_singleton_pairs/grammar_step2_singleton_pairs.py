"""
Singleton-pair robustness version of step 2.

For every E. coli K-12 query we take the MMseqs2 family its best representative belongs to,
collect that family's SINGLETON members across the 412 Gammaproteobacteria, and pull each
member's 200 bp upstream out of the 2.4 GB flanks CSV. Paired controls are also restricted to
singletons and drawn from the SAME target genome, so gene architecture and phylogenetic
distance are identical for test and control.

Controls (see METHODOLOGY.md 2):
  C1  random non-family gene in the same genome, upstream length matched (+/-10%)
  C2  as C1, additionally GC matched (+/-2 points)
  C3  dinucleotide-preserving shuffle of the test upstream itself
  C4  a random *downstream* flank from the same genome -- the "any DNA from here" floor

Writes the sequences only; metrics and statistics are step 3, so the expensive streaming pass
happens once.

    python grammar_step2.py

Output: results/upstream_grammar/ortho_frames_singleton_pairs.parquet
"""
# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
CL_ROOT = _os.environ.get("CL_ROOT", "../Contrastive_Learning_NeurIPS2026")
# -------------------------------------------------------------------

import os
import numpy as np
import pandas as pd

PROJ = f"{CL_ROOT}"
G = f"{PROJ}/results/gamma_rm"
OUT = f"{PROJ}/results/upstream_grammar"
FLANKS = f"{PROJ}/data/gamma_genus/gamma_genus_flanks.csv"
FRAME = 200
MIN_PID, MIN_QCOV = 30.0, 70.0
SEED = 20260813
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- best family per query
cols = ["qseqid", "sseqid", "pident", "length", "qlen", "slen", "bitscore", "evalue", "qcovs"]
hits = pd.read_csv(f"{OUT}/ecoli_vs_reps.tsv", sep="\t", names=cols)
hits = hits[(hits["pident"] >= MIN_PID) & (hits["qcovs"] >= MIN_QCOV)]
best = hits.sort_values("bitscore", ascending=False).groupby("qseqid", as_index=False).first()
best["rep"] = best["sseqid"].astype(str).str.split("|").str[0].astype(int)
print(f"queries with a family hit: {len(best)} of 727", flush=True)

# ---------------------------------------------------------------- family membership
clu = pd.read_csv(f"{G}/gamma_clu_cluster.tsv", sep="\t", names=["rep", "member"])
fam = clu[clu["rep"].isin(set(best["rep"]))]
print(f"family members across the 412 genomes: {len(fam):,}", flush=True)
in_tested_families = set(fam["member"])

genes = pd.read_parquet(f"{G}/gamma_genes.parquet",
                        columns=["accession", "locus_tag", "operon_size", "up_len", "row_idx"])
genes = genes.set_index("row_idx")

members = fam.join(genes, on="member", how="inner")
members = members[(members["operon_size"] == 1) & (members["up_len"] >= FRAME)]
print(f"singleton members with a >= {FRAME} bp upstream: {len(members):,}", flush=True)

# one member per (query, genome): the family is the unit, duplicates within a genome are dropped
q2rep = dict(zip(best["rep"], best["qseqid"]))
members["qseqid"] = members["rep"].map(q2rep)
members = members.sort_values("member").drop_duplicates(["qseqid", "accession"])
print(f"(query, genome) test pairs: {len(members):,}", flush=True)

need = set(members["member"])

# ---------------------------------------------------------------- control candidates
# eligible controls: singleton genes in a target genome with a >= FRAME bp upstream that are not
# in ANY tested family, so the control has the same gene architecture and cannot be a hidden
# homologue of its query.
pool = genes[(genes["operon_size"] == 1) & (genes["up_len"] >= FRAME)].reset_index()
pool = pool[~pool["row_idx"].isin(in_tested_families)]
target_accs = set(members["accession"])
pool = pool[pool["accession"].isin(target_accs)]
print(f"control pool: {len(pool):,} genes in {pool['accession'].nunique()} genomes", flush=True)

by_acc = {a: d for a, d in pool.groupby("accession")}
ctrl_rows = []
for r in members.itertuples():
    cand = by_acc.get(r.accession)
    if cand is None or not len(cand):
        ctrl_rows.append((np.nan, np.nan))
        continue
    lo, hi = r.up_len * 0.9, r.up_len * 1.1
    m = cand[(cand["up_len"] >= lo) & (cand["up_len"] <= hi)]
    if not len(m):
        m = cand
    c1 = int(m["row_idx"].iloc[rng.integers(0, len(m))])
    c4 = int(cand["row_idx"].iloc[rng.integers(0, len(cand))])   # any gene, no matching
    ctrl_rows.append((c1, c4))
members["c1_row"] = [c[0] for c in ctrl_rows]
members["c4_row"] = [c[1] for c in ctrl_rows]
need |= set(members["c1_row"].dropna().astype(int))
need |= set(members["c4_row"].dropna().astype(int))
print(f"rows to pull from the flanks CSV: {len(need):,}", flush=True)

# ---------------------------------------------------------------- stream the flanks CSV once
keep = {}
reader = pd.read_csv(FLANKS, chunksize=200_000,
                     usecols=["upstream_seq", "downstream_seq", "protein_seq", "up_len"],
                     dtype={"upstream_seq": "string", "downstream_seq": "string",
                            "protein_seq": "string"})
need_arr = np.sort(np.fromiter(need, dtype=np.int64))
start = 0
for ci, ch in enumerate(reader):
    idx = np.arange(start, start + len(ch))
    start += len(ch)
    want = np.intersect1d(idx, need_arr, assume_unique=True)
    if want.size:
        sub = ch.iloc[want - (start - len(ch))]
        for ri, up, dn, pr in zip(want, sub["upstream_seq"], sub["downstream_seq"],
                                  sub["protein_seq"]):
            keep[int(ri)] = (up, dn, pr)
    if ci % 10 == 0:
        print(f"  chunk {ci:4d}  rows={start:>10,}  kept={len(keep):>8,}", flush=True)
print(f"pulled {len(keep):,} rows", flush=True)


def up200(ri):
    v = keep.get(int(ri)) if ri == ri else None
    return v[0][-FRAME:] if v and isinstance(v[0], str) and len(v[0]) >= FRAME else None


def down200(ri):
    v = keep.get(int(ri)) if ri == ri else None
    return v[1][:FRAME] if v and isinstance(v[1], str) and len(v[1]) >= FRAME else None


def prot(ri):
    v = keep.get(int(ri)) if ri == ri else None
    return v[2] if v and isinstance(v[2], str) else None


out = pd.DataFrame({
    "qseqid": members["qseqid"].to_numpy(),
    "accession": members["accession"].to_numpy(),
    "member_row": members["member"].to_numpy(),
    "member_locus": members["locus_tag"].to_numpy(),
    "member_operon_size": members["operon_size"].to_numpy(),
    "member_up_len": members["up_len"].to_numpy(),
    "c1_row": members["c1_row"].to_numpy(),
    "test_up": [up200(r) for r in members["member"]],
    "member_prot": [prot(r) for r in members["member"]],
    "c1_up": [up200(r) for r in members["c1_row"]],
    "c4_up": [down200(r) for r in members["c4_row"]],
})
out = out.dropna(subset=["test_up", "c1_up"])
out["as_row"] = out["qseqid"].astype(str).str.split("|").str[0].astype(int)

# taxonomy of each target genome, for the ancestry strata
meta = pd.read_csv(f"{PROJ}/data/gamma_genus/gamma_genus_metadata.csv",
                   usecols=["accession", "species", "genus", "family", "order", "class"])
out = out.merge(meta, on="accession", how="left")
ECOLI = dict(genus="Escherichia", family="Enterobacteriaceae", order="Enterobacterales",
             klass="Gammaproteobacteria")
rank = np.where(out["genus"] == ECOLI["genus"], "same_genus",
        np.where(out["family"] == ECOLI["family"], "same_family",
        np.where(out["order"] == ECOLI["order"], "same_order", "same_class")))
out["taxon_rank"] = rank

os.makedirs(OUT, exist_ok=True)
assert (out["member_operon_size"] == 1).all()
control_operon_size = genes["operon_size"].to_dict()
assert all(control_operon_size[int(r)] == 1 for r in out["c1_row"])
out.to_parquet(f"{OUT}/ortho_frames_singleton_pairs.parquet", index=False)
print(out.groupby("taxon_rank").size().to_string())
print(f"\nwrote {OUT}/ortho_frames_singleton_pairs.parquet  ({len(out):,} pairs, "
      f"{out['as_row'].nunique()} queries, {out['accession'].nunique()} genomes)")
print("GRAMMAR_STEP2_OK", flush=True)
