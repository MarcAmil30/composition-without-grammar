"""
Build the E. coli CDS<->upstream RETRIEVAL gene set for the benchmark.

Same extraction as the C3P/Evo retrieval runs (data/ref2/ref2_flanks.csv,
GCF_000005845.2, operon window 50 bp) but WITHOUT the `down_len >= 100` filter:
every model in this benchmark is upstream-only or upstream-first, so requiring a
valid downstream flank threw away half the genes for no modelling reason
(mishap #4 in NOTES/C3P_ANALYSIS_BREAKDOWN.md).

Funnel:  E. coli CDS 4290 -> singleton 1896 -> up_len>=100 -> 1453
(the old C3P/Evo set = these 1453 further cut by down_len>=100 -> 727, kept as a flag)

E. coli only: C3P never saw E. coli, so it is the clean held-out test species.

Out: ICBINB_BIO_NeurIPS2026/data/benchmark/retrieval/
       ecoli_retrieval_1453.csv
       ecoli_retrieval_summary.json
"""
# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
ICBINB_ROOT = _os.environ.get("ICBINB_ROOT", str(_Path(__file__).resolve().parents[3]))
CL_ROOT = _os.environ.get("CL_ROOT", "../Contrastive_Learning_NeurIPS2026")
# -------------------------------------------------------------------

import json
import os

import pandas as pd

CL = f"{CL_ROOT}"
IC = f"{ICBINB_ROOT}"
BENCH = f"{IC}/data/benchmark"
OUT = f"{BENCH}/retrieval"
ACC = "GCF_000005845.2"
MIN_UP = 100

os.makedirs(OUT, exist_ok=True)

flanks = pd.read_csv(f"{CL}/data/ref2/ref2_flanks.csv")
ec = flanks[flanks.accession == ACC].copy()

funnel = {"all_gene_rows": len(ec)}
ec = ec[ec.gene_type == "CDS"]
funnel["CDS"] = len(ec)
ec = ec[ec.protein_seq.notna() & (ec.protein_seq.astype(str).str.len() > 0)]
funnel["protein_nonempty"] = len(ec)
ec = ec[ec.operon_size == 1]
funnel["singleton_operon_size_1"] = len(ec)
ec = ec[ec.up_len >= MIN_UP]
funnel[f"up_len_ge_{MIN_UP}"] = len(ec)
funnel["also_down_len_ge_100_old_c3p_set"] = int((ec.down_len >= 100).sum())

ec["cds_len"] = ec.cds_seq.str.len()
ec["prot_len"] = ec.protein_seq.str.len()
ec["in_c3p_727"] = ec.down_len >= 100

# ---- cross-reference the other benchmark sets -------------------------------
cdsp = pd.read_csv(f"{BENCH}/cds_prompted/ecoli_singletons_CAI.csv")
ec["in_cds_prompted"] = ec.locus_tag.isin(cdsp.locus_tag)
ec = ec.merge(cdsp[["locus_tag", "CAI"]], on="locus_tag", how="left")

master = pd.read_csv(f"{BENCH}/ecoli_upstream_master.csv", usecols=["locus_tag", "tier"])
ec = ec.merge(master.rename(columns={"tier": "bench_tier"}), on="locus_tag", how="left")

# ---- sanity checks ----------------------------------------------------------
checks = {}
checks["unique_locus_tag"] = bool(ec.locus_tag.is_unique)
checks["cds_starts_with_start_codon"] = float(
    ec.cds_seq.str[:3].isin(["ATG", "GTG", "TTG", "ATT", "CTG"]).mean()
)
checks["prot_len_eq_cds_len_div3_minus1"] = float(
    (ec.prot_len == ec.cds_len // 3 - 1).mean()
)
checks["upstream_len_matches_up_len"] = bool((ec.upstream_seq.str.len() == ec.up_len).all())
checks["min_up_len"] = int(ec.up_len.min())
checks["n_in_cds_prompted"] = int(ec.in_cds_prompted.sum())
checks["n_in_c3p_727"] = int(ec.in_c3p_727.sum())
checks["bench_tier_counts"] = ec.bench_tier.fillna("none").value_counts().to_dict()

COLS = [
    "accession", "locus_tag", "gene", "protein_id", "strand", "operon_id",
    "up_len", "down_len", "cds_len", "prot_len",
    "CAI", "bench_tier", "in_c3p_727", "in_cds_prompted",
    "upstream_seq", "cds_seq", "protein_seq",
]
ec = ec[COLS].sort_values("locus_tag").reset_index(drop=True)

csv_path = f"{OUT}/ecoli_retrieval_{len(ec)}.csv"
ec.to_csv(csv_path, index=False)

summary = {
    "species": "Escherichia coli K-12 MG1655",
    "accession": ACC,
    "n_genes": len(ec),
    "source": f"{CL}/data/ref2/ref2_flanks.csv",
    "operon_window_bp": 50,
    "filters": ["gene_type==CDS", "protein_seq non-empty", "operon_size==1", f"up_len>={MIN_UP}"],
    "dropped_filter_vs_c3p_run": "down_len>=100 (models are upstream-only; see mishap #4)",
    "funnel": funnel,
    "checks": checks,
    "csv": csv_path,
}
with open(f"{OUT}/ecoli_retrieval_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)

print(json.dumps(summary, indent=2))
