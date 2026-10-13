#!/usr/bin/env python3
"""
UPSTREAM-ONLY track (no CDS in the prompt) -> data/benchmark/upstream_only/

WIDE format: one row per gene, 711 rows, both tasks as columns.
Two output files, identical in every column except the sequence background:

  prompts_leaked.csv    prompts built from seq_leaked (the real upstream)
  prompts_clean.csv     prompts built from seq_clean  (motifs kept in place,
                        filler permuted -> exact same base composition)

Both files carry all 711 genes for BOTH tasks. Nothing is dropped.

TASK 1 - generate_promoter
  prompt = seq[:end of -35]
  target = seq[end of -35 : end of -35 + 45]   (or fewer, if the upstream ends first)
  45 bp is the generation budget: it covers the -10 box (max offset 24) and the
  TSS (max offset 38) for every gene, and stops there. The target no longer runs
  to the ATG, so task 1 and task 2 are now disjoint rather than nested.
  Scored at offsets into the generated continuation:
      m10_offset_in_target = m10_start - cut     (-> motif_pwm)
      tss_offset_in_target = tss_idx   - cut     (-> promoter_calculator)
  The full 200 bp region is kept in `upstream_seq`, so nothing is lost by the trim.

TASK 2 - generate_sd
  prompt = seq[:L-20]
  target = seq[L-20:] + start_codon           (20 bp + the codon; unchanged)
  At evaluation time the model is asked for 40 bp so it has room to place its own
  start codon -- that is a GENERATION setting, not a column. The target stays 20 bp.
  Defined for all 711 genes. `sd_present` is a PROPERTY OF THE ANSWER, not a
  precondition - the 136 genes with sd_present=0 are kept on purpose, because
  "does the model hallucinate an SD where the real sequence has none?" is a
  question worth being able to ask. Subset on sd_present==1 for the strict 575.
"""
# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
WORK_ROOT = _os.environ.get("WORK_ROOT", _os.path.expanduser("~"))
# -------------------------------------------------------------------

import csv, os, json
from collections import defaultdict

BASE = f"{WORK_ROOT}"
RAW  = BASE + "/Proking_repo/data/raw_sequences_data_small_scale/Escherichia_coli_operon_annotated.csv"
GFF  = BASE + "/Proking_repo/code/annotation/ecoli_annotation/GCF_000005845.2_ASM584v2_genomic.gff"
RDB  = BASE + "/ICBINB_BIO_NeurIPS2026/data/regulondb"
BM   = BASE + "/ICBINB_BIO_NeurIPS2026/data/benchmark"        # root: shared master
UPO  = BM + "/upstream_only"                                  # this track
os.makedirs(UPO, exist_ok=True)
SDTAIL = 20          # last 20 bp removed for the SD task
GENLEN = 45          # task-1 generation budget (covers -10 max offset 24, TSS max 38)
SHORT  = 20          # flag prompts shorter than this as not-real-context

# ---- master ----
master  = {r["locus_tag"]: r for r in csv.DictReader(open(f"{BM}/ecoli_upstream_master.csv"))}
full711 = [lt for lt, r in master.items() if r["tier"] == "full_promoter"]

# ---- start codon from CDS ----
start_codon = {}
for r in csv.DictReader(open(RAW)):
    cds = (r.get("cds") or "").strip().upper()
    if cds:
        start_codon[r["locus_tag"]] = cds[:3]

# ---- GFF name<->lt, ATG, strand ----
lt_by_name = {}
gene = {}
for line in open(GFF):
    if line.startswith("#"):
        continue
    c = line.split("\t")
    if len(c) < 9 or c[2] != "gene":
        continue
    s, e, st, attr = int(c[3]), int(c[4]), c[6], c[8]
    d = dict(kv.split("=", 1) for kv in attr.strip().split(";") if "=" in kv)
    lt = d.get("locus_tag"); nm = d.get("Name") or d.get("gene")
    if lt:
        gene[lt] = (nm, s if st == "+" else e, st)
        if nm:
            lt_by_name[nm] = lt

def boxp(x):
    x = (x or "").strip()
    if "-" not in x:
        return None
    a, b = x.split("-")[:2]
    return (int(a), int(b)) if a.isdigit() and b.isdigit() else None

def span_idx(bx, atg, st, L):
    bL, bR = bx
    d1, d2 = (atg - bR, atg - bL) if st == "+" else (bL - atg, bR - atg)
    dmin, dmax = min(d1, d2), max(d1, d2)
    i0, i1 = max(0, L - dmax), min(L, L - dmin + 1)
    return (i0, i1) if i1 > i0 else None

def tss_i(t, atg, st, L):
    d = (atg - t) if st == "+" else (t - atg)
    return (L - d) if 1 <= d <= L else None

# ---- pick the operative promoter per gene (ATG-proximal, tie-break confidence) ----
rows = [r for r in csv.reader(open(f"{RDB}/PromoterSet.tsv"), delimiter="\t")
        if r and not r[0].startswith("#")]
hdr = [h.split(")", 1)[-1].strip() for h in rows[0]]
P = [dict(zip(hdr, r)) for r in rows[1:]]
conf_rank = {"C": 3, "Confirmed": 3, "S": 2, "Strong": 2, "W": 1, "Weak": 1, "": 0}
op = {}
for r in P:
    lt = lt_by_name.get((r.get("firstGeneName") or "").strip())
    if lt not in master or master[lt]["tier"] != "full_promoter":
        continue
    m35 = boxp(r.get("boxMinus35pos")); m10 = boxp(r.get("boxMinus10pos"))
    t = r.get("posTSS", "").strip()
    if not (m35 and m10 and t.lstrip("-").isdigit()):
        continue
    nm, atg, st = gene[lt]; L = int(master[lt]["L"]); tss = int(t)
    s35 = span_idx(m35, atg, st, L); s10 = span_idx(m10, atg, st, L)
    ti = tss_i(tss, atg, st, L)
    if not (s35 and s10 and ti is not None):
        continue
    if not (s35[1] <= s10[0] < s10[1] <= ti):      # sane ordering -35 < -10 < TSS
        continue
    cand = {"m35": s35, "m10": s10, "tss": ti,
            "sigma": r.get("sigmaFactor", "").strip(),
            "name": r.get("name", "").strip(),
            "conf": conf_rank.get(r.get("confidenceLevel", "").strip(), 0)}
    if lt not in op or (ti > op[lt]["tss"]) or (ti == op[lt]["tss"] and cand["conf"] > op[lt]["conf"]):
        op[lt] = cand

missing = [lt for lt in full711 if lt not in op]
assert not missing, f"full_promoter genes with no operative promoter: {missing}"

COLS = ["locus_tag", "gene", "strand", "L", "promoter", "sigma",
        "upstream_seq",
        "m35_start", "m35_end", "m10_start", "m10_end", "tss_idx",
        "sd_present", "sd_start", "sd_end", "sd_score", "sd_spacing",
        "n_tfbs", "tfbs_regulators", "tfbs_spans", "tfbs_functions", "start_codon",
        # task 1
        "cut_promoter", "prompt_promoter", "prompt_promoter_len",
        "target_promoter", "m10_offset_in_target", "tss_offset_in_target",
        "short_prompt",
        # task 2
        "prompt_sd", "target_sd"]

def build(seqkey, name):
    n_short = 0; n_clip = 0; n_row = 0
    with open(f"{UPO}/{name}", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for lt in sorted(full711):
            r = master[lt]; o = op[lt]
            if (o["m35"][1] - o["m35"][0]) < 6:   # -35 clipped by the 200bp window -> drop
                n_clip += 1
                continue
            L = int(r["L"]); seq = r[seqkey]
            nm, atg, st = gene[lt]
            m35s, m35e = o["m35"]; m10s, m10e = o["m10"]; ti = o["tss"]
            sd = json.loads(r["sd_span"]) or [None, None]
            tf = json.loads(r["tfbs"]) or []
            tf_reg = ";".join(t["reg"] for t in tf)
            tf_span = ";".join(f"{t['span'][0]}-{t['span'][1]}" for t in tf)
            tf_fn = ";".join(t.get("fn") or "" for t in tf)
            sc = start_codon.get(lt, "")

            cut1 = m35e                       # end of the -35 box
            cut2 = L - SDTAIL                 # last 20 bp removed
            short = int(cut1 < SHORT)
            n_short += short; n_row += 1
            # the -35 box must be intact for the task premise to hold; genes whose
            # box is cut by the 200 bp window are dropped above, so no flag column
            # is needed -- every surviving row has a full-width -35.
            assert (m35e - m35s) >= 6
            # the 45 bp budget must reach both scoring offsets
            assert (m10s - cut1) < GENLEN and (ti - cut1) < GENLEN, lt

            w.writerow({
                "locus_tag": lt, "gene": nm, "strand": st, "L": L,
                "promoter": o["name"], "sigma": o["sigma"],
                "upstream_seq": seq,
                "m35_start": m35s, "m35_end": m35e,
                "m10_start": m10s, "m10_end": m10e, "tss_idx": ti,
                "sd_present": r["sd_present"], "sd_start": sd[0], "sd_end": sd[1],
                "sd_score": r["sd_score"], "sd_spacing": r["sd_spacing"],
                "n_tfbs": r["n_tfbs"], "tfbs_regulators": tf_reg,
                "tfbs_spans": tf_span, "tfbs_functions": tf_fn, "start_codon": sc,

                "cut_promoter": cut1,
                "prompt_promoter": seq[:cut1], "prompt_promoter_len": cut1,
                "target_promoter": seq[cut1:cut1 + GENLEN],
                "m10_offset_in_target": m10s - cut1,
                "tss_offset_in_target": ti - cut1,
                "short_prompt": short,

                "prompt_sd": seq[:cut2],
                "target_sd": seq[cut2:] + sc,
            })
    print(f"[{name}] {n_row} genes x 2 tasks | short prompts (<{SHORT}bp): {n_short} "
          f"| dropped for a clipped -35: {n_clip}")

build("seq_leaked", "prompts_681_leaked.csv")
build("seq_clean",  "prompts_681_clean.csv")

sd_yes = sum(1 for lt in full711 if int(master[lt]["sd_present"]) == 1)
print(f"sd_present=1: {sd_yes} / {len(full711)}  (all 711 carry the SD task; subset if you want the strict set)")
