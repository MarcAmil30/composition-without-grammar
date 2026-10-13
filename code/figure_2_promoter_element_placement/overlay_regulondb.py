#!/usr/bin/env python3
"""
Task-1 dataset prep: overlay RegulonDB PromoterSet onto the 200 bp upstream
windows of the E. coli up_only genes (operon leaders + singletons).

Answers:
  (a) how many of the 2720 up_only genes match a RegulonDB promoter
  (b) of those, how many have promoter boxes IN-WINDOW, and the per-gene
      annotation coverage of the 200 bp (fraction covered / gap).

Coordinate convention (GFF is 1-based inclusive):
  + strand gene: ATG at gene.start ; upstream window = [start-200, start-1]
  - strand gene: ATG at gene.end   ; upstream window = [end+1,  end+200]
  upstream distance d (1..200): +: start-g ; -: g-end   (d=1 is adjacent to ATG)
  window index (matches 5'->3' upstream_200bp string): idx = L - d
      where L = window length (usually 200); idx 0 = 5'-most, idx L-1 = next to ATG
"""
# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
WORK_ROOT = _os.environ.get("WORK_ROOT", _os.path.expanduser("~"))
# -------------------------------------------------------------------

import csv, sys, os
from collections import defaultdict

BASE = f"{WORK_ROOT}"
RAW  = f"{BASE}/Proking_repo/data/raw_sequences_data_small_scale/Escherichia_coli_operon_annotated.csv"
GFF  = f"{BASE}/Proking_repo/code/annotation/ecoli_annotation/GCF_000005845.2_ASM584v2_genomic.gff"
PROM = f"{BASE}/ICBINB_BIO_NeurIPS2026/data/regulondb/PromoterSet.tsv"
OUTDIR = f"{BASE}/ICBINB_BIO_NeurIPS2026/data"
UP_STATUS = {"singleton_or_non_operon", "first_operon_gene"}

def comp(s):
    return s.translate(str.maketrans("ACGTacgt", "TGCAtgca"))
def rc(s):
    return comp(s)[::-1]

# ---------- 1. up_only genes (locus_tag -> upstream seq) ----------
up = {}
with open(RAW) as f:
    for r in csv.DictReader(f):
        if r["status"] in UP_STATUS:
            up[r["locus_tag"]] = r["upstream_200bp"].strip().upper()
print(f"[up_only] genes = {len(up)}")

# ---------- 2. GFF: locus_tag <-> name, ATG coord, strand ----------
gene_by_lt = {}     # locus_tag -> (name, atg_pos, strand, start, end)
lt_by_name = {}     # gene name -> locus_tag
with open(GFF) as f:
    for line in f:
        if line.startswith("#"): continue
        c = line.rstrip("\n").split("\t")
        if len(c) < 9 or c[2] != "gene": continue
        start, end, strand, attr = int(c[3]), int(c[4]), c[6], c[8]
        d = dict(kv.split("=", 1) for kv in attr.split(";") if "=" in kv)
        lt = d.get("locus_tag"); name = d.get("Name") or d.get("gene")
        if not lt: continue
        atg = start if strand == "+" else end
        gene_by_lt[lt] = (name, atg, strand, start, end)
        if name: lt_by_name[name] = lt
print(f"[gff] genes = {len(gene_by_lt)} ; named = {len(lt_by_name)}")

# ---------- 3. PromoterSet: group promoters by locus_tag ----------
def parse_box(s):
    # "1825694-1825699" -> (1825694, 1825699) ; may be empty
    s = (s or "").strip()
    if "-" not in s: return None
    a, b = s.split("-")[:2]
    try: return (int(a), int(b))
    except ValueError: return None

proms_by_lt = defaultdict(list)
n_prom = 0; n_prom_named = 0
with open(PROM) as f:
    rdr = csv.reader(f, delimiter="\t")
    header = None
    for row in rdr:
        if not row or row[0].startswith("#"): continue
        if header is None and row[0].strip().startswith("1)"):
            header = row; continue
        if header is None: continue
        rec = dict(zip([h.split(")",1)[-1].strip() for h in header], row))
        n_prom += 1
        gname = (rec.get("firstGeneName") or "").strip()
        lt = lt_by_name.get(gname)
        if not lt: continue
        n_prom_named += 1
        proms_by_lt[lt].append({
            "name": rec.get("name","").strip(),
            "sigma": rec.get("sigmaFactor","").strip(),
            "tss": int(rec["posTSS"]) if rec.get("posTSS","").strip().lstrip("-").isdigit() else None,
            "m10": parse_box(rec.get("boxMinus10pos")),
            "m35": parse_box(rec.get("boxMinus35pos")),
            "m10seq": (rec.get("boxMinus10seq") or "").strip().upper(),
            "m35seq": (rec.get("boxMinus35seq") or "").strip().upper(),
            "conf": rec.get("confidenceLevel","").strip(),
        })
print(f"[promoters] total={n_prom} ; mapped-to-a-gene-name={n_prom_named} ; genes-with-promoter={len(proms_by_lt)}")

# ---------- 4. Overlay onto each up_only window ----------
def dist_range(box, atg, strand):
    """upstream distance range (d_min,d_max), d>=1 means upstream of ATG."""
    bL, bR = box
    if strand == "+":
        d1, d2 = atg - bR, atg - bL     # larger g -> smaller d
    else:
        d1, d2 = bL - atg, bR - atg
    return (min(d1, d2), max(d1, d2))

rows_out = []
matched = 0                 # up_only genes with >=1 mapped promoter
in_window_any = 0           # ... with >=1 box overlapping the 200 bp window
cov_boxes_list = []         # coverage fraction (just -35 + -10 boxes)
cov_footprint_list = []     # coverage fraction (-35 start .. TSS span)
validated = 0; validate_attempt = 0

for lt, useq in up.items():
    L = len(useq)
    g = gene_by_lt.get(lt)
    proms = proms_by_lt.get(lt, [])
    if not g or not proms:
        rows_out.append((lt, 0, 0, 0.0, 0.0, ""))
        continue
    matched += 1
    name, atg, strand, start, end = g
    covered_boxes = [False]*L
    covered_fp   = [False]*L
    any_in = False; details = []
    for p in proms:
        marks = []  # (label, d_min, d_max)
        for lab, box in (("m35", p["m35"]), ("m10", p["m10"])):
            if box: marks.append((lab,)+dist_range(box, atg, strand))
        # TSS as single position
        if p["tss"] is not None:
            dt = (atg - p["tss"]) if strand == "+" else (p["tss"] - atg)
            marks.append(("tss", dt, dt))
        in_this = False
        for lab, dmin, dmax in marks:
            for d in range(dmin, dmax+1):
                if 1 <= d <= L:
                    idx = L - d
                    if 0 <= idx < L:
                        if lab in ("m35","m10"): covered_boxes[idx] = True
                        covered_fp[idx] = True
                        in_this = True
        # footprint fill: from -35 start to TSS (span)
        ds = [d for _,a,b in marks for d in (a,b)]
        if ds:
            for d in range(max(1,min(ds)), min(L,max(ds))+1):
                covered_fp[L-d] = True
        if in_this:
            any_in = True
            details.append(f"{p['name']}({p['sigma']},{p['conf']})")
        # sequence validation: does m10seq sit where we computed?
        if p["m10"] and p["m10seq"]:
            validate_attempt += 1
            dmin, dmax = dist_range(p["m10"], atg, strand)
            if 1 <= dmin and dmax <= L:
                idx0 = L - dmax; idx1 = L - dmin + 1
                frag = useq[idx0:idx1]
                if frag == p["m10seq"] or rc(frag) == p["m10seq"]:
                    validated += 1
    if any_in: in_window_any += 1
    cov_boxes_list.append(sum(covered_boxes)/L)
    cov_footprint_list.append(sum(covered_fp)/L)
    rows_out.append((lt, len(proms), int(any_in),
                     sum(covered_boxes)/L, sum(covered_fp)/L, ";".join(details)))

# ---------- 5. Report ----------
N = len(up)
def pct(x, d): return f"{x} ({100*x/d:.1f}%)"
print("\n================ TASK-1 OVERLAY SUMMARY ================")
print(f"up_only genes (E. coli)                : {N}")
print(f"  matched to >=1 RegulonDB promoter    : {pct(matched, N)}")
print(f"  with >=1 promoter box IN 200bp window: {pct(in_window_any, N)}")
print(f"  matched but promoter OUTSIDE window  : {pct(matched-in_window_any, N)}")
if validate_attempt:
    print(f"seq-validation (-10 box lands where predicted): {pct(validated, validate_attempt)}")
def stats(v, label):
    if not v: print(f"  {label}: n=0"); return
    v2 = sorted(v); n=len(v2)
    mean=sum(v2)/n; med=v2[n//2]
    print(f"  {label}: mean={100*mean:.1f}%  median={100*med:.1f}%  "
          f"min={100*min(v2):.1f}%  max={100*max(v2):.1f}%  (n={n})")
print("\nAnnotation coverage of the 200 bp (genes matched IN-window):")
stats([c for c in cov_boxes_list if c>0], "  boxes only (-35 + -10)")
stats([c for c in cov_footprint_list if c>0], "  footprint (-35 .. TSS)")
print("=> Expect small coverage: promoter boxes are point features; the rest is")
print("   the ~unannotated filler that gets randomized for the CLEAN set.")

# ---------- 6. Write per-gene table ----------
os.makedirs(OUTDIR, exist_ok=True)
outp = f"{OUTDIR}/ecoli_up_only_regulondb_overlay.csv"
with open(outp, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["locus_tag","n_promoters","box_in_window",
                "cov_boxes_frac","cov_footprint_frac","promoters"])
    w.writerows(rows_out)
print(f"\n[written] {outp}  ({len(rows_out)} rows)")
