#!/usr/bin/env python3
"""
Build Task-1 benchmark dataset (E. coli up_only, 200 bp upstream).

Annotation layer (per gene, positions as indices into the 5'->3' upstream string,
idx 0 = 5'-most = -L, idx L-1 = adjacent to ATG):
  -35 box, -10 box, TSS        : RegulonDB PromoterSet (validated 99.9%)
  Shine-Dalgarno (SD)          : computed (best AGGAGG-consensus / anti-SD match in last ~20bp)
  TF binding sites (operators) : RegulonDB RISet

Outputs:
  data/benchmark/ecoli_upstream_master.csv  : one row/gene, leaked+clean seqs + full
      annotation. SHARED root table - both benchmark tracks (upstream_only/ and
      cds_prompted/) are derived from it, so it does not live in either subfolder.
LEAKED  = real extracted upstream (models have seen it)
CLEAN   = keep annotated motif bases at fixed positions; randomize the filler at
          matched per-sequence composition (positions of motifs fixed => spacing preserved)
"""
# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
WORK_ROOT = _os.environ.get("WORK_ROOT", _os.path.expanduser("~"))
# -------------------------------------------------------------------

import csv, os, random, json
from collections import defaultdict

BASE=f"{WORK_ROOT}"
RAW =BASE+"/Proking_repo/data/raw_sequences_data_small_scale/Escherichia_coli_operon_annotated.csv"
GFF =BASE+"/Proking_repo/code/annotation/ecoli_annotation/GCF_000005845.2_ASM584v2_genomic.gff"
RDB =BASE+"/ICBINB_BIO_NeurIPS2026/data/regulondb"
OUT =BASE+"/ICBINB_BIO_NeurIPS2026/data/benchmark"
UP  ={"singleton_or_non_operon","first_operon_gene"}
os.makedirs(OUT, exist_ok=True)

comp=lambda s:s.translate(str.maketrans("ACGTacgt","TGCAtgca")); rc=lambda s:comp(s)[::-1]

# ---- inputs ----
up={r["locus_tag"]:r["upstream_200bp"].strip().upper() for r in csv.DictReader(open(RAW)) if r["status"] in UP}
lt_by_name={}; gene={}
for line in open(GFF):
    if line.startswith("#"): continue
    c=line.split("\t")
    if len(c)<9 or c[2]!="gene": continue
    s,e,st,attr=int(c[3]),int(c[4]),c[6],c[8]
    d=dict(kv.split("=",1) for kv in attr.strip().split(";") if "=" in kv)
    lt=d.get("locus_tag"); nm=d.get("Name") or d.get("gene")
    if lt:
        gene[lt]=(nm,s if st=="+" else e,st)
        if nm: lt_by_name[nm]=lt
def read_tsv(p):
    rows=[r for r in csv.reader(open(p),delimiter="\t") if r and not r[0].startswith("#")]
    h=[x.split(")",1)[-1].strip() for x in rows[0]]; return [dict(zip(h,r)) for r in rows[1:]]
def box(x):
    x=(x or "").strip()
    if "-" not in x: return None
    a,b=x.split("-")[:2]; return (int(a),int(b)) if a.isdigit() and b.isdigit() else None

def span_idx(bx,atg,st,L):
    """genome [bL,bR] -> (i0,i1) slice in upstream string, or None if fully OOB."""
    bL,bR=bx
    if st=="+": d1,d2=atg-bR,atg-bL
    else:       d1,d2=bL-atg,bR-atg
    dmin,dmax=min(d1,d2),max(d1,d2)
    i0,i1=L-dmax, L-dmin+1
    i0,i1=max(0,i0),min(L,i1)
    return (i0,i1) if i1>i0 else None
def tss_idx(t,atg,st,L):
    d=(atg-t) if st=="+" else (t-atg)
    return (L-d) if 1<=d<=L else None

# ---- promoters by gene ----
proms=defaultdict(list)
for r in read_tsv(f"{RDB}/PromoterSet.tsv"):
    lt=lt_by_name.get((r.get("firstGeneName") or "").strip())
    if lt not in up: continue
    t=r.get("posTSS","").strip()
    proms[lt].append({"name":r.get("name","").strip(),"sigma":r.get("sigmaFactor","").strip(),
        "tss":int(t) if t.lstrip("-").isdigit() else None,
        "m10":box(r.get("boxMinus10pos")),"m35":box(r.get("boxMinus35pos")),
        "conf":r.get("confidenceLevel","").strip()})
# ---- TFBS by gene ----
tfbs=defaultdict(list)
for r in read_tsv(f"{RDB}/RISet.tsv"):
    lt=lt_by_name.get((r.get("firstGene") or "").strip())
    if lt not in up: continue
    l,rr=r.get("tfrsLeft","").strip(),r.get("tfrsRight","").strip()
    if l.isdigit() and rr.isdigit():
        tfbs[lt].append({"reg":r.get("regulatorName","").strip(),"box":(int(l),int(rr)),
                         "fn":r.get("riFunction","").strip()})

# ---- SD finder (consensus AGGAGG / anti-SD complementarity proxy) ----
SD_CONS="AGGAGG"
def sd_find(useq):
    """search positions with SD 3'-end 4..16 nt upstream of ATG; return (i0,i1,score,spacing)."""
    L=len(useq); best=None
    for i in range(max(0,L-20), L-3):          # SD start index
        j=i+len(SD_CONS)
        if j>L: break
        hexa=useq[i:j]
        score=sum(a==b for a,b in zip(hexa,SD_CONS))
        spacing=L-j                              # nt from SD 3'-end to ATG
        if 3<=spacing<=12:                       # plausible SD->ATG spacing
            if best is None or score>best[2] or (score==best[2] and abs(spacing-7)<abs(best[3]-7)):
                best=(i,j,score,spacing)
    return best

# ---- build ----
FULL_MIN_SD=4    # SD called present if >=4/6 consensus match
rows=[]; n_full=0
for gi,(lt,useq) in enumerate(sorted(up.items())):
    L=len(useq); g=gene.get(lt)
    if not g: continue
    nm,atg,st=g
    keep=[False]*L
    ann={"m35":[], "m10":[], "tss":[], "tfbs":[]}
    has35=has10=hastss=False
    for p in proms.get(lt,[]):
        for lab,bx in (("m35",p["m35"]),("m10",p["m10"])):
            if bx:
                sp=span_idx(bx,atg,st,L)
                if sp:
                    for k in range(*sp): keep[k]=True
                    ann[lab].append(list(sp))
                    if lab=="m35": has35=True
                    else: has10=True
        if p["tss"] is not None:
            ti=tss_idx(p["tss"],atg,st,L)
            if ti is not None and 0<=ti<L:
                keep[ti]=True; ann["tss"].append(ti); hastss=True
    for tf in tfbs.get(lt,[]):
        sp=span_idx(tf["box"],atg,st,L)
        if sp:
            for k in range(*sp): keep[k]=True
            ann["tfbs"].append({"reg":tf["reg"],"span":list(sp),"fn":tf["fn"]})
    # KEEP the RBS region = last 20 bp (contains SD + spacing to ATG); target for SD task
    for k in range(max(0,L-20),L): keep[k]=True
    sd=sd_find(useq); sd_present=bool(sd and sd[2]>=FULL_MIN_SD)   # metadata only
    if sd:
        i0,i1,sc,spac=sd
        ann["sd"]={"span":[i0,i1],"score":sc,"spacing":spac,"present":sd_present}
    else:
        ann["sd"]={"span":None,"score":0,"spacing":None,"present":False}

    full_prom = has35 and has10 and hastss
    if full_prom: n_full+=1
    tier = "full_promoter" if full_prom else "sd_only"

    # clean sequence: keep motif/RBS bases at fixed positions; RESHUFFLE the filler
    # (permutation preserves EXACT base composition -> strongest AT-content control)
    rng=random.Random(gi)
    fpos=[k for k in range(L) if not keep[k]]
    fb=[useq[k] for k in fpos]; rng.shuffle(fb)
    cl=list(useq)
    for k,b in zip(fpos,fb): cl[k]=b
    clean="".join(cl)

    rows.append({
        "locus_tag":lt,"gene":nm,"strand":st,"L":L,"tier":tier,
        "seq_leaked":useq,"seq_clean":clean,
        "keep_frac":round(sum(keep)/L,3),
        "full_promoter":int(full_prom),
        "m35":json.dumps(ann["m35"]),"m10":json.dumps(ann["m10"]),"tss":json.dumps(ann["tss"]),
        "sd_span":json.dumps(ann["sd"]["span"]),"sd_score":ann["sd"]["score"],
        "sd_spacing":ann["sd"]["spacing"],"sd_present":int(ann["sd"]["present"]),
        "n_tfbs":len(ann["tfbs"]),"tfbs":json.dumps(ann["tfbs"]),
        "promoters":";".join(f'{p["name"]}({p["sigma"]},{p["conf"]})' for p in proms.get(lt,[])),
    })

# ---- write ----
cols=["locus_tag","gene","strand","L","tier","full_promoter","keep_frac",
      "m35","m10","tss","sd_span","sd_score","sd_spacing","sd_present",
      "n_tfbs","tfbs","promoters","seq_leaked","seq_clean"]
outp=f"{OUT}/ecoli_upstream_master.csv"
with open(outp,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
    for r in rows: w.writerow(r)

# ---- summary ----
N=len(rows)
sd_present=sum(r["sd_present"] for r in rows)
tfbs_any=sum(1 for r in rows if r["n_tfbs"]>0)
kf=[r["keep_frac"] for r in rows if r["full_promoter"]]
kf_all=[r["keep_frac"] for r in rows]
def m(v): return sum(v)/len(v) if v else 0
print(f"genes written              : {N}")
print(f"  tier=full_promoter       : {n_full}")
print(f"  tier=sd_only             : {N-n_full}")
print(f"SD present (score>={FULL_MIN_SD}/6)     : {sd_present} ({100*sd_present/N:.1f}%)")
print(f"genes with >=1 TFBS        : {tfbs_any} ({100*tfbs_any/N:.1f}%)")
print(f"keep_frac mean (full-prom) : {100*m(kf):.1f}%   (all genes {100*m(kf_all):.1f}%)")
print(f"[written] {outp}")
