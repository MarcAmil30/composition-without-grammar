#!/usr/bin/env python3
"""Measure the conserved promoter grammar on the 711 full-promoter set:
 - -35 -> -10 spacer, -10 -> TSS, -35 -> TSS distances (per promoter, genome coords)
 - TSS -> ATG (5'UTR length)
 - nucleotide pattern around the TSS (+1 initiator)
So we know what 'correct generation' means beyond just -35->-10."""
# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
WORK_ROOT = _os.environ.get("WORK_ROOT", _os.path.expanduser("~"))
# -------------------------------------------------------------------

import csv
from collections import defaultdict, Counter

BASE=f"{WORK_ROOT}"
RAW =BASE+"/Proking_repo/data/raw_sequences_data_small_scale/Escherichia_coli_operon_annotated.csv"
GFF =BASE+"/Proking_repo/code/annotation/ecoli_annotation/GCF_000005845.2_ASM584v2_genomic.gff"
PROM=BASE+"/ICBINB_BIO_NeurIPS2026/data/regulondb/PromoterSet.tsv"
UP  ={"singleton_or_non_operon","first_operon_gene"}

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
def box(x):
    x=(x or "").strip()
    if "-" not in x: return None
    a,b=x.split("-")[:2]; return (int(a),int(b)) if a.isdigit() and b.isdigit() else None
rows=[r for r in csv.reader(open(PROM),delimiter="\t") if r and not r[0].startswith("#")]
hdr=[h.split(")",1)[-1].strip() for h in rows[0]]; D=[dict(zip(hdr,r)) for r in rows[1:]]

def upstream_dist(pos, atg, st):        # distance upstream of ATG (1..L adjacent)
    return (atg-pos) if st=="+" else (pos-atg)
def rel_to_tss(pos, tss, st):           # + = upstream of TSS
    return (tss-pos) if st=="+" else (pos-tss)

spacer=[]; d10=[]; d35=[]; utr=[]; tss_ctx=[]
n=0
for r in D:
    m35=box(r.get("boxMinus35pos")); m10=box(r.get("boxMinus10pos")); t=r.get("posTSS","").strip()
    if not(m35 and m10 and t.lstrip("-").isdigit()): continue
    lt=lt_by_name.get((r.get("firstGeneName") or "").strip())
    if lt not in up: continue
    nm,atg,st=gene[lt]; L=len(up[lt]); tss=int(t)
    # must be in-window (TSS + both boxes)
    if not(1<=upstream_dist(tss,atg,st)<=L): continue
    n+=1
    # strand-aware "upstream ends": use the edge of each box nearest the TSS
    if st=="+":
        m35_in, m35_out = max(m35), min(m35)   # in=closer to TSS(3'side)
        m10_in, m10_out = max(m10), min(m10)
    else:
        m35_in, m35_out = min(m35), max(m35)
        m10_in, m10_out = min(m10), max(m10)
    # spacer = gap between -35 (TSS-proximal edge) and -10 (distal edge)
    sp = abs(rel_to_tss(m10_out,tss,st) - rel_to_tss(m35_in,tss,st)) - 1
    spacer.append(sp)
    d10.append(rel_to_tss(m10_in,tss,st))     # -10 proximal edge -> TSS
    d35.append(rel_to_tss(m35_in,tss,st))
    utr.append(upstream_dist(tss,atg,st))     # TSS -> ATG (5'UTR len)
    # TSS base context from the extracted upstream: idx of TSS
    ti=L-upstream_dist(tss,atg,st)
    frag=up[lt][max(0,ti-2):ti+3]              # positions -2 -1 +1 +2 +3 (ti = +1)
    if len(frag)==5: tss_ctx.append(frag)

def summ(v,name,lo,hi):
    v=[x for x in v if lo<=x<=hi]; v.sort(); n=len(v)
    mode=Counter(v).most_common(1)[0]
    mean=sum(v)/n
    print(f"{name:24s} n={n:4d}  mean={mean:5.1f}  median={v[n//2]:3d}  mode={mode[0]} ({100*mode[1]/n:.0f}%)")
    # compact histogram
    c=Counter(v); span=range(min(v),max(v)+1)
    print("   "+"  ".join(f"{k}:{100*c.get(k,0)/n:.0f}%" for k in span if c.get(k,0)/n>=0.03))

print(f"promoters analysed (full, in-window): {n}\n")
print("=== conserved distances (bp) ===")
summ(spacer,"-35 -> -10 spacer",10,25)
summ(d10,   "-10(edge) -> TSS",0,15)
summ(d35,   "-35(edge) -> TSS",20,50)
summ(utr,   "TSS -> ATG (5'UTR)",0,200)

print("\n=== TSS initiator pattern (position -2 -1 +1 +2 +3 ; +1 = TSS) ===")
import collections
for i,lab in enumerate(["-2","-1","+1","+2","+3"]):
    c=Counter(f[i] for f in tss_ctx); tot=sum(c.values())
    order=sorted(c.items(),key=lambda x:-x[1])
    print(f"  {lab}: "+"  ".join(f"{b}:{100*n/tot:.0f}%" for b,n in order))
print(f"\n(+1 purine A/G fraction: "
      f"{100*sum(1 for f in tss_ctx if f[2] in 'AG')/len(tss_ctx):.0f}% ; "
      f"most common -1/+1 dinuc: {Counter(f[1]+f[2] for f in tss_ctx).most_common(3)})")
