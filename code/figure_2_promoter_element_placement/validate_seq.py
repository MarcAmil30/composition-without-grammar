#!/usr/bin/env python3
"""Sequence-level validation: does the RegulonDB -10/-35 box sequence actually
appear in the extracted upstream_200bp at the coordinate-predicted position?
Diagnose offsets for any mismatches (systematic shift = fixable)."""
# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
WORK_ROOT = _os.environ.get("WORK_ROOT", _os.path.expanduser("~"))
# -------------------------------------------------------------------

import csv
from collections import Counter, defaultdict

BASE=f"{WORK_ROOT}"
RAW=BASE+"/Proking_repo/data/raw_sequences_data_small_scale/Escherichia_coli_operon_annotated.csv"
GFF=BASE+"/Proking_repo/code/annotation/ecoli_annotation/GCF_000005845.2_ASM584v2_genomic.gff"
PROM=BASE+"/ICBINB_BIO_NeurIPS2026/data/regulondb/PromoterSet.tsv"
UP={"singleton_or_non_operon","first_operon_gene"}

comp=lambda s:s.translate(str.maketrans("ACGTacgt","TGCAtgca"))
rc=lambda s:comp(s)[::-1]

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
    a,b=x.split("-")[:2]
    return (int(a),int(b)) if a.isdigit() and b.isdigit() else None
rows=[r for r in csv.reader(open(PROM),delimiter="\t") if r and not r[0].startswith("#")]
hdr=[h.split(")",1)[-1].strip() for h in rows[0]]; D=[dict(zip(hdr,r)) for r in rows[1:]]

def predicted_idx(bx,atg,st,L):
    bL,bR=bx
    if st=="+": d1,d2=atg-bR,atg-bL
    else:       d1,d2=bL-atg,bR-atg
    dmin,dmax=min(d1,d2),max(d1,d2)
    return L-dmax, L-dmin+1     # slice [i0:i1] in the 5'->3' upstream string

def check(useq,bx,bseq,atg,st):
    """return ('exact'|'offset:N'|'elsewhere'|'notfound'|'oob', found_offset)"""
    L=len(useq); bseq=bseq.upper()
    if not bseq: return ("noseq",None)
    i0,i1=predicted_idx(bx,atg,st,L)
    if 0<=i0<i1<=L:
        frag=useq[i0:i1]
        if frag==bseq or rc(frag)==bseq: return ("exact",0)
    # search whole upstream (both orientations) for the box seq
    for orient in (bseq, rc(bseq)):
        j=useq.find(orient)
        if j!=-1:
            off = j-i0
            return (("offset" if 0<=i0<=L else "elsewhere"), off)
    if not(0<=i0<i1<=L): return ("oob",None)
    return ("notfound",None)

res10=Counter(); res35=Counter(); offs=Counter(); seen=set()
n=0
for r in D:
    t=r.get("posTSS","").strip(); m10=box(r.get("boxMinus10pos")); m35=box(r.get("boxMinus35pos"))
    if not(m10 and m35 and t.lstrip("-").isdigit()): continue
    nm=(r.get("firstGeneName") or "").strip(); lt=lt_by_name.get(nm)
    if lt not in up: continue
    nm2,atg,st=gene[lt]; useq=up[lt]
    # only count in-window full promoters
    i0,i1=predicted_idx(m10,atg,st,len(useq))
    if not(0<=i0<i1<=len(useq)): continue
    n+=1
    s10,o10=check(useq,m10,r.get("boxMinus10seq",""),atg,st)
    s35,o35=check(useq,m35,r.get("boxMinus35seq",""),atg,st)
    res10[s10]+=1; res35[s35]+=1
    if o10 is not None: offs[o10]+=1

print(f"full in-window promoters checked: {n}\n")
def show(name,c):
    print(name)
    for k,v in c.most_common(): print(f"   {k:10s}: {v} ({100*v/n:.1f}%)")
show("-10 box sequence vs predicted position:",res10)
show("-35 box sequence vs predicted position:",res35)
print("\noffset distribution (found_pos - predicted_pos), -10 box:")
for k,v in sorted(offs.items(), key=lambda x:-x[1])[:12]:
    print(f"   offset {k:+d}: {v}")
