# %% [markdown]
# # Task-1 dataset analysis (E. coli up_only, 711 full-promoter set)
# Basic sanity + diversity analysis of the upstream-generation benchmark dataset.

# %%
# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
WORK_ROOT = _os.environ.get("WORK_ROOT", _os.path.expanduser("~"))
# -------------------------------------------------------------------

import csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

BASE = f"{WORK_ROOT}"
BM   = f"{BASE}/ICBINB_BIO_NeurIPS2026/data/benchmark/upstream_only"
RDB  = f"{BASE}/ICBINB_BIO_NeurIPS2026/data/regulondb"
RAW  = f"{BASE}/Proking_repo/data/raw_sequences_data_small_scale/Escherichia_coli_operon_annotated.csv"
GFF  = f"{BASE}/Proking_repo/code/annotation/ecoli_annotation/GCF_000005845.2_ASM584v2_genomic.gff"
OUT  = f"{BM}/plots"
import os; os.makedirs(OUT, exist_ok=True)
UP={"singleton_or_non_operon","first_operon_gene"}

full711={r["locus_tag"]:r for r in csv.DictReader(open(f"{BM}/full_promoter_711.csv"))}
prompts=list(csv.DictReader(open(f"{BM}/prompts_leaked.csv")))
print("711 genes:",len(full711),"| prompts:",len(prompts))

# mapping helpers (for genome-coordinate distances)
lt_by_name={}; gene={}
for line in open(GFF):
    if line.startswith("#"): continue
    c=line.split("\t")
    if len(c)<9 or c[2]!="gene": continue
    s,e,st,attr=int(c[3]),int(c[4]),c[6],c[8]
    d=dict(kv.split("=",1) for kv in attr.strip().split(";") if "=" in kv)
    lt=d.get("locus_tag"); nm=d.get("Name") or d.get("gene")
    if lt:
        gene[lt]=(s if st=="+" else e, st)
        if nm: lt_by_name[nm]=lt
def boxp(x):
    x=(x or "").strip()
    if "-" not in x: return None
    a,b=x.split("-")[:2]; return (int(a),int(b)) if a.isdigit() and b.isdigit() else None
rows=[r for r in csv.reader(open(f"{RDB}/PromoterSet.tsv"),delimiter="\t") if r and not r[0].startswith("#")]
hdr=[h.split(")",1)[-1].strip() for h in rows[0]]; Prom=[dict(zip(hdr,r)) for r in rows[1:]]

# %% [markdown]
# ## 1. -35 and -10 box lengths (should both be ~6 bp)
# %%
len35=[]; len10=[]
for r in Prom:
    lt=lt_by_name.get((r.get("firstGeneName") or "").strip())
    if lt not in full711: continue
    b35=boxp(r.get("boxMinus35pos")); b10=boxp(r.get("boxMinus10pos"))
    if b35: len35.append(b35[1]-b35[0]+1)
    if b10: len10.append(b10[1]-b10[0]+1)
fig,ax=plt.subplots(1,2,figsize=(9,3.5))
for a,data,lab,col in [(ax[0],len35,"-35 box","#4C72B0"),(ax[1],len10,"-10 box","#C44E52")]:
    vals=Counter(data)
    a.bar(vals.keys(),vals.values(),color=col,edgecolor="black")
    a.set_title(f"{lab} length  (mode={Counter(data).most_common(1)[0][0]} bp, n={len(data)})")
    a.set_xlabel("box length (bp)"); a.set_ylabel("count")
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_box_lengths.png",dpi=150); plt.show()
print("-35 lengths:",dict(Counter(len35))," | -10 lengths:",dict(Counter(len10)))

# %% [markdown]
# ## 2. Distance from box (TSS-proximal edge) to the TSS
# -10 -> TSS should center ~7 bp ; -35 -> TSS ~30 bp
# %%
d10=[]; d35=[]
for r in Prom:
    lt=lt_by_name.get((r.get("firstGeneName") or "").strip())
    if lt not in full711: continue
    b35=boxp(r.get("boxMinus35pos")); b10=boxp(r.get("boxMinus10pos")); t=r.get("posTSS","").strip()
    if not(b35 and b10 and t.lstrip("-").isdigit()): continue
    atg,st=gene[lt]; tss=int(t)
    def edge_to_tss(bx):
        prox = max(bx) if st=="+" else min(bx)     # TSS-proximal edge
        return (tss-prox) if st=="+" else (prox-tss)
    x10=edge_to_tss(b10); x35=edge_to_tss(b35)
    if 0<=x10<=20: d10.append(x10)
    if 15<=x35<=55: d35.append(x35)
fig,ax=plt.subplots(1,2,figsize=(9,3.5))
ax[0].hist(d10,bins=range(0,21),color="#C44E52",edgecolor="black",align="left")
ax[0].axvline(np.median(d10),ls="--",c="k"); ax[0].set_title(f"-10 end -> TSS  (median {int(np.median(d10))} bp)")
ax[0].set_xlabel("distance (bp)"); ax[0].set_ylabel("count")
ax[1].hist(d35,bins=range(15,56),color="#4C72B0",edgecolor="black",align="left")
ax[1].axvline(np.median(d35),ls="--",c="k"); ax[1].set_title(f"-35 end -> TSS  (median {int(np.median(d35))} bp)")
ax[1].set_xlabel("distance (bp)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_dist_to_tss.png",dpi=150); plt.show()
print(f"-10->TSS median={np.median(d10):.0f} mean={np.mean(d10):.1f} | -35->TSS median={np.median(d35):.0f} mean={np.mean(d35):.1f}")

# %% [markdown]
# ## 3. How diverse are the sequences? (are prompts all the same, or vary per gene?)
# Pairwise 4-mer cosine distance within each set. 0 = identical, 1 = maximally different.
# %%
BASES="ACGT"; K=4
kmers=[a+b+c+d for a in BASES for b in BASES for c in BASES for d in BASES]
kidx={k:i for i,k in enumerate(kmers)}
def kmer_matrix(seqs):
    M=np.zeros((len(seqs),len(kmers)))
    for i,s in enumerate(seqs):
        s=s.upper()
        for j in range(len(s)-K+1):
            m=s[j:j+K]
            if m in kidx: M[i,kidx[m]]+=1
    M=M/ (M.sum(1,keepdims=True)+1e-9)          # frequency
    M=M/ (np.linalg.norm(M,axis=1,keepdims=True)+1e-9)  # L2 normalise
    return M
def pairwise_dist(seqs):
    M=kmer_matrix(seqs); S=M@M.T; D=1-S
    iu=np.triu_indices(len(seqs),k=1)
    return D[iu]

sets={
 "full 200bp\n(seq_leaked)":      [r["seq_leaked"] for r in full711.values()],
 "generate_promoter\nprompt":     [r["prompt_promoter"] for r in prompts],
 "generate_promoter\ntarget":     [r["target_promoter"] for r in prompts],
 "generate_sd\nprompt":           [r["prompt_sd"] for r in prompts],
}
dists={k:pairwise_dist(v) for k,v in sets.items()}
fig,ax=plt.subplots(figsize=(8,4.5))
parts=ax.violinplot([dists[k] for k in sets],showmedians=True)
ax.set_xticks(range(1,len(sets)+1)); ax.set_xticklabels(list(sets.keys()))
ax.set_ylabel("pairwise 4-mer cosine distance\n(higher = more diverse)")
ax.set_title("Sequence diversity within each set (n pairs per set in legend)")
ax.set_ylim(0,1)
for i,k in enumerate(sets,1):
    ax.text(i, 0.03, f"mean={dists[k].mean():.2f}", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_diversity.png",dpi=150); plt.show()

# %%
print("\n=== SUMMARY ===")
print(f"-35 box length mode: {Counter(len35).most_common(1)[0]}  | -10 box length mode: {Counter(len10).most_common(1)[0]}")
print(f"-10 end -> TSS: median {np.median(d10):.0f} bp | -35 end -> TSS: median {np.median(d35):.0f} bp")
print("Diversity (mean pairwise 4-mer distance):")
for k,v in dists.items():
    print(f"  {k.splitlines()[0]:14s}: mean={v.mean():.3f}  median={np.median(v):.3f}  n_seqs={len(sets[k])}")
