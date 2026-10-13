# Task 4 — CDS ↔ upstream retrieval

**The question.** If regulation were one-size-fits-all, no model could pick a CDS's own
upstream out of 100 upstreams from the same genome. So: can it?

This is the direct test of the paper's hypothesis — that a CDS needs a *tailored* upstream.
Tasks 1–3 ask whether a model **generates** a plausible upstream. This one asks whether it
knows **which** upstream goes with **which** gene.

---

## 1. Dataset

`ecoli_retrieval_eval500.csv` — **500 E. coli K-12 genes** (`GCF_000005845.2`).

**E. coli only.** C3P was never trained on E. coli, so the one model with a strong positive
result is being tested held-out. That makes it a credible ceiling to compare against.

Because L is fixed at 200 bp, **every gene must have a full 200 bp of real upstream**. Only
798 of the 1,453 clear that — the rest run into the neighbouring gene first. Eligible pool
= those 798. One seeded shuffle (`seed=0`), then prefixes:

```
first 150  = pilot set   (smoke test only, never reported)
first 500  = eval set    (both scorers use this)
```

The pilot is a **subset** of the eval set, so a pilot run's scores are reused by the full
run rather than thrown away.

```
E. coli CDS                        4,290
  singletons (50 bp operon window) 1,896
  + up_len >= 100                  1,453   population
  + up_len >= 200                    798   eligible pool
  → seeded draw                      500   eval set
```

Singletons only, so each gene owns its upstream. Per gene the file carries `upstream_seq`
(5'→3', ends at the start codon — the window is `upstream_seq[-200:]`), `cds_seq`, and
`protein_seq` (C3P needs protein, the DNA models need CDS), plus flags joining to the other
tasks. Median CDS 870 bp, max 7,020 → longest scored sequence 7,224 tokens, inside Evo's
8k context.

**482 / 500 (96%) are also in the CDS-prompted set** — same genome, same b-number
locus_tags. That's the point: retrieval and generation results join **per gene**.

⚠️ **Caveat.** Requiring a clean 200 bp selects genes in sparser genomic neighbourhoods. If
isolated genes have more distinctive promoters, absolute retrieval is measured on a mildly
favourable subset. It is the *same* subset for every model, so model comparisons are
unaffected — but the absolute numbers carry this qualifier.

The alternative was to take 200 bp regardless and let the window overlap the neighbour (the
convention `cds_prompted` uses, with a `contam_bp` column). That would have made the input
byte-identical to task 3 at the cost of feeding models sequence that is physically inside
another gene. Clean windows were chosen instead.

---

## 2. Methodology

One protocol, two scorers, **the same 500 genes for both**.

### The setup (identical for both scorers)

For each gene *i*, build a candidate pool of 100 upstreams:

- 1 true — gene *i*'s own upstream
- 99 decoys — upstreams of other genes in the 500

Rank all 100. Record where the true one landed. Repeat **20 times** per gene with a fresh
decoy draw (`seed=0`) → **10,000 ranking trials** per model.

Decoys come from the same genome, so GC content is controlled by construction — a model
can't win by matching base composition.

The 20 repeats are **free**: score the full 500×500 matrix once, and each repeat is just
re-sampling 99 columns from the cached matrix. They only shrink noise from the decoy draw.

**The RNG is shared across every model and both scorers** — gene *i*, repeat *r* faces the
identical 99 decoys everywhere. Every comparison is therefore paired.

### Scorer A — Embedding (cosine)

Embed each sequence **once**, then rank by cosine similarity.

- Embed the CDS → one vector. Embed each of the 100 candidate upstreams → 100 vectors.
- Mean-pool the final hidden state over positions, L2-normalise.
- **Centre before comparing**: subtract the per-candidate-set mean. Autoregressive DNA LMs
  are badly anisotropic — Evo-1 gave a true-pair cosine of 0.999, every embedding crammed
  into a narrow cone where cosine can't discriminate. Report raw *and* centred.
- **C3P is different**: it has a trained shared space, so cosine is used directly and the
  query is the **protein**, not the CDS.

Cheap — linear in the number of sequences.

### Scorer B — Scoring (likelihood)

Feed the model the **whole thing**: upstream + CDS concatenated, 5'→3'. Rank candidates by
how likely the model finds that concatenation.

Two numbers from one forward pass:

- `score_full` — mean log-prob over the whole sequence
- **`score_cond` — mean log-prob over the *second* segment only = the conditional**

`score_cond` is the principled one: it isolates the dependency between the two regions
instead of letting a generically-likely sequence dominate. On Evo-1 it was 3× better
(15.6% vs 5.3%). Which segment is "second" depends on the model — see *Model-specific
formatting* below.

Higher likelihood = better, but both directions are recorded so the data settles it.

Expensive — **pairwise**, so cost is n². The 500×500 matrix is ~7.4 h on one GH200 per
model. Matrices are cached as `.npy`, so any re-ranking afterwards is instant.

### Upstream length — fixed at 200 bp

**L = 200 bp everywhere.** Every model, both scorers, every run. No sweep.

The 200 bp closest to the start codon (`upstream_seq[-200:]`). This is the same window the
CDS-prompted task generates, so the two tasks are measured on identical sequence.

Nothing varies between runs except the model and the scorer. A table where one row used 150
genes at 300 bp and another used 500 at 200 bp isn't a comparison. The only things that
change across rows are the two things being compared.

(The pilot 150 is a smoke test to confirm a model runs end-to-end — its numbers are never
reported.)

### Model-specific formatting

Each model is fed sequence in the arrangement it was trained on. Feeding a model an
arrangement it has never seen produces finite, plausible, meaningless numbers — that is
exactly how the first C3P run produced a publishable null that was wrong.

**PROK** — trained on `<species>#<CDS><species>$<5'UTR><species>*<3'UTR>`, where the species
token is `+` for E. coli and `@` for B. subtilis, repeated before each segment tag.

- *Scoring*: feed `+#<CDS>+$<upstream>` and score the **upstream tokens** → **log P(upstream | CDS)**.
  CDS-first is PROK's native order, so this needs no concatenation trick and asks the
  question directly: given this gene, how likely is this regulatory region?
- *Embedding*: embed the **formatted** strings including the prompt/special tokens —
  `+#<CDS>+$` for the query and `+$<upstream>` for each candidate — not bare DNA. The
  special tokens are part of what PROK was trained to see; stripping them is off-distribution.

**Evo 1 / Evo 2 / HyenaDNA** — no segment tokens, and their training order is genomic
(upstream precedes CDS). So they get `<upstream><CDS>` and are scored on the **CDS tokens**
→ **log P(CDS | upstream)**. Embeddings are of bare DNA.

**C3P** — protein query against upstream, in its trained shared space. Embedding only.

⚠️ **Note the asymmetry.** PROK is scored as P(upstream | CDS); Evo/HyenaDNA as
P(CDS | upstream). Both are valid rankings over the same 100 candidates, and each is that
model's native direction — but they are not the same quantity, and the table should say so
rather than imply the numbers are interchangeable.

---

## 3. Metrics

Chance: top-1 = **1%**, top-5 = **5%**, median rank = **50**.

| metric | what it tells you |
|---|---|
| **top-1 (%)** | headline — also quote as fold over chance |
| top-5 (%) | partial credit |
| median rank | robust to the tail |
| MRR | one number for the leaderboard |
| 95% CI | bootstrap **over genes (n=500)** — the 10,000 trials are not independent, 20 share each gene |
| paired Wilcoxon vs C3P | per-gene ranks, shared decoys → a paired test is valid |

At n=500 the CI on a top-1 near 15% is about **±3.2%** — enough to separate models a few
points apart. (At n=150 it would be ±5.8%, which is why the eval set isn't the pilot.)

---

## 4. Controls

Run at L=200 per model. These separate "reads regulatory grammar" from "matches composition".

| control | what it does | C3P result |
|---|---|---|
| shuffled upstream | mononucleotide shuffle — keeps GC and 1-mer composition exactly, destroys motif order | 1.5% |
| 3-mer matcher | rank by k-mer composition similarity, no model | 2.3% |
| GC matcher | rank by GC distance alone | 1.4% |

Same 500 genes, same L=200, same decoys as the main run.

All three collapse to chance for C3P ⇒ its signal is **motif structure**, not composition.
A model that survives shuffling is reading composition and the result doesn't count.

---

## 5. Models

| model | scoring | embedding | input arrangement | note |
|---|:---:|:---:|---|---|
| **Evo 1** | ✅ | ✅ | `<up><CDS>` → P(CDS \| up) | prior run: 15.6% scoring / at chance embedding |
| **Evo 2** | ✅ | ✅ | `<up><CDS>` → P(CDS \| up) | larger, budget ~1.5–2× the runtime |
| **PROK** | ✅ | ✅ | `+#<CDS>+$<up>` → P(up \| CDS) | native — CDS-first is its training format |
| **C3P** | ❌ | ✅ | protein ↔ upstream | contrastive, no likelihood head; cosine *is* its score. Prior run: **86.1%** |
| **HyenaDNA** | ✅ | ✅ | `<up><CDS>` → P(CDS \| up) | human-trained → off-domain baseline |

Both scorers run on every model that supports them, even where one is expected to win. Evo
is a likelihood model, but its embedding null is a result worth reporting in its own right.

**Note on the prior Evo-1 / C3P numbers:** those were run on a different gene set (the old
727, with an `n=150` subsample for scoring). They are a consistency check, **not** rows in
this table — both models get re-run on the 500 for a like-for-like comparison.

---

## 6. Compute

Everything at L=200, n=500, E. coli.

| step | cost per model |
|---|---|
| embedding (500 CDS + 500 upstreams) | minutes |
| scoring (500×500 matrix) | **~7.4 h** on 1× GH200 |
| pilot smoke test (150×150) | ~40 min |

Four scoring models (Evo 1, Evo 2, PROK, HyenaDNA) ≈ **30 h GPU total**; budget 1.5–2× on
the Evo 2 row for its size. Score matrices checkpoint in blocks, so jobs survive the queue
and resume.

Fixing L at 200 removes the 4× multiplier the length sweep would have added.

---

## 7. Guards

A null result is only worth reporting if the pipeline is known to work. Each run checks:

1. **tokenizer** is one token per character, including the framing chars `+ # $`
2. **offsets** — tokenized length equals character length, per gene, so the segment
   boundary is where we think it is
3. **diagonal** — a few true-pair scores are recomputed independently and must match the
   matrix diagonal (catches index misalignment)
4. **embedding determinism** — the same sequence embedded twice gives cosine 1.0
5. **true-vs-decoy gap** — reported next to top-1 on every run

Guard 5 is the important one. A model can sit at chance top-1 and still show a clean
positive gap: that means the plumbing works and the effect is real but too weak to rank.
Chance top-1 **and** a zero gap is the signature of a wiring bug — which is exactly how the
first C3P run produced a confident, publishable, wrong null (CLS pooling instead of mean).

The ranking code itself is calibrated against noise: a random matrix returns
top-1 1.06% (chance 1%), median rank 50.2 (chance 50.5), and the bootstrap CI covers
chance in 39/40 draws. Planted signals of known size are recovered monotonically.
See `test_calib.py`.

---

## 8. Reading the result

- **High retrieval** → the upstream carries CDS-specific information the model represents.
- **At chance** → the model treats regulation as interchangeable between genes:
  one-size-fits-all.
- **Generates well (task 3) but retrieves at chance** → the paper's claim, demonstrated on
  the same genes: **species-typical, not CDS-specific.**

That last line is why the set was rebuilt around the CDS-prompted genes. The 96% overlap
means you can ask it gene by gene, not just as an average.
