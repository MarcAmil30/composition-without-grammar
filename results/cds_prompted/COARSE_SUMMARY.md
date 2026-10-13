# What PROK actually learned — the coarse-composition summary

One claim, assembled from five measurements: **PROK reproduces low-order sequence
composition (up to ~3rd–4th order) and one fixed motif, and nothing else.** Every
number below is on the same 6-mer readout over 200 bp upstream regions
(*E. coli* n = 2,719; *B. subtilis* n = 2,388).

---

## 1. The two species differ mainly in composition

| | *E. coli* | *B. subtilis* |
|---|---|---|
| upstream GC | 44.1% | 37.5% |

Correlation between the two species' **natural** profiles:

| | k=1 | k=2 | k=4 | k=6 |
|---|---|---|---|---|
| *r*(E. coli, B. subtilis) | 0.985 | 0.924 | 0.849 | 0.773 |

The species are nearly identical at low order and separate only as *k* grows —
so "species-specific" is a composition statement, not a grammar statement.

## 2. The wrong-token control *is* the other species

| | *E. coli* | *B. subtilis* |
|---|---|---|
| PROK, wrong species token | 0.792 | 0.772 |
| natural other species (k=6, above) | 0.773 | 0.773 |

Generating with the wrong token scores exactly what the other organism's **real
DNA** scores. The control therefore measures "did it pick the right composition",
and establishes no floor for regulatory structure.

## 3. The shuffle ladder: how much of a 6-mer profile is reachable without grammar

A *k*-preserving shuffle keeps every *k*-mer count exactly and destroys everything
longer — no promoter, no positioned Shine–Dalgarno, no gene identity survives.
20 replicates per rung, 95% percentile interval.

| preserve | *E. coli* | *B. subtilis* |
|---|---|---|
| 1-mer | 0.584 [0.569–0.596] | 0.756 [0.747–0.763] |
| 2-mer | 0.802 [0.797–0.806] | 0.884 [0.880–0.887] |
| 3-mer | 0.905 [0.902–0.908] | 0.962 [0.960–0.964] |
| **PROK** | **0.953** | **0.978** |
| 4-mer | 0.966 [0.965–0.968] | 0.982 [0.982–0.983] |
| 5-mer | 0.983 [0.982–0.984] | 0.988 [0.987–0.989] |

**PROK is bracketed between the 3-mer and 4-mer rungs, and sits below the 4-mer
rung in both species** — the CIs are ~0.001–0.003 wide, so the ordering is not in
doubt. A shuffle with provably zero regulatory structure scores higher than the
model.

The ladder stops at k=5 because a 6-mer-preserving shuffle reproduces a 6-mer
profile by definition (*r* = 1) and carries no information.

## 4. The generator does not commit to a gene

Divergence = normalised Hamming over aligned positions. The natural yardstick is the
**composition-matched ceiling** $1-\sum_i p_i^2$: what two unrelated sequences of that
base composition would reach.

**Over the whole 200 bp the measure is saturated and says nothing:**

| % of ceiling | *E. coli* | *B. subtilis* |
|---|---|---|
| natural, between genes | 99.5 | 98.8 |
| generated, between genes | 99.8 | 99.4 |
| generated, within gene | 99.9 | 99.5 |

Two *real* promoters are, at whole-window Hamming, indistinguishable from random
sequence of the same composition — so the metric has no headroom for anything.
Split by region, the reason is clear:

| % of ceiling | region | *E. coli* | *B. subtilis* |
|---|---|---|---|
| natural, between genes | body (−200…−21) | 100.0 | 99.8 |
| natural, between genes | **SD window (−20…−1)** | **96.2** | **93.9** |
| generated, between genes | SD window | 98.0 | 97.0 |
| generated, within gene | SD window | 97.9 | 97.2 |

Natural genes converge **only** where they share a motif. In that window:

- the model reproduces roughly **half** the natural convergence (2.0 vs 3.8 points
  below ceiling in *E. coli*; 3.0 vs 6.1 in *B. subtilis*) — its Shine–Dalgarnos are
  more variable than real ones;
- **within-gene equals between-gene** (97.9 vs 98.0; 97.2 vs 97.0), so asking twice
  for the same gene is no more consistent than asking for two different genes.

> Caveat: between-gene comparisons must use **different decode indices**.
> `models.py` sets `manual_seed(seed*100003 + sample_idx)`, so decode *i* shares an
> RNG stream across all genes; comparing matched indices gives 0.09 instead of 0.74
> and is an artefact of the seeding, not a property of the model.

## 5. The one thing above composition

Per-position predictive entropy is 1.895 of a maximum 2.0 bits — near-uniform
almost everywhere — but falls to **1.36 bits at position −9**, where P(G) = 0.54.
That is the Shine–Dalgarno, and it is the model's only positional structure. It is
gene-invariant: the same motif is stamped at the same offset regardless of the CDS.

---

## What this adds up to

| the model has | evidence |
|---|---|
| species-typical base and low-order *k*-mer composition | §1, §2 |
| local structure to ~3rd–4th order | §3 |
| one fixed motif (SD) at a fixed offset | §5 |

| the model lacks | evidence |
|---|---|
| gene-specific conditioning | §4, and Δp = 0.003 under CDS shuffling |
| positional promoter grammar | the companion upstream-only track (chance placement) |

A 6-mer correlation of 0.95 against natural sequence is therefore **not** evidence
that a generator has learned regulatory DNA. Reported without the shuffle ladder
it overstates the result by the entire distance between 0 and 0.97.

---

*Sources: `fig_stats.py`, `kladder.py`, `xspec.py`, `diversity_and_reps.py`,
`divfix.py` on Isambard under `data/benchmark/cds_prompted/code/`; all values
cached in `cds_fig_stats.npz`.*
