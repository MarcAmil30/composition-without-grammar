# Upstream-only benchmark — results

**Complete.** 5 models × 2 tasks × 2 backgrounds × 681 genes × 21 decodes =
**284,382 generations**, all 20 shards finished, plus a teacher-forced pass
recovering the per-base probability behind every one of them, plus a third run
that generates the whole upstream from the −35 to the start codon.

| figure | what it shows |
|---|---|
| `fig1_placement` | position-error curves vs chance, both tasks, leaked vs clean |
| `fig2_signal` | signal above each model's own shuffle · leakage · start codon |
| `fig3_composition` | mean P(base) per position, models × tasks |
| `fig4_bases_promoter` / `fig4_bases_sd` | one panel per model, 4 ACGT lines |
| `fig5_to_atg_5prime` / `fig5_to_atg_3prime` | prompt at −35 → generate to the ATG |
| `fig6_context` | exact-hit rate vs upstream context (200 → 1000 bp + intergenic) |

Numbers: `summary.csv`, `to_atg_*_source.csv`; full tables on Isambard under
`benchmark/upstream_only/{scores,probs,to_atg}/`.

---

## The result in one line

**Every model places the −10 box at chance and shows no positional preference
there at all — and giving them 5× more upstream context does not change it.
PROK, alone, places the Shine–Dalgarno, and does it from 73 bp away. No model
gains anything from having seen the real genome.**

---

## Task 1 — place the −10 box

Prompt ends at the −35; 45 bp generated. Chance for an exact hit is 1/40 = **2.5%**.

| model | exact hits | median error | vs own shuffle | GC |
|---|---|---|---|---|
| Evo 1 (7B, prokaryotic) | 2.3% | 10 bp | **−0.35** | 0.40 |
| Evo 2 | 2.9% | 10 bp | **−0.15** | 0.42 |
| PROK (fine-tuned on these upstreams) | 2.2% | 10 bp | **−0.08** | 0.41 |
| HyenaDNA (human hg38 only) | 2.4% | 10 bp | **−0.22** | 0.39 |
| PromoGen2 | 2.4% | 10 bp | **−0.11** | 0.20 |
| *chance / real* | *2.5%* | — | *0* | *0.42* |

Every model scores **below a shuffle of its own output** — destroying the grammar
of a generation does not make its best −10 hit worse. And a human-genome model is
indistinguishable from one fine-tuned on exactly these upstreams.

**The probabilities make this much stronger than the argmax result.** Across the
whole 45 bp window, P(G) varies by at most **0.027** and entropy by at most
**0.10 bits** in every model. The distribution never sharpens anywhere, let alone
at the −10. This does not depend on the PWM, the scan, or the chance model: the
models simply have no positional preference to express.

**Spacer.** Real promoters put the −35→−10 spacer in 15–19 bp **60.5%** of the
time. The best model (Evo 2) manages **13.5%** against a chance of 12.5%. PROK is
worst at 8.9% — below chance for the 17 bp mode.

**Composition is the one thing four of five get right**: Evo 1, Evo 2, PROK and
HyenaDNA land at 39–42% GC against the real 42%. PromoGen2 alone is off at 20%
(P(AT) = 0.78 vs ~0.58 for the rest). So four of five produce compositionally
correct bacterial DNA containing no positioned motif — *likelihood ≈ composition*,
measured directly on generation.

---

## Task 2 — place the Shine–Dalgarno

Prompt is the upstream minus its last 20 bp; 40 bp generated. Chance = 1/35 = **2.9%**.

| model | exact hits | median error | within 2 bp | vs own shuffle | start codon |
|---|---|---|---|---|---|
| **PROK** | **11.9%** | **2 bp** | **51%** | **+2.13** | 38% |
| PromoGen2 | 2.4% | 9 bp | 12% | +1.45 | 24% |
| HyenaDNA | 3.0% | 9 bp | 14% | +0.41 | 52% |
| Evo 2 | 2.9% | 9 bp | 15% | +0.11 | 53% |
| Evo 1 | 3.3% | 10 bp | 15% | +0.17 | 39% |

**PROK's SD signal is real.** Its P(G) rises from a 0.15 baseline to **0.51**
across positions 9–12 — the G-core of `AGGAG` at the right offset — and its PWM
hits pile up at 7–9, matching the true SD distribution. This is not an artefact
of a shortened search window: the effective window is 33 of 35 positions, so
chance is 3.0%, and 11.9% is 4× that.

**PromoGen2 is the instructive contrast**: +1.45 bits above its own shuffle, so it
writes SD-*like* purine-rich sequence, but places it at chance. Producing a motif
and placing it are separable, and this benchmark separates them.

---

## What the models actually believe (teacher-forced probabilities)

One forward pass over `prompt + generation` recovers exactly the distribution each
base was drawn from. Entropy is over A/C/G/T only; 2.0 bits = no preference.

| model | task | mean entropy | min entropy | mean P(AT) |
|---|---|---|---|---|
| Evo 1 | −10 | 1.76 | 1.73 | 0.60 |
| Evo 2 | −10 | 1.84 | 1.79 | 0.58 |
| PROK | −10 | 1.90 | 1.84 | 0.59 |
| HyenaDNA | −10 | 1.86 | 1.84 | 0.60 |
| PromoGen2 | −10 | 1.66 | 1.64 | **0.78** |
| **PROK** | **SD** | 1.69 | **0.05** | 0.55 |
| PromoGen2 | SD | 1.41 | **0.00** | 0.52 |
| Evo 1 / Evo 2 / HyenaDNA | SD | 1.76–1.86 | 1.72–1.84 | 0.58–0.61 |

**Two models ever commit, and neither commits at a motif.**

- **PROK collapses to 0.05 bits at position 21**, where **92% of the probability
  mass sits on a non-ACGT token** — its segment delimiter, exactly 200 bp into the
  upstream, where its training records end. On the clean background the numbers
  are identical (0.050 vs 0.051 bits). It has memorised the length of its own
  training frame, entirely independently of the prompt content.
- **PromoGen2 collapses to 0.00 bits at position 34** — its EOS. It stops early on
  **100%** of Task 2 generations.

The SD signal in PROK is *not* this collapse: it is the softer G rise at positions
9–12, which shifts the distribution without driving entropy to zero. **Frame
memorisation and SD grammar are two separate phenomena at two different positions
in the same model.**

---

## Generating the whole upstream (−35 → start codon)

Prompted at the end of the −35 and left to generate the entire remainder (median
73 bp, up to 194), so one continuous sample must traverse the spacer, −10,
discriminator, TSS and 5′UTR before reaching the SD.

**PROK is the only model with any structure, and it is at the SD:** P(G) rises
0.18 → **0.52 peaking at −8** relative to the start codon, with C dropping to 0.08
and T to 0.11. Flat everywhere from −90 to −20.

Evo 1, Evo 2 and HyenaDNA are flat lines across all 90 bp. PromoGen2 has
excursions, but at −45 and −33 — not at any annotated element.

This is the strongest single panel in the benchmark. **In one model, one sample:
flat across the entire stretch containing the −10 box, then a hard commit at the
SD.** The element it learned and the element it did not, side by side, from the
same prompt.

---

## Context length: does more upstream help?

The reviewer's objection — *you only gave it 200 bp* — tested directly. Same task
(prompt ends at the −35, generate 45 bp, same scoring offsets); the only thing
that changes is how much real upstream precedes it. Real sequence only.

Upstreams were re-extracted from the MG1655 genome and validated: the 200 bp
window reproduces the existing dataset base-for-base for 680/681 genes.

**Exact-hit rate (%), 95% CI bootstrapped over genes** (the 20 samples per gene
are not independent; treating 14,280 generations as 14,280 trials would shrink
the intervals ~4.5× too far):

| model | 200 bp | 300 | 500 | 1000 | full intergenic |
|---|---|---|---|---|---|
| Evo 1 | 2.35 | 2.27 | 2.30 | 2.51 | 2.65 |
| HyenaDNA | 2.46 | 2.57 | 2.28 | 2.51 | 2.52 |
| PROK | 2.23 | **1.96** | **2.02** | 2.32 | 2.71 |
| **Evo 2** | 2.71 | **3.37** | **3.32** | **3.92** | 2.63 |
| *chance* | *2.50* | | | | |

**Three of four models are flat at chance from 200 bp to 1000 bp.** PROK is
*worse* with more context — its 300 and 500 bp CIs lie entirely below chance
([1.74, 2.19] and [1.79, 2.26]) — exactly as its memorised 200 bp frame predicts.
At 300+ bp it is off-distribution by construction.

**Evo 2 is a real exception and should be reported as one.** It rises
monotonically to 3.92% [3.40, 4.50], and that interval does not overlap its
200 bp interval [2.42, 3.03]. The improvement is genuine. But 3.92% against 2.5%
chance is 1.6× — it is still not placing the box.

**The intergenic condition validates the ladder.** Evo 2's full-intergenic point
lands at 2.63, essentially on its 200 bp value of 2.71 — and the median
intergenic window *is* 202 bp. The effect tracks window length, not noise. All
four models land within 2.52–2.71 on this condition, i.e. at chance, which is
what "every base of regulatory sequence this gene actually has" buys you.

### Why the ladder stops mattering above ~200 bp

**The median E. coli gene has 203 bp of intergenic sequence upstream.** Beyond
that, the added context is the neighbouring gene's coding sequence:

| context | median % intergenic | genes fully intergenic |
|---|---|---|
| 200 bp | **100%** | 51% |
| 300 bp | 68% | 24% |
| 500 bp | 41% | 6% |
| 1000 bp | **20%** | **1.3%** |

At 1000 bp, 98.7% of genes are reading into an upstream CDS. So the flat result
is not "we were stingy with context" — **beyond 200 bp there is no further
regulatory sequence to supply.** The biology agrees: the core promoter spans
−40 to +1, the UP element −40 to −60, and essentially all RegulonDB TFBS fall
within −200.

**Figure:** `fig6_context` · **numbers:** `context_ladder_source.csv`


---

## Leakage

**Negligible everywhere.** The largest leaked − clean gap on exact hits is
**0.44 percentage points** (Evo 2, Task 1); most are under 0.3 pp and two are
negative. PROK's frame collapse is identical on both backgrounds.

Whatever these models are doing, it is not recall of the memorised genome — even
for Evo 1 and Evo 2, whose training corpora contain MG1655. This is the
measurement the Mechanistic Invariance paper designed away and could not make.

---

## Why this matters for the paper

The benchmark reproduces the entropy-mass story **within a single model**:

- PROK commits at the **Shine–Dalgarno** (11.9%, 4× chance; P(G) 0.18 → 0.52 from
  73 bp away) and nowhere else.
- The same model is at chance for the **−10 box** (2.2%), with a probability
  profile flat to within 0.03.

Same model, same prompt format, same sampling — the difference is which element it
was asked for. It learned the high-mass, CDS-proximal, sequence-defined element
and not the low-mass, positionally-defined promoter grammar. That is the "commits
only at the SD" claim, now shown on generation rather than on entropy alone.

---

## Caveats to state

- **The 681 genes are a biased subset** — RegulonDB-annotated *and* compact enough
  for a whole promoter inside 200 bp. Enriched for short 5′UTRs and
  well-characterised σ70 promoters (487/681).
- **PromoGen2 runs on 642 genes for Task 1**, not 681: 39 lose their −35 box in its
  native 160 bp reframe. It also has its own short-prompt population (32 prompts
  under 20 bp vs 15 for the other models).
- **Evo 1 is the 8k-context checkpoint**, not 131k — the 131k variant sizes its KV
  cache to the full context and cannot be run at any usable batch size. Same model,
  same corpus.
- **PROK is the v3 fine-tuning run.**
- **The to-ATG run has leaked-background results for all five models but will not
  complete `clean` for Evo 1 and PROK** — recording probabilities at every step
  runs without a KV cache and those two hit the wall time. The 45/40 bp benchmark
  does have both backgrounds for them, and showed ≈0 leakage.
- **Functional oracles are not run.** Promoter Calculator (σ70 Tx rate) and RBS
  Calculator (ΔG) are specified in `PROCEDURE.md` but not applied; everything above
  is motif placement, sequence and probability, not predicted function.
- **PromoGen2 is absent from the context ladder by construction** — its 160 bp
  reframe is anchored to the start codon, so extra 5′ sequence never enters its
  window and every context gives a byte-identical prompt.
- **PROK's own SD→start-codon spacing is 0–1 bp** against a real mode of 6–7, and
  only 38% of its generations contain a start codon at all, so that number rests
  on a subset and should not be over-read.

---

## What to run next

1. **The two functional oracles** — the one specified metric still missing.
2. **The gene-specificity regression** (generated score vs real score per gene).
   `per_gene.csv.gz` is already written for it. This is the shortcut detector:
   whether PROK's SD placement tracks each gene or stamps one generic motif.
   The `p_AT_at_true` and `entropy_at_true` columns in `probs/*_pergen.csv.gz`
   give a more sensitive version of the same test.
3. **Stratify PROK's Task 2 by `sd_present`** — the 133 genes whose real RBS is
   weak are where "knows this gene" and "always stamps an SD" diverge.
4. **A frame-shift control for PROK** — truncate the prompt so the upstream is no
   longer 200 bp and re-check both the delimiter collapse and the SD placement.
   The collapse should move with the frame; if the SD placement moves with it too,
   the SD result is positional counting after all rather than grammar.
