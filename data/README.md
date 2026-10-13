> **What is shipped here.** This repository carries the prompt manifests, gene
> lists, per-model rank tables and summary JSON. The `benchmark/` tree described
> below — raw generations, likelihood matrices and per-shard probability tensors —
> is tens of gigabytes and stays on the cluster; `results/` holds the aggregates
> the paper reports and `code/` the scripts that produce them.

# Benchmark layout

Split by **what the model is conditioned on**. Both tracks share the same genes
(E. coli `up_only` = singletons + operon leaders, 200 bp upstream) and the same
annotation, so results are directly comparable across tracks.

```
benchmark/
  ecoli_upstream_master.csv   <- SHARED. 2717 genes, one row each: seq_leaked,
                                 seq_clean, -35/-10/TSS/SD/TFBS annotation, tier.
                                 Both tracks derive from this; belongs to neither.
  upstream_only/              <- prompt contains NO CDS   (built)
  cds_prompted/               <- prompt contains the CDS  (not built)
```

`leaked` = the real extracted upstream (models have seen it).
`clean`  = annotated motif bases kept at their exact positions, filler permuted
at exactly matched base composition. Same gene, same motif grammar, broken
memorization -> the leaked/clean gap is the leakage measurement.

---

## How 2720 narrows to 681

| step | genes | what is dropped |
|---|---|---|
| genes in the operon-annotated CSV | 4240 | (the GFF has 4340 CDS) |
| `up_only` = singletons + operon leaders | 2720 | 1520 internal/last operon genes — their "upstream" is the preceding CDS |
| present in `ecoli_upstream_master.csv` | 2717 | b3681, b4223, b4590 |
| has ≥1 RegulonDB promoter assigned | 1151 | **1569 genes have no annotated promoter at all** |
| TSS inside the 200 bp window | 1096 | 55 |
| −10 box inside the window | 769 | 327 |
| −35 box inside the window | 713 | 56 |
| −35 + −10 + TSS all present → tier `full_promoter` | 711 | b0799, b4720 (no TSS coordinate) |
| **−35 box not clipped by the window** (box width ≥ 6 bp) | **681** | 30 genes whose −35 is cut off by the 200 bp boundary — the task premise ("you are given the −35") cannot hold for them. **These rows are dropped at source**, so every surviving row has a full-width −35 and no flag column is carried. |

Two forces do most of the cutting: **annotation coverage** (2720 → 1151; only 42%
of `up_only` genes have any RegulonDB promoter — a property of the literature,
not the sequence) and **the 200 bp window** (1151 → 713; the −35 is the most
distal element, so a long 5'UTR pushes it out).

**Selection bias — state this in the paper.** The 681 are the intersection of
"studied enough to be in RegulonDB" and "compact enough to fit a whole promoter
inside 200 bp": enriched for short 5'UTRs and well-characterised, often
highly-expressed sigma70 promoters (487 of 681 are sigma70). Results generalise
to that subset first, and to the genome only by assumption.

---

## upstream_only/  — the benchmark

Prompt is a prefix of the upstream itself; no CDS anywhere. Tests whether the
model knows regulatory *grammar* at all, independent of CDS conditioning.

```
upstream_only/
  prompts_681_leaked.csv    681 rows x 31 cols   natural background
  prompts_681_clean.csv     681 rows x 31 cols   shuffled background
  plots/                    dataset-stat figures
```

**Two files, nothing else.** One row per gene, both tasks as columns. Every
column except the four sequence columns (`upstream_seq`, `prompt_promoter`,
`target_promoter`, `prompt_sd`) is byte-identical between them — same genes, same
cut points, same scoring offsets, only the background changes. `target_sd` is
identical in both, because the clean rule keeps the last 20 bp real.

`upstream_seq` is the full region (200 bp for all but one gene). Every index in
the file — `m35_start`, `m10_start`, `tss_idx`, `sd_start`, the cut points — is an
offset into it, and both prompts are its prefixes. Nothing is lost by the
task-1 trim below.

Built by `../../code/build_prompts_wide.py` (the clipped-−35 filter is applied at
source, so a rebuild reproduces these two files exactly).

### Task 1 — `generate_promoter`

```
prompt = upstream_seq[:cut_promoter]                cut_promoter = m35_end
target = upstream_seq[cut_promoter:cut_promoter+45]  the 45 bp continuation
```

**Generation budget = 45 bp.** It reaches the −10 box (max offset 24) and the TSS
(max offset 38) for every gene, and stops there. 21 genes have fewer than 45 bp
left before the ATG and simply carry what remains (min 32 bp).

The model must produce, in order: the spacer, the −10 box, the discriminator and
the TSS. Scored at offsets *into the generated continuation*, so correct
**spacing** is required, not just correct motifs:

- `m10_offset_in_target` → score the −10 with `motif_pwm`
- `tss_offset_in_target` → score the TSS with `promoter_calculator` (σ70 only)

Prompt length: median 127 bp, range 6–168. **15 genes have a prompt under 20 bp**
(`short_prompt=1`) — the −35 is intact but sits near the 5' edge, so there is
little conditioning context. Flagged, not removed; report with and without.

Because the target stops after the TSS it no longer reaches the SD, so **task 1
and task 2 are disjoint**, not nested — their scores are independent.

### Task 2 — `generate_sd`

```
prompt = upstream_seq[:L-20]                  the 180 bp before the SD region
target = upstream_seq[L-20:] + start_codon    the real 20 bp + the codon
```

Defined for **all 681 genes**. Scorer: `rbs_calculator`.

**The target is 20 bp; the generation budget is 40 bp.** Those are different
things. The target is the ground truth against which a generation is scored; the
40 bp is how much the model is asked to write at evaluation time, so it has room
to place its own start codon rather than being told where the ATG goes. The
budget is a runner setting, not a column.

**Do not score for `ATG` literally** — 602 genes start with ATG, but 59 are GTG
and 18 TTG. Score against the gene's own `start_codon`, and report separately
which codon the model chose ("always writes ATG" is itself a result). Two genes
(`fdnG`, `fdhF`, both selenoproteins) have no annotated start codon and carry a
20-character `target_sd`; the rest carry 23.

`sd_present` is a property of the *answer*, not a precondition. The 133 genes
with `sd_present=0` are kept deliberately: their real RBS scores only 2–3/6
against the `AGGAGG` consensus, so they are the genes where "the model knows
this gene" and "the model always stamps an SD" predict **opposite** results. The
548 with a real SD cannot separate those hypotheses; these 133 can.

Better still, skip the 4/6 binarisation and regress generated `sd_score` on real
`sd_score` across all 681 — a flat line near 4–6 regardless of truth is the
shortcut, quantified.

| `sd_score` | genes | |
|---|---|---|
| 2/6 | 14 | `sd_present=0` |
| 3/6 | 119 | `sd_present=0` |
| 4/6 | 325 | `sd_present=1` |
| 5/6 | 214 | `sd_present=1` |
| 6/6 | 9 | `sd_present=1` |

### Columns (31)

**Identity** — `locus_tag`, `gene`, `strand`, `L`, `promoter`, `sigma`

**Sequence** — `upstream_seq`, the full region every index below refers to.

**Annotation** (indices into `upstream_seq`; 0 = 5' end, L−1 = adjacent to the ATG)
— `m35_start`, `m35_end`, `m10_start`, `m10_end`, `tss_idx`, `sd_present`,
`sd_start`, `sd_end`, `sd_score`, `sd_spacing`, `start_codon`

**TFBS** — `n_tfbs`, `tfbs_regulators`, `tfbs_spans`, `tfbs_functions`.
`;`-separated and index-aligned, e.g. for `b0019`:

```
n_tfbs           4
tfbs_regulators  NhaR;NhaR;NhaR;NhaR
tfbs_spans       95-112;127-144;158-175;117-134
tfbs_functions   activator;activator;activator;activator
```

**Task 1** — `cut_promoter`, `prompt_promoter`, `prompt_promoter_len`,
`target_promoter`, `m10_offset_in_target`, `tss_offset_in_target`

**Task 2** — `prompt_sd`, `target_sd`

**Flags** — `short_prompt`

Lengths and cut points that are trivially derivable are not stored:
`target_promoter_len` (= `len(target_promoter)`), `prompt_sd_len` and `cut_sd`
(= `L − 20`), `target_sd_len` (= `len(target_sd)`), `m35_width`
(= `m35_end − m35_start`) and `m35_clipped` (always 0 — those rows are dropped).

### Dataset at a glance

| | |
|---|---|
| genes | 681 |
| SD typical / weak | 548 / 133 |
| with TFBS / without | 501 / 180 |
| >1 distinct regulator | 291 |
| sigma70 | 487 |
| `short_prompt=1` | 15 |

Verified on build, for both backgrounds: `prompt_promoter` and `prompt_sd` are
exact prefixes of `upstream_seq`; `prompt_promoter + target_promoter` equals
`upstream_seq[:cut_promoter+45]`; both scoring offsets fall inside the 45 bp
budget and land on the real annotated motif; and only the four sequence columns
differ between leaked and clean.

### What "clean" actually keeps

Kept bases: the −35 box, the −10 box, the TSS (a single base), every TFBS
footprint, and the last 20 bp. Everything else — **including the body of the
5'UTR** — is permuted. Mean `keep_frac` is 0.313, so ~69% of each upstream is
shuffled. The permutation reuses the gene's own filler bases, so base composition
is preserved *exactly*, not merely in expectation.

---

## cds_prompted/  (not built yet)

Prompt = the CDS, model generates the upstream de novo. This is the track that
tests the paper thesis directly (does the model condition on the CDS at all, or
regress to the species marginal), and the home for the CDS-swap / CDS-removal
controls. Derive it from the same 681 genes so the two tracks stay comparable.
