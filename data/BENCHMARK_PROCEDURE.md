# Upstream-only benchmark — procedure

Prefix-continuation benchmark for regulatory-sequence generation in *E. coli*.
No CDS anywhere in the prompt: this track tests whether a model knows promoter
**grammar** at all, independent of CDS conditioning.

Companion file: `README.md` (dataset construction and column reference).

---

## 1. Dataset

| | |
|---|---|
| Source | *E. coli* MG1655, `up_only` genes (singletons + operon leaders), 200 bp before the start codon |
| Annotation | RegulonDB v12 — −35 box, −10 box, TSS, TFBS; Shine–Dalgarno computed |
| Genes | **681** — those with −35 + −10 + TSS all inside the window and the −35 not clipped |
| Sigma factors | σ70 487 · σ38 82 · σ32 41 · σ24 32 · σ54 29 · σ28 1 |
| Files | `prompts_681_leaked.csv`, `prompts_681_clean.csv` — 31 columns, one row per gene |

**Two backgrounds. Same genes, same cut points, same annotation.**

| | filler | motifs |
|---|---|---|
| `leaked` | the real genomic sequence | real, at their real positions |
| `clean` | permuted, **exactly** matched base composition | real, at their real positions (−35, −10, TSS, TFBS, last 20 bp) |

~69% of each upstream is permuted in `clean` (mean `keep_frac` 0.313). Grammar
intact, memorisation broken. Only four columns differ between the two files:
`upstream_seq`, `prompt_promoter`, `target_promoter`, `prompt_sd`.

**Selection bias — state this in the paper.** The 681 are the intersection of
"studied enough to be in RegulonDB" and "compact enough to fit a whole promoter
inside 200 bp". Enriched for short 5′UTRs and well-characterised, often
highly-expressed σ70 promoters. Results generalise to that subset first.

---

## 2. What the model is asked to do

|  | Task 1 — `generate_promoter` | Task 2 — `generate_sd` |
|---|---|---|
| **Prompt** | `upstream_seq[:m35_end]` | `upstream_seq[:L−20]` |
| Prompt length | median 127 bp (range 6–168) | 180 bp |
| **Generate** | **45 bp** | **40 bp** |
| Must produce | spacer → −10 box → discriminator → TSS | SD → gap → start codon |
| Ground truth | `target_promoter` (45 bp; 21 genes have 32–44) | `target_sd` (20 bp + start codon) |
| Anchor the model has | left — the −35 it was just given | none — it must place its own start codon |
| Genes | 681 | 681 |

Both tasks run on **both backgrounds** → 4 generation sets per model.

The two tasks are **disjoint**: Task 1's target stops after the TSS and never
reaches the SD, so the scores are independent rather than nested.

**Why 45 and 40.** 45 bp reaches the −10 box (max offset 24) and the TSS (max
offset 38) for every gene and stops there. 40 bp gives the model room past where
the start codon should be (offset 20) so it places its own anchor instead of
being told where the ATG goes. Both are asserted at build time.

---

## 3. Metrics

### Setup

For a motif of width $w = 6$, build a position weight matrix from the **real
annotated boxes** — sigma-specific for the −10, SD-anchored for the SD. With
$p_j(b)$ the base frequency at position $j$ and $b_0$ the upstream background
composition (A .289, C .209, G .208, T .295):

$$S(x) \;=\; \sum_{j=1}^{w} \log_2 \frac{p_j(x_j)}{b_0(x_j)}$$

Scan the generation $g$ of length $G$ and take the best hit:

$$\hat{p} \;=\; \arg\max_{0 \le p \le G-w} S\big(g[p:p{+}w]\big),
\qquad \hat{x} \;=\; g[\hat{p}:\hat{p}{+}w]$$

### Task 1 — all 681 genes

| # | Metric | Formula | Reads as |
|---|---|---|---|
| 1 | Position error | $\Delta_{\text{pos}} = \lvert \hat{p} - p^{*} \rvert$, $p^{*}=$ `m10_offset_in_target` | right place |
| 2 | Sequence error | $\Delta_{\text{seq}} = \sum_{j=1}^{6} \mathbb{1}[\hat{x}_j \neq r_j]$, 0–6 vs the gene's real box | right letters |
| 3 | Spacer length | $\hat{p}$ itself (the prompt ends at the −35, so the offset *is* the spacer); compare the **distribution** to real (mode 17, tight) | right spacing |
| 4 | Function | Promoter Calculator on prompt+generation → $\log_{10}$ Tx rate, predicted TSS offset. **σ70 only, n = 487** | does it work |

Reference for metric 1: a model with no idea places $\hat{p}$ at random over
45 bp → **median error ≈ 13 bp**, flat distribution. Peaked at 0 = it knows.

### Task 2 — all 681 genes

| # | Metric | Formula | Reads as |
|---|---|---|---|
| 1 | Position error | $\Delta_{\text{pos}} = \lvert \hat{p} - p^{*} \rvert$, $p^{*}$ from `sd_start` | right place |
| 2 | Sequence error | Hamming vs the gene's real SD hexamer, 0–6 | right letters |
| 3 | Start codon | is one emitted, and at what offset? truth = **20** | right anchor |
| 4 | Own spacing | $s = a - (\hat{p} + 6)$, $a$ = the model's own codon offset; compare distribution to real (mode 6–7, 72% in 5–8, range 3–12) | right grammar |
| 5 | Function | RBS Calculator → $\Delta G_{\text{total}}$ of the generated leader | does it work |

**Do not score for `ATG` literally.** 602 genes start with ATG, 59 with GTG,
18 with TTG. Score against the gene's own `start_codon` and report separately
which codon the model chose — "always writes ATG" is itself a result. Two genes
(`fdnG`, `fdhF`, both selenoproteins) have no annotated start codon.

**Stratify by the 133 genes with `sd_present=0`.** Their real RBS scores only
2–3/6, so they are the genes where "the model knows this gene" and "the model
always stamps an SD" predict **opposite** results. The 548 typical genes cannot
separate those hypotheses; these 133 can.

### Cross-cutting

**Leakage gap — the headline.** For any metric $M$:

$$\mathcal{G} \;=\; M_{\text{leaked}} - M_{\text{clean}}$$

A large gap means the apparent skill was memorisation. This is a **within-model**
comparison — each model is its own control — so it survives the models being
differently trained.

**Gene-specificity — the shortcut detector.** Regress the generated score on the
real score across genes:

$$y_i \;=\; \alpha + \beta x_i$$

$\beta \approx 1$: the model tracks each gene. $\beta \approx 0$ with a high
$\alpha$: it stamps the same generic motif on all 681 genes. This is invisible in
any average-based metric, which is why it is the most informative single number
in the benchmark. Report $\beta$ and Pearson $r$.

**Distribution match — the mode-collapse detector.** For spacer and spacing:

$$D_{\mathrm{KL}} \;=\; \sum_k P_{\text{real}}(k)\,
\log_2 \frac{P_{\text{real}}(k)}{P_{\text{gen}}(k)}$$

A model that always writes exactly 17 bp scores perfectly on the mode and wrongly
on the biology. Metrics 1 and 3 (Task 1) are the per-gene and per-population
views of the same number — report both.

**Ceiling.** Score the real target with the same pipeline and report the model as
a fraction of it.

---

## 4. Models

All five are run **as-is**. No fine-tuning. The only per-model adaptation is
PROK's prompt prefix.

| Model | Architecture | Trained on | How it is used here |
|---|---|---|---|
| **Evo 1** | StripedHyena, 7B, nucleotide-level | OpenGenome — prokaryotic genomes, phage, plasmid | Native. Prefix continuation *is* its training objective. MG1655 is in its corpus, so the leaked set is genuinely leaked. Both tasks. |
| **Evo 2** | StripedHyena 2, 7B / 40B | OpenGenome2 — all domains of life | Native, same as Evo 1 with a broader corpus. Both tasks. |
| **PROK** | Evo 1 + GRPO (ours) | 16-species corpus; 31,609 rows are `<species_token>$<200 bp upstream>` | The **v3** fine-tuning run. Native, **but the prompt must be prefixed with `+$`** (E. coli token + upstream delimiter) or it is off-distribution. Both tasks. |
| **HyenaDNA** | Hyena, 8 layers, `d_model` 256, single-character ACGT tokenizer | human hg38 **only** | Checkpoint **`LongSafari/hyenadna-medium-450k-seqlen-hf`** — HyenaDNA's top capacity tier, architecturally identical to `large-1m` (same `n_layer`, `d_model`, `d_inner`); only `max_seq_len` differs (450k vs 1M), which is irrelevant at 220 bp inputs. Run as-is, **not fine-tuned**. The question is: how well does an hg38-trained model do on bacterial promoter grammar? Whatever it scores is the answer. Both tasks. |
| **PromoGen2** | GPT-2, 30 layers, `n_embd` 640, `n_positions` **256**, ~0.1B params | 160 bp upstream of the translation start site, 17,806 prokaryotic genomes | Checkpoint `jinyuan22/promogen2-base`. Run in its **native 160 bp frame** (below). Its training window ends at the start codon, so it *did* see the 5′UTR and SD. **Both tasks.** |

### PromoGen2 — run it in its native 160 bp frame

PromoGen2's architectural limit is 256 positions, so nothing is blocked by
context. The real constraint is its **trained length of 160 bp**, and the frame
matches ours exactly: 160 bp ending at the start codon.

**Rule: reframe each gene to the last 160 bp of its 200 bp upstream**, then cut
the prompts as normal within that frame.

| | genes lost | why |
|---|---|---|
| Task 1 | **39** (5.7%) | their −35 box lies in the first 40 bp of the 200 bp window and falls outside the reframe |
| Task 2 | **0** | the prompt becomes 140 bp, entirely in-frame |

The rejected alternative was to use the full 256 positions and truncate only the
4 over-long prompts. It loses no genes, but **413 of 681 would then generate past
160 bp** — beyond anything the model saw in training. A low score would be
unreadable between "does not know the grammar" and "was asked to write past its
trained length". Losing 39 genes is the cheaper price.

**Report the 39 dropped genes wherever PromoGen2 appears.** They are the
long-5′UTR genes, so its subset is biased toward compact promoters.

**PromoGen2 has its own short-prompt set.** The reframe shortens every Task 1
prompt (median 127 → 90 bp), so the `short_prompt=1` flag in the CSV — computed
in the 200 bp frame — undercounts for this model. In its 160 bp frame:

| Task 1 prompt | genes |
|---|---|
| < 5 bp | 8 |
| < 10 bp | 13 |
| < 20 bp | **32** (vs 15 for every other model) |
| < 40 bp | 81 |

Use the 32 when stratifying PromoGen2, not the 15.

**EOS quirk.** PromoGen2's training sequences *end* at the start codon, so on
Task 2 it will likely emit EOS around offset 20 rather than write a codon. Use
its **EOS position as the predicted start-codon position** for Task 2 metric 3 —
a fairer and sharper test of whether it knows where the gene starts. The 40 bp
budget is unchanged; the model simply stops early, and that stopping point is the
measurement.

### Why mixed provenance does not break the result

The headline metric $\mathcal{G} = M_{\text{leaked}} - M_{\text{clean}}$ is
computed **within** each model, so a model being off-domain or differently
trained does not invalidate its own leakage measurement. Cross-model comparison
of *absolute* scores is the weaker claim and must be stated as such.

---

## 5. Fixed before generation

These are baked into the outputs and cannot be changed without re-running the
models:

| | |
|---|---|
| Generation length | 45 bp (Task 1), 40 bp (Task 2) |
| Sampling | **20 samples per prompt at T = 1.0, plus 1 greedy decode** |
| Coverage | 5 models × 2 tasks × 2 backgrounds × 681 genes (PromoGen2: 642 genes on Task 1, 681 on Task 2) |
| Sampling procedure | **identical for every model** — same temperature, same sample count, same greedy pass. No per-model decoding tweaks. |

The only per-model adaptations are to the **input framing**, never to the
sampling: PROK gets a `+$` prefix, PromoGen2 is reframed to its native 160 bp
window. Both are recorded with the generations.

Everything downstream — the PWMs, Promoter Calculator, RBS Calculator, score
thresholds — is computed after the fact and can be revised without re-generating.

### Reporting checklist

- [ ] Every metric reported for `leaked` and `clean`, plus the gap $\mathcal{G}$
- [ ] Every metric reported against the real-sequence ceiling
- [ ] Task 1 stratified with and without the 15 `short_prompt=1` genes
- [ ] Task 2 stratified by `sd_present`
- [ ] PromoGen2's 39 dropped Task-1 genes stated wherever it appears
- [ ] $\beta$ and $r$ from the gene-specificity regression, not only means
