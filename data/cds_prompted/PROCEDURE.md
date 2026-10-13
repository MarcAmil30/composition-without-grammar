# CDS-prompted benchmark — procedure

Prompt = the CDS. The model writes the 200 bp upstream de novo.

Where `upstream_only/` asked *does the model know promoter grammar*, this asks
**does it condition on the gene at all**.

Companion track: `../upstream_only/` — same species, same scorers, same sampling.

---

## 1. Dataset

Singletons / monocistronic genes only. Every gene in the set has its own
promoter **by annotation**, not by a distance heuristic.

| | E. coli MG1655 | B. subtilis 168 |
|---|---|---|
| protein-coding CDS | 4,189 | 4,229 |
| **singletons with a 200 bp upstream** | **1,885** | **1,272** |
| clean window (`contam_bp = 0`) | 781 (41%) | 406 (32%) |

`contam_bp` is measured against the **nearest upstream feature on either strand**
(CDS, tRNA, rRNA, ncRNA) — that is how much of the 200 bp window physically sits
inside another gene's body. Measured against same-strand CDS only it would read
1,215 / 704, but that ignores divergent neighbours, and 48% / 41% of upstream
neighbours here are divergent. Both distances are carried in the prompt files
(`dist_any_feature`, `dist_same_strand_cds`) so either convention can be applied
downstream.

Source: `Proking_repo/data/raw_sequences_data_small_scale/{Escherichia_coli,Bacillus_subtilis}_operon_annotated.csv`,
columns `cds` and `upstream_200bp`.

**Drop non-CDS rows first.** The B. subtilis file carries 211 tRNA / misc_RNA
entries that are not protein-coding; they sink to the bottom of any CAI ranking
and distort it. Filter to rows with a matching CDS feature in
`GCF_000009045.1_ASM904v1_genomic.gff`.

Record per gene: `contam_bp = max(0, 200 − dist_upstream_cds)`,
`neighbour_strand` (tandem / divergent), `dist_upstream_cds`.

Singleton status comes from the annotation, not from a distance heuristic. The
≥50 bp operon rule is retained as a diagnostic only — on singletons it flags 13
genes in E. coli and 5 in B. subtilis, confirming the two criteria agree. Those
genes are kept; they are excluded from the clean subset anyway, since a <50 bp
gap implies `contam_bp` > 150.

**Contamination is real and is not removed by any of the above.** 35% (E. coli)
and 44% (B. subtilis) of kept windows still run backwards into the neighbouring
gene. That is a stratifier, not a filter.

---

## 2. Prompt

Native to PROK v4 — the training set contains 19,122 `CDS_5UTR_3UTR` rows of the
form `<sp>#<CDS><sp>$<200bp upstream><sp>*<200bp downstream>`.

```
+#<CDS>+$     → generate 200 bp     (E. coli)
@#<CDS>@$     → generate 200 bp     (B. subtilis)
```

Sampling: **20 samples at T = 1.0, plus 1 greedy decode.** Identical procedure
for every condition. No per-condition decoding tweaks.

---

## 3. Conditions

Same gene, same real upstream — only the CDS in the prompt changes. The start
codon is held fixed in every condition.

| | CDS given | preserved | nt changed |
|---|---|---|---|
| **C0** | real | — *(reference)* | — |
| **C1** | synonymous codons permuted within each amino acid | protein, codon counts, CAI, GC | 20% |
| **C2** | codon order shuffled | codon counts, GC | 72% |
| **C3** | swapped with a length-matched gene | real biology, wrong gene | 72% |
| **C4** | none | — | — |
| **C5** | codons 1–30 shuffled | tail, codon counts, GC | 11% |
| **C6** | codons 30–end shuffled | head, codon counts, GC | 60% |

**Every condition preserves CDS GC exactly.** C1 permutes which synonymous codon
sits at each position of a given amino acid, so protein, codon counts, CAI and GC
are all identical to C0 and only the arrangement changes. A uniform random
synonymous recode was rejected: it shifts GC by −1.4 pp (E. coli) / +3.1 pp
(B. subtilis) for only 4.6 pp more perturbation, and would confound C1 with a
composition change — the exact failure mode this benchmark exists to detect.

C0→C4 is a severity ladder: the rung at which the output stops changing tells you
which feature the model was using. C1 vs C0 is the sharpest — same protein, same
composition, different nucleotides. C2 = C5 + C6, so the positional pair
decomposes the full shuffle.

**C5 vs C6 is the decisive pair.** The first ~90 nt of the CDS folds with the
RBS and carries the codon ramp, so biology predicts C5 matters. But in this
prompt format the CDS 3' end is what sits nearest the generation point, so
recency predicts C6 matters. The two hypotheses make opposite predictions on the
same pair of arms.

---

## 4. Tasks

| | question | conditions |
|---|---|---|
| **T1** | does the CDS help at all? | C0 vs C4 vs the finished `upstream_only/` track |
| **T2** | is the upstream sensitive to *which* CDS? | C0–C6 |
| **T3** | does it condition on the *right* thing? | C0, split high vs low CAI |

---

## 5. Metrics

### M1 — base probabilities

At every one of the 200 generated positions, store the full distribution
P(A), P(C), P(G), P(T) and entropy — **for all 21 decodes separately (20 sampled
+ greedy), not just the aggregate**. This is the raw object; M2 and T3 are
computed from it.

Two passes:

- **Teacher-forced** on the real upstream → `KL(Cᵢ ‖ C0)` per position. The
  context is byte-identical across conditions, so any divergence is attributable
  to the CDS alone. Low variance; run on every gene and every condition.
- **Free-running** → the generations themselves.

### M2 — generation divergence

*How different are the upstreams it actually writes?*

| | |
|---|---|
| **profile JS** | per-position base-frequency profile from the 20 decodes; Jensen–Shannon divergence between conditions, summed over 200 positions |
| **pairwise identity** | mean % identity between condition A's decodes and condition B's decodes |

Neither means anything without the null: **within-condition** divergence (C0's
20 decodes against each other). Report the pair side by side.

> between ≈ within → the CDS changed nothing
> between > within → it conditions

Pairs to report: C0 vs each of C1–C6, and **high-CAI vs low-CAI under C0**.

The CAI pair is a different kind of comparison — two *gene groups*, not two
perturbations of one gene — so its null is two random halves of the same group,
not the within-gene spread. Same arithmetic, different null.

### M3 — SD placement

Best AGGAGG match in the last 20 nt, spacing 3–12 nt to the ATG, called present
at ≥ 4/6. Report position error, % exact, % within 2 bp — against chance and
against the model's own shuffled output.

### M4 — greedy identity

% of genes where `greedy(C0)` and `greedy(Cᵢ)` are byte-identical. Temperature 0,
so deterministic: if the strings match, the CDS was ignored. One number, no
statistics.

*(β and r from the gene-specificity regression: computed and stored, not reported.)*

---

## 6. T3 — conservation, not strength

High-CAI genes have lower-entropy Shine–Dalgarno regions in the real genome. If
PROK conditions on the CDS, its generations should be correspondingly tighter
for high-CAI genes.

**Report information content in bits, not mean SD strength.** In the real genome
the strength correlation is weak (Spearman +0.10 E. coli, +0.14 B. subtilis) —
reporting means would find nothing and wrongly read as model failure. The signal
is in conservation.

Lists: 200 high-CAI vs 200 low-CAI per species, cut from the singleton set —
`{ecoli,bsubtilis}_singletons_top200_{high,low}_CAI.csv`.

| | E. coli | B. subtilis |
|---|---|---|
| high-CAI list | 0.605 – 0.890 | 0.574 – 0.868 |
| low-CAI list | 0.152 – 0.330 | 0.332 – 0.455 |

**Real-genome ceiling** (n = 200 per group, singletons), information over −20…−1:

| | high CAI | low CAI | Δ | p |
|---|---|---|---|---|
| E. coli | 3.55 bits | 2.70 bits | +0.85 | 0.0079 |
| B. subtilis | **6.21 bits** | **4.65 bits** | **+1.56** | 5e-5 |

Supporting numbers: SD hexamer information 0.773 / 0.588 bits (E. coli) and
1.223 / 0.979 (B. subtilis); mean SD score 4.21 / 3.95 (p = 0.0018) and
5.04 / 4.71 (p = 1e-4); %SD ≥ 4 of 86 / 72 and 96 / 94.

**Test information content with a label-permutation test, not bootstrap CIs.**
Bootstrapping an alignment resamples duplicate sequences, which inflates apparent
conservation and widens the interval upward — on this data the bootstrap CIs
overlap for E. coli while the permutation test gives p = 0.0079. Permute the
high/low labels and recompute the difference in bits.

Two readouts:

- **between-gene** — 1 generation per gene, across the 200 genes. Matches the
  ceiling above, so this is the like-for-like number.
- **within-gene** — the 20 generations of one gene, averaged over genes. Needs no
  ceiling; the comparison is high vs low CAI within the same model.

If the model conditions on the CDS, both gaps go the same direction.

B. subtilis is the stronger test: nearly twice the SD information of E. coli and
a gap almost twice as wide, despite a narrower CAI contrast.

---

## 7. Reporting

- Every metric against the **real-sequence ceiling** — same scorer run on the
  real upstream.
- **Clean subset** (`contam_bp = 0`) as the headline; full singleton set as the
  robustness check.
- **Spread reported next to every median.** The spread across the 20 decodes is
  the noise floor; a median without it is uninterpretable.
- **E. coli and B. subtilis separate, never pooled.** Different GC, different SD
  strength, different genome compactness.
- **State the subset on the same line as the number.**
- **Report what happens.** A flat ladder with high greedy identity — PROK writing
  the species average regardless of gene — is the finding, not a failure.

---

## 8. Cost

| | |
|---|---|
| teacher-forced | 3,157 genes × 7 conditions ≈ **22k forward passes** |
| free-running | 3,157 genes × 7 conditions × 21 decodes ≈ **464k generations** |

Generate on **all** singletons and stratify by `clean` at analysis time, rather
than generating only on the clean subset. The clean subset is 781 + 406 = 1,187
genes; B. subtilis at 406 is too thin to carry a headline on its own. 464k is
1.6× the finished `upstream_only/` track (284,382).

CDS length: median 858 / 750 nt, p95 2,136 / 1,941, max ~7,000. Only 6 E. coli
and 3 B. subtilis genes exceed 4,000 nt, all well inside context. PROK's training
rows reach 26,787 nt, so nothing here is off-distribution by length.

Storage: keep all 21 decodes per gene per condition rather than collapsing them —
M2 needs them. 3,157 × 7 × 21 × 200 × 4 is ~7 GB in float32; store float16 and
keep the greedy decode in full precision.

---

## 9. Build status

- [x] `{ecoli,bsubtilis}_singletons_CAI.csv` — 1,885 / 1,272 genes
- [x] `{ecoli,bsubtilis}_singletons_top200_{high,low}_CAI.csv`
- [x] `t3_ceilings.json` — T3 ceilings at n = 200
- [x] `prompts_C{0..6}_{ecoli,bsubtilis}.csv` — 14 files, 0 assertion failures
- [x] `prompts_manifest.json` — seed, per-condition perturbation and GC summary

Ready to generate.

CAI provenance: verified against the shipped `ecoli_gene_CAI.csv`
(Pearson r = 0.9985 reproducing it with their `ecol.heg.fasta` reference,
r = 0.9876 against an independent 67-gene MG1655 ribosomal reference).
B. subtilis CAI computed the same way from 68 B. subtilis ribosomal + EF genes;
translational selection in B. subtilis is well established (Shields & Sharp 1987,
NAR 15:8023). Scripts: `cai_audit.py`, `cai_audit2.py`, `cai_stats.py` on Isambard.
