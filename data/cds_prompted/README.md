# cds_prompted/ — dataset construction and column reference

Prompt = the CDS, model generates the 200 bp upstream de novo. Companion file:
`PROCEDURE.md` (tasks, conditions, metrics, reporting).

Track status: **built and ready to generate.**

---

## Source

| | E. coli MG1655 | B. subtilis 168 |
|---|---|---|
| sequences | `Proking_repo/data/raw_sequences_data_small_scale/Escherichia_coli_operon_annotated.csv` | `…/Bacillus_subtilis_operon_annotated.csv` |
| annotation | `GCF_000005845.2_ASM584v2_genomic.gff` | `GCF_000009045.1_ASM904v1_genomic.gff` |
| columns used | `locus_tag`, `cds`, `upstream_200bp`, `status` | same, `gene_type` in place of `status` |

`upstream_200bp` is the 200 bp immediately 5′ of the translation start, strand
corrected. It is populated only for singletons and operon leaders; internal and
last operon genes carry an empty string, which is correct — for those the 200 bp
before the ATG is the coding body of the gene in front.

---

## Filters, in order

1. **Protein-coding only.** Require a matching `CDS` feature in the GFF.
   Drops 51 rows in E. coli and **219 in B. subtilis** — the B. subtilis file
   carries 211 tRNA / misc_RNA entries. Skipping this step puts non-coding genes
   at the bottom of every CAI ranking and inflates the T3 effect (the low decile
   appears to have 58% SD presence instead of the true 90%).
2. **In frame, ≥ 30 codons.** CAI is unstable on very short ORFs.
3. **Singleton / monocistronic**, by annotation.
4. **Has a 200 bp upstream.**

| | E. coli | B. subtilis |
|---|---|---|
| rows in source file | 4,240 | 4,449 |
| protein-coding, in frame, ≥30 codons | 4,189 | 4,229 |
| **singletons with a 200 bp upstream** | **1,885** | **1,272** |
| of which clean window (`contam_bp = 0`) | 781 (41%) | 406 (32%) |

The ≥50 bp operon-distance rule is a diagnostic, not a filter: on singletons it
flags 13 genes in E. coli and 5 in B. subtilis, confirming that the annotation
and the distance criterion agree.

---

## Per-gene columns to carry

| column | meaning |
|---|---|
| `locus_tag` | b-number (E. coli) / BSU_ tag (B. subtilis) |
| `gene`, `species` | from the GFF / the species token |
| `condition` | `C0`…`C6` |
| `prompt` | the exact string fed to the model, delimiters included |
| `target_upstream_200bp` | the real 200 bp — the teacher-forcing target |
| `cds_len`, `cds_len_c0` | perturbed and original CDS length |
| `pct_nt_changed` | nucleotide distance from C0 (blank for C0 / C4) |
| `partner_locus_tag` | C3 only — whose CDS was substituted in |
| `CAI` | see below |
| `gc_cds`, `gc_cds_c0` | perturbed and original CDS GC — equal by construction |
| `dist_same_strand_cds` | bp to the end of the nearest same-strand upstream CDS |
| `dist_any_feature` | bp to the nearest upstream feature on either strand |
| `neighbour_strand` | tandem / divergent |
| `contam_bp` | `max(0, 200 − dist_any_feature)` |
| `clean` | 1 if `contam_bp == 0` |

**`contam_bp` is a stratifier, not a filter.** 59% (E. coli) and 68%
(B. subtilis) of kept windows still run backwards into a neighbouring gene,
because bacterial genomes are compact. It is measured against the nearest
upstream feature on **either** strand — that is the DNA physically occupied by
another gene. Against same-strand CDS only the clean counts would read 1,215 /
704, but 48% / 41% of upstream neighbours here are divergent, so that convention
overstates cleanliness. Both distances are in the files; pick either downstream.

Where the neighbour is *divergent* the shared region is genuinely regulatory for
both genes; where it is *tandem* the model is being asked to write the 3′ end of
a coding gene. Report the two apart.

---

## CAI

Standard Sharp & Li: w = codon count / max count within its synonymous family,
Met / Trp / stops excluded, CAI = geometric mean of w over the gene.

Reference sets are the ribosomal-protein and elongation-factor genes of each
genome (67 for E. coli, 68 for B. subtilis), taken from **all protein-coding
genes** — not from the singleton subset, since ribosomal genes are mostly
operon-internal and a singleton-only reference would be badly undersampled.

Verified against the shipped `ecoli_gene_CAI.csv`: Pearson **r = 0.9985**
reproducing it with its own `ecol.heg.fasta` reference, **r = 0.9876** against
the independent MG1655 ribosomal reference. The implementation is correct and
insensitive to the choice of reference. Translational selection in B. subtilis is
well established (Shields & Sharp 1987, *NAR* 15:8023), so the same method
applies there.

Known caveat: low-CAI genes are shorter (E. coli median 513 nt vs 958 nt for
high-CAI). A length-matched control leaves the T3 effect intact in B. subtilis
and slightly reduced in E. coli.

---

## Files

### Prompt files

`prompts_C{0..6}_{ecoli,bsubtilis}.csv` — 14 files, one row per gene per
condition, 1,885 / 1,272 rows each. Columns as above.

Prompt format, native to PROK v4 (`<sp>#<CDS><sp>$<upstream>`):

```
+#<CDS>+$     E. coli          @#<CDS>@$     B. subtilis
+$            E. coli, C4      @$            B. subtilis, C4
```

C4's prompt is exactly the one used by `../upstream_only/`, so T1 compares
against that track directly.

Operators, all applied with seed 20260801, start and stop codons always held
fixed:

| | operation | preserved | nt changed (E. coli / B. subtilis) |
|---|---|---|---|
| C0 | — | — | — |
| C1 | permute synonymous codons within each amino acid | protein, codon counts, CAI, GC | 20.0% / 20.6% |
| C2 | shuffle codon order | codon counts, GC | 72.0% / 71.2% |
| C3 | substitute a length-matched partner's CDS | real biology | 71.9% / 71.9% |
| C4 | remove the CDS | — | — |
| C5 | shuffle codons 1–30 | tail, codon counts, GC | 10.7% / 11.9% |
| C6 | shuffle codons 30–end | head, codon counts, GC | 60.8% / 58.7% |

**Every condition preserves CDS GC exactly.** C1 is a within-amino-acid
permutation, not a uniform synonymous recode — the latter shifts GC by −1.4 pp
(E. coli) / +3.1 pp (B. subtilis) for only 4.6 pp more perturbation, which would
confound C1 with the very composition change this benchmark tests for.

C3 pairs genes by sorted CDS length and swaps adjacent pairs, so the substituted
CDS is length-matched and the mapping is a true derangement.

C2 = C5 + C6: the positional pair decomposes the full shuffle.

Verified at build time, 0 failures over all 3,157 genes × 7 conditions: C1
preserves the translated protein, codon multiset and GC; C2 preserves the codon
multiset; C5 leaves nt 90+ byte-identical; C6 leaves nt 0–90 byte-identical; all
of C1/C2/C5/C6 preserve the start codon and total length.

### Data files

| file | contents |
|---|---|
| `prompts_C{0..6}_{ecoli,bsubtilis}.csv` | 14 prompt files, 1,885 / 1,272 rows each |
| `prompts_manifest.json` | seed, per-condition perturbation and GC summary, geometry counts |
| `ecoli_singletons_CAI.csv` | 1,885 genes, CAI 0.152 – 0.890 |
| `bsubtilis_singletons_CAI.csv` | 1,272 genes, CAI 0.332 – 0.868 |
| `ecoli_singletons_top200_high_CAI.csv` | 200 genes, 0.605 – 0.890 |
| `ecoli_singletons_top200_low_CAI.csv` | 200 genes, 0.152 – 0.330 |
| `bsubtilis_singletons_top200_high_CAI.csv` | 200 genes, 0.574 – 0.868 |
| `bsubtilis_singletons_top200_low_CAI.csv` | 200 genes, 0.332 – 0.455 |
| `t3_ceilings.json` | T3 real-genome ceilings at n = 200 |

Columns in all CAI files: `locus_tag, gene, product, CAI, cds_len, contam_bp`.

Spot check — the extremes are biologically right. E. coli high: `lpp, eno, rplL,
gapA, ompC, ahpC, cspA, tufA`; low: `y`-genes and small ORFs. B. subtilis high:
`cspB, tufA, sspA, rpmEA, rpsB, fusA`; low: sporulation and coat genes.

---

## T3 ceilings

Information content over −20…−1 in the real upstreams, n = 200 per group:

| | high CAI | low CAI | Δ | p |
|---|---|---|---|---|
| E. coli | 3.55 bits | 2.70 bits | +0.85 | 0.0079 |
| B. subtilis | 6.21 bits | 4.65 bits | +1.56 | 5e-5 |

p from a label-permutation test on the difference in bits. **Do not use bootstrap
CIs here** — resampling an alignment duplicates sequences, inflates apparent
conservation, and makes the E. coli comparison look non-significant when it is not.

The effect is in *conservation*, not mean strength: the strength correlation is
weak in both genomes (Spearman +0.10 E. coli, +0.14 B. subtilis), consistent with
Rocha et al. 1999 (*NAR* 27:3567) reporting that B. subtilis RBS strength does not
track codon usage class, while Abolbaghaei et al. 2017 (PMC5427494) find highly
expressed genes use longer SDs.

---

## Selection bias — state this in the paper

Singletons are not a random sample of the genome. They are depleted of the
co-regulated, operon-organised core (ribosomal, ATP synthase, most biosynthetic
pathways) and enriched for independently regulated genes. B. subtilis loses more
than E. coli (1,272 of 2,034 singleton-plus-leader genes, vs 1,885 of 2,418).
Results generalise to that subset first.

---

## Provenance

Built by `cai_audit.py`, `cai_audit2.py`, `cai_stats.py`, `build_lists.py`,
`perm_bits.py`, `build_prompts.py` on Isambard at `${WORK_ROOT}/`.
Random seed 20260801 throughout; re-running `build_prompts.py` reproduces the
prompt files byte for byte.
SD scorer is the one from `../../code/build_dataset.py`: best AGGAGG match in the
last 20 nt, spacing 3–12 nt to the ATG, called present at ≥ 4/6.
