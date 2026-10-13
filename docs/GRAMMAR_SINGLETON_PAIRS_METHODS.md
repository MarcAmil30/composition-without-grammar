# Grammar singleton-pair figure: methods and interpretation

## Main-paper purpose

This figure provides model-independent motivation for the ICBINB_BIO study. It tests whether
natural bacterial genes and their upstream regions have a measurable sequence association before
asking whether autoregressive models learn or use that association.

The analysis is restricted to singleton genes on both sides, so every retained gene has an
independent upstream region rather than an operon-shared promoter.

## Dataset and filtering

- Anchor set: 727 *Escherichia coli* K-12 singleton genes with at least 100 bp of available
  upstream sequence.
- Orthologues: identified through the existing MMseqs2 Gammaproteobacterial gene families after
  mapping each *E. coli* protein to a family representative with BLASTP.
- Family-hit thresholds: at least 30% protein identity and at least 70% query coverage.
- Target filter: `operon_size == 1` and at least 200 bp of available upstream sequence.
- At most one family member was retained per *E. coli* query and target genome.
- Upstream orientation: all sequences were represented in the gene's transcriptional direction.
- Query frame: the proximal 100-200 bp available for the *E. coli* gene, capped at 200 bp.
- Target frame: the proximal 200 bp upstream of the target orthologue.
- Final plotted sample: 22,393 singleton-singleton orthologue pairs from 408 target genomes.

## Non-homologous control

For every true orthologue pair, the same *E. coli* query upstream was also compared with a control
upstream. The control was:

1. a singleton gene (`operon_size == 1`);
2. from the same target genome as the true orthologue;
3. outside every tested orthologue family;
4. required to have at least 200 bp upstream; and
5. selected from candidates whose available upstream length was within +/-10% of the true target
   where possible.

The same-genome construction holds genome-wide GC composition, regulatory style, annotation
practice and phylogenetic distance from *E. coli* constant. Random sampling used seed `20260813`.

## Upstream-similarity metric

Similarity was measured with normalized D2* using overlapping 4-mers (`k = 4`, stride 1).
For each sequence, counts were obtained for all 256 possible 4-mers. The expected count of each
4-mer was estimated from an order-2 Markov background fitted to that sequence:

```text
E(w) = N(prefix of length 3) x N(suffix of length 3) / N(middle of length 2)
```

Each observed count was converted to a composition-adjusted residual:

```text
(observed - expected) / sqrt(expected)
```

D2* is the normalized dot product between the two residual vectors. Higher values indicate that
the upstreams share short sequence patterns beyond their expected local composition. The method
is alignment-free and does not require a motif to occur at exactly the same coordinate.

## Gene-relatedness axis

The x-axis is identity at third codon positions across the BLASTP-aligned coding region. Only
alignments with at least 30 informative codons were retained. Third-position identity is primarily
used as an evolutionary-relatedness or ancestry measure; it should not be described as a direct
measurement of protein function.

Points are coloured by the target genome's taxonomic relationship to *E. coli*. The black and
grey lines are medians in 5-percentage-point third-position-identity bins, requiring at least 30
pairs per bin.

## Statistical result

| Comparison | n | Spearman rho | Pearson r | Pearson r-squared |
|---|---:|---:|---:|---:|
| True singleton orthologue upstream | 22,393 | +0.437 | +0.686 | 0.470 |
| Same-genome non-homologous singleton control | 22,393 | +0.027 | +0.023 | 0.001 |

The singleton restriction therefore does not remove the relationship: more closely related genes
have more similar upstream sequences, whereas unrelated singleton upstreams from the same genome
show essentially no trend.

## Interpretation and limitation

The figure establishes a model-independent gene-upstream association and shows that it is not an
artefact of operon-shared promoters or genome-wide composition. It does not by itself prove
functional co-adaptation, because an orthologous coding region and its upstream region descend
from the same ancestral locus. The separate adjusted functional co-adaptation analysis should be
cited for the smaller protein-related component beyond ancestry.

## Suggested main-paper caption

> **Related singleton genes have similar upstream regions.** Alignment-free D2* similarity
> (`k = 4`) between the upstream regions of 22,393 singleton-singleton orthologue pairs from 408
> Gammaproteobacterial genomes increased with third-codon-position identity (Spearman
> rho = 0.437). Upstreams of non-homologous singleton genes sampled from the same target genomes
> showed no corresponding trend (rho = 0.027). Lines show running medians; colours denote the
> target genome's taxonomic relationship to *E. coli*. This establishes a model-independent,
> predominantly ancestry-linked association between genes and their upstream regions.

## Files in the ICBINB_BIO project

```text
figures/fig_grammar_singleton_pairs.pdf
figures/fig_grammar_singleton_pairs.png
figures/fig_grammar_singleton_pairs_stats.csv
code/grammar_singleton_pairs/grammar_step2_singleton_pairs.py
code/grammar_singleton_pairs/grammar_step3_singleton_pairs.py
code/grammar_singleton_pairs/grammar_step4_singleton_pairs.py
code/grammar_singleton_pairs/make_fig_grammar_singleton_pairs.py
```

The strict singleton analysis ran on KCL CREATE as job `36652178`. These files use distinct names
and do not replace any existing ICBINB_BIO figure or result.
