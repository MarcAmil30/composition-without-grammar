# Figure caption

**Fig. X | Retrieval of cognate upstream regions in *Escherichia coli*.**
For each of 500 monocistronic genes, the true 200 bp upstream sequence was ranked
against 99 upstream sequences drawn from other genes in the same genome
(20 independent decoy draws per gene; identical genes and decoys for every model).
Each model is shown with its better-performing approach: embedding cosine
similarity for C3P, and normalised log-likelihood for the autoregressive models.
**A**, Rank assigned to the true upstream (log scale; 1 = ranked first). Boxes,
interquartile range; white line and diamond, median; whiskers, 1.5 × interquartile
range; dashed line, chance (rank 50). **B**, Top-1 accuracy with 95% bootstrap
confidence intervals over genes; chance, 1%. **C**, Recall@*k*, the proportion of
genes whose true upstream falls within the top *k* candidates; dashed line, chance.
Two genes with coding sequences longer than 4.3 kb were excluded as queries but
retained as candidates.

---

# Results

## Genomic language models do not link a gene to its own regulatory region

Generating a realistic upstream sequence does not require knowing *which* gene it
belongs to. We therefore tested recognition directly: for each of 500 monocistronic
*E. coli* genes we presented the true 200 bp upstream sequence together with 99
upstream sequences from other genes in the same genome, and ranked all 100
candidates (Fig. X). Drawing decoys from the same genome controls nucleotide
composition by construction, and every model was given identical genes and
identical decoys.

We evaluated two scoring approaches for each model: cosine similarity between
independently computed embeddings of the coding and upstream sequences, and the
model's log-likelihood of the two regions concatenated. The autoregressive models
performed substantially better under likelihood scoring, with their embeddings at
or near chance (Evo 2: 1.0% by embedding versus 6.0% by likelihood). This is
expected — next-token training optimises sequence likelihood, not a metric
embedding space — and we therefore report likelihood scoring for the autoregressive
models. C3P, which is contrastively trained and has no likelihood head, is reported
by embedding similarity.

C3P ranked the correct upstream first for 82.7% of genes (median rank 1),
establishing that the task is solvable and that upstream sequences carry
gene-specific information recoverable from sequence alone. All genomic language
models performed far worse: Evo 1, 6.3% (median rank 25); Evo 2, 6.0% (rank 31);
PROK, 2.8% (rank 36); and HyenaDNA, 1.0% (rank 47), indistinguishable from chance.
For PROK the effect is statistically real but weak — the true upstream outscored the
average decoy for 63.7% of genes (*P* = 3 × 10⁻¹⁴, Wilcoxon signed-rank) — yet it
ranks the correct sequence first only once in 35 attempts.

The ordering is the informative result. PROK performed **below both Evo models from
which it was derived**, despite fine-tuning on bacterial coding–upstream pairs. Its
scores were dominated by a gene-independent term: 97.8% of the variance reflected
how intrinsically probable each candidate upstream was, identical for every query,
leaving only 2.1% attributable to the pairing itself. Fine-tuning therefore improved
the model's account of what an *E. coli* upstream looks like in general without
improving, and in fact degrading, its ability to tie a regulatory region to the gene
it regulates.

Together with the generation results, this gives a clear dissociation: these models
produce upstream sequences that are typical of the species yet not specific to the
gene.
