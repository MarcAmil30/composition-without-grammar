# CDS midpoint completion — results summary

- Genes prompted: **672**
- Models: evo1, evo2, hyenadna
- Continuation length: 40 bp, 1 greedy + 20 sampled decodes per gene per condition
- Figures: `supp_cds_completion_probabilities.pdf`
- Full metrics: `cds_completion_metrics.csv`

## Shard health

| model | shards | genes | invalid_generations_dropped |
| --- | --- | --- | --- |
| Evo 1 (8k) | 27 | 672 | 114 |
| Evo 2 (1B) | 27 | 672 | 1 |
| HyenaDNA (medium) | 27 | 672 | 0 |

Failed shards this run: **0**

## Codon periodicity (RMS Fourier amplitude at 1 cycle / 3 bp)

Observed CDS gives **0.0391**. Anything near that reproduces real
codon structure; anything near zero does not.

| model | frac_of_real_free_native | free_running/native | free_running/shuffled | teacher_forced/native | teacher_forced/shuffled |
| --- | --- | --- | --- | --- | --- |
| Evo 1 (8k) | 0.9780 | 0.0382 | 0.0011 | 0.0363 | 0.0045 |
| Evo 2 (1B) | 1.0273 | 0.0401 | 0.0001 | 0.0402 | 0.0038 |
| HyenaDNA (medium) | 0.0153 | 0.0006 | 0.0001 | 0.0040 | 0.0023 |

`free_running` = the model continues its own generated bases.
`teacher_forced` = the model is scored on the true CDS continuation.
A model that is periodic only when teacher-forced is reading codon
structure out of the given context, not generating it.

## Per-base probability range

| model | condition | A_range | C_range | G_range | T_range |
| --- | --- | --- | --- | --- | --- |
| Evo 1 (8k) | native | 0.1626 | 0.1164 | 0.1772 | 0.2119 |
| Evo 1 (8k) | shuffled | 0.0450 | 0.0440 | 0.0335 | 0.0551 |
| Evo 2 (1B) | native | 0.1881 | 0.0915 | 0.2103 | 0.1831 |
| Evo 2 (1B) | shuffled | 0.0026 | 0.0025 | 0.0028 | 0.0021 |
| HyenaDNA (medium) | native | 0.0097 | 0.0108 | 0.0166 | 0.0080 |
| HyenaDNA (medium) | shuffled | 0.0060 | 0.0035 | 0.0066 | 0.0045 |

