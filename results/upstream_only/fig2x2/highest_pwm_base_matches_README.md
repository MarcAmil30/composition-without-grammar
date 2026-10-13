# Highest-PWM base matching

For each generated sequence, scan every six-base window with the same PWM used
for placement and retain the highest-scoring window. Count how many of its six
bases match that gene's annotated proximal box or Shine--Dalgarno core.

The control first applies a deterministic, composition-preserving shuffle to
the complete generated sequence and then repeats the same PWM scan and base
comparison. Scores are averaged across generations within each gene and then
across genes. Confidence intervals are 2,000 gene-level bootstrap replicates.
The proximal-box analysis uses the 487 sigma70 genes (459 for PromoGen2); the SD
analysis uses all 681 genes.

The positive control is the identical scan run on each gene's *real*
continuation. It scores 4.99/6 for the proximal box (95% CI 4.85-5.12) and
5.01/6 for the Shine--Dalgarno (95% CI 4.88-5.15), not 6/6, because the scan
prefers a window other than the annotated element in 167 of 487 sigma70
promoters and 177 of 681 SD regions. Median in both cases is 6/6. These are the
reference lines in Fig. 2c and Fig. A.1.

Files:

- `highest_pwm_base_matches.csv`: complete reporting table.
- `highest_pwm_positive_control.csv`: the real-sequence ceiling for the metric.
- `highest_pwm_base_matches.json`: identical results in JSON form.
- `../../../code/figure_2_promoter_element_placement/highest_pwm_base_matches.py`:
  analysis code.
- `../../../code/figure_2_promoter_element_placement/highest_pwm_positive_control.py`:
  positive control; runs from `data/prompts_681_leaked.csv` and `data/pwms/`.

Example cluster command:

```bash
python highest_pwm_base_matches.py \
  --bench-dir /path/to/data/benchmark/upstream_only \
  --output highest_pwm_base_matches.csv
```
