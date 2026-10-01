# Cross-Experiment Correlation Report

## Objective

This analysis asks whether companies that receive more paired wins in **Experiment 1** also receive a larger name-related score increase in **Experiment 2**.

For each sector, we plot:

- **x-axis:** each company's mean number of paired wins in Experiment 1, averaged across the six models;
- **y-axis:** each company's mean `Named - Masked` score delta in Experiment 2, averaged across the available models and runs.

The analysis is exploratory: it tests whether the two experiments appear to measure a related company-level preference signal. It does not establish causality.

## Data sources

### Experiment 1

The following files contain company-level preference counts by model:

- `pharma_preference_counts.csv`
- `cloud_provider_preference_counts.csv`
- `consulting_preference_counts.csv`
- `banking_preference_counts.csv`

For each company, the x-axis value is the arithmetic mean of the six model columns. The `total_preferences` column was not used because it is already the sum across models.

### Experiment 2

The requested `ranking_summary_postgres.csv` contains rank-change summaries, but it does not contain company-level Named-versus-Masked score deltas. Therefore, the y-axis uses:

`experiments/brand_masking/manuscript_analysis/outputs/company_effects.csv`

Specifically, `mean_delta` was averaged within each sector and company across the available model/window rows.

Company names were matched after lowercasing and removing punctuation and whitespace. This handles minor formatting differences such as `Merck & Co` versus `Merck & Co.`.

## Results

| Sector | Companies matched | Pearson correlation | Spearman correlation | Linear-model R² |
|---|---:|---:|---:|---:|
| Pharma | 15 | -0.05 | -0.00 | 0.00 |
| Cloud | 12 | 0.77 | 0.76 | 0.59 |
| Consulting | 15 | 0.39 | 0.41 | 0.15 |
| Banking | 15 | 0.16 | 0.16 | 0.03 |

The corresponding scatter plots and fitted least-squares lines are in:

`experiment1_vs_experiment2_correlations.svg`

The numerical summary is in:

`experiment1_vs_experiment2_summary.csv`

## Interpretation by sector

### Pharma

There is effectively no relationship between Experiment-1 paired wins and the Experiment-2 brand premium. The Pearson correlation is approximately -0.05 and the linear model explains essentially none of the company-level variation (`R² ≈ 0.00`).

This suggests that the companies preferred in pairwise suitability decisions are not simply the companies receiving the largest additional score from explicit naming. Pairwise selection and name-conditioned scoring may be responding to different aspects of the profiles or scenarios.

### Cloud

Cloud shows the clearest cross-experiment relationship. The correlation is positive and strong (`r ≈ 0.77`; Spearman `ρ ≈ 0.76`), and a one-predictor linear model explains approximately 59% of the observed company-level delta variation.

This is consistent with the possibility that Experiment 1's pairwise preference ordering and Experiment 2's name-related premium capture a shared familiarity or incumbent-recognition signal in this sector. The result should still be treated as exploratory because the number of companies is small and the observations are not independent at the underlying response level.

### Consulting

Consulting shows a positive but modest relationship (`r ≈ 0.39`; Spearman `ρ ≈ 0.41`; `R² ≈ 0.15`). The direction is compatible with a shared signal, but the explanatory power is limited. Most variation in the Experiment-2 deltas remains unexplained by Experiment-1 paired wins alone.

### Banking

Banking shows only a weak positive relationship (`r ≈ 0.16`; Spearman `ρ ≈ 0.16`; `R² ≈ 0.03`). This is too small to support a useful one-variable predictive model. Other factors—such as regional representation, company-specific profile content, or scenario composition—may dominate in this sector.

## Overall conclusion

The two experiments do not produce one universal company-level ranking signal. The relationship is strongly sector-dependent:

1. **Cloud:** substantial alignment between pairwise wins and brand-premium deltas.
2. **Consulting:** weak-to-moderate alignment.
3. **Banking:** little alignment.
4. **Pharma:** no detectable alignment in this exploratory analysis.

Therefore, the most defensible interpretation is not that Experiment 1 mechanically predicts Experiment 2 everywhere. Instead, the results suggest that the two measures may overlap in some sectors—especially cloud—but reflect partially distinct mechanisms in others.

## Recommended next analysis

For a stronger paper analysis, fit a pooled model with sector interactions rather than a single global slope:

\[
\Delta_{s,c} = \alpha_s + \beta_s W_{s,c} + \epsilon_{s,c},
\]

where \(W_{s,c}\) is mean paired wins, \(\Delta_{s,c}\) is the mean brand premium, and both the intercept and slope vary by sector. Report confidence intervals using company-level bootstrap resampling, not response-level resampling.

The next useful extension is to add publicly available company variables—such as market capitalization, revenue, geographic headquarters, and sector market share—and test whether they explain the residual delta after accounting for Experiment-1 wins.

## Reproducibility

The analysis used the following operations:

1. Read each sector's Experiment-1 preference-count CSV.
2. Average the six model-specific preference columns for each company.
3. Read `company_effects.csv` and average `mean_delta` within sector/company.
4. Normalize company names by lowercasing and removing non-alphanumeric characters.
5. Inner-join the two experiment summaries by sector and normalized company name.
6. Compute Pearson correlation, rank-based Spearman correlation, and ordinary least-squares `R²` separately by sector.
7. Plot the matched company points and the sector-specific least-squares line.

The generated numerical summary is available in `experiment1_vs_experiment2_summary.csv`, and the plotted figure is available in `experiment1_vs_experiment2_correlations.svg`.

## Limitations

- The number of companies per sector is small (`n = 12–15`).
- Experiment-1 counts and Experiment-2 deltas are aggregated summaries, so uncertainty at the individual-response level is not represented in these correlations.
- The analysis is correlational and cannot determine why a relationship is stronger in cloud than in pharma or banking.
- Because `ranking_summary_postgres.csv` does not contain company-level score deltas, the Experiment-2 outcome was taken from `company_effects.csv`.
