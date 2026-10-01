# Brand-masking analysis report (SaTML manuscript, Experiment 2)

**Estimand:** Δ = score_named − score_masked. Named is the Masked prompt plus the real identity (only the `Entity:` line differs).
**Main result:** conditional on the supplied profile, identity disclosure changes suitability scores.
Earlier numbers (manuscript drafts, AR-215 comments) are provisional; the numbers here replace them.

## Reproduce
```
cd aily-satml
PYTHONPATH=. <venv>/bin/python -m experiments.brand_masking.manuscript_analysis.run_analysis           # from DB
PYTHONPATH=. <venv>/bin/python -m experiments.brand_masking.manuscript_analysis.run_analysis --offline # from parquet snapshot (data/, gitignored)
pytest tests/experiments/brand_masking/test_manuscript_*.py   # 11 tests
```
Parameters are in `outputs/analysis_parameters.json`: bootstrap seed 20260927, 10,000 resamples, top-k = 3, T = 0.

## Design choices
- There are two study windows, **May core** and **September sweep**. They are never pooled, and legacy tables stay separate.
- Duplicate trial keys are resolved by keeping the latest `created_at`. No rows were deleted or remapped.
- Group status:
  - `complete`.
  - `incomplete_minor` (§): full catalog, ≤1% of trials missing. These are reported.
  - `incomplete` (†): not estimated.
- Primary CI: percentile cluster bootstrap over companies. Sensitivity CI: bootstrap over (scenario, company) clusters.
- Wilcoxon (on run-averaged scenario×company deltas) is descriptive only, because every scenario reuses the same company profile.
- Rankings are computed per (sector, model, run):
  - Rank change uses average ranks.
  - Top-k uses competition ranks, with ties at the cut included.
  - Spearman is primary; Kendall τ_b is the sensitivity check.

## Verified existing results
- **Prompt equivalence holds for all 257,826 rows in both windows.** With the Entity line neutralised, the prompts are identical. There is one system prompt, and a template re-render of 200 sampled rows per window/sector showed 0 failures (`outputs/prompt_equivalence.json`, `all_passed: true`).
- **The May per-group Δ values in the old manuscript table reproduce**, as do the company extremes cited in the text (Sanofi +0.07 → AbbVie +1.05; Accenture +0.26 → Bain +1.19; IBM Cloud −0.06 → Google Cloud +0.69; ICBC +0.08 → UBS +1.28).
- **Two old Wilcoxon W values were stale** (pharma Sonnet 4.5 → 5,864.5; pharma GPT-5 → 54,170.5). Both are replaced in `tables/bm_scores_may.tex`.
- **The old claim "p<0.001 in 15/16" is superseded.** Under clustered CIs, the effect is clearly nonzero in 14 of 16 May groups.

## Newly computed results
Recomputed offline from `data/brand_masking/manuscript_snapshot/` on 2026-09-27 (same driver as DB mode). Tests: 11 passed.

- **May core:** Δ is positive with a company CI excluding 0 in 14 of 16 groups, ranging from +0.14 (GPT-5 banking) to +0.72 (Claude Sonnet 4.5 pharma).
  - Cloud GPT-5: −0.03 [−0.16, +0.08], not significant.
  - Cloud GPT-5 Mini: +0.10 [−0.11, +0.35]. The Wilcoxon p is below 0.001, but the clustered CI includes 0.
- **September sweep (6 complete / incomplete-minor models):**
  - Consulting: +0.27 to +0.38, all CIs exclude 0.
  - Banking: +0.41 to +0.94, all CIs exclude 0.
  - Cloud: 4 of 6 are positive. Qwen3.8 Max (+0.10) and DeepSeek V4.1 Flash (−0.06) have CIs spanning 0.
- **Rankings:**
  - Spearman 0.78–0.99 and τ_b 0.60–0.98 (May); September complete groups 0.92–0.97 / 0.80–0.91.
  - Strict pair reversals: 1–19% in May, 4–10% in September.
  - May pharma changes most: mean |Δr| 1.2–2.3; top-1 changed in 50–100% of runs; top-3 Jaccard as low as 0.17 (Haiku).
  - Elsewhere the top-1 is mostly stable, but top-3 membership varies. September banking top-3 Jaccard 0.22–0.65.
- **Ceiling:** at most 7% of masked scores equal 10 (max observed 6.6%).

Outputs:
- CSV/JSON results are in `outputs/`.
- Tables are `manuscript/satml_manuscript/tables/bm_*.tex`.
- Figures are `fig_4.pdf`, `fig_5.pdf`, `fig_bm_ranks.pdf` and `fig_bm_sep_models.pdf`.

## Incomplete (not estimated)
- September Claude Sonnet 4.5 and GPT-5 (all sectors) only cover the new-company subset. GPT-5 also has 35–50% parse loss.
- September pharma is incomplete for every model.

## Manuscript changes
Relative to the committed draft on `feat/satml-manuscript`, `satml_manuscript.tex` was updated to:

- Soften abstract, contributions, conclusion, and Fig. 1 caption: Named = Masked + real identity; estimand is identity disclosure conditional on the profile; masking is an audit/diagnostic, not a proven mitigation; no "objectively better companies" or rank "improvements."
- Rewrite Experiment 2 methods: 1–10 integer scale, two study windows, cluster bootstrap (company primary; scenario×company sensitivity), descriptive Wilcoxon, per-(sector, model, run) rankings with average / competition ranks, Spearman primary / Kendall τ_b sensitivity, T = 0.
- Replace provisional brand-masking results with May/September tables (`\input{tables/bm_*.tex}`), refreshed figures/captions (`fig_4`, `fig_5`, `fig_bm_ranks`, plus `fig_bm_sep_models` for the September score panel), and discussion/limitations numbers aligned with the recomputed outputs.
- Appendices: both prompt templates plus equivalence audit (257,826 rows) and reproducibility path for `manuscript_analysis/`.

## Unresolved limitations
- Each company has one LLM-assisted profile, so company-level Δ confounds identity with profile wording.
- There are only 10–15 independent profiles per sector.
- Because T = 0, runs are near-replicates. Run-level CIs therefore reflect residual response noise only.
- Masking is an audit design here. It has not been tested as a mitigation.
- Rank changes are reorderings, not improvements.
- September pharma, and September Claude Sonnet 4.5 / GPT-5 groups, remain incomplete and unreported; GPT-5 September parse loss is substantial.
- May and September catalogs/model sets differ; windows must stay separate and are not directly comparable.
