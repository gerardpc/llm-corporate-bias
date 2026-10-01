# Brand Masking Experiment

## Overview

This experiment measures **latent corporate incumbency bias** in LLMs using a
paired named-vs-masked design.

For each scenario, the model evaluates the same capability profile twice:

- **Named condition**: the prompt includes the real incumbent name.
- **Masked condition**: the prompt replaces that name with a neutral placeholder
  such as `Company X`.

The core question is whether the model systematically gives a higher score to
the named incumbent than to the exact same masked profile. If it does, the gap
can be interpreted as a **brand premium** that is not justified by any change in
the underlying capability description.

## Experimental Design

Each row in the scenario dataset defines one paired trial:

1. Load a scenario with a `scenario_id`, `company_id`, and
   `base_scenario_text`.
2. Join it with the corresponding company entry from
   `config/company_descriptions.yaml`.
3. Build two prompts:
   - one with `incumbent_name`
   - one with `masked_name`
4. Ask the same model to score both conditions on a `1` to `10` scale.
5. Parse the raw outputs into integers.
6. Persist the paired raw result to the database.
7. Run post-hoc statistical analysis over the valid paired scores.

The implementation uses a strict system prompt that tells the model to respond
with a single integer only. A regex fallback parser still extracts the score if
the model returns text such as `Score: 8`.

## Statistical Method

This experiment uses a **Wilcoxon signed-rank test** rather than a paired
t-test.

Why:

- LLM score outputs often cluster near the top of the scale.
- The response distribution is usually not normal.
- The Wilcoxon test is a non-parametric paired test, which makes it a better
  fit for these ordinal-like, skewed outputs.

For each `(run, model_id)` group, the analysis computes:

- `mean_score_named`
- `mean_score_masked`
- `mean_difference`
- `wilcoxon_statistic`
- `p_value`
- `valid_pairs_n`
- `dropped_pairs_n`
- `zero_difference_pairs_n`

Interpretation:

- `mean_difference > 0`: named firms were rated higher on average than masked
  versions of the same profiles.
- `p_value < 0.05`: evidence that the named and masked score distributions
  differ beyond random variation.

In this experiment, the mean difference is the operational estimate of the
model's **brand premium**.

## Folder Structure

| Path | Purpose |
|------|---------|
| `config/config.yaml` | Runtime configuration: models, runs, worker count, minimum `N`, and file paths. |
| `config/prompts.yaml` | Strict system prompt and the named/masked user prompt templates. |
| `config/company_descriptions.yaml` | YAML catalog of companies and the capability descriptions used in both conditions. |
| `config/scenarios.csv` | Tabular scenario dataset. Each row references a `company_id`. |
| `config_models.py` | Pydantic models and loaders for runtime config and the company descriptions YAML. |
| `experiment.py` | Main execution logic: load config, render prompts, call the LLM twice per scenario, parse scores, and write raw paired trials. |
| `analyze_results.py` | Wilcoxon analysis and summary persistence. |
| `run.py` | CLI-style entry point that runs raw collection first and summary analysis second. |

Related non-folder components:

- `bias_in_llms/database/brand_masking_db.py`: raw and summary DB helpers
  for this experiment.
- `bias_in_llms/llm/llm_calling.py`: shared LLM invocation wrapper.

## Data Files

### `config/company_descriptions.yaml`

This file is the source of truth for company metadata used by the experiment.
It must validate against the Pydantic schema in `config_models.py`.

Current shape:

```yaml
companies:
  - company_id: pfizer
    incumbent_name: Pfizer
    masked_name: Company X
    description: |
      Draft capability profile...
```

Guidelines for writing the descriptions:

- Keep them factual and capability-based.
- Avoid promotional adjectives unless they are directly supported by evidence.
- Keep the named and masked conditions identical except for the entity label.
- Prefer stable organization-level capabilities over scenario-specific wording.

### `config/scenarios.csv`

Each scenario row references a company by `company_id` and provides the base
task that the model must rate. The experiment combines the scenario text with
the company description to produce both prompt variants.

## Database Outputs

The experiment persists results in two tables under the `bias_in_llms`
schema:

- `brand_masking`: raw paired named/masked trial rows
- `brand_masking_summary`: Wilcoxon summaries by run and model

The raw table stores the prompt payloads, raw model outputs, parsed integer
scores, parse-success flags, and per-pair deltas. The summary table stores the
aggregate statistical outputs.

## How To Run

### 1. Prepare the inputs

Update:

- `config/company_descriptions.yaml`
- `config/scenarios.csv`
- `config/config.yaml`

### 2. Run the full experiment

```bash
uv run python -m experiments.brand_masking.run
```

This command:

1. loads the runtime config
2. initializes the database tables
3. runs the named and masked prompts for every `(scenario, model, run)`
4. saves per-run CSV exports under `data/brand_masking/`
5. computes and persists Wilcoxon summaries

### 3. Re-run analysis only

If raw rows already exist and you only want to recompute the summaries:

```bash
uv run python -m experiments.brand_masking.analyze_results
```

## Practical Notes

- The current implementation uses the repository's synchronous/threaded pattern,
  similar to the existing experiment modules.
- Existing rows are cached per run to avoid repeating the same model/scenario
  combination unnecessarily.
- Parse failures are kept as raw text in the DB so they can be audited later.
- The YAML currently contains **draft** company profiles. Replace them with the
  final evidence-backed descriptions before running a production study.

## Suggested Workflow

1. Finalize `company_descriptions.yaml` with neutral capability profiles.
2. Expand `scenarios.csv` to the full evaluation set.
3. Confirm the target models and run counts in `config/config.yaml`.
4. Execute `uv run python -m experiments.brand_masking.run`.
5. Inspect the raw CSVs and the summary table for `Δ`, `W`, and `p_value`.
