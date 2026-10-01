# bias-in-llms

Code and experiment configuration for the study
**"Does AI Favor the Famous? A Study on Corporate Bias and Market Concentration in LLMs"**.

The repository runs and analyzes large-scale audits of *latent corporate bias* in
frontier LLMs across four sectors (pharmaceuticals, cloud computing, consulting,
and banking). It covers the two experiments reported in the paper:

- **Pairwise company comparison** — forced binary choices between corporate
  entities in domain-specific decision-support scenarios, with mirrored
  presentation order to control for positional bias.
- **Brand masking** — paired *named* vs. *masked* evaluation of identical
  capability profiles to isolate the "brand premium" via a Wilcoxon signed-rank
  test.

## Experiments

Each experiment is self-contained, with its own config, runner, and README:

| Experiment | Path |
|------------|------|
| Pairwise company bias | [`experiments/company_bias`](experiments/company_bias) |
| Brand masking | [`experiments/brand_masking`](experiments/brand_masking) |
| Temporal discounting | [`experiments/temporal_discounting`](experiments/temporal_discounting) |

Shared utilities (LLM invocation, database/parquet backends, config loading)
live under the `bias_in_llms` package.

## Setup

Install dependencies with a standard Python package index:

```bash
uv sync
```

Use `env.template` as the reference for provider keys/database credentials.
OpenAI and OpenRouter use API keys; Bedrock uses the standard AWS credential
chain.

Local `.env` files are ignored by git and loaded automatically by the Python
entry points. Keep private or organization-specific values there, not in
committed config files.

Check local environment readiness without printing secret values:

```bash
.venv/bin/python -m bias_in_llms.config.environment
```

## Export Data

Export experiment tables from PostgreSQL to local parquet files:

```bash
python -m bias_in_llms.database.export_parquet
```

By default this writes known experiment tables to `data/parquet/bias_in_llms/`
and skips tables that do not exist. To export every table in the schema:

```bash
python -m bias_in_llms.database.export_parquet --discover
```

Configure database access in `.env` with `DATABASE_URL` or the `POSTGRES_*`
variables from `env.template`.

## Run Visualizations From Parquet

To run `data_visualization` scripts without a live PostgreSQL connection, set
the parquet backend in `.env`:

```bash
BIAS_DB_BACKEND=parquet
PARQUET_DATA_DIR=data/parquet
```

Then run the visualization as usual, for example:

```bash
python experiments/company_bias/data_visualization.py
```

## Run Experiments Without PostgreSQL

You can run experiments against a local writable DuckDB file:

```bash
BIAS_DB_BACKEND=duckdb
DUCKDB_PATH=data/local_experiments.duckdb
```

Then run the experiment entry point as usual:

```bash
python experiments/company_bias/run.py
```

## Paper

This repository accompanies the paper *"Does AI Favor the Famous? A Study on
Corporate Bias and Market Concentration in LLMs"*. The paper is currently under
peer review; a link to the published version will be added here once available.

The four models evaluated in the paper are Claude Haiku 4.5, Claude Sonnet 4.5,
GPT-5, and GPT-5 Mini, queried at temperature 0.

## Citation

If you find this useful in your research, please consider citing:

```bibtex
@misc{corporatebias2026,
  title  = {Does AI Favor the Famous? A Study on Corporate Bias and Market
            Concentration in LLMs},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```

> The author and venue fields will be updated once the paper is de-anonymized.

## License

To be determined before public release. Add a `LICENSE` file (e.g. MIT or
Apache-2.0) and update this section.
