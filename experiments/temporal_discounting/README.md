# Temporal Discounting Experiment

This directory contains everything required to reproduce the *temporal discounting* study – i.e. asking an LLM whether it prefers a smaller amount of money **now** or a larger amount **later**.

## File overview

| File / Folder | Purpose |
|---------------|---------|
| `config/` | YAML configuration for prompts (`prompts.yaml`) and the experimental grid (`config.yaml`). |
| `pair_generation.py` | Pure utility that creates the A/B monetary choice strings. No LangChain dependency, so you can unit-test it on its own. |
| `experiment.py` | Library function `run_experiment` that executes the LLM calls and writes the results to the SQLite database. *No* top-level side effects. |
| `run.py` | Thin CLI wrapper that loads the config, initialises the DB, builds the option pairs, instantiates each LLM, and delegates to `run_experiment`. |


## Configuration

* **Prompt templates** – edit `config/prompts.yaml`.
* **Experimental grid** – amounts, rates, durations in `config/config.yaml`.
* **Models** – list of `model_id`, `model_type`, `reasoning_model`, and `temperature` in the same config file.
