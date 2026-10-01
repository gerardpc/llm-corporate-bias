"""
CLI entry point for the temporal-discounting experiment.

Usage
-----
$ python -m experiments.temporal_discounting.run

This script:
1. Initializes the SQLite database if needed.
2. Loads the experiment grid and model list from config/config.yaml.
3. Generates A/B choice pairs.
4. Iterates over the specified number of runs.
5. For each run, checks if combinations already exist in the database.
6. Instantiates each LLM from config.yaml and calls
   experiments.temporal_discounting.experiment.run_experiment.
7. Saves per-model CSV files with the raw choices.

Heavy lifting (LLM calls, DB writes) is in experiment.py.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from bias_in_llms.config.project_paths import DATA_DIR, EXPERIMENTS_DIR
from bias_in_llms.database import initialize_temporal_discounting_db, query_db
from bias_in_llms.database.temporal_discounting_db import (
    combination_exists,
    get_existing_combination,
)
from bias_in_llms.llm.llm_calling import get_llm
from bias_in_llms.llm.prompt.prompt_loading import load_prompt_templates
from experiments.temporal_discounting.experiment import run_experiment
from experiments.temporal_discounting.pair_generation import generate_discounting_pairs

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH = EXPERIMENTS_DIR / "temporal_discounting" / "config" / "config.yaml"
PROMPT_PATH = EXPERIMENTS_DIR / "temporal_discounting" / "config" / "prompts.yaml"
DB_PATH = DATA_DIR / "temporal_discounting_results.db"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_existing_combinations_cache(models: list[dict], run_number: int) -> dict:
    """
    Load all existing combinations for the given models and run number into a cache.

    Returns:
        dict: Cache with key (model_id, now_value, later_value, delay_years,
              temperature) -> result_data
    """
    cache = {}

    for model in models:
        model_id = model["model_id"]
        temperature = model.get("temperature", 0.0)

        query = """
            SELECT now_value, later_value, delay_years, preference, answer, question,
                   full_prompt, reasoning_tokens
            FROM bias_in_llms.temporal_discounting
            WHERE model_id = %s AND run = %s AND temperature = %s
        """

        df = query_db(query, (model_id, run_number, temperature))

        for row in df.to_dict("records"):
            key = (
                model_id,
                row["now_value"],
                row["later_value"],
                row["delay_years"],
                temperature,
            )
            cache[key] = {
                "preference": row["preference"],
                "answer": row["answer"],
                "question": row["question"],
                "full_prompt": row["full_prompt"],
                "reasoning_tokens": row["reasoning_tokens"],
            }

    print(
        f"📊 Loaded {len(cache)} existing combinations into cache for run {run_number}",
    )
    return cache


def _load_config(path: Path) -> dict:
    """
    Load YAML config and perform basic validation.

    Args:
        path (Path): Path to the YAML config file.

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        KeyError: If required keys are missing in the config.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)

    required_keys: list[str] = ["llm_models", "amounts", "rates", "durations", "runs"]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise KeyError(f"Config file missing required keys: {', '.join(missing)}")

    return cfg


def _process_model_run(
    model: dict,
    run_number: int,
    option_pairs: list,
    prompt_path: Path,
    trace_tags: list[str],
    cache: dict | None = None,
) -> pd.DataFrame:
    """
    Process a single model for a single run.

    Returns:
        pd.DataFrame: Results for this model run.
    """
    model_id = model["model_id"]
    model_type = model["model_type"]
    temperature = model.get("temperature", 0.0)

    print(f"▶ Processing {model_id} (Run {run_number})")

    llm = get_llm(
        model_id,
        model_type,
        temperature=temperature,
    )

    results_data = []
    new_experiments = 0
    existing_count = 0
    total_combinations = len(option_pairs)

    for idx, (option_a, option_b, now_val, later_val, delay_years) in enumerate(
        option_pairs,
        1,
    ):
        print(f"Processing combination {idx}/{total_combinations} for {model_id}")

        # Check cache if provided (fast), otherwise use database queries (slow)
        existing_data = None
        if cache is not None:
            cache_key = (model_id, now_val, later_val, delay_years, temperature)
            existing_data = cache.get(cache_key)
        else:
            # Fallback to database queries for backward compatibility
            if combination_exists(
                model_id,
                run_number,
                now_val,
                later_val,
                delay_years,
                temperature,
            ):
                existing_data = get_existing_combination(
                    model_id,
                    run_number,
                    now_val,
                    later_val,
                    delay_years,
                    temperature,
                )

        if existing_data:
            results_data.append(
                {
                    "Run": run_number,
                    "Now ($)": now_val,
                    "Later ($)": later_val,
                    "Delay (years)": delay_years,
                    "LLM": model_id,
                    "Temperature": temperature,
                    "Preference": existing_data["preference"],
                },
            )
            existing_count += 1
        else:
            # Run experiment for this specific combination
            single_pair_results = run_experiment(
                llm=llm,
                option_pairs=[(option_a, option_b, now_val, later_val, delay_years)],
                prompt_path=prompt_path,
                model_id=model_id,
                run=run_number,
                reasoning_model=model.get("reasoning_model", False),
                temperature=temperature,
            )
            # Add the result to our results_data
            if not single_pair_results.empty:
                results_data.extend(single_pair_results.to_dict("records"))
            new_experiments += 1

    print(
        f"    ✓ Processed {new_experiments} new experiments, {existing_count} from DB",
    )
    return pd.DataFrame(results_data)


def _save_results(results: pd.DataFrame, model_id: str, run_number: int) -> None:
    """Save results to CSV with timestamped directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = DATA_DIR / "temporal_discounting" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"{model_id}_run_{run_number}_results.csv"
    results.to_csv(out_csv, index=False)
    print(f"    ✓ Results saved to {out_csv}")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Main entry point for running the temporal-discounting experiment.

    Loads config, initializes DB, generates option pairs, runs each model
    across multiple runs, and saves results to CSV.
    """
    cfg = _load_config(CONFIG_PATH)

    # Step 1 – ensure DB exists
    initialize_temporal_discounting_db()

    # Step 2 – build choice pairs
    prompt_templates = load_prompt_templates(str(PROMPT_PATH))
    option_pairs = generate_discounting_pairs(
        prompt_templates["gain_immediate_option_template"],
        prompt_templates["gain_delayed_option_template"],
        amounts=cfg["amounts"],
        rates=[r / 100 for r in cfg["rates"]],
        durations=cfg["durations"],
    )

    # Step 3 – run each model across multiple runs
    trace_tags = ["genai", "temporal_discounting_experiment"]
    num_runs = cfg["runs"]

    for run_number in range(1, num_runs + 1):
        print(f"\n{'=' * 60}")
        print(f"Starting Run {run_number}/{num_runs}")
        print(f"{'=' * 60}")

        # Load existing combinations cache for performance
        print("🗂️  Loading existing combinations cache...")
        combinations_cache = _load_existing_combinations_cache(
            cfg["llm_models"],
            run_number,
        )

        for model in cfg["llm_models"]:
            results = _process_model_run(
                model,
                run_number,
                option_pairs,
                PROMPT_PATH,
                trace_tags,
                cache=combinations_cache,
            )
            _save_results(results, model["model_id"], run_number)

    print(f"\n{'=' * 60}")
    print("All runs completed!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
