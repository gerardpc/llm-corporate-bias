"""CLI entry point for the brand-masking experiment."""

from bias_in_llms.config.project_paths import EXPERIMENTS_DIR
from experiments.brand_masking.analyze_results import summarize_brand_masking_trials
from experiments.brand_masking.config_models import load_brand_masking_config
from experiments.brand_masking.experiment import run_full_brand_masking_experiment

SECTOR = "pharma"  # change to "consulting" or "cloud" as needed


def main() -> None:
    """Run raw paired trials and then compute Wilcoxon summaries."""
    config_path = EXPERIMENTS_DIR / "brand_masking" / "config" / SECTOR / "config.yaml"

    config = load_brand_masking_config(config_path)
    run_full_brand_masking_experiment(config_path)
    summarize_brand_masking_trials(config.min_pairs_for_analysis, SECTOR)


if __name__ == "__main__":
    main()
