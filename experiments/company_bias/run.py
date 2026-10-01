"""CLI entry point for the *company-bias* experiment."""

from bias_in_llms.config.project_paths import EXPERIMENTS_DIR
from experiments.company_bias.experiment import run_full_company_bias_experiment


def main():
    """Run the company-bias experiment."""
    config_path = EXPERIMENTS_DIR / "company_bias" / "config" / "config.yaml"

    run_full_company_bias_experiment(
        config_path=config_path,
        use_recommended_models=False,  # Set to True to use recommended models
    )


if __name__ == "__main__":
    main()
