"""Tests for brand-masking YAML and config validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from experiments.brand_masking.config_models import (
    load_brand_masking_config,
    load_company_descriptions,
)


def test_load_company_descriptions_validates_catalog(tmp_path: Path) -> None:
    """Company descriptions YAML should validate into the expected model."""
    yaml_path = tmp_path / "company_descriptions.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "companies:",
                "  - company_id: pfizer",
                "    incumbent_name: Pfizer",
                "    masked_name: Company X",
                '    description: ""',
            ],
        ),
    )

    catalog = load_company_descriptions(yaml_path)

    assert len(catalog.companies) == 1
    assert catalog.companies[0].company_id == "pfizer"
    assert catalog.companies[0].masked_name == "Company X"


def test_load_company_descriptions_requires_company_id(tmp_path: Path) -> None:
    """Missing required company fields should raise Pydantic validation errors."""
    yaml_path = tmp_path / "company_descriptions.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "companies:",
                "  - incumbent_name: Pfizer",
                "    masked_name: Company X",
                '    description: ""',
            ],
        ),
    )

    with pytest.raises(ValidationError):
        load_company_descriptions(yaml_path)


def test_load_brand_masking_config_validates_runtime_settings(tmp_path: Path) -> None:
    """Runtime config YAML should validate and expose helper path resolution."""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "runs: 2",
                "max_workers: 4",
                "min_pairs_for_analysis: 5",
                "dataset_path: experiments/brand_masking/config/scenarios.csv",
                "company_descriptions_path: "
                "experiments/brand_masking/config/company_descriptions.yaml",
                "prompts_path: experiments/brand_masking/config/prompts.yaml",
                "default_masked_name: Company X",
                "llm_models:",
                "  - model_id: GPT_5_MINI",
                "    model_type: openai",
                "    reasoning_model: false",
                "    temperature: 0.0",
            ],
        ),
    )

    config = load_brand_masking_config(yaml_path)

    assert config.runs == 2
    assert config.max_workers == 4
    assert config.llm_models[0].model_id == "GPT_5_MINI"
