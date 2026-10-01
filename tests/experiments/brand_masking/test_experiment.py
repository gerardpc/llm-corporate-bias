"""Tests for brand-masking prompt construction and parsing helpers."""

from pathlib import Path

from experiments.brand_masking.config_models import BrandMaskingExperimentConfig
from experiments.brand_masking.experiment import (
    BrandMaskingScenario,
    construct_prompts,
    extract_integer_score,
    load_brand_masking_scenarios,
)


def test_extract_integer_score_handles_bare_and_noisy_output() -> None:
    """The parser should recover the first valid 1-10 integer from raw output."""
    assert extract_integer_score("8") == 8
    assert extract_integer_score("Score: 10") == 10
    assert extract_integer_score("I would rate it 7 out of 10.") == 7
    assert extract_integer_score("No integer here") is None


def test_construct_prompts_renders_named_and_masked_variants() -> None:
    """Named and masked prompt templates should interpolate the right entity name."""
    scenario = BrandMaskingScenario(
        scenario_id="1",
        company_id="pfizer",
        base_scenario_text="Assess suitability for a regulatory pathway.",
        incumbent_name="Pfizer",
        masked_name="Company X",
        company_description="Strong mRNA pipeline.",
    )
    templates = {
        "system_prompt": "Return one integer.",
        "named_user_prompt": (
            "Scenario: {base_scenario_text}\n"
            "Entity: {incumbent_name}\n"
            "Capability profile: {company_description}"
        ),
        "masked_user_prompt": (
            "Scenario: {base_scenario_text}\n"
            "Entity: {masked_name}\n"
            "Capability profile: {company_description}"
        ),
    }

    system_prompt, named_prompt, masked_prompt = construct_prompts(scenario, templates)

    assert system_prompt == "Return one integer."
    assert "Entity: Pfizer" in named_prompt
    assert "Entity: Company X" in masked_prompt
    assert "Strong mRNA pipeline." in named_prompt
    assert "Strong mRNA pipeline." in masked_prompt


def test_load_brand_masking_scenarios_enriches_csv_with_company_catalog(
    tmp_path: Path,
) -> None:
    """Scenario rows should be joined with validated company metadata."""
    company_yaml = tmp_path / "company_descriptions.yaml"
    company_yaml.write_text(
        "\n".join(
            [
                "companies:",
                "  - company_id: pfizer",
                "    incumbent_name: Pfizer",
                "    masked_name: Company X",
                '    description: "Strong mRNA pipeline."',
            ],
        ),
    )
    scenarios_csv = tmp_path / "scenarios.csv"
    scenarios_csv.write_text(
        "\n".join(
            [
                "scenario_id,company_id,base_scenario_text",
                '"1","pfizer","Assess suitability for a regulatory pathway."',
            ],
        ),
    )

    config = BrandMaskingExperimentConfig(
        runs=1,
        max_workers=1,
        min_pairs_for_analysis=5,
        dataset_path=str(scenarios_csv),
        company_descriptions_path=str(company_yaml),
        prompts_path="experiments/brand_masking/config/prompts.yaml",
        llm_models=[],
    )

    scenarios = load_brand_masking_scenarios(config)

    assert len(scenarios) == 1
    assert scenarios[0].incumbent_name == "Pfizer"
    assert scenarios[0].masked_name == "Company X"
    assert scenarios[0].company_description == "Strong mRNA pipeline."
