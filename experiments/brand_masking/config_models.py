"""Configuration models and loaders for the brand-masking experiment."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from bias_in_llms.config.project_paths import ROOT_DIR


class BrandMaskingModelConfig(BaseModel):
    """Runtime configuration for one LLM used in the experiment."""

    model_id: str = Field(
        description="Configured LLM model alias or provider model ID.",
    )
    model_type: str = Field(
        description="Model provider name (openai, bedrock, or openrouter).",
    )
    reasoning_model: bool = Field(
        default=False,
        description="Whether the configured model is a reasoning model.",
    )
    temperature: float = Field(
        default=0.0,
        description="Sampling temperature used when invoking the model.",
    )


class BrandMaskingExperimentConfig(BaseModel):
    """Top-level runtime configuration for the brand-masking experiment."""

    sector: str = Field(
        default="pharma",
        description="Industry sector for this experiment run (e.g. pharma, consulting, "
        "cloud).",
    )
    runs: int = Field(
        default=1,
        description="Number of full experiment repetitions to execute.",
    )
    max_workers: int = Field(
        default=5,
        description="Maximum worker threads used for concurrent trial execution.",
    )
    min_pairs_for_analysis: int = Field(
        default=5,
        description="Minimum number of valid paired scores required for Wilcoxon.",
    )
    dataset_path: str = Field(
        description="Repository-relative path to the scenario dataset file.",
    )
    company_descriptions_path: str = Field(
        description="Repository-relative path to the company descriptions YAML file.",
    )
    prompts_path: str = Field(
        description="Repository-relative path to the prompt templates YAML file.",
    )
    default_masked_name: str = Field(
        default="Company X",
        description="Fallback anonymized label when a scenario lacks a masked name.",
    )
    llm_models: list[BrandMaskingModelConfig] = Field(
        description="Models that should be evaluated for the experiment.",
    )

    def resolve_path(self, path_value: str) -> Path:
        """Resolve a repository-relative config path to an absolute path."""
        return ROOT_DIR / path_value


class CompanyDescription(BaseModel):
    """A company entry that can be injected into named and masked prompts."""

    company_id: str = Field(
        description="Stable identifier used to join scenarios to company metadata.",
    )
    incumbent_name: str = Field(
        description="Real company name shown in the named condition.",
    )
    masked_name: str = Field(
        default="Company X",
        description="Placeholder name shown in the masked condition.",
    )
    description: str = Field(
        default="",
        description="Capability profile text used in both named and masked prompts.",
    )


class CompanyDescriptionsCatalog(BaseModel):
    """YAML root model containing all company description entries."""

    companies: list[CompanyDescription] = Field(
        description="All companies currently defined for the experiment.",
    )


def _load_yaml(path: Path) -> dict:
    """Load a YAML file and always return a dictionary payload."""
    with open(path) as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at YAML root in {path}")

    return data


def load_brand_masking_config(sector_config_path: Path) -> BrandMaskingExperimentConfig:
    """
    Load and validate the runtime experiment configuration.

    Merges the shared base_config.yaml with the sector-specific config. Sector
    values take precedence over base values when keys overlap.
    """
    base_config_path = sector_config_path.parent.parent / "base_config.yaml"
    base = _load_yaml(base_config_path) if base_config_path.exists() else {}
    sector = _load_yaml(sector_config_path)
    merged = {**base, **sector}
    return BrandMaskingExperimentConfig.model_validate(merged)


def load_company_descriptions(path: Path) -> CompanyDescriptionsCatalog:
    """Load and validate the company descriptions catalog."""
    return CompanyDescriptionsCatalog.model_validate(_load_yaml(path))
