"""This module is needed for saving the figures of the company bias experiment."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.ticker import FuncFormatter

from bias_in_llms.config.project_paths import EXPERIMENTS_DIR
from bias_in_llms.database import query_db
from bias_in_llms.utils.data_visualization import (
    generate_color_palette,
    plot_bradley_terry_utilities,
    plot_confusion_matrix,
    plot_entity_frequency,
    plot_order_bias,
    plot_thurstone_utilities,
    plot_utility_comparison,
)
from bias_in_llms.utils.utils import save_plots_path

# Mapping of sector names to their table name prefixes (same as experiment.py)
SECTOR_TABLE_MAP = {
    "pharma": "pharma",
    "consulting": "consulting",
    "cloud_provider": "cloud_provider",
    "banking": "banking",
}

EXCLUDED_COMPANIES_BY_SECTOR = {
    "cloud_provider": {"Salesforce Platform (Salesforce Cloud)", "Salesforce"},
}

# Display-only abbreviations for cloud_provider company names
CLOUD_PROVIDER_NAME_MAP = {
    "Amazon Web Services (AWS)": "AWS",
    "Microsoft Azure": "Azure",
    "Google Cloud": "Google",
    "IBM Cloud": "IBM",
    "Oracle Cloud Infrastructure": "Oracle",
    "Alibaba Cloud": "Alibaba",
    "Tencent Cloud": "Tencent",
    "Huawei Cloud": "Huawei",
    "Salesforce Platform (Salesforce Cloud)": "Salesforce",
    "DigitalOcean": "DigitalOcean",
    "OVH cloud": "OVH",
    "Deutsche Telekom Cloud": "Deutsche Telekom",
}

BANKS_NAME_MAP = {
    "Bank of America": "Bank of America",
    "JPMorgan Chase": "JPMorgan",
    "Bank of China ": "BoC",
    "China Construction Bank": "CCB",
    "Agricultural Bank of China": "ABC",
    "Industrial and Commercial Bank of China (ICBC)": "ICBC",
    "Morgan Stanley": "Morgan Stanley",
    "HSBC": "HSBC",
    "BBVA": "BBVA",
    "Banco Santander": "Santander",
    "UBS": "UBS",
}


def _format_sector_title(sector: str) -> str:
    """Format a sector name for plot titles."""
    return sector.replace("_", " ").title()


def _short_company_label(name: str) -> str:
    """Return a shorter display label for crowded axes."""
    custom_labels = {
        "McKinsey & Company": "McKinsey",
        "Boston Consulting Group": "BCG",
        "Bain & Company": "Bain",
        "Deloitte Consulting": "Deloitte",
        "PwC Advisory": "PwC",
        "EY Consulting": "EY",
        "KPMG Advisory": "KPMG",
        "Oliver Wyman": "O. Wyman",
        "Roland Berger": "R. Berger",
        "Johnson & Johnson": "J&J",
        "GlaxoSmithKline": "GSK",
        "Merck & Co.": "Merck",
        "Bristol Myers Squibb": "BMS",
        "Regeneron Pharmaceuticals": "Regeneron",
        "Gilead Sciences": "Gilead",
        "Vertex Pharmaceuticals": "Vertex",
        "Novo Nordisk": "Novo",
        "Bank of America": "BofA",
        "Banco Santander": "Santander",
        "Morgan Stanley": "M. Stanley",
        "Agricultural Bank of China": "ABC",
        "China Construction Bank": "CCB",
        "Industrial and Commercial Bank of China (ICBC)": "ICBC",
        "Amazon Web Services (AWS)": "AWS",
        "Microsoft Azure": "Azure",
        "Google Cloud": "Google",
        "Oracle Cloud Infrastructure": "Oracle",
        "Salesforce Platform (Salesforce Cloud)": "Salesforce",
        "Deutsche Telekom Cloud": "DT Cloud",
    }
    if name in custom_labels:
        return custom_labels[name]

    if len(name) <= 14:
        return name

    tokens = [
        token for token in name.replace("(", "").replace(")", "").split(" ") if token
    ]
    if len(tokens) >= 2:
        acronym = "".join(
            token[0].upper()
            for token in tokens
            if token[0].isalnum() and token.lower() not in {"and", "of", "&"}
        )
        if 2 <= len(acronym) <= 6:
            return acronym

    return name[:14]


def _format_k(value: float) -> str:
    """Format large numbers with a k suffix."""
    if value >= 1000:
        return f"{int(round(value / 1000))}k"
    return str(int(round(value)))


def _apply_sector_display_names(data: pd.DataFrame, sector: str) -> pd.DataFrame:
    """Apply sector-specific display name abbreviations."""
    if sector == "cloud_provider":
        data = data.copy()
        data["option_a"] = data["option_a"].map(
            lambda x: CLOUD_PROVIDER_NAME_MAP.get(x, x),
        )
        data["option_b"] = data["option_b"].map(
            lambda x: CLOUD_PROVIDER_NAME_MAP.get(x, x),
        )
    elif sector == "banking":
        data = data.copy()
        data["option_a"] = data["option_a"].map(
            lambda x: BANKS_NAME_MAP.get(x, x),
        )
        data["option_b"] = data["option_b"].map(
            lambda x: BANKS_NAME_MAP.get(x, x),
        )
    return data


def _exclude_sector_companies(data: pd.DataFrame, sector: str) -> pd.DataFrame:
    """Exclude configured companies for a sector from pairwise data."""
    excluded_companies = EXCLUDED_COMPANIES_BY_SECTOR.get(sector, set())
    if not excluded_companies:
        return data
    return data[
        ~data["option_a"].isin(excluded_companies)
        & ~data["option_b"].isin(excluded_companies)
    ].copy()


def _compute_pairwise_wins(df: pd.DataFrame, entities: list[str]) -> np.ndarray:
    """Compute pairwise wins matrix (row chosen over column)."""
    n_entities = len(entities)
    pairwise = np.zeros((n_entities, n_entities), dtype=int)
    entity_to_idx = {entity: idx for idx, entity in enumerate(entities)}

    for _, row in df.iterrows():
        option_a = row["option_a"]
        option_b = row["option_b"]
        answer = row["answer"]

        if option_a not in entity_to_idx or option_b not in entity_to_idx:
            continue

        idx_a = entity_to_idx[option_a]
        idx_b = entity_to_idx[option_b]

        if answer == "A":
            pairwise[idx_a, idx_b] += 1
        elif answer == "B":
            pairwise[idx_b, idx_a] += 1

    return pairwise


def _plot_cross_sector_frequency_subplot(
    sector_data: dict[str, pd.DataFrame],
    save_dir: Path,
) -> None:
    """Plot one 2x2 figure with chosen-company frequencies across sectors."""
    bar_color = "#c9472d"
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    axes = axes.flatten()

    for idx, sector in enumerate(SECTOR_TABLE_MAP):
        ax = axes[idx]
        data = sector_data.get(sector)
        if data is None or data.empty:
            ax.set_title(f"{_format_sector_title(sector)} (No data)")
            ax.axis("off")
            continue

        chosen = data.apply(
            lambda row: row["option_a"] if row["answer"] == "A" else row["option_b"],
            axis=1,
        )
        freq = chosen.value_counts().reset_index()
        freq.columns = ["company", "frequency"]
        freq = freq.sort_values("frequency", ascending=False)
        freq["company_label"] = freq["company"].map(_short_company_label)

        sns.barplot(
            data=freq,
            x="company_label",
            y="frequency",
            color=bar_color,
            ax=ax,
        )
        ax.set_title(_format_sector_title(sector))
        ax.set_xlabel("Company", fontsize=11)
        ax.set_ylabel("Frequency")
        ax.tick_params(axis="x", rotation=45, labelsize=13)
        ax.tick_params(axis="y", labelsize=12)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: _format_k(x)))
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45)

    fig.suptitle(
        "Chosen Company Frequency Across Models by Sector",
        fontsize=20,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(
        save_dir / "cross_sector_frequency_across_models.svg",
        format="svg",
    )
    plt.close(fig)


def _plot_cross_sector_pairwise_subplot(
    sector_data: dict[str, pd.DataFrame],
    save_dir: Path,
) -> None:
    """Plot one 2x2 figure with pairwise head-to-head wins across sectors."""
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    axes = axes.flatten()

    for idx, sector in enumerate(SECTOR_TABLE_MAP):
        ax = axes[idx]
        data = sector_data.get(sector)
        if data is None or data.empty:
            ax.set_title(f"{_format_sector_title(sector)} (No data)")
            ax.axis("off")
            continue

        entities = sorted(set(data["option_a"]).union(set(data["option_b"])))
        pairwise_wins = _compute_pairwise_wins(data, entities)
        label_entities = [_short_company_label(entity) for entity in entities]

        sns.heatmap(
            pairwise_wins,
            cmap="YlOrRd",
            annot=True,
            fmt="d",
            xticklabels=label_entities,
            yticklabels=label_entities,
            cbar=False,
            ax=ax,
            annot_kws={"fontsize": 10},
        )
        ax.set_title(_format_sector_title(sector))
        ax.set_xlabel("Lost to (Column Company)", fontsize=11)
        ax.set_ylabel("Won against (Row Company)", fontsize=11)
        ax.tick_params(axis="x", rotation=45, labelsize=13)
        ax.tick_params(axis="y", rotation=0, labelsize=13)

    fig.suptitle(
        "Pairwise Head-to-Head Wins Across Models by Sector",
        fontsize=20,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(
        save_dir / "cross_sector_pairwise_grid_across_models.svg",
        format="svg",
    )
    plt.close(fig)


def generate_cross_sector_aggregated_plots() -> None:
    """
    Generate cross-sector SVG plots aggregated across all models.

    Outputs:
    - cross_sector_frequency_across_models.svg
    - cross_sector_pairwise_grid_across_models.svg
    """
    save_dir = save_plots_path(
        experiment_name="company_bias",
        model_id="cross_sector",
    )
    sector_data: dict[str, pd.DataFrame] = {}

    for sector, table_prefix in SECTOR_TABLE_MAP.items():
        table_name = f"bias_in_llms.{table_prefix}_bias"
        data = query_db(query=f"SELECT * FROM {table_name}")
        if data.empty:
            print(f"⚠️  No data found in {table_name}. Skipping sector '{sector}'.")
            continue

        data = data[data["answer"] != "ERROR"].copy()
        data = data[data["model_id"] != "all_together"].copy()
        data = _exclude_sector_companies(data, sector)
        data = _apply_sector_display_names(data, sector)
        sector_data[sector] = data

    if not sector_data:
        print("⚠️  No data available to generate cross-sector aggregated plots.")
        return

    _plot_cross_sector_frequency_subplot(sector_data, save_dir)
    _plot_cross_sector_pairwise_subplot(sector_data, save_dir)
    print(f"✅ Cross-sector aggregated SVG plots saved in: {save_dir}")


def _load_sector_config(config_dir: Path, sector: str) -> dict:
    """
    Load the sector-specific configuration file.

    Args:
        config_dir: Path to the config directory
        sector: The sector name (e.g., "pharma", "consulting")

    Returns:
        dict: The sector-specific configuration
    """
    sector_config_path = config_dir / f"{sector}_config.yaml"
    if not sector_config_path.exists():
        raise FileNotFoundError(
            f"Sector config file not found: {sector_config_path}. "
            "Please create "
            f"{sector}_config.yaml with 'companies' and 'questions' keys.",
        )

    with open(sector_config_path) as f:
        return yaml.safe_load(f)


def _merge_configs(main_config: dict, sector_config: dict) -> dict:
    """
    Merge main config with sector-specific config.

    Args:
        main_config: The main configuration
        sector_config: The sector-specific configuration (companies, questions)

    Returns:
        dict: Merged configuration
    """
    merged = main_config.copy()
    merged["companies"] = sector_config.get("companies", [])
    merged["questions"] = sector_config.get("questions", [])
    if "prompt_preceding_context_block" in sector_config:
        merged["prompt_preceding_context_block"] = sector_config[
            "prompt_preceding_context_block"
        ]
    return merged


def run_data_visualization(config_path: Path | None = None) -> None:  # noqa: C901
    """
    Run data visualization for the sector bias experiment.

    The sector is determined from the main config.yaml file. This function
    loads the appropriate sector config and queries the correct database table.

    Args:
        config_path: Path to the main config file. Defaults to the standard location.
    """
    if config_path is None:
        config_path = EXPERIMENTS_DIR / "company_bias" / "config" / "config.yaml"

    config_dir = config_path.parent

    # Load main config
    with open(config_path) as f:
        main_config = yaml.safe_load(f)

    # Get sector from config (defaults to "pharma" for backwards compatibility)
    sector = main_config.get("sector", "pharma")
    print(f"🏢 Running visualization for sector: {sector}")

    # Load sector-specific config
    sector_config = _load_sector_config(config_dir, sector)

    # Merge configs
    config = _merge_configs(main_config, sector_config)

    # Get the database table name for this sector
    table_prefix = SECTOR_TABLE_MAP.get(sector, sector)
    table_name = f"bias_in_llms.{table_prefix}_bias"

    # Query the sector-specific table
    query = f"SELECT * FROM {table_name}"
    data = query_db(query=query)

    if data.empty:
        print(f"❌ No data found in {table_name}. Run the experiment first!")
        return

    # Get unique companies from data
    companies = data["option_a"].unique()

    # Generate consistent color palette for all companies
    color_map = generate_color_palette(companies)

    # Handle preceding context blocks
    dummy_preceding_context_block_list = [{"instruction": "", "class": ""}]
    preceding_context_block_list = (
        config.get("prompt_preceding_context_block")
        if config.get("prompt_preceding_context_block")
        else dummy_preceding_context_block_list
    )

    if len(preceding_context_block_list) > 0:
        # Generate all possible filled instructions
        # (handling {company_b_name} placeholders)
        normalized_instructions_set = set()
        for block in preceding_context_block_list:
            instruction_template = block.get("instruction", "")
            if not instruction_template:
                continue
            # Check if template has placeholder
            if "{company_b_name}" in instruction_template:
                # Generate filled instruction for each company
                for company_b_name in companies:
                    filled_instruction = instruction_template.format(
                        company_b_name=company_b_name,
                    )
                    normalized_instructions_set.add(
                        filled_instruction.lower().replace(" ", ""),
                    )
            else:
                # No placeholder, use instruction as-is
                normalized_instructions_set.add(
                    instruction_template.lower().replace(" ", ""),
                )

        # Filter data if we have instructions to match
        if normalized_instructions_set:
            normalized_prompts = (
                data["full_prompt"].str.lower().str.replace(" ", "", regex=False)
            )
            filter_condition = normalized_prompts.apply(
                lambda prompt: any(
                    instruction in prompt for instruction in normalized_instructions_set
                ),
            )
            data = data[filter_condition]

    # Filter by questions from config (using unified "questions" key)
    questions = config.get("questions", [])
    if len(questions) > 0:
        data = data[data["question"].isin(questions)]

    data = _exclude_sector_companies(data, sector)
    data = _apply_sector_display_names(data, sector)

    print(f"✅ Loaded {len(data)} rows of data from {table_name}")

    companies = data["option_a"].unique()
    # Regenerate color palette using the (possibly abbreviated) company names
    color_map = generate_color_palette(companies)
    models = data["model_id"].unique()

    print(
        f"📈 Generating plots for {len(models)} models and {len(companies)} companies",
    )
    print(f"   • Sector: {sector}")
    print(f"   • Companies: {', '.join(companies)}")
    print(f"   • Models: {', '.join(models)}")
    print()

    # Use sector-specific experiment name for saving plots
    experiment_name = f"{sector}_bias"

    for preceding_context_block in preceding_context_block_list:
        for model_id in models:
            if model_id == "qwen/qwen3.6-plus":
                model_id_path = "qwen3.6-plus"
            else:
                model_id_path = model_id

            preceding_context_class = preceding_context_block.get("class", "")
            print(f"🎨 Processing model: {model_id}")
            save_path = save_plots_path(
                experiment_name=experiment_name,
                model_id=model_id_path,
                preceding_context_class=preceding_context_class,
            )
            print(f"   • Save path: {save_path}")

            target_instruction = preceding_context_block.get("instruction")
            filtered_data = data.copy()
            if target_instruction is None or target_instruction == "":
                mask = data["preceding_context_block"].isna() | (
                    data["preceding_context_block"] == ""
                )
            else:
                # Handle placeholders: generate all possible filled instructions
                if "{company_b_name}" in target_instruction:
                    # Generate filled instruction for each company
                    filled_instructions = {
                        target_instruction.format(company_b_name=company_b_name)
                        for company_b_name in companies
                    }
                    mask = data["preceding_context_block"].isin(filled_instructions)
                else:
                    # No placeholder, use instruction as-is
                    mask = data["preceding_context_block"] == target_instruction
            filtered_data = filtered_data[mask]

            # If no data matches, skip to next
            if filtered_data.empty:
                print(
                    f"⚠️  Skipping context '{preceding_context_class}':"
                    " No matching prompts found.",
                )
                continue

            if model_id != "all_together":
                model_data = filtered_data[filtered_data["model_id"] == model_id].copy()
            else:
                model_data = filtered_data

            # Filter out errors
            successful_data = model_data[model_data["answer"] != "ERROR"]

            if successful_data.empty:
                print(f"⚠️  Skipping model {model_id}: No successful results found.")
                print()
                continue

            print(f"   • Data points: {len(successful_data)} successful results")

            # Original plots with consistent colors
            plot_entity_frequency(
                successful_data,
                companies,
                model_id_path,
                save_path,
                color_map,
            )
            plot_order_bias(successful_data, model_id_path, save_path, color_map)

            # Confusion matrix showing head-to-head comparisons
            plot_confusion_matrix(
                successful_data,
                companies,
                model_id_path,
                save_path,
                color_map,
            )

            # Utility analysis plots with consistent colors
            plot_thurstone_utilities(
                successful_data,
                companies,
                model_id_path,
                save_path,
                color_map,
            )
            plot_bradley_terry_utilities(
                successful_data,
                companies,
                model_id_path,
                save_path,
                color_map,
            )

            # Comparison plot with consistent colors
            plot_utility_comparison(
                successful_data,
                companies,
                model_id_path,
                save_path,
                color_map,
            )

            print(f"   ✅ Generated 6 plots for {model_id}")
            print()
            if model_id == "all_together":
                break

    print(f"🎉 All {sector} bias visualizations completed!")
    generate_cross_sector_aggregated_plots()


if __name__ == "__main__":
    run_data_visualization()
