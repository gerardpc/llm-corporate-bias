"""This module contains the functions for visualizing the data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from bias_in_llms.utility_analysis.utility_models.bradley_terry.bradley_terry_model import (  # noqa: E501
    bradley_terry_model,
)
from bias_in_llms.utility_analysis.utility_models.bradley_terry.bradley_terry_model import (  # noqa: E501
    compute_win_matrix as compute_win_matrix_bt,
)
from bias_in_llms.utility_analysis.utility_models.thurstone.thurstone_model_v import (  # noqa: E501
    compute_win_matrix,
    thurstone_model_v,
)


def generate_color_palette(entities: list[str]) -> dict[str, str]:
    """
    Generate a consistent color palette for entities.

    Args:
        entities: List of entity names.

    Returns:
        Dictionary mapping entity names to hex color codes.
    """
    # Use a colorblind-friendly palette with distinct colors
    n_entities = len(entities)

    if n_entities <= 10:
        # Use tab10 for up to 10 entities
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_entities]
    elif n_entities <= 20:
        # Combine tab10 and tab20b for up to 20 entities
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_entities]
    else:
        # Use hsv for many entities
        colors = plt.cm.hsv(np.linspace(0, 0.9, n_entities))

    # Convert to hex
    color_hex = [
        f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        for r, g, b, _ in colors
    ]

    # Sort entities alphabetically for consistency
    sorted_entities = sorted(entities)
    return dict(zip(sorted_entities, color_hex, strict=False))


def plot_entity_frequency(
    df: pd.DataFrame,
    entities: list[str],
    model_id: str,
    save_path: Path,
    color_map: dict[str, str] | None = None,
) -> None:
    """
    Plot the entity frequency.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        entities (list[str]): The entities to plot.
        model_id (str): The model ID.
        save_path (Path): The path to save the plot.
        color_map (dict[str, str] | None): Optional color mapping for entities.

    Returns:
        None. The plot is saved to the save_path.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()
    df_copy["chosen_entity"] = df_copy.apply(
        lambda row: row["option_a"] if row["answer"] == "A" else row["option_b"],
        axis=1,
    )

    freq_series = df_copy["chosen_entity"].value_counts()

    freq_full = freq_series.reindex(entities, fill_value=0).reset_index()
    freq_full.columns = ["Entity", "Frequency"]

    freq_full = freq_full.sort_values(by="Frequency", ascending=False)

    # Generate or use provided color map
    if color_map is None:
        color_map = generate_color_palette(entities)

    # Create color list in the order of the sorted dataframe
    colors = [color_map[entity] for entity in freq_full["Entity"]]

    sns.barplot(
        data=freq_full,
        x="Entity",
        y="Frequency",
        hue="Entity",
        palette=colors,
        legend=False,
    )
    plt.xticks(rotation=45)
    plt.title(f"Frequency of Chosen Entities for {model_id}")
    plt.tight_layout()
    plt.savefig(save_path / f"entity_frequency_{model_id}.png", dpi=300)
    plt.close()


def plot_preference_frequency(df, model_id, save_path):
    """Plot the frequency of 'Now' vs 'Later' preferences."""
    preference_counts = df["preference"].value_counts().reset_index()
    preference_counts.columns = ["Preference", "Frequency"]

    sns.barplot(
        data=preference_counts,
        x="Preference",
        y="Frequency",
        hue="Preference",
        palette="pastel",
        legend=False,
    )
    plt.title(f"Preference Frequency ('Now' vs 'Later') for {model_id}")
    plt.tight_layout()
    plt.savefig(save_path / f"preference_frequency_{model_id}.png", dpi=300)
    plt.close()


def plot_order_bias(
    df: pd.DataFrame,
    model_id: str,
    save_path: Path,
    color_map: dict[str, str] | None = None,
) -> None:
    """
    Plot the order bias.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        model_id (str): The model ID.
        save_path (Path): The path to save the plot.
        color_map (dict[str, str] | None): Optional color mapping for choices.

    Returns:
        None. The plot is saved to the save_path.
    """
    order_bias = df["answer"].value_counts().reset_index()
    order_bias.columns = ["Option", "Frequency"]

    # Create color palette for A/B choices
    if color_map is None:
        # Default colors for A and B
        choice_colors = {"A": "#1f77b4", "B": "#ff7f0e"}
    else:
        # Use consistent colors if provided (though for A/B we use default)
        choice_colors = {"A": "#1f77b4", "B": "#ff7f0e"}

    colors = [choice_colors.get(opt, "#808080") for opt in order_bias["Option"]]

    sns.barplot(
        data=order_bias,
        x="Option",
        y="Frequency",
        hue="Option",
        palette=colors,
        legend=False,
    )
    plt.title(f"Choice Frequency by Option Position (A vs B) for {model_id}")
    plt.savefig(save_path / f"order_bias_{model_id}.png", dpi=300)
    plt.close()


def plot_confusion_matrix(
    df: pd.DataFrame,
    entities: list[str],
    model_id: str,
    save_path: Path,
    color_map: dict[str, str] | None = None,
) -> None:
    """
    Plot a confusion matrix showing head-to-head company "fights".

    The matrix shows how often each company (row) was chosen over each other company.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        entities (list[str]): The entities to plot.
        model_id (str): The model ID.
        save_path (Path): The path to save the plot.
        color_map (dict[str, str] | None): Optional color mapping for entities.

    Returns:
        None. The plot is saved to the save_path.
    """
    # Initialize confusion matrix
    n = len(entities)
    confusion = np.zeros((n, n))
    entity_to_idx = {entity: i for i, entity in enumerate(entities)}

    # Populate confusion matrix
    for _, row in df.iterrows():
        option_a = row["option_a"]
        option_b = row["option_b"]
        answer = row["answer"]

        if option_a not in entity_to_idx or option_b not in entity_to_idx:
            continue

        idx_a = entity_to_idx[option_a]
        idx_b = entity_to_idx[option_b]

        if answer == "A":
            # Option A was chosen over Option B
            confusion[idx_a, idx_b] += 1
        elif answer == "B":
            # Option B was chosen over Option A
            confusion[idx_b, idx_a] += 1

    # Create figure
    fig, ax = plt.subplots(figsize=(max(10, n * 0.8), max(8, n * 0.7)))

    # Plot heatmap
    im = ax.imshow(confusion, cmap="YlOrRd", aspect="auto")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(
        "Number of times row company was chosen over column company",
        rotation=270,
        labelpad=20,
    )

    # Set ticks and labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(entities, rotation=45, ha="right")
    ax.set_yticklabels(entities)

    # Add text annotations
    for i in range(n):
        for j in range(n):
            if i != j:  # Don't annotate diagonal (company vs itself)
                ax.text(
                    j,
                    i,
                    int(confusion[i, j]),
                    ha="center",
                    va="center",
                    color="white" if confusion[i, j] > confusion.max() / 2 else "black",
                    fontsize=8,
                )

    # Labels and title
    ax.set_xlabel("Lost to (Column Company)")
    ax.set_ylabel("Won against (Row Company)")
    ax.set_title(
        f"Company Head-to-Head Comparison Matrix for {model_id}\n"
        "(Read: Row company chosen X times over column company)",
        pad=20,
    )

    plt.tight_layout()
    plt.savefig(save_path / f"confusion_matrix_{model_id}.png", dpi=300)
    plt.close()


def plot_thurstone_utilities(
    df: pd.DataFrame,
    entities: list[str],
    model_id: str,
    save_path: Path,
    color_map: dict[str, str] | None = None,
) -> None:
    """
    Plot the Thurstonian utilities.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        entities (list[str]): The entities to plot.
        model_id (str): The model ID.
        save_path (Path): The path to save the plot.
        color_map (dict[str, str] | None): Optional color mapping for entities.

    Returns:
        None. The plot is saved to the save_path.
    """
    win_matrix = compute_win_matrix(df, entities)
    utilities = thurstone_model_v(win_matrix)

    df_util = utilities.sort_values(ascending=False).reset_index()
    df_util.columns = ["Entity", "Utility"]

    # Generate or use provided color map
    if color_map is None:
        color_map = generate_color_palette(entities)

    # Create color list in the order of the sorted dataframe
    colors = [color_map[entity] for entity in df_util["Entity"]]

    sns.barplot(
        data=df_util,
        x="Entity",
        y="Utility",
        hue="Entity",
        palette=colors,
        legend=False,
    )
    plt.xticks(rotation=45)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title(f"Thurstonian Utilities per Entity for {model_id}")
    plt.tight_layout()
    plt.savefig(save_path / f"thurstone_utilities_{model_id}.png", dpi=300)
    plt.close()


def plot_bradley_terry_utilities(
    df: pd.DataFrame,
    entities: list[str],
    model_id: str,
    save_path: Path,
    color_map: dict[str, str] | None = None,
) -> None:
    """
    Plot the Bradley-Terry utilities.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        entities (list[str]): The entities to plot.
        model_id (str): The model ID.
        save_path (Path): The path to save the plot.
        color_map (dict[str, str] | None): Optional color mapping for entities.

    Returns:
        None. The plot is saved to the save_path.
    """
    win_matrix = compute_win_matrix_bt(df, entities)
    utilities = bradley_terry_model(win_matrix)

    df_util = utilities.sort_values(ascending=False).reset_index()
    df_util.columns = ["Entity", "Utility"]

    # Generate or use provided color map
    if color_map is None:
        color_map = generate_color_palette(entities)

    # Create color list in the order of the sorted dataframe
    colors = [color_map[entity] for entity in df_util["Entity"]]

    sns.barplot(
        data=df_util,
        x="Entity",
        y="Utility",
        hue="Entity",
        palette=colors,
        legend=False,
    )
    plt.xticks(rotation=45)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title(f"Bradley-Terry Utilities per Entity for {model_id}")
    plt.tight_layout()
    plt.savefig(save_path / f"bradley_terry_utilities_{model_id}.png", dpi=300)
    plt.close()


def plot_utility_comparison(
    df: pd.DataFrame,
    entities: list[str],
    model_id: str,
    save_path: Path,
    color_map: dict[str, str] | None = None,
) -> None:
    """
    Plot a comparison between Thurstone and Bradley-Terry utilities.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        entities (list[str]): The entities to plot.
        model_id (str): The model ID.
        save_path (Path): The path to save the plot.
        color_map (dict[str, str] | None): Optional color mapping for entities.

    Returns:
        None. The plot is saved to the save_path.
    """
    # Compute utilities for both models
    win_matrix_thurstone = compute_win_matrix(df, entities)
    win_matrix_bt = compute_win_matrix_bt(df, entities)

    thurstone_utilities = thurstone_model_v(win_matrix_thurstone)
    bt_utilities = bradley_terry_model(win_matrix_bt)

    # Create comparison dataframe
    comparison_df = pd.DataFrame(
        {
            "Entity": entities,
            "Thurstone": thurstone_utilities.reindex(entities, fill_value=0),
            "Bradley-Terry": bt_utilities.reindex(entities, fill_value=0),
        },
    )

    # Generate or use provided color map
    if color_map is None:
        color_map = generate_color_palette(entities)

    # Create side-by-side comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Thurstone plot
    thurstone_sorted = comparison_df.sort_values("Thurstone", ascending=False)
    colors_thurstone = [color_map[entity] for entity in thurstone_sorted["Entity"]]
    sns.barplot(
        data=thurstone_sorted,
        x="Entity",
        y="Thurstone",
        hue="Entity",
        palette=colors_thurstone,
        ax=ax1,
        legend=False,
    )
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax1.set_title(f"Thurstone Utilities - {model_id}")

    # Bradley-Terry plot
    bt_sorted = comparison_df.sort_values("Bradley-Terry", ascending=False)
    colors_bt = [color_map[entity] for entity in bt_sorted["Entity"]]
    sns.barplot(
        data=bt_sorted,
        x="Entity",
        y="Bradley-Terry",
        hue="Entity",
        palette=colors_bt,
        ax=ax2,
        legend=False,
    )
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax2.set_title(f"Bradley-Terry Utilities - {model_id}")

    plt.tight_layout()
    plt.savefig(save_path / f"utility_comparison_{model_id}.png", dpi=300)
    plt.close()

    # Correlation plot with color-coded points
    plt.figure(figsize=(8, 6))
    for entity in comparison_df["Entity"]:
        entity_data = comparison_df[comparison_df["Entity"] == entity]
        plt.scatter(
            entity_data["Thurstone"],
            entity_data["Bradley-Terry"],
            color=color_map[entity],
            label=entity,
            alpha=0.7,
            s=100,
        )

    plt.xlabel("Thurstone Utilities")
    plt.ylabel("Bradley-Terry Utilities")
    plt.title(f"Correlation between Thurstone and Bradley-Terry Utilities - {model_id}")

    # Add correlation coefficient
    correlation = comparison_df["Thurstone"].corr(comparison_df["Bradley-Terry"])
    plt.text(
        0.05,
        0.95,
        f"Correlation: {correlation:.3f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / f"utility_correlation_{model_id}.png", dpi=300)
    plt.close()
