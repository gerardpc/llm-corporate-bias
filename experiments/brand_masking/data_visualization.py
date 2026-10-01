"""Visualizations for the brand-masking experiment."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

from bias_in_llms.database import query_db
from bias_in_llms.utils.utils import save_plots_path

_NAMED_COLOR = "#E07B54"
_MASKED_COLOR = "#5B8DB8"
_POSITIVE_LINE_COLOR = "#C0392B"
_NEGATIVE_LINE_COLOR = "#2980B9"
_NEUTRAL_LINE_COLOR = "#95A5A6"


def plot_score_differences(
    df: pd.DataFrame,
    model_id: str,
    save_path: Path,
    sector: str = "pharma",
) -> None:
    """
    Horizontal diverging bar chart — mean name bias per company across all runs.

    Each bar = mean delta (named − masked) across runs, with low alpha.
    Individual run dots are overlaid so variability is directly visible.
    A fixed-column label on the right shows the mean value clearly.
    """
    clean = df.dropna(subset=["score_named", "score_masked"]).copy()
    clean["score_delta"] = clean["score_named"] - clean["score_masked"]
    label_col = "incumbent_name" if "incumbent_name" in clean.columns else "scenario_id"
    clean["label"] = clean[label_col].astype(str)

    if clean.empty:
        print(f"   ⚠️  plot_score_differences: no valid pairs for {model_id}, skipping.")
        return

    n_runs = clean["run"].nunique() if "run" in clean.columns else 1

    agg = (
        clean.groupby("label")["score_delta"]
        .agg(mean_delta="mean", std_delta="std")
        .reset_index()
    )
    agg["std_delta"] = agg["std_delta"].fillna(0)
    agg = agg.sort_values("mean_delta", ascending=True).reset_index(drop=True)

    overall_mean = agg["mean_delta"].mean()
    n_positive = (agg["mean_delta"] > 0).sum()
    n_negative = (agg["mean_delta"] < 0).sum()
    n_neutral = (agg["mean_delta"] == 0).sum()

    bar_colors = [
        _POSITIVE_LINE_COLOR
        if d > 0
        else _NEGATIVE_LINE_COLOR
        if d < 0
        else _NEUTRAL_LINE_COLOR
        for d in agg["mean_delta"]
    ]

    n_companies = len(agg)
    fig_height = max(5, min(n_companies * 0.75, 13))
    fig, ax = plt.subplots(figsize=(11, fig_height))

    # Background bars (low alpha) show the mean direction
    ax.barh(
        agg["label"],
        agg["mean_delta"],
        color=bar_colors,
        alpha=0.35,
        edgecolor="none",
        height=0.55,
    )

    ax.axvline(0, color="black", linewidth=1.2, linestyle="--", zorder=3)
    ax.axvline(overall_mean, color="dimgrey", linewidth=1.5, linestyle=":", zorder=3)

    # Fixed-column mean labels on the far right (outside the data area)
    raw_max = (agg["mean_delta"].abs() + agg["std_delta"]).max()
    x_max = raw_max + 0.8
    ax.set_xlim(-x_max, x_max)

    label_x = x_max - 0.05
    for i, (_label, delta) in enumerate(
        zip(agg["label"], agg["mean_delta"], strict=False),
    ):
        ax.text(
            label_x,
            i,
            f"avg: {delta:+.2f}",
            va="center",
            ha="right",
            fontsize=8.5,
            fontweight="bold",
            color=bar_colors[i],
        )

    ax.set_xlabel(
        f"Name Bias  (Score with {sector} brand name − Score without brand name)",
        fontsize=10,
    )

    runs_label = f"{n_runs} run{'s' if n_runs > 1 else ''}"
    title = f"Brand Masking Bias: Does Knowing the {sector.capitalize()} Brand Name Inflate the Score? — {model_id} ({runs_label})"  # noqa: E501
    fig.suptitle(title, fontsize=12, fontweight="bold")

    n_pos_label = (
        f"{n_positive} {sector} {'company' if n_positive == 1 else 'companies'}"
    )
    n_neg_label = (
        f"{n_negative} {sector} {'company' if n_negative == 1 else 'companies'}"
    )
    n_neu_label = f"{n_neutral} {sector} {'company' if n_neutral == 1 else 'companies'}"

    pos_patch = plt.Rectangle(
        (0, 0),
        1,
        1,
        fc=_POSITIVE_LINE_COLOR,
        alpha=0.85,
        label=f"Name bias detected — named scored higher ({n_pos_label})",
    )
    neg_patch = plt.Rectangle(
        (0, 0),
        1,
        1,
        fc=_NEGATIVE_LINE_COLOR,
        alpha=0.85,
        label=f"Reverse name bias — masked scored higher ({n_neg_label})",
    )
    neu_patch = plt.Rectangle(
        (0, 0),
        1,
        1,
        fc=_NEUTRAL_LINE_COLOR,
        alpha=0.85,
        label=f"No name bias ({n_neu_label})",
    )
    mean_line = plt.Line2D(
        [0],
        [0],
        color="dimgrey",
        linewidth=1.5,
        linestyle=":",
        label=f"Overall mean name bias Δ = {overall_mean:+.2f} across all companies",
    )
    fig.legend(
        handles=[pos_patch, neg_patch, neu_patch, mean_line],
        fontsize=8,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
    )

    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        save_path / f"score_differences_{model_id}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_score_distributions(
    df: pd.DataFrame,
    model_id: str,
    save_path: Path,
    sector: str = "pharma",
) -> None:
    """
    Violin + strip plot with paired connections — Named vs Masked distributions.

    Shows the full distribution of scores in each condition.
    Thin lines connect each company's named and masked score.
    The median bar inside each violin is the key summary statistic.
    """
    clean = df.dropna(subset=["score_named", "score_masked"]).copy()
    if clean.empty:
        print(
            f"   ⚠️  plot_score_distributions: no valid pairs for {model_id}, skipping.",
        )
        return

    mean_named = clean["score_named"].mean()
    mean_masked = clean["score_masked"].mean()
    median_named = clean["score_named"].median()
    median_masked = clean["score_masked"].median()

    melted = pd.DataFrame(
        {
            "Condition": ["Named"] * len(clean) + ["Masked"] * len(clean),
            "Score": clean["score_named"].tolist() + clean["score_masked"].tolist(),
        },
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    conditions = ["Named", "Masked"]
    colors = [_NAMED_COLOR, _MASKED_COLOR]

    for i, (cond, color) in enumerate(zip(conditions, colors, strict=False)):
        subset = melted[melted["Condition"] == cond]["Score"]
        parts = ax.violinplot(
            subset,
            positions=[i],
            widths=0.55,
            showmedians=True,
            showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.3)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(2.5)

    for i, (cond, color) in enumerate(zip(conditions, colors, strict=False)):
        subset = melted[melted["Condition"] == cond]["Score"]
        ax.scatter(
            [i] * len(subset),
            subset,
            color=color,
            s=40,
            alpha=0.55,
            zorder=3,
            edgecolors="white",
            linewidths=0.5,
        )

    # Draw mean as a dashed horizontal line per condition
    ax.hlines(
        mean_named,
        -0.28,
        0.28,
        colors="black",
        linestyles="dashed",
        linewidth=1.8,
        zorder=4,
    )
    ax.hlines(
        mean_masked,
        0.72,
        1.28,
        colors="black",
        linestyles="dashed",
        linewidth=1.8,
        zorder=4,
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [
            f"Named\n(real {sector} brand name shown)",
            f"Masked\n({sector} brand replaced with 'Company X')",
        ],
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylim(0, 20)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.set_ylabel("LLM Score (1–10)", fontsize=11)

    title = f"Brand Masking Bias: Are {sector.capitalize()} Brand Names Inflating LLM Scores? — {model_id}"  # noqa: E501
    fig.suptitle(title, fontsize=12, fontweight="bold")

    named_dot = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=_NAMED_COLOR,
        markersize=8,
        label="Named score (individual observations)",
    )
    masked_dot = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=_MASKED_COLOR,
        markersize=8,
        label="Masked score (individual observations)",
    )
    median_patch = plt.Line2D(
        [0],
        [0],
        color="black",
        linewidth=2.5,
        label=f"Median score  (Named={median_named:.0f} · Masked={median_masked:.0f})",
    )
    mean_patch = plt.Line2D(
        [0],
        [0],
        color="black",
        linewidth=1.8,
        linestyle="dashed",
        label=f"Mean score  (Named={mean_named:.2f} · Masked={mean_masked:.2f})",
    )
    fig.legend(
        handles=[named_dot, masked_dot, median_patch, mean_patch],
        fontsize=8,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.10),
        frameon=False,
    )

    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        save_path / f"score_distributions_{model_id}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    import argparse

    from bias_in_llms.database.brand_masking_db import get_raw_full_table_name

    parser = argparse.ArgumentParser(
        description="Generate brand-masking visualizations.",
    )
    parser.add_argument(
        "--sector",
        required=True,
        help="Sector to visualize (e.g. pharma, consulting, cloud).",
    )
    args = parser.parse_args()
    _sector = args.sector

    query = f"SELECT * FROM {get_raw_full_table_name(_sector)}"
    data = query_db(query=query)

    if data.empty:
        print("❌ No data found in database. Run the experiment first!")
        exit(1)

    models = data["model_id"].unique()
    print(f"✅ Loaded {len(data)} rows of data")
    print(f"📈 Generating plots for {len(models)} model(s): {', '.join(models)}")
    print()

    for model_id in models:
        # Sanitize model_id for use in filenames and directory names:
        # some providers return ids with slashes (e.g. "google/gemini-2.5-flash-lite")
        safe_model_id = model_id.replace("/", "_").replace(":", "_")

        print(f"🎨 Processing model: {model_id}")
        save_path = save_plots_path(
            experiment_name="brand_masking",
            model_id=safe_model_id,
            preceding_context_class=_sector,
        )
        print(f"   • Save path: {save_path}")

        model_data = data[data["model_id"] == model_id].copy()
        valid_data = model_data[
            model_data["named_parse_success"] & model_data["masked_parse_success"]
        ]

        if valid_data.empty:
            print(f"   ⚠️  No valid paired scores for {model_id}, skipping.")
            print()
            continue

        print(f"   • Valid pairs: {len(valid_data)}")

        plot_score_differences(valid_data, safe_model_id, save_path, sector=_sector)
        plot_score_distributions(valid_data, safe_model_id, save_path, sector=_sector)

        print(f"   ✅ Generated 2 plots for {model_id}")
        print()

    print("🎉 All visualizations completed!")
