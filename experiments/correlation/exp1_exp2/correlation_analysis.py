#!/usr/bin/env python3
"""Compare company win frequency with its identity-disclosure effect."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CORRELATION_DIR = REPO / "experiments" / "correlation"
EXP1_FILES = {
    "pharma": CORRELATION_DIR / "pharma_preference_counts.csv",
    "cloud": CORRELATION_DIR / "cloud_provider_preference_counts.csv",
    "consulting": CORRELATION_DIR / "consulting_preference_counts.csv",
    "banking": CORRELATION_DIR / "banking_preference_counts.csv",
}
COMPANY_EFFECTS = (
    REPO
    / "experiments"
    / "brand_masking"
    / "manuscript_analysis"
    / "outputs"
    / "company_effects.csv"
)
MANUSCRIPT_FIGURE = (
    REPO / "manuscript" / "satml_manuscript" / "fig_bias_vs_win_frequency.pdf"
)
MODEL_IDS = {
    "GPT-5.6 Sol": "gpt-5.6-sol",
    "Qwen 3.8 Max": "qwen/qwen3.8-max-0902",
    "Claude Opus 5": "anthropic.claude-opus-5",
    "DeepSeek V4.1": "deepseek/deepseek-v4.1-flash",
    "GLM 5.3": "z-ai/glm-5.3-flash",
    "GPT-5.6 Luna": "gpt-5.6-luna",
}
SECTOR_ORDER = ("pharma", "cloud", "consulting", "banking")


def company_key(value: str) -> str:
    """Return a punctuation-insensitive company key."""
    return re.sub(r"[^a-z0-9]", "", value.lower().replace("&", "and"))


def complete_models(effects: pd.DataFrame, sector: str, company_count: int) -> list[str]:
    """Return matched models with complete company coverage in a sector."""
    sector_rows = effects.loc[effects["sector"].eq(sector)]
    coverage = sector_rows.groupby("model_id")["company_key"].nunique()
    return [
        model_id
        for model_id in MODEL_IDS.values()
        if coverage.get(model_id, 0) == company_count
    ]


def load_data() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Load matched company summaries using identical complete model sets."""
    effects = pd.read_csv(COMPANY_EFFECTS)
    effects = effects.loc[effects["window"].eq("sep_sweep")].copy()
    effects["company_key"] = effects["incumbent_name"].map(company_key)

    parts: list[pd.DataFrame] = []
    models_by_sector: dict[str, list[str]] = {}
    for sector in SECTOR_ORDER:
        wins = pd.read_csv(EXP1_FILES[sector])
        wins["company_key"] = wins["company"].map(company_key)
        model_ids = complete_models(effects, sector, len(wins))
        if not model_ids:
            raise ValueError(f"No completely observed matched models for {sector}")
        models_by_sector[sector] = model_ids

        model_columns = [
            exp1_name
            for exp1_name, model_id in MODEL_IDS.items()
            if model_id in model_ids
        ]
        wins["exp1_wins"] = wins[model_columns].mean(axis=1)
        deltas = (
            effects.loc[
                effects["sector"].eq(sector) & effects["model_id"].isin(model_ids)
            ]
            .groupby("company_key", as_index=False)["mean_delta"]
            .mean()
            .rename(columns={"mean_delta": "exp2_delta"})
        )
        joined = wins[["company", "company_key", "exp1_wins"]].merge(
            deltas, on="company_key", how="inner", validate="one_to_one"
        )
        if len(joined) != len(wins):
            raise ValueError(
                f"{sector}: matched {len(joined)} of {len(wins)} companies"
            )
        joined["sector"] = sector
        joined["models_averaged"] = len(model_ids)
        parts.append(joined)
    return pd.concat(parts, ignore_index=True), models_by_sector


def main() -> None:
    """Write matched data, correlation summary, and manuscript figure."""
    data, models_by_sector = load_data()
    data.to_csv(HERE / "matched_company_data.csv", index=False)

    summaries: list[dict[str, float | int | str]] = []
    fig, axes = plt.subplots(
        1, 4, figsize=(7.15, 2.25), sharey=True, constrained_layout=True
    )
    for panel, (ax, sector) in enumerate(zip(axes, SECTOR_ORDER), start=1):
        group = data.loc[data["sector"].eq(sector)]
        x = group["exp1_wins"].to_numpy()
        y = group["exp2_delta"].to_numpy()
        pearson_r, pearson_p = pearsonr(x, y)
        spearman_rho, spearman_p = spearmanr(x, y)
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 100)

        ax.scatter(x, y, s=18, color="#62A8D8", edgecolor="white", linewidth=0.4)
        ax.plot(grid, intercept + slope * grid, color="#155A8A", linewidth=1.1)
        ax.axhline(0, color="0.55", linewidth=0.6)
        ax.set_title(sector.title(), fontsize=8, pad=4)
        ax.set_xlabel("Mean paired wins (count)", fontsize=7)
        if panel == 1:
            ax.set_ylabel("Mean identity effect (score points)", fontsize=7)
        ax.text(
            0.97,
            0.97,
            rf"$r={pearson_r:.2f}$, $p={pearson_p:.3f}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=5.5,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1},
        )
        ax.grid(color="0.93", linewidth=0.45)
        ax.tick_params(labelsize=6.5)
        for spine in ax.spines.values():
            spine.set_color("0.35")
            spine.set_linewidth(0.65)
        summaries.append(
            {
                "sector": sector,
                "n_companies": len(group),
                "n_models": len(models_by_sector[sector]),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
                "slope_delta_per_win": slope,
                "intercept": intercept,
            }
        )

    figure_path = HERE / "experiment1_vs_experiment2_correlations.pdf"
    fig.savefig(figure_path, bbox_inches="tight")
    fig.savefig(MANUSCRIPT_FIGURE, bbox_inches="tight")
    fig.savefig(figure_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    summary = pd.DataFrame(summaries)
    summary.to_csv(HERE / "experiment1_vs_experiment2_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
