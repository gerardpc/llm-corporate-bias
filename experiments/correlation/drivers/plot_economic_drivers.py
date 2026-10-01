#!/usr/bin/env python3
"""Generate the manuscript figure for exploratory economic bias drivers."""
from __future__ import annotations

import csv
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
EFFECTS = REPO / "experiments/brand_masking/manuscript_analysis/outputs/company_effects.csv"
ECON = ROOT / "public_economic_data.csv"
OUT = ROOT / "plots"
SUMMARY = ROOT / "economic_driver_summary.csv"
MANUSCRIPT_FIGURE = (
    REPO / "manuscript" / "satml_manuscript" / "fig_possible_bias_drivers.pdf"
)

def read(path: Path) -> list[dict[str, str]]:
    """Read a CSV file as dictionaries."""
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def canon(value: str) -> str:
    """Normalize known naming differences across source files."""
    return {"Merck & Co.": "Merck & Co", "OVHcloud": "OVH Cloud"}.get(
        value, value
    )


def permutation_p(x: np.ndarray, y: np.ndarray, permutations: int = 50_000) -> float:
    """Return a two-sided permutation p-value for Pearson correlation."""
    observed = abs(pearsonr(x, y).statistic)
    shuffled = y.copy()
    rng = random.Random(20260929)
    exceedances = 0
    for _ in range(permutations):
        rng.shuffle(shuffled)
        if abs(pearsonr(x, shuffled).statistic) >= observed - 1e-12:
            exceedances += 1
    return (exceedances + 1) / (permutations + 1)

def main() -> None:
    """Build a two-by-two figure and write its correlation statistics."""
    try:
        os.environ.setdefault("MPLBACKEND", "Agg")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required; install it in the analysis environment") from exc
    effects = read(EFFECTS)
    econ = {canon(row["company"]): row for row in read(ECON)}
    grouped: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for r in effects:
        if r["window"] == "may_core":
            grouped[(r["sector"], canon(r["incumbent_name"]))].append(
                float(r["mean_delta"])
            )
    data: defaultdict[str, list[dict[str, float | str | None]]] = defaultdict(list)
    for (sector, company), vals in grouped.items():
        e = econ.get(company)
        if not e:
            continue
        data[sector].append(
            {
                "company": company,
                "delta": sum(vals) / len(vals),
                "cap": (
                    float(e["market_cap_usd_billion"])
                    if e["market_cap_usd_billion"]
                    else None
                ),
                "revenue": (
                    float(e["revenue_ttm_usd_billion"])
                    if e["revenue_ttm_usd_billion"]
                    else None
                ),
            }
        )
    OUT.mkdir(exist_ok=True)
    configs = [
        ("pharma", "cap", "Market cap\n(USD bn, log scale)"),
        ("cloud", "cap", "Parent market cap\n(USD bn, log scale)"),
        ("banking", "cap", "Market cap\n(USD bn, log scale)"),
        ("consulting", "revenue", "Reported revenue\n(USD bn, log scale)"),
    ]
    fig, axes = plt.subplots(
        1, 4, figsize=(7.15, 2.25), sharey=True, constrained_layout=True
    )
    summary: list[dict[str, float | int | str]] = []
    for panel, (ax, (sector, variable, axis_label)) in enumerate(
        zip(axes, configs), start=1
    ):
        rows = [r for r in data[sector] if r[variable] is not None]
        x = np.array([math.log10(float(r[variable])) for r in rows])
        y = np.array([float(r["delta"]) for r in rows])
        pearson_r, _ = pearsonr(x, y)
        pearson_p = permutation_p(x, y)
        spearman_rho, spearman_p = spearmanr(x, y)
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 100)

        ax.scatter(
            x,
            y,
            s=18,
            color="#E6A15A",
            edgecolor="white",
            linewidth=0.4,
            alpha=0.9,
        )
        ax.plot(grid, intercept + slope * grid, color="#A84232", linewidth=1.1)
        ax.axhline(0, color="0.55", linewidth=0.6)
        ax.set_xlabel(axis_label, fontsize=7)
        if panel == 1:
            ax.set_ylabel("Mean identity effect (score points)", fontsize=7)
        ax.set_title(sector.title(), fontsize=8, pad=4)
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
        summary.append(
            {
                "sector": sector,
                "driver": variable,
                "n": len(rows),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
                "slope": slope,
                "intercept": intercept,
            }
        )

    figure_path = ROOT / "economic_drivers.pdf"
    fig.savefig(figure_path, bbox_inches="tight")
    fig.savefig(MANUSCRIPT_FIGURE, bbox_inches="tight")
    fig.savefig(figure_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    with SUMMARY.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {figure_path} and {SUMMARY}")


if __name__ == "__main__":
    main()
