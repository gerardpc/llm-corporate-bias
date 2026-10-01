"""Statistical analysis helpers for the brand-masking experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bias_in_llms.config.project_paths import DATA_DIR, EXPERIMENTS_DIR
from bias_in_llms.database.brand_masking_db import (
    insert_brand_masking_summary,
    select_brand_masking_trials,
)
from experiments.brand_masking.config_models import load_brand_masking_config


@dataclass(frozen=True)
class BrandMaskingSummary:
    """Analysis summary for one model/run pair."""

    run: int
    model_id: str
    temperature: float
    valid_pairs_n: int
    dropped_pairs_n: int
    zero_difference_pairs_n: int
    mean_score_named: float | None
    mean_score_masked: float | None
    mean_difference: float | None
    wilcoxon_statistic: float | None
    p_value: float | None
    conclusion: str


def build_brand_masking_summary(
    df: pd.DataFrame,
    *,
    run: int,
    model_id: str,
    temperature: float,
    min_pairs_for_analysis: int,
) -> BrandMaskingSummary:
    """Compute Wilcoxon-based summary metrics for a model/run slice."""
    clean_df = df.copy()
    clean_df["score_named"] = pd.to_numeric(clean_df["score_named"], errors="coerce")
    clean_df["score_masked"] = pd.to_numeric(clean_df["score_masked"], errors="coerce")
    clean_df = clean_df.dropna(subset=["score_named", "score_masked"]).copy()

    valid_pairs_n = len(clean_df)
    dropped_pairs_n = len(df) - valid_pairs_n
    zero_difference_pairs_n = 0

    if valid_pairs_n == 0:
        return BrandMaskingSummary(
            run=run,
            model_id=model_id,
            temperature=temperature,
            valid_pairs_n=0,
            dropped_pairs_n=dropped_pairs_n,
            zero_difference_pairs_n=0,
            mean_score_named=None,
            mean_score_masked=None,
            mean_difference=None,
            wilcoxon_statistic=None,
            p_value=None,
            conclusion="No valid paired scores were available for analysis.",
        )

    scores_named = clean_df["score_named"].to_numpy(dtype=float)
    scores_masked = clean_df["score_masked"].to_numpy(dtype=float)
    score_differences = scores_named - scores_masked
    zero_difference_pairs_n = int(np.sum(score_differences == 0))

    mean_score_named = float(np.mean(scores_named))
    mean_score_masked = float(np.mean(scores_masked))
    mean_difference = float(mean_score_named - mean_score_masked)

    if zero_difference_pairs_n == valid_pairs_n:
        return BrandMaskingSummary(
            run=run,
            model_id=model_id,
            temperature=temperature,
            valid_pairs_n=valid_pairs_n,
            dropped_pairs_n=dropped_pairs_n,
            zero_difference_pairs_n=zero_difference_pairs_n,
            mean_score_named=mean_score_named,
            mean_score_masked=mean_score_masked,
            mean_difference=mean_difference,
            wilcoxon_statistic=None,
            p_value=None,
            conclusion="Test failed: all paired differences are zero.",
        )

    if valid_pairs_n < min_pairs_for_analysis:
        return BrandMaskingSummary(
            run=run,
            model_id=model_id,
            temperature=temperature,
            valid_pairs_n=valid_pairs_n,
            dropped_pairs_n=dropped_pairs_n,
            zero_difference_pairs_n=zero_difference_pairs_n,
            mean_score_named=mean_score_named,
            mean_score_masked=mean_score_masked,
            mean_difference=mean_difference,
            wilcoxon_statistic=None,
            p_value=None,
            conclusion="Insufficient data for reliable Wilcoxon test.",
        )

    try:
        statistic, p_value = stats.wilcoxon(
            scores_named,
            scores_masked,
            alternative="two-sided",
            zero_method="wilcox",
        )
        conclusion = (
            "Significant (Reject Null)"
            if p_value < 0.05
            else "Not Significant (Fail to Reject Null)"
        )
        return BrandMaskingSummary(
            run=run,
            model_id=model_id,
            temperature=temperature,
            valid_pairs_n=valid_pairs_n,
            dropped_pairs_n=dropped_pairs_n,
            zero_difference_pairs_n=zero_difference_pairs_n,
            mean_score_named=mean_score_named,
            mean_score_masked=mean_score_masked,
            mean_difference=mean_difference,
            wilcoxon_statistic=float(statistic),
            p_value=float(p_value),
            conclusion=conclusion,
        )
    except ValueError as exc:
        return BrandMaskingSummary(
            run=run,
            model_id=model_id,
            temperature=temperature,
            valid_pairs_n=valid_pairs_n,
            dropped_pairs_n=dropped_pairs_n,
            zero_difference_pairs_n=zero_difference_pairs_n,
            mean_score_named=mean_score_named,
            mean_score_masked=mean_score_masked,
            mean_difference=mean_difference,
            wilcoxon_statistic=None,
            p_value=None,
            conclusion=f"Test failed: {exc}",
        )


def summarize_brand_masking_trials(
    min_pairs_for_analysis: int,
    sector: str,
) -> pd.DataFrame:
    """Compute and persist Wilcoxon summaries for all raw brand-masking trials."""
    trials_df = select_brand_masking_trials(sector)
    if trials_df.empty:
        return pd.DataFrame()

    summary_rows: list[dict] = []
    grouped = trials_df.groupby(["run", "model_id", "temperature"], dropna=False)
    for (run, model_id, temperature), group_df in grouped:
        summary = build_brand_masking_summary(
            group_df,
            run=int(run),
            model_id=str(model_id),
            temperature=float(temperature),
            min_pairs_for_analysis=min_pairs_for_analysis,
        )
        insert_brand_masking_summary(**asdict(summary), sector=sector)
        summary_rows.append(asdict(summary))

    summary_df = pd.DataFrame(summary_rows)
    output_dir = (
        DATA_DIR / "brand_masking" / sector / datetime.now().strftime("%Y%m%d_%H%M")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "analysis_summary.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"✓ Analysis summary saved to {output_path}")
    return summary_df


def main(config_path: Path | None = None) -> pd.DataFrame:
    """Run post-hoc analysis using the configured minimum pair threshold."""
    resolved_config_path = config_path or (
        EXPERIMENTS_DIR / "brand_masking" / "config" / "pharma" / "config.yaml"
    )
    config = load_brand_masking_config(resolved_config_path)
    return summarize_brand_masking_trials(config.min_pairs_for_analysis, config.sector)


if __name__ == "__main__":
    main()
