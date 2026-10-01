"""Tests for brand-masking statistical analysis helpers."""

import pandas as pd
import pytest

from experiments.brand_masking.analyze_results import build_brand_masking_summary


def test_build_brand_masking_summary_computes_wilcoxon_metrics() -> None:
    """Valid paired inputs should produce effect-size and test metrics."""
    df = pd.DataFrame(
        {
            "score_named": [8, 9, 7, 8, 9],
            "score_masked": [7, 7, 6, 7, 8],
        },
    )

    summary = build_brand_masking_summary(
        df,
        run=1,
        model_id="gpt-4o",
        temperature=0.0,
        min_pairs_for_analysis=5,
    )

    assert summary.valid_pairs_n == 5
    assert summary.mean_difference == pytest.approx(1.2)
    assert summary.wilcoxon_statistic is not None
    assert summary.p_value is not None


def test_build_brand_masking_summary_handles_insufficient_data() -> None:
    """Too few valid pairs should skip the Wilcoxon calculation."""
    df = pd.DataFrame(
        {
            "score_named": [8, 9, None],
            "score_masked": [7, 7, 6],
        },
    )

    summary = build_brand_masking_summary(
        df,
        run=1,
        model_id="gpt-4o",
        temperature=0.0,
        min_pairs_for_analysis=5,
    )

    assert summary.valid_pairs_n == 2
    assert summary.wilcoxon_statistic is None
    assert summary.conclusion == "Insufficient data for reliable Wilcoxon test."


def test_build_brand_masking_summary_handles_all_zero_differences() -> None:
    """All-zero paired differences should surface a graceful test failure."""
    df = pd.DataFrame(
        {
            "score_named": [8, 8, 8, 8, 8],
            "score_masked": [8, 8, 8, 8, 8],
        },
    )

    summary = build_brand_masking_summary(
        df,
        run=1,
        model_id="gpt-4o",
        temperature=0.0,
        min_pairs_for_analysis=5,
    )

    assert summary.zero_difference_pairs_n == 5
    assert summary.wilcoxon_statistic is None
    assert summary.p_value is None
    assert summary.conclusion.startswith("Test failed:")
