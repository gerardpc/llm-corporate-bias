"""Statistics for the manuscript brand-masking analysis.

Estimand: ``delta = score_named - score_masked`` for the same (scenario, profile).
All functions are pure (DataFrame in, DataFrame/dict out) so they can be unit-tested.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

BOOTSTRAP_SEED = 20260927
N_BOOTSTRAP = 10_000
TOP_K = 3


# --------------------------------------------------------------------------- scores
def valid_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where both conditions parsed to an integer score."""
    mask = (
        df["named_parse_success"].astype(bool)
        & df["masked_parse_success"].astype(bool)
        & df["score_named"].notna()
        & df["score_masked"].notna()
    )
    out = df.loc[mask].copy()
    out["delta"] = out["score_named"].astype(float) - out["score_masked"].astype(float)
    return out


def wilcoxon_run_averaged(pairs: pd.DataFrame) -> dict:
    """Wilcoxon signed-rank on run-averaged deltas per (scenario_id, company_id).

    At temperature 0 the ten runs are near-replicates, so each (scenario, profile)
    unit contributes one averaged difference rather than ten.
    """
    unit = pairs.groupby(["scenario_id", "company_id"])["delta"].mean()
    nonzero = int((unit != 0).sum())
    if nonzero == 0:
        return {"n_units": len(unit), "n_nonzero_units": 0, "W": None, "p": None}
    res = stats.wilcoxon(unit.to_numpy(), zero_method="wilcox", alternative="two-sided")
    return {
        "n_units": int(len(unit)),
        "n_nonzero_units": nonzero,
        "W": float(res.statistic),
        "p": float(res.pvalue),
    }


def cluster_bootstrap_mean_delta(
    pairs: pd.DataFrame,
    cluster_cols: list[str],
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Percentile CI for the pair-weighted mean delta, resampling whole clusters."""
    grouped = pairs.groupby(cluster_cols)["delta"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy()
    counts = grouped["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "mean_delta": float(sums.sum() / counts.sum()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_clusters": int(len(sums)),
        "n_boot": n_boot,
        "seed": seed,
    }


# ------------------------------------------------------------------------- rankings
def company_means(pairs_run: pd.DataFrame) -> pd.DataFrame:
    """Per-company means over valid scenarios for one (sector, model, run)."""
    means = pairs_run.groupby("company_id")[["score_named", "score_masked"]].mean()
    return means.dropna()


def _ranks(values: pd.Series) -> pd.Series:
    """Rank 1 = highest mean; ties share the average rank."""
    return values.rank(ascending=False, method="average")


def _top_set(values: pd.Series, k: int) -> set[str]:
    """Companies whose competition ('min') rank is <= k; ties at the cut are kept."""
    r = values.rank(ascending=False, method="min")
    return set(r[r <= k].index)


def ranking_metrics(means: pd.DataFrame, k: int = TOP_K) -> dict:
    """Rank-change, pairwise-reversal, top-k and correlation metrics for one ranking.

    The comparison is Masked -> Named (what changes when identity is added).
    """
    named, masked = means["score_named"], means["score_masked"]
    n = len(means)
    out: dict = {"n_companies": n}
    if n < 2:
        return out
    abs_change = (_ranks(named) - _ranks(masked)).abs()
    out.update(
        mean_abs_rank_change=float(abs_change.mean()),
        median_abs_rank_change=float(abs_change.median()),
        max_abs_rank_change=float(abs_change.max()),
        mean_abs_rank_change_norm=float(abs_change.mean() / (n - 1)),
    )
    strict = ties_named_only = ties_masked_only = ties_both = 0
    ids = list(means.index)
    for a, b in combinations(ids, 2):
        sn = np.sign(named[a] - named[b])
        sm = np.sign(masked[a] - masked[b])
        if sn != 0 and sm != 0 and sn != sm:
            strict += 1
        elif sn == 0 and sm != 0:
            ties_named_only += 1  # pair enters a tie when identity is added
        elif sn != 0 and sm == 0:
            ties_masked_only += 1  # pair leaves a tie when identity is added
        elif sn == 0 and sm == 0:
            ties_both += 1
    n_pairs = n * (n - 1) // 2
    out.update(
        n_company_pairs=n_pairs,
        strict_reversals=strict,
        pct_strict_reversals=100 * strict / n_pairs,
        pairs_entering_tie=ties_named_only,
        pairs_leaving_tie=ties_masked_only,
        pairs_tied_both=ties_both,
    )
    top1_n, top1_m = _top_set(named, 1), _top_set(masked, 1)
    topk_n, topk_m = _top_set(named, k), _top_set(masked, k)
    out.update(
        top1_named=sorted(top1_n),
        top1_masked=sorted(top1_m),
        top1_changed=top1_n != top1_m,
        topk_jaccard=len(topk_n & topk_m) / len(topk_n | topk_m),
        topk_entries=sorted(topk_n - topk_m),
        topk_exits=sorted(topk_m - topk_n),
    )
    rho = stats.spearmanr(named, masked).statistic
    tau = stats.kendalltau(named, masked, variant="b").statistic
    out.update(
        spearman_rho=None if np.isnan(rho) else float(rho),
        kendall_tau_b=None if np.isnan(tau) else float(tau),
    )
    return out


RANK_SUMMARY_COLS = (
    "mean_abs_rank_change",
    "mean_abs_rank_change_norm",
    "max_abs_rank_change",
    "pct_strict_reversals",
    "top1_changed",
    "topk_jaccard",
    "spearman_rho",
    "kendall_tau_b",
)


def summarize_over_runs(
    per_run: pd.DataFrame,
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Mean / median across runs with a percentile bootstrap CI (runs resampled)."""
    rng = np.random.default_rng(seed)
    out: dict = {"n_runs": len(per_run)}
    for col in RANK_SUMMARY_COLS:
        vals = per_run[col].dropna().astype(float).to_numpy()
        if len(vals) == 0:
            continue
        boot = vals[rng.integers(0, len(vals), size=(n_boot, len(vals)))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[f"{col}_mean"] = float(vals.mean())
        out[f"{col}_median"] = float(np.median(vals))
        out[f"{col}_ci_low"] = float(lo)
        out[f"{col}_ci_high"] = float(hi)
    return out
