"""Reproducible manuscript analysis for Experiment 2 (brand masking).

Usage (from the repository root)::

    python -m experiments.brand_masking.manuscript_analysis.run_analysis [--offline]

Pulls deduplicated paired trials per study window (``sql/extract_trials.sql``),
snapshots them to ``data/brand_masking/manuscript_snapshot/`` (git-ignored) and writes
machine-readable results to ``outputs/`` and figures/tables to the manuscript folder.
``--offline`` re-runs everything from the snapshot without touching the database.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aily_bias_in_llms.config.project_paths import DATA_DIR, ROOT_DIR
from experiments.brand_masking.manuscript_analysis import metrics
from experiments.brand_masking.manuscript_analysis import prompt_equivalence as pe
from experiments.brand_masking.manuscript_analysis.study_windows import (
    RUNS_PER_MODEL,
    STUDY_WINDOWS,
    StudyWindow,
    render_sql,
)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
SNAPSHOT_DIR = DATA_DIR / "brand_masking" / "manuscript_snapshot"
MANUSCRIPT_DIR = ROOT_DIR / "manuscript" / "satml_manuscript"
PROMPT_SAMPLE_N = 200
PROMPT_SAMPLE_SEED = "satml-2026"
# A group is "complete" if every expected (run, scenario, company) row exists in the
# window; observed groups with narrower coverage remain reportable and are flagged.
PARSE_LOSS_FLAG = 0.05
# Groups missing at most this fraction of expected rows (full catalog present) are
# labelled observed_minor_gap. Other observed groups are labelled by coverage.
MINOR_GAP_FRACTION = 0.01

MODEL_LABELS = {
    "anthropic.claude-haiku-4-5-20251001-v1:0": "Claude Haiku 4.5",
    "anthropic.claude-sonnet-4-5-20250929-v1:0": "Claude Sonnet 4.5",
    "anthropic.claude-opus-5": "Claude Opus 5",
    "gpt-5": "GPT-5",
    "gpt-5-mini": "GPT-5 Mini",
    "gpt-5.6-luna": "GPT-5.6 Luna",
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "openai.gpt-5.6-luna": "GPT-5.6 Luna (openai.)",
    "openai.gpt-5.6-sol": "GPT-5.6 Sol (openai.)",
    "qwen/qwen3.8-max-0902": "Qwen3.8 Max",
    "deepseek/deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
    "z-ai/glm-5.3-flash": "GLM-5.3 Flash",
}
SECTORS = ("pharma", "consulting", "cloud", "banking")


def label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id)


# ----------------------------------------------------------------------- extraction
def _snapshot_path(w: StudyWindow) -> Path:
    return SNAPSHOT_DIR / f"{w.window}_{w.sector}.parquet"


def load_trials(w: StudyWindow, *, offline: bool) -> pd.DataFrame:
    path = _snapshot_path(w)
    if offline:
        return pd.read_parquet(path)
    from aily_bias_in_llms.database.database import query_db

    df = query_db(render_sql("extract_trials.sql", w))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


# ------------------------------------------------------------------------- coverage
def coverage(w: StudyWindow, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_id, g in df.groupby("model_id"):
        expected = RUNS_PER_MODEL * w.expected_pairs_per_run
        n_valid = len(metrics.valid_pairs(g))
        companies = set(g["company_id"])
        desc_versions = g.groupby("company_id")["description_md5"].nunique().max()
        scen_versions = g.groupby("scenario_id")["scenario_md5"].nunique().max()
        mask_versions = g.groupby("company_id")["masked_name"].nunique().max()
        complete_rows = (
            len(g) == expected
            and g["run"].nunique() == RUNS_PER_MODEL
            and companies == set(w.companies)
        )
        consistent = desc_versions == 1 and scen_versions == 1 and mask_versions == 1
        parse_rate = n_valid / len(g)
        minor_gap = (
            companies == set(w.companies)
            and 0 < expected - len(g) <= MINOR_GAP_FRACTION * expected
        )
        if minor_gap:
            # all companies present, only a handful of trials missing: reported, flagged
            status = "observed_minor_gap"
        elif not complete_rows:
            status = (
                "observed_partial_catalog"
                if g["run"].nunique() == RUNS_PER_MODEL and len(companies) > 0
                else "observed_limited"
            )
        elif not consistent:
            status = "inconsistent_profiles"
        elif 1 - parse_rate > PARSE_LOSS_FLAG:
            status = "complete_high_parse_loss"
        else:
            status = "complete"
        rows.append(
            {
                "window": w.window,
                "sector": w.sector,
                "model_id": model_id,
                "model": label(model_id),
                "runs": int(g["run"].nunique()),
                "companies_present": len(companies),
                "companies_catalog": len(w.companies),
                "rows": len(g),
                "rows_expected": expected,
                "missing_rows": expected - len(g),
                "valid_pairs": n_valid,
                "parse_failures": len(g) - n_valid,
                "named_parse_failures": int((~g["named_parse_success"].astype(bool)).sum()),
                "masked_parse_failures": int((~g["masked_parse_success"].astype(bool)).sum()),
                "parse_rate": parse_rate,
                "dedup_keys_with_multiple_versions": int((g["n_versions"] > 1).sum()),
                "max_profile_versions_per_company": int(desc_versions),
                "max_scenario_versions_per_id": int(scen_versions),
                "max_masked_labels_per_company": int(mask_versions),
                "first_created_at": str(g["created_at"].min()),
                "last_created_at": str(g["created_at"].max()),
                "status": status,
                "window_note": w.note,
            },
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- scores
def score_effects(w: StudyWindow, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_id, g in df.groupby("model_id"):
        p = metrics.valid_pairs(g)
        if p.empty:
            continue
        boot_co = metrics.cluster_bootstrap_mean_delta(p, ["company_id"])
        boot_sc = metrics.cluster_bootstrap_mean_delta(p, ["scenario_id", "company_id"])
        wil = metrics.wilcoxon_run_averaged(p)
        rows.append(
            {
                "window": w.window,
                "sector": w.sector,
                "model_id": model_id,
                "model": label(model_id),
                "n_pairs": len(p),
                "mean_named": p["score_named"].mean(),
                "mean_masked": p["score_masked"].mean(),
                "mean_delta": boot_co["mean_delta"],
                "ci_low_company": boot_co["ci_low"],
                "ci_high_company": boot_co["ci_high"],
                "n_company_clusters": boot_co["n_clusters"],
                "ci_low_scen_company": boot_sc["ci_low"],
                "ci_high_scen_company": boot_sc["ci_high"],
                "pct_pairs_named_higher": 100 * (p["delta"] > 0).mean(),
                "pct_pairs_equal": 100 * (p["delta"] == 0).mean(),
                "pct_pairs_masked_higher": 100 * (p["delta"] < 0).mean(),
                "pct_masked_at_ceiling_10": 100 * (p["score_masked"] == 10).mean(),
                "wilcoxon_n_units": wil["n_units"],
                "wilcoxon_nonzero_units": wil["n_nonzero_units"],
                "wilcoxon_W": wil["W"],
                "wilcoxon_p": wil["p"],
            },
        )
    return pd.DataFrame(rows)


def company_effects(w: StudyWindow, df: pd.DataFrame) -> pd.DataFrame:
    p = metrics.valid_pairs(df)
    out = (
        p.groupby(["model_id", "company_id", "incumbent_name"])["delta"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_delta", "count": "n_pairs"})
    )
    out.insert(0, "sector", w.sector)
    out.insert(0, "window", w.window)
    return out


# ------------------------------------------------------------------------- rankings
def _join(v) -> str:
    return ",".join(v) if isinstance(v, list) else ""


def ranking_effects(w: StudyWindow, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_run_rows, summary_rows = [], []
    p = metrics.valid_pairs(df)
    for model_id, g in p.groupby("model_id"):
        runs = []
        for run, gr in g.groupby("run"):
            m = metrics.ranking_metrics(metrics.company_means(gr))
            m.update(window=w.window, sector=w.sector, model_id=model_id, run=int(run))
            runs.append(m)
        per_run = pd.DataFrame(runs)
        per_run_rows.append(per_run)
        if "spearman_rho" not in per_run:
            continue
        s = metrics.summarize_over_runs(per_run)
        s.update(
            window=w.window,
            sector=w.sector,
            model_id=model_id,
            model=label(model_id),
            n_companies_median=float(per_run["n_companies"].median()),
            top1_masked_modal=per_run["top1_masked"].map(_join).mode().iloc[0],
            top1_named_modal=per_run["top1_named"].map(_join).mode().iloc[0],
        )
        summary_rows.append(s)
    return pd.concat(per_run_rows, ignore_index=True), pd.DataFrame(summary_rows)


# ------------------------------------------------------------------ prompt audit
def prompt_audit(windows: list[StudyWindow], *, offline: bool) -> dict:
    templates = pe.load_templates()
    report: dict = {
        "templates_path": str(pe.PROMPTS_PATH.relative_to(ROOT_DIR)),
        "placeholder": pe.PLACEHOLDER,
        "template_check": pe.template_check(templates).to_dict(),
        "rendered_checks": [],
        "db_sample_checks": [],
        "db_full_checks": [],
    }
    for w in windows:
        cat = pd.read_parquet(_snapshot_path(w))
        rep = cat.sort_values(["scenario_id", "company_id"]).drop_duplicates("company_id")
        report["rendered_checks_source"] = "snapshot identity fields + fixed profile text"
        for _, r in rep.head(3).iterrows():
            named, masked = pe.render_pair(
                templates,
                scenario=f"<scenario {r.scenario_id}>",
                incumbent_name=r.incumbent_name,
                masked_name=r.masked_name,
                description="<capability profile>",
            )
            res = pe.compare_prompts(
                named, masked,
                incumbent_name=r.incumbent_name, masked_name=r.masked_name,
                label=f"{w.window}/{w.sector}/{r.company_id}",
                system_named=templates["system_prompt"],
                system_masked=templates["system_prompt"],
            )
            report["rendered_checks"].append(res.to_dict())
    if offline:
        cached = OUT_DIR / "prompt_equivalence.json"
        if cached.exists():
            old = json.loads(cached.read_text())
            report["db_sample_checks"] = old.get("db_sample_checks", [])
            report["db_full_checks"] = old.get("db_full_checks", [])
        return _finalise_audit(report)
    from aily_bias_in_llms.database.database import query_db

    for w in windows:
        sample = query_db(
            render_sql("sample_prompts.sql", w, seed=PROMPT_SAMPLE_SEED, n=PROMPT_SAMPLE_N),
        )
        results = []
        for _, r in sample.iterrows():
            res = pe.compare_prompts(
                r.prompt_named, r.prompt_masked,
                incumbent_name=r.incumbent_name, masked_name=r.masked_name,
                label=f"id={r.id}",
                system_named=r.system_prompt, system_masked=r.system_prompt,
            )
            # the persisted prompt must also be what the template would render
            expected_named, expected_masked = pe.render_pair(
                templates,
                scenario=r.base_scenario_text,
                incumbent_name=r.incumbent_name,
                masked_name=r.masked_name,
                description=r.company_description,
            )
            res_d = res.to_dict()
            res_d["matches_template_render"] = (
                expected_named == r.prompt_named and expected_masked == r.prompt_masked
            )
            res_d["system_prompt_matches_template"] = (
                r.system_prompt == templates["system_prompt"]
            )
            results.append(res_d)
        failures = [
            x for x in results
            if not (x["equivalent"] and x["matches_template_render"]
                    and x["system_prompt_matches_template"])
        ]
        report["db_sample_checks"].append(
            {
                "window": w.window, "sector": w.sector, "n_sampled": len(results),
                "seed": PROMPT_SAMPLE_SEED, "n_failures": len(failures),
                "failures": failures[:20],
                "example_raw_diff": results[0]["raw_differing_lines"] if results else [],
            },
        )
        full = query_db(render_sql("prompt_equivalence_full.sql", w)).iloc[0].to_dict()
        full = {k: int(v) for k, v in full.items()}
        full.update(window=w.window, sector=w.sector)
        report["db_full_checks"].append(full)
    return _finalise_audit(report)


def _finalise_audit(report: dict) -> dict:
    ok = report["template_check"]["equivalent"]
    ok &= all(x["equivalent"] for x in report["rendered_checks"])
    ok &= all(x["n_failures"] == 0 for x in report["db_sample_checks"])
    ok &= all(
        x["n_equivalent"] == x["n_rows"] == x["n_profile_in_both"] == x["n_scenario_in_both"]
        and x["n_system_prompts"] == 1
        for x in report["db_full_checks"]
    )
    report["all_passed"] = bool(ok)
    return report


# -------------------------------------------------------------------------- figures
def _fmt_p(p: float | None) -> str:
    if p is None or np.isnan(p):
        return "--"
    return r"$<0.001$" if p < 0.001 else f"${p:.3f}$"


def figure_deltas(scores: pd.DataFrame, cov: pd.DataFrame, window: str, path: Path) -> None:
    s = scores.merge(cov[["window", "sector", "model_id", "status"]])
    s = s[(s.window == window) & s.status.notna()]
    sectors = [x for x in SECTORS if x in set(s.sector)]
    fig, axes = plt.subplots(1, len(sectors), figsize=(3.2 * len(sectors), 3.6), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, sec in zip(axes, sectors, strict=True):
        d = s[s.sector == sec].sort_values("mean_delta")
        y = np.arange(len(d))
        err = [d.mean_delta - d.ci_low_company, d.ci_high_company - d.mean_delta]
        colors = ["tab:orange" if st == "complete_high_parse_loss" else "tab:blue" for st in d.status]
        ax.errorbar(d.mean_delta, y, xerr=err, fmt="none", ecolor="grey", capsize=2)
        ax.scatter(d.mean_delta, y, c=colors, zorder=3, s=18)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y, d.model)
        ax.set_title(sec.capitalize())
        ax.set_xlabel(r"$\Delta$ = Named $-$ Masked")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_companies(comp: pd.DataFrame, window: str, path: Path) -> None:
    c = comp[comp.window == window]
    c = (c.assign(w=c.mean_delta * c.n_pairs)
         .groupby(["sector", "incumbent_name"]).agg(w=("w", "sum"), n=("n_pairs", "sum")))
    c = (c.w / c.n).rename("mean_delta").reset_index()
    sectors = [x for x in SECTORS if x in set(c.sector)]
    fig, axes = plt.subplots(1, len(sectors), figsize=(3.4 * len(sectors), 4.2))
    for ax, sec in zip(np.atleast_1d(axes), sectors, strict=True):
        d = c[c.sector == sec].sort_values("mean_delta")
        ax.barh(d.incumbent_name, d.mean_delta,
                color=["tab:red" if v > 0 else "tab:blue" for v in d.mean_delta])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(sec.capitalize())
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel(r"mean $\Delta$ (pooled over models)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_ranks(
    rank_sum: pd.DataFrame,
    cov: pd.DataFrame,
    window: str,
    path: Path,
) -> None:
    r = rank_sum.merge(cov[["window", "sector", "model_id", "status"]])
    r = r[(r.window == window) & r.status.notna()]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for sec, col in zip(SECTORS, plt.cm.tab10.colors, strict=False):
        e = r[r.sector == sec]
        axes[0].scatter(e.spearman_rho_mean, e.pct_strict_reversals_mean,
                        marker="o", color=col, label=sec, s=22)
        axes[1].scatter(e.mean_abs_rank_change_mean, e.topk_jaccard_mean,
                        marker="o", color=col, s=22)
    axes[0].set_xlabel(r"Spearman $\rho$ (Masked vs Named company means)")
    axes[0].set_ylabel("% company pairs strictly reversed")
    axes[1].set_xlabel("mean |rank change|")
    axes[1].set_ylabel("top-3 Jaccard overlap")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- tables
def latex_score_table(
    scores: pd.DataFrame, cov: pd.DataFrame, window: str, exclude: tuple[str, ...] = (),
) -> str:
    s = scores.merge(cov[["window", "sector", "model_id", "status", "parse_rate"]])
    s = s[s.window == window]
    lines = []
    for sec in SECTORS:
        d = s[s.sector == sec].sort_values("model")
        if d.empty or sec in exclude:
            continue
        for _, r in d.iterrows():
            flag = {
                "observed_partial_catalog": r"$^{\dagger}$",
                "observed_limited": r"$^{\ddagger}$",
                "observed_minor_gap": r"$^{\S}$",
                "complete_high_parse_loss": r"$^{\P}$",
            }.get(r.status, "")
            w = "--" if r.wilcoxon_W is None or np.isnan(r.wilcoxon_W) else f"{r.wilcoxon_W:,.1f}"
            lines.append(
                f"{r.model}{flag} & {sec.capitalize()} & {r.mean_named:.2f} & {r.mean_masked:.2f} & "
                f"${r.mean_delta:+.2f}$ & $[{r.ci_low_company:+.2f}, {r.ci_high_company:+.2f}]$ & "
                f"{w} & {_fmt_p(r.wilcoxon_p)} \\\\",
            )
        lines.append(r"\midrule")
    if lines and lines[-1] == r"\midrule":
        lines.pop()
    return "\n".join(lines) + "\n"


def latex_rank_table(rank_sum: pd.DataFrame, cov: pd.DataFrame, window: str) -> str:
    r = rank_sum.merge(cov[["window", "sector", "model_id", "status"]])
    r = r[(r.window == window) & r.status.notna()]
    lines = []
    for sec in SECTORS:
        d = r[r.sector == sec].sort_values("model")
        for _, x in d.iterrows():
            lines.append(
                f"{x.model} & {sec.capitalize()} & {int(x.n_companies_median)} & "
                f"{x.mean_abs_rank_change_mean:.2f} & {x.max_abs_rank_change_mean:.1f} & "
                f"{x.pct_strict_reversals_mean:.1f} $[{x.pct_strict_reversals_ci_low:.1f}, {x.pct_strict_reversals_ci_high:.1f}]$ & "
                f"{100 * x.top1_changed_mean:.0f} & {x.topk_jaccard_mean:.2f} & "
                f"{x.spearman_rho_mean:.2f} $[{x.spearman_rho_ci_low:.2f}, {x.spearman_rho_ci_high:.2f}]$ & "
                f"{x.kendall_tau_b_mean:.2f} \\\\",
            )
        if not d.empty:
            lines.append(r"\midrule")
    if lines and lines[-1] == r"\midrule":
        lines.pop()
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- main
def main(offline: bool = False) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    covs, scores, comps, per_runs, rank_sums = [], [], [], [], []
    for w in STUDY_WINDOWS:
        df = load_trials(w, offline=offline)
        covs.append(coverage(w, df))
        scores.append(score_effects(w, df))
        comps.append(company_effects(w, df))
        pr, rs = ranking_effects(w, df)
        per_runs.append(pr)
        rank_sums.append(rs)
    cov = pd.concat(covs, ignore_index=True)
    sc = pd.concat(scores, ignore_index=True)
    comp = pd.concat(comps, ignore_index=True)
    per_run = pd.concat(per_runs, ignore_index=True)
    rank_sum = pd.concat(rank_sums, ignore_index=True)

    cov.to_csv(OUT_DIR / "coverage.csv", index=False)
    sc.merge(cov[["window", "sector", "model_id", "status"]]).to_csv(
        OUT_DIR / "score_effects.csv", index=False,
    )
    comp.to_csv(OUT_DIR / "company_effects.csv", index=False)
    for col in ("top1_named", "top1_masked", "topk_entries", "topk_exits"):
        per_run[col] = per_run[col].map(lambda v: ";".join(v) if isinstance(v, list) else v)
    per_run.to_csv(OUT_DIR / "ranking_per_run.csv", index=False)
    rank_sum.merge(cov[["window", "sector", "model_id", "status"]]).to_csv(
        OUT_DIR / "ranking_summary.csv", index=False,
    )

    audit = prompt_audit(list(STUDY_WINDOWS), offline=offline)
    (OUT_DIR / "prompt_equivalence.json").write_text(json.dumps(audit, indent=2))

    params = {
        "bootstrap_seed": metrics.BOOTSTRAP_SEED,
        "n_bootstrap": metrics.N_BOOTSTRAP,
        "bootstrap_method": "percentile; resampling whole clusters with replacement",
        "primary_cluster": "company_id",
        "sensitivity_cluster": "(scenario_id, company_id)",
        "wilcoxon_unit": "run-averaged delta per (scenario_id, company_id); zero_method=wilcox",
        "ranking_unit": "(sector, model_id, run); company means over valid scenarios",
        "tie_policy": "average ranks for rank change; competition (min) rank <= k for top-k, ties at the cut kept",
        "top_k": metrics.TOP_K,
        "parse_loss_flag": PARSE_LOSS_FLAG,
        "minor_gap_fraction": MINOR_GAP_FRACTION,
        "windows": [
            {"window": w.window, "sector": w.sector, "start": w.start, "end_exclusive": w.end,
             "n_companies": len(w.companies), "companies": list(w.companies), "note": w.note}
            for w in STUDY_WINDOWS
        ],
    }
    (OUT_DIR / "analysis_parameters.json").write_text(json.dumps(params, indent=2))

    figure_deltas(sc, cov, "may_core", MANUSCRIPT_DIR / "fig_4.pdf")
    figure_deltas(sc, cov, "sep_sweep", MANUSCRIPT_DIR / "fig_bm_sep_models.pdf")
    figure_companies(comp, "may_core", MANUSCRIPT_DIR / "fig_5.pdf")
    figure_ranks(rank_sum, cov, "sep_sweep", MANUSCRIPT_DIR / "fig_bm_ranks.pdf")
    tables = MANUSCRIPT_DIR / "tables"
    tables.mkdir(exist_ok=True)
    (tables / "bm_scores_may.tex").write_text(latex_score_table(sc, cov, "may_core"))
    (tables / "bm_scores_sep.tex").write_text(latex_score_table(sc, cov, "sep_sweep"))
    (tables / "bm_ranks_may.tex").write_text(latex_rank_table(rank_sum, cov, "may_core"))
    (tables / "bm_ranks_sep.tex").write_text(latex_rank_table(rank_sum, cov, "sep_sweep"))
    print(cov[["window", "sector", "model", "rows", "missing_rows", "parse_rate", "status"]].to_string())
    print("prompt audit passed:", audit["all_passed"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    main(offline=parser.parse_args().offline)
