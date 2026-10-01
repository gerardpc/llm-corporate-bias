#!/usr/bin/env python3
"""Reproduce the economic-scale analysis and write economic-scale-report.md.

The supplied ranking_summary_postgres.csv is an aggregate ranking-stability file;
it has no company identifier. Company-level Experiment 2 deltas therefore come
from the accompanying manuscript-analysis/company_effects.csv, while the
ranking summary is reported as a robustness check.
"""
from __future__ import annotations
import csv, math, random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
EFFECTS = REPO / "experiments/brand_masking/manuscript_analysis/outputs/company_effects.csv"
RANKS = ROOT.parent / "ranking_summary_postgres.csv"
ECON = ROOT / "public_economic_data.csv"
REPORT = ROOT / "economic-scale-report.md"

def read(path):
    with path.open(newline="") as f: return list(csv.DictReader(f))
def canon(x): return {"Merck & Co.": "Merck & Co", "OVHcloud": "OVH Cloud"}.get(x, x)
def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i]); out = [0.0]*len(xs); i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j+1]] == xs[order[i]]: j += 1
        for k in range(i, j+1): out[order[k]] = (i+j)/2 + 1
        i = j + 1
    return out
def corr(x, y):
    mx, my = sum(x)/len(x), sum(y)/len(y)
    a = sum((u-mx)*(v-my) for u,v in zip(x,y)); b = math.sqrt(sum((u-mx)**2 for u in x)*sum((v-my)**2 for v in y))
    return a/b if b else float("nan")
def permutation_p(x, y, n=50000):
    observed = abs(corr(x, y)); rng = random.Random(20260929); hits = 0; z = list(y)
    for _ in range(n):
        rng.shuffle(z)
        if abs(corr(x, z)) >= observed - 1e-12: hits += 1
    return (hits + 1) / (n + 1)
def ols_r2(y, columns):
    X = [[1.0] + [c[i] for c in columns] for i in range(len(y))]; p = len(X[0])
    A = [[sum(X[i][j]*X[i][k] for i in range(len(y))) for k in range(p)] for j in range(p)]
    b = [sum(X[i][j]*y[i] for i in range(len(y))) for j in range(p)]
    for c in range(p):
        q = max(range(c,p), key=lambda q: abs(A[q][c])); A[c],A[q] = A[q],A[c]; b[c],b[q] = b[q],b[c]
        for q in range(c+1,p):
            z = A[q][c]/A[c][c]
            for k in range(c,p): A[q][k] -= z*A[c][k]
            b[q] -= z*b[c]
    beta = [0.0]*p
    for i in range(p-1,-1,-1): beta[i] = (b[i]-sum(A[i][j]*beta[j] for j in range(i+1,p)))/A[i][i]
    pred = [sum(beta[j]*X[i][j] for j in range(p)) for i in range(len(y))]; mean = sum(y)/len(y)
    return 1-sum((a-b)**2 for a,b in zip(y,pred))/sum((a-mean)**2 for a in y)

def main():
    effects = read(EFFECTS); economic = {canon(r["company"]): r for r in read(ECON)}
    grouped = defaultdict(list)
    for r in effects:
        if r["window"] == "may_core": grouped[(r["sector"], canon(r["incumbent_name"]))].append(float(r["mean_delta"]))
    results = {}
    for sector in ("pharma", "cloud", "consulting", "banking"):
        rows = []
        for (s, company), values in grouped.items():
            if s != sector or company not in economic: continue
            e = economic[company]
            if not e["market_cap_usd_billion"] and not e["revenue_ttm_usd_billion"]: continue
            rows.append((company, sum(values)/len(values), float(e["market_cap_usd_billion"]) if e["market_cap_usd_billion"] else None, float(e["revenue_ttm_usd_billion"]) if e["revenue_ttm_usd_billion"] else None))
        results[sector] = rows
    rank_rows = read(RANKS)
    rank_summary = {}
    for sector in ("pharma", "cloud", "consulting", "banking"):
        rr = [r for r in rank_rows if r["study_window"] == "may_core" and r["sector"] == sector and r["status"] == "rankable"]
        rank_summary[sector] = {k: sum(float(r[k]) for r in rr)/len(rr) for k in ("mean_abs_rank_change", "pct_pair_reversals", "top3_jaccard", "spearman_rho")}
    lines = ["# Economic-scale analysis of Experiment 2", "", "## Bottom line", "", "Economic scale is not a single cross-sector explanation for the name premium. With the expanded public snapshot, log market cap is positive in pharma (r=0.53, permutation p=.045), negative in banking (r=-0.40, p=.211), and weak in cloud (r=0.22, p=.567). Consulting revenue is negatively associated (r=-0.63, p=.023) but comes from only seven private-network observations with mixed fiscal years. Treat all associations as exploratory, not causal.", "", "## Data and estimand", "", "- Outcome: company-level `mean_delta = Named - Masked`, averaged over the May-core model rows in `company_effects.csv`.", "- The supplied `ranking_summary_postgres.csv` has 50 model-sector aggregates and no company identifier; it is used below as a ranking-stability check, not as the economic regression target.", "- Public economic variables are parent-company market capitalization and TTM revenue (USD billions), collected from CompaniesMarketCap and saved in `public_economic_data.csv` with source URLs and retrieval dates. Parent values are proxies for branded subsidiaries (for example, AWS uses Amazon).", "- For private consulting networks, publicly reported global-network revenue is included as a revenue-only observation; no market cap is imputed.", "- We use log10(capitalization) and log10(revenue), Pearson/Spearman correlations, 50,000-label permutation p-values, and an exploratory two-variable OLS model.", "", "## Correlations by sector", "", "`n` is shown as market-cap observations / revenue observations. The two-variable model uses only companies with both variables.", "", "| Sector | n (cap/revenue) | log market cap: r / rho / p | log revenue: r / rho / p | OLS R² (cap + revenue) |", "|---|---:|---:|---:|---:|"]
    for sector in ("pharma", "cloud", "consulting", "banking"):
        rows = results[sector]; n = len(rows)
        cap_rows = [r for r in rows if r[2] is not None]; rev_rows = [r for r in rows if r[3] is not None]; both = [r for r in rows if r[2] is not None and r[3] is not None]
        def one(var_rows, idx):
            if len(var_rows) < 3: return "not estimable"
            y = [r[1] for r in var_rows]; x = [math.log10(r[idx]) for r in var_rows]
            return f"{corr(x,y):.2f} / {corr(ranks(x),ranks(y)):.2f} / {permutation_p(x,y):.3f}"
        r2 = "not estimable" if len(both) < 3 else f"{ols_r2([r[1] for r in both], [[math.log10(r[2]) for r in both], [math.log10(r[3]) for r in both]]):.2f}"
        lines.append(f"| {sector} | {len(cap_rows)}/{len(rev_rows)} | {one(cap_rows, 2)} | {one(rev_rows, 3)} | {r2} |")
    lines += ["", "## Plots", "", "The figures below show the company-level May-core Δ values against log economic scale. Red lines are within-sector OLS fits; labels identify companies.", "", "![Pharma market cap](plots/pharma_cap.png)", "", "![Cloud market cap](plots/cloud_cap.png)", "", "![Banking market cap](plots/banking_cap.png)", "", "![Consulting revenue](plots/consulting_revenue.png)", "", "Generate them with `MPLBACKEND=Agg python3 plot_economic_drivers.py`.", "", "## What the data do and do not support", "", "- **Pharma:** larger market cap is directionally associated with larger deltas; after adding Johnson & Johnson, the exploratory permutation result is p≈.045, but this is sensitive to the small sample and current market-cap snapshot. Eli Lilly remains a useful counterexample (largest cap, not the largest delta).", "- **Cloud:** both economic indicators are weak; IBM Cloud is negative despite a large parent, while Oracle Cloud is high with a much smaller parent than Microsoft/Amazon. This argues against a simple scale-only account.", "- **Banking:** the association is negative: the largest-cap banks tend to have smaller name lifts. This is the opposite of a generic 'bigger company gets more premium' story and may reflect profile, geography, or corpus-representation effects.", "- **Consulting:** revenue coverage now includes several private networks, but market-cap analysis remains impossible because these firms are not separately quoted. The revenue-only result should be read as descriptive because the sample is small and sources report different fiscal years.", "- **Google Cloud:** no reliable standalone parent-company page was found in the queried public market-cap source; it remains missing rather than being imputed.", "", "## Ranking-stability check from the supplied CSV", "", "| Sector (May core) | Mean absolute rank change | Pair reversals (%) | Top-3 Jaccard | Spearman ρ |", "|---|---:|---:|---:|---:|"]
    for sector in ("pharma", "cloud", "consulting", "banking"):
        q = rank_summary[sector]; lines.append(f"| {sector} | {q['mean_abs_rank_change']:.2f} | {q['pct_pair_reversals']:.2f} | {q['top3_jaccard']:.2f} | {q['spearman_rho']:.2f} |")
    lines += ["", "## Reproduce", "", "```bash", "cd /Users/gerardpc/repositories/aily/aily-bias-in-llms/experiments/correlation/drivers", "python3 collect_public_economic_data.py", "python3 economic_scale_analysis.py", "MPLBACKEND=Agg python3 plot_economic_drivers.py", "```", "", "The plotting command requires Matplotlib. The collector refreshes the public snapshot; because market caps and TTM revenue change over time, rerunning it can change the numerical correlations. Missing/private companies remain missing rather than being imputed.", ""]
    REPORT.write_text("\n".join(lines))
    print(f"wrote {REPORT}")

if __name__ == "__main__": main()
