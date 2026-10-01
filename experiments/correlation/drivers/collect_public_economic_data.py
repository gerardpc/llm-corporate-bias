#!/usr/bin/env python3
"""Collect public market-cap and TTM-revenue values for Experiment 2 companies.

Source: CompaniesMarketCap company pages. Values are saved with URLs and the
retrieval date so the snapshot can be audited or refreshed later.
"""
from __future__ import annotations
import csv
import datetime as dt
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "public_economic_data.csv"

SLUGS = {
    "AbbVie":"abbvie", "Amgen":"amgen", "AstraZeneca":"astrazeneca", "Bristol Myers Squibb":"bristol-myers-squibb", "Chugai Pharmaceuticals":"chugai-pharmaceutical", "Eli Lilly":"eli-lilly", "Gilead Sciences":"gilead-sciences", "Jiangsu Hengrui Medicine":"jiangsu-hengrui-medicine", "Johnson & Johnson":"johnson-and-johnson", "Merck & Co":"merck", "Merck & Co.":"merck", "Novartis":"novartis", "Novo Nordisk":"novo-nordisk", "Pfizer":"pfizer", "Regeneron Pharmaceuticals":"regeneron-pharmaceuticals", "Roche":"roche", "Sanofi":"sanofi", "Vertex Pharmaceuticals":"vertex-pharmaceuticals", "WuXi AppTec":"wuxi-apptec",
    "Agricultural Bank of China":"agricultural-bank-of-china", "BBVA":"bbva", "Banco Santander":"santander", "Bank of America":"bank-of-america", "Bank of China":"bank-of-china", "China Construction Bank":"china-construction-bank", "Citigroup":"citigroup", "Goldman Sachs":"goldman-sachs", "HSBC":"hsbc", "Industrial and Commercial Bank of China (ICBC)":"icbc", "JPMorgan Chase":"jp-morgan-chase", "Mitsubishi UFJ Financial Group (MUFG)":"mitsubishi-ufj-financial", "Morgan Stanley":"morgan-stanley", "Royal Bank of Canada":"royal-bank-of-canada", "UBS":"ubs", "Wells Fargo":"wells-fargo",
    "Alibaba Cloud":"alibaba", "Amazon Web Services (AWS)":"amazon", "China Telecom Cloud":"china-telecom", "China Unicom Cloud":"china-unicom", "Deutsche Telekom Cloud":"deutsche-telekom", "DigitalOcean":"digitalocean", "Google Cloud":"alphabet-google", "IBM Cloud":"ibm", "Microsoft Azure":"microsoft", "OVH Cloud":"ovh", "OVHcloud":"ovh", "Oracle Cloud Infrastructure":"oracle", "Tencent Cloud":"tencent",
    "Accenture":"accenture", "FTI Consulting":"fti-consulting",
    "Bain & Company":"bain-and-company", "Boston Consulting Group":"boston-consulting-group",
    "Deloitte Consulting":"deloitte", "EY Consulting":"ernst-young", "KPMG Advisory":"kpmg",
    "McKinsey & Company":"mckinsey", "Kearney":"kearney",
}

# Publicly reported global-network revenue for private consulting firms. These
# are not market caps; they are recorded as revenue-only observations with the
# cited source URL. Values are USD billions.
MANUAL_REVENUE = {
    "Boston Consulting Group": (14.4, "https://www.bcg.com/press/23april2026-bcg-revenue-22nd-consecutive-year-growth"),
    "Bain & Company": (14.0, "https://www.forbes.com/companies/bain-and-company/"),
    "Deloitte Consulting": (70.5, "https://www.deloitte.com/global/en/about/press-room/global-revenue-announcement.html"),
    "EY Consulting": (53.219, "https://www.ey.com/en_gl/newsroom/2025/10/ey-announces-global-revenue-of-us-53-2b-for-fiscal-year-2025"),
    "KPMG Advisory": (38.4, "https://kpmg.com/xx/en/media/press-releases/2024/12/robust-growth-for-kpmg-as-global-revenues-rise-5-percent-to-us-dollar-38-point-4-billion.html"),
    "McKinsey & Company": (16.0, "https://www.ft.com/content/d17b114c-fdbe-433f-902e-5043ec51cb5a"),
    "Kearney": (2.0, "https://en.wikipedia.org/wiki/Kearney_(consulting_firm)"),
}

CAP = re.compile(r"has a market cap of <strong>\$([0-9.,]+) (Trillion|Billion|Million) USD", re.I)
REV = re.compile(r"TTM.*?\$([0-9.,]+) (Trillion|Billion|Million) USD", re.I | re.S)

def billions(value: str, unit: str) -> float:
    return float(value.replace(",", "")) * {"Trillion":1000.0, "Billion":1.0, "Million":0.001}[unit.title()]

def get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as response:
        return response.read().decode("utf-8", "ignore")

def main() -> None:
    today = dt.date.today().isoformat()
    rows = []
    for company, slug in SLUGS.items():
        cap_url = f"https://companiesmarketcap.com/{slug}/marketcap/"
        rev_url = f"https://companiesmarketcap.com/{slug}/revenue/"
        cap = rev = None
        try:
            cap_match = CAP.search(get(cap_url))
            cap = billions(*cap_match.groups()) if cap_match else None
        except Exception as exc:
            print(f"warning: {company} market cap: {exc}")
        try:
            rev_match = REV.search(get(rev_url))
            rev = billions(*rev_match.groups()) if rev_match else None
        except Exception as exc:
            print(f"warning: {company} revenue: {exc}")
        manual = MANUAL_REVENUE.get(company)
        if manual:
            rev = manual[0]
            rev_url = manual[1]
            source = "Public company/annual revenue disclosure"
            notes = "Private consulting network revenue; no market cap."
        else:
            rev_url = f"https://companiesmarketcap.com/{slug}/revenue/"
            source = "CompaniesMarketCap"
            notes = "Parent-company proxy; blank means no separately quoted/publicly parsed value."
        rows.append({"company":company, "companiesmarketcap_slug":slug,
                     "market_cap_usd_billion":"" if cap is None else f"{cap:.3f}",
                     "revenue_ttm_usd_billion":"" if rev is None else f"{rev:.3f}",
                     "source":source, "source_url_market_cap":cap_url if cap is not None else "",
                     "source_url_revenue":rev_url, "retrieved_date":today,
                     "notes":notes})
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}")

if __name__ == "__main__":
    main()
