"""Study-window and catalog definitions for the manuscript brand-masking analysis.

Two windows are analysed separately and never pooled:

* ``may_core`` -- the original four-model study (May 2026) whose company catalogs are
  the ones present in the database for that window (they predate later catalog edits).
* ``sep_sweep`` -- the September 2026 cross-sector sweep run with the AR-215 catalogs.

Catalogs are listed explicitly so the analysis is reproducible even if the experiment
configs change again.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SQL_DIR = Path(__file__).resolve().parent / "sql"
SCHEMA = "aily_bias_in_llms"
RUNS_PER_MODEL = 10
N_SCENARIOS = 51

MAY_CATALOGS: dict[str, tuple[str, ...]] = {
    "pharma": (
        "abbvie", "amgen", "astrazeneca", "bristol", "eli_lilly", "gilead",
        "johnson_and_johnson", "merck", "novartis", "novo_nordisk", "pfizer",
        "regeneron", "roche", "sanofi", "vertex",
    ),
    "consulting": (
        "accenture", "bain", "boston_consulting", "deloitte", "ey", "kpmg",
        "mckinsey", "oliver_wyman", "pwc", "roland_berger",
    ),
    "cloud": (
        "alibaba_cloud", "aws", "azure", "deutsche_telekom_cloud", "digital_ocean",
        "google_cloud", "huawei_cloud", "ibm_cloud", "oracle", "ovh_cloud",
        "tencent_cloud",
    ),
    "banking": (
        "agricultural_bank_of_china", "banco_santander", "bank_of_america",
        "bank_of_china", "bbva", "china_construction_bank", "hsbc", "icbc",
        "jpmorgan", "morgan_stanley", "ubs",
    ),
}

# AR-215 catalogs (experiments/brand_masking/config/<sector>/scenarios.csv on the
# feat/AR-215-brand-masking branch).
SEP_CATALOGS: dict[str, tuple[str, ...]] = {
    "pharma": (
        "abbvie", "amgen", "astrazeneca", "chugai", "eli_lilly", "gilead",
        "jiangsu_hengrui", "johnson_and_johnson", "merck", "novartis",
        "novo_nordisk", "pfizer", "roche", "vertex", "wuxi_apptec",
    ),
    "consulting": (
        "accenture", "bain", "bearingpoint", "beida_zongheng", "boston_consulting",
        "deloitte", "ey", "fti_consulting", "grant_thornton", "hejun", "kearney",
        "kpmg", "mckinsey", "pwc", "zhongda",
    ),
    "cloud": (
        "alibaba_cloud", "aws", "azure", "china_telecom_cloud", "china_unicom_cloud",
        "deutsche_telekom_cloud", "google_cloud", "huawei_cloud", "ibm_cloud",
        "oracle", "ovh_cloud", "tencent_cloud",
    ),
    "banking": (
        "agricultural_bank_of_china", "banco_santander", "bank_of_america",
        "bank_of_china", "china_construction_bank", "citigroup", "goldman_sachs",
        "hsbc", "icbc", "jpmorgan_chase", "morgan_stanley", "mufg",
        "royal_bank_of_canada", "ubs", "wells_fargo",
    ),
}


@dataclass(frozen=True)
class StudyWindow:
    """One sector analysed inside one time window with a fixed catalog."""

    window: str
    sector: str
    start: str
    end: str
    companies: tuple[str, ...]
    note: str = ""

    @property
    def table(self) -> str:
        return f"{SCHEMA}.brand_masking_{self.sector}"

    @property
    def expected_pairs_per_run(self) -> int:
        return N_SCENARIOS * len(self.companies)


STUDY_WINDOWS: tuple[StudyWindow, ...] = (
    *(
        StudyWindow("may_core", s, "2026-05-19", "2026-05-22", MAY_CATALOGS[s])
        for s in ("pharma", "consulting", "cloud", "banking")
    ),
    *(
        StudyWindow("sep_sweep", s, "2026-09-21", "2026-09-23", SEP_CATALOGS[s])
        for s in ("consulting", "cloud", "banking")
    ),
    StudyWindow(
        "sep_sweep",
        "pharma",
        "2026-09-16",
        "2026-09-19",
        SEP_CATALOGS["pharma"],
        note="observed trials retained; run and company coverage varies by model",
    ),
)


def render_sql(template_name: str, window: StudyWindow, **extra: object) -> str:
    """Fill an SQL template with window bounds and catalog (values are constants)."""
    template = (SQL_DIR / template_name).read_text()
    companies = ", ".join(f"'{c}'" for c in window.companies)
    return template.format(
        table=window.table,
        start=window.start,
        end=window.end,
        companies=companies,
        **extra,
    )
