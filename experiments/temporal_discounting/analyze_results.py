"""
Analysis helper for the temporal-discounting experiment.

Usage
-----
$ python -m experiments.temporal_discounting.analyze_results
# prints the DataFrame

You can also import ``load_temporal_discounting_results`` from notebooks.
"""

from pathlib import Path

import pandas as pd

from bias_in_llms.config.project_paths import DATA_DIR
from bias_in_llms.database.database import query_db

# ---------------------------------------------------------------------------

DEFAULT_DB = DATA_DIR / "temporal_discounting_results.db"


def load_temporal_discounting_results(
    db_path: Path | str = DEFAULT_DB,
) -> pd.DataFrame:
    """
    Load temporal-discounting results as a DataFrame.

    Fetches all rows from the temporal_discounting_results table and
    returns them as a pandas DataFrame with analysis-friendly column
    names.
    """
    query = (
        "SELECT\n"
        '    now_value   AS "Now ($)",\n'
        '    later_value AS "Later ($)",\n'
        '    delay_years AS "Delay (years)",\n'
        '    model_id    AS "LLM",\n'
        '    preference  AS "Preference",\n'
        '    created_at  AS "Timestamp"\n'
        "FROM temporal_discounting_results\n"
        "ORDER BY created_at\n"
    )
    return query_db(db_path, query)


def main() -> None:
    """
    Print the temporal-discounting results DataFrame.

    Loads the results from the default database and prints the
    DataFrame to stdout.
    """
    df = load_temporal_discounting_results()
    print(df)


if __name__ == "__main__":
    main()
