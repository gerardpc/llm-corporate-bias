"""Unit-tests for the shared helpers in `bias_in_llms.database.database`."""

import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
import pytest

from bias_in_llms.database import query_db, select_all_from_table


@pytest.fixture()
def populated_temp_db():
    """Create a temporary SQLite database populated with a simple table."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)

    # Build a trivial schema with a couple of rows
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)")
    cursor.executemany(
        "INSERT INTO test_table (name) VALUES (?)",
        [("Alice",), ("Bob",)],
    )
    conn.commit()
    conn.close()

    yield db_path

    # Teardown – delete the temp file
    db_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_query_db_returns_dataframe(populated_temp_db):
    """`query_db` should return the expected pandas DataFrame."""
    df: pd.DataFrame = query_db(
        populated_temp_db,
        "SELECT * FROM test_table ORDER BY id",
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2  # two rows inserted
    assert list(df["name"]) == ["Alice", "Bob"]

    # Ensure both column names are present
    assert set(df.columns) == {"id", "name"}


def test_query_db_accepts_string_path(populated_temp_db):
    """`query_db` should also work when the db_path is provided as a string."""
    df = query_db(str(populated_temp_db), "SELECT COUNT(*) AS n FROM test_table")
    assert df.loc[0, "n"] == 2


def test_select_all_from_table(populated_temp_db):
    """`select_all_from_table` should fetch the whole table as a DataFrame."""
    df = select_all_from_table("test_table", populated_temp_db)

    # Same expectations as in the `query_db` test
    assert len(df) == 2
    assert list(df["name"]) == ["Alice", "Bob"]
