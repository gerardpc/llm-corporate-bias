"""
Test insert_temporal_discounting_row functionality.

This module contains tests for the insert_temporal_discounting_row
function, ensuring correct insertion, duplicate handling, boolean
handling, and path flexibility for the temporal_discounting_results
table.
"""

import sqlite3
import tempfile
from pathlib import Path

from bias_in_llms.database import (
    initialize_temporal_discounting_db,
    insert_temporal_discounting_row,
)


def test_insert_single_temporal_discounting_row(
    temp_db,
    sample_temporal_discounting_data,
):
    """
    Test single row insertion.

    Verifies that a single row can be inserted into the
    temporal_discounting_results table and that the data matches the
    input.
    """
    data = sample_temporal_discounting_data[0]

    insert_temporal_discounting_row(temp_db, **data)

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM temporal_discounting_results")
    count = cursor.fetchone()[0]
    assert count == 1

    cursor.execute(
        "SELECT now_value, later_value, delay_years, preference "
        "FROM temporal_discounting_results",
    )
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == data["now_value"]
    assert row[1] == data["later_value"]
    assert row[2] == data["delay_years"]
    assert row[3] == data["preference"]

    conn.close()


def test_insert_multiple_temporal_discounting_rows(
    temp_db,
    sample_temporal_discounting_data,
):
    """
    Test multiple row insertion.

    Ensures that multiple rows can be inserted and that each row's
    reasoning_model and reasoning_tokens fields are correct.
    """
    for data in sample_temporal_discounting_data:
        insert_temporal_discounting_row(temp_db, **data)

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM temporal_discounting_results")
    count = cursor.fetchone()[0]
    assert count == len(sample_temporal_discounting_data)

    cursor.execute(
        "SELECT reasoning_model, reasoning_tokens "
        "FROM temporal_discounting_results ORDER BY id",
    )
    rows = cursor.fetchall()

    for i, (reasoning_model, reasoning_tokens) in enumerate(rows):
        expected = sample_temporal_discounting_data[i]
        assert reasoning_model == expected["reasoning_model"]
        assert reasoning_tokens == expected["reasoning_tokens"]

    conn.close()


def test_insert_duplicate_temporal_discounting_rows_allowed(
    temp_db,
    sample_temporal_discounting_data,
):
    """
    Test duplicate row insertion.

    Checks that duplicate rows are allowed in the absence of a
    uniqueness constraint.
    """
    data = sample_temporal_discounting_data[0]

    insert_temporal_discounting_row(temp_db, **data)
    insert_temporal_discounting_row(temp_db, **data)

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM temporal_discounting_results")
    count = cursor.fetchone()[0]
    assert count == 2
    conn.close()


def test_insert_temporal_discounting_with_debug_output(
    temp_db,
    sample_temporal_discounting_data,
    capsys,
):
    """
    Test debug output on insertion.

    Verifies that debug output is printed when inserting rows with
    debug=True.
    """
    data = sample_temporal_discounting_data[0]

    insert_temporal_discounting_row(temp_db, debug=True, **data)

    captured = capsys.readouterr()
    assert "Temporal-discounting row inserted." in captured.out

    insert_temporal_discounting_row(temp_db, debug=True, **data)

    captured = capsys.readouterr()
    assert "Temporal-discounting row inserted." in captured.out


def test_temporal_discounting_reasoning_model_boolean_handling(temp_db):
    """
    Test boolean handling for reasoning_model.

    Ensures that reasoning_model boolean values are stored as 1 (True)
    and 0 (False) in the database.
    """
    insert_temporal_discounting_row(
        temp_db,
        model_id="test-model",
        run=1,
        now_value=100.0,
        later_value=120.0,
        delay_years=1,
        preference="Now",
        full_prompt="Test prompt",
        question="Test question?",
        option_A="A",
        option_B="B",
        answer="A",
        reasoning_model=True,
        reasoning_tokens=1000,
        temperature=1.0,
    )

    insert_temporal_discounting_row(
        temp_db,
        model_id="test-model-2",
        run=1,
        now_value=100.0,
        later_value=110.0,
        delay_years=2,
        preference="Later",
        full_prompt="Test prompt 2",
        question="Test question 2?",
        option_A="A",
        option_B="B",
        answer="B",
        reasoning_model=False,
        reasoning_tokens=0,
        temperature=0.0,
    )

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT reasoning_model FROM temporal_discounting_results ORDER BY id",
    )
    rows = cursor.fetchall()

    assert len(rows) == 2
    assert rows[0][0] == 1  # SQLite stores True as 1
    assert rows[1][0] == 0  # SQLite stores False as 0

    conn.close()


def test_temporal_discounting_path_handling(
    sample_temporal_discounting_data,
):
    """
    Test db_path as string and Path.

    Checks that both string and Path objects can be used as db_path
    for initialization and insertion.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path_str = tmp.name
        db_path_obj = Path(tmp.name)

    try:
        initialize_temporal_discounting_db(db_path_str)

        data = sample_temporal_discounting_data[0]
        insert_temporal_discounting_row(db_path_obj, **data)

        conn = sqlite3.connect(db_path_str)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM temporal_discounting_results")
        count = cursor.fetchone()[0]
        assert count == 1
        conn.close()

    finally:
        Path(db_path_str).unlink()
