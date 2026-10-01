"""Test database initialization functionality for the company bias experiment."""

import sqlite3
import tempfile
from pathlib import Path

from bias_in_llms.database import initialize_company_bias_db


def test_initialize_company_bias_db_creates_database():
    """Test that initialize_company_bias_db creates a new database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tmp:
        db_path = Path(tmp.name)

    # Ensure file doesn't exist
    assert not db_path.exists()

    # Initialize database
    initialize_company_bias_db(db_path)

    # Check that file was created
    assert db_path.exists()

    # Cleanup
    db_path.unlink()


def test_initialize_company_bias_db_creates_correct_table_structure(temp_db):
    """Test that the company bias database has the correct table structure."""
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Get table info
    cursor.execute("PRAGMA table_info(company_bias_results)")
    columns = cursor.fetchall()

    # Expected columns: (cid, name, type, notnull, dflt_value, pk)
    expected_columns = {
        "id": ("INTEGER", 1),  # (type, is_primary_key)
        "run": ("INTEGER", 0),
        "full_prompt": ("TEXT", 0),
        "question": ("TEXT", 0),
        "option_A": ("TEXT", 0),
        "option_B": ("TEXT", 0),
        "answer": ("TEXT", 0),
        "model_id": ("TEXT", 0),
        "reasoning_model": ("BOOLEAN", 0),
        "reasoning_tokens": ("INTEGER", 0),
        "temperature": ("REAL", 0),
        "created_at": ("DATETIME", 0),
    }

    # Check that all expected columns exist
    column_info = {col[1]: (col[2], col[5]) for col in columns}

    for col_name, (expected_type, expected_pk) in expected_columns.items():
        assert col_name in column_info, f"Column {col_name} not found"
        actual_type, actual_pk = column_info[col_name]
        assert actual_type == expected_type, (
            f"Column {col_name} has type {actual_type}, expected {expected_type}"
        )
        assert actual_pk == expected_pk, f"Column {col_name} primary key mismatch"

    conn.close()


def test_initialize_company_bias_db_with_existing_database(temp_db, capsys):
    """Test that initialize_company_bias_db handles existing databases correctly."""
    # Initialize again (should not fail)
    initialize_company_bias_db(temp_db)

    # Check output
    captured = capsys.readouterr()
    assert "Database already exists" in captured.out
