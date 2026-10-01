"""
Tests for insert_company_bias_row functionality.

This module contains tests for the insert_company_bias_row function.
It ensures correct insertion, duplicate handling, boolean handling, and
path flexibility for the company_bias_results table.
"""

import sqlite3
import tempfile
from pathlib import Path

from bias_in_llms.database import (
    initialize_company_bias_db,
    insert_company_bias_row,
)


def test_insert_single_company_bias_row(temp_db, sample_company_bias_data):
    """
    Test single row insertion.

    Verifies that a single row can be inserted into the
    company_bias_results table and that the data matches the input.
    """
    data = sample_company_bias_data[0]

    # Insert with required parameters
    insert_company_bias_row(
        temp_db,
        model_id=data["model_id"],
        run=1,
        full_prompt=data["full_prompt"],
        question=data["question"],
        option_A=data["option_A"],
        option_B=data["option_B"],
        answer=data["answer"],
        reasoning_model=False,
        reasoning_tokens=0,
        temperature=0.0,
    )

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM company_bias_results")
    rows = cursor.fetchall()
    conn.close()

    assert len(rows) == 1
    row = rows[0]

    # Check key fields based on TEST schema:
    # 0:id, 1:run, 2:full_prompt, 3:question, 4:option_A, 5:option_B,
    # 6:answer, 7:model_id, 8:reasoning_model, 9:reasoning_tokens,
    # 10:temperature, 11:created_at
    assert row[7] == data["model_id"]  # model_id
    assert row[1] == 1  # run
    assert row[3] == data["question"]  # question
    assert row[4] == data["option_A"]  # option_A
    assert row[5] == data["option_B"]  # option_B
    assert row[6] == data["answer"]  # answer


def test_insert_multiple_company_bias_rows(temp_db, sample_company_bias_data):
    """
    Test multiple row insertion.

    Ensures that multiple rows can be inserted and that each row's
    reasoning_model and reasoning_tokens fields are correct.
    """
    for i, data in enumerate(sample_company_bias_data[:3]):  # Use first 3 for speed
        insert_company_bias_row(
            temp_db,
            model_id=data["model_id"],
            run=1,
            full_prompt=data["full_prompt"],
            question=data["question"],
            option_A=data["option_A"],
            option_B=data["option_B"],
            answer=data["answer"],
            reasoning_model=i % 2 == 0,  # Alternate between True/False
            reasoning_tokens=i * 100,
            temperature=0.0,
        )

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM company_bias_results ORDER BY id")
    rows = cursor.fetchall()
    conn.close()

    assert len(rows) == 3

    for i, row in enumerate(rows):
        expected_reasoning_model = i % 2 == 0
        expected_reasoning_tokens = i * 100

        # Check reasoning_model and reasoning_tokens fields based on TEST schema:
        # 8:reasoning_model, 9:reasoning_tokens
        assert bool(row[8]) == expected_reasoning_model  # reasoning_model
        assert row[9] == expected_reasoning_tokens  # reasoning_tokens


def test_insert_duplicate_company_bias_rows_allowed(
    temp_db,
    sample_company_bias_data,
):
    """
    Test duplicate row insertion.

    Checks that duplicate rows are allowed in the absence of a
    uniqueness constraint.
    """
    data = sample_company_bias_data[0]

    # Insert the same row twice
    for _ in range(2):
        insert_company_bias_row(
            temp_db,
            model_id=data["model_id"],
            run=1,
            full_prompt=data["full_prompt"],
            question=data["question"],
            option_A=data["option_A"],
            option_B=data["option_B"],
            answer=data["answer"],
            reasoning_model=False,
            reasoning_tokens=0,
            temperature=0.0,
        )

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM company_bias_results")
    count = cursor.fetchone()[0]
    conn.close()

    assert count == 2


def test_insert_company_bias_with_debug_output(
    temp_db,
    sample_company_bias_data,
    capsys,
):
    """
    Test debug output on insertion.

    Verifies that debug output is printed when inserting rows with
    debug=True, including for duplicate insertions.
    """
    data = sample_company_bias_data[0]

    insert_company_bias_row(
        temp_db,
        model_id=data["model_id"],
        run=1,
        full_prompt=data["full_prompt"],
        question=data["question"],
        option_A=data["option_A"],
        option_B=data["option_B"],
        answer=data["answer"],
        reasoning_model=False,
        reasoning_tokens=0,
        temperature=0.0,
        debug=True,
    )

    captured = capsys.readouterr()
    assert "Company-bias row inserted" in captured.out

    # Insert again with debug to test duplicate debug output
    insert_company_bias_row(
        temp_db,
        model_id=data["model_id"],
        run=1,
        full_prompt=data["full_prompt"],
        question=data["question"],
        option_A=data["option_A"],
        option_B=data["option_B"],
        answer=data["answer"],
        reasoning_model=False,
        reasoning_tokens=0,
        temperature=0.0,
        debug=True,
    )

    captured = capsys.readouterr()
    assert "Company-bias row inserted" in captured.out


def test_company_bias_reasoning_model_boolean_handling(temp_db):
    """
    Test boolean handling for reasoning_model field.

    Ensures that both True/False and 1/0 values are handled correctly
    for the reasoning_model field.
    """
    test_cases = [
        {"reasoning_model": True, "expected": 1},
        {"reasoning_model": False, "expected": 0},
        {"reasoning_model": 1, "expected": 1},
        {"reasoning_model": 0, "expected": 0},
    ]

    for i, case in enumerate(test_cases):
        insert_company_bias_row(
            temp_db,
            model_id=f"test-model-{i}",
            run=1,
            full_prompt="Test prompt",
            question="Test question?",
            option_A="A",
            option_B="B",
            answer="A",
            reasoning_model=case["reasoning_model"],
            reasoning_tokens=0,
            temperature=0.0,
        )

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT reasoning_model FROM company_bias_results ORDER BY id")
    results = cursor.fetchall()
    conn.close()

    for i, case in enumerate(test_cases):
        assert results[i][0] == case["expected"]


def test_company_bias_path_handling(sample_company_bias_data):
    """
    Test db_path as string and Path.

    Checks that both string and Path objects can be used as db_path
    for initialization and insertion.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path_str = tmp.name
        db_path_obj = Path(tmp.name)

    try:
        initialize_company_bias_db(db_path_str)

        data = sample_company_bias_data[0]
        insert_company_bias_row(
            db_path_obj,
            model_id=data["model_id"],
            run=1,
            full_prompt=data["full_prompt"],
            question=data["question"],
            option_A=data["option_A"],
            option_B=data["option_B"],
            answer=data["answer"],
            reasoning_model=False,
            reasoning_tokens=0,
            temperature=0.0,
        )

        conn = sqlite3.connect(db_path_str)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM company_bias_results")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1

    finally:
        # Clean up
        Path(db_path_str).unlink(missing_ok=True)
