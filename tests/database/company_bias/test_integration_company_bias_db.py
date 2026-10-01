"""
Integration tests for company bias database functionality.

This module contains comprehensive integration tests that test the complete
workflow of the company bias database system, including initialization,
insertion, querying, and data integrity.
"""

import sqlite3
import time

from bias_in_llms.database import (
    initialize_company_bias_db,
    insert_company_bias_row,
    select_all_from_table,
)


def test_full_workflow_company_bias(temp_db, sample_company_bias_data):
    """
    Test full workflow for company bias database.

    This test covers the complete process from database initialization,
    data insertion, and various queries to ensure correct storage and
    retrieval of company bias results.
    """
    # Database is already initialized by fixture, but ensure it's the
    # company bias one
    initialize_company_bias_db(temp_db)

    # Insert all sample data with required parameters
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
            reasoning_model=i % 2 == 0,  # Alternate True/False
            reasoning_tokens=i * 100,
            temperature=0.0,
        )

    # Test data retrieval using pandas
    df = select_all_from_table("company_bias_results", temp_db)

    assert len(df) == 3
    assert "model_id" in df.columns
    assert "run" in df.columns
    assert "question" in df.columns
    assert "option_A" in df.columns
    assert "option_B" in df.columns
    assert "answer" in df.columns
    assert "reasoning_model" in df.columns
    assert "reasoning_tokens" in df.columns
    assert "temperature" in df.columns

    # Test direct SQL queries
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Test count
    cursor.execute("SELECT COUNT(*) FROM company_bias_results")
    count = cursor.fetchone()[0]
    assert count == 3

    # Test filtering by model
    cursor.execute(
        "SELECT COUNT(*) FROM company_bias_results WHERE model_id = ?",
        (sample_company_bias_data[0]["model_id"],),
    )
    model_count = cursor.fetchone()[0]
    assert model_count >= 1

    # Test filtering by run
    cursor.execute(
        "SELECT COUNT(*) FROM company_bias_results WHERE run = ?",
        (1,),
    )
    run_count = cursor.fetchone()[0]
    assert run_count == 3

    conn.close()


def test_reasoning_vs_non_reasoning_models_company_bias(
    temp_db,
    sample_company_bias_data,
):
    """
    Test reasoning vs non-reasoning model queries.

    This test verifies that reasoning models have nonzero tokens and
    non-reasoning models have zero tokens in the company bias results.
    """
    initialize_company_bias_db(temp_db)

    data = sample_company_bias_data[0]

    # Insert reasoning model data
    insert_company_bias_row(
        temp_db,
        model_id=data["model_id"] + "_reasoning",
        run=1,
        full_prompt=data["full_prompt"],
        question=data["question"],
        option_A=data["option_A"],
        option_B=data["option_B"],
        answer=data["answer"],
        reasoning_model=True,
        reasoning_tokens=1500,
        temperature=0.0,
    )

    # Insert non-reasoning model data
    insert_company_bias_row(
        temp_db,
        model_id=data["model_id"] + "_non_reasoning",
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

    # Query reasoning models
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT reasoning_tokens FROM company_bias_results WHERE reasoning_model = 1",
    )
    reasoning_tokens = cursor.fetchall()

    cursor.execute(
        "SELECT reasoning_tokens FROM company_bias_results WHERE reasoning_model = 0",
    )
    non_reasoning_tokens = cursor.fetchall()

    conn.close()

    # Reasoning models should have > 0 tokens
    assert len(reasoning_tokens) == 1
    assert reasoning_tokens[0][0] > 0

    # Non-reasoning models should have 0 tokens
    assert len(non_reasoning_tokens) == 1
    assert non_reasoning_tokens[0][0] == 0


def test_data_integrity_across_operations_company_bias(
    temp_db,
    sample_company_bias_data,
):
    """
    Test data integrity across multiple operations.

    This test ensures that inserting data in batches maintains the
    integrity and order of all company bias results in the database.
    """
    initialize_company_bias_db(temp_db)
    batch1 = sample_company_bias_data[:2]
    batch2 = sample_company_bias_data[2:4] if len(sample_company_bias_data) > 2 else []

    # Insert first batch
    for _i, data in enumerate(batch1):
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

    # Verify first batch
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM company_bias_results")
    count_after_batch1 = cursor.fetchone()[0]
    assert count_after_batch1 == len(batch1)

    # Insert second batch if it exists
    if batch2:
        for _i, data in enumerate(batch2):
            insert_company_bias_row(
                temp_db,
                model_id=data["model_id"],
                run=2,  # Different run
                full_prompt=data["full_prompt"],
                question=data["question"],
                option_A=data["option_A"],
                option_B=data["option_B"],
                answer=data["answer"],
                reasoning_model=False,
                reasoning_tokens=0,
                temperature=0.0,
            )

        # Verify total count
        cursor.execute("SELECT COUNT(*) FROM company_bias_results")
        total_count = cursor.fetchone()[0]
        assert total_count == len(batch1) + len(batch2)

        # Verify run separation
        cursor.execute("SELECT COUNT(*) FROM company_bias_results WHERE run = 1")
        run1_count = cursor.fetchone()[0]
        assert run1_count == len(batch1)

        cursor.execute("SELECT COUNT(*) FROM company_bias_results WHERE run = 2")
        run2_count = cursor.fetchone()[0]
        assert run2_count == len(batch2)

    conn.close()


def test_timestamp_functionality_company_bias(
    temp_db,
    sample_company_bias_data,
):
    """
    Test timestamp creation and maintenance for company bias.

    This test checks that the created_at timestamp is set for each row
    and that later insertions have equal or later timestamps.
    """
    initialize_company_bias_db(temp_db)

    # Insert one record
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
    )

    # Get the timestamp
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT created_at FROM company_bias_results ORDER BY id LIMIT 1",
    )
    first_timestamp = cursor.fetchone()[0]

    # Wait a moment to ensure different timestamp
    time.sleep(0.01)

    # Insert another record
    insert_company_bias_row(
        temp_db,
        model_id=data["model_id"] + "_2",
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

    cursor.execute(
        "SELECT created_at FROM company_bias_results ORDER BY id DESC LIMIT 1",
    )
    second_timestamp = cursor.fetchone()[0]

    conn.close()

    # Both timestamps should exist and be valid
    assert first_timestamp is not None
    assert second_timestamp is not None

    # Second timestamp should be equal or later
    assert second_timestamp >= first_timestamp
