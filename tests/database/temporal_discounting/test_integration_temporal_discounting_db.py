"""Integration tests for temporal discounting database functionality."""

import sqlite3
import time

from bias_in_llms.database import (
    initialize_temporal_discounting_db,
    insert_temporal_discounting_row,
)


def test_full_workflow_temporal_discounting(temp_db, sample_temporal_discounting_data):
    """
    Test full workflow for temporal discounting.

    This test covers the complete workflow from database initialization
    to data retrieval for temporal discounting.
    """
    # Database is already initialized by fixture, but ensure it's the
    # temporal discounting one
    initialize_temporal_discounting_db(temp_db)

    # Insert all sample data
    for data in sample_temporal_discounting_data:
        insert_temporal_discounting_row(temp_db, **data)

    # Query and verify data
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Test various queries
    cursor.execute("SELECT COUNT(*) FROM temporal_discounting_results")
    assert cursor.fetchone()[0] == len(sample_temporal_discounting_data)

    cursor.execute(
        "SELECT COUNT(*) FROM temporal_discounting_results WHERE reasoning_model = 1",
    )
    reasoning_count = cursor.fetchone()[0]
    expected_reasoning = sum(
        1 for d in sample_temporal_discounting_data if d["reasoning_model"]
    )
    assert reasoning_count == expected_reasoning

    cursor.execute(
        "SELECT AVG(reasoning_tokens) FROM temporal_discounting_results WHERE "
        "reasoning_model = 1",
    )
    avg_tokens = cursor.fetchone()[0]
    assert avg_tokens is not None and avg_tokens >= 0  # Should be non-negative

    cursor.execute("SELECT created_at FROM temporal_discounting_results LIMIT 1")
    created_at = cursor.fetchone()[0]
    assert created_at is not None  # Should have timestamp

    conn.close()


def test_reasoning_vs_non_reasoning_models_temporal_discounting(
    temp_db,
    sample_temporal_discounting_data,
):
    """
    Test reasoning vs non-reasoning models.

    This test queries reasoning vs non-reasoning models for temporal
    discounting.
    """
    initialize_temporal_discounting_db(temp_db)
    # Insert all sample data
    for data in sample_temporal_discounting_data:
        insert_temporal_discounting_row(temp_db, **data)

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Get reasoning models
    cursor.execute(
        """
        SELECT model_id, reasoning_tokens
        FROM temporal_discounting_results
        WHERE reasoning_model = 1
        ORDER BY model_id
        """,
    )
    reasoning_models = cursor.fetchall()

    # Get non-reasoning models
    cursor.execute(
        """
        SELECT model_id, reasoning_tokens
        FROM temporal_discounting_results
        WHERE reasoning_model = 0
        ORDER BY model_id
        """,
    )
    non_reasoning_models = cursor.fetchall()

    # Verify reasoning models have tokens > 0
    for model_id, tokens in reasoning_models:
        assert tokens > 0, f"Reasoning model {model_id} should have tokens > 0"

    # Verify non-reasoning models have tokens = 0
    for model_id, tokens in non_reasoning_models:
        assert tokens == 0, f"Non-reasoning model {model_id} should have tokens = 0"

    conn.close()


def test_data_integrity_across_operations_temporal_discounting(
    temp_db,
    sample_temporal_discounting_data,
):
    """
    Test data integrity across operations.

    This test ensures that data integrity is maintained across multiple
    operations for temporal discounting.
    """
    initialize_temporal_discounting_db(temp_db)
    # Insert data in batches
    batch1 = sample_temporal_discounting_data[:1]
    batch2 = sample_temporal_discounting_data[1:]

    # Insert first batch
    for data in batch1:
        insert_temporal_discounting_row(temp_db, **data)

    # Verify first batch
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM temporal_discounting_results")
    assert cursor.fetchone()[0] == len(batch1)

    # Insert second batch
    for data in batch2:
        insert_temporal_discounting_row(temp_db, **data)

    # Verify total count
    cursor.execute("SELECT COUNT(*) FROM temporal_discounting_results")
    assert cursor.fetchone()[0] == len(sample_temporal_discounting_data)

    # Verify all data is intact
    cursor.execute(
        "SELECT question, model_id FROM temporal_discounting_results ORDER BY id",
    )
    rows = cursor.fetchall()

    for i, (question, model_id) in enumerate(rows):
        expected = sample_temporal_discounting_data[i]
        assert question == expected["question"]
        assert model_id == expected["model_id"]

    conn.close()


def test_timestamp_functionality_temporal_discounting(
    temp_db,
    sample_temporal_discounting_data,
):
    """
    Test timestamp functionality.

    This test ensures that timestamps are properly created and maintained
    for temporal discounting.
    """
    initialize_temporal_discounting_db(temp_db)
    # Insert one record
    data = sample_temporal_discounting_data[0]
    insert_temporal_discounting_row(temp_db, **data)

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Get the timestamp
    cursor.execute("SELECT created_at FROM temporal_discounting_results LIMIT 1")
    timestamp1 = cursor.fetchone()[0]
    assert timestamp1 is not None

    # Insert another record (should have different timestamp)
    time.sleep(0.1)  # Small delay to ensure different timestamp

    data2 = sample_temporal_discounting_data[1]
    insert_temporal_discounting_row(temp_db, **data2)

    cursor.execute(
        "SELECT created_at FROM temporal_discounting_results ORDER BY id DESC LIMIT 1",
    )
    timestamp2 = cursor.fetchone()[0]
    assert timestamp2 is not None
    assert timestamp2 >= timestamp1  # Second timestamp should be later or equal

    conn.close()
