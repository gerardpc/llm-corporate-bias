"""
Tests for combination_exists functionality.

This module tests the combination_exists function which checks if a specific
combination of parameters already exists in the temporal_discounting_results table.
"""

from bias_in_llms.database import insert_temporal_discounting_row
from bias_in_llms.database.temporal_discounting_db import combination_exists


def test_combination_exists_functionality(temp_db, sample_temporal_discounting_data):
    """
    Test combination_exists function.

    Verifies that combination_exists correctly identifies existing
    and non-existing combinations.
    """
    data = sample_temporal_discounting_data[0]

    # Initially, combination should not exist
    assert not combination_exists(
        temp_db,
        data["model_id"],
        data["run"],
        data["now_value"],
        data["later_value"],
        data["delay_years"],
        data["temperature"],
    )

    # Insert the row
    insert_temporal_discounting_row(temp_db, **data)

    # Now combination should exist
    assert combination_exists(
        temp_db,
        data["model_id"],
        data["run"],
        data["now_value"],
        data["later_value"],
        data["delay_years"],
        data["temperature"],
    )

    # Different run number should not exist
    assert not combination_exists(
        temp_db,
        data["model_id"],
        data["run"] + 1,  # Different run
        data["now_value"],
        data["later_value"],
        data["delay_years"],
        data["temperature"],
    )

    # Different model should not exist
    assert not combination_exists(
        temp_db,
        "different-model",  # Different model
        data["run"],
        data["now_value"],
        data["later_value"],
        data["delay_years"],
        data["temperature"],
    )


def test_combination_exists_with_multiple_entries(
    temp_db,
    sample_temporal_discounting_data,
):
    """
    Test combination_exists with multiple database entries.

    Verifies that combination_exists works correctly when there are
    multiple entries in the database.
    """
    # Insert multiple rows
    for data in sample_temporal_discounting_data:
        insert_temporal_discounting_row(temp_db, **data)

    # Check that each combination exists
    for data in sample_temporal_discounting_data:
        assert combination_exists(
            temp_db,
            data["model_id"],
            data["run"],
            data["now_value"],
            data["later_value"],
            data["delay_years"],
            data["temperature"],
        )

    # Check that a non-existent combination returns False
    assert not combination_exists(
        temp_db,
        "non-existent-model",
        999,
        99999.0,
        88888.0,
        777,
        0.0,
    )


def test_combination_exists_with_float_precision(temp_db):
    """
    Test combination_exists with float precision considerations.

    Ensures that float values are handled correctly for comparison.
    """
    insert_temporal_discounting_row(
        temp_db,
        model_id="test-model",
        run=1,
        now_value=100.5,
        later_value=120.75,
        delay_years=2,
        preference="Now",
        full_prompt="Test prompt",
        question="Test question?",
        option_A="A",
        option_B="B",
        answer="A",
        reasoning_model=False,
        reasoning_tokens=0,
        temperature=0.5,
    )

    # Exact match should exist
    assert combination_exists(
        temp_db,
        "test-model",
        1,
        100.5,
        120.75,
        2,
        0.5,
    )

    # Slightly different float should not exist
    assert not combination_exists(
        temp_db,
        "test-model",
        1,
        100.50001,  # Slightly different
        120.75,
        2,
        0.5,
    )
