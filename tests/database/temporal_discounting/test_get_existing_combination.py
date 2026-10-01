"""
Tests for get_existing_combination functionality.

This module tests the get_existing_combination function which retrieves
existing combination data from the temporal_discounting_results table.
"""

from bias_in_llms.database import insert_temporal_discounting_row
from bias_in_llms.database.temporal_discounting_db import get_existing_combination


def test_get_existing_combination_functionality(
    temp_db,
    sample_temporal_discounting_data,
):
    """
    Test get_existing_combination function.

    Verifies that get_existing_combination correctly retrieves existing
    combination data from the database.
    """
    data = sample_temporal_discounting_data[0]

    # Initially, combination should not exist
    result = get_existing_combination(
        temp_db,
        data["model_id"],
        data["run"],
        data["now_value"],
        data["later_value"],
        data["delay_years"],
        data["temperature"],
    )
    assert result is None

    # Insert the row
    insert_temporal_discounting_row(temp_db, **data)

    # Now should be able to retrieve the combination
    result = get_existing_combination(
        temp_db,
        data["model_id"],
        data["run"],
        data["now_value"],
        data["later_value"],
        data["delay_years"],
        data["temperature"],
    )

    assert result is not None
    assert result["preference"] == data["preference"]
    assert result["answer"] == data["answer"]
    assert result["option_A"] == data["option_A"]
    assert result["option_B"] == data["option_B"]
    assert result["question"] == data["question"]
    assert result["full_prompt"] == data["full_prompt"]
    assert result["reasoning_tokens"] == data["reasoning_tokens"]

    # Different combination should return None
    result = get_existing_combination(
        temp_db,
        "different-model",
        data["run"],
        data["now_value"],
        data["later_value"],
        data["delay_years"],
        data["temperature"],
    )
    assert result is None


def test_get_existing_combination_with_multiple_entries(
    temp_db,
    sample_temporal_discounting_data,
):
    """
    Test get_existing_combination with multiple database entries.

    Ensures that the function correctly retrieves the right combination
    when multiple entries exist in the database.
    """
    # Insert all sample data
    for data in sample_temporal_discounting_data:
        insert_temporal_discounting_row(temp_db, **data)

    # Retrieve each combination and verify it matches the original data
    for expected_data in sample_temporal_discounting_data:
        result = get_existing_combination(
            temp_db,
            expected_data["model_id"],
            expected_data["run"],
            expected_data["now_value"],
            expected_data["later_value"],
            expected_data["delay_years"],
            expected_data["temperature"],
        )

        assert result is not None
        assert result["preference"] == expected_data["preference"]
        assert result["answer"] == expected_data["answer"]
        assert result["option_A"] == expected_data["option_A"]
        assert result["option_B"] == expected_data["option_B"]
        assert result["question"] == expected_data["question"]
        assert result["full_prompt"] == expected_data["full_prompt"]
        assert result["reasoning_tokens"] == expected_data["reasoning_tokens"]


def test_get_existing_combination_return_format(temp_db):
    """
    Test that get_existing_combination returns the correct format.

    Verifies that the returned dictionary contains all expected keys
    and the correct data types.
    """
    insert_temporal_discounting_row(
        temp_db,
        model_id="test-model",
        run=1,
        now_value=100.0,
        later_value=120.0,
        delay_years=1,
        preference="Now",
        full_prompt="Test prompt with question?",
        question="Test question?",
        option_A="Option A",
        option_B="Option B",
        answer="A",
        reasoning_model=True,
        reasoning_tokens=500,
        temperature=0.7,
    )

    result = get_existing_combination(
        temp_db,
        "test-model",
        1,
        100.0,
        120.0,
        1,
        0.7,
    )

    assert result is not None
    assert isinstance(result, dict)

    # Check that all expected keys are present
    expected_keys = {
        "preference",
        "answer",
        "option_A",
        "option_B",
        "question",
        "full_prompt",
        "reasoning_tokens",
    }
    assert set(result.keys()) == expected_keys

    # Check data types and values
    assert isinstance(result["preference"], str)
    assert isinstance(result["answer"], str)
    assert isinstance(result["option_A"], str)
    assert isinstance(result["option_B"], str)
    assert isinstance(result["question"], str)
    assert isinstance(result["full_prompt"], str)
    assert isinstance(result["reasoning_tokens"], int)

    # Check specific values
    assert result["preference"] == "Now"
    assert result["answer"] == "A"
    assert result["option_A"] == "Option A"
    assert result["option_B"] == "Option B"
    assert result["question"] == "Test question?"
    assert result["full_prompt"] == "Test prompt with question?"
    assert result["reasoning_tokens"] == 500


def test_get_existing_combination_with_duplicate_entries(temp_db):
    """
    Test get_existing_combination with duplicate database entries.

    When there are duplicate entries (which shouldn't happen in practice),
    the function should still work correctly and return valid data.
    """
    data = {
        "model_id": "test-model",
        "run": 1,
        "now_value": 100.0,
        "later_value": 120.0,
        "delay_years": 1,
        "preference": "Now",
        "full_prompt": "Test prompt",
        "question": "Test question?",
        "option_A": "A",
        "option_B": "B",
        "answer": "A",
        "reasoning_model": False,
        "reasoning_tokens": 0,
        "temperature": 0.0,
    }

    insert_temporal_discounting_row(temp_db, **data)
    insert_temporal_discounting_row(temp_db, **data)

    # Should still be able to retrieve the combination
    result = get_existing_combination(
        temp_db,
        "test-model",
        1,
        100.0,
        120.0,
        1,
        0.0,
    )

    assert result is not None
    assert result["preference"] == "Now"
    assert result["answer"] == "A"
