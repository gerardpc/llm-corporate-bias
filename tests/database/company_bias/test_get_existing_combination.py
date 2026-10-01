"""
Tests for get_existing_combination functionality.

This module tests the get_existing_combination function which retrieves
existing combination data from the company_bias_results table.
"""

from bias_in_llms.database import insert_company_bias_row
from bias_in_llms.database.company_bias_db import get_existing_combination


def test_get_existing_combination_functionality(
    temp_db,
    sample_company_bias_data,
):
    """
    Test get_existing_combination function.

    Verifies that get_existing_combination correctly retrieves existing
    combination data from the database.
    """
    data = sample_company_bias_data[0]

    # Initially, combination should not exist
    result = get_existing_combination(
        temp_db,
        data["model_id"],
        1,  # run
        data["option_A"],
        data["option_B"],
        0.0,  # temperature
        data["question"],
    )
    assert result is None

    # Insert the row with required parameters
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

    # Now should be able to retrieve the combination
    result = get_existing_combination(
        temp_db,
        data["model_id"],
        1,  # run
        data["option_A"],
        data["option_B"],
        0.0,  # temperature
        data["question"],
    )

    assert result is not None
    assert result["answer"] == data["answer"]
    assert result["question"] == data["question"]
    assert result["full_prompt"] == data["full_prompt"]
    assert result["reasoning_tokens"] == 0


def test_get_existing_combination_with_multiple_entries(
    temp_db,
    sample_company_bias_data,
):
    """
    Test get_existing_combination with multiple database entries.

    Ensures that the function correctly retrieves the right combination
    when multiple entries exist in the database.
    """
    # Insert all sample data with proper parameters
    for _i, data in enumerate(sample_company_bias_data[:3]):  # Use only first 3
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

    # Test retrieval of each combination
    for data in sample_company_bias_data[:3]:
        result = get_existing_combination(
            temp_db,
            data["model_id"],
            1,  # run
            data["option_A"],
            data["option_B"],
            0.0,  # temperature
            data["question"],
        )

        assert result is not None
        assert result["answer"] == data["answer"]
        assert result["question"] == data["question"]


def test_get_existing_combination_return_format(temp_db):
    """
    Test that get_existing_combination returns the correct format.

    Verifies that the returned dictionary contains all expected keys
    and the correct data types.
    """
    insert_company_bias_row(
        temp_db,
        model_id="test-model",
        run=1,
        full_prompt="Test prompt with question?",
        question="Test question?",
        option_A="Option A",
        option_B="Option B",
        answer="A",
        reasoning_model=False,
        reasoning_tokens=500,
        temperature=0.7,
    )

    result = get_existing_combination(
        temp_db,
        "test-model",
        1,
        "Option A",
        "Option B",
        0.7,
        "Test question?",
    )

    # Check that result is not None and has expected keys
    assert result is not None
    assert "answer" in result
    assert "question" in result
    assert "full_prompt" in result
    assert "reasoning_tokens" in result

    # Check data types and values
    assert isinstance(result["answer"], str)
    assert isinstance(result["question"], str)
    assert isinstance(result["full_prompt"], str)
    assert isinstance(result["reasoning_tokens"], int)

    assert result["answer"] == "A"
    assert result["question"] == "Test question?"
    assert result["reasoning_tokens"] == 500


def test_get_existing_combination_with_different_runs(temp_db):
    """
    Test get_existing_combination with different run numbers.

    Verifies that the function correctly distinguishes between different runs.
    """
    # Insert same combination but different runs
    for run_number in [1, 2, 3]:
        insert_company_bias_row(
            temp_db,
            model_id="test-model",
            run=run_number,
            full_prompt="Test prompt",
            question="Test question?",
            option_A="A",
            option_B="B",
            answer=f"Answer_run_{run_number}",
            reasoning_model=False,
            reasoning_tokens=run_number * 100,
            temperature=0.0,
        )

    # Test retrieval for each run
    for run_number in [1, 2, 3]:
        result = get_existing_combination(
            temp_db,
            "test-model",
            run_number,
            "A",
            "B",
            0.0,
            "Test question?",
        )

        assert result is not None
        assert result["answer"] == f"Answer_run_{run_number}"
        assert result["reasoning_tokens"] == run_number * 100


def test_get_existing_combination_with_duplicate_entries(temp_db):
    """
    Test get_existing_combination with duplicate database entries.

    When there are duplicate entries (which shouldn't happen in practice),
    the function should still work correctly and return valid data.
    """
    data = {
        "model_id": "test-model",
        "run": 1,
        "full_prompt": "Test prompt",
        "question": "Test question?",
        "option_A": "A",
        "option_B": "B",
        "answer": "A",
        "reasoning_model": False,
        "reasoning_tokens": 0,
        "temperature": 0.0,
    }

    insert_company_bias_row(temp_db, **data)
    insert_company_bias_row(temp_db, **data)

    # Should still be able to retrieve the combination
    result = get_existing_combination(
        temp_db,
        "test-model",
        1,
        "A",  # option_A
        "B",  # option_B
        0.0,  # temperature
        "Test question?",  # question
    )

    assert result is not None
    assert result["answer"] == "A"
    assert result["question"] == "Test question?"
