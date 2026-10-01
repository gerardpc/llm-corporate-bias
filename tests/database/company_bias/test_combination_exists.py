"""
Tests for combination_exists functionality.

This module tests the combination_exists function which checks if a specific
combination of parameters already exists in the company_bias_results table.
"""

from bias_in_llms.database import insert_company_bias_row
from bias_in_llms.database.company_bias_db import combination_exists


def test_combination_exists_functionality(temp_db, sample_company_bias_data):
    """
    Test combination_exists function.

    Verifies that combination_exists correctly identifies existing
    and non-existing combinations.
    """
    data = sample_company_bias_data[0]

    # Initially, combination should not exist
    assert not combination_exists(
        temp_db,
        data["model_id"],
        1,  # run
        data["option_A"],
        data["option_B"],
        0.0,  # temperature
        data["question"],
    )

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

    # Now combination should exist
    assert combination_exists(
        temp_db,
        data["model_id"],
        1,  # run
        data["option_A"],
        data["option_B"],
        0.0,  # temperature
        data["question"],
    )

    # Different run number should not exist
    assert not combination_exists(
        temp_db,
        data["model_id"],
        2,  # Different run
        data["option_A"],
        data["option_B"],
        0.0,  # temperature
        data["question"],
    )

    # Different model should not exist
    assert not combination_exists(
        temp_db,
        "different-model",  # Different model
        1,  # run
        data["option_A"],
        data["option_B"],
        0.0,  # temperature
        data["question"],
    )

    # Different question should not exist
    assert not combination_exists(
        temp_db,
        data["model_id"],
        1,  # run
        data["option_A"],
        data["option_B"],
        0.0,  # temperature
        "Different question?",  # Different question
    )


def test_combination_exists_with_multiple_entries(
    temp_db,
    sample_company_bias_data,
):
    """
    Test combination_exists with multiple database entries.

    Verifies that combination_exists works correctly when there are
    multiple entries in the database.
    """
    # Insert multiple rows with proper parameters
    for _i, data in enumerate(
        sample_company_bias_data[:3],
    ):  # Use only first 3 to speed up test
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

    # Check that each combination exists
    for data in sample_company_bias_data[:3]:
        assert combination_exists(
            temp_db,
            data["model_id"],
            1,  # run
            data["option_A"],
            data["option_B"],
            0.0,  # temperature
            data["question"],
        )

    # Check that a non-existent combination returns False
    assert not combination_exists(
        temp_db,
        "non-existent-model",
        999,
        "non-existent-option_A",
        "non-existent-option_B",
        0.0,
        "Non-existent question?",
    )


def test_combination_exists_with_float_precision(temp_db):
    """
    Test combination_exists with float precision considerations.

    Ensures that float values are handled correctly for comparison.
    """
    insert_company_bias_row(
        temp_db,
        model_id="test-model",
        run=1,
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
        "A",
        "B",
        0.5,
        "Test question?",
    )

    # Slightly different float should not exist
    assert not combination_exists(
        temp_db,
        "test-model",
        1,
        "A",
        "B",
        0.51,  # Slightly different temperature
        "Test question?",
    )
