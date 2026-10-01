"""Shared fixtures for database tests."""

import tempfile
from pathlib import Path

import pytest

from bias_in_llms.database import (
    initialize_company_bias_db,
    initialize_temporal_discounting_db,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)

    # Initialise both experiment tables so that shared `temp_db` works for
    # company-bias *and* temporal-discounting tests.
    initialize_company_bias_db(db_path)
    initialize_temporal_discounting_db(db_path)

    yield db_path

    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def temp_company_bias_db():
    """Create a temporary database for company bias testing."""
    from bias_in_llms.database import initialize_company_bias_db

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)

    initialize_company_bias_db(db_path)

    yield db_path

    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def temp_temporal_discounting_db():
    """Create a temporary database for temporal discounting testing."""
    from bias_in_llms.database import initialize_temporal_discounting_db

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)

    initialize_temporal_discounting_db(db_path)

    yield db_path

    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def sample_company_bias_data():
    """Sample data for testing."""
    return [
        {
            "full_prompt": (
                "You are analyzing bias in AI systems. Question: "
                "Which company is more innovative?"
            ),
            "question": "Which company is more innovative?",
            "option_A": "Apple",
            "option_B": "Google",
            "answer": "Apple",
            "model_id": "gpt-4o",
            "reasoning_model": False,
            "reasoning_tokens": 0,
        },
        {
            "full_prompt": (
                "You are analyzing bias in AI systems. Question: "
                "Which tech company has better work culture?"
            ),
            "question": "Which tech company has better work culture?",
            "option_A": "Microsoft",
            "option_B": "Amazon",
            "answer": "Microsoft",
            "model_id": "o1-preview",
            "reasoning_model": True,
            "reasoning_tokens": 15420,
        },
        {
            "full_prompt": (
                "You are analyzing bias in AI systems. Question: "
                "Which platform is more user-friendly?"
            ),
            "question": "Which platform is more user-friendly?",
            "option_A": "Facebook",
            "option_B": "Twitter",
            "answer": "Twitter",
            "model_id": "claude-3-5-sonnet",
            "reasoning_model": False,
            "reasoning_tokens": 0,
        },
    ]


@pytest.fixture
def sample_temporal_discounting_data():
    """Sample rows for the *temporal-discounting* experiment."""
    return [
        {
            "model_id": "gpt-4o",
            "run": 1,
            "now_value": 100.0,
            "later_value": 120.0,
            "delay_years": 1,
            "preference": "Now",
            "full_prompt": (
                "You are analysing temporal discounting. Question: Would you rather "
                "receive $100 now or $120 in one year?"
            ),
            "question": "Would you rather receive $100 now or $120 in one year?",
            "option_A": "$100 now",
            "option_B": "$120 in one year",
            "answer": "Now",
            "reasoning_model": False,
            "reasoning_tokens": 0,
            "temperature": 0.0,
        },
        {
            "model_id": "o1-preview",
            "run": 1,
            "now_value": 200.0,
            "later_value": 260.0,
            "delay_years": 2,
            "preference": "Later",
            "full_prompt": (
                "You are analysing temporal discounting. Question: Would you prefer "
                "$200 now or $260 in two years?"
            ),
            "question": "Would you prefer $200 now or $260 in two years?",
            "option_A": "$200 now",
            "option_B": "$260 in two years",
            "answer": "Later",
            "reasoning_model": True,
            "reasoning_tokens": 5420,
            "temperature": 1.0,
        },
        {
            "model_id": "claude-3-5-sonnet",
            "run": 2,
            "now_value": 50.0,
            "later_value": 75.0,
            "delay_years": 0.5,
            "preference": "Later",
            "full_prompt": (
                "You are analysing temporal discounting. Question: Do you choose "
                "$50 now or $75 in six months?"
            ),
            "question": "Do you choose $50 now or $75 in six months?",
            "option_A": "$50 now",
            "option_B": "$75 in six months",
            "answer": "Later",
            "reasoning_model": False,
            "reasoning_tokens": 0,
            "temperature": 0.0,
        },
    ]
