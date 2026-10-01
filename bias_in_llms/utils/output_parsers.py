"""Utility Pydantic parsers usable across multiple experiments."""

from typing import Literal

from pydantic import BaseModel, Field


class BinaryChoiceOutputParser(BaseModel):
    """Parser that validates LLM responses that should be either 'A' or 'B'."""

    choice: Literal["A", "B"] = Field(
        description="The choice of the LLM",
    )

    reasoning: str = Field(
        description="The reasoning of the LLM's choice",
    )
