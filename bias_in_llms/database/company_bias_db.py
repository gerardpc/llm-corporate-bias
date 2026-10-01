"""
Database utilities for sector-based bias experiments (pharma, consulting, etc.).

This module encapsulates all SQL helpers that touch sector bias tables. It
is intentionally independent from the generic helpers that live in
:pyfile:`bias_in_llms.database.database` so that each experiment can evolve
without stepping on each other's toes.

Public API
~~~~~~~~~~
- :class:`SectorBiasDB` – main class for interacting with sector bias tables
- Legacy functions for backwards compatibility with ``company_bias`` table

The functions now use PostgreSQL instead of SQLite.
"""

import logging
from typing import Optional

from bias_in_llms.database.database import (
    execute_query,
    query_db,
    sync_id_sequence,
)

TABLE_SCHEMA = "bias_in_llms"

logger = logging.getLogger(__name__)


# Expected table columns (excluding the auto-increment primary key `id`).
# The values are the SQL type definitions used for table creation. When adding
# missing columns on an existing table we will add them as NULLable to avoid
# migration failures (i.e. "NOT NULL" is stripped during ALTER TABLE and can be
# tightened later with a backfill + constraint if desired).

EXPECTED_COLUMNS = {
    # Core experiment fields
    "run": "INTEGER NOT NULL",
    "full_prompt": "TEXT",
    "question": "TEXT",
    "option_a": "TEXT",
    "option_b": "TEXT",
    "answer": "TEXT",
    "model_id": "TEXT",
    "reasoning_model": "BOOLEAN",
    "reasoning_tokens": "INTEGER",
    "temperature": "REAL",
    "preceding_context_block": "TEXT",
    "reasoning_text": "TEXT",
    # Metadata
    "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
}

__all__ = [
    "SectorBiasDB",
    # Legacy exports for backwards compatibility
    "initialize_company_bias_db",
    "insert_company_bias_row",
    "combination_exists",
    "get_existing_combination",
]


class SectorBiasDB:
    """
    Database interface for sector-based bias experiments.

    This class provides CRUD operations for sector bias tables (e.g., company_bias,
    consulting_bias). Each sector gets its own table with identical schema.

    Args:
        sector: The sector name (e.g., "pharma", "consulting"). This determines
                the table name as ``{schema}.{sector}_bias``.

    Example:
        >>> db = SectorBiasDB("consulting")
        >>> db.initialize()
        >>> db.insert_row(model_id="gpt-4", run=1, ...)
    """

    def __init__(self, sector: str) -> None:
        """Initialize the database interface for a specific sector."""
        self.sector = sector
        self.table_name = f"{sector}_bias"
        self.full_table_name = f"{TABLE_SCHEMA}.{self.table_name}"

    def initialize(self) -> None:
        """
        Initialise the PostgreSQL table for the sector bias experiment.

        The function is idempotent – it will create the table if it does not
        exist yet but will *not* overwrite an existing database.
        """
        execute_query(f"CREATE SCHEMA IF NOT EXISTS {TABLE_SCHEMA}")

        # Check if table exists
        table_exists_query = """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = %s
                AND table_name = %s
            ) AS exists;
        """

        result_df = query_db(
            table_exists_query,
            (TABLE_SCHEMA, self.table_name),
        )
        result = result_df.to_dict("records")

        if not result[0]["exists"]:
            create_table_query = f"""
                CREATE TABLE {self.full_table_name} (
                    id SERIAL PRIMARY KEY,
                    run INTEGER NOT NULL,
                    full_prompt TEXT,
                    question TEXT,
                    option_a TEXT,
                    option_b TEXT,
                    answer TEXT,
                    model_id TEXT,
                    reasoning_model BOOLEAN,
                    reasoning_tokens INTEGER,
                    temperature REAL,
                    preceding_context_block TEXT,
                    reasoning_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """
            execute_query(create_table_query)
            print(f"Table {self.full_table_name} created successfully")
        else:
            print(f"Table {self.full_table_name} already exists")
            self._ensure_columns_exist()
            print("Synchronizing ID sequence...")
            sync_id_sequence(self.full_table_name)

    def _ensure_columns_exist(self) -> None:
        """Ensure expected columns exist (safe, idempotent migrations)."""
        column_exists_query = """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = %s
                AND table_name = %s
                AND column_name = %s
            ) AS exists;
        """

        for column_name, column_def in EXPECTED_COLUMNS.items():
            exists_df = query_db(
                column_exists_query,
                (TABLE_SCHEMA, self.table_name, column_name),
            )
            if not exists_df.to_dict("records")[0]["exists"]:
                print(f"Adding missing column: {column_name} ...")
                # Avoid NOT NULL during live migrations to prevent failures
                safe_def = column_def.replace(" NOT NULL", "")
                add_col_query = f"""
                    ALTER TABLE {self.full_table_name}
                    ADD COLUMN {column_name} {safe_def}
                """
                execute_query(add_col_query)
                print(f"Column {column_name} added successfully")

    def combination_exists(
        self,
        model_id: str,
        run: int,
        option_a: str,
        option_b: str,
        temperature: float,
        question: str,
        preceding_context_block: str,
    ) -> bool:
        """
        Check if a specific combination already exists in the table.

        Args:
            model_id: The model identifier
            run: The run number
            option_a: The text for option A
            option_b: The text for option B
            temperature: The temperature setting
            question: The question text
            preceding_context_block: The preceding context block

        Returns:
            True if the combination exists, False otherwise
        """
        query = f"""
            SELECT COUNT(*) as count
            FROM {self.full_table_name}
            WHERE model_id = %s
            AND run = %s
            AND option_a = %s
            AND option_b = %s
            AND temperature = %s
            AND question = %s
            AND preceding_context_block = %s
        """

        result_df = query_db(
            query,
            (
                model_id,
                run,
                option_a,
                option_b,
                temperature,
                question,
                preceding_context_block,
            ),
        )
        result = result_df.to_dict("records")
        return result[0]["count"] > 0 if result else False

    def get_existing_combination(
        self,
        model_id: str,
        run: int,
        option_a: str,
        option_b: str,
        temperature: float,
        question: str,
        preceding_context_block: str,
    ) -> Optional[dict]:
        """
        Retrieve an existing combination from the table.

        Args:
            model_id: The model identifier
            run: The run number
            option_a: The text for option A
            option_b: The text for option B
            temperature: The temperature setting
            question: The question text
            preceding_context_block: The preceding context block

        Returns:
            Dictionary with the existing combination data, or None if not found
        """
        query = f"""
            SELECT answer, question, full_prompt, reasoning_tokens
            FROM {self.full_table_name}
            WHERE model_id = %s
            AND run = %s
            AND option_a = %s
            AND option_b = %s
            AND temperature = %s
            AND question = %s
            AND preceding_context_block = %s
            LIMIT 1
        """

        result_df = query_db(
            query,
            (
                model_id,
                run,
                option_a,
                option_b,
                temperature,
                question,
                preceding_context_block,
            ),
        )
        result = result_df.to_dict("records")

        if result:
            row = result[0]
            return {
                "answer": row["answer"],
                "question": row["question"],
                "full_prompt": row["full_prompt"],
                "reasoning_tokens": row["reasoning_tokens"],
            }

        return None

    def insert_row(
        self,
        model_id: str,
        run: int,
        full_prompt: str,
        question: str,
        option_a: str,
        option_b: str,
        answer: str,
        reasoning_model: bool,
        reasoning_tokens: int,
        temperature: float,
        preceding_context_block: str,
        reasoning_text: str,
        debug: bool = False,
    ) -> None:
        """
        Insert a row into the sector bias table.

        Args:
            model_id: The model identifier
            run: The run number
            full_prompt: The full prompt text
            question: The question text
            option_a: The text for option A
            option_b: The text for option B
            answer: The answer choice
            reasoning_model: Whether reasoning model was used
            reasoning_tokens: Number of reasoning tokens
            temperature: The temperature setting
            preceding_context_block: The preceding context block
            reasoning_text: The reasoning text
            debug: Flag to enable debug output
        """
        query = f"""
            INSERT INTO {self.full_table_name} (
                model_id, run, full_prompt, question, option_a, option_b, answer,
                reasoning_model, reasoning_tokens, temperature, preceding_context_block,
                reasoning_text
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        params = (
            model_id,
            run,
            full_prompt,
            question,
            option_a,
            option_b,
            answer,
            reasoning_model,
            reasoning_tokens,
            temperature,
            preceding_context_block,
            reasoning_text,
        )

        execute_query(query, params)

        if debug:
            print(f"{self.sector.capitalize()}-bias row inserted into PostgreSQL.")


# ---------------------------------------------------------------------------
# Legacy functions for backwards compatibility with company_bias table
# ---------------------------------------------------------------------------

# Default instance for the original "company" sector (pharma)
_default_db = SectorBiasDB("company")


def initialize_company_bias_db() -> None:
    """Legacy wrapper: Initialize the company_bias table."""
    _default_db.initialize()


def combination_exists(
    model_id: str,
    run: int,
    option_a: str,
    option_b: str,
    temperature: float,
    question: str,
    preceding_context_block: str,
) -> bool:
    """Legacy wrapper: Check if combination exists in company_bias table."""
    return _default_db.combination_exists(
        model_id=model_id,
        run=run,
        option_a=option_a,
        option_b=option_b,
        temperature=temperature,
        question=question,
        preceding_context_block=preceding_context_block,
    )


def get_existing_combination(
    model_id: str,
    run: int,
    option_a: str,
    option_b: str,
    temperature: float,
    question: str,
    preceding_context_block: str,
) -> Optional[dict]:
    """Legacy wrapper: Retrieve existing combination from company_bias table."""
    return _default_db.get_existing_combination(
        model_id=model_id,
        run=run,
        option_a=option_a,
        option_b=option_b,
        temperature=temperature,
        question=question,
        preceding_context_block=preceding_context_block,
    )


def insert_company_bias_row(
    model_id: str,
    run: int,
    full_prompt: str,
    question: str,
    option_A: str,
    option_B: str,
    answer: str,
    reasoning_model: bool,
    reasoning_tokens: int,
    temperature: float,
    preceding_context_block: str,
    reasoning_text: str,
    debug: bool = False,
) -> None:
    """Legacy wrapper: Insert row into company_bias table."""
    _default_db.insert_row(
        model_id=model_id,
        run=run,
        full_prompt=full_prompt,
        question=question,
        option_a=option_A,
        option_b=option_B,
        answer=answer,
        reasoning_model=reasoning_model,
        reasoning_tokens=reasoning_tokens,
        temperature=temperature,
        preceding_context_block=preceding_context_block,
        reasoning_text=reasoning_text,
        debug=debug,
    )
