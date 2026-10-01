"""Common database utilities for experiment data management."""

import logging
import os

import pandas as pd

from bias_in_llms.config.environment import load_environment

logger = logging.getLogger(__name__)


def get_database_backend() -> str:
    """Return active backend name: postgres, parquet, or duckdb."""
    load_environment()
    backend = os.getenv("BIAS_DB_BACKEND", "").strip().lower()
    if backend in {"postgres", "parquet", "duckdb"}:
        return backend
    if os.getenv("BIAS_USE_PARQUET", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return "parquet"
    return "postgres"


def query_db(
    query: str,
    params: tuple = None,
) -> pd.DataFrame:
    """
    Execute a custom query on the database and return the results.

    Args:
        query: SQL query to execute.
        params: Query parameters (optional)

    Returns:
        pd.DataFrame: DataFrame containing the results of the query.
    """
    backend = get_database_backend()
    if backend == "parquet":
        from bias_in_llms.database.parquet_query import query_db_parquet

        return query_db_parquet(query, params)
    if backend == "duckdb":
        from bias_in_llms.database.duckdb_local import query_db_duckdb

        return query_db_duckdb(query, params)

    from bias_in_llms.database.postgresql_direct import query_db_direct

    return query_db_direct(query, params)


def execute_query(
    query: str,
    params: tuple = None,
) -> None:
    """
    Execute a query that doesn't return data (INSERT, UPDATE, DELETE, CREATE).

    Args:
        query: SQL query to execute.
        params: Query parameters (optional)
    """
    backend = get_database_backend()
    if backend == "parquet":
        raise RuntimeError(
            "execute_query is not supported when BIAS_DB_BACKEND=parquet "
            "(read-only mode).",
        )
    if backend == "duckdb":
        from bias_in_llms.database.duckdb_local import execute_query_duckdb

        execute_query_duckdb(query, params)
        return

    from bias_in_llms.database.postgresql_direct import execute_query_direct

    execute_query_direct(query, params)


def select_all_from_table(
    table_name: str,
) -> pd.DataFrame:
    """
    Get all rows from a table as a pandas DataFrame.

    Args:
        table_name: The name of the table to get the data from.

    Returns:
        pd.DataFrame: DataFrame containing the data from the table.
    """
    # Map legacy table names to new PostgreSQL table names
    table_mapping = {
        "company_bias_results": "company_bias",
        "brand_masking_results": "brand_masking",
        "brand_masking_summary_results": "brand_masking_summary",
        "temporal_discounting_results": "temporal_discounting",
    }

    # Use mapped table name if it exists, otherwise use original
    mapped_table = table_mapping.get(table_name, table_name)

    query = f"SELECT * FROM bias_in_llms.{mapped_table}"
    return query_db(query)


# ---------------------------------------------------------------------------
# ID Sequence Management (Generic)
# ---------------------------------------------------------------------------


def sync_id_sequence(full_table_name: str) -> None:
    """
    Synchronize the ID sequence for any table with an auto-increment primary key.

    This generic function works for any PostgreSQL table that follows the pattern:
    - Has an 'id' column as primary key
    - Uses SERIAL or has an identity column

    The function ensures the sequence is properly configured:
    - For empty tables: resets sequence to start from 1
    - For populated tables: sets sequence to continue from max(id) + 1

    Optimized implementation:
    - Uses a single query instead of two separate ones
    - Provides structured error handling and logging
    - Eliminates code duplication across experiments

    Args:
        full_table_name: Full table name including schema
    """
    if get_database_backend() != "postgres":
        return

    try:
        # Single optimized query to get both COUNT and MAX in one go
        stats_query = f"""
            SELECT
                COUNT(*) as row_count,
                COALESCE(MAX(id), 0) as max_id
            FROM {full_table_name}
        """

        result_df = query_db(stats_query)

        if result_df.empty:
            logger.warning(
                f"Could not retrieve table statistics for {full_table_name}, "
                f"assuming empty table",
            )
            row_count, max_id = 0, 0
        else:
            result = result_df.iloc[0]
            row_count = int(result["row_count"])
            max_id = int(result["max_id"])

        # Configure sequence based on table state
        sequence_name = f"pg_get_serial_sequence('{full_table_name}', 'id')"

        if row_count == 0:
            _reset_sequence_to_start(sequence_name, full_table_name)
        else:
            _set_sequence_to_continue(sequence_name, max_id, row_count, full_table_name)

    except Exception as e:
        logger.error(f"Failed to synchronize ID sequence for {full_table_name}: {e}")
        logger.info("Continuing without sequence synchronization...")


def _reset_sequence_to_start(sequence_name: str, table_name: str) -> None:
    """Reset sequence to start from 1 for empty table."""
    logger.info(f"Table {table_name} is empty, resetting ID sequence to start from 1")

    reset_query = f"SELECT SETVAL({sequence_name}, 1, false)"
    query_db(reset_query)

    logger.info(f"✅ ID sequence reset to start from 1 for {table_name}")


def _set_sequence_to_continue(
    sequence_name: str,
    max_id: int,
    row_count: int,
    table_name: str,
) -> None:
    """Set sequence to continue from max_id + 1 for populated table."""
    logger.info(f"Table {table_name} has {row_count} rows, max ID is {max_id}")
    logger.info(f"Setting sequence to continue from {max_id + 1}")

    set_query = f"SELECT SETVAL({sequence_name}, %s, true)"
    query_db(set_query, (max_id,))

    logger.info(f"✅ ID sequence set to continue from {max_id + 1} for {table_name}")
