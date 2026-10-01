"""Local DuckDB backend used when PostgreSQL is not available."""

from __future__ import annotations

import os
import re
from pathlib import Path

import duckdb
import pandas as pd

from bias_in_llms.config.environment import load_environment
from bias_in_llms.config.project_paths import ROOT_DIR

_connection: duckdb.DuckDBPyConnection | None = None
_SERIAL_PK_RE = re.compile(r"\bSERIAL\s+PRIMARY\s+KEY\b", flags=re.IGNORECASE)


def _duckdb_path() -> Path:
    load_environment()
    configured = os.getenv("DUCKDB_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return ROOT_DIR / "data" / "local_experiments.duckdb"


def _adapt_sql(query: str) -> str:
    # PostgreSQL DDL used in this project; DuckDB needs IDENTITY syntax instead.
    return _SERIAL_PK_RE.sub("BIGINT", query)


def _adapt_params(query: str, params: tuple | None) -> tuple[str, tuple | None]:
    if params is None:
        return query, None
    # Project queries use psycopg2 placeholders.
    return query.replace("%s", "?"), params


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Get or create a reusable local DuckDB connection."""
    global _connection
    if _connection is not None:
        return _connection

    path = _duckdb_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _connection = duckdb.connect(str(path))
    return _connection


def query_db_duckdb(query: str, params: tuple | None = None) -> pd.DataFrame:
    """Execute a read query on local DuckDB and return a DataFrame."""
    conn = get_duckdb_connection()
    sql = _adapt_sql(query)
    sql, adapted_params = _adapt_params(sql, params)
    if adapted_params is None:
        return conn.execute(sql).fetchdf()
    return conn.execute(sql, adapted_params).fetchdf()


def execute_query_duckdb(query: str, params: tuple | None = None) -> None:
    """Execute a mutating query on local DuckDB."""
    conn = get_duckdb_connection()
    sql = _adapt_sql(query)
    sql, adapted_params = _adapt_params(sql, params)
    if adapted_params is None:
        conn.execute(sql)
    else:
        conn.execute(sql, adapted_params)
