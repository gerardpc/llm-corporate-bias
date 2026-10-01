"""Read-only SQL querying over local parquet exports."""

from __future__ import annotations

import os
import re
from pathlib import Path

import duckdb
import pandas as pd

from bias_in_llms.config.environment import load_environment
from bias_in_llms.config.project_paths import ROOT_DIR

_FROM_OR_JOIN_TABLE = re.compile(
    r'(?i)\b(?:from|join)\s+("?[\w]+"?)\.("?[\w]+"?)',
)
_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SELECT_OR_WITH = re.compile(r"^\s*(select|with)\b", flags=re.IGNORECASE)


def _normalize_identifier(identifier: str) -> str:
    value = identifier.strip().strip('"')
    if not _SQL_IDENTIFIER.fullmatch(value):
        raise ValueError(f"Unsupported SQL identifier: {identifier!r}")
    return value


def _schema_candidates(schema_name: str) -> list[str]:
    if schema_name == "bias_in_llms":
        return [schema_name, "aily_bias_in_llms"]
    if schema_name == "aily_bias_in_llms":
        return [schema_name, "bias_in_llms"]
    return [schema_name]


def parquet_data_dir() -> Path:
    """Return parquet root dir from env or project default."""
    load_environment()
    configured = os.getenv("PARQUET_DATA_DIR")
    if configured:
        return Path(configured).expanduser()
    return ROOT_DIR / "data" / "parquet"


def _resolve_table_parquet_path(parquet_root: Path, schema: str, table: str) -> Path:
    candidates: list[Path] = []
    for schema_name in _schema_candidates(schema):
        candidates.append(parquet_root / schema_name / f"{table}.parquet")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"No parquet file found for {schema}.{table}. Tried: {tried}",
    )


def query_db_parquet(query: str, params: tuple | None = None) -> pd.DataFrame:
    """
    Execute a read-only SQL query against local parquet exports.

    The query may reference schema-qualified tables (for example:
    ``SELECT * FROM bias_in_llms.pharma_bias``). Matching parquet files must
    exist in ``PARQUET_DATA_DIR/<schema>/<table>.parquet``.
    """
    if not _SELECT_OR_WITH.search(query):
        raise RuntimeError("Parquet backend supports only read-only SELECT queries.")

    parquet_root = parquet_data_dir()
    if not parquet_root.exists():
        raise FileNotFoundError(
            f"Parquet data directory not found: {parquet_root}",
        )

    table_refs = {
        (
            _normalize_identifier(schema),
            _normalize_identifier(table),
        )
        for schema, table in _FROM_OR_JOIN_TABLE.findall(query)
    }

    with duckdb.connect(database=":memory:") as conn:
        for schema, table in table_refs:
            parquet_path = _resolve_table_parquet_path(parquet_root, schema, table)
            parquet_path_sql = str(parquet_path).replace("'", "''")
            conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
            conn.execute(
                f'CREATE OR REPLACE VIEW "{schema}"."{table}" AS '
                f"SELECT * FROM read_parquet('{parquet_path_sql}')",
            )

        if params is None:
            return conn.execute(query).fetchdf()
        return conn.execute(query, params).fetchdf()
