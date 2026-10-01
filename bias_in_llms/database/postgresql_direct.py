"""
Direct PostgreSQL connection utilities for faster database operations.

This module provides direct psycopg2-based database operations configured
through standard environment variables.
"""

import os

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

from bias_in_llms.config.environment import load_environment

# Connection pool to reuse connections
_connection_pool = {}


def get_pg_connection():
    """
    Get a direct PostgreSQL connection.

    Configure either DATABASE_URL or the POSTGRES_HOST, POSTGRES_DB,
    POSTGRES_USER, POSTGRES_PASSWORD, and optional POSTGRES_PORT variables.

    Returns:
        psycopg2.connection: Direct PostgreSQL connection
    """
    load_environment()

    database_url = os.getenv("DATABASE_URL")
    if database_url:
        connection_key = database_url
        if connection_key in _connection_pool:
            conn = _connection_pool[connection_key]
            if not conn.closed:
                return conn

        conn = psycopg2.connect(
            database_url,
            cursor_factory=RealDictCursor,
            connect_timeout=10,
        )
        _connection_pool[connection_key] = conn
        return conn

    required_env_vars = [
        "POSTGRES_HOST",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    ]
    missing = [name for name in required_env_vars if not os.getenv(name)]
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            "Database connection is not configured. Set DATABASE_URL or "
            f"the required POSTGRES_* variables. Missing: {missing_str}",
        )

    host = os.environ["POSTGRES_HOST"]
    database = os.environ["POSTGRES_DB"]
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    port = int(os.getenv("POSTGRES_PORT", "5432"))

    connection_key = f"{host}:{port}:{database}:{user}"

    # Simple connection reuse (not a full pool)
    if connection_key in _connection_pool:
        conn = _connection_pool[connection_key]
        if not conn.closed:
            return conn

    conn = psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        cursor_factory=RealDictCursor,
        connect_timeout=10,
    )

    _connection_pool[connection_key] = conn
    return conn


def query_db_direct(query: str, params: tuple = None) -> pd.DataFrame:
    """
    Execute a SELECT query directly on PostgreSQL and return DataFrame.

    Args:
        query: SQL SELECT query to execute
        params: Query parameters (optional)

    Returns:
        pd.DataFrame: Query results
    """
    try:
        conn = get_pg_connection()

        # Execute query manually to avoid pandas/psycopg2 compatibility issues
        with conn.cursor() as cur:
            cur.execute(query, params)

            # Get column names
            columns = [desc[0] for desc in cur.description]

            # Fetch all rows
            rows = cur.fetchall()

            # Convert to DataFrame
            df = pd.DataFrame(rows, columns=columns)

        return df
    except psycopg2.Error as e:
        # Reset connection on error
        connection_key = list(_connection_pool.keys())[0] if _connection_pool else None
        if connection_key and connection_key in _connection_pool:
            del _connection_pool[connection_key]
        raise RuntimeError(f"PostgreSQL query failed: {e}") from e


def execute_query_direct(query: str, params: tuple = None) -> None:
    """
    Execute a non-returning query directly on PostgreSQL.

    Args:
        query: SQL query to execute
        params: Query parameters (optional)
    """
    try:
        conn = get_pg_connection()
        with conn.cursor() as cur:
            cur.execute(query, params)
        conn.commit()
    except psycopg2.Error as e:
        # Reset connection on error and rollback
        connection_key = list(_connection_pool.keys())[0] if _connection_pool else None
        if connection_key and connection_key in _connection_pool:
            try:
                _connection_pool[connection_key].rollback()
            except Exception:
                pass
            del _connection_pool[connection_key]
        raise RuntimeError(f"PostgreSQL execute failed: {e}") from e


def close_connections():
    """Close all pooled connections."""
    for conn in _connection_pool.values():
        try:
            if not conn.closed:
                conn.close()
        except Exception:
            pass
    _connection_pool.clear()
