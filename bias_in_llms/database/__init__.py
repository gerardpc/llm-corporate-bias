"""
Database package public interface.

Re-export commonly used helper functions so they can be imported directly via

    >>> from bias_in_llms.database import initialize_db, insert_row

This is handy for users and is also required by the test-suite shipped with
the project.

All database operations now use PostgreSQL instead of SQLite.
"""

# ruff: disable=unused-import

from .brand_masking_db import (
    brand_masking_combination_exists,
    get_existing_brand_masking_combination,
    initialize_brand_masking_db,
    insert_brand_masking_summary,
    insert_brand_masking_trial,
    select_brand_masking_summaries,
    select_brand_masking_trials,
)
from .company_bias_db import (
    SectorBiasDB,
    combination_exists,
    get_existing_combination,
    initialize_company_bias_db,
    insert_company_bias_row,
)
from .database import execute_query, query_db, select_all_from_table
from .temporal_discounting_db import (
    initialize_temporal_discounting_db,
    insert_temporal_discounting_row,
)

# Update the public interface
__all__ = [
    # Sector bias (generic class for any sector)
    "SectorBiasDB",
    # Legacy company bias functions (backwards compatibility)
    "initialize_company_bias_db",
    "insert_company_bias_row",
    "combination_exists",
    "get_existing_combination",
    "initialize_brand_masking_db",
    "insert_brand_masking_trial",
    "insert_brand_masking_summary",
    "brand_masking_combination_exists",
    "get_existing_brand_masking_combination",
    "select_brand_masking_trials",
    "select_brand_masking_summaries",
    "initialize_temporal_discounting_db",
    "insert_temporal_discounting_row",
    # Common utilities
    "execute_query",
    "query_db",
    "select_all_from_table",
]
