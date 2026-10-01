"""Tests for brand-masking database helpers."""

import pandas as pd

from bias_in_llms.database import brand_masking_db


def test_initialize_brand_masking_db_creates_raw_and_summary_tables(
    monkeypatch,
) -> None:
    """Initialization should create both experiment tables and sync identities."""
    executed_queries: list[str] = []
    sync_targets: list[str] = []
    table_exists_responses = iter(
        [
            pd.DataFrame([{"exists": False}]),
            pd.DataFrame([{"exists": False}]),
        ],
    )

    def fake_query_db(query: str, params=None) -> pd.DataFrame:
        if "information_schema.tables" in query:
            return next(table_exists_responses)
        if "is_identity" in query:
            return pd.DataFrame([{"is_identity": "YES"}])
        raise AssertionError(f"Unexpected query: {query}")

    def fake_execute_query(query: str, params=None) -> None:
        executed_queries.append(query)

    def fake_sync_id_sequence(full_table_name: str) -> None:
        sync_targets.append(full_table_name)

    monkeypatch.setattr(brand_masking_db, "query_db", fake_query_db)
    monkeypatch.setattr(brand_masking_db, "execute_query", fake_execute_query)
    monkeypatch.setattr(brand_masking_db, "sync_id_sequence", fake_sync_id_sequence)

    brand_masking_db.initialize_brand_masking_db()

    assert any(
        "CREATE TABLE bias_in_llms.brand_masking" in query for query in executed_queries
    )
    assert any(
        "CREATE TABLE bias_in_llms.brand_masking_summary" in query
        for query in executed_queries
    )
    assert sync_targets == [
        "bias_in_llms.brand_masking",
        "bias_in_llms.brand_masking_summary",
    ]


def test_insert_brand_masking_trial_uses_expected_insert_shape(monkeypatch) -> None:
    """Trial insertion should send the paired prompt payload and scores to SQL."""
    captured: dict[str, object] = {}

    def fake_execute_query(query: str, params=None) -> None:
        captured["query"] = query
        captured["params"] = params

    monkeypatch.setattr(brand_masking_db, "execute_query", fake_execute_query)

    brand_masking_db.insert_brand_masking_trial(
        run=1,
        scenario_id="1",
        company_id="pfizer",
        base_scenario_text="Scenario text",
        incumbent_name="Pfizer",
        masked_name="Company X",
        company_description="",
        system_prompt="Return one integer.",
        prompt_named="Named prompt",
        prompt_masked="Masked prompt",
        response_named_raw="8",
        response_masked_raw="7",
        score_named=8,
        score_masked=7,
        score_delta=1.0,
        named_parse_success=True,
        masked_parse_success=True,
        reasoning_tokens_named=0,
        reasoning_tokens_masked=0,
        model_id="gpt-4o",
        reasoning_model=False,
        temperature=0.0,
    )

    assert "INSERT INTO bias_in_llms.brand_masking" in captured["query"]
    assert captured["params"][0:4] == (1, "1", "pfizer", "Scenario text")
    assert captured["params"][12:15] == (8, 7, 1.0)


def test_insert_brand_masking_summary_uses_expected_insert_shape(monkeypatch) -> None:
    """Summary insertion should persist Wilcoxon metrics to the summary table."""
    captured: dict[str, object] = {}

    def fake_execute_query(query: str, params=None) -> None:
        captured["query"] = query
        captured["params"] = params

    monkeypatch.setattr(brand_masking_db, "execute_query", fake_execute_query)

    brand_masking_db.insert_brand_masking_summary(
        run=1,
        model_id="gpt-4o",
        temperature=0.0,
        valid_pairs_n=8,
        dropped_pairs_n=1,
        zero_difference_pairs_n=2,
        mean_score_named=8.2,
        mean_score_masked=7.1,
        mean_difference=1.1,
        wilcoxon_statistic=3.0,
        p_value=0.03125,
        conclusion="Significant (Reject Null)",
    )

    assert "INSERT INTO bias_in_llms.brand_masking_summary" in captured["query"]
    assert captured["params"] == (
        1,
        "gpt-4o",
        0.0,
        8,
        1,
        2,
        8.2,
        7.1,
        1.1,
        3.0,
        0.03125,
        "Significant (Reject Null)",
    )
