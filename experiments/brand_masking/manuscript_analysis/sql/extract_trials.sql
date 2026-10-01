-- Deduplicated paired trials for one sector / study window.
-- Placeholders filled by study_windows.render_sql (table, window, catalog):
--   {table}        aily_bias_in_llms.brand_masking_<sector>
--   {start}/{end}  half-open created_at window [start, end)
--   {companies}    quoted, comma-separated company_id catalog for the window
-- Dedup key: (run, model_id, scenario_id, company_id, temperature); the latest
-- created_at inside the window is kept. n_versions reports how many rows shared the key.
-- Prompt bodies are not extracted here (see sample_prompts.sql).
WITH windowed AS (
    SELECT
        id, run, model_id, scenario_id, company_id, temperature, created_at,
        md5(company_description) AS description_md5,
        md5(base_scenario_text) AS scenario_md5,
        incumbent_name, masked_name,
        score_named, score_masked,
        named_parse_success, masked_parse_success,
        COUNT(*) OVER key_w AS n_versions,
        ROW_NUMBER() OVER (key_w ORDER BY created_at DESC, id DESC) AS version_rank
    FROM {table}
    WHERE created_at >= '{start}' AND created_at < '{end}'
      AND company_id IN ({companies})
    WINDOW key_w AS (PARTITION BY run, model_id, scenario_id, company_id, temperature)
)
SELECT *
FROM windowed
WHERE version_rank = 1
ORDER BY model_id, run, scenario_id, company_id;
