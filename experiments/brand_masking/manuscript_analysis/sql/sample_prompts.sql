-- Deterministic sample of persisted prompts for the prompt-equivalence audit.
-- Same window/catalog placeholders as extract_trials.sql plus {seed} and {n}.
-- Ordering by md5(id || seed) gives a reproducible pseudo-random sample.
SELECT
    id, model_id, run, scenario_id, company_id,
    incumbent_name, masked_name, base_scenario_text, company_description,
    system_prompt, prompt_named, prompt_masked
FROM {table}
WHERE created_at >= '{start}' AND created_at < '{end}'
  AND company_id IN ({companies})
ORDER BY md5(id::text || '{seed}')
LIMIT {n};
