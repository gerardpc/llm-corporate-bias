-- Full-table check (all rows in a window): named and masked prompts must be identical
-- after replacing the Entity line value with one placeholder; one system prompt only.
SELECT
    COUNT(*) AS n_rows,
    SUM((replace(prompt_named, E'Entity: ' || incumbent_name || E'\n', E'Entity: <ENTITY>\n')
       = replace(prompt_masked, E'Entity: ' || masked_name || E'\n', E'Entity: <ENTITY>\n'))::int)
        AS n_equivalent,
    SUM((strpos(prompt_named, company_description) > 0
         AND strpos(prompt_masked, company_description) > 0)::int) AS n_profile_in_both,
    SUM((strpos(prompt_named, base_scenario_text) > 0
         AND strpos(prompt_masked, base_scenario_text) > 0)::int) AS n_scenario_in_both,
    COUNT(DISTINCT system_prompt) AS n_system_prompts
FROM {table}
WHERE created_at >= '{start}' AND created_at < '{end}'
  AND company_id IN ({companies});
