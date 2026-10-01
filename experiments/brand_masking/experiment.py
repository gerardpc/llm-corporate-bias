"""Core execution loop for the brand-masking experiment."""

from __future__ import annotations

import concurrent.futures
import random
import re
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from bias_in_llms.config.project_paths import DATA_DIR
from bias_in_llms.database import query_db
from bias_in_llms.database.brand_masking_db import (
    brand_masking_combination_exists,
    get_existing_brand_masking_combination,
    get_raw_full_table_name,
    initialize_brand_masking_db,
    insert_brand_masking_trial,
)
from bias_in_llms.llm.llm_calling import resolve_model_id, run_llm_call
from bias_in_llms.llm.prompt.prompt_loading import load_prompt_templates
from bias_in_llms.utils.tracing import trace_context
from experiments.brand_masking.config_models import (
    BrandMaskingExperimentConfig,
    load_brand_masking_config,
    load_company_descriptions,
)

if TYPE_CHECKING:
    from langchain_core.prompts import ChatPromptTemplate

_db_lock = threading.Lock()
_INTEGER_PATTERN = re.compile(r"\b(10|[1-9])\b")


@dataclass(frozen=True)
class BrandMaskingScenario:
    """Fully prepared scenario row ready for prompt rendering."""

    scenario_id: str
    company_id: str
    base_scenario_text: str
    incumbent_name: str
    masked_name: str
    company_description: str


def extract_integer_score(response_text: str) -> int | None:
    """Extract a single integer score from raw model output."""
    match = _INTEGER_PATTERN.search(response_text)
    if match:
        return int(match.group(0))
    return None


def construct_prompts(
    scenario: BrandMaskingScenario,
    prompt_templates: dict[str, str],
) -> tuple[str, str, str]:
    """Construct the system, named, and masked prompts for one scenario."""
    system_prompt = prompt_templates["system_prompt"]
    named_prompt = prompt_templates["named_user_prompt"].format(
        base_scenario_text=scenario.base_scenario_text,
        incumbent_name=scenario.incumbent_name,
        masked_name=scenario.masked_name,
        company_description=scenario.company_description,
    )
    masked_prompt = prompt_templates["masked_user_prompt"].format(
        base_scenario_text=scenario.base_scenario_text,
        incumbent_name=scenario.incumbent_name,
        masked_name=scenario.masked_name,
        company_description=scenario.company_description,
    )
    return system_prompt, named_prompt, masked_prompt


def _prepare_llm_models(config: BrandMaskingExperimentConfig) -> list[dict]:
    """Resolve configured models to provider-neutral runtime descriptors."""
    llm_models: list[dict] = []
    for model_config in config.llm_models:
        llm_models.append(
            {
                "model_id": resolve_model_id(
                    model_config.model_id,
                    model_config.model_type,
                ),
                "model_type": model_config.model_type,
                "reasoning_model": model_config.reasoning_model,
                "temperature": model_config.temperature,
            },
        )

    return llm_models


def load_brand_masking_scenarios(
    config: BrandMaskingExperimentConfig,
) -> list[BrandMaskingScenario]:
    """Load scenarios CSV and enrich it with validated company descriptions."""
    dataset_path = config.resolve_path(config.dataset_path)
    catalog_path = config.resolve_path(config.company_descriptions_path)

    scenarios_df = pd.read_csv(dataset_path)
    required_columns = {"scenario_id", "company_id", "base_scenario_text"}
    missing = required_columns.difference(scenarios_df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"Scenario dataset missing required columns: {missing_str}")

    company_catalog = load_company_descriptions(catalog_path)
    companies = {company.company_id: company for company in company_catalog.companies}

    prepared_scenarios: list[BrandMaskingScenario] = []
    for row in scenarios_df.to_dict("records"):
        company_id = str(row["company_id"])
        company = companies.get(company_id)
        if company is None:
            raise KeyError(f"Scenario references unknown company_id '{company_id}'")

        masked_name_override = row.get("masked_name")
        masked_name = (
            str(masked_name_override)
            if pd.notna(masked_name_override)
            else company.masked_name or config.default_masked_name
        )

        prepared_scenarios.append(
            BrandMaskingScenario(
                scenario_id=str(row["scenario_id"]),
                company_id=company_id,
                base_scenario_text=str(row["base_scenario_text"]),
                incumbent_name=company.incumbent_name,
                masked_name=masked_name,
                company_description=company.description,
            ),
        )

    return prepared_scenarios


def _load_existing_trials_cache(
    models: list[dict],
    run_number: int,
    sector: str,
) -> dict:
    """Load existing results for the requested models into an in-memory cache."""
    cache: dict[tuple[str, str, float], dict] = {}
    raw_full = get_raw_full_table_name(sector)

    for model in models:
        model_id = model["model_id"]
        temperature = model["temperature"]
        query = f"""
            SELECT
                scenario_id,
                company_id,
                base_scenario_text,
                incumbent_name,
                masked_name,
                company_description,
                system_prompt,
                prompt_named,
                prompt_masked,
                response_named_raw,
                response_masked_raw,
                score_named,
                score_masked,
                score_delta,
                named_parse_success,
                masked_parse_success,
                reasoning_tokens_named,
                reasoning_tokens_masked
            FROM {raw_full}
            WHERE model_id = %s AND run = %s AND temperature = %s
        """
        result_df = query_db(query, (model_id, run_number, temperature))
        for row in result_df.to_dict("records"):
            cache[(model_id, row["scenario_id"], temperature)] = row

    return cache


def _check_existing_trial_and_build_result(
    *,
    model_id: str,
    run_number: int,
    scenario: BrandMaskingScenario,
    temperature: float,
    reasoning_model: bool,
    cache: dict | None,
    sector: str,
) -> tuple[bool, dict | None]:
    """Check the cache or DB for an existing paired trial and normalize its shape."""
    if cache is not None:
        existing = cache.get((model_id, scenario.scenario_id, temperature))
        if existing:
            return True, {
                "run": run_number,
                **existing,
                "model_id": model_id,
                "reasoning_model": reasoning_model,
                "temperature": temperature,
            }
        return False, None

    if brand_masking_combination_exists(
        model_id,
        run_number,
        scenario.scenario_id,
        temperature,
        sector,
    ):
        existing = get_existing_brand_masking_combination(
            model_id,
            run_number,
            scenario.scenario_id,
            temperature,
            sector,
        )
        if existing:
            return True, {
                "run": run_number,
                **existing,
                "model_id": model_id,
                "reasoning_model": reasoning_model,
                "temperature": temperature,
            }
        return True, None

    return False, None


def _run_single_trial(
    task: tuple[BrandMaskingScenario, dict],
    *,
    llm_instances: dict,
    prompt_template: ChatPromptTemplate,
    prompt_templates: dict[str, str],
    run_number: int,
    cache: dict | None,
    sector: str,
) -> tuple[bool, bool, dict | None]:
    """Run named and masked scoring for one scenario/model combination."""
    scenario, model_config = task
    model_id = model_config["model_id"]
    temperature = model_config["temperature"]
    reasoning_model = model_config["reasoning_model"]
    llm = llm_instances[model_id]

    # Cache lookup is read-only — no lock needed.
    exists, result_data = _check_existing_trial_and_build_result(
        model_id=model_id,
        run_number=run_number,
        scenario=scenario,
        temperature=temperature,
        reasoning_model=reasoning_model,
        cache=cache,
        sector=sector,
    )
    if exists:
        return True, True, result_data

    # Sleep only when an actual LLM call is about to be made, to stagger
    # concurrent requests and reduce provider-side rate-limit pressure.
    time.sleep(random.uniform(0.05, 0.2))

    system_prompt, prompt_named, prompt_masked = construct_prompts(
        scenario,
        prompt_templates,
    )

    max_retries = 5
    base_delay = 0.5
    for attempt in range(max_retries):
        try:
            with trace_context(
                trace_name="brand-masking-experiment",
                tags=["genai", "research_group", "brand_masking_experiment"],
            ) as trace:
                trace.update_current_trace(
                    input={
                        "model_id": model_id,
                        "run_number": run_number,
                        "scenario_id": scenario.scenario_id,
                        "company_id": scenario.company_id,
                    },
                )
                response_named_raw, reasoning_tokens_named = run_llm_call(
                    prompt_template=prompt_template,
                    llm=llm,
                    parameters={
                        "system_prompt": system_prompt,
                        "user_prompt": prompt_named,
                    },
                )
                response_masked_raw, reasoning_tokens_masked = run_llm_call(
                    prompt_template=prompt_template,
                    llm=llm,
                    parameters={
                        "system_prompt": system_prompt,
                        "user_prompt": prompt_masked,
                    },
                )

                score_named = extract_integer_score(response_named_raw)
                score_masked = extract_integer_score(response_masked_raw)
                score_delta = (
                    float(score_named - score_masked)
                    if score_named is not None and score_masked is not None
                    else None
                )

                trace.update_current_trace(
                    output={
                        "score_named": score_named,
                        "score_masked": score_masked,
                        "score_delta": score_delta,
                    },
                )

            with _db_lock:
                exists, result_data = _check_existing_trial_and_build_result(
                    model_id=model_id,
                    run_number=run_number,
                    scenario=scenario,
                    temperature=temperature,
                    reasoning_model=reasoning_model,
                    cache=cache,
                    sector=sector,
                )
                if exists:
                    return True, True, result_data

                insert_brand_masking_trial(
                    run=run_number,
                    scenario_id=scenario.scenario_id,
                    company_id=scenario.company_id,
                    base_scenario_text=scenario.base_scenario_text,
                    incumbent_name=scenario.incumbent_name,
                    masked_name=scenario.masked_name,
                    company_description=scenario.company_description,
                    system_prompt=system_prompt,
                    prompt_named=prompt_named,
                    prompt_masked=prompt_masked,
                    response_named_raw=response_named_raw,
                    response_masked_raw=response_masked_raw,
                    score_named=score_named,
                    score_masked=score_masked,
                    score_delta=score_delta,
                    named_parse_success=score_named is not None,
                    masked_parse_success=score_masked is not None,
                    reasoning_tokens_named=reasoning_tokens_named,
                    reasoning_tokens_masked=reasoning_tokens_masked,
                    model_id=model_id,
                    reasoning_model=reasoning_model,
                    temperature=temperature,
                    sector=sector,
                    debug=True,
                )

            result_data = {
                "run": run_number,
                **asdict(scenario),
                "system_prompt": system_prompt,
                "prompt_named": prompt_named,
                "prompt_masked": prompt_masked,
                "response_named_raw": response_named_raw,
                "response_masked_raw": response_masked_raw,
                "score_named": score_named,
                "score_masked": score_masked,
                "score_delta": score_delta,
                "named_parse_success": score_named is not None,
                "masked_parse_success": score_masked is not None,
                "reasoning_tokens_named": reasoning_tokens_named,
                "reasoning_tokens_masked": reasoning_tokens_masked,
                "model_id": model_id,
                "reasoning_model": reasoning_model,
                "temperature": temperature,
            }
            return True, False, result_data
        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)
            if "ThrottlingException" in error_msg and attempt < max_retries - 1:
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                time.sleep(delay)
                continue

            print(
                "Error on scenario "
                f"{scenario.scenario_id} with model {model_id}: {exc}",
            )
            return False, False, None

    return False, False, None


def _save_run_results(results_data: list[dict], run_number: int, sector: str) -> None:
    """Persist per-run raw results to a timestamped CSV file."""
    if not results_data:
        return

    output_dir = (
        DATA_DIR / "brand_masking" / sector / datetime.now().strftime("%Y%m%d_%H%M")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"run_{run_number}_results.csv"
    pd.DataFrame(results_data).to_csv(output_path, index=False)
    print(f"    ✓ Results saved to {output_path}")


def _initialize_run_components(
    llm_models: list[dict],
    trace_tags: list[str],
    run_number: int,
    sector: str,
) -> tuple[dict, ChatPromptTemplate, dict]:
    """Create the LLM clients, prompt template, and dedup cache for one run."""
    from langchain_core.prompts import ChatPromptTemplate

    from bias_in_llms.llm.llm_calling import get_llm

    llm_instances = {
        model["model_id"]: get_llm(
            model["model_id"],
            model["model_type"],
            temperature=model["temperature"],
            max_tokens=10,
        )
        for model in llm_models
    }
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            ("human", "{user_prompt}"),
        ],
    )
    cache = _load_existing_trials_cache(llm_models, run_number, sector)
    return llm_instances, prompt_template, cache


def _execute_run(
    *,
    tasks: list[tuple[BrandMaskingScenario, dict]],
    llm_instances: dict,
    prompt_template: ChatPromptTemplate,
    prompt_templates: dict[str, str],
    run_number: int,
    cache: dict,
    max_workers: int,
    sector: str,
) -> list[dict]:
    """Execute one full run in parallel and return the collected result rows."""
    from tqdm import tqdm

    results_data: list[dict] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        worker = partial(
            _run_single_trial,
            llm_instances=llm_instances,
            prompt_template=prompt_template,
            prompt_templates=prompt_templates,
            run_number=run_number,
            cache=cache,
            sector=sector,
        )
        future_to_task = {executor.submit(worker, task): task for task in tasks}

        for future in tqdm(
            concurrent.futures.as_completed(future_to_task),
            total=len(tasks),
            desc=f"Run {run_number}",
        ):
            try:
                success, _, result_data = future.result()
                if success and result_data:
                    results_data.append(result_data)
            except Exception as exc:  # noqa: BLE001
                task = future_to_task[future]
                print(f"Error processing task {task}: {exc}")

    return results_data


def run_full_brand_masking_experiment(config_path: Path) -> None:
    """Run the paired named-vs-masked brand-masking experiment."""
    config = load_brand_masking_config(config_path)
    sector = config.sector

    initialize_brand_masking_db(sector)

    prompt_templates = load_prompt_templates(
        str(config.resolve_path(config.prompts_path)),
    )
    scenarios = load_brand_masking_scenarios(config)
    llm_models = _prepare_llm_models(config)

    trace_tags = ["genai", "brand_masking_experiment"]
    tasks = [(scenario, model) for model in llm_models for scenario in scenarios]

    print("🧪 BRAND MASKING EXPERIMENT")
    print(f"  • Sector: {sector}")
    print(f"  • Scenarios: {len(scenarios)}")
    print(f"  • Models: {len(llm_models)}")
    print(f"  • Runs: {config.runs}")
    print(f"  • Total paired trials per run: {len(tasks)}")
    print(f"  • Max workers: {config.max_workers}")
    print()

    for run_number in range(1, config.runs + 1):
        print(f"\n{'=' * 60}")
        print(f"Starting Run {run_number}/{config.runs}")
        print(f"{'=' * 60}")

        llm_instances, prompt_template, cache = _initialize_run_components(
            llm_models,
            trace_tags,
            run_number,
            sector,
        )

        results_data = _execute_run(
            tasks=tasks,
            llm_instances=llm_instances,
            prompt_template=prompt_template,
            prompt_templates=prompt_templates,
            run_number=run_number,
            cache=cache,
            max_workers=config.max_workers,
            sector=sector,
        )
        _save_run_results(results_data, run_number, sector)
