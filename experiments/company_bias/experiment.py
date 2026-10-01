"""This script runs the company bias experiment."""

import concurrent.futures
import random
import threading
import time
from functools import partial
from pathlib import Path

import yaml
from aily_ai_brain.common.enums import (
    BedrockModelID,
    OpenAIModelID,
    OpenAIReasoningModels,
    OpenRouterModelID,
    RecommendedModels,
)
from aily_ai_brain.langfuse.utils import langfuse_trace
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm

from aily_bias_in_llms.config.project_paths import EXPERIMENTS_DIR
from aily_bias_in_llms.database import query_db
from aily_bias_in_llms.database.company_bias_db import SectorBiasDB
from aily_bias_in_llms.llm.llm_calling import get_llm, run_llm_call
from aily_bias_in_llms.llm.prompt.prompt_loading import load_prompt_templates
from aily_bias_in_llms.utils.output_parsers import BinaryChoiceOutputParser
from aily_bias_in_llms.utils.utils import generate_randomized_pairs

_db_lock = threading.Lock()

# Mapping of sector names to their table name prefixes
SECTOR_TABLE_MAP = {
    "pharma": "pharma",
    "consulting": "consulting",
    "cloud_provider": "cloud_provider",
    "banking": "banking",
}


def _load_existing_combinations_cache(
    models: list[dict],
    run_number: int,
    sector_db: SectorBiasDB,
) -> dict:
    """
    Load all existing combinations for the given models and run number into a cache.

    Args:
        models: List of model configurations
        run_number: The run number
        sector_db: The SectorBiasDB instance for the current sector

    Returns:
        dict: Cache with key (model_id, option_a, option_b, temperature, question)
              -> result_data
    """
    cache = {}

    for model in models:
        model_id = model["model_id"]
        temperature = model["temperature"]

        query = f"""
            SELECT option_a, option_b, question, preceding_context_block, answer,
            full_prompt, reasoning_tokens, reasoning_text
            FROM {sector_db.full_table_name}
            WHERE model_id = %s AND run = %s AND temperature = %s
        """

        df = query_db(query, (model_id, run_number, temperature))

        for row in df.to_dict("records"):
            key = (
                model_id,
                row["option_a"],
                row["option_b"],
                temperature,
                row["question"],
                row["preceding_context_block"],
            )
            cache[key] = {
                "answer": row["answer"],
                "question": row["question"],
                "preceding_context_block": row["preceding_context_block"],
                "full_prompt": row["full_prompt"],
                "reasoning_tokens": row["reasoning_tokens"],
                "reasoning_text": row.get("reasoning_text", ""),
            }

    print(
        f"📊 Loaded {len(cache)} existing combinations into cache for run {run_number}",
    )
    return cache


def _check_existing_combination_and_build_result(
    model_id: str,
    run_number: int,
    option_a: str,
    option_b: str,
    temperature: float,
    question: str,
    reasoning_model: bool,
    cache: dict | None = None,
    preceding_context_block: str = "",
    sector_db: SectorBiasDB | None = None,
) -> tuple[bool, dict | None]:
    """
    Check if a combination exists in the cache/database and build result data.

    Args:
        model_id: The model identifier
        run_number: The run number
        option_a: The text for option A
        option_b: The text for option B
        temperature: The temperature setting
        question: The question text
        reasoning_model: Whether reasoning model was used
        cache: Optional cache dict to use instead of database queries
        preceding_context_block: The preceding context block
        sector_db: The SectorBiasDB instance for database queries

    Returns:
        tuple[bool, dict | None]: (exists, result_data)
        - exists: True if combination exists
        - result_data: Dictionary with existing data if found, None otherwise
    """
    # Use cache if provided (much faster)
    if cache is not None:
        key = (
            model_id,
            option_a,
            option_b,
            temperature,
            question,
            preceding_context_block,
        )
        if key in cache:
            existing_data = cache[key]
            result_data = {
                "run": run_number,
                "question": existing_data["question"],
                "option_A": option_a,
                "option_B": option_b,
                "answer": existing_data["answer"],
                "model_id": model_id,
                "reasoning_model": reasoning_model,
                "reasoning_tokens": existing_data["reasoning_tokens"],
                "temperature": temperature,
                "full_prompt": existing_data["full_prompt"],
                "preceding_context_block": existing_data["preceding_context_block"],
                "reasoning_text": existing_data["reasoning_text"],
            }
            return True, result_data
        return False, None

    # Fallback to database queries (slower, for backward compatibility)
    if sector_db is None:
        return False, None

    if sector_db.combination_exists(
        model_id=model_id,
        run=run_number,
        option_a=option_a,
        option_b=option_b,
        temperature=temperature,
        question=question,
        preceding_context_block=preceding_context_block,
    ):
        existing_data = sector_db.get_existing_combination(
            model_id=model_id,
            run=run_number,
            option_a=option_a,
            option_b=option_b,
            temperature=temperature,
            question=question,
            preceding_context_block=preceding_context_block,
        )
        if existing_data:
            result_data = {
                "run": run_number,
                "question": existing_data["question"],
                "option_A": option_a,
                "option_B": option_b,
                "answer": existing_data["answer"],
                "model_id": model_id,
                "reasoning_model": reasoning_model,
                "reasoning_tokens": existing_data["reasoning_tokens"],
                "temperature": temperature,
                "full_prompt": existing_data["full_prompt"],
                "preceding_context_block": existing_data.get(
                    "preceding_context_block",
                    preceding_context_block,
                ),
                "reasoning_text": existing_data.get("reasoning_text", ""),
            }
            return True, result_data
        else:
            return True, None
    return False, None


def _run_single_experiment(
    task: tuple,
    llm_instances: dict,
    prompt_template: PromptTemplate,
    parser: PydanticOutputParser,
    run_number: int,
    cache: dict | None = None,
    sector_db: SectorBiasDB | None = None,
) -> tuple[bool, bool, dict | None]:  # Returns (success, was_existing, result_data)
    """Runs a single experiment for a given pair, question, and model."""
    # Add random delay to prevent rate limiting (0.05 to 0.2 seconds)
    time.sleep(random.uniform(0.05, 0.2))

    company_pair, question, model_config, preceding_context_block = task
    option_a, option_b = company_pair
    model_id = model_config["model_id"]
    reasoning_model = model_config["reasoning_model"]
    temperature = model_config["temperature"]
    llm = llm_instances[model_id]

    full_prompt = prompt_template.format(
        preceding_context_block=preceding_context_block,
        question=question,
        opt_A_text=option_a,
        opt_B_text=option_b,
    )
    reasoning_tokens = None

    # Exponential backoff retry logic
    max_retries = 5
    base_delay = 0.5  # Start with 0.5 seconds

    for attempt in range(max_retries):
        try:
            with langfuse_trace(
                trace_name="company-bias-experiment",
                tags=["research_group", "company_bias_experiment"],
            ) as company_bias_span:
                company_bias_span.update_current_trace(
                    input={
                        "model_id": model_id,
                        "run_number": run_number,
                        "option_a": option_a,
                        "option_b": option_b,
                        "preceding_context_block": preceding_context_block,
                    },
                )
                response, reasoning_tokens = run_llm_call(
                    prompt_template=prompt_template,
                    llm=llm,
                    parameters={
                        "preceding_context_block": preceding_context_block,
                        "question": question,
                        "opt_A_text": option_a,
                        "opt_B_text": option_b,
                    },
                    output_parser=parser,
                )
                choice = response.choice
                reasoning_text = response.reasoning
                company_bias_span.update_current_trace(
                    output={
                        "choice": choice,
                        "reasoning_text": reasoning_text,
                    },
                )

            # Second check: Post-LLM verification to prevent race conditions
            with _db_lock:
                exists, result_data = _check_existing_combination_and_build_result(
                    model_id,
                    run_number,
                    option_a,
                    option_b,
                    temperature,
                    question,
                    reasoning_model,
                    cache,
                    preceding_context_block,
                    sector_db,
                )
                if exists:
                    # Another thread inserted this combination while we
                    # were making the LLM call
                    return (True, True, result_data)  # success=True, was_existing=True

                # Safe to insert using the sector-specific DB
                sector_db.insert_row(
                    model_id=model_id,
                    run=run_number,
                    full_prompt=full_prompt,
                    question=question,
                    option_a=option_a,
                    option_b=option_b,
                    answer=choice,
                    reasoning_model=reasoning_model,
                    reasoning_tokens=reasoning_tokens,
                    temperature=temperature,
                    preceding_context_block=preceding_context_block,
                    reasoning_text=reasoning_text,
                    debug=False,
                )

            result_data = {
                "run": run_number,
                "question": question,
                "option_A": option_a,
                "option_B": option_b,
                "answer": choice,
                "model_id": model_id,
                "reasoning_model": reasoning_model,
                "reasoning_tokens": reasoning_tokens,
                "temperature": temperature,
                "full_prompt": full_prompt,
                "preceding_context_block": preceding_context_block,
                "reasoning_text": reasoning_text,
            }
            return (
                True,
                False,
                result_data,
            )  # success=True, was_existing=False, with new data

        except Exception as e:
            error_msg = str(e)
            if "ThrottlingException" in error_msg and attempt < max_retries - 1:
                # Calculate exponential backoff delay
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                print(
                    f"Rate limited on pair ({option_a}, {option_b}) "
                    f"with model {model_id}. "
                    f"Retrying in {delay:.1f} seconds... "
                    f"(attempt {attempt + 1}/{max_retries})",
                )
                time.sleep(delay)
                continue
            else:
                print(
                    f"Error on pair ({option_a}, {option_b}) "
                    f"with model {model_id}: {e}",
                )
                return False, False, None  # success=False, was_existing=False, no data

    return False, False, None  # success=False, was_existing=False, no data


def _prepare_llm_models(config: dict, use_recommended_models: bool) -> list[dict]:
    """Prepare the list of LLM models for the experiment."""
    if use_recommended_models:
        print(
            "🧠 Using recommended models for the experiment.",
        )
        llm_models = []
        reasoning_models_set = set(OpenAIReasoningModels)
        for model in RecommendedModels:
            llm_models.append(
                {
                    "model_id": model.value,
                    "model_enum": model,
                    "reasoning_model": model in reasoning_models_set,
                    "temperature": 0.0,
                },
            )
        return llm_models
    else:
        print("🧠 Using models from config.yaml.")
        llm_models_from_config = config["llm_models"]

        model_type_map = {
            "BedrockModelID": BedrockModelID,
            "OpenAIModelID": OpenAIModelID,
            "OpenRouterModelID": OpenRouterModelID,
        }

        processed_llm_models = []
        for model_config in llm_models_from_config:
            model_name = model_config["model_id"]
            model_type_str = model_config["model_type"]

            model_enum_class = model_type_map.get(model_type_str)
            if not model_enum_class:
                raise ValueError(f"Invalid model type in config: {model_type_str}")

            try:
                model_enum_member = getattr(model_enum_class, model_name)
            except AttributeError:
                raise ValueError(
                    f"Model '{model_name}' not found in enum '{model_type_str}'",
                ) from None

            processed_llm_models.append(
                {
                    "model_id": model_enum_member.value,
                    "model_enum": model_enum_member,
                    "reasoning_model": model_config.get("reasoning_model", False),
                    "temperature": model_config.get("temperature", 0.0),
                },
            )
        return processed_llm_models


def _load_sector_config(config_dir: Path, sector: str) -> dict:
    """
    Load the sector-specific configuration file.

    Args:
        config_dir: Path to the config directory
        sector: The sector name (e.g., "pharma", "consulting")

    Returns:
        dict: The sector-specific configuration
    """
    sector_config_path = config_dir / f"{sector}_config.yaml"
    if not sector_config_path.exists():
        raise FileNotFoundError(
            f"Sector config file not found: {sector_config_path}. "
            f"Please create {sector}_config.yaml with 'companies' and 'questions' keys",
        )

    with open(sector_config_path) as f:
        return yaml.safe_load(f)


def _merge_configs(main_config: dict, sector_config: dict) -> dict:
    """
    Merge main config with sector-specific config.

    The sector config provides companies and questions, while the main config
    provides experiment parameters (models, num_pairs, runs, etc.).

    Args:
        main_config: The main configuration (models, num_pairs, runs, etc.)
        sector_config: The sector-specific configuration (companies, questions)

    Returns:
        dict: Merged configuration
    """
    merged = main_config.copy()
    # Add sector-specific keys
    merged["companies"] = sector_config.get("companies", [])
    merged["questions"] = sector_config.get("questions", [])
    # Optional: prompt_preceding_context_block from sector config
    if "prompt_preceding_context_block" in sector_config:
        merged["prompt_preceding_context_block"] = sector_config[
            "prompt_preceding_context_block"
        ]
    return merged


def _setup_experiment(
    config_path: Path,
    use_recommended_models: bool,
) -> tuple[dict, list[dict], list[str], list[tuple[str, str]], str]:
    """
    Set up the experiment by loading config and preparing models and pairs.

    Args:
        config_path: Path to the main configuration file
        use_recommended_models: Whether to use recommended models

    Returns:
        tuple: (config, llm_models, companies, company_pairs, sector)
    """
    with open(config_path) as f:
        main_config = yaml.safe_load(f)

    # Get sector from main config (defaults to "pharma" for backwards compatibility)
    sector = main_config.get("sector", "pharma")
    print(f"🏢 Running experiment for sector: {sector}")

    # Load sector-specific config
    config_dir = config_path.parent
    sector_config = _load_sector_config(config_dir, sector)

    # Merge configs
    config = _merge_configs(main_config, sector_config)

    llm_models = _prepare_llm_models(config, use_recommended_models)

    companies = config["companies"]
    companies = [company["name"] for company in companies]
    company_pairs = generate_randomized_pairs(
        companies,
        config["num_pairs"],
    )

    return config, llm_models, companies, company_pairs, sector


def _print_experiment_info(
    config: dict,
    llm_models: list[dict],
    companies: list[str],
    company_pairs: list[tuple[str, str]],
    sector: str,
) -> None:
    """Print experiment setup information."""
    num_runs = config["runs"]
    total_calls_per_run = (
        len(company_pairs) * len(config["questions"]) * len(llm_models)
    )
    total_calls = total_calls_per_run * num_runs

    print(f"🧪 {sector.upper()} BIAS EXPERIMENT (FULLY PARALLEL)")
    print(f"  • Sector: {sector}")
    print(f"  • Companies: {len(companies)}")
    print(f"  • Pairs: {len(company_pairs)}")
    print(f"  • Questions: {len(config['questions'])}")
    print(f"  • Models: {len(llm_models)}")
    print(f"  • Runs: {num_runs}")
    print(f"  • Total LLM calls per run: {total_calls_per_run}")
    print(f"  • Total LLM calls (all runs): {total_calls}")
    print(f"  • Max workers: {config.get('max_workers', 10)}")
    print()


def _initialize_run_components(
    llm_models: list[dict],
    langfuse_tags: list[str],
    run_number: int,
    sector_db: SectorBiasDB,
) -> tuple[dict, PromptTemplate, PydanticOutputParser, dict]:
    """
    Initialize LLM instances, prompt template, parser, and cache for a run.

    Args:
        llm_models: List of model configurations
        langfuse_tags: Tags for langfuse tracing
        run_number: The run number
        sector_db: The SectorBiasDB instance for the current sector

    Returns:
        tuple: (llm_instances, prompt_template, parser, combinations_cache)
    """
    llm_instances = {}
    for model in llm_models:
        model_id = model["model_id"]
        model_enum = model["model_enum"]
        print(f"🔧 Initializing LLM: {model_id}")
        llm_instances[model_id] = get_llm(
            model_enum,
            langfuse_tags,
            temperature=model["temperature"],
            max_tokens=model.get("max_tokens", 2000),
        )

    prompt_path = EXPERIMENTS_DIR / "company_bias" / "config" / "prompts.yaml"
    parser = PydanticOutputParser(pydantic_object=BinaryChoiceOutputParser)
    partial_variables = {"format_instructions": parser.get_format_instructions()}
    prompt = load_prompt_templates(str(prompt_path))["prompt_template_variation"]
    prompt_template = PromptTemplate.from_template(
        template=prompt,
        partial_variables=partial_variables,
    )

    print("🗂️  Loading existing combinations cache...")
    combinations_cache = _load_existing_combinations_cache(
        llm_models,
        run_number,
        sector_db,
    )

    return llm_instances, prompt_template, parser, combinations_cache


def _generate_tasks(
    config: dict,
    company_pairs: list[tuple[str, str]],
    llm_models: list[dict],
) -> list[tuple]:
    """
    Generate all tasks for a run based on config.

    Returns:
        list: List of tasks, where each task is (pair, question, model, instruction)
    """
    tasks = []
    preceding_context_block_list = config.get("prompt_preceding_context_block") or []

    # Use unified "questions" key from merged config
    questions = config.get("questions", [])

    if len(preceding_context_block_list) > 0:
        print(
            f"🔍 Using {len(preceding_context_block_list)} preceding context blocks",
        )
        for preceding_context_block in preceding_context_block_list:
            for question in questions:
                for model in llm_models:
                    for pair in company_pairs:
                        company_b_name = pair[1]
                        instruction = preceding_context_block["instruction"].format(
                            company_b_name=company_b_name,
                        )
                        tasks.append((pair, question, model, instruction))
    else:
        # No preceding context blocks - just iterate over questions
        for question in questions:
            for model in llm_models:
                for pair in company_pairs:
                    tasks.append((pair, question, model, ""))

    return tasks


def _filter_existing_tasks(
    tasks: list[tuple],
    cache: dict,
    run_number: int,
) -> tuple[list[tuple], list[dict]]:
    """
    Split tasks into those already in cache and those that need LLM calls.

    Args:
        tasks: Full list of tasks generated by `_generate_tasks`
        cache: Pre-loaded combinations cache from `_load_existing_combinations_cache`
        run_number: The run number

    Returns:
        tuple: (new_tasks, existing_results)
        - new_tasks: Tasks whose combination is not yet in the cache
        - existing_results: Pre-built result dicts for already-cached combinations
    """
    new_tasks = []
    existing_results = []

    for task in tasks:
        company_pair, question, model_config, preceding_context_block = task
        option_a, option_b = company_pair
        model_id = model_config["model_id"]
        temperature = model_config["temperature"]
        reasoning_model = model_config["reasoning_model"]

        exists, result_data = _check_existing_combination_and_build_result(
            model_id,
            run_number,
            option_a,
            option_b,
            temperature,
            question,
            reasoning_model,
            cache,
            preceding_context_block,
        )

        if exists and result_data:
            existing_results.append(result_data)
        elif not exists:
            new_tasks.append(task)

    return new_tasks, existing_results


def _execute_run(
    tasks: list[tuple],
    llm_instances: dict,
    prompt_template: PromptTemplate,
    parser: PydanticOutputParser,
    run_number: int,
    combinations_cache: dict,
    config: dict,
    sector_db: SectorBiasDB,
) -> tuple[int, int, int, list[dict]]:
    """
    Execute a single run of experiments in parallel.

    Args:
        tasks: List of tasks to execute
        llm_instances: Dictionary of LLM instances
        prompt_template: The prompt template
        parser: The output parser
        run_number: The run number
        combinations_cache: Cache of existing combinations
        config: The experiment configuration
        sector_db: The SectorBiasDB instance for the current sector

    Returns:
        tuple: (successful_runs, existing_count, new_experiments, results_data)
    """
    print(
        f"🚀 Running {len(tasks)} experiments in parallel for run {run_number}...",
    )

    successful_runs = 0
    existing_count = 0
    new_experiments = 0
    results_data = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config.get("max_workers", 10),
    ) as executor:
        worker_func = partial(
            _run_single_experiment,
            llm_instances=llm_instances,
            prompt_template=prompt_template,
            parser=parser,
            run_number=run_number,
            cache=combinations_cache,
            sector_db=sector_db,
        )

        future_to_task = {executor.submit(worker_func, task): task for task in tasks}

        for future in tqdm(
            concurrent.futures.as_completed(future_to_task),
            total=len(tasks),
            desc=f"Run {run_number}",
        ):
            try:
                success, was_existing, result_data = future.result()
                if success:
                    successful_runs += 1
                    if was_existing:
                        existing_count += 1
                    else:
                        new_experiments += 1

                    if result_data:
                        results_data.append(result_data)

            except Exception as e:
                task = future_to_task[future]
                print(f"Error processing task {task}: {e}")

    print(
        f"✓ Run {run_number} completed: {successful_runs}/{len(tasks)} successful",
    )
    print(
        f"    ✓ Processed {new_experiments} new experiments, {existing_count} from DB",
    )

    return successful_runs, existing_count, new_experiments, results_data


def _save_run_results(
    results_data: list[dict],
    run_number: int,
    sector: str,
) -> None:
    """Save run results to CSV file."""
    if not results_data:
        return

    from datetime import datetime

    import pandas as pd

    from aily_bias_in_llms.config.project_paths import DATA_DIR

    results_df = pd.DataFrame(results_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    # Save to sector-specific directory
    output_dir = DATA_DIR / f"{sector}_bias" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"run_{run_number}_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"    ✓ Results saved to {out_csv}")


def _print_final_summary(
    start_time: float,
    sector: str,
    sector_db: SectorBiasDB,
) -> None:
    """Print final experiment summary."""
    end_time = time.time()
    duration_seconds = end_time - start_time
    minutes = int(duration_seconds // 60)
    seconds = int(duration_seconds % 60)

    print(f"\n{'=' * 60}")
    print("All runs completed!")
    print(f"{'=' * 60}")
    print()
    print(f"⏱️  Total experiment duration: {minutes} minutes and {seconds} seconds.")
    print(f"🎉 {sector.capitalize()} bias experiment completed!")

    # Query the sector-specific table for final stats
    query = f"SELECT * FROM {sector_db.full_table_name}"
    data = query_db(query)
    print(f"📊 Final Results: {len(data)} rows in {sector_db.full_table_name}")
    if not data.empty:
        print(f"   • Successful: {len(data[data['answer'] != 'ERROR'])}")
        print(f"   • Errors: {len(data[data['answer'] == 'ERROR'])}")


def run_full_company_bias_experiment(
    config_path: Path,
    use_recommended_models: bool = False,
):
    """
    Runs the entire company bias experiment based on a configuration file.

    The experiment reads the `sector` field from the main config to determine
    which sector-specific config to load (e.g., pharma_config.yaml or
    consulting_config.yaml) and which database table to use.

    Args:
        config_path (Path): Path to the main configuration YAML file.
        use_recommended_models (bool): Whether to use recommended models.
    """
    start_time = time.time()

    # Setup experiment (now returns sector as well)
    config, llm_models, companies, company_pairs, sector = _setup_experiment(
        config_path,
        use_recommended_models,
    )

    # Print experiment information
    _print_experiment_info(config, llm_models, companies, company_pairs, sector)

    # Get the database table prefix for this sector
    table_prefix = SECTOR_TABLE_MAP.get(sector, sector)
    sector_db = SectorBiasDB(table_prefix)

    # Initialize the sector-specific database table
    sector_db.initialize()

    langfuse_tags = ["genai", f"{sector}_bias_experiment"]
    num_runs = config["runs"]

    # Execute all runs
    for run_number in range(1, num_runs + 1):
        print(f"\n{'=' * 60}")
        print(f"Starting Run {run_number}/{num_runs}")
        print(f"{'=' * 60}")

        # Initialize run components
        llm_instances, prompt_template, parser, combinations_cache = (
            _initialize_run_components(llm_models, langfuse_tags, run_number, sector_db)
        )

        # Generate tasks
        tasks = _generate_tasks(config, company_pairs, llm_models)

        # Pre-filter tasks: separate already-cached from new
        new_tasks, existing_results = _filter_existing_tasks(
            tasks,
            combinations_cache,
            run_number,
        )
        print(
            f"    ✓ {len(existing_results)} already in DB, "
            f"{len(new_tasks)} need LLM calls",
        )

        # Execute run with only the new tasks
        _, _, _, new_results_data = _execute_run(
            new_tasks,
            llm_instances,
            prompt_template,
            parser,
            run_number,
            combinations_cache,
            config,
            sector_db,
        )

        # Save results
        results_data = existing_results + new_results_data
        _save_run_results(results_data, run_number, sector)

    # Print final summary
    _print_final_summary(start_time, sector, sector_db)


if __name__ == "__main__":
    run_full_company_bias_experiment(
        config_path=EXPERIMENTS_DIR / "company_bias" / "config" / "config.yaml",
        use_recommended_models=False,
    )
