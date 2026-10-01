"""
Core *library* routine for the temporal-discounting experiment.

The function :pyfunc:`run_experiment` executes a batch of binary-choice prompts
against a given LLM instance, stores each result in the experiment database,
and returns the aggregated results as a :class:`pandas.DataFrame`.

There is intentionally **no** top-level side-effectful code in this module –
for an executable entry-point refer to :pymod:`experiments.temporal_discounting.run`.
"""

from pathlib import Path
from typing import Union

import pandas as pd
from langchain_community.chat_models import BedrockChat
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from bias_in_llms.config.project_paths import EXPERIMENTS_DIR
from bias_in_llms.database.temporal_discounting_db import (
    insert_temporal_discounting_row,
)
from bias_in_llms.llm.llm_calling import run_llm_call
from bias_in_llms.llm.prompt.prompt_loading import load_prompt_templates
from bias_in_llms.utils.output_parsers import BinaryChoiceOutputParser


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def run_experiment(
    llm: Union[ChatOpenAI, BedrockChat],
    option_pairs: list[tuple[str, str, float, float, int]],
    prompt_path: Path = EXPERIMENTS_DIR
    / "temporal_discounting"
    / "config"
    / "prompts.yaml",
    model_id: str | None = None,
    run: int = 1,
    reasoning_model: bool = False,
    temperature: float = 0.0,
) -> pd.DataFrame:
    """
    Run the temporal discounting experiment.

    Args:
        llm: The LLM model to use.
        option_pairs: List of tuples [(option_A_text, option_B_text), ...]
            for new experiments.
        prompt_path (Path): The path to the prompt file.
        model_id (str): The ID of the LLM model used.
        run (int): The run number for this experiment iteration.
        reasoning_model (bool): Whether the model is a reasoning model.
        temperature (float): The temperature setting.

    Returns:
        A dataframe with the results of the experiment.
    """
    # ---------------------------------------------------------------------
    # Prepare parser and prompt template once instead of inside the loop
    # ---------------------------------------------------------------------
    parser = PydanticOutputParser(pydantic_object=BinaryChoiceOutputParser)
    partial_variables = {"format_instructions": parser.get_format_instructions()}
    prompt_template_str = load_prompt_templates(str(prompt_path))[
        "direct_choice_template"
    ]
    prompt_template = ChatPromptTemplate.from_template(
        template=prompt_template_str,
        partial_variables=partial_variables,
    )
    question = "Choose one of the following options:"

    results: list[dict[str, str | float | int]] = []

    # Process option pairs
    for option_a, option_b, now_val, later_val, delay_years in option_pairs:
        full_prompt = prompt_template_str.format(
            preceding_context_block="",
            question=question,
            opt_A_text=option_a,
            opt_B_text=option_b,
            format_instructions=parser.get_format_instructions(),
        )
        try:
            response, reasoning_tokens = run_llm_call(
                prompt_template=prompt_template,
                llm=llm,
                parameters={
                    "preceding_context_block": "",
                    "question": question,
                    "opt_A_text": option_a,
                    "opt_B_text": option_b,
                },
                output_parser=parser,
            )
            choice = response.choice
        except Exception as e:
            print(f"Error on pair ({option_a}, {option_b}): {e}")
            choice = "ERROR"

        # Determine which textual option was chosen to map to 'Now' vs 'Later'
        chosen_text = option_a if choice == "A" else option_b
        preference = "Now" if "today" in chosen_text.lower() else "Later"

        results.append(
            {
                "Run": run,
                "Now ($)": now_val,
                "Later ($)": later_val,
                "Delay (years)": delay_years,
                "LLM": model_id,
                "Temperature": temperature,
                "Preference": preference,
            },
        )

        # Experiment-specific logging
        insert_temporal_discounting_row(
            model_id=model_id,
            run=run,
            now_value=now_val,
            later_value=later_val,
            delay_years=delay_years,
            preference=preference,
            full_prompt=full_prompt,
            question=question,
            option_A=option_a,
            option_B=option_b,
            answer=choice,
            reasoning_model=reasoning_model,
            reasoning_tokens=reasoning_tokens,
            temperature=temperature,
            debug=True,
        )

    return pd.DataFrame(results)
