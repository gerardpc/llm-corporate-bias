"""This module contains functions for calling LLMs."""

import json
import os
import re
from typing import Any

import tiktoken
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import BasePromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from bias_in_llms.config.environment import load_environment

OPENAI_MODEL_ALIASES = {
    "GPT_5": "gpt-5",
    "GPT_5_MINI": "gpt-5-mini",
    "GPT41": "gpt-4.1",
    "GPT41Mini": "gpt-4.1-mini",
    "GPT41Nano": "gpt-4.1-nano",
    "GPT4O": "gpt-4o",
    "GPT4O_20241120": "gpt-4o-2024-11-20",
    "GPT4O_20240806": "gpt-4o-2024-08-06",
    "GPT4O_mini": "gpt-4o-mini",
    "O3": "o3",
    "O4Mini": "o4-mini",
}

BEDROCK_MODEL_ALIASES = {
    "Claude35Sonnet20241022V2_0": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "Claude37Sonnet20250219V1_0": "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "Claude3Haiku20240307V1_0": "anthropic.claude-3-haiku-20240307-v1:0",
    "Claude4Sonnet20250514V1_0": "anthropic.claude-sonnet-4-20250514-v1:0",
    "ClaudeSonnet45": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "Claude45Haiku": "anthropic.claude-haiku-4-5-20251001-v1:0",
    "ClaudeHaiku45": "anthropic.claude-haiku-4-5-20251001-v1:0",
}

OPENROUTER_MODEL_ALIASES = {
    "DEEPSEEK_R1": "deepseek/deepseek-r1",
    "QWEN3_6_PLUS": "qwen/qwen3.6-plus",
}

MODEL_TYPE_ALIASES = {
    "openai": "openai",
    "OpenAIModelID": "openai",
    "OpenAIReasoningModels": "openai",
    "bedrock": "bedrock",
    "BedrockModelID": "bedrock",
    "openrouter": "openrouter",
    "OpenRouterModelID": "openrouter",
}


def _preprocess_json_response(text: str) -> str:
    """
    Extract and normalise JSON from an LLM response.

    Handles two common quirks of thinking/reasoning models (e.g. Qwen3):

    1. Chain-of-thought or thinking tokens appearing *before* the JSON block —
       we locate the last ``{...}`` span so the preamble is ignored.
    2. JavaScript-style unquoted object keys (``{ choice: "B" }``) — we add
       double-quotes around bare identifiers so ``json.loads`` accepts them.
    """
    # Extract the outermost last {...} block, skipping any thinking preamble
    start = text.rfind("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]

    # Fast path: already valid JSON
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Fix unquoted object keys: { choice: "B" } → { "choice": "B" }
    fixed = re.sub(r"(?<=[{,])\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", r' "\1":', text)
    return fixed


def count_tokens(text, model_name="gpt-4"):
    """
    Count the number of tokens in a text.

    Args:
        text (str): The text to count the tokens of.
        model_name (str): The model name to use for the encoding.
    """
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _extract_response_text(response: Any) -> str:
    """Extract plain text from a LangChain chat response."""
    if isinstance(response.content, str):
        return response.content

    if isinstance(response.content, list):
        text_chunks: list[str] = []
        for chunk in response.content:
            if isinstance(chunk, str):
                text_chunks.append(chunk)
            elif isinstance(chunk, dict) and "text" in chunk:
                text_chunks.append(str(chunk["text"]))

        if text_chunks:
            return "".join(text_chunks).strip()

    return str(response.content).strip()


def run_llm_call(
    prompt_template: BasePromptTemplate,
    llm: BaseChatModel,
    parameters: dict,
    output_parser: BaseModel | None = None,
) -> str:
    """
    Run a llm call with a given prompt and input.

    Args:
        prompt_template (BasePromptTemplate): The prompt template to use.
        llm: The llm to use for the call.
        parameters (dict): The parameters for the prompt.
        output_parser (BaseModel | None): The output parser to use for the llm call.

    Returns:
        str: The output of the llm call.
    """
    chain = prompt_template | llm

    response = chain.invoke(parameters)
    usage_metadata = getattr(response, "usage_metadata", {}) or {}
    output_tokens = usage_metadata.get("output_tokens")

    response_text = _extract_response_text(response)
    reasoning_tokens = None
    if output_tokens is not None:
        reasoning_tokens = output_tokens - count_tokens(response_text)

    if output_parser:
        content = output_parser.parse(_preprocess_json_response(response_text))
    else:
        content = response_text

    return content, reasoning_tokens


def normalize_model_type(model_type: str) -> str:
    """Normalize config model type values to provider names."""
    provider = MODEL_TYPE_ALIASES.get(model_type)
    if provider is None:
        valid_types = ", ".join(sorted(MODEL_TYPE_ALIASES))
        raise ValueError(
            f"Unknown model_type '{model_type}'. Valid values: {valid_types}",
        )
    return provider


def resolve_model_id(model_id: str, model_type: str) -> str:
    """Resolve project-friendly model aliases to provider model IDs."""
    provider = normalize_model_type(model_type)
    alias_maps = {
        "openai": OPENAI_MODEL_ALIASES,
        "bedrock": BEDROCK_MODEL_ALIASES,
        "openrouter": OPENROUTER_MODEL_ALIASES,
    }
    return alias_maps[provider].get(model_id, model_id)


def _get_openai_llm(
    model_id: str,
    temperature: float,
    max_tokens: int,
) -> ChatOpenAI:
    """Create an OpenAI chat model."""
    return ChatOpenAI(
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _get_openrouter_llm(
    model_id: str,
    temperature: float,
    max_tokens: int,
) -> ChatOpenAI:
    """Create an OpenRouter chat model through the OpenAI-compatible API."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY must be set for OpenRouter models.")

    return ChatOpenAI(
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )


def _get_bedrock_llm(
    model_id: str,
    temperature: float,
    max_tokens: int,
) -> BaseChatModel:
    """Create an AWS Bedrock chat model."""
    try:
        from langchain_aws import ChatBedrock
    except ImportError as exc:
        raise RuntimeError(
            "Install langchain-aws to use Bedrock models.",
        ) from exc

    model_kwargs = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    return ChatBedrock(
        model_id=model_id,
        region_name=os.getenv("AWS_REGION"),
        model_kwargs=model_kwargs,
    )


def get_llm(
    model_id: str,
    model_type: str,
    temperature: float = 0.0,
    max_tokens: int = 2000,
) -> BaseChatModel:
    """Create a LangChain chat model from provider-neutral config values."""
    load_environment()

    provider = normalize_model_type(model_type)
    resolved_model_id = resolve_model_id(model_id, provider)

    if provider == "openai":
        return _get_openai_llm(resolved_model_id, temperature, max_tokens)
    if provider == "openrouter":
        return _get_openrouter_llm(resolved_model_id, temperature, max_tokens)
    if provider == "bedrock":
        return _get_bedrock_llm(resolved_model_id, temperature, max_tokens)

    raise ValueError(f"Unsupported model provider: {provider}")
