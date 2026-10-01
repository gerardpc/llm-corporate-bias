"""Environment loading and diagnostics."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from bias_in_llms.config.project_paths import ROOT_DIR

DEFAULT_ENV_PATH = ROOT_DIR / ".env"
DATABASE_ENV_VARS = (
    "POSTGRES_HOST",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
)
OPTIONAL_ENV_VARS = (
    "POSTGRES_PORT",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "AWS_REGION",
    "AWS_PROFILE",
)


@lru_cache(maxsize=1)
def load_environment(env_path: str | Path = DEFAULT_ENV_PATH) -> bool:
    """Load local environment variables from .env if present."""
    resolved_path = Path(env_path)
    if not resolved_path.exists():
        return False

    return load_dotenv(resolved_path, override=False)


def _is_set(name: str) -> bool:
    """Return whether an environment variable is non-empty."""
    return bool(os.getenv(name))


def database_environment_status() -> tuple[str, list[str]]:
    """Return configured database source and missing required variables."""
    load_environment()
    if _is_set("DATABASE_URL"):
        return "DATABASE_URL", []

    missing = [name for name in DATABASE_ENV_VARS if not _is_set(name)]
    if missing:
        return "not configured", missing

    return "POSTGRES_*", []


def main() -> None:
    """Print environment readiness without exposing secret values."""
    env_loaded = load_environment()
    db_source, missing = database_environment_status()

    print(f".env loaded: {'yes' if env_loaded else 'no'} ({DEFAULT_ENV_PATH})")
    print(f"database: {db_source}")
    if missing:
        print(f"missing database variables: {', '.join(missing)}")

    configured_optional = [name for name in OPTIONAL_ENV_VARS if _is_set(name)]
    if configured_optional:
        print(f"optional variables set: {', '.join(configured_optional)}")


if __name__ == "__main__":
    main()
