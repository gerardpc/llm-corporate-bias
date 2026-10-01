"""This module contains the project paths."""

from pathlib import Path

ROOT_DIR: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = ROOT_DIR / "data"

EXPERIMENTS_DIR: Path = ROOT_DIR / "experiments"

PLOTS_DIR: Path = DATA_DIR / "figures"
