"""This module contains general utility functions."""

import itertools
import random
from pathlib import Path

from bias_in_llms.config.project_paths import PLOTS_DIR


def generate_randomized_pairs(elements, n_repeats=10):
    """
    Generate randomized pairs of elements ensuring balanced positions.

    Each pair (a,b) will appear exactly n_repeats/2 times in each order.

    Args:
        elements: The elements to generate pairs from.
        n_repeats: The number of times to repeat the experiment (must be even).

    Returns:
        A list of randomized pairs with balanced positions.
    """
    if n_repeats % 2 != 0:
        raise ValueError("n_repeats must be even to ensure balanced positions")

    pairs = list(itertools.combinations(elements, 2))
    randomized_trials = []

    # For each original pair, add exactly n_repeats/2 in each order
    for a, b in pairs:
        # Add half the repeats as (a,b)
        randomized_trials.extend([(a, b)] * (n_repeats // 2))
        # Add half the repeats as (b,a)
        randomized_trials.extend([(b, a)] * (n_repeats // 2))

    # Shuffle all trials to randomize the order
    random.shuffle(randomized_trials)
    return randomized_trials


def save_plots_path(
    experiment_name: str,
    model_id: str,
    preceding_context_class: str = "",
    question_class: str = "",
) -> Path:
    """
    Save the path for the figures.

    Args:
        experiment_name (str): The name of the experiment.
        model_id (str): The ID of the model.
        preceding_context_class (str): The class of the preceding context.
        question_class (str): The class of the question.

    Returns:
        Path: The path directory to save the figures.
    """
    # Start with base path
    plots_path = PLOTS_DIR / experiment_name

    # Add optional components in order if they exist
    if preceding_context_class:
        plots_path = plots_path / preceding_context_class

    if question_class:
        plots_path = plots_path / question_class

    if model_id:
        plots_path = plots_path / model_id

    plots_path.mkdir(parents=True, exist_ok=True)
    return plots_path
