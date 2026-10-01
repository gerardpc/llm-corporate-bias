"""
Generate randomized A/B company pairs for the company-bias experiment.

This helper is **pure** (no database or LLM dependencies) which makes it easy
to reuse in notebooks and unit-test in isolation.

The algorithm enumerates all unique unordered pairs of companies and then, for
`n_repeats` rounds, randomly assigns each company to option A or B. The result
is a shuffled list of `(option_A, option_B)` tuples ready to feed into the
LLM prompts.
"""

# Standard Library
import itertools
import random
from collections.abc import Iterable

__all__: list[str] = ["generate_company_pairs"]


def generate_company_pairs(
    companies: Iterable[str],
    n_repeats: int = 10,
) -> list[tuple[str, str]]:
    """
    Create a shuffled list of binary company choices.

    Parameters
    ----------
    companies : Iterable[str]
        Collection of company names.
    n_repeats : int, default ``10``
        How many times to repeat the full set of unique pairs. A value of ``10``
        will therefore produce ``10 * (len(companies) choose 2)`` total pairs.

    Returns:
    -------
    list[tuple[str, str]]
        A list of `(option_A, option_B)` tuples where the order is randomised
        on two levels: (1) whether a given company appears as option A or B and
        (2) the overall pair order through an outer shuffle.
    """
    pairs = list(itertools.combinations(companies, 2))
    randomized_trials: list[tuple[str, str]] = []

    for _ in range(n_repeats):
        for a, b in pairs:
            if random.random() < 0.5:
                randomized_trials.append((a, b))
            else:
                randomized_trials.append((b, a))

    return randomized_trials
