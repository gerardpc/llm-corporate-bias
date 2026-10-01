"""
Generate randomized immediate-vs-delayed monetary choice pairs for experiments.

This utility builds binary-choice pairs for temporal discounting tasks, with no
dependencies on external libraries or infrastructure. It is fully pure and easy
to unit-test or reuse in notebooks and alternative experiment drivers.

The caller provides the templates for the immediate and delayed options, keeping
the generator independent of prompt file formats.
"""

# Standard Library
import random
from collections.abc import Iterable

__all__ = ["generate_discounting_pairs"]


def generate_discounting_pairs(
    immediate_template: str,
    delayed_template: str,
    *,
    amounts: Iterable[float],
    rates: Iterable[float],
    durations: Iterable[int],
) -> list[tuple[str, str, float, float, int]]:
    """
    Create a shuffled list of binary-choice strings for discounting tasks.

    Returns:
    -------
    list[tuple[str, str, float, float, int]]
        (option_A_text, option_B_text, now_value, later_value, delay_years)
    """
    option_pairs: list[tuple[str, str, float, float, int]] = []

    for amount in amounts:
        for rate in rates:
            for duration in durations:
                delayed_value = round(amount * (1 + rate) ** duration, 2)

                option_immediate = immediate_template.format(value=amount)
                option_delayed = delayed_template.format(
                    value=delayed_value,
                    duration=duration,
                )

                # Randomly decide which option becomes A vs B while still
                # tracking the numeric values
                if random.random() < 0.5:
                    option_pairs.append(
                        (
                            option_immediate,
                            option_delayed,
                            amount,
                            delayed_value,
                            duration,
                        ),
                    )
                else:
                    option_pairs.append(
                        (
                            option_delayed,
                            option_immediate,
                            amount,
                            delayed_value,
                            duration,
                        ),
                    )

    return option_pairs


if __name__ == "__main__":
    # Example usage
    immediate_template = "Receive ${value} today"
    delayed_template = "Receive ${value} in {duration} years"

    # Note: rates should be in fraction form (e.g., 0.05 for 5%)
    example_amounts = [100, 200]
    example_rates = [0.01, 0.05]  # 1% and 5%
    example_durations = [1, 5]

    pairs = generate_discounting_pairs(
        immediate_template,
        delayed_template,
        amounts=example_amounts,
        rates=example_rates,
        durations=example_durations,
    )

    print("Generated Discounting Pairs:")
    for i, (option_a, option_b, now_v, later_v, d) in enumerate(pairs):
        print(
            f"Pair {i + 1}:\n  Option A: {option_a}\n  Option B: {option_b}"
            f"\n  Now ($): {now_v}, Later ($): {later_v}, Delay (years): {d}",
        )
