"""This module contains utility analysis functions."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


def thurstone_model_v(win_matrix):
    """
    Estimates the latent utilities (mu values) using Thurstone's Case V model.

    Args:
        win_matrix (np.ndarray or pd.DataFrame):
            NxN win matrix. win_matrix[i][j] = n if i beat j n times, 0 otherwise.

    Returns:
        pd.Series: estimated utility values (mu), indexed by labels if provided
    """
    if isinstance(win_matrix, pd.DataFrame):
        labels = win_matrix.index.tolist()
        win_matrix = win_matrix.to_numpy()
    else:
        labels = [f"Item {i}" for i in range(len(win_matrix))]

    n = win_matrix.shape[0]

    def neg_log_likelihood(mu):
        ll = 0
        # Iterate over the upper triangle of the matrix to avoid double counting
        for i in range(n):
            for j in range(i + 1, n):
                wins_ij = win_matrix[i, j]
                wins_ji = win_matrix[j, i]
                if wins_ij + wins_ji == 0:
                    continue

                delta = mu[i] - mu[j]
                prob_ij = norm.cdf(delta / np.sqrt(2))

                # Add clipping for numerical stability to avoid log(0)
                prob_ij = np.clip(prob_ij, 1e-15, 1 - 1e-15)
                prob_ji = 1 - prob_ij

                if wins_ij > 0:
                    ll += wins_ij * np.log(prob_ij)
                if wins_ji > 0:
                    ll += wins_ji * np.log(prob_ji)
        return -ll

    def constraint(mu):
        return np.sum(mu)

    mu0 = np.zeros(n)
    result = minimize(
        neg_log_likelihood,
        mu0,
        constraints={"type": "eq", "fun": constraint},
    )

    if result.success:
        return pd.Series(result.x, index=labels)
    else:
        raise RuntimeError(f"Optimization failed: {result.message}")


def compute_win_matrix(df, entities: list[str]):
    """
    Compute the win matrix for a given dataframe and companies.

    Args:
        df: The dataframe with the results of the experiment.
        entities: The entities to compare.

    Returns:
        A dataframe with the win matrix.
    """
    win_matrix = pd.DataFrame(0, index=entities, columns=entities, dtype=int)

    for _, row in df.iterrows():
        a, b, choice = row["option_a"], row["option_b"], row["answer"]
        if choice == "A":
            win_matrix.loc[a, b] += 1
        elif choice == "B":
            win_matrix.loc[b, a] += 1

    return win_matrix
