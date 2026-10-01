"""This module contains the Bradley-Terry model for utility analysis."""

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def validate_win_matrix(win_matrix):
    """
    Validate that the win matrix is properly formatted.

    Args:
        win_matrix: Input matrix to validate

    Raises:
        ValueError: If matrix is invalid
    """
    if not isinstance(win_matrix, (np.ndarray, pd.DataFrame)):
        raise ValueError("win_matrix must be numpy array or pandas DataFrame")

    if win_matrix.shape[0] != win_matrix.shape[1]:
        raise ValueError("win_matrix must be square")

    if np.any(win_matrix < 0):
        raise ValueError("win_matrix cannot contain negative values")

    # Check if matrix has sufficient data
    # Convert to scalar to handle both numpy arrays and pandas DataFrames
    if isinstance(win_matrix, pd.DataFrame):
        total_comparisons = win_matrix.values.sum()
    else:
        total_comparisons = np.sum(win_matrix)

    if total_comparisons == 0:
        raise ValueError("win_matrix contains no comparisons")


def bradley_terry_model(win_matrix, method="SLSQP", max_iter=1000, tol=1e-8):
    """
    Estimates the latent utilities using the Bradley-Terry model.

    The Bradley-Terry model assumes that the probability of item i beating item j
    follows a logistic distribution based on the difference in their utilities.

    Args:
        win_matrix (np.ndarray or pd.DataFrame):
            NxN win matrix. win_matrix[i][j] = n if i beat j n times, 0 otherwise.
        method (str): Optimization method ('SLSQP', 'trust-constr', 'auto')
        max_iter (int): Maximum number of iterations
        tol (float): Tolerance for convergence

    Returns:
        pd.Series: estimated utility values (log-odds), indexed by labels if provided

    Raises:
        RuntimeError: If optimization fails to converge
        ValueError: If input matrix is invalid
    """
    # Validate input
    validate_win_matrix(win_matrix)

    if isinstance(win_matrix, pd.DataFrame):
        labels = win_matrix.index.tolist()
        win_matrix_np = win_matrix.to_numpy()
    else:
        labels = [f"Item {i}" for i in range(len(win_matrix))]
        win_matrix_np = win_matrix.copy()

    n = win_matrix_np.shape[0]

    # Pre-compute comparison mask for efficiency
    comparison_mask = (win_matrix_np + win_matrix_np.T) > 0

    def neg_log_likelihood(beta):
        ll = 0
        # Iterate over the upper triangle of the matrix to avoid double counting
        for i in range(n):
            for j in range(i + 1, n):
                if comparison_mask[i, j]:
                    wins_ij = win_matrix_np[i, j]
                    wins_ji = win_matrix_np[j, i]
                    delta = beta[i] - beta[j]
                    # Numerical stability: clip delta to prevent overflow
                    delta = np.clip(delta, -500, 500)
                    prob_ij = 1 / (1 + np.exp(-delta))

                    if wins_ij > 0:
                        ll += wins_ij * np.log(max(prob_ij, 1e-15))
                    if wins_ji > 0:
                        ll += wins_ji * np.log(max(1 - prob_ij, 1e-15))
        return -ll

    def constraint(beta):
        return np.sum(beta)

    # Initialize with smart starting point
    beta0 = np.random.normal(0, 0.1, n)

    # Try multiple optimization methods if 'auto' is selected
    methods_to_try = ["SLSQP", "trust-constr"] if method == "auto" else [method]

    for opt_method in methods_to_try:
        try:
            result = minimize(
                neg_log_likelihood,
                beta0,
                method=opt_method,
                constraints={"type": "eq", "fun": constraint},
                options={"maxiter": max_iter, "ftol": tol},
            )

            if result.success:
                return pd.Series(result.x, index=labels)

        except Exception as e:
            warnings.warn(f"Method {opt_method} failed: {e}", stacklevel=2)
            continue

    raise RuntimeError(
        f"All optimization methods failed. Last message: {result.message}",
    )


def compute_win_matrix(
    df,
    entities: list[str],
    choice_col="answer",
    option_a_col="option_a",
    option_b_col="option_b",
):
    """
    Compute the win matrix for a given dataframe and entities.

    Args:
        df: The dataframe with the results of the experiment.
        entities: The entities to compare.
        choice_col (str): Column name containing the choice ('A' or 'B')
        option_a_col (str): Column name containing option A
        option_b_col (str): Column name containing option B

    Returns:
        pd.DataFrame: A dataframe with the win matrix where entry (i,j)
                     represents how many times entity i beat entity j.

    Raises:
        ValueError: If required columns are missing or entities not found
    """
    # Validate input
    required_cols = [choice_col, option_a_col, option_b_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check if all entities are present in the data
    all_options = set(df[option_a_col].unique()) | set(df[option_b_col].unique())
    missing_entities = set(entities) - all_options
    if missing_entities:
        warnings.warn(f"Entities not found in data: {missing_entities}", stacklevel=2)

    win_matrix = pd.DataFrame(0, index=entities, columns=entities, dtype=int)

    for _, row in df.iterrows():
        a, b, choice = row[option_a_col], row[option_b_col], row[choice_col]

        # Skip if entities not in our list
        if a not in entities or b not in entities:
            continue

        if choice == "A":
            win_matrix.loc[a, b] += 1
        elif choice == "B":
            win_matrix.loc[b, a] += 1

    return win_matrix
