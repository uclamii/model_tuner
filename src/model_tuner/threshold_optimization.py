from typing import Union, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
)
from tqdm import tqdm
import warnings
import pandas as pd


def threshold_tune(
    y: Union[np.ndarray, List[int]],
    y_proba: Union[np.ndarray, List[float]],
    betas: Union[np.ndarray, List[float]],
    thresholds_range: np.ndarray = np.arange(0, 1, 0.01),
) -> float:
    """
    Tune the threshold to maximize the F-beta score.

    Parameters:
        y (array-like): True binary labels.
        y_proba (array-like): Predicted probabilities.
        betas (array-like): List of beta values for F-beta score calculation.
        thresholds_range (array-like): Range of thresholds to evaluate.

    Returns:
        float: Optimal threshold that maximizes the F-beta score.
    """

    fbeta_scores = np.zeros((len(betas), len(thresholds_range)))

    for i in range(fbeta_scores.shape[0]):
        for j in range(fbeta_scores.shape[1]):
            y_preds = (y_proba > thresholds_range[j]) * 1
            fbeta_scores[i, j] = fbeta_score(y, y_preds, beta=betas[i], zero_division=1)

    ind_beta_with_max_fscore = np.argmax(np.max(fbeta_scores, axis=1))

    return thresholds_range[np.argmax(fbeta_scores[ind_beta_with_max_fscore])]


def find_optimal_threshold_beta(
    y: Union[np.ndarray, List[int]],
    y_proba: Union[np.ndarray, List[float]],
    target_metric: Optional[str] = None,
    target_score: Optional[float] = None,
    beta_value_range: np.ndarray = np.linspace(0.01, 4, 400),
    threshold_value_range: np.ndarray = np.arange(0, 1, 0.01),
    delta: float = 0.0,
) -> Optional[Tuple[float, float]]:
    """
    Find the optimal threshold and beta for a given target metric and score.

    Parameters:
        y (array-like): True binary labels.
        y_proba (array-like): Predicted probabilities.
        target_metric (str): Metric to optimize ("precision" or "recall").
        target_score (float): Desired target metric score.
        beta_value_range (array-like): Range of beta values to evaluate.
        thresholds_value_range (array-like): Range of thresholds to evaluate.
        delta (float): Initial tolerance for matching the target score.


    Returns:
        tuple: Optimal threshold and beta if found; otherwise, None.

    Raises:
        Exception: If delta exceeds 0.2 and no threshold is found.
        ValueError: If y or y_proba are empty
        ValueError: If precision or recall are not specifid as target metrics
    """
    threshold = None

    if target_metric not in ["precision", "recall"]:
        raise ValueError("Please specify either precision or recall")

    ## Exceptions to test if y or y_proba are empty based on whether they are dataframes
    ## or numpy arrays
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        if y.empty:
            raise ValueError("y cannot be empty.")
    elif not y.size:
        raise ValueError("y cannot be empty.")

    if isinstance(y_proba, pd.DataFrame) or isinstance(y_proba, pd.Series):
        if y_proba.empty:
            raise ValueError("y_proba cannot be empty.")
    elif not y_proba.size:
        raise ValueError("y_proba cannot be empty.")

    while threshold is None:
        # Increase delta if no threshold is found
        delta += 0.01

        for beta in tqdm(beta_value_range, desc="Beta Tuning"):

            threshold = threshold_tune(
                y, y_proba, betas=[beta], thresholds_range=threshold_value_range
            )

            ## Convert probabilities to binary predictions using the current threshold
            y_pred = (y_proba > threshold).astype(int)

            if target_metric == "precision":
                metric = precision_score(y, y_pred, zero_division=0)
            else:
                metric = recall_score(y, y_pred, zero_division=0)

            if abs(target_score - metric) < delta:
                print(f"Found optimal threshold for {target_metric}: {target_score}")
                print(f"Threshold: {threshold}")
                return threshold, beta

            if delta > 0.1:
                warnings.warn(
                    f"Delta has exceeded 0.1. Continuing to increase delta..."
                )

            if delta > 0.2:
                raise Exception(
                    "Delta exceeded 0.2. Unable to find an optimal threshold."
                )

            threshold = None


def find_optimal_threshold_youden(
    y: Union[np.ndarray, List[int]],
    y_proba: Union[np.ndarray, List[float]],
    threshold_value_range: np.ndarray = np.arange(0, 1, 0.01),
) -> Tuple[float, None]:
    """
    Find the threshold maximizing Youden's J statistic (sensitivity +
    specificity - 1).

    Unlike F-beta tuning, this requires no target score and no beta value.
    Returns beta as None for signature compatibility with
    find_optimal_threshold_beta.

    Parameters:
        y (array-like): True binary labels.
        y_proba (array-like): Predicted probabilities.
        threshold_value_range (array-like): Range of thresholds to evaluate.

    Returns:
        tuple: (optimal threshold, None)

    Raises:
        ValueError: If y or y_proba are empty.
    """
    y = np.asarray(y).ravel()
    y_proba = np.asarray(y_proba).ravel()

    if not y.size:
        raise ValueError("y cannot be empty.")
    if not y_proba.size:
        raise ValueError("y_proba cannot be empty.")

    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    if n_pos == 0 or n_neg == 0:
        raise ValueError("y must contain both classes to compute Youden's J.")

    # Vectorized: (n_thresholds, n_samples)
    preds = y_proba[None, :] > threshold_value_range[:, None]

    tp = (preds & (y == 1)[None, :]).sum(axis=1)
    fp = (preds & (y == 0)[None, :]).sum(axis=1)

    sensitivity = tp / n_pos
    specificity = 1 - (fp / n_neg)
    j = sensitivity + specificity - 1

    best = int(np.argmax(j))
    threshold = float(threshold_value_range[best])

    print(f"Found optimal threshold via Youden's J: {threshold}")
    print(f"Sensitivity: {sensitivity[best]:.3f}, Specificity: {specificity[best]:.3f}")

    return threshold, None
