import numpy as np  # For creating ranges for beta and thresholds
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
)  # For evaluation metrics
from tqdm import tqdm
import warnings


def threshold_tune(
    y,
    y_proba,
    betas,
    thresholds_range=np.arange(0, 1, 0.01),
):
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
            fbeta_scores[i, j] = fbeta_score(y, y_preds, beta=betas[i])

    ind_beta_with_max_fscore = np.argmax(np.max(fbeta_scores, axis=1))

    return thresholds_range[np.argmax(fbeta_scores[ind_beta_with_max_fscore])]


def find_optimal_threshold_beta(
    y,
    y_proba,
    target_metric=None,
    target_score=None,
    beta_value_range=np.linspace(0.01, 4, 400),
    delta=0.0,
):
    """
    Find the optimal threshold and beta for a given target metric and score.

    Parameters:
        y (array-like): True binary labels.
        y_proba (array-like): Predicted probabilities.
        target_metric (str): Metric to optimize ("precision" or "recall").
        target_score (float): Desired target metric score.
        beta_value_range (array-like): Range of beta values to evaluate.
        delta (float): Initial tolerance for matching the target score.

    Returns:
        tuple: Optimal threshold and beta if found; otherwise, None.

    Raises:
        Exception: If delta exceeds 0.2 and no threshold is found.
    """
    threshold = None

    while threshold is None:
        # Increase delta if no threshold is found
        delta += 0.01

        for beta in tqdm(beta_value_range, desc="Beta Tuning"):

            threshold = threshold_tune(y, y_proba, betas=[beta])

            ## Convert probabilities to binary predictions using the current threshold
            y_pred = (y_proba >= threshold).astype(int)

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

    return None
