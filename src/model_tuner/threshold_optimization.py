import numpy as np  # For creating ranges for beta and thresholds
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
)  # For evaluation metrics


def find_optimal_threshold_beta_aware(
    model,
    X_valid,
    y_valid,
    target_precision=None,
    beta_values=None,
    threshold_range=None,
):
    """
    Finds the optimal beta and decision threshold based on precision, recall,
    and F-beta. Optionally aligns with a specific target precision.

    Parameters:
    -----------
    model : trained model object
        The trained model (pickled or otherwise) that supports `predict_proba`.
    X_valid : array-like
        Test dataset features.
    y_valid : array-like
        True labels for the test set.
    target_precision : float, optional
        The desired precision value to achieve. If None, optimizes solely for
        F-beta score.
    beta_values : array-like, optional
        Range of beta values to tune the F-beta score.
        Default is np.linspace(0.5, 2, 20).
    threshold_range : array-like, optional
        Range of thresholds to test. Default is np.linspace(0.01, 0.99, 100).

    Returns:
    --------
    optimal_beta : float
        The best beta value based on balancing precision and recall.
    optimal_threshold : float
        The threshold that maximizes F-beta or aligns with target precision.
    best_precision : float
        The precision score corresponding to the optimal threshold.
    best_recall : float
        The recall score at the optimal threshold.
    best_fbeta : float
        The F-beta score at the optimal threshold.
    """

    ## Set default ranges if not provided
    if beta_values is None:
        beta_values = np.linspace(0.5, 2, 20)  ## Default range of beta values
    if threshold_range is None:
        ## Default range of thresholds
        threshold_range = np.linspace(0.01, 0.99, 100)

    ## Get predicted probabilities for the positive class
    y_probs = model.predict_proba(X_valid)[:, 1]

    best_beta = None
    best_threshold = None
    best_precision = 0
    best_recall = 0
    best_fbeta = 0
    smallest_precision_diff = float("inf")  ## Initialize to a large value

    for beta in beta_values:
        for threshold in threshold_range:
            ## Convert probabilities to binary predictions using the current threshold
            y_pred = (y_probs >= threshold).astype(int)

            ## Compute metrics
            precision = precision_score(y_valid, y_pred, zero_division=0)
            recall = recall_score(y_valid, y_pred, zero_division=0)
            fbeta = fbeta_score(y_valid, y_pred, beta=beta, zero_division=0)

            ## If target_precision is provided, prioritize precision
            if target_precision is not None:
                precision_diff = abs(precision - target_precision)

                ## Check if this is the closest precision so far
                if precision_diff < smallest_precision_diff or (
                    precision_diff == smallest_precision_diff and fbeta > best_fbeta
                ):
                    smallest_precision_diff = precision_diff
                    best_beta = beta
                    best_threshold = threshold
                    best_precision = precision
                    best_recall = recall
                    best_fbeta = fbeta
            else:
                ## If no target_precision, simply maximize F-beta
                if fbeta > best_fbeta:
                    best_beta = beta
                    best_threshold = threshold
                    best_precision = precision
                    best_recall = recall
                    best_fbeta = fbeta

    return best_beta, best_threshold, best_precision, best_recall, best_fbeta
