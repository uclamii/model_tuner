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

import numpy as np
import pandas as pd
from typing import Tuple, Optional

from sklearn.metrics import precision_recall_curve, fbeta_score


def find_threshold_for_precision_recall(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_metric: str = "precision",
    min_target_metric: float = 0.99,
    beta: float = 1.0,
) -> Optional[Tuple[float, float, float]]:
    """
    1) Find all threshold points at which precision or recall >= `min_target_metric`.
    2) Among them, choose the threshold that maximizes the fbeta

    Args:
        y_true (np.ndarray): Ground-truth binary labels (0 or 1).
        y_proba (np.ndarray): Predicted probabilities for the positive class.
        min_target_metric (float): The minimum required precision or recall (default 0.99).
        beta (float): If `secondary_metric` is 'fbeta', this beta is used for the F-beta score.

    Returns:
        (best_threshold, best_precision, best_fbeta_socre) if found,
        else None if no threshold can achieve the required min_target_metric.

    Raises:
        ValueError: If inputs are invalid (empty arrays, wrong shapes, invalid values)
    """
    ## Error checking for valid input
    if not isinstance(y_true, np.ndarray) or not isinstance(y_proba, np.ndarray):
        raise ValueError("Both y_true and y_proba must be numpy arrays")

    if len(y_true) == 0 or len(y_proba) == 0:
        raise ValueError("Empty arrays are not allowed")

    if len(y_true) != len(y_proba):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true)}) != y_proba ({len(y_proba)})"
        )

    unique_labels = np.unique(y_true)
    if not np.all(np.isin(unique_labels, [0, 1])):
        raise ValueError("y_true must contain only binary labels (0 or 1)")
    if np.any(np.isnan(y_proba)) or np.any(np.isinf(y_proba)):
        raise ValueError("y_proba contains NaN or infinite values")
    if np.any((y_proba < 0) | (y_proba > 1)):
        raise ValueError("y_proba values must be between 0 and 1")
    if target_metric not in ["precision", "recall"]:
        raise ValueError("Please specify either precision or recall")
    if not 0 <= min_target_metric <= 1:
        raise ValueError("min_target_metric must be between 0 and 1")
    if beta <= 0:
        raise ValueError("beta must be positive")

    ### 1. Calculate precision, recall, threshold arrays
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    ## 2. find where precision or recalls are above the target
    if target_metric == "precision":
        valid_indices = np.where(precisions[:-1] >= min_target_metric)[0]
    else:  # recall
        valid_indices = np.where(recalls[:-1] >= min_target_metric)[0]

    if len(valid_indices) == 0:
        print("Unable to find a threshold that can achieve your target value.")
        return None

    ### 3. Among those thresholds, pick the one that maximizes the fbeta score.
    best_idx = None
    best_val = -1.0
    best_thresh = None

    for i in valid_indices:
        t = thresholds[i]
        y_pred = (y_proba >= t).astype(int)
        secondary_val = fbeta_score(y_true, y_pred, beta=beta, zero_division=1)

        if secondary_val > best_val:
            best_val = secondary_val
            best_thresh = t
            best_idx = i

    if best_idx is not None:
        return (best_thresh, precisions[best_idx], best_val)
    else:
        return None
