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
    """

    ### 1. Calculate precision, recall, threshold arrays
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    ## `precision_recall_curve` returns an array of thresholds of length N-1,
    ## but the precision/recall arrays are length N.

    ## 2. find where precision or recalls are above the target. This gives us our
    ## threshold search space. We ignore the last value as this doesn't correspond to
    ## a real threshold
    if target_metric == "precision":
        valid_indices = np.where(precisions[:-1] >= min_target_metric)[0]
    elif target_metric == "recall":
        valid_indices = np.where(recalls[:-1] >= min_target_metric)[0]
    else:
        raise (ValueError("Please specify either precision or recall"))

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
