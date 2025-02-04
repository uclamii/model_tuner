import numpy as np  # For creating ranges for beta and thresholds
from sklearn.metrics import (
    precision_score,
    recall_score,
    confusion_matrix,
    fbeta_score,
)  # For evaluation metrics
from tqdm import tqdm


def threshold_tune(
    y,
    y_proba,
    betas,
    thresholds_range=np.arange(0, 1, 0.01),
):

    fbeta_scores = np.zeros((len(betas), len(thresholds_range)))

    for i in tqdm(range(fbeta_scores.shape[0])):
        for j in range(fbeta_scores.shape[1]):
            y_preds = (y_proba > thresholds_range[j]) * 1
            # conf matrix
            conf_matrix = confusion_matrix(y_true=y.astype(int), y_pred=y_preds)
            tn, fp, fn, tp = conf_matrix.ravel()

            # avoid extreme cases
            if tn <= fp:
                fbeta_scores[i, j] = -1  # not desirable results
            else:
                fbeta_scores[i, j] = fbeta_score(y, y_preds, beta=betas[i])

    ind_beta_with_max_fscore = np.argmax(np.max(fbeta_scores, axis=1))

    return thresholds_range[np.argmax(fbeta_scores[ind_beta_with_max_fscore])]


def find_optimal_threshold_beta(
    y,
    y_proba,
    target_metric=None,
    target_score=None,
    beta_value_range=np.linspace(0.01, 4, 400),
    delta=0.03,
):

    for beta in tqdm(beta_value_range):

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
