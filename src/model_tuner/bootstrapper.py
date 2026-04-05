from sklearn.utils import resample
from sklearn.metrics import (
    get_scorer,
    recall_score,
    hamming_loss,
    brier_score_loss,
    r2_score,
)
import numpy as np
import scipy.stats as st
import pandas as pd
from tqdm import tqdm
from random import seed, randint


def check_input_type(x):
    """Method to check input type pandas Series or numpy.
    Sort index if pandas for sampling efficiency.
    """
    # if y is a numpy array cast it to a dataframe
    if isinstance(x, np.ndarray):
        x = pd.DataFrame(x)
    elif isinstance(x, pd.Series):
        x = x.reset_index(drop=True)  # have to reset index
    elif isinstance(x, pd.DataFrame):
        x = x.reset_index(drop=True)  # have to reset index
    else:
        raise ValueError("Only numpy or pandas types supported.")
    return x


def sampling_method(
    y, n_samples, stratify=False, balance=False, class_proportions=None
):
    """
    Method to resample a dataframe with options for balanced, stratified,
    or custom-proportioned sampling.

    Parameters:
    - y: Pandas Series to resample.
    - n_samples: Total number of samples to draw.
    - stratify: Pandas series to specify stratify according to series proportions.
    - balance: Boolean indicating if classes should be balanced.
    - class_proportions: Dict specifying the proportion to sample from each class.

    Returns resampled y.
    """
    if class_proportions:
        # Ensure that the proportions sum to 1
        if not sum(class_proportions.values()) == 1:
            raise ValueError("class_proportions values must sum to 1.")

        y_resample = pd.DataFrame()

        # Sample from each class according to the specified proportions
        for class_label, proportion in class_proportions.items():
            class_samples = y[y.values == class_label]
            n_class_samples = int(n_samples * proportion)
            resampled_class_samples = resample(
                class_samples,
                replace=True,
                n_samples=n_class_samples,
                random_state=randint(0, 1000000),
            )
            y_resample = pd.concat([y_resample, resampled_class_samples])

    elif balance:
        # Perform balanced resampling by downsampling the majority classes
        class_counts = y.value_counts()
        num_classes = len(class_counts)
        y_resample = pd.DataFrame()

        for class_label in class_counts.index:
            class_samples = y[y.values == class_label]
            resampled_class_samples = resample(
                class_samples,
                replace=True,
                n_samples=int(n_samples / num_classes),
                random_state=randint(0, 1000000),
            )
            y_resample = pd.concat([y_resample, resampled_class_samples])

    else:
        # Resample the target variable with optional stratification to the original dataset
        y_resample = resample(
            y,
            replace=True,
            n_samples=n_samples,
            stratify=stratify,
            random_state=randint(0, 1000000),
        )

    return y_resample.sort_index()


def evaluate_bootstrap_metrics(
    model=None,
    X=None,
    y=None,
    y_pred_prob=None,
    n_samples=500,
    num_resamples=1000,
    metrics=["roc_auc", "f1_weighted", "average_precision"],
    random_state=42,
    threshold=0.5,
    average="macro",
    thresholds=None,
    model_type="classification",
    stratify=None,
    balance=False,
    class_proportions=None,
    n_features=None,
    ci_method="t",
    confidence_level=0.95,
):
    """
    Evaluate model performance metrics using bootstrap resampling.

    Confidence intervals are computed using the specified ci_method:
    'percentile' (default) or 't' (t-distribution with SEM, legacy behavior).

    Parameters
    ----------
    ... (existing parameters unchanged)
    ci_method : str, default 't'
        Method for computing 95% confidence intervals. Options are
        'percentile' or 't'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Metric, Mean, 95% CI Lower, 95% CI Upper.
    """
    if y is None:
        raise ValueError("The y parameter is required and cannot be None.")

    if ci_method not in ("percentile", "t"):
        raise ValueError("ci_method must be 'percentile' or 't'.")

    regression_metrics = [
        "explained_variance",
        "max_error",
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "neg_root_mean_squared_error",
        "neg_mean_squared_log_error",
        "neg_median_absolute_error",
        "r2",
        "adjusted_r2",
        "neg_mean_poisson_deviance",
        "neg_mean_gamma_deviance",
    ]

    if "adjusted_r2" in metrics:
        if model_type != "regression":
            raise ValueError(
                "'adjusted_r2' is a regression metric; set model_type='regression'."
            )
        if n_features is None:
            if X is not None:
                n_features = np.array(X).shape[1]
            else:
                raise ValueError(
                    "n_features must be provided when 'adjusted_r2' is included "
                    "in metrics and X is not supplied."
                )
        if n_features < 1:
            raise ValueError("n_features must be a positive integer.")

    y = check_input_type(y)
    if y_pred_prob is not None:
        y_pred_prob = check_input_type(y_pred_prob)

    seed(random_state)

    if y_pred_prob is None and (model is None or X is None):
        raise ValueError("Either model and X or y_pred_prob must be provided.")

    if model_type != "regression" and any(
        metric in metrics for metric in regression_metrics
    ):
        raise ValueError(
            "If using regression metrics please specify model_type='regression'"
        )

    if model_type == "regression" and balance == True:
        raise ValueError(
            "Error: Balancing classes is not applicable for 'regression' tasks."
        )

    scores = {metric: [] for metric in metrics}

    for _ in tqdm(range(num_resamples)):

        y_resample = sampling_method(
            y=y,
            n_samples=n_samples,
            stratify=stratify,
            balance=balance,
            class_proportions=class_proportions,
        )

        if y_pred_prob is not None:
            resampled_indicies = y_resample.index
            y_pred_prob_resample = y_pred_prob.iloc[resampled_indicies]

            if model_type != "regression":
                if thresholds is not None:
                    y_pred_resample = np.zeros_like(y_pred_prob_resample)
                    if isinstance(thresholds, dict):
                        for idx, col in enumerate(y_pred_prob_resample.columns):
                            thr = thresholds.get(col, 0.5)
                            y_pred_resample[:, idx] = (
                                y_pred_prob_resample.iloc[:, idx] > thr
                            ).astype(int)
                    else:
                        for idx, thr in enumerate(thresholds):
                            y_pred_resample[:, idx] = (
                                y_pred_prob_resample.iloc[:, idx] > thr
                            ).astype(int)
                else:
                    y_pred_resample = (y_pred_prob_resample > threshold).astype(int)
            else:
                y_pred_resample = y_pred_prob_resample
        else:
            X = check_input_type(X)
            resampled_indicies = y_resample.index
            X_resample = X.iloc[resampled_indicies]

            if model_type != "regression":
                y_pred_prob_resample = model.predict_proba(X_resample)[:, 1]
                y_pred_resample = model.predict(X_resample, optimal_threshold=True)
            else:
                y_pred_prob_resample = None
                y_pred_resample = model.predict(X_resample)

        for metric in metrics:
            if metric == "adjusted_r2":
                n = len(y_resample)
                p = n_features
                r2 = r2_score(y_resample, y_pred_resample)
                if n - p - 1 <= 0:
                    raise RuntimeError(
                        "Sample Size Error: n_samples must be greater than "
                        "n_features + 1 to compute adjusted_r2."
                    )
                adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
                scores[metric].append(adj_r2)
                continue

            if metric == "neg_brier_score":
                scores[metric].append(
                    brier_score_loss(
                        y_resample,
                        y_pred_prob_resample,
                    )
                )
                continue

            if metric == "specificity":
                scores[metric].append(
                    recall_score(
                        y_resample,
                        y_pred_resample,
                        pos_label=0,
                        average=average if thresholds is not None else "binary",
                    )
                )
                continue

            if metric == "hamming_loss":
                scores[metric].append(hamming_loss(y_resample, y_pred_resample))
                continue

            scorer = get_scorer(metric)
            if metric in ["roc_auc", "average_precision", "brier_score"]:
                try:
                    scores[metric].append(
                        scorer._score_func(y_resample, y_pred_prob_resample)
                    )
                except ValueError as e:
                    if "Only one class present in y_true" in str(e):
                        raise RuntimeError(
                            "Sample Size Error: Increase n_samples, sample size "
                            "too small for metric to be valid."
                        )
                    else:
                        raise
            elif metric == "precision":
                scores[metric].append(
                    scorer._score_func(
                        y_resample,
                        y_pred_resample,
                        zero_division=0,
                        average=average if thresholds is not None else "binary",
                    )
                )
            else:
                try:
                    if thresholds:
                        scores[metric].append(
                            scorer._score_func(
                                y_resample, y_pred_resample, average=average
                            )
                        )
                    else:
                        scores[metric].append(
                            scorer._score_func(y_resample, y_pred_resample)
                        )
                except TypeError:
                    scores[metric].append(
                        scorer._score_func(y_resample, y_pred_resample)
                    )

    metrics_results = {
        "Metric": [],
        "Mean": [],
        "95% CI Lower": [],
        "95% CI Upper": [],
    }

    for metric in metrics:
        metric_scores = scores[metric]
        mean_score = np.mean(metric_scores)

        if ci_method == "percentile":
            ci_lower, ci_upper = np.percentile(
                metric_scores,
                [(1 - confidence_level) / 2 * 100, (1 + confidence_level) / 2 * 100],
            )
        elif ci_method == "t":
            ci_lower, ci_upper = st.t.interval(
                confidence_level,
                len(metric_scores) - 1,
                loc=mean_score,
                scale=st.sem(metric_scores),
            )

        metrics_results["Metric"].append(metric)
        metrics_results["Mean"].append(mean_score)
        metrics_results["95% CI Lower"].append(ci_lower)
        metrics_results["95% CI Upper"].append(ci_upper)

    metrics_df = pd.DataFrame(metrics_results)
    return metrics_df
