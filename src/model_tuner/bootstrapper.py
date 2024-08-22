from sklearn.utils import resample
from sklearn.metrics import get_scorer, recall_score
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
    y,
    n_samples,
    stratify,
    balance,
):
    """Method to resample a dataframe balanced, stratified, or none.

    Returns resampled y.
    """
    if balance:
        # Perform balanced resampling by downsampling the majority classes
        # resampling the same number of samples as the minority class
        class_counts = y.value_counts()
        num_classes = len(class_counts)
        y_resample = pd.DataFrame()

        # append each sample to y_resample
        for class_label in class_counts.index:
            class_samples = y[y == class_label]
            resampled_class_samples = resample(
                class_samples,
                replace=True,
                n_samples=int(
                    n_samples / num_classes
                ),  # same number of samples per class always same fraction
                random_state=randint(0, 1000000),
            )
            y_resample = pd.concat([y_resample, resampled_class_samples])

        y_resample = y_resample.sort_index()  # to set indx to original shuffled state
    else:
        # Resample the target variable
        y_resample = resample(
            y,
            replace=True,
            n_samples=n_samples,
            stratify=stratify,
            random_state=randint(
                0,
                1000000,
            ),
        )
    return y_resample


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
    model_type="classification",
    stratify=None,
    balance=False,
):
    """
    Evaluate various classification metrics on bootstrap samples using a
    pre-trained model or pre-computed predicted probabilities.

    Parameters:
    - model (optional): A pre-trained classifier that has a predict_proba method.
      Not required if y_pred_prob is provided.
    - X (array-like, optional): Input features. Not required if y_pred_prob is provided.
    - y (array-like): Labels.
    - y_pred_prob (array-like, optional): Pre-computed predicted probabilities.
    - n_samples (int): The number of samples in each bootstrap sample.
    - num_resamples (int): The number of resamples to generate.
    - metrics (list): List of metric names to evaluate.
    - random_state (int, optional): Random state used as the seed for each random number
      in the loop
    - threshold (float, optional): Threshold used to turn probability estimates into predictions.

    Returns:
    - DataFrame: Confidence intervals for various metrics.
    """

    regression_metrics = [
        "explained_variance",
        "max_error",
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "neg_root_mean_squared_error",
        "neg_mean_squared_log_error",
        "neg_median_absolute_error",
        "r2",
        "neg_mean_poisson_deviance",
        "neg_mean_gamma_deviance",
    ]

    # if y is a numpy array cast it to a dataframe
    y = check_input_type(y)
    if y_pred_prob is not None:
        y_pred_prob = check_input_type(y_pred_prob)

    # Set the random seed for reproducibility
    seed(random_state)

    # Ensure either model and X or y_pred_prob are provided
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

    # Initialize a dictionary to store scores for each metric
    scores = {metric: [] for metric in metrics}

    # Perform bootstrap resampling
    for _ in tqdm(range(num_resamples)):

        y_resample = sampling_method(
            y=y,
            n_samples=n_samples,
            stratify=stratify,
            balance=balance,
        )

        # If pre-computed predicted probabilities are provided
        if y_pred_prob is not None:
            resampled_indicies = y_resample.index
            y_pred_prob_resample = y_pred_prob.iloc[resampled_indicies]

            if model_type != "regression":
                y_pred_resample = (y_pred_prob_resample >= threshold).astype(int)
            else:
                y_pred_resample = y_pred_prob_resample
        else:
            X = check_input_type(X)
            # Resample the input features and compute predictions
            resampled_indicies = y_resample.index
            X_resample = X.iloc[resampled_indicies]

            # X_resample = X_resample.values  # numpy array
            if model_type != "regression":
                y_pred_prob_resample = model.predict_proba(X_resample)[:, 1]
            else:
                y_pred_prob_resample = None
            y_pred_resample = model.predict(X_resample)

        # Calculate and store metric scores
        for metric in metrics:
            if metric == "specificity":
                # Compute specificity using recall_score with pos_label=0
                scores[metric].append(
                    recall_score(
                        y_resample,
                        y_pred_resample,
                        pos_label=0,
                    )
                )
                continue
            # Get the scorer function for the given metric
            scorer = get_scorer(metric)
            if metric in ["roc_auc", "average_precision", "brier_score"]:
                # Metrics that use probability predictions
                try:
                    scores[metric].append(
                        scorer._score_func(y_resample, y_pred_prob_resample)
                    )
                except ValueError as e:
                    if "Only one class present in y_true" in str(e):
                        raise RuntimeError(
                            "Sample Size Error: Increase n_samples, sample size too small for metric to be valid."
                        )
                    else:
                        raise

            elif metric == "precision":
                # Precision with zero division handling
                scores[metric].append(
                    scorer._score_func(
                        y_resample,
                        y_pred_resample,
                        zero_division=0,
                    )
                )
            else:
                # Other metrics
                scores[metric].append(
                    scorer._score_func(
                        y_resample,
                        y_pred_resample,
                    )
                )
    # Initialize a dictionary to store results
    metrics_results = {
        "Metric": [],
        "Mean": [],
        "95% CI Lower": [],
        "95% CI Upper": [],
    }

    # Calculate mean and confidence intervals for each metric
    for metric in metrics:
        metric_scores = scores[metric]
        mean_score = np.mean(metric_scores)
        ci_lower, ci_upper = st.t.interval(
            0.95,
            len(metric_scores) - 1,
            loc=mean_score,
            scale=st.sem(
                metric_scores,
            ),
        )
        metrics_results["Metric"].append(metric)
        metrics_results["Mean"].append(mean_score)
        metrics_results["95% CI Lower"].append(ci_lower)
        metrics_results["95% CI Upper"].append(ci_upper)

    # Convert results to a DataFrame and return
    metrics_df = pd.DataFrame(metrics_results)
    return metrics_df
