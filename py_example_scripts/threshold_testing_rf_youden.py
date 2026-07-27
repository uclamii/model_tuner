"""
Threshold selection comparison on the ACTG 175 AIDS clinical trials cohort.

Compares three ways of choosing an operating point for a calibrated
RandomForest:

    1. find_optimal_threshold_beta(target_metric="precision", ...)
    2. find_optimal_threshold_beta(target_metric="recall", ...)
    3. find_optimal_threshold_youden(...)          <-- added

The first two require the caller to guess a `target_score` up front and then
re-run to discover what operating point that produced. Youden's J derives the
cut-point from the ROC curve directly: no target score, no beta.

Reference:
    Youden WJ. Index for rating diagnostic tests. Cancer. 1950;3(1):32-35.
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
import matplotlib.pyplot as plt

import model_tuner  ## import model_tuner to show version info.
from model_tuner import Model  ## Model class from model_tuner lib.
from model_tuner.threshold_optimization import (
    find_optimal_threshold_beta,
    find_optimal_threshold_youden,
)

from ucimlrepo import fetch_ucirepo

print(f"model_tuner version: {model_tuner.__version__}")

################################################################################
## Data
################################################################################

aids_clinical_trials_group_study_175 = fetch_ucirepo(id=890)

X = aids_clinical_trials_group_study_175.data.features
y = aids_clinical_trials_group_study_175.data.targets

if isinstance(y, pd.DataFrame):
    y = y.squeeze()

## Drop zero-variance columns
zero_variance_columns = X.columns[X.var() == 0]
if not zero_variance_columns.empty:
    X = X.drop(columns=zero_variance_columns)

print(f"n = {len(y)}, events = {int(y.sum())}, prevalence = {y.mean():.4f}")

################################################################################
## Model
################################################################################

rf = RandomForestClassifier(class_weight="balanced", random_state=42)

estimator_name = "rf"
tuned_parameters = [
    {
        estimator_name + "__n_estimators": [100],
        estimator_name + "__max_depth": [None, 10],
    }
]

kfold = False
calibrate = True

model = Model(
    name="Random Forest",
    estimator_name=estimator_name,
    model_type="classification",
    calibrate=calibrate,
    estimator=rf,
    kfold=kfold,
    stratify_y=True,
    stratify_cols=["gender"],
    grid=tuned_parameters,
    randomized_grid=True,
    n_iter=40,
    scoring=["roc_auc"],
    n_splits=10,
    n_jobs=-2,
    random_state=42,
)

model.grid_search_param_tuning(X, y, f1_beta_tune=True)

X_train, y_train = model.get_train_data(X, y)
X_test, y_test = model.get_test_data(X, y)
X_valid, y_valid = model.get_valid_data(X, y)

model.fit(X_train, y_train, validation_data=[X_valid, y_valid])

## Calibration curve before calibration, for reference
y_prob_uncalibrated = model.predict_proba(X_test)[:, 1]
prob_true_uncalibrated, prob_pred_uncalibrated = calibration_curve(
    y_test,
    y_prob_uncalibrated,
    n_bins=10,
)

if model.calibrate:
    model.calibrateModel(X, y, score="roc_auc")

################################################################################
## Threshold selection
################################################################################

## Thresholds are chosen on the VALIDATION set and reported on the TEST set.
y_prob_valid = model.predict_proba(X_valid)[:, 1]
y_prob_test = model.predict_proba(X_test)[:, 1]

prevalence = float(y_valid.mean())
results = {}


def evaluate(label, threshold):
    """Confusion matrix and rates on the test set at a given cut-point."""
    y_hat = (y_prob_test > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_hat, labels=[0, 1]).ravel()
    results[label] = {
        "threshold": round(float(threshold), 3),
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "precision": tp / (tp + fp) if (tp + fp) else np.nan,
        "youden_J": (
            (tp / (tp + fn) + tn / (tn + fp) - 1) if (tp + fn) and (tn + fp) else np.nan
        ),
    }
    return results[label]


## ---------------------------------------------------------------------------
## 1. Target precision = 0.5
## ---------------------------------------------------------------------------

threshold_precision, beta_precision = find_optimal_threshold_beta(
    y_valid,
    y_prob_valid,
    target_metric="precision",
    target_score=0.5,
    beta_value_range=np.linspace(0.01, 4, 40),
    threshold_value_range=[0.2, 1],
    delta=0.05,
)

model.threshold["roc_auc"] = threshold_precision
metrics_precision = model.return_metrics(
    X_valid, y_valid, optimal_threshold=True, model_metrics=True
)
evaluate("precision target 0.50", threshold_precision)

## ---------------------------------------------------------------------------
## 2. Target recall = 0.8
## ---------------------------------------------------------------------------

threshold_recall, beta_recall = find_optimal_threshold_beta(
    y_valid,
    y_prob_valid,
    target_metric="recall",
    target_score=0.8,
    beta_value_range=np.linspace(0.01, 4, 40),
    threshold_value_range=[0.3, 0.7],
    delta=0.08,
)

model.threshold["roc_auc"] = threshold_recall
metrics_recall = model.return_metrics(
    X_valid, y_valid, optimal_threshold=True, model_metrics=True
)
evaluate("recall target 0.80", threshold_recall)

## ---------------------------------------------------------------------------
## 3. Youden's J
## ---------------------------------------------------------------------------
## No target score and no beta. The cut-point maximizing sensitivity +
## specificity - 1 is read straight off the ROC curve, so nothing has to be
## guessed in advance and re-run.

threshold_youden, beta_youden = find_optimal_threshold_youden(
    y_valid,
    y_prob_valid,
    threshold_value_range=np.arange(0, 1, 0.01),
)

model.threshold["roc_auc"] = threshold_youden
metrics_youden = model.return_metrics(
    X_valid, y_valid, optimal_threshold=True, model_metrics=True
)
evaluate("Youden's J", threshold_youden)

################################################################################
## Comparison
################################################################################

comparison = pd.DataFrame(results).T
comparison.index.name = "criterion"

print("\n" + "=" * 78)
print("Threshold selected on validation, evaluated on test")
print("=" * 78)
print(f"validation prevalence: {prevalence:.4f}")
print(f"test AUC             : {roc_auc_score(y_test, y_prob_test):.4f}")
print(f"test avg precision   : {average_precision_score(y_test, y_prob_test):.4f}")
print()
print(comparison.to_string())
print()
print(f"beta (precision target): {beta_precision}")
print(f"beta (recall target)   : {beta_recall}")
print(f"beta (Youden)          : {beta_youden}   <- none required")

################################################################################
## Threshold sweep
################################################################################
## Plotting the metrics against threshold shows where each criterion landed,
## and whether it sits in a stable region of the curve or in the ragged tail
## where one or two patients move the numbers around.

grid = np.arange(0.01, 1.0, 0.01)
sens, spec, prec = [], [], []
for t in grid:
    y_hat = (y_prob_test > t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_hat, labels=[0, 1]).ravel()
    sens.append(tp / (tp + fn) if (tp + fn) else np.nan)
    spec.append(tn / (tn + fp) if (tn + fp) else np.nan)
    prec.append(tp / (tp + fp) if (tp + fp) else np.nan)

plt.figure(figsize=(10, 6))
plt.plot(grid, sens, label="Sensitivity")
plt.plot(grid, spec, label="Specificity")
plt.plot(grid, prec, label="Precision")

for label, style in [
    ("precision target 0.50", ":"),
    ("recall target 0.80", "-."),
    ("Youden's J", "--"),
]:
    plt.axvline(
        results[label]["threshold"],
        linestyle=style,
        color="black",
        alpha=0.7,
        label=f"{label} ({results[label]['threshold']:.2f})",
    )

plt.axvline(prevalence, color="grey", alpha=0.4, label=f"prevalence ({prevalence:.2f})")
plt.xlabel("Threshold")
plt.ylabel("Metric")
plt.title("Operating point selected by each criterion")
plt.legend(loc="best", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
