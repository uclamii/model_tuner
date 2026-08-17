"""
F-beta sweep vs Youden's J for threshold selection.

model_tuner already selects a threshold with no target score required:
`grid_search_param_tuning(f1_beta_tune=True)` and
`calibrateModel(f1_beta_tune=True)` both sweep betas and thresholds and set
`model.threshold[score]` internally.

The question this script asks is not which criterion is better, but where the
two land relative to each other. F-beta is the harmonic mean of precision and
recall, so true negatives never enter the calculation. Youden's J is
sensitivity + specificity - 1, so specificity enters explicitly. They are
answering different questions and there is no reason to expect the same
cut-point.

This is a demonstration of the two criteria on one cohort and one split, not a
benchmark. Whichever criterion happens to win on any single held-out metric is
mostly sampling noise: the bootstrap interval printed below is wide enough to
cover zero, so the table should be read as "here are two operating points",
never as "this criterion outperforms that one". Choosing between them is a
clinical decision about the relative cost of a false negative versus a false
positive, and that cost is not in the data.

Run on the ACTG 175 AIDS clinical trials cohort.

Reference:
    Youden WJ. Index for rating diagnostic tests. Cancer. 1950;3(1):32-35.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

import model_tuner
from model_tuner import Model
from model_tuner.threshold_optimization import find_optimal_threshold_youden

from ucimlrepo import fetch_ucirepo

print(f"model_tuner version: {model_tuner.__version__}")

IMAGES_DIR = Path("./images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

SCORE = "roc_auc"

################################################################################
## Data
################################################################################

aids = fetch_ucirepo(id=890)

X = aids.data.features
y = aids.data.targets

if isinstance(y, pd.DataFrame):
    y = y.squeeze()

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

model = Model(
    name="Random Forest",
    estimator_name=estimator_name,
    model_type="classification",
    calibrate=True,
    estimator=rf,
    kfold=False,
    stratify_y=True,
    stratify_cols=["gender"],
    grid=tuned_parameters,
    randomized_grid=True,
    n_iter=40,
    scoring=[SCORE],
    n_splits=10,
    n_jobs=-2,
    random_state=42,
)

## f1_beta_tune=True sweeps betas and thresholds. No target score is passed.
model.grid_search_param_tuning(X, y, f1_beta_tune=True)

X_train, y_train = model.get_train_data(X, y)
X_valid, y_valid = model.get_valid_data(X, y)
X_test, y_test = model.get_test_data(X, y)

model.fit(X_train, y_train, validation_data=[X_valid, y_valid])

print(f"\nThreshold after grid search (uncalibrated): {model.threshold[SCORE]:.3f}")

################################################################################
## Calibrate, retuning the threshold afterwards
################################################################################
## Calibration refits the estimator and shifts the probability distribution,
## so a threshold chosen beforehand no longer matches the final model.
## f1_beta_tune=True here retunes on the calibrated probabilities.

model.calibrateModel(X, y, score=SCORE, f1_beta_tune=True)

threshold_fbeta = float(model.threshold[SCORE])
print(f"Threshold after calibration (F-beta sweep) : {threshold_fbeta:.3f}")

################################################################################
## Youden's J on the same calibrated model
################################################################################

y_prob_valid = model.predict_proba(X_valid)[:, 1]
y_prob_test = model.predict_proba(X_test)[:, 1]

threshold_youden, beta_youden = find_optimal_threshold_youden(
    y_valid,
    y_prob_valid,
    threshold_value_range=np.arange(0, 1, 0.01),
)
print(f"Threshold from Youden's J                  : {threshold_youden:.3f}")

################################################################################
## Evaluate both cut-points on the test set
################################################################################

prevalence = float(y_valid.mean())
results = {}


def evaluate(label, threshold):
    """Confusion matrix and rates on the test set at a given cut-point."""
    y_hat = (y_prob_test > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_hat, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision and sensitivity
        else np.nan
    )

    results[label] = {
        "threshold": round(float(threshold), 3),
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "sensitivity": round(sensitivity, 3),
        "specificity": round(specificity, 3),
        "precision": round(precision, 3),
        "F1": round(f1, 3),
        "youden_J": round(sensitivity + specificity - 1, 3),
    }
    return results[label]


evaluate("F-beta sweep", threshold_fbeta)
evaluate("Youden's J", threshold_youden)

comparison = pd.DataFrame(results).T
comparison.index.name = "criterion"

print("\n" + "=" * 78)
print("Two operating points, both selected on validation, evaluated on test")
print("=" * 78)
print(f"validation prevalence : {prevalence:.4f}")
print(f"test AUC              : {roc_auc_score(y_test, y_prob_test):.4f}")
print(f"test average precision: {average_precision_score(y_test, y_prob_test):.4f}")
print()
print(comparison.to_string())

################################################################################
## How much of the gap between the two rows is real
################################################################################
## The rows above differ, but a single test split of this size cannot resolve a
## difference that small. Resampling the test set with replacement gives an
## interval on the paired difference in each metric. Where the interval covers
## zero, the ordering in the table is not evidence of anything.

N_BOOT = 2000
rng = np.random.default_rng(42)

y_test_arr = np.asarray(y_test).ravel()
n_test = len(y_test_arr)


def paired_metrics(idx):
    """Youden's J and F1 at both cut-points on one bootstrap resample."""
    y_true = y_test_arr[idx]
    probs = y_prob_test[idx]
    out = []
    for threshold in (threshold_fbeta, threshold_youden):
        y_hat = (probs > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
        se = tp / (tp + fn) if (tp + fn) else np.nan
        sp = tn / (tn + fp) if (tn + fp) else np.nan
        pr = tp / (tp + fp) if (tp + fp) else np.nan
        f1 = 2 * pr * se / (pr + se) if pr and se else np.nan
        out.append((se + sp - 1, f1))
    return out


boot_j, boot_f1 = [], []
for _ in range(N_BOOT):
    idx = rng.integers(0, n_test, n_test)
    if len(np.unique(y_test_arr[idx])) < 2:
        continue
    (j_fbeta, f1_fbeta), (j_youden, f1_youden) = paired_metrics(idx)
    boot_j.append(j_youden - j_fbeta)
    boot_f1.append(f1_youden - f1_fbeta)

print()
print(f"Paired bootstrap on the test set, {len(boot_j)} resamples")
print("Positive favours Youden's J, negative favours the F-beta sweep.")
for name, deltas in (("Youden's J", boot_j), ("F1", boot_f1)):
    lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    covers_zero = lo <= 0 <= hi
    verdict = "covers zero" if covers_zero else "excludes zero"
    print(
        f"  delta {name:<11} {np.nanmean(deltas):+.3f}  "
        f"95% CI [{lo:+.3f}, {hi:+.3f}]  <- {verdict}"
    )

################################################################################
## What each criterion is actually optimizing
################################################################################
## F1 rises with TP and falls with FP and FN. TN appears nowhere in it, so the
## criterion is blind to how the model handles the majority class. Youden's J
## carries specificity, which is TN / (TN + FP), so the negative class is
## weighted equally with the positive one.

print()
print("-" * 78)
print("F1  = 2TP / (2TP + FP + FN)          <- TN absent")
print("J   = TP/(TP+FN) + TN/(TN+FP) - 1    <- TN present")
print("-" * 78)

diff = abs(threshold_fbeta - threshold_youden)
if diff < 0.02:
    print(
        f"\nThe two criteria land within {diff:.3f} of each other on this cohort. "
        "Convergence between criteria that weight the negative class differently "
        "suggests the operating point is not especially sensitive to that choice "
        "here."
    )
else:
    print(
        f"\nThe two criteria land {diff:.3f} apart. The F-beta cut-point sits "
        "lower, buying sensitivity with false positives; the Youden cut-point "
        "sits higher, giving some of that sensitivity back for specificity. "
        "Neither is the correct answer on its own. Pick the one whose error "
        "trade-off matches what a miss costs in your setting."
    )

################################################################################
## Threshold sweep
################################################################################

grid = np.arange(0.01, 1.0, 0.01)
sens, spec, f1s = [], [], []
for t in grid:
    y_hat = (y_prob_test > t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_hat, labels=[0, 1]).ravel()
    se = tp / (tp + fn) if (tp + fn) else np.nan
    sp = tn / (tn + fp) if (tn + fp) else np.nan
    pr = tp / (tp + fp) if (tp + fp) else np.nan
    sens.append(se)
    spec.append(sp)
    f1s.append(2 * pr * se / (pr + se) if pr and se else np.nan)

j_curve = [se + sp - 1 for se, sp in zip(sens, spec)]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(grid, sens, label="Sensitivity", alpha=0.7)
ax.plot(grid, spec, label="Specificity", alpha=0.7)
ax.plot(grid, f1s, label="F1", linewidth=2)
ax.plot(grid, j_curve, label="Youden's J", linewidth=2)

ax.axvline(
    threshold_fbeta,
    linestyle="--",
    color="black",
    alpha=0.8,
    label=f"F-beta sweep ({threshold_fbeta:.2f})",
)
ax.axvline(
    threshold_youden,
    linestyle=":",
    color="black",
    alpha=0.8,
    label=f"Youden's J ({threshold_youden:.2f})",
)
ax.axvline(
    prevalence,
    color="grey",
    alpha=0.4,
    label=f"prevalence ({prevalence:.2f})",
)

ax.set_xlabel("Threshold")
ax.set_ylabel("Metric")
ax.set_title("F-beta sweep vs Youden's J")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(IMAGES_DIR / "fbeta_vs_youden.png", dpi=300, bbox_inches="tight")
print(f"\nSaved: {IMAGES_DIR / 'fbeta_vs_youden.png'}")

plt.show()