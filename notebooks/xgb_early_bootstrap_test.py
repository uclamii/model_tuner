import pandas as pd
import numpy as np
import os
import sys

from sklearn.datasets import make_classification

from sklearn.datasets import load_breast_cancer

from model_tuner.model_tuner_utils import Model
from model_tuner.bootstrapper import evaluate_bootstrap_metrics
from model_tuner.pickleObjects import dumpObjects, loadObjects

bc = load_breast_cancer(as_frame=True)["frame"]
bc_cols = [cols for cols in bc.columns if "target" not in cols]
X = bc[bc_cols]
y = bc["target"]

from xgboost import XGBClassifier


estimator = XGBClassifier(
    objective="binary:logistic",
)

estimator_name = "xgb"
xgbearly = True

tuned_parameters = {
    f"{estimator_name}__max_depth": [3, 10, 20, 200, 500],
    f"{estimator_name}__learning_rate": [1e-4],
    f"{estimator_name}__n_estimators": [30],
    f"{estimator_name}__early_stopping_rounds": [10],
    f"{estimator_name}__verbose": [True],
    f"{estimator_name}__eval_metric": ["logloss"],
}

kfold = False
calibrate = False

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Model(
    name="XGBoost Early",
    estimator_name=estimator_name,
    calibrate=calibrate,
    estimator=estimator,
    kfold=kfold,
    stratify_y=True,
    grid=tuned_parameters,
    randomized_grid=False,
    n_iter=4,
    boost_early=True,
    scoring=["roc_auc"],
    n_splits=10,
    n_jobs=-2,
    random_state=42,
)


model.grid_search_param_tuning(X, y)

X_train, y_train = model.get_train_data(X, y)
X_test, y_test = model.get_test_data(X, y)
X_valid, y_valid = model.get_valid_data(X, y)

model.fit(X_train, y_train, validation_data=[X_valid, y_valid])

print("Validation Metrics")
model.return_metrics(X_valid, y_valid)
print("Test Metrics")
model.return_metrics(X_test, y_test)

y_prob = model.predict_proba(X_test)[:, 1]

### F1 Weighted
y_pred = model.predict(X_test, optimal_threshold=True)


### model type bootstrap no stratify, balance, or class proportions
print("Bootstrap metrics \n")
print(
    evaluate_bootstrap_metrics(
        model=model,
        X=X_test,
        y=y_test,
        y_pred_prob=None,
        n_samples=500,
        num_resamples=1000,
        metrics=["roc_auc", "f1_weighted", "average_precision"],
        random_state=42,
        threshold=0.5,
        model_type="classification",
        stratify=None,
        balance=False,
    )
)

print("Bootstrap metrics - Stratified \n")
# stratified
print(
    evaluate_bootstrap_metrics(
        model=None,
        X=X_test,
        y=y_test,
        y_pred_prob=y_prob,
        n_samples=500,
        num_resamples=1000,
        metrics=["roc_auc", "f1_weighted", "average_precision"],
        random_state=42,
        threshold=0.5,
        model_type="classification",
        stratify=y_test,
        balance=False,
    )
)

print("Bootstrap metrics - Balanced \n")
# balanced
print(
    evaluate_bootstrap_metrics(
        model=None,
        X=X_test,
        y=y_test,
        y_pred_prob=y_prob,
        n_samples=500,
        num_resamples=1000,
        metrics=["roc_auc", "f1_weighted", "average_precision"],
        random_state=42,
        threshold=0.5,
        model_type="classification",
        stratify=None,
        balance=True,
    )
)


class_proportions = {
    1: 0.5,
    0: 0.5,
}
print("Bootstrap metrics - Class Proportions \n")
print("Proportions:\n", class_proportions)
# class proportions
print(
    evaluate_bootstrap_metrics(
        model=None,
        X=X_test,
        y=y_test,
        y_pred_prob=y_prob,
        n_samples=500,
        num_resamples=1000,
        metrics=["roc_auc", "f1_weighted", "average_precision"],
        random_state=42,
        threshold=0.5,
        model_type="classification",
        stratify=None,
        balance=False,
        class_proportions=class_proportions,
    )
)

class_proportions = {
    1: 0.3,
    0: 0.7,
}
print("Bootstrap metrics - Class Proportions \n")
print("Proportions:\n", class_proportions)
# class proportions
print(
    evaluate_bootstrap_metrics(
        model=None,
        X=X_test,
        y=y_test,
        y_pred_prob=y_prob,
        n_samples=500,
        num_resamples=1000,
        metrics=["roc_auc", "f1_weighted", "average_precision"],
        random_state=42,
        threshold=0.5,
        model_type="classification",
        stratify=None,
        balance=False,
        class_proportions=class_proportions,
    )
)