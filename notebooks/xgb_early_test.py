import pandas as pd
import numpy as np
import os
import sys
from imblearn.over_sampling import SMOTE

from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
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
    f"{estimator_name}__verbose": [0],
    f"{estimator_name}__eval_metric": ["logloss"],
}

kfold = False
calibrate = False


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Model(
    name="XGBoost Early",
    model_type="classification",
    estimator_name=estimator_name,
    model_type="classification",
    calibrate=calibrate,
    estimator=estimator,
    pipeline_steps=[],
    kfold=kfold,
    stratify_y=True,
    grid=tuned_parameters,
    randomized_grid=False,
    n_iter=4,
    boost_early=True,
    scoring=["roc_auc"],
    n_jobs=-2,
    random_state=42,
    imbalance_sampler=SMOTE(),
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

y_prob = model.predict_proba(X_test)

### F1 Weighted
y_pred = model.predict(X_test, optimal_threshold=True)
