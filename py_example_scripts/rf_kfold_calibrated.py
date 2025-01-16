import pandas as pd
import numpy as np
import os
import sys
import model_tuner
import warnings

# from sklearn.exceptions import ConvergenceWarning

# # Suppress ConvergenceWarnings
# warnings.filterwarnings("ignore", category=ConvergenceWarning)


from sklearn.datasets import make_classification

from sklearn.datasets import load_breast_cancer
from imblearn.over_sampling import SMOTE

from model_tuner.model_tuner_utils import Model, report_model_metrics
from model_tuner.bootstrapper import evaluate_bootstrap_metrics
from model_tuner.pickleObjects import dumpObjects, loadObjects
from sklearn.ensemble import RandomForestClassifier

from ucimlrepo import fetch_ucirepo

print()
print(f"Model Tuner version: {model_tuner.__version__}")
print(f"Model Tuner authors: {model_tuner.__author__}")
print()

# fetch dataset
aids_clinical_trials_group_study_175 = fetch_ucirepo(id=890)

aids_clinical_trials_group_study_175.data

# data (as pandas dataframes)
X = aids_clinical_trials_group_study_175.data.features

indices = [
    1024,
    1,
    387,
    12,
    1164,
    1548,
    1423,
    272,
    22,
    152,
    30,
    2122,
    33,
    675,
    1829,
    1321,
    810,
    45,
    1847,
    1976,
    55,
    60,
    1731,
    67,
    1354,
    463,
    213,
    345,
    2137,
    606,
    96,
    100,
    234,
    120,
    890,
]

X["gender"].iloc[indices] = np.nan


y = aids_clinical_trials_group_study_175.data.targets

joined_data = pd.merge(X, y, left_index=True, right_index=True)

joined_data = joined_data.dropna()

y = joined_data["cid"]

X = joined_data.drop(columns=["cid"])

rf = RandomForestClassifier(class_weight="balanced", random_state=42)

estimator_name = "rf"  # Change estimator name to reflect Random Forest
# Set the parameters by cross-validation (example: tuning n_estimators and max_depth)
tuned_parameters = [
    {
        estimator_name + "__n_estimators": [100],  # Example: number of trees
        estimator_name + "__max_depth": [None, 10],  # Example: max depth of trees
    }
]

kfold = True
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
    imbalance_sampler=SMOTE(random_state=42),
)


model.grid_search_param_tuning(X, y, f1_beta_tune=True)

model.fit(X, y)

if model.calibrate:
    model.calibrateModel(X, y, score="roc_auc")

print("Validation Metrics")
model.return_metrics(
    X,
    y,
    optimal_threshold=True,
    print_threshold=True,
    model_metrics=True,
)
print("Test Metrics")
model.return_metrics(
    X,
    y,
    optimal_threshold=True,
    print_threshold=True,
    model_metrics=True,
)

y_prob = model.predict_proba(X)

### F1 Weighted
y_pred = model.predict(X, optimal_threshold=True)

### Report Model Metrics

model.return_metrics(
    X,
    y,
    optimal_threshold=True,
    print_threshold=True,
    model_metrics=True,
    return_dict=False,
)
