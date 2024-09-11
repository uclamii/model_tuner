import pandas as pd
import numpy as np
import os
import sys

from sklearn.datasets import make_classification

from sklearn.datasets import load_breast_cancer

from functions import *

from model_tuner.model_tuner_utils import Model
from model_tuner.bootstrapper import evaluate_bootstrap_metrics
from model_tuner.pickleObjects import dumpObjects, loadObjects
from sklearn.linear_model import LogisticRegression


from ucimlrepo import fetch_ucirepo

# fetch dataset
aids_clinical_trials_group_study_175 = fetch_ucirepo(id=890)

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


lr = LogisticRegression(class_weight="balanced", max_iter=1000)

estimator_name = "lg"
# Set the parameters by cross-validation
tuned_parameters = [
    {
        estimator_name + "__C": np.logspace(-4, 0, 3),
        "selectKBest__k": [5, 10, 11, 12, 13, 8, 6, 9, 20],
    }
]
kfold = False
calibrate = False


model = Model(
    name="Logistic Regression",
    estimator_name=estimator_name,
    calibrate=calibrate,
    estimator=lr,
    kfold=kfold,
    impute=True,
    stratify_y=True,
    stratify_cols=["gender"],
    grid=tuned_parameters,
    randomized_grid=True,
    n_iter=40,
    scoring=["roc_auc"],
    n_splits=10,
    selectKBest=True,
    n_jobs=-2,
    random_state=42,
)


model.grid_search_param_tuning(X, y)

X_train, y_train = model.get_train_data(X, y)
X_test, y_test = model.get_test_data(X, y)
X_valid, y_valid = model.get_valid_data(X, y)

model.fit(X_train, y_train)

print("Validation Metrics")
model.return_metrics(X_valid, y_valid)
print("Test Metrics")
model.return_metrics(X_test, y_test)

y_prob = model.predict_proba(X_test)

### F1 Weighted
y_pred = model.predict(X_test, optimal_threshold=True)
