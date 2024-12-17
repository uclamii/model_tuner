import pandas as pd
import numpy as np
import os
import sys
import model_tuner

from sklearn.datasets import make_classification

from sklearn.datasets import load_breast_cancer
from imblearn.over_sampling import SMOTE
from model_tuner.model_tuner_utils import Model
from model_tuner.bootstrapper import evaluate_bootstrap_metrics
from model_tuner.pickleObjects import dumpObjects, loadObjects
from sklearn.linear_model import LogisticRegression


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

lr = LogisticRegression(class_weight="balanced", max_iter=1000)

estimator_name = "lg"
# Set the parameters by cross-validation
tuned_parameters = [
    {
        estimator_name + "__C": np.logspace(-4, 0, 3),
    }
]
kfold = False
calibrate = False


model = Model(
    name="Logistic Regression",
    estimator_name=estimator_name,
    model_type="classification",
    calibrate=calibrate,
    estimator=lr,
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
    imbalance_sampler=SMOTE(),
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
