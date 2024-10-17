import pandas as pd
import numpy as np
import os
import sys

from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from model_tuner.model_tuner_utils import Model
from model_tuner.bootstrapper import evaluate_bootstrap_metrics
from model_tuner.pickleObjects import dumpObjects, loadObjects
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from skopt.space import Real, Categorical, Integer
from sklearn.linear_model import LogisticRegression

bc = load_breast_cancer(as_frame=True)["frame"]
bc_cols = [cols for cols in bc.columns if "target" not in cols]
X = bc[bc_cols]
y = bc["target"]

from xgboost import XGBClassifier


estimator = XGBClassifier(
    objective="binary:logistic",
)

estimator_name = "xgb"
xgbearly = False

tuned_parameters = {
    f"{estimator_name}__max_depth": Integer(2, 1000),
    f"{estimator_name}__learning_rate": Real(1e-5, 1e-1, "log-uniform"),
    f"{estimator_name}__n_estimators": Integer(3, 1000),
    f"{estimator_name}__gamma": Real(0, 4, "uniform"),
    f"feature_selection_rfe__n_features_to_select": [5, 10],
}

kfold = False
calibrate = False

rfe_estim = LogisticRegression(C=0.1, max_iter=10)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rfe = RFE(rfe_estim)

model = Model(
    name="XGBoost Early",
    estimator_name=estimator_name,
    calibrate=calibrate,
    estimator=estimator,
    pipeline_steps=[SimpleImputer(), ("rfe", rfe)],
    kfold=True,
    bayesian=True,
    stratify_y=True,
    grid=tuned_parameters,
    randomized_grid=False,
    feature_selection=True,
    n_iter=4,
    scoring=["roc_auc"],
    n_jobs=-2,
    random_state=42,
    imbalance_sampler=SMOTE(),
)


model.grid_search_param_tuning(X, y)


model.fit(X, y)

# print("Validation Metrics")
# model.return_metrics(X, y)
