from sklearn.metrics import (
    auc,
    roc_curve,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
    precision_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    fbeta_score,
)

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import ElasticNet

import model_tuner  ## import model_tuner to show version info.
from model_tuner import Model  ## Model class from model_tuner lib.
from model_tuner.threshold_optimization import find_optimal_threshold_beta

from ucimlrepo import fetch_ucirepo

## Fetch dataset
aids_clinical_trials_group_study_175 = fetch_ucirepo(id=890)

## Data (as pandas dataframes)
X = aids_clinical_trials_group_study_175.data.features
y = aids_clinical_trials_group_study_175.data.targets

X.head()  ## Inspect the first 5 rows of data

if isinstance(y, pd.DataFrame):
    y = y.squeeze()

"""## Check for zero-variance columns and drop accordingly"""

## Check for zero-variance columns and drop them
zero_variance_columns = X.columns[X.var() == 0]
if not zero_variance_columns.empty:
    X = X.drop(columns=zero_variance_columns)

"""## Define Hyperparameters for XGBoost"""

xgb_name = "xgb"
xgb = XGBClassifier(
    objective="binary:logistic",
    random_state=222,
)
xgbearly = True
tuned_parameters_xgb = {
    f"{xgb_name}__max_depth": [3, 10, 20, 200, 500],
    f"{xgb_name}__learning_rate": [1e-4],
    f"{xgb_name}__n_estimators": [1000],
    f"{xgb_name}__early_stopping_rounds": [100],
    f"{xgb_name}__verbose": [0],
    f"{xgb_name}__eval_metric": ["logloss"],
}

xgb_definition = {
    "clc": xgb,
    "estimator_name": xgb_name,
    "tuned_parameters": tuned_parameters_xgb,
    "randomized_grid": False,
    "n_iter": 5,
    "early": xgbearly,
}

"""## Define The Model Object"""

model_type = "xgb"
clc = xgb_definition["clc"]
estimator_name = xgb_definition["estimator_name"]

tuned_parameters = xgb_definition["tuned_parameters"]
n_iter = xgb_definition["n_iter"]
rand_grid = xgb_definition["randomized_grid"]
early_stop = xgb_definition["early"]
kfold = False
calibrate = True

## Initialize and Configure the Model

model_xgb = Model(
    name=f"AIDS_Clinical_{model_type}",
    estimator_name=estimator_name,
    calibrate=calibrate,
    estimator=clc,
    model_type="classification",
    kfold=kfold,
    stratify_y=True,
    stratify_cols=["gender", "race"],
    grid=tuned_parameters,
    randomized_grid=rand_grid,
    boost_early=early_stop,
    scoring=["roc_auc"],
    random_state=222,
    n_jobs=2,
)

#### Perform Grid Search Parameter Tuning and Retrieve Split Data

model_xgb.grid_search_param_tuning(X, y, f1_beta_tune=True)

X_train, y_train = model_xgb.get_train_data(X, y)
X_test, y_test = model_xgb.get_test_data(X, y)
X_valid, y_valid = model_xgb.get_valid_data(X, y)


model_xgb.fit(
    X_train,
    y_train,
    validation_data=[X_valid, y_valid],
)


## Get the predicted probabilities for the validation data from uncalibrated model
y_prob_uncalibrated = model_xgb.predict_proba(X_test)[:, 1]

## Compute the calibration curve for the uncalibrated model
prob_true_uncalibrated, prob_pred_uncalibrated = calibration_curve(
    y_test,
    y_prob_uncalibrated,
    n_bins=10,
)

## Calibrate the model
if model_xgb.calibrate:
    model_xgb.calibrateModel(X, y, score="roc_auc")

threshold, beta = find_optimal_threshold_beta(
    y_valid,
    model_xgb.predict_proba(X_valid)[:, 1],
    target_metric="precision",
    target_score=0.5,
    beta_value_range=np.linspace(0.01, 4, 40),
)

model_xgb.threshold["roc_auc"] = threshold

metrics_df = model_xgb.return_metrics(
    X_valid, y_valid, optimal_threshold=True, model_metrics=True
)


threshold, beta = find_optimal_threshold_beta(
    y_valid,
    model_xgb.predict_proba(X_valid)[:, 1],
    target_metric="recall",
    target_score=0.8,
    beta_value_range=np.linspace(0.01, 4, 1000),
    delta=0.08,
)

model_xgb.threshold["roc_auc"] = threshold

metrics_df = model_xgb.return_metrics(
    X_valid, y_valid, optimal_threshold=True, model_metrics=True
)
