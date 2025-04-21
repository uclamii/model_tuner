import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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

## Check for zero-variance columns and drop accordingly

## Check for zero-variance columns and drop them
zero_variance_columns = X.columns[X.var() == 0]
if not zero_variance_columns.empty:
    X = X.drop(columns=zero_variance_columns)

# Define Hyperparameters for RandomForest
rf = RandomForestClassifier(class_weight="balanced", random_state=42)

estimator_name = "rf"  # Change estimator name to reflect Random Forest
# Set the parameters by cross-validation (example: tuning n_estimators and max_depth)
tuned_parameters = [
    {
        estimator_name + "__n_estimators": [100],  # Example: number of trees
        estimator_name + "__max_depth": [None, 10],  # Example: max depth of trees
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

#### Perform Grid Search Parameter Tuning and Retrieve Split Data

model.grid_search_param_tuning(X, y, f1_beta_tune=True)

X_train, y_train = model.get_train_data(X, y)
X_test, y_test = model.get_test_data(X, y)
X_valid, y_valid = model.get_valid_data(X, y)


model.fit(
    X_train,
    y_train,
    validation_data=[X_valid, y_valid],
)


## Get the predicted probabilities for the validation data from uncalibrated model
y_prob_uncalibrated = model.predict_proba(X_test)[:, 1]

## Compute the calibration curve for the uncalibrated model
prob_true_uncalibrated, prob_pred_uncalibrated = calibration_curve(
    y_test,
    y_prob_uncalibrated,
    n_bins=10,
)

## Calibrate the model
if model.calibrate:
    model.calibrateModel(X, y, score="roc_auc")

threshold, beta = find_optimal_threshold_beta(
    y_valid,
    model.predict_proba(X_valid)[:, 1],
    target_metric="precision",
    target_score=0.5,
    beta_value_range=np.linspace(0.01, 4, 40),
    threshold_value_range=[0.2, 1],
    delta=0.05,
)

model.threshold["roc_auc"] = threshold

metrics_df = model.return_metrics(
    X_valid, y_valid, optimal_threshold=True, model_metrics=True
)


threshold, beta = find_optimal_threshold_beta(
    y_valid,
    model.predict_proba(X_valid)[:, 1],
    target_metric="recall",
    target_score=0.8,
    beta_value_range=np.linspace(0.01, 4, 40),
    threshold_value_range=[0.3, 0.7],
    delta=0.08,
)

model.threshold["roc_auc"] = threshold

metrics_df = model.return_metrics(
    X_valid, y_valid, optimal_threshold=True, model_metrics=True
)
