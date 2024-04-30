import pandas as pd
import numpy as np
import os
import sys

module_path = os.path.abspath(os.path.join("../"))
if module_path not in sys.path:
    sys.path.append(module_path)

from model_tuner import *

from xgboost import XGBRegressor

df = pd.read_csv(
    "/Users/afunnell/Code/ModelTuner/UpdatedBranch/model_latest/model_tuner/data/redfin_2024-04-16-15-59-17.csv"
)
df = df.drop(df.index[0])

xg_boost = XGBRegressor(random_state=3)
estimator_name = "xgb"
# Set the parameters by cross-validation
kfold = False
calibrate = False
# Define the hyperparameters for XGBoost
xgb_learning_rates = [0.1, 0.01, 0.05][:1]  # Learning rate or eta
xgb_n_estimators = [100, 200, 300][
    :1
]  # Number of trees. Equivalent to n_estimators in GB
xgb_max_depths = [3, 5, 7][:1]  # Maximum depth of the trees
xgb_subsamples = [0.8, 1.0][:1]  # Subsample ratio of the training instances
xgb_colsample_bytree = [0.8, 1.0][:1]
xgb_eval_metric = ["logloss"]
xgb_early_stopping_rounds = [10]
# xgb_tree_method = ["gpu_hist"]
# early_stopping_mode = ['min']
# early_stopping_patience = [5]
xgb_verbose = [False]
# Subsample ratio of columns when constructing each tree

# Combining the hyperparameters in a dictionary
xgb_parameters = [
    {
        "xgb__learning_rate": xgb_learning_rates,
        "xgb__n_estimators": xgb_n_estimators,
        "xgb__max_depth": xgb_max_depths,
        "xgb__subsample": xgb_subsamples,
        "xgb__colsample_bytree": xgb_colsample_bytree,
        "xgb__eval_metric": xgb_eval_metric,
        "xgb__early_stopping_rounds": xgb_early_stopping_rounds,
        # 'xgb__early_stopping_patience': early_stopping_patience,
        # "xgb_tree_method": xgb_tree_method,
        "xgb__verbose": xgb_verbose,
    }
]

X = df[["BEDS", "BATHS", "SQUARE FEET", "LOT SIZE"]]
y = df[["PRICE"]]

model4 = Model(
    name="Redfin_model",
    estimator_name=estimator_name,
    model_type="regression",
    calibrate=calibrate,
    estimator=xg_boost,
    kfold=kfold,
    stratify=True,
    grid=xgb_parameters,
    randomized_grid=False,
    impute=True,
    # n_iter=3,
    scoring=["r2"],
    # n_splits=2,
    random_state=3,
    xgboost_early=True,
)

eval_set = [X, y]
model4.grid_search_param_tuning(X, y)

X_train, X_valid, X_test, y_train, y_valid, y_test = model4.train_val_test_split(
    X,
    y,
    stratify=True,
    stratify_by=None,
    train_size=0.6,
    validation_size=0.2,
    test_size=0.2,
    calibrate=model4.calibrate,
    random_state=model4.random_state,
)

model4.fit(X, y, validation_data=(X_valid, y_valid))
