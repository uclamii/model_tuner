from model_tuner import Model, report_model_metrics

import pandas as pd
import numpy as np
import model_tuner

from sklearn.linear_model import Lasso, Ridge, SGDRegressor
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action="ignore", category=DataConversionWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

print()
print(f"Model Tuner version: {model_tuner.__version__}")
print(f"Model Tuner authors: {model_tuner.__author__}")
print()

# Direct download link to the Excel file
url = (
    "https://github.com/uclamii/model_tuner/raw/main/public_data/"
    "redfin_2024-04-16-15-59-17.xlsx"
)

# Read the Excel file
df = pd.read_excel(url)

df.head()  # inspect first 5 rows of data

df["PROPERTY TYPE"].unique()

df = df.drop(df.index[0])  # remove first row of dataframe which is not used

"""## Categorical and Numerical Data Types"""

# >2 categories
categorical_features = [
    "PROPERTY TYPE",
]

# continuous or binary
numerical_features = ["BEDS", "BATHS", "SQUARE FEET", "LOT SIZE"]


lasso_name = "lasso"
lasso = Lasso(random_state=3)
tuned_parameters_lasso = [
    {
        f"{lasso_name}__fit_intercept": [True, False],
        f"{lasso_name}__precompute": [True, False],
        f"{lasso_name}__copy_X": [True, False],
        f"{lasso_name}__max_iter": [100, 500, 1000, 2000],
        f"{lasso_name}__tol": [1e-4, 1e-3],
        f"{lasso_name}__warm_start": [True, False],
        f"{lasso_name}__positive": [True, False],
    }
]
lasso_definition = {
    "clc": lasso,
    "estimator_name": lasso_name,
    "tuned_parameters": tuned_parameters_lasso,
    "randomized_grid": False,
    "early": False,
}

################################# XGB Regression ###############################

xgb_name = "xgb"
xgb = XGBRegressor(random_state=3)
tuned_parameters_xgb = [
    {
        f"{xgb_name}__learning_rates": [0.1, 0.01, 0.05][:1],
        f"{xgb_name}__n_estimators": [100, 200, 300][
            :1
        ],  # Number of trees. Equivalent to n_estimators in GB
        f"{xgb_name}__max_depths": [3, 5, 7][:1],  # Maximum depth of the trees
        f"{xgb_name}__subsamples": [0.8, 1.0][
            :1
        ],  # Subsample ratio of the training instances
        f"{xgb_name}__colsample_bytree": [0.8, 1.0][:1],
        f"{xgb_name}__eval_metric": ["logloss"],
        f"{xgb_name}__early_stopping_rounds": [10],
        f"{xgb_name}__tree_method": ["hist"],
        f"{xgb_name}__stopping_mode": ["min"],
        f"{xgb_name}__stopping_patience": [5],
        f"{xgb_name}__verbose": [False],
    }
]

xgb_definition = {
    "clc": xgb,
    "estimator_name": xgb_name,
    "tuned_parameters": tuned_parameters_xgb,
    "randomized_grid": False,
    "early": True,
}

"""## Set Up The Feature Space and Dependent Variable"""

outcome = "PRICE"
features = numerical_features + categorical_features
X, y = df[features], df[outcome]

model_definitions = {
    lasso_name: lasso_definition,
    xgb_name: xgb_definition,
}

"""## Set Up The Column Transformers"""

# Define transformers for different column types
numerical_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("imputer", SimpleImputer(strategy="mean")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Create the ColumnTransformer with passthrough
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="passthrough",
)

"""## Run The Model

For these examples, we will not be doing a KFold split nor will we be calibrating our models, so we will inherently set the following to `False` as follows.
"""

kfold = False
calibrate = False

"""### Lasso Regression"""

# Step 4: define model object
model_type = "lasso"
clc = model_definitions[model_type]["clc"]
estimator_name = model_definitions[model_type]["estimator_name"]

# Set the parameters by cross-validation
tuned_parameters = model_definitions[model_type]["tuned_parameters"]
rand_grid = model_definitions[model_type]["randomized_grid"]
early_stop = model_definitions[model_type]["early"]

model_lasso = Model(
    name=f"lasso_{model_type}",
    estimator_name=estimator_name,
    model_type="regression",
    calibrate=calibrate,
    estimator=clc,
    kfold=kfold,
    pipeline_steps=[("ColumnTransformer", preprocessor)],
    stratify_y=False,
    grid=tuned_parameters,
    randomized_grid=rand_grid,
    boost_early=early_stop,
    scoring=["r2"],
    random_state=3,
    n_jobs=2,
)

"""#### Perform Grid Search Parameter Tuning and Retrieve Split Data"""

model_lasso.grid_search_param_tuning(X, y)

X_train, y_train = model_lasso.get_train_data(X, y)
X_test, y_test = model_lasso.get_test_data(X, y)
X_valid, y_valid = model_lasso.get_valid_data(X, y)

"""#### Fit The Model"""

model_lasso.fit(X_train, y_train)

"""#### Return Metrics (Optional)"""

print("Validation Metrics")
model_lasso.return_metrics(X_valid, y_valid)

print("Test Metrics")
model_lasso.return_metrics(X_test, y_test)

print("Report model metrics dataframe for Lasso")

lasso_metrics_df = report_model_metrics(model_lasso, X_test, y_test)

# print(lasso_metrics_df)

"""## XGBoost

## Initialize and Configure the XGB ``Model``
"""

# Step 4: define model object
model_type = "xgb"
clc = model_definitions[model_type]["clc"]
estimator_name = model_definitions[model_type]["estimator_name"]

# Set the parameters by cross-validation
tuned_parameters = model_definitions[model_type]["tuned_parameters"]
rand_grid = model_definitions[model_type]["randomized_grid"]
early_stop = model_definitions[model_type]["early"]

model_xgb = Model(
    name=f"xgb_{model_type}",
    estimator_name=estimator_name,
    model_type="regression",
    calibrate=calibrate,
    estimator=clc,
    kfold=kfold,
    pipeline_steps=[("ColumnTransformer", preprocessor)],
    stratify_y=False,
    grid=tuned_parameters,
    randomized_grid=rand_grid,
    boost_early=early_stop,
    scoring=["r2"],
    random_state=3,
    n_jobs=2,
)

"""#### Perform Grid Search Parameter Tuning and Retrieve Split Data"""

model_xgb.grid_search_param_tuning(
    X,
    y,
)

X_train, y_train = model_xgb.get_train_data(X, y)
X_test, y_test = model_xgb.get_test_data(X, y)
X_valid, y_valid = model_xgb.get_valid_data(X, y)

"""#### Fit The Model"""

model_xgb.fit(
    X_train,
    y_train,
    validation_data=[X_valid, y_valid],
)

"""#### Return Metrics (Optional)"""

print("Validation Metrics")
model_xgb.return_metrics(X_valid, y_valid)

print("Test Metrics")
model_xgb.return_metrics(X_test, y_test)

print("Report model metrics for XGB")
xgb_metrics = report_model_metrics(model_xgb, X_test, y_test)


# print(xgb_metrics)
