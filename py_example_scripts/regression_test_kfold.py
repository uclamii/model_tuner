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


"""## Set Up The Feature Space and Dependent Variable"""

outcome = "PRICE"
features = numerical_features + categorical_features
X, y = df[features], df[outcome]

model_definitions = {
    lasso_name: lasso_definition,
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

For these examples, we will not be doing a KFold split nor will we be 
calibrating our models, so we will inherently set the following to 
`False` as follows.
"""

kfold = True
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

"""#### Fit The Model"""

model_lasso.fit(X, y)

"""#### Return Metrics (Optional)"""

print("Return Metrics")
model_lasso.return_metrics(
    X,
    y,
    print_threshold=True,
    model_metrics=True,
    print_per_fold=True,
)
