from model_tuner import Model
import pandas as pd
import numpy as np
import model_tuner

from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline

import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action="ignore", category=DataConversionWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

################################################################################
# Load dataset
################################################################################

student_performance = fetch_ucirepo(id=320)

X = student_performance.data.features
y = student_performance.data.targets

# Average of G1, G2, G3
y = (y["G1"] + y["G2"] + y["G3"]) / 3
y = y.squeeze()

################################################################################
# Feature preprocessing
################################################################################

# Keep numeric features only
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
X = X[numerical_cols]

# Missingness indicators
X_missing = X.isna().astype(int)
X_missing.columns = [f"{c}_missing" for c in X.columns]

X = pd.concat([X, X_missing], axis=1)

print(f"\nNew X shape (with missingness indicators): {X.shape}")

numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, X.columns.tolist()),
    ]
)

################################################################################
# Lasso + RFE
################################################################################

lasso_name = "lasso"
lasso = Lasso(random_state=3, max_iter=10000)

# Grid for final model
tuned_parameters_lasso = [
    {
        f"{lasso_name}__alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
        f"{lasso_name}__fit_intercept": [True],
    }
]

# RFE estimator (separate from final model)
rfe_estimator = Lasso(
    alpha=0.01,
    max_iter=10000,
    random_state=3,
)

rfe = RFE(
    estimator=rfe_estimator,
    n_features_to_select=7,
    step=1,
)

pipeline_steps_lasso = [
    ("preprocess", preprocessor),
    ("rfe", rfe),
]

model_lasso = Model(
    name="lasso_rfe_student_performance",
    estimator_name=lasso_name,
    model_type="regression",
    calibrate=False,
    estimator=lasso,
    kfold=False,
    pipeline_steps=pipeline_steps_lasso,
    feature_selection=True,
    stratify_y=False,
    grid=tuned_parameters_lasso,
    randomized_grid=False,
    boost_early=False,
    scoring=["r2"],
    random_state=3,
    n_jobs=2,
)

################################################################################
# Train / evaluate
################################################################################


model_lasso.grid_search_param_tuning(X, y)

X_train, y_train = model_lasso.get_train_data(X, y)
X_test, y_test = model_lasso.get_test_data(X, y)
X_valid, y_valid = model_lasso.get_valid_data(X, y)

model_lasso.fit(X_train, y_train)


print("\nValidation Metrics (Lasso + RFE)")
model_lasso.return_metrics(
    X_valid,
    y_valid,
    print_best_feats=False,
)

print("\nTest Metrics (Lasso + RFE)")
model_lasso.return_metrics(
    X_test,
    y_test,
    return_dict=True,
    print_best_feats=True,
)
