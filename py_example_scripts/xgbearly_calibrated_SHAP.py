import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from model_tuner.model_tuner_utils import Model
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.linear_model import ElasticNet
from sklearn.compose import ColumnTransformer
from ucimlrepo import fetch_ucirepo
import shap
import model_tuner

print()
print(f"Model Tuner version: {model_tuner.__version__}")
print(f"Model Tuner authors: {model_tuner.__author__}")
print()

## Fetch dataset
aids_clinical_trials_group_study_175 = fetch_ucirepo(id=890)

## Data (as pandas dataframes)
X = aids_clinical_trials_group_study_175.data.features
y = aids_clinical_trials_group_study_175.data.targets

if isinstance(y, pd.DataFrame):
    y = y.squeeze()

## Check for zero-variance columns and drop them
zero_variance_columns = X.columns[X.var() == 0]
if not zero_variance_columns.empty:
    X = X.drop(columns=zero_variance_columns)

# bc = load_breast_cancer(as_frame=True)["frame"]
# bc_cols = [cols for cols in bc.columns if "target" not in cols]
# X = bc[bc_cols]
# y = bc["target"]

estimator = XGBClassifier(
    objective="binary:logistic",
)

estimator_name = "xgb"
xgbearly = True

tuned_parameters = {
    f"{estimator_name}__max_depth": [3],
    f"{estimator_name}__learning_rate": [1e-4],
    f"{estimator_name}__n_estimators": [100],
    f"{estimator_name}__early_stopping_rounds": [10],
    f"{estimator_name}__verbose": [0],
    f"{estimator_name}__eval_metric": ["logloss"],
}

kfold = False
calibrate = True

model = Model(
    name="XGBoost Early",
    estimator_name=estimator_name,
    calibrate=calibrate,
    model_type="classification",
    estimator=estimator,
    pipeline_steps=[
        SimpleImputer(),
    ],
    kfold=kfold,
    stratify_y=True,
    grid=tuned_parameters,
    randomized_grid=False,
    n_iter=4,
    boost_early=True,
    scoring=["roc_auc"],
    n_jobs=-2,
    random_state=42,
)


model.grid_search_param_tuning(X, y)

X_train, y_train = model.get_train_data(X, y)
X_test, y_test = model.get_test_data(X, y)
X_valid, y_valid = model.get_valid_data(X, y)

model.fit(X_train, y_train, validation_data=[X_valid, y_valid])

if model.calibrate:
    model.calibrateModel(
        X,
        y,
        score="roc_auc",
    )


print("Validation Metrics")
model.return_metrics(X_valid, y_valid)
print("Test Metrics")
model.return_metrics(X_test, y_test)

# y_prob = model.predict_proba(X_test)

# ### F1 Weighted
# y_pred = model.predict(X_test, optimal_threshold=True)

################################################################################
################################ SHAP explainer ################################
################################################################################

## The pipeline applies preprocessing (e.g., imputation, scaling) and feature
## selection (RFE) to X_test
X_test_transformed = model.get_preprocessing_pipeline().transform(X_test)

print(X_test_transformed)
# quit()
## The last estimator in the pipeline is the XGBoost model
xgb_classifier = model.estimator.estimator[-1]

## Feature names are required for interpretability in SHAP plots
feature_names = X_train.columns.to_list()

## Initialize the SHAP explainer with the model
explainer = shap.TreeExplainer(xgb_classifier)

## Compute SHAP values for the transformed dataset
shap_values = explainer.shap_values(X_test_transformed)

## Plot SHAP values
## Summary plot of SHAP values for all features across all data points
shap.summary_plot(
    shap_values,
    X_test_transformed,
    feature_names=feature_names,
)
