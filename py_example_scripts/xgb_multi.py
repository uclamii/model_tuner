import pandas as pd
import numpy as np

from model_tuner.model_tuner_utils import Model, report_model_metrics
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.datasets import load_iris

print()
print(f"Model Tuner version: {model_tuner.__version__}")
print(f"Model Tuner authors: {model_tuner.__author__}")
print()

data = load_iris()
X = data.data
y = data.target


X = pd.DataFrame(X)
y = pd.DataFrame(y)

scalercols = X.columns

numerical_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("imputer", SimpleImputer(strategy="mean")),
    ]
)


# Create the ColumnTransformer with passthrough
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, scalercols),
    ],
    remainder="passthrough",
)


estimator = XGBClassifier(objective="multi:softprob")

estimator_name = "xgb_mc"
xgbearly = True

tuned_parameters = {
    f"{estimator_name}__max_depth": [3, 10, 15],
    f"{estimator_name}__n_estimators": [5, 10, 15, 20],
    f"{estimator_name}__eval_metric": ["mlogloss"],
    f"{estimator_name}__verbose": [0],
    f"{estimator_name}__early_stopping_rounds": [20],
}

kfold = False
calibrate = False


model = Model(
    name="XGB Multi Class",
    model_type="classification",
    estimator_name=estimator_name,
    pipeline_steps=[("ColumnTransformer", preprocessor)],
    calibrate=calibrate,
    estimator=estimator,
    kfold=kfold,
    stratify_y=True,
    boost_early=xgbearly,
    grid=tuned_parameters,
    multi_label=True,
    randomized_grid=False,
    n_iter=4,
    scoring=["roc_auc_ovr"],
    n_jobs=-2,
    random_state=42,
    class_labels=["Setosa", "Versicolor", "Virginica"],
)


model.grid_search_param_tuning(X, y)

X_train, y_train = model.get_train_data(X, y)
X_test, y_test = model.get_test_data(X, y)
X_valid, y_valid = model.get_valid_data(X, y)

model.fit(X_train, y_train, validation_data=[X_valid, y_valid])

print("Validation Metrics")
model.return_metrics(X_valid, y_valid)

y_prob = model.predict_proba(X_test)
print("Test Metrics")
model.return_metrics(X_test, y_test)

metrics_df = report_model_metrics(model, X_test, y_test)

# y_prob = model.predict_proba(X_test)[:, 1]

# ### F1 Weighted
# y_pred = model.predict(X_test, optimal_threshold=True)
