import pandas as pd
import numpy as np

from model_tuner.model_tuner_utils import Model
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import load_iris

print()
print(f"Model Tuner version: {model_tuner.__version__}")
print(f"Model Tuner authors: {model_tuner.__author__}")
print()

data = load_iris()
X = data.data
y = data.target

X = pd.DataFrame(X)
y = pd.DataFrame(y).squeeze()


estimator = RandomForestClassifier()

estimator_name = "rf_mc"
xgbearly = False

tuned_parameters = {
    f"{estimator_name}__max_depth": [3, 10, 15],
    f"{estimator_name}__n_estimators": [5, 10, 15, 20],
}

kfold = False
calibrate = False


model = Model(
    name="Random Forest Multi Class",
    model_type="classification",
    estimator_name=estimator_name,
    pipeline_steps=[("impute", SimpleImputer())],
    calibrate=calibrate,
    estimator=estimator,
    kfold=kfold,
    stratify_y=True,
    grid=tuned_parameters,
    multi_label=True,
    randomized_grid=False,
    n_iter=4,
    scoring=["roc_auc_ovr"],
    n_jobs=-2,
    random_state=42,
    class_labels=["1", "2", "3"],
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
