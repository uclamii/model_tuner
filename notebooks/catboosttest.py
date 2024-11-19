from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from model_tuner.model_tuner_utils import Model


estimator = CatBoostClassifier(verbose=0)
bc = load_breast_cancer(as_frame=True)["frame"]
bc_cols = [cols for cols in bc.columns if "target" not in cols]


X = bc[bc_cols]
y = bc["target"]

estimator_name = "cat"

tuned_parameters = {
    f"{estimator_name}__depth": [10],
    f"{estimator_name}__learning_rate": [1e-4],
    f"{estimator_name}__n_estimators": [30],
    f"{estimator_name}__early_stopping_rounds": [10],
    f"{estimator_name}__verbose": [0],
    f"{estimator_name}__eval_metric": ["Logloss"],
}

model = Model(
    name="Catboost Early",
    model_type="classification",
    estimator_name=estimator_name,
    estimator=estimator,
    pipeline_steps=[],
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

print("Validation Metrics")
model.return_metrics(X_valid, y_valid)
print("Test Metrics")
model.return_metrics(X_test, y_test)

y_prob = model.predict_proba(X_test)

### F1 Weighted
y_pred = model.predict(X_test, optimal_threshold=True)
