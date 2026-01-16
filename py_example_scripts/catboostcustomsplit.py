from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from model_tuner.model_tuner_utils import Model
import model_tuner

print()
print(f"Model Tuner version: {model_tuner.__version__}")
print(f"Model Tuner authors: {model_tuner.__author__}")
print()

estimator = CatBoostClassifier(verbose=0)
bc = load_breast_cancer(as_frame=True)["frame"]
bc_cols = [cols for cols in bc.columns if "target" not in cols]


X = bc[bc_cols]
y = bc["target"]

estimator_name = "cat"

tuned_parameters = {
    f"{estimator_name}__depth": [10],
    f"{estimator_name}__learning_rate": [1e-4],
    f"{estimator_name}__n_estimators": [1000],
    f"{estimator_name}__early_stopping_rounds": [10],
    f"{estimator_name}__verbose": [0],
    f"{estimator_name}__eval_metric": ["Logloss"],
}


X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, train_size=0.8)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, train_size=0.8
)

custom_splits = {
    "X_train": X_train,
    "y_train": y_train,
    "X_valid": X_val,
    "y_valid": y_val,
    "X_test": X_test,
    "y_test": y_test,
}

print(f"\n{'-'*80}\nCustom Data Splits Summary:\n{'-'*80}")
print(f"X_train = {X_train.shape[0]} rows")
print(f"X_valid = {X_val.shape[0]} rows")
print(f"X_test = {X_test.shape[0]} rows\n{'-'*80}")
print(f"Total = {X.shape[0]} rows\n{'-'*80}")
print(f"X_train is {X_train.shape[0]/X.shape[0]*100:.2f}% of total data")
print(f"X_valid is {X_val.shape[0]/X.shape[0]*100:.2f}% of total data")
print(f"X_test is {X_test.shape[0]/X.shape[0]*100:.2f}% of total data")

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
    calibrate=True,
)


model.grid_search_param_tuning(X, y, custom_splits=custom_splits)

X_train, y_train = model.get_train_data(X, y)
X_test, y_test = model.get_test_data(X, y)
X_valid, y_valid = model.get_valid_data(X, y)

model.fit(X_train, y_train, validation_data=[X_valid, y_valid])

model.calibrateModel(X, y, custom_splits=custom_splits)

print("Validation Metrics")
model.return_metrics(X_valid, y_valid)
print("Test Metrics")
model.return_metrics(X_test, y_test)

y_prob = model.predict_proba(X_test)

### F1 Weighted
y_pred = model.predict(X_test, optimal_threshold=True)
