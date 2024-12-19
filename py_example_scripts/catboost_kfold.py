from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
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
    f"{estimator_name}__n_estimators": [30],
}

model = Model(
    name="Catboost Early",
    estimator_name=estimator_name,
    model_type="classification",
    estimator=estimator,
    pipeline_steps=[],
    stratify_y=True,
    grid=tuned_parameters,
    randomized_grid=False,
    n_iter=4,
    boost_early=False,
    scoring=["roc_auc"],
    n_jobs=-2,
    random_state=42,
    kfold=True,
)


model.grid_search_param_tuning(X, y, f1_beta_tune=True)


model.fit(X, y)


y_prob = model.predict_proba(X)

### F1 Weighted
y_pred = model.predict(X)

metrics = model.return_metrics(X, y)
