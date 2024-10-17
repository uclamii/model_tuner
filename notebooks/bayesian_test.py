from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_breast_cancer
from model_tuner.model_tuner_utils import Model
from imblearn.over_sampling import SMOTE
from skopt.space import Real, Categorical, Integer

bc = load_breast_cancer(as_frame=True)["frame"]
bc_cols = [cols for cols in bc.columns if "target" not in cols]
X = bc[bc_cols]
y = bc["target"]

from xgboost import XGBClassifier


estimator = XGBClassifier(
    objective="binary:logistic",
)

estimator_name = "xgb"
xgbearly = False

tuned_parameters = {
    f"{estimator_name}__max_depth": Integer(2, 1000),
    f"{estimator_name}__learning_rate": Real(1e-5, 1e-1, "log-uniform"),
    f"{estimator_name}__n_estimators": Integer(3, 1000),
    f"{estimator_name}__gamma": Real(0, 4, "uniform"),
    "bayes__n_points": 1,
    "bayes__n_iter": 1,
}

kfold = False
calibrate = False


model = Model(
    name="Bayesian Test",
    estimator_name=estimator_name,
    calibrate=calibrate,
    estimator=estimator,
    pipeline_steps=[SimpleImputer()],
    kfold=True,
    bayesian=True,
    stratify_y=True,
    grid=tuned_parameters,
    randomized_grid=False,
    feature_selection=False,
    scoring=["roc_auc"],
    n_jobs=-2,
    random_state=42,
    imbalance_sampler=SMOTE(),
)


model.grid_search_param_tuning(X, y)


model.fit(X, y)

print("Validation Metrics")
model.return_metrics(X, y)
model.kfold = False

predictions = model.predict(X)
