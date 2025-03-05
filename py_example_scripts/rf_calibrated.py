from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from model_tuner.model_tuner_utils import Model
import model_tuner
from sklearn.ensemble import RandomForestClassifier

print()
print(f"Model Tuner version: {model_tuner.__version__}")
print(f"Model Tuner authors: {model_tuner.__author__}")
print()

# Load dataset
bc = load_breast_cancer(as_frame=True)["frame"]
bc_cols = [cols for cols in bc.columns if "target" not in cols]
X = bc[bc_cols]
y = bc["target"]

rstate = 42

print(X.shape)

# Define Random Forest classifier
estimator = RandomForestClassifier(
    class_weight="balanced", random_state=rstate, n_jobs=-1  # Handle class imbalance
)

estimator_name = "rf"

tuned_parameters = {
    f"{estimator_name}__n_estimators": [100, 200],  # Number of trees in the forest
    f"{estimator_name}__max_depth": [5, 10, None],  # Control tree depth
    f"{estimator_name}__min_samples_split": [2, 5],  # Minimum samples required to split
}

kfold = False
calibrate = True  # Allow calibration for probability outputs

pipeline = [
    ("StandardScalar", StandardScaler()),
    ("Preprocessor", SimpleImputer()),
]

# Define model pipeline
model = Model(
    name="Random Forest Classifier",
    estimator_name=estimator_name,
    calibrate=calibrate,
    model_type="classification",
    estimator=estimator,
    pipeline_steps=pipeline,
    kfold=kfold,
    stratify_y=True,
    grid=tuned_parameters,
    randomized_grid=False,
    n_iter=4,
    boost_early=False,  # Not applicable for Random Forest
    scoring=["roc_auc"],
    n_jobs=-2,
    random_state=rstate,
)

# Perform grid search
model.grid_search_param_tuning(X, y, f1_beta_tune=True)

# Fit model
model.fit(X, y)

if model.calibrate:
    model.calibrateModel(
        X,
        y,
        score="roc_auc",
    )

# Evaluate metrics
print("Validation Metrics")
model.return_metrics(X, y, print_threshold=True, model_metrics=True)

# Predict probabilities and classes
y_prob = model.predict_proba(X)
y_pred = model.predict(X, optimal_threshold=True)
