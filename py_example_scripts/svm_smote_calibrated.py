from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from model_tuner.model_tuner_utils import Model
import model_tuner
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer


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

# Define SVM classifier
from sklearn.svm import SVC

estimator = SVC(class_weight="balanced", probability=True, random_state=rstate)

estimator_name = "svm"

tuned_parameters = {
    f"{estimator_name}__C": [1, 10],  # Focus on fewer values for regularization
    f"{estimator_name}__kernel": [
        "linear"
    ],  # Test only a single kernel (linear or rbf)
}


kfold = True
calibrate = True

pipeline = [
    ("StandardScalar", StandardScaler()),
    ("Preprocessor", SimpleImputer()),
]

# Define model pipeline
model = Model(
    name="SVM Classifier",
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
    boost_early=False,  # Not applicable for SVM
    scoring=["roc_auc"],
    n_jobs=-2,
    random_state=rstate,
    imbalance_sampler=SMOTE(random_state=rstate),
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
        f1_beta_tune=True
    )

# Evaluate metrics
print("Validation Metrics")
model.return_metrics(X, y, print_threshold=True, model_metrics=True)

# Predict probabilities and classes
y_prob = model.predict_proba(X)
y_pred = model.predict(X, optimal_threshold=True)
