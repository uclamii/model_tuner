### Defining columns to be scaled and columns to be onehotencoded
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from model_tuner import Model
from sklearn.pipeline import Pipeline
import model_tuner

print()
print(f"Model Tuner version: {model_tuner.__version__}")
print(f"Model Tuner authors: {model_tuner.__author__}")
print()

# Load Titanic dataset from OpenML
titanic = fetch_openml("titanic", version=1, as_frame=True)
df = titanic.data

# Convert target to numeric (sklearn datasets sometimes store target as string)
df["survived"] = titanic.target.astype(int)

X = df[[col for col in df.columns if col != "survived"]]
X = X.drop(
    columns=["name", "ticket", "cabin", "boat", "body", "home.dest", "embark_town"],
    errors="ignore",
)

y = df["survived"]

# Update categorical column selections
ohcols = ["embarked", "sex"]  # Removed "embark_town" and "who" since they are missing
scalercols = ["parch", "fare", "age", "pclass"]  # Keep numeric columns


numerical_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("imputer", SimpleImputer(strategy="mean")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Create the ColumnTransformer with passthrough
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, scalercols),
        ("cat", categorical_transformer, ohcols),
    ],
    remainder="passthrough",
)

# CatBoost definition
catboost_name = "catboost"
catboost = CatBoostClassifier(verbose=0)  # verbose=0 to suppress training output
tuned_parameters_catboost = {
    f"{catboost_name}__depth": [3, 5, 10],
    f"{catboost_name}__early_stopping_rounds": [10],
    f"{catboost_name}__learning_rate": [0.01, 0.03, 0.1],
}
catboost_definition = {
    "clc": catboost,
    "estimator_name": catboost_name,
    "tuned_parameters": tuned_parameters_catboost,
    "randomized_grid": True,
    "n_iter": 20,
    "early": True,
}

kfold = False

# Initialize titanic_model
titanic_model_catboost = Model(
    name="CatBoost_Titanic",
    estimator_name=catboost_definition["estimator_name"],
    model_type="classification",
    calibrate=True,
    estimator=catboost_definition["clc"],
    kfold=kfold,
    pipeline_steps=[("Preprocessor", preprocessor)],
    stratify_y=True,
    grid=catboost_definition["tuned_parameters"],
    randomized_grid=catboost_definition["randomized_grid"],
    n_iter=catboost_definition["n_iter"],
    scoring=["roc_auc"],
    random_state=42,
    boost_early=True,
    n_jobs=-1,
)

# Perform grid search
titanic_model_catboost.grid_search_param_tuning(X, y, f1_beta_tune=True)

# Get the training and validation data
X_train, y_train = titanic_model_catboost.get_train_data(X, y)
X_valid, y_valid = titanic_model_catboost.get_valid_data(X, y)
X_test, y_test = titanic_model_catboost.get_test_data(X, y)

# Fit the model (assuming best params are applied internally)
titanic_model_catboost.fit(X_train, y_train, validation_data=[X_valid, y_valid])

# Predict probabilities
prob_uncalibrated = titanic_model_catboost.predict_proba(X_test)[:, 1]

# Calibrate if needed
if titanic_model_catboost.calibrate:
    titanic_model_catboost.calibrateModel(X, y)
