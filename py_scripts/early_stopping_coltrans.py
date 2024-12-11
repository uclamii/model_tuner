### Defining columns to be scaled and columns to be onehotencoded
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import seaborn as sns
from sklearn.impute import SimpleImputer
from model_tuner import Model
from sklearn.pipeline import Pipeline
import model_tuner

print()
print(f"Model Tuner version: {model_tuner.__version__}")
print(f"Model Tuner authors: {model_tuner.__author__}")
print()

titanic = sns.load_dataset("titanic")
titanic.head()
X = titanic[[col for col in titanic.columns if col != "survived"]]
### Removing repeated data
X = X.drop(columns=["alive", "class", "embarked"])
y = titanic["survived"]


ohcols = ["embark_town", "who", "sex", "adult_male"]

ordcols = ["deck"]

scalercols = ["parch", "fare", "age", "pclass"]

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

ordinal_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ord_encoder", OrdinalEncoder()),
    ]
)


# Create the ColumnTransformer with passthrough
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, scalercols),
        ("cat", categorical_transformer, ohcols),
        ("ord", ordinal_transformer, ordcols),
    ],
    remainder="passthrough",
)


# random forest definition
xgb_name = "xgb"
xgb = XGBClassifier()
tuned_parameters_xgb = {
    f"{xgb_name}__max_depth": [3, 5, 10],
    f"{xgb_name}__n_estimators": [1000],
    f"{xgb_name}__early_stopping_rounds": [10],
}
xgb_definition = {
    "clc": xgb,
    "estimator_name": xgb_name,
    "tuned_parameters": tuned_parameters_xgb,
    "randomized_grid": True,
    "n_iter": 20,
    "early": True,
}


kfold = False

# Initialize titanic_model
titanic_model_xgb = Model(
    name="XGB_Titanic",
    estimator_name=xgb_definition["estimator_name"],
    model_type="classification",
    calibrate=True,
    estimator=xgb_definition["clc"],
    kfold=kfold,
    pipeline_steps=[("Preprocessor", preprocessor)],
    stratify_y=True,
    grid=xgb_definition["tuned_parameters"],
    randomized_grid=xgb_definition["randomized_grid"],
    n_iter=xgb_definition["n_iter"],
    scoring=["roc_auc"],
    random_state=42,
    boost_early=True,
    n_jobs=-1,
)


titanic_model_xgb.grid_search_param_tuning(X, y, f1_beta_tune=True)

X_train, y_train = titanic_model_xgb.get_train_data(X, y)
X_valid, y_valid = titanic_model_xgb.get_valid_data(X, y)
X_test, y_test = titanic_model_xgb.get_test_data(X, y)

titanic_model_xgb.fit(X_train, y_train, validation_data=[X_valid, y_valid])


prob_uncalibrated = titanic_model_xgb.predict_proba(X_test)[:, 1]

if titanic_model_xgb.calibrate == True:
    titanic_model_xgb.calibrateModel(X, y)
