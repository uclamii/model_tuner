if __name__ == "__main__":

    # ## Adult Income Dataset

    # ## Import Requisite Libraries

    from ucimlrepo import fetch_ucirepo
    import pandas as pd
    import numpy as np
    from xgboost import XGBClassifier
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from model_tuner import Model
    import model_tuner

    print()
    print(f"Model Tuner version: {model_tuner.__version__}")
    print(f"Model Tuner authors: {model_tuner.__author__}")
    print()

    # fetch dataset
    adult = fetch_ucirepo(id=2)

    # data (as pandas dataframes)
    X = adult.data.features
    y = adult.data.targets

    print("-" * 80)
    print("X")
    print("-" * 80)

    print(X.head())  # inspect first 5 rows of X

    print("-" * 80)
    print("y")
    print("-" * 80)

    print(y.head())  # inspect first 5 rows of y

    y.loc[:, "income"] = y["income"].str.rstrip(".")  # Remove trailing periods

    # Check the updated value counts
    print(y["income"].value_counts())

    y.value_counts()

    y = y["income"].map({"<=50K": 0, ">50K": 1})

    outcome = ["y"]

    # >2 categories
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        "race",
    ]

    # continuous or binary
    numerical_features = X.select_dtypes(np.number).columns.to_list()

    xgb_name = "xgb"
    xgb = XGBClassifier(
        objective="binary:logistic",
        random_state=222,
    )
    xgbearly = True
    tuned_parameters_xgb = {
        f"{xgb_name}__max_depth": [3, 10, 20, 200, 500],
        f"{xgb_name}__learning_rate": [1e-4],
        f"{xgb_name}__n_estimators": [1000],
        f"{xgb_name}__early_stopping_rounds": [100],
        f"{xgb_name}__verbose": [0],
        f"{xgb_name}__eval_metric": ["logloss"],
    }

    xgb_definition = {
        "clc": xgb,
        "estimator_name": xgb_name,
        "tuned_parameters": tuned_parameters_xgb,
        "randomized_grid": False,
        "n_iter": 5,
        "early": xgbearly,
    }

    model_definitions = {
        xgb_name: xgb_definition,
    }

    # Define transformers for different column types
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
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",
    )

    model_type = "xgb"
    clc = xgb_definition["clc"]
    estimator_name = xgb_definition["estimator_name"]

    tuned_parameters = xgb_definition["tuned_parameters"]
    n_iter = xgb_definition["n_iter"]
    rand_grid = xgb_definition["randomized_grid"]
    early_stop = xgb_definition["early"]
    kfold = False
    calibrate = True

    # groups = pd.Series(X.index.tolist(), index=X.index)
    group_size = 10
    groups_arr = np.repeat(np.arange(np.ceil(len(X) / group_size)), group_size)[
        : len(X)
    ]
    groups = pd.Series(groups_arr, index=X.index)

    X["group_id"] = groups

    print(f"Shape of X: {X.shape}")
    print(f"X['group_id'].nunique(): {X['group_id'].nunique()}")

    model_xgb = Model(
        name=f"Adult_Income_{model_type}",
        estimator_name=estimator_name,
        calibrate=calibrate,
        estimator=clc,
        model_type="classification",
        kfold=kfold,
        pipeline_steps=[("ColumnTransformer", preprocessor)],
        stratify_y=True,
        stratify_cols=["race", "sex"],
        grid=tuned_parameters,
        randomized_grid=rand_grid,
        boost_early=early_stop,
        scoring=["roc_auc"],
        random_state=222,
        n_jobs=2,
        groups=groups,
    )

    model_xgb.grid_search_param_tuning(X, y, f1_beta_tune=True)

    X_train, y_train = model_xgb.get_train_data(X, y)
    X_test, y_test = model_xgb.get_test_data(X, y)
    X_valid, y_valid = model_xgb.get_valid_data(X, y)
    X_test, y_test = model_xgb.get_test_data(X, y)

    print("-" * 80)
    print(
        f"\nTrain_Val_Test size: Train = {X_train.shape[0]}, Validation = {X_valid.shape[0]}, Test = {X_test.shape[0]}\n"
    )
    print(
        f"Total Train_Val_Test size: {X_train.shape[0] + X_valid.shape[0] + X_test.shape[0]}"
    )

    print(
        f"\nSum of overlap with Validation Set: {groups.loc[X_train.index].isin(groups.loc[X_valid.index]).sum()}",
    )

    print(
        f"Percentage of overlap with Validation Set: {groups.loc[X_train.index].isin(groups.loc[X_valid.index]).mean()}%\n",
    )

    print(
        f"\nSum of overlap with Test Set: {groups.loc[X_train.index].isin(groups.loc[X_test.index]).sum()}",
    )

    print(
        f"Percentage of overlap with Test Set: {groups.loc[X_train.index].isin(groups.loc[X_test.index]).mean()}%\n",
    )

    print("Distributions in each split:")

    print("Train distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nValidation distribution:")
    print(y_valid.value_counts(normalize=True))

    print("\nTest distribution:")
    print(y_test.value_counts(normalize=True))

    print("-" * 80)

    model_xgb.fit(X_train, y_train, validation_data=[X_valid, y_valid])

    model_xgb.return_metrics(X_test, y_test, True)
