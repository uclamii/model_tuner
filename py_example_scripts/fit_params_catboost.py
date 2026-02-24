if __name__ == "__main__":

    # ## Adult Income Dataset

    # ## Import Requisite Libraries

    from ucimlrepo import fetch_ucirepo
    import pandas as pd
    import numpy as np
    from catboost import CatBoostClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import FunctionTransformer
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

    cat_name = "cat"
    cat = CatBoostClassifier(
        random_state=222,
    )
    xgbearly = True
    tuned_parameters_cat = {
        f"{cat_name}__max_depth": [3, 10],
        f"{cat_name}__learning_rate": [1e-4],
        f"{cat_name}__n_estimators": [1000],
        f"{cat_name}__early_stopping_rounds": [100],
        f"{cat_name}__verbose": [0],
        f"{cat_name}__eval_metric": ["Logloss"],
    }

    cat_definition = {
        "clc": cat,
        "estimator_name": cat_name,
        "tuned_parameters": tuned_parameters_cat,
        "randomized_grid": True,
        "n_iter": 1,
        "early": xgbearly,
    }

    model_definitions = {
        cat_name: cat_definition,
    }

    # Define transformers for different column types
    numerical_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("imputer", SimpleImputer(strategy="mean")),
        ]
    )

    def to_str_func(X):
        # Convert categorical values to strings.
        # CatBoost can consume string categories directly, and this ensures
        # consistent dtype handling after imputation.
        return X.astype(str)

    # Create the ColumnTransformer with passthrough
    cat_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="__MISSING__"),
            ),
            (
                "to_str",
                FunctionTransformer(
                    to_str_func,
                    # IMPORTANT:
                    # ColumnTransformer.get_feature_names_out() requires every
                    # transformer in the pipeline to implement feature name
                    # propagation. FunctionTransformer does not provide this
                    # by default, which causes:
                    #
                    # AttributeError:
                    #   Estimator to_str does not provide get_feature_names_out
                    #
                    # Setting feature_names_out="one-to-one" tells sklearn that:
                    #
                    #   • output columns == input columns
                    #   • no columns are added, removed, or renamed
                    #
                    # This enables safe feature lineage tracking and allows:
                    #
                    #   pipeline.get_feature_names_out()
                    #
                    # to work correctly.
                    feature_names_out="one-to-one",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", cat_transformer, categorical_features),
        ],
        remainder="drop",
    )

    model_type = "cat"
    clc = cat_definition["clc"]
    estimator_name = cat_definition["estimator_name"]

    tuned_parameters = cat_definition["tuned_parameters"]
    n_iter = cat_definition["n_iter"]
    rand_grid = cat_definition["randomized_grid"]
    early_stop = cat_definition["early"]
    kfold = False
    calibrate = True

    model_cat = Model(
        name=f"Adult_Income_{model_type}",
        estimator_name=estimator_name,
        calibrate=calibrate,
        estimator=clc,
        model_type="classification",
        kfold=kfold,
        pipeline_steps=[("ColumnTransformer", preprocessor)],
        stratify_y=True,
        stratify_cols=["race", "sex"],
        n_iter=n_iter,
        grid=tuned_parameters,
        randomized_grid=rand_grid,
        boost_early=early_stop,
        scoring=["roc_auc"],
        random_state=222,
        n_jobs=2,
    )

    cat_feature_indices = list(
        range(
            len(numerical_features), len(numerical_features) + len(categorical_features)
        )
    )

    fit_params = {"cat__cat_features": cat_feature_indices}

    model_cat.grid_search_param_tuning(X, y, f1_beta_tune=True, fit_params=fit_params)

    X_train, y_train = model_cat.get_train_data(X, y)
    X_test, y_test = model_cat.get_test_data(X, y)
    X_valid, y_valid = model_cat.get_valid_data(X, y)

    model_cat.fit(
        X_train, y_train, validation_data=[X_valid, y_valid], fit_params=fit_params
    )

    model_cat.calibrateModel(X, y, f1_beta_tune=True, fit_params=fit_params)

    model_cat.return_metrics(X_test, y_test, True)
