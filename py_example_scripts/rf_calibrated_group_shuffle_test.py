import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from model_tuner.model_tuner_utils import Model
import model_tuner
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def main():
    print()
    print(f"Model Tuner version: {model_tuner.__version__}")
    print(f"Model Tuner authors: {model_tuner.__author__}")
    print()

    # Load dataset
    bc = load_breast_cancer(as_frame=True)["frame"]
    bc_cols = [cols for cols in bc.columns if "target" not in cols]
    X = bc[bc_cols]
    y = bc["target"]

    n_samples = X.shape[0]
    group_size = 10
    groups = np.arange(n_samples // group_size)
    groups = np.repeat(groups, group_size)

    # handle any leftover samples
    if len(groups) < n_samples:
        groups = np.concatenate(
            [groups, np.full(n_samples - len(groups), groups[-1] + 1)]
        )

    # Add to DataFrame
    X["group_id"] = groups

    rstate = 42

    print(X.shape)

    # Define Random Forest classifier
    estimator = RandomForestClassifier(
        class_weight="balanced",
        random_state=rstate,
        n_jobs=-1,  # Handle class imbalance
    )

    estimator_name = "rf"

    tuned_parameters = {
        f"{estimator_name}__n_estimators": [100, 200],  # Number of trees in the forest
        f"{estimator_name}__max_depth": [5, 10, None],  # Control tree depth
        f"{estimator_name}__min_samples_split": [
            2,
            5,
        ],  # Minimum samples required to split
    }

    kfold = False
    calibrate = True  # Allow calibration for probability outputs

    pipeline = [
        ("StandardScalar", StandardScaler()),
        ("Preprocessor", SimpleImputer()),
    ]

    # X = X.set_index('group_id')
    # Define model pipeline
    model = Model(
        name="Random Forest Classifier",
        estimator_name=estimator_name,
        calibrate=calibrate,
        model_type="classification",
        estimator=estimator,
        pipeline_steps=pipeline,
        kfold=kfold,
        stratify_y=False,
        grid=tuned_parameters,
        randomized_grid=False,
        n_iter=4,
        boost_early=False,  # Not applicable for Random Forest
        scoring=["roc_auc"],
        n_jobs=-2,
        random_state=rstate,
        groups=X["group_id"],
    )

    # Perform grid search
    model.grid_search_param_tuning(X, y, f1_beta_tune=True)

    ### Extract Training, Validation, and Test Splits
    X_train, y_train = model.get_train_data(X, y)
    X_valid, y_valid = model.get_valid_data(X, y)
    X_test, y_test = model.get_test_data(X, y)


    print("-" * 80)
    print(
        f"\nTrain_Val_Test size: Train = {X_train.shape[0]}, Validation = {X_valid.shape[0]}, Test = {X_test.shape[0]}\n"
    )
    print(
        f"Total Train_Val_Test size: {X_train.shape[0] + X_valid.shape[0] + X_test.shape[0]}"
    )

    print(
        f"\nSum of overlap with Validation Set: {(X.loc[X_train.index, 'group_id'].isin(X.loc[X_valid.index, 'group_id'])).sum()}"
    )

    print(
        f"Percentage of overlap with Validation Set: {(X.loc[X_train.index, 'group_id'].isin(X.loc[X_valid.index, 'group_id'])).mean()}%\n",
    )

    print(
        f"\nSum of overlap with Test Set: "
        f"{X.loc[X_train.index, 'group_id'].isin(X.loc[X_test.index, 'group_id']).sum()}"
    )

    print(
        f"Percentage of overlap with Test Set: "
        f"{X.loc[X_train.index, 'group_id'].isin(X.loc[X_test.index, 'group_id']).mean()}\n"
    )

    print("Distributions in each split:")

    print("Train distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nValidation distribution:")
    print(y_valid.value_counts(normalize=True))

    print("\nTest distribution:")
    print(y_test.value_counts(normalize=True))

    print("-" * 80)

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


if __name__ == "__main__":
    main()
