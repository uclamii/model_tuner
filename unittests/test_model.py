import pytest
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from model_tuner import Model
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.linear_model import ElasticNet

from model_tuner.model_tuner_utils import train_val_test_split


@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    y = pd.Series(y)
    return X, y


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    y = pd.Series(y)
    return X, y


def test_model_initialization():
    name = "test_model"
    estimator_name = "lr"
    estimator = LogisticRegression()

    tuned_parameters = {
        estimator_name + "__C": np.logspace(-4, 0, 3),
    }
    model = Model(
        name=name,
        estimator_name=estimator_name,
        estimator=estimator,
        scoring=["roc_auc"],
        grid=tuned_parameters,
        model_type="classification",
    )
    assert model.name == "test_model"
    assert model.estimator_name == "lr"
    assert isinstance(model.estimator.named_steps[estimator_name], LogisticRegression)
    assert model.scoring == ["roc_auc"]
    assert isinstance(model.grid, ParameterGrid)
    assert isinstance(model.estimator, Pipeline)


def test_reset_estimator():
    name = "test_model"
    estimator_name = "lr"
    estimator = LogisticRegression(C=0.1)
    tuned_parameters = {
        estimator_name + "__tol": [1e-4, 1e-3],
    }
    model = Model(
        name=name,
        estimator_name=estimator_name,
        estimator=estimator,
        scoring=["roc_auc"],
        grid=tuned_parameters,
        model_type="classification",
    )

    model.estimator.named_steps[estimator_name].set_params(C=1.0)
    model.reset_estimator()
    assert model.estimator.get_params()["lr__C"] == 0.1


def test_fit_method(classification_data):
    estimator_name = "RF"
    tuned_parameters = {
        estimator_name + "__max_depth": [2, 30],
    }
    estimator = RandomForestClassifier(n_estimators=10)
    X, y = classification_data
    model = Model(
        name="test_model",
        estimator_name=estimator_name,
        estimator=estimator,
        scoring=["roc_auc"],
        grid=tuned_parameters,
        model_type="classification",
    )
    model.grid_search_param_tuning(X, y)

    X_train, y_train = model.get_train_data(X, y)
    model.fit(X_train, y_train, score="roc_auc")
    assert hasattr(model.estimator, "predict")


def test_predict_method(classification_data):
    X, y = classification_data
    estimator_name = "RF"
    tuned_parameters = {
        estimator_name + "__max_depth": [2, 30],
    }
    model = Model(
        name="test_model",
        estimator_name=estimator_name,
        estimator=RandomForestClassifier(n_estimators=10),
        scoring=["accuracy"],
        grid=tuned_parameters,
        model_type="classification",
    )
    model.grid_search_param_tuning(X, y)
    X_train, y_train = model.get_train_data(X, y)
    model.fit(X_train, y_train, score="accuracy")
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})


def test_predict_proba_method(classification_data):
    X, y = classification_data
    estimator_name = "RF"
    tuned_parameters = {
        estimator_name + "__max_depth": [2, 30],
    }

    model = Model(
        name="test_model",
        estimator_name=estimator_name,
        estimator=RandomForestClassifier(n_estimators=10),
        scoring=["accuracy"],
        grid=tuned_parameters,
        model_type="classification",
    )
    model.grid_search_param_tuning(X, y)
    X_train, y_train = model.get_train_data(X, y)
    model.fit(X_train, y_train, score="accuracy")

    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1)

    # Assertions
    assert proba is not None, "predict_proba returned None"
    assert len(proba) == len(
        X
    ), "predict_proba did not return correct number of probabilities"

    # Check if the probabilities sum to 1 for each sample
    for p in proba:
        assert abs(sum(p) - 1.0) < 1e-6, f"Probabilities do not sum to 1: {p}"


def test_grid_search_param_tuning_early(classification_data):
    X, y = classification_data
    estimator_name = "xgb"
    tuned_parameters = {
        estimator_name + "__learning_rate": [1e-4],
        estimator_name + "__n_estimators": [1000],
        estimator_name + "__early_stopping_rounds": [10],
        estimator_name + "__verbose": [0],
        estimator_name + "__eval_metric": ["logloss"],
    }
    model = Model(
        name="test_model",
        estimator_name=estimator_name,
        estimator=XGBClassifier(),
        grid=tuned_parameters,
        scoring=["accuracy"],
        boost_early=True,
        model_type="classification",
    )
    model.grid_search_param_tuning(X, y)
    assert model.best_params_per_score["accuracy"]["params"] is not None


def test_return_metrics_classification(classification_data):
    """
    Test the return_metrics method for a classification model.
    """
    X, y = classification_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Initialize the classification model
    model = Model(
        name="test_classification_model",
        estimator_name="lr",
        model_type="classification",
        estimator=LogisticRegression(),
        scoring=["roc_auc"],
        grid={"lr__C": [0.01, 0.1, 1]},
    )

    # Perform grid search to populate `best_params_per_score`
    model.grid_search_param_tuning(X_train, y_train)

    # Train the model
    model.fit(X_train, y_train)

    # Test return_metrics method
    metrics = model.return_metrics(X_test, y_test, return_dict=True)

    # Validate the structure and content of the metrics
    assert isinstance(metrics, dict), "Expected return_metrics to return a dictionary."
    assert "Classification Report" in metrics, "Classification Report is missing."
    assert "Confusion Matrix" in metrics, "Confusion Matrix is missing."
    assert isinstance(
        metrics["Classification Report"], dict
    ), "Classification Report should be a dictionary."
    assert isinstance(
        metrics["Confusion Matrix"], (np.ndarray, list)
    ), "Confusion Matrix should be an array or list."


def test_return_metrics_regression(regression_data):
    """
    Test the return_metrics method for a regression model.
    """
    X, y = regression_data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
    )

    # Initialize the regression model
    model = Model(
        name="test_regression_model",
        estimator_name="lr",
        model_type="regression",
        estimator=LinearRegression(),
        scoring=["r2"],
        grid={"lr__fit_intercept": [True, False]},
    )

    # Perform grid search to populate `best_params_per_score`
    model.grid_search_param_tuning(X_train, y_train)

    # Train the model
    model.fit(X_train, y_train)

    # Test return_metrics method
    metrics = model.return_metrics(X_test, y_test, return_dict=True)

    # Validate the results
    assert isinstance(metrics, dict), "Expected return_metrics to return a dictionary."
    expected_keys = [
        "Explained Variance",
        "Mean Absolute Error",
        "Mean Squared Error",
        "Median Absolute Error",
        "R2",
        "RMSE",
    ]
    print(metrics)
    for key in expected_keys:
        assert key in metrics, f"Expected metric '{key}' to be in the results."
        assert isinstance(
            metrics[key], (int, float)
        ), f"Metric '{key}' should be a numeric value."


def test_imbalance_sampler_integration(classification_data):
    """
    Test the integration of imbalance sampler in the model pipeline.
    """
    X, y = classification_data

    ## Define model parameters
    name = "test_model"
    estimator_name = "lr"
    estimator = LogisticRegression()
    tuned_parameters = {
        estimator_name + "__C": np.logspace(-4, 0, 3),
    }

    ## Create the model with imbalance sampler
    model = Model(
        name=name,
        estimator_name=estimator_name,
        model_type="classification",
        estimator=estimator,
        scoring=["roc_auc"],
        grid=tuned_parameters,
        imbalance_sampler=None,  # Set as needed
    )

    ## Check that model is initialized correctly
    assert model.name == name
    assert model.estimator_name == estimator_name
    assert isinstance(
        model.estimator.named_steps[estimator_name],
        LogisticRegression,
    )

    ## Run imbalance sampler logic
    if model.imbalance_sampler:
        sampler = model.estimator.named_steps["resampler"]
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        assert len(X_resampled) > len(X), "Resampling failed to increase data size"
        assert (
            y_resampled.value_counts().min() > 0
        ), "Resampling failed to balance classes"
    else:
        assert not hasattr(
            model.estimator.named_steps, "resampler"
        ), "No imbalance sampler expected"


def test_bootstrapped_metrics_consistency(classification_data):
    """
    Test the consistency of bootstrapped metrics.
    """
    X, y = classification_data

    ## Reuse model initialization from test_imbalance_sampler_integration
    name = "test_model"
    estimator_name = "lr"
    estimator = LogisticRegression()
    tuned_parameters = {
        estimator_name + "__C": np.logspace(-4, 0, 3),
    }

    model = Model(
        name=name,
        estimator_name=estimator_name,
        model_type="classification",
        estimator=estimator,
        scoring=["roc_auc"],
        grid=tuned_parameters,
    )

    ## Assert model is initialized correctly
    assert model.name == name
    assert model.estimator_name == estimator_name
    assert isinstance(model.estimator.named_steps[estimator_name], LogisticRegression)

    ## Perform grid search to populate `best_params_per_score`
    model.grid_search_param_tuning(X, y)

    ## Fit the model to avoid NotFittedError
    model.fit(X, y)

    ## Test bootstrapped metrics
    metrics = ["precision", "roc_auc"]
    bootstrap_results = model.return_bootstrap_metrics(
        X_test=X,
        y_test=y,
        metrics=metrics,
        num_resamples=100,
    )

    ## Validate the structure and content of bootstrap results
    ## Check if the result is a DataFrame
    assert isinstance(bootstrap_results, pd.DataFrame), "Expected a DataFrame"

    ## Validate that the metrics are correctly represented in the DataFrame
    for metric in metrics:
        assert metric in bootstrap_results["Metric"].values, f"{metric} not in results"

    ## Check columns for mean and confidence intervals
    required_columns = ["Metric", "Mean", "95% CI Lower", "95% CI Upper"]
    for col in required_columns:
        assert col in bootstrap_results.columns, f"Missing column: {col}"

    ## Validate that the number of rows corresponds to the metrics tested
    assert len(bootstrap_results) == len(
        metrics
    ), "Mismatch in number of metrics returned"

    print(bootstrap_results)


@pytest.fixture
def sample_dataframe():
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_df = pd.Series(y, name="target")
    return X_df, y_df


def test_train_val_test_split(sample_dataframe):
    X, y = sample_dataframe
    train_size = 0.6
    validation_size = 0.2
    test_size = 0.2

    X_train, X_valid, X_test, y_train, y_valid, y_test = train_val_test_split(
        X,
        y,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        random_state=42,
    )

    # Check the sizes of the splits
    assert len(X_train) == int(0.6 * len(X))
    assert len(X_valid) == int(0.2 * len(X))
    assert len(X_test) == int(0.2 * len(X))

    assert len(y_train) == int(0.6 * len(y))
    assert len(y_valid) == int(0.2 * len(y))
    assert len(y_test) == int(0.2 * len(y))


def test_train_val_test_split_stratify(sample_dataframe):
    X, y = sample_dataframe
    X["stratify_col"] = np.random.choice(
        [0, 1], size=len(X)
    )  # Add a stratification column
    stratify_cols = ["stratify_col"]

    X_train, X_valid, X_test, y_train, y_valid, y_test = train_val_test_split(
        X,
        y,
        stratify_y=True,
        stratify_cols=stratify_cols,
        train_size=0.6,
        validation_size=0.2,
        test_size=0.2,
        random_state=42,
    )

    # Assert that the distribution of stratification column is similar across splits
    for df in [X_train, X_valid, X_test]:
        np.testing.assert_allclose(
            df["stratify_col"].value_counts(normalize=True).values,
            X["stratify_col"].value_counts(normalize=True).values,
            rtol=1e-2,
        )


@pytest.fixture
def initialized_model():
    lr = LogisticRegression()
    param_grid = {"logistic_regression__C": [0.1, 1, 10]}  # Example param grid

    return Model(
        name="test_model",
        estimator_name="logistic_regression",
        estimator=lr,
        model_type="classification",
        grid=param_grid,
        # scoring=["roc_auc"],
    )


@pytest.fixture
def initialized_kfold_lr_model():
    lr = LogisticRegression()
    param_grid = {"logistic_regression__C": [0.1, 1, 10]}  # Example param grid

    return Model(
        name="test_model",
        estimator_name="logistic_regression",
        estimator=lr,
        model_type="classification",
        grid=param_grid,
        kfold=True,
        # scoring=["roc_auc"],
    )


def test_fit_basic(initialized_model, classification_data):
    X, y = classification_data
    model = initialized_model

    # first need to do gridsearch
    model.grid_search_param_tuning(X, y)

    # Fit the model
    model.fit(X, y)

    # Check if estimator is fitted by checking if steps exist
    assert hasattr(
        model.estimator, "steps"
    ), "Model should be fitted with steps attribute set."


def test_fit_with_validation(initialized_model, classification_data):
    X, y = classification_data
    model = initialized_model

    # Create a small validation set
    X_validation, y_validation = X.iloc[:10], y.iloc[:10]

    # first need to do gridsearch
    model.grid_search_param_tuning(X, y)

    # Fit the model with validation data
    model.fit(X, y, validation_data=(X_validation, y_validation))

    # Check if model remained consistent
    assert hasattr(
        model.estimator, "steps"
    ), "Model should be fitted even when validation data is used."


def test_fit_without_labels(initialized_model, classification_data):
    X, y = classification_data
    model = initialized_model

    with pytest.raises(ValueError):
        # first need to do gridsearch
        model.grid_search_param_tuning(X, y)
        # Fit should raise an error when no labels are provided
        model.fit(X, None)


def test_fit_with_empty_dataset(initialized_model):
    model = initialized_model
    X_empty, y_empty = pd.DataFrame(), pd.Series()

    with pytest.raises(ValueError):
        # first need to do gridsearch
        model.grid_search_param_tuning(X_empty, y_empty)
        # Fit should raise an error when the dataset is empty
        model.fit(X_empty, y_empty)


@pytest.fixture
def initialized_xgb_model():
    xgb = XGBClassifier(eval_metric="logloss")
    estimator_name = "xgbclassifier"
    param_grid = {
        f"{estimator_name}__max_depth": [3, 10, 20, 200, 500],
        f"{estimator_name}__learning_rate": [1e-4],
        f"{estimator_name}__n_estimators": [50],
        f"{estimator_name}__early_stopping_rounds": [10],
        f"{estimator_name}__verbose": [0],
        f"{estimator_name}__eval_metric": ["logloss"],
    }  # Example param grid for XGBoost

    return Model(
        name="xgboost_model",
        estimator_name=estimator_name,
        estimator=xgb,
        model_type="classification",
        grid=param_grid,
        boost_early=True,  # Enable early stopping
        # Include other necessary parameters
    )


def test_fit_with_early_stopping_and_validation(
    initialized_xgb_model, classification_data
):
    X, y = classification_data
    model = initialized_xgb_model

    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = (
        X.iloc[:50],
        X.iloc[50:],
        y.iloc[:50],
        y.iloc[50:],
    )

    # first need to do gridsearch
    model.grid_search_param_tuning(X, y)

    # Fit the model with early stopping
    model.fit(
        X=X_train,
        y=y_train,
        validation_data=(X_valid, y_valid),
    )

    # Check if estimator is fitted
    assert hasattr(
        model.estimator, "steps"
    ), "Model should have a fitted attribute set."

    # Check that the early stopping logic was applied
    assert hasattr(model.estimator.steps[0][1], "best_iteration") or hasattr(
        model.estimator.steps[0][1], "best_ntree_limit"
    ), "Model should have early stopping logic applied."

    # Ensure the regularization logic finalized correctly
    assert (
        model.estimator.steps[0][1].best_iteration <= 50
    ), "Model should stop before max estimators if early stopping is effective."


def test_get_best_score_params(initialized_kfold_lr_model, classification_data):
    X, y = classification_data
    # note this method is only used with kfold so needs to be turned on
    model = initialized_kfold_lr_model

    # first need to do gridsearch
    model.grid_search_param_tuning(X, y)

    # Run the method to find the best parameters
    model.get_best_score_params(X, y)

    # Verify that best_params_per_score is set and contains expected keys
    assert (
        "roc_auc" in model.best_params_per_score
    ), "Best score for roc_auc should be in results."

    best_params = model.best_params_per_score["roc_auc"]["params"]

    # Check that the best parameters are from the predefined grid
    assert best_params["logistic_regression__C"] in [0.1, 1, 10]
    # We could also assert that the best score is assigned (though its value may vary)
    assert (
        "score" in model.best_params_per_score["roc_auc"]
    ), "Best score value should be present."


def test_return_bootstrap_metrics_classification(
    initialized_model, classification_data
):
    X, y = classification_data
    model = initialized_model

    # first need to do gridsearch
    model.grid_search_param_tuning(X, y)

    # Fit the model
    model.fit(X, y)

    # Define a simple metric set
    metrics = ["roc_auc", "f1"]

    # Obtain bootstrap metrics
    bootstrap_metrics = model.return_bootstrap_metrics(X, y, metrics=metrics)

    print(bootstrap_metrics)

    # Check the type and content
    assert isinstance(
        bootstrap_metrics, pd.DataFrame
    ), "Bootstrap metrics should be returned as a pandas dataframe."
    assert all(
        metric in bootstrap_metrics["Metric"].tolist() for metric in metrics
    ), "All specified metrics should be in the result."
    assert all(
        col in bootstrap_metrics.columns
        for col in [
            "Mean",
            "95% CI Lower",
            "95% CI Upper",
        ]
    ), "Each metric should contain these results Mean, 95% CI Lower, 95% CI Upper."


def test_return_bootstrap_metrics_with_empty_data(initialized_model):
    model = initialized_model
    X_empty = pd.DataFrame()
    y_empty = pd.Series()

    # Expect an error with empty datasets
    with pytest.raises(ValueError):
        model.return_bootstrap_metrics(X_empty, y_empty, metrics=["roc_auc"])


def test_return_bootstrap_metrics_unrecognized_metric(
    initialized_model, classification_data
):
    X, y = classification_data
    model = initialized_model

    # first need to do gridsearch
    model.grid_search_param_tuning(X, y)

    # Fit the model
    model.fit(X, y)

    # Call method with an unrecognized metric
    with pytest.raises(ValueError):
        model.return_bootstrap_metrics(X, y, metrics=["unknown_metric"])


from collections.abc import Mapping, Iterable


# Define parameter sets for initialization
@pytest.fixture
def lr_model_parameters():
    return {
        "name": "default_model",
        "estimator_name": "logistic_regression",
        "estimator": LogisticRegression(),
        "model_type": "classification",
        "grid": {"logistic_regression__C": [0.1, 1, 10]},
    }


@pytest.fixture
def xgboost_model_parameters():
    return {
        "name": "xgboost_model",
        "estimator_name": "xgbclassifier",
        "estimator": XGBClassifier(eval_metric="logloss"),
        "model_type": "classification",
        "grid": {
            "xgbclassifier__n_estimators": [10, 50],
            "xgbclassifier__learning_rate": [0.1, 0.01],
            "xgbclassifier__max_depth": [3, 5],
        },
        "scoring": ["roc_auc"],
        "kfold": True,
        "calibrate": False,
        "train_size": 0.7,
        "validation_size": 0.15,
        "test_size": 0.15,
        "random_state": 42,
    }


def test_lr_initialization(lr_model_parameters):
    model = Model(**lr_model_parameters)

    assert model.name == lr_model_parameters["name"]
    assert model.estimator_name == lr_model_parameters["estimator_name"]
    assert isinstance(model.estimator, Pipeline)
    assert model.model_type == lr_model_parameters["model_type"]
    assert model.scoring == ["roc_auc"]  # Assuming default scoring if not provided
    assert isinstance(model.grid, (Mapping, Iterable))  # Checks for grid type
    assert model.kfold is False  # Default value
    assert model.calibrate is False  # Default value


def test_xgboost_initialization(xgboost_model_parameters):
    model = Model(**xgboost_model_parameters)

    assert model.name == xgboost_model_parameters["name"]
    assert model.estimator_name == xgboost_model_parameters["estimator_name"]
    assert isinstance(model.estimator, Pipeline)
    assert model.model_type == xgboost_model_parameters["model_type"]
    assert model.grid == xgboost_model_parameters["grid"]
    assert model.scoring == xgboost_model_parameters["scoring"]
    assert model.kfold == xgboost_model_parameters["kfold"]
    assert model.calibrate == xgboost_model_parameters["calibrate"]
    assert model.train_size == xgboost_model_parameters["train_size"]
    assert model.validation_size == xgboost_model_parameters["validation_size"]
    assert model.test_size == xgboost_model_parameters["test_size"]
    assert model.random_state == xgboost_model_parameters["random_state"]


def test_invalid_model_type(lr_model_parameters):
    lr_model_parameters["model_type"] = "unsupported_type"

    with pytest.raises(ValueError):
        Model(**lr_model_parameters)


def test_missing_estimator():
    with pytest.raises(TypeError):
        Model(
            name="model_without_estimator",
            estimator_name="missing",
            model_type="classification",
        )


# Additional edge cases:
def test_invalid_grid_type(lr_model_parameters):
    lr_model_parameters["grid"] = 12345  # Invalid grid type

    with pytest.raises(TypeError):
        Model(**lr_model_parameters)


def test_empty_param_grid(lr_model_parameters):
    lr_model_parameters["grid"] = []

    model = Model(**lr_model_parameters)
    # even though its an empty list still in init
    # set to paramGrid object
    assert isinstance(model.grid, ParameterGrid)  # Should initialize without error


from unittest.mock import MagicMock, patch


@pytest.fixture
def model():
    mock_steps = [
        (StandardScaler()),
        # (SimpleImputer()),
        ("kbest", SelectKBest(k=2)),
        ("classifier", MagicMock()),
    ]
    return Model(
        name="TestModel",
        estimator_name="mock_estimator",
        estimator=MagicMock(),
        pipeline_steps=mock_steps,
        model_type="classification",
        grid=[],
    )


from imblearn.over_sampling import SMOTE


@pytest.fixture
def model_with_sampler():
    mock_steps = [
        (StandardScaler()),
        # (SimpleImputer()),
        ("kbest", SelectKBest(k=2)),
        ("classifier", MagicMock()),
    ]
    return Model(
        name="TestModel",
        estimator_name="mock_estimator",
        estimator=MagicMock(),
        pipeline_steps=mock_steps,
        model_type="classification",
        grid=[],
        imbalance_sampler=SMOTE(),
    )


def test_get_preprocessing_and_feature_selection_pipeline(model):
    pipeline = model.get_preprocessing_and_feature_selection_pipeline()
    expected_steps = [
        ("preprocess_scaler_step_0", model.estimator.steps[0][1]),
        # ("preprocess_imputer_step_0", model.estimator.steps[1][1]),
        ("feature_selection_kbest", model.estimator.steps[1][1]),
    ]
    print(pipeline.steps)
    print(expected_steps)
    assert pipeline.steps == expected_steps
    # MockPipeline.assert_called_once_with(expected_steps)


def test_get_feature_selection_pipeline(model):
    pipeline = model.get_feature_selection_pipeline()
    expected_steps = [("feature_selection_kbest", model.estimator.steps[1][1])]
    print(pipeline.steps)
    print(expected_steps)
    assert pipeline.steps == expected_steps


def test_pipeline_structure(model):
    """Test the structure of the pipeline."""
    pipeline = model.get_preprocessing_pipeline()
    assert isinstance(pipeline, Pipeline), "The output should be a Pipeline instance."

    # Check if the pipeline contains a scaler
    print(pipeline)
    scaler = pipeline.named_steps.get("preprocess_scaler_step_0", None)
    assert isinstance(scaler, StandardScaler), "Pipeline should include a scaler."


def test_pipeline_transform(classification_data, model):
    """Test the pipeline transformation."""
    X, y = classification_data
    pipeline = model.get_preprocessing_pipeline()

    transformed_data = pipeline.fit_transform(X)
    assert transformed_data is not None, "Transformed data should not be None."
    assert isinstance(
        transformed_data, np.ndarray
    ), "Transformed data should be a numpy array."


def test_pipeline_handles_empty_data(model):
    """Test the pipeline with empty data."""
    pipeline = model.get_preprocessing_pipeline()

    empty_data = pd.DataFrame({"numerical": [], "categorical": []})
    with pytest.raises(ValueError):
        pipeline.fit_transform(empty_data)  # standard scaler should raise an error


@pytest.fixture
def imbalanced_data():
    """Fixture for imbalanced data."""
    X_train = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            "feature2": [
                10,
                20,
                30,
                40,
                50,
                60,
                10,
                20,
                30,
                40,
                50,
                60,
                10,
                20,
                30,
                40,
                50,
                60,
            ],
        }
    )
    y_train = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1])
    return X_train, y_train


@pytest.fixture
def balanced_data():
    """Fixture for balanced data."""
    X_train = pd.DataFrame({"feature1": [1, 2, 3, 4], "feature2": [10, 20, 30, 40]})
    y_train = np.array([0, 1, 0, 1])
    return X_train, y_train


@pytest.fixture
def single_class_data():
    """Fixture for single-class data."""
    X_train = pd.DataFrame({"feature1": [1, 2, 3, 4], "feature2": [10, 20, 30, 40]})
    y_train = np.array([0, 0, 0, 0])
    return X_train, y_train


def test_verify_imbalance_sampler_imbalanced_data(
    capfd, imbalanced_data, model_with_sampler
):
    """Test verify_imbalance_sampler with imbalanced data."""
    X_train, y_train = imbalanced_data

    model_with_sampler.verify_imbalance_sampler(X_train, y_train)

    # Capture printed output
    captured = capfd.readouterr()

    assert "Distribution of y values after resampling: 0" in captured.out
    assert "0    12" in captured.out, "Should correctly identify the majority class."
    assert "1    12" in captured.out, "Should print correct count for minority class."


def test__balanced_data(capfd, balanced_data, model_with_sampler):
    """Test verify_imbalance_sampler with already balanced data."""
    X_train, y_train = balanced_data

    model_with_sampler.verify_imbalance_sampler(X_train, y_train)

    # Capture printed output
    captured = capfd.readouterr()

    # Validate that the data remains unchanged
    assert "Distribution of y values after resampling: 0" in captured.out
    assert "0    2" in captured.out, "Should correctly identify the majority class."
    assert "1    2" in captured.out, "Should print correct count for minority class."


def test_verify_imbalance_sampler_single_class_data(
    single_class_data, model_with_sampler
):
    """Test verify_imbalance_sampler with single-class data."""
    X_train, y_train = single_class_data

    with pytest.raises(ValueError):
        model_with_sampler.verify_imbalance_sampler(X_train, y_train)


@pytest.fixture
def calibrated_lr_model():
    """Fixture to create a simple Calibrated Logistic Regression model."""
    lr = LogisticRegression()
    param_grid = {"logistic_regression__C": [0.1, 1, 10]}  # Example param grid

    return Model(
        name="test_model",
        estimator_name="logistic_regression",
        estimator=lr,
        model_type="classification",
        grid=param_grid,
        kfold=False,
        calibrate=True,
        calibration_method="sigmoid",
    )


@pytest.fixture
def calibrated_kfold_lr_model():
    """Fixture to create a simple Calibrated Logistic Regression model."""
    lr = LogisticRegression()
    param_grid = {"logistic_regression__C": [0.1, 1, 10]}  # Example param grid

    return Model(
        name="test_model",
        estimator_name="logistic_regression",
        estimator=lr,
        model_type="classification",
        grid=param_grid,
        kfold=True,
        calibrate=True,
        calibration_method="sigmoid",
    )


from sklearn.calibration import CalibratedClassifierCV


def test_calibrate_model_default_method(calibrated_lr_model, classification_data):
    """Test that calibrateModel works with the default method."""
    X, y = classification_data

    calibration_methods = [
        "sigmoid",
        "isotonic",
    ]

    for cab_method in calibration_methods:
        model = calibrated_lr_model

        # define cab method
        model.calibration_method = cab_method

        # first need to do gridsearch
        model.grid_search_param_tuning(X, y)

        # Fit the model
        model.fit(X, y)

        # calibrate model
        model.calibrateModel(X, y)

        # Ensure the calibrated model is an instance of CalibratedClassifierCV
        assert isinstance(
            model.estimator, CalibratedClassifierCV
        ), "Expected a CalibratedClassifierCV instance."

        # Ensure predictions are probabilistic
        probabilities = model.predict_proba(X)
        assert np.allclose(
            probabilities.sum(axis=1), 1
        ), "Probabilities should sum to 1 for each instance."


def test_calibrate_kfold_model_default_method(
    calibrated_kfold_lr_model, classification_data
):
    """Test that calibrateModel works with the default method."""
    X, y = classification_data

    calibration_methods = [
        "sigmoid",
        "isotonic",
    ]

    for cab_method in calibration_methods:
        model = type(calibrated_kfold_lr_model)(
            name="test_model",
            estimator_name="logistic_regression",
            grid={"logistic_regression__C": [0.1, 1, 10]},
            estimator=LogisticRegression(),  # Or provide a pre-initialized estimator if needed
            model_type="classification",
            kfold=True,
            calibrate=True,
            calibration_method="sigmoid",
        )

        # define cab method
        model.calibration_method = cab_method

        # first need to do gridsearch
        model.grid_search_param_tuning(X, y)

        # Fit the model
        model.fit(X, y)

        # calibrate model
        model.calibrateModel(X, y)

        # Ensure the calibrated model is an instance of CalibratedClassifierCV
        assert isinstance(
            model.estimator, CalibratedClassifierCV
        ), "Expected a CalibratedClassifierCV instance."

        # Ensure predictions are probabilistic
        probabilities = model.predict_proba(X)
        assert np.allclose(
            probabilities.sum(axis=1), 1
        ), "Probabilities should sum to 1 for each instance."


def test_calibrate_model_invalid_method(calibrated_lr_model, classification_data):
    """Test calibrateModel with an unsupported method."""

    X, y = classification_data

    model = calibrated_lr_model

    # define cab method
    model.calibration_method = "unsupported method"

    # first need to do gridsearch
    model.grid_search_param_tuning(X, y)

    # Fit the model
    model.fit(X, y)

    with pytest.raises(Exception):
        # calibrate model
        model.calibrateModel(X, y)


def test_calibrate_model_edge_probabilities(calibrated_lr_model, classification_data):
    """Test calibrateModel with edge probabilities."""
    X, y = classification_data

    model = calibrated_lr_model

    # first need to do gridsearch
    model.grid_search_param_tuning(X, y)

    # Fit the model
    model.fit(X, y)

    # calibrate model
    model.calibrateModel(X, y)

    # Simulate predictions near 0 and 1
    probabilities = model.predict_proba(X)
    assert np.all(
        (probabilities >= 0) & (probabilities <= 1)
    ), "Probabilities should be within [0, 1]."


def test_rfe_calibrate_model(classification_data):
    rfe_estimator = ElasticNet()

    rfe = RFE(rfe_estimator)

    estimator_name = "RF"
    tuned_parameters = {
        estimator_name + "__max_depth": [2, 30],
    }
    estimator = RandomForestClassifier(n_estimators=10)
    X, y = classification_data
    model = Model(
        name="test_model",
        estimator_name=estimator_name,
        pipeline_steps=[SimpleImputer(), ("rfe", rfe)],
        estimator=estimator,
        scoring=["roc_auc"],
        grid=tuned_parameters,
        model_type="classification",
        calibrate=True,
        feature_selection=True,
    )
    model.grid_search_param_tuning(X, y)

    X_train, y_train = model.get_train_data(X, y)
    X_test, y_test = model.get_test_data(X, y)
    model.fit(X_train, y_train, score="roc_auc")

    if model.calibrate:
        model.calibrateModel(
            X,
            y,
            score="roc_auc",
        )

    print("Validation Metrics")
    model.return_metrics(X_test, y_test)

    assert hasattr(model.estimator, "predict")

def test_print_selected_best_features_with_dataframe(model):
    # Mock the feature selection pipeline and its get_support method
    model.get_feature_selection_pipeline = MagicMock()
    mock_pipeline = MagicMock()
    mock_pipeline.get_support.return_value = [True, False, True]
    model.get_feature_selection_pipeline.return_value = [mock_pipeline]

    # Create a sample DataFrame
    X = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'feature3': [7, 8, 9]
    })

    # Call the method and capture the output
    selected_features = model.print_selected_best_features(X)

    # Assert the correct features are selected
    assert selected_features == ['feature1', 'feature3']

def test_print_selected_best_features_with_array(model):
    # Mock the feature selection pipeline and its get_support method
    model.get_feature_selection_pipeline = MagicMock()
    mock_pipeline = MagicMock()
    mock_pipeline.get_support.return_value = [True, False, True]
    model.get_feature_selection_pipeline.return_value = [mock_pipeline]

    # Create a sample array-like input
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    # Call the method and capture the output
    selected_features = model.print_selected_best_features(X)

    # Assert the correct features are selected
    assert selected_features == [True, False, True]
