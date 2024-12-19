import pytest
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
        model_type="classification",
        estimator=estimator,
        scoring=["roc_auc"],
        grid=tuned_parameters,
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
        model_type="classification",
        estimator=estimator,
        scoring=["roc_auc"],
        grid=tuned_parameters,
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
        model_type="classification",
        estimator=estimator,
        scoring=["roc_auc"],
        grid=tuned_parameters,
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
        model_type="classification",
        scoring=["accuracy"],
        grid=tuned_parameters,
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
        model_type="classification",
        scoring=["accuracy"],
        grid=tuned_parameters,
    )
    model.grid_search_param_tuning(X, y)
    X_train, y_train = model.get_train_data(X, y)
    model.fit(X_train, y_train, score="accuracy")

    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1)


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
        model_type="classification",
        grid=tuned_parameters,
        scoring=["accuracy"],
        boost_early=True,
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
    metrics = model.return_metrics(X_test, y_test)

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
    metrics = model.return_metrics(X_test, y_test)

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
