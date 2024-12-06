import pytest
from model_tuner import Model
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

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
