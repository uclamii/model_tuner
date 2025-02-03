from copy import deepcopy
import pytest
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from model_tuner import Model, report_model_metrics
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import RFE
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, StratifiedKFold
from model_tuner.model_tuner_utils import kfold_split
from model_tuner.model_tuner_utils import train_val_test_split, report_model_metrics


@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    y = pd.Series(y)
    return X, y


@pytest.fixture
def multi_classification_data():
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=10,
        n_redundant=0,
        random_state=42,
        n_classes=4,
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    y = pd.Series(y, name="labels")
    return X, y


@pytest.fixture
def multiclass_data():
    X, y = make_classification(
        n_samples=100, n_features=10, n_classes=3, n_informative=5, random_state=42
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    y = pd.Series(y)
    return X, y


@pytest.fixture
def classification_data_large():
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

    # testing with and without optimal_threshold
    for optimal_threshold in [False, True]:
        # testing with different options of scoring
        for score in [None, "accuracy"]:
            predictions = model.predict(
                X, score=score, optimal_threshold=optimal_threshold
            )
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


def test_predict_proba_method_kfold(classification_data):
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
        kfold=True,
    )

    model.grid_search_param_tuning(X, y)
    model.fit(X, y, score="accuracy")

    # with and without y
    for y_test in [None, y]:
        proba = model.predict_proba(X, y=y_test)
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
    metrics = model.return_metrics(
        X_test,
        y_test,
        return_dict=True,
    )

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


def test_return_metrics_multilabel_classification(multi_classification_data):
    """
    Test the return_metrics method for a classification model.
    """
    X, y = multi_classification_data
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    estimator_name = "RF"
    tuned_parameters = {
        estimator_name + "__max_depth": [2, 30],
    }

    # Initialize the classification model
    model = Model(
        name="test_classification_model",
        estimator_name="RF",
        model_type="classification",
        estimator=RandomForestClassifier(n_estimators=10),
        multi_label=True,
        class_labels=[str(x) for x in sorted(list(set(y)))],
        scoring=["roc_auc_ovr"],
        grid=tuned_parameters,
    )

    # Perform grid search to populate `best_params_per_score`
    model.grid_search_param_tuning(X_train, y_train)

    # Train the model
    model.fit(X_train, y_train)

    # Test return_metrics method
    metrics = model.return_metrics(
        X_test, y_test, model_metrics=False, return_dict=True
    )
    print(metrics)

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


def test_model_metrics_True_classification(classification_data):
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
    metrics = model.return_metrics(X_test, y_test, model_metrics=True)
    print(metrics)

    # Validate the structure and content of the metrics
    assert isinstance(metrics, dict), "Expected return_metrics to return a dictionary."
    assert "Brier Score" in metrics, "Brier Score is missing."
    assert "Precision/PPV" in metrics, "Precision/PPV is missing."


def test_model_optimal_threshold_True_classification(classification_data):
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
    model.grid_search_param_tuning(X_train, y_train, f1_beta_tune=True)

    # Train the model
    model.fit(X_train, y_train)

    # Test return_metrics method
    metrics = model.return_metrics(
        X_test,
        y_test,
        model_metrics=True,
        optimal_threshold=True,
    )
    print(metrics)

    # Validate the structure and content of the metrics
    assert isinstance(metrics, dict), "Expected return_metrics to return a dictionary."
    assert "Brier Score" in metrics, "Brier Score is missing."
    assert "Precision/PPV" in metrics, "Precision/PPV is missing."


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


def test_return_metrics_classification_feature_select(classification_data):
    """
    Test the return_metrics method for a classification model.
    """
    X, y = classification_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    rfe_estimator = ElasticNet()

    rfe = RFE(rfe_estimator)

    # Initialize the classification model
    model = Model(
        name="test_classification_model",
        estimator_name="lr",
        pipeline_steps=[rfe],
        model_type="classification",
        estimator=LogisticRegression(),
        scoring=["roc_auc"],
        grid={"lr__C": [0.01, 0.1, 1]},
        feature_selection=True,
    )

    # Perform grid search to populate `best_params_per_score`
    model.grid_search_param_tuning(X_train, y_train)

    # Train the model
    model.fit(X_train, y_train)

    # Test return_metrics method
    metrics = model.return_metrics(
        X_test,
        y_test,
        return_dict=True,
    )

    assert any(
        isinstance(step[1], RFE) for step in model.pipeline_steps
    ), "RFE not found in the pipeline steps!"

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


def test_bootstrapped_metrics_consistency_regression(regression_data):
    """
    Test the consistency of bootstrapped metrics for a regression model.
    """
    X, y = regression_data

    ## Reuse model initialization from test_imbalance_sampler_integration
    name = "test_model"
    estimator_name = "lr"
    estimator = LinearRegression()
    tuned_parameters = {
        estimator_name + "__fit_intercept": [True, False],
    }

    model = Model(
        name=name,
        estimator_name=estimator_name,
        model_type="regression",
        estimator=estimator,
        scoring=["r2"],
        grid=tuned_parameters,
    )

    ## Assert model is initialized correctly
    assert model.name == name
    assert model.estimator_name == estimator_name
    assert isinstance(model.estimator.named_steps[estimator_name], LinearRegression)

    # Perform grid search to populate `best_params_per_score`
    model.grid_search_param_tuning(X, y)

    # Fit the model
    model.fit(X, y)

    # Define a simple metric set
    metrics = ["r2", "neg_mean_squared_error"]

    # Obtain bootstrap metrics
    bootstrap_results = model.return_bootstrap_metrics(X, y, metrics=metrics)

    print(bootstrap_results)

    # Check the type and content
    assert isinstance(
        bootstrap_results, pd.DataFrame
    ), "Bootstrap metrics should be returned as a pandas dataframe."
    assert all(
        metric in bootstrap_results["Metric"].tolist() for metric in metrics
    ), "All specified metrics should be in the result."
    assert all(
        col in bootstrap_results.columns
        for col in [
            "Mean",
            "95% CI Lower",
            "95% CI Upper",
        ]
    ), "Each metric should contain these results Mean, 95% CI Lower, 95% CI Upper."


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


@pytest.fixture
def initialized_kfold_lr_model_multiple_scores():
    lr = LogisticRegression()
    param_grid = {"logistic_regression__C": [0.1, 1, 10]}  # Example param grid

    return Model(
        name="test_model",
        estimator_name="logistic_regression",
        estimator=lr,
        model_type="classification",
        grid=param_grid,
        kfold=True,
        scoring=["roc_auc", "f1"],
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


def test_fit_basic_kfold(
    initialized_kfold_lr_model,
    initialized_kfold_lr_model_multiple_scores,
    classification_data,
):
    X, y = classification_data
    for model in [
        initialized_kfold_lr_model,
        initialized_kfold_lr_model_multiple_scores,
    ]:

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


@pytest.fixture
def initialized_xgb_model_multiple_scores():
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
        scoring=["roc_auc", "f1"],
    )


def test_fit_with_early_stopping_and_validation(
    initialized_xgb_model,
    initialized_xgb_model_multiple_scores,
    classification_data,
):
    X, y = classification_data
    for model in [initialized_xgb_model, initialized_xgb_model_multiple_scores]:

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


def test_catboost_earlystop_best_iteration(initialized_model, classification_data):
    X, y = classification_data
    model = initialized_model

    # Create a small validation set
    X_validation, y_validation = X.iloc[:10], y.iloc[:10]

    estimator = CatBoostClassifier(verbose=0)
    estimator_name = "cat"

    tuned_parameters = {
        f"{estimator_name}__depth": [10],
        f"{estimator_name}__learning_rate": [1e-4],
        f"{estimator_name}__n_estimators": [1],
        f"{estimator_name}__early_stopping_rounds": [10],
        f"{estimator_name}__verbose": [0],
        f"{estimator_name}__eval_metric": ["Logloss"],
    }

    model = Model(
        name="Catboost Early",
        model_type="classification",
        estimator_name=estimator_name,
        estimator=estimator,
        pipeline_steps=[],
        stratify_y=True,
        grid=tuned_parameters,
        randomized_grid=False,
        n_iter=4,
        boost_early=True,
        scoring=["roc_auc"],
        n_jobs=-2,
        random_state=42,
    )

    model.grid_search_param_tuning(X, y)
    X_train, y_train = model.get_train_data(X, y)
    # Fit the model with validation data

    # assert (
    #     model.best_params_per_score["roc_auc"]["params"]["cat__n_estimators"] == 1
    # ), "Number of n_estimator should be 1"

    model.fit(X, y, validation_data=(X_validation, y_validation))

    # Check if model remained consistent
    assert hasattr(
        model.estimator, "steps"
    ), "Model should be fitted even when validation data is used."


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

        # Ensure the calibrated model is an instance of CalibratedClassifierCV
        # Assert that the number of classes in the classification report matches
        # the size of the confusion matrix

        model.return_metrics(X, y)

        # print(model.classification_report)
        # print(model.conf_mat)

        assert (
            model.classification_report["macro avg"]["support"] == model.conf_mat.sum()
        ), (
            f"Mismatch in the number of classes: "
            f"{model.classification_report['macro avg']['support']} samples in classification_report, "
            f"{model.conf_mat.sum()} in conf_mat."
        )

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


def test_kfold_split_stratified():
    """
    Test that kfold_split returns a StratifiedKFold when stratify=True.
    """
    splitter = kfold_split(
        classifier=None,
        X=None,
        y=None,
        stratify=True,
        scoring=["roc_auc"],
        n_splits=5,
        random_state=42,
    )

    assert isinstance(splitter, StratifiedKFold), "Expected a StratifiedKFold instance"
    assert splitter.n_splits == 5, "Expected 5 splits"
    assert splitter.random_state == 42, "Expected a random state of 42"


def test_kfold_split_kfold():
    """
    Test that kfold_split returns a KFold when stratify=False.
    """
    splitter = kfold_split(
        classifier=None,
        X=None,
        y=None,
        stratify=False,
        scoring=["roc_auc"],
        n_splits=5,
        random_state=42,
    )

    assert isinstance(splitter, KFold), "Expected a KFold instance"
    assert splitter.n_splits == 5, "Expected 5 splits"
    assert splitter.random_state == 42, "Expected a random state of 42"


from skopt.space import Real, Integer


def test_grid_search_param_tuning_bayesian(classification_data):
    """
    Test that the Model class correctly performs a Bayesian hyperparameter search
    using skopt's BayesSearchCV backend.
    """
    X, y = classification_data
    estimator_name = "xgb"

    tuned_parameters = {
        "bayes__n_iter": 5,  # Number of iterations for Bayesian optimization
        f"{estimator_name}__max_depth": Integer(1, 4),
        f"{estimator_name}__learning_rate": Real(1e-4, 1e-2, prior="log-uniform"),
        f"{estimator_name}__n_estimators": Integer(20, 50),
        f"{estimator_name}__verbosity": [0],
    }

    model = Model(
        name="bayes_model",
        estimator_name=estimator_name,
        estimator=XGBClassifier(eval_metric="logloss"),
        grid=tuned_parameters,
        model_type="classification",
        scoring=["accuracy"],
        bayesian=True,
        kfold=True,
        random_state=42,
    )

    model.grid_search_param_tuning(X, y)

    assert (
        "accuracy" in model.best_params_per_score
    ), "Bayesian search did not populate best_params_per_score for 'accuracy'."

    best_params = model.best_params_per_score["accuracy"]["params"]
    assert best_params, "Bayesian search returned an empty best_params dictionary."

    # Optionally, fit the best model and check if it runs
    model.fit(X, y, score="accuracy")

    # Basic sanity check: we can now predict, and it shouldn't error
    predictions = model.predict(X)
    assert len(predictions) == len(y), "Predictions do not match number of samples."
    assert set(predictions).issubset({0, 1}), "Predictions are outside expected labels."

    print("Best Bayesian Hyperparameters Found:", best_params)


@pytest.mark.parametrize("n_iter_value", [1, 3])
def test_grid_search_param_tuning_randomized_kfold(classification_data, n_iter_value):
    """
    Test that setting `randomized_grid=True` in the Model class
    leads to using RandomizedSearchCV.
    """
    # Unpack your classification fixture
    X, y = classification_data

    # Define some parameter grid
    # In random search, these parameters become discrete choices (or distributions).
    estimator_name = "rf"
    tuned_parameters = {
        f"{estimator_name}__n_estimators": [10, 50],
        f"{estimator_name}__max_depth": [1, 3, 5],
    }

    # Instantiate the Model with `randomized_grid=True`
    model = Model(
        name="randomized_model",
        estimator_name=estimator_name,
        estimator=RandomForestClassifier(random_state=42),
        grid=tuned_parameters,
        scoring=["accuracy"],
        kfold=True,
        n_splits=4,
        model_type="classification",
        randomized_grid=True,  # <--- This triggers RandomizedSearchCV in your code
        n_iter=n_iter_value,  # <--- Number of random samples
        random_state=42,  # <--- For reproducibility
    )

    model.grid_search_param_tuning(X, y)
    # Check that the model indeed recorded best_params_per_score
    assert (
        "accuracy" in model.best_params_per_score
    ), "Best params were not populated under the 'accuracy' key."
    best_params = model.best_params_per_score["accuracy"]
    assert best_params, "No best_params found after randomized grid search."

    # Fit final model with these params to ensure the pipeline can train
    model.fit(X, y, score="accuracy")

    # Make sure predictions come out
    predictions = model.predict(X)
    assert len(predictions) == len(y), "Prediction length mismatch."
    print(f"Test passed with n_iter={n_iter_value}. Best Params: {best_params}")


@pytest.mark.parametrize("n_iter_value", [1, 3])
def test_grid_search_param_tuning_randomized(classification_data, n_iter_value):
    """
    Test that setting `randomized_grid=True` in the Model class
    leads to using RandomizedSearchCV.
    """
    # Unpack your classification fixture
    X, y = classification_data

    # Define some parameter grid
    # In random search, these parameters become discrete choices (or distributions).
    estimator_name = "rf"
    tuned_parameters = {
        f"{estimator_name}__n_estimators": [10, 50],
        f"{estimator_name}__max_depth": [1, 3, 5],
    }

    # Instantiate the Model with `randomized_grid=True`
    model = Model(
        name="randomized_model",
        estimator_name=estimator_name,
        estimator=RandomForestClassifier(random_state=42),
        grid=tuned_parameters,
        scoring=["accuracy"],
        kfold=False,
        model_type="classification",
        randomized_grid=True,
        n_iter=n_iter_value,
        random_state=42,
    )

    model.grid_search_param_tuning(X, y)
    # Check that the model indeed recorded best_params_per_score
    assert (
        "accuracy" in model.best_params_per_score
    ), "Best params were not populated under the 'accuracy' key."
    best_params = model.best_params_per_score["accuracy"]
    assert best_params, "No best_params found after randomized grid search."

    # Fit final model with these params to ensure the pipeline can train
    model.fit(X, y, score="accuracy")

    # Make sure predictions come out
    predictions = model.predict(X)
    assert len(predictions) == len(y), "Prediction length mismatch."
    print(f"Test passed with n_iter={n_iter_value}. Best Params: {best_params}")


def test_print_selected_best_features_with_dataframe(model):
    # Mock the feature selection pipeline and its get_support method
    model.get_feature_selection_pipeline = MagicMock()
    mock_pipeline = MagicMock()
    mock_pipeline.get_support.return_value = [True, False, True]
    model.get_feature_selection_pipeline.return_value = [mock_pipeline]

    # Create a sample DataFrame
    X = pd.DataFrame(
        {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "feature3": [7, 8, 9]}
    )

    # Call the method and capture the output
    selected_features = model.print_selected_best_features(X)

    # Assert the correct features are selected
    assert selected_features == ["feature1", "feature3"]


def test_print_selected_best_features_with_array(model):
    # Mock the feature selection pipeline and its get_support method
    model.get_feature_selection_pipeline = MagicMock()
    mock_pipeline = MagicMock()
    mock_pipeline.get_support.return_value = [True, False, True]
    model.get_feature_selection_pipeline.return_value = [mock_pipeline]

    # Create a sample array-like input
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Call the method and capture the output
    selected_features = model.print_selected_best_features(X)

    # Assert the correct features are selected
    assert selected_features == [True, False, True]


from sklearn.metrics import confusion_matrix, fbeta_score


def test_tune_threshold_Fbeta_basic(model):
    y_valid = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_valid_proba = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.6, 0.3, 0.9])
    betas = [1, 2]

    with patch("sklearn.metrics.confusion_matrix") as mock_confusion_matrix, patch(
        "sklearn.metrics.fbeta_score"
    ) as mock_fbeta_score:

        mock_confusion_matrix.side_effect = lambda y_true, y_pred: confusion_matrix(
            y_true, y_pred
        )
        mock_fbeta_score.side_effect = lambda y_true, y_pred, beta: fbeta_score(
            y_true, y_pred, beta=beta
        )

        model.tune_threshold_Fbeta(
            score="f1",
            y_valid=y_valid,
            betas=betas,
            y_valid_proba=y_valid_proba,
            kfold=False,
        )

        assert model.threshold["f1"] is not None


def test_tune_threshold_Fbeta_kfold(initialized_kfold_lr_model):
    y_valid = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_valid_proba = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.6, 0.3, 0.9])
    betas = [1, 2]

    with patch("sklearn.metrics.confusion_matrix") as mock_confusion_matrix, patch(
        "sklearn.metrics.fbeta_score"
    ) as mock_fbeta_score:

        mock_confusion_matrix.side_effect = lambda y_true, y_pred: confusion_matrix(
            y_true, y_pred
        )
        mock_fbeta_score.side_effect = lambda y_true, y_pred, beta: fbeta_score(
            y_true, y_pred, beta=beta
        )

        best_threshold = initialized_kfold_lr_model.tune_threshold_Fbeta(
            score="f1",
            y_valid=y_valid,
            betas=betas,
            y_valid_proba=y_valid_proba,
            kfold=True,
        )

        assert best_threshold is not None


def test_tune_threshold_Fbeta_invalid_input(model):
    y_valid = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_valid_proba = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.6, 0.3, 0.9])
    betas = ["hello", "2"]

    with pytest.raises(ValueError):
        model.tune_threshold_Fbeta(
            score="f1",
            y_valid=y_valid,
            betas=betas,
            y_valid_proba=y_valid_proba,
        )


def test_conf_mat_class_kfold(initialized_kfold_lr_model, classification_data):
    X, y = classification_data

    initialized_kfold_lr_model.grid_search_param_tuning(X, y)
    initialized_kfold_lr_model.fit(X, y)

    test_model = deepcopy(initialized_kfold_lr_model)

    # testing dataframe and numpy array cases
    for X_test, y_test in [(X, y), (X.values, y.values)]:
        result = initialized_kfold_lr_model.conf_mat_class_kfold(
            X.values, y.values, test_model
        )

        # k=10 , total sum results in 48, 2 , 2, 48
        # since the data total X is 100 samples
        expected_conf_mat = np.array([[48, 2], [2, 48]])

        assert result is not None

        assert np.array_equal(result["Confusion Matrix"], expected_conf_mat)


def test_group_kfold_no_group_overlap():
    """
    Test that when using GroupKFold, no group is shared between
    the train and validation (test) splits.
    """

    # --- 1. Create some synthetic data with groups ---
    # For demonstration, let's make 100 rows, with 5 distinct groups of 20 rows each.
    n_samples = 100
    n_features = 10
    n_groups = 5
    group_size = n_samples // n_groups  # 20

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=5,
        random_state=42,
    )

    # Convert X, y to pandas for convenience
    X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])
    y = pd.Series(y)

    # Create a column "groups" with 5 repeated values [0,0,0... 1,1,1... 4,4,4]
    groups = np.repeat(range(n_groups), group_size)
    np.random.seed(42)
    np.random.shuffle(groups)  # shuffle groups so they're not contiguous
    # But it’s fine to leave them unshuffled if you prefer

    # --- 2. Instantiate the Model, specifying GroupKFold usage ---
    from sklearn.linear_model import LogisticRegression
    from model_tuner.model_tuner_utils import Model

    estimator_name = "lr"
    param_grid = {f"{estimator_name}__C": [0.1, 1]}  # minimal grid for speed

    # Create the Model with kfold=True and pass kfold_group=groups
    model = Model(
        name="test_group_kfold_model",
        estimator_name=estimator_name,
        estimator=LogisticRegression(),
        model_type="classification",
        kfold=True,
        kfold_group=groups,  # <--- pass the group array
        grid=param_grid,
        n_splits=3,  # e.g., use 3 folds
        random_state=42,
        scoring=["roc_auc"],
    )

    # --- 3. Run grid search and fit the model ---
    model.grid_search_param_tuning(X, y)  # populates best_params_per_score
    model.fit(X, y)  # fit on the entire data

    # --- 4. Verify that no group appears in both train and test for each fold ---
    # By design, Model’s internal cross-validation uses GroupKFold when kfold_group is set.
    # The model should have stored the splitter in `model.kf`.

    # We can re-check by iterating over model.kf manually:
    for train_idx, test_idx in model.kf.split(X, y, groups=groups):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups.intersection(test_groups)

        # If GroupKFold is working correctly, overlap should be empty
        assert len(overlap) == 0, (
            f"Groups {overlap} appear in both train and test. "
            "GroupKFold is not splitting correctly."
        )

    # If we get here without an assertion, the test passes.
    print("GroupKFold test passed: No group overlap between train and test folds!")


def test_compare_report_model_metrics_vs_model_classification_report(
    classification_data,
):
    """
    Compare the classification metrics from `report_model_metrics` DataFrame
    with the `Model` object's `self.classification_report` generated by
    `model.return_metrics()`.
    """
    X, y = classification_data

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Initialize and train the model
    model = Model(
        name="compare_test",
        estimator_name="logreg",
        estimator=LogisticRegression(),
        model_type="classification",
        grid={},  # Not doing hyperparam tuning for simplicity
        scoring=["accuracy"],  # Simplify
        class_labels=["0", "1"],  # Let’s define these for the classification report
    )

    model.grid_search_param_tuning(X_train, y_train)

    model.fit(X_train, y_train)

    # 1) Call return_metrics => populates model.classification_report internally
    model.return_metrics(X_test, y_test, return_dict=False)

    # 2) Call report_model_metrics => returns a DataFrame of metrics
    metrics_df = report_model_metrics(
        model,
        X_valid=X_test,
        y_valid=y_test,
        print_results=False,
    )
    # -------------------------------------------------------------------------
    model_positive_precision = model.classification_report["1"]["precision"]
    model_positive_recall = model.classification_report["1"]["recall"]

    precision_row = metrics_df.loc[metrics_df["Metric"] == "Precision/PPV"]
    assert (
        not precision_row.empty
    ), "No row named 'Precision/PPV' in report_model_metrics DataFrame."

    reported_precision = precision_row["Value"].values[0]

    sensitivity_row = metrics_df.loc[metrics_df["Metric"] == "Sensitivity"]
    assert (
        not sensitivity_row.empty
    ), "No row named 'Sensitivity' in report_model_metrics DataFrame."

    reported_sensitivity = sensitivity_row["Value"].values[0]

    np.testing.assert_allclose(
        model_positive_precision,
        reported_precision,
        rtol=1e-3,
        atol=1e-6,
        err_msg=(
            f"Mismatch in 'precision' for positive class:\n"
            f"self.classification_report => {model_positive_precision}\n"
            f"report_model_metrics => {reported_precision}"
        ),
    )

    np.testing.assert_allclose(
        model_positive_recall,
        reported_sensitivity,
        rtol=1e-3,
        atol=1e-6,
        err_msg=(
            f"Mismatch in 'recall' for positive class:\n"
            f"self.classification_report => {model_positive_recall}\n"
            f"report_model_metrics => {reported_sensitivity}"
        ),
    )

    print(
        "Precision and recall from Model.classification_report match report_model_metrics DataFrame."
    )


def test_compare_report_model_metrics_vs_model_classification_report_f1_tune(
    classification_data,
):
    """
    Compare the classification metrics from `report_model_metrics` DataFrame
    with the `Model` object's `self.classification_report` generated by
    `model.return_metrics()`.
    """
    X, y = classification_data

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Initialize and train the model
    model = Model(
        name="compare_test",
        estimator_name="logreg",
        estimator=LogisticRegression(),
        model_type="classification",
        grid={},  # Not doing hyperparam tuning for simplicity
        scoring=["accuracy"],  # Simplify
        class_labels=["0", "1"],  # Let’s define these for the classification report
    )

    model.grid_search_param_tuning(X_train, y_train, f1_beta_tune=True)

    model.fit(X_train, y_train)

    # 1) Call return_metrics => populates model.classification_report internally
    model.return_metrics(X_test, y_test, return_dict=False, optimal_threshold=True)

    model.threshold["accuracy"]

    # 2) Call report_model_metrics => returns a DataFrame of metrics
    metrics_df = report_model_metrics(
        model,
        X_valid=X_test,
        y_valid=y_test,
        threshold=model.threshold["accuracy"],
        print_results=False,
    )
    # -------------------------------------------------------------------------
    model_positive_precision = model.classification_report["1"]["precision"]
    model_positive_recall = model.classification_report["1"]["recall"]

    precision_row = metrics_df.loc[metrics_df["Metric"] == "Precision/PPV"]

    assert (
        not precision_row.empty
    ), "No row named 'Precision/PPV' in report_model_metrics DataFrame."

    reported_precision = precision_row["Value"].values[0]

    sensitivity_row = metrics_df.loc[metrics_df["Metric"] == "Sensitivity"]
    assert (
        not sensitivity_row.empty
    ), "No row named 'Sensitivity' in report_model_metrics DataFrame."

    reported_sensitivity = sensitivity_row["Value"].values[0]

    print("***** METRICS DF *****")
    print(metrics_df)
    print("***** CLASS REPORT *****")
    print(model.classification_report)

    np.testing.assert_allclose(
        model_positive_precision,
        reported_precision,
        rtol=1e-3,
        atol=1e-6,
        err_msg=(
            f"Mismatch in 'precision' for positive class:\n"
            f"self.classification_report => {model_positive_precision}\n"
            f"report_model_metrics => {reported_precision}"
        ),
    )

    np.testing.assert_allclose(
        model_positive_recall,
        reported_sensitivity,
        rtol=1e-3,
        atol=1e-6,
        err_msg=(
            f"Mismatch in 'recall' for positive class:\n"
            f"self.classification_report => {model_positive_recall}\n"
            f"report_model_metrics => {reported_sensitivity}"
        ),
    )

    print(
        "Precision and recall from Model.classification_report match report_model_metrics DataFrame."
    )


def test_compare_report_model_metrics_vs_model_classification_report_kfold(
    classification_data_large,
):
    """
    Compare the classification metrics from `report_model_metrics` DataFrame
    with the `Model` object's `self.classification_report` generated by
    `model.return_metrics()`.
    """
    X, y = classification_data_large

    # Initialize and train the model
    model = Model(
        name="compare_test",
        estimator_name="logreg",
        estimator=LogisticRegression(),
        model_type="classification",
        grid={},  # Not doing hyperparam tuning for simplicity
        scoring=["accuracy"],  # Simplify
        class_labels=["0", "1"],
        kfold=True,
    )

    model.grid_search_param_tuning(X, y, f1_beta_tune=True)

    model.fit(X, y)

    # 1) Call return_metrics => populates model.classification_report internally
    model.return_metrics(X, y, return_dict=False, optimal_threshold=True)

    model.threshold["accuracy"]

    # 2) Call report_model_metrics => returns a DataFrame of metrics
    metrics_df = report_model_metrics(
        model,
        X_valid=X,
        y_valid=y,
        threshold=model.threshold["accuracy"],
        print_results=False,
    )
    # -------------------------------------------------------------------------
    model_positive_precision = model.classification_report["1"]["precision"]
    model_positive_recall = model.classification_report["1"]["recall"]

    precision_row = metrics_df.loc[metrics_df["Metric"] == "Precision/PPV"]

    assert (
        not precision_row.empty
    ), "No row named 'Precision/PPV' in report_model_metrics DataFrame."

    reported_precision = precision_row["Value"].values[0]

    sensitivity_row = metrics_df.loc[metrics_df["Metric"] == "Sensitivity"]
    assert (
        not sensitivity_row.empty
    ), "No row named 'Sensitivity' in report_model_metrics DataFrame."

    reported_sensitivity = sensitivity_row["Value"].values[0]

    print("***** METRICS DF *****")
    print(metrics_df)
    print("***** CLASS REPORT *****")
    print(model.classification_report)

    np.testing.assert_allclose(
        model_positive_precision,
        reported_precision,
        rtol=1e-3,
        atol=1e-6,
        err_msg=(
            f"Mismatch in 'precision' for positive class:\n"
            f"self.classification_report => {model_positive_precision}\n"
            f"report_model_metrics => {reported_precision}"
        ),
    )

    np.testing.assert_allclose(
        model_positive_recall,
        reported_sensitivity,
        rtol=1e-3,
        atol=1e-6,
        err_msg=(
            f"Mismatch in 'recall' for positive class:\n"
            f"self.classification_report => {model_positive_recall}\n"
            f"report_model_metrics => {reported_sensitivity}"
        ),
    )

    print(
        "Precision and recall from Model.classification_report match report_model_metrics DataFrame."
    )


def test_calibrate_imbalance_sampler_and_extract(classification_data):

    from imblearn.pipeline import Pipeline

    X, y = classification_data
    lr = LogisticRegression()
    param_grid = {"logistic_regression__C": [0.1, 1, 10]}

    calibrate_imb_model = Model(
        name="calibrate_imb_test_model",
        estimator_name="logistic_regression",
        estimator=lr,
        model_type="classification",
        grid=param_grid,
        kfold=False,
        calibrate=True,
        calibration_method="sigmoid",
        imbalance_sampler=SMOTE(),
    )

    assert calibrate_imb_model.calibrate == True

    calibrate_imb_model.grid_search_param_tuning(X, y)

    X_train, y_train = calibrate_imb_model.get_train_data(X, y)
    X_valid, y_valid = calibrate_imb_model.get_valid_data(X, y)
    X_test, y_test = calibrate_imb_model.get_test_data(X, y)

    calibrate_imb_model.fit(X_train, y_train)

    calibrate_imb_model.calibrateModel(X, y)

    assert isinstance(
        calibrate_imb_model.estimator, CalibratedClassifierCV
    ), "Expected a CalibratedClassifierCV instance."

    probabilities = calibrate_imb_model.predict_proba(X)
    assert np.allclose(
        probabilities.sum(axis=1), 1
    ), "Probabilities should sum to 1 for each instance."

    preproc_feat_select = (
        calibrate_imb_model.get_preprocessing_and_feature_selection_pipeline()
    )
    preproc_pipe = calibrate_imb_model.get_preprocessing_pipeline()
    feat_select = calibrate_imb_model.get_feature_selection_pipeline()

    assert isinstance(
        preproc_feat_select, Pipeline
    ), "Expected a imbalanced learn Pipeline instance"
    assert isinstance(
        preproc_pipe, Pipeline
    ), "Expected a imbalanced learn Pipeline instance"
    assert isinstance(
        feat_select, Pipeline
    ), "Expected a imbalanced learn Pipeline instance"


def test_model_with_column_transformer(classification_data):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    X, y = classification_data

    preprocessor = ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), slice(0, 5)),  # Scale first 5 features
            ("imputer", SimpleImputer(), slice(5, 10)),  # Impute last 5 features
        ]
    )

    model = Model(
        name="column_transformer_model",
        estimator_name="lr",
        estimator=LogisticRegression(),
        model_type="classification",
        grid={"lr__C": [0.1, 1, 10]},
        pipeline_steps=[("ct", preprocessor)],
        scoring=["accuracy"],
        random_state=42,
    )

    assert "preprocess_column_transformer_ct" in model.estimator.named_steps
    assert isinstance(
        model.estimator.named_steps["preprocess_column_transformer_ct"],
        ColumnTransformer,
    )

    model.grid_search_param_tuning(X, y)
    model.fit(X, y)

    # Verify predictions
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})


def test_calculate_metrics_classification(classification_data):
    X, y = classification_data
    model = Model(
        name="test_model",
        estimator_name="lr",
        grid={"lr__C": [0.1, 1, 10]},
        estimator=LogisticRegression(),
        model_type="classification",
    )

    model.grid_search_param_tuning(X, y)
    X_train, y_train = model.get_train_data(X, y)
    X_test, y_test = model.get_test_data(X, y)
    X_valid, y_valid = model.get_valid_data(X, y)

    model.fit(X_train, y_train)

    metrics_df = report_model_metrics(model, X_valid=X, y_valid=y, print_results=False)
    assert isinstance(metrics_df, pd.DataFrame)
    assert not metrics_df.empty
    expected_metrics = {"Precision/PPV", "AUC ROC", "Sensitivity", "Specificity"}
    assert expected_metrics.issubset(set(metrics_df["Metric"]))


def test_calculate_metrics_regression(regression_data):
    X, y = regression_data
    model = Model(
        name="kfold_regression_test",
        estimator_name="linreg",
        estimator=LinearRegression(),
        model_type="regression",
        grid={"linreg__fit_intercept": [True, False]},
        kfold=True,
        n_splits=5,
        random_state=42,
        scoring=["r2"],
    )
    model.grid_search_param_tuning(X, y)
    model.fit(X, y)
    metrics_df = report_model_metrics(model, X_valid=X, y_valid=y, print_results=False)
    assert isinstance(metrics_df, pd.DataFrame)
    assert not metrics_df.empty
    expected_metrics = {
        "Mean Absolute Error (MAE)",
        "Mean Squared Error (MSE)",
        "Root Mean Squared Error (RMSE)",
        "R2 Score",
        "Explained Variance",
    }
    assert expected_metrics.issubset(set(metrics_df["Metric"]))


def test_calculate_metrics_multiclass(multiclass_data):
    X, y = multiclass_data
    tuned_parameters = {
        # "depth": [4, 6, 8],  # Example hyperparameters for CatBoost
        "catboost__learning_rate": [0.01, 0.1],
        "catboost__iterations": [100, 200],
    }
    model = Model(
        name="test_multiclass_model",
        estimator_name="catboost",
        estimator=CatBoostClassifier(verbose=0),
        model_type="classification",
        multi_label=True,
        scoring=["roc_auc_ovr"],
        class_labels=[
            "0",
            "1",
            "2",
        ],  # Let’s define these for the classification report
        grid=tuned_parameters,  # Provide the grid
    )
    model.grid_search_param_tuning(
        X,
        y,
    )
    model.fit(X, y)
    metrics_df = report_model_metrics(model, X_valid=X, y_valid=y, print_results=False)
    assert isinstance(metrics_df, pd.DataFrame)
    assert not metrics_df.empty
    expected_metrics = {"Precision", "Recall", "F1-Score"}
    assert expected_metrics.issubset(set(metrics_df["Metric"]))


def test_regression_report_kfold(regression_data):
    X, y = regression_data

    # Initialize regression model with KFold
    model = Model(
        name="kfold_regression_test",
        estimator_name="linreg",
        estimator=LinearRegression(),
        model_type="regression",
        grid={"linreg__fit_intercept": [True, False]},
        kfold=True,
        n_splits=5,
        random_state=42,
        scoring=["r2"],
    )

    model.grid_search_param_tuning(X, y)
    model.fit(X, y)

    metrics = model.regression_report_kfold(X, y, model.test_model, score="r2")

    assert isinstance(metrics, dict)
    expected_keys = [
        "R2",
        "Explained Variance",
        "Mean Absolute Error",
        "Median Absolute Error",
        "Mean Squared Error",
        "RMSE",
    ]

    for key in expected_keys:
        assert key in metrics
        assert isinstance(metrics[key], float)
        if key == "R2":
            assert -1 <= metrics[key] <= 1
        elif "Error" in key:
            assert metrics[key] >= 0

    assert model.kf.get_n_splits() == 5


def test_grid_search_param_tuning_f1_beta_tune(classification_data):
    X, y = classification_data
    for kfold in [True, False]:
        model = Model(
            name="test_model",
            estimator_name="lr",
            estimator=LogisticRegression(),
            model_type="classification",
            grid={"lr__C": [0.1, 1, 10]},
            scoring=["roc_auc"],
            kfold=kfold,
        )
        model.grid_search_param_tuning(X, y, f1_beta_tune=True)
        assert (
            "roc_auc" in model.best_params_per_score
        ), "Best score for roc_auc should be in results."


def test_grid_search_param_tuning_dataframe(classification_data):
    X, y = classification_data
    for kfold in [True, False]:
        model = Model(
            name="test_model",
            estimator_name="lr",
            estimator=LogisticRegression(),
            model_type="classification",
            grid={"lr__C": [0.1, 1, 10]},
            scoring=["roc_auc"],
            kfold=kfold,
        )
        model.grid_search_param_tuning(X, y)
        assert (
            "roc_auc" in model.best_params_per_score
        ), "Best score for roc_auc should be in results."


def test_grid_search_param_tuning_numpy(classification_data):
    X, y = classification_data
    X = X.values
    y = y.values
    for kfold in [True, False]:
        model = Model(
            name="test_model",
            estimator_name="lr",
            estimator=LogisticRegression(),
            model_type="classification",
            grid={"lr__C": [0.1, 1, 10]},
            scoring=["roc_auc"],
            kfold=kfold,
        )
        model.grid_search_param_tuning(X, y)
        assert (
            "roc_auc" in model.best_params_per_score
        ), "Best score for roc_auc should be in results."


def test_grid_search_param_tuning_dataframe_regression(regression_data):
    X, y = regression_data
    for kfold in [True, False]:
        model = Model(
            name="test_model",
            estimator_name="lr",
            estimator=LinearRegression(),
            model_type="regression",
            grid={"lr__fit_intercept": [True, False]},
            scoring=["r2"],
            kfold=kfold,
        )
        model.grid_search_param_tuning(X, y)
        assert (
            "r2" in model.best_params_per_score
        ), "Best score for r2 should be in results."


def test_grid_search_param_tuning_numpy_regression(regression_data):
    X, y = regression_data
    X = X.values
    y = y.values
    for kfold in [True, False]:
        model = Model(
            name="test_model",
            estimator_name="lr",
            estimator=LinearRegression(),
            model_type="regression",
            grid={"lr__fit_intercept": [True, False]},
            scoring=["r2"],
            kfold=kfold,
        )
        model.grid_search_param_tuning(X, y)
        assert (
            "r2" in model.best_params_per_score
        ), "Best score for r2 should be in results."


def test_get_preprocessing_pipeline_after_calibration(classification_data):
    X, y = classification_data

    # Initialize the model with preprocessing steps and calibration enabled
    model = Model(
        name="calibrated_model",
        estimator_name="lr",
        estimator=LogisticRegression(),
        model_type="classification",
        pipeline_steps=[StandardScaler(), SimpleImputer()],
        grid={"lr__C": [0.1, 1, 10]},
        calibrate=True,
        scoring=["roc_auc"],
        random_state=42,
    )

    model.grid_search_param_tuning(X, y)

    X_train, y_train = model.get_train_data(X, y)
    X_test, y_test = model.get_test_data(X, y)
    X_valid, y_valid = model.get_valid_data(X, y)

    model.fit(X_train, y_train)

    model.calibrateModel(X, y)

    # Attempt to retrieve the preprocessing pipeline
    preproc_pipeline = model.get_preprocessing_pipeline()
    x_transformed = preproc_pipeline.transform(X)
    # Verify the pipeline contains the expected steps
    assert len(preproc_pipeline.named_steps) == 2, "Preprocessing steps mismatch"
    assert "preprocess_scaler_step_0" in preproc_pipeline.named_steps, "Scaler missing"
    assert (
        "preprocess_imputer_step_0" in preproc_pipeline.named_steps
    ), "Imputer missing"


def test_get_feature_selection_pipeline_after_calibration(classification_data):
    X, y = classification_data

    # Initialize model with feature selection and calibration
    model = Model(
        name="calibrated_model_feature_select",
        estimator_name="lr",
        estimator=LogisticRegression(),
        model_type="classification",
        pipeline_steps=[("scaler", StandardScaler()), ("selector", SelectKBest(k=5))],
        grid={"lr__C": [0.1, 1, 10]},
        calibrate=True,
        scoring=["roc_auc"],
        random_state=42,
        imbalance_sampler=SMOTE(),
    )
    model.grid_search_param_tuning(X, y)

    X_train, y_train = model.get_train_data(X, y)
    X_test, y_test = model.get_test_data(X, y)
    X_valid, y_valid = model.get_valid_data(X, y)

    model.fit(X_train, y_train)

    model.calibrateModel(X, y)
    # Retrieve combined preprocessing and feature selection pipeline
    feat_pipeline = model.get_feature_selection_pipeline()
    transformed_x = feat_pipeline.transform(X_test)

    # Check if the selector is correctly identified
    assert len(feat_pipeline.named_steps) == 1, "Feature selection steps mismatch"
    assert "feature_selection_selector" in feat_pipeline.named_steps, "Selector missing"
    assert isinstance(
        feat_pipeline.named_steps["feature_selection_selector"], SelectKBest
    )


def test_combined_preproc_feat_select_after_calibration(classification_data):
    X, y = classification_data

    model = Model(
        name="combined_pipeline_model",
        estimator_name="lr",
        estimator=LogisticRegression(),
        model_type="classification",
        pipeline_steps=[
            ("scaler", StandardScaler()),
            ("imputer", SimpleImputer()),
            ("selector", SelectKBest(k=5)),
        ],
        grid={"lr__C": [0.1, 1, 10]},
        calibrate=True,
        scoring=["roc_auc"],
        random_state=42,
    )

    model.grid_search_param_tuning(X, y)

    X_train, y_train = model.get_train_data(X, y)
    X_test, y_test = model.get_test_data(X, y)
    X_valid, y_valid = model.get_valid_data(X, y)

    model.fit(X, y)

    model.calibrateModel(X_train, y_train)
    # Retrieve combined preprocessing and feature selection pipeline
    combined_pipeline = model.get_preprocessing_and_feature_selection_pipeline()
    transformed_x = combined_pipeline.transform(X_test)
    # Verify all steps are present
    expected_steps = [
        "preprocess_scaler_scaler",
        "preprocess_imputer_imputer",
        "feature_selection_selector",
    ]
    assert (
        list(combined_pipeline.named_steps.keys()) == expected_steps
    ), "Combined steps mismatch"


def test_fit_with_early_stopping_and_specific_score(
    initialized_xgb_model_multiple_scores, classification_data
):
    X, y = classification_data
    model = initialized_xgb_model_multiple_scores
    model.grid_search_param_tuning(X, y)

    # Fit with a specific score
    score = "f1"

    X_train, y_train = model.get_train_data(X, y)
    X_test, y_test = model.get_test_data(X, y)
    X_valid, y_valid = model.get_valid_data(X, y)

    model.fit(X_train, y_train, score=score, validation_data=(X_valid, y_valid))

    # Check that best params for 'f1' are used
    assert "f1" in model.best_params_per_score
    assert model.best_params_per_score[score]["params"] is not None

    # Verify the model is fitted
    assert hasattr(model.estimator, "predict")

    # Check that early stopping parameters are applied
    estimator = model.estimator.named_steps[model.estimator_name]
    assert (
        estimator.n_estimators <= 50
    )  # Initial was 50, should be <= due to early stopping


def test_fit_with_early_stopping_default_score(
    initialized_xgb_model, classification_data
):
    X, y = classification_data
    model = initialized_xgb_model
    model.grid_search_param_tuning(X, y)

    # Fit without specifying a score, default to first in scoring list
    default_score = model.scoring[0]

    X_train, y_train = model.get_train_data(X, y)
    X_test, y_test = model.get_test_data(X, y)
    X_valid, y_valid = model.get_valid_data(X, y)
    model.fit(X_train, y_train, validation_data=(X_valid, y_valid))

    # Check that best params for default score are used
    assert default_score in model.best_params_per_score
    assert model.best_params_per_score[default_score]["params"] is not None

    # Verify the model is fitted
    assert hasattr(model.estimator, "predict")

    # Check that early stopping parameters are applied
    estimator = model.estimator.named_steps[model.estimator_name]
    assert (
        estimator.n_estimators <= 50
    )  # Initial was 50, should be <= due to early stopping
