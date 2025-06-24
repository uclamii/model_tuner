import pytest
import numpy as np
import pandas as pd
from model_tuner import Model
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from src.model_tuner.bootstrapper import (
    check_input_type,
    sampling_method,
    evaluate_bootstrap_metrics,
)


def generate_classification_data(
    n_samples: int = 100,
    n_features: int = 20,
    n_classes: int = 2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic classification data for testing.

    Parameters:
    - n_samples: Number of samples.
    - n_features: Number of features.
    - n_classes: Number of classes.
    - random_state: Random seed for reproducibility.

    Returns:
    A tuple containing a DataFrame of features and a Series of labels.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        random_state=random_state,
    )
    return pd.DataFrame(X), pd.Series(y)


def test_check_input_type() -> None:
    """Test the input type conversion functionality for various inputs."""
    np_array = np.array([1, 2, 3])
    pd_series = pd.Series([1, 2, 3])
    pd_dataframe = pd.DataFrame([1, 2, 3])

    assert isinstance(
        check_input_type(np_array), pd.DataFrame
    ), "Failed to convert np.array to DataFrame"
    assert isinstance(
        check_input_type(pd_series), pd.Series
    ), "Failed to handle pd.Series correctly"
    assert isinstance(
        check_input_type(pd_dataframe), pd.DataFrame
    ), "Failed to handle pd.DataFrame correctly"

    with pytest.raises(ValueError):
        check_input_type("not a numpy or pandas type")


def test_sampling_method() -> None:
    """Test the sampling method for balanced, stratified, and custom proportions."""
    y = pd.Series([0, 1, 1, 0, 1, 0])

    # Test balance
    balanced = sampling_method(y, 6, balance=True)

    assert (
        len(balanced.value_counts()) == 2
    ), "Number of classes are not 2, sampling failed"
    assert all(balanced.value_counts() == 3), "Balanced sampling failed"

    # Test custom proportions
    custom = sampling_method(y, 6, class_proportions={0: 0.5, 1: 0.5})
    assert (
        custom.value_counts()[0] == 3
    ), "Custom proportion sampling for class 0 failed"
    assert (
        custom.value_counts()[1] == 3
    ), "Custom proportion sampling for class 1 failed"

    # Test stratify (raises error if not enough variety)
    with pytest.raises(ValueError):
        sampling_method(y, 6, stratify=False)

    # Test error in custom proportions
    with pytest.raises(ValueError):
        sampling_method(y, 20, class_proportions={0: 0.6, 1: 0.6})


def test_evaluate_bootstrap_metrics_multilabel_thresholds() -> None:
    """Test the evaluate_bootstrap_metrics function with multi-label thresholds."""
    # Generate multi-label data
    n_samples = 200
    n_labels = 3
    np.random.seed(42)

    # Create multi-label target data (3 labels)
    y_multilabel = pd.DataFrame(
        {
            "label_0": np.random.binomial(1, 0.3, n_samples),
            "label_1": np.random.binomial(1, 0.5, n_samples),
            "label_2": np.random.binomial(1, 0.4, n_samples),
        }
    )

    # Create predicted probabilities for each label
    y_pred_prob_multilabel = pd.DataFrame(
        {
            "label_0": np.random.beta(2, 5, n_samples),  # Lower probabilities
            "label_1": np.random.beta(3, 3, n_samples),  # Balanced probabilities
            "label_2": np.random.beta(2, 3, n_samples),  # Moderate probabilities
        }
    )

    # Test with list of thresholds
    custom_thresholds = [0.3, 0.5, 0.4]  # Different threshold per label

    results = evaluate_bootstrap_metrics(
        y_pred_prob=y_pred_prob_multilabel,
        y=y_multilabel,
        thresholds=custom_thresholds,
        n_samples=50,
        num_resamples=20,
        metrics=["roc_auc", "precision", "recall", "f1_weighted"],
        average="macro",
    )

    assert isinstance(results, pd.DataFrame), "The return type should be DataFrame"
    assert "Metric" in results.columns, "Results DataFrame must include 'Metric' column"
    assert len(results) == 4, "Should have 4 metrics evaluated"
    assert all(results["Mean"] >= 0), "All metric means should be non-negative"

    # Test with dictionary of thresholds
    threshold_dict = {"label_0": 0.3, "label_1": 0.5, "label_2": 0.4}

    results_dict = evaluate_bootstrap_metrics(
        y_pred_prob=y_pred_prob_multilabel,
        y=y_multilabel,
        thresholds=threshold_dict,
        n_samples=50,
        num_resamples=20,
        metrics=["roc_auc", "f1_weighted"],
        average="macro",
    )

    assert isinstance(results_dict, pd.DataFrame), "The return type should be DataFrame"
    assert len(results_dict) == 2, "Should have 2 metrics evaluated"


def test_evaluate_bootstrap_metrics() -> None:
    """Test the evaluate_bootstrap_metrics function with various configurations."""
    # Generate data
    X, y = generate_classification_data(n_samples=500)

    X_single = X.iloc[:, [0]]

    model = Model(
        name="compare_test",
        estimator_name="rf",
        estimator=RandomForestClassifier(),
        model_type="classification",
        grid={},  # Not doing hyperparam tuning for simplicity
        scoring=["accuracy"],  # Simplify
        class_labels=["0", "1"],  # Let’s define these for the classification report
    )

    model.grid_search_param_tuning(X_single, y)
    model.fit(X, y)
    # Test without y_pred_prob
    results = evaluate_bootstrap_metrics(
        model=model,
        X=X,
        y=y,
        n_samples=50,
        num_resamples=20,
        metrics=["roc_auc"],
    )
    assert isinstance(results, pd.DataFrame), "The return type should be DataFrame"
    assert "Metric" in results.columns, "Results DataFrame must include 'Metric' column"
    assert len(results) > 0, "There should be at least one metric evaluated"

    # Test with y_pred_prob
    y_pred_prob = pd.DataFrame(model.predict_proba(X)[:, 1])
    results_prob = evaluate_bootstrap_metrics(
        y_pred_prob=y_pred_prob,
        y=y,
        n_samples=50,
        num_resamples=20,
        metrics=["roc_auc"],
    )

    assert isinstance(results_prob, pd.DataFrame), "The return type should be DataFrame"

    # Test for errors with incompatible model_type and metrics
    with pytest.raises(ValueError):
        evaluate_bootstrap_metrics(
            model=model,
            X=X,
            y=y,
            n_samples=50,
            num_resamples=10,
            model_type="regression",
            metrics=["roc_auc"],
        )

    # Test error when y is not provided
    with pytest.raises(ValueError):
        evaluate_bootstrap_metrics(model=model, X=X)


def test_missing_model_and_y_pred_prob() -> None:
    X, y = generate_classification_data(n_samples=100)
    with pytest.raises(
        ValueError, match="Either model and X or y_pred_prob must be provided."
    ):
        evaluate_bootstrap_metrics(y=y, n_samples=10, num_resamples=2)


def test_mismatch_model_type_and_metrics() -> None:
    X, y = generate_classification_data(n_samples=100)
    model = RandomForestClassifier().fit(X, y)
    with pytest.raises(
        ValueError,
        match="If using regression metrics please specify model_type='regression'",
    ):
        evaluate_bootstrap_metrics(
            model=model, X=X, y=y, model_type="classification", metrics=["r2"]
        )


def test_balance_with_regression_type() -> None:
    X, y = generate_classification_data(n_samples=100)
    model = RandomForestClassifier().fit(X, y)
    with pytest.raises(
        ValueError,
        match="Error: Balancing classes is not applicable for 'regression' tasks.",
    ):
        evaluate_bootstrap_metrics(
            model=model, X=X, y=y, model_type="regression", balance=True
        )


if __name__ == "__main__":
    pytest.main()
