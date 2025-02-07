import numpy as np
import pytest
from sklearn.metrics import fbeta_score
from src.model_tuner.threshold_optimization import (
    threshold_tune,
    find_optimal_threshold_beta,
)
import pandas as pd


empty_inputs = [
    (np.array([]), np.array([0.1, 0.2, 0.3]), "y"),
    (np.array([0, 1, 0]), np.array([]), "y_proba"),
    (pd.DataFrame(), pd.Series([0.1, 0.2, 0.3]), "y"),
    (pd.Series([0, 1, 0]), pd.DataFrame(), "y_proba"),
]


@pytest.fixture
def sample_data():
    """Fixture to generate sample binary classification data"""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)  ## Binary labels
    y_proba = np.random.rand(100)  ## Probability scores
    return y_true, y_proba


def test_threshold_tune(sample_data):
    """Test threshold tuning function with valid inputs."""
    y, y_proba = sample_data
    betas = [0.5, 1.0, 2.0]
    threshold = threshold_tune(y, y_proba, betas)

    assert 0 <= threshold <= 1, "Threshold should be within the range [0,1]"


def test_threshold_tune_invalid_input():
    """Test threshold tuning with invalid inputs."""
    y = np.array([0, 1, 1, 0])
    y_proba = np.array([0.2, 0.8, 0.6, 0.4])
    betas = []  # Invalid betas list

    with pytest.raises(ValueError):
        threshold_tune(y, y_proba, betas)


@pytest.mark.parametrize("target_metric", ["precision", "recall"])
def test_find_optimal_threshold_beta(sample_data, target_metric):
    """Test find_optimal_threshold_beta function with valid input."""
    y, y_proba = sample_data
    target_score = 0.7  # Arbitrary chosen target score

    threshold, beta = find_optimal_threshold_beta(
        y,
        y_proba,
        target_metric,
        target_score,
        beta_value_range=np.linspace(0.01, 4, 40),
        delta=0.18,
    )

    assert 0 <= threshold <= 1, "Threshold should be within [0,1]"
    assert 0.01 <= beta <= 4, "Beta should be within the given range"


def test_find_optimal_threshold_invalid_metric(sample_data):
    """Test invalid target metric input."""
    y, y_proba = sample_data
    target_metric = "invalid_metric"

    with pytest.raises(ValueError):
        find_optimal_threshold_beta(
            y,
            y_proba,
            target_metric,
            target_score=0.7,
            beta_value_range=np.linspace(0.01, 4, 40),
            delta=0.18,
        )


def test_find_optimal_threshold_no_suitable_beta(sample_data):
    """Test when no suitable beta is found within delta tolerance."""
    y, y_proba = sample_data
    target_metric = "precision"
    target_score = 0.99  # Unreasonably high target precision that won't be met

    with pytest.raises(Exception):
        result = find_optimal_threshold_beta(
            y,
            y_proba,
            target_metric,
            target_score,
            beta_value_range=np.linspace(0.01, 4, 40),
            delta=0.18,
        )


@pytest.mark.parametrize("y, y_proba, which_empty", empty_inputs)
def test_threshold_tune_empty_input(y, y_proba, which_empty):
    betas = [1.0]
    with pytest.raises(ValueError):
        threshold_tune(y, y_proba, betas)


@pytest.mark.parametrize("y, y_proba, which_empty", empty_inputs)
def test_find_optimal_threshold_beta_empty_input(y, y_proba, which_empty):
    target_metric = "precision"
    target_score = 0.5
    with pytest.raises(ValueError):
        find_optimal_threshold_beta(y, y_proba, target_metric, target_score)


if __name__ == "__main__":
    pytest.main()
