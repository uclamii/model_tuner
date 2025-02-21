import numpy as np
import pytest
from src.model_tuner.threshold_optimization import find_threshold_for_precision_recall
import pandas as pd


@pytest.fixture
def sample_data():
    """Fixture to generate sample binary classification data"""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)  # Binary labels
    y_proba = np.random.rand(100)  # Probability scores
    return y_true, y_proba


def test_basic_functionality(sample_data):
    """Test basic functionality with valid inputs."""
    y_true, y_proba = sample_data
    result = find_threshold_for_precision_recall(
        y_true, y_proba, target_metric="precision", min_target_metric=0.5, beta=1.0
    )

    assert result is not None, "Should find a valid threshold for reasonable target"
    threshold, precision, fbeta = result
    assert 0 <= threshold <= 1, "Threshold should be within [0,1]"
    assert 0 <= precision <= 1, "Precision should be within [0,1]"
    assert 0 <= fbeta <= 1, "F-beta score should be within [0,1]"
    assert precision >= 0.5, "Returned precision should meet minimum target"


def test_impossible_target():
    """Test when target metric is impossibly high."""
    # Create a case where even with the highest threshold possible,
    # we can't achieve perfect precision because we have identical
    # probabilities for different classes
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.9, 0.9, 0.9, 0.9])  # Same probability for both classes

    result = find_threshold_for_precision_recall(
        y_true,
        y_proba,
        target_metric="precision",
        min_target_metric=1.0,  # Require perfect precision
    )

    assert result is None, "Should return None when target is impossible to achieve"


def test_perfect_predictions():
    """Test with perfect predictions."""
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.1, 0.9])

    result = find_threshold_for_precision_recall(
        y_true,
        y_proba,
        target_metric="precision",
        min_target_metric=1.0,
    )

    assert result is not None, "Should find threshold for perfect predictions"
    threshold, precision, fbeta = result
    assert precision == 1.0, "Should achieve perfect precision"


@pytest.mark.parametrize("target_metric", ["precision", "recall"])
def test_different_target_metrics(sample_data, target_metric):
    """Test both precision and recall as target metrics."""
    y_true, y_proba = sample_data
    result = find_threshold_for_precision_recall(
        y_true,
        y_proba,
        target_metric=target_metric,
        min_target_metric=0.5,
    )

    assert result is not None, "Should find valid threshold for both metrics"
    threshold, metric_val, fbeta = result
    assert metric_val >= 0.5, f"Should meet minimum {target_metric} target"


def test_invalid_target_metric(sample_data):
    """Test with invalid target metric."""
    y_true, y_proba = sample_data

    with pytest.raises(ValueError, match="Please specify either precision or recall"):
        find_threshold_for_precision_recall(
            y_true,
            y_proba,
            target_metric="invalid_metric",
        )


def test_different_beta_values(sample_data):
    """Test with different beta values."""
    y_true, y_proba = sample_data
    betas = [0.5, 1.0, 2.0]

    for beta in betas:
        result = find_threshold_for_precision_recall(
            y_true, y_proba, target_metric="precision", min_target_metric=0.5, beta=beta
        )
        assert result is not None, f"Should work with beta={beta}"


def test_input_validation():
    """Test input validation."""
    # Empty arrays
    with pytest.raises(ValueError):
        find_threshold_for_precision_recall(
            np.array([]), np.array([]), target_metric="precision"
        )

    # Mismatched lengths
    with pytest.raises(ValueError):
        find_threshold_for_precision_recall(
            np.array([0, 1]), np.array([0.5]), target_metric="precision"
        )

    # Invalid values in y_true
    with pytest.raises(ValueError):
        find_threshold_for_precision_recall(
            np.array([0, 1, 2]),  # Invalid label
            np.array([0.1, 0.2, 0.3]),
            target_metric="precision",
        )

    # Invalid probabilities
    with pytest.raises(ValueError):
        find_threshold_for_precision_recall(
            np.array([0, 1, 0]),
            np.array([0.1, 1.2, 0.3]),  # Invalid probability > 1
            target_metric="precision",
        )


if __name__ == "__main__":
    pytest.main()
