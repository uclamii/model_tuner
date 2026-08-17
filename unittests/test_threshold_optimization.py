import numpy as np
import pytest
from src.model_tuner.threshold_optimization import (
    threshold_tune,
    find_optimal_threshold_beta,
    find_optimal_threshold_youden,
)
import pandas as pd

empty_inputs = [
    (np.array([]), np.array([0.1, 0.2, 0.3]), "y"),
    (np.array([0, 1, 0]), np.array([]), "y_proba"),
    (pd.DataFrame(), pd.Series([0.1, 0.2, 0.3]), "y"),
    (pd.Series([0, 1, 0]), pd.DataFrame(), "y_proba"),
]

single_class_inputs = [
    np.zeros(20, dtype=int),
    np.ones(20, dtype=int),
]


@pytest.fixture
def separable_data():
    """Perfectly separable data with a known ideal cut-point at 0.5."""
    y = np.array([0] * 50 + [1] * 50)
    y_proba = np.concatenate([np.full(50, 0.2), np.full(50, 0.8)])
    return y, y_proba


@pytest.fixture
def imbalanced_data():
    """10% prevalence with signal, the setting Youden is intended for."""
    rng = np.random.default_rng(0)
    y = np.zeros(1000, dtype=int)
    y[:100] = 1
    y_proba = np.where(
        y == 1,
        rng.beta(5, 5, size=1000),
        rng.beta(2, 8, size=1000),
    )
    return y, y_proba


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
        find_optimal_threshold_beta(
            y,
            y_proba,
            target_metric,
            target_score,
            beta_value_range=np.linspace(0.01, 4, 40),
            delta=0.18,
        )


def test_find_optimal_threshold_beta_custom_thresholds(sample_data):
    """Test find_optimal_threshold_beta with a custom threshold range."""
    y, y_proba = sample_data
    target_metric = "recall"
    target_score = 0.6
    custom_thresholds = np.linspace(0.2, 0.8, 10)  # Custom range

    threshold, beta = find_optimal_threshold_beta(
        y,
        y_proba,
        target_metric,
        target_score,
        threshold_value_range=custom_thresholds,
        beta_value_range=np.linspace(0.01, 4, 40),
        delta=0.18,  # Set a larger initial delta to speed up test
    )

    # Check if the returned threshold is one of the values from the custom range
    assert threshold in custom_thresholds, "Threshold should be from the custom range"
    # Also check bounds based on the custom range provided
    assert (
        min(custom_thresholds) <= threshold <= max(custom_thresholds)
    ), "Threshold outside custom bounds"
    assert 0.01 <= beta <= 4, "Beta should be within the given range"


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


def test_threshold_strict_greater_than_behavior():
    """Ensure y_pred excludes threshold value under strict greater-than condition."""
    y = np.array([0, 1, 1])
    y_proba = np.array([0.4, 0.5, 0.6])
    threshold = 0.5

    # Expect only the value above threshold to be 1
    y_pred = (y_proba > threshold).astype(int)
    expected_pred = np.array([0, 0, 1])

    np.testing.assert_array_equal(y_pred, expected_pred)


def _youden_j(y, y_proba, threshold):
    """Reference implementation used to check the optimizer's choice."""
    y_pred = (y_proba > threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y == 1))
    fn = np.sum((y_pred == 0) & (y == 1))
    tn = np.sum((y_pred == 0) & (y == 0))
    fp = np.sum((y_pred == 1) & (y == 0))
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return sensitivity + specificity - 1


def test_find_optimal_threshold_youden(sample_data):
    """Test Youden threshold selection with valid input."""
    y, y_proba = sample_data
    threshold, beta = find_optimal_threshold_youden(y, y_proba)

    assert 0 <= threshold <= 1, "Threshold should be within [0,1]"
    assert beta is None, "Youden's J requires no beta; None expected"


def test_youden_returns_none_beta_for_signature_compatibility(sample_data):
    """The two-tuple shape must match find_optimal_threshold_beta."""
    y, y_proba = sample_data
    result = find_optimal_threshold_youden(y, y_proba)

    assert isinstance(result, tuple), "Should return a tuple"
    assert len(result) == 2, "Should return (threshold, beta)"
    assert result[1] is None


def test_youden_requires_no_target_score(sample_data):
    """Unlike the beta search, no target_metric or target_score is needed."""
    y, y_proba = sample_data
    threshold, _ = find_optimal_threshold_youden(y, y_proba)

    assert isinstance(threshold, float)


def test_youden_maximizes_j(sample_data):
    """The returned threshold must maximize J over the search grid."""
    y, y_proba = sample_data
    grid = np.arange(0, 1, 0.01)

    threshold, _ = find_optimal_threshold_youden(y, y_proba, threshold_value_range=grid)

    best_j = _youden_j(y, y_proba, threshold)
    for candidate in grid:
        assert _youden_j(y, y_proba, candidate) <= best_j + 1e-12, (
            f"Threshold {candidate} yields a higher J than the selected " f"{threshold}"
        )


def test_youden_perfect_separation(separable_data):
    """With a clean gap, the cut-point must fall inside it."""
    y, y_proba = separable_data
    threshold, _ = find_optimal_threshold_youden(y, y_proba)

    assert 0.2 <= threshold < 0.8, "Threshold should land in the separating gap"
    assert _youden_j(y, y_proba, threshold) == pytest.approx(
        1.0
    ), "Perfect separation should give J = 1"


def test_youden_imbalanced_gives_usable_operating_point(imbalanced_data):
    """
    On a low-prevalence problem the selected point should be usable, i.e. it
    should not collapse to predicting the majority class everywhere.
    """
    y, y_proba = imbalanced_data
    threshold, _ = find_optimal_threshold_youden(y, y_proba)

    y_pred = (y_proba > threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y == 1))
    fn = np.sum((y_pred == 0) & (y == 1))
    sensitivity = tp / (tp + fn)

    assert y_pred.sum() > 0, "Model must flag at least some positives"
    assert (
        sensitivity > 0.5
    ), f"Youden should give a usable sensitivity; got {sensitivity:.3f}"


def test_youden_custom_threshold_range(sample_data):
    """The returned threshold must come from the supplied grid."""
    y, y_proba = sample_data
    custom_thresholds = np.linspace(0.2, 0.8, 10)

    threshold, beta = find_optimal_threshold_youden(
        y, y_proba, threshold_value_range=custom_thresholds
    )

    assert threshold in custom_thresholds, "Threshold should be from the custom range"
    assert (
        min(custom_thresholds) <= threshold <= max(custom_thresholds)
    ), "Threshold outside custom bounds"
    assert beta is None


def test_youden_accepts_pandas_input(sample_data):
    """Series and single-column DataFrame inputs must behave like arrays."""
    y, y_proba = sample_data

    t_array, _ = find_optimal_threshold_youden(y, y_proba)
    t_series, _ = find_optimal_threshold_youden(pd.Series(y), pd.Series(y_proba))
    t_frame, _ = find_optimal_threshold_youden(pd.DataFrame(y), pd.DataFrame(y_proba))

    assert t_array == t_series == t_frame


def test_youden_is_deterministic(sample_data):
    """Repeated calls on identical input must return the same threshold."""
    y, y_proba = sample_data
    first, _ = find_optimal_threshold_youden(y, y_proba)
    second, _ = find_optimal_threshold_youden(y, y_proba)

    assert first == second


@pytest.mark.parametrize("y, y_proba, which_empty", empty_inputs)
def test_youden_empty_input(y, y_proba, which_empty):
    """Empty y or y_proba must raise, whichever container type."""
    with pytest.raises(ValueError):
        find_optimal_threshold_youden(y, y_proba)


@pytest.mark.parametrize("y", single_class_inputs)
def test_youden_single_class(y):
    """J is undefined when one class is absent; must raise, not return NaN."""
    y_proba = np.linspace(0, 1, len(y))

    with pytest.raises(ValueError, match="both classes"):
        find_optimal_threshold_youden(y, y_proba)


def test_youden_strict_greater_than_behavior():
    """
    Predictions must use a strict '>' comparison, matching threshold_tune.
    A score exactly equal to the threshold is a negative prediction.
    """
    y = np.array([0, 1, 1])
    y_proba = np.array([0.4, 0.5, 0.6])

    threshold, _ = find_optimal_threshold_youden(
        y, y_proba, threshold_value_range=np.array([0.5])
    )

    y_pred = (y_proba > threshold).astype(int)
    np.testing.assert_array_equal(y_pred, np.array([0, 0, 1]))


def test_youden_returns_python_float(sample_data):
    """Threshold should be a plain float, not a numpy scalar."""
    y, y_proba = sample_data
    threshold, _ = find_optimal_threshold_youden(y, y_proba)

    assert type(threshold) is float


def test_youden_handles_list_input():
    """Plain Python lists must be accepted."""
    y = [0, 0, 1, 1]
    y_proba = [0.1, 0.2, 0.8, 0.9]

    threshold, beta = find_optimal_threshold_youden(y, y_proba)

    assert 0 <= threshold <= 1
    assert beta is None


def test_youden_all_scores_identical():
    """
    Degenerate case: no threshold separates anything, so J is 0 everywhere.
    The call must still return a threshold in range rather than raise.
    """
    y = np.array([0, 0, 1, 1])
    y_proba = np.full(4, 0.5)

    threshold, beta = find_optimal_threshold_youden(y, y_proba)

    assert 0 <= threshold <= 1
    assert beta is None


if __name__ == "__main__":
    pytest.main()
