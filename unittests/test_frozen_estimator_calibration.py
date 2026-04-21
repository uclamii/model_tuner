"""
Tests for PR #231: sklearn >= 1.6 migration to FrozenEstimator for calibration.

Background:
    sklearn 1.6 removed the ``cv="prefit"`` value from CalibratedClassifierCV
    and introduced ``FrozenEstimator`` as its replacement. The library now
    dispatches on ``_HAS_FROZEN_ESTIMATOR`` to choose between the two
    construction patterns:

    * Frozen path (sklearn >= 1.6):
        CalibratedClassifierCV(estimator=FrozenEstimator(est), method=...)
    * Prefit fallback (sklearn < 1.6):
        CalibratedClassifierCV(est, cv="prefit", method=...)

Design:
    The prefit fallback is *unreachable* end-to-end on sklearn >= 1.6 because
    sklearn rejects ``cv='prefit'`` at parameter validation time. Testing the
    fallback by monkeypatching the flag to False and then running the real
    CalibratedClassifierCV therefore cannot succeed on a modern install.

    Instead, the prefit branch is verified with mock-based unit tests that
    replace ``mtu.CalibratedClassifierCV`` and assert the library dispatches
    the correct construction arguments. The frozen path is verified with
    real integration tests against the installed sklearn.

No tests are skipped. Every test runs on every environment. Frozen tests
import ``FrozenEstimator`` lazily so that collection never fails on
sklearn < 1.6 installs; the failures on such an install would then be the
real signal that the install does not match pyproject.toml's
``scikit-learn>=1.6`` requirement on Python 3.10+.
"""

from __future__ import annotations

import importlib
from typing import Tuple
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, brier_score_loss
from sklearn.model_selection import train_test_split

from model_tuner import Model
import model_tuner.model_tuner_utils as mtu


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


@pytest.fixture
def classification_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Moderate synthetic binary classification set, stratifiable."""
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        weights=[0.6, 0.4],
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="target")
    return X, y


@pytest.fixture
def custom_splits_60_20_20(classification_data):
    """60/20/20 stratified split passed straight through to the Model."""
    X, y = classification_data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=0
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=0
    )
    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
    }


def _build_calibrated_lr(
    calibration_method: str = "sigmoid", kfold: bool = False
) -> Model:
    """Factory for a calibrated logistic regression Model harness."""
    return Model(
        name="lr_cal",
        estimator_name="lr",
        estimator=LogisticRegression(max_iter=1000, solver="liblinear"),
        model_type="classification",
        grid={"lr__C": [0.1, 1.0]},
        scoring=["roc_auc"],
        kfold=kfold,
        calibrate=True,
        calibration_method=calibration_method,
        display=False,
        randomized_grid=False,
    )


# ----------------------------------------------------------------------------
# 1. Module level flag detection
# ----------------------------------------------------------------------------


def test_has_frozen_estimator_flag_exists():
    """The module must expose ``_HAS_FROZEN_ESTIMATOR`` as a bool."""
    assert hasattr(mtu, "_HAS_FROZEN_ESTIMATOR")
    assert isinstance(mtu._HAS_FROZEN_ESTIMATOR, bool)


def test_has_frozen_estimator_flag_matches_import_capability():
    """
    Flag must be True iff ``sklearn.frozen.FrozenEstimator`` is importable.
    Catches regressions where the try/except swallows a non-ImportError.
    """
    try:
        from sklearn.frozen import FrozenEstimator  # noqa: F401

        expected = True
    except ImportError:
        expected = False
    assert mtu._HAS_FROZEN_ESTIMATOR is expected


def test_flag_survives_module_reimport():
    """Reimporting the module should not flip the flag."""
    original = mtu._HAS_FROZEN_ESTIMATOR
    importlib.reload(mtu)
    assert mtu._HAS_FROZEN_ESTIMATOR == original


# ----------------------------------------------------------------------------
# 2. Frozen path: real integration against installed sklearn
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("calibration_method", ["sigmoid", "isotonic"])
def test_calibrate_model_frozen_path_uses_frozen_estimator(
    classification_data, custom_splits_60_20_20, calibration_method
):
    """
    With the real sklearn 1.6+ install, the calibrated estimator must wrap
    the base estimator in FrozenEstimator and must not use ``cv='prefit'``.
    """
    from sklearn.frozen import FrozenEstimator

    X, y = classification_data
    model = _build_calibrated_lr(calibration_method=calibration_method)
    model.best_params_per_score["roc_auc"] = {"params": {"lr__C": 1.0}, "score": 0.0}

    model.calibrateModel(X, y, custom_splits=custom_splits_60_20_20)

    assert isinstance(model.estimator, CalibratedClassifierCV)
    wrapped = getattr(model.estimator, "estimator", None)
    assert wrapped is not None, "CalibratedClassifierCV.estimator attribute missing"
    assert isinstance(
        wrapped, FrozenEstimator
    ), f"Expected FrozenEstimator wrapper, got {type(wrapped).__name__}"
    assert getattr(model.estimator, "cv", None) != "prefit"
    assert model.estimator.method == calibration_method


def test_frozen_path_produces_valid_probabilities(
    classification_data, custom_splits_60_20_20
):
    """
    Real frozen path: predict_proba should yield finite probabilities in
    [0, 1] that sum to 1 across classes.
    """
    X, y = classification_data
    model = _build_calibrated_lr()
    model.best_params_per_score["roc_auc"] = {"params": {"lr__C": 1.0}, "score": 0.0}

    model.calibrateModel(X, y, custom_splits=custom_splits_60_20_20)

    proba = model.predict_proba(custom_splits_60_20_20["X_test"])
    assert proba.shape == (len(custom_splits_60_20_20["X_test"]), 2)
    assert np.all(np.isfinite(proba))
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ----------------------------------------------------------------------------
# 3. Prefit fallback: mock based branch dispatch verification
#
# The prefit branch cannot run end-to-end on sklearn >= 1.6 because sklearn
# removed ``cv='prefit'`` in 1.6. We mock CalibratedClassifierCV to verify
# the library dispatches the correct construction arguments.
# ----------------------------------------------------------------------------


def _install_cccv_mock(monkeypatch) -> MagicMock:
    """
    Replace ``mtu.CalibratedClassifierCV`` with a MagicMock that mimics the
    chainable ``CalibratedClassifierCV(...).fit(...)`` pattern used in the
    library and returns a fitted-looking mock.
    """
    fit_result = MagicMock(name="FittedCCCV")
    fit_result.classes_ = np.array([0, 1])
    fit_result.predict_proba.return_value = np.tile([0.5, 0.5], (10, 1))
    fit_result.predict.return_value = np.zeros(10, dtype=int)

    cccv_ctor = MagicMock(name="CalibratedClassifierCV")
    cccv_instance = MagicMock(name="CCCVInstance")
    cccv_instance.fit.return_value = fit_result
    cccv_ctor.return_value = cccv_instance

    monkeypatch.setattr(mtu, "CalibratedClassifierCV", cccv_ctor)
    return cccv_ctor


@pytest.mark.parametrize("calibration_method", ["sigmoid", "isotonic"])
def test_prefit_branch_dispatches_cv_prefit(
    monkeypatch, classification_data, custom_splits_60_20_20, calibration_method
):
    """
    When the flag is False, calibrateModel must construct
    ``CalibratedClassifierCV(<pipeline>, cv='prefit', method=<method>)``.
    """
    monkeypatch.setattr(mtu, "_HAS_FROZEN_ESTIMATOR", False)
    cccv_mock = _install_cccv_mock(monkeypatch)

    X, y = classification_data
    model = _build_calibrated_lr(calibration_method=calibration_method)
    model.best_params_per_score["roc_auc"] = {"params": {"lr__C": 1.0}, "score": 0.0}

    model.calibrateModel(X, y, custom_splits=custom_splits_60_20_20)

    assert (
        cccv_mock.call_count == 1
    ), "CalibratedClassifierCV should be constructed exactly once"
    args, kwargs = cccv_mock.call_args

    # cv="prefit" and method="<calibration_method>" must be passed as kwargs.
    assert (
        kwargs.get("cv") == "prefit"
    ), f"Prefit branch must pass cv='prefit', got cv={kwargs.get('cv')!r}"
    assert kwargs.get("method") == calibration_method
    # The estimator must be passed as the first positional arg, not wrapped.
    assert len(args) == 1, f"Expected exactly 1 positional arg, got {len(args)}"
    # And crucially, no 'estimator' kwarg on the prefit branch.
    assert "estimator" not in kwargs


def test_prefit_branch_does_not_wrap_in_frozen_estimator(
    monkeypatch, classification_data, custom_splits_60_20_20
):
    """
    Belt and suspenders: when the flag is False, FrozenEstimator must not be
    invoked even if it happens to be importable in the module namespace.
    """
    monkeypatch.setattr(mtu, "_HAS_FROZEN_ESTIMATOR", False)
    cccv_mock = _install_cccv_mock(monkeypatch)

    frozen_mock = MagicMock(name="FrozenEstimator")
    monkeypatch.setattr(mtu, "FrozenEstimator", frozen_mock, raising=False)

    X, y = classification_data
    model = _build_calibrated_lr(calibration_method="sigmoid")
    model.best_params_per_score["roc_auc"] = {"params": {"lr__C": 1.0}, "score": 0.0}

    model.calibrateModel(X, y, custom_splits=custom_splits_60_20_20)

    # FrozenEstimator must never be called on the prefit branch.
    assert (
        frozen_mock.call_count == 0
    ), "FrozenEstimator should not be invoked when _HAS_FROZEN_ESTIMATOR is False"
    # And CalibratedClassifierCV should have been called with cv='prefit'.
    _args, kwargs = cccv_mock.call_args
    assert kwargs.get("cv") == "prefit"


# ----------------------------------------------------------------------------
# 4. Frozen path: mock based dispatch check (complement to integration test)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("calibration_method", ["sigmoid", "isotonic"])
def test_frozen_branch_dispatches_frozen_estimator_kwarg(
    monkeypatch, classification_data, custom_splits_60_20_20, calibration_method
):
    """
    When the flag is True, calibrateModel must construct
    ``CalibratedClassifierCV(estimator=FrozenEstimator(<pipeline>),
    method=<method>)`` and must not set cv='prefit'.
    """
    monkeypatch.setattr(mtu, "_HAS_FROZEN_ESTIMATOR", True)
    cccv_mock = _install_cccv_mock(monkeypatch)

    frozen_mock = MagicMock(name="FrozenEstimator")
    monkeypatch.setattr(mtu, "FrozenEstimator", frozen_mock, raising=False)

    X, y = classification_data
    model = _build_calibrated_lr(calibration_method=calibration_method)
    model.best_params_per_score["roc_auc"] = {"params": {"lr__C": 1.0}, "score": 0.0}

    model.calibrateModel(X, y, custom_splits=custom_splits_60_20_20)

    # FrozenEstimator must be called exactly once to wrap the pipeline.
    assert frozen_mock.call_count == 1
    # CalibratedClassifierCV must be passed estimator=<FrozenEstimator return value>.
    _args, kwargs = cccv_mock.call_args
    assert kwargs.get("estimator") is frozen_mock.return_value
    assert kwargs.get("method") == calibration_method
    # cv='prefit' must not appear on the frozen branch.
    assert kwargs.get("cv") != "prefit"


# ----------------------------------------------------------------------------
# 5. kfold branch: real integration against installed sklearn
#
# Note: the kfold=True branch of calibrateModel does not reliably reach the
# same frozen/prefit if/else as the non-kfold branch under default calibrate
# parameters. The non-kfold mock tests above already verify the dispatch
# logic, and the integration test below confirms the kfold + calibrate path
# produces a valid calibrated model end to end.
# ----------------------------------------------------------------------------


def test_kfold_frozen_path_produces_valid_probabilities(classification_data):
    """
    Real integration: kfold=True with the frozen path should produce a
    fitted CalibratedClassifierCV that outputs valid probabilities.
    """
    X, y = classification_data
    model = Model(
        name="lr_cal_kfold",
        estimator_name="lr",
        estimator=LogisticRegression(max_iter=1000, solver="liblinear"),
        model_type="classification",
        grid={"lr__C": [0.1, 1.0]},
        scoring=["roc_auc"],
        kfold=True,
        calibrate=True,
        calibration_method="sigmoid",
        display=False,
    )

    model.grid_search_param_tuning(X, y)
    model.fit(X, y)
    model.calibrateModel(X, y)

    assert isinstance(model.estimator, CalibratedClassifierCV)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ----------------------------------------------------------------------------
# 6. FrozenEstimator must not refit the underlying model
# ----------------------------------------------------------------------------


def test_frozen_estimator_preserves_fitted_coefficients(
    classification_data, custom_splits_60_20_20
):
    """
    The point of FrozenEstimator is to keep the underlying estimator
    unchanged while CalibratedClassifierCV learns a calibrator on top.
    Verify the LR coefficients are unchanged after calibrateModel.
    """
    from sklearn.frozen import FrozenEstimator

    X, y = classification_data
    model = _build_calibrated_lr()
    model.best_params_per_score["roc_auc"] = {"params": {"lr__C": 1.0}, "score": 0.0}

    model.fit(custom_splits_60_20_20["X_train"], custom_splits_60_20_20["y_train"])
    pre_cal_pipeline = model.estimator
    pre_cal_lr = pre_cal_pipeline.named_steps["lr"]
    coef_before = pre_cal_lr.coef_.copy()
    intercept_before = pre_cal_lr.intercept_.copy()

    model.calibrateModel(X, y, custom_splits=custom_splits_60_20_20)

    # Navigate: CalibratedClassifierCV -> FrozenEstimator -> Pipeline -> LR
    wrapped = model.estimator.estimator
    assert isinstance(wrapped, FrozenEstimator)
    inner_pipeline = wrapped.estimator
    inner_lr = inner_pipeline.named_steps["lr"]

    np.testing.assert_allclose(inner_lr.coef_, coef_before, atol=1e-10)
    np.testing.assert_allclose(inner_lr.intercept_, intercept_before, atol=1e-10)


# ----------------------------------------------------------------------------
# 7. Calibration curve AUC pattern from the updated rf_calibrated.py example
# ----------------------------------------------------------------------------


def test_calibration_curve_auc_pattern_matches_example_script(classification_data):
    """
    The updated example script computes the area under the calibration curve
    using the trapezoidal rule. Lock in the invariants of that pattern.
    """
    X, y = classification_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0
    )
    rf = RandomForestClassifier(n_estimators=50, random_state=0).fit(X_train, y_train)
    y_prob = rf.predict_proba(X_test)[:, 1]

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_prob, n_bins=10, strategy="uniform"
    )
    auc_calibration = auc(mean_predicted_value, fraction_of_positives)

    assert np.isfinite(auc_calibration)
    assert 0.0 <= auc_calibration <= 1.0
    # Perfect calibration sits near 0.5 (diagonal). A reasonably tuned RF on
    # this data should land between 0.2 and 0.9.
    assert 0.2 <= auc_calibration <= 0.9


def test_frozen_estimator_brier_score_sanity(custom_splits_60_20_20):
    """
    Calibrated probabilities via FrozenEstimator should produce a Brier
    score at least as good as the uncalibrated baseline on a held out set.
    This exercises the CalibratedClassifierCV + FrozenEstimator API
    directly (outside the Model wrapper).
    """
    from sklearn.frozen import FrozenEstimator

    X_train = custom_splits_60_20_20["X_train"]
    y_train = custom_splits_60_20_20["y_train"]
    X_valid = custom_splits_60_20_20["X_valid"]
    y_valid = custom_splits_60_20_20["y_valid"]
    X_test = custom_splits_60_20_20["X_test"]
    y_test = custom_splits_60_20_20["y_test"]

    base = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=0)
    base.fit(X_train, y_train)
    p_uncal = base.predict_proba(X_test)[:, 1]
    brier_uncal = brier_score_loss(y_test, p_uncal)

    cal = CalibratedClassifierCV(estimator=FrozenEstimator(base), method="sigmoid").fit(
        X_valid, y_valid
    )
    p_cal = cal.predict_proba(X_test)[:, 1]
    brier_cal = brier_score_loss(y_test, p_cal)

    assert brier_cal <= brier_uncal + 0.02


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
