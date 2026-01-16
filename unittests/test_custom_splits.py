import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

# IMPORTANT:
# Update this import to match where your Model + train_val_test_split live.
# Example:
#   from my_pkg.model import Model
#   import my_pkg.model as model_module
from model_tuner import Model
import model_tuner as model_module


def _make_classification_df(n=250, random_state=0):
    X, y = make_classification(
        n_samples=n,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[0.6, 0.4],
        random_state=random_state,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="target")
    return X, y


def _make_custom_splits(X, y, random_state=0):
    # 60/20/20 with stratification so each split has both classes
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=random_state
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )

    # sanity: each split should contain both classes for roc_auc
    for ys in (y_train, y_valid, y_test):
        assert set(ys.unique()) == {0, 1}

    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
    }


def test_grid_search_param_tuning_uses_custom_splits_and_skips_internal_split(monkeypatch):
    X, y = _make_classification_df()
    custom_splits = _make_custom_splits(X, y)

    # If train_val_test_split is called, this test should fail.
    def _boom(*args, **kwargs):
        raise AssertionError("train_val_test_split should NOT be called when custom_splits is provided")

    monkeypatch.setattr(model_module, "train_val_test_split", _boom)

    lr = LogisticRegression(max_iter=1000, solver="liblinear")

    model = Model(
        name="lr_test",
        estimator_name="lr",
        estimator=lr,
        model_type="classification",
        calibrate=False,
        kfold=False,
        grid={"lr__C": [0.1, 1.0]},
        scoring=["roc_auc"],
        display=False,
        randomized_grid=False,
    )

    model.grid_search_param_tuning(X, y, custom_splits=custom_splits)

    # Indices should match the provided splits (because you store them when X is a DataFrame)
    assert model.X_train_index == custom_splits["X_train"].index.to_list()
    assert model.X_valid_index == custom_splits["X_valid"].index.to_list()
    assert model.X_test_index == custom_splits["X_test"].index.to_list()

    # Best params populated
    assert "roc_auc" in model.best_params_per_score
    best = model.best_params_per_score["roc_auc"]
    assert "params" in best and "score" in best
    assert best["params"]["lr__C"] in (0.1, 1.0)
    assert isinstance(best["score"], float)

    # Optional: verify the reported best C truly corresponds to the max ROC AUC on the *provided* validation set
    # (This makes the test more "meaningful" vs just "it ran".)
    X_train = custom_splits["X_train"]
    y_train = custom_splits["y_train"]
    X_valid = custom_splits["X_valid"]
    y_valid = custom_splits["y_valid"]

    scores = {}
    for c in [0.1, 1.0]:
        pipe = Pipeline([("lr", LogisticRegression(max_iter=1000, solver="liblinear", C=c))])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_valid)[:, 1]
        scores[c] = roc_auc_score(y_valid, proba)

    expected_best_c = max(scores, key=scores.get)
    assert best["params"]["lr__C"] == expected_best_c
    # allow tiny floating diffs
    assert best["score"] == pytest.approx(scores[expected_best_c], rel=1e-6, abs=1e-6)


def test_calibrateModel_uses_custom_splits_and_skips_internal_split(monkeypatch):
    X, y = _make_classification_df()
    custom_splits = _make_custom_splits(X, y)

    # If train_val_test_split is called, this test should fail.
    def _boom(*args, **kwargs):
        raise AssertionError("train_val_test_split should NOT be called when custom_splits is provided")

    monkeypatch.setattr(model_module, "train_val_test_split", _boom)

    lr = LogisticRegression(max_iter=1000, solver="liblinear")

    model = Model(
        name="lr_cal_test",
        estimator_name="lr",
        estimator=lr,
        model_type="classification",
        calibrate=True,
        kfold=False,
        grid={"lr__C": [1.0]},
        scoring=["roc_auc"],
        display=False,
    )

    model.best_params_per_score["roc_auc"] = {"params": {"lr__C": 1.0}, "score": 0.0}

    model.calibrateModel(X, y, custom_splits=custom_splits)

    # Indices should match the provided splits
    assert model.X_train_index == custom_splits["X_train"].index.to_list()
    assert model.X_valid_index == custom_splits["X_valid"].index.to_list()
    assert model.X_test_index == custom_splits["X_test"].index.to_list()

    # Estimator should now be a fitted CalibratedClassifierCV
    assert isinstance(model.estimator, CalibratedClassifierCV)
    assert hasattr(model.estimator, "classes_")

    # And it should produce probabilities on the provided test split
    proba = model.predict_proba(custom_splits["X_test"])
    assert proba.shape[0] == len(custom_splits["X_test"])
    assert proba.shape[1] == 2
    assert np.all(np.isfinite(proba))
