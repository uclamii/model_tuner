import pandas as pd
import numpy as np
import os
import sys

from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from model_tuner.model_tuner_utils import Model, AutoKerasClassifier
from model_tuner.bootstrapper import evaluate_bootstrap_metrics
from model_tuner.pickleObjects import dumpObjects, loadObjects

from sklearn.model_selection import train_test_split

import random


# temporary method
def train_val_test_split(
    X,
    y,
    stratify_y,
    train_size,
    validation_size,
    test_size,
    random_state,
    stratify_cols,
):

    if stratify_cols is not None and stratify_y:
        # Creating stratification columns out of stratify_cols list
        if type(stratify_cols) == pd.DataFrame:
            stratify_key = pd.concat([stratify_cols, y], axis=1)
        else:
            stratify_key = pd.concat([X[stratify_cols], y], axis=1)
    elif stratify_cols is not None:
        stratify_key = X[stratify_cols]
    elif stratify_y is not None:
        stratify_key = y
    else:
        stratify_key = None

    if stratify_cols is not None:
        # stratify_key = stratify_key.copy()
        stratify_key = stratify_key.fillna("")

    X_train, X_valid_test, y_train, y_valid_test = train_test_split(
        X,
        y,
        test_size=1 - train_size,
        stratify=stratify_key,  # Use stratify_key here
        random_state=random_state,
    )

    # Determine the proportion of validation to test size in the remaining dataset
    proportion = test_size / (validation_size + test_size)

    if stratify_cols is not None and stratify_y:
        # Creating stratification columns out of stratify_cols list
        if type(stratify_cols) == pd.DataFrame:
            strat_key_val_test = pd.concat(
                [stratify_cols.loc[X_valid_test.index, :], y_valid_test], axis=1
            )
        else:
            strat_key_val_test = pd.concat(
                [X_valid_test[stratify_cols], y_valid_test], axis=1
            )
    elif stratify_cols is not None:
        strat_key_val_test = X_valid_test[stratify_cols]
    elif stratify_y is not None:
        strat_key_val_test = y_valid_test
    else:
        strat_key_val_test = None

    if stratify_cols is not None:
        strat_key_val_test = strat_key_val_test.fillna("")

    # Further split (validation + test) set into validation and test sets
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_valid_test,
        y_valid_test,
        test_size=proportion,
        stratify=strat_key_val_test,
        random_state=random_state,
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test


import keras
import tensorflow as tf
from autokeras import StructuredDataClassifier


from keras_tuner import Objective, RandomSearch

bc = load_breast_cancer(as_frame=True)["frame"]
bc_cols = [cols for cols in bc.columns if "target" not in cols]
X = bc[bc_cols]
y = bc["target"]


# Set seed for TensorFlow
tf.random.set_seed(222)

# Set seed for NumPy
np.random.seed(222)

# Set seed for Python's built-in random module
random.seed(222)

# Ensure reproducibility for Python hash-based operations
os.environ["PYTHONHASHSEED"] = str(222)

# Set environment variable for Keras Tuner
os.environ["KERAS_TUNER_SEED"] = str(222)

ak_max_trials = 3
ak_epochs = 10

ak_clf = AutoKerasClassifier(
    StructuredDataClassifier(
        max_trials=ak_max_trials,
        loss="binary_crossentropy",
        metrics=[keras.metrics.Accuracy()],
        seed=222,
        overwrite=True,
    ),
    Pipeline([("impute", SimpleImputer()), ("scaler", StandardScaler())]),
)

X_train, X_valid, X_test, y_train, y_valid, y_test = train_val_test_split(
    X,
    y,
    stratify_y=y.values,
    train_size=0.6,
    validation_size=0.2,
    test_size=0.2,
    random_state=0,
    stratify_cols=None,
)

ak_clf.fit(
    X_train,
    y_train,
    epochs=ak_epochs,
    validation_data=(X_valid, y_valid),
)

y_prob = ak_clf.predict_proba(X_test)[:, 1]

### F1 Weighted
y_pred = ak_clf.predict(X_test)

## report metrics

from sklearn.metrics import roc_auc_score, average_precision_score

print("AUCROC:", roc_auc_score(y_score=y_prob, y_true=y_test))
print("AP:", roc_auc_score(y_score=y_prob, y_true=y_test))
