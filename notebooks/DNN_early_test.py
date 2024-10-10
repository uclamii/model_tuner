import pandas as pd
import numpy as np
import os
import random

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

import tensorflow as tf
import autokeras as ak

from model_tuner.model_tuner_utils import AutoKerasClassifier

from sklearn.metrics import roc_auc_score, average_precision_score

# Set seeds for reproducibility
os.environ["PYTHONHASHSEED"] = str(222)
random.seed(222)
np.random.seed(222)
tf.random.set_seed(222)


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


# Load data
bc = load_breast_cancer(as_frame=True)["frame"]
bc_cols = [col for col in bc.columns if "target" not in col]
X = bc[bc_cols]
y = bc["target"]

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

X_train, X_valid, X_test, y_train, y_valid, y_test = (
    X_train.values,
    X_valid.values,
    X_test.values,
    y_train.values,
    y_valid.values,
    y_test.values,
)

# Preprocessing pipeline (done outside AutoKeras)
preprocessing_pipeline = Pipeline(
    [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)


print(np.bincount(y_valid))

ak_max_trials = 1
ak_epochs = 10

import keras
import kerastuner

import autokeras as ak

from tensorflow.keras import backend as K

# https://github.com/keras-team/autokeras/blob/master/docs/templates/tutorial/faq.md
# https://github.com/keras-team/autokeras/issues/867


def recall_m(y_true, y_pred):
    # Cast y_true and y_pred to float32 to avoid data type mismatches
    y_true = K.cast(y_true, "float32")
    y_pred = K.cast(y_pred, "float32")
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    # Cast y_true and y_pred to float32 to avoid data type mismatches
    y_true = K.cast(y_true, "float32")
    y_pred = K.cast(y_pred, "float32")
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_score(y_true, y_pred):
    # Cast y_true and y_pred to float32 to avoid data type mismatches
    y_true = K.cast(y_true, "float32")
    y_pred = K.cast(y_pred, "float32")
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


opt = keras.optimizers.Adam(learning_rate=0.01)

# Initialize the input and output
auto_AK_model = ak.AutoModel(
    inputs=[ak.Input()],
    outputs=[
        ak.ClassificationHead(
            loss="binary_crossentropy",
            # metrics=["f1_score"],
            metrics=[tf.keras.metrics.AUC(name="auc")],  # Adding AUC-ROC as a metric
            # metrics=["auc"],  # Adding AUC-ROC as a metric
        ),
    ],
    # loss="binary_crossentropy",
    # objective=kerastuner.Objective("val_f1_score", direction="max"),
    objective=kerastuner.Objective(
        "val_auc", direction="max"
    ),  # Optimizing for AUC-ROC
    # tuner="greedy",
    tuner="bayesian",
    # tuner="hyperband",
    overwrite=True,
    max_trials=ak_max_trials,
    max_model_size=100000,
    seed=222,
)


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)
X__train_resampled, y__train_resampled = ros.fit_resample(X_train, y_train)


ak_clf = AutoKerasClassifier(
    auto_AK_model,
    pipeline=Pipeline(
        [
            ("impute", SimpleImputer()),
            ("scaler", StandardScaler()),
        ]
    ),
)


ak_clf.fit(
    X__train_resampled,
    y__train_resampled,
    epochs=ak_epochs,
    batch_size=10,
    validation_data=(
        [X_valid],
        [y_valid],
    ),
    verbose=1,
)

ak_clf.summarize_auto_keras_params()

y_prob = ak_clf.predict_proba(X_test)[:, 1]

print(y_prob)

### F1 Weighted
y_pred = ak_clf.predict(X_test, threshold=0.5)

## report metrics

from sklearn.metrics import roc_auc_score, average_precision_score

print("AUCROC:", roc_auc_score(y_score=y_prob, y_true=y_test))
print("AP:", roc_auc_score(y_score=y_prob, y_true=y_test))


ak_clf.save_model(
    model_name="auto_model/best_test_model.keras",
    pipeline_name="auto_model/best_sk_pipeline",
)

ak_clf_loaded = AutoKerasClassifier()

ak_clf_loaded.load_saved_model(
    model_name="auto_model/best_test_model.keras",
    pipeline_name="auto_model/best_sk_pipeline",
)


y_prob = ak_clf_loaded.predict_proba(X_test)[:, 1]

print(y_prob)

### F1 Weighted
y_pred = ak_clf_loaded.predict(X_test, threshold=0.5)

## report metrics

from sklearn.metrics import roc_auc_score, average_precision_score

print("AUCROC:", roc_auc_score(y_score=y_prob, y_true=y_test))
print("AP:", roc_auc_score(y_score=y_prob, y_true=y_test))
