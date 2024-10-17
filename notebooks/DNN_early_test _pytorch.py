import pandas as pd
import numpy as np
import os
import random

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, average_precision_score

from model_tuner.pytorch_model_tuner import *


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


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)
X__train_resampled, y__train_resampled = ros.fit_resample(X_train, y_train)

# Create DataLoaders
train_loader, val_loader, test_loader = create_dataloaders(
    X__train_resampled,
    y__train_resampled,
    X_valid,
    y_valid,
    X_test,
    y_test,
    batch_size=32,
)

print(train_loader.batch_size)
print(train_loader.dataset[0][0].shape)
import torch

# for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # if using multiple GPUs


print(torch.cuda.is_available())


from sklearn.metrics import roc_auc_score

# Define the metric for optimization
# metric_to_optimize = roc_auc_score  # "val_loss"  # Can be 'val_loss' or 'val_accuracy'
metric_to_optimize = "val_loss"  # Can be 'val_loss' or 'val_accuracy'

from pytorch_lightning.callbacks import ModelCheckpoint

# Define a ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",  # Metric to monitor
    dirpath="checkpoints/",  # Directory to save checkpoints
    filename="model-{epoch:02d}",  # Use only epoch number in filename
    save_top_k=1,  # Saves the top k models, here it saves the best
    mode="min",  # 'min' for metrics where lower is better (like loss), 'max' for higher is better (like accuracy)
)

# Run the Optuna study
direction = "minimize" if metric_to_optimize == "val_loss" else "maximize"
study = optuna.create_study(
    direction=direction,
    sampler=optuna.samplers.RandomSampler(seed=42),
)
study.optimize(
    lambda trial: objective(
        trial,
        train_loader,
        val_loader,
        metric_name=metric_to_optimize,
        checkpoint_callback=checkpoint_callback,
    ),
    n_trials=1,
)

print("Best hyperparameters: ", study.best_params)

# Use the best checkpoint path to load the model
best_checkpoint_path = checkpoint_callback.best_model_path
print(f"The best model is saved at: {best_checkpoint_path}")

best_model = LitModel.load_from_checkpoint(
    best_checkpoint_path,
    # "checkpoints/model-epoch=08.ckpt",
    # input_dim=train_loader.dataset[0][0].shape,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_model.to(device)  # Move model to the specified device

print(best_model)

# If you want to handle the predictions manually (e.g., in a loop)
probability_predictions = []
for batch in test_loader:
    x, _ = batch  # Usually, the test loader might have dummy targets
    x = x.to(device)
    with torch.no_grad():  # Disable gradient calculations
        probs = best_model(x)  # Forward pass
        probability_predictions.append(
            probs.cpu()
        )  # Collect predictions and move to CPU if needed

# Optionally concatenate all the results into a single tensor
all_probabilities = torch.cat(probability_predictions, dim=0)

# Convert to numpy for further processing
all_probabilities_numpy = all_probabilities.numpy()

print(all_probabilities)

# Convert to numpy for further processing (optional)
y_prob = all_probabilities.cpu().numpy()

from sklearn.metrics import roc_auc_score, average_precision_score

print("AUCROC:", roc_auc_score(y_score=y_prob, y_true=y_test))
print("AP:", roc_auc_score(y_score=y_prob, y_true=y_test))
