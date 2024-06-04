
# Model Tuner Documentation

## Table of Contents

I. [Overview](#overview)  
II. [Dependencies](#dependencies)  
III. [Key Methods and Functionalities](#key-methods-and-functionalities)  
IV. [Helper Functions](#helper-functions)  
V. [Notes](#notes)  
VI. [Usage](#usage)  
- [Example: Using ModelTuner with Logistic Regression on Iris Dataset](#example-using-modeltuner-with-logistic-regression-on-iris-dataset)  
- [Binary Classification](#binary-classification)  
  - [Breast Cancer Example with XGBoost](#breast-cancer-example-with-xgboost)  
    
VII. [Acknowledgements](#acknowledgements)   
VIII. [License](LICENSE.md)   


## Overview

The ModelTuner class is a versatile and powerful tool designed to facilitate the training, evaluation, and tuning of machine learning models. It supports various functionalities such as handling imbalanced data, applying different scaling and imputation techniques, calibrating models, and conducting cross-validation. This class is particularly useful for model selection and hyperparameter tuning, ensuring optimal performance across different metrics.

## Dependencies
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `tqdm`

## Key Methods and Functionalities

- `__init__(...)`: Initializes the ModelTuner with various configurations such as estimator, cross-validation settings, scoring metrics, etc.

- `reset_estimator()`: Resets the estimator.
- `process_imbalance_sampler(X_train, y_train)`: Processes imbalance sampler.
- `calibrateModel(X, y, score=None, stratify=None)`: Calibrates the model.
- `get_train_data(X, y)`, `get_valid_data(X, y)`, `get_test_data(X, y)`: Methods to retrieve train, validation, and test data.
- `calibrate_report(X, y, score=None)`: Generates a calibration report.
- `fit(X, y, validation_data=None, score=None)`: Fits the model to the data.
- `return_metrics(X_test, y_test)`: Returns evaluation metrics.
- `predict(X, y=None, optimal_threshold=False)`, `predict_proba(X, y=None)`: Methods to make predictions and predict probabilities.
- `grid_search_param_tuning(X, y, f1_beta_tune=False, betas=[1, 2])`: Performs grid search parameter tuning.
- `print_k_best_features(X)`: Prints the top K best features.
- `tune_threshold_Fbeta(score, X_train, y_train, X_valid, y_valid, betas, kfold=False)`: Tunes the threshold for F-beta score.
- `train_val_test_split(X, y, stratify_y, train_size, validation_size, test_size, random_state, stratify_cols, calibrate)`: Splits the data into train, validation, and test sets.
- `get_best_score_params(X, y)`: Retrieves the best score parameters.
- `conf_mat_class_kfold(X, y, test_model, score=None)`: Generates confusion matrix for k-fold cross-validation.
- `regression_report_kfold(X, y, test_model, score=None)`: Generates regression report for k-fold cross-validation.
- `regression_report(y_true, y_pred, print_results=True)`: Generates a regression report.

## Helper Functions

- `kfold_split(classifier, X, y, stratify=False, scoring=["roc_auc"], n_splits=10, random_state=3)`: Splits data using k-fold cross-validation.
- `get_cross_validate(classifier, X, y, kf, stratify=False, scoring=["roc_auc"])`: Performs cross-validation.
- `_confusion_matrix_print(conf_matrix, labels)`: Prints the confusion matrix.

## Notes
- This class is designed to be flexible and can be extended to include additional functionalities or custom metrics.
- It is essential to properly configure the parameters during initialization to suit the specific requirements of your machine learning task.
- Ensure that all dependencies are installed and properly imported before using the ModelTuner class.

## Usage

### Example: Using ModelTuner with Logistic Regression on Iris Dataset

```python

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

# Load the dataset
iris = load_iris()
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
features = [col for col in iris_df.columns if col != 'target']
target = 'target'

X = iris_df[features].values
y = iris_df[target].values.astype(int)

# Define the model and parameters
lr = LogisticRegression(class_weight="balanced", C=1, max_iter=1000)
estimator_name = "lr"
tuned_parameters = [{estimator_name + "__C": np.logspace(-4, 0, 10)}]

# Initialize ModelTuner
model_tuner = ModelTuner(
    name="Iris_model",
    estimator_name=estimator_name,
    calibrate=True,
    estimator=lr,
    kfold=True,
    stratify_y=True,
    grid=tuned_parameters,
    randomized_grid=False,
    n_iter=3,
    scoring=["roc_auc_ovr", "precision_macro"],
    n_splits=2,
    random_state=3
)

# Perform grid search parameter tuning
model_tuner.grid_search_param_tuning(X, y)

# Fit the model
model_tuner.fit(X, y)

# Calibrate the model
if model_tuner.calibrate:
    model_tuner.calibrateModel(X, y)

# Print results
if model_tuner.kfold:
    print(model_tuner.xval_output['train_score'], model_tuner.xval_output['test_score'])
else:
    importance = model_tuner.estimator.steps[1][1].coef_[0]
    sort_imp_indx = np.argsort(importance)[::-1]
    for i in sort_imp_indx:
        print(f"Feature: {features[i]}, Score: {importance[i]}")
```

### Binary Classification

Here is an example of using the ModelTuner class for binary classification using XGBoost on the Breast Cancer dataset.

#### Breast Cancer Example with XGBoost
##### Step 1: Import Necessary Libraries

```python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from ModelTuner import ModelTuner  

```

##### Step 2: Load the Dataset

```python
# Load the breast cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

```

##### Step 3: Create an Instance of the XGBClassifier
```python
# Creating an instance of the XGBClassifier
xgb_model = xgb.XGBClassifier(
    random_state=222,
)

```

##### Step 4: Define Hyperparameters for XGBoost

```python
# Estimator name prefix for use in GridSearchCV or similar tools
estimator_name_xgb = "xgb"

# Define the hyperparameters for XGBoost
xgb_learning_rates = [0.1, 0.01, 0.05][:1]  # Learning rate or eta
xgb_n_estimators = [100, 200, 300][:1]  # Number of trees. Equivalent to n_estimators in GB
xgb_max_depths = [3, 5, 7][:1]  # Maximum depth of the trees
xgb_subsamples = [0.8, 1.0][:1]  # Subsample ratio of the training instances
xgb_colsample_bytree = [0.8, 1.0][:1]

xgb_eval_metric = ["logloss"]  # Check out "pr_auc"
xgb_early_stopping_rounds = [10][:1]
xgb_verbose = [False]  # Subsample ratio of columns when constructing each tree

# Combining the hyperparameters in a dictionary
xgb_parameters = [
    {
        "xgb__learning_rate": xgb_learning_rates,
        "xgb__n_estimators": xgb_n_estimators,
        "xgb__max_depth": xgb_max_depths,
        "xgb__subsample": xgb_subsamples,
        "xgb__colsample_bytree": xgb_colsample_bytree,
        "xgb__eval_metric": xgb_eval_metric,
        "xgb__early_stopping_rounds": xgb_early_stopping_rounds,
        "xgb__verbose": xgb_verbose,
        "selectKBest__k": [5, 10, 20],
    }
]

```

##### Step 5: Initialize and Configure the ModelTuner

```python
# Initialize ModelTuner
model_tuner = ModelTuner(
    name="XGBoost_Breast_Cancer",
    estimator_name=estimator_name_xgb,
    calibrate=True,
    estimator=xgb_model,
    xgboost_early=True,
    kfold=False,
    impute=True,
    scaler_type=None,  # Turn off scaling for XGBoost
    selectKBest=True,
    stratify_cols=None,  # Assuming stratification is not required for this example
    grid=xgb_parameters,
    randomized_grid=False,
    scoring=["roc_auc"],
    random_state=222,
    n_jobs=-1,
)

```

##### Step 6: Perform Grid Search Parameter Tuning

```python
# Perform grid search parameter tuning
model_tuner.grid_search_param_tuning(X, y)

```

##### Step 7: Fit the Model

```python
# Get the training and validation data
X_train, y_train = model_tuner.get_train_data(X, y)
X_valid, y_valid = model_tuner.get_valid_data(X, y)

# Fit the model with the validation data
model_tuner.fit(
    X_train, y_train, validation_data=(X_valid, y_valid), score="roc_auc"
)

```

##### Step 8: Evaluate the Model

```python
# Return metrics for the validation set
metrics = model_tuner.return_metrics(X_valid, y_valid)
print(metrics)


```

##### Step 9: Calibrate the Model (if needed)

```python
# Calibrate the model
if model_tuner.calibrate:
    model_tuner.calibrateModel(X, y, score="roc_auc")

# Predict on the validation set
y_valid_pred = model_tuner.predict(X_valid)

```

## Acknowledgements

This work was supported by the UCLA Medical Informatics Institute (MII) and the Clinical and Translational Science Institute (CTSI). Special thanks to Dr. Alex Bui for his invaluable guidance and support, and to Panayiotis Petousis for his original contributions to this codebase.