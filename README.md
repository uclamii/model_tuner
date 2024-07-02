
![Model Tuner Logo](https://github.com/uclamii/model_tuner/blob/main/assets/modeltunertiny.png?raw=true)

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
- `train_val_test_split(X, y, stratify_y, train_size, validation_size, test_size, random_state, stratify_cols, calibrate)`: Splits the data into train, validation, and test sets.
- `get_best_score_params(X, y)`: Retrieves the best score parameters.

## Helper Functions

- `kfold_split(classifier, X, y, stratify=False, scoring=["roc_auc"], n_splits=10, random_state=3)`: Splits data using k-fold cross-validation.
- `get_cross_validate(classifier, X, y, kf, stratify=False, scoring=["roc_auc"])`: Performs cross-validation.
- `_confusion_matrix_print(conf_matrix, labels)`: Prints the confusion matrix.
- `tune_threshold_Fbeta(score, X_train, y_train, X_valid, y_valid, betas, kfold=False)`: Tunes the threshold for F-beta score.
- `conf_mat_class_kfold(X, y, test_model, score=None)`: Generates confusion matrix for k-fold cross-validation.
- `regression_report_kfold(X, y, test_model, score=None)`: Generates regression report for k-fold cross-validation.
- `regression_report(y_true, y_pred, print_results=True)`: Generates a regression report.

## Notes
- This class is designed to be flexible and can be extended to include additional functionalities or custom metrics.
- It is essential to properly configure the parameters during initialization to suit the specific requirements of your machine learning task.
- Ensure that all dependencies are installed and properly imported before using the ModelTuner class.

## Input Parameters

   - `name` (str): A name for the model, useful for identifying the model in outputs and logs.  
   - `estimator_name` (str): The prefix for the estimator used in the pipeline. This is used in parameter tuning (e.g., estimator_name + "__param_name").  
   - `estimator` (object): The machine learning model to be tuned and trained.
   - `calibrate` (bool, optional): Whether to calibrate the classifier. Default is False.
   - `kfold` (bool, optional): Whether to use k-fold cross-validation. Default is False.
   - `imbalance_sampler` (object, optional): An imbalanced data sampler from the imblearn library, e.g., RandomUnderSampler or RandomOverSampler.
   - `train_size` (float, optional): Proportion of the data to use for training. Default is 0.6.
   - `validation_size` (float, optional): Proportion of the data to use for validation. Default is 0.2.
   - `test_size` (float, optional): Proportion of the data to use for testing. Default is 0.2.
   - `stratify_y` (bool, optional): Whether to stratify by the target variable during train/validation/test split. Default is False.
   - `stratify_cols` (list, optional): List of columns to stratify by during train/validation/test split. Default is None.
   - `drop_strat_feat` (list, optional): List of columns to drop after stratification. Default is None.
   - `grid` (list of dict): Hyperparameter grid for tuning.
   - `scoring` (list of str): Scoring metrics for evaluation.
   - `n_splits` (int, optional): Number of splits for k-fold cross-validation. Default is 10.
   - `random_state` (int, optional): Random state for reproducibility. Default is 3.
   - `n_jobs` (int, optional): Number of jobs to run in parallel for model fitting. Default is 1.
   - `display` (bool, optional): Whether to display output messages during the tuning process. Default is True.
   - `feature_names` (list, optional): List of feature names. Default is None.
   - `randomized_grid` (bool, optional): Whether to use randomized grid search. Default is False.
   - `n_iter` (int, optional): Number of iterations for randomized grid search. Default is 100.
   - `trained` (bool, optional): Whether the model has been trained. Default is False.
   - `pipeline` (bool, optional): Whether to use a pipeline. Default is True.
   - `scaler_type` (str, optional): Type of scaler to use. Options are "min_max_scaler", "standard_scaler", "max_abs_scaler", or None. Default is "min_max_scaler".
   - `impute_strategy` (str, optional): Strategy for imputation. Options are "mean", "median", "most_frequent", or "constant". Default is "mean".
   - `impute` (bool, optional): Whether to impute missing values. Default is False.
   - `pipeline_steps` (list, optional): List of pipeline steps. Default is [("min_max_scaler", MinMaxScaler())].
   - `xgboost_early` (bool, optional): Whether to use early stopping for XGBoost. Default is False.
   - `selectKBest` (bool, optional): Whether to select K best features. Default is False.
   - `model_type` (str, optional): Type of model, either "classification" or "regression". Default is "classification".
   - `class_labels` (list, optional): List of class labels for multi-class classification. Default is None.
   - `multi_label` (bool, optional): Whether the problem is a multi-label classification problem. Default is False.
   - `calibration_method` (str, optional): Method for calibration, options are "sigmoid" or "isotonic". Default is "sigmoid".
   - `custom_scorer` (dict, optional): Custom scorers for evaluation. Default is [].

## Usage

### Binary Classification

Here is an example of using the `ModelTuner` class for binary classification using XGBoost on the Breast Cancer dataset.

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
xgb_learning_rates = [0.1, 0.01, 0.05]  # Learning rate or eta
xgb_n_estimators = [100, 200, 300]  # Number of trees. Equivalent to n_estimators in GB
xgb_max_depths = [3, 5, 7]  # Maximum depth of the trees
xgb_subsamples = [0.8, 1.0]  # Subsample ratio of the training instances
xgb_colsample_bytree = [0.8, 1.0]

xgb_eval_metric = ["logloss"]  # Check out "pr_auc"
xgb_early_stopping_rounds = [10]
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
model_tuner = Model(
    name="XGBoost_Breast_Cancer",
    estimator_name=estimator_name_xgb,
    calibrate=True,
    estimator=xgb_model,
    xgboost_early=True,
    kfold=False,
    impute=True,
    scaler_type=None,  # Turn off scaling for XGBoost
    selectKBest=True,
    stratify_y=False,
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

##### Step 8: Return Metrics (Optional)

You can use this function to evaluate the model by printing the output.

```python

# Return metrics for the validation set
metrics = model_tuner.return_metrics(
    X_valid,
    y_valid,
)
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


##### Output:

```
100%|██████████| 324/324 [15:39<00:00,  2.90s/it]
Best score/param set found on validation set:
{'params': {'selectKBest__k': 20,
            'xgb__colsample_bytree': 0.8,
            'xgb__early_stopping_rounds': 10,
            'xgb__eval_metric': 'logloss',
            'xgb__learning_rate': 0.1,
            'xgb__max_depth': 3,
            'xgb__n_estimators': 200,
            'xgb__subsample': 0.8,
            'xgb__verbose': False},
 'score': 0.9987212276214834}
Best roc_auc: 0.999 

Confusion matrix on validation set: 
--------------------------------------------------------------------------------
          Predicted:
            Pos  Neg
--------------------------------------------------------------------------------
Actual: Pos 46 (tp)   0 (fn)
        Neg  3 (fp)  65 (tn)
--------------------------------------------------------------------------------

              precision    recall  f1-score   support

           0       0.94      1.00      0.97        46
           1       1.00      0.96      0.98        68

    accuracy                           0.97       114
   macro avg       0.97      0.98      0.97       114
weighted avg       0.98      0.97      0.97       114

--------------------------------------------------------------------------------

Feature names selected:
['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean compactness', 'mean concavity', 'mean concave points', 'radius error', 'perimeter error', 'area error', 'concavity error', 'concave points error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points']

{'Classification Report': {'0': {'precision': 0.9387755102040817, 'recall': 1.0, 'f1-score': 0.968421052631579, 'support': 46.0}, '1': {'precision': 1.0, 'recall': 0.9558823529411765, 'f1-score': 0.9774436090225563, 'support': 68.0}, 'accuracy': 0.9736842105263158, 'macro avg': {'precision': 0.9693877551020409, 'recall': 0.9779411764705883, 'f1-score': 0.9729323308270676, 'support': 114.0}, 'weighted avg': {'precision': 0.9752953813104189, 'recall': 0.9736842105263158, 'f1-score': 0.9738029283735655, 'support': 114.0}}, 'Confusion Matrix': array([[46,  0],
       [ 3, 65]]), 'K Best Features': ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean compactness', 'mean concavity', 'mean concave points', 'radius error', 'perimeter error', 'area error', 'concavity error', 'concave points error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points']}
Confusion matrix on validation set for roc_auc
--------------------------------------------------------------------------------
          Predicted:
            Pos  Neg
--------------------------------------------------------------------------------
Actual: Pos 46 (tp)   0 (fn)
        Neg  3 (fp)  65 (tn)
--------------------------------------------------------------------------------

              precision    recall  f1-score   support

           0       0.94      1.00      0.97        46
           1       1.00      0.96      0.98        68

    accuracy                           0.97       114
   macro avg       0.97      0.98      0.97       114
weighted avg       0.98      0.97      0.97       114

--------------------------------------------------------------------------------
roc_auc after calibration: 0.9987212276214834
```


### Regression

Here is an example of using the `ModelTuner` class for regression using XGBoost on the California Housing dataset.

#### California Housing with XGBoost
##### Step 1: Import Necessary Libraries

```python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from ModelTuner import ModelTuner  

```

##### Step 2: Load the Dataset

```python
# Load the California Housing dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")
```

##### Step 3: Create an Instance of the XGBClassifier

```python
# Creating an instance of the XGBRegressor
xgb_model = xgb.XGBRegressor(
    random_state=222,
)
```

##### Step 4: Define Hyperparameters for XGBoost

```python
# Estimator name prefix for use in GridSearchCV or similar tools
estimator_name_xgb = "xgb"

# Define the hyperparameters for XGBoost
xgb_learning_rates = [0.1, 0.01, 0.05]
xgb_n_estimators = [100, 200, 300]
xgb_max_depths = [3, 5, 7]
xgb_subsamples = [0.8, 1.0]
xgb_colsample_bytree = [0.8, 1.0]

# Combining the hyperparameters in a dictionary
xgb_parameters = [
    {
        "xgb__learning_rate": xgb_learning_rates,
        "xgb__n_estimators": xgb_n_estimators,
        "xgb__max_depth": xgb_max_depths,
        "xgb__subsample": xgb_subsamples,
        "xgb__colsample_bytree": xgb_colsample_bytree,
        "selectKBest__k": [1, 3, 5, 8],
    }
]

```

##### Step 5: Initialize and Configure the ModelTuner

```python
# Initialize ModelTuner
model_tuner = Model(
    name="XGBoost_California_Housing",
    model_type="regression",
    estimator_name=estimator_name_xgb,
    calibrate=False,
    estimator=xgb_model,
    kfold=False,
    impute=True,
    scaler_type=None,
    selectKBest=True,
    stratify_y=False,
    grid=xgb_parameters,
    randomized_grid=False,
    scoring=["neg_mean_squared_error"],
    random_state=222,
    n_jobs=-1,
)
```

##### Step 7: Fit the Model

```python
# Get the training and validation data
X_train, y_train = model_tuner.get_train_data(X, y)
X_valid, y_valid = model_tuner.get_valid_data(X, y)

# Fit the model with the validation data
model_tuner.fit(
    X_train, y_train, validation_data=(X_valid, y_valid), 
    score="neg_mean_squared_error",
)
```


##### Step 8: Return Metrics (Optional)

You can use this function to evaluate the model by printing the output.

```python

# Return metrics for the validation set
metrics = model_tuner.return_metrics(
    X_valid,
    y_valid,
)
print(metrics)

```

##### Output:

```
100%|██████████| 432/432 [04:10<00:00,  1.73it/s]
Best score/param set found on validation set:
{'params': {'selectKBest__k': 8,
            'xgb__colsample_bytree': 0.8,
            'xgb__learning_rate': 0.05,
            'xgb__max_depth': 7,
            'xgb__n_estimators': 300,
            'xgb__subsample': 0.8},
 'score': -0.21038206511437127}
Best neg_mean_squared_error: -0.210 

********************************************************************************
{'Explained Variance': 0.8385815985957561,
 'Mean Absolute Error': 0.3008222037008959,
 'Mean Squared Error': 0.21038206511437127,
 'Median Absolute Error': 0.196492121219635,
 'R2': 0.8385811859863378,
 'RMSE': 0.45867424727618106}
********************************************************************************

Feature names selected:
['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

{'Regression Report': {'Explained Variance': 0.8385815985957561, 'R2': 0.8385811859863378, 'Mean Absolute Error': 0.3008222037008959, 'Median Absolute Error': 0.196492121219635, 'Mean Squared Error': 0.21038206511437127, 'RMSE': 0.45867424727618106}, 'K Best Features': ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']}

```





## Acknowledgements

This work was supported by the UCLA Medical Informatics Institute (MII) and the Clinical and Translational Science Institute (CTSI). Special thanks to Dr. Alex Bui for his invaluable guidance and support, and to Panayiotis Petousis for his original contributions to this codebase.
