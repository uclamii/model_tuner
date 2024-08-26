.. _usage_guide:

.. _target-link:

.. raw:: html

   <div class="no-click">

.. image:: /../assets/ModelTunerTarget.png
   :alt: Model Tuner Logo
   :align: left
   :width: 250px

.. raw:: html

   </div>

.. raw:: html

   <div style="height: 150px;"></div>

\



iPython Notebooks
===================

- `Binary Classification Example <https://colab.research.google.com/drive/1D9nl8rLdwxPEpiZplsU0I0lFSAec7NzP?authuser=1#scrollTo=tumIjsNpSAKC&uniqifier=1>`_  

- `Column Transformer Example <https://colab.research.google.com/drive/1ujLL2mRtIWwGamnpWKIo2f271_Q103t-?usp=sharing#scrollTo=uMxyy0yvd2xQ>`_

- `Regression Example <https://colab.research.google.com/drive/151kdlsW-WyJ0pwwt_iWpjXDuqj1Ktam_?authuser=1#scrollTo=UhfZKVoq3sAN>`_


Key Methods and Functionalities
========================================

``__init__(...)``
    Initializes the model_tuner with configurations such as estimator, cross-validation settings, scoring metrics, etc.

``reset_estimator()``
    Resets the estimator.

``process_imbalance_sampler(X_train, y_train)``
    Processes imbalance sampler.

``calibrateModel(X, y, score=None, stratify=None)``
    Calibrates the model.

``get_train_data(X, y), get_valid_data(X, y), get_test_data(X, y)``
    Retrieves train, validation, and test data.

``calibrate_report(X, y, score=None)``
    Generates a calibration report.

``fit(X, y, validation_data=None, score=None)``
    Fits the model to the data.

``return_metrics(X_test, y_test)``
    Returns evaluation metrics.

``predict(X, y=None, optimal_threshold=False), predict_proba(X, y=None)``
    Makes predictions and predicts probabilities.

``grid_search_param_tuning(X, y, f1_beta_tune=False, betas=[1, 2])``
    Performs grid search parameter tuning.

``print_k_best_features(X)``
    Prints the top K best features.

``tune_threshold_Fbeta(score, X_train, y_train, X_valid, y_valid, betas, kfold=False)``
    Tunes the threshold for F-beta score.

``train_val_test_split(X, y, stratify_y, train_size, validation_size, test_size, random_state, stratify_cols, calibrate)``
    Splits the data into train, validation, and test sets.

``get_best_score_params(X, y)``
    Retrieves the best score parameters.

``conf_mat_class_kfold(X, y, test_model, score=None)``
    Generates confusion matrix for k-fold cross-validation.

``regression_report_kfold(X, y, test_model, score=None)``
    Generates regression report for k-fold cross-validation.

``regression_report(y_true, y_pred, print_results=True)``
    Generates a regression report.


Helper Functions
=================

``kfold_split(classifier, X, y, stratify=False, scoring=["roc_auc"], n_splits=10, random_state=3)`` 
      Splits data using k-fold cross-validation.

``get_cross_validate(classifier, X, y, kf, stratify=False, scoring=["roc_auc"])``
      Performs cross-validation.

``_confusion_matrix_print(conf_matrix, labels)``
      Prints the confusion matrix.


.. note::

   - This class is designed to be flexible and can be extended to include additional functionalities or custom metrics.
   - It is essential to properly configure the parameters during initialization to suit the specific requirements of your machine learning task.
   - Ensure that all dependencies are installed and properly imported before using the ``Model`` class from the ``model_tuner`` library.

Input Parameters
=====================

.. function:: Model(name, estimator_name, estimator, calibrate, kfold, imbalance_sampler, train_size, validation_size, test_size, stratify_y, stratify_cols, drop_strat_feat, grid, scoring, n_splits, random_state, n_jobs, display, feature_names, randomized_grid, n_iter, trained, pipeline, scaler_type, impute_strategy, impute, pipeline_steps, xgboost_early, selectKBest, model_type, class_labels, multi_label, calibration_method, custom_scorer)

   :param name: A name for the model, useful for identifying the model in outputs and logs.
   :type name: str
   :param estimator_name: The prefix for the estimator used in the pipeline. This is used in parameter tuning (e.g., estimator_name + ``__param_name``).
   :type estimator_name: str
   :param estimator: The machine learning model to be tuned and trained.
   :type estimator: object
   :param calibrate: Whether to calibrate the classifier. Default is False.
   :type calibrate: bool, optional
   :param kfold: Whether to use k-fold cross-validation. Default is False.
   :type kfold: bool, optional
   :param imbalance_sampler: An imbalanced data sampler from the imblearn library, e.g., ``RandomUnderSampler`` or ``RandomOverSampler``.
   :type imbalance_sampler: object, optional
   :param train_size: Proportion of the data to use for training. Default is 0.6.
   :type train_size: float, optional
   :param validation_size: Proportion of the data to use for validation. Default is 0.2.
   :type validation_size: float, optional
   :param test_size: Proportion of the data to use for testing. Default is 0.2.
   :type test_size: float, optional
   :param stratify_y: Whether to stratify by the target variable during train/validation/test split. Default is ``False``.
   :type stratify_y: bool, optional
   :param stratify_cols: List of columns to stratify by during train/validation/test split. Default is ``None``.
   :type stratify_cols: list, optional
   :param drop_strat_feat: List of columns to drop after stratification. Default is ``None``.
   :type drop_strat_feat: list, optional
   :param grid: Hyperparameter grid for tuning.
   :type grid: list of dict
   :param scoring: Scoring metrics for evaluation.
   :type scoring: list of str
   :param n_splits: Number of splits for k-fold cross-validation. Default is ``10``.
   :type n_splits: int, optional
   :param random_state: Random state for reproducibility. Default is ``3``.
   :type random_state: int, optional
   :param n_jobs: Number of jobs to run in parallel for model fitting. Default is ``1``.
   :type n_jobs: int, optional
   :param display: Whether to display output messages during the tuning process. Default is ``True``.
   :type display: bool, optional
   :param feature_names: List of feature names. Default is ``None``.
   :type feature_names: list, optional
   :param randomized_grid: Whether to use randomized grid search. Default is ``False``.
   :type randomized_grid: bool, optional
   :param n_iter: Number of iterations for randomized grid search. Default is ``100``.
   :type n_iter: int, optional
   :param trained: Whether the model has been trained. Default is ``False``.
   :type trained: bool, optional
   :param pipeline: Whether to use a pipeline. Default is ``True``.
   :type pipeline: bool, optional
   :param scaler_type: Type of scaler to use. Options are ``min_max_scaler``, ``standard_scaler``, ``max_abs_scaler``, or ``None``. Default is ``min_max_scaler``.
   :type scaler_type: str, optional
   :param impute_strategy: Strategy for imputation. Options are ``mean``, ``median``, ``most_frequent``, or ``constant``. Default is ``mean``.
   :type impute_strategy: str, optional
   :param impute: Whether to impute missing values. Default is ``False``.
   :type impute: bool, optional
   :param pipeline_steps: List of pipeline steps. Default is ``[(min_max_scaler, MinMaxScaler())]``.
   :type pipeline_steps: list, optional
   :param xgboost_early: Whether to use early stopping for ``XGBoost``. Default is ``False``.
   :type xgboost_early: bool, optional
   :param selectKBest: Whether to select K best features. Default is ``False``.
   :type selectKBest: bool, optional
   :param model_type: Type of model, either ``classification`` or ``regression``. Default is ``classification``.
   :type model_type: str, optional
   :param class_labels: List of class labels for multi-class classification. Default is ``None``.
   :type class_labels: list, optional
   :param multi_label: Whether the problem is a multi-label classification problem. Default is ``False``.
   :type multi_label: bool, optional
   :param calibration_method: Method for calibration, options are ``sigmoid`` or ``isotonic``. Default is ``sigmoid``.
   :type calibration_method: str, optional
   :param custom_scorer: Custom scorers for evaluation. Default is ``[]``.
   :type custom_scorer: dict, optional


   :raises ImportError: If the ``bootstrapper`` module is not found or not installed.
   :raises ValueError: In various cases, such as when an invalid parameter is passed to Scikit-learn functions like ``cross_validate``, ``fit``, or ``train_test_split``, or if the shapes of ``X`` and ``y`` do not match during operations.
   :raises AttributeError: If an expected step in the pipeline (e.g., "imputer", "Resampler") is missing from ``self.estimator.named_steps``, or if ``self.PipelineClass`` or ``self.estimator`` is not properly initialized.
   :raises TypeError: If an incorrect type is passed to a function or method, such as passing ``None`` where a numerical value or a non-NoneType object is expected.
   :raises IndexError: If the dimensions of the confusion matrix are incorrect or unexpected in ``_confusion_matrix_print_ML`` or ``_confusion_matrix_print``.
   :raises KeyError: If a key is not found in a dictionary, such as when accessing ``self.best_params_per_score`` with a score that is not in the dictionary, or when accessing configuration keys in the ``summarize_auto_keras_params`` method.
   :raises RuntimeError: If there is an unexpected issue during model fitting or transformation that does not fit into the other categories of exceptions.

Model Calibration
==================

Model calibration refers to the process of adjusting the predicted probabilities of a model so that they more accurately reflect the true likelihood of outcomes. This is crucial in machine learning, particularly for classification problems where the model outputs probabilities rather than just class labels.

Goal of Calibration
--------------------

The goal of calibration is to ensure that the predicted probability :math:`\hat{p}(x)` is equal to the true probability that :math:`y = 1` given :math:`x`. Mathematically, this can be expressed as:

.. math::

    \hat{p}(x) = P(y = 1 \mid \hat{p}(x) = p)

This equation states that for all instances where the model predicts a probability :math:`p`, the true fraction of positive cases should also be :math:`p`.

Calibration Curve
------------------

To assess calibration, we often use a *calibration curve*. This involves:

1. **Binning** the predicted probabilities :math:`\hat{p}(x)` into intervals (e.g., [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]).
2. **Calculating the mean predicted probability** :math:`\hat{p}_i` for each bin :math:`i`.
3. **Calculating the empirical frequency** :math:`f_i` (the fraction of positives) in each bin.

For a perfectly calibrated model:

.. math::

    \hat{p}_i = f_i \quad \text{for all bins } i

Brier Score
------------

The **Brier score** is one way to measure the calibration of a model. It’s calculated as:

.. math::

    \text{Brier Score} = \frac{1}{N} \sum_{i=1}^{N} (\hat{p}(x_i) - y_i)^2

Where:

- :math:`N` is the number of instances.
- :math:`\hat{p}(x_i)` is the predicted probability for instance :math:`i`.
- :math:`y_i` is the actual label for instance :math:`i` (0 or 1).

The Brier score penalizes predictions that are far from the true outcome. A lower Brier score indicates better calibration and accuracy.

Platt Scaling
--------------

One common method to calibrate a model is **Platt Scaling**. This involves fitting a logistic regression model to the predictions of the original model. The logistic regression model adjusts the raw predictions :math:`\hat{p}(x)` to output calibrated probabilities.

Mathematically, Platt scaling is expressed as:

.. math::

    \hat{p}_{\text{calibrated}}(x) = \frac{1}{1 + \exp(-(A \hat{p}(x) + B))}

Where :math:`A` and :math:`B` are parameters learned from the data. These parameters adjust the original probability estimates to better align with the true probabilities.

Isotonic Regression
--------------------

Another method is **Isotonic Regression**, a non-parametric approach that fits a piecewise constant function. Unlike Platt Scaling, which assumes a logistic function, Isotonic Regression only assumes that the function is monotonically increasing. The goal is to find a set of probabilities :math:`p_i` that are as close as possible to the true probabilities while maintaining a monotonic relationship.

The isotonic regression problem can be formulated as:

.. math::

    \min_{p_1 \leq p_2 \leq \dots \leq p_n} \sum_{i=1}^{n} (p_i - y_i)^2

Where :math:`p_i` are the adjusted probabilities, and the constraint ensures that the probabilities are non-decreasing.

Example: Calibration in Logistic Regression
---------------------------------------------

In a standard logistic regression model, the predicted probability is given by:

.. math::

    \hat{p}(x) = \sigma(w^\top x) = \frac{1}{1 + \exp(-w^\top x)}

Where :math:`w` is the vector of weights, and :math:`x` is the input feature vector.

If this model is well-calibrated, :math:`\hat{p}(x)` should closely match the true conditional probability :math:`P(y = 1 \mid x)`. If not, techniques like Platt Scaling or Isotonic Regression can be applied to adjust :math:`\hat{p}(x)` to be more accurate.

Summary
--------

- **Model calibration** is about aligning predicted probabilities with actual outcomes.
- **Mathematically**, calibration ensures :math:`\hat{p}(x) = P(y = 1 \mid \hat{p}(x) = p)`.
- **Platt Scaling** and **Isotonic Regression** are two common methods to achieve calibration.
- **Brier Score** is a metric that captures both the calibration and accuracy of probabilistic predictions.

Calibration is essential when the probabilities output by a model need to be trusted, such as in risk assessment, medical diagnosis, and other critical applications.


Binary Classification
======================

Binary classification is a type of supervised learning where a model is trained 
to distinguish between two distinct classes or categories. In essence, the model 
learns to classify input data into one of two possible outcomes, typically 
labeled as ``0`` and ``1``, or negative and positive. This is commonly used in 
scenarios such as spam detection, disease diagnosis, or fraud detection.

In our library, binary classification is handled seamlessly through the ``Model`` 
class. Users can specify a binary classifier as the estimator, and the library 
takes care of essential tasks like data preprocessing, model calibration, and 
cross-validation. The library also provides robust support for evaluating the 
model's performance using a variety of metrics, such as accuracy, precision, 
recall, and ROC-AUC, ensuring that the model's ability to distinguish between the 
two classes is thoroughly assessed. Additionally, the library supports advanced 
techniques like imbalanced data handling and model calibration to fine-tune 
decision thresholds, making it easier to deploy effective binary classifiers in 
real-world applications.


AIDS Clinical Trials Group Study
---------------------------------

The UCI Machine Learning Repository is a well-known resource for accessing a wide 
range of datasets used for machine learning research and practice. One such dataset 
is the AIDS Clinical Trials Group Study dataset, which can be used to build and 
evaluate predictive models.

You can easily fetch this dataset using the ucimlrepo package. If you haven't 
installed it yet, you can do so by running the following command:

.. code-block:: bash
   
   pip install ucimlrepo


Once installed, you can quickly load the AIDS Clinical Trials Group Study dataset 
with a simple command:

.. code-block:: python

    from ucimlrepo import fetch_ucirepo 

Step 1: Import Necessary Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import pandas as pd
    import numpy as np
    import xgboost as xgb


Step 2: Load the dataset, define X, y
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # fetch dataset 
   aids_clinical_trials_group_study_175 = fetch_ucirepo(id=890) 
   
   # data (as pandas dataframes) 
   X = aids_clinical_trials_group_study_175.data.features 
   y = aids_clinical_trials_group_study_175.data.targets 
   y = y.squeeze() # convert a DataFrame to Series when single column


Step 3: Check for zero-variance columns and drop accordingly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Check for zero-variance columns and drop them
   zero_variance_columns = X.columns[X.var() == 0]
   if not zero_variance_columns.empty:
      X = X.drop(columns=zero_variance_columns)


Step 3: Create an Instance of the XGBClassifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Creating an instance of the XGBClassifier
   xgb_model = xgb.XGBClassifier(
      random_state=222,
   )

Step 4: Define Hyperparameters for XGBoost
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

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


Step 5: Initialize and Configure the ``Model``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Initialize model_tuner
   model_tuner = Model(
      name="XGBoost_AIDS",
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

Step 6: Perform Grid Search Parameter Tuning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Perform grid search parameter tuning
   model_tuner.grid_search_param_tuning(X, y)

.. code-block:: bash

   100%|██████████| 324/324 [01:36<00:00,  3.37it/s]
   Best score/param set found on validation set:
   {'params': {'selectKBest__k': 4,
               'xgb__colsample_bytree': 1.0,
               'xgb__early_stopping_rounds': 10,
               'xgb__eval_metric': 'logloss',
               'xgb__learning_rate': 0.01,
               'xgb__max_depth': 3,
               'xgb__n_estimators': 199,
               'xgb__subsample': 0.8},
   'score': 0.9364314448541736}
   Best roc_auc: 0.936 

Step 7: Fit the Model
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Get the training and validation data
   X_train, y_train = model_tuner.get_train_data(X, y)
   X_valid, y_valid = model_tuner.get_valid_data(X, y)

   # Fit the model with the validation data
   model_tuner.fit(
      X_train,
      y_train,
      validation_data=(X_valid, y_valid),
      score="roc_auc",
   )

Step 8: Return Metrics (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use this function to evaluate the model by printing the output.

.. code-block:: python

   # Return metrics for the validation set
   metrics = model_tuner.return_metrics(
      X_valid,
      y_valid,
   )
   print(metrics)

.. code-block:: bash

   Confusion matrix on set provided: 
   --------------------------------------------------------------------------------
            Predicted:
               Pos   Neg
   --------------------------------------------------------------------------------
   Actual: Pos 291 (tp)   23 (fn)
         Neg  31 (fp)   83 (tn)
   --------------------------------------------------------------------------------

               precision    recall  f1-score   support

            0       0.90      0.93      0.92       314
            1       0.78      0.73      0.75       114

      accuracy                           0.87       428
      macro avg       0.84      0.83      0.83       428
   weighted avg       0.87      0.87      0.87       428

   --------------------------------------------------------------------------------

   Feature names selected:
   ['time', 'strat', 'cd40', 'cd420']

   {'Classification Report': {'0': {'precision': 0.9037267080745341,
      'recall': 0.9267515923566879,
      'f1-score': 0.9150943396226415,
      'support': 314.0},
   '1': {'precision': 0.7830188679245284,
      'recall': 0.7280701754385965,
      'f1-score': 0.7545454545454546,
      'support': 114.0},
   'accuracy': 0.8738317757009346,
   'macro avg': {'precision': 0.8433727879995312,
      'recall': 0.8274108838976422,
      'f1-score': 0.8348198970840481,
      'support': 428.0},
   'weighted avg': {'precision': 0.8715755543897196,
      'recall': 0.8738317757009346,
      'f1-score': 0.8723313188310543,
      'support': 428.0}},
   'Confusion Matrix': array([[291,  23],
         [ 31,  83]]),
   'K Best Features': ['time', 'strat', 'cd40', 'cd420']}   

Step 9: Calibrate the Model (if needed)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sklearn.calibration import calibration_curve

   # Get the predicted probabilities for the validation data from the 
   # uncalibrated model
   y_prob_uncalibrated = model_tuner.predict_proba(X_test)[:, 1]

   # Compute the calibration curve for the uncalibrated model
   prob_true_uncalibrated, prob_pred_uncalibrated = calibration_curve(
      y_test,
      y_prob_uncalibrated,
      n_bins=10,
   )


   # Calibrate the model
   if model_tuner.calibrate:
      model_tuner.calibrateModel(X, y, score="roc_auc")

   # Predict on the validation set
   y_test_pred = model_tuner.predict_proba(X_test)[:,1]


.. code-block:: bash


   Change back to CPU
   Confusion matrix on validation set for roc_auc
   --------------------------------------------------------------------------------
            Predicted:
               Pos   Neg
   --------------------------------------------------------------------------------
   Actual: Pos 292 (tp)   22 (fn)
         Neg  32 (fp)   82 (tn)
   --------------------------------------------------------------------------------

               precision    recall  f1-score   support

            0       0.90      0.93      0.92       314
            1       0.79      0.72      0.75       114

      accuracy                           0.87       428
      macro avg       0.84      0.82      0.83       428
   weighted avg       0.87      0.87      0.87       428

   --------------------------------------------------------------------------------
   roc_auc after calibration: 0.9364035087719298


.. code-block:: python

   import matplotlib.pyplot as plt

   # Get the predicted probabilities for the validation data from calibrated model
   y_prob_calibrated = model_tuner.predict_proba(X_test)[:, 1]

   # Compute the calibration curve for the calibrated model
   prob_true_calibrated, prob_pred_calibrated = calibration_curve(
      y_test,
      y_prob_calibrated,
      n_bins=5,
   )


   # Plot the calibration curves
   plt.figure(figsize=(5, 5))
   plt.plot(
      prob_pred_uncalibrated,
      prob_true_uncalibrated,
      marker="o",
      label="Uncalibrated XGBoost",
   )
   plt.plot(
      prob_pred_calibrated,
      prob_true_calibrated,
      marker="o",
      label="Calibrated XGBoost",
   )
   plt.plot(
      [0, 1],
      [0, 1],
      linestyle="--",
      label="Perfectly calibrated",
   )
   plt.xlabel("Predicted probability")
   plt.ylabel("True probability in each bin")
   plt.title("Calibration plot (reliability curve)")
   plt.legend()
   plt.show()


.. raw:: html

   <div class="no-click">

.. image:: /../assets/calibration_curves.png
   :alt: Model Tuner Logo
   :align: center
   :width: 400px

.. raw:: html

   </div>

.. raw:: html

   <div style="height: 50px;"></div>


Regression
===========

Here is an example of using the ``Model`` class for regression using XGBoost on the California Housing dataset.

California Housing with XGBoost
--------------------------------

Step 1: Import Necessary Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import pandas as pd
   import numpy as np
   import xgboost as xgb
   from sklearn.datasets import fetch_california_housing
   from model_tuner import model_tuner  

Step 2: Load the Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Load the California Housing dataset
   data = fetch_california_housing()
   X = pd.DataFrame(data.data, columns=data.feature_names)
   y = pd.Series(data.target, name="target")

Step 3: Create an Instance of the XGBClassifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Creating an instance of the XGBRegressor
   xgb_model = xgb.XGBRegressor(
      random_state=222,
   )

Step 4: Define Hyperparameters for XGBoost
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

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


Step 5: Initialize and Configure the ``Model``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Initialize model_tuner
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

Step 6: Fit the Model
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Get the training and validation data
   X_train, y_train = model_tuner.get_train_data(X, y)
   X_valid, y_valid = model_tuner.get_valid_data(X, y)

   # Fit the model with the validation data
   model_tuner.fit(
      X_train, y_train, validation_data=(X_valid, y_valid), 
      score="neg_mean_squared_error",
   )

Step 7: Return Metrics (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Return metrics for the validation set
   metrics = model_tuner.return_metrics(
      X_valid,
      y_valid,
   )
   print(metrics)


.. code-block:: bash

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
   ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 
   'AveOccup', 'Latitude', 'Longitude']

   {'Regression Report': {'Explained Variance': 0.8385815985957561, 'R2': 
   0.8385811859863378, 'Mean Absolute Error': 0.3008222037008959, 'Median 
   Absolute Error': 0.196492121219635, 'Mean Squared Error': 
   0.21038206511437127, 'RMSE': 0.45867424727618106}, 'K Best Features': 
   ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 
   'AveOccup', 'Latitude', 'Longitude']}






