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

Binary Classification Examples
--------------------------------

   **Google Colab Notebooks**

   - .. raw:: html

   <a href="https://colab.research.google.com/drive/1ujLL2mRtIWwGamnpWKIo2f271_Q103t-?usp=sharing#scrollTo=uMxyy0yvd2xQ" target="_blank">Binary Classification + KFold Example: Titanic Dataset - Categorical Data</a>

   - .. raw:: html

      <a href="https://colab.research.google.com/drive/12XywbGBiwlZIbi0C3JKu9NOQPPRgVwcp?usp=sharing#scrollTo=rm5TA__pC3M-" target="_blank">Binary Classification: AIDS Clinical Trials - Numerical Data</a>

   - .. raw:: html

      <a href="https://colab.research.google.com/drive/1D9nl8rLdwxPEpiZplsU0I0lFSAec7NzP?authuser=1#scrollTo=tumIjsNpSAKC&uniqifier=1" target="_blank">Binary Classification: Breast Cancer - Numerical Data</a>


   **HTML Files**

   - .. raw:: html

      <a href="Model_Tuner_Binary_Classification_AIDS_Clinical_Trials.html" target="_blank">Binary Classification: AIDS Clinical Trials HTML File</a>

   - .. raw:: html

      <a href="Model_Tuner_Binary_Classification_Breast_Cancer_Example.html" target="_blank">Binary Classification: Breast Cancer HTML File</a>



Regression Example
----------------------

   **Google Colab Notebook**

   - .. raw:: html

      <a href="https://colab.research.google.com/drive/151kdlsW-WyJ0pwwt_iWpjXDuqj1Ktam_?authuser=1#scrollTo=UhfZKVoq3sAN" target="_blank">Redfin Real Estate - Los Angeles Data Colab Notebook</a>
      

   **HTML File**
   
   - .. raw:: html

      <a href="Model_Tuner_Regression_Redfin_Real_Estate.html" target="_blank">Redfin Real Estate - Los Angeles Data HTML File</a>


Key Methods and Functionalities
========================================

``__init__(...)``
    Initializes the model tuner with configurations, including estimator, cross-validation settings, scoring metrics, pipeline steps, feature selection, imbalance sampler, Bayesian search, and model calibration options.

``reset_estimator()``
    Resets the estimator and pipeline configuration.

``process_imbalance_sampler(X_train, y_train)``
    Processes the imbalance sampler, applying it to resample the training data.

``calibrateModel(X, y, score=None)``
    Calibrates the model with cross-validation support and configurable calibration methods, improving probability estimates.

``get_train_data(X, y), get_valid_data(X, y), get_test_data(X, y)``
    Retrieves train, validation, and test data based on specified indices.

``calibrate_report(X, y, score=None)``
    Generates a calibration report, including a confusion matrix and classification report.

``fit(X, y, validation_data=None, score=None)``
    Fits the model to training data and, if applicable, tunes threshold and performs early stopping. Allows feature selection and processing steps as part of the pipeline.

``return_metrics(X_test, y_test, optimal_threshold=False)``
    Returns evaluation metrics with confusion matrix and classification report, optionally using optimized classification thresholds.

``predict(X, y=None, optimal_threshold=False), predict_proba(X, y=None)``
    Makes predictions and predicts probabilities, allowing threshold tuning.

``grid_search_param_tuning(X, y, f1_beta_tune=False, betas=[1, 2])``
    Performs grid or Bayesian search parameter tuning, optionally tuning F-beta score thresholds for classification.

``print_selected_best_features(X)``
    Prints and returns the selected top K best features based on the feature selection step.

``tune_threshold_Fbeta(score, y_valid, betas, y_valid_proba, kfold=False)``
    Tunes classification threshold for optimal F-beta score, balancing precision and recall across various thresholds.

``train_val_test_split(X, y, stratify_y, train_size, validation_size, test_size, random_state, stratify_cols)``
    Splits data into train, validation, and test sets, supporting stratification by specific columns or the target variable.

``get_best_score_params(X, y)``
    Retrieves the best hyperparameters for the model based on cross-validation scores for specified metrics.

``conf_mat_class_kfold(X, y, test_model, score=None)``
    Generates and averages confusion matrices across k-folds, producing a combined classification report.

``regression_report_kfold(X, y, test_model, score=None)``
    Generates averaged regression metrics across k-folds.

``regression_report(y_true, y_pred, print_results=True)``
    Generates a regression report with metrics like Mean Absolute Error, R-squared, and Root Mean Squared Error.


Helper Functions
=================

``kfold_split(classifier, X, y, stratify=False, scoring=["roc_auc"], n_splits=10, random_state=3)``
    Splits data using k-fold or stratified k-fold cross-validation.

``get_cross_validate(classifier, X, y, kf, scoring=["roc_auc"])``
    Performs cross-validation and returns training scores and estimator instances.

``_confusion_matrix_print(conf_matrix, labels)``
    Prints the formatted confusion matrix for binary classification.

``print_pipeline(pipeline)``
    Displays an ASCII representation of the pipeline steps for visual clarity.

``report_model_metrics(model, X_valid=None, y_valid=None, threshold=0.5)``
    Generates a DataFrame of key model performance metrics, including Precision, Sensitivity, Specificity, and AUC-ROC.


.. note::

   - This class is designed to be flexible and can be extended to include additional functionalities or custom metrics.
   - It is essential to properly configure the parameters during initialization to suit the specific requirements of your machine learning task.
   - Ensure that all dependencies are installed and properly imported before using the ``Model`` class from the ``model_tuner`` library.

Input Parameters
=====================

.. class:: Model(name, estimator_name, estimator, calibrate=False, kfold=False, imbalance_sampler=None, train_size=0.6, validation_size=0.2, test_size=0.2, stratify_y=False, stratify_cols=None, grid=None, scoring=["roc_auc"], n_splits=10, random_state=3, n_jobs=1, display=True, randomized_grid=False, n_iter=100, pipeline_steps=[], boost_early=False, feature_selection=False, model_type="classification", class_labels=None, multi_label=False, calibration_method="sigmoid", custom_scorer=[], bayesian=False)

   A class for building, tuning, and evaluating machine learning models, supporting both classification and regression tasks, as well as multi-label classification.

   :param name: A unique name for the model, helpful for tracking outputs and logs.
   :type name: str
   :param estimator_name: Prefix for the estimator in the pipeline, used for setting parameters in tuning (e.g., estimator_name + ``__param_name``).
   :type estimator_name: str
   :param estimator: The machine learning model to be trained and tuned.
   :type estimator: object
   :param calibrate: Whether to calibrate the model's probability estimates. Default is ``False``.
   :type calibrate: bool, optional
   :param kfold: Whether to perform k-fold cross-validation. Default is ``False``.
   :type kfold: bool, optional
   :param imbalance_sampler: An imbalanced data sampler from the imblearn library, e.g., ``RandomUnderSampler`` or ``RandomOverSampler``.
   :type imbalance_sampler: object, optional
   :param train_size: Proportion of the data to be used for training. Default is ``0.6``.
   :type train_size: float, optional
   :param validation_size: Proportion of the data to be used for validation. Default is ``0.2``.
   :type validation_size: float, optional
   :param test_size: Proportion of the data to be used for testing. Default is ``0.2``.
   :type test_size: float, optional
   :param stratify_y: Whether to stratify by the target variable during data splitting. Default is ``False``.
   :type stratify_y: bool, optional
   :param stratify_cols: List of columns to use for stratification during data splitting. Default is ``None``.
   :type stratify_cols: list, optional
   :param grid: Hyperparameter grid for model tuning, supporting both regular and Bayesian search.
   :type grid: list of dict
   :param scoring: List of scoring metrics for evaluation, e.g., ``["roc_auc", "accuracy"]``.
   :type scoring: list of str
   :param n_splits: Number of splits for k-fold cross-validation. Default is ``10``.
   :type n_splits: int, optional
   :param random_state: Seed for random number generation to ensure reproducibility. Default is ``3``.
   :type random_state: int, optional
   :param n_jobs: Number of parallel jobs to run for model fitting. Default is ``1``.
   :type n_jobs: int, optional
   :param display: Whether to print messages during the tuning and training process. Default is ``True``.
   :type display: bool, optional
   :param randomized_grid: Whether to use randomized grid search. Default is ``False``.
   :type randomized_grid: bool, optional
   :param n_iter: Number of iterations for randomized grid search. Default is ``100``.
   :type n_iter: int, optional
   :param pipeline_steps: List of steps for the pipeline, e.g., preprocessing and feature selection steps. Default is ``[]``.
   :type pipeline_steps: list, optional
   :param boost_early: Whether to enable early stopping for boosting algorithms like XGBoost. Default is ``False``.
   :type boost_early: bool, optional
   :param feature_selection: Whether to enable feature selection. Default is ``False``.
   :type feature_selection: bool, optional
   :param model_type: Specifies the model type, either ``classification`` or ``regression``. Default is ``classification``.
   :type model_type: str, optional
   :param class_labels: List of labels for multi-class classification. Default is ``None``.
   :type class_labels: list, optional
   :param multi_label: Whether the task is a multi-label classification problem. Default is ``False``.
   :type multi_label: bool, optional
   :param calibration_method: Method for calibration; options include ``sigmoid`` and ``isotonic``. Default is ``sigmoid``.
   :type calibration_method: str, optional
   :param custom_scorer: Dictionary of custom scoring functions, allowing additional metrics to be evaluated. Default is ``[]``.
   :type custom_scorer: dict, optional
   :param bayesian: Whether to perform Bayesian hyperparameter tuning using ``BayesSearchCV``. Default is ``False``.
   :type bayesian: bool, optional

   :raises ImportError: If the ``bootstrapper`` module is not found or not installed.
   :raises ValueError: Raised for various issues, such as invalid hyperparameter configurations, or mismatched ``X`` and ``y`` shapes.
   :raises AttributeError: Raised if an expected pipeline step is missing, or if ``self.estimator`` is improperly initialized.
   :raises TypeError: Raised when an incorrect parameter type is provided, such as passing ``None`` instead of a valid object.
   :raises IndexError: Raised for indexing issues, particularly in confusion matrix formatting functions.
   :raises KeyError: Raised when accessing dictionary keys that are not available, such as missing scores in ``self.best_params_per_score``.
   :raises RuntimeError: Raised for unexpected issues during model fitting or transformations that do not fit into the other exception categories.


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
    from model_tuner import model_tuner  
    from sklearn.impute import SimpleImputer


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


Step 4: Create an Instance of the XGBClassifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Creating an instance of the XGBClassifier
   xgb_model = xgb.XGBClassifier(
      random_state=222,
   )

Step 5: Define Hyperparameters for XGBoost
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


Step 6: Initialize and Configure the ``Model``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Initialize model_tuner
   model_tuner = Model(
      pipeline_steps=[
         ("Preprocessor", SimpleImputer()),
      ],
      name="XGBoost_AIDS",
      estimator_name=estimator_name_xgb,
      calibrate=True,
      estimator=xgb_model,
      xgboost_early=True,
      kfold=False,
      selectKBest=True,
      stratify_y=False,
      grid=xgb_parameters,
      randomized_grid=False,
      scoring=["roc_auc"],
      random_state=222,
      n_jobs=-1,
   )

Step 7: Perform Grid Search Parameter Tuning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Perform grid search parameter tuning
   model_tuner.grid_search_param_tuning(X, y)

.. code-block:: bash

   100%|██████████| 324/324 [01:34<00:00,  3.42it/s]
   Best score/param set found on validation set:
   {'params': {'selectKBest__k': 20,
               'xgb__colsample_bytree': 1.0,
               'xgb__early_stopping_rounds': 10,
               'xgb__eval_metric': 'logloss',
               'xgb__learning_rate': 0.05,
               'xgb__max_depth': 5,
               'xgb__n_estimators': 100,
               'xgb__subsample': 1.0},
   'score': 0.946877967711301}
   Best roc_auc: 0.947 

Step 8: Fit the Model
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Get the training and validation data
   X_train, y_train = model_tuner.get_train_data(X, y)
   X_valid, y_valid = model_tuner.get_valid_data(X, y)
   X_test, y_test = model_tuner.get_test_data(X, y)

   # Fit the model with the validation data
   model_tuner.fit(
      X_train,
      y_train,
      validation_data=(X_valid, y_valid),
      score="roc_auc",
   )

Step 9: Return Metrics (Optional)
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
   Actual: Pos  89 (tp)   15 (fn)
           Neg  21 (fp)  303 (tn)
   --------------------------------------------------------------------------------

               precision    recall  f1-score   support

            0       0.95      0.94      0.94       324
            1       0.81      0.86      0.83       104

      accuracy                          0.92       428
      macro avg     0.88      0.90      0.89       428
   weighted avg     0.92      0.92      0.92       428

   --------------------------------------------------------------------------------

   Feature names selected:
   ['time', 'trt', 'age', 'hemo', 'homo', 'drugs', 'karnof', 'oprior', 'z30', 'preanti', 'race', 'gender', 'str2', 'strat', 'symptom', 'treat', 'offtrt', 'cd40', 'cd420', 'cd80']

   {'Classification Report': {'0': {'precision': 0.9528301886792453,
      'recall': 0.9351851851851852,
      'f1-score': 0.9439252336448598,
      'support': 324.0},
   '1': {'precision': 0.8090909090909091,
      'recall': 0.8557692307692307,
      'f1-score': 0.8317757009345793,
      'support': 104.0},
   'accuracy': 0.9158878504672897,
   'macro avg': {'precision': 0.8809605488850771,
      'recall': 0.895477207977208,
      'f1-score': 0.8878504672897196,
      'support': 428.0},
   'weighted avg': {'precision': 0.9179028870970327,
      'recall': 0.9158878504672897,
      'f1-score': 0.9166739453227356,
      'support': 428.0}},
   'Confusion Matrix': array([[303,  21],
         [ 15,  89]]),
   'K Best Features': ['time',
   'trt',
   'age',
   'hemo',
   'homo',
   'drugs',
   'karnof',
   'oprior',
   'z30',
   'preanti',
   'race',
   'gender',
   'str2',
   'strat',
   'symptom',
   'treat',
   'offtrt',
   'cd40',
   'cd420',
   'cd80']}

Step 10: Calibrate the Model (if needed)
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

Classification Report (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A classification report is readily available at this stage, should you wish to 
print and examine it. A call to ``print(model_tuner.classification_report)`` will
output it as follows:

.. code-block:: python 

   print(model_tuner.classification_report)

.. code-block:: bash

                precision    recall  f1-score   support

             0       0.95      0.94      0.94       324
             1       0.81      0.85      0.83       104

      accuracy                           0.92       428
     macro avg       0.88      0.89      0.89       428
  weighted avg       0.92      0.92      0.92       428



Regression
===========

Here is an example of using the ``Model`` class for regression using ``XGBoost`` on the California Housing dataset.

California Housing with XGBoost
--------------------------------

Step 1: Import Necessary Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import pandas as pd
   import numpy as np
   import xgboost as xgb
   from sklearn.impute import SimpleImputer
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
   xgb_learning_rates = [0.1, 0.01, 0.05]  # Learning rate or eta
   # Number of trees. Equivalent to n_estimators in GB
   xgb_n_estimators = [100, 200, 300]  
   xgb_max_depths = [3, 5, 7][:1]  # Maximum depth of the trees
   xgb_subsamples = [0.8, 1.0][:1]  # Subsample ratio of the training instances
   xgb_colsample_bytree = [0.8, 1.0][:1]
   xgb_eval_metric = ["logloss"]
   xgb_early_stopping_rounds = [10]
   xgb_verbose = [False]

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
      }
   ]

Step 5: Initialize and Configure the ``Model``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Initialize model_tuner
   california_housing = Model(
      pipeline_steps=[
         ("Preprocessor", SimpleImputer()),
      ],
      name="Redfin_model_XGB",
      estimator_name="xgb",
      model_type="regression",
      calibrate=False,
      estimator=xgb_model,
      kfold=False,
      stratify_y=False,
      grid=xgb_parameters,
      randomized_grid=False,
      scoring=["r2"],
      random_state=3,
      xgboost_early=True,
   )

Step 6: Fit the Model
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   eval_set = [X, y]  # necessary for early stopping

   # Perform grid search parameter tuning
   california_housing.grid_search_param_tuning(X, y)

   # Get the training and validation data
   X_train, y_train = california_housing.get_train_data(X, y)
   X_valid, y_valid = california_housing.get_valid_data(X, y)

   california_housing.fit(
      X_train,
      y_train,
      validation_data=(X_valid, y_valid),
   )

   california_housing.return_metrics(X_test, y_test)

.. code-block:: bash

   100%|██████████| 9/9 [00:01<00:00,  4.81it/s]
   Best score/param set found on validation set:
   {'params': {'xgb__colsample_bytree': 0.8,
               'xgb__early_stopping_rounds': 10,
               'xgb__eval_metric': 'logloss',
               'xgb__learning_rate': 0.1,
               'xgb__max_depth': 3,
               'xgb__n_estimators': 172,
               'xgb__subsample': 0.8},
   'score': np.float64(0.7979488661159093)}
   Best r2: 0.798 

   ********************************************************************************
   {'Explained Variance': 0.7979060590722392,
   'Mean Absolute Error': np.float64(0.35007797000749163),
   'Mean Squared Error': np.float64(0.2633964855111536),
   'Median Absolute Error': np.float64(0.24205514192581173),
   'R2': 0.7979050719771986,
   'RMSE': np.float64(0.5132216728774747)}
   ********************************************************************************

   {'Explained Variance': 0.7979060590722392,
   'R2': 0.7979050719771986,
   'Mean Absolute Error': np.float64(0.35007797000749163),
   'Median Absolute Error': np.float64(0.24205514192581173),
   'Mean Squared Error': np.float64(0.2633964855111536),
   'RMSE': np.float64(0.5132216728774747)}


