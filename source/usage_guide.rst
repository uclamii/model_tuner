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

      <a href="https://colab.research.google.com/drive/16gWnRAJvpUjTIes5y1gFRdX1soASdV6m#scrollTo=3NYa_tQWy6HR" target="_blank">Binary Classification: Imbalanced Learning</a>



   **HTML Files**


   - .. raw:: html

      <a href="./example_htmls/Model_Tuner_Column_Transformer.html" target="_blank">Binary Classification + KFold Example: Titanic Dataset - Categorical Data</a>

   - .. raw:: html

      <a href="./example_htmls/Model_Tuner_Binary_Classification_AIDS_Clinical_Trials.html" target="_blank">Binary Classification: AIDS Clinical Trials HTML File</a>

   - .. raw:: html

      <a href="./example_htmls/Model_Tuner_Binary_Classification_Imbalanced_Learning.html" target="_blank">Binary Classification: Imbalanced Learning</a>



Regression Example
----------------------

   **Google Colab Notebook**

   - .. raw:: html

      <a href="https://colab.research.google.com/drive/151kdlsW-WyJ0pwwt_iWpjXDuqj1Ktam_?authuser=1#scrollTo=UhfZKVoq3sAN" target="_blank">Redfin Real Estate - Los Angeles Data Colab Notebook</a>
      

   **HTML File**
   
   - .. raw:: html

      <a href="./example_htmls/Model_Tuner_Regression_Redfin_Real_Estate.html" target="_blank">Redfin Real Estate - Los Angeles Data HTML File</a>


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


.. class:: Model(name, estimator_name, estimator, model_type, calibrate=False, kfold=False, imbalance_sampler=None, train_size=0.6, validation_size=0.2, test_size=0.2, stratify_y=False, stratify_cols=None, grid=None, scoring=["roc_auc"], n_splits=10, random_state=3, n_jobs=1, display=True, randomized_grid=False, n_iter=100, pipeline_steps=[], boost_early=False, feature_selection=False, class_labels=None, multi_label=False, calibration_method="sigmoid", custom_scorer=[], bayesian=False)

   A class for building, tuning, and evaluating machine learning models, supporting both classification and regression tasks, as well as multi-label classification.

   :param name: A unique name for the model, helpful for tracking outputs and logs.
   :type name: str
   :param estimator_name: Prefix for the estimator in the pipeline, used for setting parameters in tuning (e.g., estimator_name + ``__param_name``).
   :type estimator_name: str
   :param estimator: The machine learning model to be trained and tuned.
   :type estimator: object
   :param model_type: Specifies the type of model, must be either ``classification`` or ``regression``.
   :type model_type: str
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
   :raises ValueError: Raised for various issues, such as:
       - Invalid ``model_type`` value. The ``model_type`` must be explicitly specified as either ``classification`` or ``regression``.
       - Invalid hyperparameter configurations or mismatched ``X`` and ``y`` shapes.
   :raises AttributeError: Raised if an expected pipeline step is missing, or if ``self.estimator`` is improperly initialized.
   :raises TypeError: Raised when an incorrect parameter type is provided, such as passing ``None`` instead of a valid object.
   :raises IndexError: Raised for indexing issues, particularly in confusion matrix formatting functions.
   :raises KeyError: Raised when accessing dictionary keys that are not available, such as missing scores in ``self.best_params_per_score``.
   :raises RuntimeError: Raised for unexpected issues during model fitting or transformations that do not fit into the other exception categories.

Pipeline Management
============================================

The pipeline in the model tuner class is designed to automatically organize steps into three categories: **preprocessing**, **feature selection**, and **imbalanced sampling**. The steps are ordered in the following sequence:

1. **Preprocessing**:

   - Imputation
   - Scaling
   - Other preprocessing steps
2. **Imbalanced Sampling**
3. **Feature Selection**
4. **Classifier**

The ``pipeline_assembly`` method automatically sorts the steps into this order.

Specifying Pipeline Steps
-------------------------

Pipeline steps can be specified in multiple ways. For example, if naming a pipeline step then specify like so::

    pipeline_steps = ['imputer', SimpleImputer()]

Naming each step is optional and the steps can also be specified like so::

    pipeline_steps = [SimpleImputer(), StandardScalar(), rfe()]

- If no name is assigned, the step will be renamed automatically to follow the convention ``step_0``, ``step_1``, etc.
- Column transformers can also be included in the pipeline and are automatically categorized under the **preprocessing** section.

Helper Methods for Pipeline Extraction
--------------------------------------

To support advanced use cases, the model tuner provides helper methods to extract parts of the pipeline for later use. For example, when generating SHAP plots, users might only need the preprocessing section of the pipeline.

Here are some of the available methods:

.. py:function:: get_preprocessing_and_feature_selection_pipeline()

    Extracts both the preprocessing and feature selection parts of the pipeline.

    **Example**::

        def get_preprocessing_and_feature_selection_pipeline(self):
            steps = [
                (name, transformer)
                for name, transformer in self.estimator.steps
                if name.startswith("preprocess_") or name.startswith("feature_selection_")
            ]
            return self.PipelineClass(steps)

.. py:function:: get_feature_selection_pipeline()

    Extracts only the feature selection part of the pipeline.

    **Example**::

        def get_feature_selection_pipeline(self):
            steps = [
                (name, transformer)
                for name, transformer in self.estimator.steps
                if name.startswith("feature_selection_")
            ]
            return self.PipelineClass(steps)

.. py:function:: get_preprocessing_pipeline()

    Extracts only the preprocessing part of the pipeline.

    **Example**::

        def get_preprocessing_pipeline(self):
            preprocessing_steps = [
                (name, transformer)
                for name, transformer in self.estimator.steps
                if name.startswith("preprocess_")
            ]
            return self.PipelineClass(preprocessing_steps)

Summary
--------

By organizing pipeline steps automatically and providing helper methods for extraction, the model tuner class offers flexibility and ease of use for building and managing complex pipelines. Users can focus on specifying the steps, and the tuner handles naming, sorting, and category assignments seamlessly.

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
model's performance using a variety of metrics, such as :ref:`accuracy, precision, 
recall, and ROC-AUC <Limitations_of_Accuracy>`, ensuring that the model's ability to distinguish between the 
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

.. _xgb_hyperparams:

Step 5: Define Hyperparameters for XGBoost
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   xgb_name = "xgb"
   xgb = XGBClassifier(
      objective="binary:logistic",
      random_state=222,
   )
   xgbearly = True
   tuned_parameters_xgb = {
      f"{xgb_name}__max_depth": [3, 10, 20, 200, 500],
      f"{xgb_name}__learning_rate": [1e-4],
      f"{xgb_name}__n_estimators": [1000],
      f"{xgb_name}__early_stopping_rounds": [100],
      f"{xgb_name}__verbose": [0],
      f"{xgb_name}__eval_metric": ["logloss"],
   }

   xgb_definition = {
      "clc": xgb,
      "estimator_name": xgb_name,
      "tuned_parameters": tuned_parameters_xgb,
      "randomized_grid": False,
      "n_iter": 5,
      "early": xgbearly,
   }


Step 6: Initialize and Configure the ``Model``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model_type = "xgb"
   clc = xgb_definition["clc"]
   estimator_name = xgb_definition["estimator_name"]

   tuned_parameters = xgb_definition["tuned_parameters"]
   n_iter = xgb_definition["n_iter"]
   rand_grid = xgb_definition["randomized_grid"]
   early_stop = xgb_definition["early"]
   kfold = False
   calibrate = True

   # Initialize model_tuner
   model_xgb = Model(
      name=f"AIDS_Clinical_{model_type}",
      estimator_name=estimator_name,
      calibrate=calibrate,
      estimator=clc,
      kfold=kfold,
      stratify_y=True,
      stratify_cols=["gender", "race"],
      grid=tuned_parameters,
      randomized_grid=rand_grid,
      boost_early=early_stop,
      scoring=["roc_auc"],
      random_state=222,
      n_jobs=2,
   )

Step 7: Perform Grid Search Parameter Tuning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Perform grid search parameter tuning
   model_xgb.grid_search_param_tuning(X, y, f1_beta_tune=True)

.. code-block:: bash

   Pipeline Steps:

   ┌─────────────────┐
   │ Step 1: xgb     │
   │ XGBClassifier   │
   └─────────────────┘

   100%|██████████| 5/5 [00:19<00:00,  3.98s/it]
   Fitting model with best params and tuning for best threshold ...
   100%|██████████| 2/2 [00:00<00:00,  3.42it/s]Best score/param set found on validation set:
   {'params': {'xgb__early_stopping_rounds': 100,
               'xgb__eval_metric': 'logloss',
               'xgb__learning_rate': 0.0001,
               'xgb__max_depth': 3,
               'xgb__n_estimators': 999},
   'score': 0.9280033238366572}
   Best roc_auc: 0.928 

Step 8: Fit the Model
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   ## Get the training and validation data
   X_train, y_train = model_tuner.get_train_data(X, y)
   X_valid, y_valid = model_tuner.get_valid_data(X, y)
   X_test, y_test = model_tuner.get_test_data(X, y)

   model_xgb.fit(X_train, y_train, validation_data=[X_valid, y_valid])

Step 9: Return Metrics (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use this function to evaluate the model by printing the output.

.. code-block:: bash

   Validation Metrics
   Confusion matrix on set provided: 
   --------------------------------------------------------------------------------
            Predicted:
                Pos   Neg
   --------------------------------------------------------------------------------
   Actual: Pos  95 (tp)    9 (fn)
           Neg  79 (fp)  245 (tn)
   --------------------------------------------------------------------------------
   --------------------------------------------------------------------------------
   {'AUC ROC': 0.9280033238366572,
   'Average Precision': 0.7992275185850191,
   'Brier Score': 0.16713189436073958,
   'Precision/PPV': 0.5459770114942529,
   'Sensitivity': 0.9134615384615384,
   'Specificity': 0.7561728395061729}
   --------------------------------------------------------------------------------

                 precision    recall  f1-score   support
 
              0       0.96      0.76      0.85       324
              1       0.55      0.91      0.68       104

       accuracy                           0.79       428
      macro avg       0.76      0.83      0.77       428
   weighted avg       0.86      0.79      0.81       428

   --------------------------------------------------------------------------------

   Test Metrics
   Confusion matrix on set provided: 
   --------------------------------------------------------------------------------
            Predicted:
                Pos   Neg
   --------------------------------------------------------------------------------
   Actual: Pos  95 (tp)    9 (fn)
           Neg  78 (fp)  246 (tn)
   --------------------------------------------------------------------------------
   --------------------------------------------------------------------------------
   {'AUC ROC': 0.934576804368471,
   'Average Precision': 0.8023014087345259,
   'Brier Score': 0.16628708993634742,
   'Precision/PPV': 0.5491329479768786,
   'Sensitivity': 0.9134615384615384,
   'Specificity': 0.7592592592592593}
   --------------------------------------------------------------------------------

                 precision    recall  f1-score   support
 
              0       0.96      0.76      0.85       324
              1       0.55      0.91      0.69       104

       accuracy                           0.80       428
      macro avg       0.76      0.84      0.77       428
   weighted avg       0.86      0.80      0.81       428

   --------------------------------------------------------------------------------

Step 10: Calibrate the Model (if needed)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import matplotlib.pyplot as plt
   from sklearn.calibration import calibration_curve

   ## Get the predicted probabilities for the validation data from uncalibrated model
   y_prob_uncalibrated = model_xgb.predict_proba(X_test)[:, 1]

   ## Compute the calibration curve for the uncalibrated model
   prob_true_uncalibrated, prob_pred_uncalibrated = calibration_curve(
      y_test,
      y_prob_uncalibrated,
      n_bins=10,
   )

   ## Calibrate the model
   if model_xgb.calibrate:
      model_xgb.calibrateModel(X, y, score="roc_auc")

   ## Predict on the validation set
   y_test_pred = model_xgb.predict_proba(X_test)[:, 1]


.. code-block:: bash


   Change back to CPU
   Confusion matrix on validation set for roc_auc
   --------------------------------------------------------------------------------
            Predicted:
                Pos   Neg
   --------------------------------------------------------------------------------
   Actual: Pos  70 (tp)   34 (fn)
           Neg   9 (fp)  315 (tn)
   --------------------------------------------------------------------------------

                 precision     recall  f1-score    support

              0       0.90       0.97      0.94        324
              1       0.89       0.67      0.77        104

       accuracy                            0.90        428
      macro avg       0.89       0.82      0.85        428
   weighted avg       0.90       0.90      0.89        428

   --------------------------------------------------------------------------------
   roc_auc after calibration: 0.9280033238366572



.. code-block:: python

   ## Get the predicted probabilities for the validation data from calibrated model
   y_prob_calibrated = model_xgb.predict_proba(X_test)[:, 1]

   ## Compute the calibration curve for the calibrated model
   prob_true_calibrated, prob_pred_calibrated = calibration_curve(
      y_test,
      y_prob_calibrated,
      n_bins=10,
   )


   ## Plot the calibration curves
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
   :alt: Calibration Curve AIDs
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

            0      0.90      0.97      0.94       324
            1      0.89      0.67      0.77       104

     accuracy                          0.90       428
    macro avg      0.89      0.82      0.85       428
 weighted avg      0.90      0.90      0.89       428


Recursive Feature Elimination (RFE)
-------------------------------------

Now that we've trained the models, we can also refine them by identifying which 
features contribute most to their performance. One effective method for this is 
Recursive Feature Elimination (RFE). This technique allows us to systematically 
remove the least important features, retraining the model at each step to evaluate 
how performance is affected. By focusing only on the most impactful variables, 
RFE helps streamline the dataset, reduce noise, and improve both the accuracy and 
interpretability of the final model.

It works by recursively training a model, ranking the importance of features 
based on the model’s outputas (such as coefficients in linear models or 
importance scores in tree-based models), and then removing the least important 
features one by one. This process continues until a specified number of features 
remains or the desired performance criteria are met.

The primary advantage of RFE is its ability to streamline datasets, improving 
model performance and interpretability by focusing on features that contribute 
the most to the predictive power. However, it can be computationally expensive 
since it involves repeated model training, and its effectiveness depends on the 
underlying model’s ability to evaluate feature importance. RFE is commonly used 
with cross-validation to ensure that the selected features generalize well across 
datasets, making it a robust choice for model optimization and dimensionality 
reduction.

As an illustrative example, we will retrain the above model using RFE.

We will begin by appending the feature selection technique to our :ref:`tuned parameters dictionary <xgb_hyperparams>`.

.. code-block:: python

   xgb_definition["tuned_parameters"][f"feature_selection_rfe__n_features_to_select"] = [
      5,
      10,
   ]

Elastic Net for Feature Selection with RFE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

   You may wish to explore :ref:`this section <elastic_net>` for rationale in applying this technique.

We will use elastic net because it strikes a balance between two widely used 
regularization techniques: Lasso (:math:`L1`) and Ridge (:math:`L2`). Elastic net 
is particularly effective in scenarios where we expect the dataset to have a mix 
of strongly and weakly correlated features. Lasso alone tends to select only one 
feature from a group of highly correlated ones, ignoring the others, while Ridge 
includes all features but may not perform well when some are entirely irrelevant. 
Elastic net addresses this limitation by combining both penalties, allowing it to handle 
multicollinearity more effectively while still performing feature selection.

Additionally, elastic net provides flexibility by controlling the ratio between 
:math:`L1` and :math:`L2` penalties, enabling fine-tuning to suit the specific needs of 
our dataset. This makes it a robust choice for datasets with many features, some 
of which may be irrelevant or redundant, as it can reduce overfitting while 
retaining a manageable subset of predictors.


.. code-block:: python

   rfe_estimator = ElasticNet()

   rfe = RFE(rfe_estimator)


.. code-block:: python

   model_xgb = Model(
      name=f"AIDS_Clinical_{model_type}",
      estimator_name=estimator_name,
      calibrate=calibrate,
      estimator=clc,
      model_type="classification",
      kfold=kfold,
      pipeline_steps=[
         ("Imputer", SimpleImputer()),
         ("StandardScalar", StandardScaler()),
         ("rfe", rfe),
      ],
      stratify_y=True,
      stratify_cols=False,
      grid=tuned_parameters,
      randomized_grid=rand_grid,
      feature_selection=True,
      boost_early=early_stop,
      scoring=["roc_auc"],
      random_state=222,
      n_jobs=2,
   )

   model_xgb.grid_search_param_tuning(X, y, f1_beta_tune=True)

   X_train, y_train = model_xgb.get_train_data(X, y)
   X_test, y_test = model_xgb.get_test_data(X, y)
   X_valid, y_valid = model_xgb.get_valid_data(X, y)

   model_xgb.fit(
      X_train,
      y_train,
      validation_data=[X_valid, y_valid],
   )


   # ------------------------- VALID AND TEST METRICS -----------------------------

   print("Validation Metrics")
   model_xgb.return_metrics(
      X_valid,
      y_valid,
      optimal_threshold=True,
   )
   print()
   print("Test Metrics")
   model_xgb.return_metrics(
      X_test,
      y_test,
      optimal_threshold=True,
   )

   print()

.. code-block:: bash

   Pipeline Steps:

   ┌────────────────────────────────────────────┐
   │ Step 1: preprocess_imputer_Imputer         │
   │ SimpleImputer                              │
   └────────────────────────────────────────────┘
                        │
                        ▼
   ┌────────────────────────────────────────────┐
   │ Step 2: preprocess_scaler_StandardScalar   │
   │ StandardScaler                             │
   └────────────────────────────────────────────┘
                        │
                        ▼
   ┌────────────────────────────────────────────┐
   │ Step 3: feature_selection_rfe              │
   │ RFE                                        │
   └────────────────────────────────────────────┘
                        │
                        ▼
   ┌────────────────────────────────────────────┐
   │ Step 4: xgb                                │
   │ XGBClassifier                              │
   └────────────────────────────────────────────┘

   100%|██████████| 10/10 [00:29<00:00,  2.99s/it]
   Fitting model with best params and tuning for best threshold ...
   100%|██████████| 2/2 [00:00<00:00,  3.57it/s]
   Best score/param set found on validation set:
   {'params': {'feature_selection_rfe__n_features_to_select': 5,
               'xgb__early_stopping_rounds': 100,
               'xgb__eval_metric': 'logloss',
               'xgb__learning_rate': 0.0001,
               'xgb__max_depth': 3,
               'xgb__n_estimators': 999},
   'score': 0.7360220797720798}
   Best roc_auc: 0.736 

   Validation Metrics
   Confusion matrix on set provided: 
   --------------------------------------------------------------------------------
            Predicted:
                Pos   Neg
   --------------------------------------------------------------------------------
   Actual: Pos  77 (tp)   27 (fn)
           Neg 125 (fp)  199 (tn)
   --------------------------------------------------------------------------------
   --------------------------------------------------------------------------------
   {'AUC ROC': 0.7360220797720798,
   'Average Precision': 0.5399209924105289,
   'Brier Score': 0.17797682092982414,
   'Precision/PPV': 0.3811881188118812,
   'Sensitivity': 0.7403846153846154,
   'Specificity': 0.6141975308641975}
   --------------------------------------------------------------------------------

                 precision    recall  f1-score   support

              0       0.88      0.61      0.72       324
              1       0.38      0.74      0.50       104

       accuracy                           0.64       428
      macro avg       0.63      0.68      0.61       428
   weighted avg       0.76      0.64      0.67       428

   --------------------------------------------------------------------------------

   Feature names selected:
   ['offtrt', 'cd40', 'cd420', 'cd80', 'cd820']


   Test Metrics
   Confusion matrix on set provided: 
   --------------------------------------------------------------------------------
            Predicted:
               Pos   Neg
   --------------------------------------------------------------------------------
   Actual: Pos  74 (tp)   30 (fn)
         Neg 126 (fp)  198 (tn)
   --------------------------------------------------------------------------------
   --------------------------------------------------------------------------------
   {'AUC ROC': 0.706582383665717,
   'Average Precision': 0.44119645955771625,
   'Brier Score': 0.17935075703865283,
   'Precision/PPV': 0.37,
   'Sensitivity': 0.7115384615384616,
   'Specificity': 0.6111111111111112}
   --------------------------------------------------------------------------------

                 precision    recall  f1-score   support

              0       0.87      0.61      0.72       324
              1       0.37      0.71      0.49       104

       accuracy                           0.64       428
      macro avg       0.62      0.66      0.60       428
   weighted avg       0.75      0.64      0.66       428

   --------------------------------------------------------------------------------

   Feature names selected:
   ['offtrt', 'cd40', 'cd420', 'cd80', 'cd820']


.. important::

   Passing ``feature_selection=True`` in conjunction with accounting for ``rfe`` for 
   the ``pipeline_steps`` inside the ``Model``` class above is necessary to print the
   output of the feature names selected, thus yielding:

   .. code-block:: bash

      Feature names selected:
      ['offtrt', 'cd40', 'cd420', 'cd80', 'cd820']


Imbalanced Learning
------------------------

In machine learning, imbalanced datasets are a frequent challenge, especially in 
real-world scenarios. These datasets have an unequal distribution of target classes, 
with one class (e.g., fraudulent transactions, rare diseases, or other low-frequency events) 
being underrepresented compared to the majority class. Models trained on imbalanced data 
often struggle to generalize, as they tend to favor the majority class, leading to 
poor performance on the minority class.

To mitigate these issues, it is crucial to:

1. Understand the nature of the imbalance in the dataset.
2. Apply appropriate resampling techniques (oversampling, undersampling, or hybrid methods).
3. Use metrics beyond accuracy, such as :ref:`precision, recall, and F1-score <Limitations_of_Accuracy>`, to evaluate model performance fairly.

Generating an Imbalanced Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Demonstrated below are the steps to generate an imbalanced dataset using 
``make_classification`` from the ``sklearn.datasets`` module. The following 
parameters are specified:

- ``n_samples=1000``: The dataset contains 1,000 samples.    
- ``n_features=20``: Each sample has 20 features.    
- ``n_informative=2``: Two features are informative for predicting the target.  
- ``n_redundant=2``: Two features are linear combinations of the informative features.  
- ``weights=[0.9, 0.1]``: The target class distribution is 90% for the majority class and 10% for the minority class, creating an imbalance.  
- ``flip_y=0``: No label noise is added to the target variable.  
- ``random_state=42``: Ensures reproducibility by using a fixed random seed.

.. code-block:: python

   import pandas as pd
   import numpy as np
   from sklearn.datasets import make_classification

   X, y = make_classification(
      n_samples=1000,  
      n_features=20,  
      n_informative=2, 
      n_redundant=2,  
      n_clusters_per_class=1,
      weights=[0.9, 0.1],  
      flip_y=0,  
      random_state=42,
   )

   ## Convert to a pandas DataFrame for better visualization
   data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(1, 21)])
   data['target'] = y

   X = data[[col for col in data.columns if "target" not in col]]
   y = pd.Series(data["target"])


Below, you will see that the dataset we have generated is severely imbalanced with 
900 observations allocated to the majority class (0) and 100 observations to the minority class (1).

.. code-block:: python

   import matplotlib.pyplot as plt

   ## Create a bar plot
   value_counts = pd.Series(y).value_counts()
   ax = value_counts.plot(
      kind="bar",
      rot=0,
      width=0.9,
   )

   ## Add labels inside the bars
   for index, count in enumerate(value_counts):
      plt.text(
         index,  
         count / 2,  
         str(count),  
         ha="center",
         va="center",  
         color="yellow",  
      )

   ## Customize labels and title
   plt.xlabel("Class")
   plt.ylabel("Count")
   plt.title("Class Distribution")

   plt.show() ## Show the plot


.. raw:: html

   <div class="no-click">

.. image:: /../assets/imbalanced_classes.png
   :alt: Calibration Curve AIDs
   :align: center
   :width: 400px

.. raw:: html

   </div>

.. raw:: html

   <div style="height: 50px;"></div>


Define Hyperparameters for XGBoost
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Below, we will use an XGBoost classifier with the following hyperparameters:

.. code-block:: python

   from xgboost import XGBClassifier

   xgb_name = "xgb"
   xgb = XGBClassifier(
      random_state=222,
   )
   xgbearly = True
   tuned_parameters_xgb = {
      f"{xgb_name}__max_depth": [3, 10, 20, 200, 500],
      f"{xgb_name}__learning_rate": [1e-4],
      f"{xgb_name}__n_estimators": [1000],
      f"{xgb_name}__early_stopping_rounds": [100],
      f"{xgb_name}__verbose": [0],
      f"{xgb_name}__eval_metric": ["logloss"],
   }

   xgb_definition = {
      "clc": xgb,
      "estimator_name": xgb_name,
      "tuned_parameters": tuned_parameters_xgb,
      "randomized_grid": False,
      "n_iter": 5,
      "early": xgbearly,
   }

Define The Model object
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model_type = "xgb"
   clc = xgb_definition["clc"]
   estimator_name = xgb_definition["estimator_name"]

   tuned_parameters = xgb_definition["tuned_parameters"]
   n_iter = xgb_definition["n_iter"]
   rand_grid = xgb_definition["randomized_grid"]
   early_stop = xgb_definition["early"]
   kfold = False
   calibrate = True


Addressing Class Imbalance in Machine Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Class imbalance occurs when one class significantly outweighs another in the 
dataset, leading to biased models that perform well on the majority class but 
poorly on the minority class. Techniques like SMOTE and others aim to address 
this issue by improving the representation of the minority class, ensuring balanced 
learning and better generalization.

Techniques to Address Class Imbalance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Resampling Techniques**

- **SMOTE (Synthetic Minority Oversampling Technique)**: SMOTE generates synthetic samples for the minority class by interpolating between existing minority class data points and their nearest neighbors. This helps create a more balanced class distribution without merely duplicating data, thus avoiding overfitting.

- **Oversampling**: Randomly duplicates examples from the minority class to balance the dataset. While simple, it risks overfitting to the duplicated examples.  

- **Undersampling**: Reduces the majority class by randomly removing samples. While effective, it can lead to loss of important information.

Purpose of Using These Techniques
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The goal of using these techniques is to improve model performance on imbalanced datasets, specifically by:

- Ensuring the model captures meaningful patterns in the minority class.
- Reducing bias toward the majority class, which often dominates predictions in imbalanced datasets.
- Improving metrics like :ref:`recall, F1-score, and AUC-ROC <Limitations_of_Accuracy>` for the minority class, which are critical in applications like fraud detection, healthcare, and rare event prediction.

.. note::

   While we provide comprehensive examples for SMOTE, ADASYN, and 
   RandomUnderSampler in the `accompanying notebook <https://colab.research.google.com/drive/16gWnRAJvpUjTIes5y1gFRdX1soASdV6m#scrollTo=3NYa_tQWy6HR>`_, 
   this documentation section demonstrates the implementation of SMOTE. The other 
   examples follow a similar workflow and can be executed by simply passing the 
   respective ``imbalance_sampler`` input to ``ADASYN()`` or ``RandomUnderSampler()``, as 
   needed. For detailed examples of all methods, please refer to the linked notebook.

Synthetic Minority Oversampling Technique (SMOTE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
SMOTE (Synthetic Minority Oversampling Technique) is a method used to address 
class imbalance in datasets. It generates synthetic samples for the minority 
class by interpolating between existing minority samples and their nearest neighbors, 
effectively increasing the size of the minority class without duplicating data. 
This helps models better learn patterns from the minority class, improving 
classification performance on imbalanced datasets.

Initalize and Configure The Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::

   In the code block below, we initalize and configure the model by calling the 
   ``Model`` class, and assign it to a new variable call ``xgb_smote``. Notice that 
   we pass the ``imbalance_sampler=SMOTE()`` as a necessary step of activating 
   this imbalanced sampler. 

.. code-block:: python

   from model_tuner import Model

   xgb_smote = Model(
      name=f"Make_Classification_{model_type}",
      estimator_name=estimator_name,
      calibrate=calibrate,
      model_type="classification",
      estimator=clc,
      kfold=kfold,
      stratify_y=True,
      stratify_cols=False,
      grid=tuned_parameters,
      randomized_grid=rand_grid,
      boost_early=early_stop,
      scoring=["roc_auc"],
      random_state=222,
      n_jobs=2,
      imbalance_sampler=SMOTE(),
   )

Perform Grid Search Parameter Tuning and Retrieve Split Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   xgb_smote.grid_search_param_tuning(
      X,
      y,
      f1_beta_tune=True,
   )

   X_train, y_train = xgb_smote.get_train_data(X, y)
   X_test, y_test = xgb_smote.get_test_data(X, y)
   X_valid, y_valid = xgb_smote.get_valid_data(X, y)


.. code-block:: bash

   Pipeline Steps:

   ┌─────────────────────┐
   │ Step 1: resampler   │
   │ SMOTE               │
   └─────────────────────┘
            │
            ▼
   ┌─────────────────────┐
   │ Step 2: xgb         │
   │ XGBClassifier       │
   └─────────────────────┘

   Distribution of y values after resampling: target
   0         540
   1         540
   Name: count, dtype: int64

   100%|██████████| 5/5 [00:34<00:00,  6.87s/it]
   Fitting model with best params and tuning for best threshold ...
   100%|██████████| 2/2 [00:00<00:00,  4.37it/s]Best score/param set found on validation set:
   {'params': {'xgb__early_stopping_rounds': 100,
               'xgb__eval_metric': 'logloss',
               'xgb__learning_rate': 0.0001,
               'xgb__max_depth': 10,
               'xgb__n_estimators': 999},
   'score': 0.9990277777777777}
   Best roc_auc: 0.999 

SMOTE: Distribution of y values after resampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Notice that the target has been redistributed after SMOTE to 540 observations 
for the minority class and 540 observations for the majority class.

Fit The Model
~~~~~~~~~~~~~~~

.. code-block:: python

   xgb_smote.fit(
      X_train,
      y_train,
      validation_data=[X_valid, y_valid],
   )

Return Metrics (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   Validation Metrics
   Confusion matrix on set provided: 
   --------------------------------------------------------------------------------
            Predicted:
                Pos   Neg
   --------------------------------------------------------------------------------
   Actual: Pos  20 (tp)    0 (fn)
           Neg   6 (fp)  174 (tn)
   --------------------------------------------------------------------------------
   --------------------------------------------------------------------------------
   {'AUC ROC': 0.9955555555555555,
   'Average Precision': 0.9378696741854636,
   'Brier Score': 0.20835571676988004,
   'Precision/PPV': 0.7692307692307693,
   'Sensitivity': 1.0,
   'Specificity': 0.9666666666666667}
   --------------------------------------------------------------------------------

               precision   recall  f1-score   support

              0     1.00     0.97      0.98       180
              1     0.77     1.00      0.87        20

       accuracy                        0.97       200
      macro avg     0.88     0.98      0.93       200
   weighted avg     0.98     0.97      0.97       200

   --------------------------------------------------------------------------------

   Test Metrics
   Confusion matrix on set provided: 
   --------------------------------------------------------------------------------
            Predicted:
                Pos   Neg
   --------------------------------------------------------------------------------
   Actual: Pos  19 (tp)    1 (fn)
           Neg   3 (fp)  177 (tn)
   --------------------------------------------------------------------------------
   --------------------------------------------------------------------------------
   {'AUC ROC': 0.9945833333333333,
   'Average Precision': 0.9334649122807017,
   'Brier Score': 0.20820269480995568,
   'Precision/PPV': 0.8636363636363636,
   'Sensitivity': 0.95,
   'Specificity': 0.9833333333333333}
   --------------------------------------------------------------------------------

               precision    recall  f1-score   support

              0     0.99      0.98      0.99       180
              1     0.86      0.95      0.90        20

       accuracy                         0.98       200
      macro avg     0.93      0.97      0.95       200
   weighted avg     0.98      0.98      0.98       200

   --------------------------------------------------------------------------------
.. _Regression:

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
   from xgboost import XGBRegressor
   from sklearn.impute import SimpleImputer
   from sklearn.datasets import fetch_california_housing
   from model_tuner import Model  
  

Step 2: Load the Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Load the California Housing dataset
   data = fetch_california_housing()
   X = pd.DataFrame(data.data, columns=data.feature_names)
   y = pd.Series(data.target, name="target")


Step 3: Create an Instance of the XGBRegressor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   xgb_name = "xgb"
   xgb = XGBRegressor(random_state=222)


Step 4: Define Hyperparameters for XGBoost
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   tuned_parameters_xgb = [
      {
         f"{xgb_name}__learning_rate": [0.1, 0.01, 0.05],
         f"{xgb_name}__n_estimators": [100, 200, 300],  # Number of trees.  
         f"{xgb_name}__max_depth": [3, 5, 7][:1],    # Maximum depth of the trees
         f"{xgb_name}__subsample": [0.8, 1.0][:1],   # Subsample ratio of the 
                                                      # training instances
         f"{xgb_name}__colsample_bytree": [0.8, 1.0][:1],
         f"{xgb_name}__eval_metric": ["logloss"],
         f"{xgb_name}__early_stopping_rounds": [10],
         f"{xgb_name}__tree_method": ["hist"],
         f"{xgb_name}__verbose": [False],
      }
   ]

   xgb_definition = {
      "clc": xgb,
      "estimator_name": xgb_name,
      "tuned_parameters": tuned_parameters_xgb,
      "randomized_grid": False,
      "early": True,
   }

   model_definition = {xgb_name: xgb_definition}

Step 5: Initialize and Configure the ``Model``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``XGBRegressor`` inherently handles missing values (``NaN``) without requiring explicit 
imputation strategies. During training, ``XGBoost`` treats missing values as a 
separate category and learns how to route them within its decision trees. 
Therefore, passing a ``SimpleImputer`` or using an imputation strategy is unnecessary 
when using ``XGBRegressor``.

.. code-block:: python

   kfold = False
   calibrate = False

   # Define model object
   model_type = "xgb"
   clc = model_definition[model_type]["clc"]
   estimator_name = model_definition[model_type]["estimator_name"]

   # Set the parameters by cross-validation
   tuned_parameters = model_definition[model_type]["tuned_parameters"]
   rand_grid = model_definition[model_type]["randomized_grid"]
   early_stop = model_definition[model_type]["early"]

   model_xgb = Model(
      name=f"xgb_{model_type}",
      estimator_name=estimator_name,
      model_type="regression",
      calibrate=calibrate,
      estimator=clc,
      kfold=kfold,
      stratify_y=False,
      grid=tuned_parameters,
      randomized_grid=rand_grid,
      boost_early=early_stop,
      scoring=["r2"],
      random_state=222,
      n_jobs=2,
   )

Step 6: Perform Grid Search Parameter Tuning and Retrieve Split Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model_xgb.grid_search_param_tuning(X, y,)

   X_train, y_train = model_xgb.get_train_data(X, y)
   X_test, y_test = model_xgb.get_test_data(X, y)
   X_valid, y_valid = model_xgb.get_valid_data(X, y)


.. code-block:: bash

   Pipeline Steps:

   ┌────────────────┐
   │ Step 1: xgb    │
   │ XGBRegressor   │
   └────────────────┘

   100%|██████████| 9/9 [00:22<00:00,  2.45s/it]Best score/param set found on validation set:
   {'params': {'xgb__colsample_bytree': 0.8,
               'xgb__early_stopping_rounds': 10,
               'xgb__eval_metric': 'logloss',
               'xgb__learning_rate': 0.1,
               'xgb__max_depth': 3,
               'xgb__n_estimators': 67,
               'xgb__subsample': 0.8,
               'xgb__tree_method': 'hist'},
   'score': 0.7651490279157868}
   Best r2: 0.765 


Step 7: Fit the Model
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model_xgb.fit(
      X_train,
      y_train,
      validation_data=[X_valid, y_valid],
   )

Step 8: Return Metrics (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   Validation Metrics
   ********************************************************************************
   {'Explained Variance': 0.7647451659057567,
   'Mean Absolute Error': 0.3830825326824073,
   'Mean Squared Error': 0.3066172248224347,
   'Median Absolute Error': 0.2672762813568116,
   'R2': 0.7647433075624044,
   'RMSE': 0.5537302816556403}
   ********************************************************************************
   Test Metrics
   ********************************************************************************
   {'Explained Variance': 0.7888942913974833,
   'Mean Absolute Error': 0.3743548199982513,
   'Mean Squared Error': 0.28411432705731066,
   'Median Absolute Error': 0.26315186452865597,
   'R2': 0.7888925135381788,
   'RMSE': 0.533023758436067}
   ********************************************************************************
   {'Explained Variance': 0.7888942913974833,
   'R2': 0.7888925135381788,
   'Mean Absolute Error': 0.3743548199982513,
   'Median Absolute Error': 0.26315186452865597,
   'Mean Squared Error': 0.28411432705731066,
   'RMSE': 0.533023758436067}

Bootstrap Metrics
===========================

The ``bootstrapper.py`` module provides utility functions for input type checking, data resampling, and evaluating bootstrap metrics.

.. function:: check_input_type(x)

   Validates and normalizes the input type for data processing. Converts NumPy arrays, Pandas Series, and DataFrames into a standard Pandas DataFrame with a reset index.

   :param x: Input data (NumPy array, Pandas Series, or DataFrame).
   :type x: array-like
   :returns: Normalized input as a Pandas DataFrame.
   :rtype: pandas.DataFrame
   :raises ValueError: If the input type is not supported.

.. function:: sampling_method(y, n_samples, stratify=False, balance=False, class_proportions=None)

   Resamples a dataset based on specified options for balancing, stratification, or custom class proportions.

   :param y: Target variable to resample.
   :type y: pandas.Series
   :param n_samples: Number of samples to draw.
   :type n_samples: int
   :param stratify: Whether to stratify based on the provided target variable.
   :type stratify: bool, optional
   :param balance: Whether to balance class distributions equally.
   :type balance: bool, optional
   :param class_proportions: Custom proportions for each class. Must sum to 1.
   :type class_proportions: dict, optional
   :returns: Resampled target variable.
   :rtype: pandas.DataFrame
   :raises ValueError: If class proportions do not sum to 1.

.. function:: evaluate_bootstrap_metrics(model=None, X=None, y=None, y_pred_prob=None, n_samples=500, num_resamples=1000, metrics=["roc_auc", "f1_weighted", "average_precision"], random_state=42, threshold=0.5, model_type="classification", stratify=None, balance=False, class_proportions=None)

   Evaluates classification or regression metrics on bootstrap samples using a pre-trained model or pre-computed predictions.

   :param model: Pre-trained model with a ``predict_proba`` method. Required if ``y_pred_prob`` is not provided.
   :type model: object, optional
   :param X: Input features. Not required if ``y_pred_prob`` is provided.
   :type X: array-like, optional
   :param y: Ground truth labels.
   :type y: array-like
   :param y_pred_prob: Pre-computed predicted probabilities.
   :type y_pred_prob: array-like, optional
   :param n_samples: Number of samples per bootstrap iteration. Default is 500.
   :type n_samples: int, optional
   :param num_resamples: Number of bootstrap iterations. Default is 1000.
   :type num_resamples: int, optional
   :param metrics: List of metrics to calculate (e.g., ``"roc_auc"``, ``"f1_weighted"``).
   :type metrics: list of str
   :param random_state: Random seed for reproducibility. Default is 42.
   :type random_state: int, optional
   :param threshold: Classification threshold for probability predictions. Default is 0.5.
   :type threshold: float, optional
   :param model_type: Specifies the task type, either ``"classification"`` or ``"regression"``.
   :type model_type: str
   :param stratify: Variable for stratified sampling.
   :type stratify: pandas.Series, optional
   :param balance: Whether to balance class distributions.
   :type balance: bool, optional
   :param class_proportions: Custom class proportions for sampling.
   :type class_proportions: dict, optional
   :returns: DataFrame with mean and confidence intervals for each metric.
   :rtype: pandas.DataFrame
   :raises ValueError: If invalid parameters or metrics are provided.
   :raises RuntimeError: If sample size is insufficient for metric calculation.


.. note::

   The ``model_tuner_utils.py`` module includes utility functions for evaluating bootstrap metrics in the context of model tuning.

.. function:: return_bootstrap_metrics(X_test, y_test, metrics, threshold=0.5, num_resamples=500, n_samples=500, balance=False)

   Evaluates bootstrap metrics for a trained model using the test dataset. This function supports both classification and regression tasks by leveraging `evaluate_bootstrap_metrics` to compute confidence intervals for the specified metrics.

   :param X_test: Test dataset features.
   :type X_test: pandas.DataFrame
   :param y_test: Test dataset labels.
   :type y_test: pandas.Series or pandas.DataFrame
   :param metrics: List of metric names to calculate (e.g., ``"roc_auc"``, ``"f1_weighted"``).
   :type metrics: list of str
   :param threshold: Threshold for converting predicted probabilities into class predictions. Default is 0.5.
   :type threshold: float, optional
   :param num_resamples: Number of bootstrap iterations. Default is 500.
   :type num_resamples: int, optional
   :param n_samples: Number of samples per bootstrap iteration. Default is 500.
   :type n_samples: int, optional
   :param balance: Whether to balance the class distribution during resampling. Default is False.
   :type balance: bool, optional
   :returns: DataFrame containing mean and confidence intervals for the specified metrics.
   :rtype: pandas.DataFrame
   :raises ValueError: If ``X_test`` or ``y_test`` are not provided as Pandas DataFrames or if unsupported input types are specified.


Bootstrap Metrics Example
-----------------------------

Continuing from the model output object (``model_xgb``) from the :ref:`regression example <Regression>` above, we leverage the ``return_bootstrap_metrics`` method from ``model_tuner_utils.py`` to print bootstrap performance metrics (:math:`R^2` and :math:`\text{explained variance}`) at 95% confidence levels as shown below: 

.. code-block:: python

   print("Bootstrap Metrics")

   model_xgb.return_bootstrap_metrics(
      X_test=X_test,
      y_test=y_test,
      metrics=["r2", "explained_variance"],
      n_samples=30,
      num_resamples=300,
   )


.. code-block:: bash

   Bootstrap Metrics
   100%|██████████| 300/300 [00:00<00:00, 358.05it/s]
   Metric	              Mean  95% CI Lower  95% CI Upper  
   0	             r2   0.781523      0.770853      0.792193  
   1 explained_variance	  0.788341	0.777898      0.798785

