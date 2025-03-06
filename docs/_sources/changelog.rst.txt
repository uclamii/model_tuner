.. _target-link:

.. raw:: html

   <div class="no-click">

.. image:: /../assets/ModelTunerTarget.png
   :alt: Model Tuner Logo
   :align: left
   :width: 350px

.. raw:: html

   </div>

.. raw:: html

   <div style="height: 200px;"></div>

\

Changelog
=======================================

.. important::
   Complete version release history available `here <https://pypi.org/project/model-tuner/#history>`_


Version 0.0.28b (Beta)
-----------------------

- Updated CalibratedClassifier to use validation set by @lshpaner in https://github.com/uclamii/model_tuner/pull/193
- Updated docstring for `stratify_y` in `train_val_test_split` by @lshpaner in https://github.com/uclamii/model_tuner/pull/194

**Full Changelog**: https://github.com/uclamii/model_tuner/compare/0.0.27b...0.0.28b

Version 0.0.27b (Beta)
-----------------------

- Xgboost n estimators fix by @elemets in https://github.com/uclamii/model_tuner/pull/186
- Google colab compatibility - pandas version by @elemets in https://github.com/uclamii/model_tuner/pull/187


**Full Changelog**: https://github.com/uclamii/model_tuner/compare/0.0.26b...0.0.27b


Version 0.0.26b (Beta)
-----------------------

- Optimal threshold: Users can now specify target precision or recall and an optimal threshold is computed for that
- Finalised testing: coverage is now at 86% total
- New get_feature_names() helper function for extracting features
- n_estimators calculation for boosting algorithms is now fixed

Version 0.0.25a
----------------

- Pushed fixes for the get_feature_selection_pipeline method.
- Updated scoring blocks for calibrated KFold models and folded confusion matrix metrics.
- Added unittests for edge cases, including test_rfe_calibrate_model() and validating confusion matrix alignment.
- Fixed mismatches between confusion matrix and classification report.
- Provided fixes for all pipeline getter methods.
- Integrated verify_imb_sampler prints into KFold logic.
- Resolved typos in group split configurations and refined nested KFold bug fixes.
- Adjusted fold metric calculations in report_model_metrics.
- Moved optimal threshold logic into prediction functionality.
- Enhanced return metrics dictionary logic to handle all cases and added multilabel classification tests.
- Addressed Brier score calculation issues and optimized regression reports for KFold.
- Introduced threshold print updates for clearer reporting.
- Implemented SHAP scripts and tests for model explainability.
- Removed outdated calibration reports from documentation and codebase.
- Fixed bugs in regression metric calculations and refined KFold metric aggregation.


Version 0.0.24a
-----------------

- Updated .gitignore to incl. doctrees
- Added pickleObjects tests and updated reqs, tests passed
- Added boostrapper test and tests passed
- Adding multi class test script
- Updated Metrics Output
- Added optl' threshold print inside return_metrics
- KFold metric printing
- Augmented predict_proba test, and train_val_test_split
- Fixed pipeline_steps arg in model definition
- Refactored metrics_df in report_model_metrics for aesthetics
- Unit Tests
- Made return_dict optional in return_metrics
- Added openpyxl versions for all python versions in requirements.txt
- Refactor metrics, foldwise metrics and foldwise con_mat, class_labels
- Cleaned notebooks dir
- Added model_tuner version print to scripts
- Added fix for sort of pipeline_steps now optional:
- Added required model_tuner import to xgb_multi.py
- Added requisite model_tuner import to multi_class_test.py
- Added catboost_multi_class.py script
- Removed pip dependency from requirements

Version 0.0.23a
--------------------

- Fixed a bug found when calibrating early stopping models
- Fixed early stopping in Column Transformer application


Version 0.0.22a
--------------------

- Fixed an issue where the feature selection name was not referenced correctly, causing a bug when printing selected feature names with the updated pipeline.
- Removed resolved print statements from April, 2024.


Version 0.0.21a
--------------------

- Specified the pipeline class otherwise the method just returned a list
- Removed need to specify ``self.estimator`` when its called
- Generalized (renamed) ``"K Best Features"`` to just ``"Best Features"`` inside returns of  ``return_metrics()``
- Generalized (renamed) ``k_best_features`` to ``best_features`` 


Version 0.0.20a
--------------------

- Added flexibility between ``boolean`` and ``None`` for stratification inputs
- Added custom exception for non pandas inputs in ``return_bootstrap_metrics``
- Enforced required ``model_type`` input to be specified as ``"classification"`` or ``"regression"``
- Removed extraneous ``"="`` print below ``pipeline_steps``
- Handled missing ``pipeline_steps`` when using ``imbalance_sampler`` 
- Updated requirements for ``python==3.11``
- Fixed SMOTE for early stopping
- Removed extra ``model_type`` input from ``xgb_early_test.py``


Version 0.0.19a
--------------------

- Requirements updated again to make compatible with google colab out of the box.
- Bug in ``fit()`` method where ``best_params`` wasn't defined if we didn't specify a score
- Threshold bug now actually fixed. Specificity and other metrics should reflect this. (Defaults to 0.5 if optimal_threshold is not specified). 

Version 0.0.18a
--------------------

- Updated requirements to include ``numpy`` versions ``<1.26`` for Python 3.8-3.11.

This should stop a rerun occurring when using the library on a google colab.


Version 0.0.17a
--------------------

Major fixes:

- Verbosity variable is now popped from the parameters before the fit
- Bug with Column Transformer early stopping fixed (valid set is now transformed correctly)
- Return metrics now has a consistent naming convention  
- ``report_model_metrics`` is now using the correct threshold in all cases
- Default values updated for ``train_val_test_split``  
- ``tune_threshold_Fbeta`` is now called with the correct number of parameters in all cases
- Requirements updates: ``XGBoost`` updated to ``2.1.2`` for later Python versions.

Minor changes:

- ``help(model_tuner)`` should now be correctly formatted in google colab

Version 0.0.16a
--------------------

- Custom pipeline steps now updated (our pipeline usage has been completely changed and should now order itself and support non named steps) always ensures correct order
- This fixed multiple other issues that were occuring to do with logging of imbalanced learn 
- Reporting model metrics now works.
- ``AutoKeras`` code deprecated and removed.
- ``KFold`` bug introduced because of ``CatBoost``. This has now been fixed.
- Pretty print of pipeline.
- Boosting variable has been renamed.
- Version constraints have been updated and refactored.
- ``tune_threshold_Fbeta`` has been cleaned up to remove unused parameters.
- ``train_val_test`` unnecessary self removed and taken outside of class method.
- deprecated ``setup.py`` in favor of ``pyproject.toml`` per forthcoming ``pip25`` update.

Version 0.0.15a
--------------------

Contains all previous fixes relating to:

- ``CatBoost`` support (early stopping, and support involving resetting estimators).
- Pipeline steps now support hyperparameter tuning of the resamplers (``SMOTE``, ``ADASYN``, etc.).
- Removed older implementations of impute and scaling and moved onto supporting only custom ``pipeline_steps``. 
- Fixed bugs in stratification with regards to length mismatch of dependent variable when using column names to stratify. 
- Cleaned a removed multiple lines of unused code and unused initialisation parameters. 


Version 0.0.014a
------------------

In previous versions, the ``train_val_test_split`` method allowed for stratification 
either by `y` (``stratify_y``) or by specified columns (``stratify_cols``), but 
not both at the same time. There are use cases where stratification by both the target 
variable (`y`) and specific columns is necessary to ensure a balanced and representative 
split across different data segments.

**Enhancement**

Modified the ``train_val_test_split`` method to support simultaneous stratification 
by both ``stratify_y`` and ``stratify_cols``. This was inside the method achieved 
by implementing the following logic that ensures both y and the specified columns 
are considered during the stratification process.

.. code-block:: python

   stratify_key = pd.concat([X[stratify_cols], y], axis=1)

   strat_key_val_test = pd.concat(
      [X_valid_test[stratify_cols], y_valid_test], axis=1
   )


Version 0.0.013a
------------------

- Updated bootstrapper 
- ``evaluate_bootstrap_metrics``
- Added ``notebooks/xgb_early_bootstrap_test.py`` to test it
- Updated ``requirements.txt`` file for dev testing
- Fixed sampling error on low number of samples inside bootstrapper


Version 0.0.012a
------------------

- ``Xgboost`` bug fixes
- Zenodo updates
- Pickle model fixes with ``np`` import
- ``ADASYN`` and ``SMOTE`` fix with no fit happening when calibrating


Version 0.0.011a
------------------

- updated readme for PyPI
- previous version not saved on setup; re-release to ``0.0.11a``


Version 0.0.010a
-----------------

- updated readme for PyPI

Version 0.0.09a
----------------

- number of estimators now extracted from ``XGBoost`` model object
- early stopping fixed


Version 0.0.08a
----------------

``AutoKerasClassifier``

- Changed ``layers`` key to store count instead of list to avoid exceeding MLflow's 500-char limit.
- Simplified function by removing key filtering loop.


Version 0.0.07a
----------------

- Kfold threshold tuning fix 


Version 0.0.06a
----------------

- Updating best_params: ref before assignment bug


Version 0.0.05a
----------------

- Bootstrapper:
  - Fixed import bugs
  - Fixed Assertion bug to do with metrics not being assigned
- Early stopping:
  - Leon: fixed bug with `SelectKBest` and `ADASYN` where the wrong code chunk was being utilized
  - Arthur: Verbosity fix


Version 0.0.02a
----------------

- temporarily commented out updated apache software license string in setup.py
- updated logo resolution


