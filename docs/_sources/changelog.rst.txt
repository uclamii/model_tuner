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


