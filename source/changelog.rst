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


Version 0.0.08a
----------------

``AutoKerasClassifier``

- Changed 'layers' key to store count instead of list to avoid exceeding MLflow's 500-char limit.
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


