.. _getting_started:

.. _target-link:
.. image:: /../assets/ModelTunerTarget.png
   :alt: Model Tuner Logo
   :align: left
   :width: 350px

.. raw:: html

   <div style="height: 200px;"></div>

\

Welcome to Model Tuner's Documentation!
=======================================

.. important::
   This documentation is for ``model_tuner`` version ``0.0.08a``.


Model Tuner is a versatile and powerful tool designed to facilitate the training, evaluation, and tuning of machine learning models. It supports various functionalities such as handling imbalanced data, applying different scaling and imputation techniques, calibrating models, and conducting cross-validation. This class is particularly useful for model selection and hyperparameter tuning, ensuring optimal performance across different metrics.   

.. _prerequisites:   

Prerequisites
-------------
Before you install ``model_tuner``, ensure your system meets the following requirements:

- **Python**: Version ``3.7`` or higher is required to run ``model_tuner``.

Additionally, ``model_tuner`` depends on the following packages, which will be automatically installed when you install ``model_tuner`` using pip:

- **NumPy**: Version ``1.21.6`` or higher
- **Pandas**: Version ``1.3.5`` or higher
- **joblib**: Version ``1.3.2`` or higher
- **scikit-learn**: Version ``1.0.2`` or higher
- **scipy**: Version ``1.7.3`` or higher
- **tqdm**: Version ``4.66.4`` or higher

.. _installation:

Installation
-------------

You can install ``model_tuner`` directly from PyPI:

.. code-block:: bash

    pip install model_tuner


