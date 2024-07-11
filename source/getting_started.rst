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
   This documentation is for ``model_tuner`` version ``0.0.11a``.


What Does Model Tuner Offer?
------------------------------

Model Tuner is a versatile and powerful tool designed to facilitate the training, evaluation, and tuning of machine learning models. It supports various functionalities such as handling imbalanced data, applying different scaling and imputation techniques, calibrating models, and conducting cross-validation. This class is particularly useful for model selection and hyperparameter tuning, ensuring optimal performance across different metrics. Here are the key features it offers:

- **Hyperparameter Tuning**: Facilitates hyperparameter optimization to fine-tune model parameters.
- **Performance Metrics**: Provides bootstrap metrics and general performance metrics to evaluate model effectiveness.
- **Model Calibration**: Supports calibration methods such as ``sigmoid`` and ``isotonic`` to improve probability estimates.
- **Threshold Tuning**: Allows for threshold tuning to optimize classification performance.
- **Custom Pipelines**: Enables the creation of custom pipelines for flexible and robust model workflows.
- **K-Fold Cross-Validation**: Implements ``K-Fold cross-validation`` to assess model performance across different data splits.
- **Stratified Splits**: Ensures balanced splits of data based on target variable distribution.
- **Model Compatibility**: Compatible with all scikit-learn models and select deep learning models (e.g. ``AutoKeras``).
- **XGBoost Integration**: Includes early stopping for ``XGBoost`` to prevent overfitting and improve model performance.
- **Data Imputation**: Supports imputation using ``SimpleImputer`` and is compatible with other imputation strategies.
- **Feature Scaling**: Includes feature scaling methods like ``MinMax`` scaling.
- **Feature Selection**: Offers feature selection techniques such as ``SelectKBest`` and ``Recursive Feature Elimination (RFE)``.
- **Imbalanced Learning**: Provides oversampling methods like ``ADASYN`` and ``SMOTE`` to handle imbalanced datasets.
- **Multi-Class Label Support**: Currently under development to support multi-class classification tasks.



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


