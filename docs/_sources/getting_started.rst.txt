.. _getting_started:

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

Welcome to Model Tuner's Documentation!
========================================

.. important::
   This documentation is for ``model_tuner`` version ``0.0.028b``.


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
- **Model Compatibility**: Compatible with all scikit-learn models.
- **XGBoost and CatBoost Integration**: Includes early stopping for ``XGBoost`` and ``CatBoost`` to prevent overfitting and improve model performance.
- **Data Imputation**: Supports imputation using ``SimpleImputer`` and is compatible with other imputation strategies.
- **Feature Scaling**: Includes feature scaling methods like ``MinMax`` scaling.
- **Feature Selection**: Offers feature selection techniques such as ``SelectKBest`` and ``Recursive Feature Elimination (RFE)``.
- **Imbalanced Learning**: Provides oversampling methods like ``ADASYN`` and ``SMOTE`` to handle imbalanced datasets.
- **Multi-Class Label Support**: Currently under development to support multi-class classification tasks.


.. _prerequisites:   

Prerequisites
-------------
Before installing ``model_tuner``, ensure your system meets the following requirements:

- **Python:** version ``3.7`` or higher.

The ``model_tuner`` library includes different dependencies based on Python versions, 
which will be automatically installed when you install ``model_tuner`` using pip. Below are the key dependencies:

- For Python ``3.7``:

   - ``numpy``: version ``1.21.4``
   - ``pandas``: version ``1.1.5``
   - ``scikit-learn``: version ``0.23.2``
   - ``scipy``: version ``1.4.1``
   - ``joblib``: version ``1.3.2``
   - ``tqdm``: version ``4.66.4``
   - ``imbalanced-learn``: version ``0.7.0``
   - ``scikit-optimize``: version ``0.8.1``
   - ``xgboost``: version ``1.6.2``
   - ``pip``: version ``24.0``

- For Python ``3.8`` to ``<3.11``:

   - ``numpy``: versions between ``1.19.5`` and ``<2.0.0``
   - ``pandas``: versions between ``1.3.5`` and ``<2.2.3``
   - ``scikit-learn``: versions between ``1.0.2`` and ``<1.4.0``
   - ``scipy``: versions between ``1.6.3`` and ``<1.11``
   - ``joblib``: version ``1.3.2``
   - ``tqdm``: version ``4.66.4``
   - ``imbalanced-learn``: version ``0.12.4``
   - ``scikit-optimize``: version ``0.10.2``
   - ``xgboost``: version ``2.1.2``
   - ``pip``: version ``24.2``
   - ``setuptools``: version ``75.1.0``
   - ``wheel``: version ``0.44.0``

- For Python ``3.11`` or higher:

   - ``numpy``: versions between ``1.19.5`` and ``<2.0.0``
   - ``pandas``: versions between ``1.3.5`` and ``<2.2.2``
   - ``scikit-learn``: version ``1.5.1``
   - ``scipy``: version ``1.14.0``
   - ``joblib``: version ``1.3.2``
   - ``tqdm``: version ``4.66.4``
   - ``imbalanced-learn``: version ``0.12.4``
   - ``scikit-optimize``: version ``0.10.2``
   - ``xgboost``: version ``2.1.2``
   - ``pip``: version ``24.2``
   - ``setuptools``: version ``75.1.0``
   - ``wheel``: version ``0.44.0``

.. _installation:

Installation
-------------

You can install ``model_tuner`` directly from PyPI:

.. code-block:: bash

    pip install model_tuner


