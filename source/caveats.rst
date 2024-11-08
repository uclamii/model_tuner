.. _caveats:   

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



Zero Variance Columns
-----------------------

.. important::

   Ensure that your feature set `X` is free of zero-variance columns before using this method. 
   Zero-variance columns can lead to issues such as ``UserWarning: Features[feat_num] are constant`` 
   and ``RuntimeWarning: invalid value encountered in divide f = msb/msw`` during the model training process.

   To check for and remove zero-variance columns, you can use the following code:

   .. code-block:: python

      # Check for zero-variance columns and drop them
      zero_variance_columns = X.columns[X.var() == 0]
      if not zero_variance_columns.empty:
          X = X.drop(columns=zero_variance_columns)

Zero-variance columns in the feature set :math:`X` refer to columns where all values are identical.
Mathematically, if :math:`X_j` is a column in :math:`X`, the variance of this column is calculated as:

.. math::

   \text{Var}(X_j) = \frac{1}{n} \sum_{i=1}^{n} (X_{ij} - \bar{X}_j)^2 = 0

where :math:`X_{ij}` is the :math:`i`-th observation of feature :math:`j`, and :math:`\bar{X}_j` is the mean of the :math:`j`-th feature. 
Since all :math:`X_{ij}` are equal, :math:`\text{Var}(X_j)` is zero.

Effects on Model Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. **UserWarning:**

   During model training, algorithms often check for variability in features to determine their usefulness in predicting the target variable. A zero-variance column provides no information, leading to the following warning:

   .. code-block:: text

      UserWarning: Features[feat_num] are constant

   This indicates that the feature :math:`X_j` has no variability and, therefore, cannot contribute to the model's predictive power.

2. **RuntimeWarning:**

   When calculating metrics like the F-statistic used in Analysis of Variance (ANOVA) or feature importance metrics, the following ratio is computed:

   .. math::

      F = \frac{\text{MSB}}{\text{MSW}}

   where :math:`\text{MSB}` (Mean Square Between) and :math:`\text{MSW}` (Mean Square Within) are defined as:

   .. math::

      \text{MSB} = \frac{1}{k-1} \sum_{j=1}^{k} n_j (\bar{X}_j - \bar{X})^2

   .. math::

      \text{MSW} = \frac{1}{n-k} \sum_{j=1}^{k} \sum_{i=1}^{n_j} (X_{ij} - \bar{X}_j)^2

   If :math:`X_j` is a zero-variance column, then :math:`\text{MSW} = 0` because all :math:`X_{ij}` are equal to :math:`\bar{X}_j`. This leads to a division by zero in the calculation of :math:`F`:

   .. math::

      F = \frac{\text{MSB}}{0} \rightarrow \text{undefined}

   which triggers a runtime warning:

   .. code-block:: text

      RuntimeWarning: invalid value encountered in divide f = msb/msw

   indicating that the calculation involves dividing by zero, resulting in undefined or infinite values.

To avoid these issues, ensure that zero-variance columns are removed from :math:`X` before proceeding with model training.


Dependent Variable
-------------------

.. important::

   Additionally, ensure that `y` (the target variable) is passed as a Series and not as a DataFrame.
   Passing `y` as a DataFrame can cause issues such as ``DataConversionWarning: A column-vector y was passed 
   when a 1d array was expected. Please change the shape of y to (n_samples,)``. 

   If `y` is a DataFrame, you can convert it to a Series using the following code:

   .. code-block:: python

      # Convert y to a Series if it's a DataFrame
      if isinstance(y, pd.DataFrame):
          y = y.squeeze()

   This conversion ensures that the target variable `y` has the correct shape, preventing the aforementioned warning.


Target Variable Shape and Its Effects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The target variable :math:`y` should be passed as a 1-dimensional array (Series) and not as a 2-dimensional array (DataFrame).
If :math:`y` is passed as a DataFrame, the model training process might raise the following warning:

.. code-block:: text

   DataConversionWarning: A column-vector y was passed when a 1d array was expected. 
   Please change the shape of y to (n_samples,).

**Explanation:**

Machine learning models generally expect the target variable :math:`y` to be in the shape of a 1-dimensional array, 
denoted as :math:`y = \{y_1, y_2, \dots, y_n\}`, where :math:`n` is the number of samples. 
Mathematically, :math:`y` is represented as:

.. math::

   y = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{pmatrix}

When :math:`y` is passed as a DataFrame, it is treated as a 2-dimensional array, which has the form:

.. math::

   y = \begin{pmatrix} y_1, y_2, \dots , y_n \end{pmatrix}

or 

.. math::

   y = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{pmatrix}

where each sample is represented as a column vector. This discrepancy in dimensionality can cause the model to misinterpret the data, 
leading to the ``DataConversionWarning``.

Solution
^^^^^^^^^^
To ensure :math:`y` is interpreted correctly as a 1-dimensional array, it should be passed as a Series. 
If :math:`y` is currently a DataFrame, you can convert it to a Series using the following code:

.. code-block:: python

   # Convert y to a Series if it's a DataFrame
   if isinstance(y, pd.DataFrame):
         y = y.squeeze()

The method :code:`squeeze()` effectively removes any unnecessary dimensions, converting a 2-dimensional DataFrame 
with a single column into a 1-dimensional Series. This ensures that :math:`y` has the correct shape, preventing 
the aforementioned warning and ensuring the model processes the target variable correctly.



Imputation Before Scaling
----------------------------

**Ensuring Correct Data Preprocessing Order: Imputation Before Scaling**

.. important:: 
   It is crucial to apply imputation before scaling during the data preprocessing 
   pipeline to preserve the mathematical integrity of the transformations. The 
   correct sequence for the pipeline is as follows:

   .. code:: python

      pipeline_steps = [
         ("Preprocessor", SimpleImputer()),
         ("StandardScaler", StandardScaler()),
      ]

1. Accurate Calculation of Scaling Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Scaling methods, such as standardization or min-max scaling, rely on the calculation of statistical properties, such as the mean (:math:`\mu`), standard deviation (:math:`\sigma`), minimum (:math:`x_{\min}`), and maximum (:math:`x_{\max}`) of the dataset. These statistics are computed over the full set of available data. If missing values are present during this calculation, the resulting parameters will be incorrect, leading to improper scaling.

For example, in Z-score standardization, the transformation is defined as:

.. math::

   z = \frac{x - \mu}{\sigma}

where :math:`\mu = \frac{1}{N} \sum_{i=1}^{N} x_i` and :math:`\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2}`, with :math:`N` representing the number of data points. If missing values are not imputed first, both :math:`\mu` and :math:`\sigma` will be computed based on incomplete data, resulting in inaccurate transformations for all values.

In contrast, if we impute the missing values first (e.g., by replacing them with the mean or median), the complete dataset is used for calculating these parameters. This ensures the scaling transformation is applied consistently across all data points.

2. Consistency in Data Transformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Imputing missing values before scaling ensures that the transformation applied is consistent across the entire dataset, including previously missing values. For instance, consider a feature :math:`X = [1, 2, \text{NaN}, 4, 5]`. If we impute the missing value using the mean (:math:`\mu = \frac{1 + 2 + 4 + 5}{4} = 3`), the imputed dataset becomes:

.. math::

   X_{\text{imputed}} = [1, 2, 3, 4, 5]

Now, applying standardization on the imputed dataset results in consistent Z-scores for each value, based on the correct parameters :math:`\mu = 3` and :math:`\sigma = 1.58`.

Had scaling been applied first, without imputing, the calculated mean and standard deviation would be incorrect, leading to inconsistent transformations when imputation is subsequently applied. For example, if we calculated:

.. math::

   z_{\text{incomplete}} = \frac{x - 3}{1.58} \quad \text{(based on non-imputed data)}

and later imputed the missing value, the transformed imputed value would not be aligned with the scaled distribution.

3. Prevention of Distortion in Scaling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Placeholder values used to represent missing data (e.g., large negative numbers like -999) can severely distort scaling transformations if not handled prior to scaling. In min-max scaling, the transformation is:

.. math::

   x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}

where :math:`x_{\min}` and :math:`x_{\max}` represent the minimum and maximum values of the feature. If a placeholder value like -999 is included, the range :math:`x_{\max} - x_{\min}` will be artificially inflated, leading to a heavily skewed scaling of all values. For instance, the min-max scaling of :math:`X = [1, 2, -999, 4, 5]` would produce extreme distortions due to the influence of -999 on :math:`x_{\min}`.

By imputing missing values before scaling, we avoid these distortions, ensuring that the scaling operation reflects the true range of the data.




Column Stratification with Cross-Validation
---------------------------------------------
.. important::

   **Using** ``stratify_cols`` **with Cross-Validation**

   It is important to note that ``stratify_cols`` cannot be used when performing cross-validation.
   Cross-validation involves repeatedly splitting the dataset into training and validation sets to 
   evaluate the model's performance across different subsets of the data. 

   **Explanation:**

   When using cross-validation, the process automatically handles the stratification of the target variable :math:`y`, 
   if specified. This ensures that each fold is representative of the overall distribution of :math:`y`. However, 
   ``stratify_cols`` is designed to stratify based on specific columns in the feature set :math:`X`, which can lead to 
   inconsistencies or even errors when applied in the context of cross-validation.

   Since cross-validation inherently handles stratification based on the target variable, attempting to apply 
   additional stratification based on specific columns would conflict with the cross-validation process. 
   This can result in unpredictable behavior or failure of the cross-validation routine.

   However, you can use ``stratify_y`` during cross-validation to ensure that each fold of the dataset is representative 
   of the distribution of the target variable :math:`y`. This is a common practice to maintain consistency in the distribution 
   of the target variable across the different training and validation sets.


Cross-Validation and Stratification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`D = \{(X_i, y_i)\}_{i=1}^n` be the dataset with :math:`n` samples, where :math:`X_i` is the feature set and :math:`y_i` is the target variable.

In `k-fold` cross-validation, the dataset :math:`D` is split into :math:`k` folds :math:`\{D_1, D_2, \dots, D_k\}`.

When stratifying by :math:`y` using :code:`stratify_y`, each fold :math:`D_j` is constructed such that the distribution of :math:`y` in each fold is similar to the distribution of :math:`y` in :math:`D`.

Mathematically, if :math:`P(y=c)` is the probability of the target variable :math:`y` taking on class :math:`c`, then:

.. math::

    P(y=c \mid D_j) \approx P(y=c \mid D)

for all folds :math:`D_j` and all classes :math:`c`.

This ensures that the stratified folds preserve the same class proportions as the original dataset.

On the other hand, :code:`stratify_cols` stratifies based on specific columns of :math:`X`. However, in cross-validation, the primary focus is on the target variable :math:`y`.

Attempting to stratify based on :math:`X` columns during cross-validation can disrupt the process of ensuring a representative sample of :math:`y` in each fold. This can lead to unreliable performance estimates and, in some cases, errors.

Therefore, the use of :code:`stratify_y` is recommended during cross-validation to maintain consistency in the target variable distribution across folds, while :code:`stratify_cols` should be avoided.




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

