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


.. _model_calibration:

Model Calibration
--------------------

Model calibration refers to the process of adjusting the predicted probabilities of a model so that they more accurately reflect the true likelihood of outcomes. This is crucial in machine learning, particularly for classification problems where the model outputs probabilities rather than just class labels.

Goal of Calibration
^^^^^^^^^^^^^^^^^^^^^

The goal of calibration is to ensure that the predicted probability :math:`\hat{p}(x)` is equal to the true probability that :math:`y = 1` given :math:`x`. Mathematically, this can be expressed as:

.. math::

    \hat{p}(x) = P(y = 1 \mid \hat{p}(x) = p)

This equation states that for all instances where the model predicts a probability :math:`p`, the true fraction of positive cases should also be :math:`p`.

Calibration Curve
^^^^^^^^^^^^^^^^^^^^^

To assess calibration, we often use a *calibration curve*. This involves:

1. **Binning** the predicted probabilities :math:`\hat{p}(x)` into intervals (e.g., [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]).
2. **Calculating the mean predicted probability** :math:`\hat{p}_i` for each bin :math:`i`.
3. **Calculating the empirical frequency** :math:`f_i` (the fraction of positives) in each bin.

For a perfectly calibrated model:

.. math::

    \hat{p}_i = f_i \quad \text{for all bins } i

Brier Score
^^^^^^^^^^^^^^^^^^^^^^

The **Brier score** is one way to measure the calibration of a model. It’s calculated as:

.. math::

    \text{Brier Score} = \frac{1}{N} \sum_{i=1}^{N} (\hat{p}(x_i) - y_i)^2

Where:

- :math:`N` is the number of instances.
- :math:`\hat{p}(x_i)` is the predicted probability for instance :math:`i`.
- :math:`y_i` is the actual label for instance :math:`i` (0 or 1).

The Brier score penalizes predictions that are far from the true outcome. A lower Brier score indicates better calibration and accuracy.

Platt Scaling
^^^^^^^^^^^^^^^^^^^^^^^

One common method to calibrate a model is **Platt Scaling**. This involves fitting a logistic regression model to the predictions of the original model. The logistic regression model adjusts the raw predictions :math:`\hat{p}(x)` to output calibrated probabilities.

Mathematically, Platt scaling is expressed as:

.. math::

    \hat{p}_{\text{calibrated}}(x) = \frac{1}{1 + \exp(-(A \hat{p}(x) + B))}

Where :math:`A` and :math:`B` are parameters learned from the data. These parameters adjust the original probability estimates to better align with the true probabilities.

Isotonic Regression
^^^^^^^^^^^^^^^^^^^^^^^^

Another method is **Isotonic Regression**, a non-parametric approach that fits a piecewise constant function. Unlike Platt Scaling, which assumes a logistic function, Isotonic Regression only assumes that the function is monotonically increasing. The goal is to find a set of probabilities :math:`p_i` that are as close as possible to the true probabilities while maintaining a monotonic relationship.

The isotonic regression problem can be formulated as:

.. math::

    \min_{p_1 \leq p_2 \leq \dots \leq p_n} \sum_{i=1}^{n} (p_i - y_i)^2

Where :math:`p_i` are the adjusted probabilities, and the constraint ensures that the probabilities are non-decreasing.

Example: Calibration in Logistic Regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In a standard logistic regression model, the predicted probability is given by:

.. math::

    \hat{p}(x) = \sigma(w^\top x) = \frac{1}{1 + \exp(-w^\top x)}

Where :math:`w` is the vector of weights, and :math:`x` is the input feature vector.

If this model is well-calibrated, :math:`\hat{p}(x)` should closely match the true conditional probability :math:`P(y = 1 \mid x)`. If not, techniques like Platt Scaling or Isotonic Regression can be applied to adjust :math:`\hat{p}(x)` to be more accurate.

Summary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Model calibration** is about aligning predicted probabilities with actual outcomes.
- **Mathematically**, calibration ensures :math:`\hat{p}(x) = P(y = 1 \mid \hat{p}(x) = p)`.
- **Platt Scaling** and **Isotonic Regression** are two common methods to achieve calibration.
- **Brier Score** is a metric that captures both the calibration and accuracy of probabilistic predictions.

Calibration is essential when the probabilities output by a model need to be trusted, such as in risk assessment, medical diagnosis, and other critical applications.


Using Imputation and Scaling in Pipeline Steps for Model Preprocessing
-------------------------------------------------------------------------

The ``pipeline_steps`` parameter accepts a list of tuples, where each tuple specifies 
a transformation step to be applied to the data. For example, the code block below 
performs imputation followed by standardization on the dataset before training the model.

.. code-block:: python

   pipeline_steps=[
      ("Imputer", SimpleImputer()), 
      ("StandardScaler", StandardScaler()),
   ] 

When Is Imputation and Feature Scaling in ``pipeline_steps`` Beneficial?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Logistic Regression:** Highly sensitive to feature scaling and missing data. Preprocessing steps like imputation and standardization improve model performance significantly.
- **Linear Models (e.g., Ridge, Lasso):** Similar to Logistic Regression, these models require feature scaling for optimal performance.
- **SVMs:** Sensitive to the scale of the features, requiring preprocessing like standardization.

Models Not Benefiting From Imputation and Scaling in ``pipeline_steps``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Tree-Based Models (e.g., XGBoost, Random Forests, Decision Trees):** These models are invariant to feature scaling and can handle missing values natively. Passing preprocessing steps like StandardScaler or Imputer may be redundant or even unnecessary.

Why Doesn't XGBoost Require Imputation and Scaling in ``pipeline_steps``?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

XGBoost and similar tree-based models work on feature splits rather than feature values directly. This makes them robust to unscaled data and capable of handling missing values using default mechanisms like missing parameter handling in XGBoost. Thus, adding steps like scaling or imputation often does not improve and might complicate the training process.

To this end, it is best to use ``pipeline_steps`` strategically for algorithms that rely on numerical properties (e.g., Logistic Regression). For XGBoost, focus on other optimization techniques like hyperparameter tuning and feature engineering instead.

Caveats in Imbalanced Learning
----------------------------------

Working with imbalanced datasets introduces several challenges that must be carefully addressed 
to ensure model performance is both effective and fair. Below are key caveats to consider:

Bias from Class Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In imbalanced datasets, the prior probabilities of the classes are highly skewed:

.. math::

    P(Y = c_{\text{minority}}) \ll P(Y = c_{\text{majority}})

This imbalance can lead models to prioritize the majority class, resulting in biased predictions 
that overlook the minority class. Models may optimize for accuracy but fail to capture the true 
distribution of minority class instances.

Threshold-Dependent Predictions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many classifiers rely on a decision threshold :math:`\tau` to make predictions:

.. math::

    \text{Predict } c_{\text{minority}} \text{ if } \hat{P}(Y = c_{\text{minority}} \mid X) \geq \tau

With imbalanced data, the default threshold may favor the majority class, causing a high rate of 
false negatives for the minority class. Adjusting the threshold to account for imbalance can 
help mitigate this issue, but it requires careful tuning and validation.

.. _Limitations_of_Accuracy:

Limitations of Accuracy
^^^^^^^^^^^^^^^^^^^^^^^^^^

Traditional accuracy is a misleading metric in imbalanced datasets. For example, a model predicting 
only the majority class can achieve high accuracy despite failing to identify any minority class instances. 
Instead, alternative metrics should be used:

.. _Precision:

- **Precision** for the minority class:

   Measures the proportion of correctly predicted minority class instances out of all 
   instances predicted as the minority class.
  
  .. math::

      \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}

.. _Recall:

- **Recall** for the minority class:

   Measures the proportion of correctly predicted minority class instances out of all actual 
   minority class instances.

  .. math::

      \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}

.. _F1_Score:

- **F1-Score**, the harmonic mean of precision and recall:

   Balances precision and recall to provide a single performance measure.

  .. math::

      F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}

- **ROC AUC (Receiver Operating Characteristic - Area Under the Curve)**:

   Measures the model's ability to distinguish between classes. It is the area under the 
   ROC curve, which plots the True Positive Rate (Recall) against the False Positive Rate.

  .. math::

      \text{True Positive Rate (TPR)} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}

  .. math::

      \text{False Positive Rate (FPR)} = \frac{\text{False Positives}}{\text{False Positives} + \text{True Negatives}}

\

      The AUC (Area Under Curve) is computed by integrating the ROC curve:

      .. math::

            \text{AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}) \, d(\text{FPR})

      This integral represents the total area under the ROC curve, where:

      - A value of 0.5 indicates random guessing.
      - A value of 1.0 indicates a perfect classifier.

         Practically, the AUC is estimated using numerical integration techniques such as the trapezoidal rule 
         over the discrete points of the ROC curve.

Integration and Practical Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ROC AUC provides an aggregate measure of model performance across all classification thresholds. 

However:

- **Imbalanced Datasets**: The ROC AUC may still appear high if the classifier performs well on the majority class, even if the minority class is poorly predicted. 
  In such cases, metrics like Precision-Recall AUC are more informative.
- **Numerical Estimation**: Most implementations (e.g., in scikit-learn) compute the AUC numerically, ensuring fast and accurate computation.

These metrics provide a more balanced evaluation of model performance on imbalanced datasets. By using metrics like ROC AUC in conjunction with precision, recall, and F1-score, practitioners 
can better assess a model's effectiveness in handling imbalanced data.

Impact of Resampling Techniques
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Resampling methods such as oversampling and undersampling can address class imbalance but come with trade-offs:

**Oversampling Caveats**

  - Methods like SMOTE may introduce synthetic data that does not fully reflect the true distribution of the minority class.
  - Overfitting to the minority class is a risk if too much synthetic data is added.

**Undersampling Caveats**

  - Removing samples from the majority class can lead to loss of important information, reducing the model's generalizability.


SMOTE: A Mathematical Illustration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SMOTE (Synthetic Minority Over-sampling Technique) is a widely used algorithm for addressing 
class imbalance by generating synthetic samples for the minority class. However, while powerful, 
SMOTE comes with inherent caveats that practitioners should understand. Below is a mathematical 
illustration highlighting these caveats.

**Synthetic Sample Generation**

SMOTE generates synthetic samples by interpolating between a minority class sample and its nearest 
neighbors. Mathematically, a synthetic sample :math:`x_{synthetic}` is defined as:

.. math::

    \mathbf{x}_{\text{synthetic}} = \mathbf{x}_i + \delta \cdot (\mathbf{x}_k - \mathbf{x}_i)

where:

- :math:`\mathbf{x}_i`: A minority class sample.
- :math:`\mathbf{x}_k`: One of its :math:`k` nearest neighbors (from the same class).
- :math:`\delta`: A random value drawn from a uniform distribution, :math:`\delta \sim U(0, 1)`.

This process ensures that synthetic samples are generated along the line segments connecting 
minority class samples and their neighbors.

**Caveats in Application**

1. **Overlapping Classes**:

   - SMOTE assumes that the minority class samples are well-clustered and separable from the majority class.
   - If the minority class overlaps significantly with the majority class, synthetic samples may fall into regions dominated by the majority class, leading to misclassification.

2. **Noise Sensitivity**:

   - SMOTE generates synthetic samples based on existing minority class samples, including noisy or mislabeled ones.
   - Synthetic samples created from noisy data can amplify the noise, degrading model performance.

3. **Feature Space Assumptions**:

   - SMOTE relies on linear interpolation in the feature space, which assumes that the feature space is homogeneous.
   - In highly non-linear spaces, this assumption may not hold, leading to unrealistic synthetic samples.

4. **Dimensionality Challenges**:

   - In high-dimensional spaces, nearest neighbor calculations may become less meaningful due to the curse of dimensionality.
   - Synthetic samples may not adequately represent the true distribution of the minority class.

5. **Risk of Overfitting**:

   - If SMOTE is applied excessively, the model may overfit to the synthetic minority class samples, reducing generalizability to unseen data.

Example of Synthetic Sample Creation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To illustrate, consider a minority class sample :math:`f{x}_i = [1, 2]` and its nearest neighbor 
:math:`f{x}_k = [3, 4]`. If :math:`\delta = 0.5`, the synthetic sample is computed as:

.. math:: 
   \mathbf{x}_{\text{synthetic}} = [1, 2] + 0.5 \cdot ([3, 4] - [1, 2])

.. math:: 
   \mathbf{x}_{\text{synthetic}} = [2, 3]

This synthetic sample lies midway between the two points in the feature space.

Mitigating the Caveats
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Combine SMOTE with Undersampling**: Techniques like ``SMOTEENN`` or ``SMOTETomek`` remove noisy or overlapping samples after synthetic generation.  

- **Apply with Feature Engineering**: Ensure the feature space is meaningful and represents the underlying data structure.  

- **Tune Oversampling Ratio**: Avoid generating excessive synthetic samples to reduce overfitting.


.. _Threshold_Tuning_Considerations:

Threshold Tuning Considerations
-----------------------------------

**Mathematical Basis**:

In binary classification, the decision rule is represented as:

.. math::

   \hat{y} =
   \begin{cases} 
      1 & \text{if } P(\text{positive class} \mid X) > \tau \\
      0 & \text{otherwise}
   \end{cases}

Here:

- :math:`P(\text{positive class} \mid X)` is the predicted probability of the positive class given features :math:`X`.
- :math:`\tau` is the threshold value, which determines the decision boundary.

By default, :math:`\tau = 0.5`, but this may not always align with the desired balance between precision and recall.

The Precision-Recall Tradeoff
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When tuning the threshold :math:`\tau`, it is important to recognize its impact on precision and recall:

- :ref:`Precision <Precision>` increases as :math:`\tau` increases, since the model becomes more conservative in predicting the positive class, reducing false positives.

- :ref:`Recall <Recall>` decreases as :math:`\tau` increases, as the model's stricter criteria result in more false negatives.

This tradeoff is especially critical in domains where false positives or false negatives have significantly different costs, such as:

- Medical diagnostics: Emphasize recall to minimize false negatives.
- Spam detection: Emphasize precision to reduce false positives.

Threshold Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adjusting the threshold based solely on a single metric (e.g., maximizing precision) may lead to suboptimal performance in other metrics. For example:

- Increasing :math:`\tau` to improve precision might drastically reduce recall.
- Decreasing :math:`\tau` to maximize recall might result in an unacceptably high false positive rate.

A balanced metric like the F-beta score can address this tradeoff:

.. math::
   F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}}

Here, :math:`\beta` adjusts the weight given to recall relative to precision:

- :math:`\beta > 1`: Recall is prioritized.
- :math:`\beta < 1`: Precision is prioritized.


.. _elastic_net:

ElasticNet Regularization
----------------------------

Elastic net minimizes the following cost function:

.. math::

   \mathcal{L}(\beta) = \frac{1}{2n} \sum_{i=1}^{n} \left( y_i - \mathbf{x}_i^\top \beta \right)^2 + \lambda \left( \alpha \|\beta\|_1 + \frac{1 - \alpha}{2} \|\beta\|_2^2 \right)

where:

- :math:`\|\beta\|_1 = \sum_{j=1}^p |\beta_j|` represents the :math:`L1` norm, promoting sparsity.
- :math:`\|\beta\|_2^2 = \sum_{j=1}^p \beta_j^2` represents the :math:`L2` norm, promoting shrinkage.
- :math:`\lambda` controls the regularization strength.
- :math:`\alpha \in [0, 1]` determines the balance between the :math:`L1` and :math:`L2` penalties.

Important Considerations
^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Balance of Sparsity and Shrinkage**:

   - :math:`\alpha = 1`: Reduces to Lasso (:math:`L1` only).
   - :math:`\alpha = 0`: Reduces to Ridge (:math:`L2` only).
   - Intermediate values allow elastic net to select features while managing multicollinearity.

2. **Regularization Strength**:

   - Larger :math:`\lambda` increases bias but reduces variance, favoring simpler models.
   - Smaller :math:`\lambda` reduces bias but may increase variance, allowing more complex models.

3. **Feature Correlation**:

   - Elastic net handles correlated features better than Lasso, spreading coefficients across groups of related predictors.

4. **Hyperparameter Tuning**:

   - Both :math:`\alpha` and :math:`\lambda` should be optimized via cross-validation to achieve the best performance.

Elastic net is well-suited for datasets with mixed feature relevance, reducing overfitting while retaining important predictors.

.. important::

   - When combining elastic net with RFE, it is important to note that the recursive process may interact with the regularization in elastic net.
   - Elastic net's built-in feature selection can prioritize sparsity, but RFE explicitly removes features step-by-step. This may lead to redundancy in feature selection efforts or alter the balance between :math:`L1` and :math:`L2` penalties as features are eliminated.
   - Careful calibration of :math:`\alpha` and :math:`\lambda` is essential when using RFE alongside elastic net to prevent over-penalization or premature exclusion of relevant features.

.. _CatBoost_Training_Parameters:

CatBoost Training Parameters
-------------------------------

According to the `CatBoost documentation <https://catboost.ai/docs/en/references/training-parameters/>`_:

   "For the Python package several parameters have aliases. For example, the --iterations parameter has the following synonyms: num_boost_round, n_estimators, num_trees. Simultaneous usage of different names of one parameter raises an error."

.. important::

   Attempting to pass more than one of these synonymous hyperparameters will result in the following error:

   ``CatBoostError: only one of the parameters iterations, n_estimators, num_boost_round, num_trees should be initialized.``

   To prevent this issue, ensure you define only **one** of these parameters (e.g., ``n_estimators``) in your configuration, and avoid including any other aliases such as ``iterations`` or ``num_boost_round``.


Example: Tuning hyperparameters for CatBoost
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When defining hyperparameters for grid search, specify only one alias in your configuration. Below is an example:

.. code-block:: python

   cat = CatBoostClassifier(
       n_estimators=100,  ## num estimator/iteration/boostrounds
       learning_rate=0.1,
       depth=6,
       loss_function="Logloss",
   )

   tuned_hyperparameters_cat = {
       f"{cat_name}__n_estimators": [1500],

       ## Additional hyperparameters
       f"{cat_name}__learning_rate": [0.01, 0.1],
       f"{cat_name}__depth": [4, 6, 8],
       f"{cat_name}__loss_function": ["Logloss"],
   }



This ensures compatibility with CatBoost’s requirements and avoids errors during hyperparameter tuning.
