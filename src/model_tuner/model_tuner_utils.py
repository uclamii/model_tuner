import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold, KFold
from pprint import pprint
from sklearn.metrics import get_scorer
from sklearn.metrics import (
    fbeta_score,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
)


from skopt import BayesSearchCV
import copy
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.model_selection import (
    cross_val_predict,
    train_test_split,
    GridSearchCV,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ParameterSampler
from tqdm import tqdm
from sklearn.feature_selection import SelectKBest
from .bootstrapper import evaluate_bootstrap_metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression

"""
# Scores

| Scoring                        | Function                             | Comment                         |
|--------------------------------|--------------------------------------|---------------------------------|
| Classification                 |                                      |                                 |
| ‘accuracy’                     | metrics.accuracy_score               |                                 |
| ‘balanced_accuracy’            | metrics.balanced_accuracy_score      |                                 |
| ‘average_precision’            | metrics.average_precision_score      |                                 |
| ‘neg_brier_score’              | metrics.brier_score_loss             |                                 |
| ‘f1’                           | metrics.f1_score                     | for binary targets              |
| ‘f1_micro’                     | metrics.f1_score                     | micro-averaged                  |
| ‘f1_macro’                     | metrics.f1_score                     | macro-averaged                  |
| ‘f1_weighted’                  | metrics.f1_score                     | weighted average                |
| ‘f1_samples’                   | metrics.f1_score                     | by multilabel sample            |
| ‘neg_log_loss’                 | metrics.log_loss                     | requires predict_proba support  |
| ‘precision’etc.                | metrics.precision_score              | suffixes apply as with ‘f1’     |
| ‘recall’etc.                   | metrics.recall_score                 | suffixes apply as with ‘f1’     |
| ‘jaccard’etc.                  | metrics.jaccard_score                | suffixes apply as with ‘f1’     |
| ‘roc_auc’                      | metrics.roc_auc_score                |                                 |
| ‘roc_auc_ovr’                  | metrics.roc_auc_score                |                                 |
| ‘roc_auc_ovo’                  | metrics.roc_auc_score                |                                 |
| ‘roc_auc_ovr_weighted’         | metrics.roc_auc_score                |                                 |
| ‘roc_auc_ovo_weighted’         | metrics.roc_auc_score                |                                 |
| Clustering                     |                                      |                                 |
| ‘adjusted_mutual_info_score’   | metrics.adjusted_mutual_info_score   |                                 |
| ‘adjusted_rand_score’          | metrics.adjusted_rand_score          |                                 |
| ‘completeness_score’           | metrics.completeness_score           |                                 |
| ‘fowlkes_mallows_score’        | metrics.fowlkes_mallows_score        |                                 |
| ‘homogeneity_score’            | metrics.homogeneity_score            |                                 |
| ‘mutual_info_score’            | metrics.mutual_info_score            |                                 |
| ‘normalized_mutual_info_score’ | metrics.normalized_mutual_info_score |                                 |
| ‘v_measure_score’              | metrics.v_measure_score              |                                 |
| Regression                     |                                      |                                 |
| ‘explained_variance’           | metrics.explained_variance_score     |                                 |
| ‘max_error’                    | metrics.max_error                    |                                 |
| ‘neg_mean_absolute_error’      | metrics.mean_absolute_error          |                                 |
| ‘neg_mean_squared_error’       | metrics.mean_squared_error           |                                 |
| ‘neg_root_mean_squared_error’  | metrics.mean_squared_error           |                                 |
| ‘neg_mean_squared_log_error’   | metrics.mean_squared_log_error       |                                 |
| ‘neg_median_absolute_error’    | metrics.median_absolute_error        |                                 |
| ‘r2’                           | metrics.r2_score                     |                                 |
| ‘neg_mean_poisson_deviance’    | metrics.mean_poisson_deviance        |                                 |
| ‘neg_mean_gamma_deviance’      | metrics.mean_gamma_deviance          |                                 |

"""


class Model:

    def __init__(
        self,
        name,
        estimator_name,
        estimator,
        model_type,
        calibrate=False,
        kfold=False,
        imbalance_sampler=None,
        train_size=0.6,
        validation_size=0.2,
        test_size=0.2,
        stratify_y=False,
        stratify_cols=None,
        grid=None,
        scoring=["roc_auc"],
        n_splits=10,
        random_state=3,
        n_jobs=1,
        display=True,
        randomized_grid=False,
        n_iter=100,
        pipeline_steps=[],
        boost_early=False,
        feature_selection=False,
        class_labels=None,
        multi_label=False,
        calibration_method="sigmoid",
        custom_scorer=[],
        bayesian=False,
        sort_preprocess=True,
        kfold_group=None,
    ):

        # Check if model_type is provided and valid
        if model_type not in ["classification", "regression"]:
            raise ValueError(
                "You must specify model_type as either 'classification' or 'regression'."
            )

        self.name = name
        self.estimator_name = estimator_name
        self.calibrate = calibrate
        self.original_estimator = estimator
        self.feature_selection = feature_selection
        self.model_type = model_type
        self.multi_label = multi_label
        self.calibration_method = calibration_method
        self.imbalance_sampler = imbalance_sampler
        self.sort_preprocess = sort_preprocess

        if imbalance_sampler:
            from imblearn.pipeline import Pipeline

            self.PipelineClass = Pipeline
        else:
            from sklearn.pipeline import Pipeline

            self.PipelineClass = Pipeline

        self.pipeline_steps = pipeline_steps
        if self.pipeline_steps or self.imbalance_sampler:
            self.pipeline_assembly()
        else:
            self.estimator = self.PipelineClass(
                [(self.estimator_name, copy.deepcopy(self.original_estimator))]
            )

        self.grid = grid
        self.class_labels = class_labels
        self.kfold = kfold
        self.dropped_strat_cols = None
        self.randomized_grid = randomized_grid
        self.random_state = random_state
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        if self.randomized_grid and not self.kfold:
            self.grid = list(
                ParameterSampler(
                    self.grid, n_iter=self.n_iter, random_state=self.random_state
                )
            )
        elif self.kfold:
            self.grid = grid
        else:
            self.grid = ParameterGrid(grid)
        self.kf = None
        self.xval_output = None
        self.stratify_y = stratify_y
        self.stratify_cols = stratify_cols
        self.n_splits = n_splits
        self.scoring = scoring
        self.best_params_per_score = {score: 0 for score in self.scoring}
        self.display = display
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size
        self.threshold = {score: 0.5 for score in self.scoring}
        self.beta = 2
        self.labels = ["tn", "fp", "fn", "tp"]
        self.boost_early = boost_early
        self.custom_scorer = custom_scorer
        self.bayesian = bayesian
        self.conf_mat = (
            None  # Initialize conf_mat which is going to be filled in return_metrics
        )

        ### for the moment bayesian only works using cross validation, so
        ### we use the structure that already exists for kfold
        if self.bayesian:
            self.kfold = True

        self.kfold_group = kfold_group

        if self.kfold_group is not None:
            self.stratify_y = False

    """
    Multiple helper methods that are used to fetch different parts of the pipeline.
    These use the naming convention that we enforce in the assemble_pipeline method.
    """

    def get_preprocessing_and_feature_selection_pipeline(self):
        if hasattr(self.estimator, "steps"):
            estimator_steps = self.estimator.steps
        else:
            estimator_steps = self.estimator.estimator.steps
        steps = [
            (name, transformer)
            for name, transformer in estimator_steps
            if name.startswith("preprocess_") or name.startswith("feature_selection_")
        ]
        return self.PipelineClass(steps)

    def get_feature_selection_pipeline(self):
        if hasattr(self.estimator, "steps"):
            estimator_steps = self.estimator.steps
        else:
            estimator_steps = self.estimator.estimator.steps
        steps = [
            (name, transformer)
            for name, transformer in estimator_steps
            if name.startswith("feature_selection_")
        ]
        return self.PipelineClass(steps)

    def get_preprocessing_pipeline(self):
        if hasattr(self.estimator, "steps"):
            estimator_steps = self.estimator.steps
        else:
            estimator_steps = self.estimator.estimator.steps
        # Extract steps names that start with 'preprocess_'
        preprocessing_steps = [
            (name, transformer)
            for name, transformer in estimator_steps
            if name.startswith("preprocess_")
        ]
        return self.PipelineClass(preprocessing_steps)

    def pipeline_assembly(self):
        """
        This method will assemble the pipeline in the correct order. It contains
        helper functions which determine whether the steps are preprocessing, feature
        selection, or imbalance sampler steps.

        These are then used to sort and categorize each step so we ensure the correct
        ordering of the pipeline no matter the input order from users. Users can
        also have unnamed pipeline steps and these will still be ordered in the correct
        format.

        Below we define several helper functions that will be used to type-check parts
        of the pipeline in order to put them in the right sections.
        """

        def is_preprocessing_step(transformer):
            from sklearn.compose import ColumnTransformer

            module = transformer.__class__.__module__
            return (
                module.startswith("sklearn.preprocessing")
                or module.startswith("sklearn.impute")
                or module.startswith("sklearn.decomposition")
                or module.startswith("sklearn.feature_extraction")
                or module.startswith("sklearn.kernel_approximation")
                or module.startswith("category_encoders")
                or isinstance(transformer, ColumnTransformer)
            )

        def is_column_transformer(transformer):
            from sklearn.compose import ColumnTransformer

            return isinstance(transformer, ColumnTransformer)

        def is_imputer(transformer):
            module = transformer.__class__.__module__
            return module.startswith("sklearn.impute")

        def is_scaler(transformer):
            module = transformer.__class__.__module__
            return (
                module.startswith("sklearn.preprocessing")
                and "scal" in transformer.__class__.__name__.lower()
            )

        def is_feature_selection_step(transformer):
            module = transformer.__class__.__module__
            return module.startswith("sklearn.feature_selection")

        def is_imbalance_sampler(transformer):
            from imblearn.base import SamplerMixin

            return isinstance(transformer, SamplerMixin)

        # Initialize lists for different types of steps
        column_transformer_steps = []
        imputation_steps = []
        scaling_steps = []
        other_preprocessing_steps = []
        feature_selection_steps = []
        other_steps = []

        for step in self.pipeline_steps:
            # Unpack the step
            if isinstance(step, tuple):
                name, transformer = step
            else:
                name = None
                transformer = step

            # Categorize the transformer
            if is_column_transformer(transformer):
                self.oh_or_ct = True

                # ColumnTransformer steps
                if not name:
                    name = f"preprocess_column_transformer_step_{len(column_transformer_steps)}"
                else:
                    name = f"preprocess_column_transformer_{name}"
                column_transformer_steps.append((name, transformer))
            elif is_preprocessing_step(transformer):
                if is_imputer(transformer) and self.sort_preprocess:
                    # Imputation steps
                    if not name:
                        name = f"preprocess_imputer_step_{len(imputation_steps)}"
                    else:
                        name = f"preprocess_imputer_{name}"
                    imputation_steps.append((name, transformer))
                elif is_scaler(transformer) and self.sort_preprocess:
                    # Scaling steps
                    if not name:
                        name = f"preprocess_scaler_step_{len(scaling_steps)}"
                    else:
                        name = f"preprocess_scaler_{name}"
                    scaling_steps.append((name, transformer))
                else:
                    # Other preprocessing steps
                    if not name:
                        name = f"preprocess_step_{len(other_preprocessing_steps)}"
                    else:
                        name = f"preprocess_{name}"
                    other_preprocessing_steps.append((name, transformer))
            elif is_feature_selection_step(transformer):
                # Feature selection steps
                if not name:
                    name = f"feature_selection_step_{len(feature_selection_steps)}"
                else:
                    name = f"feature_selection_{name}"
                feature_selection_steps.append((name, transformer))
            elif is_imbalance_sampler(transformer):
                raise ValueError(
                    "Imbalance sampler should be specified via the 'imbalance_sampler' parameter."
                )
            else:
                # Other steps
                if not name:
                    name = f"other_step_{len(other_steps)}"
                else:
                    name = f"other_{name}"
                other_steps.append((name, transformer))

        # Assemble the preprocessing steps in the correct order
        if self.sort_preprocess:
            preprocessing_steps = (
                column_transformer_steps
                + scaling_steps
                + imputation_steps
                + other_preprocessing_steps
            )
        else:
            preprocessing_steps = column_transformer_steps + other_preprocessing_steps

        # Initialize the main pipeline steps list
        main_pipeline_steps = []

        # Add preprocessing steps
        main_pipeline_steps.extend(preprocessing_steps)

        # Add the imbalance sampler and import the appropriate pipeline
        if self.imbalance_sampler:
            main_pipeline_steps.append(("resampler", self.imbalance_sampler))
            from imblearn.pipeline import Pipeline
        else:
            from sklearn.pipeline import Pipeline

        # Add feature selection steps
        main_pipeline_steps.extend(feature_selection_steps)

        # Add any other steps
        main_pipeline_steps.extend(other_steps)

        # Add the estimator
        main_pipeline_steps.append(
            (self.estimator_name, copy.deepcopy(self.original_estimator))
        )

        # Construct the final pipeline
        self.PipelineClass = Pipeline
        self.pipeline_steps = main_pipeline_steps
        self.estimator = self.PipelineClass(self.pipeline_steps)

    def reset_estimator(self):
        """
        Resets the estimator to its original state, reinitializing the pipeline
        with the configured steps or the original estimator.

        This method is useful when modifications (e.g., hyperparameter tuning,
        calibration) have altered the current estimator, and it needs to be
        reverted to its initial configuration for retraining or evaluation.

        Returns:
        --------
        None
        """

        if self.pipeline_steps:
            self.estimator = self.PipelineClass(copy.deepcopy(self.pipeline_steps))
        else:
            self.estimator = self.PipelineClass(
                [(self.estimator_name, copy.deepcopy(self.original_estimator))]
            )
        return

    def verify_imbalance_sampler(self, X_train, y_train):
        """
        Applies imbalance sampling to the training data, optionally including
        preprocessing steps if a pipeline is configured.

        This method is used to handle class imbalance by resampling the training
        data using a resampling technique (e.g., SMOTE, RandomUnderSampler).
        Preprocessing steps, if defined in the pipeline, are applied prior to
        resampling.

        Parameters:
        -----------
        X_train : pandas.DataFrame or array-like
            The training features.
        y_train : pandas.Series or array-like
            The training target labels.

        Returns:
        --------
        None
        """

        ####  Preprocessor, Resampler, rfe, Estimator

        if self.pipeline_steps:
            preproc_test = self.get_preprocessing_pipeline()
        else:
            pass

        resampler_test = clone(self.estimator.named_steps["resampler"])

        ### Only apply the preprocessing when there are valid pipeline steps
        if preproc_test:
            X_train_preproc = preproc_test.fit_transform(X_train)
        else:
            X_train_preproc = X_train

        _, y_res = resampler_test.fit_resample(X_train_preproc, y_train)

        if not isinstance(y_res, pd.DataFrame):
            y_res = pd.DataFrame(y_res)
        print(f"Distribution of y values after resampling: {y_res.value_counts()}")
        print()

    def calibrateModel(self, X, y, score=None):
        """
        Calibrates the model to improve probability estimates, with support for
        k-fold cross-validation and prefit workflows. This method adjusts the
        model's predicted probabilities to align better with observed outcomes,
        using specified calibration techniques (e.g., Platt scaling or isotonic
        regression).

        Parameters:
        -----------
        X : pandas.DataFrame or array-like
            The feature dataset for calibration.
        y : pandas.Series or array-like
            The target dataset for calibration.
        score : str or list of str, optional (default=None)
            The scoring metric(s) to guide calibration. If None, the first scoring
            metric from `self.scoring` is used. For k-fold workflows, calibration
            is performed for each specified score.

        Returns:
        --------
        None
        """

        if self.kfold:
            if score == None:
                if self.calibrate:
                    # reset estimator in case of calibrated model
                    self.reset_estimator()

                    classifier = self.estimator.set_params(
                        **self.best_params_per_score[self.scoring[0]]["params"]
                    )
                    if self.imbalance_sampler:
                        self.verify_imbalance_sampler(X, y)
                    self.estimator = CalibratedClassifierCV(
                        classifier,
                        cv=self.n_splits,
                        method=self.calibration_method,
                    ).fit(X, y)
                    test_model = self.estimator
                    # self.conf_mat_class_kfold(X=X, y=y, test_model=test_model)
                else:
                    pass
            else:
                if self.calibrate:
                    # reset estimator in case of calibrated model
                    self.reset_estimator()
                    classifier = self.estimator.set_params(
                        **self.best_params_per_score[score]["params"]
                    )
                    #  calibrate model, and save output
                    if self.imbalance_sampler:
                        self.verify_imbalance_sampler(X, y)
                    self.estimator = CalibratedClassifierCV(
                        classifier,
                        cv=self.n_splits,
                        method=self.calibration_method,
                    ).fit(X, y)
                    test_model = self.estimator

        else:
            if score == None:
                if self.calibrate:
                    (
                        X_train,
                        X_valid,
                        X_test,
                        y_train,
                        y_valid,
                        y_test,
                    ) = train_val_test_split(
                        X=X,
                        y=y,
                        stratify_y=self.stratify_y,
                        stratify_cols=self.stratify_cols,
                        train_size=self.train_size,
                        validation_size=self.validation_size,
                        test_size=self.test_size,
                        random_state=self.random_state,
                    )
                    if isinstance(X, pd.DataFrame):
                        self.X_train_index = X_train.index.to_list()
                        self.X_valid_index = X_valid.index.to_list()
                        self.X_test_index = X_test.index.to_list()
                        self.y_train_index = y_train.index.to_list()
                        self.y_valid_index = y_valid.index.to_list()
                        self.y_test_index = y_test.index.to_list()

                    # reset estimator in case of calibrated model
                    self.reset_estimator()
                    # fit estimator

                    if self.imbalance_sampler:
                        self.verify_imbalance_sampler(X_train, y_train)

                    if self.boost_early:
                        self.fit(X_train, y_train, validation_data=[X_valid, y_valid])
                    else:
                        self.fit(X_train, y_train)
                    #  calibrate model, and save output
                    self.estimator = CalibratedClassifierCV(
                        self.estimator,
                        cv="prefit",
                        method=self.calibration_method,
                    ).fit(X_valid, y_valid)
                else:
                    pass
            else:
                if self.calibrate:
                    (
                        X_train,
                        X_valid,
                        X_test,
                        y_train,
                        y_valid,
                        y_test,
                    ) = train_val_test_split(
                        X=X,
                        y=y,
                        stratify_y=self.stratify_y,
                        stratify_cols=self.stratify_cols,
                        train_size=self.train_size,
                        validation_size=self.validation_size,
                        test_size=self.test_size,
                        random_state=self.random_state,
                    )
                    if isinstance(X, pd.DataFrame):
                        self.X_train_index = X_train.index.to_list()
                        self.X_valid_index = X_valid.index.to_list()
                        self.X_test_index = X_test.index.to_list()
                        self.y_train_index = y_train.index.to_list()
                        self.y_valid_index = y_valid.index.to_list()
                        self.y_test_index = y_test.index.to_list()

                    # reset estimator in case of calibrated model
                    self.reset_estimator()
                    # print(self.estimator[-1].get_params())
                    if "device" in self.estimator[-1].get_params():
                        print("Change back to CPU")
                        self.estimator[-1].set_params(**{"device": "cpu"})

                    # fit estimator
                    if self.imbalance_sampler:
                        self.verify_imbalance_sampler(X_train, y_train)

                    # fit model
                    self.fit(
                        X_train,
                        y_train,
                        score=score,
                        validation_data=(X_valid, y_valid),
                    )
                    #  calibrate model, and save output

                    self.estimator = CalibratedClassifierCV(
                        self.estimator,
                        cv="prefit",
                        method=self.calibration_method,
                    ).fit(X_valid, y_valid)
                    print(
                        f"{score} after calibration:",
                        get_scorer(score)(self.estimator, X_valid, y_valid),
                    )

                else:
                    pass

        return

    ############################################################################
    ##  These functions return subsets of the dataset (features and labels)   ##
    ##  based on predefined indices stored in the class attributes            ##
    ############################################################################
    def get_train_data(self, X, y):
        return X.loc[self.X_train_index], y.loc[self.y_train_index]

    def get_valid_data(self, X, y):
        return X.loc[self.X_valid_index], y.loc[self.y_valid_index]

    def get_test_data(self, X, y):
        return X.loc[self.X_test_index], y.loc[self.y_test_index]

    def fit(self, X, y, validation_data=None, score=None):
        """
        Trains the model using the best hyperparameters obtained from tuning
        and optionally supports k-fold cross-validation and early stopping.

        Parameters:
        -----------
        X : pandas.DataFrame or array-like
            Training feature dataset.
        y : pandas.Series or array-like
            Training target dataset.
        validation_data : tuple, optional (default=None)
            A tuple containing validation feature and target datasets
            (X_valid, y_valid). Required for early stopping.
        score : str, optional (default=None)
            The scoring metric to optimize during training. If None, the default
            scoring metric from the class is used.

        Returns:
        --------
        None
        """

        self.reset_estimator()
        if self.kfold:
            if score == None:
                classifier = self.estimator.set_params(
                    **self.best_params_per_score[self.scoring[0]]["params"]
                )
                classifier.fit(X, y)
                self.estimator = classifier
                self.xval_output = get_cross_validate(
                    classifier,
                    X,
                    y,
                    self.kf,
                    scoring=self.scoring[0],
                    kfgroups=self.kfold_group,
                )
                # self.estimator = self.x_valoutput['estimator']
            else:
                if score in self.custom_scorer:
                    scorer = self.custom_scorer[score]
                else:
                    scorer = score
                classifier = self.estimator.set_params(
                    **self.best_params_per_score[score]["params"]
                )
                classifier.fit(X, y)
                self.estimator = classifier
                self.xval_output = get_cross_validate(
                    classifier, X, y, self.kf, scoring=scorer, kfgroups=self.kfold_group
                )
        else:
            if score is None:
                best_params = self.best_params_per_score[self.scoring[0]]["params"]

                if self.boost_early:
                    X_valid, y_valid = validation_data
                    if self.feature_selection or self.pipeline_steps:
                        # Extract parameters for preprocessing and feature selection
                        params_preprocessing = {
                            key: value
                            for key, value in best_params.items()
                            if key.startswith("preprocess_")
                        }
                        params_feature_selection = {
                            key: value
                            for key, value in best_params.items()
                            if key.startswith("feature_selection_")
                        }
                        params_no_estimator = {
                            **params_preprocessing,
                            **params_feature_selection,
                        }

                        # Exclude the resampler if present
                        if self.imbalance_sampler:
                            params_no_estimator = {
                                key: value
                                for key, value in params_no_estimator.items()
                                if not key.startswith("resampler__")
                            }

                        # Get the combined preprocessing and feature selection pipeline
                        preproc_feat_select_pipe = (
                            self.get_preprocessing_and_feature_selection_pipeline()
                        )

                        ### IF we have preprocessing steps then they need applying
                        ### Otherwise do not apply them
                        if preproc_feat_select_pipe:
                            # Set parameters and fit the pipeline
                            preproc_feat_select_pipe.set_params(
                                **params_no_estimator
                            ).fit(X, y)

                            # Transform the validation data
                            X_valid_transformed = preproc_feat_select_pipe.transform(
                                X_valid
                            )
                        else:
                            X_valid_transformed = X_valid
                    else:
                        X_valid_transformed = X_valid

                    X_valid, y_valid = validation_data
                    if isinstance(X_valid, pd.DataFrame):
                        eval_set = [(X_valid_transformed, y_valid.values)]
                    else:
                        eval_set = [(X_valid_transformed, y_valid)]
                    estimator_eval_set = f"{self.estimator_name}__eval_set"
                    estimator_verbosity = f"{self.estimator_name}__verbose"

                    xgb_params = {
                        estimator_eval_set: eval_set,
                        estimator_verbosity: self.verbosity,
                    }
                    if estimator_verbosity in best_params:
                        best_params.pop(estimator_verbosity)
                    self.estimator.set_params(**best_params).fit(X, y, **xgb_params)
                else:
                    self.estimator.set_params(**best_params).fit(X, y)
            else:
                best_params = self.best_params_per_score[score]["params"]
                if self.boost_early:
                    X_valid, y_valid = validation_data
                    if self.feature_selection or self.pipeline_steps:
                        # Extract parameters for preprocessing and feature selection
                        params_preprocessing = {
                            key: value
                            for key, value in best_params.items()
                            if key.startswith("preprocess_")
                        }
                        params_feature_selection = {
                            key: value
                            for key, value in best_params.items()
                            if key.startswith("feature_selection_")
                        }
                        params_no_estimator = {
                            **params_preprocessing,
                            **params_feature_selection,
                        }

                        # Exclude the resampler if present
                        if self.imbalance_sampler:
                            params_no_estimator = {
                                key: value
                                for key, value in params_no_estimator.items()
                                if not key.startswith("resampler__")
                            }

                        # Get the combined preprocessing and feature selection pipeline
                        preproc_feat_select_pipe = (
                            self.get_preprocessing_and_feature_selection_pipeline()
                        )

                        ### IF we have preprocessing steps then they need applying
                        ### Otherwise do not apply them
                        if preproc_feat_select_pipe:
                            # Set parameters and fit the pipeline
                            preproc_feat_select_pipe.set_params(
                                **params_no_estimator
                            ).fit(X, y)

                            # Transform the validation data
                            X_valid_transformed = preproc_feat_select_pipe.transform(
                                X_valid
                            )
                        else:
                            X_valid_transformed = X_valid
                    else:
                        X_valid_transformed = X_valid

                    X_valid, y_valid = validation_data
                    if isinstance(X_valid, pd.DataFrame):
                        eval_set = [(X_valid_transformed, y_valid.values)]
                    else:
                        eval_set = [(X_valid_transformed, y_valid)]
                    estimator_eval_set = f"{self.estimator_name}__eval_set"
                    estimator_verbosity = f"{self.estimator_name}__verbose"

                    xgb_params = {
                        estimator_eval_set: eval_set,
                        estimator_verbosity: self.verbosity,
                    }
                    if (
                        estimator_verbosity
                        in self.best_params_per_score[score]["params"]
                    ):
                        self.best_params_per_score[score]["params"].pop(
                            estimator_verbosity
                        )

                    self.estimator.set_params(
                        **self.best_params_per_score[score]["params"]
                    ).fit(X, y, **xgb_params)
                else:
                    try:
                        self.estimator.set_params(
                            **self.best_params_per_score[score]["params"]
                        ).fit(X, y)
                    except ValueError as error:
                        print(
                            "Specified score not found in scoring dictionary. "
                            "Please use a score that was parsed for tuning."
                        )
                        raise error

        return

    def return_bootstrap_metrics(
        self,
        X_test,
        y_test,
        metrics,
        threshold=0.5,
        num_resamples=500,
        n_samples=500,
        balance=False,
    ):
        """
        Evaluates bootstrap metrics for a trained model using
        the test dataset. This function supports both
        classification and regression tasks by leveraging
        evaluate_bootstrap_metrics to compute confidence
        intervals for the specified metrics.

        Parameters:
        -----------
        X_test : pandas.DataFrame
            Test dataset features.
        y_test : pandas.Series or pandas.DataFrame
            Test dataset labels.
        metrics : list of str
            List of metric names to calculate (e.g., "roc_auc",
            "f1_weighted").
        threshold : float, optional
            Threshold for converting predicted probabilities
            into class predictions. Default is 0.5.
        num_resamples : int, optional
            Number of bootstrap iterations. Default is 500.
        n_samples : int, optional
            Number of samples per bootstrap iteration.
            Default is 500.
        balance : bool, optional
            Whether to balance the class distribution during
            resampling. Default is False.

        Returns:
        --------
        pandas.DataFrame
            DataFrame containing mean and confidence intervals
            for the specified metrics.

        Raises:
        -------
        ValueError
            If X_test or y_test are not provided as Pandas
            DataFrames or if unsupported input types are
            specified.
        """

        # Custom type check for X_test and y_test
        if not isinstance(X_test, pd.DataFrame) or not isinstance(
            y_test, (pd.Series, pd.DataFrame)
        ):
            raise ValueError(
                "Specifying X_test and/or y_test as anything other "
                "than pandas DataFrames is not supported."
            )

        if self.model_type != "regression":
            y_pred_prob = pd.Series(self.predict_proba(X_test)[:, 1])
            bootstrap_metrics = evaluate_bootstrap_metrics(
                model=None,
                y=y_test,
                y_pred_prob=y_pred_prob,
                metrics=metrics,
                threshold=threshold,
                num_resamples=num_resamples,
                n_samples=n_samples,
                balance=balance,
            )
        else:
            y_pred = pd.Series(self.predict(X_test))
            bootstrap_metrics = evaluate_bootstrap_metrics(
                model=None,
                y=y_test,
                y_pred_prob=y_pred,
                model_type="regression",
                metrics=metrics,
                num_resamples=num_resamples,
                n_samples=n_samples,
                balance=balance,
            )
        return bootstrap_metrics

    def return_metrics(
        self,
        X,
        y,
        optimal_threshold=False,
        model_metrics=False,
        print_threshold=False,
        return_dict=False,
        print_per_fold=False,
    ):
        """
        Evaluate the model on the given dataset, optionally using cross-validation,
        and provide metrics or insights into the model's performance.

        Parameters:
        -----------
        X : array-like or DataFrame
            The feature matrix for evaluation.
        y : array-like
            The target vector for evaluation.
        optimal_threshold : bool, optional (default=False)
            Whether to use the optimal threshold for predictions
            (for classification models). If False, a default threshold of 0.5 is used.
        model_metrics : bool, optional (default=False)
            Whether to calculate and print additional model metrics, such as
            precision, recall, and F1-score.
        print_threshold : bool, optional (default=False)
            Whether to print the threshold used for predictions.
        return_dict : bool, optional (default=False)
            Whether to return the calculated metrics as a dictionary.
        print_per_fold : bool, optional (default=False)
            If using cross-validation, whether to print metrics for each fold.

        Returns:
        --------
        dict, optional
            Returns a dictionary containing performance metrics if
            `return_dict=True`. The dictionary structure depends on the model
            type(classification or regression):
                - For classification:
                    {
                        "Classification Report": str,
                        "Confusion Matrix": array,
                        "Best Features": list (if feature selection is enabled),
                    }
                - For regression:
                    {
                        "Regression Report": dict,
                        "Best Features": list (if feature selection is enabled),
                    }
        None
            If `return_dict=False`, the function prints metrics and does not
            return anything.

        Notes:
        ------
        - For classification models, confusion matrices and classification
        reports are printed and optionally included in the returned dictionary.
        - For regression models, regression metrics are printed and optionally
        returned.
        - If cross-validation (`self.kfold`) is enabled, metrics are calculated
        and printed for each fold, and an aggregated summary is provided.
        - If feature selection is enabled (`self.feature_selection`), the top
        selected features are printed and optionally included in the output
        dictionary.

        Examples:
        ---------
        # Example usage for classification:
        metrics = model.return_metrics(
            X_test,
            y_test,
            optimal_threshold=True,
            return_dict=True,
        )

        # Example usage for regression:
        metrics = model.return_metrics(
            X_test,
            y_test,
            model_metrics=True,
            return_dict=True,
        )
        """

        if self.kfold:
            for score in self.scoring:
                if self.model_type != "regression":
                    print(
                        "\n"
                        + "Detailed classification report for %s:" % self.name
                        + "\n"
                    )
                    self.conf_mat_class_kfold(X, y, self, score, optimal_threshold)

                    print("The model is trained on the full development set.")
                    print("The scores are computed on the full evaluation set." + "\n")
                    self.return_metrics_kfold(
                        X,
                        y,
                        self,
                        score,
                        print_per_fold,
                        optimal_threshold,
                    )

                    if optimal_threshold:
                        threshold = self.threshold[self.scoring[0]]
                    else:
                        threshold = 0.5
                    if print_threshold:
                        print(f"Optimal threshold used: {threshold}")
                    if model_metrics:
                        metrics = report_model_metrics(
                            self, X, y, threshold, True, print_per_fold
                        )
                    print("-" * 80)
                    print()
                    if return_dict and model_metrics:
                        return metrics.set_index("Metric")["Value"].to_dict()
                else:
                    self.regression_report_kfold(X, y, self.test_model, score)

                if self.feature_selection:
                    print(self.get_feature_names())
        else:
            y_pred = self.predict(X, optimal_threshold=optimal_threshold)
            if self.model_type != "regression":

                if self.multi_label:
                    conf_mat = multilabel_confusion_matrix(y, y_pred)
                    self.conf_mat = conf_mat  # store it so we can ext. dict
                    self._confusion_matrix_print_ML(conf_mat)

                    print(
                        classification_report(
                            y,
                            y_pred,
                            target_names=self.class_labels,
                        )
                    )
                    print("-" * 80)

                    if optimal_threshold:
                        threshold = self.threshold[self.scoring[0]]
                    else:
                        threshold = 0.5
                    if print_threshold:
                        print(f"Optimal threshold used: {threshold}")
                    if model_metrics:
                        metrics = report_model_metrics(
                            self, X, y, threshold, True, print_per_fold
                        )
                        return metrics.set_index(["Class", "Metric"])["Value"].to_dict()

                    if return_dict:
                        self.classification_report = classification_report(
                            y, y_pred, output_dict=True
                        )
                        return {
                            "Classification Report": self.classification_report,
                            "Confusion Matrix": conf_mat,
                        }

                else:
                    conf_mat = confusion_matrix(y, y_pred)
                    self.conf_mat = conf_mat  # store it so we can ext. dict
                    print("Confusion matrix on set provided: ")
                    _confusion_matrix_print(conf_mat, self.labels)
                    if optimal_threshold:
                        threshold = self.threshold[self.scoring[0]]
                    else:
                        threshold = 0.5
                    if print_threshold:
                        print(f"Optimal threshold used: {threshold}")

                    if model_metrics:
                        metrics = report_model_metrics(
                            self, X, y, threshold, True, print_per_fold
                        )
                        return metrics.set_index("Metric")["Value"].to_dict()
                    print("-" * 80)
                print()
                self.classification_report = classification_report(
                    y, y_pred, output_dict=True
                )
                print(
                    classification_report(
                        y,
                        y_pred,
                        target_names=self.class_labels,
                    )
                )
                print("-" * 80)

                if self.feature_selection:
                    best_features = self.get_feature_names()
                    print(best_features)

                    if return_dict:
                        return {
                            "Classification Report": self.classification_report,
                            "Confusion Matrix": conf_mat,
                            "Best Features": best_features,
                        }
                else:
                    if return_dict:
                        return {
                            "Classification Report": self.classification_report,
                            "Confusion Matrix": conf_mat,
                        }
            else:
                reg_report = self.regression_report(y, y_pred)
                if self.feature_selection:
                    best_features = self.get_feature_names()
                    print(best_features)
                    if return_dict:
                        return {
                            "Regression Report": reg_report,
                            "Best Features": best_features,
                        }
                else:
                    if return_dict:
                        return reg_report

    def predict(self, X, y=None, optimal_threshold=False, score=None):
        """
        Makes predictions and predicts probabilities, allowing
        threshold tuning.

        Parameters:
        -----------
        X : The feature matrix for prediction.
        y : The true target labels, required only for k-fold predictions.
            Default is None.
        optimal_threshold : Whether to use an optimal classification threshold
                            for predictions. Default is False.

        Returns:
        --------
        Predicted class labels or predictions adjusted by the optimal threshold.

        Raises:
        -------
        ValueError
            Raised if invalid inputs or configurations are provided.
        """

        # Select the threshold if optimal_threshold is True
        if optimal_threshold:
            if score is not None:
                threshold = self.threshold[score]
            else:
                threshold = self.threshold[self.scoring[0]]

        if self.model_type == "regression":
            optimal_threshold = False
        if self.kfold and y is not None:
            return cross_val_predict(estimator=self.estimator, X=X, y=y, cv=self.kf)
        else:
            if optimal_threshold:
                return (self.predict_proba(X)[:, 1] > threshold) * 1
            else:
                return self.estimator.predict(X)

    def predict_proba(self, X, y=None):
        """
        Predicts class probabilities for the input data.

        Parameters:
        -----------
        X : pandas.DataFrame or array-like. The feature matrix for prediction.
        y : pandas.Series or array-like, optional. The true target labels,
            required only for k-fold predictions. Default is None.

        Returns:
        --------
        numpy.ndarray or array-like. Predicted probabilities for each class.

        Notes:
        ------
        - If `self.kfold` is True and `y` is provided, cross-validated
        probabilities are computed using the specified estimator.
        - If k-fold validation is not enabled, probabilities are predicted
        using the fitted estimator.
        """

        if self.kfold and y is not None:
            return cross_val_predict(
                self.estimator, X, y, cv=self.kf, method="predict_proba"
            )
        else:
            return self.estimator.predict_proba(X)

    def grid_search_param_tuning(
        self,
        X,
        y,
        f1_beta_tune=False,
        betas=[1, 2],
    ):
        """
        Performs grid or Bayesian search parameter tuning, optionally
        tuning F-beta score thresholds for classification tasks.

        Parameters:
        -----------
        X : The feature matrix for training and validation.
        y : The target vector corresponding to `X`.
        f1_beta_tune : Whether to tune F-beta score thresholds during parameter
                       tuning. Default is False.
        betas : List of beta values to use for F-beta score tuning.Default is [1, 2].

        Raises:
        -------
        ValueError
            Raised if invalid data or configurations are provided.

        KeyError
            Raised if required scoring metrics are missing.

        Description:
        ------------
        - Tunes hyperparameters for a model using grid search or Bayesian
        optimization.
        - Supports tuning F-beta thresholds for classification tasks.
        - Can handle both k-fold cross-validation and train-validation-test
        workflows.

        Output:
        -------
        - Updates `self.best_params_per_score` with the best parameters and
        scores for each metric.
        - Optionally updates `self.threshold` with tuned F-beta thresholds.
        - Prints the best parameters and scores if `self.display` is enabled.
        """

        if self.display:
            print_pipeline(self.estimator)

        if self.kfold:
            self.kf = kfold_split(
                self.estimator,
                X,
                y,
                stratify=self.stratify_y,
                kfold_group=self.kfold_group,
                scoring=self.scoring,
                n_splits=self.n_splits,
                random_state=self.random_state,
            )

            self.get_best_score_params(X, y)
            #### Threshold tuning for kfold split for each score
            if f1_beta_tune:  # tune threshold
                if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
                    for score in self.scoring:
                        thresh_list = []
                        self.kfold = False
                        for train, test in self.kf.split(X, y, groups=self.kfold_group):

                            self.fit(X.iloc[train], y.iloc[train])
                            y_pred_proba = self.predict_proba(X.iloc[test])[:, 1]
                            thresh = self.tune_threshold_Fbeta(
                                score,
                                y.iloc[test],
                                betas,
                                y_pred_proba,
                                kfold=True,
                            )
                            thresh_list.append(thresh)
                        average_threshold = np.mean(thresh_list)
                        self.threshold[score] = average_threshold
                        self.kfold = True

                else:
                    for score in self.scoring:
                        thresh_list = []
                        self.kfold = False
                        for train, test in self.kf.split(X, y, groups=self.kfold_group):

                            self.fit(X[train], y[train])
                            y_pred_proba = self.predict_proba(X[test])[:, 1]
                            thresh = self.tune_threshold_Fbeta(
                                score,
                                y[test],
                                betas,
                                y_pred_proba,
                                kfold=True,
                            )
                            thresh_list.append(thresh)
                        self.kfold = True
                        average_threshold = np.mean(thresh_list)
                        self.threshold[score] = average_threshold
        else:
            X_train, X_valid, X_test, y_train, y_valid, y_test = train_val_test_split(
                X=X,
                y=y,
                stratify_y=self.stratify_y,
                stratify_cols=self.stratify_cols,
                train_size=self.train_size,
                validation_size=self.validation_size,
                test_size=self.test_size,
                random_state=self.random_state,
            )
            if isinstance(X, pd.DataFrame):
                self.X_train_index = X_train.index.to_list()
                self.X_valid_index = X_valid.index.to_list()
                self.X_test_index = X_test.index.to_list()
                self.y_train_index = y_train.index.to_list()
                self.y_valid_index = y_valid.index.to_list()
                self.y_test_index = y_test.index.to_list()

            if self.imbalance_sampler:
                self.verify_imbalance_sampler(X_train, y_train)

            ## casting the ParameterGrid Object to a list so that we can update
            ## update the hyperparameters in both random grid and non random grid
            ## scenarios
            if not self.randomized_grid:
                self.grid = list(self.grid)

            for score in self.scoring:
                scores = []
                for index, params in enumerate(tqdm(self.grid)):
                    ### Resetting the estimator here because catboost requires
                    ### a new model to be fitted each time.
                    self.reset_estimator()
                    if self.boost_early:

                        if self.feature_selection or self.pipeline_steps:
                            # Extract parameters for preprocessing and feature selection
                            params_preprocessing = {
                                key: value
                                for key, value in params.items()
                                if key.startswith("preprocess_")
                            }
                            params_feature_selection = {
                                key: value
                                for key, value in params.items()
                                if key.startswith("feature_selection_")
                            }
                            params_no_estimator = {
                                **params_preprocessing,
                                **params_feature_selection,
                            }

                            if self.imbalance_sampler:
                                params_no_estimator = {
                                    key: value
                                    for key, value in params_no_estimator.items()
                                    if not key.startswith("resampler__")
                                }

                            preproc_feat_select_pipe = (
                                self.get_preprocessing_and_feature_selection_pipeline()
                            )

                            ### IF we have preprocessing steps then they need applying
                            ### Otherwise do not apply them
                            if preproc_feat_select_pipe:
                                preproc_feat_select_pipe.set_params(
                                    **params_no_estimator
                                ).fit(X_train, y_train)

                                X_valid_transformed = (
                                    preproc_feat_select_pipe.transform(X_valid)
                                )
                            else:
                                X_valid_transformed = X_valid
                        else:
                            X_valid_transformed = X_valid

                        if isinstance(X_valid, pd.DataFrame):
                            eval_set = [(X_valid_transformed, y_valid.values)]
                        else:
                            eval_set = [(X_valid_transformed, y_valid)]

                        estimator_eval_set = f"{self.estimator_name}__eval_set"
                        estimator_verbosity = f"{self.estimator_name}__verbose"

                        if estimator_verbosity in params:
                            self.verbosity = params[estimator_verbosity]
                            params.pop(estimator_verbosity)
                        else:
                            self.verbosity = False

                        xgb_params = {
                            estimator_eval_set: eval_set,
                            estimator_verbosity: self.verbosity,
                        }

                        clf = self.estimator.set_params(**params).fit(
                            X_train, y_train, **xgb_params
                        )

                        ### extracting the best parameters found through early stopping
                        best_early_stopping_params = clf.named_steps[
                            self.estimator_name
                        ].get_params()

                        ### updating the params in the param grid with these
                        ### updated parameters
                        for (
                            param_name,
                            param_value,
                        ) in best_early_stopping_params.items():
                            if param_name in params:
                                params[param_name] = param_value

                        ### extracting the number of estimators out of the
                        ### fitted model, this is stored in best_iteration
                        ### setting the current item in the grid to this number
                        try:
                            ### XGBoost case
                            params[f"{self.estimator_name}__n_estimators"] = (
                                clf[len(clf) - 1].best_iteration + 1
                            )
                        except:
                            ### catboost case
                            params[f"{self.estimator_name}__n_estimators"] = (
                                clf[len(clf) - 1].best_iteration_ + 1
                            )

                        # Update the parameters in the grid
                        self.grid[index] = params

                    else:
                        clf = self.estimator.set_params(**params).fit(X_train, y_train)

                    if score in self.custom_scorer:
                        scorer_func = self.custom_scorer[score]
                    else:
                        scorer_func = get_scorer(score)

                    score_value = scorer_func(clf, X_valid, y_valid)
                    scores.append(score_value)

                self.best_params_per_score[score] = {
                    "params": self.grid[np.argmax(scores)],
                    "score": np.max(scores),
                }

                if f1_beta_tune:  # tune threshold
                    y_pred_proba = clf.predict_proba(X_valid)[:, 1]
                    self.tune_threshold_Fbeta(score, y_valid, betas, y_pred_proba)

                if not self.calibrate:
                    if self.display:
                        print("Best score/param set found on validation set:")
                        pprint(self.best_params_per_score[score])
                        print("Best " + score + ": %0.3f" % (np.max(scores)), "\n")

                else:
                    if self.display:
                        print("Best score/param set found on validation set:")
                        pprint(self.best_params_per_score[score])
                        print("Best " + score + ": %0.3f" % (np.max(scores)), "\n")

    def get_feature_names(self):
        """
        Returns feature names post processing. Uses self.estimator[:-1] to access pipe.

        Returns:
        --------
            A list of the features.
        """
        if self.pipeline_steps is None or not self.pipeline_steps:
            raise ValueError("You must provide pipeline steps to use get_feature_names")
        if hasattr(self.estimator, "steps"):
            estimator_steps = self.estimator[:-1]
        else:
            estimator_steps = self.estimator.estimator[:-1]

        return estimator_steps.get_feature_names_out().tolist()

    def tune_threshold_Fbeta(
        self,
        score,
        y_valid,
        betas,
        y_valid_proba,
        kfold=False,
    ):
        """
        Tune the classification threshold for a model based on the F-beta score.

        This method finds the optimal threshold for classifying validation data,
        aiming to maximize the F-beta score for a given set of beta values. The
        F-beta score balances precision and recall, with the beta parameter
        determining the weight of recall in the score. A range of thresholds
        (from 0 to 1) is evaluated, and the best performing threshold for each
        beta value is identified.

        Parameters
        ----------
        score : str
            A label or name for the score that will be used to store the best
            threshold.

        y_valid : array-like of shape (n_samples,)
            Ground truth (actual) labels for the validation dataset.

        betas : list of float
            A list of beta values to consider when calculating the F-beta score.
            The beta parameter controls the balance between precision and recall,
            where higher beta values give more weight to recall.

        y_valid_proba : array-like of shape (n_samples,)
            Predicted probabilities for the positive class for the validation
            dataset. This is used to apply different thresholds and generate
            binary predictions.

        kfold : bool, optional, default=False
            If True, the method will return the optimal threshold based on
            k-fold cross-validation rather than updating the class's `threshold`
            attribute. Otherwise, the method updates the `threshold` attribute
            for the specified score.

        Returns
        -------
        float or None
            If `kfold` is True, the method returns the best threshold for the
            given score. If `kfold` is False, it updates the `threshold`
            attribute in place and returns None.

        Notes
        -----
        - The method iterates over a range of thresholds (0 to 1, with step
          size of 0.01) and evaluates each threshold by calculating binary
          predictions and computing the confusion matrix.
        - To avoid undesirable results (e.g., excessive false positives),
          thresholds leading to cases where false positives exceed true
          negatives are penalized.
        - The method selects the beta value that produces the maximum F-beta
          score, and for that beta, it identifies the best threshold.
        """

        print("Fitting model with best params and tuning for best threshold ...")
        # predictions
        y_valid_probs = y_valid_proba

        # threshold range
        thresholds_range = np.arange(0, 1, 0.01)

        fbeta_scores = np.zeros((len(betas), len(thresholds_range)))

        for i in tqdm(range(fbeta_scores.shape[0])):
            for j in range(fbeta_scores.shape[1]):
                y_preds = (y_valid_probs > thresholds_range[j]) * 1
                # conf matrix
                conf_matrix = confusion_matrix(
                    y_true=y_valid.astype(int), y_pred=y_preds
                )
                tn, fp, fn, tp = conf_matrix.ravel()

                # avoid extreme cases
                if tn <= fp:
                    fbeta_scores[i, j] = -1  # not desirable results
                else:
                    fbeta_scores[i, j] = fbeta_score(y_valid, y_preds, beta=betas[i])

        ind_beta_with_max_fscore = np.argmax(np.max(fbeta_scores, axis=1))
        self.beta = betas[ind_beta_with_max_fscore]

        if kfold:
            return_score = thresholds_range[
                np.argmax(fbeta_scores[ind_beta_with_max_fscore])
            ]
            return return_score
        else:
            self.threshold[score] = thresholds_range[
                np.argmax(fbeta_scores[ind_beta_with_max_fscore])
            ]
            return

    def get_best_score_params(self, X, y):
        """
        Tunes hyperparameters for the model based on specified scoring metrics
        and updates the best parameters and scores for each metric.

        Parameters:
        -----------
        X : pandas.DataFrame or array-like
            The feature matrix used for hyperparameter tuning.

        y : pandas.Series or array-like
            The target vector corresponding to `X`.

        Raises:
        -------
        ValueError
            Raised if the grid or scoring metrics are invalid.

        Description:
        ------------
        - This method performs hyperparameter tuning using grid search,
        randomized search, or Bayesian search based on the class
        configuration.
        - Supports multiple scoring metrics and stores the best
        parameters and scores for each.
        """

        for score in self.scoring:

            print("# Tuning hyper-parameters for %s" % score)

            if score in self.custom_scorer:
                scorer = self.custom_scorer[score]
            else:
                scorer = score

            if self.randomized_grid:
                clf = RandomizedSearchCV(
                    self.estimator,
                    self.grid,
                    scoring=scorer,
                    cv=self.kf,
                    random_state=self.random_state,
                    n_iter=self.n_iter,
                    n_jobs=self.n_jobs,
                    verbose=2,
                )

            elif self.bayesian:
                ### removing any bayes_params
                bayes_params = {
                    key.replace("bayes__", ""): value
                    for key, value in self.grid.items()
                    if key.startswith("bayes__")
                }
                for key in bayes_params.keys():
                    self.grid.pop(f"bayes__{key}")

                print("Performing Bayesian search:")
                clf = BayesSearchCV(
                    estimator=self.estimator,
                    search_spaces=self.grid,
                    cv=self.kf,
                    n_jobs=self.n_jobs,
                    scoring=scorer,
                    random_state=self.random_state,
                    verbose=2,
                    **bayes_params,
                )

            else:
                clf = GridSearchCV(
                    self.estimator,
                    self.grid,
                    scoring=scorer,
                    cv=self.kf,
                    n_jobs=self.n_jobs,
                    verbose=2,
                )
            if self.kfold_group is not None:
                clf.fit(X, y, groups=self.kfold_group)
            else:
                clf.fit(X, y)
            self.estimator = clf.best_estimator_
            self.test_model = clf.best_estimator_

            if self.display:
                ## Make classification report and conf matrix into function
                print("\n" + "Best score/param set found on development set:")
                pprint({clf.best_score_: clf.best_params_})
                print("\n" + "Grid scores on development set:")
                means = clf.cv_results_["mean_test_score"]
                stds = clf.cv_results_["std_test_score"]
                for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
                    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

            self.best_params_per_score[score] = {
                "params": clf.best_params_,
                "score": clf.best_score_,
            }
            # self.estimator = clf.best_estimator_

    def return_metrics_kfold(
        self,
        X,
        y,
        test_model,
        score=None,
        print_per_fold=False,
        optimal_threshold=False,
    ):
        """
        Evaluate the model's performance using K-Fold cross-validation and print
        metrics for each fold. Aggregates predictions and true labels across
        folds for overall evaluation.

        Parameters:
        -----------
        X : array-like or DataFrame
            The feature matrix to be split into K folds for cross-validation.
        y : array-like
            The target vector corresponding to the feature matrix.
        test_model : estimator object
            The model to be trained and evaluated. It must implement `fit` and
            `predict` methods.
        score : str, optional (default=None)
            The scoring metric to determine the optimal threshold. If None, the
            first metric in `self.scoring` is used.
        print_per_fold : bool, optional (default=False)
            Whether to print the confusion matrix and classification report for
            each fold.

        Returns:
        --------
        None
            This function does not return any value. It prints metrics for each
            fold if `print_per_fold=True`.

        Notes:
        ------
        - For each fold, the model is trained on the training subset and
        evaluated on the test subset.
        - Confusion matrices and classification reports are generated for each
        fold if `print_per_fold=True`.
        - The function aggregates true labels and predictions across folds for
        overall evaluation (though aggregated results are not explicitly returned).
        - Ensures `test_model` has the necessary attributes (`model_type` and
        `estimator_name`) for compatibility with downstream processing.
        """

        # Ensure test_model has necessary attributes
        if not hasattr(test_model, "model_type"):
            test_model.model_type = self.model_type
        if not hasattr(test_model, "estimator_name"):
            test_model.estimator_name = self.estimator_name

        aggregated_true_labels = []
        aggregated_predictions = []

        if isinstance(X, pd.DataFrame):
            for fold_idx, (train, test) in enumerate(
                self.kf.split(X, y, groups=self.kfold_group), start=1
            ):
                X_train, X_test = X.iloc[train], X.iloc[test]
                y_train, y_test = y.iloc[train], y.iloc[test]
                test_model.fit(X_train, y_train)
                # y_pred = test_model.predict(X_test)
                y_pred = test_model.predict(
                    X_test, optimal_threshold=optimal_threshold, score=score
                )
                conf_matrix = confusion_matrix(y_test, y_pred)
                # Aggregate true labels and predictions
                aggregated_true_labels.extend(y_test)
                aggregated_predictions.extend(y_pred)
                # Print confusion matrix for this fold
                if print_per_fold:
                    print(f"Confusion Matrix for Fold {fold_idx}:")
                    _confusion_matrix_print(conf_matrix, self.labels)

                    # Print classification report for this fold
                    print(f"Classification Report for Fold {fold_idx}:")
                    print(classification_report(y_test, y_pred, zero_division=0))
                    print("*" * 80)

        else:
            for fold_idx, (train, test) in enumerate(
                self.kf.split(X, y, groups=self.kfold_group), start=1
            ):
                X_train, X_test = X[train], X[test]
                y_train, y_test = y[train], y[test]
                test_model.fit(X_train, y_train)
                y_pred = test_model.predict(
                    X_test, optimal_threshold=optimal_threshold, score=score
                )
                conf_matrix = confusion_matrix(y_test, y_pred)

                # Aggregate true labels and predictions
                aggregated_true_labels.extend(y_test)
                aggregated_predictions.extend(y_pred)

                # Print confusion matrix for this fold
                if print_per_fold:
                    print(f"Confusion Matrix for Fold {fold_idx}:")
                    _confusion_matrix_print(conf_matrix, self.labels)
                    # Print classification report for this fold
                    print(f"Classification Report for Fold {fold_idx}:")
                    print(classification_report(y_test, y_pred, zero_division=0))
                    print("*" * 80)

    def conf_mat_class_kfold(
        self, X, y, test_model, score=None, optimal_threshold=False
    ):
        """
        Generates and averages confusion matrices across k-folds, producing
        a combined classification report.

        Parameters:
        -----------
        X : pandas.DataFrame or array-like
            The feature matrix for k-fold cross-validation.
        y : pandas.Series or array-like
            The target vector corresponding to `X`.
        test_model : object
            The model to be trained and evaluated on each fold.
        score : str, optional
            Optional scoring metric label for reporting purposes.
            Default is None.

        Returns:
        --------
        dict
            A dictionary containing the averaged classification report and
            confusion matrix:
            - "Classification Report": The averaged classification report as a
            dictionary.
            - "Confusion Matrix": The averaged confusion matrix as a NumPy array.

        Raises:
        -------
        ValueError
            Raised if the input data is incompatible with k-fold splitting.

        Description:
        ------------
        - This method performs k-fold cross-validation to generate confusion
        matrices for each fold.
        - Averages the confusion matrices across all folds and produces a
        combinedclassification report.
        - Prints the averaged confusion matrix and classification report.
        """

        aggregated_true_labels = []
        aggregated_predictions = []
        ### Confusion Matrix across multiple folds
        conf_ma_list = []

        if isinstance(X, pd.DataFrame):
            for train, test in self.kf.split(X, y, groups=self.kfold_group):
                X_train, X_test = X.iloc[train], X.iloc[test]
                y_train, y_test = y.iloc[train], y.iloc[test]
                test_model.fit(X_train, y_train)
                y_pred = test_model.predict(
                    X_test, optimal_threshold=optimal_threshold, score=score
                )
                conf_ma = confusion_matrix(y_test, y_pred)
                conf_ma_list.append(conf_ma)
                aggregated_true_labels.extend(y_test)
                aggregated_predictions.extend(y_pred)
        else:
            for train, test in self.kf.split(X, y, groups=self.kfold_group):
                X_train, X_test = X[train], X[test]
                y_train, y_test = y[train], y[test]
                test_model.fit(X_train, y_train)
                y_pred = test_model.predict(
                    X_test, optimal_threshold=optimal_threshold, score=score
                )
                conf_ma = confusion_matrix(y_test, y_pred)
                conf_ma_list.append(conf_ma)
                aggregated_true_labels.extend(y_test)
                aggregated_predictions.extend(y_pred)

        if score:
            print(f"Confusion Matrix Across All {len(conf_ma_list)} Folds for {score}:")
        else:
            print(f"Confusion Matrix Across All {len(conf_ma_list)} Folds:")
        conf_matrix = confusion_matrix(aggregated_true_labels, aggregated_predictions)
        self.conf_mat = conf_matrix
        _confusion_matrix_print(conf_matrix, self.labels)
        print()
        self.classification_report = classification_report(
            aggregated_true_labels,
            aggregated_predictions,
            zero_division=0,
            output_dict=True,
        )
        # Now, outside the fold loop, calculate and print the overall classification report
        print(f"Classification Report Averaged Across All Folds for {score}:")
        print(
            classification_report(
                aggregated_true_labels,
                aggregated_predictions,
                zero_division=0,
            )
        )
        print("-" * 80)
        return {
            "Classification Report": self.classification_report,
            "Confusion Matrix": conf_matrix,
        }

    def regression_report_kfold(self, X, y, test_model, score=None):
        """
        Generates and averages confusion matrices across k-folds, producing
        a combined classification report.

        Parameters:
        -----------
        X : pandas.DataFrame or array-like
            The feature matrix for k-fold cross-validation.
        y : pandas.Series or array-like
            The target vector corresponding to `X`.
        test_model : object
            The model to be trained and evaluated on each fold.
        score : str, optional
            Optional scoring metric label for reporting purposes.
            Default is None.

        Returns:
        --------
        dict
            A dictionary containing the averaged classification report and
            confusion matrix:
            - "Classification Report": The averaged classification report as a
            dictionary.
            - "Confusion Matrix": The averaged confusion matrix as a NumPy array.

        Raises:
        -------
        ValueError
            Raised if the input data is incompatible with k-fold splitting.

        Description:
        ------------
        - This method performs k-fold cross-validation to generate confusion
        matrices for each fold.
        - Averages the confusion matrices across all folds and produces a
        combined classification report.
        - Prints the averaged confusion matrix and classification report.
        """

        aggregated_pred_list = []

        if isinstance(X, pd.DataFrame):
            for train, test in self.kf.split(X, y, groups=self.kfold_group):
                X_train, X_test = X.iloc[train], X.iloc[test]
                y_train, y_test = y.iloc[train], y.iloc[test]
                test_model.fit(X_train, y_train)
                pred_y_test = test_model.predict(X_test)
                aggregated_pred_list.append(
                    self.regression_report(y_test, pred_y_test, print_results=False),
                )
        else:
            for train, test in self.kf.split(X, y, groups=self.kfold_group):
                X_train, X_test = X[train], X[test]
                y_train, y_test = y[train], y[test]
                test_model.fit(X_train, y_train)
                pred_y_test = test_model.predict(X_test)
                aggregated_pred_list.append(
                    self.regression_report(y_test, pred_y_test, print_results=False),
                )

        pred_df = pd.DataFrame(aggregated_pred_list)
        mean_dict = dict(pred_df.mean())
        print("*" * 80)
        print(f"Average performance across {len(aggregated_pred_list)} Folds:")
        pprint(mean_dict)
        print("*" * 80)
        return mean_dict

    def regression_report(self, y_true, y_pred, print_results=True):
        explained_variance = explained_variance_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        median_abs_error = median_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        reg_dict = {
            "R2": r2,
            "Explained Variance": explained_variance,
            "Mean Absolute Error": mae,
            "Median Absolute Error": median_abs_error,
            "Mean Squared Error": mse,
            "RMSE": np.sqrt(mse),
        }

        if print_results:
            print("\033[1m" + "*" * 80 + "\033[0m")  # Bold the line separator
            for key, value in reg_dict.items():
                # Use LaTeX keys directly in output
                print(f"{key}: {value:.4f}")  # Regular key with value, no green color
            print("\033[1m" + "*" * 80 + "\033[0m")  # Bold the line separator

        return reg_dict

    def _confusion_matrix_print_ML(self, conf_matrix_list):
        """
        Prints a formatted confusion matrix for multi-label classification.

        Parameters:
        -----------
        conf_matrix_list : list of numpy.ndarray
            A list of confusion matrices, one for each label or class.

        Description:
        ------------
        - This method iterates over a list of confusion matrices, printing each
        in a formatted table with clear labeling.
        - Assumes binary classification for each label (e.g., Positive vs. Negative).
        - Includes class labels and column/row names for readability.
        """

        border = "-" * 80
        print(border)
        for i, conf_matrix in enumerate(conf_matrix_list):
            print(self.class_labels[i])
            print(f"{'':>10}Predicted:")

            # Assuming binary classification for each label (Pos vs Neg)
            max_length = max(len(str(conf_matrix.max())), 2)
            header = f"{'':>12}{'Pos':>{max_length+1}}{'Neg':>{max_length+3}}"
            print(header)
            print(border)

            # Printing the confusion matrix for the class
            print(
                f"Actual: Pos {conf_matrix[0,0]:>{max_length}} ({self.labels[0]})  {conf_matrix[0,1]:>{max_length}} ({self.labels[1]})"
            )
            print(
                f"{'':>8}Neg {conf_matrix[1,0]:>{max_length}} ({self.labels[2]})  {conf_matrix[1,1]:>{max_length}} ({self.labels[3]})"
            )
            print(border)


def train_val_test_split(
    X,
    y,
    stratify_y=None,
    train_size=0.6,
    validation_size=0.2,
    test_size=0.2,
    random_state=3,
    stratify_cols=None,
):
    """
    Splits data into train, validation, and test sets, supporting stratification
    by specific columns or the target variable.

    Parameters:
    -----------
    X : pandas.DataFrame or array-like
        The feature matrix to split.
    y : pandas.Series or array-like
        The target vector corresponding to `X`.
    stratify_y : bool, optional
        If True, stratifies based on the target variable (`y`).
        If None or False, no stratification is applied.
        Default is None.
    train_size : float, optional
        Proportion of the data to allocate to the training set.
        Default is 0.6.
    validation_size : float, optional
        Proportion of the data to allocate to the validation set.
        Default is 0.2.
    test_size : float, optional
        Proportion of the data to allocate to the test set.
        Default is 0.2.
    random_state : int, optional
        Random seed for reproducibility. Default is 3.
    stratify_cols : list, pandas.DataFrame, or None, optional
        Columns to use for stratification, in addition to or instead of `y`.
        Default is None.

    Returns:
    --------
    tuple of (pandas.DataFrame, pandas.Series)
        A tuple containing train, validation, and test sets:
        (`X_train`, `X_valid`, `X_test`, `y_train`, `y_valid`, `y_test`).

    Raises:
    -------
    ValueError
        Raised if the sizes for train, validation, and test do not sum to 1.0
        or if invalid stratification keys are provided.

    Description:
    ------------
    - Splits data into three sets: train, validation, and test.
    - Supports stratification based on the target variable (`y`) or specific
      columns (`stratify_cols`).
    - Ensures the proportions of the split sets are consistent with the
      specified `train_size`, `validation_size`, and `test_size`.

    Notes:
    ------
    - The sum of `train_size`, `validation_size`, and `test_size` must equal 1.0.
    - Stratification ensures the distribution of classes or categories is
      preserved across splits.
    - The function works seamlessly with both `pandas.DataFrame` and array-like
      data structures.
    """

    # Standardize stratification parameters
    if stratify_y is False:
        stratify_y = None
    if stratify_cols is False:
        stratify_cols = None

    if stratify_cols is not None and stratify_y is not None:
        if type(stratify_cols) == pd.DataFrame:
            stratify_key = pd.concat([stratify_cols, y], axis=1)
        else:
            stratify_key = pd.concat([X[stratify_cols], y], axis=1)
    elif stratify_cols is not None:
        stratify_key = X[stratify_cols]
    elif stratify_y is not None:
        stratify_key = y
    else:
        stratify_key = None

    if stratify_cols is not None:
        stratify_key = stratify_key.fillna("")

    X_train, X_valid_test, y_train, y_valid_test = train_test_split(
        X,
        y,
        test_size=1 - train_size,
        stratify=stratify_key,  # Use stratify_key here
        random_state=random_state,
    )

    proportion = test_size / (validation_size + test_size)

    if stratify_cols is not None and stratify_y:
        # Creating stratification columns out of stratify_cols list
        if type(stratify_cols) == pd.DataFrame:
            strat_key_val_test = pd.concat(
                [stratify_cols.loc[X_valid_test.index, :], y_valid_test], axis=1
            )
        else:
            strat_key_val_test = pd.concat(
                [X_valid_test[stratify_cols], y_valid_test], axis=1
            )
    elif stratify_cols is not None:
        strat_key_val_test = X_valid_test[stratify_cols]
    elif stratify_y is not None:
        strat_key_val_test = y_valid_test
    else:
        strat_key_val_test = None

    if stratify_cols is not None:
        strat_key_val_test = strat_key_val_test.fillna("")

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_valid_test,
        y_valid_test,
        test_size=proportion,
        stratify=strat_key_val_test,
        random_state=random_state,
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def kfold_split(
    classifier,
    X,
    y,
    stratify=False,
    kfold_group=None,
    scoring=["roc_auc"],
    n_splits=10,
    random_state=3,
):
    """

    :param: n_split = number of k folds
    :param: random state of cv
    :param: False = Stratified KFold, True= KF
    :param: self.X = Lat/Hold float
    :param: self.y = Lat/Hold features

    :return cross validation function using params
    """
    if stratify:
        skf = StratifiedKFold(
            n_splits=n_splits, random_state=random_state, shuffle=True
        )  # Define the stratified ksplit
        return skf
    elif kfold_group is not None:
        gkf = GroupKFold(n_splits=n_splits)
        return gkf
    else:
        kf = KFold(
            n_splits=n_splits, random_state=random_state, shuffle=True
        )  # Define the stratified ksplit1

        return kf


def get_cross_validate(classifier, X, y, kf, scoring=["roc_auc"], kfgroups=None):
    """
    :param classifier: Machine learning model or pipeline to evaluate.
    :param X: Feature matrix (pandas.DataFrame or array-like).
    :param y: Target vector (pandas.Series or array-like).
    :param kf: Cross-validation splitting strategy.
    :param scoring: List of scoring metrics. Default = ["roc_auc"].

    :return: Cross-validation results including training scores, validation
             scores, and fitted estimators.
    """
    if kfgroups is not None:
        cross_validate(
            classifier,
            X,
            y,
            scoring=scoring,
            cv=kf,
            groups=kfgroups,
            return_train_score=True,
            return_estimator=True,
        )
    else:
        return cross_validate(
            classifier,
            X,
            y,
            scoring=scoring,
            cv=kf,
            return_train_score=True,
            return_estimator=True,
        )


def _confusion_matrix_print(conf_matrix, labels):
    """
    :param conf_matrix: Confusion matrix as a 2x2 NumPy array.
    :param labels: List of labels for the matrix cells.

    :return: Prints a formatted confusion matrix.
    """
    max_length = max(len(str(conf_matrix.max())), 2)
    border = "-" * 80
    print(border)
    print(f"{'':>10}Predicted:")
    print(f"{'':>12}{'Pos':>{max_length+1}}{'Neg':>{max_length+3}}")
    print(border)
    print(
        f"Actual: Pos {conf_matrix[1,1]:>{max_length}} ({labels[3]})  {conf_matrix[1,0]:>{max_length}} ({labels[2]})"
    )
    print(
        f"{'':>8}Neg {conf_matrix[0,1]:>{max_length}} ({labels[1]})  {conf_matrix[0,0]:>{max_length}} ({labels[0]})"
    )
    print(border)


def print_pipeline(pipeline):
    """
    Prints an ASCII art representation of a scikit-learn pipeline.

    Parameters:
    pipeline : sklearn.pipeline.Pipeline or similar
        The pipeline object containing different steps to display.
    """
    steps = pipeline.steps if hasattr(pipeline, "steps") else []

    if not steps:
        print("No steps found in the pipeline!")
        return

    print("\nPipeline Steps:")
    print()
    # Box Drawing Characters
    vertical_connector = "│"
    down_arrow = "▼"

    max_width = 0

    # First pass: calculate the maximum width needed for the boxes
    for idx, (name, step) in enumerate(steps, 1):
        step_name = f"Step {idx}: {name}"
        step_class = step.__class__.__name__
        max_length = max(len(step_name), len(step_class)) + 4  # Padding of 4
        if max_length > max_width:
            max_width = max_length

    # Second pass: print the pipeline with aligned boxes
    for idx, (name, step) in enumerate(steps, 1):
        step_name = f"Step {idx}: {name}"
        step_class = step.__class__.__name__

        # Create the box top, bottom, and content with dynamic width
        top_border = "┌" + "─" * max_width + "┐"
        bottom_border = "└" + "─" * max_width + "┘"

        # Ensure that text is aligned properly
        step_name_line = f"│ {step_name.ljust(max_width - 2)} │"
        step_class_line = f"│ {step_class.ljust(max_width - 2)} │"

        # Print the box with dynamic width
        print(top_border)
        print(step_name_line)
        print(step_class_line)
        print(bottom_border)

        # Connect the steps unless it's the last one
        if idx < len(steps):
            connector_padding = " " * (max_width // 2)
            print(connector_padding + vertical_connector)
            print(connector_padding + down_arrow)

    print()


def report_model_metrics(
    model,
    X_valid=None,
    y_valid=None,
    threshold=0.5,
    print_results=True,
    print_per_fold=False,
):
    """
    Generate a comprehensive report of model performance metrics for binary,
    multiclass classification, or regression problems. Supports both single
    validation and K-Fold cross-validation.

    Parameters:
    -----------
    model : fitted model
        The trained model with `predict_proba` or `predict` methods. It should
        also include attributes `model_type` (e.g., "regression", "binary", or
        "multiclass") and optionally `multi_label` for multiclass classification.

    X_valid : DataFrame or array-like, optional
        The feature set used for validation. For K-Fold validation, this should
        represent the entire dataset.

    y_valid : Series or array-like, optional
        The true labels for the validation dataset. For K-Fold validation, this
        should correspond to the entire dataset.

    threshold : float, default=0.5
        The threshold for binary classification. Predictions above this threshold
        are classified as the positive class.

    print_results : bool, optional (default=True)
        Whether to print the metrics report.

    print_per_fold : bool, optional (default=False)
        If performing K-Fold validation, specifies whether to print metrics for
        each fold.

    Returns:
    --------
    metrics_df : DataFrame
        A DataFrame containing performance metrics. The exact structure depends
        on the type of model:
        - **Binary Classification**: Includes Precision (PPV), Average Precision,
          Sensitivity, Specificity, AUC-ROC, and Brier Score.
        - **Multiclass Classification**: Includes Precision, Recall, and F1-Score
          for each class, as well as weighted averages.
        - **Regression**: Includes MAE, MSE, RMSE, R² Score, and Explained Variance.
        - **K-Fold**: Returns averaged metrics across all folds, with individual
          fold results optionally printed if `print_per_fold=True`.

    Notes:
    ------
    - Handles binary, multiclass, and regression models, adapting metrics to the
      specific task.
    - For K-Fold cross-validation, calculates metrics for each fold and aggregates
      them into an average report.
    - If `model.kfold` is set, the function performs K-Fold validation using
      `model.kf`, the K-Fold splitter.

    Examples:
    ---------
    # Binary classification example:
    metrics_df = report_model_metrics(
        model, X_valid=X_test, y_valid=y_test, threshold=0.5
    )

    # Regression example:
    metrics_df = report_model_metrics(
        model, X_valid=X_test, y_valid=y_test
    )

    # K-Fold validation example:
    metrics_df = report_model_metrics(
        model, X_valid=X, y_valid=y, print_per_fold=True
    )
    """

    def calculate_metrics(model, X, y, threshold):
        """
        Calculate and return performance metrics for regression, binary, or
        multiclass classification models.

        Parameters:
        -----------
        model : fitted model
            The trained model with the following expected attributes:
            - `model_type`: Specifies the type of model ("regression", "binary",
            or "multiclass").
            - `multi_label` (optional): Indicates if the model is for multiclass
            classification.
            - `class_labels` (required for multiclass): A list of class labels
            for generating metrics.
            - `predict_proba` or `predict`: Methods for generating predictions.
        X : DataFrame or array-like
            The feature matrix used for generating predictions.
        y : Series or array-like
            The true labels for the dataset.
        threshold : float
            The classification threshold for binary classification. Predictions
            with probabilities above this value are classified as the positive
            class.

        Returns:
        --------
        metrics_df : DataFrame
            A DataFrame containing the calculated metrics:
            - **Regression**: Includes Mean Absolute Error (MAE), Mean Squared
            Error (MSE), Root Mean Squared Error (RMSE), R² Score, and Explained
            Variance.
            - **Binary Classification**: Includes Precision (PPV), Average
            Precision, Sensitivity, Specificity, AUC ROC, and Brier Score.
            - **Multiclass Classification**: Includes Precision, Recall, and
            F1-Score for each class, as well as weighted averages and accuracy.

        Notes:
        ------
        - For regression models, standard regression metrics are calculated.
        - For binary classification models, threshold-based metrics are computed
        using probabilities from `predict_proba`.
        - For multiclass classification models, metrics are calculated for each
        class, along with weighted averages.
        - The function assumes that the model has the necessary attributes and
        methods based on the task type.
        """

        if model.model_type == "regression":
            # Regression metrics
            y_pred = model.predict(X)
            return pd.DataFrame(
                [
                    {
                        "Metric": "Mean Absolute Error (MAE)",
                        "Value": mean_absolute_error(y, y_pred),
                    },
                    {
                        "Metric": "Mean Squared Error (MSE)",
                        "Value": mean_squared_error(y, y_pred),
                    },
                    {
                        "Metric": "Root Mean Squared Error (RMSE)",
                        "Value": np.sqrt(mean_squared_error(y, y_pred)),
                    },
                    {"Metric": "R2 Score", "Value": r2_score(y, y_pred)},
                    {
                        "Metric": "Explained Variance",
                        "Value": explained_variance_score(y, y_pred),
                    },
                ]
            )
        elif hasattr(model, "multi_label") and model.multi_label:
            # Multiclass metrics
            y_pred = np.argmax(model.predict_proba(X), axis=1)
            report = classification_report(
                y,
                y_pred,
                output_dict=True,
                target_names=model.class_labels,
                zero_division=0,
            )
            metrics = []
            for label, scores in report.items():
                if label in model.class_labels:
                    metrics.extend(
                        [
                            {
                                "Class": label,
                                "Metric": "Precision",
                                "Value": scores.get("precision", 0),
                            },
                            {
                                "Class": label,
                                "Metric": "Recall",
                                "Value": scores.get("recall", 0),
                            },
                            {
                                "Class": label,
                                "Metric": "F1-Score",
                                "Value": scores.get("f1-score", 0),
                            },
                        ]
                    )
            metrics.extend(
                [
                    {
                        "Class": "Weighted Avg",
                        "Metric": "Precision",
                        "Value": report["weighted avg"]["precision"],
                    },
                    {
                        "Class": "Weighted Avg",
                        "Metric": "Recall",
                        "Value": report["weighted avg"]["recall"],
                    },
                    {
                        "Class": "Weighted Avg",
                        "Metric": "F1-Score",
                        "Value": report["weighted avg"]["f1-score"],
                    },
                    {
                        "Class": "Weighted Avg",
                        "Metric": "Accuracy",
                        "Value": report["accuracy"],
                    },
                ]
            )
            return pd.DataFrame(metrics)
        else:
            # Binary classification metrics
            y_pred_proba = model.predict_proba(X)[:, 1]
            y_pred = [1 if pred > threshold else 0 for pred in y_pred_proba]
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
            return pd.DataFrame(
                [
                    {"Metric": "Precision/PPV", "Value": precision_score(y, y_pred)},
                    {
                        "Metric": "Average Precision",
                        "Value": average_precision_score(y, y_pred_proba),
                    },
                    {"Metric": "Sensitivity", "Value": recall_score(y, y_pred)},
                    {"Metric": "Specificity", "Value": tn / (tn + fp)},
                    {"Metric": "AUC ROC", "Value": roc_auc_score(y, y_pred_proba)},
                    {
                        "Metric": "Brier Score",
                        "Value": brier_score_loss(y, y_pred_proba),
                    },
                ]
            )

    if hasattr(model, "kfold") and model.kfold:
        print("\nRunning k-fold model metrics...\n")
        aggregated_metrics = []
        aggregated_y_true = []
        aggregated_y_pred = []
        aggregated_y_prob = []

        test_model = model
        for fold_idx, (train, test) in tqdm(
            enumerate(
                model.kf.split(X_valid, y_valid, groups=model.kfold_group), start=1
            ),
            total=model.kf.get_n_splits(),
            desc="Processing Folds",
        ):
            if isinstance(X_valid, pd.DataFrame):
                X_train, X_test = X_valid.iloc[train], X_valid.iloc[test]
                y_train, y_test = y_valid.iloc[train], y_valid.iloc[test]
            else:
                X_train, X_test = X_valid[train], X_valid[test]
                y_train, y_test = y_valid[train], y_valid[test]
            # Fit and predict for this fold
            test_model.kfold = False
            test_model.fit(X_train, y_train)

            if model.model_type != "regression":
                y_pred_proba = test_model.predict_proba(X_test)[:, 1]
                y_pred = 1 * (y_pred_proba > threshold)
                aggregated_y_true.extend(y_test.values.tolist())
                aggregated_y_pred.extend(y_pred.tolist())
                aggregated_y_prob.extend(y_pred_proba.tolist())
            else:
                y_pred = test_model.predict(X_test)
                aggregated_y_true.extend(y_test.values.tolist())
                aggregated_y_pred.extend(y_pred.tolist())

            # Calculate metrics using existing logic
            fold_metrics = calculate_metrics(test_model, X_test, y_test, threshold)

            if isinstance(fold_metrics, pd.DataFrame):
                fold_metrics["Fold"] = fold_idx
                aggregated_metrics.append(fold_metrics)

            # Print fold-specific metrics
            if print_results and print_per_fold:
                print(f"Metrics for Fold {fold_idx}:")
                print(fold_metrics)
                print("*" * 80)

            test_model.kfold = True

        if model.model_type == "regression":
            avg_metrics_df = pd.DataFrame(
                [
                    {
                        "Metric": "Mean Absolute Error (MAE)",
                        "Value": mean_absolute_error(
                            aggregated_y_true, aggregated_y_pred
                        ),
                    },
                    {
                        "Metric": "Mean Squared Error (MSE)",
                        "Value": mean_squared_error(
                            aggregated_y_true, aggregated_y_pred
                        ),
                    },
                    {
                        "Metric": "Root Mean Squared Error (RMSE)",
                        "Value": np.sqrt(
                            mean_squared_error(aggregated_y_true, aggregated_y_pred)
                        ),
                    },
                    {
                        "Metric": "R2 Score",
                        "Value": r2_score(aggregated_y_true, aggregated_y_pred),
                    },
                    {
                        "Metric": "Explained Variance",
                        "Value": explained_variance_score(
                            aggregated_y_true, aggregated_y_pred
                        ),
                    },
                ]
            )
        else:
            tn, fp, fn, tp = confusion_matrix(
                aggregated_y_true, aggregated_y_pred
            ).ravel()

            avg_metrics_df = pd.DataFrame(
                [
                    {
                        "Metric": "Precision/PPV",
                        "Value": precision_score(aggregated_y_true, aggregated_y_pred),
                    },
                    {
                        "Metric": "Average Precision",
                        "Value": average_precision_score(
                            aggregated_y_true, aggregated_y_prob
                        ),
                    },
                    {
                        "Metric": "Sensitivity",
                        "Value": recall_score(aggregated_y_true, aggregated_y_pred),
                    },
                    {"Metric": "Specificity", "Value": tn / (tn + fp)},
                    {
                        "Metric": "AUC ROC",
                        "Value": roc_auc_score(aggregated_y_true, aggregated_y_prob),
                    },
                    {
                        "Metric": "Brier Score",
                        "Value": brier_score_loss(aggregated_y_true, aggregated_y_prob),
                    },
                ]
            )

        if print_results:
            print("\nAverage Metrics Across All Folds:")
            print(avg_metrics_df)
            print("-" * 80)

        return avg_metrics_df
    else:
        # Standard single validation logic
        metrics = calculate_metrics(model, X_valid, y_valid, threshold)

        if isinstance(metrics, pd.DataFrame):
            if print_results:
                print("*" * 80)
                print(f"Report Model Metrics: {model.estimator_name}")
                print()
                print(metrics)
                print("*" * 80)
            return metrics
        else:
            metrics_df = pd.DataFrame(metrics, index=[0]).T.rename(columns={0: ""})

            if print_results:
                print("*" * 80)
                print(f"Report Model Metrics: {model.estimator_name}")
                print()
                for key, value in metrics.items():
                    print(
                        f"{key}: {value:.4f}"
                        if isinstance(value, float)
                        else f"{value}"
                    )

            return metrics_df
