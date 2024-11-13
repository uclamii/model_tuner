import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold, KFold
from pprint import pprint
from sklearn.metrics import get_scorer, explained_variance_score, mean_squared_error
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
)

from skopt import BayesSearchCV
import copy
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import (
    cross_val_predict,
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.model_selection import ParameterSampler
from tqdm import tqdm
from sklearn.feature_selection import SelectKBest
from .bootstrapper import evaluate_bootstrap_metrics
from sklearn.calibration import CalibratedClassifierCV

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
        model_type="classification",
        class_labels=None,
        multi_label=False,
        calibration_method="sigmoid",  # 04_27_24 --> added calibration method
        custom_scorer=[],
        bayesian=False,
    ):
        self.name = name
        self.estimator_name = estimator_name
        self.calibrate = calibrate
        self.original_estimator = estimator
        self.feature_selection = feature_selection
        self.model_type = model_type
        self.multi_label = multi_label
        self.calibration_method = (
            calibration_method  # 04_27_24 --> added calibration method
        )
        self.imbalance_sampler = imbalance_sampler

        if imbalance_sampler:
            from imblearn.pipeline import Pipeline

            self.PipelineClass = Pipeline
        else:
            from sklearn.pipeline import Pipeline

            self.PipelineClass = Pipeline

        self.pipeline_steps = pipeline_steps
        if self.pipeline_steps:
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
        if scoring == ["roc_auc"] and model_type == "regression":
            scoring == ["r2"]
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
        self.threshold = {score: 0 for score in self.scoring}
        self.beta = 2
        self.labels = ["tn", "fp", "fn", "tp"]
        self.boost_early = boost_early
        self.custom_scorer = custom_scorer
        self.bayesian = bayesian

        ### for the moment bayesian only works using cross validation, so
        ### we use the structure that already exists for kfold
        if self.bayesian:
            self.kfold = True

    """
    Multiple helper methods that are used to fetch different parts of the pipeline.
    These use the naming convention that we enforce in the assemble_pipeline method.
    """

    def get_preprocessing_and_feature_selection_pipeline(self, pipeline):
        steps = [
            (name, transformer)
            for name, transformer in pipeline.steps
            if name.startswith("preprocess_") or name.startswith("feature_selection_")
        ]
        return self.PipelineClass(steps)

    def get_feature_selection_pipeline(self, pipeline):
        steps = [
            (name, transformer)
            for name, transformer in pipeline.steps
            if name.startswith("feature_selection_")
        ]
        return steps

    def get_preprocessing_pipeline(self, pipeline):
        # Extract steps names that start with 'preprocess_'
        preprocessing_steps = [
            (name, transformer)
            for name, transformer in pipeline.steps
            if name.startswith("preprocess_")
        ]
        # Create a new pipeline with just the preprocessing steps
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
                # ColumnTransformer steps
                if not name:
                    name = f"preprocess_column_transformer_step_{len(column_transformer_steps)}"
                else:
                    name = f"preprocess_column_transformer_{name}"
                column_transformer_steps.append((name, transformer))
            elif is_preprocessing_step(transformer):
                if is_imputer(transformer):
                    # Imputation steps
                    if not name:
                        name = f"preprocess_imputer_step_{len(imputation_steps)}"
                    else:
                        name = f"preprocess_imputer_{name}"
                    imputation_steps.append((name, transformer))
                elif is_scaler(transformer):
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
        preprocessing_steps = (
            column_transformer_steps
            + imputation_steps
            + scaling_steps
            + other_preprocessing_steps
        )

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
        if self.pipeline_steps:
            self.estimator = self.PipelineClass(copy.deepcopy(self.pipeline_steps))
        else:
            self.estimator = self.PipelineClass(
                [(self.estimator_name, copy.deepcopy(self.original_estimator))]
            )
        return

    def process_imbalance_sampler(self, X_train, y_train):

        ####  Preprocessor, Resampler, rfe, Estimator

        if self.pipeline_steps:
            preproc_test = self.get_preprocessing_pipeline(self.estimator)
        else:
            pass

        resampler_test = clone(self.estimator.named_steps["resampler"])

        X_train_preproc = preproc_test.fit_transform(X_train)

        _, y_res = resampler_test.fit_resample(X_train_preproc, y_train)

        if not isinstance(y_res, pd.DataFrame):
            y_res = pd.DataFrame(y_res)
        print(f"Distribution of y values after resampling: {y_res.value_counts()}")
        print()

    def calibrateModel(self, X, y, score=None):
        if self.kfold:
            if score == None:
                if self.calibrate:
                    # reset estimator in case of calibrated model
                    self.reset_estimator()
                    ### FIXING CODE: More efficient by removing unnecessary fit

                    classifier = self.estimator.set_params(
                        **self.best_params_per_score[self.scoring[0]]["params"]
                    )

                    self.estimator = CalibratedClassifierCV(
                        classifier,
                        cv=self.n_splits,
                        method=self.calibration_method,
                    ).fit(X, y)
                    test_model = self.estimator
                    self.conf_mat_class_kfold(X=X, y=y, test_model=test_model)
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
                    self.estimator = CalibratedClassifierCV(
                        classifier,
                        cv=self.n_splits,
                        method=self.calibration_method,
                    ).fit(X, y)
                    test_model = self.estimator
                    for s in score:
                        self.conf_mat_class_kfold(
                            X=X, y=y, test_model=test_model, score=s
                        )
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
                        self.process_imbalance_sampler(X_train, y_train)

                    self.fit(X_train, y_train)
                    #  calibrate model, and save output
                    self.estimator = CalibratedClassifierCV(
                        self.estimator,
                        cv="prefit",
                        method=self.calibration_method,
                    ).fit(X_test, y_test)
                    self.calibrate_report(X_valid, y_valid)
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
                        self.process_imbalance_sampler(X_train, y_train)

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
                    ).fit(X_test, y_test)
                    test_model = self.estimator
                    self.calibrate_report(X_valid, y_valid, score=score)
                    print(
                        f"{score} after calibration:",
                        get_scorer(score)(self.estimator, X_valid, y_valid),
                    )

                else:
                    pass

        return

    def get_train_data(self, X, y):
        return X.loc[self.X_train_index], y.loc[self.y_train_index]

    def get_valid_data(self, X, y):
        return X.loc[self.X_valid_index], y.loc[self.y_valid_index]

    def get_test_data(self, X, y):
        return X.loc[self.X_test_index], y.loc[self.y_test_index]

    def calibrate_report(self, X, y, score=None):
        y_pred_valid = self.predict(X, optimal_threshold=False)
        if self.multi_label:
            conf_mat = multilabel_confusion_matrix(y, y_pred_valid)
        else:
            conf_mat = confusion_matrix(y, y_pred_valid)
        if score:
            print(f"Confusion matrix on validation set for {score}")
        else:
            print(f"Confusion matrix on validation set:")
        _confusion_matrix_print(conf_mat, self.labels)
        print()
        self.classification_report = classification_report(y, y_pred_valid)
        print(self.classification_report)
        print("-" * 80)

    def fit(self, X, y, validation_data=None, score=None):
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
                    classifier,
                    X,
                    y,
                    self.kf,
                    scoring=scorer,
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
                            self.get_preprocessing_and_feature_selection_pipeline(
                                self.estimator
                            )
                        )

                        # Set parameters and fit the pipeline
                        preproc_feat_select_pipe.set_params(**params_no_estimator).fit(
                            X, y
                        )

                        # Transform the validation data
                        X_valid_transformed = preproc_feat_select_pipe.transform(
                            X_valid
                        )
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
                            self.get_preprocessing_and_feature_selection_pipeline(
                                self.estimator
                            )
                        )

                        # Set parameters and fit the pipeline
                        preproc_feat_select_pipe.set_params(**params_no_estimator).fit(
                            X, y
                        )

                        # Transform the validation data
                        X_valid_transformed = preproc_feat_select_pipe.transform(
                            X_valid
                        )
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
                            "Specified score not found in scoring dictionary. Please use a score that was parsed for tuning."
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

    def return_metrics(self, X, y, optimal_threshold=False):

        if self.kfold:
            for score in self.scoring:
                if self.model_type != "regression":
                    print(
                        "\n"
                        + "Detailed classification report for %s:" % self.name
                        + "\n"
                    )
                    self.conf_mat_class_kfold(X, y, self.test_model, score)

                    print("The model is trained on the full development set.")
                    print("The scores are computed on the full evaluation set." + "\n")
                    self.return_metrics_kfold(X, y, self.test_model, score)

                else:
                    self.regression_report_kfold(X, y, self.test_model, score)

                if self.feature_selection:
                    self.print_selected_best_features(X)
        else:
            y_pred_valid = self.predict(X, optimal_threshold=optimal_threshold)
            if self.model_type != "regression":

                if self.multi_label:
                    conf_mat = multilabel_confusion_matrix(y, y_pred_valid)
                    self._confusion_matrix_print_ML(conf_mat)
                else:
                    conf_mat = confusion_matrix(y, y_pred_valid)
                    print("Confusion matrix on set provided: ")
                    _confusion_matrix_print(conf_mat, self.labels)
                    if optimal_threshold:
                        threshold = self.threshold[self.scoring[0]]
                    else:
                        threshold = 0.5
                    model_metrics_df = report_model_metrics(self, X, y, threshold)
                    print("-" * 80)
                    pprint(model_metrics_df.iloc[0].to_dict())
                    print("-" * 80)
                print()
                self.classification_report = classification_report(
                    y, y_pred_valid, output_dict=True
                )
                print(classification_report(y, y_pred_valid))
                print("-" * 80)

                if self.feature_selection:
                    k_best_features = self.print_selected_best_features(X)

                    return {
                        "Classification Report": self.classification_report,
                        "Confusion Matrix": conf_mat,
                        "K Best Features": k_best_features,
                    }
                else:
                    return {
                        "Classification Report": self.classification_report,
                        "Confusion Matrix": conf_mat,
                    }
            else:
                reg_report = self.regression_report(y, y_pred_valid)
                if self.feature_selection:
                    k_best_features = self.print_selected_best_features(X)
                    return {
                        "Regression Report": reg_report,
                        "K Best Features": k_best_features,
                    }
                else:
                    return reg_report

    def predict(self, X, y=None, optimal_threshold=False):
        if self.model_type == "regression":
            optimal_threshold = False
        if self.kfold and y is not None:
            return cross_val_predict(estimator=self.estimator, X=X, y=y, cv=self.kf)
        else:
            if optimal_threshold:
                return (
                    self.predict_proba(X)[:, 1]
                    > self.threshold[
                        self.scoring[0]
                    ]  # TODO generalize so that user can select score
                ) * 1
            else:
                return self.estimator.predict(X)

    def predict_proba(self, X, y=None):
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

        if self.display:
            print_pipeline(self.estimator)

        if self.kfold:
            self.kf = kfold_split(
                self.estimator,
                X,
                y,
                stratify=self.stratify_y,
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
                        for train, test in self.kf.split(X, y):

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
                        for train, test in self.kf.split(X, y):

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
                self.process_imbalance_sampler(X_train, y_train)

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

                            # Exclude the resampler if present
                            if self.imbalance_sampler:
                                params_no_estimator = {
                                    key: value
                                    for key, value in params_no_estimator.items()
                                    if not key.startswith("resampler__")
                                }

                            # Get the combined preprocessing and feature selection pipeline
                            preproc_feat_select_pipe = (
                                self.get_preprocessing_and_feature_selection_pipeline(
                                    self.estimator
                                )
                            )

                            preproc_feat_select_pipe.set_params(
                                **params_no_estimator
                            ).fit(X_train, y_train)

                            # Transform the validation data
                            X_valid_transformed = preproc_feat_select_pipe.transform(
                                X_valid
                            )
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

                        ### updating the params in the param grid with these updated parameters
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
                            params[f"{self.estimator_name}__n_estimators"] = clf[
                                len(clf) - 1
                            ].best_iteration
                        except:
                            ### catboost case
                            params[f"{self.estimator_name}__n_estimators"] = clf[
                                len(clf) - 1
                            ].best_iteration_

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

    def print_selected_best_features(self, X):

        feat_select_pipeline = self.get_feature_selection_pipeline(self.estimator)
        feat_select_pipeline = feat_select_pipeline[0][1]
        print()
        support = feat_select_pipeline.get_support()
        if isinstance(X, pd.DataFrame):
            print("Feature names selected:")
            support = X.columns[support].to_list()
        else:
            print("Feature columns selected:")
        print(support)
        print()
        return support

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

    def return_metrics_kfold(self, X, y, test_model, score=None):

        aggregated_pred_list = []
        if score is not None:
            threshold = self.threshold[score]
        else:
            threshold = self.threshold[self.scoring[0]]

        if threshold == 0:
            threshold = 0.5

        if isinstance(X, pd.DataFrame):
            for train, test in self.kf.split(X, y):
                X_train, X_test = X.iloc[train], X.iloc[test]
                y_train, y_test = y.iloc[train], y.iloc[test]
                test_model.fit(X_train, y_train)
                aggregated_pred_list.append(
                    report_model_metrics(test_model, X_test, y_test, threshold),
                )
        else:
            for train, test in self.kf.split(X, y):
                X_train, X_test = X[train], X[test]
                y_train, y_test = y[train], y[test]
                test_model.fit(X_train, y_train)
                aggregated_pred_list.append(
                    report_model_metrics(test_model, X_test, y_test, threshold),
                )

        concat_df = pd.concat(aggregated_pred_list)
        # Calculate the mean for each column
        mean_df = concat_df.groupby(concat_df.index).mean()
        mean_dict = mean_df.iloc[0].to_dict()
        print("-" * 80)
        print(f"Average performance across {len(aggregated_pred_list)} Folds:")
        pprint(mean_dict)
        print("-" * 80)
        return mean_dict

    def conf_mat_class_kfold(self, X, y, test_model, score=None):

        aggregated_true_labels = []
        aggregated_predictions = []
        ### Confusion Matrix across multiple folds
        conf_ma_list = []

        if isinstance(X, pd.DataFrame):
            for train, test in self.kf.split(X, y):
                X_train, X_test = X.iloc[train], X.iloc[test]
                y_train, y_test = y.iloc[train], y.iloc[test]
                test_model.fit(X_train, y_train)
                pred_y_test = test_model.predict(X_test)
                conf_ma = confusion_matrix(y_test, pred_y_test)
                conf_ma_list.append(conf_ma)
                aggregated_true_labels.extend(y_test)
                aggregated_predictions.extend(pred_y_test)
        else:
            for train, test in self.kf.split(X, y):
                X_train, X_test = X[train], X[test]
                y_train, y_test = y[train], y[test]
                test_model.fit(X_train, y_train)
                pred_y_test = test_model.predict(X_test)
                conf_ma = confusion_matrix(y_test, pred_y_test)
                conf_ma_list.append(conf_ma)
                aggregated_true_labels.extend(y_test)
                aggregated_predictions.extend(pred_y_test)

        if score:
            print(
                f"Confusion Matrix Average Across {len(conf_ma_list)} Folds for {score}:"
            )
        else:
            print(f"Confusion Matrix Average Across {len(conf_ma_list)} Folds:")
        conf_matrix = np.mean(conf_ma_list, axis=0).astype(int)
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

        aggregated_pred_list = []

        if isinstance(X, pd.DataFrame):
            for train, test in self.kf.split(X, y):
                X_train, X_test = X.iloc[train], X.iloc[test]
                y_train, y_test = y.iloc[train], y.iloc[test]
                test_model.fit(X_train, y_train)
                pred_y_test = test_model.predict(X_test)
                aggregated_pred_list.append(
                    self.regression_report(y_test, pred_y_test, print_results=False),
                )
        else:
            for train, test in self.kf.split(X, y):
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
            "Explained Variance": explained_variance,
            "R2": r2,
            "Mean Absolute Error": mae,
            "Median Absolute Error": median_abs_error,
            "Mean Squared Error": mse,
            "RMSE": np.sqrt(mse),
        }

        if print_results:
            print("*" * 80)
            pprint(reg_dict)
            print("*" * 80)
        return reg_dict

    def _confusion_matrix_print_ML(self, conf_matrix_list):
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
    classifier, X, y, stratify=False, scoring=["roc_auc"], n_splits=10, random_state=3
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
    else:
        kf = KFold(
            n_splits=n_splits, random_state=random_state, shuffle=True
        )  # Define the stratified ksplit1

        return kf


def get_cross_validate(classifier, X, y, kf, scoring=["roc_auc"]):
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
    print("========================")

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
):
    """
    Generate a DataFrame of model performance metrics for given models,
    predictions, or probability estimates.

    Parameters:
    -----------

    X_valid : DataFrame, optional
        The feature set used for validating the model(s).

    y_valid : Series, optional
        The true labels for the validation set.


    Returns:
    --------
    metrics_df : DataFrame
        A DataFrame containing calculated metrics for each model or prediction
        column, with metrics including:
        - Precision/PPV
        - Average Precision
        - Sensitivity (Recall)
        - Specificity
        - AUC ROC
        - Brier Score
    """

    metrics = {}
    y_pred_proba = model.predict_proba(X_valid)[:, 1]
    y_pred = [1 if pred > threshold else 0 for pred in y_pred_proba]
    tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()
    precision = precision_score(y_valid, y_pred)
    recall = recall_score(y_valid, y_pred)
    roc_auc = roc_auc_score(y_valid, y_pred_proba)
    brier_score = brier_score_loss(y_valid, y_pred_proba)
    avg_precision = average_precision_score(y_valid, y_pred_proba)
    specificity = tn / (tn + fp)
    metrics = {
        "Precision/PPV": precision,
        "Average Precision": avg_precision,
        "Sensitivity": recall,
        "Specificity": specificity,
        "AUC ROC": roc_auc,
        "Brier Score": brier_score,
    }

    metrics_df = pd.DataFrame(metrics, index=[0])
    return metrics_df
