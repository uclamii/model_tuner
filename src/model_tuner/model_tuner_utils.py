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
)
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.impute import SimpleImputer
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

| Scoring                        | Function                             | Comment                        |
|--------------------------------|--------------------------------------|--------------------------------|
| Classification                 |                                      |                                |
| ‘accuracy’                     | metrics.accuracy_score               |                                |
| ‘balanced_accuracy’            | metrics.balanced_accuracy_score      |                                |
| ‘average_precision’            | metrics.average_precision_score      |                                |
| ‘neg_brier_score’              | metrics.brier_score_loss             |                                |
| ‘f1’                           | metrics.f1_score                     | for binary targets             |
| ‘f1_micro’                     | metrics.f1_score                     | micro-averaged                 |
| ‘f1_macro’                     | metrics.f1_score                     | macro-averaged                 |
| ‘f1_weighted’                  | metrics.f1_score                     | weighted average               |
| ‘f1_samples’                   | metrics.f1_score                     | by multilabel sample           |
| ‘neg_log_loss’                 | metrics.log_loss                     | requires predict_proba support |
| ‘precision’ etc.               | metrics.precision_score              | suffixes apply as with ‘f1’    |
| ‘recall’ etc.                  | metrics.recall_score                 | suffixes apply as with ‘f1’    |
| ‘jaccard’ etc.                 | metrics.jaccard_score                | suffixes apply as with ‘f1’    |
| ‘roc_auc’                      | metrics.roc_auc_score                |                                |
| ‘roc_auc_ovr’                  | metrics.roc_auc_score                |                                |
| ‘roc_auc_ovo’                  | metrics.roc_auc_score                |                                |
| ‘roc_auc_ovr_weighted’         | metrics.roc_auc_score                |                                |
| ‘roc_auc_ovo_weighted’         | metrics.roc_auc_score                |                                |
| Clustering                     |                                      |                                |
| ‘adjusted_mutual_info_score’   | metrics.adjusted_mutual_info_score   |                                |
| ‘adjusted_rand_score’          | metrics.adjusted_rand_score          |                                |
| ‘completeness_score’           | metrics.completeness_score           |                                |
| ‘fowlkes_mallows_score’        | metrics.fowlkes_mallows_score        |                                |
| ‘homogeneity_score’            | metrics.homogeneity_score            |                                |
| ‘mutual_info_score’            | metrics.mutual_info_score            |                                |
| ‘normalized_mutual_info_score’ | metrics.normalized_mutual_info_score |                                |
| ‘v_measure_score’              | metrics.v_measure_score              |                                |
| Regression                     |                                      |                                |
| ‘explained_variance’           | metrics.explained_variance_score     |                                |
| ‘max_error’                    | metrics.max_error                    |                                |
| ‘neg_mean_absolute_error’      | metrics.mean_absolute_error          |                                |
| ‘neg_mean_squared_error’       | metrics.mean_squared_error           |                                |
| ‘neg_root_mean_squared_error’  | metrics.mean_squared_error           |                                |
| ‘neg_mean_squared_log_error’   | metrics.mean_squared_log_error       |                                |
| ‘neg_median_absolute_error’    | metrics.median_absolute_error        |                                |
| ‘r2’                           | metrics.r2_score                     |                                |
| ‘neg_mean_poisson_deviance’    | metrics.mean_poisson_deviance        |                                |
| ‘neg_mean_gamma_deviance’      | metrics.mean_gamma_deviance          |                                |

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
        drop_strat_feat=None,
        grid=None,
        scoring=["roc_auc"],
        n_splits=10,
        random_state=3,
        n_jobs=1,
        display=True,
        feature_names=None,
        randomized_grid=False,
        n_iter=100,
        trained=False,
        pipeline=True,
        scaler_type="min_max_scaler",
        impute_strategy="mean",
        impute=False,
        pipeline_steps=[("min_max_scaler", MinMaxScaler())],
        xgboost_early=False,
        selectKBest=False,
        model_type="classification",
        class_labels=None,
        multi_label=False,
        calibration_method="sigmoid",  # 04_27_24 --> added calibration method
        custom_scorer=[],
    ):
        self.name = name
        self.estimator_name = estimator_name
        self.calibrate = calibrate
        self.pipeline = pipeline
        self.original_estimator = estimator
        self.selectKBest = selectKBest
        self.model_type = model_type
        self.multi_label = multi_label
        self.calibration_method = (
            calibration_method  # 04_27_24 --> added calibration method
        )

        if scaler_type == None:
            pipeline_steps = pipeline_steps
        pipeline_steps = [
            step
            for step in pipeline_steps
            if not isinstance(
                step[1], (SimpleImputer) or not isinstance(step[1], (StandardScaler))
            )
        ]
        if scaler_type == "standard_scaler":
            pipeline_steps.append(("standard_scaler", StandardScaler()))
        if impute:
            pipeline_steps.append(("imputer", SimpleImputer()))
        if selectKBest:
            pipeline_steps.append(("selectKBest", SelectKBest()))

        if imbalance_sampler:
            from imblearn.pipeline import Pipeline

            self.PipelineClass = Pipeline
            pipeline_steps.append(("Resampler", imbalance_sampler))
        else:
            from sklearn.pipeline import Pipeline

            self.PipelineClass = Pipeline

        self.pipeline_steps = pipeline_steps
        if self.pipeline:
            self.estimator = self.PipelineClass(
                self.pipeline_steps + [(self.estimator_name, self.original_estimator)]
            )
        else:
            self.estimator
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
        self.imbalance_sampler = imbalance_sampler
        self.kf = None
        self.xval_output = None
        self.stratify_y = stratify_y
        self.stratify_cols = stratify_cols
        self.drop_strat_feat = drop_strat_feat
        self.n_splits = n_splits
        self.scoring = scoring
        self.best_params_per_score = {score: 0 for score in self.scoring}
        self.display = display
        self.feature_names = feature_names
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size
        self.threshold = {score: 0 for score in self.scoring}
        self.beta = 2
        self.trained = trained
        self.labels = ["tp", "fn", "fp", "tn"]
        self.xgboost_early = xgboost_early
        self.custom_scorer = custom_scorer

    def reset_estimator(self):
        if self.pipeline:
            self.estimator = self.PipelineClass(
                self.pipeline_steps + [(self.estimator_name, self.original_estimator)]
            )
        else:
            self.estimator
        return

    def process_imbalance_sampler(self, X_train, y_train):
        imputer_test = clone(self.estimator.named_steps["imputer"])
        resampler_test = clone(self.estimator.named_steps["Resampler"])

        X_train_imputed = imputer_test.fit_transform(X_train)

        X_res, y_res = resampler_test.fit_resample(X_train_imputed, y_train)

        if not isinstance(y_res, pd.DataFrame):
            y_res = pd.DataFrame(y_res)
        print(f"Distribution of y values after resampling: {y_res.value_counts()}")
        print()

    def calibrateModel(
        self,
        X,
        y,
        score=None,
        stratify=None,
    ):
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
                    ) = self.train_val_test_split(
                        X=X,
                        y=y,
                        stratify_y=self.stratify_y,
                        stratify_cols=self.stratify_cols,
                        train_size=self.train_size,
                        validation_size=self.validation_size,
                        test_size=self.test_size,
                        calibrate=True,
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

                    else:
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
                    ) = self.train_val_test_split(
                        X=X,
                        y=y,
                        stratify_y=self.stratify_y,
                        stratify_cols=self.stratify_cols,
                        train_size=self.train_size,
                        validation_size=self.validation_size,
                        calibrate=True,
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
                    if 'device' in self.estimator[-1].get_params():
                        print("Change back to CPU")
                        self.estimator[-1].set_params(**{'device': 'cpu'})

                    # fit estimator
                    if self.imbalance_sampler:
                        self.process_imbalance_sampler(X_train, y_train)
                    else:
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
        if self.kfold:
            if score == None:
                classifier = self.estimator.set_params(
                    **self.best_params_per_score[self.scoring[0]]["params"]
                )
                self.xval_output = get_cross_validate(
                    classifier,
                    X,
                    y,
                    self.kf,
                    stratify=self.stratify_y,
                    scoring=self.scoring[0],
                )
            else:
                if score in self.custom_scorer:
                    scorer = self.custom_scorer[score]
                else:
                    scorer = score
                classifier = self.estimator.set_params(
                    **self.best_params_per_score[score]["params"]
                )
                self.xval_output = get_cross_validate(
                    classifier,
                    X,
                    y,
                    self.kf,
                    stratify=self.stratify_y,
                    scoring=scorer,
                )

        else:
            if score is None:
                best_params = self.best_params_per_score[self.scoring[0]]["params"]

                if self.xgboost_early:
                    X_valid, y_valid = validation_data
                    if self.selectKBest or self.pipeline:

                        params_no_estimator = {
                            key: value
                            for key, value in best_params.items()
                            if not key.startswith(f"{self.estimator_name}__")
                        }
                        if self.imbalance_sampler:
                            self.estimator[:-2].set_params(**params_no_estimator).fit(
                                X, y
                            )
                            X_valid_selected = self.estimator[:-2].transform(X_valid)
                        else:
                            self.estimator[:-1].set_params(**params_no_estimator).fit(
                                X, y
                            )
                            X_valid_selected = self.estimator[:-1].transform(X_valid)
                    else:
                        X_valid_selected = X_valid

                    X_valid, y_valid = validation_data
                    if isinstance(X_valid, pd.DataFrame):
                        eval_set = [(X_valid_selected, y_valid.values)]
                    else:
                        eval_set = [(X_valid_selected, y_valid)]
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
                if self.xgboost_early:
                    X_valid, y_valid = validation_data
                    best_params = self.best_params_per_score[score]["params"]
                    if self.selectKBest or self.pipeline:

                        params_no_estimator = {
                            key: value
                            for key, value in best_params.items()
                            if not key.startswith(f"{self.estimator_name}__")
                        }
                        if self.imbalance_sampler:
                            self.estimator[:-2].set_params(**params_no_estimator).fit(
                                X, y
                            )
                            X_valid_selected = self.estimator[:-2].transform(X_valid)
                        else:
                            self.estimator[:-1].set_params(**params_no_estimator).fit(
                                X, y
                            )
                            X_valid_selected = self.estimator[:-1].transform(X_valid)
                    else:
                        X_valid_selected = X_valid

                    X_valid, y_valid = validation_data
                    if isinstance(X_valid, pd.DataFrame):
                        eval_set = [(X_valid_selected, y_valid.values)]
                    else:
                        eval_set = [(X_valid_selected, y_valid)]
                    estimator_eval_set = f"{self.estimator_name}__eval_set"
                    estimator_verbosity = f"{self.estimator_name}__verbose"

                    xgb_params = {
                        estimator_eval_set: eval_set,
                        estimator_verbosity: self.verbosity,
                    }
                    if estimator_verbosity in self.best_params_per_score[score]["params"]:
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
        self, X_test, y_test, metrics, threshold=0.5, num_resamples=500, n_samples=500
    ):
        if self.model_type != "regression":
            y_pred_prob = pd.Series(self.predict_proba(X_test)[:, 1])
            if isinstance(y_test, pd.DataFrame) or isinstance(y_test, pd.Series):
                y_test = y_test.reset_index(drop=True)
            bootstrap_metrics = evaluate_bootstrap_metrics(
                model=None,
                y=y_test,
                y_pred_prob=y_pred_prob,
                metrics=metrics,
                threshold=threshold,
                num_resamples=num_resamples,
                n_samples=n_samples,
            )
        else:
            y_pred = pd.Series(self.predict(X_test))
            if isinstance(y_test, pd.DataFrame) or isinstance(y_test, pd.Series):
                y_test = y_test.reset_index(drop=True)
            bootstrap_metrics = evaluate_bootstrap_metrics(
                model=None,
                y=y_test,
                y_pred_prob=y_pred,
                model_type="regression",
                metrics=metrics,
                num_resamples=num_resamples,
                n_samples=n_samples,
            )
        return bootstrap_metrics

    def return_metrics(self, X_test, y_test, optimal_threshold=False):

        if self.kfold:
            for score in self.scoring:
                if self.model_type != "regression":
                    print(
                        "\n"
                        + "Detailed classification report for %s:" % self.name
                        + "\n"
                    )
                    self.conf_mat_class_kfold(X_test, y_test, self.test_model, score)

                    print("The model is trained on the full development set.")
                    print("The scores are computed on the full evaluation set." + "\n")

                else:
                    self.regression_report_kfold(X_test, y_test, self.test_model, score)

                if self.selectKBest:
                    self.print_k_best_features(X_test)
        else:
            y_pred_valid = self.predict(X_test, optimal_threshold=optimal_threshold)
            if self.model_type != "regression":

                if self.multi_label:
                    conf_mat = multilabel_confusion_matrix(y_test, y_pred_valid)
                    self._confusion_matrix_print_ML(conf_mat)
                else:
                    conf_mat = confusion_matrix(y_test, y_pred_valid)
                    print("Confusion matrix on set provided: ")
                    _confusion_matrix_print(conf_mat, self.labels)

                print()
                self.classification_report = classification_report(
                    y_test, y_pred_valid, output_dict=True
                )
                print(classification_report(y_test, y_pred_valid))
                print("-" * 80)

                if self.selectKBest:
                    k_best_features = self.print_k_best_features(X_test)

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
                reg_report = self.regression_report(y_test, y_pred_valid)
                if self.selectKBest:
                    k_best_features = self.print_k_best_features(X_test)
                    return {
                        "Regression Report": reg_report,
                        "K Best Features": k_best_features,
                    }
                else:
                    return reg_report

    def predict(self, X, y=None, optimal_threshold=False):
        if self.model_type == "regression":
            optimal_threshold = False
        if self.kfold:
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
        if self.kfold:
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
                                X.iloc[train],
                                y.iloc[train],
                                X.iloc[test],
                                y.iloc[test],
                                betas,
                                y_pred_proba,
                                kfold=True,
                            )
                            thresh_list.append(thresh)
                        average_threshold = np.mean(thresh_list)
                        self.threshold[score] = average_threshold
                else:
                    for score in self.scoring:
                        thresh_list = []
                        self.kfold = False
                        for train, test in self.kf.split(X, y):
                            self.fit(X[train], y[train])
                            y_pred_proba = self.predict_proba(X[test])[:, 1]
                            thresh = self.tune_threshold_Fbeta(
                                score,
                                X[train],
                                y[train],
                                X[test],
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
            X_train, X_valid, X_test, y_train, y_valid, y_test = (
                self.train_val_test_split(
                    X=X,
                    y=y,
                    stratify_y=self.stratify_y,
                    stratify_cols=self.stratify_cols,
                    train_size=self.train_size,
                    validation_size=self.validation_size,
                    test_size=self.test_size,
                    calibrate=False,
                    random_state=self.random_state,
                )
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
            for score in self.scoring:
                scores = []
                for params in tqdm(self.grid):
                    if self.xgboost_early:
                        estimator_verbosity = f"{self.estimator_name}__verbose"

                        if params.get(estimator_verbosity):
                            self.verbosity = params[estimator_verbosity]
                            params.pop(estimator_verbosity)
                        else:
                            self.verbosity = False

                        if self.selectKBest or self.pipeline:
                            params_no_estimator = {
                                key: value
                                for key, value in params.items()
                                if not key.startswith(f"{self.estimator_name}__")
                            }
                            if self.imbalance_sampler:
                                self.estimator[:-2].set_params(
                                    **params_no_estimator
                                ).fit(X_train, y_train)
                                X_valid_selected = self.estimator[:-2].transform(
                                    X_valid
                                )
                            else:
                                self.estimator[:-1].set_params(
                                    **params_no_estimator
                                ).fit(X_train, y_train)
                                X_valid_selected = self.estimator[:-1].transform(
                                    X_valid
                                )
                        else:
                            X_valid_selected = X_valid

                        if isinstance(X_valid, pd.DataFrame):
                            eval_set = [(X_valid_selected, y_valid.values)]
                        else:
                            eval_set = [(X_valid_selected, y_valid)]

                        estimator_eval_set = f"{self.estimator_name}__eval_set"
                        estimator_verbosity = f"{self.estimator_name}__verbose"

                        xgb_params = {
                            estimator_eval_set: eval_set,
                            estimator_verbosity: self.verbosity,
                        }

                        if estimator_verbosity in params:
                            params.pop(estimator_verbosity)

                        clf = self.estimator.set_params(**params).fit(
                            X_train, y_train, **xgb_params
                        )
                    else:
                        clf = self.estimator.set_params(**params).fit(X_train, y_train)

                    if score in self.custom_scorer:
                        scorer_func = self.custom_scorer[score]
                    else:
                        scorer_func = get_scorer(score)

                    score_value = scorer_func(clf, X_valid, y_valid)
                    # if custom_scorer
                    scores.append(score_value)

                self.best_params_per_score[score] = {
                    "params": self.grid[np.argmax(scores)],
                    "score": np.max(scores),
                }

                if f1_beta_tune:  # tune threshold
                    y_pred_proba = clf.predict_proba(X_valid)[:, 1]
                    self.tune_threshold_Fbeta(
                        score, X_train, y_train, X_valid, y_valid, betas, y_pred_proba
                    )

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

    def print_k_best_features(self, X):
        print()
        support = self.estimator.named_steps["selectKBest"].get_support()
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
        X_train,
        y_train,
        X_valid,
        y_valid,
        betas,
        y_valid_proba,
        kfold=False,
    ):
        """Method to tune threshold on validation dataset using F beta score."""

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

    def train_val_test_split(
        self,
        X,
        y,
        stratify_y,
        train_size,
        validation_size,
        test_size,
        random_state,
        stratify_cols,
        calibrate,
    ):

        # if calibrate:
        #     X = X.join(self.dropped_strat_cols)
        # Determine the stratify parameter based on stratify and stratify_cols
        if stratify_cols:
            # Creating stratification columns out of stratify_cols list
            stratify_key = X[stratify_cols]
        elif stratify_y:
            stratify_key = y
        else:
            stratify_key = None

        if self.drop_strat_feat:
            self.dropped_strat_cols = X[self.drop_strat_feat]
            X = X.drop(columns=self.drop_strat_feat)

        X_train, X_valid_test, y_train, y_valid_test = train_test_split(
            X,
            y,
            test_size=1 - train_size,
            stratify=stratify_key,  # Use stratify_key here
            random_state=random_state,
        )

        # Determine the proportion of validation to test size in the remaining dataset
        proportion = test_size / (validation_size + test_size)

        if stratify_cols:
            strat_key_val_test = X_valid_test[stratify_cols]
        elif stratify_y:
            strat_key_val_test = y_valid_test
        else:
            strat_key_val_test = None

        # Further split (validation + test) set into validation and test sets
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_valid_test,
            y_valid_test,
            test_size=proportion,
            stratify=strat_key_val_test,
            random_state=random_state,
        )

        return X_train, X_valid, X_test, y_train, y_valid, y_test

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
        print(self.classification_report)
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


def get_cross_validate(classifier, X, y, kf, stratify=False, scoring=["roc_auc"]):
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
        f"Actual: Pos {conf_matrix[0,0]:>{max_length}} ({labels[0]})  {conf_matrix[0,1]:>{max_length}} ({labels[1]})"
    )
    print(
        f"{'':>8}Neg {conf_matrix[1,0]:>{max_length}} ({labels[2]})  {conf_matrix[1,1]:>{max_length}} ({labels[3]})"
    )
    print(border)

################################################################################


class AutoKerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model, pipeline=None):
        super().__init__()
        self.model = model
        self.pipeline = pipeline

    def fit(self, X, y, **params):
        if self.pipeline:
            X = self.pipeline.fit_transform(X)
        self.model.fit(X, y, **params)
        self.model_export = self.model.export_model()
        self.best_params_per_score = self.summarize_auto_keras_params(
            self.model_export.get_config()
        )

    def predict(self, X):
        if self.pipeline:
            X = self.pipeline.transform(X)
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.pipeline:
            X = self.pipeline.transform(X)
        y_pos = self.model_export.predict(X)
        return np.c_[1 - y_pos, y_pos]

    def summarize_auto_keras_params(self, params):
        # Importing the 'deepcopy' function from the 'copy' module
        from copy import deepcopy

        # Creating a deep copy of the 'params' dictionary and storing it in 'res'
        res = deepcopy(params)

        # Initializing an empty list for the 'layers' key in the 'res' dictionary
        res["layers"] = []

        # Looping through each 'layer' in the 'params['layers']' list
        for layer in params["layers"]:
            # Appending a dictionary to 'res['layers']' with only specific keys
            # ('class_name' and 'name')
            # res['layers'].append({key: val for key, val in layer.items() if key in {'class_name', 'name'}})

            key_inc = {"class_name", "name"}
            filt_keys = {key: val for key, val in layer.items() if key in key_inc}
            res["layers"].append(filt_keys)

        # Returning the modified 'res' dictionary
        return res
