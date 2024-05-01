import pandas as pd
import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold, KFold
from pprint import pprint
from sklearn.metrics import get_scorer
from sklearn.metrics import fbeta_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    cross_val_predict,
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.model_selection import ParameterSampler
from tqdm import tqdm
from sklearn.feature_selection import SelectKBest, f_classif

# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import RandomOverSampler
from collections import Counter

from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris, load_breast_cancer

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
        balance=False,
        imbalance_sampler=None,
        train_size=0.6,
        validation_size=0.2,
        test_size=0.2,
        stratify_y=True,
        stratify_cols=None,
        stratify_test_val=True,
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
        selectKBest=-1,
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
        if scaler_type == "standard_scaler":
            pipeline_steps = [("standard_scaler", StandardScaler())]
        elif scaler_type == None:
            pipeline_steps = []
        pipeline_steps = [
            step for step in pipeline_steps if not isinstance(step[1], (SimpleImputer))
        ]
        if impute:
            pipeline_steps.append(("imputer", SimpleImputer()))
        if selectKBest != -1:
            pipeline_steps.append(
                ("selectKBest", SelectKBest(f_classif, k=selectKBest))
            )
        self.pipeline_steps = pipeline_steps
        if self.pipeline:
            self.estimator = Pipeline(
                self.pipeline_steps + [(self.estimator_name, self.original_estimator)]
            )
        else:
            self.estimator
        self.grid = grid
        self.class_labels = class_labels
        self.kfold = kfold
        self.balance = balance
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
        self.stratify_test_val = stratify_test_val
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
            self.estimator = Pipeline(
                self.pipeline_steps + [(self.estimator_name, self.original_estimator)]
            )
        else:
            self.estimator
        return

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

                    # self.xval_output = get_cross_validate(
                    # CalibratedClassifierCV(
                    #     classifier,
                    #     cv=self.n_splits,
                    #     method="sigmoid",
                    # ),
                    #     X,
                    #     y,
                    #     self.kf,
                    #     stratify=self.stratify,
                    #     scoring=self.scoring[0],
                    # )
                    # max_score_estimator = np.argmax(self.xval_output["test_score"])
                    # self.estimator = self.xval_output["estimator"][max_score_estimator]
                    self.estimator = CalibratedClassifierCV(
                        classifier,
                        cv=self.n_splits,
                        method=self.calibration_method,  # 04_27_24 --> previously hard-coded to sigmoid
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
                        method=self.calibration_method,  # 04_27_24 --> previously hard-coded to sigmoid
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
                    if self.balance:
                        rus = self.imbalance_sampler
                        X_res, y_res = rus.fit_sample(X_train, y_train)
                        print("Resampled dataset shape {}".format(Counter(y_res)))

                        # fit model based on "score_tune_params" best score params
                        self.reset_estimator()
                        self.fit(X_res, y_res)
                    else:
                        self.fit(X_train, y_train)
                    #  calibrate model, and save output
                    self.estimator = CalibratedClassifierCV(
                        self.estimator,
                        cv="prefit",
                        method=self.calibration_method,  # 04_27_24 --> previously hard-coded to sigmoid
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
                    # fit estimator
                    if self.balance:
                        rus = self.imbalance_sampler
                        X_res, y_res = rus.fit_sample(X_train, y_train)
                        print("Resampled dataset shape {}".format(Counter(y_res)))

                        # fit model based on "score_tune_params" best score params
                        self.reset_estimator()
                        self.fit(X_res, y_res, score)
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
                        method=self.calibration_method,  # 04_27_24 --> previously hard-coded to sigmoid
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
                # print(self.xval_output)
                # print(self.xval_output['estimator'])
                ## TODO: If the scores are best for the minimum then
                ## this needs to inverted max will not always be correct!
                # max_score_estimator = np.argmax(self.xval_output["test_score"])
                # self.estimator = self.xval_output["estimator"][max_score_estimator]
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
                # max_score_estimator = np.argmax(self.xval_output["test_score"])
                # self.estimator = self.xval_output["estimator"][max_score_estimator]
        else:
            if score is None:
                if self.xgboost_early:

                    ## L.S. 04_20_24
                    ## previously x, valid, y_valid was a part of self, now we
                    ## use indices
                    # if validation_data:
                    X_valid, y_valid = validation_data
                    if isinstance(X_valid, pd.DataFrame):
                        eval_set = [(X_valid.values, y_valid.values)]
                    else:
                        eval_set = [(X_valid, y_valid)]
                    # else:
                    #     eval_set = [] # changed only up until this line 04_20_24
                    estimator_eval_set = f"{self.estimator_name}__eval_set"
                    estimator_verbosity = f"{self.estimator_name}__verbose"

                    xgb_params = {
                        estimator_eval_set: eval_set,
                        estimator_verbosity: self.verbosity,
                    }
                    self.estimator.fit(X, y, **xgb_params)
                else:
                    self.estimator.fit(X, y)
            else:
                if self.xgboost_early:
                    # if validation_data:
                    X_valid, y_valid = validation_data
                    if isinstance(X_valid, pd.DataFrame):
                        eval_set = [(X_valid.values, y_valid.values)]
                    else:
                        eval_set = [(X_valid, y_valid)]
                    # else:
                    #     eval_set = []
                    estimator_eval_set = f"{self.estimator_name}__eval_set"
                    estimator_verbosity = f"{self.estimator_name}__verbose"

                    xgb_params = {
                        estimator_eval_set: eval_set,
                        estimator_verbosity: self.verbosity,
                    }
                    self.estimator.set_params(
                        **self.best_params_per_score[score]["params"]
                    ).fit(X, y, **xgb_params)
                else:
                    self.estimator.set_params(
                        **self.best_params_per_score[score]["params"]
                    ).fit(X, y)

        return

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
        else:
            # print("y:", y)
            # print("X:", X)
            # print("Stratify:", stratify)
            # print("validation_size:", self.validation_size)
            # print("test_size:", self.test_size)
            # print(f"Stratify By {self.stratify_by}")

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

            if self.balance:
                rus = self.imbalance_sampler
                X_train, y_train = rus.fit_sample(X_train, y_train)
                print("Resampled dataset shape {}".format(Counter(y_train)))

            for score in self.scoring:
                scores = []
                for params in tqdm(self.grid):
                    if self.xgboost_early:
                        if isinstance(X_valid, pd.DataFrame):
                            eval_set = [(X_valid.values, y_valid.values)]
                        else:
                            eval_set = [(X_valid, y_valid)]
                        estimator_eval_set = f"{self.estimator_name}__eval_set"
                        estimator_verbosity = f"{self.estimator_name}__verbose"

                        if params.get(estimator_verbosity):
                            self.verbosity = params[estimator_verbosity]
                        else:
                            self.verbosity = False

                        xgb_params = {
                            estimator_eval_set: eval_set,
                            estimator_verbosity: self.verbosity,
                        }
                        ### xgb_params required here in order to ensure
                        ### custom estimator name. X_valid, y_valid
                        ### needed to use .values.
                        clf = self.estimator.set_params(**params).fit(
                            X_train, y_train, **xgb_params
                        )
                    else:
                        clf = self.estimator.set_params(**params).fit(X_train, y_train)

                    if score in self.custom_scorer:
                        scorer_func = self.custom_scorer[score]
                    else:
                        scorer_func = get_scorer(score)

                    score_value = scorer_func(self.estimator, X_valid, y_valid)
                    # if custom_scorer
                    scores.append(score_value)

                self.best_params_per_score[score] = {
                    "params": self.grid[np.argmax(scores)],
                    "score": np.max(scores),
                }

                if f1_beta_tune:  # tune threshold
                    self.tune_threshold_Fbeta(
                        score, X_train, y_train, X_valid, y_valid, betas
                    )

                if not self.calibrate:
                    if self.display:
                        print("Best score/param set found on validation set:")
                        pprint(self.best_params_per_score[score])
                        print("Best " + score + ": %0.3f" % (np.max(scores)), "\n")
                        y_pred_valid = clf.predict(X_valid)
                        if self.model_type != "regression":

                            if self.multi_label:
                                conf_mat = multilabel_confusion_matrix(
                                    y_valid, y_pred_valid
                                )
                                self._confusion_matrix_print_ML(conf_mat)
                            else:
                                conf_mat = confusion_matrix(y_valid, y_pred_valid)
                                print("Confusion matrix on validation set: ")
                                _confusion_matrix_print(
                                    conf_mat, self.labels
                                )  # TODO: LS

                            print()
                            self.classification_report = classification_report(
                                y_valid, y_pred_valid, output_dict=True
                            )
                            print(classification_report(y_valid, y_pred_valid))
                            print("-" * 80)
                else:
                    if self.display:
                        print("Best score/param set found on validation set:")
                        pprint(self.best_params_per_score[score])
                        print("Best " + score + ": %0.3f" % (np.max(scores)), "\n")

            # for score in self.scoring:
            #     scores = []
            #     for params in tqdm(self.grid):
            #         clf = self.estimator.set_params(**params).fit(X_train, y_train)
            #         scores.append(get_scorer(score)(clf, X_valid, y_valid))

            #     self.best_params_per_score[score] = {
            #         "params": self.grid[np.argmax(scores)],
            #         "score": np.max(scores),
            #     }

            #     if f1_beta_tune:  # tune threshold
            #         self.tune_threshold_Fbeta(
            #             score, X_train, y_train, X_valid, y_valid, betas
            #         )

            #     if self.display:
            #         print("Best score/param set found on validation set:")
            #         pprint(self.best_params_per_score[score])
            #         print("Best " + score + ": %0.3f" % (np.max(scores)), "\n")

    def tune_threshold_Fbeta(self, score, X_train, y_train, X_valid, y_valid, betas):
        """Method to tune threshold on validation dataset using F beta score."""

        print("Fitting model with best params and tuning for best threshold ...")
        # predictions
        y_valid_probs = (
            self.estimator.set_params(**self.best_params_per_score[score]["params"])
            .fit(X_train, y_train)
            .predict_proba(X_valid)[:, 1]
        )

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
            stratify_key = X[stratify_cols] if stratify_cols else None
        elif stratify_y:
            stratify_key = y
        else:
            stratify_key = None

        if self.drop_strat_feat:
            self.dropped_strat_cols = X[self.drop_strat_feat]
            X = X.drop(columns=self.drop_strat_feat)

        # TODO: add special case to drop additional columns for stratify list\
        # Split the dataset into training and (validation + test) sets
        X_train, X_valid_test, y_train, y_valid_test = train_test_split(
            X,
            y,
            test_size=1 - train_size,
            stratify=stratify_key,  # Use stratify_key here
            random_state=random_state,
        )

        # Determine the proportion of validation to test size in the remaining dataset
        proportion = test_size / (validation_size + test_size)

        if self.stratify_test_val:
            if stratify_y and not stratify_cols:
                strat_key_val_test = y_valid_test
            elif stratify_cols:
                strat_key_val_test = X_valid_test[stratify_cols]
            else:
                strat_key_val_test = None
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
            test_model = clf.best_estimator_

            #### TODO: Implement threshold tuning for kfold split

            if self.display:
                ## Make classification report and conf matrix into function
                print("\n" + "Best score/param set found on development set:")
                pprint({clf.best_score_: clf.best_params_})
                print("\n" + "Grid scores on development set:")
                means = clf.cv_results_["mean_test_score"]
                stds = clf.cv_results_["std_test_score"]
                for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
                    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

                print(
                    "\n" + "Detailed classification report for %s:" % self.name + "\n"
                )
                self.conf_mat_class_kfold(X, y, test_model, score)

                print("The model is trained on the full development set.")
                print("The scores are computed on the full evaluation set." + "\n")

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


# def train_val_test_split(X, y, stratify, train_size, validation_size, test_size, random_state):
#     if stratify:
#         stratify_param = y
#     else:
#         stratify_param = None

#     X_train, X_valid_test, y_train, y_valid_test = train_test_split(
#         X,
#         y,
#         test_size=1 - train_size,
#         stratify=stratify_param,
#         random_state=random_state,
#     )

#     # Update to fix proportions for validation and test split
#     proportion = test_size / (validation_size + test_size)

#     X_valid, X_test, y_valid, y_test = train_test_split(
#         X_valid_test,
#         y_valid_test,
#         test_size=proportion,
#         stratify=y_valid_test if stratify else None,
#         random_state=random_state,
#     )

#     return X_train, X_valid, X_test, y_train, y_valid, y_test

# def train_val_test_split(
#     X,
#     y,
#     stratify,
#     train_size,
#     validation_size,
#     test_size,
#     random_state,
# ):
#     X_train, X_valid_test, y_train, y_valid_test = train_test_split(
#         X,
#         stratify,  # stratify contains another array and last the label
#         test_size=1 - train_size,
#         train_size=train_size,
#         stratify=stratify,
#         random_state=random_state,
#     )
#     X_valid, X_test, y_valid, y_test = train_test_split(
#         X_valid_test,
#         y_valid_test,
#         test_size=(1 - train_size - validation_size) / (1 - train_size),
#         train_size=(1 - train_size - test_size) / (1 - train_size),
#         stratify=y_valid_test,
#         random_state=random_state,
#     )
#     return X_train, X_valid, X_test, y_train[:, -1], y_valid[:, -1], y_test[:, -1]


if __name__ == "__main__":
    iris = load_iris()
    iris = pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]],
        columns=iris["feature_names"] + ["target"],
    )
    features = [col for col in iris.columns if col != "target"]
    target = "target"

    X = iris[features].values  # independant variables
    y = iris[target].values.astype(int)  # dependent variable

    # breast_sk = load_breast_cancer()
    # breast = pd.DataFrame(
    #     data=np.c_[breast_sk.data, breast_sk.target],
    # )
    # breast.columns = list(breast_sk.feature_names) + ["target"]
    # features = [col for col in breast.columns if col != "target"]
    # target = "target"

    # X = breast[features].values  # independant variables
    # y = breast[target].values.astype(int)  # dependent variable

    lr = LogisticRegression(class_weight="balanced", C=1, max_iter=1000)

    estimator_name = "lr"
    # Set the parameters by cross-validation
    tuned_parameters = [{estimator_name + "__C": np.logspace(-4, 0, 10)}]

    kfold = True
    calibrate = True

    model = Model(
        name="Iris_model",
        estimator_name=estimator_name,
        calibrate=calibrate,
        estimator=lr,
        kfold=kfold,
        stratify=True,
        grid=tuned_parameters,
        randomized_grid=False,
        n_iter=3,
        scoring=["roc_auc_ovr", "precision_macro"],
        n_splits=2,
        random_state=3,
    )

    model.grid_search_param_tuning(X, y)

    model.fit(X, y)

    ## The below calibration process replaces "base_estimator" with "estimator"
    ## for all scikit-learn versions >= 0.24

    if model.calibrate:
        model.calibrateModel(X, y)
    else:
        pass

    if kfold:
        print(model.xval_output["train_score"], model.xval_output["test_score"])
        for i in range(len(model.xval_output["estimator"])):
            print("\n" + str(i) + " Fold: ")
            if calibrate:
                importance = (
                    model.xval_output["estimator"][i]
                    .calibrated_classifiers_[i]
                    .estimator.steps[1][1]
                    .coef_[0]
                )
            else:
                importance = model.xval_output["estimator"][i].steps[1][1].coef_[0]

            sort_imp_indx = np.argsort(importance)[::-1]
            # print(importance)
            # print(sort_imp_indx)
            for i in sort_imp_indx:
                print("Feature: %s, Score: %.5f" % (features[i], importance[i]))
    else:
        if calibrate:
            importance = model.estimator.estimator.steps[1][1].coef_[0]
        else:
            importance = model.estimator.steps[1][1].coef_[0]
        sort_imp_indx = np.argsort(importance)[::-1]
        # print(importance)
        # print(sort_imp_indx)
        # summarize feature importance
        for i in sort_imp_indx:
            print("Feature: %s, Score: %.5f" % (features[i], importance[i]))

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
