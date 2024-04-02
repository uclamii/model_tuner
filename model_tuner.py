import pandas as pd
import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold, KFold
from pprint import pprint
from sklearn.metrics import get_scorer
from sklearn.metrics import fbeta_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
        stratify=True,
        stratify_by=None,
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
        early_stopping_rounds=None,  # Number of rounds to enable early stopping
        early_stopping_monitor="logloss",  # Metric to monitor for early stopping
        early_stopping_mode="min",  # Mode for the monitored quantity ('min' or 'max')
        early_stopping_patience=0,  # Patience for early stopping
    ):
        self.name = name
        self.estimator_name = estimator_name
        self.calibrate = calibrate
        self.pipeline = pipeline
        self.original_estimator = estimator
        if scaler_type == "standard_scaler":
            pipeline_steps = [("standard_scaler", StandardScaler())]
        if scaler_type == None:
            pipeline_steps = []
        if impute:
            pipeline_steps.append(("imputer", SimpleImputer(strategy=impute_strategy)))
        self.pipeline_steps = pipeline_steps
        if self.pipeline:
            self.estimator = Pipeline(
                self.pipeline_steps + [(self.estimator_name, self.original_estimator)]
            )
        else:
            self.estimator
        self.grid = grid
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
        self.imbalance_sampler = imbalance_sampler
        self.kf = None
        self.xval_output = None
        self.stratify = stratify
        self.stratify_by = stratify_by
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
        self.labels = ["TP", "FN", "FP", "TN"]
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_monitor = early_stopping_monitor
        self.early_stopping_mode = early_stopping_mode
        self.early_stopping_patience = early_stopping_patience

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
                        method="sigmoid",
                    ).fit(X, y)
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
                        method="sigmoid",
                    ).fit(X, y)

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
                        stratify=stratify,
                        stratify_by=self.stratify_by,
                        train_size=self.train_size,
                        validation_size=self.validation_size,
                        test_size=self.test_size,
                        calibrate=True,
                        random_state=self.random_state,
                    )
                    self.X_train = X_train  # returns training data as df for X
                    self.X_valid = X_valid  # returns validation data as df for X
                    self.X_test = X_test  # returns test data as df for X
                    self.y_train = y_train  # returns training data as df for y
                    self.y_valid = y_valid  # returns validation data as df for y
                    self.y_test = y_test  # returns test data as df for y
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
                        method="sigmoid",
                    ).fit(X_test, y_test)
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
                        stratify=stratify,
                        stratify_by=self.stratify_by,
                        train_size=self.train_size,
                        validation_size=self.validation_size,
                        calibrate=True,
                        test_size=self.test_size,
                        random_state=self.random_state,
                    )
                    self.X_train = X_train  # returns training data as df for X
                    self.X_valid = X_valid  # returns validation data as df for X
                    self.X_test = X_test  # returns test data as df for X
                    self.y_train = y_train  # returns training data as df for y
                    self.y_valid = y_valid  # returns validation data as df for y
                    self.y_test = y_test  # returns test data as df for y
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
                        self.fit(X_train, y_train, score)
                    #  calibrate model, and save output
                    self.estimator = CalibratedClassifierCV(
                        self.estimator,
                        cv="prefit",
                        method="sigmoid",
                    ).fit(X_test, y_test)
                    print(
                        f"{score} after calibration:",
                        get_scorer(score)(self.estimator, X_valid, y_valid),
                    )

                else:
                    pass
        return

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
                    stratify=self.stratify,
                    scoring=self.scoring[0],
                )
                # print(self.xval_output)
                # print(self.xval_output['estimator'])
                ## TODO: If the scores are best for the minimum then
                ## this needs to inverted max will not always be correct!
                # max_score_estimator = np.argmax(self.xval_output["test_score"])
                # self.estimator = self.xval_output["estimator"][max_score_estimator]
            else:
                classifier = self.estimator.set_params(
                    **self.best_params_per_score[score]["params"]
                )
                self.xval_output = get_cross_validate(
                    classifier,
                    X,
                    y,
                    self.kf,
                    stratify=self.stratify,
                    scoring=score,
                )
                # max_score_estimator = np.argmax(self.xval_output["test_score"])
                # self.estimator = self.xval_output["estimator"][max_score_estimator]
        else:
            if validation_data is not None and self.early_stopping_rounds:
                # Early stopping parameters are utilized
                eval_set = [validation_data]
                if score is None:
                    self.estimator.fit(
                        X,
                        y,
                        early_stopping_rounds=self.early_stopping_rounds,
                        eval_set=eval_set,
                        verbose=True,
                    )
                else:
                    self.estimator.set_params(
                        **self.best_params_per_score[score]["params"]
                    ).fit(
                        X,
                        y,
                        early_stopping_rounds=self.early_stopping_rounds,
                        eval_set=eval_set,
                        verbose=True,
                    )
            else:
                # Fitting without early stopping
                if score is None:
                    self.estimator.fit(X, y)
                else:
                    self.estimator.set_params(
                        **self.best_params_per_score[score]["params"]
                    ).fit(X, y)
        return

    def predict(self, X, y=None, optimal_threshold=True):
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
        stratify=None,
        stratify_by=None,
    ):

        if self.kfold:
            self.kf = kfold_split(
                self.estimator,
                X,
                y,
                stratify=self.stratify,
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
                    stratify=stratify,
                    stratify_by=self.stratify_by,
                    train_size=self.train_size,
                    validation_size=self.validation_size,
                    test_size=self.test_size,
                    calibrate=False,
                    random_state=self.random_state,
                )
            )
            self.X_train = X_train  # returns training data as df for X
            self.X_valid = X_valid  # returns validation data as df for X
            self.X_test = X_test  # returns test data as df for X
            self.y_train = y_train  # returns training data as df for y
            self.y_valid = y_valid  # returns validation data as df for y
            self.y_test = y_test  # returns test data as df for y
            if self.balance:
                rus = self.imbalance_sampler
                X_train, y_train = rus.fit_sample(X_train, y_train)
                print("Resampled dataset shape {}".format(Counter(y_train)))

            for score in self.scoring:
                scores = []
                for params in tqdm(self.grid):
                    if self.early_stopping_rounds:
                        eval_set = [(X_valid.values, y_valid.values)]
                        estimator_name = self.estimator_name + "__eval_set"
                        clf = self.estimator.set_params(**params).fit(
                            X_train,
                            y_train,
                            xgb__eval_set=eval_set,
                            xgb__verbose=False,
                            xgb__early_stopping_rounds=self.early_stopping_rounds,
                            xgb__eval_metric=self.early_stopping_monitor,
                        )
                    else:
                        clf = self.estimator.set_params(**params).fit(X_train, y_train)

                    scores.append(get_scorer(score)(clf, X_valid, y_valid))

                self.best_params_per_score[score] = {
                    "params": self.grid[np.argmax(scores)],
                    "score": np.max(scores),
                }

                if f1_beta_tune:  # tune threshold
                    self.tune_threshold_Fbeta(
                        score, X_train, y_train, X_valid, y_valid, betas
                    )

                if self.display:
                    print("Best score/param set found on validation set:")
                    pprint(self.best_params_per_score[score])
                    print("Best " + score + ": %0.3f" % (np.max(scores)), "\n")
                    y_pred_valid = clf.predict(X_valid)
                    conf_mat = confusion_matrix(y_valid, y_pred_valid)
                    print("Confusion matrix on validation set: ")
                    _confusion_matrix_print(conf_mat, self.labels)  # TODO: LS
                    print()
                    print(classification_report(y_valid, y_pred_valid))
                    print("-" * 80)

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
        stratify,
        train_size,
        validation_size,
        test_size,
        random_state,
        stratify_by,
        calibrate,
    ):

        # if calibrate:
        #     X = X.join(self.dropped_strat_cols)
        # Determine the stratify parameter based on stratify and stratify_by
        if stratify_by:
            # Creating stratification columns out of stratify_by list
            stratify_key = X[stratify_by] if stratify_by else None
        elif stratify:
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

        # Further split (validation + test) set into validation and test sets
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_valid_test,
            y_valid_test,
            test_size=proportion,
            stratify=(
                y_valid_test if stratify and stratify_by is None else None
            ),  # Adjust stratification here
            random_state=random_state,
        )

        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def get_best_score_params(self, X, y):
        aggregated_true_labels = []
        aggregated_predictions = []

        for score in self.scoring:

            print("# Tuning hyper-parameters for %s" % score)

            if self.randomized_grid:
                clf = RandomizedSearchCV(
                    self.estimator,
                    self.grid,
                    scoring=score,
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
                    scoring=score,
                    cv=self.kf,
                    n_jobs=self.n_jobs,
                    verbose=2,
                )

            # Reset lists for each scoring metric
            aggregated_true_labels.clear()
            aggregated_predictions.clear()

            ### Confusion Matrix across multiple folds
            conf_ma_list = []

            clf.fit(X, y)
            self.estimator = clf.best_estimator_
            test_model = clf.best_estimator_
            #### TODO: Confusion matrix and classification report function
            #### Takes best estimator and kfold split and give report on CM and CR

            #### TODO: Implement threshold tuning for kfold split

            for train, test in self.kf.split(X, y):
                X_train, X_test = X[train], X[test]
                y_train, y_test = y[train], y[test]
                test_model.fit(X_train, y_train)
                pred_y_test = test_model.predict(X_test)
                conf_ma = confusion_matrix(y_test, pred_y_test)
                conf_ma_list.append(conf_ma)
                aggregated_true_labels.extend(y_test)
                aggregated_predictions.extend(pred_y_test)

            print(f"Confusion Matrix Average Across {len(conf_ma_list)} Folds")
            conf_matrix = np.mean(conf_ma_list, axis=0).astype(int)
            # Assume _confusion_matrix_print is defined elsewhere
            _confusion_matrix_print(conf_matrix, self.labels)
            print()

            # Now, outside the fold loop, calculate and print the overall classification report
            print(f"Classification Report Averaged Across All Folds for {score}:")
            print(
                classification_report(
                    aggregated_true_labels, aggregated_predictions, zero_division=0
                )
            )
            print("-" * 80)

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
                print("The model is trained on the full development set.")
                print("The scores are computed on the full evaluation set." + "\n")

            self.best_params_per_score[score] = {
                "params": clf.best_params_,
                "score": clf.best_score_,
            }
            # self.estimator = clf.best_estimator_


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
