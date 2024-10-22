################################################################################
#########################  Model Performance Metrics ###########################
################################################################################


def report_model_metrics(
    df=None,
    outcome_cols=None,
    pred_cols=None,
    models=None,
    X_valid=None,
    y_valid=None,
    pred_probs_df=None,
):
    """
    Generate a DataFrame of model metrics for given models or predictions.

    Parameters:
    df (DataFrame, optional): DataFrame containing outcome and prediction cols.
    outcome_cols (list, optional): List of outcome column names in df.
    pred_cols (list, optional): List of prediction column names in df.
    models (dict, optional): Dict where key is model name and value is model.
    X_valid (DataFrame, optional): DataFrame with validation data.
    y_valid (Series, optional): Series with outcome data for validation set.
    pred_probs_df (DataFrame, optional): DataFrame with predicted probabilities.

    Returns:
    metrics_df (DataFrame): DataFrame containing model metrics.
    """

    metrics = {}

    # Calculate metrics for each outcome_col-pred_col pair
    if outcome_cols is not None and pred_cols is not None and df is not None:
        for outcome_col, pred_col in zip(outcome_cols, pred_cols):
            y_true = df[outcome_col]
            y_pred_proba = df[pred_col]
            y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_proba]
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            # Calculate precision-recall curve for y_valid and y_pred_proba.
            # This returns a tuple: (precision, recall, thresholds).
            # Only the first two values (precision and recall) are needed
            # for the auc() function, hence `[:2]`. The auc() function expects
            # its inputs in the order (recall, precision). However,
            # precision_recall_curve() returns the values in the opposite order,
            # i.e., (precision, recall). Therefore, we need to reverse this
            # order. The `[::-1]` does exactly this - it reverses the tuple.
            # pr_auc = auc(*precision_recall_curve(y_valid, y_pred_proba)[:2][::-1])
            brier_score = brier_score_loss(y_true, y_pred_proba)
            avg_precision = average_precision_score(y_true, y_pred_proba)
            specificity = tn / (tn + fp)
            metrics[pred_col] = {
                "Precision/PPV": precision,
                "Average Precision": avg_precision,
                "Sensitivity": recall,
                "Specificity": specificity,
                "AUC ROC": roc_auc,
                "Brier Score": brier_score,
            }

    # Calculate metrics for each model
    if models is not None and X_valid is not None and y_valid is not None:
        for name, model in models.items():
            y_pred = model.predict(X_valid)
            y_pred_proba = model.predict_proba(X_valid)[:, 1]
            tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()
            precision = precision_score(y_valid, y_pred)
            recall = recall_score(y_valid, y_pred)
            roc_auc = roc_auc_score(y_valid, y_pred_proba)
            # Calculate precision-recall curve for y_valid and y_pred_proba.
            # This returns a tuple: (precision, recall, thresholds).
            # Only the first two values (precision and recall) are needed
            # for the auc() function, hence `[:2]`. The auc() function expects
            # its inputs in the order (recall, precision). However,
            # precision_recall_curve() returns the values in the opposite order,
            # i.e., (precision, recall). Therefore, we need to reverse this
            # order. The `[::-1]` does exactly this - it reverses the tuple.
            # pr_auc = auc(*precision_recall_curve(y_valid, y_pred_proba)[:2][::-1])
            brier_score = brier_score_loss(y_valid, y_pred_proba)
            avg_precision = average_precision_score(y_valid, y_pred_proba)
            specificity = tn / (tn + fp)
            metrics[name] = {
                "Precision/PPV": precision,
                "Average Precision": avg_precision,
                "Sensitivity": recall,
                "Specificity": specificity,
                "AUC ROC": roc_auc,
                "Brier Score": brier_score,
            }

    # Calculate metrics for each column in pred_probs_df
    if pred_probs_df is not None:
        for col in pred_probs_df.columns:
            y_pred_proba = pred_probs_df[col]
            y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_proba]
            tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()
            precision = precision_score(y_valid, y_pred)
            recall = recall_score(y_valid, y_pred)
            roc_auc = roc_auc_score(y_valid, y_pred_proba)
            # Calculate precision-recall curve for y_valid and y_pred_proba.
            # This returns a tuple: (precision, recall, thresholds).
            # Only the first two values (precision and recall) are needed
            # for the auc() function, hence `[:2]`. The auc() function expects
            # its inputs in the order (recall, precision). However,
            # precision_recall_curve() returns the values in the opposite order,
            # i.e., (precision, recall). Therefore, we need to reverse this
            # order. The `[::-1]` does exactly this - it reverses the tuple.
            # pr_auc = auc(*precision_recall_curve(y_valid, y_pred_proba)[:2][::-1])
            brier_score = brier_score_loss(y_valid, y_pred_proba)
            avg_precision = average_precision_score(y_valid, y_pred_proba)
            specificity = tn / (tn + fp)
            metrics[col] = {
                "Precision/PPV": precision,
                "Average Precision": avg_precision,
                "Sensitivity": recall,
                "Specificity": specificity,
                "AUC ROC": roc_auc,
                "Brier Score": brier_score,
            }

    metrics_df = pd.DataFrame(metrics).round(3)
    metrics_df["Mean"] = metrics_df.mean(axis=1).round(3)
    return metrics_df

