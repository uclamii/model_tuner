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
    Generate a DataFrame of model performance metrics for given models, 
    predictions, or probability estimates.

    The function can evaluate model metrics for outcome-prediction column pairs 
    from a DataFrame, for specified models on validation data, or for columns 
    containing predicted probabilities.

    Parameters:
    -----------
    df : DataFrame, optional
        A DataFrame containing both the true outcome columns and corresponding 
        predicted probability columns for binary classification tasks.
    
    outcome_cols : list of str, optional
        A list of column names representing the true binary outcome in the 
        DataFrame `df`.
    
    pred_cols : list of str, optional
        A list of column names representing the predicted probabilities in the 
        DataFrame `df`.
    
    models : dict, optional
        A dictionary where the keys are model names (str) and the values are 
        model objects that implement the `.predict()` and `.predict_proba()` 
        methods.
    
    X_valid : DataFrame, optional
        The feature set used for validating the model(s).
    
    y_valid : Series, optional
        The true labels for the validation set.
    
    pred_probs_df : DataFrame, optional
        A DataFrame containing columns of predicted probabilities for evaluation.

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
        A "Mean" column is also provided, which gives the average score across 
        all metrics for each model or prediction column.
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

