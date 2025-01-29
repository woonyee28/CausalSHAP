import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics_regression(y_true, y_pred):
    """
    Calculate multiple regression metrics.

    Parameters:
    - y_true: Ground truth (continuous) labels array
    - y_pred: Predicted values array

    Returns:
    - Dictionary containing MSE, RMSE, MAE, and R-squared scores
    """
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r_squared': r2_score(y_true, y_pred)
    }
    return metrics


def calculate_metrics_classification(y_true, y_pred):
    """
    Calculate multiple classification metrics.
    
    Parameters:
    - y_true: Ground truth (binary) labels array
    - y_pred: Predicted probabilities array
    
    Returns:
    - Dictionary containing AUROC, Cross Entropy, and Brier scores
    """
    metrics = {
        'auroc': roc_auc_score(y_true, y_pred),
        'cross_entropy': log_loss(y_true, y_pred),
        'brier': brier_score_loss(y_true, y_pred)
    }
    return metrics

def iterative_feature_deletion_scores(model, X_test, y_test, attribution_scores, top_k=None):
    """
    Iteratively deletes features based on attribution scores and computes metrics globally.
    
    Parameters:
    - model: Trained classifier with predict_proba method
    - X_test: Test features DataFrame
    - y_test: Test labels
    - attribution_scores: Feature attributions with same index as X_test.columns
    - top_k: Number of top features to delete
    
    Returns:
    - Dictionary containing lists of metric values at each step
    """
    if top_k is None:
        top_k = len(X_test.columns)
    
    # Sort features by absolute attribution scores
    sorted_features = attribution_scores.abs().sort_values(ascending=False).index[:top_k]
    
    # Initialize modified input
    modified_X = X_test.copy().astype(float)
    
    # Initialize metric lists
    metrics_lists = {
        'auroc': [],
        'cross_entropy': [],
        'brier': [],
        'rmse': [],
        'mae': [],
        'r_squared': []
    }
    
    # Iteratively delete features
    for feature_to_delete in sorted_features:
        # Delete feature globally
        modified_X[feature_to_delete] = 0.0
        
        # Get predictions for all instances
        if hasattr(model, "predict_proba"):
            predictions = model.predict_proba(modified_X)[:, 1]
            # Calculate metrics
            step_metrics = calculate_metrics_classification(y_test, predictions)
        else:
            predictions = model.predict(modified_X)
            # Calculate metrics
            step_metrics = calculate_metrics_regression(y_test, predictions)
        
        # Store metrics
        for metric_name, value in step_metrics.items():
            metrics_lists[metric_name].append(value)
    
    average_scores = {metric: np.mean(values) for metric, values in metrics_lists.items()}
    
    # Return both stepwise metrics and averages
    return {
        'stepwise_metrics': metrics_lists,
        'average_scores': average_scores
    }

def iterative_feature_addition_scores(model, X_test, y_test, attribution_scores, top_k=None):
    """
    Iteratively adds features based on attribution scores and computes metrics globally.
    
    Parameters:
    - model: Trained classifier with predict_proba method
    - X_test: Test features DataFrame
    - y_test: Test labels
    - attribution_scores: Feature attributions with same index as X_test.columns
    - top_k: Number of top features to add
    
    Returns:
    - Dictionary containing lists of metric values at each step
    """
    if top_k is None:
        top_k = len(X_test.columns)
    
    # Sort features by absolute attribution scores
    sorted_features = attribution_scores.abs().sort_values(ascending=False).index[:top_k]
    
    # Initialize modified input with zeros
    modified_X = pd.DataFrame(0, index=X_test.index, columns=X_test.columns)
    
    # Initialize metric lists
    metrics_lists = {
        'auroc': [],
        'cross_entropy': [],
        'brier': [],
        'rmse': [],
        'mae': [],
        'r_squared': []
    }
    
    # Iteratively add features
    for feature_to_add in sorted_features:
        # Add feature globally
        modified_X[feature_to_add] = X_test[feature_to_add].astype(float)
        
        # Get predictions for all instances
        if hasattr(model, "predict_proba"):
            predictions = model.predict_proba(modified_X)[:, 1]
            # Calculate metrics
            step_metrics = calculate_metrics_classification(y_test, predictions)
        else:
            predictions = model.predict(modified_X)
            # Calculate metrics
            step_metrics = calculate_metrics_regression(y_test, predictions)
        
        # Store metrics
        for metric_name, value in step_metrics.items():
            metrics_lists[metric_name].append(value)
    
    average_scores = {metric: np.mean(values) for metric, values in metrics_lists.items()}
    
    # Return both stepwise metrics and averages
    return {
        'stepwise_metrics': metrics_lists,
        'average_scores': average_scores
    }

def evaluate_global_shap_scores(model, X_test, y_test, shap_values, causal=False):
    """
    Evaluate SHAP values using global insertion and deletion metrics.
    
    Parameters:
    - model: Trained classifier
    - X_test: Test features DataFrame
    - y_test: Test labels
    - shap_values: SHAP values array
    
    Returns:
    - Dictionary containing metric trajectories
    """
    # Calculate global feature importance (mean absolute SHAP value for each feature)
    if causal:
        global_importance = shap_values
    else:
        if isinstance(shap_values, list):
            global_importance = pd.Series({
                feature: np.abs(shap_values[:, idx, 1]).mean() 
                for idx, feature in enumerate(X_test.columns)
            })
        else:
            global_importance = pd.Series(
                np.abs(shap_values[:, :, 1]).mean(axis=0),
                index=X_test.columns
            )
        
    # Calculate deletion and insertion scores
    deletion_results = iterative_feature_deletion_scores(
        model, X_test, y_test, global_importance
    )
    
    insertion_results = iterative_feature_addition_scores(
        model, X_test, y_test, global_importance
    )
    
    return {
        'deletion': deletion_results,
        'insertion': insertion_results
    }


def iterative_feature_deletion_with_rmse(model, input_features, attribution_scores, y_predicted, top_k=None):
    """
    Iteratively deletes features based on attribution scores and computes the RMSE of model output for regression.
    
    Parameters:
    - model: Trained Scikit-learn regression model.
    - input_features (pd.Series): The instance to explain.
    - attribution_scores (pd.Series or dict): Feature attributions (e.g., SHAP values).
    - top_k (int): Number of top features to delete. If None, deletes all features.
    
    Returns:
    - avg_rmse (float): Average RMSE after deletions.
    """
    if top_k is None:
        top_k = len(input_features)
    
    # Sort features by absolute attribution scores in descending order
    sorted_features = input_features.index[np.argsort(-np.abs(attribution_scores))]
    
    # Original prediction
    modified_input = input_features.copy().astype(float)  # Ensure float dtype
    
    # Initialize list to store RMSE values
    rmse_values = []
    
    # Iteratively delete features cumulatively and calculate RMSE
    for i in range(top_k):
        feature_to_delete = sorted_features[i]
        modified_input[feature_to_delete] = 0.0  # Assign float zero
        if y_predicted:
            prediction = model.predict_proba([modified_input.to_numpy()])[0][1]
        else:
            prediction = model.predict_proba([modified_input.to_numpy()])[0][0]
        rmse = root_mean_squared_error([y_predicted], [prediction])
        rmse_values.append(rmse)
    
    # Compute average RMSE
    avg_rmse = np.mean(rmse_values)
    return avg_rmse


def iterative_feature_addition_with_rmse(model, input_features, attribution_scores, y_predicted, top_k=None):
    """
    Iteratively adds features based on attribution scores and computes the RMSE of model output for regression.
    
    Parameters:
    - model: Trained Scikit-learn regression model.
    - input_features (pd.Series): The instance to explain.
    - attribution_scores (pd.Series or dict): Feature attributions (e.g., SHAP values).
    - top_k (int): Number of top features to add. If None, adds all features.
    
    Returns:
    - avg_rmse (float): Average RMSE after additions.
    """
    if top_k is None:
        top_k = len(input_features)
    
    # Sort features by absolute attribution scores in descending order
    sorted_features = input_features.index[np.argsort(-np.abs(attribution_scores))]
    
    # Initialize modified input with baseline (e.g., zeros)
    modified_input = pd.Series(0, index=input_features.index)
    
    # Initialize list to store RMSE values
    rmse_values = []
    
    # Iteratively add features and calculate RMSE
    for i in range(top_k):
        feature_to_add = sorted_features[i]
        modified_input[feature_to_add] = input_features[feature_to_add].astype(float)
        if y_predicted:
            prediction = model.predict_proba([modified_input.to_numpy()])[0][1]
        else:
            prediction = model.predict_proba([modified_input.to_numpy()])[0][0]
        rmse = root_mean_squared_error([y_predicted], [prediction])
        rmse_values.append(rmse)
    
    # Compute average RMSE
    avg_rmse = np.mean(rmse_values)
    return avg_rmse
