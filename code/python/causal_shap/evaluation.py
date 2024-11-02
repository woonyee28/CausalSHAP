import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

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
