import numpy as np
import pandas as pd

def iterative_feature_deletion_with_avg_output_regression(model, input_features, attribution_scores, top_k=None):
    """
    Iteratively deletes features based on attribution scores and computes the average model output for regression.
    
    Parameters:
    - model: Trained Scikit-learn regression model.
    - input_features (pd.Series): The instance to explain.
    - attribution_scores (pd.Series or dict): Feature attributions (e.g., SHAP values).
    - top_k (int): Number of top features to delete. If None, deletes all features.
    
    Returns:
    - avg_output (float): Average model output after deletions.
    """
    if top_k is None:
        top_k = len(input_features)
    
    # Sort features by absolute attribution scores in descending order
    sorted_features = input_features.index[np.argsort(-np.abs(attribution_scores))]
    
    # Initialize list to store outputs
    output_values = []
    
    # Original prediction
    modified_input = input_features.copy().astype(float)  # Ensure float dtype
    original_output = model.predict_proba([modified_input])[0]
    output_values.append(original_output)
    
    # Iteratively delete features cumulatively
    for i in range(top_k):
        feature_to_delete = sorted_features[i]
        modified_input[feature_to_delete] = 0.0  # Assign float zero
        prediction = model.predict_proba([modified_input])[0]
        output_values.append(prediction)
    
    # Compute average output
    avg_output = np.mean(output_values)
    return avg_output



def iterative_feature_addition_with_avg_output_regression(model, input_features, attribution_scores, top_k=None):
    """
    Iteratively adds features based on attribution scores and computes the average model output.
    
    Parameters:
    - model: Trained Scikit-learn regression model.
    - input_features (pd.Series): The instance to explain.
    - attribution_scores (pd.Series or dict): Feature attributions (e.g., SHAP values).
    - top_k (int): Number of top features to add. If None, adds all features.
    
    Returns:
    - avg_output (float): Average model output after additions.
    """
    if top_k is None:
        top_k = len(input_features)
    
    # Sort features by attribution scores in descending order
    sorted_features = input_features.index[np.argsort(-attribution_scores)]
    
    # Initialize list to store outputs
    output_values = []
    
    # Initialize modified input with baseline (e.g., zeros)
    modified_input = pd.Series(0, index=input_features.index)
    
    # Iteratively add features
    for i in range(top_k):
        feature_to_add = sorted_features[i]
        modified_input[feature_to_add] = input_features[feature_to_add].astype(float)
        prediction = model.predict_proba([modified_input])[0]
        output_values.append(prediction)
    
    # Original prediction
    original_output = model.predict_proba([input_features])[0]
    output_values.append(original_output)
    
    # Compute average output
    avg_output = np.mean(output_values)
    return avg_output

