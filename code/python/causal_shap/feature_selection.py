# feature_selection.py
import pandas as pd
import numpy as np
import shap

class FeatureSelector:
    def __init__(self, model, X_train):
        self.model = model
        self.X_train = X_train

    def gini_importance(self, threshold=0.01):
        importances = self.model.feature_importances_
        feature_names = self.X_train.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        gini_importance_df = importance_df.sort_values(by='Importance', ascending=False)
        selected_features = gini_importance_df[gini_importance_df['Importance'] > threshold]['Feature']
        return selected_features

    def shap_importance(self, X_test, threshold=0.005):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_test)
        shap_values_abs = np.abs(shap_values)
        mean_shap_values = np.mean(shap_values_abs, axis=0)
        feature_importance = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': mean_shap_values[:, 1]
        })
        shap_importance_df = feature_importance.sort_values(by='Importance', ascending=False)
        selected_features = shap_importance_df[shap_importance_df['Importance'] > threshold]['Feature']
        return selected_features

    def causal_shap_importance(self, X_test, causal_graph, threshold=0.01):
        # Implement causal SHAP based on your requirements
        pass
