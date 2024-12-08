# causal_inference.py
import pandas as pd
import networkx as nx
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.cit import fisherz
import numpy as np
import json
import random
from math import factorial
from sklearn.linear_model import LinearRegression

class CausalInference:
    def __init__(self, data, model, target_variable):
        self.data = data  # pandas DataFrame
        self.pc_graph = None
        self.model = model  # Trained machine learning model
        self.gamma = None  # Dictionary to hold normalized causal strengths gamma_i
        self.target_variable = target_variable  # Name of the target variable
        self.ida_graph = None
        self.regression_models = {} 

    def run_pc_algorithm(self, alpha=0.05):
        data_np = self.data.to_numpy()
        pc_result = pc(data_np, alpha, fisherz)
        self.pc_graph = pc_result.G
        return self.pc_graph

    def draw_graph(self, file_path):
        pyd = GraphUtils.to_pydot(self.pc_graph)
        pyd.write_png(file_path)

    def load_causal_strengths(self, json_file_path):
        """
        Load causal strengths (beta_i) from JSON file and compute gamma_i.
        """
        # Load causal effects from JSON file
        with open(json_file_path, 'r') as f:
            causal_effects_list = json.load(f)
        
        # Build the causal graph
        G = nx.DiGraph()

        # Step 1: Add all nodes (features) from self.data columns
        nodes = list(self.data.columns)
        G.add_nodes_from(nodes)

        # Step 2: Add edges based on the causal effects from the JSON file
        for item in causal_effects_list:
            pair = item['Pair']
            mean_causal_effect = item['Mean_Causal_Effect']
            if mean_causal_effect is None:
                continue  # Skip if causal effect is None
            # Split the pair into source and target
            source, target = pair.split('->')
            source = source.strip()
            target = target.strip()
            # Add edge to the graph with the causal effect as weight
            G.add_edge(source, target, weight=mean_causal_effect)

        # Save the graph with all nodes and edges
        self.ida_graph = G.copy()

        # Now, compute the total causal effect from each feature to the target variable
        features = self.data.columns.tolist()
        beta_dict = {}
        for feature in features:
            if feature == self.target_variable:
                continue
            # Find all paths from feature to target_variable
            try:
                paths = list(nx.all_simple_paths(G, source=feature, target=self.target_variable))
            except nx.NetworkXNoPath:
                continue  # No path from this feature to target variable
            total_effect = 0
            for path in paths:
                # Compute the product of the edge weights along the path
                effect = 1
                for i in range(len(path)-1):
                    edge_weight = G[path[i]][path[i+1]]['weight']
                    effect *= edge_weight
                total_effect += effect
            if total_effect != 0:
                beta_dict[feature] = total_effect

        # Compute gamma_i = |beta_i| / sum_j |beta_j|
        total_causal_effect = sum(abs(beta) for beta in beta_dict.values())
        if total_causal_effect == 0:
            # Avoid division by zero
            self.gamma = {k: 0.0 for k in features}
        else:
            self.gamma = {k: abs(beta_dict.get(k, 0.0)) / total_causal_effect for k in features}
        return self.gamma
    
    def get_topological_order(self, S):
        """
        Returns the topological order of variables after intervening on subset S.
        """
        # Create a copy of the causal graph to modify
        G_intervened = self.ida_graph.copy()
        
        # Remove incoming edges to features in S
        for feature in S:
            G_intervened.remove_edges_from(list(G_intervened.in_edges(feature)))

        # Add any missing nodes (isolated nodes)
        missing_nodes = set(self.data.columns) - set(G_intervened.nodes)
        G_intervened.add_nodes_from(missing_nodes)
        
        # Perform topological sort
        try:
            order = list(nx.topological_sort(G_intervened))
        except nx.NetworkXUnfeasible:
            raise ValueError("The causal graph contains cycles.")
        
        return order
    
    def get_parents(self, feature):
        """
        Returns the list of parent features for a given feature in the causal graph.
        """
        return list(self.ida_graph.predecessors(feature))

    def sample_marginal(self, feature):
        """
        Sample a value from the marginal distribution of the specified feature.
        """
        return self.data[feature].sample(1).iloc[0]

    def sample_conditional(self, feature, parent_values):
        """
        Sample a value for a feature conditioned on its parent features using precomputed regression model.
        """
        # Effective parents are those not in S and not the target variable
        effective_parents = [p for p in self.get_parents(feature) if p != self.target_variable]
        
        if not effective_parents:
            return self.sample_marginal(feature)

        # Create a unique key for the feature and its effective parents
        model_key = (feature, tuple(sorted(effective_parents)))  # Sorted for consistency

        # Check if a precomputed regression model exists for the feature and its effective parents
        if model_key not in self.regression_models:
            # Fit a regression model for this feature given its effective parents
            X = self.data[effective_parents].values
            y = self.data[feature].values
            reg = LinearRegression()
            reg.fit(X, y)

            # Estimate the standard deviation from residuals
            residuals = y - reg.predict(X)
            std = residuals.std()

            # Store the regression model and std in the dictionary
            self.regression_models[model_key] = (reg, std)

        # Use the precomputed regression model and std
        reg, std = self.regression_models[model_key]

        # Prepare parent values for prediction using NumPy arrays
        parent_values_array = np.array([parent_values[parent] for parent in effective_parents]).reshape(1, -1)
        mean = reg.predict(parent_values_array)[0]

        # Sample from a normal distribution centered at the predicted mean
        sampled_value = np.random.normal(mean, std)
        return sampled_value

    def compute_v_do(self, S, x_S, num_samples=50, is_classifier=False):
        samples = []
        all_features = [col for col in self.data.columns if col != self.target_variable]
        variables_order = self.get_topological_order(S)
        for _ in range(num_samples):
            sample = {}
            for feature in S:
                sample[feature] = x_S[feature]
            for feature in variables_order:
                if feature in S or feature == self.target_variable:
                    continue
                parents = self.get_parents(feature)
                parents = [p for p in parents if p != self.target_variable]
                parent_values = {}
                for parent in parents:
                    if parent in S:
                        parent_values[parent] = x_S[parent]
                    else:
                        parent_values[parent] = sample[parent]
                if not parent_values:
                    sample[feature] = self.sample_marginal(feature)
                else:
                    sample[feature] = self.sample_conditional(feature, parent_values)
            for feature in all_features:
                if feature not in sample and feature not in S:
                    sample[feature] = self.sample_marginal(feature)
            intervened_data = pd.DataFrame([sample])
            intervened_data = intervened_data[self.model.feature_names_in_]
            if is_classifier:
                proba = self.model.predict_proba(intervened_data)[0][1]  # Assuming binary classification
            else:
                proba = self.model.predict(intervened_data)[0]
            samples.append(proba)
        v_S = np.mean(samples)
        return v_S

    def compute_modified_shap_proba(self, x, num_samples=50, shap_num_samples=50, is_classifier=False):
        features = [col for col in self.data.columns if col != self.target_variable]
        n_features = len(features)
        phi_causal = {feature: 0.0 for feature in features}

        data_without_target = self.data.drop(columns=[self.target_variable], errors='ignore')
        data_without_target = data_without_target[self.model.feature_names_in_]
        if is_classifier:
            E_fX = self.model.predict_proba(data_without_target)[:, 1].mean()  # Assuming binary classification
        else:
            E_fX = self.model.predict(data_without_target).mean()

        x_ordered = x[self.model.feature_names_in_]
        if is_classifier:
            f_x = self.model.predict_proba(x_ordered.to_frame().T)[0][1]  # Assuming binary classification
        else:
            f_x = self.model.predict(x_ordered.to_frame().T)[0]

        for _ in range(shap_num_samples):
            S_size = random.randint(0, n_features)
            S = random.sample(features, S_size)
            for i in features:
                if i in S:
                    continue
                S_without_i = S.copy()
                S_with_i = S + [i]
                x_S = x[S_without_i] if S_without_i else pd.Series(dtype=float)
                x_Si = x[S_with_i] if S_with_i else pd.Series(dtype=float)
                v_S = self.compute_v_do(S_without_i, x_S, num_samples=num_samples, is_classifier=is_classifier)
                v_Si = self.compute_v_do(S_with_i, x_Si, num_samples=num_samples, is_classifier=is_classifier)
                weight = (factorial(len(S_without_i)) * factorial(n_features - len(S_without_i) - 1)) / factorial(n_features)
                gamma_i = self.gamma.get(i, 0.0)
                weight *= gamma_i
                delta_v = v_Si - v_S
                phi_causal[i] += weight * delta_v

        sum_phi_causal = sum(phi_causal.values())
        if sum_phi_causal == 0:
            phi_normalized = {k: 0.0 for k in phi_causal.keys()}
        else:
            scaling_factor = (f_x - E_fX) / sum_phi_causal
            phi_normalized = {k: v * scaling_factor for k, v in phi_causal.items()}

        return phi_normalized