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

    def sample_conditional(self, feature, parent_values, S):
        """
        Sample a value for a feature conditioned on its parent features.
        """
        # Effective parents are those not in S and not the target variable
        effective_parents = [p for p in self.get_parents(feature) if p not in S and p != self.target_variable]
        if not effective_parents:
            return self.sample_marginal(feature)

        # Fit a regression model for this feature given its effective parents
        X = self.data[effective_parents]
        y = self.data[feature]
        reg = LinearRegression()
        reg.fit(X, y)

        # Prepare parent values for prediction
        parent_df = pd.DataFrame([parent_values])
        mean = reg.predict(parent_df)[0]

        # Estimate the standard deviation from residuals
        residuals = y - reg.predict(X)
        std = residuals.std()

        # Sample from a normal distribution centered at the predicted mean
        sampled_value = np.random.normal(mean, std)
        return sampled_value


    def compute_v_do(self, S, x_S, num_samples=100):
        samples = []
        # Determine the topological order of variables after intervention
        variables_order = self.get_topological_order(S)
        for _ in range(num_samples):
            sample = {}
            # Set intervened features
            for feature in S:
                sample[feature] = x_S[feature]
            # Sample non-intervened features
            for feature in variables_order:
                if feature in S or feature == self.target_variable:
                    continue  # Skip intervened features and the target variable
                parents = self.get_parents(feature)
                # Exclude parents that are in S or are the target variable
                effective_parents = [p for p in parents if p not in S and p != self.target_variable]
                if not effective_parents:
                    # Sample from marginal distribution
                    sample[feature] = self.sample_marginal(feature)
                else:
                    # Prepare parent values
                    parent_values = {parent: sample[parent] for parent in effective_parents}
                    # Sample conditionally
                    sample[feature] = self.sample_conditional(feature, parent_values, S)
            samples.append(sample)
        intervened_data = pd.DataFrame(samples)
        # Ensure all features are present
        features_in_order = list(self.model.feature_names_in_)
        for feature in features_in_order:
            if feature not in intervened_data.columns:
                # Fill missing features with mean values
                intervened_data[feature] = self.data[feature].mean()
        # Reorder columns to match training data
        intervened_data = intervened_data[features_in_order]
        predictions = self.model.predict(intervened_data)
        v_S = np.mean(predictions)
        return v_S



    def compute_modified_shap(self, x, num_samples=100, shap_num_samples=100):
        # Exclude the target variable from the features list
        features = [col for col in self.data.columns if col != self.target_variable]
        n_features = len(features)
        phi_causal = {feature: 0.0 for feature in features}

        # Precompute E[f(X)]
        data_without_target = self.data.drop(columns=[self.target_variable], errors='ignore')
        # Reorder columns to match training data
        data_without_target = data_without_target[self.model.feature_names_in_]
        E_fX = self.model.predict(data_without_target).mean()

        # Precompute f(x)
        x_ordered = x[self.model.feature_names_in_]
        f_x = self.model.predict(x_ordered.to_frame().T)[0]

        # ... rest of your code ...


        # Monte Carlo approximation for Shapley values
        for _ in range(shap_num_samples):
            # Randomly select a subset S from features (excluding target variable)
            S_size = random.randint(0, n_features)
            S = random.sample(features, S_size)

            # For each feature i not in S
            for i in features:
                if i in S:
                    continue
                S_without_i = S.copy()
                S_with_i = S + [i]

                # x_S is the values of features in S
                x_S = x[S_without_i] if S_without_i else pd.Series(dtype=float)

                # x_Si is the values of features in S union {i}
                x_Si = x[S_with_i] if S_with_i else pd.Series(dtype=float)

                # Compute v(S)
                v_S = self.compute_v_do(S_without_i, x_S, num_samples=num_samples)

                # Compute v(S union {i})
                v_Si = self.compute_v_do(S_with_i, x_Si, num_samples=num_samples)

                # Compute weight w(S,i) = |S|!(n - |S| - 1)! / n!
                weight = (factorial(len(S_without_i)) * factorial(n_features - len(S_without_i) - 1)) / factorial(n_features)

                # Multiply by gamma_i
                gamma_i = self.gamma.get(i, 0.0)
                weight *= gamma_i

                # Marginal contribution
                delta_v = v_Si - v_S

                # Update phi_causal[i]
                phi_causal[i] += weight * delta_v

        # Normalize phi_causal
        sum_phi_causal = sum(phi_causal.values())

        if sum_phi_causal == 0:
            # Avoid division by zero
            phi_normalized = {k: 0.0 for k in phi_causal.keys()}
        else:
            scaling_factor = (f_x - E_fX) / sum_phi_causal
            phi_normalized = {k: v * scaling_factor for k, v in phi_causal.items()}

        return phi_normalized
