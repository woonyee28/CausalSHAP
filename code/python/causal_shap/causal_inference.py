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

class CausalInference:
    def __init__(self, data, model, target_variable):
        self.data = data  # pandas DataFrame
        self.graph = None
        self.model = model  # Trained machine learning model
        self.gamma = None  # Dictionary to hold normalized causal strengths gamma_i
        self.target_variable = target_variable  # Name of the target variable

    def run_pc_algorithm(self, alpha=0.05):
        data_np = self.data.to_numpy()
        pc_result = pc(data_np, alpha, fisherz)
        self.graph = pc_result.G
        return self.graph

    def draw_graph(self, file_path):
        pyd = GraphUtils.to_pydot(self.graph)
        pyd.write_png(file_path)

    def load_causal_strengths(self, json_file_path):
        """
        Load causal strengths (beta_i) from JSON file and compute gamma_i.

        Parameters:
        - json_file_path: Path to the JSON file containing causal effects.

        The JSON file is expected to be a list of dictionaries with keys 'Pair' and 'Mean_Causal_Effect'.
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

    def compute_v_do(self, S, x_S, num_samples=100):
        """
        Compute v(S) = E[f(X) | do(X_S = x_S)]
        """
        samples = []
        for _ in range(num_samples):
            sample = {}
            for feature in self.data.columns:
                if feature in S:
                    sample[feature] = x_S[feature]
                else:
                    # Sample from the marginal distribution of the feature
                    sample[feature] = self.data[feature].sample(1).iloc[0]
            samples.append(sample)
        intervened_data = pd.DataFrame(samples)
        # Predict using the model
        predictions = self.model.predict(intervened_data)
        # Compute the expectation
        v_S = np.mean(predictions)
        return v_S


    def compute_modified_shap(self, x, num_samples=100, shap_num_samples=100):
        """
        Compute the modified SHAP values incorporating causal strengths
        Parameters:
        - x: the instance to explain, as a pandas Series
        - num_samples: number of samples for computing v(S) with do-operator
        - shap_num_samples: number of samples for SHAP value approximation
        Returns:
        - phi_normalized: dictionary of normalized SHAP values
        """
        features = self.data.columns.tolist()
        n_features = len(features)
        phi_causal = {feature: 0.0 for feature in features}

        # Precompute E[f(X)]
        E_fX = self.model.predict(self.data).mean()

        # Precompute f(x)
        f_x = self.model.predict(x.to_frame().T)[0]

        # Monte Carlo approximation for Shapley values
        for _ in range(shap_num_samples):
            # Randomly select a subset S
            S_size = random.randint(0, n_features)
            S = random.sample(features, S_size)

            # For each feature i not in S
            for i in features:
                if i in S:
                    continue
                S_without_i = S.copy()
                S_with_i = S + [i]

                # x_S is the values of features in S
                x_S = x[S_without_i]

                # x_Si is the values of features in S union {i}
                x_Si = x[S_with_i]

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
