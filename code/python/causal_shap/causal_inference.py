# causal_inference.py
import pandas as pd
import networkx as nx
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.cit import fisherz
import numpy as np

class CausalInference:
    def __init__(self, data):
        self.data = data
        self.graph = None

    def run_pc_algorithm(self, alpha=0.05):
        data_np = self.data.to_numpy()
        pc_result = pc(data_np, alpha, fisherz)
        self.graph = pc_result.G
        return self.graph

    def draw_graph(self, file_path):
        pyd = GraphUtils.to_pydot(self.graph)
        pyd.write_png(file_path)
