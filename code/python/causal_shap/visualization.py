# visualization.py
import shap
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self):
        pass

    def plot_shap_summary(self, model, X_test, file_path):
        explainer = shap.TreeExplainer(model)
        shap_obj = explainer(X_test)
        shap.plots.beeswarm(shap_obj[:,:,1], max_display=25, show=False)
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()

    def plot_classification_report(self, report):
        print(report)

    def plot_causal_graph(self, graph, file_path):
        import networkx as nx
        nx.draw(graph, with_labels=True)
        plt.savefig(file_path)
        plt.close()
    
