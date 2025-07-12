import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def plot_importance_table(global_importance, top_n=20, figsize=(10, 8), save_path=None):
    """Create a table-like visualization of feature importance"""
    # Take top N features
    if len(global_importance) > top_n:
        importance_subset = global_importance.iloc[:top_n]
    else:
        importance_subset = global_importance
    
    # Convert to DataFrame for better display
    df = pd.DataFrame({
        'Feature': importance_subset.index,
        'Importance': importance_subset.values
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.axis('off')
    
    # Create table
    table = plt.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#f2f2f2', '#f2f2f2'],
        cellColours=[[('#ffffff' if i % 2 == 0 else '#f9f9f9'), 
                     ('#ffffff' if i % 2 == 0 else '#f9f9f9')] for i in range(len(df))]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)  # Adjust row height
    
    plt.title('Feature Importance from Causal SHAP Values', fontsize=16, pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()
    
    return fig, ax

def visualize_causal_graph(json_data, target_node="Prob_Class_1", 
                         min_effect_threshold=0.0, layout_seed=42, 
                         filter_target_paths=True, figsize=(10, 8),
                         node_size_factor=0.7, edge_width_factor=1.0,
                         edge_curvature=0.1, layout_type="shell"):
    """
    Visualize a causal graph from JSON data containing causal relationships.
    
    Parameters:
    -----------
    json_data : list or str
        Either a Python list of dicts with causal data or path to JSON file
    target_node : str, optional
        Name of the target node to highlight and focus paths on
    min_effect_threshold : float, optional
        Minimum absolute causal effect strength to include in the graph
    layout_seed : int, optional
        Seed for the graph layout algorithm to ensure reproducibility
    filter_target_paths : bool, optional
        If True, only include nodes that have a path to the target node
    figsize : tuple, optional
        Size of the figure (width, height) in inches
    node_size_factor : float, optional
        Factor to multiply node sizes by for larger/smaller nodes
    edge_width_factor : float, optional
        Factor to multiply edge widths by for thicker/thinner edges
    edge_curvature : float, optional
        Controls the curvature of edges; 0 = straight lines, higher values = more curved
    layout_type : str, optional
        Type of layout algorithm to use: "kamada_kawai", "spring", "circular", or "shell"
        
    Returns:
    --------
    G : networkx.DiGraph
        The graph object created
    stats : dict
        Dictionary containing graph statistics
    """
    # Set professional font for IEEE publication
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 10
    })
    
    # Load the data - either from file or use the provided list
    if isinstance(json_data, str):
        with open(json_data, 'r') as f:
            causal_data = json.load(f)
    else:
        causal_data = json_data
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add edges with causal strength as attributes
    for item in causal_data:
        source, target = item["Pair"].split("->")
        effect = item["Mean_Causal_Effect"]
        
        # Only add edges that meet the threshold
        if abs(effect) >= min_effect_threshold:
            G.add_edge(source.strip(), target.strip(), weight=effect)
    
    # Filter the graph to only include nodes with paths to the target
    if filter_target_paths and target_node in G.nodes():
        # First, find all nodes that have a path to the target
        nodes_with_path_to_target = set()
        
        for node in G.nodes():
            try:
                if node != target_node and nx.has_path(G, node, target_node):
                    nodes_with_path_to_target.add(node)
            except nx.NetworkXNoPath:
                continue
        
        # Add the target node itself
        nodes_with_path_to_target.add(target_node)
        
        # Create a subgraph with only these nodes
        G = G.subgraph(nodes_with_path_to_target).copy()
    
    # Identify the target node and direct causes
    direct_causes = []
    if target_node in G.nodes():
        direct_causes = [n for n in G.predecessors(target_node)]
    
    other_nodes = [n for n in G.nodes() if n != target_node and n not in direct_causes]
    
    # Create node groups for easier styling
    node_groups = {
        'target': [target_node] if target_node in G.nodes() else [],
        'direct_causes': direct_causes,
        'other_nodes': other_nodes
    }
    
    # Generate layout based on the specified algorithm
    if layout_type == "kamada_kawai":
        try:
            pos = nx.kamada_kawai_layout(G, scale=2.0)
        except:
            # Fall back to spring layout if Kamada-Kawai fails
            print("Kamada-Kawai layout failed, falling back to spring layout.")
            pos = nx.spring_layout(G, k=0.5, iterations=100, seed=layout_seed)
    elif layout_type == "spring":
        pos = nx.spring_layout(G, k=0.5, iterations=100, seed=layout_seed)
    elif layout_type == "circular":
        pos = nx.circular_layout(G, scale=2.0)
    elif layout_type == "shell":
        # Create shells: target, direct causes, and other nodes
        shells = [
            node_groups['target'],
            node_groups['direct_causes'],
            node_groups['other_nodes']
        ]
        shells = [s for s in shells if s]  # Remove empty shells
        pos = nx.shell_layout(G, shells, scale=2.0)
    else:
        # Default to spring layout
        pos = nx.spring_layout(G, k=0.5, iterations=100, seed=layout_seed)
    
    # If target node exists, adjust its position to be at the bottom center
    if target_node in G.nodes():
        pos[target_node] = np.array([0.5, -0.8])
    
    # Get weight range for normalization
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    min_weight = min(weights) if weights else 0
    max_weight = max(weights) if weights else 0
    abs_max_weight = max(abs(min_weight), abs(max_weight)) if weights else 1
    
    # Create figure
    plt.figure(figsize=figsize, facecolor='white')
    
    # Professional IEEE color scheme
    # Draw nodes with different styles for different types
    if node_groups['target']:
        nx.draw_networkx_nodes(G, pos, 
                             nodelist=node_groups['target'], 
                             node_color='#4472C4',  # IEEE blue
                             node_size=4000 * node_size_factor,
                             alpha=0.9,
                             node_shape='s',  # Square for target
                             edgecolors='#2F528F')  # Darker blue outline
    
    if node_groups['direct_causes']:
        nx.draw_networkx_nodes(G, pos, 
                             nodelist=node_groups['direct_causes'], 
                             node_color='#70AD47',  # Professional green
                             node_size=3000 * node_size_factor,
                             alpha=0.8,
                             node_shape='o',  # Circle for direct causes
                             edgecolors='#507E32')  # Darker green outline
    
    if node_groups['other_nodes']:
        nx.draw_networkx_nodes(G, pos, 
                             nodelist=node_groups['other_nodes'], 
                             node_color='#A5A5A5',  # Professional gray
                             node_size=2000 * node_size_factor,
                             alpha=0.7,
                             node_shape='o',  # Circle for other nodes
                             edgecolors='#7F7F7F')  # Darker gray outline
    
    # Create a custom professional colormap for the edges
    edge_cmap = LinearSegmentedColormap.from_list(
        "professional", 
        [(0, '#2F5597'),    # Dark blue for negative
         (0.5, '#E7E6E6'),  # Light gray for neutral
         (1.0, '#C55A11')], # Dark orange for positive
        N=100
    )
    
    # Draw edges with unified color scale
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        
        # Normalize weight for color intensity
        norm_weight = (weight / abs_max_weight * 0.5) + 0.5
        
        # Determine edge width based on absolute weight
        width = (1.0 + 4 * (abs(weight) / abs_max_weight)) * edge_width_factor
        
        # Draw the edge
        nx.draw_networkx_edges(G, pos, 
                              edgelist=[(u, v)], 
                              width=width, 
                              alpha=0.8,
                              edge_color=[edge_cmap(norm_weight)],
                              arrows=True,
                              arrowsize=20,  # Professional sized arrowheads
                              arrowstyle='-|>',
                              connectionstyle=f'arc3,rad={edge_curvature}')
    
    # Add edge labels for direct effects to target with adjusted positions for curved edges
    if target_node in G.nodes():
        direct_edges = [(u, v) for u, v in G.edges() if v == target_node]
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in direct_edges}
        
        if edge_curvature == 0:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, font_family='serif')
        else:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, label_pos=0.4, font_family='serif')
    
    # Draw labels with appropriate font sizes
    if node_groups['target']:
        nx.draw_networkx_labels(G, pos, 
                               labels={n: n for n in node_groups['target']}, 
                               font_size=12, 
                               font_weight='bold',
                               font_family='serif')
    
    if node_groups['direct_causes']:
        nx.draw_networkx_labels(G, pos, 
                               labels={n: n for n in node_groups['direct_causes']}, 
                               font_size=10,
                               font_family='serif')
    
    if node_groups['other_nodes']:
        nx.draw_networkx_labels(G, pos, 
                               labels={n: n for n in node_groups['other_nodes']}, 
                               font_size=8,
                               font_family='serif')
    
    # Add a professional legend for node types
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#4472C4', markersize=10, 
                 label=f'Target ({target_node})') if node_groups['target'] else None,
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#70AD47', markersize=8, 
                 label='Direct Causal Factors') if node_groups['direct_causes'] else None,
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#A5A5A5', markersize=8, 
                 label='Indirect Factors') if node_groups['other_nodes'] else None,
        plt.Line2D([0], [0], color='#C55A11', lw=2, 
                  label='Positive Causal Effect'),
        plt.Line2D([0], [0], color='#2F5597', lw=2, 
                  label='Negative Causal Effect'),
    ]
    
    # Filter out None values
    legend_elements = [e for e in legend_elements if e is not None]
    
    if legend_elements:
        plt.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9, 
                  edgecolor='#D9D9D9')
    
    # Add title
    # plt.title('Causation Network Analysis of Bacterial Species and\nMetabolites in IBS', 
    #          fontsize=14, font='serif', fontweight='bold', pad=20)
    plt.axis('off')
    
    # Get graph statistics
    direct_effects = []
    if target_node in G.nodes():
        direct_effects = [(u, G[u][target_node]['weight']) for u in G.predecessors(target_node)]
        direct_effects.sort(key=lambda x: abs(x[1]), reverse=True)
    
    stats = {
        'total_nodes': G.number_of_nodes(),
        'total_edges': G.number_of_edges(),
        'direct_causes_count': len(direct_causes),
        'max_positive_effect': max(weights) if weights else 0,
        'max_negative_effect': min(weights) if weights else 0,
        'direct_effects': direct_effects
    }
    
    # Add summary statistics as text with a professional look
    plt.figtext(0.01, 0.01, 
               f"Graph Statistics:\n"
               f"Total Nodes: {stats['total_nodes']}\n"
               f"Total Edges: {stats['total_edges']}\n"
               f"Direct Causes: {stats['direct_causes_count']}\n"
               f"Max Positive Effect: {stats['max_positive_effect']:.3f}\n"
               f"Max Negative Effect: {stats['max_negative_effect']:.3f}", 
               fontsize=8, 
               font='serif',
               bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5', 
                       edgecolor='#D9D9D9', linewidth=0.5))
    
    plt.tight_layout()
    
    # Print nodes directly affecting the target
    if direct_effects:
        print(f"Nodes directly affecting {target_node} (sorted by effect strength):")
        for node, effect in direct_effects:
            print(f"{node}: {effect:.4f}")
    
    return G, stats

def get_causal_paths(G, source, target, max_length=None):
    """
    Find all causal paths from a source node to a target node
    
    Parameters:
    -----------
    G : networkx.DiGraph
        The graph to analyze
    source : str
        Source node
    target : str
        Target node
    max_length : int, optional
        Maximum path length to consider
        
    Returns:
    --------
    list of lists
        All paths from source to target
    """
    all_paths = []
    
    try:
        if nx.has_path(G, source, target):
            if max_length:
                all_paths = list(nx.all_simple_paths(G, source, target, cutoff=max_length))
            else:
                all_paths = list(nx.all_simple_paths(G, source, target))
    except nx.NetworkXNoPath:
        pass
    
    return [list(path) for path in all_paths]


def print_causal_path_strengths(G, source, target):
    """
    Print all causal paths from source to target with the strength of each link
    
    Parameters:
    -----------
    G : networkx.DiGraph
        The graph to analyze
    source : str
        Source node
    target : str
        Target node
    """
    paths = get_causal_paths(G, source, target)
    
    if not paths:
        print(f"No path found from {source} to {target}")
        return
    
    print(f"Causal paths from {source} to {target}:")
    
    for i, path in enumerate(paths):
        print(f"\nPath {i+1}:")
        total_effect = 1.0  # Start with 1 to later multiply all effects
        
        for j in range(len(path)-1):
            from_node = path[j]
            to_node = path[j+1]
            effect = G[from_node][to_node]['weight']
            
            print(f"  {from_node} -> {to_node}: {effect:.4f}")
            total_effect *= effect
        
        print(f"  Cumulative multiplicative effect: {total_effect:.4f}")