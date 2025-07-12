library(readxl)
library(pcalg)
library(igraph)
library(graph)

# ------------------------- Data Loading and Preparation -------------------------
dataset <- read_xlsx("C:/Users/snorl/Desktop/FYP/dataset/IBS_species/Predicted_Probabilites_PRJNA789106_genus_pivoted.xlsx")

X_columns <- c('Collinsella', 'Bacteroides' ,'Ruthenibacterium', 'Lactobacillus',
               'Faecalibacterium', 'Prevotella', 'Akkermansia', 'Oscillibacter',
               'Anaerotruncus' ,'Odoribacter', 'Lachnoclostridium', 'Butyricimonas',
               'Porphyromonas' ,'Hungatella' ,'Flavonifractor', 'Alistipes',
               'Parabacteroides', 'Anaerobutyricum' ,'Dorea', 'Bifidobacterium',
               'Fusicatenibacter', 'Leptotrichia', 'Gemella', 'Intestinimonas',
               'Fusobacterium', 'Coprococcus' ,'Tyzzerella', 'Eisenbergiella',
               'Anaerostipes', 'Paraprevotella','IBS')

X_raw <- dataset[, X_columns]
X <- scale(X_raw)
Y <- dataset[["IBS"]]

if (!all(sapply(X, is.numeric))) {
  stop("All columns in X must be numeric for correlation computation.")
}

# ------------------------- PC Algorithm -------------------------

suffStat <- list(C = cor(X), n = nrow(X)) 
alpha <- 0.06
var_names <- colnames(X)

pc_result <- pc(
  suffStat = suffStat,
  indepTest = gaussCItest,
  alpha = alpha,
  labels = var_names,
  verbose = FALSE,
  skel.method = "stable.fast"
)

nodeNames <- nodes(pc_result@graph)
nodeAttrs <- list(fontsize = setNames(rep(25, length(nodeNames)), nodeNames))
plot(pc_result@graph, nodeAttrs = nodeAttrs)

adj_matrix <- as(pc_result@graph, "matrix")

get_edges <- function(adj_matrix) {
  edges <- which(adj_matrix != 0, arr.ind = TRUE)
  edge_list <- data.frame(
    from = rownames(adj_matrix)[edges[, 1]],
    to = colnames(adj_matrix)[edges[, 2]]
  )
  return(edge_list)
}

# ------------------------- IDA Algorithm -------------------------

ida_results_list <- list()
edge_list <- get_edges(adj_matrix)

for (i in 1:nrow(edge_list)) {
  x <- edge_list[i, "from"]
  y <- edge_list[i, "to"]
  if (x != y) {  
    x_pos <- match(x, var_names)
    y_pos <- match(y, var_names)
    result <- pcalg::ida(
      y.pos = y_pos,
      x.pos = x_pos,
      mcov = cov(X),
      graphEst = pc_result@graph,
      method = "local",
      type = "cpdag"
    )
    ida_results_list[[paste(x, "->", y, sep = "")]] <- result
  }
}

ida_results_df <- do.call(rbind, lapply(names(ida_results_list), function(name) {
  data.frame(
    Pair = name,
    Causal_Effect = I(ida_results_list[[name]])
  )
}))
# print(ida_results_df)

ida_results_mean_df <- do.call(rbind, lapply(names(ida_results_list), function(name) {
  causal_effects <- ida_results_list[[name]]
  mean_causal_effect <- mean(unlist(causal_effects), na.rm = TRUE)
  data.frame(
    Pair = name,
    Mean_Causal_Effect = mean_causal_effect,
    stringsAsFactors = FALSE
  )
}))

# ------------------------- Print Causal Edges and Mean Causal Strength -------------------------

build_graph_from_ida <- function(ida_results_df, var_names) {
  g <- make_empty_graph(n = length(var_names), directed = TRUE)
  V(g)$name <- var_names
  for (i in 1:nrow(ida_results_df)) {
    pair <- unlist(strsplit(as.character(ida_results_df$Pair[i]), "->"))
    x <- pair[1]
    y <- pair[2]
    g <- add_edges(g, c(x, y))
  }
  g <- simplify(g)
  return(g)
}

g <- build_graph_from_ida(ida_results_mean_df, var_names)
plot(g, main = "Reconstructed DAG from IDA Results", vertex.size = 8, vertex.label.cex = 0.7)

# -------------------- Find Paths to Target --------------------

find_all_paths_to_target <- function(graph, target) {
  all_paths <- list()
  for (node in V(graph)$name) {
    if (node != target) {
      paths <- all_simple_paths(graph, from = node, to = target, mode = "out")
      if (length(paths) > 0) {
        all_paths[[node]] <- lapply(paths, function(p) V(graph)[p]$name)
      }
    }
  }
  return(all_paths)
}

create_subgraph_for_target <- function(graph, all_paths) {
  unique_nodes <- unique(unlist(lapply(all_paths, unlist)))
  
  # Get all edges in the original graph
  all_edges <- as_edgelist(graph)
  
  # Filter edges to only include those between the unique nodes
  filtered_edges <- all_edges[all_edges[,1] %in% unique_nodes & all_edges[,2] %in% unique_nodes, , drop = FALSE]
  
  # Create new graph from filtered edges
  if(nrow(filtered_edges) > 0) {
    subgraph <- graph_from_edgelist(filtered_edges, directed = TRUE)
  } else {
    subgraph <- make_empty_graph(directed = TRUE)
  }
  
  return(subgraph)
}

print_paths_to_target <- function(all_paths, target_node) {
  for (start_node in names(all_paths)) {
    cat(sprintf("Paths from %s to %s:\n", start_node, target_node))
    for (path in all_paths[[start_node]]) {
      cat(paste(path, collapse = " -> "), "\n")
    }
    cat("\n")
  }
}

target_node <- "IBS"
all_paths_to_target <- find_all_paths_to_target(g, target_node)
# print_paths_to_target(all_paths_to_target, target_node)

simplified_subgraph <- create_subgraph_for_target(g, all_paths_to_target)
plot(simplified_subgraph,
     layout = layout_with_fr(simplified_subgraph),
     vertex.size = 10,
     vertex.label.cex = 0.8,
     edge.arrow.size = 0.5,
     main = "Causal Paths Leading to Prob_Class_1")

# -------------------- Extract Causal Effects for Target Paths --------------------
get_causal_effects_for_target <- function(subgraph, ida_results_mean_df) {
  subgraph_edges <- as_edgelist(subgraph)
  subgraph_edge_pairs <- apply(subgraph_edges, 1, function(edge) {
    paste(edge[1], "->", edge[2], sep = "")
  })
  
  filtered_effects <- ida_results_mean_df[ida_results_mean_df$Pair %in% subgraph_edge_pairs, ]
  filtered_effects <- filtered_effects[order(abs(filtered_effects$Mean_Causal_Effect), decreasing = TRUE), ]
  return(filtered_effects)
}

target_causal_effects <- get_causal_effects_for_target(simplified_subgraph, ida_results_mean_df)
cat("Edges with paths to", target_node, "and their causal strengths:\n\n")
print(target_causal_effects)



# -------------------- Visualization --------------------

# Load required libraries
library(igraph)
library(ggraph)
library(tidyverse)
library(gridExtra)
# Function to create and visualize causal network with fixed edge visibility
plot_causal_network <- function(causal_effects_df, target_node = "IBS", 
                                title = "Causal Network Leading to Target",
                                layout_type = "fr") {
  
  # Parse the causal effects data
  edges_df <- causal_effects_df %>%
    separate(Pair, into = c("from", "to"), sep = "->", remove = FALSE) %>%
    rename(weight = Mean_Causal_Effect) %>%
    mutate(
      abs_weight = abs(weight),
      edge_type = ifelse(weight > 0, "positive", "negative"),
      weight_category = cut(abs_weight, 
                            breaks = c(0, 0.1, 0.3, 0.5, Inf),
                            labels = c("weak", "moderate", "strong", "very_strong"))
    )
  
  # Create nodes dataframe
  all_nodes <- unique(c(edges_df$from, edges_df$to))
  nodes_df <- data.frame(
    name = all_nodes,
    is_target = all_nodes == target_node,
    stringsAsFactors = FALSE
  )
  
  print(edges_df)
  print(nodes_df)
  
  # Clean the data - remove any hidden characters/whitespace and ensure character type
  edges_df$from <- as.character(trimws(edges_df$from))
  edges_df$to <- as.character(trimws(edges_df$to))
  nodes_df$name <- as.character(trimws(nodes_df$name))
  
  # Verify data types
  cat("Edge data types - from:", class(edges_df$from), "to:", class(edges_df$to), "\n")
  cat("Node data type - name:", class(nodes_df$name), "\n")
  
  # Double check for exact matches
  edge_nodes <- unique(c(edges_df$from, edges_df$to))
  missing_in_vertices <- setdiff(edge_nodes, nodes_df$name)
  if(length(missing_in_vertices) > 0) {
    cat("Still missing nodes:\n")
    print(missing_in_vertices)
    # Add them manually
    additional_nodes <- data.frame(
      name = missing_in_vertices,
      is_target = missing_in_vertices == target_node,
      stringsAsFactors = FALSE
    )
    nodes_df <- rbind(nodes_df, additional_nodes)
  }
  
  # Create igraph object
  edge_matrix <- as.matrix(edges_df[, c("from", "to")])
  
  # Create graph from edge list (this method is more reliable)
  g <- graph_from_edgelist(edge_matrix, directed = TRUE)
  
  # Add edge attributes
  E(g)$weight <- edges_df$weight
  E(g)$abs_weight <- edges_df$abs_weight
  E(g)$edge_type <- edges_df$edge_type
  E(g)$weight_category <- edges_df$weight_category
  
  # Add vertex attributes
  V(g)$is_target <- V(g)$name == target_node
  
  # Create the plot using ggraph
  p <- ggraph(g, layout = layout_type) +
    # Add edges with different styles based on effect direction and strength
    # FIXED: Removed edge_alpha mapping to make edges always visible
    geom_edge_link(aes(
      edge_width = abs_weight,
      edge_color = weight,
      linetype = edge_type
    ), 
    alpha = 0.8,  # Fixed alpha value for all edges
    arrow = arrow(length = unit(3, 'mm'), type = "closed"),
    start_cap = circle(3, 'mm'),
    end_cap = circle(3, 'mm'),
    show.legend = TRUE) +
    
    # Style the edges
    scale_edge_width(name = "Effect Strength", range = c(0.5, 2.5)) +  # Increased minimum width
    scale_edge_color_gradient2(
      name = "Effect Direction",
      low = "red", mid = "grey", high = "blue",
      midpoint = 0,
      guide = guide_edge_colorbar()
    ) +
    scale_edge_linetype_manual(
      name = "Effect Type",
      values = c("positive" = "solid", "negative" = "dashed")
    ) +
    
    # Add nodes with different colors for target vs other nodes
    geom_node_point(aes(
      color = is_target,
      shape = is_target
    ), size = 6) +  # Fixed size for all nodes
    
    # Style the nodes
    scale_color_manual(
      name = "Node Type",
      values = c("TRUE" = "red", "FALSE" = "lightblue"),
      labels = c("TRUE" = "Target", "FALSE" = "Feature")
    ) +
    scale_shape_manual(
      name = "Node Type",
      values = c("TRUE" = 17, "FALSE" = 16),  # Triangle for target, circle for others
      labels = c("TRUE" = "Target", "FALSE" = "Feature")
    ) +
    
    # Add node labels
    geom_node_text(aes(label = name), 
                   repel = TRUE, 
                   size = 3.5,  # Slightly larger text
                   max.overlaps = 50,
                   box.padding = 0.3) +
    
    # Customize the plot
    labs(title = title) +
    theme_void() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      legend.position = "right",
      legend.box = "vertical",
      legend.title = element_text(size = 10, face = "bold"),
      legend.text = element_text(size = 8)
    ) +
    
    # Arrange legends (removed node degree and edge_alpha guides)
    guides(
      edge_width = guide_legend(order = 1),
      edge_color = guide_edge_colorbar(order = 2),
      edge_linetype = guide_legend(order = 3),
      color = guide_legend(order = 4),
      shape = guide_legend(order = 5)
    )
  
  return(p)
}

# Alternative version with manual alpha scaling if you want to keep magnitude-based transparency
plot_causal_network_with_scaled_alpha <- function(causal_effects_df, target_node = "IBS", 
                                                  title = "Causal Network Leading to Target",
                                                  layout_type = "fr") {
  
  # Parse the causal effects data
  edges_df <- causal_effects_df %>%
    separate(Pair, into = c("from", "to"), sep = "->", remove = FALSE) %>%
    rename(weight = Mean_Causal_Effect) %>%
    mutate(
      abs_weight = abs(weight),
      edge_type = ifelse(weight > 0, "positive", "negative"),
      # Scale alpha manually to ensure visibility (min 0.5, max 1.0)
      scaled_alpha = pmax(0.5, pmin(1.0, 0.5 + (abs_weight / max(abs_weight, na.rm = TRUE)) * 0.5)),
      weight_category = cut(abs_weight, 
                            breaks = c(0, 0.1, 0.3, 0.5, Inf),
                            labels = c("weak", "moderate", "strong", "very_strong"))
    )
  
  # Create nodes dataframe
  all_nodes <- unique(c(edges_df$from, edges_df$to))
  nodes_df <- data.frame(
    name = all_nodes,
    is_target = all_nodes == target_node,
    stringsAsFactors = FALSE
  )
  
  # Clean the data
  edges_df$from <- as.character(trimws(edges_df$from))
  edges_df$to <- as.character(trimws(edges_df$to))
  nodes_df$name <- as.character(trimws(nodes_df$name))
  
  # Handle missing nodes
  edge_nodes <- unique(c(edges_df$from, edges_df$to))
  missing_in_vertices <- setdiff(edge_nodes, nodes_df$name)
  if(length(missing_in_vertices) > 0) {
    additional_nodes <- data.frame(
      name = missing_in_vertices,
      is_target = missing_in_vertices == target_node,
      stringsAsFactors = FALSE
    )
    nodes_df <- rbind(nodes_df, additional_nodes)
  }
  
  # Create igraph object
  edge_matrix <- as.matrix(edges_df[, c("from", "to")])
  g <- graph_from_edgelist(edge_matrix, directed = TRUE)
  
  # Add edge attributes
  E(g)$weight <- edges_df$weight
  E(g)$abs_weight <- edges_df$abs_weight
  E(g)$edge_type <- edges_df$edge_type
  E(g)$scaled_alpha <- edges_df$scaled_alpha
  
  # Add vertex attributes
  V(g)$is_target <- V(g)$name == target_node
  
  # Create the plot using ggraph
  p <- ggraph(g, layout = layout_type) +
    # Add edges with scaled alpha that ensures visibility
    geom_edge_link(aes(
      edge_width = abs_weight,
      edge_color = weight,
      edge_alpha = scaled_alpha,  # Use pre-scaled alpha
      linetype = edge_type
    ), 
    arrow = arrow(length = unit(3, 'mm'), type = "closed"),
    start_cap = circle(3, 'mm'),
    end_cap = circle(3, 'mm'),
    show.legend = TRUE) +
    
    # Style the edges
    scale_edge_width(name = "Effect Strength", range = c(0.5, 2.5)) +
    scale_edge_color_gradient2(
      name = "Effect Direction",
      low = "red", mid = "grey", high = "blue",
      midpoint = 0,
      guide = guide_edge_colorbar()
    ) +
    scale_edge_alpha_identity(name = "Effect Magnitude") +  # Use identity scale for pre-scaled alpha
    scale_edge_linetype_manual(
      name = "Effect Type",
      values = c("positive" = "solid", "negative" = "dashed")
    ) +
    
    # Add nodes
    geom_node_point(aes(
      color = is_target,
      shape = is_target
    ), size = 6) +  # Fixed size for all nodes
    
    # Style the nodes
    scale_color_manual(
      name = "Node Type",
      values = c("TRUE" = "red", "FALSE" = "lightblue"),
      labels = c("TRUE" = "Target", "FALSE" = "Feature")
    ) +
    scale_shape_manual(
      name = "Node Type",
      values = c("TRUE" = 17, "FALSE" = 16),
      labels = c("TRUE" = "Target", "FALSE" = "Feature")
    ) +
    
    # Add node labels
    geom_node_text(aes(label = name), 
                   repel = TRUE, 
                   size = 3.5,
                   max.overlaps = 50,
                   box.padding = 0.3,
                   family = "Times New Roman") +
    
    # Customize the plot
    labs(title = title) +
    theme_void(base_family = "Times New Roman") +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      legend.position = "right",
      legend.box = "vertical",
      legend.title = element_text(size = 10, face = "bold", family = "Times New Roman"),
      legend.text = element_text(size = 8, family = "Serif")
    )
  
  return(p)
}

# Usage - replace your existing function calls with these:

# Version 1: Fixed alpha (recommended for clarity)
causal_network_plot <- plot_causal_network(
  target_causal_effects, 
  target_node = "IBS",
  title = "Causal Network Leading to IBS Classification",
  layout_type = "dh"
)

# Version 2: Scaled alpha (if you want to keep magnitude-based transparency)
causal_network_plot_scaled <- plot_causal_network_with_scaled_alpha(
  target_causal_effects, 
  target_node = "IBS",
  title = "Causal Network Leading to IBS Classification",
  layout_type = "dh"
)

print(causal_network_plot)
print(causal_network_plot_scaled)


