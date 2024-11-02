# Documentation: http://cran.nexr.com/web/packages/pcalg/vignettes/pcalgDoc.pdf
# Load necessary libraries
library(readxl)
library(pcalg)
library(igraph)

# Load the dataset
dataset <- read_excel("C:/Users/snorl/Desktop/FYP/dataset/data_full_predicted_probabilities.xlsx")

# Correct Group encoding

# Specify columns for X (features)
X_columns <- c('xylose', 'xanthosine', 'valylglutamine', 'valine betaine', 
               'ursodeoxycholate sulfate (1)', 'uracil', 'tyrosine', 
               'tryptophylglycine', "trigonelline (N'-methylnicotinate)", 
               'tricarballylate', 'thymine', 'threonine', 'thiamin (Vitamin B1)', 
               'theobromine', 'syringic acid', 'succinimide', 'succinate', 
               'stearate (18:0)', 'stachydrine', 'sphingosine', 'serotonin', 
               'serine', 'salicylate', 'saccharin', 'ribulose/xylulose', 
               'riboflavin (Vitamin B2)', 'ribitol', 'quinolinate', 'quinate', 
               'pyroglutamine*','Prob_Class_1')
 
# Select X and Y
X_raw <- dataset[, X_columns]
X <- scale(X_raw)
Y <- dataset[["Prob_Class_1"]]

# Check if all selected features are numeric
if (!all(sapply(X, is.numeric))) {
  stop("All columns in X must be numeric for correlation computation.")
}

# Print the first few rows of X and Y to confirm
head(X)
#head(Y)

# PC Algorithm
suffStat <- list(C = cor(X), n = nrow(X))  # Ensure X is numeric for correlation
alpha <- 0.05
var_names <- colnames(X)

pc_result <- pc(
  suffStat = suffStat,
  indepTest = gaussCItest,
  alpha = alpha,
  labels = var_names,
  verbose = FALSE,
  skel.method = "stable"
)

# Plot the resulting graph
nodeNames <- nodes(pc_result@graph)
nodeAttrs <- list(fontsize = setNames(rep(25, length(nodeNames)), nodeNames))
plot(pc_result@graph, nodeAttrs = nodeAttrs)

adj_matrix <- as(pc_result@graph, "matrix")

# IDA Algorithm: Extract edges
get_edges <- function(adj_matrix) {
  edges <- which(adj_matrix != 0, arr.ind = TRUE)
  edge_list <- data.frame(
    from = rownames(adj_matrix)[edges[, 1]],
    to = colnames(adj_matrix)[edges[, 2]]
  )
  return(edge_list)
}

# Apply IDA
ida_results_list <- list()
edge_list <- get_edges(adj_matrix)

# Loop through each edge and apply IDA
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
      type = "pdag"
    )
    ida_results_list[[paste(x, "->", y, sep = "")]] <- result
  }
}

# Convert IDA results to DataFrame
ida_results_df <- do.call(rbind, lapply(names(ida_results_list), function(name) {
  data.frame(
    Pair = name,
    Causal_Effect = I(ida_results_list[[name]])
  )
}))
print(ida_results_df)

# Reconstruct DAG using IDA Results
g <- make_empty_graph(n = length(var_names), directed = TRUE)
V(g)$name <- var_names

# Function to add edges based on IDA results
add_edges_based_on_ida <- function(df, threshold = 0.5) {
  for (i in 1:nrow(df)) {
    pair <- unlist(strsplit(as.character(df$Pair[i]), "->"))
    x <- pair[1]
    y <- pair[2]
    effect <- df$Causal_Effect[i]
    
    # Add edges if the effect is significant based on threshold
    if (!is.na(effect) && (effect >= 0.15 || effect <= -0.15)) {
      g <<- add_edges(g, c(x, y))
    }
  }
}

# Add edges to the graph based on IDA results

add_edges_based_on_ida(ida_results_df)

# Simplify the graph (remove loops and multiple edges)
g <- simplify(g, remove.multiple = TRUE, remove.loops = TRUE)

# Plot the reconstructed DAG
plot(g, main = "Reconstructed DAG from IDA Results")

# Print the mean causal effects for each edge
ida_results_mean_df <- do.call(rbind, lapply(names(ida_results_list), function(name) {
  filtered_effects <- ida_results_list[[name]][abs(ida_results_list[[name]]) > 0.1]
  mean_causal_effect <- mean(filtered_effects, na.rm = TRUE)
  data.frame(
    Pair = name,
    Mean_Causal_Effect = mean_causal_effect
  )
}))
print(ida_results_mean_df)


# To find all paths to Prob_Class_1
# Install the required package if not already installed
# install.packages("igraph")

# Ensure the graph 'g' is directed
g <- simplify(g, remove.multiple = TRUE, remove.loops = TRUE)

# Define the target node
target_node <- "Prob_Class_1"

# Find all paths leading to the target node
find_all_paths_to_target <- function(graph, target) {
  all_paths <- list()
  
  # Loop through all nodes except the target node
  for (node in V(graph)$name) {
    if (node != target) {
      # Use igraph's all_simple_paths function to find paths from each node to the target
      paths <- all_simple_paths(graph, from = node, to = target, mode = "out")
      if (length(paths) > 0) {
        # Store all paths found
        all_paths[[node]] <- lapply(paths, function(p) V(graph)[p]$name)
      }
    }
  }
  return(all_paths)
}

# Get all paths to the target node
all_paths_to_target <- find_all_paths_to_target(g, target_node)

# Print all paths to the target node
for (start_node in names(all_paths_to_target)) {
  cat(sprintf("Paths from %s to %s:\n", start_node, target_node))
  for (path in all_paths_to_target[[start_node]]) {
    cat(paste(path, collapse = " -> "), "\n")
  }
}

# Step 2: Create a simplified subgraph based on the identified paths
# Collect all unique nodes and edges that are part of the paths
unique_nodes <- unique(unlist(lapply(all_paths_to_target, unlist)))
print(unique_nodes)

# Create a subgraph using these nodes
subgraph <- induced_subgraph(g, vids = unique_nodes)

# Simplify the subgraph to remove multiple edges and loops (if any)
simplified_subgraph <- simplify(subgraph, remove.multiple = TRUE, remove.loops = TRUE)

# Plot the simplified subgraph
plot(simplified_subgraph, main = "PC Graph Leading to Prob_Class_1")

# Save or print the subgraph structure
cat("Nodes in the simplified subgraph:\n", V(simplified_subgraph)$name, "\n")
cat("Edges in the simplified subgraph:\n")
print(as_edgelist(simplified_subgraph))


