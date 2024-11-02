library(readxl)
library(pcalg)
library(igraph)

# Load and preprocess the dataset
dataset <- read_excel("C:/Users/snorl/Desktop/FYP/dataset/data_full_predicted_probabilities.xlsx")

X_columns <- c('xylose', 'xanthosine', 'valylglutamine', 'valine betaine', 
               'ursodeoxycholate sulfate (1)', 'uracil', 'tyrosine', 
               'tryptophylglycine', "trigonelline (N'-methylnicotinate)", 
               'tricarballylate', 'thymine', 'threonine', 'thiamin (Vitamin B1)', 
               'theobromine', 'syringic acid', 'succinimide', 'succinate', 
               'stearate (18:0)', 'stachydrine', 'sphingosine', 'serotonin', 
               'serine', 'salicylate', 'saccharin', 'ribulose/xylulose', 
               'riboflavin (Vitamin B2)', 'ribitol', 'quinolinate', 'quinate', 
               'pyroglutamine*','Prob_Class_1')

X_raw <- dataset[, X_columns]
X <- scale(X_raw)

# Ensure all columns in X are numeric for correlation computation
if (!all(sapply(X, is.numeric))) {
  stop("All columns in X must be numeric for correlation computation.")
}

# Set up the sufficient statistics and parameters for FCI
suffStat <- list(C = cor(X), n = nrow(X))
alpha <- 0.05  # Standard alpha for significance
var_names <- colnames(X)

# Run the FCI algorithm
fci_result <- fci(
  suffStat = suffStat,
  indepTest = gaussCItest,
  alpha = alpha,
  labels = var_names,
  verbose = TRUE,
  skel.method = "stable.fast",
  m.max = 3, 
  numCores = 4
)

# Convert the adjacency matrix to an igraph object
amat <- fci_result@amat  # Retrieve the adjacency matrix
graph_fci <- graph_from_adjacency_matrix(amat, mode = "directed", diag = FALSE)

plot(graph_fci, main = "FCI_graph")

# Quantify edge strengths
# Here, we compute edge strengths using conditional independence p-values
edge_strengths <- matrix(NA, ncol = length(var_names), nrow = length(var_names))
for (i in 1:length(var_names)) {
  for (j in 1:length(var_names)) {
    if (amat[i, j] != 0) {  # Only if there's an edge
      test_result <- gaussCItest(x = i, y = j, S = NULL, suffStat = suffStat)
      edge_strengths[i, j] <- -log10(test_result)  # Store p-value as edge strength
    }
  }
}
print(edge_strengths)
# Find all directed paths to "Prob_Class_1"
target_node <- which(var_names == "Prob_Class_1")
all_paths_to_target <- list()

for (node in V(graph_fci)) {
  if (node != target_node) {  # Exclude the target node itself
    paths <- all_simple_paths(graph_fci, from = node, to = target_node, mode = "out")
    if (length(paths) > 0) {
      all_paths_to_target[[var_names[node]]] <- paths
    }
  }
}














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
        all_paths[[node]] <- paths
      }
    }
  }
  return(all_paths)
}

# Get all paths to the target node
all_paths_to_target <- find_all_paths_to_target(graph_fci, target_node)

# Collect all unique nodes and edges that are part of the paths
unique_nodes <- unique(unlist(lapply(all_paths_to_target, function(paths) {
  unique(unlist(paths))
})))

# Create a subgraph using these nodes
subgraph <- induced_subgraph(graph_fci, vids = unique_nodes)

# Simplify the subgraph to remove multiple edges and loops (if any)
simplified_subgraph <- simplify(subgraph, remove.multiple = TRUE, remove.loops = TRUE)
plot(simplified_subgraph, main = "FCI Graph")
