library(ggplot2)
# Set the seed for reproducibility
set.seed(42)

# Generate the data
n_samples <- 1000


# Generate the first layer of input variables
age <- round(runif(n_samples, 18, 80))
smoking <- round(rnorm(n_samples, 10, 5))
air_pollution_exposure <- round(runif(n_samples, 20, 70))
industrial_activity <- round(runif(n_samples, 10, 50))

# Generate the intermediate variables
air_quality_index <-  20 + 0.8 * air_pollution_exposure + 0.5 * industrial_activity + rnorm(n_samples, 0, 20)

# Generate the target variable (lung cancer risk) with non-linear effects
lung_cancer_risk_mean <- 0.001 * (age ^ 2) + 0.2 * smoking + 0.3 * air_quality_index 
lung_cancer_risk_std <- 2 + 0.1 * age + 0.05 * smoking + 0.05 * air_pollution_exposure
lung_cancer_risk <- lung_cancer_risk_mean + rnorm(n_samples, 0, lung_cancer_risk_std)

# Create the dataset
data <- data.frame(
  age = age,
  smoking = smoking,
  air_pollution_exposure = air_pollution_exposure,
  air_quality_index = air_quality_index,
  industrial_activity = industrial_activity,
  lung_cancer_risk = lung_cancer_risk
)

write.csv(data, "Synthetic_LC.csv", row.names = FALSE)

# Convert data to matrix and scale
X <- as.matrix(data)
X_scaled <- scale(X)

# PC Algorithm
suffStat <- list(C = cor(X_scaled), n = nrow(X_scaled))
alpha <- 0.1
var_names <- colnames(X_scaled)

pc_result <- pc(
  suffStat = suffStat,
  indepTest = gaussCItest,
  alpha = alpha,
  labels = var_names,
  verbose = FALSE,
  skel.method = "stable"
)

# Plot PC result
plot(pc_result@graph)

# Get adjacency matrix
adj_matrix <- as(pc_result@graph, "matrix")

# Function to get edges
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

# Loop through edges and apply IDA
for (i in 1:nrow(edge_list)) {
  x <- edge_list[i, "from"]
  y <- edge_list[i, "to"]
  if (x != y) {  
    x_pos <- match(x, var_names)
    y_pos <- match(y, var_names)
    result <- pcalg::ida(
      y.pos = y_pos,
      x.pos = x_pos,
      mcov = cov(X_scaled),
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

print("IDA Results:")
print(ida_results_df)

# Find paths to cv_risk
library(igraph)

# Create igraph object from PC results
g <- graph_from_adjacency_matrix(adj_matrix, mode="directed")

# Find all paths to cv_risk
target_node <- "cv_risk"

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

# Get and print paths
all_paths_to_target <- find_all_paths_to_target(g, target_node)

cat("\nPaths to cardiovascular risk:\n")
for (start_node in names(all_paths_to_target)) {
  cat(sprintf("From %s:\n", start_node))
  for (path in all_paths_to_target[[start_node]]) {
    cat(paste(path, collapse = " -> "), "\n")
  }
}

