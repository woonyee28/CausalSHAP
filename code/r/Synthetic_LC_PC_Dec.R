library(ggplot2)
library(readxl)
library(pcalg)
library(igraph)
# Set the seed for reproducibility
set.seed(42)

# Generate the data
n_samples <- 1000

# Generate the first layer of input variables
smoking <- round(rnorm(n_samples, 5, 2))
stress <- round(rnorm(5,2))

# Generate the intermediate variables
drink_coffee <-  round(stress + 2 * smoking + rnorm(n_samples, 0, 1))

lung_cancer_risk <- round(smoking * 2 + 1.2 * stress + rnorm(n_samples, 0, 3) )

# Create the dataset
data <- data.frame(
  smoking = smoking,
  stress = stress,
  drink_coffee = drink_coffee,
  lung_cancer_risk = lung_cancer_risk
)

write.csv(data, "Synthetic_LC_Dec.csv", row.names = FALSE)

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