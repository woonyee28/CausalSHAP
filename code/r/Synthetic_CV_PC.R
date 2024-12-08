# Create toy healthcare data with strong causal relationships
set.seed(123)
n <- 1000

# Layer 1: Root causes 
age <- rnorm(n, mean=65, sd=10)
age <- pmax(pmin(age, 85), 45)

diet_score <- rnorm(n, mean=50, sd=15)
diet_score <- pmax(pmin(diet_score, 100), 0)

exercise_level <- rnorm(n, mean=6, sd=2)
exercise_level <- pmax(pmin(exercise_level, 10), 0)

# Layer 2: Features with stronger dependencies
bmi_mean <- 25 + 0.7*scale(age) + 0.8*scale(diet_score)  
bmi <- rnorm(n, mean=bmi_mean, sd=1)  
bmi <- pmax(pmin(bmi, 40), 18.5)

fitness_mean <- 7 + 0.7*scale(age) + 0.8*scale(exercise_level)  
fitness <- rnorm(n, mean=fitness_mean, sd=1)  
fitness <- pmax(pmin(fitness, 14), 0)

# Layer 3: Outcome with stronger dependencies
cv_risk_mean <- 0.8*scale(age) + 0.8*scale(bmi) + 0.8*scale(fitness)  
cv_risk <- rnorm(n, mean=cv_risk_mean, sd=0.2)  

# Combine into data frame
data <- data.frame(
  age = age,
  diet_score = diet_score,
  exercise_level = exercise_level,
  bmi = bmi,
  fitness = fitness,
  cv_risk = cv_risk
)

write.csv(data, "Synthetic_CV.csv", row.names = FALSE)

# Convert data to matrix and scale
X <- as.matrix(data)
X_scaled <- scale(X)

# PC Algorithm
suffStat <- list(C = cor(X_scaled), n = nrow(X_scaled))
alpha <- 0.05
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

