# Load Libraries
library(DiagrammeR)

# Create Synthetic Data
set.seed(42)
n <- 1000
A <- rnorm(n, mean = 0, sd = 1)
epsilon_1 <- rnorm(n, mean = 0, sd = 1)
epsilon_2 <- rnorm(n, mean = 0, sd = 1)
epsilon_3 <- rnorm(n, mean = 0, sd = 1)
epsilon_4 <- rnorm(n, mean = 0, sd = 1)
B <- 2 * A + epsilon_1 
C <- -1.5 * A + epsilon_2
D <- 0.9 * B + 0.3 * C + epsilon_3
E <- 1.2 * D + epsilon_4
synthetic_data <- data.frame(
  A = A,
  B = B,
  C = C,
  D = D,
  E = E
)
head(synthetic_data)


# Define and Visualize DAG
dag_spec <- "
digraph synthetic_data_dag {
  rankdir=LR; # Left to Right layout
  
  # Define node styles
  node [shape = ellipse, style = filled, color = LightBlue];
  
  # Define nodes
  A [label = 'A'];
  B [label = 'B'];
  C [label = 'C'];
  D [label = 'D'];
  E [label = 'E'];
  
  # Define edges with labels indicating the relationships
  A -> B [label = '2 * A + ε1'];
  A -> C [label = '-1.5 * A + ε2'];
  B -> D [label = '0.9 * B'];
  C -> D [label = '0.3 * C'];
  D -> E [label = '1.2 * D + ε4'];
}
"
grViz(dag_spec)

# PC Algorithm
library(pcalg)
suffStat <- list(
  C = cor(synthetic_data),  
  n = nrow(synthetic_data) 
)
alpha <- 0.05
var_names <- colnames(synthetic_data)
pc_result <- pc(
  suffStat = suffStat,
  indepTest = gaussCItest,
  alpha = alpha,
  labels = var_names,
  verbose = FALSE
)
plot(pc_result@graph)
adj_matrix <- as(pc_result@graph, "matrix")

# IDA Algorithm
get_edges <- function(adj_matrix) {
  edges <- which(adj_matrix != 0, arr.ind = TRUE)
  edge_list <- data.frame(
    from = rownames(adj_matrix)[edges[, 1]],
    to = colnames(adj_matrix)[edges[, 2]]
  )
  return(edge_list)
}


ida_results_list <- list()
edge_list <- get_edges(adj_matrix)
print(edge_list)

# Correct position mapping
var_positions <- setNames(seq_along(var_names), var_names)

# Edge extraction in IDA
for (i in 1:nrow(edge_list)) {
  x <- edge_list[i, "from"]
  y <- edge_list[i, "to"]
  if (x != y) {  
    x_pos <- match(x, var_names)
    y_pos <- match(y, var_names)
    result <- pcalg::ida(
      y.pos = y_pos,
      x.pos = x_pos,
      mcov = cov(synthetic_data),
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
print(ida_results_df)

# Recontruct DAG
library(igraph)
g <- make_empty_graph(n = 5, directed = TRUE)
V(g)$name <- var_names
add_edges_based_on_ida <- function(df, threshold = 0) {
  for (i in 1:nrow(df)) {
    pair <- unlist(strsplit(as.character(df$Pair[i]), "->"))
    x <- pair[1]
    y <- pair[2]
    effect <- df$Causal_Effect[i]
    
    # Define a threshold to consider an effect as significant (optional)
    if (!is.na(effect) && (effect >= 0.5 || effect <=-0.5)) {
      g <<- add_edges(g, c(x, y))
    }
  }
}
add_edges_based_on_ida(ida_results_df)
g <- simplify(g, remove.multiple = TRUE, remove.loops = TRUE)
plot(g, main = "Reconstructed DAG from IDA Results")

# Print the mean effect with edges
ida_results_mean_df <- do.call(rbind, lapply(names(ida_results_list), function(name) {
  filtered_effects <- ida_results_list[[name]][abs(ida_results_list[[name]]) > 0.5]
  mean_causal_effect <- mean(filtered_effects, na.rm = TRUE)
  data.frame(
    Pair = name,
    Mean_Causal_Effect = mean_causal_effect
  )
}))
print(ida_results_mean_df)
