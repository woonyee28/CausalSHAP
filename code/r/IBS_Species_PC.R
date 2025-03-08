library(readxl)
library(pcalg)
library(igraph)

# ------------------------- Data Loading and Preparation -------------------------
dataset <- read_xlsx("C:/Users/snorl/Desktop/FYP/dataset/IBS_species/Predicted_Probabilites_PRJNA644520_class_pivoted.xlsx")

X_columns <- c('Clostridia', 'Erysipelotrichia' ,'Bacilli', 'Bacteroidia',
               'Gammaproteobacteria', 'Coriobacteriia' ,'Mammalia', 'Betaproteobacteria',
               'Actinobacteria', 'Alphaproteobacteria', 'Fusobacteriia', 'Chitinophagia',
               'Cytophagia' ,'Chlamydiia', 'Spirochaetia' ,'Epsilonproteobacteria',
               'Deltaproteobacteria', 'Magnoliopsida', 'Flavobacteriia', 'Methanobacteria',
               'Negativicutes','Prob_Class_1')

X_raw <- dataset[, X_columns]
X <- scale(X_raw)
Y <- dataset[["Prob_Class_1"]]

if (!all(sapply(X, is.numeric))) {
  stop("All columns in X must be numeric for correlation computation.")
}

# ------------------------- PC Algorithm -------------------------

suffStat <- list(C = cor(X), n = nrow(X)) 
alpha <- 0.10
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
  g <- simplify(g, remove.multiple = TRUE, remove.loops = TRUE)
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
  subgraph <- induced_subgraph(graph, vids = unique_nodes)
  simplified_subgraph <- simplify(subgraph, remove.multiple = TRUE, remove.loops = TRUE)
  return(simplified_subgraph)
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

target_node <- "Prob_Class_1"
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

