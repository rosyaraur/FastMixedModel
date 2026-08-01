library(Rcpp)
library(RcppEigen)
library(Matrix)
library(MASS)

# 1. Compile the pristine C++ backend
Rcpp::sourceCpp("CombinedMixedSolvercpp.cpp")

# 2. Define the R Wrapper (Named 'Fit_Mixed_Model' to prevent recursion)
Fit_Mixed_Model <- function(
    engine, X, Z, y, 
    K = NULL, A_inv = NULL,
    init_varE = 1.0, init_varU = 1.0,
    max_iter = 50, n_iter = 1000, burn_in = 200, tol = 1e-6
) {
  
  # Safe matrix imputation
  if (is.null(K) && is.null(A_inv)) {
    stop("Error: You must provide either 'K' or 'A_inv'.")
  }
  
  if (is.null(A_inv)) {
    # 'generalMatrix' bypasses the dgCMatrix infinite loop bug in the Matrix package
    tmp_A <- Matrix::Matrix(base::solve(K), sparse = TRUE)
    A_inv <- methods::as(tmp_A, "generalMatrix") 
  }
  if (is.null(K)) {
    K <- base::as.matrix(base::solve(A_inv))
  }
  
  # Strict base types for Rcpp
  X <- base::as.matrix(X)
  Z <- base::as.matrix(Z)
  y <- base::as.numeric(y)
  K <- base::as.matrix(K)
  if (!inherits(A_inv, "dgCMatrix")) {
    A_inv <- methods::as(methods::as(A_inv, "CsparseMatrix"), "generalMatrix")
  }
  
  # 3. Call the C++ backend
  res <- CombinedMixedSolvercpp(
    engine    = engine, 
    X         = X, 
    Z         = Z, 
    y         = y, 
    K         = K, 
    A_inv     = A_inv,
    init_varE = init_varE,
    init_varU = init_varU,
    max_iter  = max_iter,
    n_iter    = n_iter,
    burn_in   = burn_in,
    tol       = tol
  )
  
  res$engine <- engine
  class(res) <- "CombinedMixedSolver"
  return(res)
}

# =========================================================================
# Benchmark Execution
# =========================================================================
set.seed(42)
library(MASS)
library(Matrix)

# Simulate Dataset
n <- 1500
p <- 3
q <- 200

true_beta <- c(2.5, -2.0, 0.5)
true_varU <- 2.5
true_varE <- 2.0

X <- matrix(rnorm(n * p), n, p)
Z <- matrix(rbinom(n * q, 1, 0.05), n, q)

W <- matrix(rnorm(q * q), q, q)
K <- tcrossprod(W) / q
diag(K) <- diag(K) + 0.1 

u <- mvrnorm(1, mu = rep(0, q), Sigma = K * true_varU)
e <- rnorm(n, mean = 0, sd = sqrt(true_varE))
y <- X %*% true_beta + Z %*% u + e

# Pre-calculate sparse inverse once safely
A_inv_sparse <- methods::as(Matrix::Matrix(base::solve(K), sparse = TRUE), "generalMatrix")

engines <- c(
  "kernel_bglr", "block_mcmcglmm", "penalized_map_blme", "hmc_stan", 
  "reml_lme4", "moment_mbest", "spectral_rrblup", "ai_sommer", 
  "sparse_asreml", "nr_sas"
)

results <- data.frame(
  Engine = character(), Estimated_varE = numeric(),
  Estimated_varU = numeric(), Time_Seconds = numeric(),
  stringsAsFactors = FALSE
)

cat("Starting 10-Engine Benchmark...\n")

for (eng in engines) {
  cat(sprintf("Running %s... ", eng))
  
  start_time <- Sys.time()
  
  # Call the safely named wrapper
  fit <- Fit_Mixed_Model(
    engine = eng, X = X, Z = Z, y = y, K = K, A_inv = A_inv_sparse,
    init_varE = 1.0, init_varU = 1.0,
    max_iter = 100, n_iter = 2000, burn_in = 500, tol = 1e-6
  )
  
  end_time <- Sys.time()
  exec_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
  
  results <- rbind(results, data.frame(
    Engine = eng, Estimated_varE = round(fit$varE, 4),
    Estimated_varU = round(fit$varU, 4), Time_Seconds = round(exec_time, 4)
  ))
  
  cat("Done.\n")
}

print(results)

#' Extract Random Effects and Compute Engine Correlation Matrix
#'
#' @param fit_results Named list of fitted models from each engine.
#' @param true_u Numeric vector. The true simulated random effects (optional).
#' @return A list containing the combined effects matrix and the correlation matrix.
extract_engine_correlations <- function(fit_results, true_u = NULL) {
  # 1. Extract estimated random effects (u) from each engine
  effects_list <- lapply(fit_results, function(fit) {
    if (!is.null(fit$u)) return(as.vector(fit$u))
    if (!is.null(fit$post_u)) return(as.vector(fit$post_u))
    return(NULL)
  })
  
  # Remove nulls (in case an engine failed or didn't return u)
  effects_list <- effects_list[!sapply(effects_list, is.null)]
  
  # 2. Combine into a matrix (Rows = random effect levels, Columns = Engines)
  effects_mat <- do.call(cbind, effects_list)
  colnames(effects_mat) <- names(effects_list)
  
  # 3. Optionally prepend the true simulated effects for validation
  if (!is.null(true_u)) {
    effects_mat <- cbind(True_U = as.vector(true_u), effects_mat)
  }
  
  # 4. Compute pairwise Pearson correlation matrix
  cor_mat <- cor(effects_mat, use = "pairwise.complete.obs")
  
  return(list(
    effects_matrix = effects_mat,
    correlation_matrix = round(cor_mat, 4)
  ))
}

# Initialize a storage list for model fits
engine_fits <- list()

cat("Starting 10-Engine Benchmark & Storage...\n")

for (eng in engines) {
  cat(sprintf("Running %s... ", eng))
  
  fit <- Fit_Mixed_Model(
    engine = eng, X = X, Z = Z, y = y, K = K, A_inv = A_inv_sparse,
    init_varE = 1.0, init_varU = 1.0,
    max_iter = 100, n_iter = 2000, burn_in = 500, tol = 1e-6
  )
  
  # Store fit object named by engine
  engine_fits[[eng]] <- fit
  cat("Done.\n")
}

# Run the extraction and correlation function (including true 'u')
comparison <- extract_engine_correlations(engine_fits, true_u = u)

# View the correlation matrix comparing all engines and the true values
print(comparison$correlation_matrix)

library(ggplot2)
library(reshape2)

# Ensure you have already run the benchmark and have 'comparison$correlation_matrix'
cor_mat <- comparison$correlation_matrix

# Melt the correlation matrix into long format
melted_cor <- melt(cor_mat)
colnames(melted_cor) <- c("Engine1", "Engine2", "Correlation")

# Generate the heatmap plot
p_heatmap <- ggplot(melted_cor, aes(x = Engine1, y = Engine2, fill = Correlation)) +
  geom_tile(color = "white") +
  # Use a gradient color scale focused on the high-correlation range (0.90 to 1.00)
  scale_fill_gradient2(
    low = "#2b580c", mid = "#f7fcb9", high = "#e31a1c", 
    midpoint = 0.95, limits = c(0.90, 1.00), space = "Lab", 
    name = "Pearson\nCorrelation"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, face = "bold"),
    axis.text.y = element_text(face = "bold"),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    axis.ticks = element_blank(),
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5)
  ) +
  coord_fixed() +
  # Overlay numeric correlation values on each tile for clarity
  geom_text(aes(label = sprintf("%.3f", Correlation)), color = "black", size = 3) +
  labs(title = "Engine-to-Engine Random Effects Correlation Heatmap")

# Display the plot
print(p_heatmap)

# Animal model 
library(Rcpp)
library(RcppEigen)
library(Matrix)
library(MASS)

# 1. Ensure the C++ backend is compiled
# (Make sure 'CombinedMixedSolvercpp.cpp' is in your working directory)
Rcpp::sourceCpp("CombinedMixedSolvercpp.cpp")

# 2. Define the Safe R Wrapper 
Fit_Mixed_Model <- function(
    engine, X, Z, y, 
    K = NULL, A_inv = NULL,
    init_varE = 1.0, init_varU = 1.0,
    max_iter = 50, n_iter = 1000, burn_in = 200, tol = 1e-6
) {
  if (is.null(K) && is.null(A_inv)) {
    stop("Error: You must provide either 'K' or 'A_inv'.")
  }
  
  if (is.null(A_inv)) {
    tmp_A <- Matrix::Matrix(base::solve(K), sparse = TRUE)
    A_inv <- methods::as(tmp_A, "generalMatrix") 
  }
  if (is.null(K)) {
    K <- base::as.matrix(base::solve(A_inv))
  }
  
  X <- base::as.matrix(X)
  Z <- base::as.matrix(Z)
  y <- base::as.numeric(y)
  K <- base::as.matrix(K)
  if (!inherits(A_inv, "dgCMatrix")) {
    A_inv <- methods::as(methods::as(A_inv, "CsparseMatrix"), "generalMatrix")
  }
  
  res <- CombinedMixedSolvercpp(
    engine    = engine, X = X, Z = Z, y = y, 
    K = K, A_inv = A_inv,
    init_varE = init_varE, init_varU = init_varU,
    max_iter  = max_iter, n_iter  = n_iter, burn_in = burn_in, tol = tol
  )
  
  res$engine <- engine
  class(res) <- "CombinedMixedSolver"
  return(res)
}

# =========================================================================
# 3. Simulate Animal Model Dataset
# =========================================================================
set.seed(123)

n_animals <- 1500  # Number of individuals / animals
p <- 2            # Fixed effects (e.g., Intercept and Sex)
q <- n_animals    # Random genetic levels mapped 1-to-1 to animals

# True parameters
true_beta <- c(15.0, 3.2)
true_varU <- 3.5  # Additive genetic variance (sigma^2_u)
true_varE <- 1.8  # Residual variance (sigma^2_e)

# Fixed effects design matrix (X) and Sex covariate
X <- cbind(1, sample(c(0, 1), n_animals, replace = TRUE))
colnames(X) <- c("Intercept", "Sex")

# Random effects incidence matrix (Z) maps records 1-to-1 to animals
Z <- diag(n_animals)

# Generate a positive-definite Genomic Relationship / Kinship Matrix (K)
raw_markers <- matrix(rnorm(n_animals * 400), nrow = n_animals, ncol = 400)
K <- tcrossprod(raw_markers) / 400
diag(K) <- diag(K) + 0.05 # Ridge stabilization for positive definiteness

# Simulate true Breeding Values (u) and Residuals (e)
u <- mvrnorm(1, mu = rep(0, q), Sigma = K * true_varU)
e <- rnorm(n_animals, mean = 0, sd = sqrt(true_varE))

# Phenotypic response vector (y)
y <- as.vector(X %*% true_beta + Z %*% u + e)

# Pre-calculate the sparse inverse relationship matrix (A_inv)
A_inv_sparse <- methods::as(Matrix::Matrix(base::solve(K), sparse = TRUE), "generalMatrix")

# =========================================================================
# 4. Animal Model Benchmark Execution
# =========================================================================
# Select engines that natively support relationship matrices (K or A_inv)
animal_engines <- c(
  "kernel_bglr", "block_mcmcglmm", "penalized_map_blme", 
  "reml_lme4", "spectral_rrblup", "sparse_asreml", "nr_sas"
)

animal_results <- data.frame(
  Engine = character(), Estimated_varE = numeric(),
  Estimated_varU = numeric(), EBV_Correlation = numeric(),
  Time_Seconds = numeric(), stringsAsFactors = FALSE
)

cat("Starting Animal Model Benchmark...\n")
cat("True varE =", true_varE, "| True varU =", true_varU, "\n\n")

for (eng in animal_engines) {
  cat(sprintf("Running %s... ", eng))
  
  start_time <- Sys.time()
  
  fit <- Fit_Mixed_Model(
    engine = eng, X = X, Z = Z, y = y, 
    K = K, A_inv = A_inv_sparse,
    init_varE = 1.0, init_varU = 1.0,
    max_iter = 100, n_iter = 2000, burn_in = 500, tol = 1e-6
  )
  
  end_time <- Sys.time()
  exec_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
  
  # Extract estimated breeding values (u)
  est_u <- if (!is.null(fit$u)) fit$u else fit$post_u
  ebv_cor <- cor(u, as.vector(est_u))
  
  animal_results <- rbind(animal_results, data.frame(
    Engine = eng, 
    Estimated_varE = round(fit$varE, 4),
    Estimated_varU = round(fit$varU, 4), 
    EBV_Correlation = round(ebv_cor, 4),
    Time_Seconds = round(exec_time, 4)
  ))
  
  cat("Done.\n")
}

print(animal_results)

# Compute the EBV Correlation Matrix
# 1. Initialize a list storing True_U and each engine's estimated breeding values
ebv_list <- list(True_U = as.vector(u))

# 2. Re-collect or loop through the animal engines to extract est_u
for (eng in animal_engines) {
  cat(sprintf("Extracting EBVs for %s...\n", eng))
  
  fit <- Fit_Mixed_Model(
    engine = eng, X = X, Z = Z, y = y, 
    K = K, A_inv = A_inv_sparse,
    init_varE = 1.0, init_varU = 1.0,
    max_iter = 100, n_iter = 2000, burn_in = 500, tol = 1e-6
  )
  
  # Extract vector safely
  est_u <- if (!is.null(fit$u)) as.vector(fit$u) else as.vector(fit$post_u)
  ebv_list[[eng]] <- est_u
}

# 3. Combine into a matrix (Rows = animals, Columns = Engines + True_U)
ebv_matrix <- do.call(cbind, ebv_list)
colnames(ebv_matrix) <- names(ebv_list)

# 4. Compute the pairwise Pearson correlation matrix
ebv_cor_matrix <- cor(ebv_matrix, use = "pairwise.complete.obs")

# 5. Print the rounded correlation matrix
print(round(ebv_cor_matrix, 4))