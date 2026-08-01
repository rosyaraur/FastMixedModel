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