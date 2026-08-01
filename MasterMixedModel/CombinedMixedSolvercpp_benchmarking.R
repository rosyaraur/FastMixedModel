library(Rcpp)
library(RcppEigen)
library(Matrix)
library(MASS)

# 1. Compile the pristine C++ backend
Rcpp::sourceCpp("CombinedMixedSolvercpp.cpp")

# 2. Define the R Wrapper (Named 'Fit_Mixed_Model' to prevent recursion)
# =========================================================================
# 1. COMPILE C++ BACKEND & DEFINE R WRAPPER
# =========================================================================
cat("Compiling C++ backend via RcppEigen...\n")
Rcpp::sourceCpp("CombinedMixedSolvercpp.cpp")

#' Safe R Wrapper for CombinedMixedSolvercpp
Fit_Mixed_Model <- function(
    engine, X, Z, y, 
    K = NULL, A_inv = NULL,
    init_varE = 1.0, init_varU = 1.0,
    max_iter = 50, n_iter = 2000, burn_in = 200, tol = 1e-6
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
# 2. DIAGNOSTIC REPORT & VISUALIZATION FUNCTION
# =========================================================================
#' Generate Visual Diagnostics and Iteration Summary Report
GenerateModelDiagnosticReport <- function(fit_object, y, X, Z, K = NULL) {
  engine_name <- fit_object$engine
  plots <- list()
  
  cat(sprintf("\n=== Diagnostic Report for Engine: %s ===\n", engine_name))
  
  # A. MCMC Diagnostics (Trace Plots for Gibbs engines)
  if (!is.null(fit_object$chains)) {
    cat("-> Generating MCMC convergence trace plots...\n")
    chains <- fit_object$chains
    
    df_varE <- data.frame(Iteration = 1:length(chains$varE), Value = chains$varE, Parameter = "Residual Variance (varE)")
    df_varU <- data.frame(Iteration = 1:length(chains$varU), Value = chains$varU, Parameter = "Genetic Variance (varU)")
    df_chains <- rbind(df_varE, df_varU)
    
    p_trace <- ggplot(df_chains, aes(x = Iteration, y = Value, color = Parameter)) +
      geom_line(alpha = 0.7) +
      facet_wrap(~ Parameter, scales = "free_y") +
      theme_minimal() +
      labs(title = sprintf("MCMC Trace Plots (%s)", engine_name), x = "Iteration", y = "Parameter Value") +
      theme(legend.position = "none", plot.title = element_text(face = "bold"))
    
    plots$mcmc_trace <- p_trace
  } 
  
  # B. Optimization Diagnostics (Profile Likelihood / Deviance Curve for REML/MAP)
  else {
    cat("-> Evaluating Profile Likelihood Surface for boundary analysis...\n")
    n <- length(y); p <- ncol(X); q <- ncol(Z)
    M <- cbind(X, Z); MtM <- crossprod(M); Mty <- as.vector(crossprod(M, y))
    
    A_inv_grid <- if (is.null(K)) Diagonal(q) else as(solve(K), "dgCMatrix")
    A_inv_pad <- Matrix::bdiag(Diagonal(p, 0), A_inv_grid)
    
    log_lambda_grid <- seq(-10, 10, length.out = 200)
    dev_values <- sapply(log_lambda_grid, function(log_lam) {
      lambda <- exp(log_lam)
      C <- MtM + lambda * A_inv_pad
      solver <- tryCatch(Matrix::Cholesky(C, LDL = FALSE), error = function(e) NULL)
      if (is.null(solver)) return(NA)
      
      theta_hat <- as.vector(solve(solver, Mty))
      res_sq <- sum((y - M %*% theta_hat)^2)
      varE_hat <- res_sq / (n - p)
      
      log_det_C <- 2.0 * determinant(solver, logarithm = TRUE, sqrt=TRUE)$modulus
      log_det_A <- determinant(A_inv_grid, logarithm = TRUE)$modulus
      
      val <- -0.5 * ((n - p) * log(varE_hat) + log_det_C - (q * log_lam + log_det_A) + (n - p))
      return(val)
    })
    
    df_profile <- data.frame(LogLambda = log_lambda_grid, REML_Objective = dev_values)
    est_lambda <- fit_object$varE / fit_object$varU
    est_log_lambda <- log(est_lambda)
    
    p_profile <- ggplot(df_profile, aes(x = LogLambda, y = REML_Objective)) +
      geom_line(color = "blue", linewidth = 1) +
      geom_vline(xintercept = est_log_lambda, color = "red", linetype = "dashed", linewidth = 1) +
      theme_minimal() +
      labs(
        title = sprintf("Profile Likelihood Surface (%s)", engine_name),
        subtitle = "Red dashed line = Final optimizer estimate (Boundary Check)",
        x = "Log(Lambda) [log(varE / varU)]", 
        y = "Profile REML Objective"
      ) +
      theme(plot.title = element_text(face = "bold"))
    
    plots$profile_curve <- p_profile
  }
  
  # C. Summary Table
  summary_table <- data.frame(
    Metric = c("Engine Name", "Estimated varE", "Estimated varU", "Status / Convergence"),
    Value = c(
      engine_name,
      round(fit_object$varE, 4),
      round(fit_object$varU, 4),
      ifelse(fit_object$varU < 1e-4 || fit_object$varE < 1e-4, 
             "⚠️ Warning: Potential Boundary Trap", 
             "✅ Clean Convergence")
    )
  )
  
  print(summary_table)
  return(list(summary = summary_table, plots = plots))
}

# =========================================================================
# Benchmark Execution
# =========================================================================
set.seed(42)
library(MASS)
library(Matrix)

# Simulate Dataset
n <- 600
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

n_animals <- 600  # Number of individuals / animals
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
# 3. SIMULATE ANIMAL MODEL DATASET (N = 1500)
# =========================================================================
set.seed(42)

n_animals <- 600 
p <- 2            
q <- n_animals    

true_beta <- c(15.0, 3.2)
true_varU <- 3.5  
true_varE <- 1.8  

X <- cbind(1, sample(c(0, 1), n_animals, replace = TRUE))
colnames(X) <- c("Intercept", "Sex")
Z <- diag(n_animals)

raw_markers <- matrix(rnorm(n_animals * 400), nrow = n_animals, ncol = 400)
K <- tcrossprod(raw_markers) / 400
diag(K) <- diag(K) + 0.05 

u <- mvrnorm(1, mu = rep(0, q), Sigma = K * true_varU)
e <- rnorm(n_animals, mean = 0, sd = sqrt(true_varE))
y <- as.vector(X %*% true_beta + Z %*% u + e)

A_inv_sparse <- methods::as(Matrix::Matrix(base::solve(K), sparse = TRUE), "generalMatrix")

# =========================================================================
# 4. RUN BENCHMARK ACROSS ENGINES
# =========================================================================
animal_engines <- c("kernel_bglr", "block_mcmcglmm", "penalized_map_blme", 
                    "reml_lme4", "spectral_rrblup", "sparse_asreml", "nr_sas")

animal_results <- data.frame(
  Engine = character(), Estimated_varE = numeric(),
  Estimated_varU = numeric(), EBV_Correlation = numeric(),
  Time_Seconds = numeric(), stringsAsFactors = FALSE
)

engine_fits <- list()

cat("\nStarting Animal Model Benchmark & Storage...\n")
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
  engine_fits[[eng]] <- fit
  
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

cat("\n--- Benchmark Results Summary ---\n")
print(animal_results)

# =========================================================================
# 5. EBV CORRECTION MATRIX & HEATMAP GENERATION
# =========================================================================
ebv_list <- list(True_U = as.vector(u))
for (eng in animal_engines) {
  fit <- engine_fits[[eng]]
  ebv_list[[eng]] <- if (!is.null(fit$u)) as.vector(fit$u) else as.vector(fit$post_u)
}

ebv_matrix <- do.call(cbind, ebv_list)
colnames(ebv_matrix) <- names(ebv_list)
ebv_cor_matrix <- cor(ebv_matrix, use = "pairwise.complete.obs")

cat("\n--- EBV Correlation Matrix ---\n")
print(round(ebv_cor_matrix, 2))

# Heatmap Plot
melted_cor <- melt(ebv_cor_matrix)
colnames(melted_cor) <- c("Engine1", "Engine2", "Correlation")

p_heatmap <- ggplot(melted_cor, aes(x = Engine1, y = Engine2, fill = Correlation)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "#2b580c", mid = "#f7fcb9", high = "#e31a1c", midpoint = 0.95, limits = c(0.80, 1.00), name = "Pearson\nCorrelation") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, face = "bold"),
    axis.text.y = element_text(face = "bold"),
    axis.title.x = element_blank(), axis.title.y = element_blank(),
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5)
  ) +
  coord_fixed() +
  geom_text(aes(label = sprintf("%.2f", Correlation)), color = "black", size = 3) +
  labs(title = "EBV Correlation Heatmap Across Mixed Model Engines")

print(p_heatmap)

# =========================================================================
# 6. EXECUTE DIAGNOSTIC REPORTS
# =========================================================================
# Example A: Check profile likelihood optimization curve for REML lme4
diag_lme4 <- GenerateModelDiagnosticReport(fit_object = engine_fits[["reml_lme4"]], y = y, X = X, Z = Z, K = K)
print(diag_lme4$plots$profile_curve)

# Example B: Check MCMC trace plots for BGLR Gibbs sampling
diag_bglr <- GenerateModelDiagnosticReport(fit_object = engine_fits[["kernel_bglr"]], y = y, X = X, Z = Z, K = K)

# This will now display the caterpillar trace plots successfully
print(diag_bglr$plots$mcmc_trace)

# all model looping 
# Initialize a storage list for all diagnostic plots and summaries
all_diagnostics <- list()

cat("Generating diagnostic reports across all active engines...\n")

for (eng_name in names(engine_fits)) {
  fit_obj <- engine_fits[[eng_name]]
  
  # Run the universal diagnostic report function
  rep_out <- GenerateModelDiagnosticReport(
    fit_object = fit_obj, 
    y = y, X = X, Z = Z, K = K
  )
  
  all_diagnostics[[eng_name]] <- rep_out
}

# Example: Display the profile curve for REML lme4
print(all_diagnostics[["reml_lme4"]]$plots$profile_curve)

# Example: Display the trace plots for Block MCMCglmm
print(all_diagnostics[["block_mcmcglmm"]]$plots$mcmc_trace)

# Example: Display the profile curve for SAS PROC MIXED
print(all_diagnostics[["nr_sas"]]$plots$profile_curve)

print(all_diagnostics[["spectral_rrblup"]]$plots$profile_curve)

print(all_diagnostics[["penalized_map_blme"]]$plots$profile_curve)