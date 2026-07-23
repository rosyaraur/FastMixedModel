library(Rcpp)
library(Matrix)

# Compile C++ backend
sourceCpp("lmm_engine.cpp")

#' Unified Mixed Model Solver
#' 
#' @param fixed Formula for fixed effects (e.g., y ~ x1)
#' @param random Formula for random effects (e.g., ~ group)
#' @param data Data frame containing variables
#' @param algorithm Choice between "aireml" (ASReml method) and "profiled" (lme4 method)
#' @param max_iter Maximum iterations for AI-REML
#' @param tol Convergence tolerance
fit_mixed_model <- function(fixed, random, data, 
                            algorithm = c("aireml", "profiled"), 
                            max_iter = 30, tol = 1e-5) {
  
  algorithm <- match.arg(algorithm)
  start_time <- Sys.time()
  
  # 1. Parse fixed and random model matrices
  mf_fixed <- model.frame(fixed, data)
  y <- model.response(mf_fixed)
  X <- model.matrix(fixed, mf_fixed)
  
  random_term <- as.character(random)[2]
  Z_formula <- as.formula(paste("~ 0 +", random_term))
  Z <- Matrix::sparse.model.matrix(Z_formula, data)
  
  beta_names <- colnames(X)
  u_names <- colnames(Z)
  
  # 2. Dispatch to requested algorithm
  if (algorithm == "profiled") {
    # Derivative-free optimization over log(lambda)
    opt_fn <- function(log_lambda) {
      eval_profiled_reml_cpp(X, Z, y, exp(log_lambda))$deviance
    }
    
    res_opt <- optimize(opt_fn, interval = c(-10, 10))
    best_lambda <- exp(res_opt$minimum)
    
    # Extract final estimates
    fit <- eval_profiled_reml_cpp(X, Z, y, best_lambda)
    
    sigma2_e <- fit$sigma2_e
    sigma2_u <- fit$sigma2_u
    beta <- fit$beta
    u <- fit$u
    iterations <- res_opt$objective
    
  } else if (algorithm == "aireml") {
    # AI-REML Derivative Update Loop
    sigma2_e <- var(y) * 0.5
    sigma2_u <- var(y) * 0.5
    
    iter <- 0
    converged <- FALSE
    
    while (!converged && iter < max_iter) {
      iter <- iter + 1
      step <- step_aireml_cpp(X, Z, y, sigma2_e, sigma2_u)
      
      sigma2_e <- step$sigma2_e
      sigma2_u <- step$sigma2_u
      beta <- step$beta
      u <- step$u
      
      if (step$change < tol) {
        converged <- TRUE
      }
    }
    iterations <- iter
  }
  
  elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  names(beta) <- beta_names
  
  # 3. Return structured model output
  structure(
    list(
      fixed_effects = beta,
      random_effects = setNames(u, u_names),
      vcov = c(sigma2_u = sigma2_u, sigma2_e = sigma2_e),
      algorithm = algorithm,
      iterations = iterations,
      runtime_sec = elapsed
    ),
    class = "custom_lmm"
  )
}

# Custom print method
print.custom_lmm <- function(x, ...) {
  cat("==========================================\n")
  cat(" Mixed Model Fit via:", toupper(x$algorithm), "\n")
  cat("==========================================\n\n")
  cat("Variance Components:\n")
  print(x$vcov)
  cat("\nFixed Effects:\n")
  print(x$fixed_effects)
  cat(sprintf("\nConvergence in %.4f sec\n", x$runtime_sec))
}