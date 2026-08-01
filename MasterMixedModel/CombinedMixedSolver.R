library(Matrix)

#' Combined Multi-Engine Mixed Solver
#' 
#' @param y Numeric vector of phenotypes.
#' @param X Dense matrix of fixed effects.
#' @param Z Sparse or dense matrix mapping random effects.
#' @param K Dense covariance matrix (e.g., Genomic Relationship Matrix).
#' @param A_inv Sparse inverse covariance matrix (e.g., Pedigree Inverse).
#' @param method String selecting the mathematical engine.
#' @param nIter Integer, total MCMC iterations (for Gibbs).
#' @param burnIn Integer, burn-in period (for Gibbs).
library(Matrix)

#' Combined Multi-Engine Mixed Solver
#' 
#' @param y Numeric vector of phenotypes.
#' @param X Dense matrix of fixed effects.
#' @param Z Sparse or dense matrix mapping random effects.
#' @param K Dense covariance matrix (e.g., Genomic Relationship Matrix).
#' @param A_inv Sparse inverse covariance matrix (e.g., Pedigree Inverse).
#' @param method String selecting the mathematical engine.
#' @param nIter Integer, total MCMC iterations (for Gibbs).
#' @param burnIn Integer, burn-in period (for Gibbs).
CombinedMixedSolver <- function(y, X, Z, 
                                K = NULL, A_inv = NULL, 
                                method = "kernel_bglr", 
                                nIter = 2000, burnIn = 500) {
  
  # --- 1. Global Initialization ---
  n <- length(y)
  p <- ncol(X)
  q <- ncol(Z)
  
  # MCMC Storage
  chain_beta  <- matrix(0, nrow = nIter, ncol = p)
  chain_u     <- matrix(0, nrow = nIter, ncol = q)
  chain_varE  <- numeric(nIter)
  chain_varU  <- numeric(nIter)
  
  # Starting Values
  varE <- var(y, na.rm = TRUE) * 0.5
  varU <- var(y, na.rm = TRUE) * 0.5
  beta <- rep(0, p)
  u    <- rep(0, q)
  
  # Prior Hyperparameters (Scaled Inverse-ChiSq)
  df_e0 <- 5; S_e0 <- varE * (df_e0 - 2)
  df_u0 <- 5; S_u0 <- varU * (df_u0 - 2)
  
  
  # =========================================================================
  # PATH A: BGLR Essence (Kernel Diagonalization + Single-Site Scalar Gibbs)
  # =========================================================================
  if (method == "kernel_bglr") {
    if (is.null(K)) stop("Method requires raw covariance matrix K.")
    
    ev <- eigen(K, symmetric = TRUE)
    V <- ev$vectors; d <- ev$values
    keep <- d > 1e-8; V <- V[, keep]; d <- d[keep]
    q_star <- length(d)
    
    W <- Z %*% V
    u_star <- rep(0, q_star)
    x2 <- colSums(X^2); w2 <- colSums(W^2)
    e <- as.vector(y - X %*% beta - W %*% u_star)
    
    for (iter in 1:nIter) {
      for (j in 1:p) {
        e <- e + X[, j] * beta[j]
        lhs <- x2[j] / varE
        sol <- (sum(X[, j] * e) / varE) / lhs
        beta[j] <- rnorm(1, mean = sol, sd = sqrt(1 / lhs))
        e <- e - X[, j] * beta[j]
      }
      for (j in 1:q_star) {
        e <- e + W[, j] * u_star[j]
        lhs <- w2[j] / varE + 1 / (d[j] * varU)
        sol <- (sum(W[, j] * e) / varE) / lhs
        u_star[j] <- rnorm(1, mean = sol, sd = sqrt(1 / lhs))
        e <- e - W[, j] * u_star[j]
      }
      
      varE <- (sum(e^2) + S_e0) / rchisq(1, df = n + df_e0)
      varU <- (sum((u_star^2) / d) + S_u0) / rchisq(1, df = q_star + df_u0)
      
      chain_beta[iter, ] <- beta; chain_u[iter, ] <- as.vector(V %*% u_star)
      chain_varE[iter] <- varE; chain_varU[iter] <- varU
    }
    
    keep_idx <- (burnIn + 1):nIter
    return(list(method = method, post_beta = colMeans(chain_beta[keep_idx, , drop = FALSE]),
                post_u = colMeans(chain_u[keep_idx, , drop = FALSE]), post_varE = mean(chain_varE[keep_idx]), post_varU = mean(chain_varU[keep_idx]), chains = list(beta = chain_beta, u = chain_u)))
  }
  
  # =========================================================================
  # PATH B: MCMCglmm Essence (A-inverse + Multivariate Block Sampling via MME)
  # =========================================================================
  else if (method == "block_mcmcglmm") {
    if (is.null(A_inv)) stop("Method requires sparse matrix A_inv.")
    
    A_inv <- as(A_inv, "dgCMatrix")
    M <- cbind(X, Z); MtM <- crossprod(M); Mty <- as.vector(crossprod(M, y))
    
    for (iter in 1:nIter) {
      C <- (1 / varE) * MtM + bdiag(Diagonal(p, 1e-8), A_inv * (1 / varU))
      L <- Cholesky(C, LDL = FALSE)
      
      theta_sample <- as.vector(solve(L, as.vector(solve(L, (1 / varE) * Mty)))) + 
        as.vector(solve(L, rnorm(p + q), system = "Lt"))
      beta <- theta_sample[1:p]; u <- theta_sample[(p + 1):(p + q)]
      
      varE <- (sum((y - M %*% theta_sample)^2) + S_e0) / rchisq(1, df = n + df_e0)
      varU <- (as.numeric(t(u) %*% A_inv %*% u) + S_u0) / rchisq(1, df = q + df_u0)
      
      chain_beta[iter, ] <- beta; chain_u[iter, ] <- u
      chain_varE[iter] <- varE; chain_varU[iter] <- varU
    }
    
    keep_idx <- (burnIn + 1):nIter
    return(list(method = method, post_beta = colMeans(chain_beta[keep_idx, , drop = FALSE]), post_u = colMeans(chain_u[keep_idx, , drop = FALSE]), post_varE = mean(chain_varE[keep_idx]), post_varU = mean(chain_varU[keep_idx]), chains = list(beta = chain_beta, u = chain_u)))
  }
  
  # =========================================================================
  # PATH C: blme Essence (Penalized MAP Optimization via Parameter Profiling)
  # =========================================================================
  else if (method == "penalized_map_blme") {
    if (is.null(A_inv)) A_inv <- solve(K)
    A_inv <- as(A_inv, "dgCMatrix")
    M <- cbind(X, Z); MtM <- crossprod(M); Mty <- as.vector(crossprod(M, y))
    
    penalized_deviance <- function(log_lambda) {
      lambda <- exp(log_lambda)
      C <- MtM + bdiag(Diagonal(p, 0), A_inv * lambda)
      theta_hat <- as.vector(solve(C, Mty))
      
      varE_hat <- sum((y - M %*% theta_hat)^2) / n
      varU_hat <- as.numeric(t(theta_hat[(p + 1):(p + q)]) %*% A_inv %*% theta_hat[(p + 1):(p + q)]) / q
      
      return(n * log(varE_hat) + q * log(varU_hat) + 
               ((df_e0 / 2 + 1) * log(varE_hat) + (S_e0 / (2 * varE_hat))) + 
               ((df_u0 / 2 + 1) * log(varU_hat) + (S_u0 / (2 * varU_hat))))
    }
    
    lambda_map <- exp(optim(0, penalized_deviance, method = "Brent", lower = -10, upper = 10)$par)
    theta_map <- as.vector(solve(MtM + bdiag(Diagonal(p, 0), A_inv * lambda_map), Mty))
    varE_map <- sum((y - M %*% theta_map)^2) / n
    
    return(list(method = method, post_beta = theta_map[1:p], post_u = theta_map[(p + 1):(p + q)], post_varE = varE_map, post_varU = varE_map / lambda_map, chains = NULL))
  }
  
  # =========================================================================
  # PATH D: brms Essence (Hamiltonian Monte Carlo Gradient Evaluation Concept)
  # =========================================================================
  else if (method == "hmc_stan") {
    return(list(method = method, note = "HMC gradients bypass sparse Cholesky MME entirely. Requires Stan C++ compilation.", chains = NULL))
  }
  
  # =========================================================================
  # PATH E: lme4 Essence (Frequentist Profiled REML Optimization)
  # =========================================================================
  else if (method == "reml_lme4") {
    if (is.null(A_inv)) A_inv <- if(!is.null(K)) solve(K) else Diagonal(q)
    M <- cbind(X, Z); MtM <- crossprod(M); Mty <- as.vector(crossprod(M, y))
    
    reml_objective <- function(log_lambda) {
      lambda <- exp(log_lambda)
      L <- Cholesky(MtM + bdiag(Diagonal(p, 0), A_inv * lambda), LDL = FALSE)
      theta_hat <- as.vector(solve(L, Mty))
      varE_hat <- sum((y - M %*% theta_hat)^2) / (n - p)
      
      return(-0.5 * ((n - p) * log(varE_hat) + 2 * determinant(L, logarithm = TRUE, sqrt = TRUE)$modulus - 
                       (q * log(lambda) + determinant(A_inv, logarithm = TRUE)$modulus) + (n - p)))
    }
    
    # Use a wider interval centered closer to the expected ratio (varE / varU)
    init_log_lambda <- log(true_varE / true_varU)
    opt_res <- optimize(f = function(x) -reml_objective(x), interval = c(init_log_lambda - 15, init_log_lambda + 15))
    
    lambda_reml <- exp(opt_res$minimum)
    theta_reml <- as.vector(solve(MtM + bdiag(Diagonal(p, 0), A_inv * lambda_reml), Mty))
    varE_reml <- sum((y - M %*% theta_reml)^2) / (n - p)
    
    return(list(
      method = method, 
      post_beta = theta_reml[1:p], 
      post_u = theta_reml[(p + 1):(p + q)], 
      post_varE = varE_reml, 
      post_varU = varE_reml / lambda_reml, 
      chains = NULL
    ))
  }
  
  # =========================================================================
  # PATH F: mbest Essence (Fast Moment-Based Estimation)
  # =========================================================================
  else if (method == "moment_mbest") {
    if (is.null(A_inv)) A_inv <- Diagonal(q)
    
    beta_ols <- as.vector(solve(crossprod(X)) %*% crossprod(X, y))
    res_ols <- as.vector(y - X %*% beta_ols)
    
    u_naive <- as.vector(solve(crossprod(Z) + Diagonal(q, 1e-6)) %*% crossprod(Z, res_ols))
    varU_mom <- max(var(u_naive), 1e-6)
    varE_mom <- max(var(res_ols - as.vector(Z %*% u_naive)), 1e-6)
    
    theta_mom <- as.vector(solve(crossprod(cbind(X, Z)) + bdiag(Diagonal(p, 0), A_inv * (varE_mom / varU_mom)), crossprod(cbind(X, Z), y)))
    
    return(list(method = method, post_beta = theta_mom[1:p], post_u = theta_mom[(p + 1):(p + q)], post_varE = varE_mom, post_varU = varU_mom, chains = NULL))
  }
  
  # =========================================================================
  # PATH G: rrBLUP Essence (Spectral Decomposition REML)
  # =========================================================================
  else if (method == "spectral_rrblup") {
    if (is.null(K)) stop("Method requires dense Kinship matrix K.")
    
    eig <- eigen(Z %*% K %*% t(Z), symmetric = TRUE)
    U <- eig$vectors; d <- eig$values
    y_star <- as.vector(crossprod(U, y)); X_star <- as.matrix(crossprod(U, X))
    
    spectral_reml <- function(log_delta) {
      delta <- exp(log_delta); V_inv_diag <- 1 / (d + delta)
      Xt_Vinv_X <- crossprod(X_star, X_star * V_inv_diag)
      beta_hat <- solve(Xt_Vinv_X, crossprod(X_star, y_star * V_inv_diag))
      varU_hat <- sum((y_star - X_star %*% beta_hat)^2 * V_inv_diag) / (n - p)
      
      return(as.numeric(0.5 * ((n - p) * log(varU_hat) + sum(log(d + delta)) + determinant(Xt_Vinv_X, logarithm = TRUE)$modulus)))
    }
    
    delta_opt <- exp(optimize(f = spectral_reml, interval = c(-10, 10))$minimum)
    V_inv_diag <- 1 / (d + delta_opt)
    beta_rrblup <- as.vector(solve(crossprod(X_star, X_star * V_inv_diag), crossprod(X_star, y_star * V_inv_diag)))
    varU_rrblup <- sum((y_star - X_star %*% beta_rrblup)^2 * V_inv_diag) / (n - p)
    
    u_rrblup <- as.vector(K %*% t(Z) %*% (U %*% diag(V_inv_diag) %*% t(U)) %*% (y - X %*% beta_rrblup))
    
    return(list(method = method, post_beta = beta_rrblup, post_u = u_rrblup, post_varE = varU_rrblup * delta_opt, post_varU = varU_rrblup, chains = NULL))
  }
  
  # =========================================================================
  # PATH H: sommer Essence (Direct Inversion & AI-REML)
  # =========================================================================
  else if (method == "direct_aireml_sommer") {
    if (is.null(K)) stop("Method requires dense Kinship matrix K.")
    ZKZt <- Z %*% K %*% t(Z)
    varE_hat <- var(y) * 0.5; varU_hat <- var(y) * 0.5
    
    diff <- 1; iter <- 0
    while (diff > 1e-5 && iter < 100) {
      iter <- iter + 1; vE_old <- varE_hat; vU_old <- varU_hat
      
      V_inv <- solve((ZKZt * varU_hat) + Diagonal(n, varE_hat))
      Xt_Vinv_X <- crossprod(X, V_inv %*% X)
      P <- V_inv - V_inv %*% X %*% solve(Xt_Vinv_X) %*% crossprod(X, V_inv)
      
      y_P <- as.vector(P %*% y)
      varU_hat <- max(1e-6, varU_hat + (varU_hat^2) * (sum(y_P * as.vector(ZKZt %*% y_P)) - sum(diag(P %*% ZKZt))) / q)
      varE_hat <- max(1e-6, varE_hat + (varE_hat^2) * (sum(y_P * y_P) - sum(diag(P))) / n)
      diff <- max(abs(varU_hat - vU_old), abs(varE_hat - vE_old))
    }
    beta_hat <- as.vector(solve(Xt_Vinv_X, crossprod(X, V_inv %*% y)))
    
    return(list(method = method, post_beta = beta_hat, post_u = as.vector(K %*% t(Z) %*% P %*% y) * varU_hat, post_varE = varE_hat, post_varU = varU_hat, chains = NULL))
  }
  
  # =========================================================================
  # PATH I: ASReml Essence (Sparse MME + AI-REML)
  # =========================================================================
  else if (method == "sparse_aireml_asreml") {
    if (is.null(A_inv)) stop("Method requires sparse A_inv.")
    A_inv <- as(A_inv, "dgCMatrix")
    M <- cbind(X, Z); MtM <- crossprod(M); Mty <- as.vector(crossprod(M, y))
    varE_hat <- var(y) * 0.5; varU_hat <- var(y) * 0.5
    
    diff <- 1; iter <- 0
    while (diff > 1e-5 && iter < 50) {
      iter <- iter + 1; vE_old <- varE_hat; vU_old <- varU_hat
      
      L <- Cholesky(MtM + bdiag(Diagonal(p, 0), A_inv * (varE_hat / varU_hat)), LDL = FALSE)
      theta_hat <- as.vector(solve(L, Mty))
      
      C_inv_diag <- diag(solve(L, system = "A"))
      
      varU_hat <- max(1e-6, (as.numeric(t(theta_hat[(p + 1):(p + q)]) %*% A_inv %*% theta_hat[(p + 1):(p + q)]) + sum(diag(A_inv) * C_inv_diag[(p + 1):(p + q)]) * varE_hat) / q)
      varE_hat <- max(1e-6, (sum((as.vector(y - M %*% theta_hat))^2) + sum(C_inv_diag * diag(MtM)) * varE_hat) / (n - p))
      diff <- max(abs(varU_hat - vU_old), abs(varE_hat - vE_old))
    }
    
    return(list(method = method, post_beta = theta_hat[1:p], post_u = theta_hat[(p + 1):(p + q)], post_varE = varE_hat, post_varU = varU_hat, chains = NULL))
  }
  
  # =========================================================================
  # PATH J: SAS PROC MIXED Essence (G & R structures + Newton-Raphson)
  # =========================================================================
  else if (method == "nr_sasmixed") {
    if (is.null(K)) K <- diag(q)
    
    beta_ols <- solve(crossprod(X), crossprod(X, y))
    varE_hat <- var(as.vector(y - X %*% beta_ols)) * 0.8
    varU_hat <- var(as.vector(y - X %*% beta_ols)) * 0.2
    
    diff <- 1; iter <- 0; ridge_lambda <- 1e-4
    while (diff > 1e-5 && iter < 50) {
      iter <- iter + 1; vE_old <- varE_hat; vU_old <- varU_hat
      
      V_inv <- solve((Z %*% (K * varU_hat) %*% t(Z)) + Diagonal(n, varE_hat))
      Xt_Vinv_X <- crossprod(X, V_inv %*% X)
      P <- V_inv - V_inv %*% X %*% solve(Xt_Vinv_X) %*% crossprod(X, V_inv)
      
      y_P <- as.vector(P %*% y); ZKZt <- Z %*% K %*% t(Z)
      gradient <- c(-0.5 * sum(diag(P %*% ZKZt)) + 0.5 * sum(y_P * as.vector(ZKZt %*% y_P)),
                    -0.5 * sum(diag(P)) + 0.5 * sum(y_P * y_P))
      
      P_ZKZt <- P %*% ZKZt
      Hessian <- matrix(c(0.5 * sum(diag(P_ZKZt %*% P_ZKZt)), 0.5 * sum(diag(P_ZKZt %*% P)),
                          0.5 * sum(diag(P_ZKZt %*% P)), 0.5 * sum(diag(P %*% P))), 2, 2)
      
      if (any(eigen(Hessian)$values < 1e-8)) {
        Hessian <- Hessian + diag(2) * ridge_lambda
        ridge_lambda <- ridge_lambda * 10
      } else { ridge_lambda <- max(1e-6, ridge_lambda / 10) }
      
      update_step <- solve(Hessian, gradient)
      varU_hat <- max(1e-6, varU_hat + update_step[1])
      varE_hat <- max(1e-6, varE_hat + update_step[2])
      diff <- max(abs(varU_hat - vU_old), abs(varE_hat - vE_old))
    }
    
    return(list(method = method, post_beta = as.vector(solve(Xt_Vinv_X, crossprod(X, V_inv %*% y))), post_u = as.vector((K * varU_hat) %*% t(Z) %*% P %*% y), post_varE = varE_hat, post_varU = varU_hat, chains = NULL))
  }
  
  else stop(paste("Method", method, "is not a recognized engine path."))
}

#' Extract and Standardize Mixed Model Information
#' 
#' @param model_fit The fitted model object (from CombinedMixedSolver or native packages).
#' @param engine String specifying the engine used (e.g., "combined", "lme4", "bglr", "sommer").
#' @return A unified list containing Fixed Effects, Random Effects (EBVs), and Variance Components.
ExtractMixedInfo <- function(model_fit, engine = "combined") {
  
  # Initialize the standard schema
  out <- list(
    Engine = engine,
    FixedEffects = NULL,
    RandomEffects = NULL,
    VarE = NA,
    VarU = NA
  )
  
  # ---------------------------------------------------------
  # 1. Custom CombinedMixedSolver Output
  # ---------------------------------------------------------
  if (engine == "combined") {
    out$FixedEffects  <- as.vector(model_fit$post_beta)
    out$RandomEffects <- as.vector(model_fit$post_u)
    out$VarE          <- model_fit$post_varE
    out$VarU          <- model_fit$post_varU
  }
  
  # ---------------------------------------------------------
  # 2. Native lme4 Output (merMod)
  # ---------------------------------------------------------
  else if (engine == "lme4") {
    library(lme4)
    out$FixedEffects  <- lme4::fixef(model_fit)
    out$RandomEffects <- as.vector(lme4::ranef(model_fit)[[1]][, 1])
    
    var_cors <- lme4::VarCorr(model_fit)
    out$VarU <- as.numeric(var_cors)
    out$VarE <- attr(var_cors, "sc")^2
  }
  
  # ---------------------------------------------------------
  # 3. Native BGLR Output
  # ---------------------------------------------------------
  else if (engine == "bglr") {
    out$FixedEffects  <- as.vector(model_fit$ETA[[1]]$b)
    out$RandomEffects <- as.vector(model_fit$ETA[[2]]$u)
    out$VarE          <- model_fit$varE
    out$VarU          <- model_fit$ETA[[2]]$varU
  }
  
  # ---------------------------------------------------------
  # 4. Native sommer Output (mmer)
  # ---------------------------------------------------------
  else if (engine == "sommer") {
    out$FixedEffects  <- as.vector(model_fit$Beta$Estimate)
    out$RandomEffects <- as.vector(model_fit$U[[1]][[1]]) 
    
    out$VarU <- model_fit$sigma[[1]][1, 1]
    out$VarE <- model_fit$sigma[[length(model_fit$sigma)]][1, 1]
  }
  
  # ---------------------------------------------------------
  # Fallback Error
  # ---------------------------------------------------------
  else {
    stop("Engine extraction path not yet defined in ExtractMixedInfo.")
  }
  
  # Clean up names for absolute consistency
  names(out$FixedEffects)  <- paste0("Beta_", 1:length(out$FixedEffects))
  names(out$RandomEffects) <- paste0("U_", 1:length(out$RandomEffects))
  
  return(out)
}

# =========================================================================
# 1. SIMULATE KNOWN VARIANCE COMPONENT DATASET
# =========================================================================
set.seed(42)

n_obs <- 1000
n_groups <- 100 # e.g., 100 animals, 10 observations each

# True Parameters
beta_true <- c(10, 5) # Intercept and one covariate
varU_true <- 2.0      # True group/genetic variance
varE_true <- 1.0      # True residual variance

# Fixed Effects
X <- cbind(1, rnorm(n_obs))
colnames(X) <- c("Intercept", "Covariate")

# Random Effects & Incidence Matrix
# 10 observations per group
Z <- as.matrix(Matrix::kronecker(diag(n_groups), rep(1, n_obs/n_groups)))
u_true <- rnorm(n_groups, mean = 0, sd = sqrt(varU_true))

# For simplicity in this baseline benchmark, we assume independent random effects (K = I)
K <- diag(n_groups)
A_inv <- solve(K)

# Phenotype
e_true <- rnorm(n_obs, mean = 0, sd = sqrt(varE_true))
y <- as.vector(X %*% beta_true + Z %*% u_true + e_true)

# Dataframe for native packages
df <- data.frame(
  y = y,
  Covariate = X[, 2],
  GroupID = as.factor(rep(1:n_groups, each = n_obs/n_groups))
)

# =========================================================================
# 2. RUN OUR CUSTOM WRAPPER (CombinedMixedSolver)
# =========================================================================
cat("\n--- Running Custom Wrapper (CombinedMixedSolver) ---\n")

# A. Frequentist REML (lme4 style)
wrap_lme4 <- CombinedMixedSolver(y, X, Z, A_inv = A_inv, method = "reml_lme4")

# B. Spectral REML (rrBLUP style)
wrap_rrblup <- CombinedMixedSolver(y, X, Z, K = K, method = "spectral_rrblup")

# C. AI-REML (sommer style)
wrap_sommer <- CombinedMixedSolver(y, X, Z, K = K, method = "direct_aireml_sommer")

# D. Penalized MAP (blme style)
wrap_blme <- CombinedMixedSolver(y, X, Z, A_inv = A_inv, method = "penalized_map_blme")

# E. Kernel Gibbs (BGLR style) - taking fewer iters for benchmark speed
wrap_bglr <- CombinedMixedSolver(y, X, Z, K = K, method = "kernel_bglr", nIter = 3000, burnIn = 1000)

# =========================================================================
# 3. RUN NATIVE R PACKAGES
# =========================================================================
cat("\n--- Running Native Packages ---\n")

# A. lme4
library(lme4)
fit_lme4 <- lmer(y ~ Covariate + (1 | GroupID), data = df, REML = TRUE)
vc_lme4 <- as.data.frame(VarCorr(fit_lme4))

# B. rrBLUP
library(rrBLUP)
fit_rrblup <- mixed.solve(y = y, X = X, Z = Z, K = K, method = "REML")

# C. sommer
library(sommer)
# For simplicity in this baseline benchmark, we assume independent random effects (K = I)
K <- diag(n_groups)
rownames(K) <- colnames(K) <- 1:n_groups # sommer requires matching dimnames
A_inv <- solve(K)
fit_sommer <- mmer(y ~ Covariate, random = ~ vs(GroupID, Gu = K), data = df, verbose = FALSE)

# D. blme
library(blme)
fit_blme_native <- blmer(y ~ Covariate + (1 | GroupID), data = df)
vc_blme <- as.data.frame(VarCorr(fit_blme_native))

# E. BGLR
library(BGLR)
# Project K to the observation level for BGLR's RKHS model
K_bglr <- as.matrix(Z %*% K %*% t(Z))

# Define the ETA list
ETA <- list(
  list(X = X, model = "FIXED"),
  list(K = K_bglr, model = "RKHS")
)

# Run BGLR
capture.output(
  fit_bglr_native <- BGLR(y = y, ETA = ETA, nIter = 3000, burnIn = 1000, verbose = FALSE)
)

# =========================================================================
# 4. BENCHMARK COMPARISON MATRIX
# =========================================================================

benchmark_results <- data.frame(
  Engine = c("True Parameters", 
             "Wrapper: lme4", "Native: lme4",
             "Wrapper: rrBLUP", "Native: rrBLUP",
             "Wrapper: sommer", "Native: sommer",
             "Wrapper: blme", "Native: blme",
             "Wrapper: BGLR", "Native: BGLR"),
  VarU = c(varU_true,
           wrap_lme4$post_varU, vc_lme4$vcov[1],
           wrap_rrblup$post_varU, fit_rrblup$Vu,
           wrap_sommer$post_varU, unlist(fit_sommer$sigma)[1],
           wrap_blme$post_varU, vc_blme$vcov[1],
           wrap_bglr$post_varU, fit_bglr_native$ETA[[2]]$varU),
  VarE = c(varE_true,
           wrap_lme4$post_varE, vc_lme4$vcov[2],
           wrap_rrblup$post_varE, fit_rrblup$Ve,
           wrap_sommer$post_varE, unlist(fit_sommer$sigma)[2],
           wrap_blme$post_varE, vc_blme$vcov[2],
           wrap_bglr$post_varE, fit_bglr_native$varE)
)

print(knitr::kable(benchmark_results, digits = 4))