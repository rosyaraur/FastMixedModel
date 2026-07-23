# ==============================================================================
# INDEPENDENT SINGLE-STEP GENOMIC MET SOLVER (EM vs. NR)
# ==============================================================================

library(Rcpp)
library(Matrix)

# ------------------------------------------------------------------------------
# 1. Simulate MET Data
# ------------------------------------------------------------------------------
set.seed(2026)

# 100 Genotypes tested across 4 Environments
n_genotypes <- 100
n_envs <- 4
n_markers <- 200

# Simulate Marker Data (0, 1, 2) and center it
M <- matrix(rbinom(n_genotypes * n_markers, 2, 0.5), nrow = n_genotypes, ncol = n_markers)
rownames(M) <- paste0("G", 1:n_genotypes)
M_centered <- scale(M, center = TRUE, scale = FALSE)

# Calculate Additive Genomic Relationship Matrix (G)
scaling_factor <- 2 * mean(colMeans(M / 2) * (1 - colMeans(M / 2)))
G_matrix <- (M_centered %*% t(M_centered)) / scaling_factor

# Simulate Phenotypic Data based on Genotypes
met_data <- expand.grid(Genotype = rownames(G_matrix), Environment = paste0("E", 1:n_envs))
met_data$Yield <- 20 + rnorm(nrow(met_data), 0, 5)
met_data$Env_Geno <- factor(paste(met_data$Environment, met_data$Genotype, sep = ":"))

# ------------------------------------------------------------------------------
# 2. C++ Engine (EM and Newton-Raphson)
# ------------------------------------------------------------------------------
cpp_code <- "
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <string>

// [[Rcpp::export]]
Rcpp::List solve_genomic_mme_cpp(const arma::mat& X, const arma::mat& Z, 
                                 const arma::vec& y, const arma::mat& G_genomic, 
                                 int n_envs, std::string method = \"EM\",
                                 int max_iter = 100, double tol = 1e-5) {
    int n = y.n_elem;
    int nx = X.n_cols;
    int nz = Z.n_cols;
    int n_genotypes = G_genomic.n_rows;
    
    arma::mat G_genomic_inv = arma::pinv(G_genomic);
    
    // Initial variance guesses
    double var_e = arma::var(y) / 2.0;
    arma::mat G_env = arma::eye(n_envs, n_envs) * (arma::var(y) / 2.0);
    
    arma::mat XtX = X.t() * X;
    arma::mat XtZ = X.t() * Z;
    arma::mat ZtX = Z.t() * X;
    arma::mat ZtZ = Z.t() * Z;
    
    arma::vec b = arma::zeros(nx);
    arma::vec u = arma::zeros(nz);
    
    int iter_used = 0;

    for(int iter = 0; iter < max_iter; iter++) {
        iter_used++;
        arma::mat G_env_inv = arma::pinv(G_env);
        arma::mat G_full_inv = arma::kron(G_env_inv, G_genomic_inv);

        // --- 1. Expectation-Maximization (EM) Update ---
        if (method == \"EM\") {
            arma::mat LHS = arma::join_cols(
                arma::join_rows(XtX, XtZ),
                arma::join_rows(ZtX, ZtZ + (G_full_inv * var_e))
            );
            arma::mat RHS = arma::join_cols(X.t() * y, Z.t() * y);
            arma::mat sol = arma::pinv(LHS) * RHS;
            
            b = sol.rows(0, nx - 1);
            u = sol.rows(nx, nx + nz - 1);

            var_e = arma::as_scalar((y - X*b - Z*u).t() * (y - X*b - Z*u)) / (n - nx);
            
            arma::mat U_mat = arma::reshape(u, n_genotypes, n_envs); 
            arma::mat U_scaled = U_mat.t() * G_genomic_inv * U_mat;
            
            arma::mat C_inv = arma::pinv(LHS);
            arma::mat C22 = C_inv.submat(nx, nx, nx + nz - 1, nx + nz - 1);
            
            for (int i = 0; i < n_envs; ++i) {
                for (int j = 0; j < n_envs; ++j) {
                    G_env(i,j) = U_scaled(i,j) / n_genotypes; 
                }
            }
            
            arma::vec diag_C22 = C22.diag();
            arma::mat C22_reshaped = arma::reshape(diag_C22, n_genotypes, n_envs);
            arma::rowvec trace_comp = arma::sum(C22_reshaped, 0); 
            
            for(int i = 0; i < n_envs; ++i) {
                G_env(i,i) += (trace_comp(i) * var_e) / n_genotypes;
            }
        } 
        // --- 2. Newton-Raphson / Fisher Scoring (NR) Update ---
        // Analogous to sommer's direct inversion technique
        else if (method == \"NR\") {
            arma::mat G_full = arma::kron(G_env, G_genomic);
            arma::mat V = Z * G_full * Z.t() + arma::eye(n, n) * var_e;
            arma::mat V_inv = arma::pinv(V);
            
            arma::mat Xt_Vinv_X = X.t() * V_inv * X;
            arma::mat P = V_inv - V_inv * X * arma::pinv(Xt_Vinv_X) * X.t() * V_inv;
            
            // Calculate BLUPs/BLUEs explicitly via V_inv
            b = arma::pinv(Xt_Vinv_X) * X.t() * V_inv * y;
            u = G_full * Z.t() * P * y;
            
            // REML Score (First Derivative) & Information for var_e
            double score_e = -0.5 * arma::trace(P) + 0.5 * arma::as_scalar(y.t() * P * P * y);
            double info_e = 0.5 * arma::trace(P * P);
            
            // Newton-Raphson Step for residual variance
            double delta_e = score_e / info_e;
            var_e = var_e + delta_e;
            if(var_e < 1e-6) var_e = 1e-6; // Prevent negative variance
            
            // Simplified Trace-based NR step for G_env to approximate AI-REML
            arma::mat U_mat = arma::reshape(u, n_genotypes, n_envs);
            arma::mat U_scaled = U_mat.t() * G_genomic_inv * U_mat;
            
            arma::mat Zt_P_Z = Z.t() * P * Z;
            arma::vec diag_ZPZ = Zt_P_Z.diag();
            arma::mat ZPZ_reshaped = arma::reshape(diag_ZPZ, n_genotypes, n_envs);
            arma::rowvec trace_PZ = arma::sum(ZPZ_reshaped, 0);
            
            for(int i = 0; i < n_envs; ++i) {
                // Approximate Fisher step for variance parameters
                G_env(i,i) = (U_scaled(i,i) + trace_PZ(i)) / n_genotypes;
            }
        }
    }

    return Rcpp::List::create(
        Rcpp::Named(\"BLUEs\") = b, 
        Rcpp::Named(\"GBLUPs\") = u, 
        Rcpp::Named(\"G_env\") = G_env, 
        Rcpp::Named(\"var_e\") = var_e,
        Rcpp::Named(\"iterations\") = iter_used,
        Rcpp::Named(\"method\") = method
    );
}
"
sourceCpp(code = cpp_code)

# ------------------------------------------------------------------------------
# 3. Independent R Interface
# ------------------------------------------------------------------------------
fit_genomic_mme <- function(fixed, random, data, G_mat, n_envs, method = "EM") {
  resp_name <- all.vars(fixed)[1]
  y_mat <- as.matrix(data[[resp_name]])
  
  X <- model.matrix(fixed, data)
  
  Z <- model.matrix(random, data)
  ordered_levels <- paste(rep(paste0("E", 1:n_envs), each = nrow(G_mat)), 
                          rep(rownames(G_mat), times = n_envs), sep = ":")
  
  clean_Z_colnames <- gsub(all.vars(random)[1], "", colnames(Z))
  Z <- Z[, match(ordered_levels, clean_Z_colnames)]
  
  results <- solve_genomic_mme_cpp(X, Z, y_mat, G_mat, n_envs, method = method)
  rownames(results$BLUEs) <- colnames(X)
  
  gblups_list <- list()
  gblups_vector <- as.numeric(results$GBLUPs)
  names(gblups_vector) <- ordered_levels
  gblups_list$U$`rr(Environment, d = 2):Genotype`$Yield <- gblups_vector
  
  return(list(
    BLUEs = results$BLUEs,
    GBLUPs_raw = results$GBLUPs,
    U = gblups_list$U,
    G_env = results$G_env,
    var_e = results$var_e,
    iterations = results$iterations,
    method = results$method
  ))
}

# ------------------------------------------------------------------------------
# 4. Execution and Performance Comparison
# ------------------------------------------------------------------------------
cat("\n======================================================\n")
cat("          ALGORITHM PERFORMANCE COMPARISON              \n")
cat("======================================================\n\n")

# --- 1. Fit using Expectation-Maximization (EM) ---
cat("Fitting Model via EM (Stable, Linear Convergence)...\n")
time_em <- system.time({
  model_em <- fit_genomic_mme(
    fixed = Yield ~ Environment,
    random = ~ 0 + Env_Geno,
    data = met_data,
    G_mat = G_matrix,
    n_envs = n_envs,
    method = "EM"
  )
})
cat("EM Time:\n")
print(time_em)

# --- 2. Fit using Newton-Raphson (NR - Like sommer) ---
cat("\nFitting Model via Newton-Raphson (Fast, Quadratic Convergence)...\n")
time_nr <- system.time({
  model_nr <- fit_genomic_mme(
    fixed = Yield ~ Environment,
    random = ~ 0 + Env_Geno,
    data = met_data,
    G_mat = G_matrix,
    n_envs = n_envs,
    method = "NR"
  )
})
cat("NR Time:\n")
print(time_nr)

# --- 3. Compare Results ---
cat("\n--- Variance Estimates Comparison ---\n")
var_comparison <- data.frame(
  Method = c("EM", "Newton-Raphson"),
  Residual_Variance = c(model_em$var_e, model_nr$var_e),
  Avg_Env_Variance = c(mean(diag(model_em$G_env)), mean(diag(model_nr$G_env)))
)
print(var_comparison)

cat("\n--- Top 5 GBLUPs Comparison (Yield) ---\n")
#gblups_em <- head(model_em$U$`rr(Environment, d = 2):Genotype`$Yield, 5)
#gblups_nr <- head(model_nr$U$`rr(Environment, d = 2):Genotype`$Yield, 5)
gblups_em <- model_em$U$`rr(Environment, d = 2):Genotype`$Yield
gblups_nr <- model_nr$U$`rr(Environment, d = 2):Genotype`$Yield

blup_comparison <- data.frame(
  Genotype_Env = names(gblups_em),
  EM_GBLUP = gblups_em,
  NR_GBLUP = gblups_nr
)
print(blup_comparison)
plot(blup_comparison$NR_GBLUP, blup_comparison$EM_GBLUP)
