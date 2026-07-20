# ==============================================================================
# INDEPENDENT SINGLE-STEP GENOMIC MET SOLVER
# ==============================================================================

library(Rcpp)
library(Matrix)

# ------------------------------------------------------------------------------
# 1. Simulate MET Data (User Provided)
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
met_data$Yield <- 20 + rnorm(nrow(met_data), 0, 5) # Base yield + noise
met_data$Env_Geno <- factor(paste(met_data$Environment, met_data$Genotype, sep = ":"))

# ------------------------------------------------------------------------------
# 2. C++ Engine (Genomic Expectation-Maximization)
# ------------------------------------------------------------------------------
cpp_code <- "
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

// [[Rcpp::export]]
Rcpp::List solve_genomic_mme_cpp(const arma::mat& X, const arma::mat& Z, 
                                 const arma::vec& y, const arma::mat& G_genomic, 
                                 int n_envs, int max_iter = 100, double tol = 1e-5) {
    int n = y.n_elem;
    int nx = X.n_cols;
    int nz = Z.n_cols;
    int n_genotypes = G_genomic.n_rows;
    
    // Precompute Genomic Inverse
    arma::mat G_genomic_inv = arma::pinv(G_genomic);
    
    // Initial variance guesses
    double var_e = arma::var(y) / 2.0;
    arma::mat G_env = arma::eye(n_envs, n_envs) * (arma::var(y) / 2.0);
    
    arma::mat XtX = X.t() * X;
    arma::mat XtZ = X.t() * Z;
    arma::mat ZtX = Z.t() * X;
    arma::mat ZtZ = Z.t() * Z;
    
    arma::vec b, u;

    for(int iter = 0; iter < max_iter; iter++) {
        arma::mat G_env_inv = arma::pinv(G_env);
        
        // Kronecker structure: G_env (x) G_genomic
        arma::mat G_full_inv = arma::kron(G_env_inv, G_genomic_inv);

        // Build LHS using pinv for stability in singular systems
        arma::mat LHS = arma::join_cols(
            arma::join_rows(XtX, XtZ),
            arma::join_rows(ZtX, ZtZ + (G_full_inv * var_e))
        );
        arma::mat RHS = arma::join_cols(X.t() * y, Z.t() * y);
        arma::mat sol = arma::pinv(LHS) * RHS;
        
        b = sol.rows(0, nx - 1);
        u = sol.rows(nx, nx + nz - 1);

        // Update var_e
        var_e = arma::as_scalar((y - X*b - Z*u).t() * (y - X*b - Z*u)) / (n - nx);
        
        // Update G_env (Simplified EM for Env Covariance with Genomic constraints)
        arma::mat U_mat = arma::reshape(u, n_genotypes, n_envs); 
        arma::mat U_scaled = U_mat.t() * G_genomic_inv * U_mat;
        
        arma::mat C_inv = arma::pinv(LHS);
        arma::mat C22 = C_inv.submat(nx, nx, nx + nz - 1, nx + nz - 1);
        
        // Approximate Trace Update for G_env
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

    return Rcpp::List::create(
        Rcpp::Named(\"BLUEs\") = b, 
        Rcpp::Named(\"GBLUPs\") = u, 
        Rcpp::Named(\"G_env\") = G_env, 
        Rcpp::Named(\"var_e\") = var_e
    );
}
"
sourceCpp(code = cpp_code)

# ------------------------------------------------------------------------------
# 3. Independent R Interface (The "Mouth")
# ------------------------------------------------------------------------------
fit_genomic_mme <- function(fixed, random, data, G_mat, n_envs) {
  resp_name <- all.vars(fixed)[1]
  y_mat <- as.matrix(data[[resp_name]])
  
  X <- model.matrix(fixed, data)
  
  # Ensure Z matrix column order strictly matches Environment (outer) x Genotype (inner)
  # This aligns with the kronecker product in the C++ engine
  Z <- model.matrix(random, data)
  ordered_levels <- paste(rep(paste0("E", 1:n_envs), each = nrow(G_mat)), 
                          rep(rownames(G_mat), times = n_envs), sep = ":")
  
  # Reorder Z columns to match the Kronecker layout
  # Strip 'Env_Geno' prefix from model.matrix colnames for matching
  clean_Z_colnames <- gsub(all.vars(random)[1], "", colnames(Z))
  Z <- Z[, match(ordered_levels, clean_Z_colnames)]
  
  results <- solve_genomic_mme_cpp(X, Z, y_mat, G_mat, n_envs)
  rownames(results$BLUEs) <- colnames(X)
  
  # Format GBLUPs as a nested list to mimic sommer extraction syntax
  gblups_list <- list()
  gblups_vector <- as.numeric(results$GBLUPs)
  names(gblups_vector) <- ordered_levels
  
  gblups_list$U$`rr(Environment, d = 2):Genotype`$Yield <- gblups_vector
  
  return(list(
    BLUEs = results$BLUEs,
    GBLUPs_raw = results$GBLUPs,
    U = gblups_list$U,
    G_env = results$G_env,
    var_e = results$var_e
  ))
}

# ------------------------------------------------------------------------------
# 4. Execution and Extracting Genomic BLUPs
# ------------------------------------------------------------------------------
cat("\nFitting Independent Single-Step Genomic MET Model...\n")
g_met_model <- fit_genomic_mme(
  fixed = Yield ~ Environment,
  random = ~ 0 + Env_Geno,
  data = met_data,
  G_mat = G_matrix,
  n_envs = n_envs
)

# Extract the BLUPs (Estimated Genetic Values) analogous to sommer
gblups <- g_met_model$U$`rr(Environment, d = 2):Genotype`$Yield

# Preview the first 10 GBLUPs
cat("\n--- First 10 Extracted GBLUPs ---\n")
print(head(gblups, 10))