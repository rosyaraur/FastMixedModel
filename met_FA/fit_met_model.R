# ==============================================================================
# UNIFIED MET SOLVER: GENOMIC & ENVIROTYPIC INTEGRATION
# ==============================================================================

library(Rcpp)
library(Matrix)

# ------------------------------------------------------------------------------
# 1. Simulate Unified Data (Genotypes, Phenotypes, and Envirotypes)
# ------------------------------------------------------------------------------
set.seed(2026)

n_genotypes <- 100
n_envs <- 4
n_markers <- 200

# A. Simulate and Calculate Genomic Matrix (G)
M <- matrix(rbinom(n_genotypes * n_markers, 2, 0.5), nrow = n_genotypes, ncol = n_markers)
rownames(M) <- paste0("G", 1:n_genotypes)
M_centered <- scale(M, center = TRUE, scale = FALSE)
scaling_G <- 2 * mean(colMeans(M / 2) * (1 - colMeans(M / 2)))
G_matrix <- (M_centered %*% t(M_centered)) / scaling_G

# B. Simulate and Calculate Environmental Relationship Matrix (E)
n_covariates <- 3
days <- 100
env_data <- matrix(rnorm(n_envs * days * n_covariates), nrow = n_envs, ncol = days * n_covariates)
rownames(env_data) <- paste0("E", 1:n_envs)
W_centered <- scale(env_data, center = TRUE, scale = TRUE)
E_matrix <- (W_centered %*% t(W_centered)) / ncol(W_centered)

# C. Simulate Phenotypic Data
met_data <- expand.grid(Genotype = rownames(G_matrix), Environment = paste0("E", 1:n_envs))
met_data$Yield <- 20 + rnorm(nrow(met_data), 0, 5)
met_data$Env_Geno <- factor(paste(met_data$Environment, met_data$Genotype, sep = ":"))

# ------------------------------------------------------------------------------
# 2. C++ Backend (Contains both Standard and Envirotypic Engines)
# ------------------------------------------------------------------------------
cpp_code <- "
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <string>

// --- ENGINE 1: Standard MET (Estimates G_env via EM or NR) ---
// [[Rcpp::export]]
Rcpp::List solve_standard_mme_cpp(const arma::mat& X, const arma::mat& Z, 
                                 const arma::vec& y, const arma::mat& G_genomic, 
                                 int n_envs, std::string method = \"EM\",
                                 int max_iter = 100, double tol = 1e-5) {
    int n = y.n_elem;
    int nx = X.n_cols;
    int nz = Z.n_cols;
    int n_genotypes = G_genomic.n_rows;
    
    arma::mat G_genomic_inv = arma::pinv(G_genomic);
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
        if (method == \"EM\") {
            arma::mat G_env_inv = arma::pinv(G_env);
            arma::mat G_full_inv = arma::kron(G_env_inv, G_genomic_inv);
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
        } else if (method == \"NR\") {
            arma::mat G_full = arma::kron(G_env, G_genomic);
            arma::mat V = Z * G_full * Z.t() + arma::eye(n, n) * var_e;
            arma::mat V_inv = arma::pinv(V);
            arma::mat Xt_Vinv_X = X.t() * V_inv * X;
            arma::mat P = V_inv - V_inv * X * arma::pinv(Xt_Vinv_X) * X.t() * V_inv;
            
            b = arma::pinv(Xt_Vinv_X) * X.t() * V_inv * y;
            u = G_full * Z.t() * P * y;
            
            double score_e = -0.5 * arma::trace(P) + 0.5 * arma::as_scalar(y.t() * P * P * y);
            double info_e = 0.5 * arma::trace(P * P);
            var_e = var_e + (score_e / info_e);
            if(var_e < 1e-6) var_e = 1e-6; 
            
            arma::mat U_mat = arma::reshape(u, n_genotypes, n_envs);
            arma::mat U_scaled = U_mat.t() * G_genomic_inv * U_mat;
            arma::mat Zt_P_Z = Z.t() * P * Z;
            arma::rowvec trace_PZ = arma::sum(arma::reshape(Zt_P_Z.diag(), n_genotypes, n_envs), 0);
            
            for(int i = 0; i < n_envs; ++i) {
                G_env(i,i) = (U_scaled(i,i) + trace_PZ(i)) / n_genotypes;
            }
        }
    }
    return Rcpp::List::create(Rcpp::Named(\"BLUEs\") = b, Rcpp::Named(\"GBLUPs\") = u, 
                              Rcpp::Named(\"G_env\") = G_env, Rcpp::Named(\"var_e\") = var_e,
                              Rcpp::Named(\"iterations\") = iter_used);
}

// --- ENGINE 2: Envirotypic MET (Uses E Matrix, Estimates var_ge via EM) ---
// [[Rcpp::export]]
Rcpp::List solve_envirotypic_mme_cpp(const arma::mat& X, const arma::mat& Z, 
                                     const arma::vec& y, const arma::mat& G_genomic, 
                                     const arma::mat& E_matrix, 
                                     int max_iter = 100, double tol = 1e-5) {
    int n = y.n_elem; int nx = X.n_cols; int nz = Z.n_cols;
    
    arma::mat EG_full_inv = arma::kron(arma::pinv(E_matrix), arma::pinv(G_genomic));
    double var_e = arma::var(y) / 2.0;
    double var_ge = arma::var(y) / 2.0; 
    
    arma::mat XtX = X.t() * X; arma::mat XtZ = X.t() * Z;
    arma::mat ZtX = Z.t() * X; arma::mat ZtZ = Z.t() * Z;
    arma::vec b, u;

    for(int iter = 0; iter < max_iter; iter++) {
        double lambda = var_e / var_ge;
        arma::mat LHS = arma::join_cols(
            arma::join_rows(XtX, XtZ),
            arma::join_rows(ZtX, ZtZ + (EG_full_inv * lambda))
        );
        arma::mat RHS = arma::join_cols(X.t() * y, Z.t() * y);
        arma::mat C_inv = arma::pinv(LHS);
        arma::mat sol = C_inv * RHS;
        
        b = sol.rows(0, nx - 1);
        u = sol.rows(nx, nx + nz - 1);

        arma::vec residuals = y - X*b - Z*u;
        var_e = arma::as_scalar(residuals.t() * residuals) / (n - nx);
        
        arma::mat C22 = C_inv.submat(nx, nx, nx + nz - 1, nx + nz - 1);
        double trace_term = arma::trace(C22 * EG_full_inv);
        var_ge = (arma::as_scalar(u.t() * EG_full_inv * u) + (trace_term * var_e)) / nz;
    }
    return Rcpp::List::create(Rcpp::Named(\"BLUEs\") = b, Rcpp::Named(\"GBLUPs\") = u, 
                              Rcpp::Named(\"var_ge\") = var_ge, Rcpp::Named(\"var_e\") = var_e);
}
"
sourceCpp(code = cpp_code)

# ------------------------------------------------------------------------------
# 3. Master R Wrapper Router (UPDATED)
# ------------------------------------------------------------------------------
fit_met_model <- function(fixed, random, data, G_mat, E_mat = NULL, n_envs = NULL, method = "EM") {
  
  # Dynamically extract response and environment column names from the fixed formula
  resp_col <- all.vars(fixed)[1]
  env_col <- all.vars(fixed)[2] 
  y_mat <- as.matrix(data[[resp_col]])
  
  X <- model.matrix(fixed, data)
  Z <- model.matrix(random, data)
  
  is_envirotypic <- !is.null(E_mat)
  
  if(is_envirotypic) {
    # --- Envirotypic Routing ---
    ordered_levels <- paste(rep(rownames(E_mat), each = nrow(G_mat)), 
                            rep(rownames(G_mat), times = nrow(E_mat)), sep = ":")
    
    clean_Z_colnames <- gsub(all.vars(random)[1], "", colnames(Z))
    Z <- Z[, match(ordered_levels, clean_Z_colnames)]
    
    if(any(is.na(Z))) stop("NAs generated in Z matrix. Check E_mat rownames and data levels.")
    
    res <- solve_envirotypic_mme_cpp(X, Z, y_mat, G_mat, E_mat)
    
    gblups_vec <- as.numeric(res$GBLUPs)
    names(gblups_vec) <- ordered_levels
    
    return(list(Type = "Envirotypic (E x G)", BLUEs = res$BLUEs, GBLUPs = gblups_vec, 
                Var_GxE = res$var_ge, Var_Residual = res$var_e))
    
  } else {
    # --- Standard Routing ---
    # Dynamically extract the exact environment levels present in this subset of data
    env_levels <- sort(unique(as.character(data[[env_col]])))
    n_envs_actual <- length(env_levels)
    
    ordered_levels <- paste(rep(env_levels, each = nrow(G_mat)), 
                            rep(rownames(G_mat), times = n_envs_actual), sep = ":")
    
    clean_Z_colnames <- gsub(all.vars(random)[1], "", colnames(Z))
    Z <- Z[, match(ordered_levels, clean_Z_colnames)]
    
    if(any(is.na(Z))) stop("NAs generated in Z matrix. Check data factor levels.")
    
    res <- solve_standard_mme_cpp(X, Z, y_mat, G_mat, n_envs_actual, method = method)
    
    gblups_vec <- as.numeric(res$GBLUPs)
    names(gblups_vec) <- ordered_levels
    
    return(list(Type = paste("Standard GxE -", method), BLUEs = res$BLUEs, GBLUPs = gblups_vec, 
                G_env = res$G_env, Var_Residual = res$var_e))
  }
}
# ------------------------------------------------------------------------------
# 4. Execution Examples
# ------------------------------------------------------------------------------
cat("\n--- Running Standard Model (EM) ---\n")
model_standard_em <- fit_met_model(
  fixed = Yield ~ Environment, random = ~ 0 + Env_Geno, 
  data = met_data, G_mat = G_matrix, n_envs = n_envs, method = "EM"
)
print(model_standard_em$Type)

cat("\n--- Running Standard Model (NR) ---\n")
model_standard_nr <- fit_met_model(
  fixed = Yield ~ Environment, random = ~ 0 + Env_Geno, 
  data = met_data, G_mat = G_matrix, n_envs = n_envs, method = "NR"
)
print(model_standard_nr$Type)

cat("\n--- Running Envirotypic Model (E x G) ---\n")
model_envirotypic <- fit_met_model(
  fixed = Yield ~ Environment, random = ~ 0 + Env_Geno, 
  data = met_data, G_mat = G_matrix, E_mat = E_matrix
)
print(model_envirotypic$Type)
print(paste("Estimated GxE Variance (var_ge):", round(model_envirotypic$Var_GxE, 4)))


# ==============================================================================
# LEAVE-ONE-ENVIRONMENT-OUT (LOEO) CROSS-VALIDATION
# ==============================================================================

cat("\n======================================================\n")
cat("          ENVIROTYPIC CROSS-VALIDATION (LOEO)           \n")
cat("======================================================\n\n")

# Initialize a data frame to store prediction accuracies
cv_results <- data.frame(
  Untested_Environment = character(),
  Standard_Accuracy = numeric(),
  Envirotypic_Accuracy = numeric(),
  stringsAsFactors = FALSE
)

# Loop through each environment, treating it as the "untested" future environment
environments <- paste0("E", 1:n_envs)

for (target_env in environments) {
  
  cat(sprintf("Masking %s... Training on remaining environments.\n", target_env))
  
  # 1. Split Data into Training and Testing
  train_data <- met_data[met_data$Environment != target_env, ]
  test_data <- met_data[met_data$Environment == target_env, ]
  
  # Sort test data by Genotype to ensure alignment during correlation
  test_data <- test_data[order(test_data$Genotype), ]
  true_yield <- test_data$Yield
  
  # Ensure factors drop the masked level for clean model matrix generation
  train_data$Environment <- factor(train_data$Environment)
  train_data$Env_Geno <- factor(train_data$Env_Geno)
  
  # 2. Subset the E matrix for the Envirotypic Model
  train_envs <- environments[environments != target_env]
  E_train <- E_matrix[train_envs, train_envs]
  E_test_train <- E_matrix[target_env, train_envs] # Covariance between test and train
  
  # ----------------------------------------------------------------------------
  # Model A: Standard MET Model (Baseline)
  # ----------------------------------------------------------------------------
  mod_std <- fit_met_model(
    fixed = Yield ~ Environment, 
    random = ~ 0 + Env_Geno, 
    data = train_data, 
    G_mat = G_matrix, 
    n_envs = length(train_envs), 
    method = "EM"
  )
  
  # Standard Prediction: Average the GBLUPs across the tested environments
  U_mat_std <- matrix(mod_std$GBLUPs, nrow = n_genotypes, ncol = length(train_envs), byrow = FALSE)
  pred_std <- rowMeans(U_mat_std) 
  
  # ----------------------------------------------------------------------------
  # Model B: Envirotypic Model (E x G Projection)
  # ----------------------------------------------------------------------------
  mod_env <- fit_met_model(
    fixed = Yield ~ Environment, 
    random = ~ 0 + Env_Geno, 
    data = train_data, 
    G_mat = G_matrix, 
    E_mat = E_train
  )
  
  # Envirotypic Prediction: Project GBLUPs using the Environmental Covariances
  U_mat_env <- matrix(mod_env$GBLUPs, nrow = n_genotypes, ncol = length(train_envs), byrow = FALSE)
  
  # Apply projection formula: u_test = U_train %*% E_train_inv %*% E_test_train
  E_train_inv <- solve(E_train)
  projection_weights <- as.numeric(E_test_train %*% E_train_inv)
  
  # Multiply the genotype responses in the tested environments by the projection weights
  pred_env <- as.numeric(U_mat_env %*% projection_weights)
  
  # ----------------------------------------------------------------------------
  # Calculate Prediction Accuracies (Pearson Correlation)
  # ----------------------------------------------------------------------------
  # Correlate predicted genetic merit with the actual observed phenotypic yield
  acc_std <- cor(pred_std, true_yield)
  acc_env <- cor(pred_env, true_yield)
  
  # Append to results
  cv_results <- rbind(cv_results, data.frame(
    Untested_Environment = target_env,
    Standard_Accuracy = round(acc_std, 4),
    Envirotypic_Accuracy = round(acc_env, 4)
  ))
}

# ------------------------------------------------------------------------------
# View the Final Cross-Validation Results
# ------------------------------------------------------------------------------
cat("\n--- Prediction Accuracy for Untested Environments (CV0) ---\n")
print(cv_results)

cat("\n--- Summary ---\n")
mean_std <- mean(cv_results$Standard_Accuracy)
mean_env <- mean(cv_results$Envirotypic_Accuracy)

summary_df <- data.frame(
  Model = c("Standard (GxE Average)", "Envirotypic (E x G Projection)"),
  Mean_Accuracy = c(round(mean_std, 4), round(mean_env, 4))
)
print(summary_df)