# Load required packages for C++ integration
library(Rcpp)

# ---------------------------------------------------------
# Step A: Define the C++ Engine (The "Body")
# ---------------------------------------------------------
# We use RcppArmadillo to handle the heavy matrix algebra.
# It builds the Left-Hand Side (LHS) and Right-Hand Side (RHS) 
# of Henderson's Mixed Model Equations and solves them.
cpp_engine <- "
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

// [[Rcpp::export]]
Rcpp::List solve_mme_cpp(const arma::mat& X, const arma::mat& Z, const arma::vec& y, double var_g, double var_e) {
    
    // Calculate the variance ratio (lambda)
    double lambda = var_e / var_g;
    
    int nx = X.n_cols;
    int nz = Z.n_cols;

    // 1. Build the Left Hand Side (LHS) components
    arma::mat XtX = X.t() * X;
    arma::mat XtZ = X.t() * Z;
    arma::mat ZtX = Z.t() * X;
    
    // Z'Z + I * lambda
    arma::mat ZtZ = Z.t() * Z;
    arma::mat I_lambda = arma::eye(nz, nz) * lambda;
    arma::mat ZtZ_inv = ZtZ + I_lambda;

    // 2. Combine into the full LHS matrix
    arma::mat LHS = arma::join_cols(
        arma::join_rows(XtX, XtZ),
        arma::join_rows(ZtX, ZtZ_inv)
    );

    // 3. Build the Right Hand Side (RHS) components
    arma::mat Xty = X.t() * y;
    arma::mat Zty = Z.t() * y;
    arma::mat RHS = arma::join_cols(Xty, Zty);

    // 4. Solve the system of equations
    arma::mat solutions = arma::solve(LHS, RHS);

    // 5. Separate solutions into Fixed (b) and Random (u) effects
    arma::vec b = solutions.rows(0, nx - 1);
    arma::vec u = solutions.rows(nx, nx + nz - 1);

    return Rcpp::List::create(
        Rcpp::Named(\"BLUEs\") = b,
        Rcpp::Named(\"BLUPs\") = u
    );
}
"
# Compile the C++ function into the R environment
sourceCpp(code = cpp_engine)

# ---------------------------------------------------------
# Step B: Define the R Interface 
# ---------------------------------------------------------
# This function takes standard R formulas, converts them into 
# mathematical matrices, and passes them to the C++ engine.
fit_mme <- function(fixed, random, data, var_g, var_e) {
  
  # 1. Extract the Response Vector (y)
  resp_name <- all.vars(fixed)[1]
  y_mat <- as.matrix(data[[resp_name]])
  
  # 2. Build the Fixed Effects Design Matrix (X)
  mf_fixed <- model.frame(fixed, data = data)
  X <- model.matrix(fixed, mf_fixed)
  
  # Clean up X column names for cleaner output
  fixed_term <- all.vars(fixed)[2]
  colnames(X) <- gsub(fixed_term, "", colnames(X)) 
  
  # 3. Build the Random Effects Design Matrix (Z)
  # We suppress the intercept in Z to ensure it only maps group levels
  random_form <- as.formula(paste("~ 0 +", all.vars(random)[1]))
  Z <- model.matrix(random_form, data)
  
  # Clean up Z column names
  random_term <- all.vars(random)[1]
  colnames(Z) <- gsub(random_term, "", colnames(Z))
  
  # 4. Pass the dense matrices to the fast C++ solver
  results <- solve_mme_cpp(
    X = as.matrix(X), 
    Z = as.matrix(Z), 
    y = y_mat, 
    var_g = var_g, 
    var_e = var_e
  )
  
  # 5. Re-attach the row names to the outputs
  rownames(results$BLUEs) <- colnames(X)
  rownames(results$BLUPs) <- colnames(Z)
  
  return(results)
}

# Example Application
# ---------------------------------------------------------
# 1. Simulate the Data
# ---------------------------------------------------------
set.seed(42)

# 3 Environments, 10 Genotypes, 3 Replicates per genotype/environment
dat <- expand.grid(
  Rep = 1:3,
  Geno = paste0("Variety_", 1:10),
  Env = c("Farm_North", "Farm_South", "Farm_West")
)

# True Fixed Effects for the Farms
true_env <- c(Farm_North = 50, Farm_South = 65, Farm_West = 55)

# True Random Genetic Effects for the Varieties (mean = 0, variance = 16)
true_geno <- rnorm(10, mean = 0, sd = 4) 
names(true_geno) <- paste0("Variety_", 1:10)

# Simulate Yield (with residual error variance of ~4)
dat$Yield <- true_env[dat$Env] + true_geno[dat$Geno] + rnorm(nrow(dat), mean = 0, sd = 2)

# ---------------------------------------------------------
# 2. Run the Custom Solver
# ---------------------------------------------------------
# Assume historical data tells us genetic variance is ~16 and error is ~4
model_results <- fit_mme(
  fixed  = Yield ~ Env,      # Fixed formula
  random = ~ Geno,           # Random formula
  data   = dat,
  var_g  = 16.0,             # Known genetic variance (sigma^2_g)
  var_e  = 4.0               # Known error variance (sigma^2_e)
)

# ---------------------------------------------------------
# 3. View the Results
# ---------------------------------------------------------
cat("\n--- Estimated Fixed Environmental Effects (BLUEs) ---\n")
print(model_results$BLUEs)

cat("\n--- Predicted Genetic Values (BLUPs) ---\n")
# Rank the varieties from best to worst based on their BLUPs
sorted_blups <- model_results$BLUPs[order(model_results$BLUPs[,1], decreasing = TRUE), , drop = FALSE]
print(sorted_blups)

# ---------------------------------------------------------
# 4. Validation against the 'sommer' package
# ---------------------------------------------------------
# Load sommer for comparison
library(sommer)

cat("\nFitting model with sommer (this may take a moment)...\n")

# Run the exact same model in sommer
fit_sommer <- mmer(
  fixed = Yield ~ Env,
  random = ~ Geno,
  rcov = ~ units,
  data = dat,
  verbose = FALSE
)

# Extract the variance components sommer calculated via REML
vc_table <- summary(fit_sommer)$varcomp
var_g_sommer <- vc_table[1, "VarComp"] # Genotype variance
var_e_sommer <- vc_table[2, "VarComp"] # Residual (units) variance

cat("Sommer's Optimized Genetic Variance:", var_g_sommer, "\n")
cat("Sommer's Optimized Error Variance:", var_e_sommer, "\n")

# Re-run our custom solver using sommer's exact variance components
# so we are comparing apples to apples
model_results_sommer_vc <- fit_mme(
  fixed  = Yield ~ Env,
  random = ~ Geno,
  data   = dat,
  var_g  = var_g_sommer,
  var_e  = var_e_sommer
)

# ---------------------------------------------------------
# 5. Side-by-Side Comparison
# ---------------------------------------------------------
cat("\n--- Fixed Effects (BLUEs) Validation ---\n")
blue_val <- data.frame(
  Custom_Engine = as.vector(model_results_sommer_vc$BLUEs),
  Sommer_Package = fit_sommer$Beta$Estimate
)
rownames(blue_val) <- rownames(model_results_sommer_vc$BLUEs)
print(blue_val)

cat("\n--- Random Effects (BLUPs) Validation ---\n")
# Extract BLUPs from sommer (indexing the first random effect list)
blup_val <- data.frame(
  Custom_Engine = as.vector(model_results_sommer_vc$BLUPs),
  Sommer_Package = fit_sommer$U[[1]]$Yield
)
rownames(blup_val) <- rownames(model_results_sommer_vc$BLUPs)
print(blup_val)
