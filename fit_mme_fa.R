library(Rcpp)

# ---------------------------------------------------------
# Step A: The Independent C++ Engine (EM Algorithm)
# ---------------------------------------------------------
cpp_engine_independent <- "
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

// [[Rcpp::export]]
Rcpp::List solve_mme_em_cpp(const arma::mat& X, const arma::mat& Z, const arma::vec& y, int max_iter = 500, double tol = 1e-6) {
    
    int n = y.n_elem;
    int nx = X.n_cols;
    int nz = Z.n_cols;

    // Pre-compute static matrix cross-products
    arma::mat XtX = X.t() * X;
    arma::mat XtZ = X.t() * Z;
    arma::mat ZtX = Z.t() * X;
    arma::mat ZtZ = Z.t() * Z;
    
    arma::mat Xty = X.t() * y;
    arma::mat Zty = Z.t() * y;
    arma::mat RHS = arma::join_cols(Xty, Zty);
    double yty = arma::as_scalar(y.t() * y);

    // Initial naive guesses for variance components
    double var_e = arma::var(y) / 2.0;
    double var_g = var_e;

    arma::vec b(nx, arma::fill::zeros);
    arma::vec u(nz, arma::fill::zeros);
    
    // Expectation-Maximization (EM) Loop
    for(int iter = 0; iter < max_iter; iter++) {
        double old_var_g = var_g;
        double old_var_e = var_e;
        
        // 1. Calculate lambda
        double lambda = var_e / var_g;

        // 2. Build LHS
        arma::mat I_lambda = arma::eye(nz, nz) * lambda;
        arma::mat ZtZ_inv = ZtZ + I_lambda;
        arma::mat LHS = arma::join_cols(
            arma::join_rows(XtX, XtZ),
            arma::join_rows(ZtX, ZtZ_inv)
        );

        // 3. Invert LHS to get the C matrix (Coefficient Matrix)
        arma::mat C_inv = arma::pinv(LHS); 
        
        // 4. Solve for b and u
        arma::mat solutions = C_inv * RHS;
        b = solutions.rows(0, nx - 1);
        u = solutions.rows(nx, nx + nz - 1);

        // 5. Extract C22 (the random effects portion of the inverse)
        arma::mat C22 = C_inv.submat(nx, nx, nx + nz - 1, nx + nz - 1);
        double tr_C22 = arma::trace(C22);

        // 6. Henderson's EM Updates for Variance Components
        double bXty = arma::as_scalar(b.t() * Xty);
        double uZty = arma::as_scalar(u.t() * Zty);
        
        var_e = (yty - bXty - uZty) / (n - nx);
        
        double utu = arma::as_scalar(u.t() * u);
        var_g = (utu + var_e * tr_C22) / nz;

        // 7. Convergence Check
        if (std::abs(var_g - old_var_g) < tol && std::abs(var_e - old_var_e) < tol) {
            break;
        }
    }

    return Rcpp::List::create(
        Rcpp::Named(\"BLUEs\") = b,
        Rcpp::Named(\"BLUPs\") = u,
        Rcpp::Named(\"var_g\") = var_g,
        Rcpp::Named(\"var_e\") = var_e
    );
}
"
sourceCpp(code = cpp_engine_independent)

# ---------------------------------------------------------
# Step B: The Independent R Interface
# ---------------------------------------------------------
# Notice this function no longer takes var_g or var_e as inputs.
fit_mme_fa <- function(fixed, random, data) {
  
  resp_name <- all.vars(fixed)[1]
  y_mat <- as.matrix(data[[resp_name]])
  
  mf_fixed <- model.frame(fixed, data = data)
  X <- model.matrix(fixed, mf_fixed)
  colnames(X) <- gsub(all.vars(fixed)[2], "", colnames(X))
  
  random_form <- as.formula(paste("~ 0 +", all.vars(random)[1]))
  Z <- model.matrix(random_form, data)
  colnames(Z) <- gsub(all.vars(random)[1], "", colnames(Z))
  
  # Pass only data to C++ to solve completely independently
  results <- solve_mme_em_cpp(as.matrix(X), as.matrix(Z), y_mat)
  
  rownames(results$BLUEs) <- colnames(X)
  rownames(results$BLUPs) <- colnames(Z)
  
  return(results)
}

library(sommer)

# ---------------------------------------------------------
# 1. Simulate the Dataset
# ---------------------------------------------------------
set.seed(42)
dat <- expand.grid(
  Rep = 1:5,
  Geno = paste0("Variety_", 1:10),
  Env = c("Farm_North", "Farm_South")
)

true_env <- c(Farm_North = 50, Farm_South = 65)
true_geno <- rnorm(10, mean = 0, sd = 4) 
names(true_geno) <- paste0("Variety_", 1:10)
dat$Yield <- true_env[dat$Env] + true_geno[dat$Geno] + rnorm(nrow(dat), mean = 0, sd = 2)

# ---------------------------------------------------------
# 2. Run the Independent Custom Engine
# ---------------------------------------------------------
cat("Running Custom Independent Engine (EM Algorithm)...\n")
fit_custom <- fit_mme_fa(
  fixed  = Yield ~ Env,
  random = ~ Geno,
  data   = dat
)

# ---------------------------------------------------------
# 3. Run Sommer (AI-REML Algorithm)
# ---------------------------------------------------------
cat("Running Sommer Engine (AI-REML Algorithm)...\n\n")
fit_sommer <- mmer(
  fixed = Yield ~ Env,
  random = ~ Geno,
  rcov = ~ units,
  data = dat,
  verbose = FALSE
)
vc_table <- summary(fit_sommer)$varcomp

# ---------------------------------------------------------
# 4. Side-by-Side Comparison
# ---------------------------------------------------------
cat("--- 1. Variance Component Comparison ---\n")
# Minor differences are expected here because EM and AI-REML traverse 
# the likelihood surface slightly differently, but they will converge very closely.
var_comp <- data.frame(
  Custom_EM = c(fit_custom$var_g, fit_custom$var_e),
  Sommer_AI = c(vc_table[1, "VarComp"], vc_table[2, "VarComp"]),
  row.names = c("Genetic_Variance (var_g)", "Error_Variance (var_e)")
)
print(var_comp)

cat("\n--- 2. Fixed Effects (BLUEs) Comparison ---\n")
blue_val <- data.frame(
  Custom_EM = as.vector(fit_custom$BLUEs),
  Sommer_AI = fit_sommer$Beta$Estimate
)
rownames(blue_val) <- rownames(fit_custom$BLUEs)
print(blue_val)

cat("\n--- 3. Random Effects (BLUPs) Comparison (First 5) ---\n")
blup_val <- data.frame(
  Custom_EM = as.vector(fit_custom$BLUPs)[1:5],
  Sommer_AI = fit_sommer$U[[1]]$Yield[1:5]
)
rownames(blup_val) <- rownames(fit_custom$BLUPs)[1:5]
print(blup_val)