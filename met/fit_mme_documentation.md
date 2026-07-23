## `fit_mme`: Fast Mixed Model Equation Solver via Rcpp

### Description

The `fit_mme` function acts as a streamlined interface for solving Henderson's Mixed Model Equations (MMEs). By parsing standard R formulas into design matrices ($X$ and $Z$) and passing them to a highly optimized C++ backend (`RcppArmadillo`), it rapidly computes Best Linear Unbiased Estimators (BLUEs) for fixed effects and Best Linear Unbiased Predictors (BLUPs) for a single random effect.

### Usage

```r
fit_mme(fixed, random, data, var_g, var_e)

```

### Arguments

* **`fixed`**: A two-sided linear formula object describing the fixed effects (e.g., `Yield ~ Env`).
* **`random`**: A one-sided formula object describing the random effect (e.g., `~ Geno`).
* **`data`**: A data frame containing the variables named in the `fixed` and `random` formulas.
* **`var_g`**: A numeric value specifying the known or previously estimated genetic (or random effect) variance, $\sigma^2_g$.
* **`var_e`**: A numeric value specifying the known or previously estimated residual error variance, $\sigma^2_e$.

### Value

Returns a named list containing two elements:

* **`BLUEs`**: A matrix of the Best Linear Unbiased Estimators for the fixed effects.
* **`BLUPs`**: A matrix of the Best Linear Unbiased Predictors for the random effects.

### Details

The function solves the following system of equations:

$$\begin{bmatrix} X'X & X'Z \\ Z'X & Z'Z + I\lambda \end{bmatrix} \begin{bmatrix} b \\ u \end{bmatrix} = \begin{bmatrix} X'y \\ Z'y \end{bmatrix}$$

Where $X$ is the fixed effects design matrix, $Z$ is the random effects design matrix, $y$ is the response vector, $b$ represents the fixed effect estimates, and $u$ represents the random effect predictions. The penalization term is defined as $\lambda = \sigma^2_e / \sigma^2_g$.

---

### Setup & Implementation

To use `fit_mme`, you must first compile the C++ engine using the `Rcpp` package.

```r
library(Rcpp)

# 1. Compile the C++ Engine
cpp_engine <- "
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

// [[Rcpp::export]]
Rcpp::List solve_mme_cpp(const arma::mat& X, const arma::mat& Z, const arma::vec& y, double var_g, double var_e) {
    double lambda = var_e / var_g;
    int nx = X.n_cols;
    int nz = Z.n_cols;

    arma::mat XtX = X.t() * X;
    arma::mat XtZ = X.t() * Z;
    arma::mat ZtX = Z.t() * X;
    arma::mat ZtZ = Z.t() * Z;
    
    arma::mat I_lambda = arma::eye(nz, nz) * lambda;
    arma::mat ZtZ_inv = ZtZ + I_lambda;

    arma::mat LHS = arma::join_cols(
        arma::join_rows(XtX, XtZ),
        arma::join_rows(ZtX, ZtZ_inv)
    );

    arma::mat RHS = arma::join_cols(X.t() * y, Z.t() * y);
    arma::mat solutions = arma::solve(LHS, RHS);

    return Rcpp::List::create(
        Rcpp::Named(\"BLUEs\") = solutions.rows(0, nx - 1),
        Rcpp::Named(\"BLUPs\") = solutions.rows(nx, nx + nz - 1)
    );
}
"
sourceCpp(code = cpp_engine)

# 2. Define the R Wrapper
fit_mme <- function(fixed, random, data, var_g, var_e) {
  resp_name <- all.vars(fixed)[1]
  y_mat <- as.matrix(data[[resp_name]])
  
  mf_fixed <- model.frame(fixed, data = data)
  X <- model.matrix(fixed, mf_fixed)
  colnames(X) <- gsub(all.vars(fixed)[2], "", colnames(X)) 
  
  random_form <- as.formula(paste("~ 0 +", all.vars(random)[1]))
  Z <- model.matrix(random_form, data)
  colnames(Z) <- gsub(all.vars(random)[1], "", colnames(Z))
  
  results <- solve_mme_cpp(as.matrix(X), as.matrix(Z), y_mat, var_g, var_e)
  rownames(results$BLUEs) <- colnames(X)
  rownames(results$BLUPs) <- colnames(Z)
  
  return(results)
}

```

---

### Example: Agricultural Plant Breeding and `sommer` Validation

In this use case, we simulate an agricultural trial evaluating 10 crop varieties across 3 environments. We first fit the model with `fit_mme` and then validate the exact results using the `sommer` package.

```r
# 1. Simulate the Data
set.seed(42)
dat <- expand.grid(
  Rep = 1:3,
  Geno = paste0("Variety_", 1:10),
  Env = c("Farm_North", "Farm_South", "Farm_West")
)

true_env <- c(Farm_North = 50, Farm_South = 65, Farm_West = 55)
true_geno <- rnorm(10, mean = 0, sd = 4) 
names(true_geno) <- paste0("Variety_", 1:10)

dat$Yield <- true_env[dat$Env] + true_geno[dat$Geno] + rnorm(nrow(dat), mean = 0, sd = 2)

# 2. Extract Variance Components via sommer
library(sommer)
fit_sommer <- mmer(
  fixed = Yield ~ Env,
  random = ~ Geno,
  rcov = ~ units,
  data = dat,
  verbose = FALSE
)

vc_table <- summary(fit_sommer)$varcomp
var_g_sommer <- vc_table[1, "VarComp"]
var_e_sommer <- vc_table[2, "VarComp"]

# 3. Run Custom C++ Solver
model_results <- fit_mme(
  fixed  = Yield ~ Env,
  random = ~ Geno,
  data   = dat,
  var_g  = var_g_sommer,
  var_e  = var_e_sommer
)

# 4. Validate and Compare
cat("\n--- Fixed Effects (BLUEs) Validation ---\n")
blue_val <- data.frame(
  Custom_Engine = as.vector(model_results$BLUEs),
  Sommer_Package = fit_sommer$Beta$Estimate
)
rownames(blue_val) <- rownames(model_results$BLUEs)
print(blue_val)

cat("\n--- Random Effects (BLUPs) Validation ---\n")
blup_val <- data.frame(
  Custom_Engine = as.vector(model_results$BLUPs),
  Sommer_Package = fit_sommer$U[[1]]$Yield
)
rownames(blup_val) <- rownames(model_results$BLUPs)
print(blup_val)

```