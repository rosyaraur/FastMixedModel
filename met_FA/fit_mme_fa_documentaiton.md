# Documentation: `fit_mme_fa`

### Overview

`fit_mme_independent` is a self-contained R function that estimates variance components and solves Mixed Model Equations (MMEs) without external dependencies (other than `Rcpp` for compilation). It implements the **Expectation-Maximization (EM) algorithm** in C++ to iteratively optimize variance components, providing a robust, standalone alternative to complex breeding software packages like `sommer`.

---

### Mathematical Foundation

The function iteratively optimizes the variance components by alternating between two primary steps:

1. **Solving the MMEs**: Given current variance estimates ($\sigma^2_g, \sigma^2_e$), the system is solved for fixed effects ($b$) and random effects ($u$):

$$\begin{bmatrix} X'X & X'Z \\ Z'X & Z'Z + I(\sigma^2_e/\sigma^2_g) \end{bmatrix} \begin{bmatrix} b \\ u \end{bmatrix} = \begin{bmatrix} X'y \\ Z'y \end{bmatrix}$$


2. **Updating Variance Components**: The estimates are updated using Henderson's formulas:
* **Error Variance**: $\hat{\sigma}^2_e = \frac{y'y - \hat{b}'X'y - \hat{u}'Z'y}{N - \text{rank}(X)}$
* **Genetic Variance**: $\hat{\sigma}^2_g = \frac{\hat{u}'\hat{u} + \hat{\sigma}^2_e \text{tr}(C^{22})}{q}$



---

### Implementation Setup

The tool consists of an R interface ("mouth") that prepares data matrices and a C++ body that performs high-speed iterations.

#### 1. C++ Computational Engine

The engine uses `RcppArmadillo` for efficient linear algebra and matrix operations:

```cpp
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

// [[Rcpp::export]]
Rcpp::List solve_mme_em_cpp(const arma::mat& X, const arma::mat& Z, const arma::vec& y, int max_iter = 500, double tol = 1e-6) {
    // ... [Implementation uses EM algorithm to converge on REML estimates] ...
}

```

#### 2. R Interface

The R function processes standard formula objects into the numeric matrices required by the C++ backend:

```r
fit_mme_fa<- function(fixed, random, data) {
  # Parses formulas to design matrices and calls solve_mme_em_cpp
  # Returns: list(BLUEs, BLUPs, var_g, var_e)
}

```

---

### Use Case Example: Agricultural Field Trial

This function is ideal for plant breeding scenarios where you need to partition variance to estimate the "breeding value" of different varieties across different test environments.

```r
# Example usage with simulated data
model_results <- fit_mme_independent(
  fixed  = Yield ~ Env,
  random = ~ Geno,
  data   = my_trial_data
)

# Access estimates
print(model_results$BLUEs) # Environment performance
print(model_results$BLUPs) # Genotype genetic values

```

---

### Validation

Because this function uses the EM algorithm, its results are statistically convergent with other REML-based tools (like `sommer`'s AI-REML). While the specific internal iteration paths differ, the resulting variance components and predictions typically match with extremely high precision.

* **Custom Engine**: Uses EM algorithm (robust, guaranteed to increase likelihood).
* **Sommer Engine**: Uses AI-REML (faster convergence, but more complex implementation).
* **Comparison**: When tested on the same dataset, both produce nearly identical estimates for $\sigma^2_g$ and $\sigma^2_e$, confirming the validity of the independent implementation.