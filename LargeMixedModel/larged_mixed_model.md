# Methodological Details: Large-Scale Mixed Model / Genomic Prediction Approximations

This documentation outlines the mathematical framework, C++ functions, and R implementation for bypassing the memory and computational bottlenecks of massive Mixed Model Equations (MME) in genomic evaluations.

Two primary bottlenecks are addressed:

1. **Computational Time ($O(N^3)$):** Variance component (VC) estimation requires repeatedly inverting dense relationship matrices.
2. **Memory Allocation ($O(N^2)$):** Constructing the dense genomic relationship matrix ($G$) exceeds standard RAM limits as the number of individuals ($N$) grows large.

---

## 1. Two-Step Variance Component Approximation

Traditional Genomic Best Linear Unbiased Prediction (GBLUP) solves the following system to estimate genetic variance ($\sigma^2_g$) and residual variance ($\sigma^2_e$):

$$\begin{bmatrix} X'X & X'Z \\ Z'X & Z'Z + G^{-1}\lambda \end{bmatrix} \begin{bmatrix} \hat{\beta} \\ \hat{u} \end{bmatrix} = \begin{bmatrix} X'y \\ Z'y \end{bmatrix}$$

Where $\lambda = \sigma^2_e / \sigma^2_g$. Inverting $G$ iteratively during Restricted Maximum Likelihood (REML) is computationally infeasible for $N > 50,000$.

### Methodology

Instead of estimating VC on the full dataset, we implement a two-step approximation:

1. **Subset Estimation:** Select a representative subset of individuals ($n \approx 5,000$) to estimate a highly accurate $\lambda$.
2. **Conjugate Gradient Solver:** Fix $\lambda$ and solve the full system iteratively. The exact GBLUP solution can be re-written without $G^{-1}$ as a linear system $Ax = b$, where $A = (G + \lambda I)$ and $b = y$. We solve for $x$ using a Conjugate Gradient (CG) solver, which relies purely on matrix-vector multiplications ($O(N^2)$). Final estimated breeding values (EBVs) are obtained via $\hat{g} = G x$.

### Core Function: C++ Conjugate Gradient

```cpp
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

// [[Rcpp::export]]
arma::vec cg_solve(const arma::mat& A, const arma::vec& b, int max_iter = 1000, double tol = 1e-6) {
    int n = A.n_rows;
    arma::vec x = arma::zeros(n);
    arma::vec r = b - A * x;
    arma::vec p = r;
    double rsold = arma::dot(r, r);
    
    for (int i = 0; i < max_iter; ++i) {
        arma::vec Ap = A * p;
        double alpha = rsold / arma::dot(p, Ap);
        x = x + alpha * p;
        r = r - alpha * Ap;
        double rsnew = arma::dot(r, r);
        
        if (sqrt(rsnew) < tol) break;
        
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    }
    return x;
}

```

### Embedded Example: Subset VC + CG Pipeline

```R
library(rrBLUP)
library(Rcpp)
# Note: Ensure cg_solve is sourced via Rcpp::sourceCpp()

# 1. Simulate full dataset
N <- 10000; M <- 2000
Z <- matrix(sample(c(-1, 0, 1), N * M, replace = TRUE), N, M)
G <- tcrossprod(Z) / M 
y_c <- scale(Z %*% rnorm(M, 0, 1) + rnorm(N, 0, 1), scale = FALSE)

# 2. Estimate lambda on a subset (N = 500)
idx_sub <- sample(1:N, 500)
fit_sub <- mixed.solve(y_c[idx_sub], K = G[idx_sub, idx_sub])
lambda_est <- fit_sub$Ve / fit_sub$Vu

# 3. Solve full system using C++ CG solver
A <- G + (lambda_est * diag(N))
x_alpha <- cg_solve(A, as.numeric(y_c))

# 4. Compute Final EBVs
ebv_approx <- G %*% x_alpha

```

---

## 2. Memory-Efficient $G$ Matrix Construction

For a genotype matrix $Z$ of dimensions $N \times M$, the standard R calculation `tcrossprod(Z)` creates a massive intermediate object before returning the $N \times N$ matrix. At $N=100,000$, this requires approximately 80 GB of contiguous RAM.

### Methodology

To bypass memory limits without altering the mathematical output, the $G$ matrix is computed using a **Block Matrix (Chunking) Algorithm**. The algorithm divides the $N$ individuals into sequential blocks. It iteratively computes the cross-products of these blocks, filling in a pre-allocated output matrix. This restricts the maximum contiguous memory allocation to the size of the block ($b \times b$) rather than $N \times N$.

### Core Function: C++ Block Builder

```cpp
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

// [[Rcpp::export]]
void compute_G_blocked(const arma::mat& Z, arma::mat& G, int block_size) {
    int N = Z.n_rows;
    double M = Z.n_cols;
    
    for (int i = 0; i < N; i += block_size) {
        int end_i = std::min(i + block_size - 1, N - 1);
        arma::mat Z_block_i = Z.rows(i, end_i);
        
        // Diagonal block
        G.submat(i, i, end_i, end_i) = (Z_block_i * Z_block_i.t()) / M;
        
        // Off-diagonal blocks (filling symmetrically)
        for (int j = i + block_size; j < N; j += block_size) {
            int end_j = std::min(j + block_size - 1, N - 1);
            arma::mat Z_block_j = Z.rows(j, end_j);
            arma::mat G_sub = (Z_block_i * Z_block_j.t()) / M;
            
            G.submat(i, j, end_i, end_j) = G_sub;
            G.submat(j, i, end_j, end_i) = G_sub.t();
        }
        Rcpp::checkUserInterrupt(); 
    }
}

```

### Embedded Example: Chunked Generation

```R
# Note: Ensure compute_G_blocked is sourced via Rcpp::sourceCpp()

N <- 20000
M <- 5000
Z <- matrix(sample(c(-1, 0, 1), N * M, replace = TRUE), N, M)

# Pre-allocate the output matrix to avoid R-level copying
G_chunked <- matrix(0, nrow = N, ncol = N)

# Compute G in blocks of 2000 individuals
compute_G_blocked(Z, G_chunked, block_size = 2000)

```

---

## 3. Alternative Dimensionality Reduction: SNP-BLUP Equivalence

If $N > M$ (more individuals than markers), constructing $G$ is mathematically redundant. The Woodbury matrix identity demonstrates that solving the model in the marker space (SNP-BLUP) is strictly equivalent to solving in the individual space (GBLUP).

### Methodology

Instead of calculating $Z Z'$ ($N \times N$), compute the marker cross-product $Z' Z$ ($M \times M$).

1. Calculate marker effects ($\hat{m}$):

$$(Z'Z + I\lambda) \hat{m} = Z'y$$


2. Project breeding values ($\hat{g}$):

$$\hat{g} = Z \hat{m}$$



**Complexity Comparison:**

| Method | Target Matrix | Dimension | Primary Use Case |
| --- | --- | --- | --- |
| **GBLUP (Chunked)** | $Z Z'$ | $N \times N$ | $M \gg N$, or utilizing APY inverse |
| **SNP-BLUP** | $Z' Z$ | $M \times M$ | $N \gg M$ |