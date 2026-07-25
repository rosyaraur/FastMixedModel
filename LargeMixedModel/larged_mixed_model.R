# Load required libraries
if (!requireNamespace("rrBLUP", quietly = TRUE)) install.packages("rrBLUP")
if (!requireNamespace("Rcpp", quietly = TRUE)) install.packages("Rcpp")
if (!requireNamespace("RcppArmadillo", quietly = TRUE)) install.packages("RcppArmadillo")

library(rrBLUP)
library(Rcpp)

# ==========================================
# 1. C++ CONJUGATE GRADIENT SOLVER
# ==========================================
# This compiles a fast CG solver using Armadillo
sourceCpp(code = '
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
        
        if (sqrt(rsnew) < tol) {
            Rcpp::Rcout << "CG Converged in " << i << " iterations.\\n";
            break;
        }
        
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    }
    return x;
}
')

# ==========================================
# 2. SIMULATE LARGE DATASET
# ==========================================
cat("Simulating Genotypes and Phenotypes...\n")
set.seed(42)
N <- 3000      # Total individuals (large enough to show time difference)
M <- 1000      # Markers
N_sub <- 500   # Subset size for VC estimation

# Simulate genotypes (-1, 0, 1) and create G matrix
Z <- matrix(sample(c(-1, 0, 1), N * M, replace = TRUE), N, M)
G <- tcrossprod(Z) / M 

# Simulate true genetic effects and phenotypes (h2 = 0.5)
u_true <- rnorm(M, 0, 1)
g_true <- Z %*% u_true
var_g <- var(g_true)
var_e <- var_g 
y <- g_true + rnorm(N, 0, sqrt(var_e))
y_c <- scale(y, center = TRUE, scale = FALSE) # Mean center

# ==========================================
# 3. METHOD 1: FULL EXACT GBLUP (The Bottleneck)
# ==========================================
cat("\nRunning Full Exact GBLUP (Estimating VC on all data)...\n")
start_time <- Sys.time()
fit_full <- mixed.solve(y_c, K = G)
ebv_exact <- fit_full$u
time_exact <- Sys.time() - start_time
cat("Time taken:", round(time_exact, 2), "seconds\n")

# ==========================================
# 4. METHOD 2: SUBSET + C++ CG APPROXIMATION
# ==========================================
cat("\nRunning Subset VC + C++ CG Solver...\n")
start_time <- Sys.time()

# Step A: Estimate Variance Components on Subset
idx_sub <- sample(1:N, N_sub)
fit_sub <- mixed.solve(y_c[idx_sub], K = G[idx_sub, idx_sub])
lambda_est <- fit_sub$Ve / fit_sub$Vu

# Step B: Setup the linear system A = (G + lambda * I)
I <- diag(N)
A <- G + (lambda_est * I)

# Step C: Solve Ax = y using our C++ Conjugate Gradient
x_alpha <- cg_solve(A, as.numeric(y_c))

# Step D: Calculate Final EBVs
ebv_approx <- G %*% x_alpha

time_approx <- Sys.time() - start_time
cat("Time taken:", round(time_approx, 2), "seconds\n")

# ==========================================
# 5. COMPARE RESULTS
# ==========================================
cat("\n--- RESULTS ---\n")
correlation <- cor(ebv_exact, ebv_approx)
cat("Correlation between Exact and Approximated EBVs:", round(correlation, 6), "\n")

# ==========================================
# optimize the G matrix construction
# ==========================================

# When your sample size ($N$) approaches 100,000, calling tcrossprod(Z) in R is a guaranteed fatal error. 
# R attempts to allocate the input, the intermediate steps, and the massive $N \times N$ output matrix 
# (which alone is $\sim 80$ GB in double-precision RAM) all at once.

# Strategy 1: The Math Bypass (SNP-BLUP)
# Strategy 2: Block Matrix Computation (Chunking)
library(Rcpp)
library(RcppArmadillo)

# C++ function to compute G in blocks
sourceCpp(code = '
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

// [[Rcpp::export]]
void compute_G_blocked(const arma::mat& Z, arma::mat& G, int block_size) {
    int N = Z.n_rows;
    double M = Z.n_cols;
    
    // Iterate over row blocks of Z
    for (int i = 0; i < N; i += block_size) {
        int end_i = std::min(i + block_size - 1, N - 1);
        arma::mat Z_block_i = Z.rows(i, end_i);
        
        // Compute diagonal block
        G.submat(i, i, end_i, end_i) = (Z_block_i * Z_block_i.t()) / M;
        
        // Compute off-diagonal blocks
        for (int j = i + block_size; j < N; j += block_size) {
            int end_j = std::min(j + block_size - 1, N - 1);
            arma::mat Z_block_j = Z.rows(j, end_j);
            
            arma::mat G_sub = (Z_block_i * Z_block_j.t()) / M;
            
            // Fill symmetric parts
            G.submat(i, j, end_i, end_j) = G_sub;
            G.submat(j, i, end_j, end_i) = G_sub.t();
        }
        
        // Force garbage collection in R if memory gets tight
        Rcpp::checkUserInterrupt(); 
    }
}
')

# ==========================================
# TEST THE CHUNKED COMPUTATION
# ==========================================
N <- 5000  # Scale this up based on your RAM
M <- 1000

# 1. Simulate Genotypes
Z <- matrix(sample(c(-1, 0, 1), N * M, replace = TRUE), N, M)

# 2. Pre-allocate the G matrix to avoid R copying overhead
G_chunked <- matrix(0, nrow = N, ncol = N)

# 3. Compute using block size of 1000
cat("Computing chunked G matrix...\n")
start_time <- Sys.time()
compute_G_blocked(Z, G_chunked, block_size = 1000)
print(Sys.time() - start_time)

# 4. Verify against standard tcrossprod (only do this for small N!)
G_standard <- tcrossprod(Z) / M
cat("Max difference:", max(abs(G_chunked - G_standard)), "\n")



