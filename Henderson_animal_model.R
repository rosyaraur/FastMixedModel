# Load the required library
library(Rcpp)

# 1. Source the C++ Function using Rcpp
# We define the C++ code as a string and compile it automatically.
# The [[Rcpp::depends(RcppEigen)]] tag tells Rcpp to link the Eigen library.
sourceCpp(code = '
#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export]]
Eigen::VectorXd solveAnimalModelR(
    const Eigen::MatrixXd& X, 
    const Eigen::MatrixXd& Za, 
    const Eigen::MatrixXd& A_inv, 
    double variance_ratio, 
    const Eigen::VectorXd& y) 
{
    int n_fixed = X.cols();
    int n_animals = Za.cols();
    int total_cols = n_fixed + n_animals;
    
    // Pre-compute transposed matrices
    Eigen::MatrixXd X_t = X.transpose();
    Eigen::MatrixXd Za_t = Za.transpose();
    
    // Initialize Left Hand Side (LHS)
    Eigen::MatrixXd LHS = Eigen::MatrixXd::Zero(total_cols, total_cols);
    
    LHS.block(0, 0, n_fixed, n_fixed) = X_t * X;
    LHS.block(0, n_fixed, n_fixed, n_animals) = X_t * Za;
    LHS.block(n_fixed, 0, n_animals, n_fixed) = Za_t * X;
    LHS.block(n_fixed, n_fixed, n_animals, n_animals) = (Za_t * Za) + (A_inv * variance_ratio);
    
    // Initialize Right Hand Side (RHS)
    Eigen::VectorXd RHS = Eigen::VectorXd::Zero(total_cols);
    RHS.segment(0, n_fixed) = X_t * y;
    RHS.segment(n_fixed, n_animals) = Za_t * y;
    
    // Solve the system
    Eigen::VectorXd solutions = LHS.colPivHouseholderQr().solve(RHS);
    
    return solutions;
}
')

# 2. Setup the Data for Henderson's Dam-Daughter Example
n_records <- 10
n_animals <- 10

# Vector y (Observations)
y <- c(5, 4, 3, 2, 6, 6, 7, 3, 5, 4)

# Matrix X (Fixed Effects Design Matrix)
X <- matrix(0, nrow = n_records, ncol = 2)
X[1:5, 1] <- 1   # Period 1
X[6:10, 2] <- 1  # Period 2

# Matrix Za (Random Additive Genetic Design Matrix)
# Identity matrix since each animal has 1 record
Za <- diag(n_records)

# Matrix A (Numerator Relationship Matrix)
A <- diag(n_animals)
I_5 <- diag(5)
A[1:5, 6:10] <- 0.5 * I_5  # Relationship from dams to daughters
A[6:10, 1:5] <- 0.5 * I_5  # Relationship from daughters to dams

# Calculate Inverse of A
A_inv <- solve(A)

# Variance ratio (sigma_e^2 / sigma_a^2 = 5)
variance_ratio <- 5.0

# 3. Call the Compiled C++ Function
cat("Solving Mixed Model Equations via RcppEigen...\n\n")
solutions <- solveAnimalModelR(X, Za, A_inv, variance_ratio, y)

# 4. Format and Print the Output
expected_values <- c(4, 5, 0.23077, 0.13986, -0.30070, -0.32168, 0.25175, 
                     0.23077, 0.32168, -0.39161, -0.13986, -0.20298)

results_df <- data.frame(
  Parameter = c("p1", "p2", paste0("a", 1:n_animals)),
  Expected_From_Text = expected_values,
  Calculated_C_plus_plus = as.numeric(solutions)
)

print(results_df, row.names = FALSE)


# For large A matrix 

# To handle large datasets efficiently, it is computationally prohibitive to construct the numerator relationship matrix $\mathbf{A}$ and then invert it. Instead, you can compute $\mathbf{A}^{-1}$ directly from a list of pedigrees using Henderson's (1976) rules.

# Below is the updated R script. It includes a new C++ function, `computeAInverseR`, which applies Henderson’s rapid inversion algorithm assuming the base population is non-inbred (which matches the dam-daughter example). This function is seamlessly integrated with the `solveAnimalModelR` function from the previous step.

### Complete R Script with Integrated C++ Henderson (1976) Algorithm

# Load the required library
library(Rcpp)

# 1. Source the C++ Functions using Rcpp
# This block compiles both Henderson's A-inverse method and the MME solver.
sourceCpp(code = '
#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export]]
Eigen::MatrixXd computeAInverseR(const Eigen::MatrixXi& pedigree) {
    // pedigree is an n x 2 matrix containing (Sire, Dam) indices.
    // Indices should be 1-based (R style). 0 indicates an unknown parent.
    // Animals must be ordered such that parents appear before their progeny.
    
    int n = pedigree.rows();
    Eigen::MatrixXd A_inv = Eigen::MatrixXd::Zero(n, n);
    
    for (int i = 0; i < n; ++i) {
        // Convert to 0-based indexing for C++. Unknown (0) becomes -1.
        int s = pedigree(i, 0) - 1; 
        int d = pedigree(i, 1) - 1;
        
        // Determine the diagonal element of D (variance of Mendelian sampling)
        // Assuming base population is non-inbred for this implementation.
        double di = 1.0;
        if (s >= 0 && d >= 0) {
            di = 0.5;      // Both parents known
        } else if (s >= 0 || d >= 0) {
            di = 0.75;     // One parent known
        }
        
        double val = 1.0 / di;
        
        // Apply Henderson\'s (1976) rules to build A inverse directly
        A_inv(i, i) += val;
        
        if (s >= 0) {
            A_inv(s, i) -= 0.5 * val;
            A_inv(i, s) -= 0.5 * val;
            A_inv(s, s) += 0.25 * val;
        }
        if (d >= 0) {
            A_inv(d, i) -= 0.5 * val;
            A_inv(i, d) -= 0.5 * val;
            A_inv(d, d) += 0.25 * val;
        }
        if (s >= 0 && d >= 0) {
            A_inv(s, d) += 0.25 * val;
            A_inv(d, s) += 0.25 * val;
        }
    }
    
    return A_inv;
}

// [[Rcpp::export]]
Eigen::VectorXd solveAnimalModelR(
    const Eigen::MatrixXd& X, 
    const Eigen::MatrixXd& Za, 
    const Eigen::MatrixXd& A_inv, 
    double variance_ratio, 
    const Eigen::VectorXd& y) 
{
    int n_fixed = X.cols();
    int n_animals = Za.cols();
    int total_cols = n_fixed + n_animals;
    
    Eigen::MatrixXd X_t = X.transpose();
    Eigen::MatrixXd Za_t = Za.transpose();
    
    Eigen::MatrixXd LHS = Eigen::MatrixXd::Zero(total_cols, total_cols);
    
    LHS.block(0, 0, n_fixed, n_fixed) = X_t * X;
    LHS.block(0, n_fixed, n_fixed, n_animals) = X_t * Za;
    LHS.block(n_fixed, 0, n_animals, n_fixed) = Za_t * X;
    LHS.block(n_fixed, n_fixed, n_animals, n_animals) = (Za_t * Za) + (A_inv * variance_ratio);
    
    Eigen::VectorXd RHS = Eigen::VectorXd::Zero(total_cols);
    RHS.segment(0, n_fixed) = X_t * y;
    RHS.segment(n_fixed, n_animals) = Za_t * y;
    
    Eigen::VectorXd solutions = LHS.colPivHouseholderQr().solve(RHS);
    
    return solutions;
}
')

# 2. Setup Data using the Pedigree Approach
n_records <- 10
n_animals <- 10

# Vector y (Observations)
y <- c(5, 4, 3, 2, 6, 6, 7, 3, 5, 4)

# Matrix X (Fixed Effects)
X <- matrix(0, nrow = n_records, ncol = 2)
X[1:5, 1] <- 1   # Period 1
X[6:10, 2] <- 1  # Period 2

# Matrix Za (Identity matrix since each animal has 1 record)
Za <- diag(n_records)

# --- NEW: Define Pedigree instead of building matrix A ---
# 2 columns: Sire, Dam. 0 means unknown.
# Animals 1-5 (Dams) have unknown parents.
# Animals 6-10 (Daughters) have unknown sires, and Dams are 1-5.
pedigree <- matrix(c(
  0, 0,  # Animal 1 (Dam 1)
  0, 0,  # Animal 2 (Dam 2)
  0, 0,  # Animal 3 (Dam 3)
  0, 0,  # Animal 4 (Dam 4)
  0, 0,  # Animal 5 (Dam 5)
  0, 1,  # Animal 6 (Daughter of Dam 1)
  0, 2,  # Animal 7 (Daughter of Dam 2)
  0, 3,  # Animal 8 (Daughter of Dam 3)
  0, 4,  # Animal 9 (Daughter of Dam 4)
  0, 5   # Animal 10 (Daughter of Dam 5)
), ncol = 2, byrow = TRUE)

# Calculate Inverse of A DIRECTLY using Henderson's C++ method
A_inv_rapid <- computeAInverseR(pedigree)

# Variance ratio
variance_ratio <- 5.0

# 3. Call the Compiled C++ MME Solver
cat("Computing solutions using rapid A-inverse...\n\n")
solutions <- solveAnimalModelR(X, Za, A_inv_rapid, variance_ratio, y)

# 4. Format Output
expected_values <- c(4, 5, 0.23077, 0.13986, -0.30070, -0.32168, 0.25175, 
                     0.23077, 0.32168, -0.39161, -0.13986, -0.20298)

results_df <- data.frame(
  Parameter = c("p1", "p2", paste0("a", 1:n_animals)),
  Expected_From_Text = expected_values,
  Calculated = round(as.numeric(solutions), 5)
)

print(results_df, row.names = FALSE)

