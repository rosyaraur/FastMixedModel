#include <Eigen/Dense>

using namespace Eigen;

/**
 * Computes the inverse of the numerator relationship matrix (A^-1) directly 
 * from a pedigree using Henderson's (1976) rules for large datasets.
 * 
 * @param pedigree An n x 2 matrix of (Sire, Dam) indices. 
 *                 Indices must be 0-based. Use -1 for an unknown parent.
 *                 Animals MUST be ordered such that parents appear before their progeny.
 * @return         The A^-1 matrix.
 */
MatrixXd computeAInverse(const MatrixXi& pedigree) {
    int n = pedigree.rows();
    MatrixXd A_inv = MatrixXd::Zero(n, n);
    
    for (int i = 0; i < n; ++i) {
        int s = pedigree(i, 0); 
        int d = pedigree(i, 1);
        
        // Determine the diagonal element of D (variance of Mendelian sampling)
        // Assuming the base population is non-inbred for this implementation.
        double di = 1.0;
        if (s >= 0 && d >= 0) {
            di = 0.5;      // Both parents known
        } else if (s >= 0 || d >= 0) {
            di = 0.75;     // One parent known
        }
        
        double val = 1.0 / di;
        
        // Apply Henderson's (1976) rules to build A inverse directly
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

/**
 * Solves Henderson's Mixed Model Equations (MME) for the Animal Model.
 * 
 * @param X              Design matrix for fixed effects
 * @param Za             Design matrix for random additive genetic values
 * @param A_inv          Inverse of the numerator relationship matrix (A^-1)
 * @param variance_ratio Ratio of residual to additive genetic variance (sigma_e^2 / sigma_a^2)
 * @param y              Vector of observations
 * @return               Vector containing [beta_hat, a_hat]^T
 */
VectorXd solveAnimalModel(
    const MatrixXd& X, 
    const MatrixXd& Za, 
    const MatrixXd& A_inv, 
    double variance_ratio, 
    const VectorXd& y) 
{
    int n_fixed = X.cols();
    int n_animals = Za.cols();
    int total_cols = n_fixed + n_animals;
    
    // Pre-compute transposed matrices
    MatrixXd X_t = X.transpose();
    MatrixXd Za_t = Za.transpose();
    
    // Initialize Left Hand Side (LHS) matrix
    MatrixXd LHS = MatrixXd::Zero(total_cols, total_cols);
    
    LHS.block(0, 0, n_fixed, n_fixed) = X_t * X;
    LHS.block(0, n_fixed, n_fixed, n_animals) = X_t * Za;
    LHS.block(n_fixed, 0, n_animals, n_fixed) = Za_t * X;
    LHS.block(n_fixed, n_fixed, n_animals, n_animals) = (Za_t * Za) + (A_inv * variance_ratio);
    
    // Initialize Right Hand Side (RHS) vector
    VectorXd RHS = VectorXd::Zero(total_cols);
    RHS.segment(0, n_fixed) = X_t * y;
    RHS.segment(n_fixed, n_animals) = Za_t * y;
    
    // Solve the system: LHS * solutions = RHS
    VectorXd solutions = LHS.colPivHouseholderQr().solve(RHS);
    
    return solutions;
}