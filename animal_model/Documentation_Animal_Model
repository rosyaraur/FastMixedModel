
# Fast Animal Model of Henderson 

Here is a C++ implementation of the mixed model equations (MME) for the animal model as outlined by C. R. Henderson.

To handle the matrix algebra efficiently in C++, this implementation uses the popular **Eigen** library. You will need to have Eigen installed and included in your project's include path.

### Henderson's General Animal Model

Based on the provided document, the general vector representation for the set of records is:


$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \mathbf{Z}\mathbf{u} + \mathbf{Z}_a\mathbf{a} + \mathbf{e}$$

The implementation below constructs and solves the general mixed model equations (MME) when $\mathbf{Z}_a \neq \mathbf{I}$:


$$\begin{pmatrix} \mathbf{X}'\mathbf{R}^{-1}\mathbf{X} & \mathbf{X}'\mathbf{R}^{-1}\mathbf{Z} & \mathbf{X}'\mathbf{R}^{-1}\mathbf{Z}_a \\ \mathbf{Z}'\mathbf{R}^{-1}\mathbf{X} & \mathbf{Z}'\mathbf{R}^{-1}\mathbf{Z} + \mathbf{G}^{-1} & \mathbf{Z}'\mathbf{R}^{-1}\mathbf{Z}_a \\ \mathbf{Z}_a'\mathbf{R}^{-1}\mathbf{X} & \mathbf{Z}_a'\mathbf{R}^{-1}\mathbf{Z} & \mathbf{Z}_a'\mathbf{R}^{-1}\mathbf{Z}_a + \mathbf{A}^{-1}/\sigma_a^2 \end{pmatrix} \begin{pmatrix} \hat{\boldsymbol{\beta}} \\ \hat{\mathbf{u}} \\ \hat{\mathbf{a}} \end{pmatrix} = \begin{pmatrix} \mathbf{X}'\mathbf{R}^{-1}\mathbf{y} \\ \mathbf{Z}'\mathbf{R}^{-1}\mathbf{y} \\ \mathbf{Z}_a'\mathbf{R}^{-1}\mathbf{y} \end{pmatrix}$$

### C++ Implementation (Using Eigen)

```cpp
#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

/**
 * Solves Henderson's Mixed Model Equations (MME) for the Animal Model.
 * 
 * @param X      Design matrix for fixed effects
 * @param Z      Design matrix for random effects (other than breeding values)
 * @param Za     Design matrix for additive genetic values
 * @param R_inv  Inverse of the residual variance-covariance matrix (R^-1)
 * @param G_inv  Inverse of the variance-covariance matrix for random effects (G^-1)
 * @param A_inv  Inverse of the numerator relationship matrix (A^-1)
 * @param var_a  Additive genetic variance (sigma^2_a)
 * @param y      Vector of observations
 * @return       Vector containing [beta_hat, u_hat, a_hat]^T
 */
VectorXd solveAnimalModel(
    const MatrixXd& X, 
    const MatrixXd& Z, 
    const MatrixXd& Za, 
    const MatrixXd& R_inv, 
    const MatrixXd& G_inv, 
    const MatrixXd& A_inv, 
    double var_a, 
    const VectorXd& y) 
{
    // Determine dimensions
    int n_beta = X.cols();
    int n_u = Z.cols();
    int n_a = Za.cols();
    int total_cols = n_beta + n_u + n_a;

    // Pre-compute common transposed matrices to save overhead
    MatrixXd X_t = X.transpose();
    MatrixXd Z_t = Z.transpose();
    MatrixXd Za_t = Za.transpose();

    // Initialize LHS (Left Hand Side) matrix
    MatrixXd LHS = MatrixXd::Zero(total_cols, total_cols);

    // Row 1: Fixed effects (beta)
    LHS.block(0, 0, n_beta, n_beta)           = X_t * R_inv * X;
    LHS.block(0, n_beta, n_beta, n_u)         = X_t * R_inv * Z;
    LHS.block(0, n_beta + n_u, n_beta, n_a)   = X_t * R_inv * Za;

    // Row 2: Random effects (u)
    LHS.block(n_beta, 0, n_u, n_beta)         = Z_t * R_inv * X;
    LHS.block(n_beta, n_beta, n_u, n_u)       = Z_t * R_inv * Z + G_inv;
    LHS.block(n_beta, n_beta + n_u, n_u, n_a) = Z_t * R_inv * Za;

    // Row 3: Additive genetic values (a)
    LHS.block(n_beta + n_u, 0, n_a, n_beta)   = Za_t * R_inv * X;
    LHS.block(n_beta + n_u, n_beta, n_a, n_u) = Za_t * R_inv * Z;
    LHS.block(n_beta + n_u, n_beta + n_u, n_a, n_a) = Za_t * R_inv * Za + A_inv / var_a;

    // Initialize RHS (Right Hand Side) vector
    VectorXd RHS = VectorXd::Zero(total_cols);

    RHS.segment(0, n_beta)            = X_t * R_inv * y;
    RHS.segment(n_beta, n_u)          = Z_t * R_inv * y;
    RHS.segment(n_beta + n_u, n_a)    = Za_t * R_inv * y;

    // Solve the system: LHS * solutions = RHS
    // Using ColPivHouseholderQR for numerical stability with semi-definite matrices
    VectorXd solutions = LHS.colPivHouseholderQr().solve(RHS);

    return solutions;
}

```

### Notes on Simplification

If you are dealing with a scenario where the animal model is simplified (e.g., every animal has a record, so $\mathbf{Z}_a = \mathbf{I}$, and $\mathbf{R} = \mathbf{I}\sigma_e^2$), you can still use this function by passing identity matrices scaled appropriately for `Za` and `R_inv`.

For large datasets, directly inverting $\mathbf{A}$ is computationally expensive. As Henderson notes, his 1976 method for computing $\mathbf{A}^{-1}$ should be used in production systems to form the inverse directly without building the standard covariance matrix $\mathbf{A}$.