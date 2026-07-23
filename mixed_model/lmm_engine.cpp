// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <cmath>

using namespace Rcpp;
using namespace Eigen;

typedef SparseMatrix<double> SpMat;
typedef SimplicialLDLT<SpMat> SparseCholesky;

// Helper function to build Henderson's MME matrix C
SpMat build_mme_lhs(const MatrixXd& X, const SpMat& Z, double lambda) {
    int p = X.cols();
    int q = Z.cols();
    
    SpMat ZtZ = Z.transpose() * Z;
    SpMat I(q, q);
    I.setIdentity();
    SpMat ZtZ_reg = ZtZ + lambda * I;

    std::vector<Triplet<double>> triplets;
    
    // X'X block
    MatrixXd XtX = X.transpose() * X;
    for (int i = 0; i < p; ++i)
        for (int j = 0; j < p; ++j)
            triplets.push_back(Triplet<double>(i, j, XtX(i, j)));

    // X'Z and Z'X blocks
    // Note: X'Z evaluates to a Dense matrix, so we store it in a MatrixXd
    MatrixXd XtZ = X.transpose() * Z;
    for (int i = 0; i < p; ++i) {
      for (int j = 0; j < q; ++j) {
        // Only add non-zero elements to the sparse triplet list
        if (XtZ(i, j) != 0.0) {
          triplets.push_back(Triplet<double>(i, p + j, XtZ(i, j)));
          triplets.push_back(Triplet<double>(p + j, i, XtZ(i, j)));
        }
      }
    }
    
    // Z'Z + lambda*I block
    for (int k = 0; k < ZtZ_reg.outerSize(); ++k) {
        for (SpMat::InnerIterator it(ZtZ_reg, k); it; ++it) {
            triplets.push_back(Triplet<double>(p + it.row(), p + it.col(), it.value()));
        }
    }

    SpMat C(p + q, p + q);
    C.setFromTriplets(triplets.begin(), triplets.end());
    return C;
}

// --------------------------------------------------------------------------
// 1. Profiled Likelihood Evaluator (lme4 approach)
// --------------------------------------------------------------------------
// [[Rcpp::export]]
List eval_profiled_reml_cpp(const Map<MatrixXd> X, 
                            const MappedSparseMatrix<double> Z, 
                            const Map<VectorXd> y, 
                            double lambda) {
    int n = X.rows();
    int p = X.cols();
    int q = Z.cols();

    SpMat C = build_mme_lhs(X, Z, lambda);

    SparseCholesky solver;
    solver.compute(C);
    if (solver.info() != Success) {
        return List::create(Named("deviance") = 1e10);
    }

    VectorXd RHS(p + q);
    RHS.head(p) = X.transpose() * y;
    RHS.tail(q) = Z.transpose() * y;

    VectorXd sol = solver.solve(RHS);
    VectorXd beta = sol.head(p);
    VectorXd u = sol.tail(q);

    // Compute Penalized Residual Sum of Squares (PRSS)
    VectorXd e = y - X * beta - Z * u;
    double prss = e.squaredNorm() + lambda * u.squaredNorm();
    double sigma2_e = prss / (n - p);

    // Log-determinant of C from LDLT diagonal
    VectorXd D = solver.vectorD();
    double log_det_C = 0.0;
    for (int i = 0; i < D.size(); ++i) {
        log_det_C += std::log(D(i));
    }

    // REML Deviance (-2 * logLik)
    double deviance = (n - p) * std::log(sigma2_e) + log_det_C - q * std::log(lambda);

    return List::create(
        Named("deviance") = deviance,
        Named("sigma2_e") = sigma2_e,
        Named("sigma2_u") = sigma2_e / lambda,
        Named("beta") = beta,
        Named("u") = u
    );
}

// --------------------------------------------------------------------------
// 2. Single AI-REML Iteration Step (ASReml approach)
// --------------------------------------------------------------------------
// [[Rcpp::export]]
List step_aireml_cpp(const Map<MatrixXd> X, 
                     const MappedSparseMatrix<double> Z, 
                     const Map<VectorXd> y, 
                     double sigma2_e, 
                     double sigma2_u) {
    int n = X.rows();
    int p = X.cols();
    int q = Z.cols();
    double lambda = sigma2_e / sigma2_u;

    SpMat C = build_mme_lhs(X, Z, lambda);

    SparseCholesky solver;
    solver.compute(C);

    VectorXd RHS(p + q);
    RHS.head(p) = X.transpose() * y;
    RHS.tail(q) = Z.transpose() * y;

    VectorXd sol = solver.solve(RHS);
    VectorXd beta = sol.head(p);
    VectorXd u = sol.tail(q);

    VectorXd e = y - X * beta - Z * u;

    // Working vectors for AI matrix construction (Gilmour et al. 1995)
    VectorXd Py = e / sigma2_e;
    VectorXd ZtPy = Z.transpose() * Py;

    // Working Score Equations
    double tr_P_approx = (n - p - q) / sigma2_e; 
    double score_e = -0.5 * (tr_P_approx - Py.squaredNorm());
    double score_u = -0.5 * (q / sigma2_u - (u.squaredNorm() / (sigma2_u * sigma2_u) + ZtPy.squaredNorm()));

    // Average Information (AI) Matrix
    double AI_ee = 0.5 * Py.squaredNorm() / sigma2_e;
    double AI_eu = 0.5 * ZtPy.squaredNorm() / sigma2_e;
    double AI_uu = 0.5 * ZtPy.squaredNorm() / sigma2_u;

    Matrix2d AI;
    AI << AI_ee, AI_eu,
          AI_eu, AI_uu;

    Vector2d score;
    score << score_e, score_u;

    // AI Newton-Raphson Step: theta_new = theta + AI^{-1} * score
    Vector2d theta;
    theta << sigma2_e, sigma2_u;
    Vector2d update = AI.ldlt().solve(score);
    Vector2d theta_new = theta + update;

    // Positivity constraint bounds
    if (theta_new(0) <= 1e-5) theta_new(0) = 1e-5;
    if (theta_new(1) <= 1e-5) theta_new(1) = 1e-5;

    return List::create(
        Named("sigma2_e") = theta_new(0),
        Named("sigma2_u") = theta_new(1),
        Named("beta") = beta,
        Named("u") = u,
        Named("change") = update.norm()
    );
}