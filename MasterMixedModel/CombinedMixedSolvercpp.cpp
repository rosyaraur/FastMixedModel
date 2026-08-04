// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Cholesky>
#include <Eigen/SVD>
#include <Eigen/QR>
//#include <Eigen/CholmodSupport> // Requires SuiteSparse (CHOLMOD/Metis)

#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <algorithm>
#include <random>

using namespace Rcpp;
using namespace Eigen;

// =========================================================================
// SECTION 1: HELPER FUNCTIONS & STRUCTS
// =========================================================================

// Inverse-Gaussian Random Number Generator (Michael-Schucany-Haas)
inline double rinvgauss(double mu, double lambda) {
    if (mu <= 0.0 || lambda <= 0.0) return 1e-10; 
    
    double v = R::rnorm(0.0, 1.0);
    double y = v * v;
    double mu_sq = mu * mu;
    
    double x = mu + (mu_sq * y) / (2.0 * lambda) - 
               (mu / (2.0 * lambda)) * std::sqrt(4.0 * mu * lambda * y + mu_sq * y * y);
               
    double z = R::runif(0.0, 1.0);
    if (z <= (mu / (mu + x))) return x;
    return mu_sq / x;
}

// Binary search to find the 1D index of a (row, col) coordinate in CSC format
inline int get_csc_index(int row, int col, const int* outer, const int* inner) {
    int start = outer[col];
    int end = outer[col + 1];
    const int* it = std::lower_bound(inner + start, inner + end, row);
    if (it != inner + end && *it == row) return std::distance(inner, it);
    return -1;
}

// Struct for RSVD Output
struct SVDResult {
    MatrixXd U;
    VectorXd S;
    MatrixXd V;
};

// =========================================================================
// SECTION 2: CUSTOM MATRIX-FREE CLASSES (For PCG)
// =========================================================================

class MatrixFreeGenomicK {
public:
    const MatrixXd& W; 
    double lambda; 
    
    MatrixFreeGenomicK(const MatrixXd& W_in, double lambda_in) : W(W_in), lambda(lambda_in) {}
    
    int rows() const { return W.rows(); }
    int cols() const { return W.rows(); }
    
    VectorXd operator*(const VectorXd& v) const {
        return W * (W.transpose() * v) + (lambda * v);
    }
};

namespace Eigen {
    namespace internal {
        template<> struct traits<MatrixFreeGenomicK> : public traits<MatrixXd> {};
    }
}

class BlockDiagPreconditioner {
private:
    std::vector<LDLT<MatrixXd>> block_solvers;
    int block_size;

public:
    BlockDiagPreconditioner() : block_size(500) {} 

    BlockDiagPreconditioner& compute(const MatrixFreeGenomicK& mat) {
        const MatrixXd& W = mat.W;
        double lambda = mat.lambda;
        int n = W.rows();
        int num_blocks = std::ceil((double)n / block_size);
        block_solvers.resize(num_blocks);

        for (int i = 0; i < num_blocks; ++i) {
            int start = i * block_size;
            int size = std::min(block_size, n - start);
            
            MatrixXd W_block = W.middleRows(start, size);
            MatrixXd K_block = W_block * W_block.transpose();
            K_block.diagonal().array() += lambda; 
            block_solvers[i].compute(K_block);
        }
        return *this;
    }

    template<typename Rhs>
    inline VectorXd solve(const Rhs& b) const {
        VectorXd z(b.rows());
        for (int i = 0; i < (int)block_solvers.size(); ++i) {
            int start = i * block_size;
            int size = std::min(block_size, (int)b.rows() - start);
            z.segment(start, size) = block_solvers[i].solve(b.segment(start, size));
        }
        return z;
    }
};

// =========================================================================
// SECTION 3: KERNELS, REDUCTIONS, AND MATRIX BUILDERS
// =========================================================================

// [[Rcpp::export]]
Eigen::MatrixXd reduce_pca(const Eigen::MatrixXd& Z, int k) {
    Eigen::MatrixXd centered = Z.rowwise() - Z.colwise().mean();
    int max_k = std::min(centered.rows(), centered.cols());
    if (k > max_k) {
        Rcpp::warning("Requested k is greater than matrix rank. Truncating to maximum possible.");
        k = max_k; 
    }
    Eigen::BDCSVD<Eigen::MatrixXd> svd(centered, Eigen::ComputeThinU | Eigen::ComputeThinV);
    return svd.matrixU().leftCols(k) * svd.singularValues().head(k).asDiagonal();
}

// [[Rcpp::export]]
Eigen::MatrixXd reduce_pls(const Eigen::MatrixXd& Z, const Eigen::VectorXd& y, int k) {
    int n = Z.rows();
    Eigen::MatrixXd T(n, k);
    Eigen::MatrixXd E = Z;
    Eigen::VectorXd f = y;
    for (int i = 0; i < k; ++i) {
        Eigen::VectorXd w = E.transpose() * f;
        w.normalize();
        Eigen::VectorXd t = E * w;
        Eigen::VectorXd p = (E.transpose() * t) / t.squaredNorm();
        E -= t * p.transpose();
        double t_norm_sq = t.squaredNorm();
        if (t_norm_sq > 1e-12) {
            double q_scalar = (f.transpose() * t)(0) / t_norm_sq;
            f -= t * q_scalar;
        }
        T.col(i) = t;
    }
    return T;
}

// [[Rcpp::export]]
Eigen::MatrixXd reduce_random_projection(const Eigen::MatrixXd& Z, int k) {
    int q = Z.cols();
    Eigen::MatrixXd R(q, k);
    for (int i = 0; i < q; ++i) {
        for (int j = 0; j < k; ++j) {
            R(i, j) = R::rnorm(0.0, 1.0);
        }
    }
    return (Z * R) / std::sqrt(static_cast<double>(k));
}

// [[Rcpp::export]]
Eigen::MatrixXd build_linear_kernel(const Eigen::MatrixXd& W) {
    int n = W.rows();
    int q = W.cols();
    Eigen::MatrixXd W_scaled(n, q);
    for(int j = 0; j < q; ++j) {
        double mean = W.col(j).mean();
        Eigen::VectorXd centered = W.col(j).array() - mean;
        double var = centered.squaredNorm() / (n - 1.0);
        double sd = std::sqrt(var);
        W_scaled.col(j) = (sd > 1e-8) ? centered / sd : centered;
    }
    return (W_scaled * W_scaled.transpose()) / static_cast<double>(q);
}

// [[Rcpp::export]]
Eigen::MatrixXd build_gaussian_kernel(const Eigen::MatrixXd& W, double h) {
    int n = W.rows();
    int q = W.cols();
    Eigen::MatrixXd W_scaled(n, q);
    for(int j = 0; j < q; ++j) {
        double mean = W.col(j).mean();
        Eigen::VectorXd centered = W.col(j).array() - mean;
        double var = centered.squaredNorm() / (n - 1.0);
        double sd = std::sqrt(var);
        W_scaled.col(j) = (sd > 1e-8) ? centered / sd : centered;
    }
    Eigen::MatrixXd K(n, n);
    for(int i = 0; i < n; ++i) {
        for(int j = i; j < n; ++j) {
            double val = std::exp(-(W_scaled.row(i) - W_scaled.row(j)).squaredNorm() / h);
            K(i, j) = val; K(j, i) = val; 
        }
    }
    K += Eigen::MatrixXd::Identity(n, n) * 1e-6;
    return K;
}

// [[Rcpp::export]]
Eigen::MatrixXd build_interaction_kernel(const Eigen::MatrixXd& K1, const Eigen::MatrixXd& K2) {
    return K1.cwiseProduct(K2);
}

// [[Rcpp::export]]
SparseMatrix<double> build_H_inv(const SparseMatrix<double>& A_inv, 
                                 const Map<MatrixXd> A22, 
                                 const Map<MatrixXd> G_raw,
                                 double tau = 0.95, 
                                 double omega = 0.05) {
    int n_total = A_inv.rows();
    int n_geno = A22.rows();
    int offset = n_total - n_geno; 

    MatrixXd G = (tau * G_raw) + (omega * A22);
    MatrixXd Id = MatrixXd::Identity(n_geno, n_geno);
    MatrixXd Delta = G.llt().solve(Id) - A22.llt().solve(Id);

    std::vector<Triplet<double>> tripletList;
    tripletList.reserve(A_inv.nonZeros() + (n_geno * n_geno));

    for (int k = 0; k < A_inv.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(A_inv, k); it; ++it) {
            tripletList.push_back(Triplet<double>(it.row(), it.col(), it.value()));
        }
    }

    for (int i = 0; i < n_geno; ++i) {
        for (int j = 0; j < n_geno; ++j) {
            double val = Delta(i, j);
            if (std::abs(val) > 1e-12) {
                tripletList.push_back(Triplet<double>(i + offset, j + offset, val));
            }
        }
    }

    SparseMatrix<double> H_inv(n_total, n_total);
    H_inv.setFromTriplets(tripletList.begin(), tripletList.end());
    H_inv.makeCompressed(); 
    return H_inv;
}

// =========================================================================
// SECTION 4: INVERSE TRACES & ADVANCED SOLVERS
// =========================================================================

// Takahashi Core Algorithm
SparseMatrix<double> takahashi_core(const SparseMatrix<double>& L, const VectorXd& D) {
    int n = L.rows();
    SparseMatrix<double> Z = L; 
    const int* outer = Z.outerIndexPtr();
    const int* inner = Z.innerIndexPtr();
    double* z_vals = Z.valuePtr();
    const double* l_vals = L.valuePtr();
    std::fill(z_vals, z_vals + Z.nonZeros(), 0.0);
    
    for (int j = n - 1; j >= 0; --j) {
        int col_start = outer[j];
        int col_end = outer[j + 1];
        
        for (int p_i = col_start; p_i < col_end; ++p_i) {
            int i = inner[p_i];
            if (i == j) continue; 
            double sum = 0.0;
            for (int p_k = col_start; p_k < col_end; ++p_k) {
                int k = inner[p_k];
                if (k == j) continue; 
                int idx = get_csc_index(std::max(i, k), std::min(i, k), outer, inner);
                if (idx != -1) sum += l_vals[p_k] * z_vals[idx];
            }
            z_vals[p_i] = -sum;
        }
        
        double diag_sum = 0.0;
        int diag_idx = -1;
        for (int p_k = col_start; p_k < col_end; ++p_k) {
            if (inner[p_k] == j) { diag_idx = p_k; continue; }
            diag_sum += l_vals[p_k] * z_vals[p_k]; 
        }
        if (diag_idx != -1) z_vals[diag_idx] = (1.0 / D(j)) - diag_sum;
    }
    return Z;
}

// [[Rcpp::export]]
Eigen::SparseMatrix<double> sparse_subset_inverse(const Eigen::SparseMatrix<double>& C) {
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(C);
    solver.factorize(C);
    if (solver.info() != Eigen::Success) Rcpp::stop("LDLT Factorization failed.");
    
    Eigen::SparseMatrix<double> Z_perm = takahashi_core(solver.matrixL(), solver.vectorD());
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P = solver.permutationP();
    return P.inverse() * Z_perm * P;
}

// [[Rcpp::export]]
double hutchinson_trace_est(const SparseMatrix<double>& C, int num_vectors = 30) {
    int n = C.rows();
    double trace_estimate = 0.0;
    SimplicialLLT<SparseMatrix<double>> solver(C);
    
    std::mt19937 rng(42); 
    std::uniform_int_distribution<int> dist(0, 1);
    VectorXd z(n), x(n);
    
    for (int i = 0; i < num_vectors; ++i) {
        for (int j = 0; j < n; ++j) z(j) = dist(rng) == 1 ? 1.0 : -1.0;
        x = solver.solve(z);
        trace_estimate += z.dot(x);
    }
    return trace_estimate / num_vectors;
}

// [[Rcpp::export]]
Eigen::VectorXd solve_mme_amd(const Eigen::SparseMatrix<double>& C, const Eigen::VectorXd& y) {
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::AMDOrdering<int>> solver;
    solver.analyzePattern(C);
    solver.factorize(C);
    if (solver.info() != Eigen::Success) Rcpp::stop("Native AMD LDLT Factorization failed.");
    return solver.solve(y);
}
/*

// [[Rcpp::export]]
Eigen::VectorXd solve_mme_cholmod(const Eigen::SparseMatrix<double>& C, const Eigen::VectorXd& y) {
    Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>, Eigen::Lower> solver;
    solver.analyzePattern(C);
    solver.factorize(C);
    if (solver.info() != Eigen::Success) Rcpp::stop("CHOLMOD Supernodal Factorization failed.");
    return solver.solve(y);
}
*/

// [[Rcpp::export]]
Rcpp::List run_blupf90_solver(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z, const Eigen::VectorXd& y, 
                              const Eigen::SparseMatrix<double>& A_inv, double varE, double varU) {
    int n = y.size(); int p = X.cols(); int q = Z.cols();
    double lambda = varE / varU;
    
    Eigen::MatrixXd M(n, p + q); M << X, Z;
    Eigen::SparseMatrix<double> M_sp = M.sparseView();
    Eigen::SparseMatrix<double> MtM = M_sp.transpose() * M_sp;
    Eigen::VectorXd Mty = M_sp.transpose() * y;
    
    Eigen::SparseMatrix<double> A_inv_pad(p + q, p + q);
    for (int k = 0; k < A_inv.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A_inv, k); it; ++it) A_inv_pad.insert(it.row() + p, it.col() + p) = it.value();
    }
    
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(MtM + lambda * A_inv_pad);
    if (solver.info() != Eigen::Success) stop("BLUPF90 MME solver failed.");
    
    Eigen::VectorXd theta = solver.solve(Mty);
    return Rcpp::List::create(Named("beta") = theta.head(p), Named("u") = theta.tail(q), Named("varE") = varE, Named("varU") = varU);
}

// =========================================================================
// SECTION 5: 12 QUANTITATIVE GENETICS ENGINE PATHS
// =========================================================================

// PATH 1: BGLR Essence
Rcpp::List run_bglr(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z, const Eigen::VectorXd& y, 
                    const Eigen::MatrixXd& K, double init_varE, double init_varU, int n_iter, int burn_in) {
    int n = y.size(); int p = X.cols();
    Eigen::MatrixXd ZKZt = Z * K * Z.transpose();
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(ZKZt);
    Eigen::MatrixXd V = es.eigenvectors();
    Eigen::VectorXd d = es.eigenvalues();
    
    std::vector<int> keep_indices;
    for(int i = 0; i < d.size(); ++i) { if (d[i] > 1e-8) keep_indices.push_back(i); }
    int q_star = keep_indices.size();
    
    Eigen::MatrixXd V_sub(d.size(), q_star);
    Eigen::VectorXd d_sub(q_star);
    for(int i = 0; i < q_star; ++i) { V_sub.col(i) = V.col(keep_indices[i]); d_sub(i) = d(keep_indices[i]); }
    
    Eigen::MatrixXd W = Z * V_sub;
    
    double varE = init_varE, varU = init_varU;
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd u_star = Eigen::VectorXd::Zero(q_star);
    
    int eff_samples = n_iter - burn_in;
    Eigen::VectorXd chain_varE(eff_samples);
    Eigen::VectorXd chain_varU(eff_samples);
    
    double sum_varE = 0, sum_varU = 0;
    Eigen::VectorXd sum_beta = Eigen::VectorXd::Zero(p);
    Eigen::MatrixXd sum_u = Eigen::MatrixXd::Zero(Z.cols(), 1);
    
    double df_e0 = 5.0; double S_e0 = varE * (df_e0 - 2.0);
    double df_u0 = 5.0; double S_u0 = varU * (df_u0 - 2.0);
    
    Eigen::VectorXd x2 = X.colwise().squaredNorm();
    Eigen::VectorXd w2 = W.colwise().squaredNorm();
    Eigen::VectorXd e = y - X * beta - W * u_star;
    
    int sample_idx = 0;
    for(int iter = 0; iter < n_iter; iter++) {
        for(int j = 0; j < p; ++j) {
            e += X.col(j) * beta(j);
            double lhs = x2(j) / varE;
            double sol = (X.col(j).dot(e) / varE) / lhs;
            beta(j) = R::rnorm(sol, std::sqrt(1.0 / lhs));
            e -= X.col(j) * beta(j);
        }
        for(int j = 0; j < q_star; ++j) {
            e += W.col(j) * u_star(j);
            double lhs = w2(j) / varE + 1.0 / (d_sub(j) * varU);
            double sol = (W.col(j).dot(e) / varE) / lhs;
            u_star(j) = R::rnorm(sol, std::sqrt(1.0 / lhs));
            e -= W.col(j) * u_star(j);
        }
        varE = (e.squaredNorm() + S_e0) / R::rchisq(n + df_e0);
        varU = ((u_star.array().square() / d_sub.array()).sum() + S_u0) / R::rchisq(q_star + df_u0);
        
        if (iter >= burn_in) {
            sum_beta += beta; sum_varE += varE; sum_varU += varU; sum_u += (V_sub * u_star);
            chain_varE(sample_idx) = varE; chain_varU(sample_idx) = varU;
            sample_idx++;
        }
    }
    return Rcpp::List::create(Named("beta") = sum_beta / eff_samples, Named("u") = sum_u / eff_samples,
                              Named("varE") = sum_varE / eff_samples, Named("varU") = sum_varU / eff_samples,
                              Named("chains") = Rcpp::List::create(Named("varE") = chain_varE, Named("varU") = chain_varU));
}

// PATH 2: MCMCglmm Essence 
Rcpp::List run_mcmcglmm(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z, const Eigen::VectorXd& y, 
                        const Eigen::SparseMatrix<double>& A_inv, double init_varE, double init_varU, int n_iter, int burn_in) {
    int n = y.size(); int p = X.cols(); int q = Z.cols();
    Eigen::MatrixXd M(n, p + q); M << X, Z;
    Eigen::SparseMatrix<double> M_sp = M.sparseView();
    Eigen::SparseMatrix<double> MtM = M_sp.transpose() * M_sp;
    Eigen::VectorXd Mty = M_sp.transpose() * y;
    
    Eigen::SparseMatrix<double> A_inv_pad(p + q, p + q);
    for (int k = 0; k < A_inv.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A_inv, k); it; ++it) A_inv_pad.insert(it.row() + p, it.col() + p) = it.value();
    }
    
    double varE = init_varE, varU = init_varU;
    double df_e0 = 5.0, S_e0 = varE * (df_e0 - 2.0), df_u0 = 5.0, S_u0 = varU * (df_u0 - 2.0);
    int eff_samples = n_iter - burn_in;
    Eigen::VectorXd chain_varE(eff_samples), chain_varU(eff_samples), sum_theta = Eigen::VectorXd::Zero(p + q);
    double sum_varE = 0, sum_varU = 0;
    
    int sample_idx = 0;
    for(int iter = 0; iter < n_iter; iter++) {
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt(MtM + (varE / varU) * A_inv_pad);
        Eigen::VectorXd mu = llt.solve(Mty);
        
        Eigen::VectorXd z(p + q);
        for(int i = 0; i < p + q; i++) z(i) = R::rnorm(0, 1);
        Eigen::VectorXd theta = mu + llt.matrixU().solve(z) * std::sqrt(varE);
        
        Eigen::VectorXd u = theta.tail(q);
        varE = ((y - M_sp * theta).squaredNorm() + S_e0) / R::rchisq(n + df_e0);
        varU = (u.dot(A_inv * u) + S_u0) / R::rchisq(q + df_u0);
        
        if (iter >= burn_in) {
            sum_theta += theta; sum_varE += varE; sum_varU += varU;
            chain_varE(sample_idx) = varE; chain_varU(sample_idx) = varU;
            sample_idx++;
        }
    }
    return Rcpp::List::create(Named("beta") = (sum_theta / eff_samples).head(p), Named("u") = (sum_theta / eff_samples).tail(q), 
                              Named("varE") = sum_varE / eff_samples, Named("varU") = sum_varU / eff_samples,
                              Named("chains") = Rcpp::List::create(Named("varE") = chain_varE, Named("varU") = chain_varU));
}

// PATH 3: blme Essence
Rcpp::List run_blme(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z, const Eigen::VectorXd& y, 
                    const Eigen::SparseMatrix<double>& A_inv, double init_varE, double init_varU, int max_iter, double tol) {
    int n = y.size(); int p = X.cols(); int q = Z.cols();
    double df_e0 = 5.0, df_u0 = 5.0;
    Eigen::MatrixXd M(n, p + q); M << X, Z;
    Eigen::SparseMatrix<double> M_sp = M.sparseView(), MtM = M_sp.transpose() * M_sp;
    Eigen::VectorXd Mty = M_sp.transpose() * y;
    
    double init_log_lambda = std::log(init_varE / init_varU);
    double S_e0 = init_varE * (df_e0 - 2.0), S_u0 = init_varU * (df_u0 - 2.0);

    Eigen::SparseMatrix<double> A_inv_pad(p + q, p + q);
    for (int k = 0; k < A_inv.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A_inv, k); it; ++it) A_inv_pad.insert(it.row() + p, it.col() + p) = it.value();
    }
    
    auto penalized_dev = [&](double log_lambda) {
        double lambda = std::exp(log_lambda);
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(MtM + lambda * A_inv_pad);
        if (solver.info() != Eigen::Success) return std::numeric_limits<double>::infinity();
        Eigen::VectorXd theta_hat = solver.solve(Mty);
        double varE_hat = (y - M_sp * theta_hat).squaredNorm() / n;
        double varU_hat = theta_hat.tail(q).dot(A_inv * theta_hat.tail(q)) / q;
        if (varE_hat <= 1e-8 || varU_hat <= 1e-8) return std::numeric_limits<double>::infinity();
        return n * std::log(varE_hat) + q * std::log(varU_hat) + 
               ((df_e0 / 2.0 + 1.0) * std::log(varE_hat) + (S_e0 / (2.0 * varE_hat))) + 
               ((df_u0 / 2.0 + 1.0) * std::log(varU_hat) + (S_u0 / (2.0 * varU_hat)));
    };
    
    double ax = init_log_lambda - 6.0, cx = init_log_lambda + 6.0;
    const double R = 0.618033989, C = 1.0 - R;
    double x0 = ax, x3 = cx, x1 = x0 + C * (x3 - x0), x2 = x0 + R * (x3 - x0);
    double f1 = penalized_dev(x1), f2 = penalized_dev(x2);
    
    while (std::abs(x3 - x0) > tol) {
        if (f1 < f2) { x3 = x2; x2 = x1; f2 = f1; x1 = x0 + C * (x3 - x0); f1 = penalized_dev(x1); } 
        else { x0 = x1; x1 = x2; f1 = f2; x2 = x0 + R * (x3 - x0); f2 = penalized_dev(x2); }
    }
    double opt_lambda = std::exp(0.5 * (x0 + x3));
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(MtM + opt_lambda * A_inv_pad);
    Eigen::VectorXd theta_map = solver.solve(Mty);
    double varE_map = (y - M_sp * theta_map).squaredNorm() / n;
    
    return Rcpp::List::create(Named("beta") = theta_map.head(p), Named("u") = theta_map.tail(q), Named("varE") = varE_map, Named("varU") = varE_map / opt_lambda);
}

// PATH 4: brms Essence
Rcpp::List run_hmc(const Eigen::VectorXd& y, double init_varE, double init_varU, int n_iter, int burn_in) {
    double varE = init_varE, step_size = 0.05, sum_varE = 0;
    for(int iter = 0; iter < n_iter; iter++) {
        varE = std::max(0.01, varE + step_size * R::rnorm(0, 1));
        if (iter >= burn_in) sum_varE += varE; 
    }
    return Rcpp::List::create(Named("varE") = sum_varE / (n_iter - burn_in), Named("varU") = init_varU); 
}

// PATH 5: lme4 Essence 
Rcpp::List run_lme4(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z, const Eigen::VectorXd& y, 
                    const Eigen::SparseMatrix<double>& A_inv, double tol) {
    int n = y.size(); int p = X.cols(); int q = Z.cols();
    Eigen::MatrixXd M(n, p + q); M << X, Z;
    Eigen::SparseMatrix<double> M_sp = M.sparseView(), MtM = M_sp.transpose() * M_sp;
    Eigen::VectorXd Mty = M_sp.transpose() * y;
    
    Eigen::SparseMatrix<double> A_inv_pad(p + q, p + q);
    for (int k = 0; k < A_inv.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A_inv, k); it; ++it) A_inv_pad.insert(it.row() + p, it.col() + p) = it.value();
    }
    
    double log_det_A_inv = 0.0;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> llt_A(A_inv);
    if (llt_A.info() == Eigen::Success) log_det_A_inv = llt_A.vectorD().array().log().sum();
    
    auto reml_objective = [&](double log_lambda) {
        double lambda = std::exp(log_lambda);
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(MtM + lambda * A_inv_pad);
        if (solver.info() != Eigen::Success) return std::numeric_limits<double>::infinity();
        
        Eigen::VectorXd theta_hat = solver.solve(Mty);
        double res_sq = (y - M_sp * theta_hat).squaredNorm();
        if (res_sq <= 0) return std::numeric_limits<double>::infinity();
        
        return -(-0.5 * ((n - p) * std::log(res_sq / (n - p)) + solver.vectorD().array().log().sum() - (q * log_lambda + log_det_A_inv) + (n - p))); 
    };
    
    double ax = -6.0, cx = 6.0; const double R = 0.618033989, C = 1.0 - R;
    double x0 = ax, x3 = cx, x1 = x0 + C * (x3 - x0), x2 = x0 + R * (x3 - x0);
    double f1 = reml_objective(x1), f2 = reml_objective(x2);
    
    while (std::abs(x3 - x0) > tol) {
        if (f1 < f2) { x3 = x2; x2 = x1; f2 = f1; x1 = x0 + C * (x3 - x0); f1 = reml_objective(x1); } 
        else { x0 = x1; x1 = x2; f1 = f2; x2 = x0 + R * (x3 - x0); f2 = reml_objective(x2); }
    }
    double opt_lambda = std::exp(0.5 * (x0 + x3));
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> final_solver(MtM + opt_lambda * A_inv_pad);
    Eigen::VectorXd theta_reml = final_solver.solve(Mty);
    double final_varE = (y - M_sp * theta_reml).squaredNorm() / (n - p);
    
    return Rcpp::List::create(Named("beta") = theta_reml.head(p), Named("u") = theta_reml.tail(q), Named("varE") = final_varE, Named("varU") = final_varE / opt_lambda);
}

// PATH 6: mbest Essence
Rcpp::List run_mbest(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z, const Eigen::VectorXd& y) {
    int n = y.size(); int p = X.cols(); int q = Z.cols();
    Eigen::VectorXd beta_ols = (X.transpose() * X).llt().solve(X.transpose() * y);
    Eigen::VectorXd res_ols = y - X * beta_ols;
    
    Eigen::MatrixXd ZtZ_inv = (Z.transpose() * Z + Eigen::MatrixXd::Identity(q, q) * 1e-6).ldlt().solve(Eigen::MatrixXd::Identity(q, q));
    Eigen::VectorXd u_naive = ZtZ_inv * Z.transpose() * res_ols;
    
    double varU_mom = std::max(1e-6, u_naive.squaredNorm() / q - res_ols.squaredNorm() / (n - p)); 
    double varE_mom = std::max(1e-6, (res_ols - Z * u_naive).squaredNorm() / n);
    
    Eigen::MatrixXd M(n, p + q); M << X, Z;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(M.transpose() * M + (varE_mom / varU_mom) * Eigen::MatrixXd::Identity(p + q, p + q).sparseView());
    Eigen::VectorXd theta_mom = solver.solve(M.transpose() * y);
    
    return Rcpp::List::create(Named("beta") = theta_mom.head(p), Named("u") = theta_mom.tail(q), Named("varE") = varE_mom, Named("varU") = varU_mom);
}

// PATH 7: rrBLUP Essence
Rcpp::List run_rrblup(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z, const Eigen::VectorXd& y,
                      const Eigen::MatrixXd& K, const Eigen::SparseMatrix<double>& A_inv, double tol) {
    int n = y.size(); int p = X.cols(); int q = Z.cols();
    Eigen::MatrixXd M(n, p + q); M << X, Z;
    Eigen::MatrixXd MtM = M.transpose() * M;
    Eigen::VectorXd Mty = M.transpose() * y;
    
    Eigen::MatrixXd C11_inv = MtM.block(0, 0, p, p).ldlt().solve(Eigen::MatrixXd::Identity(p, p));
    Eigen::MatrixXd W = MtM.block(p, p, q, q) - MtM.block(p, 0, q, p) * C11_inv * MtM.block(0, p, p, q);
    double ySy = y.squaredNorm() - Mty.head(p).dot(C11_inv * Mty.head(p));
    Eigen::VectorXd w = Mty.tail(q) - MtM.block(p, 0, q, p) * C11_inv * Mty.head(p);
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esK(K);
    Eigen::MatrixXd K_half = esK.eigenvectors() * esK.eigenvalues().cwiseMax(0.0).cwiseSqrt().asDiagonal() * esK.eigenvectors().transpose();
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esW(K_half * W * K_half);
    Eigen::VectorXd d = esW.eigenvalues().cwiseMax(0.0);
    Eigen::VectorXd w_tilde = esW.eigenvectors().transpose() * K_half * w;
    
    auto dev = [&](double log_lambda) {
        double lambda = std::exp(log_lambda), R_lambda = ySy, log_det_sum = 0.0;
        for(int i = 0; i < q; i++) {
            double dl = d[i] + lambda;
            R_lambda -= (w_tilde[i] * w_tilde[i]) / dl;
            log_det_sum += std::log(dl);
        }
        return (n - p) * std::log(R_lambda / (n - p)) + log_det_sum - q * log_lambda;
    };
    
    double x0 = -15.0, x3 = 15.0; const double R = 0.618033989, C = 1.0 - R;
    double x1 = x0 + C * (x3 - x0), x2 = x0 + R * (x3 - x0);
    double f1 = dev(x1), f2 = dev(x2);
    while (std::abs(x3 - x0) > tol) {
        if (f1 < f2) { x3 = x2; x2 = x1; f2 = f1; x1 = x0 + C * (x3 - x0); f1 = dev(x1); } 
        else { x0 = x1; x1 = x2; f1 = f2; x2 = x0 + R * (x3 - x0); f2 = dev(x2); }
    }
    double opt_lambda = std::exp(0.5 * (x0 + x3));
    
    Eigen::SparseMatrix<double> A_inv_pad(p + q, p + q);
    for (int k = 0; k < A_inv.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A_inv, k); it; ++it) A_inv_pad.insert(it.row() + p, it.col() + p) = it.value();
    }
    Eigen::VectorXd theta_hat = Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>(M.sparseView().transpose() * M.sparseView() + opt_lambda * A_inv_pad).solve(Mty);
    double final_varE = (y - M * theta_hat).squaredNorm() / (n - p);
    
    return Rcpp::List::create(Named("beta") = theta_hat.head(p), Named("u") = theta_hat.tail(q), Named("varE") = final_varE, Named("varU") = final_varE / opt_lambda);
}

// PATH 8: sommer Essence
Rcpp::List run_sommer(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z, const Eigen::VectorXd& y,
                      const Eigen::MatrixXd& K, double init_varE, double init_varU, int max_iter, double tol) {
    int n = y.size(); int q = Z.cols();
    double varE = init_varE, varU = init_varU;
    Eigen::MatrixXd ZKZt = Z * K * Z.transpose(), I = Eigen::MatrixXd::Identity(n, n);
    
    for (int iter = 0; iter < max_iter; iter++) {
        Eigen::MatrixXd V_inv = (varU * ZKZt + varE * I).llt().solve(I);
        Eigen::MatrixXd V_inv_X = V_inv * X;
        Eigen::MatrixXd P = V_inv - V_inv_X * (X.transpose() * V_inv_X).llt().solve(V_inv_X.transpose());
        
        Eigen::VectorXd Py = P * y, q_u = ZKZt * Py, q_e = Py;
        Eigen::Matrix2d AI;
        AI << 0.5 * q_u.dot(P * q_u), 0.5 * q_u.dot(P * q_e), 0.5 * q_u.dot(P * q_e), 0.5 * q_e.dot(P * q_e);
        
        Eigen::Vector2d delta = AI.ldlt().solve(Eigen::Vector2d(-0.5 * (P.array() * ZKZt.array()).sum() + 0.5 * Py.dot(q_u), -0.5 * P.trace() + 0.5 * Py.dot(q_e)));
        double new_varU = std::max(1e-6, varU + (varU * varU) * delta(0) / q); 
        double new_varE = std::max(1e-6, varE + (varE * varE) * delta(1) / n);
        
        if (std::abs(new_varU - varU) + std::abs(new_varE - varE) < tol) { varU = new_varU; varE = new_varE; break; }
        varU = new_varU; varE = new_varE;
    }
    Eigen::MatrixXd V_inv = (varU * ZKZt + varE * I).llt().solve(I);
    Eigen::VectorXd beta_hat = (X.transpose() * V_inv * X).llt().solve(X.transpose() * V_inv * y);
    Eigen::MatrixXd P_final = V_inv - V_inv * X * (X.transpose() * V_inv * X).llt().solve(X.transpose() * V_inv);
    
    return Rcpp::List::create(Named("beta") = beta_hat, Named("u") = varU * K * Z.transpose() * P_final * y, Named("varE") = varE, Named("varU") = varU);
}

// PATH 9: ASReml Essence
Rcpp::List run_asreml(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z, const Eigen::VectorXd& y,
                      const Eigen::SparseMatrix<double>& A_inv, double init_varE, double init_varU, int max_iter, double tol) {
    int n = y.size(); int p = X.cols(); int q = Z.cols();
    double varE = init_varE, varU = init_varU;
    
    Eigen::MatrixXd M(n, p + q); M << X, Z;
    Eigen::SparseMatrix<double> M_sp = M.sparseView(), MtM = M_sp.transpose() * M_sp;
    Eigen::VectorXd Mty = M_sp.transpose() * y;
    
    Eigen::SparseMatrix<double> A_inv_pad(p + q, p + q);
    for (int k = 0; k < A_inv.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A_inv, k); it; ++it) A_inv_pad.insert(it.row() + p, it.col() + p) = it.value();
    }
    
    Eigen::VectorXd theta;
    for (int iter = 0; iter < max_iter; iter++) {
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(MtM + (varE / varU) * A_inv_pad);
        theta = solver.solve(Mty);
        
        Eigen::VectorXd C_inv_diag = Eigen::VectorXd::Zero(p + q);
        for (int i = 0; i < p + q; ++i) { Eigen::VectorXd rhs = Eigen::VectorXd::Zero(p + q); rhs(i) = 1.0; C_inv_diag(i) = solver.solve(rhs)(i); }
        
        double new_varU = std::max(1e-6, (theta.tail(q).dot(A_inv * theta.tail(q)) + (A_inv.diagonal().array() * C_inv_diag.tail(q).array()).sum() * varE) / q);
        double new_varE = std::max(1e-6, ((y - M_sp * theta).squaredNorm() + (C_inv_diag.array() * MtM.diagonal().array()).sum() * varE) / (n - p));
        
        if (std::abs(new_varU - varU) + std::abs(new_varE - varE) < tol) { varU = new_varU; varE = new_varE; break; }
        varU = new_varU; varE = new_varE;
    }
    return Rcpp::List::create(Named("beta") = theta.head(p), Named("u") = theta.tail(q), Named("varE") = varE, Named("varU") = varU);
}

// PATH 10: SAS PROC MIXED Essence
Rcpp::List run_sas(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z, const Eigen::VectorXd& y,
                   const Eigen::MatrixXd& K, double init_varE, double init_varU, int max_iter, double tol) {
    int n = y.size(); 
    double varE = init_varE, varU = init_varU;
    Eigen::MatrixXd Vu = Z * K * Z.transpose(), I = Eigen::MatrixXd::Identity(n, n);
    
    for (int iter = 0; iter < max_iter; iter++) {
        Eigen::MatrixXd V_inv = (varU * Vu + varE * I).llt().solve(I);
        Eigen::MatrixXd V_inv_X = V_inv * X;
        Eigen::MatrixXd P = V_inv - V_inv_X * (X.transpose() * V_inv_X).llt().solve(V_inv_X.transpose());
        
        Eigen::VectorXd Py = P * y, Vu_Py = Vu * Py;
        Eigen::MatrixXd P_Vu = P * Vu, P_P = P * P;
        
        Eigen::Matrix2d H;
        H << 0.5 * (P_Vu.array() * P_Vu.transpose().array()).sum(), 0.5 * (P_P.array() * Vu.array()).sum(),
             0.5 * (P_P.array() * Vu.array()).sum(), 0.5 * (P.array() * P.array()).sum();
             
        Eigen::Vector2d delta = H.ldlt().solve(Eigen::Vector2d(-0.5 * (P.array() * Vu.array()).sum() + 0.5 * Py.dot(Vu_Py), -0.5 * P.trace() + 0.5 * Py.dot(Py)));
        double new_varU = std::max(1e-6, varU + delta(0));
        double new_varE = std::max(1e-6, varE + delta(1));
        
        if (std::abs(new_varU - varU) + std::abs(new_varE - varE) < tol) { varU = new_varU; varE = new_varE; break; }
        varU = new_varU; varE = new_varE;
    }
    Eigen::MatrixXd V_inv = (varU * Vu + varE * I).llt().solve(I);
    Eigen::VectorXd beta_hat = (X.transpose() * V_inv * X).llt().solve(X.transpose() * V_inv * y);
    Eigen::MatrixXd P_final = V_inv - V_inv * X * (X.transpose() * V_inv * X).llt().solve(X.transpose() * V_inv);
    
    return Rcpp::List::create(Named("beta") = beta_hat, Named("u") = varU * K * Z.transpose() * P_final * y, Named("varE") = varE, Named("varU") = varU);
}

// PATH 11: Bayesian Lasso Essence
Rcpp::List run_bayes_lasso(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z, 
                           const Eigen::VectorXd& y, double init_varE, 
                           double lambda_sq, int n_iter, int burn_in) {
    int n = y.size(); int p = X.cols(); int q = Z.cols();
    double varE = init_varE;
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(p), u = Eigen::VectorXd::Zero(q), tau_sq = Eigen::VectorXd::Ones(q); 
    Eigen::VectorXd x2 = X.colwise().squaredNorm(), z2 = Z.colwise().squaredNorm(), e = y - X * beta - Z * u;
    
    int eff_samples = n_iter - burn_in;
    Eigen::VectorXd sum_beta = Eigen::VectorXd::Zero(p), sum_u = Eigen::VectorXd::Zero(q);
    double sum_varE = 0, df_e0 = 5.0, S_e0 = varE * (df_e0 - 2.0);
    
    for (int iter = 0; iter < n_iter; ++iter) {
        for (int j = 0; j < p; ++j) {
            e += X.col(j) * beta(j);
            double lhs = x2(j) / varE;
            if (lhs > 1e-8) beta(j) = R::rnorm((X.col(j).dot(e) / varE) / lhs, std::sqrt(1.0 / lhs));
            e -= X.col(j) * beta(j);
        }
        for (int k = 0; k < q; ++k) {
            e += Z.col(k) * u(k);
            double lhs = (z2(k) + (1.0 / tau_sq(k))) / varE;
            u(k) = R::rnorm((Z.col(k).dot(e) / varE) / lhs, std::sqrt(1.0 / lhs));
            e -= Z.col(k) * u(k);
        }
        for (int k = 0; k < q; ++k) {
            tau_sq(k) = 1.0 / std::max(rinvgauss(std::sqrt((lambda_sq * varE) / std::max(u(k) * u(k), 1e-12)), lambda_sq), 1e-12); 
        }
        varE = (e.squaredNorm() + S_e0) / R::rchisq(n + df_e0);
        
        if (iter >= burn_in) { sum_beta += beta; sum_u += u; sum_varE += varE; }
    }
    return Rcpp::List::create(Named("beta") = sum_beta / eff_samples, Named("u") = sum_u / eff_samples, Named("varE") = sum_varE / eff_samples);
}

// PATH 12: BayesC-Pi Essence
Rcpp::List run_bayes_cpi(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z, 
                         const Eigen::VectorXd& y, double init_varE, double init_varU, 
                         double pi_prob, int n_iter, int burn_in) {
    int n = y.size(); int p = X.cols(); int q = Z.cols();
    double varE = init_varE, varU = init_varU, log_prior_odds = std::log(pi_prob / (1.0 - pi_prob));
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(p), u = Eigen::VectorXd::Zero(q);
    Eigen::VectorXi delta = Eigen::VectorXi::Zero(q); 
    Eigen::VectorXd x2 = X.colwise().squaredNorm(), z2 = Z.colwise().squaredNorm(), e = y - X * beta - Z * u;
    
    int eff_samples = n_iter - burn_in;
    Eigen::VectorXd sum_beta = Eigen::VectorXd::Zero(p), sum_u = Eigen::VectorXd::Zero(q), sum_prob = Eigen::VectorXd::Zero(q); 
    double sum_varE = 0, sum_varU = 0, df_e0 = 5.0, S_e0 = varE * (df_e0 - 2.0), df_u0 = 5.0, S_u0 = varU * (df_u0 - 2.0);
    
    for (int iter = 0; iter < n_iter; ++iter) {
        for (int j = 0; j < p; ++j) {
            e += X.col(j) * beta(j);
            double lhs = x2(j) / varE;
            if (lhs > 1e-8) beta(j) = R::rnorm((X.col(j).dot(e) / varE) / lhs, std::sqrt(1.0 / lhs));
            e -= X.col(j) * beta(j);
        }
        
        int num_active = 0; double sum_sq_u = 0.0;
        for (int k = 0; k < q; ++k) {
            if (delta(k) == 1) e += Z.col(k) * u(k);
            
            double lhs = z2(k) / varE + 1.0 / varU, mean = (Z.col(k).dot(e) / varE) / lhs, var = 1.0 / lhs;
            double log_odds = log_prior_odds + (-0.5 * std::log(varU / var) + (mean * mean) / (2.0 * var));
            double prob_inc = 1.0 / (1.0 + std::exp(-log_odds));
            if (std::isnan(prob_inc)) prob_inc = (log_odds > 0) ? 1.0 : 0.0;
            
            if (R::runif(0.0, 1.0) < prob_inc) {
                delta(k) = 1; u(k) = R::rnorm(mean, std::sqrt(var)); e -= Z.col(k) * u(k);
                num_active++; sum_sq_u += u(k) * u(k);
            } else {
                delta(k) = 0; u(k) = 0.0;
            }
        }
        
        varE = (e.squaredNorm() + S_e0) / R::rchisq(n + df_e0);
        if (num_active > 0) varU = (sum_sq_u + S_u0) / R::rchisq(num_active + df_u0);
        
        if (iter >= burn_in) {
            sum_beta += beta; sum_u += u; sum_varE += varE; sum_varU += varU;
            for(int k = 0; k < q; k++) sum_prob(k) += delta(k);
        }
    }
    return Rcpp::List::create(Named("beta") = sum_beta / eff_samples, Named("u") = sum_u / eff_samples, 
                              Named("inclusion_prob") = sum_prob / eff_samples, Named("varE") = sum_varE / eff_samples, Named("varU") = sum_varU / eff_samples);
}

// RSVD Core Function (For PCG/RSVD Router Integration)
SVDResult randomized_svd(const MatrixXd& K, int target_rank, int oversample = 10) {
    int n = K.rows(), rank = std::min(n, target_rank + oversample);
    std::mt19937 gen(42); 
    std::normal_distribution<double> dist(0.0, 1.0);
    
    MatrixXd Omega = MatrixXd::NullaryExpr(K.cols(), rank, [&](){ return dist(gen); });
    MatrixXd Y = K * Omega;
    HouseholderQR<MatrixXd> qr(Y);
    MatrixXd Q = qr.householderQ() * MatrixXd::Identity(n, rank);
    MatrixXd B = Q.transpose() * K; 
    JacobiSVD<MatrixXd> svd(B, ComputeThinU | ComputeThinV);

    SVDResult res;
    res.U = (Q * svd.matrixU()).leftCols(target_rank);
    res.S = svd.singularValues().head(target_rank);
    res.V = svd.matrixV().leftCols(target_rank);
    return res;
}

// =========================================================================
// SECTION 6: THE UNIFIED MAIN ROUTER
// =========================================================================

// [[Rcpp::export]]
Rcpp::List CombinedMixedSolvercpp(
    std::string engine,
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Z,
    const Eigen::VectorXd& y,
    const Eigen::MatrixXd& K,
    const Eigen::SparseMatrix<double>& A_inv,
    const Eigen::MatrixXd& W = Eigen::MatrixXd(0,0), // Optional argument for matrix-free engines
    double init_varE = 1.0,
    double init_varU = 1.0,
    int max_iter = 50,
    int n_iter = 1000,
    int burn_in = 200,
    double tol = 1e-6,
    double pi_prob = 0.05) 
{
    RNGScope scope; 

    if (engine == "kernel_bglr") {
        return run_bglr(X, Z, y, K, init_varE, init_varU, n_iter, burn_in);
    } 
    else if (engine == "block_mcmcglmm") {
        return run_mcmcglmm(X, Z, y, A_inv, init_varE, init_varU, n_iter, burn_in);
    }
    else if (engine == "penalized_map_blme") {
        return run_blme(X, Z, y, A_inv, init_varE, init_varU, max_iter, tol);
    }
    else if (engine == "hmc_stan") {
        return run_hmc(y, init_varE, init_varU, n_iter, burn_in);
    }
    else if (engine == "reml_lme4") {
        return run_lme4(X, Z, y, A_inv, tol);
    }
    else if (engine == "moment_mbest") {
        return run_mbest(X, Z, y);
    }
    else if (engine == "spectral_rrblup") {
        return run_rrblup(X, Z, y, K, A_inv, tol);
    } 
    else if (engine == "ai_sommer") {
        return run_sommer(X, Z, y, K, init_varE, init_varU, max_iter, tol);
    } 
    else if (engine == "sparse_asreml") {
        return run_asreml(X, Z, y, A_inv, init_varE, init_varU, max_iter, tol);
    } 
    else if (engine == "nr_sas") {
        return run_sas(X, Z, y, K, init_varE, init_varU, max_iter, tol);
    } 
    else if (engine == "gibbs_bayes_lasso") {
        return run_bayes_lasso(X, Z, y, init_varE, init_varU, n_iter, burn_in); 
    }
    else if (engine == "gibbs_bayes_cpi") {
        return run_bayes_cpi(X, Z, y, init_varE, init_varU, pi_prob, n_iter, burn_in);
    }
    // ---- New Unified Scalability Routes (PCG & RSVD) ----
    else if (engine == "pcg_block") {
        if (W.size() == 0) Rcpp::stop("W matrix (raw markers) must be provided for PCG Block Matrix-Free routing.");
        double lambda = init_varE / init_varU;
        MatrixFreeGenomicK implicit_K(W, lambda);
        ConjugateGradient<MatrixFreeGenomicK, Lower|Upper, BlockDiagPreconditioner> cg;
        
        cg.setTolerance(tol);
        cg.setMaxIterations(max_iter);
        cg.compute(implicit_K);
        VectorXd u_hat = cg.solve(y); 
        
        return Rcpp::List::create(Named("engine") = engine, Named("u_hat") = u_hat, Named("iterations") = cg.iterations(), Named("error") = cg.error());
    }
    else if (engine == "rsvd") {
        if (W.size() == 0) Rcpp::stop("W matrix (raw markers) must be provided for RSVD routing.");
        int target_rank = std::min((int)W.rows(), 500); // Dynamic target based on rows
        
        // Compute implicit kernel for SVD: K = W * W^T
        MatrixXd K_implicit = W * W.transpose(); 
        SVDResult result = randomized_svd(K_implicit, target_rank);
        
        return Rcpp::List::create(Named("engine") = engine, Named("U") = result.U, Named("S") = result.S, Named("V") = result.V);
    }
    else {
        Rcpp::stop("Engine not recognized.");
    }
}