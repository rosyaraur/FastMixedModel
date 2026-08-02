// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <algorithm> 

using namespace Rcpp;

// =========================================================================
// HELPER: Inverse-Gaussian Random Number Generator (Michael-Schucany-Haas)
// =========================================================================
inline double rinvgauss(double mu, double lambda) {
    if (mu <= 0.0 || lambda <= 0.0) return 1e-10; // Prevent domain errors
    
    double v = R::rnorm(0.0, 1.0);
    double y = v * v;
    double mu_sq = mu * mu;
    
    double x = mu + (mu_sq * y) / (2.0 * lambda) - 
               (mu / (2.0 * lambda)) * std::sqrt(4.0 * mu * lambda * y + mu_sq * y * y);
               
    double z = R::runif(0.0, 1.0);
    if (z <= (mu / (mu + x))) {
        return x;
    } else {
        return mu_sq / x;
    }
}

// =========================================================================
// PATH 1: BGLR Essence (Kernel Diagonalization Gibbs with Chain Storage)
// =========================================================================
Rcpp::List run_bglr(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z, const Eigen::VectorXd& y, 
                    const Eigen::MatrixXd& K, double init_varE, double init_varU, int n_iter, int burn_in) {
    int n = y.size(); int p = X.cols();
    Eigen::MatrixXd ZKZt = Z * K * Z.transpose();
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(ZKZt);
    Eigen::MatrixXd V = es.eigenvectors();
    Eigen::VectorXd d = es.eigenvalues();
    
    std::vector<int> keep_indices;
    for(int i = 0; i < d.size(); ++i) {
        if (d[i] > 1e-8) keep_indices.push_back(i);
    }
    int q_star = keep_indices.size();
    
    Eigen::MatrixXd V_sub(d.size(), q_star);
    Eigen::VectorXd d_sub(q_star);
    for(int i = 0; i < q_star; ++i) {
        V_sub.col(i) = V.col(keep_indices[i]);
        d_sub(i) = d(keep_indices[i]);
    }
    
    Eigen::MatrixXd W = Z * V_sub;
    Eigen::VectorXd y_rot = V.transpose() * y; 
    Eigen::MatrixXd X_rot = V.transpose() * X;
    
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
            sum_beta += beta; 
            sum_varE += varE; 
            sum_varU += varU;
            sum_u += (V_sub * u_star);
            
            chain_varE(sample_idx) = varE;
            chain_varU(sample_idx) = varU;
            sample_idx++;
        }
    }
    
    return Rcpp::List::create(
        Named("beta") = sum_beta / eff_samples, 
        Named("u") = sum_u / eff_samples,
        Named("varE") = sum_varE / eff_samples, 
        Named("varU") = sum_varU / eff_samples,
        Named("chains") = Rcpp::List::create(
            Named("varE") = chain_varE,
            Named("varU") = chain_varU
        )
    );
}

// =========================================================================
// PATH 2: MCMCglmm Essence (Sparse MME Block Gibbs with Chain Storage)
// =========================================================================
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
    double df_e0 = 5.0, S_e0 = varE * (df_e0 - 2.0);
    double df_u0 = 5.0, S_u0 = varU * (df_u0 - 2.0);
    
    int eff_samples = n_iter - burn_in;
    Eigen::VectorXd chain_varE(eff_samples);
    Eigen::VectorXd chain_varU(eff_samples);
    
    Eigen::VectorXd sum_theta = Eigen::VectorXd::Zero(p + q);
    double sum_varE = 0, sum_varU = 0;
    
    int sample_idx = 0;
    for(int iter = 0; iter < n_iter; iter++) {
        double lambda = varE / varU;
        Eigen::SparseMatrix<double> C = MtM + lambda * A_inv_pad;
        
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt(C);
        Eigen::VectorXd mu = llt.solve(Mty);
        
        Eigen::VectorXd z(p + q);
        for(int i = 0; i < p + q; i++) z(i) = R::rnorm(0, 1);
        Eigen::VectorXd theta = mu + llt.matrixU().solve(z) * std::sqrt(varE);
        
        Eigen::VectorXd u = theta.tail(q);
        Eigen::VectorXd res = y - M_sp * theta;
        
        varE = (res.squaredNorm() + S_e0) / R::rchisq(n + df_e0);
        varU = (u.dot(A_inv * u) + S_u0) / R::rchisq(q + df_u0);
        
        if (iter >= burn_in) {
            sum_theta += theta; 
            sum_varE += varE; 
            sum_varU += varU;
            
            chain_varE(sample_idx) = varE;
            chain_varU(sample_idx) = varU;
            sample_idx++;
        }
    }
    
    return Rcpp::List::create(
        Named("beta") = (sum_theta / eff_samples).head(p), 
        Named("u") = (sum_theta / eff_samples).tail(q), 
        Named("varE") = sum_varE / eff_samples, 
        Named("varU") = sum_varU / eff_samples,
        Named("chains") = Rcpp::List::create(
            Named("varE") = chain_varE,
            Named("varU") = chain_varU
        )
    );
}

// =========================================================================
// PATH 3: blme Essence (Penalized MAP with Warm-Start & Adaptive Bounds)
// =========================================================================
Rcpp::List run_blme(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z, const Eigen::VectorXd& y, 
                    const Eigen::SparseMatrix<double>& A_inv, double init_varE, double init_varU, int max_iter, double tol) {
    int n = y.size(); int p = X.cols(); int q = Z.cols();
    double df_e0 = 5.0, df_u0 = 5.0;
    
    Eigen::MatrixXd M(n, p + q); M << X, Z;
    Eigen::SparseMatrix<double> M_sp = M.sparseView();
    Eigen::SparseMatrix<double> MtM = M_sp.transpose() * M_sp;
    Eigen::VectorXd Mty = M_sp.transpose() * y;
    
    Eigen::MatrixXd XtX = X.transpose() * X;
    Eigen::VectorXd beta_ols = XtX.llt().solve(X.transpose() * y);
    Eigen::VectorXd res_ols = y - X * beta_ols;
    double warm_varE = std::max(1e-4, res_ols.squaredNorm() / (n - p));
    double warm_varU = std::max(1e-4, warm_varE * 0.5);
    double init_log_lambda = std::log(warm_varE / warm_varU);
    
    double S_e0 = warm_varE * (df_e0 - 2.0);
    double S_u0 = warm_varU * (df_u0 - 2.0);

    Eigen::SparseMatrix<double> A_inv_pad(p + q, p + q);
    for (int k = 0; k < A_inv.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A_inv, k); it; ++it) A_inv_pad.insert(it.row() + p, it.col() + p) = it.value();
    }
    
    auto penalized_dev = [&](double log_lambda) {
        double lambda = std::exp(log_lambda);
        if (lambda < 1e-8 || lambda > 1e8) return std::numeric_limits<double>::infinity();
        
        Eigen::SparseMatrix<double> C = MtM + lambda * A_inv_pad;
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(C);
        if (solver.info() != Eigen::Success) return std::numeric_limits<double>::infinity();
        
        Eigen::VectorXd theta_hat = solver.solve(Mty);
        
        double varE_hat = (y - M_sp * theta_hat).squaredNorm() / n;
        Eigen::VectorXd u_hat = theta_hat.tail(q);
        double varU_hat = u_hat.dot(A_inv * u_hat) / q;
        if (varE_hat <= 1e-8 || varU_hat <= 1e-8) return std::numeric_limits<double>::infinity();
        
        return n * std::log(varE_hat) + q * std::log(varU_hat) + 
               ((df_e0 / 2.0 + 1.0) * std::log(varE_hat) + (S_e0 / (2.0 * varE_hat))) + 
               ((df_u0 / 2.0 + 1.0) * std::log(varU_hat) + (S_u0 / (2.0 * varU_hat)));
    };
    
    double ax = init_log_lambda - 6.0; 
    double cx = init_log_lambda + 6.0;
    const double R = 0.618033989, C = 1.0 - R;
    double x0 = ax, x3 = cx;
    double x1 = x0 + C * (x3 - x0), x2 = x0 + R * (x3 - x0);
    double f1 = penalized_dev(x1), f2 = penalized_dev(x2);
    
    while (std::abs(x3 - x0) > tol) {
        if (f1 < f2) { x3 = x2; x2 = x1; f2 = f1; x1 = x0 + C * (x3 - x0); f1 = penalized_dev(x1); } 
        else { x0 = x1; x1 = x2; f1 = f2; x2 = x0 + R * (x3 - x0); f2 = penalized_dev(x2); }
    }
    double opt_lambda = std::exp(0.5 * (x0 + x3));
    
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(MtM + opt_lambda * A_inv_pad);
    Eigen::VectorXd theta_map = solver.solve(Mty);
    double varE_map = (y - M_sp * theta_map).squaredNorm() / n;
    
    return Rcpp::List::create(Named("beta") = theta_map.head(p), Named("u") = theta_map.tail(q), 
                              Named("varE") = varE_map, Named("varU") = varE_map / opt_lambda);
}

// =========================================================================
// PATH 4: brms Essence (Placeholder)
// =========================================================================
Rcpp::List run_hmc(const Eigen::VectorXd& y, double init_varE, double init_varU, int n_iter, int burn_in) {
    double varE = init_varE;
    double step_size = 0.05;
    double sum_varE = 0;
    for(int iter = 0; iter < n_iter; iter++) {
        varE = std::max(0.01, varE + step_size * R::rnorm(0, 1));
        if (iter >= burn_in) { sum_varE += varE; }
    }
    return Rcpp::List::create(Named("varE") = sum_varE / (n_iter - burn_in), Named("varU") = init_varU); 
}

// =========================================================================
// PATH 5: lme4 Essence (Profiled REML with Automated Warm-Start & Bounds)
// =========================================================================
Rcpp::List run_lme4(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z, const Eigen::VectorXd& y, 
                    const Eigen::SparseMatrix<double>& A_inv, double tol) {
    int n = y.size(); int p = X.cols(); int q = Z.cols();
    Eigen::MatrixXd M(n, p + q); M << X, Z;
    Eigen::SparseMatrix<double> M_sp = M.sparseView();
    Eigen::SparseMatrix<double> MtM = M_sp.transpose() * M_sp;
    Eigen::VectorXd Mty = M_sp.transpose() * y;
    
    Eigen::MatrixXd XtX = X.transpose() * X;
    Eigen::VectorXd beta_ols = XtX.llt().solve(X.transpose() * y);
    Eigen::VectorXd res_ols = y - X * beta_ols;
    double init_varE = std::max(1e-4, res_ols.squaredNorm() / (n - p));
    double init_varU = std::max(1e-4, init_varE * 0.5); 
    double init_lambda = init_varE / init_varU;
    double init_log_lambda = std::log(init_lambda);

    Eigen::SparseMatrix<double> A_inv_pad(p + q, p + q);
    for (int k = 0; k < A_inv.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A_inv, k); it; ++it) A_inv_pad.insert(it.row() + p, it.col() + p) = it.value();
    }
    
    double log_det_A_inv = 0.0;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> llt_A(A_inv);
    if (llt_A.info() == Eigen::Success) log_det_A_inv = llt_A.vectorD().array().log().sum();
    
    auto reml_objective = [&](double log_lambda) {
        double lambda = std::exp(log_lambda);
        if (lambda < 1e-8 || lambda > 1e8) return std::numeric_limits<double>::infinity();
        
        Eigen::SparseMatrix<double> C = MtM + lambda * A_inv_pad;
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(C);
        if (solver.info() != Eigen::Success) return std::numeric_limits<double>::infinity();
        
        Eigen::VectorXd theta_hat = solver.solve(Mty);
        double res_sq = (y - M_sp * theta_hat).squaredNorm();
        if (res_sq <= 0) return std::numeric_limits<double>::infinity();
        
        double varE_hat = res_sq / (n - p);
        double log_det_C = solver.vectorD().array().log().sum();
        
        return -(-0.5 * ((n - p) * std::log(varE_hat) + log_det_C - (q * log_lambda + log_det_A_inv) + (n - p))); 
    };
    
    double ax = init_log_lambda - 6.0; double cx = init_log_lambda + 6.0;
    const double R = 0.618033989, C = 1.0 - R;
    double x0 = ax, x3 = cx;
    double x1 = x0 + C * (x3 - x0), x2 = x0 + R * (x3 - x0);
    double f1 = reml_objective(x1), f2 = reml_objective(x2);
    
    while (std::abs(x3 - x0) > tol) {
        if (f1 < f2) { x3 = x2; x2 = x1; f2 = f1; x1 = x0 + C * (x3 - x0); f1 = reml_objective(x1); } 
        else { x0 = x1; x1 = x2; f1 = f2; x2 = x0 + R * (x3 - x0); f2 = reml_objective(x2); }
    }
    double opt_lambda = std::exp(0.5 * (x0 + x3));
    
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> final_solver(MtM + opt_lambda * A_inv_pad);
    Eigen::VectorXd theta_reml = final_solver.solve(Mty);
    double final_varE = (y - M_sp * theta_reml).squaredNorm() / (n - p);
    
    return Rcpp::List::create(Named("beta") = theta_reml.head(p), Named("u") = theta_reml.tail(q), 
                              Named("varE") = final_varE, Named("varU") = final_varE / opt_lambda);
}

// =========================================================================
// PATH 6: mbest Essence (Method of Moments - CORRECTED)
// =========================================================================
Rcpp::List run_mbest(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z, const Eigen::VectorXd& y) {
    int n = y.size(); int p = X.cols(); int q = Z.cols();
    
    Eigen::MatrixXd XtX = X.transpose() * X;
    Eigen::VectorXd beta_ols = XtX.llt().solve(X.transpose() * y);
    Eigen::VectorXd res_ols = y - X * beta_ols;
    
    Eigen::MatrixXd ZtZ_inv = (Z.transpose() * Z + Eigen::MatrixXd::Identity(q, q) * 1e-6).ldlt().solve(Eigen::MatrixXd::Identity(q, q));
    Eigen::VectorXd u_naive = ZtZ_inv * Z.transpose() * res_ols;
    
    double varU_mom = std::max(1e-6, u_naive.squaredNorm() / q - res_ols.squaredNorm() / (n - p)); 
    double varE_mom = std::max(1e-6, (res_ols - Z * u_naive).squaredNorm() / n);
    
    Eigen::MatrixXd M(n, p + q); M << X, Z;
    Eigen::MatrixXd MtM = M.transpose() * M;
    Eigen::SparseMatrix<double> A_inv_pad = Eigen::MatrixXd::Identity(p + q, p + q).sparseView();
    
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(MtM + (varE_mom / varU_mom) * A_inv_pad);
    Eigen::VectorXd theta_mom = solver.solve(M.transpose() * y);
    
    return Rcpp::List::create(Named("beta") = theta_mom.head(p), Named("u") = theta_mom.tail(q), 
                              Named("varE") = varE_mom, Named("varU") = varU_mom);
}

// =========================================================================
// PATH 7: rrBLUP Essence (Spectral Decomposition + GSS)
// =========================================================================
Rcpp::List run_rrblup(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z, const Eigen::VectorXd& y,
                      const Eigen::MatrixXd& K, const Eigen::SparseMatrix<double>& A_inv, double tol) {
    int n = y.size(); int p = X.cols(); int q = Z.cols();
    Eigen::MatrixXd M(n, p + q); M << X, Z;
    Eigen::MatrixXd MtM_dense = M.transpose() * M;
    Eigen::VectorXd Mty = M.transpose() * y;
    double yTy = y.squaredNorm();
    
    Eigen::MatrixXd C11 = MtM_dense.block(0, 0, p, p);
    Eigen::MatrixXd C12 = MtM_dense.block(0, p, p, q);
    Eigen::MatrixXd C21 = MtM_dense.block(p, 0, q, p);
    Eigen::MatrixXd C22 = MtM_dense.block(p, p, q, q);
    
    Eigen::VectorXd v1 = Mty.head(p);
    Eigen::VectorXd v2 = Mty.tail(q);
    
    Eigen::MatrixXd C11_inv = C11.ldlt().solve(Eigen::MatrixXd::Identity(p, p));
    Eigen::MatrixXd W = C22 - C21 * C11_inv * C12;
    double ySy = yTy - v1.dot(C11_inv * v1);
    Eigen::VectorXd w = v2 - C21 * C11_inv * v1;
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esK(K);
    Eigen::MatrixXd K_half = esK.eigenvectors() * esK.eigenvalues().cwiseMax(0.0).cwiseSqrt().asDiagonal() * esK.eigenvectors().transpose();
    
    Eigen::MatrixXd W_tilde = K_half * W * K_half;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esW(W_tilde);
    Eigen::VectorXd d = esW.eigenvalues().cwiseMax(0.0);
    Eigen::VectorXd w_tilde = esW.eigenvectors().transpose() * K_half * w;
    
    auto dev = [&](double log_lambda) {
        double lambda = std::exp(log_lambda);
        double R_lambda = ySy, log_det_sum = 0.0;
        for(int i = 0; i < q; i++) {
            double dl = d[i] + lambda;
            R_lambda -= (w_tilde[i] * w_tilde[i]) / dl;
            log_det_sum += std::log(dl);
        }
        if (R_lambda <= 0) return std::numeric_limits<double>::infinity();
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
    Eigen::SparseMatrix<double> M_sp = M.sparseView();
    Eigen::SparseMatrix<double> MtM_sparse = M_sp.transpose() * M_sp;
    
    Eigen::SparseMatrix<double> A_inv_pad(p + q, p + q);
    for (int k = 0; k < A_inv.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A_inv, k); it; ++it) A_inv_pad.insert(it.row() + p, it.col() + p) = it.value();
    }
    
    Eigen::VectorXd theta_hat = Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>(MtM_sparse + opt_lambda * A_inv_pad).solve(Mty);
    double final_varE = (yTy - 2.0 * theta_hat.dot(Mty) + theta_hat.dot(MtM_sparse * theta_hat)) / (n - p);
    
    return Rcpp::List::create(Named("beta") = theta_hat.head(p), Named("u") = theta_hat.tail(q), Named("varE") = final_varE, Named("varU") = final_varE / opt_lambda);
}

// =========================================================================
// PATH 8: sommer Essence (Dense AI-REML)
// =========================================================================
Rcpp::List run_sommer(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z, const Eigen::VectorXd& y,
                      const Eigen::MatrixXd& K, double init_varE, double init_varU, int max_iter, double tol) {
    int n = y.size(); int q = Z.cols();
    double varE = init_varE, varU = init_varU;
    
    Eigen::MatrixXd ZKZt = Z * K * Z.transpose();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
    
    for (int iter = 0; iter < max_iter; iter++) {
        Eigen::MatrixXd V = varU * ZKZt + varE * I;
        Eigen::MatrixXd V_inv = V.llt().solve(I);
        
        Eigen::MatrixXd V_inv_X = V_inv * X;
        Eigen::MatrixXd P = V_inv - V_inv_X * (X.transpose() * V_inv_X).llt().solve(V_inv_X.transpose());
        
        Eigen::VectorXd Py = P * y;
        Eigen::VectorXd q_u = ZKZt * Py, q_e = Py;
        
        double S_u = -0.5 * (P.array() * ZKZt.array()).sum() + 0.5 * Py.dot(q_u);
        double S_e = -0.5 * P.trace() + 0.5 * Py.dot(q_e);
        
        Eigen::Matrix2d AI;
        AI << 0.5 * q_u.dot(P * q_u), 0.5 * q_u.dot(P * q_e),
              0.5 * q_u.dot(P * q_e), 0.5 * q_e.dot(P * q_e);
        
        Eigen::Vector2d delta = AI.ldlt().solve(Eigen::Vector2d(S_u, S_e));
        double new_varU = std::max(1e-6, varU + (varU * varU) * delta(0) / q); 
        double new_varE = std::max(1e-6, varE + (varE * varE) * delta(1) / n);
        
        if (std::abs(new_varU - varU) + std::abs(new_varE - varE) < tol) { varU = new_varU; varE = new_varE; break; }
        varU = new_varU; varE = new_varE;
    }
    
    Eigen::MatrixXd V_inv = (varU * ZKZt + varE * I).llt().solve(I);
    Eigen::VectorXd beta_hat = (X.transpose() * V_inv * X).llt().solve(X.transpose() * V_inv * y);
    Eigen::MatrixXd P_final = V_inv - V_inv * X * (X.transpose() * V_inv * X).llt().solve(X.transpose() * V_inv);
    Eigen::VectorXd u_hat = varU * K * Z.transpose() * P_final * y;
    
    return Rcpp::List::create(Named("beta") = beta_hat, Named("u") = u_hat, Named("varE") = varE, Named("varU") = varU);
}

// =========================================================================
// PATH 9: ASReml Essence (Sparse MME Exact AI-REML)
// =========================================================================
Rcpp::List run_asreml(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z, const Eigen::VectorXd& y,
                      const Eigen::SparseMatrix<double>& A_inv, double init_varE, double init_varU, int max_iter, double tol) {
    int n = y.size(); int p = X.cols(); int q = Z.cols();
    double varE = init_varE, varU = init_varU;
    
    Eigen::MatrixXd M(n, p + q); M << X, Z;
    Eigen::SparseMatrix<double> M_sp = M.sparseView();
    Eigen::SparseMatrix<double> MtM = M_sp.transpose() * M_sp;
    Eigen::VectorXd Mty = M_sp.transpose() * y;
    
    Eigen::SparseMatrix<double> A_inv_pad(p + q, p + q);
    for (int k = 0; k < A_inv.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A_inv, k); it; ++it) A_inv_pad.insert(it.row() + p, it.col() + p) = it.value();
    }
    
    Eigen::VectorXd theta;
    for (int iter = 0; iter < max_iter; iter++) {
        double lambda = varE / varU;
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(MtM + lambda * A_inv_pad);
        theta = solver.solve(Mty);
        Eigen::VectorXd u_hat = theta.tail(q);
        
        Eigen::VectorXd C_inv_diag(p + q);
        for (int i = 0; i < p + q; ++i) {
            Eigen::VectorXd rhs = Eigen::VectorXd::Zero(p + q);
            rhs(i) = 1.0;
            C_inv_diag(i) = solver.solve(rhs)(i);
        }
        
        double new_varU = std::max(1e-6, (u_hat.dot(A_inv * u_hat) + (A_inv.diagonal().array() * C_inv_diag.tail(q).array()).sum() * varE) / q);
        double new_varE = std::max(1e-6, ((y - M_sp * theta).squaredNorm() + (C_inv_diag.array() * MtM.diagonal().array()).sum() * varE) / (n - p));
        
        if (std::abs(new_varU - varU) + std::abs(new_varE - varE) < tol) { varU = new_varU; varE = new_varE; break; }
        varU = new_varU; varE = new_varE;
    }
    return Rcpp::List::create(Named("beta") = theta.head(p), Named("u") = theta.tail(q), Named("varE") = varE, Named("varU") = varU);
}

// =========================================================================
// PATH 10: SAS PROC MIXED Essence (Exact Newton-Raphson)
// =========================================================================
Rcpp::List run_sas(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z, const Eigen::VectorXd& y,
                   const Eigen::MatrixXd& K, double init_varE, double init_varU, int max_iter, double tol) {
    int n = y.size(); 
    double varE = init_varE, varU = init_varU;
    
    Eigen::MatrixXd Vu = Z * K * Z.transpose();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
    
    for (int iter = 0; iter < max_iter; iter++) {
        Eigen::MatrixXd V_inv = (varU * Vu + varE * I).llt().solve(I);
        Eigen::MatrixXd V_inv_X = V_inv * X;
        Eigen::MatrixXd P = V_inv - V_inv_X * (X.transpose() * V_inv_X).llt().solve(V_inv_X.transpose());
        
        Eigen::VectorXd Py = P * y;
        Eigen::VectorXd Vu_Py = Vu * Py;
        
        Eigen::MatrixXd P_Vu = P * Vu;
        Eigen::MatrixXd P_P = P * P;
        
        double Su = -0.5 * (P.array() * Vu.array()).sum() + 0.5 * Py.dot(Vu_Py);
        double Se = -0.5 * P.trace() + 0.5 * Py.dot(Py);
        
        Eigen::Matrix2d H;
        H << 0.5 * (P_Vu.array() * P_Vu.transpose().array()).sum(),
             0.5 * (P_P.array() * Vu.array()).sum(),
             0.5 * (P_P.array() * Vu.array()).sum(),
             0.5 * (P.array() * P.array()).sum();
             
        Eigen::Vector2d delta = H.ldlt().solve(Eigen::Vector2d(Su, Se));
        double new_varU = std::max(1e-6, varU + delta(0));
        double new_varE = std::max(1e-6, varE + delta(1));
        
        if (std::abs(new_varU - varU) + std::abs(new_varE - varE) < tol) { varU = new_varU; varE = new_varE; break; }
        varU = new_varU; varE = new_varE;
    }
    
    Eigen::MatrixXd V_inv = (varU * Vu + varE * I).llt().solve(I);
    Eigen::VectorXd beta_hat = (X.transpose() * V_inv * X).llt().solve(X.transpose() * V_inv * y);
    Eigen::MatrixXd P_final = V_inv - V_inv * X * (X.transpose() * V_inv * X).llt().solve(X.transpose() * V_inv);
    Eigen::VectorXd u_hat = varU * K * Z.transpose() * P_final * y;
    
    return Rcpp::List::create(Named("beta") = beta_hat, Named("u") = u_hat, Named("varE") = varE, Named("varU") = varU);
}

// =========================================================================
// PATH 11: Bayesian Lasso Essence (Scale-Mixture of Normals Gibbs)
// =========================================================================
Rcpp::List run_bayes_lasso(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z, 
                           const Eigen::VectorXd& y, double init_varE, 
                           double lambda_sq, int n_iter, int burn_in) {
    int n = y.size(); int p = X.cols(); int q = Z.cols();
    
    double varE = init_varE;
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd u = Eigen::VectorXd::Zero(q);
    Eigen::VectorXd tau_sq = Eigen::VectorXd::Ones(q); 
    
    Eigen::VectorXd x2 = X.colwise().squaredNorm();
    Eigen::VectorXd z2 = Z.colwise().squaredNorm();
    Eigen::VectorXd e = y - X * beta - Z * u;
    
    int eff_samples = n_iter - burn_in;
    Eigen::VectorXd sum_beta = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd sum_u = Eigen::VectorXd::Zero(q);
    double sum_varE = 0;
    
    double df_e0 = 5.0; double S_e0 = varE * (df_e0 - 2.0);
    
    for (int iter = 0; iter < n_iter; ++iter) {
        
        for (int j = 0; j < p; ++j) {
            e += X.col(j) * beta(j);
            double lhs = x2(j) / varE;
            if (lhs > 1e-8) {
                double rhs = X.col(j).dot(e) / varE;
                beta(j) = R::rnorm(rhs / lhs, std::sqrt(1.0 / lhs));
            }
            e -= X.col(j) * beta(j);
        }
        
        for (int k = 0; k < q; ++k) {
            e += Z.col(k) * u(k);
            double lhs = (z2(k) + (1.0 / tau_sq(k))) / varE;
            double rhs = Z.col(k).dot(e) / varE;
            
            u(k) = R::rnorm(rhs / lhs, std::sqrt(1.0 / lhs));
            e -= Z.col(k) * u(k);
        }
        
        for (int k = 0; k < q; ++k) {
            double u_sq = std::max(u(k) * u(k), 1e-12); 
            double mu_prime = std::sqrt((lambda_sq * varE) / u_sq);
            double inv_tau2 = rinvgauss(mu_prime, lambda_sq);
            tau_sq(k) = 1.0 / std::max(inv_tau2, 1e-12); 
        }
        
        varE = (e.squaredNorm() + S_e0) / R::rchisq(n + df_e0);
        
        if (iter >= burn_in) {
            sum_beta += beta;
            sum_u += u;
            sum_varE += varE;
        }
    }
    
    return Rcpp::List::create(Named("beta") = sum_beta / eff_samples, Named("u") = sum_u / eff_samples, Named("varE") = sum_varE / eff_samples);
}

// =========================================================================
// PATH 12: BayesC-Pi Essence (Spike-and-Slab Selection)
// =========================================================================
Rcpp::List run_bayes_cpi(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z, 
                         const Eigen::VectorXd& y, double init_varE, double init_varU, 
                         double pi_prob, int n_iter, int burn_in) {
    int n = y.size(); int p = X.cols(); int q = Z.cols();
    
    double varE = init_varE, varU = init_varU;
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd u = Eigen::VectorXd::Zero(q);
    Eigen::VectorXi delta = Eigen::VectorXi::Zero(q); // Indicator variable
    
    Eigen::VectorXd x2 = X.colwise().squaredNorm();
    Eigen::VectorXd z2 = Z.colwise().squaredNorm();
    Eigen::VectorXd e = y - X * beta - Z * u;
    
    int eff_samples = n_iter - burn_in;
    Eigen::VectorXd sum_beta = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd sum_u = Eigen::VectorXd::Zero(q);
    Eigen::VectorXd sum_prob = Eigen::VectorXd::Zero(q); 
    double sum_varE = 0, sum_varU = 0;
    
    double df_e0 = 5.0; double S_e0 = varE * (df_e0 - 2.0);
    double df_u0 = 5.0; double S_u0 = varU * (df_u0 - 2.0);
    
    // Convert inclusion probability to prior log odds
    double log_prior_odds = std::log(pi_prob / (1.0 - pi_prob));
    
    for (int iter = 0; iter < n_iter; ++iter) {
        
        // 1. Fixed Effects (Always active)
        for (int j = 0; j < p; ++j) {
            e += X.col(j) * beta(j);
            double lhs = x2(j) / varE;
            if (lhs > 1e-8) {
                double rhs = X.col(j).dot(e) / varE;
                beta(j) = R::rnorm(rhs / lhs, std::sqrt(1.0 / lhs));
            }
            e -= X.col(j) * beta(j);
        }
        
        // 2. Penalized Markers (Spike and Slab via Indicator delta)
        int num_active = 0;
        double sum_sq_u = 0.0;
        
        for (int k = 0; k < q; ++k) {
            if (delta(k) == 1) { e += Z.col(k) * u(k); }
            
            double lhs = z2(k) / varE + 1.0 / varU;
            double rhs = Z.col(k).dot(e) / varE;
            double mean = rhs / lhs;
            double var = 1.0 / lhs;
            
            // Bayes Factor (Log Scale) & Inclusion Probability
            double log_bf = -0.5 * std::log(varU / var) + (mean * mean) / (2.0 * var);
            double log_odds = log_prior_odds + log_bf;
            double prob_inc = 1.0 / (1.0 + std::exp(-log_odds));
            
            // Prevent NaN if exp overflows
            if (std::isnan(prob_inc)) prob_inc = (log_odds > 0) ? 1.0 : 0.0;
            
            if (R::runif(0.0, 1.0) < prob_inc) {
                delta(k) = 1;
                u(k) = R::rnorm(mean, std::sqrt(var));
                e -= Z.col(k) * u(k);
                num_active++;
                sum_sq_u += u(k) * u(k);
            } else {
                delta(k) = 0;
                u(k) = 0.0;
            }
        }
        
        // 3. Variance Updates
        varE = (e.squaredNorm() + S_e0) / R::rchisq(n + df_e0);
        if (num_active > 0) {
            varU = (sum_sq_u + S_u0) / R::rchisq(num_active + df_u0);
        }
        
        if (iter >= burn_in) {
            sum_beta += beta;
            sum_u += u;
            for(int k = 0; k < q; k++) sum_prob(k) += delta(k);
            sum_varE += varE;
            sum_varU += varU;
        }
    }
    
    return Rcpp::List::create(
        Named("beta") = sum_beta / eff_samples,
        Named("u") = sum_u / eff_samples,
        Named("inclusion_prob") = sum_prob / eff_samples,
        Named("varE") = sum_varE / eff_samples,
        Named("varU") = sum_varU / eff_samples
    );
}

// =========================================================================
// MAIN ROUTER: CombinedMixedSolvercpp
// =========================================================================
// [[Rcpp::export]]
Rcpp::List CombinedMixedSolvercpp(
    std::string engine,
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Z,
    const Eigen::VectorXd& y,
    const Eigen::MatrixXd& K,
    const Eigen::SparseMatrix<double>& A_inv,
    double init_varE = 1.0,
    double init_varU = 1.0,
    int max_iter = 50,
    int n_iter = 1000,
    int burn_in = 200,
    double tol = 1e-6,
    double pi_prob = 0.05) // <--- Added parameter for BayesCpi
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
    else {
        Rcpp::stop("Engine not recognized.");
    }
}


// =========================================================================
// 1. Principal Component Analysis (Safe Version)
// =========================================================================
// [[Rcpp::export]]
Eigen::MatrixXd reduce_pca(const Eigen::MatrixXd& Z, int k) {
    // 1. Column-center the matrix
    Eigen::MatrixXd centered = Z.rowwise() - Z.colwise().mean();
    
    // 2. Safety Check: k cannot exceed the rank of the matrix
    int max_k = std::min(centered.rows(), centered.cols());
    if (k > max_k) {
        Rcpp::warning("Requested k is greater than matrix rank. Truncating to maximum possible.");
        k = max_k; 
    }
    
    // 3. Perform SVD 
    // (Note: BDCSVD is fast, but if your Mac STILL crashes, 
    // change BDCSVD to JacobiSVD for maximum stability on Apple Silicon)
    Eigen::BDCSVD<Eigen::MatrixXd> svd(centered, Eigen::ComputeThinU | Eigen::ComputeThinV);
    
    // 4. THE FIX: Explicitly evaluate the matrix into a concrete object
    // Do NOT return the expression (svd.matrixU... * svd.singular...) directly.
    Eigen::MatrixXd Z_reduced = svd.matrixU().leftCols(k) * svd.singularValues().head(k).asDiagonal();
    
    // 5. Safely pass the realized matrix back to R
    return Z_reduced;
}

// =========================================================================
// 2. Partial Least Squares (PLS - NIPALS Algorithm)
// =========================================================================
// Supervised reduction maximizing covariance between features Z and phenotype y.
// [[Rcpp::export]]
Eigen::MatrixXd reduce_pls(const Eigen::MatrixXd& Z, const Eigen::VectorXd& y, int k) {
    int n = Z.rows();
    
    Eigen::MatrixXd T(n, k);   // Latent score matrix
    Eigen::MatrixXd E = Z;     // Deflated feature matrix
    Eigen::VectorXd f = y;     // Deflated response vector
    
    for (int i = 0; i < k; ++i) {
        // Step 1: Calculate weights proportional to covariance
        Eigen::VectorXd w = E.transpose() * f;
        w.normalize();
        
        // Step 2: Calculate latent scores
        Eigen::VectorXd t = E * w;
        
        // Step 3: Calculate loadings
        Eigen::VectorXd p = (E.transpose() * t) / t.squaredNorm();
        
        // Step 4: Deflate matrices
        E -= t * p.transpose();
        
        // Prevent division by zero if vectors become perfectly orthogonal
        double t_norm_sq = t.squaredNorm();
        if (t_norm_sq > 1e-12) {
            double q_scalar = (f.transpose() * t)(0) / t_norm_sq;
            f -= t * q_scalar;
        }
        
        T.col(i) = t;
    }
    
    return T;
}

// =========================================================================
// 3. Random Projections (Johnson-Lindenstrauss Transform)
// =========================================================================
// Ultra-fast, data-agnostic reduction preserving pairwise distances.
// [[Rcpp::export]]
Eigen::MatrixXd reduce_random_projection(const Eigen::MatrixXd& Z, int k) {
    int q = Z.cols();
    Eigen::MatrixXd R(q, k);
    
    // Generate a Gaussian random matrix
    for (int i = 0; i < q; ++i) {
        for (int j = 0; j < k; ++j) {
            R(i, j) = R::rnorm(0.0, 1.0);
        }
    }
    
    // Project and scale
    return (Z * R) / std::sqrt(static_cast<double>(k));
}



// =========================================================================
// KERNEL 1: Linear Environmental Kernel (Analogous to GBLUP)
// =========================================================================
// [[Rcpp::export]]
Eigen::MatrixXd build_linear_kernel(const Eigen::MatrixXd& W) {
    int n = W.rows();
    int q = W.cols();
    Eigen::MatrixXd W_scaled(n, q);
    
    // Standardize each column (mean = 0, variance = 1)
    for(int j = 0; j < q; ++j) {
        double mean = W.col(j).mean();
        Eigen::VectorXd centered = W.col(j).array() - mean;
        double var = centered.squaredNorm() / (n - 1.0);
        double sd = std::sqrt(var);
        
        if(sd > 1e-8) {
            W_scaled.col(j) = centered / sd;
        } else {
            W_scaled.col(j) = centered; // Prevent division by zero for constants
        }
    }
    
    // Compute cross-product and scale by number of variables
    return (W_scaled * W_scaled.transpose()) / static_cast<double>(q);
}

// =========================================================================
// KERNEL 2: Non-Linear Gaussian Kernel (RKHS)
// =========================================================================
// [[Rcpp::export]]
Eigen::MatrixXd build_gaussian_kernel(const Eigen::MatrixXd& W, double h) {
    int n = W.rows();
    int q = W.cols();
    Eigen::MatrixXd W_scaled(n, q);
    
    // Standardize each column
    for(int j = 0; j < q; ++j) {
        double mean = W.col(j).mean();
        Eigen::VectorXd centered = W.col(j).array() - mean;
        double var = centered.squaredNorm() / (n - 1.0);
        double sd = std::sqrt(var);
        
        if(sd > 1e-8) W_scaled.col(j) = centered / sd;
        else W_scaled.col(j) = centered;
    }
    
    Eigen::MatrixXd K(n, n);
    for(int i = 0; i < n; ++i) {
        for(int j = i; j < n; ++j) {
            double dist_sq = (W_scaled.row(i) - W_scaled.row(j)).squaredNorm();
            double val = std::exp(-dist_sq / h);
            K(i, j) = val;
            K(j, i) = val; // Matrix is symmetric
        }
    }
    
    // Add small ridge to diagonal for numerical stability during inversion
    K += Eigen::MatrixXd::Identity(n, n) * 1e-6;
    return K;
}

// =========================================================================
// KERNEL 3: Interaction Matrix (Hadamard Product for GxE)
// =========================================================================
// [[Rcpp::export]]
Eigen::MatrixXd build_interaction_kernel(const Eigen::MatrixXd& K1, const Eigen::MatrixXd& K2) {
    return K1.cwiseProduct(K2);
}
