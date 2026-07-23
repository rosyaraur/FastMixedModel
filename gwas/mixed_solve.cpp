// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <cmath>

using namespace Rcpp;
using namespace arma;

// Fast log-space Golden Section Search to replace R's `optimize`
template <typename F>
double optimize_log_gss(F f, double lower, double upper, double tol = 1e-6) {
    const double r = (std::sqrt(5.0) - 1.0) / 2.0;
    double log_l = std::log(lower);
    double log_u = std::log(upper);
    
    double x1 = log_l + (1.0 - r) * (log_u - log_l);
    double x2 = log_l + r * (log_u - log_l);
    
    double f1 = f(std::exp(x1));
    double f2 = f(std::exp(x2));
    
    while (log_u - log_l > tol) {
        if (f1 < f2) {
            log_u = x2; x2 = x1; f2 = f1;
            x1 = log_l + (1.0 - r) * (log_u - log_l);
            f1 = f(std::exp(x1));
        } else {
            log_l = x1; x1 = x2; f1 = f2;
            x2 = log_l + r * (log_u - log_l);
            f2 = f(std::exp(x2));
        }
    }
    return std::exp(0.5 * (log_l + log_u));
}

// [[Rcpp::export]]
List mixed_solve_cpp(arma::vec y,
                     Nullable<NumericMatrix> Z_in = R_NilValue,
                     Nullable<NumericMatrix> K_in = R_NilValue,
                     Nullable<NumericMatrix> X_in = R_NilValue,
                     std::string method = "REML",
                     NumericVector bounds = NumericVector::create(1e-9, 1e9),
                     bool SE = false,
                     bool return_Hinv = false) {
    
    uvec not_NA = find_finite(y);
    int n_full = y.n_elem;
    int n = not_NA.n_elem;
    vec y_sub = y.elem(not_NA);
    
    mat X, Z;
    if (X_in.isNull()) {
        X = ones<mat>(n_full, 1);
    } else {
        X = as<mat>(X_in);
    }
    int p = X.n_cols;
    
    if (Z_in.isNull()) {
        Z = eye<mat>(n_full, n_full);
    } else {
        Z = as<mat>(Z_in);
    }
    int m = Z.n_cols;
    
    mat K;
    bool has_K = false;
    if (!K_in.isNull()) {
        K = as<mat>(K_in);
        has_K = true;
    }
    
    X = X.rows(not_NA);
    Z = Z.rows(not_NA);
    
    mat XtX = X.t() * X;
    if (arma::rank(XtX) < p) {
        stop("X not full rank");
    }
    mat XtXinv = inv_sympd(XtX);
    mat S = eye<mat>(n, n) - X * XtXinv * X.t();
    
    std::string spectral_method = (n <= m + p) ? "eigen" : "cholesky";
    mat B;
    
    if (spectral_method == "cholesky" && has_K) {
        mat K_copy = K;
        K_copy.diag() += 1e-6;
        if (!chol(B, K_copy)) {
            stop("K not positive semi-definite.");
        }
    }
    
    vec phi, theta;
    mat U, Q;
    bool cholesky_success = true;
    
    if (spectral_method == "cholesky") {
        mat ZBt;
        if (has_K) {
            ZBt = Z * B.t();
        } else {
            ZBt = Z;
        }
        
        mat U_ZBt, V_ZBt; vec d_ZBt;
        svd(U_ZBt, d_ZBt, V_ZBt, ZBt); 
        
        U = U_ZBt;
        phi = zeros<vec>(n);
        for(size_t i = 0; i < d_ZBt.n_elem; ++i) phi(i) = std::pow(d_ZBt(i), 2);
        
        mat SZBt = S * ZBt;
        mat U_SZBt, V_SZBt; vec d_SZBt;
        if (!svd(U_SZBt, d_SZBt, V_SZBt, SZBt)) {
            svd(U_SZBt, d_SZBt, V_SZBt, SZBt + 1e-10 * ones<mat>(SZBt.n_rows, SZBt.n_cols));
        }
        
        mat X_Usz = join_rows(X, U_SZBt);
        mat Q_full, R_full;
        qr(Q_full, R_full, X_Usz);
        
        Q = Q_full.cols(p, n - 1);
        mat R = R_full.submat(p, p, p + m - 1, p + m - 1);
        
        vec ans_vec;
        if (!solve(ans_vec, square(R).t(), square(d_SZBt))) {
            cholesky_success = false;
        } else {
            theta = zeros<vec>(n);
            theta.subvec(0, ans_vec.n_elem - 1) = ans_vec;
        }
    }
    
    if (spectral_method == "eigen" || !cholesky_success) {
        double offset = std::sqrt(n);
        mat Hb;
        if (has_K) {
            Hb = Z * K * Z.t() + offset * eye<mat>(n, n);
        } else {
            Hb = Z * Z.t() + offset * eye<mat>(n, n);
        }
                       
        vec eigval; mat eigvec;
        eig_sym(eigval, eigvec, Hb);
        eigval = reverse(eigval);
        eigvec = fliplr(eigvec);
        
        phi = eigval - offset;
        if (phi.min() < -1e-6) stop("K not positive semi-definite.");
        U = eigvec;
        
        mat SHbS = S * Hb * S;
        vec eigval_S; mat eigvec_S;
        eig_sym(eigval_S, eigvec_S, SHbS);
        eigval_S = reverse(eigval_S);
        eigvec_S = fliplr(eigvec_S);
        
        theta = eigval_S.subvec(0, n - p - 1) - offset;
        Q = eigvec_S.cols(0, n - p - 1);
    }
    
    vec omega = Q.t() * y_sub;
    vec omega_sq = square(omega);
    
    auto f_ML = [&](double lambda) {
        return n * std::log(arma::sum(omega_sq / (theta + lambda))) + arma::sum(arma::log(phi + lambda));
    };
    
    auto f_REML = [&](double lambda) {
        return (n - p) * std::log(arma::sum(omega_sq / (theta + lambda))) + arma::sum(arma::log(theta + lambda));
    };
    
    double lambda_opt, df, min_obj;
    if (method == "ML") {
        lambda_opt = optimize_log_gss(f_ML, bounds[0], bounds[1]);
        df = n;
        min_obj = f_ML(lambda_opt);
    } else {
        lambda_opt = optimize_log_gss(f_REML, bounds[0], bounds[1]);
        df = n - p;
        min_obj = f_REML(lambda_opt);
    }
    
    double Vu_opt = arma::sum(omega_sq / (theta + lambda_opt)) / df;
    double Ve_opt = lambda_opt * Vu_opt;
    
    mat Hinv = U * diagmat(1.0 / (phi + lambda_opt)) * U.t();
    mat W = X.t() * Hinv * X;
    vec beta = solve(W, X.t() * Hinv * y_sub);
    
    mat KZt;
    if (has_K) {
        KZt = K * Z.t();
    } else {
        KZt = Z.t();
    }
    
    mat KZt_Hinv = KZt * Hinv;
    vec u = KZt_Hinv * (y_sub - X * beta);
    
    double LL = -0.5 * (min_obj + df + df * std::log(2 * M_PI / df));
    
    List res = List::create(
        Named("Vu") = Vu_opt,
        Named("Ve") = Ve_opt,
        Named("beta") = beta,
        Named("u") = u,
        Named("LL") = LL
    );
    
    if (SE) {
        mat Winv = inv_sympd(W);
        res["beta.SE"] = sqrt(Vu_opt * Winv.diag());
        
        mat WW = KZt_Hinv * KZt.t();
        mat WWW = KZt_Hinv * X;
        
        // Explicitly evaluate to mat before calling .diag()
        vec term3 = mat(WWW * Winv * WWW.t()).diag();
        
        if (has_K) {
            res["u.SE"] = sqrt(Vu_opt * (K.diag() - WW.diag() + term3));
        } else {
            res["u.SE"] = sqrt(Vu_opt * (ones<vec>(m) - WW.diag() + term3));
        }
    }
    if (return_Hinv) res["Hinv"] = Hinv;
    
    return res;
}