// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <cmath>

using namespace Rcpp;
using namespace arma;

//' @title Fast GWAS Chunk Evaluator for Distributed Computing
//' @export
// [[Rcpp::export]]
List fastGWASpoly_chunk(const arma::vec& y, 
                        const arma::mat& X_null, 
                        const arma::mat& Hinv, 
                        double Vu, 
                        const arma::mat& M_chunk) {
    
    int n_markers = M_chunk.n_cols;
    int n_null = X_null.n_cols;
    
    vec beta_m(n_markers);
    vec se_m(n_markers);
    vec pval(n_markers);
    
    // Pre-compute constant terms to save CPU cycles on worker nodes
    mat Hinv_Xnull = Hinv * X_null;
    mat Xnull_Hinv_Xnull = X_null.t() * Hinv_Xnull;
    vec Hinv_y = Hinv * y;
    vec Xnull_Hinv_y = X_null.t() * Hinv_y;
    
    for(int i = 0; i < n_markers; ++i) {
        vec m = M_chunk.col(i);
        
        // Build the full W matrix: W = X_full^T * Hinv * X_full
        // Using block matrices prevents having to concatenate large matrices in memory
        vec Hinv_m = Hinv * m;
        mat Xnull_Hinv_m = X_null.t() * Hinv_m;
        double m_Hinv_m = as_scalar(m.t() * Hinv_m);
        
        mat W = zeros<mat>(n_null + 1, n_null + 1);
        W.submat(0, 0, n_null - 1, n_null - 1) = Xnull_Hinv_Xnull;
        W.submat(0, n_null, n_null - 1, n_null) = Xnull_Hinv_m;
        W.submat(n_null, 0, n_null, n_null - 1) = Xnull_Hinv_m.t();
        W(n_null, n_null) = m_Hinv_m;
        
        // Use pseudo-inverse (pinv) to prevent crashes on monomorphic/collinear markers
        mat Winv = pinv(W); 
        
        // RHS = X_full^T * Hinv * y
        vec RHS = zeros<vec>(n_null + 1);
        RHS.subvec(0, n_null - 1) = Xnull_Hinv_y;
        RHS(n_null) = as_scalar(m.t() * Hinv_y);
        
        // Calculate Beta and SE
        vec beta_full = Winv * RHS;
        
        double beta_marker = beta_full(n_null);
        double var_beta_marker = Vu * Winv(n_null, n_null);
        double se_marker = std::sqrt(std::max(0.0, var_beta_marker)); // max() prevents negative floating point errors
        
        // Wald Test
        double z_score = (se_marker > 0.0) ? (beta_marker / se_marker) : 0.0;
        double p = (se_marker > 0.0) ? (2.0 * R::pnorm(-std::abs(z_score), 0.0, 1.0, 1, 0)) : 1.0; 
        
        beta_m(i) = beta_marker;
        se_m(i) = se_marker;
        pval(i) = p;
    }
    
    return List::create(
        Named("beta") = beta_m,
        Named("SE") = se_m,
        Named("p_value") = pval
    );
}