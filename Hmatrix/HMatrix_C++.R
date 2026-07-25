
# ==============================================================================
# MODULAR SINGLE-STEP GBLUP SIMULATION SCRIPT 
# ==============================================================================

# ---------------------------------------------------------
# 1. LOAD LIBRARIES & COMPILE C++ ENGINE
# ---------------------------------------------------------
library(BGLR)
library(ggplot2)
library(tidyr)
library(dplyr)
library(scales)
library(Rcpp)

sourceCpp(code = '
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
List fit_rkhs_cpp(arma::vec y, arma::mat K, int nIter, int burnIn) {
    int n = y.n_elem;
    
    arma::uvec miss_idx = arma::find_nonfinite(y);
    arma::uvec obs_idx = arma::find_finite(y);
    
    arma::vec y_star = y;
    double mean_y = arma::mean(y(obs_idx));
    y_star(miss_idx).fill(mean_y);
    
    arma::vec D;
    arma::mat U;
    arma::eig_sym(D, U, K);
    D.elem(arma::find(D < 1e-6)).fill(1e-6); 
    
    arma::vec one = arma::ones<arma::vec>(n);
    arma::vec x_tilde = U.t() * one;
    
    double df_e = 5.0, S_e = 0.5 * arma::var(y(obs_idx));
    double df_u = 5.0, S_u = 0.5 * arma::var(y(obs_idx));
    
    double mu = mean_y;
    arma::vec a = arma::zeros<arma::vec>(n);
    double sigma2_e = S_e;
    double sigma2_u = S_u;
    
    arma::vec post_u = arma::zeros<arma::vec>(n);
    arma::vec post_yHat = arma::zeros<arma::vec>(n);
    int n_samples = 0;
    
    for(int iter = 1; iter <= nIter; iter++) {
        arma::vec y_tilde = U.t() * y_star;
        
        double var_mu = sigma2_e / arma::accu(arma::square(x_tilde));
        double mean_mu = var_mu * arma::accu(x_tilde % (y_tilde - a)) / sigma2_e;
        mu = R::rnorm(mean_mu, std::sqrt(var_mu));
        
        arma::vec var_a = 1.0 / (1.0 / sigma2_e + 1.0 / (D * sigma2_u));
        arma::vec mean_a = var_a % (y_tilde - x_tilde * mu) / sigma2_e;
        for(int i = 0; i < n; i++) {
            a(i) = R::rnorm(mean_a(i), std::sqrt(var_a(i)));
        }
        
        arma::vec u = U * a;
        arma::vec y_pred = mu + u;
        
        for(size_t i = 0; i < miss_idx.n_elem; i++) {
            y_star(miss_idx(i)) = R::rnorm(y_pred(miss_idx(i)), std::sqrt(sigma2_e));
        }
        
        arma::vec e = y_star - y_pred;
        double scale_e = arma::dot(e, e) + S_e;
        sigma2_e = scale_e / R::rchisq(n + df_e);
        
        double scale_u = arma::accu(arma::square(a) / D) + S_u;
        sigma2_u = scale_u / R::rchisq(n + df_u);
        
        if(iter > burnIn) {
            post_u += u;
            post_yHat += y_pred;
            n_samples++;
        }
    }
    
    post_u /= n_samples;
    post_yHat /= n_samples;
    
    return List::create(Named("u") = post_u,
                        Named("yHat") = post_yHat);
}
')

# ---------------------------------------------------------
# MODULE 1: H MATRIX BUILDER
# ---------------------------------------------------------
build_H_matrix <- function(A, G, genotyped_ids, w = 0.95) {
  id_all <- rownames(A)
  id2 <- as.character(genotyped_ids)
  id1 <- setdiff(id_all, id2)
  
  A11 <- A[id1, id1, drop = FALSE]
  A12 <- A[id1, id2, drop = FALSE]
  A21 <- A[id2, id1, drop = FALSE]
  A22 <- A[id2, id2, drop = FALSE]
  
  Gw <- (w * G) + ((1 - w) * A22)
  A22_inv <- solve(A22)
  
  H22 <- Gw
  H12 <- A12 %*% A22_inv %*% Gw
  H21 <- t(H12)
  diff_matrix <- Gw - A22
  H11 <- A11 + (A12 %*% A22_inv %*% diff_matrix %*% A22_inv %*% A21)
  
  H <- matrix(NA, nrow = length(id_all), ncol = length(id_all))
  rownames(H) <- colnames(H) <- id_all
  
  H[id1, id1] <- H11
  H[id1, id2] <- H12
  H[id2, id1] <- H21
  H[id2, id2] <- H22
  
  return(H[id_all, id_all])
}

# ---------------------------------------------------------
# MODULE 2: MODEL FITTER
# ---------------------------------------------------------
fit_rkhs_model <- function(y, K, engine = "BGLR", nIter = 6000, burnIn = 1000) {
  if(engine == "BGLR") {
    # Generate unique temp prefix to prevent file overwriting during loops
    prefix <- file.path(tempdir(), paste0("M_", runif(1), "_"))
    fit <- BGLR(y = y, 
                ETA = list(list(K = K, model = "RKHS")), 
                nIter = nIter, 
                burnIn = burnIn, 
                saveAt = prefix, 
                verbose = FALSE)
    return(fit$yHat)
    
  } else if (engine == "CPP") {
    # Call the embedded C++ function
    fit <- fit_rkhs_cpp(y = y, K = K, nIter = nIter, burnIn = burnIn)
    return(as.numeric(fit$yHat))
    
  } else {
    stop("Engine must be either 'BGLR' or 'CPP'")
  }
}

# ---------------------------------------------------------
# MODULE 3: SCENARIO EVALUATOR
# ---------------------------------------------------------
evaluate_masking_scenario <- function(prop_mask_pheno = 0.20, 
                                      prop_mask_geno = 0.33, 
                                      engine = "BGLR",
                                      nIter = 6000, 
                                      burnIn = 1000,
                                      seed = NULL) {
  
  if(!is.null(seed)) set.seed(seed)
  
  # 1. Load and prep baseline data
  data(wheat, envir = environment())
  A <- wheat.A
  X <- wheat.X
  y_true <- wheat.Y[, 1]
  
  n <- nrow(A)
  id_names <- paste0("Line_", 1:n)
  rownames(A) <- colnames(A) <- id_names
  rownames(X) <- id_names
  names(y_true) <- id_names
  
  # 2. Apply masking logic
  n_test <- floor(n * prop_mask_pheno)
  n_ungenotyped <- floor(n * prop_mask_geno)
  n_genotyped <- n - n_ungenotyped
  
  test_idx <- sample(1:n, n_test)
  y_na <- y_true
  y_na[test_idx] <- NA
  
  genotyped_idx <- sample(1:n, n_genotyped)
  genotyped_ids <- id_names[genotyped_idx]
  
  # 3. Build Kernel Matrices
  X_scaled <- scale(X)
  G_sub <- (tcrossprod(X_scaled) / ncol(X_scaled))[genotyped_ids, genotyped_ids]
  H <- build_H_matrix(A = A, G = G_sub, genotyped_ids = genotyped_ids)
  
  # 4. Fit Models using the modular fitter
  yHat_A <- fit_rkhs_model(y = y_na, K = A, engine = engine, nIter = nIter, burnIn = burnIn)
  yHat_H <- fit_rkhs_model(y = y_na, K = H, engine = engine, nIter = nIter, burnIn = burnIn)
  
  # 5. Extract results
  actual <- y_true[test_idx]
  
  return(data.frame(
    Engine = engine,
    Prop_Masked_Pheno = prop_mask_pheno,
    Prop_Masked_Geno = prop_mask_geno,
    Accuracy_A = cor(yHat_A[test_idx], actual),
    Accuracy_H = cor(yHat_H[test_idx], actual)
  ))
}

# ---------------------------------------------------------
# EXECUTE EXPERIMENT
# ---------------------------------------------------------
mask_levels <- c(0.1, 0.3, 0.5, 0.7)

results_list <- lapply(mask_levels, function(p) {
  cat("Running simulation for", p * 100, "% ungenotyped...\n")
  evaluate_masking_scenario(prop_mask_pheno = 0.20, 
                            prop_mask_geno = p, 
                            engine = "CPP", # Readily swap between CPP and BGLR here
                            nIter = 3000, 
                            burnIn = 500, 
                            seed = 123) 
})

final_results <- do.call(rbind, results_list)
print(final_results)

# ---------------------------------------------------------
# VISUALIZE RESULTS
# ---------------------------------------------------------
plot_data <- final_results %>%
  pivot_longer(
    cols = c(Accuracy_A, Accuracy_H),
    names_to = "Model",
    values_to = "Accuracy"
  ) %>%
  mutate(Model = ifelse(Model == "Accuracy_A", 
                        "Pedigree Only (A Matrix)", 
                        "Single-Step (H Matrix)"))

ggplot(plot_data, aes(x = Prop_Masked_Geno, y = Accuracy, color = Model, group = Model)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  scale_x_continuous(labels = percent_format(accuracy = 1), 
                     breaks = seq(0.1, 0.7, by = 0.2)) +
  scale_color_manual(values = c("Pedigree Only (A Matrix)" = "#E69F00", 
                                "Single-Step (H Matrix)" = "#0072B2")) +
  labs(
    title = paste("Prediction Accuracy (Engine:", final_results$Engine[1], ")"),
    subtitle = "Impact of increasing the proportion of ungenotyped individuals",
    x = "Proportion of Ungenotyped Lines",
    y = "Prediction Accuracy (Pearson r)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    panel.grid.minor = element_blank()
  )
