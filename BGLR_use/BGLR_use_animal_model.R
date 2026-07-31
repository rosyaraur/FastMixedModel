# Install packages if missing
# install.packages(c("BGLR", "sommer", "MCMCglmm"))

library(BGLR)
library(sommer)
library(MCMCglmm)

# ==========================================
# 1. SETUP A COMMON DATASET
# ==========================================
# We use the built-in 'mice' dataset from BGLR for a realistic relationship matrix
data(mice, package = "BGLR")
y <- mice.pheno$Obesity.BMI
A <- mice.A # Numerator Relationship Matrix (or Genomic Relationship Matrix)

# MCMCglmm and sommer prefer data frames
pheno_df <- data.frame(id = rownames(A), y = y)

# MCMCglmm specifically requires the INVERSE of the relationship matrix
A_inv <- solve(A)
rownames(A_inv) <- rownames(A)
colnames(A_inv) <- colnames(A)


# ==========================================
# 2. BGLR (Bayesian - Gibbs Sampler)
# ==========================================
# Setup the linear predictor using Reproducing Kernel Hilbert Spaces (RKHS)
ETA_bglr <- list(list(K = A, model = "RKHS"))

# Fit model
fit_bglr <- BGLR(y = y, 
                 ETA = ETA_bglr, 
                 nIter = 5000, 
                 burnIn = 1000, 
                 verbose = FALSE)

# Extract Heritability
var_g_bglr <- fit_bglr$ETA[[1]]$varU
var_e_bglr <- fit_bglr$varE
h2_bglr <- var_g_bglr / (var_g_bglr + var_e_bglr)


# ==========================================
# 3. sommer (Frequentist - REML)
# ==========================================
# sommer uses the vsr() function to map the covariance matrix (Gu) to the ID
fit_sommer <- mmer(fixed = y ~ 1, 
                   random = ~ vsr(id, Gu = A), 
                   rcov = ~ vsr(units),
                   data = pheno_df, 
                   verbose = FALSE)

# Extract Heritability from the variance components summary
vc_sommer <- summary(fit_sommer)$varcomp
var_g_sommer <- vc_sommer[1, "VarComp"] # Genetic variance
var_e_sommer <- vc_sommer[2, "VarComp"] # Residual variance
h2_sommer <- var_g_sommer / (var_g_sommer + var_e_sommer)


# ==========================================
# 4. MCMCglmm (Bayesian - MCMC)
# ==========================================
# Define priors for variance components (V = 1, nu = 0.002 is a standard uninformative prior)
library(Matrix) # Required for sparse matrices

# 1. Setup the prior
prior_mcmc <- list(G = list(G1 = list(V = 1, nu = 0.002)), 
                   R = list(V = 1, nu = 0.002))

# 2. Invert and convert the A matrix to sparse format
A_inv_base <- solve(A)
A_inv_sparse <- as(A_inv_base, "dgCMatrix")
rownames(A_inv_sparse) <- rownames(A)
colnames(A_inv_sparse) <- colnames(A)

# 3. Fit the model using the SPARSE matrix
fit_mcmc <- MCMCglmm(fixed = y ~ 1, 
                     random = ~ id, 
                     ginverse = list(id = A_inv_sparse), # <--- This is the crucial change
                     data = pheno_df, 
                     prior = prior_mcmc, 
                     nitt = 5000, 
                     burnin = 1000, 
                     verbose = FALSE)

# 4. Extract Heritability
var_g_mcmc <- mean(fit_mcmc$VCV[, "id"])
var_e_mcmc <- mean(fit_mcmc$VCV[, "units"])
h2_mcmc <- var_g_mcmc / (var_g_mcmc + var_e_mcmc)

print(paste("Heritability (MCMCglmm):", round(h2_mcmc, 3)))


# ==========================================
# 5. COMPARE HERITABILITY ESTIMATES
# ==========================================
cat("Heritability (BGLR):     ", round(h2_bglr, 3), "\n")
cat("Heritability (sommer):   ", round(h2_sommer, 3), "\n")
cat("Heritability (MCMCglmm): ", round(h2_mcmc, 3), "\n")