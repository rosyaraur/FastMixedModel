# Implementation examples 
# Full Multi-Engine Demonstration Script (sleepstudy)
library(lme4)
library(BGLR)

# Load the classic dataset
data("sleepstudy", package = "lme4")

# 1. Phenotype / Response Vector (y)
y <- sleepstudy$Reaction

# 2. Fixed Effects Matrix (X) - Intercept and Days
X <- model.matrix(~ Days, data = sleepstudy)

# 3. Incidence Matrix for Random Effects (Z) - Subject intercepts
Z <- model.matrix(~ 0 + Subject, data = sleepstudy)

# 4. Covariance/Kinship Matrix (K) for Unrelated Subjects
subjects <- unique(sleepstudy$Subject)
q <- length(subjects)
K <- diag(q)
rownames(K) <- colnames(K) <- paste0("Subject", subjects)
colnames(Z) <- rownames(K)

# 5. Project K to the observation level for BGLR's RKHS model
K_bglr <- as.matrix(Z %*% K %*% t(Z))

# =========================================================================
# RUN NATIVE PACKAGES
# =========================================================================
cat("Running native package fits...\n")

# A. Native lme4
fit_native_lme4 <- lmer(Reaction ~ Days + (1 | Subject), data = sleepstudy, REML = TRUE)

# B. Native BGLR (passing X without global intercept to avoid collinearity)
ETA_bglr <- list(
  list(X = X[, 2, drop = FALSE], model = "FIXED"),
  list(K = K_bglr, model = "RKHS")
)
capture.output({
  fit_native_bglr <- BGLR(y = y, ETA = ETA_bglr, nIter = 3000, burnIn = 1000, verbose = FALSE)
})

# =========================================================================
# RUN UNIFIED ARCHITECTURE PATHS
# =========================================================================
cat("Running Unified Mixed Solver paths...\n")

# A. Path E: Frequentist REML (lme4 style)
fit_path_lme4 <- CombinedMixedSolver(y = y, X = X, Z = Z, A_inv = solve(K), method = "reml_lme4")

# B. Path A: Kernel Gibbs (BGLR style)
fit_path_bglr <- CombinedMixedSolver(y = y, X = X, Z = Z, K = K, method = "kernel_bglr", nIter = 3000, burnIn = 1000)

# C. Path G: Spectral REML (rrBLUP style)
fit_path_rrblup <- CombinedMixedSolver(y = y, X = X, Z = Z, K = K, method = "spectral_rrblup")

# D. Path C: Penalized MAP (blme style)
fit_path_blme <- CombinedMixedSolver(y = y, X = X, Z = Z, A_inv = solve(K), method = "penalized_map_blme")


# =========================================================================
# EXTRACTION AND COMPARISON TABLE
# =========================================================================
info_native_lme4  <- ExtractMixedInfo(fit_native_lme4, engine = "lme4")
info_native_bglr  <- ExtractMixedInfo(fit_native_bglr, engine = "bglr")
info_path_lme4    <- ExtractMixedInfo(fit_path_lme4, engine = "combined")
info_path_bglr    <- ExtractMixedInfo(fit_path_bglr, engine = "combined")
info_path_rrblup  <- ExtractMixedInfo(fit_path_rrblup, engine = "combined")
info_path_blme    <- ExtractMixedInfo(fit_path_blme, engine = "combined")

comparison_table <- data.frame(
  Engine = c("Native: lme4", "Unified: reml_lme4", "Unified: spectral_rrblup", 
             "Unified: penalized_map (blme)", "Native: BGLR", "Unified: kernel_bglr"),
  VarU_Subject = c(info_native_lme4$VarU, info_path_lme4$VarU, info_path_rrblup$VarU, 
                   info_path_blme$VarU, info_native_bglr$VarU, info_path_bglr$VarU),
  VarE_Residual = c(info_native_lme4$VarE, info_path_lme4$VarE, info_path_rrblup$VarE, 
                    info_path_blme$VarE, info_native_bglr$VarE, info_path_bglr$VarE)
)

print(comparison_table)

###########################################################################
# Animal model demonstration script
library(Matrix)
library(BGLR)
library(sommer)

# =========================================================================
# 1. SIMULATE AN ANIMAL MODEL DATASET (Pedigree / Kinship Structure)
# =========================================================================
set.seed(123)

n_animals <- 200

# Generate a synthetic Kinship / Genomic Relationship Matrix (K) for animals
raw_genotypes <- matrix(rnorm(n_animals * 500), nrow = n_animals, ncol = 500)
K <- tcrossprod(raw_genotypes) / 500
# Ensure diagonal has variability and scale
K <- K + diag(n_animals) * 0.05 
rownames(K) <- colnames(K) <- paste0("Animal", 1:n_animals)

# True parameters
varU_true <- 3.0  # Additive genetic variance
varE_true <- 2.0  # Residual variance

# Fixed effects (e.g., Sex or Management Group)
X <- cbind(1, sample(c(0, 1), n_animals, replace = TRUE))
colnames(X) <- c("Intercept", "Sex")

# Random effects incidence matrix Z maps observations 1-to-1 to animals
Z <- diag(n_animals)
colnames(Z) <- rownames(K)

# Simulate breeding values (u) using Cholesky factor of K
u_true <- as.vector(t(chol(K)) %*% rnorm(n_animals, mean = 0, sd = sqrt(varU_true)))
e_true <- rnorm(n_animals, mean = 0, sd = sqrt(varE_true))
# 1. Define the true fixed effects vector explicitly
beta_true <- c(20, 3.5)

# 2. Calculate y correctly using standard matrix multiplication (%*%) and addition
y <- as.vector(X %*% beta_true + u_true + e_true)

# Project K to the observation level for BGLR's RKHS model
K_bglr <- as.matrix(Z %*% K %*% t(Z))

# =========================================================================
# 2. RUN NATIVE ANIMAL MODEL PACKAGES
# =========================================================================
cat("Fitting native animal models...\n")

# A. Native sommer (AI-REML)
df_animal <- data.frame(y = y, Sex = as.factor(X[, 2]), AnimalID = rownames(K))
fit_native_sommer <- mmer(y ~ Sex, random = ~ vs(AnimalID, Gu = K), data = df_animal, verbose = FALSE)

# B. Native BGLR (Gibbs Sampling)
ETA_bglr <- list(
  list(X = X[, 2, drop = FALSE], model = "FIXED"),
  list(K = K_bglr, model = "RKHS")
)
capture.output({
  fit_native_bglr <- BGLR(y = y, ETA = ETA_bglr, nIter = 3000, burnIn = 1000, verbose = FALSE)
})

# =========================================================================
# 3. RUN UNIFIED ARCHITECTURE ANIMAL MODEL PATHS
# =========================================================================
cat("Running Unified Animal Model Solver paths...\n")

# A. Path H: sommer AI-REML Engine via CombinedMixedSolver
fit_path_sommer <- CombinedMixedSolver(y = y, X = X, Z = Z, K = K, method = "direct_aireml_sommer")

# B. Path A: BGLR Kernel Gibbs via CombinedMixedSolver
fit_path_bglr <- CombinedMixedSolver(y = y, X = X, Z = Z, K = K, method = "kernel_bglr", nIter = 3000, burnIn = 1000)

# =========================================================================
# 4. STANDARDIZED EXTRACTION & COMPARISON
# =========================================================================
info_native_sommer <- ExtractMixedInfo(fit_native_sommer, engine = "sommer")
info_native_bglr   <- ExtractMixedInfo(fit_native_bglr, engine = "bglr")
info_path_sommer   <- ExtractMixedInfo(fit_path_sommer, engine = "combined")
info_path_bglr     <- ExtractMixedInfo(fit_path_bglr, engine = "combined")

animal_comparison <- data.frame(
  Engine = c("Native: sommer", "Unified: direct_aireml_sommer", "Native: BGLR", "Unified: kernel_bglr"),
  VarU_Additive = c(info_native_sommer$VarU, info_path_sommer$VarU, info_native_bglr$VarU, info_path_bglr$VarU),
  VarE_Residual = c(info_native_sommer$VarE, info_path_sommer$VarE, info_native_bglr$VarE, info_path_bglr$VarE)
)

print(animal_comparison)

# Check correlation of Estimated Breeding Values (EBVs / u)
ebv_cor <- cor(info_native_sommer$RandomEffects, info_path_sommer$RandomEffects)
cat("\nCorrelation between Native sommer EBVs and Unified Solver EBVs:", ebv_cor, "\n")


################################################################################
# Genomic Prediction Demonstration Script
################################################################################
library(Matrix)
library(BGLR)

# =========================================================================
# 1. SIMULATE GENOMIC DATA & QUANTITATIVE TRAIT (Genomic Selection)
# =========================================================================
set.seed(42)

n_lines <- 300   # Number of breeding lines / individuals
n_snps  <- 1000  # Number of genetic markers (SNPs)

# Simulate marker genotype matrix (coded as -1, 0, 1)
W <- matrix(sample(c(-1, 0, 1), n_lines * n_snps, replace = TRUE, prob = c(0.2, 0.5, 0.3)), 
            nrow = n_lines, ncol = n_snps)
rownames(W) <- paste0("Line", 1:n_lines)

# Compute the Genomic Relationship Matrix (GRM / Kinship K) using VanRaden's method
# 1. Compute allele frequencies per marker (column)
p_freq <- colMeans(W + 1) / 2 

# 2. Construct the centered genomic matrix Z_gen using sweep
P_mat <- matrix(2 * (p_freq - 0.5), nrow = nrow(W), ncol = ncol(W), byrow = TRUE)
Z_gen <- W - P_mat

# 3. Calculate VanRaden's Genomic Relationship Matrix (K)
denom <- sum(2 * p_freq * (1 - p_freq))
K <- tcrossprod(Z_gen) / denom
rownames(K) <- colnames(K) <- rownames(W)

# Simulate Phenotypes with a known heritability (h2 = 0.5)
h2_true <- 0.5
qtl_idx <- sample(1:n_snps, size = 20) # 20 causal QTLs
marker_effects <- rnorm(n_snps, mean = 0, sd = 1)
marker_effects[-qtl_idx] <- 0 # Only 20 markers have true effects

genetic_values <- as.vector(W %*% marker_effects)
genetic_values <- scale(genetic_values) * sqrt(h2_true)
residual_val   <- rnorm(n_lines, mean = 0, sd = sqrt(1 - h2_true))
y_full         <- 50 + genetic_values + residual_val

# Fixed effects matrix (e.g., intercept and an environmental covariate)
X <- cbind(1, rnorm(n_lines))
Z_id <- diag(n_lines) # Incidence matrix mapping lines 1-to-1

# =========================================================================
# 2. CREATE A PREDICTION SCENARIO (Masking Phenotypes)
# =========================================================================
# Mask 30% of the lines to simulate unphenotyped selection candidates
test_idx <- sample(1:n_lines, size = 90)
y_train <- y_full
y_train[test_idx] <- NA # Hidden phenotypes for validation
train_idx <- which(!is.na(y_train))

# =========================================================================
# 3. FIT THE MODEL VIA UNIFIED ARCHITECTURE
# =========================================================================
cat("Running Genomic Prediction via Unified Solver...\n")

# A. Frequentist G-BLUP via Spectral Decomposition (rrBLUP style)
# Note: For prediction, we can pass y_train directly; our solver handles regression.
fit_gp_reml <- CombinedMixedSolver(
  y = y_train[train_idx], 
  X = X[train_idx, , drop = FALSE], 
  Z = Z_id[train_idx, , drop = FALSE], 
  K = K, 
  method = "spectral_rrblup"
)
# B. Bayesian G-BLUP via Kernel Gibbs Sampling (BGLR style)
K_bglr <- as.matrix(Z_id %*% K %*% t(Z_id))
# Ensure X remains a proper matrix with drop = FALSE
X_train <- as.matrix(X[train_idx, , drop = FALSE])
Z_train <- as.matrix(Z_id[train_idx, , drop = FALSE])

# Run kernel_bglr with clean subset dimensions
fit_gp_gibbs <- CombinedMixedSolver(
  y = y_train[train_idx], 
  X = X_train, 
  Z = Z_train, 
  K = K, 
  method = "kernel_bglr", 
  nIter = 3000, 
  burnIn = 1000
)

# =========================================================================
# 4. PREDICT AND VALIDATE ON MASKED LINES
# =========================================================================
# Predicted Genomic Estimated Breeding Values (GEBVs) = X*beta + u
pred_reml <- as.vector(X %*% fit_gp_reml$post_beta + fit_gp_reml$post_u)
pred_gibbs <- as.vector(X %*% fit_gp_gibbs$post_beta + fit_gp_gibbs$post_u)

# Calculate Predictive Ability (Correlation between true genetic value and prediction)
acc_reml  <- cor(pred_reml[test_idx], genetic_values[test_idx])
acc_gibbs <- cor(pred_gibbs[test_idx], genetic_values[test_idx])

cat("\n--- Genomic Prediction Results ---\n")
cat("Predictive Ability (Spectral REML):", round(acc_reml, 4), "\n")
cat("Predictive Ability (Kernel Gibbs): ", round(acc_gibbs, 4), "\n")
cat("Estimated Variance Ratio (REML):   ", round(fit_gp_reml$post_varU / (fit_gp_reml$post_varU + fit_gp_reml$post_varE), 4), "\n")
cat("Estimated Variance Ratio (Gibbs):  ", round(fit_gp_gibbs$post_varU / (fit_gp_gibbs$post_varU + fit_gp_gibbs$post_varE), 4), "\n")

#########################################################
# MCMCglmm Pedigree Animal Model Demonstration Script
#########################################################
library(Matrix)
library(MCMCglmm)

# =========================================================================
# 1. LOAD DATA & BUILD PEDIGREE MATRICES (MCMCglmm Standard)
# =========================================================================
# Using the built-in PlodiaPO dataset from MCMCglmm which contains family/pedigree structure
data(PlodiaPO)

# Define response (PO: Phenoloxidase activity) and fixed effect (plates/trt)
y <- PlodiaPO$PO
X <- model.matrix(~ 1, data = PlodiaPO) # Intercept only model

# Define the random grouping factor (Full-sib family acting as our animal/group ID)
group_factor <- PlodiaPO$FSfamily
Z <- model.matrix(~ 0 + group_factor, data = PlodiaPO)

# For a true pedigree demonstration, we can generate a numerator relationship matrix (A) 
# and its sparse inverse (A_inv) using MCMCglmm's inverseA utility.
# (Here we mock a simple pedigree structure based on the levels of FSfamily)
library(MasterBayes)

# 1. Clean up and structure the dummy pedigree correctly for MCMCglmm
dummy_pedigree <- data.frame(
  id = 1:n_groups,
  dam = c(NA, NA, sample(1:2, n_groups - 2, replace = TRUE)),
  sire = c(NA, NA, sample(1:2, n_groups - 2, replace = TRUE))
)

# 2. Use MasterBayes tools to insert missing parent records and order chronologically
ped_matrix <- as.matrix(dummy_pedigree)
ped_matrix <- insertPed(ped_matrix)
ped_matrix <- orderPed(ped_matrix)

# 3. Convert back to data frame and compute the inverse cleanly
ordered_pedigree <- as.data.frame(ped_matrix)
inv_obj <- inverseA(ordered_pedigree, nodes = "ALL", scale = TRUE)
A_inv <- inv_obj$Ainv

# Align dimensions of A_inv with Z
rownames(A_inv) <- colnames(A_inv) <- levels(group_factor)

# =========================================================================
# 2. RUN NATIVE MCMCglmm PACKAGE
# =========================================================================
cat("Fitting native MCMCglmm model...\n")

# Calculate phenotypic variance of the trait to set an informed prior scale
V_pheno <- var(PlodiaPO$PO, na.rm = TRUE)

prior_mcmc_informed <- list(
  R = list(V = V_pheno * 0.5, nu = 1),
  G = list(G1 = list(V = V_pheno * 0.5, nu = 1))
)

# 1. Bind the grouping factor directly into the PlodiaPO dataset
PlodiaPO$group_factor <- PlodiaPO$FSfamily

fit_native_mcmc_informed <- MCMCglmm(
  PO ~ 1, 
  random = ~ group_factor, 
  ginverse = list(group_factor = A_inv), 
  data = PlodiaPO, 
  prior = prior_mcmc_informed, 
  nitt = 30000, 
  burnin = 10000, 
  thin = 10, 
  verbose = FALSE
)


# =========================================================================
# 3. RUN UNIFIED SOLVER (Path B: block_mcmcglmm)
# =========================================================================
cat("Running Unified Architecture Sparse MME Block Gibbs Solver...\n")

# 1. Ensure complete cases and clear scaling for Path B
complete_idx <- which(!is.na(PlodiaPO$PO))
y_clean <- PlodiaPO$PO[complete_idx]
X_clean <- model.matrix(~ 1, data = PlodiaPO[complete_idx, ])

# Reconstruct Z incidence matrix dynamically matching the levels of group_factor
group_fct <- factor(PlodiaPO$FSfamily[complete_idx])
Z_clean <- model.matrix(~ group_fct - 1)

# 2. Rerun Path B with clean dimensions
fit_path_mcmc <- CombinedMixedSolver(
  y = y_clean, 
  X = X_clean, 
  Z = Z_clean, 
  A_inv = A_inv, 
  method = "block_mcmcglmm", 
  nIter = 30000, 
  burnIn = 10000
)

# =========================================================================
# 4. COMPARISON MATRIX
# =========================================================================
# Extract posterior means from MCMCglmm chains
native_varU <- mean(fit_native_mcmc$VCV[, "group_factor"])
native_varE <- mean(fit_native_mcmc$VCV[, "units"])

mcmc_comparison <- data.frame(
  Engine = c("Native: MCMCglmm", "Unified: block_mcmcglmm (Path B)"),
  VarU_Genetic = c(native_varU, fit_path_mcmc$post_varU),
  VarE_Residual = c(native_varE, fit_path_mcmc$post_varE)
)

print(mcmc_comparison)

##############################################################################
# Implementation Demonstration: Complex Random Slopes & Intercepts
library(lme4)
library(blme)
library(brms)

# Load the classic sleep dataset
data("sleepstudy", package = "lme4")

# =========================================================================
# 1. FREQUENTIST OPTIMIZATION (lme4) - Complex Random Slopes & Intercepts
# =========================================================================
cat("Fitting complex model with lme4 (REML)...\n")
fit_complex_lme4 <- lmer(Reaction ~ Days + (Days | Subject), data = sleepstudy, REML = TRUE)

# View the subject-specific intercepts and slopes
head(coef(fit_complex_lme4)$Subject)


# =========================================================================
# 2. PENALIZED MAP OPTIMIZATION (blme) - Constraining Boundary Estimates
# =========================================================================
# Complex models often result in singular fits in lme4. 
# blme injects Wishart/Gamma priors on the 2x2 covariance matrix to stabilize it.
cat("Fitting complex model with blme (MAP with priors)...\n")
fit_complex_blme <- blmer(Reaction ~ Days + (Days | Subject), data = sleepstudy,
                          cov.prior = "wishart") # Applies default LKJ/Wishart penalty

# Compare variance components between lme4 and blme
print(VarCorr(fit_complex_lme4))
print(VarCorr(fit_complex_blme))


# =========================================================================
# 3. FULL BAYESIAN HMC (brms) - Exploring Posterior Funnel Geometries
# =========================================================================
# brms compiles this complex hierarchical structure into C++ via Stan
cat("Fitting complex model with brms (HMC Sampling)...\n")
fit_complex_brms <- brm(
  formula = Reaction ~ Days + (Days | Subject),
  data = sleepstudy,
  family = gaussian(),
  chains = 2, iter = 2000, warmup = 1000,
  refresh = 0
)

# Print full posterior summary including correlation between slope and intercept
print(summary(fit_complex_brms))

################################################################################
# comparision of sas mixed and asreml proxy functions 
# =========================================================================
# 1. SIMULATE DATA WITH KNOWN VARIANCE COMPONENTS
# =========================================================================
library(Matrix)
library(lme4)

# =========================================================================
# 1. SIMULATE BENCHMARK DATA
# =========================================================================
set.seed(42)
n <- 400
p <- 2  
q <- 50 

true_varU <- 15.0
true_varE <- 4.0

X <- cbind(1, rnorm(n))
beta_true <- c(10, 2.5)

group_ids <- sample(1:q, n, replace = TRUE)
Z <- sparse.model.matrix(~ factor(group_ids) - 1)

u_true <- rnorm(q, mean = 0, sd = sqrt(true_varU))
e_true <- rnorm(n, mean = 0, sd = sqrt(true_varE))

y <- as.vector(X %*% beta_true + Z %*% u_true + e_true)
A_inv <- Diagonal(q) 
K_matrix <- as.matrix(solve(A_inv))

# =========================================================================
# 2. RUN PATH COMPARISONS VIA CombinedMixedSolver
# =========================================================================
fit_asreml_sparse <- CombinedMixedSolver(y = y, X = X, Z = Z, A_inv = A_inv, method = "sparse_aireml_asreml")
fit_sas_nr        <- CombinedMixedSolver(y = y, X = X, Z = Z, K = K_matrix, method = "nr_sasmixed")
fit_lme4_reml     <- CombinedMixedSolver(y = y, X = X, Z = Z, A_inv = A_inv, method = "reml_lme4")
fit_sommer_ai     <- CombinedMixedSolver(y = y, X = X, Z = Z, K = K_matrix, method = "direct_aireml_sommer")

# Native lme4 benchmark for cross-validation
native_lmer <- lmer(y ~ X[,2] + (1 | group_ids))
native_vc   <- VarCorr(native_lmer)

# =========================================================================
# 3. COMPILE AND PRINT RESULTS TABLE
# =========================================================================
comparison_table <- data.frame(
  Engine = c(
    "True Parameters",
    "Path I: sparse_aireml_asreml (ASReml Essence)",
    "Path J: nr_sasmixed (SAS PROC MIXED Essence)",
    "Path E: reml_lme4 (Profiled REML)",
    "Path H: direct_aireml_sommer (Sommer AI-REML)",
    "Native lme4 (Benchmark)"
  ),
  VarU_Genetic = c(
    true_varU,
    fit_asreml_sparse$post_varU,
    fit_sas_nr$post_varU,
    fit_lme4_reml$post_varU,
    fit_sommer_ai$post_varU,
    as.numeric(native_vc$group_ids)
  ),
  VarE_Residual = c(
    true_varE,
    fit_asreml_sparse$post_varE,
    fit_sas_nr$post_varE,
    fit_lme4_reml$post_varE,
    fit_sommer_ai$post_varE,
    attr(native_vc, "sc")^2
  )
)

print(comparison_table)

library(Matrix)
library(lme4)

# =========================================================================
# 1. SIMULATE BENCHMARK DATA
# =========================================================================
set.seed(42)
n <- 400
p <- 2  
q <- 50 

true_varU <- 15.0
true_varE <- 4.0

X <- cbind(1, rnorm(n))
beta_true <- c(10, 2.5)

group_ids <- sample(1:q, n, replace = TRUE)
Z <- sparse.model.matrix(~ factor(group_ids) - 1)

u_true <- rnorm(q, mean = 0, sd = sqrt(true_varU))
e_true <- rnorm(n, mean = 0, sd = sqrt(true_varE))

y <- as.vector(X %*% beta_true + Z %*% u_true + e_true)
A_inv <- Diagonal(q) 
K_matrix <- as.matrix(solve(A_inv))

# =========================================================================
# 2. RUN PATH COMPARISONS VIA CombinedMixedSolver
# =========================================================================
fit_asreml_sparse <- CombinedMixedSolver(y = y, X = X, Z = Z, A_inv = A_inv, method = "sparse_aireml_asreml")
fit_sas_nr        <- CombinedMixedSolver(y = y, X = X, Z = Z, K = K_matrix, method = "nr_sasmixed")
fit_lme4_reml     <- CombinedMixedSolver(y = y, X = X, Z = Z, A_inv = A_inv, method = "reml_lme4")
fit_sommer_ai     <- CombinedMixedSolver(y = y, X = X, Z = Z, K = K_matrix, method = "direct_aireml_sommer")

# Native lme4 benchmark for cross-validation
native_lmer <- lmer(y ~ X[,2] + (1 | group_ids))
native_vc   <- VarCorr(native_lmer)
native_u    <- ranef(native_lmer)$group_ids[, 1]

# =========================================================================
# 3. COMPILE AND PRINT VARIANCE COMPONENTS TABLE
# =========================================================================
comparison_table <- data.frame(
  Engine = c(
    "True Parameters",
    "Path I: sparse_aireml_asreml (ASReml Essence)",
    "Path J: nr_sasmixed (SAS PROC MIXED Essence)",
    "Path E: reml_lme4 (Profiled REML)",
    "Path H: direct_aireml_sommer (Sommer AI-REML)",
    "Native lme4 (Benchmark)"
  ),
  VarU_Genetic = c(
    true_varU,
    fit_asreml_sparse$post_varU,
    fit_sas_nr$post_varU,
    fit_lme4_reml$post_varU,
    fit_sommer_ai$post_varU,
    as.numeric(native_vc$group_ids)
  ),
  VarE_Residual = c(
    true_varE,
    fit_asreml_sparse$post_varE,
    fit_sas_nr$post_varE,
    fit_lme4_reml$post_varE,
    fit_sommer_ai$post_varE,
    attr(native_vc, "sc")^2
  )
)

cat("\n--- Variance Component Estimates ---\n")
print(comparison_table)

# =========================================================================
# 4. EXTRACT AND COMPARE EFFECTS (BLUPs / Random Effects)
# =========================================================================
effects_matrix <- data.frame(
  True_u        = u_true,
  ASReml_Sparse = fit_asreml_sparse$post_u,
  SAS_NR        = fit_sas_nr$post_u,
  REML_lme4     = fit_lme4_reml$post_u,
  Sommer_AI     = fit_sommer_ai$post_u,
  Native_lme4   = native_u
)

# Calculate Pearson correlation matrix across predicted and true effects
effect_cor_matrix <- cor(effects_matrix)

cat("\n--- Random Effects Correlation Matrix ---\n")
print(round(effect_cor_matrix, 4))