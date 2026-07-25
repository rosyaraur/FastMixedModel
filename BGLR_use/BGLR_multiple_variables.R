# ==========================================
# 1. Multiple Predictors WITHOUT an A/G Matrix
# ==========================================

# When you do not have a pedigree or genomic relationship matrix, you are essentially fitting a standard Multiple Linear Mixed Model.
# We will use a continuous variable (Age), a categorical variable (Treatment), and a random grouping variable (Herd).

library(BGLR)
library(lme4)

# ==========================================
# 1. SIMULATE DATA (Multiple Predictors)
# ==========================================
set.seed(123)
n_obs <- 300

dat <- data.frame(
  id = 1:n_obs,
  age = rnorm(n_obs, mean = 50, sd = 10),               # Continuous X1
  treatment = as.factor(sample(c("A", "B", "C"), n_obs, replace = TRUE)), # Categorical X2
  herd = as.factor(rep(1:15, each = 20))                # Random Group
)

# True effects: Base=100, Age=0.5, TrtB=5, TrtC=-3, Herd Variance=4^2
herd_eff <- rnorm(15, 0, 4)
dat$y <- 100 + 
  (0.5 * dat$age) + 
  ifelse(dat$treatment == "B", 5, 0) + 
  ifelse(dat$treatment == "C", -3, 0) + 
  herd_eff[dat$herd] + 
  rnorm(n_obs, 0, 2) # Residual error


# ==========================================
# 2. BGLR: MULTIPLE PREDICTORS
# ==========================================
# Combine all fixed predictors into a single matrix. 
# MUST drop the first column [-1] because BGLR fits an intercept automatically.
X_multi <- model.matrix(~ age + treatment, data = dat)[, -1, drop = FALSE]

# Random effect matrix
Z_herd <- model.matrix(~ herd - 1, data = dat)

ETA_multi <- list(
  list(X = X_multi, model = "FIXED"),
  list(X = Z_herd,  model = "BRR") # BRR handles the random intercepts
)

fit_bglr <- BGLR(y = dat$y, 
                 ETA = ETA_multi, 
                 nIter = 5000, burnIn = 1000, verbose = FALSE)


# ==========================================
# 3. lme4: MULTIPLE PREDICTORS
# ==========================================
fit_lme4 <- lmer(y ~ age + treatment + (1 | herd), data = dat)


# ==========================================
# 4. COMPARISON
# ==========================================
cat("--- FIXED EFFECTS ---\n")
cat("lme4:\n")
print(fixef(fit_lme4))
cat("\nBGLR:\n")
# BGLR stores the overall intercept in 'mu', and the other fixed effects in the ETA list
print(c(Intercept = fit_bglr$mu, fit_bglr$ETA[[1]]$b))

cat("\n--- HERD VARIANCE ---\n")
vc_lme4 <- as.data.frame(VarCorr(fit_lme4))
cat("lme4:", vc_lme4$vcov[vc_lme4$grp == "herd"], "\n")
cat("BGLR:", fit_bglr$ETA[[2]]$varB, "\n")

# ==========================================
# 2. Multiple Predictors WITH an A/G Matrix
# ==========================================
# When you introduce a Numerator Relationship Matrix ($A$) or a Genomic Relationship Matrix ($G$), 
# you are fitting a classic Animal Model with Fixed Covariates.The comparable package for this task
# is sommer, as it natively handles custom covariance matrices alongside standard formula inputs. We will
# use the built-in mice dataset and simulate two additional fixed covariates (Age and Diet) to demonstrate.
library(BGLR)
library(sommer)

# ==========================================
# 1. SETUP DATA AND A-MATRIX
# ==========================================
data(mice, package = "BGLR")
y_bmi <- mice.pheno$Obesity.BMI
A_mat <- mice.A

# Simulate multiple predictors for the mice
set.seed(42)
n_mice <- length(y_bmi)
df_mice <- data.frame(
  id = rownames(A_mat),
  age_weeks = runif(n_mice, 8, 20),
  diet = as.factor(sample(c("Control", "HighFat"), n_mice, replace = TRUE)),
  y = y_bmi
)


# ==========================================
# 2. BGLR: COVARIATES + A-MATRIX
# ==========================================
# Combine covariates into one fixed matrix (dropping intercept)
X_covariates <- model.matrix(~ age_weeks + diet, data = df_mice)[, -1, drop = FALSE]

# ETA list: Stack the fixed covariates and the RKHS relationship matrix
ETA_animal_multi <- list(
  list(X = X_covariates, model = "FIXED"),
  list(K = A_mat,        model = "RKHS") 
)

fit_bglr_animal <- BGLR(y = df_mice$y, 
                        ETA = ETA_animal_multi, 
                        nIter = 5000, burnIn = 1000, verbose = FALSE)


# ==========================================
# 3. SOMMER: COVARIATES + A-MATRIX
# ==========================================
# sommer uses a standard formula for fixed effects, and vsr() for the A-matrix
fit_sommer_animal <- mmer(fixed = y ~ age_weeks + diet, 
                          random = ~ vsr(id, Gu = A_mat), 
                          rcov = ~ vsr(units),
                          data = df_mice, 
                          verbose = FALSE)


# ==========================================
# 4. COMPARISON
# ==========================================
cat("--- FIXED EFFECTS ---\n")
cat("sommer:\n")
print(fit_sommer_animal$Beta)
cat("\nBGLR:\n")
print(c(Intercept = fit_bglr_animal$mu, fit_bglr_animal$ETA[[1]]$b))

cat("\n--- ADDITIVE GENETIC VARIANCE (A-Matrix) ---\n")
vc_sommer <- summary(fit_sommer_animal)$varcomp
cat("sommer:", vc_sommer[1, "VarComp"], "\n")
cat("BGLR:  ", fit_bglr_animal$ETA[[2]]$varU, "\n")

# ==========================================
# GBLUP 
# ==========================================
library(BGLR)

# ==========================================
# 1. SIMULATE DATA (Major Genes + Noise)
# ==========================================
set.seed(42)
n <- 300   # 300 individuals
p <- 1000  # 1000 genetic markers

# Simulate a genotype matrix (values 0, 1, or 2)
X_markers <- matrix(rbinom(n * p, 2, 0.5), nrow = n, ncol = p)

# Simulate True Marker Effects (Only the first 5 markers matter)
true_effects <- rep(0, p)
true_effects[1:5] <- c(10, -8, 12, -10, 9) # The "Major Genes"

# Create the true phenotype with some residual noise
y_true <- X_markers %*% true_effects
y <- y_true + rnorm(n, mean = 0, sd = 5)


# ==========================================
# 2. FIT THE FREQUENTIST EQUIVALENT (BRR / GBLUP)
# ==========================================
# BRR assumes all markers have effects drawn from the same normal distribution
ETA_ridge <- list(list(X = X_markers, model = "BRR"))

fit_ridge <- BGLR(y = y, 
                  ETA = ETA_ridge, 
                  nIter = 5000, burnIn = 1000, 
                  verbose = FALSE)


# ==========================================
# 3. FIT THE BAYESIAN ADVANTAGE (BayesB)
# ==========================================
# BayesB assumes most markers have zero effect (prob = probIn), 
# allowing the real ones to stay large.
ETA_bayesB <- list(list(X = X_markers, model = "BayesB", probIn = 0.05))

fit_bayesB <- BGLR(y = y, 
                   ETA = ETA_bayesB, 
                   nIter = 5000, burnIn = 1000, 
                   verbose = FALSE)


# ==========================================
# 4. COMPARE PREDICTIVE ACCURACY
# ==========================================
# Extract the estimated marker effects
eff_ridge <- fit_ridge$ETA[[1]]$b
eff_bayesB <- fit_bayesB$ETA[[1]]$b

# Predict phenotypes based on the models
y_hat_ridge <- X_markers %*% eff_ridge
y_hat_bayesB <- X_markers %*% eff_bayesB

# Calculate correlation between predicted and true genetic merit
acc_ridge <- cor(y_hat_ridge, y_true)
acc_bayesB <- cor(y_hat_bayesB, y_true)

cat("Accuracy of Ridge (Frequentist Equivalent):", round(acc_ridge, 3), "\n")
cat("Accuracy of BayesB (Bayesian Prior):       ", round(acc_bayesB, 3), "\n\n")

# Look at how the models handled the first major gene (True effect = 10)
cat("True Effect of Marker 1:  10.00\n")
cat("Estimated by Ridge:      ", round(eff_ridge[1], 2), "\n")
cat("Estimated by BayesB:     ", round(eff_bayesB[1], 2), "\n")

# The absolute greatest advantage of BGLR—and the primary reason it was created—is its ability to 
# handle the Curse of Dimensionality ($p \gg n$) using differential shrinkage priors.In a frequentist 
# framework, if you have more predictors than observations (e.g., $n = 500$ animals, but $p = 50,000$ genetic markers),
# a standard linear model simply crashes because the design matrix is rank-deficient.To bypass this, frequentist mixed 
# models (like sommer) use GBLUP (Genomic Best Linear Unbiased Prediction). 

# Mathematically, GBLUP is identical to Ridge Regression. It gets around the $p \gg n$ problem by applying a blanket penalty,
# shrinking all 50,000 markers toward zero equally.The Problem with Frequentist GBLUPBiology rarely works like Ridge Regression. 
# Often, a trait is controlled by one or two major genes (large effects) and thousands of minor background genes (small effects). 
# Because the frequentist approach shrinks everything equally, it severely over-shrinks the major genes, reducing your predictive accuracy.
#The BGLR Advantage: Spike-and-Slab PriorsBGLR allows you to apply complex Bayesian priors, such as BayesB. BayesB uses a "spike-and-slab"
# prior:The Spike: A high probability ($\pi$) that a marker has exactly zero effect.The Slab: A t-distribution for the few markers that
# do have an effect, allowing them to remain large without being brutally shrunk.Here is an example demonstrating exactly how using 
# BayesB outperforms the frequentist Ridge approach when major genes are present.

library(BGLR)

# ==========================================
# 1. SIMULATE DATA (Major Genes + Noise)
# ==========================================
set.seed(82)
n <- 300   # 300 individuals
p <- 1000  # 1000 genetic markers

# Simulate a genotype matrix (values 0, 1, or 2)
X_markers <- matrix(rbinom(n * p, 2, 0.5), nrow = n, ncol = p)

# Simulate True Marker Effects (Only the first 5 markers matter)
true_effects <- rep(0, p)
true_effects[1:5] <- c(10, -8, 12, -10, 9) # The "Major Genes"

# Create the true phenotype with some residual noise
y_true <- X_markers %*% true_effects
y <- y_true + rnorm(n, mean = 0, sd = 5)


# ==========================================
# 2. FIT THE FREQUENTIST EQUIVALENT (BRR / GBLUP)
# ==========================================
# BRR assumes all markers have effects drawn from the same normal distribution
ETA_ridge <- list(list(X = X_markers, model = "BRR"))

fit_ridge <- BGLR(y = y, 
                  ETA = ETA_ridge, 
                  nIter = 5000, burnIn = 1000, 
                  verbose = FALSE)


# ==========================================
# 3. FIT THE BAYESIAN ADVANTAGE (BayesB)
# ==========================================
# BayesB assumes most markers have zero effect (prob = probIn), 
# allowing the real ones to stay large.
ETA_bayesB <- list(list(X = X_markers, model = "BayesB", probIn = 0.05))

fit_bayesB <- BGLR(y = y, 
                   ETA = ETA_bayesB, 
                   nIter = 5000, burnIn = 1000, 
                   verbose = FALSE)


# ==========================================
# 4. COMPARE PREDICTIVE ACCURACY
# ==========================================
# Extract the estimated marker effects
eff_ridge <- fit_ridge$ETA[[1]]$b
eff_bayesB <- fit_bayesB$ETA[[1]]$b

# Predict phenotypes based on the models
y_hat_ridge <- X_markers %*% eff_ridge
y_hat_bayesB <- X_markers %*% eff_bayesB

# Calculate correlation between predicted and true genetic merit
acc_ridge <- cor(y_hat_ridge, y_true)
acc_bayesB <- cor(y_hat_bayesB, y_true)

cat("Accuracy of Ridge (Frequentist Equivalent):", round(acc_ridge, 3), "\n")
cat("Accuracy of BayesB (Bayesian Prior):       ", round(acc_bayesB, 3), "\n\n")

# Look at how the models handled the first major gene (True effect = 10)
cat("True Effect of Marker 1:  10.00\n")
cat("Estimated by Ridge:      ", round(eff_ridge[1], 2), "\n")
cat("Estimated by BayesB:     ", round(eff_bayesB[1], 2), "\n")

# ==========================================
# Predicting flowering time in soybean 
# ==========================================
# We will simulate a dataset with 100 plant lines grown across 4 different environments. We will measure temperature 
# and photoperiod (day length) for each environment, and use 500 genetic markers to represent the plant genomes.

# Load required packages
# install.packages(c("BGLR", "sommer"))
library(BGLR)
library(sommer)

set.seed(123)
n_lines <- 100
n_envs <- 4
n_obs <- n_lines * n_envs

# ---------------------------------------------------------
# A. Simulate Genotypes (100 lines, 500 markers)
# ---------------------------------------------------------
M <- matrix(rbinom(n_lines * 500, 2, 0.5), nrow = n_lines)
rownames(M) <- paste0("Line_", 1:n_lines)

# Simulate 5 Major Genes for flowering time (e.g., Ppd and Vrn genes)
true_marker_eff <- rep(0, 500)
true_marker_eff[c(10, 55, 120, 300, 450)] <- c(-5, 4, -6, 7, -3) 

# Calculate true genetic merit
genotype_value <- M %*% true_marker_eff

# ---------------------------------------------------------
# B. Simulate Environmental Data (4 Environments)
# ---------------------------------------------------------
env_data <- data.frame(
  Env = paste0("E", 1:n_envs),
  Temp_C = c(15, 18, 22, 26),        # Average Temperature
  Photoperiod_hr = c(10, 12, 14, 16) # Day length
)

# ---------------------------------------------------------
# C. Build the Full Phenotypic Dataset
# ---------------------------------------------------------
df <- expand.grid(Line = rownames(M), Env = env_data$Env)
df <- merge(df, env_data, by = "Env")

# Map genetic value to the expanded dataframe
df$G_Value <- genotype_value[match(df$Line, rownames(M))]

# Environmental Effect: Warmer temps and longer days accelerate flowering (reduce days)
df$E_Value <- (-0.8 * df$Temp_C) + (-1.5 * df$Photoperiod_hr)

# Calculate final phenotype: Base (120 days) + Genes + Environment + Noise
df$Flowering_Days <- 120 + df$G_Value + df$E_Value + rnorm(n_obs, mean = 0, sd = 3)

# ==========================================
# 2. BGLR: Bayesian Prior + Environmental Covariates
# ==========================================
# 1. Prepare Environmental Fixed Effects Matrix
# We drop the intercept because BGLR fits a global mu automatically
X_env <- model.matrix(~ Temp_C + Photoperiod_hr, data = df)[, -1, drop = FALSE]

# 2. Expand Marker Matrix to match observations
# This matches the 100 lines to the 400 total observations
X_markers_expanded <- M[match(df$Line, rownames(M)), ]

# 3. Define the ETA predictor list
ETA_bglr <- list(
  list(X = X_env, model = "FIXED"),
  list(X = X_markers_expanded, model = "BayesB", probIn = 0.05) # Spike-and-slab for major genes
)

# 4. Run the model
fit_bglr <- BGLR(y = df$Flowering_Days, 
                 ETA = ETA_bglr, 
                 nIter = 5000, burnIn = 1000, 
                 verbose = FALSE)

# Extract BGLR Predictions
y_hat_bglr <- fit_bglr$yHat

# ==========================================
# 3. sommer: The Frequentist GBLUP Alternative
# ==========================================
# 1. Compute the Genomic Relationship Matrix (G-matrix)
# A simple normalization of the marker matrix
p_marker <- colMeans(M) / 2
M_centered <- scale(M, center = TRUE, scale = FALSE)
G_matrix <- (M_centered %*% t(M_centered)) / (2 * sum(p_marker * (1 - p_marker)))

# Ensure the dataframe uses factor type for the Line ID to match the G matrix
df$Line <- as.factor(df$Line)

# 2. Run the sommer model (GBLUP)
fit_sommer <- mmer(fixed = Flowering_Days ~ Temp_C + Photoperiod_hr, 
                   random = ~ vsr(Line, Gu = G_matrix), 
                   rcov = ~ vsr(units), 
                   data = df, 
                   verbose = FALSE)

# Extract sommer Predictions
# y_hat = Fixed effects + Random genetic effects
fixed_preds <- as.vector(model.matrix(~ Temp_C + Photoperiod_hr, data = df) %*% fit_sommer$Beta$Estimate)
random_preds <- fit_sommer$U$`u:Line`$Flowering_Days[match(df$Line, rownames(G_matrix))]
y_hat_sommer <- fixed_preds + random_preds

# ==========================================
# 4. Comparing the Results
# ==========================================
cat("--- ENVIRONMENTAL EFFECTS ---\n")
cat("True Temp Effect: -0.80 | True Photo Effect: -1.50\n\n")

cat("BGLR Estimates:\n")
cat("Temp:", round(fit_bglr$ETA[[1]]$b[1], 3), "| Photo:", round(fit_bglr$ETA[[1]]$b[2], 3), "\n\n")

cat("sommer Estimates:\n")
cat("Temp:", round(fit_sommer$Beta$Estimate[2], 3), "| Photo:", round(fit_sommer$Beta$Estimate[3], 3), "\n\n")

cat("--- PREDICTION ACCURACY (Correlation with True y) ---\n")
cat("BGLR (BayesB):", round(cor(y_hat_bglr, df$Flowering_Days), 3), "\n")
cat("sommer (GBLUP):", round(cor(y_hat_sommer, df$Flowering_Days), 3), "\n")