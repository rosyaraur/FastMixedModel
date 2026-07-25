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
prior_mcmc <- list(G = list(G1 = list(V = 1, nu = 0.002)), 
                   R = list(V = 1, nu = 0.002))

library(Matrix) # Load the Matrix package to handle S4 sparse matrices

# 1. Invert the A matrix
A_inv_base <- solve(A)

# 2. Convert it to a sparse matrix (dgCMatrix)
A_inv_sparse <- as(A_inv_base, "dgCMatrix")

# 3. Ensure row and column names are intact
rownames(A_inv_sparse) <- rownames(A)
colnames(A_inv_sparse) <- colnames(A)

# Fit model (Make sure to use A_inv_sparse here!)
fit_mcmc <- MCMCglmm(fixed = y ~ 1, 
                     random = ~ id, 
                     ginverse = list(id = A_inv_sparse), 
                     data = pheno_df, 
                     prior = prior_mcmc, 
                     nitt = 5000, 
                     burnin = 1000, 
                     verbose = FALSE)

# Extract Heritability (using the posterior means of the variance components)
var_g_mcmc <- mean(fit_mcmc$VCV[, "id"])
var_e_mcmc <- mean(fit_mcmc$VCV[, "units"])
h2_mcmc <- var_g_mcmc / (var_g_mcmc + var_e_mcmc)

print(h2_mcmc)


# ==========================================
# 5. COMPARE HERITABILITY ESTIMATES
# ==========================================
cat("Heritability (BGLR):     ", round(h2_bglr, 3), "\n")
cat("Heritability (sommer):   ", round(h2_sommer, 3), "\n")
cat("Heritability (MCMCglmm): ", round(h2_mcmc, 3), "\n")

# ==========================================
## Example usecase 
# ==========================================

# 1. Healthcare: Random Intercepts & Slopes (sleepstudy)

library(lme4)
library(BGLR)
data("sleepstudy")

# ==========================================
# 1. BGLR SETUP (Random Intercept + Slope)
# ==========================================
# Fixed effect matrix (Drop the intercept, BGLR adds it)
X_fixed <- model.matrix(~ Days, data = sleepstudy)[, -1, drop = FALSE]

# Random Intercept matrix (Z_int)
Z_int <- model.matrix(~ Subject - 1, data = sleepstudy)

# Random Slope matrix (Z_slope): Intercept matrix multiplied by Days
Z_slope <- Z_int * sleepstudy$Days

ETA_sleep <- list(
  list(X = X_fixed, model = "FIXED"),
  list(X = Z_int,   model = "BRR"),  # Subject Intercepts
  list(X = Z_slope, model = "BRR")   # Subject Slopes
)

fit_bglr_sleep <- BGLR(y = sleepstudy$Reaction, 
                       ETA = ETA_sleep, 
                       nIter = 5000, burnIn = 1000, verbose = FALSE)

# ==========================================
# 2. lme4 SETUP
# ==========================================
fit_lme4_sleep <- lmer(Reaction ~ Days + (Days | Subject), data = sleepstudy)

# ==========================================
# 3. COMPARISON
# ==========================================
cat("Fixed Effect (Days):\n lme4:", fixef(fit_lme4_sleep)["Days"], 
    "\n BGLR:", fit_bglr_sleep$ETA[[1]]$b, "\n")

cat("\nVariance Components:\n lme4 Intercept:", VarCorr(fit_lme4_sleep)$Subject[1,1],
    "\n BGLR Intercept:", fit_bglr_sleep$ETA[[2]]$varB,
    "\n\n lme4 Slope:", VarCorr(fit_lme4_sleep)$Subject[2,2],
    "\n BGLR Slope:", fit_bglr_sleep$ETA[[3]]$varB, "\n")

# GLR assumes the random intercept and random slope are independent (covariance = 0). lme4 estimates their covariance. 
# If subjects who start with high reaction times also degrade faster, lme4 captures that correlation, 
# while this basic BGLR setup misses it.

# ==========================================
# 2. Epidemiology: Generalized Mixed Models (cbpp)
# ==========================================
# BGLR handles binary and categorical data beautifully via response_type = "ordinal". However, lme4 accepts 
# aggregated binomial data (successes and failures). To use BGLR, we must first "unroll" the aggregated dataset 
# into individual binary rows (0 for healthy, 1 for diseased).
data("cbpp")

# ==========================================
# 1. UNROLL DATA TO BINARY FOR BGLR
# ==========================================
# Expand data so each animal has its own row (1 = sick, 0 = healthy)
expanded_cbpp <- data.frame(
  period = rep(cbpp$period, times = cbpp$size),
  herd   = rep(cbpp$herd, times = cbpp$size),
  y      = unlist(apply(cbpp, 1, function(row) {
    sick <- as.numeric(row["incidence"])
    total <- as.numeric(row["size"])
    c(rep(1, sick), rep(0, total - sick))
  }))
)

# ==========================================
# 2. BGLR SETUP (Threshold / Probit Model)
# ==========================================
X_period <- model.matrix(~ period, data = expanded_cbpp)[, -1, drop = FALSE]
Z_herd   <- model.matrix(~ herd - 1, data = expanded_cbpp)

ETA_epi <- list(
  list(X = X_period, model = "FIXED"),
  list(X = Z_herd,   model = "BRR")
)

# response_type = "ordinal" fits a probit link for binary data
fit_bglr_epi <- BGLR(y = expanded_cbpp$y, 
                     ETA = ETA_epi, 
                     response_type = "ordinal", 
                     nIter = 5000, burnIn = 1000, verbose = FALSE)

# ==========================================
# 3. lme4 SETUP (Logit Model)
# ==========================================
# We use family = binomial(link="probit") to ensure a 1:1 math comparison with BGLR
fit_lme4_epi <- glmer(cbind(incidence, size - incidence) ~ period + (1 | herd), 
                      family = binomial(link = "probit"), 
                      data = cbpp)

# ==========================================
# 4. COMPARISON
# ==========================================
cat("Fixed Effect (Period 2):\n lme4:", fixef(fit_lme4_epi)["period2"], 
    "\n BGLR:", fit_bglr_epi$ETA[[1]]$b[1], "\n")

# To compare herd variance on the liability scale
cat("\nHerd Variance (Liability Scale):\n lme4:", as.data.frame(VarCorr(fit_lme4_epi))$vcov, 
    "\n BGLR:", fit_bglr_epi$ETA[[2]]$varB, "\n")

# ==========================================
# 3. Education: Deeply Nested Data
# ==========================================
# Deep nesting in BGLR is actually quite straightforward—you simply map each level of the hierarchy 
# to its own independent incidence matrix and add it as a new "layer" in the ETA list.
# (Assuming the df_edu dataset from the previous step is still in your environment)

# Install packages if missing
# install.packages(c("BGLR", "lme4"))

# Install packages if missing
# install.packages(c("BGLR", "lme4"))

library(BGLR)
library(lme4)

# ==========================================
# 1. SIMULATE THE DATA (Creates df_edu)
# ==========================================
set.seed(42)

# Hierarchy: 5 Schools -> 4 Classrooms per School -> 20 Students per Class
n_schools <- 5
n_classrooms_per_school <- 4
n_students_per_class <- 20
total_n <- n_schools * n_classrooms_per_school * n_students_per_class

school_id <- rep(1:n_schools, each = n_classrooms_per_school * n_students_per_class)
class_id  <- rep(1:(n_schools * n_classrooms_per_school), each = n_students_per_class)

curriculum <- sample(c("Standard", "New"), total_n, replace = TRUE)
student_ses <- rnorm(total_n, mean = 0, sd = 1) 

school_effects <- rnorm(n_schools, 0, 5)
class_effects  <- rnorm(n_schools * n_classrooms_per_school, 0, 3)

math_score <- 50 + 
  ifelse(curriculum == "New", 10, 0) + 
  (5 * student_ses) + 
  school_effects[school_id] + 
  class_effects[class_id] + 
  rnorm(total_n, 0, 2) 

# This creates the dataframe that was missing!
df_edu <- data.frame(school_id, class_id, curriculum, student_ses, math_score)


# ==========================================
# 2. BGLR SETUP (Multiple BRR Layers)
# ==========================================
# Fixed effects: Curriculum and SES
X_edu <- model.matrix(~ curriculum + student_ses, data = df_edu)[, -1, drop = FALSE]

# Random effects: School and Class
Z_school <- model.matrix(~ as.factor(school_id) - 1, data = df_edu)
Z_class  <- model.matrix(~ as.factor(class_id) - 1, data = df_edu)

ETA_edu <- list(
  list(X = X_edu,    model = "FIXED"),
  list(X = Z_school, model = "BRR"),  # Layer 1: School variance
  list(X = Z_class,  model = "BRR")   # Layer 2: Class variance
)

fit_bglr_edu <- BGLR(y = df_edu$math_score, 
                     ETA = ETA_edu, 
                     nIter = 5000, 
                     burnIn = 1000, 
                     verbose = FALSE)


# ==========================================
# 3. lme4 SETUP
# ==========================================
fit_lme4_edu <- lmer(math_score ~ curriculum + student_ses + (1 | school_id/class_id), 
                     data = df_edu)


# ==========================================
# 4. COMPARISON
# ==========================================
vc_lme4 <- as.data.frame(VarCorr(fit_lme4_edu))

cat("\n--- SCHOOL VARIANCE (True = ~25) ---\n")
cat("lme4: ", vc_lme4$vcov[vc_lme4$grp == "school_id"], "\n")
cat("BGLR: ", fit_bglr_edu$ETA[[2]]$varB, "\n\n")

cat("--- CLASSROOM VARIANCE (True = ~9) ---\n")
cat("lme4: ", vc_lme4$vcov[vc_lme4$grp == "class_id:school_id"], "\n")
cat("BGLR: ", fit_bglr_edu$ETA[[3]]$varB, "\n")

