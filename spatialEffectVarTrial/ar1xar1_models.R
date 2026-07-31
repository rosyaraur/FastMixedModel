# ==============================================================================
# 0. LOAD LIBRARIES
# ==============================================================================
# install.packages(c("glmmTMB", "emmeans"))
library(glmmTMB)
library(emmeans)

# ==============================================================================
# 1. THE BULLETPROOF WRAPPER FUNCTION
# ==============================================================================
# Required Libraries
# install.packages(c("glmmTMB", "emmeans", "SpATS"))
library(glmmTMB)
library(emmeans)
library(SpATS)

#' Fit Agricultural Spatial Models (Including 2D Splines)
fit_trial_model <- function(data, trait, genotype, rep, row, col, 
                            estimate_type = c("BLUE", "BLUP"), 
                            spatial_model = c("RCB", "AR1_Row", "AR1_Col", "AR1_2D", "Spline_2D")) {
  
  estimate_type <- match.arg(estimate_type)
  spatial_model <- match.arg(spatial_model)
  
  # Standardize dataframe column names and types
  df <- data
  df$Trait <- as.numeric(df[[trait]])
  df$Genotype <- as.factor(df[[genotype]])
  df$Rep <- as.factor(df[[rep]])
  
  # glmmTMB needs factor coordinates; SpATS needs numeric coordinates
  df$Row_f <- as.factor(df[[row]])
  df$Col_f <- as.factor(df[[col]])
  df$Row_n <- as.numeric(as.character(df[[row]]))
  df$Col_n <- as.numeric(as.character(df[[col]]))
  
  # ==============================================================================
  # PATH A: The SpATS Engine (Spline_2D)
  # ==============================================================================
  if (spatial_model == "Spline_2D") {
    
    is_random <- (estimate_type == "BLUP")
    
    mod <- SpATS(response = "Trait", 
                 genotype = "Genotype", 
                 genotype.as.random = is_random, 
                 fixed = ~ Rep, 
                 spatial = ~ PSANOVA(Col_n, Row_n, nseg = c(10, 10)), 
                 data = df)
    
    # 1. Variance Components
    var_list <- list()
    if (is_random) var_list[["Genotype"]] <- as.numeric(mod$var.comp["Genotype"])
    var_list[["Residual"]] <- as.numeric(mod$psi[1])
    
    var_comps_df <- data.frame(
      grp = names(var_list),
      vcov = unlist(var_list),
      row.names = NULL
    )
    
    # 2. Heritability
    heritability <- ifelse(is_random, getHeritability(mod), NA)
    
    # 3. Estimates (Suppressing the verbose prediction messages)
    pred <- suppressMessages(predict(mod, which = "Genotype"))
    estimates <- data.frame(
      Genotype = pred$Genotype,
      Predicted_Yield = pred$predicted.values
    )
    
    # 4. Diagnostics (SpATS does not use standard AIC/BIC comparable to glmmTMB)
    diagnostics <- data.frame(
      Model_Type = spatial_model, Estimate = estimate_type,
      AIC = NA, BIC = NA, LogLik = NA, Formula = "PSANOVA(Col_n, Row_n)"
    )
    
    return(list(Diagnostics = diagnostics, Heritability = heritability, 
                Variance_Components = var_comps_df, Estimates = estimates, Model_Object = mod))
  }
  
  # ==============================================================================
  # PATH B: The glmmTMB Engine (RCB, AR1)
  # ==============================================================================
  df$pos <- numFactor(df$Col_n, df$Row_n)
  df$field_dummy <- factor(1) 
  
  if (estimate_type == "BLUE") {
    base_form <- "Trait ~ Genotype + (1 | Rep)"
  } else {
    base_form <- "Trait ~ (1 | Genotype) + (1 | Rep)"
  }
  
  if (spatial_model == "RCB") form_str <- base_form
  else if (spatial_model == "AR1_Row") form_str <- paste(base_form, "+ ar1(Row_f + 0 | Col_f)")
  else if (spatial_model == "AR1_Col") form_str <- paste(base_form, "+ ar1(Col_f + 0 | Row_f)")
  else if (spatial_model == "AR1_2D") form_str <- paste(base_form, "+ exp(pos + 0 | field_dummy)")
  
  mod <- glmmTMB(as.formula(form_str), data = df)
  
  # 1. Variance Components
  vc <- VarCorr(mod)$cond
  var_list <- list()
  for (grp_name in names(vc)) var_list[[grp_name]] <- as.numeric(vc[[grp_name]][1, 1])
  var_list[["Residual"]] <- as.numeric(attr(vc, "sc")^2)
  var_comps_df <- data.frame(grp = names(var_list), vcov = unlist(var_list), row.names = NULL)
  
  # 2. Estimates & Heritability
  heritability <- NA
  if (estimate_type == "BLUE") {
    em <- emmeans(mod, ~ Genotype)
    estimates <- as.data.frame(em)
    names(estimates)[names(estimates) == "emmean"] <- "Predicted_Yield"
  } else {
    ranef_vals <- ranef(mod)$cond$Genotype
    intercept <- fixef(mod)$cond["(Intercept)"]
    estimates <- data.frame(
      Genotype = rownames(ranef_vals),
      Predicted_Yield = ranef_vals[,"(Intercept)"] + intercept
    )
    
    var_g <- var_comps_df$vcov[var_comps_df$grp == "Genotype"]
    var_e <- var_comps_df$vcov[var_comps_df$grp == "Residual"]
    n_rep <- length(unique(df$Rep))
    if (length(var_g) > 0 && length(var_e) > 0) heritability <- var_g / (var_g + (var_e / n_rep))
  }
  
  # 3. Diagnostics
  diagnostics <- data.frame(
    Model_Type = spatial_model, Estimate = estimate_type,
    AIC = AIC(mod), BIC = BIC(mod), LogLik = as.numeric(logLik(mod)), Formula = form_str
  )
  
  return(list(Diagnostics = diagnostics, Heritability = heritability, 
              Variance_Components = var_comps_df, Estimates = estimates, Model_Object = mod))
}

# ==============================================================================
# 2. RUN THE MOCK DATASET
# ==============================================================================
my_field_data <- data.frame(
  plot_row = rep(1:4, times = 5),
  plot_col = rep(1:5, each = 4),
  block = rep(1:4, each = 5),
  seed_variety = rep(c("Alpha", "Beta", "Gamma", "Delta", "Echo"), times = 4),
  harvest_yield = c(
    12.1, 14.5, 11.2, 15.3, 13.0,
    13.2, 16.1, 12.0, 14.8, 15.1,
    11.8, 15.5, 10.9, 14.1, 12.7,
    14.0, 17.2, 13.4, 16.5, 14.4
  )
)

# Note: Changed to "RCB" since 20 plots is too small for "AR1_2D" to converge
my_results <- fit_trial_model(
  data = my_field_data, 
  trait = "harvest_yield",      
  genotype = "seed_variety",    
  rep = "block",                
  row = "plot_row",             
  col = "plot_col",             
  estimate_type = "BLUP",       
  spatial_model = "RCB"         
)

# Print the Heritability
cat("Trial Heritability:", round(my_results$Heritability, 3), "\n\n")

# Print the adjusted yields
blups <- my_results$Estimates
sorted_blups <- blups[order(blups$Predicted_Yield, decreasing = TRUE), ]
print(sorted_blups)

# ==============================================================================
# 2. THE SIMULATION FUNCTION
# ==============================================================================
simulate_trial <- function(var_G, var_E, var_Rep, spatial_var, rho_row = 0.8, rho_col = 0.8) {
  n_rows <- 15
  n_cols <- 10
  n_plots <- n_rows * n_cols
  n_reps <- 3
  n_varieties <- 50
  
  df <- expand.grid(Row = 1:n_rows, Column = 1:n_cols)
  df$Replicate <- as.factor(rep(1:n_reps, each = n_plots/n_reps))
  
  # Randomize varieties across reps
  varieties <- paste0("V", 1:n_varieties)
  df$Variety <- as.factor(c(sample(varieties), sample(varieties), sample(varieties)))
  
  # Generate true biological effects
  rep_eff <- rnorm(n_reps, 0, sqrt(var_Rep))
  gen_eff <- rnorm(n_varieties, 0, sqrt(var_G))
  
  # Map true effects to plots
  df$Rep_Effect <- rep_eff[as.numeric(df$Replicate)]
  df$Gen_Effect <- gen_eff[as.numeric(df$Variety)]
  
  # --- NEW: Generate TRUE 2D Spatial Autocorrelation ---
  if (spatial_var > 0) {
    # 1. Create Row and Column AR1 correlation matrices
    R_cor <- rho_row ^ abs(outer(1:n_rows, 1:n_rows, "-"))
    C_cor <- rho_col ^ abs(outer(1:n_cols, 1:n_cols, "-"))
    
    # 2. Combine into a 2D spatial covariance matrix via Kronecker product
    Sigma_spatial <- kronecker(C_cor, R_cor) * spatial_var
    
    # 3. Simulate the spatial field using Cholesky decomposition
    df$Spatial_Effect <- as.numeric(t(chol(Sigma_spatial)) %*% rnorm(n_plots))
  } else {
    df$Spatial_Effect <- 0
  }
  
  # Generate Random Error
  df$Error <- rnorm(n_plots, 0, sqrt(var_E))
  
  # Calculate final plot yield
  df$yield <- 100 + df$Rep_Effect + df$Gen_Effect + df$Spatial_Effect + df$Error
  
  return(df)
}

# ==============================================================================
# 3. RUN THE SIMULATION & COMPARE MODELS
# ==============================================================================
# ==============================================================================
# RE-RUN THE TEST
# ==============================================================================
set.seed(42) # New seed for the new process

# We replaced 'spatial_amp' with 'spatial_var' (True spatial variance = 30)
data_uniform <- simulate_trial(var_G = 15, var_E = 5, var_Rep = 2, spatial_var = 0)
data_wavy <- simulate_trial(var_G = 15, var_E = 5, var_Rep = 2, spatial_var = 30)

# Run Models on Uniform Field
uni_rcb <- fit_trial_model(data_uniform, "yield", "Variety", "Replicate", "Row", "Column", "BLUP", "RCB")
uni_2d  <- fit_trial_model(data_uniform, "yield", "Variety", "Replicate", "Row", "Column", "BLUP", "AR1_2D")

# Run Models on the Spatially Correlated Field
wave_rcb <- fit_trial_model(data_wavy, "yield", "Variety", "Replicate", "Row", "Column", "BLUP", "RCB")
wave_2d  <- fit_trial_model(data_wavy, "yield", "Variety", "Replicate", "Row", "Column", "BLUP", "AR1_2D")
# ==============================================================================
# 4. EXTRACT RESULTS SAFELY
# ==============================================================================
# Helper function to safely extract variance by name (prevents indexing errors)
get_vcov <- function(model_out, group_name) {
  vc <- model_out$Variance_Components
  val <- vc$vcov[vc$grp == group_name]
  return(ifelse(length(val) > 0, val, NA))
}

results_table <- data.frame(
  Dataset = c("Uniform (No Spatial)", "Uniform (No Spatial)", "Wavy (High Spatial)", "Wavy (High Spatial)"),
  Model = c("RCB", "2D Spatial", "RCB", "2D Spatial"),
  AIC = c(uni_rcb$Diagnostics$AIC, uni_2d$Diagnostics$AIC, 
          wave_rcb$Diagnostics$AIC, wave_2d$Diagnostics$AIC),
  Genetic_Var = c(get_vcov(uni_rcb, "Genotype"), get_vcov(uni_2d, "Genotype"), 
                  get_vcov(wave_rcb, "Genotype"), get_vcov(wave_2d, "Genotype")),
  Error_Var = c(get_vcov(uni_rcb, "Residual"), get_vcov(uni_2d, "Residual"), 
                get_vcov(wave_rcb, "Residual"), get_vcov(wave_2d, "Residual")),
  Calculated_H2 = c(uni_rcb$Heritability, uni_2d$Heritability, 
                    wave_rcb$Heritability, wave_2d$Heritability)
)

# Round only the numeric columns so base R doesn't crash trying to round text strings
numeric_cols <- c("AIC", "Genetic_Var", "Error_Var", "Calculated_H2")
results_table[numeric_cols] <- lapply(results_table[numeric_cols], round, 2)

print(results_table)

# --- 5. THE ULTIMATE COMPARISON TABLE ---

# We know the exact true values because we simulated them:
true_G <- 15
true_E <- 5
true_H2 <- true_G / (true_G + (true_E / 3))

# Build a comparative dataframe
comparison_table <- data.frame(
  Dataset = c("Uniform", "Uniform", "Correlated", "Correlated"),
  Model = c("RCB", "2D Spatial", "RCB", "2D Spatial"),
  
  # Genetic Variance Comparison
  True_Gen_Var = true_G,
  Est_Gen_Var = c(get_vcov(uni_rcb, "Genotype"), get_vcov(uni_2d, "Genotype"), 
                  get_vcov(wave_rcb, "Genotype"), get_vcov(wave_2d, "Genotype")),
  
  # Error (Residual) Variance Comparison
  True_Err_Var = true_E,
  Est_Err_Var = c(get_vcov(uni_rcb, "Residual"), get_vcov(uni_2d, "Residual"), 
                  get_vcov(wave_rcb, "Residual"), get_vcov(wave_2d, "Residual")),
  
  # Heritability Comparison
  True_H2 = true_H2,
  Est_H2 = c(uni_rcb$Heritability, uni_2d$Heritability, 
             wave_rcb$Heritability, wave_2d$Heritability)
)

# Clean up rounding for display
numeric_cols <- c("True_Gen_Var", "Est_Gen_Var", "True_Err_Var", "Est_Err_Var", "True_H2", "Est_H2")
comparison_table[numeric_cols] <- lapply(comparison_table[numeric_cols], round, 2)

print(comparison_table)

# Testing AR1_2D vs. Spline_2D
set.seed(42)

# Generate a field with heavy spatial autocorrelation
# (Assumes simulate_trial is already loaded in your environment)
data_wavy <- simulate_trial(var_G = 15, var_E = 5, var_Rep = 2, spatial_var = 30)

# Fit the three distinct models
wave_rcb    <- fit_trial_model(data_wavy, "yield", "Variety", "Replicate", "Row", "Column", "BLUP", "RCB")
wave_ar1_2d <- fit_trial_model(data_wavy, "yield", "Variety", "Replicate", "Row", "Column", "BLUP", "AR1_2D")
wave_spline <- fit_trial_model(data_wavy, "yield", "Variety", "Replicate", "Row", "Column", "BLUP", "Spline_2D")

# Extract and Compare
get_vcov <- function(model_out, group_name) {
  vc <- model_out$Variance_Components
  val <- vc$vcov[vc$grp == group_name]
  return(ifelse(length(val) > 0, val, NA))
}

comparison_table <- data.frame(
  Model = c("Standard RCB", "glmmTMB (AR1_2D)", "SpATS (Spline_2D)"),
  Genetic_Var = c(get_vcov(wave_rcb, "Genotype"), get_vcov(wave_ar1_2d, "Genotype"), get_vcov(wave_spline, "Genotype")),
  Error_Var   = c(get_vcov(wave_rcb, "Residual"), get_vcov(wave_ar1_2d, "Residual"), get_vcov(wave_spline, "Residual")),
  Heritability = c(wave_rcb$Heritability, wave_ar1_2d$Heritability, wave_spline$Heritability)
)

numeric_cols <- c("Genetic_Var", "Error_Var", "Heritability")
comparison_table[numeric_cols] <- lapply(comparison_table[numeric_cols], round, 2)

print(comparison_table)