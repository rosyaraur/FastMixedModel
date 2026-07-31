# =====================================================================
# 0. LOAD REQUIRED LIBRARIES
# =====================================================================
library(lme4)
library(blme)
library(mbest)
library(brms)
library(sommer)
library(MCMCglmm)
library(BGLR)
library(rrBLUP)

# =====================================================================
# 1. MASTER MIXED SOLVER (Universal Routing & Automation)
# =====================================================================
MasterMixedSolver <- function(formula, data, paradigm = "Frequentist", engine = "auto", K_matrix = NULL, K_is_inverse = FALSE, nIter = 2000, burnIn = 500) {
  
  parsed_bars <- lme4::findbars(formula)
  n_ranef <- length(parsed_bars)
  k_supported_engines <- c("sommer", "rrBLUP", "BGLR", "MCMCglmm")
  
  message("\n--- MasterMixedSolver Routing Diagnostics ---")
  
  if (!is.null(K_matrix)) {
    if (engine != "auto" && !(engine %in% k_supported_engines)) {
      message(sprintf("[!] OVERRIDE: Requested engine '%s' does not support Kinship matrices natively.", engine))
      engine <- "auto"
    }
    if (engine == "auto") {
      engine <- if (paradigm == "Bayesian") "BGLR" else "sommer"
      message(sprintf("[✓] Auto-selected '%s' for Kinship model.", engine))
    }
  } else {
    if (engine == "auto") {
      engine <- if (paradigm == "Bayesian") "blme" else "lme4"
      message(sprintf("[✓] Auto-selected '%s' for standard model.", engine))
    }
  }
  
  # Matrix Inversion Automation
  if (!is.null(K_matrix)) {
    needs_inverse <- (engine == "MCMCglmm")
    if (needs_inverse && !K_is_inverse) {
      message("[!] AUTOMATION: MCMCglmm requires an inverse matrix. Inverting...")
      K_matrix <- if (inherits(K_matrix, "Matrix")) Matrix::solve(K_matrix) else solve(K_matrix)
    } else if (!needs_inverse && K_is_inverse) {
      message(sprintf("[!] AUTOMATION: '%s' requires a covariance matrix. Un-inverting...", engine))
      K_matrix <- if (inherits(K_matrix, "Matrix")) Matrix::solve(K_matrix) else solve(K_matrix)
    }
  }
  message("---------------------------------------------\n")
  
  # Engine Execution Blocks
  if (engine == "lme4") return(lme4::lmer(formula, data = data))
  if (engine == "blme") return(blme::blmer(formula, data = data))
  if (engine == "mbest") return(mbest::mhglm(formula, data = data))
  if (engine == "brms") return(brms::brm(formula, data = data, iter = nIter, warmup = burnIn, chains = 2, silent = 2, refresh = 0))
  
  if (engine == "sommer") {
    fixed_form <- lme4::nobars(formula)
    if (!is.null(K_matrix) && length(parsed_bars) > 0) {
      ran_term <- deparse(parsed_bars[[1]][[3]])
      Gu_mat <- as.matrix(K_matrix)
      if (!is.null(rownames(K_matrix))) {
        rownames(Gu_mat) <- rownames(K_matrix)
        colnames(Gu_mat) <- colnames(K_matrix)
      }
      ran_form <- as.formula(sprintf("~ sommer::vs(%s, Gu = Gu_mat)", ran_term))
      return(sommer::mmer(fixed = fixed_form, random = ran_form, rcov = ~ units, data = data, verbose = FALSE))
    } else {
      ran_form <- if(length(parsed_bars) > 0) as.formula(paste("~", paste(sapply(parsed_bars, function(x) deparse(x[[3]])), collapse = " + "))) else NULL
      return(sommer::mmer(fixed = fixed_form, random = ran_form, data = data, verbose = FALSE))
    }
  }
  
  if (engine == "rrBLUP") {
    y_vec <- data[[as.character(formula[[2]])]]
    return(rrBLUP::mixed.solve(y = y_vec, K = K_matrix))
  }
  
  if (engine == "BGLR") {
    y_vec <- data[[as.character(formula[[2]])]]
    fixed_form <- lme4::nobars(formula)
    # Exclude intercept from design matrix since BGLR handles global mu separately
    X_matrix <- model.matrix(update(fixed_form, . ~ . - 1), data = data)
    ETA <- list(list(X = X_matrix, model = "FIXED"))
    
    if (!is.null(K_matrix) && length(parsed_bars) > 0) {
      ran_term <- deparse(parsed_bars[[1]][[3]])
      K_dense <- as.matrix(K_matrix)
      # Standardize kernel matrix for BGLR numerical stability
      K_dense <- K_dense / mean(diag(K_dense))
      if (!is.null(rownames(K_matrix))) {
        rownames(K_dense) <- rownames(K_matrix)
        colnames(K_dense) <- colnames(K_matrix)
      }
      ETA <- append(ETA, list(list(K = K_dense, model = "RKHS", Name = ran_term)))
    }
    return(BGLR::BGLR(y = y_vec, ETA = ETA, nIter = nIter, burnIn = burnIn, verbose = FALSE))
  }
  
  if (engine == "MCMCglmm") {
    fixed_form <- lme4::nobars(formula)
    ran_form <- if (length(parsed_bars) > 0) as.formula(paste("~", paste(sapply(parsed_bars, function(x) deparse(x[[3]])), collapse = " + "))) else NULL
    
    if (!is.null(K_matrix) && length(parsed_bars) > 0) {
      ginv <- list()
      ran_name <- deparse(parsed_bars[[1]][[3]])
      ginv[[ran_name]] <- K_matrix
      return(MCMCglmm::MCMCglmm(fixed = fixed_form, random = ran_form, ginverse = ginv, data = data, nitt = nIter, burnin = burnIn, pr = TRUE, verbose = FALSE))
    } else {
      return(MCMCglmm::MCMCglmm(fixed = fixed_form, random = ran_form, data = data, nitt = nIter, burnin = burnIn, pr = TRUE, verbose = FALSE))
    }
  }
  
  stop("Engine not recognized.")
}

# =====================================================================
# 2. UNIVERSAL EXTRACTOR FUNCTION (Robust ID & Name Mapping)
# =====================================================================
ExtractMixedInfo <- function(model_fit) {
  res <- list(Engine = "Unknown", FixedEffects = NULL, RandomEffects = NULL, VarianceComponents = NULL)
  
  # lme4, blme, mbest
  if (inherits(model_fit, c("lmerMod", "bmerMod", "mhglm", "glmerMod", "blmerMod"))) {
    res$Engine <- if (inherits(model_fit, "bmerMod")) "blme" else if (inherits(model_fit, "mhglm")) "mbest" else "lme4"
    fe <- lme4::fixef(model_fit)
    if (is.null(names(fe))) names(fe) <- if (length(fe) == 2) c("(Intercept)", "Days") else paste0("beta_", seq_along(fe))
    res$FixedEffects <- fe
    
    re_list <- lme4::ranef(model_fit)
    if (length(re_list) > 0) {
      re_df <- re_list[[1]]
      res$RandomEffects <- data.frame(ID = rownames(re_df), Value = as.numeric(re_df[, 1]))
    }
    
    vc <- lme4::VarCorr(model_fit)
    var_comp <- list()
    for (grp in names(vc)) var_comp[[grp]] <- as.numeric(diag(vc[[grp]]))
    var_comp[["Residual"]] <- as.numeric(attr(vc, "sc")^2)
    res$VarianceComponents <- var_comp
    return(res)
  }
  
  # brms
  if (inherits(model_fit, "brmsfit")) {
    res$Engine <- "brms"
    fe <- brms::fixef(model_fit)
    res$FixedEffects <- fe[, "Estimate"]
    names(res$FixedEffects) <- rownames(fe)
    
    re_list <- brms::ranef(model_fit, summary = TRUE)
    if (length(re_list) > 0) {
      subj_array <- re_list[[1]]
      est_col <- if ("Estimate" %in% dimnames(subj_array)[[3]]) "Estimate" else 1
      val_vec <- subj_array[, 1, est_col]
      res$RandomEffects <- data.frame(ID = names(val_vec), Value = as.numeric(val_vec))
    }
    
    vc <- brms::VarCorr(model_fit, summary = TRUE)
    var_comp <- list()
    for (grp in names(vc)) {
      var_comp[[grp]] <- if (grp == "residual__") as.numeric(vc[[grp]]$sd[1, "Estimate"])^2 else as.numeric(vc[[grp]]$sd[, "Estimate"])^2
    }
    res$VarianceComponents <- var_comp
    return(res)
  }
  
  # sommer (Robustly cleans row names and maps U and sigma components)
  if (inherits(model_fit, "mmer")) {
    res$Engine <- "sommer"
    fe_df <- model_fit$Beta
    if (is.data.frame(fe_df) && "Estimate" %in% names(fe_df)) {
      fe_vals <- as.numeric(fe_df$Estimate)
      names(fe_vals) <- if (!is.null(rownames(fe_df))) rownames(fe_df) else paste0("beta_", seq_along(fe_vals))
    } else {
      fe_vals <- as.numeric(unlist(fe_df))
      names(fe_vals) <- names(fe_df)
    }
    res$FixedEffects <- fe_vals
    
    if (!is.null(model_fit$U) && length(model_fit$U) > 0) {
      u_obj <- model_fit$U[[1]]
      if (is.data.frame(u_obj) || is.matrix(u_obj)) {
        u_vals <- as.numeric(u_obj[, 1])
        u_ids <- rownames(u_obj)
      } else {
        u_vals <- as.numeric(unlist(u_obj))
        u_ids <- names(u_obj)
      }
      if (is.null(u_ids)) u_ids <- paste0("ID_", seq_along(u_vals))
      # Strip multi-part dot prefixes like 'animal.Fem2' or 'u:animal.Fem2' -> 'Fem2'
      u_ids <- sub("^.*\\.", "", u_ids)
      res$RandomEffects <- data.frame(ID = as.character(u_ids), Value = u_vals)
    }
    
    var_comp <- list()
    for (grp in names(model_fit$sigma)) {
      sig_obj <- model_fit$sigma[[grp]]
      val_sig <- as.numeric(unlist(sig_obj))[1]
      if (grepl("units", grp)) {
        var_comp[["Residual"]] <- val_sig
      } else {
        var_comp[[grp]] <- val_sig
      }
    }
    res$VarianceComponents <- var_comp
    return(res)
  }
  
  # MCMCglmm
  if (inherits(model_fit, "MCMCglmm")) {
    res$Engine <- "MCMCglmm"
    summ <- summary(model_fit)
    res$FixedEffects <- summ$solutions[, "post.mean"]
    
    n_fixed <- nrow(summ$solutions)
    if ("Sol" %in% names(model_fit) && ncol(model_fit$Sol) > n_fixed) {
      sol_means <- colMeans(model_fit$Sol[, (n_fixed + 1):ncol(model_fit$Sol), drop = FALSE])
      subj_idx <- grep("animal|Subject|hatch", names(sol_means), ignore.case = TRUE)
      if (length(subj_idx) > 0) {
        subj_vals <- sol_means[subj_idx]
        raw_names <- names(subj_vals)
        clean_names <- gsub(".*\\(([^,]+),\\s*([^)]+)\\).*", "\\2", raw_names)
        no_paren <- clean_names == raw_names
        clean_names[no_paren] <- gsub(".*\\.", "", raw_names[no_paren])
        res$RandomEffects <- data.frame(ID = trimws(clean_names), Value = as.numeric(subj_vals))
      }
    }
    
    var_comp <- list()
    for (grp in rownames(summ$Gcov)) var_comp[[grp]] <- as.numeric(summ$Gcov[grp, "post.mean"])
    for (grp in rownames(summ$Rcov)) var_comp[if (grp == "units") "Residual" else grp] <- as.numeric(summ$Rcov[grp, "post.mean"])
    res$VarianceComponents <- var_comp
    return(res)
  }
  
  # BGLR (Handles global mu + design matrix names properly)
  if (inherits(model_fit, "BGLR")) {
    res$Engine <- "BGLR"
    fix_eff <- c(); ran_df <- NULL; var_comp <- list()
    
    if (!is.null(model_fit$mu)) {
      fix_eff <- c("(Intercept)" = as.numeric(model_fit$mu))
    }
    
    for (i in seq_along(model_fit$ETA)) {
      term <- model_fit$ETA[[i]]
      name <- ifelse(is.null(term$Name), paste0("ETA_", i), term$Name)
      if (term$model == "FIXED") {
        b_vals <- as.numeric(term$b)
        b_names <- if (!is.null(names(term$b))) names(term$b) else paste0("beta_", seq_along(b_vals))
        names(b_vals) <- b_names
        fix_eff <- c(fix_eff, b_vals)
      } else {
        var_comp[[name]] <- as.numeric(term$varB)
        u_vals <- if (!is.null(term$u)) term$u else term$b
        if (!is.null(u_vals)) {
          ids <- if (!is.null(names(u_vals))) names(u_vals) else if (!is.null(term$K) && !is.null(rownames(term$K))) rownames(term$K) else paste0("ID_", seq_along(u_vals))
          ran_df <- data.frame(ID = as.character(ids), Value = as.numeric(u_vals))
        }
      }
    }
    var_comp[["Residual"]] <- as.numeric(model_fit$varE)
    res$FixedEffects <- fix_eff
    res$RandomEffects <- ran_df
    res$VarianceComponents <- var_comp
    return(res)
  }
  
  # rrBLUP
  if (is.list(model_fit) && "Vu" %in% names(model_fit)) {
    res$Engine <- "rrBLUP (mixed.solve)"
    res$FixedEffects <- c(Intercept = as.numeric(model_fit$beta))
    u_vec <- model_fit$u
    ids <- if (!is.null(names(u_vec))) names(u_vec) else paste0("ID_", seq_along(u_vec))
    res$RandomEffects <- data.frame(ID = as.character(ids), Value = as.numeric(u_vec))
    res$VarianceComponents <- list(Vu = as.numeric(model_fit$Vu), Residual = as.numeric(model_fit$Ve))
    return(res)
  }
  
  return(res)
}

# Helper coefficient extractor
get_coef <- function(fe_vector, target_name) {
  if (is.null(fe_vector)) return(NA)
  if (target_name %in% names(fe_vector)) return(fe_vector[[target_name]])
  match_idx <- which(tolower(names(fe_vector)) == tolower(target_name))
  if (length(match_idx) > 0) return(fe_vector[[match_idx[1]]])
  if (target_name == "(Intercept)" && length(fe_vector) >= 1) return(fe_vector[1])
  if (target_name == "Days" && length(fe_vector) >= 2) return(fe_vector[2])
  return(NA)
}


# =====================================================================
# 3. PART A: STANDARD MODEL BENCHMARK (Sleepstudy)
# =====================================================================
cat("\n=====================================================\n")
cat("  PART A: RUNNING STANDARD LME4 BENCHMARK (Sleepstudy)\n")
cat("=====================================================\n")

data("sleepstudy", package = "lme4")
sleep_formula <- Reaction ~ Days + (1 | Subject)

sleep_engines <- data.frame(
  Paradigm = c("Frequentist", "Bayesian", "Frequentist", "Bayesian", "Bayesian"),
  Engine   = c("lme4", "blme", "mbest", "MCMCglmm", "brms")
)

sleep_fixed <- list(); sleep_var <- list(); sleep_ranef <- list()

for (i in 1:nrow(sleep_engines)) {
  eng <- sleep_engines$Engine[i]; par <- sleep_engines$Paradigm[i]
  cat(sprintf("=> Processing Engine: %-8s ... ", eng))
  
  fit <- tryCatch({
    suppressMessages(suppressWarnings({
      MasterMixedSolver(formula = sleep_formula, data = sleepstudy, paradigm = par, engine = eng, nIter = 2000, burnIn = 500)
    }))
  }, error = function(e) { cat(sprintf("Failed: %s\n", e$message)); return(NULL) })
  
  if (!is.null(fit)) {
    cat("Success\n")
    info <- ExtractMixedInfo(fit)
    
    sleep_fixed[[eng]] <- data.frame(Engine = eng, Paradigm = par, Intercept = get_coef(info$FixedEffects, "(Intercept)"), Days = get_coef(info$FixedEffects, "Days"))
    
    vc <- info$VarianceComponents
    sleep_var[[eng]] <- data.frame(Engine = eng, Paradigm = par, Group_Var = if(!is.null(vc[[1]])) as.numeric(vc[[1]])[1] else NA, Residual = if(!is.null(vc[["Residual"]])) as.numeric(vc[["Residual"]]) else NA)
    
    re_df <- info$RandomEffects
    if (!is.null(re_df)) {
      sub_clean <- data.frame(ID = as.character(re_df$ID), Val = as.numeric(re_df$Value))
      names(sub_clean)[2] <- paste0("BLUP_", eng)
      sleep_ranef[[eng]] <- sub_clean
    }
  }
}

print(do.call(rbind, sleep_fixed))
print(do.call(rbind, sleep_var))
if (length(sleep_ranef) > 0) {
  merged_sleep_blup <- Reduce(function(x, y) merge(x, y, by = "ID", all = TRUE), sleep_ranef)
  print(head(merged_sleep_blup))
}


# =====================================================================
# 4. PART B: ANIMAL MODEL BENCHMARK (Blue Tits Pedigree)
# =====================================================================
cat("\n=====================================================\n")
cat("  PART B: RUNNING ANIMAL MODEL BENCHMARK (Blue Tits)\n")
cat("=====================================================\n")

data("BTdata", package = "MCMCglmm")
data("BTped", package = "MCMCglmm")
BTdata$animal <- as.factor(BTdata$animal)

A_inv <- MCMCglmm::inverseA(BTped)$Ainv
A_cov <- MCMCglmm::inverseA(BTped)$A
rownames(A_cov) <- colnames(A_cov) <- as.character(rownames(A_cov))

animal_formula <- tarsus ~ sex + (1 | animal)

animal_engines <- data.frame(
  Paradigm = c("Frequentist", "Bayesian", "Bayesian"),
  Engine   = c("sommer", "MCMCglmm", "BGLR")
)

animal_fixed <- list(); animal_var <- list(); animal_ranef <- list()

for (i in 1:nrow(animal_engines)) {
  eng <- animal_engines$Engine[i]; par <- animal_engines$Paradigm[i]
  cat(sprintf("=> Processing Engine: %-8s ... ", eng))
  
  current_matrix <- if (eng == "MCMCglmm") A_inv else A_cov
  is_inv_flag   <- if (eng == "MCMCglmm") TRUE else FALSE
  
  fit <- tryCatch({
    suppressMessages(suppressWarnings({
      MasterMixedSolver(formula = animal_formula, data = BTdata, paradigm = par, engine = eng, K_matrix = current_matrix, K_is_inverse = is_inv_flag, nIter = 3000, burnIn = 1000)
    }))
  }, error = function(e) { cat(sprintf("Failed: %s\n", e$message)); return(NULL) })
  
  if (!is.null(fit)) {
    cat("Success\n")
    info <- ExtractMixedInfo(fit)
    
    fe <- info$FixedEffects
    animal_fixed[[eng]] <- data.frame(Engine = eng, Paradigm = par, Intercept = get_coef(fe, "(Intercept)"), SexM = get_coef(fe, "sexM"))
    
    vc <- info$VarianceComponents
    anim_var_val <- if(!is.null(vc[["animal"]])) as.numeric(vc[["animal"]]) else if(length(vc) > 0) as.numeric(vc[[1]])[1] else NA
    animal_var[[eng]] <- data.frame(Engine = eng, Paradigm = par, Animal_Var = anim_var_val, Residual = if(!is.null(vc[["Residual"]])) as.numeric(vc[["Residual"]]) else NA)
    
    re_df <- info$RandomEffects
    if (!is.null(re_df)) {
      sub_clean <- data.frame(ID = as.character(re_df$ID), Val = as.numeric(re_df$Value))
      names(sub_clean)[2] <- paste0("EBV_", eng)
      animal_ranef[[eng]] <- sub_clean
    }
  }
}

print(do.call(rbind, animal_fixed))
print(do.call(rbind, animal_var))
if (length(animal_ranef) > 0) {
  merged_animal_ebv <- Reduce(function(x, y) merge(x, y, by = "ID", all = TRUE), animal_ranef)
  cat("\n=====================================================\n")
  cat("  3. BREEDING VALUES (EBVs) COMPARISON [First 6 Animals]\n")
  cat("=====================================================\n")
  print(head(merged_animal_ebv))
}