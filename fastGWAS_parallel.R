# Load libraries
library(Rcpp)
library(future)
library(future.apply)

# Compile functions
Rcpp::sourceCpp("mixed_solve.cpp")       # The previous script
Rcpp::sourceCpp("FastGWASpoly_chunk.cpp")    # The new chunk script

# ==============================================================
# 1. Server Configuration (Define your parallel backend)
# ==============================================================

# Option A: Run in parallel on a single large server (e.g., 32 cores)
plan(multisession, workers = 4)

# Option B: Run on multiple remote servers (SSH clusters)
# plan(cluster, workers = c("server1.domain.com", "server2.domain.com", "server3.domain.com"))

# Option C: Run on an HPC Scheduler (requires future.batchtools)
# library(future.batchtools)
# plan(batchtools_slurm)

# ==============================================================
# 2. Master Node: Fit the Null Model
# ==============================================================
cat("Fitting null model on master node...\n")

# Assuming 'y' is phenotypes, 'K_mat' is kinship, and 'geno_matrix' is an N x M genotype matrix
X_null <- matrix(1, nrow = length(y), ncol = 1) 

# Run mixed_solve_cpp ONCE requesting the Hinv matrix
null_fit <- mixed_solve_cpp(y = y, 
                            K_in = K_mat, 
                            X_in = X_null, 
                            method = "REML", 
                            return_Hinv = TRUE)

Vu_est <- null_fit$Vu
Hinv_est <- null_fit$Hinv

# ==============================================================
# 3. Master Node: Split Data into Chunks
# ==============================================================
# CORRECTED: Use nrow() because markers are the rows in geno_matrix
n_total_markers <- nrow(geno_matrix) 
chunk_size <- 5000  

# Create indices for the chunks
chunk_indices <- split(1:n_total_markers, ceiling(seq_along(1:n_total_markers) / chunk_size))

cat(sprintf("Distributing %d markers across %d chunks to servers...\n", 
            n_total_markers, length(chunk_indices)))

# ==============================================================
# 4. Worker Nodes: Run fastGWASpoly_chunk in Parallel
# ==============================================================
results_list <- future_lapply(chunk_indices, function(idx) {
  
  # CORRECTED: Subset rows (markers) and transpose so individuals are rows
  # Resulting M_chunk dimensions: (N_individuals x N_markers_in_chunk)
  M_chunk <- t(geno_matrix[idx, , drop = FALSE])
  
  # Run the C++ function
  res <- fastGWASpoly_chunk(y = y, 
                            X_null = X_null, 
                            Hinv = Hinv_est, 
                            Vu = Vu_est, 
                            M_chunk = M_chunk)
  
  # Return as a data frame for easy combining later
  data.frame(
    Marker_Index = idx,
    Beta = res$beta,
    SE = res$SE,
    P_value = res$p_value
  )
  
}, future.seed = TRUE)

# ==============================================================
# 5. Master Node: Reassemble Results
# ==============================================================
# Combine the list of data frames back into one large results table
gwas_results <- do.call(rbind, results_list)
gwas_results$MinusLog10P <- -log10(gwas_results$P_value)

cat("Distributed GWAS Complete.\n")
head(gwas_results[order(gwas_results$P_value), ])