# 1. Load Required Libraries
library(GWASpoly)
library(Rcpp)
library(RcppArmadillo)

# Compile our C++ functions (Ensure these files are in your working directory)
#setwd("/Users/umeshrosyara/Documents/githubdir/FastMixedModel/")
Rcpp::sourceCpp("mixed_solve.cpp")
Rcpp::sourceCpp("FastGWASpoly_chunk.cpp")

# ==========================================
# 2. Simulate Data
# ==========================================
set.seed(123)
N_ind <- 200   
N_mark <- 1000 

# Simulate diploid genotype dosages (0, 1, 2)
geno_matrix <- matrix(sample(0:2, N_ind * N_mark, replace = TRUE), 
                      nrow = N_mark, ncol = N_ind)
rownames(geno_matrix) <- paste0("M", 1:N_mark)
colnames(geno_matrix) <- paste0("Ind_", 1:N_ind)

# Create true causal effects for marker M150 and M800
causal_effects <- rep(0, N_mark)
causal_effects[150] <- 3.0
causal_effects[800] <- -2.5

# Generate Phenotypes
g <- as.vector(crossprod(geno_matrix, causal_effects))
h2 <- 0.6 # Heritability
y <- g + rnorm(N_ind, mean = 0, sd = sqrt((1 - h2) / h2 * var(g)))

# Format data for GWASpoly
pheno_df <- data.frame(Name = colnames(geno_matrix), Trait1 = y)
geno_df <- data.frame(
  Marker = rownames(geno_matrix),
  Chrom = rep(1, N_mark),
  Position = 1:N_mark
)
geno_df <- cbind(geno_df, geno_matrix)

write.csv(pheno_df, "temp_pheno.csv", row.names = FALSE)
write.csv(geno_df, "temp_geno.csv", row.names = FALSE)

# ==========================================
# 3. Run GWASpoly
# ==========================================
cat("Running GWASpoly...\n")
data_gwaspoly <- read.GWASpoly(ploidy = 2, 
                               pheno.file = "temp_pheno.csv", 
                               geno.file = "temp_geno.csv", 
                               format = "numeric", 
                               n.traits = 1, 
                               delim = ",")

# GWASpoly automatically uses the P3D approach by default
data_gwaspoly <- set.K(data_gwaspoly, LOCO=FALSE)
params <- set.params(fixed=NULL, fixed.type=NULL)
data_gwaspoly <- GWASpoly(data_gwaspoly, models = "additive", 
                          traits = "Trait1", params = params)

# Extract -log10(p) scores
scores_gwaspoly <- data_gwaspoly@scores$Trait1$additive

# ==========================================
# 4. Run fastGWASpoly_chunk (C++ Method)
# ==========================================
cat("Running fastGWASpoly_chunk...\n")

# A. Calculate Kinship (Centered GRM)
M_centered <- scale(t(geno_matrix), center = TRUE, scale = FALSE)
K_mat <- tcrossprod(M_centered) / ncol(M_centered)

# B. Fit the Null Model (Intercept only)
X_null <- matrix(1, nrow = N_ind, ncol = 1) 
null_fit <- mixed_solve_cpp(y = y, 
                            K_in = K_mat, 
                            X_in = X_null, 
                            method = "REML", 
                            return_Hinv = TRUE)

# C. Run the Chunk Evaluator (passing all markers as a single chunk for testing)
# We transpose geno_matrix because the C++ function expects markers as columns
res_chunk <- fastGWASpoly_chunk(y = y, 
                                X_null = X_null, 
                                Hinv = null_fit$Hinv, 
                                Vu = null_fit$Vu, 
                                M_chunk = t(geno_matrix))

# Convert custom p-values to -log10(p) scores
scores_chunk <- -log10(res_chunk$p_value)

# ==========================================
# 5. Compare Results
# ==========================================
cat("\n--- Comparison Results ---\n")

# Calculate Pearson correlation
correlation <- cor(scores_gwaspoly, scores_chunk, use = "complete.obs")
cat(sprintf("Correlation of -log10(p) scores: %.6f\n", correlation))

# Check maximum absolute difference
max_diff <- max(abs(scores_gwaspoly - scores_chunk), na.rm = TRUE)
cat(sprintf("Maximum absolute difference in scores: %.6f\n", max_diff))

# Compare Top 5 Hits
top_gwaspoly <- order(scores_gwaspoly, decreasing = TRUE)[1:5]
top_chunk <- order(scores_chunk, decreasing = TRUE)[1:5]

cat("\nTop 5 Marker Indices (GWASpoly):     ", top_gwaspoly, "\n")
cat("Top 5 Marker Indices (C++ Chunk):    ", top_chunk, "\n")

# Cleanup
file.remove("temp_pheno.csv", "temp_geno.csv")