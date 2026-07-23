# Install Rcpp and RcppArmadillo if you haven't already:
# install.packages(c("Rcpp", "RcppArmadillo", "rrBLUP"))

library(Rcpp)
library(rrBLUP)

# Compile the C++ function (make sure mixed_solve.cpp is in your working directory)
Rcpp::sourceCpp("mixed_solver.cpp")
#Rcpp::sourceCpp("/Users/umeshrosyara/Documents/githubdir/FastMixedModel/mixed_solver.cpp")

# Set up the test case parameters
set.seed(42)
N_lines <- 2000
N_markers <- 10000

# Random population of lines with markers
M <- matrix(ifelse(runif(N_lines * N_markers) < 0.5, -1, 1), N_lines, N_markers)

# Random phenotypes
u_true <- rnorm(N_markers)
g <- as.vector(crossprod(t(M), u_true))
h2 <- 0.5  # heritability
y <- g + rnorm(N_lines, mean = 0, sd = sqrt((1 - h2) / h2 * var(g)))

# ---------------------------------------------------------
# Test 1: Predict Marker Effects (Z = M)
# ---------------------------------------------------------

cat("\n--- Testing Marker Effects (Z=M) ---\n")
# Original R Version
t1 <- system.time({
  ans_r <- mixed.solve(y, Z = M)
})

# C++ Version
t2 <- system.time({
  ans_cpp <- mixed_solve_cpp(y, Z_in = M)
})

cat(sprintf("R Time: %.3f sec | C++ Time: %.3f sec\n", t1["elapsed"], t2["elapsed"]))
cat("Accuracy R  : ", cor(u_true, ans_r$u), "\n")
cat("Accuracy C++: ", cor(u_true, as.vector(ans_cpp$u)), "\n")
cat("Difference in predicted u: ", max(abs(ans_r$u - as.vector(ans_cpp$u))), "\n")


# ---------------------------------------------------------
# Test 2: Predict Breeding Values (K = A.mat)
# ---------------------------------------------------------

cat("\n--- Testing Breeding Values (K=A.mat) ---\n")
K_mat <- A.mat(M)

# Original R Version
t3 <- system.time({
  ans_r2 <- mixed.solve(y, K = K_mat)
})

# C++ Version
t4 <- system.time({
  ans_cpp2 <- mixed_solve_cpp(y, K_in = K_mat)
})

cat(sprintf("R Time: %.3f sec | C++ Time: %.3f sec\n", t3["elapsed"], t4["elapsed"]))
cat("Accuracy R  : ", cor(g, ans_r2$u), "\n")
cat("Accuracy C++: ", cor(g, as.vector(ans_cpp2$u)), "\n")
cat("Difference in predicted u: ", max(abs(ans_r2$u - as.vector(ans_cpp2$u))), "\n")