# FastMixedModel: Distributed C++ GWAS Pipeline

A high-performance, distributed pipeline for solving Linear Mixed Models (LMM) and performing Genome-Wide Association Studies (GWAS) in R.

This project provides highly optimized C++ implementations of the mixed model solvers found in popular packages like `rrBLUP` and `GWASpoly`. By leveraging the **RcppArmadillo** linear algebra library and the **P3D (Population Parameters Previously Determined)** approximation, this pipeline bypasses traditional computational bottlenecks and scales natively across local multicore machines, SSH clusters, and HPC schedulers using the R `future` ecosystem.

## Features

* **Blazing Fast Mixed Models:** A native C++ port of `rrBLUP::mixed.solve` featuring custom log-space Golden Section Search and optimized matrix inversions.
* **P3D / EMMAX Approximation:** Evaluates null model variance components ($V_u$, $V_e$) exactly once, bypassing iterative spectral decompositions for individual genetic markers.
* **Distributed Computing:** Uses block-matrix Generalized Least Squares (GLS) on "chunks" of markers, allowing the workload to be split asynchronously across hundreds of cores or remote servers.
* **Exact Comparability:** Mathematically mirrors the additive model of `GWASpoly`, yielding $>0.999$ correlation in $-\log_{10}(p)$ scores in a fraction of the time for massive datasets.

## Repository Structure

* `mixed_solve.cpp`: The core C++ mixed model solver. Calculates fixed effects (BLUES), random effects (BLUPS), and variance components.
* `fastGWAS_chunk.cpp`: The C++ worker function designed for distributed nodes. Evaluates chunks of markers using Generalized Least Squares against a pre-computed inverse phenotypic variance matrix ($H^{-1}$).
* `fastGWAS_parallel.R`: The R orchestration script. Handles data formatting, null model evaluation, chunk dispatching, and result aggregation.

## Prerequisites

To compile and run the C++ code, your system must have a working C++ compiler and Fortran libraries (required for Armadillo's LAPACK/BLAS bindings).

### System Requirements

* **Windows:** Install [Rtools](https://cran.r-project.org/bin/windows/Rtools/) corresponding to your R version.
* **macOS (Intel/Apple Silicon):** Install Xcode Command Line Tools (`xcode-select --install`). **Crucially**, Apple Silicon (M1/M2/M3) users must install the official CRAN GNU Fortran (`gfortran`) binaries from [mac.r-project.org/tools/](https://mac.r-project.org/tools/) to avoid linker errors.
* **Linux:** Install build-essential and gfortran (`sudo apt install build-essential gfortran`).

### R Dependencies

```R
install.packages(c("Rcpp", "RcppArmadillo", "future", "future.apply"))

```

## Quick Start

### 1. Standalone Mixed Model Prediction

Use `mixed_solve_cpp` as a direct, faster drop-in replacement for `rrBLUP::mixed.solve`.

```R
library(Rcpp)
Rcpp::sourceCpp("mixed_solve.cpp")

# y: phenotype vector, K_mat: Kinship Matrix
fit <- mixed_solve_cpp(y = y, K_in = K_mat, method = "REML")

print(fit$Vu)     # Genetic variance
print(fit$u)      # Predicted breeding values (BLUPs)

```

### 2. Distributed GWAS

To run a full GWAS, utilize the orchestration script. The Master node will fit the null model, and Worker nodes will evaluate the markers.

```R
library(Rcpp)
library(future)
library(future.apply)

Rcpp::sourceCpp("mixed_solve.cpp")
Rcpp::sourceCpp("fastGWAS_chunk.cpp")

# Define your compute backend (e.g., 8 local cores)
plan(multisession, workers = 8)

# 1. Fit Null Model (Master Node)
X_null <- matrix(1, nrow = length(y), ncol = 1) 
null_fit <- mixed_solve_cpp(y = y, K_in = K_mat, X_in = X_null, return_Hinv = TRUE)

# 2. Split Genotypes into Chunks (Rows = Markers, Cols = Individuals)
chunk_size <- 5000
chunk_indices <- split(1:nrow(geno_matrix), ceiling(seq_along(1:nrow(geno_matrix)) / chunk_size))

# 3. Distributed Evaluation (Worker Nodes)
results_list <- future_lapply(chunk_indices, function(idx) {
  
  # Subset and transpose so individuals are rows
  M_chunk <- t(geno_matrix[idx, , drop = FALSE])
  
  res <- fastGWASpoly_chunk(y = y, 
                            X_null = X_null, 
                            Hinv = null_fit$Hinv, 
                            Vu = null_fit$Vu, 
                            M_chunk = M_chunk)
  
  data.frame(Marker_Index = idx, Beta = res$beta, SE = res$SE, P_value = res$p_value)
}, future.seed = TRUE)

# 4. Aggregate Results
gwas_results <- do.call(rbind, results_list)
gwas_results$MinusLog10P <- -log10(gwas_results$P_value)

```

## Optimization Tips

For maximum performance, create a `Makevars` file (`~/.R/Makevars` on Mac/Linux, `~/.R/Makevars.win` on Windows) to enable compiler optimizations:

```makefile
CXX11FLAGS=-O3 -Wall -mtune=native -march=native
CXX14FLAGS=-O3 -Wall -mtune=native -march=native

```

