# SparseLMM: High-Performance Mixed Model Engine in R/C++

SparseLMM is a custom, high-performance computational engine for solving linear mixed-effects models. Designed for biometrics data scientists and quantitative geneticists, it bridges R's flexible formula interface with a blazing-fast C++ backend powered by `RcppEigen`.

This engine implements two distinct mathematical pathways for maximizing the Restricted Maximum Likelihood (REML), allowing users to switch between derivative-free and Newton-Raphson optimization strategies depending on the analytical pipeline.

## 🚀 Features

* **Dual Algorithm Dispatch:**
* **AI-REML:** Uses the Average Information (AI) algorithm (mimicking ASReml) for rapid convergence using first and second derivatives.
* **Profiled REML:** Uses a derivative-free profiled likelihood optimization (mimicking lme4) for high stability.


* **Sparse Matrix Algebra:** Leverages Eigen's `SimplicialLDLT` sparse Cholesky factorization to handle highly dimensional random effects (e.g., blocking factors in crop breeding programs) without exorbitant memory overhead.
* **Zero Matrix Inversion:** Profiled likelihood evaluations bypass computationally heavy matrix inversions entirely by extracting the log-determinant directly from the Cholesky factor.

## 🛠️ Architecture

### The Mathematics (Henderson's MME)

At its core, the C++ engine (`lmm_engine.cpp`) constructs and solves Henderson's Mixed Model Equations:

$$\begin{bmatrix} X^T X & X^T Z \\ Z^T X & Z^T Z + \lambda I \end{bmatrix} \begin{bmatrix} \hat{\beta} \\ \hat{u} \end{bmatrix} = \begin{bmatrix} X^T y \\ Z^T y \end{bmatrix}$$

### The C++ Backend (`lmm_engine.cpp`)

* **`build_mme_lhs`:** Dynamically constructs the sparse Left-Hand Side matrix ($C$) using Eigen's `Triplet` insertion.
* **`eval_profiled_reml_cpp`:** Evaluates the profiled REML deviance given a variance ratio ($\lambda = \sigma^2_e / \sigma^2_u$) and returns the Penalized Residual Sum of Squares (PRSS).
* **`step_aireml_cpp`:** Executes a single Newton-Raphson update step using algebraic approximations of the AI matrix (Gilmour et al., 1995).

### The R Wrapper (`fit_mixed_model.R`)

Acts as the memory manager and traffic controller. It parses formulas, generates the dense $X$ and sparse $Z$ (`dgCMatrix`) model matrices, and orchestrates the optimization loops (either via `optimize()` or a custom AI-REML `while` loop).

## 💻 Usage & Example

*Note: The simulated multi-environment trial datasets included in this repository are intended strictly as user examples for testing the engine, rather than primary data for biological analysis.*

```R
library(Rcpp)
library(Matrix)

# Compile the C++ backend
sourceCpp("src/lmm_engine.cpp")
source("R/fit_mixed_model.R")

# 1. Generate Example Simulation Data
set.seed(2026)
n_groups <- 50
obs_per_group <- 40
N <- n_groups * obs_per_group

group_ids <- factor(rep(1:n_groups, each = obs_per_group))
x1 <- rnorm(N, mean = 2, sd = 1)
u_true <- rnorm(n_groups, mean = 0, sd = sqrt(1.25))
e_true <- rnorm(N, mean = 0, sd = sqrt(0.75))

y <- 3.5 + 1.8 * x1 + u_true[group_ids] + e_true
sim_data <- data.frame(y = y, x1 = x1, group = group_ids)

# 2. Fit Models
# AI-REML Approach
fit_aireml <- fit_mixed_model(y ~ x1, ~ group, data = sim_data, algorithm = "aireml")
print(fit_aireml)

# Profiled REML Approach
fit_profiled <- fit_mixed_model(y ~ x1, ~ group, data = sim_data, algorithm = "profiled")
print(fit_profiled)

```

## ⚠️ Known Limitations & Roadmap

This engine is foundational. The following architectural limitations are actively being addressed for the next release to fully support complex spatial trial corrections and genomic models:

1. **Dense Intermediate Memory Spikes:** The sparse triplet builder currently forces a dense evaluation of $X^T Z$. For massive marker matrices or complex fixed effects, this can trigger out-of-memory (OOM) errors. *Roadmap: Implement column-wise sparse iterator dot products.*
2. **Single Random Effect Assumption:** The formula parser and $Z$ matrix constructor are currently hardcoded for a single variance component ($\lambda$). *Roadmap: Expand parser to accept a list of $Z_i$ matrices and $n$-dimensional optimization.*
3. **AI Matrix Exact Inverses:** The AI-REML step relies on an algebraic shortcut ($Z^T P y$) valid only for independent random effects ($G = \sigma^2_u I$). To support Genomic Relationship Matrices (A or G matrices), this approximation fails. *Roadmap: Integrate the Takahashi equations (Sparse Inverse Subset algorithm) into the `SimplicialLDLT` factor to extract the exact trace of $C^{-1}$.*
4. **Boundary Constraint Bouncing:** The AI update utilizes a hard boundary constraint (`1e-5`). If true variances are zero, the Newton-Raphson step may bounce indefinitely against the boundary. *Roadmap: Implement iterative step-halving for negative updates.*

## 📄 Dependencies

* R (>= 4.0.0)
* `Rcpp`
* `RcppEigen`
* `Matrix`
* A C++20 compliant compiler (e.g., Apple clang >= 17.0)