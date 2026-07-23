# Methodological Documentation: Sparse Mixed Model Engine

## 1. Overview

This software provides a custom R and C++ pipeline for solving linear mixed-effects models. It bridges R's user-friendly formula interface with a high-performance C++ backend powered by `RcppEigen`. The engine implements two distinct mathematical pathways for maximizing the Restricted Maximum Likelihood (REML):

1. **Profiled REML:** A derivative-free approach scaling the variance ratio (mimicking `lme4`).
2. **Average Information REML (AI-REML):** A Newton-Raphson derivative approach (mimicking ASReml).

---

## 2. The C++ Computation Engine (`lmm_engine.cpp`)

The backend is built entirely on Eigen's sparse matrix algebra, which is crucial for memory efficiency when scaling up to high-dimensional datasets with thousands of random effects levels.

### Step 2.1: Constructing Henderson's Mixed Model Equations (MMEs)

The helper function `build_mme_lhs` constructs the sparse Left-Hand Side (LHS) matrix, $C$, from the dense fixed-effect matrix $X$ and the sparse random-effect matrix $Z$.

$$C = \begin{bmatrix} X^T X & X^T Z \\ Z^T X & Z^T Z + \lambda I \end{bmatrix}$$

* **Methodology:** To avoid illegal dense-to-sparse implicit conversions (which trigger compile-time errors in Eigen), $X^T Z$ is evaluated as a dense matrix. Non-zero elements are then explicitly extracted into a `std::vector<Triplet<double>>` to securely populate the final sparse block matrix $C$.

### Step 2.2: Profiled REML Evaluator (`eval_profiled_reml_cpp`)

This function evaluates the log-likelihood for a proposed variance ratio $\lambda = \sigma^2_e / \sigma^2_u$.

* **Methodology:**
1. It performs a sparse Cholesky factorization ($C = L D L^T$) via Eigen's `SimplicialLDLT`.
2. It solves the MME for $\hat{\beta}$ and $\hat{u}$.
3. It extracts the log-determinant of $C$ directly from the diagonal matrix $D$ generated during factorization, completely avoiding explicit matrix inversion.
4. It returns the Penalized Residual Sum of Squares (PRSS) and the REML deviance back to R.



### Step 2.3: AI-REML Step (`step_aireml_cpp`)

This function performs a single Newton-Raphson update of the variance components $\theta = [\sigma^2_e, \sigma^2_u]^T$.

* **Methodology:**
1. Solves the MME for the current $\theta$.
2. Calculates the working residuals $e = y - X\hat{\beta} - Z\hat{u}$ and the projection vector $Py = e / \sigma^2_e$.
3. Computes the Average Information matrix $AI$ using algebraic projections (e.g., $AI_{ee} = \frac{1}{2} Py^T Py / \sigma^2_e$) based on Gilmour et al. (1995).
4. Computes the score equations (first derivatives) and applies the update: $\theta_{new} = \theta_{old} + AI^{-1} \times Score$.
5. Applies a strict boundary constraint ($\ge 1\text{e-}5$) to prevent negative variance estimates.



---

## 3. The R Wrapper (`fit_mixed_model`)

The R function acts as the interface, memory allocator, and optimization director.

### Step 3.1: Formula Parsing and Matrix Allocation

* **Methodology:** The wrapper leverages R's native `model.matrix` to generate the dense $X$ matrix for fixed effects. For the random effects, it intercepts the formula and forces the use of `Matrix::sparse.model.matrix`. This ensures that $Z$, which is mostly zeros, is passed to C++ as an efficient `dgCMatrix`.

### Step 3.2: Algorithm Dispatch

* **Profiled Method:** Uses R's native `optimize()` function. It performs a 1D derivative-free search over the bounds of $\log(\lambda)$, repeatedly calling the C++ profiled evaluator until the minimum deviance is found.
* **AI-REML Method:** Initializes variance components to $0.5 \times \text{var}(y)$ and triggers a `while` loop. It repeatedly calls the C++ AI-REML step function, overwriting the variance components until the $L_2$ norm of the parameter change vector falls below the tolerance threshold ($1\text{e-}5$).

---

## 4. Simulated Test Case Documentation

The simulation serves as a unit test to verify that both independent algorithms converge on the same maximum likelihood surface.

### Step 4.1: Data Generation

* **Methodology:** A dataset of $N=2,000$ is simulated across 50 discrete groups. The ground truth parameters are hardcoded:
* $\beta_0 = 3.5, \beta_1 = 1.8$
* $\sigma^2_u = 1.25$ (Group variance)
* $\sigma^2_e = 0.75$ (Residual variance)


* The response $y$ is generated via $y = X\beta + Zu + e$.

### Step 4.2: Execution and Benchmarking

* **Methodology:** Both algorithms are executed sequentially on the same simulated dataset. Their variance component outputs and fixed effect estimates are compared up to 5 decimal places to ensure mathematical parity.

---

## 5. Potential Bugs & Scaling Limitations

While mathematically sound for the test case, this foundational engine has specific limitations that must be addressed before deployment into complex spatial or quantitative genetic pipelines.

### 5.1. Single Random Effect Limitation (Architecture)

* **Bug/Limitation:** The current formula parser and C++ backend strictly assume a single random effect vector ($q$ columns, single $\lambda$).
* **Fix:** To support multiple overlapping random effects (e.g., `~ Block + Genotype`), the engine must be refactored to accept a list of $Z_i$ matrices and a vector of $\lambda_i$ parameters. The MME builder must dynamically expand the $Z^T Z$ block to include off-diagonal $Z_i^T Z_j$ blocks.

### 5.2. AI Matrix Approximation (Mathematical)

* **Bug/Limitation:** True AI-REML requires calculating the trace of the $C$ inverse for the exact score equations. Because exact sparse inversion is computationally prohibitive, the current C++ code relies on an algebraic shortcut using $Z^T P y$. This works perfectly for simple variance components but will cause the Newton-Raphson iterations to fail or oscillate if a complex covariance structure (like an Identity By Descent kinship matrix, $G$) is introduced.
* **Fix:** Implement a "Sparse Inverse Subset" algorithm (Takahashi equations) operating on the `SimplicialLDLT` factor $L$ to extract the exact trace elements of $C^{-1}$ required for the rigorous score equations.

### 5.3. Dense Intermediate Matrix Memory Spike (Memory)

* **Bug/Limitation:** In `build_mme_lhs`, the line `MatrixXd XtZ = X.transpose() * Z;` correctly forces a dense evaluation to satisfy Eigen's type checker. However, if $X$ contains many fixed effects (e.g., multi-environment trial markers) and $Z$ has millions of columns, this temporary dense matrix will trigger a massive RAM spike and potential out-of-memory (OOM) crash before the triplets are even built.
* **Fix:** Iterate through the sparse non-zero elements of $Z$ column-by-column, computing the dot products with the columns of $X$ manually, and pushing directly to the triplet list to maintain a strictly $O(N_{nonzero})$ memory footprint.

### 5.4. Positivity Constraint Boundary (Convergence)

* **Bug/Limitation:** The AI-REML step enforces `theta_new(i) = 1e-5` if the update step pushes a variance component below zero. If the true variance is virtually zero, the Newton-Raphson algorithm will continuously "slam" against this boundary, causing the `step$change` delta to remain above the convergence tolerance, resulting in an infinite loop (until `max_iter` is reached).
* **Fix:** Implement a step-halving routine. If an AI update pushes a component negative, halve the update step iteratively until the new value is strictly positive, rather than artificially pinning it to a boundary constant.

# Function-by-function documentation

### 1. `build_mme_lhs` (C++ Helper Function)

**Inputs:**

* `X` (`const MatrixXd&`): The dense model matrix for fixed effects (dimensions $N \times p$).
* `Z` (`const SpMat&`): The sparse model matrix for random effects (dimensions $N \times q$).
* `lambda` (`double`): The current variance ratio ($\sigma^2_e / \sigma^2_u$).

**Outputs:**

* **Returns:** `SpMat` (Sparse Matrix). The fully assembled, sparse Left-Hand Side coefficient matrix $C$ of dimensions $(p+q) \times (p+q)$.

**Methodology:**
This function dynamically constructs the sparse coefficient matrix, $C$, of Henderson's Mixed Model Equations. It utilizes Eigen's `Triplet` insertion method, which is the most efficient way to build large sparse matrices in C++. It maps the cross-products ($X^T X$, $X^T Z$, $Z^T X$, and $Z^T Z + \lambda I$) into a unified block structure.

**Watchouts & Potential Issues:**

* **Dense Intermediate Memory Spike:** The operation `MatrixXd XtZ = X.transpose() * Z;` forces Eigen to evaluate the cross-product as a dense matrix to bypass strict type-conversion errors. If $X$ contains a high number of fixed effects and $Z$ has tens of thousands of columns, this temporary matrix will cause a massive memory spike and potential Out-Of-Memory (OOM) crash before the triplets are even generated. To scale this efficiently, the dense multiplication must be rewritten to iterate only over the non-zero elements of $Z$.

---

### 2. `eval_profiled_reml_cpp` (C++ Engine)

**Inputs:**

* `X` (`const Map<MatrixXd>`): Pointer to the dense fixed-effect matrix from R.
* `Z` (`const MappedSparseMatrix<double>`): Pointer to the sparse random-effect matrix from R.
* `y` (`const Map<VectorXd>`): Pointer to the dense response vector from R.
* `lambda` (`double`): The proposed variance ratio to evaluate.

**Outputs:**

* **Returns:** `Rcpp::List` containing:
* `deviance` (`double`): The evaluated REML deviance ($-2 \times \text{logLik}$).
* `sigma2_e` (`double`): The conditionally optimal residual variance.
* `sigma2_u` (`double`): The conditionally optimal random effect variance.
* `beta` (`VectorXd`): The solved fixed-effect coefficients.
* `u` (`VectorXd`): The solved random-effect BLUPs.



**Methodology:**
This function executes a single evaluation of the profiled REML deviance. It solves the MME using Eigen's `SimplicialLDLT` (sparse Cholesky factorization, $C = LDL^T$). Because the 1D optimization only requires the log-likelihood, it calculates the log-determinant of $C$ directly from the trace of the diagonal matrix $D$ produced during factorization, completely bypassing exact matrix inversion.

**Watchouts & Potential Issues:**

* **Rank Deficiency Failures:** If the experimental design is highly unbalanced or contains collinear fixed effects, the $C$ matrix will not be positive definite. In these cases, `solver.info() != Success` triggers an early exit, returning a massive deviance penalty (`1e10`). While this keeps the optimizer moving, frequent rank deficiencies will break the 1D search space.

---

### 3. `step_aireml_cpp` (C++ Engine)

**Inputs:**

* `X` (`const Map<MatrixXd>`): Pointer to the dense fixed-effect matrix.
* `Z` (`const MappedSparseMatrix<double>`): Pointer to the sparse random-effect matrix.
* `y` (`const Map<VectorXd>`): Pointer to the dense response vector.
* `sigma2_e` (`double`): The current estimate of residual variance.
* `sigma2_u` (`double`): The current estimate of random effect variance.

**Outputs:**

* **Returns:** `Rcpp::List` containing:
* `sigma2_e` (`double`): The updated residual variance.
* `sigma2_u` (`double`): The updated random effect variance.
* `beta` (`VectorXd`): Current fixed-effect estimates.
* `u` (`VectorXd`): Current random-effect estimates.
* `change` (`double`): The $L_2$ norm of the parameter update vector (used to check convergence).



**Methodology:**
Executes one Newton-Raphson update step using the Average Information (AI) algorithm. It solves the MME, calculates the projection vector $P y$ using the working residuals, and computes the AI matrix based on algebraic approximations. It then solves the score equations to propose the next parameter values and applies a strict boundary constraint to prevent negative variances.

**Watchouts & Potential Issues:**

* **Covariance Structure Assumptions:** The current AI score equations rely on an algebraic simplification that is only valid when random effects are independent ($G = \sigma^2_u I$). If you extend this to handle genomic relationship matrices (IBD matrices) or spatial autoregressive corrections, this approximation fails. You will need to implement a Sparse Inverse Subset algorithm (Takahashi equations) to extract the exact trace of $C^{-1}$.
* **Boundary Traps:** The function currently forces a hard boundary constraint (`if (theta_new <= 1e-5)`). If the true variance is near zero, the Newton step will aggressively overshoot into negative territory, hit the boundary, and bounce back indefinitely. A robust implementation requires iterative step-halving.

---

### 4. `fit_mixed_model` (R Wrapper)

**Inputs:**

* `fixed` (`formula`): The standard R formula for fixed effects (e.g., `y ~ x1 + x2`).
* `random` (`formula`): A one-sided formula for random effects (e.g., `~ group`).
* `data` (`data.frame`): The dataset containing the variables.
* `algorithm` (`character`): User selection between `"aireml"` and `"profiled"`.
* `max_iter` (`numeric`, default = 30): Maximum number of iterations for the AI-REML loop.
* `tol` (`numeric`, default = 1e-5): Convergence threshold for the AI parameter change delta.

**Outputs:**

* **Returns:** An S3 object of class `custom_lmm` containing:
* `fixed_effects` (`numeric vector`): Named vector of $\hat{\beta}$ estimates.
* `random_effects` (`numeric vector`): Named vector of $\hat{u}$ BLUPs.
* `vcov` (`numeric vector`): Final variance components ($\sigma^2_u$ and $\sigma^2_e$).
* `algorithm` (`character`): The algorithm used to fit the model.
* `iterations` (`numeric`): Number of solver loops or optimization evaluations.
* `runtime_sec` (`numeric`): Total elapsed execution time.



**Methodology:**
Acts as the user interface and memory manager. It utilizes R's native `model.matrix` for dense matrices and `Matrix::sparse.model.matrix` to ensure the $Z$ matrix is passed to C++ as an efficient `dgCMatrix`. It then routes execution to either `optimize()` for a 1D derivative-free search (Profiled method) or a custom `while` loop (AI-REML method) until convergence is reached.

**Watchouts & Potential Issues:**

* **Single Random Effect Lock:** The formula parser `as.character(random)[2]` strictly assumes a single random effect term. Attempting to pass a model with multiple variance components (e.g., `~ Genotype + Block`) will cause the parser to break.
* **Optimizer Limitations:** Because it is locked to a single random effect, the profiled method relies on `optimize()`, which only performs 1-dimensional searches. Transitioning to a multi-component model requires swapping `optimize()` for a multivariate solver like `optim` (L-BFGS-B) to handle multidimensional optimization over several $\lambda_i$ parameters.
