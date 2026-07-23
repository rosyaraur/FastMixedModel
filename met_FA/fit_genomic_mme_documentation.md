#  Single-Step Genomic MET Solver Documentation

This document outlines the architecture, mathematical framework, and usage of the custom C++-backed R solver for Multi-Environment Trial (MET) genomic analysis. This tool efficiently estimates marker-based genetic values across environments using the Expectation-Maximization (EM) algorithm.

---

## 1. Overview

The **Independent Single-Step Genomic MET Solver** is designed to bypass standard package memory constraints by utilizing `RcppArmadillo` for computationally heavy matrix operations. It is specifically tailored for handling singular systems (using the Moore-Penrose pseudo-inverse) and incorporates a user-provided Genomic Relationship Matrix ($G$) to estimate genomic BLUPs (GBLUPs) across multiple environments.

### Key Features

* **Native C++ EM Loop**: Iteratively solves the Mixed Model Equations natively in C++.
* **Singularity Handling**: Utilizes `arma::pinv()` to gracefully solve rank-deficient systems without failing.
* **Kronecker Covariance**: Models the random effect covariance as $G_{env} \otimes G_{genomic}$.
* **`sommer`-Compatible Output**: Structures the output list so that GBLUP extraction mimics standard genomic R packages.

---

## 2. Mathematical Framework

The solver estimates fixed effects ($b$) and random genomic effects ($u$) by solving the Mixed Model Equations (MME):

$$\begin{bmatrix} X'X & X'Z \\ Z'X & Z'Z + (G_{env} \otimes G_{genomic})^{-1} \sigma^2_e \end{bmatrix} \begin{bmatrix} \hat{b} \\ \hat{u} \end{bmatrix} = \begin{bmatrix} X'y \\ Z'y \end{bmatrix}$$

Where:

* $X$ and $Z$ are the design matrices for fixed and random effects, respectively.
* $G_{env}$ is the unstructured covariance matrix between environments.
* $G_{genomic}$ is the additive genomic relationship matrix.
* $\sigma^2_e$ is the residual variance.

During each EM iteration, the environmental covariance matrix ($G_{env}$) and residual variance ($\sigma^2_e$) are updated based on the current estimates of $\hat{u}$ and the pseudo-inverse of the Left-Hand Side (LHS) matrix.

---

## 3. Function Reference

### 3.1. C++ Core: `solve_genomic_mme_cpp`

The high-performance engine executing the EM algorithm.

**Arguments:**

* `X` *(arma::mat)*: Fixed effects design matrix.
* `Z` *(arma::mat)*: Random effects design matrix (Strictly ordered as Environment $\times$ Genotype).
* `y` *(arma::vec)*: Phenotypic response vector.
* `G_genomic` *(arma::mat)*: The additive Genomic Relationship Matrix.
* `n_envs` *(int)*: Number of environments.
* `max_iter` *(int)*: Maximum number of EM iterations (default: 100).
* `tol` *(double)*: Convergence tolerance (default: 1e-5).

**Returns:** A list containing `BLUEs`, `GBLUPs`, the estimated `G_env` matrix, and `var_e`.

### 3.2. R Wrapper: `fit_genomic_mme`

Parses standard R formulas, handles matrix alignment, and dispatches to the C++ engine.

**Arguments:**

* `fixed` *(formula)*: Fixed effects formula (e.g., `Yield ~ Environment`).
* `random` *(formula)*: Random effects formula (e.g., `~ 0 + Env_Geno`).
* `data` *(data.frame)*: The dataset containing the phenotype and factors.
* `G_mat` *(matrix)*: The pre-computed square $G$ matrix with rownames matching the genotypes.
* `n_envs` *(integer)*: The number of unique environments in the dataset.

---

## 4. Usage Pipeline

### Step 1: Prepare the Genomic Matrix ($G$)

Ensure your marker data is centered and the $G$ matrix is calculated and scaled appropriately. The row names of the $G$ matrix **must** exactly match the genotype identifiers used in your phenotype dataset.

```r
# Example Calculation
M_centered <- scale(M, center = TRUE, scale = FALSE)
scaling_factor <- 2 * mean(colMeans(M / 2) * (1 - colMeans(M / 2)))
G_matrix <- (M_centered %*% t(M_centered)) / scaling_factor

```

### Step 2: Format the Phenotypic Data

Create a composite factor combining the environment and genotype identifiers. The formula expects this exact format to construct the $Z$ matrix properly.

```r
met_data$Env_Geno <- factor(paste(met_data$Environment, met_data$Genotype, sep = ":"))

```

### Step 3: Fit the Model

Pass the formulas, dataset, and $G$ matrix to the wrapper function.

```r
g_met_model <- fit_genomic_mme(
  fixed = Yield ~ Environment,
  random = ~ 0 + Env_Geno,
  data = met_data,
  G_mat = G_matrix,
  n_envs = 4
)

```

### Step 4: Extract the GBLUPs

The solver automatically formats the output list to match the extraction syntax used by packages like `sommer`.

```r
# Extract Estimated Genetic Values
gblups <- g_met_model$U$`rr(Environment, d = 2):Genotype`$Yield

# View the results
head(gblups)

```