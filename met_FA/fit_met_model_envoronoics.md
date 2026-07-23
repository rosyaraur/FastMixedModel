# Unified Multi-Environment Trial (MET) Solver Documentation : Envirotyping data added 

## 1. Overview

This toolkit provides a high-performance, custom-built Mixed Model Equation (MME) solver for Multi-Environment Trials (MET). It utilizes a C++ backend (`RcppArmadillo`) for computationally heavy matrix operations and an R frontend for formula processing and data routing.

The pipeline supports two primary modeling strategies:

1. **Standard MET Modeling:** Estimates an unstructured environmental covariance matrix ($G_{env}$) using either Expectation-Maximization (EM) or Newton-Raphson (NR) algorithms.
2. **Envirotypic MET Modeling:** Replaces the estimation of an unstructured covariance matrix with a known Environmental Relationship Matrix ($E$) derived from envirotyping data (e.g., weather, soil), vastly improving predictions for unobserved environments.

---

## 2. Mathematical Foundations

### Standard MET Model

In the standard approach without envirotypic data, the model must estimate the covariance between environments from the phenotypic data itself. The random Genotype $\times$ Environment (GxE) effects ($u$) are modeled as:


$$Var(u) = G_{env} \otimes G$$


Where $G_{env}$ is the unstructured $k \times k$ covariance matrix for $k$ environments, and $G$ is the genomic relationship matrix.

### Envirotypic Model (Reaction Norm)

When environmental covariates are available, we compute a known Environmental Relationship Matrix ($E$). The complex $G_{env}$ estimation is reduced to estimating a single scalar variance component for the GxE interaction ($\sigma^2_{ge}$):


$$Var(u) = \sigma^2_{ge} (E \otimes G)$$

### Cross-Environment Projection (CV0)

Using the Envirotypic model, GBLUPs for an untested environment ($u_{test}$) can be mathematically projected from the tested environments ($u_{train}$) using the covariances in the $E$ matrix:


$$u_{test} = u_{train} E_{train}^{-1} E_{test, train}$$

---

## 3. Core Functions

### Master Wrapper: `fit_met_model()`

This is the primary user-facing function in R. It parses the model formulas, dynamically checks the data environment, constructs the design matrices ($X$ and $Z$), and routes the data to the appropriate C++ engine.

**Arguments:**

* `fixed`: A two-sided formula for fixed effects (e.g., `Yield ~ Environment`).
* `random`: A one-sided formula for random GxE effects (e.g., `~ 0 + Env_Geno`).
* `data`: The primary phenotypic `data.frame`.
* `G_mat`: The scaled Genomic Relationship Matrix ($G$). Rownames must match genotype IDs.
* `E_mat`: (Optional) The Environmental Relationship Matrix ($E$). If provided, the solver automatically routes to the Envirotypic Engine.
* `n_envs`: (Optional) Integer. Required only if `E_mat` is `NULL`. Represents the number of environments.
* `method`: Character string (`"EM"` or `"NR"`). Defines the optimization algorithm used in the standard model. Ignored if `E_mat` is provided.

**Returns:**
A list containing:

* `Type`: The model routing type used.
* `BLUEs`: Best Linear Unbiased Estimates for fixed effects.
* `GBLUPs`: Best Linear Unbiased Predictors for the random effects.
* `Var_Residual`: Estimated residual variance ($\sigma^2_{e}$).
* `G_env` / `Var_GxE`: Estimated environmental covariance matrix (Standard) or single scalar variance (Envirotypic).

---

## 4. Data Preparation Requirements

To ensure successful matrix inversion in C++, your data must be strictly aligned:

1. **Matrix Naming:** Both `G_mat` and `E_mat` must have proper `rownames` corresponding to the Genotype and Environment IDs found in your phenotypic dataset.
2. **Factor Concatenation:** The random effect column in your dataset (e.g., `Env_Geno`) must be formatted as `Environment:Genotype`. The R wrapper utilizes this exact string matching to correctly order the columns of the $Z$ matrix against the Kronecker product $E \otimes G$.
3. **Scaling:** Relationship matrices must be properly scaled so their diagonals average roughly 1.

---

## 5. Troubleshooting & Common Errors

### Error: `pinv(): svd failed`

**Cause:** This error originates from the C++ `RcppArmadillo` engine. It indicates that the Singular Value Decomposition (SVD) algorithm failed, which happens almost exclusively when `NA` or `NaN` values are passed into a matrix (usually the $Z$ matrix).
**Solution:**

* Ensure that the levels in your random effect factor (`Env_Geno`) exactly match the combined `rownames` of your $E$ and $G$ matrices.
* If performing subsetting or cross-validation (like Leave-One-Environment-Out), ensure you are using the updated `fit_met_model()` function that dynamically recalculates the exact environment levels present in the subsetted data to avoid generating empty columns.

---

Would you like to expand this documentation by adding a specific section detailing the differences in convergence behavior and computational memory load between the Expectation-Maximization (EM) and Newton-Raphson (NR) algorithms?