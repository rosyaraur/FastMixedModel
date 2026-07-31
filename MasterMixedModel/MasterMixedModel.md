# Unified Multi-Engine Mixed-Effects and Animal Model Architecture

## Methodological Documentation & Implementation Guide

---

## 1. Executive Summary & Architecture Overview

Fitting linear mixed-effects models (LMMs) and quantitative genetic animal models across different statistical paradigms (**Frequentist** vs. **Bayesian**) and software ecosystems (`lme4`, `sommer`, `MCMCglmm`, `BGLR`, etc.) typically requires writing distinct, engine-specific syntax, managing alternative matrix structures (e.g., covariance matrices $A$ vs. inverse matrices $A^{-1}$), and parsing heterogeneous output objects.

The **Unified Multi-Engine Mixed-Effects Architecture** solves this interoperability challenge by introducing a two-tier abstraction wrapper:

1. **`MasterMixedSolver`**: A smart routing and execution engine that automates input diagnostics, paradigm mapping, and matrix transformation (such as sparse-to-dense conversion and automated matrix inversion).
2. **`ExtractMixedInfo`**: A universal standardization parser that homogenizes engine-specific output objects into a unified schema for fixed effects, variance components, and individual-level random effects (BLUPs / Estimated Breeding Values [EBVs]).

---

## 2. Core Module 1: `MasterMixedSolver`

The solver acts as a dispatch router that analyzes the model formula, dataset characteristics, and user preferences to execute the appropriate backend solver.

### Function Signature & Parameters

```R
MasterMixedSolver <- function(formula, data, paradigm = "Frequentist", engine = "auto", K_matrix = NULL, K_is_inverse = FALSE, nIter = 2000, burnIn = 500)

```

* **`formula`**: Standard R formula using lme4 syntax (e.g., `Reaction ~ Days + (1 | Subject)` or `tarsus ~ sex + (1 | animal)`).
* **`data`**: A data frame containing the phenotypic and grouping variables.
* **`paradigm`**: Character string specifying `"Frequentist"` or `"Bayesian"`.
* **`engine`**: Target computational backend (`"auto"`, `"lme4"`, `"blme"`, `"mbest"`, `"brms"`, `"sommer"`, `"MCMCglmm"`, `"BGLR"`, `"rrBLUP"`).
* **`K_matrix`**: Optional relationship matrix (Kinship or Pedigree Relationship Matrix $A$).
* **`K_is_inverse`**: Logical flag indicating whether `K_matrix` is supplied as an inverse ($A^{-1}$) or covariance ($A$) matrix.
* **`nIter` / `burnIn**`: MCMC chain parameters for Bayesian engines.

### Routing Logic & Automation Rules

| Condition / Input | Default Auto-Selected Engine | Paradigm | Key Implementation Detail |
| --- | --- | --- | --- |
| **No Kinship Matrix** | `lme4` | Frequentist | Standard REML estimation via `lmer()`. |
| **No Kinship Matrix** | `blme` | Bayesian | Maximum A Posteriori (MAP) estimation with default priors. |
| **Kinship Matrix Provided** | `sommer` | Frequentist | AI-REML estimation via `mmer()`, mapping `Gu` structure. |
| **Kinship Matrix Provided** | `BGLR` | Bayesian | Reproducing Kernel Hilbert Spaces (RKHS) Gibbs sampling. |

### Automated Matrix Inversion Layer

Different engines expect different matrix parameterizations for random genetic effects:

* **`MCMCglmm`**: Strictly requires the inverse pedigree relationship matrix ($A^{-1}$). If a covariance matrix is supplied, the solver automatically computes `Matrix::solve(K_matrix)`.
* **`sommer` & `BGLR**`: Require the raw covariance matrix ($A$). If an inverse matrix is supplied, the solver automatically un-inverts it and ensures dense matrix compliance with retained row/column names.

---

## 3. Core Module 2: `ExtractMixedInfo`

Because every R package returns model coefficients, variance components, and random effects in unique data structures (e.g., `lme4` returns 3D lists/S4 classes, `sommer` returns nested data frames, `MCMCglmm` returns MCMC chain matrices), `ExtractMixedInfo` acts as a universal adapter.

### Standardized Output Schema

Every fitted model object, regardless of the underlying engine, is coerced into a standardized R list format:

* **`Engine`**: Character tag identifying the source package.
* **`FixedEffects`**: Named numeric vector of population-level fixed coefficients.
* **`RandomEffects`**: Standardized data frame containing two explicit columns:
* `ID`: Character vector of group levels (e.g., Subject IDs or Animal pedigree IDs).
* `Value`: Numeric vector of conditional modes, BLUPs, or EBVs.


* **`VarianceComponents`**: Named list of extracted variance parameters (group variance and residual variance).

---

## 4. Backend Engine Support Matrix

| Engine | Paradigm | Primary Function | Kinship / Pedigree Support | Output Standardization Target |
| --- | --- | --- | --- | --- |
| **`lme4`** | Frequentist | `lme4::lmer()` | No | `lmerMod` S4 class |
| **`blme`** | Bayesian (MAP) | `blme::blmer()` | No | `bmerMod` S4 class |
| **`mbest`** | Frequentist | `mbest::mhglm()` | No | `mhglm` object |
| **`brms`** | Bayesian (HMC) | `brms::brm()` | No | `brmsfit` S4 class |
| **`sommer`** | Frequentist | `sommer::mmer()` | Yes (via `Gu`) | `mmer` S3 list |
| **`MCMCglmm`** | Bayesian (MCMC) | `MCMCglmm::MCMCglmm()` | Yes (via `ginverse`) | `MCMCglmm` S3 object |
| **`BGLR`** | Bayesian (Gibbs) | `BGLR::BGLR()` | Yes (via RKHS `ETA`) | `BGLR` list object |
| **`rrBLUP`** | Frequentist | `rrBLUP::mixed.solve()` | Yes (via `K`) | List object with `Vu` / `Ve` |

---

## 5. Implementation Workflow & Execution Blueprint

The architecture divides execution workflows into distinct operational tiers:

```
[ Raw Data & Pedigree ] 
        │
        ▼
[ MasterMixedSolver ] ──(Auto-Routing & Matrix Inversion)
        │
        ▼
[ Backend Engine Execution ] (lme4, sommer, MCMCglmm, BGLR, etc.)
        │
        ▼
[ ExtractMixedInfo ] ──(Standardization to ID + Value Schema)
        │
        ▼
[ Comparative Analytics ] (Fixed Effects, Variance Components, EBV Merging)

```

1. **Environment Initialization**: Load all target mixed-model libraries simultaneously.
2. **Model Dispatch**: Pass formulas and data into `MasterMixedSolver()`.
3. **Information Extraction**: Feed fitted objects into `ExtractMixedInfo()` to sanitize names, strip prefixes (e.g., converting sommer's `animal.Fem2` to clean identifiers like `Fem2`), and extract posterior means or BLUPs.
4. **Comparative Analysis**: Combine results across engines using standard data frame merging (`merge(..., by = "ID", all = TRUE)`) to evaluate parameter convergence across software implementations.

Here is the comprehensive **Methodological Addendum & Manual Correction Guide** to resolve the four remaining issues (`SexM` as `NA`, BGLR animal variance as `NA`, BGLR residual variance explosion > 2000, and `EBV_sommer` as `NA`).

---

### Root Causes & Manual Corrections

#### 1. BGLR Residual Variance Explosion (`> 2000`) & `SexM` as `NA`

* **The Cause**: BGLR automatically estimates a global intercept (`mu`) internally. In our previous script, if the design matrix `X_matrix` included an intercept column (or if formula parsing retained it alongside `mu`), it created **collinearity/rank deficiency** in the linear predictor. This causes the Gibbs sampler to fail at partitioning variance, leading to an inflated residual variance (`varE > 2000`) and dropping/corrupting fixed effect coefficients like `SexM`.
* **The Manual Correction**:
1. Strip the intercept explicitly from the fixed design matrix when passing it to BGLR using `model.matrix(~ sex - 1, data = data)`.
2. Explicitly map `colnames(X_matrix)` to BGLR's fixed effect coefficients (`term$b`) so that `SexM` preserves its proper name.



#### 2. BGLR Animal Variance as `NA`

* **The Cause**: BGLR stores the genetic variance under `term$varB`, but the extractor loop was looking for a generic list key instead of mapping the component name directly to `"animal"` or the active random term name.
* **The Manual Correction**: Force the BGLR variance component parser to explicitly assign the RKHS variance to `var_comp[["animal"]]` (matching the model's random effect group name).

#### 3. `EBV_sommer` as `NA`

* **The Cause**: Sommer prefixes the row names of its random effects matrix with varying syntax depending on the version (e.g., `animal.Fem2`, `u:animal.Fem2`, or `animal:Fem2`). A simple single-dot regex failed to strip these compound prefixes entirely.
* **The Manual Correction**: Use a robust regular expression (`gsub("^.*[:\\.]", "", u_ids)`) that strips any combination of colons, dots, and package prefixes to isolate the pure animal ID (`Fem2`).

---

