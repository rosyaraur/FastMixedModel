# Advanced Modeling in R

**A Comprehensive Guide to BGLR, Animal Models, and Hierarchical Data across Disciplines**

---

## 1. Introduction to BGLR

The `BGLR` (Bayesian Generalized Linear Regression) R-package is a powerful framework originally designed for genomic prediction and whole-genome regression. It is particularly well-suited for high-dimensional datasets where the number of predictors vastly exceeds the sample size ($p \gg n$).

While BGLR relies on Bayesian shrinkage priors (such as BayesA, BayesB, and the Bayesian Lasso) for marker data, its inclusion of the **Reproducing Kernel Hilbert Spaces (RKHS)** and **Bayesian Ridge Regression (BRR)** allows it to act as a highly flexible mixed-model solver for both genomic and standard hierarchical applications.

---

## 2. Pedigree & Genomic Animal Models

In quantitative genetics, standard hierarchical models are insufficient because individuals are not independent; their relatedness must be modeled using a covariance structure, typically a Numerator Relationship Matrix ($A$) or a Genomic Relationship Matrix ($G$).

### Comparing BGLR, sommer, and MCMCglmm

When fitting these models, researchers typically choose between Frequentist and Bayesian frameworks. Below is a summary of how the major packages handle custom covariance matrices.

| Package | Framework | Best Use Case | Key Implementation Details |
| --- | --- | --- | --- |
| **BGLR** | Bayesian (Gibbs) | Genomic prediction, variable marker selection.

 | Uses `model = "RKHS"` to process relationship matrices.

 |
| **sommer** | Frequentist (REML) | Rapid prediction in large commercial datasets; multivariate traits.

 | Uses `vsr(id, Gu = A)` to directly inject the matrix.

 |
| **MCMCglmm** | Bayesian (MCMC) | Complex error structures, non-Gaussian traits, evolutionary ecology.

 | Requires the **inverse** matrix in a sparse format (`dgCMatrix`).

 |

### The Sparse Matrix Requirement in MCMCglmm

A common pitfall when transitioning from `sommer` or `BGLR` to `MCMCglmm` is the handling of the inverse relationship matrix ($A^{-1}$). Standard base R matrices will trigger an `Error in x@Dim`. The matrix must be coerced using the `Matrix` package.

```R
library(Matrix)
library(MCMCglmm)

# Correctly preparing the sparse inverse matrix for MCMCglmm
A_inv_base <- solve(A)
A_inv_sparse <- as(A_inv_base, "dgCMatrix")
rownames(A_inv_sparse) <- rownames(A)
colnames(A_inv_sparse) <- colnames(A)

fit_mcmc <- MCMCglmm(fixed = y ~ 1, 
                     random = ~ id, 
                     ginverse = list(id = A_inv_sparse), 
                     data = pheno_df, prior = prior_mcmc, verbose = FALSE)

```

---

## 3. Standard Hierarchical Models

Hierarchical (or Multi-Level) models are utilized when data is grouped or nested (e.g., students within classrooms, repeated measures on a single patient). Unlike animal models, the groups are assumed to be independent of one another.

### BGLR vs. lme4

For non-genomic hierarchical modeling, `lme4` is universally preferred due to its formula syntax and rapid Maximum Likelihood execution. To replicate standard mixed models in BGLR, one must manually build incidence matrices ($X$ for fixed, $Z$ for random) and stack them in an `ETA` list using Bayesian Ridge Regression (`model = "BRR"`).

> **Key Distinction: Random Slopes**
> When specifying random slopes and intercepts (e.g., `(Days | Subject)`), `lme4` automatically estimates the correlation between the slope and the intercept. In BGLR, constructing separate $Z$ matrices for intercepts and slopes strictly assumes they are independent (covariance = 0) unless modeled through complex multi-trait frameworks.
> 
> 

---

## 4. Cross-Disciplinary Use Cases

Mixed models extend far beyond animal breeding. Here is how hierarchical modeling is applied across various scientific fields.

### Healthcare: Longitudinal Data

In clinical trials, taking multiple measurements from the same patient over time violates independence. Models utilize a **Random Intercept** to account for baseline patient differences and a **Random Slope** to capture how differently individuals respond to treatment over time.

```R
# lme4 syntax for clinical trials
# Reaction ~ Days + Random Intercept & Slope per Subject
fit_clinical <- lmer(Reaction ~ Days + (Days | Subject), data = sleepstudy)

```

### Epidemiology: Generalized Mixed Models (GLMMs)

When outcomes are binary (disease presence) or proportional, standard linear assumptions fail. GLMMs apply a link function (like Logit or Probit) while still estimating herd or regional variance.

```R
# lme4 syntax for proportional disease incidence
fit_epi <- glmer(cbind(incidence, size - incidence) ~ period + (1 | herd), 
                 family = binomial(link = "probit"), data = cbpp)

```

*Note:* BGLR handles this natively using `response_type = "ordinal"`, provided the data is unrolled into individual binary records.

### Education & Social Sciences: Deep Nesting

Educational data is notoriously nested: students test scores are influenced by their specific classroom, which is influenced by their overarching school. We account for this partitioned variance using deep nesting syntax.

```R
# lme4 syntax for deep nesting (Classrooms within Schools)
fit_edu <- lmer(math_score ~ curriculum + student_ses + 
                (1 | school_id/class_id), data = df_edu)

# BGLR equivalent mapping requires two independent BRR matrices:
ETA_edu <- list(
  list(X = X_fixed, model = "FIXED"),
  list(X = Z_school, model = "BRR"),
  list(X = Z_class, model = "BRR")
)

```

---

## 5. Conclusion and Package Selection

Selecting the right R package for mixed modeling comes down to the specific data structure and the desired inference framework:

* **Use `lme4**` for standard hierarchical models, longitudinal data, and cross-disciplinary grouped statistics where no covariance matrices are required.


* **Use `sommer**` for rapid evaluation of animal/plant models with pedigree or genomic relationship matrices, especially in commercial breeding.


* **Use `MCMCglmm**` for complex ecological models, highly non-Gaussian distributions, or explicit multi-trait Bayesian variance estimation (ensuring sparse matrices are used for inverses).


* **Use `BGLR**` when performing genomic prediction using raw marker data, when applying specific shrinkage priors to variables, or when integrating multiple "omics" layers into a unified Bayesian framework.