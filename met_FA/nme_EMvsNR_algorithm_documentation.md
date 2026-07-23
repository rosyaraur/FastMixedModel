# Documentation: EM vs. Newton-Raphson Algorithm Comparison for Genomic MET Solvers

### 1. Overview

This documentation accompanies the **Independent Single-Step Genomic MET Solver**, which features a dual-algorithm C++ backend. The script allows users to directly compare two fundamental approaches to solving Restricted Maximum Likelihood (REML) equations for Multi-Environment Trials (MET): the **Expectation-Maximization (EM)** algorithm and the **Newton-Raphson (NR)** algorithm (the latter being analogous to the methodology used by the `sommer` R package).

This comparison is vital for understanding the trade-offs between computational stability, memory footprint, and convergence speed when processing large-scale genomic datasets.

---

### 2. Algorithmic Frameworks

#### 2.1. Expectation-Maximization (EM)

The EM algorithm iteratively updates variance components by treating the random effects as missing data. It relies on Henderson's Mixed Model Equations (MME) to implicitly solve for the variance structures.

* **Mathematical Approach:** Constructs the Left-Hand Side (LHS) matrix:

$$LHS = \begin{bmatrix} X'X & X'Z \\ Z'X & Z'Z + (G_{env} \otimes G_{genomic})^{-1} \sigma^2_e \end{bmatrix}$$


* **Characteristics:**
* **Convergence:** Linear (requires a high number of iterations).
* **Stability:** Highly stable; practically guarantees that variance components remain strictly positive within the parameter space.
* **Memory Efficiency:** Excellent. Because it utilizes the sparse nature of the $Z$ matrix and the inverse of the relationship matrices within the MME, it avoids computing the massive $N \times N$ marginal phenotypic variance matrix.



#### 2.2. Newton-Raphson / Fisher Scoring (NR)

The NR algorithm (often implemented as Average Information or Fisher Scoring in packages like `sommer`) uses the first derivatives (Score) and second derivatives (Information) of the restricted log-likelihood to find the optimal variance components.

* **Mathematical Approach:** Explicitly calculates the marginal phenotypic variance matrix ($V$) and the projection matrix ($P$) via Direct Inversion:

$$V = Z (G_{env} \otimes G_{genomic}) Z' + I \sigma^2_e$$


$$P = V^{-1} - V^{-1}X(X'V^{-1}X)^{-1}X'V^{-1}$$



Residual variance ($\sigma^2_e$) is updated using the Score ($\mathcal{S}$) and Information ($\mathcal{I}$):

$$\sigma^2_{new} = \sigma^2_{old} + \mathcal{I}^{-1} \mathcal{S}$$


* **Characteristics:**
* **Convergence:** Quadratic (reaches the maximum likelihood in significantly fewer iterations than EM).
* **Speed:** Extremely fast per convergence, provided the matrix inversions can be handled efficiently.
* **Memory/Computational Cost:** High. It requires the direct inversion of the dense $N \times N$ matrix $V$, which scales cubically $O(N^3)$ and can lead to memory exhaustion (out-of-memory errors) on standard machines when processing large genomic datasets.



---

### 3. Executing the Comparison in R

The `fit_genomic_mme` wrapper function includes a `method` argument to seamlessly toggle between the two C++ engines.

#### Running EM

To execute the model using the stable Expectation-Maximization solver:

```r
model_em <- fit_genomic_mme(
  fixed = Yield ~ Environment,
  random = ~ 0 + Env_Geno,
  data = met_data,
  G_mat = G_matrix,
  n_envs = 4,
  method = "EM" # Triggers the MME-based loop
)

```

#### Running Newton-Raphson

To execute the model using the direct-inversion Newton-Raphson solver:

```r
model_nr <- fit_genomic_mme(
  fixed = Yield ~ Environment,
  random = ~ 0 + Env_Geno,
  data = met_data,
  G_mat = G_matrix,
  n_envs = 4,
  method = "NR" # Triggers the Projection Matrix loop
)

```

---

### 4. Interpreting the Output

When you run the provided performance comparison script, it will generate three distinct evaluation metrics:

* **Execution Time (`system.time`):**
Compare the `elapsed` time. For smaller datasets (like the simulated 100 genotypes), NR will typically finish faster due to requiring fewer iterations. As $N$ grows, the EM algorithm may become the only viable option if the $V$ matrix exceeds system RAM.
* **Variance Estimates Comparison:**
A table contrasting the final estimated Residual Variance ($\sigma^2_e$) and the Average Environmental Variance from both models. Due to differing likelihood maximization pathways, small numerical differences may exist, but they should converge toward the same general magnitude.
* **Top 5 GBLUPs Comparison:**
A side-by-side dataframe of the extracted Genomic BLUPs for specific Genotype-Environment interactions. This confirms that despite differing internal mathematics, both the EM and NR engines accurately predict the same top-performing genetic lines.