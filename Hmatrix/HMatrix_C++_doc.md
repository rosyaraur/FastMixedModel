# Single-Step Genomic Evaluation (ssGBLUP) & RKHS Simulation Pipeline

This document provides comprehensive documentation for the modular R/C++ pipeline designed to evaluate Single-Step Genomic Best Linear Unbiased Prediction (ssGBLUP) using Reproducing Kernel Hilbert Space (RKHS) models.

---

## 1. Methodology

### Single-Step GBLUP and the $H$ Matrix

Traditional genomic selection models require all individuals to have both phenotypic and genotypic data. Single-step GBLUP (ssGBLUP) elegantly circumvents this by combining the pedigree-based numerator relationship matrix ($A$) and the marker-based genomic relationship matrix ($G$) into a single, unified matrix ($H$).

The $H$ matrix propagates genomic information from genotyped individuals to ungenotyped relatives via pedigree linkages. It is partitioned by ungenotyped (subscript 1) and genotyped (subscript 2) individuals:

$$H = \begin{bmatrix} A_{11} + A_{12} A_{22}^{-1} (G_w - A_{22}) A_{22}^{-1} A_{21} & A_{12} A_{22}^{-1} G_w \\ G_w A_{22}^{-1} A_{21} & G_w \end{bmatrix}$$

To ensure positive-definiteness and invertibility, the $G$ matrix is blended with the pedigree matrix using a weighting parameter $w$ (typically 0.95):


$$G_w = wG + (1-w)A_{22}$$

### Reproducing Kernel Hilbert Space (RKHS) Modeling

Instead of fitting linear marker effects directly, the pipeline uses an RKHS framework. The model evaluates genetic merit by using a Kernel matrix ($K$)—which in our case is either the $A$ matrix or the $H$ matrix—as the covariance structure for a multivariate normal distribution of genetic effects.

### MCMC and Eigen-Decomposition (The C++ Engine)

Gibbs sampling for large multivariate mixed models typically requires inverting large covariance matrices at every iteration, creating massive computational bottlenecks. The custom C++ engine implements a mathematical shortcut:

1. **Eigen-Decomposition:** Before the MCMC loop begins, the kernel is decomposed: $K = UDU'$.
2. **Orthogonal Rotation:** The phenotype vector is rotated into an orthogonal space using the eigenvectors ($U$).
3. **Independent Sampling:** In this rotated space, the genetic effects are entirely uncorrelated. The complex multivariate sampling collapses into $n$ simple, independent univariate normal samples.
4. **Data Augmentation:** Missing phenotypes are treated as latent variables and sampled sequentially from their posterior predictive distribution during the MCMC steps.

---

## 2. Function Implementation

### `fit_rkhs_cpp` (Embedded C++ Engine)

A high-performance Gibbs sampler written in C++ using `RcppArmadillo`.

* **Purpose:** Fits the RKHS model using Eigen-decomposition and handles missing phenotypes via data augmentation.
* **Inputs:**
* `y` (arma::vec): Numeric vector of phenotypes (can contain `NA` / non-finite values).
* `K` (arma::mat): The Kernel matrix ($A$ or $H$).
* `nIter` (int): Total MCMC iterations.
* `burnIn` (int): Number of initial iterations discarded.


* **Outputs:** An R `List` containing the posterior mean of genetic effects (`u`) and the fitted values (`yHat`).

### Module 1: `build_H_matrix`

* **Purpose:** Constructs the single-step $H$ matrix from pedigree and genomic data.
* **Inputs:**

| Argument | Type | Description |
| --- | --- | --- |
| `A` | Matrix | Full pedigree-based relationship matrix. |
| `G` | Matrix | Genomic relationship matrix matching the `genotyped_ids`. |
| `genotyped_ids` | Character Vector | The exact IDs of the individuals that possess genomic data. |
| `w` | Numeric | Blending weight for $G$ and $A_{22}$ (Default: 0.95). |

* **Outputs:** A symmetric matrix ($H$) containing blended relationships, with rows and columns identically ordered to the input $A$ matrix.

### Module 2: `fit_rkhs_model`

* **Purpose:** A unified wrapper function that routes the modeling task to either the standard `BGLR` package or the custom `CPP` engine.
* **Inputs:**
* `y` (Numeric Vector): Phenotypes (includes `NA`s).
* `K` (Matrix): Kernel matrix.
* `engine` (Character): Set to `"BGLR"` or `"CPP"`.
* `nIter` / `burnIn` (Numeric): MCMC parameters.


* **Outputs:** A numeric vector of fitted values (`yHat`) matching the length of `y`.

### Module 3: `evaluate_masking_scenario`

* **Purpose:** Orchestrates a single iteration of the missing-data experiment. It masks a specific percentage of phenotypes (test set) and genotypes, builds the matrices, fits both the $A$ and $H$ models, and returns the prediction accuracy.
* **Inputs:**

| Argument | Type | Description |
| --- | --- | --- |
| `prop_mask_pheno` | Numeric | Proportion of total lines to mask phenotypes for (e.g., 0.20 = 20%). |
| `prop_mask_geno` | Numeric | Proportion of total lines treated as ungenotyped (e.g., 0.33 = 33%). |
| `engine` | Character | Backend for MCMC (`"BGLR"` or `"CPP"`). |
| `nIter` / `burnIn` | Numeric | MCMC configuration. |
| `seed` | Integer | Random seed to ensure consistent data masking across runs. |

* **Outputs:** A single-row `data.frame` recording the proportions used and the resulting Pearson correlations (`Accuracy_A`, `Accuracy_H`).

---

## 3. Prerequisites & System Requirements

To run this code, your environment must be equipped with the following:

1. **R Packages:**
* Statistical/MCMC: `BGLR`, `Rcpp`, `RcppArmadillo`
* Data Manipulation: `dplyr`, `tidyr`
* Visualization: `ggplot2`, `scales`


2. **C++ Compilation Toolchain:**
Because the code compiles C++ on the fly via `sourceCpp()`, your system requires a C++ compiler:
* **Windows:** Install [Rtools](https://cran.r-project.org/bin/windows/Rtools/). Ensure the Rtools version matches your R version.
* **macOS:** Install Xcode Command Line Tools (run `xcode-select --install` in terminal).
* **Linux:** Install `build-essential` or `gcc`/`g++` via your package manager.



---

## 4. Execution Workflow

### Step 1: Initialize the Environment

Run the section of the script labeled `1. LOAD LIBRARIES & COMPILE C++ ENGINE`. The `sourceCpp()` block will take 5-15 seconds to compile the MCMC engine. If you see no errors, the C++ function is securely bound to your R session.

### Step 2: Load the Modules

Highlight and run Modules 1, 2, and 3. This registers `build_H_matrix`, `fit_rkhs_model`, and `evaluate_masking_scenario` in your global R environment.

### Step 3: Run the Simulation Array

Modify the `mask_levels` vector to define the proportions of ungenotyped individuals you wish to test.

```r
mask_levels <- c(0.1, 0.3, 0.5, 0.7)

```

Pass this to the `lapply` loop. **Note on computation time:** For quick testing, leave `nIter = 3000`. For publication-ready model convergence, increase `nIter = 12000` and `burnIn = 2000`.

### Step 4: Interpret Outputs

The execution loop will output a `final_results` dataframe in the console.

* `Accuracy_A`: Represents the baseline prediction accuracy if no genomic data was available (pedigree only).
* `Accuracy_H`: Represents the accuracy of the single-step model.

Run the final `ggplot2` visualization block. The resulting plot will visualize the decay in `Accuracy_H` as the proportion of ungenotyped lines increases, ultimately converging toward the baseline `Accuracy_A`.