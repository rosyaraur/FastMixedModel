
# Linking external C/C++ libraries like SuiteSparse

Linking external C/C++ libraries like SuiteSparse to an R package is often considered a rite of passage in high-performance computing. It requires bridging the gap between your operating system's system libraries and R's compiler.

Here is the exact step-by-step breakdown of how to perform both paths: setting up the full integration for a formal R package, or bypassing it for quick script testing.


### Path A: Full Installation for an R Package (The "Right" Way)

If you are building your framework into a formal R package, you must install SuiteSparse on your operating system and instruct R on how to find it during compilation.

#### Step 1: Install SuiteSparse on your Operating System

You need the system-level development libraries for SuiteSparse. Open your terminal and run the command appropriate for your machine:

* **macOS (via Homebrew):**
```bash
brew install suite-sparse

```


* **Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install libsuitesparse-dev

```


* **Linux (CentOS/Fedora/RHEL):**
```bash
sudo yum install suitesparse-devel

```


* **Windows (via Rtools40 or newer):**
Open the "Rtools Bash" or "MSYS2" terminal installed with Rtools and run:
```bash
pacman -S mingw-w64-x86_64-suitesparse

```



#### Step 2: Create the `Makevars` Files

In the root directory of your R package, you should have a `src/` folder (this is where `CombinedMixedSolvercpp.cpp` lives). Inside this `src/` folder, you must create two text files with **no file extensions**.

**1. Create a file named `Makevars**` (Used by Mac and Linux) and paste this inside:

```makefile
# Use standard R BLAS and LAPACK routines
PKG_CXXFLAGS = $(SHLIB_OPENMP_CXXFLAGS)

# Link standard math libraries PLUS the SuiteSparse dependencies required by Eigen
PKG_LIBS = $(SHLIB_OPENMP_CXXFLAGS) $(LAPACK_LIBS) $(BLAS_LIBS) $(FLIBS) -lcholmod -lamd -lcolamd -lsuitesparseconfig

```

**2. Create a file named `Makevars.win**` (Used by Windows/Rtools) and paste this inside:

```makefile
PKG_CXXFLAGS = $(SHLIB_OPENMP_CXXFLAGS)
PKG_LIBS = $(SHLIB_OPENMP_CXXFLAGS) $(LAPACK_LIBS) $(BLAS_LIBS) $(FLIBS) -lcholmod -lamd -lcolamd -lsuitesparseconfig

```

*Note: R will automatically read these files when you run `devtools::build()` or `devtools::document()`, and it will append those `-l` (link) flags to the compiler, allowing it to successfully locate the CHOLMOD algorithms.*

---

### Path B: The Quick Testing Bypass (`sourceCpp`)

If you are not yet building a formal R package and just want to compile the C++ script directly in an R session using `Rcpp::sourceCpp("CombinedMixedSolvercpp.cpp")`, linking system libraries is highly tedious.

To bypass the SuiteSparse requirement so the rest of your engines (like PCG, RSVD, and AMD) compile flawlessly, you just need to comment out the specific lines invoking it.

**1. Comment out the Header (Line 10)**
Find the SuiteSparse/CHOLMOD header near the top of the file and add `//` to disable it:

```cpp
// #include <Eigen/CholmodSupport> // Requires SuiteSparse (CHOLMOD/Metis)

```

**2. Comment out the CHOLMOD Solver Function**
Scroll to Section 4 (Inverse Traces & Advanced Solvers) and comment out the entire `solve_mme_cholmod` function block:

```cpp
/* 
// [[Rcpp::export]]
Eigen::VectorXd solve_mme_cholmod(const Eigen::SparseMatrix<double>& C, const Eigen::VectorXd& y) {
    Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>, Eigen::Lower> solver;
    solver.analyzePattern(C);
    solver.factorize(C);
    if (solver.info() != Eigen::Success) Rcpp::stop("CHOLMOD Supernodal Factorization failed.");
    return solver.solve(y);
}
*/

```

Once those two sections are commented out, you can run `sourceCpp("CombinedMixedSolvercpp.cpp")` directly in your R console, and it will compile using only native C++ and Eigen math.