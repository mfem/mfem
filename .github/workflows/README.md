                    Finite Element Discretization Library
                                   __
                       _ __ ___   / _|  ___  _ __ ___
                      | '_ ` _ \ | |_  / _ \| '_ ` _ \
                      | | | | | ||  _||  __/| | | | | |
                      |_| |_| |_||_|   \___||_| |_| |_|

                               https://mfem.org

This directory contains the GitHub CI scripts for MFEM.

Note that some of these scripts use the shared MFEM GitHub Actions from the external mfem/github-actions repository:

  <https://github.com/mfem/github-actions>

For a particular action, e.g. `mfem/github-actions/build-mfem@v2.5`, the `v2.5` suffix denotes the branch in the above from which the action is taken.

The current CI workflows are:

## `repo-check.yml`

Runs a number of static repository-level sanity checks.

- `file-headers-check` checks copyright date, license, etc. using the `--copyright`, `--license` and `--release` options of the `config/githooks/pre-push` script.

- `code-style` checks the code style using the `--style` option of the `config/githooks/pre-push` script.

- `documentation` checks the documentation build using the `tests/scripts/documentation` script.

- `branch-history` guards against accidental commits of large files using the `--history` option of the `config/githooks/pre-push` script.

## `mfem-analysis.yml` (`build-analysis`)

Checks if the code builds and satisfies minimal requirements.

- `gitignore` builds hypre, METIS, and MFEM using `mfem/github-actions/build-hypre`, `mfem/github-actions/build-metis`, and `mfem/github-actions/build-mfem` and checks for correct `.gitignore` settings by running the `tests/scripts/gitignore` script.

## `builds-and-tests.yml`

Runs a matrix of builds and tests runs with different compilers, OS, mfem/hypre settings, etc. Also processes and upload Codecov reports.

Uses the following GitHub Actions from <https://github.com/mfem/github-actions>:

- `mfem/github-actions/build-hypre`
- `mfem/github-actions/build-metis`
- `mfem/github-actions/build-mfem`
- `mfem/github-actions/upload-coverage`

## Sanitizer Workflow for MFEM Verification

This workflow validates MFEM unit tests, examples, and miniapps using sanitizer tools.

- `sanitizers.yml` orchestrates:
  - Building and caching dependencies: HYPRE, METIS, LSAN suppression file, and LLVM libcxx.
  - Launching fine-grained jobs for serial (ASAN, MSAN, UBSAN) and parallel (ASAN, UBSAN) sanitizers.
- `sanitize-tests.yml` is a reusable workflow accepting `par` mode (`true` for parallel) and `sanitizer` (ASAN, MSAN, or UBSAN) as inputs. It executes the following jobs:
  - **Build**: Compiles the MFEM library with specified parallel and sanitizer settings.
  - **Check**: Runs verification checks.
  - Parallel jobs to test the following: **Examples**, **Miniapps** and **Unit tests**

The workflow leverages composite actions in `.github/actions/sanitize/`:

- `config`: Centralizes settings for the sanitizer workflow.
- `mfem`: Manages the MFEM library build process.
- `mpi`: Installs MPI and applies additional compilation flags.
- `restore`: Restores the testing environment state.
- `setup`: Builds or restores cached dependencies.
