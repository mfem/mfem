                    Finite Element Discretization Library
                                   __
                       _ __ ___   / _|  ___  _ __ ___
                      | '_ ` _ \ | |_  / _ \| '_ ` _ \
                      | | | | | ||  _||  __/| | | | | |
                      |_| |_| |_||_|   \___||_| |_| |_|

                               https://mfem.org

This directory contains [Microsoft Vcpkg](https://vcpkg.io) configuration
files. Those files express the dependencies to install for MFEM. They are
primarily used in CI.

We chose to use those files, called manifests, to enforce the version of
packages, and easily track changes in the CI configuration by hashing those
manifests files in CI.

In order to ensure reproducibility, we force Vcpkg to use a particular commit
of its baseline, so that we can then freeze
([override](https://vcpkg.io/en/docs/users/versioning.html#overrides) the
versions of the installed dependencies. There is [no other
way](https://github.com/microsoft/vcpkg/discussions/25622) to do that in
Vcpkg.
