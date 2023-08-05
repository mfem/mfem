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

We chose to use those files, called ports and manifests, to enforce the version
of packages, and easily track changes in the CI configuration by hashing those
manifests files in CI.
