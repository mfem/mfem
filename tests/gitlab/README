                    Finite Element Discretization Library
                                   __
                       _ __ ___   / _|  ___  _ __ ___
                      | '_ ` _ \ | |_  / _ \| '_ ` _ \
                      | | | | | ||  _||  __/| | | | | |
                      |_| |_| |_||_|   \___||_| |_| |_|

                               https://mfem.org

This directory contains utility scripts related to Gitlab testing at LLNL.

* `build_and_test` is used in CI to build TPLs (dependencies) and MFEM and to
  perform testing.

   While designed to be used in CI context, this script can also be used
   standalone on LLNL's LC in order to reproduce a similar build. The script
   uses environment variables for configuration (a place for improvement), such
   as "BUILD_ROOT", "HOST_CONFIG", "SPEC", etc. Some are mandatory, while others
   have default values.

* `get_mfem_uberenv` sets uberenv up for use with MFEM, notably to install TPLs
  with Spack.

  Uberenv, configured for MFEM, is maintained in a separate repo. This script
  downloads and places it in `tests/uberenv`. The exact commit to extract is
  hard-coded.

* `generate_spack_upstream` can be used to generate a Spack upstream instance.

  This script addresses a much less common use case, where the TPLs for any MFEM
  target appearing in CI are built in Spack instance using Uberenv. Configuring
  permissions accordingly allows this instance to be reused to prevent multiple
  installations. Such an upstream instance can be found on LC in
  `/usr/workspace/mfem/spack-upstream`.
