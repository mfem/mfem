# Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

# Sets the following variables:
#   - STRUMPACK_FOUND
#   - STRUMPACK_INCLUDE_DIRS
#   - STRUMPACK_LIBRARIES

include(MfemCmakeUtilities)
mfem_find_package(STRUMPACK STRUMPACK STRUMPACK_DIR
  "include" "StrumpackSparseSolverMPIDist.hpp"
  "lib" "strumpack;strumpack_sparse" # add NAMES_PER_DIR?
  "Paths to headers required by STRUMPACK."
  "Libraries required by STRUMPACK."
  CHECK_BUILD STRUMPACK_VERSION_OK TRUE
"
#include <StrumpackSparseSolverMPIDist.hpp>
using namespace strumpack;
int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);
   MPI_Comm comm = MPI_COMM_WORLD;
   StrumpackSparseSolverMPIDist<double,int> solver(comm, argc, argv, false);
   solver.options().set_from_command_line();
   return 0;
}
"
  )
