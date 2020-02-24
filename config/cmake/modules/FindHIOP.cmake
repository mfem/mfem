# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

# Sets the following variables:
#   - HIOP_FOUND
#   - HIOP_INCLUDE_DIRS
#   - HIOP_LIBRARIES

include(MfemCmakeUtilities)
mfem_find_package(HIOP HIOP HIOP_DIR
  "include" "hiopInterface.hpp"
  "lib" "hiop"
  "Paths to headers required by HIOP."
  "Libraries required by HIOP.")

# this test fails with parallel MFEM since mpi.h is not available (cxx compiler is used for some reason)
#  CHECK_BUILD HIOP_VERSION_OK TRUE
#"
##include <hiopInterface.hpp>
#using namespace hiop;
#int main(int argc, char *argv[])
#{
#   MPI_Init(&argc, &argv);
#   MPI_Comm comm = MPI_COMM_WORLD;
#
#   return 0;
#}
#")
