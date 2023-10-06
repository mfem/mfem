# Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

# Defines the following variables:
#   - NETCDF_FOUND
#   - NETCDF_LIBRARIES
#   - NETCDF_INCLUDE_DIRS

include(MfemCmakeUtilities)

# FindHDF5.cmake uses HDF5_ROOT, so we "translate" from the MFEM convention
set(HDF5_ROOT ${HDF5_DIR} CACHE PATH "")
# We need to guard against the case where HDF5 was already found but without
# the HL extensions (in which case mfem_find_package will treat the package
# as already having been found), so we reset the variable to force FindHDF5.cmake
# to be called for a second time
set(HDF5_FOUND OFF)
enable_language(C) # FindHDF5.cmake uses the C compiler
mfem_find_package(NetCDF NETCDF NETCDF_DIR "include" netcdf.h "lib" netcdf
  "Paths to headers required by NetCDF." "Libraries required by NetCDF.")
# The HL extension libraries are in a separate variable and must precede
# the "regular" hdf5 library, as hdf5_hl depends on hdf5
# The netcdf library will always be the first element of NETCDF_LIBRARIES
# and we need to insert after that library but before the hdf5 library, so
# position 1 is used
list(INSERT NETCDF_LIBRARIES 1 ${HDF5_C_LIBRARY_hdf5_hl})
