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

# Defines the following variables:
#   - CONDUIT_FOUND
#   - CONDUIT_LIBRARIES
#   - CONDUIT_INCLUDE_DIRS

# check to see if relay requires hdf5, if so make sure to set HDF5
# as a required dep
if(EXISTS ${CONDUIT_DIR}/include/conduit/conduit_relay_hdf5.hpp)
  message(STATUS "Conduit Relay HDF5 Support is ENABLED")
  # we only need  HDF5 if Conduit was built with HDF5 support
  set(Conduit_REQUIRED_PACKAGES "HDF5" CACHE STRING
      "Additional packages required by Conduit.")
  else()
    message(STATUS "Conduit Relay HDF5 Support is DISABLED")
endif()

include(MfemCmakeUtilities)
mfem_find_package(Conduit CONDUIT CONDUIT_DIR
  "include;include/conduit" conduit.hpp "lib" conduit
  "Paths to headers required by Conduit." "Libraries required by Conduit."
  ADD_COMPONENT relay
    "include;include/conduit" conduit_relay.hpp "lib" conduit_relay
  ADD_COMPONENT blueprint
      "include;include/conduit" conduit_blueprint.hpp "lib" conduit_blueprint)
