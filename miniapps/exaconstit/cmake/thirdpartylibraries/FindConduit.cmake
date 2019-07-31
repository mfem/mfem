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

      find_library( CONDUIT_RELAY_LIBRARY NAMES relay conduit_relay
              PATHS ${CONDUIT_DIR}/lib/
              NO_DEFAULT_PATH
              NO_CMAKE_ENVIRONMENT_PATH
              NO_CMAKE_PATH
              NO_SYSTEM_ENVIRONMENT_PATH
              NO_CMAKE_SYSTEM_PATH)

find_library( CONDUIT_BLUEPRINT_LIBRARY NAMES blueprint conduit_blueprint
              PATHS ${CONDUIT_DIR}/lib/
              NO_DEFAULT_PATH
              NO_CMAKE_ENVIRONMENT_PATH
              NO_CMAKE_PATH
              NO_SYSTEM_ENVIRONMENT_PATH
              NO_CMAKE_SYSTEM_PATH)