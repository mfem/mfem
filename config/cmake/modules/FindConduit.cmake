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

include(MfemCmakeUtilities)
mfem_find_package(Conduit CONDUIT CONDUIT_DIR
  "include;include/conduit" conduit.hpp "lib" conduit
  "Paths to headers required by Conduit." "Libraries required by Conduit."
  ADD_COMPONENT relay
    "include;include/conduit" conduit_relay.hpp "lib" conduit_relay)
