# Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

if(MFEM_USE_MPI)

list(APPEND AN_SOURCES linear_anisotropic_elasticity.hpp)
list(APPEND AN_HEADERS linear_anisotropic_elasticity.cpp)

convert_filenames_to_full_paths(AN_SOURCES)
convert_filenames_to_full_paths(AN_HEADERS)

set(EX_COMMON_FILES
	EXTRA_SOURCES ${AN_SOURCES}
	EXTRA_HEADERS ${AN_HEADERS})

add_mfem_miniapp(mtop_test_anisotropic_le
                      MAIN test_anisotropic_le.cpp
		      ${EX_COMMON_FILES}
                      LIBRARIES mfem)


endif (MFEM_USE_MPI)


