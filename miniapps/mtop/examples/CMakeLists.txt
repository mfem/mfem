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

list(APPEND EX_SOURCES linear_elasticity.hpp)
list(APPEND EX_HEADERS linear_elasticity.cpp)

convert_filenames_to_full_paths(EX_SOURCES)
convert_filenames_to_full_paths(EX_HEADERS)

set(EX_COMMON_FILES
	EXTRA_SOURCES ${EX_SOURCES}
	EXTRA_HEADERS ${EX_HEADERS})

add_mfem_miniapp(mtop_test_forward_le
                      MAIN test_forward_le.cpp
		      ${EX_COMMON_FILES}
                      LIBRARIES mfem)


endif (MFEM_USE_MPI)


