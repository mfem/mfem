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

# Skip dFEM qfunction with floats
if(MFEM_USE_MPI AND MFEM_USE_DOUBLE)

	list(APPEND PAR_MTOP_SOURCES mtop_solvers.cpp)
	list(APPEND PAR_MTOP_HEADERS mtop_solvers.hpp)

	convert_filenames_to_full_paths(PAR_MTOP_SOURCES)
	convert_filenames_to_full_paths(PAR_MTOP_HEADERS)

	set(PAR_MTOP_COMMON_FILES
		EXTRA_SOURCES ${PAR_MTOP_SOURCES}
		EXTRA_HEADERS ${PAR_MTOP_HEADERS})

	add_mfem_miniapp(mtop_test_iso_elasticity
					 MAIN mtop_test_iso_elasticity.cpp
					 ${PAR_MTOP_COMMON_FILES}
					 LIBRARIES mfem)

endif (MFEM_USE_MPI AND MFEM_USE_DOUBLE)
