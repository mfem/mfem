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

list(APPEND CHPT_SOURCES )
list(APPEND CHPT_HEADERS )

convert_filenames_to_full_paths(CHPT_SOURCES)
convert_filenames_to_full_paths(CHPT_HEADERS)

set(CHPT_COMMON_FILES
	EXTRA_SOURCES ${CHPT_SOURCES}
	EXTRA_HEADERS ${CHPT_HEADERS})

add_mfem_miniapp(mtop_ex_scalar_chpt
	                      MAIN ex_scalar_field.cpp
		      ${CHPT_COMMON_FILES}
                      LIBRARIES mfem)

add_mfem_miniapp(mtop_ex_adapt_chpt
	                      MAIN ex_scalar_adaptive.cpp
                      ${CHPT_COMMON_FILES}
                      LIBRARIES mfem)


add_mfem_miniapp(mtop_ex_vector_chpt
	                      MAIN ex_vector_field.cpp
                      ${CHPT_COMMON_FILES}
                      LIBRARIES mfem)
	      
add_mfem_miniapp(mtop_ex_fixedslot
			 MAIN ex_fixedslot_scalar_vector.cpp
		 ${CHPT_COMMON_FILES}
                 LIBRARIES mfem)

	 # add_mfem_miniapp(mtop_ex_segmentio
	 #        MAIN ex_segmentio_scalar_vector_mfem.cpp
	 #	 ${CHPT_COMMON_FILES}
	 #        LIBRARIES mfem)

add_mfem_miniapp(mtop_ex_file
			 MAIN ex_fileio_scalar_vector_mfem.cpp
		 ${CHPT_COMMON_FILES}
                 LIBRARIES mfem)


add_mfem_miniapp(mtop_ex_revolve
	                 MAIN ex_revolve.cpp
                 ${CHPT_COMMON_FILES}
                 LIBRARIES mfem)

add_mfem_miniapp(mtop_ex_dyn
	                 MAIN ex_dynamic_chpt.cpp
                 ${CHPT_COMMON_FILES}
                 LIBRARIES mfem)

add_mfem_miniapp(mtop_ex_rev_fixedslot
	                 MAIN ex_revolve_chpt_fixedslot.cpp
                 ${CHPT_COMMON_FILES}
                 LIBRARIES mfem)


