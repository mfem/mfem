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

if (MFEM_USE_MPI)
    add_mfem_miniapp(dfem-minimal-surface
        MAIN dfem-minimal-surface.cpp
        LIBRARIES mfem)

    # Add parallel tests.
    if (MFEM_ENABLE_TESTING)
        add_test(NAME dfem-minimal-surface=${MFEM_MPI_NP}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:dfem-minimal-surface> -der 0 -o 1 -r 2 -no-vis
        ${MPIEXEC_POSTFLAGS})
    endif()
endif()
