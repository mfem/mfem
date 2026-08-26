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

    add_mfem_miniapp(dfem-hyperelasticity
        MAIN dfem-hyperelasticity.cpp
        LIBRARIES mfem)

    add_mfem_miniapp(dfem-nonlinear-poisson
        MAIN dfem-nonlinear-poisson.cpp
        LIBRARIES mfem)

    add_mfem_miniapp(dfem-hyperbolic-heat
        MAIN dfem-hyperbolic-heat.cpp
        LIBRARIES mfem)

    add_mfem_miniapp(dfem-reaction-diffusion
        MAIN dfem-reaction-diffusion.cpp
        LIBRARIES mfem)

    # Build the dfem-navier library and link it to the different examples
    set(DFEM_NAVIER_SOURCES
        dfem-navier-stokes/lib/navier_preconditioners.cpp
        dfem-navier-stokes/lib/navier_solver.cpp)
    set(DFEM_NAVIER_HEADERS
        dfem-navier-stokes/lib/navier_preconditioners.hpp
        dfem-navier-stokes/lib/navier_qfunctions.hpp
        dfem-navier-stokes/lib/navier_solver.hpp)

    # If CUDA/HIP is enabled, tag the sources the same way add_mfem_miniapp
    # would have done for EXTRA_SOURCES.
    if (MFEM_USE_CUDA)
        set_source_files_properties(${DFEM_NAVIER_SOURCES}
            PROPERTIES LANGUAGE CUDA)
    endif()
    if (MFEM_USE_HIP)
        set_source_files_properties(${DFEM_NAVIER_SOURCES}
            PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)
    endif()

    add_library(dfem-navier STATIC
        ${DFEM_NAVIER_SOURCES} ${DFEM_NAVIER_HEADERS})
    target_link_libraries(dfem-navier PUBLIC mfem)
    add_dependencies(dfem-navier ${MFEM_EXEC_PREREQUISITES_TARGET_NAME})

    add_mfem_miniapp(dfem-navier-convergence
        MAIN dfem-navier-stokes/dfem-navier-convergence.cpp
        LIBRARIES dfem-navier)

    add_mfem_miniapp(dfem-navier-cavity
        MAIN dfem-navier-stokes/dfem-navier-cavity.cpp
        LIBRARIES dfem-navier)

    add_mfem_miniapp(dfem-navier-kovasznay
        MAIN dfem-navier-stokes/dfem-navier-kovasznay.cpp
        LIBRARIES dfem-navier)

    # Add parallel tests.
    if (MFEM_ENABLE_TESTING)
        add_test(NAME dfem-minimal-surface=${MFEM_MPI_NP}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:dfem-minimal-surface> -der 0 -o 1 -r 2 -no-vis
        ${MPIEXEC_POSTFLAGS})

        add_test(NAME dfem-hyperelasticity=${MFEM_MPI_NP}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:dfem-hyperelasticity> -o 1 -rs 0 -no-vis
        ${MPIEXEC_POSTFLAGS})

        add_test(NAME dfem-nonlinear-poisson=${MFEM_MPI_NP}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:dfem-nonlinear-poisson>
        -k grad -f energy -o 1 -r 2 -no-vis
        ${MPIEXEC_POSTFLAGS})

        add_test(NAME dfem-nonlinear-poisson-residual=${MFEM_MPI_NP}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:dfem-nonlinear-poisson>
        -k sol -f residual -o 1 -r 2 -no-vis
        ${MPIEXEC_POSTFLAGS})

        add_test(NAME dfem-reaction-diffusion=${MFEM_MPI_NP}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:dfem-reaction-diffusion>
        -b 1 -o 1 -r 2 -pc 2 -no-vis
        ${MPIEXEC_POSTFLAGS})

        add_test(NAME dfem-navier-convergence=${MFEM_MPI_NP}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:dfem-navier-convergence>
        -o 1 -r 0
        ${MPIEXEC_POSTFLAGS})

        add_test(NAME dfem-navier-cavity=${MFEM_MPI_NP}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:dfem-navier-cavity>
        -o 1 -e 2 -tf 0.01 -dt 0.01 -no-pv
        ${MPIEXEC_POSTFLAGS})

        add_test(NAME dfem-navier-kovasznay=${MFEM_MPI_NP}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:dfem-navier-kovasznay>
        -o 1 -r 1 -tf 0.01 -dt 0.01 -no-pv
        ${MPIEXEC_POSTFLAGS})

        foreach(AL partial legacy full)
            add_test(NAME dfem-hyperbolic-heat-${AL}=${MFEM_MPI_NP}
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:dfem-hyperbolic-heat>
            -al ${AL} -o 1 -r 0 -tf 0.1 -dt 0.05 -no-vis
            ${MPIEXEC_POSTFLAGS})
        endforeach()
    endif()
endif()
