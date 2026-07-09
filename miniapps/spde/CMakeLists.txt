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
  add_mfem_miniapp(generate_random_field
    MAIN generate_random_field.cpp
    EXTRA_SOURCES material_metrics.cpp spde_solver.cpp util.cpp visualizer.cpp
                  transformation.cpp
    LIBRARIES mfem)

   add_mfem_miniapp(tst_rnd	
     MAIN tst_rnd.cpp
     EXTRA_SOURCES frac_noise.cpp spde_solver.cpp
     LIBRARIES mfem
   )

   add_mfem_miniapp(tst_rnd_periodic
     MAIN tst_rnd_periodic.cpp
     EXTRA_SOURCES frac_noise.cpp spde_solver.cpp
     LIBRARIES mfem
   )

   add_mfem_miniapp(tst_periodic_fraclap_mms
     MAIN tst_periodic_fraclap_mms.cpp
     EXTRA_SOURCES frac_noise.cpp spde_solver.cpp
     LIBRARIES mfem
   )

   add_mfem_miniapp(tst_additive_gll_single_cell
     MAIN tst_additive_gll_single_cell.cpp
     EXTRA_SOURCES frac_noise.cpp spde_solver.cpp
     LIBRARIES mfem
   )

   add_mfem_miniapp(tst_diffusion_mass_gmg_mms
     MAIN tst_diffusion_mass_gmg_mms.cpp
     EXTRA_SOURCES diffusion_mass_gmg.cpp frac_noise.cpp spde_solver.cpp
     LIBRARIES mfem
   )

   add_mfem_miniapp(tst_diffusion_mass_gmg_lor_mms
     MAIN tst_diffusion_mass_gmg_lor_mms.cpp
     EXTRA_SOURCES diffusion_mass_gmg.cpp frac_noise.cpp spde_solver.cpp
     LIBRARIES mfem
   )

   add_mfem_miniapp(tst_diffusion_mass_solver_mms
     MAIN tst_diffusion_mass_solver_mms.cpp
     EXTRA_SOURCES diffusion_mass_solver.cpp
     LIBRARIES mfem
   )

   add_mfem_miniapp(tst_diffusion_solver_mms
     MAIN tst_diffusion_solver_mms.cpp
     EXTRA_SOURCES diffusion_mass_solver.cpp
     LIBRARIES mfem
   )

   add_mfem_miniapp(tst_diffusion_solver_compliance_gradient
     MAIN tst_diffusion_solver_compliance_gradient.cpp
     EXTRA_SOURCES diffusion_mass_solver.cpp
     LIBRARIES mfem
   )

   add_mfem_miniapp(tst_stokes_solver_mms
     MAIN tst_stokes_solver_mms.cpp
     EXTRA_SOURCES diffusion_mass_solver.cpp
     LIBRARIES mfem
   )

   add_mfem_miniapp(tst_brinkman_stokes_solver_mms
     MAIN tst_brinkman_stokes_solver_mms.cpp
     EXTRA_SOURCES diffusion_mass_solver.cpp
     LIBRARIES mfem
   )

   add_mfem_miniapp(tst_pde_filter
     MAIN tst_pde_filter.cpp
     EXTRA_SOURCES diffusion_mass_solver.cpp
     LIBRARIES mfem
   )

   add_mfem_miniapp(tst_quadrature_pde_filter
     MAIN tst_quadrature_pde_filter.cpp
     EXTRA_SOURCES diffusion_mass_solver.cpp
     LIBRARIES mfem
   )

   add_mfem_miniapp(tst_balakrishnan_fractional_solver
     MAIN tst_balakrishnan_fractional_solver.cpp
     EXTRA_SOURCES diffusion_mass_solver.cpp
     LIBRARIES mfem
   )

   add_mfem_miniapp(tst_fractional_diffusion_solver
     MAIN tst_fractional_diffusion_solver.cpp
     EXTRA_SOURCES diffusion_mass_solver.cpp
     LIBRARIES mfem
   )

   add_mfem_miniapp(tst_fractional_diffusion_mass_compare
     MAIN tst_fractional_diffusion_mass_compare.cpp
     EXTRA_SOURCES diffusion_mass_solver.cpp frac_noise.cpp spde_solver.cpp
     LIBRARIES mfem
   )

   add_mfem_miniapp(tst_riesz_map
     MAIN tst_riesz_map.cpp
     EXTRA_SOURCES diffusion_mass_solver.cpp
     LIBRARIES mfem
   )

   add_mfem_miniapp(tst_frac_inv
     MAIN tst_frac_inv.cpp
     EXTRA_SOURCES polylib.cpp
     LIBRARIES mfem
   )


  # Copy files to the build folder.
  file(COPY ${PROJECT_SOURCE_DIR}/miniapps/spde/README.md
       DESTINATION ${PROJECT_BINARY_DIR}/miniapps/spde)

  if (MFEM_ENABLE_TESTING)
    if (MFEM_USE_DOUBLE)
      # otherwise returns MFEM_SKIP_RETURN_VALUE
      add_test(NAME generate_random_field_np${MFEM_MPI_NP}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:generate_random_field> -no-vis
        -r 3 -rp 2 -nu 4 -l1 0.09 -l2 0.03 -l3 0.05 -s 0.01 -t 0.08 -top 1
        -no-rs -m ../../data/ref-square.mesh
        ${MPIEXEC_POSTFLAGS})
    endif()
  endif()
endif()
