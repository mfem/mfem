if(MFEM_USE_MPI AND MFEM_USE_DOUBLE)
  add_custom_target(stokes_package)
  add_library(stokes_core STATIC stokes.cpp)
  target_link_libraries(stokes_core PUBLIC mfem)
  target_include_directories(stokes_core PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
  target_compile_features(stokes_core PUBLIC cxx_std_17)
  if(MFEM_ENABLE_TESTING)
    target_compile_definitions(stokes_core PRIVATE MFEM_STOKES_TESTING)
  endif()
  add_library(stokes_example_support STATIC
    examples/support/manufactured.cpp examples/support/mesh_manufactured.cpp)
  target_link_libraries(stokes_example_support PUBLIC stokes_core)
  if(MFEM_USE_CUDA)
    get_target_property(stokes_core_sources stokes_core SOURCES)
    get_target_property(stokes_support_sources stokes_example_support SOURCES)
    set_source_files_properties(${stokes_core_sources} ${stokes_support_sources} PROPERTIES LANGUAGE CUDA)
  endif()
  function(add_stokes_executable target main)
    add_mfem_miniapp(${target}
      MAIN ${main}
      EXTRA_SOURCES ${ARGN}
      EXTRA_HEADERS
      EXTRA_OPTIONS
      EXTRA_DEFINES
      LIBRARIES stokes_core)
    add_dependencies(stokes_package ${target})
  endfunction()
  add_stokes_executable(stokes_pspg examples/stokes_pspg.cpp)
  add_stokes_executable(flow_example examples/flow.cpp)
  add_stokes_executable(channel_flow examples/channel_flow.cpp)
  add_stokes_executable(refine_channel_serial examples/refine_channel_serial.cpp)
  add_stokes_executable(stokes_mms examples/stokes_mms_main.cpp)
  add_stokes_executable(stokes_adjoint_example examples/stokes_adjoint_example.cpp)
  add_stokes_executable(steady_navier_stokes_setup examples/steady_navier_stokes.cpp)
  add_stokes_executable(steady_navier_stokes_workflow examples/steady_navier_stokes_workflow.cpp)
  if(MFEM_USE_MUMPS)
    add_stokes_executable(steady_navier_stokes_gmg_benchmark examples/steady_navier_stokes_gmg_benchmark.cpp)
    add_stokes_executable(steady_navier_stokes examples/steady_navier_stokes_example.cpp)
    add_stokes_executable(stokes_brinkman examples/stokes_brinkman.cpp)
    add_stokes_executable(stokes_brinkman_linear_example examples/stokes_brinkman_linear_example.cpp)
  endif()
  foreach(target stokes_pspg stokes_mms stokes_adjoint_example)
    target_link_libraries(${target} PRIVATE stokes_example_support)
  endforeach()
  if(MFEM_ENABLE_TESTING)
    add_stokes_executable(stokes_package_headers tests/package_headers.cpp)
    add_stokes_executable(stokes_flow_namespace tests/flow_namespace.cpp)
    add_stokes_executable(stokes_flow_interface tests/flow_interface.cpp)
    add_executable(stokes_driver_checks tests/driver_checks.cpp)
    target_compile_features(stokes_driver_checks PRIVATE cxx_std_17)
    add_dependencies(stokes_package stokes_driver_checks)
    add_dependencies(${MFEM_ALL_MINIAPPS_TARGET_NAME} stokes_driver_checks)
    add_test(NAME stokes_driver_checks COMMAND stokes_driver_checks self-test)
    set_tests_properties(stokes_driver_checks PROPERTIES LABELS "stokes;stokes-package" TIMEOUT 30)
    foreach(ranks 1 4)
      foreach(method 0 1 2 3)
        foreach(backend iterative projected direct)
          if(backend STREQUAL "direct" AND NOT MFEM_USE_MUMPS)
            continue()
          endif()
          add_test(NAME stokes_flow_interface_${backend}_${method}_np${ranks}
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:stokes_flow_interface> --backend ${backend} --method ${method}
              ${MPIEXEC_POSTFLAGS})
          set_tests_properties(stokes_flow_interface_${backend}_${method}_np${ranks} PROPERTIES
            PROCESSORS ${ranks} TIMEOUT 180 LABELS "stokes;stokes-flow-interface"
            ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
        endforeach()
      endforeach()
    endforeach()
    function(add_unified_flow_configuration name ranks)
      add_test(NAME stokes_flow_unified_${name}_np${ranks}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
          $<TARGET_FILE:stokes_flow_interface> --automatic-gauge ${ARGN} ${MPIEXEC_POSTFLAGS})
      set_tests_properties(stokes_flow_unified_${name}_np${ranks} PROPERTIES
        PROCESSORS ${ranks} TIMEOUT 180 LABELS "stokes;stokes-flow-unified"
        ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
    endfunction()
    foreach(ranks 1 4)
      foreach(krylov gmres fgmres)
        foreach(preconditioner RANGE 0 6)
          add_unified_flow_configuration(${krylov}_${preconditioner} ${ranks}
            --krylov ${krylov} --preconditioner ${preconditioner})
        endforeach()
      endforeach()
      foreach(method 2 3)
        foreach(preconditioner 5 6)
          add_unified_flow_configuration(residual_${method}_${preconditioner} ${ranks}
            --krylov gmres --method ${method} --preconditioner ${preconditioner})
        endforeach()
      endforeach()
    endforeach()
    set(header_source "${CMAKE_CURRENT_BINARY_DIR}/headers/stokes.hpp.cpp")
    file(GENERATE OUTPUT "${header_source}" CONTENT "#include \"stokes.hpp\"\n#include \"stokes.hpp\"\n")
    target_sources(stokes_package_headers PRIVATE "${header_source}")
    add_test(NAME stokes_package_headers COMMAND stokes_package_headers)
    set_tests_properties(stokes_package_headers PROPERTIES LABELS "stokes;stokes-package" TIMEOUT 30)
    add_test(NAME stokes_flow_namespace COMMAND stokes_flow_namespace)
    set_tests_properties(stokes_flow_namespace PROPERTIES LABELS "stokes;stokes-package" TIMEOUT 30)
    if(MFEM_USE_MUMPS)
      add_stokes_executable(steady_navier_stokes_gmg_tests tests/steady_navier_stokes_gmg_tests.cpp)
      function(add_navier_stokes_gmg_test name ranks)
        add_test(NAME navier_stokes_gmg_${name}_np${ranks}
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:steady_navier_stokes_gmg_tests> ${ARGN} ${MPIEXEC_POSTFLAGS})
        set_tests_properties(navier_stokes_gmg_${name}_np${ranks} PROPERTIES
          PROCESSORS ${ranks} TIMEOUT 900 LABELS "stokes;navier-stokes-gmg"
          ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
      endfunction()
      foreach(ranks 1 2 4)
        foreach(geometry simplex tensor)
          foreach(active_levels 1 2 3)
            add_navier_stokes_gmg_test(levels_operators_2d_${active_levels}_${geometry} ${ranks}
              --suite operators --${geometry} --gmg-levels ${active_levels})
            if(active_levels LESS 3)
              foreach(method 0 3)
                foreach(boundary closed open projected)
                  add_navier_stokes_gmg_test(levels_solves_${active_levels}_${method}_${boundary}_${geometry} ${ranks}
                    --suite solves --refinements 2 --gmg-levels ${active_levels}
                    --method ${method} --${boundary} --${geometry})
                endforeach()
              endforeach()
            endif()
          endforeach()
          add_navier_stokes_gmg_test(levels_operators_3d_${geometry} ${ranks}
            --suite operators --dimension 3 --${geometry} --gmg-levels 2)
          foreach(dimension 2 3)
            add_navier_stokes_gmg_test(operators_${dimension}d_${geometry} ${ranks}
              --suite operators --dimension ${dimension} --${geometry})
            add_navier_stokes_gmg_test(root_operators_${dimension}d_${geometry} ${ranks}
              --suite operators --dimension ${dimension} --${geometry} --root-coarse)
          endforeach()
          foreach(order 3 4)
            add_navier_stokes_gmg_test(operators_order${order}_${geometry} ${ranks}
              --suite operators --order ${order} --${geometry})
          endforeach()
          add_navier_stokes_gmg_test(fixed_operators_${geometry} ${ranks}
            --suite operators --${geometry} --fixed-jacobi)
          foreach(method 0 1 2 3)
            foreach(boundary closed open projected)
              add_navier_stokes_gmg_test(solves_${method}_${boundary}_${geometry} ${ranks}
                --suite solves --method ${method} --${boundary} --${geometry})
            endforeach()
          endforeach()
          foreach(method 0 3)
            foreach(boundary closed open projected)
              add_navier_stokes_gmg_test(root_solves_${method}_${boundary}_${geometry} ${ranks}
                --suite solves --method ${method} --${boundary} --${geometry} --root-coarse)
            endforeach()
            foreach(boundary closed open)
              add_navier_stokes_gmg_test(solves_3d_${method}_${boundary}_${geometry} ${ranks}
                --suite solves --dimension 3 --method ${method} --${boundary} --${geometry})
            endforeach()
            add_navier_stokes_gmg_test(reaction_${method}_${geometry} ${ranks}
              --suite solves --method ${method} --${geometry} --reaction)
            add_navier_stokes_gmg_test(order3_${method}_${geometry} ${ranks}
              --suite solves --method ${method} --${geometry} --order 3)
          endforeach()
          add_navier_stokes_gmg_test(scaling_${geometry} ${ranks} --suite scaling --${geometry})
          add_navier_stokes_gmg_test(root_reaction_${geometry} ${ranks}
            --suite solves --method 3 --reaction --${geometry} --root-coarse)
          add_navier_stokes_gmg_test(root_solves_3d_${geometry} ${ranks}
            --suite solves --dimension 3 --${geometry} --root-coarse)
        endforeach()
        add_navier_stokes_gmg_test(fixed_solve ${ranks} --suite solves --fixed-jacobi)
        set(gmg_options "Invalid GMG options")
        set(gmg_options-ranks "Navier-Stokes configuration differs across ranks")
        set(gmg_provider "GMG requires UniformVelocityHierarchy")
        set(gmg_space "Incompatible GMG finest space or provider")
        set(gmg_revision "Stale GMG inverse")
        set(gmg_geometry "GMG mesh/spaces changed")
        set(gmg_coarse-geometry "GMG mesh/spaces changed")
        set(gmg_mesh "GMG mesh/spaces changed")
        set(gmg_level "Invalid GMG level")
        set(gmg_size "Invalid GMG input size")
        set(gmg_order "GMG requires velocity order 2-4")
        set(gmg_refinements "GMG requires a nonempty conforming")
        set(gmg_coordinate-order "GMG requires simplex/tensor cells")
        set(gmg_nonfinite "Nonfinite GMG input")
        set(gmg_root-dofs "GMG root coarse matrix exceeds configured limits")
        set(gmg_root-nonzeros "GMG root coarse matrix exceeds configured limits")
        set(gmg_levels-negative "Invalid GMG options")
        set(gmg_levels-excess "Invalid GMG level count for hierarchy")
        set(gmg_levels-ranks "Navier-Stokes configuration differs across ranks")
        foreach(scenario options options-ranks provider space revision geometry coarse-geometry mesh level size order refinements coordinate-order nonfinite root-dofs root-nonzeros levels-negative levels-excess levels-ranks)
          if(scenario MATCHES "^(options-ranks|levels-ranks)$" AND ranks EQUAL 1)
            continue()
          endif()
          add_test(NAME navier_stokes_gmg_invalid_${scenario}_np${ranks}
            COMMAND ${CMAKE_COMMAND} "-DLAUNCHER=${MPIEXEC}" "-DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
              "-DPREFLAGS=${MPIEXEC_PREFLAGS}" "-DPOSTFLAGS=${MPIEXEC_POSTFLAGS}"
              "-DEXECUTABLE=$<TARGET_FILE:steady_navier_stokes_gmg_tests>" "-DSCENARIO=${scenario}"
              "-DRANKS=${ranks}" "-DEXPECTED_MESSAGE=${gmg_${scenario}}"
              -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/expect_failure.cmake)
          set_tests_properties(navier_stokes_gmg_invalid_${scenario}_np${ranks} PROPERTIES
            PROCESSORS ${ranks} TIMEOUT 90 LABELS "stokes;navier-stokes-gmg"
            ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
        endforeach()
      endforeach()
      add_stokes_executable(steady_navier_stokes_adjoint_tests tests/steady_navier_stokes_adjoint_tests.cpp)
      set(adjoint_unprepared "Prepare the Navier-Stokes adjoint before layout access")
      set(adjoint_gauge "Inconsistent Navier-Stokes supplied-state gauge")
      set(adjoint_field-gauge "Inconsistent Navier-Stokes supplied-state gauge")
      set(adjoint_state-finite "Nonfinite Navier-Stokes supplied state")
      set(adjoint_spaces "Incompatible Navier-Stokes supplied-state spaces")
      set(adjoint_nonconverged "Supplied Navier-Stokes state is not converged")
      set(adjoint_nonsteady-sensitivity "Navier-Stokes sensitivity requires a converged adjoint and valid steady state")
      set(adjoint_objective-value "Nonfinite Navier-Stokes objective value")
      set(adjoint_objective-ranks "Navier-Stokes configuration differs across ranks")
      set(adjoint_gradient-size "Invalid Navier-Stokes algebraic vector")
      set(adjoint_gradient-finite "Invalid Navier-Stokes algebraic vector")
      set(adjoint_stale "Stale Navier-Stokes layout or linearization")
      set(adjoint_changed "Navier-Stokes adjoint supplied state changed")
      foreach(ranks 1 2 4)
        foreach(scenario unprepared gauge field-gauge state-finite spaces nonconverged nonsteady-sensitivity
                         objective-value objective-ranks gradient-size gradient-finite stale changed)
          if(scenario STREQUAL "objective-ranks" AND ranks EQUAL 1)
            continue()
          endif()
          add_test(NAME navier_stokes_adjoint_invalid_${scenario}_np${ranks}
            COMMAND ${CMAKE_COMMAND} "-DLAUNCHER=${MPIEXEC}" "-DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
              "-DPREFLAGS=${MPIEXEC_PREFLAGS}" "-DPOSTFLAGS=${MPIEXEC_POSTFLAGS}"
              "-DEXECUTABLE=$<TARGET_FILE:steady_navier_stokes_adjoint_tests>" "-DSCENARIO=${scenario}"
              "-DRANKS=${ranks}" "-DEXPECTED_MESSAGE=${adjoint_${scenario}}"
              -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/expect_failure.cmake)
          set_tests_properties(navier_stokes_adjoint_invalid_${scenario}_np${ranks}
            PROPERTIES PROCESSORS ${ranks} TIMEOUT 90 LABELS "stokes;navier-stokes-adjoint"
            ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
        endforeach()
        add_test(NAME navier_stokes_adjoint_lifecycle_np${ranks}
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:steady_navier_stokes_adjoint_tests> --suite lifecycle ${MPIEXEC_POSTFLAGS})
        set_tests_properties(navier_stokes_adjoint_lifecycle_np${ranks}
          PROPERTIES PROCESSORS ${ranks} TIMEOUT 600 LABELS "stokes;navier-stokes-adjoint"
          ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
        foreach(geometry simplex tensor)
          foreach(boundary closed open)
            add_test(NAME navier_stokes_adjoint_stokes_${geometry}_${boundary}_np${ranks}
              COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:steady_navier_stokes_adjoint_tests> --stokes --${geometry} --${boundary} ${MPIEXEC_POSTFLAGS})
            set_tests_properties(navier_stokes_adjoint_stokes_${geometry}_${boundary}_np${ranks}
              PROPERTIES PROCESSORS ${ranks} TIMEOUT 600 LABELS "stokes;navier-stokes-adjoint"
              ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
            add_test(NAME navier_stokes_adjoint_sensitivity_${geometry}_${boundary}_np${ranks}
              COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:steady_navier_stokes_adjoint_tests> --suite sensitivity --${geometry} --${boundary} ${MPIEXEC_POSTFLAGS})
            set_tests_properties(navier_stokes_adjoint_sensitivity_${geometry}_${boundary}_np${ranks}
              PROPERTIES PROCESSORS ${ranks} TIMEOUT 900 LABELS "stokes;navier-stokes-adjoint"
              ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
          endforeach()
          foreach(dimension 2 3)
            foreach(boundary closed open projected)
              add_test(NAME navier_stokes_adjoint_operators_${dimension}d_${geometry}_${boundary}_np${ranks}
                COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
                  $<TARGET_FILE:steady_navier_stokes_adjoint_tests> --dimension ${dimension} --${geometry} --${boundary} ${MPIEXEC_POSTFLAGS})
              set_tests_properties(navier_stokes_adjoint_operators_${dimension}d_${geometry}_${boundary}_np${ranks}
                PROPERTIES PROCESSORS ${ranks} TIMEOUT 900 LABELS "stokes;navier-stokes-adjoint"
                ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
            endforeach()
          endforeach()
        endforeach()
      endforeach()
      add_stokes_executable(steady_navier_stokes_nonlinear_tests tests/steady_navier_stokes_nonlinear_tests.cpp)
      set(nonlinear_pspg "Convection with PSPG/GLS requires Stage D")
      foreach(ranks 1 2 4)
        foreach(suite zero parameters sweep)
          foreach(geometry simplex tensor)
            add_test(NAME navier_stokes_stabilized_${suite}_${geometry}_np${ranks}
              COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:steady_navier_stokes_nonlinear_tests> --suite stabilized-${suite} --${geometry} ${MPIEXEC_POSTFLAGS})
            set_tests_properties(navier_stokes_stabilized_${suite}_${geometry}_np${ranks}
              PROPERTIES PROCESSORS ${ranks} TIMEOUT 600 LABELS "stokes;navier-stokes-stabilization"
              ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
          endforeach()
        endforeach()
        foreach(suite derivatives manufactured recovery)
          foreach(dimension 2 3)
            foreach(geometry simplex tensor)
              add_test(NAME navier_stokes_stabilized_${suite}_${dimension}d_${geometry}_np${ranks}
                COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
                  $<TARGET_FILE:steady_navier_stokes_nonlinear_tests> --suite stabilized-${suite}
                  --dimension ${dimension} --${geometry} ${MPIEXEC_POSTFLAGS})
              set_tests_properties(navier_stokes_stabilized_${suite}_${dimension}d_${geometry}_np${ranks}
                PROPERTIES PROCESSORS ${ranks} TIMEOUT 600 LABELS "stokes;navier-stokes-stabilization"
                ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
            endforeach()
          endforeach()
        endforeach()
        foreach(geometry simplex tensor)
          add_test(NAME navier_stokes_stabilized_convergence_${geometry}_np${ranks}
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:steady_navier_stokes_nonlinear_tests> --suite stabilized-convergence --${geometry} ${MPIEXEC_POSTFLAGS})
          set_tests_properties(navier_stokes_stabilized_convergence_${geometry}_np${ranks}
            PROPERTIES PROCESSORS ${ranks} TIMEOUT 600 LABELS "stokes;navier-stokes-stabilization"
            ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
        endforeach()
        foreach(scenario experimental_policy experimental_scale experimental_rank)
          if(scenario STREQUAL "experimental_rank" AND ranks EQUAL 1)
            continue()
          endif()
          set(experimental_policy "Experimental residual stabilization requires PSPG or GLS")
          set(experimental_scale "Invalid experimental residual stabilization scales")
          set(experimental_rank "Navier-Stokes configuration differs across ranks")
          add_test(NAME navier_stokes_stabilized_invalid_${scenario}_np${ranks}
            COMMAND ${CMAKE_COMMAND} "-DLAUNCHER=${MPIEXEC}" "-DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
              "-DPREFLAGS=${MPIEXEC_PREFLAGS}" "-DPOSTFLAGS=${MPIEXEC_POSTFLAGS}"
              "-DEXECUTABLE=$<TARGET_FILE:steady_navier_stokes_nonlinear_tests>" "-DSCENARIO=${scenario}"
              "-DRANKS=${ranks}" "-DEXPECTED_MESSAGE=${${scenario}}"
              -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/expect_failure.cmake)
          set_tests_properties(navier_stokes_stabilized_invalid_${scenario}_np${ranks}
            PROPERTIES PROCESSORS ${ranks} TIMEOUT 60 LABELS "stokes;navier-stokes-stabilization"
            ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
        endforeach()
      endforeach()
      set(nonlinear_gls "${nonlinear_pspg}")
      set(iterative_schur "Invalid Navier-Stokes Schur approximation")
      set(iterative_rank "Navier-Stokes configuration differs across ranks")
      set(iterative_factory "Invalid iterative inverse factory result")
      set(iterative_pivot "Reaction Schur requires strictly positive finite velocity diagonal pivots")
      set(projection_mean "Nullspace projection requires a pressure mean")
      set(projection_multiplier "Projected initial multiplier must be zero")
      set(projection_left "Unsupported projected left pressure nullspace")
      set(projection_right "Unsupported projected right pressure nullspace")
      foreach(ranks 1 2 4)
        foreach(scenario iterative_schur iterative_rank iterative_factory iterative_pivot
                         projection_mean projection_multiplier projection_left projection_right)
          if(scenario STREQUAL "iterative_rank" AND ranks EQUAL 1)
            continue()
          endif()
          add_test(NAME navier_stokes_iterative_invalid_${scenario}_np${ranks}
            COMMAND ${CMAKE_COMMAND} "-DLAUNCHER=${MPIEXEC}" "-DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
              "-DPREFLAGS=${MPIEXEC_PREFLAGS}" "-DPOSTFLAGS=${MPIEXEC_POSTFLAGS}"
              "-DEXECUTABLE=$<TARGET_FILE:steady_navier_stokes_nonlinear_tests>" "-DSCENARIO=${scenario}"
              "-DRANKS=${ranks}" "-DEXPECTED_MESSAGE=${${scenario}}"
              -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/expect_failure.cmake)
          set_tests_properties(navier_stokes_iterative_invalid_${scenario}_np${ranks}
            PROPERTIES PROCESSORS ${ranks} TIMEOUT 60 LABELS "stokes;navier-stokes-iterative"
            ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
        endforeach()
        add_test(NAME navier_stokes_iterative_lifecycle_np${ranks}
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:steady_navier_stokes_nonlinear_tests> --suite iterative-lifecycle ${MPIEXEC_POSTFLAGS})
        set_tests_properties(navier_stokes_iterative_lifecycle_np${ranks}
          PROPERTIES PROCESSORS ${ranks} TIMEOUT 600 LABELS "stokes;navier-stokes-iterative"
          ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
        foreach(geometry simplex tensor)
          add_test(NAME navier_stokes_iterative_recovery_${geometry}_np${ranks}
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:steady_navier_stokes_nonlinear_tests> --suite iterative-recovery --${geometry} ${MPIEXEC_POSTFLAGS})
          set_tests_properties(navier_stokes_iterative_recovery_${geometry}_np${ranks}
            PROPERTIES PROCESSORS ${ranks} TIMEOUT 600 LABELS "stokes;navier-stokes-iterative"
            ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
          add_test(NAME navier_stokes_iterative_sweep_${geometry}_np${ranks}
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:steady_navier_stokes_nonlinear_tests> --suite iterative-sweep --${geometry} ${MPIEXEC_POSTFLAGS})
          set_tests_properties(navier_stokes_iterative_sweep_${geometry}_np${ranks}
            PROPERTIES PROCESSORS ${ranks} TIMEOUT 600 LABELS "stokes;navier-stokes-iterative"
            ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
        endforeach()
        foreach(suite manufactured projection operators convection lsc pcd)
          foreach(dimension 2 3)
            foreach(geometry simplex tensor)
              add_test(NAME navier_stokes_iterative_${suite}_${dimension}d_${geometry}_np${ranks}
                COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
                  $<TARGET_FILE:steady_navier_stokes_nonlinear_tests> --suite iterative-${suite}
                  --dimension ${dimension} --${geometry} ${MPIEXEC_POSTFLAGS})
              set_tests_properties(navier_stokes_iterative_${suite}_${dimension}d_${geometry}_np${ranks}
                PROPERTIES PROCESSORS ${ranks} TIMEOUT 600 LABELS "stokes;navier-stokes-iterative"
                ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
            endforeach()
          endforeach()
        endforeach()
      endforeach()
      set(nonlinear_iterative "Invalid Navier-Stokes iterative options")
      set(nonlinear_initial_policy "Invalid Navier-Stokes initial guess policy")
      set(nonlinear_initial_gauge "An initial gauge multiplier requires a provided, augmented initial guess")
      set(nonlinear_wrong_output "Incompatible Navier-Stokes output spaces")
      set(nonlinear_residual_alias "Navier-Stokes residual output must not alias primal fields")
      set(nonlinear_block "Invalid Navier-Stokes Jacobian block")
      set(nonlinear_stale "Stale Navier-Stokes layout or linearization")
      set(nonlinear_convection_geometry "Invalid Navier-Stokes geometry: nonpositive/nonfinite convection-quadrature signed Jacobian")
      foreach(method cip pspg gls)
        set(nonlinear_pcd_${method} "Experimental PCD requires unstabilized Taylor-Hood flow")
      endforeach()
      set(nonlinear_pcd_partial "PCD requires fully prescribed or traction-only velocity boundary attributes")
      set(nonlinear_pcd_explicit "PCD requires boundary-attribute velocity conditions, not explicit DOF constraints")
      set(nonlinear_pcd_solver "Invalid PCD pressure inverse options")
      set(nonlinear_pcd_cycles "Invalid PCD pressure inverse options")
      foreach(ranks 1 2 4)
        add_test(NAME navier_stokes_nonlinear_example_np${ranks}
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:steady_navier_stokes> ${MPIEXEC_POSTFLAGS})
        add_test(NAME navier_stokes_nonlinear_globalization_np${ranks}
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:steady_navier_stokes_nonlinear_tests> --suite globalization ${MPIEXEC_POSTFLAGS})
        set_tests_properties(navier_stokes_nonlinear_example_np${ranks} navier_stokes_nonlinear_globalization_np${ranks}
          PROPERTIES PROCESSORS ${ranks} TIMEOUT 600 LABELS "stokes;navier-stokes-nonlinear"
          ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
        foreach(suite derivatives manufactured stokes-limit)
          foreach(dimension 2 3)
            foreach(geometry simplex tensor)
              add_test(NAME navier_stokes_nonlinear_${suite}_${dimension}d_${geometry}_np${ranks}
                COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
                  $<TARGET_FILE:steady_navier_stokes_nonlinear_tests> --suite ${suite} --dimension ${dimension}
                    --${geometry} ${MPIEXEC_POSTFLAGS})
              set_tests_properties(navier_stokes_nonlinear_${suite}_${dimension}d_${geometry}_np${ranks}
                PROPERTIES PROCESSORS ${ranks} TIMEOUT 600 LABELS "stokes;navier-stokes-nonlinear"
                ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
            endforeach()
          endforeach()
        endforeach()
        foreach(geometry simplex tensor)
          add_test(NAME navier_stokes_nonlinear_convergence_${geometry}_np${ranks}
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:steady_navier_stokes_nonlinear_tests> --suite convergence --${geometry} ${MPIEXEC_POSTFLAGS})
          set_tests_properties(navier_stokes_nonlinear_convergence_${geometry}_np${ranks}
            PROPERTIES PROCESSORS ${ranks} TIMEOUT 600 LABELS "stokes;navier-stokes-nonlinear"
            ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
        endforeach()
        foreach(scenario pspg gls iterative initial_policy initial_gauge wrong_output residual_alias block stale convection_geometry
                         pcd_cip pcd_pspg pcd_gls pcd_partial pcd_explicit pcd_solver pcd_cycles)
          add_test(NAME navier_stokes_nonlinear_invalid_${scenario}_np${ranks}
            COMMAND ${CMAKE_COMMAND}
              "-DLAUNCHER=${MPIEXEC}" "-DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
              "-DPREFLAGS=${MPIEXEC_PREFLAGS}" "-DPOSTFLAGS=${MPIEXEC_POSTFLAGS}"
              "-DEXECUTABLE=$<TARGET_FILE:steady_navier_stokes_nonlinear_tests>" "-DSCENARIO=${scenario}"
              "-DRANKS=${ranks}" "-DEXPECTED_MESSAGE=${nonlinear_${scenario}}"
              -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/expect_failure.cmake)
          set_tests_properties(navier_stokes_nonlinear_invalid_${scenario}_np${ranks}
            PROPERTIES PROCESSORS ${ranks} TIMEOUT 60 LABELS "stokes;navier-stokes-nonlinear"
            ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
        endforeach()
      endforeach()
      add_stokes_executable(stokes_brinkman_tests tests/stokes_brinkman_tests.cpp)
      add_stokes_executable(stokes_brinkman_linear_tests tests/stokes_brinkman_linear_tests.cpp)
      function(add_linear_brinkman_test name ranks)
        add_test(NAME stokes_brinkman_linear_${name}_np${ranks}
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:stokes_brinkman_linear_tests> ${ARGN} ${MPIEXEC_POSTFLAGS})
        set_tests_properties(stokes_brinkman_linear_${name}_np${ranks} PROPERTIES
          PROCESSORS ${ranks} TIMEOUT 600 LABELS "stokes;stokes-brinkman;stokes-brinkman-linear"
          ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
      endfunction()
      foreach(ranks 1 4)
        foreach(penalty 0 1000)
          foreach(refinement 1 2)
            foreach(schur lsc diagonal)
              add_linear_brinkman_test(sweep_${schur}_${penalty}_${refinement} ${ranks}
                --schur ${schur} --penalty ${penalty} --refinements ${refinement})
            endforeach()
          endforeach()
        endforeach()
        foreach(coarse mumps amg jacobi)
          add_test(NAME stokes_brinkman_linear_example_${coarse}_np${ranks}
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:stokes_brinkman_linear_example> --coarse ${coarse} ${MPIEXEC_POSTFLAGS})
          set_tests_properties(stokes_brinkman_linear_example_${coarse}_np${ranks}
            PROPERTIES PROCESSORS ${ranks} TIMEOUT 600 LABELS "stokes;stokes-brinkman;stokes-brinkman-linear"
            ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
        endforeach()
        foreach(geometry simplex tensor)
          foreach(method 0 3)
            add_linear_brinkman_test(order3_${method}_${geometry} ${ranks}
              --method ${method} --order 3 --schur diagonal --${geometry})
          endforeach()
          foreach(method 0 1 2 3)
            foreach(backend mumps p1 gmg)
              foreach(schur mass diagonal)
                if(NOT backend STREQUAL "mumps" OR schur STREQUAL "mass")
                  add_linear_brinkman_test(${method}_${backend}_${schur}_${geometry} ${ranks}
                    --method ${method} --backend ${backend} --schur ${schur} --${geometry})
                endif()
              endforeach()
            endforeach()
          endforeach()
          foreach(backend p1 gmg)
            foreach(pressure_solver mumps amg)
              foreach(boundary closed open projected)
                add_linear_brinkman_test(lsc_${backend}_${pressure_solver}_${boundary}_${geometry} ${ranks}
                  --backend ${backend} --schur lsc --lsc-solver ${pressure_solver} --${boundary} --${geometry})
              endforeach()
            endforeach()
          endforeach()
        endforeach()
        foreach(coarse mumps root jacobi amg user)
          foreach(levels 1 2)
            add_linear_brinkman_test(coarse_${coarse}_levels_${levels} ${ranks}
              --coarse ${coarse} --gmg-levels ${levels})
          endforeach()
        endforeach()
        foreach(method 0 3)
          foreach(geometry simplex tensor)
            add_linear_brinkman_test(3d_${method}_${geometry} ${ranks}
              --dimension 3 --method ${method} --schur diagonal --${geometry})
          endforeach()
        endforeach()
        foreach(scenario missing_selector incompatible_selector lsc_cip lsc_pspg lsc_gls)
          set(message "List-based velocity constraints require an auxiliary level constraint selector")
          if(scenario STREQUAL "incompatible_selector")
            set(message "Incompatible auxiliary constraints")
          endif()
          if(scenario MATCHES "^lsc_")
            set(message "Experimental LSC requires unstabilized Taylor-Hood flow")
          endif()
          add_test(NAME stokes_brinkman_linear_invalid_${scenario}_np${ranks}
            COMMAND ${CMAKE_COMMAND}
              "-DLAUNCHER=${MPIEXEC}" "-DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
              "-DPREFLAGS=${MPIEXEC_PREFLAGS}" "-DPOSTFLAGS=${MPIEXEC_POSTFLAGS}"
              "-DEXECUTABLE=$<TARGET_FILE:stokes_brinkman_linear_tests>" "-DSCENARIO=${scenario}"
              "-DRANKS=${ranks}" "-DEXPECTED_MESSAGE=${message}"
              -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/expect_failure.cmake)
          set_tests_properties(stokes_brinkman_linear_invalid_${scenario}_np${ranks}
            PROPERTIES PROCESSORS ${ranks} TIMEOUT 60 LABELS "stokes;stokes-brinkman;stokes-brinkman-linear"
            ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
        endforeach()
      endforeach()
      set(brinkman_pspg_drag "Positive Brinkman reaction with PSPG/GLS requires Stage D")
      set(brinkman_gls_drag "${brinkman_pspg_drag}")
      set(brinkman_iterative "Invalid Navier-Stokes iterative options")
      set(brinkman_missing_gauge "Pressure gauge none leaves an unresolved constant-pressure nullspace")
      set(brinkman_excess_gauge "Pressure mean constraint requires a constant-pressure nullspace")
      set(brinkman_rigid "Velocity BCs leave an unconstrained rigid translation/rotation")
      set(brinkman_conflict "Conflicting or nonfinite velocity boundary values")
      set(brinkman_dof_index "Prescribed velocity DOFs require valid owned true indices")
      set(brinkman_dof_conflict "Conflicting or nonfinite velocity boundary values or prescribed DOF values")
      set(brinkman_dof_nonfinite "${brinkman_dof_conflict}")
      set(brinkman_dof_dimension "${brinkman_dof_index}")
      set(brinkman_flux "Prescribed velocity has incompatible net boundary flux")
      set(brinkman_stale "Stale Navier-Stokes layout or linearization")
      set(brinkman_wrong_output "Incompatible flow output spaces")
      foreach(ranks 1 2 4)
        foreach(geometry simplex tensor)
          foreach(dimension 2 3)
            add_test(NAME stokes_brinkman_prescribed_dofs_${dimension}d_${geometry}_np${ranks}
              COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:stokes_brinkman_tests> --suite prescribed-dofs
                --dimension ${dimension} --${geometry} ${MPIEXEC_POSTFLAGS})
            set_tests_properties(stokes_brinkman_prescribed_dofs_${dimension}d_${geometry}_np${ranks}
              PROPERTIES PROCESSORS ${ranks} TIMEOUT 300 LABELS "stokes;stokes-brinkman"
              ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
          endforeach()
        endforeach()
        add_test(NAME stokes_brinkman_example_np${ranks}
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:stokes_brinkman> ${MPIEXEC_POSTFLAGS})
        add_test(NAME stokes_brinkman_channel_np${ranks}
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:stokes_brinkman_tests> --suite channel ${MPIEXEC_POSTFLAGS})
        add_test(NAME stokes_brinkman_high-quadrature_np${ranks}
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:stokes_brinkman_tests> --suite high-quadrature ${MPIEXEC_POSTFLAGS})
        set_tests_properties(stokes_brinkman_example_np${ranks} stokes_brinkman_channel_np${ranks}
          stokes_brinkman_high-quadrature_np${ranks}
          PROPERTIES PROCESSORS ${ranks} TIMEOUT 180 LABELS "stokes;stokes-brinkman"
          ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
        foreach(suite recovery manufactured)
          foreach(dimension 2 3)
            foreach(geometry simplex tensor)
              add_test(NAME stokes_brinkman_${suite}_${dimension}d_${geometry}_np${ranks}
                COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
                  $<TARGET_FILE:stokes_brinkman_tests> --suite ${suite} --dimension ${dimension}
                    --${geometry} ${MPIEXEC_POSTFLAGS})
              set_tests_properties(stokes_brinkman_${suite}_${dimension}d_${geometry}_np${ranks}
                PROPERTIES PROCESSORS ${ranks} TIMEOUT 600 LABELS "stokes;stokes-brinkman"
                ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
            endforeach()
          endforeach()
        endforeach()
        foreach(scenario pspg_drag gls_drag iterative missing_gauge excess_gauge rigid conflict flux stale wrong_output dof_index dof_conflict dof_nonfinite dof_dimension)
          add_test(NAME stokes_brinkman_invalid_${scenario}_np${ranks}
            COMMAND ${CMAKE_COMMAND}
              "-DLAUNCHER=${MPIEXEC}" "-DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
              "-DPREFLAGS=${MPIEXEC_PREFLAGS}" "-DPOSTFLAGS=${MPIEXEC_POSTFLAGS}"
              "-DEXECUTABLE=$<TARGET_FILE:stokes_brinkman_tests>" "-DSCENARIO=${scenario}"
              "-DRANKS=${ranks}" "-DEXPECTED_MESSAGE=${brinkman_${scenario}}"
              -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/expect_failure.cmake)
          set_tests_properties(stokes_brinkman_invalid_${scenario}_np${ranks}
            PROPERTIES PROCESSORS ${ranks} TIMEOUT 60 LABELS "stokes;stokes-brinkman"
            ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
        endforeach()
      endforeach()
    endif()
    add_stokes_executable(steady_navier_stokes_tests tests/steady_navier_stokes_tests.cpp)
    set(navier_boundary_attribute "Unknown Navier-Stokes boundary attribute")
    set(navier_boundary_nonfinite "Nonfinite sampled Navier-Stokes boundary data")
    set(navier_boundary_dimension "Invalid Navier-Stokes component or coefficient dimension")
    set(navier_direction "Invalid Navier-Stokes component or coefficient dimension")
    set(navier_domain_attribute "Unknown Navier-Stokes domain attribute")
    set(navier_penalty "Invalid Navier-Stokes Brinkman penalty")
    set(navier_penalty_coefficient "Invalid sampled Navier-Stokes material or force data")
    set(navier_force_nonfinite "Invalid sampled Navier-Stokes material or force data")
    set(navier_boundary_overlap "Overlapping Navier-Stokes boundary conditions")
    set(navier_pressure_overlap "Overlapping Navier-Stokes boundary conditions")
    set(navier_essential_trace "Navier-Stokes EssentialTrace has no supported velocity closure")
    set(navier_equal_order_none "Unsupported Navier-Stokes element orders or pair")
    set(navier_missing_gradient "Navier-Stokes residual stabilization requires a viscosity gradient")
    set(navier_stabilization "Invalid Navier-Stokes stabilization scale")
    set(navier_options "Invalid Navier-Stokes nonlinear options")
    set(navier_projection "Navier-Stokes nullspace projection requires the iterative backend")
    set(navier_rank_options "Navier-Stokes configuration differs across ranks")
    set(navier_rank_boundaries "Navier-Stokes configuration differs across ranks")
    set(navier_rank_force "Navier-Stokes configuration differs across ranks")
    set(navier_rank_mean "Navier-Stokes configuration differs across ranks")
    set(navier_wrong_state "Incompatible Navier-Stokes supplied-state spaces")
    set(navier_missing_gauge "Inconsistent Navier-Stokes supplied-state gauge")
    set(navier_nonfinite_gauge "Nonfinite Navier-Stokes configuration")
    set(navier_wrong_vector "Invalid Navier-Stokes algebraic vector")
    set(navier_stale "Stale Navier-Stokes layout or linearization")
    set(navier_mesh_change "Navier-Stokes mesh/spaces changed; reconstruct the problem")
    set(navier_operator_size "Invalid Navier-Stokes linearization operator")
    foreach(ranks 1 4)
      add_test(NAME navier_stokes_setup_np${ranks}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
          $<TARGET_FILE:steady_navier_stokes_setup> ${MPIEXEC_POSTFLAGS})
      set_tests_properties(navier_stokes_setup_np${ranks} PROPERTIES
        PROCESSORS ${ranks} TIMEOUT 120 LABELS "stokes;navier-stokes-contracts"
        ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
      foreach(suite registration spaces states options-adapter)
        add_test(NAME navier_stokes_${suite}_np${ranks}
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:steady_navier_stokes_tests> --suite ${suite} ${MPIEXEC_POSTFLAGS})
        set_tests_properties(navier_stokes_${suite}_np${ranks} PROPERTIES
          PROCESSORS ${ranks} TIMEOUT 120 LABELS "stokes;navier-stokes-contracts"
          ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
      endforeach()
      foreach(scenario boundary_attribute boundary_nonfinite boundary_dimension direction domain_attribute
              penalty penalty_coefficient force_nonfinite boundary_overlap pressure_overlap
              essential_trace equal_order_none missing_gradient stabilization options projection
              rank_options rank_boundaries rank_force rank_mean wrong_state missing_gauge
              nonfinite_gauge wrong_vector stale mesh_change operator_size)
        if(ranks EQUAL 1 AND scenario MATCHES "^rank_")
          continue()
        endif()
        add_test(NAME navier_stokes_invalid_${scenario}_np${ranks}
          COMMAND ${CMAKE_COMMAND}
            "-DLAUNCHER=${MPIEXEC}" "-DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
            "-DPREFLAGS=${MPIEXEC_PREFLAGS}" "-DPOSTFLAGS=${MPIEXEC_POSTFLAGS}"
            "-DEXECUTABLE=$<TARGET_FILE:steady_navier_stokes_tests>"
            "-DSCENARIO=${scenario}" "-DRANKS=${ranks}"
            "-DEXPECTED_MESSAGE=${navier_${scenario}}"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/expect_failure.cmake)
        set_tests_properties(navier_stokes_invalid_${scenario}_np${ranks} PROPERTIES
          PROCESSORS ${ranks} TIMEOUT 90 LABELS "stokes;navier-stokes-contracts"
          ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
      endforeach()
    endforeach()
    set(channel_mesh_directory "${CMAKE_SOURCE_DIR}/../meshes/channel-3d")
    foreach(geometry hex tet)
      if(EXISTS "${channel_mesh_directory}/channel_${geometry}.mesh")
        foreach(ranks 1 4)
          set(channel_suites smoke)
          if(ranks EQUAL 1)
            list(APPEND channel_suites resolution)
          else()
            list(APPEND channel_suites parallel features invalid)
          endif()
          foreach(suite IN LISTS channel_suites)
            add_test(NAME stokes_channel_${geometry}_${suite}_np${ranks}
              COMMAND ${CMAKE_COMMAND}
                "-DEXECUTABLE=$<TARGET_FILE:channel_flow>" "-DLAUNCHER=${MPIEXEC}"
                "-DCHECKER=$<TARGET_FILE:stokes_driver_checks>"
                "-DREFINER=$<TARGET_FILE:refine_channel_serial>"
                "-DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}" "-DPREFLAGS=${MPIEXEC_PREFLAGS}" "-DPOSTFLAGS=${MPIEXEC_POSTFLAGS}"
                "-DMESH=${channel_mesh_directory}/channel_${geometry}.mesh" "-DRANKS=${ranks}"
                "-DUSE_MUMPS=${MFEM_USE_MUMPS}" "-DSUITE=${suite}"
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_channel_flow.cmake)
            set_tests_properties(stokes_channel_${geometry}_${suite}_np${ranks} PROPERTIES
              PROCESSORS ${ranks} TIMEOUT 900 LABELS "stokes;stokes-channel"
              ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
          endforeach()
        endforeach()
      endif()
    endforeach()
    if(MFEM_USE_MUMPS)
      foreach(ranks 1 4)
        add_test(NAME navier_stokes_gmg_benchmark_levels_np${ranks}
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:steady_navier_stokes_gmg_benchmark> --cells 8 --refinements 2
            --gmg-levels 2 --backend gmg --coarse distributed --expected-nodes 0 ${MPIEXEC_POSTFLAGS})
        set_tests_properties(navier_stokes_gmg_benchmark_levels_np${ranks} PROPERTIES
          PROCESSORS ${ranks} TIMEOUT 180 LABELS "stokes;navier-stokes-gmg-benchmark"
          ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
      endforeach()
      foreach(ranks 1 4)
        foreach(variant p1-amg gmg gmg-root)
          set(benchmark_backend gmg)
          set(benchmark_coarse distributed)
          if(variant STREQUAL "p1-amg")
            set(benchmark_backend p1-amg)
          elseif(variant STREQUAL "gmg-root")
            set(benchmark_coarse root)
          endif()
          add_test(NAME navier_stokes_gmg_benchmark_${variant}_np${ranks}
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:steady_navier_stokes_gmg_benchmark> --cells 8 --refinements 1
              --backend ${benchmark_backend} --coarse ${benchmark_coarse} --expected-nodes 0 ${MPIEXEC_POSTFLAGS})
          set_tests_properties(navier_stokes_gmg_benchmark_${variant}_np${ranks} PROPERTIES
            PROCESSORS ${ranks} TIMEOUT 180 LABELS "stokes;navier-stokes-gmg-benchmark"
            ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
        endforeach()
      endforeach()
      function(stokes_add_workflow_test name ranks)
        add_test(NAME navier_stokes_workflow_${name}_np${ranks}
          COMMAND ${CMAKE_COMMAND}
            "-DEXECUTABLE=$<TARGET_FILE:steady_navier_stokes_workflow>" "-DLAUNCHER=${MPIEXEC}"
            "-DCHECKER=$<TARGET_FILE:stokes_driver_checks>"
            "-DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}" "-DPREFLAGS=${MPIEXEC_PREFLAGS}" "-DPOSTFLAGS=${MPIEXEC_POSTFLAGS}"
            "-DRANKS=${ranks}" ${ARGN}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_navier_stokes_workflow.cmake)
        set_tests_properties(navier_stokes_workflow_${name}_np${ranks} PROPERTIES
          PROCESSORS ${ranks} TIMEOUT 600 LABELS "stokes;navier-stokes-workflow"
          ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
      endfunction()
      foreach(ranks 1 2 4)
        stokes_add_workflow_test(invalid ${ranks} -DSUITE=invalid)
        set_tests_properties(navier_stokes_workflow_invalid_np${ranks} PROPERTIES TIMEOUT 300)
        foreach(geometry simplex tensor)
          foreach(method taylor-hood cip pspg gls)
            foreach(scenario channel-pressure channel-traction closed-stationary closed-moving)
              stokes_add_workflow_test(${method}_${scenario}_${geometry} ${ranks}
                -DSUITE=round-trip "-DMETHOD=${method}" "-DSCENARIO=${scenario}" "-DGEOMETRY=${geometry}")
            endforeach()
          endforeach()
          foreach(method taylor-hood gls)
            stokes_add_workflow_test(order3_${method}_${geometry} ${ranks}
              -DSUITE=round-trip "-DMETHOD=${method}" -DSCENARIO=channel-pressure "-DGEOMETRY=${geometry}" -DORDER=3)
          endforeach()
        endforeach()
      endforeach()
    endif()
    add_test(NAME stokes_package_layout
      COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_package_layout.cmake)
    set_tests_properties(stokes_package_layout PROPERTIES TIMEOUT 30 LABELS "stokes;stokes-package")
    find_program(STOKES_MAKE_EXECUTABLE NAMES gmake make)
    if(STOKES_MAKE_EXECUTABLE)
      add_test(NAME stokes_makefile
        COMMAND ${CMAKE_COMMAND} "-DMAKE_EXECUTABLE=${STOKES_MAKE_EXECUTABLE}"
          -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_makefile.cmake)
      set_tests_properties(stokes_makefile PROPERTIES TIMEOUT 60 LABELS "stokes;stokes-package")
    endif()
    add_stokes_executable(stokes_pspg_tests tests/stokes_tests.cpp tests/stokes_adjoint_tests.cpp)
    target_link_libraries(stokes_pspg_tests PRIVATE stokes_example_support)
    target_compile_definitions(stokes_pspg_tests PRIVATE MFEM_STOKES_TESTING)
    set(adjoint_vector_message "Invalid Navier-Stokes algebraic vector")
    set(adjoint_nonfinite_message "Invalid Navier-Stokes algebraic vector")
    set(adjoint_state_message "Supplied Navier-Stokes state is not converged for the current residual")
    set(adjoint_spaces_message "Incompatible Navier-Stokes supplied-state spaces")
    set(adjoint_gauge_message "Inconsistent Navier-Stokes supplied-state gauge")
    set(adjoint_stale_message "Stale or unprepared adjoint; call Prepare or Solve")
    set(adjoint_unconverged_message "Navier-Stokes sensitivity requires a converged adjoint and valid steady state")
    set(adjoint_missing_message "Adjoint requires a converged forward solve")
    set(adjoint_options_message "Invalid Navier-Stokes linear tolerances")
    set(adjoint_state_nonfinite_message "Nonfinite Navier-Stokes supplied state")
    set(adjoint_gauge_value_message "Nonfinite Navier-Stokes configuration")
    set(adjoint_derivative_height_message "Invalid Navier-Stokes parameter derivative layout or values")
    set(adjoint_derivative_width_message "Invalid Navier-Stokes parameter derivative layout or values")
    set(adjoint_derivative_nonfinite_message "Invalid Navier-Stokes residual parameter transpose output")
    set(adjoint_mesh_message "Navier-Stokes mesh/spaces changed; reconstruct the problem")
    foreach(ranks 1 4)
      if(ranks EQUAL 1)
        set(adjoint_scalar_message "Nonfinite explicit Navier-Stokes objective derivative")
        set(adjoint_options_rank_message "Invalid Navier-Stokes linear tolerances")
      else()
        set(adjoint_scalar_message "Navier-Stokes configuration differs across ranks")
        set(adjoint_options_rank_message "Navier-Stokes configuration differs across ranks")
      endif()
      foreach(backend gmres fgmres)
        add_test(NAME stokes_adjoint_example_${backend}_np${ranks}
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:stokes_adjoint_example> --solver ${backend} ${MPIEXEC_POSTFLAGS})
        set_tests_properties(stokes_adjoint_example_${backend}_np${ranks} PROPERTIES
          PROCESSORS ${ranks} TIMEOUT 120 LABELS "stokes;stokes-adjoint"
          ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
      endforeach()
      if(MFEM_USE_MUMPS)
        add_test(NAME stokes_adjoint_example_mumps_np${ranks}
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:stokes_adjoint_example> --solver mumps ${MPIEXEC_POSTFLAGS})
        set_tests_properties(stokes_adjoint_example_mumps_np${ranks} PROPERTIES
          PROCESSORS ${ranks} TIMEOUT 120 LABELS "stokes;stokes-adjoint"
          ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
      endif()
      foreach(suite matrix geometry sensitivity lifecycle)
        add_test(NAME stokes_adjoint_${suite}_np${ranks}
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:stokes_pspg_tests> --suite adjoint-${suite} ${MPIEXEC_POSTFLAGS})
        set_tests_properties(stokes_adjoint_${suite}_np${ranks} PROPERTIES
          PROCESSORS ${ranks} TIMEOUT 1800 LABELS "stokes;stokes-adjoint"
          ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
      endforeach()
      foreach(scenario vector nonfinite state spaces gauge scalar stale unconverged missing options
                       options_rank state_nonfinite gauge_value derivative_height derivative_width derivative_nonfinite mesh)
        add_test(NAME stokes_adjoint_invalid_${scenario}_np${ranks}
          COMMAND ${CMAKE_COMMAND}
            "-DLAUNCHER=${MPIEXEC}" "-DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
            "-DPREFLAGS=${MPIEXEC_PREFLAGS}" "-DPOSTFLAGS=${MPIEXEC_POSTFLAGS}"
            "-DEXECUTABLE=$<TARGET_FILE:stokes_pspg_tests>" "-DSCENARIO=adjoint_${scenario}"
            "-DRANKS=${ranks}" "-DEXPECTED_MESSAGE=${adjoint_${scenario}_message}"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/expect_failure.cmake)
        set_tests_properties(stokes_adjoint_invalid_${scenario}_np${ranks} PROPERTIES
          PROCESSORS ${ranks} TIMEOUT 60 LABELS "stokes;stokes-adjoint"
          ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
      endforeach()
      add_test(NAME stokes_timing_np${ranks}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
          $<TARGET_FILE:stokes_pspg_tests> --suite timing ${MPIEXEC_POSTFLAGS})
      set_tests_properties(stokes_timing_np${ranks} PROPERTIES
        PROCESSORS ${ranks} TIMEOUT 600 LABELS "stokes;stokes-timing"
        ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
      add_test(NAME stokes_robustness_np${ranks}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
          $<TARGET_FILE:stokes_pspg_tests> --suite robustness ${MPIEXEC_POSTFLAGS})
      set_tests_properties(stokes_robustness_np${ranks} PROPERTIES
        PROCESSORS ${ranks} TIMEOUT 600 LABELS "stokes;stokes-robustness"
        ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
      foreach(suite flux-scaling bernstein-boundary)
        add_test(NAME stokes_${suite}_np${ranks}
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:stokes_pspg_tests> --suite ${suite} ${MPIEXEC_POSTFLAGS})
        set_tests_properties(stokes_${suite}_np${ranks} PROPERTIES
          PROCESSORS ${ranks} TIMEOUT 600 LABELS "stokes;stokes-boundary-correctness"
          ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
      endforeach()
      set(robustness_scenarios overflow flux_pspg flux_gls flux_cip flux_taylor_hood positive_pspg positive_gls serendipity_pspg serendipity_gls)
      foreach(method pspg gls cip taylor_hood)
        foreach(exponent 0 3 6)
          list(APPEND robustness_scenarios flux_scale${exponent}_${method})
        endforeach()
      endforeach()
      foreach(method cip taylor_hood)
        list(APPEND robustness_scenarios positive_boundary_conflict_${method} positive_boundary_nonfinite_${method})
      endforeach()
      foreach(scenario ${robustness_scenarios})
        if(scenario STREQUAL "overflow")
          set(expected_message "Nonfinite Stokes matrix or right-hand side")
        elseif(scenario MATCHES "^flux_")
          set(expected_message "Prescribed velocity has incompatible net boundary flux")
        elseif(scenario MATCHES "^positive_boundary_nonfinite_")
          set(expected_message "Nonfinite sampled Navier-Stokes boundary data")
        elseif(scenario MATCHES "^positive_boundary_conflict_")
          set(expected_message "Conflicting or nonfinite velocity boundary values")
        else()
          set(expected_message "Unsupported Navier-Stokes residual-stabilization velocity basis")
        endif()
        add_test(NAME stokes_invalid_robustness_${scenario}_np${ranks}
          COMMAND ${CMAKE_COMMAND}
            "-DLAUNCHER=${MPIEXEC}" "-DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
            "-DPREFLAGS=${MPIEXEC_PREFLAGS}" "-DPOSTFLAGS=${MPIEXEC_POSTFLAGS}"
            "-DEXECUTABLE=$<TARGET_FILE:stokes_pspg_tests>" "-DSCENARIO=robustness_${scenario}"
            "-DRANKS=${ranks}" "-DEXPECTED_MESSAGE=${expected_message}"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/expect_failure.cmake)
        set_tests_properties(stokes_invalid_robustness_${scenario}_np${ranks} PROPERTIES
          PROCESSORS ${ranks} TIMEOUT 60 LABELS "stokes;stokes-robustness"
          ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
        if(scenario MATCHES "^flux_scale|^positive_boundary_")
          set_property(TEST stokes_invalid_robustness_${scenario}_np${ranks} APPEND PROPERTY LABELS stokes-boundary-correctness)
        endif()
      endforeach()
      foreach(suite operators solvers boundary sensitivity partition reassembly preconditioner-operators mesh-mms-fields geometry)
        add_test(NAME stokes_${suite}_np${ranks}
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:stokes_pspg_tests> --suite ${suite} ${MPIEXEC_POSTFLAGS})
        set_tests_properties(stokes_${suite}_np${ranks} PROPERTIES
          PROCESSORS ${ranks} TIMEOUT 600 LABELS stokes
          ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
      endforeach()
    endforeach()
    foreach(ranks 1 4)
      foreach(method pspg cip gls taylor-hood)
        add_test(NAME stokes_${method}_pressure-gauge_np${ranks}
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:stokes_pspg_tests> --suite pressure-gauge --method ${method} ${MPIEXEC_POSTFLAGS})
        set_tests_properties(stokes_${method}_pressure-gauge_np${ranks} PROPERTIES
          PROCESSORS ${ranks} TIMEOUT 600 LABELS "stokes;stokes-pressure-gauge"
          ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
        foreach(scenario none mean_zero flux_none flux_mean_zero reassembly_none reassembly_mean_zero)
          if(scenario MATCHES "^flux_")
            set(expected_message "Prescribed velocity has incompatible net boundary flux")
          elseif(scenario MATCHES "none$")
            set(expected_message "Pressure gauge none leaves an unresolved constant-pressure nullspace")
          else()
            set(expected_message "Pressure mean constraint requires a constant-pressure nullspace")
          endif()
          add_test(NAME stokes_invalid_gauge_${scenario}_${method}_np${ranks}
            COMMAND ${CMAKE_COMMAND}
              "-DLAUNCHER=${MPIEXEC}" "-DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
              "-DPREFLAGS=${MPIEXEC_PREFLAGS}" "-DPOSTFLAGS=${MPIEXEC_POSTFLAGS}"
              "-DEXECUTABLE=$<TARGET_FILE:stokes_pspg_tests>" "-DSCENARIO=gauge_${scenario}_${method}"
              "-DRANKS=${ranks}" "-DEXPECTED_MESSAGE=${expected_message}"
              -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/expect_failure.cmake)
          set_tests_properties(stokes_invalid_gauge_${scenario}_${method}_np${ranks} PROPERTIES
            PROCESSORS ${ranks} TIMEOUT 60 LABELS "stokes;stokes-pressure-gauge"
            ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
        endforeach()
      endforeach()
      add_test(NAME stokes_invalid_gauge_mean_nonfinite_np${ranks}
        COMMAND ${CMAKE_COMMAND}
          "-DLAUNCHER=${MPIEXEC}" "-DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
          "-DPREFLAGS=${MPIEXEC_PREFLAGS}" "-DPOSTFLAGS=${MPIEXEC_POSTFLAGS}"
          "-DEXECUTABLE=$<TARGET_FILE:stokes_pspg_tests>" "-DSCENARIO=gauge_mean_nonfinite"
          "-DRANKS=${ranks}" "-DEXPECTED_MESSAGE=Invalid Navier-Stokes pressure mean"
          -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/expect_failure.cmake)
      set_tests_properties(stokes_invalid_gauge_mean_nonfinite_np${ranks} PROPERTIES
        PROCESSORS ${ranks} TIMEOUT 60 LABELS "stokes;stokes-pressure-gauge"
        ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
      foreach(policy auto mean-zero)
        add_test(NAME stokes_pressure-gauge_${policy}_cli_np${ranks}
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:stokes_pspg> --cells 3 --pressure-gauge ${policy} ${MPIEXEC_POSTFLAGS})
        set_tests_properties(stokes_pressure-gauge_${policy}_cli_np${ranks} PROPERTIES
          PROCESSORS ${ranks} TIMEOUT 120 LABELS "stokes;stokes-pressure-gauge"
          PASS_REGULAR_EXPRESSION "RESULT .*pressure_gauge=${policy} pressure_nullspace_detected=1 pressure_gauge_active=1"
          FAIL_REGULAR_EXPRESSION "Stokes: converged=0"
          ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
      endforeach()
    endforeach()
    foreach(method pspg cip gls taylor-hood)
      foreach(suite curved-solvers curved-boundary curved-convergence curved-sensitivity curved-partition)
        foreach(ranks 1 4)
          if(ranks EQUAL 1 AND NOT suite STREQUAL "curved-solvers")
            continue()
          endif()
          add_test(NAME stokes_${method}_${suite}_np${ranks}
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:stokes_pspg_tests> --suite ${suite} --method ${method} ${MPIEXEC_POSTFLAGS})
          set_tests_properties(stokes_${method}_${suite}_np${ranks} PROPERTIES
            PROCESSORS ${ranks} TIMEOUT 1800 LABELS "stokes;stokes-geometry"
            ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
        endforeach()
      endforeach()
    endforeach()
    foreach(ranks 1 4)
      foreach(scenario dimension embedded)
        if(scenario STREQUAL "dimension")
          set(invalid_mesh "${CMAKE_CURRENT_SOURCE_DIR}/../../data/inline-segment.mesh")
        else()
          set(invalid_mesh "${CMAKE_CURRENT_SOURCE_DIR}/../../data/square-disc-surf.mesh")
        endif()
        add_test(NAME stokes_mms_invalid_${scenario}_np${ranks}
          COMMAND ${CMAKE_COMMAND}
            "-DLAUNCHER=${MPIEXEC}" "-DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
            "-DPREFLAGS=${MPIEXEC_PREFLAGS}" "-DPOSTFLAGS=${MPIEXEC_POSTFLAGS}"
            "-DEXECUTABLE=$<TARGET_FILE:stokes_mms>" "-DSCENARIO=mms_${scenario}"
            "-DCLI_ARGUMENTS=--mesh;${invalid_mesh}" "-DRANKS=${ranks}"
            "-DEXPECTED_MESSAGE=MMS requires a full-dimensional 2D or 3D mesh"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/expect_failure.cmake)
        set_tests_properties(stokes_mms_invalid_${scenario}_np${ranks} PROPERTIES
          PROCESSORS ${ranks} TIMEOUT 60 LABELS "stokes;stokes-mms"
          ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
      endforeach()
    endforeach()
    set(STOKES_MESH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../meshes")
    function(stokes_add_mms_test name dimension ranks)
      add_test(NAME stokes_mms_${name}_np${ranks}
        COMMAND ${CMAKE_COMMAND}
          "-DLAUNCHER=${MPIEXEC}" "-DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
          "-DPREFLAGS=${MPIEXEC_PREFLAGS}" "-DPOSTFLAGS=${MPIEXEC_POSTFLAGS}"
          "-DRANKS=${ranks}" "-DEXECUTABLE=$<TARGET_FILE:stokes_mms>"
          "-DDIMENSION=${dimension}"
          "-DCSV=${CMAKE_CURRENT_BINARY_DIR}/mms-${name}-np${ranks}.csv"
          ${ARGN} -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_mms_driver.cmake)
      set_tests_properties(stokes_mms_${name}_np${ranks} PROPERTIES
        PROCESSORS ${ranks} TIMEOUT 900 LABELS "stokes;stokes-mms"
        ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
    endfunction()
    function(stokes_add_smooth_mms_test name dimension ranks)
      stokes_add_mms_test(smooth_${name} ${dimension} ${ranks}
        "-DMMS_CASE=smooth" "-DVERIFY_SMOOTH=ON" "-DMETHOD=pspg"
        "-DRELATIVE_TOLERANCE=1e-12" "-DPARALLEL_REFINEMENTS=0" ${ARGN})
      set_tests_properties(stokes_mms_smooth_${name}_np${ranks} PROPERTIES
        LABELS "stokes;stokes-mms;stokes-smooth-mms")
    endfunction()
    foreach(dimension 2 3)
      if(dimension EQUAL 2)
        set(smooth_families triangles quadrilaterals)
        set(simplex triangles)
      else()
        set(smooth_families tetrahedra hexahedra)
        set(simplex tetrahedra)
      endif()
      foreach(geometry_order 1 2)
        if(geometry_order EQUAL 1)
          set(suffix "")
        else()
          set(suffix "-order2")
        endif()
        foreach(family ${smooth_families})
          set(smooth_base "${STOKES_MESH_ROOT}/holes-${dimension}d${suffix}/${family}/level-1/void")
          foreach(boundary periodic nonperiodic)
            if(EXISTS "${smooth_base}/${boundary}.mesh")
              if(boundary STREQUAL "periodic")
                set(smooth_periodic 1)
              else()
                set(smooth_periodic 0)
              endif()
              stokes_add_smooth_mms_test(${dimension}d_g${geometry_order}_${family}_${boundary} ${dimension} 4
                "-DMESH=${smooth_base}/${boundary}.mesh" "-DGEOMETRY_ORDER=${geometry_order}"
                "-DPERIODIC=${smooth_periodic}" "-DINTERFACES=0")
            endif()
          endforeach()
        endforeach()
        set(smooth_mesh "${STOKES_MESH_ROOT}/holes-${dimension}d${suffix}/${simplex}/level-1/void/periodic.mesh")
        if(EXISTS "${smooth_mesh}")
          stokes_add_smooth_mms_test(${dimension}d_g${geometry_order}_serial ${dimension} 1
            "-DMESH=${smooth_mesh}" "-DGEOMETRY_ORDER=${geometry_order}" "-DPERIODIC=1" "-DINTERFACES=0")
          stokes_add_smooth_mms_test(${dimension}d_g${geometry_order}_convergence ${dimension} 4
            "-DMESH=${smooth_mesh}" "-DGEOMETRY_ORDER=${geometry_order}" "-DPARALLEL_REFINEMENTS=1"
            "-DPERIODIC=1" "-DINTERFACES=0")
        endif()
      endforeach()
      set(smooth_filled "${STOKES_MESH_ROOT}/holes-${dimension}d-order2/${simplex}/level-1/filled/periodic.mesh")
      if(EXISTS "${smooth_filled}")
        stokes_add_smooth_mms_test(${dimension}d_filled ${dimension} 4
          "-DMESH=${smooth_filled}" "-DINTERFACES=1" "-DPERIODIC=1" "-DGEOMETRY_ORDER=2")
      endif()
    endforeach()
    set(smooth_comparison_mesh "${STOKES_MESH_ROOT}/holes-2d-order2/triangles/level-1/void/nonperiodic.mesh")
    if(EXISTS "${smooth_comparison_mesh}")
      foreach(method gls cip taylor-hood)
        stokes_add_smooth_mms_test(${method} 2 4 "-DMESH=${smooth_comparison_mesh}"
          "-DMETHOD=${method}" "-DPARALLEL_REFINEMENTS=1" "-DVISUALIZATION=ON"
          "-DPERIODIC=0" "-DINTERFACES=0" "-DGEOMETRY_ORDER=2")
      endforeach()
      if(MFEM_USE_MUMPS)
        stokes_add_smooth_mms_test(mumps 2 4 "-DMESH=${smooth_comparison_mesh}" "-DSOLVER=mumps")
      endif()
      stokes_add_smooth_mms_test(gmres 2 4 "-DMESH=${smooth_comparison_mesh}" "-DSOLVER=gmres")
    endif()
    foreach(ranks 1 4)
      set_tests_properties(stokes_mesh-mms-fields_np${ranks} PROPERTIES
        LABELS "stokes;stokes-smooth-mms")
    endforeach()
    foreach(dimension 2 3)
      if(dimension EQUAL 2)
        set(family triangles)
      else()
        set(family tetrahedra)
      endif()
      set(mesh_base "${STOKES_MESH_ROOT}/holes-${dimension}d/${family}")
      if(EXISTS "${mesh_base}/level-1/void/nonperiodic.mesh")
        foreach(policy unsupported none)
          if(policy STREQUAL "unsupported")
            set(expected_message "Unknown Stokes pressure-gauge policy")
          else()
            set(expected_message "Pressure gauge none leaves an unresolved constant-pressure nullspace")
          endif()
          add_test(NAME stokes_mms_${dimension}d_invalid_pressure_gauge_${policy}
            COMMAND ${CMAKE_COMMAND}
              "-DLAUNCHER=${MPIEXEC}" "-DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
              "-DPREFLAGS=${MPIEXEC_PREFLAGS}" "-DPOSTFLAGS=${MPIEXEC_POSTFLAGS}"
              "-DEXECUTABLE=$<TARGET_FILE:stokes_mms>" "-DSCENARIO=mms_pressure_gauge_${policy}"
              "-DCLI_ARGUMENTS=--mesh;${mesh_base}/level-1/void/nonperiodic.mesh;--pressure-gauge;${policy}"
              "-DEXPECTED_MESSAGE=${expected_message}"
              -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/expect_failure.cmake)
          set_tests_properties(stokes_mms_${dimension}d_invalid_pressure_gauge_${policy} PROPERTIES
            PROCESSORS 4 TIMEOUT 60 LABELS "stokes;stokes-mms;stokes-pressure-gauge"
            ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
        endforeach()
        foreach(ranks 1 4)
          if(dimension EQUAL 2)
            set(visualization_mesh "-DMESH=${mesh_base}/level-1/void/nonperiodic.mesh")
          else()
            set(visualization_mesh "-DMESH=${mesh_base}/level-1/void/periodic.mesh")
          endif()
          stokes_add_mms_test(paraview_${dimension}d ${dimension} ${ranks}
            ${visualization_mesh} "-DVISUALIZATION=ON")
          set_tests_properties(stokes_mms_paraview_${dimension}d_np${ranks} PROPERTIES
            LABELS "stokes;stokes-mms;stokes-visualization")
          stokes_add_mms_test(serial_${dimension}d ${dimension} ${ranks}
            "-DMESH=${mesh_base}/level-1/void/periodic.mesh"
            "-DSERIAL_REFINEMENTS=1" "-DPARALLEL_REFINEMENTS=0"
            "-DLONG_REFINEMENT_OPTIONS=ON" "-DPERIODIC=1" "-DINTERFACES=0")
        endforeach()
        foreach(boundary periodic nonperiodic)
          if(boundary STREQUAL "periodic")
            set(periodic 1)
          else()
            set(periodic 0)
          endif()
          stokes_add_mms_test(pressure_gauge_${dimension}d_${boundary} ${dimension} 4
            "-DMESH=${mesh_base}/level-1/void/${boundary}.mesh"
            "-DPRESSURE_GAUGE=mean-zero" "-DPARALLEL_REFINEMENTS=0"
            "-DPERIODIC=${periodic}" "-DINTERFACES=0")
          set_tests_properties(stokes_mms_pressure_gauge_${dimension}d_${boundary}_np4 PROPERTIES
            LABELS "stokes;stokes-mms;stokes-pressure-gauge")
          foreach(ranks 1 4)
            stokes_add_mms_test(${dimension}d_${boundary} ${dimension} ${ranks}
              "-DMESH=${mesh_base}/level-1/void/${boundary}.mesh"
              "-DPERIODIC=${periodic}" "-DINTERFACES=0")
          endforeach()
        endforeach()
      endif()
    endforeach()
    foreach(dimension 2 3)
      if(dimension EQUAL 2)
        set(family quadrilaterals)
        set(method cip)
      else()
        set(family tetrahedra)
        set(method gls)
      endif()
      set(assembled_mesh "${STOKES_MESH_ROOT}/holes-${dimension}d-order2/${family}/level-1/void/periodic.mesh")
      if(EXISTS "${assembled_mesh}")
        foreach(block diagonal upper lower)
          stokes_add_mms_test(assembled_${block}_${dimension}d ${dimension} 4
            "-DMESH=${assembled_mesh}" "-DPARALLEL_REFINEMENTS=0" "-DMETHOD=${method}"
            "-DPRECONDITIONER=block-${block}-p1-amg-assembled" "-DPERIODIC=1" "-DGEOMETRY_ORDER=2")
          set_tests_properties(stokes_mms_assembled_${block}_${dimension}d_np4 PROPERTIES
            LABELS "stokes;stokes-mms;stokes-p1-assembled")
        endforeach()
      endif()
    endforeach()
    set(quad_mesh "${STOKES_MESH_ROOT}/holes-2d/quadrilaterals/level-1/void")
    if(EXISTS "${quad_mesh}/periodic.mesh")
      stokes_add_mms_test(paraview_disabled 2 4
        "-DMESH=${quad_mesh}/periodic.mesh" "-DPARALLEL_REFINEMENTS=0" "-DVISUALIZATION=OFF")
      stokes_add_mms_test(paraview_nonconvergence 2 4
        "-DMESH=${quad_mesh}/periodic.mesh" "-DEXPECT_CONVERGED=0" "-DVISUALIZATION=ON")
      set_tests_properties(stokes_mms_paraview_disabled_np4 stokes_mms_paraview_nonconvergence_np4 PROPERTIES
        LABELS "stokes;stokes-mms;stokes-visualization")
      stokes_add_mms_test(default_refinements 2 4
        "-DMESH=${quad_mesh}/periodic.mesh" "-DDEFAULT_REFINEMENTS=ON"
        "-DPARALLEL_REFINEMENTS=0" "-DPERIODIC=1")
      stokes_add_mms_test(mixed_refinements 2 4
        "-DMESH=${quad_mesh}/nonperiodic.mesh" "-DSERIAL_REFINEMENTS=1"
        "-DPARALLEL_REFINEMENTS=1" "-DPERIODIC=0" "-DVISUALIZATION=ON")
      stokes_add_mms_test(uniform_gmres 2 4
        "-DMESH=${quad_mesh}/nonperiodic.mesh" "-DSOLVER=gmres"
        "-DPRECONDITIONER=block-diagonal-amg" "-DPERIODIC=0")
      stokes_add_mms_test(uniform_fgmres 2 4
        "-DMESH=${quad_mesh}/periodic.mesh" "-DMETHOD=pspg"
        "-DPRECONDITIONER=block-upper-p1-amg" "-DPERIODIC=1")
      stokes_add_mms_test(nonconvergence 2 4
        "-DMESH=${quad_mesh}/periodic.mesh" "-DEXPECT_CONVERGED=0")
      if(MFEM_USE_MUMPS)
        stokes_add_mms_test(uniform_mumps 2 4
          "-DMESH=${quad_mesh}/nonperiodic.mesh" "-DSOLVER=mumps")
      endif()
      if(EXISTS "${quad_mesh}/periodic.msh")
        stokes_add_mms_test(gmsh_periodic 2 4
          "-DMESH=${quad_mesh}/periodic.msh" "-DPARALLEL_REFINEMENTS=0" "-DPERIODIC=1")
      endif()
      set(mms_mesh_arguments --mesh "${quad_mesh}/periodic.mesh")
      set(mms_solver_arguments ${mms_mesh_arguments} --solver unsupported)
      set(mms_solver_message "Unknown MMS solver")
      set(mms_preconditioner_arguments ${mms_mesh_arguments} --preconditioner unsupported)
      set(mms_preconditioner_message "Unknown MMS method or preconditioner")
      set(mms_case_arguments ${mms_mesh_arguments} --mms-case unsupported)
      set(mms_case_message "Unknown mesh MMS case")
      set(mms_missing_mesh_arguments)
      set(mms_missing_mesh_message "Specify an input mesh with --mesh")
      set(mms_serial_refinements_arguments ${mms_mesh_arguments} -rs -1)
      set(mms_serial_refinements_message "MMS refinement counts must be nonnegative")
      set(mms_parallel_refinements_arguments ${mms_mesh_arguments} --refine-parallel -1)
      set(mms_parallel_refinements_message "MMS refinement counts must be nonnegative")
      set(mms_invalid_scenarios solver preconditioner case missing_mesh serial_refinements parallel_refinements)
      foreach(removed_option -mp -r -fl -ll --mesh-pattern --refine --first-level --last-level)
        string(REGEX REPLACE "^-+" "" option_name "${removed_option}")
        set(scenario removed_${option_name})
        set(mms_${scenario}_arguments ${mms_mesh_arguments} ${removed_option} 1)
        set(mms_${scenario}_message "Unrecognized option: ${removed_option}")
        list(APPEND mms_invalid_scenarios ${scenario})
      endforeach()
      foreach(scenario ${mms_invalid_scenarios})
        add_test(NAME stokes_mms_invalid_${scenario}
          COMMAND ${CMAKE_COMMAND}
            "-DLAUNCHER=${MPIEXEC}" "-DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
            "-DPREFLAGS=${MPIEXEC_PREFLAGS}" "-DPOSTFLAGS=${MPIEXEC_POSTFLAGS}"
            "-DEXECUTABLE=$<TARGET_FILE:stokes_mms>" "-DSCENARIO=mms_${scenario}"
            "-DCLI_ARGUMENTS=${mms_${scenario}_arguments}"
            "-DEXPECTED_MESSAGE=${mms_${scenario}_message}"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/expect_failure.cmake)
        set_tests_properties(stokes_mms_invalid_${scenario} PROPERTIES
          PROCESSORS 4 TIMEOUT 60 LABELS "stokes;stokes-mms"
          ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
        if(scenario STREQUAL "case")
          set_tests_properties(stokes_mms_invalid_${scenario} PROPERTIES
            LABELS "stokes;stokes-mms;stokes-smooth-mms")
        endif()
      endforeach()
    endif()
    set(filled_mesh "${STOKES_MESH_ROOT}/holes-3d/tetrahedra/level-1/filled/periodic.mesh")
    if(EXISTS "${filled_mesh}")
      stokes_add_mms_test(filled_periodic 3 4
        "-DMESH=${filled_mesh}" "-DPARALLEL_REFINEMENTS=0" "-DMETHOD=gls"
        "-DPRECONDITIONER=block-diagonal-p1-amg" "-DPERIODIC=1" "-DINTERFACES=1")
    endif()
    set(hex_mesh "${STOKES_MESH_ROOT}/holes-3d/hexahedra/level-1/void/nonperiodic.mesh")
    if(EXISTS "${hex_mesh}")
      stokes_add_mms_test(hex_nonperiodic 3 4
        "-DMESH=${hex_mesh}" "-DPARALLEL_REFINEMENTS=0" "-DPERIODIC=0")
    endif()
    foreach(dimension 2 3)
      if(dimension EQUAL 2)
        set(families triangles quadrilaterals)
      else()
        set(families tetrahedra hexahedra)
      endif()
      foreach(family ${families})
        set(mesh_base "${STOKES_MESH_ROOT}/holes-${dimension}d-order2/${family}")
        if(NOT EXISTS "${mesh_base}/level-1/void/periodic.mesh")
          continue()
        endif()
        foreach(boundary periodic nonperiodic)
          if(family STREQUAL "triangles" OR family STREQUAL "tetrahedra")
            if(boundary STREQUAL "periodic")
              set(method gls)
            else()
              set(method pspg)
            endif()
          else()
            if(boundary STREQUAL "periodic")
              set(method taylor-hood)
            else()
              set(method cip)
            endif()
          endif()
          set(name order2_${dimension}d_${family}_${boundary})
          if(family STREQUAL "hexahedra")
            set(mesh_arguments "-DMESH=${mesh_base}/level-1/void/${boundary}.mesh" "-DPARALLEL_REFINEMENTS=0")
          else()
            set(mesh_arguments "-DMESH=${mesh_base}/level-1/void/${boundary}.mesh")
          endif()
          stokes_add_mms_test(${name} ${dimension} 4 ${mesh_arguments}
            "-DMETHOD=${method}" "-DGEOMETRY_ORDER=2")
          set_tests_properties(stokes_mms_${name}_np4 PROPERTIES LABELS "stokes;stokes-mms;stokes-geometry")
          if(family STREQUAL "hexahedra")
            set_tests_properties(stokes_mms_${name}_np4 PROPERTIES LABELS "stokes;stokes-mms;stokes-geometry;slow")
          endif()
        endforeach()
        if(family STREQUAL "triangles" OR family STREQUAL "tetrahedra")
          stokes_add_mms_test(order2_${dimension}d_serial ${dimension} 4
            "-DMESH=${mesh_base}/level-1/void/periodic.mesh" "-DGEOMETRY_ORDER=2"
            "-DSERIAL_REFINEMENTS=1" "-DPARALLEL_REFINEMENTS=0")
        endif()
      endforeach()
    endforeach()
    set(curved_base "${STOKES_MESH_ROOT}/holes-2d-order2/quadrilaterals/level-1")
    if(EXISTS "${curved_base}/void/periodic.msh")
      stokes_add_mms_test(order2_gmsh_periodic 2 4
        "-DMESH=${curved_base}/void/periodic.msh" "-DPARALLEL_REFINEMENTS=0" "-DMETHOD=gls" "-DPERIODIC=1" "-DGEOMETRY_ORDER=2")
      stokes_add_mms_test(order2_filled_periodic 2 4
        "-DMESH=${curved_base}/filled/periodic.mesh" "-DPARALLEL_REFINEMENTS=0" "-DMETHOD=pspg" "-DPERIODIC=1" "-DINTERFACES=1" "-DGEOMETRY_ORDER=2")
    endif()
    add_test(NAME stokes_convergence_np4
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:stokes_pspg_tests> --suite convergence ${MPIEXEC_POSTFLAGS})
    set_tests_properties(stokes_convergence_np4 PROPERTIES
      PROCESSORS 4 TIMEOUT 1800 LABELS "stokes;slow"
      ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
    foreach(method cip gls taylor-hood)
      foreach(ranks 1 4)
        foreach(suite solvers boundary sensitivity partition reassembly)
          add_test(NAME stokes_${method}_${suite}_np${ranks}
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:stokes_pspg_tests> --suite ${suite} --method ${method} ${MPIEXEC_POSTFLAGS})
          set_tests_properties(stokes_${method}_${suite}_np${ranks} PROPERTIES
            PROCESSORS ${ranks} TIMEOUT 600 LABELS stokes
            ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
        endforeach()
      endforeach()
      add_test(NAME stokes_${method}_convergence_np4
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
          $<TARGET_FILE:stokes_pspg_tests> --suite convergence --method ${method} ${MPIEXEC_POSTFLAGS})
      set_tests_properties(stokes_${method}_convergence_np4 PROPERTIES
        PROCESSORS 4 TIMEOUT 1800 LABELS "stokes;slow"
        ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
    endforeach()
    foreach(ranks 1 4)
      add_test(NAME stokes_extension_operators_np${ranks}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
          $<TARGET_FILE:stokes_pspg_tests> --suite extension-operators ${MPIEXEC_POSTFLAGS})
      set_tests_properties(stokes_extension_operators_np${ranks} PROPERTIES
        PROCESSORS ${ranks} TIMEOUT 600 LABELS stokes
        ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
    endforeach()
    foreach(method pspg cip gls taylor-hood)
      foreach(preconditioner block-diagonal-p1-amg block-upper-p1-amg block-lower-p1-amg
          block-diagonal-p1-amg-assembled block-upper-p1-amg-assembled block-lower-p1-amg-assembled)
        add_test(NAME stokes_${method}_${preconditioner}_cli_np4
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:stokes_pspg> --method ${method} --element quad --order 3
            --cells 4 --variable-viscosity --shift 0.2 --solver fgmres
            --preconditioner ${preconditioner} --p1-smoothing-steps 3 ${MPIEXEC_POSTFLAGS})
        set_tests_properties(stokes_${method}_${preconditioner}_cli_np4 PROPERTIES
          PROCESSORS 4 TIMEOUT 180 LABELS stokes
          ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
        if(preconditioner MATCHES "-assembled$")
          set_tests_properties(stokes_${method}_${preconditioner}_cli_np4 PROPERTIES
            LABELS "stokes;stokes-p1-assembled")
        endif()
      endforeach()
      foreach(ranks 1 4)
        foreach(suite p1-assembly assembled-preconditioners)
          add_test(NAME stokes_${method}_${suite}_np${ranks}
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:stokes_pspg_tests> --suite ${suite} --method ${method} ${MPIEXEC_POSTFLAGS})
          set_tests_properties(stokes_${method}_${suite}_np${ranks} PROPERTIES
            PROCESSORS ${ranks} TIMEOUT 1800 LABELS "stokes;stokes-p1-assembled"
            ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
        endforeach()
        add_test(NAME stokes_${method}_preconditioners_np${ranks}
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:stokes_pspg_tests> --suite preconditioners --method ${method} ${MPIEXEC_POSTFLAGS})
        set_tests_properties(stokes_${method}_preconditioners_np${ranks} PROPERTIES
          PROCESSORS ${ranks} TIMEOUT 1200 LABELS stokes
          ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
      endforeach()
      add_test(NAME stokes_${method}_cli_np4
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
          $<TARGET_FILE:stokes_pspg> --method ${method} --element quad --order 3
          --cells 4 --variable-viscosity --shift 0.2 --solver fgmres ${MPIEXEC_POSTFLAGS})
      set_tests_properties(stokes_${method}_cli_np4 PROPERTIES
        PROCESSORS 4 TIMEOUT 120 LABELS stokes
        ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
    endforeach()
    set(cli_method_arguments --method unsupported)
    set(cli_preconditioner_arguments --preconditioner unsupported)
    set(cli_preconditioner_message "Unknown Stokes preconditioner")
    set(cli_method_message "Unknown Stokes method")
    set(cli_order_arguments --method taylor-hood --order 1)
    set(cli_order_message "Require velocity order >=2")
    set(cli_pressure_arguments --method taylor-hood --order 2 --pressure-order 2)
    set(cli_pressure_message "method-compatible pressure order")
    set(cli_pressure_gauge_arguments --pressure-gauge unsupported)
    set(cli_pressure_gauge_message "Unknown Stokes pressure-gauge policy")
    set(cli_pressure_gauge_none_arguments --pressure-gauge none --cells 3)
    set(cli_pressure_gauge_none_message "Pressure gauge none leaves an unresolved constant-pressure nullspace")
    foreach(scenario method order pressure preconditioner pressure_gauge pressure_gauge_none)
      add_test(NAME stokes_cli_invalid_${scenario}
        COMMAND ${CMAKE_COMMAND}
          "-DLAUNCHER=${MPIEXEC}" "-DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
          "-DPREFLAGS=${MPIEXEC_PREFLAGS}" "-DPOSTFLAGS=${MPIEXEC_POSTFLAGS}"
          "-DEXECUTABLE=$<TARGET_FILE:stokes_pspg>" "-DSCENARIO=cli_${scenario}"
          "-DCLI_ARGUMENTS=${cli_${scenario}_arguments}"
          "-DEXPECTED_MESSAGE=${cli_${scenario}_message}"
          -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/expect_failure.cmake)
      set_tests_properties(stokes_cli_invalid_${scenario} PROPERTIES
        PROCESSORS 4 TIMEOUT 60 LABELS stokes
        ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
      if(scenario MATCHES "^pressure_gauge")
        set_tests_properties(stokes_cli_invalid_${scenario} PROPERTIES LABELS "stokes;stokes-pressure-gauge")
      endif()
    endforeach()
    foreach(scenario ordering viscosity direction attribute dimension conflict flux rigid stabilization pressure_order cip_gamma gls_tau cip_mesh preconditioner smoothing folded_geometry coordinate_family)
      add_test(NAME stokes_invalid_${scenario}
        COMMAND ${CMAKE_COMMAND}
          "-DLAUNCHER=${MPIEXEC}" "-DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
          "-DPREFLAGS=${MPIEXEC_PREFLAGS}" "-DPOSTFLAGS=${MPIEXEC_POSTFLAGS}"
          "-DEXECUTABLE=$<TARGET_FILE:stokes_pspg_tests>" "-DSCENARIO=${scenario}"
          -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/expect_failure.cmake)
      set_tests_properties(stokes_invalid_${scenario} PROPERTIES
        PROCESSORS 4 TIMEOUT 60 LABELS stokes
        ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
    endforeach()
  endif()
endif()
