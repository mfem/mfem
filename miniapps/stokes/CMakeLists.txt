if(MFEM_USE_MPI AND MFEM_USE_DOUBLE)
  set(STOKES_SOURCES stokes_solver.cpp stokes_integrators.cpp stokes_mms.cpp stokes_preconditioners.cpp stokes_mesh_mms.cpp)
  set(STOKES_HEADERS stokes_solver.hpp stokes_integrators.hpp stokes_mms.hpp stokes_preconditioners.hpp stokes_mesh_mms.hpp)
  add_mfem_miniapp(stokes_pspg
    MAIN stokes_pspg.cpp
    EXTRA_SOURCES ${STOKES_SOURCES}
    EXTRA_HEADERS ${STOKES_HEADERS}
    LIBRARIES mfem)
  add_mfem_miniapp(stokes_mms
    MAIN stokes_mms_main.cpp
    EXTRA_SOURCES ${STOKES_SOURCES}
    EXTRA_HEADERS ${STOKES_HEADERS}
    LIBRARIES mfem)
  if(MFEM_ENABLE_TESTING)
    find_package(Python3 3.8 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
      add_test(NAME stokes_benchmark-runner
        COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/test_benchmark_2d.py -v)
      set_tests_properties(stokes_benchmark-runner PROPERTIES
        TIMEOUT 120 LABELS "stokes;stokes-benchmark-runner"
        ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
    endif()
    add_mfem_miniapp(stokes_pspg_tests
      MAIN stokes_tests.cpp
      EXTRA_SOURCES ${STOKES_SOURCES}
      EXTRA_HEADERS ${STOKES_HEADERS}
      LIBRARIES mfem)
    foreach(ranks 1 4)
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
        elseif(scenario MATCHES "^positive_boundary_")
          set(expected_message "Conflicting or nonfinite velocity boundary values")
        else()
          set(expected_message "PSPG/GLS require a velocity basis with implemented Hessians")
        endif()
        add_test(NAME stokes_invalid_robustness_${scenario}_np${ranks}
          COMMAND ${CMAKE_COMMAND}
            "-DLAUNCHER=${MPIEXEC}" "-DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
            "-DPREFLAGS=${MPIEXEC_PREFLAGS}" "-DPOSTFLAGS=${MPIEXEC_POSTFLAGS}"
            "-DEXECUTABLE=$<TARGET_FILE:stokes_pspg_tests>" "-DSCENARIO=robustness_${scenario}"
            "-DRANKS=${ranks}" "-DEXPECTED_MESSAGE=${expected_message}"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake)
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
            set(expected_message "Pressure gauge mean-zero requires a constant-pressure nullspace")
          endif()
          add_test(NAME stokes_invalid_gauge_${scenario}_${method}_np${ranks}
            COMMAND ${CMAKE_COMMAND}
              "-DLAUNCHER=${MPIEXEC}" "-DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
              "-DPREFLAGS=${MPIEXEC_PREFLAGS}" "-DPOSTFLAGS=${MPIEXEC_POSTFLAGS}"
              "-DEXECUTABLE=$<TARGET_FILE:stokes_pspg_tests>" "-DSCENARIO=gauge_${scenario}_${method}"
              "-DRANKS=${ranks}" "-DEXPECTED_MESSAGE=${expected_message}"
              -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake)
          set_tests_properties(stokes_invalid_gauge_${scenario}_${method}_np${ranks} PROPERTIES
            PROCESSORS ${ranks} TIMEOUT 60 LABELS "stokes;stokes-pressure-gauge"
            ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
        endforeach()
      endforeach()
      add_test(NAME stokes_invalid_gauge_enum_np${ranks}
        COMMAND ${CMAKE_COMMAND}
          "-DLAUNCHER=${MPIEXEC}" "-DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
          "-DPREFLAGS=${MPIEXEC_PREFLAGS}" "-DPOSTFLAGS=${MPIEXEC_POSTFLAGS}"
          "-DEXECUTABLE=$<TARGET_FILE:stokes_pspg_tests>" "-DSCENARIO=gauge_enum"
          "-DRANKS=${ranks}" "-DEXPECTED_MESSAGE=Unknown Stokes pressure-gauge policy"
          -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake)
      set_tests_properties(stokes_invalid_gauge_enum_np${ranks} PROPERTIES
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
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake)
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
          ${ARGN} -P ${CMAKE_CURRENT_SOURCE_DIR}/check_mms_driver.cmake)
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
              -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake)
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
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake)
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
          -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake)
      set_tests_properties(stokes_cli_invalid_${scenario} PROPERTIES
        PROCESSORS 4 TIMEOUT 60 LABELS stokes
        ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
      if(scenario MATCHES "^pressure_gauge")
        set_tests_properties(stokes_cli_invalid_${scenario} PROPERTIES LABELS "stokes;stokes-pressure-gauge")
      endif()
    endforeach()
    foreach(scenario ordering viscosity direction attribute dimension conflict flux rigid method pressure_order cip_gamma gls_tau cip_mesh preconditioner smoothing folded_geometry coordinate_family)
      add_test(NAME stokes_invalid_${scenario}
        COMMAND ${CMAKE_COMMAND}
          "-DLAUNCHER=${MPIEXEC}" "-DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
          "-DPREFLAGS=${MPIEXEC_PREFLAGS}" "-DPOSTFLAGS=${MPIEXEC_POSTFLAGS}"
          "-DEXECUTABLE=$<TARGET_FILE:stokes_pspg_tests>" "-DSCENARIO=${scenario}"
          -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake)
      set_tests_properties(stokes_invalid_${scenario} PROPERTIES
        PROCESSORS 4 TIMEOUT 60 LABELS stokes
        ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
    endforeach()
  endif()
endif()
