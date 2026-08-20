# Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All rights reserved. See files
# LICENSE and NOTICE for details.

cmake_minimum_required(VERSION 3.16)

# This directory can be configured on its own or added by an MFEM build.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(topopt_maxlen LANGUAGES CXX)
endif()

if(NOT TARGET mfem AND NOT TARGET MFEM::MFEM)
  if(DEFINED MFEM_BUILD_DIR)
    list(PREPEND CMAKE_PREFIX_PATH
      "${MFEM_BUILD_DIR}"
      "${MFEM_BUILD_DIR}/lib/cmake/mfem"
      "${MFEM_BUILD_DIR}/share/mfem")
  endif()
  find_package(MFEM CONFIG REQUIRED)
endif()

if(TARGET MFEM::MFEM)
  set(_mfem_target MFEM::MFEM)
elseif(TARGET mfem)
  set(_mfem_target mfem)
else()
  message(FATAL_ERROR "MFEM was found, but it did not provide an MFEM library target")
endif()

# The GNU make build intentionally has no serial executables.
if(DEFINED MFEM_USE_MPI AND NOT MFEM_USE_MPI)
  message(STATUS "topopt_maxlen miniapps require an MPI-enabled MFEM; no targets added")
  return()
endif()

set(_topopt_root "${CMAKE_CURRENT_SOURCE_DIR}/../..")

# These are COMMON_OBJ in the makefile.  Building them once avoids recompiling
# the same implementation for every miniapp.
add_library(topopt_maxlen_common STATIC
  "${_topopt_root}/mma/MMA_MFEM.cpp"
  "${_topopt_root}/mtop_solvers.cpp"
  "${_topopt_root}/pde_filter.cpp"
  "${_topopt_root}/diffusion_mass_solver.cpp"
  pseudo_transient_solver.cpp
  "${_topopt_root}/linear_elasticity_solver.cpp")

target_include_directories(topopt_maxlen_common PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${_topopt_root}"
  "${_topopt_root}/mma")
target_link_libraries(topopt_maxlen_common PUBLIC ${_mfem_target})

set(_topopt_maxlen_apps
  ElastTopOpt_maxlen
  ElastTopOpt_maxthick
  ElastTopOpt_adv
  ElastTopOpt_ct
  ElastTopOpt_std
  ElastTopOpt_pnorm
  ElastTopOpt_ctpnorm
  ElastTopOpt_3d
  DiffuTopOpt)

foreach(_app IN LISTS _topopt_maxlen_apps)
  add_executable(${_app} "${_app}.cpp")
  target_link_libraries(${_app} PRIVATE topopt_maxlen_common)
endforeach()
