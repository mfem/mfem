# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

list(APPEND SRCS
  blockmatrix.cpp
  blockoperator.cpp
  blockvector.cpp
  complex_operator.cpp
  densemat.cpp
  handle.cpp
  matrix.cpp
  ode.cpp
  operator.cpp
  solvers.cpp
  sparsemat.cpp
  sparsesmoothers.cpp
  vector.cpp
  )

list(APPEND HDRS
  blockmatrix.hpp
  blockoperator.hpp
  blockvector.hpp
  complex_operator.hpp
  densemat.hpp
  dtensor.hpp
  handle.hpp
  invariants.hpp
  linalg.hpp
  matrix.hpp
  ode.hpp
  operator.hpp
  solvers.hpp
  sparsemat.hpp
  sparsesmoothers.hpp
  tlayout.hpp
  tmatrix.hpp
  ttensor.hpp
  vector.hpp
  )

if (MFEM_USE_MPI)
  list(APPEND SRCS
    hypre.cpp
    hypre_parcsr.cpp)
  # If this list (HDRS -> HEADERS) is used for install, we probably want the
  # headers added all the time.
  list(APPEND HDRS
    hypre.hpp
    hypre_parcsr.hpp)
  if (MFEM_USE_PETSC)
    list(APPEND SRCS
      petsc.cpp)
    list(APPEND HDRS
      petsc.hpp)
  endif()
endif()

if (MFEM_USE_GINKGO)
  list(APPEND SRCS ginkgo.cpp)
  list(APPEND HDRS ginkgo.hpp)
endif()

if (MFEM_USE_SUNDIALS)
  list(APPEND SRCS sundials.cpp)
  list(APPEND HDRS sundials.hpp)
endif()

if (MFEM_USE_SUPERLU)
  list(APPEND SRCS superlu.cpp)
  # If this list (HDRS -> HEADERS) is used for install, we probably want the
  # header added all the time.
  list(APPEND HDRS superlu.hpp)
endif()

if (MFEM_USE_STRUMPACK)
  list(APPEND SRCS strumpack.cpp)
  list(APPEND HDRS strumpack.hpp)
endif()

if (MFEM_USE_HIOP)
  list(APPEND SRCS hiop.cpp)
  list(APPEND HDRS hiop.hpp)
endif()

convert_filenames_to_full_paths(SRCS)
convert_filenames_to_full_paths(HDRS)

set(SOURCES ${SOURCES} ${SRCS} PARENT_SCOPE)
set(HEADERS ${HEADERS} ${HDRS} PARENT_SCOPE)
