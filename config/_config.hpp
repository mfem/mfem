// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_CONFIG_HEADER
#define MFEM_CONFIG_HEADER

// Build the parallel MFEM library.
// Requires an MPI compiler, and the libraries HYPRE and METIS.
#define MFEM_USE_MPI

// Enable debug checks in MFEM.
// #define MFEM_DEBUG

// Enable gzstream in MFEM.
// #define MFEM_USE_GZSTREAM

// Enable backtraces for mfem_error through libunwind.
// #define MFEM_USE_LIBUNWIND

// Enable this option if linking with METIS version 5 (parallel MFEM).
#define MFEM_USE_METIS_5

// Use LAPACK routines for various dense linear algebra operations.
#define MFEM_USE_LAPACK

// Use thread-safe implementation. This comes at the cost of extra memory
// allocation and de-allocation.
// #define MFEM_THREAD_SAFE

// Enable experimental OpenMP support. Requires MFEM_THREAD_SAFE.
// #define MFEM_USE_OPENMP

// Internal MFEM option: enable group/batch allocation for some small objects.
#define MFEM_USE_MEMALLOC

// Which library functions to use in class StopWatch for measuring time.
// The available options are:
//   0 - use std::clock from <ctime>
//   1 - use times from <sys/times.h>
//   2 - use high-resolution POSIX clocks
//   3 - use QueryPerformanceCounter from <windows.h>
// If not defined, an option is selected automatically.
#define MFEM_TIMER_TYPE 2

// Enable MFEM functionality based on the SUNDIALS libraries.
// #define MFEM_USE_SUNDIALS

// Enable MFEM functionality based on the Mesquite library.
// #define MFEM_USE_MESQUITE

// Enable MFEM functionality based on the SuiteSparse library.
#define MFEM_USE_SUITESPARSE

// Enable MFEM functionality based on the SuperLU library.
// #define MFEM_USE_SUPERLU

// Enable functionality based on the Gecko library
// #define MFEM_USE_GECKO

// Enable secure socket streams based on the GNUTLS library
// #define MFEM_USE_GNUTLS

// Enable Sidre support
// #define MFEM_USE_SIDRE

// Enable functionality based on the NetCDF library (reading CUBIT files)
// #define MFEM_USE_NETCDF

// Enable functionality based on the PETSc library
// #define MFEM_USE_PETSC

// Enable functionality based on the MPFR library.
// #define MFEM_USE_MPFR

// Windows specific options
#ifdef _WIN32
// Macro needed to get defines like M_PI from <cmath>. (Visual Studio C++ only?)
#define _USE_MATH_DEFINES
#endif

#endif // MFEM_CONFIG_HEADER
