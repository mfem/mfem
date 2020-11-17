// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.


// Support out-of-source builds: if MFEM_CONFIG_FILE is defined, include it.
//
// Otherwise, use the local file: _config.hpp.

#ifndef MFEM_CONFIG_HPP
#define MFEM_CONFIG_HPP

#ifdef MFEM_CONFIG_FILE
#include MFEM_CONFIG_FILE
#else
#include "_config.hpp"
#endif

// Common configuration macros

#if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 7)) || defined(__clang__)
#define MFEM_HAVE_GCC_PRAGMA_DIAGNOSTIC
#endif

// Windows specific options
#ifdef _WIN32
// Macro needed to get defines like M_PI from <cmath>. (Visual Studio C++ only?)
#define _USE_MATH_DEFINES
#endif
// On Cygwin the option -std=c++11 prevents the definition of M_PI. Defining
// the following macro allows us to get M_PI and some needed functions, e.g.
// posix_memalign(), strdup(), strerror_r().
#ifdef __CYGWIN__
#define _XOPEN_SOURCE 600
#endif

// Check dependencies:

// Options that require MPI
#ifndef MFEM_USE_MPI
#ifdef MFEM_USE_SUPERLU
#error Building with SuperLU_DIST (MFEM_USE_SUPERLU=YES) requires MPI (MFEM_USE_MPI=YES)
#endif
#ifdef MFEM_USE_MUMPS
#error Building with MUMPS (MFEM_USE_MUMPS=YES) requires MPI (MFEM_USE_MPI=YES)
#endif
#ifdef MFEM_USE_STRUMPACK
#error Building with STRUMPACK (MFEM_USE_STRUMPACK=YES) requires MPI (MFEM_USE_MPI=YES)
#endif
#ifdef MFEM_USE_MKL_CPARDISO
#error Building with MKL CPARDISO (MFEM_USE_MKL_CPARDISO=YES) requires MPI (MFEM_USE_MPI=YES)
#endif
#ifdef MFEM_USE_PETSC
#error Building with PETSc (MFEM_USE_PETSC=YES) requires MPI (MFEM_USE_MPI=YES)
#endif
#ifdef MFEM_USE_SLEPC
#error Building with SLEPc (MFEM_USE_SLEPC=YES) requires MPI (MFEM_USE_MPI=YES)
#endif
#ifdef MFEM_USE_PUMI
#error Building with PUMI (MFEM_USE_PUMI=YES) requires MPI (MFEM_USE_MPI=YES)
#endif
#endif // MFEM_USE_MPI not defined

#endif // MFEM_CONFIG_HPP
