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


// Support out-of-source builds: if MFEM_BUILD_DIR is defined, load the config
// file MFEM_BUILD_DIR/config/_config.hpp.
//
// Otherwise, use the local file: _config.hpp.

#ifdef MFEM_BUILD_DIR
#define MFEM_QUOTE(a) #a
#define MFEM_MAKE_PATH(x,y) MFEM_QUOTE(x/y)
#include MFEM_MAKE_PATH(MFEM_BUILD_DIR,config/_config.hpp)
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

// Check dependencies:

// Options that require MPI
#ifndef MFEM_USE_MPI
#ifdef MFEM_USE_SUPERLU
#error Building with SuperLU_DIST (MFEM_USE_SUPERLU=YES) requires MPI (MFEM_USE_MPI=YES)
#endif
#ifdef MFEM_USE_STRUMPACK
#error Building with STRUMPACK (MFEM_USE_STRUMPACK=YES) requires MPI (MFEM_USE_MPI=YES)
#endif
#ifdef MFEM_USE_PETSC
#error Building with PETSc (MFEM_USE_PETSC=YES) requires MPI (MFEM_USE_MPI=YES)
#endif
#ifdef MFEM_USE_PUMI
#error Building with PUMI (MFEM_USE_PUMI=YES) requires MPI (MFEM_USE_MPI=YES)
#endif
#endif // MFEM_USE_MPI not defined
