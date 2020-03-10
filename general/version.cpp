// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license.  We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../config/config.hpp"
#include "version.hpp"

#define QUOTE(str) #str
#define EXPAND_AND_QUOTE(str) QUOTE(str)

namespace mfem
{


int GetVersion()
{
   return MFEM_VERSION;
}


int GetVersionMajor()
{
   return MFEM_VERSION_MAJOR;
}


int GetVersionMinor()
{
   return MFEM_VERSION_MINOR;
}


int GetVersionPatch()
{
   return MFEM_VERSION_PATCH;
}


const char *GetVersionStr()
{
#if MFEM_VERSION_TYPE == MFEM_VERSION_TYPE_RELEASE
#define MFEM_VERSION_TYPE_STR " (release)"
#elif MFEM_VERSION_TYPE == MFEM_VERSION_TYPE_DEVELOPMENT
#define MFEM_VERSION_TYPE_STR " (development)"
#endif
   static const char *version_str =
      "MFEM v" MFEM_VERSION_STRING MFEM_VERSION_TYPE_STR;
   return version_str;
}


const char *GetGitStr()
{
   static const char *git_str = MFEM_GIT_STRING;
   return git_str;
}


const char *GetConfigStr()
{
   static const char *config_str =
      ""
#ifdef MFEM_USE_MPI
      "MFEM_USE_MPI\n"
#endif
#ifdef MFEM_USE_METIS
      "MFEM_USE_METIS\n"
#endif
#ifdef MFEM_USE_METIS_5
      "MFEM_USE_METIS_5\n"
#endif
#ifdef MFEM_DEBUG
      "MFEM_DEBUG\n"
#endif
#ifdef MFEM_USE_EXCEPTIONS
      "MFEM_USE_EXCEPTIONS\n"
#endif
#ifdef MFEM_USE_ZLIB
      "MFEM_USE_ZLIB\n"
#endif
#ifdef MFEM_USE_LIBUNWIND
      "MFEM_USE_LIBUNWIND\n"
#endif
#ifdef MFEM_USE_LAPACK
      "MFEM_USE_LAPACK\n"
#endif
#ifdef MFEM_THREAD_SAFE
      "MFEM_THREAD_SAFE\n"
#endif
#ifdef MFEM_USE_LEGACY_OPENMP
      "MFEM_USE_LEGACY_OPENMP\n"
#endif
#ifdef MFEM_USE_MEMALLOC
      "MFEM_USE_MEMALLOC\n"
#endif
#ifdef MFEM_USE_SUNDIALS
      "MFEM_USE_SUNDIALS\n"
#endif
#ifdef MFEM_USE_MESQUITE
      "MFEM_USE_MESQUITE\n"
#endif
#ifdef MFEM_USE_SUITESPARSE
      "MFEM_USE_SUITESPARSE\n"
#endif
#ifdef MFEM_USE_SUPERLU
      "MFEM_USE_SUPERLU\n"
#endif
#ifdef MFEM_USE_STRUMPACK
      "MFEM_USE_STRUMPACK\n"
#endif
#ifdef MFEM_USE_GECKO
      "MFEM_USE_GECKO\n"
#endif
#ifdef MFEM_USE_GNUTLS
      "MFEM_USE_GNUTLS\n"
#endif
#ifdef MFEM_USE_NETCDF
      "MFEM_USE_NETCDF\n"
#endif
#ifdef MFEM_USE_PETSC
      "MFEM_USE_PETSC\n"
#endif
#ifdef MFEM_USE_MPFR
      "MFEM_USE_MPFR\n"
#endif
#ifdef MFEM_USE_CONDUIT
      "MFEM_USE_CONDUIT\n"
#endif
#ifdef MFEM_USE_SIDRE
      "MFEM_USE_SIDRE\n"
#endif
#ifdef MFEM_USE_PUMI
      "MFEM_USE_PUMI\n"
#endif
#ifdef MFEM_USE_CUDA
      "MFEM_USE_CUDA\n"
#endif
#ifdef MFEM_USE_RAJA
      "MFEM_USE_RAJA\n"
#endif
#ifdef MFEM_USE_UMPIRE
      "MFEM_USE_UMPIRE\n"
#endif
#ifdef MFEM_USE_OCCA
      "MFEM_USE_OCCA\n"
#endif
      "MFEM_TIMER_TYPE = " EXPAND_AND_QUOTE(MFEM_TIMER_TYPE)
      ;

   return config_str;
}

}
