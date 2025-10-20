// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_PETSCINTERNALS
#define MFEM_PETSCINTERNALS

#include "../general/error.hpp"
#include "petsc.h"

// Error handling
// Prints PETSc's stacktrace and then calls MFEM_ABORT
// We cannot use PETSc's CHKERRQ since it returns a PetscErrorCode
#define PCHKERRQ(obj,err) do {                                                         \
     if ((err))                                                                        \
     {                                                                                 \
        (void)PetscError(PetscObjectComm((PetscObject)(obj)),__LINE__,_MFEM_FUNC_NAME, \
                   __FILE__,(err),PETSC_ERROR_REPEAT,NULL);                            \
        MFEM_ABORT("Error in PETSc. See stacktrace above.");                           \
     }                                                                                 \
  } while(0);
#define CCHKERRQ(comm,err) do {                                             \
     if ((err))                                                             \
     {                                                                      \
        (void)PetscError(comm,__LINE__,_MFEM_FUNC_NAME,                     \
                   __FILE__,(PetscErrorCode)(err),PETSC_ERROR_REPEAT,NULL); \
        MFEM_ABORT("Error in PETSc. See stacktrace above.");                \
     }                                                                      \
  } while(0);

#endif
