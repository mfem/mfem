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

#ifndef MFEM_LINALG
#define MFEM_LINALG

// Linear algebra header file

#include "vector.hpp"
#include "operator.hpp"
#include "matrix.hpp"
#include "sparsemat.hpp"
#include "complex_operator.hpp"
#include "blockvector.hpp"
#include "blockmatrix.hpp"
#include "blockoperator.hpp"
#include "sparsesmoothers.hpp"
#include "densemat.hpp"
#include "ode.hpp"
#include "solvers.hpp"
#include "handle.hpp"
#include "invariants.hpp"

#ifdef MFEM_USE_SUNDIALS
#include "sundials.hpp"
#endif

#ifdef MFEM_USE_HIOP
#include "hiop.hpp"
#endif

#ifdef MFEM_USE_GINKGO
#include "ginkgo.hpp"
#endif

#ifdef MFEM_USE_MPI
#include "hypre_parcsr.hpp"
#include "hypre.hpp"

#ifdef MFEM_USE_PETSC
#include "petsc.hpp"
#endif

#ifdef MFEM_USE_SLEPC
#include "slepc.hpp"
#endif

#ifdef MFEM_USE_SUPERLU
#include "superlu.hpp"
#endif

#ifdef MFEM_USE_STRUMPACK
#include "strumpack.hpp"
#endif

#endif // MFEM_USE_MPI

#endif
