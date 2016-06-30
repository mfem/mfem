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

#ifndef MFEM_LINALG
#define MFEM_LINALG

// Linear algebra header file

#include "vector.hpp"
#include "operator.hpp"
#include "matrix.hpp"
#include "sparsemat.hpp"
#include "blockvector.hpp"
#include "blockmatrix.hpp"
#include "blockoperator.hpp"
#include "sparsesmoothers.hpp"
#include "densemat.hpp"
#include "ode.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#include "hypre.hpp"
#endif

#include "solvers.hpp"
#include "superlu.hpp"

#endif
