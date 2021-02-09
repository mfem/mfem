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


#ifndef MFEM_CEEDSOLVERS_SPARSIFY_H
#define MFEM_CEEDSOLVERS_SPARSIFY_H

#include "../../config/config.hpp"

#ifdef MFEM_USE_CEED

#include <ceed.h>

typedef enum {
  SPARSIFY_LARGEST_ABS = 0,
  SPARSIFY_LARGEST_POSITIVE = 1,
  SPARSIFY_NEARBY = 2,
  SPARSIFY_LARGEST_GRAD_ABS = 3,
} SparsifySelectionStrategy;

/**
   For each (fixed) quadrature point, you sort the dof indices
   based on the value of the basis function corresponding to
   that dof. Then you zero the basis function and the gradient
   for all but the largest <parameter> values.

   basisin must be a tensor basis, ie, have 1d tensors available

   scaling is to ensure that rows of interp1d have rowsum == 1,
   and to ensure that rows of grad1d have rowsum == 0
*/
int CeedBasisSparsifyScaling(CeedBasis basisin, CeedBasis* basisout,
                             SparsifySelectionStrategy sel_strategy,
                             int parameter);

/**
   This is all modeled after H1 operators, do not expect it to work
   on anything else. You also must have some symmetry in the basis
   (input/output)

   The only original thing happens in (basis_sparsify); this tries to
   just copy the pointers/data/etc for everything else.

   Note well that if you ask me to sparsify a different operator, mass
   or something else, this will all fall apart.

   Caller is responsible for deleting sparse_basis_out and out
   out is what you actually want to use, I only return sparse_basis_out
   so you can delete it cleanly.
*/
int CeedSparsifyH1Operator(CeedOperator oper, int sparse_parameter,
                           SparsifySelectionStrategy sel_strategy,
                           int (*basis_sparsify)(CeedBasis, CeedBasis*, SparsifySelectionStrategy, int),
                           CeedBasis* sparse_basis_out,
                           CeedOperator* out);

#endif // MFEM_USE_CEED

#endif // include guard
