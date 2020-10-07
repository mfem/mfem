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

#ifndef MFEM_CEEDSOLVERS_ATPMG_H
#define MFEM_CEEDSOLVERS_ATPMG_H

#include "../../config/config.hpp"

#ifdef MFEM_USE_CEED

#include <ceed.h>

int CeedATPMGElemRestriction(int order,
                             int order_reduction,
                             CeedElemRestriction er_in,
                             CeedElemRestriction* er_out,
                             CeedInt *&dof_map);

// Create coarse-to-fine basis, given number of input nodes and order reduction.
// Assumes Gauss-Lobatto basis. This is useful because it does not require an
// input CeedBasis object, which depends on choice of quadrature rule, whereas
// the coarse-to-fine operator is independent of quadrature.
int CeedBasisATPMGCToF(Ceed ceed, int P1d, int dim, int order_reduction,
                       CeedBasis *basisc2f);

int CeedBasisATPMGCoarsen(CeedBasis basisin, CeedBasis* basisout,
                          CeedBasis* basis_ctof,
                          int order_reduction);

int CeedATPMGOperator(CeedOperator oper, int order_reduction,
                      CeedElemRestriction coarse_er,
                      CeedBasis* coarse_basis_out,
                      CeedBasis* basis_ctof_out,
                      CeedOperator* out);

/**
   Intended to encapsulate some of the above; we could consider a whole
   Ceed object that would create and destroy this stuff.
*/
int CeedATPMGBundle(CeedOperator oper, int order_reduction,
                    CeedBasis* coarse_basis_out,
                    CeedBasis* basis_ctof_out,
                    CeedElemRestriction* er_out,
                    CeedOperator* coarse_oper,
                    CeedInt *&dof_map);

#endif // MFEM_USE_CEED

#endif // include guard
