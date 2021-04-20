// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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

namespace mfem
{

namespace ceed
{

/** @brief Take given (high-order) CeedElemRestriction and make a new
    CeedElemRestriction, which corresponds to a lower-order problem.

    Assumes a Gauss-Lobatto basis and tensor product elements, and assumes that
    the nodes in er_in are ordered in a tensor-product way.

    This is a setup routine that operates on the host.

    The caller is responsible for freeing er_out and dof_map. */
int CeedATPMGElemRestriction(int order,
                             int order_reduction,
                             CeedElemRestriction er_in,
                             CeedElemRestriction* er_out,
                             CeedInt *&dof_map);

/** @brief Create coarse-to-fine basis, given number of input nodes and order
    reduction.

    Assumes Gauss-Lobatto basis. This is useful because it does not require an
    input CeedBasis object, which depends on choice of quadrature rule, whereas
    the coarse-to-fine operator is independent of quadrature. */
int CeedBasisATPMGCoarseToFine(Ceed ceed, int P1d, int dim, int order_reduction,
                               CeedBasis *basisc2f);

/** @brief Given basis basisin, reduces its order by order_reduction and return
    basisout (which has the same height (Q1d) but is narrower (smaller P1d))

    The algorithm takes the locations of the fine nodes as input, but this
    particular implementation simply assumes Gauss-Lobatto, and furthermore
    assumes the MFEM [0, 1] reference element (rather than the Ceed/Petsc [-1,
    1] element) */
int CeedBasisATPMGCoarsen(CeedBasis basisin, CeedBasis* basisout,
                          CeedBasis* basis_ctof,
                          int order_reduction);

/** @brief Coarsen a CeedOperator using semi-algebraic p-multigrid

    This implementation does not coarsen the integration points at all.

    @param[in] oper              the operator to coarsen
    @param[in] order_reduction   how much to coarsen (order p)
    @param[in] coarse_er         CeedElemRestriction for coarse operator
                                 (see CeedATPMGElemRestriction)
    @param[out] coarse_basis_out CeedBasis for coarser operator
    @param[out] out              coarsened CeedOperator
*/
int CeedATPMGOperator(CeedOperator oper, int order_reduction,
                      CeedElemRestriction coarse_er,
                      CeedBasis* coarse_basis_out,
                      CeedBasis* basis_ctof_out,
                      CeedOperator* out);

/** @brief Given (fine) CeedOperator, produces everything you need for a coarse
    level (operator and interpolation).

    @param[in]  oper             Fine CeedOperator to coarsen
    @param[in]  order_reduction  Amount to reduce the order (p) of the operator
    @param[out] coarse_basis_out CeedBasis for coarse operator
    @param[out] basis_ctof_out   CeedBasis describing interpolation from coarse to fine
    @param[out] er_out           CeedElemRestriction for coarse operator
    @param[out] coarse_oper      coarse operator itself
    @param[out] dof_map          maps high-order ldof to low-order ldof, needed for
                                 further coarsening
*/
int CeedATPMGBundle(CeedOperator oper, int order_reduction,
                    CeedBasis* coarse_basis_out,
                    CeedBasis* basis_ctof_out,
                    CeedElemRestriction* er_out,
                    CeedOperator* coarse_oper,
                    CeedInt *&dof_map);

} // namespace ceed

} // namespace mfem

#endif // MFEM_USE_CEED

#endif // include guard
