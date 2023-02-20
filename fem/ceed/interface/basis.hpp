// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_LIBCEED_BASIS
#define MFEM_LIBCEED_BASIS

#include "ceed.hpp"

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED

/** @brief Initialize a CeedBasis for non-mixed meshes.

   @param[in] fes Input finite element space.
   @param[in] ir Input integration rule.
   @param[in] ceed Input Ceed object.
   @param[out] basis The address of the initialized CeedBasis object.
*/
void InitBasis(const FiniteElementSpace &fes,
               const IntegrationRule &ir,
               Ceed ceed, CeedBasis *basis);

/** @brief Initialize a CeedBasis for mixed meshes.

    @param[in] fes The finite element space.
    @param[in] ir is the integration rule for the operator.
    @param[in] nelem The number of elements.
    @param[in] indices The indices of the elements of same type in the
                       `FiniteElementSpace`.
    @param[in] ceed The Ceed object.
    @param[out] basis The `CeedBasis` to initialize. */
void InitBasisWithIndices(const FiniteElementSpace &fes,
                          const IntegrationRule &ir,
                          int nelem,
                          const int* indices,
                          Ceed ceed, CeedBasis *basis);

#endif

} // namespace ceed

} // namespace mfem

#endif // MFEM_LIBCEED_BASIS
