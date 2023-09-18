// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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

#include "../../fespace.hpp"
#include "ceed.hpp"

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED

/** @brief Initialize a CeedBasis based on an mfem::FiniteElementSpace @a fes,
    an mfem::FiniteElement @a fe, and an mfem::IntegrationRule @a ir.

    @param[in] fes The finite element space.
    @param[in] fe The finite element.
    @param[in] ir The integration rule.
    @param[in] ceed The Ceed object.
    @param[out] basis The `CeedBasis` to initialize. */
void InitBasis(const FiniteElementSpace &fes,
               const FiniteElement &fe,
               const IntegrationRule &ir,
               Ceed ceed,
               CeedBasis *basis);

/** @brief Initialize a CeedBasis based on an mfem::FiniteElementSpace @a fes,
    an mfem::IntegrationRule @a ir, and an optional list of element indices
    @a indices.

    @param[in] fes The finite element space.
    @param[in] ir The integration rule.
    @param[in] use_bdr Create the basis and restriction for boundary elements.
    @param[in] indices The indices of the elements of same type in the
                       `FiniteElementSpace`. If `indices == nullptr`, assumes
                       that the `FiniteElementSpace` is not mixed.
    @param[in] ceed The Ceed object.
    @param[out] basis The `CeedBasis` to initialize. */
inline void InitBasis(const FiniteElementSpace &fes,
                      const IntegrationRule &ir,
                      bool use_bdr,
                      const int *indices,
                      Ceed ceed,
                      CeedBasis *basis)
{
   const mfem::FiniteElement *fe;
   if (indices)
   {
      fe = use_bdr ? fes.GetBE(indices[0]) : fes.GetFE(indices[0]);
   }
   else
   {
      fe = use_bdr ? fes.GetBE(0) : fes.GetFE(0);
   }
   InitBasis(fes, *fe, ir, ceed, basis);
}

inline void InitBasis(const FiniteElementSpace &fes,
                      const IntegrationRule &ir,
                      bool use_bdr,
                      Ceed ceed,
                      CeedBasis *basis)
{
   InitBasis(fes, ir, use_bdr, nullptr, ceed, basis);
}

/** @brief Initialize a CeedBasis based on an interpolation from
    mfem::FiniteElementSpace @a trial_fes to @a test_fes. The type of
    interpolation will be chosen based on the map type of the provided
    mfem::FiniteElement objects.

    @param[in] trial_fes The trial finite element space.
    @param[in] test_fes The test finite element space.
    @param[in] trial_fe The trial finite element.
    @param[in] test_fe The test finite element.
    @param[in] ceed The Ceed object.
    @param[out] basis The `CeedBasis` to initialize. */
void InitInterpolatorBasis(const FiniteElementSpace &trial_fes,
                           const FiniteElementSpace &test_fes,
                           const FiniteElement &trial_fe,
                           const FiniteElement &test_fe,
                           Ceed ceed,
                           CeedBasis *basis);

/** @brief Initialize a CeedBasis based on an interpolation from
    mfem::FiniteElementSpace @a trial_fes to @a test_fes, with an optional list
    of element indices @a indices. The type of interpolation will be chosen
    based on the map type of the provided spaces.

    @param[in] trial_fes The trial finite element space.
    @param[in] test_fes The test finite element space.
    @param[in] indices The indices of the elements of same type in the
                       `FiniteElementSpace`. If `indices == nullptr`, assumes
                       that the `FiniteElementSpace` is not mixed.
    @param[in] ceed The Ceed object.
    @param[out] basis The `CeedBasis` to initialize. */
inline void InitInterpolatorBasis(const FiniteElementSpace &trial_fes,
                                  const FiniteElementSpace &test_fes,
                                  const int *indices,
                                  Ceed ceed,
                                  CeedBasis *basis)
{
   const int first_index = indices ? indices[0] : 0;
   const mfem::FiniteElement &trial_fe = *trial_fes.GetFE(first_index);
   const mfem::FiniteElement &test_fe = *test_fes.GetFE(first_index);
   InitInterpolatorBasis(trial_fes, test_fes, trial_fe, test_fe, ceed, basis);
}

inline void InitInterpolatorBasis(const FiniteElementSpace &trial_fes,
                                  const FiniteElementSpace &test_fes,
                                  Ceed ceed,
                                  CeedBasis *basis)
{
   InitInterpolatorBasis(trial_fes, test_fes, nullptr, ceed, basis);
}

#endif

} // namespace ceed

} // namespace mfem

#endif // MFEM_LIBCEED_BASIS
