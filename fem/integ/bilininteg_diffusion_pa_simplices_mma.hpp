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
#pragma once

#include "../bilininteg.hpp"
#include "mma/form/form.hpp"

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal
{

/** Simplex diffusion PA setup from mesh nodes (defined in .cpp). */
void PADiffusionSetupSimplexFromNodes(const int dim,
                                      const int coeffDim,
                                      const int NE,
                                      const int NQ,
                                      const int ND,
                                      const Array<real_t> &w,
                                      const Array<real_t> &g,
                                      const Vector &nodes_e,
                                      const Vector &c,
                                      Vector &d);

} // namespace internal

namespace internal::mma::form
{

/** Diffusion metric matvec at a quadrature point: y = A * u. */
template <int DIM, bool SYM = true>
struct DiffusionMetric
{
   static constexpr bool symmetric_pa = SYM;

   MFEM_HOST_DEVICE void operator()(const grad_t<DIM> &u, grad_t<DIM> &y,
                                    const tensor<real_t, DIM, DIM> &A) const
   {
      y = A * u;
   }
};

template <int DIM, bool SYM>
struct qfn_traits<DiffusionMetric<DIM, SYM>> : GradGradQFnTraits<DIM, SYM> {};

/** Dispatch on runtime `symmetric` for DiffusionIntegrator registration. */
template <int DIM, int D1D, int QND>
inline void ApplyDiffusionDispatch(const int NE,
                                   const bool symmetric,
                                   const Array<real_t> &g,
                                   const Vector &d,
                                   const Vector &x,
                                   Vector &y)
{
   if (symmetric)
   {
      Apply<DiffusionMetric<DIM, true>, DIM, D1D, QND>(NE, g, d, x, y);
   }
   else
   {
      Apply<DiffusionMetric<DIM, false>, DIM, D1D, QND>(NE, g, d, x, y);
   }
}

template <int DIM>
inline void ApplyDiffusionDispatch(const int NE,
                                   const bool symmetric,
                                   const Array<real_t> &g,
                                   const Vector &d,
                                   const Vector &x,
                                   Vector &y)
{
   if (symmetric)
   {
      Apply<DiffusionMetric<DIM, true>, DIM>(NE, g, d, x, y);
   }
   else
   {
      Apply<DiffusionMetric<DIM, false>, DIM>(NE, g, d, x, y);
   }
}

} // namespace internal::mma::form

// NB/Q-tile policy: mma/batch.hpp (used by form plans).

template<int DIM, int D1D, int QND>
DiffusionIntegrator::ApplySimplexMmaKernelType
DiffusionIntegrator::ApplySimplexMmaPAKernels::Kernel()
{
   using internal::mma::form::ApplyDiffusionDispatch;
   if constexpr (DIM == 2)
   {
      return ApplyDiffusionDispatch<2, D1D, QND>;
   }
   else if constexpr (DIM == 3)
   {
      return ApplyDiffusionDispatch<3, D1D, QND>;
   }
   else
   {
      MFEM_ABORT("Simplex MMA diffusion only supports DIM 2 or 3");
      return nullptr;
   }
}

inline DiffusionIntegrator::ApplySimplexMmaKernelType
DiffusionIntegrator::ApplySimplexMmaPAKernels::Fallback(int dim, int, int)
{
   using internal::mma::form::ApplyDiffusionDispatch;
   using Fn = ApplySimplexMmaKernelType;
   MFEM_VERIFY(dim == 2 || dim == 3,
               "Simplex MMA diffusion PA is only implemented for triangles/tets");
   if (dim == 2)
   {
      return static_cast<Fn>(ApplyDiffusionDispatch<2>);
   }
   return static_cast<Fn>(ApplyDiffusionDispatch<3>);
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
