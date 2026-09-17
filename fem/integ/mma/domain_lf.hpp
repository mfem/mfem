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

/** @file domain_lf.hpp
    DomainLF / VectorFE DomainLF PA MMA — IdentityLoad QFn + simplex Kernel decls.
*/

#include "../../lininteg.hpp"
#include "form/form.hpp"

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

/** Assemble DomainLF PA coefficient data for simplex MMA (defined in .cpp). */
void DLFEvalAssembleSimplexMma(const FiniteElementSpace &fes,
                               const IntegrationRule *ir,
                               const Array<int> &markers,
                               const Vector &coeff,
                               Vector &y);

/** Assemble VectorFE DomainLF PA data for simplex MMA (ND adj(J)f, RT J^T f). */
void VectorFEDLFAssembleSimplexMma(const FiniteElementSpace &fes,
                                   const IntegrationRule *ir,
                                   const Array<int> &markers,
                                   const Vector &coeff,
                                   Vector &y);

namespace internal::mma::form
{

/** Linear-form load at a quadrature point: y = d (no trial DOFs). */
struct IdentityLoad
{
   MFEM_HOST_DEVICE void operator()(eval_t &y, real_t d) const
   {
      y = d;
   }
};

template <>
struct qfn_traits<IdentityLoad> : NoneEvalQFnTraits {};

} // namespace internal::mma::form

template<int DIM, int D1D, int QND>
DomainLFIntegrator::AssembleSimplexMmaKernelType
DomainLFIntegrator::AssembleSimplexMmaKernels::Kernel()
{
   using internal::mma::form::ApplyLF;
   using internal::mma::form::IdentityLoad;
   if constexpr (DIM == 2)
   {
      return ApplyLF<IdentityLoad, 2, D1D, QND>;
   }
   else if constexpr (DIM == 3)
   {
      return ApplyLF<IdentityLoad, 3, D1D, QND>;
   }
   else
   {
      MFEM_ABORT("Simplex MMA DomainLF only supports DIM 2 or 3");
      return nullptr;
   }
}

inline DomainLFIntegrator::AssembleSimplexMmaKernelType
DomainLFIntegrator::AssembleSimplexMmaKernels::Fallback(int dim, int, int)
{
   using internal::mma::form::ApplyLF;
   using internal::mma::form::IdentityLoad;
   using Fn = AssembleSimplexMmaKernelType;
   MFEM_VERIFY(dim == 2 || dim == 3,
               "Simplex MMA DomainLF is only implemented for triangles/tets");
   if (dim == 2)
   {
      return static_cast<Fn>(ApplyLF<IdentityLoad, 2>);
   }
   return static_cast<Fn>(ApplyLF<IdentityLoad, 3>);
}

inline void ApplyVecLF(const int NE, const int nd, const int nq, const int sdim,
                       const Array<real_t> &B, const Vector &D, Vector &y)
{
   internal::mma::form::detail::SimplexVecLFApply(NE, nd, nq, sdim, B, D, y);
}

template<int DIM, int D1D, int QND>
VectorFEDomainLFIntegrator::AssembleSimplexMmaKernelType
VectorFEDomainLFIntegrator::AssembleSimplexMmaKernels::Kernel()
{
   static_assert(DIM == 2 || DIM == 3, "VectorFE simplex LF DIM 2 or 3");
   static_assert(D1D > 0 && QND > 0, "requires specialized D1D/QND");
   return ApplyVecLF;
}

inline VectorFEDomainLFIntegrator::AssembleSimplexMmaKernelType
VectorFEDomainLFIntegrator::AssembleSimplexMmaKernels::Fallback(int dim, int, int)
{
   MFEM_VERIFY(dim == 2 || dim == 3,
               "Simplex MMA VectorFE DomainLF is only implemented for triangles/tets");
   return ApplyVecLF;
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
