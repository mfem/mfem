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
#include "mma/mma.hpp"
#include "mma/form/pipeline.hpp"

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal
{
namespace mma
{
namespace form
{

/** Mass density scale at a quadrature point: y = d * u. */
struct MassScale
{
   MFEM_HOST_DEVICE void operator()(const eval_t &u, eval_t &y, real_t d) const
   {
      y = d * u;
   }
};

template <>
struct qfn_traits<MassScale> : EvalEvalQFnTraits {};

} // namespace form
} // namespace mma
} // namespace internal

// AssembleSimplexMmaPA / RegisterSimplexMmaKernels live in the .cpp.

template<int DIM, int D1D, int QND>
MassIntegrator::ApplySimplexMmaKernelType
MassIntegrator::ApplySimplexMmaPAKernels::Kernel()
{
   using internal::mma::form::Apply;
   using internal::mma::form::MassScale;
   if constexpr (DIM == 2)
   {
      return Apply<MassScale, 2, D1D, QND>;
   }
   else if constexpr (DIM == 3)
   {
      return Apply<MassScale, 3, D1D, QND>;
   }
   else
   {
      MFEM_ABORT("Simplex MMA mass only supports DIM 2 or 3");
      return nullptr;
   }
}

inline MassIntegrator::ApplySimplexMmaKernelType
MassIntegrator::ApplySimplexMmaPAKernels::Fallback(int dim, int, int)
{
   using internal::mma::form::Apply;
   using internal::mma::form::MassScale;
   using Fn = ApplySimplexMmaKernelType;
   MFEM_VERIFY(dim == 2 || dim == 3,
               "Simplex MMA mass PA is only implemented for triangles/tets");
   if (dim == 2)
   {
      return static_cast<Fn>(Apply<MassScale, 2>);
   }
   return static_cast<Fn>(Apply<MassScale, 3>);
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
