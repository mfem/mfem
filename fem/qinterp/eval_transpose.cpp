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

#include "../quadinterpolator.hpp"
#include "eval_transpose.hpp"

namespace mfem
{

/// @cond Suppress_Doxygen_warnings

QuadratureInterpolator::TensorEvalTransposeKernelType
QuadratureInterpolator::TensorEvalTransposeKernels::Fallback(
   int DIM, QVectorLayout Q_LAYOUT, int, int, int)
{
   using namespace internal::quadrature_interpolator;

   if (Q_LAYOUT == QVectorLayout::byNODES)
   {
      if (DIM == 1) { return ValuesTranspose1D<QVectorLayout::byNODES>; }
      else if (DIM == 2) { return ValuesTranspose2D<QVectorLayout::byNODES>; }
      else if (DIM == 3) { return ValuesTranspose3D<QVectorLayout::byNODES>; }
   }
   else
   {
      if (DIM == 1) { return ValuesTranspose1D<QVectorLayout::byVDIM>; }
      else if (DIM == 2) { return ValuesTranspose2D<QVectorLayout::byVDIM>; }
      else if (DIM == 3) { return ValuesTranspose3D<QVectorLayout::byVDIM>; }
   }
   MFEM_ABORT("Invalid dimension");
   return nullptr;
}

/// @endcond

} // namespace mfem
