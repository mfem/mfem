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

#include "det.hpp"

namespace mfem
{

namespace internal
{
namespace quadrature_interpolator
{

void InitDetKernels()
{
   using k = QuadratureInterpolator::DetKernels;
   // 2D
   k::Specialization<2,2,2,2>::Add();
   k::Specialization<2,2,2,3>::Add();
   k::Specialization<2,2,2,4>::Add();
   k::Specialization<2,2,2,6>::Add();
   k::Specialization<2,2,3,4>::Add();
   k::Specialization<2,2,3,6>::Add();
   k::Specialization<2,2,4,4>::Add();
   k::Specialization<2,2,4,6>::Add();
   k::Specialization<2,2,5,6>::Add();
   // 3D
   k::Specialization<3,3,2,4>::Add();
   k::Specialization<3,3,3,3>::Add();
   k::Specialization<3,3,3,5>::Add();
   k::Specialization<3,3,3,6>::Add();
   k::Specialization<3,3,4,6>::Add();
   k::Specialization<3,3,3,4>::Add();
}

} // namespace quadrature_interpolator
} // namespace internal

/// @cond Suppress_Doxygen_warnings

QuadratureInterpolator::DetKernelType
QuadratureInterpolator::DetKernels::Fallback(
   int DIM, int SDIM, int D1D, int Q1D)
{
   if (DIM == 1)
   {
      if (SDIM == 1) { return internal::quadrature_interpolator::Det1D; }
      else if (SDIM == 2) { return internal::quadrature_interpolator::Det1DSurface<0,0,2>; }
      else if (SDIM == 3) { return internal::quadrature_interpolator::Det1DSurface<0,0,3>; }
      else { MFEM_ABORT(""); }
   }
   else if (DIM == 2 && SDIM == 2) { return internal::quadrature_interpolator::Det2D; }
   else if (DIM == 2 && SDIM == 3) { return internal::quadrature_interpolator::Det2DSurface; }
   else if (DIM == 3)
   {
      const int MD = DeviceDofQuadLimits::Get().MAX_DET_1D;
      const int MQ = DeviceDofQuadLimits::Get().MAX_DET_1D;
      if (D1D <= MD && Q1D <= MQ) { return internal::quadrature_interpolator::Det3D<0,0,true>; }
      else { return internal::quadrature_interpolator::Det3D<0,0,false>; }
   }
   else { MFEM_ABORT(""); }
}

/// @endcond

} // namespace mfem
