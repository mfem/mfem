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

#include "eval_hdiv.hpp"

namespace mfem
{

namespace internal
{
namespace quadrature_interpolator
{

void InitTensorEvalHDivKernels()
{
   using k = QuadratureInterpolator::TensorEvalHDivKernels;
   constexpr auto LNODES = QVectorLayout::byNODES;
   constexpr auto LVDIM  = QVectorLayout::byVDIM;
   constexpr auto PV = QuadratureInterpolator::PHYSICAL_VALUES;
   constexpr auto PM = QuadratureInterpolator::PHYSICAL_MAGNITUDES;

   // Do not instantiate FLAGS = QuadratureInterpolator::VALUES, for now.

   // FLAGS = QuadratureInterpolator::PHYSICAL_VALUES:

   // 2D, QVectorLayout::byNODES
   k::Specialization<2,LNODES,PV,2,2>::Add(); // RT(0), 2^2 qpts
   // 2D, QVectorLayout::byVDIM
   k::Specialization<2,LVDIM, PV,2,2>::Add(); // RT(0), 2^2 qpts
   k::Specialization<2,LVDIM, PV,3,3>::Add(); // RT(1), 3^2 qpts
   k::Specialization<2,LVDIM, PV,4,4>::Add(); // RT(2), 4^2 qpts
   k::Specialization<2,LVDIM, PV,5,5>::Add(); // RT(3), 5^2 qpts

   // 3D, QVectorLayout::byNODES
   k::Specialization<3,LNODES,PV,2,3>::Add(); // RT(0), 3^3 qpts
   // 3D, QVectorLayout::byVDIM
   k::Specialization<3,LVDIM, PV,2,3>::Add(); // RT(0), 3^3 qpts
   k::Specialization<3,LVDIM, PV,3,4>::Add(); // RT(1), 4^3 qpts
   k::Specialization<3,LVDIM, PV,4,5>::Add(); // RT(2), 5^3 qpts
   k::Specialization<3,LVDIM, PV,5,6>::Add(); // RT(3), 6^3 qpts

   // FLAGS = QuadratureInterpolator::PHYSICAL_MAGNITUDES:

   // For vdim = 1: QVectorLayout::byNODES = QVectorLayout::byVDIM, so
   // we use just QVectorLayout::byNODES:
   // 2D
   k::Specialization<2,LNODES,PM,2,2>::Add(); // RT(0), 2^2 qpts
   k::Specialization<2,LNODES,PM,3,3>::Add(); // RT(1), 3^2 qpts
   k::Specialization<2,LNODES,PM,4,4>::Add(); // RT(2), 4^2 qpts
   k::Specialization<2,LNODES,PM,5,5>::Add(); // RT(3), 5^2 qpts

   // 3D
   k::Specialization<3,LNODES,PM,2,3>::Add(); // RT(0), 3^3 qpts
   k::Specialization<3,LNODES,PM,3,4>::Add(); // RT(1), 4^3 qpts
   k::Specialization<3,LNODES,PM,4,5>::Add(); // RT(2), 5^3 qpts
   k::Specialization<3,LNODES,PM,5,6>::Add(); // RT(3), 6^3 qpts
}

} // namespace quadrature_interpolator
} // namespace internal

/// @cond Suppress_Doxygen_warnings

QuadratureInterpolator::TensorEvalHDivKernelType
QuadratureInterpolator::TensorEvalHDivKernels::Fallback(
   int DIM, QVectorLayout Q_LAYOUT, unsigned FLAGS, int D1D, int Q1D)
{
   using namespace internal::quadrature_interpolator;
   MFEM_CONTRACT_VAR(D1D);
   MFEM_CONTRACT_VAR(Q1D);
   constexpr auto RV = QuadratureInterpolator::VALUES;
   constexpr auto PV = QuadratureInterpolator::PHYSICAL_VALUES;
   constexpr auto PM = QuadratureInterpolator::PHYSICAL_MAGNITUDES;
   if (DIM == 2)
   {
      if (FLAGS & RV)
      {
         return (Q_LAYOUT == QVectorLayout::byNODES) ?
                EvalHDiv2D<QVectorLayout::byNODES,RV> :
                EvalHDiv2D<QVectorLayout::byVDIM,RV>;
      }
      else if (FLAGS & PV)
      {
         return (Q_LAYOUT == QVectorLayout::byNODES) ?
                EvalHDiv2D<QVectorLayout::byNODES,PV> :
                EvalHDiv2D<QVectorLayout::byVDIM,PV>;
      }
      else
      {
         return EvalHDiv2D<QVectorLayout::byNODES,PM>;
      }
   }
   else if (DIM == 3)
   {
      if (FLAGS & RV)
      {
         return (Q_LAYOUT == QVectorLayout::byNODES) ?
                EvalHDiv3D<QVectorLayout::byNODES,RV> :
                EvalHDiv3D<QVectorLayout::byVDIM,RV>;
      }
      else if (FLAGS & PV)
      {
         return (Q_LAYOUT == QVectorLayout::byNODES) ?
                EvalHDiv3D<QVectorLayout::byNODES,PV> :
                EvalHDiv3D<QVectorLayout::byVDIM,PV>;
      }
      else
      {
         return EvalHDiv3D<QVectorLayout::byNODES,PM>;
      }
   }
   MFEM_ABORT("DIM = " << DIM << " is not implemented!");
}

/// @endcond

} // namespace mfem
