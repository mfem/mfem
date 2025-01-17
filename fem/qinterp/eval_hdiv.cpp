// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

   // Do not instantiate PHYS = false, for now.

   // PHYS = true
   // 2D, QVectorLayout::byNODES
   k::Specialization<2,LNODES,1,2,2>::Add(); // RT(0), 2^2 qpts
   // 2D, QVectorLayout::byVDIM
   k::Specialization<2,LVDIM,1,2,2>::Add();  // RT(0), 2^2 qpts
   k::Specialization<2,LVDIM,1,3,3>::Add();  // RT(1), 3^2 qpts
   k::Specialization<2,LVDIM,1,4,4>::Add();  // RT(2), 4^2 qpts
   k::Specialization<2,LVDIM,1,5,5>::Add();  // RT(3), 5^2 qpts

   // 3D, QVectorLayout::byNODES
   k::Specialization<3,LNODES,1,2,3>::Add(); // RT(0), 3^3 qpts
   // 3D, QVectorLayout::byVDIM
   k::Specialization<3,LVDIM, 1,2,3>::Add(); // RT(0), 3^3 qpts
   k::Specialization<3,LVDIM, 1,3,4>::Add(); // RT(1), 4^3 qpts
   k::Specialization<3,LVDIM, 1,4,5>::Add(); // RT(2), 5^3 qpts
   k::Specialization<3,LVDIM, 1,5,6>::Add(); // RT(3), 6^3 qpts
}

} // namespace quadrature_interpolator
} // namespace internal

QuadratureInterpolator::TensorEvalHDivKernelType
QuadratureInterpolator::TensorEvalHDivKernels::Fallback(
   int DIM, QVectorLayout Q_LAYOUT, bool PHYS, int D1D, int Q1D)
{
   using namespace internal::quadrature_interpolator;
   MFEM_CONTRACT_VAR(D1D);
   MFEM_CONTRACT_VAR(Q1D);
   if (DIM == 2)
   {
      if (PHYS)
      {
         return (Q_LAYOUT == QVectorLayout::byNODES) ?
                EvalHDiv2D<QVectorLayout::byNODES,true> :
                EvalHDiv2D<QVectorLayout::byVDIM,true>;
      }
      else
      {
         return (Q_LAYOUT == QVectorLayout::byNODES) ?
                EvalHDiv2D<QVectorLayout::byNODES,false> :
                EvalHDiv2D<QVectorLayout::byVDIM,false>;
      }
   }
   else if (DIM == 3)
   {
      if (PHYS)
      {
         return (Q_LAYOUT == QVectorLayout::byNODES) ?
                EvalHDiv3D<QVectorLayout::byNODES,true> :
                EvalHDiv3D<QVectorLayout::byVDIM,true>;
      }
      else
      {
         return (Q_LAYOUT == QVectorLayout::byNODES) ?
                EvalHDiv3D<QVectorLayout::byNODES,false> :
                EvalHDiv3D<QVectorLayout::byVDIM,false>;
      }
   }
   MFEM_ABORT("DIM = " << DIM << " is not implemented!");
}

} // namespace mfem
