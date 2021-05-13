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

#include "../tmop.hpp"
#include "tmop_pa.hpp"
#include "../gridfunc.hpp"
#include "../kernels.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"

using namespace mfem;

namespace mfem
{

MFEM_REGISTER_TMOP_KERNELS(bool, TC_IDEAL_SHAPE_UNIT_SIZE_2D_KERNEL,
                           const int NE,
                           const DenseMatrix w_, // copy
                           DenseTensor &j_,
                           const int d1d,
                           const int q1d)
{
   constexpr int DIM = 2;
   constexpr int NBZ = 1;

   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto W = Reshape(w_.Read(), DIM,DIM);
   auto J = Reshape(j_.Write(), DIM,DIM, Q1D,Q1D, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            kernels::Set(DIM,DIM, 1.0, &W(0,0), &J(0,0,qx,qy,e));
         }
      }
   });
   return true;
}

MFEM_REGISTER_TMOP_KERNELS(bool, TC_IDEAL_SHAPE_GIVEN_SIZE_2D_KERNEL,
                           const int NE,
                           const Array<double> &b_,
                           const Array<double> &g_,
                           const DenseMatrix w_ideal_, // copy
                           const Vector &x_,
                           DenseTensor &j_,
                           const int d1d,
                           const int q1d)
{
   constexpr int DIM = 2;
   constexpr int NBZ = 1;

   const double detW = w_ideal_.Det();
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto Wideal = Reshape(w_ideal_.Read(), DIM,DIM);
   const auto X = Reshape(x_.Read(), D1D, D1D, DIM, NE);
   auto J = Reshape(j_.Write(), DIM,DIM, Q1D,Q1D, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      constexpr int NBZ = 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED double BG[2][MQ1*MD1];
      MFEM_SHARED double XY[2][NBZ][MD1*MD1];
      MFEM_SHARED double DQ[4][NBZ][MD1*MQ1];
      MFEM_SHARED double QQ[4][NBZ][MQ1*MQ1];

      kernels::internal::LoadX<MD1,NBZ>(e,D1D,X,XY);
      kernels::internal::LoadBG<MD1,MQ1>(D1D,Q1D,b,g,BG);

      kernels::internal::GradX<MD1,MQ1,NBZ>(D1D,Q1D,BG,XY,DQ);
      kernels::internal::GradY<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ,QQ);

      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double Jtr[4];
            const double *Wid = &Wideal(0,0);
            kernels::internal::PullGrad<MQ1,NBZ>(Q1D,qx,qy,QQ,Jtr);
            const double detJ = kernels::Det<2>(Jtr);
            const double alpha = std::pow(detJ/detW,1./2);
            kernels::Set(DIM,DIM,alpha,Wid,&J(0,0,qx,qy,e));
         }
      }
   });
   return true;
}

template<> bool
TargetConstructor::ComputeElementTargetsPA<2>(const FiniteElementSpace *fes,
                                              const IntegrationRule *ir,
                                              DenseTensor &Jtr,
                                              const Vector&) const
{
   MFEM_ASSERT(target_type == IDEAL_SHAPE_UNIT_SIZE || nodes != nullptr, "");
   MFEM_VERIFY(fes->GetFE(0)->GetGeomType() == Geometry::SQUARE, "");
   const DenseMatrix &W = Geometries.GetGeomToPerfGeomJac(Geometry::SQUARE);
   const FiniteElement *fe = fes->GetFE(0);
   const int NE = fes->GetMesh()->GetNE();
   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   const DofToQuad &maps = fe->GetDofToQuad(*ir, mode);
   const int D1D = maps.ndof;
   const int Q1D = maps.nqpt;
   const int id = (D1D << 4 ) | Q1D;

   const Array<double> &B = maps.B;
   const Array<double> &G = maps.G;

   switch (target_type)
   {
      case IDEAL_SHAPE_UNIT_SIZE: // Jtr(i) = Wideal;
      {
         MFEM_LAUNCH_TMOP_KERNEL(TC_IDEAL_SHAPE_UNIT_SIZE_2D_KERNEL,
                                 id,NE,W,Jtr);
      }
      case IDEAL_SHAPE_EQUAL_SIZE: return false;
      case IDEAL_SHAPE_GIVEN_SIZE:
      {
         MFEM_VERIFY(nodes, "");
         const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
         const Operator *R = fes->GetElementRestriction(ordering);
         Vector X(R->Height(), Device::GetDeviceMemoryType());
         X.UseDevice(true);
         R->Mult(*nodes, X);
         MFEM_ASSERT(nodes->FESpace()->GetVDim() == 2, "");
         MFEM_LAUNCH_TMOP_KERNEL(TC_IDEAL_SHAPE_GIVEN_SIZE_2D_KERNEL,
                                 id,NE,B,G,W,X,Jtr);
      }
      case GIVEN_SHAPE_AND_SIZE: return false;
      default: return false;
   }
   return false;
}

} // namespace mfem
