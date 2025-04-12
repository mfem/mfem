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

#include "../tmop.hpp"
#include "tmop_pa.hpp"
#include "../tmop_tools.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"

namespace mfem
{

MFEM_REGISTER_TMOP_KERNELS(real_t, MinDetJpr_Kernel_3D,
                           const int NE,
                           const Array<real_t> &b_,
                           const Array<real_t> &g_,
                           const Vector &x_,
                           Vector &DetJ,
                           const int d1d,
                           const int q1d)
{
   constexpr int DIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto X = Reshape(x_.Read(), D1D, D1D, D1D, DIM, NE);

   auto E = Reshape(DetJ.Write(), Q1D, Q1D, Q1D, NE);

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED real_t BG[2][MQ1*MD1];
      MFEM_SHARED real_t DDD[3][MD1*MD1*MD1];
      MFEM_SHARED real_t DDQ[6][MD1*MD1*MQ1];
      MFEM_SHARED real_t DQQ[9][MD1*MQ1*MQ1];
      MFEM_SHARED real_t QQQ[9][MQ1*MQ1*MQ1];

      kernels::internal::LoadX<MD1>(e,D1D,X,DDD);
      kernels::internal::LoadBG<MD1,MQ1>(D1D,Q1D,b,g,BG);

      kernels::internal::GradX<MD1,MQ1>(D1D,Q1D,BG,DDD,DDQ);
      kernels::internal::GradY<MD1,MQ1>(D1D,Q1D,BG,DDQ,DQQ);
      kernels::internal::GradZ<MD1,MQ1>(D1D,Q1D,BG,DQQ,QQQ);

      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t J[9];
               kernels::internal::PullGrad<MQ1>(Q1D,qx,qy,qz,QQQ,J);
               E(qx,qy,qz,e) = kernels::Det<3>(J);
            }
         }
      }
   });

   return DetJ.Min();
}

real_t TMOPNewtonSolver::MinDetJpr_3D(const FiniteElementSpace *fes,
                                      const Vector &D) const
{
   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;

   const Operator *RD = fes->GetElementRestriction(ordering);
   Vector DE(RD->Height(), Device::GetDeviceMemoryType());
   DE.UseDevice(true);
   RD->Mult(D, DE);

   const Operator *RX = x_0.FESpace()->GetElementRestriction(ordering);
   Vector XE(RX->Height(), Device::GetDeviceMemoryType());
   XE.UseDevice(true);
   RX->Mult(x_0, XE);
   XE += DE;

   auto maps = fes->GetTypicalFE()->GetDofToQuad(ir, DofToQuad::TENSOR);
   const int NE = fes->GetMesh()->GetNE();
   const int NQ = ir.GetNPoints();
   const int D1D = maps.ndof;
   const int Q1D = maps.nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const Array<real_t> &B = maps.B;
   const Array<real_t> &G = maps.G;

   Vector E(NE*NQ);
   E.UseDevice(true);

   MFEM_LAUNCH_TMOP_KERNEL(MinDetJpr_Kernel_3D,id,NE,B,G,XE,E);
}

} // namespace mfem
