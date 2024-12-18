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

<<<<<<<< HEAD:fem/tmop/tools/2d/det_jpr.cpp
#include "../../pa.hpp"
#include "../../../tmop.hpp"
#include "../../../tmop_tools.hpp"
#include "../../../kernels.hpp"
#include "../../../../general/forall.hpp"
#include "../../../../linalg/kernels.hpp"
========
#include "../pa.hpp"
#include "../../tmop.hpp"
#include "../../tmop_tools.hpp"
#include "../../kernels.hpp"
#include "../../../general/forall.hpp"
#include "../../../linalg/kernels.hpp"
>>>>>>>> main:fem/tmop/tools/det2_jpr.cpp

namespace mfem
{

template <int T_D1D = 0, int T_Q1D = 0, int T_MAX = 4>
void TMOP_MinDetJpr_2D(const int NE,
                       const ConstDeviceMatrix &B,
                       const ConstDeviceMatrix &G,
                       const DeviceTensor<4, const real_t> &X,
                       DeviceTensor<3> &E,
                       const int d1d,
                       const int q1d,
                       const int max)
{
   constexpr int NBZ = 1;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ, [=] MFEM_HOST_DEVICE(int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED real_t BG[2][MQ1 * MD1];
      MFEM_SHARED real_t XY[2][NBZ][MD1 * MD1];
      MFEM_SHARED real_t DQ[4][NBZ][MD1 * MQ1];
      MFEM_SHARED real_t QQ[4][NBZ][MQ1 * MQ1];

      kernels::internal::LoadX<MD1, NBZ>(e, D1D, X, XY);
      kernels::internal::LoadBG<MD1, MQ1>(D1D, Q1D, B, G, BG);

      kernels::internal::GradX<MD1, MQ1, NBZ>(D1D, Q1D, BG, XY, DQ);
      kernels::internal::GradY<MD1, MQ1, NBZ>(D1D, Q1D, BG, DQ, QQ);

      MFEM_FOREACH_THREAD(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD(qx, x, Q1D)
         {
            real_t J[4];
            kernels::internal::PullGrad<MQ1, NBZ>(Q1D, qx, qy, QQ, J);
            E(qx, qy, e) = kernels::Det<2>(J);
         }
      }
   });
}

TMOP_REGISTER_KERNELS(TMOPMinDetJpr2D, TMOP_MinDetJpr_2D);

real_t TMOPNewtonSolver::MinDetJpr_2D(const FiniteElementSpace *fes,
                                      const Vector &x) const
{
   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *R = fes->GetElementRestriction(ordering);
   Vector xe(R->Height(), Device::GetDeviceMemoryType());
   xe.UseDevice(true);
   R->Mult(x, xe);

   const DofToQuad &maps = fes->GetFE(0)->GetDofToQuad(ir, DofToQuad::TENSOR);
   const int NE = fes->GetMesh()->GetNE(), NQ = ir.GetNPoints();
   const int d = maps.ndof, q = maps.nqpt;

   constexpr int DIM = 2;
   const auto B = Reshape(maps.B.Read(), q, d);
   const auto G = Reshape(maps.G.Read(), q, d);
   const auto XE = Reshape(xe.Read(), d, d, DIM, NE);

   Vector e(NE * NQ);
   e.UseDevice(true);
   auto E = Reshape(e.Write(), q, q, NE);

   const static auto specialized_kernels = []
   { return tmop::KernelSpecializations<TMOPMinDetJpr2D>(); }();

   TMOPMinDetJpr2D::Run(d, q, NE, B, G, XE, E, d, q, 4);
   return e.Min();
}

} // namespace mfem
