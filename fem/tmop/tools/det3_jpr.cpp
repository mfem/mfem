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

#include "../../tmop.hpp"
#include "../../tmop_tools.hpp"
#include "../../kernels.hpp"
#include "../../../general/forall.hpp"
#include "../../../linalg/kernels.hpp"

namespace mfem
{

template <int T_D1D = 0, int T_Q1D = 0, int T_MAX = 4>
void TMOP_MinDetJpr_3D(const int NE,
                       const ConstDeviceMatrix &B,
                       const ConstDeviceMatrix &G,
                       const DeviceTensor<5, const real_t> &X,
                       DeviceTensor<4> &E,
                       const int d1d,
                       const int q1d,
                       const int max)
{
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED real_t BG[2][MQ1 * MD1];
      MFEM_SHARED real_t DDD[3][MD1 * MD1 * MD1];
      MFEM_SHARED real_t DDQ[6][MD1 * MD1 * MQ1];
      MFEM_SHARED real_t DQQ[9][MD1 * MQ1 * MQ1];
      MFEM_SHARED real_t QQQ[9][MQ1 * MQ1 * MQ1];

      kernels::internal::LoadX<MD1>(e, D1D, X, DDD);
      kernels::internal::LoadBG<MD1, MQ1>(D1D, Q1D, B, G, BG);

      kernels::internal::GradX<MD1, MQ1>(D1D, Q1D, BG, DDD, DDQ);
      kernels::internal::GradY<MD1, MQ1>(D1D, Q1D, BG, DDQ, DQQ);
      kernels::internal::GradZ<MD1, MQ1>(D1D, Q1D, BG, DQQ, QQQ);

      MFEM_FOREACH_THREAD(qz, z, Q1D)
      {
         MFEM_FOREACH_THREAD(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               real_t J[9];
               kernels::internal::PullGrad<MQ1>(Q1D, qx, qy, qz, QQQ, J);
               E(qx, qy, qz, e) = kernels::Det<3>(J);
            }
         }
      }
   });
}

real_t TMOPNewtonSolver::MinDetJpr_3D(const FiniteElementSpace *fes,
                                      const Vector &x) const
{
   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *R = fes->GetElementRestriction(ordering);
   Vector xe(R->Height(), Device::GetDeviceMemoryType());
   xe.UseDevice(true);
   R->Mult(x, xe);

   const DofToQuad &maps = fes->GetFE(0)->GetDofToQuad(ir, DofToQuad::TENSOR);
   const int NE = fes->GetMesh()->GetNE();
   const int NQ = ir.GetNPoints();
   const int d = maps.ndof, q = maps.nqpt;

   constexpr int DIM = 3;
   const auto B = Reshape(maps.B.Read(), q, d);
   const auto G = Reshape(maps.G.Read(), q, d);
   const auto XE = Reshape(xe.Read(), d, d, d, DIM, NE);

   Vector e(NE * NQ);
   e.UseDevice(true);
   auto E = Reshape(e.Write(), q, q, q, NE);

   decltype(&TMOP_MinDetJpr_3D<>) ker = TMOP_MinDetJpr_3D;

   if (d == 2 && q == 2) { ker = TMOP_MinDetJpr_3D<2, 2>; }
   if (d == 2 && q == 3) { ker = TMOP_MinDetJpr_3D<2, 3>; }
   if (d == 2 && q == 4) { ker = TMOP_MinDetJpr_3D<2, 4>; }
   if (d == 2 && q == 5) { ker = TMOP_MinDetJpr_3D<2, 5>; }
   if (d == 2 && q == 6) { ker = TMOP_MinDetJpr_3D<2, 6>; }

   if (d == 3 && q == 3) { ker = TMOP_MinDetJpr_3D<3, 3>; }
   if (d == 3 && q == 4) { ker = TMOP_MinDetJpr_3D<3, 4>; }
   if (d == 3 && q == 5) { ker = TMOP_MinDetJpr_3D<3, 5>; }
   if (d == 3 && q == 6) { ker = TMOP_MinDetJpr_3D<3, 6>; }

   if (d == 4 && q == 4) { ker = TMOP_MinDetJpr_3D<4, 4>; }
   if (d == 4 && q == 5) { ker = TMOP_MinDetJpr_3D<4, 5>; }
   if (d == 4 && q == 6) { ker = TMOP_MinDetJpr_3D<4, 6>; }

   if (d == 5 && q == 5) { ker = TMOP_MinDetJpr_3D<5, 5>; }
   if (d == 5 && q == 6) { ker = TMOP_MinDetJpr_3D<5, 6>; }

   ker(NE, B, G, XE, E, d, q, 4);
   return e.Min();
}

} // namespace mfem
