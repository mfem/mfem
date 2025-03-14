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

#include "../pa.hpp"
#include "../../tmop.hpp"
#include "../../tmop_tools.hpp"
#include "../../kernels.hpp"
#include "../../kernels_smem.hpp"
#include "../../../general/forall.hpp"
#include "../../../linalg/kernels.hpp"

namespace mfem
{

template <int T_D1D = 0, int T_Q1D = 0>
void TMOP_MinDetJpr_3D(const int NE,
                       const ConstDeviceMatrix &B,
                       const ConstDeviceMatrix &G,
                       const DeviceTensor<5, const real_t> &X,
                       DeviceTensor<4> &E,
                       const int d1d,
                       const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_TMOP_1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_TMOP_1D, "");

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_TMOP_1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_TMOP_1D;
      constexpr int MDQ = MQ1 > MD1 ? MQ1 : MD1;

      MFEM_SHARED real_t BG[2][MQ1 * MD1];
      MFEM_SHARED real_t sm0[9][MDQ * MDQ * MDQ];
      MFEM_SHARED real_t sm1[9][MDQ * MDQ * MDQ];

      kernels::internal::sm::LoadX<MDQ>(e, D1D, X, sm0);
      kernels::internal::LoadBG<MD1, MQ1>(D1D, Q1D, B, G, BG);

      kernels::internal::sm::GradX<MD1, MQ1>(D1D, Q1D, BG, sm0, sm1);
      kernels::internal::sm::GradY<MD1, MQ1>(D1D, Q1D, BG, sm1, sm0);
      kernels::internal::sm::GradZ<MD1, MQ1>(D1D, Q1D, BG, sm0, sm1);

      MFEM_FOREACH_THREAD(qz, z, Q1D)
      {
         MFEM_FOREACH_THREAD(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               real_t J[9];
               kernels::internal::PullGrad<MDQ>(Q1D, qx, qy, qz, sm1, J);
               E(qx, qy, qz, e) = kernels::Det<3>(J);
            }
         }
      }
   });
}

MFEM_TMOP_REGISTER_KERNELS(TMOPMinDetJpr3D, TMOP_MinDetJpr_3D);
MFEM_TMOP_ADD_SPECIALIZED_KERNELS(TMOPMinDetJpr3D);

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

   TMOPMinDetJpr3D::Run(d,q, NE, B, G, XE, E, d, q);

   return e.Min();
}

} // namespace mfem
