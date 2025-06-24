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
#include "../../kernels.hpp"
#include "../../tmop_tools.hpp"
#include "../../../general/forall.hpp"
#include "../../../linalg/kernels.hpp"

namespace mfem
{

using namespace kernels::internal;

template <int MD1, int MQ1, int T_D1D = 0, int T_Q1D = 0>
void TMOP_MinDetJpr_3D(const int NE,
                       const real_t *b,
                       const real_t *g,
                       const DeviceTensor<5, const real_t> &X,
                       DeviceTensor<4> &DetJ,
                       const int d1d, const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];
      vd_regs3d_t<3, 3, MQ1> r0, r1;

      LoadMatrix(D1D, Q1D, b, sB);
      LoadMatrix(D1D, Q1D, g, sG);

      LoadDofs3d(e, D1D, X, r0);
      Grad3d(D1D, Q1D, smem, sB, sG, r0, r1);

      for (int qz = 0; qz < Q1D; ++qz)
      {
         mfem::tmop::foreach_y_thread(Q1D, [&](int qy)
         {
            mfem::tmop::foreach_x_thread(Q1D, [&](int qx)
            {
               const real_t J[9] =
               {
                  r1(0, 0, qz, qy, qx), r1(1, 0, qz, qy, qx), r1(2, 0, qz, qy, qx),
                  r1(0, 1, qz, qy, qx), r1(1, 1, qz, qy, qx), r1(2, 1, qz, qy, qx),
                  r1(0, 2, qz, qy, qx), r1(1, 2, qz, qy, qx), r1(2, 2, qz, qy, qx)
               };
               DetJ(qx, qy, qz, e) = kernels::Det<3>(J);
            });
         });
      }
   });
}

MFEM_TMOP_MDQ_REGISTER(TMOPMinDetJpr3D, TMOP_MinDetJpr_3D);
MFEM_TMOP_MDQ_SPECIALIZE(TMOPMinDetJpr3D);

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

   const auto maps = fes->GetFE(0)->GetDofToQuad(ir, DofToQuad::TENSOR);
   const int NE = fes->GetMesh()->GetNE();
   const int NQ = ir.GetNPoints();

   const int d = maps.ndof, q = maps.nqpt;
   MFEM_VERIFY(d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const auto *b = maps.B.Read(), *g = maps.G.Read();
   const auto xe = Reshape(XE.Read(), d, d, d, 3, NE);

   Vector E(NE * NQ);
   E.UseDevice(true);

   auto DetJ = Reshape(E.Write(), q, q, q, NE);

   TMOPMinDetJpr3D::Run(d, q, NE, b, g, xe, DetJ, d, q);

   return E.Min();
}

} // namespace mfem
