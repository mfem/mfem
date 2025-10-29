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

template <int MD1, int MQ1, int T_D1D = 0, int T_Q1D = 0>
void TMOP_MinDetJpr_2D(const int NE,
                       const real_t *b,
                       const real_t *g,
                       const DeviceTensor<4, const real_t> &X,
                       const DeviceTensor<3, real_t> &DetJ,
                       const int d1d,
                       const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];
      kernels::internal::vd_regs2d_t<2, 2, MQ1> r0, r1;

      kernels::internal::LoadMatrix(D1D, Q1D, b, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, g, sG);

      kernels::internal::LoadDofs2d(e, D1D, X, r0);
      kernels::internal::Grad2d(D1D, Q1D, smem, sB, sG, r0, r1);

      MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            const real_t J[4] =
            {
               r1[0][0][qy][qx], r1[1][0][qy][qx],
               r1[0][1][qy][qx], r1[1][1][qy][qx]
            };
            DetJ(qx, qy, e) = kernels::Det<2>(J);
         }
      }
   });
}

MFEM_TMOP_MDQ_REGISTER(TMOPMinDetJpr2D, TMOP_MinDetJpr_2D);
MFEM_TMOP_MDQ_SPECIALIZE(TMOPMinDetJpr2D);

real_t TMOPNewtonSolver::MinDetJpr_2D(const FiniteElementSpace *fes,
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

   auto maps = fes->GetFE(0)->GetDofToQuad(ir, DofToQuad::TENSOR);
   const int NE = fes->GetMesh()->GetNE();
   const int NQ = ir.GetNPoints();

   const int d = maps.ndof, q = maps.nqpt;
   MFEM_VERIFY(d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   Vector E(NE*NQ);
   E.UseDevice(true);

   const auto *b = maps.B.Read(), *g = maps.G.Read();
   const auto xe = Reshape(XE.Read(), d, d, 2, NE);
   auto e = Reshape(E.Write(), q, q, NE);

   TMOPMinDetJpr2D::Run(d, q, NE, b, g, xe, e, d, q);

   return E.Min();
}

} // namespace mfem
