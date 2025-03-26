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
#include "../../../general/forall.hpp"
#include "../../../linalg/kernels.hpp"

namespace mfem
{

template <int T_D1D = 0, int T_Q1D = 0>
void TMOP_MinDetJpr_3D(const int NE,
                       const real_t *b,
                       const real_t *g,
                       const DeviceTensor<5, const real_t> &X,
                       DeviceTensor<4> &E,
                       const int d1d,
                       const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int DIM = 3, VDIM = 3;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];
      regs5d_t<VDIM, DIM, MQ1> r0, r1;

      LoadMatrix(D1D, Q1D, b, sB);
      LoadMatrix(D1D, Q1D, g, sG);

      LoadDofs3d(e, D1D, X, r0);
      Grad3d(D1D, Q1D, smem, sB, sG, r0, r1);

      for (int qz = 0; qz < Q1D; ++qz)
      {
         foreach_y_thread(Q1D, [&](int qy)
         {
            foreach_x_thread(Q1D, [&](int qx)
            {
               const real_t J[9] =
               {
                  r1(0, 0, qz, qy, qx), r1(1, 0, qz, qy, qx), r1(2, 0, qz, qy, qx),
                  r1(0, 1, qz, qy, qx), r1(1, 1, qz, qy, qx), r1(2, 1, qz, qy, qx),
                  r1(0, 2, qz, qy, qx), r1(1, 2, qz, qy, qx), r1(2, 2, qz, qy, qx)
               };
               E(qx, qy, qz, e) = kernels::Det<3>(J);
            });
         });
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
   const int NE = fes->GetMesh()->GetNE(), NQ = ir.GetNPoints();
   const int d = maps.ndof, q = maps.nqpt;

   MFEM_VERIFY(d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   constexpr int DIM = 3;
   const auto *B = maps.B.Read(), *G = maps.G.Read();
   const auto XE = Reshape(xe.Read(), d, d, d, DIM, NE);

   Vector e(NE * NQ);
   e.UseDevice(true);
   auto E = Reshape(e.Write(), q, q, q, NE);

   TMOPMinDetJpr3D::Run(d,q, NE, B, G, XE, E, d, q);

   return e.Min();
}

} // namespace mfem
