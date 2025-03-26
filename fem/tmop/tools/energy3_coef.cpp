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
#include "../../../general/forall.hpp"
#include "../../../linalg/kernels.hpp"

namespace mfem
{

template <int T_D1D = 0, int T_Q1D = 0>
void TMOP_EnergyPA_C0_3D(const real_t lim_normal,
                         const DeviceTensor<4, const real_t> &LD,
                         const bool const_c0,
                         const DeviceTensor<4, const real_t> &C0,
                         const int NE,
                         const DeviceTensor<6, const real_t> &J,
                         const ConstDeviceCube &W,
                         const real_t *b,
                         const real_t *bld,
                         const DeviceTensor<5, const real_t> &X0,
                         const DeviceTensor<5, const real_t> &X1,
                         DeviceTensor<4> &E,
                         const bool exp_lim,
                         const int d1d,
                         const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      static constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      static constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MD1][MQ1];
      LoadMatrix(D1D, Q1D, bld, sB);

      regs5d_t<1,1,MQ1> rm0, rm1; // scalar LD
      LoadDofs3d(e, D1D, LD, rm0);
      Eval3d(D1D, Q1D, smem, sB, rm0, rm1);

      LoadMatrix(D1D, Q1D, b, sB);

      regs5d_t<3,1,MQ1> r00, r01; // vector X0
      LoadDofs3d(e, D1D, X0, r00);
      Eval3d(D1D, Q1D, smem, sB, r00, r01);

      regs5d_t<3,1,MQ1> r10, r11; // vector X1
      LoadDofs3d(e, D1D, X1, r10);
      Eval3d(D1D, Q1D, smem, sB, r10, r11);

      for (int qz = 0; qz < Q1D; ++qz)
      {
         mfem::tmop::foreach_y_thread(Q1D, [&](int qy)
         {
            mfem::tmop::foreach_x_thread(Q1D, [&](int qx)
            {
               const real_t *Jtr = &J(0, 0, qx, qy, qz, e);
               const real_t detJtr = kernels::Det<3>(Jtr);
               const real_t weight = W(qx, qy, qz) * detJtr;
               const real_t coeff0 = const_c0
                                     ? C0(0, 0, 0, 0)
                                     : C0(qx, qy, qz, e);

               const real_t D = rm1(0, 0, qz, qy, qx);
               const real_t p0[3] =
               {
                  r01(0, 0, qz, qy, qx),
                  r01(1, 0, qz, qy, qx),
                  r01(2, 0, qz, qy, qx)
               };
               const real_t p1[3] =
               {
                  r11(0, 0, qz, qy, qx),
                  r11(1, 0, qz, qy, qx),
                  r11(2, 0, qz, qy, qx)
               };

               const real_t dist = D; // GetValues, default comp set to 0
               real_t id2 = 0.0;
               real_t dsq = 0.0;
               if (!exp_lim)
               {
                  id2 = 0.5 / (dist * dist);
                  dsq = kernels::DistanceSquared<3>(p1, p0) * id2;
                  E(qx, qy, qz, e) = weight * lim_normal * dsq * coeff0;
               }
               else
               {
                  id2 = 1.0 / (dist * dist);
                  dsq = kernels::DistanceSquared<3>(p1, p0) * id2;
                  E(qx, qy, qz, e) =
                     weight * lim_normal * exp(10.0 * (dsq - 1.0)) * coeff0;
               }
            });
         });
      }
   });
}

MFEM_TMOP_REGISTER_KERNELS(TMOPEnergyPAC03D, TMOP_EnergyPA_C0_3D);
MFEM_TMOP_ADD_SPECIALIZED_KERNELS(TMOPEnergyPAC03D);

real_t TMOP_Integrator::GetLocalStateEnergyPA_C0_3D(const Vector &x) const
{
   constexpr int DIM = 3;

   const real_t ln = lim_normal;
   const bool const_c0 = PA.C0.Size() == 1;
   const int NE = PA.ne, d = PA.maps->ndof, q = PA.maps->nqpt;

   MFEM_VERIFY(PA.maps_lim->ndof == d, "");
   MFEM_VERIFY(PA.maps_lim->nqpt == q, "");
   MFEM_VERIFY(d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const auto C0 = const_c0 ? Reshape(PA.C0.Read(), 1, 1, 1, 1)
                   : Reshape(PA.C0.Read(), q, q, q, NE);
   const auto LD = Reshape(PA.LD.Read(), d, d, d, NE);
   const auto J = Reshape(PA.Jtr.Read(), DIM, DIM, q, q, q, NE);
   const auto *b = PA.maps->B.Read();
   const auto *bld = PA.maps_lim->B.Read();
   const auto W = Reshape(PA.ir->GetWeights().Read(), q, q, q);
   const auto XL = Reshape(PA.XL.Read(), d, d, d, DIM, NE);
   const auto X = Reshape(x.Read(), d, d, d, DIM, NE);
   auto E = Reshape(PA.E.Write(), q, q, q, NE);

   auto el = dynamic_cast<TMOP_ExponentialLimiter *>(lim_func);
   const bool exp_lim = (el) ? true : false;

   TMOPEnergyPAC03D::Run(d, q,
                         ln, LD, const_c0, C0, NE, J, W,
                         b, bld, XL, X, E, exp_lim, d, q);

   return PA.E * PA.O;
}

} // namespace mfem
