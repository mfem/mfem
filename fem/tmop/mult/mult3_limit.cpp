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
#include "../../../general/forall.hpp"
#include "../../../linalg/kernels.hpp"

namespace mfem
{

template <int MD1, int MQ1, int T_D1D = 0, int T_Q1D = 0>
void TMOP_AddMultPA_C0_3D(const real_t lim_normal,
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
                          DeviceTensor<5> &Y,
                          const bool exp_lim,
                          const int d1d,
                          const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MD1][MQ1];
      kernels::internal::LoadMatrix(D1D, Q1D, bld, sB);

      kernels::internal::s_regs3d_t<MQ1> rm0, rm1; // scalar LD
      kernels::internal::LoadDofs3d(e, D1D, LD, rm0);
      kernels::internal::Eval3d(D1D, Q1D, smem, sB, rm0, rm1);

      kernels::internal::LoadMatrix(D1D, Q1D, b, sB);

      kernels::internal::v_regs3d_t<3,MQ1> r00, r01; // vector X0
      kernels::internal::LoadDofs3d(e, D1D, X0, r00);
      kernels::internal::Eval3d(D1D, Q1D, smem, sB, r00, r01);

      kernels::internal::v_regs3d_t<3,MQ1> r10, r11; // vector X1
      kernels::internal::LoadDofs3d(e, D1D, X1, r10);
      kernels::internal::Eval3d(D1D, Q1D, smem, sB, r10, r11);

      for (int qz = 0; qz < Q1D; ++qz)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               const real_t *Jtr = &J(0, 0, qx, qy, qz, e);
               const real_t detJtr = kernels::Det<3>(Jtr);
               const real_t weight = W(qx, qy, qz) * detJtr;
               const real_t D = rm1(qz, qy, qx);
               const real_t p0[3] = { r01(0, qz, qy, qx),
                                      r01(1, qz, qy, qx),
                                      r01(2, qz, qy, qx)
                                    };
               const real_t p1[3] = { r11(0, qz, qy, qx),
                                      r11(1, qz, qy, qx),
                                      r11(2, qz, qy, qx)
                                    };
               const real_t coeff0 = const_c0 ? C0(0, 0, 0, 0) : C0(qx, qy, qz, e);

               real_t d1[3];
               // Eval_d1 (Quadratic Limiter)
               // subtract(1.0 / (dist * dist), x, x0, d1);
               // z = a * (x - y)
               // grad = a * (x - x0)

               // Eval_d1 (Exponential Limiter)
               // real_t dist_squared = dist*dist;
               // subtract(20.0*exp(10.0*((x.DistanceSquaredTo(x0) /
               // dist_squared)
               // - 1.0)) / dist_squared, x, x0, d1); z = a * (x - y) grad = a *
               // (x - x0)
               const real_t dist = D; // GetValues, default comp set to 0
               real_t a = 0.0;
               const real_t w = weight * lim_normal * coeff0;
               const real_t dist_squared = dist * dist;

               if (!exp_lim) { a = 1.0 / dist_squared; }
               else
               {
                  real_t dsq =
                     kernels::DistanceSquared<3>(p1, p0) / dist_squared;
                  a = 20.0 * exp(10.0 * (dsq - 1.0)) / dist_squared;
               }

               kernels::Subtract<3>(w * a, p1, p0, d1);
               r00(0,qz,qy,qx) = d1[0];
               r00(1,qz,qy,qx) = d1[1];
               r00(2,qz,qy,qx) = d1[2];
            }
         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::EvalTranspose3d(D1D, Q1D, smem, sB, r00, r01);
      kernels::internal::WriteDofs3d(e, D1D, r01, Y);
   });
}

MFEM_TMOP_MDQ_REGISTER(TMOPMultCoefKernels3D, TMOP_AddMultPA_C0_3D);
MFEM_TMOP_MDQ_SPECIALIZE(TMOPMultCoefKernels3D);

void TMOP_Integrator::AddMultPA_C0_3D(const Vector &x, Vector &y) const
{
   const real_t ln = lim_normal;
   const bool const_c0 = PA.C0.Size() == 1;
   const int NE = PA.ne, d = PA.maps->ndof, q = PA.maps->nqpt;

   MFEM_VERIFY(d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   MFEM_VERIFY(PA.maps_lim->ndof == d, "");
   MFEM_VERIFY(PA.maps_lim->nqpt == q, "");

   const auto C0 = const_c0
                   ? Reshape(PA.C0.Read(), 1, 1, 1, 1)
                   : Reshape(PA.C0.Read(), q, q, q, NE);
   const auto LD = Reshape(PA.LD.Read(), d, d, d, NE);
   const auto J = Reshape(PA.Jtr.Read(), 3, 3, q, q, q, NE);
   const auto *b = PA.maps->B.Read(), *bld = PA.maps_lim->B.Read();
   const auto W = Reshape(PA.ir->GetWeights().Read(), q, q, q);
   const auto XL = Reshape(PA.XL.Read(), d, d, d, 3, NE);
   const auto X = Reshape(x.Read(), d, d, d, 3, NE);
   auto Y = Reshape(y.ReadWrite(), d, d, d, 3, NE);

   auto el = dynamic_cast<TMOP_ExponentialLimiter *>(lim_func);
   const bool exp_lim = (el) ? true : false;

   TMOPMultCoefKernels3D::Run(d, q, ln, LD, const_c0, C0, NE, J, W, b, bld, XL,
                              X, Y, exp_lim, d, q);
}

} // namespace mfem
