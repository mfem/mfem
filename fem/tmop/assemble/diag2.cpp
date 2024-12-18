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

<<<<<<<< HEAD:fem/tmop/assemble/2d/diag.cpp
#include "../../pa.hpp"
#include "../../../tmop.hpp"
#include "../../../../general/forall.hpp"
#include "../../../../linalg/kernels.hpp"
========
#include "../pa.hpp"
#include "../../tmop.hpp"
#include "../../../general/forall.hpp"
#include "../../../linalg/kernels.hpp"
>>>>>>>> main:fem/tmop/assemble/diag2.cpp

namespace mfem
{

/* // Original i-j assembly (old invariants code).
   for (int e = 0; e < NE; e++)
   {
      for (int q = 0; q < nqp; q++)
      {
         el.CalcDShape(ip, DSh);
         Mult(DSh, Jrt, DS);
         for (int i = 0; i < dof; i++)
         {
            for (int j = 0; j < dof; j++)
            {
               for (int r = 0; r < dim; r++)
               {
                  for (int c = 0; c < dim; c++)
                  {
                     for (int rr = 0; rr < dim; rr++)
                     {
                        for (int cc = 0; cc < dim; cc++)
                        {
                           const real_t H = h(r, c, rr, cc);
                           A(e, i + r*dof, j + rr*dof) +=
                                 weight_q * DS(i, c) * DS(j, cc) * H;
                        }
                     }
                  }
               }
            }
         }
      }
   }*/

template <int T_D1D = 0, int T_Q1D = 0, int T_MAX = 4>
void TMOP_AssembleDiagonalPA_2D(const int NE,
                                const ConstDeviceMatrix &B,
                                const ConstDeviceMatrix &G,
                                const DeviceTensor<5, const real_t> &J,
                                const DeviceTensor<7, const real_t> &H,
                                DeviceTensor<4> &D,
                                const int d1d = 0,
                                const int q1d = 0,
                                const int max = 0)
{
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int DIM = 2;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      // Takes into account Jtr by replacing H with Href at all quad points.
      MFEM_SHARED real_t Href_data[DIM * DIM * DIM * MQ1 * MQ1];
      DeviceTensor<5, real_t> Href(Href_data, DIM, DIM, DIM, MQ1, MQ1);
      for (int v = 0; v < DIM; v++)
      {
         MFEM_FOREACH_THREAD(qx, x, Q1D)
         {
            MFEM_FOREACH_THREAD(qy, y, Q1D)
            {
               const real_t *Jtr = &J(0, 0, qx, qy, e);
               real_t Jrt_data[4];
               ConstDeviceMatrix Jrt(Jrt_data, 2, 2);
               kernels::CalcInverse<2>(Jtr, Jrt_data);

               for (int m = 0; m < DIM; m++)
               {
                  for (int n = 0; n < DIM; n++)
                  {
                     // Hr_{v,m,n,q} = \sum_{s,t=1}^d
                     //                Jrt_{m,s,q} H_{v,s,v,t,q} Jrt_{n,t,q}
                     Href(v, m, n, qx, qy) = 0.0;
                     for (int s = 0; s < DIM; s++)
                     {
                        for (int t = 0; t < DIM; t++)
                        {
                           Href(v, m, n, qx, qy) +=
                              Jrt(m, s) * H(v, s, v, t, qx, qy, e) * Jrt(n, t);
                        }
                     }
                  }
               }
            }
         }
      }

      MFEM_SHARED real_t qd[DIM * DIM * MQ1 * MD1];
      DeviceTensor<4, real_t> QD(qd, DIM, DIM, MQ1, MD1);

      for (int v = 0; v < DIM; v++)
      {
         // Contract in y.
         MFEM_FOREACH_THREAD(qx, x, Q1D)
         {
            MFEM_FOREACH_THREAD(dy, y, D1D)
            {
               for (int m = 0; m < DIM; m++)
               {
                  for (int n = 0; n < DIM; n++) { QD(m, n, qx, dy) = 0.0; }
               }

               MFEM_UNROLL(MQ1)
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t By = B(qy, dy);
                  const real_t Gy = G(qy, dy);
                  for (int m = 0; m < DIM; m++)
                  {
                     for (int n = 0; n < DIM; n++)
                     {
                        const real_t L = (m == 1 ? Gy : By);
                        const real_t R = (n == 1 ? Gy : By);
                        QD(m, n, qx, dy) += L * Href(v, m, n, qx, qy) * R;
                     }
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;

         // Contract in x.
         MFEM_FOREACH_THREAD(dy, y, D1D)
         {
            MFEM_FOREACH_THREAD(dx, x, D1D)
            {
               real_t d = 0.0;
               MFEM_UNROLL(MQ1)
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const real_t Bx = B(qx, dx);
                  const real_t Gx = G(qx, dx);

                  for (int m = 0; m < DIM; m++)
                  {
                     for (int n = 0; n < DIM; n++)
                     {
                        const real_t L = (m == 0 ? Gx : Bx);
                        const real_t R = (n == 0 ? Gx : Bx);
                        d += L * QD(m, n, qx, dy) * R;
                     }
                  }
               }
               D(dx, dy, v, e) += d;
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

TMOP_REGISTER_KERNELS(TMOPAssembleDiag2D, TMOP_AssembleDiagonalPA_2D);

void TMOP_Integrator::AssembleDiagonalPA_2D(Vector &diagonal) const
{
   const int NE = PA.ne;
   constexpr int DIM = 2;
   const int d = PA.maps->ndof, q = PA.maps->nqpt;

   const DenseTensor &j = PA.Jtr;
   const Array<real_t> &b = PA.maps->B;
   const Array<real_t> &g = PA.maps->G;
   const Vector &h = PA.H;

   const auto B = Reshape(b.Read(), q, d);
   const auto G = Reshape(g.Read(), q, d);
   const auto J = Reshape(j.Read(), DIM, DIM, q, q, NE);
   const auto H = Reshape(h.Read(), DIM, DIM, DIM, DIM, q, q, NE);
   auto D = Reshape(diagonal.ReadWrite(), d, d, DIM, NE);

   const static auto specialized_kernels = []
   { return tmop::KernelSpecializations<TMOPAssembleDiag2D>(); }();

   TMOPAssembleDiag2D::Run(d, q, NE, B, G, J, H, D, d, q, 4);
}

} // namespace mfem
