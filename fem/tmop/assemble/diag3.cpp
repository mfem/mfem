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
void TMOP_AssembleDiagPA_3D(const int NE,
                            const ConstDeviceMatrix &B,
                            const ConstDeviceMatrix &G,
                            const DeviceTensor<6, const real_t> &J,
                            const DeviceTensor<8, const real_t> &H,
                            DeviceTensor<5> &D,
                            const int d1d = 0,
                            const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      MFEM_SHARED real_t smem[3][3][MQ1][MQ1];
      kernels::internal::vd_regs3d_t<3, 3, MQ1> rH, r0, r1;

      for (int v = 0; v < 3; ++v)
      {
         // Takes into account Jtr by replacing H with Href at all quad points.
         for (int qz = 0; qz < Q1D; ++qz)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  const real_t *Jtr = &J(0, 0, qx, qy, qz, e);
                  real_t Jrt_data[9];
                  ConstDeviceMatrix Jrt(Jrt_data, 3, 3);
                  kernels::CalcInverse<3>(Jtr, Jrt_data);

                  real_t h[3][3];
                  for (int s = 0; s < 3; s++)
                  {
                     for (int t = 0; t < 3; t++)
                     {
                        h[s][t] = H(v, s, v, t, qx, qy, qz, e);
                     }
                  }

                  for (int m = 0; m < 3; m++)
                  {
                     for (int n = 0; n < 3; n++)
                     {
                        // Hr_{v,m,n,q} = \sum_{s,t=1}^d
                        //                Jrt_{m,s,q} H_{v,s,v,t,q} Jrt_{n,t,q}
                        rH(m, n, qz, qy, qx) = 0.0;
                        for (int s = 0; s < 3; s++)
                        {
                           for (int t = 0; t < 3; t++)
                           {
                              rH(m, n, qz, qy, qx) += Jrt(m, s) * h[s][t] * Jrt(n, t);
                           }
                        }
                     }
                  }
               }
            }
            MFEM_SYNC_THREAD;
         }

         // Contract in z.
         for (int dz = 0; dz < D1D; ++dz)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  for (int m = 0; m < 3; m++)
                  {
                     for (int n = 0; n < 3; n++)
                     {
                        r0(m, n, dz, qy, qx) = 0.0;
                     }
                  }

                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const real_t Bz = B(qz, dz), Gz = G(qz, dz);
                     for (int m = 0; m < 3; m++)
                     {
                        for (int n = 0; n < 3; n++)
                        {
                           const real_t L = (m == 2 ? Gz : Bz);
                           const real_t R = (n == 2 ? Gz : Bz);
                           r0(m, n, dz, qy, qx) += L * rH(m, n, qz, qy, qx) * R;
                        }
                     }
                  }
               }
            }
            MFEM_SYNC_THREAD;
         }

         // Contract in y.
         for (int dz = 0; dz < D1D; ++dz)
         {
            for (int m = 0; m < 3; m++)
            {
               for (int n = 0; n < 3; n++)
               {
                  MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
                  {
                     MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
                     {
                        smem[m][n][qy][qx] = r0(m, n, dz, qy, qx);
                     }
                  }
               }
            }
            MFEM_SYNC_THREAD;

            MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  for (int m = 0; m < 3; m++)
                  {
                     for (int n = 0; n < 3; n++)
                     {
                        r1(m, n, dz, dy, qx) = 0.0;
                     }
                  }

                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const real_t By = B(qy, dy);
                     const real_t Gy = G(qy, dy);
                     for (int m = 0; m < 3; m++)
                     {
                        for (int n = 0; n < 3; n++)
                        {
                           const real_t L = (m == 1 ? Gy : By);
                           const real_t R = (n == 1 ? Gy : By);
                           r1(m, n, dz, dy, qx) += L * smem[m][n][qy][qx] * R;
                        }
                     }
                  }
               }
            }
            MFEM_SYNC_THREAD;
         }

         // Contract in x.
         for (int dz = 0; dz < D1D; ++dz)
         {
            for (int m = 0; m < 3; m++)
            {
               for (int n = 0; n < 3; n++)
               {
                  MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
                  {
                     MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
                     {
                        smem[m][n][dy][qx] = r1(m, n, dz, dy, qx);
                     }
                  }
               }
            }
            MFEM_SYNC_THREAD;

            MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(dx, x, D1D)
               {
                  real_t d = 0.0;
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const real_t Bx = B(qx, dx);
                     const real_t Gx = G(qx, dx);
                     for (int m = 0; m < 3; m++)
                     {
                        for (int n = 0; n < 3; n++)
                        {
                           const real_t L = (m == 0 ? Gx : Bx);
                           const real_t R = (n == 0 ? Gx : Bx);
                           d += L * smem[m][n][dy][qx] * R;
                        }
                     }
                  }
                  D(dx, dy, dz, v, e) += d;
               }
            }
            MFEM_SYNC_THREAD;
         }
      }
   });
}

MFEM_TMOP_MDQ_REGISTER(TMOPAssembleDiag3D, TMOP_AssembleDiagPA_3D);
MFEM_TMOP_MDQ_SPECIALIZE(TMOPAssembleDiag3D);

void TMOP_Integrator::AssembleDiagonalPA_3D(Vector &diagonal) const
{
   const int NE = PA.ne, d = PA.maps->ndof, q = PA.maps->nqpt;
   MFEM_VERIFY(d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const auto B = Reshape(PA.maps->B.Read(), q, d);
   const auto G = Reshape(PA.maps->G.Read(), q, d);
   const auto J = Reshape(PA.Jtr.Read(), 3, 3, q, q, q, NE);
   const auto H = Reshape(PA.H.Read(), 3, 3, 3, 3, q, q, q, NE);
   auto D = Reshape(diagonal.ReadWrite(), d, d, d, 3, NE);

   TMOPAssembleDiag3D::Run(d, q, NE, B, G, J, H, D, d, q);
}

} // namespace mfem
