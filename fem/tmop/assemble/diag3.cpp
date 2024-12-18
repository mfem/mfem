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

#include "../pa.hpp"
#include "../../tmop.hpp"
// #include "../../kernel_dispatch.hpp"
#include "../../../general/forall.hpp"
#include "../../../linalg/kernels.hpp"

namespace mfem
{

template <int T_D1D = 0, int T_Q1D = 0, int T_MAX = 4>
void TMOP_AssembleDiagonalPA_3D(const int NE,
                                const ConstDeviceMatrix &B,
                                const ConstDeviceMatrix &G,
                                const DeviceTensor<6, const real_t> &J,
                                const DeviceTensor<8, const real_t> &H,
                                DeviceTensor<5> &D,
                                const int d1d = 0,
                                const int q1d = 0,
                                const int max = 4)
{
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
   // This kernel uses its own CUDA/ROCM limits: compile time values:
#if defined(__CUDA_ARCH__)
      constexpr int MAX_D1D = 6;
      constexpr int MAX_Q1D = 7;
#elif defined(__HIP_DEVICE_COMPILE__)
      constexpr int MAX_D1D = 7;
      constexpr int MAX_Q1D = 7;
#else
      constexpr int MAX_D1D = DofQuadLimits::MAX_D1D;
      constexpr int MAX_Q1D = DofQuadLimits::MAX_Q1D; // 123
#endif

      constexpr int DIM = 3;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;

      MFEM_SHARED real_t bg[2 * MQ1 * MD1];
      DeviceMatrix B_sm(bg, MQ1, MD1);
      DeviceMatrix G_sm(bg + MQ1 * MD1, MQ1, MD1);

      MFEM_SHARED real_t qqq[DIM * DIM * MQ1 * MQ1 * MQ1];
      MFEM_SHARED real_t qqd[DIM * DIM * MQ1 * MQ1 * MD1];
      DeviceTensor<5, real_t> Href(qqq, DIM, DIM, MQ1, MQ1, MQ1);
      DeviceTensor<5, real_t> QQD(qqd, DIM, DIM, MQ1, MQ1, MD1);
      DeviceTensor<5, real_t> QDD(qqq, DIM, DIM, MQ1, MD1,
                                  MD1); // reuse qqq

      // Load B + G into shared memory
      MFEM_FOREACH_THREAD(q, x, Q1D)
      {
         MFEM_FOREACH_THREAD(d, y, D1D)
         {
            MFEM_FOREACH_THREAD(dummy, z, 1)
            {
               B_sm(q, d) = B(q, d);
               G_sm(q, d) = G(q, d);
            }
         }
      }

      for (int v = 0; v < DIM; ++v)
      {
         // Takes into account Jtr by replacing H with Href at all quad points.
         MFEM_FOREACH_THREAD(qx, x, Q1D)
         {
            MFEM_FOREACH_THREAD(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD(qz, z, Q1D)
               {
                  const real_t *Jtr = &J(0, 0, qx, qy, qz, e);
                  real_t Jrt_data[9];
                  ConstDeviceMatrix Jrt(Jrt_data, 3, 3);
                  kernels::CalcInverse<3>(Jtr, Jrt_data);

                  real_t H_loc_data[DIM * DIM];
                  DeviceMatrix H_loc(H_loc_data, DIM, DIM);
                  for (int s = 0; s < DIM; s++)
                  {
                     for (int t = 0; t < DIM; t++)
                     {
                        H_loc(s, t) = H(v, s, v, t, qx, qy, qz, e);
                     }
                  }

                  for (int m = 0; m < DIM; m++)
                  {
                     for (int n = 0; n < DIM; n++)
                     {
                        // Hr_{v,m,n,q} = \sum_{s,t=1}^d
                        //                Jrt_{m,s,q} H_{v,s,v,t,q} Jrt_{n,t,q}
                        Href(m, n, qx, qy, qz) = 0.0;
                        for (int s = 0; s < DIM; s++)
                        {
                           for (int t = 0; t < DIM; t++)
                           {
                              Href(m, n, qx, qy, qz) +=
                                 Jrt(m, s) * H_loc(s, t) * Jrt(n, t);
                           }
                        }
                     }
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;

         // Contract in z.
         MFEM_FOREACH_THREAD(qx, x, Q1D)
         {
            MFEM_FOREACH_THREAD(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD(dz, z, D1D)
               {
                  for (int m = 0; m < DIM; m++)
                  {
                     for (int n = 0; n < DIM; n++)
                     {
                        QQD(m, n, qx, qy, dz) = 0.0;
                     }
                  }

                  MFEM_UNROLL(MQ1)
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const real_t Bz = B_sm(qz, dz);
                     const real_t Gz = G_sm(qz, dz);
                     for (int m = 0; m < DIM; m++)
                     {
                        for (int n = 0; n < DIM; n++)
                        {
                           const real_t L = (m == 2 ? Gz : Bz);
                           const real_t R = (n == 2 ? Gz : Bz);
                           QQD(m, n, qx, qy, dz) +=
                              L * Href(m, n, qx, qy, qz) * R;
                        }
                     }
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;

         // Contract in y.
         MFEM_FOREACH_THREAD(qx, x, Q1D)
         {
            MFEM_FOREACH_THREAD(dz, z, D1D)
            {
               MFEM_FOREACH_THREAD(dy, y, D1D)
               {
                  for (int m = 0; m < DIM; m++)
                  {
                     for (int n = 0; n < DIM; n++)
                     {
                        QDD(m, n, qx, dy, dz) = 0.0;
                     }
                  }

                  MFEM_UNROLL(MQ1)
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const real_t By = B_sm(qy, dy);
                     const real_t Gy = G_sm(qy, dy);
                     for (int m = 0; m < DIM; m++)
                     {
                        for (int n = 0; n < DIM; n++)
                        {
                           const real_t L = (m == 1 ? Gy : By);
                           const real_t R = (n == 1 ? Gy : By);
                           QDD(m, n, qx, dy, dz) +=
                              L * QQD(m, n, qx, qy, dz) * R;
                        }
                     }
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;

         // Contract in x.
         MFEM_FOREACH_THREAD(dz, z, D1D)
         {
            MFEM_FOREACH_THREAD(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD(dx, x, D1D)
               {
                  real_t d = 0.0;
                  MFEM_UNROLL(MQ1)
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const real_t Bx = B_sm(qx, dx);
                     const real_t Gx = G_sm(qx, dx);
                     for (int m = 0; m < DIM; m++)
                     {
                        for (int n = 0; n < DIM; n++)
                        {
                           const real_t L = (m == 0 ? Gx : Bx);
                           const real_t R = (n == 0 ? Gx : Bx);
                           d += L * QDD(m, n, qx, dy, dz) * R;
                        }
                     }
                  }
                  D(dx, dy, dz, v, e) += d;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

TMOP_REGISTER_KERNELS(TMOPAssembleDiag3D, TMOP_AssembleDiagonalPA_3D);

void TMOP_Integrator::AssembleDiagonalPA_3D(Vector &diagonal) const
{
   constexpr int DIM = 3;
   const int NE = PA.ne, d = PA.maps->ndof, q = PA.maps->nqpt;

   const auto B = Reshape(PA.maps->B.Read(), q, d);
   const auto G = Reshape(PA.maps->G.Read(), q, d);
   const auto J = Reshape(PA.Jtr.Read(), DIM, DIM, q, q, q, NE);
   const auto H = Reshape(PA.H.Read(), DIM, DIM, DIM, DIM, q, q, q, NE);
   auto D = Reshape(diagonal.ReadWrite(), d, d, d, DIM, NE);

   const static auto specialized_kernels = []
   { return tmop::KernelSpecializations<TMOPAssembleDiag3D>(); }();

   TMOPAssembleDiag3D::Run(d, q, NE, B, G, J, H, D, d, q, 4);
}

} // namespace mfem
