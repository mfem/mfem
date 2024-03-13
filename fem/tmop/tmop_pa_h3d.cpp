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

#include "../tmop.hpp"
#include "tmop_pa.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"

namespace mfem
{

MFEM_REGISTER_TMOP_KERNELS(void, AssembleDiagonalPA_Kernel_3D,
                           const int NE,
                           const Array<double> &b,
                           const Array<double> &g,
                           const DenseTensor &j,
                           const Vector &h,
                           Vector &diagonal,
                           const int d1d,
                           const int q1d)
{
   constexpr int DIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto G = Reshape(g.Read(), Q1D, D1D);
   const auto J = Reshape(j.Read(), DIM, DIM, Q1D, Q1D, Q1D, NE);
   const auto H = Reshape(h.Read(), DIM, DIM, DIM, DIM, Q1D, Q1D, Q1D, NE);

   auto D = Reshape(diagonal.ReadWrite(), D1D, D1D, D1D, DIM, NE);

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int DIM = 3;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      // Takes into account Jtr by replacing H with Href at all quad points.
      MFEM_SHARED double Href_data[DIM*DIM*DIM*MQ1*MQ1*MQ1];
      DeviceTensor<6, double> Href(Href_data, DIM, DIM, DIM, MQ1, MQ1, MQ1);
      for (int v = 0; v < DIM; v++)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qz,z,Q1D)
               {
                  const double *Jtr = &J(0,0,qx,qy,qz,e);
                  double Jrt_data[9];
                  ConstDeviceMatrix Jrt(Jrt_data,3,3);
                  kernels::CalcInverse<3>(Jtr, Jrt_data);

                  for (int m = 0; m < DIM; m++)
                  {
                     for (int n = 0; n < DIM; n++)
                     {
                        // Hr_{v,m,n,q} = \sum_{s,t=1}^d
                        //                Jrt_{m,s,q} H_{v,s,v,t,q} Jrt_{n,t,q}
                        Href(v,m,n,qx,qy,qz) = 0.0;
                        for (int s = 0; s < DIM; s++)
                        {
                           for (int t = 0; t < DIM; t++)
                           {
                              Href(v,m,n,qx,qy,qz) +=
                                 Jrt(m,s) * H(v,s,v,t,qx,qy,qz,e) * Jrt(n,t);
                           }
                        }
                     }
                  }
               }
            }
         }
      }

      MFEM_SHARED double qqd[DIM*DIM*MQ1*MQ1*MD1];
      MFEM_SHARED double qdd[DIM*DIM*MQ1*MD1*MD1];
      DeviceTensor<5,double> QQD(qqd, DIM, DIM, MQ1, MQ1, MD1);
      DeviceTensor<5,double> QDD(qdd, DIM, DIM, MQ1, MD1, MD1);

      for (int v = 0; v < DIM; ++v)
      {
         // Contract in z.
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(dz,z,D1D)
               {
                  for (int m = 0; m < DIM; m++)
                  {
                     for (int n = 0; n < DIM; n++)
                     {
                        QQD(m,n,qx,qy,dz) = 0.0;
                     }
                  }

                  MFEM_UNROLL(MQ1)
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const double Bz = B(qz,dz);
                     const double Gz = G(qz,dz);
                     for (int m = 0; m < DIM; m++)
                     {
                        for (int n = 0; n < DIM; n++)
                        {
                           const double L = (m == 2 ? Gz : Bz);
                           const double R = (n == 2 ? Gz : Bz);
                           QQD(m,n,qx,qy,dz) += L * Href(v,m,n,qx,qy,qz) * R;
                        }
                     }
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;

         // Contract in y.
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(dz,z,D1D)
            {
               MFEM_FOREACH_THREAD(dy,y,D1D)
               {
                  for (int m = 0; m < DIM; m++)
                  {
                     for (int n = 0; n < DIM; n++)
                     {
                        QDD(m,n,qx,dy,dz) = 0.0;
                     }
                  }

                  MFEM_UNROLL(MQ1)
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const double By = B(qy,dy);
                     const double Gy = G(qy,dy);
                     for (int m = 0; m < DIM; m++)
                     {
                        for (int n = 0; n < DIM; n++)
                        {
                           const double L = (m == 1 ? Gy : By);
                           const double R = (n == 1 ? Gy : By);
                           QDD(m,n,qx,dy,dz) += L * QQD(m,n,qx,qy,dz) * R;
                        }
                     }
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;

         // Contract in x.
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  double d = 0.0;
                  MFEM_UNROLL(MQ1)
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double Bx = B(qx,dx);
                     const double Gx = G(qx,dx);
                     for (int m = 0; m < DIM; m++)
                     {
                        for (int n = 0; n < DIM; n++)
                        {
                           const double L = (m == 0 ? Gx : Bx);
                           const double R = (n == 0 ? Gx : Bx);
                           d += L * QDD(m,n,qx,dy,dz) * R;
                        }
                     }
                  }
                  D(dx,dy,dz,v,e) += d;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

void TMOP_Integrator::AssembleDiagonalPA_3D(Vector &D) const
{
   const int N = PA.ne;
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const DenseTensor &J = PA.Jtr;
   const Array<double> &B = PA.maps->B;
   const Array<double> &G = PA.maps->G;
   const Vector &H = PA.H;

   MFEM_LAUNCH_TMOP_KERNEL(AssembleDiagonalPA_Kernel_3D,id,N,B,G,J,H,D);
}

} // namespace mfem
