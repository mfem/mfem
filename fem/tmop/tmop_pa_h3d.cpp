// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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

      MFEM_SHARED double qqd[DIM*DIM*DIM*DIM*MQ1*MQ1*MD1];
      MFEM_SHARED double qdd[DIM*DIM*DIM*DIM*MQ1*MD1*MD1];
      DeviceTensor<7,double> QQD(qqd, DIM, DIM, DIM, DIM, MQ1, MQ1, MD1);
      DeviceTensor<7,double> QDD(qdd, DIM, DIM, DIM, DIM, MQ1, MD1, MD1);

      for (int v = 0; v < DIM; ++v)
      {
         // W_inv = (w_ij)_3x3.
         // ( d_dx w00 + d_dy w_10 + d_dz w_20,
         //   d_dx w01 + d_dy w_11 + d_dz w_21,
         //   d_dx w02 + d_dy w_12 + d_dz w_22 )   on both sides of H.

         // Contract in z.
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(dz,z,D1D)
               {
                  for (int d1 = 0; d1 < DIM; d1++)
                  {
                     for (int d2 = 0; d2 < DIM; d2++)
                     {
                        for (int d3 = 0; d3 < DIM; d3++)
                        {
                           for (int d4 = 0; d4 < DIM; d4++)
                           {
                              QQD(d1,d2,d3,d4,qx,qy,dz) = 0.0;
                           }
                        }
                     }
                  }

                  MFEM_UNROLL(MQ1)
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const double *Jtr = &J(0,0,qx,qy,qz,e);
                     double jrt[9];
                     ConstDeviceMatrix Jrt(jrt,3,3);
                     kernels::CalcInverse<3>(Jtr, jrt);

                     const double Bz = B(qz,dz);
                     const double Gz = G(qz,dz);

                     for (int l1 = 0; l1 < DIM; l1++)
                     {
                        for (int l2 = 0; l2 < DIM; l2++)
                        {
                           for (int r1 = 0; r1 < DIM; r1++)
                           {
                              for (int r2 = 0; r2 < DIM; r2++)
                              {
                                 const double L = (l1 == 2 ?Gz:Bz) * Jrt(l1,l2);
                                 const double h = H(v,l2,v,r2,qx,qy,qz,e);
                                 const double R = (r1 == 2 ?Gz:Bz) * Jrt(r1,r2);
                                 QQD(l1,l2,r1,r2,qx,qy,dz) += L * h * R;
                              }
                           }
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
                  for (int d1 = 0; d1 < DIM; d1++)
                  {
                     for (int d2 = 0; d2 < DIM; d2++)
                     {
                        for (int d3 = 0; d3 < DIM; d3++)
                        {
                           for (int d4 = 0; d4 < DIM; d4++)
                           {
                              QDD(d1,d2,d3,d4,qx,dy,dz) = 0.0;
                           }
                        }
                     }
                  }

                  MFEM_UNROLL(MQ1)
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const double By = B(qy,dy);
                     const double Gy = G(qy,dy);

                     for (int l1 = 0; l1 < DIM; l1++)
                     {
                        for (int l2 = 0; l2 < DIM; l2++)
                        {
                           for (int r1 = 0; r1 < DIM; r1++)
                           {
                              for (int r2 = 0; r2 < DIM; r2++)
                              {
                                 const double L = (l1 == 1 ? Gy : By);
                                 const double R = (r1 == 1 ? Gy : By);
                                 QDD(l1,l2,r1,r2,qx,dy,dz) +=
                                    L * QQD(l1,l2,r1,r2,qx,qy,dz) * R;
                              }
                           }
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

                     for (int l1 = 0; l1 < DIM; l1++)
                     {
                        for (int l2 = 0; l2 < DIM; l2++)
                        {
                           for (int r1 = 0; r1 < DIM; r1++)
                           {
                              for (int r2 = 0; r2 < DIM; r2++)
                              {
                                 const double L = (l1 == 0 ? Gx : Bx);
                                 const double R = (r1 == 0 ? Gx : Bx);
                                 d += L * QDD(l1,l2,r1,r2,qx,dy,dz) * R;
                              }
                           }
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
