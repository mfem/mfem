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
                           const double H = h(r, c, rr, cc);
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

MFEM_REGISTER_TMOP_KERNELS(void, AssembleDiagonalPA_Kernel_2D,
                           const int NE,
                           const Array<double> &b,
                           const Array<double> &g,
                           const DenseTensor &j,
                           const Vector &h,
                           Vector &diagonal,
                           const int d1d,
                           const int q1d)
{
   constexpr int DIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto G = Reshape(g.Read(), Q1D, D1D);
   const auto J = Reshape(j.Read(), DIM, DIM, Q1D, Q1D, NE);
   const auto H = Reshape(h.Read(), DIM, DIM, DIM, DIM, Q1D, Q1D, NE);

   auto D = Reshape(diagonal.ReadWrite(), D1D, D1D, DIM, NE);

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int DIM = 2;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      MFEM_SHARED double qd[DIM*DIM*DIM*DIM*MQ1*MD1];
      DeviceTensor<6,double> QD(qd, DIM, DIM, DIM, DIM, MQ1, MD1);

      for (int v = 0; v < DIM; v++)
      {
         // W_inv = (w00, w10, w01, w11).
         // (d_dx w00 + d_dy w_10, d_dx w01 + d_dy w_11)   on both sides of H.
         //     0,0        1,0        0,1        1,1

         // Contract in y.
         MFEM_FOREACH_THREAD(qx,x,Q1D)
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
                           QD(d1,d2,d3,d4,qx,dy) = 0.0;
                        }
                     }
                  }
               }

               MFEM_UNROLL(MQ1)
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double *Jtr = &J(0,0,qx,qy,e);

                  // Jrt = Jtr^{-1}
                  double jrt_data[4];
                  ConstDeviceMatrix Jrt(jrt_data,2,2);
                  kernels::CalcInverse<2>(Jtr, jrt_data);

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
                              const double L = (l1 == 0 ? By : Gy) * Jrt(l1,l2);
                              const double h = H(v,l2,v,r2,qx,qy,e);
                              const double R = (r1 == 0 ? By : Gy) * Jrt(r1,r2);
                              QD(l1,l2,r1,r2,qx,dy) += L * h * R;
                           }
                        }
                     }
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;

         // Contract in x.
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
                              d += L * QD(l1,l2,r1,r2,qx,dy) * R;
                           }
                        }
                     }
                  }
               }
               D(dx,dy,v,e) += d;
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

void TMOP_Integrator::AssembleDiagonalPA_2D(Vector &D) const
{
   const int N = PA.ne;
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const DenseTensor &J = PA.Jtr;
   const Array<double> &B = PA.maps->B;
   const Array<double> &G = PA.maps->G;
   const Vector &H = PA.H;

   MFEM_LAUNCH_TMOP_KERNEL(AssembleDiagonalPA_Kernel_2D,id,N,B,G,J,H,D);
}

} // namespace mfem
