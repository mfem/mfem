// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "quadinterpolator.hpp"
#include "../general/forall.hpp"
#include "../linalg/dtensor.hpp"
#include "../linalg/kernels.hpp"

namespace mfem
{

template<QVectorLayout Q_LAYOUT,
         int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0,
         int T_NBZ = 1, int MAX_D1D = 0, int MAX_Q1D = 0>
static void PhysGrad2D(const int NE,
                       const double *b_,
                       const double *g_,
                       const double *j_,
                       const double *x_,
                       double *y_,
                       const int vdim = 0,
                       const int d1d = 0,
                       const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int VDIM = T_VDIM ? T_VDIM : vdim;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;

   const auto b = Reshape(b_, Q1D, D1D);
   const auto g = Reshape(g_, Q1D, D1D);
   const auto j = Reshape(j_, Q1D, Q1D, 2, 2, NE);
   const auto x = Reshape(x_, D1D, D1D, VDIM, NE);
   auto y = Q_LAYOUT ==QVectorLayout:: byNODES ?
            Reshape(y_, Q1D, Q1D, VDIM, 2, NE):
            Reshape(y_, VDIM, 2, Q1D, Q1D, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      const int tidz = MFEM_THREAD_ID(z);
      MFEM_SHARED double s_B[MQ1][MD1];
      MFEM_SHARED double s_G[MQ1][MD1];
      DeviceTensor<2,double> B((double*)(s_B), Q1D, D1D);
      DeviceTensor<2,double> G((double*)(s_G), Q1D, D1D);

      MFEM_SHARED double s_X[NBZ][MD1*MD1];
      DeviceTensor<2,double> X((double*)(s_X+tidz), MD1, MD1);

      MFEM_SHARED double sm[2][NBZ][MD1*MQ1];
      DeviceTensor<2,double> DQ0((double*)(sm[0]+tidz), MD1, MQ1);
      DeviceTensor<2,double> DQ1((double*)(sm[1]+tidz), MD1, MQ1);

      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B(q,d) = b(q,d);
               G(q,d) = g(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;

      for (int c = 0; c < VDIM; ++c)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               X(dx,dy) = x(dx,dy,c,e);
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               double v = 0.0;
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double input = X(dx,dy);
                  u += input * B(qx,dx);
                  v += input * G(qx,dx);
               }
               DQ0(dy,qx) = u;
               DQ1(dy,qx) = v;
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               double v = 0.0;
               for (int dy = 0; dy < D1D; ++dy)
               {
                  u += DQ1(dy,qx) * B(qy,dy);
                  v += DQ0(dy,qx) * G(qy,dy);
               }
               double Jloc[4], Jinv[4];
               Jloc[0] = j(qx,qy,0,0,e);
               Jloc[1] = j(qx,qy,1,0,e);
               Jloc[2] = j(qx,qy,0,1,e);
               Jloc[3] = j(qx,qy,1,1,e);
               kernels::CalcInverse<2>(Jloc, Jinv);
               if (Q_LAYOUT == QVectorLayout::byVDIM)
               {
                  y(c,0,qx,qy,e) = Jinv[0]*u + Jinv[1]*v;
                  y(c,1,qx,qy,e) = Jinv[2]*u + Jinv[3]*v;
               }
               if (Q_LAYOUT == QVectorLayout::byNODES)
               {
                  y(qx,qy,c,0,e) = Jinv[0]*u + Jinv[1]*v;
                  y(qx,qy,c,1,e) = Jinv[2]*u + Jinv[3]*v;

               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

template<QVectorLayout Q_LAYOUT,
         int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0,
         int MAX_D = 0, int MAX_Q = 0>
static  void PhysGrad3D(const int NE,
                        const double *b_,
                        const double *g_,
                        const double *j_,
                        const double *x_,
                        double *y_,
                        const int vdim = 1,
                        const int d1d = 0,
                        const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int VDIM = T_VDIM ? T_VDIM : vdim;

   const auto b = Reshape(b_, Q1D, D1D);
   const auto g = Reshape(g_, Q1D, D1D);
   const auto j = Reshape(j_, Q1D, Q1D, Q1D, 3, 3, NE);
   const auto x = Reshape(x_, D1D, D1D, D1D, VDIM, NE);
   auto y = Q_LAYOUT ==QVectorLayout:: byNODES ?
            Reshape(y_, Q1D, Q1D, Q1D, VDIM, 3, NE):
            Reshape(y_, VDIM, 3, Q1D, Q1D, Q1D, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D;
      const int tidz = MFEM_THREAD_ID(z);
      MFEM_SHARED double s_B[MQ1][MD1];
      MFEM_SHARED double s_G[MQ1][MD1];
      DeviceTensor<2,double> B((double*)(s_B), Q1D, D1D);
      DeviceTensor<2,double> G((double*)(s_G), Q1D, D1D);

      MFEM_SHARED double sm0[3][MQ1*MQ1*MQ1];
      MFEM_SHARED double sm1[3][MQ1*MQ1*MQ1];
      DeviceTensor<3,double> X((double*)(sm0+2), MD1, MD1, MD1);
      DeviceTensor<3,double> DDQ0((double*)(sm0+0), MD1, MD1, MQ1);
      DeviceTensor<3,double> DDQ1((double*)(sm0+1), MD1, MD1, MQ1);
      DeviceTensor<3,double> DQQ0((double*)(sm1+0), MD1, MQ1, MQ1);
      DeviceTensor<3,double> DQQ1((double*)(sm1+1), MD1, MQ1, MQ1);
      DeviceTensor<3,double> DQQ2((double*)(sm1+2), MD1, MQ1, MQ1);

      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B(q,d) = b(q,d);
               G(q,d) = g(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;

      for (int c = 0; c < VDIM; ++c)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dz,z,D1D)
               {
                  X(dx,dy,dz) = x(dx,dy,dz,c,e);
               }
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  double u = 0.0;
                  double v = 0.0;
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     const double coords = X(dx,dy,dz);
                     u += coords * B(qx,dx);
                     v += coords * G(qx,dx);
                  }
                  DDQ0(dz,dy,qx) = u;
                  DDQ1(dz,dy,qx) = v;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  double u = 0.0;
                  double v = 0.0;
                  double w = 0.0;
                  for (int dy = 0; dy < D1D; ++dy)
                  {
                     u += DDQ1(dz,dy,qx) * B(qy,dy);
                     v += DDQ0(dz,dy,qx) * G(qy,dy);
                     w += DDQ0(dz,dy,qx) * B(qy,dy);
                  }
                  DQQ0(dz,qy,qx) = u;
                  DQQ1(dz,qy,qx) = v;
                  DQQ2(dz,qy,qx) = w;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  double u = 0.0;
                  double v = 0.0;
                  double w = 0.0;
                  for (int dz = 0; dz < D1D; ++dz)
                  {
                     u += DQQ0(dz,qy,qx) * B(qz,dz);
                     v += DQQ1(dz,qy,qx) * B(qz,dz);
                     w += DQQ2(dz,qy,qx) * G(qz,dz);
                  }
                  double Jloc[9], Jinv[9];
                  for (int col = 0; col < 3; col++)
                  {
                     for (int row = 0; row < 3; row++)
                     {
                        Jloc[row+3*col] = j(qx,qy,qz,row,col,e);
                     }
                  }
                  kernels::CalcInverse<3>(Jloc, Jinv);
                  if (Q_LAYOUT == QVectorLayout::byNODES)
                  {
                     y(qx,qy,qz,c,0,e) = Jinv[0]*u + Jinv[1]*v + Jinv[2]*w;
                     y(qx,qy,qz,c,1,e) = Jinv[3]*u + Jinv[4]*v + Jinv[5]*w;
                     y(qx,qy,qz,c,2,e) = Jinv[6]*u + Jinv[7]*v + Jinv[8]*w;
                  }
                  if (Q_LAYOUT == QVectorLayout::byVDIM)
                  {
                     y(c,0,qx,qy,qz,e) = Jinv[0]*u + Jinv[1]*v + Jinv[2]*w;
                     y(c,1,qx,qy,qz,e) = Jinv[3]*u + Jinv[4]*v + Jinv[5]*w;
                     y(c,2,qx,qy,qz,e) = Jinv[6]*u + Jinv[7]*v + Jinv[8]*w;
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

} // namespace mfem
