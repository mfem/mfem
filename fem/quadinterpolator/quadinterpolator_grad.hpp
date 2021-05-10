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

// Internal header, included only by .cpp files.
// Template function implementations.

#include "../quadinterpolator.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../fem/kernels.hpp"
#include "../../linalg/kernels.hpp"

namespace mfem
{

namespace internal
{

namespace quadrature_interpolator
{

// Template compute kernel for derivatives in 2D: tensor product version.
template<QVectorLayout Q_LAYOUT, bool GRAD_PHYS,
         int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0,
         int T_NBZ = 1, int MAX_D1D = 0, int MAX_Q1D = 0>
static void Derivatives2D(const int NE,
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
   static constexpr int NBZ = T_NBZ ? T_NBZ : 1;

   const auto b = Reshape(b_, Q1D, D1D);
   const auto g = Reshape(g_, Q1D, D1D);
   const auto j = Reshape(j_, Q1D, Q1D, 2, 2, NE);
   const auto x = Reshape(x_, D1D, D1D, VDIM, NE);
   auto y = Q_LAYOUT == QVectorLayout:: byNODES ?
            Reshape(y_, Q1D, Q1D, VDIM, 2, NE):
            Reshape(y_, VDIM, 2, Q1D, Q1D, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;

      const int tidz = MFEM_THREAD_ID(z);
      MFEM_SHARED double BG[2][MQ1*MD1];
      kernels::internal::LoadBG<MD1,MQ1>(D1D,Q1D,b,g,BG);
      DeviceMatrix B(BG[0], MD1, MQ1);
      DeviceMatrix G(BG[1], MD1, MQ1);

      MFEM_SHARED double XY[NBZ][MD1*MD1];
      DeviceTensor<2> X((double*)(XY+tidz), MD1, MD1);

      MFEM_SHARED double s_DQ[2][NBZ][MD1*MQ1];
      DeviceTensor<2> DQ0((double*)(s_DQ[0]+tidz), MD1, MQ1);
      DeviceTensor<2> DQ1((double*)(s_DQ[1]+tidz), MD1, MQ1);

      for (int c = 0; c < VDIM; ++c)
      {
         kernels::internal::LoadX<MD1,NBZ>(e,D1D,c,x,XY);
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               double v = 0.0;
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double input = X(dx,dy);
                  u += input * B(dx,qx);
                  v += input * G(dx,qx);
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
                  u += DQ1(dy,qx) * B(dy,qy);
                  v += DQ0(dy,qx) * G(dy,qy);
               }
               if (GRAD_PHYS)
               {
                  double Jloc[4], Jinv[4];
                  Jloc[0] = j(qx,qy,0,0,e);
                  Jloc[1] = j(qx,qy,1,0,e);
                  Jloc[2] = j(qx,qy,0,1,e);
                  Jloc[3] = j(qx,qy,1,1,e);
                  kernels::CalcInverse<2>(Jloc, Jinv);
                  const double U = Jinv[0]*u + Jinv[1]*v;
                  const double V = Jinv[2]*u + Jinv[3]*v;
                  u = U; v = V;
               }
               if (Q_LAYOUT == QVectorLayout::byVDIM)
               {
                  y(c,0,qx,qy,e) = u;
                  y(c,1,qx,qy,e) = v;
               }
               if (Q_LAYOUT == QVectorLayout::byNODES)
               {
                  y(qx,qy,c,0,e) = u;
                  y(qx,qy,c,1,e) = v;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

// Template compute kernel for derivatives in 3D: tensor product version.
template<QVectorLayout Q_LAYOUT, bool GRAD_PHYS,
         int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0,
         int MAX_D1D = 0, int MAX_Q1D = 0>
static void Derivatives3D(const int NE,
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

   const auto b = Reshape(b_, Q1D, D1D);
   const auto g = Reshape(g_, Q1D, D1D);
   const auto j = Reshape(j_, Q1D, Q1D, Q1D, 3, 3, NE);
   const auto x = Reshape(x_, D1D, D1D, D1D, VDIM, NE);
   auto y = Q_LAYOUT == QVectorLayout:: byNODES ?
            Reshape(y_, Q1D, Q1D, Q1D, VDIM, 3, NE):
            Reshape(y_, VDIM, 3, Q1D, Q1D, Q1D, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;

      MFEM_SHARED double BG[2][MQ1*MD1];
      kernels::internal::LoadBG<MD1,MQ1>(D1D,Q1D,b,g,BG);
      DeviceMatrix B(BG[0], MD1, MQ1);
      DeviceMatrix G(BG[1], MD1, MQ1);

      MFEM_SHARED double sm0[3][MQ1*MQ1*MQ1];
      MFEM_SHARED double sm1[3][MQ1*MQ1*MQ1];
      DeviceTensor<3> X((double*)(sm0+2), MD1, MD1, MD1);
      DeviceTensor<3> DDQ0((double*)(sm0+0), MD1, MD1, MQ1);
      DeviceTensor<3> DDQ1((double*)(sm0+1), MD1, MD1, MQ1);
      DeviceTensor<3> DQQ0((double*)(sm1+0), MD1, MQ1, MQ1);
      DeviceTensor<3> DQQ1((double*)(sm1+1), MD1, MQ1, MQ1);
      DeviceTensor<3> DQQ2((double*)(sm1+2), MD1, MQ1, MQ1);

      for (int c = 0; c < VDIM; ++c)
      {
         kernels::internal::LoadX<MD1>(e,D1D,c,x,X);//sm0[2]);
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
                     const double input = X(dx,dy,dz);
                     u += input * B(dx,qx);
                     v += input * G(dx,qx);
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
                     u += DDQ1(dz,dy,qx) * B(dy,qy);
                     v += DDQ0(dz,dy,qx) * G(dy,qy);
                     w += DDQ0(dz,dy,qx) * B(dy,qy);
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
                     u += DQQ0(dz,qy,qx) * B(dz,qz);
                     v += DQQ1(dz,qy,qx) * B(dz,qz);
                     w += DQQ2(dz,qy,qx) * G(dz,qz);
                  }
                  if (GRAD_PHYS)
                  {
                     double Jloc[9], Jinv[9];
                     for (int col = 0; col < 3; col++)
                     {
                        for (int row = 0; row < 3; row++)
                        {
                           Jloc[row+3*col] = j(qx,qy,qz,row,col,e);
                        }
                     }
                     kernels::CalcInverse<3>(Jloc, Jinv);
                     const double U = Jinv[0]*u + Jinv[1]*v + Jinv[2]*w;
                     const double V = Jinv[3]*u + Jinv[4]*v + Jinv[5]*w;
                     const double W = Jinv[6]*u + Jinv[7]*v + Jinv[8]*w;
                     u = U; v = V; w = W;
                  }
                  if (Q_LAYOUT == QVectorLayout::byVDIM)
                  {
                     y(c,0,qx,qy,qz,e) = u;
                     y(c,1,qx,qy,qz,e) = v;
                     y(c,2,qx,qy,qz,e) = w;
                  }
                  if (Q_LAYOUT == QVectorLayout::byNODES)
                  {
                     y(qx,qy,qz,c,0,e) = u;
                     y(qx,qy,qz,c,1,e) = v;
                     y(qx,qy,qz,c,2,e) = w;
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

} // namespace quadrature_interpolator

} // namespace internal

} // namespace mfem
