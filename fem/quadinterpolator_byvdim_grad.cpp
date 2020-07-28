// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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

#define MFEM_DEBUG_COLOR 226
#include "../general/debug.hpp"

namespace mfem
{

template<int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0>
static void GradByVDim2D(const int NE,
                         const double *b_,
                         const double *g_,
                         const double *x_,
                         double *y_,
                         const int vdim = 1,
                         const int d1d = 0,
                         const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;
   const int VDIM = T_VDIM ? T_VDIM : vdim;

   auto b = Reshape(b_, Q1D, D1D);
   auto g = Reshape(g_, Q1D, D1D);
   auto x = Reshape(x_, D1D, D1D, VDIM, NE);
   auto y = Reshape(y_, VDIM, 2, Q1D, Q1D, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      const int tidz = MFEM_THREAD_ID(z);
      MFEM_SHARED double B[MQ1][MD1];
      MFEM_SHARED double G[MQ1][MD1];

      MFEM_SHARED double Xz[NBZ][MD1][MD1];
      double (*X)[MD1] = (double (*)[MD1])(Xz + tidz);

      MFEM_SHARED double GD[2][NBZ][MD1][MQ1];
      double (*DQ0)[MQ1] = (double (*)[MQ1])(GD[0] + tidz);
      double (*DQ1)[MQ1] = (double (*)[MQ1])(GD[1] + tidz);

      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][d] = b(q,d);
               G[q][d] = g(q,d);
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
               X[dx][dy] = x(dx,dy,c,e);
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
                  const double input = X[dx][dy];
                  u += B[qx][dx] * input;
                  v += G[qx][dx] * input;
               }
               DQ0[dy][qx] = u;
               DQ1[dy][qx] = v;
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
                  u += DQ1[dy][qx] * B[qy][dy];
                  v += DQ0[dy][qx] * G[qy][dy];
               }
               y(c,0,qx,qy,e) = u;
               y(c,1,qx,qy,e) = v;
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

template<int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0,
         int MAX_D = 0, int MAX_Q = 0>
static  void GradByVDim3D(const int NE,
                          const double *b_,
                          const double *g_,
                          const double *x_,
                          double *y_,
                          const int vdim = 1,
                          const int d1d = 0,
                          const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int VDIM = T_VDIM ? T_VDIM : vdim;

   auto b = Reshape(b_, Q1D, D1D);
   auto g = Reshape(g_, Q1D, D1D);
   auto x = Reshape(x_, D1D, D1D, D1D, VDIM, NE);
   auto y = Reshape(y_, VDIM, 3, Q1D, Q1D, Q1D, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D;
      const int tidz = MFEM_THREAD_ID(z);
      MFEM_SHARED double B[MQ1][MD1];
      MFEM_SHARED double G[MQ1][MD1];

      MFEM_SHARED double sm0[3][MQ1*MQ1*MQ1];
      MFEM_SHARED double sm1[3][MQ1*MQ1*MQ1];
      double (*X)[MD1][MD1]    = (double (*)[MD1][MD1]) (sm0+2);
      double (*DDQ0)[MD1][MQ1] = (double (*)[MD1][MQ1]) (sm0+0);
      double (*DDQ1)[MD1][MQ1] = (double (*)[MD1][MQ1]) (sm0+1);
      double (*DQQ0)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+0);
      double (*DQQ1)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+1);
      double (*DQQ2)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+2);

      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][d] = b(q,d);
               G[q][d] = g(q,d);
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
                  X[dx][dy][dz] = x(dx,dy,dz,c,e);
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
                     const double coords = X[dx][dy][dz];
                     u += coords * B[qx][dx];
                     v += coords * G[qx][dx];
                  }
                  DDQ0[dz][dy][qx] = u;
                  DDQ1[dz][dy][qx] = v;
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
                     u += DDQ1[dz][dy][qx] * B[qy][dy];
                     v += DDQ0[dz][dy][qx] * G[qy][dy];
                     w += DDQ0[dz][dy][qx] * B[qy][dy];
                  }
                  DQQ0[dz][qy][qx] = u;
                  DQQ1[dz][qy][qx] = v;
                  DQQ2[dz][qy][qx] = w;
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
                     u += DQQ0[dz][qy][qx] * B[qz][dz];
                     v += DQQ1[dz][qy][qx] * B[qz][dz];
                     w += DQQ2[dz][qy][qx] * G[qz][dz];
                  }
                  y(c,0,qx,qy,qz,e) = u;
                  y(c,1,qx,qy,qz,e) = v;
                  y(c,2,qx,qy,qz,e) = w;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

static void D2QGrad(const FiniteElementSpace &fes,
                    const DofToQuad *maps,
                    const Vector &e_vec,
                    Vector &q_der)
{
   const int dim = fes.GetMesh()->Dimension();
   const int vdim = fes.GetVDim();
   const int NE = fes.GetNE();
   const int D1D = maps->ndof;
   const int Q1D = maps->nqpt;
   const int id = (vdim<<8) | (D1D<<4) | Q1D;
   const double *B = maps->B.Read();
   const double *G = maps->G.Read();
   const double *X = e_vec.Read();
   double *Y = q_der.Write();
   if (dim == 2)
   {
      switch (id)
      {
         case 0x134: return GradByVDim2D<1,3,4,8>(NE, B, G, X, Y);
         case 0x146: return GradByVDim2D<1,4,6,4>(NE, B, G, X, Y);
         case 0x158: return GradByVDim2D<1,5,8,2>(NE, B, G, X, Y);
         case 0x234: return GradByVDim2D<2,3,4,8>(NE, B, G, X, Y);
         case 0x246: return GradByVDim2D<2,4,6,4>(NE, B, G, X, Y);
         case 0x258: return GradByVDim2D<2,5,8,2>(NE, B, G, X, Y);
         default:
         {
            MFEM_VERIFY(D1D <= MAX_D1D, "Orders higher than " << MAX_D1D-1
                        << " are not supported!");
            MFEM_VERIFY(Q1D <= MAX_Q1D, "Quadrature rules with more than "
                        << MAX_Q1D << " 1D points are not supported!");
            GradByVDim2D(NE, B, G, X, Y, vdim, D1D, Q1D);
            return;
         }
      }
   }
   if (dim == 3)
   {
      switch (id)
      {
         case 0x134: return GradByVDim3D<1,3,4>(NE, B, G, X, Y);
         case 0x146: return GradByVDim3D<1,4,6>(NE, B, G, X, Y);
         case 0x158: return GradByVDim3D<1,5,8>(NE, B, G, X, Y);
         case 0x334: return GradByVDim3D<3,3,4>(NE, B, G, X, Y);
         case 0x346: return GradByVDim3D<3,4,6>(NE, B, G, X, Y);
         case 0x358: return GradByVDim3D<3,5,8>(NE, B, G, X, Y);
         default:
         {
            constexpr int MD = 8;
            constexpr int MQ = 8;
            MFEM_VERIFY(D1D <= MD, "Orders higher than " << MD-1
                        << " are not supported!");
            MFEM_VERIFY(Q1D <= MQ, "Quadrature rules with more than " << MQ
                        << " 1D points are not supported!");
            GradByVDim3D<0,0,0,MD,MQ>(NE, B, G, X, Y, vdim, D1D, Q1D);
            return;
         }
      }
   }
   mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
   MFEM_ABORT("Unknown kernel");
}

template<>
void QuadratureInterpolator::Derivatives<QVectorLayout::byVDIM>(
   const Vector &e_vec, Vector &q_der) const
{
   MFEM_VERIFY(q_layout == QVectorLayout::byVDIM, "");
   if (fespace->GetNE() == 0) { return; }
   const IntegrationRule &ir = *IntRule;
   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   const DofToQuad &d2q = fespace->GetFE(0)->GetDofToQuad(ir, mode);
   D2QGrad(*fespace, &d2q, e_vec, q_der);
}

} // namespace mfem
