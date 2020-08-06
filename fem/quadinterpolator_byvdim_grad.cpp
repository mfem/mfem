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

template<int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0,
         int T_NBZ = 0, int MAX_D1D = 0, int MAX_Q1D = 0>
static void GradByVDim2D(const int NE,
                         const double *b_,
                         const double *g_,
                         const double *x_,
                         double *y_,
                         const int vdim = 0,
                         const int d1d = 0,
                         const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;
   const int VDIM = T_VDIM ? T_VDIM : vdim;

   const auto b = Reshape(b_, Q1D, D1D);
   const auto g = Reshape(g_, Q1D, D1D);
   const auto x = Reshape(x_, D1D, D1D, VDIM, NE);
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
      MFEM_SHARED double s_B[MQ1*MD1];
      MFEM_SHARED double s_G[MQ1*MD1];
      DeviceTensor<2,double> B(s_B, Q1D, D1D);
      DeviceTensor<2,double> G(s_G, Q1D, D1D);

      MFEM_SHARED double s_X[NBZ][MD1*MD1];
      DeviceTensor<2,double> X((double*)(s_X+tidz), MD1, MD1);

      MFEM_SHARED double s_DQ[2][NBZ][MD1*MQ1];
      DeviceTensor<2,double> DQ0((double*)(s_DQ[0]+tidz), MD1, MQ1);
      DeviceTensor<2,double> DQ1((double*)(s_DQ[1]+tidz), MD1, MQ1);

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
               y(c,0,qx,qy,e) = u;
               y(c,1,qx,qy,e) = v;
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

template<int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0,
         int MAX_D1D = 0, int MAX_Q1D = 0>
static  void GradByVDim3D(const int NE,
                          const double *b_,
                          const double *g_,
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
   const auto x = Reshape(x_, D1D, D1D, D1D, VDIM, NE);
   auto y = Reshape(y_, VDIM, 3, Q1D, Q1D, Q1D, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;

      const int tidz = MFEM_THREAD_ID(z);
      MFEM_SHARED double s_B[MQ1*MD1];
      MFEM_SHARED double s_G[MQ1*MD1];
      DeviceTensor<2,double> B(s_B, Q1D, D1D);
      DeviceTensor<2,double> G(s_G, Q1D, D1D);

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
                     const double input = X(dx,dy,dz);
                     u += input * B(qx,dx);
                     v += input * G(qx,dx);
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

template<>
void QuadratureInterpolator::Derivatives<QVectorLayout::byVDIM>(
   const Vector &e_vec, Vector &q_der) const
{
   const int NE = fespace->GetNE();
   if (NE == 0) { return; }
   const int vdim = fespace->GetVDim();
   const int dim = fespace->GetMesh()->Dimension();
   const FiniteElement *fe = fespace->GetFE(0);
   const IntegrationRule *ir =
      IntRule ? IntRule : &qspace->GetElementIntRule(0);
   const DofToQuad &maps = fe->GetDofToQuad(*ir, DofToQuad::TENSOR);
   const int D1D = maps.ndof;
   const int Q1D = maps.nqpt;
   const double *B = maps.B.Read();
   const double *G = maps.G.Read();
   const double *X = e_vec.Read();
   double *Y = q_der.Write();

   const int id = (dim<<12) | (vdim<<8) | (D1D<<4) | Q1D;

   switch (id)
   {
      case 0x2134: return GradByVDim2D<1,3,4,8>(NE,B,G,X,Y);
      case 0x2146: return GradByVDim2D<1,4,6,4>(NE,B,G,X,Y);
      case 0x2158: return GradByVDim2D<1,5,8,2>(NE,B,G,X,Y);

      case 0x2234: return GradByVDim2D<2,3,4,8>(NE,B,G,X,Y);
      case 0x2246: return GradByVDim2D<2,4,6,4>(NE,B,G,X,Y);
      case 0x2258: return GradByVDim2D<2,5,8,2>(NE,B,G,X,Y);

      case 0x3134: return GradByVDim3D<1,3,4>(NE,B,G,X,Y);
      case 0x3146: return GradByVDim3D<1,4,6>(NE,B,G,X,Y);
      case 0x3158: return GradByVDim3D<1,5,8>(NE,B,G,X,Y);

      case 0x3334: return GradByVDim3D<3,3,4>(NE,B,G,X,Y);
      case 0x3346: return GradByVDim3D<3,4,6>(NE,B,G,X,Y);
      case 0x3358: return GradByVDim3D<3,5,8>(NE,B,G,X,Y);
      default:
      {
         constexpr int MD1 = 8;
         constexpr int MQ1 = 8;
         dbg("Using standard kernel #id 0x%x", id);
         MFEM_VERIFY(D1D <= MD1, "Orders higher than " << MD1-1
                     << " are not supported!");
         MFEM_VERIFY(Q1D <= MQ1, "Quadrature rules with more than "
                     << MQ1 << " 1D points are not supported!");
         if (dim == 2)
         {
            return GradByVDim2D<0,0,0,0,MD1,MQ1>(NE,B,G,X,Y,vdim,D1D,Q1D);
         }
         if (dim == 3)
         {
            return GradByVDim3D<0,0,0,MD1,MQ1>(NE,B,G,X,Y,vdim,D1D,Q1D);
         }
      }
   }
   mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
   MFEM_ABORT("Kernel not supported yet");
}

} // namespace mfem
