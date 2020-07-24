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

template<int T_VDIM, int T_D1D, int T_Q1D,
         const int MAX_ND3D = QuadratureInterpolator::MAX_ND3D,
         const int MAX_NQ3D = QuadratureInterpolator::MAX_NQ3D,
         const int MAX_VDIM3D = QuadratureInterpolator::MAX_VDIM3D>
static void GradTensor3D(const int NE,
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

   const auto b = Reshape(b_, Q1D, D1D);
   const auto g = Reshape(g_, Q1D, D1D);
   const auto x = Reshape(x_, D1D, D1D, D1D, VDIM, NE);
   auto y = Reshape(y_, Q1D, Q1D, Q1D, VDIM, 3, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_NQ3D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_ND3D;

      const int tidz = MFEM_THREAD_ID(z);
      MFEM_SHARED double s_B[MQ1][MD1];
      MFEM_SHARED double s_G[MQ1][MD1];
      DeviceTensor<2,double> B((double*)(s_B+0), Q1D, D1D);
      DeviceTensor<2,double> G((double*)(s_G+0), Q1D, D1D);

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
                  y(qx,qy,qz,c,0,e) = u;
                  y(qx,qy,qz,c,1,e) = v;
                  y(qx,qy,qz,c,2,e) = w;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

void QuadratureInterpolator::GradByNodesTensor(
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

   const int id = (vdim<<8) | (D1D<<4) | Q1D;

   if (dim == 3)
   {
      switch (id)
      {
         case 0x124: return GradTensor3D<1,2,4>(NE,B,G,X,Y);
         case 0x136: return GradTensor3D<1,3,6>(NE,B,G,X,Y);
         case 0x148: return GradTensor3D<1,4,8>(NE,B,G,X,Y);

         case 0x324: return GradTensor3D<3,2,4>(NE,B,G,X,Y);
         case 0x333: return GradTensor3D<3,3,3>(NE,B,G,X,Y);
         case 0x335: return GradTensor3D<3,3,5>(NE,B,G,X,Y);
         case 0x336: return GradTensor3D<3,3,6>(NE,B,G,X,Y);
         case 0x348: return GradTensor3D<3,4,8>(NE,B,G,X,Y);
      }
   }
   dbg("0x%x",id);
   MFEM_ABORT("");
}

} // namespace mfem
