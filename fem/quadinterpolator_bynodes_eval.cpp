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

namespace mfem
{

template<int T_VDIM, int T_D1D, int T_Q1D, int T_NBZ = 1,
         int MAX_D1D = 0, int MAX_Q1D = 0>
static void EvalByNodes2D(const int NE,
                          const double *b_,
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
   const auto x = Reshape(x_, D1D, D1D, VDIM, NE);
   auto y = Reshape(y_, Q1D, Q1D, VDIM, NE);

   constexpr int NBZ = T_NBZ ? T_NBZ : 1;
   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      const int tidz = MFEM_THREAD_ID(z);
      MFEM_SHARED double s_B[MQ1][MD1];
      DeviceTensor<2,double> B((double*)(s_B), Q1D, D1D);

      MFEM_SHARED double s_X[NBZ][MD1*MD1];
      DeviceTensor<2,double> X((double*)(s_X+tidz), MD1, MD1);

      MFEM_SHARED double sm[NBZ][MD1*MQ1];
      DeviceTensor<2,double> DQ((double*)(sm+tidz), MD1, MQ1);

      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B(q,d) = b(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;

      for (int c = 0; c < VDIM; c++)
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
               for (int dx = 0; dx < D1D; ++dx)
               {
                  u += B(qx,dx) *  X(dx,dy);
               }
               DQ(dy,qx) = u;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               for (int dy = 0; dy < D1D; ++dy)
               {
                  u += DQ(dy,qx) * B(qy,dy);
               }
               y(qx,qy,c,e) = u;
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

template<int T_VDIM, int T_D1D, int T_Q1D, int MAX_D1D = 0, int MAX_Q1D = 0>
static void EvalByNodes3D(const int NE,
                          const double *b_,
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
   const auto x = Reshape(x_, D1D, D1D, D1D, VDIM, NE);
   auto y = Reshape(y_, Q1D, Q1D, Q1D, VDIM, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
      const int tidz = MFEM_THREAD_ID(z);
      MFEM_SHARED double B[MQ1][MD1];
      MFEM_SHARED double sm0[MDQ*MDQ*MDQ];
      MFEM_SHARED double sm1[MDQ*MDQ*MDQ];
      double (*X)[MD1][MD1]   = (double (*)[MD1][MD1]) sm0;
      double (*DDQ)[MD1][MQ1] = (double (*)[MD1][MQ1]) sm1;
      double (*DQQ)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) sm0;

      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][d] = b(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;

      for (int c = 0; c < VDIM; c++)
      {
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  X[dz][dy][dx] = x(dx,dy,dz,c,e);
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
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     u += B[qx][dx] * X[dz][dy][dx];
                  }
                  DDQ[dz][dy][qx] = u;
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
                  for (int dy = 0; dy < D1D; ++dy)
                  {
                     u += DDQ[dz][dy][qx] * B[qy][dy];
                  }
                  DQQ[dz][qy][qx] = u;
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
                  for (int dz = 0; dz < D1D; ++dz)
                  {
                     u += DQQ[dz][qy][qx] * B[qz][dz];
                  }
                  y(qx,qy,qz,c,e) = u;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

template<>
void QuadratureInterpolator::Values<QVectorLayout::byNODES>(
   const Vector &e_vec, Vector &q_val) const
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
   const double *X = e_vec.Read();
   double *Y = q_val.Write();

   const int id = (dim<<12) | (vdim<<8) | (D1D<<4) | Q1D;

   switch (id)
   {
      case 0x2124: return EvalByNodes2D<1,2,4>(NE,B,X,Y);

      case 0x2222: return EvalByNodes2D<2,2,2>(NE,B,X,Y);
      case 0x2223: return EvalByNodes2D<2,2,3>(NE,B,X,Y);
      case 0x2224: return EvalByNodes2D<2,2,4>(NE,B,X,Y);
      case 0x2225: return EvalByNodes2D<2,2,5>(NE,B,X,Y);
      case 0x2226: return EvalByNodes2D<2,2,6>(NE,B,X,Y);
      case 0x2233: return EvalByNodes2D<2,3,3>(NE,B,X,Y);
      case 0x2234: return EvalByNodes2D<2,3,4>(NE,B,X,Y);
      case 0x2236: return EvalByNodes2D<2,3,6>(NE,B,X,Y);
      case 0x2244: return EvalByNodes2D<2,4,4>(NE,B,X,Y);
      case 0x2245: return EvalByNodes2D<2,4,5>(NE,B,X,Y);
      case 0x2246: return EvalByNodes2D<2,4,6>(NE,B,X,Y);
      case 0x2247: return EvalByNodes2D<2,4,7>(NE,B,X,Y);
      case 0x2256: return EvalByNodes2D<2,5,6>(NE,B,X,Y);

      case 0x3124: return EvalByNodes3D<1,2,4>(NE,B,X,Y);
      case 0x3136: return EvalByNodes3D<1,3,6>(NE,B,X,Y);
      case 0x3148: return EvalByNodes3D<1,4,8>(NE,B,X,Y);

      case 0x3222: return EvalByNodes3D<2,2,2>(NE,B,X,Y);
      case 0x3223: return EvalByNodes3D<2,2,3>(NE,B,X,Y);
      case 0x3234: return EvalByNodes3D<2,3,4>(NE,B,X,Y);

      case 0x3323: return EvalByNodes3D<3,2,3>(NE,B,X,Y);
      case 0x3324: return EvalByNodes3D<3,2,4>(NE,B,X,Y);
      case 0x3325: return EvalByNodes3D<3,2,5>(NE,B,X,Y);
      case 0x3326: return EvalByNodes3D<3,2,6>(NE,B,X,Y);
      case 0x3333: return EvalByNodes3D<3,3,3>(NE,B,X,Y);
      case 0x3335: return EvalByNodes3D<3,3,5>(NE,B,X,Y);
      case 0x3336: return EvalByNodes3D<3,3,6>(NE,B,X,Y);
      case 0x3347: return EvalByNodes3D<3,4,7>(NE,B,X,Y);
      case 0x3348: return EvalByNodes3D<3,4,8>(NE,B,X,Y);
      default:
      {
         constexpr int MD1 = 8;
         constexpr int MQ1 = 8;
         printf("\033[7m[EvalByNodes] 0x%x\033[m", id); fflush(0);
         MFEM_VERIFY(D1D <= MD1, "Orders higher than " << MD1-1
                     << " are not supported!");
         MFEM_VERIFY(Q1D <= MQ1, "Quadrature rules with more than "
                     << MQ1 << " 1D points are not supported!");
         if (dim == 2)
         {
            return EvalByNodes2D<0,0,0,1,MD1,MQ1>(NE,B,X,Y,vdim,D1D,Q1D);
         }
         if (dim == 3)
         {
            return EvalByNodes3D<0,0,0,MD1,MQ1>(NE,B,X,Y,vdim,D1D,Q1D);
         }
      }
   }
   MFEM_ABORT("Kernel not supported yet");
}

} // namespace mfem
