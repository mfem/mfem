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
static void D2QValues2D(const int NE,
                        const Array<double> &b_,
                        const Vector &x_,
                        Vector &y_,
                        const int vdim = 1,
                        const int d1d = 0,
                        const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;
   const int VDIM = T_VDIM ? T_VDIM : vdim;

   auto b = Reshape(b_.Read(), Q1D, D1D);
   auto x = Reshape(x_.Read(), D1D, D1D, VDIM, NE);
   auto y = Reshape(y_.Write(), VDIM, Q1D, Q1D, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      const int zid = MFEM_THREAD_ID(z);
      MFEM_SHARED double B[MQ1][MD1];

      MFEM_SHARED double DDz[NBZ][MD1*MD1];
      double (*DD)[MD1] = (double (*)[MD1])(DDz + zid);

      MFEM_SHARED double DQz[NBZ][MD1*MQ1];
      double (*DQ)[MQ1] = (double (*)[MQ1])(DQz + zid);

      if (zid == 0)
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
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               DD[dy][dx] = x(dx,dy,c,e);
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double dq = 0.0;
               for (int dx = 0; dx < D1D; ++dx)
               {
                  dq += B[qx][dx] * DD[dy][dx];
               }
               DQ[dy][qx] = dq;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double qq = 0.0;
               for (int dy = 0; dy < D1D; ++dy)
               {
                  qq += DQ[dy][qx] * B[qy][dy];
               }
               y(c,qx,qy,e) = qq;
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

template<int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0,
         int MAX_D = 0, int MAX_Q = 0>
static void D2QValues3D(const int NE,
                        const Array<double> &b_,
                        const Vector &x_,
                        Vector &y_,
                        const int vdim = 1,
                        const int d1d = 0,
                        const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int VDIM = T_VDIM ? T_VDIM : vdim;

   auto b = Reshape(b_.Read(), Q1D, D1D);
   auto x = Reshape(x_.Read(), D1D, D1D, D1D, VDIM, NE);
   auto y = Reshape(y_.Write(), VDIM, Q1D, Q1D, Q1D, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D;
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
                  y(c,qx,qy,qz,e) = u;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

void QuadratureInterpolator::D2QValues(const FiniteElementSpace &fes,
                                       const DofToQuad *maps,
                                       const Vector &e_vec,
                                       Vector &q_val)
{
   const int dim = fes.GetMesh()->Dimension();
   const int vdim = fes.GetVDim();
   const int NE = fes.GetNE();
   const int D1D = maps->ndof;
   const int Q1D = maps->nqpt;
   const int id = (vdim<<8) | (D1D<<4) | Q1D;

   if (dim == 2)
   {
      switch (id)
      {
         case 0x124: return D2QValues2D<1,2,4,8>(NE, maps->B, e_vec, q_val);
         case 0x136: return D2QValues2D<1,3,6,4>(NE, maps->B, e_vec, q_val);
         case 0x148: return D2QValues2D<1,4,8,2>(NE, maps->B, e_vec, q_val);
         case 0x224: return D2QValues2D<2,2,4,8>(NE, maps->B, e_vec, q_val);
         case 0x236: return D2QValues2D<2,3,6,4>(NE, maps->B, e_vec, q_val);
         case 0x248: return D2QValues2D<2,4,8,2>(NE, maps->B, e_vec, q_val);
         default:
         {
            MFEM_VERIFY(D1D <= MAX_D1D, "Orders higher than " << MAX_D1D-1
                        << " are not supported!");
            MFEM_VERIFY(Q1D <= MAX_Q1D, "Quadrature rules with more than "
                        << MAX_Q1D << " 1D points are not supported!");
            D2QValues2D(NE, maps->B, e_vec, q_val, vdim, D1D, Q1D);
            return;
         }
      }
   }
   if (dim == 3)
   {
      switch (id)
      {
         case 0x124: return D2QValues3D<1,2,4>(NE, maps->B, e_vec, q_val);
         case 0x136: return D2QValues3D<1,3,6>(NE, maps->B, e_vec, q_val);
         case 0x148: return D2QValues3D<1,4,8>(NE, maps->B, e_vec, q_val);
         case 0x324: return D2QValues3D<3,2,4>(NE, maps->B, e_vec, q_val);
         case 0x336: return D2QValues3D<3,3,6>(NE, maps->B, e_vec, q_val);
         case 0x348: return D2QValues3D<3,4,8>(NE, maps->B, e_vec, q_val);
         default:
         {
            constexpr int MD = 8;
            constexpr int MQ = 8;
            MFEM_VERIFY(D1D <= MD, "Orders higher than " << MD-1
                        << " are not supported!");
            MFEM_VERIFY(Q1D <= MQ, "Quadrature rules with more than " << MQ
                        << " 1D points are not supported!");
            D2QValues3D<0,0,0,MD,MQ>(NE, maps->B, e_vec, q_val, vdim, D1D, Q1D);
            return;
         }
      }
   }
   mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
   MFEM_ABORT("Unknown kernel");
}

} // namespace mfem
