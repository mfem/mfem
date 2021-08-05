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

#include "../general/forall.hpp"
#include "gridfunc.hpp"
#include "restriction.hpp"

using namespace std;

namespace mfem
{

template<int D1D, int Q1D>
static void NDK_SmemPAMassDiag3D(const int ndofs,
                                 const int NE,
                                 const int *map_,
                                 const double *b_,
                                 const double *d_,
                                 double *y_)
{
   const auto MAP = Reshape(map_, D1D,D1D,D1D, NE);
   const auto b = Reshape(b_, Q1D, D1D);
   const auto D = Reshape(d_, Q1D, Q1D, Q1D, NE);
   auto Y = Reshape(y_, ndofs);
   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int tidz = MFEM_THREAD_ID(z);
      MFEM_SHARED double B[Q1D][D1D];
      MFEM_SHARED double QQD[Q1D][Q1D][D1D];
      MFEM_SHARED double QDD[Q1D][D1D][D1D];
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
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(dz,z,D1D)
            {
               QQD[qx][qy][dz] = 0.0;
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  QQD[qx][qy][dz] += B[qz][dz] * B[qz][dz] * D(qx, qy, qz, e);
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               QDD[qx][dy][dz] = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  QDD[qx][dy][dz] += B[qy][dy] * B[qy][dy] * QQD[qx][qy][dz];
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               double t = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  t += B[qx][dx] * B[qx][dx] * QDD[qx][dy][dz];
               }
               const int gid = MAP(dx, dy, dz, e);
               const int idx = gid >= 0 ? gid : -1 - gid;
               AtomicAdd(Y(idx), t);
            }
         }
      }
   });
}

template<int D1D, int Q1D>
static void NDK_RegsPAMassDiag3D(const int ndofs,
                                 const int NE,
                                 const int *map_,
                                 const double *b_,
                                 const double *d_,
                                 double *y_)
{
   const auto MAP = Reshape(map_, D1D,D1D,D1D, NE);
   const auto B = Reshape(b_, Q1D, D1D);
   const auto D = Reshape(d_, Q1D, Q1D, Q1D, NE);
   auto Y = Reshape(y_, ndofs);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, 1,
   {
      double r_wk[Q1D];
      MFEM_SHARED double s_B[Q1D][D1D];
      MFEM_SHARED double s_q[Q1D][Q1D][Q1D];


      MFEM_FOREACH_THREAD(d,y,Q1D)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            if (d<D1D) { s_B[q][d] = B(q,d); }
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(j,y,Q1D)
      {
         MFEM_FOREACH_THREAD(i,x,Q1D)
         {
            MFEM_UNROLL(Q1D)
            for (int k=0; k<Q1D; ++k) { r_wk[k] = D(i,j,k,e); }
            for (int c=0; c<D1D; ++c)
            {
               double q_cji = 0.0;
               MFEM_UNROLL(Q1D)
               for (int k=0; k<Q1D; ++k)
               {
                  const double Bkc = s_B[k][c];
                  q_cji += Bkc * Bkc * r_wk[k];
               }
               s_q[c][j][i] = q_cji;
            }
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(c,y,D1D)
      {
         MFEM_FOREACH_THREAD(i,x,Q1D)
         {
            for (int j=0; j<Q1D; ++j) { r_wk[j] = s_q[c][j][i]; }
            MFEM_UNROLL(D1D)
            for (int b=0; b<D1D; ++b)
            {
               double q_cbi = 0.0;
               MFEM_UNROLL(Q1D)
               for (int j=0; j<Q1D; ++j)
               {
                  const double Bjb = s_B[j][b];
                  q_cbi += Bjb * Bjb * r_wk[j];
               }
               s_q[c][b][i] = q_cbi;
            }
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(c,y,D1D)
      {
         MFEM_FOREACH_THREAD(b,x,D1D)
         {
            for (int i=0; i<Q1D; ++i) { r_wk[i] = s_q[c][b][i]; }
            MFEM_UNROLL(D1D)
            for (int a=0; a<D1D; ++a)
            {
               double q_cba = 0.0;
               MFEM_UNROLL(Q1D)
               for (int i=0; i<Q1D; ++i)
               {
                  const double Bia = s_B[i][a];
                  q_cba += Bia * Bia * r_wk[i];
               }
               s_q[c][b][a] = q_cba;
            }
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(b,y,D1D)
      {
         MFEM_FOREACH_THREAD(a,x,D1D)
         {
            MFEM_UNROLL(D1D)
            for (int c=0; c<D1D; ++c)
            {
               const double q_cba = s_q[c][b][a];
               const int gid = MAP(a,b,c,e);
               const int idx = gid >= 0 ? gid : -1 - gid;
               AtomicAdd(Y(idx), q_cba);
            }
         }
      }
      MFEM_SYNC_THREAD;
   });
}

void NDK_PAMassAssembleDiagonal(const int dim,
                                const int D1D,
                                const int Q1D,
                                const int NE,
                                const FiniteElementSpace *fes,
                                const DofToQuad *maps,
                                const Vector &D,
                                Vector &Y)
{
   const int ND = fes->GetNDofs();
   constexpr ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *ERop = fes->GetElementRestriction(ordering);
   const ElementRestriction* ER = dynamic_cast<const ElementRestriction*>(ERop);
   assert(ER);
   const int *map = ER->GatherMap().Read();
   const double *b = maps->B.Read();
   const double *d = D.Read();
   double *y = Y.ReadWrite();

   assert(dim == 3);
   const int ver = DeviceKernelsVersion();
   const int id = (ver << 8) | (D1D << 4) | Q1D;

   switch (id) // orders 1~6
   {
      case 0x023: return NDK_SmemPAMassDiag3D<2,3>(ND,NE,map,b,d,y);
      case 0x024: return NDK_SmemPAMassDiag3D<2,4>(ND,NE,map,b,d,y);
      case 0x034: return NDK_SmemPAMassDiag3D<3,4>(ND,NE,map,b,d,y);
      case 0x045: return NDK_SmemPAMassDiag3D<4,5>(ND,NE,map,b,d,y);
      case 0x046: return NDK_SmemPAMassDiag3D<4,6>(ND,NE,map,b,d,y);
      case 0x056: return NDK_SmemPAMassDiag3D<5,6>(ND,NE,map,b,d,y);
      case 0x058: return NDK_SmemPAMassDiag3D<5,8>(ND,NE,map,b,d,y);
      case 0x067: return NDK_SmemPAMassDiag3D<6,7>(ND,NE,map,b,d,y);
      case 0x078: return NDK_SmemPAMassDiag3D<7,8>(ND,NE,map,b,d,y);

      case 0x123: return NDK_RegsPAMassDiag3D<2,3>(ND,NE,map,b,d,y);
      case 0x124: return NDK_RegsPAMassDiag3D<2,4>(ND,NE,map,b,d,y);
      case 0x134: return NDK_RegsPAMassDiag3D<3,4>(ND,NE,map,b,d,y);
      case 0x145: return NDK_RegsPAMassDiag3D<4,5>(ND,NE,map,b,d,y);
      case 0x146: return NDK_RegsPAMassDiag3D<4,6>(ND,NE,map,b,d,y);
      case 0x156: return NDK_RegsPAMassDiag3D<5,6>(ND,NE,map,b,d,y);
      case 0x158: return NDK_RegsPAMassDiag3D<5,8>(ND,NE,map,b,d,y);
      case 0x167: return NDK_RegsPAMassDiag3D<6,7>(ND,NE,map,b,d,y);
      case 0x178: return NDK_RegsPAMassDiag3D<7,8>(ND,NE,map,b,d,y);
      default: break;
   }

   MFEM_ABORT("Unknown kernel 0x" << std::hex << id);
}

} // namespace mfem
