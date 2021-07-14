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
void AMD_SmemPAMassApply3D(const int ndofs,
                           const int NE,
                           const int *m_,
                           const double *b_,
                           const double *d_,
                           const double *x_,
                           double *y_)
{
   const auto MAP = Reshape(m_, D1D,D1D,D1D, NE);
   const auto b = Reshape(b_, Q1D, D1D);
   const auto D = Reshape(d_, Q1D, Q1D, Q1D, NE);
   const auto X = Reshape(x_, ndofs);
   auto Y = Reshape(y_, ndofs);
   MFEM_FORALL_3D(e, NE, Q1D, Q1D, 1,
   {
      MFEM_SHARED double sDQ[Q1D*Q1D];
      double (*B)[D1D] = (double (*)[D1D]) sDQ;
      double (*Bt)[Q1D] = (double (*)[Q1D]) sDQ;
      MFEM_SHARED double sm0[Q1D*Q1D*Q1D];
      MFEM_SHARED double sm1[Q1D*Q1D*Q1D];
      double (*DDQ)[D1D][Q1D] = (double (*)[D1D][Q1D]) sm1;
      double (*DQQ)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) sm0;
      double (*QQQ)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) sm1;
      double (*QQD)[Q1D][D1D] = (double (*)[Q1D][D1D]) sm0;
      double (*QDD)[D1D][D1D] = (double (*)[D1D][D1D]) sm1;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,Q1D)
         {
            B[dx][dy] = b(dx,dy);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u[D1D];
            MFEM_UNROLL(D1D)
            for (int dz = 0; dz < D1D; dz++)
            {
               u[dz] = 0;
            }
            MFEM_UNROLL(D1D)
            for (int dx = 0; dx < D1D; ++dx)
            {
               MFEM_UNROLL(D1D)
               for (int dz = 0; dz < D1D; ++dz)
               {
                  const int gid = MAP(dx, dy, dz, e);
                  const int idx = gid >= 0 ? gid : -1 - gid;
                  u[dz] += X(idx) * B[qx][dx];
               }
            }
            MFEM_UNROLL(D1D)
            for (int dz = 0; dz < D1D; ++dz)
            {
               DDQ[dz][dy][qx] = u[dz];
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u[D1D];
            MFEM_UNROLL(D1D)
            for (int dz = 0; dz < D1D; dz++)
            {
               u[dz] = 0;
            }
            MFEM_UNROLL(D1D)
            for (int dy = 0; dy < D1D; ++dy)
            {
               MFEM_UNROLL(D1D)
               for (int dz = 0; dz < D1D; dz++)
               {
                  u[dz] += DDQ[dz][dy][qx] * B[qy][dy];
               }
            }
            MFEM_UNROLL(D1D)
            for (int dz = 0; dz < D1D; dz++)
            {
               DQQ[dz][qy][qx] = u[dz];
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u[Q1D];
            MFEM_UNROLL(Q1D)
            for (int qz = 0; qz < Q1D; qz++)
            {
               u[qz] = 0;
            }
            MFEM_UNROLL(D1D)
            for (int dz = 0; dz < D1D; ++dz)
            {
               MFEM_UNROLL(Q1D)
               for (int qz = 0; qz < Q1D; qz++)
               {
                  u[qz] += DQQ[dz][qy][qx] * B[qz][dz];
               }
            }
            MFEM_UNROLL(Q1D)
            for (int qz = 0; qz < Q1D; qz++)
            {
               QQQ[qz][qy][qx] = u[qz] * D(qx,qy,qz,e);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(d,y,D1D)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            Bt[d][q] = b(q,d);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u[Q1D];
            MFEM_UNROLL(Q1D)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               u[qz] = 0;
            }
            MFEM_UNROLL(Q1D)
            for (int qx = 0; qx < Q1D; ++qx)
            {
               MFEM_UNROLL(Q1D)
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  u[qz] += QQQ[qz][qy][qx] * Bt[dx][qx];
               }
            }
            MFEM_UNROLL(Q1D)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               QQD[qz][qy][dx] = u[qz];
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u[Q1D];
            MFEM_UNROLL(Q1D)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               u[qz] = 0;
            }
            MFEM_UNROLL(Q1D)
            for (int qy = 0; qy < Q1D; ++qy)
            {
               MFEM_UNROLL(Q1D)
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  u[qz] += QQD[qz][qy][dx] * Bt[dy][qy];
               }
            }
            MFEM_UNROLL(Q1D)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               QDD[qz][dy][dx] = u[qz];
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u[D1D];
            MFEM_UNROLL(D1D)
            for (int dz = 0; dz < D1D; ++dz)
            {
               u[dz] = 0;
            }
            MFEM_UNROLL(Q1D)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               MFEM_UNROLL(D1D)
               for (int dz = 0; dz < D1D; ++dz)
               {
                  u[dz] += QDD[qz][dy][dx] * Bt[dz][qz];
               }
            }
            MFEM_UNROLL(D1D)
            for (int dz = 0; dz < D1D; ++dz)
            {
               const int gid = MAP(dx, dy, dz, e);
               const int idx = gid >= 0 ? gid : -1 - gid;
               AtomicAdd(Y(idx), u[dz]);
            }
         }
      }
   });
}

struct XElementRestriction : public ElementRestriction
{
   XElementRestriction(const FiniteElementSpace *fes,
                       ElementDofOrdering ordering)
      : ElementRestriction(*fes, ordering) { }
   const Array<int> &GatherMap() const { return gatherMap; }
};

void AMD_PAMassApply(const int dim,
                     const int D1D,
                     const int Q1D,
                     const int NE,
                     const FiniteElementSpace *fes,
                     const DofToQuad *maps,
                     const Vector &D,
                     const Vector &X,
                     Vector &Y)
{
   const int ndofs = fes->GetNDofs();
   static XElementRestriction ER(fes, ElementDofOrdering::LEXICOGRAPHIC);
   const int *map = ER.GatherMap().Read();
   const double *b = maps->B.Read();
   const double *d = D.Read();
   const double *x = X.Read();
   double *y = Y.ReadWrite();

   assert(dim == 3);
   const int id = (D1D << 4) | Q1D;

   switch (id) // orders 1~6
   {
      case 0x23: return AMD_SmemPAMassApply3D<2,3>(ndofs,NE,map,b,d,x,y);
      case 0x34: return AMD_SmemPAMassApply3D<3,4>(ndofs,NE,map,b,d,x,y);
      case 0x45: return AMD_SmemPAMassApply3D<4,5>(ndofs,NE,map,b,d,x,y);
      case 0x56: return AMD_SmemPAMassApply3D<5,6>(ndofs,NE,map,b,d,x,y);
      case 0x67: return AMD_SmemPAMassApply3D<6,7>(ndofs,NE,map,b,d,x,y);
      case 0x78: return AMD_SmemPAMassApply3D<7,8>(ndofs,NE,map,b,d,x,y);
      default: break;
   }

   MFEM_ABORT("Unknown kernel 0x" << std::hex << id);
}

} // namespace mfem
