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

#define MFEM_NVTX_COLOR Pink
#include "../general/nvtx.hpp"

#define MFEM_DEBUG_COLOR 206
#include "../general/debug.hpp"
#include "../general/forall.hpp"
#include "bilininteg.hpp"
#include "gridfunc.hpp"
#include "ceed/diffusion.hpp"

using namespace std;

namespace mfem
{

// Shared memory PA Diffusion Diagonal 3D kernel
template<int D1D, int Q1D>
static void NDK_SmemPADiffusionDiag3D(const int ndofs,
                                      const int NE,
                                      const bool symmetric,
                                      const int *map_,
                                      const double *b_,
                                      const double *g_,
                                      const double *d_,
                                      double *y_)
{
   MFEM_NVTX;
   constexpr int DIM = 3;
   const auto MAP = Reshape(map_, D1D,D1D,D1D, NE);
   auto b = Reshape(b_, Q1D, D1D);
   auto g = Reshape(g_, Q1D, D1D);
   auto D = Reshape(d_, Q1D*Q1D*Q1D, symmetric ? 6 : 9, NE);
   auto Y = Reshape(y_, D1D, D1D, D1D, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int tidz = MFEM_THREAD_ID(z);
      MFEM_SHARED double BG[2][Q1D*D1D];
      double (*B)[D1D] = (double (*)[D1D]) (BG+0);
      double (*G)[D1D] = (double (*)[D1D]) (BG+1);
      MFEM_SHARED double QQD[Q1D][Q1D][D1D];
      MFEM_SHARED double QDD[Q1D][D1D][D1D];
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
      for (int i = 0; i < DIM; ++i)
      {
         for (int j = 0; j < DIM; ++j)
         {
            // first tensor contraction, along z direction
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               MFEM_FOREACH_THREAD(qy,y,Q1D)
               {
                  MFEM_FOREACH_THREAD(dz,z,D1D)
                  {
                     QQD[qx][qy][dz] = 0.0;
                     for (int qz = 0; qz < Q1D; ++qz)
                     {
                        const int q = qx + (qy + qz * Q1D) * Q1D;
                        const int ksym = j >= i ?
                                         3 - (3-i)*(2-i)/2 + j:
                                         3 - (3-j)*(2-j)/2 + i;
                        const int k = symmetric ? ksym : (i*DIM) + j;
                        const double O = D(q,k,e);
                        const double Bz = B[qz][dz];
                        const double Gz = G[qz][dz];
                        const double L = i==2 ? Gz : Bz;
                        const double R = j==2 ? Gz : Bz;
                        QQD[qx][qy][dz] += L * O * R;
                     }
                  }
               }
            }
            MFEM_SYNC_THREAD;
            // second tensor contraction, along y direction
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               MFEM_FOREACH_THREAD(dz,z,D1D)
               {
                  MFEM_FOREACH_THREAD(dy,y,D1D)
                  {
                     QDD[qx][dy][dz] = 0.0;
                     for (int qy = 0; qy < Q1D; ++qy)
                     {
                        const double By = B[qy][dy];
                        const double Gy = G[qy][dy];
                        const double L = i==1 ? Gy : By;
                        const double R = j==1 ? Gy : By;
                        QDD[qx][dy][dz] += L * QQD[qx][qy][dz] * R;
                     }
                  }
               }
            }
            MFEM_SYNC_THREAD;
            // third tensor contraction, along x direction
            MFEM_FOREACH_THREAD(dz,z,D1D)
            {
               MFEM_FOREACH_THREAD(dy,y,D1D)
               {
                  MFEM_FOREACH_THREAD(dx,x,D1D)
                  {
                     for (int qx = 0; qx < Q1D; ++qx)
                     {
                        const double Bx = B[qx][dx];
                        const double Gx = G[qx][dx];
                        const double L = i==0 ? Gx : Bx;
                        const double R = j==0 ? Gx : Bx;
                        const double lvr = L * QDD[qx][dy][dz] * R;
                        const int gid = MAP(dx,dy,dz,e);
                        const int idx = gid >= 0 ? gid : -1 - gid;
                        AtomicAdd(Y[idx], lvr);
                     }
                  }
               }
            }
            MFEM_SYNC_THREAD;
         }
      }
   });
}

void NDK_PADiffusionAssembleDiagonal(const int dim,
                                     const int D1D,
                                     const int Q1D,
                                     const int NE,
                                     const bool symm,
                                     const FiniteElementSpace *fes,
                                     const DofToQuad *maps,
                                     const Vector &D,
                                     Vector &Y)
{
   MFEM_NVTX;
   const int ND = fes->GetNDofs();
   constexpr ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *ERop = fes->GetElementRestriction(ordering);
   const ElementRestriction* ER = dynamic_cast<const ElementRestriction*>(ERop);
   assert(ER);
   const int *m = ER->GatherMap().Read();
   const double *b = maps->B.Read();
   const double *g = maps->G.Read();
   const double *d = D.Read();
   double *y = Y.ReadWrite();

   if (dim == 2) {  assert(false); }
   else if (dim == 3)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x22: return NDK_SmemPADiffusionDiag3D<2,2>(ND,NE,symm,m,b,g,d,y);
         case 0x23: return NDK_SmemPADiffusionDiag3D<2,3>(ND,NE,symm,m,b,g,d,y);
         case 0x34: return NDK_SmemPADiffusionDiag3D<3,4>(ND,NE,symm,m,b,g,d,y);
         case 0x45: return NDK_SmemPADiffusionDiag3D<4,5>(ND,NE,symm,m,b,g,d,y);
         case 0x46: return NDK_SmemPADiffusionDiag3D<4,6>(ND,NE,symm,m,b,g,d,y);
         case 0x56: return NDK_SmemPADiffusionDiag3D<5,6>(ND,NE,symm,m,b,g,d,y);
         case 0x67: return NDK_SmemPADiffusionDiag3D<6,7>(ND,NE,symm,m,b,g,d,y);
         case 0x78: return NDK_SmemPADiffusionDiag3D<7,8>(ND,NE,symm,m,b,g,d,y);
         //case 0x89: return NDK_SmemPADiffusionDiag3D<8,9>(ND,NE,symm,m,b,g,d,y);
         //case 0x9A: return NDK_SmemPADiffusionDiag3D<9,10>(ND,NE,symm,m,b,g,d,y);
         default: MFEM_ABORT("Not implemented!");
      }
   }
   MFEM_ABORT("Unknown kernel.");
}

} // namespace mfem
