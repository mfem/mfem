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

//#define MFEM_DEBUG_COLOR 206
//#include "../general/debug.hpp"

#include "../general/forall.hpp"
#include "gridfunc.hpp"
#include "restriction.hpp"

using namespace std;

namespace mfem
{

/*
MFEM_DEVICE int barnos[MFEM_CUDA_BLOCKS];

void MFEM_DEVICE __synblocks()
{
   const int bid_x = MFEM_BLOCK_ID(x);
   const int grid_dim_x = MFEM_GRID_DIM(x);
   const int tid_x = MFEM_THREAD_ID(x);

   // First, sync within each Block
   MFEM_SYNC_THREAD;
   // Pick a representative from each (here, 1D) block
   if (tid_x == 0)
   {
      // Get my barrier number
      const int barno = barnos[bid_x] + 1;
      int hisbarno;
      int who = (bid_x + 1) % grid_dim_x;
      // Check in at barrier
      barnos[bid_x] = barno;
      // Scan for all here or somebody passed
      do
      {
         // Wait for who
         do
         {
            hisbarno = barnos[who];
         }
         while (hisbarno < barno);
         // Bump to next who
         if (++who >= grid_dim_x) { who = 0; }
      }
      while ((hisbarno == barno) && (who != bid_x));
      // Tell others we are all here
      barnos[MFEM_BLOCK_ID(x)] = barno + 1;
   }
   // Rejoin with rest of my Block
   MFEM_SYNC_THREAD;
}*/

// Fast '4' deterministic 3D mass kernel
// Smem version melded toward registers
template<int D1D, int Q1D>
void SmRgPAMassApply3D(const int ndofs,
                       const int NE,
                       const int *map,
                       const int *indices_,
                       const int *offsets_,
                       const double *b_,
                       const double *d_,
                       const double *x_,
                       double *y_)
{
   MFEM_CONTRACT_VAR(indices_);
   const auto MAP = Reshape(map, D1D,D1D,D1D, NE);
   const auto INDICES = Reshape(indices_, ndofs);
   const auto OFFSETS = Reshape(offsets_, ndofs);
   const auto B = Reshape(b_, Q1D, D1D);
   const auto D = Reshape(d_, Q1D, Q1D, Q1D, NE);
   const auto X = Reshape(x_, ndofs);
   const auto X1 = Reshape(x_, D1D,D1D,D1D, NE);
   auto Y = Reshape(y_, ndofs);
   auto Y1 = Reshape(y_, D1D,D1D,D1D, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, 1,
   {

      double u[Q1D];
      MFEM_SHARED double s_B[Q1D][D1D];
      MFEM_SHARED double s_q[Q1D][Q1D][Q1D];

      // Load input, B & X interpolation
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            s_B[qx][dy] = B(qx,dy);
            MFEM_UNROLL(D1D)
            for (int dz = 0; dz < D1D; ++dz) { u[dz] = 0.0; }
            MFEM_UNROLL(D1D)
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double Bx = B(qx,dx);
               MFEM_UNROLL(D1D)
               for (int dz = 0; dz < D1D; ++dz)
               {
                  const int gid = map ? MAP(dx,dy,dz,e) : 0;
                  const int idx = gid >= 0 ? gid : -1 - gid;
                  u[dz] += (map ? X(idx) : X1(dx,dy,dz,e)) * Bx;
               }
            }
            MFEM_UNROLL(D1D)
            for (int dz = 0; dz < D1D; ++dz) { s_q[dz][dy][qx] = u[dz]; }
         }
      }
      MFEM_SYNC_THREAD;

      // Y interpolation
      MFEM_FOREACH_THREAD(dz,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_UNROLL(Q1D)
            for (int qy = 0; qy < Q1D; ++qy) { u[qy] = 0.0; }
            MFEM_UNROLL(D1D)
            for (int dy = 0; dy < D1D; ++dy)
            {
               const double zyX = s_q[dz][dy][qx];
               MFEM_UNROLL(Q1D)
               for (int qy = 0; qy < Q1D; ++qy) { u[qy] += zyX * s_B[qy][dy]; }
            }
            MFEM_UNROLL(Q1D)
            for (int qy = 0; qy < Q1D; ++qy) { s_q[dz][qy][qx] = u[qy]; }
         }
      }
      MFEM_SYNC_THREAD;

      // Z interpolation, Q-function & Zt projection
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            // Z interpolation
            MFEM_UNROLL(Q1D)
            for (int qz = 0; qz < Q1D; ++qz) { u[qz] = 0.0; }
            MFEM_UNROLL(D1D)
            for (int dz = 0; dz < D1D; ++dz)
            {
               const double zYX = s_q[dz][qy][qx];
               MFEM_UNROLL(Q1D)
               for (int qz = 0; qz < Q1D; ++qz) { u[qz] += zYX * s_B[qz][dz]; }
            }

            // Q-function
            MFEM_UNROLL(Q1D)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               s_q[qz][qy][qx] = u[qz] * D(qx,qy,qz,e);
            }

            // Zt projection
            MFEM_UNROLL(D1D)
            for (int dz = 0; dz < D1D; ++dz) { u[dz] = 0.0; }
            MFEM_UNROLL(Q1D)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double ZYX = s_q[qz][qy][qx];
               MFEM_UNROLL(D1D)
               for (int dz = 0; dz < D1D; ++dz) { u[dz] += ZYX * s_B[qz][dz]; }
            }
            MFEM_UNROLL(D1D)
            for (int dz = 0; dz < D1D; ++dz) { s_q[dz][qy][qx] = u[dz]; }
         }
      }
      MFEM_SYNC_THREAD;

      // Yt projection
      MFEM_FOREACH_THREAD(dz,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_UNROLL(D1D)
            for (int dy = 0; dy < D1D; ++dy) { u[dy] = 0.0; }
            MFEM_UNROLL(Q1D)
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double zYX = s_q[dz][qy][qx];
               MFEM_UNROLL(D1D)
               for (int dy = 0; dy < D1D; ++dy) { u[dy] += zYX * s_B[qy][dy]; }
            }
            MFEM_UNROLL(D1D)
            for (int dy = 0; dy < D1D; ++dy) { s_q[dz][dy][qx] = u[dy]; }
         }
      }
      MFEM_SYNC_THREAD;

      // Xt projection & save output
      MFEM_FOREACH_THREAD(dz,y,D1D)
      {
         MFEM_FOREACH_THREAD(dy,x,D1D)
         {
            MFEM_UNROLL(D1D)
            for (int dx = 0; dx < D1D; ++dx) { u[dx] = 0.0; }
            MFEM_UNROLL(Q1D)
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double zyX = s_q[dz][dy][qx];
               MFEM_UNROLL(D1D)
               for (int dx = 0; dx < D1D; ++dx) { u[dx] += zyX * s_B[qx][dx]; }
            }
            MFEM_UNROLL(D1D)
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double output = u[dx];
               if (map)
               {
                  const int gid = MAP(dx,dy,dz,e);
                  const int idx = gid >= 0 ? gid : -1 - gid;
                  const int offset = OFFSETS[idx];
                  const int nextOffset = OFFSETS[idx+1];
                  const int n = nextOffset - offset;
                  //printf("%d:%.15e ",n,output);
                  //assert(n==1);
                  AtomicAdd(Y(idx), output);
               }
               else
               {
                  Y1(dx,dy,dz,e) += output;
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      //MFEM_GRID_SYNC;
      //printf("x");
   });
}

void NDK_DMZ_PAMassApply(const int dim,
                         const int D1D,
                         const int Q1D,
                         const int NE,
                         const FiniteElementSpace *fes,
                         const DofToQuad *maps,
                         const Vector &D,
                         const Vector &X,
                         Vector &Y)
{
   const int ND = fes->GetNDofs();
   constexpr ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *ERop = fes->GetElementRestriction(ordering);
   const ElementRestriction* ER = dynamic_cast<const ElementRestriction*>(ERop);
   const int *map = ER ? ER->GatherMap().Read() : nullptr;
   const int *indices = ER ? ER->Indices().Read() : nullptr;
   const int *offsets = ER ? ER->Offsets().Read() : nullptr;
   const double *b = maps->B.Read();
   //printf("b:%p",(void*)b); fflush(0);
   const double *d = D.Read();
   const double *x = X.Read();
   double *y = Y.ReadWrite();

   assert(dim == 3);
   const int ver = DeviceKernelsVersion();
   const int id = (ver << 8) | (D1D << 4) | Q1D;

   //printf("\033[32mkernel #0x%x\033[m\n",id); fflush(0);

   switch (id) // orders 1~6
   {
      // Fast '4' deterministic 3D mass kernel
      // Smem version melded toward registers

      case 0x423: return SmRgPAMassApply3D<2,3>(ND,NE,
                                                   map,indices,offsets,
                                                   b,d,x,y);
      case 0x456: return SmRgPAMassApply3D<5,6>(ND,NE,
                                                   map,indices,offsets,
                                                   b,d,x,y);

      default: break;
   }

   MFEM_ABORT("Unknown kernel 0x" << std::hex << id);
}

} // namespace mfem
