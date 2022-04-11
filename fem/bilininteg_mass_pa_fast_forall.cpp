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

#define MFEM_NEW_FORALL_3D(i,N,X,Y,Z,...)                   \
   NewForallWrap<3>(true,N,                                 \
                 [=] MFEM_HOST_DEVICE (int i) {__VA_ARGS__},\
                 X,Y,Z)

/// The forall kernel body wrapper
template <const int DIM, typename HDBODY>
inline void forall(const int N,
                   const int X, const int Y, const int Z, const int G,
                   HDBODY &&hd_body)
{
   MFEM_CONTRACT_VAR(X);
   MFEM_CONTRACT_VAR(Y);
   MFEM_CONTRACT_VAR(Z);
   MFEM_CONTRACT_VAR(G);

#ifdef MFEM_USE_CUDA
   // If Backend::CUDA is allowed, use it
   if (Device::Allows(Backend::CUDA))
   {
      return CuWrap<DIM>::run(N, hd_body, X, Y, Z, G);
   }
#endif

   for (int k = 0; k < N; k++) { hd_body(k); }
}

template <int DIM, typename HDBODY>
inline void forall(const int N, const int X, const int Y, const int Z,
                   HDBODY &&hd_body)
{
   forall<DIM>(N,X,Y,Z,0, hd_body);
}

// Fast '8' FORALL non-deterministic 3D mass kernel
// Smem version melded toward registers + BZ-batch
template<int D1D, int Q1D, int NBZ>
void NDK_FORALL_SmemPAMassApply3D(const int ndofs,
                                  const int NE,
                                  const int *map,
                                  const double *b_,
                                  const double *d_,
                                  const double *x_,
                                  double *y_)
{
   const auto MAP = Reshape(map, D1D,D1D,D1D, NE);
   const auto B = Reshape(b_, Q1D, D1D);
   const auto D = Reshape(d_, Q1D, Q1D, Q1D, NE);
   const auto X = Reshape(x_, ndofs);
   const auto X1 = Reshape(x_, D1D,D1D,D1D, NE);
   auto Y = Reshape(y_, ndofs);
   auto Y1 = Reshape(y_, D1D,D1D,D1D, NE);

   mfem::forall<3>((NE+NBZ-1)/NBZ, Q1D, Q1D, NBZ,
                   [=] MFEM_HOST_DEVICE (int be)
   {
      double u[Q1D];
      const int tz = MFEM_THREAD_ID(z);
      const int e = be * MFEM_THREAD_SIZE(z) + tz;

      MFEM_SHARED double s_B[Q1D][D1D];
      MFEM_SHARED double s_q[NBZ][Q1D][Q1D][Q1D];

      // Load input, B & X interpolation
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            if (tz == 0) { s_B[qx][dy] = B(qx,dy); }
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
            for (int dz = 0; dz < D1D; ++dz) { s_q[tz][dz][dy][qx] = u[dz]; }
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
               const double zyX = s_q[tz][dz][dy][qx];
               MFEM_UNROLL(D1D)
               for (int qy = 0; qy < Q1D; ++qy) { u[qy] += zyX * s_B[qy][dy]; }
            }
            MFEM_UNROLL(Q1D)
            for (int qy = 0; qy < Q1D; ++qy) { s_q[tz][dz][qy][qx] = u[qy]; }
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
               const double zYX = s_q[tz][dz][qy][qx];
               MFEM_UNROLL(Q1D)
               for (int qz = 0; qz < Q1D; ++qz) { u[qz] += zYX * s_B[qz][dz]; }
            }

            // Q-function
            MFEM_UNROLL(Q1D)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               s_q[tz][qz][qy][qx] = u[qz] * D(qx,qy,qz,e);
            }

            // Zt projection
            MFEM_UNROLL(D1D)
            for (int dz = 0; dz < D1D; ++dz) { u[dz] = 0.0; }
            MFEM_UNROLL(Q1D)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double ZYX = s_q[tz][qz][qy][qx];
               MFEM_UNROLL(D1D)
               for (int dz = 0; dz < D1D; ++dz) { u[dz] += ZYX * s_B[qz][dz]; }
            }
            MFEM_UNROLL(D1D)
            for (int dz = 0; dz < D1D; ++dz) { s_q[tz][dz][qy][qx] = u[dz]; }
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
               const double zYX = s_q[tz][dz][qy][qx];
               MFEM_UNROLL(D1D)
               for (int dy = 0; dy < D1D; ++dy) { u[dy] += zYX * s_B[qy][dy]; }
            }
            MFEM_UNROLL(D1D)
            for (int dy = 0; dy < D1D; ++dy) { s_q[tz][dz][dy][qx] = u[dy]; }
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
               const double zyX = s_q[tz][dz][dy][qx];
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
   });
}

void NDK_FORALL_PAMassApply(const int dim,
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
   const double *b = maps->B.Read();
   const double *d = D.Read();
   const double *x = X.Read();
   double *y = Y.ReadWrite();

   assert(dim == 3);
   const int ver = Device::KernelsVersion();
   const int id = (ver << 8) | (D1D << 4) | Q1D;

   //printf("\033[32mkernel #0x%x\033[m\n",id); fflush(0);

   switch (id) // orders 1~6
   {
      // Fast '8': Legacy & half smem non-deterministic 3D mass kernel + Z-batch
      case 0x823: return NDK_FORALL_SmemPAMassApply3D<2,3,32>(ND,NE,map,b,d,x,y);//1
      case 0x824: return NDK_FORALL_SmemPAMassApply3D<2,4,16>(ND,NE,map,b,d,x,y);
      case 0x834: return NDK_FORALL_SmemPAMassApply3D<3,4,16>(ND,NE,map,b,d,x,y);//2
      case 0x836: return NDK_FORALL_SmemPAMassApply3D<3,6,8>(ND,NE,map,b,d,x,y);
      case 0x845: return NDK_FORALL_SmemPAMassApply3D<4,5,4>(ND,NE,map,b,d,x,y);//3
      case 0x846: return NDK_FORALL_SmemPAMassApply3D<4,6,4>(ND,NE,map,b,d,x,y);
      case 0x848: return NDK_FORALL_SmemPAMassApply3D<4,8,4>(ND,NE,map,b,d,x,y);
      case 0x856: return NDK_FORALL_SmemPAMassApply3D<5,6,4>(ND,NE,map,b,d,x,y);//4
      case 0x858: return NDK_FORALL_SmemPAMassApply3D<5,8,1>(ND,NE,map,b,d,x,y);
      case 0x867: return NDK_FORALL_SmemPAMassApply3D<6,7,1>(ND,NE,map,b,d,x,y);//5
      case 0x878: return NDK_FORALL_SmemPAMassApply3D<7,8,1>(ND,NE,map,b,d,x,y);//6

      default: break;
   }

   MFEM_ABORT("Unknown kernel 0x" << std::hex << id);
}

} // namespace mfem
