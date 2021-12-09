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

// Fast '7' HIP
template<int D1D, int Q1D, int NBZ=1, int NBK=1> MFEM_GLOBAL static
//MFEM_LAUNCH_BOUNDS(Q1D*Q1D*NBZ,NBK)
void HIP_PAMassApply(const int NE,
                     const int* MAP,
                     const double* B,
                     const double* D,
                     const double* X,
                     double* Y)
{
   double u[Q1D];
   const int tz = MFEM_THREAD_ID(z);
   MFEM_SHARED double s_B[Q1D][D1D];
   MFEM_SHARED double s_q[NBZ][Q1D][Q1D][Q1D];

   for (int be = MFEM_BLOCK_ID(x); be < (NE+NBZ-1)/NBZ; be += MFEM_GRID_DIM(x))
   {
      const int e = be * NBZ + tz;
      if (e>=NE) { return; }

      // Load input, B & X interpolation
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            if (tz == 0) { s_B[qx][dy] = B[qx+Q1D*dy]; }
            MFEM_UNROLL(D1D)
            for (int dz = 0; dz < D1D; ++dz) { u[dz] = 0.0; }
            MFEM_UNROLL(D1D)
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double Bx = B[qx+Q1D*dx];
               MFEM_UNROLL(D1D)
               for (int dz = 0; dz < D1D; ++dz)
               {
                  const int gid = MAP[e*D1D*D1D*D1D+dz*D1D*D1D+dy*D1D+dx];
                  const int idx = gid >= 0 ? gid : -1 - gid;
                  u[dz] = fma(X[idx], Bx, u[dz]);
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
               MFEM_UNROLL(Q1D)
               for (int qy = 0; qy < Q1D; ++qy) { u[qy] = fma(zyX,s_B[qy][dy],u[qy]); }
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
               for (int qz = 0; qz < Q1D; ++qz) { u[qz] = fma(zYX,s_B[qz][dz],u[qz]); }
            }

            // Q-function
            MFEM_UNROLL(Q1D)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               const int idx = e*Q1D*Q1D*Q1D + qx + qy*Q1D + qz*Q1D*Q1D;
               s_q[tz][qz][qy][qx] = u[qz] * D[idx];
            }

            // Zt projection
            MFEM_UNROLL(D1D)
            for (int dz = 0; dz < D1D; ++dz) { u[dz] = 0.0; }
            MFEM_UNROLL(Q1D)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double ZYX = s_q[tz][qz][qy][qx];
               MFEM_UNROLL(D1D)
               for (int dz = 0; dz < D1D; ++dz) { u[dz] = fma(ZYX,s_B[qz][dz],u[dz]); }
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
               for (int dy = 0; dy < D1D; ++dy) { u[dy] = fma(zYX,s_B[qy][dy],u[dy]); }
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
               for (int dx = 0; dx < D1D; ++dx) { u[dx] = fma(zyX,s_B[qx][dx],u[dx]); }
            }
            MFEM_UNROLL(D1D)
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double output = u[dx];
               const int gid = MAP[e*D1D*D1D*D1D+dz*D1D*D1D+dy*D1D+dx];
               const int idx = gid >= 0 ? gid : -1 - gid;
               AtomicAdd(Y[idx], output);
            }
         }
      }
      MFEM_SYNC_THREAD;
   }
}


void NDK_HIP_PAMassApply(const int dim,
                         const int D1D,
                         const int Q1D,
                         const int NE,
                         const FiniteElementSpace *fes,
                         const DofToQuad *maps,
                         const Vector &d,
                         const Vector &x,
                         Vector &y)
{
   assert(dim == 3);

   constexpr ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *ERop = fes->GetElementRestriction(ordering);
   const ElementRestriction *ER = dynamic_cast<const ElementRestriction*>(ERop);
   assert(ER);

   const auto M = ER->GatherMap().Read();
   const double *B = maps->B.Read();
   const double *D = d.Read();
   const double *X = x.Read();
   double *Y = y.ReadWrite();

   void (*Ker)(const int NE,
               const int *M,
               const double *B,
               const double *D,
               const double *X,
               double *Y) = nullptr;

   const int ver = Device::KernelsVersion();
   const int id = (ver << 8) | (D1D << 4) | Q1D;

   switch (id) // orders 1~8
   {
      case 0x723: Ker=HIP_PAMassApply<2,3>; break; // 1
      case 0x734: Ker=HIP_PAMassApply<3,4>; break; // 2
      case 0x745: Ker=HIP_PAMassApply<4,5>; break; // 3
      case 0x756: Ker=HIP_PAMassApply<5,6>; break; // 4
      case 0x767: Ker=HIP_PAMassApply<6,7>; break; // 5
      case 0x778: Ker=HIP_PAMassApply<7,8>; break; // 6
      case 0x789: Ker=HIP_PAMassApply<8,9>; break; // 7
      case 0x79A: Ker=HIP_PAMassApply<9,10>; break; // 8
      default: MFEM_ABORT("Unknown kernel 0x" << std::hex << id << std::dec);
   }
   MFEM_LAUNCH_KERNEL(Ker,NE,dim3(Q1D,Q1D,1),0,NE,M,B,D,X,Y);
}

} // namespace mfem
