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

////////////////////////////////////////////////////////////////////////////////
template<int D1D, int Q1D, int NBZ, int NBK> static
MFEM_GLOBAL MFEM_LAUNCH_BOUNDS(Q1D*Q1D*NBZ,NBK)
void NDK_PADiffApply(const int NE,
                     const int *MAP,
                     const double *B,
                     const double *G,
                     const double *D,
                     const double *X,
                     double *Y)
{
   double r_qt, r_q[Q1D];
   MFEM_SHARED double s_B[D1D][Q1D];
   MFEM_SHARED double s_G[Q1D][Q1D];
   MFEM_SHARED double s_Iq[NBZ][Q1D][Q1D][Q1D];
   MFEM_SHARED double s_Gqr[NBZ][Q1D][Q1D];
   MFEM_SHARED double s_Gqs[NBZ][Q1D][Q1D];

   for (int be = MFEM_BLOCK_ID(x); be < (NE+NBZ-1)/NBZ; be += MFEM_GRID_DIM(x))
   {
      const int tz = MFEM_THREAD_ID(z);
      const int e = be * NBZ + tz;
      if (e>=NE) { return; }

      // Scatter X
      MFEM_FOREACH_THREAD(j,y,Q1D)
      {
         MFEM_FOREACH_THREAD(i,x,Q1D)
         {
            if (tz == 0) { s_G[j][i] = G[i+Q1D*j]; } // ok with init
            if (tz == 0 && j<D1D) { s_B[j][i] = B[i+Q1D*j]; } // ok
            if (j<D1D && i<D1D)
            {
               MFEM_UNROLL(D1D)
               for (int k = 0; k < D1D; k++)
               {
                  const int gid = MAP[e*D1D*D1D*D1D + k*D1D*D1D + j*D1D + i];
                  const int idx = gid >= 0 ? gid : -1 - gid;
                  r_q[k] = X[idx];
               }
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Grad1X
      MFEM_FOREACH_THREAD(b,y,D1D)
      {
         MFEM_FOREACH_THREAD(a,x,D1D)
         {
            MFEM_UNROLL(Q1D)
            for (int k=0; k<Q1D; ++k)
            {
               double u = 0.0;
               MFEM_UNROLL(D1D)
               for (int c=0; c<D1D; ++c) { u += s_B[c][k] * r_q[c]; }
               s_Iq[tz][k][b][a] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Grad1Y
      MFEM_FOREACH_THREAD(k,y,Q1D)
      {
         MFEM_FOREACH_THREAD(a,x,D1D)
         {
            MFEM_UNROLL(D1D)
            for (int b=0; b<D1D; ++b) { r_q[b] = s_Iq[tz][k][b][a]; }
            MFEM_UNROLL(Q1D)
            for (int j=0; j<Q1D; ++j)
            {
               double u = 0.0;
               MFEM_UNROLL(D1D)
               for (int b=0; b<D1D; ++b) { u += s_B[b][j] * r_q[b]; }
               s_Iq[tz][k][j][a] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Grad1Z
      MFEM_FOREACH_THREAD(k,y,Q1D)
      {
         MFEM_FOREACH_THREAD(j,x,Q1D)
         {
            MFEM_UNROLL(D1D)
            for (int a=0; a<D1D; ++a) { r_q[a] = s_Iq[tz][k][j][a]; }
            MFEM_UNROLL(Q1D)
            for (int i=0; i<Q1D; ++i)
            {
               double u = 0.0;
               MFEM_UNROLL(D1D)
               for (int a=0; a<D1D; ++a) { u += s_B[a][i] * r_q[a]; }
               s_Iq[tz][k][j][i] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Flush
      MFEM_FOREACH_THREAD(j,y,Q1D)
      {
         MFEM_FOREACH_THREAD(i,x,Q1D)
         {
            MFEM_UNROLL(Q1D)
            for (int k = 0; k < Q1D; ++k) { r_q[k] = 0.0; }
         }
      }
      MFEM_SYNC_THREAD;

      // Q-Function
      MFEM_UNROLL(Q1D)
      for (int k = 0; k < Q1D; ++k)
      {
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(j,y,Q1D)
         {
            MFEM_FOREACH_THREAD(i,x,Q1D)
            {
               double qr = 0.0, qs = 0.0, qt = 0.0;
               MFEM_UNROLL(Q1D)
               for (int m = 0; m < Q1D; ++m)
               {
                  const double Dim = s_G[i][m];
                  const double Djm = s_G[j][m];
                  const double Dkm = s_G[k][m];
                  qr += Dim * s_Iq[tz][k][j][m];
                  qs += Djm * s_Iq[tz][k][m][i];
                  qt += Dkm * s_Iq[tz][m][j][i];
               }
               //(d, Q1D,Q1D,Q1D, 6, NE);
               const int ebase = e * 6*Q1D*Q1D*Q1D;
               const int gbase = i + j*Q1D + k*Q1D*Q1D;
               const double D00 = D[0*Q1D*Q1D*Q1D + gbase + ebase];
               const double D01 = D[1*Q1D*Q1D*Q1D + gbase + ebase];
               const double D02 = D[2*Q1D*Q1D*Q1D + gbase + ebase];
               const double D11 = D[3*Q1D*Q1D*Q1D + gbase + ebase];
               const double D12 = D[4*Q1D*Q1D*Q1D + gbase + ebase];
               const double D22 = D[5*Q1D*Q1D*Q1D + gbase + ebase];

               s_Gqr[tz][j][i] = D00*qr + D01*qs + D02*qt;
               s_Gqs[tz][j][i] = D01*qr + D11*qs + D12*qt;
               r_qt = D02*qr + D12*qs + D22*qt;
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(j,y,Q1D)
         {
            MFEM_FOREACH_THREAD(i,x,Q1D)
            {
               double Aqtmp = 0.0;
               MFEM_UNROLL(Q1D)
               for (int m = 0; m < Q1D; ++m)
               {
                  const double Dmi = s_G[m][i];
                  const double Dmj = s_G[m][j];
                  const double Dkm = s_G[k][m];
                  Aqtmp += Dmi * s_Gqr[tz][j][m];
                  Aqtmp += Dmj * s_Gqs[tz][m][i];
                  r_q[m] += Dkm * r_qt;
               }
               r_q[k] += Aqtmp;
            }
         }
         MFEM_SYNC_THREAD;
      }

      // GradZT
      MFEM_FOREACH_THREAD(j,y,Q1D)
      {
         MFEM_FOREACH_THREAD(i,x,Q1D)
         {
            MFEM_UNROLL(D1D)
            for (int c=0; c<D1D; ++c)
            {
               double u = 0.0;
               MFEM_UNROLL(Q1D)
               for (int k=0; k<Q1D; ++k) { u += s_B[c][k] * r_q[k]; }
               s_Iq[tz][c][j][i] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;

      // GradYT
      MFEM_FOREACH_THREAD(c,y,D1D)
      {
         MFEM_FOREACH_THREAD(i,x,Q1D)
         {
            MFEM_UNROLL(Q1D)
            for (int j=0; j<Q1D; ++j) { r_q[j] = s_Iq[tz][c][j][i]; }
            MFEM_UNROLL(D1D)
            for (int b=0; b<D1D; ++b)
            {
               double u = 0.0;
               MFEM_UNROLL(Q1D)
               for (int j=0; j<Q1D; ++j) { u += s_B[b][j] * r_q[j]; }
               s_Iq[tz][c][b][i] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;

      // GradXT
      MFEM_FOREACH_THREAD(c,y,D1D)
      {
         MFEM_FOREACH_THREAD(b,x,D1D)
         {
            MFEM_UNROLL(Q1D)
            for (int i=0; i<Q1D; ++i) { r_q[i] = s_Iq[tz][c][b][i]; }
            MFEM_UNROLL(D1D)
            for (int a=0; a<D1D; ++a)
            {
               double u = 0.0;
               MFEM_UNROLL(Q1D)
               for (int i=0; i<Q1D; ++i) { u += s_B[a][i] * r_q[i]; }
               s_Iq[tz][c][b][a] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Gather
      MFEM_FOREACH_THREAD(j,y,D1D)
      {
         MFEM_FOREACH_THREAD(i,x,D1D)
         {
            MFEM_UNROLL(D1D)
            for (int k = 0; k < D1D; k++)
            {
               const int gid = MAP[e*D1D*D1D*D1D + k*D1D*D1D + j*D1D + i];
               const int idx = gid >= 0 ? gid : -1 - gid;
               const double output = s_Iq[tz][k][j][i];
               AtomicAdd(Y[idx], output);
            }
         }
      }
      MFEM_SYNC_THREAD;
   }
}

void NDK_PADiffusionApply(const int dim,
                          const int D1D,
                          const int Q1D,
                          const int NE,
                          const Vector &CoG,
                          const FiniteElementSpace *fes,
                          const DofToQuad *maps,
                          const Vector &D,
                          const Vector &X,
                          Vector &Y)
{
   //dbg();
   MFEM_NVTX;
   assert(dim == 3);

   const double *b = maps->B.Read();
   const double *d = D.Read();
   const double *x = X.Read();
   double *y = Y.ReadWrite();

   const int id = (D1D << 4) | Q1D;

   const int ND = fes->GetNDofs();
   constexpr ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *ERop = fes->GetElementRestriction(ordering);
   const ElementRestriction *ER = dynamic_cast<const ElementRestriction*>(ERop);
   assert(ER);
   const int *map = ER->GatherMap().Read();

   const auto dM = Reshape(map, D1D,D1D,D1D, NE);
   const auto dB = Reshape(b, Q1D,D1D);
   const auto dG = Reshape(CoG.Read(), Q1D,Q1D);
   const auto dD = Reshape(d, Q1D,Q1D,Q1D, 6, NE);
   const auto dX = Reshape(x, ND);
   auto dY = Reshape(y, ND);

   void (*Ker)(const int NE,
               const int *MAP,
               const double *B,
               const double *G,
               const double *D,
               const double *X,
               double *Y) = nullptr;

   int NBZ = 1;

   switch (id) // orders 1~8
   {
      case 0x23: Ker=NDK_PADiffApply<2,3,16,5>; NBZ=16; break; // 1
      case 0x34: Ker=NDK_PADiffApply<3,4,8,6>; NBZ=8; break; // 2
      case 0x45: Ker=NDK_PADiffApply<4,5,4,4>; NBZ=4; break; // 3
      case 0x56: Ker=NDK_PADiffApply<5,6,4,3>; NBZ=4; break; // 4
      case 0x67: Ker=NDK_PADiffApply<6,7,1,3>; break; // 5
      case 0x78: Ker=NDK_PADiffApply<7,8,1,0>; break; // 6
      //case 0x89: Ker=NDK_PADiffApply<8,9,1,0>; break; // 7
      //case 0x9A: Ker=NDK_PADiffApply<9,10,1,1>; break; // 8
      default: MFEM_ABORT("Unknown kernel 0x" << std::hex << id << std::dec);
   }
   MFEM_CONTRACT_VAR(NBZ);
   MFEM_LAUNCH_KERNEL(Ker,(NE+NBZ-1)/NBZ,dim3(Q1D,Q1D,NBZ),0,NE,dM,dB,dG,dD,dX,dY);
}

} // namespace mfem
