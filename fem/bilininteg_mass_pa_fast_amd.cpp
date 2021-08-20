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

// Fast '3' non-deterministic 3D mass kernel
template<int D1D, int Q1D>
void NDK_AMD_PAMassApply3D(const int ndofs,
                           const int NE,
                           const int *map,
                           const double *b_,
                           const double *d_,
                           const double *x_,
                           double *y_)
{
   const auto MAP = Reshape(map, D1D,D1D,D1D, NE);
   const auto B = Reshape(b_, Q1D,D1D);
   const auto D = Reshape(d_, Q1D,Q1D,Q1D, NE);
   const auto X = Reshape(x_, ndofs);
   const auto X1 = Reshape(x_, D1D,D1D,D1D, NE);
   auto Y = Reshape(y_, ndofs);
   auto Y1 = Reshape(y_, D1D,D1D,D1D, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, 1,
   {
      double r_wk[Q1D];
      MFEM_SHARED double s_B[Q1D][D1D];
      MFEM_SHARED double s_q[Q1D][Q1D][Q1D];

      // Load s_B, load X in shared memory
      MFEM_FOREACH_THREAD(b,y,Q1D)
      {
         MFEM_FOREACH_THREAD(a,x,Q1D)
         {
            if (a<D1D) { s_B[b][a] = B(b,a); }

            MFEM_UNROLL(Q1D)
            for (int i=0; i<Q1D; ++i) { r_wk[i] = 0.0; }

            if (a<D1D && b<D1D)
            {
               MFEM_UNROLL(D1D)
               for (int c=0; c<D1D; ++c)
               {
                  const int gid = map ? MAP(a,b,c,e) : 0;
                  const int idx = gid >= 0 ? gid : -1 - gid;
                  s_q[c][b][a] = map ? X(idx) : X1(a,b,c,e);
               }
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Interpolate in X direction
      MFEM_FOREACH_THREAD(c,y,Q1D)
      {
         MFEM_FOREACH_THREAD(b,x,Q1D)
         {
            if (b<D1D && c<D1D)
            {
               MFEM_UNROLL(D1D)
               for (int a=0; a<D1D; ++a)
               {
                  const double q_cba = s_q[c][b][a];
                  MFEM_UNROLL(Q1D)
                  for (int i=0; i<Q1D; ++i) { r_wk[i] += s_B[i][a]*q_cba; }
               }
               // reg => s_mem
               MFEM_UNROLL(Q1D)
               for (int i=0; i<Q1D; ++i) { s_q[c][b][i] = r_wk[i]; }
            }
            MFEM_UNROLL(Q1D)
            for (int j=0; j<Q1D; ++j) { r_wk[j] = 0.0; }
         }
      }
      MFEM_SYNC_THREAD;

      // Interpolate in Y direction
      MFEM_FOREACH_THREAD(c,y,Q1D)
      {
         MFEM_FOREACH_THREAD(i,x,Q1D)
         {
            if (c<D1D)
            {
               MFEM_UNROLL(D1D)
               for (int b=0; b<D1D; ++b)
               {
                  const double q_cbi = s_q[c][b][i];
                  MFEM_UNROLL(Q1D)
                  for (int j=0; j<Q1D; ++j) { r_wk[j] += s_B[j][b]*q_cbi; }
               }
               MFEM_UNROLL(Q1D)
               for (int j=0; j<Q1D; ++j) { s_q[c][j][i] = r_wk[j]; }
            }
            MFEM_UNROLL(Q1D)
            for (int k=0; k<Q1D; ++k) { r_wk[k] = 0.0; }
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(j,y,Q1D)
      {
         MFEM_FOREACH_THREAD(i,x,Q1D)
         {
            // Interpolate in Z direction
            MFEM_UNROLL(D1D)
            for (int c=0; c<D1D; ++c)
            {
               const double q_cji = s_q[c][j][i];
               MFEM_UNROLL(Q1D)
               for (int k=0; k<Q1D; ++k) { r_wk[k] += s_B[k][c]*q_cji; }
            }

            // Scale by Jacobian and integration weights
            MFEM_UNROLL(Q1D)
            for (int k=0; k<Q1D; ++k) { r_wk[k] *= D(i,j,k,e); }

            // Project back in Z direction
            MFEM_UNROLL(D1D)
            for (int c=0; c<D1D; ++c)
            {
               double q_cji = 0.0;
               MFEM_UNROLL(Q1D)
               for (int k=0; k<Q1D; ++k) { q_cji += s_B[k][c] * r_wk[k]; }
               s_q[c][j][i] = q_cji;
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Project back in Y direction
      MFEM_FOREACH_THREAD(c,y,D1D)
      {
         MFEM_FOREACH_THREAD(i,x,Q1D)
         {
            MFEM_UNROLL(Q1D)
            for (int j=0; j<Q1D; ++j) { r_wk[j] = s_q[c][j][i]; }
            MFEM_UNROLL(D1D)
            for (int b=0; b<D1D; ++b)
            {
               double q_cbi = 0.0;
               MFEM_UNROLL(Q1D)
               for (int j=0; j<Q1D; ++j) { q_cbi += s_B[j][b] * r_wk[j]; }
               s_q[c][b][i] = q_cbi;
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Project back in X direction
      MFEM_FOREACH_THREAD(c,y,D1D)
      {
         MFEM_FOREACH_THREAD(b,x,D1D)
         {
            MFEM_UNROLL(Q1D)
            for (int i=0; i<Q1D; ++i) { r_wk[i] = s_q[c][b][i]; }
            MFEM_UNROLL(D1D)
            for (int a=0; a<D1D; ++a)
            {
               double q_cba = 0.0;
               MFEM_UNROLL(Q1D)
               for (int i=0; i<Q1D; ++i) { q_cba += s_B[i][a] * r_wk[i]; }
               s_q[c][b][a] = q_cba;
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Save back to memory
      MFEM_FOREACH_THREAD(b,y,D1D)
      {
         MFEM_FOREACH_THREAD(a,x,D1D)
         {
            MFEM_UNROLL(D1D)
            for (int c=0; c<D1D; ++c)
            {
               const double q_cba = s_q[c][b][a];
               const int gid = map ? MAP(a,b,c,e) : 0;
               const int idx = gid >= 0 ? gid : -1 - gid;
               AtomicAdd(map?Y(idx):Y1(a,b,c,e), q_cba);
            }
         }
      }
      MFEM_SYNC_THREAD;
   });
}

void NDK_AMD_PAMassApply(const int dim,
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
      // Fast '3': libP + AMD specific non-deterministic 3D mass kernel
      case 0x323: return NDK_AMD_PAMassApply3D<2,3>(ND,NE,map,b,d,x,y);
      case 0x324: return NDK_AMD_PAMassApply3D<2,4>(ND,NE,map,b,d,x,y);
      case 0x334: return NDK_AMD_PAMassApply3D<3,4>(ND,NE,map,b,d,x,y);
      case 0x336: return NDK_AMD_PAMassApply3D<3,6>(ND,NE,map,b,d,x,y);
      case 0x345: return NDK_AMD_PAMassApply3D<4,5>(ND,NE,map,b,d,x,y);
      case 0x346: return NDK_AMD_PAMassApply3D<4,6>(ND,NE,map,b,d,x,y);
      case 0x348: return NDK_AMD_PAMassApply3D<4,8>(ND,NE,map,b,d,x,y);
      case 0x356: return NDK_AMD_PAMassApply3D<5,6>(ND,NE,map,b,d,x,y);
      case 0x358: return NDK_AMD_PAMassApply3D<5,8>(ND,NE,map,b,d,x,y);
      case 0x367: return NDK_AMD_PAMassApply3D<6,7>(ND,NE,map,b,d,x,y);
      case 0x378: return NDK_AMD_PAMassApply3D<7,8>(ND,NE,map,b,d,x,y);

      default: break;
   }

   MFEM_ABORT("Unknown kernel 0x" << std::hex << id);
}

} // namespace mfem
