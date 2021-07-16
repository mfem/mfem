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
static void AMD_PAMassDiag3D(const int ndofs,
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

void AMD_PAMassAssembleDiagonal(const int dim,
                                const int D1D,
                                const int Q1D,
                                const int NE,
                                const FiniteElementSpace *fes,
                                const DofToQuad *maps,
                                const Vector &D,
                                Vector &Y)
{
   const int ndofs = fes->GetNDofs();
   constexpr ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *ERop = fes->GetElementRestriction(ordering);
   const ElementRestriction* ER = dynamic_cast<const ElementRestriction*>(ERop);
   assert(ER);
   const int *map = ER->GatherMap().Read();
   const double *b = maps->B.Read();
   const double *d = D.Read();
   double *y = Y.ReadWrite();

   assert(dim == 3);
   const int id = (D1D << 4) | Q1D;

   switch (id) // orders 1~6
   {
      case 0x23: return AMD_PAMassDiag3D<2,3>(ndofs,NE,map,b,d,y);
      case 0x24: return AMD_PAMassDiag3D<2,4>(ndofs,NE,map,b,d,y);
      case 0x34: return AMD_PAMassDiag3D<3,4>(ndofs,NE,map,b,d,y);
      case 0x45: return AMD_PAMassDiag3D<4,5>(ndofs,NE,map,b,d,y);
      case 0x56: return AMD_PAMassDiag3D<5,6>(ndofs,NE,map,b,d,y);
      case 0x67: return AMD_PAMassDiag3D<6,7>(ndofs,NE,map,b,d,y);
      case 0x78: return AMD_PAMassDiag3D<7,8>(ndofs,NE,map,b,d,y);
      default: break;
   }

   MFEM_ABORT("Unknown kernel 0x" << std::hex << id);
}

} // namespace mfem
