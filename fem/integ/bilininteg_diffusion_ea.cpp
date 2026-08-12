// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../../general/forall.hpp"
#include "../bilininteg.hpp"
#include "../gridfunc.hpp"

namespace mfem
{

template<int T_D1D = 0, int T_Q1D = 0>
static void EADiffusionAssemble1D(const int NE,
                                  const Array<real_t> &b,
                                  const Array<real_t> &g,
                                  const Vector &padata,
                                  Vector &eadata,
                                  const bool add,
                                  const int d1d = 0,
                                  const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   auto G = Reshape(g.Read(), Q1D, D1D);
   auto D = Reshape(padata.Read(), Q1D, NE);
   auto A = Reshape(eadata.ReadWrite(), D1D, D1D, NE);
   mfem::forall_2D(NE, D1D, D1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      real_t r_Gi[MQ1];
      real_t r_Gj[MQ1];
      for (int q = 0; q < Q1D; q++)
      {
         r_Gi[q] = G(q,MFEM_THREAD_ID(x));
         r_Gj[q] = G(q,MFEM_THREAD_ID(y));
      }
      MFEM_FOREACH_THREAD(i1,x,D1D)
      {
         MFEM_FOREACH_THREAD(j1,y,D1D)
         {
            real_t val = 0.0;
            for (int k1 = 0; k1 < Q1D; ++k1)
            {
               val += r_Gj[k1] * D(k1, e) * r_Gi[k1];
            }
            if (add)
            {
               A(i1, j1, e) += val;
            }
            else
            {
               A(i1, j1, e) = val;
            }
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
static void EADiffusionAssemble2D(const int NE,
                                  const Array<real_t> &b,
                                  const Array<real_t> &g,
                                  const Vector &padata,
                                  Vector &eadata,
                                  const bool add,
                                  const int d1d = 0,
                                  const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   auto D = Reshape(padata.Read(), Q1D, Q1D, 3, NE);
   auto A = Reshape(eadata.ReadWrite(), D1D, D1D, D1D, D1D, NE);
   mfem::forall_2D(NE, D1D, D1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      real_t r_B[MQ1][MD1];
      real_t r_G[MQ1][MD1];
      for (int d = 0; d < D1D; d++)
      {
         for (int q = 0; q < Q1D; q++)
         {
            r_B[q][d] = B(q,d);
            r_G[q][d] = G(q,d);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(i1,x,D1D)
      {
         MFEM_FOREACH_THREAD(i2,y,D1D)
         {
            for (int j1 = 0; j1 < D1D; ++j1)
            {
               for (int j2 = 0; j2 < D1D; ++j2)
               {
                  real_t val = 0.0;
                  for (int k1 = 0; k1 < Q1D; ++k1)
                  {
                     for (int k2 = 0; k2 < Q1D; ++k2)
                     {
                        real_t bgi = r_G[k1][i1] * r_B[k2][i2];
                        real_t gbi = r_B[k1][i1] * r_G[k2][i2];
                        real_t bgj = r_G[k1][j1] * r_B[k2][j2];
                        real_t gbj = r_B[k1][j1] * r_G[k2][j2];
                        real_t D00 = D(k1,k2,0,e);
                        real_t D10 = D(k1,k2,1,e);
                        real_t D01 = D10;
                        real_t D11 = D(k1,k2,2,e);
                        val += bgi * D00 * bgj
                               + gbi * D01 * bgj
                               + bgi * D10 * gbj
                               + gbi * D11 * gbj;
                     }
                  }
                  if (add)
                  {
                     A(i1, i2, j1, j2, e) += val;
                  }
                  else
                  {
                     A(i1, i2, j1, j2, e) = val;
                  }
               }
            }
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
static void EADiffusionAssemble3D(const int NE,
                                  const Array<real_t> &b,
                                  const Array<real_t> &g,
                                  const Vector &padata,
                                  Vector &eadata,
                                  const bool add,
                                  const int d1d = 0,
                                  const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   auto D = Reshape(padata.Read(), Q1D, Q1D, Q1D, 6, NE);
   auto A = Reshape(eadata.ReadWrite(), D1D, D1D, D1D, D1D, D1D, D1D, NE);
   mfem::forall_3D(NE, D1D, D1D, D1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      real_t r_B[MQ1][MD1];
      real_t r_G[MQ1][MD1];
      for (int d = 0; d < D1D; d++)
      {
         for (int q = 0; q < Q1D; q++)
         {
            r_B[q][d] = B(q,d);
            r_G[q][d] = G(q,d);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(i1,x,D1D)
      {
         MFEM_FOREACH_THREAD(i2,y,D1D)
         {
            MFEM_FOREACH_THREAD(i3,z,D1D)
            {
               for (int j1 = 0; j1 < D1D; ++j1)
               {
                  for (int j2 = 0; j2 < D1D; ++j2)
                  {
                     for (int j3 = 0; j3 < D1D; ++j3)
                     {
                        real_t val = 0.0;
                        for (int k1 = 0; k1 < Q1D; ++k1)
                        {
                           for (int k2 = 0; k2 < Q1D; ++k2)
                           {
                              for (int k3 = 0; k3 < Q1D; ++k3)
                              {
                                 real_t bbgi = r_G[k1][i1] * r_B[k2][i2] * r_B[k3][i3];
                                 real_t bgbi = r_B[k1][i1] * r_G[k2][i2] * r_B[k3][i3];
                                 real_t gbbi = r_B[k1][i1] * r_B[k2][i2] * r_G[k3][i3];
                                 real_t bbgj = r_G[k1][j1] * r_B[k2][j2] * r_B[k3][j3];
                                 real_t bgbj = r_B[k1][j1] * r_G[k2][j2] * r_B[k3][j3];
                                 real_t gbbj = r_B[k1][j1] * r_B[k2][j2] * r_G[k3][j3];
                                 real_t D00 = D(k1,k2,k3,0,e);
                                 real_t D10 = D(k1,k2,k3,1,e);
                                 real_t D20 = D(k1,k2,k3,2,e);
                                 real_t D01 = D10;
                                 real_t D11 = D(k1,k2,k3,3,e);
                                 real_t D21 = D(k1,k2,k3,4,e);
                                 real_t D02 = D20;
                                 real_t D12 = D21;
                                 real_t D22 = D(k1,k2,k3,5,e);
                                 val += bbgi * D00 * bbgj
                                        + bgbi * D10 * bbgj
                                        + gbbi * D20 * bbgj
                                        + bbgi * D01 * bgbj
                                        + bgbi * D11 * bgbj
                                        + gbbi * D21 * bgbj
                                        + bbgi * D02 * gbbj
                                        + bgbi * D12 * gbbj
                                        + gbbi * D22 * gbbj;
                              }
                           }
                        }
                        if (add)
                        {
                           A(i1, i2, i3, j1, j2, j3, e) += val;
                        }
                        else
                        {
                           A(i1, i2, i3, j1, j2, j3, e) = val;
                        }
                     }
                  }
               }
            }
         }
      }
   });
}

void DiffusionIntegrator::AssembleEA(const FiniteElementSpace &fes,
                                     Vector &ea_data,
                                     const bool add)
{
   AssemblePA(fes);
   ne = fes.GetMesh()->GetNE();
   const Array<real_t> &B = maps->B;
   const Array<real_t> &G = maps->G;
   if (dim == 1)
   {
      switch ((dofs1D << 4 ) | quad1D)
      {
         case 0x22: return EADiffusionAssemble1D<2,2>(ne,B,G,pa_data,ea_data,add);
         case 0x33: return EADiffusionAssemble1D<3,3>(ne,B,G,pa_data,ea_data,add);
         case 0x44: return EADiffusionAssemble1D<4,4>(ne,B,G,pa_data,ea_data,add);
         case 0x55: return EADiffusionAssemble1D<5,5>(ne,B,G,pa_data,ea_data,add);
         case 0x66: return EADiffusionAssemble1D<6,6>(ne,B,G,pa_data,ea_data,add);
         case 0x77: return EADiffusionAssemble1D<7,7>(ne,B,G,pa_data,ea_data,add);
         case 0x88: return EADiffusionAssemble1D<8,8>(ne,B,G,pa_data,ea_data,add);
         case 0x99: return EADiffusionAssemble1D<9,9>(ne,B,G,pa_data,ea_data,add);
         default:   return EADiffusionAssemble1D(ne,B,G,pa_data,ea_data,add,
                                                    dofs1D,quad1D);
      }
   }
   else if (dim == 2)
   {
      switch ((dofs1D << 4 ) | quad1D)
      {
         case 0x22: return EADiffusionAssemble2D<2,2>(ne,B,G,pa_data,ea_data,add);
         case 0x33: return EADiffusionAssemble2D<3,3>(ne,B,G,pa_data,ea_data,add);
         case 0x44: return EADiffusionAssemble2D<4,4>(ne,B,G,pa_data,ea_data,add);
         case 0x55: return EADiffusionAssemble2D<5,5>(ne,B,G,pa_data,ea_data,add);
         case 0x66: return EADiffusionAssemble2D<6,6>(ne,B,G,pa_data,ea_data,add);
         case 0x77: return EADiffusionAssemble2D<7,7>(ne,B,G,pa_data,ea_data,add);
         case 0x88: return EADiffusionAssemble2D<8,8>(ne,B,G,pa_data,ea_data,add);
         case 0x99: return EADiffusionAssemble2D<9,9>(ne,B,G,pa_data,ea_data,add);
         default:   return EADiffusionAssemble2D(ne,B,G,pa_data,ea_data,add,
                                                    dofs1D,quad1D);
      }
   }
   else if (dim == 3)
   {
      switch ((dofs1D << 4 ) | quad1D)
      {
         case 0x23: return EADiffusionAssemble3D<2,3>(ne,B,G,pa_data,ea_data,add);
         case 0x34: return EADiffusionAssemble3D<3,4>(ne,B,G,pa_data,ea_data,add);
         case 0x45: return EADiffusionAssemble3D<4,5>(ne,B,G,pa_data,ea_data,add);
         case 0x56: return EADiffusionAssemble3D<5,6>(ne,B,G,pa_data,ea_data,add);
         case 0x67: return EADiffusionAssemble3D<6,7>(ne,B,G,pa_data,ea_data,add);
         case 0x78: return EADiffusionAssemble3D<7,8>(ne,B,G,pa_data,ea_data,add);
         case 0x89: return EADiffusionAssemble3D<8,9>(ne,B,G,pa_data,ea_data,add);
         default:   return EADiffusionAssemble3D(ne,B,G,pa_data,ea_data,add,
                                                    dofs1D,quad1D);
      }
   }
   MFEM_ABORT("Unknown kernel.");
}

}
