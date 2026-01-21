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

namespace mfem
{
template <int T_D1D = 0, int T_Q1D = 0>
static void EAVectorDiffusionAssemble2D(const int NE, const Array<real_t> &b,
                                        const Array<real_t> &g,
                                        const Vector &padata, Vector &eadata,
                                        const bool add, const int VDIM,
                                        const int d1d = 0, const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);

   // for the W J^{-1} J^T 2x2 tensor
   constexpr int pa_size = 4;
   MFEM_VERIFY(padata.Size() % (Q1D * Q1D * pa_size * NE) == 0,
               "This must be cleanly divisible for determing the channel size "
               "related to the diffusion coefficient type");
   const int num_channels = padata.Size() / (Q1D * Q1D * pa_size * NE);
   const bool diag_blocks = (num_channels ==
                             VDIM); // scalar or vector diffusion coefficient
   const bool full_blocks = (num_channels ==
                             VDIM * VDIM); // matrix diffusion coefficient
   MFEM_VERIFY(diag_blocks || full_blocks,
               "The channel size does not match what we would expect for "
               "different types of supported diffusion coefficients");
   auto D = Reshape(padata.Read(), Q1D, Q1D, pa_size, num_channels, NE);

   auto A = Reshape(eadata.ReadWrite(), D1D, D1D, VDIM, D1D, D1D, VDIM, NE);

   forall_2D(NE, D1D, D1D,
             [=] MFEM_HOST_DEVICE(int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      real_t r_B[MQ1][MD1];
      real_t r_G[MQ1][MD1];
      for (int d = 0; d < D1D; ++d)
      {
         for (int q = 0; q < Q1D; ++q)
         {
            r_B[q][d] = B(q, d);
            r_G[q][d] = G(q, d);
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(i1, x, D1D)
      {
         MFEM_FOREACH_THREAD(i2, y, D1D)
         {
            for (int j1 = 0; j1 < D1D; ++j1)
            {
               for (int j2 = 0; j2 < D1D; ++j2)
               {
                  real_t val_ch[4]; // Only used fully for matrix coefficients
                  for (int ch = 0; ch < num_channels; ++ch)
                  {
                     val_ch[ch] = 0.;
                  }

                  for (int k1 = 0; k1 < Q1D; ++k1)
                  {
                     for (int k2 = 0; k2 < Q1D; ++k2)
                     {
                        real_t g0i = r_G[k1][i1] * r_B[k2][i2];
                        real_t g1i = r_B[k1][i1] * r_G[k2][i2];
                        real_t g0j = r_G[k1][j1] * r_B[k2][j2];
                        real_t g1j = r_B[k1][j1] * r_G[k2][j2];

                        for (int ch = 0; ch < num_channels; ++ch)
                        {
                           real_t D00 = D(k1, k2, 0, ch, e);
                           real_t D01 = D(k1, k2, 1, ch, e);
                           real_t D10 = D(k1, k2, 2, ch, e);
                           real_t D11 = D(k1, k2, 3, ch, e);

                           val_ch[ch] += g0i * D00 * g0j +
                                         g0i * D01 * g1j +
                                         g1i * D10 * g0j +
                                         g1i * D11 * g1j;
                        }
                     }
                  }

                  if (diag_blocks)
                  {
                     for (int ci = 0; ci < VDIM; ++ci)
                     {
                        for (int cj = 0; cj < VDIM; ++cj)
                        {
                           const real_t val = ci == cj ? val_ch[ci]
                                              : 0.;
                           if (add)
                           {
                              A(i1, i2, ci, j1, j2, cj, e) += val;
                           }
                           else
                           {
                              A(i1, i2, ci, j1, j2, cj, e) = val;
                           }
                        }
                     }
                  }
                  else
                  {
                     // Full coupling. Need to honor mapping in PA
                     const int inv_map[4] = {0, 2, 1, 3};
                     for (int ci = 0; ci < VDIM; ++ci)
                     {
                        for (int cj = 0; cj < VDIM; ++cj)
                        {
                           const int index = ci * VDIM + cj;
                           const int ch = inv_map[index];
                           const real_t val = val_ch[ch];
                           if (add)
                           {
                              A(i1, i2, ci, j1, j2, cj, e) += val;
                           }
                           else
                           {
                              A(i1, i2, ci, j1, j2, cj, e) = val;
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   });
}

template <int T_D1D = 0, int T_Q1D = 0>
static void EAVectorDiffusionAssemble3D(const int NE, const Array<real_t> &b,
                                        const Array<real_t> &g,
                                        const Vector &padata, Vector &eadata,
                                        const bool add, const int VDIM,
                                        const int d1d = 0, const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   auto D = Reshape(padata.Read(), Q1D, Q1D, Q1D, 6, NE);

   auto A = Reshape(eadata.ReadWrite(), D1D, D1D, D1D, VDIM, D1D, D1D, D1D,
                    VDIM, NE);

   forall_3D(NE, D1D, D1D, D1D,
             [=] MFEM_HOST_DEVICE(int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      real_t r_B[MQ1][MD1];
      real_t r_G[MQ1][MD1];
      for (int d = 0; d < D1D; ++d)
      {
         for (int q = 0; q < Q1D; ++q)
         {
            r_B[q][d] = B(q, d);
            r_G[q][d] = G(q, d);
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(i1, x, D1D)
      {
         MFEM_FOREACH_THREAD(i2, y, D1D)
         {
            MFEM_FOREACH_THREAD(i3, z, D1D)
            {
               for (int j1 = 0; j1 < D1D; ++j1)
               {
                  for (int j2 = 0; j2 < D1D; ++j2)
                  {
                     for (int j3 = 0; j3 < D1D; ++j3)
                     {
                        real_t on_diag_val = 0.;
                        for (int k1 = 0; k1 < Q1D; ++k1)
                        {
                           for (int k2 = 0; k2 < Q1D; ++k2)
                           {
                              for (int k3 = 0; k3 < Q1D; ++k3)
                              {
                                 real_t g0i = r_G[k1][i1] *
                                              r_B[k2][i2] *
                                              r_B[k3][i3];
                                 real_t g1i = r_B[k1][i1] *
                                              r_G[k2][i2] *
                                              r_B[k3][i3];
                                 real_t g2i = r_B[k1][i1] *
                                              r_B[k2][i2] *
                                              r_G[k3][i3];
                                 real_t g0j = r_G[k1][j1] *
                                              r_B[k2][j2] *
                                              r_B[k3][j3];
                                 real_t g1j = r_B[k1][j1] *
                                              r_G[k2][j2] *
                                              r_B[k3][j3];
                                 real_t g2j = r_B[k1][j1] *
                                              r_B[k2][j2] *
                                              r_G[k3][j3];

                                 real_t D00 = D(k1, k2, k3, 0, e);
                                 real_t D10 = D(k1, k2, k3, 1, e);
                                 real_t D01 = D10;
                                 real_t D20 = D(k1, k2, k3, 2, e);
                                 real_t D02 = D20;
                                 real_t D11 = D(k1, k2, k3, 3, e);
                                 real_t D21 = D(k1, k2, k3, 4, e);
                                 real_t D12 = D21;
                                 real_t D22 = D(k1, k2, k3, 5, e);

                                 on_diag_val += g0i * D00 * g0j +
                                                g1i * D10 * g0j +
                                                g2i * D20 * g0j +
                                                g0i * D01 * g1j +
                                                g1i * D11 * g1j +
                                                g2i * D21 * g1j +
                                                g0i * D02 * g2j +
                                                g1i * D12 * g2j +
                                                g2i * D22 * g2j;
                              }
                           }
                        }

                        for (int ci = 0; ci < VDIM; ++ci)
                        {
                           for (int cj = 0; cj < VDIM; ++cj)
                           {
                              const real_t val = ci == cj ? on_diag_val
                                                 : 0.;
                              if (add)
                              {
                                 A(i1, i2, i3, ci, j1, j2, j3, cj,
                                   e) += val;
                              }
                              else
                              {
                                 A(i1, i2, i3, ci, j1, j2, j3, cj,
                                   e) = val;
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   });
}

void VectorDiffusionIntegrator::AssembleEA(const FiniteElementSpace &fes,
                                           Vector &ea_data, const bool add)
{
   AssemblePA(fes);
   ne = fes.GetMesh()->GetNE();
   const Array<real_t> &B = maps->B;
   const Array<real_t> &G = maps->G;
   if (dim == 1)
   {
      MFEM_ABORT("VectorDiffusionIntegrator::AssembleEA doesn't support a "
                 "single dimension");
   }
   else if (dim == 2)
   {
      switch ((dofs1D << 4) | quad1D)
      {
         case 0x22:
            return EAVectorDiffusionAssemble2D<2, 2>(ne, B, G, pa_data, ea_data,
                                                     add, vdim);
         case 0x33:
            return EAVectorDiffusionAssemble2D<3, 3>(ne, B, G, pa_data, ea_data,
                                                     add, vdim);
         case 0x44:
            return EAVectorDiffusionAssemble2D<4, 4>(ne, B, G, pa_data, ea_data,
                                                     add, vdim);
         case 0x55:
            return EAVectorDiffusionAssemble2D<5, 5>(ne, B, G, pa_data, ea_data,
                                                     add, vdim);
         case 0x66:
            return EAVectorDiffusionAssemble2D<6, 6>(ne, B, G, pa_data, ea_data,
                                                     add, vdim);
         case 0x77:
            return EAVectorDiffusionAssemble2D<7, 7>(ne, B, G, pa_data, ea_data,
                                                     add, vdim);
         case 0x88:
            return EAVectorDiffusionAssemble2D<8, 8>(ne, B, G, pa_data, ea_data,
                                                     add, vdim);
         case 0x99:
            return EAVectorDiffusionAssemble2D<9, 9>(ne, B, G, pa_data, ea_data,
                                                     add, vdim);
         default:
            return EAVectorDiffusionAssemble2D(ne, B, G, pa_data, ea_data, add,
                                               vdim, dofs1D, quad1D);
      }
   }
   else if (dim == 3)
   {
      switch ((dofs1D << 4) | quad1D)
      {
         case 0x23:
            return EAVectorDiffusionAssemble3D<2, 3>(ne, B, G, pa_data, ea_data,
                                                     add, vdim);
         case 0x34:
            return EAVectorDiffusionAssemble3D<3, 4>(ne, B, G, pa_data, ea_data,
                                                     add, vdim);
         case 0x45:
            return EAVectorDiffusionAssemble3D<4, 5>(ne, B, G, pa_data, ea_data,
                                                     add, vdim);
         case 0x56:
            return EAVectorDiffusionAssemble3D<5, 6>(ne, B, G, pa_data, ea_data,
                                                     add, vdim);
         case 0x67:
            return EAVectorDiffusionAssemble3D<6, 7>(ne, B, G, pa_data, ea_data,
                                                     add, vdim);
         case 0x78:
            return EAVectorDiffusionAssemble3D<7, 8>(ne, B, G, pa_data, ea_data,
                                                     add, vdim);
         case 0x89:
            return EAVectorDiffusionAssemble3D<8, 9>(ne, B, G, pa_data, ea_data,
                                                     add, vdim);
         default:
            return EAVectorDiffusionAssemble3D(ne, B, G, pa_data, ea_data, add,
                                               vdim, dofs1D, quad1D);
      }
   }
   MFEM_ABORT("Unknown kernel.");
}

} // namespace mfem
