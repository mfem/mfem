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

#ifndef MFEM_BILININTEG_DGDIFFUSION_KERNELS_HPP
#define MFEM_BILININTEG_DGDIFFUSION_KERNELS_HPP

#include "../../general/forall.hpp"
#include "../../mesh/face_nbr_geom.hpp"
#include "../fe/face_map_utils.hpp"
#include "../gridfunc.hpp"
#include "../qfunction.hpp"

/// \cond DO_NOT_DOCUMENT
namespace mfem
{

namespace internal
{

template <int T_D1D = 0, int T_Q1D = 0>
static void PADGDiffusionApply2D(const int NF, const Array<real_t> &b,
                                 const Array<real_t> &bt,
                                 const Array<real_t> &g,
                                 const Array<real_t> &gt, const real_t sigma,
                                 const Vector &pa_data, const Vector &x_,
                                 const Vector &dxdn_, Vector &y_, Vector &dydn_,
                                 const int d1d = 0, const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   auto B_ = Reshape(b.Read(), Q1D, D1D);
   auto G_ = Reshape(g.Read(), Q1D, D1D);

   auto pa =
      Reshape(pa_data.Read(), 5, Q1D, NF); // (J00, J01, J10, J11, q/h)

   auto x = Reshape(x_.Read(), D1D, 2, NF);
   auto y = Reshape(y_.ReadWrite(), D1D, 2, NF);
   auto dxdn = Reshape(dxdn_.Read(), D1D, 2, NF);
   auto dydn = Reshape(dydn_.ReadWrite(), D1D, 2, NF);

   const int NBX = std::max(D1D, Q1D);

   mfem::forall_2D(NF, NBX, 2, [=] MFEM_HOST_DEVICE(int f) -> void
   {
      constexpr int max_D1D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      MFEM_SHARED real_t u0[max_D1D];
      MFEM_SHARED real_t u1[max_D1D];
      MFEM_SHARED real_t du0[max_D1D];
      MFEM_SHARED real_t du1[max_D1D];

      MFEM_SHARED real_t Bu0[max_Q1D];
      MFEM_SHARED real_t Bu1[max_Q1D];
      MFEM_SHARED real_t Bdu0[max_Q1D];
      MFEM_SHARED real_t Bdu1[max_Q1D];

      MFEM_SHARED real_t r[max_Q1D];

      MFEM_SHARED real_t BG[2 * max_D1D * max_Q1D];
      DeviceMatrix B(BG, Q1D, D1D);
      DeviceMatrix G(BG + D1D * Q1D, Q1D, D1D);

      if (MFEM_THREAD_ID(y) == 0)
      {
         MFEM_FOREACH_THREAD(p, x, Q1D)
         {
            for (int d = 0; d < D1D; ++d)
            {
               B(p, d) = B_(p, d);
               G(p, d) = G_(p, d);
            }
         }
      }
      MFEM_SYNC_THREAD;

      // copy edge values to u0, u1 and copy edge normals to du0, du1
      MFEM_FOREACH_THREAD(side, y, 2)
      {
         real_t *u = (side == 0) ? u0 : u1;
         real_t *du = (side == 0) ? du0 : du1;
         MFEM_FOREACH_THREAD(d, x, D1D)
         {
            u[d] = x(d, side, f);
            du[d] = dxdn(d, side, f);
         }
      }
      MFEM_SYNC_THREAD;

      // eval @ quad points
      MFEM_FOREACH_THREAD(side, y, 2)
      {
         real_t *u = (side == 0) ? u0 : u1;
         real_t *du = (side == 0) ? du0 : du1;
         real_t *Bu = (side == 0) ? Bu0 : Bu1;
         real_t *Bdu = (side == 0) ? Bdu0 : Bdu1;

         MFEM_FOREACH_THREAD(p, x, Q1D)
         {
            const real_t Je_side[] = {pa(2 * side + 0, p, f),
                                      pa(2 * side + 1, p, f)
                                     };

            Bu[p] = 0.0;
            Bdu[p] = 0.0;

            for (int d = 0; d < D1D; ++d)
            {
               const real_t b = B(p, d);
               const real_t g = G(p, d);

               Bu[p] += b * u[d];
               Bdu[p] += Je_side[0] * b * du[d] + Je_side[1] * g * u[d];
            }
         }
      }
      MFEM_SYNC_THREAD;

      // term - < {Q du/dn}, [v] > +  kappa * < {Q/h} [u], [v] >:
      if (MFEM_THREAD_ID(y) == 0)
      {
         MFEM_FOREACH_THREAD(p, x, Q1D)
         {
            const real_t q = pa(4, p, f);
            const real_t jump = Bu0[p] - Bu1[p];
            const real_t avg = Bdu0[p] + Bdu1[p]; // = {Q du/dn} * w * det(J)
            r[p] = -avg + q * jump;
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(d, x, D1D)
      {
         real_t Br = 0.0;

         for (int p = 0; p < Q1D; ++p)
         {
            Br += B(p, d) * r[p];
         }

         u0[d] = Br; // overwrite u0, u1
         u1[d] = -Br;
      } // for d
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(side, y, 2)
      {
         real_t *du = (side == 0) ? du0 : du1;
         MFEM_FOREACH_THREAD(d, x, D1D) { du[d] = 0.0; }
      }
      MFEM_SYNC_THREAD;

      // term sigma * < [u], {Q dv/dn} >
      MFEM_FOREACH_THREAD(side, y, 2)
      {
         real_t *const du = (side == 0) ? du0 : du1;
         real_t *const u = (side == 0) ? u0 : u1;

         MFEM_FOREACH_THREAD(d, x, D1D)
         {
            for (int p = 0; p < Q1D; ++p)
            {
               const real_t Je[] = {pa(2 * side + 0, p, f),
                                    pa(2 * side + 1, p, f)
                                   };
               const real_t jump = Bu0[p] - Bu1[p];
               const real_t r_p = Je[0] * jump; // normal
               const real_t w_p = Je[1] * jump; // tangential
               du[d] += sigma * B(p, d) * r_p;
               u[d] += sigma * G(p, d) * w_p;
            }
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(side, y, 2)
      {
         real_t *u = (side == 0) ? u0 : u1;
         real_t *du = (side == 0) ? du0 : du1;
         MFEM_FOREACH_THREAD(d, x, D1D)
         {
            y(d, side, f) += u[d];
            dydn(d, side, f) += du[d];
         }
      }
   }); // mfem::forall
}

template <int T_D1D = 0, int T_Q1D = 0>
static void PADGDiffusionApply3D(const int NF, const Array<real_t> &b,
                                 const Array<real_t> &bt,
                                 const Array<real_t> &g,
                                 const Array<real_t> &gt, const real_t sigma,
                                 const Vector &pa_data, const Vector &x_,
                                 const Vector &dxdn_, Vector &y_, Vector &dydn_,
                                 const int d1d = 0, const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   auto B_ = Reshape(b.Read(), Q1D, D1D);
   auto G_ = Reshape(g.Read(), Q1D, D1D);

   // (nJ[0], nJ[1], nJ[2], kappa * {Q}/h)
   const auto pa = Reshape(pa_data.Read(), 4, Q1D, Q1D, 2, NF);

   auto x = Reshape(x_.Read(), D1D, D1D, 2, NF);
   auto y = Reshape(y_.ReadWrite(), D1D, D1D, 2, NF);
   auto dxdn = Reshape(dxdn_.Read(), D1D, D1D, 2, NF);
   auto dydn = Reshape(dydn_.ReadWrite(), D1D, D1D, 2, NF);

   const int NBX = std::max(D1D, Q1D);

   mfem::forall_3D(NF, NBX, NBX, 2, [=] MFEM_HOST_DEVICE(int f) -> void
   {
      constexpr int max_D1D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      MFEM_SHARED real_t u[2][max_Q1D][max_Q1D];
      MFEM_SHARED real_t Bu[2][max_Q1D][max_Q1D];

      MFEM_SHARED real_t jmp[max_Q1D][max_Q1D];

      MFEM_SHARED real_t Y[2][max_Q1D][max_Q1D];

      MFEM_SHARED real_t B[max_Q1D][max_Q1D]; // not Q1D x D1D because we apply inplace transpose later
      MFEM_SHARED real_t G[max_Q1D][max_Q1D];

      // some buffers are reused multiple times, but for clarity have new names:
      real_t(*const du)[max_Q1D][max_Q1D] = u;
      
      real_t(*const Gu)[max_Q1D][max_Q1D] = Bu;
      real_t(*const Bdu)[max_Q1D][max_Q1D] = Bu;

      real_t(*const avg)[max_Q1D] = Y[0];
      real_t(*const R)[max_Q1D] = Y[0];

      real_t(*const Bj)[max_Q1D][max_Q1D] = Bu;
      real_t(*const Bjn)[max_Q1D][max_Q1D] = Bu;
      real_t(*const Gj)[max_Q1D][max_Q1D] = Bu;

      real_t(*const nJj)[max_Q1D][max_Q1D] = u;

      #ifdef __CUDA_ARCH__
      real_t _pa[4];
      {
         int p1 = MFEM_THREAD_ID(x);
         int p2 = MFEM_THREAD_ID(y);
         int side = MFEM_THREAD_ID(z);

         #ifdef MFEM_USE_SINGLE
         using vec_t = float4;
         #else // MFEM_USE_DOUBLE
         using vec_t = double4; // use double4_16a for cuda >= 13
         #endif

         // coallesced read
         reinterpret_cast<vec_t&>(_pa) = *reinterpret_cast<const vec_t*>(&pa(0, p1, p2, side, f));
         // for (int l = 0; l < 4; ++l)
         //       _pa[l] = pa(l, p1, p2, side, f);
      }

      auto nJ = [&_pa](int l, int p1, int p2, int side) -> real_t
      {
         MFEM_ASSERT_KERNEL(p1 == MFEM_THREAD_ID(x)
                           && p2 == MFEM_THREAD_ID(y)
                           && side == MFEM_THREAD_ID(z),
                           "nJ accessed incorrectly by threads.");
         return _pa[l];
      };

      auto kappa_Qh = [&_pa](int p1, int p2) -> real_t
      {
         MFEM_ASSERT_KERNEL(p1 == MFEM_THREAD_ID(x) && p2 == MFEM_THREAD_ID(y),
                           "kappa_Qh accessed incorrectly by threads.");
         return _pa[3];
      };
      #else
      auto nJ = [&](int l, int p1, int p2, int side) -> real_t
      {
         return pa(l, p1, p2, side, f);
      };

      auto kappa_Qh = [&](int p1, int p2) -> real_t
      {
         return pa(3, p1, p2, 0, f);
      };
      #endif

      MFEM_FOREACH_THREAD_DIRECT(side, z, 2)
      {
         MFEM_FOREACH_THREAD_DIRECT(p1, x, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(p2, y, Q1D)
            {
               if (side == 0)
                  avg[p1][p2] = 0.0;
               else
                  jmp[p1][p2] = 0.0;
            }
         }

         MFEM_FOREACH_THREAD_DIRECT(p, x, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(d, y, D1D)
            {
               if (side == 0)
                  B[d][p] = B_(p, d);
               else
                  G[d][p] = G_(p, d);
            }
         }

         MFEM_FOREACH_THREAD_DIRECT(d1, x, D1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(d2, y, D1D)
            {
               u[side][d1][d2] = x(d1, d2, side, f);
            }
         }
      }
      MFEM_SYNC_THREAD;

      // compute u and Dy*u on quad points
      MFEM_FOREACH_THREAD_DIRECT(side, z, 2)
      {
         MFEM_FOREACH_THREAD_DIRECT(p1, x, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(d2, y, D1D)
            {
               real_t bu = 0.0;

               MFEM_UNROLL(max_D1D)
               for (int d1 = 0; d1 < D1D; ++d1)
               {
                  bu += B[d1][p1] * u[side][d1][d2];
               }

               Bu[side][d2][p1] = bu;
            }
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD_DIRECT(side, z, 2)
      {
         MFEM_FOREACH_THREAD_DIRECT(p1, x, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(p2, y, Q1D)
            {
               real_t bbu = 0.0;
               real_t gbu = 0.0;

               MFEM_UNROLL(max_D1D)
               for (int d2 = 0; d2 < D1D; ++d2)
               {
                  bbu += B[d2][p2] * Bu[side][d2][p1];
                  gbu += G[d2][p2] * Bu[side][d2][p1];
               }

               gbu *= nJ(2, p1, p2, side);
               bbu *= (side == 0) ? 1.0 : -1.0;

               AtomicAdd(avg[p1][p2], gbu); // at worst serialization of `side` by atomic add
               AtomicAdd(jmp[p1][p2], bbu);
            }
         }
      }
      MFEM_SYNC_THREAD;

      // compute Dx*u on quad points
      MFEM_FOREACH_THREAD_DIRECT(side, z, 2)
      {
         MFEM_FOREACH_THREAD_DIRECT(p1, x, Q1D)
         {
            MFEM_FOREACH_THREAD(d2, y, D1D)
            {
               real_t gu = 0.0;

               MFEM_UNROLL(max_D1D)
               for (int d1 = 0; d1 < D1D; ++d1)
               {
                  gu += G[d1][p1] * u[side][d1][d2];
               }

               Gu[side][d2][p1] = gu;
            }
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD_DIRECT(side, z, 2)
      {
         MFEM_FOREACH_THREAD_DIRECT(p1, x, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(p2, y, Q1D)
            {
               real_t bgu = 0.0;

               MFEM_UNROLL(max_D1D)
               for (int d2 = 0; d2 < D1D; ++d2)
               {
                  bgu += B[d2][p2] * Gu[side][d2][p1];
               }

               bgu *= nJ(1, p1, p2, side);

               AtomicAdd(avg[p1][p2], bgu);
            }
         }
      }
      // MFEM_SYNC_THREAD;

      // compute du on quadrature points
      MFEM_FOREACH_THREAD_DIRECT(side, z, 2)
      {
         MFEM_FOREACH_THREAD_DIRECT(d1, x, D1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(d2, y, D1D)
            {
               du[side][d1][d2] = dxdn(d1, d2, side, f);
            }
         }
      }
      MFEM_SYNC_THREAD;
      
      MFEM_FOREACH_THREAD_DIRECT(side, z, 2)
      {
         MFEM_FOREACH_THREAD_DIRECT(p1, x, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(d2, y, D1D)
            {
               real_t bdu = 0.0;

               MFEM_UNROLL(max_D1D)
               for (int d1 = 0; d1 < D1D; ++d1)
               {
                  bdu += B[d1][p1] * du[side][d1][d2];
               }

               Bdu[side][d2][p1] = bdu;
            }
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD_DIRECT(side, z, 2)
      {
         MFEM_FOREACH_THREAD_DIRECT(p1, x, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(p2, y, Q1D)
            {
               real_t bbdu = 0.0;

               MFEM_UNROLL(max_D1D)
               for (int d2 = 0; d2 < D1D; ++d2)
               {
                  bbdu += B[d2][p2] * Bdu[side][d2][p1];
               }

               bbdu *= nJ(0, p1, p2, side);

               AtomicAdd(avg[p1][p2], bbdu);
            }
         }
      }
      MFEM_SYNC_THREAD;

      // integrate
      MFEM_FOREACH_THREAD_DIRECT(side, z, 2)
      {
         real_t (*const mat)[max_Q1D] = (side == 0) ? B : G;

         MFEM_FOREACH_THREAD_DIRECT(p1, x, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(p2, y, Q1D)
            {
               if (side == 0)
               {
                  // r = - < {Q du/dn}, [v] > + kappa * < {Q/h} [u], [v] >
                  R[p1][p2] = -avg[p1][p2] + kappa_Qh(p1, p2) * jmp[p1][p2];
               }

               nJj[side][p1][p2] = nJ(1, p1, p2, side) * jmp[p1][p2];


               // transpose B and G for better access
               if (p1 < p2)
               {
                  real_t tmp = mat[p1][p2];
                  mat[p1][p2] = mat[p2][p1];
                  mat[p2][p1] = tmp;
               }
            }
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD_DIRECT(side, z, 2)
      {
         MFEM_FOREACH_THREAD_DIRECT(d1, x, D1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(p2, y, Q1D)
            {
               real_t br = 0.0, gj = 0.0;

               MFEM_UNROLL(max_Q1D)
               for (int p1 = 0; p1 < Q1D; ++p1)
               {
                  br += B[p1][d1] * R[p1][p2];
                  gj += G[p1][d1] * nJj[side][p1][p2];
               }

               const real_t sgn = (side == 0) ? 1.0 : -1.0;
               Gj[side][p2][d1] = sgn * br + sigma * gj;
            }
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD_DIRECT(side, z, 2)
      {
         MFEM_FOREACH_THREAD_DIRECT(d1, x, D1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(d2, y, D1D)
            {
               real_t bgj = 0.0;

               MFEM_UNROLL(max_Q1D)
               for (int p2 = 0; p2 < Q1D; ++p2)
               {
                  bgj += B[p2][d2] * Gj[side][p2][d1];
               }

               Y[side][d1][d2] = bgj;
            }
         }
      }
      // MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD_DIRECT(side, z, 2)
      {
         MFEM_FOREACH_THREAD_DIRECT(p1, x, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(p2, y, Q1D)
            {
               nJj[side][p1][p2] = nJ(2, p1, p2, side) * jmp[p1][p2];
            }
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD_DIRECT(side, z, 2)
      {
         MFEM_FOREACH_THREAD_DIRECT(d1, x, D1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(p2, y, Q1D)
            {
               real_t bjn = 0.0;

               MFEM_UNROLL(max_Q1D)
               for (int p1 = 0; p1 < Q1D; ++p1)
               {
                  bjn += B[p1][d1] * nJj[side][p1][p2];
               }

               Bjn[side][p2][d1] = sigma * bjn;
            }
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD_DIRECT(side, z, 2)
      {
         MFEM_FOREACH_THREAD_DIRECT(d1, x, D1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(d2, y, D1D)
            {
               real_t gbj = 0.0;

               MFEM_UNROLL(max_Q1D)
               for (int p2 = 0; p2 < Q1D; ++p2)
               {
                  gbj += G[p2][d2] * Bjn[side][p2][d1];
               }

               y(d1, d2, side, f) += gbj + Y[side][d1][d2];
            }
         }
      }
      // MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD_DIRECT(side, z, 2)
      {
         MFEM_FOREACH_THREAD_DIRECT(p1, x, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(p2, y, Q1D)
            {
               nJj[side][p1][p2] = nJ(0, p1, p2, side) * jmp[p1][p2];
            }
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD_DIRECT(side, z, 2)
      {
         MFEM_FOREACH_THREAD_DIRECT(d1, x, D1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(p2, y, Q1D)
            {
               real_t bj = 0.0;

               MFEM_UNROLL(max_Q1D)
               for (int p1 = 0; p1 < Q1D; ++p1)
               {
                  bj += B[p1][d1] * nJj[side][p1][p2];
               }

               Bj[side][p2][d1] = sigma * bj;
            }
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD_DIRECT(side, z, 2)
      {
         MFEM_FOREACH_THREAD_DIRECT(d1, x, D1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(d2, y, D1D)
            {
               real_t bbj = 0.0;

               MFEM_UNROLL(max_Q1D)
               for (int p2 = 0; p2 < Q1D; ++p2)
               {
                  bbj += B[p2][d2] * Bj[side][p2][d1];
               }

               dydn(d1, d2, side, f) += bbj;
            }
         }
      }
   });
}

} // namespace internal

template <int DIM, int D1D, int Q1D>
DGDiffusionIntegrator::ApplyKernelType
DGDiffusionIntegrator::ApplyPAKernels::Kernel()
{
   if constexpr (DIM == 2)
   {
      return internal::PADGDiffusionApply2D<D1D, Q1D>;
   }
   else if constexpr (DIM == 3)
   {
      return internal::PADGDiffusionApply3D<D1D, Q1D>;
   }
   MFEM_ABORT("");
}
} // namespace mfem
/// \endcond DO_NOT_DOCUMENT
#endif
