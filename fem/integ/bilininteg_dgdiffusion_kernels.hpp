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
      Reshape(pa_data.Read(), 6, Q1D, NF); // (q, 1/h, J00, J01, J10, J11)

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
            const real_t Je_side[] = {pa(2 + 2 * side, p, f),
                                      pa(2 + 2 * side + 1, p, f)
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
            const real_t q = pa(0, p, f);
            const real_t hi = pa(1, p, f);
            const real_t jump = Bu0[p] - Bu1[p];
            const real_t avg = Bdu0[p] + Bdu1[p]; // = {Q du/dn} * w * det(J)
            r[p] = -avg + hi * q * jump;
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
               const real_t Je[] = {pa(2 + 2 * side, p, f),
                                    pa(2 + 2 * side + 1, p, f)
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

   // (J0[0], J0[1], J0[2], J1[0], J1[1], J1[2], q/h)
   auto pa = Reshape(pa_data.Read(), 7, Q1D, Q1D, NF);

   auto x = Reshape(x_.Read(), D1D, D1D, 2, NF);
   auto y = Reshape(y_.ReadWrite(), D1D, D1D, 2, NF);
   auto dxdn = Reshape(dxdn_.Read(), D1D, D1D, 2, NF);
   auto dydn = Reshape(dydn_.ReadWrite(), D1D, D1D, 2, NF);

   const int NBX = std::max(D1D, Q1D);

   mfem::forall_3D(NF, NBX, NBX, 2, [=] MFEM_HOST_DEVICE(int f) -> void
   {
      constexpr int max_D1D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      MFEM_SHARED real_t u0[max_Q1D][max_Q1D];
      MFEM_SHARED real_t u1[max_Q1D][max_Q1D];

      MFEM_SHARED real_t du0[max_Q1D][max_Q1D];
      MFEM_SHARED real_t du1[max_Q1D][max_Q1D];

      MFEM_SHARED real_t Gu0[max_Q1D][max_Q1D];
      MFEM_SHARED real_t Gu1[max_Q1D][max_Q1D];

      MFEM_SHARED real_t Bu0[max_Q1D][max_Q1D];
      MFEM_SHARED real_t Bu1[max_Q1D][max_Q1D];

      MFEM_SHARED real_t Bdu0[max_Q1D][max_Q1D];
      MFEM_SHARED real_t Bdu1[max_Q1D][max_Q1D];

      MFEM_SHARED real_t kappa_Qh[max_Q1D][max_Q1D];

      MFEM_SHARED real_t nJe[2][max_Q1D][max_Q1D][3];
      MFEM_SHARED real_t BG[2 * max_D1D * max_Q1D];

      // some buffers are reused multiple times, but for clarity have new names:
      real_t(*Bj0)[max_Q1D] = Bu0;
      real_t(*Bj1)[max_Q1D] = Bu1;
      real_t(*Bjn0)[max_Q1D] = Bdu0;
      real_t(*Bjn1)[max_Q1D] = Bdu1;
      real_t(*Gj0)[max_Q1D] = Gu0;
      real_t(*Gj1)[max_Q1D] = Gu1;

      DeviceMatrix B(BG, Q1D, D1D);
      DeviceMatrix G(BG + D1D * Q1D, Q1D, D1D);

      // copy face values to u0, u1 and copy normals to du0, du1
      MFEM_FOREACH_THREAD(side, z, 2)
      {
         real_t(*u)[max_Q1D] = (side == 0) ? u0 : u1;
         real_t(*du)[max_Q1D] = (side == 0) ? du0 : du1;

         MFEM_FOREACH_THREAD(d2, x, D1D)
         {
            MFEM_FOREACH_THREAD(d1, y, D1D)
            {
               u[d2][d1] = x(d1, d2, side,
                             f); // copy transposed for better memory access
               du[d2][d1] = dxdn(d1, d2, side, f);
            }
         }

         MFEM_FOREACH_THREAD(p1, x, Q1D)
         {
            MFEM_FOREACH_THREAD(p2, y, Q1D)
            {
               for (int l = 0; l < 3; ++l)
               {
                  nJe[side][p2][p1][l] = pa(3 * side + l, p1, p2, f);
               }

               if (side == 0)
               {
                  kappa_Qh[p2][p1] = pa(6, p1, p2, f);
               }
            }
         }

         if (side == 0)
         {
            MFEM_FOREACH_THREAD(p, x, Q1D)
            {
               MFEM_FOREACH_THREAD(d, y, D1D)
               {
                  B(p, d) = B_(p, d);
                  G(p, d) = G_(p, d);
               }
            }
         }
      }
      MFEM_SYNC_THREAD;

      // eval u and normal derivative @ quad points
      MFEM_FOREACH_THREAD(side, z, 2)
      {
         real_t(*u)[max_Q1D] = (side == 0) ? u0 : u1;
         real_t(*du)[max_Q1D] = (side == 0) ? du0 : du1;
         real_t(*Bu)[max_Q1D] = (side == 0) ? Bu0 : Bu1;
         real_t(*Bdu)[max_Q1D] = (side == 0) ? Bdu0 : Bdu1;
         real_t(*Gu)[max_Q1D] = (side == 0) ? Gu0 : Gu1;

         MFEM_FOREACH_THREAD(p1, x, Q1D)
         {
            MFEM_FOREACH_THREAD(d2, y, D1D)
            {
               real_t bu = 0.0;
               real_t bdu = 0.0;
               real_t gu = 0.0;

               for (int d1 = 0; d1 < D1D; ++d1)
               {
                  const real_t b = B(p1, d1);
                  const real_t g = G(p1, d1);

                  bu += b * u[d2][d1];
                  bdu += b * du[d2][d1];
                  gu += g * u[d2][d1];
               }

               Bu[p1][d2] = bu;
               Bdu[p1][d2] = bdu;
               Gu[p1][d2] = gu;
            }
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(side, z, 2)
      {
         real_t(*u)[max_Q1D] = (side == 0) ? u0 : u1;
         real_t(*du)[max_Q1D] = (side == 0) ? du0 : du1;
         real_t(*Bu)[max_Q1D] = (side == 0) ? Bu0 : Bu1;
         real_t(*Gu)[max_Q1D] = (side == 0) ? Gu0 : Gu1;
         real_t(*Bdu)[max_Q1D] = (side == 0) ? Bdu0 : Bdu1;

         MFEM_FOREACH_THREAD(p2, x, Q1D)
         {
            MFEM_FOREACH_THREAD(p1, y, Q1D)
            {
               const real_t *Je = nJe[side][p2][p1];

               real_t bbu = 0.0;
               real_t bgu = 0.0;
               real_t gbu = 0.0;
               real_t bbdu = 0.0;

               for (int d2 = 0; d2 < D1D; ++d2)
               {
                  const real_t b = B(p2, d2);
                  const real_t g = G(p2, d2);
                  bbu += b * Bu[p1][d2];
                  gbu += g * Bu[p1][d2];
                  bgu += b * Gu[p1][d2];
                  bbdu += b * Bdu[p1][d2];
               }

               u[p2][p1] = bbu;
               // du <- Q du/dn * w * det(J)
               du[p2][p1] = Je[0] * bbdu + Je[1] * bgu + Je[2] * gbu;
            }
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(side, z, 2)
      {
         real_t(*Bj)[max_Q1D] = (side == 0) ? Bj0 : Bj1;
         real_t(*Bjn)[max_Q1D] = (side == 0) ? Bjn0 : Bjn1;
         real_t(*Gj)[max_Q1D] = (side == 0) ? Gj0 : Gj1;

         MFEM_FOREACH_THREAD(d1, x, D1D)
         {
            MFEM_FOREACH_THREAD(p2, y, Q1D)
            {
               real_t bj = 0.0;
               real_t bjn = 0.0;
               real_t gj = 0.0;
               real_t br = 0.0;

               for (int p1 = 0; p1 < Q1D; ++p1)
               {
                  const real_t b = B(p1, d1);
                  const real_t g = G(p1, d1);

                  const real_t *Je = nJe[side][p2][p1];

                  const real_t jump = u0[p2][p1] - u1[p2][p1];
                  const real_t avg = du0[p2][p1] + du1[p2][p1];

                  // r = - < {Q du/dn}, [v] > + kappa * < {Q/h} [u], [v] >
                  const real_t r = -avg + kappa_Qh[p2][p1] * jump;

                  // bj, gj, bjn contribute to sigma term
                  bj += b * Je[0] * jump;
                  gj += g * Je[1] * jump;
                  bjn += b * Je[2] * jump;

                  br += b * r;
               }

               Bj[d1][p2] = sigma * bj;
               Bjn[d1][p2] = sigma * bjn;

               // group br and gj together since we will multiply them both by B
               // and then sum
               const real_t sgn = (side == 0) ? 1.0 : -1.0;
               Gj[d1][p2] = sgn * br + sigma * gj;
            }
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(side, z, 2)
      {
         real_t(*u)[max_Q1D] = (side == 0) ? u0 : u1;
         real_t(*du)[max_Q1D] = (side == 0) ? du0 : du1;
         real_t(*Bj)[max_Q1D] = (side == 0) ? Bj0 : Bj1;
         real_t(*Bjn)[max_Q1D] = (side == 0) ? Bjn0 : Bjn1;
         real_t(*Gj)[max_Q1D] = (side == 0) ? Gj0 : Gj1;

         MFEM_FOREACH_THREAD(d2, x, D1D)
         {
            MFEM_FOREACH_THREAD(d1, y, D1D)
            {
               real_t bbj = 0.0;
               real_t gbj = 0.0;
               real_t bgj = 0.0;

               for (int p2 = 0; p2 < Q1D; ++p2)
               {
                  const real_t b = B(p2, d2);
                  const real_t g = G(p2, d2);

                  bbj += b * Bj[d1][p2];
                  bgj += b * Gj[d1][p2];
                  gbj += g * Bjn[d1][p2];
               }

               du[d2][d1] = bbj;
               u[d2][d1] = bgj + gbj;
            }
         }
      }
      MFEM_SYNC_THREAD;

      // map back to y and dydn
      MFEM_FOREACH_THREAD(side, z, 2)
      {
         const real_t(*u)[max_Q1D] = (side == 0) ? u0 : u1;
         const real_t(*du)[max_Q1D] = (side == 0) ? du0 : du1;

         MFEM_FOREACH_THREAD(d2, x, D1D)
         {
            MFEM_FOREACH_THREAD(d1, y, D1D)
            {
               y(d1, d2, side, f) += u[d2][d1];
               dydn(d1, d2, side, f) += du[d2][d1];
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
