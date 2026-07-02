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

#include "../bilininteg.hpp"
#include "../gridfunc.hpp"
#include "../qfunction.hpp"
#include "bilininteg_hcurl_kernels.hpp"
#include "bilininteg_hcurlhdiv_kernels.hpp"

namespace mfem
{

namespace
{

void PAHcurlDotSetup2D(const int q1d,
                       const int ne,
                       const bool test_map_integral,
                       const Array<real_t> &w,
                       const Vector &jacobians,
                       const Vector &coeff,
                       Vector &op)
{
   auto W = Reshape(w.Read(), q1d, q1d);
   auto J = Reshape(jacobians.Read(), q1d, q1d, 2, 2, ne);
   auto C = Reshape(coeff.Read(), 2, q1d, q1d, ne);
   auto O = Reshape(op.Write(), 2, q1d, q1d, ne);

   mfem::forall_2D(ne, q1d, q1d, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(qy, y, q1d)
      {
         MFEM_FOREACH_THREAD(qx, x, q1d)
         {
            const real_t J11 = J(qx, qy, 0, 0, e);
            const real_t J12 = J(qx, qy, 1, 0, e);
            const real_t J21 = J(qx, qy, 0, 1, e);
            const real_t J22 = J(qx, qy, 1, 1, e);
            const real_t detJ = (J11 * J22) - (J21 * J12);
            const real_t scale = W(qx, qy) * (test_map_integral ? 1.0 / detJ : 1.0);
            const real_t Vx = C(0, qx, qy, e);
            const real_t Vy = C(1, qx, qy, e);

            O(0, qx, qy, e) = scale * ( J22 * Vx - J12 * Vy);
            O(1, qx, qy, e) = scale * (-J21 * Vx + J11 * Vy);
         }
      }
   });
}

void PAHcurlDotSetup3D(const int q1d,
                       const int ne,
                       const bool test_map_integral,
                       const Array<real_t> &w,
                       const Vector &jacobians,
                       const Vector &coeff,
                       Vector &op)
{
   auto W = Reshape(w.Read(), q1d, q1d, q1d);
   auto J = Reshape(jacobians.Read(), q1d, q1d, q1d, 3, 3, ne);
   auto C = Reshape(coeff.Read(), 3, q1d, q1d, q1d, ne);
   auto O = Reshape(op.Write(), 3, q1d, q1d, q1d, ne);

   mfem::forall_3D(ne, q1d, q1d, q1d, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(qz, z, q1d)
      {
         MFEM_FOREACH_THREAD(qy, y, q1d)
         {
            MFEM_FOREACH_THREAD(qx, x, q1d)
            {
               const real_t J11 = J(qx, qy, qz, 0, 0, e);
               const real_t J12 = J(qx, qy, qz, 0, 1, e);
               const real_t J13 = J(qx, qy, qz, 0, 2, e);
               const real_t J21 = J(qx, qy, qz, 1, 0, e);
               const real_t J22 = J(qx, qy, qz, 1, 1, e);
               const real_t J23 = J(qx, qy, qz, 1, 2, e);
               const real_t J31 = J(qx, qy, qz, 2, 0, e);
               const real_t J32 = J(qx, qy, qz, 2, 1, e);
               const real_t J33 = J(qx, qy, qz, 2, 2, e);
               const real_t detJ = J11 * (J22 * J33 - J32 * J23)
                                   - J21 * (J12 * J33 - J32 * J13)
                                   + J31 * (J12 * J23 - J22 * J13);
               const real_t scale = W(qx, qy, qz) *
                                    (test_map_integral ? 1.0 / detJ : 1.0);
               const real_t Vx = C(0, qx, qy, qz, e);
               const real_t Vy = C(1, qx, qy, qz, e);
               const real_t Vz = C(2, qx, qy, qz, e);

               O(0, qx, qy, qz, e) = scale *
                                     ((J22 * J33 - J23 * J32) * Vx +
                                      (J13 * J32 - J12 * J33) * Vy +
                                      (J12 * J23 - J13 * J22) * Vz);
               O(1, qx, qy, qz, e) = scale *
                                     ((J23 * J31 - J21 * J33) * Vx +
                                      (J11 * J33 - J13 * J31) * Vy +
                                      (J13 * J21 - J11 * J23) * Vz);
               O(2, qx, qy, qz, e) = scale *
                                     ((J21 * J32 - J22 * J31) * Vx +
                                      (J12 * J31 - J11 * J32) * Vy +
                                      (J11 * J22 - J12 * J21) * Vz);
            }
         }
      }
   });
}

void PAHcurlDotApply2D(const int d1d,
                       const int d1d_test,
                       const int q1d,
                       const int ne,
                       const Array<real_t> &bo,
                       const Array<real_t> &bc,
                       const Array<real_t> &bt,
                       const Vector &pa_data,
                       const Vector &x,
                       Vector &y)
{
   MFEM_VERIFY(d1d <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D, "");
   MFEM_VERIFY(d1d_test <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q1d <= DeviceDofQuadLimits::Get().HCURL_MAX_Q1D, "");

   auto Bo = Reshape(bo.Read(), q1d, d1d - 1);
   auto Bc = Reshape(bc.Read(), q1d, d1d);
   auto Bt = Reshape(bt.Read(), d1d_test, q1d);
   auto O = Reshape(pa_data.Read(), 2, q1d, q1d, ne);
   auto X = Reshape(x.Read(), 2 * (d1d - 1) * d1d, ne);
   auto Y = Reshape(y.ReadWrite(), d1d_test, d1d_test, ne);

   mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int MAX_D1D = DofQuadLimits::MAX_D1D;
      constexpr int MAX_Q1D = DofQuadLimits::HCURL_MAX_Q1D;

      real_t u0[MAX_Q1D][MAX_Q1D];
      real_t u1[MAX_Q1D][MAX_Q1D];

      for (int qy = 0; qy < q1d; ++qy)
      {
         for (int qx = 0; qx < q1d; ++qx)
         {
            u0[qy][qx] = 0.0;
            u1[qy][qx] = 0.0;
         }
      }

      int osc = 0;
      for (int dy = 0; dy < d1d; ++dy)
      {
         real_t mass_x[MAX_Q1D];
         for (int qx = 0; qx < q1d; ++qx) { mass_x[qx] = 0.0; }
         for (int dx = 0; dx < d1d - 1; ++dx)
         {
            const real_t t = X(dx + (dy * (d1d - 1)) + osc, e);
            for (int qx = 0; qx < q1d; ++qx) { mass_x[qx] += t * Bo(qx, dx); }
         }
         for (int qy = 0; qy < q1d; ++qy)
         {
            const real_t wy = Bc(qy, dy);
            for (int qx = 0; qx < q1d; ++qx) { u0[qy][qx] += mass_x[qx] * wy; }
         }
      }

      osc += (d1d - 1) * d1d;
      for (int dy = 0; dy < d1d - 1; ++dy)
      {
         real_t mass_x[MAX_Q1D];
         for (int qx = 0; qx < q1d; ++qx) { mass_x[qx] = 0.0; }
         for (int dx = 0; dx < d1d; ++dx)
         {
            const real_t t = X(dx + (dy * d1d) + osc, e);
            for (int qx = 0; qx < q1d; ++qx) { mass_x[qx] += t * Bc(qx, dx); }
         }
         for (int qy = 0; qy < q1d; ++qy)
         {
            const real_t wy = Bo(qy, dy);
            for (int qx = 0; qx < q1d; ++qx) { u1[qy][qx] += mass_x[qx] * wy; }
         }
      }

      for (int qy = 0; qy < q1d; ++qy)
      {
         real_t sol_x[MAX_D1D];
         for (int dx = 0; dx < d1d_test; ++dx) { sol_x[dx] = 0.0; }
         for (int qx = 0; qx < q1d; ++qx)
         {
            const real_t s = O(0, qx, qy, e) * u0[qy][qx]
                             + O(1, qx, qy, e) * u1[qy][qx];
            for (int dx = 0; dx < d1d_test; ++dx)
            {
               sol_x[dx] += s * Bt(dx, qx);
            }
         }
         for (int dy = 0; dy < d1d_test; ++dy)
         {
            const real_t wy = Bt(dy, qy);
            for (int dx = 0; dx < d1d_test; ++dx)
            {
               Y(dx, dy, e) += sol_x[dx] * wy;
            }
         }
      }
   });
}

void PAHcurlDotApplyTranspose2D(const int d1d,
                                const int d1d_test,
                                const int q1d,
                                const int ne,
                                const Array<real_t> &bo,
                                const Array<real_t> &bc,
                                const Array<real_t> &b,
                                const Vector &pa_data,
                                const Vector &x,
                                Vector &y)
{
   MFEM_VERIFY(d1d <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D, "");
   MFEM_VERIFY(d1d_test <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q1d <= DeviceDofQuadLimits::Get().HCURL_MAX_Q1D, "");

   auto Bo = Reshape(bo.Read(), q1d, d1d - 1);
   auto Bc = Reshape(bc.Read(), q1d, d1d);
   auto B = Reshape(b.Read(), q1d, d1d_test);
   auto O = Reshape(pa_data.Read(), 2, q1d, q1d, ne);
   auto X = Reshape(x.Read(), d1d_test, d1d_test, ne);
   auto Y = Reshape(y.ReadWrite(), 2 * (d1d - 1) * d1d, ne);

   mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int MAX_D1D = DofQuadLimits::MAX_D1D;
      constexpr int MAX_Q1D = DofQuadLimits::HCURL_MAX_Q1D;

      real_t mass[MAX_Q1D][MAX_Q1D];
      for (int qy = 0; qy < q1d; ++qy)
      {
         for (int qx = 0; qx < q1d; ++qx)
         {
            mass[qy][qx] = 0.0;
         }
      }

      for (int dy = 0; dy < d1d_test; ++dy)
      {
         real_t sol_x[MAX_Q1D];
         for (int qx = 0; qx < q1d; ++qx) { sol_x[qx] = 0.0; }
         for (int dx = 0; dx < d1d_test; ++dx)
         {
            const real_t t = X(dx, dy, e);
            for (int qx = 0; qx < q1d; ++qx) { sol_x[qx] += t * B(qx, dx); }
         }
         for (int qy = 0; qy < q1d; ++qy)
         {
            const real_t wy = B(qy, dy);
            for (int qx = 0; qx < q1d; ++qx) { mass[qy][qx] += sol_x[qx] * wy; }
         }
      }

      int osc = 0;
      for (int qy = 0; qy < q1d; ++qy)
      {
         real_t mass_x[MAX_D1D];
         for (int dx = 0; dx < d1d - 1; ++dx) { mass_x[dx] = 0.0; }
         for (int qx = 0; qx < q1d; ++qx)
         {
            const real_t s = O(0, qx, qy, e) * mass[qy][qx];
            for (int dx = 0; dx < d1d - 1; ++dx) { mass_x[dx] += s * Bo(qx, dx); }
         }
         for (int dy = 0; dy < d1d; ++dy)
         {
            const real_t wy = Bc(qy, dy);
            for (int dx = 0; dx < d1d - 1; ++dx)
            {
               Y(dx + (dy * (d1d - 1)) + osc, e) += mass_x[dx] * wy;
            }
         }
      }

      osc += (d1d - 1) * d1d;
      for (int qy = 0; qy < q1d; ++qy)
      {
         real_t mass_x[MAX_D1D];
         for (int dx = 0; dx < d1d; ++dx) { mass_x[dx] = 0.0; }
         for (int qx = 0; qx < q1d; ++qx)
         {
            const real_t s = O(1, qx, qy, e) * mass[qy][qx];
            for (int dx = 0; dx < d1d; ++dx) { mass_x[dx] += s * Bc(qx, dx); }
         }
         for (int dy = 0; dy < d1d - 1; ++dy)
         {
            const real_t wy = Bo(qy, dy);
            for (int dx = 0; dx < d1d; ++dx)
            {
               Y(dx + (dy * d1d) + osc, e) += mass_x[dx] * wy;
            }
         }
      }
   });
}

void PAHcurlDotApply3D(const int d1d,
                       const int d1d_test,
                       const int q1d,
                       const int ne,
                       const Array<real_t> &bo,
                       const Array<real_t> &bc,
                       const Array<real_t> &bt,
                       const Vector &pa_data,
                       const Vector &x,
                       Vector &y)
{
   MFEM_VERIFY(d1d <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D, "");
   MFEM_VERIFY(d1d_test <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q1d <= DeviceDofQuadLimits::Get().HCURL_MAX_Q1D, "");

   auto Bo = Reshape(bo.Read(), q1d, d1d - 1);
   auto Bc = Reshape(bc.Read(), q1d, d1d);
   auto Bt = Reshape(bt.Read(), d1d_test, q1d);
   auto O = Reshape(pa_data.Read(), 3, q1d, q1d, q1d, ne);
   auto X = Reshape(x.Read(), 3 * (d1d - 1) * d1d * d1d, ne);
   auto Y = Reshape(y.ReadWrite(), d1d_test, d1d_test, d1d_test, ne);

   mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int MAX_D1D = DofQuadLimits::MAX_D1D;
      constexpr int MAX_Q1D = DofQuadLimits::HCURL_MAX_Q1D;

      real_t u[MAX_Q1D][MAX_Q1D][MAX_Q1D][3];
      for (int qz = 0; qz < q1d; ++qz)
      {
         for (int qy = 0; qy < q1d; ++qy)
         {
            for (int qx = 0; qx < q1d; ++qx)
            {
               for (int c = 0; c < 3; ++c) { u[qz][qy][qx][c] = 0.0; }
            }
         }
      }

      int osc = 0;
      for (int dz = 0; dz < d1d; ++dz)
      {
         real_t mass_xy[MAX_Q1D][MAX_Q1D];
         for (int qy = 0; qy < q1d; ++qy)
         {
            for (int qx = 0; qx < q1d; ++qx) { mass_xy[qy][qx] = 0.0; }
         }

         for (int dy = 0; dy < d1d; ++dy)
         {
            real_t mass_x[MAX_Q1D];
            for (int qx = 0; qx < q1d; ++qx) { mass_x[qx] = 0.0; }
            for (int dx = 0; dx < d1d - 1; ++dx)
            {
               const real_t t = X(dx + ((dy + (dz * d1d)) * (d1d - 1)) + osc, e);
               for (int qx = 0; qx < q1d; ++qx) { mass_x[qx] += t * Bo(qx, dx); }
            }
            for (int qy = 0; qy < q1d; ++qy)
            {
               const real_t wy = Bc(qy, dy);
               for (int qx = 0; qx < q1d; ++qx) { mass_xy[qy][qx] += mass_x[qx] * wy; }
            }
         }

         for (int qz = 0; qz < q1d; ++qz)
         {
            const real_t wz = Bc(qz, dz);
            for (int qy = 0; qy < q1d; ++qy)
            {
               for (int qx = 0; qx < q1d; ++qx) { u[qz][qy][qx][0] += mass_xy[qy][qx] * wz; }
            }
         }
      }

      osc += (d1d - 1) * d1d * d1d;
      for (int dz = 0; dz < d1d; ++dz)
      {
         real_t mass_xy[MAX_Q1D][MAX_Q1D];
         for (int qy = 0; qy < q1d; ++qy)
         {
            for (int qx = 0; qx < q1d; ++qx) { mass_xy[qy][qx] = 0.0; }
         }

         for (int dy = 0; dy < d1d - 1; ++dy)
         {
            real_t mass_x[MAX_Q1D];
            for (int qx = 0; qx < q1d; ++qx) { mass_x[qx] = 0.0; }
            for (int dx = 0; dx < d1d; ++dx)
            {
               const real_t t = X(dx + ((dy + (dz * (d1d - 1))) * d1d) + osc, e);
               for (int qx = 0; qx < q1d; ++qx) { mass_x[qx] += t * Bc(qx, dx); }
            }
            for (int qy = 0; qy < q1d; ++qy)
            {
               const real_t wy = Bo(qy, dy);
               for (int qx = 0; qx < q1d; ++qx) { mass_xy[qy][qx] += mass_x[qx] * wy; }
            }
         }

         for (int qz = 0; qz < q1d; ++qz)
         {
            const real_t wz = Bc(qz, dz);
            for (int qy = 0; qy < q1d; ++qy)
            {
               for (int qx = 0; qx < q1d; ++qx) { u[qz][qy][qx][1] += mass_xy[qy][qx] * wz; }
            }
         }
      }

      osc += (d1d - 1) * d1d * d1d;
      for (int dz = 0; dz < d1d - 1; ++dz)
      {
         real_t mass_xy[MAX_Q1D][MAX_Q1D];
         for (int qy = 0; qy < q1d; ++qy)
         {
            for (int qx = 0; qx < q1d; ++qx) { mass_xy[qy][qx] = 0.0; }
         }

         for (int dy = 0; dy < d1d; ++dy)
         {
            real_t mass_x[MAX_Q1D];
            for (int qx = 0; qx < q1d; ++qx) { mass_x[qx] = 0.0; }
            for (int dx = 0; dx < d1d; ++dx)
            {
               const real_t t = X(dx + ((dy + (dz * d1d)) * d1d) + osc, e);
               for (int qx = 0; qx < q1d; ++qx) { mass_x[qx] += t * Bc(qx, dx); }
            }
            for (int qy = 0; qy < q1d; ++qy)
            {
               const real_t wy = Bc(qy, dy);
               for (int qx = 0; qx < q1d; ++qx) { mass_xy[qy][qx] += mass_x[qx] * wy; }
            }
         }

         for (int qz = 0; qz < q1d; ++qz)
         {
            const real_t wz = Bo(qz, dz);
            for (int qy = 0; qy < q1d; ++qy)
            {
               for (int qx = 0; qx < q1d; ++qx) { u[qz][qy][qx][2] += mass_xy[qy][qx] * wz; }
            }
         }
      }

      for (int qz = 0; qz < q1d; ++qz)
      {
         real_t mass_xy[MAX_D1D][MAX_D1D];
         for (int dy = 0; dy < d1d_test; ++dy)
         {
            for (int dx = 0; dx < d1d_test; ++dx) { mass_xy[dy][dx] = 0.0; }
         }

         for (int qy = 0; qy < q1d; ++qy)
         {
            real_t mass_x[MAX_D1D];
            for (int dx = 0; dx < d1d_test; ++dx) { mass_x[dx] = 0.0; }
            for (int qx = 0; qx < q1d; ++qx)
            {
               const real_t s = O(0, qx, qy, qz, e) * u[qz][qy][qx][0]
                                + O(1, qx, qy, qz, e) * u[qz][qy][qx][1]
                                + O(2, qx, qy, qz, e) * u[qz][qy][qx][2];
               for (int dx = 0; dx < d1d_test; ++dx) { mass_x[dx] += s * Bt(dx, qx); }
            }
            for (int dy = 0; dy < d1d_test; ++dy)
            {
               const real_t wy = Bt(dy, qy);
               for (int dx = 0; dx < d1d_test; ++dx) { mass_xy[dy][dx] += mass_x[dx] * wy; }
            }
         }

         for (int dz = 0; dz < d1d_test; ++dz)
         {
            const real_t wz = Bt(dz, qz);
            for (int dy = 0; dy < d1d_test; ++dy)
            {
               for (int dx = 0; dx < d1d_test; ++dx)
               {
                  Y(dx, dy, dz, e) += mass_xy[dy][dx] * wz;
               }
            }
         }
      }
   });
}

void PAHcurlDotApplyTranspose3D(const int d1d,
                                const int d1d_test,
                                const int q1d,
                                const int ne,
                                const Array<real_t> &bo,
                                const Array<real_t> &bc,
                                const Array<real_t> &b,
                                const Vector &pa_data,
                                const Vector &x,
                                Vector &y)
{
   MFEM_VERIFY(d1d <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D, "");
   MFEM_VERIFY(d1d_test <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q1d <= DeviceDofQuadLimits::Get().HCURL_MAX_Q1D, "");

   auto Bo = Reshape(bo.Read(), q1d, d1d - 1);
   auto Bc = Reshape(bc.Read(), q1d, d1d);
   auto B = Reshape(b.Read(), q1d, d1d_test);
   auto O = Reshape(pa_data.Read(), 3, q1d, q1d, q1d, ne);
   auto X = Reshape(x.Read(), d1d_test, d1d_test, d1d_test, ne);
   auto Y = Reshape(y.ReadWrite(), 3 * (d1d - 1) * d1d * d1d, ne);

   mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int MAX_D1D = DofQuadLimits::MAX_D1D;
      constexpr int MAX_Q1D = DofQuadLimits::HCURL_MAX_Q1D;

      real_t mass[MAX_Q1D][MAX_Q1D][MAX_Q1D];
      for (int qz = 0; qz < q1d; ++qz)
      {
         for (int qy = 0; qy < q1d; ++qy)
         {
            for (int qx = 0; qx < q1d; ++qx) { mass[qz][qy][qx] = 0.0; }
         }
      }

      for (int dz = 0; dz < d1d_test; ++dz)
      {
         real_t mass_xy[MAX_Q1D][MAX_Q1D];
         for (int qy = 0; qy < q1d; ++qy)
         {
            for (int qx = 0; qx < q1d; ++qx) { mass_xy[qy][qx] = 0.0; }
         }

         for (int dy = 0; dy < d1d_test; ++dy)
         {
            real_t mass_x[MAX_Q1D];
            for (int qx = 0; qx < q1d; ++qx) { mass_x[qx] = 0.0; }
            for (int dx = 0; dx < d1d_test; ++dx)
            {
               const real_t t = X(dx, dy, dz, e);
               for (int qx = 0; qx < q1d; ++qx) { mass_x[qx] += t * B(qx, dx); }
            }
            for (int qy = 0; qy < q1d; ++qy)
            {
               const real_t wy = B(qy, dy);
               for (int qx = 0; qx < q1d; ++qx) { mass_xy[qy][qx] += mass_x[qx] * wy; }
            }
         }

         for (int qz = 0; qz < q1d; ++qz)
         {
            const real_t wz = B(qz, dz);
            for (int qy = 0; qy < q1d; ++qy)
            {
               for (int qx = 0; qx < q1d; ++qx) { mass[qz][qy][qx] += mass_xy[qy][qx] * wz; }
            }
         }
      }

      int osc = 0;
      for (int qz = 0; qz < q1d; ++qz)
      {
         real_t mass_xy[MAX_D1D][MAX_D1D];
         for (int dy = 0; dy < d1d; ++dy)
         {
            for (int dx = 0; dx < d1d - 1; ++dx) { mass_xy[dy][dx] = 0.0; }
         }

         for (int qy = 0; qy < q1d; ++qy)
         {
            real_t mass_x[MAX_D1D];
            for (int dx = 0; dx < d1d - 1; ++dx) { mass_x[dx] = 0.0; }
            for (int qx = 0; qx < q1d; ++qx)
            {
               const real_t s = O(0, qx, qy, qz, e) * mass[qz][qy][qx];
               for (int dx = 0; dx < d1d - 1; ++dx) { mass_x[dx] += s * Bo(qx, dx); }
            }
            for (int dy = 0; dy < d1d; ++dy)
            {
               const real_t wy = Bc(qy, dy);
               for (int dx = 0; dx < d1d - 1; ++dx) { mass_xy[dy][dx] += mass_x[dx] * wy; }
            }
         }

         for (int dz = 0; dz < d1d; ++dz)
         {
            const real_t wz = Bc(qz, dz);
            for (int dy = 0; dy < d1d; ++dy)
            {
               for (int dx = 0; dx < d1d - 1; ++dx)
               {
                  Y(dx + ((dy + (dz * d1d)) * (d1d - 1)) + osc, e) += mass_xy[dy][dx] * wz;
               }
            }
         }
      }

      osc += (d1d - 1) * d1d * d1d;
      for (int qz = 0; qz < q1d; ++qz)
      {
         real_t mass_xy[MAX_D1D][MAX_D1D];
         for (int dy = 0; dy < d1d - 1; ++dy)
         {
            for (int dx = 0; dx < d1d; ++dx) { mass_xy[dy][dx] = 0.0; }
         }

         for (int qy = 0; qy < q1d; ++qy)
         {
            real_t mass_x[MAX_D1D];
            for (int dx = 0; dx < d1d; ++dx) { mass_x[dx] = 0.0; }
            for (int qx = 0; qx < q1d; ++qx)
            {
               const real_t s = O(1, qx, qy, qz, e) * mass[qz][qy][qx];
               for (int dx = 0; dx < d1d; ++dx) { mass_x[dx] += s * Bc(qx, dx); }
            }
            for (int dy = 0; dy < d1d - 1; ++dy)
            {
               const real_t wy = Bo(qy, dy);
               for (int dx = 0; dx < d1d; ++dx) { mass_xy[dy][dx] += mass_x[dx] * wy; }
            }
         }

         for (int dz = 0; dz < d1d; ++dz)
         {
            const real_t wz = Bc(qz, dz);
            for (int dy = 0; dy < d1d - 1; ++dy)
            {
               for (int dx = 0; dx < d1d; ++dx)
               {
                  Y(dx + ((dy + (dz * (d1d - 1))) * d1d) + osc, e) += mass_xy[dy][dx] * wz;
               }
            }
         }
      }

      osc += (d1d - 1) * d1d * d1d;
      for (int qz = 0; qz < q1d; ++qz)
      {
         real_t mass_xy[MAX_D1D][MAX_D1D];
         for (int dy = 0; dy < d1d; ++dy)
         {
            for (int dx = 0; dx < d1d; ++dx) { mass_xy[dy][dx] = 0.0; }
         }

         for (int qy = 0; qy < q1d; ++qy)
         {
            real_t mass_x[MAX_D1D];
            for (int dx = 0; dx < d1d; ++dx) { mass_x[dx] = 0.0; }
            for (int qx = 0; qx < q1d; ++qx)
            {
               const real_t s = O(2, qx, qy, qz, e) * mass[qz][qy][qx];
               for (int dx = 0; dx < d1d; ++dx) { mass_x[dx] += s * Bc(qx, dx); }
            }
            for (int dy = 0; dy < d1d; ++dy)
            {
               const real_t wy = Bc(qy, dy);
               for (int dx = 0; dx < d1d; ++dx) { mass_xy[dy][dx] += mass_x[dx] * wy; }
            }
         }

         for (int dz = 0; dz < d1d - 1; ++dz)
         {
            const real_t wz = Bo(qz, dz);
            for (int dy = 0; dy < d1d; ++dy)
            {
               for (int dx = 0; dx < d1d; ++dx)
               {
                  Y(dx + ((dy + (dz * d1d)) * d1d) + osc, e) += mass_xy[dy][dx] * wz;
               }
            }
         }
      }
   });
}

} // namespace

void MixedDotProductIntegrator::AssemblePA(const FiniteElementSpace &trial_fes,
                                           const FiniteElementSpace &test_fes)
{
   Mesh *mesh = trial_fes.GetMesh();
   const FiniteElement *trial_fel = trial_fes.GetTypicalFE();
   const FiniteElement *test_fel = test_fes.GetTypicalFE();

   const VectorTensorFiniteElement *trial_el =
      dynamic_cast<const VectorTensorFiniteElement*>(trial_fel);
   MFEM_VERIFY(trial_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const TensorBasisElement *test_tensor_el =
      dynamic_cast<const TensorBasisElement*>(test_fel);
   MFEM_VERIFY(test_tensor_el != NULL,
               "Only tensor-product scalar test elements are supported!");

   MFEM_VERIFY(trial_el->GetDerivType() == mfem::FiniteElement::CURL,
               "Only H(curl) trial spaces are supported!");

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      const int order = trial_fel->GetOrder() + test_fel->GetOrder()
                        + mesh->GetTypicalElementTransformation()->OrderW();
      ir = &IntRules.Get(trial_fel->GetGeomType(), order);
   }

   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "Unsupported dimension!");
   MFEM_VERIFY(trial_el->GetDim() == dim && test_fel->GetDim() == dim,
               "Trial/test dimension mismatch.");

   ne = trial_fes.GetNE();
   MFEM_VERIFY(ne == test_fes.GetNE(),
               "Different meshes for test and trial spaces");

   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   mapsC = &trial_el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &trial_el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   mapsTest = &test_fel->GetDofToQuad(*ir, DofToQuad::TENSOR);

   dofs1D = mapsC->ndof;
   dofs1Dtest = mapsTest->ndof;
   quad1D = mapsC->nqpt;
   test_map_integral = (test_fel->GetMapType() == FiniteElement::INTEGRAL);

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");
   MFEM_VERIFY(quad1D == mapsTest->nqpt, "Trial/test quadrature mismatch");
   MFEM_VERIFY(dofs1D <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D, "");
   MFEM_VERIFY(dofs1Dtest <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(quad1D <= DeviceDofQuadLimits::Get().HCURL_MAX_Q1D, "");

   const int nq = ir->GetNPoints();
   if (dim == 2) { MFEM_VERIFY(nq == quad1D * quad1D, ""); }
   else { MFEM_VERIFY(nq == quad1D * quad1D * quad1D, ""); }

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(*VQ, qs, CoefficientStorage::FULL);
   MFEM_VERIFY(coeff.GetVDim() == dim, "Vector coefficient dimension mismatch.");

   pa_data.SetSize(dim * nq * ne, Device::GetMemoryType());

   if (dim == 2)
   {
      PAHcurlDotSetup2D(quad1D, ne, test_map_integral, ir->GetWeights(),
                        geom->J, coeff, pa_data);
   }
   else
   {
      PAHcurlDotSetup3D(quad1D, ne, test_map_integral, ir->GetWeights(),
                        geom->J, coeff, pa_data);
   }
}

void MixedDotProductIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (dim == 2)
   {
      PAHcurlDotApply2D(dofs1D, dofs1Dtest, quad1D, ne,
                        mapsO->B, mapsC->B, mapsTest->Bt, pa_data, x, y);
   }
   else if (dim == 3)
   {
      PAHcurlDotApply3D(dofs1D, dofs1Dtest, quad1D, ne,
                        mapsO->B, mapsC->B, mapsTest->Bt, pa_data, x, y);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension!");
   }
}

void MixedDotProductIntegrator::AddMultTransposePA(const Vector &x,
                                                   Vector &y) const
{
   if (dim == 2)
   {
      PAHcurlDotApplyTranspose2D(dofs1D, dofs1Dtest, quad1D, ne,
                                 mapsO->B, mapsC->B, mapsTest->B,
                                 pa_data, x, y);
   }
   else if (dim == 3)
   {
      PAHcurlDotApplyTranspose3D(dofs1D, dofs1Dtest, quad1D, ne,
                                 mapsO->B, mapsC->B, mapsTest->B,
                                 pa_data, x, y);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension!");
   }
}

void MixedScalarCurlIntegrator::AssemblePA(const FiniteElementSpace &trial_fes,
                                           const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements
   Mesh *mesh = trial_fes.GetMesh();
   const FiniteElement *fel = trial_fes.GetTypicalFE(); // In H(curl)
   const FiniteElement *eltest = test_fes.GetTypicalFE(); // In scalar space

   const VectorTensorFiniteElement *el =
      dynamic_cast<const VectorTensorFiniteElement*>(fel);
   MFEM_VERIFY(el != NULL, "Only VectorTensorFiniteElement is supported!");

   if (el->GetDerivType() != mfem::FiniteElement::CURL)
   {
      MFEM_ABORT("Unknown kernel.");
   }

   const IntegrationRule *ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(*eltest, *eltest,
                                                     *mesh->GetTypicalElementTransformation());

   const int dims = el->GetDim();
   MFEM_VERIFY(dims == 2, "");

   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2, "");

   ne = test_fes.GetNE();
   mapsC = &el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

   if (el->GetOrder() == eltest->GetOrder())
   {
      dofs1Dtest = dofs1D;
   }
   else
   {
      dofs1Dtest = dofs1D - 1;
   }

   pa_data.SetSize(nq * ne, Device::GetMemoryType());

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(Q, qs, CoefficientStorage::FULL);

   if (dim == 2)
   {
      internal::PAHcurlL2Setup2D(quad1D, ne, ir->GetWeights(), coeff, pa_data);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension!");
   }
}

void MixedScalarCurlIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (dim == 2)
   {
      internal::PAHcurlL2Apply2D(dofs1D, dofs1Dtest, quad1D, ne, mapsO->B,
                                 mapsO->Bt, mapsC->Bt, mapsC->G, pa_data,
                                 x, y);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension!");
   }
}

void MixedScalarCurlIntegrator::AddMultTransposePA(const Vector &x,
                                                   Vector &y) const
{
   if (dim == 2)
   {
      internal::PAHcurlL2ApplyTranspose2D(dofs1D, dofs1Dtest, quad1D, ne, mapsO->B,
                                          mapsO->Bt, mapsC->B, mapsC->Gt, pa_data,
                                          x, y);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension!");
   }
}

void MixedVectorCurlIntegrator::AssemblePA(const FiniteElementSpace &trial_fes,
                                           const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements, with vector test and trial spaces.
   Mesh *mesh = trial_fes.GetMesh();
   const FiniteElement *trial_fel = trial_fes.GetTypicalFE();
   const FiniteElement *test_fel = test_fes.GetTypicalFE();

   const VectorTensorFiniteElement *trial_el =
      dynamic_cast<const VectorTensorFiniteElement*>(trial_fel);
   MFEM_VERIFY(trial_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const VectorTensorFiniteElement *test_el =
      dynamic_cast<const VectorTensorFiniteElement*>(test_fel);
   MFEM_VERIFY(test_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const IntegrationRule *ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(*trial_el, *trial_el,
                                                     *mesh->GetTypicalElementTransformation());
   const int dims = trial_el->GetDim();
   MFEM_VERIFY(dims == 3, "");

   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 3, "");

   MFEM_VERIFY(trial_el->GetOrder() == test_el->GetOrder(), "");

   ne = trial_fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   mapsC = &trial_el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &trial_el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   mapsCtest = &test_el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsOtest = &test_el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;
   dofs1Dtest = mapsCtest->ndof;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

   testType = test_el->GetDerivType();
   trialType = trial_el->GetDerivType();

   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   coeffDim = (DQ ? 3 : 1);

   const bool curlSpaces = (testType == mfem::FiniteElement::CURL &&
                            trialType == mfem::FiniteElement::CURL);

   const int ndata = curlSpaces ? (coeffDim == 1 ? 1 : 9) : symmDims;
   pa_data.SetSize(ndata * nq * ne, Device::GetMemoryType());

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(qs, CoefficientStorage::FULL);
   if (Q) { coeff.Project(*Q); }
   else if (DQ) { coeff.Project(*DQ); }
   else { coeff.SetConstant(1.0); }

   if (testType == mfem::FiniteElement::CURL &&
       trialType == mfem::FiniteElement::CURL && dim == 3)
   {
      if (coeffDim == 1)
      {
         internal::PAHcurlL2Setup3D(nq, coeffDim, ne, ir->GetWeights(), coeff, pa_data);
      }
      else
      {
         internal::PAHcurlHdivMassSetup3D(quad1D, coeffDim, ne, false, ir->GetWeights(),
                                          geom->J, coeff, pa_data);
      }
   }
   else if (testType == mfem::FiniteElement::DIV &&
            trialType == mfem::FiniteElement::CURL && dim == 3 &&
            test_fel->GetOrder() == trial_fel->GetOrder())
   {
      internal::PACurlCurlSetup3D(quad1D, coeffDim, ne, ir->GetWeights(), geom->J,
                                  coeff, pa_data);
   }
   else
   {
      MFEM_ABORT("Unknown kernel.");
   }
}

void MixedVectorCurlIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (testType == mfem::FiniteElement::CURL &&
       trialType == mfem::FiniteElement::CURL && dim == 3)
   {
      const int ndata = coeffDim == 1 ? 1 : 9;

      if (Device::Allows(Backend::DEVICE_MASK))
      {
         const int ID = (dofs1D << 4) | quad1D;
         switch (ID)
         {
            case 0x23:
               return internal::SmemPAHcurlL2Apply3D<2,3>(
                         dofs1D, quad1D, ndata, ne,
                         mapsO->B, mapsC->B, mapsC->G,
                         pa_data, x, y);
            case 0x34:
               return internal::SmemPAHcurlL2Apply3D<3,4>(
                         dofs1D, quad1D, ndata, ne,
                         mapsO->B, mapsC->B, mapsC->G,
                         pa_data, x, y);
            case 0x45:
               return internal::SmemPAHcurlL2Apply3D<4,5>(
                         dofs1D, quad1D, ndata, ne,
                         mapsO->B, mapsC->B, mapsC->G,
                         pa_data, x, y);
            case 0x56:
               return internal::SmemPAHcurlL2Apply3D<5,6>(
                         dofs1D, quad1D, ndata, ne,
                         mapsO->B, mapsC->B, mapsC->G,
                         pa_data, x, y);
            default:
               return internal::SmemPAHcurlL2Apply3D(
                         dofs1D, quad1D, ndata, ne,
                         mapsO->B, mapsC->B, mapsC->G,
                         pa_data, x, y);
         }
      }
      else
      {
         internal::PAHcurlL2Apply3D(dofs1D, quad1D, ndata, ne, mapsO->B, mapsC->B,
                                    mapsO->Bt, mapsC->Bt, mapsC->G, pa_data, x, y);
      }
   }
   else if (testType == mfem::FiniteElement::DIV &&
            trialType == mfem::FiniteElement::CURL && dim == 3)
   {
      internal::PAHcurlHdivApply3D(dofs1D, dofs1Dtest, quad1D, ne, mapsO->B,
                                   mapsC->B, mapsOtest->Bt, mapsCtest->Bt, mapsC->G,
                                   pa_data, x, y);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension or space!");
   }
}

void MixedVectorCurlIntegrator::AddMultTransposePA(const Vector &x,
                                                   Vector &y) const
{
   if (testType == mfem::FiniteElement::DIV &&
       trialType == mfem::FiniteElement::CURL && dim == 3)
   {
      internal::PAHcurlHdivApplyTranspose3D(dofs1D, dofs1Dtest, quad1D, ne, mapsO->B,
                                            mapsC->B, mapsOtest->Bt, mapsCtest->Bt,
                                            mapsC->Gt, pa_data, x, y);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension or space!");
   }
}

void MixedVectorWeakCurlIntegrator::AssemblePA(const FiniteElementSpace
                                               &trial_fes,
                                               const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements, with vector test and trial spaces.
   Mesh *mesh = trial_fes.GetMesh();
   const FiniteElement *trial_fel = trial_fes.GetTypicalFE();
   const FiniteElement *test_fel = test_fes.GetTypicalFE();

   const VectorTensorFiniteElement *trial_el =
      dynamic_cast<const VectorTensorFiniteElement*>(trial_fel);
   MFEM_VERIFY(trial_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const VectorTensorFiniteElement *test_el =
      dynamic_cast<const VectorTensorFiniteElement*>(test_fel);
   MFEM_VERIFY(test_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const IntegrationRule *ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(*trial_el, *trial_el,
                                                     *mesh->GetTypicalElementTransformation());
   const int dims = trial_el->GetDim();
   MFEM_VERIFY(dims == 3, "");

   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 3, "");

   MFEM_VERIFY(trial_el->GetOrder() == test_el->GetOrder(), "");

   ne = trial_fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   mapsC = &test_el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &test_el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

   testType = test_el->GetDerivType();
   trialType = trial_el->GetDerivType();

   const bool curlSpaces = (testType == mfem::FiniteElement::CURL &&
                            trialType == mfem::FiniteElement::CURL);

   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6

   coeffDim = DQ ? 3 : 1;
   const int ndata = curlSpaces ? (DQ ? 9 : 1) : symmDims;

   pa_data.SetSize(ndata * nq * ne, Device::GetMemoryType());

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(qs, CoefficientStorage::FULL);
   if (Q) { coeff.Project(*Q); }
   else if (DQ) { coeff.Project(*DQ); }
   else if (MQ) { MFEM_ABORT("Not implemented."); }
   else { coeff.SetConstant(1.0); }

   if (trialType == mfem::FiniteElement::CURL && dim == 3)
   {
      if (coeffDim == 1)
      {
         internal::PAHcurlL2Setup3D(nq, coeffDim, ne, ir->GetWeights(), coeff, pa_data);
      }
      else
      {
         internal::PAHcurlHdivMassSetup3D(quad1D, coeffDim, ne, false, ir->GetWeights(),
                                          geom->J, coeff, pa_data);
      }
   }
   else if (trialType == mfem::FiniteElement::DIV && dim == 3 &&
            test_el->GetOrder() == trial_el->GetOrder())
   {
      internal::PACurlCurlSetup3D(quad1D, coeffDim, ne, ir->GetWeights(), geom->J,
                                  coeff, pa_data);
   }
   else
   {
      MFEM_ABORT("Unknown kernel.");
   }
}

void MixedVectorWeakCurlIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (testType == mfem::FiniteElement::CURL &&
       trialType == mfem::FiniteElement::CURL && dim == 3)
   {
      const int ndata = coeffDim == 1 ? 1 : 9;
      if (Device::Allows(Backend::DEVICE_MASK))
      {
         const int ID = (dofs1D << 4) | quad1D;
         switch (ID)
         {
            case 0x23:
               return internal::SmemPAHcurlL2ApplyTranspose3D<2,3>(
                         dofs1D, quad1D, ndata,
                         ne, mapsO->B, mapsC->B,
                         mapsC->G, pa_data, x, y);
            case 0x34:
               return internal::SmemPAHcurlL2ApplyTranspose3D<3,4>(
                         dofs1D, quad1D, ndata,
                         ne, mapsO->B, mapsC->B,
                         mapsC->G, pa_data, x, y);
            case 0x45:
               return internal::SmemPAHcurlL2ApplyTranspose3D<4,5>(
                         dofs1D, quad1D, ndata,
                         ne, mapsO->B, mapsC->B,
                         mapsC->G, pa_data, x, y);
            case 0x56:
               return internal::SmemPAHcurlL2ApplyTranspose3D<5,6>(
                         dofs1D, quad1D, ndata,
                         ne, mapsO->B, mapsC->B,
                         mapsC->G, pa_data, x, y);
            default:
               return internal::SmemPAHcurlL2ApplyTranspose3D(
                         dofs1D, quad1D, ndata, ne,
                         mapsO->B, mapsC->B,
                         mapsC->G, pa_data, x, y);
         }
      }
      else
      {
         internal::PAHcurlL2ApplyTranspose3D(dofs1D, quad1D, ndata, ne, mapsO->B,
                                             mapsC->B, mapsO->Bt, mapsC->Bt, mapsC->Gt,
                                             pa_data, x, y);
      }
   }
   else if (testType == mfem::FiniteElement::CURL &&
            trialType == mfem::FiniteElement::DIV && dim == 3)
   {
      internal::PAHcurlHdivApplyTranspose3D(dofs1D, dofs1D, quad1D, ne, mapsO->B,
                                            mapsC->B, mapsO->Bt, mapsC->Bt,
                                            mapsC->Gt, pa_data, x, y);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension or space!");
   }
}

void MixedVectorWeakCurlIntegrator::AddMultTransposePA(const Vector &x,
                                                       Vector &y) const
{
   if (testType == mfem::FiniteElement::CURL &&
       trialType == mfem::FiniteElement::DIV && dim == 3)
   {
      internal::PAHcurlHdivApply3D(dofs1D, dofs1D, quad1D, ne, mapsO->B,
                                   mapsC->B, mapsO->Bt, mapsC->Bt, mapsC->G,
                                   pa_data, x, y);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension or space!");
   }
}

} // namespace mfem
