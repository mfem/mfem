// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
#include "../qfunction.hpp"

namespace mfem
{

// Apply to x corresponding to DOFs in H^1 (domain) the (topological) gradient
// to get a dof in H(curl) (range). You can think of the range as the "test" space
// and the domain as the "trial" space, but there's no integration.
static void PAHcurlApplyGradient2D(const int c_dofs1D,
                                   const int o_dofs1D,
                                   const int NE,
                                   const Array<real_t> &B_,
                                   const Array<real_t> &G_,
                                   const Vector &x_,
                                   Vector &y_)
{
   auto B = Reshape(B_.Read(), c_dofs1D, c_dofs1D);
   auto G = Reshape(G_.Read(), o_dofs1D, c_dofs1D);

   auto x = Reshape(x_.Read(), c_dofs1D, c_dofs1D, NE);
   auto y = Reshape(y_.ReadWrite(), 2 * c_dofs1D * o_dofs1D, NE);

   MFEM_VERIFY(c_dofs1D <= DeviceDofQuadLimits::Get().MAX_D1D &&
               o_dofs1D <= c_dofs1D, "");

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int MAX_D1D = DofQuadLimits::HCURL_MAX_D1D;
      real_t w[MAX_D1D][MAX_D1D];

      // horizontal part
      for (int dx = 0; dx < c_dofs1D; ++dx)
      {
         for (int ey = 0; ey < c_dofs1D; ++ey)
         {
            w[dx][ey] = 0.0;
            for (int dy = 0; dy < c_dofs1D; ++dy)
            {
               w[dx][ey] += B(ey, dy) * x(dx, dy, e);
            }
         }
      }

      for (int ey = 0; ey < c_dofs1D; ++ey)
      {
         for (int ex = 0; ex < o_dofs1D; ++ex)
         {
            real_t s = 0.0;
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               s += G(ex, dx) * w[dx][ey];
            }
            const int local_index = ey*o_dofs1D + ex;
            y(local_index, e) += s;
         }
      }

      // vertical part
      for (int dx = 0; dx < c_dofs1D; ++dx)
      {
         for (int ey = 0; ey < o_dofs1D; ++ey)
         {
            w[dx][ey] = 0.0;
            for (int dy = 0; dy < c_dofs1D; ++dy)
            {
               w[dx][ey] += G(ey, dy) * x(dx, dy, e);
            }
         }
      }

      for (int ey = 0; ey < o_dofs1D; ++ey)
      {
         for (int ex = 0; ex < c_dofs1D; ++ex)
         {
            real_t s = 0.0;
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               s += B(ex, dx) * w[dx][ey];
            }
            const int local_index = c_dofs1D * o_dofs1D + ey*c_dofs1D + ex;
            y(local_index, e) += s;
         }
      }
   });
}

// Specialization of PAHcurlApplyGradient2D to the case where B is identity
static void PAHcurlApplyGradient2DBId(const int c_dofs1D,
                                      const int o_dofs1D,
                                      const int NE,
                                      const Array<real_t> &G_,
                                      const Vector &x_,
                                      Vector &y_)
{
   auto G = Reshape(G_.Read(), o_dofs1D, c_dofs1D);

   auto x = Reshape(x_.Read(), c_dofs1D, c_dofs1D, NE);
   auto y = Reshape(y_.ReadWrite(), 2 * c_dofs1D * o_dofs1D, NE);

   MFEM_VERIFY(c_dofs1D <= DeviceDofQuadLimits::Get().MAX_D1D &&
               o_dofs1D <= c_dofs1D, "");

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int MAX_D1D = DofQuadLimits::HCURL_MAX_D1D;
      real_t w[MAX_D1D][MAX_D1D];

      // horizontal part
      for (int dx = 0; dx < c_dofs1D; ++dx)
      {
         for (int ey = 0; ey < c_dofs1D; ++ey)
         {
            const int dy = ey;
            w[dx][ey] = x(dx, dy, e);
         }
      }

      for (int ey = 0; ey < c_dofs1D; ++ey)
      {
         for (int ex = 0; ex < o_dofs1D; ++ex)
         {
            real_t s = 0.0;
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               s += G(ex, dx) * w[dx][ey];
            }
            const int local_index = ey*o_dofs1D + ex;
            y(local_index, e) += s;
         }
      }

      // vertical part
      for (int dx = 0; dx < c_dofs1D; ++dx)
      {
         for (int ey = 0; ey < o_dofs1D; ++ey)
         {
            w[dx][ey] = 0.0;
            for (int dy = 0; dy < c_dofs1D; ++dy)
            {
               w[dx][ey] += G(ey, dy) * x(dx, dy, e);
            }
         }
      }

      for (int ey = 0; ey < o_dofs1D; ++ey)
      {
         for (int ex = 0; ex < c_dofs1D; ++ex)
         {
            const int dx = ex;
            const real_t s = w[dx][ey];
            const int local_index = c_dofs1D * o_dofs1D + ey*c_dofs1D + ex;
            y(local_index, e) += s;
         }
      }
   });
}

static void PAHcurlApplyGradientTranspose2D(
   const int c_dofs1D, const int o_dofs1D, const int NE,
   const Array<real_t> &B_, const Array<real_t> &G_,
   const Vector &x_, Vector &y_)
{
   auto B = Reshape(B_.Read(), c_dofs1D, c_dofs1D);
   auto G = Reshape(G_.Read(), o_dofs1D, c_dofs1D);

   auto x = Reshape(x_.Read(), 2 * c_dofs1D * o_dofs1D, NE);
   auto y = Reshape(y_.ReadWrite(), c_dofs1D, c_dofs1D, NE);

   MFEM_VERIFY(c_dofs1D <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D &&
               o_dofs1D <= c_dofs1D, "");

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int MAX_D1D = DofQuadLimits::HCURL_MAX_D1D;
      real_t w[MAX_D1D][MAX_D1D];

      // horizontal part (open x, closed y)
      for (int dy = 0; dy < c_dofs1D; ++dy)
      {
         for (int ex = 0; ex < o_dofs1D; ++ex)
         {
            w[dy][ex] = 0.0;
            for (int ey = 0; ey < c_dofs1D; ++ey)
            {
               const int local_index = ey*o_dofs1D + ex;
               w[dy][ex] += B(ey, dy) * x(local_index, e);
            }
         }
      }

      for (int dy = 0; dy < c_dofs1D; ++dy)
      {
         for (int dx = 0; dx < c_dofs1D; ++dx)
         {
            real_t s = 0.0;
            for (int ex = 0; ex < o_dofs1D; ++ex)
            {
               s += G(ex, dx) * w[dy][ex];
            }
            y(dx, dy, e) += s;
         }
      }

      // vertical part (open y, closed x)
      for (int dy = 0; dy < c_dofs1D; ++dy)
      {
         for (int ex = 0; ex < c_dofs1D; ++ex)
         {
            w[dy][ex] = 0.0;
            for (int ey = 0; ey < o_dofs1D; ++ey)
            {
               const int local_index = c_dofs1D * o_dofs1D + ey*c_dofs1D + ex;
               w[dy][ex] += G(ey, dy) * x(local_index, e);
            }
         }
      }

      for (int dy = 0; dy < c_dofs1D; ++dy)
      {
         for (int dx = 0; dx < c_dofs1D; ++dx)
         {
            real_t s = 0.0;
            for (int ex = 0; ex < c_dofs1D; ++ex)
            {
               s += B(ex, dx) * w[dy][ex];
            }
            y(dx, dy, e) += s;
         }
      }
   });
}

// Specialization of PAHcurlApplyGradientTranspose2D to the case where
// B is identity
static void PAHcurlApplyGradientTranspose2DBId(
   const int c_dofs1D, const int o_dofs1D, const int NE,
   const Array<real_t> &G_,
   const Vector &x_, Vector &y_)
{
   auto G = Reshape(G_.Read(), o_dofs1D, c_dofs1D);

   auto x = Reshape(x_.Read(), 2 * c_dofs1D * o_dofs1D, NE);
   auto y = Reshape(y_.ReadWrite(), c_dofs1D, c_dofs1D, NE);

   MFEM_VERIFY(c_dofs1D <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D &&
               o_dofs1D <= c_dofs1D, "");

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int MAX_D1D = DofQuadLimits::HCURL_MAX_D1D;
      real_t w[MAX_D1D][MAX_D1D];

      // horizontal part (open x, closed y)
      for (int dy = 0; dy < c_dofs1D; ++dy)
      {
         for (int ex = 0; ex < o_dofs1D; ++ex)
         {
            const int ey = dy;
            const int local_index = ey*o_dofs1D + ex;
            w[dy][ex] = x(local_index, e);
         }
      }

      for (int dy = 0; dy < c_dofs1D; ++dy)
      {
         for (int dx = 0; dx < c_dofs1D; ++dx)
         {
            real_t s = 0.0;
            for (int ex = 0; ex < o_dofs1D; ++ex)
            {
               s += G(ex, dx) * w[dy][ex];
            }
            y(dx, dy, e) += s;
         }
      }

      // vertical part (open y, closed x)
      for (int dy = 0; dy < c_dofs1D; ++dy)
      {
         for (int ex = 0; ex < c_dofs1D; ++ex)
         {
            w[dy][ex] = 0.0;
            for (int ey = 0; ey < o_dofs1D; ++ey)
            {
               const int local_index = c_dofs1D * o_dofs1D + ey*c_dofs1D + ex;
               w[dy][ex] += G(ey, dy) * x(local_index, e);
            }
         }
      }

      for (int dy = 0; dy < c_dofs1D; ++dy)
      {
         for (int dx = 0; dx < c_dofs1D; ++dx)
         {
            const int ex = dx;
            const real_t s = w[dy][ex];
            y(dx, dy, e) += s;
         }
      }
   });
}

static void PAHcurlApplyGradient3D(const int c_dofs1D,
                                   const int o_dofs1D,
                                   const int NE,
                                   const Array<real_t> &B_,
                                   const Array<real_t> &G_,
                                   const Vector &x_,
                                   Vector &y_)
{
   auto B = Reshape(B_.Read(), c_dofs1D, c_dofs1D);
   auto G = Reshape(G_.Read(), o_dofs1D, c_dofs1D);

   auto x = Reshape(x_.Read(), c_dofs1D, c_dofs1D, c_dofs1D, NE);
   auto y = Reshape(y_.ReadWrite(), (3 * c_dofs1D * c_dofs1D * o_dofs1D), NE);

   MFEM_VERIFY(c_dofs1D <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D &&
               o_dofs1D <= c_dofs1D, "");

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int MAX_D1D = DofQuadLimits::HCURL_MAX_D1D;
      real_t w1[MAX_D1D][MAX_D1D][MAX_D1D];
      real_t w2[MAX_D1D][MAX_D1D][MAX_D1D];

      // ---
      // dofs that point parallel to x-axis (open in x, closed in y, z)
      // ---

      // contract in z
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int dx = 0; dx < c_dofs1D; ++dx)
         {
            for (int dy = 0; dy < c_dofs1D; ++dy)
            {
               w1[dx][dy][ez] = 0.0;
               for (int dz = 0; dz < c_dofs1D; ++dz)
               {
                  w1[dx][dy][ez] += B(ez, dz) * x(dx, dy, dz, e);
               }
            }
         }
      }

      // contract in y
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int ey = 0; ey < c_dofs1D; ++ey)
         {
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               w2[dx][ey][ez] = 0.0;
               for (int dy = 0; dy < c_dofs1D; ++dy)
               {
                  w2[dx][ey][ez] += B(ey, dy) * w1[dx][dy][ez];
               }
            }
         }
      }

      // contract in x
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int ey = 0; ey < c_dofs1D; ++ey)
         {
            for (int ex = 0; ex < o_dofs1D; ++ex)
            {
               real_t s = 0.0;
               for (int dx = 0; dx < c_dofs1D; ++dx)
               {
                  s += G(ex, dx) * w2[dx][ey][ez];
               }
               const int local_index = ez*c_dofs1D*o_dofs1D + ey*o_dofs1D + ex;
               y(local_index, e) += s;
            }
         }
      }

      // ---
      // dofs that point parallel to y-axis (open in y, closed in x, z)
      // ---

      // contract in z
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int dx = 0; dx < c_dofs1D; ++dx)
         {
            for (int dy = 0; dy < c_dofs1D; ++dy)
            {
               w1[dx][dy][ez] = 0.0;
               for (int dz = 0; dz < c_dofs1D; ++dz)
               {
                  w1[dx][dy][ez] += B(ez, dz) * x(dx, dy, dz, e);
               }
            }
         }
      }

      // contract in y
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int ey = 0; ey < o_dofs1D; ++ey)
         {
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               w2[dx][ey][ez] = 0.0;
               for (int dy = 0; dy < c_dofs1D; ++dy)
               {
                  w2[dx][ey][ez] += G(ey, dy) * w1[dx][dy][ez];
               }
            }
         }
      }

      // contract in x
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int ey = 0; ey < o_dofs1D; ++ey)
         {
            for (int ex = 0; ex < c_dofs1D; ++ex)
            {
               real_t s = 0.0;
               for (int dx = 0; dx < c_dofs1D; ++dx)
               {
                  s += B(ex, dx) * w2[dx][ey][ez];
               }
               const int local_index = c_dofs1D*c_dofs1D*o_dofs1D +
                                       ez*c_dofs1D*o_dofs1D + ey*c_dofs1D + ex;
               y(local_index, e) += s;
            }
         }
      }

      // ---
      // dofs that point parallel to z-axis (open in z, closed in x, y)
      // ---

      // contract in z
      for (int ez = 0; ez < o_dofs1D; ++ez)
      {
         for (int dx = 0; dx < c_dofs1D; ++dx)
         {
            for (int dy = 0; dy < c_dofs1D; ++dy)
            {
               w1[dx][dy][ez] = 0.0;
               for (int dz = 0; dz < c_dofs1D; ++dz)
               {
                  w1[dx][dy][ez] += G(ez, dz) * x(dx, dy, dz, e);
               }
            }
         }
      }

      // contract in y
      for (int ez = 0; ez < o_dofs1D; ++ez)
      {
         for (int ey = 0; ey < c_dofs1D; ++ey)
         {
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               w2[dx][ey][ez] = 0.0;
               for (int dy = 0; dy < c_dofs1D; ++dy)
               {
                  w2[dx][ey][ez] += B(ey, dy) * w1[dx][dy][ez];
               }
            }
         }
      }

      // contract in x
      for (int ez = 0; ez < o_dofs1D; ++ez)
      {
         for (int ey = 0; ey < c_dofs1D; ++ey)
         {
            for (int ex = 0; ex < c_dofs1D; ++ex)
            {
               real_t s = 0.0;
               for (int dx = 0; dx < c_dofs1D; ++dx)
               {
                  s += B(ex, dx) * w2[dx][ey][ez];
               }
               const int local_index = 2*c_dofs1D*c_dofs1D*o_dofs1D +
                                       ez*c_dofs1D*c_dofs1D + ey*c_dofs1D + ex;
               y(local_index, e) += s;
            }
         }
      }
   });
}

// Specialization of PAHcurlApplyGradient3D to the case where B is identity
static void PAHcurlApplyGradient3DBId(const int c_dofs1D,
                                      const int o_dofs1D,
                                      const int NE,
                                      const Array<real_t> &G_,
                                      const Vector &x_,
                                      Vector &y_)
{
   auto G = Reshape(G_.Read(), o_dofs1D, c_dofs1D);

   auto x = Reshape(x_.Read(), c_dofs1D, c_dofs1D, c_dofs1D, NE);
   auto y = Reshape(y_.ReadWrite(), (3 * c_dofs1D * c_dofs1D * o_dofs1D), NE);

   MFEM_VERIFY(c_dofs1D <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D &&
               o_dofs1D <= c_dofs1D, "");

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int MAX_D1D = DofQuadLimits::HCURL_MAX_D1D;

      real_t w1[MAX_D1D][MAX_D1D][MAX_D1D];
      real_t w2[MAX_D1D][MAX_D1D][MAX_D1D];

      // ---
      // dofs that point parallel to x-axis (open in x, closed in y, z)
      // ---

      // contract in z
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int dx = 0; dx < c_dofs1D; ++dx)
         {
            for (int dy = 0; dy < c_dofs1D; ++dy)
            {
               const int dz = ez;
               w1[dx][dy][ez] = x(dx, dy, dz, e);
            }
         }
      }

      // contract in y
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int ey = 0; ey < c_dofs1D; ++ey)
         {
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               const int dy = ey;
               w2[dx][ey][ez] = w1[dx][dy][ez];
            }
         }
      }

      // contract in x
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int ey = 0; ey < c_dofs1D; ++ey)
         {
            for (int ex = 0; ex < o_dofs1D; ++ex)
            {
               real_t s = 0.0;
               for (int dx = 0; dx < c_dofs1D; ++dx)
               {
                  s += G(ex, dx) * w2[dx][ey][ez];
               }
               const int local_index = ez*c_dofs1D*o_dofs1D + ey*o_dofs1D + ex;
               y(local_index, e) += s;
            }
         }
      }

      // ---
      // dofs that point parallel to y-axis (open in y, closed in x, z)
      // ---

      // contract in z
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int dx = 0; dx < c_dofs1D; ++dx)
         {
            for (int dy = 0; dy < c_dofs1D; ++dy)
            {
               const int dz = ez;
               w1[dx][dy][ez] = x(dx, dy, dz, e);
            }
         }
      }

      // contract in y
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int ey = 0; ey < o_dofs1D; ++ey)
         {
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               w2[dx][ey][ez] = 0.0;
               for (int dy = 0; dy < c_dofs1D; ++dy)
               {
                  w2[dx][ey][ez] += G(ey, dy) * w1[dx][dy][ez];
               }
            }
         }
      }

      // contract in x
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int ey = 0; ey < o_dofs1D; ++ey)
         {
            for (int ex = 0; ex < c_dofs1D; ++ex)
            {
               const int dx = ex;
               const real_t s = w2[dx][ey][ez];
               const int local_index = c_dofs1D*c_dofs1D*o_dofs1D +
                                       ez*c_dofs1D*o_dofs1D + ey*c_dofs1D + ex;
               y(local_index, e) += s;
            }
         }
      }

      // ---
      // dofs that point parallel to z-axis (open in z, closed in x, y)
      // ---

      // contract in z
      for (int ez = 0; ez < o_dofs1D; ++ez)
      {
         for (int dx = 0; dx < c_dofs1D; ++dx)
         {
            for (int dy = 0; dy < c_dofs1D; ++dy)
            {
               w1[dx][dy][ez] = 0.0;
               for (int dz = 0; dz < c_dofs1D; ++dz)
               {
                  w1[dx][dy][ez] += G(ez, dz) * x(dx, dy, dz, e);
               }
            }
         }
      }

      // contract in y
      for (int ez = 0; ez < o_dofs1D; ++ez)
      {
         for (int ey = 0; ey < c_dofs1D; ++ey)
         {
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               const int dy = ey;
               w2[dx][ey][ez] = w1[dx][dy][ez];
            }
         }
      }

      // contract in x
      for (int ez = 0; ez < o_dofs1D; ++ez)
      {
         for (int ey = 0; ey < c_dofs1D; ++ey)
         {
            for (int ex = 0; ex < c_dofs1D; ++ex)
            {
               const int dx = ex;
               const real_t s = w2[dx][ey][ez];
               const int local_index = 2*c_dofs1D*c_dofs1D*o_dofs1D +
                                       ez*c_dofs1D*c_dofs1D + ey*c_dofs1D + ex;
               y(local_index, e) += s;
            }
         }
      }
   });
}

static void PAHcurlApplyGradientTranspose3D(
   const int c_dofs1D, const int o_dofs1D, const int NE,
   const Array<real_t> &B_, const Array<real_t> &G_,
   const Vector &x_, Vector &y_)
{
   auto B = Reshape(B_.Read(), c_dofs1D, c_dofs1D);
   auto G = Reshape(G_.Read(), o_dofs1D, c_dofs1D);

   auto x = Reshape(x_.Read(), (3 * c_dofs1D * c_dofs1D * o_dofs1D), NE);
   auto y = Reshape(y_.ReadWrite(), c_dofs1D, c_dofs1D, c_dofs1D, NE);

   MFEM_VERIFY(c_dofs1D <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D &&
               o_dofs1D <= c_dofs1D, "");

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int MAX_D1D = DofQuadLimits::HCURL_MAX_D1D;
      real_t w1[MAX_D1D][MAX_D1D][MAX_D1D];
      real_t w2[MAX_D1D][MAX_D1D][MAX_D1D];
      // ---
      // dofs that point parallel to x-axis (open in x, closed in y, z)
      // ---

      // contract in z
      for (int dz = 0; dz < c_dofs1D; ++dz)
      {
         for (int ex = 0; ex < o_dofs1D; ++ex)
         {
            for (int ey = 0; ey < c_dofs1D; ++ey)
            {
               w1[ex][ey][dz] = 0.0;
               for (int ez = 0; ez < c_dofs1D; ++ez)
               {
                  const int local_index = ez*c_dofs1D*o_dofs1D + ey*o_dofs1D + ex;
                  w1[ex][ey][dz] += B(ez, dz) * x(local_index, e);
               }
            }
         }
      }

      // contract in y
      for (int dz = 0; dz < c_dofs1D; ++dz)
      {
         for (int dy = 0; dy < c_dofs1D; ++dy)
         {
            for (int ex = 0; ex < o_dofs1D; ++ex)
            {
               w2[ex][dy][dz] = 0.0;
               for (int ey = 0; ey < c_dofs1D; ++ey)
               {
                  w2[ex][dy][dz] += B(ey, dy) * w1[ex][ey][dz];
               }
            }
         }
      }

      // contract in x
      for (int dz = 0; dz < c_dofs1D; ++dz)
      {
         for (int dy = 0; dy < c_dofs1D; ++dy)
         {
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               real_t s = 0.0;
               for (int ex = 0; ex < o_dofs1D; ++ex)
               {
                  s += G(ex, dx) * w2[ex][dy][dz];
               }
               y(dx, dy, dz, e) += s;
            }
         }
      }

      // ---
      // dofs that point parallel to y-axis (open in y, closed in x, z)
      // ---

      // contract in z
      for (int dz = 0; dz < c_dofs1D; ++dz)
      {
         for (int ex = 0; ex < c_dofs1D; ++ex)
         {
            for (int ey = 0; ey < o_dofs1D; ++ey)
            {
               w1[ex][ey][dz] = 0.0;
               for (int ez = 0; ez < c_dofs1D; ++ez)
               {
                  const int local_index = c_dofs1D*c_dofs1D*o_dofs1D +
                                          ez*c_dofs1D*o_dofs1D + ey*c_dofs1D + ex;
                  w1[ex][ey][dz] += B(ez, dz) * x(local_index, e);
               }
            }
         }
      }

      // contract in y
      for (int dz = 0; dz < c_dofs1D; ++dz)
      {
         for (int dy = 0; dy < c_dofs1D; ++dy)
         {
            for (int ex = 0; ex < c_dofs1D; ++ex)
            {
               w2[ex][dy][dz] = 0.0;
               for (int ey = 0; ey < o_dofs1D; ++ey)
               {
                  w2[ex][dy][dz] += G(ey, dy) * w1[ex][ey][dz];
               }
            }
         }
      }

      // contract in x
      for (int dz = 0; dz < c_dofs1D; ++dz)
      {
         for (int dy = 0; dy < c_dofs1D; ++dy)
         {
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               real_t s = 0.0;
               for (int ex = 0; ex < c_dofs1D; ++ex)
               {
                  s += B(ex, dx) * w2[ex][dy][dz];
               }
               y(dx, dy, dz, e) += s;
            }
         }
      }

      // ---
      // dofs that point parallel to z-axis (open in z, closed in x, y)
      // ---

      // contract in z
      for (int dz = 0; dz < c_dofs1D; ++dz)
      {
         for (int ex = 0; ex < c_dofs1D; ++ex)
         {
            for (int ey = 0; ey < c_dofs1D; ++ey)
            {
               w1[ex][ey][dz] = 0.0;
               for (int ez = 0; ez < o_dofs1D; ++ez)
               {
                  const int local_index = 2*c_dofs1D*c_dofs1D*o_dofs1D +
                                          ez*c_dofs1D*c_dofs1D + ey*c_dofs1D + ex;
                  w1[ex][ey][dz] += G(ez, dz) * x(local_index, e);
               }
            }
         }
      }

      // contract in y
      for (int dz = 0; dz < c_dofs1D; ++dz)
      {
         for (int dy = 0; dy < c_dofs1D; ++dy)
         {
            for (int ex = 0; ex < c_dofs1D; ++ex)
            {
               w2[ex][dy][dz] = 0.0;
               for (int ey = 0; ey < c_dofs1D; ++ey)
               {
                  w2[ex][dy][dz] += B(ey, dy) * w1[ex][ey][dz];
               }
            }
         }
      }

      // contract in x
      for (int dz = 0; dz < c_dofs1D; ++dz)
      {
         for (int dy = 0; dy < c_dofs1D; ++dy)
         {
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               real_t s = 0.0;
               for (int ex = 0; ex < c_dofs1D; ++ex)
               {
                  s += B(ex, dx) * w2[ex][dy][dz];
               }
               y(dx, dy, dz, e) += s;
            }
         }
      }
   });
}

// Specialization of PAHcurlApplyGradientTranspose3D to the case where
// B is identity
static void PAHcurlApplyGradientTranspose3DBId(
   const int c_dofs1D, const int o_dofs1D, const int NE,
   const Array<real_t> &G_,
   const Vector &x_, Vector &y_)
{
   auto G = Reshape(G_.Read(), o_dofs1D, c_dofs1D);

   auto x = Reshape(x_.Read(), (3 * c_dofs1D * c_dofs1D * o_dofs1D), NE);
   auto y = Reshape(y_.ReadWrite(), c_dofs1D, c_dofs1D, c_dofs1D, NE);

   MFEM_VERIFY(c_dofs1D <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D &&
               o_dofs1D <= c_dofs1D, "");

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int MAX_D1D = DofQuadLimits::HCURL_MAX_D1D;

      real_t w1[MAX_D1D][MAX_D1D][MAX_D1D];
      real_t w2[MAX_D1D][MAX_D1D][MAX_D1D];
      // ---
      // dofs that point parallel to x-axis (open in x, closed in y, z)
      // ---

      // contract in z
      for (int dz = 0; dz < c_dofs1D; ++dz)
      {
         for (int ex = 0; ex < o_dofs1D; ++ex)
         {
            for (int ey = 0; ey < c_dofs1D; ++ey)
            {
               const int ez = dz;
               const int local_index = ez*c_dofs1D*o_dofs1D + ey*o_dofs1D + ex;
               w1[ex][ey][dz] = x(local_index, e);
            }
         }
      }

      // contract in y
      for (int dz = 0; dz < c_dofs1D; ++dz)
      {
         for (int dy = 0; dy < c_dofs1D; ++dy)
         {
            for (int ex = 0; ex < o_dofs1D; ++ex)
            {
               const int ey = dy;
               w2[ex][dy][dz] = w1[ex][ey][dz];
            }
         }
      }

      // contract in x
      for (int dz = 0; dz < c_dofs1D; ++dz)
      {
         for (int dy = 0; dy < c_dofs1D; ++dy)
         {
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               real_t s = 0.0;
               for (int ex = 0; ex < o_dofs1D; ++ex)
               {
                  s += G(ex, dx) * w2[ex][dy][dz];
               }
               y(dx, dy, dz, e) += s;
            }
         }
      }

      // ---
      // dofs that point parallel to y-axis (open in y, closed in x, z)
      // ---

      // contract in z
      for (int dz = 0; dz < c_dofs1D; ++dz)
      {
         for (int ex = 0; ex < c_dofs1D; ++ex)
         {
            for (int ey = 0; ey < o_dofs1D; ++ey)
            {
               const int ez = dz;
               const int local_index = c_dofs1D*c_dofs1D*o_dofs1D +
                                       ez*c_dofs1D*o_dofs1D + ey*c_dofs1D + ex;
               w1[ex][ey][dz] = x(local_index, e);
            }
         }
      }

      // contract in y
      for (int dz = 0; dz < c_dofs1D; ++dz)
      {
         for (int dy = 0; dy < c_dofs1D; ++dy)
         {
            for (int ex = 0; ex < c_dofs1D; ++ex)
            {
               w2[ex][dy][dz] = 0.0;
               for (int ey = 0; ey < o_dofs1D; ++ey)
               {
                  w2[ex][dy][dz] += G(ey, dy) * w1[ex][ey][dz];
               }
            }
         }
      }

      // contract in x
      for (int dz = 0; dz < c_dofs1D; ++dz)
      {
         for (int dy = 0; dy < c_dofs1D; ++dy)
         {
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               const int ex = dx;
               real_t s = w2[ex][dy][dz];
               y(dx, dy, dz, e) += s;
            }
         }
      }

      // ---
      // dofs that point parallel to z-axis (open in z, closed in x, y)
      // ---

      // contract in z
      for (int dz = 0; dz < c_dofs1D; ++dz)
      {
         for (int ex = 0; ex < c_dofs1D; ++ex)
         {
            for (int ey = 0; ey < c_dofs1D; ++ey)
            {
               w1[ex][ey][dz] = 0.0;
               for (int ez = 0; ez < o_dofs1D; ++ez)
               {
                  const int local_index = 2*c_dofs1D*c_dofs1D*o_dofs1D +
                                          ez*c_dofs1D*c_dofs1D + ey*c_dofs1D + ex;
                  w1[ex][ey][dz] += G(ez, dz) * x(local_index, e);
               }
            }
         }
      }

      // contract in y
      for (int dz = 0; dz < c_dofs1D; ++dz)
      {
         for (int dy = 0; dy < c_dofs1D; ++dy)
         {
            for (int ex = 0; ex < c_dofs1D; ++ex)
            {
               const int ey = dy;
               w2[ex][dy][dz] = w1[ex][ey][dz];
            }
         }
      }

      // contract in x
      for (int dz = 0; dz < c_dofs1D; ++dz)
      {
         for (int dy = 0; dy < c_dofs1D; ++dy)
         {
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               const int ex = dx;
               real_t s = w2[ex][dy][dz];
               y(dx, dy, dz, e) += s;
            }
         }
      }
   });
}

void GradientInterpolator::AssemblePA(const FiniteElementSpace &trial_fes,
                                      const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements, with a vector test space and H^1 trial space.
   Mesh *mesh = trial_fes.GetMesh();
   const FiniteElement *trial_fel = trial_fes.GetTypicalFE();
   const FiniteElement *test_fel = test_fes.GetTypicalFE();

   const NodalTensorFiniteElement *trial_el =
      dynamic_cast<const NodalTensorFiniteElement*>(trial_fel);
   MFEM_VERIFY(trial_el != NULL, "Only NodalTensorFiniteElement is supported!");

   const VectorTensorFiniteElement *test_el =
      dynamic_cast<const VectorTensorFiniteElement*>(test_fel);
   MFEM_VERIFY(test_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const int dims = trial_el->GetDim();
   MFEM_VERIFY(dims == 2 || dims == 3, "Bad dimension!");
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "Bad dimension!");
   MFEM_VERIFY(trial_el->GetOrder() == test_el->GetOrder(),
               "Orders do not match!");
   ne = trial_fes.GetNE();

   const int order = trial_el->GetOrder();
   dofquad_fe = new H1_SegmentElement(order, trial_el->GetBasisType());
   mfem::QuadratureFunctions1D qf1d;
   mfem::IntegrationRule closed_ir;
   closed_ir.SetSize(order + 1);
   qf1d.GaussLobatto(order + 1, &closed_ir);
   mfem::IntegrationRule open_ir;
   open_ir.SetSize(order);
   qf1d.GaussLegendre(order, &open_ir);

   maps_O_C = &dofquad_fe->GetDofToQuad(open_ir, DofToQuad::TENSOR);
   o_dofs1D = maps_O_C->nqpt;
   if (trial_el->GetBasisType() == BasisType::GaussLobatto)
   {
      B_id = true;
      c_dofs1D = maps_O_C->ndof;
   }
   else
   {
      B_id = false;
      maps_C_C = &dofquad_fe->GetDofToQuad(closed_ir, DofToQuad::TENSOR);
      c_dofs1D = maps_C_C->nqpt;
   }
}

void GradientInterpolator::AddMultPA(const Vector &x, Vector &y) const
{
   if (dim == 3)
   {
      if (B_id)
      {
         PAHcurlApplyGradient3DBId(c_dofs1D, o_dofs1D, ne,
                                   maps_O_C->G, x, y);
      }
      else
      {
         PAHcurlApplyGradient3D(c_dofs1D, o_dofs1D, ne, maps_C_C->B,
                                maps_O_C->G, x, y);
      }
   }
   else if (dim == 2)
   {
      if (B_id)
      {
         PAHcurlApplyGradient2DBId(c_dofs1D, o_dofs1D, ne,
                                   maps_O_C->G, x, y);
      }
      else
      {
         PAHcurlApplyGradient2D(c_dofs1D, o_dofs1D, ne, maps_C_C->B, maps_O_C->G,
                                x, y);
      }
   }
   else
   {
      mfem_error("Bad dimension!");
   }
}

void GradientInterpolator::AddMultTransposePA(const Vector &x, Vector &y) const
{
   if (dim == 3)
   {
      if (B_id)
      {
         PAHcurlApplyGradientTranspose3DBId(c_dofs1D, o_dofs1D, ne,
                                            maps_O_C->G, x, y);
      }
      else
      {
         PAHcurlApplyGradientTranspose3D(c_dofs1D, o_dofs1D, ne, maps_C_C->B,
                                         maps_O_C->G, x, y);
      }
   }
   else if (dim == 2)
   {
      if (B_id)
      {
         PAHcurlApplyGradientTranspose2DBId(c_dofs1D, o_dofs1D, ne,
                                            maps_O_C->G, x, y);
      }
      else
      {
         PAHcurlApplyGradientTranspose2D(c_dofs1D, o_dofs1D, ne, maps_C_C->B,
                                         maps_O_C->G, x, y);
      }
   }
   else
   {
      mfem_error("Bad dimension!");
   }
}

static void PAHcurlVecH1IdentityApply2D(const int c_dofs1D,
                                        const int o_dofs1D,
                                        const int NE,
                                        const Array<real_t> &Bclosed,
                                        const Array<real_t> &Bopen,
                                        const Vector &pa_data,
                                        const Vector &x_,
                                        Vector &y_)
{
   auto Bc = Reshape(Bclosed.Read(), c_dofs1D, c_dofs1D);
   auto Bo = Reshape(Bopen.Read(), o_dofs1D, c_dofs1D);

   auto x = Reshape(x_.Read(), c_dofs1D, c_dofs1D, 2, NE);
   auto y = Reshape(y_.ReadWrite(), (2 * c_dofs1D * o_dofs1D), NE);

   auto vk = Reshape(pa_data.Read(), 2, (2 * c_dofs1D * o_dofs1D), NE);

   MFEM_VERIFY(c_dofs1D <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D &&
               o_dofs1D <= c_dofs1D, "");

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int MAX_D1D = DofQuadLimits::HCURL_MAX_D1D;

      real_t w[2][MAX_D1D][MAX_D1D];

      // dofs that point parallel to x-axis (open in x, closed in y)

      // contract in y
      for (int ey = 0; ey < c_dofs1D; ++ey)
      {
         for (int dx = 0; dx < c_dofs1D; ++dx)
         {
            for (int j=0; j<2; ++j)
            {
               w[j][dx][ey] = 0.0;
               for (int dy = 0; dy < c_dofs1D; ++dy)
               {
                  w[j][dx][ey] += Bc(ey, dy) * x(dx, dy, j, e);
               }
            }
         }
      }

      // contract in x
      for (int ey = 0; ey < c_dofs1D; ++ey)
      {
         for (int ex = 0; ex < o_dofs1D; ++ex)
         {
            for (int j=0; j<2; ++j)
            {
               real_t s = 0.0;
               for (int dx = 0; dx < c_dofs1D; ++dx)
               {
                  s += Bo(ex, dx) * w[j][dx][ey];
               }
               const int local_index = ey*o_dofs1D + ex;
               y(local_index, e) += s * vk(j, local_index, e);
            }
         }
      }

      // dofs that point parallel to y-axis (open in y, closed in x)

      // contract in y
      for (int ey = 0; ey < o_dofs1D; ++ey)
      {
         for (int dx = 0; dx < c_dofs1D; ++dx)
         {
            for (int j=0; j<2; ++j)
            {
               w[j][dx][ey] = 0.0;
               for (int dy = 0; dy < c_dofs1D; ++dy)
               {
                  w[j][dx][ey] += Bo(ey, dy) * x(dx, dy, j, e);
               }
            }
         }
      }

      // contract in x
      for (int ey = 0; ey < o_dofs1D; ++ey)
      {
         for (int ex = 0; ex < c_dofs1D; ++ex)
         {
            for (int j=0; j<2; ++j)
            {
               real_t s = 0.0;
               for (int dx = 0; dx < c_dofs1D; ++dx)
               {
                  s += Bc(ex, dx) * w[j][dx][ey];
               }
               const int local_index = c_dofs1D*o_dofs1D + ey*c_dofs1D + ex;
               y(local_index, e) += s * vk(j, local_index, e);
            }
         }
      }
   });
}

static void PAHcurlVecH1IdentityApplyTranspose2D(const int c_dofs1D,
                                                 const int o_dofs1D,
                                                 const int NE,
                                                 const Array<real_t> &Bclosed,
                                                 const Array<real_t> &Bopen,
                                                 const Vector &pa_data,
                                                 const Vector &x_,
                                                 Vector &y_)
{
   auto Bc = Reshape(Bclosed.Read(), c_dofs1D, c_dofs1D);
   auto Bo = Reshape(Bopen.Read(), o_dofs1D, c_dofs1D);

   auto x = Reshape(x_.Read(), (2 * c_dofs1D * o_dofs1D), NE);
   auto y = Reshape(y_.ReadWrite(), c_dofs1D, c_dofs1D, 2, NE);

   auto vk = Reshape(pa_data.Read(), 2, (2 * c_dofs1D * o_dofs1D), NE);

   MFEM_VERIFY(c_dofs1D <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D &&
               o_dofs1D <= c_dofs1D, "");

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int MAX_D1D = DofQuadLimits::HCURL_MAX_D1D;

      real_t w[2][MAX_D1D][MAX_D1D];

      // dofs that point parallel to x-axis (open in x, closed in y)

      // contract in x
      for (int ey = 0; ey < c_dofs1D; ++ey)
      {
         for (int dx = 0; dx < c_dofs1D; ++dx)
         {
            for (int j=0; j<2; ++j) { w[j][dx][ey] = 0.0; }
         }
         for (int ex = 0; ex < o_dofs1D; ++ex)
         {
            const int local_index = ey*o_dofs1D + ex;
            const real_t xd = x(local_index, e);

            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               for (int j=0; j<2; ++j)
               {
                  w[j][dx][ey] += Bo(ex, dx) * xd * vk(j, local_index, e);
               }
            }
         }
      }

      // contract in y
      for (int dx = 0; dx < c_dofs1D; ++dx)
      {
         for (int dy = 0; dy < c_dofs1D; ++dy)
         {
            for (int j=0; j<2; ++j)
            {
               real_t s = 0.0;
               for (int ey = 0; ey < c_dofs1D; ++ey)
               {
                  s += w[j][dx][ey] * Bc(ey, dy);
               }
               y(dx, dy, j, e) += s;
            }
         }
      }

      // dofs that point parallel to y-axis (open in y, closed in x)

      // contract in x
      for (int ey = 0; ey < o_dofs1D; ++ey)
      {
         for (int dx = 0; dx < c_dofs1D; ++dx)
         {
            for (int j=0; j<2; ++j) { w[j][dx][ey] = 0.0; }
         }
         for (int ex = 0; ex < c_dofs1D; ++ex)
         {
            const int local_index = c_dofs1D*o_dofs1D + ey*c_dofs1D + ex;
            const real_t xd = x(local_index, e);
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               for (int j=0; j<2; ++j)
               {
                  w[j][dx][ey] += Bc(ex, dx) * xd * vk(j, local_index, e);
               }
            }
         }
      }

      // contract in y
      for (int dx = 0; dx < c_dofs1D; ++dx)
      {
         for (int dy = 0; dy < c_dofs1D; ++dy)
         {
            for (int j=0; j<2; ++j)
            {
               real_t s = 0.0;
               for (int ey = 0; ey < o_dofs1D; ++ey)
               {
                  s += w[j][dx][ey] * Bo(ey, dy);
               }
               y(dx, dy, j, e) += s;
            }
         }
      }
   });
}

static void PAHcurlVecH1IdentityApply3D(const int c_dofs1D,
                                        const int o_dofs1D,
                                        const int NE,
                                        const Array<real_t> &Bclosed,
                                        const Array<real_t> &Bopen,
                                        const Vector &pa_data,
                                        const Vector &x_,
                                        Vector &y_)
{
   auto Bc = Reshape(Bclosed.Read(), c_dofs1D, c_dofs1D);
   auto Bo = Reshape(Bopen.Read(), o_dofs1D, c_dofs1D);

   auto x = Reshape(x_.Read(), c_dofs1D, c_dofs1D, c_dofs1D, 3, NE);
   auto y = Reshape(y_.ReadWrite(), (3 * c_dofs1D * c_dofs1D * o_dofs1D), NE);

   auto vk = Reshape(pa_data.Read(), 3, (3 * c_dofs1D * c_dofs1D * o_dofs1D),
                     NE);
   MFEM_VERIFY(c_dofs1D <= DeviceDofQuadLimits::Get().MAX_D1D &&
               o_dofs1D <= c_dofs1D, "");

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int MAX_D1D = DofQuadLimits::HCURL_MAX_D1D;

      real_t w1[3][MAX_D1D][MAX_D1D][MAX_D1D];
      real_t w2[3][MAX_D1D][MAX_D1D][MAX_D1D];

      // dofs that point parallel to x-axis (open in x, closed in y, z)

      // contract in z
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int dx = 0; dx < c_dofs1D; ++dx)
         {
            for (int dy = 0; dy < c_dofs1D; ++dy)
            {
               for (int j=0; j<3; ++j)
               {
                  w1[j][dx][dy][ez] = 0.0;
                  for (int dz = 0; dz < c_dofs1D; ++dz)
                  {
                     w1[j][dx][dy][ez] += Bc(ez, dz) * x(dx, dy, dz, j, e);
                  }
               }
            }
         }
      }

      // contract in y
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int ey = 0; ey < c_dofs1D; ++ey)
         {
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               for (int j=0; j<3; ++j)
               {
                  w2[j][dx][ey][ez] = 0.0;
                  for (int dy = 0; dy < c_dofs1D; ++dy)
                  {
                     w2[j][dx][ey][ez] += Bc(ey, dy) * w1[j][dx][dy][ez];
                  }
               }
            }
         }
      }

      // contract in x
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int ey = 0; ey < c_dofs1D; ++ey)
         {
            for (int ex = 0; ex < o_dofs1D; ++ex)
            {
               for (int j=0; j<3; ++j)
               {
                  real_t s = 0.0;
                  for (int dx = 0; dx < c_dofs1D; ++dx)
                  {
                     s += Bo(ex, dx) * w2[j][dx][ey][ez];
                  }
                  const int local_index = ez*c_dofs1D*o_dofs1D + ey*o_dofs1D + ex;
                  y(local_index, e) += s * vk(j, local_index, e);
               }
            }
         }
      }

      // dofs that point parallel to y-axis (open in y, closed in x, z)

      // contract in z
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int dx = 0; dx < c_dofs1D; ++dx)
         {
            for (int dy = 0; dy < c_dofs1D; ++dy)
            {
               for (int j=0; j<3; ++j)
               {
                  w1[j][dx][dy][ez] = 0.0;
                  for (int dz = 0; dz < c_dofs1D; ++dz)
                  {
                     w1[j][dx][dy][ez] += Bc(ez, dz) * x(dx, dy, dz, j, e);
                  }
               }
            }
         }
      }

      // contract in y
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int ey = 0; ey < o_dofs1D; ++ey)
         {
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               for (int j=0; j<3; ++j)
               {
                  w2[j][dx][ey][ez] = 0.0;
                  for (int dy = 0; dy < c_dofs1D; ++dy)
                  {
                     w2[j][dx][ey][ez] += Bo(ey, dy) * w1[j][dx][dy][ez];
                  }
               }
            }
         }
      }

      // contract in x
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int ey = 0; ey < o_dofs1D; ++ey)
         {
            for (int ex = 0; ex < c_dofs1D; ++ex)
            {
               for (int j=0; j<3; ++j)
               {
                  real_t s = 0.0;
                  for (int dx = 0; dx < c_dofs1D; ++dx)
                  {
                     s += Bc(ex, dx) * w2[j][dx][ey][ez];
                  }
                  const int local_index = c_dofs1D*c_dofs1D*o_dofs1D +
                                          ez*c_dofs1D*o_dofs1D + ey*c_dofs1D + ex;
                  y(local_index, e) += s * vk(j, local_index, e);
               }
            }
         }
      }

      // dofs that point parallel to z-axis (open in z, closed in x, y)

      // contract in z
      for (int ez = 0; ez < o_dofs1D; ++ez)
      {
         for (int dx = 0; dx < c_dofs1D; ++dx)
         {
            for (int dy = 0; dy < c_dofs1D; ++dy)
            {
               for (int j=0; j<3; ++j)
               {
                  w1[j][dx][dy][ez] = 0.0;
                  for (int dz = 0; dz < c_dofs1D; ++dz)
                  {
                     w1[j][dx][dy][ez] += Bo(ez, dz) * x(dx, dy, dz, j, e);
                  }
               }
            }
         }
      }

      // contract in y
      for (int ez = 0; ez < o_dofs1D; ++ez)
      {
         for (int ey = 0; ey < c_dofs1D; ++ey)
         {
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               for (int j=0; j<3; ++j)
               {
                  w2[j][dx][ey][ez] = 0.0;
                  for (int dy = 0; dy < c_dofs1D; ++dy)
                  {
                     w2[j][dx][ey][ez] += Bc(ey, dy) * w1[j][dx][dy][ez];
                  }
               }
            }
         }
      }

      // contract in x
      for (int ez = 0; ez < o_dofs1D; ++ez)
      {
         for (int ey = 0; ey < c_dofs1D; ++ey)
         {
            for (int ex = 0; ex < c_dofs1D; ++ex)
            {
               for (int j=0; j<3; ++j)
               {
                  real_t s = 0.0;
                  for (int dx = 0; dx < c_dofs1D; ++dx)
                  {
                     s += Bc(ex, dx) * w2[j][dx][ey][ez];
                  }
                  const int local_index = 2*c_dofs1D*c_dofs1D*o_dofs1D +
                                          ez*c_dofs1D*c_dofs1D + ey*c_dofs1D + ex;
                  y(local_index, e) += s * vk(j, local_index, e);
               }
            }
         }
      }
   });
}

static void PAHcurlVecH1IdentityApplyTranspose3D(const int c_dofs1D,
                                                 const int o_dofs1D,
                                                 const int NE,
                                                 const Array<real_t> &Bclosed,
                                                 const Array<real_t> &Bopen,
                                                 const Vector &pa_data,
                                                 const Vector &x_,
                                                 Vector &y_)
{
   auto Bc = Reshape(Bclosed.Read(), c_dofs1D, c_dofs1D);
   auto Bo = Reshape(Bopen.Read(), o_dofs1D, c_dofs1D);

   auto x = Reshape(x_.Read(), (3 * c_dofs1D * c_dofs1D * o_dofs1D), NE);
   auto y = Reshape(y_.ReadWrite(), c_dofs1D, c_dofs1D, c_dofs1D, 3, NE);

   auto vk = Reshape(pa_data.Read(), 3, (3 * c_dofs1D * c_dofs1D * o_dofs1D),
                     NE);

   MFEM_VERIFY(c_dofs1D <= DeviceDofQuadLimits::Get().MAX_D1D &&
               o_dofs1D <= c_dofs1D, "");

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int MAX_D1D = DofQuadLimits::HCURL_MAX_D1D;

      real_t w1[3][MAX_D1D][MAX_D1D][MAX_D1D];
      real_t w2[3][MAX_D1D][MAX_D1D][MAX_D1D];

      // dofs that point parallel to x-axis (open in x, closed in y, z)

      // contract in x
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int ey = 0; ey < c_dofs1D; ++ey)
         {
            for (int j=0; j<3; ++j)
            {
               for (int dx = 0; dx < c_dofs1D; ++dx)
               {
                  w2[j][dx][ey][ez] = 0.0;
               }
               for (int ex = 0; ex < o_dofs1D; ++ex)
               {
                  const int local_index = ez*c_dofs1D*o_dofs1D + ey*o_dofs1D + ex;
                  const real_t xv = x(local_index, e) * vk(j, local_index, e);
                  for (int dx = 0; dx < c_dofs1D; ++dx)
                  {
                     w2[j][dx][ey][ez] += xv * Bo(ex, dx);
                  }
               }
            }
         }
      }

      // contract in y
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int dx = 0; dx < c_dofs1D; ++dx)
         {
            for (int dy = 0; dy < c_dofs1D; ++dy)
            {
               for (int j=0; j<3; ++j)
               {
                  w1[j][dx][dy][ez] = 0.0;
                  for (int ey = 0; ey < c_dofs1D; ++ey)
                  {
                     w1[j][dx][dy][ez] += w2[j][dx][ey][ez] * Bc(ey, dy);
                  }
               }
            }
         }
      }

      // contract in z
      for (int dx = 0; dx < c_dofs1D; ++dx)
      {
         for (int dy = 0; dy < c_dofs1D; ++dy)
         {
            for (int dz = 0; dz < c_dofs1D; ++dz)
            {
               for (int j=0; j<3; ++j)
               {
                  real_t s = 0.0;
                  for (int ez = 0; ez < c_dofs1D; ++ez)
                  {
                     s += w1[j][dx][dy][ez] * Bc(ez, dz);
                  }
                  y(dx, dy, dz, j, e) += s;
               }
            }
         }
      }

      // dofs that point parallel to y-axis (open in y, closed in x, z)

      // contract in x
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int ey = 0; ey < o_dofs1D; ++ey)
         {
            for (int j=0; j<3; ++j)
            {
               for (int dx = 0; dx < c_dofs1D; ++dx)
               {
                  w2[j][dx][ey][ez] = 0.0;
               }
               for (int ex = 0; ex < c_dofs1D; ++ex)
               {
                  const int local_index = c_dofs1D*c_dofs1D*o_dofs1D +
                                          ez*c_dofs1D*o_dofs1D + ey*c_dofs1D + ex;
                  const real_t xv = x(local_index, e) * vk(j, local_index, e);
                  for (int dx = 0; dx < c_dofs1D; ++dx)
                  {
                     w2[j][dx][ey][ez] += xv * Bc(ex, dx);
                  }
               }
            }
         }
      }

      // contract in y
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int dx = 0; dx < c_dofs1D; ++dx)
         {
            for (int dy = 0; dy < c_dofs1D; ++dy)
            {
               for (int j=0; j<3; ++j)
               {
                  w1[j][dx][dy][ez] = 0.0;
                  for (int ey = 0; ey < o_dofs1D; ++ey)
                  {
                     w1[j][dx][dy][ez] += w2[j][dx][ey][ez] * Bo(ey, dy);
                  }
               }
            }
         }
      }

      // contract in z
      for (int dx = 0; dx < c_dofs1D; ++dx)
      {
         for (int dy = 0; dy < c_dofs1D; ++dy)
         {
            for (int dz = 0; dz < c_dofs1D; ++dz)
            {
               for (int j=0; j<3; ++j)
               {
                  real_t s = 0.0;
                  for (int ez = 0; ez < c_dofs1D; ++ez)
                  {
                     s += w1[j][dx][dy][ez] * Bc(ez, dz);
                  }
                  y(dx, dy, dz, j, e) += s;
               }
            }
         }
      }

      // dofs that point parallel to z-axis (open in z, closed in x, y)

      // contract in x
      for (int ez = 0; ez < o_dofs1D; ++ez)
      {
         for (int ey = 0; ey < c_dofs1D; ++ey)
         {
            for (int j=0; j<3; ++j)
            {
               for (int dx = 0; dx < c_dofs1D; ++dx)
               {
                  w2[j][dx][ey][ez] = 0.0;
               }
               for (int ex = 0; ex < c_dofs1D; ++ex)
               {
                  const int local_index = 2*c_dofs1D*c_dofs1D*o_dofs1D +
                                          ez*c_dofs1D*c_dofs1D + ey*c_dofs1D + ex;
                  const real_t xv = x(local_index, e) * vk(j, local_index, e);
                  for (int dx = 0; dx < c_dofs1D; ++dx)
                  {
                     w2[j][dx][ey][ez] += xv * Bc(ex, dx);
                  }
               }
            }
         }
      }

      // contract in y
      for (int ez = 0; ez < o_dofs1D; ++ez)
      {
         for (int dx = 0; dx < c_dofs1D; ++dx)
         {
            for (int dy = 0; dy < c_dofs1D; ++dy)
            {
               for (int j=0; j<3; ++j)
               {
                  w1[j][dx][dy][ez] = 0.0;
                  for (int ey = 0; ey < c_dofs1D; ++ey)
                  {
                     w1[j][dx][dy][ez] += w2[j][dx][ey][ez] * Bc(ey, dy);
                  }
               }
            }
         }
      }

      // contract in z
      for (int dx = 0; dx < c_dofs1D; ++dx)
      {
         for (int dy = 0; dy < c_dofs1D; ++dy)
         {
            for (int dz = 0; dz < c_dofs1D; ++dz)
            {
               for (int j=0; j<3; ++j)
               {
                  real_t s = 0.0;
                  for (int ez = 0; ez < o_dofs1D; ++ez)
                  {
                     s += w1[j][dx][dy][ez] * Bo(ez, dz);
                  }
                  y(dx, dy, dz, j, e) += s;
               }
            }
         }
      }
   });
}

void IdentityInterpolator::AssemblePA(const FiniteElementSpace &trial_fes,
                                      const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements, with a vector test space and H^1 trial space.
   Mesh *mesh = trial_fes.GetMesh();
   const FiniteElement *trial_fel = trial_fes.GetTypicalFE();
   const FiniteElement *test_fel = test_fes.GetTypicalFE();

   const NodalTensorFiniteElement *trial_el =
      dynamic_cast<const NodalTensorFiniteElement*>(trial_fel);
   MFEM_VERIFY(trial_el != NULL, "Only NodalTensorFiniteElement is supported!");

   const VectorTensorFiniteElement *test_el =
      dynamic_cast<const VectorTensorFiniteElement*>(test_fel);
   MFEM_VERIFY(test_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const int dims = trial_el->GetDim();
   MFEM_VERIFY(dims == 2 || dims == 3, "");

   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");

   MFEM_VERIFY(trial_el->GetOrder() == test_el->GetOrder(), "");

   MFEM_VERIFY(vdim == 1, "vdim != 1 with PA is not supported yet!");

   ne = trial_fes.GetNE();

   const int order = trial_el->GetOrder();
   dofquad_fe.reset(new H1_SegmentElement(order));
   mfem::QuadratureFunctions1D qf1d;
   mfem::IntegrationRule closed_ir;
   closed_ir.SetSize(order + 1);
   qf1d.GaussLobatto(order + 1, &closed_ir);
   mfem::IntegrationRule open_ir;
   open_ir.SetSize(order);
   qf1d.GaussLegendre(order, &open_ir);

   maps_C_C = &dofquad_fe->GetDofToQuad(closed_ir, DofToQuad::TENSOR);
   maps_O_C = &dofquad_fe->GetDofToQuad(open_ir, DofToQuad::TENSOR);

   o_dofs1D = maps_O_C->nqpt;
   c_dofs1D = maps_C_C->nqpt;
   MFEM_VERIFY(maps_O_C->ndof == c_dofs1D &&
               maps_C_C->ndof == c_dofs1D, "Discrepancy in the number of DOFs");

   const int ndof_test = (dim == 3) ? 3 * c_dofs1D * c_dofs1D * o_dofs1D
                         : 2 * c_dofs1D * o_dofs1D;

   const IntegrationRule & Nodes = test_el->GetNodes();

   pa_data.SetSize(dim * ndof_test * ne, Device::GetMemoryType());
   auto op = Reshape(pa_data.HostWrite(), dim, ndof_test, ne);

   const Array<int> &dofmap = test_el->GetDofMap();

   if (dim == 3)
   {
      // Note that ND_HexahedronElement uses 6 vectors in tk rather than 3, with
      // the last 3 having negative signs. Here the signs are all positive, as
      // signs are applied in ElementRestriction.

      const real_t tk[9] = { 1.,0.,0.,  0.,1.,0.,  0.,0.,1. };

      for (int c=0; c<3; ++c)
      {
         for (int i=0; i<ndof_test/3; ++i)
         {
            const int d = (c*ndof_test/3) + i;
            // ND_HexahedronElement sets dof2tk = (dofmap < 0) ? 3+c : c, but here
            // no signs should be applied due to ElementRestriction.
            const int dof2tk = c;
            const int id = (dofmap[d] >= 0) ? dofmap[d] : -1 - dofmap[d];

            for (int e=0; e<ne; ++e)
            {
               real_t v[3];
               ElementTransformation *tr = mesh->GetElementTransformation(e);
               tr->SetIntPoint(&Nodes.IntPoint(id));
               tr->Jacobian().Mult(tk + dof2tk*dim, v);

               for (int j=0; j<3; ++j)
               {
                  op(j,d,e) = v[j];
               }
            }
         }
      }
   }
   else // 2D case
   {
      const real_t tk[4] = { 1.,0.,  0.,1. };
      for (int c=0; c<2; ++c)
      {
         for (int i=0; i<ndof_test/2; ++i)
         {
            const int d = (c*ndof_test/2) + i;
            // ND_QuadrilateralElement sets dof2tk = (dofmap < 0) ? 2+c : c, but here
            // no signs should be applied due to ElementRestriction.
            const int dof2tk = c;
            const int id = (dofmap[d] >= 0) ? dofmap[d] : -1 - dofmap[d];

            for (int e=0; e<ne; ++e)
            {
               real_t v[2];
               ElementTransformation *tr = mesh->GetElementTransformation(e);
               tr->SetIntPoint(&Nodes.IntPoint(id));
               tr->Jacobian().Mult(tk + dof2tk*dim, v);

               for (int j=0; j<2; ++j)
               {
                  op(j,d,e) = v[j];
               }
            }
         }
      }
   }
}

void IdentityInterpolator::AddMultPA(const Vector &x, Vector &y) const
{
   if (dim == 3)
   {
      PAHcurlVecH1IdentityApply3D(c_dofs1D, o_dofs1D, ne, maps_C_C->B, maps_O_C->B,
                                  pa_data, x, y);
   }
   else if (dim == 2)
   {
      PAHcurlVecH1IdentityApply2D(c_dofs1D, o_dofs1D, ne, maps_C_C->B, maps_O_C->B,
                                  pa_data, x, y);
   }
   else
   {
      mfem_error("Bad dimension!");
   }
}

void IdentityInterpolator::AddMultTransposePA(const Vector &x, Vector &y) const
{
   if (dim == 3)
   {
      PAHcurlVecH1IdentityApplyTranspose3D(c_dofs1D, o_dofs1D, ne, maps_C_C->B,
                                           maps_O_C->B, pa_data, x, y);
   }
   else if (dim == 2)
   {
      PAHcurlVecH1IdentityApplyTranspose2D(c_dofs1D, o_dofs1D, ne, maps_C_C->B,
                                           maps_O_C->B, pa_data, x, y);
   }
   else
   {
      mfem_error("Bad dimension!");
   }
}

} // namespace mfem
