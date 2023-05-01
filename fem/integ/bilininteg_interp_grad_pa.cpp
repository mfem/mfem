// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
#include "../ceed/integrators/interp/interp.hpp"

namespace mfem
{

void GradientInterpolator::AssemblePA(const FiniteElementSpace &trial_fes,
                                      const FiniteElementSpace &test_fes)
{
   Mesh *mesh = trial_fes.GetMesh();
   if (mesh->GetNE() == 0) { return; }
   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      ceedOp = new ceed::PADiscreteInterpolator(*this, trial_fes, test_fes);
      return;
   }

   // Assumes tensor-product elements, with a vector test space and H^1 trial space.
   const FiniteElement *trial_fel = trial_fes.GetFE(0);
   const FiniteElement *test_fel = test_fes.GetFE(0);

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

// Apply to x corresponding to DOFs in H^1 (domain) the (topological) gradient
// to get a dof in H(curl) (range). You can think of the range as the "test" space
// and the domain as the "trial" space, but there's no integration.
static void PAHcurlApplyGradient2D(const int c_dofs1D,
                                   const int o_dofs1D,
                                   const int NE,
                                   const Array<double> &B_,
                                   const Array<double> &G_,
                                   const Vector &x_,
                                   Vector &y_)
{
   auto B = Reshape(B_.Read(), c_dofs1D, c_dofs1D);
   auto G = Reshape(G_.Read(), o_dofs1D, c_dofs1D);

   auto x = Reshape(x_.Read(), c_dofs1D, c_dofs1D, NE);
   auto y = Reshape(y_.ReadWrite(), 2 * c_dofs1D * o_dofs1D, NE);

   constexpr static int MAX_D1D = HCURL_MAX_D1D;
   MFEM_VERIFY(c_dofs1D <= MAX_D1D && o_dofs1D <= c_dofs1D, "");

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      double w[MAX_D1D][MAX_D1D];

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
            double s = 0.0;
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
            double s = 0.0;
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
                                      const Array<double> &G_,
                                      const Vector &x_,
                                      Vector &y_)
{
   auto G = Reshape(G_.Read(), o_dofs1D, c_dofs1D);

   auto x = Reshape(x_.Read(), c_dofs1D, c_dofs1D, NE);
   auto y = Reshape(y_.ReadWrite(), 2 * c_dofs1D * o_dofs1D, NE);

   constexpr static int MAX_D1D = HCURL_MAX_D1D;
   MFEM_VERIFY(c_dofs1D <= MAX_D1D && o_dofs1D <= c_dofs1D, "");

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      double w[MAX_D1D][MAX_D1D];

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
            double s = 0.0;
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
            const double s = w[dx][ey];
            const int local_index = c_dofs1D * o_dofs1D + ey*c_dofs1D + ex;
            y(local_index, e) += s;
         }
      }
   });
}

static void PAHcurlApplyGradientTranspose2D(
   const int c_dofs1D, const int o_dofs1D, const int NE,
   const Array<double> &B_, const Array<double> &G_,
   const Vector &x_, Vector &y_)
{
   auto B = Reshape(B_.Read(), c_dofs1D, c_dofs1D);
   auto G = Reshape(G_.Read(), o_dofs1D, c_dofs1D);

   auto x = Reshape(x_.Read(), 2 * c_dofs1D * o_dofs1D, NE);
   auto y = Reshape(y_.ReadWrite(), c_dofs1D, c_dofs1D, NE);

   constexpr static int MAX_D1D = HCURL_MAX_D1D;
   MFEM_VERIFY(c_dofs1D <= MAX_D1D && o_dofs1D <= c_dofs1D, "");

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      double w[MAX_D1D][MAX_D1D];

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
            double s = 0.0;
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
            double s = 0.0;
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
   const Array<double> &G_,
   const Vector &x_, Vector &y_)
{
   auto G = Reshape(G_.Read(), o_dofs1D, c_dofs1D);

   auto x = Reshape(x_.Read(), 2 * c_dofs1D * o_dofs1D, NE);
   auto y = Reshape(y_.ReadWrite(), c_dofs1D, c_dofs1D, NE);

   constexpr static int MAX_D1D = HCURL_MAX_D1D;
   MFEM_VERIFY(c_dofs1D <= MAX_D1D && o_dofs1D <= c_dofs1D, "");

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      double w[MAX_D1D][MAX_D1D];

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
            double s = 0.0;
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
            const double s = w[dy][ex];
            y(dx, dy, e) += s;
         }
      }
   });
}

static void PAHcurlApplyGradient3D(const int c_dofs1D,
                                   const int o_dofs1D,
                                   const int NE,
                                   const Array<double> &B_,
                                   const Array<double> &G_,
                                   const Vector &x_,
                                   Vector &y_)
{
   auto B = Reshape(B_.Read(), c_dofs1D, c_dofs1D);
   auto G = Reshape(G_.Read(), o_dofs1D, c_dofs1D);

   auto x = Reshape(x_.Read(), c_dofs1D, c_dofs1D, c_dofs1D, NE);
   auto y = Reshape(y_.ReadWrite(), (3 * c_dofs1D * c_dofs1D * o_dofs1D), NE);

   constexpr static int MAX_D1D = HCURL_MAX_D1D;
   MFEM_VERIFY(c_dofs1D <= MAX_D1D && o_dofs1D <= c_dofs1D, "");

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      double w1[MAX_D1D][MAX_D1D][MAX_D1D];
      double w2[MAX_D1D][MAX_D1D][MAX_D1D];

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
               double s = 0.0;
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
               double s = 0.0;
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
               double s = 0.0;
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
                                      const Array<double> &G_,
                                      const Vector &x_,
                                      Vector &y_)
{
   auto G = Reshape(G_.Read(), o_dofs1D, c_dofs1D);

   auto x = Reshape(x_.Read(), c_dofs1D, c_dofs1D, c_dofs1D, NE);
   auto y = Reshape(y_.ReadWrite(), (3 * c_dofs1D * c_dofs1D * o_dofs1D), NE);

   constexpr static int MAX_D1D = HCURL_MAX_D1D;
   MFEM_VERIFY(c_dofs1D <= MAX_D1D && o_dofs1D <= c_dofs1D, "");

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      double w1[MAX_D1D][MAX_D1D][MAX_D1D];
      double w2[MAX_D1D][MAX_D1D][MAX_D1D];

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
               double s = 0.0;
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
               const double s = w2[dx][ey][ez];
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
               const double s = w2[dx][ey][ez];
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
   const Array<double> &B_, const Array<double> &G_,
   const Vector &x_, Vector &y_)
{
   auto B = Reshape(B_.Read(), c_dofs1D, c_dofs1D);
   auto G = Reshape(G_.Read(), o_dofs1D, c_dofs1D);

   auto x = Reshape(x_.Read(), (3 * c_dofs1D * c_dofs1D * o_dofs1D), NE);
   auto y = Reshape(y_.ReadWrite(), c_dofs1D, c_dofs1D, c_dofs1D, NE);

   constexpr static int MAX_D1D = HCURL_MAX_D1D;
   MFEM_VERIFY(c_dofs1D <= MAX_D1D && o_dofs1D <= c_dofs1D, "");

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      double w1[MAX_D1D][MAX_D1D][MAX_D1D];
      double w2[MAX_D1D][MAX_D1D][MAX_D1D];
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
               double s = 0.0;
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
               double s = 0.0;
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
               double s = 0.0;
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
   const Array<double> &G_,
   const Vector &x_, Vector &y_)
{
   auto G = Reshape(G_.Read(), o_dofs1D, c_dofs1D);

   auto x = Reshape(x_.Read(), (3 * c_dofs1D * c_dofs1D * o_dofs1D), NE);
   auto y = Reshape(y_.ReadWrite(), c_dofs1D, c_dofs1D, c_dofs1D, NE);

   constexpr static int MAX_D1D = HCURL_MAX_D1D;
   MFEM_VERIFY(c_dofs1D <= MAX_D1D && o_dofs1D <= c_dofs1D, "");

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      double w1[MAX_D1D][MAX_D1D][MAX_D1D];
      double w2[MAX_D1D][MAX_D1D][MAX_D1D];
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
               double s = 0.0;
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
               double s = w2[ex][dy][dz];
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
               double s = w2[ex][dy][dz];
               y(dx, dy, dz, e) += s;
            }
         }
      }
   });
}

void GradientInterpolator::AddMultPA(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      ceedOp->AddMult(x, y);
   }
   else
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
         MFEM_ABORT("Bad dimension!");
      }
   }
}

void GradientInterpolator::AddMultTransposePA(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      ceedOp->AddMultTranspose(x, y);
   }
   else
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
         MFEM_ABORT("Bad dimension!");
      }
   }
}

} // namespace mfem
