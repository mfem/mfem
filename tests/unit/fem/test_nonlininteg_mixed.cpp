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

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

namespace nonlininteg_mixed
{

// A MixedFluxFunction supplies its own dual-flux Jacobian rather than having
// one differenced out of it. That is worth having, and it is also invisible
// when wrong: in a hybridized method the Jacobian is never assembled globally,
// so an error in it does not produce a wrong answer, only slow Newton
// convergence -- which a passing regression suite will not notice. The only
// thing that catches it is differencing the dual flux and comparing.

/// Central difference of ComputeDualFlux with respect to the state.
void FDStateJacobian(const MixedFluxFunction &fun, const Vector &u,
                     const DenseMatrix &F, ElementTransformation &Tr,
                     DenseMatrix &J_u_fd)
{
   const int dim = F.Width();
   const real_t h = 1e-6;

   Vector up(u), um(u);
   up(0) += h;
   um(0) -= h;

   DenseMatrix Dp(1, dim), Dm(1, dim);
   fun.ComputeDualFlux(up, F, Tr, Dp);
   fun.ComputeDualFlux(um, F, Tr, Dm);

   J_u_fd.SetSize(dim, 1);
   for (int i = 0; i < dim; i++)
   {
      J_u_fd(i, 0) = (Dp(0, i) - Dm(0, i)) / (2.0 * h);
   }
}

/// Central difference of ComputeDualFlux with respect to the flux.
void FDFluxJacobian(const MixedFluxFunction &fun, const Vector &u,
                    const DenseMatrix &F, ElementTransformation &Tr,
                    DenseMatrix &J_F_fd)
{
   const int dim = F.Width();
   const real_t h = 1e-6;

   J_F_fd.SetSize(dim, dim);
   for (int j = 0; j < dim; j++)
   {
      DenseMatrix Fp(F), Fm(F);
      Fp(0, j) += h;
      Fm(0, j) -= h;

      DenseMatrix Dp(1, dim), Dm(1, dim);
      fun.ComputeDualFlux(u, Fp, Tr, Dp);
      fun.ComputeDualFlux(u, Fm, Tr, Dm);

      for (int i = 0; i < dim; i++)
      {
         J_F_fd(i, j) = (Dp(0, i) - Dm(0, i)) / (2.0 * h);
      }
   }
}

/// Compare the analytic Jacobian of @a fun against a central difference of its
/// own dual flux, at the given state and flux.
void CheckJacobian(const MixedFluxFunction &fun, const Vector &u,
                   const DenseMatrix &F, ElementTransformation &Tr,
                   real_t tol = 1e-6)
{
   const int dim = F.Width();

   DenseMatrix J_u(dim, 1), J_F(dim, dim);
   J_u = 0.0;
   J_F = 0.0;
   fun.ComputeDualFluxJacobian(u, F, Tr, J_u, J_F);

   DenseMatrix J_u_fd, J_F_fd;
   FDStateJacobian(fun, u, F, Tr, J_u_fd);
   FDFluxJacobian(fun, u, F, Tr, J_F_fd);

   for (int i = 0; i < dim; i++)
   {
      INFO("d(dualFlux_" << i << ")/du : analytic " << J_u(i, 0)
           << " vs difference " << J_u_fd(i, 0));
      REQUIRE(J_u(i, 0) == MFEM_Approx(J_u_fd(i, 0), tol, tol));
   }

   for (int i = 0; i < dim; i++)
   {
      for (int j = 0; j < dim; j++)
      {
         INFO("d(dualFlux_" << i << ")/d(flux_" << j << ") : analytic "
              << J_F(i, j) << " vs difference " << J_F_fd(i, j));
         REQUIRE(J_F(i, j) == MFEM_Approx(J_F_fd(i, j), tol, tol));
      }
   }
}

/// A single element, with an interior integration point set, so the flux
/// functions have somewhere to evaluate their coefficients.
struct OneElement
{
   Mesh mesh;
   ElementTransformation *Tr;

   OneElement(int dim)
      : mesh((dim == 2)
             ? Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL, false,
                                     1.0, 1.0)
             : Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON,
                                     1.0, 1.0, 1.0))
   {
      Tr = mesh.GetElementTransformation(0);
      IntegrationPoint ip;
      ip.Set3(0.3, 0.6, 0.2);
      Tr->SetIntPoint(&ip);
   }
};

} // namespace nonlininteg_mixed

TEST_CASE("LinearDiffusionFlux Jacobian matches its dual flux",
          "[MixedFluxFunction]")
{
   using namespace nonlininteg_mixed;

   const int dim = GENERATE(2, 3);
   OneElement el(dim);

   Vector u(1);
   u(0) = 0.7;

   DenseMatrix F(1, dim);
   for (int i = 0; i < dim; i++) { F(0, i) = 0.5 * (i + 1); }

   SECTION("scalar coefficient")
   {
      ConstantCoefficient c(2.5);
      LinearDiffusionFlux fun(dim, c);
      CheckJacobian(fun, u, F, *el.Tr);
   }

   SECTION("vector coefficient")
   {
      Vector kv(dim);
      for (int i = 0; i < dim; i++) { kv(i) = 1.0 + i; }   // anisotropic
      VectorConstantCoefficient vc(kv);
      LinearDiffusionFlux fun(vc);
      CheckJacobian(fun, u, F, *el.Tr);
   }

   SECTION("matrix coefficient")
   {
      DenseMatrix km(dim);
      km = 0.0;
      for (int i = 0; i < dim; i++) { km(i, i) = 1.0 + i; }
      km(0, 1) = km(1, 0) = 0.25;                          // off-diagonal
      MatrixConstantCoefficient mc(km);
      LinearDiffusionFlux fun(mc);
      CheckJacobian(fun, u, F, *el.Tr);
   }
}

TEST_CASE("FunctionDiffusionFlux Jacobian matches its dual flux",
          "[MixedFluxFunction]")
{
   using namespace nonlininteg_mixed;

   const int dim = GENERATE(2, 3);
   OneElement el(dim);

   Vector u(1);
   u(0) = 0.7;

   DenseMatrix F(1, dim);
   for (int i = 0; i < dim; i++) { F(0, i) = 0.5 * (i + 1); }

   SECTION("scalar function of the state")
   {
      // kappa^-1 = 1 + u^2, so the state Jacobian is genuinely nonzero.
      auto f  = [](const Vector &, real_t s) { return 1.0 + s * s; };
      auto df = [](const Vector &, real_t s) { return 2.0 * s; };
      FunctionDiffusionFlux fun(dim, f, df);
      CheckJacobian(fun, u, F, *el.Tr);
   }

   SECTION("vector function of the state")
   {
      auto f = [](const Vector &, real_t s, Vector &k)
      {
         for (int i = 0; i < k.Size(); i++) { k(i) = (1.0 + i) + s * s; }
      };
      auto df = [](const Vector &, real_t s, Vector &k)
      {
         k = 2.0 * s;
      };
      FunctionDiffusionFlux fun(dim,
                                std::function<void(const Vector &, real_t, Vector &)>(f),
                                std::function<void(const Vector &, real_t, Vector &)>(df));
      CheckJacobian(fun, u, F, *el.Tr);
   }

   SECTION("matrix function of the state")
   {
      auto f = [](const Vector &, real_t s, DenseMatrix &k)
      {
         const int d = k.Height();
         k = 0.0;
         for (int i = 0; i < d; i++) { k(i, i) = (1.0 + i) + s * s; }
         k(0, 1) = k(1, 0) = 0.25;
      };
      auto df = [](const Vector &, real_t s, DenseMatrix &k)
      {
         const int d = k.Height();
         k = 0.0;
         for (int i = 0; i < d; i++) { k(i, i) = 2.0 * s; }
      };
      FunctionDiffusionFlux fun(dim,
                                std::function<void(const Vector &, real_t, DenseMatrix &)>(f),
                                std::function<void(const Vector &, real_t, DenseMatrix &)>(df));
      CheckJacobian(fun, u, F, *el.Tr);
   }
}
