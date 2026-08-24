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

namespace hyperbolic_hdg
{

/// A single element with an interior face, giving both an
/// ElementTransformation and a FaceElementTransformations with an integration
/// point set -- what the flux functions need to evaluate coefficients.
struct Geom
{
   Mesh mesh;
   ElementTransformation *Tr;
   FaceElementTransformations *FTr;

   Geom(int dim)
      : mesh((dim == 2)
             ? Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL, false,
                                     2.0, 1.0)
             : Mesh::MakeCartesian3D(2, 1, 1, Element::HEXAHEDRON,
                                     2.0, 1.0, 1.0))
   {
      Tr = mesh.GetElementTransformation(0);
      IntegrationPoint ip;
      ip.Set3(0.3, 0.4, 0.2);
      Tr->SetIntPoint(&ip);

      int f = -1;
      for (int i = 0; i < mesh.GetNumFaces(); i++)
      {
         if (mesh.FaceIsInterior(i)) { f = i; break; }
      }
      FTr = mesh.GetFaceElementTransformations(f);
      IntegrationPoint fip;
      fip.Set2(0.37, 0.0);
      FTr->SetAllIntPoints(&fip);
   }
};

/// A state with a positive density and a non-symmetric momentum, so that no
/// component can be confused with another.
void MakeState(int dim, Vector &U)
{
   U.SetSize(dim + 1);
   U(0) = 1.3;                                     // density
   for (int d = 0; d < dim; d++) { U(1 + d) = 0.4 * (d + 1) - 0.15; }
}

void RequireClose(const Vector &a, const Vector &b, real_t tol = 1e-12)
{
   REQUIRE(a.Size() == b.Size());
   for (int i = 0; i < a.Size(); i++)
   {
      CAPTURE(i, a(i), b(i));
      REQUIRE(a(i) == MFEM_Approx(b(i), tol, 1e-11));
   }
}

} // namespace hyperbolic_hdg

TEST_CASE("IsothermalFlux satisfies the FluxFunction contract",
          "[HyperbolicFlux]")
{
   using namespace hyperbolic_hdg;

   // These identities are required of any FluxFunction, so they test the
   // implementation without re-deriving the physics it encodes.
   // dim is 2 or 3 only: the flux writes dim entries, and pairing a
   // one-dimensional flux with a two-dimensional face transformation
   // overruns the normal vector and corrupts the heap. That is a property of
   // the test rather than of the code, but it is worth not repeating.
   const int dim = GENERATE(2, 3);
   const real_t cs = 1.7;
   CAPTURE(dim);

   Geom g(dim);
   IsothermalFlux flux(dim, cs);
   REQUIRE(flux.num_equations == dim + 1);
   REQUIRE(flux.dim == dim);

   Vector U;
   MakeState(dim, U);

   Vector nor(dim);
   for (int d = 0; d < dim; d++) { nor(d) = 0.6 - 0.35 * d; }

   SECTION("the normal flux is the flux contracted with the normal")
   {
      DenseMatrix F(dim + 1, dim);
      flux.ComputeFlux(U, *g.Tr, F);

      Vector FdotN(dim + 1);
      flux.ComputeFluxDotN(U, nor, *g.FTr, FdotN);

      Vector expect(dim + 1);
      F.Mult(nor, expect);
      RequireClose(FdotN, expect);
   }

   SECTION("the average of a state with itself is the flux at that state")
   {
      DenseMatrix F(dim + 1, dim), Favg(dim + 1, dim);
      flux.ComputeFlux(U, *g.Tr, F);
      flux.ComputeAvgFlux(U, U, *g.Tr, Favg);

      for (int i = 0; i < dim + 1; i++)
         for (int d = 0; d < dim; d++)
         {
            CAPTURE(i, d);
            REQUIRE(Favg(i, d) == MFEM_Approx(F(i, d), 1e-12, 1e-11));
         }

      Vector FdotN(dim + 1), FavgN(dim + 1);
      flux.ComputeFluxDotN(U, nor, *g.FTr, FdotN);
      flux.ComputeAvgFluxDotN(U, U, nor, *g.FTr, FavgN);
      RequireClose(FavgN, FdotN);
   }

   SECTION("the vacuum state gives no flux")
   {
      Vector V(dim + 1);
      V = 0.0;
      DenseMatrix F(dim + 1, dim);
      F = 1.0;
      const real_t speed = flux.ComputeFlux(V, *g.Tr, F);

      REQUIRE(speed == MFEM_Approx(0.0));
      REQUIRE(F.MaxMaxNorm() == MFEM_Approx(0.0));
   }

   SECTION("the characteristic speed is the flow speed plus the sound speed")
   {
      DenseMatrix F(dim + 1, dim);
      const real_t speed = flux.ComputeFlux(U, *g.Tr, F);

      real_t v2 = 0.0;
      for (int d = 0; d < dim; d++) { v2 += std::pow(U(1 + d) / U(0), 2); }
      const real_t expect = std::sqrt(v2) + cs;

      INFO("returned " << speed << ", |v| + c = " << expect);
      REQUIRE(speed == MFEM_Approx(expect, 1e-10, 1e-9));
   }
}

TEST_CASE("CompoundFlux replicates a scalar flux over its components",
          "[HyperbolicFlux]")
{
   using namespace hyperbolic_hdg;

   const int dim = GENERATE(2, 3);
   const int neq = 3;
   CAPTURE(dim, neq);

   Geom g(dim);

   Vector adv(dim);
   for (int d = 0; d < dim; d++) { adv(d) = 1.0 - 0.4 * d; }
   VectorConstantCoefficient acoeff(adv);
   AdvectionFlux scalar(acoeff);

   CompoundFlux compound(neq, scalar);
   REQUIRE(compound.num_equations == neq);
   REQUIRE(compound.dim == dim);

   Vector U(neq);
   for (int i = 0; i < neq; i++) { U(i) = 0.7 - 0.45 * i; }

   Vector nor(dim);
   for (int d = 0; d < dim; d++) { nor(d) = 0.6 - 0.35 * d; }

   SECTION("each row is the scalar flux of that component")
   {
      DenseMatrix Fc(neq, dim);
      const real_t speed = compound.ComputeFlux(U, *g.Tr, Fc);

      real_t max_speed = 0.0;
      for (int i = 0; i < neq; i++)
      {
         Vector Ui(1);
         Ui(0) = U(i);
         DenseMatrix Fi(1, dim);
         max_speed = std::max(max_speed, scalar.ComputeFlux(Ui, *g.Tr, Fi));

         for (int d = 0; d < dim; d++)
         {
            CAPTURE(i, d);
            REQUIRE(Fc(i, d) == MFEM_Approx(Fi(0, d), 1e-12, 1e-11));
         }
      }
      REQUIRE(speed == MFEM_Approx(max_speed, 1e-12, 1e-11));
   }

   SECTION("the normal flux is the flux contracted with the normal")
   {
      DenseMatrix Fc(neq, dim);
      compound.ComputeFlux(U, *g.Tr, Fc);

      Vector FdotN(neq);
      compound.ComputeFluxDotN(U, nor, *g.FTr, FdotN);

      Vector expect(neq);
      Fc.Mult(nor, expect);
      RequireClose(FdotN, expect);
   }
}

TEST_CASE("HDGFlux::Average selects a side and adds a scaled jump",
          "[HyperbolicFlux]")
{
   using namespace hyperbolic_hdg;

   // HDGFlux overrides Average, not Eval -- Eval is inherited Rusanov and does
   // not see the scheme or Ctau at all, which is worth knowing before writing
   // a test against it.
   const int dim = GENERATE(2, 3);
   const int neq = 3;
   CAPTURE(dim);

   Geom g(dim);

   Vector adv(dim);
   for (int d = 0; d < dim; d++) { adv(d) = 1.0 - 0.4 * d; }
   VectorConstantCoefficient acoeff(adv);
   AdvectionFlux scalar(acoeff);
   CompoundFlux flux(neq, scalar);

   Vector nor(dim);
   for (int d = 0; d < dim; d++) { nor(d) = 0.6 - 0.35 * d; }
   const real_t nlen = nor.Norml2();

   Vector U1(neq), U2(neq);
   for (int i = 0; i < neq; i++)
   {
      U1(i) = 0.83 - 0.31 * i;
      U2(i) = -0.41 + 0.22 * i;
   }

   SECTION("it is consistent: equal states give the physical flux")
   {
      for (auto s : {HDGFlux::HDGScheme::HDG_1, HDGFlux::HDGScheme::HDG_2})
      {
         HDGFlux hdg(flux, s, 1.0);
         Vector avg(neq), phys(neq);
         hdg.Average(U1, U1, nor, *g.FTr, avg);
         flux.ComputeFluxDotN(U1, nor, *g.FTr, phys);
         CAPTURE(int(s));
         RequireClose(avg, phys, 1e-11);
      }
   }

   SECTION("each scheme takes the flux from its own side")
   {
      Vector f1(neq), f2(neq);
      flux.ComputeFluxDotN(U1, nor, *g.FTr, f1);
      flux.ComputeFluxDotN(U2, nor, *g.FTr, f2);

      HDGFlux h1(flux, HDGFlux::HDGScheme::HDG_1, 0.0);
      HDGFlux h2(flux, HDGFlux::HDGScheme::HDG_2, 0.0);

      Vector a1(neq), a2(neq);
      h1.Average(U1, U2, nor, *g.FTr, a1);
      h2.Average(U1, U2, nor, *g.FTr, a2);

      RequireClose(a1, f1, 1e-11);
      RequireClose(a2, f2, 1e-11);

      // And with a jump present the two schemes genuinely differ.
      Vector d(a1);
      d -= a2;
      REQUIRE(d.Normlinf() > 1e-8);
   }

   SECTION("Ctau scales the jump term linearly")
   {
      const real_t Ctau = 2.5;
      HDGFlux h0(flux, HDGFlux::HDGScheme::HDG_1, 0.0);
      HDGFlux hc(flux, HDGFlux::HDGScheme::HDG_1, Ctau);

      Vector a0(neq), ac(neq);
      h0.Average(U1, U2, nor, *g.FTr, a0);
      hc.Average(U1, U2, nor, *g.FTr, ac);

      for (int i = 0; i < neq; i++)
      {
         const real_t expect = a0(i) + Ctau * (U2(i) - U1(i)) * nlen;
         CAPTURE(i, ac(i), expect);
         REQUIRE(ac(i) == MFEM_Approx(expect, 1e-11, 1e-10));
      }
   }

   SECTION("AverageGrad is the derivative of Average")
   {
      // The same discipline as everywhere else in this branch: an analytic
      // Jacobian is only worth having if it is checked against a difference of
      // the thing it differentiates.
      const auto s = GENERATE(HDGFlux::HDGScheme::HDG_1,
                              HDGFlux::HDGScheme::HDG_2);
      const int side = GENERATE(1, 2);
      CAPTURE(int(s), side);

      HDGFlux hdg(flux, s, 1.3);

      DenseMatrix grad(neq, neq);
      hdg.AverageGrad(side, U1, U2, nor, *g.FTr, grad);

      const real_t h = std::cbrt(std::numeric_limits<real_t>::epsilon());
      for (int j = 0; j < neq; j++)
      {
         Vector p1(U1), p2(U2), m1(U1), m2(U2);
         if (side == 1) { p1(j) += h; m1(j) -= h; }
         else           { p2(j) += h; m2(j) -= h; }

         Vector ap(neq), am(neq);
         hdg.Average(p1, p2, nor, *g.FTr, ap);
         hdg.Average(m1, m2, nor, *g.FTr, am);

         for (int i = 0; i < neq; i++)
         {
            const real_t fd = (ap(i) - am(i)) / (2.0 * h);
            CAPTURE(i, j, grad(i, j), fd);
            REQUIRE(grad(i, j) == MFEM_Approx(fd, 1e-7, 1e-6));
         }
      }
   }
}
