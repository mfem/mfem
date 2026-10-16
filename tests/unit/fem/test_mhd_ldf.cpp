// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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
#include "../../../examples/ex42.hpp"
#include <limits>

using namespace mfem;

namespace
{

// Independently assemble the weak physical divergence operator. For quads,
// use the full nodal Q^p test basis, including its redundant constraint;
// the projector implementation instead uses a reduced Legendre basis.
DenseMatrix AssembleMagneticDivergence(const FiniteElement &fe,
                                       ElementTransformation &Tr)
{
   const int p = fe.GetOrder();
   const int qorder = fe.GetGeomType() == Geometry::TRIANGLE ?
                      std::max(0, p-1) : p;
   L2_FECollection qfec(qorder, 2);
   const FiniteElement &qfe =
      *qfec.FiniteElementForGeometry(fe.GetGeomType());
   const int dof = fe.GetDof();
   DenseMatrix D(qfe.GetDof(), 2*dof);
   D = 0.0;
   Vector qshape(qfe.GetDof());
   DenseMatrix grad(dof, 2);
   const IntegrationRule &ir = IntRules.Get(fe.GetGeomType(), 2*p+4);
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      Tr.SetIntPoint(&ip);
      qfe.CalcShape(ip, qshape);
      fe.CalcPhysDShape(Tr, grad);
      const real_t weight = ip.weight * Tr.Weight();
      for (int a = 0; a < qfe.GetDof(); a++)
      {
         for (int j = 0; j < dof; j++)
         {
            D(a,j) += weight * qshape(a) * grad(j,0);
            D(a,dof+j) += weight * qshape(a) * grad(j,1);
         }
      }
   }
   return D;
}

}

TEST_CASE("Magnetic LDF projectors have zero physical divergence",
          "[LDF][MHD]")
{
   for (int components : {2, 3})
      for (auto geometry : {Element::TRIANGLE, Element::QUADRILATERAL})
      {
         for (int order = 0; order <= 4; order++)
         {
            Mesh mesh = Mesh::MakeCartesian2D(2, 2, geometry, true);
            DG_FECollection fec(order, 2, BasisType::GaussLobatto);
            FiniteElementSpace fes(&mesh, &fec, 2*components+2, Ordering::byNODES);
            IdealMHDFlux flux(2, 5.0/3.0, components);
            RusanovFlux numerical_flux(flux);
            GPsource source(2, components);
            RusanovGPsource numerical_source(source);
            DG_MHD_GPsource op(fes,
                               std::unique_ptr<HyperbolicFormIntegrator>(
                                  new HyperbolicFormIntegrator(numerical_flux)),
                               std::unique_ptr<GPsourceFormIntegrator>(
                                  new GPsourceFormIntegrator(numerical_source)),
                               true, true);
            for (int deformed = 0; deformed < 2; deformed++)
            {
               if (deformed)
               {
                  // Triangles remain affine; quads become non-affine bilinear
                  // cells, so the updated physical divergence operator changes.
                  mesh.Transform([](const Vector &x, Vector &y)
                  {
                     y.SetSize(2);
                     y(0) = x(0) + 0.25*x(0)*x(1) + 0.1*x(1);
                     y(1) = x(1) + 0.12*x(0)*x(1);
                  });
                  op.Update();
               }
               for (int element = 0; element < fes.GetNE(); element++)
               {
                  const FiniteElement &fe = *fes.GetFE(element);
                  const DenseMatrix D = AssembleMagneticDivergence(
                                           fe, *fes.GetElementTransformation(element));
                  const DenseMatrix &P = op.GetLDFProjector(element);
                  DenseMatrix DP(D.Height(), P.Width());
                  mfem::Mult(D, P, DP);
                  const real_t scale = std::max(real_t(1.0), D.MaxMaxNorm());
                  const real_t tolerance = 256 *
                                           std::numeric_limits<real_t>::epsilon() * P.Height() * scale;
                  CAPTURE(geometry, order, deformed, element, DP.MaxMaxNorm(), tolerance);
                  REQUIRE(DP.MaxMaxNorm() <= tolerance);
                  DenseMatrix PP(P.Height());
                  mfem::Mult(P, P, PP);
                  PP -= P;
                  REQUIRE(PP.MaxMaxNorm() <= tolerance);
                  // A zero matrix also satisfies D*P=0 and P*P=P. Preserve
                  // each constant magnetic component independently as well.
                  for (int component = 0; component < 2; component++)
                  {
                     Vector constant(P.Width()), projected(P.Height());
                     constant = 0.0;
                     for (int j = 0; j < fe.GetDof(); j++)
                     {
                        constant(component*fe.GetDof()+j) = 1.0;
                     }
                     P.Mult(constant, projected);
                     projected -= constant;
                     REQUIRE(projected.Normlinf() <= tolerance);
                  }
               }
            }
         }
      }
}

TEST_CASE("Ideal MHD normal flux includes magnetic stresses", "[MHD]")
{
   const real_t gamma = 5.0/3.0;
   Vector state({2.0, 2.0, -1.0, 0.3, 0.4, 2.875});
   IdealMHDFlux flux(2, gamma);
   Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
   ElementTransformation &Tr = *mesh.GetElementTransformation(0);
   FaceElementTransformations face;
   DenseMatrix F(6, 2);
   flux.ComputeFlux(state, Tr, F);
   Vector normal({2.0, 0.0}), Fn(6);
   const real_t speed = flux.ComputeFluxDotN(state, normal, face, Fn);
   // p=1, p_total=1.125, u=(1,-0.5), u dot B=0.1.
   Vector expected({4.0, 6.07, -2.24, 0.0, 1.1, 7.94});
   const real_t tol = 512*std::numeric_limits<real_t>::epsilon();
   for (int i = 0; i < 6; i++)
   {
      REQUIRE(std::abs(Fn(i)-expected(i)) <= tol);
      REQUIRE(std::abs(Fn(i)-2*F(i,0)) <= tol);
   }
   const real_t cf = std::sqrt((gamma+0.25 +
                                std::sqrt((gamma-0.25)*(gamma-0.25)+
                                          4*gamma*0.16))/4);
   REQUIRE(std::abs(speed-(1+cf)) <= tol);
}

TEST_CASE("Godunov-Powell sources have the physical sign and state ordering",
          "[MHD]")
{
   Vector state({2.0, 2.0, -1.0, 0.3, 0.4, 2.875}), value(6);
   GPsource source(2);
   Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
   ElementTransformation &Tr = *mesh.GetElementTransformation(0);
   source.ComputeSource(state, 2.0, Tr, value);
   Vector expected({0.0, -0.6, -0.8, -2.0, 1.0, -0.2});
   value -= expected;
   const real_t tol = 512*std::numeric_limits<real_t>::epsilon();
   REQUIRE(value.Normlinf() <= tol);
   RusanovGPsource numerical_source(source);
   FaceElementTransformations face;
   Vector normal({2.0, -1.0}), left(6), right(6);
   numerical_source.Eval(state, state, normal, face, left, right);
   REQUIRE(left.Normlinf() <= tol);
   REQUIRE(right.Normlinf() <= tol);
}

TEST_CASE("2.5D MHD retains transverse fluxes and GP terms", "[MHD]")
{
   Vector state({2.0, 2.0, -1.0, 0.6, 0.3, 0.4, 0.5, 3.09});
   IdealMHDFlux flux(2, 5.0/3.0, 3);
   GPsource source(2, 3);
   Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
   ElementTransformation &Tr = *mesh.GetElementTransformation(0);
   FaceElementTransformations face;
   DenseMatrix F(8, 2);
   flux.ComputeFlux(state, Tr, F);
   Vector normal({2.0, 0.0}), Fn(8), value(8);
   flux.ComputeFluxDotN(state, normal, face, Fn);
   Vector expected({4.0, 6.32, -2.24, 0.9, 0.0, 1.1, 0.82, 8.53});
   const real_t tol = 512*std::numeric_limits<real_t>::epsilon();
   for (int i = 0; i < 8; i++)
   {
      REQUIRE(std::abs(Fn(i)-expected(i)) <= tol);
      REQUIRE(std::abs(Fn(i)-2*F(i,0)) <= tol);
   }
   source.ComputeSource(state, 2.0, Tr, value);
   Vector expected_source({0.0, -0.6, -0.8, -1.0, -2.0, 1.0, -0.6, -0.5});
   value -= expected_source;
   REQUIRE(value.Normlinf() <= tol);
}

TEST_CASE("2.5D LDF changes only the in-plane magnetic components",
          "[MHD][LDF]")
{
   Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
   DG_FECollection fec(2, 2, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec, 8, Ordering::byNODES);
   IdealMHDFlux flux(2, 5.0/3.0, 3);
   RusanovFlux numerical_flux(flux);
   GPsource source(2, 3);
   RusanovGPsource numerical_source(source);
   DG_MHD_GPsource op(fes,
                      std::unique_ptr<HyperbolicFormIntegrator>(
                         new HyperbolicFormIntegrator(numerical_flux)),
                      std::unique_ptr<GPsourceFormIntegrator>(
                         new GPsourceFormIntegrator(numerical_source)),
                      true, true);
   VectorFunctionCoefficient initial(8, [](const Vector &x, Vector &u)
   {
      for (int i = 0; i < 8; i++) { u(i) = (i+1)*(1+x(0)+2*x(1)); }
   });
   GridFunction state(&fes);
   state.ProjectCoefficient(initial);
   Vector before(state);
   op.ApplyLDF(state);
   const int dof = fes.GetNDofs();
   const real_t tol = 512*std::numeric_limits<real_t>::epsilon();
   for (int component : {0, 1, 2, 3, 6, 7})
   {
      for (int j = 0; j < dof; j++)
      {
         REQUIRE(state(component*dof+j) == before(component*dof+j));
      }
   }
   Vector magnetic(state.GetData()+4*dof, 2*dof);
   const DenseMatrix D = AssembleMagneticDivergence(
                            *fes.GetFE(0), *fes.GetElementTransformation(0));
   Vector divergence(D.Height());
   D.Mult(magnetic, divergence);
   REQUIRE(divergence.Normlinf() <= tol*D.MaxMaxNorm()*magnetic.Normlinf());
   state -= before;
   REQUIRE(state.Normlinf() > 0.1);
}

TEST_CASE("Smooth Alfven data satisfy the time-dependent MHD equations",
          "[MHD]")
{
   Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
   ElementTransformation &Tr = *mesh.GetElementTransformation(0);
   auto exact = MHDInitialCondition(2, 5.0/3.0);
   IdealMHDFlux flux(2, 5.0/3.0, 3);
   IntegrationPoint ip;
   ip.Set2(0.271, 0.387);
   const real_t t = 0.137, h = 1e-4;
   Vector plus(8), minus(8), residual(8);
   exact.SetTime(t+h);
   exact.Eval(plus, Tr, ip);
   exact.SetTime(t-h);
   exact.Eval(minus, Tr, ip);
   subtract(plus, minus, residual);
   residual /= 2*h;
   exact.SetTime(t);
   DenseMatrix Fplus(8, 2), Fminus(8, 2);
   for (int direction = 0; direction < 2; direction++)
   {
      IntegrationPoint shifted(ip);
      if (direction == 0) { shifted.x += h; }
      else { shifted.y += h; }
      exact.Eval(plus, Tr, shifted);
      Tr.SetIntPoint(&shifted);
      flux.ComputeFlux(plus, Tr, Fplus);
      if (direction == 0) { shifted.x -= 2*h; }
      else { shifted.y -= 2*h; }
      exact.Eval(minus, Tr, shifted);
      Tr.SetIntPoint(&shifted);
      flux.ComputeFlux(minus, Tr, Fminus);
      for (int i = 0; i < 8; i++)
      {
         residual(i) += (Fplus(i,direction)-Fminus(i,direction))/(2*h);
      }
   }
   REQUIRE(residual.Normlinf() < 1e-7);
}
