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

// Independently assemble the weak physical divergence operator. For tensor-product cells,
// use the full nodal Q^p test basis, including its redundant constraint;
// the projector implementation instead uses a reduced Legendre basis.
DenseMatrix AssembleMagneticDivergence(const FiniteElement &fe,
                                       ElementTransformation &Tr)
{
   const int p = fe.GetOrder();
   const int dim = fe.GetDim();
   const bool simplex = fe.GetGeomType() == Geometry::SEGMENT ||
                        fe.GetGeomType() == Geometry::TRIANGLE ||
                        fe.GetGeomType() == Geometry::TETRAHEDRON;
   const int qorder = simplex ? std::max(0, p-1) : p;
   L2_FECollection qfec(qorder, dim);
   const FiniteElement &qfe =
      *qfec.FiniteElementForGeometry(fe.GetGeomType());
   const int dof = fe.GetDof();
   DenseMatrix D(qfe.GetDof(), dim*dof);
   D = 0.0;
   Vector qshape(qfe.GetDof());
   DenseMatrix grad(dof, dim);
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
            for (int c = 0; c < dim; c++)
            {
               D(a,c*dof+j) += weight * qshape(a) * grad(j,c);
            }
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
            LDF_DG_GP_MHD op(fes,
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
   LDF_DG_GP_MHD op(fes,
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
   for (int dim : {1, 2, 3})
   {
      Mesh mesh = dim == 1 ? Mesh::MakeCartesian1D(1) :
                  dim == 2 ? Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL) :
                  Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON);
      ElementTransformation &Tr = *mesh.GetElementTransformation(0);
      auto exact = MHDInitialCondition(2, 5.0/3.0, dim);
      IdealMHDFlux flux(dim, 5.0/3.0, 3);
      IntegrationPoint ip;
      ip.Set3(0.271, 0.387, 0.153);
      const real_t t = 0.137, h = 1e-4;
      Vector plus(8), minus(8), residual(8);
      exact.SetTime(t+h);
      exact.Eval(plus, Tr, ip);
      exact.SetTime(t-h);
      exact.Eval(minus, Tr, ip);
      subtract(plus, minus, residual);
      residual /= 2*h;
      exact.SetTime(t);
      DenseMatrix Fplus(8, dim), Fminus(8, dim);
      for (int direction = 0; direction < dim; direction++)
      {
         IntegrationPoint shifted(ip);
         real_t &coordinate = direction == 0 ? shifted.x :
                              direction == 1 ? shifted.y : shifted.z;
         coordinate += h;
         exact.Eval(plus, Tr, shifted);
         Tr.SetIntPoint(&shifted);
         flux.ComputeFlux(plus, Tr, Fplus);
         coordinate -= 2*h;
         exact.Eval(minus, Tr, shifted);
         Tr.SetIntPoint(&shifted);
         flux.ComputeFlux(minus, Tr, Fminus);
         for (int i = 0; i < 8; i++)
         {
            residual(i) += (Fplus(i,direction)-Fminus(i,direction))/(2*h);
         }
      }
      CAPTURE(dim, residual.Normlinf());
      REQUIRE(residual.Normlinf() < 1e-7);
      exact.SetTime(0.0);
      exact.Eval(plus, Tr, ip);
      exact.SetTime(1.0);
      exact.Eval(minus, Tr, ip);
      plus -= minus;
      REQUIRE(plus.Normlinf() < 128*std::numeric_limits<real_t>::epsilon());
   }
}

TEST_CASE("1D LDF is the mean projection of the longitudinal magnetic field",
          "[MHD][LDF]")
{
   for (int order = 0; order <= 4; order++)
   {
      Mesh mesh = Mesh::MakeCartesian1D(3);
      DG_FECollection fec(order, 1, BasisType::GaussLobatto);
      FiniteElementSpace fes(&mesh, &fec, 8, Ordering::byNODES);
      IdealMHDFlux flux(1, 5.0/3.0, 3);
      RusanovFlux numerical_flux(flux);
      GPsource source(1, 3);
      RusanovGPsource numerical_source(source);
      LDF_DG_GP_MHD op(fes,
                       std::unique_ptr<HyperbolicFormIntegrator>(
                          new HyperbolicFormIntegrator(numerical_flux)),
                       std::unique_ptr<GPsourceFormIntegrator>(
                          new GPsourceFormIntegrator(numerical_source)),
                       true, true);
      VectorFunctionCoefficient initial(8, [](const Vector &x, Vector &u)
      {
         for (int i = 0; i < 8; i++) { u(i) = (i+1)*(1+x(0)); }
      });
      GridFunction state(&fes);
      state.ProjectCoefficient(initial);
      Vector before(state);
      op.ApplyLDF(state);
      const int ndofs = fes.GetNDofs();
      // Only B_x (component 4) may change; transverse magnetic components,
      // density, all momenta, and total energy must be exactly unchanged.
      for (int c : {0, 1, 2, 3, 5, 6, 7})
      {
         for (int j = 0; j < ndofs; j++)
         {
            REQUIRE(state(c*ndofs+j) == before(c*ndofs+j));
         }
      }
      for (int k = 0; k < fes.GetNE(); k++)
      {
         const FiniteElement &fe = *fes.GetFE(k);
         const int dof = fe.GetDof();
         ElementTransformation &Tr = *fes.GetElementTransformation(k);
         const DenseMatrix &P = op.GetLDFProjector(k);
         DenseMatrix D = AssembleMagneticDivergence(fe, Tr);
         DenseMatrix DP(D.Height(), P.Width()), PP(P.Height());
         mfem::Mult(D, P, DP);
         mfem::Mult(P, P, PP);
         PP -= P;
         const real_t tol = 1024*std::numeric_limits<real_t>::epsilon()*dof;
         CAPTURE(order, k);
         REQUIRE(DP.MaxMaxNorm() <= tol);
         REQUIRE(PP.MaxMaxNorm() <= tol);
         Vector constant(dof), projected(dof);
         constant = 1.0;
         P.Mult(constant, projected);
         projected -= constant;
         REQUIRE(projected.Normlinf() <= tol);
         // Independently check preservation of the physical cell integral.
         DenseMatrix mass;
         MassIntegrator().AssembleElementMatrix(fe, Tr, mass);
         Vector data(dof), delta(dof), mass_delta(dof);
         for (int j = 0; j < dof; j++) { data(j) = (j+1)*(j+1); }
         P.Mult(data, delta);
         delta -= data;
         mass.Mult(delta, mass_delta);
         REQUIRE(std::abs(mass_delta.Sum()) <= tol*data.Normlinf());
      }
   }
}

TEST_CASE("3D affine LDF projectors are mass-orthogonal and divergence-free",
          "[MHD][LDF]")
{
   for (auto geometry : {Element::TETRAHEDRON, Element::HEXAHEDRON, Element::WEDGE})
      for (int order = 0; order <= 3; order++)
      {
         Mesh mesh = Mesh::MakeCartesian3D(1, 1, 1, geometry);
         // Include a shear: a diagonal Cartesian Jacobian alone would miss
         // mistakes in the coupling of physical derivative directions.
         mesh.Transform([](const Vector &x, Vector &y)
         {
            y.SetSize(3);
            y(0) = 1.2*x(0)+0.3*x(1)-0.1*x(2);
            y(1) = 0.2*x(0)+0.9*x(1)+0.2*x(2);
            y(2) = 0.1*x(0)-0.2*x(1)+1.1*x(2);
         });
         DG_FECollection fec(order, 3, BasisType::GaussLobatto);
         FiniteElementSpace fes(&mesh, &fec, 8, Ordering::byNODES);
         IdealMHDFlux flux(3, 5.0/3.0);
         RusanovFlux numerical_flux(flux);
         GPsource source(3);
         RusanovGPsource numerical_source(source);
         LDF_DG_GP_MHD op(fes,
                          std::unique_ptr<HyperbolicFormIntegrator>(
                             new MHDFormIntegrator(numerical_flux)),
                          std::unique_ptr<GPsourceFormIntegrator>(
                             new GPsourceFormIntegrator(numerical_source)),
                          true, true);
         for (int k = 0; k < fes.GetNE(); k++)
         {
            const FiniteElement &fe = *fes.GetFE(k);
            ElementTransformation &Tr = *fes.GetElementTransformation(k);
            const int dof = fe.GetDof();
            const DenseMatrix &P = op.GetLDFProjector(k);
            DenseMatrix D = AssembleMagneticDivergence(fe, Tr);
            DenseMatrix DP(D.Height(), P.Width()), PP(P.Height());
            mfem::Mult(D, P, DP);
            mfem::Mult(P, P, PP);
            PP -= P;
            const real_t tol = 1024*std::numeric_limits<real_t>::epsilon()*P.Height();
            CAPTURE(geometry, order, k);
            REQUIRE(DP.MaxMaxNorm() <= tol*std::max(1_r, D.MaxMaxNorm()));
            REQUIRE(PP.MaxMaxNorm() <= tol);
            DenseMatrix mass;
            MassIntegrator().AssembleElementMatrix(fe, Tr, mass);
            DenseMatrix HP(P.Height());
            HP = 0.0;
            for (int c = 0; c < 3; c++)
               for (int i = 0; i < dof; i++)
                  for (int j = 0; j < P.Width(); j++)
                     for (int l = 0; l < dof; l++)
                     {
                        HP(c*dof+i,j) += mass(i,l)*P(c*dof+l,j);
                     }
            real_t asymmetry = 0.0;
            for (int i = 0; i < HP.Height(); i++)
               for (int j = 0; j < i; j++)
               {
                  asymmetry = std::max(asymmetry, std::abs(HP(i,j)-HP(j,i)));
               }
            REQUIRE(asymmetry <= tol*mass.MaxMaxNorm());
            for (int c = 0; c < 3; c++)
            {
               Vector constant(P.Width()), projected(P.Height());
               constant = 0.0;
               for (int j = 0; j < dof; j++) { constant(c*dof+j) = 1.0; }
               P.Mult(constant, projected);
               projected -= constant;
               REQUIRE(projected.Normlinf() <= tol);
            }
            // Check strong divergence at independent quadrature points,
            // rather than only against the weak constraints used to build P.
            Vector data(P.Width()), projected(P.Height());
            for (int j = 0; j < data.Size(); j++) { data(j) = std::sin(real_t(j+1)); }
            P.Mult(data, projected);
            DenseMatrix grad(dof, 3);
            const IntegrationRule &ir = IntRules.Get(fe.GetGeomType(), 2*order+3);
            for (int q = 0; q < ir.GetNPoints(); q++)
            {
               Tr.SetIntPoint(&ir.IntPoint(q));
               fe.CalcPhysDShape(Tr, grad);
               real_t divergence = 0.0;
               for (int c = 0; c < 3; c++)
                  for (int j = 0; j < dof; j++)
                  {
                     divergence += projected(c*dof+j)*grad(j,c);
                  }
               REQUIRE(std::abs(divergence) <= tol*std::max(1_r, grad.MaxMaxNorm()));
            }
         }
      }
}

TEST_CASE("MHD default quadrature uses cell degree 2k and face degree 2k+1",
          "[MHD]")
{
   for (auto geometry : {Element::TRIANGLE, Element::TETRAHEDRON, Element::WEDGE})
   {
      const int dim = geometry == Element::TRIANGLE ? 2 : 3;
      Mesh mesh = dim == 2 ? Mesh::MakeCartesian2D(1, 1, geometry) :
                  Mesh::MakeCartesian3D(1, 1, 2, geometry);
      const int order = 2;
      DG_FECollection fec(order, dim, BasisType::GaussLobatto);
      FiniteElementSpace fes(&mesh, &fec, 8, Ordering::byNODES);
      auto exact = MHDInitialCondition(2, 5.0/3.0, dim);
      GridFunction state(&fes);
      state.ProjectCoefficient(exact);
      IdealMHDFlux flux(dim, 5.0/3.0, 3);
      RusanovFlux numerical_flux(flux);
      GPsource source(dim, 3);
      RusanovGPsource numerical_source(source);
      MHDFormIntegrator default_flux(numerical_flux), reference_flux(numerical_flux);
      GPsourceFormIntegrator default_gp(numerical_source),
                             reference_gp(numerical_source);
      Array<int> dofs;
      Vector local, actual, expected;
      const FiniteElement &fe = *fes.GetFE(0);
      ElementTransformation &Tr = *fes.GetElementTransformation(0);
      fes.GetElementVDofs(0, dofs);
      state.GetSubVector(dofs, local);
      const IntegrationRule &cell_rule = IntRules.Get(fe.GetGeomType(), 2*order);
      reference_flux.SetIntRule(&cell_rule);
      reference_gp.SetIntRule(&cell_rule);
      default_flux.AssembleElementVector(fe, Tr, local, actual);
      reference_flux.AssembleElementVector(fe, Tr, local, expected);
      actual -= expected;
      REQUIRE(actual.Normlinf() < 64*std::numeric_limits<real_t>::epsilon());
      default_gp.AssembleElementVector(fe, Tr, local, actual);
      reference_gp.AssembleElementVector(fe, Tr, local, expected);
      actual -= expected;
      REQUIRE(actual.Normlinf() < 64*std::numeric_limits<real_t>::epsilon());
      for (int f = 0; f < mesh.GetNumFaces(); f++)
      {
         FaceElementTransformations *face = mesh.GetInteriorFaceTransformations(f);
         if (!face) { continue; }
         const FiniteElement &left = *fes.GetFE(face->Elem1No);
         const FiniteElement &right = *fes.GetFE(face->Elem2No);
         Array<int> right_dofs;
         fes.GetElementVDofs(face->Elem1No, dofs);
         fes.GetElementVDofs(face->Elem2No, right_dofs);
         dofs.Append(right_dofs);
         state.GetSubVector(dofs, local);
         const IntegrationRule &face_rule = IntRules.Get(face->GetGeometryType(),
                                                         2*order+1);
         reference_flux.SetIntRule(&face_rule);
         reference_gp.SetIntRule(&face_rule);
         default_flux.AssembleFaceVector(left, right, *face, local, actual);
         reference_flux.AssembleFaceVector(left, right, *face, local, expected);
         actual -= expected;
         REQUIRE(actual.Normlinf() < 64*std::numeric_limits<real_t>::epsilon());
         default_gp.AssembleFaceVector(left, right, *face, local, actual);
         reference_gp.AssembleFaceVector(left, right, *face, local, expected);
         actual -= expected;
         REQUIRE(actual.Normlinf() < 64*std::numeric_limits<real_t>::epsilon());
      }
      // A face call must not replace the default cell quadrature rule.
      fes.GetElementVDofs(0, dofs);
      state.GetSubVector(dofs, local);
      default_flux.AssembleElementVector(fe, Tr, local, actual);
      reference_flux.SetIntRule(&cell_rule);
      reference_flux.AssembleElementVector(fe, Tr, local, expected);
      actual -= expected;
      REQUIRE(actual.Normlinf() < 64*std::numeric_limits<real_t>::epsilon());
   }
}

TEST_CASE("Generated Alfven prism mesh is fully periodic", "[MHD]")
{
   Mesh mesh = MHDMesh(2, 3, false, true);
   for (int k = 0; k < mesh.GetNE(); k++)
   {
      REQUIRE(mesh.GetElementBaseGeometry(k) == Geometry::PRISM);
   }
   for (int f = 0; f < mesh.GetNumFaces(); f++)
   {
      int first, second;
      mesh.GetFaceElements(f, &first, &second);
      REQUIRE(first >= 0);
      REQUIRE(second >= 0);
   }
}

TEST_CASE("MHD normal wave speeds include transverse fields without scratch vectors",
          "[MHD]")
{
   for (int dim = 1; dim <= 3; dim++)
      for (int components = dim; components <= 3; components++)
         for (bool parallel : {false, true})
         {
            const real_t rho = 1.3, pressure = 0.8, gamma = 5.0/3.0;
            IdealMHDFlux flux(dim, gamma, components);
            FaceElementTransformations Tr;
            Vector normal(dim), momentum(components), magnetic(components);
            for (int d = 0; d < dim; d++) { normal(d) = 0.7+0.2*d; }
            for (int d = 0; d < components; d++)
            {
               momentum(d) = -0.2+0.1*d;
               magnetic(d) = parallel ? (d < dim ? 0.4*normal(d) : 0_r) :
                             0.3-0.2*d;
            }
            Vector state(2*components+2), flux_normal(state.Size());
            state(0) = rho;
            for (int d = 0; d < components; d++)
            {
               state(1+d) = momentum(d);
               state(1+components+d) = magnetic(d);
            }
            state(state.Size()-1) = pressure/(gamma-1)+
                                    0.5*(momentum*momentum)/rho+0.5*(magnetic*magnetic);
            const real_t speed = flux.ComputeFluxDotN(state, normal, Tr, flux_normal);
            // Independent reference: explicitly form tangential B as a vector.
            Vector tangent(magnetic);
            real_t Bn = 0.0, mn = 0.0;
            for (int d = 0; d < dim; d++)
            {
               Bn += magnetic(d)*normal(d);
               mn += momentum(d)*normal(d);
            }
            for (int d = 0; d < dim; d++) { tangent(d) -= Bn/(normal*normal)*normal(d); }
            const real_t gp = gamma*pressure, B2 = magnetic*magnetic;
            const real_t reference = std::abs(mn)/(rho*normal.Norml2())+
                                     std::sqrt((gp+B2+std::sqrt((gp-B2)*(gp-B2)+
                                                                4*gp*(tangent*tangent)))/(2*rho));
            CAPTURE(dim, components, parallel);
            REQUIRE(std::abs(speed-reference) <=
                    64*std::numeric_limits<real_t>::epsilon()*reference);
            DenseMatrix F(state.Size(), dim);
            flux.ComputeFlux(state, Tr, F);
            Vector contracted(state.Size());
            F.Mult(normal, contracted);
            contracted -= flux_normal;
            REQUIRE(contracted.Normlinf() <=
                    64*std::numeric_limits<real_t>::epsilon()*std::max(1_r,
                                                                       flux_normal.Normlinf()));
         }
}

TEST_CASE("MHD residual supports disabling GP without a source integrator",
          "[MHD]")
{
   Mesh mesh = MHDMesh(2, 1);
   DG_FECollection fec(1, 1, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec, 8, Ordering::byNODES);
   IdealMHDFlux flux(1, 5.0/3.0, 3);
   RusanovFlux numerical_flux(flux);
   LDF_DG_GP_MHD op(fes,
                    std::unique_ptr<HyperbolicFormIntegrator>(new MHDFormIntegrator(
                                                                 numerical_flux)),
                    nullptr, false, false);
   VectorFunctionCoefficient constant(8, [](const Vector &, Vector &u)
   {
      u = 0.0;
      u(0) = 1.0;
      u(4) = 1.0;
      u(7) = 2.0;
   });
   GridFunction state(&fes);
   state.ProjectCoefficient(constant);
   Vector residual(fes.GetVSize());
   op.Mult(state, residual);
   REQUIRE(residual.Normlinf() <= 256*std::numeric_limits<real_t>::epsilon());
   REQUIRE(op.GetMaxCharSpeed() > 0.0);
}

#ifdef MFEM_USE_EXCEPTIONS
TEST_CASE("MHD operator rejects incompatible spaces and integrators", "[MHD]")
{
   Mesh mesh = MHDMesh(2, 1);
   DG_FECollection fec(1, 1, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec, 8, Ordering::byNODES);
   IdealMHDFlux flux(1, 5.0/3.0, 3);
   RusanovFlux numerical_flux(flux);
   auto make_flux_integrator = [&]()
   {
      return std::unique_ptr<HyperbolicFormIntegrator>(
                new MHDFormIntegrator(numerical_flux));
   };

   SECTION("The space must match the flux")
   {
      FiniteElementSpace wrong_fes(&mesh, &fec, 6, Ordering::byNODES);
      REQUIRE_THROWS_AS(LDF_DG_GP_MHD(wrong_fes, make_flux_integrator(),
                                      nullptr, false, false), ErrorException);
   }
   SECTION("The flux integrator is required")
   {
      REQUIRE_THROWS_AS(LDF_DG_GP_MHD(fes, nullptr, nullptr, false, false),
                        ErrorException);
   }
   SECTION("Enabled GP sources require an integrator")
   {
      REQUIRE_THROWS_AS(LDF_DG_GP_MHD(fes, make_flux_integrator(),
                                      nullptr, true, false), ErrorException);
   }
   SECTION("The GP source must match the flux")
   {
      GPsource source(1, 1);
      RusanovGPsource numerical_source(source);
      auto source_integrator = std::unique_ptr<GPsourceFormIntegrator>(
                                  new GPsourceFormIntegrator(numerical_source));
      REQUIRE_THROWS_AS(LDF_DG_GP_MHD(fes, make_flux_integrator(),
                                      std::move(source_integrator), true, false),
                        ErrorException);
   }
}
#endif

TEST_CASE("Rusanov flux retains upwinding for small and larger systems",
          "[MHD]")
{
   class LinearSystemFlux : public FluxFunction
   {
   public:
      LinearSystemFlux(int equations) : FluxFunction(equations, 1) { }
      real_t ComputeFlux(const Vector &state, ElementTransformation &,
                         DenseMatrix &flux) const override
      {
         for (int i = 0; i < num_equations; i++) { flux(i,0) = 2*state(i); }
         return 2.0;
      }
   };
   for (int equations : {1, 6, 8, 11})
   {
      LinearSystemFlux physical_flux(equations);
      RusanovFlux numerical_flux(physical_flux);
      FaceElementTransformations Tr;
      Vector left(equations), right(equations), flux(equations), normal(1);
      for (int i = 0; i < equations; i++)
      {
         left(i) = 0.1*(i+1);
         right(i) = -0.2*(i+1);
      }
      for (real_t direction : {-0.3_r, 0.7_r})
      {
         normal(0) = direction;
         const real_t speed = numerical_flux.Eval(left, right, normal, Tr, flux);
         REQUIRE(speed == 2.0);
         const Vector &upwind = direction > 0 ? left : right;
         for (int i = 0; i < equations; i++)
         {
            REQUIRE(std::abs(flux(i)-2*direction*upwind(i)) <=
                    64*std::numeric_limits<real_t>::epsilon()*std::max(1_r, std::abs(flux(i))));
         }
      }
   }
}

#ifdef MFEM_USE_MPI
TEST_CASE("Distributed MHD residual matches serial GP and LDF assembly",
          "[MHD][Parallel]")
{
   const int ranks = Mpi::WorldSize(), rank = Mpi::WorldRank();
   Mesh mesh = MHDMesh(2, 2);
   REQUIRE(ranks <= mesh.GetNE());
   Array<int> partition(mesh.GetNE());
   for (int k = 0; k < mesh.GetNE(); k++) { partition[k] = k*ranks/mesh.GetNE(); }
   ParMesh pmesh(MPI_COMM_WORLD, mesh, partition.GetData());
   DG_FECollection fec(2, 2, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec, 8, Ordering::byNODES);
   ParFiniteElementSpace pfes(&pmesh, &fec, 8, Ordering::byNODES);
   IdealMHDFlux flux(2, 5.0/3.0, 3);
   RusanovFlux numerical_flux(flux);
   GPsource source(2, 3);
   RusanovGPsource numerical_source(source);
   auto make_operator = [&](FiniteElementSpace &space)
   {
      return std::unique_ptr<LDF_DG_GP_MHD>(new LDF_DG_GP_MHD(space,
                                                              std::unique_ptr<HyperbolicFormIntegrator>(new MHDFormIntegrator(
                                                                       numerical_flux)),
                                                              std::unique_ptr<GPsourceFormIntegrator>(new GPsourceFormIntegrator(
                                                                       numerical_source)),
                                                              true, true));
   };
   auto serial_op = make_operator(fes);
   auto parallel_op = make_operator(pfes);
   auto initial = MHDInitialCondition(2, 5.0/3.0, 2);
   GridFunction state(&fes);
   ParGridFunction pstate(&pfes);
   state.ProjectCoefficient(initial);
   pstate.ProjectCoefficient(initial);
   serial_op->ApplyLDF(state);
   parallel_op->ApplyLDF(pstate);
   Vector residual(fes.GetVSize()), presmult(pfes.GetTrueVSize());
   serial_op->Mult(state, residual);
   parallel_op->Mult(pstate, presmult);
   real_t difference = 0.0;
   int local_element = 0;
   for (int k = 0; k < mesh.GetNE(); k++)
   {
      if (partition[k] != rank) { continue; }
      Array<int> dofs, pdofs;
      fes.GetElementVDofs(k, dofs);
      pfes.GetElementVDofs(local_element++, pdofs);
      REQUIRE(dofs.Size() == pdofs.Size());
      for (int j = 0; j < dofs.Size(); j++)
      {
         difference = std::max(difference,
                               std::abs(residual(dofs[j])-presmult(pdofs[j])));
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &difference, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MAX, MPI_COMM_WORLD);
   real_t speed = parallel_op->GetMaxCharSpeed();
   MPI_Allreduce(MPI_IN_PLACE, &speed, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MAX, MPI_COMM_WORLD);
   CAPTURE(ranks, difference);
   REQUIRE(difference <= 1e-11);
   REQUIRE(std::abs(speed-serial_op->GetMaxCharSpeed()) <= 1e-12);
}
#endif
