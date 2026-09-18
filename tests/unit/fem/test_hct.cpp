// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.

#include "mfem.hpp"
#include "unit_tests.hpp"
#include <cmath>
#include <memory>

using namespace mfem;

namespace
{

constexpr real_t vertices[3][2] = {{0.0,0.0}, {1.0,0.0}, {0.0,1.0}};
constexpr int edge_vertices[3][2] = {{0,1}, {1,2}, {2,0}};

} // namespace

TEST_CASE("HCT triangle degrees of freedom", "[FiniteElement][HCT]")
{
   HCTTriangleFiniteElement fe;
   Vector shape(12);
   DenseMatrix dshape(12, 2);
   for (int vertex = 0; vertex < 3; vertex++)
   {
      IntegrationPoint ip;
      ip.Set2(vertices[vertex][0], vertices[vertex][1]);
      fe.CalcShape(ip, shape);
      fe.CalcDShape(ip, dshape);
      for (int i = 0; i < 12; i++)
      {
         REQUIRE(shape(i) == MFEM_Approx(i == 3*vertex ? 1.0 : 0.0));
         REQUIRE(dshape(i,0) ==
                 MFEM_Approx(i == 3*vertex + 1 ? 1.0 : 0.0));
         REQUIRE(dshape(i,1) ==
                 MFEM_Approx(i == 3*vertex + 2 ? 1.0 : 0.0));
      }
   }
   for (int edge = 0; edge < 3; edge++)
   {
      const int v0 = edge_vertices[edge][0];
      const int v1 = edge_vertices[edge][1];
      const real_t tx = vertices[v1][0] - vertices[v0][0];
      const real_t ty = vertices[v1][1] - vertices[v0][1];
      IntegrationPoint ip;
      ip.Set2(0.5*(vertices[v0][0] + vertices[v1][0]),
              0.5*(vertices[v0][1] + vertices[v1][1]));
      fe.CalcDShape(ip, dshape);
      for (int i = 0; i < 12; i++)
      {
         const real_t normal_derivative =
            ty*dshape(i,0) - tx*dshape(i,1);
         REQUIRE(normal_derivative ==
                 MFEM_Approx(i == 9 + edge ? 1.0 : 0.0));
      }
   }

   HCT_FECollection collection;
   REQUIRE(collection.DofForGeometry(Geometry::POINT) == 3);
   REQUIRE(collection.DofForGeometry(Geometry::SEGMENT) == 1);
   REQUIRE(collection.DofForGeometry(Geometry::TRIANGLE) == 0);
   std::unique_ptr<FiniteElementCollection> copy(
      FiniteElementCollection::New(collection.Name()));
   REQUIRE(copy != nullptr);
}

TEST_CASE("Reduced HCT triangle basis", "[FiniteElement][HCT][ReducedHCT]")
{
   ReducedHCTTriangleFiniteElement fe;
   REQUIRE(fe.GetDof() == 9);
   REQUIRE(fe.GetOrder() == 3);
   REQUIRE(fe.GetIntegrationPartition() ==
           FiniteElement::IntegrationPartition::ALFELD);

   IsoparametricTransformation T;
   T.SetIdentityTransformation(Geometry::TRIANGLE);
   const int geometry = GENERATE(0, 1, 2);
   if (geometry != 0)
   {
      DenseMatrix points(2, 3);
      points(0,0) = 0.2; points(1,0) = -0.3;
      points(0,1) = 1.6; points(1,1) = 0.1;
      points(0,2) = 0.5; points(1,2) = 0.8;
      if (geometry == 2)
      {
         std::swap(points(0,1), points(0,2));
         std::swap(points(1,1), points(1,2));
      }
      T.SetPointMat(points);
   }

   Vector shape(9), reference_shape(9);
   DenseMatrix dshape(9, 2), hessian(9, 3);
   DenseMatrix reference_dshape(9, 2), reference_hessian(9, 3);
   for (int vertex = 0; vertex < 3; vertex++)
   {
      IntegrationPoint ip;
      ip.Set2(vertices[vertex][0], vertices[vertex][1]);
      T.SetIntPoint(&ip);
      fe.CalcPhysShape(T, shape);
      fe.CalcPhysDShape(T, dshape);
      for (int i = 0; i < 9; i++)
      {
         REQUIRE(shape(i) == MFEM_Approx(i == 3*vertex ? 1.0 : 0.0));
         REQUIRE(dshape(i,0) ==
                 MFEM_Approx(i == 3*vertex + 1 ? 1.0 : 0.0));
         REQUIRE(dshape(i,1) ==
                 MFEM_Approx(i == 3*vertex + 2 ? 1.0 : 0.0));
      }
   }

   // Every basis function has a linear normal derivative on physical edges.
   for (int edge = 0; edge < 3; edge++)
   {
      const int v0 = edge_vertices[edge][0];
      const int v1 = edge_vertices[edge][1];
      const DenseMatrix &points = T.GetPointMat();
      const real_t nx = points(1,v1) - points(1,v0);
      const real_t ny = points(0,v0) - points(0,v1);
      for (const real_t t : {0.2, 0.5, 0.8})
      {
         IntegrationPoint ip;
         ip.Set2((1-t)*vertices[v0][0] + t*vertices[v1][0],
                 (1-t)*vertices[v0][1] + t*vertices[v1][1]);
         T.SetIntPoint(&ip);
         fe.CalcPhysDShape(T, dshape);
         for (int i = 0; i < 9; i++)
         {
            const real_t expected =
               (1-t)*(nx*(i == 3*v0+1) + ny*(i == 3*v0+2)) +
               t*(nx*(i == 3*v1+1) + ny*(i == 3*v1+2));
            REQUIRE(nx*dshape(i,0) + ny*dshape(i,1) == MFEM_Approx(expected));
         }
      }
   }

   // Reproduce a complete basis of physical P2, including both derivatives.
   const IntegrationRule &base = IntRules.Get(Geometry::TRIANGLE, 4);
   std::unique_ptr<IntegrationRule> rule(base.ApplyToTriangleAlfeldSplit());
   for (int px = 0; px <= 2; px++)
   {
      for (int py = 0; py <= 2-px; py++)
      {
         auto monomial = [px, py](real_t x, real_t y, int dx, int dy)
         {
            if (px < dx || py < dy) { return real_t(0); }
            real_t value = std::pow(x, px-dx)*std::pow(y, py-dy);
            for (int i = 0; i < dx; i++) { value *= px-i; }
            for (int i = 0; i < dy; i++) { value *= py-i; }
            return value;
         };
         Vector dofs(9), point(2), gradient(2), second(3);
         for (int vertex = 0; vertex < 3; vertex++)
         {
            T.Transform(fe.GetNodes().IntPoint(3*vertex), point);
            dofs(3*vertex) = monomial(point(0), point(1), 0, 0);
            dofs(3*vertex+1) = monomial(point(0), point(1), 1, 0);
            dofs(3*vertex+2) = monomial(point(0), point(1), 0, 1);
         }
         for (int q = 0; q < rule->GetNPoints(); q++)
         {
            const IntegrationPoint &ip = rule->IntPoint(q);
            T.SetIntPoint(&ip);
            T.Transform(ip, point);
            fe.CalcPhysShape(T, shape);
            fe.CalcPhysDShape(T, dshape);
            fe.CalcPhysHessian(T, hessian);
            if (geometry == 0)
            {
               fe.CalcShape(ip, reference_shape);
               fe.CalcDShape(ip, reference_dshape);
               fe.CalcHessian(ip, reference_hessian);
               reference_shape -= shape;
               reference_dshape -= dshape;
               reference_hessian -= hessian;
               REQUIRE(reference_shape.Normlinf() < 1e-11);
               REQUIRE(reference_dshape.MaxMaxNorm() < 1e-11);
               REQUIRE(reference_hessian.MaxMaxNorm() < 1e-11);
            }
            dshape.MultTranspose(dofs, gradient);
            hessian.MultTranspose(dofs, second);
            REQUIRE(shape*dofs == MFEM_Approx(monomial(point(0), point(1), 0, 0)));
            REQUIRE(gradient(0) == MFEM_Approx(monomial(point(0), point(1), 1, 0)));
            REQUIRE(gradient(1) == MFEM_Approx(monomial(point(0), point(1), 0, 1)));
            REQUIRE(second(0) == MFEM_Approx(monomial(point(0), point(1), 2, 0)));
            REQUIRE(second(1) == MFEM_Approx(monomial(point(0), point(1), 1, 1)));
            REQUIRE(second(2) == MFEM_Approx(monomial(point(0), point(1), 0, 2)));
         }
      }
   }
}

TEST_CASE("Reduced HCT collection and continuity",
          "[FiniteElement][HCT][ReducedHCT]")
{
   ReducedHCT_FECollection collection;
   REQUIRE(collection.DofForGeometry(Geometry::POINT) == 3);
   REQUIRE(collection.DofForGeometry(Geometry::SEGMENT) == 0);
   REQUIRE(collection.DofForGeometry(Geometry::TRIANGLE) == 0);
   std::unique_ptr<FiniteElementCollection> copy(
      FiniteElementCollection::New(collection.Name()));
   REQUIRE(dynamic_cast<ReducedHCT_FECollection *>(copy.get()) != nullptr);

   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::TRIANGLE, true);
   for (int vertex = 0; vertex < mesh.GetNV(); vertex++)
   {
      real_t *point = mesh.GetVertex(vertex);
      const real_t x = point[0], y = point[1];
      point[0] = 0.2 + 1.4*x + 0.3*y;
      point[1] = -0.1 + 0.2*x + 0.8*y;
   }
   FiniteElementSpace fes(&mesh, &collection);
   REQUIRE(fes.GetVSize() == 3*mesh.GetNV());
   Vector coefficients(fes.GetVSize());
   for (int i = 0; i < coefficients.Size(); i++)
   { coefficients(i) = std::sin(real_t(i+1)); }

   const IntegrationRule &rule = IntRules.Get(Geometry::SEGMENT, 6);
   for (int face = 0; face < mesh.GetNumFaces(); face++)
   {
      FaceElementTransformations *T = mesh.GetInteriorFaceTransformations(face);
      if (!T) { continue; }
      Array<int> dofs1, dofs2;
      fes.GetElementDofs(T->Elem1No, dofs1);
      fes.GetElementDofs(T->Elem2No, dofs2);
      Vector local1, local2, shape1(9), shape2(9), grad1(2), grad2(2);
      coefficients.GetSubVector(dofs1, local1);
      coefficients.GetSubVector(dofs2, local2);
      DenseMatrix dshape1(9, 2), dshape2(9, 2);
      for (int q = 0; q < rule.GetNPoints(); q++)
      {
         T->SetAllIntPoints(&rule.IntPoint(q));
         fes.GetFE(T->Elem1No)->CalcPhysShape(*T->Elem1, shape1);
         fes.GetFE(T->Elem2No)->CalcPhysShape(*T->Elem2, shape2);
         fes.GetFE(T->Elem1No)->CalcPhysDShape(*T->Elem1, dshape1);
         fes.GetFE(T->Elem2No)->CalcPhysDShape(*T->Elem2, dshape2);
         REQUIRE(shape1*local1 == MFEM_Approx(shape2*local2));
         dshape1.MultTranspose(local1, grad1);
         dshape2.MultTranspose(local2, grad2);
         grad1 -= grad2;
         REQUIRE(grad1.Normlinf() < 1e-10);
      }
   }
}

TEST_CASE("HCT to Johnson-Mercier Airy interpolation",
          "[DiscreteInterpolator][AiryInterpolator]")
{
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::TRIANGLE, true);
   for (int vertex = 0; vertex < mesh.GetNV(); vertex++)
   {
      real_t *point = mesh.GetVertex(vertex);
      const real_t x = point[0];
      const real_t y = point[1];
      point[0] = 0.2 + 1.4*x + 0.3*y;
      point[1] = -0.1 + 0.2*x + 0.8*y;
   }
   HCT_FECollection hct_collection;
   JohnsonMercierFECollection jm_collection;
   FiniteElementSpace hct_space(&mesh, &hct_collection);
   FiniteElementSpace jm_space(&mesh, &jm_collection);
   DiscreteLinearOperator airy(&hct_space, &jm_space);
   airy.AddDomainInterpolator(new AiryInterpolator);
   airy.Assemble();
   airy.Finalize();

   Vector potential(hct_space.GetVSize()), stress(jm_space.GetVSize());
   for (int i = 0; i < potential.Size(); i++)
   {
      potential(i) = std::sin(real_t(i + 1));
   }
   airy.Mult(potential, stress);

   const IntegrationRule &base = IntRules.Get(Geometry::TRIANGLE, 2);
   std::unique_ptr<IntegrationRule> rule(base.ApplyToTriangleAlfeldSplit());
   real_t error = 0.0;
   real_t divergence = 0.0;
   for (int element = 0; element < mesh.GetNE(); element++)
   {
      Array<int> hct_dofs, jm_dofs;
      hct_space.GetElementDofs(element, hct_dofs);
      jm_space.GetElementDofs(element, jm_dofs);
      Vector local_potential, local_stress;
      potential.GetSubVector(hct_dofs, local_potential);
      stress.GetSubVector(jm_dofs, local_stress);
      ElementTransformation *T = mesh.GetElementTransformation(element);
      DenseMatrix hessian(12, 3), divshape(15, 2);
      DenseTensor matrix_shape(2, 2, 15);
      for (int q = 0; q < rule->GetNPoints(); q++)
      {
         const IntegrationPoint &ip = rule->IntPoint(q);
         T->SetIntPoint(&ip);
         hct_space.GetFE(element)->CalcPhysHessian(*T, hessian);
         jm_space.GetFE(element)->CalcMShape(*T, matrix_shape);
         jm_space.GetFE(element)->CalcPhysDivShape(*T, divshape);
         real_t computed[3] = {0.0, 0.0, 0.0};
         real_t div[2] = {0.0, 0.0};
         for (int i = 0; i < 15; i++)
         {
            computed[0] += local_stress(i)*matrix_shape(0,0,i);
            computed[1] += local_stress(i)*matrix_shape(0,1,i);
            computed[2] += local_stress(i)*matrix_shape(1,1,i);
            div[0] += local_stress(i)*divshape(i,0);
            div[1] += local_stress(i)*divshape(i,1);
         }
         real_t expected[3] = {0.0, 0.0, 0.0};
         for (int i = 0; i < 12; i++)
         {
            expected[0] += local_potential(i)*hessian(i,2);
            expected[1] -= local_potential(i)*hessian(i,1);
            expected[2] += local_potential(i)*hessian(i,0);
         }
         for (int i = 0; i < 3; i++)
         {
            error = std::max(error, std::abs(computed[i] - expected[i]));
         }
         divergence = std::max(divergence,
                               std::hypot(div[0], div[1]));
      }
   }
   REQUIRE(error < 1e-9);
   REQUIRE(divergence < 1e-9);
}
