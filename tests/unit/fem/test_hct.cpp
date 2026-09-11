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
