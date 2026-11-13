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
#include "../../../mesh/ncmesh_face_partition.hpp"
#include "mesh_test_utils.hpp"
#include "unit_tests.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <sstream>
#include <vector>
namespace mfem
{

constexpr real_t EPS = 1e-10;

namespace
{

struct RefinedElementSummary
{
   std::array<int, Geometry::NumGeom> geometry_counts = {};
   real_t volume = 0.0;
};

RefinedElementSummary SummarizeRefinedElements(Mesh &mesh)
{
   RefinedElementSummary summary;
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      const Geometry::Type geom = mesh.GetElementBaseGeometry(i);
      summary.geometry_counts[geom]++;
      summary.volume += mesh.GetElementVolume(i);
   }
   return summary;
}

struct FacePatternCase
{
   Geometry::Type geometry;
   Refinement::Pattern pattern;
   int children;
};

constexpr FacePatternCase geometry_specific_2d_patterns[] =
{
   {Geometry::SQUARE, Refinement::Quadrilateral::DIAGONAL_02_2_TRIS, 2},
   {Geometry::SQUARE, Refinement::Quadrilateral::DIAGONAL_13_2_TRIS, 2},
   {Geometry::SQUARE, Refinement::Quadrilateral::DIAGONALS_4_TRIS, 4},
   {Geometry::SQUARE, Refinement::Quadrilateral::EDGE_01_3_TRIS, 3},
   {Geometry::SQUARE, Refinement::Quadrilateral::EDGE_12_3_TRIS, 3},
   {Geometry::SQUARE, Refinement::Quadrilateral::EDGE_23_3_TRIS, 3},
   {Geometry::SQUARE, Refinement::Quadrilateral::EDGE_30_3_TRIS, 3},
   {Geometry::TRIANGLE, Refinement::Triangle::CORNER_0_TRI_QUAD, 2},
   {Geometry::TRIANGLE, Refinement::Triangle::CORNER_1_TRI_QUAD, 2},
   {Geometry::TRIANGLE, Refinement::Triangle::CORNER_2_TRI_QUAD, 2},
   {Geometry::TRIANGLE, Refinement::Triangle::CENTER_3_TRIS, 3},
   {Geometry::TRIANGLE, Refinement::Triangle::CENTER_3_QUADS, 3}
};
constexpr int num_geometry_specific_2d_patterns =
   sizeof(geometry_specific_2d_patterns)/sizeof(FacePatternCase);

// Extract leaf faces on a plane of a refined 3D element, then express them
// in a common pair of face coordinates.
NCFacePartition MakePlanarFacePartition(
   Mesh &mesh, const std::array<real_t, 3> &normal, int u, int v, real_t plane)
{
   NCFacePartition partition;
   for (int elem = 0; elem < mesh.GetNE(); elem++)
   {
      const Element *element = mesh.GetElement(elem);
      const int *vertices = element->GetVertices();
      for (int face = 0; face < element->GetNFaces(); face++)
      {
         const int count = element->GetNFaceVertices(face);
         const int *local = element->GetFaceVertices(face);
         bool on_plane = true;
         for (int i = 0; i < count; i++)
         {
            const real_t *point = mesh.GetVertex(vertices[local[i]]);
            const real_t position = normal[0]*point[0] +
                                    normal[1]*point[1] +
                                    normal[2]*point[2];
            on_plane &= std::abs(position - plane) < EPS;
         }
         if (!on_plane) { continue; }

         NCFacePartition::Cell cell = {};
         cell.geometry = count == 3 ? Geometry::TRIANGLE : Geometry::SQUARE;
         for (int i = 0; i < count; i++)
         {
            const real_t *point = mesh.GetVertex(vertices[local[i]]);
            cell.vertex[i] = { point[u], point[v] };
         }
         partition.cells.Append(cell);
      }
   }
   return partition;
}

NCFacePartition MakeRefinedFacePartition(
   const char *mesh_file, const std::array<real_t, 3> &normal,
   int u, int v, real_t plane, int pattern = 0)
{
   Mesh mesh(mesh_file, 1, 1);
   if (pattern)
   {
      mesh.EnsureNCMesh(true);
      Array<Refinement> refs(1);
      refs[0].Set(0, pattern);
      mesh.GeneralRefinement(refs);
   }
   return MakePlanarFacePartition(mesh, normal, u, v, plane);
}

NCFacePartition MakePatternFacePartition(const Refinement::Pattern &pattern)
{
   const Geometry::Type geometry = pattern.GetParentGeometry();
   Mesh mesh(geometry == Geometry::TRIANGLE ?
             "../../data/ref-triangle.mesh" : "../../data/ref-square.mesh", 1, 1);
   if (pattern.GetId())
   {
      mesh.EnsureNCMesh(true);
      Array<Refinement> refs(1);
      refs[0].Set(0, pattern);
      mesh.GeneralRefinement(refs);
   }
   NCFacePartition partition;
   Array<int> vertices;
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      NCFacePartition::Cell cell = {};
      cell.geometry = mesh.GetElementBaseGeometry(i);
      mesh.GetElementVertices(i, vertices);
      for (int j = 0; j < vertices.Size(); j++)
      {
         const real_t *point = mesh.GetVertex(vertices[j]);
         cell.vertex[j] = { point[0], point[1] };
      }
      partition.cells.Append(cell);
   }
   return partition;
}

real_t IntegrateGridFunction(Mesh &mesh, const GridFunction &field)
{
   real_t integral = 0.0;
   for (int elem = 0; elem < mesh.GetNE(); elem++)
   {
      ElementTransformation &trans = *mesh.GetElementTransformation(elem);
      const IntegrationRule &ir =
         IntRules.Get(mesh.GetElementBaseGeometry(elem), 16);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         const real_t value = field.GetValue(elem, ip);
         trans.SetIntPoint(&ip);
         integral += ip.weight * trans.Weight() * value;
      }
   }
   return integral;
}

Refinement TetRefinement(int element, char type)
{
   if (type == Refinement::Tetrahedron::EDGE_12_2_TETS.GetId())
   {
      return Refinement(element, Refinement::Tetrahedron::EDGE_12_2_TETS);
   }
   if (type == Refinement::Tetrahedron::EDGE_13_2_TETS.GetId())
   {
      return Refinement(element, Refinement::Tetrahedron::EDGE_13_2_TETS);
   }
   if (type == Refinement::Tetrahedron::EDGE_23_2_TETS.GetId())
   {
      return Refinement(element, Refinement::Tetrahedron::EDGE_23_2_TETS);
   }
   return Refinement(element, type);
}

#ifdef MFEM_USE_MPI
Vector IntegrateGridFunctionComponents(Mesh &mesh, const GridFunction &field)
{
   Vector integral(field.FESpace()->GetVDim());
   integral = 0.0;
   Vector value;
   for (int elem = 0; elem < mesh.GetNE(); elem++)
   {
      ElementTransformation &trans = *mesh.GetElementTransformation(elem);
      const IntegrationRule &ir =
         IntRules.Get(mesh.GetElementBaseGeometry(elem), 16);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         field.GetVectorValue(elem, ip, value);
         trans.SetIntPoint(&ip);
         integral.Add(ip.weight * trans.Weight(), value);
      }
   }
   return integral;
}
#endif

// The 2D mesh of DividingPlaneMesh with its interior boundary at x = 0.5,
// refined twice, after which the elements on one side of the interior boundary
// are refined once more. The interior boundary edges on the coarse side are
// then the master edges of the slave edges of the boundary elements on the
// refined side.
Mesh OneSidedRefinedDividingLineMesh(bool tri_mesh)
{
   auto mesh = DividingPlaneMesh(tri_mesh, true, false);
   mesh.EnsureNCMesh(true);
   mesh.UniformRefinement();
   mesh.UniformRefinement();
   Array<int> refs;
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      if (mesh.GetAttribute(e) == 2) { refs.Append(e); }
   }
   mesh.GeneralRefinement(refs);
   return mesh;
}

// Check that the DOFs on the interior boundary of a mesh from
// OneSidedRefinedDividingLineMesh, except those on the exterior boundary, are
// essential if and only if the interior boundary is. The array ess_marker marks
// the essential DOFs, true or local, and the function dof_of returns the DOF of
// a local DOF in ess_marker, or -1 for a DOF not to check (e.g. a constrained
// or non-owned DOF). Returns the number of checked DOFs.
template <typename DofOf>
int CheckInteriorBoundaryEssentialDofs(const Mesh &mesh,
                                       const FiniteElementSpace &fes,
                                       const Array<int> &ess_marker,
                                       bool ess_interior, DofOf &&dof_of)
{
   auto on_interior = [&](int v)
   {
      const real_t *x = mesh.GetVertex(v);
      return std::abs(x[0] - 0.5) < EPS && x[1] > EPS && x[1] < 1.0 - EPS;
   };
   int checked = 0;
   auto check_essential_dofs = [&](const Array<int> &dofs)
   {
      for (int d : dofs)
      {
         const int t = dof_of(UnsignIndex(d));
         if (t < 0) { continue; }
         CHECK((ess_marker[t] != 0) == ess_interior);
         checked++;
      }
   };
   Array<int> vert, dofs;
   for (int e = 0; e < mesh.GetNEdges(); e++)
   {
      mesh.GetEdgeVertices(e, vert);
      const real_t *x0 = mesh.GetVertex(vert[0]), *x1 = mesh.GetVertex(vert[1]);
      if (std::abs(x0[0] - 0.5) > EPS || std::abs(x1[0] - 0.5) > EPS)
      {
         continue;
      }
      fes.GetEdgeInteriorDofs(e, dofs);
      check_essential_dofs(dofs);
   }
   for (int v = 0; v < mesh.GetNV(); v++)
   {
      if (!on_interior(v)) { continue; }
      fes.GetVertexDofs(v, dofs);
      check_essential_dofs(dofs);
   }
   return checked;
}

} // namespace

// Test case: Verify that a conforming mesh yields the same norm for the
//            assembled diagonal with PA when using the standard (conforming)
//            Mesh vs. the corresponding (non-conforming) NCMesh. (note:
//            permutations of the values in the diagonal are expected)
TEST_CASE("NCMesh PA diagonal", "[NCMesh]")
{
   SECTION("Quad mesh")
   {
      int ne = 2;
      Mesh mesh = Mesh::MakeCartesian2D(
                     ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
      Mesh nc_mesh = Mesh::MakeCartesian2D(
                        ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
      nc_mesh.EnsureNCMesh();

      mesh.UniformRefinement();
      nc_mesh.UniformRefinement();

      int dim = 2;
      for (int order = 1; order <= 3; ++order)
      {
         ND_FECollection fec(order, dim);

         FiniteElementSpace fes(&mesh, &fec);
         FiniteElementSpace nc_fes(&nc_mesh, &fec);

         BilinearForm a(&fes);
         BilinearForm nc_a(&nc_fes);

         a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         nc_a.SetAssemblyLevel(AssemblyLevel::PARTIAL);

         ConstantCoefficient coef(1.0);
         a.AddDomainIntegrator(new CurlCurlIntegrator(coef));
         nc_a.AddDomainIntegrator(new CurlCurlIntegrator(coef));

         a.Assemble();
         nc_a.Assemble();

         Vector diag(fes.GetTrueVSize());
         Vector nc_diag(nc_fes.GetTrueVSize());
         a.AssembleDiagonal(diag);
         nc_a.AssembleDiagonal(nc_diag);

         real_t error = fabs(diag.Norml2() - nc_diag.Norml2());
         CAPTURE(order, error);
         REQUIRE(error == MFEM_Approx(0.0, EPS));
      }
   }

   SECTION("Hexa mesh")
   {
      int ne = 2;
      Mesh mesh = Mesh::MakeCartesian3D(
                     ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
      Mesh nc_mesh = Mesh::MakeCartesian3D(
                        ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
      nc_mesh.EnsureNCMesh();

      mesh.UniformRefinement();
      nc_mesh.UniformRefinement();

      int dim = 3;
      for (int order = 1; order <= 3; ++order)
      {
         ND_FECollection fec(order, dim);

         FiniteElementSpace fes(&mesh, &fec);
         FiniteElementSpace nc_fes(&nc_mesh, &fec);

         BilinearForm a(&fes);
         BilinearForm nc_a(&nc_fes);

         a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         nc_a.SetAssemblyLevel(AssemblyLevel::PARTIAL);

         ConstantCoefficient coef(1.0);
         a.AddDomainIntegrator(new CurlCurlIntegrator(coef));
         nc_a.AddDomainIntegrator(new CurlCurlIntegrator(coef));

         a.Assemble();
         nc_a.Assemble();

         Vector diag(fes.GetTrueVSize());
         Vector nc_diag(nc_fes.GetTrueVSize());
         a.AssembleDiagonal(diag);
         nc_a.AssembleDiagonal(nc_diag);

         real_t error = fabs(diag.Sum() - nc_diag.Sum());
         CAPTURE(order, error);
         REQUIRE(error == MFEM_Approx(0.0, EPS));
      }
   }

} // test case

TEST_CASE("NCMesh 3D Refined Volume", "[NCMesh]")
{
   auto mesh_fname = GENERATE("../../data/ref-tetrahedron.mesh",
                              "../../data/ref-cube.mesh",
                              "../../data/ref-prism.mesh",
                              "../../data/ref-pyramid.mesh"
                             );

   auto ref_type = GENERATE(Refinement::X,
                            Refinement::Y,
                            Refinement::Z,
                            Refinement::XY,
                            Refinement::XZ,
                            Refinement::YZ,
                            Refinement::XYZ);

   const real_t scale = GENERATE(0.5, 0.25);  // Only affects hex mesh so far

   if (scale != 0.5 && std::strcmp(mesh_fname, "../../data/ref-cube.mesh") != 0)
   {
      return;
   }

   Mesh mesh(mesh_fname, 1, 1);
   mesh.EnsureNCMesh(true);
   real_t original_volume = mesh.GetElementVolume(0);
   Array<Refinement> ref(1);
   ref[0].Set(0, ref_type, scale);

   mesh.GeneralRefinement(ref, 1);
   real_t summed_volume = 0.0;
   for (int i = 0; i < mesh.GetNE(); ++i)
   {
      summed_volume += mesh.GetElementVolume(i);
   }
   REQUIRE(summed_volume == MFEM_Approx(original_volume));
} // test case

TEST_CASE("NCMesh geometry-specific refinement patterns", "[NCMesh]")
{
   SECTION("pattern descriptors")
   {
      const std::array<Refinement::Pattern, 36> patterns =
      {
         Refinement::Segment::BISECTION,
         Refinement::Triangle::EDGE_01_2_TRIS,
         Refinement::Triangle::EDGE_02_2_TRIS,
         Refinement::Triangle::EDGE_12_2_TRIS,
         Refinement::Triangle::UNIFORM,
         Refinement::Triangle::CORNER_0_TRI_QUAD,
         Refinement::Triangle::CORNER_1_TRI_QUAD,
         Refinement::Triangle::CORNER_2_TRI_QUAD,
         Refinement::Triangle::CENTER_3_TRIS,
         Refinement::Triangle::CENTER_3_QUADS,
         Refinement::Quadrilateral::SPLIT_X,
         Refinement::Quadrilateral::SPLIT_Y,
         Refinement::Quadrilateral::UNIFORM,
         Refinement::Quadrilateral::DIAGONAL_02_2_TRIS,
         Refinement::Quadrilateral::DIAGONAL_13_2_TRIS,
         Refinement::Quadrilateral::DIAGONALS_4_TRIS,
         Refinement::Quadrilateral::EDGE_01_3_TRIS,
         Refinement::Quadrilateral::EDGE_12_3_TRIS,
         Refinement::Quadrilateral::EDGE_23_3_TRIS,
         Refinement::Quadrilateral::EDGE_30_3_TRIS,
         Refinement::Tetrahedron::EDGE_01_2_TETS,
         Refinement::Tetrahedron::EDGE_02_2_TETS,
         Refinement::Tetrahedron::EDGE_03_2_TETS,
         Refinement::Tetrahedron::EDGE_12_2_TETS,
         Refinement::Tetrahedron::UNIFORM,
         Refinement::Hexahedron::SPLIT_X,
         Refinement::Hexahedron::SPLIT_Y,
         Refinement::Hexahedron::SPLIT_Z,
         Refinement::Hexahedron::SPLIT_XY,
         Refinement::Hexahedron::SPLIT_XZ,
         Refinement::Hexahedron::SPLIT_YZ,
         Refinement::Hexahedron::UNIFORM,
         Refinement::Prism::TRIANGLE_UNIFORM,
         Refinement::Prism::SPLIT_Z,
         Refinement::Prism::UNIFORM,
         Refinement::Pyramid::UNIFORM
      };

      Array<Refinement> refs;
      for (const Refinement::Pattern &pattern : patterns)
      {
         refs.Append(Refinement(0, pattern));
      }
      for (int i = 0; i < refs.Size(); i++)
      {
         CHECK(refs[i].HasPattern());
         CHECK(refs[i].GetPatternGeometry() == patterns[i].GetParentGeometry());
         CHECK(refs[i].GetType() == patterns[i].GetId());
      }

      Refinement legacy(0, Refinement::X);
      CHECK_FALSE(legacy.HasPattern());
   }

   SECTION("quadrilateral pattern")
   {
      Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
      mesh.EnsureNCMesh();

      Array<Refinement> ref(1);
      ref[0].Set(0, Refinement::Quadrilateral::SPLIT_Y, 0.25);
      CHECK(ref[0].HasPattern());
      CHECK(ref[0].GetPatternGeometry() == Geometry::SQUARE);
      CHECK(ref[0].GetType() == Refinement::Y);
      CHECK(ref[0].s[0] == 0.0);
      CHECK(ref[0].s[1] == 0.25);

      mesh.GeneralRefinement(ref);
      REQUIRE(mesh.GetNE() == 2);
   }

#ifdef MFEM_USE_EXCEPTIONS
   SECTION("pattern parent geometry is validated")
   {
      Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
      mesh.EnsureNCMesh();

      Array<Refinement> ref(1);
      ref[0].Set(0, Refinement::Triangle::UNIFORM);
      REQUIRE_THROWS(mesh.GeneralRefinement(ref));
   }
#endif
} // test case

TEST_CASE("NCMesh 3D Derefined Volume", "[NCMesh]")
{
   auto mesh_fname = GENERATE("../../data/ref-tetrahedron.mesh",
                              "../../data/ref-cube.mesh",
                              "../../data/ref-prism.mesh",
                              "../../data/ref-pyramid.mesh"
                             );

   auto ref_type = GENERATE(Refinement::X, Refinement::Y, Refinement::Z,
                            Refinement::XY, Refinement::XZ, Refinement::YZ,
                            Refinement::XYZ);

   if (ref_type != Refinement::XYZ &&
       std::strcmp(mesh_fname, "../../data/ref-cube.mesh") != 0 &&
       std::strcmp(mesh_fname, "../../data/ref-prism.mesh") != 0)
   {
      return;
   }

   Mesh mesh(mesh_fname, 1, 1);
   mesh.EnsureNCMesh(true);
   real_t original_volume = mesh.GetElementVolume(0);
   Array<Refinement> ref(1);
   ref[0].Set(0, ref_type);

   mesh.GeneralRefinement(ref, 1);

   Array<real_t> elem_error(mesh.GetNE());
   for (int i = 0; i < mesh.GetNE(); ++i)
   {
      elem_error[i] = 0.0;
   }
   REQUIRE(mesh.DerefineByError(elem_error, 1.0));

   REQUIRE(mesh.GetNE() == 1);
   real_t derefined_volume = mesh.GetElementVolume(0);
   REQUIRE(derefined_volume == MFEM_Approx(original_volume));
} // test case

TEST_CASE("NCMesh mixed pyramid children refine and derefine", "[NCMesh]")
{
   const int order = GENERATE(1, 2, 3, 4);
   const int pyr_type = GENERATE(0, 1);
   CAPTURE(order, pyr_type);
   Mesh mesh("../../data/ref-pyramid.mesh", 1, 1);
   const real_t original_volume = mesh.GetElementVolume(0);
   H1_FECollection fec(order, 3, BasisType::GaussLobatto, pyr_type);
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction field(&fes);
   GridFunction arbitrary(&fes);
   for (int i = 0; i < arbitrary.Size(); i++)
   {
      arbitrary(i) = (i % 11 - 5) / 7.0;
   }
   const Vector original_values(arbitrary);
   FunctionCoefficient affine([](const Vector &x)
   {
      return x[0] - 2.0*x[1] + 3.0*x[2] + 1.0;
   });
   field.ProjectCoefficient(affine);

   mesh.EnsureNCMesh(true);
   Array<Refinement> refinement(1);
   refinement[0].Set(0, Refinement::XYZ);
   mesh.GeneralRefinement(refinement, 1);

   const RefinedElementSummary refined = SummarizeRefinedElements(mesh);
   REQUIRE(mesh.GetNE() == 10);
   REQUIRE(refined.geometry_counts[Geometry::PYRAMID] == 6);
   REQUIRE(refined.geometry_counts[Geometry::TETRAHEDRON] == 4);
   REQUIRE(refined.volume == MFEM_Approx(original_volume));
   REQUIRE(mesh.CheckElementOrientation(false) == 0);
   REQUIRE(mesh.CheckBdrElementOrientation(false) == 0);

   const CoarseFineTransformations &trans = mesh.GetRefinementTransforms();
   const DenseMatrixStack &pmats =
      trans.point_matrices[Geometry::PYRAMID];
   REQUIRE(trans.embeddings.Size() == mesh.GetNE());
   // NCMesh keeps the identity transform at matrix zero.
   REQUIRE(pmats.SizeK() == mesh.GetNE() + 1);
   REQUIRE(pmats.SizeI(0) == 3);
   REQUIRE(pmats.SizeJ(0) == Geometry::NumVerts[Geometry::PYRAMID]);
   for (int child = 0; child < mesh.GetNE(); child++)
   {
      const Geometry::Type child_geom = mesh.GetElementBaseGeometry(child);
      const Embedding &embedding = trans.embeddings[child];
      CAPTURE(child);
      REQUIRE(embedding.parent == 0);
      REQUIRE(embedding.geom == Geometry::PYRAMID);
      REQUIRE(embedding.matrix == child + 1);
      REQUIRE(pmats.SizeI(embedding.matrix) == 3);
      REQUIRE(pmats.SizeJ(embedding.matrix) == Geometry::NumVerts[child_geom]);
   }

   fes.Update();
   field.Update();
   arbitrary.Update();
   {
      GridFunction exact(&fes);
      GridFunction error(&fes);
      exact.ProjectCoefficient(affine);
      error = field;
      error -= exact;
      REQUIRE(error.Normlinf() == MFEM_Approx(0.0, EPS));
   }

   Array<real_t> elem_error(mesh.GetNE());
   elem_error = 0.0;
   REQUIRE(mesh.DerefineByError(elem_error, 1.0));

   const RefinedElementSummary derefined = SummarizeRefinedElements(mesh);
   REQUIRE(mesh.GetNE() == 1);
   REQUIRE(derefined.geometry_counts[Geometry::PYRAMID] == 1);
   REQUIRE(derefined.geometry_counts[Geometry::TETRAHEDRON] == 0);
   REQUIRE(derefined.volume == MFEM_Approx(original_volume));
   REQUIRE(mesh.CheckElementOrientation(false) == 0);
   REQUIRE(mesh.CheckBdrElementOrientation(false) == 0);

   fes.Update();
   field.Update();
   arbitrary.Update();
   GridFunction exact(&fes);
   GridFunction error(&fes);
   exact.ProjectCoefficient(affine);
   error = field;
   error -= exact;
   REQUIRE(error.Normlinf() == MFEM_Approx(0.0, EPS));
   if (order <= 2)
   {
      arbitrary -= original_values;
      REQUIRE(arbitrary.Normlinf() == MFEM_Approx(0.0, EPS));
   }
}

TEST_CASE("NCMesh legacy quad and hex pattern save and reload", "[NCMesh]")
{
   const bool hex = GENERATE(false, true);
   const int ref_type = GENERATE(Refinement::X, Refinement::Y,
                                 Refinement::XY, Refinement::Z,
                                 Refinement::XZ, Refinement::YZ,
                                 Refinement::XYZ);
   if (!hex && ref_type > Refinement::XY) { return; }
   CAPTURE(hex, ref_type);

   Mesh mesh(hex ? "../../data/ref-cube.mesh" : "../../data/ref-square.mesh",
             1, 1);
   const real_t original_volume = mesh.GetElementVolume(0);
   mesh.EnsureNCMesh(true);
   Array<Refinement> refs(1);
   refs[0].Set(0, ref_type);
   mesh.GeneralRefinement(refs);
   const int num_children = mesh.GetNE();

   std::stringstream serialized;
   mesh.Print(serialized);
   REQUIRE(serialized.str().find("MFEM NC mesh v1.0") == 0);
   Mesh reloaded(serialized, 1, 1);
   REQUIRE(reloaded.GetNE() == num_children);
   REQUIRE(SummarizeRefinedElements(reloaded).volume ==
           MFEM_Approx(original_volume));
   REQUIRE(reloaded.CheckElementOrientation(false) == 0);

   Vector errors(reloaded.GetNE());
   errors = 0.0;
   REQUIRE(reloaded.DerefineByError(errors, 1.0));
   REQUIRE(reloaded.GetNE() == 1);
   REQUIRE(reloaded.GetElementVolume(0) == MFEM_Approx(original_volume));
}

TEST_CASE("NCMesh geometry-specific face partition relations", "[NCMesh]")
{
   for (Geometry::Type geometry : {Geometry::TRIANGLE, Geometry::SQUARE})
   {
      std::vector<Refinement::Pattern> patterns =
      {
         {geometry, 0}, {geometry, Refinement::X}, {geometry, Refinement::Y},
         {geometry, Refinement::XY}
      };
      if (geometry == Geometry::TRIANGLE)
      { patterns.push_back({geometry, Refinement::Z}); }
      for (const FacePatternCase &test : geometry_specific_2d_patterns)
      {
         if (test.geometry == geometry) { patterns.push_back(test.pattern); }
      }

      std::vector<NCFacePartition> partitions;
      for (const Refinement::Pattern &pattern : patterns)
      { partitions.push_back(MakePatternFacePartition(pattern)); }

      for (size_t i = 0; i < patterns.size(); i++)
      {
         CAPTURE(geometry, patterns[i].GetId());
         REQUIRE(CompareNCFacePartitions(partitions[i], partitions[i]) ==
                 NCFaceRelation::Matching);
         REQUIRE(CompareNCFacePartitions(partitions[0], partitions[i]) ==
                 (i == 0 ? NCFaceRelation::Matching :
                  NCFaceRelation::SecondRefinesFirst));
         for (size_t j = 0; j < patterns.size(); j++)
         {
            const NCFaceRelation forward =
               CompareNCFacePartitions(partitions[i], partitions[j]);
            const NCFaceRelation reverse =
               CompareNCFacePartitions(partitions[j], partitions[i]);
            if (forward == NCFaceRelation::FirstRefinesSecond)
            { REQUIRE(reverse == NCFaceRelation::SecondRefinesFirst); }
            else if (forward == NCFaceRelation::SecondRefinesFirst)
            { REQUIRE(reverse == NCFaceRelation::FirstRefinesSecond); }
            else
            { REQUIRE(reverse == forward); }
         }
      }
   }
}

TEST_CASE("NCMesh face partition comparison with a narrow cell", "[NCMesh]")
{
   const real_t delta = 32*std::numeric_limits<real_t>::epsilon();
   const auto quad = [](real_t left, real_t right)
   {
      NCFacePartition::Cell cell = {};
      cell.geometry = Geometry::SQUARE;
      cell.vertex[0] = {left, 0.0};
      cell.vertex[1] = {right, 0.0};
      cell.vertex[2] = {right, 1.0};
      cell.vertex[3] = {left, 1.0};
      return cell;
   };
   NCFacePartition coarse, fine;
   coarse.cells.Append(quad(0.0, 1.0));
   fine.cells.Append(quad(0.0, 1.0-delta));
   fine.cells.Append(quad(1.0-delta, 1.0));
   REQUIRE(CompareNCFacePartitions(coarse, fine) ==
           NCFaceRelation::SecondRefinesFirst);
}

TEST_CASE("NCMesh three-dimensional face partition relations", "[NCMesh]")
{
   const auto tet = [](int pattern)
   {
      return MakeRefinedFacePartition("../../data/ref-tetrahedron.mesh",
      {1.0, 1.0, 1.0}, 0, 1, 1.0, pattern);
   };
   const auto hex = [](int pattern)
   {
      return MakeRefinedFacePartition("../../data/ref-cube.mesh",
      {0.0, 0.0, 1.0}, 0, 1, 0.0, pattern);
   };
   const auto prism_base = [](int pattern)
   {
      return MakeRefinedFacePartition("../../data/ref-prism.mesh",
      {0.0, 0.0, 1.0}, 0, 1, 0.0, pattern);
   };
   const auto prism_side = [](int pattern)
   {
      return MakeRefinedFacePartition("../../data/ref-prism.mesh",
      {0.0, 1.0, 0.0}, 0, 2, 0.0, pattern);
   };
   const auto pyramid_base = [](int pattern)
   {
      return MakeRefinedFacePartition("../../data/ref-pyramid.mesh",
      {0.0, 0.0, 1.0}, 0, 1, 0.0, pattern);
   };
   const auto pyramid_side = [](int pattern)
   {
      return MakeRefinedFacePartition("../../data/ref-pyramid.mesh",
      {0.0, 1.0, 0.0}, 0, 2, 0.0, pattern);
   };

   const NCFacePartition tet_coarse = tet(0);
   const NCFacePartition tet_x = tet(Refinement::X);
   const NCFacePartition tet_y = tet(Refinement::Y);
   const NCFacePartition tet_iso = tet(Refinement::XYZ);
   REQUIRE(tet_coarse.cells.Size() == 1);
   REQUIRE(tet_x.cells.Size() == 2);
   REQUIRE(tet_iso.cells.Size() == 4);
   REQUIRE(CompareNCFacePartitions(tet_coarse, tet_x) ==
           NCFaceRelation::SecondRefinesFirst);
   REQUIRE(CompareNCFacePartitions(tet_x, tet_y) ==
           NCFaceRelation::Incompatible);
   REQUIRE(CompareNCFacePartitions(tet_y, tet_x) ==
           NCFaceRelation::Incompatible);
   REQUIRE(CompareNCFacePartitions(tet_x, tet_iso) ==
           NCFaceRelation::Incompatible);
   REQUIRE(CompareNCFacePartitions(tet_x, tet_x) ==
           NCFaceRelation::Matching);
   NCFacePartition reversed = tet_x;
   for (auto &cell : reversed.cells)
   { std::swap(cell.vertex[1], cell.vertex[2]); }
   REQUIRE(CompareNCFacePartitions(tet_x, reversed) ==
           NCFaceRelation::Matching);
   REQUIRE(CompareNCFacePartitions(tet_iso, prism_base(Refinement::XY)) ==
           NCFaceRelation::Matching);
   REQUIRE(CompareNCFacePartitions(tet_iso, pyramid_side(Refinement::XYZ)) ==
           NCFaceRelation::Matching);

   const NCFacePartition hex_x = hex(Refinement::X);
   const NCFacePartition hex_y = hex(Refinement::Y);
   const NCFacePartition hex_xy = hex(Refinement::XY);
   REQUIRE(hex_x.cells.Size() == 2);
   REQUIRE(hex_xy.cells.Size() == 4);
   REQUIRE(CompareNCFacePartitions(hex_x, hex_y) ==
           NCFaceRelation::Incompatible);
   REQUIRE(CompareNCFacePartitions(hex_x, hex_xy) ==
           NCFaceRelation::SecondRefinesFirst);
   REQUIRE(CompareNCFacePartitions(hex_xy, hex_x) ==
           NCFaceRelation::FirstRefinesSecond);
   REQUIRE(CompareNCFacePartitions(hex_xy, pyramid_base(Refinement::XYZ)) ==
           NCFaceRelation::Matching);
   REQUIRE(CompareNCFacePartitions(prism_side(Refinement::Z), hex_xy) ==
           NCFaceRelation::SecondRefinesFirst);
}

TEST_CASE("NCMesh geometry-specific H1 field transfer round trip", "[NCMesh]")
{
   FunctionCoefficient affine([](const Vector &x)
   { return 1.0 + 2.0*x[0] - 3.0*x[1]; });

   for (const FacePatternCase &test : geometry_specific_2d_patterns)
   {
      CAPTURE(test.geometry, test.pattern.GetId());
      Mesh mesh = Mesh::MakeCartesian2D(2, 1,
                                        test.geometry == Geometry::SQUARE ?
                                        Element::QUADRILATERAL : Element::TRIANGLE);
      const int original_elements = mesh.GetNE();
      mesh.EnsureNCMesh(true);
      H1_FECollection fec(2, 2);
      FiniteElementSpace fes(&mesh, &fec);
      GridFunction field(&fes);
      field.ProjectCoefficient(affine);

      Array<Refinement> refs(1);
      refs[0].Set(0, test.pattern);
      mesh.GeneralRefinement(refs);
      REQUIRE(mesh.GetNE() == original_elements + test.children - 1);
      REQUIRE(SummarizeRefinedElements(mesh).volume == MFEM_Approx(1.0));
      REQUIRE(mesh.CheckElementOrientation(false) == 0);
      fes.Update();
      field.Update();
      GridFunction exact(&fes);
      exact.ProjectCoefficient(affine);
      exact -= field;
      REQUIRE(exact.Normlinf() == MFEM_Approx(0.0, EPS));

      Vector errors(mesh.GetNE());
      errors = 0.0;
      REQUIRE(mesh.DerefineByError(errors, 1.0));
      REQUIRE(mesh.GetNE() == original_elements);
      fes.Update();
      field.Update();
      exact.Update();
      exact.ProjectCoefficient(affine);
      exact -= field;
      REQUIRE(exact.Normlinf() == MFEM_Approx(0.0, EPS));
   }
}

TEST_CASE("NCMesh mixed 2D attribute pattern round trip", "[NCMesh]")
{
   Mesh mesh("../../data/rings-mixed.mesh", 1, 0);
   REQUIRE(mesh.GetNE() == 21);
   mesh.EnsureNCMesh(true);
   H1_FECollection fec(2, 2);
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction field(&fes);
   FunctionCoefficient affine([](const Vector &x)
   { return 1.0 + 2.0*x[0] - 3.0*x[1]; });
   field.ProjectCoefficient(affine);

   Array<Refinement> refs;
   for (int elem = 0; elem < mesh.GetNE(); elem++)
   {
      const int pattern = mesh.GetElement(elem)->GetAttribute();
      if (pattern != 15)
      { refs.Append(Refinement(elem, static_cast<char>(pattern))); }
   }
   REQUIRE(refs.Size() == 19);
   mesh.GeneralRefinement(refs);
   {
      std::ofstream mesh_ofs("rings-mixed-refined.mesh");
      mesh_ofs.precision(8);
      mesh.Print(mesh_ofs);
   }

   REQUIRE(mesh.GetNE() == 52);
   REQUIRE(mesh.CheckElementOrientation(false) == 0);
   fes.Update();
   field.Update();
   GridFunction difference(&fes);
   difference.ProjectCoefficient(affine);
   difference -= field;
   REQUIRE(difference.Normlinf() == MFEM_Approx(0.0, EPS));

   Vector errors(mesh.GetNE());
   errors = 0.0;
   REQUIRE(mesh.DerefineByError(errors, 1.0));
   REQUIRE(mesh.GetNE() == 21);
   fes.Update();
   field.Update();
   difference.Update();
   difference.ProjectCoefficient(affine);
   difference -= field;
   REQUIRE(difference.Normlinf() == MFEM_Approx(0.0, EPS));
}

TEST_CASE("NCMesh geometry-specific Nedelec refinement transfer", "[NCMesh]")
{
   const int order = GENERATE(1, 2);
   VectorFunctionCoefficient field_coefficient(2,
                                               [](const Vector &x, Vector &value)
   {
      value[0] = 2.0 - x[1];
      value[1] = 1.0 + x[0];
   });

   for (const FacePatternCase &test : geometry_specific_2d_patterns)
   {
      CAPTURE(order, test.geometry, test.pattern.GetId());
      Mesh mesh = Mesh::MakeCartesian2D(2, 1,
                                        test.geometry == Geometry::SQUARE ?
                                        Element::QUADRILATERAL : Element::TRIANGLE);
      mesh.EnsureNCMesh(true);
      ND_FECollection fec(order, 2);
      FiniteElementSpace fes(&mesh, &fec);
      GridFunction field(&fes);
      field.ProjectCoefficient(field_coefficient);

      Array<Refinement> refs(1);
      refs[0].Set(0, test.pattern);
      mesh.GeneralRefinement(refs);
      fes.Update();
      field.Update();
      GridFunction exact(&fes);
      exact.ProjectCoefficient(field_coefficient);
      exact -= field;
      REQUIRE(exact.Normlinf() == MFEM_Approx(0.0, EPS));
   }
}

TEST_CASE("NCMesh quadrilateral triangle-child patterns", "[NCMesh]")
{
   const Refinement::Pattern pattern = GENERATE(
                                          Refinement::Quadrilateral::DIAGONAL_02_2_TRIS,
                                          Refinement::Quadrilateral::DIAGONAL_13_2_TRIS,
                                          Refinement::Quadrilateral::DIAGONALS_4_TRIS,
                                          Refinement::Quadrilateral::EDGE_01_3_TRIS,
                                          Refinement::Quadrilateral::EDGE_12_3_TRIS,
                                          Refinement::Quadrilateral::EDGE_23_3_TRIS,
                                          Refinement::Quadrilateral::EDGE_30_3_TRIS);
   const int nested_child = GENERATE(-1, 0, 1, 2, 3);
   const int children =
      pattern.GetId() == Refinement::Quadrilateral::DIAGONALS_4_TRIS.GetId() ? 4 :
      pattern.GetId() >= Refinement::Quadrilateral::EDGE_01_3_TRIS.GetId() ? 3 : 2;
   if (nested_child >= children) { return; }
   CAPTURE(pattern.GetId(), nested_child);
   Mesh mesh("../../data/ref-square.mesh", 1, 1);
   mesh.EnsureNCMesh(true);
   H1_FECollection fec(2, 2);
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction field(&fes);
   FunctionCoefficient affine([](const Vector &x)
   {
      return 1.0 + 2.0*x[0] - 3.0*x[1];
   });
   field.ProjectCoefficient(affine);

   Array<Refinement> refs(1);
   refs[0].Set(0, pattern);
   mesh.GeneralRefinement(refs);
   REQUIRE(mesh.GetNE() == children);
   REQUIRE(SummarizeRefinedElements(mesh).geometry_counts[Geometry::TRIANGLE]
           == children);
   REQUIRE(SummarizeRefinedElements(mesh).volume == MFEM_Approx(1.0));
   REQUIRE(mesh.CheckElementOrientation(false) == 0);
   REQUIRE(mesh.CheckBdrElementOrientation(false) == 0);
   if (pattern.GetId() >= Refinement::Quadrilateral::EDGE_01_3_TRIS.GetId())
   {
      const int edge =
         pattern.GetId() - Refinement::Quadrilateral::EDGE_01_3_TRIS.GetId();
      const real_t *v0 = mesh.GetVertex(edge);
      const real_t *v1 = mesh.GetVertex((edge + 1) % 4);
      const real_t *mid = mesh.GetVertex(4);
      REQUIRE(mesh.GetNV() == 5);
      REQUIRE(mesh.GetNBE() == 5);
      REQUIRE(mid[0] == MFEM_Approx(0.5*(v0[0] + v1[0])));
      REQUIRE(mid[1] == MFEM_Approx(0.5*(v0[1] + v1[1])));
   }
   fes.Update();
   field.Update();
   GridFunction exact(&fes);
   exact.ProjectCoefficient(affine);
   exact -= field;
   REQUIRE(exact.Normlinf() == MFEM_Approx(0.0, EPS));

   if (nested_child >= 0)
   {
      refs[0].Set(nested_child, Refinement::X);
      mesh.GeneralRefinement(refs);
      REQUIRE(mesh.GetNE() == children + 1);
      fes.Update();
      field.Update();
      exact.Update();
      exact.ProjectCoefficient(affine);
      exact -= field;
      REQUIRE(exact.Normlinf() == MFEM_Approx(0.0, EPS));
   }

   std::stringstream serialized;
   mesh.Print(serialized);
   REQUIRE(serialized.str().find("MFEM NC mesh v1.2") == 0);
   Mesh reloaded(serialized, 1, 1);
   REQUIRE(reloaded.GetNE() == mesh.GetNE());
   REQUIRE(SummarizeRefinedElements(reloaded).volume == MFEM_Approx(1.0));
   for (int level = 0; level < (nested_child >= 0 ? 2 : 1); level++)
   {
      Vector errors(reloaded.GetNE());
      errors = 0.0;
      REQUIRE(reloaded.DerefineByError(errors, 1.0));
   }
   REQUIRE(reloaded.GetNE() == 1);
   REQUIRE(reloaded.GetElementVolume(0) == MFEM_Approx(1.0));
   REQUIRE(reloaded.GetNBE() == 4);
   int boundary_count[5] = {0, 0, 0, 0, 0};
   for (int i = 0; i < reloaded.GetNBE(); i++)
   {
      const int attribute = reloaded.GetBdrAttribute(i);
      REQUIRE(attribute >= 1);
      REQUIRE(attribute <= 4);
      boundary_count[attribute]++;
   }
   for (int attribute = 1; attribute <= 4; attribute++)
   {
      REQUIRE(boundary_count[attribute] == 1);
   }
}

TEST_CASE("NCMesh geometry-specific 2D L2 transfer", "[NCMesh]")
{
   const int case_index = GENERATE(range(0, num_geometry_specific_2d_patterns));
   const FacePatternCase &test = geometry_specific_2d_patterns[case_index];
   CAPTURE(test.geometry, test.pattern.GetId());
   Mesh mesh(test.geometry == Geometry::SQUARE ?
             "../../data/ref-square.mesh" : "../../data/ref-triangle.mesh", 1, 1);
   mesh.EnsureNCMesh(true);
   L2_FECollection fec(2, 2, BasisType::GaussLegendre,
                       FiniteElement::VALUE);
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction field(&fes);
   for (int i = 0; i < field.Size(); i++)
   {
      field(i) = (i % 11 - 5) / 7.0;
   }
   const real_t coarse_integral = IntegrateGridFunction(mesh, field);
   Array<Refinement> refs(1);
   refs[0].Set(0, test.pattern);
   mesh.GeneralRefinement(refs);
   REQUIRE(mesh.GetNE() == test.children);
   fes.Update();
   field.Update();
   REQUIRE(IntegrateGridFunction(mesh, field) ==
           MFEM_Approx(coarse_integral, EPS));

   Vector errors(mesh.GetNE());
   errors = 0.0;
   REQUIRE(mesh.DerefineByError(errors, 1.0));
   fes.Update();
   field.Update();
   REQUIRE(IntegrateGridFunction(mesh, field) ==
           MFEM_Approx(coarse_integral, EPS));
}

TEST_CASE("NCMesh quadrilateral diagonal center on skew element", "[NCMesh]")
{
   Mesh mesh("../../data/ref-square.mesh", 1, 1);
   mesh.GetVertex(2)[0] = 1.8;
   mesh.GetVertex(2)[1] = 1.2;
   mesh.GetVertex(3)[0] = -0.2;
   real_t center[2] = {0.0, 0.0};
   for (int i = 0; i < 4; i++)
   {
      center[0] += mesh.GetVertex(i)[0] / 4.0;
      center[1] += mesh.GetVertex(i)[1] / 4.0;
   }
   const real_t original_area = mesh.GetElementVolume(0);
   mesh.EnsureNCMesh(true);
   Array<Refinement> refs(1);
   refs[0].Set(0, Refinement::Quadrilateral::DIAGONALS_4_TRIS);
   mesh.GeneralRefinement(refs);
   REQUIRE(mesh.GetNV() == 5);
   REQUIRE(mesh.GetVertex(4)[0] == MFEM_Approx(center[0]));
   REQUIRE(mesh.GetVertex(4)[1] == MFEM_Approx(center[1]));
   REQUIRE(SummarizeRefinedElements(mesh).volume == MFEM_Approx(original_area));
   REQUIRE(mesh.CheckElementOrientation(false) == 0);
   std::stringstream serialized;
   mesh.Print(serialized);
   Mesh reloaded(serialized, 1, 1);
   REQUIRE(reloaded.GetNV() == 5);
   REQUIRE(reloaded.GetVertex(4)[0] == MFEM_Approx(center[0]));
   REQUIRE(reloaded.GetVertex(4)[1] == MFEM_Approx(center[1]));
   REQUIRE(SummarizeRefinedElements(reloaded).volume ==
           MFEM_Approx(original_area));
}

TEST_CASE("NCMesh triangle geometry-specific patterns", "[NCMesh]")
{
   const Refinement::Pattern pattern = GENERATE(
                                          Refinement::Triangle::CORNER_0_TRI_QUAD,
                                          Refinement::Triangle::CORNER_1_TRI_QUAD,
                                          Refinement::Triangle::CORNER_2_TRI_QUAD,
                                          Refinement::Triangle::CENTER_3_TRIS,
                                          Refinement::Triangle::CENTER_3_QUADS);
   const int nested_child = GENERATE(-1, 0, 1, 2);
   const bool center_pattern =
      pattern.GetId() >= Refinement::Triangle::CENTER_3_TRIS.GetId();
   const int children = center_pattern ? 3 : 2;
   if (nested_child >= children) { return; }
   CAPTURE(pattern.GetId(), nested_child);
   Mesh mesh("../../data/ref-triangle.mesh", 1, 1);
   const real_t area = mesh.GetElementVolume(0);
   mesh.EnsureNCMesh(true);
   H1_FECollection fec(2, 2);
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction field(&fes);
   FunctionCoefficient affine([](const Vector &x)
   {
      return 1.0 + 2.0*x[0] - 3.0*x[1];
   });
   field.ProjectCoefficient(affine);

   Array<Refinement> refs(1);
   refs[0].Set(0, pattern);
   mesh.GeneralRefinement(refs);
   const RefinedElementSummary refined = SummarizeRefinedElements(mesh);
   REQUIRE(mesh.GetNE() == children);
   REQUIRE(refined.volume == MFEM_Approx(area));
   if (!center_pattern)
   {
      REQUIRE(refined.geometry_counts[Geometry::TRIANGLE] == 1);
      REQUIRE(refined.geometry_counts[Geometry::SQUARE] == 1);
   }
   else if (pattern.GetId() == Refinement::Triangle::CENTER_3_TRIS.GetId())
   {
      REQUIRE(refined.geometry_counts[Geometry::TRIANGLE] == 3);
   }
   else
   {
      REQUIRE(refined.geometry_counts[Geometry::SQUARE] == 3);
   }
   REQUIRE(mesh.CheckElementOrientation(false) == 0);
   REQUIRE(mesh.CheckBdrElementOrientation(false) == 0);
   fes.Update();
   field.Update();
   GridFunction exact(&fes);
   exact.ProjectCoefficient(affine);
   exact -= field;
   REQUIRE(exact.Normlinf() == MFEM_Approx(0.0, EPS));

   if (center_pattern)
   {
      bool found_center = false;
      for (int i = 0; i < mesh.GetNV(); i++)
      {
         const real_t *v = mesh.GetVertex(i);
         if (v[0] == MFEM_Approx(1.0/3.0, EPS) &&
             v[1] == MFEM_Approx(1.0/3.0, EPS))
         { found_center = true; }
      }
      REQUIRE(found_center);
   }

   if (nested_child >= 0)
   {
      refs[0].Set(nested_child, Refinement::X);
      mesh.GeneralRefinement(refs);
      REQUIRE(mesh.GetNE() == children + 1);
      fes.Update();
      field.Update();
      exact.Update();
      exact.ProjectCoefficient(affine);
      exact -= field;
      REQUIRE(exact.Normlinf() == MFEM_Approx(0.0, EPS));
   }

   std::stringstream serialized;
   mesh.Print(serialized);
   REQUIRE(serialized.str().find(center_pattern ?
                                 "MFEM NC mesh v1.3" : "MFEM NC mesh v1.2") == 0);
   Mesh reloaded(serialized, 1, 1);
   REQUIRE(reloaded.GetNE() == mesh.GetNE());
   REQUIRE(SummarizeRefinedElements(reloaded).volume == MFEM_Approx(area));
   for (int level = 0; level < (nested_child >= 0 ? 2 : 1); level++)
   {
      Vector errors(reloaded.GetNE());
      errors = 0.0;
      REQUIRE(reloaded.DerefineByError(errors, 1.0));
   }
   REQUIRE(reloaded.GetNE() == 1);
   REQUIRE(reloaded.GetElementVolume(0) == MFEM_Approx(area));
   REQUIRE(reloaded.GetNBE() == 3);
   int boundary_count[4] = {0, 0, 0, 0};
   for (int i = 0; i < reloaded.GetNBE(); i++)
   {
      const int attribute = reloaded.GetBdrAttribute(i);
      REQUIRE(attribute >= 1);
      REQUIRE(attribute <= 3);
      boundary_count[attribute]++;
   }
   for (int attribute = 1; attribute <= 3; attribute++)
   { REQUIRE(boundary_count[attribute] == 1); }
}

TEST_CASE("NCMesh mixed pyramid children save and reload", "[NCMesh]")
{
   Mesh mesh("../../data/ref-pyramid.mesh", 1, 1);
   const real_t original_volume = mesh.GetElementVolume(0);
   mesh.EnsureNCMesh(true);
   Array<Refinement> refs(1);
   refs[0].Set(0, Refinement::XYZ);
   mesh.GeneralRefinement(refs);

   std::stringstream serialized;
   mesh.Print(serialized);
   Mesh reloaded(serialized, 1, 1);
   const RefinedElementSummary refined = SummarizeRefinedElements(reloaded);
   REQUIRE(reloaded.GetNE() == 10);
   REQUIRE(refined.geometry_counts[Geometry::PYRAMID] == 6);
   REQUIRE(refined.geometry_counts[Geometry::TETRAHEDRON] == 4);
   REQUIRE(refined.volume == MFEM_Approx(original_volume));
   REQUIRE(reloaded.CheckElementOrientation(false) == 0);
   REQUIRE(reloaded.CheckBdrElementOrientation(false) == 0);

   Vector errors(reloaded.GetNE());
   errors = 0.0;
   REQUIRE(reloaded.DerefineByError(errors, 1.0));
   REQUIRE(reloaded.GetNE() == 1);
   REQUIRE(reloaded.GetElementVolume(0) == MFEM_Approx(original_volume));
}

TEST_CASE("NCMesh explicit tet edge patterns", "[NCMesh]")
{
   const Refinement::Pattern pattern = GENERATE(
                                          Refinement::Tetrahedron::EDGE_12_2_TETS,
                                          Refinement::Tetrahedron::EDGE_13_2_TETS,
                                          Refinement::Tetrahedron::EDGE_23_2_TETS);
   const bool nested = GENERATE(false, true);
   const bool scaled = GENERATE(false, true);
   CAPTURE(pattern.GetId(), nested, scaled);
   Mesh mesh("../../data/ref-tetrahedron.mesh", 1, 1);
   const real_t original_volume = mesh.GetElementVolume(0);
   mesh.EnsureNCMesh(true);
   if (scaled) { mesh.SetScaledNCMesh(); }
   H1_FECollection fec(2, 3);
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction field(&fes);
   FunctionCoefficient affine([](const Vector &x)
   {
      return 1.0 + x[0] - 2.0*x[1] + 3.0*x[2];
   });
   field.ProjectCoefficient(affine);

   Array<Refinement> refs(1);
   refs[0].Set(0, pattern);
   REQUIRE(refs[0].GetType() == pattern.GetId());
   mesh.GeneralRefinement(refs);
   REQUIRE(mesh.ncmesh->GetNumElements() == 3);
   if (nested)
   {
      fes.Update();
      field.Update();
      refs[0].Set(0, Refinement::XYZ);
      mesh.GeneralRefinement(refs);
   }
   REQUIRE(mesh.GetNE() == (nested ? 9 : 2));
   REQUIRE(SummarizeRefinedElements(mesh).volume ==
           MFEM_Approx(original_volume));
   REQUIRE(mesh.CheckElementOrientation(false) == 0);
   REQUIRE(mesh.CheckBdrElementOrientation(false) == 0);
   fes.Update();
   field.Update();
   GridFunction exact(&fes);
   exact.ProjectCoefficient(affine);
   exact -= field;
   REQUIRE(exact.Normlinf() == MFEM_Approx(0.0, EPS));

   std::stringstream serialized;
   mesh.Print(serialized);
   REQUIRE(serialized.str().find(scaled ? "MFEM NC mesh v1.3" :
                                 "MFEM NC mesh v1.2") == 0);
   Mesh reloaded(serialized, 1, 1);
   REQUIRE(reloaded.GetNE() == mesh.GetNE());
   REQUIRE(SummarizeRefinedElements(reloaded).volume ==
           MFEM_Approx(original_volume));
   for (int level = 0; level < (nested ? 2 : 1); level++)
   {
      Vector errors(reloaded.GetNE());
      errors = 0.0;
      REQUIRE(reloaded.DerefineByError(errors, 1.0));
   }
   REQUIRE(reloaded.GetNE() == 1);
   REQUIRE(reloaded.GetElementVolume(0) == MFEM_Approx(original_volume));
}

TEST_CASE("NCMesh anisotropic tet edge bisection", "[NCMesh]")
{
   const char ref_type = GENERATE(
                            Refinement::X, Refinement::Y, Refinement::Z,
                            Refinement::Tetrahedron::EDGE_12_2_TETS.GetId(),
                            Refinement::Tetrahedron::EDGE_13_2_TETS.GetId(),
                            Refinement::Tetrahedron::EDGE_23_2_TETS.GetId());
   CAPTURE(ref_type);
   Mesh mesh("../../data/ref-tetrahedron.mesh", 1, 1);
   const real_t original_volume = mesh.GetElementVolume(0);
   mesh.EnsureNCMesh(true);
   H1_FECollection fec(2, 3);
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction field(&fes);
   FunctionCoefficient affine([](const Vector &x)
   {
      return 1.0 + x[0] - 2.0*x[1] + 3.0*x[2];
   });
   field.ProjectCoefficient(affine);

   Array<Refinement> refs(1);
   refs[0] = TetRefinement(0, ref_type);
   mesh.GeneralRefinement(refs);
   REQUIRE(mesh.GetNE() == 2);
   REQUIRE(SummarizeRefinedElements(mesh).volume ==
           MFEM_Approx(original_volume));
   REQUIRE(mesh.CheckElementOrientation(false) == 0);
   REQUIRE(mesh.CheckBdrElementOrientation(false) == 0);
   REQUIRE(mesh.GetNBE() == 6);
   fes.Update();
   field.Update();
   GridFunction exact(&fes);
   exact.ProjectCoefficient(affine);
   exact -= field;
   REQUIRE(exact.Normlinf() == MFEM_Approx(0.0, EPS));

   std::stringstream serialized;
   mesh.Print(serialized);
   Mesh reloaded(serialized, 1, 1);
   REQUIRE(reloaded.GetNE() == 2);
   Vector reload_errors(reloaded.GetNE());
   reload_errors = 0.0;
   REQUIRE(reloaded.DerefineByError(reload_errors, 1.0));
   REQUIRE(reloaded.GetNE() == 1);
   REQUIRE(reloaded.GetElementVolume(0) == MFEM_Approx(original_volume));

   Vector errors(mesh.GetNE());
   errors = 0.0;
   REQUIRE(mesh.DerefineByError(errors, 1.0));
   REQUIRE(mesh.GetNE() == 1);
   REQUIRE(mesh.GetElementVolume(0) == MFEM_Approx(original_volume));
   REQUIRE(mesh.GetNBE() == 4);
   int boundary_count[5] = {0, 0, 0, 0, 0};
   for (int i = 0; i < mesh.GetNBE(); i++)
   {
      const int attribute = mesh.GetBdrAttribute(i);
      REQUIRE(attribute >= 1);
      REQUIRE(attribute <= 4);
      boundary_count[attribute]++;
   }
   for (int attribute = 1; attribute <= 4; attribute++)
   {
      REQUIRE(boundary_count[attribute] == 1);
   }
   fes.Update();
   field.Update();
   GridFunction coarse_exact(&fes);
   coarse_exact.ProjectCoefficient(affine);
   coarse_exact -= field;
   REQUIRE(coarse_exact.Normlinf() == MFEM_Approx(0.0, EPS));
}

TEST_CASE("NCMesh anisotropic tet L2 derefinement", "[NCMesh]")
{
   const char ref_type = GENERATE(
                            Refinement::X, Refinement::Y, Refinement::Z,
                            Refinement::Tetrahedron::EDGE_12_2_TETS.GetId(),
                            Refinement::Tetrahedron::EDGE_13_2_TETS.GetId(),
                            Refinement::Tetrahedron::EDGE_23_2_TETS.GetId());
   const bool nested = GENERATE(false, true);
   const bool variable_order = GENERATE(false, true);
   CAPTURE(ref_type, nested, variable_order);

   Mesh mesh("../../data/ref-tetrahedron.mesh", 1, 1);
   const real_t original_volume = mesh.GetElementVolume(0);
   mesh.EnsureNCMesh(true);
   L2_FECollection fec(2, 3, BasisType::GaussLegendre,
                       FiniteElement::VALUE);
   FiniteElementSpace fes(&mesh, &fec);
   if (variable_order)
   {
      fes.SetElementOrder(0, 3);
      fes.Update(false);
   }
   GridFunction field(&fes);
   FunctionCoefficient affine([](const Vector &x)
   {
      return 1.0 + x[0] - 2.0*x[1] + 3.0*x[2];
   });
   field.ProjectCoefficient(affine);

   Array<Refinement> refs(1);
   refs[0] = TetRefinement(0, ref_type);
   mesh.GeneralRefinement(refs);
   fes.Update();
   field.Update();
   if (nested)
   {
      refs[0].Set(0, Refinement::XYZ);
      mesh.GeneralRefinement(refs);
      fes.Update();
      field.Update();
   }
   REQUIRE(mesh.GetNE() == (nested ? 9 : 2));
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      REQUIRE(fes.GetElementOrder(i) == 2 + int(variable_order));
   }

   for (int i = 0; i < field.Size(); i++)
   {
      field(i) = (i % 13 - 6) / 9.0;
   }
   const real_t fine_integral = IntegrateGridFunction(mesh, field);
   for (int level = 0; level < (nested ? 2 : 1); level++)
   {
      Vector errors(mesh.GetNE());
      errors = 0.0;
      REQUIRE(mesh.DerefineByError(errors, 1.0));
      fes.Update();
      field.Update();
      REQUIRE(IntegrateGridFunction(mesh, field) ==
              MFEM_Approx(fine_integral, EPS));
   }
   REQUIRE(mesh.GetNE() == 1);
   REQUIRE(fes.GetElementOrder(0) == 2 + int(variable_order));
   REQUIRE(mesh.GetElementVolume(0) == MFEM_Approx(original_volume));
}

TEST_CASE("NCMesh anisotropic tet shared face", "[NCMesh]")
{
   const char ref_type = GENERATE(
                            Refinement::X, Refinement::Y, Refinement::Z,
                            Refinement::Tetrahedron::EDGE_12_2_TETS.GetId(),
                            Refinement::Tetrahedron::EDGE_13_2_TETS.GetId(),
                            Refinement::Tetrahedron::EDGE_23_2_TETS.GetId());
   const int shared_face = GENERATE(1, 2, 3);
   if ((ref_type == Refinement::X && shared_face == 1) ||
       (ref_type == Refinement::Y && shared_face == 2) ||
       (ref_type == Refinement::Z && shared_face == 3) ||
       (ref_type == Refinement::Tetrahedron::EDGE_12_2_TETS.GetId() &&
        shared_face != 3) ||
       (ref_type == Refinement::Tetrahedron::EDGE_13_2_TETS.GetId() &&
        shared_face != 2) ||
       (ref_type == Refinement::Tetrahedron::EDGE_23_2_TETS.GetId() &&
        shared_face != 1)) { return; }
   CAPTURE(ref_type, shared_face);
   Mesh mesh(3, 5, 2, 0);
   mesh.AddVertex(0.0, 0.0, 0.0);
   mesh.AddVertex(1.0, 0.0, 0.0);
   mesh.AddVertex(0.0, 1.0, 0.0);
   mesh.AddVertex(0.0, 0.0, 1.0);
   if (shared_face == 1) { mesh.AddVertex(-1.0, 0.0, 0.0); }
   else if (shared_face == 2) { mesh.AddVertex(0.0, -1.0, 0.0); }
   else { mesh.AddVertex(0.0, 0.0, -1.0); }
   mesh.AddTet(0, 1, 2, 3);
   if (shared_face == 1) { mesh.AddTet(0, 3, 2, 4); }
   else if (shared_face == 2) { mesh.AddTet(0, 1, 3, 4); }
   else { mesh.AddTet(0, 2, 1, 4); }
   mesh.FinalizeTetMesh(1, 0, true);
   const real_t original_volume = SummarizeRefinedElements(mesh).volume;
   mesh.EnsureNCMesh(true);
   H1_FECollection fec(2, 3);
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction field(&fes);
   FunctionCoefficient affine([](const Vector &x)
   {
      return 1.0 + x[0] - 2.0*x[1] + 3.0*x[2];
   });
   field.ProjectCoefficient(affine);

   Array<Refinement> refs(1);
   refs[0] = TetRefinement(0, ref_type);
   mesh.GeneralRefinement(refs);
   REQUIRE(mesh.GetNE() == 3);
   REQUIRE(SummarizeRefinedElements(mesh).volume ==
           MFEM_Approx(original_volume));
   REQUIRE(mesh.CheckElementOrientation(false) == 0);
   REQUIRE(mesh.CheckBdrElementOrientation(false) == 0);
   const auto &faces = mesh.ncmesh->GetFaceList();
   REQUIRE(faces.masters.Size() == 1);
   REQUIRE(faces.slaves.Size() == 2);
   fes.Update();
   field.Update();
   GridFunction exact(&fes);
   exact.ProjectCoefficient(affine);
   exact -= field;
   REQUIRE(exact.Normlinf() == MFEM_Approx(0.0, EPS));

   refs[0].Set(0, Refinement::XYZ);
   mesh.GeneralRefinement(refs);
   REQUIRE(mesh.GetNE() == 10);
   const auto &nested_faces = mesh.ncmesh->GetFaceList();
   REQUIRE(nested_faces.masters.Size() == 2);
   REQUIRE(nested_faces.slaves.Size() == 10);
   fes.Update();
   field.Update();
   exact.Update();
   exact.ProjectCoefficient(affine);
   exact -= field;
   REQUIRE(exact.Normlinf() == MFEM_Approx(0.0, EPS));
}

TEST_CASE("NCMesh anisotropic triangle round trip", "[NCMesh]")
{
   const int ref_type = GENERATE(Refinement::X, Refinement::Y,
                                 Refinement::Z);
   CAPTURE(ref_type);

   Mesh mesh("../../data/ref-triangle.mesh");
   const int *triangle_vertices = mesh.GetElement(0)->GetVertices();
   const int split_vertex0 = triangle_vertices[ref_type == Refinement::Z ? 1 : 0];
   const int split_vertex1 = triangle_vertices[ref_type == Refinement::X ? 1 : 2];
   int split_attribute = -1;
   Array<int> boundary_vertices;
   for (int i = 0; i < mesh.GetNBE(); i++)
   {
      mesh.GetBdrElementVertices(i, boundary_vertices);
      if ((boundary_vertices[0] == split_vertex0 &&
           boundary_vertices[1] == split_vertex1) ||
          (boundary_vertices[1] == split_vertex0 &&
           boundary_vertices[0] == split_vertex1))
      {
         split_attribute = mesh.GetBdrAttribute(i);
      }
   }
   REQUIRE(split_attribute >= 1);
   mesh.EnsureNCMesh(true);
   H1_FECollection fec(2, 2);
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction field(&fes);
   FunctionCoefficient affine([](const Vector &x)
   {
      return 1.0 + 2.0*x[0] - 3.0*x[1];
   });
   field.ProjectCoefficient(affine);

   Array<Refinement> refs(1);
   refs[0].Set(0, ref_type);
   mesh.GeneralRefinement(refs, 1);
   REQUIRE(mesh.GetNE() == 2);
   REQUIRE(SummarizeRefinedElements(mesh).volume == MFEM_Approx(0.5));
   REQUIRE(mesh.CheckElementOrientation(false) == 0);
   REQUIRE(mesh.CheckBdrElementOrientation(false) == 0);
   REQUIRE(mesh.GetNBE() == 4);
   int boundary_count[4] = {0, 0, 0, 0};
   for (int i = 0; i < mesh.GetNBE(); i++)
   {
      const int attribute = mesh.GetBdrAttribute(i);
      REQUIRE(attribute >= 1);
      REQUIRE(attribute <= 3);
      boundary_count[attribute]++;
   }
   for (int attribute = 1; attribute <= 3; attribute++)
   {
      REQUIRE(boundary_count[attribute] ==
              (attribute == split_attribute ? 2 : 1));
   }
   const CoarseFineTransformations &trans = mesh.GetRefinementTransforms();
   REQUIRE(trans.embeddings.Size() == 2);
   fes.Update();
   field.Update();
   GridFunction exact(&fes);
   exact.ProjectCoefficient(affine);
   exact -= field;
   REQUIRE(exact.Normlinf() == MFEM_Approx(0.0, EPS));

   std::stringstream serialized;
   mesh.Print(serialized);
   Mesh reloaded(serialized, 1, 1);
   REQUIRE(reloaded.GetNE() == 2);
   REQUIRE(SummarizeRefinedElements(reloaded).volume == MFEM_Approx(0.5));
   Vector reload_errors(reloaded.GetNE());
   reload_errors = 0.0;
   REQUIRE(reloaded.DerefineByError(reload_errors, 1.0));
   REQUIRE(reloaded.GetNE() == 1);

   Vector errors(mesh.GetNE());
   errors = 0.0;
   REQUIRE(mesh.DerefineByError(errors, 1.0));
   REQUIRE(mesh.GetNE() == 1);
   REQUIRE(SummarizeRefinedElements(mesh).volume == MFEM_Approx(0.5));
   REQUIRE(mesh.CheckElementOrientation(false) == 0);
   REQUIRE(mesh.CheckBdrElementOrientation(false) == 0);
   REQUIRE(mesh.GetNBE() == 3);
   boundary_count[1] = boundary_count[2] = boundary_count[3] = 0;
   for (int i = 0; i < mesh.GetNBE(); i++)
   {
      const int attribute = mesh.GetBdrAttribute(i);
      REQUIRE(attribute >= 1);
      REQUIRE(attribute <= 3);
      boundary_count[attribute]++;
   }
   REQUIRE(boundary_count[1] == 1);
   REQUIRE(boundary_count[2] == 1);
   REQUIRE(boundary_count[3] == 1);
   fes.Update();
   field.Update();
   GridFunction coarse_exact(&fes);
   coarse_exact.ProjectCoefficient(affine);
   coarse_exact -= field;
   REQUIRE(coarse_exact.Normlinf() == MFEM_Approx(0.0, EPS));
}

TEST_CASE("NCMesh anisotropic triangle hanging edge", "[NCMesh]")
{
   const int ref_type = GENERATE(Refinement::X, Refinement::Y,
                                 Refinement::Z);
   CAPTURE(ref_type);

   Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::TRIANGLE);
   const real_t original_area = SummarizeRefinedElements(mesh).volume;
   mesh.EnsureNCMesh(true);
   H1_FECollection fec(2, 2);
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction field(&fes);
   FunctionCoefficient affine([](const Vector &x)
   {
      return 1.0 + 2.0*x[0] - 3.0*x[1];
   });
   field.ProjectCoefficient(affine);

   Array<Refinement> refs(1);
   refs[0].Set(0, ref_type);
   mesh.GeneralRefinement(refs);
   REQUIRE(mesh.GetNE() == 3);
   REQUIRE(SummarizeRefinedElements(mesh).volume ==
           MFEM_Approx(original_area));
   REQUIRE(mesh.CheckElementOrientation(false) == 0);
   REQUIRE(mesh.CheckBdrElementOrientation(false) == 0);
   fes.Update();
   field.Update();
   GridFunction exact(&fes);
   exact.ProjectCoefficient(affine);
   exact -= field;
   REQUIRE(exact.Normlinf() == MFEM_Approx(0.0, EPS));

   // Exercise the sheared reference maps through a second refinement.
   refs[0].Set(0, ref_type == Refinement::X ?
               Refinement::Y : Refinement::X);
   mesh.GeneralRefinement(refs);
   REQUIRE(mesh.GetNE() == 4);
   REQUIRE(SummarizeRefinedElements(mesh).volume ==
           MFEM_Approx(original_area));
   fes.Update();
   field.Update();
   exact.Update();
   exact.ProjectCoefficient(affine);
   exact -= field;
   REQUIRE(exact.Normlinf() == MFEM_Approx(0.0, EPS));
}

TEST_CASE("NCMesh anisotropic triangle L2 derefinement", "[NCMesh]")
{
   const int ref_type = GENERATE(Refinement::X, Refinement::Y,
                                 Refinement::Z);
   const bool nested = GENERATE(false, true);
   const bool variable_order = GENERATE(false, true);
   CAPTURE(ref_type, nested, variable_order);

   Mesh mesh("../../data/ref-triangle.mesh");
   mesh.EnsureNCMesh(true);
   L2_FECollection fec(2, 2, BasisType::GaussLegendre,
                       FiniteElement::VALUE);
   FiniteElementSpace fes(&mesh, &fec);
   if (variable_order)
   {
      fes.SetElementOrder(0, 3);
      fes.Update(false);
   }
   GridFunction field(&fes);
   FunctionCoefficient affine([](const Vector &x)
   {
      return 1.0 + 2.0*x[0] - 3.0*x[1];
   });
   field.ProjectCoefficient(affine);

   Array<Refinement> refs(1);
   refs[0].Set(0, ref_type);
   mesh.GeneralRefinement(refs);
   fes.Update();
   field.Update();
   if (nested)
   {
      refs[0].Set(0, ref_type == Refinement::X ?
                  Refinement::Y : Refinement::X);
      mesh.GeneralRefinement(refs);
      fes.Update();
      field.Update();
   }
   REQUIRE(mesh.GetNE() == (nested ? 3 : 2));
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      REQUIRE(fes.GetElementOrder(i) == 2 + int(variable_order));
   }

   for (int i = 0; i < field.Size(); i++)
   {
      field(i) = (i % 11 - 5) / 7.0;
   }
   const real_t fine_integral = IntegrateGridFunction(mesh, field);
   for (int level = 0; level < (nested ? 2 : 1); level++)
   {
      Vector errors(mesh.GetNE());
      errors = 0.0;
      REQUIRE(mesh.DerefineByError(errors, 1.0));
      fes.Update();
      field.Update();
      REQUIRE(IntegrateGridFunction(mesh, field) ==
              MFEM_Approx(fine_integral, EPS));
   }
   REQUIRE(mesh.GetNE() == 1);
   REQUIRE(fes.GetElementOrder(0) == 2 + int(variable_order));
   REQUIRE(SummarizeRefinedElements(mesh).volume == MFEM_Approx(0.5));
   REQUIRE(mesh.CheckElementOrientation(false) == 0);
   REQUIRE(mesh.CheckBdrElementOrientation(false) == 0);
}

TEST_CASE("NCMesh anisotropic prism refinement in mixed mesh", "[NCMesh]")
{
   const int prism_attribute = GENERATE(5, 6, 7, 8);
   const int ref_type = GENERATE(Refinement::XY, Refinement::Z);
   CAPTURE(prism_attribute, ref_type);

   Mesh mesh("../../data/l-mixed.mesh", 1, 1);
   const real_t original_volume = SummarizeRefinedElements(mesh).volume;
   H1_FECollection fec(2, 3);
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction field(&fes);
   FunctionCoefficient affine([](const Vector &x)
   {
      return 1.0 + x[0] - 2.0*x[1] + 3.0*x[2];
   });
   field.ProjectCoefficient(affine);

   mesh.EnsureNCMesh(true);
   Array<Refinement> refs;
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      if (mesh.GetAttribute(i) == prism_attribute)
      {
         REQUIRE(mesh.GetElementBaseGeometry(i) == Geometry::PRISM);
         refs.Append(Refinement(i, ref_type));
      }
   }
   REQUIRE(refs.Size() == 1);
   mesh.GeneralRefinement(refs);

   int prism_children = 0;
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      if (mesh.GetAttribute(i) == prism_attribute)
      {
         REQUIRE(mesh.GetElementBaseGeometry(i) == Geometry::PRISM);
         prism_children++;
      }
   }
   REQUIRE(prism_children == (ref_type == Refinement::XY ? 4 : 2));
   REQUIRE(SummarizeRefinedElements(mesh).volume ==
           MFEM_Approx(original_volume));
   REQUIRE(mesh.CheckElementOrientation(false) == 0);
   REQUIRE(mesh.CheckBdrElementOrientation(false) == 0);
   fes.Update();
   field.Update();
   GridFunction exact(&fes);
   exact.ProjectCoefficient(affine);
   exact -= field;
   REQUIRE(exact.Normlinf() == MFEM_Approx(0.0, EPS));
}

TEST_CASE("NCMesh anisotropic prism derefinement in mixed mesh", "[NCMesh]")
{
   const int prism_attribute = GENERATE(5, 6, 7, 8);
   const int ref_type = GENERATE(Refinement::XY, Refinement::Z);
   const bool nested = GENERATE(false, true);
   const bool continuous = GENERATE(false, true);
   const bool variable_order = GENERATE(false, true);
   CAPTURE(prism_attribute, ref_type, nested, continuous, variable_order);

   Mesh mesh("../../data/l-mixed.mesh", 1, 1);
   const RefinedElementSummary original = SummarizeRefinedElements(mesh);
   mesh.EnsureNCMesh(true);
   H1_FECollection h1_fec(2, 3);
   L2_FECollection l2_fec(2, 3, BasisType::GaussLegendre,
                          FiniteElement::VALUE);
   FiniteElementSpace fes(&mesh, continuous ?
                          static_cast<FiniteElementCollection*>(&h1_fec) :
                          static_cast<FiniteElementCollection*>(&l2_fec));
   if (variable_order)
   {
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         if (mesh.GetAttribute(i) == prism_attribute)
         {
            fes.SetElementOrder(i, 3);
         }
      }
      fes.Update(false);
   }
   GridFunction field(&fes);
   FunctionCoefficient affine([](const Vector &x)
   {
      return 1.0 + x[0] - 2.0*x[1] + 3.0*x[2];
   });
   field.ProjectCoefficient(affine);

   Array<Refinement> refs;
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      if (mesh.GetAttribute(i) == prism_attribute)
      {
         refs.Append(Refinement(i, ref_type));
      }
   }
   REQUIRE(refs.Size() == 1);
   mesh.GeneralRefinement(refs);
   fes.Update();
   field.Update();
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      if (mesh.GetAttribute(i) == prism_attribute)
      {
         REQUIRE(fes.GetElementOrder(i) == 2 + int(variable_order));
      }
   }

   if (nested)
   {
      refs.SetSize(0);
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         if (mesh.GetAttribute(i) == prism_attribute)
         {
            const int second_type = ref_type == Refinement::XY ?
                                    Refinement::Z : Refinement::XY;
            refs.Append(Refinement(i, second_type));
            break;
         }
      }
      REQUIRE(refs.Size() == 1);
      mesh.GeneralRefinement(refs);
      fes.Update();
      field.Update();
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         if (mesh.GetAttribute(i) == prism_attribute)
         {
            REQUIRE(fes.GetElementOrder(i) == 2 + int(variable_order));
         }
      }
   }

   real_t fine_integral = 0.0;
   if (!continuous)
   {
      for (int i = 0; i < field.Size(); i++)
      {
         field(i) = (i % 13 - 6) / 9.0;
      }
      fine_integral = IntegrateGridFunction(mesh, field);
   }

   for (int level = 0; level < (nested ? 2 : 1); level++)
   {
      Vector error(mesh.GetNE());
      error = 0.0;
      REQUIRE(mesh.DerefineByError(error, 1.0));
      fes.Update();
      field.Update();
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         if (mesh.GetAttribute(i) == prism_attribute)
         {
            REQUIRE(fes.GetElementOrder(i) == 2 + int(variable_order));
         }
      }
   }

   const RefinedElementSummary restored = SummarizeRefinedElements(mesh);
   REQUIRE(mesh.GetNE() == 9);
   REQUIRE(restored.geometry_counts == original.geometry_counts);
   REQUIRE(restored.volume == MFEM_Approx(original.volume));
   REQUIRE(mesh.CheckElementOrientation(false) == 0);
   REQUIRE(mesh.CheckBdrElementOrientation(false) == 0);
   if (continuous)
   {
      GridFunction exact(&fes);
      exact.ProjectCoefficient(affine);
      exact -= field;
      REQUIRE(exact.Normlinf() == MFEM_Approx(0.0, EPS));
   }
   else
   {
      REQUIRE(IntegrateGridFunction(mesh, field) ==
              MFEM_Approx(fine_integral, EPS));
   }
}

TEST_CASE("NCMesh nested mixed child transforms", "[NCMesh]")
{
   Mesh parent("../../data/ref-pyramid.mesh", 1, 1);
   NCMesh ncmesh(&parent);
   ncmesh.MarkCoarseLevel();
   Array<Refinement> refs(1);
   refs[0].Set(0, Refinement::XYZ);
   ncmesh.Refine(refs);

   int tet = -1;
   for (int i = 0; i < 10; i++)
   {
      if (ncmesh.GetElementGeometry(i) == Geometry::TETRAHEDRON)
      {
         tet = i;
         break;
      }
   }
   REQUIRE(tet >= 0);
   refs[0].Set(tet, Refinement::XYZ);
   ncmesh.Refine(refs);

   const CoarseFineTransformations &trans = ncmesh.GetRefinementTransforms();
   const DenseMatrixStack &pmats = trans.point_matrices[Geometry::PYRAMID];
   REQUIRE(trans.embeddings.Size() == 17);
   std::array<int, Geometry::NumGeom> counts = {};
   for (int i = 0; i < trans.embeddings.Size(); i++)
   {
      const Embedding &embedding = trans.embeddings[i];
      const Geometry::Type geom = ncmesh.GetElementGeometry(i);
      counts[geom]++;
      CAPTURE(i);
      REQUIRE(embedding.parent == 0);
      REQUIRE(embedding.geom == Geometry::PYRAMID);
      REQUIRE(pmats.SizeJ(embedding.matrix) ==
              Geometry::NumVerts[geom]);
   }
   REQUIRE(counts[Geometry::PYRAMID] == 6);
   REQUIRE(counts[Geometry::TETRAHEDRON] == 11);
}

TEST_CASE("NCMesh derived tetrahedron face edge topology", "[NCMesh]")
{
   const int pyramid_attribute = GENERATE(1, 2, 3, 4, 9);
   const int tet_child = GENERATE(0, 1, 2, 3);
   // The unrelated root tet must not change constraints around tetrahedra
   // created from a pyramid; both meshes need tetrahedral face-edge traversal.
   auto get_nc_counts = [pyramid_attribute, tet_child](bool add_root_tet)
   {
      Mesh mesh("../../data/l-mixed.mesh", 1, 1);
      if (add_root_tet)
      {
         const int v0 = mesh.AddVertex(2.0, 0.0, 0.0);
         const int v1 = mesh.AddVertex(3.0, 0.0, 0.0);
         const int v2 = mesh.AddVertex(2.0, 1.0, 0.0);
         const int v3 = mesh.AddVertex(2.0, 0.0, 1.0);
         mesh.AddTet(v0, v1, v2, v3, 99);
         mesh.FinalizeTopology();
         mesh.Finalize();
      }
      mesh.EnsureNCMesh(true);
      Array<Refinement> refs;
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         if (mesh.GetAttribute(i) == pyramid_attribute)
         {
            refs.Append(Refinement(i, Refinement::XYZ));
         }
      }
      mesh.GeneralRefinement(refs);
      refs.SetSize(0);
      int tet = 0;
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         if (mesh.GetAttribute(i) == pyramid_attribute &&
             mesh.GetElementBaseGeometry(i) == Geometry::TETRAHEDRON &&
             tet++ == tet_child)
         {
            refs.Append(Refinement(i, Refinement::XYZ));
            break;
         }
      }
      REQUIRE(refs.Size() == 1);
      mesh.GeneralRefinement(refs);
      const auto &faces = mesh.ncmesh->GetFaceList();
      const auto &edges = mesh.ncmesh->GetEdgeList();
      return std::array<int, 4> {faces.masters.Size(), faces.slaves.Size(),
                                 edges.masters.Size(), edges.slaves.Size()
                                };
   };
   const auto without = get_nc_counts(false);
   const auto with = get_nc_counts(true);
   CAPTURE(pyramid_attribute, tet_child,
           without[0], without[1], without[2], without[3],
           with[0], with[1], with[2], with[3]);
   REQUIRE(without == with);
}

TEST_CASE("NCMesh mixed pyramid parent field derefinement", "[NCMesh]")
{
   const int order = GENERATE(1, 2);
   CAPTURE(order);
   Mesh mesh("../../data/l-mixed.mesh", 1, 1);
   mesh.EnsureNCMesh(true);
   H1_FECollection fec(order, 3);
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction field(&fes);
   FunctionCoefficient affine([](const Vector &x)
   {
      return 1.0 + x[0] - 2.0*x[1] + 3.0*x[2];
   });
   field.ProjectCoefficient(affine);
   const int coarse_ne = mesh.GetNE();

   Array<Refinement> refs(1);
   refs[0].Set(coarse_ne - 1, Refinement::XYZ);
   mesh.GeneralRefinement(refs, 1);
   fes.Update();
   field.Update();
   GridFunction exact_fine(&fes);
   exact_fine.ProjectCoefficient(affine);
   exact_fine -= field;
   REQUIRE(exact_fine.Normlinf() == MFEM_Approx(0.0, EPS));

   Array<real_t> error(mesh.GetNE());
   error = 0.0;
   REQUIRE(mesh.DerefineByError(error, 1.0));
   REQUIRE(mesh.GetNE() == coarse_ne);
   fes.Update();
   field.Update();
   GridFunction exact_coarse(&fes);
   exact_coarse.ProjectCoefficient(affine);
   exact_coarse -= field;
   REQUIRE(exact_coarse.Normlinf() == MFEM_Approx(0.0, EPS));
}

TEST_CASE("NCMesh mixed pyramid L2 derefinement", "[NCMesh]")
{
   const char *mesh_name = GENERATE("../../data/ref-pyramid.mesh",
                                    "../../data/l-mixed.mesh");
   const int order = GENERATE(0, 1, 2);
   const int pyr_type = GENERATE(0, 1);
   const bool arbitrary_fine = GENERATE(false, true);
   CAPTURE(mesh_name, order, pyr_type, arbitrary_fine);

   Mesh mesh(mesh_name, 1, 1);
   L2_FECollection fec(order, 3, BasisType::GaussLegendre,
                       FiniteElement::VALUE, pyr_type);
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction field(&fes);
   ConstantCoefficient constant(1.5);
   FunctionCoefficient affine([](const Vector &x)
   {
      return 1.0 + x[0] - 2.0*x[1] + 3.0*x[2];
   });
   field.ProjectCoefficient(order == 0 ?
                            static_cast<Coefficient&>(constant) :
                            static_cast<Coefficient&>(affine));
   const Vector original_values(field);
   const int coarse_ne = mesh.GetNE();

   mesh.EnsureNCMesh(true);
   Array<Refinement> refs(1);
   refs[0].Set(coarse_ne - 1, Refinement::XYZ);
   mesh.GeneralRefinement(refs, 1);
   fes.Update();
   field.Update();
   if (arbitrary_fine)
   {
      for (int i = 0; i < field.Size(); i++)
      {
         field(i) = (i % 13 - 6) / 9.0;
      }
   }
   const real_t fine_integral = IntegrateGridFunction(mesh, field);

   Array<real_t> error(mesh.GetNE());
   error = 0.0;
   REQUIRE(mesh.DerefineByError(error, 1.0));
   fes.Update();
   field.Update();
   REQUIRE(mesh.GetNE() == coarse_ne);
   REQUIRE(IntegrateGridFunction(mesh, field) ==
           MFEM_Approx(fine_integral, 1e-10));
   if (!arbitrary_fine)
   {
      field -= original_values;
      REQUIRE(field.Normlinf() == MFEM_Approx(0.0, 1e-10));
   }
}

TEST_CASE("NCMesh variable-order mixed pyramid derefinement", "[NCMesh]")
{
   const bool continuous = GENERATE(false, true);
   const int pyr_type = GENERATE(0, 1);
   const int pyr_order = GENERATE(2, 3, 4);
   const bool arbitrary_fine = GENERATE(false, true);
   if ((continuous && arbitrary_fine) || (!continuous && pyr_order == 4))
   {
      return;
   }
   CAPTURE(continuous, pyr_type, pyr_order, arbitrary_fine);

   Mesh mesh("../../data/l-mixed.mesh", 1, 1);
   mesh.EnsureNCMesh(true);
   H1_FECollection h1_fec(1, 3, BasisType::GaussLobatto, pyr_type);
   L2_FECollection l2_fec(1, 3, BasisType::GaussLegendre,
                          FiniteElement::VALUE, pyr_type);
   FiniteElementSpace fes(&mesh, continuous ?
                          static_cast<FiniteElementCollection*>(&h1_fec) :
                          static_cast<FiniteElementCollection*>(&l2_fec));
   const int coarse_ne = mesh.GetNE();
   fes.SetElementOrder(coarse_ne - 1, pyr_order);
   fes.Update(false);
   GridFunction field(&fes);
   FunctionCoefficient affine([](const Vector &x)
   {
      return 1.0 + x[0] - 2.0*x[1] + 3.0*x[2];
   });
   field.ProjectCoefficient(affine);
   const Vector original_values(field);

   Array<Refinement> refs(1);
   refs[0].Set(coarse_ne - 1, Refinement::XYZ);
   mesh.GeneralRefinement(refs, 1);
   fes.Update();
   field.Update();
   GridFunction fine_exact(&fes);
   fine_exact.ProjectCoefficient(affine);
   fine_exact -= field;
   REQUIRE(fine_exact.Normlinf() == MFEM_Approx(0.0, 1e-10));
   if (arbitrary_fine)
   {
      for (int i = 0; i < field.Size(); i++)
      {
         field(i) = (i % 13 - 6) / 9.0;
      }
   }
   const real_t fine_integral = continuous ? 0.0 :
                                IntegrateGridFunction(mesh, field);

   Array<real_t> error(mesh.GetNE());
   error = 0.0;
   REQUIRE(mesh.DerefineByError(error, 1.0));
   fes.Update();
   field.Update();
   REQUIRE(mesh.GetNE() == coarse_ne);
   REQUIRE(fes.GetElementOrder(coarse_ne - 1) == pyr_order);
   if (arbitrary_fine)
   {
      REQUIRE(IntegrateGridFunction(mesh, field) ==
              MFEM_Approx(fine_integral, 1e-10));
   }
   else
   {
      field -= original_values;
      REQUIRE(field.Normlinf() == MFEM_Approx(0.0, 1e-10));
   }
}

TEST_CASE("NCMesh high-order pyramid affine interpolation", "[NCMesh]")
{
   const int pyr_type = GENERATE(0, 1);
   const int order = GENERATE(3, 4);
   CAPTURE(pyr_type, order);
   Mesh mesh("../../data/ref-pyramid.mesh", 1, 1);
   H1_FECollection fec(order, 3, BasisType::GaussLobatto, pyr_type);
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction field(&fes);
   FunctionCoefficient affine([](const Vector &x)
   {
      return 1.0 + x[0] - 2.0*x[1] + 3.0*x[2];
   });
   field.ProjectCoefficient(affine);
   real_t coarse_error = 0.0;
   ElementTransformation &coarse_trans = *mesh.GetElementTransformation(0);
   const IntegrationRule &coarse_ir = IntRules.Get(Geometry::PYRAMID, 6);
   for (int q = 0; q < coarse_ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = coarse_ir.IntPoint(q);
      const real_t value = field.GetValue(0, ip);
      Vector x(3);
      coarse_trans.Transform(ip, x);
      coarse_error = std::max(coarse_error,
                              std::abs(value - (1.0 + x[0] - 2.0*x[1] +
                                                3.0*x[2])));
   }
   REQUIRE(coarse_error == MFEM_Approx(0.0, 1e-10));
   mesh.EnsureNCMesh(true);
   Array<Refinement> refs(1);
   refs[0].Set(0, Refinement::XYZ);
   mesh.GeneralRefinement(refs, 1);
   fes.Update();
   field.Update();
   real_t max_error = 0.0;
   for (int elem = 0; elem < mesh.GetNE(); elem++)
   {
      ElementTransformation &trans = *mesh.GetElementTransformation(elem);
      const Geometry::Type geom = mesh.GetElementBaseGeometry(elem);
      const FiniteElement &fe = *fes.GetFE(elem);
      real_t node_error = 0.0;
      for (int n = 0; n < fe.GetDof(); n++)
      {
         const IntegrationPoint &ip = fe.GetNodes().IntPoint(n);
         const real_t value = field.GetValue(elem, ip);
         Vector x(3);
         trans.Transform(ip, x);
         node_error = std::max(node_error,
                               std::abs(value - (1.0 + x[0] - 2.0*x[1] +
                                                 3.0*x[2])));
      }
      CAPTURE(elem, geom);
      REQUIRE(node_error == MFEM_Approx(0.0, 1e-10));
      const IntegrationRule &ir = IntRules.Get(geom, 6);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         const real_t value = field.GetValue(elem, ip);
         Vector x(3);
         trans.Transform(ip, x);
         max_error = std::max(max_error,
                              std::abs(value - (1.0 + x[0] - 2.0*x[1] +
                                                3.0*x[2])));
      }
   }
   REQUIRE(max_error == MFEM_Approx(0.0, 1e-10));
}


#ifdef MFEM_USE_MPI

TEST_CASE("ParNCMesh geometry-specific refinement patterns",
          "[Parallel], [NCMesh]")
{
   if (Mpi::WorldSize() < 2) { return; }

   Mesh mesh = Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL);
   mesh.EnsureNCMesh();

   const int partition[] = {0, 1};
   ParMesh pmesh(MPI_COMM_WORLD, mesh, partition);
   Array<Refinement> refs;
   if (Mpi::WorldRank() == 0)
   {
      refs.Append(Refinement(0, Refinement::Quadrilateral::SPLIT_Y, 0.25));
   }
   pmesh.GeneralRefinement(refs);

   REQUIRE(pmesh.GetGlobalNE() == 3);
}

TEST_CASE("ParNCMesh geometry-specific 2D patterns", "[Parallel][NCMesh]")
{
   if (Mpi::WorldSize() != 2) { return; }
   const int case_index = GENERATE(range(0, num_geometry_specific_2d_patterns));
   const FacePatternCase &test = geometry_specific_2d_patterns[case_index];
   CAPTURE(test.geometry, test.pattern.GetId());
   const bool quad = test.geometry == Geometry::SQUARE;
   Mesh serial_mesh = Mesh::MakeCartesian2D(
                         quad ? 2 : 1, 1,
                         quad ? Element::QUADRILATERAL : Element::TRIANGLE);
   serial_mesh.EnsureNCMesh(true);
   int partitioning[2] = {0, 1};
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh, partitioning);
   H1_FECollection fec(2, 2);
   ParFiniteElementSpace fes(&mesh, &fec);
   ParGridFunction field(&fes);
   FunctionCoefficient affine([](const Vector &x)
   {
      return 1.0 + 2.0*x[0] - 3.0*x[1];
   });
   field.ProjectCoefficient(affine);

   Array<Refinement> refs;
   if (Mpi::WorldRank() == 0)
   {
      REQUIRE(mesh.GetNE() == 1);
      refs.Append(Refinement(0, test.pattern));
   }
   mesh.GeneralRefinement(refs);
   REQUIRE(mesh.GetGlobalNE() == test.children + 1);
   REQUIRE(mesh.CheckElementOrientation(false) == 0);
   fes.Update();
   field.Update();
   ParGridFunction exact(&fes);
   exact.ProjectCoefficient(affine);
   exact -= field;
   REQUIRE(exact.Normlinf() == MFEM_Approx(0.0, EPS));

   Vector errors(mesh.GetNE());
   errors = 0.0;
   REQUIRE(mesh.DerefineByError(errors, 1.0));
   REQUIRE(mesh.GetGlobalNE() == 2);
}

TEST_CASE("ParNCMesh anisotropic tet shared face", "[Parallel][NCMesh]")
{
   if (Mpi::WorldSize() != 2) { return; }
   const char ref_type = GENERATE(
                            Refinement::X, Refinement::Y, Refinement::Z,
                            Refinement::Tetrahedron::EDGE_12_2_TETS.GetId(),
                            Refinement::Tetrahedron::EDGE_13_2_TETS.GetId(),
                            Refinement::Tetrahedron::EDGE_23_2_TETS.GetId());
   CAPTURE(ref_type);
   Mesh serial_mesh(3, 5, 2, 0);
   serial_mesh.AddVertex(0.0, 0.0, 0.0);
   serial_mesh.AddVertex(1.0, 0.0, 0.0);
   serial_mesh.AddVertex(0.0, 1.0, 0.0);
   serial_mesh.AddVertex(0.0, 0.0, 1.0);
   if (ref_type == Refinement::Z ||
       ref_type == Refinement::Tetrahedron::EDGE_13_2_TETS.GetId())
   {
      serial_mesh.AddVertex(0.0, -1.0, 0.0);
   }
   else if (ref_type == Refinement::Tetrahedron::EDGE_23_2_TETS.GetId())
   {
      serial_mesh.AddVertex(-1.0, 0.0, 0.0);
   }
   else
   {
      serial_mesh.AddVertex(0.0, 0.0, -1.0);
   }
   serial_mesh.AddTet(0, 1, 2, 3);
   if (ref_type == Refinement::Z ||
       ref_type == Refinement::Tetrahedron::EDGE_13_2_TETS.GetId())
   {
      serial_mesh.AddTet(0, 1, 3, 4);
   }
   else if (ref_type == Refinement::Tetrahedron::EDGE_23_2_TETS.GetId())
   {
      serial_mesh.AddTet(0, 3, 2, 4);
   }
   else
   {
      serial_mesh.AddTet(0, 2, 1, 4);
   }
   serial_mesh.FinalizeTetMesh(1, 0, true);
   serial_mesh.EnsureNCMesh(true);
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   H1_FECollection fec(2, 3);
   ParFiniteElementSpace fes(&mesh, &fec);
   ParGridFunction field(&fes);
   FunctionCoefficient affine([](const Vector &x)
   {
      return 1.0 + x[0] - 2.0*x[1] + 3.0*x[2];
   });
   field.ProjectCoefficient(affine);

   Array<Refinement> refs;
   if (Mpi::WorldRank() == 0)
   {
      REQUIRE(mesh.GetNE() == 1);
      refs.Append(TetRefinement(0, ref_type));
   }
   mesh.GeneralRefinement(refs);
   REQUIRE(mesh.GetGlobalNE() == 3);
   REQUIRE(mesh.CheckElementOrientation(false) == 0);
   REQUIRE(mesh.CheckBdrElementOrientation(false) == 0);
   fes.Update();
   field.Update();
   ParGridFunction exact(&fes);
   exact.ProjectCoefficient(affine);
   exact -= field;
   REQUIRE(exact.Normlinf() == MFEM_Approx(0.0, EPS));

   Vector errors(mesh.GetNE());
   errors = 0.0;
   REQUIRE(mesh.DerefineByError(errors, 1.0));
   REQUIRE(mesh.GetGlobalNE() == 2);
   fes.Update();
   field.Update();
   ParGridFunction coarse_exact(&fes);
   coarse_exact.ProjectCoefficient(affine);
   coarse_exact -= field;
   REQUIRE(coarse_exact.Normlinf() == MFEM_Approx(0.0, EPS));
}

TEST_CASE("ParNCMesh anisotropic triangle hanging edge", "[Parallel][NCMesh]")
{
   if (Mpi::WorldSize() != 2) { return; }
   const int ref_type = GENERATE(Refinement::X, Refinement::Y,
                                 Refinement::Z);
   CAPTURE(ref_type);

   Mesh serial_mesh = Mesh::MakeCartesian2D(1, 1, Element::TRIANGLE);
   serial_mesh.EnsureNCMesh(true);
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   H1_FECollection fec(2, 2);
   ParFiniteElementSpace fes(&mesh, &fec);
   ParGridFunction field(&fes);
   FunctionCoefficient affine([](const Vector &x)
   {
      return 1.0 + 2.0*x[0] - 3.0*x[1];
   });
   field.ProjectCoefficient(affine);

   Array<Refinement> refs;
   if (Mpi::WorldRank() == 0)
   {
      REQUIRE(mesh.GetNE() == 1);
      refs.Append(Refinement(0, ref_type));
   }
   mesh.GeneralRefinement(refs);
   REQUIRE(mesh.GetGlobalNE() == 3);
   REQUIRE(mesh.CheckElementOrientation(false) == 0);
   REQUIRE(mesh.CheckBdrElementOrientation(false) == 0);
   fes.Update();
   field.Update();
   ParGridFunction exact(&fes);
   exact.ProjectCoefficient(affine);
   exact -= field;
   REQUIRE(exact.Normlinf() == MFEM_Approx(0.0, EPS));

   Vector errors(mesh.GetNE());
   errors = 0.0;
   REQUIRE(mesh.DerefineByError(errors, 1.0));
   REQUIRE(mesh.GetGlobalNE() == 2);
   fes.Update();
   field.Update();
   ParGridFunction coarse_exact(&fes);
   coarse_exact.ProjectCoefficient(affine);
   coarse_exact -= field;
   REQUIRE(coarse_exact.Normlinf() == MFEM_Approx(0.0, EPS));
}

TEST_CASE("ParNCMesh mixed pyramid derefinement", "[Parallel][NCMesh]")
{
   const int nranks = Mpi::WorldSize();
   if (nranks != 2 && nranks != 3) { return; }
   const bool continuous = GENERATE(false, true);
   const bool variable_order = GENERATE(false, true);
   const bool arbitrary_fine = GENERATE(false, true);
   const int redistribution = GENERATE(0, 1, 2);
   const int order = GENERATE(1, 2);
   if (continuous && arbitrary_fine) { return; }
   CAPTURE(continuous, variable_order, arbitrary_fine, redistribution, order);

   Mesh serial_mesh("../../data/l-mixed.mesh", 1, 1);
   serial_mesh.EnsureNCMesh(true);
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   H1_FECollection h1_fec(order, 3);
   L2_FECollection l2_fec(order, 3, BasisType::GaussLegendre,
                          FiniteElement::VALUE);
   ParFiniteElementSpace fes(&mesh, continuous ?
                             static_cast<FiniteElementCollection*>(&h1_fec) :
                             static_cast<FiniteElementCollection*>(&l2_fec));
   if (variable_order)
   {
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         if (mesh.GetAttribute(i) == 9)
         {
            fes.SetElementOrder(i, order + 1);
         }
      }
      fes.Update(false);
   }
   ParGridFunction field(&fes);
   FunctionCoefficient affine([](const Vector &x)
   {
      return 1.0 + x[0] - 2.0*x[1] + 3.0*x[2];
   });
   field.ProjectCoefficient(affine);
   const int coarse_global_ne = mesh.GetGlobalNE();

   Array<Refinement> refs;
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      if (mesh.GetAttribute(i) == 9)
      {
         refs.Append(Refinement(i, Refinement::XYZ));
      }
   }
   mesh.GeneralRefinement(refs);
   fes.Update();
   field.Update();
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      if (mesh.GetAttribute(i) == 9)
      {
         REQUIRE(fes.GetElementOrder(i) == order + int(variable_order));
      }
   }
   ParGridFunction fine_exact(&fes);
   fine_exact.ProjectCoefficient(affine);
   fine_exact -= field;
   REQUIRE(fine_exact.Normlinf() == MFEM_Approx(0.0, EPS));

   if (redistribution)
   {
      Array<int> partition(mesh.GetNE());
      const int rank = Mpi::WorldRank();
      int pyramid_child = 0;
      int moved_local = 0;
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         partition[i] = rank;
         if (mesh.GetAttribute(i) == 9)
         {
            const int child = pyramid_child++;
            // The first partition spreads children over all ranks; the
            // second moves children 8 and 9 together.
            if (redistribution == 1)
            {
               partition[i] = child % nranks;
            }
            else if (child >= 5)
            {
               partition[i] = (rank + 1) % nranks;
            }
         }
         moved_local += partition[i] != rank;
      }
      int moved_global = 0;
      MPI_Allreduce(&moved_local, &moved_global, 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
      REQUIRE(moved_global > 0);
      mesh.Rebalance(partition);
      fes.Update();
      field.Update();
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         if (mesh.GetAttribute(i) == 9)
         {
            REQUIRE(fes.GetElementOrder(i) == order + int(variable_order));
         }
      }
      ParGridFunction balanced_exact(&fes);
      balanced_exact.ProjectCoefficient(affine);
      balanced_exact -= field;
      REQUIRE(balanced_exact.Normlinf() == MFEM_Approx(0.0, 1e-10));
   }

   real_t fine_integral = 0.0;
   if (arbitrary_fine)
   {
      const int rank = Mpi::WorldRank();
      for (int i = 0; i < field.Size(); i++)
      {
         field(i) = (i % 13 - 6) / 9.0 + rank / 10.0;
      }
      const real_t local_integral = IntegrateGridFunction(mesh, field);
      MPI_Allreduce(&local_integral, &fine_integral, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
   }

   Vector error(mesh.GetNE());
   error = 0.0;
   REQUIRE(mesh.DerefineByError(error, 1.0));
   fes.Update();
   field.Update();
   REQUIRE(mesh.GetGlobalNE() == coarse_global_ne);
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      if (mesh.GetAttribute(i) == 9)
      {
         REQUIRE(fes.GetElementOrder(i) == order + int(variable_order));
      }
   }
   if (arbitrary_fine)
   {
      const real_t local_integral = IntegrateGridFunction(mesh, field);
      real_t coarse_integral = 0.0;
      MPI_Allreduce(&local_integral, &coarse_integral, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
      REQUIRE(coarse_integral == MFEM_Approx(fine_integral, 1e-10));
   }
   else
   {
      ParGridFunction coarse_exact(&fes);
      coarse_exact.ProjectCoefficient(affine);
      coarse_exact -= field;
      REQUIRE(coarse_exact.Normlinf() == MFEM_Approx(0.0, 1e-10));
   }
}

TEST_CASE("ParNCMesh anisotropic prism refinement and derefinement in mixed mesh",
          "[Parallel][NCMesh]")
{
   const int nranks = Mpi::WorldSize();
   if (nranks != 2 && nranks != 3) { return; }
   const int prism_attribute = GENERATE(5, 6, 7, 8);
   const int ref_type = GENERATE(Refinement::XY, Refinement::Z);
   const bool redistribute = GENERATE(false, true);
   const bool nested = GENERATE(false, true);
   const bool continuous = GENERATE(false, true);
   const bool variable_order = GENERATE(false, true);
   CAPTURE(prism_attribute, ref_type, redistribute, nested, continuous,
           variable_order);

   Mesh serial_mesh("../../data/l-mixed.mesh", 1, 1);
   serial_mesh.EnsureNCMesh(true);
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   H1_FECollection h1_fec(2, 3);
   L2_FECollection l2_fec(2, 3, BasisType::GaussLegendre,
                          FiniteElement::VALUE);
   ParFiniteElementSpace fes(&mesh, continuous ?
                             static_cast<FiniteElementCollection*>(&h1_fec) :
                             static_cast<FiniteElementCollection*>(&l2_fec));
   if (variable_order)
   {
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         if (mesh.GetAttribute(i) == prism_attribute)
         {
            fes.SetElementOrder(i, 3);
         }
      }
      fes.Update(false);
   }
   auto check_order = [&]()
   {
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         if (mesh.GetAttribute(i) == prism_attribute)
         {
            REQUIRE(fes.GetElementOrder(i) == 2 + int(variable_order));
         }
      }
   };
   ParGridFunction field(&fes);
   FunctionCoefficient affine([](const Vector &x)
   {
      return 1.0 + x[0] - 2.0*x[1] + 3.0*x[2];
   });
   field.ProjectCoefficient(affine);

   real_t original_volume = SummarizeRefinedElements(mesh).volume;
   MPI_Allreduce(MPI_IN_PLACE, &original_volume, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
   Array<Refinement> refs;
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      if (mesh.GetAttribute(i) == prism_attribute)
      {
         REQUIRE(mesh.GetElementBaseGeometry(i) == Geometry::PRISM);
         refs.Append(Refinement(i, ref_type));
      }
   }
   int global_refs = refs.Size();
   MPI_Allreduce(MPI_IN_PLACE, &global_refs, 1, MPI_INT, MPI_SUM,
                 MPI_COMM_WORLD);
   REQUIRE(global_refs == 1);
   mesh.GeneralRefinement(refs);
   fes.Update();
   field.Update();
   check_order();

   int prism_children = 0;
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      if (mesh.GetAttribute(i) == prism_attribute)
      {
         REQUIRE(mesh.GetElementBaseGeometry(i) == Geometry::PRISM);
         prism_children++;
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &prism_children, 1, MPI_INT, MPI_SUM,
                 MPI_COMM_WORLD);
   REQUIRE(prism_children == (ref_type == Refinement::XY ? 4 : 2));

   if (redistribute)
   {
      Array<int> partition(mesh.GetNE());
      const int rank = Mpi::WorldRank();
      int child = 0;
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         partition[i] = mesh.GetAttribute(i) == prism_attribute ?
                        (rank + 1 + child++) % nranks : rank;
      }
      mesh.Rebalance(partition);
      fes.Update();
      field.Update();
      check_order();
   }

   if (nested)
   {
      const int rank = Mpi::WorldRank();
      int owner = nranks;
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         if (mesh.GetAttribute(i) == prism_attribute) { owner = rank; break; }
      }
      MPI_Allreduce(MPI_IN_PLACE, &owner, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD);
      REQUIRE(owner < nranks);
      refs.SetSize(0);
      if (rank == owner)
      {
         for (int i = 0; i < mesh.GetNE(); i++)
         {
            if (mesh.GetAttribute(i) == prism_attribute)
            {
               const int second_type = ref_type == Refinement::XY ?
                                       Refinement::Z : Refinement::XY;
               refs.Append(Refinement(i, second_type));
               break;
            }
         }
      }
      mesh.GeneralRefinement(refs);
      fes.Update();
      field.Update();
      check_order();
   }

   real_t refined_volume = SummarizeRefinedElements(mesh).volume;
   MPI_Allreduce(MPI_IN_PLACE, &refined_volume, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
   REQUIRE(refined_volume == MFEM_Approx(original_volume));
   REQUIRE(mesh.CheckElementOrientation(false) == 0);
   REQUIRE(mesh.CheckBdrElementOrientation(false) == 0);
   ParGridFunction exact(&fes);
   exact.ProjectCoefficient(affine);
   exact -= field;
   REQUIRE(exact.Normlinf() == MFEM_Approx(0.0, EPS));

   real_t fine_integral = 0.0;
   if (!continuous)
   {
      const int rank = Mpi::WorldRank();
      for (int i = 0; i < field.Size(); i++)
      {
         field(i) = (i % 13 - 6) / 9.0 + rank / 10.0;
      }
      const real_t local_integral = IntegrateGridFunction(mesh, field);
      MPI_Allreduce(&local_integral, &fine_integral, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
   }

   for (int level = 0; level < (nested ? 2 : 1); level++)
   {
      Vector error(mesh.GetNE());
      error = 0.0;
      REQUIRE(mesh.DerefineByError(error, 1.0));
      fes.Update();
      field.Update();
      check_order();
   }
   REQUIRE(mesh.GetGlobalNE() == 9);
   real_t restored_volume = SummarizeRefinedElements(mesh).volume;
   MPI_Allreduce(MPI_IN_PLACE, &restored_volume, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
   REQUIRE(restored_volume == MFEM_Approx(original_volume));
   if (continuous)
   {
      ParGridFunction restored_exact(&fes);
      restored_exact.ProjectCoefficient(affine);
      restored_exact -= field;
      REQUIRE(restored_exact.Normlinf() == MFEM_Approx(0.0, EPS));
   }
   else
   {
      const real_t local_integral = IntegrateGridFunction(mesh, field);
      real_t restored_integral = 0.0;
      MPI_Allreduce(&local_integral, &restored_integral, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
      REQUIRE(restored_integral == MFEM_Approx(fine_integral, 1e-10));
   }
}

TEST_CASE("ParNCMesh anisotropic hex derefinement", "[Parallel][NCMesh]")
{
   if (Mpi::WorldSize() != 2) { return; }
   const int ref_type = GENERATE(Refinement::X, Refinement::Y, Refinement::Z,
                                 Refinement::XY, Refinement::XZ, Refinement::YZ,
                                 Refinement::XYZ);
   const bool continuous = GENERATE(false, true);
   CAPTURE(ref_type, continuous);

   Mesh serial_mesh = Mesh::MakeCartesian3D(
                         2, 1, 1, Element::HEXAHEDRON, 2.0, 1.0, 1.0);
   serial_mesh.EnsureNCMesh(true);
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   H1_FECollection h1_fec(2, 3);
   L2_FECollection l2_fec(2, 3, BasisType::GaussLegendre,
                          FiniteElement::VALUE);
   ParFiniteElementSpace fes(&mesh, continuous ?
                             static_cast<FiniteElementCollection*>(&h1_fec) :
                             static_cast<FiniteElementCollection*>(&l2_fec));
   ParGridFunction field(&fes);
   FunctionCoefficient affine([](const Vector &x)
   {
      return 1.0 + x[0] - 2.0*x[1] + 3.0*x[2];
   });
   field.ProjectCoefficient(affine);

   Array<Refinement> refs;
   if (Mpi::WorldRank() == 0)
   {
      REQUIRE(mesh.GetNE() == 1);
      refs.Append(Refinement(0, ref_type));
   }
   mesh.GeneralRefinement(refs);
   fes.Update();
   field.Update();

   real_t fine_integral = 0.0;
   if (!continuous)
   {
      const int rank = Mpi::WorldRank();
      for (int i = 0; i < field.Size(); i++)
      {
         field(i) = (i % 13 - 6) / 9.0 + rank / 10.0;
      }
      const real_t local_integral = IntegrateGridFunction(mesh, field);
      MPI_Allreduce(&local_integral, &fine_integral, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
   }

   Vector error(mesh.GetNE());
   error = 0.0;
   REQUIRE(mesh.DerefineByError(error, 1.0));
   fes.Update();
   field.Update();
   REQUIRE(mesh.GetGlobalNE() == 2);
   if (continuous)
   {
      ParGridFunction exact(&fes);
      exact.ProjectCoefficient(affine);
      exact -= field;
      REQUIRE(exact.Normlinf() == MFEM_Approx(0.0, EPS));
   }
   else
   {
      const real_t local_integral = IntegrateGridFunction(mesh, field);
      real_t restored_integral = 0.0;
      MPI_Allreduce(&local_integral, &restored_integral, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
      REQUIRE(restored_integral == MFEM_Approx(fine_integral, 1e-10));
   }
}

TEST_CASE("ParNCMesh mixed pyramid vector L2 derefinement",
          "[Parallel][NCMesh]")
{
   const int nranks = Mpi::WorldSize();
   if (nranks != 2 && nranks != 3) { return; }
   const bool variable_order = GENERATE(false, true);
   const bool redistribute = GENERATE(false, true);
   const Ordering::Type ordering = GENERATE(Ordering::byNODES,
                                            Ordering::byVDIM);
   CAPTURE(variable_order, redistribute, ordering);

   Mesh serial_mesh("../../data/l-mixed.mesh", 1, 1);
   serial_mesh.EnsureNCMesh(true);
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   L2_FECollection fec(2, 3, BasisType::GaussLegendre,
                       FiniteElement::VALUE);
   ParFiniteElementSpace fes(&mesh, &fec, 3, ordering);
   if (variable_order)
   {
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         if (mesh.GetAttribute(i) == 9) { fes.SetElementOrder(i, 3); }
      }
      fes.Update(false);
   }
   ParGridFunction field(&fes);
   field = 0.0;
   const int coarse_global_ne = mesh.GetGlobalNE();

   Array<Refinement> refs;
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      if (mesh.GetAttribute(i) == 9)
      {
         refs.Append(Refinement(i, Refinement::XYZ));
      }
   }
   mesh.GeneralRefinement(refs);
   fes.Update();
   field.Update();

   if (redistribute)
   {
      Array<int> partition(mesh.GetNE());
      const int rank = Mpi::WorldRank();
      int pyramid_child = 0;
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         partition[i] = rank;
         if (mesh.GetAttribute(i) == 9)
         {
            partition[i] = pyramid_child++ % nranks;
         }
      }
      mesh.Rebalance(partition);
      fes.Update();
      field.Update();
   }

   for (int i = 0; i < field.Size(); i++)
   {
      field(i) = (i % 17 - 8) / 11.0 + Mpi::WorldRank() / 10.0;
   }
   Vector local_fine = IntegrateGridFunctionComponents(mesh, field);
   Vector global_fine(3);
   MPI_Allreduce(local_fine.GetData(), global_fine.GetData(), 3,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);

   Vector error(mesh.GetNE());
   error = 0.0;
   REQUIRE(mesh.DerefineByError(error, 1.0));
   fes.Update();
   field.Update();
   REQUIRE(mesh.GetGlobalNE() == coarse_global_ne);
   Vector local_coarse = IntegrateGridFunctionComponents(mesh, field);
   Vector global_coarse(3);
   MPI_Allreduce(local_coarse.GetData(), global_coarse.GetData(), 3,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
   for (int component = 0; component < 3; component++)
   {
      REQUIRE(global_coarse(component) ==
              MFEM_Approx(global_fine(component), 1e-10));
   }
}

TEST_CASE("ParNCMesh nested mixed pyramid derefinement",
          "[Parallel][NCMesh]")
{
   const int nranks = Mpi::WorldSize();
   if (nranks != 2 && nranks != 3) { return; }
   const bool continuous = GENERATE(false, true);
   const bool variable_order = GENERATE(false, true);
   const bool redistribute = GENERATE(false, true);
   CAPTURE(continuous, variable_order, redistribute);

   Mesh serial_mesh("../../data/l-mixed.mesh", 1, 1);
   serial_mesh.EnsureNCMesh(true);
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   H1_FECollection h1_fec(2, 3);
   L2_FECollection l2_fec(2, 3, BasisType::GaussLegendre,
                          FiniteElement::VALUE);
   ParFiniteElementSpace fes(&mesh, continuous ?
                             static_cast<FiniteElementCollection*>(&h1_fec) :
                             static_cast<FiniteElementCollection*>(&l2_fec));
   if (variable_order)
   {
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         if (mesh.GetAttribute(i) == 9) { fes.SetElementOrder(i, 3); }
      }
      fes.Update(false);
   }
   ParGridFunction field(&fes);
   FunctionCoefficient affine([](const Vector &x)
   {
      return 1.0 + x[0] - 2.0*x[1] + 3.0*x[2];
   });
   field.ProjectCoefficient(affine);
   const int coarse_global_ne = mesh.GetGlobalNE();

   Array<Refinement> refs;
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      if (mesh.GetAttribute(i) == 9)
      {
         refs.Append(Refinement(i, Refinement::XYZ));
      }
   }
   mesh.GeneralRefinement(refs);
   fes.Update();
   field.Update();
   REQUIRE(mesh.GetGlobalNE() == coarse_global_ne + 9);

   refs.SetSize(0);
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      if (mesh.GetAttribute(i) == 9 &&
          mesh.GetElementBaseGeometry(i) == Geometry::TETRAHEDRON)
      {
         refs.Append(Refinement(i, Refinement::XYZ));
         break;
      }
   }
   int local_nested = refs.Size(), global_nested = 0;
   MPI_Allreduce(&local_nested, &global_nested, 1, MPI_INT, MPI_SUM,
                 MPI_COMM_WORLD);
   REQUIRE(global_nested == 1);
   mesh.GeneralRefinement(refs);
   fes.Update();
   field.Update();
   REQUIRE(mesh.GetGlobalNE() == coarse_global_ne + 16);

   if (redistribute)
   {
      Array<int> partition(mesh.GetNE());
      const int rank = Mpi::WorldRank();
      int pyramid_child = 0;
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         partition[i] = rank;
         if (mesh.GetAttribute(i) == 9)
         {
            partition[i] = pyramid_child++ % nranks;
         }
      }
      mesh.Rebalance(partition);
      fes.Update();
      field.Update();
   }

   auto check_field = [&]()
   {
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         if (mesh.GetAttribute(i) == 9)
         {
            REQUIRE(fes.GetElementOrder(i) == 2 + int(variable_order));
         }
      }
      ParGridFunction exact(&fes);
      exact.ProjectCoefficient(affine);
      exact -= field;
      REQUIRE(exact.Normlinf() == MFEM_Approx(0.0, 1e-10));
   };
   check_field();

   for (int level = 0; level < 2; level++)
   {
      Vector error(mesh.GetNE());
      error = 0.0;
      REQUIRE(mesh.DerefineByError(error, 1.0));
      fes.Update();
      field.Update();
      REQUIRE(mesh.GetGlobalNE() == coarse_global_ne + (level == 0 ? 9 : 0));
      check_field();
   }
}

// Test case: Verify that a conforming mesh yields the same norm for the
//            assembled diagonal with PA when using the standard (conforming)
//            Mesh vs. the corresponding (non-conforming) NCMesh. (note:
//            permutations of the values in the diagonal are expected)
TEST_CASE("pNCMesh PA diagonal",  "[Parallel], [NCMesh]")
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   SECTION("Quad pmesh")
   {
      int ne = 2;
      Mesh mesh = Mesh::MakeCartesian2D(
                     ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
      Mesh nc_mesh = Mesh::MakeCartesian2D(
                        ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
      nc_mesh.EnsureNCMesh();

      mesh.UniformRefinement();
      nc_mesh.UniformRefinement();

      ParMesh pmesh(MPI_COMM_WORLD, mesh);
      ParMesh nc_pmesh(MPI_COMM_WORLD, nc_mesh);

      int dim = 2;
      for (int order = 1; order <= 3; ++order)
      {
         ND_FECollection fec(order, dim);

         ParFiniteElementSpace pfes(&pmesh, &fec);
         ParFiniteElementSpace nc_pfes(&nc_pmesh, &fec);

         ParBilinearForm a(&pfes);
         ParBilinearForm nc_a(&nc_pfes);

         a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         nc_a.SetAssemblyLevel(AssemblyLevel::PARTIAL);

         ConstantCoefficient coef(1.0);
         a.AddDomainIntegrator(new CurlCurlIntegrator(coef));
         nc_a.AddDomainIntegrator(new CurlCurlIntegrator(coef));

         a.Assemble();
         nc_a.Assemble();

         Vector diag(pfes.GetTrueVSize());
         Vector nc_diag(nc_pfes.GetTrueVSize());
         a.AssembleDiagonal(diag);
         nc_a.AssembleDiagonal(nc_diag);

         real_t diag_lsum = diag.Sum(), nc_diag_lsum = nc_diag.Sum();
         real_t diag_gsum = 0.0, nc_diag_gsum = 0.0;
         MPI_Allreduce(&diag_lsum, &diag_gsum, 1,
                       MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
         MPI_Allreduce(&nc_diag_lsum, &nc_diag_gsum, 1,
                       MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);

         real_t error = fabs(diag_gsum - nc_diag_gsum);
         CAPTURE(order, error);
         REQUIRE(error == MFEM_Approx(0.0, EPS));
         MPI_Barrier(MPI_COMM_WORLD);
      }
   }

   SECTION("Hexa pmesh")
   {
      int ne = 2;
      Mesh mesh = Mesh::MakeCartesian3D(
                     ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
      Mesh nc_mesh = Mesh::MakeCartesian3D(
                        ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
      nc_mesh.EnsureNCMesh();

      mesh.UniformRefinement();
      nc_mesh.UniformRefinement();

      ParMesh pmesh(MPI_COMM_WORLD, mesh);
      ParMesh nc_pmesh(MPI_COMM_WORLD, nc_mesh);

      int dim = 3;
      for (int order = 1; order <= 3; ++order)
      {
         ND_FECollection fec(order, dim);

         ParFiniteElementSpace pfes(&pmesh, &fec);
         ParFiniteElementSpace nc_pfes(&nc_pmesh, &fec);

         ParBilinearForm a(&pfes);
         ParBilinearForm nc_a(&nc_pfes);

         a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         nc_a.SetAssemblyLevel(AssemblyLevel::PARTIAL);

         ConstantCoefficient coef(1.0);
         a.AddDomainIntegrator(new CurlCurlIntegrator(coef));
         nc_a.AddDomainIntegrator(new CurlCurlIntegrator(coef));

         a.Assemble();
         nc_a.Assemble();

         Vector diag(pfes.GetTrueVSize());
         Vector nc_diag(nc_pfes.GetTrueVSize());
         a.AssembleDiagonal(diag);
         nc_a.AssembleDiagonal(nc_diag);

         real_t diag_lsum = diag.Sum(), nc_diag_lsum = nc_diag.Sum();
         real_t diag_gsum = 0.0, nc_diag_gsum = 0.0;
         MPI_Allreduce(&diag_lsum, &diag_gsum, 1, MPITypeMap<real_t>::mpi_type,
                       MPI_SUM, MPI_COMM_WORLD);
         MPI_Allreduce(&nc_diag_lsum, &nc_diag_gsum, 1,
                       MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);

         CAPTURE(order, diag_gsum, nc_diag_gsum);
         REQUIRE(nc_diag_gsum == MFEM_Approx(diag_gsum));
         MPI_Barrier(MPI_COMM_WORLD);
      }
   }
} // test case

TEST_CASE("ParNCMesh Rebalance preserves element attributes",
          "[Parallel], [NCMesh]")
{
   const int rank = Mpi::WorldRank();
   const int nranks = Mpi::WorldSize();
   if (nranks < 2) { return; }

   auto mesh_fname = GENERATE("../../data/star.mesh",
                              "../../data/fichera.mesh");
   CAPTURE(mesh_fname);

   auto CheckRebalance = [rank, nranks, mesh_fname](bool refine,
                                                    bool custom_partition)
   {
      Mesh mesh(mesh_fname);
      mesh.EnsureNCMesh();
      ParMesh pmesh(MPI_COMM_WORLD, mesh);

      const int attribute = 1234 + (custom_partition ? rank : 0);
      for (int i = 0; i < pmesh.GetNE(); i++)
      {
         pmesh.SetAttribute(i, attribute);
      }
      pmesh.SetAttributes();

      if (refine)
      {
         Array<int> refinements;
         if (pmesh.GetNE() && (custom_partition || rank == 0))
         {
            refinements.Append(0);
         }
         pmesh.GeneralRefinement(refinements);
      }

      int expected_attribute = attribute;
      if (custom_partition)
      {
         // Move every element to the next rank, as in GitHub issue #4009.
         Array<int> partition(pmesh.GetNE());
         partition = (rank + 1) % nranks;
         pmesh.Rebalance(partition);
         expected_attribute = 1234 + (rank + nranks - 1) % nranks;
      }
      else
      {
         pmesh.Rebalance();
      }

      for (int i = 0; i < pmesh.GetNE(); i++)
      {
         CHECK(pmesh.GetAttribute(i) == expected_attribute);
      }
   };

   SECTION("Custom partition, unrefined") { CheckRebalance(false, true); }
   SECTION("Custom partition, refined") { CheckRebalance(true, true); }
   SECTION("Default partition, refined") { CheckRebalance(true, false); }
}

TEST_CASE("EdgeFaceConstraint", "[Parallel], [NCMesh]")
{
   auto exact_soln = [](const Vector& x)
   {
      // sin(|| x - d ||^2) -> non polynomial but very smooth.
      Vector d(3);
      d[0] = -0.5; d[1] = -1; d[2] = -2; // arbitrary
      d -= x;
      return std::sin(d * d);
   };

   SECTION("ReferenceTet")
   {
      constexpr int refining_rank = 0;
      auto smesh = Mesh("../../data/ref-tetrahedron.mesh");

      REQUIRE(smesh.GetNE() == 1);
      {
         // Start the test with two tetrahedra attached by triangle.
         auto single_edge_refine = Array<Refinement>(1);
         single_edge_refine[0].Set(0, Refinement::X);
         smesh.GeneralRefinement(single_edge_refine, 0); // conformal
      }

      REQUIRE(smesh.GetNE() == 2);
      smesh.EnsureNCMesh(true);
      smesh.Finalize();

      auto partition = std::unique_ptr<int[]>(new int[smesh.GetNE()]);
      partition[0] = 0;
      partition[1] = Mpi::WorldSize() > 1 ? 1 : 0;

      auto pmesh = ParMesh(MPI_COMM_WORLD, smesh, partition.get());

      // Construct the NC refined mesh in parallel and serial. Once constructed
      // a global L2 projected solution should match exactly on each.
      Array<int> refines, serial_refines(1);
      if (Mpi::WorldRank() == refining_rank)
      {
         refines.Append(0);
      }

      // Must be called on all ranks as it uses MPI calls internally. All ranks
      // will use the global element number dictated by rank 0 though.
      serial_refines[0] = pmesh.GetGlobalElementNum(0);
      MPI_Bcast(&serial_refines[0], 1, MPI_INT, refining_rank, MPI_COMM_WORLD);

      // Rank 0 refines the parallel mesh, all ranks refine the serial mesh
      smesh.GeneralRefinement(serial_refines, 1); // nonconformal
      pmesh.GeneralRefinement(refines, 1); // nonconformal

      REQUIRE(pmesh.GetGlobalNE() == 8 + 1);
      REQUIRE(smesh.GetNE() == 8 + 1);

      // RT0 has no edge DOFs; skip its empty edge-face constraints.
      {
         RT0_3DFECollection fec;
         FiniteElementSpace fes(&smesh, &fec);
         ParFiniteElementSpace pfes(&pmesh, &fec);
         CHECK(pfes.GlobalTrueVSize() == fes.GetTrueVSize());
      }

      // Each pair of indices here represents sequential element indices to
      // refine. First the i element is refined, then in the resulting mesh the
      // j element is refined. These pairs were arrived at by looping over all
      // possible i,j pairs and checking for the addition of a face-edge
      // constraint.
      std::vector<std::pair<int,int>> indices{{2,13}, {3,13}, {6,2}, {6,3}};

      // Rank 0 has all but one element in the parallel mesh. The remaining
      // element is owned by another processor if the number of ranks is greater
      // than one.
      for (const auto &ij : indices)
      {
         int i = ij.first;
         int j = ij.second;
         if (Mpi::WorldRank() == refining_rank)
         {
            refines[0] = i;
         }
         // Inform all ranks of the serial mesh
         serial_refines[0] = pmesh.GetGlobalElementNum(i);
         MPI_Bcast(&serial_refines[0], 1, MPI_INT, 0, MPI_COMM_WORLD);

         ParMesh tmp(pmesh);
         tmp.GeneralRefinement(refines);

         REQUIRE(tmp.GetGlobalNE() == 1 + 8 - 1 + 8); // 16 elements

         Mesh stmp(smesh);
         stmp.GeneralRefinement(serial_refines);
         REQUIRE(stmp.GetNE() == 1 + 8 - 1 + 8); // 16 elements

         if (Mpi::WorldRank() == refining_rank)
         {
            refines[0] = j;
         }
         // Inform all ranks of the serial mesh
         serial_refines[0] = tmp.GetGlobalElementNum(j);
         MPI_Bcast(&serial_refines[0], 1, MPI_INT, 0, MPI_COMM_WORLD);

         ParMesh ttmp(tmp);
         ttmp.GeneralRefinement(refines);

         REQUIRE(ttmp.GetGlobalNE() == 1 + 8 - 1 + 8 - 1 + 8); // 23 elements

         Mesh sttmp(stmp);
         sttmp.GeneralRefinement(serial_refines);
         REQUIRE(sttmp.GetNE() == 1 + 8 - 1 + 8 - 1 + 8); // 23 elements

         // Loop over interior faces, fill and check face transform on the
         // serial.
         for (int iface = 0; iface < sttmp.GetNumFaces(); ++iface)
         {
            const auto face_transform = sttmp.GetFaceElementTransformations(iface);
            CHECK(face_transform->CheckConsistency(0) < 1e-12);
         }

         for (int iface = 0; iface < ttmp.GetNumFacesWithGhost(); ++iface)
         {
            const auto face_transform = ttmp.GetFaceElementTransformations(iface);
            CHECK(face_transform->CheckConsistency(0) < 1e-12);
         }

         // Use P4 to ensure there's a few fully interior DOF.
         {
            auto error = CheckL2Projection(ttmp, sttmp, 4, exact_soln);
            real_t constexpr tol = 1e-9;
            CHECK(std::abs(error[1] - error[0]) < tol);
         }
         ttmp.ExchangeFaceNbrData();
         ttmp.Rebalance();
         {
            auto error = CheckL2Projection(ttmp, sttmp, 4, exact_soln);
            real_t constexpr tol = 1e-9;
            CHECK(std::abs(error[1] - error[0]) < tol);
         }
      }
   }

   auto check_serial_parallel_h1_equivalance = [](Mesh &smesh)
   {
      constexpr int dim = 3;
      constexpr int order = 2;
      H1_FECollection nd_fec(order, dim);
      FiniteElementSpace fes(&smesh, &nd_fec);
      const auto serial_ntdof = fes.GetTrueVSize();

      ParMesh mesh(MPI_COMM_WORLD, smesh);
      ParFiniteElementSpace pfes(&mesh, &nd_fec);
      const auto parallel_ntdof = pfes.GlobalTrueVSize();

      // If nc constraints have been observed correctly, the number of true dof
      // in parallel should match the number of true dof in serial. If the
      // number of parallel dofs is greater, then a slave constraint has not
      // been fully labeled.
      CHECK(serial_ntdof == parallel_ntdof);
   };

   auto check_serial_parallel_nd_equivalence = [](Mesh &smesh)
   {
      constexpr int dim = 3;
      constexpr int order = 1;
      ND_FECollection nd_fec(order, dim);
      FiniteElementSpace fes(&smesh, &nd_fec);
      const auto serial_ntdof = fes.GetTrueVSize();

      ParMesh mesh(MPI_COMM_WORLD, smesh);
      ParFiniteElementSpace pfes(&mesh, &nd_fec);
      const auto parallel_ntdof = pfes.GlobalTrueVSize();

      // If nc constraints have been observed correctly, the number of true dof
      // in parallel should match the number of true dof in serial. If the
      // number of parallel dofs is greater, then a slave constraint has not
      // been fully labeled.
      CHECK(serial_ntdof == parallel_ntdof);
   };

   SECTION("LevelTwoRefinement")
   {
      Mesh smesh("../../data/ref-tetrahedron.mesh");
      Array<Refinement> aniso_ref(1);
      aniso_ref[0].Set(0, Refinement::X);
      smesh.GeneralRefinement(aniso_ref);
      smesh.UniformRefinement();
      smesh.EnsureNCMesh(true);
      Array<int> el_to_refine(1);

      for (int n = 0; n < smesh.GetNE(); n++)
      {
         Mesh smesh2(smesh);
         el_to_refine[0] = n;
         smesh2.GeneralRefinement(el_to_refine);
         for (int m = 0; m < smesh2.GetNE(); m++)
         {
            Mesh smesh3(smesh2);
            el_to_refine[0] = m;
            smesh3.GeneralRefinement(el_to_refine);
            CAPTURE(n,m);
            check_serial_parallel_nd_equivalence(smesh3);
            check_serial_parallel_h1_equivalance(smesh3);
         }
      }
   }

   SECTION("EdgeCasePartition")
   {
      Mesh smesh("../../data/ref-tetrahedron.mesh");
      smesh.UniformRefinement();
      smesh.EnsureNCMesh(true);
      Array<int> el_to_refine(1);

      el_to_refine[0] = 0;
      smesh.GeneralRefinement(el_to_refine);

      // This particular partition was found by brute force search. The default
      // rebalancing can in rare cases produce similar local patterns,
      // particularly for highly adapted meshes.
      auto partition = std::unique_ptr<int[]>(new int[smesh.GetNE()]);
      if (Mpi::WorldSize() > 1)
      {
         auto bad_partition = std::vector<int> {0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0};
         std::copy(bad_partition.begin(), bad_partition.end(), partition.get());
      }
      else
      {
         for (int i = 0; i < smesh.GetNE(); i++)
         {
            partition[i] = 0;
         }
      }
      ParMesh pmesh(MPI_COMM_WORLD, smesh, partition.get());
      {
         constexpr int dim = 3;
         constexpr int order = 1;
         ND_FECollection nd_fec(order, dim);
         FiniteElementSpace fes(&smesh, &nd_fec);
         const auto serial_ntdof = fes.GetTrueVSize();
         ParFiniteElementSpace pfes(&pmesh, &nd_fec);
         pfes.ExchangeFaceNbrData();
         const auto parallel_ntdof = pfes.GlobalTrueVSize();
         CHECK(serial_ntdof == parallel_ntdof);
      }
      for (int order = 1; order <= 4; order++)
      {
         CAPTURE(order);
         auto error = CheckL2Projection(pmesh, smesh, order, exact_soln);
         real_t constexpr tol = 1e-9;
         CHECK(std::abs(error[1] - error[0]) < tol);
      }
   }

} // test case

TEST_CASE("TetEdgeFaceCommunication", "[Parallel], [NCMesh]")
{
   if (Mpi::WorldSize() < 3) { return; }

   Mesh smesh("../../data/ref-tetrahedron.mesh");
   smesh.UniformRefinement();
   smesh.EnsureNCMesh(true);
   Array<int> ref(1);
   ref[0] = 0;
   smesh.GeneralRefinement(ref);
   REQUIRE(smesh.GetNE() == 15);

   // On rank 1, SelfGroupId still means self. No faces should be shared.
   {
      Array<int> local_partition(smesh.GetNE());
      local_partition = 1;
      ParMesh local(MPI_COMM_WORLD, smesh, local_partition.GetData());
      int shared = local.GetNE() ? local.pncmesh->GetSharedFaces().masters.Size() : 0;
      MPI_Allreduce(MPI_IN_PLACE, &shared, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      CHECK(shared == 0);
   }

   // Rank 0 owns an interior edge of rank 2's slave face, but no local
   // slave face constrains that edge.
   constexpr std::array<int, 15> partition =
   {0, 1, 2, 1, 0, 1, 1, 0, 2, 2, 0, 2, 1, 0, 0};
   ParMesh pmesh(MPI_COMM_WORLD, smesh, partition.data());
   auto &nc = *pmesh.pncmesh;
   const auto &faces = nc.GetFaceList();
   const auto &edges = nc.GetEdgeList();

   int checked = 0, remote = 0, missing = 0;
   for (const auto &master : faces.masters)
   {
      if (nc.IsGhost(2, master.index)) { continue; }
      int mv[4], me[4], mo[4];
      const int nmv = nc.GetFaceVerticesEdges(master, mv, me, mo);
      for (int i = master.slaves_begin; i < master.slaves_end; i++)
      {
         const auto &slave = faces.slaves[i];
         if (slave.index < 0 || nc.IsGhost(2, slave.index)) { continue; }
         int sv[4], se[4], so[4];
         const int nsv = nc.GetFaceVerticesEdges(slave, sv, se, so);
         for (int j = 0; j < nsv; j++)
         {
            const int edge = se[j];
            if (std::find(me, me + nmv, edge) != me + nmv) { continue; }
            const auto type = edges.GetMeshIdAndType(edge).type;
            if (type != NCMesh::NCList::MeshIdType::MASTER &&
                type != NCMesh::NCList::MeshIdType::CONFORMING) { continue; }

            // Use the slave face's edges, not the edge-face records.
            const int owner = nc.GetGroup(nc.GetEntityOwnerId(1, edge))[0];
            checked++;
            remote += (owner != Mpi::WorldRank());
            const auto group = nc.GetEntityGroupId(2, master.index);
            missing += !nc.GroupContains(group, owner);
         }
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &checked, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &remote, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &missing, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   REQUIRE(checked > 0);
   // The partition must exercise an edge owned by another rank, which is the
   // case that previously left that rank out of the master face's group.
   REQUIRE(remote > 0);
   // Fail on all ranks before a missing row can hang space construction.
   REQUIRE(missing == 0);

   ND_FECollection fec(2, 3);
   FiniteElementSpace fes(&smesh, &fec);
   ParFiniteElementSpace pfes(&pmesh, &fec);
   CHECK(pfes.GlobalTrueVSize() == fes.GetTrueVSize());
}

TEST_CASE("HangingVertexOwnership", "[Parallel], [NCMesh]")
{
   if (Mpi::WorldSize() < 3) { return; }

   auto check_ownership = [](Mesh &smesh, const Array<int> &base)
   {
      std::array<int, 3> labels = {0, 1, 2};
      do
      {
         Array<int> partition(base.Size());
         for (int i = 0; i < partition.Size(); i++)
         {
            partition[i] = labels[base[i]];
         }

         CAPTURE(labels[0], labels[1], labels[2]);
         ParMesh pmesh(MPI_COMM_WORLD, smesh, partition.GetData());
         for (int order : {1, 2, 3})
         {
            CAPTURE(order);
            H1_FECollection fec(order, smesh.Dimension());
            FiniteElementSpace fes(&smesh, &fec);
            ParFiniteElementSpace pfes(&pmesh, &fec);
            CHECK(pfes.GlobalTrueVSize() == fes.GetTrueVSize());
            if (order != 1) { continue; }

            // Compare serial and parallel P R for every P1 nodal basis vector.
            // Affine fields can hide extra true DOFs.
            GridFunction serial_values(&fes);
            Vector true_values;
            int finite = 1;
            real_t error = 0.0;
            for (int i = 0; i < fes.GetVSize(); i++)
            {
               serial_values = 0.0;
               serial_values[i] = 1.0;
               ParGridFunction actual(&pmesh, &serial_values, partition.GetData());
               actual.GetTrueDofs(true_values);
               actual.SetFromTrueDofs(true_values);
               serial_values.GetTrueDofs(true_values);
               serial_values.SetFromTrueDofs(true_values);
               ParGridFunction expected(&pmesh, &serial_values, partition.GetData());
               actual -= expected;
               if (actual.CheckFinite()) { finite = 0; }
               else { error = std::max(error, actual.Normlinf()); }
            }
            MPI_Allreduce(MPI_IN_PLACE, &finite, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPITypeMap<real_t>::mpi_type,
                          MPI_MAX, MPI_COMM_WORLD);
            CHECK(finite);
            CHECK(error == MFEM_Approx(0.0));
         }
      }
      while (std::next_permutation(labels.begin(), labels.end()));
   };

   SECTION("VertexFan")
   {
      Mesh smesh("../../data/ref-tetrahedron.mesh");
      smesh.UniformRefinement();
      smesh.EnsureNCMesh(true);
      smesh.GeneralRefinement(Array<int>({0}));
      REQUIRE(smesh.GetNE() == 15);
      const Array<int> partition({2, 2, 1, 2, 0, 1, 2, 2, 0, 2, 1, 1, 2, 1, 0});
      check_ownership(smesh, partition);
   }

   SECTION("Edge")
   {
      // Child 3 on rank 0 touches the midpoint, but neither half-edge.
      // Rank 1 holds both constraining half-edges.
      Mesh smesh = Mesh::MakeCartesian2D(1, 1, Element::TRIANGLE);
      smesh.EnsureNCMesh(true);
      smesh.GeneralRefinement(Array<int>({0}));
      REQUIRE(smesh.GetNE() == 5);
      const Array<int> partition({1, 1, 2, 0, 2});
      check_ownership(smesh, partition);
   }

   SECTION("Face")
   {
      // Refine both children sharing edge [(0,1/2,0),(1/2,1/2,0)].
      // Its midpoint then depends on the coarse face, not a master edge.
      Mesh smesh(3, 5, 2, 0);
      smesh.AddVertex(0, 0, 0);
      smesh.AddVertex(1, 0, 0);
      smesh.AddVertex(0, 1, 0);
      smesh.AddVertex(0, 0, 1);
      smesh.AddVertex(0, 0, -1);
      smesh.AddTet(0, 1, 2, 3);
      smesh.AddTet(0, 2, 1, 4);
      smesh.FinalizeTetMesh(1, 0, true);
      smesh.EnsureNCMesh(true);
      smesh.GeneralRefinement(Array<int>({0}));
      smesh.GeneralRefinement(Array<int>({2, 7}));
      REQUIRE(smesh.GetNE() == 23);

      // Element 20 touches the coarse face only at (1/4,1/2,0).
      // Rank 1 holds all constraining faces; element 22 is the lower tet.
      Array<int> partition(smesh.GetNE());
      partition = 1;
      partition[20] = 0;
      partition[22] = 2;
      check_ownership(smesh, partition);
   }
}

TEST_CASE("P2Q1PureTetHexPri",  "[Parallel], [NCMesh]")
{
   auto exact_soln = [](const Vector& x)
   {
      // sin(|| x - d ||^2) -> non polynomial but very smooth.
      Vector d(3);
      d[0] = -0.5; d[1] = -1; d[2] = -2; // arbitrary
      d -= x;
      return std::sin(d * d);
   };

   auto el_type = GENERATE(Geometry::TETRAHEDRON,
                           Geometry::CUBE,
                           Geometry::PRISM);
   int variant = GENERATE(0,1,2);

   if (variant > 0 && el_type != Geometry::PRISM)
   {
      return;
   }

   CAPTURE(el_type, variant);

   auto smesh = CylinderMesh(el_type, false, variant);

   for (auto ref : {0,1,2})
   {
      if (ref == 1) { smesh.UniformRefinement(); }

      smesh.EnsureNCMesh(true);

      if (ref == 2) { smesh.UniformRefinement(); }

      smesh.Finalize();

      auto pmesh = ParMesh(MPI_COMM_WORLD, smesh);

      // P2 ensures there are triangles without dofs
      auto error = CheckL2Projection(pmesh, smesh, 2, exact_soln);
      CHECK(std::abs(error[1] - error[0]) < 1e-9);
   }
} // test case

TEST_CASE("PNQ2PureTetHexPri",  "[Parallel], [NCMesh]")
{
   auto exact_soln = [](const Vector& x)
   {
      // sin(|| x - d ||^2) -> non polynomial but very smooth.
      Vector d(3);
      d[0] = -0.5; d[1] = -1; d[2] = -2; // arbitrary
      d -= x;
      return std::sin(d * d);
   };

   auto el_type = GENERATE(Geometry::TETRAHEDRON,
                           Geometry::CUBE,
                           Geometry::PRISM);
   int variant = GENERATE(0,1,2);

   if (variant > 0 && el_type != Geometry::PRISM)
   {
      return;
   }

   CAPTURE(el_type, variant);

   auto smesh = CylinderMesh(el_type, true);

   for (auto ref : {0,1,2})
   {
      if (ref == 1) { smesh.UniformRefinement(); }

      smesh.EnsureNCMesh(true);

      if (ref == 2) { smesh.UniformRefinement(); }

      smesh.Finalize();

      auto pmesh = ParMesh(MPI_COMM_WORLD, smesh);

      for (int p = 1; p < 3; ++p)
      {
         auto error = CheckL2Projection(pmesh, smesh, p, exact_soln);
         CHECK(std::abs(error[1] - error[0]) < 1e-9);
      }
   }
} // test case

TEST_CASE("GetVectorValueInFaceNeighborElement", "[Parallel], [NCMesh]")
{
   // The aim of this test is to verify the correct behaviour of the
   // GetVectorValue method when called on face neighbor elements in a non
   // conforming mesh.
   auto smesh = Mesh("../../data/beam-tet.mesh");

   for (int nc_level : {0,1,2,3})
   {
      for (int skip : {1,2})
      {
         for (bool use_ND : {false, true})
         {
            TestVectorValueInVolume(smesh, nc_level, skip, use_ND);
         }
      }
   }
}

TEST_CASE("TetCornerRefines", "[Parallel], [NCMesh]")
{
   auto p = GENERATE(1,2,3);
   CAPTURE(p);

   auto smesh = OrientedTriFaceMesh(1, true);
   smesh.EnsureNCMesh(true);

   auto pmesh = CheckParMeshNBE(smesh);

   pmesh->FinalizeTopology();
   pmesh->Finalize();
   pmesh->ExchangeFaceNbrData();

   REQUIRE(pmesh->Nonconforming());

   std::map<int, int> local_to_shared;
   for (int i = 0; i < pmesh->GetNSharedFaces(); ++i)
   {
      local_to_shared[pmesh->GetSharedFace(i)] = i;
   }

   // Count the number of internal faces via the boundary elements
   int num_internal = 0;
   for (int n = 0; n < pmesh->GetNBE(); ++n)
   {
      int f, o;
      pmesh->GetBdrElementFace(n, &f, &o);
      if (CheckFaceInternal(*pmesh, f, local_to_shared))
      {
         ++num_internal;
      }
   }

   MPI_Allreduce(MPI_IN_PLACE, &num_internal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   CHECK(num_internal == 1);

   CheckPoisson(*pmesh, p, pmesh->bdr_attributes.Max());
   CheckPoisson(*pmesh, p);
}

TEST_CASE("InteriorBoundaryReferenceTets", "[Parallel], [NCMesh]")
{
   constexpr auto seed = 314159;
   srand(seed);
   auto p = GENERATE(1,2,3);
   CAPTURE(p);

   auto smesh = OrientedTriFaceMesh(1, true);
   smesh.EnsureNCMesh(true);

   auto pmesh = CheckParMeshNBE(smesh);

   pmesh->FinalizeTopology();
   pmesh->Finalize();
   pmesh->ExchangeFaceNbrData();

   REQUIRE(pmesh->Nonconforming());

   std::map<int, int> local_to_shared;
   for (int i = 0; i < pmesh->GetNSharedFaces(); ++i)
   {
      local_to_shared[pmesh->GetSharedFace(i)] = i;
   }

   // Count the number of internal faces via the boundary elements
   int num_internal = 0;
   for (int n = 0; n < pmesh->GetNBE(); ++n)
   {
      int f, o;
      pmesh->GetBdrElementFace(n, &f, &o);
      if (CheckFaceInternal(*pmesh, f, local_to_shared))
      {
         ++num_internal;
      }
   }

   MPI_Allreduce(MPI_IN_PLACE, &num_internal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   CHECK(num_internal == 1);

   CheckPoisson(*pmesh, p, pmesh->bdr_attributes.Max());
   CheckPoisson(*pmesh, p);

   int num_initial_ess_tdof = CountEssentialDof<H1_FECollection>(*pmesh, p,
                                                                 smesh.bdr_attributes.Max());
   if (Mpi::Root())
   {
      REQUIRE(num_initial_ess_tdof > 0);
   }
   // Level of refinement difference across the processor boundary from root
   // zero to the others
   auto ref_level = GENERATE(1,2,3);
   auto refined_attribute = GENERATE(1,2);
   CAPTURE(ref_level);
   CAPTURE(refined_attribute);

   Mesh modified_smesh(smesh);
   for (int r = 0; r < ref_level; r++)
   {
      Array<int> el_to_refine;
      for (int n = 0; n < modified_smesh.GetNE(); n++)
      {
         if (modified_smesh.GetAttribute(n) == refined_attribute)
         {
            el_to_refine.Append(n);
         }
      }
      modified_smesh.GeneralRefinement(el_to_refine);
   }

   // There should now be some internal boundary elements, where there was one
   // before.
   CHECK(modified_smesh.GetNBE() == 3 /* external boundaries of unrefined  */
         + std::pow(4, ref_level) /* internal boundaries */
         + (3 * std::pow(4, ref_level)) /* external boundaries of refined */);

   // Force the partition to have the edge case of a parent and child being
   // divided across the processor boundary.
   auto partition = std::unique_ptr<int[]>(new int[modified_smesh.GetNE()]);
   for (int i = 0; i < modified_smesh.GetNE(); i++)
   {
      // Randomly assign to any processor but zero.
      partition[i] = Mpi::WorldSize() > 1 ? 1 + rand() % (Mpi::WorldSize() - 1) : 0;
   }
   if (Mpi::WorldSize() > 0)
   {
      // Make sure rank 0 has the non-refined attribute. This ensures it will
      // have a parent face with only ghost children.
      const int unrefined_attribute = refined_attribute == 1 ? 2 : 1;
      Array<int> root_element;
      for (int n = 0; n < modified_smesh.GetNE(); n++)
      {
         if (modified_smesh.GetAttribute(n) == unrefined_attribute)
         {
            root_element.Append(n);
         }
      }
      REQUIRE(root_element.Size() == 1);
      partition[root_element[0]] = 0;
   }

   pmesh = CheckParMeshNBE(modified_smesh, partition);
   pmesh->Finalize();
   pmesh->FinalizeTopology();
   pmesh->ExchangeFaceNbrData();

   auto check_faces = [&]()
   {
      // repopulate the local to shared map.
      local_to_shared.clear();
      for (int i = 0; i < pmesh->GetNSharedFaces(); ++i)
      {
         local_to_shared[pmesh->GetSharedFace(i)] = i;
      }

      // Count the number of internal faces via the boundary elements
      num_internal = 0;
      for (int n = 0; n < pmesh->GetNBE(); ++n)
      {
         int f, o;
         pmesh->GetBdrElementFace(n, &f, &o);
         if (CheckFaceInternal(*pmesh, f, local_to_shared))
         {
            ++num_internal;
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, &num_internal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      CHECK(num_internal == std::pow(4, ref_level));
      CheckPoisson(*pmesh, p, smesh.bdr_attributes.Max());
      CheckPoisson(*pmesh, p);
   };

   check_faces();
   pmesh->Rebalance();
   pmesh->ExchangeFaceNbrData();
   check_faces();
}

TEST_CASE("InteriorBoundaryInlineRefines", "[Parallel], [NCMesh]")
{
   const auto use_tet = GENERATE(false, true);
   const int p = use_tet ? GENERATE(1,2,3) : GENERATE(1,2);
   CAPTURE(p);

   const auto fname = use_tet ? "../../data/inline-tet.mesh" :
                      "../../data/inline-hex.mesh";
   auto smesh = Mesh(fname);
   smesh.FinalizeTopology();
   smesh.Finalize();

   // Mark even and odd elements with different attributes
   const auto num_attributes = 3;
   for (int i = 0; i < smesh.GetNE(); ++i)
   {
      smesh.SetAttribute(i, (i % num_attributes) + 1);
   }

   smesh.SetAttributes();
   const int initial_nbe = smesh.GetNBE();

   // Introduce internal boundary elements
   const int new_attribute = smesh.bdr_attributes.Max() + 1;
   for (int f = 0; f < smesh.GetNumFaces(); ++f)
   {
      int e1, e2;
      smesh.GetFaceElements(f, &e1, &e2);
      if (e1 >= 0 && e2 >= 0 && smesh.GetAttribute(e1) != smesh.GetAttribute(e2))
      {
         // This is the internal face between attributes.
         auto *new_elem = smesh.GetFace(f)->Duplicate(&smesh);
         new_elem->SetAttribute(new_attribute);
         smesh.AddBdrElement(new_elem);
      }
   }

   smesh.FinalizeTopology(); // Finalize to build relevant tables
   smesh.Finalize();

   smesh.EnsureNCMesh(true);

   // Boundary elements must've been added to make the test valid
   int num_internal_serial = smesh.GetNBE() - initial_nbe;
   REQUIRE(num_internal_serial > 0);

   auto partition = std::unique_ptr<int[]>(new int[smesh.GetNE()]);

   for (int i = 0; i < smesh.GetNE(); ++i)
   {
      partition[i] = i % Mpi::WorldSize(); // checkerboard partition
   }

   auto pmesh = CheckParMeshNBE(smesh, partition);

   pmesh->FinalizeTopology();
   pmesh->Finalize();
   pmesh->ExchangeFaceNbrData();

   std::map<int, int> local_to_shared;
   for (int i = 0; i < pmesh->GetNSharedFaces(); ++i)
   {
      local_to_shared[pmesh->GetSharedFace(i)] = i;
   }

   // Count the number of internal faces via the boundary elements
   int num_internal = 0;
   for (int n = 0; n < pmesh->GetNBE(); ++n)
   {
      int f, o;
      pmesh->GetBdrElementFace(n, &f, &o);
      if (CheckFaceInternal(*pmesh, f, local_to_shared))
      {
         ++num_internal;
      }
   }

   MPI_Allreduce(MPI_IN_PLACE, &num_internal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   CHECK(num_internal == num_internal_serial);

   CheckPoisson(*pmesh, p, new_attribute);
   CheckPoisson(*pmesh, p);

   // Mark all elements of a given attribute for refinement to a given depth.
   const auto ref_level = GENERATE(1,2);
   const auto marked_attribute = GENERATE(1,2,3);
   REQUIRE(marked_attribute <= num_attributes);
   CAPTURE(ref_level);
   CAPTURE(marked_attribute);
   for (int r = 0; r < ref_level; r++)
   {
      Array<int> elem_to_refine;
      for (int i = 0; i < smesh.GetNE(); ++i)
      {
         if (smesh.GetAttribute(i) == marked_attribute)
         {
            elem_to_refine.Append(i);
         }
      }
      smesh.GeneralRefinement(elem_to_refine);
   }

   pmesh = CheckParMeshNBE(smesh);
   pmesh->FinalizeTopology();
   pmesh->Finalize();
   pmesh->ExchangeFaceNbrData();

   // Count the number of internal boundary elements
   num_internal_serial = 0;
   for (int n = 0; n < smesh.GetNBE(); ++n)
   {
      int f, o;
      smesh.GetBdrElementFace(n, &f, &o);
      int e1, e2;
      smesh.GetFaceElements(f, &e1, &e2);
      if (e1 >= 0 && e2 >= 0 && smesh.GetAttribute(e1) != smesh.GetAttribute(e2))
      {
         ++num_internal_serial;
      }
   }

   auto check_faces = [&]()
   {
      // repopulate the local to shared map.
      local_to_shared.clear();
      for (int i = 0; i < pmesh->GetNSharedFaces(); ++i)
      {
         local_to_shared[pmesh->GetSharedFace(i)] = i;
      }

      num_internal = 0;
      for (int n = 0; n < pmesh->GetNBE(); ++n)
      {
         int f, o;
         pmesh->GetBdrElementFace(n, &f, &o);
         if (CheckFaceInternal(*pmesh, f, local_to_shared))
         {
            ++num_internal;
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, &num_internal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      CHECK(num_internal == num_internal_serial);

      CheckPoisson(*pmesh, p, pmesh->bdr_attributes.Max());
      CheckPoisson(*pmesh, p);
   };

   check_faces();
   pmesh->Rebalance();
   pmesh->ExchangeFaceNbrData();
   check_faces();
}

TEST_CASE("InteriorBoundaryReferenceCubes", "[Parallel], [NCMesh]")
{
   auto p = GENERATE(1,2);
   CAPTURE(p);

   auto smesh = DividingPlaneMesh(false, true);

   auto pmesh = CheckParMeshNBE(smesh);

   pmesh->FinalizeTopology();
   pmesh->Finalize();
   pmesh->ExchangeFaceNbrData();

   REQUIRE(pmesh->Conforming());

   std::map<int, int> local_to_shared;
   for (int i = 0; i < pmesh->GetNSharedFaces(); ++i)
   {
      local_to_shared[pmesh->GetSharedFace(i)] = i;
   }

   // Count the number of internal faces via the boundary elements
   int num_internal = 0;
   for (int n = 0; n < pmesh->GetNBE(); ++n)
   {
      int f, o;
      pmesh->GetBdrElementFace(n, &f, &o);
      if (CheckFaceInternal(*pmesh, f, local_to_shared))
      {
         ++num_internal;
      }
   }

   MPI_Allreduce(MPI_IN_PLACE, &num_internal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   CHECK(num_internal == 1);

   CheckPoisson(*pmesh, p, pmesh->bdr_attributes.Max());
   CheckPoisson(*pmesh, p);

   for (int refined_elem : {0, 1})
   {
      // Now NC refine one of the attached elements, this should result in 4
      // internal boundary elements.
      Array<int> el_to_refine;
      el_to_refine.Append(refined_elem);

      Mesh modified_smesh(smesh);
      modified_smesh.GeneralRefinement(el_to_refine);

      // There should now be four internal boundary elements, where there was
      // one before.
      CHECK(modified_smesh.GetNBE() == 5 /* external boundaries of unrefined  */
            + 4 /* internal boundaries */
            + (5 * 4) /* external boundaries of refined */);

      // Force the partition to have the edge case of a parent and child being
      // divided across the processor boundary. This necessitates the
      // GhostBoundaryElement treatment.
      auto partition = std::unique_ptr<int[]>(new int[modified_smesh.GetNE()]);
      srand(314159);
      for (int i = 0; i < modified_smesh.GetNE(); ++i)
      {
         // Randomly assign to any processor but zero.
         partition[i] = Mpi::WorldSize() > 1 ? 1 + rand() % (Mpi::WorldSize() - 1) : 0;
      }
      if (Mpi::WorldSize() > 0)
      {
         // Make sure on rank 1 there is a parent face with only ghost child
         // faces. This can cause issues with higher order dofs being
         // uncontrolled.
         partition[refined_elem == 0 ? modified_smesh.GetNE() - 1 : 0] = 0;
      }

      pmesh = CheckParMeshNBE(modified_smesh, partition);
      pmesh->Finalize();
      pmesh->FinalizeTopology();
      pmesh->ExchangeFaceNbrData();

      auto check_faces = [&]()
      {
         // repopulate the local to shared map.
         local_to_shared.clear();
         for (int i = 0; i < pmesh->GetNSharedFaces(); ++i)
         {
            local_to_shared[pmesh->GetSharedFace(i)] = i;
         }

         // Count the number of internal faces via the boundary elements
         num_internal = 0;
         for (int n = 0; n < pmesh->GetNBE(); ++n)
         {
            int f, o;
            pmesh->GetBdrElementFace(n, &f, &o);
            if (CheckFaceInternal(*pmesh, f, local_to_shared))
            {
               ++num_internal;
            }
         }
         MPI_Allreduce(MPI_IN_PLACE, &num_internal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
         CHECK(num_internal == 4);

         CAPTURE(refined_elem);
         CheckPoisson(*pmesh, p, smesh.bdr_attributes.Max());
         CheckPoisson(*pmesh, p);
      };

      check_faces();
      pmesh->Rebalance();
      pmesh->ExchangeFaceNbrData();
      check_faces();
   }
}

TEST_CASE("ParMeshInternalBoundaryTetStarMesh", "[Parallel], [NCMesh]")
{
   auto smesh = TetStarMesh();
   smesh.EnsureNCMesh(true);

   if (Mpi::WorldSize() < 5) { return; }

   auto partition = std::unique_ptr<int[]>(new int[5]);
   for (int i = 0; i < 5; i++)
   {
      partition[i] = i;
   }
   auto pmesh = CheckParMeshNBE(smesh, partition);
   pmesh->FinalizeTopology();
   pmesh->Finalize();
   pmesh->ExchangeFaceNbrData();

   REQUIRE(pmesh->Nonconforming());

   std::map<int, int> local_to_shared;
   for (int i = 0; i < pmesh->GetNSharedFaces(); ++i)
   {
      local_to_shared[pmesh->GetSharedFace(i)] = i;
   }

   // Count the number of internal faces via the boundary elements
   int num_internal = 0;
   for (int n = 0; n < pmesh->GetNBE(); ++n)
   {
      int f, o;
      pmesh->GetBdrElementFace(n, &f, &o);
      if (CheckFaceInternal(*pmesh, f, local_to_shared))
      {
         ++num_internal;
      }
   }

   const int rank = Mpi::WorldRank();
   SECTION("Unrefined")
   {
      MPI_Allreduce(MPI_IN_PLACE, &num_internal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      CHECK(num_internal == 4);

      CHECK(CountEssentialDof<H1_FECollection>(*pmesh, 1,
                                               smesh.bdr_attributes.Max()) == (rank == 0 ? 4 : 0));
      CHECK(CountEssentialDof<H1_FECollection>(*pmesh, 2,
                                               smesh.bdr_attributes.Max()) == (rank == 0 ? 4 + 6 : 0));
      CHECK(CountEssentialDof<H1_FECollection>(*pmesh, 3,
                                               smesh.bdr_attributes.Max()) == (rank == 0 ? 4 + 6*2 + 4*1 : 0));
      CHECK(CountEssentialDof<H1_FECollection>(*pmesh, 4,
                                               smesh.bdr_attributes.Max()) == (rank == 0 ? 4 + 6*3 + 4*3 : 0));

      CHECK(CountEssentialDof<ND_FECollection>(*pmesh, 1,
                                               smesh.bdr_attributes.Max()) == (rank == 0 ? 6 : 0));
      CHECK(CountEssentialDof<ND_FECollection>(*pmesh, 2,
                                               smesh.bdr_attributes.Max()) == (rank == 0 ? 20 : 0));
      CHECK(CountEssentialDof<ND_FECollection>(*pmesh, 3,
                                               smesh.bdr_attributes.Max()) == (rank == 0 ? 42 : 0));
      CHECK(CountEssentialDof<ND_FECollection>(*pmesh, 4,
                                               smesh.bdr_attributes.Max()) == (rank == 0 ? 72 : 0));
      CHECK(pmesh->GetNBE() == (rank == 0 ? 4 : (rank < 5 ? 3 : 0)));
   }

   SECTION("Refinement")
   {
      // Refining an element attached to the core should not change the number
      // of essential DOF, or the owner of them.

      const int refined_attribute = GENERATE(1,2,3,4,5); // equal to rank of owner + 1
      int ref_level = GENERATE(0, 1, 2, 3);
      for (int r = 0; r < ref_level; r++)
      {
         Array<int> el_to_refine;
         for (int n = 0; n < pmesh->GetNE(); n++)
         {
            if (pmesh->GetAttribute(n) == refined_attribute)
            {
               el_to_refine.Append(n);
            }
         }
         pmesh->GeneralRefinement(el_to_refine);
      }
      pmesh->ExchangeFaceNbrData();

      CAPTURE(rank);
      CAPTURE(refined_attribute);
      CAPTURE(ref_level);
      CHECK(pmesh->GetNE() == (rank == refined_attribute - 1 ? std::pow(8,
                                                                        ref_level) : 1));
      CHECK(pmesh->GetNBE() == (rank == refined_attribute - 1
                                ? std::pow(4, ref_level + 1)
                                : (ref_level == 0 && rank == 0 ? 4 : 3)));

      // Refining on only one side of the boundary face should not change the
      // number of essential true dofs, which should match the number within the
      // original face.
      CHECK(ParCountEssentialDof<H1_FECollection>(*pmesh, 1,
                                                  smesh.bdr_attributes.Max()) == 4);
      CHECK(ParCountEssentialDof<H1_FECollection>(*pmesh, 2,
                                                  smesh.bdr_attributes.Max()) == 4 + 6);
      CHECK(ParCountEssentialDof<H1_FECollection>(*pmesh, 3,
                                                  smesh.bdr_attributes.Max()) == 4 + 6*2 + 4*1);
      CHECK(ParCountEssentialDof<H1_FECollection>(*pmesh, 4,
                                                  smesh.bdr_attributes.Max()) == 4 + 6*3 + 4*3);

      CHECK(ParCountEssentialDof<ND_FECollection>(*pmesh, 1,
                                                  smesh.bdr_attributes.Max()) == (rank == 0 ? 6 : 0));
      CHECK(ParCountEssentialDof<ND_FECollection>(*pmesh, 2,
                                                  smesh.bdr_attributes.Max()) == (rank == 0 ? 20 : 0));
      CHECK(ParCountEssentialDof<ND_FECollection>(*pmesh, 3,
                                                  smesh.bdr_attributes.Max()) == (rank == 0 ? 42 : 0));
      CHECK(ParCountEssentialDof<ND_FECollection>(*pmesh, 4,
                                                  smesh.bdr_attributes.Max()) == (rank == 0 ? 72 : 0));
   }
}

TEST_CASE("ParDividingPlaneMesh", "[Parallel], [NCMesh]")
{
   auto refine_attribute = [](Mesh& mesh, int attr, int ref_level)
   {
      for (int r = 0; r < ref_level; r++)
      {
         Array<int> el_to_refine;
         for (int n = 0; n < mesh.GetNE(); n++)
         {
            if (mesh.GetAttribute(n) == attr)
            {
               el_to_refine.Append(n);
            }
         }
         mesh.GeneralRefinement(el_to_refine);
      }
   };

   SECTION("Hex")
   {
      auto mesh = DividingPlaneMesh(false);
      mesh.EnsureNCMesh(true);

      CHECK(mesh.GetNBE() == 2 * 5 + 1);
      CHECK(mesh.GetNE() == 2);

      SECTION("H1Hex")
      {
         mesh.UniformRefinement();
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 1,
                                                  mesh.bdr_attributes.Max()) == 3*3);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 2,
                                                  mesh.bdr_attributes.Max()) == 5*5);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 3,
                                                  mesh.bdr_attributes.Max()) == 7*7);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 4,
                                                  mesh.bdr_attributes.Max()) == 9*9);

         auto attr = GENERATE(1,2);
         auto ref_level = GENERATE(1,2);
         refine_attribute(mesh, attr, ref_level);

         CHECK(CountEssentialDof<H1_FECollection>(mesh, 1,
                                                  mesh.bdr_attributes.Max()) == 3*3);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 2,
                                                  mesh.bdr_attributes.Max()) == 5*5);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 3,
                                                  mesh.bdr_attributes.Max()) == 7*7);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 4,
                                                  mesh.bdr_attributes.Max()) == 9*9);
      }
   }

   SECTION("Tet")
   {
      auto mesh = DividingPlaneMesh(true, true);
      mesh.EnsureNCMesh(true);
      auto pmesh = CheckParMeshNBE(mesh);
      pmesh->FinalizeTopology();
      pmesh->Finalize();
      pmesh->ExchangeFaceNbrData();

      CHECK(pmesh->bdr_attributes.Max() == mesh.bdr_attributes.Max());

      auto attr = GENERATE(1,2);
      auto ref_level = GENERATE(1,2);
      CAPTURE(attr);
      CAPTURE(ref_level);

      const int initial_num_vert = 4;
      const int initial_num_edge = 5;
      const int initial_num_face = 2;
      SECTION("H1Tet")
      {
         CHECK(ParCountEssentialDof<H1_FECollection>(*pmesh, 1,
                                                     mesh.bdr_attributes.Max()) == initial_num_vert);
         CHECK(ParCountEssentialDof<H1_FECollection>(*pmesh, 2,
                                                     mesh.bdr_attributes.Max()) == initial_num_vert + initial_num_edge);
         CHECK(ParCountEssentialDof<H1_FECollection>(*pmesh, 3,
                                                     mesh.bdr_attributes.Max()) == initial_num_vert + 2*initial_num_edge +
               initial_num_face);
         CHECK(ParCountEssentialDof<H1_FECollection>(*pmesh, 4,
                                                     mesh.bdr_attributes.Max()) == initial_num_vert + 3*initial_num_edge +
               3*initial_num_face);

         refine_attribute(*pmesh, attr, ref_level);

         CHECK(ParCountEssentialDof<H1_FECollection>(*pmesh, 1,
                                                     mesh.bdr_attributes.Max()) == initial_num_vert);
         CHECK(ParCountEssentialDof<H1_FECollection>(*pmesh, 2,
                                                     mesh.bdr_attributes.Max()) == initial_num_vert + initial_num_edge);
         CHECK(ParCountEssentialDof<H1_FECollection>(*pmesh, 3,
                                                     mesh.bdr_attributes.Max()) == initial_num_vert + 2*initial_num_edge +
               initial_num_face);
         CHECK(ParCountEssentialDof<H1_FECollection>(*pmesh, 4,
                                                     mesh.bdr_attributes.Max()) == initial_num_vert + 3*initial_num_edge +
               3*initial_num_face);
      }

      SECTION("NDTet")
      {
         CHECK(ParCountEssentialDof<ND_FECollection>(*pmesh, 1,
                                                     mesh.bdr_attributes.Max()) == 5);
         CHECK(ParCountEssentialDof<ND_FECollection>(*pmesh, 2,
                                                     mesh.bdr_attributes.Max()) == 14);
         CHECK(ParCountEssentialDof<ND_FECollection>(*pmesh, 3,
                                                     mesh.bdr_attributes.Max()) == 27);
         CHECK(ParCountEssentialDof<ND_FECollection>(*pmesh, 4,
                                                     mesh.bdr_attributes.Max()) == 44);

         refine_attribute(*pmesh, attr, ref_level);
         CHECK(ParCountEssentialDof<ND_FECollection>(*pmesh, 1,
                                                     mesh.bdr_attributes.Max()) == 5);
         CHECK(ParCountEssentialDof<ND_FECollection>(*pmesh, 2,
                                                     mesh.bdr_attributes.Max()) == 14);
         CHECK(ParCountEssentialDof<ND_FECollection>(*pmesh, 3,
                                                     mesh.bdr_attributes.Max()) == 27);
         CHECK(ParCountEssentialDof<ND_FECollection>(*pmesh, 4,
                                                     mesh.bdr_attributes.Max()) == 44);
      }
   }
}

TEST_CASE("ParTetFaceFlips", "[Parallel], [NCMesh]")
{
   /*
      1. Define an ND space, and project a smooth non polynomial function onto
         the space.
      2. Compute y-z components in the face, and check that they are equal when
         evaluated from either side of the face. Tangential continuity of the ND
         space should ensure they are identical, if orientations are correctly
         accounted for.
      3. Mark the mesh as NC, build a new FESpace, and repeat. There should be
         no change as the faces are "conformal" though they are within the NC
         structure.
      3. Partition the mesh, create a ParFESpace and repeat the above. There
         should be no difference in conformal parallel.
      4. Construct the ParMesh from the NCMesh and repeat. As above, there
         should be no change.
      5. Perform NC refinement on one side of the internal face, the number of
         conformal dof in the face will not change, so there should also be no
         difference. This will be complicated by ensuring the slave evaluations
         are at the same points.
   */

   auto orientation = GENERATE(1,3,5);
   auto smesh = OrientedTriFaceMesh(orientation);
   smesh.EnsureNodes();

   CHECK(smesh.GetNBE() == 1);

   // A smooth function in each vector component
   constexpr int order = 3, dim = 3, quadrature_order = 4;
   constexpr real_t kappa = 2 * M_PI;
   auto E_exact = [=](const Vector &x, Vector &E)
   {
      E(0) = cos(kappa * x(1));
      E(1) = cos(kappa * x(2));
      E(2) = cos(kappa * x(0));
   };
   VectorFunctionCoefficient E_coeff(dim, E_exact);

   // Helper for evaluating the ND grid function on either side of the first
   // conformal shared face. Specific to the pair of tet mesh described above,
   // but can be generalized.
   auto check_parallel_nc_conformal = [&](ParMesh &mesh)
   {
      ND_FECollection fe_collection(order, dim);
      ParFiniteElementSpace fe_space(&mesh, &fe_collection);
      ParGridFunction E(&fe_space);

      E.ProjectCoefficient(E_coeff);
      E.ExchangeFaceNbrData();

      auto *P = fe_space.GetProlongationMatrix();
      if (P != nullptr)
      {
         // Projection does not respect the non-conformal constraints. Extract
         // the true (conformal) and prolongate to get the NC respecting
         // projection.
         auto E_true = E.GetTrueVector();
         P->Mult(E_true, E);
         E.ExchangeFaceNbrData();
      }
      ParGridFunction * const coords = dynamic_cast<ParGridFunction*>
                                       (mesh.GetNodes());

      const auto &ir = IntRules.Get(Geometry::Type::TRIANGLE, quadrature_order);
      IntegrationRule left_eir(ir.GetNPoints()),
                      right_eir(ir.GetNPoints()); // element integration rules

      bool y_valid = true, z_valid = true;
      for (int n = 0; n < mesh.GetNBE(); n++)
      {
         auto f = mesh.GetBdrElementFaceIndex(n);

         auto finfo = mesh.GetFaceInformation(f);
         auto &face_element_transform = finfo.IsShared()
                                        ? *mesh.GetSharedFaceTransformationsByLocalIndex(f, true)
                                        : *mesh.GetFaceElementTransformations(f);

         face_element_transform.Loc1.Transform(ir, left_eir);
         face_element_transform.Loc2.Transform(ir, right_eir);

         constexpr real_t tol = 1e-14;
         REQUIRE(left_eir.GetNPoints() == ir.GetNPoints());
         REQUIRE(right_eir.GetNPoints() == ir.GetNPoints());
         Vector left_val, right_val;
         for (int i = 0; i < ir.GetNPoints(); i++)
         {
            face_element_transform.Elem1->SetIntPoint(&left_eir[i]);
            coords->GetVectorValue(*face_element_transform.Elem1, left_eir[i], left_val);
            coords->GetVectorValue(*face_element_transform.Elem1, left_eir[i], right_val);
            REQUIRE(std::abs(left_val(0) - right_val(0)) < tol);
            REQUIRE(std::abs(left_val(1) - right_val(1)) < tol);
            REQUIRE(std::abs(left_val(2) - right_val(2)) < tol);
            E.GetVectorValue(*face_element_transform.Elem1, left_eir[i], left_val);

            face_element_transform.Elem2->SetIntPoint(&right_eir[i]);
            E.GetVectorValue(*face_element_transform.Elem2, right_eir[i], right_val);

            // Check that the second and third rows agree. The y and z should
            // agree as the normal is in the x direction.
            y_valid &= (std::abs(left_val(1) - right_val(1)) < tol);
            z_valid &= (std::abs(left_val(2) - right_val(2)) < tol);
         }
      }
      CHECK(y_valid);
      CHECK(z_valid);

      return fe_space.GlobalTrueVSize();
   };

   SECTION("Conformal")
   {
      auto partition_flag = GENERATE(false, true);
      CAPTURE(partition_flag);
      auto partition = std::unique_ptr<int[]>(new int[2]);
      if (Mpi::WorldSize() > 1)
      {
         partition[0] = partition_flag ? 0 : 1; partition[1] = partition_flag  ? 1 : 0;
      }
      else
      {
         partition[0] = 0; partition[1] = 0;
      }
      auto pmesh = CheckParMeshNBE(smesh);
      pmesh->Finalize();
      pmesh->ExchangeFaceNbrData();

      check_parallel_nc_conformal(*pmesh);
   }

   SECTION("ConformalSerialUniformRefined")
   {
      smesh.UniformRefinement();
      auto pmesh = CheckParMeshNBE(smesh);
      pmesh->Finalize();
      pmesh->ExchangeFaceNbrData();

      check_parallel_nc_conformal(*pmesh);
   }

   SECTION("ConformalParallelUniformRefined")
   {
      auto partition_flag = GENERATE(false, true);
      CAPTURE(partition_flag);
      auto partition = std::unique_ptr<int[]>(new int[2]);
      if (Mpi::WorldSize() > 1)
      {
         partition[0] = partition_flag ? 0 : 1; partition[1] = partition_flag  ? 1 : 0;
      }
      else
      {
         partition[0] = 0; partition[1] = 0;
      }
      auto pmesh = CheckParMeshNBE(smesh);
      pmesh->UniformRefinement();
      pmesh->Finalize();
      pmesh->ExchangeFaceNbrData();
      check_parallel_nc_conformal(*pmesh);
   }

   SECTION("Nonconformal")
   {
      auto partition_flag = GENERATE(false, true);
      CAPTURE(partition_flag);
      auto partition = std::unique_ptr<int[]>(new int[2]);
      if (Mpi::WorldSize() > 1)
      {
         partition[0] = partition_flag ? 0 : 1; partition[1] = partition_flag  ? 1 : 0;
      }
      else
      {
         partition[0] = 0; partition[1] = 0;
      }
      smesh.EnsureNCMesh(true);
      auto pmesh = CheckParMeshNBE(smesh);
      pmesh->Finalize();
      pmesh->ExchangeFaceNbrData();

      check_parallel_nc_conformal(*pmesh);
   }

   SECTION("NonconformalSerialUniformRefined")
   {
      smesh.UniformRefinement();
      smesh.EnsureNCMesh(true);
      auto pmesh = CheckParMeshNBE(smesh);
      pmesh->Finalize();
      pmesh->ExchangeFaceNbrData();

      check_parallel_nc_conformal(*pmesh);
   }

   SECTION("NonconformalSerialRefined")
   {
      smesh.EnsureNCMesh(true);
      int ref_level = GENERATE(1, 2);
      for (int r = 0; r < ref_level; r++)
      {
         Array<int> el_to_refine;
         for (int n = 0; n < smesh.GetNE(); n++)
         {
            if (smesh.GetAttribute(n) == 2)
            {
               el_to_refine.Append(n);
            }
         }
         smesh.GeneralRefinement(el_to_refine);
      }
      auto pmesh = CheckParMeshNBE(smesh);
      pmesh->Finalize();
      pmesh->ExchangeFaceNbrData();

      check_parallel_nc_conformal(*pmesh);
   }

   SECTION("NonconformalParallelUniformRefined")
   {
      auto partition_flag = GENERATE(false, true);
      CAPTURE(partition_flag);
      auto partition = std::unique_ptr<int[]>(new int[2]);
      if (Mpi::WorldSize() > 1)
      {
         partition[0] = partition_flag ? 0 : 1; partition[1] = partition_flag  ? 1 : 0;
      }
      else
      {
         partition[0] = 0; partition[1] = 0;
      }
      smesh.EnsureNCMesh(true);
      auto pmesh = CheckParMeshNBE(smesh);
      pmesh->UniformRefinement();
      pmesh->Finalize();
      pmesh->ExchangeFaceNbrData();

      check_parallel_nc_conformal(*pmesh);
   }

   SECTION("NonconformalParallelRefined")
   {
      auto partition_flag = GENERATE(false, true);
      CAPTURE(partition_flag);
      auto partition = std::unique_ptr<int[]>(new int[2]);
      if (Mpi::WorldSize() > 1)
      {
         partition[0] = partition_flag ? 0 : 1; partition[1] = partition_flag  ? 1 : 0;
      }
      else
      {
         partition[0] = 0; partition[1] = 0;
      }
      smesh.EnsureNCMesh(true);
      auto pmesh = CheckParMeshNBE(smesh);
      int ref_level = GENERATE(1, 2);
      for (int r = 0; r < ref_level; r++)
      {
         Array<int> el_to_refine;
         for (int n = 0; n < pmesh->GetNE(); n++)
         {
            if (pmesh->GetAttribute(n) == 2)
            {
               el_to_refine.Append(n);
            }
         }
         pmesh->GeneralRefinement(el_to_refine);
      }
      pmesh->Finalize();
      pmesh->ExchangeFaceNbrData();

      check_parallel_nc_conformal(*pmesh);
   }

   SECTION("NonconformalLevelTwoRefined")
   {
      smesh.EnsureNCMesh(true);
      smesh.UniformRefinement();
      Array<int> el_to_refine(1);
      for (int n = 0; n < smesh.GetNE(); n++)
      {
         if (smesh.GetAttribute(n) == 2)
         {
            CAPTURE(n);
            Mesh smesh2(smesh);
            el_to_refine[0] = n;
            smesh2.GeneralRefinement(el_to_refine);
            for (int m = 0; m < smesh2.GetNE(); m++)
            {
               if (smesh2.GetAttribute(m) == 2)
               {
                  CAPTURE(m);
                  Mesh smesh3(smesh2);
                  el_to_refine[0] = m;
                  smesh3.GeneralRefinement(el_to_refine);
                  check_parallel_nc_conformal(*CheckParMeshNBE(smesh3));
               }
            }
         }
      }
   }

}

TEST_CASE("Parallel RP=I", "[Parallel], [NCMesh]")
{
   const int order = GENERATE(1, 2, 3);
   CAPTURE(order);
   const int dim = 3;

   SECTION("Hex")
   {
      auto smesh = DividingPlaneMesh(false, true);
      Array<int> refinements(1);
      refinements[0] = 0;
      smesh.GeneralRefinement(refinements);
      ParMesh mesh(MPI_COMM_WORLD, smesh);
      SECTION("ND")
      {
         ND_FECollection fec(order, dim);
         ParFiniteElementSpace fespace(&mesh, &fec);
         CHECK(CheckRPIdentity(fespace));
      }
      SECTION("H1")
      {
         H1_FECollection fec(order, dim);
         ParFiniteElementSpace fespace(&mesh, &fec);
         CHECK(CheckRPIdentity(fespace));
      }
   }

   SECTION("Tet")
   {
      auto orientation = GENERATE(1,3,5);
      Array<int> refinements(1);
      refinements[0] = 0;
      auto smesh = OrientedTriFaceMesh(orientation, true);
      smesh.EnsureNCMesh(true); // Always checking NC
      ParMesh mesh(MPI_COMM_WORLD, smesh);
      SECTION("ND")
      {
         ND_FECollection fec(order, dim);
         ParFiniteElementSpace fespace(&mesh, &fec);
         CHECK(CheckRPIdentity(fespace));
      }
      SECTION("H1")
      {
         H1_FECollection fec(order, dim);
         ParFiniteElementSpace fespace(&mesh, &fec);
         CHECK(CheckRPIdentity(fespace));
      }
   }
}

// Test case: Essential true DOFs of an interior boundary of a 2D parallel
//            non-conforming mesh refined on one side of it. The master edges
//            on the coarse side have no boundary elements, and must be marked
//            from the slave boundary elements on the refined side only if the
//            interior boundary is essential. The same is checked on the part
//            of the mesh of each rank loaded as a serial mesh, in which the
//            slave edges of a master edge may all be in ghost elements.
TEST_CASE("ParInteriorBoundaryEssentialDofs2D", "[Parallel], [NCMesh]")
{
   const bool tri_mesh = GENERATE(false, true);
   const int order = GENERATE(1, 2, 3);
   const bool ess_interior = GENERATE(true, false);
   CAPTURE(tri_mesh, order, ess_interior);

   auto smesh = OneSidedRefinedDividingLineMesh(tri_mesh);
   ParMesh mesh(MPI_COMM_WORLD, smesh);
   std::stringstream part_stream;
   mesh.ParPrint(part_stream);
   Mesh part(part_stream, 1, 1);
   const int interior_attr = mesh.bdr_attributes.Max();
   Array<int> ess_bdr(interior_attr);
   ess_bdr = ess_interior ? 0 : 1;
   ess_bdr[interior_attr - 1] = ess_interior ? 1 : 0;

   H1_FECollection h1_fec(order, 2);
   ND_FECollection nd_fec(order, 2);
   for (const FiniteElementCollection *fec :
        std::vector<const FiniteElementCollection *> {&h1_fec, &nd_fec})
   {
      CAPTURE(fec->Name());
      ParFiniteElementSpace fes(&mesh, fec);
      Array<int> ess_tdof_list, ess_tdof_marker;
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      FiniteElementSpace::ListToMarker(ess_tdof_list, fes.GetTrueVSize(),
                                       ess_tdof_marker);
      auto tdof_of = [&](int ldof) { return fes.GetLocalTDofNumber(ldof); };
      int checked = CheckInteriorBoundaryEssentialDofs(
                       mesh, fes, ess_tdof_marker, ess_interior, tdof_of);
      MPI_Allreduce(MPI_IN_PLACE, &checked, 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
      REQUIRE(checked > 0);

      // The true DOFs of a serial space on the part are not supported when the
      // master edges of some of its slave edges are ghosts, so check its local
      // DOFs instead.
      FiniteElementSpace part_fes(&part, fec);
      Array<int> ess_dof_marker;
      part_fes.GetEssentialVDofs(ess_bdr, ess_dof_marker);
      auto ldof_of = [](int ldof) { return ldof; };
      checked = CheckInteriorBoundaryEssentialDofs(
                   part, part_fes, ess_dof_marker, ess_interior, ldof_of);
      MPI_Allreduce(MPI_IN_PLACE, &checked, 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
      REQUIRE(checked > 0);
   }
}

#endif // MFEM_USE_MPI

TEST_CASE("ReferenceCubeInternalBoundaries", "[NCMesh]")
{
   auto p = GENERATE(1,2,3);
   CAPTURE(p);

   auto smesh = DividingPlaneMesh(false, true);
   smesh.EnsureNCMesh();
   CHECK(smesh.GetNBE() == 2 * 5 + 1);

   auto with_internal = CheckPoisson(smesh, p); // Include the internal boundary
   auto without_internal = CheckPoisson(smesh, p,
                                        smesh.bdr_attributes.Max()); // Exclude the internal boundary

   switch (p)
   {
      case 1:
         CHECK(with_internal == without_internal); break;
      case 2:
         CHECK(with_internal == without_internal + 1); break;
      case 3:
         CHECK(with_internal == without_internal + 4); break;
   }

   auto ref_type = char(GENERATE(Refinement::XYZ));
   Array<Refinement> refs(1);
   for (auto ref : {0,1})
   {
      auto ssmesh = Mesh(smesh);

      CAPTURE(ref_type);

      // Now NC refine one of the attached elements, this should result in 2
      // internal boundary elements.
      refs[0].index = ref;
      refs[0].SetType(ref_type);

      ssmesh.GeneralRefinement(refs);

      // There should now be four internal boundary elements, where there was
      // one before.
      if (ref_type == 2 /* Y */ || ref_type == 4 /* Z */)
      {
         CHECK(ssmesh.GetNBE() == 5 /* external boundaries of unrefined element  */
               + 2 /* internal boundaries */
               + (2 * 4) /* external boundaries of refined elements */);
      }
      else if (ref_type == 6)
      {
         CHECK(ssmesh.GetNBE() == 5 /* external boundaries of unrefined element  */
               + 4 /* internal boundaries */
               + (4 * 3) /* external boundaries of refined elements */);
      }
      else if (ref_type == 7)
      {
         CHECK(ssmesh.GetNBE() == 5 /* external boundaries of unrefined element  */
               + 4 /* internal boundaries */
               + (4 * 3 + 4 * 2) /* external boundaries of refined elements */);
      }
      else
      {
         MFEM_ABORT("!");
      }

      // Count the number of internal boundary elements
      int num_internal = 0;
      for (int n = 0; n < ssmesh.GetNBE(); ++n)
      {
         int f, o;
         ssmesh.GetBdrElementFace(n, &f, &o);
         int e1, e2;
         ssmesh.GetFaceElements(f, &e1, &e2);
         if (e1 >= 0 && e2 >= 0 && ssmesh.GetAttribute(e1) != ssmesh.GetAttribute(e2))
         {
            ++num_internal;
         }
      }
      CHECK(num_internal == (ref_type <= 4 ? 2 : 4));

      ssmesh.FinalizeTopology();
      ssmesh.Finalize();

      without_internal = CheckPoisson(ssmesh, p,
                                      ssmesh.bdr_attributes.Max()); // Exclude the internal boundary
      with_internal = CheckPoisson(ssmesh, p); // Include the internal boundary

      // All slaves dofs that are introduced on the face are constrained by the
      // master dofs, thus the additional constraints on the internal boundary
      // are purely on the master face, which matches the initial unrefined
      // case.
      switch (p)
      {
         case 1:
            CHECK(with_internal == without_internal); break;
         case 2:
            CHECK(with_internal == without_internal + 1); break;
         case 3:
            CHECK(with_internal == without_internal + 4); break;
      }
   }
}

TEST_CASE("RefinedCubesInternalBoundaries", "[NCMesh]")
{
   auto p = GENERATE(1,2,3);
   CAPTURE(p);

   auto smesh = DividingPlaneMesh(false, true);

   smesh.UniformRefinement();

   smesh.FinalizeTopology();
   smesh.Finalize();

   // Exactly four boundary elements must be added
   CHECK(smesh.GetNBE() == 2 * 5 * 4 + 4);

   smesh.EnsureNCMesh();
   CHECK(smesh.GetNBE() == 2 * 5 * 4 + 4);

   int without_internal = CheckPoisson(smesh, p,
                                       7); // Exclude the internal boundary
   int with_internal = CheckPoisson(smesh, p); // Include the internal boundary

   switch (p)
   {
      case 1:
         CHECK(with_internal == without_internal + 1); break;
      case 2:
         CHECK(with_internal == without_internal + 3 * 3); break;
      case 3:
         CHECK(with_internal == without_internal + 5 * 5); break;
   }

   // Mark all elements on one side of the attribute boundary to refine
   Array<Refinement> refs;
   for (int n = 0; n < smesh.GetNE(); ++n)
   {
      if (smesh.GetAttribute(n) == 2)
      {
         refs.Append(Refinement{n});
      }
   }

   smesh.GeneralRefinement(refs);
   smesh.FinalizeTopology();
   smesh.Finalize();

   // There should now be 16 internal boundary elements, where there were 4
   // before.
   CHECK(smesh.GetNBE() == 5 * 4 /* external boundaries of unrefined domain  */
         + 4 * 4 /* internal boundaries */
         + 5 * 16 /* external boundaries of refined elements */);

   // Count the number of internal boundary elements
   int num_internal = 0;
   for (int n = 0; n < smesh.GetNBE(); ++n)
   {
      int f, o;
      smesh.GetBdrElementFace(n, &f, &o);
      int e1, e2;
      smesh.GetFaceElements(f, &e1, &e2);
      if (e1 >= 0 && e2 >= 0 && smesh.GetAttribute(e1) != smesh.GetAttribute(e2))
      {
         ++num_internal;
      }
   }
   CHECK(num_internal == 16);


   without_internal = CheckPoisson(smesh, p,
                                   smesh.bdr_attributes.Max()); // Exclude the internal boundary
   with_internal = CheckPoisson(smesh, p); // Include the internal boundary

   switch (p)
   {
      case 1:
         CHECK(with_internal == without_internal + 1); break;
      case 2:
         CHECK(with_internal == without_internal + 3 * 3); break;
      case 3:
         CHECK(with_internal == without_internal + 5 * 5); break;
   }
}

TEST_CASE("ReferenceTetInternalBoundaries", "[NCMesh]")
{
   auto p = GENERATE(1,2,3);
   CAPTURE(p);

   auto smesh = Mesh("../../data/ref-tetrahedron.mesh");
   Array<Refinement> refs;
   refs.Append(Refinement(0, Refinement::X));
   smesh.GeneralRefinement(refs);

   // Now have a pair of elements, make the second element a different
   // attribute.
   smesh.SetAttribute(1, 2);

   REQUIRE(smesh.GetNE() == 2);
   REQUIRE(smesh.GetNBE() == 2 * 3);

   // Introduce an internal boundary element
   for (int f = 0; f < smesh.GetNumFaces(); ++f)
   {
      int e1, e2;
      smesh.GetFaceElements(f, &e1, &e2);
      if (e1 >= 0 && e2 >= 0 && smesh.GetAttribute(e1) != smesh.GetAttribute(e2))
      {
         // This is the internal face between attributes.
         auto *new_elem = smesh.GetFace(f)->Duplicate(&smesh);
         new_elem->SetAttribute(5);
         smesh.AddBdrElement(new_elem);
      }
   }

   // Exactly one boundary element must be added
   CHECK(smesh.GetNBE() == 2 * 3 + 1);

   smesh.EnsureNCMesh(true);

   // Still exactly one boundary element must be added
   CHECK(smesh.GetNBE() == 2 * 3 + 1);

   smesh.FinalizeTopology();
   smesh.Finalize();

   auto without_internal = CheckPoisson(smesh, p,
                                        5); // Exclude the internal boundary
   auto with_internal = CheckPoisson(smesh, p); // Include the internal boundary

   switch (p)
   {
      case 1:
         CHECK(with_internal == without_internal); break;
      case 2:
         CHECK(with_internal == without_internal); break;
      case 3:
         CHECK(with_internal == without_internal + 1); break;
   }

   // Now NC refine one of the attached elements, this should result in 2
   // internal boundary elements.
   for (int ref : {0, 1})
   {
      refs[0].Set(ref, Refinement::XYZ);
      auto ssmesh = Mesh(smesh);
      ssmesh.GeneralRefinement(refs);

      // There should now be four internal boundary elements, where there was
      // one before.
      CHECK(ssmesh.GetNBE() == 3 /* external boundaries of unrefined element  */
            + 4 /* internal boundaries */
            + (3 * 4) /* external boundaries of refined element */);

      // Count the number of internal boundary elements
      int num_internal = 0;
      for (int n = 0; n < ssmesh.GetNBE(); ++n)
      {
         int f, o;
         ssmesh.GetBdrElementFace(n, &f, &o);
         int e1, e2;
         ssmesh.GetFaceElements(f, &e1, &e2);
         if (e1 >= 0 && e2 >= 0 && ssmesh.GetAttribute(e1) != ssmesh.GetAttribute(e2))
         {
            ++num_internal;
         }
      }
      CHECK(num_internal == 4);

      without_internal = CheckPoisson(ssmesh, p, 5); // Exclude the internal boundary
      with_internal = CheckPoisson(ssmesh, p); // Include the internal boundary

      switch (p)
      {
         case 1:
            CHECK(with_internal == without_internal); break;
         case 2:
            CHECK(with_internal == without_internal); break;
         case 3:
            CHECK(with_internal == without_internal + 1); break;
      }
   }
}

TEST_CASE("RefinedTetsInternalBoundaries", "[NCMesh]")
{
   auto p = GENERATE(1,2,3);
   CAPTURE(p);

   auto smesh = Mesh("../../data/ref-tetrahedron.mesh");
   Array<Refinement> refs;
   refs.Append(Refinement(0, Refinement::X));
   smesh.GeneralRefinement(refs);

   // Now have a pair of elements, make the second element a different
   // attribute.
   smesh.SetAttribute(1, 2);

   REQUIRE(smesh.GetNE() == 2);
   REQUIRE(smesh.GetNBE() == 2 * 3);

   smesh.UniformRefinement();

   CHECK(smesh.GetNBE() == 2 * 3 * 4);

   // Introduce internal boundary elements
   for (int f = 0; f < smesh.GetNumFaces(); ++f)
   {
      int e1, e2;
      smesh.GetFaceElements(f, &e1, &e2);
      if (e1 >= 0 && e2 >= 0 && smesh.GetAttribute(e1) != smesh.GetAttribute(e2))
      {
         // This is the internal face between attributes.
         auto *new_elem = smesh.GetFace(f)->Duplicate(&smesh);
         new_elem->SetAttribute(5);
         smesh.AddBdrElement(new_elem);
      }
   }

   // Exactly four boundary elements must be added
   CHECK(smesh.GetNBE() == 2 * 3 * 4 + 4);

   smesh.EnsureNCMesh(true);

   // Still exactly one boundary element must be added
   CHECK(smesh.GetNBE() == 2 * 3 * 4 + 4);

   smesh.FinalizeTopology();
   smesh.Finalize();

   auto without_internal = CheckPoisson(smesh, p,
                                        5); // Exclude the internal boundary
   auto with_internal = CheckPoisson(smesh, p); // Include the internal boundary

   switch (p)
   {
      case 1:
         CHECK(with_internal == without_internal); break;
      case 2:
         CHECK(with_internal == without_internal + 3); break;
      case 3:
         CHECK(with_internal == without_internal + 10); break;
   }

   // Now NC refine all elements with the 2 attribute.

   // Mark all elements on one side of the attribute boundary to refine
   refs.DeleteAll();
   for (int n = 0; n < smesh.GetNE(); ++n)
   {
      if (smesh.GetAttribute(n) == 2)
      {
         refs.Append(Refinement{n});
      }
   }

   smesh.GeneralRefinement(refs);

   // There should now be four internal boundary elements, where there was one
   // before.
   CHECK(smesh.GetNBE() == 3 * 4 /* external boundaries of unrefined elements  */
         + 4 * 4 /* internal boundaries */
         + (3 * 4 * 4) /* external boundaries of refined elements */);

   // Count the number of internal boundary elements
   int num_internal = 0;
   for (int n = 0; n < smesh.GetNBE(); ++n)
   {
      int f, o;
      smesh.GetBdrElementFace(n, &f, &o);
      int e1, e2;
      smesh.GetFaceElements(f, &e1, &e2);
      if (e1 >= 0 && e2 >= 0 && smesh.GetAttribute(e1) != smesh.GetAttribute(e2))
      {
         ++num_internal;
      }
   }
   CHECK(num_internal == 4 * 4);

   without_internal = CheckPoisson(smesh, p, 5); // Exclude the internal boundary
   with_internal = CheckPoisson(smesh, p); // Include the internal boundary

   switch (p)
   {
      case 1:
         CHECK(with_internal == without_internal); break;
      case 2:
         CHECK(with_internal == without_internal + 3); break;
      case 3:
         CHECK(with_internal == without_internal + 10); break;
   }
}

TEST_CASE("PoissonOnReferenceCubeNC", "[NCMesh]")
{
   auto smesh = DividingPlaneMesh(false, true);
   auto p = GENERATE(1, 2, 3);
   CAPTURE(p);

   // Check that Poisson can be solved on the domain
   CheckPoisson(smesh, p);
   auto ref_type = char(GENERATE(Refinement::X, Refinement::Y, Refinement::Z,
                                 Refinement::XY, Refinement::XZ, Refinement::YZ,
                                 Refinement::XYZ));
   CAPTURE(ref_type);

   const real_t scale = GENERATE(0.5, 0.25);
   CAPTURE(scale);

   Array<Refinement> refs(1);
   for (auto refined_elem : {0}) // The left or the right element
   {
      auto ssmesh = Mesh(smesh);
      refs[0].Set(refined_elem, ref_type, scale);

      ssmesh.GeneralRefinement(refs);
      ssmesh.FinalizeTopology();
      ssmesh.Finalize();

      CAPTURE(refined_elem);
      CheckPoisson(ssmesh, p);
   }
}

TEST_CASE("PoissonOnReferenceTetNC", "[NCMesh]")
{
   auto p = GENERATE(1, 2, 3);
   CAPTURE(p);
   auto orientation = GENERATE(1,3,5);
   auto smesh = OrientedTriFaceMesh(orientation, true);
   smesh.EnsureNCMesh(true);

   CheckPoisson(smesh, p);
   Array<int> refs(1);
   for (auto refined_elem : {0, 1})
   {
      auto ssmesh = Mesh(smesh);
      refs[0] = refined_elem;
      ssmesh.GeneralRefinement(refs);
      ssmesh.FinalizeTopology();
      ssmesh.Finalize();

      CAPTURE(refined_elem);
      CheckPoisson(ssmesh, p);
   }
}

TEST_CASE("TetBoundaryRefinement", "[NCMesh]")
{
   auto smesh = Mesh("../../data/ref-tetrahedron.mesh");

   smesh.FinalizeTopology();
   smesh.Finalize(true);
   smesh.UniformRefinement();

   smesh.EnsureNCMesh(true);

   CHECK(smesh.GetNBE() == 4 * 4);

   // Loop over elements and mark for refinement if any vertices match the
   // original
   auto refine_corners = [&]()
   {
      Array<int> vertices, elements;
      // reference vertices of (0,0,0), (1,0,0), (0,1,0), (0,0,1) are [0,3]
      auto original_vert = [](int i) { return i >= 0 && i <= 3; };
      for (int n = 0; n < smesh.GetNE(); ++n)
      {
         smesh.GetElementVertices(n, vertices);
         if (std::any_of(vertices.begin(), vertices.end(), original_vert))
         {
            elements.Append(n);
         }
      }

      smesh.GeneralRefinement(elements);
      smesh.FinalizeTopology();
      smesh.Finalize();
   };

   constexpr int max_ref_levels = 4;
   for (int r = 0; r < max_ref_levels; r++)
   {
      refine_corners();
      CHECK(smesh.GetNBE() == 4 * (4 + 3 * 3 * (r + 1)));
   }
}

TEST_CASE("TetInternalBoundaryRefinement", "[NCMesh]")
{
   auto orientation = GENERATE(1,3,5);
   auto smesh = OrientedTriFaceMesh(orientation, true);

   smesh.FinalizeTopology();
   smesh.Finalize(true);
   smesh.UniformRefinement();
   smesh.EnsureNCMesh(true);
   CHECK(smesh.GetNBE() == (2*3 + 1) * 4);

   CHECK(CountEssentialDof<H1_FECollection>(smesh, 1,
                                            smesh.bdr_attributes.Max()) == 6);
   CHECK(CountEssentialDof<H1_FECollection>(smesh, 2,
                                            smesh.bdr_attributes.Max()) == 6 + 3 * 3);
   CHECK(CountEssentialDof<H1_FECollection>(smesh, 3,
                                            smesh.bdr_attributes.Max()) == 10 + 3 * 6);

   int refined_attribute = GENERATE(1,2);
   int ref_level = GENERATE(1,2,3);
   for (int r = 0; r < ref_level; r++)
   {
      Array<int> el_to_refine;
      for (int n = 0; n < smesh.GetNE(); n++)
      {
         if (smesh.GetAttribute(n) == refined_attribute)
         {
            el_to_refine.Append(n);
         }
      }
      smesh.GeneralRefinement(el_to_refine);
   }

   // Refining on only one side of the boundary face should not change the
   // number of essential true dofs
   CHECK(CountEssentialDof<H1_FECollection>(smesh, 1,
                                            smesh.bdr_attributes.Max()) == 6);
   CHECK(CountEssentialDof<H1_FECollection>(smesh, 2,
                                            smesh.bdr_attributes.Max()) == 6 + 3 * 3);
   CHECK(CountEssentialDof<H1_FECollection>(smesh, 3,
                                            smesh.bdr_attributes.Max()) == 10 + 3 * 6);

   // The number of boundary faces should have increased.
   CHECK(smesh.GetNBE() == 3 * 4 + (3 + 1) * std::pow(4, 1+ref_level));
}

TEST_CASE("TetInternalBoundaryTetStarMesh", "[NCMesh]")
{
   auto smesh = TetStarMesh();
   smesh.EnsureNCMesh(true);

   SECTION("Unrefined")
   {
      CHECK(smesh.GetNBE() == 4 * 3 + 4);
      CHECK(CountEssentialDof<H1_FECollection>(smesh, 1,
                                               smesh.bdr_attributes.Max()) == 4);
      CHECK(CountEssentialDof<H1_FECollection>(smesh, 2,
                                               smesh.bdr_attributes.Max()) == 4 + 6);
      CHECK(CountEssentialDof<H1_FECollection>(smesh, 3,
                                               smesh.bdr_attributes.Max()) == 4 + 6*2 + 4*1);
      CHECK(CountEssentialDof<H1_FECollection>(smesh, 4,
                                               smesh.bdr_attributes.Max()) == 4 + 6*3 + 4*3);

      CHECK(CountEssentialDof<ND_FECollection>(smesh, 1,
                                               smesh.bdr_attributes.Max()) == 6);
      CHECK(CountEssentialDof<ND_FECollection>(smesh, 2,
                                               smesh.bdr_attributes.Max()) == 20);
      CHECK(CountEssentialDof<ND_FECollection>(smesh, 3,
                                               smesh.bdr_attributes.Max()) == 42);
      CHECK(CountEssentialDof<ND_FECollection>(smesh, 4,
                                               smesh.bdr_attributes.Max()) == 72);
   }

   SECTION("Refined")
   {
      int refined_attribute = GENERATE(1,2,3,4,5);
      int ref_level = GENERATE(1,2,3);
      for (int r = 0; r < ref_level; r++)
      {
         Array<int> el_to_refine;
         for (int n = 0; n < smesh.GetNE(); n++)
         {
            if (smesh.GetAttribute(n) == refined_attribute)
            {
               el_to_refine.Append(n);
            }
         }
         smesh.GeneralRefinement(el_to_refine);
      }

      // Refining on only one side of the boundary face should not change the
      // number of essential true dofs
      CHECK(CountEssentialDof<H1_FECollection>(smesh, 1,
                                               smesh.bdr_attributes.Max()) == 4);
      CHECK(CountEssentialDof<H1_FECollection>(smesh, 2,
                                               smesh.bdr_attributes.Max()) == 4 + 6);
      CHECK(CountEssentialDof<H1_FECollection>(smesh, 3,
                                               smesh.bdr_attributes.Max()) == 4 + 6*2 + 4*1);
      CHECK(CountEssentialDof<H1_FECollection>(smesh, 4,
                                               smesh.bdr_attributes.Max()) == 4 + 6*3 + 4*3);

      CHECK(CountEssentialDof<ND_FECollection>(smesh, 1,
                                               smesh.bdr_attributes.Max()) == 6);
      CHECK(CountEssentialDof<ND_FECollection>(smesh, 2,
                                               smesh.bdr_attributes.Max()) == 6 * 2 + 4 * 2); // 2 per edge, 2 per face
      CHECK(CountEssentialDof<ND_FECollection>(smesh, 3,
                                               smesh.bdr_attributes.Max()) == 42);
      CHECK(CountEssentialDof<ND_FECollection>(smesh, 4,
                                               smesh.bdr_attributes.Max()) == 72);

      // The number of boundary faces should have increased.
      CHECK(smesh.GetNBE() == 3 * 4 + 4 * std::pow(4,ref_level));
   }
}

TEST_CASE("DividingPlaneMesh", "[NCMesh]")
{
   auto refine_attribute = [](Mesh& mesh, int attr, int ref_level)
   {
      for (int r = 0; r < ref_level; r++)
      {
         Array<int> el_to_refine;
         for (int n = 0; n < mesh.GetNE(); n++)
         {
            if (mesh.GetAttribute(n) == attr)
            {
               el_to_refine.Append(n);
            }
         }
         mesh.GeneralRefinement(el_to_refine);
      }
   };

   SECTION("Hex")
   {
      auto mesh = DividingPlaneMesh(false);
      mesh.EnsureNCMesh(true);

      CHECK(mesh.GetNBE() == 2 * 5 + 1);
      CHECK(mesh.GetNE() == 2);

      auto attr = GENERATE(1,2);
      auto ref_level = GENERATE(1,2);

      const int num_vert = ref_level == 1 ? 5*5 : 9*9;
      const int num_edge = ref_level == 1 ? 2*4*5 : 2*8*9;
      const int num_face = ref_level == 1 ? 4*4 : 8*8;

      SECTION("H1Hex")
      {
         mesh.UniformRefinement();
         CHECK(CountEssentialDof<H1_FECollection, true>(mesh, 1,
                                                        mesh.bdr_attributes.Max()) == 3*3);
         CHECK(CountEssentialDof<H1_FECollection, true>(mesh, 2,
                                                        mesh.bdr_attributes.Max()) == 5*5);
         CHECK(CountEssentialDof<H1_FECollection, true>(mesh, 3,
                                                        mesh.bdr_attributes.Max()) == 7*7);
         CHECK(CountEssentialDof<H1_FECollection, false>(mesh, 1,
                                                         mesh.bdr_attributes.Max()) == 3*3);
         CHECK(CountEssentialDof<H1_FECollection, false>(mesh, 2,
                                                         mesh.bdr_attributes.Max()) == 5*5);
         CHECK(CountEssentialDof<H1_FECollection, false>(mesh, 3,
                                                         mesh.bdr_attributes.Max()) == 7*7);

         refine_attribute(mesh, attr, ref_level);

         CHECK(CountEssentialDof<H1_FECollection>(mesh, 1,
                                                  mesh.bdr_attributes.Max()) == 3*3);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 2,
                                                  mesh.bdr_attributes.Max()) == 5*5);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 3,
                                                  mesh.bdr_attributes.Max()) == 7*7);

         // Add the slave face dofs, then subtract off the vertex dofs which are
         // double counted due to being shared.
         CHECK(CountEssentialDof<H1_FECollection, false>(mesh, 1,
                                                         mesh.bdr_attributes.Max()) == 3*3 + num_vert - 3*3);
         CHECK(CountEssentialDof<H1_FECollection, false>(mesh, 2,
                                                         mesh.bdr_attributes.Max()) == 5*5 + num_vert + num_edge + num_face - 3*3);
         CHECK(CountEssentialDof<H1_FECollection, false>(mesh, 3,
                                                         mesh.bdr_attributes.Max()) == 7*7 + num_vert + 2*num_edge + 4*num_face - 3*3);

      }

      SECTION("NDHex")
      {
         CHECK(CountEssentialDof<ND_FECollection, true>(mesh, 1,
                                                        mesh.bdr_attributes.Max()) == 4);
         CHECK(CountEssentialDof<ND_FECollection, true>(mesh, 2,
                                                        mesh.bdr_attributes.Max()) == 4*2 + 2*2);
         CHECK(CountEssentialDof<ND_FECollection, true>(mesh, 3,
                                                        mesh.bdr_attributes.Max()) == 4*3 + 2*2*3);
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 1,
                                                         mesh.bdr_attributes.Max()) == 4);
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 2,
                                                         mesh.bdr_attributes.Max()) == 4*2 + 2*2);
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 3,
                                                         mesh.bdr_attributes.Max()) == 4*3 + 2*2*3);

         mesh.UniformRefinement();
         const int initial_num_edge = 12;
         const int initial_num_face = 4;
         CHECK(CountEssentialDof<ND_FECollection, true>(mesh, 1,
                                                        mesh.bdr_attributes.Max()) == initial_num_edge);
         CHECK(CountEssentialDof<ND_FECollection, true>(mesh, 2,
                                                        mesh.bdr_attributes.Max()) == initial_num_edge*2 + initial_num_face*2*2);
         CHECK(CountEssentialDof<ND_FECollection, true>(mesh, 3,
                                                        mesh.bdr_attributes.Max()) == initial_num_edge*3 + initial_num_face*2*2*3);
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 1,
                                                         mesh.bdr_attributes.Max()) == initial_num_edge);
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 2,
                                                         mesh.bdr_attributes.Max()) == initial_num_edge*2 + initial_num_face*2*2);
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 3,
                                                         mesh.bdr_attributes.Max()) == initial_num_edge*3 + initial_num_face*2*2*3);

         refine_attribute(mesh, attr, ref_level);
         CHECK(CountEssentialDof<ND_FECollection, true>(mesh, 1,
                                                        mesh.bdr_attributes.Max()) == initial_num_edge);
         CHECK(CountEssentialDof<ND_FECollection, true>(mesh, 2,
                                                        mesh.bdr_attributes.Max()) == initial_num_edge*2 + initial_num_face*2*2);
         CHECK(CountEssentialDof<ND_FECollection, true>(mesh, 3,
                                                        mesh.bdr_attributes.Max()) == initial_num_edge*3 + initial_num_face*2*2*3);
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 1,
                                                         mesh.bdr_attributes.Max()) == (num_edge+initial_num_edge));
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 2,
                                                         mesh.bdr_attributes.Max()) == (num_edge+initial_num_edge)*2 +
               (num_face+initial_num_face)*2*2);
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 3,
                                                         mesh.bdr_attributes.Max()) == (num_edge+initial_num_edge)*3 +
               (num_face+initial_num_face)*2*2*3);
      }
   }

   SECTION("Tet")
   {
      auto mesh = DividingPlaneMesh(true);
      mesh.EnsureNCMesh(true);

      CHECK(mesh.GetNBE() == 2 * 5 * 2 + 2);
      CHECK(mesh.GetNE() == 2 * 6);

      auto attr = GENERATE(1,2);
      auto ref_level = GENERATE(1,2);
      CAPTURE(attr);
      CAPTURE(ref_level);

      const int initial_num_vert = 4;
      const int initial_num_edge = 5;
      const int initial_num_face = 2;

      const int num_vert = ref_level == 1 ? 9 : 25;
      const int num_edge = ref_level == 1 ? 16 : 56;
      const int num_face = ref_level == 1 ? 8 : 32;

      SECTION("H1Tet")
      {
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 1,
                                                  mesh.bdr_attributes.Max()) == initial_num_vert);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 2,
                                                  mesh.bdr_attributes.Max()) == initial_num_vert + initial_num_edge);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 3,
                                                  mesh.bdr_attributes.Max()) == initial_num_vert + 2*initial_num_edge +
               initial_num_face);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 4,
                                                  mesh.bdr_attributes.Max()) == initial_num_vert + 3*initial_num_edge +
               3*initial_num_face);
         CHECK(CountEssentialDof<H1_FECollection, false>(mesh, 1,
                                                         mesh.bdr_attributes.Max()) == initial_num_vert);
         CHECK(CountEssentialDof<H1_FECollection, false>(mesh, 2,
                                                         mesh.bdr_attributes.Max()) == initial_num_vert + initial_num_edge);
         CHECK(CountEssentialDof<H1_FECollection, false>(mesh, 3,
                                                         mesh.bdr_attributes.Max()) == initial_num_vert + 2*initial_num_edge +
               initial_num_face);
         CHECK(CountEssentialDof<H1_FECollection, false>(mesh, 4,
                                                         mesh.bdr_attributes.Max()) == initial_num_vert + 3*initial_num_edge +
               3*initial_num_face);

         refine_attribute(mesh, attr, ref_level);

         CHECK(CountEssentialDof<H1_FECollection>(mesh, 1,
                                                  mesh.bdr_attributes.Max()) == initial_num_vert);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 2,
                                                  mesh.bdr_attributes.Max()) == initial_num_vert + initial_num_edge);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 3,
                                                  mesh.bdr_attributes.Max()) == initial_num_vert + 2*initial_num_edge +
               initial_num_face);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 4,
                                                  mesh.bdr_attributes.Max()) == initial_num_vert + 3*initial_num_edge +
               3*initial_num_face);
         CHECK(CountEssentialDof<H1_FECollection, false>(mesh, 1,
                                                         mesh.bdr_attributes.Max()) == num_vert);
         CHECK(CountEssentialDof<H1_FECollection, false>(mesh, 2,
                                                         mesh.bdr_attributes.Max()) == num_vert + num_edge + initial_num_edge);
         CHECK(CountEssentialDof<H1_FECollection, false>(mesh, 3,
                                                         mesh.bdr_attributes.Max()) == num_vert + 2*num_edge + num_face +
               2*initial_num_edge + initial_num_face);
         CHECK(CountEssentialDof<H1_FECollection, false>(mesh, 4,
                                                         mesh.bdr_attributes.Max()) == num_vert + 3*num_edge + 3*num_face +
               3*initial_num_edge + 3*initial_num_face);
      }

      SECTION("NDTet")
      {
         CHECK(CountEssentialDof<ND_FECollection>(mesh, 1,
                                                  mesh.bdr_attributes.Max()) == 5);
         CHECK(CountEssentialDof<ND_FECollection>(mesh, 2,
                                                  mesh.bdr_attributes.Max()) == 14);
         CHECK(CountEssentialDof<ND_FECollection>(mesh, 3,
                                                  mesh.bdr_attributes.Max()) == 27);
         CHECK(CountEssentialDof<ND_FECollection>(mesh, 4,
                                                  mesh.bdr_attributes.Max()) == 44);
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 1,
                                                         mesh.bdr_attributes.Max()) == 5);
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 2,
                                                         mesh.bdr_attributes.Max()) == 14);
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 3,
                                                         mesh.bdr_attributes.Max()) == 27);
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 4,
                                                         mesh.bdr_attributes.Max()) == 44);

         refine_attribute(mesh, attr, ref_level);

         CHECK(CountEssentialDof<ND_FECollection>(mesh, 1,
                                                  mesh.bdr_attributes.Max()) == 5);
         CHECK(CountEssentialDof<ND_FECollection>(mesh, 2,
                                                  mesh.bdr_attributes.Max()) == 14);
         CHECK(CountEssentialDof<ND_FECollection>(mesh, 3,
                                                  mesh.bdr_attributes.Max()) == 27);
         CHECK(CountEssentialDof<ND_FECollection>(mesh, 4,
                                                  mesh.bdr_attributes.Max()) == 44);

         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 1,
                                                         mesh.bdr_attributes.Max()) == 5 + num_edge);
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 2,
                                                         mesh.bdr_attributes.Max()) == 14 + 2 * num_edge + 2*num_face);
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 3,
                                                         mesh.bdr_attributes.Max()) == 27 + 3 * num_edge + 2*3*num_face);
      }
   }
}

TEST_CASE("TetFaceFlips", "[NCMesh]")
{
   auto orientation = GENERATE(1,3,5);
   CAPTURE(orientation);
   auto smesh = OrientedTriFaceMesh(orientation, true);

   // A smooth function in each vector component
   constexpr int order = 3, dim = 3, quadrature_order = 4;
   constexpr real_t kappa = 2 * M_PI;
   auto E_exact = [=](const Vector &x, Vector &E)
   {
      E(0) = cos(kappa * x(1));
      E(1) = cos(kappa * x(2));
      E(2) = cos(kappa * x(0));
   };
   VectorFunctionCoefficient E_coeff(dim, E_exact);

   auto check_serial_nd_conformal = [&](Mesh &mesh, int num_essential_tdof,
                                        int num_essential_vdof)
   {
      ND_FECollection fe_collection(order, dim);
      FiniteElementSpace fe_space(&mesh, &fe_collection);
      GridFunction E(&fe_space);

      E.ProjectCoefficient(E_coeff);

      auto *P = fe_space.GetProlongationMatrix();
      if (P != nullptr)
      {
         // Projection does not respect the non-conformal constraints. Extract
         // the true (conformal) and prolongate to get the NC respecting
         // projection.
         auto E_true = E.GetTrueVector();
         P->Mult(E_true, E);
      }
      mesh.EnsureNodes();
      GridFunction * const coords = mesh.GetNodes();

      const auto &ir = IntRules.Get(Geometry::Type::TRIANGLE, quadrature_order);
      IntegrationRule left_eir(ir.GetNPoints()),
                      right_eir(ir.GetNPoints()); // element integration rules

      Array<int> bdr_attr_is_ess = mesh.bdr_attributes, tdof_list;
      bdr_attr_is_ess = 0;
      bdr_attr_is_ess.Last() = 1;
      fe_space.GetEssentialTrueDofs(bdr_attr_is_ess, tdof_list);

      Array<int> ess_vdof_marker, vdof_list;
      fe_space.GetEssentialVDofs(bdr_attr_is_ess, ess_vdof_marker);
      fe_space.MarkerToList(ess_vdof_marker, vdof_list);

      CHECK(num_essential_tdof == tdof_list.Size());
      if (num_essential_vdof != -1)
      {
         CHECK(num_essential_vdof == vdof_list.Size());
      }

      bool y_valid = true, z_valid = true;
      for (int n = 0; n < mesh.GetNBE(); n++)
      {
         // NOTE: only works for internal boundaries
         if (bdr_attr_is_ess[mesh.GetBdrAttribute(n) - 1])
         {
            auto f = mesh.GetBdrElementFaceIndex(n);
            auto &face_element_transform = *mesh.GetFaceElementTransformations(f);

            if (face_element_transform.Elem2 == nullptr)
            {
               // not internal, nothing to check.
               continue;
            }

            face_element_transform.Loc1.Transform(ir, left_eir);
            face_element_transform.Loc2.Transform(ir, right_eir);

            constexpr real_t tol = 1e-14;
            REQUIRE(left_eir.GetNPoints() == ir.GetNPoints());
            REQUIRE(right_eir.GetNPoints() == ir.GetNPoints());
            Vector left_val, right_val;
            for (int i = 0; i < ir.GetNPoints(); i++)
            {
               face_element_transform.Elem1->SetIntPoint(&left_eir[i]);
               coords->GetVectorValue(*face_element_transform.Elem1, left_eir[i], left_val);
               coords->GetVectorValue(*face_element_transform.Elem1, left_eir[i], right_val);
               REQUIRE(std::abs(left_val(0) - right_val(0)) < tol);
               REQUIRE(std::abs(left_val(1) - right_val(1)) < tol);
               REQUIRE(std::abs(left_val(2) - right_val(2)) < tol);
               E.GetVectorValue(*face_element_transform.Elem1, left_eir[i], left_val);

               face_element_transform.Elem2->SetIntPoint(&right_eir[i]);
               E.GetVectorValue(*face_element_transform.Elem2, right_eir[i], right_val);

               // Check that the second and third rows agree. The y and z should
               // agree as the normal is in the x direction
               y_valid &= (std::abs(left_val(1) - right_val(1)) < tol);
               z_valid &= (std::abs(left_val(2) - right_val(2)) < tol);
            }
         }
      }
      CHECK(y_valid);
      CHECK(z_valid);
   };

   SECTION("Conformal")
   {
      const int ntdof = 3*3 + 3*2;
      const int nvdof = ntdof;
      check_serial_nd_conformal(smesh, ntdof, nvdof);
   }

   SECTION("Nonconformal")
   {
      smesh.EnsureNCMesh(true);
      const int ntdof = 3*3 + 3*2;
      const int nvdof = ntdof;
      check_serial_nd_conformal(smesh, ntdof, nvdof);
   }

   SECTION("ConformalUniformRefined")
   {
      smesh.UniformRefinement();
      const int ntdof = 9*3 + 4*3*2;
      const int nvdof = ntdof;
      check_serial_nd_conformal(smesh, ntdof, nvdof);
   }

   SECTION("NonconformalUniformRefined")
   {
      smesh.EnsureNCMesh(true);
      smesh.UniformRefinement();
      const int ntdof = 9*3 + 4*3*2;
      const int nvdof = ntdof;
      check_serial_nd_conformal(smesh, ntdof, nvdof);
   }

   SECTION("NonconformalRefined")
   {
      smesh.EnsureNCMesh(true);
      int ref_level = GENERATE(1, 2);
      CAPTURE(ref_level);
      for (int r = 0; r < ref_level; r++)
      {
         Array<int> el_to_refine;
         for (int n = 0; n < smesh.GetNE(); n++)
         {
            if (smesh.GetAttribute(n) == 2)
            {
               el_to_refine.Append(n);
            }
         }
         smesh.GeneralRefinement(el_to_refine);
      }
      const int ntdof = 3*3 + 3*2;
      const int nvdof = ntdof + (ref_level == 1 ? 9*3 + 4*3*2 : 30*3 + 16*3*2);
      check_serial_nd_conformal(smesh, ntdof, nvdof);
   }

   SECTION("NonconformalLevelTwoRefined")
   {
      smesh.EnsureNCMesh(true);
      Array<int> el_to_refine;

      smesh.UniformRefinement();

      const int ntdof = 9*3 + 4*3*2;
      el_to_refine.SetSize(1);

      auto n = GENERATE(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
      auto m = GENERATE(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22);

      if (n < smesh.GetNE() && smesh.GetAttribute(n) == 2)
      {
         el_to_refine[0] = n;
         CAPTURE(n);
         smesh.GeneralRefinement(el_to_refine);
         check_serial_nd_conformal(smesh, ntdof, -1);

         if (smesh.GetAttribute(m) == 2)
         {
            el_to_refine[0] = m;
            CAPTURE(m);
            smesh.GeneralRefinement(el_to_refine);
            check_serial_nd_conformal(smesh, ntdof, -1);
         }
      }

   }
}

TEST_CASE("RP=I", "[NCMesh]")
{
   auto check_fespace = [](const FiniteElementSpace& fespace)
   {
      auto * const R = fespace.GetConformingRestriction();
      auto * const P = fespace.GetConformingProlongation();

      REQUIRE(R != nullptr);
      REQUIRE(P != nullptr);

      // Vector notation
      Vector e_i(R->Height()), e_j(P->Width());
      Vector Rrow(R->Width()), Pcol(P->Height());
      bool valid = true;
      for (int i = 0; i < R->Height(); i++)
      {
         e_i = 0.0;
         e_i.HostReadWrite();
         e_i(i) = 1.0;
         R->MultTranspose(e_i, Rrow);
         for (int j = 0; j < P->Width(); j++)
         {
            e_j = 0.0;
            e_j.HostReadWrite();
            e_j(j) = 1.0;
            P->Mult(e_j, Pcol);

            valid &= (Rrow * Pcol == (i == j ? 1.0 : 0.0));
         }
      }
      CHECK(valid);

      // Index notation
      CHECK(R->Height() == P->Width());
      CHECK(R->Width() == P->Height());
      valid = true;
      for (int i = 0; i < R->Height(); i++)
         for (int j = 0; j < P->Width(); j++)
         {
            real_t dot = 0.0;
            for (int k = 0; k < R->Width(); k++)
            {
               dot += (*R)(i,k)*(*P)(k,j);
            }
            CHECK(dot == (i == j ? 1.0 : 0.0));
            valid &= (dot == (i == j ? 1.0 : 0.0));
         }
      CHECK(valid);
   };

   SECTION("Hex")
   {
      const int dim = 3;
      const int order = GENERATE(1, 2);
      // Split the hex into a pair, then isotropically refine one of them.
      Mesh mesh("../../data/ref-cube.mesh");
      Array<Refinement> refinements(1);
      refinements[0].Set(0, Refinement::X);

      mesh.GeneralRefinement(refinements);
      refinements[0].SetType(Refinement::XYZ);

      mesh.GeneralRefinement(refinements);
      SECTION("ND")
      {
         ND_FECollection fec(order, dim);
         FiniteElementSpace fespace(&mesh, &fec);
         check_fespace(fespace);
      }
      SECTION("H1")
      {
         H1_FECollection fec(order, dim);
         FiniteElementSpace fespace(&mesh, &fec);
         check_fespace(fespace);
      }
   }

   SECTION("Tet")
   {
      const int dim = 3;
      const int order = GENERATE(1, 2);
      // Split the hex into a pair, then isotropically refine one of them.
      Mesh mesh("../../data/ref-tetrahedron.mesh");
      Array<Refinement> refinements(1);
      refinements[0].Set(0, Refinement::X);
      mesh.GeneralRefinement(refinements);
      mesh.EnsureNCMesh(true);
      refinements[0].SetType(Refinement::XYZ);
      mesh.GeneralRefinement(refinements);
      SECTION("ND")
      {
         ND_FECollection fec(order, dim);
         FiniteElementSpace fespace(&mesh, &fec);
         check_fespace(fespace);
      }
      SECTION("H1")
      {
         H1_FECollection fec(order, dim);
         FiniteElementSpace fespace(&mesh, &fec);
         check_fespace(fespace);
      }
   }
}

TEST_CASE("InternalBoundaryProjectBdrCoefficient", "[NCMesh]")
{
   auto test_project_H1 = [](Mesh &mesh, int order, double coef)
   {
      MFEM_ASSERT(std::abs(coef) > 0,
                  "Non zero coef value required for meaningful test.");
      H1_FECollection fe_collection(order, mesh.SpaceDimension());
      FiniteElementSpace fe_space(&mesh, &fe_collection);
      GridFunction x(&fe_space);
      x = -coef;
      ConstantCoefficient c(coef);

      // Check projecting on the internal face sets essential dof.
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 0;
      ess_bdr.Last() = 1; // internal boundary
      x.ProjectBdrCoefficient(c, ess_bdr);

      Array<int> ess_vdofs_list, ess_vdofs_marker;
      fe_space.GetEssentialVDofs(ess_bdr, ess_vdofs_marker);
      fe_space.MarkerToList(ess_vdofs_marker, ess_vdofs_list);
      for (auto ess_dof : ess_vdofs_list)
      {
         CHECK(x[ess_dof] == Approx(coef).epsilon(1e-8));
      }

      int iess = 0;
      for (int i = 0; i < x.Size(); i++)
      {
         if (iess < ess_vdofs_list.Size() && i == ess_vdofs_list[iess])
         {
            iess++;
            continue;
         }
         CHECK(x[i] == Approx(-coef).epsilon(1e-8));
      }

   };

   auto OneSidedNCRefine = [](Mesh &mesh)
   {
      // Pick one element attached to the new boundary attribute and refine.
      const auto interface_attr = mesh.bdr_attributes.Max();
      Array<int> el_to_ref;
      for (int nbe = 0; nbe < mesh.GetNBE(); nbe++)
      {
         if (mesh.GetBdrAttribute(nbe) == interface_attr)
         {
            int f, o, e1, e2;
            mesh.GetBdrElementFace(nbe, &f, &o);
            mesh.GetFaceElements(f, &e1, &e2);
            el_to_ref.Append(e1);
         }
      }
      mesh.GeneralRefinement(el_to_ref);
      return;
   };

   SECTION("Hex")
   {
      auto smesh = DividingPlaneMesh(false, true);
      smesh.EnsureNCMesh(true);
      OneSidedNCRefine(smesh);
      test_project_H1(smesh, 2, 0.25);
   }

   SECTION("Tet")
   {
      auto smesh = DividingPlaneMesh(true, true);
      smesh.EnsureNCMesh(true);
      OneSidedNCRefine(smesh);
      test_project_H1(smesh, 3, 0.25);
   }
}

// Test case: Essential true DOFs of an interior boundary of a 2D non-conforming
//            mesh refined on one side of it, see the parallel version above.
TEST_CASE("InteriorBoundaryEssentialDofs2D", "[NCMesh]")
{
   const bool tri_mesh = GENERATE(false, true);
   const int order = GENERATE(1, 2, 3);
   const bool ess_interior = GENERATE(true, false);
   CAPTURE(tri_mesh, order, ess_interior);

   auto mesh = OneSidedRefinedDividingLineMesh(tri_mesh);
   const int interior_attr = mesh.bdr_attributes.Max();
   Array<int> ess_bdr(interior_attr);
   ess_bdr = ess_interior ? 0 : 1;
   ess_bdr[interior_attr - 1] = ess_interior ? 1 : 0;

   H1_FECollection h1_fec(order, 2);
   ND_FECollection nd_fec(order, 2);
   for (const FiniteElementCollection *fec :
        std::vector<const FiniteElementCollection *> {&h1_fec, &nd_fec})
   {
      CAPTURE(fec->Name());
      FiniteElementSpace fes(&mesh, fec);
      Array<int> ess_tdof_list, ess_tdof_marker;
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      FiniteElementSpace::ListToMarker(ess_tdof_list, fes.GetTrueVSize(),
                                       ess_tdof_marker);
      const SparseMatrix *R = fes.GetConformingRestriction();
      REQUIRE(R != nullptr);
      std::vector<int> tdof_of_ldof(fes.GetVSize(), -1);
      for (int t = 0; t < R->Height(); t++)
      {
         tdof_of_ldof[R->GetRowColumns(t)[0]] = t;
      }
      auto tdof_of = [&](int ldof) { return tdof_of_ldof[ldof]; };
      const int checked = CheckInteriorBoundaryEssentialDofs(
                             mesh, fes, ess_tdof_marker, ess_interior, tdof_of);
      REQUIRE(checked > 0);
   }
}

TEST_CASE("NCMesh high vertex reference count", "[NCMesh]")
{
   // Test large vertex reference counts, supported up to the limit of
   // std::uint16_t. K tets share a central axis edge, so the two axis
   // vertices each have reference count K.
   auto make_spindle = [](int K)
   {
      Mesh mesh(3, K + 2, K);
      Array<int> ring(K);
      for (int i = 0; i < K; i++)
      {
         const real_t t = 2.0 * M_PI * i / K;
         ring[i] = mesh.AddVertex(std::cos(t), std::sin(t), 0.0);
      }
      const int c = mesh.AddVertex(0.0, 0.0, -1.0);
      const int a = mesh.AddVertex(0.0, 0.0,  1.0);
      for (int i = 0; i < K; i++)
      {
         int v[4] = {c, a, ring[i], ring[(i + 1) % K]};
         mesh.AddTet(v, 1);
      }
      mesh.FinalizeTetMesh(1, 0, true);
      return mesh;
   };

   const int K = GENERATE(255, 256, 512);

   Mesh mesh = make_spindle(K);
   const int nv = mesh.GetNV();
   REQUIRE(mesh.GetNE() == K);

   mesh.EnsureNCMesh(true);

   REQUIRE(mesh.GetNV() == nv);
   REQUIRE_FALSE(mesh.Conforming());

   // The non-conforming structure must be usable.
   H1_FECollection fec(1, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);
   REQUIRE(fes.GetVSize() == nv);

   mesh.UniformRefinement();
   REQUIRE(mesh.GetNE() == 8 * K);
}

} // namespace mfem
