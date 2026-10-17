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

#include <sstream>

using namespace mfem;

namespace
{

Mesh MakeEntitySetQuadMesh()
{
   std::istringstream input(R"(MFEM mesh v1.0

dimension
2

elements
1
1 3 0 1 2 3

boundary
4
1 1 0 1
1 1 1 2
1 1 2 3
1 1 3 0

vertices
4
2
0 0
1 0
1 1
0 1

MFEM sets v1.0

vertex_sets
1

Corner
1
0

edge_sets
1

Bottom
1
0 1

element_sets
1

Domain
1
0
)");
   return Mesh(input, 1, 0);
}

Mesh MakeEntitySetTetraMesh()
{
   std::istringstream input(R"(MFEM mesh v1.0

dimension
3

elements
1
1 4 0 1 2 3

boundary
4
1 2 0 2 1
1 2 0 1 3
1 2 1 2 3
1 2 2 0 3

vertices
4
3
0 0 0
1 0 0
0 1 0
0 0 1

MFEM sets v1.0

vertex_sets
1

Corner
1
0

edge_sets
1

Edge
1
0 1

face_sets
1

Face
1
2 0 1 2

element_sets
1

Domain
1
0
)");
   return Mesh(input, 1, 0);
}

Mesh MakeEntitySetHexMesh()
{
   std::istringstream input(R"(MFEM mesh v1.0

dimension
3

elements
1
1 5 0 1 2 3 4 5 6 7

boundary
6
1 3 0 3 2 1
1 3 0 1 5 4
1 3 1 2 6 5
1 3 2 3 7 6
1 3 3 0 4 7
1 3 4 5 6 7

vertices
8
3
0 0 0
1 0 0
1 1 0
0 1 0
0 0 1
1 0 1
1 1 1
0 1 1

MFEM sets v1.0

vertex_sets
1

Corner
1
0

edge_sets
1

Edge
1
0 1

face_sets
1

Face
1
3 0 1 2 3

element_sets
1

Domain
1
0
)");
   return Mesh(input, 1, 0);
}

void AddAllEntitySets(Mesh &mesh)
{
   EntitySets &sets = mesh.GetOrCreateEntitySets();
   const int vertices = sets.AddSet(EntitySets::VERTEX, "Vertices");
   const int edges = sets.AddSet(EntitySets::EDGE, "Edges");
   const int elements = sets.AddSet(EntitySets::ELEMENT, "Elements");

   for (int i = 0; i < mesh.GetNV(); i++)
   {
      sets.AddEntity(EntitySets::VERTEX, vertices, i);
   }
   for (int i = 0; i < mesh.GetNEdges(); i++)
   {
      sets.AddEntity(EntitySets::EDGE, edges, i);
   }
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      sets.AddEntity(EntitySets::ELEMENT, elements, i);
   }
}

} // namespace

TEST_CASE("Entity sets can be created from physical point predicates", "[Mesh]")
{
   Mesh mesh = Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL, false,
                                     2.0, 1.0);
   EntitySets &sets = mesh.GetOrCreateEntitySets();
   constexpr real_t tolerance = 1.e-12;

   const int center_set = sets.AddElementSetByPredicate(
                              "Center", [](const Vector &x)
   {
      return x[0] < 1.0;
   });
   const int vertex_set = sets.AddElementSetByPredicate(
                              "Vertices", [=](const Vector &x)
   {
      return x[0] <= 1.0 + tolerance;
   }, EntitySets::ElementSetPointRule::ALL_VERTICES);

   FunctionCoefficient level_set([](const Vector &x) { return x[0] - 1.0; });
   const int level_set_elements = sets.AddElementSetByLevelSet(
                                      "LevelSet", level_set, tolerance,
                                      EntitySets::ElementSetPointRule::ALL_VERTICES);

   for (const int set : {center_set, vertex_set, level_set_elements})
   {
      REQUIRE(sets.GetNumEntities(EntitySets::ELEMENT, set) == 1);
      REQUIRE(sets(EntitySets::ELEMENT, set).count(0) == 1);
   }
}

TEST_CASE("Entity sets can be created from element lists and markers", "[Mesh]")
{
   Mesh mesh = Mesh::MakeCartesian2D(3, 1, Element::QUADRILATERAL, false,
                                     3.0, 1.0);
   mesh.GetElement(0)->SetAttribute(1);
   mesh.GetElement(1)->SetAttribute(2);
   mesh.GetElement(2)->SetAttribute(2);
   EntitySets &sets = mesh.GetOrCreateEntitySets();

   Array<int> elements({2, 0, 2});
   const int element_list = sets.AddElementSet("List", elements);
   const int attribute = sets.AddElementSetByAttribute("Attribute", 2);
   Array<int> marker({1, 0, -1});
   const int marked = sets.AddElementSetByMarker("Marker", marker);

   for (const int set : {element_list, marked})
   {
      REQUIRE(sets.GetNumEntities(EntitySets::ELEMENT, set) == 2);
      REQUIRE(sets(EntitySets::ELEMENT, set).count(0) == 1);
      REQUIRE(sets(EntitySets::ELEMENT, set).count(2) == 1);
   }
   REQUIRE(sets.GetNumEntities(EntitySets::ELEMENT, attribute) == 2);
   REQUIRE(sets(EntitySets::ELEMENT, attribute).count(1) == 1);
   REQUIRE(sets(EntitySets::ELEMENT, attribute).count(2) == 1);
}

TEST_CASE("Vertex sets are pointwise topological selections", "[Mesh]")
{
   Mesh mesh = Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL, false,
                                     2.0, 1.0);
   EntitySets &sets = mesh.GetOrCreateEntitySets();

   Array<int> vertices({0, mesh.GetNV() - 1, 0});
   const int listed = sets.AddVertexSet("Listed vertices", vertices);
   Array<int> marker(mesh.GetNV());
   marker = 0;
   marker[1] = 1;
   marker[mesh.GetNV() - 1] = -1;
   const int marked = sets.AddVertexSetByMarker("Marked vertices", marker);
   const int left_boundary = sets.AddVertexSetByPredicate(
                                 "Left boundary", [](const Vector &x)
   {
      return x[0] < 1.e-12;
   });

   REQUIRE(sets.GetNumEntities(EntitySets::VERTEX, listed) == 2);
   REQUIRE(sets(EntitySets::VERTEX, listed).count(0) == 1);
   REQUIRE(sets(EntitySets::VERTEX, listed).count(mesh.GetNV() - 1) == 1);
   REQUIRE(sets.GetNumEntities(EntitySets::VERTEX, marked) == 2);
   REQUIRE(sets(EntitySets::VERTEX, marked).count(1) == 1);
   REQUIRE(sets(EntitySets::VERTEX, marked).count(mesh.GetNV() - 1) == 1);
   REQUIRE(sets.GetNumEntities(EntitySets::VERTEX, left_boundary) == 2);
}

TEST_CASE("Vertex sets can be created from nearest physical points", "[Mesh]")
{
   Mesh mesh = Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL, false,
                                     2.0, 1.0);
   EntitySets &sets = mesh.GetOrCreateEntitySets();
   DenseMatrix points(mesh.SpaceDimension(), 3);
   points.SetCol(0, mesh.GetVertex(0));
   points.SetCol(1, mesh.GetVertex(mesh.GetNV() - 1));
   points.SetCol(2, mesh.GetVertex(0));

   Array<int> nearest;
   Array<real_t> distances;
   sets.FindNearestVertices(points, nearest, distances);
   REQUIRE(nearest.Size() == 3);
   REQUIRE(nearest[0] == 0);
   REQUIRE(nearest[1] == mesh.GetNV() - 1);
   REQUIRE(nearest[2] == 0);
   for (const real_t distance : distances)
   {
      REQUIRE(distance == MFEM_Approx(0.0));
   }

   const int selected = sets.AddVertexSetByNearestPoints(
                            "Measurement points", points, 1.e-12,
                            &nearest, &distances);
   REQUIRE(sets.GetNumEntities(EntitySets::VERTEX, selected) == 2);
   REQUIRE(sets(EntitySets::VERTEX, selected).count(0) == 1);
   REQUIRE(sets(EntitySets::VERTEX, selected).count(mesh.GetNV() - 1) == 1);
}

TEST_CASE("Edge sets can be created from mesh incidence and geometry", "[Mesh]")
{
   Mesh mesh = Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL, false,
                                     2.0, 1.0);
   EntitySets &sets = mesh.GetOrCreateEntitySets();
   const int elements = sets.AddElementSet("Left element", Array<int>({0}));
   const int all_edges = sets.AddEdgeSetFromElementSet(
                            "Element skeleton", elements,
                            EntitySets::ElementSetEdgeRule::ALL);
   const int interior_edges = sets.AddEdgeSetFromElementSet(
                                 "Element interior", elements,
                                 EntitySets::ElementSetEdgeRule::INTERIOR);
   const int frontier_edges = sets.AddEdgeSetFromElementSet(
                                 "Element frontier", elements,
                                 EntitySets::ElementSetEdgeRule::FRONTIER);

   REQUIRE(sets.GetNumEntities(EntitySets::EDGE, all_edges) == 4);
   REQUIRE(sets.GetNumEntities(EntitySets::EDGE, interior_edges) == 3);
   REQUIRE(sets.GetNumEntities(EntitySets::EDGE, frontier_edges) == 1);

   mesh.SetBdrAttribute(0, 7);
   const int boundary_edges =
      sets.AddEdgeSetByBoundaryAttribute("Boundary attribute", 7);
   REQUIRE(sets.GetNumEntities(EntitySets::EDGE, boundary_edges) == 1);
   REQUIRE(sets(EntitySets::EDGE, boundary_edges).count(
              mesh.GetBdrElementFaceIndex(0)) == 1);

   const int center_edges = sets.AddEdgeSetByPredicate(
                               "Centers", [](const Vector &x)
   {
      return x[0] < 0.75;
   });
   const int vertex_edges = sets.AddEdgeSetByPredicate(
                               "Vertices", [](const Vector &x)
   {
      return x[0] < 1.e-12;
   }, EntitySets::EdgeSetPointRule::ALL_VERTICES);
   REQUIRE(sets.GetNumEntities(EntitySets::EDGE, center_edges) == 3);
   REQUIRE(sets.GetNumEntities(EntitySets::EDGE, vertex_edges) == 1);
}

TEST_CASE("Entity sets follow uniform refinement descendants", "[Mesh]")
{
   Mesh mesh = MakeEntitySetQuadMesh();
   REQUIRE(mesh.ent_sets != nullptr);

   const auto vertex = EntitySets::VERTEX;
   const auto edge = EntitySets::EDGE;
   const auto element = EntitySets::ELEMENT;

   REQUIRE(mesh.ent_sets->GetNumEntities(vertex, 0) == 1);
   REQUIRE(mesh.ent_sets->GetNumEntities(edge, 0) == 1);
   REQUIRE(mesh.ent_sets->GetNumEntities(element, 0) == 1);

   mesh.UniformRefinement();

   const auto &vertices = (*mesh.ent_sets)(vertex, 0);
   REQUIRE(vertices.size() == 1);
   REQUIRE(vertices.count(0) == 1);

   const auto &edges = (*mesh.ent_sets)(edge, 0);
   REQUIRE(edges.size() == 2);
   std::set<int> edge_vertices;
   for (const int e : edges)
   {
      Array<int> v;
      mesh.GetEdgeVertices(e, v);
      edge_vertices.insert(v[0]);
      edge_vertices.insert(v[1]);
   }
   REQUIRE(edge_vertices.count(0) == 1);
   REQUIRE(edge_vertices.count(1) == 1);
   REQUIRE(edge_vertices.size() == 3);

   const auto &elements = (*mesh.ent_sets)(element, 0);
   REQUIRE(elements.size() == 4);
   for (int e = 0; e < 4; e++) { REQUIRE(elements.count(e) == 1); }

   mesh.UniformRefinement();

   REQUIRE((*mesh.ent_sets)(vertex, 0).size() == 1);
   REQUIRE((*mesh.ent_sets)(vertex, 0).count(0) == 1);
   REQUIRE((*mesh.ent_sets)(edge, 0).size() == 4);
   REQUIRE((*mesh.ent_sets)(element, 0).size() == 16);
   for (int e = 0; e < 16; e++)
   {
      REQUIRE((*mesh.ent_sets)(element, 0).count(e) == 1);
   }
}

TEST_CASE("Entity sets follow nonconforming refinement descendants", "[Mesh][NCMesh]")
{
   Mesh mesh = MakeEntitySetQuadMesh();
   mesh.EnsureNCMesh(true);

   Array<Refinement> refinements(1);
   refinements[0].Set(0, Refinement::XY);
   mesh.GeneralRefinement(refinements, 1);

   const auto vertex = EntitySets::VERTEX;
   const auto edge = EntitySets::EDGE;
   const auto element = EntitySets::ELEMENT;

   const auto &vertices = (*mesh.ent_sets)(vertex, 0);
   REQUIRE(vertices.size() == 1);
   REQUIRE(vertices.count(0) == 1);

   const auto &edges = (*mesh.ent_sets)(edge, 0);
   REQUIRE(edges.size() == 2);
   for (const int e : edges)
   {
      Array<int> v;
      mesh.GetEdgeVertices(e, v);
      REQUIRE((v[0] == 0 || v[1] == 0 || v[0] == 1 || v[1] == 1));
   }

   const auto &elements = (*mesh.ent_sets)(element, 0);
   REQUIRE(elements.size() == 4);
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      REQUIRE(elements.count(e) == 1);
   }
}

TEST_CASE("Entity sets follow 3D triangular-face refinement descendants", "[Mesh]")
{
   Mesh mesh = MakeEntitySetTetraMesh();
   mesh.UniformRefinement();

   REQUIRE((*mesh.ent_sets)(EntitySets::VERTEX, 0).size() == 1);
   REQUIRE((*mesh.ent_sets)(EntitySets::VERTEX, 0).count(0) == 1);
   REQUIRE((*mesh.ent_sets)(EntitySets::EDGE, 0).size() == 2);
   REQUIRE((*mesh.ent_sets)(EntitySets::FACE, 0).size() == 4);
   REQUIRE((*mesh.ent_sets)(EntitySets::ELEMENT, 0).size() == 8);
}

TEST_CASE("Entity sets follow 3D quadrilateral-face refinement descendants",
          "[Mesh]")
{
   Mesh mesh = MakeEntitySetHexMesh();
   mesh.UniformRefinement();

   REQUIRE((*mesh.ent_sets)(EntitySets::VERTEX, 0).size() == 1);
   REQUIRE((*mesh.ent_sets)(EntitySets::VERTEX, 0).count(0) == 1);
   REQUIRE((*mesh.ent_sets)(EntitySets::EDGE, 0).size() == 2);
   REQUIRE((*mesh.ent_sets)(EntitySets::FACE, 0).size() == 4);
   REQUIRE((*mesh.ent_sets)(EntitySets::ELEMENT, 0).size() == 8);
}

TEST_CASE("Entity sets follow nonconforming 3D face refinement descendants",
          "[Mesh][NCMesh]")
{
   SECTION("triangular face")
   {
      Mesh mesh = MakeEntitySetTetraMesh();
      mesh.EnsureNCMesh(true);

      Array<Refinement> refinements(1);
      refinements[0].Set(0, Refinement::XYZ);
      mesh.GeneralRefinement(refinements, 1);

      REQUIRE((*mesh.ent_sets)(EntitySets::VERTEX, 0).size() == 1);
      REQUIRE((*mesh.ent_sets)(EntitySets::EDGE, 0).size() == 2);
      REQUIRE((*mesh.ent_sets)(EntitySets::FACE, 0).size() == 4);
      REQUIRE((*mesh.ent_sets)(EntitySets::ELEMENT, 0).size() == 8);
   }

   SECTION("quadrilateral face")
   {
      Mesh mesh = MakeEntitySetHexMesh();
      mesh.EnsureNCMesh(true);

      Array<Refinement> refinements(1);
      refinements[0].Set(0, Refinement::XYZ);
      mesh.GeneralRefinement(refinements, 1);

      REQUIRE((*mesh.ent_sets)(EntitySets::VERTEX, 0).size() == 1);
      REQUIRE((*mesh.ent_sets)(EntitySets::EDGE, 0).size() == 2);
      REQUIRE((*mesh.ent_sets)(EntitySets::FACE, 0).size() == 4);
      REQUIRE((*mesh.ent_sets)(EntitySets::ELEMENT, 0).size() == 8);
   }
}

TEST_CASE("Entity set creation and printing validate mesh topology", "[Mesh]")
{
   Mesh mesh = Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON);
   EntitySets &sets = mesh.GetOrCreateEntitySets();
   REQUIRE(&sets == &mesh.GetOrCreateEntitySets());

   const int vertices = sets.AddSet(EntitySets::VERTEX, "Corners");
   const int edges = sets.AddSet(EntitySets::EDGE, "Bottom");
   const int faces = sets.AddSet(EntitySets::FACE, "Base");
   const int elements = sets.AddSet(EntitySets::ELEMENT, "Domain");

   sets.AddEntity(EntitySets::VERTEX, vertices, 0);
   sets.AddEdge(edges, 0, 1);
   sets.AddQuadFace(faces, 0, 1, 2, 3);
   sets.AddEntity(EntitySets::ELEMENT, elements, 0);

   std::stringstream output;
   sets.Print(output);
   EntitySets restored(mesh);
   restored.Load(output);

   for (const auto type : {EntitySets::VERTEX, EntitySets::EDGE,
                           EntitySets::FACE, EntitySets::ELEMENT})
   {
      REQUIRE(restored.GetNumSets(type) == 1);
      REQUIRE(restored.GetSetName(type, 0) == sets.GetSetName(type, 0));
      REQUIRE(restored(type, 0) == sets(type, 0));
   }
}

#ifdef MFEM_USE_MPI

namespace
{

int GlobalEntitySetSize(const ParMesh &mesh, EntitySets::EntityType type)
{
   const auto *sets = dynamic_cast<const ParEntitySets *>(mesh.ent_sets);
   REQUIRE(sets != nullptr);

   int local_size = static_cast<int>((*sets)(type, 0).size());
   int global_size = 0;
   MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM,
                 mesh.GetComm());
   return global_size;
}

} // namespace

TEST_CASE("Parallel nearest-vertex sets use a global search", "[Parallel][Mesh]")
{
   Mesh serial_mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL,
                                             false, 2.0, 2.0);
   Array<int> partitioning(serial_mesh.GetNE());
   for (int element = 0; element < partitioning.Size(); element++)
   {
      partitioning[element] = element % Mpi::WorldSize();
   }
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh, partitioning.GetData());
   EntitySets &sets = mesh.GetOrCreateEntitySets();
   DenseMatrix points(mesh.SpaceDimension(), 1);
   points = 0.0;

   Array<int> nearest;
   Array<real_t> distances;
   sets.FindNearestVertices(points, nearest, distances);
   REQUIRE(distances[0] == MFEM_Approx(0.0));

   int locally_present = nearest[0] >= 0 ? 1 : 0;
   int globally_present = 0;
   MPI_Allreduce(&locally_present, &globally_present, 1, MPI_INT, MPI_SUM,
                 mesh.GetComm());
   REQUIRE(globally_present > 0);
   if (nearest[0] >= 0)
   {
      const real_t *vertex = mesh.GetVertex(nearest[0]);
      REQUIRE(vertex[0] == MFEM_Approx(0.0));
      REQUIRE(vertex[1] == MFEM_Approx(0.0));
   }

   const int selected = sets.AddVertexSetByNearestPoints(
                            "Global measurement point", points, 1.e-12);
   int local_size = sets.GetNumEntities(EntitySets::VERTEX, selected);
   int global_size = 0;
   MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM,
                 mesh.GetComm());
   REQUIRE(global_size == globally_present);
}

TEST_CASE("Parallel entity sets follow nonconforming refinement descendants",
          "[Parallel][Mesh][NCMesh]")
{
   Mesh serial_mesh = MakeEntitySetQuadMesh();
   serial_mesh.EnsureNCMesh(true);
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);

   REQUIRE(dynamic_cast<ParEntitySets *>(mesh.ent_sets) != nullptr);
   REQUIRE(GlobalEntitySetSize(mesh, EntitySets::VERTEX) == 1);
   REQUIRE(GlobalEntitySetSize(mesh, EntitySets::EDGE) == 1);
   REQUIRE(GlobalEntitySetSize(mesh, EntitySets::ELEMENT) == 1);

   Array<Refinement> refinements;
   if (mesh.GetNE() > 0)
   {
      refinements.Append(Refinement(0, Refinement::XY));
   }
   mesh.GeneralRefinement(refinements, 1);

   REQUIRE(dynamic_cast<ParEntitySets *>(mesh.ent_sets) != nullptr);
   REQUIRE(GlobalEntitySetSize(mesh, EntitySets::VERTEX) == 1);
   REQUIRE(GlobalEntitySetSize(mesh, EntitySets::EDGE) == 2);
   REQUIRE(GlobalEntitySetSize(mesh, EntitySets::ELEMENT) == 4);

   Array<real_t> error(mesh.GetNE());
   error = 0.0;
   REQUIRE(mesh.DerefineByError(error, 1.0));

   REQUIRE(dynamic_cast<ParEntitySets *>(mesh.ent_sets) != nullptr);
   REQUIRE(GlobalEntitySetSize(mesh, EntitySets::VERTEX) == 1);
   REQUIRE(GlobalEntitySetSize(mesh, EntitySets::EDGE) == 1);
   REQUIRE(GlobalEntitySetSize(mesh, EntitySets::ELEMENT) == 1);
}

TEST_CASE("Parallel entity sets retain shared descendants through rebalance",
          "[Parallel][Mesh][NCMesh]")
{
   const int nranks = Mpi::WorldSize();
   Mesh expected = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);
   AddAllEntitySets(expected);
   const int original_vertices = expected.GetNV();
   expected.EnsureNCMesh(true);

   Array<Refinement> expected_refinements(1);
   expected_refinements[0].Set(0, Refinement::XY);
   expected.GeneralRefinement(expected_refinements, 1);

   Mesh serial_mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);
   AddAllEntitySets(serial_mesh);
   serial_mesh.EnsureNCMesh(true);
   Array<int> partitioning(serial_mesh.GetNE());
   for (int i = 0; i < partitioning.Size(); i++) { partitioning[i] = i % nranks; }
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh, partitioning.GetData());

   Array<Refinement> refinements;
   if (Mpi::Root()) { refinements.Append(Refinement(0, Refinement::XY)); }
   mesh.GeneralRefinement(refinements, 1);

   REQUIRE(GlobalEntitySetSize(mesh, EntitySets::VERTEX) == original_vertices);
   REQUIRE(GlobalEntitySetSize(mesh, EntitySets::EDGE) ==
           static_cast<int>(expected.ent_sets->GetNumEntities(EntitySets::EDGE, 0)));
   REQUIRE(GlobalEntitySetSize(mesh, EntitySets::ELEMENT) ==
           static_cast<int>(expected.ent_sets->GetNumEntities(EntitySets::ELEMENT, 0)));

   mesh.Rebalance();

   REQUIRE(dynamic_cast<ParEntitySets *>(mesh.ent_sets) != nullptr);
   REQUIRE(GlobalEntitySetSize(mesh, EntitySets::VERTEX) == original_vertices);
   REQUIRE(GlobalEntitySetSize(mesh, EntitySets::EDGE) ==
           static_cast<int>(expected.ent_sets->GetNumEntities(EntitySets::EDGE, 0)));
   REQUIRE(GlobalEntitySetSize(mesh, EntitySets::ELEMENT) ==
           static_cast<int>(expected.ent_sets->GetNumEntities(EntitySets::ELEMENT, 0)));

   Array<real_t> error(mesh.GetNE());
   error = 0.0;
   REQUIRE(mesh.DerefineByError(error, 1.0));
   REQUIRE(GlobalEntitySetSize(mesh, EntitySets::VERTEX) == original_vertices);
   REQUIRE(GlobalEntitySetSize(mesh, EntitySets::EDGE) == serial_mesh.GetNEdges());
   REQUIRE(GlobalEntitySetSize(mesh, EntitySets::ELEMENT) == serial_mesh.GetNE());
}

#endif
