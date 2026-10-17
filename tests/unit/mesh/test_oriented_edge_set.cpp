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

using namespace mfem;

namespace
{

Mesh MakeBranchedPathMesh()
{
   Mesh mesh(1, 5, 4, 0, 2);
   mesh.AddVertex(0.0, 0.0);
   mesh.AddVertex(1.0, 0.0);
   mesh.AddVertex(0.0, 1.0);
   mesh.AddVertex(1.0, 1.0);
   mesh.AddVertex(2.0, 1.0);
   mesh.AddSegment(0, 1);
   mesh.AddSegment(0, 2);
   mesh.AddSegment(2, 3);
   mesh.AddSegment(3, 4);
   mesh.FinalizeMesh();
   return mesh;
}

void RequireEqual(const Array<int> &actual, const Array<int> &expected)
{
   REQUIRE(actual.Size() == expected.Size());
   for (int i = 0; i < actual.Size(); i++) { REQUIRE(actual[i] == expected[i]); }
}

void RequirePathConnectivity(const Mesh &mesh, const OrientedEdgeSet &path)
{
   REQUIRE(path.GetVertices().Size() == path.GetEdges().Size() + 1);
   int current_vertex = path.GetVertices()[0];
   for (int i = 0; i < path.GetEdges().Size(); i++)
   {
      Array<int> edge_vertices;
      if (mesh.Dimension() == 1)
      {
         mesh.GetElementVertices(path.GetEdges()[i], edge_vertices);
      }
      else
      {
         mesh.GetEdgeVertices(path.GetEdges()[i], edge_vertices);
      }
      const int next_vertex = path.GetVertices()[i + 1];
      const int orientation = (edge_vertices[0] == current_vertex) ? 1 : -1;
      REQUIRE((orientation > 0 ? edge_vertices[1] : edge_vertices[0]) ==
              next_vertex);
      REQUIRE(path.GetEdgeOrientations()[i] == orientation);
      current_vertex = next_vertex;
   }
}

} // namespace

TEST_CASE("Oriented edge sets provide path construction", "[Mesh]")
{
   Mesh mesh = MakeBranchedPathMesh();
   OrientedEdgeSet path(mesh, 0, 4);
   OrientedEdgeSet repeat_path(mesh, 0, 4);

   // The edge toward vertex 1 is more aligned with the target but is a dead
   // end. The path search therefore backtracks and takes the upper branch.
   RequireEqual(path.GetVertices(), Array<int>({0, 2, 3, 4}));
   RequireEqual(path.GetVertices(), repeat_path.GetVertices());
   RequireEqual(path.GetEdges(), repeat_path.GetEdges());
   RequireEqual(path.GetEdgeOrientations(), repeat_path.GetEdgeOrientations());

   RequirePathConnectivity(mesh, path);

   OrientedEdgeSet zero_length_path(mesh, 2, 2);
   RequireEqual(zero_length_path.GetVertices(), Array<int>({2}));
   REQUIRE(zero_length_path.GetEdges().Size() == 0);
   REQUIRE(zero_length_path.GetEdgeOrientations().Size() == 0);

   OrientedEdgeSet closed_path(mesh, Array<int>({0, 3, 0}));
   RequireEqual(closed_path.GetVertices(), Array<int>({0, 2, 3, 2, 0}));
   REQUIRE(closed_path.GetVertices()[0] == closed_path.GetVertices().Last());
   RequirePathConnectivity(mesh, closed_path);

   Mesh quad_mesh = Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL,
                                          false, 2.0, 1.0);
   int start_vertex = -1;
   int end_vertex = -1;
   for (int vertex = 0; vertex < quad_mesh.GetNV(); vertex++)
   {
      const real_t *x = quad_mesh.GetVertex(vertex);
      if (x[0] == 0.0 && x[1] == 0.0) { start_vertex = vertex; }
      if (x[0] == 2.0 && x[1] == 1.0) { end_vertex = vertex; }
   }
   REQUIRE(start_vertex >= 0);
   REQUIRE(end_vertex >= 0);
   OrientedEdgeSet quad_path(quad_mesh, start_vertex, end_vertex);
   REQUIRE(quad_path.GetVertices()[0] == start_vertex);
   REQUIRE(quad_path.GetVertices().Last() == end_vertex);
   RequirePathConnectivity(quad_mesh, quad_path);

   OrientedEdgeSet explicit_set(quad_mesh, quad_path.GetEdges(),
                                quad_path.GetEdgeOrientations());
   RequireEqual(explicit_set.GetEdges(), quad_path.GetEdges());
   RequireEqual(explicit_set.GetEdgeOrientations(), quad_path.GetEdgeOrientations());
   REQUIRE(explicit_set.GetVertices().Size() == 0);

   OrientedEdgeSet disconnected_set(quad_mesh, Array<int>({0, 1}),
                                    Array<int>({1, -1}));
   REQUIRE(disconnected_set.Size() == 2);
   REQUIRE(disconnected_set.GetVertices().Size() == 0);

   OrientedEdgeSet constrained_path;
   constrained_path.SetEndpoints(quad_mesh, start_vertex, end_vertex,
                                 quad_path.GetEdges());
   RequireEqual(constrained_path.GetEdges(), quad_path.GetEdges());
   RequirePathConnectivity(quad_mesh, constrained_path);

   EntitySets &sets = quad_mesh.GetOrCreateEntitySets();
   const int edge_set = sets.AddSet(EntitySets::EDGE, "Path edges");
   for (const int edge : quad_path.GetEdges())
   {
      sets.AddEntity(EntitySets::EDGE, edge_set, edge);
   }
   OrientedEdgeSet entity_set_path;
   entity_set_path.SetFromEdgeSet(sets, edge_set, start_vertex);
   RequireEqual(entity_set_path.GetEdges(), quad_path.GetEdges());
   RequirePathConnectivity(quad_mesh, entity_set_path);

   const int element_set = sets.AddElementSet("All elements",
                                              Array<int>({0, 1}));
   OrientedEdgeSet element_set_path;
   element_set_path.SetEndpointsInElementSet(quad_mesh, start_vertex,
                                             end_vertex, sets, element_set);
   RequirePathConnectivity(quad_mesh, element_set_path);

   Array<int> boundary_elements(quad_mesh.GetNBE());
   for (int i = 0; i < boundary_elements.Size(); i++)
   {
      boundary_elements[i] = i;
   }
   OrientedEdgeSet boundary_path;
   boundary_path.SetEndpointsOnBoundary(quad_mesh, start_vertex, end_vertex,
                                        boundary_elements);
   RequirePathConnectivity(quad_mesh, boundary_path);

   const int boundary_edge_set = sets.AddSet(EntitySets::EDGE, "Boundary");
   for (const int boundary_element : boundary_elements)
   {
      sets.AddEntity(EntitySets::EDGE, boundary_edge_set,
                     quad_mesh.GetBdrElementFaceIndex(boundary_element));
   }
   OrientedEdgeSet boundary_loop;
   boundary_loop.SetFromEdgeSet(sets, boundary_edge_set, start_vertex);
   REQUIRE(boundary_loop.GetVertices()[0] == start_vertex);
   REQUIRE(boundary_loop.GetVertices().Last() == start_vertex);
   RequirePathConnectivity(quad_mesh, boundary_loop);

   int score_calls = 0;
   OrientedEdgeSet scored_path(quad_mesh, start_vertex, end_vertex,
                               [&score_calls](const Mesh &, int edge, int, int, int)
   {
      score_calls++;
      return -edge;
   });
   REQUIRE(score_calls > 0);
   RequirePathConnectivity(quad_mesh, scored_path);
}
