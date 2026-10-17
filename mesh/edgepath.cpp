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

#include "edgepath.hpp"
#include "entsets.hpp"
#include "mesh.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace mfem
{

namespace
{

struct PathEdge
{
   int vertex;
   int edge;
   int orientation;
   real_t alignment;
};

struct PathSearchState
{
   int vertex;
   int next_candidate;
};

bool ComparePathEdges(const PathEdge &a, const PathEdge &b)
{
   if (a.alignment != b.alignment) { return a.alignment > b.alignment; }
   if (a.vertex != b.vertex) { return a.vertex < b.vertex; }
   return a.edge < b.edge;
}

int GetPathEdgeCount(const Mesh &mesh)
{
   return (mesh.Dimension() == 1) ? mesh.GetNE() : mesh.GetNEdges();
}

void GetPathEdgeVertices(const Mesh &mesh, int edge, Array<int> &vertices)
{
   if (mesh.Dimension() == 1)
   {
      mesh.GetElementVertices(edge, vertices);
   }
   else
   {
      mesh.GetEdgeVertices(edge, vertices);
   }
}

} // namespace

EdgePath::EdgePath(const Mesh &mesh, int start_vertex, int end_vertex)
{
   SetEndpoints(mesh, start_vertex, end_vertex);
}

EdgePath::EdgePath(const Mesh &mesh, int start_vertex, int end_vertex,
                   const EdgeScorer &scorer)
{
   SetEndpoints(mesh, start_vertex, end_vertex, scorer);
}

EdgePath::EdgePath(const Mesh &mesh, const Array<int> &vertices)
{
   SetWaypoints(mesh, vertices);
}

EdgePath::EdgePath(const Mesh &mesh, const Array<int> &vertices,
                   const EdgeScorer &scorer)
{
   SetWaypoints(mesh, vertices, scorer);
}

EdgePath::EdgePath(const Mesh &mesh, const Array<int> &edges,
                   const Array<int> &edge_orientations)
{
   SetEdges(mesh, edges, edge_orientations);
}

void EdgePath::SetEndpoints(const Mesh &mesh, int start_vertex, int end_vertex)
{
   SetEndpointsImpl(mesh, start_vertex, end_vertex, nullptr, EdgeScorer());
}

void EdgePath::SetEndpoints(const Mesh &mesh, int start_vertex, int end_vertex,
                            const EdgeScorer &scorer)
{
   SetEndpointsImpl(mesh, start_vertex, end_vertex, nullptr, scorer);
}

void EdgePath::SetEndpoints(const Mesh &mesh, int start_vertex, int end_vertex,
                            const Array<int> &allowed_edges,
                            const EdgeScorer &scorer)
{
   std::set<int> allowed(allowed_edges.begin(), allowed_edges.end());
   const int num_edges = GetPathEdgeCount(mesh);
   for (const int edge : allowed)
   {
      MFEM_VERIFY(edge >= 0 && edge < num_edges,
                  "An allowed edge index is out of range.");
   }
   SetEndpointsImpl(mesh, start_vertex, end_vertex, &allowed, scorer);
}

void EdgePath::SetEndpointsImpl(const Mesh &mesh, int start_vertex,
                                int end_vertex,
                                const std::set<int> *allowed_edges,
                                const EdgeScorer &scorer)
{
   mesh_ = &mesh;
   const int num_vertices = mesh.GetNV();
   MFEM_VERIFY(start_vertex >= 0 && start_vertex < num_vertices,
               "The start vertex index is out of range.");
   MFEM_VERIFY(end_vertex >= 0 && end_vertex < num_vertices,
               "The end vertex index is out of range.");

   vertices_.SetSize(0);
   edges_.SetSize(0);
   edge_orientations_.SetSize(0);

   if (start_vertex == end_vertex)
   {
      vertices_.Append(start_vertex);
      return;
   }

   const int space_dim = mesh.SpaceDimension();
   const real_t *end = mesh.GetVertex(end_vertex);
   std::vector<std::vector<PathEdge>> adjacency(num_vertices);
   const int num_edges = GetPathEdgeCount(mesh);

   for (int edge = 0; edge < num_edges; edge++)
   {
      if (allowed_edges && allowed_edges->count(edge) == 0) { continue; }
      Array<int> edge_vertices;
      GetPathEdgeVertices(mesh, edge, edge_vertices);
      MFEM_ASSERT(edge_vertices.Size() == 2, "Invalid mesh edge.");

      for (int endpoint = 0; endpoint < 2; endpoint++)
      {
         const int vertex = edge_vertices[endpoint];
         const int next_vertex = edge_vertices[1 - endpoint];
         real_t alignment = 0.0;
         if (scorer)
         {
            alignment = scorer(mesh, edge, vertex, next_vertex, end_vertex);
         }
         else
         {
            const real_t *current = mesh.GetVertex(vertex);
            const real_t *next = mesh.GetVertex(next_vertex);
            real_t edge_length_sq = 0.0;
            real_t target_length_sq = 0.0;
            real_t dot_product = 0.0;
            for (int d = 0; d < space_dim; d++)
            {
               const real_t edge_component = next[d] - current[d];
               const real_t target_component = end[d] - current[d];
               edge_length_sq += edge_component * edge_component;
               target_length_sq += target_component * target_component;
               dot_product += edge_component * target_component;
            }
            const real_t denominator = sqrt(edge_length_sq * target_length_sq);
            alignment = (denominator > 0.0) ? dot_product / denominator : 0.0;
         }
         adjacency[vertex].push_back({next_vertex, edge,
                                      endpoint == 0 ? 1 : -1, alignment});
      }
   }

   for (std::vector<PathEdge> &candidates : adjacency)
   {
      std::sort(candidates.begin(), candidates.end(), ComparePathEdges);
   }

   std::vector<char> visited(num_vertices, false);
   std::vector<int> parent_vertex(num_vertices, -1);
   std::vector<int> parent_edge(num_vertices, -1);
   std::vector<int> parent_orientation(num_vertices, 0);
   std::vector<PathSearchState> worklist;
   worklist.push_back({start_vertex, 0});
   visited[start_vertex] = true;

   while (!worklist.empty() && !visited[end_vertex])
   {
      PathSearchState &state = worklist.back();
      const std::vector<PathEdge> &candidates = adjacency[state.vertex];
      bool advanced = false;
      while (state.next_candidate < int(candidates.size()))
      {
         const PathEdge &candidate = candidates[state.next_candidate++];
         if (!visited[candidate.vertex])
         {
            visited[candidate.vertex] = true;
            parent_vertex[candidate.vertex] = state.vertex;
            parent_edge[candidate.vertex] = candidate.edge;
            parent_orientation[candidate.vertex] = candidate.orientation;
            worklist.push_back({candidate.vertex, 0});
            advanced = true;
            break;
         }
      }
      if (!advanced) { worklist.pop_back(); }
   }

   MFEM_VERIFY(visited[end_vertex],
               "The endpoint vertices are not connected by mesh edges.");

   std::vector<int> reverse_vertices;
   std::vector<int> reverse_edges;
   std::vector<int> reverse_orientations;
   for (int vertex = end_vertex; vertex != start_vertex;
        vertex = parent_vertex[vertex])
   {
      reverse_vertices.push_back(vertex);
      reverse_edges.push_back(parent_edge[vertex]);
      reverse_orientations.push_back(parent_orientation[vertex]);
   }
   reverse_vertices.push_back(start_vertex);

   vertices_.SetSize(reverse_vertices.size());
   edges_.SetSize(reverse_edges.size());
   edge_orientations_.SetSize(reverse_orientations.size());
   for (int i = 0; i < vertices_.Size(); i++)
   {
      vertices_[i] = reverse_vertices[vertices_.Size() - 1 - i];
   }
   for (int i = 0; i < edges_.Size(); i++)
   {
      edges_[i] = reverse_edges[edges_.Size() - 1 - i];
      edge_orientations_[i] = reverse_orientations[edges_.Size() - 1 - i];
   }
}

void EdgePath::SetEndpointsInElementSet(const Mesh &mesh, int start_vertex,
                                        int end_vertex, const EntitySets &sets,
                                        int element_set,
                                        const EdgeScorer &scorer)
{
   MFEM_VERIFY(sets.GetMesh() == &mesh,
               "The entity set belongs to a different mesh.");
   std::set<int> allowed_edges;
   for (const int element : sets(EntitySets::ELEMENT, element_set))
   {
      Array<int> element_edges, orientations;
      mesh.GetElementEdges(element, element_edges, orientations);
      allowed_edges.insert(element_edges.begin(), element_edges.end());
   }
   SetEndpointsImpl(mesh, start_vertex, end_vertex, &allowed_edges, scorer);
}

void EdgePath::SetEndpointsOnBoundary(const Mesh &mesh, int start_vertex,
                                      int end_vertex,
                                      const Array<int> &boundary_elements,
                                      const EdgeScorer &scorer)
{
   MFEM_VERIFY(mesh.Dimension() > 1,
               "Boundary edge paths require a mesh of dimension at least two.");
   std::set<int> allowed_edges;
   for (const int boundary_element : boundary_elements)
   {
      MFEM_VERIFY(boundary_element >= 0 && boundary_element < mesh.GetNBE(),
                  "A boundary element index is out of range.");
      Array<int> boundary_edges, orientations;
      mesh.GetBdrElementEdges(boundary_element, boundary_edges, orientations);
      allowed_edges.insert(boundary_edges.begin(), boundary_edges.end());
   }
   SetEndpointsImpl(mesh, start_vertex, end_vertex, &allowed_edges, scorer);
}

void EdgePath::SetWaypoints(const Mesh &mesh, const Array<int> &vertices)
{
   SetWaypoints(mesh, vertices, EdgeScorer());
}

void EdgePath::SetWaypoints(const Mesh &mesh, const Array<int> &vertices,
                            const EdgeScorer &scorer)
{
   mesh_ = &mesh;
   MFEM_VERIFY(vertices.Size() > 0, "An edge path requires at least one vertex.");
   const Array<int> waypoints(vertices);

   vertices_.SetSize(0);
   edges_.SetSize(0);
   edge_orientations_.SetSize(0);

   for (int waypoint = 0; waypoint + 1 < waypoints.Size(); waypoint++)
   {
      EdgePath segment(mesh, waypoints[waypoint], waypoints[waypoint + 1],
                       scorer);
      const int first_vertex = (waypoint == 0) ? 0 : 1;
      for (int i = first_vertex; i < segment.vertices_.Size(); i++)
      {
         vertices_.Append(segment.vertices_[i]);
      }
      edges_.Append(segment.edges_);
      edge_orientations_.Append(segment.edge_orientations_);
   }

   if (waypoints.Size() == 1) { vertices_.Append(waypoints[0]); }
}

void EdgePath::SetEdges(const Mesh &mesh, const Array<int> &edges,
                        const Array<int> &edge_orientations)
{
   mesh_ = &mesh;
   MFEM_VERIFY(edges.Size() == edge_orientations.Size(),
               "The number of edges and orientations must match.");

   vertices_.SetSize(0);
   edges_.SetSize(0);
   edge_orientations_.SetSize(0);
   const int num_edges = GetPathEdgeCount(mesh);

   for (int i = 0; i < edges.Size(); i++)
   {
      const int edge = edges[i];
      const int orientation = edge_orientations[i];
      MFEM_VERIFY(edge >= 0 && edge < num_edges,
                  "An edge index is out of range.");
      MFEM_VERIFY(orientation == 1 || orientation == -1,
                  "Edge orientations must be +1 or -1.");

      Array<int> edge_vertices;
      GetPathEdgeVertices(mesh, edge, edge_vertices);
      const int first = edge_vertices[(orientation > 0) ? 0 : 1];
      const int second = edge_vertices[(orientation > 0) ? 1 : 0];
      if (i == 0)
      {
         vertices_.Append(first);
      }
      else
      {
         MFEM_VERIFY(vertices_.Last() == first,
                     "The ordered edges do not form a continuous path.");
      }
      vertices_.Append(second);
      edges_.Append(edge);
      edge_orientations_.Append(orientation);
   }
}

void EdgePath::SetFromEdgeSet(const EntitySets &sets, int edge_set,
                              int start_vertex, int next_vertex)
{
   const Mesh &mesh = *sets.GetMesh();
   mesh_ = &mesh;
   const std::set<int> &selected_edges = sets(EntitySets::EDGE, edge_set);
   MFEM_VERIFY(!selected_edges.empty(), "The edge set is empty.");

   std::vector<std::vector<int>> incident_edges(mesh.GetNV());
   for (const int edge : selected_edges)
   {
      Array<int> edge_vertices;
      GetPathEdgeVertices(mesh, edge, edge_vertices);
      incident_edges[edge_vertices[0]].push_back(edge);
      incident_edges[edge_vertices[1]].push_back(edge);
   }

   Array<int> endpoints;
   for (int vertex = 0; vertex < mesh.GetNV(); vertex++)
   {
      const int degree = incident_edges[vertex].size();
      MFEM_VERIFY(degree <= 2,
                  "The edge set branches and cannot define a single path.");
      if (degree == 1) { endpoints.Append(vertex); }
   }
   MFEM_VERIFY(endpoints.Size() == 0 || endpoints.Size() == 2,
               "The edge set is disconnected and cannot define a single path.");

   if (endpoints.Size() == 2)
   {
      if (start_vertex < 0) { start_vertex = std::min(endpoints[0], endpoints[1]); }
      MFEM_VERIFY(start_vertex == endpoints[0] || start_vertex == endpoints[1],
                  "An open edge set path must start at an endpoint.");
   }
   else
   {
      MFEM_VERIFY(start_vertex >= 0 && start_vertex < mesh.GetNV() &&
                  incident_edges[start_vertex].size() == 2,
                  "A closed edge set path requires a vertex on the loop.");
   }

   vertices_.SetSize(0);
   edges_.SetSize(0);
   edge_orientations_.SetSize(0);
   std::set<int> used_edges;
   int current_vertex = start_vertex;
   int previous_edge = -1;
   vertices_.Append(current_vertex);

   while (used_edges.size() < selected_edges.size())
   {
      const std::vector<int> &candidates = incident_edges[current_vertex];
      int edge = -1;
      if (previous_edge < 0 && next_vertex >= 0)
      {
         for (const int candidate : candidates)
         {
            Array<int> edge_vertices;
            GetPathEdgeVertices(mesh, candidate, edge_vertices);
            const int candidate_next = (edge_vertices[0] == current_vertex) ?
                                       edge_vertices[1] : edge_vertices[0];
            if (candidate_next == next_vertex)
            {
               edge = candidate;
               break;
            }
         }
         MFEM_VERIFY(edge >= 0,
                     "The requested next vertex is not connected to the start.");
      }
      else
      {
         for (const int candidate : candidates)
         {
            if (candidate != previous_edge && used_edges.count(candidate) == 0 &&
                (edge < 0 || candidate < edge))
            {
               edge = candidate;
            }
         }
      }
      MFEM_VERIFY(edge >= 0,
                  "The edge set is disconnected and cannot define a single path.");

      Array<int> edge_vertices;
      GetPathEdgeVertices(mesh, edge, edge_vertices);
      const int orientation = (edge_vertices[0] == current_vertex) ? 1 : -1;
      const int next = edge_vertices[(orientation > 0) ? 1 : 0];
      used_edges.insert(edge);
      edges_.Append(edge);
      edge_orientations_.Append(orientation);
      vertices_.Append(next);
      previous_edge = edge;
      current_vertex = next;
   }

   MFEM_VERIFY(endpoints.Size() == 0 ? current_vertex == start_vertex :
               current_vertex != start_vertex,
               "The edge set is disconnected and cannot define a single path.");
}

} // namespace mfem
