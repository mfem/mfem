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

#ifndef MFEM_EDGE_PATH
#define MFEM_EDGE_PATH

#include "../general/array.hpp"

#include <functional>
#include <set>

namespace mfem
{

class Mesh;
class EntitySets;

/** @brief An ordered, directed path along mesh edges.

    Edge orientations are relative to MFEM's global mesh-edge orientation. A
    value of +1 traverses an edge from its first vertex to its second vertex;
    a value of -1 traverses it in the opposite direction. */
class EdgePath
{
public:
   /** @brief Assign a score to a candidate directed edge.

       Larger scores are explored first. The arguments are the mesh, edge
       index, current vertex, candidate next vertex, and final target vertex. */
   using EdgeScorer = std::function<real_t(const Mesh &, int, int, int, int)>;

   EdgePath() = default;

   /** @brief Construct a path between two mesh vertices.

       Candidate edges are explored in decreasing order of alignment with the
       vector from the current vertex to @a end_vertex. The search backtracks
       as necessary, so it finds a path whenever the vertices are connected by
       mesh edges. */
   EdgePath(const Mesh &mesh, int start_vertex, int end_vertex);

   /// Construct a path using a caller-supplied candidate-edge scorer.
   EdgePath(const Mesh &mesh, int start_vertex, int end_vertex,
            const EdgeScorer &scorer);

   /** @brief Construct a path that visits the given vertices in order.

       Consecutive vertices are connected using the same directional search as
       SetEndpoints(). Repeating the first vertex as the last entry defines a
       closed path. */
   EdgePath(const Mesh &mesh, const Array<int> &vertices);

   /// Construct a path through ordered waypoints using a custom scorer.
   EdgePath(const Mesh &mesh, const Array<int> &vertices,
            const EdgeScorer &scorer);

   /// Construct a path from explicitly ordered edges and orientations.
   EdgePath(const Mesh &mesh, const Array<int> &edges,
            const Array<int> &edge_orientations);

   /// Reset this path using the mesh-edge graph and the given endpoint vertices.
   void SetEndpoints(const Mesh &mesh, int start_vertex, int end_vertex);

   /// Reset this path using a caller-supplied candidate-edge scorer.
   void SetEndpoints(const Mesh &mesh, int start_vertex, int end_vertex,
                     const EdgeScorer &scorer);

   /// Reset this path while restricting the search to @a allowed_edges.
   void SetEndpoints(const Mesh &mesh, int start_vertex, int end_vertex,
                     const Array<int> &allowed_edges,
                     const EdgeScorer &scorer = EdgeScorer());

   /** @brief Reset this path while restricting the search to an element set.

       All edges of the selected elements are allowed. */
   void SetEndpointsInElementSet(const Mesh &mesh, int start_vertex,
                                 int end_vertex, const EntitySets &sets,
                                 int element_set,
                                 const EdgeScorer &scorer = EdgeScorer());

   /** @brief Reset this path while restricting the search to boundary edges.

       @a boundary_elements contains local boundary-element indices. */
   void SetEndpointsOnBoundary(const Mesh &mesh, int start_vertex,
                               int end_vertex,
                               const Array<int> &boundary_elements,
                               const EdgeScorer &scorer = EdgeScorer());

   /// Reset this path so that it visits the given vertices in order.
   void SetWaypoints(const Mesh &mesh, const Array<int> &vertices);

   /// Reset this path through ordered waypoints using a custom scorer.
   void SetWaypoints(const Mesh &mesh, const Array<int> &vertices,
                     const EdgeScorer &scorer);

   /// Reset this path from explicitly ordered edges and orientations.
   void SetEdges(const Mesh &mesh, const Array<int> &edges,
                 const Array<int> &edge_orientations);

   /** @brief Reconstruct a simple chain or loop from an edge entity set.

       For a loop, @a start_vertex is required. @a next_vertex optionally
       selects the initial direction; otherwise a deterministic direction is
       selected. */
   void SetFromEdgeSet(const EntitySets &sets, int edge_set,
                       int start_vertex = -1, int next_vertex = -1);

   /// Ordered vertex indices, including the two endpoint vertices.
   const Array<int> &GetVertices() const { return vertices_; }

   /// Ordered mesh-edge indices traversed by this path.
   const Array<int> &GetEdges() const { return edges_; }

   /// Orientations of GetEdges() relative to the global mesh-edge orientation.
   const Array<int> &GetEdgeOrientations() const { return edge_orientations_; }

private:
   void SetEndpointsImpl(const Mesh &mesh, int start_vertex, int end_vertex,
                         const std::set<int> *allowed_edges,
                         const EdgeScorer &scorer);

   Array<int> vertices_;
   Array<int> edges_;
   Array<int> edge_orientations_;
};

} // namespace mfem

#endif // MFEM_EDGE_PATH
