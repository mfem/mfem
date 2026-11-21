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

#ifndef MFEM_TMOP_SIZE_HELPER_HPP
#define MFEM_TMOP_SIZE_HELPER_HPP

#include "mfem.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace tmop_size
{
#ifdef MFEM_USE_MPI
// The mesh communicators number shared entities consecutively by group.
inline void ReduceShared(const mfem::GroupCommunicator &comm,
                         const mfem::Array<int> &entities,
                         mfem::Array<int> &values,
                         void (*op)(mfem::GroupCommunicator::OpData<int>))
{
   mfem::Array<int> shared(entities.Size());
   for (int i = 0; i < entities.Size(); i++) { shared[i] = values[entities[i]]; }
   comm.Reduce(shared, op);
   comm.Bcast(shared);
   for (int i = 0; i < entities.Size(); i++) { values[entities[i]] = shared[i]; }
}

// Mark each labeled topological face on exactly one rank, including interior
// faces recorded on either or both sides. Duplicate boundary records must not
// turn an exposed outline into an apparent patch-interior edge.
inline void GetOwnedBoundaryFaces(mfem::ParMesh &mesh,
                                  mfem::Array<int> &face_present)
{
   using namespace mfem;
   const int dim = mesh.Dimension();
   face_present.SetSize(mesh.GetNumFaces());
   face_present = 0;
   for (int b = 0; b < mesh.GetNBE(); b++)
   {
      face_present[mesh.GetBdrElementFaceIndex(b)] = 1;
   }
   // In 2D, faces are edges. In 3D, triangles and quadrilaterals have
   // separate communicators, each numbered consecutively by group.
   for (int type = 0; type < (dim == 3 ? 2 : 1); type++)
   {
      GroupCommunicator comm(mesh.gtopo);
      if (dim == 2) { mesh.GetSharedEdgeCommunicator(comm); }
      else if (type == 0) { mesh.GetSharedTriCommunicator(comm); }
      else { mesh.GetSharedQuadCommunicator(comm); }
      Array<int> shared_faces, owned;
      for (int g = 1; g < mesh.GetNGroups(); g++)
      {
         const int count = dim == 2 ? mesh.GroupNEdges(g) :
                           (type == 0 ? mesh.GroupNTriangles(g) :
                            mesh.GroupNQuadrilaterals(g));
         for (int i = 0; i < count; i++)
         {
            const int face = dim == 2 ? mesh.GroupEdge(g, i) :
                             (type == 0 ? mesh.GroupTriangle(g, i) :
                              mesh.GroupQuadrilateral(g, i));
            shared_faces.Append(face);
            owned.Append(mesh.gtopo.IAmMaster(g));
         }
      }
      ReduceShared(comm, shared_faces, face_present,
                   GroupCommunicator::Max<int>);
      for (int i = 0; i < shared_faces.Size(); i++)
      {
         if (!owned[i]) { face_present[shared_faces[i]] = 0; }
      }
   }
}

// Call collectively, with the original physical boundary attributes, after
// creating the mesh's conforming, continuous H1 nodal space. By default, fix
// all coordinate DOFs where different boundary labels meet. Also fix exposed
// outlines of open labeled surfaces, even when allow_seam_sliding is true. If
// allow_seam_sliding is true, straight Cartesian seams in 3D can slide along
// their axis; other seams, endpoints, corners and junctions remain fixed.
// In 2D, both modes fix label junctions and open-curve endpoints.
// Use the same mode on every MPI rank.
// This supplements, rather than replaces, the surface-normal constraints.
inline void GetEssentialTrueDofs(mfem::ParGridFunction &nodes,
                                 mfem::Array<int> &essential,
                                 bool allow_seam_sliding = false)
{
   using namespace mfem;
   ParFiniteElementSpace &fes = *nodes.ParFESpace();
   ParMesh &mesh = *fes.GetParMesh();
   const int dim = mesh.Dimension();
   MFEM_VERIFY(mesh.Conforming() && fes.GetVDim() == dim &&
               dynamic_cast<const H1_FECollection *>(fes.FEColl()),
               "Surface-boundary constraints require conforming H1 mesh nodes.");
   const real_t *x = allow_seam_sliding ? nodes.HostRead() : nullptr;
   const int unset = std::numeric_limits<int>::max();
   Array<int> vertex_min(mesh.GetNV()), vertex_max(mesh.GetNV());
   Array<int> edge_min(mesh.GetNEdges()), edge_max(mesh.GetNEdges());
   Array<int> vertex_faces(dim == 2 ? mesh.GetNV() : 0);
   Array<int> edge_faces(dim == 3 ? mesh.GetNEdges() : 0);
   vertex_min = unset; vertex_max = 0;
   edge_min = unset; edge_max = 0;
   vertex_faces = 0; edge_faces = 0;
   Array<int> vertices, edges, orientations, dofs;
   for (int b = 0; b < mesh.GetNBE(); b++)
   {
      const int attr = mesh.GetBdrAttribute(b);
      mesh.GetBdrElementVertices(b, vertices);
      for (int v : vertices)
      {
         vertex_min[v] = std::min(vertex_min[v], attr);
         vertex_max[v] = std::max(vertex_max[v], attr);
      }
      if (dim == 3)
      {
         mesh.GetBdrElementEdges(b, edges, orientations);
         for (int e : edges)
         {
            edge_min[e] = std::min(edge_min[e], attr);
            edge_max[e] = std::max(edge_max[e], attr);
         }
      }
   }

   Array<int> owned_faces;
   GetOwnedBoundaryFaces(mesh, owned_faces);
   for (int f = 0; f < owned_faces.Size(); f++)
   {
      if (!owned_faces[f]) { continue; }
      if (dim == 2)
      {
         mesh.GetFaceVertices(f, vertices);
         for (int v : vertices) { vertex_faces[v]++; }
      }
      else
      {
         mesh.GetFaceEdges(f, edges, orientations);
         for (int e : edges) { edge_faces[e]++; }
      }
   }

   GroupCommunicator vertex_comm(mesh.gtopo);
   mesh.GetSharedVertexCommunicator(vertex_comm);
   Array<int> shared_vertices;
   for (int g = 1; g < mesh.GetNGroups(); g++)
   {
      for (int i = 0; i < mesh.GroupNVertices(g); i++)
      { shared_vertices.Append(mesh.GroupVertex(g, i)); }
   }
   ReduceShared(vertex_comm, shared_vertices, vertex_min,
                GroupCommunicator::Min<int>);
   ReduceShared(vertex_comm, shared_vertices, vertex_max,
                GroupCommunicator::Max<int>);
   if (dim == 2)
   {
      ReduceShared(vertex_comm, shared_vertices, vertex_faces,
                   GroupCommunicator::Sum<int>);
   }

   Array<int> marker(fes.GetVSize());
   Array<int> degree(allow_seam_sliding ? mesh.GetNV() : 0);
   Array<int> junction(allow_seam_sliding ? mesh.GetNV() : 0);
   marker = 0; degree = 0; junction = 0;
   auto mark = [&](int dof, int component)
   {
      dof = dof >= 0 ? dof : -1-dof;
      marker[fes.DofToVDof(dof, component)] = 1;
   };
   if (dim == 3)
   {
      GroupCommunicator edge_comm(mesh.gtopo);
      mesh.GetSharedEdgeCommunicator(edge_comm);
      Array<int> shared_edges;
      Array<int> edge_owner(allow_seam_sliding ? mesh.GetNEdges() : 0);
      edge_owner = 1;
      for (int g = 1; g < mesh.GetNGroups(); g++)
      {
         for (int i = 0; i < mesh.GroupNEdges(g); i++)
         {
            const int e = mesh.GroupEdge(g, i);
            shared_edges.Append(e);
            if (allow_seam_sliding)
            { edge_owner[e] = mesh.gtopo.IAmMaster(g); }
         }
      }
      ReduceShared(edge_comm, shared_edges, edge_min,
                   GroupCommunicator::Min<int>);
      ReduceShared(edge_comm, shared_edges, edge_max,
                   GroupCommunicator::Max<int>);
      // Each unique labeled face contributes on one rank. Count all incident
      // faces before classifying an exposed edge: a rank-local count of one
      // can be an interior patch edge whose other face is on another rank.
      ReduceShared(edge_comm, shared_edges, edge_faces,
                   GroupCommunicator::Sum<int>);
      for (int e = 0; e < mesh.GetNEdges(); e++)
      {
         const bool open_outline = edge_faces[e] == 1;
         if (!open_outline && edge_min[e] >= edge_max[e]) { continue; }
         fes.GetEdgeDofs(e, dofs);
         int axis = -1;
         // An exposed outline stays fully fixed, including its vertices and
         // high-order edge nodes. Only seams between labels may slide.
         if (allow_seam_sliding && !open_outline)
         {
            mesh.GetEdgeVertices(e, vertices);
            if (edge_owner[e])
            {
               for (int v : vertices) { degree[v]++; }
            }
            int varying = 0;
            for (int d = 0; d < dim; d++)
            {
               real_t lo = std::numeric_limits<real_t>::infinity(), hi = -lo;
               for (int dof : dofs)
               {
                  dof = dof >= 0 ? dof : -1-dof;
                  const real_t value = x[fes.DofToVDof(dof, d)];
                  lo = std::min(lo, value); hi = std::max(hi, value);
               }
               const real_t tol = 8 * std::numeric_limits<real_t>::epsilon() *
                                  std::max({real_t(1), std::abs(lo), std::abs(hi)});
               if (hi-lo > tol) { axis = d; varying++; }
            }
            if (varying != 1) { axis = -1; }
         }
         for (int dof : dofs)
         {
            for (int d = 0; d < dim; d++)
            { if (d != axis) { mark(dof, d); } }
         }
      }
      if (allow_seam_sliding)
      {
         ReduceShared(vertex_comm, shared_vertices, degree,
                      GroupCommunicator::Sum<int>);

         // Detect a third physical label after sharing the extrema. This also
         // fixes changes in the label pair along an otherwise straight seam.
         for (int b = 0; b < mesh.GetNBE(); b++)
         {
            const int attr = mesh.GetBdrAttribute(b);
            mesh.GetBdrElementVertices(b, vertices);
            for (int v : vertices)
            {
               if (attr != vertex_min[v] && attr != vertex_max[v])
               { junction[v] = 1; }
            }
         }
         ReduceShared(vertex_comm, shared_vertices, junction,
                      GroupCommunicator::Max<int>);
      }
   }
   for (int v = 0; v < mesh.GetNV(); v++)
   {
      const bool open_endpoint = dim == 2 && vertex_faces[v] == 1;
      if (!open_endpoint && vertex_min[v] >= vertex_max[v]) { continue; }
      if (!allow_seam_sliding || dim == 2 || degree[v] != 2 || junction[v])
      {
         fes.GetVertexDofs(v, dofs);
         for (int dof : dofs)
         {
            for (int d = 0; d < dim; d++) { mark(dof, d); }
         }
      }
   }
   // Combine constraints across ranks. In sliding mode, a corner receives
   // transverse constraints from both incident seam axes, including when
   // those edges or their boundary faces are on other ranks.
   fes.Synchronize(marker);
   essential.SetSize(0);
   for (int i = 0; i < marker.Size(); i++)
   {
      if (!marker[i]) { continue; }
      const int tdof = fes.GetLocalTDofNumber(i);
      if (tdof >= 0) { essential.Append(tdof); }
   }
}
#endif // MFEM_USE_MPI

} // namespace tmop_size

#endif // MFEM_TMOP_SIZE_HELPER_HPP
