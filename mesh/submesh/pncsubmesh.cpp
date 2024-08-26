// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "pncsubmesh.hpp"

#include <numeric>
#include <unordered_map>
#include "submesh_utils.hpp"
#include "psubmesh.hpp"
#include "../ncmesh_tables.hpp"

namespace mfem
{

using namespace SubMeshUtils;


ParNCSubMesh::ParNCSubMesh(ParSubMesh& submesh,
   const ParNCMesh &parent, From from, const Array<int> &attributes)
: ParNCMesh(), parent_(&parent), from_(from), attributes_(attributes)
{
   MyComm = submesh.GetComm();
   NRanks = submesh.GetNRanks();
   MyRank = submesh.GetMyRank();

   Dim = submesh.Dimension();
   spaceDim = submesh.SpaceDimension();
   Iso = true;
   Legacy = false;

   // Loop over parent leaf elements and add nodes for all vertices. Register as top level
   // nodes, will reparent when looping over edges. Cannot add edge nodes at same time
   // because top level vertex nodes must be contiguous and first in node list (see
   // coordinates).
   if (from == From::Domain)
   {
      UniqueIndexGenerator node_ids;
      // Loop over elements of the parent NCMesh. If the element has the attribute, copy it.
      // parent_to_submesh_element_ids_.SetSize(parent.elements.Size());
      // parent_to_submesh_element_ids_ = -1;
      parent_to_submesh_element_ids_.reserve(parent.elements.Size());

      std::set<int> new_nodes;
      for (int ipe = 0; ipe < parent.elements.Size(); ipe++)
      {
         const auto& pe = parent.elements[ipe];
         if (!HasAttribute(pe, attributes)) { continue; }

         const int elem_id = AddElement(pe);
         NCMesh::Element &el = elements[elem_id];
         parent_element_ids_.Append(ipe); // submesh -> parent
         parent_to_submesh_element_ids_[ipe] = elem_id; // parent -> submesh
         el.index = submesh.GetSubMeshElementFromParent(el.index);
         if (!pe.IsLeaf()) { continue; }
         const auto gi = GI[pe.geom];
         bool new_id = false;
         for (int n = 0; n < gi.nv; n++)
         {
            new_nodes.insert(el.node[n]); // el.node are still from parent mesh at this stage.
         }
         for (int e = 0; e < gi.ne; e++)
         {
            new_nodes.insert(parent.nodes.FindId(el.node[gi.edges[e][0]],
                                                 el.node[gi.edges[e][1]]));
         }
      }

      parent_node_ids_.Reserve(static_cast<int>(new_nodes.size()));
      parent_to_submesh_node_ids_.reserve(new_nodes.size());
      for (const auto &n : new_nodes)
      {
         bool new_node;
         auto new_node_id = node_ids.Get(n, new_node);
         MFEM_ASSERT(new_node, "!");
         nodes.Alloc(new_node_id, new_node_id, new_node_id);

         parent_node_ids_.Append(n);
         parent_to_submesh_node_ids_[n] = new_node_id;
      }

      // Loop over submesh vertices, and add each node. Given submesh vertices respect
      // ordering of vertices in the parent mesh, this ensures all top level vertices are
      // added first as top level nodes. Some of these nodes will not be top level nodes,
      // and will require reparenting based on edge data.
      for (int iv = 0; iv < submesh.GetNV(); iv++)
      {
         bool new_node;
         int parent_vertex_id = submesh.GetParentVertexIDMap()[iv];
         int parent_node_id = parent.vertex_nodeId[parent_vertex_id];
         auto new_node_id = node_ids.Get(parent_node_id, new_node);
         MFEM_ASSERT(!new_node, "Each vertex's node should have already been added");
         nodes[new_node_id].vert_index = iv;
      }

      // Loop over elements and reference edges and faces (creating any nodes on first encounter).
      for (auto &el : elements)
      {
         if (el.IsLeaf())
         {
            const auto gi = GI[el.geom];
            bool new_id = false;

            for (int n = 0; n < gi.nv; n++)
            {
               // Relabel nodes from parent to submesh.
               el.node[n] = node_ids.Get(el.node[n], new_id);
               MFEM_ASSERT(new_id == false, "Should not be new.");
               nodes[el.node[n]].vert_refc++;
            }
            for (int e = 0; e < gi.ne; e++)
            {
               const int pid = parent.nodes.FindId(
                  parent_node_ids_[el.node[gi.edges[e][0]]],
                  parent_node_ids_[el.node[gi.edges[e][1]]]);
               MFEM_ASSERT(pid >= 0, "Edge not found");
               auto submesh_node_id = node_ids.Get(pid, new_id); // Convert parent id to a new submesh id.
               if (new_id)
               {
                  nodes.Alloc(submesh_node_id, submesh_node_id, submesh_node_id);
                  parent_node_ids_.Append(pid);
                  parent_to_submesh_node_ids_[pid] = submesh_node_id;
               }
               nodes[submesh_node_id].edge_refc++; // Register the edge
            }
            for (int f = 0; f < gi.nf; f++)
            {
               const int *fv = gi.faces[f];
               const int pid = parent.faces.FindId(
                  parent_node_ids_[el.node[fv[0]]],
                  parent_node_ids_[el.node[fv[1]]],
                  parent_node_ids_[el.node[fv[2]]],
                  el.node[fv[3]] >= 0 ? parent_node_ids_[el.node[fv[3]]]: - 1);
               MFEM_ASSERT(pid >= 0, "Face not found");
               const int id = faces.GetId(el.node[fv[0]], el.node[fv[1]], el.node[fv[2]], el.node[fv[3]]);
               // parent_face_ids_.Append(pid);
               // parent_to_submesh_face_ids_[pid] = id;
               faces[id].attribute = parent.faces[pid].attribute;
            }
         }
         else
         {
            // All elements have been collected, remap the child ids.
            for (int i = 0; i < ref_type_num_children[el.ref_type]; i++)
            {
               el.child[i] = parent_to_submesh_element_ids_[el.child[i]];
            }
         }
         el.parent = el.parent < 0 ? el.parent : parent_to_submesh_element_ids_.at(el.parent);
      }
   }
   else if (from == From::Boundary)
   {
      SubMeshUtils::ConstructFaceTree(parent, *this, attributes);
   }

   // Loop over all nodes, and reparent based on the node relations of the parent
   for (int i = 0; i < parent_node_ids_.Size(); i++)
   {
      const auto &parent_node = parent.nodes[parent_node_ids_[i]];
      const int submesh_p1 = parent_to_submesh_node_ids_[parent_node.p1];
      const int submesh_p2 = parent_to_submesh_node_ids_[parent_node.p2];
      nodes.Reparent(i, submesh_p1, submesh_p2);
   }

   nodes.UpdateUnused();
   for (int i = 0; i < elements.Size(); i++)
   {
      if (elements[i].IsLeaf())
      {
         // Register all faces
         RegisterFaces(i);
      }
   }

   InitRootElements();
   InitRootState(root_state.Size());
   InitGeomFlags();

#ifdef MFEM_DEBUG
   // Check all processors have the same number of roots
   {
      int p[2] = {root_state.Size(), -root_state.Size()};
      MPI_Allreduce(MPI_IN_PLACE, p, 2, MPI_INT, MPI_MIN, submesh.GetComm());
      MFEM_ASSERT(p[0] == -p[1], "Ranks must agree on number of root elements: min "
         << p[0] << " max " << -p[1] << " local " << root_state.Size() << " MyRank " << submesh.GetMyRank());
   }
#endif

   Update(); // Fills in secondary information based off of elements, nodes and faces.

   // If parent has coordinates defined, copy the relevant portion
   if (parent.coordinates.Size() > 0)
   {
      // Loop over new_nodes -> coordinates is indexed by node.
      coordinates.SetSize(3*parent_node_ids_.Size());
      parent.tmp_vertex = new TmpVertex[parent.nodes.NumIds()];
      // for (auto pn : parent_node_ids_)
      for (int n = 0; n < parent_node_ids_.Size(); n++)
      {

         // bool new_node = false;
         // auto n = node_ids.Get(pn, new_node);
         // MFEM_ASSERT(!new_node, "Should not be new");
         std::memcpy(&coordinates[3*n], parent.CalcVertexPos(parent_node_ids_[n]), 3*sizeof(real_t));
      }
   }

   // The element indexing was changed as part of generation of leaf elements. We need to
   // update the map.
   if (from == From::Domain)
   {
      // The element indexing was changed as part of generation of leaf elements. We need to
      // update the map.
      submesh.parent_to_submesh_element_ids_ = -1;
      for (int i = 0; i < submesh.parent_element_ids_.Size(); i++)
      {
         submesh.parent_element_ids_[i] =
            parent.elements[parent_element_ids_[leaf_elements[i]]].index;
         submesh.parent_to_submesh_element_ids_[submesh.parent_element_ids_[i]] = i;
      }
   }
   else
   {
      submesh.parent_to_submesh_element_ids_ = -1;
      // parent elements are BOUNDARY elements, need to map face index to be.
      const auto &parent_face_to_be = submesh.GetParent()->GetFaceToBdrElMap();

      MFEM_ASSERT(NElements == submesh.GetNE(), "!");

      auto new_parent_to_submesh_element_ids = submesh.parent_to_submesh_element_ids_;
      Array<int> new_parent_element_ids;
      new_parent_element_ids.Reserve(submesh.parent_element_ids_.Size());
      for (int i = 0; i < submesh.parent_element_ids_.Size(); i++)
      {
         auto leaf = leaf_elements[i];
         auto pe = parent_element_ids_[leaf];
         auto pfi = parent.faces[pe].index;
         auto pbe = parent_face_to_be[pfi];
         new_parent_element_ids.Append(
            parent_face_to_be[parent.faces[parent_element_ids_[leaf_elements[i]]].index]);
         new_parent_to_submesh_element_ids[new_parent_element_ids[i]] = i;
      }

      MFEM_ASSERT(new_parent_element_ids.Size() == submesh.parent_element_ids_.Size(), "!");
#ifdef MFEM_DEBUG
      for (auto x : new_parent_element_ids)
      {
         MFEM_ASSERT(std::find(submesh.parent_element_ids_.begin(),
                               submesh.parent_element_ids_.end(), x)
                     != submesh.parent_element_ids_.end(), x << " not found in submesh.parent_element_ids_");
      }
      for (auto x : submesh.parent_element_ids_)
      {
         MFEM_ASSERT(std::find(new_parent_element_ids.begin(),
                               new_parent_element_ids.end(), x)
                     != new_parent_element_ids.end(), x << " not found in new_parent_element_ids_");
      }
#endif
      submesh.parent_element_ids_ = new_parent_element_ids;
      submesh.parent_to_submesh_element_ids_ = new_parent_to_submesh_element_ids;
   }
}

} // namespace mfem

#endif // MFEM_USE_MPI