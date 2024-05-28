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

#include <unordered_map>
#include "submesh_utils.hpp"
#include "psubmesh.hpp"
#include "../ncmesh_tables.hpp"

namespace mfem
{

using namespace SubMeshUtils;

namespace
{

template <typename T>
bool ElementHasAttribute(const T &el, const Array<int> &attributes)
{
   for (int a = 0; a < attributes.Size(); a++)
   {
      if (el.GetAttribute() == attributes[a])
      {
         return true;
      }
   }
   return false;
}

}


ParNCSubMesh::ParNCSubMesh(ParSubMesh& submesh,
   const ParNCMesh &parent, From from, const Array<int> &attributes)
: ParNCMesh(), parent_(&parent), from_(from), attributes_(attributes)
{
   MyComm = submesh.GetComm();
   NRanks = submesh.GetNRanks();
   MyRank = submesh.GetMyRank();

   // get global rank
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   std::cout << "MyRank " << MyRank << " rank " << rank << std::endl;

   Dim = submesh.Dimension();
   spaceDim = submesh.SpaceDimension();
   Iso = true;
   Legacy = false;

   UniqueIndexGenerator node_ids;
   // Loop over parent leaf elements and add nodes for all vertices. Register as top level
   // nodes, will reparent when looping over edges. Cannot add edge nodes at same time
   // because top level vertex nodes must be contiguous and first in node list (see
   // coordinates).
   if (from == From::Domain)
   {
      // Loop over elements of the parent NCMesh. If the element has the attribute, copy it.
      parent_to_submesh_element_ids_.SetSize(parent.elements.Size());
      parent_to_submesh_element_ids_ = -1;

      std::set<int> new_nodes;
      for (int ipe = 0; ipe < parent.elements.Size(); ipe++)
      {
         const auto& pe = parent.elements[ipe];
         if (!ElementHasAttribute(pe, attributes)) { continue; }

         const int elem_id = AddElement(pe);
         NCMesh::Element &el = elements[elem_id];
         parent_element_ids_.Append(ipe); // submesh -> parent
         parent_to_submesh_element_ids_[ipe] = elem_id; // parent -> submesh

         std::cout << "el.rank " << el.rank << std::endl;
         if (pe.index >= submesh.parent_to_submesh_element_ids_.Size())
         {
            el.index = -1;
         }
         else
         {
            el.index = submesh.GetSubMeshElementFromParent(el.index);
         }
         if (!pe.IsLeaf()) { continue; }

         const auto gi = GI[pe.geom];
         bool new_id = false;
         for (int n = 0; n < gi.nv; n++)
         {
            new_nodes.insert(el.node[n]); // el.node are still from parent mesh at this stage.
         }
      }
      // std::cout << "new_nodes.size() " << new_nodes.size() << std::endl;
      for (const auto &n : new_nodes)
      {
         bool new_node;
         auto new_node_id = node_ids.Get(n, new_node);
         MFEM_ASSERT(new_node, "!");
         nodes.Alloc(new_node_id, new_node_id, new_node_id);

         parent_node_ids_.Append(n);
         parent_to_submesh_node_ids_[n] = new_node_id;
      }

      std::cout << "parent_node_ids_.Size() " << parent_node_ids_.Size() << std::endl;
      for (const auto x : parent_node_ids_)
      {
         std::cout << x << ' ';
      }
      std::cout << std::endl;

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
      std::cout << "node_ids.counter " << node_ids.counter << std::endl;

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
               parent_face_ids_.Append(pid);
               parent_to_submesh_face_ids_[pid] = id;
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
         el.parent = el.parent < 0 ? el.parent : parent_to_submesh_element_ids_[el.parent];
      }
   }
   else if (from == From::Boundary)
   {
      // Loop over nc faces, check if form a boundary element. Collect nodes and build maps
      // from elements to faces.
      const auto &parent_face_to_be = submesh.GetParent()->GetFaceToBdrElMap();
            // Loop over elements of the parent NCMesh. If the element has the attribute, copy it.
      parent_to_submesh_element_ids_.SetSize(parent.faces.Size());
      parent_to_submesh_element_ids_ = -1;
      std::set<int> new_nodes;
      std::cout << "elements.Size " << elements.Size() << std::endl;
      std::cout << "parent_face_to_be.Size() " << parent_face_to_be.Size() << std::endl;
      std::cout << "NFaces " << NFaces << std::endl;
      for (int i = 0, ipe = 0; ipe < parent.faces.Size(); /* nothing */)
      {
         const auto &f = parent.faces[i++];
         if (f.Unused()) { continue; }
         if (f.index >= parent.GetNFaces()) { ipe++; continue; }
         int parent_be = parent_face_to_be[f.index];
         std::cout << "i " << i << " ipe " << ipe << " f.index " << f.index << " parent_be " << parent_be << std::endl;
         if (parent_be < 0 || !ElementHasAttribute(f, attributes)) { ipe++; continue; }

         // Create new element and collect nodes
         const auto &elem = parent.elements[f.elem[0] >= 0 ? f.elem[0] : f.elem[1]];

         // If elem[0] and elem[1] are present, this is a conformal internal face, so nodes
         // can be discovered from either side. If only one is present, it is a boundary or
         // a slave face, and the nodes are discovered from the only element.
         const auto &mesh_face = *submesh.GetParent()->GetFace(f.index);
         int new_elem_id = AddElement(NCMesh::Element(mesh_face.GetGeometryType(), f.attribute));
         NCMesh::Element &new_elem = elements[new_elem_id];
         new_elem.index = submesh.GetSubMeshElementFromParent(parent_be);
         new_elem.rank = MyRank;
         new_elem.attribute = f.attribute;
         std::cout << "parent_be " << parent_be << " new_elem.index " << new_elem.index << std::endl;
         parent_element_ids_.Append(ipe); // submesh nc element -> parent nc face
         parent_to_submesh_element_ids_[ipe] = new_elem_id; // parent nc face -> submesh nc element

         MFEM_ASSERT(elem.IsLeaf(), "Parent element must be a leaf.");

         int local_face = find_local_face(elem.Geom(),
                                          find_node(elem, f.p1),
                                          find_node(elem, f.p2),
                                          find_node(elem, f.p3));

         const int * fv = GI[elem.Geom()].faces[local_face];

         // Collect vertex nodes from parent face
         constexpr int max_nodes_per_face = 4; // Largest face is quad.
         for (int n = 0; n < 4; n++)
         {
            new_elem.node[n] = elem.node[fv[n]]; // copy in parent nodes
            if (elem.node[fv[n]] >= 0)
            {
               new_nodes.insert(elem.node[fv[n]]);
            }
         }

         // Collect edge nodes from parent face
         auto &gi = GI[mesh_face.GetGeometryType()];
         gi.InitGeom(mesh_face.GetGeometryType());
         for (int e = 0; e < gi.ne; e++)
         {
            const int pid = parent.nodes.FindId(
               new_elem.node[gi.edges[e][0]],
               new_elem.node[gi.edges[e][1]]);
            MFEM_ASSERT(pid >= 0, "Edge not found");
            new_nodes.insert(pid);
         }

         ipe++;
      }

      std::cout << "new_nodes ";
      // Add nodes respecting the ordering of the parent ncmesh.
      for (const auto &n : new_nodes)
      {
         bool new_node;
         auto new_node_id = node_ids.Get(n, new_node);
         MFEM_ASSERT(new_node, "!");
         nodes.Alloc(new_node_id, new_node_id, new_node_id);
         parent_node_ids_.Append(n);
         parent_to_submesh_node_ids_[n] = new_node_id;
         std::cout << n << ' ';
      }
      std::cout << '\n';

      std::cout << "elements.Size() " << elements.Size() << std::endl;

      // Loop over elements relabeling nodes
      for (auto &el : elements)
      {
         MFEM_ASSERT(el.IsLeaf(), "Non-leaf elements not currently supported.");
         const auto gi = GI[el.geom];
         bool new_id = false;

         std::cout << "el.index " << el.index << " el.attribute " << el.attribute << std::endl;

         for (int n = 0; n < gi.nv; n++)
         {
            auto new_node_id = node_ids.Get(el.node[n], new_id);
            MFEM_ASSERT(new_id == false, "Should not be new.");
            el.node[n] = new_node_id;
            nodes[el.node[n]].vert_refc++; // Register the vertex
         }
         for (int e = 0; e < gi.ne; e++)
         {
            const int pid = parent.nodes.FindId(
               parent_node_ids_[el.node[gi.edges[e][0]]],
               parent_node_ids_[el.node[gi.edges[e][1]]]);
            MFEM_ASSERT(pid >= 0, "Edge not found");
            auto submesh_node_id = node_ids.Get(pid, new_id); // Convert parent id to a new submesh id.
            MFEM_ASSERT(!new_id, "Should not be new.");
            nodes[submesh_node_id].edge_refc++; // Register the edge
         }
      }

      // Register faces
      const auto &face_to_be = submesh.GetFaceToBdrElMap();
      Array<int> nodes;
      for (int i = 0; i < submesh.GetNumFaces(); i++)
      {
         const auto &sf = *submesh.GetFace(i);
         auto &gi = GI[sf.GetGeometryType()];
         gi.InitGeom(sf.GetGeometryType());

         // sub vert -> parent vert -> parent node -> submesh node
         sf.GetVertices(nodes);
         for (auto &n : nodes)
         {
            n = submesh.parent_vertex_ids_[n];
            n = submesh.GetParent()->ncmesh->vertex_nodeId[n];
            n = parent_to_submesh_node_ids_[n];
         }

         std::array<int, 4> nc_nodes{{-1,-1,-1,-1}};
         if (nodes.Size() == 2)
         {
            // NCMesh uses the convention that edges are keyed (a a b b) for (a, b).
            // Additionally it orders key entries.
            auto n0 = nodes.Min();
            auto n1 = nodes.Max();
            nc_nodes[0] = n0; nc_nodes[1] = n0;
            nc_nodes[2] = n1; nc_nodes[3] = n1;
         }
         else
         {
            std::copy(nodes.begin(), nodes.end(), nc_nodes.begin());
         }

         // instantiate face and bind attribute.
         auto face_id = faces.GetId(nc_nodes[0], nc_nodes[1], nc_nodes[2], nc_nodes[3]);
         faces[face_id].attribute = face_to_be[i] < 0 ? -1 : submesh.GetBdrAttribute(face_to_be[i]);
         faces[face_id].index = i;
      }
   }


   // std::cout << __FILE__ << ':' << __LINE__ << std::endl;
   // Loop over all nodes, and reparent based on the node relations of the parent
   for (int i = 0; i < parent_node_ids_.Size(); i++)
   {
      const auto &parent_node = parent.nodes[parent_node_ids_[i]];
      const int submesh_p1 = parent_to_submesh_node_ids_[parent_node.p1];
      const int submesh_p2 = parent_to_submesh_node_ids_[parent_node.p2];

      std::cout << "Reparenting " << i << " with " << submesh_p1 << ' ' << submesh_p2;
      std::cout << " vert_index " << nodes[i].vert_index << std::endl;
      nodes.Reparent(i, submesh_p1, submesh_p2);
   }

   // std::cout << __FILE__ << ':' << __LINE__ << std::endl;
   nodes.UpdateUnused();
   for (int i = 0; i < elements.Size(); i++)
   {
      if (elements[i].IsLeaf())
      {
         // Register all faces
         RegisterFaces(i);
      }
   }

   // std::cout << "faces.Size() " << faces.Size() << std::endl;


   InitRootElements();
   InitRootState(root_state.Size());
   InitGeomFlags();

   Update(); // Fills in secondary information based off of elements, nodes and faces.

   // copy top-level vertex coordinates (leave empty if the mesh is curved)
   if (!submesh.GetNodes())
   {
      // // Map parent coordinates to submesh coordinates
      // std::cout << __FILE__ << ':' << __LINE__ << std::endl;
      // std::cout << "vertex_nodeId.Size() " << vertex_nodeId.Size() << std::endl;
      // coordinates.SetSize(3*vertex_nodeId.Size());
      // parent.tmp_vertex = new TmpVertex[parent.nodes.NumIds()];
      // for (int i = 0; i < vertex_nodeId.Size(); i++)
      // {
      //    // const auto& parent_node = &parent.nodes[parent_node_ids_[vertex_nodeId[i]]];
      //    std::memcpy(&coordinates[3*i], parent.CalcVertexPos(parent_node_ids_[vertex_nodeId[i]]), 3*sizeof(real_t));
      //    std::cout << "coord " << i << ' ' << coordinates[3*i] << ' ' << coordinates[3*i+1] << ' ' << coordinates[3*i+2] << std::endl;
      // }

      // Loop over new_nodes -> coordinates is indexed by node.
      coordinates.SetSize(3*parent_node_ids_.Size());
      parent.tmp_vertex = new TmpVertex[parent.nodes.NumIds()];
      for (auto pn : parent_node_ids_)
      {
         bool new_node = false;
         auto n = node_ids.Get(pn, new_node);
         MFEM_ASSERT(!new_node, "Should not be new");
         std::memcpy(&coordinates[3*n], parent.CalcVertexPos(pn), 3*sizeof(real_t));
      }

   }

   if (from == From::Domain)
   {
      // The element indexing was changed as part of generation of leaf elements. We need to
      // update the map.
      std::cout << "leaf_elements.Size() " << leaf_elements.Size() << std::endl;
      for (int i = 0; i < submesh.parent_element_ids_.Size(); i++)
      {
         submesh.parent_element_ids_[i] =
            parent.elements[parent_element_ids_[leaf_elements[i]]].index;
      }
   }
}

} // namespace mfem

#endif // MFEM_USE_MPI