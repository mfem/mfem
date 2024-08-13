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

#include "ncsubmesh.hpp"

#include <unordered_map>
#include "submesh_utils.hpp"
#include "submesh.hpp"
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


NCSubMesh::NCSubMesh(SubMesh& submesh, const NCMesh &parent, From from,
                     const Array<int> &attributes)
: NCMesh(), parent_(&parent), from_(from), attributes_(attributes)
{
   Dim = submesh.Dimension();
   spaceDim = submesh.SpaceDimension();
   MyRank = 0;
   Iso = true;
   Legacy = false;

   UniqueIndexGenerator node_ids;
   // Loop over parent leaf elements and add nodes for all vertices. Register as top level
   // nodes, will reparent when looping over edges. Cannot add edge nodes at same time
   // because top level vertex nodes must be contiguous and first in node list (see
   // coordinates).

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
      MFEM_ASSERT(new_node && (new_node_id == iv), "Adding vertices in order, should match exactly");
      nodes.Alloc(new_node_id, new_node_id, new_node_id);
      nodes[new_node_id].vert_index = new_node_id;
      // vertex_nodeId is now implicitly the identity.

      parent_node_ids_.Append(parent_node_id);
      parent_to_submesh_node_ids_[parent_node_id] = new_node_id;
      // std::cout << "Added node " << new_node_id << " w/ vertex " << new_node_id << std::endl;
   }

   std::cout << "parent_node_ids_.Size() " << parent_node_ids_.Size() << std::endl;
   for (const auto x : parent_node_ids_)
   {
      std::cout << x << ' ';
   }
   std::cout << std::endl;

   if (from == From::Domain)
   {
      // Loop over elements of the parent NCMesh. If the element has the attribute, copy it.
      parent_to_submesh_element_ids_.SetSize(parent.elements.Size());
      parent_to_submesh_element_ids_ = -1;

      // Loop over parent elements and copy those with the attribute.
      for (int ipe = 0; ipe < parent.elements.Size(); ipe++)
      {
         const auto& pe = parent.elements[ipe];
         if (!ElementHasAttribute(pe, attributes)) { continue; }

         const int elem_id = AddElement(pe);
         NCMesh::Element &el = elements[elem_id];
         parent_element_ids_.Append(ipe); // submesh -> parent
         parent_to_submesh_element_ids_[ipe] = elem_id; // parent -> submesh

         el.rank = MyRank; // Only serial.
         el.index = submesh.GetSubMeshElementFromParent(el.index);
         if (!pe.IsLeaf()) { continue; }

         const auto gi = GI[pe.geom];
         bool new_id = false;
         for (int n = 0; n < gi.nv; n++)
         {
            el.node[n] = node_ids.Get(el.node[n], new_id); // Relabel nodes from parent to submesh.
            MFEM_ASSERT(new_id == false, "Should not be new.");
            nodes[el.node[n]].vert_refc++;
         }
      }

      nodes.UpdateUnused();
      // Loop over elements and reference edges and faces (creating any nodes on first encounter).
      for (auto &el : elements)
      {
         if (el.IsLeaf())
         {
            const auto gi = GI[el.geom];
            bool new_id = false;
            for (int e = 0; e < gi.ne; e++)
            {
               const int pid = parent.nodes.FindId(
                  parent_node_ids_[el.node[gi.edges[e][0]]],
                  parent_node_ids_[el.node[gi.edges[e][1]]]);
               MFEM_ASSERT(pid >= 0, "Edge not found");
               auto submesh_node_id = node_ids.Get(pid, new_id); // Convert parent id to a new submesh id.
               if (new_id)
               {
                  // std::cout << "Adding edge node " << pid << " parents " << parent.nodes[pid].p1 << " " << parent.nodes[pid].p2 << std::endl;
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

      if constexpr (false)
      {

      }
      else
      {

         // Loop over faces, and check for attributes.
         // If matching attribute, then go to submesh, and get face vertices. The nodes were
         // added matching the vertex ordering, so can use the face vertex numbers as the node
         // numbers.
         // Initialize similarly to a mesh with hanging nodes directly, thus do not construct
         // the root elements, effectively not recognizing coarsening opportunities?
         parent_to_submesh_element_ids_.SetSize(parent.faces.Size());
         parent_to_submesh_element_ids_ = -1;
         const auto &parent_face_to_be = submesh.GetParent()->GetFaceToBdrElMap();

         const auto &mesh_parent = *submesh.GetParent();
         // Loop over parent nc faces, and find those that match the boundary attribute. Add these
         // as elements in the ncsubmesh, use the identity map between node and vertex indices
         // right now.
         Array<int> vertices; // storage for vertices extracted
         std::cout << "parent.faces.Size() " << parent.faces.Size() << std::endl;
         for (int i = 0, ipe = 0; ipe < parent.faces.Size(); /* nothing */)
         {
            const auto &f = parent.faces[i++];
            std::cout << "i " << i << " ipe " << ipe << " f.index " << f.index << " f.attribute " << f.attribute;
            std::cout << std::boolalpha << " f.Unused() " << f.Unused() << std::endl;

            // TODO: To allow for coarsening, need to use unused faces and those with
            // attribute, but f.index == -1 (internal master faces) to construct elements with
            // children.

            if (f.Unused())
            {
               // Do not increment ipe counter, this isn't a face in the actual mesh.
               continue;
            }

            // Face that maps to a face in the actual mesh
            int parent_be = parent_face_to_be[f.index];
            if (!ElementHasAttribute(f, attributes) || parent_be < 0)
            {
               ipe++; continue;
            }

            std::cout << "Adding boundary face " << i << std::endl;

            // face index -> submesh element
            auto submesh_elem_id = submesh.GetSubMeshElementFromParent(parent_be);
            const auto & elem = *submesh.GetElement(submesh_elem_id);
            elem.GetVertices(vertices);

            // Use the implicit vertex -> node identity map. TODO: Can we build a face
            // refinement tree? This would be needed for non-root elements.
            NCMesh::Element new_elem(elem.GetGeometryType(), f.attribute);
            for (int n = 0; n < vertices.Size(); n++)
            {
               new_elem.node[n] = vertices[n];
               nodes[new_elem.node[n]].vert_refc++;
               std::cout << "nodes[" << new_elem.node[n] << "].vert_refc " << nodes[new_elem.node[n]].vert_refc << std::endl;
            }
            new_elem.index = submesh_elem_id;
            new_elem.rank = MyRank;
            auto new_elem_id = AddElement(new_elem);
            parent_element_ids_.Append(ipe); // submesh nc element -> parent nc face
            parent_to_submesh_element_ids_[ipe] = new_elem_id; // parent nc face -> submesh nc element

            // Loop over the edges and register
            auto &gi = GI[elem.GetGeometryType()];
            gi.InitGeom(elem.GetGeometryType());
            bool new_id = false;
            for (int e = 0; e < elem.GetNEdges(); e++)
            {
               const int pid = parent.nodes.FindId(
                  parent_node_ids_[new_elem.node[gi.edges[e][0]]],
                  parent_node_ids_[new_elem.node[gi.edges[e][1]]]
               );
               MFEM_ASSERT(pid >= 0, "Edge not found");
               auto submesh_node_id = node_ids.Get(pid, new_id);
               if (new_id)
               {
                  std::cout << "Adding edge node " << pid << " parents " << parent.nodes[pid].p1 << " " << parent.nodes[pid].p2 << std::endl;
                  nodes.Alloc(submesh_node_id, submesh_node_id, submesh_node_id);
                  parent_node_ids_.Append(pid);
                  parent_to_submesh_node_ids_[pid] = submesh_node_id;
               }
               nodes[submesh_node_id].edge_refc++;
            }
            ipe++;
         }

         // TODO: How to register faces? Do we need to?
         // Loop over the faces of the submesh -> use the attribute assigned there, and set
         // the faces to be equal to the faces from there. Basically equivalent to
         // initialization from a Mesh as usual.
         const auto &face_to_be = submesh.GetFaceToBdrElMap();
         for (int i = 0; i < submesh.GetNumFaces(); i++)
         {
            const auto &sf = *submesh.GetFace(i);
            auto &gi = GI[sf.GetGeometryType()];
            gi.InitGeom(sf.GetGeometryType());

            const int * v = sf.GetVertices();
            const int nv = sf.GetNVertices();
            int nc_v[4];
            if (nv == 2)
            {
               // NCMesh uses the convention that edges are keyed (a a b b) for (a, b).
               // Additionally it orders key entries.
               int v0 = std::min(v[0], v[1]);
               int v1 = std::max(v[0], v[1]);
               nc_v[0] = v0; nc_v[1] = v0;
               nc_v[2] = v1; nc_v[3] = v1;
            }
            else
            {
               std::copy(v, v + nv, nc_v);
            }

            // instantiate face and bind attribute.
            auto face_id = faces.GetId(nc_v[0], nc_v[1], nc_v[2], nv > 3 ? nc_v[3]: -1);
            faces[face_id].attribute = face_to_be[i] < 0 ? -1 : submesh.GetBdrAttribute(face_to_be[i]);
            faces[face_id].index = i;
         }

         // NOTE: All surface submesh elements are leaves. There is no possibility of
         // coarsening, as updating the relationship to the parent mesh would be very
         // complicated. As a consequence, marking for coarsening based on the submesh is not
         // supported.
      }
   }

   // Loop over all nodes, and reparent based on the node relations of the parent
   for (int i = 0; i < parent_node_ids_.Size(); i++)
   {
      const auto &parent_node = parent.nodes[parent_node_ids_[i]];
      const int submesh_p1 = parent_to_submesh_node_ids_[parent_node.p1];
      const int submesh_p2 = parent_to_submesh_node_ids_[parent_node.p2];
      nodes.Reparent(i, submesh_p1, submesh_p2);
   }

   std::cout << __FILE__ << ':' << __LINE__ << std::endl;
   std::cout << "nodes.Size() " << nodes.Size() << std::endl;
   for (auto it = nodes.begin(); it != nodes.end(); ++it)
   {
      std::cout << it.index() << ' ' << it->p1 << ' ' << it->p2;
      if (it->p1 == it->p2)
      {
         std::cout << " (root)";
      }
      std::cout << std::endl;
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


   std::cout << __FILE__<< ':' << __LINE__ << std::endl;
   std::cout << "parent_element_ids_ ";
   for (auto x : parent_element_ids_)
   {
      std::cout << x << ' ';
   }
   std::cout << '\n';
   std::cout << __FILE__<< ':' << __LINE__ << std::endl;
   std::cout << "submesh.parent_element_ids_ ";
   for (auto x : submesh.parent_element_ids_)
   {
      std::cout << x << ' ';
   }
   std::cout << '\n';


   Update(); // Fills in secondary information based off of elements, nodes and faces.
   if (from == From::Domain)
   {
      // The element indexing was changed as part of generation of leaf elements. We need to
      // update the map.
      for (int i = 0; i < leaf_elements.Size(); i++)
      {
         submesh.parent_element_ids_[i] =
            parent.elements[parent_element_ids_[leaf_elements[i]]].index;
      }
   }

   // copy top-level vertex coordinates (leave empty if the mesh is curved)
   if (!submesh.GetNodes())
   {
      // // Map parent coordinates to submesh coordinates
      // int nroot = 0;
      // coordinates.SetSize(3*vertex_nodeId.Size());
      // coordinates = 0.0;
      // for (int i = 0; i < vertex_nodeId.Size(); i++)
      // {
      //    const auto& parent_node = parent.nodes[parent_node_ids_[vertex_nodeId[i]]];
      //    if (parent_node.p1 != parent_node.p2) {continue;}
      //    std::memcpy(&coordinates[3*i], &parent.coordinates[3*parent_node.vert_index], 3*sizeof(real_t));
      // }
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
   std::cout << __FILE__ << ':' << __LINE__ << std::endl;
   for (const auto &x : coordinates)
   {
      std::cout << x << ' ';
   }
   std::cout << std::endl;
   // std::terminate();
}

} // namespace mfem
