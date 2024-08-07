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

namespace
{

template <typename T>
bool HasAttribute(const T &el, const Array<int> &attributes)
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

template <typename T1>
void Permute(const Array<int>& indices, T1& t1)
{
   Permute(Array<int>(indices), t1);
}
template <typename T1>
void Permute(const Array<int>& indices, T1&& t1)
{
   Permute(Array<int>(indices), t1);
}

template <typename T1>
void Permute(Array<int>&& indices, T1& t1)
{
   for (int i = 0; i < indices.Size(); i++)
   {
      auto current = i;
      while (i != indices[current])
      {
         auto next = indices[current];
         std::swap(t1[current], t1[next]);
         indices[current] = current;
         current = next;
      }
      indices[current] = current;
   }
}

template <typename T1, typename T2, typename T3>
void Permute(const Array<int>& indices, T1& t1, T2& t2, T3& t3)
{
   Permute(Array<int>(indices), t1, t2, t3);
}

template <typename T1, typename T2, typename T3>
void Permute(Array<int>&& indices, T1& t1, T2& t2, T3& t3)
{
   for (int i = 0; i < indices.Size(); i++)
   {
      auto current = i;
      while (i != indices[current])
      {
         auto next = indices[current];
         std::swap(t1[current], t1[next]);
         std::swap(t2[current], t2[next]);
         std::swap(t3[current], t3[next]);
         indices[current] = current;
         current = next;
      }
      indices[current] = current;
   }
}

template <typename FaceNodes>
void ReorientFaceNodesByOrientation(FaceNodes &nodes, Geometry::Type geom, int orientation)
{
   auto permute = [&]() -> std::array<int, NCMesh::MaxFaceNodes>
   {
      if (geom == Geometry::Type::SEGMENT)
      {
         switch (orientation) // degenerate (0,0,1,1)
         {
            case 0: return {0,1,2,3};
            case 1: return {2,3,0,1};
            default: MFEM_ABORT("Unexpected orientation!");
         }
      }
      else if (geom == Geometry::Type::TRIANGLE)
      {
         switch (orientation)
         {
            case 0: return {0,1,2,3};
            case 5: return {0,2,1,3};
            case 2: return {1,2,0,3};
            case 1: return {1,0,2,3};
            case 4: return {2,0,1,3};
            case 3: return {2,1,0,3};
            default: MFEM_ABORT("Unexpected orientation!");
         }
      }
      else if (geom == Geometry::Type::SQUARE)
      {
         switch (orientation)
         {
            case 0: return {0,1,2,3};
            case 1: return {0,3,2,1};
            case 2: return {1,2,3,0};
            case 3: return {1,0,3,2};
            case 4: return {2,3,0,1};
            case 5: return {2,1,0,3};
            case 6: return {3,0,1,2};
            case 7: return {3,2,1,0};
            default: MFEM_ABORT("Unexpected orientation!");
         }
      }
      else { MFEM_ABORT("Unexpected face geometry!"); }
   }();
   Permute(Array<int>(permute.data(), NCMesh::MaxFaceNodes), nodes);
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

         std::cout << "el.rank " << el.rank << std::endl;

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
      // std::cout << "new_nodes.size() " << new_nodes.size() << std::endl;

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
         el.parent = el.parent < 0 ? el.parent : parent_to_submesh_element_ids_.at(el.parent);
      }
   }
   else if (from == From::Boundary)
   {

      /*
         1) Loop over elements,
            2) Loop over the faces
               i - if match attributes set nodes equal to the face nodes and start while
               loop
                  a) while not in map, add element
                  Use method
                     int AddElementFromFace(elem, face)
               which return the index of the element that's added.
               ii - Get nodes of parent face using NCMesh::ParentFaceNodes, maps from face
                  nodes in the parent ncmesh to parent face nodes in the parent face mesh.
                  Returned child index will be used to connect the previously added element
                  index.
               iii - while loop
                  - If the nodes aren't

      */

      if constexpr (true)
      {
         auto face_geom_from_nodes = [](const std::array<int, MaxFaceNodes> &nodes)
         {
            if (nodes[3] == -1){ return Geometry::Type::TRIANGLE; }
            if (nodes[0] == nodes[1] && nodes[2] == nodes[3]) { return Geometry::Type::SEGMENT; }
            return Geometry::Type::SQUARE;
         };

         // Helper struct for storing FaceNodes and allowing comparisons based on the sorted
         struct FaceNodes
         {
            std::array<int, MaxFaceNodes> nodes;
            bool operator<(FaceNodes t2) const
            {
               std::array<int, MaxFaceNodes> t1 = nodes;
               std::sort(t1.begin(), t1.end());
               std::sort(t2.nodes.begin(), t2.nodes.end());
               return std::lexicographical_compare(t1.begin(), t1.end(),
                  t2.nodes.begin(), t2.nodes.end());
            };
            // auto begin() -> decltype(nodes.begin()) { return nodes.begin(); }
            // auto end() -> decltype(nodes.end()) { return nodes.end(); }
            // auto begin() -> decltype(nodes.begin()) const { return nodes.begin(); }
            // auto end() -> decltype(nodes.end()) const { return nodes.end(); }
         };


         // Map from parent nodes to the new element in the ncsubmesh.
         std::map<FaceNodes, int> pnodes_new_elem;
         // Collect parent vertex nodes to add in sequence.
         std::set<int> new_nodes;
         // parent_to_submesh_element_ids_.SetSize(parent.faces.Size());
         // parent_to_submesh_element_ids_ = -1;
         parent_to_submesh_element_ids_.reserve(parent.faces.Size());
         parent_element_ids_.Reserve(parent.faces.Size());
         const auto &face_list = const_cast<ParNCMesh&>(parent).GetFaceList();

         const auto &face_to_be = submesh.GetParent()->GetFaceToBdrElMap();

         // Double indexing loop because parent.faces begin() and end() do not align with
         // index 0 and size-1.
         for (int i = 0, ipe = 0; ipe < parent.faces.Size(); i++)
         {
            const auto &face = parent.faces[i];
            if (face.Unused()) { continue; }
            ipe++; // actual possible parent element.
            const auto &elem = parent.elements[face.elem[0] >= 0 ? face.elem[0] : face.elem[1]];
            if (!HasAttribute(face, attributes)
            || face_list.GetMeshIdType(face.index) == NCList::MeshIdType::MASTER
            ) { continue; }

            auto fn = FaceNodes{parent.FindFaceNodes(face)};
            if (pnodes_new_elem.find(fn) != pnodes_new_elem.end())
            {
               std::cout << "Found before\n"; continue;
            }

            auto mesh_id_type = face_list.GetMeshIdType(face.index);

            if (mesh_id_type == NCList::MeshIdType::CONFORMING)
               std::cout << "CONFORMING\n";
            if (mesh_id_type == NCList::MeshIdType::MASTER)
               std::cout << "MASTER\n";
            if (mesh_id_type == NCList::MeshIdType::SLAVE)
               std::cout << "SLAVE\n";
            if (mesh_id_type == NCList::MeshIdType::UNRECOGNIZED)
               std::cout << "UNRECOGNIZED\n";

            std::cout << "face.elem " << face.elem[0] << ' ' << face.elem[1] << std::endl;
            if (face.elem[0] >= 0)
            { std::cout << " elem 0 IsLeaf " << parent.elements[face.elem[0]].IsLeaf() << std::endl; }

            if (face.elem[1] >= 0)
            { std::cout << " elem 1 IsLeaf " << parent.elements[face.elem[1]].IsLeaf() << std::endl; }

            std::cout << " face.index " << face.index << " parent.IsGhost(2, face.index) " << parent.IsGhost(2, face.index) << std::endl;

            auto face_geom = face_geom_from_nodes(fn.nodes);
            int new_elem_id = AddElement(NCMesh::Element(face_geom, face.attribute));
            elements[new_elem_id].rank = [&parent, &face]()
            {
               // Assign rank from lowest index element attached.
               if (face.elem[0] < 0) { return parent.elements[face.elem[1]].rank; }
               if (face.elem[1] < 0) { return parent.elements[face.elem[0]].rank; }
               return parent.elements[face.elem[0] < face.elem[1] ? face.elem[0] : face.elem[1]].rank;
            }();

            auto orientation = [&]()
            {
               int f, o;
               std::cout << "i " << i << " ipe " << ipe << " face.index " << face.index << std::endl;
               if (face.index >= face_to_be.Size() || face_to_be[face.index] < 0)
               { o = 0; std::cout << " face.index >= face_to_be.Size() " << (face.index >= face_to_be.Size()); }
               else
               {
               submesh.GetParent()->GetBdrElementFace(face_to_be[face.index], &f, &o);
               }
               std::cout << " o " << o << '\n';
               return o;
            }();
            ReorientFaceNodesByOrientation(fn.nodes, face_geom, orientation);


            // // Hack to manually fix the orientation
            // std::swap(fn[0], fn[1]);
            // std::swap(fn[2], fn[3]);
            pnodes_new_elem[fn] = new_elem_id;
            parent_element_ids_.Append(i);
            parent_to_submesh_element_ids_[i] = new_elem_id;

            std::cout << " new_elem_id " << new_elem_id << " fn ";

            // Copy in the parent nodes. These will be relabeled once the tree is built.
            std::copy(fn.nodes.begin(), fn.nodes.end(), elements[new_elem_id].node);
            for (auto x : fn.nodes)
               if (x != -1)
               {
                  new_nodes.insert(x);
                  std::cout << x << ' ';
               }
            std::cout << '\n';
            auto &gi = GI[face_geom];
            gi.InitGeom(face_geom);
            for (int e = 0; e < gi.ne; e++)
            {
               new_nodes.insert(parent.nodes.FindId(fn.nodes[gi.edges[e][0]],
                                                    fn.nodes[gi.edges[e][1]]));
            }

            /*
               - Check not top level face
               - Check for parent of the newly entered element
                  - if not present, add in
               - Set .child in the parent of the newly entered element
               - Set .parent in the newly entered element

               Break if top level face or joined existing branch.
            */
            while (true)
            {
               int child = parent.ParentFaceNodes(fn.nodes);
               if (child == -1) // A root face
               {
                  elements[new_elem_id].parent = -1;
                  break;
               }

               auto pelem = pnodes_new_elem.find(fn);
               bool new_parent = pelem == pnodes_new_elem.end();
               bool fix_parent = false;
               if (new_parent)
               {
                  // Add in this parent
                  int pelem_id = AddElement(NCMesh::Element(face_geom_from_nodes(fn.nodes), face.attribute));
                  pelem = pnodes_new_elem.emplace(fn, pelem_id).first;
                  auto parent_face_id = parent.faces.FindId(fn.nodes[0], fn.nodes[1], fn.nodes[2], fn.nodes[3]);
                  parent_element_ids_.Append(parent_face_id);
                  std::cout << "Adding new parent: pelem_id " << pelem_id << " fn.nodes " << fn.nodes[0] << ' ' << fn.nodes[1] << ' ' << fn.nodes[2] << ' ' << fn.nodes[3] << '\n';
               }
               else
               {
                  std::cout << "Found parent " << pelem->second << " with ";
                  for (auto x : pelem->first.nodes)
                  {
                     std::cout << x << ' ';
                  }
                  std::cout << " new ";
                  for (auto x : fn.nodes)
                  {
                     std::cout << x << ' ';
                  }
                  std::cout << '\n';
                  // Check that the parent face nodes ordering matches the ParentFaceNodes ordering.
                  if (!std::equal(fn.nodes.begin(), fn.nodes.end(), pelem->first.nodes.begin()))
                  {
                     fix_parent = true;
                     auto pelem_id = pelem->second;
                     std::cout << "Fixing " << pelem_id;
                     auto &parent_elem = elements[pelem->second];
                     if (parent_elem.IsLeaf())
                     {
                        // Set all node to -1. Check that they are all filled appropriately.
                        for (int n = 0; n < MaxElemNodes; n++)
                        {
                           elements[pelem->second].node[n] = -1;
                        }
                     }
                     else
                     {
                        // This face already had children, reorder them to match the
                        // permutation from the original nodes to the new face nodes. The
                        // discovered parent order should be the same for all descendent
                        // faces, if this branch is triggered twice for a given parent face,
                        // duplicate child elements will be marked.
                        int child[MaxFaceNodes];
                        for (int i1 = 0; i1 < MaxFaceNodes; i1++)
                           for (int i2 = 0; i2 < MaxFaceNodes; i2++)
                              if (fn.nodes[i1] == pelem->first.nodes[i2])
                              {
                                 child[i2] = parent_elem.child[i1]; break;
                              }
                        std::copy(child, child+MaxFaceNodes, parent_elem.child);
                        std::cout << " w/child ";
                        for (auto x : parent_elem.child)
                           if (x != -1)
                           {
                              std::cout << x << ' ';
                           }
                        std::cout << '\n';
                     }
                     // Re-key the map
                     pnodes_new_elem.erase(pelem->first);
                     pelem = pnodes_new_elem.emplace(fn, pelem_id).first;
                     std::cout << " to " << pelem->first.nodes[0] << ' '
                               << pelem->first.nodes[1] << ' '
                               << pelem->first.nodes[2] << ' '
                               << pelem->first.nodes[3] << " -> " << pelem->second << '\n';

                  }
               }
               // Ensure parent element is marked as non-leaf.
               elements[pelem->second].ref_type = Dim == 2 ? Refinement::XY : Refinement::X;
               // Know that the parent element exists, connect parent and child
               elements[pelem->second].child[child] = new_elem_id;
               elements[new_elem_id].parent = pelem->second;

               std::cout << " modified child ";
               for (auto x : elements[pelem->second].child)
                  if (x != -1)
                     {
                        std::cout << x << ' ';
                     }
               std::cout << '\n';

               // If this was neither new nor a fixed parent, the higher levels of the tree have been built,
               // otherwise we recurse up the tree to add/fix more parents.
               if (!new_parent && !fix_parent) { break; }
               new_elem_id = pelem->second;
            }
         }
         parent_element_ids_.ShrinkToFit();

         MFEM_ASSERT(parent_element_ids_.Size() == elements.Size(), parent_element_ids_.Size() << ' ' << elements.Size());

         std::cout << "R" << submesh.GetMyRank() << " pnodes_new_elem.size() " << pnodes_new_elem.size() << std::endl;
         std::vector<FaceNodes> new_elem_to_parent_face_nodes(pnodes_new_elem.size());
         for (const auto &kv : pnodes_new_elem)
         {
            const auto &n = kv.first.nodes;
            std::cout << kv.second << ":\t" << n[0] << ' ' << n[1] << ' ' << n[2] << ' ' << n[3] << '\n';
         }
         for (const auto &kv : pnodes_new_elem)
         {
            new_elem_to_parent_face_nodes.at(kv.second) = kv.first;
         }

         /*
            All elements have been added into the tree but
            a) The nodes are all from the parent ncmesh
            b) The nodes do not know their parents
            c) The element ordering is wrong, root elements are not first
            d) The parent and child element numbers reflect the incorrect ordering

            1. Add in nodes in the same order from the parent ncmesh
            2. Compute reordering of elements with root elements first.
         */

         // Add new nodes preserving parent mesh ordering
         parent_node_ids_.Reserve(static_cast<int>(new_nodes.size()));
         parent_to_submesh_node_ids_.reserve(new_nodes.size());
         std::cout << __FILE__ << ':' << __LINE__ << std::endl;
         std::cout << "new_nodes ";
         for (auto n : new_nodes)
         {
            bool new_node;
            auto new_node_id = node_ids.Get(n, new_node);
            MFEM_ASSERT(new_node, "!");
            nodes.Alloc(new_node_id, new_node_id, new_node_id);
            parent_node_ids_.Append(n);
            parent_to_submesh_node_ids_[n] = new_node_id;
            std::cout << n << ' ';
         }
         parent_node_ids_.ShrinkToFit();
         std::cout << '\n';
         new_nodes.clear(); // not needed any more.


         std::cout << "parent_node_ids_.Size() " << parent_node_ids_.Size() << std::endl;
         for (const auto x : parent_node_ids_)
         {
            std::cout << x << ' ';
         }
         std::cout << std::endl;

         // Comparator for deciding order of elements. Building the ordering from the parent
         // ncmesh ensures the root ordering is common across ranks.
         auto comp_elements = [&](int l, int r)
         {
            const auto &elem_l = elements[l];
            const auto &elem_r = elements[r];
            if (elem_l.parent == elem_r.parent)
            {
               const auto &fnl = new_elem_to_parent_face_nodes.at(l).nodes;
               const auto &fnr = new_elem_to_parent_face_nodes.at(r).nodes;
               return std::lexicographical_compare(fnl.begin(), fnl.end(), fnr.begin(), fnr.end());
            }
            else
            {
               return elem_l.parent < elem_r.parent;
            }
         };

         auto parental_sorted = [&](const BlockArray<Element> &elements)
         {
            Array<int> indices(elements.Size());
            std::iota(indices.begin(), indices.end(), 0);

            return std::is_sorted(indices.begin(), indices.end(), comp_elements);
         };

         auto print_elements = [&](bool parent_nodes = true){
            for (int e = 0; e < elements.Size(); e++)
            {
               auto &elem = elements[e];
                  auto &gi = GI[elem.geom];
                  gi.InitGeom(elem.Geom());
               std::cout << "element " << e
               << " elem.attribute " << elem.attribute;

               if (elem.IsLeaf())
               {
                  std::cout << " node ";
                  for (int n = 0; n < gi.nv; n++)
                  {
                     std::cout << (parent_nodes ? parent_to_submesh_node_ids_[elem.node[n]] : elem.node[n]) << ' ';
                  }
               }
               else
               {
                  std::cout << " child ";
                  for (int c = 0; c < MaxElemChildren && elem.child[c] >= 0; c++)
                  {
                     std::cout << elem.child[c] << ' ';
                  }
               }

               std::cout << "parent " << elem.parent << '\n';
            }
            std::cout << "new_elem_to_parent_face_nodes\n";
            for (int i = 0; i < new_elem_to_parent_face_nodes.size(); i++)
            {
               const auto &n = new_elem_to_parent_face_nodes[i].nodes;
               std::cout << i << ": " << n[0] << ' ' << n[1] << ' ' << n[2] << ' ' << n[3] << '\n';
            }
         };

         Array<int> new_to_old(elements.Size()), old_to_new(elements.Size());
         int sorts = 0;
         while (!parental_sorted(elements))
         {
            std::cout << "\n\nsort : " << sorts++ << "\n\n";
            // Stably reorder elements in order of refinement, and by parental nodes within
            // a nuclear family.
            new_to_old.SetSize(elements.Size()), old_to_new.SetSize(elements.Size());
            std::iota(new_to_old.begin(), new_to_old.end(), 0);
            std::stable_sort(new_to_old.begin(), new_to_old.end(), comp_elements);

            // Build the inverse relation -> for converting the old elements to new
            for (int i = 0; i < elements.Size(); i++)
            {
               old_to_new[new_to_old[i]] = i;
            }

            std::cout << __FILE__ << ':' << __LINE__ << std::endl;
            print_elements();

            std::cout << "new_to_old ";
            for (auto x : new_to_old)
            {
               std::cout << x << ' ';
            }
            std::cout << '\n';


            // // Destructively use the new_to_old map to reorder elements without
            // // needing a temporary copy of elements.
            // for (int i = 0; i < new_to_old.size(); ++i)
            // {
            //    while (i != new_to_old[i])
            //    {
            //       std::cout << i << " -> " << new_to_old[i] << '\n';
            //       int target_index = new_to_old[i];
            //       std::swap(elements[i], elements[target_index]);
            //       std::swap(new_to_old[i], new_to_old[target_index]);
            //    }
            // }

            // TODO: Write Permute for BlockArray
            // std::vector<Element> new_elements;
            // new_elements.reserve(elements.Size());
            // for (auto i : new_to_old)
            // {
            //    new_elements.emplace_back(elements[i]);
            // }
            // for (int i = 0; i < elements.Size(); i++)
            // {
            //    elements[i] = std::move(new_elements[i]);
            // }
            std::cout << __FILE__ << ':' << __LINE__ << '\n';
            for (auto x : parent_element_ids_)
               std::cout << x << ' ';
            std::cout << '\n';

            // parent_element_ids_.Permute(std::move(new_to_old)); // Destroys new_to_old

            // Permute whilst reordering new_to_old. Avoids unnecessary copies.
            Permute(std::move(new_to_old), elements, parent_element_ids_, new_elem_to_parent_face_nodes);


            std::cout << __FILE__ << ':' << __LINE__ << '\n';
            for (auto x : parent_element_ids_)
               std::cout << x << ' ';
            std::cout << '\n';

            parent_to_submesh_element_ids_.clear();
            for (int i = 0; i < parent_element_ids_.Size(); i++)
            {
               if (parent_element_ids_[i] == -1) {continue;}
               parent_to_submesh_element_ids_[parent_element_ids_[i]] = i;
            }

            // Apply the new ordering to child and parent elements
            for (auto &elem : elements)
            {
               if (!elem.IsLeaf())
               {
                  // Parent rank is minimum of child ranks.
                  elem.rank = std::numeric_limits<int>::max();
                  std::cout << " child ";
                  for (int c = 0; c < MaxElemChildren && elem.child[c] >= 0; c++)
                  {
                     elem.child[c] = old_to_new[elem.child[c]];
                     elem.rank = std::min(elem.rank, elements[elem.child[c]].rank);
                     std::cout << elem.child[c] << ' ';
                  }
                  std::cout << '\n';
               }
               elem.parent = elem.parent == -1 ? -1 : old_to_new[elem.parent];
            }
         }

         std::cout << __FILE__ << ':' << __LINE__ << std::endl;
         print_elements();

         // Apply new node ordering to relations, and sign in on edges/vertices
         for (auto &elem : elements)
         {
            if (elem.IsLeaf())
            {
               bool new_id;
               auto &gi = GI[elem.geom];
               gi.InitGeom(elem.Geom());
               for (int e = 0; e < gi.ne; e++)
               {
                  const int pid = parent.nodes.FindId(
                     elem.node[gi.edges[e][0]], elem.node[gi.edges[e][1]]);
                  MFEM_ASSERT(pid >= 0, elem.node[gi.edges[e][0]] << ' ' << elem.node[gi.edges[e][1]]);
                  auto submesh_node_id = node_ids.Get(pid, new_id);
                  MFEM_ASSERT(!new_id, "!");
                  nodes[submesh_node_id].edge_refc++;
               }
               for (int n = 0; n < gi.nv; n++)
               {
                  MFEM_ASSERT(parent_to_submesh_node_ids_.find(elem.node[n]) != parent_to_submesh_node_ids_.end(), "!");
                  elem.node[n] = parent_to_submesh_node_ids_[elem.node[n]];
                  nodes[elem.node[n]].vert_refc++;
               }
               // Register faces
               for (int f = 0; f < gi.nf; f++)
               {
                  auto *face = faces.Get(
                     elem.node[gi.faces[f][0]],
                     elem.node[gi.faces[f][1]],
                     elem.node[gi.faces[f][2]],
                     elem.node[gi.faces[f][3]]);
                  face->attribute = -1;
                  face->index = -1;
               }
            }
         }

         std::cout << __FILE__ << ':' << __LINE__ << std::endl;
         print_elements(false);
      }
      else
      {
         // Loop over nc faces, check if form a boundary element. Collect nodes and build maps
         // from elements to faces.
         const auto &parent_face_to_be = submesh.GetParent()->GetFaceToBdrElMap();
         const auto &face_list = const_cast<ParNCMesh&>(parent).GetFaceList();
         // Loop over elements of the parent NCMesh. If the element has the attribute, copy it.
         // parent_to_submesh_element_ids_.SetSize(parent.faces.Size());
         parent_to_submesh_element_ids_.reserve(parent.faces.Size());
         // parent_to_submesh_element_ids_ = -1;
         std::set<int> new_nodes;
         std::cout << "elements.Size " << elements.Size() << std::endl;
         std::cout << "parent_face_to_be.Size() " << parent_face_to_be.Size() << std::endl;
         std::cout << "NFaces " << NFaces << std::endl;
         for (int i = 0, ipe = 0; ipe < parent.faces.Size(); /* nothing */)
         {
            const auto &f = parent.faces[i++];
            std::cout << "i " << i << " ipe " << ipe << " f.index " << f.index << " f.attribute " << f.attribute << " f.elem " << f.elem[0] << ' ' << f.elem[1] << std::endl;
            if (f.Unused()) { continue; }
            if (!HasAttribute(f, attributes)){ ipe++; continue; }

            auto mesh_id_type = face_list.GetMeshIdType(f.index);
            if (face_list.GetMeshIdType(f.index) == NCList::MeshIdType::MASTER) { ipe++; continue; }

            if (mesh_id_type == NCList::MeshIdType::CONFORMING)
               std::cout << "CONFORMING\n";
            if (mesh_id_type == NCList::MeshIdType::MASTER)
               std::cout << "MASTER\n";
            if (mesh_id_type == NCList::MeshIdType::SLAVE)
               std::cout << "SLAVE\n";
            if (mesh_id_type == NCList::MeshIdType::UNRECOGNIZED)
               std::cout << "UNRECOGNIZED\n";

            // If don't find a boundary element, this face is either a ghost master OR be is on
            // a different rank.
            int parent_be = f.index >= parent.GetNFaces() ? -1 : parent_face_to_be[f.index];
            // If elem[0] and elem[1] are present, this is a conformal internal face, so nodes
            // can be discovered from either side. If only one is present, it is a boundary or
            // a slave face, and the nodes are discovered from the only element.
            const auto &elem = parent.elements[f.elem[0] >= 0 ? f.elem[0] : f.elem[1]];

            // Create new element and collect nodes
            auto face_geom_from_parent = [&elem, &f](const Geometry::Type &parent_geom)
            {
               switch (parent_geom)
               {
                  case Geometry::Type::TETRAHEDRON:
                     return Geometry::Type::TRIANGLE;
                  case Geometry::Type::CUBE:
                     return Geometry::Type::SQUARE;
                  case Geometry::Type::SQUARE:
                  case Geometry::Type::TRIANGLE:
                     return Geometry::Type::SEGMENT;
                  case Geometry::Type::SEGMENT:
                     return Geometry::Type::POINT;
                  case Geometry::Type::PRISM:
                  case Geometry::Type::PYRAMID:
                     {
                        // Have to figure out if square or triangle. Select the nodes for each
                        // face, then check if all those from f are contained
                        std::array<int, MaxFaceNodes> face_nodes;
                        auto &gi = GI[elem.Geom()];
                        for (int lf = 0; lf < gi.nf; lf++)
                        {
                           for (int j = 0; j < MaxFaceNodes; j++)
                           {
                              face_nodes[j] = elem.node[gi.faces[lf][j]];
                           }
                           if (std::find(face_nodes.begin(), face_nodes.end(), f.p1) != face_nodes.end()
                              && std::find(face_nodes.begin(), face_nodes.end(), f.p2) != face_nodes.end()
                              && std::find(face_nodes.begin(), face_nodes.end(), f.p3) != face_nodes.end())
                           {
                              break; // found!
                           }
                        }
                        return face_nodes.back() == -1 ? Geometry::Type::TRIANGLE : Geometry::Type::SQUARE;
                     }
                  default:
                     return Geometry::Type::INVALID;
               }
            };

            int new_elem_id = AddElement(NCMesh::Element(face_geom_from_parent(elem.Geom()), f.attribute));
            NCMesh::Element &new_elem = elements[new_elem_id];
            new_elem.index = submesh.GetSubMeshElementFromParent(parent_be);
            new_elem.rank = elem.rank; // Inherit rank from the controlling element
            new_elem.attribute = f.attribute;
            new_elem.ref_type = elem.ref_type;
            std::cout << "parent_be " << parent_be << " new_elem.index " << new_elem.index << " elem.ref_type " << int(elem.ref_type) << std::endl;
            parent_element_ids_.Append(ipe); // submesh nc element -> parent nc face
            parent_to_submesh_element_ids_[ipe] = new_elem_id; // parent nc face -> submesh nc element

            MFEM_ASSERT(elem.IsLeaf(), "Parent element must be a leaf.");

            int local_face = find_local_face(elem.Geom(),
                                             find_node(elem, f.p1),
                                             find_node(elem, f.p2),
                                             find_node(elem, f.p3));

            const int * fv = GI[elem.Geom()].faces[local_face];

            // Collect vertex nodes from parent face
            for (int n = 0; n < MaxFaceNodes; n++)
            {
               new_elem.node[n] = elem.node[fv[n]]; // copy in parent nodes
               if (elem.node[fv[n]] >= 0)
               {
                  new_nodes.insert(elem.node[fv[n]]);
               }
            }

            // Collect edge nodes from parent face
            auto &gi = GI[new_elem.Geom()];
            gi.InitGeom(new_elem.Geom());
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

         std::cout << __FILE__ << ':' << __LINE__ << std::endl;
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
         parent_node_ids_.ShrinkToFit();
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
               std::cout << "parent_node_ids_[el.node[gi.edges[e][0]]] " << parent_node_ids_[el.node[gi.edges[e][0]]]
               << " parent_node_ids_[el.node[gi.edges[e][1]]] " << parent_node_ids_[el.node[gi.edges[e][1]]] << std::endl;
               MFEM_ASSERT(pid >= 0, "Edge not found");
               auto submesh_node_id = node_ids.Get(pid, new_id); // Convert parent id to a new submesh id.
               MFEM_ASSERT(!new_id, "Should not be new.");
               nodes[submesh_node_id].edge_refc++; // Register the edge
            }
            // All the nodes for vertices and edges have been added in. Build the faces now
            for (int f = 0; f < gi.nf; f++)
            {
               auto face_id = faces.GetId(el.node[gi.faces[f][0]], el.node[gi.faces[f][1]], el.node[gi.faces[f][2]], el.node[gi.faces[f][3]]);
               // TODO: Add in face, but do not give any data.
               faces[face_id].attribute = -1;
               faces[face_id].index = -1;
            }
         }
      }
   }

   // for (const auto &e : elements)
   for (int i = 0; i < elements.Size(); i++)
   {
      const auto &e = elements[i];
      std::cout << std::boolalpha;
      std::cout << "elem " << i << " e.attribute " << e.attribute
      << " e.IsLeaf() " << e.IsLeaf()
      << " e.parent " << e.parent
      << " e.ref_type " << int(e.ref_type)
      << " e.rank " << e.rank;
      if (e.IsLeaf())
      {
         std::cout << " node ";
         for (int n = 0; n < 8; n++)
         {
            std::cout << e.node[n] << ' ';
         }
      }
      else
      {
         std::cout << " child ";
         for (int n = 0; n < 10; n++)
         {
            std::cout << e.child[n] << ' ';
         }
      }
      std::cout << std::endl;
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

   // Check all processors have the same number of roots
   {
      int p[2] = {root_state.Size(), -root_state.Size()};
      MPI_Allreduce(MPI_IN_PLACE, p, 2, MPI_INT, MPI_MIN, submesh.GetComm());
      MFEM_ASSERT(p[0] == -p[1], "Ranks must agree on number of root elements: min "
         << p[0] << " max " << -p[1] << " local " << root_state.Size() << " MyRank " << submesh.GetMyRank());
   }


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
      std::cout << __FILE__ << ':' << __LINE__ << std::endl;
      for (int i = 0; i < coordinates.Size() / 3; i++)
      {
         std::cout << "node " << i
         << '\t' << coordinates[3*i + 0]
         << '\t' << coordinates[3*i + 1]
         << '\t' << coordinates[3*i + 2] << '\n';
      }

   }

   // The element indexing was changed as part of generation of leaf elements. We need to
   // update the map.
   if (from == From::Domain)
   {
      // The element indexing was changed as part of generation of leaf elements. We need to
      // update the map.
      std::cout << "leaf_elements.Size() " << leaf_elements.Size() << std::endl;
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

      std::cout << "leaf_elements.Size() " << leaf_elements.Size() << std::endl;
      submesh.parent_to_submesh_element_ids_ = -1;
      // parent elements are BOUNDARY elements, need to map face index to be.
      const auto &parent_face_to_be = submesh.GetParent()->GetFaceToBdrElMap();
      for (int i = 0; i < submesh.parent_element_ids_.Size(); i++)
      {
         auto leaf = leaf_elements[i];
         auto pe = parent_element_ids_[leaf];
         auto pfi = parent.faces[pe].index;
         auto pbe = parent_face_to_be[pfi];
         std::cout << i << ' ' << leaf << ' ' << pe << ' ' << pfi << ' ' << pbe << '\n';
         submesh.parent_element_ids_[i] =
            parent_face_to_be[parent.faces[parent_element_ids_[leaf_elements[i]]].index];
         submesh.parent_to_submesh_element_ids_[submesh.parent_element_ids_[i]] = i;
      }

      // auto new_parent_element_ids = parent_element_ids_;
      // auto new_parent_to_submesh_element_ids = parent_to_submesh_element_ids_;
      // new_parent_to_submesh_element_ids = -1;
      // std::vector<int> reorder{7,6,5,0,2,1,4,3};
      // submesh.parent_element_ids_.Permute(std::move(reorder));
      // parent_to_submesh_element_ids_ = -1;
      // for (int i = 0; i < submesh.parent_element_ids_.Size(); i++)
      // {
      //    parent_to_submesh_element_ids_[submesh.parent_element_ids_[i]] = i;
      // }
   }

}

} // namespace mfem

#endif // MFEM_USE_MPI