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

#include "submesh_utils.hpp"
#include "submesh.hpp"
#include "psubmesh.hpp"

namespace mfem
{
namespace SubMeshUtils
{
int UniqueIndexGenerator::Get(int i, bool &new_index)
{
   auto f = idx.find(i);
   if (f == idx.end())
   {
      idx[i] = counter;
      new_index = true;
      return counter++;
   }
   else
   {
      new_index = false;
      return (*f).second;
   }
}

template <typename ElementT>
bool ElementHasAttribute(const ElementT &el, const Array<int> &attributes)
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

std::tuple< Array<int>, Array<int> >
AddElementsToMesh(const Mesh& parent,
                  Mesh& mesh,
                  const Array<int> &attributes,
                  bool from_boundary)
{
   // Collect all vertices to be added.
   Array<int> parent_vertex_ids, parent_element_ids;
   const int ne = from_boundary ? parent.GetNBE() : parent.GetNE();
   Array<int> vert, submesh_vert;
   for (int i = 0; i < ne; i++)
   {
      const Element *pel = from_boundary ?
                           parent.GetBdrElement(i) : parent.GetElement(i);
      if (!ElementHasAttribute(*pel, attributes)) { continue; }

      pel->GetVertices(vert);
      parent_vertex_ids.Append(vert);
   }
   // Add vertices -> sorting ensures their ordering matches that in the original mesh. This
   // is key for being able to reconstruct vertex <-> node mappings with ncmeshes.
   parent_vertex_ids.Sort();
   parent_vertex_ids.Unique();

   UniqueIndexGenerator vertex_ids;
   vertex_ids.idx.reserve(parent_vertex_ids.Size());
   bool new_vert;
   for (auto v : parent_vertex_ids)
   {
      auto mesh_vertex_id = vertex_ids.Get(v, new_vert);
      MFEM_ASSERT(new_vert, "Vertex should be unique");
      mesh.AddVertex(parent.GetVertex(v));
   }

   for (int i = 0; i < ne; i++)
   {
      const Element *pel = from_boundary ?
                           parent.GetBdrElement(i) : parent.GetElement(i);
      if (!ElementHasAttribute(*pel, attributes)) { continue; }

      pel->GetVertices(vert);
      submesh_vert.SetSize(vert.Size());

      for (int iv = 0; iv < vert.Size(); iv++)
      {
         bool new_vertex;
         int mesh_vertex_id = vert[iv];
         int submesh_vertex_id = vertex_ids.Get(mesh_vertex_id, new_vertex);
         if (new_vertex)
         {
            mesh.AddVertex(parent.GetVertex(mesh_vertex_id));
            parent_vertex_ids.Append(mesh_vertex_id);
         }
         submesh_vert[iv] = submesh_vertex_id;
      }

      Element *el = mesh.NewElement(from_boundary ?
                                    parent.GetBdrElementType(i) : parent.GetElementType(i));
      el->SetVertices(submesh_vert);
      el->SetAttribute(pel->GetAttribute());
      mesh.AddElement(el);
      parent_element_ids.Append(i);
   }
   return {parent_vertex_ids, parent_element_ids};
}

void BuildVdofToVdofMap(const FiniteElementSpace& subfes,
                        const FiniteElementSpace& parentfes,
                        const SubMesh::From& from,
                        const Array<int>& parent_element_ids,
                        Array<int>& vdof_to_vdof_map)
{
   auto *m = subfes.GetMesh();
   vdof_to_vdof_map.SetSize(subfes.GetVSize());

   const int vdim = parentfes.GetVDim();

   IntegrationPointTransformation Tr;
   DenseMatrix T;
   Array<int> z1;
   for (int i = 0; i < m->GetNE(); i++)
   {
      Array<int> parent_vdofs;
      if (from == SubMesh::From::Domain)
      {
         parentfes.GetElementVDofs(parent_element_ids[i], parent_vdofs);
      }
      else if (from == SubMesh::From::Boundary)
      {
         if (parentfes.IsDGSpace())
         {
            MFEM_ASSERT(static_cast<const L2_FECollection*>
                        (parentfes.FEColl())->GetBasisType() == BasisType::GaussLobatto,
                        "Only BasisType::GaussLobatto is supported for L2 spaces");

            auto pm = parentfes.GetMesh();

            const Geometry::Type face_geom =
               pm->GetBdrElementGeometry(parent_element_ids[i]);
            int face_info, parent_volel_id;
            pm->GetBdrElementAdjacentElement(
               parent_element_ids[i], parent_volel_id, face_info);
            face_info = Mesh::EncodeFaceInfo(
                           Mesh::DecodeFaceInfoLocalIndex(face_info),
                           Geometry::GetInverseOrientation(
                              face_geom, Mesh::DecodeFaceInfoOrientation(face_info)));
            pm->GetLocalFaceTransformation(
               pm->GetBdrElementType(parent_element_ids[i]),
               pm->GetElementType(parent_volel_id),
               Tr.Transf,
               face_info);

            const FiniteElement *face_el =
               parentfes.GetTraceElement(parent_element_ids[i], face_geom);
            MFEM_VERIFY(dynamic_cast<const NodalFiniteElement*>(face_el),
                        "Nodal Finite Element is required");

            face_el->GetTransferMatrix(*parentfes.GetFE(parent_volel_id),
                                       Tr.Transf,
                                       T);

            parentfes.GetElementVDofs(parent_volel_id, z1);

            parent_vdofs.SetSize(vdim * T.Height());
            for (int j = 0; j < T.Height(); j++)
            {
               for (int k = 0; k < T.Width(); k++)
               {
                  if (T(j, k) != 0.0)
                  {
                     for (int vd=0; vd<vdim; vd++)
                     {
                        int sub_vdof = j + T.Height() * vd;
                        int parent_vdof = k + T.Width() * vd;
                        parent_vdofs[sub_vdof] =
                           z1[static_cast<int>(parent_vdof)];
                     }
                  }
               }
            }
         }
         else
         {
            parentfes.GetBdrElementVDofs(parent_element_ids[i], parent_vdofs);
         }
      }
      else
      {
         MFEM_ABORT("SubMesh::From type unknown");
      }

      Array<int> sub_vdofs;
      subfes.GetElementVDofs(i, sub_vdofs);

      MFEM_ASSERT(parent_vdofs.Size() == sub_vdofs.Size(),
         "elem " << i << ' ' << parent_vdofs.Size() << ' ' << sub_vdofs.Size());
      for (int j = 0; j < parent_vdofs.Size(); j++)
      {
         real_t sub_sign = 1.0;
         int sub_vdof = subfes.DecodeDof(sub_vdofs[j], sub_sign);

         real_t parent_sign = 1.0;
         int parent_vdof = parentfes.DecodeDof(parent_vdofs[j], parent_sign);

         vdof_to_vdof_map[sub_vdof] =
            (sub_sign * parent_sign > 0.0) ? parent_vdof : (-1-parent_vdof);
      }
   }

#ifdef MFEM_DEBUG
   auto tmp = vdof_to_vdof_map;
   tmp.Sort();
   tmp.Unique();

   if (tmp.Size() != vdof_to_vdof_map.Size())
   {
      std::stringstream msg;
      for (int i = 0; i < vdof_to_vdof_map.Size(); i++)
         for (int j = i + 1; j < vdof_to_vdof_map.Size(); j++)
         {
            auto x = vdof_to_vdof_map[i];
            auto y = vdof_to_vdof_map[j];
            if (x == y)
            {
               msg << "i " << i << " (" << x << ") j " << j << " (" << y << ")\n";
            }
         }
      MFEM_ABORT("vdof_to_vdof_map should be 1 to 1:\n" << msg.str());
   }
#endif

}

Array<int> BuildFaceMap(const Mesh& pm, const Mesh& sm,
                        const Array<int> &parent_element_ids)
{
   // TODO: Check if parent is really a parent of mesh

   Array<int> pfids(sm.GetNumFaces());
   pfids = -1;
   for (int i = 0; i < sm.GetNE(); i++)
   {
      int peid = parent_element_ids[i];

      Array<int> sel_faces, pel_faces, o;
      if (pm.Dimension() == 2)
      {
         sm.GetElementEdges(i, sel_faces, o);
         pm.GetElementEdges(peid, pel_faces, o);
      }
      else
      {
         sm.GetElementFaces(i, sel_faces, o);
         pm.GetElementFaces(peid, pel_faces, o);
      }

      MFEM_ASSERT(sel_faces.Size() == pel_faces.Size(), "internal error");

      for (int j = 0; j < sel_faces.Size(); j++)
      {
         if (pfids[sel_faces[j]] != -1)
         {
            MFEM_ASSERT(pfids[sel_faces[j]] == pel_faces[j], "internal error");
         }
         pfids[sel_faces[j]] = pel_faces[j];
      }
   }
   return pfids;
}

template <typename SubMeshT>
void AddBoundaryElements(SubMeshT &mesh, const std::unordered_map<int,int> &lface_to_boundary_attribute)
{
   mesh.Dimension();
   // TODO: Check if the mesh is a SubMesh or ParSubMesh.
   const int num_codim_1 = [&mesh]()
   {
      auto Dim = mesh.Dimension();
      if (Dim == 1) { return mesh.GetNV(); }
      else if (Dim == 2) { return mesh.GetNEdges(); }
      else if (Dim == 3) { return mesh.GetNFaces(); }
      else { MFEM_ABORT("Invalid dimension."); return -1; }
   }();

   if (mesh.Dimension() == 3)
   {
      // In 3D we check for `bel_to_edge`. It shouldn't have been set
      // previously.
      mesh.RemoveBoundaryElementToEdge();
   }

   int NumOfBdrElements = 0;
   for (int i = 0; i < num_codim_1; i++)
   {
      if (mesh.GetFaceInformation(i).IsBoundary())
      {
         NumOfBdrElements++;
      }
   }

   Array<Element *> boundary;
   Array<int> be_to_face;
   boundary.Reserve(NumOfBdrElements);
   be_to_face.Reserve(NumOfBdrElements);

   const auto &parent = *mesh.GetParent();
   const auto &parent_face_ids = mesh.GetParentFaceIDMap();
   const auto &parent_edge_ids = mesh.GetParentEdgeIDMap();
   const auto &parent_vertex_ids = mesh.GetParentVertexIDMap();

   const auto &parent_face_to_be = parent.GetFaceToBdrElMap();
   int max_bdr_attr = parent.bdr_attributes.Max();

   for (int i = 0; i < num_codim_1; i++)
   {
      if (mesh.GetFaceInformation(i).IsBoundary())
      {
         auto * be = mesh.GetFace(i)->Duplicate(&mesh);

         auto pfid = [&](int i)
         {
            switch (mesh.Dimension())
            {
               case 3: return parent_face_ids[i];
               case 2: return parent_edge_ids[i];
               case 1: return parent_vertex_ids[i];
            }
            MFEM_ABORT("!");
            return -1;
         };

         if (mesh.GetFrom() == SubMesh::From::Domain && mesh.Dimension() >= 2)
         {
            int pbeid = parent_face_to_be[pfid(i)];
            if (pbeid != -1)
            {
               be->SetAttribute(parent.GetBdrAttribute(pbeid));
            }
            else
            {
               auto ghost_attr = lface_to_boundary_attribute.find(mesh.Dimension() == 3 ? parent_face_ids[i] : parent_edge_ids[i]);
               int battr = ghost_attr != lface_to_boundary_attribute.end() ? ghost_attr->second : max_bdr_attr + 1;
               be->SetAttribute(battr);
            }
         }
         else
         {
            auto ghost_attr = lface_to_boundary_attribute.find(pfid(i));
            int battr = ghost_attr != lface_to_boundary_attribute.end() ? ghost_attr->second : max_bdr_attr + 1;
            be->SetAttribute(battr);
         }
         be_to_face.Append(i);
         boundary.Append(be);
      }
   }

   if (mesh.GetFrom() == SubMesh::From::Domain && mesh.Dimension() >= 2)
   {
      // Search for and count interior boundary elements
      int InteriorBdrElems = 0;
      for (int i=0; i<parent.GetNBE(); i++)
      {
         const int parentFaceIdx = parent.GetBdrElementFaceIndex(i);
         const int submeshFaceIdx =
            mesh.Dimension() == 3 ?
            mesh.GetSubMeshFaceFromParent(parentFaceIdx) :
            mesh.GetSubMeshEdgeFromParent(parentFaceIdx);

         if (submeshFaceIdx == -1) { continue; }
         if (mesh.GetFaceInformation(submeshFaceIdx).IsBoundary()) { continue; }

         InteriorBdrElems++;
      }

      if (InteriorBdrElems > 0)
      {
         const int OldNumOfBdrElements = NumOfBdrElements;
         NumOfBdrElements += InteriorBdrElems;
         boundary.Reserve(NumOfBdrElements);
         be_to_face.Reserve(NumOfBdrElements);

         // Search for and transfer interior boundary elements
         for (int i = 0; i < parent.GetNBE(); i++)
         {
            const int parentFaceIdx = parent.GetBdrElementFaceIndex(i);
            const int submeshFaceIdx =
               mesh.GetSubMeshFaceFromParent(parentFaceIdx);

            if (submeshFaceIdx == -1) { continue; }
            if (mesh.GetFaceInformation(submeshFaceIdx).IsBoundary())
            { continue; }

            auto * be = mesh.GetFace(submeshFaceIdx)->Duplicate(&mesh);
            be->SetAttribute(parent.GetBdrAttribute(i));

            boundary.Append(be);
            be_to_face.Append(submeshFaceIdx);
         }
      }
   }
   mesh.AddBdrElements(boundary, be_to_face);
}

// Explicit instantiations
template void AddBoundaryElements(SubMesh &mesh, const std::unordered_map<int,int> &);

#ifdef MFEM_USE_MPI
template void AddBoundaryElements(ParSubMesh &mesh, const std::unordered_map<int,int> &);
#endif


// namespace
// {
//    struct FaceNodes
//    {
//       std::array<int, NCMesh::MaxFaceNodes> nodes;
//       bool operator<(FaceNodes t2) const
//       {
//          std::array<int, MaxFaceNodes> t1 = nodes;
//          std::sort(t1.begin(), t1.end());
//          std::sort(t2.nodes.begin(), t2.nodes.end());
//          return std::lexicographical_compare(t1.begin(), t1.end(),
//             t2.nodes.begin(), t2.nodes.end());
//       };
//    };

// Geometry::Type FaceGeomFromNodes(const std::array<int, NCMesh::MaxFaceNodes> &nodes)
// {
//    if (nodes[3] == -1){ return Geometry::Type::TRIANGLE; }
//    if (nodes[0] == nodes[1] && nodes[2] == nodes[3]) { return Geometry::Type::SEGMENT; }
//    return Geometry::Type::SQUARE;
// };

// template <typename BaseT>
// struct ExposedNCMesh : public BaseT
// {
//    using BaseT::faces;
// };

// } // namespace

// template<typename NCMeshT, typename NCSubMeshT>
// void ConstructFaceTree(const NCMeshT &parent, NCSubMeshT &submesh)
// {
//    // Map from parent nodes to the new element in the ncsubmesh.
//    std::map<FaceNodes, int> pnodes_new_elem;
//    // Collect parent vertex nodes to add in sequence.
//    std::set<int> new_nodes;
//    std::unordered_map<int,int> parent_to_submesh_element_ids;
//    parent_to_submesh_element_ids.reserve(parent.GetNumFaces());
//    Array<int> parent_element_ids;
//    parent_element_ids.Reserve(parent.GetNumFaces());
//    const auto &face_list = const_cast<NCMeshT&>(parent).GetFaceList();
//    const auto &face_to_be = submesh.GetParent()->GetFaceToBdrElMap();
//    // Double indexing loop because parent.faces begin() and end() do not align with
//    // index 0 and size-1.
//    for (int i = 0, ipe = 0; ipe < parent.GetNumFaces(); i++)
//    {
//       const auto &face = parent.GetFace(i);
//       if (face.Unused()) { continue; }
//       ipe++; // actual possible parent element.
//       const auto &elem = parent.GetElement(face.elem[0] >= 0 ? face.elem[0] : face.elem[1]);
//       if (!HasAttribute(face, attributes)
//       || face_list.GetMeshIdType(face.index) == NCMesh::NCList::MeshIdType::MASTER
//       ) { continue; }

//       auto face_type = face_list.GetMeshIdType(face.index);
//       auto fn = FaceNodes{parent.FindFaceNodes(face)};
//       if (pnodes_new_elem.find(fn) != pnodes_new_elem.end()){ continue; }

//       // TODO: Internal nc submesh can be constructed and solved on, but the transfer
//       // to the parent mesh can be erroneous, this is likely due to not treating the
//       // changing orientation of internal faces for ncmesh within the ptransfermap.
//       MFEM_ASSERT(face.elem[0] < 0 || face.elem[1] < 0,
//          "Internal nonconforming boundaries are not reliably supported yet.");

//       auto face_geom = face_geom_from_nodes(fn.nodes);
//       int new_elem_id = AddElement(NCMesh::Element(face_geom, face.attribute));

//       // Rank needs to be established by presence (or lack of) in the submesh.
//       elements[new_elem_id].rank = [&parent, &face, &submesh, &face_to_be]()
//       {
//          auto rank0 = face.elem[0] >= 0 ? parent.elements[face.elem[0]].rank : -1;
//          auto rank1 = face.elem[1] >= 0 ? parent.elements[face.elem[1]].rank : -1;

//          if (rank0 < 0) { return rank1; }
//          if (rank1 < 0) { return rank0; }

//          // A left and right element are present, need to establish which side
//          // received the boundary element.
//          return rank0 < rank1 ? rank0 : rank1;
//       }();

//       pnodes_new_elem[fn] = new_elem_id;
//       parent_element_ids_.Append(i);
//       parent_to_submesh_element_ids_[i] = new_elem_id;


//       // Copy in the parent nodes. These will be relabeled once the tree is built.
//       std::copy(fn.nodes.begin(), fn.nodes.end(), elements[new_elem_id].node);
//       for (auto x : fn.nodes)
//          if (x != -1)
//          {
//             new_nodes.insert(x);
//          }
//       auto &gi = GI[face_geom];
//       gi.InitGeom(face_geom);
//       for (int e = 0; e < gi.ne; e++)
//       {
//          new_nodes.insert(parent.nodes.FindId(fn.nodes[gi.edges[e][0]],
//                                                 fn.nodes[gi.edges[e][1]]));
//       }

//       /*
//          - Check not top level face
//          - Check for parent of the newly entered element
//             - if not present, add in
//             - if present but different order, reorder so consistent with child
//                elements.
//          - Set .child in the parent of the newly entered element
//          - Set .parent in the newly entered element

//          Break if top level face or joined existing branch (without reordering).
//       */
//       while (true)
//       {
//          int child = parent.ParentFaceNodes(fn.nodes);
//          if (child == -1) // A root face
//          {
//             elements[new_elem_id].parent = -1;
//             break;
//          }

//          auto pelem = pnodes_new_elem.find(fn);
//          bool new_parent = pelem == pnodes_new_elem.end();
//          bool fix_parent = false;
//          if (new_parent)
//          {
//             // Add in this parent
//             int pelem_id = AddElement(NCMesh::Element(face_geom_from_nodes(fn.nodes), face.attribute));
//             pelem = pnodes_new_elem.emplace(fn, pelem_id).first;
//             auto parent_face_id = parent.faces.FindId(fn.nodes[0], fn.nodes[1], fn.nodes[2], fn.nodes[3]);
//             parent_element_ids_.Append(parent_face_id);
//          }
//          else
//          {
//             // There are two scenarios where the parent nodes should be rearranged:
//             // 1. The found face is a slave, then the master might have been added in
//             //    reverse orientation
//             // 2. The parent face was added from the central face of a triangle, the
//             //    orientation of the parent face is only fixed relative to the outer
//             //    child faces not the interior.
//             //    If either of these scenarios, and there's a mismatch, then reorder
//             //    the parent and all ancestors if necessary.
//             if (((elem.Geom() == Geometry::Type::TRIANGLE && child != 3) || face_type != NCList::MeshIdType::UNRECOGNIZED)
//             && !std::equal(fn.nodes.begin(), fn.nodes.end(), pelem->first.nodes.begin()))
//             {
//                fix_parent = true;
//                auto pelem_id = pelem->second;
//                auto &parent_elem = elements[pelem->second];
//                if (parent_elem.IsLeaf())
//                {
//                   // Set all node to -1. Check that they are all filled appropriately.
//                   for (int n = 0; n < MaxElemNodes; n++)
//                   {
//                      elements[pelem->second].node[n] = -1;
//                   }
//                }
//                else
//                {
//                   // This face already had children, reorder them to match the
//                   // permutation from the original nodes to the new face nodes. The
//                   // discovered parent order should be the same for all descendent
//                   // faces, if this branch is triggered twice for a given parent face,
//                   // duplicate child elements will be marked.
//                   int child[MaxFaceNodes];
//                   for (int i1 = 0; i1 < MaxFaceNodes; i1++)
//                      for (int i2 = 0; i2 < MaxFaceNodes; i2++)
//                         if (fn.nodes[i1] == pelem->first.nodes[i2])
//                         {
//                            child[i2] = parent_elem.child[i1]; break;
//                         }
//                   std::copy(child, child+MaxFaceNodes, parent_elem.child);
//                }
//                // Re-key the map
//                pnodes_new_elem.erase(pelem->first);
//                pelem = pnodes_new_elem.emplace(fn, pelem_id).first;
//             }
//          }
//          // Ensure parent element is marked as non-leaf.
//          elements[pelem->second].ref_type = Dim == 2 ? Refinement::XY : Refinement::X;
//          // Know that the parent element exists, connect parent and child
//          elements[pelem->second].child[child] = new_elem_id;
//          elements[new_elem_id].parent = pelem->second;

//          // If this was neither new nor a fixed parent, the higher levels of the tree have been built,
//          // otherwise we recurse up the tree to add/fix more parents.
//          if (!new_parent && !fix_parent) { break; }
//          new_elem_id = pelem->second;
//       }
//    }
//    parent_element_ids_.ShrinkToFit();

//    MFEM_ASSERT(parent_element_ids_.Size() == elements.Size(), parent_element_ids_.Size() << ' ' << elements.Size());
//    std::vector<FaceNodes> new_elem_to_parent_face_nodes(pnodes_new_elem.size());
//    /*
//       All elements have been added into the tree but
//       a) The nodes are all from the parent ncmesh
//       b) The nodes do not know their parents
//       c) The element ordering is wrong, root elements are not first
//       d) The parent and child element numbers reflect the incorrect ordering

//       1. Add in nodes in the same order from the parent ncmesh
//       2. Compute reordering of elements with root elements first.
//    */

//    // Add new nodes preserving parent mesh ordering
//    parent_node_ids_.Reserve(static_cast<int>(new_nodes.size()));
//    parent_to_submesh_node_ids_.reserve(new_nodes.size());
//    for (auto n : new_nodes)
//    {
//       bool new_node;
//       auto new_node_id = node_ids.Get(n, new_node);
//       MFEM_ASSERT(new_node, "!");
//       nodes.Alloc(new_node_id, new_node_id, new_node_id);
//       parent_node_ids_.Append(n);
//       parent_to_submesh_node_ids_[n] = new_node_id;
//    }
//    parent_node_ids_.ShrinkToFit();
//    new_nodes.clear(); // not needed any more.

//    // Comparator for deciding order of elements. Building the ordering from the parent
//    // ncmesh ensures the root ordering is common across ranks.
//    auto comp_elements = [&](int l, int r)
//    {
//       const auto &elem_l = elements[l];
//       const auto &elem_r = elements[r];
//       if (elem_l.parent == elem_r.parent)
//       {
//          const auto &fnl = new_elem_to_parent_face_nodes.at(l).nodes;
//          const auto &fnr = new_elem_to_parent_face_nodes.at(r).nodes;
//          return std::lexicographical_compare(fnl.begin(), fnl.end(), fnr.begin(), fnr.end());
//       }
//       else
//       {
//          return elem_l.parent < elem_r.parent;
//       }
//    };

//    auto parental_sorted = [&](const BlockArray<Element> &elements)
//    {
//       Array<int> indices(elements.Size());
//       std::iota(indices.begin(), indices.end(), 0);

//       return std::is_sorted(indices.begin(), indices.end(), comp_elements);
//    };

//    auto print_elements = [&](bool parent_nodes = true){
//       for (int e = 0; e < elements.Size(); e++)
//       {
//          auto &elem = elements[e];
//             auto &gi = GI[elem.geom];
//             gi.InitGeom(elem.Geom());
//          std::cout << "element " << e
//          << " elem.attribute " << elem.attribute;

//          if (elem.IsLeaf())
//          {
//             std::cout << " node ";
//             for (int n = 0; n < gi.nv; n++)
//             {
//                std::cout << (parent_nodes ? parent_to_submesh_node_ids_[elem.node[n]] : elem.node[n]) << ' ';
//             }
//          }
//          else
//          {
//             std::cout << " child ";
//             for (int c = 0; c < MaxElemChildren && elem.child[c] >= 0; c++)
//             {
//                std::cout << elem.child[c] << ' ';
//             }
//          }

//          std::cout << "parent " << elem.parent << '\n';
//       }
//       std::cout << "new_elem_to_parent_face_nodes\n";
//       for (int i = 0; i < new_elem_to_parent_face_nodes.size(); i++)
//       {
//          const auto &n = new_elem_to_parent_face_nodes[i].nodes;
//          std::cout << i << ": " << n[0] << ' ' << n[1] << ' ' << n[2] << ' ' << n[3] << '\n';
//       }
//    };

//    Array<int> new_to_old(elements.Size()), old_to_new(elements.Size());
//    int sorts = 0;
//    while (!parental_sorted(elements))
//    {
//       // Stably reorder elements in order of refinement, and by parental nodes within
//       // a nuclear family.
//       new_to_old.SetSize(elements.Size()), old_to_new.SetSize(elements.Size());
//       std::iota(new_to_old.begin(), new_to_old.end(), 0);
//       std::stable_sort(new_to_old.begin(), new_to_old.end(), comp_elements);

//       // Build the inverse relation -> for converting the old elements to new
//       for (int i = 0; i < elements.Size(); i++)
//       {
//          old_to_new[new_to_old[i]] = i;
//       }

//       // Permute whilst reordering new_to_old. Avoids unnecessary copies.
//       Permute(std::move(new_to_old), elements, parent_element_ids_, new_elem_to_parent_face_nodes);

//       parent_to_submesh_element_ids_.clear();
//       for (int i = 0; i < parent_element_ids_.Size(); i++)
//       {
//          if (parent_element_ids_[i] == -1) {continue;}
//          parent_to_submesh_element_ids_[parent_element_ids_[i]] = i;
//       }

//       // Apply the new ordering to child and parent elements
//       for (auto &elem : elements)
//       {
//          if (!elem.IsLeaf())
//          {
//             // Parent rank is minimum of child ranks.
//             elem.rank = std::numeric_limits<int>::max();
//             for (int c = 0; c < MaxElemChildren && elem.child[c] >= 0; c++)
//             {
//                elem.child[c] = old_to_new[elem.child[c]];
//                elem.rank = std::min(elem.rank, elements[elem.child[c]].rank);
//             }
//          }
//          elem.parent = elem.parent == -1 ? -1 : old_to_new[elem.parent];
//       }
//    }

//    // Apply new node ordering to relations, and sign in on edges/vertices
//    for (auto &elem : elements)
//    {
//       if (elem.IsLeaf())
//       {
//          bool new_id;
//          auto &gi = GI[elem.geom];
//          gi.InitGeom(elem.Geom());
//          for (int e = 0; e < gi.ne; e++)
//          {
//             const int pid = parent.nodes.FindId(
//                elem.node[gi.edges[e][0]], elem.node[gi.edges[e][1]]);
//             MFEM_ASSERT(pid >= 0, elem.node[gi.edges[e][0]] << ' ' << elem.node[gi.edges[e][1]]);
//             auto submesh_node_id = node_ids.Get(pid, new_id);
//             MFEM_ASSERT(!new_id, "!");
//             nodes[submesh_node_id].edge_refc++;
//          }
//          for (int n = 0; n < gi.nv; n++)
//          {
//             MFEM_ASSERT(parent_to_submesh_node_ids_.find(elem.node[n]) != parent_to_submesh_node_ids_.end(), "!");
//             elem.node[n] = parent_to_submesh_node_ids_[elem.node[n]];
//             nodes[elem.node[n]].vert_refc++;
//          }
//          // Register faces
//          for (int f = 0; f < gi.nf; f++)
//          {
//             auto *face = faces.Get(
//                elem.node[gi.faces[f][0]],
//                elem.node[gi.faces[f][1]],
//                elem.node[gi.faces[f][2]],
//                elem.node[gi.faces[f][3]]);
//             face->attribute = -1;
//             face->index = -1;
//          }
//       }
//    }
// }


} // namespace SubMeshUtils
} // namespace mfem
