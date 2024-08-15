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
            std::cout << __FILE__ << ':' << __LINE__ << std::endl;
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

         // for (auto x : {4,6,9,11})
         //    if (sub_vdof == x)
         //    {
         //       std::cout << "element " << i << " maps " << x << " to " << parent_vdof << '\n';
         //    }
         // for (auto x : {114, 119})
         //    if (parent_vdof == x)
         //    {
         //       std::cout << "element " << i << " maps " << sub_vdof << " to " << x << std::endl;
         //       // std::cout << "parent element " << parent_element_ids[i] << " parent contains " << x << std::endl;
         //    }

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
      std::cout << "duplicates found in dof map\n";
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
      MFEM_ABORT("vdof_to_vdof_map should be 1 to 1: " << msg.str());
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
void AddBoundaryElements(SubMeshT &mesh)
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

   const auto &parent_face_to_be = parent.GetFaceToBdrElMap();
   int max_bdr_attr = parent.bdr_attributes.Max();

   for (int i = 0; i < num_codim_1; i++)
   {
      if (mesh.GetFaceInformation(i).IsBoundary())
      {
         auto * be = mesh.GetFace(i)->Duplicate(&mesh);

         if (mesh.GetFrom() == SubMesh::From::Domain && mesh.Dimension() >= 2)
         {
            int pbeid = mesh.Dimension() == 3 ? parent_face_to_be[parent_face_ids[i]] :
                        parent_face_to_be[parent_edge_ids[i]];
            if (pbeid != -1)
            {
               be->SetAttribute(parent.GetBdrAttribute(pbeid));
            }
            else
            {
               be->SetAttribute(max_bdr_attr + 1);
            }
         }
         else
         {
            be->SetAttribute(max_bdr_attr + 1);
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
template void AddBoundaryElements(SubMesh &mesh);

#ifdef MFEM_USE_MPI
template void AddBoundaryElements(ParSubMesh &mesh);
#endif

} // namespace SubMeshUtils
} // namespace mfem
