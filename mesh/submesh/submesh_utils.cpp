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

bool ElementHasAttribute(const Element &el, const Array<int> &attributes)
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
   Array<int> parent_vertex_ids, parent_element_ids;
   UniqueIndexGenerator vertex_ids;
   const int ne = from_boundary ? parent.GetNBE() : parent.GetNE();
   for (int i = 0; i < ne; i++)
   {
      const Element *pel = from_boundary ?
                           parent.GetBdrElement(i) : parent.GetElement(i);
      if (!ElementHasAttribute(*pel, attributes)) { continue; }

      Array<int> v;
      pel->GetVertices(v);
      Array<int> submesh_v(v.Size());

      for (int iv = 0; iv < v.Size(); iv++)
      {
         bool new_vertex;
         int mesh_vertex_id = v[iv];
         int submesh_vertex_id = vertex_ids.Get(mesh_vertex_id, new_vertex);
         if (new_vertex)
         {
            mesh.AddVertex(parent.GetVertex(mesh_vertex_id));
            parent_vertex_ids.Append(mesh_vertex_id);
         }
         submesh_v[iv] = submesh_vertex_id;
      }

      Element *el = mesh.NewElement(from_boundary ?
                                    parent.GetBdrElementType(i) : parent.GetElementType(i));
      el->SetVertices(submesh_v);
      el->SetAttribute(pel->GetAttribute());
      mesh.AddElement(el);
      parent_element_ids.Append(i);
   }
   return std::tuple<Array<int>, Array<int>>(parent_vertex_ids,
                                             parent_element_ids);
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

} // namespace SubMeshUtils
} // namespace mfem
