// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "submesh.hpp"
#include "submesh_utils.hpp"
#include "../../fem/gridfunc.hpp"

using namespace mfem;

SubMesh SubMesh::CreateFromDomain(const Mesh &parent,
                                  Array<int> domain_attributes)
{
   return SubMesh(parent, From::Domain, domain_attributes);
}

SubMesh SubMesh::CreateFromBoundary(const Mesh &parent,
                                    Array<int> boundary_attributes)
{
   return SubMesh(parent, From::Boundary, boundary_attributes);
}

SubMesh::SubMesh(const Mesh &parent, From from,
                 Array<int> attributes) : parent_(parent), from_(from), attributes_(attributes)
{
   if (from == From::Domain)
   {
      InitMesh(parent.Dimension(), parent.SpaceDimension(), 0, 0, 0);

      std::tie(parent_vertex_ids_,
               parent_element_ids_) = SubMeshUtils::AddElementsToMesh(parent_, *this,
                                                                      attributes_);
   }
   else if (from == From::Boundary)
   {
      InitMesh(parent.Dimension() - 1, parent.SpaceDimension(), 0, 0, 0);

      std::tie(parent_vertex_ids_,
               parent_element_ids_) = SubMeshUtils::AddElementsToMesh(parent_, *this,
                                                                      attributes_, true);
   }

   FinalizeTopology(true);

   if (Dim == 3)
   {
      parent_face_ids_ = SubMeshUtils::BuildFaceMap(parent, *this,
                                                    parent_element_ids_);

      Array<int> parent_face_to_be = parent.GetFaceToBdrElMap();

      for (int i = 0; i < NumOfBdrElements; i++)
      {
         int pbeid = parent_face_to_be[parent_face_ids_[GetBdrFace(i)]];
         if (pbeid != -1)
         {
            int attr = parent.GetBdrElement(pbeid)->GetAttribute();
            GetBdrElement(i)->SetAttribute(attr);
         }
         else
         {
            // This case happens when a domain is extracted, but the root parent
            // mesh didn't have a boundary element on the surface that defined
            // it's boundary. It still creates a valid mesh, so we allow it.
            GetBdrElement(i)->SetAttribute(GENERATED_ATTRIBUTE);
         }
      }
   }

   // If the parent Mesh has nodes and therefore is defined on a higher order
   // geometry, we define this SubMesh as a curved Mesh and transfer the
   // GridFunction from the parent Mesh to the SubMesh.
   const GridFunction *parent_nodes = parent.GetNodes();
   if (parent_nodes)
   {
      const FiniteElementSpace *parent_fes = parent_nodes->FESpace();

      SetCurvature(
         parent_fes->FEColl()->GetOrder(),
         parent_fes->IsDGSpace(),
         spaceDim,
         parent_fes->GetOrdering());

      Transfer(*parent.GetNodes(), *GetNodes());
   }

   SetAttributes();
   Finalize();
}

SubMesh::~SubMesh() {}

void SubMesh::Transfer(const GridFunction &src, GridFunction &dst)
{
   Array<int> src_vdofs;
   Array<int> dst_vdofs;
   Vector vec;

   if (IsSubMesh(src.FESpace()->GetMesh()) && !IsSubMesh(dst.FESpace()->GetMesh()))
   {
      // SubMesh to Mesh transfer
      SubMesh *src_mesh = static_cast<SubMesh *>(src.FESpace()->GetMesh());

      MFEM_ASSERT(src_mesh->GetParent() == dst.FESpace()->GetMesh(),
                  "The Meshes of the specified GridFunction are not related in a SubMesh -> Mesh relationship.");

      auto &parent_element_ids = src_mesh->GetParentElementIDMap();

      for (int i = 0; i < src_mesh->GetNE(); i++)
      {
         src.FESpace()->GetElementVDofs(i, src_vdofs);
         if (src.FESpace()->IsDGSpace() && src_mesh->GetFrom() == From::Boundary)
         {
            MFEM_ABORT("Transferring from a surface SubMesh to a volume Mesh using L2 spaces is not implemented.");
         }
         else
         {
            if (src_mesh->GetFrom() == From::Domain)
            {
               dst.FESpace()->GetElementVDofs(parent_element_ids[i], dst_vdofs);
            }
            else if (src_mesh->GetFrom() == From::Boundary)
            {
               dst.FESpace()->GetBdrElementVDofs(parent_element_ids[i], dst_vdofs);
            }
            src.GetSubVector(src_vdofs, vec);
            dst.SetSubVector(dst_vdofs, vec);
         }
      }
   }
   else if (IsSubMesh(dst.FESpace()->GetMesh()))
   {
      Mesh *src_mesh = src.FESpace()->GetMesh();
      SubMesh *dst_mesh = static_cast<SubMesh *>(dst.FESpace()->GetMesh());

      // Check if there is an immediate relation
      if (dst_mesh->GetParent() == src_mesh)
      {
         auto &parent_element_ids = dst_mesh->GetParentElementIDMap();

         IntegrationPointTransformation Tr;
         DenseMatrix vals, vals_transpose;
         for (int i = 0; i < dst_mesh->GetNE(); i++)
         {
            dst.FESpace()->GetElementVDofs(i, dst_vdofs);
            if (src.FESpace()->IsDGSpace() && dst_mesh->GetFrom() == From::Boundary)
            {
               const FiniteElement *el = dst.FESpace()->GetFE(i);
               MFEM_VERIFY(dynamic_cast<const NodalFiniteElement *>(el),
                           "Destination FESpace must use nodal Finite Elements.");

               int face_info, parent_volel_id;
               src_mesh->GetBdrElementAdjacentElement(parent_element_ids[i], parent_volel_id,
                                                      face_info);
               src_mesh->GetLocalFaceTransformation(
                  src_mesh->GetBdrElementType(parent_element_ids[i]),
                  src_mesh->GetElementType(parent_volel_id),
                  Tr.Transf,
                  face_info);

               IntegrationRule src_el_ir(el->GetDof());
               Tr.Transf.ElementNo = parent_volel_id;
               Tr.Transf.ElementType = ElementTransformation::ELEMENT;
               Tr.Transform(el->GetNodes(), src_el_ir);

               src.GetVectorValues(Tr.Transf, src_el_ir, vals);
               // vals_transpose = vals^T
               vals_transpose.Transpose(vals);
               dst.SetSubVector(dst_vdofs, vals_transpose.GetData());
            }
            else
            {
               if (dst_mesh->GetFrom() == From::Domain)
               {
                  src.FESpace()->GetElementVDofs(parent_element_ids[i], src_vdofs);
               }
               else if (dst_mesh->GetFrom() == From::Boundary)
               {
                  src.FESpace()->GetBdrElementVDofs(parent_element_ids[i], src_vdofs);
               }
               src.GetSubVector(src_vdofs, vec);
               dst.SetSubVector(dst_vdofs, vec);
            }
         }
      }
      else if (IsSubMesh(src.FESpace()->GetMesh()))
      {
         SubMesh* src_sm = static_cast<SubMesh*>(src.FESpace()->GetMesh());
         SubMesh* dst_sm = static_cast<SubMesh*>(dst.FESpace()->GetMesh());
         // There is no immediate relation and both src and dst come from a
         // SubMesh, check if they have an equivalent root parent.
         if (SubMeshUtils::GetRootParent<SubMesh, Mesh>(*src_sm) !=
             SubMeshUtils::GetRootParent<SubMesh, Mesh>(*dst_sm))
         {
            MFEM_ABORT("Can't find a relation between the two GridFunctions");
         }

         if (src_sm->GetFrom() == From::Domain &&
             dst_sm->GetFrom() == From::Boundary)
         {
            const Array<int> *src_parent_fids = nullptr, *dst_parent_fids = nullptr;

            src_parent_fids = &src_sm->GetParentFaceIDMap();
            dst_parent_fids = &dst_sm->GetParentElementIDMap();

            const auto& src_parent_vids = src_sm->GetParentVertexIDMap();
            const auto& dst_parent_vids = dst_sm->GetParentVertexIDMap();

            Array<int> src_v, dst_v, src_to_parent_v, dst_to_parent_v,
                  dst_vdofs_reordered;

            for (int i = 0; i < dst_sm->GetNE(); i++)
            {
               int parent_fid = dst_sm->GetParent()->GetBdrElementEdgeIndex(
                                   (*dst_parent_fids)[i]);

               int src_fid = src_parent_fids->Find(parent_fid);

               src.FESpace()->GetFaceVDofs(src_fid, src_vdofs);
               dst.FESpace()->GetElementVDofs(i, dst_vdofs);

               // Take care of possible rotation of face/element vertices
               src_sm->GetFaceVertices(src_fid, src_v);
               dst_sm->GetElementVertices(i, dst_v);

               int nv = src_v.Size();
               src_to_parent_v.SetSize(nv);
               dst_to_parent_v.SetSize(nv);
               for (int j = 0; j < nv; j++)
               {
                  src_to_parent_v[j] = src_parent_vids[src_v[j]];
                  dst_to_parent_v[j] = dst_parent_vids[dst_v[j]];
               }

               int dst_relto_src_orientation = 0;
               if (dst_sm->GetElementGeometry(i) == Geometry::SQUARE)
               {
                  dst_relto_src_orientation = Mesh::GetQuadOrientation(src_to_parent_v,
                                                                       dst_to_parent_v);
               }
               else if (dst_sm->GetElementGeometry(i) == Geometry::TRIANGLE)
               {
                  dst_relto_src_orientation = Mesh::GetTriOrientation(src_to_parent_v,
                                                                      dst_to_parent_v);
               }
               else
               {
                  MFEM_ABORT("element geometry not supported")
               }

               Array<int> dof_order;
               dst.FESpace()->FEColl()->SubDofOrder(dst_sm->GetElementGeometry(i), 2,
                                                    dst_relto_src_orientation, dof_order);

               dst_vdofs_reordered.SetSize(dst_vdofs.Size());
               for (int j = 0; j < dst_vdofs_reordered.Size(); j++)
               {
                  dst_vdofs_reordered[j] = dst_vdofs[dof_order[j]];
               }

               src.GetSubVector(src_vdofs, vec);
               dst.SetSubVector(dst_vdofs_reordered, vec);
            }
         }
         else if (src_sm->GetFrom() == From::Boundary &&
                  dst_sm->GetFrom() == From::Domain)
         {
            const Array<int> *src_parent_fids = nullptr, *dst_parent_fids = nullptr;

            src_parent_fids = &src_sm->GetParentElementIDMap();
            dst_parent_fids = &dst_sm->GetParentFaceIDMap();

            const auto& src_parent_vids = src_sm->GetParentVertexIDMap();
            const auto& dst_parent_vids = dst_sm->GetParentVertexIDMap();

            Array<int> src_v, dst_v, src_to_parent_v, dst_to_parent_v,
                  dst_vdofs_reordered;

            for (int i = 0; i < src_sm->GetNE(); i++)
            {
               int parent_fid = src_sm->GetParent()->GetBdrElementEdgeIndex(
                                   (*src_parent_fids)[i]);

               int dst_fid = dst_parent_fids->Find(parent_fid);

               src.FESpace()->GetElementVDofs(i, src_vdofs);
               dst.FESpace()->GetFaceVDofs(dst_fid, dst_vdofs);

               // Take care of possible rotation of face/element vertices
               src_sm->GetElementVertices(i, src_v);
               dst_sm->GetFaceVertices(dst_fid, dst_v);

               int nv = src_v.Size();
               src_to_parent_v.SetSize(nv);
               dst_to_parent_v.SetSize(nv);
               for (int j = 0; j < nv; j++)
               {
                  src_to_parent_v[j] = src_parent_vids[src_v[j]];
                  dst_to_parent_v[j] = dst_parent_vids[dst_v[j]];
               }

               int dst_relto_src_orientation = 0;
               if (src_sm->GetElementGeometry(i) == Geometry::SQUARE)
               {
                  dst_relto_src_orientation = Mesh::GetQuadOrientation(src_to_parent_v,
                                                                       dst_to_parent_v);
               }
               else if (src_sm->GetElementGeometry(i) == Geometry::TRIANGLE)
               {
                  dst_relto_src_orientation = Mesh::GetTriOrientation(src_to_parent_v,
                                                                      dst_to_parent_v);
               }
               else
               {
                  MFEM_ABORT("element geometry not supported")
               }

               Array<int> dof_order;
               src.FESpace()->FEColl()->SubDofOrder(src_sm->GetElementGeometry(i), 2,
                                                    dst_relto_src_orientation, dof_order);

               dst_vdofs_reordered.SetSize(dst_vdofs.Size());
               for (int j = 0; j < dst_vdofs_reordered.Size(); j++)
               {
                  dst_vdofs_reordered[j] = dst_vdofs[dof_order[j]];
               }

               src.GetSubVector(src_vdofs, vec);
               dst.SetSubVector(dst_vdofs_reordered, vec);
            }
         }
         else
         {
            MFEM_ABORT("Can't find a supported transfer between the two GridFunctions");
         }
      }
      else
      {
         MFEM_ABORT("Can't find a relation between the two GridFunctions");
      }
   }
   else
   {
      MFEM_ABORT("Trying to do a transfer between GridFunctions but none of them is defined on a SubMesh");
   }
}
