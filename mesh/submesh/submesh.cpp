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

#include "submesh.hpp"
#include "submesh_utils.hpp"
#include "../../fem/gridfunc.hpp"
#include "../ncmesh.hpp"
#include "ncsubmesh.hpp"

namespace mfem
{

SubMesh SubMesh::CreateFromDomain(const Mesh &parent,
                                  const Array<int> &domain_attributes)
{
   return SubMesh(parent, From::Domain, domain_attributes);
}

SubMesh SubMesh::CreateFromBoundary(const Mesh &parent,
                                    const Array<int> &boundary_attributes)
{
   return SubMesh(parent, From::Boundary, boundary_attributes);
}

SubMesh::SubMesh(const Mesh &parent, From from,
                 const Array<int> &attributes) : parent_(&parent), from_(from),
   attributes_(attributes)
{
   if (from == From::Domain)
   {
      InitMesh(parent.Dimension(), parent.SpaceDimension(), 0, 0, 0);

      std::tie(parent_vertex_ids_,
               parent_element_ids_) = SubMeshUtils::AddElementsToMesh(parent, *this,
                                                                      attributes_);
   }
   else if (from == From::Boundary)
   {
      InitMesh(parent.Dimension() - 1, parent.SpaceDimension(), 0, 0, 0);

      std::tie(parent_vertex_ids_,
               parent_element_ids_) = SubMeshUtils::AddElementsToMesh(parent, *this,
                                                                      attributes_, true);
   }

   parent_to_submesh_vertex_ids_.SetSize(parent.GetNV());
   parent_to_submesh_vertex_ids_ = -1;
   for (int i = 0; i < parent_vertex_ids_.Size(); i++)
   {
      parent_to_submesh_vertex_ids_[parent_vertex_ids_[i]] = i;
   }

   parent_to_submesh_element_ids_.SetSize(from == From::Boundary ? parent.GetNBE()
                                          : parent.GetNE());
   parent_to_submesh_element_ids_ = -1;
   for (int i = 0; i < parent_element_ids_.Size(); i++)
   {
      parent_to_submesh_element_ids_[parent_element_ids_[i]] = i;
   }

   FinalizeTopology(false);

   if (parent.Nonconforming())
   {
      ncmesh = new NCSubMesh(*this, *parent.ncmesh, from, attributes);
      ncsubmesh_ = dynamic_cast<NCSubMesh*>(ncmesh);
      InitFromNCMesh(*ncsubmesh_);
      ncsubmesh_->OnMeshUpdated(this);

      // Update the submesh to parent vertex mapping, ncsubmesh_ reordered the
      // vertices so the map to parent is no longer valid.
      parent_to_submesh_vertex_ids_ = -1;
      for (int i = 0; i < parent_vertex_ids_.Size(); i++)
      {
         // vertex -> node -> parent node -> parent vertex
         auto node = ncsubmesh_->vertex_nodeId[i];
         auto parent_node = ncsubmesh_->parent_node_ids_[node];
         auto parent_vertex = parent.ncmesh->GetNodeVertex(parent_node);
         parent_vertex_ids_[i] = parent_vertex;
         parent_to_submesh_vertex_ids_[parent_vertex] = i;
      }
      GenerateNCFaceInfo();
      SetAttributes();
   }

   DSTable v2v(parent_->GetNV());
   parent_->GetVertexToVertexTable(v2v);
   for (int i = 0; i < NumOfEdges; i++)
   {
      Array<int> lv;
      GetEdgeVertices(i, lv);

      // Find vertices/edge in parent mesh
      int parent_edge_id = v2v(parent_vertex_ids_[lv[0]],
                               parent_vertex_ids_[lv[1]]);
      parent_edge_ids_.Append(parent_edge_id);
   }

   parent_to_submesh_edge_ids_.SetSize(parent.GetNEdges());
   parent_to_submesh_edge_ids_ = -1;
   for (int i = 0; i < parent_edge_ids_.Size(); i++)
   {
      parent_to_submesh_edge_ids_[parent_edge_ids_[i]] = i;
   }

   if (Dim == 3)
   {
      parent_face_ids_ = SubMeshUtils::BuildFaceMap(parent, *this,
                                                    parent_element_ids_);

      parent_to_submesh_face_ids_.SetSize(parent.GetNFaces());
      parent_to_submesh_face_ids_ = -1;
      for (int i = 0; i < parent_face_ids_.Size(); i++)
      {
         parent_to_submesh_face_ids_[parent_face_ids_[i]] = i;
      }

      parent_face_ori_.SetSize(NumOfFaces);
      for (int i = 0; i < NumOfFaces; i++)
      {
         Array<int> sub_vert;
         GetFaceVertices(i, sub_vert);

         Array<int> sub_par_vert(sub_vert.Size());
         for (int j = 0; j < sub_vert.Size(); j++)
         {
            sub_par_vert[j] = parent_vertex_ids_[sub_vert[j]];
         }

         Array<int> par_vert;
         parent.GetFaceVertices(parent_face_ids_[i], par_vert);
         if (par_vert.Size() == 3)
         {
            parent_face_ori_[i] = GetTriOrientation(par_vert, sub_par_vert);
         }
         else
         {
            parent_face_ori_[i] = GetQuadOrientation(par_vert, sub_par_vert);
         }
      }
   }
   else if (Dim == 2)
   {
      if (from == From::Domain)
      {
         parent_edge_ids_ = SubMeshUtils::BuildFaceMap(parent, *this,
                                                       parent_element_ids_);

         parent_to_submesh_edge_ids_.SetSize(parent.GetNEdges());
         parent_to_submesh_edge_ids_ = -1;
         for (int i = 0; i < parent_edge_ids_.Size(); i++)
         {
            parent_to_submesh_edge_ids_[parent_edge_ids_[i]] = i;
         }

         Array<int> parent_face_to_be = parent.GetFaceToBdrElMap();
         int max_bdr_attr = parent.bdr_attributes.Size() ?
                            parent.bdr_attributes.Max() : 1;

         for (int i = 0; i < NumOfBdrElements; i++)
         {
            int pbeid = parent_face_to_be[parent_edge_ids_[GetBdrElementFaceIndex(i)]];
            if (pbeid != -1)
            {
               int attr = parent.GetBdrElement(pbeid)->GetAttribute();
               GetBdrElement(i)->SetAttribute(attr);
            }
            else
            {
               // This case happens when a domain is extracted, but the root
               // parent mesh didn't have a boundary element on the surface that
               // defined it's boundary. It still creates a valid mesh, so we
               // allow it.
               GetBdrElement(i)->SetAttribute(max_bdr_attr + 1);
            }
         }
      }

      parent_face_ori_.SetSize(NumOfElements);

      for (int i = 0; i < NumOfElements; i++)
      {
         Array<int> sub_vert;
         GetElementVertices(i, sub_vert);

         Array<int> sub_par_vert(sub_vert.Size());
         for (int j = 0; j < sub_vert.Size(); j++)
         {
            sub_par_vert[j] = parent_vertex_ids_[sub_vert[j]];
         }

         Array<int> par_vert;
         int be_ori = 0;
         if (from == From::Boundary)
         {
            parent.GetBdrElementVertices(parent_element_ids_[i], par_vert);

            int f = -1;
            parent.GetBdrElementFace(parent_element_ids_[i], &f, &be_ori);
         }
         else
         {
            parent.GetElementVertices(parent_element_ids_[i], par_vert);
         }

         if (par_vert.Size() == 3)
         {
            int se_ori = GetTriOrientation(par_vert, sub_par_vert);
            parent_face_ori_[i] = ComposeTriOrientations(be_ori, se_ori);
         }
         else
         {
            parent_face_ori_[i] = GetQuadOrientation(par_vert, sub_par_vert);
         }
      }
   }

   SubMeshUtils::AddBoundaryElements(*this);

   if (Dim > 1)
   {
      delete el_to_edge;
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge);
   }
   if (Dim > 2)
   {
      GetElementToFaceTable();
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

void SubMesh::Transfer(const GridFunction &src, GridFunction &dst)
{
   CreateTransferMap(src, dst).Transfer(src, dst);
}

TransferMap SubMesh::CreateTransferMap(const GridFunction &src,
                                       const GridFunction &dst)
{
   return TransferMap(src, dst);
}

} // namespace mfem
