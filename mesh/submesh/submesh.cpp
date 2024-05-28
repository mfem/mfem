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
                 const Array<int> &attributes) : parent_(&parent), from_(from), attributes_(attributes)
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

   parent_to_submesh_element_ids_.SetSize(from == From::Boundary ? parent.GetNBE() : parent.GetNE());
   parent_to_submesh_element_ids_ = -1;
   for (int i = 0; i < parent_element_ids_.Size(); i++)
   {
      parent_to_submesh_element_ids_[parent_element_ids_[i]] = i;
   }

   FinalizeTopology(false);

   {
      std::cout << __FILE__ << ':' << __LINE__ << std::endl;
      Array<int> verts;
      for (int e = 0; e < GetNE(); e++)
      {
         auto * elem = GetElement(e);
         elem->GetVertices(verts);

         std::cout << "Element " << e << " : ";
         for (auto x : verts)
         {
            std::cout << x << ' ';
         }
         std::cout << std::endl;
      }
      for (int v = 0; v < GetNV(); v++)
      {
         auto *vv = GetVertex(v);
         std::cout << "Vertex " << v << " : ";
         for (int i = 0; i < 3; i++)
         {
            std::cout << vv[i] << ' ';
         }
         std::cout << std::endl;
      }
   }



   if (parent.Nonconforming())
   {
      ncmesh = new NCSubMesh(*this, *parent.ncmesh, from, attributes);
      auto ncsubmesh = dynamic_cast<NCSubMesh*>(ncmesh);

      std::cout << __FILE__ << ':' << __LINE__ << std::endl;
      for (const auto & n : ncmesh->nodes)
      {
         std::cout << n.vert_index << ' ';
      }
      std::cout << std::endl;
      auto old_parent_vertex_ids = parent_vertex_ids_;

      std::unordered_map<int,int> nodeId_old_vertex;
      nodeId_old_vertex.reserve(ncmesh->vertex_nodeId.Size());
      for (int i = 0; i < ncmesh->vertex_nodeId.Size(); i++)
      {
         nodeId_old_vertex[ncmesh->vertex_nodeId[i]] = i;
      }

      InitFromNCMesh(*ncmesh);
      ncmesh->OnMeshUpdated(this);

      std::cout << __FILE__ << ':' << __LINE__ << std::endl;
      for (const auto & n : ncmesh->nodes)
      {
         std::cout << n.vert_index << ' ';
      }
      std::cout << std::endl;

      std::vector<std::pair<int, std::array<double, 3>>> old_coordinates;
      for (int i = 0; i < parent_vertex_ids_.Size(); i++)
      {
         auto *vert = parent.GetVertex(old_parent_vertex_ids[i]);
         old_coordinates.push_back({parent_vertex_ids_[i], {vert[0], vert[1], vert[2]}});
      }

      // std::cout << __FILE__ << ':' << __LINE__ << std::endl;
      // auto new_parent_vertex_ids = parent_vertex_ids_;
      // for (int i = 0; i < GetNV(); i++)
      // {
      //    auto *vert = parent.GetVertex(i);
      //    auto key = std::array<double, 3>{{vert[0], vert[1], vert[2]}};
      //    auto comp = [&](const std::pair<int, std::array<double, 3>> v){
      //       return std::abs(v.second[0] - key[0]) < 1e-12
      //       && std::abs(v.second[1] - key[1]) < 1e-12
      //       && std::abs(v.second[2] - key[2]) < 1e-12;
      //       };
      //    auto it = std::find_if(old_coordinates.begin(), old_coordinates.end(), comp);

      //    new_parent_vertex_ids[i] = (it != old_coordinates.end()) ? it->first : -1;
      //    std::cout << i << " -> " << it->first << std::endl;
      // }
      // parent_vertex_ids_ = new_parent_vertex_ids;


      // Update the submesh to parent vertex mapping, NCSubMesh reordered the vertices so
      // the map to parent is no longer valid.
      const auto &new_to_old_vertex = ncmesh->vertex_nodeId; // newmesh vert -> oldmesh vert
      auto new_parent_vertex_ids = parent_vertex_ids_;
      for (int i = 0; i < new_parent_vertex_ids.Size(); i++)
      {
         // auto j = ncmesh->vertex_nodeId[i];
         // auto k = nodeId_old_vertex[j];
         // auto l = old_parent_vertex_ids[k];
         // new_parent_vertex_ids[i] = l;

         new_parent_vertex_ids[i] = parent_vertex_ids_[new_to_old_vertex[i]];
      }
      parent_vertex_ids_ = new_parent_vertex_ids;

      auto alt_parent_vertex_ids = parent_vertex_ids_;
      for (int i = 0; i < parent_vertex_ids_.Size(); i++)
      {
         // vertex -> node -> parent node -> parent vertex
         auto node = ncmesh->vertex_nodeId[i];
         auto parent_node = ncsubmesh->parent_node_ids_[node];
         auto parent_vertex = parent.ncmesh->nodes[parent_node].vert_index;
         std::cout << i << " node " << node << " parent_node " << parent_node << " parent_vertex " << parent_vertex << std::endl;
         alt_parent_vertex_ids[i] = parent_vertex;
      }
      parent_vertex_ids_ = alt_parent_vertex_ids;




      std::cout << __FILE__ << ':' << __LINE__ << " v ";
      for (const auto & v: parent_vertex_ids_)
      {
         std::cout << v << ' ';
      }
      std::cout << std::endl;

      parent_vertex_ids_ = new_parent_vertex_ids;


      std::cout << __FILE__ << ':' << __LINE__ << " v ";
      for (const auto & v: parent_vertex_ids_)
      {
         std::cout << v << ' ';
      }
      std::cout << std::endl;

      if (from == From::Domain)
      {
         std::set<int> remaining_elements;
         std::cout << "parent elements :";
         for (auto x : parent_element_ids_)
         {
            remaining_elements.insert(x);
            std::cout << x << ' ';
         }
         std::cout << '\n';

         auto candidate_parent_element_ids = parent_element_ids_;
         candidate_parent_element_ids = -1;
         for (int i = 0; i < parent_element_ids_.Size(); i++)
         {
            auto * elem = GetElement(i);
            auto * vert = elem->GetVertices();
            for (auto j : remaining_elements)
            {
               auto * pelem = parent.GetElement(j);
               auto * pvert = pelem->GetVertices();
               bool match = true;
               for (int v = 0; v < elem->GetNVertices(); v++)
               {
                  match &= (pvert[v] == parent_vertex_ids_[vert[v]]);
            }
               if (match) // all vertices match, found
               {
                  remaining_elements.erase(j);
                  candidate_parent_element_ids[i] = j;
                  break;
               }
            }
         }

         std::cout << __FILE__ << ':' << __LINE__ << std::endl;
         std::cout << "Brute force calculated solution\n";
         for (auto x : candidate_parent_element_ids)
         {
            std::cout << x << ' ';
         }
         std::cout << "\n";
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

      Array<int> parent_face_to_be = parent.GetFaceToBdrElMap();
      int max_bdr_attr = parent.bdr_attributes.Max();

      for (int i = 0; i < NumOfBdrElements; i++)
      {
         int pbeid = parent_face_to_be[parent_face_ids_[GetBdrElementFaceIndex(i)]];
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
            GetBdrElement(i)->SetAttribute(max_bdr_attr + 1);
         }
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
         int max_bdr_attr = parent.bdr_attributes.Max();

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
               // This case happens when a domain is extracted, but the root parent
               // mesh didn't have a boundary element on the surface that defined
               // it's boundary. It still creates a valid mesh, so we allow it.
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

   {
      Array<int> verts;
      for (int e = 0; e < GetNE(); e++)
      {
         auto * elem = GetElement(e);
         elem->GetVertices(verts);

         std::cout << "Element " << e << " : ";
         for (auto x : verts)
         {
            std::cout << x << ' ';
         }
         std::cout << std::endl;
      }
      for (int v = 0; v < GetNV(); v++)
      {
         auto *vv = GetVertex(v);
         std::cout << "Vertex " << v << " : ";
         for (int i = 0; i < 3; i++)
         {
            std::cout << vv[i] << ' ';
         }
         std::cout << std::endl;
      }

      std::cout << __FILE__ << ':' << __LINE__ << std::endl;
      if (ncmesh)
      {
         for (int f : ncmesh->boundary_faces)
         {
            std::cout << "f " << f << " attribute " << faces[f]->GetAttribute() << std::endl;
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
