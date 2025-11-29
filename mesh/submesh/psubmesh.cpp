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

#include <iostream>
#include <unordered_set>
#include <algorithm>
#include "psubmesh.hpp"
#include "submesh_utils.hpp"
#include "../segment.hpp"

namespace mfem
{

ParSubMesh ParSubMesh::CreateFromDomain(const ParMesh &parent,
                                        Array<int> &domain_attributes)
{
   return ParSubMesh(parent, SubMesh::From::Domain, domain_attributes);
}

ParSubMesh ParSubMesh::CreateFromBoundary(const ParMesh &parent,
                                          Array<int> &boundary_attributes)
{
   return ParSubMesh(parent, SubMesh::From::Boundary, boundary_attributes);
}

ParSubMesh::ParSubMesh(const ParMesh &parent, SubMesh::From from,
                       Array<int> &attributes) : parent_(parent), from_(from), attributes_(attributes)
{
   if (Nonconforming())
   {
      MFEM_ABORT("SubMesh does not support non-conforming meshes");
   }

   MyComm = parent.GetComm();
   NRanks = parent.GetNRanks();
   MyRank = parent.GetMyRank();

   // This violation of const-ness may be justified in this instance because
   // the exchange of face neighbor information only establishes or updates
   // derived information without altering the primary mesh information,
   // i.e., the topology, geometry, or region attributes.
   const_cast<ParMesh&>(parent).ExchangeFaceNbrData();

   if (from == SubMesh::From::Domain)
   {
      InitMesh(parent.Dimension(), parent.SpaceDimension(), 0, 0, 0);

      std::tie(parent_vertex_ids_,
               parent_element_ids_) = SubMeshUtils::AddElementsToMesh(parent_, *this,
                                                                      attributes_);
   }
   else if (from == SubMesh::From::Boundary)
   {
      InitMesh(parent.Dimension() - 1, parent.SpaceDimension(), 0, 0, 0);

      std::tie(parent_vertex_ids_,
               parent_element_ids_) = SubMeshUtils::AddElementsToMesh(parent_, *this,
                                                                      attributes_, true);
   }

   // Don't let boundary elements get generated automatically. This would
   // generate boundary elements on each rank locally, which is topologically
   // wrong for the distributed SubMesh.
   FinalizeTopology(false);

   parent_to_submesh_vertex_ids_.SetSize(parent_.GetNV());
   parent_to_submesh_vertex_ids_ = -1;
   for (int i = 0; i < parent_vertex_ids_.Size(); i++)
   {
      parent_to_submesh_vertex_ids_[parent_vertex_ids_[i]] = i;
   }

   DSTable v2v(parent_.GetNV());
   parent_.GetVertexToVertexTable(v2v);
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
      parent_face_ids_ = SubMeshUtils::BuildFaceMap(parent_, *this,
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
         if (from == SubMesh::From::Boundary)
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
            int se_ori = GetQuadOrientation(par_vert, sub_par_vert);
            parent_face_ori_[i] = ComposeQuadOrientations(be_ori, se_ori);
         }
      }
   }

   ListOfIntegerSets groups;
   IntegerSet group;
   // the first group is the local one
   group.Recreate(1, &MyRank);
   groups.Insert(group);

   // Every rank containing elements of the ParSubMesh attributes now has a
   // local ParSubMesh. We have to connect the local meshes and assign global
   // boundaries correctly.

   Array<int> rhvtx;
   FindSharedVerticesRanks(rhvtx);
   AppendSharedVerticesGroups(groups, rhvtx);

   Array<int> rhe;
   FindSharedEdgesRanks(rhe);
   AppendSharedEdgesGroups(groups, rhe);

   Array<int> rhq, rht;
   if (Dim == 3)
   {
      FindSharedFacesRanks(rht, rhq);
      AppendSharedFacesGroups(groups, rht, rhq);
   }

   // Build the group communication topology
   gtopo.SetComm(MyComm);
   gtopo.Create(groups, 822);
   int ngroups = groups.Size()-1;

   int nsverts, nsedges, nstrias, nsquads;
   BuildVertexGroup(ngroups, rhvtx, nsverts);
   BuildEdgeGroup(ngroups, rhe, nsedges);
   if (Dim == 3)
   {
      BuildFaceGroup(ngroups, rht, nstrias, rhq, nsquads);
   }
   else
   {
      group_stria.MakeI(ngroups);
      group_stria.MakeJ();
      group_stria.ShiftUpI();

      group_squad.MakeI(ngroups);
      group_squad.MakeJ();
      group_squad.ShiftUpI();
   }

   BuildSharedVerticesMapping(nsverts, rhvtx);
   BuildSharedEdgesMapping(nsedges, rhe);
   if (Dim == 3)
   {
      BuildSharedFacesMapping(nstrias, rht, nsquads, rhq);
   }

   ExchangeFaceNbrData();

   // Add boundaries
   {
      const int num_codim_1 = [this]()
      {
         if (Dim == 1) { return NumOfVertices; }
         else if (Dim == 2) { return NumOfEdges; }
         else if (Dim == 3) { return NumOfFaces; }
         else { MFEM_ABORT("Invalid dimension."); return -1; }
      }();

      if (Dim == 3)
      {
         // In 3D we check for `bel_to_edge`. It shouldn't have been set
         // previously.
         delete bel_to_edge;
         bel_to_edge = nullptr;
      }

      NumOfBdrElements = 0;
      for (int i = 0; i < num_codim_1; i++)
      {
         if (GetFaceInformation(i).IsBoundary())
         {
            NumOfBdrElements++;
         }
      }

      boundary.SetSize(NumOfBdrElements);
      be_to_face.SetSize(NumOfBdrElements);
      Array<int> parent_face_to_be = parent.GetFaceToBdrElMap();
      int max_bdr_attr = parent.bdr_attributes.Max();

      for (int i = 0, j = 0; i < num_codim_1; i++)
      {
         if (GetFaceInformation(i).IsBoundary())
         {
            boundary[j] = faces[i]->Duplicate(this);
            be_to_face[j] = i;

            if (from == SubMesh::From::Domain && Dim >= 2)
            {
               int pbeid = Dim == 3 ? parent_face_to_be[parent_face_ids_[i]] :
                           parent_face_to_be[parent_edge_ids_[i]];
               if (pbeid != -1)
               {
                  boundary[j]->SetAttribute(parent.GetBdrAttribute(pbeid));
               }
               else
               {
                  boundary[j]->SetAttribute(max_bdr_attr + 1);
               }
            }
            else
            {
               boundary[j]->SetAttribute(SubMesh::GENERATED_ATTRIBUTE);
            }
            ++j;
         }
      }

      if (from == SubMesh::From::Domain && Dim >= 2)
      {
         // Search for and count interior boundary elements
         int InteriorBdrElems = 0;
         for (int i=0; i<parent.GetNBE(); i++)
         {
            const int parentFaceIdx = parent.GetBdrElementFaceIndex(i);
            const int submeshFaceIdx =
               Dim == 3 ?
               parent_to_submesh_face_ids_[parentFaceIdx] :
               parent_to_submesh_edge_ids_[parentFaceIdx];

            if (submeshFaceIdx == -1) { continue; }
            if (GetFaceInformation(submeshFaceIdx).IsBoundary()) { continue; }

            InteriorBdrElems++;
         }

         if (InteriorBdrElems > 0)
         {
            const int OldNumOfBdrElements = NumOfBdrElements;
            NumOfBdrElements += InteriorBdrElems;
            boundary.SetSize(NumOfBdrElements);
            be_to_face.SetSize(NumOfBdrElements);

            // Search for and transfer interior boundary elements
            for (int i=0, j = OldNumOfBdrElements; i<parent.GetNBE(); i++)
            {
               const int parentFaceIdx = parent.GetBdrElementFaceIndex(i);
               const int submeshFaceIdx =
                  parent_to_submesh_face_ids_[parentFaceIdx];

               if (submeshFaceIdx == -1) { continue; }
               if (GetFaceInformation(submeshFaceIdx).IsBoundary())
               { continue; }

               boundary[j] = faces[submeshFaceIdx]->Duplicate(this);
               be_to_face[j] = submeshFaceIdx;
               boundary[j]->SetAttribute(parent.GetBdrAttribute(i));

               ++j;
            }
         }
      }
   }

   if (Dim == 3)
   {
      GetElementToFaceTable();
   }

   // If the parent ParMesh has nodes and therefore is defined on a higher order
   // geometry, we define this ParSubMesh as a curved ParSubMesh and transfer
   // the GridFunction from the parent ParMesh to the ParSubMesh.
   const GridFunction *parent_nodes = parent_.GetNodes();
   if (parent_nodes)
   {
      const FiniteElementSpace *parent_fes = parent_nodes->FESpace();

      SetCurvature(
         parent_fes->FEColl()->GetOrder(),
         parent_fes->IsDGSpace(),
         spaceDim,
         parent_fes->GetOrdering());

      const ParGridFunction* pn = dynamic_cast<const ParGridFunction*>
                                  (parent_.GetNodes());
      MFEM_ASSERT(pn,
                  "Internal error. Object is supposed to be ParGridFunction.");

      ParGridFunction* n = dynamic_cast<ParGridFunction*>
                           (this->GetNodes());
      MFEM_ASSERT(n,
                  "Internal error. Object is supposed to be ParGridFunction.");

      Transfer(*pn, *n);
   }

   if (Dim > 1)
   {
      if (!el_to_edge) { el_to_edge = new Table; }
      NumOfEdges = GetElementToEdgeTable(*el_to_edge);
   }

   if (Dim > 1 && from == SubMesh::From::Domain)
   {
      // Order 0 Raviart-Thomas space will have precisely 1 DoF per face.
      // We can use this DoF to communicate boundary attribute numbers.
      RT_FECollection fec_rt(0, Dim);
      ParFiniteElementSpace parent_fes_rt(const_cast<ParMesh*>(&parent),
                                          &fec_rt);

      ParGridFunction parent_bdr_attr_gf(&parent_fes_rt);
      parent_bdr_attr_gf = 0.0;

      Array<int> vdofs;
      DofTransformation doftrans;
      int dof, faceIdx;
      real_t sign, w;

      // Copy boundary attribute numbers into local portion of a parallel
      // grid function
      parent_bdr_attr_gf.HostReadWrite(); // not modifying all entries
      for (int i=0; i<parent.GetNBE(); i++)
      {
         faceIdx = parent.GetBdrElementFaceIndex(i);
         const FaceInformation &faceInfo = parent.GetFaceInformation(faceIdx);
         parent_fes_rt.GetBdrElementDofs(i, vdofs, doftrans);
         dof = ParFiniteElementSpace::DecodeDof(vdofs[0], sign);

         // Shared interior boundary elements are not duplicated across
         // processor boundaries but ParGridFunction::ParallelAverage will
         // assume both processors contribute to the averaged DoF value. So,
         // we multiply shared boundary values by 2 so that the average
         // produces the desired value.
         w = faceInfo.IsShared() ? 2.0 : 1.0;

         // The DoF sign is needed to ensure that non-shared interior
         // boundary values sum properly rather than canceling.
         parent_bdr_attr_gf[dof] = sign * w * parent.GetBdrAttribute(i);
      }

      Vector parent_bdr_attr(parent_fes_rt.GetTrueVSize());

      // Compute the average of the attribute numbers
      parent_bdr_attr_gf.ParallelAverage(parent_bdr_attr);
      // Distribute boundary attributes to neighboring processors
      parent_bdr_attr_gf.Distribute(parent_bdr_attr);

      ParFiniteElementSpace submesh_fes_rt(this,
                                           &fec_rt);

      ParGridFunction submesh_bdr_attr_gf(&submesh_fes_rt);

      // Transfer the averaged boundary attribute values to the submesh
      auto transfer_map = ParSubMesh::CreateTransferMap(parent_bdr_attr_gf,
                                                        submesh_bdr_attr_gf);
      transfer_map.Transfer(parent_bdr_attr_gf, submesh_bdr_attr_gf);

      // Extract the boundary attribute numbers from the local portion
      // of the ParGridFunction and set the corresponding boundary element
      // attributes.
      int attr;
      for (int i=0; i<NumOfBdrElements; i++)
      {
         submesh_fes_rt.GetBdrElementDofs(i, vdofs, doftrans);
         dof = ParFiniteElementSpace::DecodeDof(vdofs[0], sign);
         attr = (int)std::round(std::abs(submesh_bdr_attr_gf[dof]));
         if (attr != 0)
         {
            SetBdrAttribute(i, attr);
         }
      }
   }

   SetAttributes();
   Finalize();
}

void ParSubMesh::FindSharedVerticesRanks(Array<int> &rhvtx)
{
   // create a GroupCommunicator on the shared vertices
   GroupCommunicator svert_comm(parent_.gtopo);
   parent_.GetSharedVertexCommunicator(svert_comm);
   // Number of shared vertices
   int nsvtx = svert_comm.GroupLDofTable().Size_of_connections();

   rhvtx.SetSize(nsvtx);
   rhvtx = 0;

   // On each rank of the group, locally determine if the shared vertex is in
   // the SubMesh.
   for (int g = 1, sv = 0; g < parent_.GetNGroups(); g++)
   {
      const int group_sz = parent_.gtopo.GetGroupSize(g);
      MFEM_VERIFY((unsigned int)group_sz <= 8*sizeof(int), // 32
                  "Group size too large. Groups with more than 32 ranks are not supported, yet.");
      const int* group_lproc = parent_.gtopo.GetGroup(g);

      const int* my_group_id_ptr = std::find(group_lproc, group_lproc+group_sz, 0);
      MFEM_ASSERT(my_group_id_ptr != group_lproc+group_sz, "internal error");

      const int my_group_id = my_group_id_ptr-group_lproc;

      for (int gv = 0; gv < parent_.GroupNVertices(g); gv++, sv++)
      {
         int plvtx = parent_.GroupVertex(g, gv);
         int submesh_vertex_id = parent_to_submesh_vertex_ids_[plvtx];
         if (submesh_vertex_id != -1)
         {
            rhvtx[sv] |= 1 << my_group_id;
         }
      }
   }

   // Compute the sum on the root rank and broadcast the result to all ranks.
   svert_comm.Reduce(rhvtx, GroupCommunicator::Sum);
   svert_comm.Bcast<int>(rhvtx, 0);
}

void ParSubMesh::FindSharedEdgesRanks(Array<int> &rhe)
{
   // create a GroupCommunicator on the shared edges
   GroupCommunicator sedge_comm(parent_.gtopo);
   parent_.GetSharedEdgeCommunicator(sedge_comm);

   int nsedges = sedge_comm.GroupLDofTable().Size_of_connections();

   // see rhvtx description
   rhe.SetSize(nsedges);
   rhe = 0;

   // On each rank of the group, locally determine if the shared edge is in
   // the SubMesh.
   for (int g = 1, se = 0; g < parent_.GetNGroups(); g++)
   {
      const int group_sz = parent_.gtopo.GetGroupSize(g);
      MFEM_VERIFY((unsigned int)group_sz <= 8*sizeof(int), // 32
                  "Group size too large. Groups with more than 32 ranks are not supported, yet.");
      const int* group_lproc = parent_.gtopo.GetGroup(g);

      const int* my_group_id_ptr = std::find(group_lproc, group_lproc+group_sz, 0);
      MFEM_ASSERT(my_group_id_ptr != group_lproc+group_sz, "internal error");

      // rank id inside this group
      const int my_group_id = my_group_id_ptr-group_lproc;

      for (int ge = 0; ge < parent_.GroupNEdges(g); ge++, se++)
      {
         int ple, o;
         parent_.GroupEdge(g, ge, ple, o);
         int submesh_edge_id = parent_to_submesh_edge_ids_[ple];
         if (submesh_edge_id != -1)
         {
            rhe[se] |= 1 << my_group_id;
         }
      }
   }

   // Compute the sum on the root rank and broadcast the result to all ranks.
   sedge_comm.Reduce(rhe, GroupCommunicator::Sum);
   sedge_comm.Bcast<int>(rhe, 0);
}

void ParSubMesh::FindSharedFacesRanks(Array<int>& rht, Array<int> &rhq)
{
   GroupCommunicator squad_comm(parent_.gtopo);
   parent_.GetSharedQuadCommunicator(squad_comm);

   int nsquad = squad_comm.GroupLDofTable().Size_of_connections();

   rhq.SetSize(nsquad);
   rhq = 0;

   for (int g = 1, sq = 0; g < parent_.GetNGroups(); g++)
   {
      for (int gq = 0; gq < parent_.GroupNQuadrilaterals(g); gq++, sq++)
      {
         // Group size of a shared face is always 2

         int plq, o;
         parent_.GroupQuadrilateral(g, gq, plq, o);
         int submesh_face_id = parent_to_submesh_face_ids_[plq];
         if (submesh_face_id != -1)
         {
            rhq[sq] = 1;
         }
      }
   }

   // Compute the sum on the root rank and broadcast the result to all ranks.
   squad_comm.Reduce(rhq, GroupCommunicator::Sum);
   squad_comm.Bcast<int>(rhq, 0);

   GroupCommunicator stria_comm(parent_.gtopo);
   parent_.GetSharedTriCommunicator(stria_comm);

   int nstria = stria_comm.GroupLDofTable().Size_of_connections();

   rht.SetSize(nstria);
   rht = 0;

   for (int g = 1, st = 0; g < parent_.GetNGroups(); g++)
   {
      for (int gt = 0; gt < parent_.GroupNTriangles(g); gt++, st++)
      {
         // Group size of a shared face is always 2

         int plt, o;
         parent_.GroupTriangle(g, gt, plt, o);
         int submesh_face_id = parent_to_submesh_face_ids_[plt];
         if (submesh_face_id != -1)
         {
            rht[st] = 1;
         }
      }
   }

   // Compute the sum on the root rank and broadcast the result to all ranks.
   stria_comm.Reduce(rht, GroupCommunicator::Sum);
   stria_comm.Bcast<int>(rht, 0);
}


void ParSubMesh::AppendSharedVerticesGroups(ListOfIntegerSets &groups,
                                            Array<int> &rhvtx)
{
   IntegerSet group;

   for (int g = 1, sv = 0; g < parent_.GetNGroups(); g++)
   {
      const int group_sz = parent_.gtopo.GetGroupSize(g);
      MFEM_VERIFY((unsigned int)group_sz <= 8*sizeof(int), // 32
                  "Group size too large. Groups with more than 32 ranks are not supported, yet.");
      const int* group_lproc = parent_.gtopo.GetGroup(g);

      const int* my_group_id_ptr = std::find(group_lproc, group_lproc+group_sz, 0);
      MFEM_ASSERT(my_group_id_ptr != group_lproc+group_sz, "internal error");

      const int my_group_id = my_group_id_ptr-group_lproc;

      for (int gv = 0; gv < parent_.GroupNVertices(g); gv++, sv++)
      {
         // Returns the parents local vertex id
         int plvtx = parent_.GroupVertex(g, gv);
         int submesh_vtx = parent_to_submesh_vertex_ids_[plvtx];

         // Reusing the `rhvtx` array as shared vertex to group array.
         if (submesh_vtx == -1)
         {
            // parent shared vertex is not in SubMesh
            rhvtx[sv] = -1;
         }
         else if (rhvtx[sv] & ~(1 << my_group_id))
         {
            // shared vertex is present on this rank and others
            MFEM_ASSERT(rhvtx[sv] & (1 << my_group_id), "error again");

            // determine which other ranks have the shared vertex
            Array<int> &ranks = group;
            ranks.SetSize(0);
            for (int i = 0; i < group_sz; i++)
            {
               if ((rhvtx[sv] >> i) & 1)
               {
                  ranks.Append(parent_.gtopo.GetNeighborRank(group_lproc[i]));
               }
            }
            MFEM_ASSERT(ranks.Size() >= 2, "internal error");

            rhvtx[sv] = groups.Insert(group) - 1;
         }
         else
         {
            // previously shared vertex is only present on this rank
            rhvtx[sv] = -1;
         }
      }
   }
}

void ParSubMesh::AppendSharedEdgesGroups(ListOfIntegerSets &groups,
                                         Array<int> &rhe)
{
   IntegerSet group;

   for (int g = 1, se = 0; g < parent_.GetNGroups(); g++)
   {
      const int group_sz = parent_.gtopo.GetGroupSize(g);
      MFEM_VERIFY((unsigned int)group_sz <= 8*sizeof(int), // 32
                  "Group size too large. Groups with more than 32 ranks are not supported, yet.");
      const int* group_lproc = parent_.gtopo.GetGroup(g);

      const int* my_group_id_ptr = std::find(group_lproc, group_lproc+group_sz, 0);
      MFEM_ASSERT(my_group_id_ptr != group_lproc+group_sz, "internal error");

      const int my_group_id = my_group_id_ptr-group_lproc;

      for (int ge = 0; ge < parent_.GroupNEdges(g); ge++, se++)
      {
         int ple, o;
         parent_.GroupEdge(g, ge, ple, o);
         int submesh_edge = parent_to_submesh_edge_ids_[ple];

         // Reusing the `rhe` array as shared edge to group array.
         if (submesh_edge == -1)
         {
            // parent shared edge is not in SubMesh
            rhe[se] = -1;
         }
         else if (rhe[se] & ~(1 << my_group_id))
         {
            // shared edge is present on this rank and others

            // determine which other ranks have the shared edge
            Array<int> &ranks = group;
            ranks.SetSize(0);
            for (int i = 0; i < group_sz; i++)
            {
               if ((rhe[se] >> i) & 1)
               {
                  ranks.Append(parent_.gtopo.GetNeighborRank(group_lproc[i]));
               }
            }
            MFEM_ASSERT(ranks.Size() >= 2, "internal error");

            rhe[se] = groups.Insert(group) - 1;
         }
         else
         {
            // previously shared edge is only present on this rank
            rhe[se] = -1;
         }
      }
   }
}

void ParSubMesh::AppendSharedFacesGroups(ListOfIntegerSets &groups,
                                         Array<int>& rht, Array<int> &rhq)
{
   IntegerSet quad_group;

   for (int g = 1, sq = 0; g < parent_.GetNGroups(); g++)
   {
      const int* group_lproc = parent_.gtopo.GetGroup(g);
      for (int gq = 0; gq < parent_.GroupNQuadrilaterals(g); gq++, sq++)
      {
         const int group_sz = parent_.gtopo.GetGroupSize(g);
         MFEM_ASSERT(group_sz == 2, "internal error");

         int plq, o;
         parent_.GroupQuadrilateral(g, gq, plq, o);
         int submesh_face_id = parent_to_submesh_face_ids_[plq];

         // Reusing the `rhq` array as shared face to group array.
         if (submesh_face_id == -1)
         {
            // parent shared face is not in SubMesh
            rhq[sq] = -1;
         }
         else if (rhq[sq] == group_sz)
         {
            // shared face is present on this rank and others

            // There can only be two ranks in this group sharing faces. Add
            // all ranks to a new communication group.
            Array<int> &ranks = quad_group;
            ranks.SetSize(0);
            ranks.Append(parent_.gtopo.GetNeighborRank(group_lproc[0]));
            ranks.Append(parent_.gtopo.GetNeighborRank(group_lproc[1]));

            rhq[sq] = groups.Insert(quad_group) - 1;
         }
         else
         {
            // previously shared edge is only present on this rank
            rhq[sq] = -1;
         }
      }
   }

   IntegerSet tria_group;

   for (int g = 1, st = 0; g < parent_.GetNGroups(); g++)
   {
      const int* group_lproc = parent_.gtopo.GetGroup(g);
      for (int gt = 0; gt < parent_.GroupNTriangles(g); gt++, st++)
      {
         const int group_sz = parent_.gtopo.GetGroupSize(g);
         MFEM_ASSERT(group_sz == 2, "internal error");

         int plt, o;
         parent_.GroupTriangle(g, gt, plt, o);
         int submesh_face_id = parent_to_submesh_face_ids_[plt];

         // Reusing the `rht` array as shared face to group array.
         if (submesh_face_id == -1)
         {
            // parent shared face is not in SubMesh
            rht[st] = -1;
         }
         else if (rht[st] == group_sz)
         {
            // shared face is present on this rank and others

            // There can only be two ranks in this group sharing faces. Add
            // all ranks to a new communication group.
            Array<int> &ranks = tria_group;
            ranks.SetSize(0);
            ranks.Append(parent_.gtopo.GetNeighborRank(group_lproc[0]));
            ranks.Append(parent_.gtopo.GetNeighborRank(group_lproc[1]));

            rht[st] = groups.Insert(tria_group) - 1;
         }
         else
         {
            // previously shared edge is only present on this rank
            rht[st] = -1;
         }
      }
   }
}

void ParSubMesh::BuildVertexGroup(int ngroups, const Array<int>& rhvtx,
                                  int& nsverts)
{
   group_svert.MakeI(ngroups);
   for (int i = 0; i < rhvtx.Size(); i++)
   {
      if (rhvtx[i] >= 0)
      {
         group_svert.AddAColumnInRow(rhvtx[i]);
      }
   }

   group_svert.MakeJ();
   nsverts = 0;
   for (int i = 0; i < rhvtx.Size(); i++)
   {
      if (rhvtx[i] >= 0)
      {
         group_svert.AddConnection(rhvtx[i], nsverts++);
      }
   }
   group_svert.ShiftUpI();
}

void ParSubMesh::BuildEdgeGroup(int ngroups, const Array<int>& rhe,
                                int& nsedges)
{
   group_sedge.MakeI(ngroups);
   for (int i = 0; i < rhe.Size(); i++)
   {
      if (rhe[i] >= 0)
      {
         group_sedge.AddAColumnInRow(rhe[i]);
      }
   }

   group_sedge.MakeJ();
   nsedges = 0;
   for (int i = 0; i < rhe.Size(); i++)
   {
      if (rhe[i] >= 0)
      {
         group_sedge.AddConnection(rhe[i], nsedges++);
      }
   }
   group_sedge.ShiftUpI();
}

void ParSubMesh::BuildFaceGroup(int ngroups, const Array<int>& rht,
                                int& nstrias, const Array<int>& rhq, int& nsquads)
{
   group_squad.MakeI(ngroups);
   for (int i = 0; i < rhq.Size(); i++)
   {
      if (rhq[i] >= 0)
      {
         group_squad.AddAColumnInRow(rhq[i]);
      }
   }

   group_squad.MakeJ();
   nsquads = 0;
   for (int i = 0; i < rhq.Size(); i++)
   {
      if (rhq[i] >= 0)
      {
         group_squad.AddConnection(rhq[i], nsquads++);
      }
   }
   group_squad.ShiftUpI();

   group_stria.MakeI(ngroups);
   for (int i = 0; i < rht.Size(); i++)
   {
      if (rht[i] >= 0)
      {
         group_stria.AddAColumnInRow(rht[i]);
      }
   }

   group_stria.MakeJ();
   nstrias = 0;
   for (int i = 0; i < rht.Size(); i++)
   {
      if (rht[i] >= 0)
      {
         group_stria.AddConnection(rht[i], nstrias++);
      }
   }
   group_stria.ShiftUpI();
}

void ParSubMesh::BuildSharedVerticesMapping(const int nsverts,
                                            const Array<int>& rhvtx)
{
   svert_lvert.Reserve(nsverts);

   for (int g = 1, sv = 0; g < parent_.GetNGroups(); g++)
   {
      for (int gv = 0; gv < parent_.GroupNVertices(g); gv++, sv++)
      {
         // Returns the parents local vertex id
         int plvtx = parent_.GroupVertex(g, gv);
         int submesh_vtx_id = parent_to_submesh_vertex_ids_[plvtx];
         if ((submesh_vtx_id == -1) || (rhvtx[sv] == -1))
         {
            // parent shared vertex is not in SubMesh or is not shared
         }
         else
         {
            svert_lvert.Append(submesh_vtx_id);
         }
      }
   }
}

void ParSubMesh::BuildSharedEdgesMapping(const int sedges_ct,
                                         const Array<int>& rhe)
{
   shared_edges.Reserve(sedges_ct);
   sedge_ledge.Reserve(sedges_ct);

   for (int g = 1, se = 0; g < parent_.GetNGroups(); g++)
   {
      for (int ge = 0; ge < parent_.GroupNEdges(g); ge++, se++)
      {
         int ple, o;
         parent_.GroupEdge(g, ge, ple, o);
         int submesh_edge_id = parent_to_submesh_edge_ids_[ple];
         if ((submesh_edge_id == -1) || rhe[se] == -1)
         {
            // parent shared edge is not in SubMesh or is not shared
         }
         else
         {
            Array<int> vert;
            parent_.GetEdgeVertices(ple, vert);
            // Swap order of vertices if orientation in parent group is -1
            int v0 = parent_to_submesh_vertex_ids_[vert[(1-o)/2]];
            int v1 = parent_to_submesh_vertex_ids_[vert[(1+o)/2]];

            // The orienation of the shared edge relative to the local edge
            // will be determined by whether v0 < v1 or v1 < v0
            shared_edges.Append(new Segment(v0, v1, 1));
            sedge_ledge.Append(submesh_edge_id);
         }
      }
   }
}

void ParSubMesh::BuildSharedFacesMapping(const int nstrias,
                                         const Array<int>& rht,
                                         const int nsquads, const Array<int>& rhq)
{
   shared_trias.Reserve(nstrias);
   shared_quads.Reserve(nsquads);
   sface_lface.Reserve(nstrias + nsquads);

   // sface_lface should list the triangular shared faces first
   // followed by the quadrilateral shared faces.

   for (int g = 1, st = 0; g < parent_.GetNGroups(); g++)
   {
      for (int gt = 0; gt < parent_.GroupNTriangles(g); gt++, st++)
      {
         int plt, o;
         parent_.GroupTriangle(g, gt, plt, o);
         int submesh_face_id = parent_to_submesh_face_ids_[plt];
         if ((submesh_face_id == -1) || rht[st] == -1)
         {
            // parent shared face is not in SubMesh or is not shared
         }
         else
         {
            Array<int> vert;

            GetFaceVertices(submesh_face_id, vert);

            int v0 = vert[0];
            int v1 = vert[1];
            int v2 = vert[2];

            // See Mesh::GetTriOrientation for info on interpretting "o"
            switch (o)
            {
               case 1:
                  std::swap(v0,v1);
                  break;
               case 3:
                  std::swap(v2,v0);
                  break;
               case 5:
                  std::swap(v1,v2);
                  break;
               default:
                  // Do nothing
                  break;
            }

            shared_trias.Append(Vert3(v0, v1, v2));
            sface_lface.Append(submesh_face_id);
         }
      }
   }

   for (int g = 1, sq = 0; g < parent_.GetNGroups(); g++)
   {
      for (int gq = 0; gq < parent_.GroupNQuadrilaterals(g); gq++, sq++)
      {
         int plq, o;
         parent_.GroupQuadrilateral(g, gq, plq, o);
         int submesh_face_id = parent_to_submesh_face_ids_[plq];
         if ((submesh_face_id == -1) || rhq[sq] == -1)
         {
            // parent shared face is not in SubMesh or is not shared
         }
         else
         {
            Array<int> vert;
            GetFaceVertices(submesh_face_id, vert);

            int v0 = vert[0];
            int v1 = vert[1];
            int v2 = vert[2];
            int v3 = vert[3];

            // See Mesh::GetQuadOrientation for info on interpretting "o"
            switch (o)
            {
               case 1:
                  std::swap(v1,v3);
                  break;
               case 3:
                  std::swap(v0,v1);
                  std::swap(v2,v3);
                  break;
               case 5:
                  std::swap(v0,v2);
                  break;
               case 7:
                  std::swap(v0,v3);
                  std::swap(v1,v2);
                  break;
               default:
                  // Do nothing
                  break;
            }

            shared_quads.Append(Vert4(v0, v1, v2, v3));
            sface_lface.Append(submesh_face_id);
         }
      }
   }
}

void ParSubMesh::Transfer(const ParGridFunction &src, ParGridFunction &dst)
{
   ParTransferMap map(src, dst);
   map.Transfer(src, dst);
}

ParTransferMap ParSubMesh::CreateTransferMap(const ParGridFunction &src,
                                             const ParGridFunction &dst)
{
   return ParTransferMap(src, dst);
}

} // namespace mfem

#endif // MFEM_USE_MPI
