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
#include "pncsubmesh.hpp"
#include "submesh_utils.hpp"
#include "../segment.hpp"

namespace mfem
{

ParSubMesh ParSubMesh::CreateFromDomain(const ParMesh &parent,
                                        const Array<int> &domain_attributes)
{
   return ParSubMesh(parent, SubMesh::From::Domain, domain_attributes);
}

ParSubMesh ParSubMesh::CreateFromBoundary(const ParMesh &parent,
                                          const Array<int> &boundary_attributes)
{
   return ParSubMesh(parent, SubMesh::From::Boundary, boundary_attributes);
}

ParSubMesh::ParSubMesh(const ParMesh &parent, SubMesh::From from,
                       const Array<int> &attributes) : parent_(parent), from_(from),
   attributes_(attributes)
{
   MyComm = parent.GetComm();
   NRanks = parent.GetNRanks();
   MyRank = parent.GetMyRank();

   // This violation of const-ness may be justified in this instance because the
   // exchange of face neighbor information only establishes or updates derived
   // information without altering the primary mesh information, i.e., the
   // topology, geometry, or region attributes.
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

   parent_to_submesh_vertex_ids_.SetSize(parent_.GetNV());
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

   // Don't let boundary elements get generated automatically. This would
   // generate boundary elements on each rank locally, which is topologically
   // wrong for the distributed SubMesh.
   FinalizeTopology(false);

   if (parent.Nonconforming())
   {
      pncmesh = new ParNCSubMesh(*this, *parent.pncmesh, from, attributes);
      pncsubmesh_ = dynamic_cast<ParNCSubMesh*>(pncmesh);
      ncmesh = pncmesh;
      InitFromNCMesh(*pncmesh);
      pncmesh->OnMeshUpdated(this);

      // Update the submesh to parent vertex mapping, NCSubMesh reordered the
      // vertices so the map to parent is no longer valid.
      parent_to_submesh_vertex_ids_ = -1;
      for (int i = 0; i < parent_vertex_ids_.Size(); i++)
      {
         // vertex -> node -> parent node -> parent vertex
         auto node = pncsubmesh_->vertex_nodeId[i];
         auto parent_node = pncsubmesh_->parent_node_ids_[node];
         auto parent_vertex = parent.pncmesh->GetNodeVertex(parent_node);
         parent_vertex_ids_[i] = parent_vertex;
         parent_to_submesh_vertex_ids_[parent_vertex] = i;
      }
      GenerateNCFaceInfo();
      SetAttributes();
   }

   ReduceMeshGen();
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

   SubMeshUtils::AddBoundaryElements(*this,
                                     (from == SubMesh::From::Domain)
                                     ? FindGhostBoundaryElementAttributes()
                                     : std::unordered_map<int,int> {});

   if (Dim > 1)
   {
      if (!el_to_edge) { el_to_edge = new Table; }
      NumOfEdges = GetElementToEdgeTable(*el_to_edge);
   }
   if (Dim > 2)
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

   // On each rank of the group, locally determine if the shared edge is in the
   // SubMesh.
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
         int ple = parent_.GroupEdge(g, ge);
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
   GroupCommunicator stria_comm(parent_.gtopo);
   parent_.GetSharedTriCommunicator(stria_comm);
   int nstria = stria_comm.GroupLDofTable().Size_of_connections();
   rht.SetSize(nstria);
   rht = 0;

   for (int g = 1, st = 0; g < parent_.GetNGroups(); g++)
   {
      MFEM_ASSERT(parent_.gtopo.GetGroupSize(g) == 2
                  || parent_.GroupNTriangles(g) == 0,
                  parent_.gtopo.GetGroupSize(g) << ' ' << parent_.GroupNTriangles(g));
      for (int gt = 0; gt < parent_.GroupNTriangles(g); gt++, st++)
      {
         // Group size of a shared face is always 2
         int plt = parent_.GroupTriangle(g, gt);
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

   GroupCommunicator squad_comm(parent_.gtopo);
   parent_.GetSharedQuadCommunicator(squad_comm);
   int nsquad = squad_comm.GroupLDofTable().Size_of_connections();
   rhq.SetSize(nsquad);
   rhq = 0;

   for (int g = 1, sq = 0; g < parent_.GetNGroups(); g++)
   {
      MFEM_ASSERT(parent_.gtopo.GetGroupSize(g) == 2
                  || parent_.GroupNQuadrilaterals(g) == 0,
                  parent_.gtopo.GetGroupSize(g) << ' ' << parent_.GroupNQuadrilaterals(g));
      for (int gq = 0; gq < parent_.GroupNQuadrilaterals(g); gq++, sq++)
      {
         // Group size of a shared face is always 2
         int plq = parent_.GroupQuadrilateral(g, gq);
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
}


void ParSubMesh::AppendSharedVerticesGroups(ListOfIntegerSets &groups,
                                            Array<int> &rhvtx)
{
   IntegerSet group;

   // g = 0 corresponds to the singleton group of each rank alone.
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
         int ple = parent_.GroupEdge(g, ge);
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

         int plq = parent_.GroupQuadrilateral(g, gq);
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

            // There can only be two ranks in this group sharing faces. Add all
            // ranks to a new communication group.
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

         int plt = parent_.GroupTriangle(g, gt);
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

            // There can only be two ranks in this group sharing faces. Add all
            // ranks to a new communication group.
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

void BuildGroup(Table &group, int ngroups, const Array<int>& rh, int &ns)
{
   group.MakeI(ngroups);
   for (int i = 0; i < rh.Size(); i++)
   {
      if (rh[i] >= 0)
      {
         group.AddAColumnInRow(rh[i]);
      }
   }

   group.MakeJ();
   ns = 0;
   for (int i = 0; i < rh.Size(); i++)
   {
      if (rh[i] >= 0)
      {
         group.AddConnection(rh[i], ns++);
      }
   }
   group.ShiftUpI();
}

void ParSubMesh::BuildVertexGroup(int ngroups, const Array<int>& rhvtx,
                                  int& nsverts)
{
   BuildGroup(group_svert, ngroups, rhvtx, nsverts);
}

void ParSubMesh::BuildEdgeGroup(int ngroups, const Array<int>& rhe,
                                int& nsedges)
{
   BuildGroup(group_sedge, ngroups, rhe, nsedges);
}

void ParSubMesh::BuildFaceGroup(int ngroups, const Array<int>& rht,
                                int& nstrias, const Array<int>& rhq, int& nsquads)
{
   BuildGroup(group_squad, ngroups, rhq, nsquads);
   BuildGroup(group_stria, ngroups, rht, nstrias);
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

            // The orienation of the shared edge relative to the local edge will
            // be determined by whether v0 < v1 or v1 < v0
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

   // sface_lface should list the triangular shared faces first followed by the
   // quadrilateral shared faces.
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

            // See Mesh::GetQuadOrientation for info on interpreting "o"
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

std::unordered_map<int, int>
ParSubMesh::FindGhostBoundaryElementAttributes() const
{
   // Loop over shared faces in the parent mesh, find their attributes if they
   // exist, and map to local faces in the submesh.
   std::unordered_map<int,int> lface_boundary_attribute;
   const auto &face_to_be = parent_.GetFaceToBdrElMap();
   if (Dim == 3)
   {
      GroupCommunicator squad_comm(parent_.gtopo);
      parent_.GetSharedQuadCommunicator(squad_comm);
      int nsquad = squad_comm.GroupLDofTable().Size_of_connections();

      GroupCommunicator stria_comm(parent_.gtopo);
      parent_.GetSharedTriCommunicator(stria_comm);
      int nstria = stria_comm.GroupLDofTable().Size_of_connections();

      Array<int> stba(nstria), sqba(nsquad);
      Array<int> parent_ltface(nstria), parent_lqface(nsquad);
      stba = 0; sqba = 0;
      parent_ltface = -1; parent_lqface = -1;
      for (int g = 1, st = 0; g < parent_.GetNGroups(); g++)
      {
         for (int gt = 0; gt < parent_.GroupNTriangles(g); gt++, st++)
         {
            // Group size of a shared face is always 2
            int plt = parent_.GroupTriangle(g, gt);
            auto pbe = face_to_be[plt];
            if (pbe >= 0)
            {
               stba[st] = parent_.GetBdrAttribute(pbe);
            }
            parent_ltface[st] = plt;
         }
      }
      for (int g = 1, sq = 0; g < parent_.GetNGroups(); g++)
      {
         for (int gq = 0; gq < parent_.GroupNQuadrilaterals(g); gq++, sq++)
         {
            // Group size of a shared face is always 2
            int plq = parent_.GroupQuadrilateral(g, gq);
            auto pbe = face_to_be[plq];
            if (pbe >= 0)
            {
               sqba[sq] = parent_.GetBdrAttribute(pbe);
            }
            parent_lqface[sq] = plq;
         }
      }
#ifdef MFEM_DEBUG
      auto pre_stba = stba;
      auto pre_sqba = sqba;
#endif
      stria_comm.Reduce(stba, GroupCommunicator::Sum);
      stria_comm.Bcast<int>(stba, 0);
      squad_comm.Reduce(sqba, GroupCommunicator::Sum);
      squad_comm.Bcast<int>(sqba, 0);
#ifdef MFEM_DEBUG
      {
         Array<int> fail_indices;
         fail_indices.Reserve(stba.Size());
         for (int i = 0; i < stba.Size(); i++)
            if (pre_stba[i] != 0 && pre_stba[i] != stba[i])
            {
               fail_indices.Append(i);
            }
         MFEM_ASSERT(fail_indices.Size() == 0, [&]()
         {
            std::stringstream msg;
            msg << "More than one rank found attribute on shared tri face: ";
            for (auto x : fail_indices)
            {
               msg << x << ' ';
            }
            return msg.str();
         }());
      }

      {
         Array<int> fail_indices;
         fail_indices.Reserve(sqba.Size());
         for (int i = 0; i < sqba.Size(); i++)
            if (pre_sqba[i] != 0 && pre_sqba[i] != sqba[i])
            {
               fail_indices.Append(i);
            }
         MFEM_ASSERT(fail_indices.Size() == 0, [&]()
         {
            std::stringstream msg;
            msg << "More than one rank found attribute on shared quad face: ";
            for (auto x : fail_indices)
            {
               msg << x << ' ';
            }
            return msg.str();
         }());
      }
#endif
      int nghost = 0;
      for (auto x : stba)
         if (x > 0) { ++nghost; }

      for (auto x : sqba)
         if (x > 0) { ++nghost; }

      lface_boundary_attribute.reserve(nghost);
      for (int i = 0; i < stba.Size(); i++)
         if (stba[i] > 0)
         {
            MFEM_ASSERT(parent_ltface[i] > -1, i);
            lface_boundary_attribute[parent_ltface[i]] = stba[i];
         }
      for (int i = 0; i < sqba.Size(); i++)
         if (sqba[i] > 0)
         {
            MFEM_ASSERT(parent_lqface[i] > -1, i);
            lface_boundary_attribute[parent_lqface[i]] = sqba[i];
         }
   }
   else if (Dim == 2)
   {
      GroupCommunicator sedge_comm(parent_.gtopo);
      parent_.GetSharedEdgeCommunicator(sedge_comm);
      int nsedge = sedge_comm.GroupLDofTable().Size_of_connections();

      Array<int> seba(nsedge), parent_ledge(nsedge);
      seba = 0; parent_ledge = -1;
      for (int g = 1, se = 0; g < parent_.GetNGroups(); g++)
      {
         for (int ge = 0; ge < parent_.GroupNEdges(g); ge++, se++)
         {
            // Group size of a shared edge is always 2
            int ple = parent_.GroupEdge(g, ge);
            auto pbe = face_to_be[ple];
            if (pbe >= 0)
            {
               seba[se] = parent_.GetBdrAttribute(pbe);
            }
            parent_ledge[se] = ple;
         }
      }

#ifdef MFEM_DEBUG
      auto pre_seba = seba;
#endif
      sedge_comm.Reduce(seba, GroupCommunicator::Sum);
      sedge_comm.Bcast<int>(seba, 0);
#ifdef MFEM_DEBUG
      {
         Array<int> fail_indices;
         fail_indices.Reserve(seba.Size());
         for (int i = 0; i < seba.Size(); i++)
            if (pre_seba[i] != 0 && pre_seba[i] != seba[i])
            {
               fail_indices.Append(i);
            }
         MFEM_ASSERT(fail_indices.Size() == 0, [&]()
         {
            std::stringstream msg;
            msg << "More than one rank found attribute on shared edge: ";
            for (auto x : fail_indices)
            {
               msg << x << ' ';
            }
            return msg.str();
         }());
      }
#endif
      int nghost = 0;
      for (auto x : seba)
         if (x > 0) { ++nghost; }

      lface_boundary_attribute.reserve(nghost);
      for (int i = 0; i < seba.Size(); i++)
         if (seba[i] > 0)
         {
            MFEM_ASSERT(parent_ledge[i] > -1, i);
            lface_boundary_attribute[parent_ledge[i]] = seba[i];
         }
   }
   else if (Dim == 1)
   {
      GroupCommunicator svert_comm(parent_.gtopo);
      parent_.GetSharedVertexCommunicator(svert_comm);
      int nsvtx = svert_comm.GroupLDofTable().Size_of_connections();

      Array<int> svba(nsvtx), parent_lvtx(nsvtx);
      svba = 0; parent_lvtx = -1;
      for (int g = 1, sv = 0; g < parent_.GetNGroups(); g++)
      {
         for (int gv = 0; gv < parent_.GroupNVertices(g); gv++, sv++)
         {
            // Group size of a shared vertex is always 2
            int plv = parent_.GroupVertex(g, gv);
            auto pbe = face_to_be[plv];
            if (pbe >= 0)
            {
               svba[sv] = parent_.GetBdrAttribute(pbe);
            }
            parent_lvtx[sv] = plv;
         }
      }

#ifdef MFEM_DEBUG
      auto pre_svba = svba;
#endif
      svert_comm.Reduce(svba, GroupCommunicator::Sum);
      svert_comm.Bcast<int>(svba, 0);
#ifdef MFEM_DEBUG
      {
         Array<int> fail_indices;
         fail_indices.Reserve(svba.Size());
         for (int i = 0; i < svba.Size(); i++)
            if (pre_svba[i] != 0 && pre_svba[i] != svba[i])
            {
               fail_indices.Append(i);
            }
         MFEM_ASSERT(fail_indices.Size() == 0, [&]()
         {
            std::stringstream msg;
            msg << "More than one rank found attribute on shared vertex: ";
            for (auto x : fail_indices)
            {
               msg << x << ' ';
            }
            return msg.str();
         }());
      }
#endif
      int nghost = 0;
      for (auto x : svba)
         if (x > 0) { ++nghost; }

      lface_boundary_attribute.reserve(nghost);
      for (int i = 0; i < svba.Size(); i++)
         if (svba[i] > 0)
         {
            MFEM_ASSERT(parent_lvtx[i] > -1, i);
            lface_boundary_attribute[parent_lvtx[i]] = svba[i];
         }
   }
   return lface_boundary_attribute;
}


void ParSubMesh::Transfer(const ParGridFunction &src, ParGridFunction &dst)
{
   CreateTransferMap(src, dst).Transfer(src, dst);
}

ParTransferMap ParSubMesh::CreateTransferMap(const ParGridFunction &src,
                                             const ParGridFunction &dst)
{
   return ParTransferMap(src, dst);
}

} // namespace mfem

#endif // MFEM_USE_MPI
