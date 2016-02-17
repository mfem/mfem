// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "fem.hpp"
#include "../general/sort_pairs.hpp"

#include <climits>
#include <list>

namespace mfem
{

ParFiniteElementSpace::ParFiniteElementSpace(ParFiniteElementSpace &pf)
   : FiniteElementSpace(pf)
{
   MyComm = pf.MyComm;
   NRanks = pf.NRanks;
   MyRank = pf.MyRank;
   pmesh = pf.pmesh;
   gcomm = pf.gcomm;
   pf.gcomm = NULL;
   ltdof_size = pf.ltdof_size;
   Swap(ldof_group, pf.ldof_group);
   Swap(ldof_ltdof, pf.ldof_ltdof);
   Swap(dof_offsets, pf.dof_offsets);
   Swap(tdof_offsets, pf.tdof_offsets);
   Swap(tdof_nb_offsets, pf.tdof_nb_offsets);
   Swap(ldof_sign, pf.ldof_sign);
   P = pf.P;
   pf.P = NULL;
   R = pf.R;
   pf.R = NULL;
   num_face_nbr_dofs = pf.num_face_nbr_dofs;
   pf.num_face_nbr_dofs = -1;
   Swap<Table>(face_nbr_element_dof, pf.face_nbr_element_dof);
   Swap<Table>(face_nbr_ldof, pf.face_nbr_ldof);
   Swap(face_nbr_glob_dof_map, pf.face_nbr_glob_dof_map);
   Swap<Table>(send_face_nbr_ldof, pf.send_face_nbr_ldof);
}

ParFiniteElementSpace::ParFiniteElementSpace(
   ParMesh *pm, const FiniteElementCollection *f, int dim, int ordering)
   : FiniteElementSpace(pm, f, dim, ordering)
{
   mesh = pmesh = pm;

   MyComm = pmesh->GetComm();
   MPI_Comm_size(MyComm, &NRanks);
   MPI_Comm_rank(MyComm, &MyRank);

   num_face_nbr_dofs = -1;

   P = NULL;
   R = NULL;

   if (Nonconforming())
   {
      gcomm = NULL;
      // needed here to initialize DOF offsets etc.
      GetParallelConformingInterpolation();
      return;
   }

   if (NURBSext)
   {
      if (own_ext)
      {
         // the FiniteElementSpace constructor created a serial
         // NURBSExtension of higher order than the mesh NURBSExtension

         ParNURBSExtension *pNe = new ParNURBSExtension(
            NURBSext, dynamic_cast<ParNURBSExtension *>(pmesh->NURBSext));
         // serial NURBSext is destroyed by the above constructor
         NURBSext = pNe;
         UpdateNURBS();
      }

      ConstructTrueNURBSDofs();
   }
   else
   {
      ConstructTrueDofs();
   }

   GenerateGlobalOffsets();
}

void ParFiniteElementSpace::GetGroupComm(
   GroupCommunicator &gc, int ldof_type, Array<int> *ldof_sign)
{
   int gr;
   int ng = pmesh->GetNGroups();
   int nvd, ned, nfd;
   Array<int> dofs;

   int group_ldof_counter;
   Table &group_ldof = gc.GroupLDofTable();

   nvd = fec->DofForGeometry(Geometry::POINT);
   ned = fec->DofForGeometry(Geometry::SEGMENT);
   nfd = (fdofs) ? (fdofs[1]-fdofs[0]) : (0);

   if (ldof_sign)
   {
      ldof_sign->SetSize(GetNDofs());
      *ldof_sign = 1;
   }

   // count the number of ldofs in all groups (excluding the local group 0)
   group_ldof_counter = 0;
   for (gr = 1; gr < ng; gr++)
   {
      group_ldof_counter += nvd * pmesh->GroupNVertices(gr);
      group_ldof_counter += ned * pmesh->GroupNEdges(gr);
      group_ldof_counter += nfd * pmesh->GroupNFaces(gr);
   }
   if (ldof_type)
   {
      group_ldof_counter *= vdim;
   }
   // allocate the I and J arrays in group_ldof
   group_ldof.SetDims(ng, group_ldof_counter);

   // build the full group_ldof table
   group_ldof_counter = 0;
   group_ldof.GetI()[0] = group_ldof.GetI()[1] = 0;
   for (gr = 1; gr < ng; gr++)
   {
      int j, k, l, m, o, nv, ne, nf;
      const int *ind;

      nv = pmesh->GroupNVertices(gr);
      ne = pmesh->GroupNEdges(gr);
      nf = pmesh->GroupNFaces(gr);

      // vertices
      if (nvd > 0)
      {
         for (j = 0; j < nv; j++)
         {
            k = pmesh->GroupVertex(gr, j);

            dofs.SetSize(nvd);
            m = nvd * k;
            for (l = 0; l < nvd; l++, m++)
            {
               dofs[l] = m;
            }

            if (ldof_type)
            {
               DofsToVDofs(dofs);
            }

            for (l = 0; l < dofs.Size(); l++)
            {
               group_ldof.GetJ()[group_ldof_counter++] = dofs[l];
            }
         }
      }

      // edges
      if (ned > 0)
      {
         for (j = 0; j < ne; j++)
         {
            pmesh->GroupEdge(gr, j, k, o);

            dofs.SetSize(ned);
            m = nvdofs+k*ned;
            ind = fec->DofOrderForOrientation(Geometry::SEGMENT, o);
            for (l = 0; l < ned; l++)
               if (ind[l] < 0)
               {
                  dofs[l] = m + (-1-ind[l]);
                  if (ldof_sign)
                  {
                     (*ldof_sign)[dofs[l]] = -1;
                  }
               }
               else
               {
                  dofs[l] = m + ind[l];
               }

            if (ldof_type)
            {
               DofsToVDofs(dofs);
            }

            for (l = 0; l < dofs.Size(); l++)
            {
               group_ldof.GetJ()[group_ldof_counter++] = dofs[l];
            }
         }
      }

      // faces
      if (nfd > 0)
      {
         for (j = 0; j < nf; j++)
         {
            pmesh->GroupFace(gr, j, k, o);

            dofs.SetSize(nfd);
            m = nvdofs+nedofs+fdofs[k];
            ind = fec->DofOrderForOrientation(
                     mesh->GetFaceBaseGeometry(k), o);
            for (l = 0; l < nfd; l++)
               if (ind[l] < 0)
               {
                  dofs[l] = m + (-1-ind[l]);
                  if (ldof_sign)
                  {
                     (*ldof_sign)[dofs[l]] = -1;
                  }
               }
               else
               {
                  dofs[l] = m + ind[l];
               }

            if (ldof_type)
            {
               DofsToVDofs(dofs);
            }

            for (l = 0; l < dofs.Size(); l++)
            {
               group_ldof.GetJ()[group_ldof_counter++] = dofs[l];
            }
         }
      }

      group_ldof.GetI()[gr+1] = group_ldof_counter;
   }

   gc.Finalize();
}

void ParFiniteElementSpace::ApplyLDofSigns(Array<int> &dofs) const
{
   for (int i = 0; i < dofs.Size(); i++)
   {
      if (dofs[i] < 0)
      {
         if (ldof_sign[-1-dofs[i]] < 0)
         {
            dofs[i] = -1-dofs[i];
         }
      }
      else
      {
         if (ldof_sign[dofs[i]] < 0)
         {
            dofs[i] = -1-dofs[i];
         }
      }
   }
}

void ParFiniteElementSpace::GetElementDofs(int i, Array<int> &dofs) const
{
   if (elem_dof)
   {
      elem_dof->GetRow(i, dofs);
      return;
   }
   FiniteElementSpace::GetElementDofs(i, dofs);
   if (Conforming())
   {
      ApplyLDofSigns(dofs);
   }
}

void ParFiniteElementSpace::GetBdrElementDofs(int i, Array<int> &dofs) const
{
   if (bdrElem_dof)
   {
      bdrElem_dof->GetRow(i, dofs);
      return;
   }
   FiniteElementSpace::GetBdrElementDofs(i, dofs);
   if (Conforming())
   {
      ApplyLDofSigns(dofs);
   }
}

void ParFiniteElementSpace::GetFaceDofs(int i, Array<int> &dofs) const
{
   FiniteElementSpace::GetFaceDofs(i, dofs);
   if (Conforming())
   {
      ApplyLDofSigns(dofs);
   }
}

void ParFiniteElementSpace::GenerateGlobalOffsets()
{
   if (HYPRE_AssumedPartitionCheck())
   {
      HYPRE_Int ldof[2];

      ldof[0] = GetVSize();
      ldof[1] = TrueVSize();

      dof_offsets.SetSize(3);
      tdof_offsets.SetSize(3);

      MPI_Scan(ldof, &dof_offsets[0], 2, HYPRE_MPI_INT, MPI_SUM, MyComm);

      tdof_offsets[1] = dof_offsets[1];
      tdof_offsets[0] = tdof_offsets[1] - ldof[1];

      dof_offsets[1] = dof_offsets[0];
      dof_offsets[0] = dof_offsets[1] - ldof[0];

      // get the global sizes in (t)dof_offsets[2]
      if (MyRank == NRanks-1)
      {
         ldof[0] = dof_offsets[1];
         ldof[1] = tdof_offsets[1];
      }

      MPI_Bcast(ldof, 2, HYPRE_MPI_INT, NRanks-1, MyComm);
      dof_offsets[2] = ldof[0];
      tdof_offsets[2] = ldof[1];

      // Check for overflow
      MFEM_VERIFY(dof_offsets[0] >= 0 && dof_offsets[1] >= 0,
                  "overflow in global dof_offsets");
      MFEM_VERIFY(tdof_offsets[0] >= 0 && tdof_offsets[1] >= 0,
                  "overflow in global tdof_offsets");

      if (Conforming())
      {
         // Communicate the neighbor offsets in tdof_nb_offsets
         GroupTopology &gt = GetGroupTopo();
         int nsize = gt.GetNumNeighbors()-1;
         MPI_Request *requests = new MPI_Request[2*nsize];
         MPI_Status  *statuses = new MPI_Status[2*nsize];
         tdof_nb_offsets.SetSize(nsize+1);
         tdof_nb_offsets[0] = tdof_offsets[0];

         int request_counter = 0;
         // send and receive neighbors' local tdof offsets
         for (int i = 1; i <= nsize; i++)
         {
            MPI_Irecv(&tdof_nb_offsets[i], 1, HYPRE_MPI_INT,
                      gt.GetNeighborRank(i), 5365, MyComm,
                      &requests[request_counter++]);
         }
         for (int i = 1; i <= nsize; i++)
         {
            MPI_Isend(&tdof_nb_offsets[0], 1, HYPRE_MPI_INT,
                      gt.GetNeighborRank(i), 5365, MyComm,
                      &requests[request_counter++]);
         }
         MPI_Waitall(request_counter, requests, statuses);

         delete [] statuses;
         delete [] requests;
      }
      else
      {
         // TODO
      }
   }
   else
   {
      int i;
      HYPRE_Int ldof  = GetVSize();
      HYPRE_Int ltdof = TrueVSize();

      dof_offsets.SetSize (NRanks+1);
      tdof_offsets.SetSize(NRanks+1);

      MPI_Allgather(&ldof, 1, HYPRE_MPI_INT,
                    &dof_offsets[1], 1, HYPRE_MPI_INT, MyComm);
      MPI_Allgather(&ltdof, 1, HYPRE_MPI_INT,
                    &tdof_offsets[1], 1, HYPRE_MPI_INT, MyComm);

      dof_offsets[0] = tdof_offsets[0] = 0;
      for (i = 1; i < NRanks; i++)
      {
         dof_offsets [i+1] += dof_offsets [i];
         tdof_offsets[i+1] += tdof_offsets[i];
      }

      // Check for overflow
      MFEM_VERIFY(dof_offsets[MyRank] >= 0 && dof_offsets[MyRank+1] >= 0,
                  "overflow in global dof_offsets");
      MFEM_VERIFY(tdof_offsets[MyRank] >= 0 && tdof_offsets[MyRank+1] >= 0,
                  "overflow in global tdof_offsets");
   }
}

HypreParMatrix *ParFiniteElementSpace::Dof_TrueDof_Matrix() // matrix P
{
   if (P)
   {
      return P;
   }

   if (Nonconforming())
   {
      GetParallelConformingInterpolation();
      return P;
   }

   int ldof  = GetVSize();
   int ltdof = TrueVSize();

   HYPRE_Int *i_diag = new HYPRE_Int[ldof+1];
   HYPRE_Int *j_diag = new HYPRE_Int[ltdof];
   int diag_counter;

   HYPRE_Int *i_offd = new HYPRE_Int[ldof+1];
   HYPRE_Int *j_offd = new HYPRE_Int[ldof-ltdof];
   int offd_counter;

   HYPRE_Int *cmap   = new HYPRE_Int[ldof-ltdof];

   HYPRE_Int *col_starts = GetTrueDofOffsets();
   HYPRE_Int *row_starts = GetDofOffsets();

   Array<Pair<HYPRE_Int, int> > cmap_j_offd(ldof-ltdof);

   i_diag[0] = i_offd[0] = 0;
   diag_counter = offd_counter = 0;
   for (int i = 0; i < ldof; i++)
   {
      int ltdof = GetLocalTDofNumber(i);
      if (ltdof >= 0)
      {
         j_diag[diag_counter++] = ltdof;
      }
      else
      {
         cmap_j_offd[offd_counter].one = GetGlobalTDofNumber(i);
         cmap_j_offd[offd_counter].two = offd_counter;
         offd_counter++;
      }
      i_diag[i+1] = diag_counter;
      i_offd[i+1] = offd_counter;
   }

   SortPairs<HYPRE_Int, int>(cmap_j_offd, offd_counter);

   for (int i = 0; i < offd_counter; i++)
   {
      cmap[i] = cmap_j_offd[i].one;
      j_offd[cmap_j_offd[i].two] = i;
   }

   P = new HypreParMatrix(MyComm, MyRank, NRanks, row_starts, col_starts,
                          i_diag, j_diag, i_offd, j_offd, cmap, offd_counter);

   SparseMatrix Pdiag(0, 0);
   P->GetDiag(Pdiag);
   R = Transpose(Pdiag);

   return P;
}

void ParFiniteElementSpace::DivideByGroupSize(double *vec)
{
   if (Nonconforming())
   {
      MFEM_ABORT("Not implemented for NC mesh.");
   }

   GroupTopology &gt = GetGroupTopo();

   for (int i = 0; i < ldof_group.Size(); i++)
      if (gt.IAmMaster(ldof_group[i])) // we are the master
      {
         vec[ldof_ltdof[i]] /= gt.GetGroupSize(ldof_group[i]);
      }
}

GroupCommunicator *ParFiniteElementSpace::ScalarGroupComm()
{
   if (Nonconforming())
   {
      MFEM_WARNING("Not implemented for NC mesh.");
      return NULL;
   }

   GroupCommunicator *gc = new GroupCommunicator(GetGroupTopo());
   if (NURBSext)
   {
      gc->Create(pNURBSext()->ldof_group);
   }
   else
   {
      GetGroupComm(*gc, 0);
   }
   return gc;
}

void ParFiniteElementSpace::Synchronize(Array<int> &ldof_marker) const
{
   if (Nonconforming())
   {
      MFEM_ABORT("Not implemented for NC mesh.");
   }

   if (ldof_marker.Size() != GetVSize())
   {
      mfem_error("ParFiniteElementSpace::Synchronize");
   }

   // implement allreduce(|) as reduce(|) + broadcast
   gcomm->Reduce<int>(ldof_marker, GroupCommunicator::BitOR);
   gcomm->Bcast(ldof_marker);
}

void ParFiniteElementSpace::GetEssentialVDofs(const Array<int> &bdr_attr_is_ess,
                                              Array<int> &ess_dofs) const
{
   FiniteElementSpace::GetEssentialVDofs(bdr_attr_is_ess, ess_dofs);

   if (Conforming())
   {
      // Make sure that processors without boundary elements mark
      // their boundary dofs (if they have any).
      Synchronize(ess_dofs);
   }
}

void ParFiniteElementSpace::GetEssentialTrueDofs(
   const Array<int> &bdr_attr_is_ess, Array<int> &ess_tdof_list)
{
   Array<int> ess_dofs, true_ess_dofs;

   GetEssentialVDofs(bdr_attr_is_ess, ess_dofs);
   GetRestrictionMatrix()->BooleanMult(ess_dofs, true_ess_dofs);
   MarkerToList(true_ess_dofs, ess_tdof_list);
}

int ParFiniteElementSpace::GetLocalTDofNumber(int ldof)
{
   if (Nonconforming())
   {
      MFEM_VERIFY(P, "Dof_TrueDof_Matrix() needs to be called before "
                  "GetLocalTDofNumber()");

      return ldof_ltdof[ldof]; // NOTE: contains -1 for slaves/DOFs we don't own
   }
   else
   {
      if (GetGroupTopo().IAmMaster(ldof_group[ldof]))
      {
         return ldof_ltdof[ldof];
      }
      else
      {
         return -1;
      }
   }
}

HYPRE_Int ParFiniteElementSpace::GetGlobalTDofNumber(int ldof)
{
   if (Nonconforming())
   {
      MFEM_VERIFY(ldof_ltdof[ldof] >= 0, "ldof " << ldof << " not a true DOF.");

      return GetMyTDofOffset() + ldof_ltdof[ldof];
   }
   else
   {
      if (HYPRE_AssumedPartitionCheck())
      {
         return ldof_ltdof[ldof] +
                tdof_nb_offsets[GetGroupTopo().GetGroupMaster(ldof_group[ldof])];
      }
      else
      {
         return ldof_ltdof[ldof] +
                tdof_offsets[GetGroupTopo().GetGroupMasterRank(ldof_group[ldof])];
      }
   }
}

HYPRE_Int ParFiniteElementSpace::GetGlobalScalarTDofNumber(int sldof)
{
   if (Nonconforming())
   {
      MFEM_ABORT("Not implemented for NC mesh.");
   }

   if (HYPRE_AssumedPartitionCheck())
   {
      if (ordering == Ordering::byNODES)
      {
         return ldof_ltdof[sldof] +
                tdof_nb_offsets[GetGroupTopo().GetGroupMaster(
                                   ldof_group[sldof])] / vdim;
      }
      else
      {
         return (ldof_ltdof[sldof*vdim] +
                 tdof_nb_offsets[GetGroupTopo().GetGroupMaster(
                                    ldof_group[sldof*vdim])]) / vdim;
      }
   }

   if (ordering == Ordering::byNODES)
   {
      return ldof_ltdof[sldof] +
             tdof_offsets[GetGroupTopo().GetGroupMasterRank(
                             ldof_group[sldof])] / vdim;
   }
   else
   {
      return (ldof_ltdof[sldof*vdim] +
              tdof_offsets[GetGroupTopo().GetGroupMasterRank(
                              ldof_group[sldof*vdim])]) / vdim;
   }
}

HYPRE_Int ParFiniteElementSpace::GetMyDofOffset() const
{
   return HYPRE_AssumedPartitionCheck() ? dof_offsets[0] : dof_offsets[MyRank];
}

HYPRE_Int ParFiniteElementSpace::GetMyTDofOffset() const
{
   return HYPRE_AssumedPartitionCheck()? tdof_offsets[0] : tdof_offsets[MyRank];
}

void ParFiniteElementSpace::ExchangeFaceNbrData()
{
   if (num_face_nbr_dofs >= 0) { return; }

   pmesh->ExchangeFaceNbrData();

   int num_face_nbrs = pmesh->GetNFaceNeighbors();

   if (num_face_nbrs == 0)
   {
      num_face_nbr_dofs = 0;
      return;
   }

   MPI_Request *requests = new MPI_Request[2*num_face_nbrs];
   MPI_Request *send_requests = requests;
   MPI_Request *recv_requests = requests + num_face_nbrs;
   MPI_Status  *statuses = new MPI_Status[num_face_nbrs];

   Array<int> ldofs;
   Array<int> ldof_marker(GetVSize());
   ldof_marker = -1;

   Table send_nbr_elem_dof;

   send_nbr_elem_dof.MakeI(pmesh->send_face_nbr_elements.Size_of_connections());
   send_face_nbr_ldof.MakeI(num_face_nbrs);
   face_nbr_ldof.MakeI(num_face_nbrs);
   int *send_el_off = pmesh->send_face_nbr_elements.GetI();
   int *recv_el_off = pmesh->face_nbr_elements_offset;
   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      int *my_elems = pmesh->send_face_nbr_elements.GetRow(fn);
      int  num_my_elems = pmesh->send_face_nbr_elements.RowSize(fn);

      for (int i = 0; i < num_my_elems; i++)
      {
         GetElementVDofs(my_elems[i], ldofs);
         for (int j = 0; j < ldofs.Size(); j++)
            if (ldof_marker[ldofs[j]] != fn)
            {
               ldof_marker[ldofs[j]] = fn;
               send_face_nbr_ldof.AddAColumnInRow(fn);
            }
         send_nbr_elem_dof.AddColumnsInRow(send_el_off[fn] + i, ldofs.Size());
      }

      int nbr_rank = pmesh->GetFaceNbrRank(fn);
      int tag = 0;
      MPI_Isend(&send_face_nbr_ldof.GetI()[fn], 1, MPI_INT, nbr_rank, tag,
                MyComm, &send_requests[fn]);

      MPI_Irecv(&face_nbr_ldof.GetI()[fn], 1, MPI_INT, nbr_rank, tag,
                MyComm, &recv_requests[fn]);
   }

   MPI_Waitall(num_face_nbrs, recv_requests, statuses);
   face_nbr_ldof.MakeJ();

   num_face_nbr_dofs = face_nbr_ldof.Size_of_connections();

   MPI_Waitall(num_face_nbrs, send_requests, statuses);
   send_face_nbr_ldof.MakeJ();

   // send/receive the I arrays of send_nbr_elem_dof/face_nbr_element_dof,
   // respectively (they contain the number of dofs for each face-neighbor
   // element)
   face_nbr_element_dof.MakeI(recv_el_off[num_face_nbrs]);

   int *send_I = send_nbr_elem_dof.GetI();
   int *recv_I = face_nbr_element_dof.GetI();
   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      int nbr_rank = pmesh->GetFaceNbrRank(fn);
      int tag = 0;
      MPI_Isend(send_I + send_el_off[fn], send_el_off[fn+1] - send_el_off[fn],
                MPI_INT, nbr_rank, tag, MyComm, &send_requests[fn]);

      MPI_Irecv(recv_I + recv_el_off[fn], recv_el_off[fn+1] - recv_el_off[fn],
                MPI_INT, nbr_rank, tag, MyComm, &recv_requests[fn]);
   }

   MPI_Waitall(num_face_nbrs, send_requests, statuses);
   send_nbr_elem_dof.MakeJ();

   ldof_marker = -1;

   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      int *my_elems = pmesh->send_face_nbr_elements.GetRow(fn);
      int  num_my_elems = pmesh->send_face_nbr_elements.RowSize(fn);

      for (int i = 0; i < num_my_elems; i++)
      {
         GetElementVDofs(my_elems[i], ldofs);
         for (int j = 0; j < ldofs.Size(); j++)
         {
            if (ldof_marker[ldofs[j]] != fn)
            {
               ldof_marker[ldofs[j]] = fn;
               send_face_nbr_ldof.AddConnection(fn, ldofs[j]);
            }
         }
         send_nbr_elem_dof.AddConnections(
            send_el_off[fn] + i, ldofs, ldofs.Size());
      }
   }
   send_face_nbr_ldof.ShiftUpI();
   send_nbr_elem_dof.ShiftUpI();

   // convert the ldof indices in send_nbr_elem_dof
   int *send_J = send_nbr_elem_dof.GetJ();
   for (int fn = 0, j = 0; fn < num_face_nbrs; fn++)
   {
      int  num_ldofs = send_face_nbr_ldof.RowSize(fn);
      int *ldofs     = send_face_nbr_ldof.GetRow(fn);
      int  j_end     = send_I[send_el_off[fn+1]];

      for (int i = 0; i < num_ldofs; i++)
      {
         ldof_marker[ldofs[i]] = i;
      }

      for ( ; j < j_end; j++)
      {
         send_J[j] = ldof_marker[send_J[j]];
      }
   }

   MPI_Waitall(num_face_nbrs, recv_requests, statuses);
   face_nbr_element_dof.MakeJ();

   // send/receive the J arrays of send_nbr_elem_dof/face_nbr_element_dof,
   // respectively (they contain the element dofs in enumeration local for
   // the face-neighbor pair)
   int *recv_J = face_nbr_element_dof.GetJ();
   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      int nbr_rank = pmesh->GetFaceNbrRank(fn);
      int tag = 0;

      MPI_Isend(send_J + send_I[send_el_off[fn]],
                send_I[send_el_off[fn+1]] - send_I[send_el_off[fn]],
                MPI_INT, nbr_rank, tag, MyComm, &send_requests[fn]);

      MPI_Irecv(recv_J + recv_I[recv_el_off[fn]],
                recv_I[recv_el_off[fn+1]] - recv_I[recv_el_off[fn]],
                MPI_INT, nbr_rank, tag, MyComm, &recv_requests[fn]);
   }

   MPI_Waitall(num_face_nbrs, recv_requests, statuses);

   // shift the J array of face_nbr_element_dof
   for (int fn = 0, j = 0; fn < num_face_nbrs; fn++)
   {
      int shift = face_nbr_ldof.GetI()[fn];
      int j_end = recv_I[recv_el_off[fn+1]];

      for ( ; j < j_end; j++)
      {
         recv_J[j] += shift;
      }
   }

   MPI_Waitall(num_face_nbrs, send_requests, statuses);

   // send/receive the J arrays of send_face_nbr_ldof/face_nbr_ldof,
   // respectively
   send_J = send_face_nbr_ldof.GetJ();
   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      int nbr_rank = pmesh->GetFaceNbrRank(fn);
      int tag = 0;

      MPI_Isend(send_face_nbr_ldof.GetRow(fn),
                send_face_nbr_ldof.RowSize(fn),
                MPI_INT, nbr_rank, tag, MyComm, &send_requests[fn]);

      MPI_Irecv(face_nbr_ldof.GetRow(fn),
                face_nbr_ldof.RowSize(fn),
                MPI_INT, nbr_rank, tag, MyComm, &recv_requests[fn]);
   }

   MPI_Waitall(num_face_nbrs, recv_requests, statuses);
   MPI_Waitall(num_face_nbrs, send_requests, statuses);

   // send my_dof_offset (i.e. my_ldof_offset) to face neighbors and receive
   // their offset in dof_face_nbr_offsets, used to define face_nbr_glob_dof_map
   face_nbr_glob_dof_map.SetSize(num_face_nbr_dofs);
   Array<HYPRE_Int> dof_face_nbr_offsets(num_face_nbrs);
   HYPRE_Int my_dof_offset = GetMyDofOffset();
   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      int nbr_rank = pmesh->GetFaceNbrRank(fn);
      int tag = 0;

      MPI_Isend(&my_dof_offset, 1, HYPRE_MPI_INT, nbr_rank, tag,
                MyComm, &send_requests[fn]);

      MPI_Irecv(&dof_face_nbr_offsets[fn], 1, HYPRE_MPI_INT, nbr_rank, tag,
                MyComm, &recv_requests[fn]);
   }

   MPI_Waitall(num_face_nbrs, recv_requests, statuses);

   // set the array face_nbr_glob_dof_map which holds the global ldof indices of
   // the face-neighbor dofs
   for (int fn = 0, j = 0; fn < num_face_nbrs; fn++)
   {
      for (int j_end = face_nbr_ldof.GetI()[fn+1]; j < j_end; j++)
         face_nbr_glob_dof_map[j] =
            dof_face_nbr_offsets[fn] + face_nbr_ldof.GetJ()[j];
   }

   MPI_Waitall(num_face_nbrs, send_requests, statuses);

   delete [] statuses;
   delete [] requests;
}

void ParFiniteElementSpace::GetFaceNbrElementVDofs(
   int i, Array<int> &vdofs) const
{
   face_nbr_element_dof.GetRow(i, vdofs);
}

const FiniteElement *ParFiniteElementSpace::GetFaceNbrFE(int i) const
{
   const FiniteElement *FE =
      fec->FiniteElementForGeometry(
         pmesh->face_nbr_elements[i]->GetGeometryType());

   if (NURBSext)
   {
      mfem_error("ParFiniteElementSpace::GetFaceNbrFE"
                 " does not support NURBS!");
   }
   return FE;
}

void ParFiniteElementSpace::Lose_Dof_TrueDof_Matrix()
{
   hypre_ParCSRMatrix *csrP = (hypre_ParCSRMatrix*)(*P);
   hypre_ParCSRMatrixOwnsRowStarts(csrP) = 1;
   hypre_ParCSRMatrixOwnsColStarts(csrP) = 1;
   P -> StealData();
   dof_offsets.LoseData();
   tdof_offsets.LoseData();
}

void ParFiniteElementSpace::ConstructTrueDofs()
{
   int i, gr, n = GetVSize();
   GroupTopology &gt = pmesh->gtopo;
   gcomm = new GroupCommunicator(gt);
   Table &group_ldof = gcomm->GroupLDofTable();

   GetGroupComm(*gcomm, 1, &ldof_sign);

   // Define ldof_group and mark ldof_ltdof with
   //   -1 for ldof that is ours
   //   -2 for ldof that is in a group with another master
   ldof_group.SetSize(n);
   ldof_ltdof.SetSize(n);
   ldof_group = 0;
   ldof_ltdof = -1;

   for (gr = 1; gr < group_ldof.Size(); gr++)
   {
      const int *ldofs = group_ldof.GetRow(gr);
      const int nldofs = group_ldof.RowSize(gr);
      for (i = 0; i < nldofs; i++)
      {
         ldof_group[ldofs[i]] = gr;
      }

      if (!gt.IAmMaster(gr)) // we are not the master
      {
         for (i = 0; i < nldofs; i++)
         {
            ldof_ltdof[ldofs[i]] = -2;
         }
      }
   }

   // count ltdof_size
   ltdof_size = 0;
   for (i = 0; i < n; i++)
   {
      if (ldof_ltdof[i] == -1)
      {
         ldof_ltdof[i] = ltdof_size++;
      }
   }

   // have the group masters broadcast their ltdofs to the rest of the group
   gcomm->Bcast(ldof_ltdof);
}

void ParFiniteElementSpace::ConstructTrueNURBSDofs()
{
   int n = GetVSize();
   GroupTopology &gt = pNURBSext()->gtopo;
   gcomm = new GroupCommunicator(gt);

   // pNURBSext()->ldof_group is for scalar space!
   if (vdim == 1)
   {
      ldof_group.MakeRef(pNURBSext()->ldof_group);
   }
   else
   {
      const int *scalar_ldof_group = pNURBSext()->ldof_group;
      ldof_group.SetSize(n);
      for (int i = 0; i < n; i++)
      {
         ldof_group[i] = scalar_ldof_group[VDofToDof(i)];
      }
   }

   gcomm->Create(ldof_group);

   // ldof_sign.SetSize(n);
   // ldof_sign = 1;
   ldof_sign.DeleteAll();

   ltdof_size = 0;
   ldof_ltdof.SetSize(n);
   for (int i = 0; i < n; i++)
   {
      if (gt.IAmMaster(ldof_group[i]))
      {
         ldof_ltdof[i] = ltdof_size;
         ltdof_size++;
      }
      else
      {
         ldof_ltdof[i] = -2;
      }
   }

   // have the group masters broadcast their ltdofs to the rest of the group
   gcomm->Bcast(ldof_ltdof);
}

inline int decode_dof(int dof, double& sign)
{
   return (dof >= 0) ? (sign = 1, dof) : (sign = -1, (-1 - dof));
}

const int INVALID_DOF = INT_MAX;

static void MaskSlaveDofs(Array<int> &slave_dofs, const DenseMatrix &pm,
                          const FiniteElementCollection *fec)
{
   // If a slave edge/face shares a vertex with its master, that vertex is
   // not really a slave. In serial this is easily taken care of with the
   // statement "if (mdof != sdof)" inside AddDependencies. In parallel this
   // doesn't work as 'mdof' and 'sdof' may be on different processors.
   // Here we detect these cases from the slave point matrix.

   int k, i;
   int nv = fec->DofForGeometry(Geometry::POINT);
   int ne = fec->DofForGeometry(Geometry::SEGMENT);

   if (pm.Width() == 2) // edge: exclude master endpoint vertices
   {
      if (nv)
      {
         for (i = 0; i < 2; i++)
         {
            if (pm(0, i) == 0.0 || pm(0, i) == 1.0)
            {
               for (k = 0; k < nv; k++) { slave_dofs[i*nv + k] = INVALID_DOF; }
            }
         }
      }
   }
   else // face: exclude master corner vertices + full edges if present
   {
      MFEM_ASSERT(pm.Width() == 4, "");

      bool corner[4];
      for (i = 0; i < 4; i++)
      {
         double x = pm(0,i), y = pm(1,i);
         corner[i] = ((x == 0.0 || x == 1.0) && (y == 0.0 || y == 1.0));
      }
      if (nv)
      {
         for (i = 0; i < 4; i++)
         {
            if (corner[i])
            {
               for (k = 0; k < nv; k++) { slave_dofs[i*nv + k] = INVALID_DOF; }
            }
         }
      }
      if (ne)
      {
         for (i = 0; i < 4; i++)
         {
            if (corner[i] && corner[(i+1) % 4])
            {
               for (k = 0; k < ne; k++)
               {
                  slave_dofs[4*nv + i*ne + k] = INVALID_DOF;
               }
            }
         }
      }
   }
}

void ParFiniteElementSpace
::AddSlaveDependencies(DepList deps[], int master_rank,
                       const Array<int> &master_dofs, int master_ndofs,
                       const Array<int> &slave_dofs, DenseMatrix& I)
{
   // make each slave DOF dependent on all master DOFs
   for (int i = 0; i < slave_dofs.Size(); i++)
   {
      double ss, ms;
      int sdof = decode_dof(slave_dofs[i], ss);
      if (sdof == INVALID_DOF) { continue; }

      for (int vd = 0; vd < vdim; vd++)
      {
         DepList &dl = deps[DofToVDof(sdof, vd, ndofs)];
         if (dl.type < 2) // slave dependencies override 1-to-1 dependencies
         {
            Array<Dependency> tmp_list; // TODO remove, precalculate list size
            for (int j = 0; j < master_dofs.Size(); j++)
            {
               double coef = I(i, j);
               if (std::abs(coef) > 1e-12)
               {
                  int mdof = decode_dof(master_dofs[j], ms);
                  int mvdof = DofToVDof(mdof, vd, master_ndofs);
                  tmp_list.Append(Dependency(master_rank, mvdof, coef*ms*ss));
               }
            }
            dl.type = 2;
            tmp_list.Copy(dl.list);
         }
      }
   }
}

void ParFiniteElementSpace
::Add1To1Dependencies(DepList deps[], int owner_rank,
                      const Array<int> &owner_dofs, int owner_ndofs,
                      const Array<int> &dependent_dofs)
{
   MFEM_ASSERT(owner_dofs.Size() == dependent_dofs.Size(), "");

   for (int vd = 0; vd < vdim; vd++)
   {
      for (int i = 0; i < owner_dofs.Size(); i++)
      {
         double osign, dsign;
         int odof = decode_dof(owner_dofs[i], osign);
         int ddof = decode_dof(dependent_dofs[i], dsign);
         if (odof == INVALID_DOF || ddof == INVALID_DOF) { continue; }

         int ovdof = DofToVDof(odof, vd, owner_ndofs);
         int dvdof = DofToVDof(ddof, vd, ndofs);

         DepList &dl = deps[dvdof];
         if (dl.type == 0)
         {
            dl.type = 1;
            dl.list.Append(Dependency(owner_rank, ovdof, osign*dsign));
         }
         else if (dl.type == 1 && dl.list[0].rank > owner_rank)
         {
            // 1-to-1 dependency already exists but lower rank takes precedence
            dl.list[0] = Dependency(owner_rank, ovdof, osign*dsign);
         }
      }
   }
}

void ParFiniteElementSpace
::ReorderFaceDofs(Array<int> &dofs, int orient)
{
   Array<int> tmp;
   dofs.Copy(tmp);

   // NOTE: assuming quad faces here
   int nv = fec->DofForGeometry(Geometry::POINT);
   int ne = fec->DofForGeometry(Geometry::SEGMENT);
   const int* ind = fec->DofOrderForOrientation(Geometry::SQUARE, orient);

   int ve_dofs = 4*(nv + ne);
   for (int i = 0; i < ve_dofs; i++)
   {
      dofs[i] = INVALID_DOF;
   }

   int f_dofs = dofs.Size() - ve_dofs;
   for (int i = 0; i < f_dofs; i++)
   {
      if (ind[i] >= 0)
      {
         dofs[ve_dofs + i] = tmp[ve_dofs + ind[i]];
      }
      else
      {
         dofs[ve_dofs + i] = -1 - tmp[ve_dofs + (-1 - ind[i])];
      }
   }
}

void ParFiniteElementSpace::GetDofs(int type, int index, Array<int>& dofs)
{
   switch (type)
   {
      case 0: GetVertexDofs(index, dofs); break;
      case 1: GetEdgeDofs(index, dofs); break;
      default: GetFaceDofs(index, dofs);
   }
}

void ParFiniteElementSpace::GetParallelConformingInterpolation()
{
   ParNCMesh* pncmesh = pmesh->pncmesh;

   // *** STEP 1: exchange shared vertex/edge/face DOFs with neighbors ***

   NeighborDofMessage::Map send_dofs, recv_dofs;

   // prepare neighbor DOF messages for shared vertices/edges/faces
   for (int type = 0; type < 3; type++)
   {
      const NCMesh::NCList &list = pncmesh->GetSharedList(type);
      Array<int> dofs;

      int cs = list.conforming.size(), ms = list.masters.size();
      for (int i = 0; i < cs+ms; i++)
      {
         // loop through all (shared) conforming+master vertices/edges/faces
         const NCMesh::MeshId& id =
            (i < cs) ? (const NCMesh::MeshId&) list.conforming[i]
            /*    */ : (const NCMesh::MeshId&) list.masters[i-cs];

         int owner = pncmesh->GetOwner(type, id.index), gsize;
         if (owner == MyRank)
         {
            // we own a shared v/e/f, send its DOFs to others in group
            GetDofs(type, id.index, dofs);
            // TODO: send dofs only when dofs.Size() > 0 ???
            const int *group = pncmesh->GetGroup(type, id.index, gsize);
            for (int j = 0; j < gsize; j++)
            {
               if (group[j] != MyRank)
               {
                  NeighborDofMessage &send_msg = send_dofs[group[j]];
                  send_msg.Init(pncmesh, fec, ndofs);
                  send_msg.AddDofs(type, id, dofs);
               }
            }
         }
         else
         {
            // we don't own this v/e/f and expect to receive DOFs for it
            recv_dofs[owner].Init(pncmesh, fec, 0);
         }
      }
   }

   // send/receive all DOF messages
   NeighborDofMessage::IsendAll(send_dofs, MyComm);
   NeighborDofMessage::RecvAll(recv_dofs, MyComm);

   // *** STEP 2: build dependency lists ***

   int num_dofs = ndofs * vdim;
   DepList* deps = new DepList[num_dofs]; // NOTE: 'deps' is over vdofs

   Array<int> master_dofs, slave_dofs;
   Array<int> owner_dofs, my_dofs;

   // loop through *all* master edges/faces, constrain their slaves
   for (int type = 1; type < 3; type++)
   {
      const NCMesh::NCList &list = (type > 1) ? pncmesh->GetFaceList()
                                   /*      */ : pncmesh->GetEdgeList();
      if (!list.masters.size()) { continue; }

      IsoparametricTransformation T;
      if (type > 1) { T.SetFE(&QuadrilateralFE); }
      else { T.SetFE(&SegmentFE); }

      int geom = (type > 1) ? Geometry::SQUARE : Geometry::SEGMENT;
      const FiniteElement* fe = fec->FiniteElementForGeometry(geom);
      if (!fe) { continue; }

      DenseMatrix I(fe->GetDof());

      // process masters that we own or that affect our edges/faces
      for (unsigned mi = 0; mi < list.masters.size(); mi++)
      {
         const NCMesh::Master &mf = list.masters[mi];
         if (!pncmesh->RankInGroup(type, mf.index, MyRank)) { continue; }

         // get master DOFs
         int master_ndofs, master_rank = pncmesh->GetOwner(type, mf.index);
         if (master_rank == MyRank)
         {
            GetDofs(type, mf.index, master_dofs);
            master_ndofs = ndofs;
         }
         else
         {
            recv_dofs[master_rank].GetDofs(type, mf, master_dofs, master_ndofs);
         }

         if (!master_dofs.Size()) { continue; }

         // constrain slaves that exist in our mesh
         for (int si = mf.slaves_begin; si < mf.slaves_end; si++)
         {
            const NCMesh::Slave &sf = list.slaves[si];
            if (pncmesh->IsGhost(type, sf.index)) { continue; }

            GetDofs(type, sf.index, slave_dofs);
            if (!slave_dofs.Size()) { continue; }

            T.GetPointMat() = sf.point_matrix;
            fe->GetLocalInterpolation(T, I);

            // make each slave DOF dependent on all master DOFs
            MaskSlaveDofs(slave_dofs, sf.point_matrix, fec);
            AddSlaveDependencies(deps, master_rank, master_dofs, master_ndofs,
                                 slave_dofs, I);
         }

         // special case for master edges that we don't own but still exist in
         // our mesh: this is a conforming-like situation, create 1-to-1 deps
         if (master_rank != MyRank && !pncmesh->IsGhost(type, mf.index))
         {
            GetDofs(type, mf.index, my_dofs);
            Add1To1Dependencies(deps, master_rank, master_dofs, master_ndofs,
                                my_dofs);
         }
      }
   }

   // add one-to-one dependencies between shared conforming vertices/edges/faces
   for (int type = 0; type < 3; type++)
   {
      const NCMesh::NCList &list = pncmesh->GetSharedList(type);
      for (unsigned i = 0; i < list.conforming.size(); i++)
      {
         const NCMesh::MeshId &id = list.conforming[i];
         GetDofs(type, id.index, my_dofs);
         // TODO: skip if my_dofs.Size() == 0

         int owner_ndofs, owner = pncmesh->GetOwner(type, id.index);
         if (owner != MyRank)
         {
            recv_dofs[owner].GetDofs(type, id, owner_dofs, owner_ndofs);
            if (type == 2)
            {
               int fo = pncmesh->GetFaceOrientation(id.index);
               ReorderFaceDofs(owner_dofs, fo);
            }
            Add1To1Dependencies(deps, owner, owner_dofs, owner_ndofs, my_dofs);
         }
         else
         {
            // we own this v/e/f, assert ownership of the DOFs
            Add1To1Dependencies(deps, owner, my_dofs, ndofs, my_dofs);
         }
      }
   }

   // *** STEP 3: request P matrix rows that we need from neighbors ***

   NeighborRowRequest::Map send_requests, recv_requests;

   // copy communication topology from the DOF messages
   NeighborDofMessage::Map::iterator it;
   for (it = send_dofs.begin(); it != send_dofs.end(); ++it)
   {
      recv_requests[it->first];
   }
   for (it = recv_dofs.begin(); it != recv_dofs.end(); ++it)
   {
      send_requests[it->first];
   }

   // request rows we depend on
   for (int i = 0; i < num_dofs; i++)
   {
      const DepList &dl = deps[i];
      for (int j = 0; j < dl.list.Size(); j++)
      {
         const Dependency &dep = dl.list[j];
         if (dep.rank != MyRank)
         {
            send_requests[dep.rank].RequestRow(dep.dof);
         }
      }
   }

   NeighborRowRequest::IsendAll(send_requests, MyComm);
   NeighborRowRequest::RecvAll(recv_requests, MyComm);

   // *** STEP 4: iteratively build the P matrix ***

   // DOFs that stayed independent or are ours are true DOFs
   ltdof_size = 0;
   for (int i = 0; i < num_dofs; i++)
   {
      if (deps[i].IsTrueDof(MyRank)) { ltdof_size++; }
   }

   GenerateGlobalOffsets();

   HYPRE_Int glob_true_dofs = tdof_offsets.Last();
   HYPRE_Int glob_cdofs = dof_offsets.Last();

   // create the local part (local rows) of the P matrix
#ifdef HYPRE_BIGINT
   MFEM_VERIFY(glob_true_dofs >= 0 && glob_true_dofs < (1ll << 31),
               "64bit matrix size not supported yet in non-conforming P.")
#else
   MFEM_VERIFY(glob_true_dofs >= 0,
               "overflow of non-conforming P matrix columns.")
#endif
   SparseMatrix localP(num_dofs, glob_true_dofs); // FIXME bigint

   // initialize the R matrix (also parallel but block-diagonal)
   R = new SparseMatrix(ltdof_size, num_dofs);

   Array<bool> finalized(num_dofs);
   finalized = false;

   // put identity in P and R for true DOFs, set ldof_ltdof
   HYPRE_Int my_tdof_offset = GetMyTDofOffset();
   ldof_ltdof.SetSize(num_dofs);
   ldof_ltdof = -1;
   for (int i = 0, true_dof = 0; i < num_dofs; i++)
   {
      if (deps[i].IsTrueDof(MyRank))
      {
         localP.Add(i, my_tdof_offset + true_dof, 1.0);
         R->Add(true_dof, i, 1.0);
         finalized[i] = true;
         ldof_ltdof[i] = true_dof;
         true_dof++;
      }
   }

   Array<int> cols;
   Vector srow;

   NeighborRowReply::Map recv_replies;
   std::list<NeighborRowReply::Map> send_replies;

   int num_finalized = ltdof_size;
   while (1)
   {
      // finalize all rows that can currently be finalized
      bool done;
      do
      {
         done = true;
         for (int dof = 0, i; dof < num_dofs; dof++)
         {
            if (finalized[dof]) { continue; }

            // check that rows of all constraining DOFs are available
            const DepList &dl = deps[dof];
            for (i = 0; i < dl.list.Size(); i++)
            {
               const Dependency &dep = dl.list[i];
               if (dep.rank == MyRank)
               {
                  if (!finalized[dep.dof]) { break; }
               }
               else if (!recv_replies[dep.rank].HaveRow(dep.dof))
               {
                  break;
               }
            }
            if (i < dl.list.Size()) { continue; }

            // form a linear combination of rows that 'dof' depends on
            for (i = 0; i < dl.list.Size(); i++)
            {
               const Dependency &dep = dl.list[i];
               if (dep.rank == MyRank)
               {
                  localP.GetRow(dep.dof, cols, srow);
               }
               else
               {
                  recv_replies[dep.rank].GetRow(dep.dof, cols, srow);
               }
               srow *= dep.coef;
               localP.AddRow(dof, cols, srow);
            }

            finalized[dof] = true;
            num_finalized++;
            done = false;
         }
      }
      while (!done);

      // send rows that are requested by neighbors and are available
      send_replies.push_back(NeighborRowReply::Map());

      NeighborRowRequest::Map::iterator it;
      for (it = recv_requests.begin(); it != recv_requests.end(); ++it)
      {
         NeighborRowRequest &req = it->second;
         std::set<int>::iterator row;
         for (row = req.rows.begin(); row != req.rows.end(); )
         {
            if (finalized[*row])
            {
               localP.GetRow(*row, cols, srow);
               send_replies.back()[it->first].AddRow(*row, cols, srow);
               req.rows.erase(row++);
            }
            else { ++row; }
         }
      }
      NeighborRowReply::IsendAll(send_replies.back(), MyComm);

      // are we finished?
      if (num_finalized >= num_dofs) { break; }

      // wait for a reply from neighbors
      int rank, size;
      NeighborRowReply::Probe(rank, size, MyComm);
      recv_replies[rank].Recv(rank, size, MyComm);

      // there may be more, receive all replies available
      while (NeighborRowReply::IProbe(rank, size, MyComm))
      {
         recv_replies[rank].Recv(rank, size, MyComm);
      }
   }

   delete [] deps;
   localP.Finalize();

   // create the parallel matrix P
#ifndef HYPRE_BIGINT
   P = new HypreParMatrix(MyComm, num_dofs, glob_cdofs, glob_true_dofs,
                          localP.GetI(), localP.GetJ(), localP.GetData(),
                          dof_offsets.GetData(), tdof_offsets.GetData());
#else
   (void) glob_cdofs;
   MFEM_ABORT("HYPRE_BIGINT not supported yet.");
#endif

   R->Finalize();

   // make sure we can discard all send buffers
   NeighborDofMessage::WaitAllSent(send_dofs);
   NeighborRowRequest::WaitAllSent(send_requests);

   for (std::list<NeighborRowReply::Map>::iterator
        it = send_replies.begin(); it != send_replies.end(); ++it)
   {
      NeighborRowReply::WaitAllSent(*it);
   }
}

void ParFiniteElementSpace::Update()
{
   FiniteElementSpace::Update();

   ldof_group.DeleteAll();
   ldof_ltdof.DeleteAll();
   dof_offsets.DeleteAll();
   tdof_offsets.DeleteAll();
   tdof_nb_offsets.DeleteAll();
   ldof_sign.DeleteAll();
   delete P;
   P = NULL;
   delete R;
   R = NULL;
   delete gcomm;
   gcomm = NULL;
   num_face_nbr_dofs = -1;
   face_nbr_element_dof.Clear();
   face_nbr_ldof.Clear();
   face_nbr_glob_dof_map.DeleteAll();
   send_face_nbr_ldof.Clear();
   if (Conforming())
   {
      ConstructTrueDofs();
      GenerateGlobalOffsets();
   }
   else
   {
      GetParallelConformingInterpolation();
   }
}

FiniteElementSpace *ParFiniteElementSpace::SaveUpdate()
{
   ParFiniteElementSpace *cpfes = new ParFiniteElementSpace(*this);
   Constructor();
   if (Conforming())
   {
      ConstructTrueDofs();
      GenerateGlobalOffsets();
   }
   else
   {
      GetParallelConformingInterpolation();
   }
   return cpfes;
}

}

#endif
