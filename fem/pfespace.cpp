// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifdef MFEM_USE_MPI

#include "fem.hpp"
#include "../general/sort_pairs.hpp"

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
}

ParFiniteElementSpace::ParFiniteElementSpace(
   ParMesh *pm, FiniteElementCollection *f, int dim, int order)
   : FiniteElementSpace(pm, f, dim, order)
{
   mesh = pmesh = pm;

   MyComm = pmesh->GetComm();
   MPI_Comm_size(MyComm, &NRanks);
   MPI_Comm_rank(MyComm, &MyRank);

   P = NULL;

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
      group_ldof_counter *= vdim;
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
         for (j = 0; j < nv; j++)
         {
            k = pmesh->GroupVertex(gr, j);

            dofs.SetSize(nvd);
            m = nvd * k;
            for (l = 0; l < nvd; l++, m++)
               dofs[l] = m;

            if (ldof_type)
               DofsToVDofs(dofs);

            for (l = 0; l < dofs.Size(); l++)
               group_ldof.GetJ()[group_ldof_counter++] = dofs[l];
         }

      // edges
      if (ned > 0)
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
                     (*ldof_sign)[dofs[l]] = -1;
               }
               else
                  dofs[l] = m + ind[l];

            if (ldof_type)
               DofsToVDofs(dofs);

            for (l = 0; l < dofs.Size(); l++)
               group_ldof.GetJ()[group_ldof_counter++] = dofs[l];
         }

      // faces
      if (nfd > 0)
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
                     (*ldof_sign)[dofs[l]] = -1;
               }
               else
                  dofs[l] = m + ind[l];

            if (ldof_type)
               DofsToVDofs(dofs);

            for (l = 0; l < dofs.Size(); l++)
               group_ldof.GetJ()[group_ldof_counter++] = dofs[l];
         }

      group_ldof.GetI()[gr+1] = group_ldof_counter;
   }

   gc.Finalize();
}

void ParFiniteElementSpace::GetElementDofs(int i, Array<int> &dofs) const
{
   if (elem_dof)
   {
      elem_dof->GetRow(i, dofs);
      return;
   }
   FiniteElementSpace::GetElementDofs(i, dofs);
   for (i = 0; i < dofs.Size(); i++)
      if (dofs[i] < 0)
      {
         if (ldof_sign[-1-dofs[i]] < 0)
            dofs[i] = -1-dofs[i];
      }
      else
      {
         if (ldof_sign[dofs[i]] < 0)
            dofs[i] = -1-dofs[i];
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
   for (i = 0; i < dofs.Size(); i++)
      if (dofs[i] < 0)
      {
         if (ldof_sign[-1-dofs[i]] < 0)
            dofs[i] = -1-dofs[i];
      }
      else
      {
         if (ldof_sign[dofs[i]] < 0)
            dofs[i] = -1-dofs[i];
      }
}

void ParFiniteElementSpace::GenerateGlobalOffsets()
{
   if (HYPRE_AssumedPartitionCheck())
   {
      int ldof[2];

      ldof[0] = GetVSize();
      ldof[1] = TrueVSize();

      dof_offsets.SetSize(3);
      tdof_offsets.SetSize(3);

      MPI_Scan(&ldof, &dof_offsets[0], 2, MPI_INT, MPI_SUM, MyComm);

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

      MPI_Bcast(&ldof, 2, MPI_INT, NRanks-1, MyComm);
      dof_offsets[2] = ldof[0];
      tdof_offsets[2] = ldof[1];
   }
   else
   {
      int i;
      int ldof  = GetVSize();
      int ltdof = TrueVSize();

      dof_offsets.SetSize (NRanks+1);
      tdof_offsets.SetSize(NRanks+1);

      MPI_Allgather(&ldof, 1, MPI_INT, &dof_offsets[1], 1, MPI_INT, MyComm);
      MPI_Allgather(&ltdof, 1, MPI_INT, &tdof_offsets[1], 1, MPI_INT, MyComm);

      dof_offsets[0] = tdof_offsets[0] = 0;
      for (i = 1; i < NRanks; i++)
      {
         dof_offsets [i+1] += dof_offsets [i];
         tdof_offsets[i+1] += tdof_offsets[i];
      }
   }
}

HypreParMatrix *ParFiniteElementSpace::Dof_TrueDof_Matrix() // matrix P
{
   int  i;

   if (P)
      return P;

   int  ldof = GetVSize();
   int  ltdof = TrueVSize();

   GroupTopology &gt = GetGroupTopo();

   int *i_diag;
   int *j_diag;
   int  diag_counter;

   int *i_offd;
   int *j_offd;
   int  offd_counter;

   int *cmap;
   int *col_starts;
   int *row_starts;

   col_starts = GetTrueDofOffsets();
   row_starts = GetDofOffsets();

   i_diag = hypre_TAlloc(HYPRE_Int, ldof+1);
   j_diag = hypre_TAlloc(HYPRE_Int, ltdof);

   i_offd = hypre_TAlloc(HYPRE_Int, ldof+1);
   j_offd = hypre_TAlloc(HYPRE_Int, ldof-ltdof);

   cmap   = hypre_TAlloc(HYPRE_Int, ldof-ltdof);

   Array<Pair<int, int> > cmap_j_offd(ldof-ltdof);

   if (HYPRE_AssumedPartitionCheck())
   {
      int nsize = gt.GetNumNeighbors()-1;
      MPI_Request *requests = new MPI_Request[2*nsize];
      MPI_Status  *statuses = new MPI_Status[2*nsize];
      tdof_nb_offsets.SetSize(nsize+1);
      tdof_nb_offsets[0] = col_starts[0];

      int request_counter = 0;
      // send and receive neighbors' local tdof offsets
      for (i = 1; i <= nsize; i++)
         MPI_Irecv(&tdof_nb_offsets[i], 1, MPI_INT, gt.GetNeighborRank(i), 5365,
                   MyComm, &requests[request_counter++]);

      for (i = 1; i <= nsize; i++)
         MPI_Isend(&tdof_nb_offsets[0], 1, MPI_INT, gt.GetNeighborRank(i), 5365,
                   MyComm, &requests[request_counter++]);

      MPI_Waitall(request_counter, requests, statuses);

      delete [] statuses;
      delete [] requests;
   }

   i_diag[0] = i_offd[0] = 0;
   diag_counter = offd_counter = 0;
   for (i = 0; i < ldof; i++)
   {
      int proc = gt.GetGroupMasterRank(ldof_group[i]);
      if (proc == MyRank)
      {
         j_diag[diag_counter++] = ldof_ltdof[i];
      }
      else
      {
         if (HYPRE_AssumedPartitionCheck())
            cmap_j_offd[offd_counter].one =
               tdof_nb_offsets[gt.GetGroupMaster(ldof_group[i])] + ldof_ltdof[i];
         else
            cmap_j_offd[offd_counter].one = col_starts[proc] + ldof_ltdof[i];
         cmap_j_offd[offd_counter].two = offd_counter;
         offd_counter++;
      }
      i_diag[i+1] = diag_counter;
      i_offd[i+1] = offd_counter;
   }

   SortPairs<int, int>(cmap_j_offd, offd_counter);

   for (i = 0; i < offd_counter; i++)
   {
      cmap[i] = cmap_j_offd[i].one;
      j_offd[cmap_j_offd[i].two] = i;
   }

   P = new HypreParMatrix(MyComm, MyRank, NRanks, row_starts, col_starts,
                          i_diag, j_diag, i_offd, j_offd, cmap, offd_counter);

   return P;
}

void ParFiniteElementSpace::DivideByGroupSize(double *vec)
{
   GroupTopology &gt = GetGroupTopo();

   for (int i = 0; i < ldof_group.Size(); i++)
      if (gt.IAmMaster(ldof_group[i])) // we are the master
         vec[ldof_ltdof[i]] /= gt.GetGroupSize(ldof_group[i]);
}

GroupCommunicator *ParFiniteElementSpace::ScalarGroupComm()
{
   GroupCommunicator *gc = new GroupCommunicator(GetGroupTopo());
   if (NURBSext)
      gc->Create(pNURBSext()->ldof_group);
   else
      GetGroupComm(*gc, 0);
   return gc;
}

void ParFiniteElementSpace::Synchronize(Array<int> &ldof_marker)
{
   if (ldof_marker.Size() != GetVSize())
      mfem_error("ParFiniteElementSpace::Synchronize");

   // implement allreduce(|) as reduce(|) + broadcast
   gcomm->Reduce(ldof_marker);
   gcomm->Bcast(ldof_marker);
}

void ParFiniteElementSpace::GetEssentialVDofs(Array<int> &bdr_attr_is_ess,
                                              Array<int> &ess_dofs)
{
   FiniteElementSpace::GetEssentialVDofs(bdr_attr_is_ess, ess_dofs);

   // Make sure that processors without boundary elements mark
   // their boundary dofs (if they have any).
   Synchronize(ess_dofs);
}

int ParFiniteElementSpace::GetLocalTDofNumber(int ldof)
{
   if (GetGroupTopo().IAmMaster(ldof_group[ldof]))
      return ldof_ltdof[ldof];
   else
      return -1;
}

int ParFiniteElementSpace::GetGlobalTDofNumber(int ldof)
{
   if (HYPRE_AssumedPartitionCheck())
   {
      if (!P)
         Dof_TrueDof_Matrix();
      return ldof_ltdof[ldof] +
         tdof_nb_offsets[GetGroupTopo().GetGroupMaster(ldof_group[ldof])];
   }

   return ldof_ltdof[ldof] +
      tdof_offsets[GetGroupTopo().GetGroupMasterRank(ldof_group[ldof])];
}

int ParFiniteElementSpace::GetGlobalScalarTDofNumber(int sldof)
{
   if (HYPRE_AssumedPartitionCheck())
   {
      if (!P)
         Dof_TrueDof_Matrix();
      if (ordering == Ordering::byNODES)
         return ldof_ltdof[sldof] +
            tdof_nb_offsets[GetGroupTopo().GetGroupMaster(
               ldof_group[sldof])] / vdim;
      else
         return (ldof_ltdof[sldof*vdim] +
                 tdof_nb_offsets[GetGroupTopo().GetGroupMaster(
                       ldof_group[sldof*vdim])]) / vdim;
   }

   if (ordering == Ordering::byNODES)
      return ldof_ltdof[sldof] +
         tdof_offsets[GetGroupTopo().GetGroupMasterRank(
            ldof_group[sldof])] / vdim;
   else
      return (ldof_ltdof[sldof*vdim] +
              tdof_offsets[GetGroupTopo().GetGroupMasterRank(
                    ldof_group[sldof*vdim])]) / vdim;
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
         ldof_group[ldofs[i]] = gr;

      if (!gt.IAmMaster(gr)) // we are not the master
         for (i = 0; i < nldofs; i++)
            ldof_ltdof[ldofs[i]] = -2;
   }

   // count ltdof_size
   ltdof_size = 0;
   for (i = 0; i < n; i++)
      if (ldof_ltdof[i] == -1)
         ldof_ltdof[i] = ltdof_size++;

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
         ldof_group[i] = scalar_ldof_group[VDofToDof(i)];
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
   delete gcomm;
   gcomm = NULL;
   ConstructTrueDofs();
   GenerateGlobalOffsets();
}

FiniteElementSpace *ParFiniteElementSpace::SaveUpdate()
{
   ParFiniteElementSpace *cpfes = new ParFiniteElementSpace(*this);
   Constructor();
   ConstructTrueDofs();
   GenerateGlobalOffsets();
   return cpfes;
}

#endif
