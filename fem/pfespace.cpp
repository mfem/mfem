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
   ltdof_size = pf.ltdof_size;
   Swap(ldof_group, pf.ldof_group);
   Swap(ldof_ltdof, pf.ldof_ltdof);
   Swap(dof_offsets, pf.dof_offsets);
   Swap(tdof_offsets, pf.tdof_offsets);
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

   ConstructTrueDofs();

   GenerateGlobalOffsets();
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

   int *lproc_proc = pmesh -> lproc_proc;
   int *groupmaster_lproc = pmesh -> groupmaster_lproc;

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

   i_diag = new int[ldof+1];
   j_diag = new int[ltdof];

   i_offd = new int[ldof+1];
   j_offd = new int[ldof-ltdof];

   cmap   = new int[ldof-ltdof];

   Array<Pair<int, int> > cmap_j_offd(ldof-ltdof);

   int *ncol_starts = NULL;
   if (HYPRE_AssumedPartitionCheck())
   {
      int nsize = pmesh -> lproc_proc.Size()-1;
      MPI_Request *requests = new MPI_Request[2*nsize];
      MPI_Status  *statuses = new MPI_Status[2*nsize];
      ncol_starts = hypre_CTAlloc(int, nsize+1);

      int request_counter = 0;
      // send and receive neighbors' local tdof offsets
      for (i = 1; i < nsize+1; i++)
         MPI_Irecv(&ncol_starts[i], 1, MPI_INT, lproc_proc[i], 5365,
                   MyComm, &requests[request_counter++]);

      for (i = 1; i < nsize+1; i++)
         MPI_Isend(&col_starts[0], 1, MPI_INT, lproc_proc[i], 5365,
                   MyComm, &requests[request_counter++]);

      MPI_Waitall(request_counter, requests, statuses);

      delete [] statuses;
      delete [] requests;
   }

   i_diag[0] = i_offd[0] = 0;
   diag_counter = offd_counter = 0;
   for (i = 0; i < ldof; i++)
   {
      int proc = lproc_proc[groupmaster_lproc[ldof_group[i]]];
      if (proc == MyRank)
      {
         j_diag[diag_counter++] = ldof_ltdof[i];
      }
      else
      {
         if (HYPRE_AssumedPartitionCheck())
            cmap_j_offd[offd_counter].one =
               ncol_starts[groupmaster_lproc[ldof_group[i]]] + ldof_ltdof[i];
         else
            cmap_j_offd[offd_counter].one = col_starts[proc] + ldof_ltdof[i];
         cmap_j_offd[offd_counter].two = offd_counter;
         offd_counter++;
      }
      i_diag[i+1] = diag_counter;
      i_offd[i+1] = offd_counter;
   }

   if (HYPRE_AssumedPartitionCheck())
      delete [] ncol_starts;

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

void ParFiniteElementSpace::DivideByGroupSize(double * vec)
{
   for (int i = 0; i < ldof_group.Size(); i++)
      if (pmesh -> groupmaster_lproc[ldof_group[i]] == 0) // we are the master
         vec[ldof_ltdof[i]] /= pmesh -> group_lproc.RowSize(ldof_group[i]);
}

void ParFiniteElementSpace::GetEssentialVDofs(Array<int> &bdr_attr_is_ess,
                                              Array<int> &ess_dofs)
{
   FiniteElementSpace::GetEssentialVDofs(bdr_attr_is_ess, ess_dofs);

   // Make sure that processors without boundary elements mark
   // their boundary dofs (if they have any).
   Vector d_ess_dofs(ess_dofs.Size());
   for (int i = 0; i < ess_dofs.Size(); i++)
      d_ess_dofs(i) = ess_dofs[i];

   // vector on (all) dofs
   HypreParVector *v_ess_dofs;
   if (HYPRE_AssumedPartitionCheck())
      v_ess_dofs = new HypreParVector(dof_offsets[2], d_ess_dofs, dof_offsets);
   else
      v_ess_dofs = new HypreParVector(dof_offsets[NRanks], d_ess_dofs,
                                      dof_offsets);

   // vector on true dofs
   HypreParVector *v_ess_tdofs;
   if (HYPRE_AssumedPartitionCheck())
      v_ess_tdofs = new HypreParVector(tdof_offsets[2], tdof_offsets);
   else
      v_ess_tdofs = new HypreParVector(tdof_offsets[NRanks], tdof_offsets);

   Dof_TrueDof_Matrix()->MultTranspose(*v_ess_dofs, *v_ess_tdofs);
   Dof_TrueDof_Matrix()->Mult(*v_ess_tdofs, *v_ess_dofs);

   for (int i = 0; i < ess_dofs.Size(); i++)
      if (d_ess_dofs(i) < 0.0)
         ess_dofs[i] = -1;

   delete v_ess_tdofs;
   delete v_ess_dofs;
}

int ParFiniteElementSpace::GetLocalTDofNumber(int ldof)
{
   if (pmesh -> groupmaster_lproc[ldof_group[ldof]] == 0)
      return ldof_ltdof[ldof];
   else
      return -1;
}

int ParFiniteElementSpace::GetGlobalTDofNumber(int ldof)
{
   if (HYPRE_AssumedPartitionCheck())
   {
      if (MyRank == 0)
         cerr << "ParFiniteElementSpace::GetGlobalTDofNumber "
              << "does not support Assumed Partitioning!\n" << endl;
      MPI_Finalize();
      mfem_error();
   }

   if (pmesh -> groupmaster_lproc[ldof_group[ldof]] == 0)
      return ldof_ltdof[ldof] + tdof_offsets[MyRank];
   else
      return ldof_ltdof[ldof] +
         tdof_offsets[pmesh->lproc_proc[
            pmesh->groupmaster_lproc[ldof_group[ldof]]]];
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
   int i, gr;
   int n  = GetVSize();
   int ng = pmesh -> GetNGroups();

   int nvd, ned, nfd;
   Array<int> dofs;

   Table group_ldof;
   int group_ldof_counter;
   Array<int> j_group_ltdof;

   int request_counter;
   MPI_Request *requests;
   MPI_Status  *statuses;

#if 0
   MPI_Barrier (MPI_COMM_WORLD);
   cout << MyRank << ": group_lproc :" << endl;
   pmesh -> group_lproc.Print (cout);
   cout << MyRank << ": lproc_proc :" << endl;
   pmesh -> lproc_proc.Print (cout, 1);
   cout << MyRank << ": groupmaster_lproc :" << endl;
   pmesh -> groupmaster_lproc.Print (cout, 1);
   MPI_Barrier (MPI_COMM_WORLD);
#endif

   nvd = fec->DofForGeometry(Geometry::POINT);
   ned = fec->DofForGeometry(Geometry::SEGMENT);
   nfd = (fdofs) ? (fdofs[1]-fdofs[0]) : (0);

   // Define ldof_group and mark ldof_ltdof with
   //   -1 for ldof that is ours
   //   -2 for ldof that is in a group with another master
   ldof_group.SetSize(n);
   ldof_ltdof.SetSize(n);
   group_ldof.SetDims(ng, n);

   ldof_group = 0;
   ldof_ltdof = -1;

   ldof_sign.SetSize(GetNDofs());
   ldof_sign = 1;

   request_counter = 0;
   group_ldof_counter = 0;
   group_ldof.GetI()[0] = group_ldof.GetI()[1] = 0;
   for (gr = 1; gr < ng; gr++)
   {
      int j, k, l, m, o, nv, ne, nf;
      const int *ind;

      nv = pmesh -> GroupNVertices(gr);
      ne = pmesh -> GroupNEdges(gr);
      nf = pmesh -> GroupNFaces(gr);

      // vertices
      if (nvd > 0)
         for (j = 0; j < nv; j++)
         {
            k = pmesh -> GroupVertex(gr, j);

            dofs.SetSize(nvd);
            m = nvd * k;
            for (l = 0; l < nvd; l++, m++)
               dofs[l] = m;

            DofsToVDofs(dofs);

            for (l = 0; l < dofs.Size(); l++)
               ldof_group[dofs[l]] = gr;

            if (pmesh -> groupmaster_lproc[gr] != 0) // we are not the master
               for (l = 0; l < dofs.Size(); l++)
                  ldof_ltdof[dofs[l]] = -2;

            for (l = 0; l < dofs.Size(); l++)
               group_ldof.GetJ()[group_ldof_counter++] = dofs[l];
         }

      // edges
      if (ned > 0)
         for (j = 0; j < ne; j++)
         {
            pmesh -> GroupEdge(gr, j, k, o);

            dofs.SetSize(ned);
            m = nvdofs+k*ned;
            ind = fec->DofOrderForOrientation(Geometry::SEGMENT, o);
            for (l = 0; l < ned; l++)
               if (ind[l] < 0)
               {
                  dofs[l] = m + (-1-ind[l]);
                  ldof_sign[dofs[l]] = -1;
               }
               else
                  dofs[l] = m + ind[l];

            DofsToVDofs(dofs);

            for (l = 0; l < dofs.Size(); l++)
               ldof_group[dofs[l]] = gr;

            if (pmesh -> groupmaster_lproc[gr] != 0) // we are not the master
               for (l = 0; l < dofs.Size(); l++)
                  ldof_ltdof[dofs[l]] = -2;

            for (l = 0; l < dofs.Size(); l++)
               group_ldof.GetJ()[group_ldof_counter++] = dofs[l];
         }

      // faces
      if (nfd > 0)
         for (j = 0; j < nf; j++)
         {
            pmesh -> GroupFace(gr, j, k, o);

            dofs.SetSize(nfd);
            m = nvdofs+nedofs+fdofs[k];
            ind = fec->DofOrderForOrientation(
               mesh->GetFaceBaseGeometry(k), o);
            for (l = 0; l < nfd; l++)
               if (ind[l] < 0)
               {
                  dofs[l] = m + (-1-ind[l]);
                  ldof_sign[dofs[l]] = -1;
               }
               else
                  dofs[l] = m + ind[l];

            DofsToVDofs(dofs);

            for (l = 0; l < dofs.Size(); l++)
               ldof_group[dofs[l]] = gr;

            if (pmesh -> groupmaster_lproc[gr] != 0) // we are not the master
               for (l = 0; l < dofs.Size(); l++)
                  ldof_ltdof[dofs[l]] = -2;

            for (l = 0; l < dofs.Size(); l++)
               group_ldof.GetJ()[group_ldof_counter++] = dofs[l];
         }

      group_ldof.GetI()[gr+1] = group_ldof_counter;

      if (group_ldof.RowSize(gr) != 0)
         if (pmesh -> groupmaster_lproc[gr] != 0) // we are not the master
            request_counter++;
         else
            request_counter += pmesh -> group_lproc.RowSize(gr)-1;
   }

   // count ltdof_size
   ltdof_size = 0;
   for (i = 0; i < n; i++)
      if (ldof_ltdof[i] == -1)
         ldof_ltdof[i] = ltdof_size++;

   j_group_ltdof.SetSize(group_ldof_counter); // send and receive buffers

   if (group_ldof_counter > 0)
   {
      requests = new MPI_Request[request_counter];
      statuses = new MPI_Status[request_counter];

      request_counter = 0;
      for (gr = 1; gr < ng; gr++) {

         // ignore groups without dofs
         if (group_ldof.RowSize(gr) == 0)
            continue;

         if (pmesh -> groupmaster_lproc[gr] != 0) // we are not the master
         {
            MPI_Irecv(&j_group_ltdof[group_ldof.GetI()[gr]],
                      group_ldof.RowSize(gr),
                      MPI_INT,
                      pmesh -> lproc_proc[pmesh -> groupmaster_lproc[gr]],
                      40822 + pmesh -> group_mgroup[gr],
                      MyComm,
                      &requests[request_counter]);
            request_counter++;
         }
         else // we are the master
         {
            // fill send buffer
            for (i = group_ldof.GetI()[gr]; i < group_ldof.GetI()[gr+1]; i++)
               j_group_ltdof[i] = ldof_ltdof[group_ldof.GetJ()[i]];

            for (i = pmesh -> group_lproc.GetI()[gr];
                 i < pmesh -> group_lproc.GetI()[gr+1]; i++)
            {
               if (pmesh -> group_lproc.GetJ()[i] != 0)
               {
                  MPI_Isend(&j_group_ltdof[group_ldof.GetI()[gr]],
                            group_ldof.RowSize (gr),
                            MPI_INT,
                            pmesh -> lproc_proc[pmesh -> group_lproc.GetJ()[i]],
                            40822 + pmesh -> group_mgroup[gr],
                            MyComm,
                            &requests[request_counter]);
                  request_counter++;
               }
            }
         }
      }

      MPI_Waitall(request_counter, requests, statuses);

      delete [] statuses;
      delete [] requests;
   }

   // set ldof_ltdof for groups that have another master
   for (gr = 1; gr < ng; gr++)
      if (pmesh -> groupmaster_lproc[gr] != 0) // we are not the master
      {
         for (i = group_ldof.GetI()[gr]; i < group_ldof.GetI()[gr+1]; i++)
            ldof_ltdof[group_ldof.GetJ()[i]] = j_group_ltdof[i];
      }
}

void ParFiniteElementSpace::Update()
{
   FiniteElementSpace::Update();

   ldof_group.DeleteAll();
   ldof_ltdof.DeleteAll();
   dof_offsets.DeleteAll();
   tdof_offsets.DeleteAll();
   ldof_sign.DeleteAll();
   delete P;
   P = NULL;
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
