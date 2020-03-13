// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "pfespace.hpp"
#include "prestriction.hpp"
#include "../general/forall.hpp"
#include "../general/sort_pairs.hpp"
#include "../mesh/mesh_headers.hpp"
#include "../general/binaryio.hpp"

#include <climits> // INT_MAX
#include <limits>
#include <list>

namespace mfem
{

ParFiniteElementSpace::ParFiniteElementSpace(
   const ParFiniteElementSpace &orig, ParMesh *pmesh,
   const FiniteElementCollection *fec)
   : FiniteElementSpace(orig, pmesh, fec)
{
   ParInit(pmesh ? pmesh : orig.pmesh);
}

ParFiniteElementSpace::ParFiniteElementSpace(
   const FiniteElementSpace &orig, ParMesh &pmesh,
   const FiniteElementCollection *fec)
   : FiniteElementSpace(orig, &pmesh, fec)
{
   ParInit(&pmesh);
}

ParFiniteElementSpace::ParFiniteElementSpace(
   ParMesh *pm, const FiniteElementSpace *global_fes, const int *partitioning,
   const FiniteElementCollection *f)
   : FiniteElementSpace(pm, MakeLocalNURBSext(global_fes->GetNURBSext(),
                                              pm->NURBSext),
                        f ? f : global_fes->FEColl(),
                        global_fes->GetVDim(), global_fes->GetOrdering())
{
   ParInit(pm);
   // For NURBS spaces, the variable-order data is contained in the
   // NURBSExtension of 'global_fes' and inside the ParNURBSExtension of 'pm'.

   // TODO: when general variable-order support is added, copy the local portion
   // of the variable-order data from 'global_fes' to 'this'.
}

ParFiniteElementSpace::ParFiniteElementSpace(
   ParMesh *pm, const FiniteElementCollection *f, int dim, int ordering)
   : FiniteElementSpace(pm, f, dim, ordering)
{
   ParInit(pm);
}

ParFiniteElementSpace::ParFiniteElementSpace(
   ParMesh *pm, NURBSExtension *ext, const FiniteElementCollection *f,
   int dim, int ordering)
   : FiniteElementSpace(pm, ext, f, dim, ordering)
{
   ParInit(pm);
}

// static method
ParNURBSExtension *ParFiniteElementSpace::MakeLocalNURBSext(
   const NURBSExtension *globNURBSext, const NURBSExtension *parNURBSext)
{
   if (globNURBSext == NULL) { return NULL; }
   const ParNURBSExtension *pNURBSext =
      dynamic_cast<const ParNURBSExtension*>(parNURBSext);
   MFEM_ASSERT(pNURBSext, "need a ParNURBSExtension");
   // make a copy of globNURBSext:
   NURBSExtension *tmp_globNURBSext = new NURBSExtension(*globNURBSext);
   // tmp_globNURBSext will be deleted by the following ParNURBSExtension ctor:
   return new ParNURBSExtension(tmp_globNURBSext, pNURBSext);
}

void ParFiniteElementSpace::ParInit(ParMesh *pm)
{
   pmesh = pm;
   pncmesh = pm->pncmesh;

   MyComm = pmesh->GetComm();
   NRanks = pmesh->GetNRanks();
   MyRank = pmesh->GetMyRank();

   gcomm = NULL;

   P = NULL;
   Pconf = NULL;
   R = NULL;

   num_face_nbr_dofs = -1;

   if (NURBSext && !pNURBSext())
   {
      // This is necessary in some cases: e.g. when the FiniteElementSpace
      // constructor creates a serial NURBSExtension of higher order than the
      // mesh NURBSExtension.
      MFEM_ASSERT(own_ext, "internal error");

      ParNURBSExtension *pNe = new ParNURBSExtension(
         NURBSext, dynamic_cast<ParNURBSExtension *>(pmesh->NURBSext));
      // serial NURBSext is destroyed by the above constructor
      NURBSext = pNe;
      UpdateNURBS();
   }

   Construct(); // parallel version of Construct().

   // Apply the ldof_signs to the elem_dof Table
   if (Conforming() && !NURBSext)
   {
      ApplyLDofSigns(*elem_dof);
   }
}

void ParFiniteElementSpace::Construct()
{
   if (NURBSext)
   {
      ConstructTrueNURBSDofs();
      GenerateGlobalOffsets();
   }
   else if (Conforming())
   {
      ConstructTrueDofs();
      GenerateGlobalOffsets();
   }
   else // Nonconforming()
   {
      // Initialize 'gcomm' for the cut (aka "partially conforming") space.
      // In the process, the array 'ldof_ltdof' is also initialized (for the cut
      // space) and used; however, it will be overwritten below with the real
      // true dofs. Also, 'ldof_sign' and 'ldof_group' are constructed for the
      // cut space.
      ConstructTrueDofs();

      ngedofs = ngfdofs = 0;

      // calculate number of ghost DOFs
      ngvdofs = pncmesh->GetNGhostVertices()
                * fec->DofForGeometry(Geometry::POINT);

      if (pmesh->Dimension() > 1)
      {
         ngedofs = pncmesh->GetNGhostEdges()
                   * fec->DofForGeometry(Geometry::SEGMENT);
      }

      if (pmesh->Dimension() > 2)
      {
         int stride = fec->DofForGeometry(Geometry::SQUARE);
         ngfdofs = pncmesh->GetNGhostFaces() * stride;
      }

      // total number of ghost DOFs. Ghost DOFs start at index 'ndofs', i.e.,
      // after all regular DOFs
      ngdofs = ngvdofs + ngedofs + ngfdofs;

      // get P and R matrices, initialize DOF offsets, etc. NOTE: in the NC
      // case this needs to be done here to get the number of true DOFs
      ltdof_size = BuildParallelConformingInterpolation(
                      &P, &R, dof_offsets, tdof_offsets, &ldof_ltdof, false);

      // TODO future: split BuildParallelConformingInterpolation into two parts
      // to overlap its communication with processing between this constructor
      // and the point where the P matrix is actually needed.
   }
}

void ParFiniteElementSpace::PrintPartitionStats()
{
   long ltdofs = ltdof_size;
   long min_ltdofs, max_ltdofs, sum_ltdofs;

   MPI_Reduce(&ltdofs, &min_ltdofs, 1, MPI_LONG, MPI_MIN, 0, MyComm);
   MPI_Reduce(&ltdofs, &max_ltdofs, 1, MPI_LONG, MPI_MAX, 0, MyComm);
   MPI_Reduce(&ltdofs, &sum_ltdofs, 1, MPI_LONG, MPI_SUM, 0, MyComm);

   if (MyRank == 0)
   {
      double avg = double(sum_ltdofs) / NRanks;
      mfem::out << "True DOF partitioning: min " << min_ltdofs
                << ", avg " << std::fixed << std::setprecision(1) << avg
                << ", max " << max_ltdofs
                << ", (max-avg)/avg " << 100.0*(max_ltdofs - avg)/avg
                << "%" << std::endl;
   }

   if (NRanks <= 32)
   {
      if (MyRank == 0)
      {
         mfem::out << "True DOFs by rank: " << ltdofs;
         for (int i = 1; i < NRanks; i++)
         {
            MPI_Status status;
            MPI_Recv(&ltdofs, 1, MPI_LONG, i, 123, MyComm, &status);
            mfem::out << " " << ltdofs;
         }
         mfem::out << "\n";
      }
      else
      {
         MPI_Send(&ltdofs, 1, MPI_LONG, 0, 123, MyComm);
      }
   }
}

void ParFiniteElementSpace::GetGroupComm(
   GroupCommunicator &gc, int ldof_type, Array<int> *ldof_sign)
{
   int gr;
   int ng = pmesh->GetNGroups();
   int nvd, ned, ntd = 0, nqd = 0;
   Array<int> dofs;

   int group_ldof_counter;
   Table &group_ldof = gc.GroupLDofTable();

   nvd = fec->DofForGeometry(Geometry::POINT);
   ned = fec->DofForGeometry(Geometry::SEGMENT);

   if (fdofs)
   {
      if (mesh->HasGeometry(Geometry::TRIANGLE))
      {
         ntd = fec->DofForGeometry(Geometry::TRIANGLE);
      }
      if (mesh->HasGeometry(Geometry::SQUARE))
      {
         nqd = fec->DofForGeometry(Geometry::SQUARE);
      }
   }

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
      group_ldof_counter += ntd * pmesh->GroupNTriangles(gr);
      group_ldof_counter += nqd * pmesh->GroupNQuadrilaterals(gr);
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
      int j, k, l, m, o, nv, ne, nt, nq;
      const int *ind;

      nv = pmesh->GroupNVertices(gr);
      ne = pmesh->GroupNEdges(gr);
      nt = pmesh->GroupNTriangles(gr);
      nq = pmesh->GroupNQuadrilaterals(gr);

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
            {
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

      // triangles
      if (ntd > 0)
      {
         for (j = 0; j < nt; j++)
         {
            pmesh->GroupTriangle(gr, j, k, o);

            dofs.SetSize(ntd);
            m = nvdofs+nedofs+fdofs[k];
            ind = fec->DofOrderForOrientation(Geometry::TRIANGLE, o);
            for (l = 0; l < ntd; l++)
            {
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

      // quadrilaterals
      if (nqd > 0)
      {
         for (j = 0; j < nq; j++)
         {
            pmesh->GroupQuadrilateral(gr, j, k, o);

            dofs.SetSize(nqd);
            m = nvdofs+nedofs+fdofs[k];
            ind = fec->DofOrderForOrientation(Geometry::SQUARE, o);
            for (l = 0; l < nqd; l++)
            {
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
   MFEM_ASSERT(Conforming(), "wrong code path");

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

void ParFiniteElementSpace::ApplyLDofSigns(Table &el_dof) const
{
   Array<int> all_dofs(el_dof.GetJ(), el_dof.Size_of_connections());
   ApplyLDofSigns(all_dofs);
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


const Operator *ParFiniteElementSpace::GetFaceRestriction(
   ElementDofOrdering e_ordering, FaceType type, L2FaceValues mul) const
{
   const bool is_dg_space = dynamic_cast<const L2_FECollection*>(fec)!=nullptr;
   const L2FaceValues m = (is_dg_space && mul==L2FaceValues::DoubleValued) ?
                          L2FaceValues::DoubleValued : L2FaceValues::SingleValued;
   auto key = std::make_tuple(is_dg_space, e_ordering, type, m);
   auto itr = L2F.find(key);
   if (itr != L2F.end())
   {
      return itr->second;
   }
   else
   {
      Operator* res;
      if (is_dg_space)
      {
         res = new ParL2FaceRestriction(*this, e_ordering, type, m);
      }
      else
      {
         res = new H1FaceRestriction(*this, e_ordering, type);
      }
      L2F[key] = res;
      return res;
   }
}

void ParFiniteElementSpace::GetSharedEdgeDofs(
   int group, int ei, Array<int> &dofs) const
{
   int l_edge, ori;
   MFEM_ASSERT(0 <= ei && ei < pmesh->GroupNEdges(group), "invalid edge index");
   pmesh->GroupEdge(group, ei, l_edge, ori);
   if (ori > 0) // ori = +1 or -1
   {
      GetEdgeDofs(l_edge, dofs);
   }
   else
   {
      Array<int> rdofs;
      fec->SubDofOrder(Geometry::SEGMENT, 1, 1, dofs);
      GetEdgeDofs(l_edge, rdofs);
      for (int i = 0; i < dofs.Size(); i++)
      {
         const int di = dofs[i];
         dofs[i] = (di >= 0) ? rdofs[di] : -1-rdofs[-1-di];
      }
   }
}

void ParFiniteElementSpace::GetSharedTriangleDofs(
   int group, int fi, Array<int> &dofs) const
{
   int l_face, ori;
   MFEM_ASSERT(0 <= fi && fi < pmesh->GroupNTriangles(group),
               "invalid triangular face index");
   pmesh->GroupTriangle(group, fi, l_face, ori);
   if (ori == 0)
   {
      GetFaceDofs(l_face, dofs);
   }
   else
   {
      Array<int> rdofs;
      fec->SubDofOrder(Geometry::TRIANGLE, 2, ori, dofs);
      GetFaceDofs(l_face, rdofs);
      for (int i = 0; i < dofs.Size(); i++)
      {
         const int di = dofs[i];
         dofs[i] = (di >= 0) ? rdofs[di] : -1-rdofs[-1-di];
      }
   }
}

void ParFiniteElementSpace::GetSharedQuadrilateralDofs(
   int group, int fi, Array<int> &dofs) const
{
   int l_face, ori;
   MFEM_ASSERT(0 <= fi && fi < pmesh->GroupNQuadrilaterals(group),
               "invalid quadrilateral face index");
   pmesh->GroupQuadrilateral(group, fi, l_face, ori);
   if (ori == 0)
   {
      GetFaceDofs(l_face, dofs);
   }
   else
   {
      Array<int> rdofs;
      fec->SubDofOrder(Geometry::SQUARE, 2, ori, dofs);
      GetFaceDofs(l_face, rdofs);
      for (int i = 0; i < dofs.Size(); i++)
      {
         const int di = dofs[i];
         dofs[i] = (di >= 0) ? rdofs[di] : -1-rdofs[-1-di];
      }
   }
}

void ParFiniteElementSpace::GenerateGlobalOffsets() const
{
   MFEM_ASSERT(Conforming(), "wrong code path");

   HYPRE_Int ldof[2];
   Array<HYPRE_Int> *offsets[2] = { &dof_offsets, &tdof_offsets };

   ldof[0] = GetVSize();
   ldof[1] = TrueVSize();

   pmesh->GenerateOffsets(2, ldof, offsets);

   if (HYPRE_AssumedPartitionCheck())
   {
      // communicate the neighbor offsets in tdof_nb_offsets
      GroupTopology &gt = GetGroupTopo();
      int nsize = gt.GetNumNeighbors()-1;
      MPI_Request *requests = new MPI_Request[2*nsize];
      MPI_Status  *statuses = new MPI_Status[2*nsize];
      tdof_nb_offsets.SetSize(nsize+1);
      tdof_nb_offsets[0] = tdof_offsets[0];

      // send and receive neighbors' local tdof offsets
      int request_counter = 0;
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
}

void ParFiniteElementSpace::Build_Dof_TrueDof_Matrix() const // matrix P
{
   MFEM_ASSERT(Conforming(), "wrong code path");

   if (P) { return; }

   int ldof  = GetVSize();
   int ltdof = TrueVSize();

   HYPRE_Int *i_diag = Memory<HYPRE_Int>(ldof+1);
   HYPRE_Int *j_diag = Memory<HYPRE_Int>(ltdof);
   int diag_counter;

   HYPRE_Int *i_offd = Memory<HYPRE_Int>(ldof+1);
   HYPRE_Int *j_offd = Memory<HYPRE_Int>(ldof-ltdof);
   int offd_counter;

   HYPRE_Int *cmap   = Memory<HYPRE_Int>(ldof-ltdof);

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

   SparseMatrix Pdiag;
   P->GetDiag(Pdiag);
   R = Transpose(Pdiag);
}

HypreParMatrix *ParFiniteElementSpace::GetPartialConformingInterpolation()
{
   HypreParMatrix *P_pc;
   Array<HYPRE_Int> P_pc_row_starts, P_pc_col_starts;
   BuildParallelConformingInterpolation(&P_pc, NULL, P_pc_row_starts,
                                        P_pc_col_starts, NULL, true);
   P_pc->CopyRowStarts();
   P_pc->CopyColStarts();
   return P_pc;
}

void ParFiniteElementSpace::DivideByGroupSize(double *vec)
{
   GroupTopology &gt = GetGroupTopo();
   for (int i = 0; i < ldof_group.Size(); i++)
   {
      if (gt.IAmMaster(ldof_group[i])) // we are the master
      {
         if (ldof_ltdof[i] >= 0) // see note below
         {
            vec[ldof_ltdof[i]] /= gt.GetGroupSize(ldof_group[i]);
         }
         // NOTE: in NC meshes, ldof_ltdof generated for the gtopo
         // groups by ConstructTrueDofs gets overwritten by
         // BuildParallelConformingInterpolation. Some DOFs that are
         // seen as true by the conforming code are actually slaves and
         // end up with a -1 in ldof_ltdof.
      }
   }
}

GroupCommunicator *ParFiniteElementSpace::ScalarGroupComm()
{
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
   // For non-conforming mesh, synchronization is performed on the cut (aka
   // "partially conforming") space.

   MFEM_VERIFY(ldof_marker.Size() == GetVSize(), "invalid in/out array");

   // implement allreduce(|) as reduce(|) + broadcast
   gcomm->Reduce<int>(ldof_marker, GroupCommunicator::BitOR);
   gcomm->Bcast(ldof_marker);
}

void ParFiniteElementSpace::GetEssentialVDofs(const Array<int> &bdr_attr_is_ess,
                                              Array<int> &ess_dofs,
                                              int component) const
{
   FiniteElementSpace::GetEssentialVDofs(bdr_attr_is_ess, ess_dofs, component);

   if (Conforming())
   {
      // Make sure that processors without boundary elements mark
      // their boundary dofs (if they have any).
      Synchronize(ess_dofs);
   }
}

void ParFiniteElementSpace::GetEssentialTrueDofs(const Array<int>
                                                 &bdr_attr_is_ess,
                                                 Array<int> &ess_tdof_list,
                                                 int component)
{
   Array<int> ess_dofs, true_ess_dofs;

   GetEssentialVDofs(bdr_attr_is_ess, ess_dofs, component);
   GetRestrictionMatrix()->BooleanMult(ess_dofs, true_ess_dofs);

#ifdef MFEM_DEBUG
   // Verify that in boolean arithmetic: P^T ess_dofs = R ess_dofs.
   Array<int> true_ess_dofs2(true_ess_dofs.Size());
   HypreParMatrix *Pt = Dof_TrueDof_Matrix()->Transpose();
   const int *ess_dofs_data = ess_dofs.HostRead();
   Pt->BooleanMult(1, ess_dofs_data, 0, true_ess_dofs2);
   delete Pt;
   int counter = 0;
   const int *ted = true_ess_dofs.HostRead();
   for (int i = 0; i < true_ess_dofs.Size(); i++)
   {
      if (bool(ted[i]) != bool(true_ess_dofs2[i])) { counter++; }
   }
   MFEM_VERIFY(counter == 0, "internal MFEM error: counter = " << counter);
#endif

   MarkerToList(true_ess_dofs, ess_tdof_list);
}

int ParFiniteElementSpace::GetLocalTDofNumber(int ldof) const
{
   if (Nonconforming())
   {
      Dof_TrueDof_Matrix(); // make sure P has been built

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

HYPRE_Int ParFiniteElementSpace::GetGlobalTDofNumber(int ldof) const
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

const Operator *ParFiniteElementSpace::GetProlongationMatrix() const
{
   if (Conforming())
   {
      if (Pconf) { return Pconf; }

      if (NRanks == 1)
      {
         Pconf = new IdentityOperator(GetTrueVSize());
      }
      else
      {
         if (!Device::Allows(Backend::DEVICE_MASK))
         {
            Pconf = new ConformingProlongationOperator(*this);
         }
         else
         {
            Pconf = new DeviceConformingProlongationOperator(*this);
         }
      }
      return Pconf;
   }
   else
   {
      return Dof_TrueDof_Matrix();
   }
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
         {
            int ldof = (ldofs[j] >= 0 ? ldofs[j] : -1-ldofs[j]);

            if (ldof_marker[ldof] != fn)
            {
               ldof_marker[ldof] = fn;
               send_face_nbr_ldof.AddAColumnInRow(fn);
            }
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
            int ldof = (ldofs[j] >= 0 ? ldofs[j] : -1-ldofs[j]);

            if (ldof_marker[ldof] != fn)
            {
               ldof_marker[ldof] = fn;
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
         int ldof = (ldofs[i] >= 0 ? ldofs[i] : -1-ldofs[i]);
         ldof_marker[ldof] = i;
      }

      for ( ; j < j_end; j++)
      {
         int ldof = (send_J[j] >= 0 ? send_J[j] : -1-send_J[j]);
         send_J[j] = (send_J[j] >= 0 ? ldof_marker[ldof] : -1-ldof_marker[ldof]);
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
         if (recv_J[j] >= 0)
         {
            recv_J[j] += shift;
         }
         else
         {
            recv_J[j] -= shift;
         }
      }
   }

   MPI_Waitall(num_face_nbrs, send_requests, statuses);

   // send/receive the J arrays of send_face_nbr_ldof/face_nbr_ldof,
   // respectively
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
      {
         int ldof = face_nbr_ldof.GetJ()[j];
         if (ldof < 0)
         {
            ldof = -1-ldof;
         }

         face_nbr_glob_dof_map[j] = dof_face_nbr_offsets[fn] + ldof;
      }
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

void ParFiniteElementSpace::GetFaceNbrFaceVDofs(int i, Array<int> &vdofs) const
{
   // Works for NC mesh where 'i' is an index returned by
   // ParMesh::GetSharedFace() such that i >= Mesh::GetNumFaces(), i.e. 'i' is
   // the index of a ghost.
   MFEM_ASSERT(Nonconforming() && i >= pmesh->GetNumFaces(), "");
   int el1, el2, inf1, inf2;
   pmesh->GetFaceElements(i, &el1, &el2);
   el2 = -1 - el2;
   pmesh->GetFaceInfos(i, &inf1, &inf2);
   MFEM_ASSERT(0 <= el2 && el2 < face_nbr_element_dof.Size(), "");
   const int nd = face_nbr_element_dof.RowSize(el2);
   const int *vol_vdofs = face_nbr_element_dof.GetRow(el2);
   const Element *face_nbr_el = pmesh->face_nbr_elements[el2];
   Geometry::Type geom = face_nbr_el->GetGeometryType();
   const int face_dim = Geometry::Dimension[geom]-1;

   fec->SubDofOrder(geom, face_dim, inf2, vdofs);
   // Convert local dofs to local vdofs.
   Ordering::DofsToVDofs<Ordering::byNODES>(nd/vdim, vdim, vdofs);
   // Convert local vdofs to global vdofs.
   for (int j = 0; j < vdofs.Size(); j++)
   {
      const int ldof = vdofs[j];
      vdofs[j] = (ldof >= 0) ? vol_vdofs[ldof] : -1-vol_vdofs[-1-ldof];
   }
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

const FiniteElement *ParFiniteElementSpace::GetFaceNbrFaceFE(int i) const
{
   // Works in tandem with GetFaceNbrFaceVDofs() defined above.
   MFEM_ASSERT(Nonconforming() && !NURBSext, "");
   Geometry::Type geom = (pmesh->Dimension() == 2) ?
                         Geometry::SEGMENT : Geometry::SQUARE;
   return fec->FiniteElementForGeometry(geom);
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
   gcomm->SetLTDofTable(ldof_ltdof);

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
   gcomm->SetLTDofTable(ldof_ltdof);

   // have the group masters broadcast their ltdofs to the rest of the group
   gcomm->Bcast(ldof_ltdof);
}

void ParFiniteElementSpace::GetGhostVertexDofs(const MeshId &id,
                                               Array<int> &dofs) const
{
   int nv = fec->DofForGeometry(Geometry::POINT);
   dofs.SetSize(nv);
   for (int j = 0; j < nv; j++)
   {
      dofs[j] = ndofs + nv*id.index + j;
   }
}

void ParFiniteElementSpace::GetGhostEdgeDofs(const MeshId &edge_id,
                                             Array<int> &dofs) const
{
   int nv = fec->DofForGeometry(Geometry::POINT);
   int ne = fec->DofForGeometry(Geometry::SEGMENT);
   dofs.SetSize(2*nv + ne);

   int V[2], ghost = pncmesh->GetNVertices();
   pmesh->pncmesh->GetEdgeVertices(edge_id, V);

   for (int i = 0; i < 2; i++)
   {
      int k = (V[i] < ghost) ? V[i]*nv : (ndofs + (V[i] - ghost)*nv);
      for (int j = 0; j < nv; j++)
      {
         dofs[i*nv + j] = k++;
      }
   }

   int k = ndofs + ngvdofs + (edge_id.index - pncmesh->GetNEdges())*ne;
   for (int j = 0; j < ne; j++)
   {
      dofs[2*nv + j] = k++;
   }
}

void ParFiniteElementSpace::GetGhostFaceDofs(const MeshId &face_id,
                                             Array<int> &dofs) const
{
   int nfv, V[4], E[4], Eo[4];
   nfv = pmesh->pncmesh->GetFaceVerticesEdges(face_id, V, E, Eo);

   int nv = fec->DofForGeometry(Geometry::POINT);
   int ne = fec->DofForGeometry(Geometry::SEGMENT);
   int nf_tri = fec->DofForGeometry(Geometry::TRIANGLE);
   int nf_quad = fec->DofForGeometry(Geometry::SQUARE);
   int nf = (nfv == 3) ? nf_tri : nf_quad;

   dofs.SetSize(nfv*(nv + ne) + nf);

   int offset = 0;
   for (int i = 0; i < nfv; i++)
   {
      int ghost = pncmesh->GetNVertices();
      int first = (V[i] < ghost) ? V[i]*nv : (ndofs + (V[i] - ghost)*nv);
      for (int j = 0; j < nv; j++)
      {
         dofs[offset++] = first + j;
      }
   }

   for (int i = 0; i < nfv; i++)
   {
      int ghost = pncmesh->GetNEdges();
      int first = (E[i] < ghost) ? nvdofs + E[i]*ne
                  /*          */ : ndofs + ngvdofs + (E[i] - ghost)*ne;
      const int *ind = fec->DofOrderForOrientation(Geometry::SEGMENT, Eo[i]);
      for (int j = 0; j < ne; j++)
      {
         dofs[offset++] = (ind[j] >= 0) ? (first + ind[j])
                          /*         */ : (-1 - (first + (-1 - ind[j])));
      }
   }

   const int ghost_face_index = face_id.index - pncmesh->GetNFaces();
   int first = ndofs + ngvdofs + ngedofs + nf_quad*ghost_face_index;

   for (int j = 0; j < nf; j++)
   {
      dofs[offset++] = first + j;
   }
}

void ParFiniteElementSpace::GetGhostDofs(int entity, const MeshId &id,
                                         Array<int> &dofs) const
{
   // helper to get ghost vertex, ghost edge or ghost face DOFs
   switch (entity)
   {
      case 0: GetGhostVertexDofs(id, dofs); break;
      case 1: GetGhostEdgeDofs(id, dofs); break;
      case 2: GetGhostFaceDofs(id, dofs); break;
   }
}

void ParFiniteElementSpace::GetBareDofs(int entity, int index,
                                        Array<int> &dofs) const
{
   int ned, ghost, first;
   switch (entity)
   {
      case 0:
         ned = fec->DofForGeometry(Geometry::POINT);
         ghost = pncmesh->GetNVertices();
         first = (index < ghost)
                 ? index*ned // regular vertex
                 : ndofs + (index - ghost)*ned; // ghost vertex
         break;

      case 1:
         ned = fec->DofForGeometry(Geometry::SEGMENT);
         ghost = pncmesh->GetNEdges();
         first = (index < ghost)
                 ? nvdofs + index*ned // regular edge
                 : ndofs + ngvdofs + (index - ghost)*ned; // ghost edge
         break;

      default:
         Geometry::Type geom = pncmesh->GetFaceGeometry(index);
         MFEM_ASSERT(geom == Geometry::SQUARE ||
                     geom == Geometry::TRIANGLE, "");

         ned = fec->DofForGeometry(geom);
         ghost = pncmesh->GetNFaces();

         if (index < ghost) // regular face
         {
            first = nvdofs + nedofs + (fdofs ? fdofs[index] : index*ned);
         }
         else // ghost face
         {
            index -= ghost;
            int stride = fec->DofForGeometry(Geometry::SQUARE);
            first = ndofs + ngvdofs + ngedofs + index*stride;
         }
         break;
   }

   dofs.SetSize(ned);
   for (int i = 0; i < ned; i++)
   {
      dofs[i] = first + i;
   }
}

int ParFiniteElementSpace::PackDof(int entity, int index, int edof) const
{
   // DOFs are ordered as follows:
   // vertices | edges | faces | internal | ghost vert. | g. edges | g. faces

   int ghost, ned;
   switch (entity)
   {
      case 0:
         ghost = pncmesh->GetNVertices();
         ned = fec->DofForGeometry(Geometry::POINT);

         return (index < ghost)
                ? index*ned + edof // regular vertex
                : ndofs + (index - ghost)*ned + edof; // ghost vertex

      case 1:
         ghost = pncmesh->GetNEdges();
         ned = fec->DofForGeometry(Geometry::SEGMENT);

         return (index < ghost)
                ? nvdofs + index*ned + edof // regular edge
                : ndofs + ngvdofs + (index - ghost)*ned + edof; // ghost edge

      default:
         ghost = pncmesh->GetNFaces();
         ned = fec->DofForGeometry(pncmesh->GetFaceGeometry(index));

         if (index < ghost) // regular face
         {
            return nvdofs + nedofs + (fdofs ? fdofs[index] : index*ned) + edof;
         }
         else // ghost face
         {
            index -= ghost;
            int stride = fec->DofForGeometry(Geometry::SQUARE);
            return ndofs + ngvdofs + ngedofs + index*stride + edof;
         }
   }
}

static int bisect(int* array, int size, int value)
{
   int* end = array + size;
   int* pos = std::upper_bound(array, end, value);
   MFEM_VERIFY(pos != end, "value not found");
   return pos - array;
}

/** Dissect a DOF number to obtain the entity type (0=vertex, 1=edge, 2=face),
 *  entity index and the DOF number within the entity.
 */
void ParFiniteElementSpace::UnpackDof(int dof,
                                      int &entity, int &index, int &edof) const
{
   MFEM_VERIFY(dof >= 0, "");
   if (dof < ndofs)
   {
      if (dof < nvdofs) // regular vertex
      {
         int nv = fec->DofForGeometry(Geometry::POINT);
         entity = 0, index = dof / nv, edof = dof % nv;
         return;
      }
      dof -= nvdofs;
      if (dof < nedofs) // regular edge
      {
         int ne = fec->DofForGeometry(Geometry::SEGMENT);
         entity = 1, index = dof / ne, edof = dof % ne;
         return;
      }
      dof -= nedofs;
      if (dof < nfdofs) // regular face
      {
         if (fdofs) // have mixed faces
         {
            index = bisect(fdofs+1, mesh->GetNFaces(), dof);
            edof = dof - fdofs[index];
         }
         else // uniform faces
         {
            int nf = fec->DofForGeometry(pncmesh->GetFaceGeometry(0));
            index = dof / nf, edof = dof % nf;
         }
         entity = 2;
         return;
      }
      MFEM_ABORT("Cannot unpack internal DOF");
   }
   else
   {
      dof -= ndofs;
      if (dof < ngvdofs) // ghost vertex
      {
         int nv = fec->DofForGeometry(Geometry::POINT);
         entity = 0, index = pncmesh->GetNVertices() + dof / nv, edof = dof % nv;
         return;
      }
      dof -= ngvdofs;
      if (dof < ngedofs) // ghost edge
      {
         int ne = fec->DofForGeometry(Geometry::SEGMENT);
         entity = 1, index = pncmesh->GetNEdges() + dof / ne, edof = dof % ne;
         return;
      }
      dof -= ngedofs;
      if (dof < ngfdofs) // ghost face
      {
         int stride = fec->DofForGeometry(Geometry::SQUARE);
         index = pncmesh->GetNFaces() + dof / stride, edof = dof % stride;
         entity = 2;
         return;
      }
      MFEM_ABORT("Out of range DOF.");
   }
}

/** Represents an element of the P matrix. The column number is global and
 *  corresponds to vector dimension 0. The other dimension columns are offset
 *  by 'stride'.
 */
struct PMatrixElement
{
   HYPRE_Int column, stride;
   double value;

   PMatrixElement(HYPRE_Int col = 0, HYPRE_Int str = 0, double val = 0)
      : column(col), stride(str), value(val) {}

   bool operator<(const PMatrixElement &other) const
   { return column < other.column; }

   typedef std::vector<PMatrixElement> List;
};

/** Represents one row of the P matrix, for the construction code below.
 *  The row is complete: diagonal and offdiagonal elements are not distinguished.
 */
struct PMatrixRow
{
   PMatrixElement::List elems;

   /// Add other row, times 'coef'.
   void AddRow(const PMatrixRow &other, double coef)
   {
      elems.reserve(elems.size() + other.elems.size());
      for (unsigned i = 0; i < other.elems.size(); i++)
      {
         const PMatrixElement &oei = other.elems[i];
         elems.push_back(
            PMatrixElement(oei.column, oei.stride, coef * oei.value));
      }
   }

   /// Remove duplicate columns and sum their values.
   void Collapse()
   {
      if (!elems.size()) { return; }
      std::sort(elems.begin(), elems.end());

      int j = 0;
      for (unsigned i = 1; i < elems.size(); i++)
      {
         if (elems[j].column == elems[i].column)
         {
            elems[j].value += elems[i].value;
         }
         else
         {
            elems[++j] = elems[i];
         }
      }
      elems.resize(j+1);
   }

   void write(std::ostream &os, double sign) const
   {
      bin_io::write<int>(os, elems.size());
      for (unsigned i = 0; i < elems.size(); i++)
      {
         const PMatrixElement &e = elems[i];
         bin_io::write<HYPRE_Int>(os, e.column);
         bin_io::write<int>(os, e.stride); // truncate HYPRE_Int -> int
         bin_io::write<double>(os, e.value * sign);
      }
   }

   void read(std::istream &is, double sign)
   {
      elems.resize(bin_io::read<int>(is));
      for (unsigned i = 0; i < elems.size(); i++)
      {
         PMatrixElement &e = elems[i];
         e.column = bin_io::read<HYPRE_Int>(is);
         e.stride = bin_io::read<int>(is);
         e.value = bin_io::read<double>(is) * sign;
      }
   }
};

/** Represents a message to another processor containing P matrix rows.
 *  Used by ParFiniteElementSpace::ParallelConformingInterpolation.
 */
class NeighborRowMessage : public VarMessage<314>
{
public:
   typedef NCMesh::MeshId MeshId;
   typedef ParNCMesh::GroupId GroupId;

   struct RowInfo
   {
      int entity, index, edof;
      GroupId group;
      PMatrixRow row;

      RowInfo(int ent, int idx, int edof, GroupId grp, const PMatrixRow &row)
         : entity(ent), index(idx), edof(edof), group(grp), row(row) {}

      RowInfo(int ent, int idx, int edof, GroupId grp)
         : entity(ent), index(idx), edof(edof), group(grp) {}

      typedef std::vector<RowInfo> List;
   };

   NeighborRowMessage() : pncmesh(NULL) {}

   void AddRow(int entity, int index, int edof, GroupId group,
               const PMatrixRow &row)
   {
      rows.push_back(RowInfo(entity, index, edof, group, row));
   }

   const RowInfo::List& GetRows() const { return rows; }

   void SetNCMesh(ParNCMesh* pnc) { pncmesh = pnc; }
   void SetFEC(const FiniteElementCollection* fec) { this->fec = fec; }

   typedef std::map<int, NeighborRowMessage> Map;

protected:
   RowInfo::List rows;

   ParNCMesh *pncmesh;
   const FiniteElementCollection* fec;

   virtual void Encode(int rank);
   virtual void Decode(int);
};


void NeighborRowMessage::Encode(int rank)
{
   std::ostringstream stream;

   Array<MeshId> ent_ids[3];
   Array<GroupId> group_ids[3];
   Array<int> row_idx[3];

   // encode MeshIds and groups
   for (unsigned i = 0; i < rows.size(); i++)
   {
      const RowInfo &ri = rows[i];
      const MeshId &id = pncmesh->GetNCList(ri.entity).LookUp(ri.index);
      ent_ids[ri.entity].Append(id);
      row_idx[ri.entity].Append(i);
      group_ids[ri.entity].Append(ri.group);
   }

   Array<GroupId> all_group_ids;
   all_group_ids.Reserve(rows.size());
   for (int i = 0; i < 3; i++)
   {
      all_group_ids.Append(group_ids[i]);
   }

   pncmesh->AdjustMeshIds(ent_ids, rank);
   pncmesh->EncodeMeshIds(stream, ent_ids);
   pncmesh->EncodeGroups(stream, all_group_ids);

   // write all rows to the stream
   for (int ent = 0; ent < 3; ent++)
   {
      const Array<MeshId> &ids = ent_ids[ent];
      for (int i = 0; i < ids.Size(); i++)
      {
         const MeshId &id = ids[i];
         const RowInfo &ri = rows[row_idx[ent][i]];
         MFEM_ASSERT(ent == ri.entity, "");

#ifdef MFEM_DEBUG_PMATRIX
         mfem::out << "Rank " << pncmesh->MyRank << " sending to " << rank
                   << ": ent " << ri.entity << ", index " << ri.index
                   << ", edof " << ri.edof << " (id " << id.element << "/"
                   << int(id.local) << ")" << std::endl;
#endif

         // handle orientation and sign change
         int edof = ri.edof;
         double s = 1.0;
         if (ent == 1)
         {
            int eo = pncmesh->GetEdgeNCOrientation(id);
            const int* ind = fec->DofOrderForOrientation(Geometry::SEGMENT, eo);
            if ((edof = ind[edof]) < 0)
            {
               edof = -1 - edof;
               s = -1;
            }
         }

         bin_io::write<int>(stream, edof);
         ri.row.write(stream, s);
      }
   }

   rows.clear();
   stream.str().swap(data);
}

void NeighborRowMessage::Decode(int rank)
{
   std::istringstream stream(data);

   Array<MeshId> ent_ids[3];
   Array<GroupId> group_ids;

   // decode vertex/edge/face IDs and groups
   pncmesh->DecodeMeshIds(stream, ent_ids);
   pncmesh->DecodeGroups(stream, group_ids);

   int nrows = ent_ids[0].Size() + ent_ids[1].Size() + ent_ids[2].Size();
   MFEM_ASSERT(nrows == group_ids.Size(), "");

   rows.clear();
   rows.reserve(nrows);

   // read rows
   for (int ent = 0, gi = 0; ent < 3; ent++)
   {
      const Array<MeshId> &ids = ent_ids[ent];
      for (int i = 0; i < ids.Size(); i++)
      {
         const MeshId &id = ids[i];
         int edof = bin_io::read<int>(stream);

         // handle orientation and sign change
         const int *ind = NULL;
         if (ent == 1)
         {
            int eo = pncmesh->GetEdgeNCOrientation(id);
            ind = fec->DofOrderForOrientation(Geometry::SEGMENT, eo);
         }
         else if (ent == 2)
         {
            Geometry::Type geom = pncmesh->GetFaceGeometry(id.index);
            int fo = pncmesh->GetFaceOrientation(id.index);
            ind = fec->DofOrderForOrientation(geom, fo);
         }

         double s = 1.0;
         if (ind && (edof = ind[edof]) < 0)
         {
            edof = -1 - edof;
            s = -1.0;
         }

         rows.push_back(RowInfo(ent, id.index, edof, group_ids[gi++]));
         rows.back().row.read(stream, s);

#ifdef MFEM_DEBUG_PMATRIX
         mfem::out << "Rank " << pncmesh->MyRank << " receiving from " << rank
                   << ": ent " << rows.back().entity << ", index "
                   << rows.back().index << ", edof " << rows.back().edof
                   << std::endl;
#endif
      }
   }
}

void
ParFiniteElementSpace::ScheduleSendRow(const PMatrixRow &row, int dof,
                                       GroupId group_id,
                                       NeighborRowMessage::Map &send_msg) const
{
   int ent, idx, edof;
   UnpackDof(dof, ent, idx, edof);

   const ParNCMesh::CommGroup &group = pncmesh->GetGroup(group_id);
   for (unsigned i = 0; i < group.size(); i++)
   {
      int rank = group[i];
      if (rank != MyRank)
      {
         NeighborRowMessage &msg = send_msg[rank];
         msg.AddRow(ent, idx, edof, group_id, row);
         msg.SetNCMesh(pncmesh);
         msg.SetFEC(fec);
#ifdef MFEM_PMATRIX_STATS
         n_rows_sent++;
#endif
      }
   }
}

void ParFiniteElementSpace::ForwardRow(const PMatrixRow &row, int dof,
                                       GroupId group_sent_id, GroupId group_id,
                                       NeighborRowMessage::Map &send_msg) const
{
   int ent, idx, edof;
   UnpackDof(dof, ent, idx, edof);

   const ParNCMesh::CommGroup &group = pncmesh->GetGroup(group_id);
   for (unsigned i = 0; i < group.size(); i++)
   {
      int rank = group[i];
      if (rank != MyRank && !pncmesh->GroupContains(group_sent_id, rank))
      {
         NeighborRowMessage &msg = send_msg[rank];
         GroupId invalid = -1; // to prevent forwarding again
         msg.AddRow(ent, idx, edof, invalid, row);
         msg.SetNCMesh(pncmesh);
         msg.SetFEC(fec);
#ifdef MFEM_PMATRIX_STATS
         n_rows_fwd++;
#endif
#ifdef MFEM_DEBUG_PMATRIX
         mfem::out << "Rank " << pncmesh->GetMyRank() << " forwarding to "
                   << rank << ": ent " << ent << ", index" << idx
                   << ", edof " << edof << std::endl;
#endif
      }
   }
}

#ifdef MFEM_DEBUG_PMATRIX
void ParFiniteElementSpace
::DebugDumpDOFs(std::ostream &os,
                const SparseMatrix &deps,
                const Array<GroupId> &dof_group,
                const Array<GroupId> &dof_owner,
                const Array<bool> &finalized) const
{
   for (int i = 0; i < dof_group.Size(); i++)
   {
      os << i << ": ";
      if (i < (nvdofs + nedofs + nfdofs) || i >= ndofs)
      {
         int ent, idx, edof;
         UnpackDof(i, ent, idx, edof);

         os << edof << " @ ";
         if (i > ndofs) { os << "ghost "; }
         switch (ent)
         {
            case 0: os << "vertex "; break;
            case 1: os << "edge "; break;
            default: os << "face "; break;
         }
         os << idx << "; ";

         if (i < deps.Height() && deps.RowSize(i))
         {
            os << "depends on ";
            for (int j = 0; j < deps.RowSize(i); j++)
            {
               os << deps.GetRowColumns(i)[j] << " ("
                  << deps.GetRowEntries(i)[j] << ")";
               if (j < deps.RowSize(i)-1) { os << ", "; }
            }
            os << "; ";
         }
         else
         {
            os << "no deps; ";
         }

         os << "group " << dof_group[i] << " (";
         const ParNCMesh::CommGroup &g = pncmesh->GetGroup(dof_group[i]);
         for (unsigned j = 0; j < g.size(); j++)
         {
            if (j) { os << ", "; }
            os << g[j];
         }

         os << "), owner " << dof_owner[i] << " (rank "
            << pncmesh->GetGroup(dof_owner[i])[0] << "); "
            << (finalized[i] ? "finalized" : "NOT finalized");
      }
      else
      {
         os << "internal";
      }
      os << "\n";
   }
}
#endif

int ParFiniteElementSpace
::BuildParallelConformingInterpolation(HypreParMatrix **P, SparseMatrix **R,
                                       Array<HYPRE_Int> &dof_offs,
                                       Array<HYPRE_Int> &tdof_offs,
                                       Array<int> *dof_tdof,
                                       bool partial) const
{
   bool dg = (nvdofs == 0 && nedofs == 0 && nfdofs == 0);

#ifdef MFEM_PMATRIX_STATS
   n_msgs_sent = n_msgs_recv = 0;
   n_rows_sent = n_rows_recv = n_rows_fwd = 0;
#endif

   // *** STEP 1: build master-slave dependency lists ***

   int total_dofs = ndofs + ngdofs;
   SparseMatrix deps(ndofs, total_dofs);

   if (!dg && !partial)
   {
      Array<int> master_dofs, slave_dofs;

      // loop through *all* master edges/faces, constrain their slaves
      for (int entity = 0; entity <= 2; entity++)
      {
         const NCMesh::NCList &list = pncmesh->GetNCList(entity);
         if (!list.masters.size()) { continue; }

         IsoparametricTransformation T;
         DenseMatrix I;

         // process masters that we own or that affect our edges/faces
         for (unsigned mi = 0; mi < list.masters.size(); mi++)
         {
            const NCMesh::Master &mf = list.masters[mi];

            // get master DOFs
            pncmesh->IsGhost(entity, mf.index)
            ? GetGhostDofs(entity, mf, master_dofs)
            : GetEntityDofs(entity, mf.index, master_dofs);

            if (!master_dofs.Size()) { continue; }

            const FiniteElement* fe = fec->FiniteElementForGeometry(mf.Geom());
            if (!fe) { continue; }

            switch (mf.Geom())
            {
               case Geometry::SQUARE:   T.SetFE(&QuadrilateralFE); break;
               case Geometry::TRIANGLE: T.SetFE(&TriangleFE); break;
               case Geometry::SEGMENT:  T.SetFE(&SegmentFE); break;
               default: MFEM_ABORT("unsupported geometry");
            }

            // constrain slaves that exist in our mesh
            for (int si = mf.slaves_begin; si < mf.slaves_end; si++)
            {
               const NCMesh::Slave &sf = list.slaves[si];
               if (pncmesh->IsGhost(entity, sf.index)) { continue; }

               GetEntityDofs(entity, sf.index, slave_dofs, mf.Geom());
               if (!slave_dofs.Size()) { continue; }

               sf.OrientedPointMatrix(T.GetPointMat());
               T.FinalizeTransformation();
               fe->GetLocalInterpolation(T, I);

               // make each slave DOF dependent on all master DOFs
               AddDependencies(deps, master_dofs, slave_dofs, I);
            }
         }
      }

      deps.Finalize();
   }

   // *** STEP 2: initialize group and owner ID for each DOF ***

   Array<GroupId> dof_group(total_dofs);
   Array<GroupId> dof_owner(total_dofs);
   dof_group = 0;
   dof_owner = 0;

   if (!dg)
   {
      Array<int> dofs;

      // initialize dof_group[], dof_owner[]
      for (int entity = 0; entity <= 2; entity++)
      {
         const NCMesh::NCList &list = pncmesh->GetNCList(entity);

         std::size_t lsize[3] =
         { list.conforming.size(), list.masters.size(), list.slaves.size() };

         for (int l = 0; l < 3; l++)
         {
            for (std::size_t i = 0; i < lsize[l]; i++)
            {
               const MeshId &id =
                  (l == 0) ? list.conforming[i] :
                  (l == 1) ? (const MeshId&) list.masters[i]
                  /*    */ : (const MeshId&) list.slaves[i];

               if (id.index < 0) { continue; }

               GroupId owner = pncmesh->GetEntityOwnerId(entity, id.index);
               GroupId group = pncmesh->GetEntityGroupId(entity, id.index);

               GetBareDofs(entity, id.index, dofs);

               for (int j = 0; j < dofs.Size(); j++)
               {
                  int dof = dofs[j];
                  dof_owner[dof] = owner;
                  dof_group[dof] = group;
               }
            }
         }
      }
   }

   // *** STEP 3: count true DOFs and calculate P row/column partitions ***

   Array<bool> finalized(total_dofs);
   finalized = false;

   // DOFs that stayed independent and are ours are true DOFs
   int num_true_dofs = 0;
   for (int i = 0; i < ndofs; i++)
   {
      if (dof_owner[i] == 0 && deps.RowSize(i) == 0)
      {
         num_true_dofs++;
         finalized[i] = true;
      }
   }

   // calculate global offsets
   HYPRE_Int loc_sizes[2] = { ndofs*vdim, num_true_dofs*vdim };
   Array<HYPRE_Int>* offsets[2] = { &dof_offs, &tdof_offs };
   pmesh->GenerateOffsets(2, loc_sizes, offsets); // calls MPI_Scan, MPI_Bcast

   HYPRE_Int my_tdof_offset =
      tdof_offs[HYPRE_AssumedPartitionCheck() ? 0 : MyRank];

   if (R)
   {
      // initialize the restriction matrix (also parallel but block-diagonal)
      *R = new SparseMatrix(num_true_dofs*vdim, ndofs*vdim);
   }
   if (dof_tdof)
   {
      dof_tdof->SetSize(ndofs*vdim);
      *dof_tdof = -1;
   }

   std::vector<PMatrixRow> pmatrix(total_dofs);

   bool bynodes = (ordering == Ordering::byNODES);
   int vdim_factor = bynodes ? 1 : vdim;
   int dof_stride = bynodes ? ndofs : 1;
   int tdof_stride = bynodes ? num_true_dofs : 1;

   // big container for all messages we send (the list is for iterations)
   std::list<NeighborRowMessage::Map> send_msg;
   send_msg.push_back(NeighborRowMessage::Map());

   // put identity in P and R for true DOFs, set ldof_ltdof
   for (int dof = 0, tdof = 0; dof < ndofs; dof++)
   {
      if (finalized[dof])
      {
         pmatrix[dof].elems.push_back(
            PMatrixElement(my_tdof_offset + vdim_factor*tdof, tdof_stride, 1.));

         // prepare messages to neighbors with identity rows
         if (dof_group[dof] != 0)
         {
            ScheduleSendRow(pmatrix[dof], dof, dof_group[dof], send_msg.back());
         }

         for (int vd = 0; vd < vdim; vd++)
         {
            int vdof = dof*vdim_factor + vd*dof_stride;
            int vtdof = tdof*vdim_factor + vd*tdof_stride;

            if (R) { (*R)->Add(vtdof, vdof, 1.0); }
            if (dof_tdof) { (*dof_tdof)[vdof] = vtdof; }
         }
         tdof++;
      }
   }

   // send identity rows
   NeighborRowMessage::IsendAll(send_msg.back(), MyComm);
#ifdef MFEM_PMATRIX_STATS
   n_msgs_sent += send_msg.back().size();
#endif

   if (R) { (*R)->Finalize(); }

   // *** STEP 4: main loop ***

   // a single instance (recv_msg) is reused for all incoming messages
   NeighborRowMessage recv_msg;
   recv_msg.SetNCMesh(pncmesh);
   recv_msg.SetFEC(fec);

   int num_finalized = num_true_dofs;
   PMatrixRow buffer;
   buffer.elems.reserve(1024);

   while (num_finalized < ndofs)
   {
      // prepare a new round of send buffers
      if (send_msg.back().size())
      {
         send_msg.push_back(NeighborRowMessage::Map());
      }

      // check for incoming messages, receive PMatrixRows
      int rank, size;
      while (NeighborRowMessage::IProbe(rank, size, MyComm))
      {
         recv_msg.Recv(rank, size, MyComm);
#ifdef MFEM_PMATRIX_STATS
         n_msgs_recv++;
         n_rows_recv += recv_msg.GetRows().size();
#endif

         const NeighborRowMessage::RowInfo::List &rows = recv_msg.GetRows();
         for (unsigned i = 0; i < rows.size(); i++)
         {
            const NeighborRowMessage::RowInfo &ri = rows[i];
            int dof = PackDof(ri.entity, ri.index, ri.edof);
            pmatrix[dof] = ri.row;

            if (dof < ndofs && !finalized[dof]) { num_finalized++; }
            finalized[dof] = true;

            if (ri.group >= 0 && dof_group[dof] != ri.group)
            {
               // the sender didn't see the complete group, forward the message
               ForwardRow(ri.row, dof, ri.group, dof_group[dof], send_msg.back());
            }
         }
      }

      // finalize all rows that can currently be finalized
      bool done = false;
      while (!done)
      {
         done = true;
         for (int dof = 0; dof < ndofs; dof++)
         {
            if (finalized[dof]) { continue; }

            bool owned = (dof_owner[dof] == 0);
            bool shared = (dof_group[dof] != 0);

            if (owned && DofFinalizable(dof, finalized, deps))
            {
               const int* dep_col = deps.GetRowColumns(dof);
               const double* dep_coef = deps.GetRowEntries(dof);
               int num_dep = deps.RowSize(dof);

               // form linear combination of rows
               buffer.elems.clear();
               for (int j = 0; j < num_dep; j++)
               {
                  buffer.AddRow(pmatrix[dep_col[j]], dep_coef[j]);
               }
               buffer.Collapse();
               pmatrix[dof] = buffer;

               finalized[dof] = true;
               num_finalized++;
               done = false;

               // send row to neighbors who need it
               if (shared)
               {
                  ScheduleSendRow(pmatrix[dof], dof, dof_group[dof],
                                  send_msg.back());
               }
            }
         }
      }

#ifdef MFEM_DEBUG_PMATRIX
      /*static int dump = 0;
      if (dump < 10)
      {
         char fname[100];
         sprintf(fname, "dofs%02d.txt", MyRank);
         std::ofstream f(fname);
         DebugDumpDOFs(f, deps, dof_group, dof_owner, finalized);
         dump++;
      }*/
#endif

      // send current batch of messages
      NeighborRowMessage::IsendAll(send_msg.back(), MyComm);
#ifdef MFEM_PMATRIX_STATS
      n_msgs_sent += send_msg.back().size();
#endif
   }

   if (P)
   {
      *P = MakeVDimHypreMatrix(pmatrix, ndofs, num_true_dofs,
                               dof_offs, tdof_offs);
   }

   // clean up possible remaining messages in the queue to avoid receiving
   // them erroneously in the next run
   int rank, size;
   while (NeighborRowMessage::IProbe(rank, size, MyComm))
   {
      recv_msg.RecvDrop(rank, size, MyComm);
   }

   // make sure we can discard all send buffers
   for (std::list<NeighborRowMessage::Map>::iterator
        it = send_msg.begin(); it != send_msg.end(); ++it)
   {
      NeighborRowMessage::WaitAllSent(*it);
   }

#ifdef MFEM_PMATRIX_STATS
   int n_rounds = send_msg.size();
   int glob_rounds, glob_msgs_sent, glob_msgs_recv;
   int glob_rows_sent, glob_rows_recv, glob_rows_fwd;

   MPI_Reduce(&n_rounds,    &glob_rounds,    1, MPI_INT, MPI_SUM, 0, MyComm);
   MPI_Reduce(&n_msgs_sent, &glob_msgs_sent, 1, MPI_INT, MPI_SUM, 0, MyComm);
   MPI_Reduce(&n_msgs_recv, &glob_msgs_recv, 1, MPI_INT, MPI_SUM, 0, MyComm);
   MPI_Reduce(&n_rows_sent, &glob_rows_sent, 1, MPI_INT, MPI_SUM, 0, MyComm);
   MPI_Reduce(&n_rows_recv, &glob_rows_recv, 1, MPI_INT, MPI_SUM, 0, MyComm);
   MPI_Reduce(&n_rows_fwd,  &glob_rows_fwd,  1, MPI_INT, MPI_SUM, 0, MyComm);

   if (MyRank == 0)
   {
      mfem::out << "P matrix stats (avg per rank): "
                << double(glob_rounds)/NRanks << " rounds, "
                << double(glob_msgs_sent)/NRanks << " msgs sent, "
                << double(glob_msgs_recv)/NRanks << " msgs recv, "
                << double(glob_rows_sent)/NRanks << " rows sent, "
                << double(glob_rows_recv)/NRanks << " rows recv, "
                << double(glob_rows_fwd)/NRanks << " rows forwarded."
                << std::endl;
   }
#endif

   return num_true_dofs*vdim;
}


HypreParMatrix* ParFiniteElementSpace
::MakeVDimHypreMatrix(const std::vector<PMatrixRow> &rows,
                      int local_rows, int local_cols,
                      Array<HYPRE_Int> &row_starts,
                      Array<HYPRE_Int> &col_starts) const
{
   bool assumed = HYPRE_AssumedPartitionCheck();
   bool bynodes = (ordering == Ordering::byNODES);

   HYPRE_Int first_col = col_starts[assumed ? 0 : MyRank];
   HYPRE_Int next_col = col_starts[assumed ? 1 : MyRank+1];

   // count nonzeros in diagonal/offdiagonal parts
   HYPRE_Int nnz_diag = 0, nnz_offd = 0;
   std::map<HYPRE_Int, int> col_map;
   for (int i = 0; i < local_rows; i++)
   {
      for (unsigned j = 0; j < rows[i].elems.size(); j++)
      {
         const PMatrixElement &elem = rows[i].elems[j];
         HYPRE_Int col = elem.column;
         if (col >= first_col && col < next_col)
         {
            nnz_diag += vdim;
         }
         else
         {
            nnz_offd += vdim;
            for (int vd = 0; vd < vdim; vd++)
            {
               col_map[col] = -1;
               col += elem.stride;
            }
         }
      }
   }

   // create offd column mapping
   HYPRE_Int *cmap = Memory<HYPRE_Int>(col_map.size());
   int offd_col = 0;
   for (std::map<HYPRE_Int, int>::iterator
        it = col_map.begin(); it != col_map.end(); ++it)
   {
      cmap[offd_col] = it->first;
      it->second = offd_col++;
   }

   HYPRE_Int *I_diag = Memory<HYPRE_Int>(vdim*local_rows + 1);
   HYPRE_Int *I_offd = Memory<HYPRE_Int>(vdim*local_rows + 1);

   HYPRE_Int *J_diag = Memory<HYPRE_Int>(nnz_diag);
   HYPRE_Int *J_offd = Memory<HYPRE_Int>(nnz_offd);

   double *A_diag = Memory<double>(nnz_diag);
   double *A_offd = Memory<double>(nnz_offd);

   int vdim1 = bynodes ? vdim : 1;
   int vdim2 = bynodes ? 1 : vdim;
   int vdim_offset = bynodes ? local_cols : 1;

   // copy the diag/offd elements
   nnz_diag = nnz_offd = 0;
   int vrow = 0;
   for (int vd1 = 0; vd1 < vdim1; vd1++)
   {
      for (int i = 0; i < local_rows; i++)
      {
         for (int vd2 = 0; vd2 < vdim2; vd2++)
         {
            I_diag[vrow] = nnz_diag;
            I_offd[vrow++] = nnz_offd;

            int vd = bynodes ? vd1 : vd2;
            for (unsigned j = 0; j < rows[i].elems.size(); j++)
            {
               const PMatrixElement &elem = rows[i].elems[j];
               if (elem.column >= first_col && elem.column < next_col)
               {
                  J_diag[nnz_diag] = elem.column + vd*vdim_offset - first_col;
                  A_diag[nnz_diag++] = elem.value;
               }
               else
               {
                  J_offd[nnz_offd] = col_map[elem.column + vd*elem.stride];
                  A_offd[nnz_offd++] = elem.value;
               }
            }
         }
      }
   }
   MFEM_ASSERT(vrow == vdim*local_rows, "");
   I_diag[vrow] = nnz_diag;
   I_offd[vrow] = nnz_offd;

   return new HypreParMatrix(MyComm,
                             row_starts.Last(), col_starts.Last(),
                             row_starts.GetData(), col_starts.GetData(),
                             I_diag, J_diag, A_diag,
                             I_offd, J_offd, A_offd,
                             col_map.size(), cmap);
}


static HYPRE_Int* make_i_array(int nrows)
{
   HYPRE_Int *I = Memory<HYPRE_Int>(nrows+1);
   for (int i = 0; i <= nrows; i++) { I[i] = -1; }
   return I;
}

static HYPRE_Int* make_j_array(HYPRE_Int* I, int nrows)
{
   int nnz = 0;
   for (int i = 0; i < nrows; i++)
   {
      if (I[i] >= 0) { nnz++; }
   }
   HYPRE_Int *J = Memory<HYPRE_Int>(nnz);

   I[nrows] = -1;
   for (int i = 0, k = 0; i <= nrows; i++)
   {
      HYPRE_Int col = I[i];
      I[i] = k;
      if (col >= 0) { J[k++] = col; }
   }
   return J;
}

HypreParMatrix*
ParFiniteElementSpace::RebalanceMatrix(int old_ndofs,
                                       const Table* old_elem_dof)
{
   MFEM_VERIFY(Nonconforming(), "Only supported for nonconforming meshes.");
   MFEM_VERIFY(old_dof_offsets.Size(), "ParFiniteElementSpace::Update needs to "
               "be called before ParFiniteElementSpace::RebalanceMatrix");

   HYPRE_Int old_offset = HYPRE_AssumedPartitionCheck()
                          ? old_dof_offsets[0] : old_dof_offsets[MyRank];

   // send old DOFs of elements we used to own
   ParNCMesh* pncmesh = pmesh->pncmesh;
   pncmesh->SendRebalanceDofs(old_ndofs, *old_elem_dof, old_offset, this);

   Array<int> dofs;
   int vsize = GetVSize();

   const Array<int> &old_index = pncmesh->GetRebalanceOldIndex();
   MFEM_VERIFY(old_index.Size() == pmesh->GetNE(),
               "Mesh::Rebalance was not called before "
               "ParFiniteElementSpace::RebalanceMatrix");

   // prepare the local (diagonal) part of the matrix
   HYPRE_Int* i_diag = make_i_array(vsize);
   for (int i = 0; i < pmesh->GetNE(); i++)
   {
      if (old_index[i] >= 0) // we had this element before
      {
         const int* old_dofs = old_elem_dof->GetRow(old_index[i]);
         GetElementDofs(i, dofs);

         for (int vd = 0; vd < vdim; vd++)
         {
            for (int j = 0; j < dofs.Size(); j++)
            {
               int row = DofToVDof(dofs[j], vd);
               if (row < 0) { row = -1 - row; }

               int col = DofToVDof(old_dofs[j], vd, old_ndofs);
               if (col < 0) { col = -1 - col; }

               i_diag[row] = col;
            }
         }
      }
   }
   HYPRE_Int* j_diag = make_j_array(i_diag, vsize);

   // receive old DOFs for elements we obtained from others in Rebalance
   Array<int> new_elements;
   Array<long> old_remote_dofs;
   pncmesh->RecvRebalanceDofs(new_elements, old_remote_dofs);

   // create the offdiagonal part of the matrix
   HYPRE_Int* i_offd = make_i_array(vsize);
   for (int i = 0, pos = 0; i < new_elements.Size(); i++)
   {
      GetElementDofs(new_elements[i], dofs);
      const long* old_dofs = &old_remote_dofs[pos];
      pos += dofs.Size() * vdim;

      for (int vd = 0; vd < vdim; vd++)
      {
         for (int j = 0; j < dofs.Size(); j++)
         {
            int row = DofToVDof(dofs[j], vd);
            if (row < 0) { row = -1 - row; }

            if (i_diag[row] == i_diag[row+1]) // diag row empty?
            {
               i_offd[row] = old_dofs[j + vd * dofs.Size()];
            }
         }
      }
   }
   HYPRE_Int* j_offd = make_j_array(i_offd, vsize);

   // create the offd column map
   int offd_cols = i_offd[vsize];
   Array<Pair<HYPRE_Int, int> > cmap_offd(offd_cols);
   for (int i = 0; i < offd_cols; i++)
   {
      cmap_offd[i].one = j_offd[i];
      cmap_offd[i].two = i;
   }
   SortPairs<HYPRE_Int, int>(cmap_offd, offd_cols);

   HYPRE_Int* cmap = Memory<HYPRE_Int>(offd_cols);
   for (int i = 0; i < offd_cols; i++)
   {
      cmap[i] = cmap_offd[i].one;
      j_offd[cmap_offd[i].two] = i;
   }

   HypreParMatrix *M;
   M = new HypreParMatrix(MyComm, MyRank, NRanks, dof_offsets, old_dof_offsets,
                          i_diag, j_diag, i_offd, j_offd, cmap, offd_cols);
   return M;
}


struct DerefDofMessage
{
   std::vector<HYPRE_Int> dofs;
   MPI_Request request;
};

HypreParMatrix*
ParFiniteElementSpace::ParallelDerefinementMatrix(int old_ndofs,
                                                  const Table* old_elem_dof)
{
   int nrk = HYPRE_AssumedPartitionCheck() ? 2 : NRanks;

   MFEM_VERIFY(Nonconforming(), "Not implemented for conforming meshes.");
   MFEM_VERIFY(old_dof_offsets[nrk], "Missing previous (finer) space.");

#if 0 // check no longer seems to work with NC tet refinement
   MFEM_VERIFY(dof_offsets[nrk] <= old_dof_offsets[nrk],
               "Previous space is not finer.");
#endif

   // Note to the reader: please make sure you first read
   // FiniteElementSpace::RefinementMatrix, then
   // FiniteElementSpace::DerefinementMatrix, and only then this function.
   // You have been warned! :-)

   Mesh::GeometryList elem_geoms(*mesh);

   Array<int> dofs, old_dofs, old_vdofs;
   Vector row;

   ParNCMesh* pncmesh = pmesh->pncmesh;

   int ldof[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; i++)
   {
      ldof[i] = 0;
   }
   for (int i = 0; i < elem_geoms.Size(); i++)
   {
      Geometry::Type geom = elem_geoms[i];
      ldof[geom] = fec->FiniteElementForGeometry(geom)->GetDof();
   }

   const CoarseFineTransformations &dtrans = pncmesh->GetDerefinementTransforms();
   const Array<int> &old_ranks = pncmesh->GetDerefineOldRanks();

   std::map<int, DerefDofMessage> messages;

   HYPRE_Int old_offset = HYPRE_AssumedPartitionCheck()
                          ? old_dof_offsets[0] : old_dof_offsets[MyRank];

   // communicate DOFs for derefinements that straddle processor boundaries,
   // note that this is infrequent due to the way elements are ordered
   for (int k = 0; k < dtrans.embeddings.Size(); k++)
   {
      const Embedding &emb = dtrans.embeddings[k];

      int fine_rank = old_ranks[k];
      int coarse_rank = (emb.parent < 0) ? (-1 - emb.parent)
                        : pncmesh->ElementRank(emb.parent);

      if (coarse_rank != MyRank && fine_rank == MyRank)
      {
         old_elem_dof->GetRow(k, dofs);
         DofsToVDofs(dofs, old_ndofs);

         DerefDofMessage &msg = messages[k];
         msg.dofs.resize(dofs.Size());
         for (int i = 0; i < dofs.Size(); i++)
         {
            msg.dofs[i] = old_offset + dofs[i];
         }

         MPI_Isend(&msg.dofs[0], msg.dofs.size(), HYPRE_MPI_INT,
                   coarse_rank, 291, MyComm, &msg.request);
      }
      else if (coarse_rank == MyRank && fine_rank != MyRank)
      {
         MFEM_ASSERT(emb.parent >= 0, "");
         Geometry::Type geom = mesh->GetElementBaseGeometry(emb.parent);

         DerefDofMessage &msg = messages[k];
         msg.dofs.resize(ldof[geom]*vdim);

         MPI_Irecv(&msg.dofs[0], ldof[geom]*vdim, HYPRE_MPI_INT,
                   fine_rank, 291, MyComm, &msg.request);
      }
      // TODO: coalesce Isends/Irecvs to the same rank. Typically, on uniform
      // derefinement, there should be just one send to MyRank-1 and one recv
      // from MyRank+1
   }

   DenseTensor localR[Geometry::NumGeom];
   for (int i = 0; i < elem_geoms.Size(); i++)
   {
      GetLocalDerefinementMatrices(elem_geoms[i], localR[elem_geoms[i]]);
   }

   // create the diagonal part of the derefinement matrix
   SparseMatrix *diag = new SparseMatrix(ndofs*vdim, old_ndofs*vdim);

   Array<char> mark(diag->Height());
   mark = 0;

   for (int k = 0; k < dtrans.embeddings.Size(); k++)
   {
      const Embedding &emb = dtrans.embeddings[k];
      if (emb.parent < 0) { continue; }

      int coarse_rank = pncmesh->ElementRank(emb.parent);
      int fine_rank = old_ranks[k];

      if (coarse_rank == MyRank && fine_rank == MyRank)
      {
         Geometry::Type geom = mesh->GetElementBaseGeometry(emb.parent);
         DenseMatrix &lR = localR[geom](emb.matrix);

         elem_dof->GetRow(emb.parent, dofs);
         old_elem_dof->GetRow(k, old_dofs);

         for (int vd = 0; vd < vdim; vd++)
         {
            old_dofs.Copy(old_vdofs);
            DofsToVDofs(vd, old_vdofs, old_ndofs);

            for (int i = 0; i < lR.Height(); i++)
            {
               if (!std::isfinite(lR(i, 0))) { continue; }

               int r = DofToVDof(dofs[i], vd);
               int m = (r >= 0) ? r : (-1 - r);

               if (!mark[m])
               {
                  lR.GetRow(i, row);
                  diag->SetRow(r, old_vdofs, row);
                  mark[m] = 1;
               }
            }
         }
      }
   }
   diag->Finalize();

   // wait for all sends/receives to complete
   for (auto it = messages.begin(); it != messages.end(); ++it)
   {
      MPI_Wait(&it->second.request, MPI_STATUS_IGNORE);
   }

   // create the offdiagonal part of the derefinement matrix
   SparseMatrix *offd = new SparseMatrix(ndofs*vdim, 1);

   std::map<HYPRE_Int, int> col_map;
   for (int k = 0; k < dtrans.embeddings.Size(); k++)
   {
      const Embedding &emb = dtrans.embeddings[k];
      if (emb.parent < 0) { continue; }

      int coarse_rank = pncmesh->ElementRank(emb.parent);
      int fine_rank = old_ranks[k];

      if (coarse_rank == MyRank && fine_rank != MyRank)
      {
         Geometry::Type geom = mesh->GetElementBaseGeometry(emb.parent);
         DenseMatrix &lR = localR[geom](emb.matrix);

         elem_dof->GetRow(emb.parent, dofs);

         DerefDofMessage &msg = messages[k];
         MFEM_ASSERT(msg.dofs.size(), "");

         for (int vd = 0; vd < vdim; vd++)
         {
            MFEM_ASSERT(ldof[geom], "");
            HYPRE_Int* remote_dofs = &msg.dofs[vd*ldof[geom]];

            for (int i = 0; i < lR.Height(); i++)
            {
               if (!std::isfinite(lR(i, 0))) { continue; }

               int r = DofToVDof(dofs[i], vd);
               int m = (r >= 0) ? r : (-1 - r);

               if (!mark[m])
               {
                  lR.GetRow(i, row);
                  MFEM_ASSERT(ldof[geom] == row.Size(), "");
                  for (int j = 0; j < ldof[geom]; j++)
                  {
                     if (row[j] == 0.0) { continue; } // NOTE: lR thresholded
                     int &lcol = col_map[remote_dofs[j]];
                     if (!lcol) { lcol = col_map.size(); }
                     offd->_Set_(m, lcol-1, row[j]);
                  }
                  mark[m] = 1;
               }
            }
         }
      }
   }
   messages.clear();
   offd->Finalize(0);
   offd->SetWidth(col_map.size());

   // create offd column mapping for use by hypre
   HYPRE_Int *cmap = Memory<HYPRE_Int>(offd->Width());
   for (std::map<HYPRE_Int, int>::iterator
        it = col_map.begin(); it != col_map.end(); ++it)
   {
      cmap[it->second-1] = it->first;
   }

   // reorder offd columns so that 'cmap' is monotonic
   // NOTE: this is easier and probably faster (offd is small) than making
   // sure cmap is determined and sorted before the offd matrix is created
   {
      int width = offd->Width();
      Array<Pair<HYPRE_Int, int> > reorder(width);
      for (int i = 0; i < width; i++)
      {
         reorder[i].one = cmap[i];
         reorder[i].two = i;
      }
      reorder.Sort();

      Array<int> reindex(width);
      for (int i = 0; i < width; i++)
      {
         reindex[reorder[i].two] = i;
         cmap[i] = reorder[i].one;
      }

      int *J = offd->GetJ();
      for (int i = 0; i < offd->NumNonZeroElems(); i++)
      {
         J[i] = reindex[J[i]];
      }
      offd->SortColumnIndices();
   }

   HypreParMatrix* R;
   R = new HypreParMatrix(MyComm, dof_offsets[nrk], old_dof_offsets[nrk],
                          dof_offsets, old_dof_offsets, diag, offd, cmap);

#ifndef HYPRE_BIGINT
   diag->LoseData();
   offd->LoseData();
#else
   diag->SetDataOwner(false);
   offd->SetDataOwner(false);
#endif
   delete diag;
   delete offd;

   R->SetOwnerFlags(3, 3, 1);

   return R;
}

void ParFiniteElementSpace::Destroy()
{
   ldof_group.DeleteAll();
   ldof_ltdof.DeleteAll();
   dof_offsets.DeleteAll();
   tdof_offsets.DeleteAll();
   tdof_nb_offsets.DeleteAll();
   // preserve old_dof_offsets
   ldof_sign.DeleteAll();

   delete P; P = NULL;
   delete Pconf; Pconf = NULL;
   delete R; R = NULL;

   delete gcomm; gcomm = NULL;

   num_face_nbr_dofs = -1;
   face_nbr_element_dof.Clear();
   face_nbr_ldof.Clear();
   face_nbr_glob_dof_map.DeleteAll();
   send_face_nbr_ldof.Clear();
}

void ParFiniteElementSpace::GetTrueTransferOperator(
   const FiniteElementSpace &coarse_fes, OperatorHandle &T) const
{
   OperatorHandle Tgf(T.Type() == Operator::Hypre_ParCSR ?
                      Operator::MFEM_SPARSEMAT : Operator::ANY_TYPE);
   GetTransferOperator(coarse_fes, Tgf);
   Dof_TrueDof_Matrix(); // Make sure R is built - we need R in all cases.
   if (T.Type() == Operator::Hypre_ParCSR)
   {
      const ParFiniteElementSpace *c_pfes =
         dynamic_cast<const ParFiniteElementSpace *>(&coarse_fes);
      MFEM_ASSERT(c_pfes != NULL, "coarse_fes must be a parallel space");
      SparseMatrix *RA = mfem::Mult(*R, *Tgf.As<SparseMatrix>());
      Tgf.Clear();
      T.Reset(c_pfes->Dof_TrueDof_Matrix()->
              LeftDiagMult(*RA, GetTrueDofOffsets()));
      delete RA;
   }
   else
   {
      T.Reset(new TripleProductOperator(R, Tgf.Ptr(),
                                        coarse_fes.GetProlongationMatrix(),
                                        false, Tgf.OwnsOperator(), false));
      Tgf.SetOperatorOwner(false);
   }
}

void ParFiniteElementSpace::Update(bool want_transform)
{
   if (mesh->GetSequence() == sequence)
   {
      return; // no need to update, no-op
   }
   if (want_transform && mesh->GetSequence() != sequence + 1)
   {
      MFEM_ABORT("Error in update sequence. Space needs to be updated after "
                 "each mesh modification.");
   }
   sequence = mesh->GetSequence();

   if (NURBSext)
   {
      UpdateNURBS();
      return;
   }

   Table* old_elem_dof = NULL;
   int old_ndofs;

   // save old DOF table
   if (want_transform)
   {
      old_elem_dof = elem_dof;
      elem_dof = NULL;
      old_ndofs = ndofs;
      Swap(dof_offsets, old_dof_offsets);
   }

   Destroy();
   FiniteElementSpace::Destroy(); // calls Th.Clear()

   FiniteElementSpace::Construct();
   Construct();

   BuildElementToDofTable();

   if (want_transform)
   {
      // calculate appropriate GridFunction transformation
      switch (mesh->GetLastOperation())
      {
         case Mesh::REFINE:
         {
            if (Th.Type() != Operator::MFEM_SPARSEMAT)
            {
               Th.Reset(new RefinementOperator(this, old_elem_dof, old_ndofs));
               // The RefinementOperator takes ownership of 'old_elem_dofs', so
               // we no longer own it:
               old_elem_dof = NULL;
            }
            else
            {
               // calculate fully assembled matrix
               Th.Reset(RefinementMatrix(old_ndofs, old_elem_dof));
            }
            break;
         }

         case Mesh::DEREFINE:
         {
            Th.Reset(ParallelDerefinementMatrix(old_ndofs, old_elem_dof));
            if (Nonconforming())
            {
               Th.SetOperatorOwner(false);
               Th.Reset(new TripleProductOperator(P, R, Th.Ptr(),
                                                  false, false, true));
            }
            break;
         }

         case Mesh::REBALANCE:
         {
            Th.Reset(RebalanceMatrix(old_ndofs, old_elem_dof));
            break;
         }

         default:
            break;
      }

      delete old_elem_dof;
   }
}


ConformingProlongationOperator::ConformingProlongationOperator(
   const ParFiniteElementSpace &pfes)
   : Operator(pfes.GetVSize(), pfes.GetTrueVSize()),
     external_ldofs(),
     gc(pfes.GroupComm())
{
   MFEM_VERIFY(pfes.Conforming(), "");
   const Table &group_ldof = gc.GroupLDofTable();
   external_ldofs.Reserve(Height()-Width());
   for (int gr = 1; gr < group_ldof.Size(); gr++)
   {
      if (!gc.GetGroupTopology().IAmMaster(gr))
      {
         external_ldofs.Append(group_ldof.GetRow(gr), group_ldof.RowSize(gr));
      }
   }
   external_ldofs.Sort();
   MFEM_ASSERT(external_ldofs.Size() == Height()-Width(), "");
#ifdef MFEM_DEBUG
   for (int j = 1; j < external_ldofs.Size(); j++)
   {
      // Check for repeated ldofs.
      MFEM_VERIFY(external_ldofs[j-1] < external_ldofs[j], "");
   }
   int j = 0;
   for (int i = 0; i < external_ldofs.Size(); i++)
   {
      const int end = external_ldofs[i];
      for ( ; j < end; j++)
      {
         MFEM_VERIFY(j-i == pfes.GetLocalTDofNumber(j), "");
      }
      j = end+1;
   }
   for ( ; j < Height(); j++)
   {
      MFEM_VERIFY(j-external_ldofs.Size() == pfes.GetLocalTDofNumber(j), "");
   }
   // gc.PrintInfo();
   // pfes.Dof_TrueDof_Matrix()->PrintCommPkg();
#endif
}

void ConformingProlongationOperator::Mult(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(x.Size() == Width(), "");
   MFEM_ASSERT(y.Size() == Height(), "");

   const double *xdata = x.HostRead();
   double *ydata = y.HostWrite();
   const int m = external_ldofs.Size();

   const int in_layout = 2; // 2 - input is ltdofs array
   gc.BcastBegin(const_cast<double*>(xdata), in_layout);

   int j = 0;
   for (int i = 0; i < m; i++)
   {
      const int end = external_ldofs[i];
      std::copy(xdata+j-i, xdata+end-i, ydata+j);
      j = end+1;
   }
   std::copy(xdata+j-m, xdata+Width(), ydata+j);

   const int out_layout = 0; // 0 - output is ldofs array
   gc.BcastEnd(ydata, out_layout);
}

void ConformingProlongationOperator::MultTranspose(
   const Vector &x, Vector &y) const
{
   MFEM_ASSERT(x.Size() == Height(), "");
   MFEM_ASSERT(y.Size() == Width(), "");

   const double *xdata = x.HostRead();
   double *ydata = y.HostWrite();
   const int m = external_ldofs.Size();

   gc.ReduceBegin(xdata);

   int j = 0;
   for (int i = 0; i < m; i++)
   {
      const int end = external_ldofs[i];
      std::copy(xdata+j, xdata+end, ydata+j-i);
      j = end+1;
   }
   std::copy(xdata+j, xdata+Height(), ydata+j-m);

   const int out_layout = 2; // 2 - output is an array on all ltdofs
   gc.ReduceEnd<double>(ydata, out_layout, GroupCommunicator::Sum);
}

DeviceConformingProlongationOperator::DeviceConformingProlongationOperator(
   const ParFiniteElementSpace &pfes) :
   ConformingProlongationOperator(pfes),
   mpi_gpu_aware(Device::GetGPUAwareMPI())
{
   MFEM_ASSERT(pfes.Conforming(), "internal error");
   const SparseMatrix *R = pfes.GetRestrictionMatrix();
   MFEM_ASSERT(R->Finalized(), "");
   const int tdofs = R->Height();
   MFEM_ASSERT(tdofs == pfes.GetTrueVSize(), "");
   MFEM_ASSERT(tdofs == R->HostReadI()[tdofs], "");
   ltdof_ldof = Array<int>(const_cast<int*>(R->HostReadJ()), tdofs);
   ltdof_ldof.UseDevice();
   {
      Table nbr_ltdof;
      gc.GetNeighborLTDofTable(nbr_ltdof);
      const int nb_connections = nbr_ltdof.Size_of_connections();
      shr_ltdof.SetSize(nb_connections);
      shr_ltdof.CopyFrom(nbr_ltdof.GetJ());
      shr_buf.SetSize(nb_connections);
      shr_buf.UseDevice(true);
      shr_buf_offsets = nbr_ltdof.GetIMemory();
      {
         Array<int> shr_ltdof(nbr_ltdof.GetJ(), nb_connections);
         Array<int> unique_ltdof(shr_ltdof);
         unique_ltdof.Sort();
         unique_ltdof.Unique();
         // Note: the next loop modifies the J array of nbr_ltdof
         for (int i = 0; i < shr_ltdof.Size(); i++)
         {
            shr_ltdof[i] = unique_ltdof.FindSorted(shr_ltdof[i]);
            MFEM_ASSERT(shr_ltdof[i] != -1, "internal error");
         }
         Table unique_shr;
         Transpose(shr_ltdof, unique_shr, unique_ltdof.Size());
         unq_ltdof = Array<int>(unique_ltdof, unique_ltdof.Size());
         unq_shr_i = Array<int>(unique_shr.GetI(), unique_shr.Size()+1);
         unq_shr_j = Array<int>(unique_shr.GetJ(), unique_shr.Size_of_connections());
      }
      nbr_ltdof.GetJMemory().Delete();
      nbr_ltdof.LoseData();
   }
   {
      Table nbr_ldof;
      gc.GetNeighborLDofTable(nbr_ldof);
      const int nb_connections = nbr_ldof.Size_of_connections();
      ext_ldof.SetSize(nb_connections);
      ext_ldof.CopyFrom(nbr_ldof.GetJ());
      ext_buf.SetSize(nb_connections);
      ext_buf.UseDevice(true);
      ext_buf_offsets = nbr_ldof.GetIMemory();
      nbr_ldof.GetJMemory().Delete();
      nbr_ldof.LoseData();
   }
   const GroupTopology &gtopo = gc.GetGroupTopology();
   int req_counter = 0;
   for (int nbr = 1; nbr < gtopo.GetNumNeighbors(); nbr++)
   {
      const int send_offset = shr_buf_offsets[nbr];
      const int send_size = shr_buf_offsets[nbr+1] - send_offset;
      if (send_size > 0) { req_counter++; }

      const int recv_offset = ext_buf_offsets[nbr];
      const int recv_size = ext_buf_offsets[nbr+1] - recv_offset;
      if (recv_size > 0) { req_counter++; }
   }
   requests = new MPI_Request[req_counter];
}

static void ExtractSubVector(const int N,
                             const Array<int> &indices,
                             const Vector &in, Vector &out)
{
   auto y = out.Write();
   const auto x = in.Read();
   const auto I = indices.Read();
   MFEM_FORALL(i, N, y[i] = x[I[i]];); // indices can be repeated
}

void DeviceConformingProlongationOperator::BcastBeginCopy(
   const Vector &x) const
{
   // shr_buf[i] = src[shr_ltdof[i]]
   if (shr_ltdof.Size() == 0) { return; }
   ExtractSubVector(shr_ltdof.Size(), shr_ltdof, x, shr_buf);
   // If the above kernel is executed asynchronously, we should wait for it to
   // complete
   if (mpi_gpu_aware) { MFEM_STREAM_SYNC; }
}

static void SetSubVector(const int N,
                         const Array<int> &indices,
                         const Vector &in, Vector &out)
{
   auto y = out.Write();
   const auto x = in.Read();
   const auto I = indices.Read();
   MFEM_FORALL(i, N, y[I[i]] = x[i];);
}

void DeviceConformingProlongationOperator::BcastLocalCopy(
   const Vector &x, Vector &y) const
{
   // dst[ltdof_ldof[i]] = src[i]
   if (ltdof_ldof.Size() == 0) { return; }
   SetSubVector(ltdof_ldof.Size(), ltdof_ldof, x, y);
}

void DeviceConformingProlongationOperator::BcastEndCopy(
   Vector &y) const
{
   // dst[ext_ldof[i]] = ext_buf[i]
   if (ext_ldof.Size() == 0) { return; }
   SetSubVector(ext_ldof.Size(), ext_ldof, ext_buf, y);
}

void DeviceConformingProlongationOperator::Mult(const Vector &x,
                                                Vector &y) const
{
   const GroupTopology &gtopo = gc.GetGroupTopology();
   BcastBeginCopy(x); // copy to 'shr_buf'
   int req_counter = 0;
   for (int nbr = 1; nbr < gtopo.GetNumNeighbors(); nbr++)
   {
      const int send_offset = shr_buf_offsets[nbr];
      const int send_size = shr_buf_offsets[nbr+1] - send_offset;
      if (send_size > 0)
      {
         auto send_buf = mpi_gpu_aware ? shr_buf.Read() : shr_buf.HostRead();
         MPI_Isend(send_buf + send_offset, send_size, MPI_DOUBLE,
                   gtopo.GetNeighborRank(nbr), 41822,
                   gtopo.GetComm(), &requests[req_counter++]);
      }
      const int recv_offset = ext_buf_offsets[nbr];
      const int recv_size = ext_buf_offsets[nbr+1] - recv_offset;
      if (recv_size > 0)
      {
         auto recv_buf = mpi_gpu_aware ? ext_buf.Write() : ext_buf.HostWrite();
         MPI_Irecv(recv_buf + recv_offset, recv_size, MPI_DOUBLE,
                   gtopo.GetNeighborRank(nbr), 41822,
                   gtopo.GetComm(), &requests[req_counter++]);
      }
   }
   BcastLocalCopy(x, y);
   MPI_Waitall(req_counter, requests, MPI_STATUSES_IGNORE);
   BcastEndCopy(y); // copy from 'ext_buf'
}

DeviceConformingProlongationOperator::~DeviceConformingProlongationOperator()
{
   delete [] requests;
   ext_buf_offsets.Delete();
   shr_buf_offsets.Delete();
}

void DeviceConformingProlongationOperator::ReduceBeginCopy(
   const Vector &x) const
{
   // ext_buf[i] = src[ext_ldof[i]]
   if (ext_ldof.Size() == 0) { return; }
   ExtractSubVector(ext_ldof.Size(), ext_ldof, x, ext_buf);
   // If the above kernel is executed asynchronously, we should wait for it to
   // complete
   if (mpi_gpu_aware) { MFEM_STREAM_SYNC; }
}

void DeviceConformingProlongationOperator::ReduceLocalCopy(
   const Vector &x, Vector &y) const
{
   // dst[i] = src[ltdof_ldof[i]]
   if (ltdof_ldof.Size() == 0) { return; }
   ExtractSubVector(ltdof_ldof.Size(), ltdof_ldof, x, y);
}

static void AddSubVector(const int num_unique_dst_indices,
                         const Array<int> &unique_dst_indices,
                         const Array<int> &unique_to_src_offsets,
                         const Array<int> &unique_to_src_indices,
                         const Vector &src,
                         Vector &dst)
{
   auto y = dst.Write();
   const auto x = src.Read();
   const auto DST_I = unique_dst_indices.Read();
   const auto SRC_O = unique_to_src_offsets.Read();
   const auto SRC_I = unique_to_src_indices.Read();
   MFEM_FORALL(i, num_unique_dst_indices,
   {
      const int dst_idx = DST_I[i];
      double sum = y[dst_idx];
      const int end = SRC_O[i+1];
      for (int j = SRC_O[i]; j != end; ++j) { sum += x[SRC_I[j]]; }
      y[dst_idx] = sum;
   });
}

void DeviceConformingProlongationOperator::ReduceEndAssemble(Vector &y) const
{
   // dst[shr_ltdof[i]] += shr_buf[i]
   const int unq_ltdof_size = unq_ltdof.Size();
   if (unq_ltdof_size == 0) { return; }
   AddSubVector(unq_ltdof_size, unq_ltdof, unq_shr_i, unq_shr_j, shr_buf, y);
}

void DeviceConformingProlongationOperator::MultTranspose(const Vector &x,
                                                         Vector &y) const
{
   const GroupTopology &gtopo = gc.GetGroupTopology();
   ReduceBeginCopy(x); // copy to 'ext_buf'
   int req_counter = 0;
   for (int nbr = 1; nbr < gtopo.GetNumNeighbors(); nbr++)
   {
      const int send_offset = ext_buf_offsets[nbr];
      const int send_size = ext_buf_offsets[nbr+1] - send_offset;
      if (send_size > 0)
      {
         auto send_buf = mpi_gpu_aware ? ext_buf.Read() : ext_buf.HostRead();
         MPI_Isend(send_buf + send_offset, send_size, MPI_DOUBLE,
                   gtopo.GetNeighborRank(nbr), 41823,
                   gtopo.GetComm(), &requests[req_counter++]);
      }
      const int recv_offset = shr_buf_offsets[nbr];
      const int recv_size = shr_buf_offsets[nbr+1] - recv_offset;
      if (recv_size > 0)
      {
         auto recv_buf = mpi_gpu_aware ? shr_buf.Write() : shr_buf.HostWrite();
         MPI_Irecv(recv_buf + recv_offset, recv_size, MPI_DOUBLE,
                   gtopo.GetNeighborRank(nbr), 41823,
                   gtopo.GetComm(), &requests[req_counter++]);
      }
   }
   ReduceLocalCopy(x, y);
   MPI_Waitall(req_counter, requests, MPI_STATUSES_IGNORE);
   ReduceEndAssemble(y); // assemble from 'shr_buf'
}

} // namespace mfem

#endif
