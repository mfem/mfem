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

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "pfespace.hpp"
#include "prestriction.hpp"
#include "../general/forall.hpp"
#include "../general/sort_pairs.hpp"
#include "../mesh/mesh_headers.hpp"
#include "../general/binaryio.hpp"

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
   pncmesh = nullptr;

   MyComm = pmesh->GetComm();
   NRanks = pmesh->GetNRanks();
   MyRank = pmesh->GetMyRank();

   gcomm = nullptr;

   P = nullptr;
   Pconf = nullptr;
   nonconf_P = false;
   Rconf = nullptr;
   R = nullptr;
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

   // Check for shared triangular faces with interior Nedelec DoFs
   CheckNDSTriaDofs();
}

void ParFiniteElementSpace::Construct()
{
   MFEM_VERIFY(!IsVariableOrder(), "variable orders are not implemented"
               " for ParFiniteElementSpace yet.");

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
      pncmesh = pmesh->pncmesh;

      // Initialize 'gcomm' for the cut (aka "partially conforming") space.
      // In the process, the array 'ldof_ltdof' is also initialized (for the cut
      // space) and used; however, it will be overwritten below with the real
      // true dofs. Also, 'ldof_sign' and 'ldof_group' are constructed for the
      // cut space.
      ConstructTrueDofs();

      ngedofs = ngfdofs = 0;

      // calculate number of ghost DOFs
      ngvdofs = pncmesh->GetNGhostVertices()
                * fec->DofForGeometry(Geometry::Type::POINT);

      if (pmesh->Dimension() > 1)
      {
         ngedofs = pncmesh->GetNGhostEdges()
                   * fec->DofForGeometry(Geometry::Type::SEGMENT);
      }

      if (pmesh->Dimension() > 2)
      {
         ngfdofs = pncmesh->GetNGhostFaces()
                   * fec->DofForGeometry(Geometry::Type::SQUARE);
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
   long long ltdofs = ltdof_size;
   long long min_ltdofs, max_ltdofs, sum_ltdofs;

   MPI_Reduce(&ltdofs, &min_ltdofs, 1, MPI_LONG_LONG, MPI_MIN, 0, MyComm);
   MPI_Reduce(&ltdofs, &max_ltdofs, 1, MPI_LONG_LONG, MPI_MAX, 0, MyComm);
   MPI_Reduce(&ltdofs, &sum_ltdofs, 1, MPI_LONG_LONG, MPI_SUM, 0, MyComm);

   if (MyRank == 0)
   {
      real_t avg = real_t(sum_ltdofs) / NRanks;
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
            MPI_Recv(&ltdofs, 1, MPI_LONG_LONG, i, 123, MyComm, &status);
            mfem::out << " " << ltdofs;
         }
         mfem::out << "\n";
      }
      else
      {
         MPI_Send(&ltdofs, 1, MPI_LONG_LONG, 0, 123, MyComm);
      }
   }
}

void ParFiniteElementSpace::GetGroupComm(
   GroupCommunicator &gc, int ldof_type, Array<int> *g_ldof_sign)
{
   int gr;
   int ng = pmesh->GetNGroups();
   int nvd, ned, ntd = 0, nqd = 0;
   Array<int> dofs;

   int group_ldof_counter;
   Table &group_ldof = gc.GroupLDofTable();

   nvd = fec->DofForGeometry(Geometry::POINT);
   ned = fec->DofForGeometry(Geometry::SEGMENT);

   if (mesh->Dimension() >= 3)
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

   if (g_ldof_sign)
   {
      g_ldof_sign->SetSize(GetNDofs());
      *g_ldof_sign = 1;
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
                  if (g_ldof_sign)
                  {
                     (*g_ldof_sign)[dofs[l]] = -1;
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
            m = nvdofs + nedofs + FirstFaceDof(k);
            ind = fec->DofOrderForOrientation(Geometry::TRIANGLE, o);
            for (l = 0; l < ntd; l++)
            {
               if (ind[l] < 0)
               {
                  dofs[l] = m + (-1-ind[l]);
                  if (g_ldof_sign)
                  {
                     (*g_ldof_sign)[dofs[l]] = -1;
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
            m = nvdofs + nedofs + FirstFaceDof(k);
            ind = fec->DofOrderForOrientation(Geometry::SQUARE, o);
            for (l = 0; l < nqd; l++)
            {
               if (ind[l] < 0)
               {
                  dofs[l] = m + (-1-ind[l]);
                  if (g_ldof_sign)
                  {
                     (*g_ldof_sign)[dofs[l]] = -1;
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

void ParFiniteElementSpace::GetElementDofs(int i, Array<int> &dofs,
                                           DofTransformation &doftrans) const
{
   if (elem_dof)
   {
      elem_dof->GetRow(i, dofs);

      if (DoFTransArray[mesh->GetElementBaseGeometry(i)])
      {
         Array<int> Fo;
         elem_fos->GetRow(i, Fo);
         doftrans.SetDofTransformation(
            *DoFTransArray[mesh->GetElementBaseGeometry(i)]);
         doftrans.SetFaceOrientations(Fo);
         doftrans.SetVDim();
      }
      return;
   }
   FiniteElementSpace::GetElementDofs(i, dofs, doftrans);
   if (Conforming())
   {
      ApplyLDofSigns(dofs);
   }
}

void ParFiniteElementSpace::GetBdrElementDofs(int i, Array<int> &dofs,
                                              DofTransformation &doftrans) const
{
   if (bdr_elem_dof)
   {
      bdr_elem_dof->GetRow(i, dofs);

      if (DoFTransArray[mesh->GetBdrElementGeometry(i)])
      {
         Array<int> Fo;
         bdr_elem_fos->GetRow(i, Fo);
         doftrans.SetDofTransformation(
            *DoFTransArray[mesh->GetBdrElementGeometry(i)]);
         doftrans.SetFaceOrientations(Fo);
         doftrans.SetVDim();
      }
      return;
   }
   FiniteElementSpace::GetBdrElementDofs(i, dofs, doftrans);
   if (Conforming())
   {
      ApplyLDofSigns(dofs);
   }
}

int ParFiniteElementSpace::GetFaceDofs(int i, Array<int> &dofs,
                                       int variant) const
{
   if (face_dof != nullptr && variant == 0)
   {
      face_dof->GetRow(i, dofs);
      return fec->GetOrder();
   }
   int p = FiniteElementSpace::GetFaceDofs(i, dofs, variant);
   if (Conforming())
   {
      ApplyLDofSigns(dofs);
   }
   return p;
}

const FiniteElement *ParFiniteElementSpace::GetFE(int i) const
{
   int ne = mesh->GetNE();
   if (i >= ne) { return GetFaceNbrFE(i - ne); }
   else { return FiniteElementSpace::GetFE(i); }
}

const FaceRestriction *ParFiniteElementSpace::GetFaceRestriction(
   ElementDofOrdering f_ordering, FaceType type, L2FaceValues mul) const
{
   const bool is_dg_space = IsDGSpace();
   const L2FaceValues m = (is_dg_space && mul==L2FaceValues::DoubleValued) ?
                          L2FaceValues::DoubleValued : L2FaceValues::SingleValued;
   auto key = std::make_tuple(is_dg_space, f_ordering, type, m);
   auto itr = L2F.find(key);
   if (itr != L2F.end())
   {
      return itr->second;
   }
   else
   {
      FaceRestriction *res;
      if (is_dg_space)
      {
         if (Conforming())
         {
            res = new ParL2FaceRestriction(*this, f_ordering, type, m);
         }
         else
         {
            res = new ParNCL2FaceRestriction(*this, f_ordering, type, m);
         }
      }
      else
      {
         if (Conforming())
         {
            res = new ConformingFaceRestriction(*this, f_ordering, type);
         }
         else
         {
            res = new ParNCH1FaceRestriction(*this, f_ordering, type);
         }
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

   HYPRE_BigInt ldof[2];
   Array<HYPRE_BigInt> *offsets[2] = { &dof_offsets, &tdof_offsets };

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
         MPI_Irecv(&tdof_nb_offsets[i], 1, HYPRE_MPI_BIG_INT,
                   gt.GetNeighborRank(i), 5365, MyComm,
                   &requests[request_counter++]);
      }
      for (int i = 1; i <= nsize; i++)
      {
         MPI_Isend(&tdof_nb_offsets[0], 1, HYPRE_MPI_BIG_INT,
                   gt.GetNeighborRank(i), 5365, MyComm,
                   &requests[request_counter++]);
      }
      MPI_Waitall(request_counter, requests, statuses);

      delete [] statuses;
      delete [] requests;
   }
}

void ParFiniteElementSpace::CheckNDSTriaDofs()
{
   // Check for Nedelec basis
   bool nd_basis = dynamic_cast<const ND_FECollection*>(fec);
   if (!nd_basis)
   {
      nd_strias = false;
      return;
   }

   // Check for interior face dofs on triangles (the use of TETRAHEDRON
   // is not an error)
   bool nd_fdof  = fec->HasFaceDofs(Geometry::TETRAHEDRON,
                                    GetMaxElementOrder());
   if (!nd_fdof)
   {
      nd_strias = false;
      return;
   }

   // Check for shared triangle faces
   bool strias   = false;
   {
      int ngrps = pmesh->GetNGroups();
      for (int g = 1; g < ngrps; g++)
      {
         strias |= pmesh->GroupNTriangles(g);
      }
   }

   // Combine results
   int loc_nd_strias = strias ? 1 : 0;
   int glb_nd_strias = 0;
   MPI_Allreduce(&loc_nd_strias, &glb_nd_strias, 1,
                 MPI_INTEGER, MPI_SUM, MyComm);
   nd_strias = glb_nd_strias > 0;
}

void ParFiniteElementSpace::Build_Dof_TrueDof_Matrix() const // matrix P
{
   MFEM_ASSERT(Conforming(), "wrong code path");

   if (P) { return; }

   if (!nd_strias)
   {
      // Safe to assume 1-1 correspondence between shared dofs
      int ldof  = GetVSize();
      int ltdof = TrueVSize();

      HYPRE_Int *i_diag = Memory<HYPRE_Int>(ldof+1);
      HYPRE_Int *j_diag = Memory<HYPRE_Int>(ltdof);
      int diag_counter;

      HYPRE_Int *i_offd = Memory<HYPRE_Int>(ldof+1);
      HYPRE_Int *j_offd = Memory<HYPRE_Int>(ldof-ltdof);
      int offd_counter;

      HYPRE_BigInt *cmap   = Memory<HYPRE_BigInt>(ldof-ltdof);

      HYPRE_BigInt *col_starts = GetTrueDofOffsets();
      HYPRE_BigInt *row_starts = GetDofOffsets();

      Array<Pair<HYPRE_BigInt, int> > cmap_j_offd(ldof-ltdof);

      i_diag[0] = i_offd[0] = 0;
      diag_counter = offd_counter = 0;
      for (int i = 0; i < ldof; i++)
      {
         int ltdof_i = GetLocalTDofNumber(i);
         if (ltdof_i >= 0)
         {
            j_diag[diag_counter++] = ltdof_i;
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

      SortPairs<HYPRE_BigInt, int>(cmap_j_offd, offd_counter);

      for (int i = 0; i < offd_counter; i++)
      {
         cmap[i] = cmap_j_offd[i].one;
         j_offd[cmap_j_offd[i].two] = i;
      }

      P = new HypreParMatrix(MyComm, MyRank, NRanks, row_starts, col_starts,
                             i_diag, j_diag, i_offd, j_offd,
                             cmap, offd_counter);
   }
   else
   {
      // Some shared dofs will be linear combinations of others
      HYPRE_BigInt ldof  = GetVSize();
      HYPRE_BigInt ltdof = TrueVSize();

      HYPRE_BigInt gdof  = -1;
      HYPRE_BigInt gtdof = -1;

      MPI_Allreduce(&ldof, &gdof, 1, HYPRE_MPI_BIG_INT, MPI_SUM, MyComm);
      MPI_Allreduce(&ltdof, &gtdof, 1, HYPRE_MPI_BIG_INT, MPI_SUM, MyComm);

      // Ensure face orientations have been communicated
      pmesh->ExchangeFaceNbrData();

      // Locate and count non-zeros in off-diagonal portion of P
      int nnz_offd = 0;
      Array<int> ldsize(ldof); ldsize = 0;
      Array<int> ltori(ldof);  ltori = 0; // Local triangle orientations
      {
         int ngrps = pmesh->GetNGroups();
         int nedofs = fec->DofForGeometry(Geometry::SEGMENT);
         Array<int> sdofs;
         for (int g = 1; g < ngrps; g++)
         {
            if (pmesh->gtopo.IAmMaster(g))
            {
               continue;
            }
            for (int ei=0; ei<pmesh->GroupNEdges(g); ei++)
            {
               this->GetSharedEdgeDofs(g, ei, sdofs);
               for (int i=0; i<sdofs.Size(); i++)
               {
                  int ind = (sdofs[i]>=0) ? sdofs[i] : (-sdofs[i]-1);
                  if (ldsize[ind] == 0) { nnz_offd++; }
                  ldsize[ind] = 1;
               }
            }
            for (int fi=0; fi<pmesh->GroupNTriangles(g); fi++)
            {
               int face, ori, info1, info2;
               pmesh->GroupTriangle(g, fi, face, ori);
               pmesh->GetFaceInfos(face, &info1, &info2);
               this->GetSharedTriangleDofs(g, fi, sdofs);
               for (int i=0; i<3*nedofs; i++)
               {
                  int ind = (sdofs[i]>=0) ? sdofs[i] : (-sdofs[i]-1);
                  if (ldsize[ind] == 0) { nnz_offd++; }
                  ldsize[ind] = 1;
               }
               for (int i=3*nedofs; i<sdofs.Size(); i++)
               {
                  if (ldsize[sdofs[i]] == 0) { nnz_offd += 2; }
                  ldsize[sdofs[i]] = 2;
                  ltori[sdofs[i]]  = info2 % 64;
               }
            }
            for (int fi=0; fi<pmesh->GroupNQuadrilaterals(g); fi++)
            {
               this->GetSharedQuadrilateralDofs(g, fi, sdofs);
               for (int i=0; i<sdofs.Size(); i++)
               {
                  int ind = (sdofs[i]>=0) ? sdofs[i] : (-sdofs[i]-1);
                  if (ldsize[ind] == 0) { nnz_offd++; }
                  ldsize[ind] = 1;
               }
            }
         }
      }

      HYPRE_Int *i_diag = Memory<HYPRE_Int>(ldof+1);
      HYPRE_Int *j_diag = Memory<HYPRE_Int>(ltdof);
      real_t    *d_diag = Memory<real_t>(ltdof);
      int diag_counter;

      HYPRE_Int *i_offd = Memory<HYPRE_Int>(ldof+1);
      HYPRE_Int *j_offd = Memory<HYPRE_Int>(nnz_offd);
      real_t    *d_offd = Memory<real_t>(nnz_offd);
      int offd_counter;

      HYPRE_BigInt *cmap   = Memory<HYPRE_BigInt>(ldof-ltdof);

      HYPRE_BigInt *col_starts = GetTrueDofOffsets();
      HYPRE_BigInt *row_starts = GetDofOffsets();

      Array<Pair<HYPRE_BigInt, int> > cmap_j_offd(ldof-ltdof);

      i_diag[0] = i_offd[0] = 0;
      diag_counter = offd_counter = 0;
      int offd_col_counter = 0;
      for (int i = 0; i < ldof; i++)
      {
         int ltdofi = GetLocalTDofNumber(i);
         if (ltdofi >= 0)
         {
            j_diag[diag_counter]   = ltdofi;
            d_diag[diag_counter++] = 1.0;
         }
         else
         {
            if (ldsize[i] == 1)
            {
               cmap_j_offd[offd_col_counter].one = GetGlobalTDofNumber(i);
               cmap_j_offd[offd_col_counter].two = offd_counter;
               offd_counter++;
               offd_col_counter++;
            }
            else
            {
               cmap_j_offd[offd_col_counter].one = GetGlobalTDofNumber(i);
               cmap_j_offd[offd_col_counter].two = offd_counter;
               offd_counter += 2;
               offd_col_counter++;
               i_diag[i+1] = diag_counter;
               i_offd[i+1] = offd_counter;
               i++;
               cmap_j_offd[offd_col_counter].one = GetGlobalTDofNumber(i);
               cmap_j_offd[offd_col_counter].two = offd_counter;
               offd_counter += 2;
               offd_col_counter++;
            }
         }
         i_diag[i+1] = diag_counter;
         i_offd[i+1] = offd_counter;
      }

      SortPairs<HYPRE_BigInt, int>(cmap_j_offd, offd_col_counter);

      for (int i = 0; i < nnz_offd; i++)
      {
         j_offd[i] = -1;
         d_offd[i] = 0.0;
      }

      for (int i = 0; i < offd_col_counter; i++)
      {
         cmap[i] = cmap_j_offd[i].one;
         j_offd[cmap_j_offd[i].two] = i;
      }

      for (int i = 0; i < ldof; i++)
      {
         if (i_offd[i+1] == i_offd[i] + 1)
         {
            d_offd[i_offd[i]] = 1.0;
         }
         else if (i_offd[i+1] == i_offd[i] + 2)
         {
            const real_t *T =
               ND_DofTransformation::GetFaceTransform(ltori[i]).GetData();
            j_offd[i_offd[i] + 1] = j_offd[i_offd[i]] + 1;
            d_offd[i_offd[i]] = T[0]; d_offd[i_offd[i] + 1] = T[2];
            i++;
            j_offd[i_offd[i] + 1] = j_offd[i_offd[i]];
            j_offd[i_offd[i]] = j_offd[i_offd[i] + 1] - 1;
            d_offd[i_offd[i]] = T[1]; d_offd[i_offd[i] + 1] = T[3];
         }
      }

      P = new HypreParMatrix(MyComm, gdof, gtdof, row_starts, col_starts,
                             i_diag, j_diag, d_diag, i_offd, j_offd, d_offd,
                             offd_col_counter, cmap);
   }

   SparseMatrix Pdiag;
   P->GetDiag(Pdiag);
   R = Transpose(Pdiag);
}

HypreParMatrix *ParFiniteElementSpace::GetPartialConformingInterpolation()
{
   HypreParMatrix *P_pc;
   Array<HYPRE_BigInt> P_pc_row_starts, P_pc_col_starts;
   BuildParallelConformingInterpolation(&P_pc, NULL, P_pc_row_starts,
                                        P_pc_col_starts, NULL, true);
   P_pc->CopyRowStarts();
   P_pc->CopyColStarts();
   return P_pc;
}

void ParFiniteElementSpace::DivideByGroupSize(real_t *vec)
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

   // Make sure that processors without boundary elements mark
   // their boundary dofs (if they have any).
   Synchronize(ess_dofs);
}

void ParFiniteElementSpace::GetEssentialTrueDofs(const Array<int>
                                                 &bdr_attr_is_ess,
                                                 Array<int> &ess_tdof_list,
                                                 int component) const
{
   Array<int> ess_dofs, true_ess_dofs;

   GetEssentialVDofs(bdr_attr_is_ess, ess_dofs, component);
   GetRestrictionMatrix()->BooleanMult(ess_dofs, true_ess_dofs);

#ifdef MFEM_DEBUG
   // Verify that in boolean arithmetic: P^T ess_dofs = R ess_dofs.
   Array<int> true_ess_dofs2(true_ess_dofs.Size());
   auto Pt = std::unique_ptr<HypreParMatrix>(Dof_TrueDof_Matrix()->Transpose());

   const int *ess_dofs_data = ess_dofs.HostRead();
   Pt->BooleanMult(1, ess_dofs_data, 0, true_ess_dofs2);
   int counter = 0;
   const int *ted = true_ess_dofs.HostRead();
   std::string error_msg = "failed dof: ";
   for (int i = 0; i < true_ess_dofs.Size(); i++)
   {
      if (bool(ted[i]) != bool(true_ess_dofs2[i]))
      {
         error_msg += std::to_string(i) += "(R ";
         error_msg += std::to_string(bool(ted[i])) += " P^T ";
         error_msg += std::to_string(bool(true_ess_dofs2[i])) += ") ";
         ++counter;
      }
   }
   MFEM_ASSERT(R->Height() == P->Width(), "!");
   MFEM_ASSERT(R->Width() == P->Height(), "!");
   MFEM_ASSERT(R->Width() == ess_dofs.Size(), "!");
   MFEM_VERIFY(counter == 0, "internal MFEM error: counter = " << counter
               << ", rank = " << MyRank << ", " << error_msg);
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

HYPRE_BigInt ParFiniteElementSpace::GetGlobalTDofNumber(int ldof) const
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

HYPRE_BigInt ParFiniteElementSpace::GetGlobalScalarTDofNumber(int sldof)
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

HYPRE_BigInt ParFiniteElementSpace::GetMyDofOffset() const
{
   return HYPRE_AssumedPartitionCheck() ? dof_offsets[0] : dof_offsets[MyRank];
}

HYPRE_BigInt ParFiniteElementSpace::GetMyTDofOffset() const
{
   return HYPRE_AssumedPartitionCheck()? tdof_offsets[0] : tdof_offsets[MyRank];
}

const Operator *ParFiniteElementSpace::GetProlongationMatrix() const
{
   if (Conforming())
   {
      if (Pconf) { return Pconf; }

      if (nd_strias) { return Dof_TrueDof_Matrix(); }

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

const Operator *ParFiniteElementSpace::GetRestrictionOperator() const
{
   if (Conforming())
   {
      if (Rconf) { return Rconf; }

      if (NRanks == 1)
      {
         R_transpose.reset(new IdentityOperator(GetTrueVSize()));
      }
      else
      {
         if (!Device::Allows(Backend::DEVICE_MASK))
         {
            R_transpose.reset(new ConformingProlongationOperator(*this, true));
         }
         else
         {
            R_transpose.reset(
               new DeviceConformingProlongationOperator(*this, true));
         }
      }
      Rconf = new TransposeOperator(*R_transpose);
      return Rconf;
   }
   else
   {
      Dof_TrueDof_Matrix();
      if (!R_transpose) { R_transpose.reset(new TransposeOperator(R)); }
      return R;
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
      int *ldofs_fn  = send_face_nbr_ldof.GetRow(fn);
      int  j_end     = send_I[send_el_off[fn+1]];

      for (int i = 0; i < num_ldofs; i++)
      {
         int ldof = (ldofs_fn[i] >= 0 ? ldofs_fn[i] : -1-ldofs_fn[i]);
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
   Array<HYPRE_BigInt> dof_face_nbr_offsets(num_face_nbrs);
   HYPRE_BigInt my_dof_offset = GetMyDofOffset();
   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      int nbr_rank = pmesh->GetFaceNbrRank(fn);
      int tag = 0;

      MPI_Isend(&my_dof_offset, 1, HYPRE_MPI_BIG_INT, nbr_rank, tag,
                MyComm, &send_requests[fn]);

      MPI_Irecv(&dof_face_nbr_offsets[fn], 1, HYPRE_MPI_BIG_INT, nbr_rank, tag,
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
   int i, Array<int> &vdofs, DofTransformation &doftrans) const
{
   face_nbr_element_dof.GetRow(i, vdofs);

   if (DoFTransArray[GetFaceNbrFE(i)->GetGeomType()])
   {
      Array<int> F, Fo;
      pmesh->GetFaceNbrElementFaces(pmesh->GetNE() + i, F, Fo);
      doftrans.SetDofTransformation(
         *DoFTransArray[GetFaceNbrFE(i)->GetGeomType()]);
      doftrans.SetFaceOrientations(Fo);
      doftrans.SetVDim(vdim, ordering);
   }
}

DofTransformation *ParFiniteElementSpace::GetFaceNbrElementVDofs(
   int i, Array<int> &vdofs) const
{
   DoFTrans.SetDofTransformation(NULL);
   GetFaceNbrElementVDofs(i, vdofs, DoFTrans);
   return DoFTrans.GetDofTransformation() ? &DoFTrans : NULL;
}

void ParFiniteElementSpace::GetFaceNbrFaceVDofs(int i, Array<int> &vdofs) const
{
   // Works for NC mesh where 'i' is an index returned by
   // ParMesh::GetSharedFace() such that i >= Mesh::GetNumFaces(), i.e. 'i' is
   // the index of a ghost face.
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
   // Works for NC mesh where 'i' is an index returned by
   // ParMesh::GetSharedFace() such that i >= Mesh::GetNumFaces(), i.e. 'i' is
   // the index of a ghost face.
   // Works in tandem with GetFaceNbrFaceVDofs() defined above.

   MFEM_ASSERT(Nonconforming() && !NURBSext, "");
   Geometry::Type face_geom = pmesh->GetFaceGeometry(i);
   return fec->FiniteElementForGeometry(face_geom);
}

void ParFiniteElementSpace::Lose_Dof_TrueDof_Matrix()
{
   P -> StealData();
#if MFEM_HYPRE_VERSION <= 22200
   hypre_ParCSRMatrix *csrP = (hypre_ParCSRMatrix*)(*P);
   hypre_ParCSRMatrixOwnsRowStarts(csrP) = 1;
   hypre_ParCSRMatrixOwnsColStarts(csrP) = 1;
   dof_offsets.LoseData();
   tdof_offsets.LoseData();
#else
   dof_offsets.DeleteAll();
   tdof_offsets.DeleteAll();
#endif
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
            first = nvdofs + nedofs + FirstFaceDof(index);
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
            return nvdofs + nedofs + FirstFaceDof(index) + edof;
         }
         else // ghost face
         {
            index -= ghost;
            int stride = fec->DofForGeometry(Geometry::SQUARE);
            return ndofs + ngvdofs + ngedofs + index*stride + edof;
         }
   }
}

static int bisect(const int* array, int size, int value)
{
   const int* end = array + size;
   const int* pos = std::upper_bound(array, end, value);
   MFEM_VERIFY(pos != array, "value not found");
   if (pos == end)
   {
      MFEM_VERIFY(*(array+size - 1) == value, "Last entry must be exact")
   }
   return pos - array - 1;
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
         if (uni_fdof >= 0) // uniform faces
         {
            int nf = fec->DofForGeometry(pmesh->GetTypicalFaceGeometry());
            index = dof / nf, edof = dof % nf;
         }
         else // mixed faces or var-order space
         {
            const Table &table = var_face_dofs;

            MFEM_ASSERT(table.Size() > 0, "");
            int jpos = bisect(table.GetJ(), table.Size_of_connections(), dof);
            index = bisect(table.GetI(), table.Size(), jpos);
            edof = dof - table.GetRow(index)[0];
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
   HYPRE_BigInt column;
   int stride;
   real_t value;

   PMatrixElement(HYPRE_BigInt col = 0, int str = 0, real_t val = 0)
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
   void AddRow(const PMatrixRow &other, real_t coef)
   {
      elems.reserve(elems.size() + other.elems.size());
      for (const PMatrixElement &oei : other.elems)
      {
         elems.emplace_back(oei.column, oei.stride, coef * oei.value);
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

   void write(std::ostream &os, real_t sign) const
   {
      bin_io::write<int>(os, static_cast<int>(elems.size()));
      for (unsigned i = 0; i < elems.size(); i++)
      {
         const PMatrixElement &e = elems[i];
         bin_io::write<HYPRE_BigInt>(os, e.column);
         bin_io::write<int>(os, e.stride);
         bin_io::write<real_t>(os, e.value * sign);
      }
   }

   void read(std::istream &is, real_t sign)
   {
      elems.resize(bin_io::read<int>(is));
      for (unsigned i = 0; i < elems.size(); i++)
      {
         PMatrixElement &e = elems[i];
         e.column = bin_io::read<HYPRE_BigInt>(is);
         e.stride = bin_io::read<int>(is);
         e.value = bin_io::read<real_t>(is) * sign;
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
   };

   NeighborRowMessage() : pncmesh(NULL) {}

   void AddRow(int entity, int index, int edof, GroupId group,
               const PMatrixRow &row)
   {
      rows.emplace_back(entity, index, edof, group, row);
   }

   const std::vector<RowInfo>& GetRows() const { return rows; }

   void SetNCMesh(ParNCMesh* pnc) { pncmesh = pnc; }
   void SetFEC(const FiniteElementCollection* fec_) { this->fec = fec_; }

   typedef std::map<int, NeighborRowMessage> Map;

protected:
   std::vector<RowInfo> rows;

   ParNCMesh *pncmesh;
   const FiniteElementCollection* fec;

   /// Encode a NeighborRowMessage for sending via MPI.
   void Encode(int rank) override;
   /// Decode a NeighborRowMessage received via MPI.
   void Decode(int rank) override;
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
      const MeshId &id = *pncmesh->GetNCList(ri.entity).GetMeshIdAndType(ri.index).id;
      ent_ids[ri.entity].Append(id);
      row_idx[ri.entity].Append(i);
      group_ids[ri.entity].Append(ri.group);
   }

   Array<GroupId> all_group_ids;
   all_group_ids.Reserve(static_cast<int>(rows.size()));
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
         real_t s = 1.0;
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

   // read rows ent = {0,1,2} means vertex, edge and face entity
   for (int ent = 0, gi = 0; ent < 3; ent++)
   {
      // extract the vertex list, edge list or face list.
      const Array<MeshId> &ids = ent_ids[ent];
      for (int i = 0; i < ids.Size(); i++)
      {
         const MeshId &id = ids[i];
         // read the particular element dof value off the stream.
         int edof = bin_io::read<int>(stream);

         // Handle orientation and sign change. This flips the sign on dofs
         // where necessary, and for edges and faces also reorders if flipped,
         // i.e. an edge with 1 -> 2 -> 3 -> 4 might become -4 -> -3 -> -2 -> -1
         // This cannot treat all face dofs, as they can have rotations and
         // reflections.
         const int *ind = nullptr;
         Geometry::Type geom = Geometry::Type::INVALID;
         if (ent == 1)
         {
            // edge NC orientation is element defined.
            int eo = pncmesh->GetEdgeNCOrientation(id);
            ind = fec->DofOrderForOrientation(Geometry::SEGMENT, eo);
         }
         else if (ent == 2)
         {
            geom = pncmesh->GetFaceGeometry(id.index);
            int fo = pncmesh->GetFaceOrientation(id.index);
            ind = fec->DofOrderForOrientation(geom, fo);
         }
         // Tri faces with second order basis have dofs that must be processed
         // in pairs, as the doftransformation is not diagonal.
         const bool process_dof_pairs = (ent == 2 &&
                                         fec->GetContType() == FiniteElementCollection::TANGENTIAL
                                         && !Geometry::IsTensorProduct(geom));

#ifdef MFEM_DEBUG_PMATRIX
         mfem::out << "Rank " << pncmesh->MyRank << " receiving from " << rank
                   << ": ent " << ent << ", index " << id.index
                   << ", edof " << edof << " (id " << id.element << "/"
                   << int(id.local) << ")" << std::endl;
#endif

         // If edof arrived with a negative index, flip it, and the scaling.
         real_t s = (edof < 0) ? -1.0 : 1.0;
         edof = (edof < 0) ? -1 - edof : edof;
         if (ind && (edof = ind[edof]) < 0)
         {
            edof = -1 - edof;
            s *= -1.0;
         }

         // Create a row for this entity, recording the index of the mesh
         // element
         rows.emplace_back(ent, id.index, edof, group_ids[gi++]);
         rows.back().row.read(stream, s);

#ifdef MFEM_DEBUG_PMATRIX
         mfem::out << "Rank " << pncmesh->MyRank << " receiving from " << rank
                   << ": ent " << rows.back().entity << ", index "
                   << rows.back().index << ", edof " << rows.back().edof
                   << std::endl;
#endif

         if (process_dof_pairs)
         {
            // ND face dofs need to be processed together, as the transformation
            // is given by a 2x2 matrix, so we manually apply an extra increment
            // to the loop counter and add in a new row. Once these rows are
            // placed, they represent the Identity transformation. To map across
            // the processor boundary, we also need to apply a Primal
            // Transformation (see doftrans.hpp) to a notional "global dof"
            // orientation. For simplicity we perform the action of these 2x2
            // matrices manually using the AddRow capability, followed by a
            // Collapse.

            // To perform the operations, we add and subtract initial versions
            // of the rows, that represent [1 0; 0 1] in row major notation. The
            // first row represents the 1 at (0,0) in [1 0; 0 1] The second row
            // represents the 1 at (1,1) in [1 0; 0 1]

            // We can safely bind this reference as rows was reserved above so
            // there is no hidden copying that could result in a dangling
            // reference.
            auto &first_row = rows.back().row;
            // This is the first "fundamental unit" used in the transformation.
            const auto initial_first_row = first_row;
            // Extract the next dof too, and apply any dof order transformation
            // expected.
            const MeshId &next_id = ids[++i];
            const int fo = pncmesh->GetFaceOrientation(next_id.index);
            ind = fec->DofOrderForOrientation(geom, fo);
            edof = bin_io::read<int>(stream);

            // If edof arrived with a negative index, flip it, and the scaling.
            s = (edof < 0) ? -1.0 : 1.0;
            edof = (edof < 0) ? -1 - edof : edof;
            if (ind && (edof = ind[edof]) < 0)
            {
               edof = -1 - edof;
               s *= -1.0;
            }

            rows.emplace_back(ent, next_id.index, edof, group_ids[gi++]);
            rows.back().row.read(stream, s);
            auto &second_row = rows.back().row;

            // This is the second "fundamental unit" used in the transformation.
            const auto initial_second_row = second_row;

            // Transform the received dofs by the primal transform. This is
            // because within mfem as a face is visited its orientation is
            // assigned to match the element that visited it first. Thus on
            // processor boundaries, the transform will always be identity going
            // into the element. However, the sending processor also thought the
            // face orientation was zero, so it has sent the information in a
            // different orientation. To map onto the local orientation
            // definition, extract the orientation of the sending rank (the
            // lower rank face defines the orientation fo), then apply the
            // transform to the dependencies. The action of this transform on
            // the dependencies is performed by adding scaled versions of the
            // original two rows (which by the mfem assumption of face
            // orientation, represent the identity transform).
            const real_t *T =
               ND_DofTransformation::GetFaceTransform(fo).GetData();

            MFEM_ASSERT(fo != 2 &&
                        fo != 4, "This code branch is ambiguous for face orientations 2 and 4."
                        " Please report this mesh for further testing.\n");

            first_row.AddRow(initial_first_row, T[0] - 1.0);   // (0,0)
            first_row.AddRow(initial_second_row, T[2]);        // (0,1)
            second_row.AddRow(initial_first_row, T[1]);        // (1,0)
            second_row.AddRow(initial_second_row, T[3] - 1.0); // (1,1)

            first_row.Collapse();
            second_row.Collapse();
         }
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

   for (const auto &rank : pncmesh->GetGroup(group_id))
   {
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
::BuildParallelConformingInterpolation(HypreParMatrix **P_, SparseMatrix **R_,
                                       Array<HYPRE_BigInt> &dof_offs,
                                       Array<HYPRE_BigInt> &tdof_offs,
                                       Array<int> *dof_tdof,
                                       bool partial) const
{
   const bool dg = (nvdofs == 0 && nedofs == 0 && nfdofs == 0);

#ifdef MFEM_PMATRIX_STATS
   n_msgs_sent = n_msgs_recv = 0;
   n_rows_sent = n_rows_recv = n_rows_fwd = 0;
#endif

   // *** STEP 1: build master-slave dependency lists ***

   const int total_dofs = ndofs + ngdofs;
   SparseMatrix deps(ndofs, total_dofs);

   if (!dg && !partial)
   {
      Array<int> master_dofs, slave_dofs;

      // loop through *all* master edges/faces, constrain their slaves
      for (int entity = 0; entity <= 2; entity++)
      {
         const NCMesh::NCList &list = pncmesh->GetNCList(entity);
         if (list.masters.Size() == 0) { continue; }

         IsoparametricTransformation T;
         DenseMatrix I;

         // process masters that we own or that affect our edges/faces
         for (const auto &mf : list.masters)
         {
            // get master DOFs
            if (pncmesh->IsGhost(entity, mf.index))
            {
               GetGhostDofs(entity, mf, master_dofs);
            }
            else
            {
               GetEntityDofs(entity, mf.index, master_dofs, mf.Geom());
            }

            if (master_dofs.Size() == 0) { continue; }

            const FiniteElement * const fe = fec->FiniteElementForGeometry(mf.Geom());
            if (fe == nullptr) { continue; }

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

               constexpr int variant = 0; // TODO parallel var-order
               GetEntityDofs(entity, sf.index, slave_dofs, mf.Geom(), variant);
               if (!slave_dofs.Size()) { continue; }

               list.OrientedPointMatrix(sf, T.GetPointMat());
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

      auto initialize_group_and_owner = [&dof_group, &dof_owner, &dofs,
                                                     this](int entity, const MeshId &id)
      {
         if (id.index < 0) { return; }

         GroupId owner = pncmesh->GetEntityOwnerId(entity, id.index);
         GroupId group = pncmesh->GetEntityGroupId(entity, id.index);

         GetBareDofs(entity, id.index, dofs);

         for (auto dof : dofs)
         {
            dof_owner[dof] = owner;
            dof_group[dof] = group;
         }
      };

      // initialize dof_group[], dof_owner[] in sequence
      for (int entity : {0,1,2})
      {
         for (const auto &id : pncmesh->GetNCList(entity).conforming)
         {
            initialize_group_and_owner(entity, id);
         }
         for (const auto &id : pncmesh->GetNCList(entity).masters)
         {
            initialize_group_and_owner(entity, id);
         }
         for (const auto &id : pncmesh->GetNCList(entity).slaves)
         {
            initialize_group_and_owner(entity, id);
         }
      }
   }

   // *** STEP 3: count true DOFs and calculate P row/column partitions ***

   Array<bool> finalized(total_dofs);
   finalized = false;

   // DOFs that stayed independent and are ours are true DOFs
   int num_true_dofs = 0;
   for (int i = 0; i < ndofs; ++i)
   {
      if (dof_owner[i] == 0 && deps.RowSize(i) == 0)
      {
         ++num_true_dofs;
         finalized[i] = true;
      }
   }

#ifdef MFEM_DEBUG_PMATRIX
   // Helper for dumping diagnostics on one dof
   auto dof_diagnostics = [&](int dof, bool print_diagnostic)
   {
      const auto &comm_group = pncmesh->GetGroup(dof_group[dof]);
      std::stringstream msg;
      msg << std::boolalpha;
      msg << "R" << Mpi::WorldRank() << " dof " << dof
          << " owner_rank " << pncmesh->GetGroup(dof_owner[dof])[0] << " CommGroup {";
      for (const auto &x : comm_group)
      {
         msg << x << ' ';
      }
      msg << "} finalized " << finalized[dof];

      Array<int> cols;
      if (dof < ndofs)
      {
         Vector row;
         deps.GetRow(dof, cols, row);
         msg << " deps cols {";
         for (const auto &x : cols)
         {
            msg << x << ' ';
         }
         msg << '}';
      }

      int entity, index, edof;
      UnpackDof(dof, entity, index, edof);
      msg << " entity " << entity << " index " << index << " edof " << edof;
      return msg.str();
   };
#endif

   // calculate global offsets
   HYPRE_BigInt loc_sizes[2] = { ndofs*vdim, num_true_dofs*vdim };
   Array<HYPRE_BigInt>* offsets[2] = { &dof_offs, &tdof_offs };
   pmesh->GenerateOffsets(2, loc_sizes, offsets); // calls MPI_Scan, MPI_Bcast

   HYPRE_BigInt my_tdof_offset =
      tdof_offs[HYPRE_AssumedPartitionCheck() ? 0 : MyRank];

   if (R_)
   {
      // initialize the restriction matrix (also parallel but block-diagonal)
      *R_ = new SparseMatrix(num_true_dofs*vdim, ndofs*vdim);
   }
   if (dof_tdof)
   {
      dof_tdof->SetSize(ndofs*vdim);
      *dof_tdof = -1;
   }

   std::vector<PMatrixRow> pmatrix(total_dofs);

   const bool bynodes = (ordering == Ordering::byNODES);
   const int vdim_factor = bynodes ? 1 : vdim;
   const int dof_stride = bynodes ? ndofs : 1;
   const int tdof_stride = bynodes ? num_true_dofs : 1;

   // big container for all messages we send (the list is for iterations)
   std::list<NeighborRowMessage::Map> send_msg;
   send_msg.emplace_back();

   // put identity in P and R for true DOFs, set ldof_ltdof
   for (int dof = 0, tdof = 0; dof < ndofs; dof++)
   {
      if (finalized[dof])
      {
         pmatrix[dof].elems.emplace_back(
            my_tdof_offset + vdim_factor*tdof, tdof_stride, 1.);

         // prepare messages to neighbors with identity rows
         if (dof_group[dof] != 0)
         {
            ScheduleSendRow(pmatrix[dof], dof, dof_group[dof], send_msg.back());
         }

         for (int vd = 0; vd < vdim; vd++)
         {
            const int vdof = dof*vdim_factor + vd*dof_stride;
            const int vtdof = tdof*vdim_factor + vd*tdof_stride;

            if (R_) { (*R_)->Add(vtdof, vdof, 1.0); }
            if (dof_tdof) { (*dof_tdof)[vdof] = vtdof; }
         }
         ++tdof;
      }
   }

   // send identity rows
   NeighborRowMessage::IsendAll(send_msg.back(), MyComm);
#ifdef MFEM_PMATRIX_STATS
   n_msgs_sent += send_msg.back().size();
#endif

   if (R_) { (*R_)->Finalize(); }

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
         send_msg.emplace_back();
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

         for (const auto &ri : recv_msg.GetRows())
         {
            const int dof = PackDof(ri.entity, ri.index, ri.edof);
            pmatrix[dof] = ri.row;

            if (dof < ndofs && !finalized[dof]) { ++num_finalized; }
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
            const bool owned = (dof_owner[dof] == 0);
            if (!finalized[dof]
                && owned
                && DofFinalizable(dof, finalized, deps))
            {
               int ent, idx, edof;
               UnpackDof(dof, ent, idx, edof);

               const int* dep_col = deps.GetRowColumns(dof);
               const real_t* dep_coef = deps.GetRowEntries(dof);
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
               ++num_finalized;
               done = false;

               // send row to neighbors who need it
               const bool shared = (dof_group[dof] != 0);
               if (shared)
               {
                  ScheduleSendRow(pmatrix[dof], dof, dof_group[dof],
                                  send_msg.back());
               }
            }
         }
      }

#ifdef MFEM_DEBUG_PMATRIX
      static int dump = 0;
      if (dump < 10)
      {
         char fname[100];
         snprintf(fname, 100, "dofs%02d.txt", MyRank);
         std::ofstream f(fname);
         DebugDumpDOFs(f, deps, dof_group, dof_owner, finalized);
         dump++;
      }
#endif

      // send current batch of messages
      NeighborRowMessage::IsendAll(send_msg.back(), MyComm);
#ifdef MFEM_PMATRIX_STATS
      n_msgs_sent += send_msg.back().size();
#endif
   }

   if (P_)
   {
      *P_ = MakeVDimHypreMatrix(pmatrix, ndofs, num_true_dofs,
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
   for (auto &msg : send_msg)
   {
      NeighborRowMessage::WaitAllSent(msg);
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
                << real_t(glob_rounds)/NRanks << " rounds, "
                << real_t(glob_msgs_sent)/NRanks << " msgs sent, "
                << real_t(glob_msgs_recv)/NRanks << " msgs recv, "
                << real_t(glob_rows_sent)/NRanks << " rows sent, "
                << real_t(glob_rows_recv)/NRanks << " rows recv, "
                << real_t(glob_rows_fwd)/NRanks << " rows forwarded."
                << std::endl;
   }
#endif

   return num_true_dofs*vdim;
}


HypreParMatrix* ParFiniteElementSpace
::MakeVDimHypreMatrix(const std::vector<PMatrixRow> &rows,
                      int local_rows, int local_cols,
                      Array<HYPRE_BigInt> &row_starts,
                      Array<HYPRE_BigInt> &col_starts) const
{
   bool assumed = HYPRE_AssumedPartitionCheck();
   bool bynodes = (ordering == Ordering::byNODES);

   HYPRE_BigInt first_col = col_starts[assumed ? 0 : MyRank];
   HYPRE_BigInt next_col = col_starts[assumed ? 1 : MyRank+1];

   // count nonzeros in diagonal/offdiagonal parts
   HYPRE_Int nnz_diag = 0, nnz_offd = 0;
   std::map<HYPRE_BigInt, int> col_map;
   for (int i = 0; i < local_rows; i++)
   {
      for (unsigned j = 0; j < rows[i].elems.size(); j++)
      {
         const PMatrixElement &elem = rows[i].elems[j];
         HYPRE_BigInt col = elem.column;
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
   HYPRE_BigInt *cmap = Memory<HYPRE_BigInt>(static_cast<int>(col_map.size()));
   int offd_col = 0;
   for (auto it = col_map.begin(); it != col_map.end(); ++it)
   {
      cmap[offd_col] = it->first;
      it->second = offd_col++;
   }

   HYPRE_Int *I_diag = Memory<HYPRE_Int>(vdim*local_rows + 1);
   HYPRE_Int *I_offd = Memory<HYPRE_Int>(vdim*local_rows + 1);

   HYPRE_Int *J_diag = Memory<HYPRE_Int>(nnz_diag);
   HYPRE_Int *J_offd = Memory<HYPRE_Int>(nnz_offd);

   real_t *A_diag = Memory<real_t>(nnz_diag);
   real_t *A_offd = Memory<real_t>(nnz_offd);

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
                             static_cast<HYPRE_Int>(col_map.size()), cmap);
}

template <typename int_type>
static int_type* make_i_array(int nrows)
{
   int_type *I = Memory<int_type>(nrows+1);
   for (int i = 0; i <= nrows; i++) { I[i] = -1; }
   return I;
}

template <typename int_type>
static int_type* make_j_array(int_type* I, int nrows)
{
   int nnz = 0;
   for (int i = 0; i < nrows; i++)
   {
      if (I[i] >= 0) { nnz++; }
   }
   int_type *J = Memory<int_type>(nnz);

   I[nrows] = -1;
   for (int i = 0, k = 0; i <= nrows; i++)
   {
      int_type col = I[i];
      I[i] = k;
      if (col >= 0) { J[k++] = col; }
   }
   return J;
}

HypreParMatrix*
ParFiniteElementSpace::RebalanceMatrix(int old_ndofs,
                                       const Table* old_elem_dof,
                                       const Table* old_elem_fos)
{
   MFEM_VERIFY(Nonconforming(), "Only supported for nonconforming meshes.");
   MFEM_VERIFY(old_dof_offsets.Size(), "ParFiniteElementSpace::Update needs to "
               "be called before ParFiniteElementSpace::RebalanceMatrix");

   HYPRE_BigInt old_offset = HYPRE_AssumedPartitionCheck()
                             ? old_dof_offsets[0] : old_dof_offsets[MyRank];

   // send old DOFs of elements we used to own
   ParNCMesh* old_pncmesh = pmesh->pncmesh;
   old_pncmesh->SendRebalanceDofs(old_ndofs, *old_elem_dof, old_offset, this);

   Array<int> dofs;
   int vsize = GetVSize();

   const Array<int> &old_index = old_pncmesh->GetRebalanceOldIndex();
   MFEM_VERIFY(old_index.Size() == pmesh->GetNE(),
               "Mesh::Rebalance was not called before "
               "ParFiniteElementSpace::RebalanceMatrix");

   // prepare the local (diagonal) part of the matrix
   HYPRE_Int* i_diag = make_i_array<HYPRE_Int>(vsize);
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
   old_pncmesh->RecvRebalanceDofs(new_elements, old_remote_dofs);

   // create the offdiagonal part of the matrix
   HYPRE_BigInt* i_offd = make_i_array<HYPRE_BigInt>(vsize);
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
   HYPRE_BigInt* j_offd = make_j_array(i_offd, vsize);

#ifndef HYPRE_MIXEDINT
   HYPRE_Int *i_offd_hi = i_offd;
#else
   // Copy of i_offd array as array of HYPRE_Int
   HYPRE_Int *i_offd_hi = Memory<HYPRE_Int>(vsize + 1);
   std::copy(i_offd, i_offd + vsize + 1, i_offd_hi);
   Memory<HYPRE_BigInt>(i_offd, vsize + 1, true).Delete();
#endif

   // create the offd column map
   int offd_cols = i_offd_hi[vsize];
   Array<Pair<HYPRE_BigInt, int> > cmap_offd(offd_cols);
   for (int i = 0; i < offd_cols; i++)
   {
      cmap_offd[i].one = j_offd[i];
      cmap_offd[i].two = i;
   }

#ifndef HYPRE_MIXEDINT
   HYPRE_Int *j_offd_hi = j_offd;
#else
   HYPRE_Int *j_offd_hi = Memory<HYPRE_Int>(offd_cols);
   Memory<HYPRE_BigInt>(j_offd, offd_cols, true).Delete();
#endif

   SortPairs<HYPRE_BigInt, int>(cmap_offd, offd_cols);

   HYPRE_BigInt* cmap = Memory<HYPRE_BigInt>(offd_cols);
   for (int i = 0; i < offd_cols; i++)
   {
      cmap[i] = cmap_offd[i].one;
      j_offd_hi[cmap_offd[i].two] = i;
   }

   HypreParMatrix *M;
   M = new HypreParMatrix(MyComm, MyRank, NRanks, dof_offsets, old_dof_offsets,
                          i_diag, j_diag, i_offd_hi, j_offd_hi, cmap, offd_cols);
   return M;
}


struct DerefDofMessage
{
   std::vector<HYPRE_BigInt> dofs;
   MPI_Request request;
};

HypreParMatrix*
ParFiniteElementSpace::ParallelDerefinementMatrix(int old_ndofs,
                                                  const Table* old_elem_dof,
                                                  const Table *old_elem_fos)
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

   ParNCMesh* old_pncmesh = pmesh->pncmesh;

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

   const CoarseFineTransformations &dtrans =
      old_pncmesh->GetDerefinementTransforms();
   const Array<int> &old_ranks = old_pncmesh->GetDerefineOldRanks();

   std::map<int, DerefDofMessage> messages;

   HYPRE_BigInt old_offset = HYPRE_AssumedPartitionCheck()
                             ? old_dof_offsets[0] : old_dof_offsets[MyRank];

   // communicate DOFs for derefinements that straddle processor boundaries,
   // note that this is infrequent due to the way elements are ordered
   for (int k = 0; k < dtrans.embeddings.Size(); k++)
   {
      const Embedding &emb = dtrans.embeddings[k];

      int fine_rank = old_ranks[k];
      int coarse_rank = (emb.parent < 0) ? (-1 - emb.parent)
                        : old_pncmesh->ElementRank(emb.parent);

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

         MPI_Isend(&msg.dofs[0], static_cast<int>(msg.dofs.size()), HYPRE_MPI_BIG_INT,
                   coarse_rank, 291, MyComm, &msg.request);
      }
      else if (coarse_rank == MyRank && fine_rank != MyRank)
      {
         MFEM_ASSERT(emb.parent >= 0, "");
         Geometry::Type geom = mesh->GetElementBaseGeometry(emb.parent);

         DerefDofMessage &msg = messages[k];
         msg.dofs.resize(ldof[geom]*vdim);

         MPI_Irecv(&msg.dofs[0], ldof[geom]*vdim, HYPRE_MPI_BIG_INT,
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

   bool is_dg = FEColl()->GetContType() == FiniteElementCollection::DISCONTINUOUS;

   for (int k = 0; k < dtrans.embeddings.Size(); k++)
   {
      const Embedding &emb = dtrans.embeddings[k];
      if (emb.parent < 0) { continue; }

      int coarse_rank = old_pncmesh->ElementRank(emb.parent);
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

               if (is_dg || !mark[m])
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

   std::map<HYPRE_BigInt, int> col_map;
   for (int k = 0; k < dtrans.embeddings.Size(); k++)
   {
      const Embedding &emb = dtrans.embeddings[k];
      if (emb.parent < 0) { continue; }

      int coarse_rank = old_pncmesh->ElementRank(emb.parent);
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
            HYPRE_BigInt* remote_dofs = &msg.dofs[vd*ldof[geom]];

            for (int i = 0; i < lR.Height(); i++)
            {
               if (!std::isfinite(lR(i, 0))) { continue; }

               int r = DofToVDof(dofs[i], vd);
               int m = (r >= 0) ? r : (-1 - r);

               if (is_dg || !mark[m])
               {
                  lR.GetRow(i, row);
                  MFEM_ASSERT(ldof[geom] == row.Size(), "");
                  for (int j = 0; j < ldof[geom]; j++)
                  {
                     if (row[j] == 0.0) { continue; } // NOTE: lR thresholded
                     int &lcol = col_map[remote_dofs[j]];
                     if (!lcol) { lcol = static_cast<int>(col_map.size()); }
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
   offd->SetWidth(static_cast<int>(col_map.size()));

   // create offd column mapping for use by hypre
   HYPRE_BigInt *cmap = Memory<HYPRE_BigInt>(offd->Width());
   for (auto it = col_map.begin(); it != col_map.end(); ++it)
   {
      cmap[it->second-1] = it->first;
   }

   // reorder offd columns so that 'cmap' is monotonic
   // NOTE: this is easier and probably faster (offd is small) than making
   // sure cmap is determined and sorted before the offd matrix is created
   {
      int width = offd->Width();
      Array<Pair<HYPRE_BigInt, int> > reorder(width);
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

   HypreParMatrix* new_R;
   new_R = new HypreParMatrix(MyComm, dof_offsets[nrk], old_dof_offsets[nrk],
                              dof_offsets, old_dof_offsets, diag, offd, cmap,
                              true);

   new_R->SetOwnerFlags(new_R->OwnsDiag(), new_R->OwnsOffd(), 1);

   return new_R;
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
   delete Rconf; Rconf = NULL;
   delete R; R = NULL;

   delete gcomm; gcomm = NULL;

   num_face_nbr_dofs = -1;
   face_nbr_element_dof.Clear();
   face_nbr_ldof.Clear();
   face_nbr_glob_dof_map.DeleteAll();
   send_face_nbr_ldof.Clear();
}

void ParFiniteElementSpace::CopyProlongationAndRestriction(
   const FiniteElementSpace &fes, const Array<int> *perm)
{
   const ParFiniteElementSpace *pfes
      = dynamic_cast<const ParFiniteElementSpace*>(&fes);
   MFEM_VERIFY(pfes != NULL, "");
   MFEM_VERIFY(P == NULL, "");
   MFEM_VERIFY(R == NULL, "");

   // Ensure R and P matrices are built
   pfes->Dof_TrueDof_Matrix();

   SparseMatrix *perm_mat = NULL, *perm_mat_tr = NULL;
   if (perm)
   {
      // Note: although n and fes.GetVSize() are typically equal, in
      // variable-order spaces they may differ, since nonconforming edges/faces
      // my have fictitious DOFs.
      int n = perm->Size();
      perm_mat = new SparseMatrix(n, fes.GetVSize());
      for (int i=0; i<n; ++i)
      {
         real_t s;
         int j = DecodeDof((*perm)[i], s);
         perm_mat->Set(i, j, s);
      }
      perm_mat->Finalize();
      perm_mat_tr = Transpose(*perm_mat);
   }

   if (pfes->P != NULL)
   {
      if (perm) { P = pfes->P->LeftDiagMult(*perm_mat); }
      else { P = new HypreParMatrix(*pfes->P); }
      nonconf_P = true;
   }
   else if (perm != NULL)
   {
      HYPRE_BigInt glob_nrows = GlobalVSize();
      HYPRE_BigInt glob_ncols = GlobalTrueVSize();
      HYPRE_BigInt *col_starts = GetTrueDofOffsets();
      HYPRE_BigInt *row_starts = GetDofOffsets();
      P = new HypreParMatrix(MyComm, glob_nrows, glob_ncols, row_starts,
                             col_starts, perm_mat);
      nonconf_P = true;
   }
   if (pfes->R != NULL)
   {
      if (perm) { R = Mult(*pfes->R, *perm_mat_tr); }
      else { R = new SparseMatrix(*pfes->R); }
   }
   else if (perm != NULL)
   {
      R = perm_mat_tr;
      perm_mat_tr = NULL;
   }

   delete perm_mat;
   delete perm_mat_tr;
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
   MFEM_VERIFY(!IsVariableOrder(),
               "Parallel variable order space not supported yet.");

   if (mesh->GetSequence() == mesh_sequence)
   {
      return; // no need to update, no-op
   }
   if (want_transform && mesh->GetSequence() != mesh_sequence + 1)
   {
      MFEM_ABORT("Error in update sequence. Space needs to be updated after "
                 "each mesh modification.");
   }

   if (NURBSext)
   {
      UpdateNURBS();
      return;
   }

   Table* old_elem_dof = NULL;
   Table* old_elem_fos = NULL;
   int old_ndofs;

   // save old DOF table
   if (want_transform)
   {
      old_elem_dof = elem_dof;
      old_elem_fos = elem_fos;
      elem_dof = NULL;
      elem_fos = NULL;
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
               Th.Reset(new RefinementOperator(this, old_elem_dof,
                                               old_elem_fos, old_ndofs));
               // The RefinementOperator takes ownership of 'old_elem_dofs', so
               // we no longer own it:
               old_elem_dof = NULL;
               old_elem_fos = NULL;
            }
            else
            {
               // calculate fully assembled matrix
               Th.Reset(RefinementMatrix(old_ndofs, old_elem_dof, old_elem_fos));
            }
            break;
         }

         case Mesh::DEREFINE:
         {
            Th.Reset(ParallelDerefinementMatrix(old_ndofs, old_elem_dof,
                                                old_elem_fos));
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
            Th.Reset(RebalanceMatrix(old_ndofs, old_elem_dof, old_elem_fos));
            break;
         }

         default:
            break;
      }

      delete old_elem_dof;
      delete old_elem_fos;
   }
}

void ParFiniteElementSpace::UpdateMeshPointer(Mesh *new_mesh)
{
   ParMesh *new_pmesh = dynamic_cast<ParMesh*>(new_mesh);
   MFEM_VERIFY(new_pmesh != NULL,
               "ParFiniteElementSpace::UpdateMeshPointer(...) must be a ParMesh");
   mesh = new_mesh;
   pmesh = new_pmesh;
}

ConformingProlongationOperator::ConformingProlongationOperator(
   int lsize, const GroupCommunicator &gc_, bool local_)
   : gc(gc_), local(local_)
{
   const Table &group_ldof = gc.GroupLDofTable();

   int n_external = 0;
   for (int g=1; g<group_ldof.Size(); ++g)
   {
      if (!gc.GetGroupTopology().IAmMaster(g))
      {
         n_external += group_ldof.RowSize(g);
      }
   }
   int tsize = lsize - n_external;

   height = lsize;
   width = tsize;

   external_ldofs.Reserve(n_external);
   for (int gr = 1; gr < group_ldof.Size(); gr++)
   {
      if (!gc.GetGroupTopology().IAmMaster(gr))
      {
         external_ldofs.Append(group_ldof.GetRow(gr), group_ldof.RowSize(gr));
      }
   }
   external_ldofs.Sort();
}

const GroupCommunicator &ConformingProlongationOperator::GetGroupCommunicator()
const
{
   return gc;
}

ConformingProlongationOperator::ConformingProlongationOperator(
   const ParFiniteElementSpace &pfes, bool local_)
   : Operator(pfes.GetVSize(), pfes.GetTrueVSize()),
     external_ldofs(),
     gc(pfes.GroupComm()),
     local(local_)
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

   const real_t *xdata = x.HostRead();
   real_t *ydata = y.HostWrite();
   const int m = external_ldofs.Size();

   const int in_layout = 2; // 2 - input is ltdofs array
   if (local)
   {
      y = 0.0;
   }
   else
   {
      gc.BcastBegin(const_cast<real_t*>(xdata), in_layout);
   }

   int j = 0;
   for (int i = 0; i < m; i++)
   {
      const int end = external_ldofs[i];
      std::copy(xdata+j-i, xdata+end-i, ydata+j);
      j = end+1;
   }
   std::copy(xdata+j-m, xdata+Width(), ydata+j);

   const int out_layout = 0; // 0 - output is ldofs array
   if (!local)
   {
      gc.BcastEnd(ydata, out_layout);
   }
}

void ConformingProlongationOperator::MultTranspose(
   const Vector &x, Vector &y) const
{
   MFEM_ASSERT(x.Size() == Height(), "");
   MFEM_ASSERT(y.Size() == Width(), "");

   const real_t *xdata = x.HostRead();
   real_t *ydata = y.HostWrite();
   const int m = external_ldofs.Size();

   if (!local)
   {
      gc.ReduceBegin(xdata);
   }

   int j = 0;
   for (int i = 0; i < m; i++)
   {
      const int end = external_ldofs[i];
      std::copy(xdata+j, xdata+end, ydata+j-i);
      j = end+1;
   }
   std::copy(xdata+j, xdata+Height(), ydata+j-m);

   const int out_layout = 2; // 2 - output is an array on all ltdofs
   if (!local)
   {
      gc.ReduceEnd<real_t>(ydata, out_layout, GroupCommunicator::Sum);
   }
}

DeviceConformingProlongationOperator::DeviceConformingProlongationOperator(
   const GroupCommunicator &gc_, const SparseMatrix *R, bool local_)
   : ConformingProlongationOperator(R->Width(), gc_, local_),
     mpi_gpu_aware(Device::GetGPUAwareMPI())
{
   MFEM_ASSERT(R->Finalized(), "");
   const int tdofs = R->Height();
   MFEM_ASSERT(tdofs == R->HostReadI()[tdofs], "");
   ltdof_ldof = Array<int>(const_cast<int*>(R->HostReadJ()), tdofs);
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
         Array<int> shared_ltdof(nbr_ltdof.GetJ(), nb_connections);
         Array<int> unique_ltdof(shared_ltdof);
         unique_ltdof.Sort();
         unique_ltdof.Unique();
         // Note: the next loop modifies the J array of nbr_ltdof
         for (int i = 0; i < shared_ltdof.Size(); i++)
         {
            shared_ltdof[i] = unique_ltdof.FindSorted(shared_ltdof[i]);
            MFEM_ASSERT(shared_ltdof[i] != -1, "internal error");
         }
         Table unique_shr;
         Transpose(shared_ltdof, unique_shr, unique_ltdof.Size());
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
      ext_ldof.GetMemory().UseDevice(true);
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

DeviceConformingProlongationOperator::DeviceConformingProlongationOperator(
   const ParFiniteElementSpace &pfes, bool local_)
   : DeviceConformingProlongationOperator(pfes.GroupComm(),
                                          pfes.GetRestrictionMatrix(),
                                          local_)
{
   MFEM_ASSERT(pfes.Conforming(), "internal error");
   MFEM_ASSERT(pfes.GetRestrictionMatrix()->Height() == pfes.GetTrueVSize(), "");
}

static void ExtractSubVector(const Array<int> &indices,
                             const Vector &vin, Vector &vout)
{
   MFEM_ASSERT(indices.Size() == vout.Size(), "incompatible sizes!");
   auto y = vout.Write();
   const auto x = vin.Read();
   const auto I = indices.Read();
   mfem::forall(indices.Size(), [=] MFEM_HOST_DEVICE (int i)
   {
      y[i] = x[I[i]];
   }); // indices can be repeated
}

void DeviceConformingProlongationOperator::BcastBeginCopy(
   const Vector &x) const
{
   // shr_buf[i] = src[shr_ltdof[i]]
   if (shr_ltdof.Size() == 0) { return; }
   ExtractSubVector(shr_ltdof, x, shr_buf);
   // If the above kernel is executed asynchronously, we should wait for it to
   // complete
   if (mpi_gpu_aware) { MFEM_STREAM_SYNC; }
}

static void SetSubVector(const Array<int> &indices,
                         const Vector &vin, Vector &vout)
{
   MFEM_ASSERT(indices.Size() == vin.Size(), "incompatible sizes!");
   // Use ReadWrite() since we modify only a subset of the indices:
   auto y = vout.ReadWrite();
   const auto x = vin.Read();
   const auto I = indices.Read();
   mfem::forall(indices.Size(), [=] MFEM_HOST_DEVICE (int i)
   {
      y[I[i]] = x[i];
   });
}

void DeviceConformingProlongationOperator::BcastLocalCopy(
   const Vector &x, Vector &y) const
{
   // dst[ltdof_ldof[i]] = src[i]
   if (ltdof_ldof.Size() == 0) { return; }
   SetSubVector(ltdof_ldof, x, y);
}

void DeviceConformingProlongationOperator::BcastEndCopy(
   Vector &y) const
{
   // dst[ext_ldof[i]] = ext_buf[i]
   if (ext_ldof.Size() == 0) { return; }
   SetSubVector(ext_ldof, ext_buf, y);
}

void DeviceConformingProlongationOperator::Mult(const Vector &x,
                                                Vector &y) const
{
   const GroupTopology &gtopo = gc.GetGroupTopology();
   int req_counter = 0;
   // Make sure 'y' is marked as valid on device and for use on device.
   // This ensures that there is no unnecessary host to device copy when the
   // input 'y' is valid on host (in 'y.SetSubVector(ext_ldof, 0.0)' when local
   // is true) or BcastLocalCopy (when local is false).
   y.Write();
   if (local)
   {
      // done on device since we've marked ext_ldof for use on device:
      y.SetSubVector(ext_ldof, 0.0);
   }
   else
   {
      BcastBeginCopy(x); // copy to 'shr_buf'
      for (int nbr = 1; nbr < gtopo.GetNumNeighbors(); nbr++)
      {
         const int send_offset = shr_buf_offsets[nbr];
         const int send_size = shr_buf_offsets[nbr+1] - send_offset;
         if (send_size > 0)
         {
            auto send_buf = mpi_gpu_aware ? shr_buf.Read() : shr_buf.HostRead();
            MPI_Isend(send_buf + send_offset, send_size, MPITypeMap<real_t>::mpi_type,
                      gtopo.GetNeighborRank(nbr), 41822,
                      gtopo.GetComm(), &requests[req_counter++]);
         }
         const int recv_offset = ext_buf_offsets[nbr];
         const int recv_size = ext_buf_offsets[nbr+1] - recv_offset;
         if (recv_size > 0)
         {
            auto recv_buf = mpi_gpu_aware ? ext_buf.Write() : ext_buf.HostWrite();
            MPI_Irecv(recv_buf + recv_offset, recv_size, MPITypeMap<real_t>::mpi_type,
                      gtopo.GetNeighborRank(nbr), 41822,
                      gtopo.GetComm(), &requests[req_counter++]);
         }
      }
   }
   BcastLocalCopy(x, y);
   if (!local)
   {
      MPI_Waitall(req_counter, requests, MPI_STATUSES_IGNORE);
      BcastEndCopy(y); // copy from 'ext_buf'
   }
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
   ExtractSubVector(ext_ldof, x, ext_buf);
   // If the above kernel is executed asynchronously, we should wait for it to
   // complete
   if (mpi_gpu_aware) { MFEM_STREAM_SYNC; }
}

void DeviceConformingProlongationOperator::ReduceLocalCopy(
   const Vector &x, Vector &y) const
{
   // dst[i] = src[ltdof_ldof[i]]
   if (ltdof_ldof.Size() == 0) { return; }
   ExtractSubVector(ltdof_ldof, x, y);
}

static void AddSubVector(const Array<int> &unique_dst_indices,
                         const Array<int> &unique_to_src_offsets,
                         const Array<int> &unique_to_src_indices,
                         const Vector &src,
                         Vector &dst)
{
   auto y = dst.ReadWrite();
   const auto x = src.Read();
   const auto DST_I = unique_dst_indices.Read();
   const auto SRC_O = unique_to_src_offsets.Read();
   const auto SRC_I = unique_to_src_indices.Read();
   mfem::forall(unique_dst_indices.Size(), [=] MFEM_HOST_DEVICE (int i)
   {
      const int dst_idx = DST_I[i];
      real_t sum = y[dst_idx];
      const int end = SRC_O[i+1];
      for (int j = SRC_O[i]; j != end; ++j) { sum += x[SRC_I[j]]; }
      y[dst_idx] = sum;
   });
}

void DeviceConformingProlongationOperator::ReduceEndAssemble(Vector &y) const
{
   // dst[shr_ltdof[i]] += shr_buf[i]
   if (unq_ltdof.Size() == 0) { return; }
   AddSubVector(unq_ltdof, unq_shr_i, unq_shr_j, shr_buf, y);
}

void DeviceConformingProlongationOperator::MultTranspose(const Vector &x,
                                                         Vector &y) const
{
   const GroupTopology &gtopo = gc.GetGroupTopology();
   int req_counter = 0;
   if (!local)
   {
      ReduceBeginCopy(x); // copy to 'ext_buf'
      for (int nbr = 1; nbr < gtopo.GetNumNeighbors(); nbr++)
      {
         const int send_offset = ext_buf_offsets[nbr];
         const int send_size = ext_buf_offsets[nbr+1] - send_offset;
         if (send_size > 0)
         {
            auto send_buf = mpi_gpu_aware ? ext_buf.Read() : ext_buf.HostRead();
            MPI_Isend(send_buf + send_offset, send_size, MPITypeMap<real_t>::mpi_type,
                      gtopo.GetNeighborRank(nbr), 41823,
                      gtopo.GetComm(), &requests[req_counter++]);
         }
         const int recv_offset = shr_buf_offsets[nbr];
         const int recv_size = shr_buf_offsets[nbr+1] - recv_offset;
         if (recv_size > 0)
         {
            auto recv_buf = mpi_gpu_aware ? shr_buf.Write() : shr_buf.HostWrite();
            MPI_Irecv(recv_buf + recv_offset, recv_size, MPITypeMap<real_t>::mpi_type,
                      gtopo.GetNeighborRank(nbr), 41823,
                      gtopo.GetComm(), &requests[req_counter++]);
         }
      }
   }
   ReduceLocalCopy(x, y);
   if (!local)
   {
      MPI_Waitall(req_counter, requests, MPI_STATUSES_IGNORE);
      ReduceEndAssemble(y); // assemble from 'shr_buf'
   }
}

} // namespace mfem

#endif
