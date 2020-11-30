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

#include "hybridization.hpp"
#include "gridfunc.hpp"

#ifdef MFEM_USE_MPI
#include "pfespace.hpp"
#endif

#include <map>

// uncomment next line for debugging: write C and P to file
// #define MFEM_DEBUG_HYBRIDIZATION_CP
#ifdef MFEM_DEBUG_HYBRIDIZATION_CP
#include <fstream>
#endif

namespace mfem
{

Hybridization::Hybridization(FiniteElementSpace *fespace,
                             FiniteElementSpace *c_fespace)
   : fes(fespace), c_fes(c_fespace), c_bfi(NULL), Ct(NULL), H(NULL),
     Af_data(NULL), Af_ipiv(NULL)
{
#ifdef MFEM_USE_MPI
   pC = P_pc = NULL;
   pH.SetType(Operator::Hypre_ParCSR);
#endif
}

Hybridization::~Hybridization()
{
#ifdef MFEM_USE_MPI
   delete P_pc;
   delete pC;
#endif
   delete [] Af_ipiv;
   delete [] Af_data;
   delete H;
   delete Ct;
   delete c_bfi;
}

void Hybridization::ConstructC()
{
   const int NE = fes->GetNE();
   int num_hat_dofs = hat_offsets[NE];
   Array<int> vdofs, c_vdofs;

   int c_num_face_nbr_dofs = 0;
#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *c_pfes = dynamic_cast<ParFiniteElementSpace*>(c_fes);
   ParMesh *pmesh = c_pfes ? c_pfes->GetParMesh() : NULL;
   HYPRE_Int num_shared_slave_faces = 0, glob_num_shared_slave_faces = 0;
   if (c_pfes)
   {
      if (pmesh->Nonconforming())
      {
         const int dim = pmesh->Dimension();
         const NCMesh::NCList &shared = pmesh->pncmesh->GetSharedList(dim-1);
         num_shared_slave_faces = (HYPRE_Int) shared.slaves.Size();
         MPI_Allreduce(&num_shared_slave_faces, &glob_num_shared_slave_faces, 1,
                       HYPRE_MPI_INT, MPI_SUM, pmesh->GetComm());
         MFEM_ASSERT(glob_num_shared_slave_faces%2 == 0, "");
         glob_num_shared_slave_faces /= 2;
         if (glob_num_shared_slave_faces)
         {
            c_pfes->ExchangeFaceNbrData();
            c_num_face_nbr_dofs = c_pfes->GetFaceNbrVSize();
         }
#ifdef MFEM_DEBUG_HERE
         MFEM_WARNING('[' << c_pfes->GetMyRank() <<
                      "] num_shared_slave_faces = " << num_shared_slave_faces
                      << ", glob_num_shared_slave_faces = "
                      << glob_num_shared_slave_faces
                      << "\n   num_face_nbr_dofs = " << c_num_face_nbr_dofs
                      << ", num_shared_faces = " << pmesh->GetNSharedFaces());
#undef MFEM_DEBUG_HERE
#endif
      }
   }
#endif

   const int c_vsize = c_fes->GetVSize();
   Ct = new SparseMatrix(num_hat_dofs, c_vsize + c_num_face_nbr_dofs);

   if (c_bfi)
   {
      const int skip_zeros = 1;
      DenseMatrix elmat;
      FaceElementTransformations *FTr;
      Mesh *mesh = fes->GetMesh();
      int num_faces = mesh->GetNumFaces();
      for (int i = 0; i < num_faces; i++)
      {
         FTr = mesh->GetInteriorFaceTransformations(i);
         if (!FTr) { continue; }

         int o1 = hat_offsets[FTr->Elem1No];
         int s1 = hat_offsets[FTr->Elem1No+1] - o1;
         int o2 = hat_offsets[FTr->Elem2No];
         int s2 = hat_offsets[FTr->Elem2No+1] - o2;
         vdofs.SetSize(s1 + s2);
         for (int j = 0; j < s1; j++)
         {
            vdofs[j] = o1 + j;
         }
         for (int j = 0; j < s2; j++)
         {
            vdofs[s1+j] = o2 + j;
         }
         c_fes->GetFaceVDofs(i, c_vdofs);
         c_bfi->AssembleFaceMatrix(*c_fes->GetFaceElement(i),
                                   *fes->GetFE(FTr->Elem1No),
                                   *fes->GetFE(FTr->Elem2No),
                                   *FTr, elmat);
         // zero-out small elements in elmat
         elmat.Threshold(1e-12 * elmat.MaxMaxNorm());
         Ct->AddSubMatrix(vdofs, c_vdofs, elmat, skip_zeros);
      }
#ifdef MFEM_USE_MPI
      if (pmesh)
      {
         // Assemble local contribution to Ct from shared faces
         const int num_shared_faces = pmesh->GetNSharedFaces();
         for (int i = 0; i < num_shared_faces; i++)
         {
            const int face_no = pmesh->GetSharedFace(i);
            const bool ghost_sface = (face_no >= num_faces);
            const FiniteElement *fe, *face_fe;
            if (!ghost_sface)
            {
               FTr = pmesh->GetFaceElementTransformations(face_no);
               MFEM_ASSERT(FTr->Elem2No < 0, "");
               face_fe = c_fes->GetFaceElement(face_no);
               c_fes->GetFaceVDofs(face_no, c_vdofs);
            }
            else
            {
               const int fill2 = false; // only need side "1" data
               FTr = pmesh->GetSharedFaceTransformations(i, fill2);
               face_fe = c_pfes->GetFaceNbrFaceFE(face_no);
               c_pfes->GetFaceNbrFaceVDofs(face_no, c_vdofs);
               FiniteElementSpace::AdjustVDofs(c_vdofs);
               for (int j = 0; j < c_vdofs.Size(); j++)
               {
                  c_vdofs[j] += c_vsize;
               }
            }
            int o1 = hat_offsets[FTr->Elem1No];
            int s1 = hat_offsets[FTr->Elem1No+1] - o1;
            vdofs.SetSize(s1);
            for (int j = 0; j < s1; j++)
            {
               vdofs[j] = o1 + j;
            }
            fe = fes->GetFE(FTr->Elem1No);
            c_bfi->AssembleFaceMatrix(*face_fe, *fe, *fe, *FTr, elmat);
            // zero-out small elements in elmat
            elmat.Threshold(1e-12 * elmat.MaxMaxNorm());
            Ct->AddSubMatrix(vdofs, c_vdofs, elmat, skip_zeros);
         }
         if (glob_num_shared_slave_faces)
         {
            // Convert Ct to parallel and then transpose it:
            Ct->Finalize(skip_zeros);
            HYPRE_Int Ct_num_rows = Ct->Height();
            Array<HYPRE_Int> Ct_rows, *offsets[1] = { &Ct_rows };
            pmesh->GenerateOffsets(1, &Ct_num_rows, offsets);
            Array<HYPRE_Int> Ct_J(Ct->NumNonZeroElems());
            HYPRE_Int c_ldof_offset = c_pfes->GetMyDofOffset();
            const HYPRE_Int *c_face_nbr_glob_ldof =
               c_pfes->GetFaceNbrGlobalDofMap();
            int *J = Ct->GetJ();
            for (int i = 0; i < Ct_J.Size(); i++)
            {
               Ct_J[i] = J[i] < c_vsize ?
                         J[i] + c_ldof_offset :
                         c_face_nbr_glob_ldof[J[i] - c_vsize];
            }
            HypreParMatrix pCt(pmesh->GetComm(), Ct->Height(),
                               Ct_rows.Last(), c_pfes->GlobalVSize(),
                               Ct->GetI(), Ct_J.GetData(), Ct->GetData(),
                               Ct_rows, c_pfes->GetDofOffsets());
            Ct_J.DeleteAll();
            pC = pCt.Transpose();
         }
         if (pmesh->Nonconforming())
         {
            // TODO - Construct P_pc directly in the pH format
            P_pc = c_pfes->GetPartialConformingInterpolation();
         }
      }
#endif
      Ct->Finalize(skip_zeros);
   }
   else
   {
      // Check if c_fes is really needed here.
      MFEM_ABORT("TODO: algebraic definition of C");
   }
}

void Hybridization::Init(const Array<int> &ess_tdof_list)
{
   if (Ct) { return; }

   // count the number of dofs in the discontinuous version of fes:
   const int NE = fes->GetNE();
   Array<int> vdofs;
   int num_hat_dofs = 0;
   hat_offsets.SetSize(NE+1);
   hat_offsets[0] = 0;
   for (int i = 0; i < NE; i++)
   {
      fes->GetElementVDofs(i, vdofs);
      num_hat_dofs += vdofs.Size();
      hat_offsets[i+1] = num_hat_dofs;
   }

   // Assemble the constraint matrix C
   ConstructC();

#ifdef MFEM_DEBUG_HYBRIDIZATION_CP
   // Debug: write C and P to file
   {
      std::ofstream C_file("C_matrix.txt");
      SparseMatrix *C = Transpose(*Ct);
      C->PrintMatlab(C_file);
      delete C;

      const SparseMatrix *P = fes->GetConformingProlongation();
      if (P)
      {
         std::ofstream P_file("P_matrix.txt");
         P->PrintMatlab(P_file);
      }
   }
#endif

   // Define the "free" (0) and "essential" (1) hat_dofs.
   // The "essential" hat_dofs are those that depend only on essential cdofs;
   // all other hat_dofs are "free".
   hat_dofs_marker.SetSize(num_hat_dofs);
   Array<int> free_tdof_marker;
#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *pfes = dynamic_cast<ParFiniteElementSpace*>(fes);
   free_tdof_marker.SetSize(pfes ? pfes->TrueVSize() :
                            fes->GetConformingVSize());
#else
   free_tdof_marker.SetSize(fes->GetConformingVSize());
#endif
   free_tdof_marker = 1;
   for (int i = 0; i < ess_tdof_list.Size(); i++)
   {
      free_tdof_marker[ess_tdof_list[i]] = 0;
   }
   Array<int> free_vdofs_marker;
#ifdef MFEM_USE_MPI
   if (!pfes)
   {
      const SparseMatrix *cP = fes->GetConformingProlongation();
      if (!cP)
      {
         free_vdofs_marker.MakeRef(free_tdof_marker);
      }
      else
      {
         free_vdofs_marker.SetSize(fes->GetVSize());
         cP->BooleanMult(free_tdof_marker, free_vdofs_marker);
      }
   }
   else
   {
      HypreParMatrix *P = pfes->Dof_TrueDof_Matrix();
      free_vdofs_marker.SetSize(fes->GetVSize());
      P->BooleanMult(1, free_tdof_marker, 0, free_vdofs_marker);
   }
#else
   const SparseMatrix *cP = fes->GetConformingProlongation();
   if (!cP)
   {
      free_vdofs_marker.MakeRef(free_tdof_marker);
   }
   else
   {
      free_vdofs_marker.SetSize(fes->GetVSize());
      cP->BooleanMult(free_tdof_marker, free_vdofs_marker);
   }
#endif
   for (int i = 0; i < NE; i++)
   {
      fes->GetElementVDofs(i, vdofs);
      FiniteElementSpace::AdjustVDofs(vdofs);
      for (int j = 0; j < vdofs.Size(); j++)
      {
         hat_dofs_marker[hat_offsets[i]+j] = ! free_vdofs_marker[vdofs[j]];
      }
   }
#ifndef MFEM_DEBUG
   // In DEBUG mode this array is used below.
   free_tdof_marker.DeleteAll();
#endif
   free_vdofs_marker.DeleteAll();
   // Split the "free" (0) hat_dofs into "internal" (0) or "boundary" (-1).
   // The "internal" hat_dofs are those "free" hat_dofs for which the
   // corresponding column in C is zero; otherwise the free hat_dof is
   // "boundary".
   for (int i = 0; i < num_hat_dofs; i++)
   {
      // skip "essential" hat_dofs and empty rows in Ct
      if (hat_dofs_marker[i] != 1 && Ct->RowSize(i) > 0)
      {
         hat_dofs_marker[i] = -1; // mark this hat_dof as "boundary"
      }
   }

   // Define Af_offsets and Af_f_offsets
   Af_offsets.SetSize(NE+1);
   Af_offsets[0] = 0;
   Af_f_offsets.SetSize(NE+1);
   Af_f_offsets[0] = 0;
   // #define MFEM_DEBUG_HERE // uncomment to enable printing of hat dofs stats
#ifdef MFEM_DEBUG_HERE
   int b_size = 0;
#endif
   for (int i = 0; i < NE; i++)
   {
      int f_size = 0; // count the "free" hat_dofs in element i
      for (int j = hat_offsets[i]; j < hat_offsets[i+1]; j++)
      {
         if (hat_dofs_marker[j] != 1) { f_size++; }
#ifdef MFEM_DEBUG_HERE
         if (hat_dofs_marker[j] == -1) { b_size++; }
#endif
      }
      Af_offsets[i+1] = Af_offsets[i] + f_size*f_size;
      Af_f_offsets[i+1] = Af_f_offsets[i] + f_size;
   }

#ifdef MFEM_DEBUG_HERE
#ifndef MFEM_USE_MPI
   int myid = 0;
#else
   int myid = pmesh ? pmesh->GetMyRank() : 0;
#endif
   int i_size = Af_f_offsets[NE] - b_size;
   int e_size = num_hat_dofs - (i_size + b_size);
   mfem::out << "\nHybridization::Init:"
             << " [" << myid << "] hat dofs - \"internal\": " << i_size
             << ", \"boundary\": " << b_size
             << ", \"essential\": " << e_size << '\n' << std::endl;
#undef MFEM_DEBUG_HERE
#endif

   Af_data = new double[Af_offsets[NE]];
   Af_ipiv = new int[Af_f_offsets[NE]];

#ifdef MFEM_DEBUG
   // check that Ref = 0
   const SparseMatrix *R = fes->GetRestrictionMatrix();
   if (!R) { return; }
   Array<int> vdof_marker(fes->GetVSize()); // 0 - f, 1 - e
   vdof_marker = 0;
   for (int i = 0; i < NE; i++)
   {
      fes->GetElementVDofs(i, vdofs);
      FiniteElementSpace::AdjustVDofs(vdofs);
      for (int j = 0; j < vdofs.Size(); j++)
      {
         if (hat_dofs_marker[hat_offsets[i]+j] == 1) // "essential" hat dof
         {
            vdof_marker[vdofs[j]] = 1;
         }
      }
   }
   for (int tdof = 0; tdof < R->Height(); tdof++)
   {
      if (free_tdof_marker[tdof]) { continue; }

      const int ncols = R->RowSize(tdof);
      const int *cols = R->GetRowColumns(tdof);
      const double *vals = R->GetRowEntries(tdof);
      for (int j = 0; j < ncols; j++)
      {
         if (std::abs(vals[j]) != 0.0 && vdof_marker[cols[j]] == 0)
         {
            MFEM_ABORT("Ref != 0");
         }
      }
   }
#endif
}

void Hybridization::GetIBDofs(
   int el, Array<int> &i_dofs, Array<int> &b_dofs) const
{
   // returns local indices in i_dofs and b_dofs
   int h_start, h_end;

   h_start = hat_offsets[el];
   h_end = hat_offsets[el+1];
   i_dofs.Reserve(h_end-h_start);
   i_dofs.SetSize(0);
   b_dofs.Reserve(h_end-h_start);
   b_dofs.SetSize(0);
   for (int i = h_start; i < h_end; i++)
   {
      int mark = hat_dofs_marker[i];
      if (mark == 0) { i_dofs.Append(i-h_start); }
      else if (mark == -1) { b_dofs.Append(i-h_start); }
   }
}

void Hybridization::GetBDofs(int el, int &num_idofs, Array<int> &b_dofs) const
{
   // returns global indices in b_dofs
   const int h_start = hat_offsets[el];
   const int h_end = hat_offsets[el+1];
   b_dofs.Reserve(h_end-h_start);
   b_dofs.SetSize(0);
   num_idofs = 0;
   for (int i = h_start; i < h_end; i++)
   {
      int mark = hat_dofs_marker[i];
      if (mark == 0) { num_idofs++; }
      else if (mark == -1) { b_dofs.Append(i); }
   }
}

void Hybridization::AssembleMatrix(int el, const DenseMatrix &A)
{
   Array<int> i_dofs, b_dofs;

   GetIBDofs(el, i_dofs, b_dofs);

   DenseMatrix A_ii(Af_data + Af_offsets[el], i_dofs.Size(), i_dofs.Size());
   DenseMatrix A_ib(A_ii.Data() + i_dofs.Size()*i_dofs.Size(),
                    i_dofs.Size(), b_dofs.Size());
   DenseMatrix A_bi(A_ib.Data() + i_dofs.Size()*b_dofs.Size(),
                    b_dofs.Size(), i_dofs.Size());
   DenseMatrix A_bb(A_bi.Data() + b_dofs.Size()*i_dofs.Size(),
                    b_dofs.Size(), b_dofs.Size());

   for (int j = 0; j < i_dofs.Size(); j++)
   {
      int j_dof = i_dofs[j];
      for (int i = 0; i < i_dofs.Size(); i++)
      {
         A_ii(i,j) = A(i_dofs[i],j_dof);
      }
      for (int i = 0; i < b_dofs.Size(); i++)
      {
         A_bi(i,j) = A(b_dofs[i],j_dof);
      }
   }
   for (int j = 0; j < b_dofs.Size(); j++)
   {
      int j_dof = b_dofs[j];
      for (int i = 0; i < i_dofs.Size(); i++)
      {
         A_ib(i,j) = A(i_dofs[i],j_dof);
      }
      for (int i = 0; i < b_dofs.Size(); i++)
      {
         A_bb(i,j) = A(b_dofs[i],j_dof);
      }
   }
}

void Hybridization::AssembleBdrMatrix(int bdr_el, const DenseMatrix &A)
{
   // Not tested.
#ifdef MFEM_DEBUG
   Array<int> vdofs, bvdofs;
   fes->GetBdrElementVDofs(bdr_el, bvdofs);
#endif

   int el;
   DenseMatrix B(A);
   Array<int> i_dofs, b_dofs, e2f;

   {
      int info, vdim = fes->GetVDim();
      Array<int> lvdofs;
      Mesh *mesh = fes->GetMesh();
      mesh->GetBdrElementAdjacentElement(bdr_el, el, info);
      e2f.SetSize(hat_offsets[el+1]-hat_offsets[el], -1);
      lvdofs.Reserve(A.Height());
      fes->FEColl()->SubDofOrder(mesh->GetElementBaseGeometry(el),
                                 mesh->Dimension()-1, info, lvdofs);
      // Convert local element dofs to local element vdofs.
      Ordering::DofsToVDofs<Ordering::byNODES>(e2f.Size()/vdim, vdim, lvdofs);
      MFEM_ASSERT(lvdofs.Size() == A.Height(), "internal error");
#ifdef MFEM_DEBUG
      fes->GetElementVDofs(el, vdofs);
      for (int i = 0; i < lvdofs.Size(); i++)
      {
         int bd = lvdofs[i];
         bd = (bd >= 0) ? vdofs[bd] : -1-vdofs[-1-bd];
         MFEM_ASSERT(bvdofs[i] == bd, "internal error");
      }
#endif
      B.AdjustDofDirection(lvdofs);
      FiniteElementSpace::AdjustVDofs(lvdofs);
      // Create a map from local element vdofs to local boundary (face) vdofs.
      for (int i = 0; i < lvdofs.Size(); i++)
      {
         e2f[lvdofs[i]] = i;
      }
   }

   GetIBDofs(el, i_dofs, b_dofs);

   DenseMatrix A_ii(Af_data + Af_offsets[el], i_dofs.Size(), i_dofs.Size());
   DenseMatrix A_ib(A_ii.Data() + i_dofs.Size()*i_dofs.Size(),
                    i_dofs.Size(), b_dofs.Size());
   DenseMatrix A_bi(A_ib.Data() + i_dofs.Size()*b_dofs.Size(),
                    b_dofs.Size(), i_dofs.Size());
   DenseMatrix A_bb(A_bi.Data() + b_dofs.Size()*i_dofs.Size(),
                    b_dofs.Size(), b_dofs.Size());

   for (int j = 0; j < i_dofs.Size(); j++)
   {
      int j_f = e2f[i_dofs[j]];
      if (j_f == -1) { continue; }
      for (int i = 0; i < i_dofs.Size(); i++)
      {
         int i_f = e2f[i_dofs[i]];
         if (i_f == -1) { continue; }
         A_ii(i,j) += B(i_f,j_f);
      }
      for (int i = 0; i < b_dofs.Size(); i++)
      {
         int i_f = e2f[b_dofs[i]];
         if (i_f == -1) { continue; }
         A_bi(i,j) += B(i_f,j_f);
      }
   }
   for (int j = 0; j < b_dofs.Size(); j++)
   {
      int j_f = e2f[b_dofs[j]];
      if (j_f == -1) { continue; }
      for (int i = 0; i < i_dofs.Size(); i++)
      {
         int i_f = e2f[i_dofs[i]];
         if (i_f == -1) { continue; }
         A_ib(i,j) += B(i_f,j_f);
      }
      for (int i = 0; i < b_dofs.Size(); i++)
      {
         int i_f = e2f[b_dofs[i]];
         if (i_f == -1) { continue; }
         A_bb(i,j) += B(i_f,j_f);
      }
   }
}

void Hybridization::ComputeH()
{
   const int skip_zeros = 1;
   Array<int> c_dof_marker(Ct->Width());
   Array<int> b_dofs, c_dofs;
   const int NE = fes->GetNE();
   DenseMatrix Cb_t, Sb_inv_Cb_t, Hb;
#ifndef MFEM_USE_MPI
   H = new SparseMatrix(Ct->Width());
#else
   H = pC ? NULL : new SparseMatrix(Ct->Width());
   // V = Sb^{-1} Cb^T, for parallel non-conforming meshes
   SparseMatrix *V = pC ? new SparseMatrix(Ct->Height(), Ct->Width()) : NULL;
#endif

   c_dof_marker = -1;
   int c_mark_start = 0;
   for (int el = 0; el < NE; el++)
   {
      int i_dofs_size;
      GetBDofs(el, i_dofs_size, b_dofs);

      LUFactors LU_ii(Af_data + Af_offsets[el], Af_ipiv + Af_f_offsets[el]);
      double *A_ib_data = LU_ii.data + i_dofs_size*i_dofs_size;
      double *A_bi_data = A_ib_data + i_dofs_size*b_dofs.Size();
      LUFactors LU_bb(A_bi_data + i_dofs_size*b_dofs.Size(),
                      LU_ii.ipiv + i_dofs_size);

      LU_ii.Factor(i_dofs_size);
      LU_ii.BlockFactor(i_dofs_size, b_dofs.Size(),
                        A_ib_data, A_bi_data, LU_bb.data);
      LU_bb.Factor(b_dofs.Size());

      // Extract Cb_t from Ct, define c_dofs
      c_dofs.SetSize(0);
      for (int i = 0; i < b_dofs.Size(); i++)
      {
         const int row = b_dofs[i];
         const int ncols = Ct->RowSize(row);
         const int *cols = Ct->GetRowColumns(row);
         for (int j = 0; j < ncols; j++)
         {
            const int c_dof = cols[j];
            if (c_dof_marker[c_dof] < c_mark_start)
            {
               c_dof_marker[c_dof] = c_mark_start + c_dofs.Size();
               c_dofs.Append(c_dof);
            }
         }
      }
      Cb_t.SetSize(b_dofs.Size(), c_dofs.Size());
      Cb_t = 0.0;
      for (int i = 0; i < b_dofs.Size(); i++)
      {
         const int row = b_dofs[i];
         const int ncols = Ct->RowSize(row);
         const int *cols = Ct->GetRowColumns(row);
         const double *vals = Ct->GetRowEntries(row);
         for (int j = 0; j < ncols; j++)
         {
            const int loc_j = c_dof_marker[cols[j]] - c_mark_start;
            Cb_t(i,loc_j) = vals[j];
         }
      }

      // Compute Hb = Cb Sb^{-1} Cb^t
      Sb_inv_Cb_t = Cb_t;
      LU_bb.Solve(Cb_t.Height(), Cb_t.Width(), Sb_inv_Cb_t.Data());
#ifdef MFEM_USE_MPI
      if (!pC)
#endif
      {
         Hb.SetSize(Cb_t.Width());
         MultAtB(Cb_t, Sb_inv_Cb_t, Hb);

         // Assemble Hb into H
         H->AddSubMatrix(c_dofs, c_dofs, Hb, skip_zeros);
      }
#ifdef MFEM_USE_MPI
      else
      {
         V->AddSubMatrix(b_dofs, c_dofs, Sb_inv_Cb_t, skip_zeros);
      }
#endif

      c_mark_start += c_dofs.Size();
      MFEM_VERIFY(c_mark_start >= 0, "overflow"); // check for overflow
   }
   const bool fix_empty_rows = true;
#ifndef MFEM_USE_MPI
   H->Finalize(skip_zeros, fix_empty_rows);
#else
   ParFiniteElementSpace *c_pfes = dynamic_cast<ParFiniteElementSpace*>(c_fes);
   if (!pC)
   {
      H->Finalize(skip_zeros, fix_empty_rows);
      if (!c_pfes) { return; }

      OperatorHandle pP(pH.Type()), dH(pH.Type());
      // TODO - construct P_pc / Dof_TrueDof_Matrix directly in the pH format
      pP.ConvertFrom(P_pc ? P_pc : c_pfes->Dof_TrueDof_Matrix());
      dH.MakeSquareBlockDiag(c_pfes->GetComm(),c_pfes->GlobalVSize(),
                             c_pfes->GetDofOffsets(), H);
      pH.MakePtAP(dH, pP);
      delete H;
      H = NULL;
   }
   else
   {
      // TODO: add ones on the diagonal of zero rows
      V->Finalize();
      Array<HYPRE_Int> V_J(V->NumNonZeroElems());
      MFEM_ASSERT(c_pfes, "");
      const int c_vsize = c_fes->GetVSize();
      HYPRE_Int c_ldof_offset = c_pfes->GetMyDofOffset();
      const HYPRE_Int *c_face_nbr_glob_ldof = c_pfes->GetFaceNbrGlobalDofMap();
      int *J = V->GetJ();
      for (int i = 0; i < V_J.Size(); i++)
      {
         V_J[i] = J[i] < c_vsize ?
                  J[i] + c_ldof_offset :
                  c_face_nbr_glob_ldof[J[i] - c_vsize];
      }
      // TODO - lpH directly in the pH format
      HypreParMatrix *lpH;
      {
         HypreParMatrix pV(c_pfes->GetComm(), V->Height(),
                           pC->GetGlobalNumCols(), pC->GetGlobalNumRows(),
                           V->GetI(), V_J.GetData(), V->GetData(),
                           pC->ColPart(), pC->RowPart());
         // The above constructor makes copies of all input arrays, so we can
         // safely delete V_J and V:
         V_J.DeleteAll();
         delete V;
         lpH = ParMult(pC, &pV);
      }
      OperatorHandle pP(pH.Type()), plpH(pH.Type());
      // TODO - construct P_pc directly in the pH format
      pP.ConvertFrom(P_pc);
      plpH.ConvertFrom(lpH);
      MFEM_VERIFY(pH.Type() != Operator::PETSC_MATIS, "To be implemented");
      pH.MakePtAP(plpH, pP);
      delete lpH;
   }
#endif
}

void Hybridization::Finalize()
{
#ifndef MFEM_USE_MPI
   if (!H) { ComputeH(); }
#else
   if (!H && !pH.Ptr()) { ComputeH(); }
#endif
}

void Hybridization::MultAfInv(const Vector &b, const Vector &lambda, Vector &bf,
                              int mode) const
{
   // b1 = Rf^t b (assuming that Ref = 0)
   Vector b1;
   const SparseMatrix *R = fes->GetRestrictionMatrix();
   if (!R)
   {
      b1.SetDataAndSize(b.GetData(), b.Size());
   }
   else
   {
      b1.SetSize(fes->GetVSize());
      R->MultTranspose(b, b1);
   }

   const int NE = fes->GetMesh()->GetNE();
   Array<int> vdofs, i_dofs, b_dofs;
   Vector el_vals, bf_i, i_vals, b_vals;
   bf.SetSize(hat_offsets[NE]);
   if (mode == 1)
   {
#ifdef MFEM_USE_MPI
      ParFiniteElementSpace *c_pfes =
         dynamic_cast<ParFiniteElementSpace*>(c_fes);
      if (!c_pfes)
      {
         Ct->Mult(lambda, bf);
      }
      else
      {
         Vector L(c_pfes->GetVSize());
         (P_pc ? P_pc : c_pfes->GetProlongationMatrix())->Mult(lambda, L);
         pC ? pC->MultTranspose(L, bf) : Ct->Mult(L, bf);
      }
#else
      Ct->Mult(lambda, bf);
#endif
   }
   // Apply Af^{-1}
   Array<bool> vdof_marker(b1.Size());
   vdof_marker = false;
   for (int i = 0; i < NE; i++)
   {
      fes->GetElementVDofs(i, vdofs);
      b1.GetSubVector(vdofs, el_vals);
      for (int j = 0; j < vdofs.Size(); j++)
      {
         int vdof = vdofs[j];
         if (vdof < 0) { vdof = -1 - vdof; }
         if (vdof_marker[vdof]) { el_vals(j) = 0.0; }
         else { vdof_marker[vdof] = true; }
      }
      bf_i.SetDataAndSize(&bf[hat_offsets[i]], vdofs.Size());
      if (mode == 1)
      {
         el_vals -= bf_i;
      }
      GetIBDofs(i, i_dofs, b_dofs);
      el_vals.GetSubVector(i_dofs, i_vals);
      el_vals.GetSubVector(b_dofs, b_vals);

      LUFactors LU_ii(Af_data + Af_offsets[i], Af_ipiv + Af_f_offsets[i]);
      double *U_ib = LU_ii.data + i_dofs.Size()*i_dofs.Size();
      double *L_bi = U_ib + i_dofs.Size()*b_dofs.Size();
      LUFactors LU_bb(L_bi + b_dofs.Size()*i_dofs.Size(),
                      LU_ii.ipiv + i_dofs.Size());
      LU_ii.BlockForwSolve(i_dofs.Size(), b_dofs.Size(), 1, L_bi,
                           i_vals.GetData(), b_vals.GetData());
      LU_bb.Solve(b_dofs.Size(), 1, b_vals.GetData());
      bf_i = 0.0;
      if (mode == 1)
      {
         LU_ii.BlockBackSolve(i_dofs.Size(), b_dofs.Size(), 1, U_ib,
                              b_vals.GetData(), i_vals.GetData());
         bf_i.SetSubVector(i_dofs, i_vals);
      }
      bf_i.SetSubVector(b_dofs, b_vals);
   }
}

void Hybridization::ReduceRHS(const Vector &b, Vector &b_r) const
{
   // bf = Af^{-1} Rf^t b
   Vector bf;
   MultAfInv(b, b, bf, 0);

   // b_r = Cf bf
#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *c_pfes = dynamic_cast<ParFiniteElementSpace*>(c_fes);
   if (!c_pfes)
   {
      b_r.SetSize(Ct->Width());
      Ct->MultTranspose(bf, b_r);
   }
   else
   {
      Vector bl(pC ? pC->Height() : Ct->Width());
      pC ? pC->Mult(bf, bl) : Ct->MultTranspose(bf, bl);
      b_r.SetSize(pH.Ptr()->Height());
      (P_pc ? P_pc : c_pfes->GetProlongationMatrix())->MultTranspose(bl, b_r);
   }
#else
   b_r.SetSize(Ct->Width());
   Ct->MultTranspose(bf, b_r);
#endif
}

void Hybridization::ComputeSolution(const Vector &b, const Vector &sol_r,
                                    Vector &sol) const
{
   // bf = Af^{-1} ( Rf^t - Cf^t sol_r )
   Vector bf;
   MultAfInv(b, sol_r, bf, 1);

   // sol = Rf bf
   GridFunction s;
   const SparseMatrix *R = fes->GetRestrictionMatrix();
   if (!R)
   {
      MFEM_ASSERT(sol.Size() == fes->GetVSize(), "");
      s.MakeRef(fes, sol, 0);
   }
   else
   {
      s.SetSpace(fes);
      R->MultTranspose(sol, s);
   }
   const int NE = fes->GetMesh()->GetNE();
   Array<int> vdofs;
   for (int i = 0; i < NE; i++)
   {
      fes->GetElementVDofs(i, vdofs);
      for (int j = hat_offsets[i]; j < hat_offsets[i+1]; j++)
      {
         if (hat_dofs_marker[j] == 1) { continue; } // skip essential b.c.
         int vdof = vdofs[j-hat_offsets[i]];
         if (vdof >= 0) { s(vdof) = bf(j); }
         else { s(-1-vdof) = -bf(j); }
      }
   }
   if (R)
   {
      R->Mult(s, sol); // assuming that Ref = 0
   }
}

void Hybridization::Reset()
{
   delete H;
   H = NULL;
#ifdef MFEM_USE_MPI
   pH.Clear();
#endif
}

}
