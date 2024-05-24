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

#include "hybridization_ext.hpp"
#include "hybridization.hpp"

namespace mfem
{

HybridizationExtension::HybridizationExtension(Hybridization &hybridization_)
   : h(hybridization_)
{ }

void HybridizationExtension::ConstructC()
{
   FiniteElementSpace &fes = h.fes;

   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *R_op = fes.GetElementRestriction(ordering);
   const auto *R = dynamic_cast<const ElementRestriction*>(R_op);

   // const int NE = fes.GetNE();
   int num_hat_dofs = R->Height();

   Array<int> vdofs, c_vdofs;
   int c_num_face_nbr_dofs = 0;

   const int c_vsize = h.c_fes.GetVSize();
   h.Ct.reset(new SparseMatrix(num_hat_dofs, c_vsize + c_num_face_nbr_dofs));

   MFEM_VERIFY(h.c_bfi, "");

   const int skip_zeros = 1;
   DenseMatrix elmat;
   FaceElementTransformations *FTr;
   Mesh &mesh = *h.fes.GetMesh();
   int num_faces = mesh.GetNumFaces();
   for (int i = 0; i < num_faces; i++)
   {
      FTr = mesh.GetInteriorFaceTransformations(i);
      if (!FTr) { continue; }

      int o1 = h.hat_offsets[FTr->Elem1No];
      int s1 = h.hat_offsets[FTr->Elem1No+1] - o1;
      int o2 = h.hat_offsets[FTr->Elem2No];
      int s2 = h.hat_offsets[FTr->Elem2No+1] - o2;
      vdofs.SetSize(s1 + s2);
      for (int j = 0; j < s1; j++)
      {
         vdofs[j] = o1 + j;
      }
      for (int j = 0; j < s2; j++)
      {
         vdofs[s1+j] = o2 + j;
      }
      h.c_fes.GetFaceVDofs(i, c_vdofs);
      h.c_bfi->AssembleFaceMatrix(*h.c_fes.GetFaceElement(i),
                                  *h.fes.GetFE(FTr->Elem1No),
                                  *h.fes.GetFE(FTr->Elem2No),
                                  *FTr, elmat);
      // zero-out small elements in elmat
      elmat.Threshold(1e-12 * elmat.MaxMaxNorm());
      h.Ct->AddSubMatrix(vdofs, c_vdofs, elmat, skip_zeros);
   }
   h.Ct->Finalize(skip_zeros);
}

void HybridizationExtension::Init(const Array<int> &ess_tdof_list)
{
   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *R_op = h.fes.GetElementRestriction(ordering);
   const auto *R = dynamic_cast<const ElementRestriction*>(R_op);
   MFEM_VERIFY(R, "");

   Array<int> vdofs;
   const int NE = h.fes.GetNE();

   // count the number of dofs in the discontinuous version of fes:
   int num_hat_dofs = 0;
   h.hat_offsets.SetSize(NE+1);
   h.hat_offsets[0] = 0;
   for (int i = 0; i < NE; i++)
   {
      h.fes.GetElementVDofs(i, vdofs);
      num_hat_dofs += vdofs.Size();
      h.hat_offsets[i+1] = num_hat_dofs;
   }

   ConstructC();

   // Define the "free" (0) and "essential" (1) hat_dofs.
   // The "essential" hat_dofs are those that depend only on essential cdofs;
   // all other hat_dofs are "free".
   h.hat_dofs_marker.SetSize(num_hat_dofs);
   Array<int> free_tdof_marker;
   free_tdof_marker.SetSize(h.fes.GetConformingVSize());
   free_tdof_marker = 1;
   for (int i = 0; i < ess_tdof_list.Size(); i++)
   {
      free_tdof_marker[ess_tdof_list[i]] = 0;
   }
   Array<int> free_vdofs_marker;

   const SparseMatrix *cP = h.fes.GetConformingProlongation();
   if (!cP)
   {
      free_vdofs_marker.MakeRef(free_tdof_marker);
   }
   else
   {
      free_vdofs_marker.SetSize(h.fes.GetVSize());
      cP->BooleanMult(free_tdof_marker, free_vdofs_marker);
   }

   for (int i = 0; i < NE; i++)
   {
      h.fes.GetElementVDofs(i, vdofs);
      FiniteElementSpace::AdjustVDofs(vdofs);
      for (int j = 0; j < vdofs.Size(); j++)
      {
         h.hat_dofs_marker[h.hat_offsets[i]+j] = ! free_vdofs_marker[vdofs[j]];
      }
   }
   free_tdof_marker.DeleteAll();
   free_vdofs_marker.DeleteAll();
   // Split the "free" (0) hat_dofs into "internal" (0) or "boundary" (-1).
   // The "internal" hat_dofs are those "free" hat_dofs for which the
   // corresponding column in C is zero; otherwise the free hat_dof is
   // "boundary".
   for (int i = 0; i < num_hat_dofs; i++)
   {
      // skip "essential" hat_dofs and empty rows in Ct
      if (h.hat_dofs_marker[i] != 1 && h.Ct->RowSize(i) > 0)
      {
         h.hat_dofs_marker[i] = -1; // mark this hat_dof as "boundary"
      }
   }

   // Define Af_offsets and Af_f_offsets
   h.Af_offsets.SetSize(NE+1);
   h.Af_offsets[0] = 0;
   h.Af_f_offsets.SetSize(NE+1);
   h.Af_f_offsets[0] = 0;
   for (int i = 0; i < NE; i++)
   {
      int f_size = 0; // count the "free" hat_dofs in element i
      for (int j = h.hat_offsets[i]; j < h.hat_offsets[i+1]; j++)
      {
         if (h.hat_dofs_marker[j] != 1) { f_size++; }
      }
      h.Af_offsets[i+1] = h.Af_offsets[i] + f_size*f_size;
      h.Af_f_offsets[i+1] = h.Af_f_offsets[i] + f_size;
   }

   h.Af_data.SetSize(h.Af_offsets[NE]);
   h.Af_ipiv.SetSize(h.Af_f_offsets[NE]);
}

}
