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
#include "../general/forall.hpp"

namespace mfem
{

HybridizationExtension::HybridizationExtension(Hybridization &hybridization_)
   : h(hybridization_)
{ }

void HybridizationExtension::ConstructC()
{
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
   const ElementDofOrdering ordering = ElementDofOrdering::NATIVE;

   const Operator *R_op = h.fes.GetElementRestriction(ordering);
   const auto *R = dynamic_cast<const ElementRestriction*>(R_op);
   MFEM_VERIFY(R, "");

   const int ne = h.fes.GetNE();

   // count the number of dofs in the discontinuous version of fes:
   const int ndof_per_el = h.fes.GetFE(0)->GetDof();
   num_hat_dofs = ne*ndof_per_el;
   {
      h.hat_offsets.SetSize(ne + 1);
      int *d_hat_offsets = h.hat_offsets.Write();
      mfem::forall(ne + 1, [=] MFEM_HOST_DEVICE (int i)
      {
         d_hat_offsets[i] = i*ndof_per_el;
      });
   }

   ConstructC();

   // We now split the "hat DOFs" (broken DOFs) into three classes of DOFs, each
   // marked with an integer:
   //
   //  1: essential DOFs (depending only on essential Lagrange multiplier DOFs)
   //  0: free interior DOFs (the corresponding column in the C matrix is zero,
   //     this happens when the DOF is in the interior on an element)
   // -1: free boundary DOFs (free DOFs that lie on the interface between two
   //     elements)
   //
   // These integers are used to define the values in HatDofType enum.
   {
      const int ntdofs = h.fes.GetTrueVSize();
      // free_tdof_marker is 1 if the DOF is free, 0 if the DOF is essential
      Array<int> free_tdof_marker(ntdofs);
      {
         int *d_free_tdof_marker = free_tdof_marker.Write();
         mfem::forall(ntdofs, [=] MFEM_HOST_DEVICE (int i)
         {
            d_free_tdof_marker[i] = 1;
         });
         const int n_ess_dofs = ess_tdof_list.Size();
         const int *d_ess_tdof_list = ess_tdof_list.Read();
         mfem::forall(n_ess_dofs, [=] MFEM_HOST_DEVICE (int i)
         {
            d_free_tdof_marker[d_ess_tdof_list[i]] = 0;
         });
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

      h.hat_dofs_marker.SetSize(num_hat_dofs);
      {
         // The gather map from the ElementRestriction operator gives us the
         // index of the L-dof corresponding to a given (element, local DOF)
         // index pair.
         const int *gather_map = R->GatherMap().Read();
         const int *d_free_vdofs_marker = free_vdofs_marker.Read();
         int *d_hat_dofs_marker = h.hat_dofs_marker.Write();

         // Set the hat_dofs_marker to 1 or 0 according to whether the DOF is
         // "free" or "essential". (For now, we mark all free DOFs as free
         // interior as a placeholder). Then, as a later step, the "free" DOFs
         // will be further classified as "interior free" or "boundary free".
         mfem::forall(num_hat_dofs, [=] MFEM_HOST_DEVICE (int i)
         {
            const int j_s = gather_map[i];
            const int j = (j_s >= 0) ? j_s : -1 - j_s;
            d_hat_dofs_marker[i] = d_free_vdofs_marker[j] ? FREE_INTERIOR : ESSENTIAL;
         });

         // A free DOF is "interior" or "internal" if the corresponding column
         // of C (row of C^T) is zero. We mark the free DOFs with non-zero
         // columns of C as boundary free DOFs.
         const int *d_I = h.Ct->ReadI();
         mfem::forall(num_hat_dofs, [=] MFEM_HOST_DEVICE (int i)
         {
            const int row_size = d_I[i+1] - d_I[i];
            if (d_hat_dofs_marker[i] == 0 && row_size > 0)
            {
               d_hat_dofs_marker[i] = FREE_BOUNDARY;
            }
         });
      }
   }

   // Define Af_offsets and Af_f_offsets. Af_offsets are the offsets of the
   // matrix blocks into the data array Af_data.
   //
   // Af_f_offsets are the offets of the pivots (coming from LU factorization)
   // that are stored in the data array Af_ipiv.
   //
   // NOTE: as opposed to the non-device version of hybridization, the essential
   // DOFs are included in these matrices to ensure that all matrices have
   // identical sizes. This enabled efficient batched matrix computations.

   h.Af_offsets.SetSize(ne+1);
   h.Af_f_offsets.SetSize(ne+1);
   {
      int *Af_offsets = h.Af_offsets.Write();
      int *Af_f_offsets = h.Af_f_offsets.Write();
      mfem::forall(ne + 1, [=] MFEM_HOST_DEVICE (int i)
      {
         Af_f_offsets[i] = i*ndof_per_el;
         Af_offsets[i] = i*ndof_per_el*ndof_per_el;
      });
   }

   h.Af_data.SetSize(ne*ndof_per_el*ndof_per_el);
   h.Af_ipiv.SetSize(ne*ndof_per_el);
}

void HybridizationExtension::ComputeSolution(
   const Vector &b, const Vector &sol_r, Vector &sol) const
{
   // First, represent the solution on the "broken" space with vector 'tmp1'.

   // tmp1 = Af^{-1} ( Rf^t - Cf^t sol_r )
   tmp1.SetSize(num_hat_dofs);
   h.MultAfInv(b, sol_r, tmp1, 1);

   // The T-vector 'sol' has the correct essential boundary conditions. We
   // covert sol to an L-vector ('tmp2') to preserve those boundary values in
   // the output solution.
   const Operator *R = h.fes.GetRestrictionOperator();
   if (!R)
   {
      MFEM_ASSERT(sol.Size() == h.fes.GetVSize(), "");
      tmp2.MakeRef(sol, 0);
   }
   else
   {
      tmp2.SetSize(R->Width());
      R->MultTranspose(sol, tmp2);
   }

   // Move vector 'tmp1' from broken space to L-vector 'tmp2'.
   {
      const ElementDofOrdering ordering = ElementDofOrdering::NATIVE;
      const auto *R = static_cast<const ElementRestriction*>(
                         h.fes.GetElementRestriction(ordering));
      const int *gather_map = R->GatherMap().Read();
      const int *d_hat_dofs_marker = h.hat_dofs_marker.Read();
      const double *d_evec = tmp1.ReadWrite();
      double *d_lvec = tmp2.ReadWrite();
      mfem::forall(num_hat_dofs, [=] MFEM_HOST_DEVICE (int i)
      {
         // Skip essential DOFs
         if (d_hat_dofs_marker[i] == ESSENTIAL) { return; }

         const int j_s = gather_map[i];
         const int sgn = (j_s >= 0) ? 1 : -1;
         const int j = (j_s >= 0) ? j_s : -1 - j_s;

         d_lvec[j] = sgn*d_evec[i];
      });
   }

   // Finally, convert from L-vector to T-vector.
   if (R)
   {
      R->Mult(tmp2, sol);
   }
}

}
