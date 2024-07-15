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

#include "mfem.hpp"
#include "discrete_divergence.hpp"
#include "../../general/forall.hpp"

namespace mfem
{

/// @brief Eliminates columns in the given HypreParMatrix.
///
/// This is similar to HypreParMatrix::EliminateBC, except that only the columns
/// are eliminated.
void EliminateColumns(HypreParMatrix &D, const Array<int> &ess_dofs)
{

   hypre_ParCSRMatrix *A_hypre = D;
   D.HypreReadWrite();

   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(A_hypre);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A_hypre);

   HYPRE_Int diag_ncols = hypre_CSRMatrixNumCols(diag);
   HYPRE_Int offd_ncols = hypre_CSRMatrixNumCols(offd);

   const int n_ess_dofs = ess_dofs.Size();

   // Start communication to figure out which columns need to be eliminated in
   // the off-diagonal block
   hypre_ParCSRCommHandle *comm_handle;
   HYPRE_Int *int_buf_data, *eliminate_col_diag, *eliminate_col_offd;
   {
      eliminate_col_diag = mfem_hypre_CTAlloc_host(HYPRE_Int, diag_ncols);
      eliminate_col_offd = mfem_hypre_CTAlloc_host(HYPRE_Int, offd_ncols);

      // Make sure A has a communication package
      hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A_hypre);
      if (!comm_pkg)
      {
         hypre_MatvecCommPkgCreate(A_hypre);
         comm_pkg = hypre_ParCSRMatrixCommPkg(A_hypre);
      }

      // Which of the local columns are to be eliminated?
      for (int i = 0; i < diag_ncols; i++)
      {
         eliminate_col_diag[i] = 0;
      }

      ess_dofs.HostRead();
      for (int i = 0; i < n_ess_dofs; i++)
      {
         eliminate_col_diag[ess_dofs[i]] = 1;
      }

      // Use a matvec communication pattern to find (in eliminate_col_offd)
      // which of the local offd columns are to be eliminated
      HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      HYPRE_Int int_buf_sz = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
      int_buf_data = mfem_hypre_CTAlloc_host(HYPRE_Int, int_buf_sz);
      HYPRE_Int *send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
      for (int i = 0; i < int_buf_sz; ++i)
      {
         const int k = send_map_elmts[i];
         int_buf_data[i] = eliminate_col_diag[k];
      }
      comm_handle = hypre_ParCSRCommHandleCreate(
                       11, comm_pkg, int_buf_data, eliminate_col_offd);
   }

   // Eliminate columns in the diagonal block
   {
      Memory<HYPRE_Int> col_mem(eliminate_col_diag, diag_ncols, false);
      const auto cols = col_mem.Read(GetHypreMemoryClass(), diag_ncols);
      const int nrows_diag = hypre_CSRMatrixNumRows(diag);
      const auto I = diag->i;
      const auto J = diag->j;
      auto data = diag->data;
      mfem::hypre_forall(nrows_diag, [=] MFEM_HOST_DEVICE (int i)
      {
         for (int jj=I[i]; jj<I[i+1]; ++jj)
         {
            const int j = J[jj];
            data[jj] *= 1 - cols[j];
         }
      });
      col_mem.Delete();
   }

   // Wait for MPI communication to finish
   hypre_ParCSRCommHandleDestroy(comm_handle);
   mfem_hypre_TFree_host(int_buf_data);
   mfem_hypre_TFree_host(eliminate_col_diag);

   // Eliminate columns in the off-diagonal block
   {
      Memory<HYPRE_Int> col_mem(eliminate_col_offd, offd_ncols, false);
      const auto cols = col_mem.Read(GetHypreMemoryClass(), offd_ncols);
      const int nrows_offd = hypre_CSRMatrixNumRows(offd);
      const auto I = offd->i;
      const auto J = offd->j;
      auto data = offd->data;
      mfem::hypre_forall(nrows_offd, [=] MFEM_HOST_DEVICE (int i)
      {
         for (int jj=I[i]; jj<I[i+1]; ++jj)
         {
            const int j = J[jj];
            data[jj] *= 1 - cols[j];
         }
      });
      col_mem.Delete();
   }

   mfem_hypre_TFree_host(eliminate_col_offd);
}

void FormElementToFace2D(int order, Array<int> &element2face)
{
   const int o = order;
   const int op1 = order + 1;

   for (int iy = 0; iy < o; ++iy)
   {
      for (int ix = 0; ix < o; ++ix)
      {
         const int ivol = ix + iy*o;
         element2face[0 + 4*ivol] = -1 - (ix + iy*op1); // left, x = 0
         element2face[1 + 4*ivol] = ix+1 + iy*op1; // right, x = 1
         element2face[2 + 4*ivol] = -1 - (ix + iy*o + o*op1); // bottom, y = 0
         element2face[3 + 4*ivol] = ix + (iy+1)*o + o*op1; // top, y = 1
      }
   }
}

void FormElementToFace3D(int order, Array<int> &element2face)
{
   const int o = order;
   const int op1 = order + 1;

   const int n = o*o*op1; // number of faces per dimension

   for (int iz = 0; iz < o; ++iz)
   {
      for (int iy = 0; iy < o; ++iy)
      {
         for (int ix = 0; ix < o; ++ix)
         {
            const int ivol = ix + iy*o + iz*o*o;
            element2face[0 + 6*ivol] = -1 - (ix + iy*op1 + iz*o*op1); // x = 0
            element2face[1 + 6*ivol] = ix+1 + iy*op1 + iz*o*op1; // x = 1
            element2face[2 + 6*ivol] = -1 - (ix + iy*o + iz*o*op1 + n); // y = 0
            element2face[3 + 6*ivol] = ix + (iy+1)*o + iz*o*op1 + n; // y = 1
            element2face[4 + 6*ivol] = -1 - (ix + iy*o + iz*o*o + 2*n); // z = 0
            element2face[5 + 6*ivol] = ix + iy*o + (iz+1)*o*o + 2*n; // z = 1
         }
      }
   }
}

HypreParMatrix *FormDiscreteDivergenceMatrix(ParFiniteElementSpace &fes_rt,
                                             ParFiniteElementSpace &fes_l2,
                                             const Array<int> &ess_dofs)
{
   const Mesh &mesh = *fes_rt.GetMesh();
   const int dim = mesh.Dimension();
   const int order = fes_rt.GetMaxElementOrder();

   const int n_rt = fes_rt.GetNDofs();
   const int n_l2 = fes_l2.GetNDofs();

   SparseMatrix D_local;
   D_local.OverrideSize(n_l2, n_rt);

   D_local.GetMemoryI().New(n_l2 + 1);
   // Each row always has 2*dim nonzeros (one for each face of the element)
   const int nnz = n_l2*2*dim;
   auto I = D_local.WriteI();
   MFEM_FORALL(i, n_l2+1, I[i] = 2*dim*i; );

   const int nel_ho = mesh.GetNE();
   const int nface_per_el = dim*pow(order, dim-1)*(order+1);
   const int nvol_per_el = pow(order, dim);

   // element2face is a mapping of size (2*dim, nvol_per_el) such that with a
   // macro element, subelement i (in lexicographic ordering) has faces (also
   // in lexicographic order) given by the entries (j, i).
   Array<int> element2face;
   element2face.SetSize(2*dim*nvol_per_el);

   if (dim == 2) { FormElementToFace2D(order, element2face); }
   else if (dim == 3) { FormElementToFace3D(order, element2face); }
   else { MFEM_ABORT("Unsupported dimension.") }

   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const auto *R_rt = dynamic_cast<const ElementRestriction*>(
                         fes_rt.GetElementRestriction(ordering));
   const auto gather_rt = Reshape(R_rt->GatherMap().Read(), nface_per_el, nel_ho);
   const auto e2f = Reshape(element2face.Read(), 2*dim, nvol_per_el);

   // Fill J and data
   D_local.GetMemoryJ().New(nnz);
   D_local.GetMemoryData().New(nnz);

   auto J = D_local.WriteJ();
   auto V = D_local.WriteData();

   const int two_dim = 2*dim;

   // Loop over L2 DOFs
   MFEM_FORALL(ii, n_l2*2*dim,
   {
      const int k = ii % (two_dim);
      const int i = ii / (two_dim);
      const int i_loc = i%nvol_per_el;
      const int i_el = i/nvol_per_el;

      const int sjv_loc = e2f(k, i_loc);
      const int jv_loc = (sjv_loc >= 0) ? sjv_loc : -1 - sjv_loc;
      const int sgn1 = (sjv_loc >= 0) ? 1 : -1;
      const int sj = gather_rt(jv_loc, i_el);
      const int j = (sj >= 0) ? sj : -1 - sj;
      const int sgn2 = (sj >= 0) ? 1 : -1;

      J[k + 2*dim*i] = j;
      V[k + 2*dim*i] = sgn1*sgn2;
   });

   // Create a block diagonal parallel matrix
   OperatorHandle D_diag(Operator::Hypre_ParCSR);
   D_diag.MakeRectangularBlockDiag(fes_rt.GetComm(),
                                   fes_l2.GlobalVSize(),
                                   fes_rt.GlobalVSize(),
                                   fes_l2.GetDofOffsets(),
                                   fes_rt.GetDofOffsets(),
                                   &D_local);

   HypreParMatrix *D;
   // Assemble the parallel gradient matrix, must be deleted by the caller
   if (IsIdentityProlongation(fes_rt.GetProlongationMatrix()))
   {
      D = D_diag.As<HypreParMatrix>();
      D_diag.SetOperatorOwner(false);
      HypreStealOwnership(*D, D_local);
   }
   else
   {
      OperatorHandle Rt(Transpose(*fes_l2.GetRestrictionMatrix()));
      OperatorHandle Rt_diag(Operator::Hypre_ParCSR);
      Rt_diag.MakeRectangularBlockDiag(fes_l2.GetComm(),
                                       fes_l2.GlobalVSize(),
                                       fes_l2.GlobalTrueVSize(),
                                       fes_l2.GetDofOffsets(),
                                       fes_l2.GetTrueDofOffsets(),
                                       Rt.As<SparseMatrix>());
      D = RAP(Rt_diag.As<HypreParMatrix>(),
              D_diag.As<HypreParMatrix>(),
              fes_rt.Dof_TrueDof_Matrix());
   }
   D->CopyRowStarts();
   D->CopyColStarts();

   // Eliminate the boundary conditions
   EliminateColumns(*D, ess_dofs);

   return D;
}

} // namespace mfem
