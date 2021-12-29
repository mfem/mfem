// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "lor.hpp"
#include "lor_assembly.hpp"
#include "../linalg/dtensor.hpp"
#include "../general/forall.hpp"

#define MFEM_DEBUG_COLOR 226
#include "../general/debug.hpp"

#define MFEM_NVTX_COLOR DarkOrchid
#include "../general/nvtx.hpp"

namespace mfem
{

// Defined in lor_assembly_ker.cpp
template <int order, bool use_smem = true>
void Assemble3DBatchedLOR(const Array<int> &dof_glob2loc,
                          const Array<int> &dof_glob2loc_offsets,
                          const Array<int> &el_dof_lex,
                          Mesh &mesh_ho,
                          SparseMatrix &A_mat);

static void AssembleBatchedLORWithoutBC(LORBase &lor_disc,
                                        BilinearForm &form_lor,
                                        FiniteElementSpace &fes_ho,
                                        OperatorHandle &Ah)
{
   MFEM_NVTX;
   Mesh &mesh_ho = *fes_ho.GetMesh();
   const int dim = mesh_ho.Dimension();
   const int order = fes_ho.GetMaxElementOrder();
   dbg("dim:%d order:%d",dim,order);

   const bool has_to_init = Ah.Ptr() == nullptr;
   dbg("has_to_init: %s", has_to_init?"yes":"no");
   SparseMatrix *A = has_to_init ? nullptr : Ah.As<SparseMatrix>();

   dbg("GetLORRestriction");
   const LORRestriction *R = lor_disc.GetLORRestriction();
   MFEM_VERIFY(R,"LOR Restriction error!");

   if (has_to_init)
   {
      MFEM_VERIFY(UsesTensorBasis(fes_ho),
                  "Batched LOR assembly requires tensor basis");
      if (Device::IsEnabled()||true)
      {
         dbg("Device::IsEnabled()");
#ifdef MFEM_USE_MPI
         ParFiniteElementSpace *pfes_ho =
            dynamic_cast<ParFiniteElementSpace*>(&fes_ho);
         if (pfes_ho && pfes_ho->GetNRanks() > 1)
         {
            dbg("Device::IsEnabled() and multiple ranks!");
            const int width = pfes_ho->GetVSize();
            const int height = pfes_ho->GetVSize();
            dbg("HxW: %dx%d",height,width);
            A = new SparseMatrix(height, width, 0);
         }
         else
#endif
         {
            dbg("Device::IsEnabled() but one rank!");
            const int width = fes_ho.GetVSize();
            const int height = fes_ho.GetVSize();
            dbg("HxW: %dx%d",height,width);
            A = new SparseMatrix(height, width, 0);
         }
         A->GetMemoryI().New(A->Height()+1, A->GetMemoryI().GetMemoryType());
         const int nnz = R->FillI(*A);
         A->GetMemoryJ().New(nnz, A->GetMemoryJ().GetMemoryType());
         A->GetMemoryData().New(nnz, A->GetMemoryData().GetMemoryType());
         R->FillJAndZeroData(*A); // J, A = 0.0
      }
      else
      {
         MFEM_ABORT("");
         dbg("NOT Device::IsEnabled()");
         // the sparsity pattern is defined from the map: element->dof
         const int ndofs = fes_ho.GetVSize();
         dbg("ndofs:%d",ndofs);
         const Table &elem_dof = form_lor.FESpace()->GetElementToDofTable();
         Table dof_dof, dof_elem;
         Transpose(elem_dof, dof_elem, ndofs);
         mfem::Mult(dof_elem, elem_dof, dof_dof);
         dof_dof.SortRows();
         int *I = dof_dof.GetI();
         int *J = dof_dof.GetJ();
         double *data = Memory<double>(I[ndofs]);
         A = new SparseMatrix(I,J,data,ndofs,ndofs,true,true,true);
         dof_dof.LoseData();
         *A = 0.0;
         dbg("done");
      }
   }

   void (*Kernel)(const Array<int> &dof_glob2loc,
                  const Array<int> &dof_glob2loc_offsets,
                  const Array<int> &el_dof_lex,
                  Mesh &mesh_ho,
                  SparseMatrix &A_mat) = nullptr;

   if (dim == 2) { MFEM_ABORT("Unsuported!"); }
   else if (dim == 3)
   {
      switch (order)
      {
         case 1: Kernel = Assemble3DBatchedLOR<1>; break;
         case 2: Kernel = Assemble3DBatchedLOR<2>; break;
         case 3: Kernel = Assemble3DBatchedLOR<3>; break;
         case 4: Kernel = Assemble3DBatchedLOR<4>; break;
         case 5: Kernel = Assemble3DBatchedLOR<5>; break;
         case 6: Kernel = Assemble3DBatchedLOR<6,false>; break;/*
         case 7: Kernel = Assemble3DBatchedLOR<7,false>; break;
         case 8: Kernel = Assemble3DBatchedLOR<8,false>; break;
         case 9: Kernel = Assemble3DBatchedLOR<9,false>; break;
         case 10: Kernel = Assemble3DBatchedLOR<10,false>; break;
         case 11: Kernel = Assemble3DBatchedLOR<11,false>; break;
         case 12: Kernel = Assemble3DBatchedLOR<12,false>; break;
         case 13: Kernel = Assemble3DBatchedLOR<13,false>; break;
         case 14: Kernel = Assemble3DBatchedLOR<14,false>; break;
         case 15: Kernel = Assemble3DBatchedLOR<15,false>; break;
         case 16: Kernel = Assemble3DBatchedLOR<16,false>; break;*/
         default: MFEM_ABORT("Kernel not ready!");
      }
   }

   Kernel(R->Indices(),
          R->Offsets(),
          R->GatherMap(),
          mesh_ho, *A);

   A->Finalize();

   if (has_to_init) { Ah.Reset(A); } // A now owns A_mat
}


void AssembleBatchedLOR(LORBase &lor_disc,
                        BilinearForm &form_lor,
                        FiniteElementSpace &fes_ho,
                        const Array<int> &ess_dofs,
                        OperatorHandle &Ah)
{
   MFEM_NVTX;
   AssembleBatchedLORWithoutBC(lor_disc, form_lor, fes_ho, Ah);

   // Set essential dofs to 0.0
   const int n_ess_dofs = ess_dofs.Size();
   const auto ess_dofs_d = ess_dofs.Read();

   const auto I = Ah.As<SparseMatrix>()->ReadI();
   const auto J = Ah.As<SparseMatrix>()->ReadJ();
   auto dA = Ah.As<SparseMatrix>()->ReadWriteData();

   MFEM_FORALL(i, n_ess_dofs,
   {
      const int idof = ess_dofs_d[i];
      for (int j=I[idof]; j<I[idof+1]; ++j)
      {
         const int jdof = J[j];
         if (jdof != idof)
         {
            dA[j] = 0.0;
            for (int k=I[jdof]; k<I[jdof+1]; ++k)
            {
               if (J[k] == idof)
               {
                  dA[k] = 0.0;
                  break;
               }
            }
         }
      }
   });
}


#ifdef MFEM_USE_MPI

void ParAssembleBatchedLOR(LORBase &lor_disc,
                           BilinearForm &form_lor,
                           FiniteElementSpace &fes_ho,
                           const Array<int> &ess_dofs,
                           OperatorHandle &Ah)
{
   dbg();
   MFEM_NVTX;
   ParFiniteElementSpace *pfes_ho =
      dynamic_cast<ParFiniteElementSpace*>(&fes_ho);
   assert(pfes_ho);

   OperatorHandle A_local(Operator::MFEM_SPARSEMAT);
   AssembleBatchedLORWithoutBC(lor_disc, form_lor, fes_ho, A_local);
   MFEM_VERIFY(A_local.As<SparseMatrix>()->Finalized(),
               "the local matrix must be finalized");

   NVTX("Parallel");
   OperatorHandle dA(Operator::Hypre_ParCSR),
                  Ph(Operator::Hypre_ParCSR);
   {
      NVTX("MakeSquareBlockDiag");
      dA.MakeSquareBlockDiag(pfes_ho->GetComm(),
                             pfes_ho->GlobalVSize(),
                             pfes_ho->GetDofOffsets(),
                             A_local.As<SparseMatrix>());
   }
   Ph.ConvertFrom(pfes_ho->Dof_TrueDof_Matrix());

   {
      NVTX("MakePtAP");
      Ah.MakePtAP(dA, Ph);
   }

   /*{
      NVTX("EliminateRowsCols");
      Ah.As<HypreParMatrix>()->EliminateRowsCols(ess_dofs);
   }*/

   {
      dbg("EliminateRowsCols");
      NVTX("EliminateRowsCols");
      HypreParMatrix *A_mat = Ah.As<HypreParMatrix>();
      hypre_ParCSRMatrix *A = *A_mat;
      A_mat->HypreReadWrite();

      hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(A);
      hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A);

      HYPRE_Int diag_nrows = hypre_CSRMatrixNumRows(diag);
      HYPRE_Int offd_ncols = hypre_CSRMatrixNumCols(offd);
      dbg("diag_nrows:%d offd_ncols:%d", diag_nrows, offd_ncols);

      const int n_ess_dofs = ess_dofs.Size();
      const auto ess_dofs_d = ess_dofs.Read();
      dbg("n_ess_dofs:%d", n_ess_dofs);

      // Start communication to figure out which columns need to be eliminated in
      // the off-diagonal block
      hypre_ParCSRCommHandle *comm_handle;
      HYPRE_Int *int_buf_data, *eliminate_row, *eliminate_col;
      {
         eliminate_row = hypre_CTAlloc(HYPRE_Int, diag_nrows, HYPRE_MEMORY_HOST);
         eliminate_col = hypre_CTAlloc(HYPRE_Int, offd_ncols, HYPRE_MEMORY_HOST);

         // Make sure A has a communication package
         hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
         if (!comm_pkg)
         {
            hypre_MatvecCommPkgCreate(A);
            comm_pkg = hypre_ParCSRMatrixCommPkg(A);
         }

         // Which of the local rows are to be eliminated?
         for (int i = 0; i < diag_nrows; i++)
         {
            eliminate_row[i] = 0;
         }

         ess_dofs.HostRead();
         for (int i = 0; i < n_ess_dofs; i++)
         {
            eliminate_row[ess_dofs[i]] = 1;
         }

         // Use a matvec communication pattern to find (in eliminate_col) which of
         // the local offd columns are to be eliminated
         HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
         dbg("num_sends:%d", num_sends);
         int_buf_data =
            hypre_CTAlloc(HYPRE_Int,
                          hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                          HYPRE_MEMORY_HOST);
         int index = 0;
         for (int i = 0; i < num_sends; i++)
         {
            int start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (int j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            {
               int k = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
               int_buf_data[index++] = eliminate_row[k];
            }
         }
         comm_handle = hypre_ParCSRCommHandleCreate(
                          11, comm_pkg, int_buf_data, eliminate_col);
      }

      // Eliminate rows and columns in the diagonal block
      {
         dbg("Eliminate rows and columns in the diagonal block");
         const auto I = diag->i;
         const auto J = diag->j;
         auto data = diag->data;

         MFEM_FORALL(i, n_ess_dofs,
         {
            const int idof = ess_dofs_d[i];
            for (int j=I[idof]; j<I[idof+1]; ++j)
            {
               const int jdof = J[j];
               if (jdof != idof)
               {
                  data[j] = 0.0;
                  for (int k=I[jdof]; k<I[jdof+1]; ++k)
                  {
                     if (J[k] == idof)
                     {
                        data[k] = 0.0;
                        break;
                     }
                  }
               }
            }
         });
      }

      // Eliminate rows in the off-diagonal block
      {
         dbg("Eliminate rows in the off-diagonal block");
         const auto I = offd->i;
         auto data = offd->data;
         MFEM_FORALL(i, n_ess_dofs,
         {
            const int idof = ess_dofs_d[i];
            for (int j=I[idof]; j<I[idof+1]; ++j)
            {
               data[j] = 0.0;
            }
         });
      }

      // Wait for MPI communication to finish
      Array<HYPRE_Int> cols_to_eliminate;
      {
         dbg("Wait for MPI communication to finish");
         hypre_ParCSRCommHandleDestroy(comm_handle);

         // set the array cols_to_eliminate
         int ncols_to_eliminate = 0;
         for (int i = 0; i < offd_ncols; i++)
         {
            if (eliminate_col[i]) { ncols_to_eliminate++; }
         }

         cols_to_eliminate.SetSize(ncols_to_eliminate);
         cols_to_eliminate = 0.0;

         ncols_to_eliminate = 0;
         for (int i = 0; i < offd_ncols; i++)
         {
            if (eliminate_col[i])
            {
               cols_to_eliminate[ncols_to_eliminate++] = i;
            }
         }

         hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
         hypre_TFree(eliminate_row, HYPRE_MEMORY_HOST);
         hypre_TFree(eliminate_col, HYPRE_MEMORY_HOST);
      }

      // Eliminate columns in the off-diagonal block
      {
         dbg("Eliminate columns in the off-diagonal block");
         const int ncols_to_eliminate = cols_to_eliminate.Size();
         const int nrows_offd = hypre_CSRMatrixNumRows(offd);
         const auto cols = cols_to_eliminate.Read();
         const auto I = offd->i;
         const auto J = offd->j;
         auto data = offd->data;
         dbg("ncols_to_eliminate:%d nrows_offd:%d", ncols_to_eliminate, nrows_offd);
         // Note: could also try a different strategy, looping over nnz in the
         // matrix and then doing a binary search in ncols_to_eliminate to see if
         // the column should be eliminated.
         MFEM_FORALL(idx, ncols_to_eliminate,
         {
            const int j = cols[idx];
            for (int i=0; i<nrows_offd; ++i)
            {
               for (int jj=I[i]; jj<I[i+1]; ++jj)
               {
                  if (J[jj] == j)
                  {
                     data[jj] = 0.0;
                     break;
                  }
               }
            }
         });
      }
   }
}

#endif // MFEM_USE_MPI

} // namespace mfem
