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

namespace mfem
{

template <int order, bool use_smem = true>
void Assemble3DBatchedLOR_GPU(Mesh &mesh_lor,
                              const Array<int> &dof_glob2loc,
                              const Array<int> &dof_glob2loc_offsets,
                              const Array<int> &el_dof_lex,
                              const Array<int> &ess_dofs,
                              Mesh &mesh_ho,
                              FiniteElementSpace &fes_ho,
                              SparseMatrix &A_mat);

void AssembleBatchedLOR_GPU(LORBase &lor_disc,
                            BilinearForm &form_lor,
                            FiniteElementSpace &fes_ho,
                            const Array<int> &ess_dofs,
                            OperatorHandle &Ah)
{
   Mesh &mesh_lor = *form_lor.FESpace()->GetMesh();
   Mesh &mesh_ho = *fes_ho.GetMesh();
   const int dim = mesh_ho.Dimension();
   const int order = fes_ho.GetMaxElementOrder();

   const bool has_to_init = Ah.Ptr() == nullptr;
   //dbg("has_to_init: %s", has_to_init?"yes":"no");
   SparseMatrix *A = has_to_init ? nullptr : Ah.As<SparseMatrix>();

   const Operator *Rop = lor_disc.GetLORRestriction();
   const LORRestriction *R = static_cast<const LORRestriction*>(Rop);
   MFEM_VERIFY(R,"LOR Restriction error!");

   if (has_to_init)
   {
      MFEM_VERIFY(UsesTensorBasis(fes_ho),
                  "Batched LOR assembly requires tensor basis");
      if (Device::IsEnabled())
      {
         FiniteElementSpace &fes_lo = *form_lor.FESpace();
         const int width = fes_lo.GetVSize();
         const int height = fes_lo.GetVSize();
         A = new SparseMatrix(height, width, 0);
         A->GetMemoryI().New(A->Height()+1, A->GetMemoryI().GetMemoryType());
         const int nnz = R->FillI(*A);
         A->GetMemoryJ().New(nnz, A->GetMemoryJ().GetMemoryType());
         A->GetMemoryData().New(nnz, A->GetMemoryData().GetMemoryType());
         R->FillJAndZeroData(*A); // J, A = 0.0
      }
      else
      {
         // the sparsity pattern is defined from the map: element->dof
         const int ndofs = fes_ho.GetTrueVSize();
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
      }
   }

   void (*Kernel)(Mesh &mesh_lor,
                  const Array<int> &dof_glob2loc,
                  const Array<int> &dof_glob2loc_offsets,
                  const Array<int> &el_dof_lex,
                  const Array<int> &ess_dofs,
                  Mesh &mesh_ho,
                  FiniteElementSpace &fes_ho,
                  SparseMatrix &A_mat) = nullptr;

   if (dim == 2) { MFEM_ABORT("Unsuported!"); }
   else if (dim == 3)
   {
      switch (order)
      {
         case 1: Kernel = Assemble3DBatchedLOR_GPU<1>; break;
         case 2: Kernel = Assemble3DBatchedLOR_GPU<2>; break;
         case 3: Kernel = Assemble3DBatchedLOR_GPU<3>; break;
         case 4: Kernel = Assemble3DBatchedLOR_GPU<4>; break;
         case 5: Kernel = Assemble3DBatchedLOR_GPU<5>; break;
         case 6: Kernel = Assemble3DBatchedLOR_GPU<6,false>; break;
         case 7: Kernel = Assemble3DBatchedLOR_GPU<7,false>; break;
         case 8: Kernel = Assemble3DBatchedLOR_GPU<8,false>; break;
         case 9: Kernel = Assemble3DBatchedLOR_GPU<9,false>; break;
         case 10: Kernel = Assemble3DBatchedLOR_GPU<10,false>; break;
         case 11: Kernel = Assemble3DBatchedLOR_GPU<11,false>; break;
         case 12: Kernel = Assemble3DBatchedLOR_GPU<12,false>; break;
         case 13: Kernel = Assemble3DBatchedLOR_GPU<13,false>; break;
         case 14: Kernel = Assemble3DBatchedLOR_GPU<14,false>; break;
         case 15: Kernel = Assemble3DBatchedLOR_GPU<15,false>; break;
         case 16: Kernel = Assemble3DBatchedLOR_GPU<16,false>; break;
         default: MFEM_ABORT("Kernel not ready!");
      }
   }

   Kernel(mesh_lor,
          R->Indices(),
          R->Offsets(),
          R->GatherMap(),
          ess_dofs,
          mesh_ho, fes_ho, *A);

   if (has_to_init) { Ah.Reset(A); } // A now owns A_mat
}

} // namespace mfem
