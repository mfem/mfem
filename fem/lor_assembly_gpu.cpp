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

//#include "../general/nvvp.hpp"
#define NvtxPush(...)
#define NvtxPop(...)

namespace mfem
{

template <int order>
void Assemble3DBatchedLOR_GPU(Mesh &mesh_lor,
                              const Array<int> &dof_glob2loc,
                              const Array<int> &dof_glob2loc_offsets,
                              const Array<int> &el_dof_lex,
                              const Array<int> &ess_dofs,
                              Vector &Q_,
                              Mesh &mesh_ho,
                              FiniteElementSpace &fes_ho,
                              SparseMatrix &A_mat);

void AssembleBatchedLOR_GPU(LORBase &lor_disc,
                            BilinearForm &form_lor,
                            FiniteElementSpace &fes_ho,
                            const Array<int> &ess_dofs,
                            OperatorHandle &Ah)
{
   NvtxPush(AssembleBatchedLOR_GPU, LawnGreen);

   Mesh &mesh_lor = *form_lor.FESpace()->GetMesh();
   Mesh &mesh_ho = *fes_ho.GetMesh();
   const int dim = mesh_ho.Dimension();
   const int order = fes_ho.GetMaxElementOrder();

   const bool has_to_init = Ah.Ptr() == nullptr;
   //dbg("has_to_init: %s", has_to_init?"yes":"no");
   SparseMatrix *A = has_to_init ? nullptr : Ah.As<SparseMatrix>();

   NvtxPush(R,Purple);
   const Operator *Rop = lor_disc.GetLORRestriction();
   const LORRestriction *R = static_cast<const LORRestriction*>(Rop);
   MFEM_VERIFY(R,"LOR Restriction error!");
   NvtxPop();

   if (has_to_init)
   {
      MFEM_VERIFY(UsesTensorBasis(fes_ho),
                  "Batched LOR assembly requires tensor basis");

      NvtxPush(Sparsity, PaleTurquoise);

      if (Device::IsEnabled())
      {
         NvtxPush(DEV, LightGoldenrod);
         FiniteElementSpace &fes_lo = *form_lor.FESpace();
         const int width = fes_lo.GetVSize();
         const int height = fes_lo.GetVSize();

         NvtxPush(new mat,PapayaWhip);
         A = new SparseMatrix(height, width, 0);
         NvtxPop();

         NvtxPush(newI,Purple);
         A->GetMemoryI().New(A->Height()+1, A->GetMemoryI().GetMemoryType());
         NvtxPop();

         NvtxPush(I,Azure);
         const int nnz = R->FillI(*A);
         NvtxPop();

         NvtxPush(NewJ,Orchid);
         A->GetMemoryJ().New(nnz, A->GetMemoryJ().GetMemoryType());
         NvtxPop();

         NvtxPush(NewData,Khaki);
         A->GetMemoryData().New(nnz, A->GetMemoryData().GetMemoryType());
         NvtxPop();

         NvtxPush(J,Azure);
         R->FillJAndZeroData(*A); // J, A = 0.0
         NvtxPop();

         NvtxPop(DEV);
      }
      else
      {
         NvtxPush(CPU,DarkKhaki);
         const int ndofs = fes_ho.GetTrueVSize();

         // the sparsity pattern is defined from the map: element->dof
         const Table &elem_dof = form_lor.FESpace()->GetElementToDofTable();

         Table dof_dof, dof_elem;

         NvtxPush(Transpose, LightGoldenrod);
         Transpose(elem_dof, dof_elem, ndofs);
         NvtxPop(Transpose);

         NvtxPush(Mult, LightGoldenrod);
         mfem::Mult(dof_elem, elem_dof, dof_dof);
         NvtxPop();

         NvtxPush(SortRows, LightGoldenrod);
         dof_dof.SortRows();
         int *I = dof_dof.GetI();
         int *J = dof_dof.GetJ();
         NvtxPop();

         NvtxPush(A_Allocate, Cyan);
         double *data = Memory<double>(I[ndofs]);
         NvtxPop();

         NvtxPush(newSparseMatrix, PeachPuff);
         A = new SparseMatrix(I,J,data,ndofs,ndofs,true,true,true);
         NvtxPop();

         NvtxPush(LoseData, PaleTurquoise);
         dof_dof.LoseData();
         NvtxPop();
         NvtxPop(CPU);

         NvtxPush(A=0.0, Peru);
         *A = 0.0;
         NvtxPop();
      } // IJ

      NvtxPop(Sparsity);
   }

   void (*Kernel)(Mesh &mesh_lor,
                  const Array<int> &dof_glob2loc,
                  const Array<int> &dof_glob2loc_offsets,
                  const Array<int> &el_dof_lex,
                  const Array<int> &ess_dofs,
                  Vector &Q_,
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
         default: MFEM_ABORT("Kernel not ready!");
      }
   }

   NvtxPush(KerBatched, GreenYellow);
   Kernel(mesh_lor,
          R->Indices(),
          R->Offsets(),
          R->GatherMap(),
          ess_dofs,
          R->GetQ(),
          mesh_ho, fes_ho, *A);
   NvtxPop(KerBatched);

   if (has_to_init) { Ah.Reset(A); } // A now owns A_mat
   NvtxPop(AssembleBatchedLOR_GPU);
}

} // namespace mfem
