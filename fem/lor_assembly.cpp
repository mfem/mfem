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

// Defined in lor_assembly_ker.cpp
template <int order, bool use_smem = true>
void Assemble3DBatchedLOR(Mesh &mesh_lor,
                          const Array<int> &dof_glob2loc,
                          const Array<int> &dof_glob2loc_offsets,
                          const Array<int> &el_dof_lex,
                          Mesh &mesh_ho,
                          SparseMatrix &A_mat);

void AssembleBatchedLOR(LORBase &lor_disc,
                        BilinearForm &form_lor,
                        FiniteElementSpace &fes_ho,
                        const Array<int> &ess_dofs_lo,
                        OperatorHandle &Ah)
{
   Mesh &mesh_lor = *form_lor.FESpace()->GetMesh();
   Mesh &mesh_ho = *fes_ho.GetMesh();
   const int dim = mesh_ho.Dimension();
   const int order = fes_ho.GetMaxElementOrder();
   dbg("dim:%d order:%d",dim,order);

   const bool has_to_init = Ah.Ptr() == nullptr;
   dbg("has_to_init: %s", has_to_init?"yes":"no");
   SparseMatrix *A = has_to_init ? nullptr : Ah.As<SparseMatrix>();

   dbg("GetLORRestriction");
   const Operator *Rop = lor_disc.GetLORRestriction();
   dbg("R");
   const LORRestriction *R = static_cast<const LORRestriction*>(Rop);
   MFEM_VERIFY(R,"LOR Restriction error!");

   if (has_to_init)
   {
      MFEM_VERIFY(UsesTensorBasis(fes_ho),
                  "Batched LOR assembly requires tensor basis");
      if (Device::IsEnabled())
      {
         dbg("Device::IsEnabled()");
#ifdef MFEM_USE_MPI
         ParBilinearForm *pform_lor = dynamic_cast<ParBilinearForm*>(&form_lor);
         assert(pform_lor);
         //ParFiniteElementSpace *pfes_ho = dynamic_cast<ParFiniteElementSpace*>(&fes_ho);

         ParFiniteElementSpace &pfes_lo = *pform_lor->ParFESpace();
         const int width = pfes_lo.GetTrueVSize();
         const int height = pfes_lo.GetTrueVSize();
         dbg("HxW: %dx%d",height,width);
#else
         FiniteElementSpace &fes_lo = *form_lor.FESpace();
         const int width = fes_lo.GetVSize();
         const int height = fes_lo.GetVSize();
#endif
         A = new SparseMatrix(height, width, 0);
         A->GetMemoryI().New(A->Height()+1, A->GetMemoryI().GetMemoryType());
         const int nnz = R->FillI(*A);
         A->GetMemoryJ().New(nnz, A->GetMemoryJ().GetMemoryType());
         A->GetMemoryData().New(nnz, A->GetMemoryData().GetMemoryType());
         R->FillJAndZeroData(*A); // J, A = 0.0
      }
      else
      {
         dbg("NOT Device::IsEnabled()");
         // the sparsity pattern is defined from the map: element->dof
         const int ndofs = fes_ho.GetTrueVSize();
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

   void (*Kernel)(Mesh &mesh_lor,
                  const Array<int> &dof_glob2loc,
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
         case 6: Kernel = Assemble3DBatchedLOR<6,false>; break;
         case 7: Kernel = Assemble3DBatchedLOR<7,false>; break;
         case 8: Kernel = Assemble3DBatchedLOR<8,false>; break;
         case 9: Kernel = Assemble3DBatchedLOR<9,false>; break;
         case 10: Kernel = Assemble3DBatchedLOR<10,false>; break;
         case 11: Kernel = Assemble3DBatchedLOR<11,false>; break;
         case 12: Kernel = Assemble3DBatchedLOR<12,false>; break;
         case 13: Kernel = Assemble3DBatchedLOR<13,false>; break;
         case 14: Kernel = Assemble3DBatchedLOR<14,false>; break;
         case 15: Kernel = Assemble3DBatchedLOR<15,false>; break;
         case 16: Kernel = Assemble3DBatchedLOR<16,false>; break;
         default: MFEM_ABORT("Kernel not ready!");
      }
   }

   Kernel(mesh_lor,
          R->Indices(),
          R->Offsets(),
          R->GatherMap(),
          mesh_ho, *A);

   // Set essential dofs to 0.0
   const int n_ess_dofs = ess_dofs_lo.Size();
   const auto ess_dofs_d = ess_dofs_lo.Read();

   const auto I = A->ReadI();
   const auto J = A->ReadJ();
   auto dA = A->ReadWriteData();

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

   A->Finalize();

   if (has_to_init) { Ah.Reset(A); } // A now owns A_mat
}

#ifdef MFEM_USE_MPI

void ParAssembleBatchedLOR(LORBase &lor_disc,
                           BilinearForm &form_lor,
                           FiniteElementSpace &fes_ho,
                           const Array<int> &ess_dofs,
                           OperatorHandle &Ah)
{
   dbg();
   //ParBilinearForm &pform_lor = static_cast<ParBilinearForm&>(form_lor);
   ParFiniteElementSpace *pfes_ho = dynamic_cast<ParFiniteElementSpace*>(&fes_ho);
   assert(pfes_ho);

   OperatorHandle A_local;
   dbg("AssembleBatchedLOR");
   AssembleBatchedLOR(lor_disc, form_lor, fes_ho, ess_dofs, A_local);

   dbg("HypreParMatrix");
   HypreParMatrix dA(pfes_ho->GetComm(), pfes_ho->GlobalVSize(),
                     pfes_ho->GetDofOffsets(), A_local.As<SparseMatrix>());
   dbg("RAP");
   Ah.Reset(RAP(&dA, pfes_ho->Dof_TrueDof_Matrix()));
}

#endif

} // namespace mfem
