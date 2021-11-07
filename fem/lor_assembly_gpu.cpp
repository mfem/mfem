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

#include "lor_assembly.hpp"
#include "../linalg/dtensor.hpp"
#include "../general/forall.hpp"

#define MFEM_DEBUG_COLOR 226
#include "../general/debug.hpp"
#include "../general/nvvp.hpp"


namespace mfem
{

template <int order>
void Assemble3DBatchedLOR_GPU(Mesh &mesh_lor,
                              Array<int> &dof_glob2loc_,
                              Array<int> &dof_glob2loc_offsets_,
                              Array<int> &el_dof_lex_,
                              Vector &Q_,
                              //Vector &el_vert_,
                              Mesh &mesh_ho,
                              FiniteElementSpace &fes_ho,
                              SparseMatrix &A_mat);

void AssembleBatchedLOR_GPU(BilinearForm &form_lor,
                            FiniteElementSpace &fes_ho,
                            const Array<int> &ess_dofs,
                            OperatorHandle &Ah)
{
   Mesh &mesh_lor = *form_lor.FESpace()->GetMesh();
   Mesh &mesh_ho = *fes_ho.GetMesh();
   const int dim = mesh_ho.Dimension();
   const int order = fes_ho.GetMaxElementOrder();
   const int ndofs = fes_ho.GetTrueVSize();

   const int nel_ho = mesh_ho.GetNE();
   //const int nel_lor = mesh_lor.GetNE();
   const int ndof = fes_ho.GetVSize();
   constexpr int nv = 8;
   const int ddm2 = (dim*(dim+1))/2;
   const int nd1d = order + 1;
   const int ndof_per_el = nd1d*nd1d*nd1d;

   const bool has_to_init = Ah.Ptr() == nullptr;
   //dbg("has_to_init: %s", has_to_init?"yes":"no");
   SparseMatrix *A_mat = Ah.As<SparseMatrix>();

   Array<int> dof_glob2loc_(2*ndof_per_el*nel_ho);
   Array<int> dof_glob2loc_offsets_(ndof+1);
   Array<int> el_dof_lex_(ndof_per_el*nel_ho);
   Vector Q_(nel_ho*pow(order,dim)*nv*ddm2);
   //Vector el_vert_(dim*nv*nel_lor);


   if (has_to_init)
   {
      // Seems to need this each time...
      //if (!mesh_lor.GetNodes())
      {
         mesh_lor.SetCurvature(1, false, -1, Ordering::byNODES);
      }
      NvtxPush(AssembleBatchedLOR_GPU, LawnGreen);
      MFEM_VERIFY(UsesTensorBasis(fes_ho),
                  "Batched LOR assembly requires tensor basis");

      // the sparsity pattern is defined from the map: element->dof
      const Table &elem_dof = form_lor.FESpace()->GetElementToDofTable();

      NvtxPush(Sparsity, PaleTurquoise);
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
      A_mat = new SparseMatrix(I,J,data,ndofs,ndofs,true,true,true);
      NvtxPop();

      NvtxPush(Ah=0.0, Peru);
      *A_mat = 0.0;
      NvtxPop();

      NvtxPush(LoseData, PaleTurquoise);
      dof_dof.LoseData();
      NvtxPop();

      {
         NvtxPush(BlockMapping, Olive);
         Array<int> dofs;
         const Array<int> &lex_map =
            dynamic_cast<const NodalFiniteElement&>
            (*fes_ho.GetFE(0)).GetLexicographicOrdering();
         dof_glob2loc_offsets_ = 0;
         for (int iel_ho=0; iel_ho<nel_ho; ++iel_ho)
         {
            fes_ho.GetElementDofs(iel_ho, dofs);
            for (int i=0; i<ndof_per_el; ++i)
            {
               const int dof = dofs[lex_map[i]];
               el_dof_lex_[i + iel_ho*ndof_per_el] = dof;
               dof_glob2loc_offsets_[dof+1] += 2;
            }
         }
         dof_glob2loc_offsets_.PartialSum();
         // Sanity check
         MFEM_VERIFY(dof_glob2loc_offsets_[ndof] == dof_glob2loc_.Size(), "");
         Array<int> dof_ptr(ndof);
         for (int i=0; i<ndof; ++i) { dof_ptr[i] = dof_glob2loc_offsets_[i]; }
         for (int iel_ho=0; iel_ho<nel_ho; ++iel_ho)
         {
            fes_ho.GetElementDofs(iel_ho, dofs);
            for (int i=0; i<ndof_per_el; ++i)
            {
               const int dof = dofs[lex_map[i]];
               dof_glob2loc_[dof_ptr[dof]++] = iel_ho;
               dof_glob2loc_[dof_ptr[dof]++] = i;
            }
         }
         NvtxPop(BlockMapping);
      }

      /*{
         NvtxPush(GetVertices, IndianRed);
         for (int iel_lor=0; iel_lor<nel_lor; ++iel_lor)
         {
            Array<int> v;
            mesh_lor.GetElementVertices(iel_lor, v);
            for (int iv=0; iv<nv; ++iv)
            {
               const double *vc = mesh_lor.GetVertex(v[iv]);
               for (int d=0; d<dim; ++d)
               {
                  el_vert_[d + iv*dim + iel_lor*nv*dim] = vc[d];
               }
            }
         }
         NvtxPop(GetVertices);
      }*/
      NvtxPop(Sparsity);
   }

   void (*Kernel)(Mesh &mesh_lor,
                  Array<int> &dof_glob2loc_,
                  Array<int> &dof_glob2loc_offsets_,
                  Array<int> &el_dof_lex_,
                  Vector &Q_,
                  //Vector &el_vert_,
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

   Kernel(mesh_lor,
          dof_glob2loc_,
          dof_glob2loc_offsets_,
          el_dof_lex_,
          Q_,
          //el_vert_,
          mesh_ho,fes_ho,*A_mat);

   {
      NvtxPush(Diag=0.0, DarkGoldenrod);
      const auto I_d = A_mat->ReadI();
      const auto J_d = A_mat->ReadJ();
      auto A_d = A_mat->ReadWriteData();
      const int n_ess_dofs = ess_dofs.Size();

      MFEM_FORALL(i, n_ess_dofs,
      {
         for (int j=I_d[i]; j<I_d[i+1]; ++j)
         {
            if (J_d[j] != i)
            {
               A_d[j] = 0.0;
            }
         }
      });
      NvtxPop();
   }

   if (has_to_init) { Ah.Reset(A_mat); } // A now owns A_mat
   NvtxPop(AssembleBatchedLOR_GPU);
}

} // namespace mfem
