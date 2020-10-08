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

#include "ceed_algebraic.hpp"

#ifdef MFEM_USE_CEED
#include "../fem/bilinearform.hpp"
#include "../fem/fespace.hpp"
#include "../fem/libceed/ceedsolvers-atpmg.h"
#include "../fem/libceed/ceedsolvers-interpolation.h"
#include "../fem/pfespace.hpp"

namespace mfem
{

Solver *BuildSmootherFromCeed(
   Operator *mfem_op,
   CeedOperator ceed_op,
   const Array<int>& ess_tdofs,
   const Operator *P,
   bool chebyshev
)
{
   // this is a local diagonal, in the sense of l-vector
   CeedVector diagceed;
   CeedInt length;
   Ceed ceed;
   CeedOperatorGetCeed(ceed_op, &ceed);
   CeedOperatorGetSize(ceed_op, &length);
   CeedVectorCreate(ceed, length, &diagceed);
   CeedVectorSetValue(diagceed, 0.0);
   CeedOperatorLinearAssembleDiagonal(ceed_op, diagceed, CEED_REQUEST_IMMEDIATE);
   const CeedScalar * diagvals;
   CeedMemType mem;
   CeedGetPreferredMemType(ceed, &mem);
   if ( Device::Allows(Backend::CUDA) && mem==CEED_MEM_DEVICE )
   {
      // intentional no-op
   }
   else
   {
      mem = CEED_MEM_HOST;
   }
   CeedVectorGetArrayRead(diagceed, mem, &diagvals);
   Vector local_diag(const_cast<CeedScalar*>(diagvals), length);
   Vector mfem_diag;
   if (P)
   {
      mfem_diag.SetSize(P->Width());
      P->MultTranspose(local_diag, mfem_diag);
   }
   else
   {
      mfem_diag.SetDataAndSize(local_diag, local_diag.Size());
   }
   Solver * out = NULL;
   if (chebyshev)
   {
      const int cheb_order = 3;
      out = new OperatorChebyshevSmoother(mfem_op, mfem_diag, ess_tdofs, cheb_order);
   }
   else
   {
      const double jacobi_scale = 0.65;
      out = new OperatorJacobiSmoother(mfem_diag, ess_tdofs, jacobi_scale);
   }
   CeedVectorRestoreArrayRead(diagceed, &diagvals);
   CeedVectorDestroy(&diagceed);
   return out;
}

void CoarsenEssentialDofs(const Operator& mfem_interp,
                          const Array<int>& ho_ess_tdof_list,
                          Array<int>& alg_lo_ess_tdof_list)
{
   Vector ho_boundary_ones(mfem_interp.Height());
   ho_boundary_ones = 0.0;
   for (int k : ho_ess_tdof_list)
   {
      ho_boundary_ones(k) = 1.0;
   }
   Vector lo_boundary_ones(mfem_interp.Width());
   mfem_interp.MultTranspose(ho_boundary_ones, lo_boundary_ones);
   auto lobo = lo_boundary_ones.HostRead();
   for (int i = 0; i < lo_boundary_ones.Size(); ++i)
   {
      if (lobo[i] > 0.9)
      {
         alg_lo_ess_tdof_list.Append(i);
      }
   }
}

template <typename INTEG>
void TryToAddCeedSubOperator(BilinearFormIntegrator *integ_in, CeedOperator op)
{
   INTEG *integ = dynamic_cast<INTEG*>(integ_in);
   if (integ != NULL)
   {
      CeedCompositeOperatorAddSub(op, integ->GetCeedData()->oper);
   }
}

CeedOperator CreateCeedCompositeOperatorFromBilinearForm(BilinearForm &form)
{
   CeedOperator op;
   CeedCompositeOperatorCreate(internal::ceed, &op);

   // Get the domain bilinear form integrators (DBFIs)
   Array<BilinearFormIntegrator*> *bffis = form.GetDBFI();
   int num_integrators = bffis->Size();

   for (int i = 0; i < num_integrators; ++i)
   {
      BilinearFormIntegrator *integ = (*bffis)[i];
      TryToAddCeedSubOperator<DiffusionIntegrator>(integ, op);
      TryToAddCeedSubOperator<MassIntegrator>(integ, op);
      TryToAddCeedSubOperator<VectorDiffusionIntegrator>(integ, op);
      TryToAddCeedSubOperator<VectorMassIntegrator>(integ, op);
   }
   return op;
}

CeedOperator CoarsenCeedCompositeOperator(
   CeedOperator op,
   CeedElemRestriction er,
   CeedBasis c2f,
   int order_reduction
)
{
   bool isComposite;
   CeedOperatorIsComposite(op, &isComposite);
   MFEM_ASSERT(isComposite, "");

   CeedOperator op_coarse;
   CeedCompositeOperatorCreate(internal::ceed, &op_coarse);

   int nsub;
   CeedOperatorGetNumSub(op, &nsub);
   CeedOperator *subops;
   CeedOperatorGetSubList(op, &subops);
   for (int isub=0; isub<nsub; ++isub)
   {
      CeedOperator subop = subops[isub];
      CeedBasis basis_coarse, basis_c2f;
      CeedOperator subop_coarse;
      CeedATPMGOperator(subop, order_reduction, er, &basis_coarse, &basis_c2f, &subop_coarse);
      CeedBasisDestroy(&basis_coarse); // refcounted by subop_coarse
      CeedBasisDestroy(&basis_c2f);
      CeedCompositeOperatorAddSub(op_coarse, subop_coarse);
      CeedOperatorDestroy(&subop_coarse); // refcounted by composite operator
   }
   return op_coarse;
}

AlgebraicCeedMultigrid::AlgebraicCeedMultigrid(
   AlgebraicFESpaceHierarchy &hierarchy,
   BilinearForm &form,
   Array<int> ess_tdofs
) : Multigrid(hierarchy)
{
   int nlevels = fespaces.GetNumLevels();
   ceed_operators.SetSize(nlevels);
   essentialTrueDofs.SetSize(nlevels);

   // Construct finest level
   ceed_operators[nlevels-1] = CreateCeedCompositeOperatorFromBilinearForm(form);
   essentialTrueDofs[nlevels-1] = new Array<int>;
   *essentialTrueDofs[nlevels-1] = ess_tdofs;

   // Construct hierarchy by coarsening
   for (int ilevel=nlevels-2; ilevel>=0; --ilevel)
   {
      AlgebraicCoarseFESpace &space = hierarchy.GetAlgebraicCoarseFESpace(ilevel);
      ceed_operators[ilevel] = CoarsenCeedCompositeOperator(
         ceed_operators[ilevel+1], space.GetCeedElemRestriction(),
         space.GetCeedCoarseToFine(), space.GetOrderReduction());
      Operator *P = hierarchy.GetProlongationAtLevel(ilevel);
      essentialTrueDofs[ilevel] = new Array<int>;
      CoarsenEssentialDofs(*P, *essentialTrueDofs[ilevel+1], *essentialTrueDofs[ilevel]);
   }

   // Add the operators and smoothers
   for (int ilevel=0; ilevel<nlevels; ++ilevel)
   {
      FiniteElementSpace &space = hierarchy.GetFESpaceAtLevel(ilevel);
      const Operator *P = space.GetProlongationMatrix();
      Operator *op = new MFEMCeedOperator(ceed_operators[ilevel],
                                          *essentialTrueDofs[ilevel], P);
      Solver *solv;
      if (ilevel == 0)
      {
         solv = BuildSmootherFromCeed(op, ceed_operators[ilevel],
                                      *essentialTrueDofs[ilevel], P, true);
      }
      else {
         solv = BuildSmootherFromCeed(op, ceed_operators[ilevel],
                                      *essentialTrueDofs[ilevel], P, true);
      }
      AddLevel(op, solv, true, true);
   }
}

AlgebraicCeedMultigrid::~AlgebraicCeedMultigrid()
{
   for (int i=0; i<ceed_operators.Size(); ++i)
   {
      CeedOperatorDestroy(&ceed_operators[i]);
   }
}

AlgebraicFESpaceHierarchy::AlgebraicFESpaceHierarchy(FiniteElementSpace &fes)
{
   int order = fes.GetOrder(0);

   int nlevels = 0;
   int current_order = order;
   while (current_order > 0)
   {
      nlevels++;
      current_order = current_order / 2;
   }

   meshes.SetSize(nlevels);
   ownedMeshes.SetSize(nlevels);
   meshes = fes.GetMesh();
   ownedMeshes = false;

   fespaces.SetSize(nlevels);
   ownedFES.SetSize(nlevels);
   // Own all FESpaces except for the finest, own all prolongations
   ownedFES = true;
   fespaces[nlevels-1] = &fes;
   ownedFES[nlevels-1] = false;

   ceed_interpolations.SetSize(nlevels-1);
   R_tr.SetSize(nlevels-1);
   prolongations.SetSize(nlevels-1);
   ownedProlongations.SetSize(nlevels-1);

   current_order = order;

   Ceed ceed = internal::ceed;
   InitCeedTensorRestriction(fes, ceed, &fine_er);
   CeedElemRestriction er = fine_er;

   int dim = fes.GetMesh()->Dimension();

#ifdef MFEM_USE_MPI
   GroupCommunicator *gc = NULL;
   ParFiniteElementSpace *pfes = dynamic_cast<ParFiniteElementSpace*>(&fes);
   if (pfes)
   {
      gc = &pfes->GroupComm();
   }
#endif

   for (int ilevel=nlevels-2; ilevel>=0; --ilevel)
   {
      const int order_reduction = current_order - (current_order/2);
      AlgebraicCoarseFESpace *space;

#ifdef MFEM_USE_MPI
      if (pfes)
      {
         ParAlgebraicCoarseFESpace *parspace = new ParAlgebraicCoarseFESpace(
            *fespaces[ilevel+1],
            er,
            current_order,
            dim,
            order_reduction,
            gc
         );
         gc = parspace->GetGroupCommunicator();
         space = parspace;
      }
      else
#endif
      {
         space = new AlgebraicCoarseFESpace(
            *fespaces[ilevel+1],
            er,
            current_order,
            dim,
            order_reduction
         );
      }
      current_order = current_order/2;
      fespaces[ilevel] = space;
      ceed_interpolations[ilevel] = new MFEMCeedInterpolation(
         ceed,
         space->GetCeedCoarseToFine(),
         space->GetCeedElemRestriction(),
         er
      );
      const SparseMatrix *R = fespaces[ilevel+1]->GetRestrictionMatrix();
      if (R)
      {
         R->BuildTranspose();
         R_tr[ilevel] = new TransposeOperator(*R);
      }
      else
      {
         R_tr[ilevel] = NULL;
      }
      prolongations[ilevel] = ceed_interpolations[ilevel]->SetupRAP(
         space->GetProlongationMatrix(), R_tr[ilevel]);
      ownedProlongations[ilevel]
         = prolongations[ilevel] != ceed_interpolations[ilevel];

      er = space->GetCeedElemRestriction();
   }
}

AlgebraicCoarseFESpace::AlgebraicCoarseFESpace(
   FiniteElementSpace &fine_fes,
   CeedElemRestriction fine_er,
   int order,
   int dim,
   int order_reduction_
) : order_reduction(order_reduction_)
{
   order_reduction = order_reduction_;

   CeedATPMGElemRestriction(
      order,
      order_reduction,
      fine_er,
      &ceed_elem_restriction,
      dof_map
   );
   CeedBasisATPMGCToF(
      internal::ceed,
      order+1,
      dim,
      order_reduction,
      &coarse_to_fine
   );
   vdim = fine_fes.GetVDim();
   CeedElemRestrictionGetLVectorSize(ceed_elem_restriction, &ndofs);
}

AlgebraicCoarseFESpace::~AlgebraicCoarseFESpace()
{
   free(dof_map);
}

ParAlgebraicCoarseFESpace::ParAlgebraicCoarseFESpace(
   FiniteElementSpace &fine_fes,
   CeedElemRestriction fine_er,
   int order,
   int dim,
   int order_reduction_,
   GroupCommunicator *gc_fine)
 : AlgebraicCoarseFESpace(fine_fes, fine_er, order, dim, order_reduction_)
{
   int lsize;
   CeedElemRestrictionGetLVectorSize(ceed_elem_restriction, &lsize);
   const Table &group_ldof_fine = gc_fine->GroupLDofTable();

   Array<int> ldof_group(lsize);
   ldof_group = 0;

   GroupTopology &group_topo = gc_fine->GetGroupTopology();
   gc = new GroupCommunicator(group_topo);
   Table &group_ldof = gc->GroupLDofTable();
   group_ldof.MakeI(group_ldof_fine.Size());
   for (int g=1; g<group_ldof_fine.Size(); ++g)
   {
      int nldof_fine_g = group_ldof_fine.RowSize(g);
      const int *ldof_fine_g = group_ldof_fine.GetRow(g);
      for (int i=0; i<nldof_fine_g; ++i)
      {
         int icoarse = dof_map[ldof_fine_g[i]];
         if (icoarse >= 0)
         {
            group_ldof.AddAColumnInRow(g);
            ldof_group[icoarse] = g;
         }
      }
   }
   group_ldof.MakeJ();
   for (int g=1; g<group_ldof_fine.Size(); ++g)
   {
      int nldof_fine_g = group_ldof_fine.RowSize(g);
      const int *ldof_fine_g = group_ldof_fine.GetRow(g);
      for (int i=0; i<nldof_fine_g; ++i)
      {
         int icoarse = dof_map[ldof_fine_g[i]];
         if (icoarse >= 0)
         {
            group_ldof.AddConnection(g, icoarse);
         }
      }
   }
   group_ldof.ShiftUpI();
   gc->Finalize();
   Array<int> ldof_ltdof(lsize);
   ldof_ltdof = -2;
   int ltsize = 0;
   for (int i=0; i<lsize; ++i)
   {
      int g = ldof_group[i];
      if (group_topo.IAmMaster(g))
      {
         ldof_ltdof[i] = ltsize;
         ++ltsize;
      }
   }
   gc->SetLTDofTable(ldof_ltdof);
   gc->Bcast(ldof_ltdof);

   R_mat = new SparseMatrix(ltsize, lsize);
   for (int j=0; j<lsize; ++j)
   {
      if (group_topo.IAmMaster(ldof_group[j]))
      {
         int i = ldof_ltdof[j];
         R_mat->Set(i,j,1.0);
      }
   }
   R_mat->Finalize();

   P = new ConformingProlongationOperator(lsize, *gc);
}

} // namespace mfem
#endif // MFEM_USE_CEED
