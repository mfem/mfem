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
#include "../fem/libceed/ceed-assemble.hpp"
#include "../fem/pfespace.hpp"

namespace mfem
{

Solver *BuildSmootherFromCeed(MFEMCeedOperator &op, bool chebyshev)
{
   CeedOperator ceed_op = op.GetCeedOperator();
   const Array<int> &ess_tdofs = op.GetEssentialTrueDofs();
   const Operator *P = op.GetProlongation();
   // Assemble the a local diagonal, in the sense of L-vector
   CeedVector diagceed;
   CeedInt length;
   CeedOperatorGetSize(ceed_op, &length);
   CeedVectorCreate(internal::ceed, length, &diagceed);
   CeedOperatorLinearAssembleDiagonal(ceed_op, diagceed, CEED_REQUEST_IMMEDIATE);
   const CeedScalar *diagvals;
   CeedMemType mem;
   CeedGetPreferredMemType(internal::ceed, &mem);
   if (!Device::Allows(Backend::CUDA) || mem != CEED_MEM_DEVICE)
   {
      mem = CEED_MEM_HOST;
   }
   CeedVectorGetArrayRead(diagceed, mem, &diagvals);

   Vector t_diag;
   if (P)
   {
      Vector local_diag(const_cast<CeedScalar*>(diagvals), length);
      t_diag.SetSize(P->Width());
      P->MultTranspose(local_diag, t_diag);
   }
   else
   {
      t_diag.SetDataAndSize(const_cast<CeedScalar*>(diagvals), length);
   }
   Solver *out = NULL;
   if (chebyshev)
   {
      const int cheb_order = 3;
      out = new OperatorChebyshevSmoother(&op, t_diag, ess_tdofs, cheb_order);
   }
   else
   {
      const double jacobi_scale = 0.65;
      out = new OperatorJacobiSmoother(t_diag, ess_tdofs, jacobi_scale);
   }
   CeedVectorRestoreArrayRead(diagceed, &diagvals);
   CeedVectorDestroy(&diagceed);
   return out;
}

#ifdef MFEM_USE_MPI

class CeedAMG : public Solver
{
public:
   CeedAMG(MFEMCeedOperator &oper, HypreParMatrix *P)
   {
      const Array<int> ess_tdofs = oper.GetEssentialTrueDofs();
      height = width = oper.Height();

      CeedOperatorFullAssemble(oper.GetCeedOperator(), &mat_local);
      HypreParMatrix *hypre_local = new HypreParMatrix(
         P->GetComm(), P->GetGlobalNumRows(), P->RowPart(), mat_local);
      if (P)
      {
         op_assembled = RAP(hypre_local, P);
         delete hypre_local;
      }
      else
      {
         op_assembled = hypre_local;
      }
      HypreParMatrix *mat_e = op_assembled->EliminateRowsCols(ess_tdofs);
      delete mat_e;
      amg = new HypreBoomerAMG(*op_assembled);
      amg->SetPrintLevel(0);
   }
   void SetOperator(const Operator &op) { }
   void Mult(const Vector &x, Vector &y) const { amg->Mult(x, y); }
   ~CeedAMG()
   {
      delete op_assembled;
      delete amg;
   }
private:
   SparseMatrix *mat_local;
   HypreParMatrix *op_assembled;
   HypreBoomerAMG *amg;
};

#endif

void CoarsenEssentialDofs(const Operator &interp,
                          const Array<int> &ho_ess_tdofs,
                          Array<int> &alg_lo_ess_tdofs)
{
   Vector ho_boundary_ones(interp.Height());
   ho_boundary_ones = 0.0;
   for (int k : ho_ess_tdofs)
   {
      ho_boundary_ones(k) = 1.0;
   }
   Vector lo_boundary_ones(interp.Width());
   interp.MultTranspose(ho_boundary_ones, lo_boundary_ones);
   auto lobo = lo_boundary_ones.HostRead();
   for (int i = 0; i < lo_boundary_ones.Size(); ++i)
   {
      if (lobo[i] > 0.9)
      {
         alg_lo_ess_tdofs.Append(i);
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
   AlgebraicSpaceHierarchy &hierarchy,
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

   // Construct operators at all levels of hierarchy by coarsening
   for (int ilevel=nlevels-2; ilevel>=0; --ilevel)
   {
      AlgebraicCoarseSpace &space = hierarchy.GetAlgebraicCoarseSpace(ilevel);
      ceed_operators[ilevel] = CoarsenCeedCompositeOperator(
         ceed_operators[ilevel+1], space.GetCeedElemRestriction(),
         space.GetCeedCoarseToFine(), space.GetOrderReduction());
      Operator *P = hierarchy.GetProlongationAtLevel(ilevel);
      essentialTrueDofs[ilevel] = new Array<int>;
      CoarsenEssentialDofs(*P, *essentialTrueDofs[ilevel+1], *essentialTrueDofs[ilevel]);
   }

   // Add the operators and smoothers to the hierarchy, from coarse to fine
   for (int ilevel=0; ilevel<nlevels; ++ilevel)
   {
      FiniteElementSpace &space = hierarchy.GetFESpaceAtLevel(ilevel);
      const Operator *P = space.GetProlongationMatrix();
      MFEMCeedOperator *op = new MFEMCeedOperator(
         ceed_operators[ilevel], *essentialTrueDofs[ilevel], P);
      Solver *smoother;
#ifdef MFEM_USE_MPI
      if (ilevel == 0)
      {
         HypreParMatrix *P_mat = NULL;
         if (nlevels == 1)
         {
            // Only one level -- no coarsening, finest level
            ParFiniteElementSpace *pfes
               = dynamic_cast<ParFiniteElementSpace*>(&space);
            if (pfes) { P_mat = pfes->Dof_TrueDof_Matrix(); }
         }
         else
         {
            ParAlgebraicCoarseSpace *pspace
               = dynamic_cast<ParAlgebraicCoarseSpace*>(&space);
            if (pspace) { P_mat = pspace->GetProlongationHypreParMatrix(); }
         }
         smoother = new CeedAMG(*op, P_mat);
      }
      else
#endif
      {
         smoother = BuildSmootherFromCeed(*op, true);
      }
      AddLevel(op, smoother, true, true);
   }
}

AlgebraicCeedMultigrid::~AlgebraicCeedMultigrid()
{
   for (int i=0; i<ceed_operators.Size(); ++i)
   {
      CeedOperatorDestroy(&ceed_operators[i]);
   }
}

AlgebraicSpaceHierarchy::AlgebraicSpaceHierarchy(FiniteElementSpace &fes)
{
   int order = fes.GetOrder(0);
   int nlevels = 0;
   int current_order = order;
   while (current_order > 0)
   {
      nlevels++;
      current_order = current_order/2;
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
      AlgebraicCoarseSpace *space;

#ifdef MFEM_USE_MPI
      if (pfes)
      {
         ParAlgebraicCoarseSpace *parspace = new ParAlgebraicCoarseSpace(
            *fespaces[ilevel+1], er, current_order, dim, order_reduction, gc);
         gc = parspace->GetGroupCommunicator();
         space = parspace;
      }
      else
#endif
      {
         space = new AlgebraicCoarseSpace(
            *fespaces[ilevel+1], er, current_order, dim, order_reduction);
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

AlgebraicCoarseSpace::AlgebraicCoarseSpace(
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
   CeedElemRestrictionGetLVectorSize(ceed_elem_restriction, &ndofs);
   mesh = fine_fes.GetMesh();
}

AlgebraicCoarseSpace::~AlgebraicCoarseSpace()
{
   free(dof_map);
}

#ifdef MFEM_USE_MPI

ParAlgebraicCoarseSpace::ParAlgebraicCoarseSpace(
   FiniteElementSpace &fine_fes,
   CeedElemRestriction fine_er,
   int order,
   int dim,
   int order_reduction_,
   GroupCommunicator *gc_fine)
 : AlgebraicCoarseSpace(fine_fes, fine_er, order, dim, order_reduction_)
{
   int lsize;
   CeedElemRestrictionGetLVectorSize(ceed_elem_restriction, &lsize);
   const Table &group_ldof_fine = gc_fine->GroupLDofTable();

   ldof_group.SetSize(lsize);
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
   ldof_ltdof.SetSize(lsize);
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
   P_mat = NULL;
}

HypreParMatrix *ParAlgebraicCoarseSpace::GetProlongationHypreParMatrix()
{
   if (P_mat) { return P_mat; }

   ParMesh *pmesh = dynamic_cast<ParMesh*>(mesh);
   MFEM_VERIFY(pmesh != NULL, "");
   Array<HYPRE_Int> dof_offsets, tdof_offsets, tdof_nb_offsets;
   Array<HYPRE_Int> *offsets[2] = {&dof_offsets, &tdof_offsets};
   int lsize = P->Height();
   int ltsize = P->Width();
   HYPRE_Int loc_sizes[2] = {lsize, ltsize};
   pmesh->GenerateOffsets(2, loc_sizes, offsets);

   MPI_Comm comm = pmesh->GetComm();

   const GroupTopology &group_topo = gc->GetGroupTopology();

   if (HYPRE_AssumedPartitionCheck())
   {
      // communicate the neighbor offsets in tdof_nb_offsets
      int nsize = group_topo.GetNumNeighbors()-1;
      MPI_Request *requests = new MPI_Request[2*nsize];
      MPI_Status  *statuses = new MPI_Status[2*nsize];
      tdof_nb_offsets.SetSize(nsize+1);
      tdof_nb_offsets[0] = tdof_offsets[0];

      // send and receive neighbors' local tdof offsets
      int request_counter = 0;
      for (int i = 1; i <= nsize; i++)
      {
         MPI_Irecv(&tdof_nb_offsets[i], 1, HYPRE_MPI_INT,
                  group_topo.GetNeighborRank(i), 5365, comm,
                  &requests[request_counter++]);
      }
      for (int i = 1; i <= nsize; i++)
      {
         MPI_Isend(&tdof_nb_offsets[0], 1, HYPRE_MPI_INT,
                  group_topo.GetNeighborRank(i), 5365, comm,
                  &requests[request_counter++]);
      }
      MPI_Waitall(request_counter, requests, statuses);

      delete [] statuses;
      delete [] requests;
   }

   HYPRE_Int *i_diag = Memory<HYPRE_Int>(lsize+1);
   HYPRE_Int *j_diag = Memory<HYPRE_Int>(ltsize);
   int diag_counter;

   HYPRE_Int *i_offd = Memory<HYPRE_Int>(lsize+1);
   HYPRE_Int *j_offd = Memory<HYPRE_Int>(lsize-ltsize);
   int offd_counter;

   HYPRE_Int *cmap   = Memory<HYPRE_Int>(lsize-ltsize);

   HYPRE_Int *col_starts = tdof_offsets;
   HYPRE_Int *row_starts = dof_offsets;

   Array<Pair<HYPRE_Int, int> > cmap_j_offd(lsize-ltsize);

   i_diag[0] = i_offd[0] = 0;
   diag_counter = offd_counter = 0;
   for (int i_ldof = 0; i_ldof < lsize; i_ldof++)
   {
      int g = ldof_group[i_ldof];
      int i_ltdof = ldof_ltdof[i_ldof];
      if (group_topo.IAmMaster(g))
      {
         j_diag[diag_counter++] = i_ltdof;
      }
      else
      {
         HYPRE_Int global_tdof_number;
         int g = ldof_group[i_ldof];
         if (HYPRE_AssumedPartitionCheck())
         {
            global_tdof_number
               = i_ltdof + tdof_nb_offsets[group_topo.GetGroupMaster(g)];
         }
         else
         {
            global_tdof_number
               = i_ltdof + tdof_offsets[group_topo.GetGroupMasterRank(g)];
         }

         cmap_j_offd[offd_counter].one = global_tdof_number;
         cmap_j_offd[offd_counter].two = offd_counter;
         offd_counter++;
      }
      i_diag[i_ldof+1] = diag_counter;
      i_offd[i_ldof+1] = offd_counter;
   }

   SortPairs<HYPRE_Int, int>(cmap_j_offd, offd_counter);

   for (int i = 0; i < offd_counter; i++)
   {
      cmap[i] = cmap_j_offd[i].one;
      j_offd[cmap_j_offd[i].two] = i;
   }

   P_mat = new HypreParMatrix(
      comm, pmesh->GetMyRank(), pmesh->GetNRanks(),
      row_starts, col_starts,
      i_diag, j_diag, i_offd, j_offd,
      cmap, offd_counter
   );

   P_mat->CopyRowStarts();
   P_mat->CopyColStarts();

   return P_mat;
}

ParAlgebraicCoarseSpace::~ParAlgebraicCoarseSpace()
{
   delete P;
   delete R_mat;
   delete P_mat;
   delete gc;
}

#endif

} // namespace mfem
#endif // MFEM_USE_CEED
