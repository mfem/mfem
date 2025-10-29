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

#include "../../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "pdarcyform.hpp"

namespace mfem
{
ParDarcyForm::ParDarcyForm(ParFiniteElementSpace *fes_u,
                           ParFiniteElementSpace *fes_p, bool bsymmetrize)
   : DarcyForm(fes_u, fes_p, bsymmetrize), pfes_u(*fes_u), pfes_p(*fes_p)
{
   UpdateTOffsets();
}

void ParDarcyForm::UpdateTOffsets()
{
   if (!toffsets.OwnsData()) { toffsets.DeleteAll(); }
   toffsets.SetSize(3);
   toffsets[0] = 0;
   toffsets[1] = pfes_u.GetTrueVSize();
   toffsets[2] = pfes_p.GetTrueVSize();
   toffsets.PartialSum();
}

BilinearForm *ParDarcyForm::GetFluxMassForm()
{
   if (!pM_u) { M_u.reset(pM_u = new ParBilinearForm(&pfes_u)); }
   return pM_u;
}

BilinearForm *ParDarcyForm::GetPotentialMassForm()
{
   if (!pM_p) { M_p.reset(pM_p = new ParBilinearForm(&pfes_p)); }
   return pM_p;
}

NonlinearForm *ParDarcyForm::GetFluxMassNonlinearForm()
{
   if (!pMnl_u) { Mnl_u.reset(pMnl_u = new ParNonlinearForm(&pfes_u)); }
   return pMnl_u;
}

NonlinearForm *ParDarcyForm::GetPotentialMassNonlinearForm()
{
   if (!pMnl_p) { Mnl_p.reset(pMnl_p = new ParNonlinearForm(&pfes_p)); }
   return pMnl_p;
}

MixedBilinearForm *ParDarcyForm::GetFluxDivForm()
{
   if (!pB) { B.reset(pB = new ParMixedBilinearForm(&pfes_u, &pfes_p)); }
   return pB;
}

ParBlockNonlinearForm *ParDarcyForm::GetParBlockNonlinearForm()
{
   if (!pMnl)
   {
      Array<ParFiniteElementSpace*> fes({&pfes_u, &pfes_p});
      Mnl.reset(pMnl = new ParBlockNonlinearForm(fes));
   }
   return pMnl;
}

void ParDarcyForm::Assemble(int skip_zeros)
{
   if (pB || pM_p || pMnl_p)
   {
      pfes_u.ExchangeFaceNbrData();
      pfes_p.ExchangeFaceNbrData();
   }

   if (pM_u)
   {
      if (hybridization)
      {
         DenseMatrix elmat;

         // Element-wise integration
         for (int i = 0; i < fes_u -> GetNE(); i++)
         {
            pM_u->ComputeElementMatrix(i, elmat);
#ifndef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
            pM_u->AssembleElementMatrix(i, elmat, skip_zeros);
#endif //!MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
            hybridization->AssembleFluxMassMatrix(i, elmat);
         }
      }
      else if (reduction)
      {
         DenseMatrix elmat;

         // Element-wise integration
         for (int i = 0; i < fes_u -> GetNE(); i++)
         {
            pM_u->ComputeElementMatrix(i, elmat);
#ifndef MFEM_DARCY_REDUCTION_ELIM_BCS
            pM_u->AssembleElementMatrix(i, elmat, skip_zeros);
#endif //!MFEM_DARCY_REDUCTION_ELIM_BCS
            reduction->AssembleFluxMassMatrix(i, elmat);
         }
      }
      else
      {
         pM_u->Assemble(skip_zeros);
      }
   }
   else if (Mnl_u)
   {
      Mnl_u->Setup();
   }

   if (pB)
   {
      if (hybridization)
      {
         DenseMatrix elmat;

         // Element-wise integration
         for (int i = 0; i < fes_u -> GetNE(); i++)
         {
            pB->ComputeElementMatrix(i, elmat);
#ifndef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
            pB->AssembleElementMatrix(i, elmat, skip_zeros);
#endif //!MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
            hybridization->AssembleDivMatrix(i, elmat);
         }
      }
      else if (reduction)
      {
         DenseMatrix elmat;

         // Element-wise integration
         for (int i = 0; i < fes_u -> GetNE(); i++)
         {
            pB->ComputeElementMatrix(i, elmat);
#ifndef MFEM_DARCY_REDUCTION_ELIM_BCS
            pB->AssembleElementMatrix(i, elmat, skip_zeros);
#endif //!MFEM_DARCY_REDUCTION_ELIM_BCS
            reduction->AssembleDivMatrix(i, elmat);
         }

         AssembleDivLDGFaces(skip_zeros);
         AssembleDivLDGSharedFaces(skip_zeros);
      }
      else
      {
         pB->Assemble(skip_zeros);
      }
   }

   if (pM_p)
   {
      if (hybridization)
      {
         DenseMatrix elmat;

         // Element-wise integration
         for (int i = 0; i < fes_p -> GetNE(); i++)
         {
            pM_p->ComputeElementMatrix(i, elmat);
#ifndef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
            pM_p->AssembleElementMatrix(i, elmat, skip_zeros);
#endif //!MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
            hybridization->AssemblePotMassMatrix(i, elmat);
         }

         AssemblePotHDGFaces(skip_zeros);
         AssemblePotHDGSharedFaces(skip_zeros);
      }
      else if (reduction)
      {
         DenseMatrix elmat;

         // Element-wise integration
         for (int i = 0; i < fes_p -> GetNE(); i++)
         {
            pM_p->ComputeElementMatrix(i, elmat);
#ifndef MFEM_DARCY_REDUCTION_ELIM_BCS
            pM_p->AssembleElementMatrix(i, elmat, skip_zeros);
#endif //!MFEM_DARCY_REDUCTION_ELIM_BCS
            reduction->AssemblePotMassMatrix(i, elmat);
         }

         AssemblePotLDGFaces(skip_zeros);
         AssemblePotLDGSharedFaces(skip_zeros);
      }
      else
      {
         pM_p->Assemble(skip_zeros);
      }
   }
   else if (Mnl_p)
   {
      Mnl_p->Setup();
   }
}

void ParDarcyForm::Finalize(int skip_zeros)
{
   AllocBlockOp();

   if (block_op)
   {
      if (M_u)
      {
         M_u->Finalize(skip_zeros);
      }

      if (M_p)
      {
         M_p->Finalize(skip_zeros);
      }

      if (B)
      {
         B->Finalize(skip_zeros);
      }
   }

   if (hybridization)
   {
      hybridization->Finalize();
   }
   else if (reduction)
   {
      reduction->Finalize();
   }
}

void ParDarcyForm::ParallelAssembleInternal()
{
   if (pM_u) { pM_u->ParallelAssembleInternalMatrix(); }
   if (pM_p) { pM_p->ParallelAssembleInternalMatrix(); }
   if (pB) { pB->ParallelAssembleInternalMatrix(); }
}

void ParDarcyForm::FormLinearSystem(
   const Array<int> &ess_flux_tdof_list, BlockVector &x, BlockVector &b,
   OperatorHandle &A, Vector &X_, Vector &B_, int copy_interior)
{
   MFEM_VERIFY(assembly == AssemblyLevel::LEGACY,
               "Only legacy assembly is supported");

   // Finish the matrix assembly and perform BC elimination, storing the
   // eliminated part of the matrix.
   FormSystemMatrix(ess_flux_tdof_list, A);

   // Transform the system and perform the elimination in B, based on the
   // essential BC values from x. Restrict the BC part of x in X, and set the
   // non-BC part to zero. Since there is no good initial guess for the Lagrange
   // multipliers, set X = 0.0 for hybridization.
   if (reduction || hybridization)
   {
      // Reduction to the single equation system
      BlockVector true_X(toffsets), true_B(toffsets);

      const Operator &P_u = *pfes_u.GetProlongationMatrix();
      const Operator &R_u = *pfes_u.GetRestrictionOperator();
      P_u.MultTranspose(b.GetBlock(0), true_B.GetBlock(0));
      R_u.Mult(x.GetBlock(0), true_X.GetBlock(0));

      const Operator &P_p = *pfes_p.GetProlongationMatrix();
      const Operator &R_p = *pfes_p.GetRestrictionOperator();
      P_p.MultTranspose(b.GetBlock(1), true_B.GetBlock(1));
      R_p.Mult(x.GetBlock(1), true_X.GetBlock(1));

      ParallelEliminateTDofsInRHS(ess_flux_tdof_list, true_X, true_B);

      R_u.MultTranspose(true_B.GetBlock(0), b.GetBlock(0));
      R_p.MultTranspose(true_B.GetBlock(1), b.GetBlock(1));

      if (hybridization)
      {
         hybridization->ReduceRHS(true_B, B_);
      }
      else
      {
         reduction->ReduceRHS(true_B, B_);
      }

      if (X_.Size() != B_.Size())
      {
         X_.SetSize(B_.Size());
         X_ = 0.0;
      }
      else if (!copy_interior)
      {
         X_ = 0.0;
      }
   }
   else
   {
      X_.SetSize(toffsets.Last());
      B_.SetSize(toffsets.Last());
      BlockVector X(X_, toffsets), B(B_, toffsets);

      // Variational restriction with P
      const Operator &P_u = *pfes_u.GetProlongationMatrix();
      const Operator &R_u = *pfes_u.GetRestrictionOperator();
      P_u.MultTranspose(b.GetBlock(0), B.GetBlock(0));
      R_u.Mult(x.GetBlock(0), X.GetBlock(0));

      const Operator &P_p = *pfes_p.GetProlongationMatrix();
      const Operator &R_p = *pfes_p.GetRestrictionOperator();
      P_p.MultTranspose(b.GetBlock(1), B.GetBlock(1));
      R_p.Mult(x.GetBlock(1), X.GetBlock(1));

      ParallelEliminateTDofsInRHS(ess_flux_tdof_list, X, B);
      if (!copy_interior)
      {
         X.GetBlock(0).SetSubVectorComplement(ess_flux_tdof_list, 0.0);
         X.GetBlock(1) = 0.;
      }
   }
}

void ParDarcyForm::FormSystemMatrix(const Array<int> &ess_flux_tdof_list,
                                    OperatorHandle &A)
{
   AllocBlockOp();

   if (block_op)
   {
      Array<int> ess_pot_tdof_list;//empty for discontinuous potentials

      if (pM_u)
      {
         pM_u->FormSystemMatrix(ess_flux_tdof_list, opM_u);
         block_op->SetDiagonalBlock(0, opM_u.Ptr());
      }
      else if (Mnl_u)
      {
         pMnl_u->SetEssentialTrueDofs(ess_flux_tdof_list);
         block_op->SetDiagonalBlock(0, Mnl_u.get());
      }
      else if (Mnl)
      {
         Array<Array<int>*> ess_tdof_lists
         {
            const_cast<Array<int>*>(&ess_flux_tdof_list),
            const_cast<Array<int>*>(&ess_pot_tdof_list)
         };
         Array<Vector*> rhss(2); rhss = NULL;
         pMnl->SetEssentialTrueDofs(ess_tdof_lists, rhss);
      }

      if (pM_p)
      {
         pM_p->FormSystemMatrix(ess_pot_tdof_list, opM_p);
         block_op->SetDiagonalBlock(1, opM_p.Ptr(), (bsym)?(-1.):(+1.));
      }
      else if (Mnl_p)
      {
         block_op->SetDiagonalBlock(1, Mnl_p.get(), (bsym)?(-1.):(+1.));
      }

      if (pB)
      {
         pB->FormRectangularSystemMatrix(ess_flux_tdof_list, ess_pot_tdof_list, opB);

         ConstructBT(opB);

         block_op->SetBlock(0, 1, opBt.Ptr(), (bsym)?(-1.):(+1.));
         block_op->SetBlock(1, 0, opB.Ptr(), (bsym)?(-1.):(+1.));
      }

   }

   if (hybridization)
   {
      hybridization->Finalize();
      hybridization->GetParallelOperator(A);
   }
   else if (reduction)
   {
      reduction->Finalize();
      if (!Mnl_u && !Mnl_p && !Mnl)
      {
         reduction->GetParallelMatrix(A);
      }
      else
      {
         MFEM_ABORT("Not implemented");
         //A.Reset(reduction, false);
      }
   }
   else
   {
      A.Reset(new ParOperator(*this));
   }
}

void ParDarcyForm::RecoverFEMSolution(const Vector &X_, const BlockVector &b,
                                      BlockVector &x)
{
   if (reduction || hybridization)
   {
      // Primal unknowns recovery
      BlockVector true_X(toffsets), true_B(toffsets);

      const Operator &P_u = *pfes_u.GetProlongationMatrix();
      const Operator &R_u = *pfes_u.GetRestrictionOperator();
      P_u.MultTranspose(b.GetBlock(0), true_B.GetBlock(0));
      R_u.Mult(x.GetBlock(0), true_X.GetBlock(0));

      const Operator &P_p = *pfes_p.GetProlongationMatrix();
      const Operator &R_p = *pfes_p.GetRestrictionOperator();
      P_p.MultTranspose(b.GetBlock(1), true_B.GetBlock(1));
      R_p.Mult(x.GetBlock(1), true_X.GetBlock(1));

      if (hybridization)
      {
         hybridization->ComputeSolution(true_B, X_, true_X);
      }
      else if (reduction)
      {
         reduction->ComputeSolution(true_B, X_, true_X);
      }

      P_u.Mult(true_X.GetBlock(0), x.GetBlock(0));
      P_p.Mult(true_X.GetBlock(1), x.GetBlock(1));
   }
   else
   {
      BlockVector X(const_cast<Vector&>(X_), toffsets);
      if (pM_u)
      {
         pM_u->RecoverFEMSolution(X.GetBlock(0), b.GetBlock(0), x.GetBlock(0));
      }
      else
      {
         // Apply conforming prolongation
         const Operator &P_u = *pfes_u.GetProlongationMatrix();
         P_u.Mult(X.GetBlock(0), x.GetBlock(0));
      }

      if (pM_p)
      {
         pM_p->RecoverFEMSolution(X.GetBlock(1), b.GetBlock(1), x.GetBlock(1));
      }
      else
      {
         // Apply conforming prolongation
         const Operator &P_p = *pfes_p.GetProlongationMatrix();
         P_p.Mult(X.GetBlock(1), x.GetBlock(1));
      }
   }
}

void ParDarcyForm::ParallelEliminateTDofsInRHS(const Array<int> &tdofs_flux,
                                               const BlockVector &x, BlockVector &b)
{
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   if (hybridization)
   {
      hybridization->EliminateTrueDofsInRHS(tdofs_flux, x, b);
      return;
   }
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
   if (reduction)
   {
      reduction->EliminateTrueDofsInRHS(tdofs_flux, x, b);
      return;
   }
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
   if (pB)
   {
      if (bsym)
      {
         //In the case of the symmetrized system, the sign is oppposite!
         Vector b_(fes_p->GetVSize());
         b_ = 0.;
         pB->ParallelEliminateTrialTDofsInRHS(tdofs_flux, x.GetBlock(0), b_);
         b.GetBlock(1) -= b_;
      }
      else
      {
         pB->ParallelEliminateTrialTDofsInRHS(tdofs_flux, x.GetBlock(0), b.GetBlock(1));
      }
   }
   if (pM_u)
   {
      pM_u->ParallelEliminateTDofsInRHS(tdofs_flux, x.GetBlock(0), b.GetBlock(0));
   }
   else if (pMnl_u || pMnl)
   {
      b.GetBlock(0).SetSubVector(tdofs_flux, 0.);
   }
}

void ParDarcyForm::Mult(const Vector &x, Vector &y) const
{
   const BlockVector xb(const_cast<Vector&>(x), offsets);
   BlockVector yb(y, offsets);

   if (pM_u) { pM_u->Mult(xb.GetBlock(0), yb.GetBlock(0)); }
   else { yb.GetBlock(0) = 0.; }

   if (pM_p)
   {
      pM_p->Mult(xb.GetBlock(1), yb.GetBlock(1));
      if (bsym) { yb.GetBlock(1).Neg(); }
   }
   else { yb.GetBlock(1) = 0.; }

   if (pB)
   {
      pB->AddMult(xb.GetBlock(1), yb.GetBlock(0), (bsym)?(-1.):(+1.));
      pB->AddMultTranspose(xb.GetBlock(0), yb.GetBlock(1), (bsym)?(-1.):(+1.));
   }
}

void ParDarcyForm::ParOperator::Mult(const Vector &x, Vector &y) const
{
   darcy.block_op->Mult(x, y);
   if (darcy.pMnl)
   {
      if (darcy.bsym)
      {
         BlockVector ynl(darcy.toffsets);
         darcy.pMnl->Mult(x, ynl);
         ynl.GetBlock(1).Neg();
         y += ynl;
      }
      else
      {
         darcy.pMnl->AddMult(x, y);
      }
   }
}

Operator &ParDarcyForm::ParOperator::GetGradient(const Vector &x) const
{
   const BlockVector bx(const_cast<Vector&>(x), darcy.toffsets);

   if (!darcy.Mnl && !darcy.Mnl_u && !darcy.Mnl_p) { return *darcy.block_op; }

   if (darcy.Mnl_u || darcy.Mnl_p)
   {
      if (!block_grad)
      {
         block_grad.reset(new BlockOperator(darcy.toffsets));
      }

      if (darcy.opM_u.Ptr())
      {
         block_grad->SetDiagonalBlock(0, darcy.opM_u.Ptr());
      }
      else if (darcy.pM_u)
      {
         block_grad->SetDiagonalBlock(0, darcy.pM_u->ParallelAssembleInternalMatrix());
      }
      else if (darcy.pMnl_u)
      {
         block_grad->SetDiagonalBlock(0, &darcy.pMnl_u->GetGradient(bx.GetBlock(0)));
      }

      if (darcy.opM_p.Ptr())
      {
         block_grad->SetDiagonalBlock(1, darcy.opM_p.Ptr(), (darcy.bsym)?(-1.):(+1.));
      }
      else if (darcy.pM_p)
      {
         block_grad->SetDiagonalBlock(1, darcy.pM_p->ParallelAssembleInternalMatrix(),
                                      (darcy.bsym)?(-1.):(+1.));
      }
      else if (darcy.pMnl_p)
      {
         block_grad->SetDiagonalBlock(1, &darcy.pMnl_p->GetGradient(bx.GetBlock(1)),
                                      (darcy.bsym)?(-1.):(+1.));
      }

      if (darcy.pB)
      {
         if (!darcy.opB.Ptr() || !darcy.opBt.Ptr())
         {
            HypreParMatrix *hB = darcy.pB->ParallelAssembleInternalMatrix();
            darcy.opB.Reset(hB, false);
            darcy.ConstructBT(darcy.opB);
         }
         block_grad->SetBlock(0, 1, darcy.opBt.Ptr(), (darcy.bsym)?(-1.):(+1.));
         block_grad->SetBlock(1, 0, darcy.opB.Ptr(), (darcy.bsym)?(-1.):(+1.));
      }

      if (!darcy.Mnl) { return *block_grad; }
   }

   opG.Reset(new ParGradient(*this, x));
   return *opG.Ptr();
}

void ParDarcyForm::ParGradient::Mult(const Vector &x, Vector &y) const
{
   if (block_grad)
   {
      block_grad->Mult(x, y);
      return;
   }

   if (p.block_grad)
   {
      p.block_grad->Mult(x, y);
   }
   else
   {
      p.darcy.block_op->Mult(x, y);
   }

   if (p.darcy.bsym)
   {
      BlockVector ynl(p.darcy.toffsets);
      G.Mult(x, ynl);
      ynl.GetBlock(1).Neg();
      y += ynl;
   }
   else
   {
      G.AddMult(x, y);
   }
}

const BlockOperator& ParDarcyForm::ParGradient::BlockMatrices() const
{
   if (block_grad) { return *block_grad.get(); }

   block_grad.reset(new BlockOperator(p.darcy.toffsets));

   const BlockOperator *bop = (p.block_grad)?(p.block_grad.get()):
                              (p.darcy.block_op.get());
   const BlockOperator *bgrad = static_cast<const BlockOperator*>(&G);

   for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
      {
         //off-diagonals of bgrad are expected to be zero
         if (i == j && !bop->IsZeroBlock(i,j) && !bgrad->IsZeroBlock(i,j))
         {
            const HypreParMatrix *hop = dynamic_cast<const HypreParMatrix*>(
                                           &(bop->GetBlock(i,j)));
            const HypreParMatrix *hgrad = dynamic_cast<const HypreParMatrix*>(
                                             &(bgrad->GetBlock(i,j)));

            MFEM_ASSERT(hop && hgrad, "Not a HypreParMatrix!");

            hmats[i][j].reset(ParAdd(hop, hgrad));
            block_grad->SetBlock(i, j, hmats[i][j].get(), bop->GetBlockCoef(i,j));
         }
         else
         {
            const Operator *op;
            real_t c;
            if (!bop->IsZeroBlock(i,j))
            {
               op = &(bop->GetBlock(i,j));
               c = bop->GetBlockCoef(i,j);
            }
            else
            {
               op = &(bgrad->GetBlock(i,j));
               c = (i != 0 && p.darcy.bsym)?(-1):(+1.);
            }
            // transpose operator is passed as is
            MFEM_ASSERT((i == 0 && j == 1) ||
                        dynamic_cast<const HypreParMatrix*>(op), "Not a HypreParMatrix!");
            block_grad->SetBlock(i, j, const_cast<Operator*>(op), c);
         }
      }

   return *block_grad;
}

void ParDarcyForm::Update()
{
   DarcyForm::Update();
   UpdateTOffsets();
}

ParDarcyForm::~ParDarcyForm()
{
}

void ParDarcyForm::AllocBlockOp()
{
   bool noblock = false;
#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
   noblock = noblock || reduction || hybridization;
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS

   if (!noblock)
   {
      block_op.reset(new BlockOperator(toffsets));
   }
}

void ParDarcyForm::AssembleDivLDGSharedFaces(int skip_zeros)
{
   ParMesh *pmesh = pfes_p.GetParMesh();
   const int NE = pmesh->GetNE();
   FaceElementTransformations *tr;
#ifndef MFEM_DARCY_REDUCTION_ELIM_BCS
   MFEM_ABORT("Not supported");
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS

   auto &interior_face_integs = *B->GetFBFI();

   if (interior_face_integs.Size())
   {
      DenseMatrix elmat, elem_mat;

      const int nsfaces = pmesh->GetNSharedFaces();
      for (int i = 0; i < nsfaces; i++)
      {
         tr = pmesh->GetSharedFaceTransformations(i);
         if (tr == NULL) { continue; }

         const FiniteElement *trial_fe1 = pfes_u.GetFE(tr->Elem1No);
         const FiniteElement *trial_fe2 = pfes_u.GetFaceNbrFE(tr->Elem2No - NE);
         const FiniteElement *test_fe1 = pfes_p.GetFE(tr->Elem1No);
         const FiniteElement *test_fe2 = pfes_p.GetFaceNbrFE(tr->Elem2No - NE);

         interior_face_integs[0]->AssembleFaceMatrix(*trial_fe1, *test_fe1, *trial_fe2,
                                                     *test_fe2, *tr, elmat);
         for (int i = 1; i < interior_face_integs.Size(); i++)
         {
            interior_face_integs[i]->AssembleFaceMatrix(*trial_fe1, *test_fe1, *trial_fe2,
                                                        *test_fe2, *tr, elmat);
            elmat += elem_mat;
         }

         reduction->AssembleDivSharedFaceMatrix(i, elmat);
      }
   }
}

void ParDarcyForm::AssemblePotLDGSharedFaces(int skip_zeros)
{
   ParMesh *pmesh = pfes_p.GetParMesh();
   const int NE = pmesh->GetNE();
   FaceElementTransformations *tr;
#ifndef MFEM_DARCY_REDUCTION_ELIM_BCS
   MFEM_ABORT("Not supported");
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS

   auto &interior_face_integs = *M_p->GetFBFI();

   if (interior_face_integs.Size())
   {
      DenseMatrix elmat, elem_mat;

      const int nsfaces = pmesh->GetNSharedFaces();
      for (int i = 0; i < nsfaces; i++)
      {
         tr = pmesh->GetSharedFaceTransformations(i);

         const FiniteElement *fe1 = pfes_p.GetFE(tr->Elem1No);
         const FiniteElement *fe2 = pfes_p.GetFaceNbrFE(tr->Elem2No - NE);

         interior_face_integs[0]->AssembleFaceMatrix(*fe1, *fe2, *tr, elmat);
         for (int i = 1; i < interior_face_integs.Size(); i++)
         {
            interior_face_integs[i]->AssembleFaceMatrix(*fe1, *fe2, *tr, elem_mat);
            elmat += elem_mat;
         }

         reduction->AssemblePotSharedFaceMatrix(i, elmat);
      }
   }
}

void ParDarcyForm::AssemblePotHDGSharedFaces(int skip_zeros)
{
   ParMesh *pmesh = pfes_p.GetParMesh();
   DenseMatrix elmat1, elmat2;
   Array<int> vdofs1, vdofs2;
#ifndef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   MFEM_ABORT("Not supported");
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS

   if (hybridization->GetPotConstraintIntegrator())
   {
      int nsfaces = pmesh->GetNSharedFaces();
      for (int i = 0; i < nsfaces; i++)
      {
         const int f = pmesh->GetSharedFace(i);
         hybridization->ComputeAndAssemblePotFaceMatrix(f, elmat1, elmat2, vdofs1,
                                                        vdofs2, skip_zeros);
      }
   }
}
}

#endif // MFEM_USE_MPI
