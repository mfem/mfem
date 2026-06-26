// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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

MixedBilinearForm *ParDarcyForm::GetFluxDivForm()
{
   if (!pB) { B.reset(pB = new ParMixedBilinearForm(&pfes_u, &pfes_p)); }
   return pB;
}

LinearForm *ParDarcyForm::GetFluxRHS()
{
   if (!pb_u)
   {
      AllocRHS();
      pb_u = new ParLinearForm();
      pb_u->MakeRef(&pfes_u, block_b->GetBlock(0), 0);
      b_u.reset(pb_u);
   }
   return pb_u;
}

LinearForm *ParDarcyForm::GetPotentialRHS()
{
   if (!pb_p)
   {
      AllocRHS();
      pb_p = new ParLinearForm();
      pb_p->MakeRef(&pfes_p, block_b->GetBlock(1), 0);
      b_p.reset(pb_p);
   }
   return pb_p;
}

void ParDarcyForm::Assemble(int skip_zeros)
{
   if (pB || pM_p)
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
            reduction->AssembleFluxMassMatrix(i, elmat);
         }
      }
      else
      {
         pM_u->Assemble(skip_zeros);
      }
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

   if (pb_u)
   {
      pb_u->Assemble();
      pb_u->SyncAliasMemory(*block_b);
   }

   if (pb_p)
   {
      pb_p->Assemble();
      pb_p->SyncAliasMemory(*block_b);
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
   if (assembly != AssemblyLevel::LEGACY)
   {
      AllocBlockOp();

      X_.SetSize(toffsets.Last());
      B_.SetSize(toffsets.Last());

      BlockVector X_b(X_, toffsets), B_b(B_, toffsets);

      Array<int> ess_pot_tdof_list;//empty for discontinuous potentials

      // flux
      if (pM_u)
      {
         pM_u->FormLinearSystem(ess_flux_tdof_list, x.GetBlock(0), b.GetBlock(0), opM_u,
                                X_b.GetBlock(0), B_b.GetBlock(0), copy_interior);
         block_op->SetDiagonalBlock(0, opM_u.Ptr());
      }
      else
      {
         const Operator *P = pfes_u.GetProlongationMatrix();
         P->MultTranspose(b.GetBlock(0), B_b.GetBlock(0));
         const Operator *R = pfes_u.GetRestrictionOperator();
         R->Mult(x.GetBlock(0), X_b.GetBlock(0));

         if (!copy_interior)
         {
            X_b.GetBlock(0).SetSubVectorComplement(ess_flux_tdof_list, 0.0);
         }
      }

      // potential
      if (pM_p)
      {
         Operator *oper_M;
         pM_p->FormSystemOperator(ess_pot_tdof_list, oper_M);
         opM_p.Reset(oper_M);
         block_op->SetDiagonalBlock(1, opM_p.Ptr(), (bsym)?(-1.):(+1.));
      }

      B_b.GetBlock(1) = b.GetBlock(1);

      if (copy_interior)
      {
         X_b.GetBlock(1) = x.GetBlock(1);
      }
      else
      {
         X_b.GetBlock(1) = 0.;
      }

      // divergence
      if (pB)
      {
         Vector bp(fes_p->GetVSize()), Bp;
         bp = 0.;

         pB->FormRectangularLinearSystem(ess_flux_tdof_list, ess_pot_tdof_list,
                                         x.GetBlock(0), bp, opB, X_b.GetBlock(0), Bp);

         if (bsym)
         {
            //In the case of the symmetrized system, the sign is oppposite!
            B_b.GetBlock(1) -= Bp;
         }
         else
         {
            B_b.GetBlock(1) += Bp;
         }

         ConstructBT(opB);

         block_op->SetBlock(0, 1, opBt.Ptr(), (bsym)?(-1.):(+1.));
         block_op->SetBlock(1, 0, opB.Ptr(), (bsym)?(-1.):(+1.));
      }

      A.Reset(block_op.get(), false);

      return;
   }

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

      const Operator &P = *pfes_u.GetProlongationMatrix();
      const Operator &R = *pfes_u.GetRestrictionOperator();
      P.MultTranspose(b.GetBlock(0), true_B.GetBlock(0));
      R.Mult(x.GetBlock(0), true_X.GetBlock(0));

      true_B.GetBlock(1) = b.GetBlock(1);
      true_X.GetBlock(1) = x.GetBlock(1);

      ParallelEliminateTDofsInRHS(ess_flux_tdof_list, true_X, true_B);

      R.MultTranspose(true_B.GetBlock(0), b.GetBlock(0));
      b.GetBlock(1) = true_B.GetBlock(1);

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
      else if (hybridization)
      {
         hybridization->EliminateTraceTrueDofsInRHS(X_, B_);
      }
   }
   else
   {
      X_.SetSize(toffsets.Last());
      B_.SetSize(toffsets.Last());
      BlockVector block_X(X_, toffsets), block_B(B_, toffsets);

      // Variational restriction with P
      const Operator &P = *pfes_u.GetProlongationMatrix();
      const Operator &R = *pfes_u.GetRestrictionOperator();
      P.MultTranspose(b.GetBlock(0), block_B.GetBlock(0));
      R.Mult(x.GetBlock(0), block_X.GetBlock(0));

      block_B.GetBlock(1) = b.GetBlock(1);
      block_X.GetBlock(1) = x.GetBlock(1);

      ParallelEliminateTDofsInRHS(ess_flux_tdof_list, block_X, block_B);
      if (!copy_interior)
      {
         block_X.GetBlock(0).SetSubVectorComplement(ess_flux_tdof_list, 0.0);
         block_X.GetBlock(1) = 0.;
      }
   }
}

void ParDarcyForm::FormLinearSystem(const Array<int> &ess_flux_tdof_list,
                                    BlockVector &x, OperatorHandle &A, Vector &X_, Vector &B_, int copy_interior)
{
   AllocRHS();

   FormLinearSystem(ess_flux_tdof_list, x, *block_b, A, X_, B_, copy_interior);
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

      if (pM_p)
      {
         pM_p->FormSystemMatrix(ess_pot_tdof_list, opM_p);
         block_op->SetDiagonalBlock(1, opM_p.Ptr(), (bsym)?(-1.):(+1.));
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
      hybridization->GetParallelMatrix(A);
   }
   else if (reduction)
   {
      reduction->Finalize();
      reduction->GetParallelMatrix(A);
   }
   else
   {
      A.Reset(block_op.get(), false);
   }
}

void ParDarcyForm::RecoverFEMSolution(const Vector &X_, const BlockVector &b,
                                      BlockVector &x)
{
   if (reduction || hybridization)
   {
      // Primal unknowns recovery
      BlockVector true_X(toffsets), true_B(toffsets);

      const Operator &P = *pfes_u.GetProlongationMatrix();
      const Operator &R = *pfes_u.GetRestrictionOperator();
      P.MultTranspose(b.GetBlock(0), true_B.GetBlock(0));
      R.Mult(x.GetBlock(0), true_X.GetBlock(0));

      true_B.GetBlock(1) = b.GetBlock(1);
      true_X.GetBlock(1) = x.GetBlock(1);

      if (hybridization)
      {
         hybridization->ComputeSolution(true_B, X_, true_X);
      }
      else if (reduction)
      {
         reduction->ComputeSolution(true_B, X_, true_X);
      }

      P.Mult(true_X.GetBlock(0), x.GetBlock(0));
      x.GetBlock(1) = true_X.GetBlock(1);
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
         const Operator &P = *pfes_u.GetProlongationMatrix();
         P.Mult(X.GetBlock(0), x.GetBlock(0));
      }

      if (pM_p)
      {
         pM_p->RecoverFEMSolution(X.GetBlock(1), b.GetBlock(1), x.GetBlock(1));
      }
      else
      {
         x.GetBlock(1) = X.GetBlock(1);
      }
   }
}

void ParDarcyForm::RecoverFEMSolution(const Vector &X, BlockVector &x)
{
   MFEM_ASSERT(block_b, "RHS does not exist");

   RecoverFEMSolution(X, *block_b, x);
}

void ParDarcyForm::ParallelEliminateTDofsInRHS(const Array<int> &tdofs_flux,
                                               const BlockVector &x, BlockVector &b)
{
   if (hybridization)
   {
      hybridization->EliminateTrueDofsInRHS(tdofs_flux, x, b);
      return;
   }
   if (reduction)
   {
      reduction->EliminateTrueDofsInRHS(tdofs_flux, x, b);
      return;
   }
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
   bool noblock = reduction || hybridization;

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

   auto &interior_face_integs = *B->GetFBFI();

   if (interior_face_integs.Size())
   {
      DenseMatrix elmat, elem_mat;

      const int nsfaces = pmesh->GetNSharedFaces();
      for (int sf = 0; sf < nsfaces; sf++)
      {
         tr = pmesh->GetSharedFaceTransformations(sf);
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

         reduction->AssembleDivSharedFaceMatrix(sf, elmat);
      }
   }
}

void ParDarcyForm::AssemblePotLDGSharedFaces(int skip_zeros)
{
   ParMesh *pmesh = pfes_p.GetParMesh();
   const int NE = pmesh->GetNE();
   FaceElementTransformations *tr;

   auto &interior_face_integs = *M_p->GetFBFI();

   if (interior_face_integs.Size())
   {
      DenseMatrix elmat, elem_mat;

      const int nsfaces = pmesh->GetNSharedFaces();
      for (int sf = 0; sf < nsfaces; sf++)
      {
         tr = pmesh->GetSharedFaceTransformations(sf);

         const FiniteElement *fe1 = pfes_p.GetFE(tr->Elem1No);
         const FiniteElement *fe2 = pfes_p.GetFaceNbrFE(tr->Elem2No - NE);

         interior_face_integs[0]->AssembleFaceMatrix(*fe1, *fe2, *tr, elmat);
         for (int i = 1; i < interior_face_integs.Size(); i++)
         {
            interior_face_integs[i]->AssembleFaceMatrix(*fe1, *fe2, *tr, elem_mat);
            elmat += elem_mat;
         }

         reduction->AssemblePotSharedFaceMatrix(sf, elmat);
      }
   }
}

void ParDarcyForm::AssemblePotHDGSharedFaces(int skip_zeros)
{
   ParMesh *pmesh = pfes_p.GetParMesh();
   DenseMatrix elmat1, elmat2;
   Array<int> vdofs1, vdofs2;

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
