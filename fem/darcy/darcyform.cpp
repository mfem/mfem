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

#include "darcyform.hpp"

namespace mfem
{

DarcyForm::DarcyForm(FiniteElementSpace *fes_u_, FiniteElementSpace *fes_p_,
                     bool bsymmetrize)
   : fes_u(fes_u_), fes_p(fes_p_), bsym(bsymmetrize)
{
   offsets.SetSize(3);
   offsets[0] = 0;
   offsets[1] = fes_u->GetVSize();
   offsets[2] = fes_p->GetVSize();
   offsets.PartialSum();

   width = height = offsets.Last();
}

BilinearForm* DarcyForm::GetFluxMassForm()
{
   if (!M_u) { M_u = new BilinearForm(fes_u); }
   return M_u;
}

BilinearForm* DarcyForm::GetPotentialMassForm()
{
   if (!M_p) { M_p = new BilinearForm(fes_p); }
   return M_p;
}

NonlinearForm *DarcyForm::GetFluxMassNonlinearForm()
{
   if (!Mnl_u) { Mnl_u = new NonlinearForm(fes_u); }
   return Mnl_u;
}

NonlinearForm* DarcyForm::GetPotentialMassNonlinearForm()
{
   if (!Mnl_p) { Mnl_p = new NonlinearForm(fes_p); }
   return Mnl_p;
}

MixedBilinearForm* DarcyForm::GetFluxDivForm()
{
   if (!B) { B = new MixedBilinearForm(fes_u, fes_p); }
   return B;
}

BlockNonlinearForm *DarcyForm::GetBlockNonlinearForm()
{
   if (!Mnl)
   {
      Array<FiniteElementSpace*> fes({fes_u, fes_p});
      Mnl = new BlockNonlinearForm(fes);
   }
   return Mnl;
}

void DarcyForm::SetAssemblyLevel(AssemblyLevel assembly_level)
{
   assembly = assembly_level;

   if (M_u) { M_u->SetAssemblyLevel(assembly); }
   if (M_p) { M_p->SetAssemblyLevel(assembly); }
   if (Mnl_u) { Mnl_u->SetAssemblyLevel(assembly); }
   if (Mnl_p) { Mnl_p->SetAssemblyLevel(assembly); }
   if (B) { B->SetAssemblyLevel(assembly); }
}

void DarcyForm::EnableReduction(const Array<int> &ess_flux_tdof_list,
                                DarcyReduction *reduction_)
{
   MFEM_ASSERT(!Mnl, "Reduction cannot be used with block nonlinear forms");
   MFEM_ASSERT((M_u || Mnl_u) && (M_p || Mnl_p),
               "Mass forms for the fluxes and potentials must be set prior to this call!");

   delete reduction;
   if (assembly != AssemblyLevel::LEGACY)
   {
      reduction = NULL;
      MFEM_WARNING("Reduction not supported for this assembly level");
      return;
   }
   reduction = reduction_;

   // Automatically load the flux mass integrators
   if (Mnl_u)
   {
      NonlinearFormIntegrator *flux_integ = NULL;
      auto dnlfi = Mnl_u->GetDNFI();
      if (dnlfi->Size())
      {
         SumNLFIntegrator *snlfi = new SumNLFIntegrator(false);
         for (NonlinearFormIntegrator *nlfi : *dnlfi)
         {
            snlfi->AddIntegrator(nlfi);
         }
         flux_integ = snlfi;
      }
      reduction->SetFluxMassNonlinearIntegrator(flux_integ);
   }

   // Automatically load the potential mass integrators
   if (Mnl_p)
   {
      NonlinearFormIntegrator *pot_integ = NULL;
      auto dnlfi = Mnl_p->GetDNFI();
      if (dnlfi->Size())
      {
         SumNLFIntegrator *snlfi = new SumNLFIntegrator(false);
         for (NonlinearFormIntegrator *nlfi : *dnlfi)
         {
            snlfi->AddIntegrator(nlfi);
         }
         pot_integ = snlfi;
      }
      reduction->SetPotMassNonlinearIntegrator(pot_integ);
   }

   reduction->Init(ess_flux_tdof_list);
}

void DarcyForm::EnableHybridization(FiniteElementSpace *constr_space,
                                    BilinearFormIntegrator *constr_flux_integ,
                                    const Array<int> &ess_flux_tdof_list)
{
   MFEM_ASSERT(M_u || Mnl_u || Mnl,
               "Mass form for the fluxes must be set prior to this call!");
   delete hybridization;
   if (assembly != AssemblyLevel::LEGACY)
   {
      delete constr_flux_integ;
      hybridization = NULL;
      MFEM_WARNING("Hybridization not supported for this assembly level");
      return;
   }
   hybridization = new DarcyHybridization(fes_u, fes_p, constr_space, bsym);

   // Automatically load the potential constraint operator from the face integrators
   if (M_p)
   {
      BilinearFormIntegrator *constr_pot_integ = NULL;
      auto fbfi = M_p->GetFBFI();
      if (fbfi->Size())
      {
         SumIntegrator *sbfi = new SumIntegrator(false);
         for (BilinearFormIntegrator *bfi : *fbfi)
         {
            sbfi->AddIntegrator(bfi);
         }
         constr_pot_integ = sbfi;
      }
      hybridization->SetConstraintIntegrators(constr_flux_integ, constr_pot_integ);
   }
   else if (Mnl_p)
   {
      NonlinearFormIntegrator *constr_pot_integ = NULL;
      auto fnlfi = Mnl_p->GetInteriorFaceIntegrators();
      if (fnlfi.Size())
      {
         SumNLFIntegrator *snlfi = new SumNLFIntegrator(false);
         for (NonlinearFormIntegrator *nlfi : fnlfi)
         {
            snlfi->AddIntegrator(nlfi);
         }
         constr_pot_integ = snlfi;
      }
      hybridization->SetConstraintIntegrators(constr_flux_integ, constr_pot_integ);
   }
   else if (Mnl)
   {
      BlockNonlinearFormIntegrator *constr_integ = NULL;
      auto fnlfi = Mnl->GetInteriorFaceIntegrators();
      if (fnlfi.Size())
      {
         SumBlockNLFIntegrator *snlfi = new SumBlockNLFIntegrator(false);
         for (BlockNonlinearFormIntegrator *nlfi : fnlfi)
         {
            snlfi->AddIntegrator(nlfi);
         }
         constr_integ = snlfi;
      }
      hybridization->SetConstraintIntegrators(constr_flux_integ, constr_integ);
   }
   else
   {
      hybridization->SetConstraintIntegrators(constr_flux_integ,
                                              (BilinearFormIntegrator*)NULL);
   }

   // Automatically load the flux mass integrators
   if (Mnl_u)
   {
      NonlinearFormIntegrator *flux_integ = NULL;
      auto dnlfi = Mnl_u->GetDNFI();
      if (dnlfi->Size())
      {
         SumNLFIntegrator *snlfi = new SumNLFIntegrator(false);
         for (NonlinearFormIntegrator *nlfi : *dnlfi)
         {
            snlfi->AddIntegrator(nlfi);
         }
         flux_integ = snlfi;
      }
      hybridization->SetFluxMassNonlinearIntegrator(flux_integ);
   }

   // Automatically load the potential mass integrators
   if (Mnl_p)
   {
      NonlinearFormIntegrator *pot_integ = NULL;
      auto dnlfi = Mnl_p->GetDNFI();
      if (dnlfi->Size())
      {
         SumNLFIntegrator *snlfi = new SumNLFIntegrator(false);
         for (NonlinearFormIntegrator *nlfi : *dnlfi)
         {
            snlfi->AddIntegrator(nlfi);
         }
         pot_integ = snlfi;
      }
      hybridization->SetPotMassNonlinearIntegrator(pot_integ);
   }

   // Automatically load the block integrators
   if (Mnl)
   {
      BlockNonlinearFormIntegrator *block_integ = NULL;
      auto &dnlfi = Mnl->GetDomainIntegrators();
      block_integ = dnlfi[0];
      hybridization->SetBlockNonlinearIntegrator(block_integ, false);
   }

   // Automatically add the boundary flux constraint integrators
   if (B)
   {
      auto bfbfi_marker = B->GetBFBFI_Marker();
      hybridization->UseExternalBdrFluxConstraintIntegrators();

      for (Array<int> *bfi_marker : *bfbfi_marker)
      {
         if (bfi_marker)
         {
            hybridization->AddBdrFluxConstraintIntegrator(constr_flux_integ, *bfi_marker);
         }
         else
         {
            hybridization->AddBdrFluxConstraintIntegrator(constr_flux_integ);
         }
      }
   }

   // Automatically add the boundary potential constraint integrators
   if (M_p)
   {
      auto bfbfi = M_p->GetBFBFI();
      auto bfbfi_marker = M_p->GetBFBFI_Marker();
      hybridization->UseExternalBdrPotConstraintIntegrators();

      for (int i = 0; i < bfbfi->Size(); i++)
      {
         BilinearFormIntegrator *bfi = (*bfbfi)[i];
         Array<int> *bfi_marker = (*bfbfi_marker)[i];
         if (bfi_marker)
         {
            hybridization->AddBdrPotConstraintIntegrator(bfi, *bfi_marker);
         }
         else
         {
            hybridization->AddBdrPotConstraintIntegrator(bfi);
         }
      }
   }
   else if (Mnl_p)
   {
      auto bfnlfi = Mnl_p->GetBdrFaceIntegrators();
      auto bfnlfi_marker = Mnl_p->GetBdrFaceIntegratorsMarkers();
      hybridization->UseExternalBdrPotConstraintIntegrators();

      for (int i = 0; i < bfnlfi.Size(); i++)
      {
         NonlinearFormIntegrator *nlfi = bfnlfi[i];
         Array<int> *nlfi_marker = bfnlfi_marker[i];
         if (nlfi_marker)
         {
            hybridization->AddBdrPotConstraintIntegrator(nlfi, *nlfi_marker);
         }
         else
         {
            hybridization->AddBdrPotConstraintIntegrator(nlfi);
         }
      }
   }
   else if (Mnl)
   {
      auto bfnlfi = Mnl->GetBdrFaceIntegrators();
      auto bfnlfi_marker = Mnl->GetBdrFaceIntegratorsMarkers();
      hybridization->UseExternalBdrPotConstraintIntegrators();

      for (int i = 0; i < bfnlfi.Size(); i++)
      {
         BlockNonlinearFormIntegrator *nlfi = bfnlfi[i];
         Array<int> *nlfi_marker = bfnlfi_marker[i];
         if (nlfi_marker)
         {
            hybridization->AddBdrConstraintIntegrator(nlfi, *nlfi_marker);
         }
         else
         {
            hybridization->AddBdrConstraintIntegrator(nlfi);
         }
      }
   }

   hybridization->Init(ess_flux_tdof_list);
}

void DarcyForm::Assemble(int skip_zeros)
{
   if (M_u)
   {
      if (hybridization)
      {
         DenseMatrix elmat;

         // Element-wise integration
         for (int i = 0; i < fes_u -> GetNE(); i++)
         {
            M_u->ComputeElementMatrix(i, elmat);
#ifndef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
            M_u->AssembleElementMatrix(i, elmat, skip_zeros);
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
            M_u->ComputeElementMatrix(i, elmat);
#ifndef MFEM_DARCY_REDUCTION_ELIM_BCS
            M_u->AssembleElementMatrix(i, elmat, skip_zeros);
#endif //!MFEM_DARCY_REDUCTION_ELIM_BCS
            reduction->AssembleFluxMassMatrix(i, elmat);
         }
      }
      else
      {
         M_u->Assemble(skip_zeros);
      }
   }
   else if (Mnl_u)
   {
      Mnl_u->Setup();
   }

   if (B)
   {
      if (hybridization)
      {
         DenseMatrix elmat;

         // Element-wise integration
         for (int i = 0; i < fes_u -> GetNE(); i++)
         {
            B->ComputeElementMatrix(i, elmat);
#ifndef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
            B->AssembleElementMatrix(i, elmat, skip_zeros);
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
            B->ComputeElementMatrix(i, elmat);
#ifndef MFEM_DARCY_REDUCTION_ELIM_BCS
            B->AssembleElementMatrix(i, elmat, skip_zeros);
#endif //!MFEM_DARCY_REDUCTION_ELIM_BCS
            reduction->AssembleDivMatrix(i, elmat);
         }

         AssembleDivLDGFaces(skip_zeros);
      }
      else
      {
         B->Assemble(skip_zeros);
      }
   }

   if (M_p)
   {
      if (hybridization)
      {
         DenseMatrix elmat;

         // Element-wise integration
         for (int i = 0; i < fes_p -> GetNE(); i++)
         {
            M_p->ComputeElementMatrix(i, elmat);
#ifndef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
            M_p->AssembleElementMatrix(i, elmat, skip_zeros);
#endif //!MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
            hybridization->AssemblePotMassMatrix(i, elmat);
         }

         AssemblePotHDGFaces(skip_zeros);
      }
      else if (reduction)
      {
         DenseMatrix elmat;

         // Element-wise integration
         for (int i = 0; i < fes_p -> GetNE(); i++)
         {
            M_p->ComputeElementMatrix(i, elmat);
#ifndef MFEM_DARCY_REDUCTION_ELIM_BCS
            M_p->AssembleElementMatrix(i, elmat, skip_zeros);
#endif //!MFEM_DARCY_REDUCTION_ELIM_BCS
            reduction->AssemblePotMassMatrix(i, elmat);
         }

         AssemblePotLDGFaces(skip_zeros);
      }
      else
      {
         M_p->Assemble(skip_zeros);
      }
   }
   else if (Mnl_p)
   {
      Mnl_p->Setup();
   }
}

void DarcyForm::Finalize(int skip_zeros)
{
   AllocBlockOp();

   if (block_op)
   {
      if (M_u)
      {
         M_u->Finalize(skip_zeros);
         block_op->SetDiagonalBlock(0, M_u);
      }
      else if (Mnl_u)
      {
         block_op->SetDiagonalBlock(0, Mnl_u);
      }
      else if (Mnl)
      {
         opM.Reset(Mnl, false);
      }

      if (M_p)
      {
         M_p->Finalize(skip_zeros);
         block_op->SetDiagonalBlock(1, M_p, (bsym)?(-1.):(+1.));
      }
      else if (Mnl_p)
      {
         block_op->SetDiagonalBlock(1, Mnl_p, (bsym)?(-1.):(+1.));
      }

      if (B)
      {
         B->Finalize(skip_zeros);

         if (!opBt.Ptr()) { ConstructBT(B); }

         block_op->SetBlock(0, 1, opBt.Ptr(), (bsym)?(-1.):(+1.));
         block_op->SetBlock(1, 0, B, (bsym)?(-1.):(+1.));
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

void DarcyForm::FormLinearSystem(const Array<int> &ess_flux_tdof_list,
                                 BlockVector &x, BlockVector &b, OperatorHandle &A, Vector &X_, Vector &B_,
                                 int copy_interior)
{
   if (assembly != AssemblyLevel::LEGACY)
   {
      Array<int> ess_pot_tdof_list;//empty for discontinuous potentials

      //conforming

      if (M_u)
      {
         M_u->FormLinearSystem(ess_flux_tdof_list, x.GetBlock(0), b.GetBlock(0), opM_u,
                               X_, B_, copy_interior);
         block_op->SetDiagonalBlock(0, opM_u.Ptr());
      }
      else if (Mnl_u)
      {
         Operator *oper_M;
         Mnl_u->FormLinearSystem(ess_flux_tdof_list, x.GetBlock(0), b.GetBlock(0),
                                 oper_M, X_, B_, copy_interior);
         opM_u.Reset(oper_M);
         block_op->SetDiagonalBlock(0, opM_u.Ptr());
      }
      else if (Mnl)
      {
         Operator *oper_M;
         Mnl->FormLinearSystem(ess_flux_tdof_list, x, b, oper_M, X_, B_, copy_interior);
         opM.Reset(oper_M);
      }

      if (M_p)
      {
         M_p->FormLinearSystem(ess_pot_tdof_list, x.GetBlock(1), b.GetBlock(1), opM_p,
                               X_,
                               B_, copy_interior);
         block_op->SetDiagonalBlock(1, opM_p.Ptr(), (bsym)?(-1.):(+1.));
      }
      else if (Mnl_p)
      {
         block_op->SetDiagonalBlock(1, Mnl_p, (bsym)?(-1.):(+1.));
      }

      if (B)
      {
         if (bsym)
         {
            //In the case of the symmetrized system, the sign is oppposite!
            Vector b_(fes_p->GetVSize());
            b_ = 0.;
            B->FormRectangularLinearSystem(ess_flux_tdof_list, ess_pot_tdof_list,
                                           x.GetBlock(0), b_, opB, X_, B_);
            b.GetBlock(1) -= b_;
         }
         else
         {
            B->FormRectangularLinearSystem(ess_flux_tdof_list, ess_pot_tdof_list,
                                           x.GetBlock(0), b.GetBlock(1), opB, X_, B_);
         }

         ConstructBT(opB.Ptr());

         block_op->SetBlock(0, 1, opBt.Ptr(), (bsym)?(-1.):(+1.));
         block_op->SetBlock(1, 0, opB.Ptr(), (bsym)?(-1.):(+1.));
      }

      if (Mnl && opM.Ptr())
      {
         A.Reset(this, false);
      }
      else
      {
         A.Reset(block_op, false);
      }

      X_.MakeRef(x, 0, x.Size());
      B_.MakeRef(b, 0, b.Size());

      return;
   }

   FormSystemMatrix(ess_flux_tdof_list, A);

   //conforming

   if (hybridization)
   {
      // Reduction to the Lagrange multipliers system
      EliminateVDofsInRHS(ess_flux_tdof_list, x, b);
      hybridization->ReduceRHS(b, B_);
      X_.SetSize(B_.Size());
      X_ = 0.0;
   }
   else if (reduction)
   {
      // Reduction to the Lagrange multipliers system
      EliminateVDofsInRHS(ess_flux_tdof_list, x, b);
      reduction->ReduceRHS(b, B_);
      X_.SetSize(B_.Size());
      X_ = 0.0;
   }
   else
   {
      // A, X and B point to the same data as mat, x and b
      EliminateVDofsInRHS(ess_flux_tdof_list, x, b);
      X_.MakeRef(x, 0, x.Size());
      B_.MakeRef(b, 0, b.Size());
      if (!copy_interior)
      {
         x.GetBlock(0).SetSubVectorComplement(ess_flux_tdof_list, 0.0);
         x.GetBlock(1) = 0.;
      }
   }
}

void DarcyForm::FormSystemMatrix(const Array<int> &ess_flux_tdof_list,
                                 OperatorHandle &A)
{
   AllocBlockOp();

   if (block_op)
   {
      Array<int> ess_pot_tdof_list;//empty for discontinuous potentials

      if (M_u)
      {
         M_u->FormSystemMatrix(ess_flux_tdof_list, opM_u);
         block_op->SetDiagonalBlock(0, opM_u.Ptr());
      }
      else if (Mnl_u)
      {
         Operator *oper_M;
         Mnl_u->FormSystemOperator(ess_flux_tdof_list, oper_M);
         opM_u.Reset(oper_M);
         block_op->SetDiagonalBlock(0, opM_u.Ptr());
      }
      else if (Mnl)
      {
         Operator *oper_M;
         Mnl->FormSystemOperator(ess_flux_tdof_list, oper_M);
         opM.Reset(oper_M);
      }

      if (M_p)
      {
         M_p->FormSystemMatrix(ess_pot_tdof_list, opM_p);
         block_op->SetDiagonalBlock(1, opM_p.Ptr(), (bsym)?(-1.):(+1.));
      }
      else if (Mnl_p)
      {
         block_op->SetDiagonalBlock(1, Mnl_p, (bsym)?(-1.):(+1.));
      }

      if (B)
      {
         B->FormRectangularSystemMatrix(ess_flux_tdof_list, ess_pot_tdof_list, opB);

         ConstructBT(opB.Ptr());

         block_op->SetBlock(0, 1, opBt.Ptr(), (bsym)?(-1.):(+1.));
         block_op->SetBlock(1, 0, opB.Ptr(), (bsym)?(-1.):(+1.));
      }
   }

   if (hybridization)
   {
      hybridization->Finalize();
      if (!Mnl_u && !Mnl_p && !Mnl)
      {
         A.Reset(&hybridization->GetMatrix(), false);
      }
      else
      {
         A.Reset(hybridization, false);
      }
   }
   else if (reduction)
   {
      reduction->Finalize();
      if (!Mnl_u && !Mnl_p && !Mnl)
      {
         A.Reset(&reduction->GetMatrix(), false);
      }
      else
      {
         A.Reset(reduction, false);
      }
   }
   else
   {
      if (Mnl && opM.Ptr())
      {
         A.Reset(this, false);
      }
      else
      {
         A.Reset(block_op, false);
      }
   }
}

void DarcyForm::RecoverFEMSolution(const Vector &X, const BlockVector &b,
                                   BlockVector &x)
{
   if (hybridization)
   {
      //conforming
      hybridization->ComputeSolution(b, X, x);
   }
   else if (reduction)
   {
      //conforming
      reduction->ComputeSolution(b, X, x);
   }
   else
   {
      BlockVector X_b(const_cast<Vector&>(X), offsets);
      if (M_u)
      {
         M_u->RecoverFEMSolution(X_b.GetBlock(0), b.GetBlock(0), x.GetBlock(0));
      }
      if (M_p)
      {
         M_p->RecoverFEMSolution(X_b.GetBlock(1), b.GetBlock(1), x.GetBlock(1));
      }
   }
}

void DarcyForm::EliminateVDofsInRHS(const Array<int> &vdofs_flux,
                                    const BlockVector &x, BlockVector &b)
{
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   if (hybridization)
   {
      hybridization->EliminateVDofsInRHS(vdofs_flux, x, b);
      return;
   }
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
   if (reduction)
   {
      reduction->EliminateVDofsInRHS(vdofs_flux, x, b);
      return;
   }
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
   if (B)
   {
      if (bsym)
      {
         //In the case of the symmetrized system, the sign is oppposite!
         Vector b_(fes_p->GetVSize());
         b_ = 0.;
         B->EliminateTrialVDofsInRHS(vdofs_flux, x.GetBlock(0), b_);
         b.GetBlock(1) -= b_;
      }
      else
      {
         B->EliminateTrialVDofsInRHS(vdofs_flux, x.GetBlock(0), b.GetBlock(1));
      }
   }
   if (M_u)
   {
      M_u->EliminateVDofsInRHS(vdofs_flux, x.GetBlock(0), b.GetBlock(0));
   }
   else if (Mnl_u && opM_u.Ptr())
   {
      opM_u.As<ConstrainedOperator>()->EliminateRHS(x.GetBlock(0), b.GetBlock(0));
   }
   else if (Mnl && opM.Ptr())
   {
      opM.As<ConstrainedOperator>()->EliminateRHS(x, b);
   }
}

void DarcyForm::Mult(const Vector &x, Vector &y) const
{
   block_op->Mult(x, y);
   if (opM.Ptr())
   {
      if (bsym)
      {
         BlockVector ynl(offsets);
         opM->Mult(x, ynl);
         ynl.GetBlock(1).Neg();
         y += ynl;
      }
      else
      {
         opM->AddMult(x, y);
      }
   }
}

Operator &DarcyForm::GetGradient(const Vector &x) const
{
   const BlockVector bx(const_cast<Vector&>(x), offsets);

   if (!Mnl && !Mnl_u && !Mnl_p) { return *block_op; }

   if (Mnl_u || Mnl_p)
   {
      if (!block_grad)
      {
         block_grad = new BlockOperator(offsets);
      }

      if (M_u)
      {
         block_grad->SetDiagonalBlock(0, M_u);
      }
      else if (Mnl_u)
      {
         block_grad->SetDiagonalBlock(0, &Mnl_u->GetGradient(bx.GetBlock(0)));
      }

      if (M_p)
      {
         block_grad->SetDiagonalBlock(1, M_p, (bsym)?(-1.):(+1.));
      }
      else if (Mnl_p)
      {
         block_grad->SetDiagonalBlock(1, &Mnl_p->GetGradient(bx.GetBlock(1)),
                                      (bsym)?(-1.):(+1.));
      }

      if (B)
      {
         block_grad->SetBlock(0, 1, opBt.Ptr(), (bsym)?(-1.):(+1.));
         block_grad->SetBlock(1, 0, B, (bsym)?(-1.):(+1.));
      }

      if (!Mnl) { return *block_grad; }
   }

   opG.Reset(new Gradient(*this, x));
   return *opG.Ptr();
}

void DarcyForm::Gradient::Mult(const Vector &x, Vector &y) const
{
   if (p.block_grad)
   {
      p.block_grad->Mult(x, y);
   }
   else
   {
      p.block_op->Mult(x, y);
   }

   if (p.bsym)
   {
      BlockVector ynl(p.offsets);
      G.Mult(x, ynl);
      ynl.GetBlock(1).Neg();
      y += ynl;
   }
   else
   {
      G.AddMult(x, y);
   }
}

void DarcyForm::Update()
{
   if (M_u) { M_u->Update(); }
   if (M_p) { M_p->Update(); }
   if (Mnl_u) { Mnl_u->Update(); }
   if (Mnl_p) { Mnl_p->Update(); }
   if (B) { B->Update(); }
   if (Mnl) { Mnl->Update(); }

   opBt.Clear();

   if (reduction) { reduction->Reset(); }
   if (hybridization) { hybridization->Reset(); }
}

DarcyForm::~DarcyForm()
{
   delete M_u;
   delete M_p;
   delete Mnl_u;
   delete Mnl_p;
   delete B;
   delete Mnl;

   delete block_op;
   delete block_grad;

   delete reduction;
   delete hybridization;
}

void DarcyForm::AssembleDivLDGFaces(int skip_zeros)
{
   Mesh *mesh = fes_p->GetMesh();
   FaceElementTransformations *tr;
#ifndef MFEM_DARCY_REDUCTION_ELIM_BCS
   DenseMatrix elmat1, elmat2;
   Array<int> tr_vdofs1, te_vdofs1, tr_vdofs2, te_vdofs2;
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS

   auto &interior_face_integs = *B->GetFBFI();

   if (interior_face_integs.Size())
   {
      DenseMatrix elmat, elem_mat;

      int nfaces = mesh->GetNumFaces();
      for (int i = 0; i < nfaces; i++)
      {
         tr = mesh -> GetInteriorFaceTransformations (i);
         if (tr == NULL) { continue; }

         const FiniteElement *trial_fe1 = fes_u->GetFE(tr->Elem1No);
         const FiniteElement *trial_fe2 = fes_u->GetFE(tr->Elem2No);
         const FiniteElement *test_fe1 = fes_p->GetFE(tr->Elem1No);
         const FiniteElement *test_fe2 = fes_p->GetFE(tr->Elem2No);

         interior_face_integs[0]->AssembleFaceMatrix(*trial_fe1, *test_fe1, *trial_fe2,
                                                     *test_fe2, *tr, elmat);
         for (int i = 1; i < interior_face_integs.Size(); i++)
         {
            interior_face_integs[i]->AssembleFaceMatrix(*trial_fe1, *test_fe1, *trial_fe2,
                                                        *test_fe2, *tr, elmat);
            elmat += elem_mat;
         }

         reduction->AssembleDivFaceMatrix(i, elmat);

#ifndef MFEM_DARCY_REDUCTION_ELIM_BCS
         fes_u->GetElementVDofs(tr->Elem1No, tr_vdofs1);
         fes_p->GetElementVDofs(tr->Elem1No, te_vdofs1);
         fes_u->GetElementVDofs(tr->Elem2No, tr_vdofs2);
         fes_p->GetElementVDofs(tr->Elem2No, te_vdofs2);
         tr_vdofs1.Append(tr_vdofs2);
         te_vdofs1.Append(te_vdofs2);
         B->SpMat().AddSubMatrix(te_vdofs1, tr_vdofs1, elmat, skip_zeros);
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
      }
   }

   auto &boundary_face_integs = *B->GetBFBFI();
   auto &boundary_face_integs_marker = *B->GetBFBFI_Marker();

   if (boundary_face_integs.Size())
   {
      DenseMatrix elmat, elem_mat;

      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < boundary_face_integs_marker.Size(); k++)
      {
         if (boundary_face_integs_marker[k] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         Array<int> &bdr_marker = *boundary_face_integs_marker[k];
         MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                     "invalid boundary marker for boundary face integrator #"
                     << k << ", counting from zero");
         for (int i = 0; i < bdr_attr_marker.Size(); i++)
         {
            bdr_attr_marker[i] |= bdr_marker[i];
         }
      }

      for (int i = 0; i < fes_p -> GetNBE(); i++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

         tr = mesh -> GetBdrFaceTransformations (i);
         if (tr != NULL)
         {
            const FiniteElement *trial_fe1 = fes_u->GetFE(tr->Elem1No);
            const FiniteElement *test_fe1 = fes_p->GetFE(tr->Elem1No);
            const int tr_ndof1 = trial_fe1->GetDof() * fes_u->GetVDim();
            const int te_ndof1 = test_fe1->GetDof();

            elmat.SetSize(te_ndof1, tr_ndof1);
            elmat = 0.;

            for (int i = 0; i < boundary_face_integs.Size(); i++)
            {
               if (boundary_face_integs_marker[i]
                   && (*boundary_face_integs_marker[i])[bdr_attr-1] == 0) { continue; }

               boundary_face_integs[i]->AssembleFaceMatrix(*trial_fe1, *test_fe1, *trial_fe1,
                                                           *test_fe1, *tr, elem_mat);
               elmat += elem_mat;
            }

            const int face = mesh->GetBdrElementFaceIndex(i);
            reduction->AssembleDivFaceMatrix(face, elmat);

#ifndef MFEM_DARCY_REDUCTION_ELIM_BCS
            fes_u->GetElementVDofs(tr->Elem1No, tr_vdofs1);
            fes_p->GetElementVDofs(tr->Elem1No, te_vdofs1);
            B->SpMat().AddSubMatrix(te_vdofs1, tr_vdofs1, elmat, skip_zeros);
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
         }
      }
   }
}

void DarcyForm::AssemblePotLDGFaces(int skip_zeros)
{
   Mesh *mesh = fes_p->GetMesh();
   FaceElementTransformations *tr;
#ifndef MFEM_DARCY_REDUCTION_ELIM_BCS
   DenseMatrix elmat1, elmat2;
   Array<int> vdofs1, vdofs2;
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS

   auto &interior_face_integs = *M_p->GetFBFI();

   if (interior_face_integs.Size())
   {
      DenseMatrix elmat, elem_mat;

      int nfaces = mesh->GetNumFaces();
      for (int i = 0; i < nfaces; i++)
      {
         tr = mesh -> GetInteriorFaceTransformations (i);
         if (tr == NULL) { continue; }

         const FiniteElement *fe1 = fes_p->GetFE(tr->Elem1No);
         const FiniteElement *fe2 = fes_p->GetFE(tr->Elem2No);

         interior_face_integs[0]->AssembleFaceMatrix(*fe1, *fe2, *tr, elmat);
         for (int i = 1; i < interior_face_integs.Size(); i++)
         {
            interior_face_integs[i]->AssembleFaceMatrix(*fe1, *fe2, *tr, elem_mat);
            elmat += elem_mat;
         }

         reduction->AssemblePotFaceMatrix(i, elmat);

#ifndef MFEM_DARCY_REDUCTION_ELIM_BCS
         const int ndof1 = fe1->GetDof();
         fes_p->GetElementVDofs(tr->Elem1No, vdofs1);
         elmat1.CopyMN(elmat, ndof1, ndof1, 0, 0);
         M_p->SpMat().AddSubMatrix(vdofs1, vdofs1, elmat1, skip_zeros);

         const int ndof2 = fe2->GetDof();
         fes_p->GetElementVDofs(tr->Elem2No, vdofs2);
         elmat2.CopyMN(elmat, ndof2, ndof2, ndof1, ndof1);
         M_p->SpMat().AddSubMatrix(vdofs2, vdofs2, elmat2, skip_zeros);
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
      }
   }

   auto &boundary_face_integs = *M_p->GetBFBFI();
   auto &boundary_face_integs_marker = *M_p->GetBFBFI_Marker();

   if (boundary_face_integs.Size())
   {
      DenseMatrix elmat, elem_mat;

      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < boundary_face_integs_marker.Size(); k++)
      {
         if (boundary_face_integs_marker[k] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         Array<int> &bdr_marker = *boundary_face_integs_marker[k];
         MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                     "invalid boundary marker for boundary face integrator #"
                     << k << ", counting from zero");
         for (int i = 0; i < bdr_attr_marker.Size(); i++)
         {
            bdr_attr_marker[i] |= bdr_marker[i];
         }
      }

      for (int i = 0; i < fes_p -> GetNBE(); i++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

         tr = mesh -> GetBdrFaceTransformations (i);
         if (tr != NULL)
         {
            const FiniteElement *fe1 = fes_p->GetFE(tr->Elem1No);
            const int ndof1 = fe1->GetDof();

            elmat.SetSize(ndof1);
            elmat = 0.;

            for (int i = 0; i < boundary_face_integs.Size(); i++)
            {
               if (boundary_face_integs_marker[i]
                   && (*boundary_face_integs_marker[i])[bdr_attr-1] == 0) { continue; }

               boundary_face_integs[i]->AssembleFaceMatrix(*fe1, *fe1, *tr, elem_mat);
               elmat += elem_mat;
            }

            const int face = mesh->GetBdrElementFaceIndex(i);
            reduction->AssemblePotFaceMatrix(face, elmat);

#ifndef MFEM_DARCY_REDUCTION_ELIM_BCS
            fes_p->GetElementVDofs(tr->Elem1No, vdofs1);
            M_p->SpMat().AddSubMatrix(vdofs1, vdofs1, elmat, skip_zeros);
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
         }
      }
   }
}

void DarcyForm::AssemblePotHDGFaces(int skip_zeros)
{
   Mesh *mesh = fes_p->GetMesh();
   FaceElementTransformations *tr;
   DenseMatrix elmat1, elmat2;
   Array<int> vdofs1, vdofs2;

   if (hybridization->GetPotConstraintIntegrator())
   {
      int nfaces = mesh->GetNumFaces();
      for (int i = 0; i < nfaces; i++)
      {
         tr = mesh -> GetInteriorFaceTransformations (i);
         if (tr == NULL) { continue; }

         hybridization->ComputeAndAssemblePotFaceMatrix(i, elmat1, elmat2, vdofs1,
                                                        vdofs2);
#ifndef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
         M_p->SpMat().AddSubMatrix(vdofs1, vdofs1, elmat1, skip_zeros);
         M_p->SpMat().AddSubMatrix(vdofs2, vdofs2, elmat2, skip_zeros);
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
      }
   }

   auto &boundary_face_integs_marker = *hybridization->GetPotBCBFI_Marker();

   if (boundary_face_integs_marker.Size())
   {
      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < boundary_face_integs_marker.Size(); k++)
      {
         if (boundary_face_integs_marker[k] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         Array<int> &bdr_marker = *boundary_face_integs_marker[k];
         MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                     "invalid boundary marker for boundary face integrator #"
                     << k << ", counting from zero");
         for (int i = 0; i < bdr_attr_marker.Size(); i++)
         {
            bdr_attr_marker[i] |= bdr_marker[i];
         }
      }

      for (int i = 0; i < fes_p -> GetNBE(); i++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

         tr = mesh -> GetBdrFaceTransformations (i);
         if (tr != NULL)
         {
            hybridization->ComputeAndAssemblePotBdrFaceMatrix(i, elmat1, vdofs1);
#ifndef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
            M_p->SpMat().AddSubMatrix(vdofs1, vdofs1, elmat1, skip_zeros);
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
         }
      }
   }
}

void DarcyForm::AllocBlockOp()
{
   bool noblock = false;
#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
   noblock = noblock || reduction;
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   noblock = noblock || hybridization;
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS

   if (!noblock)
   {
      delete block_op;
      block_op = new BlockOperator(offsets);
   }
}

const Operator *DarcyForm::ConstructBT(const MixedBilinearForm *B)
{
   opBt.Reset(Transpose(B->SpMat()));
   return opBt.Ptr();
}

const Operator* DarcyForm::ConstructBT(const Operator *opB)
{
   opBt.Reset(new TransposeOperator(opB));
   return opBt.Ptr();
}

}
