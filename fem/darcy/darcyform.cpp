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

#include "darcyform.hpp"
#include "bilininteg_hdg.hpp"
#include "../hyperbolic.hpp"
#include "../nonlininteg_mixed.hpp"

namespace mfem
{

DarcyForm::DarcyForm(FiniteElementSpace *fes_u_, FiniteElementSpace *fes_p_,
                     bool bsym_)
   : fes_u(fes_u_), fes_p(fes_p_), bsym(bsym_)
{
   sequence = fes_u->GetSequence();
   UpdateOffsetsAndSize();
}

void DarcyForm::UpdateOffsetsAndSize()
{
   offsets.SetSize(3);
   offsets[0] = 0;
   offsets[1] = fes_u->GetVSize();
   offsets[2] = fes_p->GetVSize();
   offsets.PartialSum();

   toffsets.MakeRef(offsets);

   width = height = offsets.Last();

   block_op.reset();
   block_grad.reset();
   if (block_b) { block_b->Update(offsets); *block_b = 0.; }
}

void DarcyForm::UpdateTOffsetsAndSize()
{
   if (!toffsets.OwnsData()) { toffsets.DeleteAll(); }
   toffsets.SetSize(3);
   toffsets[0] = 0;
   toffsets[1] = fes_u->GetTrueVSize();
   toffsets[2] = fes_p->GetVSize();
   toffsets.PartialSum();

   width = height = toffsets.Last();

   block_op.reset();
   block_grad.reset();
}

BilinearForm* DarcyForm::GetFluxMassForm()
{
   if (!M_u)
   {
      M_u.reset(new BilinearForm(fes_u));
      M_u->SetAssemblyLevel(assembly);
   }
   return M_u.get();
}

BilinearForm* DarcyForm::GetPotentialMassForm()
{
   if (!M_p)
   {
      M_p.reset(new BilinearForm(fes_p));
      M_p->SetAssemblyLevel(assembly);
   }
   return M_p.get();
}

NonlinearForm *DarcyForm::GetFluxMassNonlinearForm()
{
   if (!Mnl_u) { Mnl_u.reset(new NonlinearForm(fes_u)); }
   return Mnl_u.get();
}

NonlinearForm* DarcyForm::GetPotentialMassNonlinearForm()
{
   if (!Mnl_p) { Mnl_p.reset(new NonlinearForm(fes_p)); }
   return Mnl_p.get();
}

MixedBilinearForm* DarcyForm::GetFluxDivForm()
{
   if (!B)
   {
      B.reset(new MixedBilinearForm(fes_u, fes_p));
      B->SetAssemblyLevel(assembly);
   }
   return B.get();
}

BlockNonlinearForm *DarcyForm::GetBlockNonlinearForm()
{
   if (!Mnl)
   {
      Array<FiniteElementSpace*> fes({fes_u, fes_p});
      Mnl.reset(new BlockNonlinearForm(fes));
   }
   return Mnl.get();
}

LinearForm *DarcyForm::GetFluxRHS()
{
   if (!b_u)
   {
      AllocRHS();
      b_u.reset(new LinearForm());
      b_u->MakeRef(fes_u, block_b->GetBlock(0), 0);
   }
   return b_u.get();
}

LinearForm *DarcyForm::GetPotentialRHS()
{
   if (!b_p)
   {
      AllocRHS();
      b_p.reset(new LinearForm());
      b_p->MakeRef(fes_p, block_b->GetBlock(1), 0);
   }
   return b_p.get();
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

   reduction.reset();
   if (assembly != AssemblyLevel::LEGACY)
   {
      MFEM_WARNING("Reduction not supported for this assembly level");
      delete reduction_;
      return;
   }
   reduction.reset(reduction_);

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

void DarcyForm::EnableFluxReduction()
{
   MFEM_ASSERT(M_u || Mnl_u,
               "Mass forms for the fluxes must be set prior to this call!");

   Array<int> ess_flux_tdof_list; //empty
   EnableReduction(ess_flux_tdof_list, new DarcyFluxReduction(fes_u, fes_p));
}

void DarcyForm::EnablePotentialReduction(const Array<int> &ess_flux_tdof_list)
{
   MFEM_ASSERT((M_u || Mnl_u) && (M_p || Mnl_p),
               "Mass forms for the fluxes and potentials must be set prior to this call!");

   EnableReduction(ess_flux_tdof_list, new DarcyPotentialReduction(fes_u, fes_p));
}

void DarcyForm::EnableHybridization(FiniteElementSpace *constr_space,
                                    BilinearFormIntegrator *constr_flux_integ,
                                    const Array<int> &ess_flux_tdof_list)
{
   MFEM_ASSERT(M_u || Mnl_u || Mnl,
               "Mass form for the fluxes must be set prior to this call!");

   hybridization.reset();
   if (assembly != AssemblyLevel::LEGACY)
   {
      delete constr_flux_integ;
      MFEM_WARNING("Hybridization not supported for this assembly level");
      return;
   }
   hybridization.reset(new DarcyHybridization(fes_u, fes_p, constr_space, bsym));

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

         AssembleFluxMassBdrFaces(skip_zeros);
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

   if (b_u)
   {
      b_u->Assemble();
      b_u->SyncAliasMemory(*block_b);
   }

   if (b_p)
   {
      b_p->Assemble();
      b_p->SyncAliasMemory(*block_b);
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
         block_op->SetDiagonalBlock(0, M_u.get());
      }
      else if (Mnl_u)
      {
         block_op->SetDiagonalBlock(0, Mnl_u.get());
      }

      if (M_p)
      {
         M_p->Finalize(skip_zeros);
         block_op->SetDiagonalBlock(1, M_p.get(), (bsym)?(-1.):(+1.));
      }
      else if (Mnl_p)
      {
         block_op->SetDiagonalBlock(1, Mnl_p.get(), (bsym)?(-1.):(+1.));
      }

      if (B)
      {
         B->Finalize(skip_zeros);

         if (!opBt.Ptr()) { ConstructBT(B.get()); }

         block_op->SetBlock(0, 1, opBt.Ptr(), (bsym)?(-1.):(+1.));
         block_op->SetBlock(1, 0, B.get(), (bsym)?(-1.):(+1.));
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
   const SparseMatrix *P = fes_u->GetConformingProlongation();

   if (assembly != AssemblyLevel::LEGACY)
   {
      AllocBlockOp(true);

      if (!P)
      {
         X_.MakeRef(x, 0, x.Size());
         B_.MakeRef(b, 0, b.Size());
      }
      else
      {
         X_.SetSize(toffsets.Last());
         B_.SetSize(toffsets.Last());
      }

      BlockVector X_b(X_, toffsets), B_b(B_, toffsets);

      Array<int> ess_pot_tdof_list;//empty for discontinuous potentials

      // flux
      if (M_u)
      {
         M_u->FormLinearSystem(ess_flux_tdof_list, x.GetBlock(0), b.GetBlock(0), opM_u,
                               X_b.GetBlock(0), B_b.GetBlock(0), copy_interior);
         block_op->SetDiagonalBlock(0, opM_u.Ptr());
      }
      else
      {
         if (Mnl_u)
         {
            Mnl_u->SetEssentialTrueDofs(ess_flux_tdof_list);
            B_b.GetBlock(0).SetSubVector(ess_flux_tdof_list, 0.);
            block_op->SetDiagonalBlock(0, Mnl_u.get());
         }
         else if (Mnl)
         {
            Array<Array<int>*> ess_tdof_lists
            {
               const_cast<Array<int>*>(&ess_flux_tdof_list),
               const_cast<Array<int>*>(&ess_pot_tdof_list)
            };
            Array<Vector*> rhss
            {
               &B_b.GetBlock(0),
               &B_b.GetBlock(1)
            };
            Mnl->SetEssentialTrueDofs(ess_tdof_lists, rhss);
         }

         if (P)
         {
            P->MultTranspose(b.GetBlock(0), B_b.GetBlock(0));
            const Operator *R = fes_u->GetRestrictionOperator();
            R->Mult(x.GetBlock(0), X_b.GetBlock(0));
         }

         if (!copy_interior)
         {
            X_b.GetBlock(0).SetSubVectorComplement(ess_flux_tdof_list, 0.0);
         }
      }

      // potential
      if (M_p)
      {
         Operator *oper_M;
         M_p->FormSystemOperator(ess_pot_tdof_list, oper_M);
         opM_p.Reset(oper_M);
         block_op->SetDiagonalBlock(1, opM_p.Ptr(), (bsym)?(-1.):(+1.));
      }
      else if (Mnl_p)
      {
         block_op->SetDiagonalBlock(1, Mnl_p.get(), (bsym)?(-1.):(+1.));
      }

      if (P)
      {
         B_b.GetBlock(1) = b.GetBlock(1);
      }

      if (copy_interior && P)
      {
         X_b.GetBlock(1) = x.GetBlock(1);
      }
      else
      {
         X_b.GetBlock(1) = 0.;
      }

      // divergence
      if (B)
      {
         Vector bp(fes_p->GetVSize()), Bp;
         bp = 0.;

         B->FormRectangularLinearSystem(ess_flux_tdof_list, ess_pot_tdof_list,
                                        x.GetBlock(0), bp, opB, X_b.GetBlock(0), Bp);

         if (bsym)
         {
            //In the case of the symmetrized system, the sign is opposite!
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

      if (Mnl)
      {
         A.Reset(this, false);
      }
      else
      {
         A.Reset(block_op.get(), false);
      }

      return;
   }

   FormSystemMatrix(ess_flux_tdof_list, A);

   if (!P) // conforming space
   {
      if (hybridization || reduction)
      {
         // Reduction to the single equation system
         EliminateVDofsInRHS(ess_flux_tdof_list, x, b);
         if (hybridization)
         {
            hybridization->ReduceRHS(b, B_);
         }
         else
         {
            reduction->ReduceRHS(b, B_);
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
   else // non-conforming space
   {
      if (hybridization || reduction)
      {
         // Reduction to the Lagrange multipliers system
         const SparseMatrix *R = fes_u->GetConformingRestriction();
         BlockVector conf_b(toffsets), conf_x(toffsets);
         P->MultTranspose(b.GetBlock(0), conf_b.GetBlock(0));
         conf_b.GetBlock(1) = b.GetBlock(1);
         R->Mult(x.GetBlock(0), conf_x.GetBlock(0));
         conf_x.GetBlock(1) = x.GetBlock(1);
         EliminateTrueDofsInRHS(ess_flux_tdof_list, conf_x, conf_b);
         R->MultTranspose(conf_b.GetBlock(0),
                          b.GetBlock(0)); // store eliminated rhs in b
         b.GetBlock(1) = conf_b.GetBlock(1);
         if (hybridization)
         {
            hybridization->ReduceRHS(conf_b, B_);
         }
         else
         {
            reduction->ReduceRHS(conf_b, B_);
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
         // Variational restriction with P
         const SparseMatrix *R = fes_u->GetConformingRestriction();
         B_.SetSize(toffsets.Last());
         BlockVector block_B(B_, toffsets);
         P->MultTranspose(b.GetBlock(0), block_B.GetBlock(0));
         block_B.GetBlock(1) = b.GetBlock(1);
         X_.SetSize(toffsets.Last());
         BlockVector block_X(X_, toffsets);
         R->Mult(x.GetBlock(0), block_X.GetBlock(0));
         block_X.GetBlock(1) = x.GetBlock(1);
         EliminateTrueDofsInRHS(ess_flux_tdof_list, block_X, block_B);
         if (!copy_interior)
         {
            block_X.GetBlock(0).SetSubVectorComplement(ess_flux_tdof_list, 0.0);
            block_X.GetBlock(1) = 0.;
         }
      }
   }
}

void DarcyForm::FormLinearSystem(const Array<int> &ess_flux_tdof_list,
                                 BlockVector &x, OperatorHandle &A,
                                 Vector &X_, Vector &B_, int copy_interior)
{
   AllocRHS();

   FormLinearSystem(ess_flux_tdof_list, x, *block_b, A, X_, B_, copy_interior);
}

void DarcyForm::FormSystemMatrix(const Array<int> &ess_flux_tdof_list,
                                 OperatorHandle &A)
{
   AllocBlockOp(true);

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
         Mnl_u->SetEssentialTrueDofs(ess_flux_tdof_list);
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
         Mnl->SetEssentialTrueDofs(ess_tdof_lists, rhss);
      }

      if (M_p)
      {
         M_p->FormSystemMatrix(ess_pot_tdof_list, opM_p);
         block_op->SetDiagonalBlock(1, opM_p.Ptr(), (bsym)?(-1.):(+1.));
      }
      else if (Mnl_p)
      {
         block_op->SetDiagonalBlock(1, Mnl_p.get(), (bsym)?(-1.):(+1.));
      }

      if (B)
      {
         B->FormRectangularSystemMatrix(ess_flux_tdof_list, ess_pot_tdof_list, opB);

         ConstructBT(opB);

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
         A.Reset(hybridization.get(), false);
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
         A.Reset(reduction.get(), false);
      }
   }
   else
   {
      A.Reset(this, false);
   }
}

void DarcyForm::RecoverFEMSolution(const Vector &X, const BlockVector &b,
                                   BlockVector &x)
{
   const SparseMatrix *P = fes_u->GetConformingProlongation();
   if (!P) // conforming space
   {
      if (hybridization)
      {
         hybridization->ComputeSolution(b, X, x);
      }
      else if (reduction)
      {
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
   else // non-conforming space
   {
      if (hybridization || reduction)
      {
         // Primal unknowns recovery
         const SparseMatrix *R = fes_u->GetConformingRestriction();
         BlockVector conf_b(toffsets), conf_x(toffsets);
         P->MultTranspose(b.GetBlock(0), conf_b.GetBlock(0));
         conf_b.GetBlock(1) = b.GetBlock(1);
         R->Mult(x.GetBlock(0), conf_x.GetBlock(0));
         conf_x.GetBlock(1) = x.GetBlock(1);

         if (hybridization)
         {
            hybridization->ComputeSolution(conf_b, X, conf_x);
         }
         else
         {
            reduction->ComputeSolution(conf_b, X, conf_x);
         }

         P->Mult(conf_x.GetBlock(0), x.GetBlock(0));
         x.GetBlock(1) = conf_x.GetBlock(1);
      }
      else
      {
         // Apply conforming prolongation
         BlockVector X_b(const_cast<Vector&>(X), toffsets);
         P->Mult(X_b.GetBlock(0), x.GetBlock(0));
         x.GetBlock(1) = X_b.GetBlock(1);
      }
   }
}

void DarcyForm::RecoverFEMSolution(const Vector &X, BlockVector &x)
{
   MFEM_ASSERT(block_b, "RHS does not exist");

   RecoverFEMSolution(X, *block_b, x);
}

/** @brief Whether @a integ is a T, looking inside a block-diagonal wrapper.

    A system installs one VectorBlockDiagonalIntegrator per term, replicating a
    scalar integrator once per equation. A dynamic_cast for a particular
    integrator type therefore has to look inside it: the wrapper is not a
    ConvectionIntegrator however convective its blocks are, and a filter that
    does not unwrap classifies every term of a system as the opposite of what
    it is. BilinearFormIntegrator derives from NonlinearFormIntegrator, so this
    serves the non-linear lists too. */
template <typename T>
static bool IsOrWraps(NonlinearFormIntegrator *integ)
{
   if (dynamic_cast<T*>(integ)) { return true; }
   if (auto *blk = dynamic_cast<VectorBlockDiagonalIntegrator*>(integ))
   {
      // The blocks of one wrapper are replicas of a single term, so the first
      // decides for all of them.
      return dynamic_cast<T*>(blk->GetIntegrator(0)) != NULL;
   }
   return false;
}

void DarcyForm::ReconstructTotalFlux(const BlockVector &sol,
                                     const Vector &sol_r, GridFunction &ut) const
{
   if (!hybridization) { return; }

   // One field or many. The total flux carries one block per equation and the
   // block layout is the one the whole branch uses -- equation outermost --
   // so the space below is built with vdim rather than as a scalar. The three
   // spaces have to agree on the count: the flux law is stated per equation
   // (see DarcyHybridization::total_flux_fun) and the constraint's face dofs
   // are what the total flux's are matched against.
   const int neq = fes_p->GetVDim();
   MFEM_VERIFY(neq == hybridization->ConstraintFESpace()->GetVDim(),
               "the potential carries " << neq << " field(s) and the trace "
               << hybridization->ConstraintFESpace()->GetVDim()
               << "; reconstruction needs them equal");

   // automatically set up the finite element space
   if (!ut.FESpace())
   {
      Mesh *mesh = fes_u->GetMesh();
      const int dim = fes_u->GetMesh()->Dimension();
      const FiniteElementCollection *u_coll = fes_u->FEColl();
      int ut_order = u_coll->GetOrder();
      if (dynamic_cast<const RT_FECollection*>(u_coll)
          || dynamic_cast<const BrokenRT_FECollection*>(u_coll)) { ut_order--; }
      FiniteElementCollection *ut_coll = new RT_FECollection(ut_order, dim);
      FiniteElementSpace *ut_space = NULL;
#ifdef MFEM_USE_MPI
      ParMesh *pmesh = dynamic_cast<ParMesh*>(mesh);
      if (pmesh)
      {
         ut_space = new ParFiniteElementSpace(pmesh, ut_coll, neq);
      }
      else
#endif //MFEM_USE_MPI
      {
         ut_space = new FiniteElementSpace(mesh, ut_coll, neq);
      }

      ut.SetSpace(ut_space);
      ut.MakeOwner(ut_coll);
   }

   VectorCoefficient *vel = NULL;
   const FluxFunction *flux_fun = NULL;
   if (M_p && M_p->GetDBFI())
   {
      auto &dbfis = *M_p->GetDBFI();
      if (dbfis.Size())
      {
         for (BilinearFormIntegrator *dbfi : dbfis)
         {
            auto *ci = dynamic_cast<ConvectionIntegrator*>(dbfi);
            if (ci) { vel = &ci->GetVelocity(); break; }

            auto *cci = dynamic_cast<ConservativeConvectionIntegrator*>(dbfi);
            if (cci) { vel = &cci->GetVelocity(); break; }
         }
      }
   }
   else if (Mnl_p && Mnl_p->GetDNFI())
   {
      auto &dnfis = *Mnl_p->GetDNFI();
      if (dnfis.Size())
      {
         for (NonlinearFormIntegrator *dnfi : dnfis)
         {
            auto *ci = dynamic_cast<ConvectionIntegrator*>(dnfi);
            if (ci) { vel = &ci->GetVelocity(); break; }

            auto *cci = dynamic_cast<ConservativeConvectionIntegrator*>(dnfi);
            if (cci) { vel = &cci->GetVelocity(); break; }

            auto *hi = dynamic_cast<HyperbolicFormIntegrator*>(dnfi);
            if (hi) { flux_fun = &hi->GetFluxFunction(); break; }
         }
      }
   }

   if (vel)
   {
      auto fx = [vel](ElementTransformation &Tr, const Vector &q,
                      const Vector &p, Vector &qt)
      {
         qt = q;

         // One velocity, every equation convected by it: the block of
         // equation e picks up p(e) times it.
         const int neq = p.Size();
         const int dim = q.Size() / neq;
         Vector cp(dim);
         vel->Eval(cp, Tr, Tr.GetIntPoint());
         for (int e = 0; e < neq; e++)
         {
            Vector qt_e(qt.GetData() + e * dim, dim);
            qt_e.Add(p(e), cp);
         }
      };
      hybridization->ReconstructTotalFlux(sol, sol_r, fx, ut);
   }
   else if (flux_fun)
   {
      auto fx = [flux_fun](ElementTransformation &Tr, const Vector &q,
                           const Vector &p, Vector &qt)
      {
         qt = q;

         // ComputeFlux() is already stated per equation -- a state vector in,
         // an (neq x dim) flux out -- so only the shape passed to it moves.
         const int neq = p.Size();
         Vector qc(q.Size());
         DenseMatrix flux(qc.GetData(), neq, q.Size() / neq);
         flux_fun->ComputeFlux(p, Tr, flux);
         qt += qc;
      };
      hybridization->ReconstructTotalFlux(sol, sol_r, fx, ut);
   }
   else
   {
      auto fx = [](ElementTransformation &Tr, const Vector &q, const Vector &p,
                   Vector &qt)
      {
         qt = q;
      };
      hybridization->ReconstructTotalFlux(sol, sol_r, fx, ut);
   }
}

void DarcyForm::ReconstructFluxAndPot(const BlockVector &sol,
                                      const GridFunction &ut, GridFunction &u,
                                      GridFunction &p, GridFunction &tr) const
{
   if (!hybridization) { return; }

   // One field or many; see ReconstructTotalFlux() above for the layout and
   // why the two counts have to agree.
   const int neq = fes_p->GetVDim();
   MFEM_VERIFY(neq == hybridization->ConstraintFESpace()->GetVDim(),
               "the potential carries " << neq << " field(s) and the trace "
               << hybridization->ConstraintFESpace()->GetVDim()
               << "; reconstruction needs them equal");

   // flux space
   if (!u.FESpace())
   {
      Mesh *mesh = fes_u->GetMesh();
      const FiniteElementCollection *u_coll = fes_u->FEColl();
      int us_order = u_coll->GetOrder() + 1;
      if (dynamic_cast<const RT_FECollection*>(u_coll)
          || dynamic_cast<const BrokenRT_FECollection*>(u_coll)) { us_order--; }
      const int vdim = fes_u->GetVDim();
      FiniteElementCollection *us_coll = u_coll->Clone(us_order);
      FiniteElementSpace *us_space;
#ifdef MFEM_USE_MPI
      ParMesh *pmesh = dynamic_cast<ParMesh*>(mesh);
      if (pmesh)
      {
         us_space = new ParFiniteElementSpace(pmesh, us_coll, vdim);
      }
      else
#endif //MFEM_USE_MPI
      {
         us_space = new FiniteElementSpace(mesh, us_coll, vdim);
      }

      u.SetSpace(us_space);
      u.MakeOwner(us_coll);
   }

   // potential space
   if (!p.FESpace())
   {
      Mesh *mesh = fes_p->GetMesh();
      const FiniteElementCollection *p_coll = fes_p->FEColl();
      const int ps_order = p_coll->GetOrder() + 1;
      FiniteElementCollection *ps_coll = p_coll->Clone(ps_order);
      FiniteElementSpace *ps_space;
#ifdef MFEM_USE_MPI
      ParMesh *pmesh = dynamic_cast<ParMesh*>(mesh);
      if (pmesh)
      {
         ps_space = new ParFiniteElementSpace(pmesh, ps_coll, neq);
      }
      else
#endif //MFEM_USE_MPI
      {
         ps_space = new FiniteElementSpace(mesh, ps_coll, neq);
      }
      p.SetSpace(ps_space);
      p.MakeOwner(ps_coll);
   }

   // trace space
   if (!tr.FESpace())
   {
      Mesh *mesh = fes_u->GetMesh();
      const FiniteElementCollection *tr_coll =
         hybridization->ConstraintFESpace()->FEColl();
      int trs_order = tr_coll->GetOrder() + 1;
      if (dynamic_cast<const RT_FECollection*>(tr_coll)) { trs_order--; }
      FiniteElementCollection *trs_coll = tr_coll->Clone(trs_order);
      FiniteElementSpace *trs_space;
#ifdef MFEM_USE_MPI
      ParMesh *pmesh = dynamic_cast<ParMesh*>(mesh);
      if (pmesh)
      {
         trs_space = new ParFiniteElementSpace(pmesh, trs_coll, neq);
      }
      else
#endif //MFEM_USE_MPI
      {
         trs_space = new FiniteElementSpace(mesh, trs_coll, neq);
      }

      tr.SetSpace(trs_space);
      tr.MakeOwner(trs_coll);
   }

   GridFunction pc(const_cast<FiniteElementSpace*>(fes_p),
                   const_cast<Vector&>(sol.GetBlock(1)), 0);

   // A solution-dependent flux law has no linear mass form to lift onto the
   // enriched space, so one is built by freezing the law at the computed
   // potential. That has to be redone every call, hence the form cannot be
   // cached in that case.
   const bool frozen_flux = !M_u && !(Mnl_u && Mnl_u->GetDNFI()->Size());

   // define reconstructed DarcyForm
   if (frozen_flux || !reconstruction ||
       reconstruction->FluxFESpace() != u.FESpace() ||
       reconstruction->PotentialFESpace() != p.FESpace())
   {
      reconstruction.reset(new DarcyForm(u.FESpace(), p.FESpace()));

      // Only the domain integrators of the flux mass are lifted, and that is
      // deliberate rather than an omission -- see the note above the class
      // definition of the reconstruction in darcyform.hpp. The local problem
      // is not the assembled problem restricted to an element: its trace is
      // free on every face, boundary faces included, and the boundary
      // condition reaches it through the reconstructed total flux and the
      // element average rather than through the forms. Lifting a boundary
      // face term onto it was tried and measured, and it costs the
      // postprocessed potential its order -- k+2 falls to about 1.25 on the
      // extension miniapp, with the error 5e4 times larger at k = 2.
      BilinearForm *Mu_s = reconstruction->GetFluxMassForm();
      if (M_u)
      {
         auto Mu_dbfi = *M_u->GetDBFI();
         for (BilinearFormIntegrator *bfi : Mu_dbfi)
         {
            Mu_s->AddDomainIntegrator(bfi);
         }
         Mu_s->UseExternalIntegrators();
      }
      else if (!frozen_flux)
      {
         // A nonlinear flux form carrying linear integrators -- convdiff's
         // -nlu -- can have its mass reused exactly as it stands.
         for (NonlinearFormIntegrator *nlfi : *Mnl_u->GetDNFI())
         {
            auto *bfi = dynamic_cast<BilinearFormIntegrator*>(nlfi);
            MFEM_VERIFY(bfi, "Reconstruction needs a flux mass that assembles "
                        "as a bilinear form.");
            Mu_s->AddDomainIntegrator(bfi);
         }
         Mu_s->UseExternalIntegrators();
      }
      else
      {
         const MixedFluxFunction *flux_fun = NULL;
         if (Mnl)
         {
            for (BlockNonlinearFormIntegrator *bnlfi : Mnl->GetDomainIntegrators())
            {
               auto *mc = dynamic_cast<MixedConductionNLFIntegrator*>(bnlfi);
               if (mc) { flux_fun = &mc->GetFluxFunction(); break; }
            }
         }
         MFEM_VERIFY(flux_fun, "Reconstruction found no flux mass: neither a "
                     "bilinear form nor a MixedConductionNLFIntegrator to "
                     "linearise.");

         // Linearise about the computed potential. Mu_s owns the integrator;
         // the coefficient it points at is held by this DarcyForm, and is
         // replaced only after the old form has been destroyed above.
         Mu_nl_coeff.reset(new FrozenDualFluxCoefficient(*flux_fun, pc));
         const FiniteElement *fe_u = u.FESpace()->GetFE(0);
         if (fe_u->GetRangeType() == FiniteElement::VECTOR)
         {
            // An H(div) flux carrying a system has no integrator to lift onto.
            // FrozenDualFluxCoefficient is neq*dim square, which is what
            // VectorMassIntegrator wants for a scalar-range flux space of that
            // vdim; VectorFEMassIntegrator instead reads a dim-square
            // coefficient and returns an ndof-square block, so at neq > 1 the
            // element matrix comes out neq times too small in each direction
            // and the local solve runs off the end of it -- a segfault in
            // LUFactors::Solve, found by running this path rather than by
            // reading it.
            //
            // A block-diagonal wrapper is NOT the repair. The law couples the
            // fields -- D is neq-square -- so the flux mass block (i,j) is
            // the integral of D_ij(p) phi_a . phi_b, which is not block
            // diagonal and which no integrator in the tree assembles. What is
            // missing is a genuinely coupled vector-FE mass, and writing one
            // is a piece of work rather than a fix. The scalar-range flux
            // space has this at every neq today.
            MFEM_VERIFY(neq == 1,
                        "the rich reconstruction cannot lift a solution-"
                        "dependent flux law onto an H(div) flux space "
                        "carrying " << neq << " fields: the frozen law "
                        "couples them and VectorFEMassIntegrator assembles "
                        "one field's block. Use a discontinuous flux space, "
                        "or HDGPotentialPostprocessor, which is general in "
                        "vdim");
            Mu_s->AddDomainIntegrator(new VectorFEMassIntegrator(*Mu_nl_coeff));
         }
         else
         {
            Mu_s->AddDomainIntegrator(new VectorMassIntegrator(*Mu_nl_coeff));
         }
      }

      MixedBilinearForm *B_s = reconstruction->GetFluxDivForm();
      auto B_dbfi = *B->GetDBFI();
      for (BilinearFormIntegrator *bfi : B_dbfi)
      {
         B_s->AddDomainIntegrator(bfi);
      }
      B_s->UseExternalIntegrators();

      if (M_p)
      {
         BilinearForm *Mp_s = reconstruction->GetPotentialMassForm();
         auto Mp_dbfi = *M_p->GetDBFI();
         for (BilinearFormIntegrator *bfi : Mp_dbfi)
         {
            // A reaction-like term is not part of the local problem the
            // postprocessing solves. NPC eq (25) is a pure Neumann problem in
            // the enriched potential, driven by the total flux and closed by
            // the element average; it carries the diffusion and the
            // stabilisation and nothing else. Convective and hyperbolic terms
            // are kept because they *are* the local operator here; the rest
            // are omitted rather than moved to the right-hand side.
            if (IsOrWraps<ConvectionIntegrator>(bfi)
                || IsOrWraps<ConservativeConvectionIntegrator>(bfi))
            {
               Mp_s->AddDomainIntegrator(bfi);
            }
         }

         auto Mt_fbfi = *M_p->GetFBFI();
         for (BilinearFormIntegrator *fbfi : Mt_fbfi)
         {
            Mp_s->AddInteriorFaceIntegrator(fbfi);
         }

         Mp_s->UseExternalIntegrators();
      }
   }

   // The potential block on a non-linear form. There is nothing to lift onto
   // the enriched space -- a non-linear integrator has no element matrix -- so
   // the element term is taken as the Jacobian frozen at the computed
   // potential, which is the same treatment the flux mass gets above and
   // reduces to the M_p branch exactly when the integrators are bilinear.
   // The list is refreshed on every call because the freezing point moves.
   Mp_nl_lift.DeleteAll();
   if (!M_p && Mnl_p)
   {
      for (NonlinearFormIntegrator *nlfi : *Mnl_p->GetDNFI())
      {
         Mp_nl_lift.Append(nlfi);
      }

      // The face constraint comes from the hybridization, whose non-linear
      // potential integrator is the sum of exactly these. Its gradient is
      // taken at the computed potential and at a zero trace, which is only
      // the block the local problem needs if the trace cannot enter it --
      // the reconstruction is never given the trace solution. Refuse rather
      // than linearise about a state that is not the computed one.
      for (NonlinearFormIntegrator *nlfi : Mnl_p->GetInteriorFaceIntegrators())
      {
         auto *bfi = dynamic_cast<BilinearFormIntegrator*>(nlfi);
         MFEM_VERIFY(bfi, "Reconstruction needs a potential constraint whose "
                     "gradient does not depend on the trace; this one is "
                     "genuinely non-linear on the face.");
         auto *hdi = dynamic_cast<HDGDiffusionIntegrator*>(nlfi);
         MFEM_VERIFY(!hdi || !hdi->GetStabilization()
                     || hdi->GetStabilization()->IsConstant(),
                     "Reconstruction needs a potential constraint whose "
                     "gradient does not depend on the trace; this "
                     "stabilization is state dependent.");
      }
   }

   reconstruction->ReconstructFluxAndPot(*hybridization, pc, ut, u, p, tr,
                                         &Mp_nl_lift);
}

void DarcyForm::ReconstructFluxAndPot(const DarcyHybridization &h,
                                      const GridFunction &pc,
                                      const GridFunction &ut, GridFunction &u,
                                      GridFunction &p, GridFunction &tr,
                                      const Array<NonlinearFormIntegrator*> *Mp_nl)
const
{
   BilinearFormIntegrator *c_bfi = h.GetFluxConstraintIntegrator();
   BilinearFormIntegrator *c_bfi_p = h.GetPotConstraintIntegrator();
   NonlinearFormIntegrator *c_nlfi_p = h.GetPotConstraintNonlinearIntegrator();
   const bool pot_nl = (Mp_nl && Mp_nl->Size());
   FiniteElementSpace *fes_tr = tr.FESpace();
   const FiniteElementSpace *fes_pc = pc.FESpace();
   const FiniteElementSpace *fes_ut = ut.FESpace();
   Mesh *mesh = fes_u->GetMesh();
#ifdef MFEM_USE_MPI
   ParMesh *pmesh = dynamic_cast<ParMesh*>(mesh);
#endif //MFEM_USE_MPI
   const int dim = mesh->Dimension();

   // The number of fields, and the one thing that makes the local problem
   // below a system rather than a scalar one. Every block of the element
   // matrix is neq copies of a scalar block laid out equation outermost --
   // that is what VectorBlockDiagonalIntegrator builds, what the base
   // BilinearFormIntegrator::AssembleHDGFaceMatrix(int side, ...) slices, and
   // what GetElementVDofs()/GetFaceVDofs() give locally under either
   // Ordering. So the *slicing* below needs nothing new; only the sizes, the
   // right-hand side and the closure rows do.
   const int neq = fes_p->GetVDim();
   MFEM_VERIFY(fes_tr->GetVDim() == neq && fes_ut->GetVDim() == neq,
               "the enriched trace and the total flux must carry " << neq
               << " field(s), got " << fes_tr->GetVDim() << " and "
               << fes_ut->GetVDim());

   DenseMatrix elmat, Mu_z, Mp_z, B_z, Ct_f, Ct_fz, DEGH_f, D_fz, E_fz, G_fz, H_f,
               Mp_k, P_lift;
   DenseMatrixInverse inv;
   Vector rhs, rhs_p, shape_p, shape_pc;
   Vector shape_ut, shape_tr, ut_f, rhs_f;
   Vector p_lift, trfun_z;
   Array<int> faces, oris, vdofs_ut;

   // The element driver, per field. At neq == 1 this is bit for bit the
   // scalar DomainLFIntegrator(DivergenceGridFunctionCoefficient) it replaced
   // -- see the note in VectorDivergenceGridFunctionCoefficient::Eval(), which
   // was written to divide rather than scale by a reciprocal so that it is.
   VectorDivergenceGridFunctionCoefficient bp_coeff(&ut, neq);
   VectorDomainLFIntegrator bp(bp_coeff);

   Vector sol, sol_u, sol_p, sol_pc, sol_tr_f, elmat_e, rhs_e, mass_p, sum_pc;
   Array<int> vdofs_u, dofs_p, dofs_pc, vdofs_tr;

   u = 0.;
   p = 0.;
   tr = 0.;

   for (int z = 0; z < mesh->GetNE(); z++)
   {
      fes_u->GetElementVDofs(z, vdofs_u);
      fes_p->GetElementVDofs(z, dofs_p);
      const int ndof_u = vdofs_u.Size();
      const int ndof_p = dofs_p.Size();
      // The scalar count, which is what every shape function and every
      // per-field offset is stated in. ndof_p is neq of these.
      const int ndof_p_s = ndof_p / neq;

      switch (dim)
      {
         case 1:
            mesh->GetElementVertices(z, faces);
            break;
         case 2:
            mesh->GetElementEdges(z, faces, oris);
            break;
         case 3:
            mesh->GetElementFaces(z, faces, oris);
            break;
      }

      int ndof_tr = 0;
      for (int f : faces)
      {
         ndof_tr += neq * fes_tr->GetFaceElement(f)->GetDof();
      }

      const int elmat_w = ndof_u + ndof_p + ndof_tr;
      const int elmat_h = elmat_w;
      elmat.SetSize(elmat_h, elmat_w);
      elmat = 0.;

      rhs.SetSize(elmat_h);
      rhs = 0.;

      const FiniteElement *fe_p = fes_p->GetFE(z);
      ElementTransformation *Tr = mesh->GetElementTransformation(z);

      fes_pc->GetElementVDofs(z, dofs_pc);
      pc.GetSubVector(dofs_pc, sol_pc);
      const int ndof_pc_s = dofs_pc.Size() / neq;

      // The computed potential lifted onto the enriched space, the state that
      // a frozen non-linear block is taken at. The enriched space contains the
      // original one, so the embedding is exact and the lift adds nothing of
      // its own.
      if (pot_nl || c_nlfi_p)
      {
         // Project() is between the two SCALAR elements, so the lift matrix
         // is shared by every field and applied to each block in turn.
         fe_p->Project(*fes_pc->GetFE(z), *Tr, P_lift);
         p_lift.SetSize(ndof_p);
         for (int e = 0; e < neq; e++)
         {
            const Vector src(sol_pc.GetData() + e * ndof_pc_s, ndof_pc_s);
            Vector dst(p_lift.GetData() + e * ndof_p_s, ndof_p_s);
            P_lift.Mult(src, dst);
         }
      }

      M_u->ComputeElementMatrix(z, Mu_z);
      elmat.CopyMN(Mu_z, 0, 0);

      B->ComputeElementMatrix(z, B_z);
      elmat.CopyMN(B_z, ndof_u, 0);
      B_z.Neg();
      elmat.CopyMNt(B_z, 0, ndof_u);

      if (M_p)
      {
         M_p->ComputeElementMatrix(z, Mp_z);
         elmat.CopyMN(Mp_z, ndof_u, ndof_u);
      }
      else if (pot_nl)
      {
         Mp_z.SetSize(ndof_p);
         Mp_z = 0.;

         for (NonlinearFormIntegrator *nlfi : *Mp_nl)
         {
            // As in the M_p branch above: only the terms that are part of the
            // local operator are taken, and for the same reason.
            if (!IsOrWraps<ConvectionIntegrator>(nlfi)
                && !IsOrWraps<ConservativeConvectionIntegrator>(nlfi)
                && !IsOrWraps<HyperbolicFormIntegrator>(nlfi))
            {
               continue;
            }

            // For a bilinear integrator this is the element matrix itself, so
            // the branch coincides with the M_p one term by term.
            nlfi->AssembleElementGrad(*fe_p, *Tr, p_lift, Mp_k);
            Mp_z += Mp_k;
         }
         elmat.CopyMN(Mp_z, ndof_u, ndof_u);
      }

      // rhs

      rhs_p.MakeRef(rhs, ndof_u, ndof_p);

      // element term
      bp.AssembleRHSElementVect(*fe_p, *Tr, rhs_p);

      // face terms

      int off_tr = ndof_u + ndof_p;
      for (int f : faces)
      {
         const FiniteElement *fe_tr = fes_tr->GetFaceElement(f);
         const int ndof_tr_fs = fe_tr->GetDof();
         const int ndof_tr_f = neq * ndof_tr_fs;
         FaceElementTransformations *FTr = mesh->GetFaceElementTransformations(f);
#ifdef MFEM_USE_MPI
         if (FTr->Elem2No < 0 && pmesh && pmesh->FaceIsTrueInterior(f))
         {
            FTr = pmesh->GetSharedFaceTransformationsByLocalIndex(f);
         }
#endif //MFEM_USE_MPI

         // flux constraint
         const FiniteElement *fe_u1 = fes_u->GetFE(FTr->Elem1No);
         const FiniteElement *fe_u2 = (FTr->Elem2No >= 0)?(fes_u->GetFE(FTr->Elem2No)):
                                      (fe_u1);

         c_bfi->AssembleFaceMatrix(*fe_tr, *fe_u1, *fe_u2, *FTr, Ct_f);

         const int off_u = (FTr->Elem1No == z)?(0):(fe_u1->GetDof() * fes_u->GetVDim());
         Ct_fz.CopyMN(Ct_f, ndof_u, ndof_tr_f, off_u, 0);

         elmat.CopyMN(Ct_fz, 0, off_tr);
         elmat.CopyMNt(Ct_fz, off_tr, 0);

         //potential constraint
         if (c_bfi_p || c_nlfi_p)
         {
            const int side = (FTr->Elem1No == z)?(0):(1);
            if (c_bfi_p)
            {
               c_bfi_p->AssembleHDGFaceMatrix(side, *fe_tr, *fe_p, *FTr, DEGH_f);
            }
            else
            {
               // The constraint of a non-linear potential form. Its gradient
               // is the block the local problem needs, and it is laid out just
               // as AssembleHDGFaceMatrix() lays out its own when the whole
               // mask is asked for. The trace is passed as zero: the caller
               // has already refused every integrator whose gradient could
               // notice, because the reconstruction is not given the trace
               // solution to linearise about.
               constexpr int mask = NonlinearFormIntegrator::HDGFaceType::ELEM
                                    | NonlinearFormIntegrator::HDGFaceType::TRACE
                                    | NonlinearFormIntegrator::HDGFaceType::CONSTR
                                    | NonlinearFormIntegrator::HDGFaceType::FACE;
               trfun_z.SetSize(ndof_tr_f);
               trfun_z = 0.;
               c_nlfi_p->AssembleHDGFaceGrad(mask | side, *fe_tr, *fe_p, *FTr,
                                             trfun_z, p_lift, DEGH_f);
            }

            D_fz.CopyMN(DEGH_f, ndof_p, ndof_p, 0, 0);
            elmat.AddMatrix(D_fz, ndof_u, ndof_u);
            elmat.CopyMN(DEGH_f, ndof_p, ndof_tr_f, 0, ndof_p, ndof_u, off_tr);
            elmat.CopyMN(DEGH_f, ndof_tr_f, ndof_p, ndof_p, 0, off_tr, ndof_u);
            elmat.CopyMN(DEGH_f, ndof_tr_f, ndof_tr_f, ndof_p, ndof_p, off_tr, off_tr);
         }

         // rhs
         const FiniteElement *fe_ut = fes_ut->GetFaceElement(f);
         const int ndof_ut_f = fe_ut->GetDof();
         int order = fe_ut->GetOrder() + fe_tr->GetOrder();
         if (fe_tr->GetMapType() != FiniteElement::VALUE) { order += FTr->OrderW(); }
         const IntegrationRule &ir = IntRules.Get(fe_ut->GetGeomType(), order);

         // Both shapes are scalar; the fields share them and differ only in
         // which block of ut_f is contracted and which block of rhs_f is
         // written.
         shape_ut.SetSize(ndof_ut_f);
         shape_tr.SetSize(ndof_tr_fs);
         rhs_f.MakeRef(rhs, off_tr, ndof_tr_f);
         rhs_f = 0.;

         fes_ut->GetFaceVDofs(f, vdofs_ut);
         ut.GetSubVector(vdofs_ut, ut_f);
         MFEM_ASSERT(vdofs_ut.Size() == neq * ndof_ut_f,
                     "the total flux has " << vdofs_ut.Size()
                     << " face vdofs, expected " << neq * ndof_ut_f);

         MFEM_ASSERT(fe_ut->GetMapType() == FiniteElement::INTEGRAL,
                     "Non-integral face");

         for (int q = 0; q < ir.GetNPoints(); q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            fe_ut->CalcShape(ip, shape_ut);
            fe_tr->CalcShape(ip, shape_tr);

            for (int e = 0; e < neq; e++)
            {
               const Vector ut_fe(ut_f.GetData() + e * ndof_ut_f, ndof_ut_f);
               const real_t ut_q = shape_ut * ut_fe;

               // The arithmetic is written in exactly the order the scalar
               // version used -- multiply by ut_q, then divide by the face
               // weight -- because reassociating it moves the last bit and
               // this path has to reproduce the one-field answer exactly.
               real_t w = ip.weight * ut_q;
               if (fe_tr->GetMapType() == FiniteElement::INTEGRAL)
               {
                  FTr->SetIntPoint(&ip);
                  w /= FTr->Weight();
               }

               Vector rhs_fe(rhs_f.GetData() + e * ndof_tr_fs, ndof_tr_fs);
               rhs_fe.Add(w, shape_tr);
            }
         }

         if (FTr->Elem1No != z) { rhs_f.Neg(); }

         off_tr += ndof_tr_f;
      }

      // Close the local problem with the element average of the computed
      // potential -- the second part of NPC eq (25), and unconditional. The
      // problem is a pure Neumann one by construction: the total flux driving
      // it is in H(div), so the element's flux balance is already satisfied
      // and the potential is determined only up to a constant. There is
      // nothing to decide here, and so nothing to get wrong.
      //
      // **For a system there are neq constants, not one, and neq closure rows.**
      // The count and the placement are both forced, and neither is a choice:
      //
      //  * The count. The right-hand side of the potential rows is driven
      //    entirely by ut, and ut is in H(div) with vdim == neq, so field e's
      //    own element balance int_K div ut_e = int_dK ut_e.n holds
      //    identically for EACH e. Each field's potential rows are therefore
      //    rank deficient by exactly one and its potential is fixed only up to
      //    a constant. The null space is neq dimensional -- spanned by the
      //    per-field constants -- so neq conditions are needed and neq
      //    equations may be dropped.
      //  * The placement. Row ndof_u + e*ndof_p_s + i belongs to field e; a
      //    row of field e's block says nothing about field f's constant. Put
      //    all neq closures in one block and neq-1 constants stay free while
      //    that block is overdetermined -- a singular matrix, silently solved.
      //    So one closure lands in each field's own block, and its datum is
      //    that field's own element average, which is what makes each
      //    component superconverge for the same reason the scalar one does.
      //
      // Which row within the block is free; the first is what the scalar code
      // used, so e == 0 reproduces it exactly.
      {
         // adjust the element average of potential
         const FiniteElement *fe_pc = fes_pc->GetFE(z);
         const int order = fe_p->GetOrder() + Tr->OrderW();
         const IntegrationRule &ir = IntRules.Get(fe_p->GetGeomType(), order);
         Tr = mesh->GetElementTransformation(z);//just to be sure
         shape_p.SetSize(ndof_p_s);
         shape_pc.SetSize(ndof_pc_s);

         // The mass row is the same for every field -- it is built from the
         // scalar shape -- so only the datum is per field.
         sum_pc.SetSize(neq);
         sum_pc = 0.;
         mass_p.SetSize(ndof_p_s);
         mass_p = 0.;
         for (int q = 0; q < ir.GetNPoints(); q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            Tr->SetIntPoint(&ip);
            fe_pc->CalcShape(ip, shape_pc);
            const real_t w = ip.weight * Tr->Weight();
            for (int e = 0; e < neq; e++)
            {
               const Vector pc_e(sol_pc.GetData() + e * ndof_pc_s, ndof_pc_s);
               const real_t val = shape_pc * pc_e;
               sum_pc(e) += val * w;
            }

            fe_p->CalcShape(ip, shape_p);
            mass_p.Add(w, shape_p);
         }

         // replace one potential equation per field by that field's average
         for (int e = 0; e < neq; e++)
         {
            const int i_p = e * ndof_p_s;
            elmat.SetRow(ndof_u + i_p, 0.);
            for (int i = 0; i < ndof_p_s; i++)
            {
               elmat(ndof_u + i_p, ndof_u + i_p + i) = mass_p(i);
            }
            rhs(ndof_u + i_p) = sum_pc(e);
         }
      }

      // LU decompose
      inv.Factor(elmat);
      sol.SetSize(rhs.Size());
      inv.Mult(rhs, sol);

      // save the reconstructed flux and potential
      sol_u.MakeRef(sol, 0, ndof_u);
      u.SetSubVector(vdofs_u, sol_u);
      sol_p.MakeRef(sol, ndof_u, ndof_p);
      p.SetSubVector(dofs_p, sol_p);

      // save the traces
      off_tr = ndof_u + ndof_p;
      for (int f : faces)
      {
         fes_tr->GetFaceVDofs(f, vdofs_tr);
         sol_tr_f.MakeRef(sol, off_tr, vdofs_tr.Size());
         tr.SetSubVector(vdofs_tr, sol_tr_f);
         off_tr += vdofs_tr.Size();
      }
   }
}

void DarcyForm::EliminateTrueDofsInRHS(const Array<int> &tdofs_flux,
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
   EliminateVDofsInRHS(tdofs_flux, x, b);
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
         //In the case of the symmetrized system, the sign is opposite!
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
   else if (Mnl_u || Mnl)
   {
      b.GetBlock(0).SetSubVector(vdofs_flux, 0.);
   }
}

void DarcyForm::Mult(const Vector &x, Vector &y) const
{
   if (!block_op)
   {
      NonblockMult(x, y);
   }
   else
   {
      block_op->Mult(x, y);
   }
   if (Mnl)
   {
      if (bsym)
      {
         BlockVector ynl(toffsets);
         Mnl->Mult(x, ynl);
         ynl.GetBlock(1).Neg();
         y += ynl;
      }
      else
      {
         Mnl->AddMult(x, y);
      }
   }
}

void DarcyForm::NonblockMult(const Vector &x, Vector &y) const
{
   const BlockVector xb(const_cast<Vector&>(x), offsets);
   BlockVector yb(y, offsets);

   if (M_u) { M_u->Mult(xb.GetBlock(0), yb.GetBlock(0)); }
   else { yb.GetBlock(0) = 0.; }

   if (M_p)
   {
      M_p->Mult(xb.GetBlock(1), yb.GetBlock(1));
      if (bsym) { yb.GetBlock(1).Neg(); }
   }
   else { yb.GetBlock(1) = 0.; }

   if (B)
   {
      B->AddMult(xb.GetBlock(0), yb.GetBlock(1), (bsym)?(-1.):(+1.));
      B->AddMultTranspose(xb.GetBlock(1), yb.GetBlock(0), (bsym)?(-1.):(+1.));
   }
}

Operator &DarcyForm::GetGradient(const Vector &x) const
{
   const BlockVector bx(const_cast<Vector&>(x), toffsets);

   if (!Mnl && !Mnl_u && !Mnl_p)
   {
      MFEM_VERIFY(block_op, "DarcyForm must be finalized!");
      return *block_op;
   }

   if (Mnl_u || Mnl_p)
   {
      if (!block_grad)
      {
         block_grad.reset(new BlockOperator(toffsets));
      }

      if (opM_u.Ptr())
      {
         block_grad->SetDiagonalBlock(0, opM_u.Ptr());
      }
      else if (M_u)
      {
         block_grad->SetDiagonalBlock(0, M_u.get());
      }
      else if (Mnl_u)
      {
         block_grad->SetDiagonalBlock(0, &Mnl_u->GetGradient(bx.GetBlock(0)));
      }

      if (opM_p.Ptr())
      {
         block_grad->SetDiagonalBlock(1, opM_p.Ptr(), (bsym)?(-1.):(+1.));
      }
      else if (M_p)
      {
         block_grad->SetDiagonalBlock(1, M_p.get(), (bsym)?(-1.):(+1.));
      }
      else if (Mnl_p)
      {
         block_grad->SetDiagonalBlock(1, &Mnl_p->GetGradient(bx.GetBlock(1)),
                                      (bsym)?(-1.):(+1.));
      }

      if (B)
      {
         if (!opB.Ptr() || !opBt.Ptr())
         {
            opB.Reset(B.get(), false);
            ConstructBT(B.get());
         }
         block_grad->SetBlock(0, 1, opBt.Ptr(), (bsym)?(-1.):(+1.));
         block_grad->SetBlock(1, 0, opB.Ptr(), (bsym)?(-1.):(+1.));
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
      BlockVector ynl(p.toffsets);
      G.Mult(x, ynl);
      ynl.GetBlock(1).Neg();
      y += ynl;
   }
   else
   {
      G.AddMult(x, y);
   }
}

const BlockOperator &DarcyForm::Gradient::BlockMatrices() const
{
   if (block_grad) { return *block_grad.get(); }

   block_grad.reset(new BlockOperator(p.toffsets));

   const BlockOperator *bop = (p.block_grad)?(p.block_grad.get()):
                              (p.block_op.get());
   const BlockOperator *bgrad = static_cast<const BlockOperator*>(&G);

   for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
      {
         //off-diagonals of bgrad are expected to be zero
         if (i == j && !bop->IsZeroBlock(i,j) && !bgrad->IsZeroBlock(i,j))
         {
            const SparseMatrix *sop = dynamic_cast<const SparseMatrix*>(
                                         &(bop->GetBlock(i,j)));
            const SparseMatrix *sgrad = dynamic_cast<const SparseMatrix*>(
                                           &(bgrad->GetBlock(i,j)));

            MFEM_ASSERT(sop && sgrad, "Not a SparseMatrix!");

            smats[i][j].reset(mfem::Add(*sop, *sgrad));
            block_grad->SetBlock(i, j, smats[i][j].get(), bop->GetBlockCoef(i,j));
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
               c = (i != 0 && p.bsym)?(-1):(+1.);
            }
            // transpose operator is passed as is
            MFEM_ASSERT((i == 0 && j == 1) ||
                        dynamic_cast<const SparseMatrix*>(op), "Not a SparseMatrix!");
            block_grad->SetBlock(i, j, const_cast<Operator*>(op), c);
         }
      }

   return *block_grad;
}

void DarcyForm::Update()
{
   // Check for different size (e.g. assembled form on non-conforming space)
   // or different sequence number.
   const bool full_update = (fes_u->GetVSize() != offsets[1] - offsets[0]
                             || fes_p->GetVSize() != offsets[2] - offsets[1]
                             || sequence < fes_u->GetSequence());

   UpdateOffsetsAndSize();

   if (M_u) { M_u->Update(); }
   if (M_p) { M_p->Update(); }
   if (Mnl_u) { Mnl_u->Update(); }
   if (Mnl_p) { Mnl_p->Update(); }
   if (B) { B->Update(); }
   if (Mnl) { Mnl->Update(); }
   if (b_u) { b_u->Update(fes_u, block_b->GetBlock(0), 0); }
   if (b_p) { b_p->Update(fes_p, block_b->GetBlock(1), 0); }

   opBt.Clear();

   if (full_update)
   {
      reduction.reset();
      hybridization.reset();
      sequence = fes_u->GetSequence();
   }
   else
   {
      if (reduction) { reduction->Reset(); }
      if (hybridization) { hybridization->Reset(); }
   }
   reconstruction.reset();
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
      for (int f = 0; f < nfaces; f++)
      {
         tr = mesh -> GetInteriorFaceTransformations (f);
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
                                                        *test_fe2, *tr, elem_mat);
            elmat += elem_mat;
         }

         reduction->AssembleDivFaceMatrix(f, elmat);

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

      for (int be = 0; be < fes_p -> GetNBE(); be++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(be);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

         tr = mesh -> GetBdrFaceTransformations (be);
         if (tr != NULL)
         {
            const FiniteElement *trial_fe1 = fes_u->GetFE(tr->Elem1No);
            const FiniteElement *test_fe1 = fes_p->GetFE(tr->Elem1No);
            const int tr_ndof1 = trial_fe1->GetDof() * fes_u->GetVDim();
            const int te_ndof1 = test_fe1->GetDof() * fes_p->GetVDim();

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

            const int face = mesh->GetBdrElementFaceIndex(be);
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
      for (int f = 0; f < nfaces; f++)
      {
         tr = mesh -> GetInteriorFaceTransformations (f);
         if (tr == NULL) { continue; }

         const FiniteElement *fe1 = fes_p->GetFE(tr->Elem1No);
         const FiniteElement *fe2 = fes_p->GetFE(tr->Elem2No);

         interior_face_integs[0]->AssembleFaceMatrix(*fe1, *fe2, *tr, elmat);
         for (int i = 1; i < interior_face_integs.Size(); i++)
         {
            interior_face_integs[i]->AssembleFaceMatrix(*fe1, *fe2, *tr, elem_mat);
            elmat += elem_mat;
         }

         reduction->AssemblePotFaceMatrix(f, elmat);

#ifndef MFEM_DARCY_REDUCTION_ELIM_BCS
         fes_p->GetElementVDofs(tr->Elem1No, vdofs1);
         const int ndof1 = vdofs1.Size();
         elmat1.CopyMN(elmat, ndof1, ndof1, 0, 0);
         M_p->SpMat().AddSubMatrix(vdofs1, vdofs1, elmat1, skip_zeros);

         fes_p->GetElementVDofs(tr->Elem2No, vdofs2);
         const int ndof2 = vdofs2.Size();
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

      for (int be = 0; be < fes_p -> GetNBE(); be++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(be);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

         tr = mesh -> GetBdrFaceTransformations (be);
         if (tr != NULL)
         {
            const FiniteElement *fe1 = fes_p->GetFE(tr->Elem1No);
            const int ndof1 = fe1->GetDof() * fes_p->GetVDim();

            elmat.SetSize(ndof1);
            elmat = 0.;

            for (int i = 0; i < boundary_face_integs.Size(); i++)
            {
               if (boundary_face_integs_marker[i]
                   && (*boundary_face_integs_marker[i])[bdr_attr-1] == 0) { continue; }

               boundary_face_integs[i]->AssembleFaceMatrix(*fe1, *fe1, *tr, elem_mat);
               elmat += elem_mat;
            }

            const int face = mesh->GetBdrElementFaceIndex(be);
            reduction->AssemblePotFaceMatrix(face, elmat);

#ifndef MFEM_DARCY_REDUCTION_ELIM_BCS
            fes_p->GetElementVDofs(tr->Elem1No, vdofs1);
            M_p->SpMat().AddSubMatrix(vdofs1, vdofs1, elmat, skip_zeros);
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
         }
      }
   }
}

void DarcyForm::AssembleFluxMassBdrFaces(int skip_zeros)
{
   Array<BilinearFormIntegrator*> &boundary_face_integs = *M_u->GetBFBFI();
   const int num_boundary_face_integs = boundary_face_integs.Size();

   if (num_boundary_face_integs <= 0) { return; }

   Array<Array<int>*> &boundary_face_integs_marker = *M_u->GetBFBFI_Marker();
   Mesh *mesh = fes_u->GetMesh();
   DenseMatrix elmat;

   // Which boundary attributes need to be processed?
   Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                              mesh->bdr_attributes.Max() : 0);
   bdr_attr_marker = 0;
   for (int k = 0; k < num_boundary_face_integs; k++)
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

   for (int i = 0; i < fes_u->GetNBE(); i++)
   {
      const int bdr_attr = mesh->GetBdrAttribute(i);
      if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

      FaceElementTransformations *FTr = mesh->GetBdrFaceTransformations(i);
      if (!FTr) { continue; }

      const FiniteElement *fe1 = fes_u->GetFE(FTr->Elem1No);
      // The second element is a dummy on a boundary face, as elsewhere: it is
      // never used, but a null reference cannot be formed.
      const FiniteElement *fe2 = fe1;

      for (int k = 0; k < num_boundary_face_integs; k++)
      {
         if (boundary_face_integs_marker[k] &&
             (*boundary_face_integs_marker[k])[bdr_attr-1] == 0) { continue; }

         boundary_face_integs[k]->AssembleFaceMatrix(*fe1, *fe2, *FTr, elmat);
         MFEM_VERIFY(elmat.Height() == fe1->GetDof() * fes_u->GetVDim() &&
                     elmat.Width() == elmat.Height(),
                     "the flux mass boundary face integrator must return the "
                     "block of the adjacent element alone");
         hybridization->AssembleFluxMassMatrix(FTr->Elem1No, elmat);
#ifndef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
         Array<int> vdofs;
         fes_u->GetElementVDofs(FTr->Elem1No, vdofs);
         M_u->SpMat().AddSubMatrix(vdofs, vdofs, elmat, skip_zeros);
#endif //!MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
      }
   }
}

void DarcyForm::AssemblePotHDGFaces(int skip_zeros)
{
   Mesh *mesh = fes_p->GetMesh();
   DenseMatrix elmat1, elmat2;
   Array<int> vdofs1, vdofs2;

   if (hybridization->GetPotConstraintIntegrator())
   {
      int nfaces = mesh->GetNumFaces();
      for (int f = 0; f < nfaces; f++)
      {
         if (!mesh->FaceIsInterior(f)) { continue; }

         hybridization->ComputeAndAssemblePotFaceMatrix(f, elmat1, elmat2, vdofs1,
                                                        vdofs2);
#ifndef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
         M_p->SpMat().AddSubMatrix(vdofs1, vdofs1, elmat1, skip_zeros);
         M_p->SpMat().AddSubMatrix(vdofs2, vdofs2, elmat2, skip_zeros);
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
      }
   }

   const int num_boundary_face_integs =
      hybridization->NumBdrPotConstraintIntegrators();

   if (num_boundary_face_integs > 0)
   {
      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < num_boundary_face_integs; k++)
      {
         Array<int> *boundary_face_integs_marker =
            hybridization->GetBdrPotConstraintIntegratorMarker(k);
         if (boundary_face_integs_marker == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         Array<int> &bdr_marker = *boundary_face_integs_marker;
         MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                     "invalid boundary marker for boundary face integrator #"
                     << k << ", counting from zero");
         for (int i = 0; i < bdr_attr_marker.Size(); i++)
         {
            bdr_attr_marker[i] |= bdr_marker[i];
         }
      }

      for (int f = 0; f < fes_p->GetNBE(); f++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(f);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

         hybridization->ComputeAndAssemblePotBdrFaceMatrix(f, elmat1, vdofs1);
#ifndef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
         M_p->SpMat().AddSubMatrix(vdofs1, vdofs1, elmat1, skip_zeros);
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
      }
   }
}

void DarcyForm::AllocBlockOp(bool nonconforming)
{
   if (nonconforming) { UpdateTOffsetsAndSize(); }

   bool noblock = false;
#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
   noblock = noblock || reduction;
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   noblock = noblock || hybridization;
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS

   if (!noblock)
   {
      block_op.reset(new BlockOperator(toffsets));
   }
}

void DarcyForm::AllocRHS()
{
   if (block_b) { return; }
   block_b.reset(new BlockVector(offsets));
   *block_b = 0.;
}

const Operator *DarcyForm::ConstructBT(const MixedBilinearForm *B_) const
{
   if (B_->HasSpMat())
   {
      opBt.Reset(Transpose(B_->SpMat()));
   }
   else
   {
      opBt.Reset(new TransposeOperator(B_));
   }
   return opBt.Ptr();
}

const Operator* DarcyForm::ConstructBT(const OperatorHandle &B_) const
{
   if (B_.Type() == Operator::Type::MFEM_SPARSEMAT)
   {
      opBt.Reset(Transpose(*B_.As<SparseMatrix>()));
   }
#ifdef MFEM_USE_MPI
   else if (B_.Type() == Operator::Type::Hypre_ParCSR)
   {
      opBt.Reset(B_.As<HypreParMatrix>()->Transpose());
   }
#endif //MFEM_USE_MPI
   else
   {
      opBt.Reset(new TransposeOperator(B_.Ptr()));
   }
   return opBt.Ptr();
}

}
