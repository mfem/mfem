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

// Implementation of class LinearForm

#include "darcyform.hpp"

namespace mfem
{

DarcyForm::DarcyForm(FiniteElementSpace *fes_u_, FiniteElementSpace *fes_p_)
   : fes_u(fes_u_), fes_p(fes_p_)
{
   offsets.SetSize(3);
   offsets[0] = 0;
   offsets[1] = fes_u->GetVSize();
   offsets[2] = fes_p->GetVSize();
   offsets.PartialSum();

   width = height = offsets.Last();

   M_u = NULL;
   M_p = NULL;
   B = NULL;

   block_op = new BlockOperator(offsets);

   hybridization = NULL;
}

BilinearForm* DarcyForm::GetFluxMassForm()
{
   if (!M_u) { M_u = new BilinearForm(fes_u); }
   return M_u;
}

const BilinearForm* DarcyForm::GetFluxMassForm() const
{
   MFEM_ASSERT(M_u, "Flux mass form not allocated!");
   return M_u;
}

BilinearForm* DarcyForm::GetPotentialMassForm()
{
   if (!M_p) { M_p = new BilinearForm(fes_p); }
   return M_p;
}

const BilinearForm* DarcyForm::GetPotentialMassForm() const
{
   MFEM_ASSERT(M_p, "Potential mass form not allocated!");
   return M_p;
}

MixedBilinearForm* DarcyForm::GetFluxDivForm()
{
   if (!B) { B = new MixedBilinearForm(fes_u, fes_p); }
   return B;
}

const MixedBilinearForm* DarcyForm::GetFluxDivForm() const
{
   MFEM_ASSERT(B, "Flux div form not allocated!");
   return B;
}

void DarcyForm::SetAssemblyLevel(AssemblyLevel assembly_level)
{
   assembly = assembly_level;

   if (M_u) { M_u->SetAssemblyLevel(assembly); }
   if (M_p) { M_p->SetAssemblyLevel(assembly); }
   if (B) { B->SetAssemblyLevel(assembly); }
}

void DarcyForm::EnableHybridization(FiniteElementSpace *constr_space,
                                    BilinearFormIntegrator *constr_flux_integ,
                                    BilinearFormIntegrator *constr_pot_integ,
                                    const Array<int> &ess_flux_tdof_list)
{
   delete hybridization;
   if (assembly != AssemblyLevel::LEGACY)
   {
      delete constr_flux_integ;
      delete constr_pot_integ;
      hybridization = NULL;
      MFEM_WARNING("Hybridization not supported for this assembly level");
      return;
   }
   hybridization = new DarcyHybridization(fes_u, fes_p, constr_space);
   hybridization->SetConstraintIntegrators(constr_flux_integ, constr_pot_integ);
   hybridization->Init(ess_flux_tdof_list);
}

void DarcyForm::Assemble(int skip_zeros)
{
   if (M_u) { M_u->Assemble(skip_zeros); }
   if (M_p) { M_p->Assemble(skip_zeros); }
   if (B) { B->Assemble(skip_zeros); }
}

void DarcyForm::Finalize(int skip_zeros)
{
   if (M_u)
   {
      M_u->Finalize(skip_zeros);
      block_op->SetDiagonalBlock(0, M_u);
   }

   if (M_p)
   {
      M_p->Finalize(skip_zeros);
      block_op->SetDiagonalBlock(1, M_p);
   }

   if (B)
   {
      B->Finalize(skip_zeros);

      if (!pBt.Ptr()) { ConstructBT(B); }

      block_op->SetBlock(0, 1, pBt.Ptr(), -1.);
      block_op->SetBlock(1, 0, B, -1.);
   }

   if (hybridization)
   {
      hybridization->Finalize();
   }
}

void DarcyForm::FormLinearSystem(const Array<int> &ess_flux_tdof_list,
                                 BlockVector &x, BlockVector &b, OperatorHandle &A, Vector &X_, Vector &B_,
                                 int copy_interior)
{
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
   Array<int> ess_pot_tdof_list;//empty for discontinuous potentials

   if (M_u)
   {
      M_u->FormSystemMatrix(ess_flux_tdof_list, pM_u);
      block_op->SetDiagonalBlock(0, pM_u.Ptr());
   }

   if (M_p)
   {
      M_p->FormSystemMatrix(ess_pot_tdof_list, pM_p);
      block_op->SetDiagonalBlock(1, pM_p.Ptr());
   }

   if (B)
   {
      B->FormRectangularSystemMatrix(ess_flux_tdof_list, ess_pot_tdof_list, pB);

      ConstructBT(pB.Ptr());

      block_op->SetBlock(0, 1, pBt.Ptr(), -1.);
      block_op->SetBlock(1, 0, pB.Ptr(), -1.);
   }

   if (hybridization)
   {
      A.Reset(&hybridization->GetMatrix(), false);
   }
   else
   {
      A.Reset(block_op, false);
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
   if (B)
   {
      if (assembly != AssemblyLevel::LEGACY && assembly != AssemblyLevel::FULL)
      {
         //TODO
         MFEM_ABORT("");
      }
      else
      {
         Array<int> vdofs_flux_marker;
         FiniteElementSpace::ListToMarker(vdofs_flux, fes_u->GetTrueVSize(),
                                          vdofs_flux_marker);
         B->EliminateEssentialBCFromTrialDofs(vdofs_flux_marker, x.GetBlock(0),
                                              b.GetBlock(1));
      }
   }
   if (M_u)
   {
      M_u->EliminateVDofsInRHS(vdofs_flux, x.GetBlock(0), b.GetBlock(0));
   }
}

DarcyForm::~DarcyForm()
{
   if (M_u) { delete M_u; }
   if (M_p) { delete M_p; }
   if (B) { delete B; }

   delete block_op;

   delete hybridization;
}

const Operator* DarcyForm::ConstructBT(const MixedBilinearForm *B)
{
   pBt.Reset(Transpose(B->SpMat()));
   return pBt.Ptr();
}

const Operator* DarcyForm::ConstructBT(const Operator *opB)
{
   pBt.Reset(new TransposeOperator(opB));
   return pBt.Ptr();
}

DarcyHybridization::DarcyHybridization(FiniteElementSpace *fes_u_,
                                       FiniteElementSpace *fes_p_,
                                       FiniteElementSpace *fes_c_)
   : fes_u(fes_u_), fes_p(fes_p_), fes_c(fes_c_)
{
   c_bfi_u = NULL;
   c_bfi_p = NULL;

   H = NULL;
}

DarcyHybridization::~DarcyHybridization()
{
   delete c_bfi_u;
   delete c_bfi_p;

   delete H;
}

void DarcyHybridization::SetConstraintIntegrators(BilinearFormIntegrator
                                                  *c_flux_integ, BilinearFormIntegrator *c_pot_integ)
{
   delete c_bfi_u;
   c_bfi_u = c_flux_integ;
   delete c_bfi_p;
   c_bfi_p = c_pot_integ;
}

void DarcyHybridization::Init(const Array<int> &ess_flux_tdof_list)
{
}

void DarcyHybridization::Finalize()
{
}

void DarcyHybridization::ReduceRHS(const BlockVector &b, Vector &b_r) const
{
}

void DarcyHybridization::ComputeSolution(const BlockVector &b,
                                         const Vector &sol_r, BlockVector &sol) const
{
}

}
