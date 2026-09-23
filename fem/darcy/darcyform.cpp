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

LinearForm *DarcyForm::GetTraceRHS()
{
   if (!b_t)
   {
      MFEM_VERIFY(hybridization,
                  "GetTraceRHS() needs the constraint space, so it has to "
                  "follow EnableHybridization()");
      b_t.reset(new LinearForm(
                   const_cast<FiniteElementSpace*>(
                      hybridization->ConstraintFESpace())));
      // Registered HERE and not in Assemble(), so that the order the caller
      // reaches for the form in cannot matter. Doing it in Assemble() meant a
      // caller who filled the form afterwards -- which is the natural order
      // for a load that is computed rather than assembled from integrators --
      // silently got no load at all.
      hybridization->SetTraceRHS(b_t.get());
   }
   return b_t.get();
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

/** @brief Is every face integrator this NonlinearForm carries actually a
    BilinearFormIntegrator, i.e. linear in the state?

    A caller whose potential mass is nonlinear has to put the potential mass on
    a NonlinearForm, and the HDG face stabilization then goes on the SAME form
    -- `convdiff` installs the identical HDGDiffusionIntegrator on the linear
    form when the mass is linear and on the nonlinear one when it is not. So a
    constraint that is linear in every term routinely arrives as a
    NonlinearForm's face integrators, and taking it as nonlinear costs a great
    deal: the constraint blocks E, G, H and D are then rebuilt from the
    integrator once per element per Newton evaluation instead of once at
    assembly, the batched face kernel is unreachable (its gate is
    DarcyHybridization::PotFaceConstraintIntegrators(), which reads c_bfi_p),
    and the residual evaluates the integrator rather than applying an
    assembled block.

    Measured on `convdiff -p 1 -o 2 -dg -hb -nl -npc -nls 3`, 128x128, with a
    direct trace solve: ConstructGrad() was 0.36 s of a 2.33 s NPC step
    before, essentially all of it this constraint.

    It is all-or-nothing across the interior AND boundary lists on purpose.
    The two are stored separately (c_bfi_p / c_nlfi_p against the boundary
    lists), but the residual's boundary loop lives inside the c_nlfi_p branch,
    so a linear interior with a nonlinear boundary would set c_bfi_p and leave
    the boundary terms with nothing to evaluate them.

    **And it requires the problem to be nonlinear for some OTHER reason**,
    which is not an optimisation but a correctness condition.
    DarcyHybridization::IsNonlinear() reads c_nlfi_p, so moving a constraint
    off it can make a problem classify as LINEAR -- and for a problem whose
    only nonlinear-form content is this constraint, that is exactly what
    happens. `convdiff -p 1 -dg -hb -nl` is such a case: the sole occupant of
    the potential mass NonlinearForm is an HDGDiffusionIntegrator, so the
    problem really is linear and `-nl` merely forces it down the nonlinear
    path. Reclassifying it changes the whole solve route, and measured, it
    produced a NaN in GMRES on the first Newton step. Requiring another
    nonlinearity keeps the classification exactly where it was, so nothing
    that used to be nonlinear stops being so.

    There is nothing to gain in the reclassifying case anyway: with no other
    nonlinearity there is no Newton loop re-evaluating the constraint.

    **The dynamic_cast is not by itself a linearity test, and it does not have
    to be.** A BilinearFormIntegrator may override AssembleHDGFaceVector() and
    AssembleHDGFaceGrad() and carry state through them --
    HDGDiffusionIntegrator does exactly that, so that a solution dependent
    SetStabilization() can contribute its derivatives, and says so on
    AssembleHDGFaceGrad(). What makes the cast safe is that the route below
    reaches such an integrator only through AssembleHDGFaceMatrix(), which
    cannot see the state and REFUSES rather than guesses:
    `MFEM_VERIFY(!stab || stab->IsConstant(), "A state dependent stabilization
    makes the face term nonlinear")`. So a genuinely nonlinear one aborts
    loudly at assembly instead of quietly assembling a different operator. An
    integrator added later that carries state through the vector form WITHOUT
    such a guard would be admitted here wrongly; the guard belongs on the
    integrator, next to the state it hides. */
static bool FaceIntegratorsAreLinear(const std::unique_ptr<NonlinearForm>
                                     &Mnl_p,
                                     bool nonlinear_elsewhere)
{
   if (!nonlinear_elsewhere) { return false; }
   auto fnlfi = Mnl_p->GetInteriorFaceIntegrators();
   auto bfnlfi = Mnl_p->GetBdrFaceIntegrators();
   if (fnlfi.Size() == 0) { return false; }
   for (int i = 0; i < fnlfi.Size(); i++)
   {
      if (!dynamic_cast<BilinearFormIntegrator*>(fnlfi[i])) { return false; }
   }
   for (int i = 0; i < bfnlfi.Size(); i++)
   {
      if (!dynamic_cast<BilinearFormIntegrator*>(bfnlfi[i])) { return false; }
   }
   return true;
}

/** @brief Are ALL of @a Mnl_p's face integrators, interior and boundary,
    plain BilinearFormIntegrators?

    FaceIntegratorsAreLinear() asks the same question and then some: it also
    requires the problem to be nonlinear for some OTHER reason, because it
    decides whether to MOVE a constraint off the nonlinear route and
    IsNonlinear() reads c_nlfi_p. This variant is for the case where the
    linear route is already taken -- M_p exists, so c_bfi_p is being set
    whatever we do here -- and there that extra condition would refuse a
    perfectly bilinear integrator for a reason that does not apply. */
static bool AllFaceIntegratorsAreBilinear(const std::unique_ptr<NonlinearForm>
                                          &Mnl_p)
{
   for (NonlinearFormIntegrator *nlfi : Mnl_p->GetInteriorFaceIntegrators())
   { if (!dynamic_cast<BilinearFormIntegrator*>(nlfi)) { return false; } }
   for (NonlinearFormIntegrator *nlfi : Mnl_p->GetBdrFaceIntegrators())
   { if (!dynamic_cast<BilinearFormIntegrator*>(nlfi)) { return false; } }
   return true;
}

/// What to say when a face term on Mnl_p cannot be folded into c_bfi_p.
static const char *MnlPFaceRefusal()
{
   return "A face integrator on the potential mass NONLINEAR form is not read "
          "when a LINEAR potential mass form exists as well, and this one is "
          "not a BilinearFormIntegrator, so it cannot be folded into the "
          "linear constraint either. Put the HDG face constraint on the "
          "linear potential mass form (GetPotentialMassForm()), or move the "
          "whole potential mass onto the nonlinear form so the constraint is "
          "read from there. Carrying a linear and a nonlinear face constraint "
          "at once is what SetFaceConstraintMode(FaceConstraintMode::Live) "
          "does, and it needs EnableNPC(); see "
          "DarcyHybridization::SetPotConstraintNonlinearIntegrator().";
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

   // Is the problem nonlinear for a reason OTHER than the potential face
   // constraint? Required before that constraint may be moved to the linear
   // route, because IsNonlinear() reads c_nlfi_p -- see
   // FaceIntegratorsAreLinear().
   const bool nl_elsewhere =
      (Mnl_u && Mnl_u->GetDNFI() && Mnl_u->GetDNFI()->Size() > 0) ||
      (Mnl_p && Mnl_p->GetDNFI() && Mnl_p->GetDNFI()->Size() > 0) ||
      (Mnl != nullptr);

   // Automatically load the potential constraint operator from the face integrators
   if (M_p)
   {
      SumIntegrator *sbfi = NULL;
      auto fbfi = M_p->GetFBFI();
      if (fbfi->Size())
      {
         sbfi = new SumIntegrator(false);
         for (BilinearFormIntegrator *bfi : *fbfi)
         {
            sbfi->AddIntegrator(bfi);
         }
      }

      // **Mnl_p's face integrators are read HERE too, and used to be
      // dropped.** This branch tests M_p alone, so it shadows both `else if
      // (Mnl_p)` branches below, and a face constraint put on the nonlinear
      // potential mass form while a linear one existed reached nobody: no
      // warning, the stabilization simply absent. Measured by toggling one
      // such integrator with a loud coefficient (7.5, td 4.0) and comparing
      // bit for bit -- |r_tr| came back 2.9217004681959e+00 either way, to
      // every digit. It is worse than it sounds, because
      // GetPotentialMassForm() CONSTRUCTS the form on demand: merely asking
      // for M_p was enough to silence a constraint on Mnl_p.
      //
      // Folding them into the same SumIntegrator is the whole repair for the
      // bilinear case, and it is the case that occurs -- an HDG face
      // stabilization is a BilinearFormIntegrator, and so is a parametric
      // HDGConvectionUpwindedIntegrator, which derives from
      // DGTraceIntegrator. They are then assembled ONCE, which is what the
      // linear route is for.
      //
      // A genuinely nonlinear one is REFUSED rather than folded or dropped:
      // c_bfi_p and c_nlfi_p are one slot (each SetConstraintIntegrators()
      // overload resets the others), and making them coexist needs a linear
      // backup for E, G and H. Refusing names that, which a silent drop
      // did not.
      //
      // **Unless the caller asked for them to stay LIVE.** Folding freezes
      // the coefficient as well as the operator, and a face integrator whose
      // coefficient is a function of another field of a coupled system has to
      // be re-read rather than re-assembled. FaceConstraintMode::Live puts
      // them on the hybridization's nonlinear slot instead, where M_p's stay
      // frozen beside them; see DarcyForm::SetFaceConstraintMode().
      SumNLFIntegrator *live = NULL;
      if (Mnl_p && Mnl_p->GetInteriorFaceIntegrators().Size() > 0)
      {
         if (fc_mode == FaceConstraintMode::Live)
         {
            live = new SumNLFIntegrator(false);
            for (NonlinearFormIntegrator *nlfi : Mnl_p->GetInteriorFaceIntegrators())
            {
               live->AddIntegrator(nlfi);
            }
         }
         else
         {
            MFEM_VERIFY(AllFaceIntegratorsAreBilinear(Mnl_p), MnlPFaceRefusal());
            if (!sbfi) { sbfi = new SumIntegrator(false); }
            for (NonlinearFormIntegrator *nlfi : Mnl_p->GetInteriorFaceIntegrators())
            {
               sbfi->AddIntegrator(static_cast<BilinearFormIntegrator*>(nlfi));
            }
         }
      }
      hybridization->SetConstraintIntegrators(constr_flux_integ,
                                              (BilinearFormIntegrator*)sbfi);
      // AFTER SetConstraintIntegrators(), which clears this slot.
      if (live) { hybridization->SetPotConstraintNonlinearIntegrator(live); }
   }
   else if (Mnl_p && fc_mode == FaceConstraintMode::Frozen
            && FaceIntegratorsAreLinear(Mnl_p, nl_elsewhere))
   {
      // A linear constraint that merely happens to sit on a NonlinearForm.
      // Taking the c_bfi_p route assembles E, G, H and D once instead of once
      // per Newton evaluation, and is what makes the batched face kernel
      // reachable on a nonlinear problem. See FaceIntegratorsAreLinear().
      SumIntegrator *sbfi = new SumIntegrator(false);
      for (NonlinearFormIntegrator *nlfi : Mnl_p->GetInteriorFaceIntegrators())
      {
         sbfi->AddIntegrator(static_cast<BilinearFormIntegrator*>(nlfi));
      }
      hybridization->SetConstraintIntegrators(constr_flux_integ, sbfi);
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
      // REACHED BY NOTHING IN THIS TREE, and that was measured rather than
      // read off the conditions. Printing which branch above fires for each
      // of the 152 serial regression references: 17 fill c_nlfi_p, 5 reach
      // here with an EMPTY interior-face list, and none reaches here with a
      // MixedConductionNLFIntegrator in hand -- because every `-nld -hb`
      // configuration in convdiff and anisodiff also puts an
      // HDGDiffusionIntegrator on the potential mass form, which makes M_p or
      // Mnl_p non-null and takes one of the branches above.
      //
      // The consequence is a SILENT DROP: in 12 of the 20 `-nld -hb`
      // references the block nonlinear form carries one interior-face
      // integrator that nothing here ever reads, so its face stabilization
      // vanishes with no warning. The answer is not wrong in those cases --
      // the HDGDiffusionIntegrator on the mass form supplies a stabilization
      // of the same shape and the same `-td` -- so the references are what
      // they always were, and repairing it means letting c_bfi_p and c_nlfi
      // coexist, which is a change to four readers and a new linear backup
      // for E, G and H. Recorded here rather than repaired.
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

      // And Mnl_p's boundary face integrators, dropped by this branch for
      // the same reason as the interior ones -- see the interior chain
      // above. This half is additive rather than a single slot, since the
      // boundary constraints are a LIST, so folding costs nothing.
      //
      // Bilinear only, and that is not conservatism: ConstructGrad()'s
      // boundary loop lives INSIDE `if (c_nlfi_p)`, so a nonlinear boundary
      // constraint added while the interior route is linear would be
      // accepted here and then read by nobody -- the same silent drop one
      // level down. Refused with the same message.
      if (Mnl_p && Mnl_p->GetBdrFaceIntegrators().Size() > 0)
      {
         auto bfnlfi = Mnl_p->GetBdrFaceIntegrators();
         auto bfnlfi_marker = Mnl_p->GetBdrFaceIntegratorsMarkers();
         // Live mode sends them to the NONLINEAR boundary list, which is the
         // list ConstructGrad()'s and LocalNLOperator's boundary loops read --
         // both of which sit inside `if (c_nlfi_p)`, which live mode fills.
         // The interior choice above and this one therefore agree by
         // construction, which is what the frozen branch's own comment asks
         // for.
         const bool to_live = (fc_mode == FaceConstraintMode::Live);
         if (!to_live)
         {
            MFEM_VERIFY(AllFaceIntegratorsAreBilinear(Mnl_p), MnlPFaceRefusal());
         }
         for (int i = 0; i < bfnlfi.Size(); i++)
         {
            Array<int> *nlfi_marker = bfnlfi_marker[i];
            if (to_live)
            {
               NonlinearFormIntegrator *nlfi = bfnlfi[i];
               if (nlfi_marker)
               {
                  hybridization->AddBdrPotConstraintIntegrator(nlfi, *nlfi_marker);
               }
               else
               {
                  hybridization->AddBdrPotConstraintIntegrator(nlfi);
               }
               continue;
            }
            BilinearFormIntegrator *bfi =
               static_cast<BilinearFormIntegrator*>(bfnlfi[i]);
            if (nlfi_marker)
            {
               hybridization->AddBdrPotConstraintIntegrator(bfi, *nlfi_marker);
            }
            else
            {
               hybridization->AddBdrPotConstraintIntegrator(bfi);
            }
         }
      }
   }
   else if (Mnl_p)
   {
      // Matches the interior choice above -- FaceIntegratorsAreLinear() is
      // all-or-nothing across both lists, so the two never disagree.
      const bool linear = FaceIntegratorsAreLinear(Mnl_p, nl_elsewhere);
      auto bfnlfi = Mnl_p->GetBdrFaceIntegrators();
      auto bfnlfi_marker = Mnl_p->GetBdrFaceIntegratorsMarkers();
      hybridization->UseExternalBdrPotConstraintIntegrators();

      for (int i = 0; i < bfnlfi.Size(); i++)
      {
         NonlinearFormIntegrator *nlfi = bfnlfi[i];
         Array<int> *nlfi_marker = bfnlfi_marker[i];
         if (linear)
         {
            BilinearFormIntegrator *bfi =
               static_cast<BilinearFormIntegrator*>(nlfi);
            if (nlfi_marker)
            {
               hybridization->AddBdrPotConstraintIntegrator(bfi, *nlfi_marker);
            }
            else
            {
               hybridization->AddBdrPotConstraintIntegrator(bfi);
            }
         }
         else if (nlfi_marker)
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

/** @brief The flux components stated on the divergence and on the flux
    constraint have to agree.

    The two shape guards inside DarcyHybridization compare each block against
    what the SPACE owns, which catches an unrestricted integrator on a short
    flux space and catches a restriction of the wrong SIZE. It cannot catch a
    restriction of the right size to the wrong DIRECTIONS, because both blocks
    are then exactly the shape the space expects. This can, and it is the only
    place that sees both objects.

    Silent on a configuration that restricts neither. */
static void CheckRestrictedFluxAgreement(MixedBilinearForm *B,
                                         const BilinearFormIntegrator *c_bfi)
{
   const Array<int> *cc = GetRestrictedFluxComponents(c_bfi);
   Array<BilinearFormIntegrator*> *dbfi = B->GetDBFI();
   for (int k = 0; k < dbfi->Size(); k++)
   {
      const Array<int> *bc = GetRestrictedFluxComponents((*dbfi)[k]);
      if (!bc && !cc) { continue; }
      MFEM_VERIFY(bc && cc,
                  "one of the flux divergence and the flux constraint restricts "
                  "the flux to a subset of the Cartesian directions and the "
                  "other does not. Both take the same list, or neither: "
                  "RestrictedVectorDivergenceIntegrator on the divergence form "
                  "and RestrictedNormalTraceJumpIntegrator as the constraint.");
      MFEM_VERIFY(bc->Size() == cc->Size(),
                  "the flux divergence restricts to " << bc->Size()
                  << " component(s) and the flux constraint to " << cc->Size());
      for (int i = 0; i < bc->Size(); i++)
      {
         MFEM_VERIFY((*bc)[i] == (*cc)[i],
                     "the flux divergence and the flux constraint disagree on "
                     "flux component " << i << ": direction " << (*bc)[i]
                     << " against " << (*cc)[i] << ". They describe one space "
                     "and its component order is part of it.");
      }
   }
}

bool DarcyForm::CanThreadAssembly() const
{
#if defined(MFEM_USE_OPENMP) && defined(MFEM_DARCY_HYBRIDIZATION_ELIM_BCS)
   if (!hybridization || !hybridization->ThreadHostLoops() ||
       !hybridization->GetIntegratorsThreadSafe())
   {
      return false;
   }

   // A variable-order space builds its element FE lazily, through
   // FiniteElementSpace's `var_orders` cache, and two threads asking for an
   // order that is not there yet race on it. Every other reader in this loop
   // is const. Refused rather than locked: the loop is worth having on the
   // uniform-order case it was measured on, and a lock around GetFE() would
   // be paid on every element for a configuration that is not this one.
   if ((fes_u && fes_u->IsVariableOrder()) ||
       (fes_p && fes_p->IsVariableOrder()))
   {
      return false;
   }

   return true;
#else
   return false;
#endif
}

void DarcyForm::Assemble(int skip_zeros)
{
   // Checked here rather than where the load is assembled, because the load is
   // usually assembled by the CALLER -- convdiff's gform is Update()d onto a
   // block of its own right-hand side -- while every caller reaches this. It
   // is a check on the REGISTRATION, so it costs one dynamic_cast per boundary
   // face integrator per assembly and fires before any arithmetic happens.
   CheckRestrictedFluxLoad(b_u.get(), fes_u->GetVDim(),
                           fes_u->GetMesh()->SpaceDimension());

   if (M_u)
   {
      if (hybridization)
      {
         // The batched route does the same element loop in one kernel, from
         // the form's DOMAIN integrators only -- the boundary faces below are
         // routed separately either way. It refuses unless
         // AssemblyMode::Batched is asked for; see
         // AssembleFluxMassMatricesBatched().
         if (!hybridization->AssembleFluxMassMatricesBatched(M_u.get()))
         {
            const bool threaded = CanThreadAssembly();
            const int NE = fes_u->GetNE();

            // Element-wise integration
#ifdef MFEM_USE_OPENMP
            #pragma omp parallel if (threaded)
#endif
            {
               // Per thread, which is the whole of what the reentrant
               // ComputeElementMatrix() overload needs from its caller.
               DenseMatrix elmat, work;
               IsoparametricTransformation eltrans;
#ifdef MFEM_USE_OPENMP
               #pragma omp for schedule(static)
#endif
               for (int i = 0; i < NE; i++)
               {
                  M_u->ComputeElementMatrix(i, elmat, eltrans, work);
#ifndef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
                  M_u->AssembleElementMatrix(i, elmat, skip_zeros);
#endif //!MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
                  hybridization->AssembleFluxMassMatrix(i, elmat);
               }
            }
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
         // **The divergence and the constraint have to name the SAME
         // directions, and shapes alone cannot say so.** Two lists of equal
         // length that differ in their entries -- {0,1} against {1,2} of
         // three -- pass every size check in the tree and assemble a
         // divergence of one pair of directions against a normal trace of
         // another. The result is a consistent-looking operator for a problem
         // nobody posed. One comparison per assembly.
         CheckRestrictedFluxAgreement(B.get(),
                                      hybridization->GetFluxConstraintIntegrator());

         // As for the two masses: the domain integrators only. B's FACE
         // integrators are a marker for the constraint on the hybridized path
         // and are never evaluated here, so there is nothing for the kernel to
         // miss; see EnableHybridization().
         if (!hybridization->AssembleDivMatricesBatched(B.get()))
         {
            const bool threaded = CanThreadAssembly();
            const int NE = fes_u->GetNE();

            // Element-wise integration
#ifdef MFEM_USE_OPENMP
            #pragma omp parallel if (threaded)
#endif
            {
               DenseMatrix elmat, work;
               IsoparametricTransformation eltrans;
#ifdef MFEM_USE_OPENMP
               #pragma omp for schedule(static)
#endif
               for (int i = 0; i < NE; i++)
               {
                  B->ComputeElementMatrix(i, elmat, eltrans, work);
#ifndef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
                  B->AssembleElementMatrix(i, elmat, skip_zeros);
#endif //!MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
                  hybridization->AssembleDivMatrix(i, elmat);
               }
            }
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
         // As for the flux mass: the domain integrators only, the faces below
         // routed separately. See AssemblePotMassMatricesBatched().
         if (!hybridization->AssemblePotMassMatricesBatched(M_p.get()))
         {
            const bool threaded = CanThreadAssembly();
            const int NE = fes_p->GetNE();

            // Element-wise integration
#ifdef MFEM_USE_OPENMP
            #pragma omp parallel if (threaded)
#endif
            {
               DenseMatrix elmat, work;
               IsoparametricTransformation eltrans;
#ifdef MFEM_USE_OPENMP
               #pragma omp for schedule(static)
#endif
               for (int i = 0; i < NE; i++)
               {
                  M_p->ComputeElementMatrix(i, elmat, eltrans, work);
#ifndef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
                  M_p->AssembleElementMatrix(i, elmat, skip_zeros);
#endif //!MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
                  hybridization->AssemblePotMassMatrix(i, elmat);
               }
            }
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

      /* The potential MASS is on this NonlinearForm -- so there is no M_p and
         the block above did not run -- but its FACE constraint may be linear,
         in which case EnableHybridization() routed it to c_bfi_p. This pass is
         what fills E, G, H and D from it. Without it they stay zero, the trace
         system is singular, and GMRES returns beta = -nan on the first Newton
         step. GetPotConstraintIntegrator() is exactly "the constraint took the
         linear route", so it is the right test and not a proxy for one. */
      if (hybridization && hybridization->GetPotConstraintIntegrator())
      {
         AssemblePotHDGFaces(skip_zeros);
      }
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

   // The skeleton load owns its storage and is not part of block_b, so it is
   // assembled here and handed to the hybridization, which is what makes both
   // routes carry it without the caller wiring anything.
   if (b_t) { b_t->Assemble(); }
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

namespace
{

/** @brief Per-thread scratch for the element body of
    DarcyForm::ReconstructFluxAndPot().

    Everything that body used to declare above its element loop, plus the two
    pieces the reentrant ComputeElementMatrix() overloads take from their
    caller and the three transformations that stand in for the Mesh's shared
    ones. Hoisted rather than declared per element for the reason this branch
    has measured twice: a per-element DenseMatrix is a malloc, and the element
    loops here were 90,949 of them before the allocation rounds.

    It is a FILE-LOCAL type, held by value on each thread's stack, so it is
    nobody's layout and no header moves for it -- the trap this branch has
    paid for eight times. */
struct ReconstructWorkspace
{
   DenseMatrix elmat, Mu_z, Mp_z, B_z, Ct_f, Ct_fz, DEGH_f, D_fz, Mp_k, P_lift;
   /// The accumulator the reentrant ComputeElementMatrix() needs when a form
   /// carries more than one domain integrator; untouched when it carries one.
   DenseMatrix ce_work;
   DenseMatrixInverse inv;
   Vector rhs, rhs_p, shape_p, shape_pc;
   Vector shape_ut, shape_tr, ut_f, rhs_f;
   Vector p_lift, trfun_z;
   Vector sol_pc, mass_p, sum_pc;
   Array<int> vdofs_u, dofs_p, dofs_pc, vdofs_ut;

   /// In place of the Mesh's shared element transformation, which is one
   /// object per Mesh and so a race the moment two elements are live.
   IsoparametricTransformation eltrans;
   /// In place of the Mesh's shared face transformation and its two sides.
   FaceElementTransformations facetrans;
   IsoparametricTransformation f1, f2;

   /// The element driver, one instance per thread; see its construction.
   VectorDomainLFIntegrator *bp = nullptr;

   /** @brief In place of FiniteElementSpace's own `mutable DofTransformation
       DoFTrans`, which the one-argument GetElementVDofs() writes into.

       Nothing here READS the returned transformation -- the dof array itself
       is transformation independent -- so this looks like a write nobody
       cares about. It is not: SetFaceOrientations() is `Fo_ = Fo`, an
       Array<int> copy assignment, so two threads gathering dofs from one
       space at once can be allocating and freeing that member's buffer
       simultaneously. It fires only where DoFTransArray[geom] is non-null,
       i.e. an H(curl) or H(div) space whose dofs depend on face orientation,
       which is exactly the RT flux this routine also serves. Taking the
       explicit overload removes the question rather than arguing it is
       benign, and costs one object per thread. */
   DofTransformation doftrans;

   /// The scatter's own, so that it allocates nothing per element either.
   Array<int> s_vdofs_u, s_dofs_p, s_vdofs_tr;
   Vector s_sol_u, s_sol_p, s_sol_tr_f;
};

} // anonymous namespace

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
   const int NE = mesh->GetNE();

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

   // The element driver, per field. At neq == 1 this is bit for bit the
   // scalar DomainLFIntegrator(DivergenceGridFunctionCoefficient) it replaced
   // -- see the note in VectorDivergenceGridFunctionCoefficient::Eval(), which
   // was written to divide rather than scale by a reciprocal so that it is.
   //
   // The COEFFICIENT is shared across threads and the INTEGRATOR is not.
   // Eval() above reads the grid function and writes only its output and its
   // own locals, so one instance serves every thread; VectorDomainLFIntegrator
   // holds `shape` and `Qvec`, and although both sit behind
   // `#ifndef MFEM_THREAD_SAFE` upstream, giving each thread its own instance
   // costs nothing and takes this integrator out of the promise the caller
   // makes with SetIntegratorsThreadSafe() altogether. It is ours, not the
   // caller's, so it should not be in that promise.
   VectorDivergenceGridFunctionCoefficient bp_coeff(&ut, neq);

   u = 0.;
   p = 0.;
   tr = 0.;

   // The element's faces, by whatever they are called in this dimension.
   auto element_faces = [&](int z, Array<int> &faces, Array<int> &oris)
   {
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
   };

   // ------------------------------------------------------------------
   // One element's local problem: assembled, closed and solved into @a sol.
   // It reads the computed solution and the mesh and writes nothing outside
   // @a w and @a sol, which is what makes the loop over it parallelisable.
   // ------------------------------------------------------------------
   auto solve_element = [&](int z, const Array<int> &faces,
                            ReconstructWorkspace &w, Vector &sol)
   {
      fes_u->GetElementVDofs(z, w.vdofs_u, w.doftrans);
      fes_p->GetElementVDofs(z, w.dofs_p, w.doftrans);
      const int ndof_u = w.vdofs_u.Size();
      const int ndof_p = w.dofs_p.Size();
      // The scalar count, which is what every shape function and every
      // per-field offset is stated in. ndof_p is neq of these.
      const int ndof_p_s = ndof_p / neq;

      int ndof_tr = 0;
      for (int f : faces)
      {
         ndof_tr += neq * fes_tr->GetFaceElement(f)->GetDof();
      }

      const int elmat_w = ndof_u + ndof_p + ndof_tr;
      const int elmat_h = elmat_w;
      w.elmat.SetSize(elmat_h, elmat_w);
      w.elmat = 0.;

      w.rhs.SetSize(elmat_h);
      w.rhs = 0.;

      const FiniteElement *fe_p = fes_p->GetFE(z);
      // The caller's own transformation rather than the Mesh's shared one.
      // The three ComputeElementMatrix() calls below refill it with this very
      // element's geometry, exactly as the one-argument overloads refilled the
      // shared one that this routine used to hold a pointer to -- so the
      // aliasing, and the answer, are what they always were.
      mesh->GetElementTransformation(z, &w.eltrans);
      ElementTransformation *Tr = &w.eltrans;

      fes_pc->GetElementVDofs(z, w.dofs_pc, w.doftrans);
      pc.GetSubVector(w.dofs_pc, w.sol_pc);
      const int ndof_pc_s = w.dofs_pc.Size() / neq;

      // The computed potential lifted onto the enriched space, the state that
      // a frozen non-linear block is taken at. The enriched space contains the
      // original one, so the embedding is exact and the lift adds nothing of
      // its own.
      if (pot_nl || c_nlfi_p)
      {
         // Project() is between the two SCALAR elements, so the lift matrix
         // is shared by every field and applied to each block in turn.
         fe_p->Project(*fes_pc->GetFE(z), *Tr, w.P_lift);
         w.p_lift.SetSize(ndof_p);
         for (int e = 0; e < neq; e++)
         {
            const Vector src(w.sol_pc.GetData() + e * ndof_pc_s, ndof_pc_s);
            Vector dst(w.p_lift.GetData() + e * ndof_p_s, ndof_p_s);
            w.P_lift.Mult(src, dst);
         }
      }

      M_u->ComputeElementMatrix(z, w.Mu_z, w.eltrans, w.ce_work);
      w.elmat.CopyMN(w.Mu_z, 0, 0);

      B->ComputeElementMatrix(z, w.B_z, w.eltrans, w.ce_work);
      w.elmat.CopyMN(w.B_z, ndof_u, 0);
      w.B_z.Neg();
      w.elmat.CopyMNt(w.B_z, 0, ndof_u);

      if (M_p)
      {
         M_p->ComputeElementMatrix(z, w.Mp_z, w.eltrans, w.ce_work);
         w.elmat.CopyMN(w.Mp_z, ndof_u, ndof_u);
      }
      else if (pot_nl)
      {
         w.Mp_z.SetSize(ndof_p);
         w.Mp_z = 0.;

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
            nlfi->AssembleElementGrad(*fe_p, *Tr, w.p_lift, w.Mp_k);
            w.Mp_z += w.Mp_k;
         }
         w.elmat.CopyMN(w.Mp_z, ndof_u, ndof_u);
      }

      // rhs

      w.rhs_p.MakeRef(w.rhs, ndof_u, ndof_p);

      // element term
      w.bp->AssembleRHSElementVect(*fe_p, *Tr, w.rhs_p);

      // face terms

      int off_tr = ndof_u + ndof_p;
      for (int f : faces)
      {
         const FiniteElement *fe_tr = fes_tr->GetFaceElement(f);
         const int ndof_tr_fs = fe_tr->GetDof();
         const int ndof_tr_f = neq * ndof_tr_fs;
         // Again the caller's own rather than the Mesh's shared cache, which
         // is one FaceElementTransformations and two IsoparametricTransforms
         // for the whole Mesh.
         mesh->GetFaceElementTransformations(f, w.facetrans, w.f1, w.f2);
         FaceElementTransformations *FTr = &w.facetrans;
#ifdef MFEM_USE_MPI
         if (FTr->Elem2No < 0 && pmesh && pmesh->FaceIsTrueInterior(f))
         {
            pmesh->GetSharedFaceTransformationsByLocalIndex(f, w.facetrans,
                                                            w.f1, w.f2);
         }
#endif //MFEM_USE_MPI

         // flux constraint
         const FiniteElement *fe_u1 = fes_u->GetFE(FTr->Elem1No);
         const FiniteElement *fe_u2 = (FTr->Elem2No >= 0)?(fes_u->GetFE(FTr->Elem2No)):
                                      (fe_u1);

         c_bfi->AssembleFaceMatrix(*fe_tr, *fe_u1, *fe_u2, *FTr, w.Ct_f);

         const int off_u = (FTr->Elem1No == z)?(0):(fe_u1->GetDof() * fes_u->GetVDim());
         w.Ct_fz.CopyMN(w.Ct_f, ndof_u, ndof_tr_f, off_u, 0);

         w.elmat.CopyMN(w.Ct_fz, 0, off_tr);
         w.elmat.CopyMNt(w.Ct_fz, off_tr, 0);

         //potential constraint
         if (c_bfi_p || c_nlfi_p)
         {
            const int side = (FTr->Elem1No == z)?(0):(1);
            if (c_bfi_p)
            {
               c_bfi_p->AssembleHDGFaceMatrix(side, *fe_tr, *fe_p, *FTr, w.DEGH_f);
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
               w.trfun_z.SetSize(ndof_tr_f);
               w.trfun_z = 0.;
               c_nlfi_p->AssembleHDGFaceGrad(mask | side, *fe_tr, *fe_p, *FTr,
                                             w.trfun_z, w.p_lift, w.DEGH_f);
            }

            w.D_fz.CopyMN(w.DEGH_f, ndof_p, ndof_p, 0, 0);
            w.elmat.AddMatrix(w.D_fz, ndof_u, ndof_u);
            w.elmat.CopyMN(w.DEGH_f, ndof_p, ndof_tr_f, 0, ndof_p, ndof_u, off_tr);
            w.elmat.CopyMN(w.DEGH_f, ndof_tr_f, ndof_p, ndof_p, 0, off_tr, ndof_u);
            w.elmat.CopyMN(w.DEGH_f, ndof_tr_f, ndof_tr_f, ndof_p, ndof_p, off_tr,
                           off_tr);
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
         w.shape_ut.SetSize(ndof_ut_f);
         w.shape_tr.SetSize(ndof_tr_fs);
         w.rhs_f.MakeRef(w.rhs, off_tr, ndof_tr_f);
         w.rhs_f = 0.;

         fes_ut->GetFaceVDofs(f, w.vdofs_ut);
         ut.GetSubVector(w.vdofs_ut, w.ut_f);
         MFEM_ASSERT(w.vdofs_ut.Size() == neq * ndof_ut_f,
                     "the total flux has " << w.vdofs_ut.Size()
                     << " face vdofs, expected " << neq * ndof_ut_f);

         MFEM_ASSERT(fe_ut->GetMapType() == FiniteElement::INTEGRAL,
                     "Non-integral face");

         for (int q = 0; q < ir.GetNPoints(); q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            fe_ut->CalcShape(ip, w.shape_ut);
            fe_tr->CalcShape(ip, w.shape_tr);

            for (int e = 0; e < neq; e++)
            {
               const Vector ut_fe(w.ut_f.GetData() + e * ndof_ut_f, ndof_ut_f);
               const real_t ut_q = w.shape_ut * ut_fe;

               // The arithmetic is written in exactly the order the scalar
               // version used -- multiply by ut_q, then divide by the face
               // weight -- because reassociating it moves the last bit and
               // this path has to reproduce the one-field answer exactly.
               real_t w_q = ip.weight * ut_q;
               if (fe_tr->GetMapType() == FiniteElement::INTEGRAL)
               {
                  FTr->SetIntPoint(&ip);
                  w_q /= FTr->Weight();
               }

               Vector rhs_fe(w.rhs_f.GetData() + e * ndof_tr_fs, ndof_tr_fs);
               rhs_fe.Add(w_q, w.shape_tr);
            }
         }

         if (FTr->Elem1No != z) { w.rhs_f.Neg(); }

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
      //
      // **The closure is unconditional, and it is only CORRECT while the
      // lifted local operator keeps the per-field constant in its null space.**
      // If that operator is full rank, replacing a row does not remove a
      // redundancy -- it discards a real equation and adds one already implied.
      // The matrix stays invertible, so there is no abort and no warning, only
      // a wrong postprocessed field.
      //
      // Not every term lifted here keeps it. Measured, ||M.1||_inf / ||M||_inf
      // on one element's potential block at order 2:
      //
      //     DiffusionIntegrator (control)            9.5e-17   keeps it
      //     ConvectionIntegrator, (b.grad u, v)      1.7e-16   keeps it
      //     ConservativeConvectionIntegrator         1.03      DOES NOT
      //     HyperbolicFormIntegrator, neq = 1        1.03      DOES NOT
      //     HyperbolicFormIntegrator, Euler, neq = 4 0.94      DOES NOT
      //
      // The split is the CONSERVATIVE (divergence) form, not hyperbolicity:
      // (b.grad u, v) differentiates the state so a constant dies, while
      // -(u, b.grad v) and (F(u), grad v) differentiate the test function and
      // leave int_dK (J.n) phi_i. ConservativeConvectionIntegrator is bilinear
      // and convdiff -p 2 puts it on the potential mass form, so the
      // configuration is reachable at neq == 1 and not only for a system.
      //
      // **Whether anything actually breaks is NOT established**, and the two
      // reasons matter more than the table. First, the null vector of this
      // local problem is (u = 0, p = c, tr = c) -- the trace is an unknown here
      // and each row's cancellation pairs a potential term with a trace one, so
      // the element block alone is the wrong object to test. Second, an
      // end-to-end sweep of convdiff -p 2 -rec against the convection-free -p 1
      // gave the same rates either way, and both gave k+1 where the unit tests
      // give k+2, because convdiff's error quadrature is 2*order+1 against an
      // enriched field of order order+1: that study measures the miniapp's
      // rule, not this closure. The decisive experiment is a manufactured
      // problem with a constant divergence-free b at 2*order+6 quadrature.
      //
      // HDGPotentialPostprocessor is immune to all of it, structurally rather
      // than by luck: its matrix is AddMult_a_AAt(w, dshape_s, A) and nothing
      // else -- the Neumann stiffness, whatever the PDE -- with the physics
      // entering only through its right-hand side. No convective term, no
      // trace, no coupling between fields.
      {
         // adjust the element average of potential
         const FiniteElement *fe_pc = fes_pc->GetFE(z);
         const int order = fe_p->GetOrder() + Tr->OrderW();
         const IntegrationRule &ir = IntRules.Get(fe_p->GetGeomType(), order);
         // Refill after the integrators above have walked it to their own last
         // quadrature point; the object is the caller's, the call is the same.
         mesh->GetElementTransformation(z, &w.eltrans);
         w.shape_p.SetSize(ndof_p_s);
         w.shape_pc.SetSize(ndof_pc_s);

         // The mass row is the same for every field -- it is built from the
         // scalar shape -- so only the datum is per field.
         w.sum_pc.SetSize(neq);
         w.sum_pc = 0.;
         w.mass_p.SetSize(ndof_p_s);
         w.mass_p = 0.;
         for (int q = 0; q < ir.GetNPoints(); q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            Tr->SetIntPoint(&ip);
            fe_pc->CalcShape(ip, w.shape_pc);
            const real_t w_q = ip.weight * Tr->Weight();
            for (int e = 0; e < neq; e++)
            {
               const Vector pc_e(w.sol_pc.GetData() + e * ndof_pc_s, ndof_pc_s);
               const real_t val = w.shape_pc * pc_e;
               w.sum_pc(e) += val * w_q;
            }

            fe_p->CalcShape(ip, w.shape_p);
            w.mass_p.Add(w_q, w.shape_p);
         }

         // replace one potential equation per field by that field's average
         for (int e = 0; e < neq; e++)
         {
            const int i_p = e * ndof_p_s;
            w.elmat.SetRow(ndof_u + i_p, 0.);
            for (int i = 0; i < ndof_p_s; i++)
            {
               w.elmat(ndof_u + i_p, ndof_u + i_p + i) = w.mass_p(i);
            }
            w.rhs(ndof_u + i_p) = w.sum_pc(e);
         }
      }

      // LU decompose
      w.inv.Factor(w.elmat);
      sol.SetSize(w.rhs.Size());
      w.inv.Mult(w.rhs, sol);
   };

   // ------------------------------------------------------------------
   // One element's share of the answer, written out. SEPARATE from the solve
   // above, and that separation is the whole reason this loop can be threaded
   // at all -- see the note below.
   // ------------------------------------------------------------------
   auto scatter_element = [&](int z, const Array<int> &faces,
                              Vector &sol, ReconstructWorkspace &w)
   {
      fes_u->GetElementVDofs(z, w.s_vdofs_u);
      fes_p->GetElementVDofs(z, w.s_dofs_p);

      // save the reconstructed flux and potential
      w.s_sol_u.MakeRef(sol, 0, w.s_vdofs_u.Size());
      u.SetSubVector(w.s_vdofs_u, w.s_sol_u);
      w.s_sol_p.MakeRef(sol, w.s_vdofs_u.Size(), w.s_dofs_p.Size());
      p.SetSubVector(w.s_dofs_p, w.s_sol_p);

      // save the traces
      int off_tr = w.s_vdofs_u.Size() + w.s_dofs_p.Size();
      for (int f : faces)
      {
         fes_tr->GetFaceVDofs(f, w.s_vdofs_tr);
         w.s_sol_tr_f.MakeRef(sol, off_tr, w.s_vdofs_tr.Size());
         tr.SetSubVector(w.s_vdofs_tr, w.s_sol_tr_f);
         off_tr += w.s_vdofs_tr.Size();
      }
   };

   /* **Threading, and why the loop is split in two rather than simply run in
      parallel.**

      The solve above is element local: it reads the mesh, the computed
      solution and the total flux, and writes only its own workspace. The
      SCATTER below is not, and the reason is sharper than "two elements might
      touch the same dof".

      **The enriched trace is whichever element visited the face last, and
      nothing said so before this comment.** Each element solves a local
      problem whose trace is free on EVERY face, so the two elements either
      side of an interior face produce two different values for that face's
      trace dofs -- and `tr.SetSubVector()` ASSIGNS. Reversing the element
      order leaves the reconstructed flux and potential identical to 17 digits
      (|u*| = 7.9321522926801267, |p*| = 5.7077841841727244, both ways) and
      moves the trace: |tr*| = 5.4581137616313953 forwards, 5.3635341954733962
      backwards.

      The FLUX is face-shared too when its enriched space is conforming -- an
      RT or broken-RT flux -- so formally the same reaches it. **Measured, it
      does not, to any size that matters**: racing the scatter on the RT
      configuration of tests/unit/fem/test_darcy_threaded_assembly.cpp moves
      `tr` by 1.0e+00 to 8.5e+07 and `u` by 1.8e-15 to 6.0e-08. The two local
      problems disagree about the trace and agree about the flux, whose normal
      component both of them pin to the same total flux on the face. The
      potential's enriched space is discontinuous, so it is exactly order
      independent.

      So there is no colouring that preserves the answer: a colouring changes
      which element writes last, which changes `tr` by 2% of its own norm.
      **Compute in parallel, then replay the writes serially in element
      order** -- the only arrangement that is bit for bit the serial one,
      which is the standard this branch holds its threaded loops to and the
      standard `AssemblyMode::Threaded` states in its own doxygen. The price
      is a buffer of local solutions, bounded rather than proportional to the
      mesh; see the chunk loop below.

      **What the caller is promising.** The gate is the assembly mode plus
      SetIntegratorsThreadSafe(), and this loop reaches integrators the
      earlier audits behind that flag did NOT cover: `c_bfi`'s
      AssembleFaceMatrix() and `c_bfi_p`'s AssembleHDGFaceMatrix(), which are
      LINEAR face integrators and so outside MultNL()'s promise, and the
      frozen flux-law coefficient a solution-dependent law is lifted through.
      That widening is written on SetIntegratorsThreadSafe() itself, which
      used to name this routine as the un-audited gap. The stock integrators
      this path installs -- NormalTraceJumpIntegrator, HDGDiffusionIntegrator,
      VectorMassIntegrator, VectorFEMassIntegrator, VectorDivergenceIntegrator
      and the two convection integrators -- are all behind
      `#ifndef MFEM_THREAD_SAFE`, checked rather than assumed, and
      AssemblyMode::Threaded already refuses a build without it. */
   const bool threaded = h.ThreadHostLoops() && h.GetIntegratorsThreadSafe();

   if (!threaded)
   {
      ReconstructWorkspace w;
      VectorDomainLFIntegrator bp(bp_coeff);
      w.bp = &bp;
      Array<int> faces, oris;
      Vector sol;

      for (int z = 0; z < NE; z++)
      {
         element_faces(z, faces, oris);
         solve_element(z, faces, w, sol);
         scatter_element(z, faces, sol, w);
      }
      return;
   }

   // The face lists and the local-solution offsets, taken once. This pass is
   // also what warms every lazily-built table the accessors above reach --
   // the space's element-to-dof tables, the mesh's element-to-face tables and
   // the trace and total-flux face dofs -- so that the parallel region below
   // only READS them. A table built for the first time from inside an OpenMP
   // region is a race that no amount of per-thread scratch can fix.
   Array<int> sol_offs(NE + 1), face_offs(NE + 1), all_faces;
   {
      Array<int> faces, oris, vdofs_u, dofs_p, vdofs_w;
      sol_offs[0] = 0;
      face_offs[0] = 0;
      for (int z = 0; z < NE; z++)
      {
         fes_u->GetElementVDofs(z, vdofs_u);
         fes_p->GetElementVDofs(z, dofs_p);
         fes_pc->GetElementVDofs(z, vdofs_w);

         element_faces(z, faces, oris);
         int ndof_tr = 0;
         for (int f : faces)
         {
            ndof_tr += neq * fes_tr->GetFaceElement(f)->GetDof();
            fes_tr->GetFaceVDofs(f, vdofs_w);
            fes_ut->GetFaceVDofs(f, vdofs_w);
            fes_ut->GetFaceElement(f);
         }

         all_faces.Append(faces);
         face_offs[z + 1] = all_faces.Size();
         sol_offs[z + 1] = sol_offs[z] + vdofs_u.Size() + dofs_p.Size() + ndof_tr;
      }
   }

   // A CHUNK of local solutions at a time, rather than all of them. Holding
   // every element's at once is the obvious arrangement and it costs about as
   // much again as the three enriched fields this routine is producing --
   // 735 MB on a 64-cubed hex mesh at order 2, which is not a buffer to take
   // without asking. A chunk bounded by its own size in reals gives the same
   // answer for a fixed, small one: the chunks are processed in order and the
   // replay inside each is in element order, so the last writer to any shared
   // dof is the element the serial loop would have left there. The cap is in
   // REALS and not in elements because the per-element size runs from a
   // handful at order 0 to a few thousand for a 3-D system, and it is the
   // product that has to be bounded.
   constexpr int chunk_cap = 1 << 20;   // 8 MB at double precision
   Vector sol_buf;

   // The replay's workspace, outside the chunk loop so that it grows once.
   ReconstructWorkspace scat;
   Array<int> scat_faces;
   Vector scat_sol;

   int z_beg = 0;
   while (z_beg < NE)
   {
      // At least one element, whatever its size, and then as many more as fit.
      int z_end = z_beg + 1;
      while (z_end < NE && sol_offs[z_end + 1] - sol_offs[z_beg] <= chunk_cap)
      {
         z_end++;
      }
      const int base = sol_offs[z_beg];
      sol_buf.SetSize(sol_offs[z_end] - base);

#ifdef MFEM_USE_OPENMP
      #pragma omp parallel
#endif
      {
         ReconstructWorkspace w;
         VectorDomainLFIntegrator bp(bp_coeff);
         w.bp = &bp;
         Array<int> faces;
         Vector sol;

#ifdef MFEM_USE_OPENMP
         #pragma omp for schedule(static)
#endif
         for (int z = z_beg; z < z_end; z++)
         {
            faces.MakeRef(all_faces.GetData() + face_offs[z],
                          face_offs[z + 1] - face_offs[z]);
            sol.MakeRef(sol_buf, sol_offs[z] - base,
                        sol_offs[z + 1] - sol_offs[z]);
            solve_element(z, faces, w, sol);
            // The slice was sized from the same three counts the local problem
            // is built from, so a mismatch means one of them moved between the
            // two passes -- and Vector::SetSize() would have shrunk the view
            // rather than said so, leaving the tail of the slice unwritten.
            MFEM_ASSERT(sol.Size() == sol_offs[z + 1] - sol_offs[z],
                        "element " << z << "'s local problem is "
                        << sol.Size() << " rows, its slice "
                        << sol_offs[z + 1] - sol_offs[z]);
         }
      }

      // The replay, serial and in element order. See the note above.
      for (int z = z_beg; z < z_end; z++)
      {
         scat_faces.MakeRef(all_faces.GetData() + face_offs[z],
                            face_offs[z + 1] - face_offs[z]);
         scat_sol.MakeRef(sol_buf, sol_offs[z] - base,
                          sol_offs[z + 1] - sol_offs[z]);
         scatter_element(z, scat_faces, scat_sol, scat);
      }

      z_beg = z_end;
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
   // The skeleton load owns its storage, so it re-sizes rather than
   // re-references. The hybridization's borrowed pointer stays valid.
   if (b_t) { b_t->Update(); }

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

   // The batched pass replaces the whole loop below, markers and the
   // periodic-mesh guard included; it builds its own work list from the same
   // two rules. See AssembleFluxMassBdrMatricesBatched(), and note that what
   // it batches is the SCATTER -- the integrators are evaluated on the host
   // either way, there being no integrator family here to dispatch on.
   if (num_boundary_face_integs > 0 &&
       !hybridization->AssembleFluxMassBdrMatricesBatched(M_u.get(), skip_zeros))
   {
      // BEFORE the loop, because the loop is host code accumulating into Af
      // and Ae through raw pointers and the element pass above may have been
      // a kernel. The one at the end of this routine is then a no-op on this
      // branch, the host copies already being valid.
      //
      // **Not reachable today, and saying so is the point.**
      // CanBatchFluxMassBdrFaces() accepts exactly when this loop would have
      // work to do, so under AssemblyMode::Batched the two are never both
      // live: work admitted means the kernel was taken, and no work admitted
      // means this loop reads nothing. It is here because the gate is the only
      // thing making that true -- add one refusal to it (a vdim condition, a
      // geometry condition, ParallelC()) and this line is what stands between
      // that and a fault naming neither the array nor the routine that left it
      // there. Measured, with the gate forced false by a probe while work was
      // admitted: without this line the fault lands inside
      // DarcyHybridization::AssembleFluxMassMatrix() under Device("debug").
      hybridization->SyncLocalBlocksToHost();

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
         // The second element is a dummy on a boundary face, as elsewhere: it
         // is never used, but a null reference cannot be formed.
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

   // Once, after BOTH passes of the flux mass group, and this is where the
   // sync that used to sit at the end of AssembleFluxMassMatricesBatched()
   // went. Either pass may have been a device kernel, and everything
   // downstream -- ComputeElementH(), the local factorisation, the per-element
   // MultInv() -- reads Af, Ae and the offset arrays through raw pointers,
   // which do not sync. Same sequencing, and the same reason, as the end of
   // AssemblePotHDGFaces().
   //
   // It is UNCONDITIONAL, including when there are no boundary integrators at
   // all: the element pass alone can have left the blocks device-valid, and
   // ComputeH() syncs only under LocalFactorMode::Batched, so nothing further
   // down guarantees it.
   hybridization->SyncLocalBlocksToHost();
}

void DarcyForm::AssemblePotHDGFaces(int skip_zeros)
{
   Mesh *mesh = fes_p->GetMesh();
   DenseMatrix elmat1, elmat2;
   Array<int> vdofs1, vdofs2;

   if (hybridization->GetPotConstraintIntegrator() &&
       !hybridization->AssemblePotFaceMatricesBatched())
   {
      int nfaces = mesh->GetNumFaces();
      for (int f = 0; f < nfaces; f++)
      {
         if (!mesh->FaceIsInterior(f)) { continue; }

         hybridization->ComputeAndAssemblePotFaceMatrix(f, elmat1, elmat2, vdofs1,
                                                        vdofs2);
#ifndef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
         // M_p is NULL when the potential mass lives on a NonlinearForm and
         // only the constraint came down the linear route -- there is then no
         // sparse potential mass to accumulate into, and nothing wants one.
         if (M_p)
         {
            if (M_p)
            {
               M_p->SpMat().AddSubMatrix(vdofs1, vdofs1, elmat1, skip_zeros);
            }
            M_p->SpMat().AddSubMatrix(vdofs2, vdofs2, elmat2, skip_zeros);
         }
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
      }
   }

   const int num_boundary_face_integs =
      hybridization->NumBdrPotConstraintIntegrators();

   // The batched boundary pass replaces the whole loop below, markers and
   // periodic-mesh guard included; it builds its own face lists from the same
   // two rules. It refuses far more often than it accepts -- see
   // CanBatchPotBdrFaceAssembly().
   if (num_boundary_face_integs > 0 &&
       !hybridization->AssemblePotBdrFaceMatricesBatched())
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

         // A PERIODIC MESH KEEPS THE BOUNDARY ELEMENTS WHOSE FACES THE
         // IDENTIFICATION TURNED INTERIOR, and GetBdrElementFaceIndex() then
         // hands back an interior face. ComputeAndAssemblePotBdrFaceMatrix()
         // writes ONE element's E, G and H over that face's slot with
         // CopyMN, which assigns -- so on an interior face it destroys the
         // two-sided blocks the interior pass has already assembled, whatever
         // the integrator's own value. An identically-zero boundary
         // integrator was measured to move the recovered potential by 23%.
         // Every other boundary loop in this file and in
         // DarcyHybridization::ConstructC drops these through
         // GetBdrFaceTransformations() returning null; this one has to as well.
         if (!mesh->GetBdrFaceTransformations(f)) { continue; }

         hybridization->ComputeAndAssemblePotBdrFaceMatrix(f, elmat1, vdofs1);
#ifndef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
         M_p->SpMat().AddSubMatrix(vdofs1, vdofs1, elmat1, skip_zeros);
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
      }
   }

   // Once, after BOTH passes. Either may have been a device kernel, and
   // everything downstream -- ComputeElementH(), the face loops, the boundary
   // flux pass -- reads E, G, H and D through raw pointers, which do not sync.
   hybridization->SyncLocalBlocksToHost();
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
