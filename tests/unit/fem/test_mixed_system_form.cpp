// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

enum class MixedSystemMode
{
   Mixed,
   FluxReduced,
   PotentialReduced,
   Hybridized
};

void FluxRHS(const Vector &x, Vector &f)
{
   f.SetSize(x.Size());
   f(0) = 1.0 + x(0);
   f(1) = 2.0 - x(1);
   if (x.Size() == 3)
   {
      f(2) = 3.0 + x(2);
   }
}

real_t PotentialRHS(const Vector &x)
{
   return 1.0 + x(0) - 2.0*x(1);
}

TEST_CASE("MixedSystemForm basic assembly", "[MixedSystemForm]")
{
   // Check operator identities, not solutions: no global solve or recovery is
   // needed. Use reproducible vectors for every geometry and sign convention.
   constexpr int seed = 17;
   const bool bsym = GENERATE(false, true);
   const MixedSystemMode mode = GENERATE(MixedSystemMode::Mixed,
                                         MixedSystemMode::FluxReduced,
                                         MixedSystemMode::PotentialReduced,
                                         MixedSystemMode::Hybridized);
   const Element::Type geometry = GENERATE(Element::TRIANGLE,
                                           Element::QUADRILATERAL,
                                           Element::TETRAHEDRON,
                                           Element::HEXAHEDRON);
   CAPTURE(bsym, static_cast<int>(mode), static_cast<int>(geometry));

   const bool is_3d = geometry != Element::TRIANGLE &&
                      geometry != Element::QUADRILATERAL;
   Mesh mesh = is_3d ? Mesh::MakeCartesian3D(2, 2, 2, geometry)
               : Mesh::MakeCartesian2D(2, 2, geometry);
   const int dim = mesh.Dimension();
   constexpr int order = 0;

   // Broken flux and discontinuous potential spaces permit element-local
   // elimination. Hybridization restores normal-flux continuity weakly through
   // a trace multiplier; the unconstrained mixed operator uses the same spaces.
   BrokenRT_FECollection flux_fec(order, dim);
   L2_FECollection potential_fec(order, dim);
   FiniteElementSpace flux_fes(&mesh, &flux_fec);
   FiniteElementSpace potential_fes(&mesh, &potential_fec);

   MixedSystemForm mixed(&flux_fes, &potential_fes, bsym);

   VectorFunctionCoefficient flux_rhs(dim, FluxRHS);
   FunctionCoefficient potential_rhs(PotentialRHS);
   mixed.GetFluxRHS()->AddDomainIntegrator(
      new VectorFEDomainLFIntegrator(flux_rhs));
   mixed.GetPotentialRHS()->AddDomainIntegrator(
      new DomainLFIntegrator(potential_rhs));

   ConstantCoefficient one(1.0);
   mixed.GetFluxMassForm()->AddDomainIntegrator(
      new VectorFEMassIntegrator(one));
   mixed.GetFluxDivForm()->AddDomainIntegrator(
      new VectorFEDivergenceIntegrator);
   mixed.GetPotentialMassForm()->AddDomainIntegrator(new MassIntegrator(one));

   // Both mass blocks are invertible, so either primary field can be eliminated.
   // Leave flux essential DOFs empty to test the unmodified mixed block algebra.
   Array<int> ess_flux_tdofs;
   std::unique_ptr<DG_Interface_FECollection> trace_fec;
   std::unique_ptr<FiniteElementSpace> trace_fes;

   switch (mode)
   {
      case MixedSystemMode::Mixed:
         break;
      case MixedSystemMode::FluxReduced:
         mixed.EnableFluxReduction();
         break;
      case MixedSystemMode::PotentialReduced:
         mixed.EnablePotentialReduction(ess_flux_tdofs);
         break;
      case MixedSystemMode::Hybridized:
         trace_fec = std::make_unique<DG_Interface_FECollection>(order, dim);
         trace_fes = std::make_unique<FiniteElementSpace>(&mesh, trace_fec.get());
         mixed.EnableHybridization(trace_fes.get(), new NormalTraceJumpIntegrator(),
                                   ess_flux_tdofs);
         break;
   }

   mixed.Assemble();

   OperatorHandle assembled;
   mixed.FormSystemMatrix(ess_flux_tdofs, assembled);

   // Use the existing component forms for the unreduced operator action.
   auto &mass_u = *mixed.GetFluxMassForm();
   auto &div = *mixed.GetFluxDivForm();
   auto &mass_p = *mixed.GetPotentialMassForm();
   if (mode != MixedSystemMode::Mixed)
   {
      // Condensation assembles local matrices, rather than these global forms.
      mass_u.Assemble();
      mass_u.Finalize();
      div.Assemble();
      div.Finalize();
      mass_p.Assemble();
      mass_p.Finalize();
   }

   // K = [ M_u, s B^T; s B, s M_p ], with s = -1 for bsym.
   // Build its action from the component forms, separately from the condensed
   // operator. The block operator borrows these forms and the transpose wrapper.
   const real_t sign = bsym ? -1.0 : 1.0;
   BlockOperator full_operator(mixed.GetTrueOffsets());
   TransposeOperator div_transpose(&div);
   full_operator.SetDiagonalBlock(0, &mass_u);
   full_operator.SetDiagonalBlock(1, &mass_p, sign);
   full_operator.SetBlock(0, 1, &div_transpose, sign);
   full_operator.SetBlock(1, 0, &div, sign);

   if (mode == MixedSystemMode::Mixed)
   {
      // Check that FormSystemMatrix applies the same signed blocks as K.
      Vector x(assembled->Width()), actual(assembled->Height());
      Vector reference(full_operator.Height());
      x.Randomize(seed);
      assembled->Mult(x, actual);
      full_operator.Mult(x, reference);
      actual -= reference;
      REQUIRE(actual.Norml2() / reference.Norml2() == MFEM_Approx(0.0));
   }
   else
   {
      // Match the RHS to K's sign convention: bsym negates the potential
      // equation, including its load. These loads are not manufactured from x.
      BlockVector rhs(mixed.GetTrueOffsets());
      rhs.GetBlock(0) = *mixed.GetFluxRHS();
      rhs.GetBlock(1) = *mixed.GetPotentialRHS();
      if (bsym) { rhs.GetBlock(1).Neg(); }

      std::unique_ptr<MixedBilinearForm> constraint_transpose;
      Array<int> boundary_trace_dofs;
      if (mode == MixedSystemMode::Hybridized)
      {
         // C is the weak normal-flux jump, not a pointwise continuity test.
         // Hybridization uses interior constraints only in this configuration.
         constraint_transpose = std::make_unique<MixedBilinearForm>(
                                   trace_fes.get(), &flux_fes);
         constraint_transpose->AddTraceFaceIntegrator(
            new NormalTraceJumpIntegrator);
         constraint_transpose->Assemble();
         constraint_transpose->Finalize();
         // AddTraceFaceIntegrator also assembles boundary faces. Remove their
         // columns to match the interior-only constraint used by hybridization.
         Array<int> boundary_marker(mesh.bdr_attributes.Max());
         boundary_marker = 1;
         trace_fes->GetEssentialTrueDofs(boundary_marker, boundary_trace_dofs);
         for (int dof : boundary_trace_dofs)
         {
            constraint_transpose->SpMat().EliminateCol(dof);
         }
      }

      // R denotes the public RHS reduction map. Its sign normalization is
      // included, so the identities below use the returned S or H directly.
      auto reduce = [&](const BlockVector &input, Vector &output)
      {
         // Hybridization adds to an existing trace RHS, so reset each time.
         output.SetSize(assembled->Height());
         output = 0.0;
         if (mode == MixedSystemMode::Hybridized)
         {
            mixed.GetHybridization()->ReduceRHS(input, output);
         }
         else
         {
            mixed.GetReduction()->ReduceRHS(input, output);
         }
      };

      Vector reduced_rhs;
      reduce(rhs, reduced_rhs);
      // Probe K with an arbitrary mixed vector, rather than a reference
      // solution. Reduction must cancel the eliminated field's contribution.
      BlockVector x(mixed.GetTrueOffsets()), action(mixed.GetTrueOffsets());
      x.Randomize(seed);
      full_operator.Mult(x, action);
      Vector expected(assembled->Height());

      if (mode == MixedSystemMode::Hybridized)
      {
         // R(K x + C^T lambda) = H lambda - C x_flux.
         Vector lambda(assembled->Width());
         // Use a distinct reproducible probe for the independent trace field.
         lambda.Randomize(seed + 1);
         // Boundary trace rows are replaced by identity during finalization;
         // test the weak equivalence on the free (interior) trace space.
         lambda.SetSubVector(boundary_trace_dofs, 0.0);
         Vector trace_forcing(flux_fes.GetVSize());
         constraint_transpose->Mult(lambda, trace_forcing);
         action.GetBlock(0) += trace_forcing;
         assembled->Mult(lambda, expected);
         Vector flux_jump(assembled->Height());
         constraint_transpose->MultTranspose(x.GetBlock(0), flux_jump);
         expected -= flux_jump;
      }
      else
      {
         // R K x = S x_retained: flux elimination retains potential;
         // potential elimination retains flux, for either sign convention.
         const int retained = mode == MixedSystemMode::FluxReduced ? 1 : 0;
         assembled->Mult(x.GetBlock(retained), expected);
      }

      Vector actual;
      reduce(action, actual);
      actual -= expected;
      REQUIRE(actual.Norml2() / expected.Norml2() == MFEM_Approx(0.0));

      // The same equivalence must hold for the affine residual with the
      // assembled, nonzero RHS: R(action - b) = expected - R(b).
      action -= rhs;
      reduce(action, actual);
      expected -= reduced_rhs;
      actual -= expected;
      REQUIRE(actual.Norml2() / expected.Norml2() == MFEM_Approx(0.0));
   }
}
