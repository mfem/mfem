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

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

namespace darcy_reduction
{

// Same mixed Darcy problem as test_darcy_hybridization.cpp:
//
//    k u + grad p = f
//      - div u    = g
//
// with the natural boundary condition -p = <given pressure> and k = 1, plus a
// zeroth-order term 'a p' on the potential.
//
// The reaction term is not decoration. Eliminating the potentials element by
// element inverts the potential mass block, and in the pure Darcy problem that
// block is identically zero -- CGSolver aborts on a non-finite residual, which
// is the library telling the truth about an inapplicable reduction rather than
// a defect. With a > 0 the block is invertible and the reduction is defined.
//
// What is checked is an algebraic identity that holds for any right-hand side:
// eliminating one block of unknowns locally must not change the discrete
// solution of the other. So no exact solution is compared against here; that
// is test_darcy_hybridization.cpp's job.

real_t pExact(const Vector &x)
{
   const real_t z = (x.Size() == 3) ? x(2) : 0.0;
   return exp(x(0)) * sin(x(1)) * cos(z);
}

real_t gExact(const Vector &x)
{
   return (x.Size() == 3) ? -pExact(x) : 0.0;
}

real_t pNatural(const Vector &x) { return -pExact(x); }

enum class Reduction { None, Potential, Flux };

struct Result
{
   Vector u, p;
   int    solved_size;
};

/// Solve with continuous RT fluxes, optionally eliminating the potentials
/// element by element.
Result SolveRT(Mesh &mesh, int order, Reduction red)
{
   const int dim = mesh.Dimension();

   RT_FECollection u_coll(order, dim);
   L2_FECollection p_coll(order, dim);
   FiniteElementSpace fes_u(&mesh, &u_coll);
   FiniteElementSpace fes_p(&mesh, &p_coll);

   ConstantCoefficient k(1.0);
   ConstantCoefficient a(1.0);
   VectorFunctionCoefficient fcoeff(dim, [](const Vector &, Vector &f)
   {
      f = 0.0;
   });
   FunctionCoefficient gcoeff(gExact), natcoeff(pNatural);

   DarcyForm darcy(&fes_u, &fes_p);
   darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorFEMassIntegrator(k));
   darcy.GetFluxDivForm()->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   darcy.GetPotentialMassForm()->AddDomainIntegrator(new MassIntegrator(a));

   LinearForm *fform = darcy.GetFluxRHS();
   fform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
   fform->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(natcoeff));
   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));

   Array<int> ess_flux_tdofs;
   if (red == Reduction::Potential)
   {
      darcy.EnablePotentialReduction(ess_flux_tdofs);
   }

   darcy.Assemble();

   BlockVector x(darcy.GetOffsets());
   x = 0.0;

   OperatorPtr A;
   Vector X, B;
   darcy.FormLinearSystem(ess_flux_tdofs, x, A, X, B, true);

   Result res;
   res.solved_size = X.Size();

   MINRESSolver solver;
   solver.SetMaxIter(20000);
   solver.SetRelTol(0.0);
   solver.SetAbsTol(1e-14);
   solver.SetOperator(*A);
   solver.Mult(B, X);
   REQUIRE(solver.GetConverged());

   darcy.RecoverFEMSolution(X, x);

   res.u = x.GetBlock(0);
   res.p = x.GetBlock(1);
   return res;
}

/// Solve with broken Raviart-Thomas fluxes -- discontinuous, so the flux mass
/// block is block-diagonal and can be eliminated element by element. The face
/// term on the divergence form is what makes the broken space consistent; it
/// needs no stabilization beyond that.
Result SolveBRT(Mesh &mesh, int order, bool reduce_flux)
{
   const int dim = mesh.Dimension();

   BrokenRT_FECollection u_coll(order, dim);
   L2_FECollection p_coll(order, dim);
   FiniteElementSpace fes_u(&mesh, &u_coll);
   FiniteElementSpace fes_p(&mesh, &p_coll);

   ConstantCoefficient k(1.0);
   VectorFunctionCoefficient fcoeff(dim, [](const Vector &, Vector &f)
   {
      f = 0.0;
   });
   FunctionCoefficient gcoeff(gExact), natcoeff(pNatural);

   DarcyForm darcy(&fes_u, &fes_p);
   darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorFEMassIntegrator(k));

   MixedBilinearForm *bVarf = darcy.GetFluxDivForm();
   bVarf->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   bVarf->AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.)));

   LinearForm *fform = darcy.GetFluxRHS();
   fform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
   fform->AddBdrFaceIntegrator(new VectorFEBoundaryFluxLFIntegrator(natcoeff));
   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));

   Array<int> ess_flux_tdofs;
   if (reduce_flux) { darcy.EnableFluxReduction(); }

   darcy.Assemble();

   BlockVector x(darcy.GetOffsets());
   x = 0.0;

   OperatorPtr A;
   Vector X, B;
   darcy.FormLinearSystem(ess_flux_tdofs, x, A, X, B, true);

   Result res;
   res.solved_size = X.Size();

   GMRESSolver solver;
   solver.SetKDim(2000);
   solver.SetMaxIter(5000);
   solver.SetRelTol(0.0);
   solver.SetAbsTol(1e-13);
   solver.SetOperator(*A);
   solver.Mult(B, X);
   REQUIRE(solver.GetConverged());

   darcy.RecoverFEMSolution(X, x);

   res.u = x.GetBlock(0);
   res.p = x.GetBlock(1);
   return res;
}

} // namespace darcy_reduction

TEST_CASE("Potential reduction reproduces the monolithic mixed solve",
          "[DarcyForm][DarcyReduction]")
{
   using namespace darcy_reduction;

   // Eliminating the block-diagonal potential mass block element by element is
   // exact arithmetic, so the recovered solution must agree with the unreduced
   // one to solver tolerance, and the system actually solved must be smaller.
   const int order = GENERATE(0, 1, 2);
   const Element::Type elem = GENERATE(Element::QUADRILATERAL,
                                       Element::TRIANGLE);

   Mesh mesh = Mesh::MakeCartesian2D(4, 4, elem, false, 1.0, 1.0);

   const Result full = SolveRT(mesh, order, Reduction::None);
   const Result red  = SolveRT(mesh, order, Reduction::Potential);

   CAPTURE(order, int(elem), full.solved_size, red.solved_size);

   REQUIRE(red.solved_size < full.solved_size);

   Vector du(red.u), dp(red.p);
   du -= full.u;
   dp -= full.p;

   REQUIRE(du.Normlinf() < 1e-8 * std::max(full.u.Normlinf(), real_t(1.0)));
   REQUIRE(dp.Normlinf() < 1e-8 * std::max(full.p.Normlinf(), real_t(1.0)));
}

TEST_CASE("Flux reduction reproduces the monolithic broken-RT solve",
          "[DarcyForm][DarcyReduction][BrokenRT]")
{
   using namespace darcy_reduction;

   // The mirror image of the potential reduction: with a broken flux space the
   // flux mass block is block-diagonal, so it is the flux that can be
   // eliminated locally. Same algebraic identity, other block.
   const int order = GENERATE(0, 1);
   const Element::Type elem = GENERATE(Element::QUADRILATERAL,
                                       Element::TRIANGLE);

   Mesh mesh = Mesh::MakeCartesian2D(3, 3, elem, false, 1.0, 1.0);

   const Result full = SolveBRT(mesh, order, false);
   const Result red  = SolveBRT(mesh, order, true);

   CAPTURE(order, int(elem), full.solved_size, red.solved_size);

   REQUIRE(red.solved_size < full.solved_size);

   Vector du(red.u), dp(red.p);
   du -= full.u;
   dp -= full.p;

   REQUIRE(du.Normlinf() < 1e-7 * std::max(full.u.Normlinf(), real_t(1.0)));
   REQUIRE(dp.Normlinf() < 1e-7 * std::max(full.p.Normlinf(), real_t(1.0)));
}
