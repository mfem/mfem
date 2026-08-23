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

namespace darcy_hybridization
{

// The mixed Darcy problem of examples/hdg/ex5.cpp,
//
//    k u + grad p = f
//      - div u    = g
//
// with the natural boundary condition -p = <given pressure>, k = 1, and the
// exact solution below -- for which f vanishes identically.

real_t pExact(const Vector &x)
{
   const real_t z = (x.Size() == 3) ? x(2) : 0.0;
   return exp(x(0)) * sin(x(1)) * cos(z);
}

void uExact(const Vector &x, Vector &u)
{
   const real_t z = (x.Size() == 3) ? x(2) : 0.0;
   u(0) = -exp(x(0)) * sin(x(1)) * cos(z);
   u(1) = -exp(x(0)) * cos(x(1)) * cos(z);
   if (x.Size() == 3) { u(2) = exp(x(0)) * sin(x(1)) * sin(z); }
}

// g = -div u = laplace p, which is -p in 3D and zero in 2D.
real_t gExact(const Vector &x)
{
   return (x.Size() == 3) ? -pExact(x) : 0.0;
}

real_t pNatural(const Vector &x) { return -pExact(x); }

struct Result
{
   Vector u, p;         ///< the recovered flux and potential
   real_t err_u, err_p; ///< L2 errors against the exact solution
   int    solved_size;  ///< size of the system actually solved
};

/// Solve the problem above with RT fluxes and L2 potentials, either as the
/// full block system or hybridized down to the traces. Everything except the
/// call to EnableHybridization() is shared, so a difference between the two
/// results is a property of the hybridization and of nothing else.
Result Solve(Mesh &mesh, int order, bool hybridize)
{
   const int dim = mesh.Dimension();

   RT_FECollection u_coll(order, dim);
   L2_FECollection p_coll(order, dim);
   FiniteElementSpace fes_u(&mesh, &u_coll);
   FiniteElementSpace fes_p(&mesh, &p_coll);

   ConstantCoefficient k(1.0);
   VectorFunctionCoefficient fcoeff(dim, [](const Vector &, Vector &f)
   {
      f = 0.0;
   });
   FunctionCoefficient gcoeff(gExact);
   FunctionCoefficient natcoeff(pNatural);
   FunctionCoefficient pcoeff(pExact);
   VectorFunctionCoefficient ucoeff(dim, uExact);

   DarcyForm darcy(&fes_u, &fes_p);

   darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorFEMassIntegrator(k));
   darcy.GetFluxDivForm()->AddDomainIntegrator(new VectorFEDivergenceIntegrator);

   LinearForm *fform = darcy.GetFluxRHS();
   fform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
   fform->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(natcoeff));
   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));

   Array<int> ess_flux_tdofs;   // the pressure enters naturally; none are essential

   // The trace space is only built when it is used, but it must outlive the
   // DarcyForm's hybridization, hence the scope of these two.
   DG_Interface_FECollection trace_coll(order, dim);
   FiniteElementSpace fes_t(&mesh, &trace_coll);
   if (hybridize)
   {
      darcy.EnableHybridization(&fes_t, new NormalTraceJumpIntegrator(),
                                ess_flux_tdofs);
   }

   darcy.Assemble();

   BlockVector x(darcy.GetOffsets());
   x = 0.0;

   OperatorPtr A;
   Vector X, B;
   darcy.FormLinearSystem(ess_flux_tdofs, x, A, X, B, true);

   Result res;
   res.solved_size = X.Size();

   if (hybridize)
   {
      GSSmoother prec;
      GMRESSolver solver;
      solver.SetKDim(1000);
      solver.SetMaxIter(2000);
      solver.SetRelTol(0.0);
      solver.SetAbsTol(1e-14);
      solver.SetPreconditioner(prec);
      solver.SetOperator(*A);
      solver.Mult(B, X);
      REQUIRE(solver.GetConverged());
   }
   else
   {
      // Symmetric indefinite saddle-point system; the meshes here are small
      // enough that unpreconditioned MINRES is both adequate and boring.
      MINRESSolver solver;
      solver.SetMaxIter(20000);
      solver.SetRelTol(0.0);
      solver.SetAbsTol(1e-14);
      solver.SetOperator(*A);
      solver.Mult(B, X);
      REQUIRE(solver.GetConverged());
   }

   darcy.RecoverFEMSolution(X, x);

   GridFunction u_h(&fes_u, x.GetBlock(0));
   GridFunction p_h(&fes_p, x.GetBlock(1));

   const int quad_order = 2 * order + 3;
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; i++)
   {
      irs[i] = &(IntRules.Get(i, quad_order));
   }

   res.err_u = u_h.ComputeL2Error(ucoeff, irs);
   res.err_p = p_h.ComputeL2Error(pcoeff, irs);
   res.u = x.GetBlock(0);
   res.p = x.GetBlock(1);
   return res;
}

} // namespace darcy_hybridization

TEST_CASE("Hybridized Darcy reproduces the monolithic mixed solve",
          "[DarcyForm][DarcyHybridization]")
{
   using namespace darcy_hybridization;

   // Hybridization of the mixed RT/L2 method is exact: eliminating the element
   // interiors in favour of a single trace unknown per face must return the
   // same discrete solution, not merely a comparable one. Anything above
   // solver tolerance here is a defect in DarcyHybridization.
   const int order = GENERATE(0, 1, 2);
   const Element::Type elem = GENERATE(Element::QUADRILATERAL,
                                       Element::TRIANGLE);

   Mesh mesh = Mesh::MakeCartesian2D(4, 4, elem, false, 1.0, 1.0);

   const Result mono = Solve(mesh, order, false);
   const Result hyb  = Solve(mesh, order, true);

   CAPTURE(order, int(elem), mono.solved_size, hyb.solved_size);

   // Hybridization must actually reduce the system it solves.
   REQUIRE(hyb.solved_size < mono.solved_size);

   Vector du(hyb.u), dp(hyb.p);
   du -= mono.u;
   dp -= mono.p;

   REQUIRE(du.Normlinf() < 1e-8 * std::max(mono.u.Normlinf(), real_t(1.0)));
   REQUIRE(dp.Normlinf() < 1e-8 * std::max(mono.p.Normlinf(), real_t(1.0)));

   // ... and the two must agree on the errors as well, to the same tolerance.
   REQUIRE(hyb.err_u == MFEM_Approx(mono.err_u, 1e-12, 1e-8));
   REQUIRE(hyb.err_p == MFEM_Approx(mono.err_p, 1e-12, 1e-8));
}

TEST_CASE("Hybridized Darcy converges at the design order",
          "[DarcyForm][DarcyHybridization]")
{
   using namespace darcy_hybridization;

   // A rate, not a value: this catches a scheme that solves a nearby problem,
   // which comparison against the monolithic path cannot, since both would be
   // wrong together.
   const int order = GENERATE(0, 1);

   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, false,
                                     1.0, 1.0);

   real_t prev_p = -1.0, prev_u = -1.0;
   for (int ref = 0; ref < 3; ref++)
   {
      const Result r = Solve(mesh, order, true);
      if (prev_p > 0.0)
      {
         const real_t rate_p = std::log2(prev_p / r.err_p);
         const real_t rate_u = std::log2(prev_u / r.err_u);
         CAPTURE(order, ref, rate_p, rate_u, r.err_p, r.err_u);
         REQUIRE(rate_p > order + 0.7);
         REQUIRE(rate_u > order + 0.7);
      }
      prev_p = r.err_p;
      prev_u = r.err_u;
      mesh.UniformRefinement();
   }
}
