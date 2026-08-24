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

/// Which discretisation of the flux. DG is the Nguyen-Peraire-Cockburn
/// setting -- both variables discontinuous, coupled only through the trace and
/// a stabilisation tau. RT is the hybridized mixed method, kept as a control
/// because hybridization of it is algebraically exact and carries no tau at
/// all, which makes it a reference with nothing to tune.
enum class Form { RT, DG };

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
Result Solve(Mesh &mesh, int order, bool hybridize, Form form = Form::RT,
             real_t td = 0.5)
{
   const int dim = mesh.Dimension();

   std::unique_ptr<FiniteElementCollection> u_coll;
   if (form == Form::DG)
   {
      u_coll.reset(new L2_FECollection(order, dim, BasisType::GaussLobatto));
   }
   else
   {
      u_coll.reset(new RT_FECollection(order, dim));
   }
   L2_FECollection p_coll(order, dim);
   FiniteElementSpace fes_u(&mesh, u_coll.get(),
                            (form == Form::DG) ? dim : 1);
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

   RatioCoefficient ik(1.0, k);

   DarcyForm darcy(&fes_u, &fes_p);

   LinearForm *fform = darcy.GetFluxRHS();
   if (form == Form::DG)
   {
      // Both variables discontinuous. The normal trace term on the divergence
      // form and the stabilisation on the potential mass form are what replace
      // the H(div) conformity that RT supplies for free.
      darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorMassIntegrator(k));
      MixedBilinearForm *B = darcy.GetFluxDivForm();
      B->AddDomainIntegrator(new VectorDivergenceIntegrator());
      B->AddInteriorFaceIntegrator(
         new TransposeIntegrator(new DGNormalTraceIntegrator(-1.)));
      darcy.GetPotentialMassForm()->AddInteriorFaceIntegrator(
         new HDGDiffusionIntegrator(ik, td));

      fform->AddDomainIntegrator(new VectorDomainLFIntegrator(fcoeff));
      fform->AddBdrFaceIntegrator(new VectorBoundaryFluxLFIntegrator(natcoeff));
   }
   else
   {
      darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorFEMassIntegrator(k));
      darcy.GetFluxDivForm()->AddDomainIntegrator(
         new VectorFEDivergenceIntegrator);

      fform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
      fform->AddBoundaryIntegrator(
         new VectorFEBoundaryFluxLFIntegrator(natcoeff));
   }
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

TEST_CASE("Hybridized Darcy in three dimensions on hexahedra",
          "[DarcyForm][DarcyHybridization]")
{
   using namespace darcy_hybridization;

   // Nothing in fem/darcy has ever run in 3D: every HDG miniapp and example
   // builds a 2D mesh, so DarcyHybridization's three-dimensional face handling
   // is unexercised. Establish that before blaming anything on element type.
   const int order = launch_all_non_regression_tests ? GENERATE(0, 1, 2)
                     : GENERATE(0, 1);

   Mesh mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON,
                                     1.0, 1.0, 1.0);

   const Result mono = Solve(mesh, order, false);
   const Result hyb  = Solve(mesh, order, true);

   CAPTURE(order, mono.solved_size, hyb.solved_size);

   REQUIRE(hyb.solved_size < mono.solved_size);

   Vector du(hyb.u), dp(hyb.p);
   du -= mono.u;
   dp -= mono.p;

   REQUIRE(du.Normlinf() < 1e-8 * std::max(mono.u.Normlinf(), real_t(1.0)));
   REQUIRE(dp.Normlinf() < 1e-8 * std::max(mono.p.Normlinf(), real_t(1.0)));
}

TEST_CASE("Hybridized Darcy on wedges", "[DarcyForm][DarcyHybridization][Wedge]")
{
   using namespace darcy_hybridization;

   // A wedge carries two triangular and three quadrilateral faces, so a single
   // element has mixed face geometry -- the structural difference from every
   // element this code has been run on, and the element the extruded velocity
   // mesh of the application is made of.
   //
   // Order 2 in 3D is minutes rather than seconds, which is more than MFEM's
   // suite budgets for, so it runs only under --all.
   const int order = launch_all_non_regression_tests ? GENERATE(0, 1, 2)
                     : GENERATE(0, 1);

   Mesh mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::WEDGE, 1.0, 1.0, 1.0);
   REQUIRE(mesh.GetElementType(0) == Element::WEDGE);

   const Result mono = Solve(mesh, order, false);
   const Result hyb  = Solve(mesh, order, true);

   CAPTURE(order, mono.solved_size, hyb.solved_size);

   REQUIRE(hyb.solved_size < mono.solved_size);

   Vector du(hyb.u), dp(hyb.p);
   du -= mono.u;
   dp -= mono.p;

   REQUIRE(du.Normlinf() < 1e-8 * std::max(mono.u.Normlinf(), real_t(1.0)));
   REQUIRE(dp.Normlinf() < 1e-8 * std::max(mono.p.Normlinf(), real_t(1.0)));
}

TEST_CASE("Hybridized Darcy converges on wedges",
          "[DarcyForm][DarcyHybridization][Wedge]")
{
   using namespace darcy_hybridization;

   const int order = launch_all_non_regression_tests ? GENERATE(0, 1, 2)
                     : GENERATE(0, 1);

   Mesh mesh = Mesh::MakeCartesian3D(1, 1, 1, Element::WEDGE, 1.0, 1.0, 1.0);

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

// HDGDiffusionIntegrator does not take tau. Its parameter enters as
//
//     tau = td * kappa / h,     1/h = |nor|/det(J)
//
// which the integrator's own source comment states. So holding td fixed while
// refining makes tau grow like 1/h, and a sweep over td at fixed mesh sequence
// measures the coefficient of a 1/h-scaled stabilization rather than tau.
//
// Nguyen, Peraire and Cockburn take eta_d = kappa/ell with ell a fixed length
// of the problem (NPC-1 section 3.6.3), so their tau is O(1). To hold tau at T
// here, pass td = T*h; the meshes are uniform on the unit square, so h = 1/n.
//
// The two scalings are different methods, both legitimate:
//
//     tau fixed (NPC)   flux k+1, scalar k+1
//     td fixed          flux k,   scalar about k+1.5   (scalar superconverges)
//
// measured below and cross-checked against convdiff -p 2 -dg -hb, which
// reproduces NPC-1 Table 1 to within 0.15 of an order when tau is held fixed.

namespace darcy_hybridization
{

struct DGRate { real_t p, u; };

/// Solve on a sequence of meshes and return the rates between the two finest.
/// With @a fixed_tau the stabilization is held at T under refinement, which is
/// the NPC scaling; otherwise td is held at T and tau grows like 1/h.
DGRate DGRates(int order, real_t T, bool fixed_tau = true, int nref = 3,
               int n0 = 2, Element::Type elem = Element::QUADRILATERAL)
{
   Mesh mesh = (elem == Element::WEDGE)
               ? Mesh::MakeCartesian3D(n0, n0, n0, elem, 1.0, 1.0, 1.0)
               : Mesh::MakeCartesian2D(n0, n0, elem, false, 1.0, 1.0);

   real_t prev_p = -1.0, prev_u = -1.0;
   DGRate out{0.0, 0.0};
   int n = n0;
   for (int r = 0; r <= nref; r++)
   {
      const real_t td = fixed_tau ? (T / n) : T;
      const Result res = Solve(mesh, order, true, Form::DG, td);
      if (prev_p > 0.0)
      {
         out.p = std::log2(prev_p / res.err_p);
         out.u = std::log2(prev_u / res.err_u);
      }
      prev_p = res.err_p;
      prev_u = res.err_u;
      if (r < nref) { mesh.UniformRefinement(); n *= 2; }
   }
   return out;
}

} // namespace darcy_hybridization

TEST_CASE("HDG converges at k+1 in both variables for a fixed tau",
          "[DarcyForm][DarcyHybridization][HDG]")
{
   using namespace darcy_hybridization;

   // The Nguyen-Peraire-Cockburn result: with tau held fixed, both the scalar
   // and the flux converge at the design order.
   //
   // At k = 0 the flux needs tau to be large enough -- measured rates 0.49,
   // 0.67, 0.89, 1.10 at tau = 0.5, 1, 2, 4 -- so the sweep starts at 2 there.
   // That is the opposite of a degradation with large tau, and it is consistent
   // with NPC-1 Example 1, whose stabilization is |c.n| + kappa/ell and so is
   // itself of order 2 for that problem. At k >= 1 the whole range is optimal.
   const int order = GENERATE(0, 1, 2);
   const real_t T = (order == 0) ? GENERATE(2.0, 4.0)
                    : GENERATE(0.5, 1.0, 2.0, 4.0);

   const DGRate r = DGRates(order, T, true);
   CAPTURE(order, T, r.p, r.u);

   REQUIRE(r.p > order + 0.7);
   REQUIRE(r.u > order + 0.7);
}

TEST_CASE("HDG: the 1/h scaling trades flux order for scalar superconvergence",
          "[DarcyForm][HDG]")
{
   using namespace darcy_hybridization;

   // Holding td fixed instead is a different method, and this pins what it
   // does rather than calling it a degradation: the flux drops to k while the
   // scalar gains about half an order over k+1. Both are real, and confusing
   // this with the NPC scaling is what produced a wrong entry in
   // HDG-REQUIREMENTS section 5, since at a single resolution the two are
   // indistinguishable.
   const int order = GENERATE(1, 2);

   const DGRate fixed_tau = DGRates(order, 1.0, true);
   const DGRate fixed_td  = DGRates(order, 1.0, false);
   CAPTURE(order, fixed_tau.p, fixed_tau.u, fixed_td.p, fixed_td.u);

   REQUIRE(fixed_tau.u > order + 0.7);        // optimal
   REQUIRE(fixed_td.u  < fixed_tau.u - 0.5);  // and the 1/h flux is lower
   REQUIRE(fixed_td.p  > fixed_tau.p);        // while its scalar is higher
}

TEST_CASE("HDG: the discontinuous formulation on wedges",
          "[DarcyForm][DarcyHybridization][HDG][Wedge]")
{
   using namespace darcy_hybridization;

   // The element the application has chosen, in the formulation it will
   // actually use.
   const int order = launch_all_non_regression_tests ? GENERATE(0, 1, 2)
                     : GENERATE(0, 1);

   const DGRate r = DGRates(order, 1.0, true, 2, 2, Element::WEDGE);
   CAPTURE(order, r.p, r.u);

   REQUIRE(r.p > order + 0.7);
}
