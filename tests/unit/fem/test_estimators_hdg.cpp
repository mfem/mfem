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

namespace estimators_hdg
{

// HDGErrorEstimator measures the trace jump |p - lambda|, the same quantity the
// scheme stabilizes with, so it needs no postprocessing pass. The tests below
// check the properties that follow from that: it vanishes when the element
// value and the trace agree, it has one entry per element, its total is the
// root of the sum of squares of the local values, and it falls under
// refinement.

real_t pExact(const Vector &x)
{
   const real_t z = (x.Size() == 3) ? x(2) : 0.0;
   return std::exp(x(0)) * std::sin(x(1)) * std::cos(z);
}
real_t gExact(const Vector &x) { return (x.Size() == 3) ? -pExact(x) : 0.0; }
real_t pNatural(const Vector &x) { return -pExact(x); }

struct Solution
{
   std::unique_ptr<Mesh> mesh;
   std::unique_ptr<L2_FECollection> u_coll, p_coll;
   std::unique_ptr<DG_Interface_FECollection> t_coll;
   std::unique_ptr<FiniteElementSpace> fes_u, fes_p, fes_t;
   std::unique_ptr<GridFunction> p_h, tr_h;
   real_t err_p{};
};

/// Solve the hybridized HDG problem and keep the potential and the trace,
/// which is what the estimator consumes.
///
/// @a ess_trace selects the boundary treatment, and **defaults to the
/// essential one**, which is the standard for the fully discontinuous spaces
/// on this branch. The boundary traces carry the projected datum, so lambda
/// approximates p there and the trace jump means on a boundary face what it
/// means everywhere else.
///
/// False is the older arrangement, copied from convdiff: the faces are
/// stabilized on the interior only and the datum enters weakly through the
/// flux equation, which leaves the boundary trace unknowns with an empty row
/// and an empty column -- dead, and left at zero. It solves correctly and is
/// still what the Raviart-Thomas path uses, having no alternative, but the
/// estimator cannot be used with it: see the test below.
void Solve(Solution &s, int n, int order, real_t T = 1.0,
           bool ess_trace = true)
{
   s.mesh.reset(new Mesh(Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL,
                                               false, 1.0, 1.0)));
   const int dim = s.mesh->Dimension();
   const real_t td = T / n;   // hold tau fixed under refinement

   s.u_coll.reset(new L2_FECollection(order, dim, BasisType::GaussLobatto));
   s.p_coll.reset(new L2_FECollection(order, dim));
   s.t_coll.reset(new DG_Interface_FECollection(order, dim));
   s.fes_u.reset(new FiniteElementSpace(s.mesh.get(), s.u_coll.get(), dim));
   s.fes_p.reset(new FiniteElementSpace(s.mesh.get(), s.p_coll.get()));
   s.fes_t.reset(new FiniteElementSpace(s.mesh.get(), s.t_coll.get()));

   ConstantCoefficient k(1.0);
   RatioCoefficient ik(1.0, k);
   VectorFunctionCoefficient fcoeff(dim, [](const Vector &, Vector &f)
   {
      f = 0.0;
   });
   FunctionCoefficient gcoeff(gExact), natcoeff(pNatural), pcoeff(pExact);

   DarcyForm darcy(s.fes_u.get(), s.fes_p.get());
   darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorMassIntegrator(k));
   MixedBilinearForm *B = darcy.GetFluxDivForm();
   B->AddDomainIntegrator(new VectorDivergenceIntegrator());
   B->AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.)));
   darcy.GetPotentialMassForm()->AddInteriorFaceIntegrator(
      new HDGDiffusionIntegrator(ik, td));

   LinearForm *fform = darcy.GetFluxRHS();
   fform->AddDomainIntegrator(new VectorDomainLFIntegrator(fcoeff));
   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));

   Array<int> bdr_ess(s.mesh->bdr_attributes.Max());
   bdr_ess = 1;

   if (ess_trace)
   {
      // The boundary faces join in, and the datum rides on the trace instead
      // of on the flux equation. The factor two against the interior's one is
      // convdiff's, and is there because only one side contributes.
      B->AddBdrFaceIntegrator(
         new TransposeIntegrator(new DGNormalTraceIntegrator(-2.)), bdr_ess);
      darcy.GetPotentialMassForm()->AddBdrFaceIntegrator(
         new HDGDiffusionIntegrator(ik, td), bdr_ess);
   }
   else
   {
      fform->AddBdrFaceIntegrator(new VectorBoundaryFluxLFIntegrator(natcoeff));
   }

   Array<int> ess;
   darcy.EnableHybridization(s.fes_t.get(), new NormalTraceJumpIntegrator(),
                             ess);
   if (ess_trace) { darcy.GetHybridization()->SetEssentialBC(bdr_ess); }
   darcy.Assemble();

   BlockVector x(darcy.GetOffsets());
   x = 0.0;
   OperatorPtr A;
   Vector X, Bv;
   if (ess_trace)
   {
      // FormLinearSystem takes the essential trace values from X, so it has to
      // arrive sized and carrying them.
      GridFunction tr0(s.fes_t.get());
      tr0 = 0.0;
      tr0.ProjectBdrCoefficient(pcoeff, bdr_ess);
      X = tr0;
   }
   darcy.FormLinearSystem(ess, x, A, X, Bv, true);

   GSSmoother prec;
   GMRESSolver solver;
   solver.SetKDim(2000);
   solver.SetMaxIter(20000);
   solver.SetRelTol(0.0);
   solver.SetAbsTol(1e-13);
   solver.SetPreconditioner(prec);
   solver.SetOperator(*A);
   solver.Mult(Bv, X);
   REQUIRE(solver.GetConverged());

   darcy.RecoverFEMSolution(X, x);

   s.p_h.reset(new GridFunction(s.fes_p.get()));
   *s.p_h = x.GetBlock(1);
   s.tr_h.reset(new GridFunction(s.fes_t.get()));
   *s.tr_h = X;

   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; i++)
   {
      irs[i] = &(IntRules.Get(i, 2 * order + 4));
   }
   s.err_p = s.p_h->ComputeL2Error(pcoeff, irs);
}

} // namespace estimators_hdg

TEST_CASE("HDGErrorEstimator vanishes when the trace matches the element",
          "[HDGErrorEstimator]")
{
   using namespace estimators_hdg;

   // The estimator is the trace jump, so a state in which the potential and
   // its trace are the same constant must produce no error at all. No solve is
   // needed to check that, and it is the sharpest statement available.
   const int order = GENERATE(0, 1, 2);
   CAPTURE(order);

   Mesh mesh = Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL, false,
                                     1.0, 1.0);
   const int dim = mesh.Dimension();

   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace fes_p(&mesh, &p_coll), fes_t(&mesh, &t_coll);

   GridFunction p_h(&fes_p), tr_h(&fes_t);
   p_h = 1.0;
   tr_h = 1.0;

   ConstantCoefficient q(2.5);
   HDGDiffusionIntegrator integ(q);

   for (auto type :
        {
           HDGErrorEstimator::Type::Energy,
           HDGErrorEstimator::Type::Residual
        })
   {
      HDGErrorEstimator est(integ, tr_h, p_h, type);
      const Vector &loc = est.GetLocalErrors();

      CAPTURE(int(type));
      REQUIRE(loc.Size() == mesh.GetNE());
      REQUIRE(loc.Normlinf() < 1e-12);
      REQUIRE(std::abs(est.GetTotalError()) < 1e-12);
   }
}

TEST_CASE("HDGErrorEstimator aggregates its local values",
          "[HDGErrorEstimator]")
{
   using namespace estimators_hdg;

   const int order = GENERATE(1, 2);
   CAPTURE(order);

   Solution s;
   Solve(s, 4, order);

   ConstantCoefficient k(1.0);
   RatioCoefficient ik(1.0, k);
   HDGDiffusionIntegrator integ(ik, 1.0 / 4);

   HDGErrorEstimator est(integ, *s.tr_h, *s.p_h,
                         HDGErrorEstimator::Type::Energy);

   const Vector &loc = est.GetLocalErrors();
   REQUIRE(loc.Size() == s.mesh->GetNE());
   REQUIRE(loc.Min() >= 0.0);
   REQUIRE(loc.Normlinf() > 0.0);   // a real solution has a nonzero jump

   // For the energy type the total is the root of the sum of squares.
   real_t sum2 = 0.0;
   for (int i = 0; i < loc.Size(); i++) { sum2 += loc(i) * loc(i); }
   INFO("total " << est.GetTotalError() << " vs sqrt(sum of squares) "
        << std::sqrt(sum2));
   REQUIRE(est.GetTotalError() == MFEM_Approx(std::sqrt(sum2), 1e-10, 1e-9));
}

namespace estimators_hdg
{

/// The estimator's local values and total on one mesh.
struct Estimate
{
   real_t lmax, total, err_p;
};

Estimate Measure(int n, int order, bool ess_trace = true)
{
   Solution s;
   Solve(s, n, order, 1.0, ess_trace);

   ConstantCoefficient k(1.0);
   RatioCoefficient ik(1.0, k);
   HDGDiffusionIntegrator integ(ik, 1.0 / n);
   HDGErrorEstimator est(integ, *s.tr_h, *s.p_h,
                         HDGErrorEstimator::Type::Energy);

   // GetTotalError() returns a cached value and does not itself trigger the
   // computation; only GetLocalErrors() and GetAnisotropicFlags() do, so
   // asking for the total first gives zero. That is the convention across all
   // of MFEM's estimators, not something particular to this one.
   const Vector &loc = est.GetLocalErrors();
   return { loc.Normlinf(), est.GetTotalError(), s.err_p };
}

} // namespace estimators_hdg

TEST_CASE("HDGErrorEstimator on unconstrained boundary traces measures p, not the error",
          "[HDGErrorEstimator]")
{
   using namespace estimators_hdg;

   // The trap, pinned. With the branch's usual DG boundary arrangement the
   // boundary trace unknowns are dead -- empty row, empty column, left at zero
   // -- so on a boundary face the jump the estimator integrates is p_h itself,
   // which does not converge to anything. The indicator is then not an error
   // measure at all there.
   //
   // The arithmetic is exact enough to assert. The energy is
   // sum_F integral_F tau (p - lambda)^2, tau is one here, and lambda is zero
   // on the boundary, so the total tends to the boundary norm of the exact
   // solution:
   //
   //     ||p||_{L2(dOmega)}^2 = 0.272675 + 2.014766 + 0 + 2.261961 = 4.549402
   //
   // for p = exp(x) sin(y) on the unit square, whose root is 2.132933.
   const int order = GENERATE(1, 2);
   CAPTURE(order);

   //     k=1   n     max local     total      true L2 error
   //            4      1.4542       2.133883     5.77e-3
   //            8      1.0855       2.133017     1.53e-3
   //           16      0.7880       2.132926     3.84e-4
   //     k=2   16      0.7881       2.132902     1.70e-6
   //
   // The total agrees with the closed form to six figures and does not care
   // what the polynomial degree is, which on its own settles that it is not
   // measuring a discretization error. The local values fall at h^0.45 -- the
   // square root of h, out of the face measure -- while the true error falls
   // at h^{k+1}, so the boundary elements outweigh the interior ones by
   // h^{-(k+1)} and grow to dominate. Marking on this would refine the
   // boundary and nothing else, which is a worse failure than the stopping
   // criterion this test used to be about.
   const real_t bdr_norm = 2.132933;

   real_t first_loc = -1.0, last_loc = -1.0, first_err = -1.0, last_err = -1.0;
   for (int n = 4; n <= 16; n *= 2)
   {
      const Estimate e = Measure(n, order, false);
      CAPTURE(n, e.lmax, e.total, e.err_p);
      // Not merely flat: flat at the boundary norm of p.
      REQUIRE(e.total == MFEM_Approx(bdr_norm, 2e-3, 2e-3));
      if (first_loc < 0.0) { first_loc = e.lmax; first_err = e.err_p; }
      last_loc = e.lmax;
      last_err = e.err_p;
   }

   const real_t r_loc = std::log2(first_loc / last_loc) / 2.0;
   const real_t r_err = std::log2(first_err / last_err) / 2.0;
   CAPTURE(r_loc, r_err);
   REQUIRE(r_loc < 0.6);                 // the square root of h, not h^{k+1}
   REQUIRE(r_err > order + 0.7);         // while the solution is converging
}

TEST_CASE("HDGErrorEstimator falls once the boundary traces are constrained",
          "[HDGErrorEstimator]")
{
   using namespace estimators_hdg;

   const int order = GENERATE(1, 2);
   CAPTURE(order);

   std::vector<real_t> lm, tot, err;
   for (int n = 4; n <= 16; n *= 2)
   {
      const Estimate e = Measure(n, order, true);
      lm.push_back(e.lmax);
      tot.push_back(e.total);
      err.push_back(e.err_p);
   }

   const int m = tot.size();
   auto rate = [&](const std::vector<real_t> &v)
   {
      return std::log2(v[m-2] / v[m-1]);
   };
   const real_t r_loc = rate(lm), r_tot = rate(tot), r_err = rate(err);
   CAPTURE(r_loc, r_tot, r_err, lm[m-1], tot[m-1], err[m-1]);

   // Constrain the boundary traces and the indicator behaves as an energy-norm
   // indicator should. Measured over 8x8 to 16x16, against what the scaling
   // predicts -- the energy is sum_F integral_F tau (p - lambda)^2, so with a
   // fixed tau a local value carries one power of the face measure that the
   // pointwise jump does not:
   //
   //       k    local        total        true L2
   //            pred  meas   pred  meas   pred  meas
   //       1    2.5   2.42   1.5   1.41   2     1.90
   //       2    3.5   3.41   2.5   2.46   3     2.96
   //
   // The total falls half an order slower than the L2 error, which is not a
   // defect: it is an energy-norm quantity and a different norm. What matters
   // is that it falls at all, and that the local values fall faster than the
   // error rather than slower, which is what makes marking meaningful.
   REQUIRE(r_loc > order + 1.2);
   REQUIRE(r_tot > order + 0.2);
   REQUIRE(r_err > order + 0.7);

   // And real reduction over the three meshes, so that a rate taken between
   // two of them cannot flatter a stagnant sequence. Two refinements at the
   // measured 1.41 is a factor of 6.9 at k=1, and far more at k=2; four is the
   // floor.
   REQUIRE(tot[m-1] < 0.25 * tot[0]);
}

TEST_CASE("HDGErrorEstimator produces anisotropic flags",
          "[HDGErrorEstimator]")
{
   using namespace estimators_hdg;

   // The flags are what make this estimator usable on the extruded prism mesh
   // the application has chosen, where the two directions refine independently.
   Solution s;
   Solve(s, 4, 1);

   ConstantCoefficient k(1.0);
   RatioCoefficient ik(1.0, k);
   HDGDiffusionIntegrator integ(ik, 1.0 / 4);

   HDGErrorEstimator est(integ, *s.tr_h, *s.p_h,
                         HDGErrorEstimator::Type::Energy);
   est.SetAnisotropic(true);

   const Array<int> &flags = est.GetAnisotropicFlags();
   REQUIRE(flags.Size() == s.mesh->GetNE());

   // The flags are a refinement-type bitmask, so every entry has to be one the
   // mesh would accept.
   for (int i = 0; i < flags.Size(); i++)
   {
      CAPTURE(i, flags[i]);
      REQUIRE(flags[i] >= 0);
      REQUIRE(flags[i] <= 3);
   }
}

