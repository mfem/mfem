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
void Solve(Solution &s, int n, int order, real_t T = 1.0)
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
   fform->AddBdrFaceIntegrator(new VectorBoundaryFluxLFIntegrator(natcoeff));
   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));

   Array<int> ess;
   darcy.EnableHybridization(s.fes_t.get(), new NormalTraceJumpIntegrator(),
                             ess);
   darcy.Assemble();

   BlockVector x(darcy.GetOffsets());
   x = 0.0;
   OperatorPtr A;
   Vector X, Bv;
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

   for (auto type : {HDGErrorEstimator::Type::Energy,
                     HDGErrorEstimator::Type::Residual})
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

TEST_CASE("HDGErrorEstimator: local values fall, the total does not",
          "[HDGErrorEstimator]")
{
   using namespace estimators_hdg;

   // Measured, on a smooth manufactured solution with tau held fixed:
   //
   //     n     max local     total     true L2 error
   //     4      1.454        2.1339      5.77e-3
   //     8      1.085        2.1330      1.53e-3
   //    16      0.788        2.1329      3.84e-4
   //
   // The per-element value falls, at about h^0.45 here, and the total does not
   // move at all, because the element count grows faster than the per-element
   // value shrinks. For marking -- which is what an estimator is usually for,
   // and where only the relative sizes matter -- that is harmless. For
   // GetTotalError() as a stopping criterion it is not, and section 7 of
   // HDG-REQUIREMENTS should not assume otherwise without checking.
   //
   // Whether the flat total is intended is not established here. What is
   // asserted is only what was measured.
   const int order = GENERATE(1, 2);
   CAPTURE(order);

   real_t prev_loc = -1.0, prev_err = -1.0;
   real_t first_total = -1.0, last_total = -1.0;

   for (int n = 4; n <= 16; n *= 2)
   {
      Solution s;
      Solve(s, n, order);

      ConstantCoefficient k(1.0);
      RatioCoefficient ik(1.0, k);
      HDGDiffusionIntegrator integ(ik, 1.0 / n);
      HDGErrorEstimator est(integ, *s.tr_h, *s.p_h,
                            HDGErrorEstimator::Type::Energy);

      // GetTotalError() returns a cached value and does not itself trigger the
      // computation; only GetLocalErrors() and GetAnisotropicFlags() do, so
      // asking for the total first gives zero. That is the convention across
      // all of MFEM's estimators, not something particular to this one.
      const Vector &loc = est.GetLocalErrors();
      const real_t lmax = loc.Normlinf();
      const real_t total = est.GetTotalError();

      CAPTURE(n, lmax, total, s.err_p);
      REQUIRE(lmax > 0.0);
      REQUIRE(total > 0.0);

      if (prev_loc > 0.0)
      {
         // The local values shrink, which is what marking needs.
         REQUIRE(lmax < prev_loc);
         // ... and the true error shrinks faster, which is why the total
         // cannot be read as a proxy for it.
         REQUIRE(s.err_p < prev_err);
      }
      if (first_total < 0.0) { first_total = total; }
      last_total = total;

      prev_loc = lmax;
      prev_err = s.err_p;
   }

   INFO("total went from " << first_total << " to " << last_total);
   REQUIRE(last_total == MFEM_Approx(first_total, 1e-2, 1e-2));
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

