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

/** A flux-mass boundary face integrator whose only property is that it is
    symmetric positive definite on the block of the element adjacent to the
    face: @a s times the identity. It discretises nothing. What is under test
    is the plumbing -- that the block reaches the element it belongs to, and
    that it is *added* to what the domain integrators have already put there
    rather than replacing it. */
class ScaledIdentityFaceIntegrator : public BilinearFormIntegrator
{
   const real_t s;
   const int vdim;

public:
   ScaledIdentityFaceIntegrator(real_t s_, int vdim_) : s(s_), vdim(vdim_) { }

   void AssembleFaceMatrix(const FiniteElement &el1, const FiniteElement &el2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override
   {
      const int n = el1.GetDof() * vdim;
      elmat.SetSize(n);
      elmat = 0.0;
      for (int i = 0; i < n; i++) { elmat(i, i) = s; }
   }
};

/// Which discretisation of the flux. DG is the Nguyen-Peraire-Cockburn
/// setting -- both variables discontinuous, coupled only through the trace and
/// a stabilisation tau. RT is the hybridized mixed method, kept as a control
/// because hybridization of it is algebraically exact and carries no tau at
/// all, which makes it a reference with nothing to tune.
enum class Form { RT, DG };

struct Result
{
   Vector u, p;         ///< the recovered flux and potential
   Vector t;            ///< the trace, as solved for (empty if not hybridized)
   real_t err_u, err_p; ///< L2 errors against the exact solution
   int    solved_size;  ///< size of the system actually solved
};

/// Solve the problem above with RT fluxes and L2 potentials, either as the
/// full block system or hybridized down to the traces. Everything except the
/// call to EnableHybridization() is shared, so a difference between the two
/// results is a property of the hybridization and of nothing else.
/** @a m_u_bdr installs one flux-mass boundary face integrator per entry, of
    the given scale. Empty for every caller but the boundary-face assembly
    test below. */
/** @a trace_order is the degree of the trace space; negative means @a order,
    which is what every caller but the redundant-trace test below wants. */
Result Solve(Mesh &mesh, int order, bool hybridize, Form form = Form::RT,
             real_t td = 0.5, const std::vector<real_t> &m_u_bdr = {},
             bool ess_trace = false, real_t marker_scale = -2.0,
             int trace_order = -1)
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
      if (!ess_trace)
      {
         fform->AddBoundaryIntegrator(
            new VectorFEBoundaryFluxLFIntegrator(natcoeff));
      }
      else
      {
         // The boundary block of C is registered from the divergence form's
         // boundary face *markers*: DarcyForm::Assemble() reads
         // B->GetBFBFI_Marker() and installs constr_flux_integ on each marker
         // it finds. The integrator object itself is never assembled on the
         // hybridized path -- only AssembleDivLDGFaces(), which the reduction
         // branch calls, touches it -- so what this line supplies is the
         // marker and nothing else.
         Array<int> all(mesh.bdr_attributes.Max());
         all = 1;
         darcy.GetFluxDivForm()->AddBdrFaceIntegrator(
            new TransposeIntegrator(new DGNormalTraceIntegrator(marker_scale)),
            all);
      }
   }
   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));

   for (real_t s : m_u_bdr)
   {
      darcy.GetFluxMassForm()->AddBdrFaceIntegrator(
         new ScaledIdentityFaceIntegrator(s, fes_u.GetVDim()));
   }

   // The pressure enters naturally, so none of the flux dofs are essential.
   Array<int> ess_flux_tdofs;

   // The trace space is only built when it is used, but it must outlive the
   // DarcyForm's hybridization, hence the scope of these two.
   DG_Interface_FECollection trace_coll((trace_order < 0) ? order : trace_order,
                                        dim);
   FiniteElementSpace fes_t(&mesh, &trace_coll);
   Array<int> bdr_all(mesh.bdr_attributes.Max());
   bdr_all = 1;
   if (hybridize)
   {
      darcy.EnableHybridization(&fes_t, new NormalTraceJumpIntegrator(),
                                ess_flux_tdofs);
      if (ess_trace) { darcy.GetHybridization()->SetEssentialBC(bdr_all); }
   }

   darcy.Assemble();

   BlockVector x(darcy.GetOffsets());
   x = 0.0;

   OperatorPtr A;
   Vector X, B;
   if (ess_trace)
   {
      GridFunction tr0(&fes_t);
      tr0 = 0.0;
      tr0.ProjectBdrCoefficient(pcoeff, bdr_all);
      X = tr0;
   }
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

   if (hybridize) { res.t = X; }

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
   const int order = GENERATE(0, 1, 2);

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
   const int order = GENERATE(0, 1, 2);

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

TEST_CASE("Hybridized Darcy on wedges",
          "[DarcyForm][DarcyHybridization][Wedge]")
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

   const int order = GENERATE(0, 1, 2);

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
   const int order = GENERATE(0, 1, 2);

   const DGRate r = DGRates(order, 1.0, true, 2, 2, Element::WEDGE);
   CAPTURE(order, r.p, r.u);

   REQUIRE(r.p > order + 0.7);
}

TEST_CASE("A boundary face term on the flux mass reaches the hybridized solve",
          "[DarcyForm][DarcyHybridization]")
{
   using namespace darcy_hybridization;

   // BilinearForm::ComputeElementMatrix(), which the hybridized assembly uses
   // to obtain each element's flux mass block, sums the domain integrators
   // only. DarcyForm::AssembleFluxMassBdrFaces() is what carries a boundary
   // face integrator of that form into the hybridization, and it does so by
   // calling AssembleFluxMassMatrix() a second time for the element owning the
   // face. So that routine has to accumulate. Assigning instead does not drop
   // the term -- it drops everything else, replacing the element's whole
   // block with the boundary contribution, silently and only on the elements
   // that touch the boundary.
   //
   // RT is the form to test it in, because hybridizing the mixed method is
   // algebraically exact: the monolithic solve assembles the same integrator
   // through BilinearForm::Assemble()'s own boundary face loop, which uses the
   // identical fe2 = fe1 convention, so the two must agree to solver
   // tolerance and the reference is right by construction rather than by
   // measurement.
   const int order = GENERATE(0, 1, 2);
   const Element::Type elem = GENERATE(Element::QUADRILATERAL,
                                       Element::TRIANGLE);
   CAPTURE(order, int(elem));

   Mesh mesh = Mesh::MakeCartesian2D(4, 4, elem, false, 1.0, 1.0);

   const Result mono = Solve(mesh, order, false, Form::RT, 0.5, {0.7});
   const Result hyb  = Solve(mesh, order, true,  Form::RT, 0.5, {0.7});

   Vector du(hyb.u), dp(hyb.p);
   du -= mono.u;
   dp -= mono.p;
   REQUIRE(du.Normlinf() < 1e-8 * std::max(mono.u.Normlinf(), real_t(1.0)));
   REQUIRE(dp.Normlinf() < 1e-8 * std::max(mono.p.Normlinf(), real_t(1.0)));

   // The term has to have done something, or the agreement above is vacuous.
   const Result plain = Solve(mesh, order, true, Form::RT);
   Vector d0(hyb.u);
   d0 -= plain.u;
   REQUIRE(d0.Normlinf() > 1e-3 * plain.u.Normlinf());

   // And the contributions accumulate across integrators as well as with the
   // domain block: two of scale s must equal one of scale 2s. A corner element
   // carries two boundary faces and so makes the same demand of the face loop.
   const Result twice = Solve(mesh, order, true, Form::RT, 0.5, {0.35, 0.35});
   Vector d2(twice.u);
   d2 -= hyb.u;
   REQUIRE(d2.Normlinf() < 1e-8 * std::max(hyb.u.Normlinf(), real_t(1.0)));
}

TEST_CASE("Raviart-Thomas takes its Dirichlet datum on an essential trace",
          "[DarcyForm][DarcyHybridization]")
{
   using namespace darcy_hybridization;

   // The boundary block of the constraint matrix C is registered from the
   // *divergence form's* boundary face markers: DarcyForm::Assemble() reads
   // B->GetBFBFI_Marker() and installs constr_flux_integ wherever it finds
   // one. The RT harnesses add no B face integrators at all, so nothing
   // registers one, so lambda on a boundary face has no entry in C -- and the
   // solve, being otherwise well posed through the natural boundary term on
   // the flux right-hand side, returns zero there without complaint.
   //
   // That is the whole of the gap. Supplying the marker closes it, and it
   // costs the discretisation nothing, which is the second half of this case:
   // on the hybridized path the marker's integrator is never assembled. Only
   // AssembleDivLDGFaces(), which the *reduction* branch calls, evaluates
   // B's face integrators; the hybridized branch takes the element matrices
   // alone and gets its face coupling from C. So the object handed to
   // AddBdrFaceIntegrator() here is read for its marker and for nothing else.
   const int order = GENERATE(0, 1, 2);
   const int n = GENERATE(4, 8);
   CAPTURE(order, n);

   Mesh mesh = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, false, 1., 1.);

   const Result nat = Solve(mesh, order, true, Form::RT);
   const Result ess = Solve(mesh, order, true, Form::RT, 0.5, {}, true);

   DG_Interface_FECollection trace_coll(order, 2);
   FiniteElementSpace fes_t(&mesh, &trace_coll);
   REQUIRE(fes_t.GetVSize() == nat.t.Size());

   // Without the marker, every boundary trace dof is exactly zero; with it,
   // they carry the datum they were given, to round-off.
   Array<int> bdr_all(mesh.bdr_attributes.Max());
   bdr_all = 1;
   GridFunction tr(&fes_t);
   tr = 0.0;
   FunctionCoefficient pcoeff(pExact);
   tr.ProjectBdrCoefficient(pcoeff, bdr_all);

   Array<int> bdr_dofs;
   fes_t.GetEssentialTrueDofs(bdr_all, bdr_dofs);
   REQUIRE(bdr_dofs.Size() > 0);

   real_t nat_bdr = 0.0, ess_drift = 0.0, datum = 0.0;
   for (int i = 0; i < bdr_dofs.Size(); i++)
   {
      const int d = bdr_dofs[i];
      nat_bdr   = std::max(nat_bdr, std::abs(nat.t(d)));
      ess_drift = std::max(ess_drift, std::abs(ess.t(d) - tr(d)));
      datum     = std::max(datum, std::abs(tr(d)));
   }
   INFO("boundary trace dofs " << bdr_dofs.Size() << " of " << nat.t.Size()
        << ", datum magnitude " << datum);
   REQUIRE(datum > 0.1);
   REQUIRE(nat_bdr == 0.0);
   REQUIRE(ess_drift < 1e-12 * datum);

   // And the two routes are the same discrete method, not merely comparable
   // ones: for RT_k the normal trace on a face is a polynomial of degree k and
   // the trace space is of that degree, so eliminating the essential trace
   // reproduces the natural term <p_D, v.n> exactly.
   Vector du(ess.u), dp(ess.p);
   du -= nat.u;
   dp -= nat.p;
   REQUIRE(du.Normlinf() < 1e-10 * std::max(nat.u.Normlinf(), real_t(1.0)));
   REQUIRE(dp.Normlinf() < 1e-10 * std::max(nat.p.Normlinf(), real_t(1.0)));
   REQUIRE(ess.err_u == MFEM_Approx(nat.err_u, 1e-12, 1e-10));
   REQUIRE(ess.err_p == MFEM_Approx(nat.err_p, 1e-12, 1e-10));

   // The marker's integrator is not assembled: changing it changes nothing.
   const Result other = Solve(mesh, order, true, Form::RT, 0.5, {}, true, 7.0);
   Vector d2(other.u);
   d2 -= ess.u;
   REQUIRE(d2.Normlinf() < 1e-12 * std::max(ess.u.Normlinf(), real_t(1.0)));
}

TEST_CASE("A trace richer than both its elements is exactly redundant",
          "[DarcyHybridization]")
{
   using namespace darcy_hybridization;

   // Raising the trace degree above the element degree is what p-adaptivity's
   // usual rule -- p_F = max over the two neighbours -- produces on a face
   // between elements of different degree. On a face whose two neighbours are
   // the SAME degree the extra trace modes buy nothing, and this says so
   // exactly rather than approximately: the trace equation for a mode that is
   // L2-orthogonal to everything the elements can put on the face reads
   // -(tau_1 + tau_2) <uhat, mu> = 0, so that mode is annihilated, and the
   // remaining discrete problem is the equal-order one unchanged.
   //
   // That is the measurement behind choosing p_F = min for a p-adaptive trace:
   // max costs dofs and, wherever the two neighbours agree, returns nothing.
   // It is not an argument against max at a genuine p-interface, where the
   // richer element does reach the extra modes; only against paying for it
   // where the degrees already match.
   //
   // This could not be run at all until the HDG face quadrature took the trace
   // element's order into account -- before that the trace-trace block was
   // rank-deficient by one per face and the reduced system was singular. See
   // the note at the top of fem/darcy/bilininteg_hdg.cpp.
   const int order = GENERATE(0, 1, 2);
   const int nx = 4;
   CAPTURE(order);

   Mesh mesh = Mesh::MakeCartesian2D(nx, nx, Element::QUADRILATERAL, false,
                                     1.0, 1.0);

   const Result base = Solve(mesh, order, true, Form::DG);
   const Result rich = Solve(mesh, order, true, Form::DG, 0.5, {}, false, -2.0,
                             order + 1);

   // The richer space really is bigger, or the comparison below is vacuous.
   REQUIRE(rich.solved_size > base.solved_size);

   INFO("errors " << base.err_u << " / " << base.err_p << " against "
        << rich.err_u << " / " << rich.err_p);
   REQUIRE(rich.err_u == Approx(base.err_u).margin(1e-12).epsilon(1e-10));
   REQUIRE(rich.err_p == Approx(base.err_p).margin(1e-12).epsilon(1e-10));

   // and the recovered fields agree, not merely their error norms
   Vector du(rich.u);
   du -= base.u;
   Vector dp(rich.p);
   dp -= base.p;
   REQUIRE(du.Normlinf() < 1e-10 * std::max(base.u.Normlinf(), real_t(1.0)));
   REQUIRE(dp.Normlinf() < 1e-10 * std::max(base.p.Normlinf(), real_t(1.0)));
}
