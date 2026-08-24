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

namespace darcy_degenerate
{

// A diffusion coefficient that vanishes somewhere in the domain, solved in the
// mixed form
//
//    k^{-1} u + grad p = 0,   -div u = g,
//
// with the natural boundary condition -p = <given pressure>. The point of the
// test is that the mixed form assembles k^{-1}, so a k that vanishes puts the
// blow-up in the flux mass block. Whether that is tolerable is a question
// about integrability, and the answer differs by codimension:
//
//   * vanishing linearly at a *point* in 2D: the mass integrand goes like
//     1/r and the area element like r dr, so the entries stay finite;
//   * vanishing linearly along a *line*: the integrand goes like 1/y with no
//     compensating factor, and the entries diverge logarithmically.
//
// Both are measured here. Neither locus is ever sampled exactly, because
// interior quadrature points do not sit on element boundaries or vertices.

enum class Degeneracy { None, Point, Line };

/// RT is the hybridized mixed method, which carries no stabilization at all.
/// DG is the Nguyen-Peraire-Cockburn setting, where tau is built from the same
/// diffusion coefficient that vanishes -- which is the half of
/// HDG-REQUIREMENTS section 3(d) the RT measurements could not reach.
enum class Form { RT, DG };

/// A stabilization that ignores the coefficient the integrator built its value
/// from and returns a fixed number. With tau = T * kappa(x) / h the built-in
/// stabilization vanishes wherever the coefficient does, which is precisely
/// what section 3(d) means by a tau that misbehaves as w -> 0; this is the
/// alternative.
class FloorTau : public HDGStabilization
{
   real_t tau;
public:
   FloorTau(real_t t) : tau(t) { }
   real_t Eval(real_t, real_t, real_t, real_t,
               ElementTransformation &) const override
   { return tau; }
};

Degeneracy active = Degeneracy::None;

// The vanishing point is a vertex of every mesh used below, so the degeneracy
// sits exactly on the mesh rather than somewhere inside an element.
const real_t centre = 0.5;

real_t kFun(const Vector &x)
{
   switch (active)
   {
      case Degeneracy::Point:
      {
         real_t r2 = 0.0;
         for (int d = 0; d < x.Size(); d++)
         {
            const real_t dx = x(d) - centre;
            r2 += dx * dx;
         }
         return std::sqrt(r2);
      }
      case Degeneracy::Line:
         // A whole face of the domain in 3D, an edge in 2D.
         return x(x.Size() - 1);
      default:
         return 1.0;
   }
}

real_t ikFun(const Vector &x) { return 1.0 / kFun(x); }

real_t pExact(const Vector &x)
{
   real_t p = 1.0;
   for (int d = 0; d < x.Size(); d++) { p *= std::sin(M_PI * x(d)); }
   return p;
}

void GradP(const Vector &x, Vector &g)
{
   const int dim = x.Size();
   g.SetSize(dim);
   for (int i = 0; i < dim; i++)
   {
      real_t v = M_PI;
      for (int d = 0; d < dim; d++)
      {
         v *= (d == i) ? std::cos(M_PI * x(d)) : std::sin(M_PI * x(d));
      }
      g(i) = v;
   }
}

void uExact(const Vector &x, Vector &u)
{
   GradP(x, u);
   u *= -kFun(x);
}

// g = -div u = div(k grad p) = grad k . grad p + k laplace p, with
// laplace p = -2 pi^2 p.
real_t gExact(const Vector &x)
{
   Vector gp;
   GradP(x, gp);

   const int dim = x.Size();

   real_t gk_dot_gp = 0.0;
   switch (active)
   {
      case Degeneracy::Point:
      {
         const real_t r = kFun(x);
         if (r > 0.0)
         {
            for (int d = 0; d < dim; d++)
            {
               gk_dot_gp += (x(d) - centre) * gp(d);
            }
            gk_dot_gp /= r;
         }
         break;
      }
      case Degeneracy::Line:
         gk_dot_gp = gp(dim - 1);
         break;
      default:
         gk_dot_gp = 0.0;
   }

   return gk_dot_gp - dim * M_PI * M_PI * kFun(x) * pExact(x);
}

real_t pNatural(const Vector &x) { return -pExact(x); }

struct Result
{
   real_t err_p, err_u;
   int    solved_size;
   int    iterations;
};

Result Solve(Mesh &mesh, int order, Form form = Form::RT, real_t td = 0.1,
             int ubasis = BasisType::GaussLobatto,
             int pbasis = BasisType::GaussLegendre,
             int qbump = 0, const HDGStabilization *stab = nullptr)
{
   const int dim = mesh.Dimension();

   std::unique_ptr<FiniteElementCollection> u_coll;
   if (form == Form::DG)
   {
      u_coll.reset(new L2_FECollection(order, dim, ubasis));
   }
   else
   {
      u_coll.reset(new RT_FECollection(order, dim));
   }
   L2_FECollection p_coll(order, dim, pbasis);
   FiniteElementSpace fes_u(&mesh, u_coll.get(),
                            (form == Form::DG) ? dim : 1);
   FiniteElementSpace fes_p(&mesh, &p_coll);

   FunctionCoefficient ikcoeff(ikFun), kcoeff(kFun);
   FunctionCoefficient gcoeff(gExact), natcoeff(pNatural), pcoeff(pExact);
   VectorFunctionCoefficient ucoeff(dim, uExact);

   DarcyForm darcy(&fes_u, &fes_p);
   LinearForm *fform = darcy.GetFluxRHS();

   if (form == Form::DG)
   {
      // Optionally force a much higher quadrature rule, to test whether the
      // loss on a degenerate coefficient is an integration error.
      const IntegrationRule *eir = nullptr, *fir = nullptr;
      if (qbump > 0)
      {
         eir = &IntRules.Get(mesh.GetElementBaseGeometry(0),
                             2*order + qbump);
         fir = &IntRules.Get(mesh.GetFaceGeometry(0), 2*order + qbump);
      }

      auto *vmi = new VectorMassIntegrator(ikcoeff);
      if (eir) { vmi->SetIntRule(eir); }
      darcy.GetFluxMassForm()->AddDomainIntegrator(vmi);
      MixedBilinearForm *B = darcy.GetFluxDivForm();
      B->AddDomainIntegrator(new VectorDivergenceIntegrator());
      B->AddInteriorFaceIntegrator(
         new TransposeIntegrator(new DGNormalTraceIntegrator(-1.)));
      // tau is {h^-1 Q} with Q the diffusion coefficient, so where the
      // coefficient vanishes the stabilization vanishes with it. Whether that
      // is benign is the open half of section 3(d).
      auto *hdi = new HDGDiffusionIntegrator(kcoeff, td);
      if (fir) { hdi->SetIntRule(fir); }
      if (stab) { hdi->SetStabilization(*stab); }
      darcy.GetPotentialMassForm()->AddInteriorFaceIntegrator(hdi);

      fform->AddBdrFaceIntegrator(new VectorBoundaryFluxLFIntegrator(natcoeff));
   }
   else
   {
      darcy.GetFluxMassForm()->AddDomainIntegrator(
         new VectorFEMassIntegrator(ikcoeff));
      darcy.GetFluxDivForm()->AddDomainIntegrator(
         new VectorFEDivergenceIntegrator);

      fform->AddBoundaryIntegrator(
         new VectorFEBoundaryFluxLFIntegrator(natcoeff));
   }
   darcy.GetPotentialRHS()->AddDomainIntegrator(
      new DomainLFIntegrator(gcoeff, 6, 12));

   Array<int> ess_flux_tdofs;
   DG_Interface_FECollection trace_coll(order, dim);
   FiniteElementSpace fes_t(&mesh, &trace_coll);
   darcy.EnableHybridization(&fes_t, new NormalTraceJumpIntegrator(),
                             ess_flux_tdofs);

   darcy.Assemble();

   BlockVector x(darcy.GetOffsets());
   x = 0.0;

   OperatorPtr A;
   Vector X, B;
   darcy.FormLinearSystem(ess_flux_tdofs, x, A, X, B, true);

   GSSmoother prec;
   GMRESSolver solver;
   solver.SetKDim(2000);
   solver.SetMaxIter(20000);
   solver.SetRelTol(1e-13);
   solver.SetAbsTol(1e-14);
   solver.SetPreconditioner(prec);
   solver.SetOperator(*A);
   solver.Mult(B, X);
   REQUIRE(solver.GetConverged());

   darcy.RecoverFEMSolution(X, x);

   GridFunction u_h(&fes_u, x.GetBlock(0));
   GridFunction p_h(&fes_p, x.GetBlock(1));

   const int quad_order = 2 * order + 6;
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; i++)
   {
      irs[i] = &(IntRules.Get(i, quad_order));
   }

   Result res;
   res.err_p = p_h.ComputeL2Error(pcoeff, irs);
   res.err_u = u_h.ComputeL2Error(ucoeff, irs);
   res.solved_size = X.Size();
   res.iterations = solver.GetNumIterations();
   return res;
}

Mesh MakeMesh(Element::Type type)
{
   switch (type)
   {
      case Element::QUADRILATERAL:
         return Mesh::MakeCartesian2D(4, 4, type, false, 1.0, 1.0);
      case Element::TRIANGLE:
         return Mesh::MakeCartesian2D(4, 4, type, false, 1.0, 1.0);
      default:
         // Wedges and hexahedra: start coarser, since three refinements in 3D
         // is the expensive direction.
         return Mesh::MakeCartesian3D(2, 2, 2, type, 1.0, 1.0, 1.0);
   }
}

/// Refine three times and return the observed rates for the potential.
void Rates(int order, Degeneracy d, real_t &rate_p, real_t &rate_u,
           real_t &last_p, Element::Type type = Element::QUADRILATERAL,
           Form form = Form::RT, real_t T = 1.0, int nsolve = 3,
           bool fixed_tau = true, int ubasis = BasisType::GaussLobatto,
           int pbasis = BasisType::GaussLegendre, int qbump = 0,
           const HDGStabilization *stab = nullptr)
{
   active = d;
   Mesh mesh = MakeMesh(type);

   // tau = td * kappa / h in the integrator, so holding td fixed makes tau
   // grow like 1/h. NPC use a fixed tau, which needs td = T*h; these meshes
   // are uniform, so h is 1/n with n doubling each refinement.
   int n = (type == Element::QUADRILATERAL || type == Element::TRIANGLE)?(4):(2);
   real_t prev_p = -1.0, prev_u = -1.0;
   rate_p = rate_u = 0.0;
   for (int ref = 0; ref < nsolve; ref++)
   {
      const real_t td = fixed_tau ? (T / n) : T;
      const Result r = Solve(mesh, order, form, td, ubasis, pbasis, qbump,
                             stab);
      if (prev_p > 0.0)
      {
         rate_p = std::log2(prev_p / r.err_p);
         rate_u = std::log2(prev_u / r.err_u);
      }
      prev_p = r.err_p;
      prev_u = r.err_u;
      if (ref < nsolve - 1) { mesh.UniformRefinement(); n *= 2; }
   }
   last_p = prev_p;
   active = Degeneracy::None;
}

} // namespace darcy_degenerate

TEST_CASE("Hybridized Darcy with a diffusion coefficient vanishing at a point",
          "[DarcyForm][DarcyHybridization][Degenerate]")
{
   using namespace darcy_degenerate;

   // The quick check: k = |x - c| with c at a mesh vertex, so the coefficient
   // vanishes at one point of the domain and the inverse it is assembled
   // through is unbounded there. In two dimensions that inverse is still
   // integrable, so the order of accuracy should survive.
   const int order = GENERATE(0, 1, 2);

   real_t rate_p, rate_u, last_p;
   Rates(order, Degeneracy::Point, rate_p, rate_u, last_p);

   real_t base_p, base_u, base_last;
   Rates(order, Degeneracy::None, base_p, base_u, base_last);

   CAPTURE(order, rate_p, rate_u, base_p, base_u);

   REQUIRE(rate_p > order + 0.7);
   REQUIRE(rate_u > order + 0.7);
}

TEST_CASE("Hybridized Darcy with a diffusion coefficient vanishing on a line",
          "[DarcyForm][DarcyHybridization][Degenerate]")
{
   using namespace darcy_degenerate;

   // The case the application actually has: k vanishing linearly along a whole
   // boundary edge. Here the inverse is not integrable, so this is really the
   // singular case in disguise, and the measurement is the point.
   const int order = GENERATE(0, 1, 2);

   real_t rate_p, rate_u, last_p;
   Rates(order, Degeneracy::Line, rate_p, rate_u, last_p);

   CAPTURE(order, rate_p, rate_u, last_p);

   REQUIRE(rate_p > order + 0.7);
   REQUIRE(rate_u > order + 0.7);
}

TEST_CASE("Degenerate diffusion on wedges",
          "[DarcyForm][DarcyHybridization][Degenerate][Wedge]")
{
   using namespace darcy_degenerate;

   // The combination the application actually has: a coefficient that vanishes
   // on part of the domain, on the extruded prism mesh. Both halves are known
   // to work separately -- wedges in test_darcy_hybridization.cpp and the
   // degeneracies above in 2D -- so this is the check that they compose.
   //
   // The point case puts the zero at a single mesh vertex; the line case puts
   // it on the whole face x_2 = 0, which in the application's coordinates is
   // the mu = 0 axis where the collision tensor degenerates.
   const int order = launch_all_non_regression_tests ? GENERATE(0, 1, 2)
                     : GENERATE(0, 1);
   const Degeneracy d = GENERATE(Degeneracy::Point, Degeneracy::Line);

   real_t rate_p, rate_u, last_p;
   Rates(order, d, rate_p, rate_u, last_p, Element::WEDGE);

   real_t base_p, base_u, base_last;
   Rates(order, Degeneracy::None, base_p, base_u, base_last, Element::WEDGE);

   CAPTURE(order, int(d), rate_p, rate_u, base_p, base_u, last_p);

   REQUIRE(rate_p > order + 0.7);
   REQUIRE(rate_u > order + 0.7);
}

TEST_CASE("Degenerate diffusion on hexahedra",
          "[DarcyForm][DarcyHybridization][Degenerate]")
{
   using namespace darcy_degenerate;

   // The same in 3D on hexahedra, so that a wedge-only failure above would be
   // attributable to the element rather than to the extra dimension.
   const int order = launch_all_non_regression_tests ? GENERATE(0, 1, 2)
                     : GENERATE(0, 1);
   const Degeneracy d = GENERATE(Degeneracy::Point, Degeneracy::Line);

   real_t rate_p, rate_u, last_p;
   Rates(order, d, rate_p, rate_u, last_p, Element::HEXAHEDRON);

   CAPTURE(order, int(d), rate_p, rate_u, last_p);

   REQUIRE(rate_p > order + 0.7);
   REQUIRE(rate_u > order + 0.7);
}

// Section 3(d) asks for a tau that does not misbehave as the coefficient
// vanishes. Measured with tau held fixed under refinement, which is the NPC
// scaling (see the note in test_darcy_hybridization.cpp -- holding td fixed
// instead makes tau grow like 1/h and is a different method):
//
//   k  locus   rate_p at tau = 0.5, 1, 2, 4        rate_u
//   0  none     1.03  1.08  1.21  1.21              1.01  1.00  0.96  0.84
//   0  point    1.03  1.07  1.17  1.31              0.65  0.81  0.95  0.97
//   0  face     0.32  0.29  0.30  0.39              0.74  0.89  0.96  0.88
//   1  none     1.86  1.99  2.06  2.06              1.85  1.99  2.06  2.03
//   1  point    1.64  1.71  1.74  1.73              1.64  1.76  1.84  1.91
//   1  face     1.81  1.91  1.96  1.98              1.82  1.94  2.02  2.02
//   2  none     2.89  2.99  3.08  3.15              2.89  2.99  3.05  3.00
//   2  point    2.15  2.18  2.19  2.23              2.59  2.66  2.71  2.79
//   2  face     2.29  2.36  2.40  2.45              2.97  3.08  3.14  3.11
//
// **The degeneracy costs accuracy in the potential, and the cause is the
// stabilization, not the coefficient.**
//
// With the built-in stabilization the loss at k = 2 is about 0.8 of an order
// for the point degeneracy and 0.6 for the face. Three candidate explanations
// were measured:
//
//   nodal basis     Gauss-Lobatto, Gauss-Legendre and Bernstein give
//                   bit-identical results. They span the same space and MFEM
//                   chooses quadrature independently of the basis nodes, so
//                   moving nodes off the degenerate locus is a change of basis
//                   and not of method. The Gauss-Legendre-Radau trick belongs
//                   to collocation settings where the nodes are the quadrature
//                   points; here it is a no-op.
//
//   quadrature      raising the rule order by 14 moves the k = 2 point rate
//                   from 2.176 to 2.179. It is not an integration error.
//
//   tau             this is it. The integrator builds tau from the diffusion
//                   coefficient, tau = td * kappa(x) / h, so tau vanishes
//                   wherever kappa does and the face loses its stabilization
//                   exactly where the operator degenerates.
//
// Replacing it with a tau that does not vanish recovers the order completely:
//
//   k  locus   tau = kappa(x)   tau = 1   tau = 4
//   1  point       1.709         1.883     1.953
//   1  face        1.908         1.989     2.027
//   2  point       2.176         3.063     3.222
//   2  face        2.358         3.128     3.180
//
// and the errors fall by a factor of five to seven as well. So section 3(d)'s
// requirement of "a tau that does not misbehave as w -> 0" is real, and the
// misbehaviour is tau going to zero rather than blowing up. The remedy is
// neither weighted quadrature nor node placement, which is what that section
// originally proposed; it is a floor on the stabilization, which the
// HDGStabilization interface expresses in three lines.

TEST_CASE("HDG: a clean coefficient converges at k+1 on these meshes",
          "[DarcyForm][Degenerate][HDG]")
{
   using namespace darcy_degenerate;

   // The control the degenerate cases are measured against.
   const int order = GENERATE(1, 2);
   const real_t T = GENERATE(1.0, 4.0);

   real_t rate_p, rate_u, last_p;
   Rates(order, Degeneracy::None, rate_p, rate_u, last_p,
         Element::QUADRILATERAL, Form::DG, T, 3, true);

   CAPTURE(order, T, rate_p, rate_u);
   REQUIRE(rate_p > order + 0.7);
   REQUIRE(rate_u > order + 0.7);
}

TEST_CASE("HDG: a degenerate coefficient still converges",
          "[DarcyForm][Degenerate][HDG]")
{
   using namespace darcy_degenerate;

   // Weaker than the control on purpose: the rates above are what they are,
   // and asserting k+1 here would be asserting something false.
   const int order = GENERATE(1, 2);
   const real_t T = GENERATE(1.0, 4.0);
   const Degeneracy d = GENERATE(Degeneracy::Point, Degeneracy::Line);

   real_t rate_p, rate_u, last_p;
   Rates(order, d, rate_p, rate_u, last_p, Element::QUADRILATERAL,
         Form::DG, T, 3, true);

   CAPTURE(order, T, int(d), rate_p, rate_u);
   REQUIRE(rate_p > order + 0.1);
   REQUIRE(rate_u > order + 0.5);
}

TEST_CASE("HDG: the degeneracy costs potential accuracy at k = 2",
          "[DarcyForm][Degenerate][HDG]")
{
   using namespace darcy_degenerate;

   // The loss itself, pinned against a control on the same meshes. If a change
   // to the stabilization recovered the order, this breaks and the table above
   // gets corrected rather than quietly becoming wrong.
   const Degeneracy d = GENERATE(Degeneracy::Point, Degeneracy::Line);

   real_t deg_p, deg_u, deg_last, base_p, base_u, base_last;
   Rates(2, d, deg_p, deg_u, deg_last, Element::QUADRILATERAL,
         Form::DG, 1.0, 3, true);
   Rates(2, Degeneracy::None, base_p, base_u, base_last,
         Element::QUADRILATERAL, Form::DG, 1.0, 3, true);

   CAPTURE(int(d), deg_p, deg_u, base_p, base_u);

   REQUIRE(base_p > 2.7);              // the control is optimal
   REQUIRE(deg_p < base_p - 0.4);      // and the degenerate case is not
}

TEST_CASE("HDG: degenerate diffusion on wedges",
          "[DarcyForm][Degenerate][HDG][Wedge]")
{
   using namespace darcy_degenerate;

   const int order = GENERATE(0, 1);
   const Degeneracy d = GENERATE(Degeneracy::Point, Degeneracy::Line);

   real_t rate_p, rate_u, last_p;
   Rates(order, d, rate_p, rate_u, last_p, Element::WEDGE, Form::DG, 1.0,
         3, true);

   CAPTURE(order, int(d), rate_p, rate_u, last_p);
   REQUIRE(rate_p > order + 0.1);
}

TEST_CASE("HDG: a tau floor recovers the order a degeneracy costs",
          "[DarcyForm][Degenerate][HDG][Stabilization]")
{
   using namespace darcy_degenerate;

   // The finding above, as a test. Both halves matter: that the built-in
   // coefficient-scaled tau loses order, and that a floored tau does not.
   const Degeneracy d = GENERATE(Degeneracy::Point, Degeneracy::Line);

   real_t scaled_p, scaled_u, l1;
   Rates(2, d, scaled_p, scaled_u, l1, Element::QUADRILATERAL, Form::DG, 1.0,
         3, true);

   FloorTau floor(1.0);
   real_t floored_p, floored_u, l2;
   Rates(2, d, floored_p, floored_u, l2, Element::QUADRILATERAL, Form::DG, 1.0,
         3, true, BasisType::GaussLobatto, BasisType::GaussLegendre, 0, &floor);

   CAPTURE(int(d), scaled_p, scaled_u, floored_p, floored_u, l1, l2);

   REQUIRE(scaled_p < 2.6);        // the coefficient-scaled tau loses order
   REQUIRE(floored_p > 2.9);       // the floored one does not
   REQUIRE(l2 < 0.5 * l1);         // and is several times more accurate
}

TEST_CASE("HDG: the nodal basis is irrelevant to the degenerate case",
          "[DarcyForm][Degenerate][HDG]")
{
   using namespace darcy_degenerate;

   // Section 3(d) proposed moving nodes off the degenerate locus. It cannot
   // help: these bases span the same space and the quadrature does not depend
   // on them, so the discrete solution is the same function. Pinned so that
   // the proposal is not revisited.
   const Degeneracy d = GENERATE(Degeneracy::Point, Degeneracy::Line);
   const int basis = GENERATE(BasisType::GaussLegendre, BasisType::Positive);

   real_t lob_p, lob_u, l1;
   Rates(2, d, lob_p, lob_u, l1, Element::QUADRILATERAL, Form::DG, 1.0, 3,
         true, BasisType::GaussLobatto, BasisType::GaussLobatto);

   real_t alt_p, alt_u, l2;
   Rates(2, d, alt_p, alt_u, l2, Element::QUADRILATERAL, Form::DG, 1.0, 3,
         true, basis, basis);

   CAPTURE(int(d), basis, lob_p, alt_p, l1, l2);
   REQUIRE(alt_p == MFEM_Approx(lob_p, 1e-9, 1e-8));
   REQUIRE(l2 == MFEM_Approx(l1, 1e-12, 1e-10));
}
