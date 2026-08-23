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

Result Solve(Mesh &mesh, int order)
{
   const int dim = mesh.Dimension();

   RT_FECollection u_coll(order, dim);
   L2_FECollection p_coll(order, dim);
   FiniteElementSpace fes_u(&mesh, &u_coll);
   FiniteElementSpace fes_p(&mesh, &p_coll);

   FunctionCoefficient ikcoeff(ikFun);
   FunctionCoefficient gcoeff(gExact), natcoeff(pNatural), pcoeff(pExact);
   VectorFunctionCoefficient ucoeff(dim, uExact);

   DarcyForm darcy(&fes_u, &fes_p);
   darcy.GetFluxMassForm()->AddDomainIntegrator(
      new VectorFEMassIntegrator(ikcoeff));
   darcy.GetFluxDivForm()->AddDomainIntegrator(new VectorFEDivergenceIntegrator);

   LinearForm *fform = darcy.GetFluxRHS();
   fform->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(natcoeff));
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
           real_t &last_p, Element::Type type = Element::QUADRILATERAL)
{
   active = d;
   Mesh mesh = MakeMesh(type);

   real_t prev_p = -1.0, prev_u = -1.0;
   rate_p = rate_u = 0.0;
   for (int ref = 0; ref < 3; ref++)
   {
      const Result r = Solve(mesh, order);
      if (prev_p > 0.0)
      {
         rate_p = std::log2(prev_p / r.err_p);
         rate_u = std::log2(prev_u / r.err_u);
      }
      prev_p = r.err_p;
      prev_u = r.err_u;
      if (ref < 2) { mesh.UniformRefinement(); }
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
