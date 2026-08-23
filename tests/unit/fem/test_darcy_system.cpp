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

namespace darcy_system
{

// Several Darcy-like problems solved as one system, in the arrangement
// miniapps/plasma/braginskii_hdg.cpp uses: spaces of vdim = neq, and scalar
// integrators replicated across the equations by
// VectorBlockDiagonalIntegrator.
//
//    k_i^{-1} u_i + grad p_i = 0,     -div u_i = g_i,      i = 1 .. neq
//
// with the natural boundary condition -p_i, and k_i a different constant for
// each equation so that a mistake mixing the blocks up cannot cancel.

constexpr int NEQ = 2;

const real_t kk[NEQ]  = {1.0, 0.4};   // conductivities
const real_t amp[NEQ] = {1.0, -0.6};  // solution amplitudes

// Cross-equation coupling, added to the potential mass block as
// VectorMassIntegrator with a matrix coefficient. Deliberately non-symmetric
// and with both off-diagonal entries nonzero, so neither a transposed block
// nor a dropped one can pass unnoticed.
const real_t QQ[NEQ][NEQ] = {{0.7, 0.3}, {-0.2, 0.5}};

bool coupled = false;

// The sign with which the potential mass block enters the assembled system.
// DarcyForm's default is bsym = true, whose second row is -B u - Mp p. The
// value was fixed by measurement rather than read off the convention: with the
// opposite sign the order study collapses to rate 0.53 instead of 2, which is
// also the control showing this test is not vacuous. Note the hybridization
// check in the same test case passes either way, since equivalence between two
// solves of the same operator says nothing about which operator it is.
constexpr real_t MP_SIGN = -1.0;

real_t pComp(const Vector &x, int i)
{
   real_t p = amp[i];
   for (int d = 0; d < x.Size(); d++) { p *= std::sin(M_PI * x(d)); }
   return p;
}

void pExact(const Vector &x, Vector &p)
{
   p.SetSize(NEQ);
   for (int i = 0; i < NEQ; i++) { p(i) = pComp(x, i); }
}

/// The flux of equation i, u_i = -k_i grad p_i, laid out as the RT space of
/// vdim = NEQ expects: component i occupies entries [i*dim, (i+1)*dim).
void uExact(const Vector &x, Vector &u)
{
   const int dim = x.Size();
   u.SetSize(NEQ * dim);
   for (int i = 0; i < NEQ; i++)
   {
      for (int j = 0; j < dim; j++)
      {
         real_t g = M_PI * amp[i];
         for (int d = 0; d < dim; d++)
         {
            g *= (d == j) ? std::cos(M_PI * x(d)) : std::sin(M_PI * x(d));
         }
         u(i * dim + j) = -kk[i] * g;
      }
   }
}

// -div u_i = k_i laplace p_i = -k_i dim pi^2 p_i, plus the coupling term.
void gExact(const Vector &x, Vector &g)
{
   const int dim = x.Size();
   g.SetSize(NEQ);
   for (int i = 0; i < NEQ; i++)
   {
      g(i) = -kk[i] * dim * M_PI * M_PI * pComp(x, i);
      if (coupled)
      {
         for (int j = 0; j < NEQ; j++)
         {
            g(i) += MP_SIGN * QQ[i][j] * pComp(x, j);
         }
      }
   }
}

void pNatural(const Vector &x, Vector &v)
{
   pExact(x, v);
   v.Neg();
}

struct Result
{
   Vector u, p;
   real_t err_p, err_u;
   int    solved_size;
};

/// Solve the neq-equation system. With neq = 1 this is the ordinary single
/// field problem, which is what the per-equation comparison below leans on.
Result Solve(Mesh &mesh, int order, int neq, bool hybridize, int only = -1)
{
   const int dim = mesh.Dimension();

   RT_FECollection u_coll(order, dim);
   L2_FECollection p_coll(order, dim);
   FiniteElementSpace fes_u(&mesh, &u_coll, neq, Ordering::byNODES);
   FiniteElementSpace fes_p(&mesh, &p_coll, neq, Ordering::byNODES);

   DarcyForm darcy(&fes_u, &fes_p);

   std::vector<BilinearFormIntegrator *> mass(neq);
   std::vector<Coefficient *> ik(neq);
   for (int i = 0; i < neq; i++)
   {
      const int eq = (only >= 0) ? only : i;
      ik[i] = new ConstantCoefficient(1.0 / kk[eq]);
      mass[i] = new VectorFEMassIntegrator(*ik[i]);
   }
   darcy.GetFluxMassForm()->AddDomainIntegrator(
      new VectorBlockDiagonalIntegrator(mass));
   darcy.GetFluxDivForm()->AddDomainIntegrator(
      new VectorBlockDiagonalIntegrator(neq, new VectorFEDivergenceIntegrator));

   // The coupling. VectorBlockDiagonalIntegrator cannot express this: it
   // replicates one integrator down the diagonal, and what is wanted here is a
   // dense neq x neq block. A matrix coefficient on the potential mass form is
   // the only route the branch offers for linear cross-equation coupling.
   DenseMatrix Qm(neq);
   Qm = 0.0;
   if (coupled && only < 0)
   {
      for (int i = 0; i < neq; i++)
         for (int j = 0; j < neq; j++)
         {
            Qm(i, j) = QQ[i][j];
         }
   }
   MatrixConstantCoefficient Qcoeff(Qm);
   if (coupled && only < 0)
   {
      darcy.GetPotentialMassForm()->AddDomainIntegrator(
         new VectorMassIntegrator(Qcoeff));
   }

   // Right-hand sides, as vector coefficients over the equations.
   auto nat = [only](const Vector & x, Vector & v)
   {
      Vector all;
      pNatural(x, all);
      if (only >= 0) { v.SetSize(1); v(0) = all(only); }
      else { v = all; }
   };
   auto src = [only](const Vector & x, Vector & v)
   {
      Vector all;
      gExact(x, all);
      if (only >= 0) { v.SetSize(1); v(0) = all(only); }
      else { v = all; }
   };
   VectorFunctionCoefficient natcoeff(neq, nat), gcoeff(neq, src);

   darcy.GetFluxRHS()->AddBoundaryIntegrator(
      new VectorFEBoundaryFluxLFIntegrator(natcoeff));
   darcy.GetPotentialRHS()->AddDomainIntegrator(
      new VectorDomainLFIntegrator(gcoeff));

   Array<int> ess_flux_tdofs;
   DG_Interface_FECollection trace_coll(order, dim);
   FiniteElementSpace fes_t(&mesh, &trace_coll, neq, Ordering::byNODES);
   if (hybridize)
   {
      darcy.EnableHybridization(
         &fes_t,
         new VectorBlockDiagonalIntegrator(neq, new NormalTraceJumpIntegrator),
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
      solver.SetKDim(2000);
      solver.SetMaxIter(20000);
      solver.SetRelTol(0.0);
      solver.SetAbsTol(1e-13);
      solver.SetPreconditioner(prec);
      solver.SetOperator(*A);
      solver.Mult(B, X);
      REQUIRE(solver.GetConverged());
   }
   else
   {
      MINRESSolver solver;
      solver.SetMaxIter(50000);
      solver.SetRelTol(0.0);
      solver.SetAbsTol(1e-13);
      solver.SetOperator(*A);
      solver.Mult(B, X);
      REQUIRE(solver.GetConverged());
   }

   darcy.RecoverFEMSolution(X, x);

   GridFunction u_h(&fes_u, x.GetBlock(0));
   GridFunction p_h(&fes_p, x.GetBlock(1));

   const int quad_order = 2 * order + 4;
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; i++)
   {
      irs[i] = &(IntRules.Get(i, quad_order));
   }

   auto pfun = [only](const Vector & x, Vector & v)
   {
      Vector all;
      pExact(x, all);
      if (only >= 0) { v.SetSize(1); v(0) = all(only); }
      else { v = all; }
   };
   auto ufun = [only, dim](const Vector & x, Vector & v)
   {
      Vector all;
      uExact(x, all);
      if (only >= 0)
      {
         v.SetSize(dim);
         for (int d = 0; d < dim; d++) { v(d) = all(only * dim + d); }
      }
      else { v = all; }
   };
   VectorFunctionCoefficient pcoeff(neq, pfun), ucoeff(neq * dim, ufun);

   res.err_p = p_h.ComputeL2Error(pcoeff, irs);
   res.err_u = u_h.ComputeL2Error(ucoeff, irs);
   res.u = x.GetBlock(0);
   res.p = x.GetBlock(1);
   for (int i = 0; i < neq; i++) { delete ik[i]; }
   return res;
}

} // namespace darcy_system

TEST_CASE("A block-diagonal Darcy system reproduces its equations one by one",
          "[DarcyForm][DarcyHybridization][System]")
{
   using namespace darcy_system;

   // The first question about "systems of equations": does the vdim > 1 path
   // solve each equation as it would be solved alone? The conductivities and
   // amplitudes differ per equation, so a block mix-up cannot cancel.
   const int order = GENERATE(0, 1);
   const Element::Type elem = GENERATE(Element::QUADRILATERAL,
                                       Element::TRIANGLE);

   Mesh mesh = Mesh::MakeCartesian2D(4, 4, elem, false, 1.0, 1.0);

   const Result sys = Solve(mesh, order, NEQ, true);

   CAPTURE(order, int(elem), sys.solved_size);

   const int np = sys.p.Size() / NEQ;
   const int nu = sys.u.Size() / NEQ;

   for (int i = 0; i < NEQ; i++)
   {
      const Result one = Solve(mesh, order, 1, true, i);
      REQUIRE(one.p.Size() == np);
      REQUIRE(one.u.Size() == nu);

      for (int k = 0; k < np; k++)
      {
         CAPTURE(i, k);
         REQUIRE(sys.p(i * np + k) == MFEM_Approx(one.p(k), 1e-9, 1e-8));
      }
      for (int k = 0; k < nu; k++)
      {
         CAPTURE(i, k);
         REQUIRE(sys.u(i * nu + k) == MFEM_Approx(one.u(k), 1e-9, 1e-8));
      }
   }
}

TEST_CASE("A Darcy system hybridizes to the same solution as the block solve",
          "[DarcyForm][DarcyHybridization][System]")
{
   using namespace darcy_system;

   const int order = GENERATE(0, 1);

   Mesh mesh = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL, false,
                                     1.0, 1.0);

   const Result mono = Solve(mesh, order, NEQ, false);
   const Result hyb  = Solve(mesh, order, NEQ, true);

   CAPTURE(order, mono.solved_size, hyb.solved_size);
   REQUIRE(hyb.solved_size < mono.solved_size);

   Vector du(hyb.u), dp(hyb.p);
   du -= mono.u;
   dp -= mono.p;
   REQUIRE(du.Normlinf() < 1e-8 * std::max(mono.u.Normlinf(), real_t(1.0)));
   REQUIRE(dp.Normlinf() < 1e-8 * std::max(mono.p.Normlinf(), real_t(1.0)));
}

TEST_CASE("A Darcy system converges at the design order",
          "[DarcyForm][DarcyHybridization][System]")
{
   using namespace darcy_system;

   const int order = GENERATE(0, 1);

   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, false,
                                     1.0, 1.0);

   real_t prev_p = -1.0, prev_u = -1.0;
   for (int ref = 0; ref < 3; ref++)
   {
      const Result r = Solve(mesh, order, NEQ, true);
      if (prev_p > 0.0)
      {
         const real_t rate_p = std::log2(prev_p / r.err_p);
         const real_t rate_u = std::log2(prev_u / r.err_u);
         CAPTURE(order, ref, rate_p, rate_u);
         REQUIRE(rate_p > order + 0.7);
         REQUIRE(rate_u > order + 0.7);
      }
      prev_p = r.err_p;
      prev_u = r.err_u;
      mesh.UniformRefinement();
   }
}

TEST_CASE("A Darcy system with cross-equation coupling",
          "[DarcyForm][DarcyHybridization][System]")
{
   using namespace darcy_system;

   // Block-diagonal replication is not a system in any interesting sense: the
   // equations never speak to each other. This adds a dense neq x neq
   // zeroth-order block, which is the only linear cross-equation coupling the
   // branch can express, and is the shape the potential cascade needs.
   const int order = GENERATE(0, 1);
   coupled = true;

   SECTION("hybridization is still exact")
   {
      Mesh mesh = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL, false,
                                        1.0, 1.0);
      const Result mono = Solve(mesh, order, NEQ, false);
      const Result hyb  = Solve(mesh, order, NEQ, true);

      CAPTURE(order, mono.solved_size, hyb.solved_size);

      Vector du(hyb.u), dp(hyb.p);
      du -= mono.u;
      dp -= mono.p;
      REQUIRE(du.Normlinf() < 1e-8 * std::max(mono.u.Normlinf(), real_t(1.0)));
      REQUIRE(dp.Normlinf() < 1e-8 * std::max(mono.p.Normlinf(), real_t(1.0)));
   }

   SECTION("and the coupled system converges at the design order")
   {
      Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, false,
                                        1.0, 1.0);
      real_t prev_p = -1.0, prev_u = -1.0;
      for (int ref = 0; ref < 3; ref++)
      {
         const Result r = Solve(mesh, order, NEQ, true);
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

   coupled = false;
}
