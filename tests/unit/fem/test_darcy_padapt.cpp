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

namespace darcy_padapt
{

// An HDG Darcy problem on the unit square, discontinuous throughout:
//
//    q + grad p = 0,   div q = g
//
// solved hybridized, so the only globally coupled unknowns are the traces.
// The tests below never compare against an exact solution -- they compare two
// runs of the *same* discrete problem against each other -- so the source only
// has to make the problem well posed and non-trivial.

real_t gExact(const Vector &x)
{
   return 2.0 * M_PI * M_PI * sin(M_PI * x(0)) * sin(M_PI * x(1));
}

struct Result
{
   Vector tr;        ///< the trace solution, as solved for
   Vector q, p;      ///< the recovered flux and potential
   int    size;      ///< size of the trace system
};

/// @a set says whether SetTraceOrders() is called at all, which is what
/// separates "never configured" from "configured with an empty array".
Result Solve(int order, int n, const Array<int> &trace_orders, bool set = true)
{
   Mesh mesh = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, false,
                                     1.0, 1.0);
   const int dim = mesh.Dimension();

   L2_FECollection q_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes_q(&mesh, &q_coll, dim);
   FiniteElementSpace fes_p(&mesh, &p_coll);

   ConstantCoefficient one(1.0);
   FunctionCoefficient gcoeff(gExact);

   DarcyForm darcy(&fes_q, &fes_p);
   darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorMassIntegrator(one));

   MixedBilinearForm *B = darcy.GetFluxDivForm();
   B->AddDomainIntegrator(new VectorDivergenceIntegrator());
   B->AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));

   darcy.GetPotentialMassForm()->AddInteriorFaceIntegrator(
      new HDGDiffusionIntegrator(one, 0.5));
   darcy.GetPotentialRHS()->AddDomainIntegrator(
      new DomainLFIntegrator(gcoeff, 6, 12));

   Array<int> ess;
   DG_Interface_FECollection trace_coll(order, dim);
   FiniteElementSpace fes_t(&mesh, &trace_coll);
   darcy.EnableHybridization(&fes_t, new NormalTraceJumpIntegrator(), ess);

   if (set) { darcy.GetHybridization()->SetTraceOrders(trace_orders); }

   darcy.Assemble();

   BlockVector x(darcy.GetOffsets());
   x = 0.0;
   OperatorPtr A;
   Vector X, RHS;
   darcy.FormLinearSystem(ess, x, A, X, RHS, true);

   GSSmoother prec;
   GMRESSolver lin;
   lin.SetKDim(500);
   lin.SetMaxIter(5000);
   lin.SetRelTol(1e-14);
   lin.SetAbsTol(1e-16);
   lin.SetPreconditioner(prec);
   lin.SetOperator(*A);
   lin.Mult(RHS, X);
   REQUIRE(lin.GetConverged());

   darcy.RecoverFEMSolution(X, x);

   Result res;
   res.size = X.Size();
   res.tr = X;
   res.q = x.GetBlock(0);
   res.p = x.GetBlock(1);
   return res;
}

} // namespace darcy_padapt

TEST_CASE("Setting every trace order to the uniform one changes nothing",
          "[DarcyHybridization][PAdapt]")
{
   using namespace darcy_padapt;

   // The null test of the per-face trace machinery, and the one that has to
   // hold before any of it is worth anything: asking for the degree the space
   // already has must reproduce the uniform answer exactly, not nearly.
   //
   // It is not vacuous, because SetTraceOrders() with a non-empty array takes
   // the *other* branch of both accessors. TraceFE() goes through
   // FiniteElementCollection::GetFE(geom, p) instead of
   // FiniteElementSpace::GetFaceElement(f), and this is what says those two
   // agree; TraceVDofs() runs its nt_f == nt_max early return, which is what
   // says the truncation is a no-op at full degree rather than an off-by-one.
   const int order = GENERATE(0, 1, 2);
   const int n = 4;
   CAPTURE(order);

   Array<int> none;
   const Result uniform = Solve(order, n, none, false);

   Mesh probe = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, false,
                                      1.0, 1.0);
   Array<int> all_max(probe.GetNumFaces());
   all_max = order;
   const Result stated = Solve(order, n, all_max);

   REQUIRE(stated.size == uniform.size);

   auto same = [](const Vector &a, const Vector &b, const char *what)
   {
      REQUIRE(a.Size() == b.Size());
      Vector d(a);
      d -= b;
      INFO(what << ": max difference " << d.Normlinf() << " on "
           << b.Normlinf());
      REQUIRE(d.Normlinf() == 0.0);
   };

   same(stated.tr, uniform.tr, "trace");
   same(stated.q, uniform.q, "flux");
   same(stated.p, uniform.p, "potential");
}

TEST_CASE("An empty trace order array is the uniform trace",
          "[DarcyHybridization][PAdapt]")
{
   using namespace darcy_padapt;

   // SetTraceOrders({}) has to be a way back to the uniform space, not a way
   // to a zero-sized one -- the accessors read tr_order.Size() == 0 as "no
   // per-face degrees", so an empty array must clear rather than configure.
   const int order = 2;
   const int n = 4;

   Array<int> none;
   const Result a = Solve(order, n, none, false);

   Mesh probe = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, false,
                                      1.0, 1.0);
   Array<int> empty(0);
   const Result b = Solve(order, n, empty);

   REQUIRE(b.size == a.size);
   Vector d(b.tr);
   d -= a.tr;
   REQUIRE(d.Normlinf() == 0.0);
}
