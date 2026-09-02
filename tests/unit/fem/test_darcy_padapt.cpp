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
   int    size;      ///< size of the trace system, storage included
   int    ess;       ///< essential trace dofs, the retired surplus included
   int    active() const { return size - ess; }   ///< what is actually solved
};

/// @a set says whether SetTraceOrders() is called at all, which is what
/// separates "never configured" from "configured with an empty array".
Result Solve(int order, int n, const Array<int> &trace_orders, bool set = true,
             int trace_ceiling = -1)
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
   DG_Interface_FECollection trace_coll(
      (trace_ceiling < 0) ? order : trace_ceiling, dim);
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
   res.ess = darcy.GetHybridization()->GetEssentialTrueDofs().Size();
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

TEST_CASE("A raised ceiling with every face at the old degree solves the old problem",
          "[DarcyHybridization][PAdapt]")
{
   using namespace darcy_padapt;

   // The acceptance test for retiring the surplus, and the one that makes the
   // ceiling usable at all. Build the constraint space one degree above the
   // elements and put every face back at the element degree: every face then
   // carries surplus slots, so this exercises the retirement everywhere at
   // once -- and the active space is face-for-face the one the plain uniform
   // run uses, with the same basis, so it must return the same solution.
   //
   // Same solution, not the same vector: the trace is embedded in bigger
   // storage under a different global numbering, so only the recovered fields
   // are comparable, and only to the linear solver's tolerance rather than
   // bitwise.
   const int order = GENERATE(0, 1, 2);
   const int n = 4;
   CAPTURE(order);

   Array<int> none;
   const Result plain = Solve(order, n, none, false);

   Mesh probe = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, false,
                                      1.0, 1.0);
   const int nfaces = probe.GetNumFaces();
   Array<int> at_old(nfaces);
   at_old = order;
   const Result raised = Solve(order, n, at_old, true, order + 1);

   // The storage really did grow, or the surplus is not being exercised.
   REQUIRE(raised.size > plain.size);
   REQUIRE(raised.ess > plain.ess);

   // ...and what is left after retiring it is exactly the old system's size.
   INFO("active " << raised.active() << " against " << plain.active());
   REQUIRE(raised.active() == plain.active());

   auto close = [](const Vector &a, const Vector &b, const char *what)
   {
      REQUIRE(a.Size() == b.Size());
      Vector d(a);
      d -= b;
      INFO(what << ": max difference " << d.Normlinf() << " on "
           << b.Normlinf());
      CHECK(d.Normlinf() < 1e-10 * std::max(b.Normlinf(), real_t(1.0)));
   };

   close(raised.q, plain.q, "flux");
   close(raised.p, plain.p, "potential");
}

TEST_CASE("A genuinely non-uniform trace solves, and its size is the sum of its faces",
          "[DarcyHybridization][PAdapt]")
{
   using namespace darcy_padapt;

   // Degrees that actually differ face to face. The arithmetic is the check
   // worth having: with no essential boundary trace condition on this problem,
   // every trace dof is free except the retired surplus, so the number of dofs
   // actually solved for must come to sum over faces of nt(p_f) -- here p_f+1,
   // the faces being segments. That is a statement about the whole machinery
   // (degrees stored, elements chosen, slots retired) and it is exact.
   const int order = GENERATE(1, 2);
   const int n = 4;
   CAPTURE(order);

   Mesh probe = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, false,
                                      1.0, 1.0);
   const int nfaces = probe.GetNumFaces();

   Array<int> mixed(nfaces);
   int expected = 0;
   for (int f = 0; f < nfaces; f++)
   {
      mixed[f] = (f % 2 == 0) ? order : (order - 1);
      expected += mixed[f] + 1;          // nt(p) on a segment
   }

   const Result r = Solve(order, n, mixed);

   INFO("active " << r.active() << ", expected " << expected);
   REQUIRE(r.active() == expected);

   // and it is a genuinely different discretisation, not a relabelling
   Array<int> none;
   const Result uniform = Solve(order, n, none, false);
   REQUIRE(r.active() < uniform.active());
   Vector d(r.p);
   d -= uniform.p;
   REQUIRE(d.Normlinf() > 1e-12);
}

TEST_CASE("Postprocessing enriches a p-adapted potential element by element",
          "[DarcyHybridization][PAdapt]")
{
   // The classic local postprocessing is the reconstruction a per-face trace
   // degree cannot disturb: it reads the flux and the potential on the element
   // it is working on and nothing else. What it *can* get wrong is its own
   // enriched space -- built one degree above the collection, it would sit at
   // the wrong degree on every element a p-adapted potential has moved.
   //
   // No solve here on purpose. The flux is handed the exact gradient, so the
   // local problem's data is exact and any error left is the enrichment's own.
   const int order = 1;
   const int n = 4;

   Mesh mesh = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, false,
                                     1.0, 1.0);
   mesh.EnsureNCMesh();
   const int dim = mesh.Dimension();

   L2_FECollection q_coll(order + 1, dim), p_coll(order, dim);
   FiniteElementSpace fes_q(&mesh, &q_coll, dim);
   FiniteElementSpace fes_p(&mesh, &p_coll);

   // Half the elements one degree up, the half a driver would have marked.
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      Vector c;
      mesh.GetElementCenter(e, c);
      if (c(0) < 0.5)
      {
         fes_p.SetElementOrder(e, order + 1);
         fes_q.SetElementOrder(e, order + 2);
      }
   }
   fes_p.Update(false);
   fes_q.Update(false);
   REQUIRE(fes_p.IsVariableOrder());

   auto pfun = [](const Vector &x)
   { return sin(M_PI * x(0)) * sin(M_PI * x(1)); };
   FunctionCoefficient pcoeff(pfun);
   VectorFunctionCoefficient qcoeff(dim, [](const Vector &x, Vector &q)
   {
      q(0) = -M_PI * cos(M_PI * x(0)) * sin(M_PI * x(1));
      q(1) = -M_PI * sin(M_PI * x(0)) * cos(M_PI * x(1));
   });

   GridFunction q_h(&fes_q), p_h(&fes_p);
   q_h.ProjectCoefficient(qcoeff);
   p_h.ProjectCoefficient(pcoeff);

   HDGPotentialPostprocessor pp(q_h, p_h);
   GridFunction p_s;
   pp.Compute(p_s);

   // The enrichment followed the potential rather than the collection.
   REQUIRE(p_s.FESpace()->IsVariableOrder());
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      INFO("element " << e);
      REQUIRE(p_s.FESpace()->GetElementOrder(e) ==
              fes_p.GetElementOrder(e) + 1);
   }

   // ...and it is worth having: one degree above a projection beats it.
   const int quad = 2 * (order + 3) + 2;
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; i++)
   {
      irs[i] = &(IntRules.Get(i, quad));
   }
   const real_t err_p = p_h.ComputeL2Error(pcoeff, irs);
   const real_t err_s = p_s.ComputeL2Error(pcoeff, irs);
   INFO("computed " << err_p << ", postprocessed " << err_s);
   REQUIRE(err_s < 0.5 * err_p);
}

TEST_CASE("The smoothness sensor reads the top degree's share of the energy",
          "[DarcyHybridization][PAdapt]")
{
   // Persson & Peraire eq (7), which is the other half of an hp decision: an
   // error estimator says where to spend, this says whether to spend it on h
   // or on p.
   const int dim = 2;
   const int n = 4;
   Mesh mesh = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, false,
                                     1.0, 1.0);

   SECTION("a function already of degree p-1 senses as perfectly smooth")
   {
      // The sharpest check available: if u is in P_{p-1} then the truncation
      // is u itself and eq (7)'s numerator is *mathematically* zero. It is not
      // numerically zero, and the tolerance is what says which -- the
      // implementation forms |u|^2 - |Pu|^2 rather than |u - Pu|^2, leaning on
      // the orthogonality of the projection, and a difference of two nearly
      // equal norms floors at round-off however exact the algebra is. What
      // this pins is that the floor is round-off and not a method error.
      const int order = GENERATE(1, 2, 3);
      CAPTURE(order);

      L2_FECollection fec(order, dim);
      FiniteElementSpace fes(&mesh, &fec);
      GridFunction u(&fes);

      // Degree order-1 exactly, and not a constant, so it is not smooth by
      // accident.
      FunctionCoefficient c([order](const Vector &x)
      { return pow(x(0) + 0.3 * x(1), order - 1) + 0.5; });
      u.ProjectCoefficient(c);

      PerssonPeraireSmoothness sm(u);
      const Vector &S = sm.GetSensor();
      REQUIRE(S.Size() == mesh.GetNE());
      INFO("largest S_e " << S.Max());
      REQUIRE(S.Max() < 1e-12);
   }

   SECTION("a jump inside one element is sensed there and not elsewhere")
   {
      const int order = 3;
      L2_FECollection fec(order, dim);
      FiniteElementSpace fes(&mesh, &fec);
      GridFunction u(&fes);

      // A step at x = 0.4, which falls strictly inside the second column of
      // elements rather than on a mesh line -- on a line every element would
      // hold a smooth half of it and the sensor would be right to say so.
      FunctionCoefficient c([](const Vector &x)
      { return (x(0) < 0.4) ? 0.0 : 1.0; });
      u.ProjectCoefficient(c);

      PerssonPeraireSmoothness sm(u);
      const Vector &S = sm.GetSensor();

      real_t cut = 0.0, away = 0.0;
      for (int e = 0; e < mesh.GetNE(); e++)
      {
         Vector ctr;
         mesh.GetElementCenter(e, ctr);
         const bool crossed = (ctr(0) > 0.25 && ctr(0) < 0.5);
         if (crossed) { cut = std::max(cut, S(e)); }
         else { away = std::max(away, S(e)); }
      }
      INFO("cut " << cut << ", away " << away);
      REQUIRE(cut > 1e3 * away);
   }

   SECTION("it follows a variable order rather than the collection's")
   {
      mesh.EnsureNCMesh();
      const int order = 1;
      L2_FECollection fec(order, dim);
      FiniteElementSpace fes(&mesh, &fec);
      for (int e = 0; e < mesh.GetNE(); e++)
      {
         Vector ctr;
         mesh.GetElementCenter(e, ctr);
         if (ctr(0) < 0.5) { fes.SetElementOrder(e, order + 2); }
      }
      fes.Update(false);
      REQUIRE(fes.IsVariableOrder());

      GridFunction u(&fes);
      // Degree 2: inside P_{p-1} on the raised elements (p = 3), and not on
      // the others (p = 1), so the sensor has to be reading each element's
      // own degree to tell them apart.
      FunctionCoefficient c([](const Vector &x)
      { return x(0) * x(0) + 0.4 * x(1) * x(1) + 0.1; });
      u.ProjectCoefficient(c);

      PerssonPeraireSmoothness sm(u);
      const Vector &S = sm.GetSensor();
      for (int e = 0; e < mesh.GetNE(); e++)
      {
         Vector ctr;
         mesh.GetElementCenter(e, ctr);
         INFO("element " << e << " order " << fes.GetElementOrder(e)
              << " S_e " << S(e));
         if (ctr(0) < 0.5) { REQUIRE(S(e) < 1e-12); }
         else              { REQUIRE(S(e) > 1e-6); }
      }
   }
}

TEST_CASE("A hanging-node family takes the ceiling degree",
          "[DarcyHybridization][PAdapt]")
{
   // A family's members reach one shared trace unknown through
   // FiniteElement::GetTransferMatrix() and, one level below that, through the
   // constraint space's conforming prolongation -- which interpolates in the
   // CEILING basis and knows nothing of a per-face degree. So the family has
   // to sit at the ceiling, and the derivation rule is what has to put it
   // there; a family below it solves a different problem and, at the degrees
   // this branch allows, an unstable one.
   Mesh mesh = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL, false,
                                     1.0, 1.0);
   mesh.EnsureNCMesh();

   Array<Refinement> refs;
   refs.Append(Refinement(5));
   mesh.GeneralRefinement(refs, 1, 0);

   const int cap = 5;
   Array<int> elem_order(mesh.GetNE());
   elem_order = 2;

   Array<int> face_order;
   DarcyHybridization::FaceOrdersFromElementOrders(
      mesh, elem_order, DarcyHybridization::TraceOrderRule::Min, cap,
      face_order);

   REQUIRE(face_order.Size() == mesh.GetNumFaces());

   // Mark the faces the NC list knows about, so the assertion below is about
   // every family member rather than about the ones that happened to be found.
   Array<int> in_family(mesh.GetNumFaces());
   in_family = 0;
   const NCMesh::NCList &nclist = mesh.ncmesh->GetNCList(mesh.Dimension() - 1);
   for (int m = 0; m < nclist.masters.Size(); m++)
   {
      const NCMesh::Master &master = nclist.masters[m];
      if (master.index < 0 || master.index >= mesh.GetNumFaces()) { continue; }
      in_family[master.index] = 1;
      for (int s = master.slaves_begin; s < master.slaves_end; s++)
      {
         const int sf = nclist.slaves[s].index;
         if (sf >= 0 && sf < mesh.GetNumFaces()) { in_family[sf] = 1; }
      }
   }

   // Refining one quad of sixteen has to produce hanging nodes, or the test
   // asserts nothing.
   REQUIRE(in_family.Sum() > 0);

   for (int f = 0; f < face_order.Size(); f++)
   {
      CAPTURE(f, in_family[f]);
      REQUIRE(face_order[f] == (in_family[f] ? cap : 2));
   }
}
