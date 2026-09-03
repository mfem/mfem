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
   Vector tr_v;      ///< the same, prolonged onto the constraint space
   Vector q, p;      ///< the recovered flux and potential
   int    size;      ///< size of the trace system
   int    ess;       ///< essential trace dofs
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
   darcy.GetHybridization()->ProlongTrace(X, res.tr_v);
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
   // It is not vacuous: a non-empty array builds the constraint, so this is
   // what says an identity constraint really is one -- E square, R E = I, and
   // every face's block placed where the space put it.
   //
   /* "The same answer", not "the same vector", and to round-off rather than
      bitwise. Both weakenings are the same fact and it is worth stating
      rather than hiding in a tolerance.

      The constrained unknowns are ours and are numbered per face in the
      coarse element's own order, while the space numbers them in an order
      that carries the face orientation -- GetFaceVDofs() comes back as {5, 4}
      on a reversed face. At a uniform degree E is the identity, so the
      constraint is exactly that permutation: the reduced system is
      Pi^T H Pi with load Pi^T b, a symmetric permutation of the old one.
      Prolonging undoes the relabelling exactly, so the trace is comparable as
      a FUNCTION; but GMRES on a permuted system reaches the same answer only
      to round-off, so nothing here can be bitwise.

      "An empty trace order array is the uniform trace" below is still
      bitwise, and that is the pair: an empty array configures nothing at all,
      so there is not even a permutation. */
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
      REQUIRE(d.Normlinf() < 1e-13 * std::max(b.Normlinf(), real_t(1.0)));
   };

   same(stated.tr_v, uniform.tr_v, "trace");
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
   // Bitwise on the reduced vector here, unlike the test above: an empty
   // array configures nothing at all, so there is not even a relabelling.
}

TEST_CASE("A raised ceiling with every face at the old degree solves the old problem",
          "[DarcyHybridization][PAdapt]")
{
   using namespace darcy_padapt;

   /* The acceptance test for the ceiling, and the one that makes it usable at
      all. Build the constraint space one degree above the elements and put
      every face back at the element degree: every face then has a surplus, so
      this exercises the constraint everywhere at once -- and the constrained
      space is face-for-face the one the plain uniform run uses, so it must
      return the same solution.

      Same solution, not the same vector: the trace lives in bigger storage
      under a different numbering, so only the recovered fields are comparable,
      and only to the linear solver's tolerance rather than bitwise.

      AND THE REDUCED SYSTEM DOES NOT GROW. This used to assert the opposite
      -- that raising the ceiling grew the trace system and the essential list
      with it, the surplus being retired into unit rows. Constraining it
      instead means the ceiling costs nothing the solver ever sees: the same
      number of unknowns, none of them essential, and only the local blocks
      and the trace vector's storage follow the ceiling. */
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

   // The ceiling really is raised, or nothing is being exercised: the
   // constraint space carries more storage per face than the faces use.
   REQUIRE(raised.tr_v.Size() > plain.tr_v.Size());

   // And it costs the solver nothing at all.
   INFO("size " << raised.size << " against " << plain.size << ", essential "
        << raised.ess << " against " << plain.ess);
   REQUIRE(raised.size == plain.size);
   REQUIRE(raised.ess == plain.ess);
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

TEST_CASE("A hanging-node family takes ONE degree, and not the ceiling",
          "[DarcyHybridization][PAdapt]")
{
   /* A family carries one trace unknown, on its master face, so it has one
      degree -- the rule taken over its members. The master's own entry says
      nothing, a master face never being integrated directly.

      This asserted the opposite until the surplus was constrained rather than
      retired: the family had to sit AT THE CEILING, because the conforming
      prolongation interpolates master onto slave in the ceiling basis and the
      retired route's slots held a coarser basis's coefficients, so the two
      could not both hold on a face the prolongation touches. Storing ceiling
      coefficients removes the conflict, and with it the rule. What is left is
      the requirement that the family agree with itself, which is what this
      now checks -- along with the coarsening it was blocking. */
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

   // Every face is at the element degree, family members included, and the
   // ceiling of 5 is not reached anywhere: that is the coarsening the old
   // rule forbade.
   for (int f = 0; f < face_order.Size(); f++)
   {
      CAPTURE(f, in_family[f]);
      REQUIRE(face_order[f] == 2);
   }
   REQUIRE(cap > 2);

   // And a family agrees with itself, which is the one thing still required
   // of it: the master's degree is the unknown's, and a slave that disagreed
   // would size its local block against a basis nothing else uses.
   for (int m = 0; m < nclist.masters.Size(); m++)
   {
      const NCMesh::Master &master = nclist.masters[m];
      if (master.index < 0 || master.index >= mesh.GetNumFaces()) { continue; }
      for (int sl = master.slaves_begin; sl < master.slaves_end; sl++)
      {
         const int sf = nclist.slaves[sl].index;
         if (sf < 0 || sf >= mesh.GetNumFaces()) { continue; }
         CAPTURE(master.index, sf);
         REQUIRE(face_order[sf] == face_order[master.index]);
      }
   }
}

namespace darcy_padapt
{

/** @brief An HDG solve that keeps its spaces alive, so an estimator can be
    built on the solution afterwards. @a ceiling raises the constraint space's
    degree above @a order without changing what any face carries. */
struct Solved
{
   std::unique_ptr<Mesh> mesh;
   std::unique_ptr<L2_FECollection> q_coll, p_coll;
   std::unique_ptr<DG_Interface_FECollection> t_coll;
   std::unique_ptr<FiniteElementSpace> fes_q, fes_p, fes_t;
   std::unique_ptr<DarcyForm> darcy;
   OperatorPtr A;    ///< the reduced trace system
   BlockVector x;
   Vector X;         ///< the reduced trace solution
   GridFunction q_h, p_h, tr_h;
};

/// @a set_orders false leaves the trace uniform AT the ceiling, which is the
/// ceiling discretisation itself rather than a face-by-face coarsening of it.
void SolveKeeping(int order, int n, int ceiling, Solved &s,
                  bool set_orders = true, bool hanging = false)
{
   s.mesh.reset(new Mesh(Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL,
                                               false, 1.0, 1.0)));
   if (hanging)
   {
      s.mesh->EnsureNCMesh(true);
      Array<int> refs;
      refs.Append(0);
      s.mesh->GeneralRefinement(refs, -1, 0);
   }
   const int dim = s.mesh->Dimension();

   s.q_coll.reset(new L2_FECollection(order, dim, BasisType::GaussLobatto));
   s.p_coll.reset(new L2_FECollection(order, dim, BasisType::GaussLobatto));
   s.fes_q.reset(new FiniteElementSpace(s.mesh.get(), s.q_coll.get(), dim));
   s.fes_p.reset(new FiniteElementSpace(s.mesh.get(), s.p_coll.get()));

   static ConstantCoefficient one(1.0);
   static FunctionCoefficient gcoeff(gExact);

   s.darcy.reset(new DarcyForm(s.fes_q.get(), s.fes_p.get()));
   s.darcy->GetFluxMassForm()->AddDomainIntegrator(
      new VectorMassIntegrator(one));

   MixedBilinearForm *B = s.darcy->GetFluxDivForm();
   B->AddDomainIntegrator(new VectorDivergenceIntegrator());
   B->AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));

   s.darcy->GetPotentialMassForm()->AddInteriorFaceIntegrator(
      new HDGDiffusionIntegrator(one, 0.5));
   s.darcy->GetPotentialRHS()->AddDomainIntegrator(
      new DomainLFIntegrator(gcoeff, 6, 12));

   Array<int> ess;
   s.t_coll.reset(new DG_Interface_FECollection(ceiling, dim));
   s.fes_t.reset(new FiniteElementSpace(s.mesh.get(), s.t_coll.get()));
   s.darcy->EnableHybridization(s.fes_t.get(), new NormalTraceJumpIntegrator(),
                                ess);

   if (ceiling != order && set_orders)
   {
      Array<int> elem_order(s.mesh->GetNE());
      elem_order = order;
      Array<int> face_order;
      DarcyHybridization::FaceOrdersFromElementOrders(
         *s.mesh, elem_order, DarcyHybridization::TraceOrderRule::Min, ceiling,
         face_order);
      // Every face at the element degree, which is BELOW the ceiling: the
      // configuration whose answer must not depend on the ceiling.
      for (int f = 0; f < face_order.Size(); f++) { face_order[f] = order; }
      s.darcy->GetHybridization()->SetTraceOrders(face_order);
   }

   s.darcy->Assemble();

   s.x.Update(s.darcy->GetOffsets());
   s.x = 0.0;
   Vector RHS;
   s.darcy->FormLinearSystem(ess, s.x, s.A, s.X, RHS, true);

   GSSmoother prec;
   GMRESSolver lin;
   lin.SetKDim(500);
   lin.SetMaxIter(5000);
   lin.SetRelTol(1e-14);
   lin.SetAbsTol(1e-16);
   lin.SetPreconditioner(prec);
   lin.SetOperator(*s.A);
   lin.Mult(RHS, s.X);
   REQUIRE(lin.GetConverged());

   s.darcy->RecoverFEMSolution(s.X, s.x);

   s.q_h.MakeRef(s.fes_q.get(), s.x.GetBlock(0), 0);
   s.p_h.MakeRef(s.fes_p.get(), s.x.GetBlock(1), 0);

   // DarcyForm::GetOffsets() is the two field blocks; the trace is the reduced
   // unknown and comes back in X. The mesh here is conforming, so its true
   // dofs and its vdofs are the same numbering and this is a copy rather than
   // a prolongation.
   /* DarcyForm::GetOffsets() is the two field blocks; the trace is the
      reduced unknown and comes back in X. Through ProlongTrace() rather than
      by copying, because the reduced unknowns stop being the space's VDOFs
      the moment a face is constrained -- and what lands is then a genuine
      ceiling-basis representation, which is what lets an estimator read it. */
   Vector tv;
   s.darcy->GetHybridization()->ProlongTrace(s.X, tv);
   REQUIRE(tv.Size() == s.fes_t->GetVSize());
   s.tr_h.SetSpace(s.fes_t.get());
   s.tr_h = tv;
}

} // namespace darcy_padapt

TEST_CASE("A hanging-node family can sit below the trace ceiling",
          "[DarcyHybridization][PAdapt]")
{
   using namespace darcy_padapt;

   /* The first of the three refusals turned round.

      A nonconforming trace space carries a conforming prolongation, and the
      trace system is P^T H P with the solution P X; P interpolates a master
      face's coefficients onto its slaves' nodes IN THE CEILING BASIS. The
      retired route's convention was the opposite -- the first nt(p_f) slots
      held a coarser basis's coefficients and the rest were zeroed -- and a
      zero tail in the ceiling basis is a different function, not the coarse
      one. The two cannot both hold on a face P touches, so a family had to
      run at the ceiling; ignoring that gave errors of 0.284, 1.06, 3.67 as
      the mesh refined, against 0.284, 0.118, 0.091.

      Constraining stores ceiling-basis coefficients, so P is right and a
      family can be coarsened like anything else. The check is the one that
      would have caught the old failure: raising the ceiling, with every face
      still at the element degree, must not move the answer -- on a mesh
      whose hanging-node families are among the faces below it. */
   const int order = GENERATE(0, 1, 2);
   const int gap = GENERATE(1, 3);
   const int n = 4;
   CAPTURE(order, gap);

   Solved base, raised;
   SolveKeeping(order, n, order, base, true, true);
   SolveKeeping(order, n, order + gap, raised, true, true);

   // The mesh really does have hanging nodes, or this asserts nothing.
   REQUIRE(base.mesh->Nonconforming());
   REQUIRE(base.mesh->ncmesh != nullptr);
   REQUIRE(base.mesh->ncmesh->GetNCList(1).slaves.Size() > 0);

   // And the families really are below the ceiling in the raised run.
   {
      const Array<int> &ord =
         raised.darcy->GetHybridization()->GetTraceOrders();
      REQUIRE(ord.Size() == raised.mesh->GetNumFaces());
      const NCMesh::NCList &nc = raised.mesh->ncmesh->GetNCList(1);
      int below = 0;
      for (int i = 0; i < nc.slaves.Size(); i++)
      {
         const int f = nc.slaves[i].index;
         if (f >= 0 && f < ord.Size() && ord[f] < order + gap) { below++; }
      }
      REQUIRE(below > 0);
   }

   // The recovered fields are the same discretisation, so they agree to
   // round-off and not merely to the discretisation error.
   const Vector &bq = base.x.GetBlock(0), &bp = base.x.GetBlock(1);
   const Vector &rq = raised.x.GetBlock(0), &rp = raised.x.GetBlock(1);
   REQUIRE(bq.Size() == rq.Size());
   REQUIRE(bp.Size() == rp.Size());
   Vector dq(bq); dq -= rq;
   Vector dp(bp); dp -= rp;
   INFO("flux differs by " << dq.Normlinf() << " on " << bq.Normlinf()
        << ", potential by " << dp.Normlinf() << " on " << bp.Normlinf());
   REQUIRE(dq.Normlinf() < 1e-10 * std::max(bq.Normlinf(), (real_t)1.0));
   REQUIRE(dp.Normlinf() < 1e-10 * std::max(bp.Normlinf(), (real_t)1.0));

   // And the reduced system is the size the degrees say, not the ceiling's.
   REQUIRE(raised.X.Size() ==
           raised.darcy->GetHybridization()->GetTraceTrueVSize());
   REQUIRE(raised.X.Size() < raised.fes_t->GetTrueVSize());
}

TEST_CASE("The constrained ceiling system IS the coarse system",
          "[DarcyHybridization][PAdapt]")
{
   using namespace darcy_padapt;

   /* The acceptance test for the whole "constrain rather than retire"
      redesign, and it is buildable before any of it is ported.

      The claim is that a face of degree p_f under a ceiling p_max is the same
      discretisation as a face of degree p_f under a ceiling p_f, in a
      different basis -- so that constraining the ceiling's storage by
      E(j,i) = phi_i^lo(node_j^hi) cannot change any answer. Written as
      matrices, with Pi the prolongation from the constrained unknowns to the
      constraint space:

          Pi^T H(ceiling) Pi  ==  H(coarse)

      entry for entry. The reduced trace matrix is what a hybridized solve
      actually inverts, so this is not a proxy for the claim, it is the claim.

      It holds because phi_i^lo = sum_j E(j,i) phi_j^hi POINTWISE, which makes
      C_lo = C_hi E and H_face_lo = E^T H_face_hi E, and the element blocks
      never see the trace at all. The two runs choose different quadrature
      rules -- each follows its own trace degree -- and both are exact here,
      which is the one assumption this test also happens to check: if either
      rule were short, the two sides would differ. */
   const int order = GENERATE(0, 1, 2);
   const int gap = GENERATE(1, 2);
   const int n = 3;
   CAPTURE(order, gap);

   // The coarse discretisation: trace space AT the element degree.
   Solved lo;
   SolveKeeping(order, n, order, lo);

   // The ceiling discretisation: trace space at order+gap, nothing coarsened.
   Solved hi;
   SolveKeeping(order, n, order + gap, hi, false);

   // And the constrained one, which is only here for its prolongation: every
   // face at `order` under the ceiling of `order + gap`.
   Solved con;
   SolveKeeping(order, n, order + gap, con, true);

   const SparseMatrix *P =
      con.darcy->GetHybridization()->GetTraceProlongationMatrix();
   REQUIRE(P != nullptr);

   const SparseMatrix *H_hi = hi.A.As<SparseMatrix>();
   const SparseMatrix *H_lo = lo.A.As<SparseMatrix>();
   REQUIRE(H_hi != nullptr);
   REQUIRE(H_lo != nullptr);

   // The mesh is conforming, so the constraint space's true DOFs are its
   // VDOFs and Pi is exactly the block-diagonal embedding.
   REQUIRE(P->Height() == H_hi->Height());
   REQUIRE(P->Width() == H_lo->Height());
   REQUIRE(P->Width() ==
           con.darcy->GetHybridization()->GetTraceTrueVSize());

   std::unique_ptr<SparseMatrix> red(mfem::RAP(*P, *H_hi, *P));
   std::unique_ptr<DenseMatrix> A1(red->ToDenseMatrix());
   std::unique_ptr<DenseMatrix> A2(H_lo->ToDenseMatrix());

   /* A BOUNDARY FACE CARRIES NO CONSTRAINT IN THIS PROBLEM, and its rows have
      to come out of the comparison -- not because the identity fails there,
      but because there is no identity to test.

      The forms here register interior face integrators only, so nothing ever
      reaches a boundary face's trace unknown: its row is empty, and DIAG_ONE
      makes ComputeH() put a 1 on the diagonal afterwards so the system can be
      solved at all. That 1 is a fix-up, not a discretisation, and restricting
      a unit block gives E^T E rather than I -- for order 0 and a gap of 1, E
      is a column of ones and E^T E is 2, so the difference is exactly 1. It
      was, which is how this was attributed rather than guessed.

      The exclusion is verified rather than asserted: each excluded row must be
      exactly the fix-up, a single diagonal 1 and nothing else. */
   /* THE CONSTRAINED NUMBERING IS NOT THE COARSE SPACE'S, and the comparison
      has to go through the map rather than assume they coincide.

      FiniteElementSpace::GetFaceVDofs() does not return a face's DOFs in
      ascending order: it returns them in the face ELEMENT's dof order, and
      where the face's orientation is reversed that list is descending. On the
      3x3 mesh here, face 2 comes back as {5, 4}. The constrained unknowns are
      ours and are numbered per face in the coarse element's own order, so the
      two differ by exactly that per-face permutation.

      It is a relabelling of our own unknowns and nothing more -- the rows of E
      go in at vdofs[j], so the subspace being imposed is the right one either
      way -- but a comparison that ignores it reads as a broken identity. This
      one did, by exactly the reversal, which is how it was attributed. */
   const int nfaces = lo.mesh->GetNumFaces();
   const int nlo = A2->Height() / nfaces;
   REQUIRE(nlo * nfaces == A2->Height());
   Array<int> cmap(A2->Height()), vl;
   for (int f = 0; f < nfaces; f++)
   {
      lo.fes_t->GetFaceVDofs(f, vl);
      REQUIRE(vl.Size() == nlo);
      for (int i = 0; i < nlo; i++) { cmap[f*nlo + i] = vl[i]; }
   }
   Array<int> keep(A2->Height());
   keep = 1;
   int excluded = 0;
   for (int f = 0; f < nfaces; f++)
   {
      if (lo.mesh->FaceIsInterior(f)) { continue; }
      for (int i = 0; i < nlo; i++)
      {
         const int r = f*nlo + i;
         for (int c = 0; c < A2->Width(); c++)
         {
            const real_t want = (c == cmap[r]) ? 1.0 : 0.0;
            REQUIRE(std::abs((*A2)(cmap[r], c) - want) < 1e-12);
         }
         keep[r] = 0;
         excluded++;
      }
   }
   REQUIRE(excluded > 0);
   REQUIRE(excluded < A2->Height());

   const real_t scale = A2->MaxMaxNorm();
   REQUIRE(scale > 0.0);
   real_t diff = 0.0;
   for (int r = 0; r < A1->Height(); r++)
   {
      if (!keep[r]) { continue; }
      for (int c = 0; c < A1->Width(); c++)
      {
         if (!keep[c]) { continue; }
         diff = std::max(diff,
                         std::abs((*A1)(r, c) - (*A2)(cmap[r], cmap[c])));
      }
   }
   INFO("max difference " << diff << " on " << scale
        << ", excluding " << excluded << " boundary rows");
   REQUIRE(diff < 1e-10 * scale);

   /* And the control, because "two matrices agree" is worth nothing unless
      they could have disagreed. Restricting the ceiling matrix with the
      WRONG embedding -- a plain selection of the first nt(p_f) slots, which
      is what the retire route's storage convention amounts to reading in the
      ceiling basis -- must not reproduce the coarse system. That selection is
      exactly the thing this redesign replaces. */
   const int nt_lo = P->Width() / con.mesh->GetNumFaces();
   const int nt_hi = P->Height() / con.mesh->GetNumFaces();
   REQUIRE(nt_hi > nt_lo);
   SparseMatrix sel(P->Height(), P->Width());
   for (int f = 0; f < con.mesh->GetNumFaces(); f++)
      for (int i = 0; i < nt_lo; i++)
      {
         sel.Add(f*nt_hi + i, f*nt_lo + i, 1.0);
      }
   sel.Finalize();
   std::unique_ptr<SparseMatrix> bad(mfem::RAP(sel, *H_hi, sel));
   std::unique_ptr<DenseMatrix> B1(bad->ToDenseMatrix());
   real_t bdiff = 0.0;
   for (int r = 0; r < B1->Height(); r++)
   {
      if (!keep[r]) { continue; }
      for (int c = 0; c < B1->Width(); c++)
      {
         if (!keep[c]) { continue; }
         bdiff = std::max(bdiff,
                          std::abs((*B1)(r, c) - (*A2)(cmap[r], cmap[c])));
      }
   }
   INFO("the selection differs by " << bdiff);
   REQUIRE(bdiff > 1e-8 * scale);
}

TEST_CASE("The error estimate does not depend on the trace ceiling",
          "[HDGErrorEstimator][PAdapt]")
{
   using namespace darcy_padapt;

   /* The estimator reads the trace solution face by face, and it used to need
      telling about the hybridization to do it: the retired route stored a
      coarse function as COARSE coefficients in the ceiling's slots, so
      applying the constraint space's own face element to them evaluated a
      different function -- not an approximation of one, because the two nodal
      bases sit at different points. Without SetHybridization() the estimate
      came out 1.957 against 0.0269, wrong with nothing failing.

      Constraining rather than retiring removes the need. The slots hold
      ceiling-basis coefficients now, so the space's own face element IS the
      right one and a generic reader is right by default. This checks both
      halves: the estimate does not move when the ceiling is raised, AND it
      does not move when the estimator is told nothing -- which is the
      workaround going away rather than being satisfied. */
   const int order = 2, n = 4;

   Solved plain, raised;
   SolveKeeping(order, n, order, plain);
   SolveKeeping(order, n, order + 3, raised);

   ConstantCoefficient one(1.0);

   // The solutions must agree first, or the estimates could differ for an
   // honest reason.
   Vector dq(plain.q_h);
   dq -= raised.q_h;
   INFO("flux differs by " << dq.Normlinf());
   REQUIRE(dq.Normlinf() < 1e-12);

   HDGDiffusionIntegrator bfi_a(one, 0.5), bfi_b(one, 0.5), bfi_c(one, 0.5);
   HDGErrorEstimator est_plain(bfi_a, plain.tr_h, plain.p_h);
   HDGErrorEstimator est_raised(bfi_b, raised.tr_h, raised.p_h);
   est_raised.SetHybridization(*raised.darcy->GetHybridization());
   HDGErrorEstimator est_bare(bfi_c, raised.tr_h, raised.p_h);

   const Vector &ea = est_plain.GetLocalErrors();
   const Vector &eb = est_raised.GetLocalErrors();
   const Vector &ec = est_bare.GetLocalErrors();

   REQUIRE(ea.Size() == eb.Size());
   REQUIRE(ea.Size() == ec.Size());
   Vector d(ea);
   d -= eb;
   INFO("estimate " << est_plain.GetTotalError() << " against "
        << est_raised.GetTotalError() << " and, told nothing, "
        << est_bare.GetTotalError());
   REQUIRE(d.Normlinf() < 1e-12 * ea.Normlinf());

   Vector d2(ea);
   d2 -= ec;
   REQUIRE(d2.Normlinf() < 1e-12 * ea.Normlinf());
}

TEST_CASE("Projecting the trace comparison changes nothing at equal degrees",
          "[HDGErrorEstimator][PAdapt]")
{
   using namespace darcy_padapt;

   // TraceComparison::Projected exists for the case where the potential
   // carries a higher degree than the trace. Where it does not, an element's
   // trace on a face is already in the trace space and the projection is the
   // identity -- so this must be a no-op, and a difference here would mean the
   // face mass matrix or its quadrature rule is wrong rather than that the
   // idea is.
   const int order = GENERATE(1, 2);
   const int n = 4;
   CAPTURE(order);

   Solved s;
   SolveKeeping(order, n, order, s);

   ConstantCoefficient one(1.0);
   HDGDiffusionIntegrator bfi_a(one, 0.5), bfi_b(one, 0.5);

   HDGErrorEstimator literal(bfi_a, s.tr_h, s.p_h);
   HDGErrorEstimator projected(bfi_b, s.tr_h, s.p_h);
   projected.SetTraceComparison(HDGErrorEstimator::TraceComparison::Projected);

   const Vector &ea = literal.GetLocalErrors();
   const Vector &eb = projected.GetLocalErrors();

   Vector d(ea);
   d -= eb;
   INFO("literal " << literal.GetTotalError() << ", projected "
        << projected.GetTotalError());
   REQUIRE(d.Normlinf() < 1e-11 * ea.Normlinf());
}

TEST_CASE("Excluding a boundary attribute removes its faces and nothing else",
          "[HDGErrorEstimator][PAdapt]")
{
   using namespace darcy_padapt;

   // The exclusion exists because |p^ - lambda| is not an error on a face
   // whose Dirichlet datum is imposed weakly. What it must not do is reach any
   // other face: an element away from the excluded attribute has to come back
   // with the same number it had.
   const int order = 2, n = 4;

   Solved s;
   SolveKeeping(order, n, order, s);

   ConstantCoefficient one(1.0);
   HDGDiffusionIntegrator bfi_a(one, 0.5), bfi_b(one, 0.5);

   HDGErrorEstimator all(bfi_a, s.tr_h, s.p_h);
   HDGErrorEstimator less(bfi_b, s.tr_h, s.p_h);

   Array<int> marker(s.mesh->bdr_attributes.Max());
   marker = 0;
   marker[0] = 1;                     // attribute 1, the y = 0 side
   less.SetExcludedBoundary(marker);

   const Vector &ea = all.GetLocalErrors();
   const Vector &eb = less.GetLocalErrors();

   // Which elements touch attribute 1.
   Array<int> touches(s.mesh->GetNE());
   touches = 0;
   for (int b = 0; b < s.mesh->GetNBE(); b++)
   {
      if (s.mesh->GetBdrAttribute(b) != 1) { continue; }
      int e1, e2;
      s.mesh->GetFaceElements(s.mesh->GetBdrElementFaceIndex(b), &e1, &e2);
      touches[e1] = 1;
   }
   REQUIRE(touches.Sum() == n);

   int changed = 0;
   for (int e = 0; e < ea.Size(); e++)
   {
      CAPTURE(e, touches[e], ea(e), eb(e));
      if (touches[e])
      {
         REQUIRE(eb(e) <= ea(e));
         if (eb(e) < ea(e) * (1.0 - 1e-12)) { changed++; }
      }
      else
      {
         REQUIRE(eb(e) == Approx(ea(e)).margin(1e-14 * ea.Normlinf()));
      }
   }

   // And it must actually have removed something, or the test passes for the
   // wrong reason.
   REQUIRE(changed == n);
}

#ifdef MFEM_USE_MPI

TEST_CASE("The constrained trace size does not depend on the rank count",
          "[DarcyHybridization][PAdapt][Parallel]")
{
   /* The third refusal turned round, and the one that had no repair short of
      this redesign.

      The retired route read a face's first nt(p_f) slots as coefficients of
      the coarse basis. Across a shared face the two ranks order that face's
      DOFs by their own view of its orientation, so "the first nt(p_f)" named
      a different subspace on each side and each retired slots the other was
      still using. Measured on a fixed mesh with every face at degree 2 under
      a degree-3 ceiling, so the answer is a property of the discretisation
      and cannot depend on the partition: 144 retired true DOFs at one rank,
      152 at two, 162 at three, and the relative L2 error going from 5.9e-4 to
      0.56. SetTraceOrders() refused it outright.

      Constraining removes the question. The constraint acts on TRUE DOFs, so
      only the owner of a face builds an E for it, in its own ordering, and
      every other rank receives values through the space's map exactly as it
      does for a uniform trace.

      Checked against the SERIAL answer rather than against another rank
      count, which is what makes it independent: the same mesh and the same
      degrees, built serially on every rank, must give the same global total
      as the parallel run's ranks summed. */
   const int order = 2, cap = 3, n = 8, dim = 2;

   auto degree_at = [order](const Vector &c)
   {
      const int band = (int)(4.0 * c(0)) + (int)(4.0 * c(1));
      return order + (band % 2);   // 2 or 3, so faces really are coarsened
   };

   // Build a hybridization on @a m with degrees from the geometry, and return
   // the number of trace unknowns it would solve for.
   ConstantCoefficient one(1.0);   // outlives every form built below
   auto constrained_size = [&](Mesh &m, FiniteElementSpace &fq,
                               FiniteElementSpace &fp, FiniteElementSpace &ft,
                               DarcyForm &darcy)
   {
      darcy.GetFluxMassForm()->AddDomainIntegrator(
         new VectorMassIntegrator(one));
      Array<int> ess;
      darcy.EnableHybridization(&ft, new NormalTraceJumpIntegrator(), ess);

      Array<int> elem_order(m.GetNE());
      for (int e = 0; e < m.GetNE(); e++)
      {
         Vector c;
         m.GetElementCenter(e, c);
         elem_order[e] = degree_at(c);
      }
      Array<int> face_order;
      DarcyHybridization::FaceOrdersFromElementOrders(
         m, elem_order, DarcyHybridization::TraceOrderRule::Min, cap,
         face_order);
      darcy.GetHybridization()->SetTraceOrders(face_order);
      return darcy.GetHybridization()->GetTraceTrueVSize();
   };

   HYPRE_BigInt serial_size = 0;
   {
      Mesh m = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, false,
                                     1.0, 1.0);
      L2_FECollection qc(order, dim, BasisType::GaussLobatto);
      L2_FECollection pc(order, dim, BasisType::GaussLobatto);
      FiniteElementSpace fq(&m, &qc, dim), fp(&m, &pc);
      DG_Interface_FECollection tc(cap, dim);
      FiniteElementSpace ft(&m, &tc);
      DarcyForm darcy(&fq, &fp);
      serial_size = constrained_size(m, fq, fp, ft, darcy);
   }

   HYPRE_BigInt par_size = 0;
   {
      Mesh serial = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, false,
                                          1.0, 1.0);
      ParMesh m(MPI_COMM_WORLD, serial);
      serial.Clear();
      L2_FECollection qc(order, dim, BasisType::GaussLobatto);
      L2_FECollection pc(order, dim, BasisType::GaussLobatto);
      ParFiniteElementSpace fq(&m, &qc, dim), fp(&m, &pc);
      DG_Interface_FECollection tc(cap, dim);
      ParFiniteElementSpace ft(&m, &tc);
      ParDarcyForm darcy(&fq, &fp);
      par_size = constrained_size(m, fq, fp, ft, darcy);
   }

   MPI_Allreduce(MPI_IN_PLACE, &par_size, 1, HYPRE_MPI_BIG_INT, MPI_SUM,
                 MPI_COMM_WORLD);

   // The degrees really do coarsen, or nothing is being exercised: at the
   // ceiling everywhere the total would be the space's own size.
   const int nfaces_flat = 2 * n * (n + 1);
   REQUIRE(serial_size < (HYPRE_BigInt)(nfaces_flat * (cap + 1)));
   REQUIRE(serial_size > 0);

   INFO("serial " << serial_size << " against " << par_size << " summed");
   REQUIRE(par_size == serial_size);
}

TEST_CASE("A shared face gets both ranks' element degrees",
          "[DarcyHybridization][PAdapt][Parallel]")
{
   // The face rule needs the degree of the element on the far side of a
   // shared face, and the far side belongs to another rank. What is checked
   // here is not that the exchange runs but that it agrees with an
   // INDEPENDENT computation of the same thing: the degrees are a function of
   // the element centre, so every rank can work out what its neighbour must
   // have had without being told, and the two answers must match on every
   // face.
   const int order = 2, cap = 5, n = 8;

   Mesh serial = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, false,
                                       1.0, 1.0);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   serial.Clear();

   // Degree by geometry, and deliberately not monotone in x, so that a
   // partition cutting anywhere still produces faces whose two sides differ.
   auto degree_at = [order](const Vector &c)
   {
      const int band = (int)(4.0 * c(0)) + (int)(4.0 * c(1));
      return order + (band % 3);
   };

   Array<int> elem_order(mesh.GetNE());
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      Vector c;
      mesh.GetElementCenter(e, c);
      elem_order[e] = degree_at(c);
   }

   const auto rule = GENERATE(DarcyHybridization::TraceOrderRule::Min,
                              DarcyHybridization::TraceOrderRule::Max);

   Array<int> face_order;
   DarcyHybridization::FaceOrdersFromElementOrders(mesh, elem_order, rule, cap,
                                                   face_order);
   REQUIRE(face_order.Size() == mesh.GetNumFaces());

   // The shared faces are the ones that could not have been done locally.
   int checked = 0;
   for (int sf = 0; sf < mesh.GetNSharedFaces(); sf++)
   {
      const int f = mesh.GetSharedFace(sf);
      if (f < 0 || f >= mesh.GetNumFaces()) { continue; }

      FaceElementTransformations *FTr = mesh.GetSharedFaceTransformations(sf);
      REQUIRE(FTr != nullptr);

      // The neighbour's centre, from the face's own geometry rather than from
      // anything the exchange produced: step across the face from this side's
      // centre by twice the distance to the face.
      Vector c_el, c_f;
      mesh.GetElementCenter(FTr->Elem1No, c_el);
      c_f.SetSize(c_el.Size());
      {
         IntegrationPoint ip;
         ip.Set2(0.5, 0.0);
         FTr->SetAllIntPoints(&ip);
         FTr->Transform(ip, c_f);
      }
      Vector c_nbr(c_el.Size());
      for (int d = 0; d < c_el.Size(); d++)
      { c_nbr(d) = c_el(d) + 2.0 * (c_f(d) - c_el(d)); }

      const int p_here = elem_order[FTr->Elem1No];
      const int p_there = degree_at(c_nbr);
      const int want = std::min(cap,
                                (rule == DarcyHybridization::TraceOrderRule::Min)
                                ? std::min(p_here, p_there)
                                : std::max(p_here, p_there));

      CAPTURE(f, p_here, p_there, face_order[f], want);
      REQUIRE(face_order[f] == want);
      checked++;
   }

   // On one rank there is nothing shared and the test is vacuous, which is
   // fine and is worth saying; on more than one there must be something.
   int total = checked;
   MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   if (Mpi::WorldSize() > 1) { REQUIRE(total > 0); }
}

#endif // MFEM_USE_MPI

TEST_CASE("A coarse trace basis is an exact combination of the ceiling's",
          "[DarcyHybridization][PAdapt]")
{
   /* The identity the "constrain rather than retire" redesign rests on, and
      the reason that redesign cannot change any answer.

      This route stores a degree-p_f face function as p_f-basis coefficients in
      the first nt(p_f) of the face's ceiling-degree slots. Everything that has
      gone wrong with it -- a hanging-node family, an essential datum on a
      coarsened face, a face shared between ranks -- is one thing: an outside
      reader of those slots assumes the CEILING basis, because that is the
      space's own. The proposed repair is to store the same function as its
      ceiling-basis coefficients, constrained to the degree-p_f subspace by
      E, whose columns are the coarse basis functions written in the fine one:

          E(j,i) = phi_i^lo( node_j^hi )

      For that to be a change of representation and not of discretisation, the
      face matrix assembled against the coarse trace must equal the one
      assembled against the ceiling trace, restricted by E. It does, because a
      degree-p_f polynomial IS a degree-p_max polynomial and so
      phi_i^lo = sum_j E(j,i) phi_j^hi exactly -- but that is an argument, and
      the argument is worth one test, because it is what says the port has
      nothing to prove beyond bookkeeping. */
   const int dim = 2;
   const int p_lo = GENERATE(0, 1, 2), gap = GENERATE(1, 2, 3);
   const int p_hi = p_lo + gap;
   CAPTURE(p_lo, p_hi);

   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, false,
                                     1.0, 1.0);
   L2_FECollection fec(2, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);

   DG_Interface_FECollection tr_coll(p_hi, dim);

   // An interior face, and the two elements it separates.
   int face = -1;
   for (int f = 0; f < mesh.GetNumFaces(); f++)
   {
      if (mesh.FaceIsInterior(f)) { face = f; break; }
   }
   REQUIRE(face >= 0);

   FaceElementTransformations *FTr =
      mesh.GetInteriorFaceTransformations(face);
   REQUIRE(FTr != nullptr);

   const Geometry::Type geom = mesh.GetFaceGeometry(face);
   const FiniteElement *fe_lo = tr_coll.GetFE(geom, p_lo);
   const FiniteElement *fe_hi = tr_coll.GetFE(geom, p_hi);
   REQUIRE(fe_lo->GetOrder() == p_lo);
   REQUIRE(fe_hi->GetOrder() == p_hi);

   const FiniteElement *el1 = fes.GetFE(FTr->Elem1No);
   const FiniteElement *el2 = fes.GetFE(FTr->Elem2No);

   NormalTraceJumpIntegrator integ;
   DenseMatrix M_lo, M_hi;
   integ.AssembleFaceMatrix(*fe_lo, *el1, *el2, *FTr, M_lo);
   integ.AssembleFaceMatrix(*fe_hi, *el1, *el2, *FTr, M_hi);

   const int n_lo = fe_lo->GetDof(), n_hi = fe_hi->GetDof();
   REQUIRE(M_lo.Width() == n_lo);
   REQUIRE(M_hi.Width() == n_hi);
   REQUIRE(M_lo.Height() == M_hi.Height());

   // E: the coarse basis functions evaluated at the ceiling element's nodes.
   DenseMatrix E(n_hi, n_lo);
   const IntegrationRule &nodes = fe_hi->GetNodes();
   Vector shape(n_lo);
   for (int j = 0; j < n_hi; j++)
   {
      fe_lo->CalcShape(nodes.IntPoint(j), shape);
      for (int i = 0; i < n_lo; i++) { E(j, i) = shape(i); }
   }

   DenseMatrix prod(M_hi.Height(), n_lo);
   mfem::Mult(M_hi, E, prod);

   prod -= M_lo;
   const real_t scale = std::max(M_lo.MaxMaxNorm(), (real_t)1.0);
   INFO("max difference " << prod.MaxMaxNorm() << " on " << scale);
   REQUIRE(prod.MaxMaxNorm() < 1e-12 * scale);

   /* And E has full column rank, which is the other half of the constraint
      being well posed: the constrained face carries exactly nt(p_f) unknowns,
      neither fewer -- which would lose the function -- nor more. E^T E is
      then invertible, and DenseMatrixInverse is the check available here,
      MFEM_USE_LAPACK being off so SingularValues() aborts. */
   DenseMatrix EtE(n_lo);
   MultAtB(E, E, EtE);
   DenseMatrixInverse EtEi(EtE);
   DenseMatrix I(n_lo), should_be_I(n_lo);
   EtEi.GetInverseMatrix(I);
   mfem::Mult(EtE, I, should_be_I);
   for (int i = 0; i < n_lo; i++) { should_be_I(i, i) -= 1.0; }
   INFO("E^T E inverse residual " << should_be_I.MaxMaxNorm());
   REQUIRE(should_be_I.MaxMaxNorm() < 1e-10);

   /* R, the other half of the pair, and the identity that makes it usable.

      E embeds a coarse function in the ceiling; R reads a ceiling function
      back at the coarse nodes:

          R(i,j) = phi_j^hi( node_i^lo )

      and R E = I exactly, because interpolating a degree-p_f polynomial at
      the ceiling's nodes and then evaluating it at the coarse ones returns
      the polynomial. E is the solution map and E^T the residual map; R is
      the DATA map, and it is NOT the transpose of either.

      That distinction is the whole reason to build R rather than reach for
      E^+ = (E^T E)^-1 E^T, which the block above has just shown exists. A
      least-squares fit over the CEILING's nodes depends on where those nodes
      are, so restricting a boundary datum with it would give a different
      answer at every ceiling -- and "the answer does not move when the
      ceiling moves" is the property the essential-datum refusal exists to
      protect. R interpolates at the coarse nodes and so cannot depend on the
      ceiling at all. */
   DenseMatrix R(n_lo, n_hi);
   {
      const IntegrationRule &nodes_lo = fe_lo->GetNodes();
      Vector shape_hi(n_hi);
      for (int i = 0; i < n_lo; i++)
      {
         fe_hi->CalcShape(nodes_lo.IntPoint(i), shape_hi);
         for (int j = 0; j < n_hi; j++) { R(i, j) = shape_hi(j); }
      }
   }

   DenseMatrix RE(n_lo);
   mfem::Mult(R, E, RE);
   for (int i = 0; i < n_lo; i++) { RE(i, i) -= 1.0; }
   INFO("R E - I is " << RE.MaxMaxNorm());
   REQUIRE(RE.MaxMaxNorm() < 1e-12);

   /* And the property that picks R over E^+, stated so that it can fail.

      Both are left inverses, so both are exact on the coarse space and no
      test restricted to it can tell them apart. The difference shows on a
      datum the coarse space cannot represent, and it is this: R is a
      function of the FUNCTION -- interpolate it at the coarse nodes -- while
      E^+ is a function of the ceiling's NODE SET, being least squares over
      those nodes. So take one function, represent it exactly at two
      different ceilings, and restrict it both ways. R must give the same
      answer twice. E^+ need not, and does not.

      That is what "the answer must not move when the ceiling moves" means
      for a boundary datum, and it is the whole reason the essential-datum
      refusal exists. A datum of degree p_hi is representable at ceiling
      p_hi and at p_hi + 1 both, which is what makes the comparison exact
      rather than asymptotic. */
   /* A SECOND collection, not tr_coll.GetFE(geom, p_hi + 1). GetFE() takes
      the collection's order, DG_Interface_FECollection(p)::GetOrder() is
      p + 1, and at q == GetOrder() it short-circuits to the base collection
      and hands back degree q - 1 -- silently, which is how this first
      presented: 5 == 6. */
   DG_Interface_FECollection tr_coll2(p_hi + 1, dim);
   const FiniteElement *fe_hi2 = tr_coll2.GetFE(geom, p_hi + 1);
   REQUIRE(fe_hi2->GetOrder() == p_hi + 1);
   const int n_hi2 = fe_hi2->GetDof();

   // A polynomial of degree p_hi in the face's reference coordinate, so both
   // ceilings carry it exactly and neither is approximating anything.
   auto datum = [p_hi](const IntegrationPoint &ip)
   { return std::pow(0.3 + 0.7 * ip.x, p_hi); };

   Vector g_hi(n_hi), g_hi2(n_hi2);
   for (int j = 0; j < n_hi; j++)
   { g_hi(j) = datum(fe_hi->GetNodes().IntPoint(j)); }
   for (int j = 0; j < n_hi2; j++)
   { g_hi2(j) = datum(fe_hi2->GetNodes().IntPoint(j)); }

   auto restrict_R = [&](const FiniteElement &fe_h, const Vector &g,
                         Vector &c)
   {
      DenseMatrix Rm(n_lo, fe_h.GetDof());
      Vector sh(fe_h.GetDof());
      for (int i = 0; i < n_lo; i++)
      {
         fe_h.CalcShape(fe_lo->GetNodes().IntPoint(i), sh);
         for (int j = 0; j < fe_h.GetDof(); j++) { Rm(i, j) = sh(j); }
      }
      c.SetSize(n_lo);
      Rm.Mult(g, c);
   };

   Vector c1, c2;
   restrict_R(*fe_hi, g_hi, c1);
   restrict_R(*fe_hi2, g_hi2, c2);
   Vector dc(c1); dc -= c2;
   INFO("R at two ceilings differs by " << dc.Normlinf());
   REQUIRE(dc.Normlinf() < 1e-12);

   /* The control, and it is the point: the same comparison with E^+ moves.
      Without it "R is ceiling-independent" is a property nothing was shown
      to lack.

      One combination cannot show it, and the reason is structural rather
      than a tolerance being missed: a least-squares CONSTANT is the mean of
      the nodal values, Gauss-Lobatto nodes are symmetric about the midpoint,
      and the mean of a LINEAR function over a symmetric node set is its
      midpoint value whatever the set. So at p_lo = 0 with a degree-1 datum
      the two ceilings agree exactly, for both maps. Every other combination
      here separates them. */
   auto restrict_Eplus = [&](const FiniteElement &fe_h, const Vector &g,
                             Vector &c)
   {
      const int nh = fe_h.GetDof();
      DenseMatrix Em(nh, n_lo);
      Vector sh(n_lo);
      for (int j = 0; j < nh; j++)
      {
         fe_lo->CalcShape(fe_h.GetNodes().IntPoint(j), sh);
         for (int i = 0; i < n_lo; i++) { Em(j, i) = sh(i); }
      }
      DenseMatrix G(n_lo);
      MultAtB(Em, Em, G);
      Vector rhs(n_lo);
      Em.MultTranspose(g, rhs);
      c.SetSize(n_lo);
      DenseMatrixInverse(G).Mult(rhs, c);
   };

   Vector e1, e2;
   restrict_Eplus(*fe_hi, g_hi, e1);
   restrict_Eplus(*fe_hi2, g_hi2, e2);
   Vector de(e1); de -= e2;
   INFO("E^+ at two ceilings differs by " << de.Normlinf());
   if (!(p_lo == 0 && p_hi == 1)) { REQUIRE(de.Normlinf() > 1e-8); }
}
