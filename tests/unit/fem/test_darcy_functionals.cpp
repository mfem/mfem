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

namespace darcy_functionals
{

real_t pExact(const Vector &x)
{
   real_t r = 1.0;
   for (int i = 0; i < x.Size(); i++) { r *= sin(M_PI * x(i)); }
   return r;
}

/// -div u = g with u = -grad p.
real_t gExact(const Vector &x)
{
   return -x.Size() * M_PI * M_PI * pExact(x);
}

/// Everything a functional needs: the total flux and the assembled source.
struct Solved
{
   std::unique_ptr<Mesh> mesh;
   std::unique_ptr<L2_FECollection> u_coll, p_coll;
   std::unique_ptr<DG_Interface_FECollection> t_coll;
   std::unique_ptr<FiniteElementSpace> fes_u, fes_p, fes_t;
   std::unique_ptr<DarcyForm> darcy;
   GridFunction ut;

   /// The assembled source over the marked elements.
   /** Taken from the *assembled* right-hand side rather than by re-integrating
       g, because the identity under test is a property of the discrete
       equations: the potential equation tested against the indicator of the
       subdomain. The potential space is discontinuous and its basis is nodal,
       so setting every dof of an element to one gives that indicator exactly,
       and no quadrature error enters the comparison. */
   real_t SourceOver(const Array<int> &elem_marker)
   {
      GridFunction ind(fes_p.get());
      ind = 0.0;
      Array<int> dofs;
      for (int e = 0; e < mesh->GetNE(); e++)
      {
         if (!elem_marker[e]) { continue; }
         fes_p->GetElementDofs(e, dofs);
         for (int i = 0; i < dofs.Size(); i++) { ind(dofs[i]) = 1.0; }
      }
      return (*darcy->GetPotentialRHS()) * ind;
   }
};

Solved Solve(int n, int order, int dim)
{
   Solved S;
   S.mesh.reset(new Mesh(
                   (dim == 3)
                   ? Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON)
                   : Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL)));

   S.u_coll.reset(new L2_FECollection(order, dim));
   S.p_coll.reset(new L2_FECollection(order, dim));
   S.t_coll.reset(new DG_Interface_FECollection(order, dim));
   S.fes_u.reset(new FiniteElementSpace(S.mesh.get(), S.u_coll.get(), dim));
   S.fes_p.reset(new FiniteElementSpace(S.mesh.get(), S.p_coll.get()));
   S.fes_t.reset(new FiniteElementSpace(S.mesh.get(), S.t_coll.get()));

   ConstantCoefficient one(1.0);
   FunctionCoefficient gcoeff(gExact);
   RatioCoefficient ik(1.0, one);

   S.darcy.reset(new DarcyForm(S.fes_u.get(), S.fes_p.get()));
   S.darcy->GetFluxMassForm()->AddDomainIntegrator(
      new VectorMassIntegrator(one));
   MixedBilinearForm *B = S.darcy->GetFluxDivForm();
   B->AddDomainIntegrator(new VectorDivergenceIntegrator());
   B->AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));
   S.darcy->GetPotentialMassForm()->AddInteriorFaceIntegrator(
      new HDGDiffusionIntegrator(ik, 0.5));
   S.darcy->GetPotentialRHS()->AddDomainIntegrator(
      new DomainLFIntegrator(gcoeff, 6, 12));

   Array<int> ess;
   S.darcy->EnableHybridization(S.fes_t.get(), new NormalTraceJumpIntegrator(),
                                ess);
   S.darcy->Assemble();

   BlockVector x(S.darcy->GetOffsets());
   x = 0.0;
   OperatorPtr A;
   Vector X, RHS;
   S.darcy->FormLinearSystem(ess, x, A, X, RHS, true);

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
   S.darcy->RecoverFEMSolution(X, x);

   S.darcy->ReconstructTotalFlux(x, X, S.ut);
   return S;
}

/// Elements whose centre satisfies @a in.
Array<int> Mark(Mesh &mesh, std::function<bool(const Vector &)> in)
{
   Array<int> m(mesh.GetNE());
   Vector c;
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      mesh.GetElementCenter(e, c);
      m[e] = in(c) ? 1 : 0;
   }
   return m;
}

} // namespace darcy_functionals

TEST_CASE("The flux through a surface balances the source inside it",
          "[DarcyForm][Functional]")
{
   // For -div q = g the divergence theorem gives, over any union of elements,
   //
   //     integral of q.n over its boundary  ==  -integral of g inside
   //
   // and hybridization makes that hold *discretely*, because the total flux is
   // normally continuous and the potential equation tested against the
   // indicator of the subdomain is exactly this statement. So the two numbers
   // agree to round-off, not to discretisation error, and that makes this the
   // sharpest available test of the whole assembly: the local solves, the
   // trace solve and the flux reconstruction all have to be right for it.
   //
   // Note it holds on a *single element* too, which is local conservation, and
   // on a subdomain touching the mesh boundary, which is what fixes the
   // convention that outside the mesh counts as outside the subdomain.
   using namespace darcy_functionals;

   const int dim = GENERATE(2, 3);
   const int order = GENERATE(0, 1, 2);
   const int n = (dim == 3) ? 3 : 6;
   CAPTURE(dim, order, n);

   Solved S = Solve(n, order, dim);
   Mesh &mesh = *S.mesh;

   auto check = [&](const char *what, const Array<int> &m)
   {
      CAPTURE(what);
      const real_t flux = ComputeOutwardFlux(S.ut, m);
      const real_t src  = S.SourceOver(m);
      CAPTURE(flux, src);
      REQUIRE(flux == Approx(-src).epsilon(1e-11).margin(1e-12));
   };

   check("whole domain", Mark(mesh, [](const Vector &)
   {
      return true;
   }));
   check("left half", Mark(mesh, [](const Vector &c)
   {
      return c(0) < 0.5;
   }));
   check("a corner block", Mark(mesh, [](const Vector &c)
   {
      return c(0) < 0.5 && c(1) < 0.5;
   }));

   // Local conservation, one element at a time.
   for (int e = 0; e < mesh.GetNE(); e += 1 + mesh.GetNE() / 5)
   {
      Array<int> m(mesh.GetNE());
      m = 0;
      m[e] = 1;
      CAPTURE(e);
      const real_t flux = ComputeOutwardFlux(S.ut, m);
      const real_t src  = S.SourceOver(m);
      REQUIRE(flux == Approx(-src).epsilon(1e-11).margin(1e-12));
   }
}

TEST_CASE("The boundary functional agrees with the subdomain one",
          "[DarcyForm][Functional]")
{
   // Two entry points, one quantity: the flux out of the whole mesh is both
   // the outward flux of the everything-subdomain and the flux through every
   // boundary attribute. They share no code path beyond the face integral, so
   // agreement checks the orientation conventions of each against the other --
   // a sign error in either would show here and nowhere else.
   using namespace darcy_functionals;

   Solved S = Solve(6, 2, 2);
   Array<int> all_elems(S.mesh->GetNE());
   all_elems = 1;
   Array<int> all_bdr(S.mesh->bdr_attributes.Max());
   all_bdr = 1;

   const real_t by_subdomain = ComputeOutwardFlux(S.ut, all_elems);
   const real_t by_boundary  = ComputeBoundaryFlux(S.ut, all_bdr);
   CAPTURE(by_subdomain, by_boundary);
   REQUIRE(by_boundary == Approx(by_subdomain).epsilon(1e-11));

   // And a single attribute is a proper part of it, not the whole.
   Array<int> one_bdr(S.mesh->bdr_attributes.Max());
   one_bdr = 0;
   one_bdr[0] = 1;
   const real_t part = ComputeBoundaryFlux(S.ut, one_bdr);
   REQUIRE(std::abs(part) < std::abs(by_boundary));
}
