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

namespace darcy_postprocess
{

/// Equation e gets its own frequency, so the blocks cannot be confused.
real_t pEx(int e, const Vector &x)
{
   real_t r = 1.0;
   for (int i = 0; i < x.Size(); i++) { r *= sin((e + 1) * M_PI * x(i)); }
   return r;
}

real_t gEx(int e, const Vector &x)
{
   return -x.Size() * (e + 1) * (e + 1) * M_PI * M_PI * pEx(e, x);
}

struct Solved
{
   std::unique_ptr<Mesh> mesh;
   std::unique_ptr<L2_FECollection> u_coll, p_coll;
   std::unique_ptr<DG_Interface_FECollection> t_coll;
   std::unique_ptr<FiniteElementSpace> fes_u, fes_p, fes_t;
   std::unique_ptr<DarcyForm> darcy;
   std::unique_ptr<GridFunction> q_h, p_h;
   BlockVector x;
   Vector X;
   int neq;

   /// L2 error of block @a e of @a gf against the exact potential.
   real_t BlockError(const GridFunction &gf, int e) const
   {
      const FiniteElementSpace *fes = gf.FESpace();
      const int nd = fes->GetNDofs();
      FiniteElementSpace scalar(fes->GetMesh(), fes->FEColl());
      GridFunction blk(&scalar);
      // byNODES: block e is the contiguous dof range [e*nd, (e+1)*nd).
      for (int i = 0; i < nd; i++) { blk(i) = gf(e * nd + i); }

      FunctionCoefficient c([e](const Vector &x) { return pEx(e, x); });
      const int qo = 2 * fes->GetMaxElementOrder() + 6;
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i = 0; i < Geometry::NumGeom; i++)
      { irs[i] = &(IntRules.Get(i, qo)); }
      return blk.ComputeL2Error(c, irs);
   }
};

/// A block-diagonal Darcy system: neq copies of the same operator, each with
/// its own source, hybridized and solved together.
std::unique_ptr<Solved> Solve(int n, int order, int neq, int dim = 2)
{
   auto S = std::unique_ptr<Solved>(new Solved);
   S->neq = neq;
   S->mesh.reset(new Mesh(
                    (dim == 3)
                    ? Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON)
                    : Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL)));
   Mesh &mesh = *S->mesh;

   S->u_coll.reset(new L2_FECollection(order, dim));
   S->p_coll.reset(new L2_FECollection(order, dim));
   S->t_coll.reset(new DG_Interface_FECollection(order, dim));
   S->fes_u.reset(new FiniteElementSpace(&mesh, S->u_coll.get(), neq * dim,
                                         Ordering::byNODES));
   S->fes_p.reset(new FiniteElementSpace(&mesh, S->p_coll.get(), neq,
                                         Ordering::byNODES));
   S->fes_t.reset(new FiniteElementSpace(&mesh, S->t_coll.get(), neq,
                                         Ordering::byNODES));

   ConstantCoefficient one(1.0);
   RatioCoefficient ik(1.0, one);
   VectorFunctionCoefficient gcoeff(neq, [neq](const Vector &x, Vector &v)
   {
      for (int e = 0; e < neq; e++) { v(e) = gEx(e, x); }
   });

   S->darcy.reset(new DarcyForm(S->fes_u.get(), S->fes_p.get()));
   S->darcy->GetFluxMassForm()->AddDomainIntegrator(
      new VectorBlockDiagonalIntegrator(neq, new VectorMassIntegrator(one)));
   MixedBilinearForm *B = S->darcy->GetFluxDivForm();
   B->AddDomainIntegrator(
      new VectorBlockDiagonalIntegrator(neq, new VectorDivergenceIntegrator()));
   B->AddInteriorFaceIntegrator(new VectorBlockDiagonalIntegrator(
                                   neq,
                                   new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0))));
   S->darcy->GetPotentialMassForm()->AddInteriorFaceIntegrator(
      new VectorBlockDiagonalIntegrator(
         neq, new HDGDiffusionIntegrator(ik, 0.5)));
   S->darcy->GetPotentialRHS()->AddDomainIntegrator(
      new VectorDomainLFIntegrator(gcoeff));

   Array<int> ess;
   S->darcy->EnableHybridization(
      S->fes_t.get(),
      new VectorBlockDiagonalIntegrator(neq, new NormalTraceJumpIntegrator()),
      ess);
   S->darcy->Assemble();

   S->x.Update(S->darcy->GetOffsets());
   S->x = 0.0;
   OperatorPtr A;
   Vector RHS;
   S->darcy->FormLinearSystem(ess, S->x, A, S->X, RHS, true);

   GSSmoother prec;
   GMRESSolver lin;
   lin.SetKDim(500);
   lin.SetMaxIter(8000);
   lin.SetRelTol(1e-14);
   lin.SetAbsTol(1e-16);
   lin.SetPreconditioner(prec);
   lin.SetOperator(*A);
   lin.Mult(RHS, S->X);
   REQUIRE(lin.GetConverged());
   S->darcy->RecoverFEMSolution(S->X, S->x);

   S->q_h.reset(new GridFunction(S->fes_u.get(), S->x.GetBlock(0)));
   S->p_h.reset(new GridFunction(S->fes_p.get(), S->x.GetBlock(1)));
   return S;
}

} // namespace darcy_postprocess

TEST_CASE("Local postprocessing improves the potential, one field or many",
          "[DarcyForm][Postprocess]")
{
   // The classic HDG postprocessing -- NPC eq (25) -- one Neumann problem per
   // element per equation, closed by the element average. Unlike the branch's
   // mixed reconstruction it needs neither the trace space nor the
   // hybridization, only the computed flux and potential, which is what lets
   // it be general in vdim cheaply.
   using namespace darcy_postprocess;

   const int neq = GENERATE(1, 2, 3);
   const int order = GENERATE(1, 2);
   CAPTURE(neq, order);

   ConstantCoefficient one(1.0);
   RatioCoefficient ik(1.0, one);

   auto S = Solve(8, order, neq);
   HDGPotentialPostprocessor pp(*S->q_h, *S->p_h);
   pp.SetDiffusionInverse(ik);
   REQUIRE(pp.GetNumEquations() == neq);

   GridFunction p_s;
   pp.Compute(p_s);
   REQUIRE(p_s.FESpace()->GetVDim() == neq);
   REQUIRE(p_s.FESpace()->GetMaxElementOrder() == order + 1);

   // Every equation improves, not just the first: a postprocessing that had
   // the block indexing wrong would leave the later ones alone or worse.
   for (int e = 0; e < neq; e++)
   {
      CAPTURE(e);
      const real_t raw  = S->BlockError(*S->p_h, e);
      const real_t post = S->BlockError(p_s, e);
      CAPTURE(raw, post);
      REQUIRE(post < 0.5 * raw);
   }
}

TEST_CASE("Local postprocessing treats the equations independently",
          "[DarcyForm][Postprocess]")
{
   // The system above is block diagonal, so equation 0 of a many-equation
   // solve is the same discrete problem as a one-equation solve. Its
   // postprocessed potential must therefore agree to round-off, and this is
   // the sharp test of the block indexing: a coefficient read from the wrong
   // block, or a right-hand side accumulated across blocks, moves this far
   // above round-off while every convergence rate still looks plausible.
   using namespace darcy_postprocess;

   ConstantCoefficient one(1.0);
   RatioCoefficient ik(1.0, one);
   const int order = 2;

   auto S1 = Solve(6, order, 1);
   auto S3 = Solve(6, order, 3);

   HDGPotentialPostprocessor pp1(*S1->q_h, *S1->p_h);
   pp1.SetDiffusionInverse(ik);
   GridFunction ps1;
   pp1.Compute(ps1);

   HDGPotentialPostprocessor pp3(*S3->q_h, *S3->p_h);
   pp3.SetDiffusionInverse(ik);
   GridFunction ps3;
   pp3.Compute(ps3);

   const real_t e1 = S1->BlockError(ps1, 0);
   const real_t e3 = S3->BlockError(ps3, 0);
   CAPTURE(e1, e3);
   REQUIRE(e3 == Approx(e1).epsilon(1e-10));
}

TEST_CASE("Local postprocessing reads an H(div) flux",
          "[DarcyForm][Postprocess]")
{
   // The other flux layout. An H(div) space carries the vector in the element,
   // so neq equations need vdim == neq and a block is one component -- the
   // opposite of the L2 case. The constructor checks which it has been given
   // rather than assuming, and mismatching the two is how a block would be
   // read past its end.
   using namespace darcy_postprocess;

   Mesh mesh = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL);
   const int dim = 2, order = 2, neq = 2;

   RT_FECollection q_coll(order, dim);
   L2_FECollection p_coll(order, dim);
   FiniteElementSpace fes_q(&mesh, &q_coll, neq);        // vdim == neq
   FiniteElementSpace fes_p(&mesh, &p_coll, neq);

   GridFunction q(&fes_q), p(&fes_p);
   q = 0.0;
   p = 0.0;

   HDGPotentialPostprocessor pp(q, p);
   REQUIRE(pp.GetNumEquations() == neq);

   // The wrong layout is refused by an MFEM_VERIFY in the constructor. That is
   // not asserted here: without MFEM_USE_EXCEPTIONS the verify aborts rather
   // than throws, and a test that kills the process is worse than no test.
}
