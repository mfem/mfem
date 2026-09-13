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

#include <cmath>
#include <vector>

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


/** @brief A polynomial of total degree @a K in @a dim variables, and its
    gradient. Deterministic, so a failure reproduces. */
struct Poly
{
   int dim, K;
   std::vector<std::vector<int>> exps;
   std::vector<real_t> c;

   Poly(int dim_, int K_, int seed) : dim(dim_), K(K_)
   {
      unsigned long s = 1000003u * (unsigned long)(seed + 1);
      auto next = [&s]()
      {
         s = s * 6364136223846793005UL + 1442695040888963407UL;
         return 2.0 * ((real_t)((s >> 11) & 0xFFFFFFFFUL) / 4294967295.0) - 1.0;
      };
      std::vector<int> a(dim, 0);
      while (true)
      {
         int t = 0;
         for (int i = 0; i < dim; i++) { t += a[i]; }
         if (t <= K) { exps.push_back(a); c.push_back(next()); }
         int i = 0;
         for (; i < dim; i++)
         {
            if (++a[i] <= K) { break; }
            a[i] = 0;
         }
         if (i == dim) { break; }
      }
   }

   real_t Val(const Vector &x) const
   {
      real_t r = 0.0;
      for (size_t m = 0; m < exps.size(); m++)
      {
         real_t t = c[m];
         for (int i = 0; i < dim; i++) { t *= std::pow(x(i), exps[m][i]); }
         r += t;
      }
      return r;
   }

   void Grad(const Vector &x, Vector &g) const
   {
      g.SetSize(dim);
      g = 0.0;
      for (size_t m = 0; m < exps.size(); m++)
         for (int d = 0; d < dim; d++)
         {
            if (exps[m][d] == 0) { continue; }
            real_t t = c[m] * exps[m][d];
            for (int i = 0; i < dim; i++)
            {
               const int e = (i == d) ? exps[m][i] - 1 : exps[m][i];
               t *= std::pow(x(i), e);
            }
            g(d) += t;
         }
   }
};

/// The elementwise L2 projection of @a P onto @a fes, block by block.
void ProjectL2(const std::vector<Poly> &P, FiniteElementSpace &fes,
               GridFunction &gf)
{
   Mesh *mesh = fes.GetMesh();
   const int neq = fes.GetVDim();
   MassIntegrator mi;
   DenseMatrix M;
   DenseMatrixInverse Mi;
   Array<int> vd;
   for (int z = 0; z < mesh->GetNE(); z++)
   {
      const FiniteElement *fe = fes.GetFE(z);
      ElementTransformation *T = mesh->GetElementTransformation(z);
      mi.AssembleElementMatrix(*fe, *T, M);
      Mi.Factor(M);
      fes.GetElementVDofs(z, vd);
      const int nd = fe->GetDof();
      for (int e = 0; e < neq; e++)
      {
         FunctionCoefficient pc([&P, e](const Vector &x)
         { return P[e].Val(x); });
         DomainLFIntegrator lf(pc);
         lf.SetIntRule(&IntRules.Get(fe->GetGeomType(),
                                     2 * fe->GetOrder() + 4));
         Vector rhs, sol(nd);
         lf.AssembleRHSElementVect(*fe, *T, rhs);
         Mi.Mult(rhs, sol);
         for (int i = 0; i < nd; i++) { gf(vd[e * nd + i]) = sol(i); }
      }
   }
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

TEST_CASE("The postprocessing blocks reproduce a polynomial exactly",
          "[DarcyForm][Postprocess]")
{
   // The acceptance check for HDGPostprocessBlocks, and it is ARITHMETIC
   // rather than a comparison against the route it replaces. With a flux that
   // is exactly -grad P for P of degree k+1, the local Neumann problem has P
   // as its solution and the mean row pins the constant, so u* must BE P --
   // an answer that does not depend on any implementation agreeing with any
   // other. A comparison against the old element loop could not serve: the
   // cached route contracts a precomputed matrix with the flux COEFFICIENTS
   // where the old one accumulated over quadrature points from the flux
   // VALUES, so the two differ in the last digits by construction.
   using namespace darcy_postprocess;

   const int dim = GENERATE(2, 3);
   const int order = GENERATE(1, 2);
   const int neq = GENERATE(1, 2);
   CAPTURE(dim, order, neq);

   const int n = (dim == 3) ? 2 : 3;
   Mesh mesh = (dim == 3)
               ? Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON)
               : Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL);

   L2_FECollection q_coll(order, dim), p_coll(order, dim);
   L2_FECollection s_coll(order + 1, dim);
   FiniteElementSpace fes_q(&mesh, &q_coll, neq * dim, Ordering::byNODES);
   FiniteElementSpace fes_p(&mesh, &p_coll, neq, Ordering::byNODES);
   FiniteElementSpace fes_s(&mesh, &s_coll, neq, Ordering::byNODES);

   std::vector<Poly> P;
   for (int e = 0; e < neq; e++) { P.emplace_back(dim, order + 1, e); }

   // grad P has degree k, so nodal interpolation onto L2_k is exact.
   VectorFunctionCoefficient qc(neq * dim, [&P, neq, dim](const Vector &x,
                                                          Vector &v)
   {
      Vector g;
      for (int e = 0; e < neq; e++)
      {
         P[e].Grad(x, g);
         for (int d = 0; d < dim; d++) { v(e * dim + d) = -g(d); }
      }
   });
   GridFunction q(&fes_q);
   q.ProjectCoefficient(qc);

   // Only the element MEAN of the potential reaches the local problem, and
   // the L2 projection reproduces it exactly because constants lie in P^k.
   GridFunction p(&fes_p);
   ProjectL2(P, fes_p, p);

   // P itself is degree k+1 and L2_{k+1} is nodal, so this is exact.
   GridFunction pex(&fes_s);
   VectorFunctionCoefficient pv(neq, [&P, neq](const Vector &x, Vector &v)
   {
      for (int e = 0; e < neq; e++) { v(e) = P[e].Val(x); }
   });
   pex.ProjectCoefficient(pv);

   ConstantCoefficient one(1.0);
   RatioCoefficient ik(1.0, one);
   HDGPostprocessBlocks blocks(fes_q, fes_p, fes_s);
   blocks.SetDiffusionInverse(ik);
   blocks.Assemble();

   Array<int> vq, vp, vs;
   Vector lq, lp, gamma, ex;
   real_t worst = 0.0, scale = 0.0;
   for (int z = 0; z < mesh.GetNE(); z++)
   {
      fes_q.GetElementVDofs(z, vq);
      q.GetSubVector(vq, lq);
      fes_p.GetElementVDofs(z, vp);
      p.GetSubVector(vp, lp);
      fes_s.GetElementVDofs(z, vs);
      pex.GetSubVector(vs, ex);

      blocks.Apply(z, lq, lp, gamma);
      REQUIRE(gamma.Size() == ex.Size());
      for (int i = 0; i < gamma.Size(); i++)
      {
         worst = std::max(worst, std::abs(gamma(i) - ex(i)));
         scale = std::max(scale, std::abs(ex(i)));
      }
   }
   CAPTURE(worst, scale);
   REQUIRE(scale > 1e-2);              // the check has something to destroy
   REQUIRE(worst / scale < 1e-11);
}

TEST_CASE("The postprocessing blocks are the map Apply() applies",
          "[DarcyForm][Postprocess]")
{
   // Three properties of the METHOD, not restatements of the code: the two
   // accessors are the same map; B12 is rank one, because the local problem
   // is told nothing about the potential but its element average; and a move
   // of the potential that leaves that average alone leaves u* alone. Each
   // comes with the reading that must be LARGE, since a draft returning zero
   // for every input would otherwise pass all three.
   using namespace darcy_postprocess;

   const int order = GENERATE(1, 2);
   const int neq = GENERATE(1, 3);
   const int dim = 2;
   CAPTURE(order, neq);

   Mesh mesh = Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL);
   L2_FECollection q_coll(order, dim), p_coll(order, dim);
   L2_FECollection s_coll(order + 1, dim);
   FiniteElementSpace fes_q(&mesh, &q_coll, neq * dim, Ordering::byNODES);
   FiniteElementSpace fes_p(&mesh, &p_coll, neq, Ordering::byNODES);
   FiniteElementSpace fes_s(&mesh, &s_coll, neq, Ordering::byNODES);

   // Any state at all: the postprocessing is a local map and does not care
   // whether its input solves anything.
   GridFunction q(&fes_q), p(&fes_p);
   unsigned long seed = 20250912u;
   auto next = [&seed]()
   {
      seed = seed * 6364136223846793005UL + 1442695040888963407UL;
      return 2.0 * ((real_t)((seed >> 11) & 0xFFFFFFFFUL) / 4294967295.0) - 1.0;
   };
   for (int i = 0; i < fes_q.GetVSize(); i++) { q(i) = next(); }
   for (int i = 0; i < fes_p.GetVSize(); i++) { p(i) = next(); }

   ConstantCoefficient one(1.0);
   RatioCoefficient ik(1.0, one);
   HDGPostprocessBlocks blocks(fes_q, fes_p, fes_s);
   blocks.SetDiffusionInverse(ik);
   blocks.Assemble();

   Array<int> vq, vp;
   Vector lq, lp, gamma;
   DenseMatrix B11, B12;

   SECTION("Apply() is B11 u + B12 p")
   {
      real_t worst = 0.0, scale = 0.0;
      for (int z = 0; z < mesh.GetNE(); z++)
      {
         fes_q.GetElementVDofs(z, vq);
         q.GetSubVector(vq, lq);
         fes_p.GetElementVDofs(z, vp);
         p.GetSubVector(vp, lp);
         blocks.Apply(z, lq, lp, gamma);
         blocks.GetBlocks(z, B11, B12);

         const int ns = blocks.NumNodes(z);
         const int na = blocks.NumFluxDofs(z);
         const int nd = blocks.NumPotentialDofs(z);
         REQUIRE(B11.Height() == ns);
         REQUIRE(B11.Width() == na);
         REQUIRE(B12.Height() == ns);
         REQUIRE(B12.Width() == nd);

         for (int e = 0; e < neq; e++)
         {
            const Vector ue(lq.GetData() + e * na, na);
            const Vector pe(lp.GetData() + e * nd, nd);
            Vector g1(ns), g2(ns);
            B11.Mult(ue, g1);
            B12.Mult(pe, g2);
            g1 += g2;
            for (int i = 0; i < ns; i++)
            {
               worst = std::max(worst, std::abs(g1(i) - gamma(e * ns + i)));
               scale = std::max(scale, std::abs(g1(i)));
            }
         }
      }
      CAPTURE(worst, scale);
      REQUIRE(scale > 1e-3);
      REQUIRE(worst / scale < 1e-12);
   }

   SECTION("B12 is rank one and neither block is empty")
   {
      real_t minor = 0.0, s12 = 0.0, s11 = 0.0;
      for (int z = 0; z < mesh.GetNE(); z++)
      {
         blocks.GetBlocks(z, B11, B12);
         s12 = std::max(s12, B12.MaxMaxNorm());
         s11 = std::max(s11, B11.MaxMaxNorm());
         for (int i = 1; i < B12.Height(); i++)
            for (int j = 1; j < B12.Width(); j++)
            {
               const real_t m =
                  B12(i, j) * B12(0, 0) - B12(i, 0) * B12(0, j);
               minor = std::max(minor, std::abs(m));
            }
      }
      CAPTURE(minor, s12, s11);
      REQUIRE(s12 > 1e-3);             // a zero B12 has rank one too
      REQUIRE(s11 > 1e-3);
      REQUIRE(minor / (s12 * s12) < 1e-12);
   }

   SECTION("a potential move with zero element mean changes nothing")
   {
      // The sharp statement of rank one. The mean-CHANGING move beside it is
      // what says the check could have failed.
      real_t worst_zero = 0.0, worst_mean = 0.0, scale = 0.0;
      for (int z = 0; z < mesh.GetNE(); z++)
      {
         fes_q.GetElementVDofs(z, vq);
         q.GetSubVector(vq, lq);
         fes_p.GetElementVDofs(z, vp);
         p.GetSubVector(vp, lp);

         Vector g0;
         blocks.Apply(z, lq, lp, g0);

         const int nd = blocks.NumPotentialDofs(z);
         DenseMatrix b11, b12;
         blocks.GetBlocks(z, b11, b12);

         // The mass row is B12's own second factor, recovered from any
         // nonzero row of it -- so the perturbation is built from the object
         // under test and a wrong mass row would show up as a failure here
         // rather than being quietly compensated.
         int r0 = 0;
         for (int i = 0; i < b12.Height(); i++)
         {
            if (std::abs(b12(i, 0)) > std::abs(b12(r0, 0))) { r0 = i; }
         }
         Vector mass(nd);
         for (int j = 0; j < nd; j++) { mass(j) = b12(r0, j); }

         Vector lpz(lp), lpm(lp);
         for (int e = 0; e < neq; e++)
         {
            Vector del(nd);
            for (int i = 0; i < nd; i++) { del(i) = next(); }
            del.Add(-(mass * del) / (mass * mass), mass);
            for (int i = 0; i < nd; i++)
            {
               lpz(e * nd + i) += del(i);
               lpm(e * nd + i) += 1.0;
            }
         }

         Vector gz, gm;
         blocks.Apply(z, lq, lpz, gz);
         blocks.Apply(z, lq, lpm, gm);
         for (int i = 0; i < g0.Size(); i++)
         {
            worst_zero = std::max(worst_zero, std::abs(gz(i) - g0(i)));
            worst_mean = std::max(worst_mean, std::abs(gm(i) - g0(i)));
            scale = std::max(scale, std::abs(g0(i)));
         }
      }
      CAPTURE(worst_zero, worst_mean, scale);
      REQUIRE(worst_zero / scale < 1e-12);
      REQUIRE(worst_mean / scale > 1e-2);
   }
}

TEST_CASE("Compute() and the blocks it is built on agree to round-off",
          "[DarcyForm][Postprocess]")
{
   // HDGPotentialPostprocessor::Compute() IS Apply() plus a scatter, so this
   // is not an independent check of the answer -- it is a check that the
   // scatter puts each equation's block where the enriched space expects it.
   // The layout is the branch's usual one, equation outermost, and getting it
   // wrong is invisible at neq == 1.
   using namespace darcy_postprocess;

   const int order = 2, dim = 2, neq = 3;
   Mesh mesh = Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL);
   L2_FECollection q_coll(order, dim), p_coll(order, dim);
   L2_FECollection s_coll(order + 1, dim);
   FiniteElementSpace fes_q(&mesh, &q_coll, neq * dim, Ordering::byNODES);
   FiniteElementSpace fes_p(&mesh, &p_coll, neq, Ordering::byNODES);
   FiniteElementSpace fes_s(&mesh, &s_coll, neq, Ordering::byNODES);

   GridFunction q(&fes_q), p(&fes_p), ps(&fes_s);
   unsigned long seed = 777u;
   auto next = [&seed]()
   {
      seed = seed * 6364136223846793005UL + 1442695040888963407UL;
      return 2.0 * ((real_t)((seed >> 11) & 0xFFFFFFFFUL) / 4294967295.0) - 1.0;
   };
   for (int i = 0; i < fes_q.GetVSize(); i++) { q(i) = next(); }
   for (int i = 0; i < fes_p.GetVSize(); i++) { p(i) = next(); }

   ConstantCoefficient one(1.0);
   RatioCoefficient ik(1.0, one);

   HDGPotentialPostprocessor pp(q, p);
   pp.SetDiffusionInverse(ik);
   pp.Compute(ps);

   HDGPostprocessBlocks blocks(fes_q, fes_p, fes_s);
   blocks.SetDiffusionInverse(ik);
   blocks.Assemble();

   Array<int> vq, vp, vs;
   Vector lq, lp, gamma, scattered;
   real_t worst = 0.0, scale = 0.0;
   for (int z = 0; z < mesh.GetNE(); z++)
   {
      fes_q.GetElementVDofs(z, vq);
      q.GetSubVector(vq, lq);
      fes_p.GetElementVDofs(z, vp);
      p.GetSubVector(vp, lp);
      fes_s.GetElementVDofs(z, vs);
      ps.GetSubVector(vs, scattered);

      blocks.Apply(z, lq, lp, gamma);
      for (int i = 0; i < gamma.Size(); i++)
      {
         worst = std::max(worst, std::abs(gamma(i) - scattered(i)));
         scale = std::max(scale, std::abs(scattered(i)));
      }
   }
   CAPTURE(worst, scale);
   REQUIRE(scale > 1e-3);
   REQUIRE(worst / scale < 1e-12);
}
