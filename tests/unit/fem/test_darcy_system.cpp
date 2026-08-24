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

/// Which mixed space the flux lives in. RT is the hybridized-mixed method;
/// DG is the fully discontinuous one the NPC papers use, where the flux is an
/// L2 vector field and the faces carry an explicit stabilization.
enum class Form { RT, DG };

/// The branch parameterizes the stabilization as tau = td * kappa / h, so a
/// fixed td is not a fixed tau: it is tau growing like 1/h under refinement.
/// NPC-1 section 3.6.3 asks instead for eta_d = kappa/l with l a *fixed*
/// problem length scale, and the difference is not cosmetic. Measured on this
/// problem, over the same 8x8 to 16x16 pair the tests use:
///
///        td fixed (tau ~ 1/h)      td = h (tau fixed)
///   k=0  p 0.05,  u 0.02           p 1.10,  u 1.00
///   k=1  p 2.18,  u 1.18           p 1.99,  u 1.99
///
/// At k=1 the growing tau costs the flux its order, which is the textbook
/// result; at k=0 it stalls both variables, the scheme locking as tau runs
/// away. So td is set to TAU * h / L below, giving tau = TAU * kappa / L with
/// the domain size L = 1 -- the constant tau that CLAUDE.md recommends
/// trying first.
constexpr real_t TAU = 1.0;
constexpr real_t LSCALE = 1.0;

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
Result Solve(Mesh &mesh, int order, int neq, bool hybridize, int only = -1,
             Form form = Form::RT)
{
   const int dim = mesh.Dimension();
   const bool dg = (form == Form::DG);

   // The DG flux is an ordinary L2 vector field, so a system needs vdim
   // neq*dim rather than neq: the equation index is outermost, and component
   // eq*dim + d of the space is direction d of equation eq, which is the
   // layout uExact already writes and the one VectorBlockDiagonalIntegrator
   // produces when it replicates a dim-wide block down the diagonal.
   std::unique_ptr<FiniteElementCollection> u_coll;
   if (dg) { u_coll.reset(new L2_FECollection(order, dim)); }
   else    { u_coll.reset(new RT_FECollection(order, dim)); }
   L2_FECollection p_coll(order, dim);
   FiniteElementSpace fes_u(&mesh, u_coll.get(), dg ? neq * dim : neq,
                            Ordering::byNODES);
   FiniteElementSpace fes_p(&mesh, &p_coll, neq, Ordering::byNODES);

   DarcyForm darcy(&fes_u, &fes_p);

   std::vector<BilinearFormIntegrator *> mass(neq);
   std::vector<Coefficient *> ik(neq), kc(neq);
   for (int i = 0; i < neq; i++)
   {
      const int eq = (only >= 0) ? only : i;
      ik[i] = new ConstantCoefficient(1.0 / kk[eq]);
      kc[i] = new ConstantCoefficient(kk[eq]);
      mass[i] = dg ? (BilinearFormIntegrator *) new VectorMassIntegrator(*ik[i])
                : (BilinearFormIntegrator *) new VectorFEMassIntegrator(*ik[i]);
   }
   darcy.GetFluxMassForm()->AddDomainIntegrator(
      new VectorBlockDiagonalIntegrator(mass));

   MixedBilinearForm *Bform = darcy.GetFluxDivForm();
   if (dg)
   {
      Bform->AddDomainIntegrator(
         new VectorBlockDiagonalIntegrator(neq, new VectorDivergenceIntegrator));
      Bform->AddInteriorFaceIntegrator(
         new VectorBlockDiagonalIntegrator(
            neq, new TransposeIntegrator(new DGNormalTraceIntegrator(-1.))));

      // The DG method has no inter-element continuity to lean on, so the
      // faces carry the stabilization explicitly -- each equation its own,
      // since the conductivities differ.
      const real_t td = TAU * mesh.GetElementSize(0) / LSCALE;
      std::vector<BilinearFormIntegrator *> stab(neq);
      for (int i = 0; i < neq; i++)
      {
         stab[i] = new HDGDiffusionIntegrator(*kc[i], td);
      }
      darcy.GetPotentialMassForm()->AddInteriorFaceIntegrator(
         new VectorBlockDiagonalIntegrator(stab));
   }
   else
   {
      Bform->AddDomainIntegrator(
         new VectorBlockDiagonalIntegrator(neq, new VectorFEDivergenceIntegrator));
   }

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

   // The Dirichlet datum. The two spaces take it differently, and which one
   // is used is not a free choice -- see the boundary treatment sweep in
   // section 7 of doc/HDG-ROADMAP.md.
   //
   //   RT   weakly, through the flux equation's natural term. The datum would
   //        otherwise have to arrive through C^T, and C is assembled on
   //        interior faces only. This is the classical hybridized mixed
   //        condition; the price is that lambda on a boundary face is
   //        meaningless and must not be read.
   //   DG   on the trace, which is essential and carries the projection of the
   //        datum. The boundary faces join the divergence form and the
   //        stabilization to make that couple. This is the default for the
   //        fully discontinuous spaces because it leaves lambda meaning the
   //        same thing everywhere, which is what the estimator and any
   //        enriched-potential variant need.
   Array<int> bdr_ess(mesh.bdr_attributes.Max());
   bdr_ess = 1;
   const bool ess_trace = dg && hybridize;

   if (!ess_trace)
   {
      // VectorBoundaryFluxLFIntegrator already takes a VectorCoefficient and
      // lays the result out as (v*dim + k)*dof + j, which is the same
      // equation-outermost layout, so no block wrapper is needed here.
      if (dg)
      {
         darcy.GetFluxRHS()->AddBdrFaceIntegrator(
            new VectorBoundaryFluxLFIntegrator(natcoeff));
      }
      else
      {
         darcy.GetFluxRHS()->AddBoundaryIntegrator(
            new VectorFEBoundaryFluxLFIntegrator(natcoeff));
      }
   }
   else
   {
      // The factor two against the interior's one is convdiff's, and is there
      // because only one side contributes on a boundary face.
      Bform->AddBdrFaceIntegrator(
         new VectorBlockDiagonalIntegrator(
            neq, new TransposeIntegrator(new DGNormalTraceIntegrator(-2.))),
         bdr_ess);
      const real_t td = TAU * mesh.GetElementSize(0) / LSCALE;
      std::vector<BilinearFormIntegrator *> bstab(neq);
      for (int i = 0; i < neq; i++)
      {
         bstab[i] = new HDGDiffusionIntegrator(*kc[i], td);
      }
      darcy.GetPotentialMassForm()->AddBdrFaceIntegrator(
         new VectorBlockDiagonalIntegrator(bstab), bdr_ess);
   }
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
      if (ess_trace) { darcy.GetHybridization()->SetEssentialBC(bdr_ess); }
   }

   darcy.Assemble();

   BlockVector x(darcy.GetOffsets());
   x = 0.0;

   OperatorPtr A;
   Vector X, B;
   if (ess_trace)
   {
      // FormLinearSystem reads the essential trace values out of X, so it has
      // to arrive sized and carrying them.
      GridFunction tr0(&fes_t);
      tr0 = 0.0;
      auto pfun = [only](const Vector & x, Vector & v)
      {
         Vector all;
         pExact(x, all);
         if (only >= 0) { v.SetSize(1); v(0) = all(only); }
         else { v = all; }
      };
      VectorFunctionCoefficient pc(neq, pfun);
      tr0.ProjectBdrCoefficient(pc, bdr_ess);
      X = tr0;
   }
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
   for (int i = 0; i < neq; i++) { delete ik[i]; delete kc[i]; }
   return res;
}

} // namespace darcy_system

TEST_CASE("An off-diagonal block integrator places exactly one block",
          "[DarcyForm][System]")
{
   // VectorBlockDiagonalIntegrator can only replicate down the diagonal, so it
   // cannot express a coupling between different fields of a system.
   // VectorBlockIntegrator is the off-diagonal counterpart, and this pins what
   // it does: the wrapped integrator's matrix in block (row, col) of a
   // byNODES layout, zero everywhere else.
   const int dim = 2;
   const int order = GENERATE(0, 1, 2);
   const int n_test = 3, n_trial = 4;
   CAPTURE(order);

   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, false,
                                     1.0, 1.0);
   L2_FECollection tr_coll(order, dim), te_coll(order, dim);
   FiniteElementSpace fes_tr(&mesh, &tr_coll), fes_te(&mesh, &te_coll);

   const FiniteElement &trial_fe = *fes_tr.GetFE(0);
   const FiniteElement &test_fe = *fes_te.GetFE(0);
   ElementTransformation &Tr = *mesh.GetElementTransformation(0);

   // A coefficient that is not symmetric in x and y, so a transposed or
   // mirrored block cannot pass unnoticed.
   FunctionCoefficient q([](const Vector &x)
   {
      return 1.0 + x(0) + 3.0 * x(1) * x(1);
   });

   DenseMatrix ref;
   MassIntegrator plain(q);
   plain.AssembleElementMatrix2(trial_fe, test_fe, Tr, ref);
   const int h = ref.Height(), w = ref.Width();
   REQUIRE(h > 0);
   REQUIRE(w > 0);
   REQUIRE(ref.MaxMaxNorm() > 0.0);

   for (int row = 0; row < n_test; row++)
      for (int col = 0; col < n_trial; col++)
      {
         CAPTURE(row, col);
         DenseMatrix elmat;
         VectorBlockIntegrator vb(n_test, n_trial, row, col,
                                  new MassIntegrator(q));
         vb.AssembleElementMatrix2(trial_fe, test_fe, Tr, elmat);

         REQUIRE(elmat.Height() == n_test * h);
         REQUIRE(elmat.Width() == n_trial * w);

         for (int bi = 0; bi < n_test; bi++)
            for (int bj = 0; bj < n_trial; bj++)
            {
               const bool wanted = (bi == row && bj == col);
               for (int i = 0; i < h; i++)
                  for (int j = 0; j < w; j++)
                  {
                     const real_t e = elmat(bi * h + i, bj * w + j);
                     const real_t want = wanted ? ref(i, j) : 0.0;
                     if (e != want) { CAPTURE(bi, bj, i, j, e, want); }
                     REQUIRE(e == want);   // bit for bit, not merely close
                  }
            }
      }
}

TEST_CASE("An HDG face block wrapper handles unequal field counts",
          "[DarcyForm][DarcyHybridization][System][HDG]")
{
   // VectorBlockDiagonalIntegrator infers one multiplicity for every space, so
   // it cannot describe a potential space with more fields than the trace
   // space -- Stokes, where the pressure is a potential with no trace and no
   // stabilization. VectorBlockDiagonalHDGIntegrator is told both counts.
   //
   // What is checked is that each stabilized field's block lands where
   // DarcyHybridization will look for it -- element 1's fields, then element
   // 2's, then the trace's -- and that the unstabilized potential field is
   // left untouched.
   const int order = GENERATE(0, 1, 2);
   const int n_pot = 3, n_tr = 2;      // two velocities and a pressure
   CAPTURE(order);

   const int dim = 2;
   Mesh mesh = Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL, false,
                                     1.0, 1.0);
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace fes_p(&mesh, &p_coll), fes_t(&mesh, &t_coll);

   int f = -1;
   for (int i = 0; i < mesh.GetNumFaces(); i++)
   {
      if (mesh.FaceIsInterior(i)) { f = i; break; }
   }
   REQUIRE(f >= 0);
   FaceElementTransformations *ftr = mesh.GetFaceElementTransformations(f);
   const FiniteElement &el1 = *fes_p.GetFE(ftr->Elem1No);
   const FiniteElement &el2 = *fes_p.GetFE(ftr->Elem2No);
   const FiniteElement &tr_el = *fes_t.GetFaceElement(f);

   ConstantCoefficient kap(1.0);

   // The reference: one scalar HDG face matrix.
   DenseMatrix ref;
   HDGDiffusionIntegrator scalar(kap, 0.7);
   scalar.AssembleHDGFaceMatrix(tr_el, el1, el2, *ftr, ref);

   const int n1 = el1.GetDof(), n2 = el2.GetDof(), nt = tr_el.GetDof();
   REQUIRE(ref.Height() == n1 + n2 + nt);

   std::vector<BilinearFormIntegrator *> stab(n_tr);
   for (int i = 0; i < n_tr; i++)
   {
      stab[i] = new HDGDiffusionIntegrator(kap, 0.7);
   }
   VectorBlockDiagonalHDGIntegrator blocked(n_pot, n_tr, stab);

   DenseMatrix elmat;
   blocked.AssembleHDGFaceMatrix(tr_el, el1, el2, *ftr, elmat);

   const int tot = n_pot * n1 + n_pot * n2 + n_tr * nt;
   REQUIRE(elmat.Height() == tot);
   REQUIRE(elmat.Width() == tot);

   // Group offsets, in the order DarcyHybridization slices them.
   const int gdof[3] = {n1, n2, nt};
   const int gmul[3] = {n_pot, n_pot, n_tr};
   int goff[3] = {0, 0, 0}, soff[3] = {0, 0, 0};
   for (int g = 1; g < 3; g++)
   {
      goff[g] = goff[g-1] + gmul[g-1] * gdof[g-1];
      soff[g] = soff[g-1] + gdof[g-1];
   }

   // Every entry: the scalar block on field i of both groups, zero elsewhere.
   // In particular the pressure's potential rows and columns are untouched.
   for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++)
         for (int fa = 0; fa < gmul[a]; fa++)
            for (int fb = 0; fb < gmul[b]; fb++)
               for (int i = 0; i < gdof[a]; i++)
                  for (int j = 0; j < gdof[b]; j++)
                  {
                     const real_t e = elmat(goff[a] + fa * gdof[a] + i,
                                            goff[b] + fb * gdof[b] + j);
                     const bool on = (fa == fb) && (fa < n_tr);
                     const real_t want = on ? ref(soff[a] + i, soff[b] + j) : 0.0;
                     if (e != want) { CAPTURE(a, b, fa, fb, i, j, e, want); }
                     REQUIRE(e == want);
                  }

   // A boundary face is the same layout with the second group empty, which is
   // where the old wrapper produced a matrix of the wrong shape that the
   // caller then silently discarded.
   int bf = -1;
   for (int i = 0; i < mesh.GetNumFaces(); i++)
   {
      if (!mesh.FaceIsInterior(i)) { bf = i; break; }
   }
   REQUIRE(bf >= 0);
   FaceElementTransformations *btr = mesh.GetFaceElementTransformations(bf);
   const FiniteElement &bel = *fes_p.GetFE(btr->Elem1No);
   const FiniteElement &btr_el = *fes_t.GetFaceElement(bf);
   DenseMatrix belmat;
   blocked.AssembleHDGFaceMatrix(btr_el, bel, bel, *btr, belmat);
   const int btot = n_pot * bel.GetDof() + n_tr * btr_el.GetDof();
   REQUIRE(belmat.Height() == btot);
   REQUIRE(belmat.Width() == btot);
   REQUIRE(belmat.MaxMaxNorm() > 0.0);
}

TEST_CASE("A block-diagonal Darcy system reproduces its equations one by one",
          "[DarcyForm][DarcyHybridization][System]")
{
   using namespace darcy_system;

   // The first question about "systems of equations": does the vdim > 1 path
   // solve each equation as it would be solved alone? The conductivities and
   // amplitudes differ per equation, so a block mix-up cannot cancel.
   const int order = GENERATE(0, 1, 2);
   const Element::Type elem = GENERATE(Element::QUADRILATERAL,
                                       Element::TRIANGLE);
   const Form form = GENERATE(Form::RT, Form::DG);

   Mesh mesh = Mesh::MakeCartesian2D(4, 4, elem, false, 1.0, 1.0);

   const Result sys = Solve(mesh, order, NEQ, true, -1, form);

   CAPTURE(order, int(elem), int(form), sys.solved_size);

   const int np = sys.p.Size() / NEQ;
   const int nu = sys.u.Size() / NEQ;

   for (int i = 0; i < NEQ; i++)
   {
      const Result one = Solve(mesh, order, 1, true, i, form);
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

   // RT only. DarcyForm::Assemble assembles the potential faces through
   // AssemblePotHDGFaces when hybridization is on and AssemblePotLDGFaces
   // when it is off, so for a DG flux the two are not the same operator and
   // there is nothing here to compare. The equivalence is a property of the
   // hybridized mixed method, not of the branch's DG path.
   const int order = GENERATE(0, 1, 2);

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

   const int order = GENERATE(0, 1, 2);
   const Form form = GENERATE(Form::RT, Form::DG);

   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, false,
                                     1.0, 1.0);

   // Rates are taken between the two finest meshes. The coarsest pair here is
   // 2x2 to 4x4, which for the DG form is still pre-asymptotic -- it reports
   // about 1.6 where 2 is wanted, and reaches 2 on the next pair.
   std::vector<real_t> ep, eu;
   for (int ref = 0; ref < 4; ref++)
   {
      const Result r = Solve(mesh, order, NEQ, true, -1, form);
      ep.push_back(r.err_p);
      eu.push_back(r.err_u);
      mesh.UniformRefinement();
   }

   const int n = ep.size();
   const real_t rate_p = std::log2(ep[n-2] / ep[n-1]);
   const real_t rate_u = std::log2(eu[n-2] / eu[n-1]);
   CAPTURE(order, int(form), rate_p, rate_u, ep[n-1], eu[n-1]);
   REQUIRE(rate_p > order + 0.7);
   REQUIRE(rate_u > order + 0.7);
}

TEST_CASE("A Darcy system with cross-equation coupling",
          "[DarcyForm][DarcyHybridization][System]")
{
   using namespace darcy_system;

   // Block-diagonal replication is not a system in any interesting sense: the
   // equations never speak to each other. This adds a dense neq x neq
   // zeroth-order block, which is the only linear cross-equation coupling the
   // branch can express, and is the shape the potential cascade needs.
   const int order = GENERATE(0, 1, 2);
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
      const Form form = GENERATE(Form::RT, Form::DG);
      Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, false,
                                        1.0, 1.0);
      std::vector<real_t> ep, eu;
      for (int ref = 0; ref < 4; ref++)
      {
         const Result r = Solve(mesh, order, NEQ, true, -1, form);
         ep.push_back(r.err_p);
         eu.push_back(r.err_u);
         mesh.UniformRefinement();
      }

      const int n = ep.size();
      const real_t rate_p = std::log2(ep[n-2] / ep[n-1]);
      const real_t rate_u = std::log2(eu[n-2] / eu[n-1]);
      CAPTURE(order, int(form), rate_p, rate_u, ep[n-1], eu[n-1]);
      REQUIRE(rate_p > order + 0.7);
      REQUIRE(rate_u > order + 0.7);
   }

   coupled = false;
}
