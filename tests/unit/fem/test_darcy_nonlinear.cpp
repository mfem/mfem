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

namespace darcy_nonlinear
{

// A nonlinear Darcy problem: the conductivity depends on the potential, wired
// the way miniapps/hdg/convdiff.cpp does it -- a MixedConductionNLFIntegrator
// on DarcyForm's block nonlinear form.
//
// What is checked is the Jacobian, against a central difference of the
// residual the same object produces. That comparison is the only thing that
// finds an error in it: the Jacobian is never assembled globally in a
// hybridized method, so a wrong one does not give a wrong answer, only slow
// Newton convergence, and a passing regression suite will not notice.

void FillVarying(Vector &v, real_t shift, real_t scale = 1.0)
{
   for (int i = 0; i < v.Size(); i++)
   {
      v(i) = scale * (std::sin(1.7 * i + shift) + 0.5 * std::cos(0.3 * i));
   }
}


/// A two-equation conductivity in which every entry of the Jacobian is
/// nonzero: each equation's dual flux depends on both potentials and on both
/// fluxes. A block-diagonal or transposed Jacobian cannot survive this.
///
///   D(p) = [ 1 + p0^2/2      c(p)      ]      c(p) = 1/4 + p0 p1 / 10
///          [ c(p)         2 + p1^2/2   ]
///
/// applied to the flux rows, component by component in space.
class CoupledDiffusionFlux : public MixedFluxFunction
{
public:
   CoupledDiffusionFlux(int dim_) : MixedFluxFunction(2, dim_) { }

   real_t ComputeDualFlux(const Vector &u, const DenseMatrix &flux,
                          ElementTransformation &, DenseMatrix &df) const override
   {
      const real_t a00 = 1.0 + 0.5 * u(0) * u(0);
      const real_t a11 = 2.0 + 0.5 * u(1) * u(1);
      const real_t a01 = 0.25 + 0.1 * u(0) * u(1);

      df.SetSize(2, dim);
      for (int d = 0; d < dim; d++)
      {
         df(0, d) = a00 * flux(0, d) + a01 * flux(1, d);
         df(1, d) = a01 * flux(0, d) + a11 * flux(1, d);
      }
      return std::max(a00, a11);
   }

   real_t ComputeFlux(const Vector &, ElementTransformation &,
                      DenseMatrix &flux) const override
   {
      flux = 0.0;
      return 0.0;
   }

   void ComputeDualFluxJacobian(const Vector &u, const DenseMatrix &flux,
                                ElementTransformation &,
                                DenseMatrix &J_u, DenseMatrix &J_F) const override
   {
      const real_t a00 = 1.0 + 0.5 * u(0) * u(0);
      const real_t a11 = 2.0 + 0.5 * u(1) * u(1);
      const real_t a01 = 0.25 + 0.1 * u(0) * u(1);

      J_F.SetSize(2 * dim, 2 * dim);
      J_F = 0.0;
      J_u.SetSize(2 * dim, 2);
      J_u = 0.0;

      for (int d = 0; d < dim; d++)
      {
         J_F(0 * dim + d, 0 * dim + d) = a00;
         J_F(0 * dim + d, 1 * dim + d) = a01;
         J_F(1 * dim + d, 0 * dim + d) = a01;
         J_F(1 * dim + d, 1 * dim + d) = a11;

         J_u(0 * dim + d, 0) = u(0) * flux(0, d) + 0.1 * u(1) * flux(1, d);
         J_u(0 * dim + d, 1) = 0.1 * u(0) * flux(1, d);
         J_u(1 * dim + d, 0) = 0.1 * u(1) * flux(0, d);
         J_u(1 * dim + d, 1) = u(1) * flux(1, d) + 0.1 * u(0) * flux(0, d);
      }
   }

   /// Break one entry, to check the comparison below is measuring something.
   bool sabotage = false;
};

/// Assemble the two-equation nonlinear Darcy operator and return the
/// difference between J dy and a central difference of the residual.
real_t CoupledJacobianError(int dim, int order, MixedFluxFunction &flux)
{
   Mesh mesh = (dim == 2)
               ? Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL, false,
                                       1.0, 1.0)
               : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON,
                                       1.0, 1.0, 1.0);

   const int neq = flux.num_equations;

   RT_FECollection u_coll(order, dim);
   L2_FECollection p_coll(order, dim);
   FiniteElementSpace fes_u(&mesh, &u_coll, neq, Ordering::byNODES);
   FiniteElementSpace fes_p(&mesh, &p_coll, neq, Ordering::byNODES);

   DarcyForm darcy(&fes_u, &fes_p);
   darcy.GetBlockNonlinearForm()->AddDomainIntegrator(
      new MixedConductionNLFIntegrator(flux));
   darcy.GetFluxDivForm()->AddDomainIntegrator(
      new VectorBlockDiagonalIntegrator(neq, new VectorFEDivergenceIntegrator));

   darcy.Assemble();
   darcy.Finalize();

   BlockVector x(darcy.GetOffsets());
   FillVarying(x, 0.0, 0.3);

   Vector dy(x.Size());
   FillVarying(dy, 2.4, 0.5);

   Vector Jdy(x.Size());
   darcy.GetGradient(x).Mult(dy, Jdy);

   const real_t h = std::cbrt(std::numeric_limits<real_t>::epsilon());
   Vector xp(x), xm(x), rp(x.Size()), rm(x.Size());
   xp.Add(h, dy);
   xm.Add(-h, dy);
   darcy.Mult(xp, rp);
   darcy.Mult(xm, rm);

   Vector fd(rp);
   fd -= rm;
   fd /= (2.0 * h);

   Vector diff(Jdy);
   diff -= fd;
   return diff.Normlinf() / std::max(fd.Normlinf(), real_t(1.0));
}

} // namespace darcy_nonlinear

TEST_CASE("Nonlinear Darcy: the analytic Jacobian matches a differenced residual",
          "[DarcyForm][NonlinearDarcy]")
{
   using namespace darcy_nonlinear;

   const int dim = GENERATE(2, 3);
   const int order = GENERATE(0, 1, 2);
   CAPTURE(dim, order);

   Mesh mesh = (dim == 2)
               ? Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL, false,
                                       1.0, 1.0)
               : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON,
                                       1.0, 1.0, 1.0);

   RT_FECollection u_coll(order, dim);
   L2_FECollection p_coll(order, dim);
   FiniteElementSpace fes_u(&mesh, &u_coll);
   FiniteElementSpace fes_p(&mesh, &p_coll);

   // kappa^{-1}(p) = 1 + p^2 / 2, so the state derivative is genuinely nonzero
   // and bounded away from zero for any state.
   auto kinv  = [](const Vector &, real_t s) { return 1.0 + 0.5 * s * s; };
   auto dkinv = [](const Vector &, real_t s) { return s; };
   FunctionDiffusionFlux flux(dim, kinv, dkinv);

   DarcyForm darcy(&fes_u, &fes_p);
   darcy.GetBlockNonlinearForm()->AddDomainIntegrator(
      new MixedConductionNLFIntegrator(flux));
   darcy.GetFluxDivForm()->AddDomainIntegrator(new VectorFEDivergenceIntegrator);

   darcy.Assemble();
   darcy.Finalize();

   BlockVector x(darcy.GetOffsets());
   FillVarying(x, 0.0, 0.3);

   Vector r0(x.Size());
   darcy.Mult(x, r0);

   Vector dy(x.Size());
   FillVarying(dy, 2.4, 0.5);

   Operator &J = darcy.GetGradient(x);
   REQUIRE(J.Height() == x.Size());
   REQUIRE(J.Width() == x.Size());

   Vector Jdy(x.Size());
   J.Mult(dy, Jdy);

   // Central difference of the residual along dy. cbrt(eps) is the balance
   // point for a central difference; the residual is smooth in the state.
   const real_t h = std::cbrt(std::numeric_limits<real_t>::epsilon());

   Vector xp(x), xm(x), rp(x.Size()), rm(x.Size());
   xp.Add(h, dy);
   xm.Add(-h, dy);
   darcy.Mult(xp, rp);
   darcy.Mult(xm, rm);

   Vector fd(rp);
   fd -= rm;
   fd /= (2.0 * h);

   Vector diff(Jdy);
   diff -= fd;

   const real_t scale = std::max(fd.Normlinf(), real_t(1.0));
   INFO("||J dy - (r(x+h dy) - r(x-h dy))/2h||_inf = " << diff.Normlinf()
        << " against ||fd||_inf = " << fd.Normlinf());
   REQUIRE(diff.Normlinf() < 1e-5 * scale);
}

TEST_CASE("Nonlinear Darcy: a wrong state derivative is visible",
          "[DarcyForm][NonlinearDarcy]")
{
   using namespace darcy_nonlinear;

   // The control for the test above. If the analytic state derivative is
   // wrong, the comparison must fail -- otherwise the check is measuring
   // nothing. Here dkinv is deliberately off by a factor of two.
   const int dim = 2;
   const int order = 1;

   Mesh mesh = Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL, false,
                                     1.0, 1.0);

   RT_FECollection u_coll(order, dim);
   L2_FECollection p_coll(order, dim);
   FiniteElementSpace fes_u(&mesh, &u_coll);
   FiniteElementSpace fes_p(&mesh, &p_coll);

   auto kinv       = [](const Vector &, real_t s) { return 1.0 + 0.5 * s * s; };
   auto dkinv_bad  = [](const Vector &, real_t s) { return 2.0 * s; };
   FunctionDiffusionFlux flux(dim, kinv, dkinv_bad);

   DarcyForm darcy(&fes_u, &fes_p);
   darcy.GetBlockNonlinearForm()->AddDomainIntegrator(
      new MixedConductionNLFIntegrator(flux));
   darcy.GetFluxDivForm()->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   darcy.Assemble();
   darcy.Finalize();

   BlockVector x(darcy.GetOffsets());
   FillVarying(x, 0.0, 0.3);

   Vector dy(x.Size());
   FillVarying(dy, 2.4, 0.5);

   Vector Jdy(x.Size());
   darcy.GetGradient(x).Mult(dy, Jdy);

   const real_t h = std::cbrt(std::numeric_limits<real_t>::epsilon());
   Vector xp(x), xm(x), rp(x.Size()), rm(x.Size());
   xp.Add(h, dy);
   xm.Add(-h, dy);
   darcy.Mult(xp, rp);
   darcy.Mult(xm, rm);

   Vector fd(rp);
   fd -= rm;
   fd /= (2.0 * h);

   Vector diff(Jdy);
   diff -= fd;

   INFO("a doubled state derivative shifts J dy by " << diff.Normlinf());
   REQUIRE(diff.Normlinf() > 1e-3 * std::max(fd.Normlinf(), real_t(1.0)));
}

TEST_CASE("A coupled two-equation nonlinear Darcy Jacobian",
          "[DarcyForm][NonlinearDarcy][System]")
{
   using namespace darcy_nonlinear;

   // MixedConductionNLFIntegrator carried a scalar potential -- Vector p(1) --
   // in every assembly path, so a nonlinear constitutive law could not couple
   // equations at all. This is that generalization exercised: two equations
   // whose dual fluxes each depend on both potentials and both fluxes, checked
   // the only way an unassembled Jacobian can be, against a difference of the
   // residual the same operator produces.
   const int dim = GENERATE(2, 3);
   const int order = GENERATE(0, 1, 2);
   CAPTURE(dim, order);

   CoupledDiffusionFlux flux(dim);
   const real_t err = CoupledJacobianError(dim, order, flux);

   INFO("relative ||J dy - fd|| = " << err);
   REQUIRE(err < 1e-5);
}

TEST_CASE("A coupled nonlinear Jacobian with a broken cross term is caught",
          "[DarcyForm][NonlinearDarcy][System]")
{
   using namespace darcy_nonlinear;

   // The control. Dropping the off-diagonal block of J_F -- exactly the term
   // that a block-diagonal implementation would omit -- must be visible.
   class BrokenFlux : public CoupledDiffusionFlux
   {
   public:
      BrokenFlux(int d) : CoupledDiffusionFlux(d) { }
      void ComputeDualFluxJacobian(const Vector &u, const DenseMatrix &flux,
                                   ElementTransformation &Tr,
                                   DenseMatrix &J_u,
                                   DenseMatrix &J_F) const override
      {
         CoupledDiffusionFlux::ComputeDualFluxJacobian(u, flux, Tr, J_u, J_F);
         for (int d = 0; d < dim; d++)
         {
            J_F(0 * dim + d, 1 * dim + d) = 0.0;
            J_F(1 * dim + d, 0 * dim + d) = 0.0;
         }
      }
   };

   BrokenFlux flux(2);
   const real_t err = CoupledJacobianError(2, 1, flux);

   INFO("dropping the cross block shifts J dy by " << err);
   REQUIRE(err > 1e-3);
}

namespace darcy_nonlinear
{

/// A face and the two elements meeting at it, with everything the HDG face
/// methods of MixedConductionNLFIntegrator need to be called directly.
struct FaceFixture
{
   Mesh mesh;
   L2_FECollection p_coll;
   DG_Interface_FECollection t_coll;
   FiniteElementSpace fes_p, fes_t;
   FaceElementTransformations *Tr{};
   const FiniteElement *fe_p{}, *fe_t{};
   Array<const FiniteElement *> el;

   FaceFixture(int dim, int order)
      : mesh((dim == 2)
             ? Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL, false, 1., 1.)
             : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON, 1., 1., 1.)),
        p_coll(order, dim), t_coll(order, dim),
        fes_p(&mesh, &p_coll), fes_t(&mesh, &t_coll), el(2)
   {
      int f = -1;
      for (int i = 0; i < mesh.GetNumFaces(); i++)
      {
         if (mesh.FaceIsInterior(i)) { f = i; break; }
      }
      Tr = mesh.GetFaceElementTransformations(f);
      fe_p = fes_p.GetFE(Tr->Elem1No);
      fe_t = fes_t.GetFaceElement(f);
      el[0] = fe_p;   // el_u is only read for its order and dimension
      el[1] = fe_p;
   }

   int NP() const { return fe_p->GetDof(); }
   int NT() const { return fe_t->GetDof(); }
};

const int kAllFaceTypes =
   BlockNonlinearFormIntegrator::HDGFaceType::ELEM
   | BlockNonlinearFormIntegrator::HDGFaceType::TRACE
   | BlockNonlinearFormIntegrator::HDGFaceType::CONSTR
   | BlockNonlinearFormIntegrator::HDGFaceType::FACE;

/// Residual of the HDG face terms, potential and trace parts stacked.
void HDGFaceResidual(MixedConductionNLFIntegrator &integ, FaceFixture &fx,
                     const Vector &p, const Vector &tr, Vector &r)
{
   Vector vu, vp, vt;
   Array<Vector *> out(3);
   out[0] = &vu;
   out[1] = &vp;
   out[2] = &vt;
   Array<const Vector *> in(2);
   in[0] = &p;   // el_u's coefficients are not read by these terms
   in[1] = &p;

   integ.AssembleHDGFaceVector(kAllFaceTypes, *fx.fe_t, fx.el, *fx.Tr, tr, in,
                               out);

   r.SetSize(vp.Size() + vt.Size());
   for (int i = 0; i < vp.Size(); i++) { r(i) = vp(i); }
   for (int i = 0; i < vt.Size(); i++) { r(vp.Size() + i) = vt(i); }
}

} // namespace darcy_nonlinear

TEST_CASE("HDG face terms of a system: the Jacobian matches a difference",
          "[DarcyForm][NonlinearDarcy][System][HDG]")
{
   using namespace darcy_nonlinear;

   // The face terms of MixedConductionNLFIntegrator were single-equation.
   // What blocked generalizing them was not the indexing but a question about
   // the formulation: for a system the stabilization could be a matrix over
   // the variables. It need not be. The spatial directions are already the
   // flux function's business, so the only structure a matrix could carry is
   // over the variable index, and a scalar per variable is the natural
   // choice. That leaves every face block diagonal in the variables.
   //
   // These are the terms hybridization actually calls -- AssembleFaceVector
   // and AssembleFaceGrad are the LDG pair, used only without hybridization.
   const int dim = GENERATE(2, 3);
   const int order = GENERATE(0, 1, 2);
   CAPTURE(dim, order);

   FaceFixture fx(dim, order);
   CoupledDiffusionFlux flux(dim);
   const int neq = flux.num_equations;
   const int np = fx.NP(), nt = fx.NT();

   MixedConductionNLFIntegrator integ(flux);
   Vector taus(neq);
   taus(0) = 1.0;
   taus(1) = 1.0;                    // tau = 1 for every variable
   integ.SetVariableStabilization(taus);

   Vector p(neq * np), tr(neq * nt);
   FillVarying(p, 0.3);
   FillVarying(tr, 1.7);

   // The analytic Jacobian, as the four blocks the hybridization consumes.
   DenseMatrix A, D, E, G, H;
   Array2D<DenseMatrix *> mats(3, 3);
   mats = nullptr;
   mats(0, 0) = &A;
   mats(1, 1) = &D;
   mats(1, 2) = &E;
   mats(2, 1) = &G;
   mats(2, 2) = &H;

   Array<const Vector *> in(2);
   in[0] = &p;
   in[1] = &p;
   integ.AssembleHDGFaceGrad(kAllFaceTypes, *fx.fe_t, fx.el, *fx.Tr, tr, in,
                             mats);

   REQUIRE(D.Height() == neq * np);
   REQUIRE(E.Width() == neq * nt);
   REQUIRE(G.Height() == neq * nt);
   REQUIRE(H.Height() == neq * nt);

   // Its action on a direction in (p, tr) ...
   Vector dp(neq * np), dt(neq * nt);
   FillVarying(dp, 2.4, 0.5);
   FillVarying(dt, 0.9, 0.5);

   Vector Jd(neq * np + neq * nt);
   Jd = 0.0;
   {
      Vector top(Jd.GetData(), neq * np), bot(Jd.GetData() + neq * np, neq * nt);
      D.AddMult(dp, top);
      E.AddMult(dt, top);
      G.AddMult(dp, bot);
      H.AddMult(dt, bot);
   }

   // ... against a central difference of the residual in the same direction.
   const real_t h = std::cbrt(std::numeric_limits<real_t>::epsilon());
   Vector pp(p), pm(p), tp(tr), tm(tr), rp, rm;
   pp.Add(h, dp);
   pm.Add(-h, dp);
   tp.Add(h, dt);
   tm.Add(-h, dt);
   HDGFaceResidual(integ, fx, pp, tp, rp);
   HDGFaceResidual(integ, fx, pm, tm, rm);

   Vector fd(rp);
   fd -= rm;
   fd /= (2.0 * h);

   Vector diff(Jd);
   diff -= fd;
   const real_t rel = diff.Normlinf() / std::max(fd.Normlinf(), real_t(1.0));

   INFO("relative ||J d - fd|| = " << rel);
   REQUIRE(rel < 1e-8);
}

TEST_CASE("HDG face terms of a system are block diagonal in the variable",
          "[DarcyForm][NonlinearDarcy][System][HDG]")
{
   using namespace darcy_nonlinear;

   // The claim that makes a per-variable scalar tau sufficient: with one, no
   // face block couples different variables. If any of these off-diagonal
   // blocks were nonzero the scalar would be hiding a coupling it cannot
   // represent, and a matrix tau would be the honest choice after all.
   const int dim = GENERATE(2, 3);
   CAPTURE(dim);

   FaceFixture fx(dim, 1);
   CoupledDiffusionFlux flux(dim);
   const int neq = flux.num_equations;
   const int np = fx.NP(), nt = fx.NT();

   MixedConductionNLFIntegrator integ(flux);
   Vector taus(neq);
   taus(0) = 1.0;
   taus(1) = 2.5;
   integ.SetVariableStabilization(taus);

   Vector p(neq * np), tr(neq * nt);
   FillVarying(p, 0.3);
   FillVarying(tr, 1.7);

   DenseMatrix A, D, E, G, H;
   Array2D<DenseMatrix *> mats(3, 3);
   mats = nullptr;
   mats(0, 0) = &A;
   mats(1, 1) = &D;
   mats(1, 2) = &E;
   mats(2, 1) = &G;
   mats(2, 2) = &H;
   Array<const Vector *> in(2);
   in[0] = &p;
   in[1] = &p;
   integ.AssembleHDGFaceGrad(kAllFaceTypes, *fx.fe_t, fx.el, *fx.Tr, tr, in,
                             mats);

   auto off_diagonal_norm = [](const DenseMatrix &M, int nr, int nc)
   {
      real_t m = 0.;
      for (int bi = 0; bi < 2; bi++)
         for (int bj = 0; bj < 2; bj++)
         {
            if (bi == bj) { continue; }
            for (int i = 0; i < nr; i++)
               for (int j = 0; j < nc; j++)
               {
                  m = std::max(m, std::abs(M(bi * nr + i, bj * nc + j)));
               }
         }
      return m;
   };

   REQUIRE(off_diagonal_norm(D, np, np) == MFEM_Approx(0.0, 1e-14, 1e-14));
   REQUIRE(off_diagonal_norm(E, np, nt) == MFEM_Approx(0.0, 1e-14, 1e-14));
   REQUIRE(off_diagonal_norm(G, nt, np) == MFEM_Approx(0.0, 1e-14, 1e-14));
   REQUIRE(off_diagonal_norm(H, nt, nt) == MFEM_Approx(0.0, 1e-14, 1e-14));

   // And each diagonal block scales with its own tau, since the term is
   // linear in it. Variable 1's blocks are 2.5x variable 0's.
   for (int i = 0; i < np; i++)
      for (int j = 0; j < np; j++)
      {
         CAPTURE(i, j);
         REQUIRE(D(np + i, np + j) == MFEM_Approx(2.5 * D(i, j), 1e-12, 1e-13));
      }
   for (int i = 0; i < nt; i++)
      for (int j = 0; j < nt; j++)
      {
         CAPTURE(i, j);
         REQUIRE(H(nt + i, nt + j) == MFEM_Approx(2.5 * H(i, j), 1e-12, 1e-13));
      }
}

TEST_CASE("A per-variable tau does not disturb the single-equation path",
          "[DarcyForm][NonlinearDarcy][System][HDG]")
{
   using namespace darcy_nonlinear;

   // One scalar equation has to keep going down exactly the route it went
   // down before: the stabilization derived from the inverse flux Jacobian,
   // with the tau vector ignored. Setting a wild one must change nothing.
   const int dim = GENERATE(2, 3);
   const int order = GENERATE(0, 1, 2);
   CAPTURE(dim, order);

   FaceFixture fx(dim, order);
   ConstantCoefficient kappa(2.75);
   LinearDiffusionFlux flux(dim, kappa);
   REQUIRE(flux.num_equations == 1);

   const int np = fx.NP(), nt = fx.NT();
   Vector p(np), tr(nt);
   FillVarying(p, 0.3);
   FillVarying(tr, 1.7);

   MixedConductionNLFIntegrator plain(flux);
   Vector r_plain;
   HDGFaceResidual(plain, fx, p, tr, r_plain);

   MixedConductionNLFIntegrator stabilized(flux);
   Vector taus(1);
   taus(0) = 37.0;
   stabilized.SetVariableStabilization(taus);
   Vector r_stab;
   HDGFaceResidual(stabilized, fx, p, tr, r_stab);

   REQUIRE(r_plain.Size() == r_stab.Size());
   for (int i = 0; i < r_plain.Size(); i++)
   {
      CAPTURE(i);
      REQUIRE(r_plain(i) == r_stab(i));   // bit for bit, not merely close
   }
}

TEST_CASE("HDG face terms with a velocity: the Jacobian matches its residual",
          "[DarcyForm][NonlinearDarcy][HDG]")
{
   using namespace darcy_nonlinear;

   // AssembleHDGFaceGrad had a live upwinding branch whose counterpart in
   // AssembleHDGFaceVector is commented out, so with a velocity coefficient
   // the two disagreed: b + a came to beta + alpha/2 in the gradient against
   // beta in the residual, which for this constructor (beta = alpha/2) is a
   // clean factor of two, at every quadrature point and any velocity. It is
   // reachable only through an upwinded nonlinear diffusion under
   // hybridization, which convdiff.cpp never builds -- so nothing was wrong
   // in any example, and Newton would simply have halved its steps.
   const int dim = GENERATE(2, 3);
   CAPTURE(dim);

   FaceFixture fx(dim, 1);
   ConstantCoefficient kappa(1.0);
   LinearDiffusionFlux flux(dim, kappa);

   Vector vel(dim);
   vel = 1.0;
   VectorConstantCoefficient vcoeff(vel);
   MixedConductionNLFIntegrator integ(flux, vcoeff, 0.7);

   const int np = fx.NP(), nt = fx.NT();
   Vector p(np), tr(nt);
   FillVarying(p, 0.3);
   FillVarying(tr, 1.7);

   DenseMatrix A, D, E, G, H;
   Array2D<DenseMatrix *> mats(3, 3);
   mats = nullptr;
   mats(0, 0) = &A;
   mats(1, 1) = &D;
   mats(1, 2) = &E;
   mats(2, 1) = &G;
   mats(2, 2) = &H;
   Array<const Vector *> in(2);
   in[0] = &p;
   in[1] = &p;
   integ.AssembleHDGFaceGrad(kAllFaceTypes, *fx.fe_t, fx.el, *fx.Tr, tr, in,
                             mats);

   Vector dp(np), dt(nt);
   FillVarying(dp, 2.4, 0.5);
   FillVarying(dt, 0.9, 0.5);

   Vector Jd(np + nt);
   Jd = 0.0;
   {
      Vector top(Jd.GetData(), np), bot(Jd.GetData() + np, nt);
      D.AddMult(dp, top);
      E.AddMult(dt, top);
      G.AddMult(dp, bot);
      H.AddMult(dt, bot);
   }

   const real_t h = std::cbrt(std::numeric_limits<real_t>::epsilon());
   Vector pp(p), pm(p), tp(tr), tm(tr), rp, rm;
   pp.Add(h, dp);
   pm.Add(-h, dp);
   tp.Add(h, dt);
   tm.Add(-h, dt);
   HDGFaceResidual(integ, fx, pp, tp, rp);
   HDGFaceResidual(integ, fx, pm, tm, rm);

   Vector fd(rp);
   fd -= rm;
   fd /= (2.0 * h);

   REQUIRE(fd.Normlinf() > 1e-3);      // the terms are not simply absent

   Vector diff(Jd);
   diff -= fd;
   const real_t rel = diff.Normlinf() / fd.Normlinf();
   INFO("relative ||J d - fd|| = " << rel << " (was 1, a factor of two)");
   REQUIRE(rel < 1e-8);
}

namespace darcy_nonlinear
{

int g_neq = 2;

void gCoupled(const Vector &x, Vector &g)
{
   g.SetSize(g_neq);
   real_t s = 1.0;
   for (int d = 0; d < x.Size(); d++) { s *= std::sin(M_PI * x(d)); }
   g(0) = s;
   if (g_neq > 1) { g(1) = -0.7 * s; }
}

/// A diffusion matrix whose dependence on the potential is scaled by eps, so
/// the nonlinearity can be turned down continuously to nothing while every
/// other property of the problem is held fixed.
class ScaledCoupledFlux : public MixedFluxFunction
{
   real_t eps;

   void Entries(const Vector &u, real_t &a00, real_t &a11, real_t &a01) const
   {
      a00 = 1.0 + eps * 0.5 * u(0) * u(0);
      a11 = 2.0 + eps * 0.5 * u(1) * u(1);
      a01 = 0.25 + eps * 0.1 * u(0) * u(1);
   }

public:
   ScaledCoupledFlux(int dim_, real_t e)
      : MixedFluxFunction(2, dim_), eps(e) { }

   real_t ComputeDualFlux(const Vector &u, const DenseMatrix &flux,
                          ElementTransformation &, DenseMatrix &df) const override
   {
      real_t a00, a11, a01;
      Entries(u, a00, a11, a01);
      df.SetSize(2, dim);
      for (int d = 0; d < dim; d++)
      {
         df(0, d) = a00 * flux(0, d) + a01 * flux(1, d);
         df(1, d) = a01 * flux(0, d) + a11 * flux(1, d);
      }
      return std::max(a00, a11);
   }

   real_t ComputeFlux(const Vector &, ElementTransformation &,
                      DenseMatrix &flux) const override
   { flux = 0.0; return 0.0; }

   void ComputeDualFluxJacobian(const Vector &u, const DenseMatrix &flux,
                                ElementTransformation &,
                                DenseMatrix &J_u, DenseMatrix &J_F) const override
   {
      real_t a00, a11, a01;
      Entries(u, a00, a11, a01);
      J_F.SetSize(2*dim, 2*dim);
      J_F = 0.0;
      J_u.SetSize(2*dim, 2);
      J_u = 0.0;
      for (int d = 0; d < dim; d++)
      {
         J_F(d, d)             = a00;
         J_F(d, dim + d)       = a01;
         J_F(dim + d, d)       = a01;
         J_F(dim + d, dim + d) = a11;

         J_u(d, 0)       = eps * (u(0)*flux(0,d) + 0.1*u(1)*flux(1,d));
         J_u(d, 1)       = eps * (0.1*u(0)*flux(1,d));
         J_u(dim + d, 0) = eps * (0.1*u(1)*flux(0,d));
         J_u(dim + d, 1) = eps * (u(1)*flux(1,d) + 0.1*u(0)*flux(0,d));
      }
   }
};

/// What the trace space and the reduced operator came out as, for the one
/// case below that is about sizes rather than about residuals. Filled only
/// when asked for.
struct TraceSizes
{
   int vsize = 0, tvsize = 0, op_height = 0, op_width = 0, rhs_size = 0;
   bool cP = false;
   /// Return as soon as these are filled, leaving r0 and r1 untouched.
   bool stop_after_form = false;
};

/// One Newton step on a nonlinear DG system under hybridization, returning
/// the residual before and after. One step is enough to see what the local
/// Jacobian is worth, and avoids the drift a stalled Newton shows if it is
/// allowed to keep iterating.
void OneNewtonStep(Mesh &mesh, int order, MixedFluxFunction &flux,
                   real_t &r0, real_t &r1, Vector *p_out = nullptr,
                   TraceSizes *sizes = nullptr)
{
   const int dim = mesh.Dimension();
   const int neq = flux.num_equations;

   L2_FECollection u_coll(order, dim), p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace fes_u(&mesh, &u_coll, neq * dim, Ordering::byNODES);
   FiniteElementSpace fes_p(&mesh, &p_coll, neq, Ordering::byNODES);
   FiniteElementSpace fes_t(&mesh, &t_coll, neq, Ordering::byNODES);

   DarcyForm darcy(&fes_u, &fes_p);

   BlockNonlinearForm *Mnl = darcy.GetBlockNonlinearForm();
   Mnl->AddDomainIntegrator(new MixedConductionNLFIntegrator(flux));

   // tau = 1 for every variable, which is where the NPC papers say to start
   // and what SetVariableStabilization defaults to.
   {
      auto *face = new MixedConductionNLFIntegrator(flux);
      Vector taus(neq);
      taus = 1.0;
      face->SetVariableStabilization(taus);
      Mnl->AddInteriorFaceIntegrator(face);
   }

   MixedBilinearForm *Bform = darcy.GetFluxDivForm();
   Bform->AddDomainIntegrator(
      new VectorBlockDiagonalIntegrator(neq, new VectorDivergenceIntegrator));
   Bform->AddInteriorFaceIntegrator(
      new VectorBlockDiagonalIntegrator(
         neq, new TransposeIntegrator(new DGNormalTraceIntegrator(-1.))));

   g_neq = neq;
   VectorFunctionCoefficient gcoeff(neq, gCoupled);
   darcy.GetPotentialRHS()->AddDomainIntegrator(
      new VectorDomainLFIntegrator(gcoeff));

   Array<int> ess;
   darcy.EnableHybridization(
      &fes_t,
      new VectorBlockDiagonalIntegrator(neq, new NormalTraceJumpIntegrator),
      ess);

   darcy.Assemble();

   // The element-local solves are nonlinear too, and get their own Newton.
   darcy.GetHybridization()->SetLocalNLSolver(
      DarcyHybridization::LSsolveType::Newton, 100, 1e-13, 1e-15, -1);

   BlockVector x(darcy.GetOffsets());
   x = 0.0;

   OperatorPtr op;
   Vector X, RHS;
   darcy.FormLinearSystem(ess, x, op, X, RHS, true);

   if (sizes)
   {
      sizes->vsize     = fes_t.GetVSize();
      sizes->tvsize    = fes_t.GetTrueVSize();
      sizes->cP        = (fes_t.GetConformingProlongation() != nullptr);
      sizes->op_height = op->Height();
      sizes->op_width  = op->Width();
      sizes->rhs_size  = RHS.Size();
      // Nothing below this point runs when only the sizes were asked for.
      // The one caller that asks is about how long the operator is, and it
      // is on a mesh whose SOLVE hits a separate defect; see it.
      if (sizes->stop_after_form) { return; }
   }

   GSSmoother prec;
   GMRESSolver lin;
   lin.SetKDim(500);
   lin.SetMaxIter(2000);
   lin.SetRelTol(1e-12);
   lin.SetAbsTol(0.0);
   lin.SetPreconditioner(prec);

   NewtonSolver newton;
   newton.SetSolver(lin);
   newton.SetOperator(*op);
   newton.SetRelTol(0.0);
   newton.SetAbsTol(0.0);
   newton.SetMaxIter(1);          // exactly one step
   newton.SetPrintLevel(-1);
   newton.Mult(RHS, X);

   Vector res(X.Size());
   op->Mult(X, res);
   res -= RHS;

   r0 = newton.GetInitialNorm();
   r1 = res.Norml2();

   if (p_out)
   {
      darcy.RecoverFEMSolution(X, x);
      *p_out = x.GetBlock(1);
   }
}

} // namespace darcy_nonlinear

TEST_CASE("A nonlinear DG system assembles and solves under hybridization",
          "[DarcyForm][NonlinearDarcy][System][HDG]")
{
   using namespace darcy_nonlinear;

   // A two-equation nonlinear system, fully discontinuous, hybridized, with
   // the per-variable stabilization at its default of one. With the state
   // dependence switched off the problem is linear, so one Newton step has to
   // land on the answer exactly -- which is the check that the whole path
   // assembles consistently for neq > 1.
   const int order = GENERATE(0, 1, 2);
   CAPTURE(order);

   Mesh mesh = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL, false,
                                     1.0, 1.0);

   ScaledCoupledFlux linear(2, 0.0);
   real_t r0 = 0., r1 = 0.;
   Vector p;
   OneNewtonStep(mesh, order, linear, r0, r1, &p);

   CAPTURE(r0, r1);
   REQUIRE(r0 > 1e-3);                 // the source really is in there
   REQUIRE(r1 < 1e-11 * r0);           // and one step solves it
   REQUIRE(p.Normlinf() > 1e-4);       // to something that is not zero
}

TEST_CASE("The reduced trace operator is sized in the trace's TRUE dofs",
          "[DarcyForm][NonlinearDarcy][HDG]")
{
   using namespace darcy_nonlinear;

   /* Hybridization's constructor announces c_fes.GetVSize() and every entry
      point of DarcyHybridization works in the trace's TRUE unknowns. On a
      conforming trace space those are the same number, which is why this
      went unnoticed -- and a DG trace on a NONCONFORMING mesh is not the
      conforming case, because DG_Interface_FECollection derives from
      RT_FECollection and reports GetContType() == NORMAL, so
      FiniteElementSpace::BuildConformingInterpolation() does not take its
      early exit for it.

      The consequence was a read past the end rather than a wrong answer:
      NewtonSolver sizes its correction from Width(), the Krylov solver sizes
      its own vectors from the assembled gradient, and GMRES's Update() ran
      off them -- 512 invalid reads under valgrind on convdiff, with the
      printed answer correct to six digits because the surplus entries sat
      past the solution vector's own length. So a residual comparison cannot
      see this and a SIZE comparison is the whole test.

      The first two requirements are the control: on a mesh whose trace space
      happens to be conforming there is nothing here to get wrong, and the
      case would pass without asserting anything. */
   const int order = GENERATE(1, 2);
   CAPTURE(order);

   Mesh mesh("../../data/amr-quad.mesh");
   mesh.UniformRefinement();

   /* The flux law is LINEAR and the integrator is a nonlinear one, which is
      the combination this case needs: IsNonlinear() is what makes DarcyForm
      hand out the hybridization ITSELF instead of an assembled SparseMatrix,
      and the SparseMatrix was always sized correctly -- so a genuinely
      linear problem cannot see this at all.

      **The nonlinear FACE constraint is ON, and it did not used to be.**
      DarcyHybridization::Finalize() refused one on a mesh with hanging nodes
      until the element-major loops learned to visit the slave sub-faces, so
      this case had to ask for a configuration that was admitted and said so
      at length. It no longer does, and the case is the stronger for it: the
      operator whose size is being asked about is now the one a caller would
      actually build.

      It still stops after FormLinearSystem(), which is a separate matter --
      see the note in OneNewtonStep().

      The behaviour half is covered elsewhere: `convdiff -m
      ../../data/amr-quad.mesh -r 1 -o 1 -dg -hb -nl -nld -nls 3` reports
      zero valgrind errors after the fix against 512 before, at order 1 and
      order 3, with the two error norms identical to every printed digit. */
   ConstantCoefficient one(1.0);
   LinearDiffusionFlux linear(mesh.Dimension(), one);
   real_t r0 = 0., r1 = 0.;
   TraceSizes ts;
   ts.stop_after_form = true;
   OneNewtonStep(mesh, order, linear, r0, r1, nullptr, &ts);

   CAPTURE(ts.vsize, ts.tvsize, ts.cP);
   REQUIRE(ts.cP);                     // the mesh really is hanging-noded
   REQUIRE(ts.tvsize < ts.vsize);      // and the two sizes really differ

   CAPTURE(ts.op_height, ts.op_width, ts.rhs_size);
   REQUIRE(ts.op_height == ts.tvsize);
   REQUIRE(ts.op_width == ts.tvsize);
   // The invariant the size exists to satisfy: the operator is exactly as
   // long as the right-hand side that was reduced for it.
   REQUIRE(ts.rhs_size == ts.op_height);
}

TEST_CASE("A nonconforming master face is not a boundary face",
          "[DarcyForm][NonlinearDarcy][HDG]")
{
   /* The condition DarcyHybridization::GetNCMasterSlaves() exists to detect,
      asserted on the mesh itself so that it runs in every build.

      Every element-major face loop in the hybridization used to decide
      "boundary" by `Elem2No < 0`, which is Mesh::FaceIsInterior() inverted.
      On a conforming mesh that is exactly the boundary faces. On a
      nonconforming one a MASTER face -- the coarse side of a hanging node --
      also has no second element, and Mesh::GetFaceToBdrElMap() maps it to
      -1, so the boundary branch reached GetBdrAttribute(-1) and dereferenced
      Mesh::boundary at -1. The failure was a segfault and not a wrong
      answer, and it fired with NO boundary integrator installed, the
      attribute being read before the loop that would have been empty.

      Those loops now test for a master FIRST and expand it onto its slave
      sub-faces; the third section is the end-to-end statement of that, and
      "Frozen and Live agree at a hanging node" below is what says the answer
      is right rather than merely produced.

      The two counts are the whole of the first two sections, and the
      conforming arm is what makes the nonconforming one mean something: the
      same accessor pair returns nothing surprising there. */
   auto count_master_slots = [](Mesh &mesh)
   {
      Array<int> f_2_b = mesh.GetFaceToBdrElMap();
      Array<int> faces, oris;
      int n_master = 0;
      for (int el = 0; el < mesh.GetNE(); el++)
      {
         if (mesh.Dimension() == 2) { mesh.GetElementEdges(el, faces, oris); }
         else { mesh.GetElementFaces(el, faces, oris); }
         for (int j = 0; j < faces.Size(); j++)
         {
            int el1, el2;
            mesh.GetFaceElements(faces[j], &el1, &el2);
            if (el2 < 0 && f_2_b[faces[j]] < 0) { n_master++; }
         }
      }
      return n_master;
   };

   SECTION("a hanging-node mesh has element faces that are neither")
   {
      Mesh mesh("../../data/amr-quad.mesh");
      mesh.UniformRefinement();
      REQUIRE(mesh.Nonconforming());
      REQUIRE(count_master_slots(mesh) > 0);
   }

   SECTION("a conforming mesh has none")
   {
      Mesh mesh = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL);
      REQUIRE_FALSE(mesh.Nonconforming());
      REQUIRE(count_master_slots(mesh) == 0);
   }

   /* And the end of the story: such a mesh now carries a nonlinear face
      constraint. This section is where the crash was -- a BLOCK face
      integrator, so it exercises the c_nlfi half of the expansion, which the
      Frozen-against-Live case below cannot reach (nothing in the tree fills
      c_nlfi through DarcyForm, and that case goes through it).

      It asserts only that the system is built and is the right size, which
      is all a "does not fall over" section can honestly claim. What the
      answer is worth is the next case's business. */
   SECTION("and a nonlinear face constraint on such a mesh now builds")
   {
      using namespace darcy_nonlinear;
      Mesh mesh("../../data/amr-quad.mesh");
      mesh.UniformRefinement();

      ConstantCoefficient one(1.0);
      LinearDiffusionFlux linear(mesh.Dimension(), one);
      real_t r0 = 0., r1 = 0.;

      TraceSizes ts;
      ts.stop_after_form = true;
      OneNewtonStep(mesh, 1, linear, r0, r1, nullptr, &ts);

      CAPTURE(ts.vsize, ts.tvsize, ts.op_height);
      REQUIRE(ts.cP);                      // the mesh really is hanging-noded
      REQUIRE(ts.tvsize < ts.vsize);
      REQUIRE(ts.op_height == ts.tvsize);
      REQUIRE(ts.rhs_size == ts.op_height);
   }
}

TEST_CASE("A one-sided master integral is not the compounded slave sum",
          "[DarcyForm][NonlinearDarcy][HDG]")
{
   /* The attribution that decided how a nonlinear face constraint on a mesh
      with hanging nodes had to be built.

      The linear route is face-major: a face term at a hanging node is
      integrated on each SLAVE sub-face, in the slave's own geometry and
      quadrature, and transferred onto the master's dofs. The element-major
      nonlinear route reaches only the MASTER face, and the repair that looks
      cheapest -- integrate it one-sidedly there and rescale -- was built,
      measured wrong by six percent, and reverted. The loops visit the slaves
      instead, which is what this case's factors say they must.

      This is why, and it is geometry rather than state. The built-in
      stabilization is tau ~ q_e/h_e, formed as
      `wq = ip.weight * |nor|^2 / |J_el|`, which is QUADRATIC in the face scale
      where tiling a master with its slaves supplies only one factor of it. So
      the one-sided integral and the compounded sum differ by exact rational
      factors, and -- the point of the case -- the elem-trace block and the
      trace-trace block want DIFFERENT ones. No rescaling of a one-sided master
      integral can reproduce the slave sum for both at once, which is what
      forces a repair to visit the slaves.

      Nothing nonlinear is needed to see it: a constant coefficient and an
      ordinary HDGDiffusionIntegrator reproduce the whole thing. And no
      transfer matrix is needed either, because the CONSTANT trace function is
      transferred to itself (I*1 = 1) -- checked here rather than assumed, by
      requiring the trace prolongation to carry an all-ones true-dof vector to
      an all-ones ldof vector. */

   const int order = GENERATE(1, 2, 3);

   auto factors = [order](const char *mesh_file, int ref,
                          real_t &ratio_E, real_t &ratio_H, int &n_master)
   {
      Mesh mesh(mesh_file);
      for (int i = 0; i < ref; i++) { mesh.UniformRefinement(); }
      REQUIRE(mesh.Nonconforming());

      const int dim = mesh.Dimension();
      const int num_faces = mesh.GetNumFaces();

      L2_FECollection l2_fec(order, dim);
      FiniteElementSpace fes_p(&mesh, &l2_fec);
      DG_Interface_FECollection tr_fec(order, dim);
      FiniteElementSpace c_fes(&mesh, &tr_fec);

      // I*1 = 1, which is what lets the constant stand in for the transfer
      const SparseMatrix *cP = c_fes.GetConformingProlongation();
      REQUIRE(cP != nullptr);
      {
         Vector t(c_fes.GetTrueVSize()); t = 1.0;
         Vector l(c_fes.GetVSize());
         cP->Mult(t, l);
         l -= 1.0;
         REQUIRE(l.Normlinf() == MFEM_Approx(0.0));
      }

      ConstantCoefficient one(1.0);
      HDGDiffusionIntegrator dif(one);

      Array<int> f2b = mesh.GetFaceToBdrElMap();
      const NCMesh::NCList &nclist = mesh.ncmesh->GetNCList(dim - 1);

      ratio_E = ratio_H = 0.0;
      n_master = 0;

      for (int f = 0; f < num_faces; f++)
      {
         int el1, el2;
         mesh.GetFaceElements(f, &el1, &el2);
         if (!(el2 < 0 && f2b[f] < 0)) { continue; }   // not a master face

         // the one-sided integral over the master face
         FaceElementTransformations *ftr_m =
            mesh.GetFaceElementTransformations(f);
         const FiniteElement *tr_m = c_fes.GetFaceElement(f);
         const FiniteElement *fe_m = fes_p.GetFE(ftr_m->Elem1No);
         const int nm = fe_m->GetDof(), cd = tr_m->GetDof();

         DenseMatrix em_m;
         dif.AssembleHDGFaceMatrix(0, *tr_m, *fe_m, *ftr_m, em_m);
         REQUIRE(em_m.Height() == nm + cd);

         Vector ones(cd); ones = 1.0;
         DenseMatrix H_m(cd), E_m(nm, cd);
         H_m.CopyMN(em_m, cd, cd, nm, nm);
         E_m.CopyMN(em_m, nm, cd, 0, nm);
         Vector v(cd), u(nm);
         H_m.Mult(ones, v);
         E_m.Mult(ones, u);
         const real_t one_H = ones * v;
         real_t one_E = 0.0;
         for (int i = 0; i < nm; i++) { one_E += u(i); }

         // the same two blocks compounded over this master's slaves
         real_t sum_H = 0.0, sum_E = 0.0;
         int n_slaves = 0;
         for (int si = 0; si < nclist.slaves.Size(); si++)
         {
            const NCMesh::Slave &sl = nclist.slaves[si];
            if (sl.master != f || sl.index >= num_faces) { continue; }
            n_slaves++;

            FaceElementTransformations *ftr_s =
               mesh.GetFaceElementTransformations(sl.index);
            REQUIRE(ftr_s->Elem2No >= 0);
            const FiniteElement *tr_s = c_fes.GetFaceElement(sl.index);
            const FiniteElement *fe1 = fes_p.GetFE(ftr_s->Elem1No);
            const FiniteElement *fe2 = fes_p.GetFE(ftr_s->Elem2No);
            const int n1 = fe1->GetDof(), n2 = fe2->GetDof();

            DenseMatrix em_s;
            dif.AssembleHDGFaceMatrix(*tr_s, *fe1, *fe2, *ftr_s, em_s);

            const bool coarse_is_2 = (ftr_s->Elem2No == el1);
            REQUIRE((coarse_is_2 || ftr_s->Elem1No == el1));
            const int off = coarse_is_2 ? n1 : 0;
            const int nc = coarse_is_2 ? n2 : n1;
            REQUIRE(nc == nm);

            DenseMatrix H_s(cd), E_s(nc, cd);
            H_s.CopyMN(em_s, cd, cd, n1 + n2, n1 + n2);
            E_s.CopyMN(em_s, nc, cd, off, n1 + n2);
            Vector w(cd), z(nc);
            H_s.Mult(ones, w);
            E_s.Mult(ones, z);
            sum_H += ones * w;
            for (int i = 0; i < nc; i++) { sum_E += z(i); }
         }
         if (n_slaves == 0) { continue; }

         REQUIRE(sum_E != MFEM_Approx(0.0));
         REQUIRE(sum_H != MFEM_Approx(0.0));
         const real_t rE = one_E / sum_E, rH = one_H / sum_H;
         if (n_master == 0) { ratio_E = rE; ratio_H = rH; }
         else
         {
            // every master face on these meshes carries the same factor
            REQUIRE(rE == MFEM_Approx(ratio_E));
            REQUIRE(rH == MFEM_Approx(ratio_H));
         }
         n_master++;
      }
   };

   SECTION("in two dimensions the factors are 2 and 1:2.5")
   {
      real_t rE = 0., rH = 0.;
      int n_master = 0;
      factors("../../data/amr-quad.mesh", 1, rE, rH, n_master);
      REQUIRE(n_master > 0);
      // the coarse element's own constraint: |nor| halves, |J| does not
      REQUIRE(rE == MFEM_Approx(2.0));
      // the trace row carries the FINE side's tau too, whose |J| is quartered
      REQUIRE(rH == MFEM_Approx(1.0 / 2.5));
   }

   SECTION("in three dimensions they are 4 and 1:2.25")
   {
      real_t rE = 0., rH = 0.;
      int n_master = 0;
      factors("../../data/amr-hex.mesh", 0, rE, rH, n_master);
      REQUIRE(n_master > 0);
      REQUIRE(rE == MFEM_Approx(4.0));
      REQUIRE(rH == MFEM_Approx(1.0 / 2.25));
   }

   /* The two sections together are the discriminator, and either alone would
      be worth much less: it is the factors DIFFERING between the two blocks
      and MOVING with the dimension that says no fixed rescaling exists. */
}

TEST_CASE("Frozen and Live agree at a hanging node",
          "[DarcyForm][NonlinearDarcy][HDG]")
{
   /* THE acceptance test for a nonlinear face constraint on a mesh with
      hanging nodes, and it needs no exact solution, no solve and no tolerance
      of its own.

      An HDGDiffusionIntegrator on the potential mass form is the SAME
      discrete problem whichever route assembles it. FaceConstraintMode::Frozen
      sends it down the face-major linear route, which has always been
      nonconforming-aware -- it integrates on each SLAVE sub-face and transfers
      the blocks onto the master's dofs. FaceConstraintMode::Live sends it down
      the element-major nonlinear route, whose coarse element sees only the
      MASTER face. So applying both operators to one fixed trace vector must
      agree to round-off, and it is the element-major route's expansion onto
      the slaves that makes them.

      BOTH the residual and the gradient are compared, because a repair can be
      right in one and wrong in the other -- this branch has paid for that
      twice, and here the two halves are genuinely separate code: the residual
      carries its trace row back with `I^T` in MultNL() while the gradient
      carries E and G onto the master's slot in AssembleHDGGrad(), and H moves
      in neither.

      THE CASE DISCRIMINATES, measured by breaking each of the three pieces in
      turn and watching which half fails. On amr-quad at order 1, one
      refinement (amr-hex in brackets), against a control of 4.4e-16 residual
      and 2.8e-17 gradient:

        visit only the FIRST slave    residual 3.0e-01 [1.0e-02],
                                      gradient 3.5e-03 [4.8e-06]
        drop the E/G transfer         residual UNCHANGED at 4.4e-16,
                                      gradient 1.1e-02 [3.1e-05]
        drop the `I^T` carry-back     residual 3.3e-01 [1.1e-02],
                                      gradient UNCHANGED at 2.8e-17

      The two "UNCHANGED" rows are the sharp part: they say the two halves are
      independent and that neither is echoing the other.

      The one-sided master integral that was tried first -- integrate the
      master face as a one-sided interior face rather than visiting its slaves
      -- fails this at 3.9e-01, relative 5.8e-02. "A one-sided master integral
      is not the compounded slave sum" above is the attribution of that number
      and of why no rescaling could have repaired it.

      M_p CARRIES AN INERT DOMAIN INTEGRATOR IN BOTH ARMS, and that is
      load-bearing rather than tidy. DarcyForm::Assemble() reaches its
      to-live branch only when M_p exists AND the nonlinear potential mass
      carries face integrators; with no M_p a LINEAR face integrator on the
      nonlinear form is routed to the FROZEN slots whatever the mode says. The
      first version of this comparison had no M_p, so neither arm was live, the
      two agreed at 2.2e-16 on every mesh, and it measured nothing. */
   const int order = GENERATE(1, 2);
   CAPTURE(order);

   struct Result { Vector r_tr, Sy; };

   auto run = [order](DarcyForm::FaceConstraintMode mode, Mesh &mesh,
                      int n_frozen, int n_live)
   {
      const int dim = mesh.Dimension();

      L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
      L2_FECollection p_coll(order, dim);
      DG_Interface_FECollection t_coll(order, dim);
      FiniteElementSpace Vh(&mesh, &u_coll, dim), Wh(&mesh, &p_coll),
                         Mh(&mesh, &t_coll);

      DarcyForm darcy(&Vh, &Wh);
      darcy.SetFaceConstraintMode(mode);

      ConstantCoefficient one(1.0), zero(0.0);
      darcy.GetFluxMassForm()->AddDomainIntegrator(
         new VectorMassIntegrator(one));
      darcy.GetFluxDivForm()->AddDomainIntegrator(
         new VectorDivergenceIntegrator());
      Array<int> all(mesh.bdr_attributes.Max());
      all = 1;
      darcy.GetFluxDivForm()->AddBdrFaceIntegrator(
         new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)), all);

      // The inert filler; see the note at the top of the case.
      BilinearForm *M_p = darcy.GetPotentialMassForm();
      M_p->AddDomainIntegrator(new MassIntegrator(zero));

      /* The integrators that matter, @a n_frozen on the linear form and
         @a n_live on the nonlinear one. Copies of ONE term, so any split of a
         given total between the two slots is the same discrete problem --
         which is what lets the half-frozen section below compare 1 + 1
         against 2 + 0. */
      for (int i = 0; i < n_frozen; i++)
      {
         M_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
      }
      for (int i = 0; i < n_live; i++)
      {
         darcy.GetPotentialMassNonlinearForm()->AddInteriorFaceIntegrator(
            new HDGDiffusionIntegrator(one, 1.0));
      }

      Array<int> ess_flux;
      darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(),
                                ess_flux);
      darcy.GetHybridization()->EnableNPC();

      darcy.Assemble();
      darcy.Finalize();

      // Arbitrary but FIXED states, the same in both arms. Nothing here is
      // a solution of anything and nothing needs to be: two spellings of one
      // operator have to agree wherever they are evaluated.
      BlockVector b(darcy.GetOffsets()), x(darcy.GetOffsets());
      b = 0.0;
      for (int i = 0; i < x.Size(); i++) { x(i) = 0.25 + 0.5 * std::sin(1.0*i); }
      Vector x_tr(Mh.GetTrueVSize());
      for (int i = 0; i < x_tr.Size(); i++) { x_tr(i) = 0.3 * std::cos(2.0*i); }

      Result res;
      BlockVector r(darcy.GetOffsets());
      darcy.GetHybridization()->NPCResidual(b, x, x_tr, r, res.r_tr);

      Operator &S = darcy.GetHybridization()->NPCGradient(x, x_tr);
      Vector y(S.Width());
      for (int i = 0; i < y.Size(); i++) { y(i) = 1.0 / (1.0 + i); }
      res.Sy.SetSize(S.Height());
      S.Mult(y, res.Sy);
      return res;
   };

   /* @a nf and @a nl are how the LIVE arm splits the total: @a nf frozen
      copies beside @a nl live ones. The frozen arm always takes the whole
      total, so the two arms are the same discrete problem by construction and
      the comparison stays an identity rather than an approximation. */
   auto compare = [&run](Mesh &fmesh, Mesh &lmesh, int nf = 0, int nl = 1)
   {
      Result f = run(DarcyForm::FaceConstraintMode::Frozen, fmesh, nf + nl, 0);
      Result l = run(DarcyForm::FaceConstraintMode::Live, lmesh, nf, nl);

      REQUIRE(l.r_tr.Size() == f.r_tr.Size());
      REQUIRE(l.Sy.Size() == f.Sy.Size());

      Vector dr(l.r_tr); dr -= f.r_tr;
      Vector ds(l.Sy);   ds -= f.Sy;
      const real_t rel_r = dr.Norml2() / f.r_tr.Norml2();
      const real_t rel_s = ds.Norml2() / f.Sy.Norml2();
      CAPTURE(rel_r, rel_s);
      // Relative, and generously: the two routes reach the same numbers by
      // different orders of operation, so this is a round-off comparison and
      // not an equality. Every measured value is below 3e-16; the three
      // broken variants are at 2e-04 and above.
      REQUIRE(rel_r < 1e-12);
      REQUIRE(rel_s < 1e-12);
   };

   /* The CONTROL, and it is what makes the rest mean anything: on a
      conforming mesh the element loop never meets a master face, so the two
      routes have always agreed and a broken expansion cannot show here. */
   SECTION("conforming")
   {
      Mesh fm = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL);
      Mesh lm = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL);
      compare(fm, lm);
   }

   /* The second control, which separates "nonconforming" from "has hanging
      nodes": the NC flag is on and there is nothing to expand. It is the arm
      that would still pass if GetNCMasterSlaves() were keyed on the mesh
      being nonconforming rather than on the face being a master. */
   SECTION("nonconforming flag, no hanging nodes")
   {
      Mesh fm = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL);
      Mesh lm = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL);
      fm.EnsureNCMesh();
      lm.EnsureNCMesh();
      REQUIRE(fm.Nonconforming());
      compare(fm, lm);
   }

   SECTION("hanging nodes, 2-D")
   {
      Mesh fm("../../data/amr-quad.mesh");
      Mesh lm("../../data/amr-quad.mesh");
      fm.UniformRefinement();
      lm.UniformRefinement();
      REQUIRE(fm.Nonconforming());
      compare(fm, lm);
   }

   /* Three dimensions, where a master carries four slaves rather than two and
      the transfer's orientation handling is a different branch of
      GetNCSlaveTransfer() -- 3-D takes its dof ordering from the face info
      where 2-D takes it from the edge orientations. Order 1 only: the 3-D
      arm at order 2 is minutes rather than seconds and says the same thing,
      and it is checked outside the suite. */
   SECTION("hanging nodes, 3-D")
   {
      if (order > 1) { return; }
      Mesh fm("../../data/amr-hex.mesh");
      Mesh lm("../../data/amr-hex.mesh");
      REQUIRE(fm.Nonconforming());
      compare(fm, lm);
   }

   /* A LIVE face constraint sitting BESIDE a frozen one, at a hanging node.
      This is the configuration gffp reported on, and on a nonconforming mesh
      it reaches a second piece of machinery: SeedLinearEG() puts the frozen
      half of E and G back before the live pass accumulates onto them, and on
      a master face it has to seed the MASTER's slot at the master's own dof
      count. It does so through exactly the offsets AssembleHDGGrad() uses,
      which is why it needed no change -- and "needed no change" is worth one
      section rather than an argument. */
   SECTION("half frozen, half live, at a hanging node")
   {
      Mesh fm("../../data/amr-quad.mesh");
      Mesh lm("../../data/amr-quad.mesh");
      fm.UniformRefinement();
      lm.UniformRefinement();
      REQUIRE(fm.Nonconforming());
      compare(fm, lm, 1, 1);
   }
}
TEST_CASE("The hybridized Jacobian carries d(flux residual)/dp",
          "[DarcyForm][NonlinearDarcy][HDG]")
{
   using namespace darcy_nonlinear;

   // DarcyHybridization::ConstructGrad and LocalNLOperator::GetGradient both
   // used to set the local Jacobian's (0,1) block to +/-B^T, the transpose of
   // the linear divergence form, and never ask the integrator for
   // d(flux residual)/dp. For a flux law q = D(p) u that term is the J_u the
   // flux function supplies, and leaving it out costs Newton convergence:
   // convdiff's own p8_o1_hb_nld_newton went from nine iterations to four
   // when it was restored.
   //
   // The check is the trace operator differenced against its own gradient.
   // An earlier attempt to infer the same defect from a Newton convergence
   // history was wrong, because the harness it used had no boundary condition
   // and therefore a null space -- the residual is identically zero on the
   // unconstrained boundary traces while the gradient is not -- and the
   // wandering that produced was read as a stall. Hence the boundary face
   // penalty below: it constrains those traces, and without it a quarter of
   // this comparison would be meaningless.
   //
   // With no state dependence the block is zero and the two agree trivially;
   // eps > 0 is the case that fails if the block is dropped.
   const real_t eps = GENERATE(0.0, 1.0);
   CAPTURE(eps);

   Mesh mesh = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL, false,
                                     1.0, 1.0);
   const int order = 1, dim = 2;
   ScaledCoupledFlux flux(dim, eps);
   const int neq = flux.num_equations;

   L2_FECollection u_coll(order, dim), p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace fes_u(&mesh, &u_coll, neq * dim, Ordering::byNODES);
   FiniteElementSpace fes_p(&mesh, &p_coll, neq, Ordering::byNODES);
   FiniteElementSpace fes_t(&mesh, &t_coll, neq, Ordering::byNODES);

   DarcyForm darcy(&fes_u, &fes_p);

   Vector taus(neq);
   taus = 1.0;

   BlockNonlinearForm *Mnl = darcy.GetBlockNonlinearForm();
   Mnl->AddDomainIntegrator(new MixedConductionNLFIntegrator(flux));

   auto *face = new MixedConductionNLFIntegrator(flux);
   face->SetVariableStabilization(taus);
   Mnl->AddInteriorFaceIntegrator(face);

   // The boundary faces carry the same penalty, which is what pins the
   // boundary traces and makes the operator nonsingular.
   auto *bface = new MixedConductionNLFIntegrator(flux);
   bface->SetVariableStabilization(taus);
   Mnl->AddBdrFaceIntegrator(bface);

   MixedBilinearForm *Bform = darcy.GetFluxDivForm();
   Bform->AddDomainIntegrator(
      new VectorBlockDiagonalIntegrator(neq, new VectorDivergenceIntegrator));
   Bform->AddInteriorFaceIntegrator(
      new VectorBlockDiagonalIntegrator(
         neq, new TransposeIntegrator(new DGNormalTraceIntegrator(-1.))));

   g_neq = neq;
   VectorFunctionCoefficient gcoeff(neq, gCoupled);
   darcy.GetPotentialRHS()->AddDomainIntegrator(
      new VectorDomainLFIntegrator(gcoeff));

   Array<int> ess;
   darcy.EnableHybridization(
      &fes_t,
      new VectorBlockDiagonalIntegrator(neq, new NormalTraceJumpIntegrator),
      ess);
   darcy.Assemble();
   darcy.GetHybridization()->SetLocalNLSolver(
      DarcyHybridization::LSsolveType::Newton, 100, 1e-14, 1e-16, -1);

   BlockVector x(darcy.GetOffsets());
   x = 0.0;
   OperatorPtr op;
   Vector X, RHS;
   darcy.FormLinearSystem(ess, x, op, X, RHS, true);

   Vector X0(X.Size()), dy(X.Size());
   for (int i = 0; i < X0.Size(); i++)
   {
      X0(i) = 0.03 * std::sin(1.7 * i + 0.4);
      dy(i) = 0.01 * std::cos(0.9 * i + 1.1);
   }

   // Residual first, gradient second, which is the order NewtonSolver uses.
   const real_t h = 1e-6;
   Vector xp(X0), xm(X0), rp(X.Size()), rm(X.Size());
   xp.Add(h, dy);
   xm.Add(-h, dy);
   op->Mult(xp, rp);
   op->Mult(xm, rm);
   Vector fd(rp);
   fd -= rm;
   fd /= (2.0 * h);

   Vector r0(X.Size());
   op->Mult(X0, r0);
   Vector Jdy(X.Size());
   op->GetGradient(X0).Mult(dy, Jdy);

   REQUIRE(fd.Normlinf() > 1e-6);          // the operator is not trivial

   // No dof is exempt: with the boundary constrained there is no null space
   // to excuse. The tolerance is set by the central difference, not by the
   // Jacobian, which is why it is 1e-8 and not machine precision.
   int nullish = 0;
   for (int i = 0; i < fd.Size(); i++) { if (fd(i) == 0.0) { nullish++; } }
   INFO("dofs with an identically zero residual: " << nullish);
   REQUIRE(nullish == 0);

   Vector d(Jdy);
   d -= fd;
   const real_t rel = d.Normlinf() / fd.Normlinf();
   INFO("relative ||J dy - fd|| = " << rel);
   REQUIRE(rel < 1e-7);
}
