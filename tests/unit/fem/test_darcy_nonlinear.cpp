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
   const int order = GENERATE(0, 1);
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
   const int order = GENERATE(0, 1);
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
