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
