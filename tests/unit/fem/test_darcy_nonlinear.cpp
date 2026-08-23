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
