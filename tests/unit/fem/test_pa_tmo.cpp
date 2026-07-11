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

#include "unit_tests.hpp"
#include "mfem.hpp"

#include <cstdlib>

using namespace mfem;

namespace pa_tmo
{

void SetEnv(const char *name, const char *val)
{
#ifdef _WIN32
   _putenv_s(name, val);
#else
   setenv(name, val, 1);
#endif
}

void ClearTmoEnv()
{
   SetEnv("MFEM_USE_TMO_DUFFY", "0");
   SetEnv("MFEM_USE_TMO_TENSOR", "0");
   SetEnv("MFEM_USE_TMO", "0");
}

void MapSquareToTriangle(int k, real_t s, real_t t, real_t &xi, real_t &eta)
{
   if (k == 0) { xi = s; eta = t; }
   else if (k == 1) { xi = real_t(1) - s - t; eta = s; }
   else { xi = t; eta = real_t(1) - s - t; }
}

void MapSquareToParallelogram(int k, real_t s, real_t t, real_t &x, real_t &y)
{
   if (k == 0) { x = s; y = t; }
   else if (k == 1) { x = real_t(1) - s - t; y = t; }
   else { x = t; y = real_t(1) - s - t; }
}

void EvenEvalPoint(int k, real_t s, real_t t, real_t &x, real_t &y)
{
   real_t sf = s, tf = t;
   if (s + t > real_t(1))
   {
      sf = real_t(1) - t;
      tf = real_t(1) - s;
   }
   MapSquareToParallelogram(k, sf, tf, x, y);
}

TEST_CASE("TMO Geometry Helpers", "[PartialAssembly][TMO]")
{
   real_t xi, eta;
   MapSquareToTriangle(0, 0.2, 0.3, xi, eta);
   REQUIRE(xi == MFEM_Approx(0.2));
   REQUIRE(eta == MFEM_Approx(0.3));

   MapSquareToTriangle(1, 0.2, 0.3, xi, eta);
   REQUIRE(xi == MFEM_Approx(0.5));
   REQUIRE(eta == MFEM_Approx(0.2));

   MapSquareToTriangle(2, 0.2, 0.3, xi, eta);
   REQUIRE(xi == MFEM_Approx(0.3));
   REQUIRE(eta == MFEM_Approx(0.5));

   real_t x, y;
   EvenEvalPoint(0, 0.7, 0.6, x, y);
   REQUIRE(x + y <= MFEM_Approx(1.0));
   EvenEvalPoint(1, 1.0, 1.0, x, y);
   REQUIRE(x + y <= MFEM_Approx(1.0));
   EvenEvalPoint(2, 1.0, 1.0, x, y);
   REQUIRE(x + y <= MFEM_Approx(1.0));
}

int NatIdx(int i, int j, int p)
{
   return j * (2 * p - j + 3) / 2 + i;
}

int ProlongSrc(int k, int i, int j, int p)
{
   if (k == 0) { return NatIdx(i, j, p); }
   if (k == 1) { return NatIdx(p - i - j, i, p); }
   return NatIdx(j, p - i - j, p);
}

TEST_CASE("TMO Bernstein multi-index prolong", "[PartialAssembly][TMO][Duffy]")
{
   const int p = GENERATE(1, 2, 3);
   const int ndof = (p + 1) * (p + 2) / 2;
   Vector X(ndof), Xk(ndof), shape0(ndof), shapek(ndof);
   X.Randomize(0xC001D00D);

   const real_t s = 0.2, t = 0.3;
   for (int k = 0; k < 3; k++)
   {
      CAPTURE(p, k);
      for (int i = 0; i <= p; i++)
      {
         for (int j = 0; j <= p - i; j++)
         {
            Xk(NatIdx(i, j, p)) = X(ProlongSrc(k, i, j, p));
         }
      }
      real_t xi, eta;
      MapSquareToTriangle(k, s, t, xi, eta);
      H1Pos_TriangleElement::CalcShape(p, s, t, shape0.GetData());
      H1Pos_TriangleElement::CalcShape(p, xi, eta, shapek.GetData());
      real_t u0 = 0.0, uk = 0.0;
      for (int n = 0; n < ndof; n++)
      {
         u0 += X(n) * shapek(n);
         uk += Xk(n) * shape0(n);
      }
      REQUIRE(u0 == MFEM_Approx(uk));
   }
}

void test_pa_tmo_mass(Mesh mesh, int p, int btype, const char *env_name)
{
   CAPTURE(p, btype, env_name);
   ClearTmoEnv();
   SetEnv(env_name, "1");

   if (mesh.GetTypicalElementGeometry() == Geometry::SQUARE ||
       mesh.GetTypicalElementGeometry() == Geometry::CUBE)
   {
      mesh = Mesh::MakeSimplicial(mesh);
   }
   REQUIRE(mesh.Dimension() == 2);
   MFEM_VERIFY(!mesh.IsMixedMesh(), "Mesh is mixed");

   H1_FECollection fec(p, mesh.Dimension(), btype);
   FiniteElementSpace fes(&mesh, &fec);

   GridFunction x(&fes), y_fa(&fes), y_pa(&fes);
   x.Randomize(0x100001b3);
   y_fa.Randomize(0x9e3779b9);
   y_pa = y_fa;

   const auto &fe = *fes.GetTypicalFE();
   const auto &Tr = *mesh.GetTypicalElementTransformation();
   const auto order = 2 * fe.GetOrder() + Tr.OrderW() + 4;
   const IntegrationRule *ir = (btype == BasisType::Positive)
                               ? &StroudIntRules.Get(fe.GetGeomType(), order)
                               : &IntRules.Get(fe.GetGeomType(), order);

   ConstantCoefficient const_coeff(M_2_SQRTPI);
   FunctionCoefficient funct_coeff([](const Vector &pt)
   { return M_1_PI + pt[0] * pt[0]; });

   BilinearForm fa(&fes), pa(&fes);
   fa.AddDomainIntegrator(new MassIntegrator(ir));
   fa.AddDomainIntegrator(new MassIntegrator(const_coeff, ir));
   fa.AddDomainIntegrator(new MassIntegrator(funct_coeff, ir));
   fa.Assemble();
   fa.Finalize();

   pa.AddDomainIntegrator(new MassIntegrator(ir));
   pa.AddDomainIntegrator(new MassIntegrator(const_coeff, ir));
   pa.AddDomainIntegrator(new MassIntegrator(funct_coeff, ir));
   pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   pa.Assemble();

   fa.Mult(x, y_fa);
   pa.Mult(x, y_pa);
   y_fa -= y_pa;
   REQUIRE(y_fa.Norml2() == MFEM_Approx(0.0));

   ClearTmoEnv();
}

TEST_CASE("PA TMO Duffy Mass", "[PartialAssembly][TMO][Duffy][GPU]")
{
   const auto all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? GENERATE(1, 2) : GENERATE(1, 2, 3, 4);

   SECTION("single triangle")
   {
      test_pa_tmo_mass(Mesh::MakeCartesian2D(1, 1, Element::TRIANGLE), p,
                       BasisType::Positive, "MFEM_USE_TMO_DUFFY");
   }

   SECTION("beam-tri")
   {
      test_pa_tmo_mass(Mesh("../../data/beam-tri.mesh"), p,
                       BasisType::Positive, "MFEM_USE_TMO_DUFFY");
   }
}

TEST_CASE("PA TMO Tensor Mass", "[PartialAssembly][TMO][Tensor][GPU]")
{
   const auto all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? GENERATE(1, 2) : GENERATE(1, 2, 3, 4);

   SECTION("single triangle")
   {
      test_pa_tmo_mass(Mesh::MakeCartesian2D(1, 1, Element::TRIANGLE), p,
                       BasisType::GaussLobatto, "MFEM_USE_TMO_TENSOR");
   }

   SECTION("beam-tri")
   {
      test_pa_tmo_mass(Mesh("../../data/beam-tri.mesh"), p,
                       BasisType::GaussLobatto, "MFEM_USE_TMO_TENSOR");
   }
}

} // namespace pa_tmo
