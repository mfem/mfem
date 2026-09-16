// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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
#include "fem/integ/mma/mma.hpp"
#include "fem/integ/mma/form/fields.hpp"
#include "fem/integ/mma/hcurl.hpp"
#include "fem/integ/mma/hdiv.hpp"

using namespace mfem;
using mfem::internal::mma::form::curl_t;
using mfem::internal::mma::form::grad_t;
using mfem::internal::mma::form::CurlCurlQFn;
using mfem::internal::mma::form::DivDivQFn;
using mfem::internal::mma::form::HcurlMass;
using mfem::future::tensor;
// Avoid clash with C stdlib ::div_t
using mma_div_t = mfem::internal::mma::form::div_t;

namespace pa_hcurl_hdiv_mma
{

enum class Op { Mass, CurlCurl, DivDiv };

void test_fa_vs_mma(Mesh &mesh, int p, bool hcurl, Op op, bool tensor)
{
   const int dim = mesh.Dimension();
   CAPTURE(dim, p, hcurl, int(op), tensor, mesh.GetNE());

   std::unique_ptr<FiniteElementCollection> fec;
   if (hcurl) { fec.reset(new ND_FECollection(p, dim)); }
   else { fec.reset(new RT_FECollection(p, dim)); }
   FiniteElementSpace fes(&mesh, fec.get());

   {
      MMAForce on(true);
      if (tensor)
      {
         if (hcurl) { REQUIRE(UsesTensorMmaHcurl(fes)); }
         else { REQUIRE(UsesTensorMmaHdiv(fes)); }
      }
      else
      {
         if (hcurl) { REQUIRE(UsesSimplexMmaHcurl(fes)); }
         else { REQUIRE(UsesSimplexMmaHdiv(fes)); }
      }
   }
   {
      MMAForce off(false);
      if (tensor)
      {
         REQUIRE_FALSE((UsesTensorMmaHcurl(fes) || UsesTensorMmaHdiv(fes)));
      }
      else
      {
         REQUIRE_FALSE((UsesSimplexMmaHcurl(fes) || UsesSimplexMmaHdiv(fes)));
      }
   }

   GridFunction x(&fes), y_fa(&fes), y_mma(&fes);
   x.Randomize(0x100001b3);
   y_fa = 0.0;
   y_mma = 0.0;

   ConstantCoefficient one(1.0);
   ConstantCoefficient c2(M_2_SQRTPI);
   FunctionCoefficient cf([](const Vector &pt)
   { return M_1_PI + pt[0] * pt[0]; });

   // Shared IR so FA and PA use the same quadrature (Mass rule / PA default).
   const auto &fe = *fes.GetTypicalFE();
   ElementTransformation &T = *mesh.GetTypicalElementTransformation();
   const IntegrationRule *ir = &MassIntegrator::GetRule(fe, fe, T);

   auto add_ops = [&](BilinearForm &a)
   {
      if (op == Op::Mass)
      {
         auto *i0 = new VectorFEMassIntegrator;
         auto *i1 = new VectorFEMassIntegrator(c2);
         auto *i2 = new VectorFEMassIntegrator(cf);
         i0->SetIntRule(ir); i1->SetIntRule(ir); i2->SetIntRule(ir);
         a.AddDomainIntegrator(i0);
         a.AddDomainIntegrator(i1);
         a.AddDomainIntegrator(i2);
      }
      else if (op == Op::CurlCurl)
      {
         auto *i0 = new CurlCurlIntegrator;
         auto *i1 = new CurlCurlIntegrator(c2);
         auto *i2 = new CurlCurlIntegrator(cf);
         i0->SetIntRule(ir); i1->SetIntRule(ir); i2->SetIntRule(ir);
         a.AddDomainIntegrator(i0);
         a.AddDomainIntegrator(i1);
         a.AddDomainIntegrator(i2);
      }
      else
      {
         auto *i0 = new DivDivIntegrator(one);
         auto *i1 = new DivDivIntegrator(c2);
         auto *i2 = new DivDivIntegrator(cf);
         i0->SetIntRule(ir); i1->SetIntRule(ir); i2->SetIntRule(ir);
         a.AddDomainIntegrator(i0);
         a.AddDomainIntegrator(i1);
         a.AddDomainIntegrator(i2);
      }
   };

   BilinearForm fa(&fes), pa(&fes);
   add_ops(fa);
   add_ops(pa);
   pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);

   fa.Assemble();
   fa.Finalize();
   {
      MMAForce on(true);
      pa.Assemble();
   }

   fa.Mult(x, y_fa);
   pa.Mult(x, y_mma);
   y_fa -= y_mma;
   REQUIRE(y_fa.Normlinf() == MFEM_Approx(0.0, 1e-8, 1e-8));
}

TEST_CASE("Hcurl/Hdiv MMA QFn traits", "[MMA][Form][Hcurl][Hdiv]")
{
   SECTION("curl_t planes")
   {
      REQUIRE(curl_t<2>::planes(2) == 1);
      REQUIRE(curl_t<3>::planes(3) == 3);
      REQUIRE(mma_div_t::planes(3) == 1);
   }
   SECTION("CurlCurlQFn 2D")
   {
      CurlCurlQFn<2> q;
      curl_t<2> u, y;
      u[0] = 2.0;
      q(u, y, real_t(3.0));
      REQUIRE(y[0] == MFEM_Approx(6.0));
   }
   SECTION("DivDivQFn")
   {
      DivDivQFn q;
      mma_div_t u(2.0), y;
      q(u, y, real_t(4.0));
      REQUIRE(real_t(y) == MFEM_Approx(8.0));
   }
   SECTION("HcurlMass 2D")
   {
      HcurlMass<2> q;
      grad_t<2> u, y;
      u[0] = 1.0; u[1] = 2.0;
      tensor<real_t, 2, 2> A{};
      A(0, 0) = 2.0; A(1, 1) = 3.0;
      q(u, y, A);
      REQUIRE(y[0] == MFEM_Approx(2.0));
      REQUIRE(y[1] == MFEM_Approx(6.0));
   }
}

TEST_CASE("Hcurl/Hdiv tensor MMA PA vs FA", "[PA][MMA][Hcurl][Hdiv][CPU]")
{
   const int p = GENERATE(3, 4);
   SECTION("2D ND mass + curlcurl")
   {
      Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);
      test_fa_vs_mma(mesh, p, true, Op::Mass, true);
      test_fa_vs_mma(mesh, p, true, Op::CurlCurl, true);
   }
   SECTION("2D RT mass + divdiv")
   {
      Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);
      test_fa_vs_mma(mesh, p, false, Op::Mass, true);
      test_fa_vs_mma(mesh, p, false, Op::DivDiv, true);
   }
   SECTION("3D ND mass + curlcurl")
   {
      Mesh mesh = Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON);
      test_fa_vs_mma(mesh, p, true, Op::Mass, true);
      test_fa_vs_mma(mesh, p, true, Op::CurlCurl, true);
   }
   SECTION("3D RT mass + divdiv")
   {
      Mesh mesh = Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON);
      test_fa_vs_mma(mesh, p, false, Op::Mass, true);
      test_fa_vs_mma(mesh, p, false, Op::DivDiv, true);
   }
}

TEST_CASE("Hcurl/Hdiv simplex MMA PA vs FA", "[PA][MMA][Hcurl][Hdiv][Simplex][CPU]")
{
   const int p = GENERATE(1, 2, 3);
   SECTION("2D ND triangle")
   {
      Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::TRIANGLE);
      test_fa_vs_mma(mesh, p, true, Op::Mass, false);
      test_fa_vs_mma(mesh, p, true, Op::CurlCurl, false);
   }
   SECTION("2D RT triangle")
   {
      Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::TRIANGLE);
      test_fa_vs_mma(mesh, p, false, Op::Mass, false);
      test_fa_vs_mma(mesh, p, false, Op::DivDiv, false);
   }
   SECTION("3D ND tet")
   {
      Mesh mesh = Mesh::MakeCartesian3D(1, 1, 1, Element::TETRAHEDRON);
      test_fa_vs_mma(mesh, p, true, Op::Mass, false);
      test_fa_vs_mma(mesh, p, true, Op::CurlCurl, false);
   }
   SECTION("3D RT tet")
   {
      Mesh mesh = Mesh::MakeCartesian3D(1, 1, 1, Element::TETRAHEDRON);
      test_fa_vs_mma(mesh, p, false, Op::Mass, false);
      test_fa_vs_mma(mesh, p, false, Op::DivDiv, false);
   }
}

TEST_CASE("Hcurl/Hdiv ForceMMA off keeps stock tensor PA",
          "[PA][MMA][Hcurl][Hdiv][CPU]")
{
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);
   ND_FECollection fec(3, 2);
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction x(&fes), y_stock(&fes), y_off(&fes);
   x.Randomize(7);
   y_stock = 0.0;
   y_off = 0.0;
   ConstantCoefficient c(1.25);

   BilinearForm stock(&fes), off(&fes);
   stock.AddDomainIntegrator(new VectorFEMassIntegrator(c));
   stock.AddDomainIntegrator(new CurlCurlIntegrator(c));
   off.AddDomainIntegrator(new VectorFEMassIntegrator(c));
   off.AddDomainIntegrator(new CurlCurlIntegrator(c));
   stock.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   off.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   {
      MMAForce force(false);
      stock.Assemble();
      off.Assemble();
   }
   stock.Mult(x, y_stock);
   off.Mult(x, y_off);
   y_stock -= y_off;
   REQUIRE(y_stock.Normlinf() == MFEM_Approx(0.0, 1e-12, 1e-12));
   REQUIRE_FALSE(UsesTensorMmaHcurl(fes));
}

} // namespace pa_hcurl_hdiv_mma
