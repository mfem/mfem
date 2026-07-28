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

#ifdef _WIN32
#define _USE_MATH_DEFINES
#include <cmath>
#endif

#include "unit_tests.hpp"
#include "mfem.hpp"
#include "fem/integ/bilininteg_pa_mma.hpp"

using namespace mfem;

namespace pa_simplices_mma
{

namespace
{

void AddMassDiffIntegrators(BilinearForm &a, const IntegrationRule *ir,
                            Coefficient &const_coeff,
                            Coefficient &funct_coeff)
{
   a.AddDomainIntegrator(new MassIntegrator(ir));
   a.AddDomainIntegrator(new MassIntegrator(const_coeff, ir));
   a.AddDomainIntegrator(new MassIntegrator(funct_coeff, ir));
   a.AddDomainIntegrator(new DiffusionIntegrator(ir));
   a.AddDomainIntegrator(new DiffusionIntegrator(const_coeff, ir));
   a.AddDomainIntegrator(new DiffusionIntegrator(funct_coeff, ir));
}

/** MMA-PA vs stock Simplex PA (Positive / MMAForce). FA vs PA is covered in
    test_pa_simplices.cpp. */
void test_pa_simplices_mma_positive(const char *filename, int p)
{
   CAPTURE(filename, p);

   Mesh mesh(filename);
   MFEM_VERIFY((mesh.Dimension() == 2 || mesh.Dimension() == 3),
               "Mesh dimension must be 2 or 3");
   MFEM_VERIFY(!mesh.IsMixedMesh(), "Mesh is mixed");
   MFEM_VERIFY(mesh.SpaceDimension() == mesh.Dimension(),
               "Simplex MMA requires volumetric meshes (sdim == dim)");

   H1_FECollection fec(p, mesh.Dimension(), BasisType::Positive);
   FiniteElementSpace fes(&mesh, &fec);

   {
      MMAForce on(true);
      if (!UsesSimplexMMA(fes)) { return; }
   }

   GridFunction x(&fes), y_mma(&fes), y_sum(&fes);
   x.Randomize(0x100001b3);
   y_mma.Randomize(0x9e3779b9);
   y_sum = y_mma;

   const auto &fe = *fes.GetTypicalFE();
   const auto &Tr = *mesh.GetTypicalElementTransformation();
   // Stock Positive PA uses ragged-tensor maps with Stroud rules; MMA accepts
   // the same IR via CalcShape, so both paths share Stroud quadrature.
   const auto order = 2 * fe.GetOrder() + Tr.OrderW();
   const IntegrationRule *ir = &StroudIntRules.Get(fe.GetGeomType(), order);

   // Runtime (non-specialized) simplex MMA apply caps.
   const int max_q1d = DeviceDofQuadLimits::Get().MAX_Q1D;
   const int max_nq = (mesh.Dimension() == 2) ? max_q1d * max_q1d : 256;
   if (ir->GetNPoints() > max_nq) { return; }

   ConstantCoefficient const_coeff(M_2_SQRTPI);
   FunctionCoefficient funct_coeff([](const Vector &pt)
   { return M_1_PI + pt[0] * pt[0]; });

   BilinearForm pa_mma(&fes), pa_sum(&fes);
   AddMassDiffIntegrators(pa_mma, ir, const_coeff, funct_coeff);
   AddMassDiffIntegrators(pa_sum, ir, const_coeff, funct_coeff);
   pa_mma.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   pa_sum.SetAssemblyLevel(AssemblyLevel::PARTIAL);

   {
      MMAForce on(true);
      pa_mma.Assemble();
   }
   {
      MMAForce off(false);
      pa_sum.Assemble();
   }

   pa_mma.Mult(x, y_mma);
   pa_sum.Mult(x, y_sum);

   y_sum -= y_mma;
   REQUIRE(y_sum.Normlinf() == MFEM_Approx(0.0, 1e-9, 1e-9));
}

/** GLL simplex MMA is default-on; smoke Assemble/Mult without an FA oracle. */
void test_pa_simplices_mma_gll_smoke(const char *filename, int p)
{
   CAPTURE(filename, p);

   Mesh mesh(filename);
   MFEM_VERIFY((mesh.Dimension() == 2 || mesh.Dimension() == 3),
               "Mesh dimension must be 2 or 3");
   MFEM_VERIFY(!mesh.IsMixedMesh(), "Mesh is mixed");
   MFEM_VERIFY(mesh.SpaceDimension() == mesh.Dimension(),
               "Simplex MMA requires volumetric meshes (sdim == dim)");

   H1_FECollection fec(p, mesh.Dimension(), BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);

   if (!UsesSimplexMMA(fes)) { return; }

   const auto &fe = *fes.GetTypicalFE();
   const auto &Tr = *mesh.GetTypicalElementTransformation();
   const auto order = 2 * fe.GetOrder() + Tr.OrderW() + 4;
   const IntegrationRule *ir = &IntRules.Get(fe.GetGeomType(), order);

   const int max_q1d = DeviceDofQuadLimits::Get().MAX_Q1D;
   const int max_nq = (mesh.Dimension() == 2) ? max_q1d * max_q1d : 256;
   if (ir->GetNPoints() > max_nq) { return; }

   ConstantCoefficient const_coeff(M_2_SQRTPI);
   FunctionCoefficient funct_coeff([](const Vector &pt)
   { return M_1_PI + pt[0] * pt[0]; });

   GridFunction x(&fes), y(&fes);
   x.Randomize(0x100001b3);
   y.Randomize(0x9e3779b9);

   BilinearForm pa(&fes);
   AddMassDiffIntegrators(pa, ir, const_coeff, funct_coeff);
   pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   pa.Assemble();
   pa.Mult(x, y);
   REQUIRE(y.Norml2() >= 0.0);
}

} // namespace

TEST_CASE("PA Simplices MMA vs stock PA", "[PartialAssembly][SimplexMMA][GPU]")
{
   const auto all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? GENERATE(1, 2, 5, 6) : GENERATE(1, 2, 3, 4, 5, 6);

   const auto GenMesh = [&](const auto &meshs, const auto &extra)
   {
      return !all_tests
             ? GENERATE_REF(from_range(meshs))
             : GENERATE_REF(from_range(meshs), from_range(extra));
   };

   SECTION("2D")
   {
      auto meshs = { "../../data/beam-tri.mesh",
                     "../../data/inline-tri.mesh",
                     "../../data/ref-triangle.mesh",
                     "../../data/rt-2d-p4-tri.mesh",
                     "../../data/square-disc-p2.mesh",
                     "../../data/square-disc-p3.mesh",
                     "../../data/periodic-annulus-sector.msh"
                   };
      test_pa_simplices_mma_positive(GENERATE_REF(from_range(meshs)), p);
   }

   SECTION("3D")
   {
      auto meshs = { "../../data/beam-tet.mesh",
                     "../../data/inline-tet.mesh",
                     "../../data/ref-tetrahedron.mesh"
                   };
      auto extra = { "../../data/escher.mesh",
                     "../../data/escher-p2.mesh"
                   };
      test_pa_simplices_mma_positive(GenMesh(meshs, extra), p);
   }
}

TEST_CASE("PA Simplices MMA GLL smoke", "[PartialAssembly][SimplexMMA][GPU]")
{
   const auto all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? GENERATE(1, 2, 5, 6) : GENERATE(1, 2, 3, 4, 5, 6);

   SECTION("2D")
   {
      auto meshs = { "../../data/ref-triangle.mesh",
                     "../../data/inline-tri.mesh",
                     "../../data/beam-tri.mesh"
                   };
      test_pa_simplices_mma_gll_smoke(GENERATE_REF(from_range(meshs)), p);
   }

   SECTION("3D")
   {
      auto meshs = { "../../data/ref-tetrahedron.mesh",
                     "../../data/inline-tet.mesh",
                     "../../data/beam-tet.mesh"
                   };
      test_pa_simplices_mma_gll_smoke(GENERATE_REF(from_range(meshs)), p);
   }
}

TEST_CASE("PA Simplices Positive force MMA",
          "[PartialAssembly][SimplexMMA][GPU]")
{
   Mesh mesh("../../data/ref-triangle.mesh");
   H1_FECollection fec(3, mesh.Dimension(), BasisType::Positive);
   FiniteElementSpace fes(&mesh, &fec);

   REQUIRE_FALSE(UsesSimplexMMA(fes));
   REQUIRE(GetEVectorOrdering(fes) == ElementDofOrdering::LEXICOGRAPHIC);

   const auto &fe = *fes.GetTypicalFE();
   const auto &Tr = *mesh.GetTypicalElementTransformation();
   const IntegrationRule *ir =
      &StroudIntRules.Get(fe.GetGeomType(), 2 * fe.GetOrder() + Tr.OrderW());

   GridFunction x(&fes), y_mma(&fes), y_sum(&fes);
   x.Randomize(0x100001b3);
   y_mma.Randomize(0x9e3779b9);
   y_sum = y_mma;

   BilinearForm pa_mma(&fes), pa_sum(&fes);
   pa_mma.AddDomainIntegrator(new MassIntegrator(ir));
   pa_mma.AddDomainIntegrator(new DiffusionIntegrator(ir));
   pa_sum.AddDomainIntegrator(new MassIntegrator(ir));
   pa_sum.AddDomainIntegrator(new DiffusionIntegrator(ir));
   pa_mma.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   pa_sum.SetAssemblyLevel(AssemblyLevel::PARTIAL);

   {
      MMAForce on(true);
      REQUIRE(UsesSimplexMMA(fes));
      REQUIRE(GetEVectorOrdering(fes) == ElementDofOrdering::NATIVE);
      pa_mma.Assemble();
   }
   {
      MMAForce off(false);
      pa_sum.Assemble();
   }

   pa_mma.Mult(x, y_mma);
   pa_sum.Mult(x, y_sum);
   y_sum -= y_mma;
   REQUIRE(y_sum.Normlinf() == MFEM_Approx(0.0, 1e-9, 1e-9));

   REQUIRE_FALSE(UsesSimplexMMA(fes));
   REQUIRE(GetEVectorOrdering(fes) == ElementDofOrdering::LEXICOGRAPHIC);
}

} // namespace pa_simplices_mma
