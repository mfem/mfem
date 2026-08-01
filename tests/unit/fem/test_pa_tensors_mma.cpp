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
#include "fem/integ/bilininteg_pa_mma.hpp"

using namespace mfem;

namespace pa_tensors_mma
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

/** SUM-PA vs MMA-PA on a tensor H1 mesh (FA vs PA covered elsewhere). */
void test_pa_tensors_mma(Mesh &mesh, int p)
{
   const int dim = mesh.Dimension();
   CAPTURE(dim, p, mesh.GetNE());

   H1_FECollection fec(p, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);

   {
      MMAForce on(true);
      REQUIRE(UsesTensorMMA(fes));
   }

   GridFunction x(&fes), y_mma(&fes), y_sum(&fes);
   x.Randomize(0x100001b3);
   y_mma.Randomize(0x9e3779b9);
   y_sum = y_mma;

   const auto &fe = *fes.GetTypicalFE();
   // Match specialized (D1D,Q1D)=(p+1,p+2) pairs used by Tensors MMA kernels
   const IntegrationRule *ir = &IntRules.Get(fe.GetGeomType(), 2 * p + 2);

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

void test_pa_tensors_mma_cartesian(int dim, int p)
{
   Mesh mesh = (dim == 2)
               ? Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL)
               : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
   test_pa_tensors_mma(mesh, p);
}

} // namespace

TEST_CASE("Tensors MMA PA vs SUM-PA", "[MMA][GPU]")
{
   const int dim = GENERATE(2, 3);
   // p=2 uses SUM (m8n8k4 pad); MMA path starts at p>=3.
   const int p = GENERATE(3, 4, 5, 6, 7);
   test_pa_tensors_mma_cartesian(dim, p);
}

TEST_CASE("Tensors MMA PA vs SUM-PA uneven NE", "[MMA][GPU]")
{
   // Partial serial NB batches: mass NB=8, diffusion NB=4.
   SECTION("quad 5x3")
   {
      const int p = GENERATE(3, 4, 7);
      Mesh mesh = Mesh::MakeCartesian2D(5, 3, Element::QUADRILATERAL);
      test_pa_tensors_mma(mesh, p);
   }
   SECTION("hex 3x3x2")
   {
      const int p = GENERATE(3, 4, 7);
      Mesh mesh = Mesh::MakeCartesian3D(3, 3, 2, Element::HEXAHEDRON);
      test_pa_tensors_mma(mesh, p);
   }
}

TEST_CASE("Tensors MMA PA vs SUM-PA on meshes", "[MMA][GPU]")
{
   // Paths assume cwd = build/tests/unit (MFEM unit-test convention).
   const int p = GENERATE(3, 4, 7);

   SECTION("2d")
   {
      const char *filename =
         GENERATE("../../data/inline-quad.mesh",
                  "../../data/periodic-square.mesh",
                  "../../data/fichera-quad.mesh");
      CAPTURE(filename);
      Mesh mesh(filename);
      REQUIRE(mesh.Dimension() == 2);
      test_pa_tensors_mma(mesh, p);
   }

   SECTION("3d")
   {
      const char *filename =
         GENERATE("../../data/fichera.mesh",
                  "../../data/fichera-q3.mesh",
                  "../../data/inline-hex.mesh",
                  "../../data/toroid-hex.mesh",
                  "../../data/periodic-cube.mesh");
      CAPTURE(filename);
      Mesh mesh(filename);
      REQUIRE(mesh.Dimension() == 3);
      test_pa_tensors_mma(mesh, p);
   }
}

TEST_CASE("Tensors MMA eligibility", "[MMA][GPU]")
{
   Mesh mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
   H1_FECollection fec(3, 3, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);

   {
      MMAForce off(false);
      REQUIRE_FALSE(UsesTensorMMA(fes));
   }

   {
      MMAForce on(true);
      REQUIRE(UsesTensorMMA(fes));
   }

   REQUIRE_FALSE(UsesTensorMMA(fes));

   // p=1 and p=2 are intentionally unsupported (pad / SUM preferred)
   H1_FECollection fec1(1, 3, BasisType::GaussLobatto);
   FiniteElementSpace fes1(&mesh, &fec1);
   {
      MMAForce on(true);
      REQUIRE_FALSE(UsesTensorMMA(fes1));
      H1_FECollection fec2(2, 3, BasisType::GaussLobatto);
      FiniteElementSpace fes2(&mesh, &fec2);
      REQUIRE_FALSE(UsesTensorMMA(fes2));
   }
}

} // namespace pa_tensors_mma
