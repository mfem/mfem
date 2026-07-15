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

using namespace mfem;

namespace pa_simplex_mma
{

void test_pa_simplex_mass(Mesh mesh, int p, int btype)
{
   CAPTURE(p, btype);
   if (mesh.GetTypicalElementGeometry() == Geometry::SQUARE ||
       mesh.GetTypicalElementGeometry() == Geometry::CUBE)
   {
      mesh = Mesh::MakeSimplicial(mesh);
   }
   REQUIRE((mesh.Dimension() == 2 || mesh.Dimension() == 3));
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
   const bool stroud = (btype == BasisType::Positive);
   const IntegrationRule *ir = stroud
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
}

TEST_CASE("PA Simplex MMA Mass", "[PartialAssembly][SimplexMMA][GPU]")
{
   // GLL on CUDA uses the single-chart simplex MMA path automatically.
   const auto all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? GENERATE(1, 2, 5, 6) : GENERATE(1, 2, 3, 4, 5, 6);

   SECTION("single triangle")
   {
      test_pa_simplex_mass(Mesh::MakeCartesian2D(1, 1, Element::TRIANGLE), p,
                           BasisType::GaussLobatto);
   }

   SECTION("beam-tri")
   {
      test_pa_simplex_mass(Mesh("../../data/beam-tri.mesh"), p,
                           BasisType::GaussLobatto);
   }

   SECTION("single tet")
   {
      test_pa_simplex_mass(Mesh::MakeCartesian3D(1, 1, 1, Element::TETRAHEDRON), p,
                           BasisType::GaussLobatto);
   }
}

TEST_CASE("PA Simplex MMA Mass HO mesh", "[PartialAssembly][SimplexMMA][GPU]")
{
   // Quadratic geometry: assemble must use per-quad detJ.
   const auto p = GENERATE(1, 2, 3);

   SECTION("quadratic triangle mesh")
   {
      Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::TRIANGLE);
      mesh.SetCurvature(2);
      test_pa_simplex_mass(std::move(mesh), p, BasisType::GaussLobatto);
   }

   SECTION("quadratic tet mesh")
   {
      Mesh mesh = Mesh::MakeCartesian3D(1, 1, 1, Element::TETRAHEDRON);
      mesh.SetCurvature(2);
      test_pa_simplex_mass(std::move(mesh), p, BasisType::GaussLobatto);
   }
}

TEST_CASE("PA Simplex Positive Mass", "[PartialAssembly][SimplexPA][GPU]")
{
   // Positive / Bernstein uses existing Stroud ragged-tensor simplex PA.
   const auto all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? GENERATE(1, 2) : GENERATE(1, 2, 3, 4);

   SECTION("single triangle")
   {
      test_pa_simplex_mass(Mesh::MakeCartesian2D(1, 1, Element::TRIANGLE), p,
                           BasisType::Positive);
   }

   SECTION("beam-tri")
   {
      test_pa_simplex_mass(Mesh("../../data/beam-tri.mesh"), p,
                           BasisType::Positive);
   }
}

} // namespace pa_simplex_mma
