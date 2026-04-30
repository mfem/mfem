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

using namespace mfem;

namespace pa_kernels
{

template<int DIM>
void test_pa_simplices(const char *filename, int p)
{
   CAPTURE(filename, DIM, p);

   Mesh mesh(filename);
   MFEM_VERIFY(mesh.Dimension() == DIM, "Mesh dimension mismatch");

   H1_FECollection fec(p, DIM, BasisType::Positive);
   FiniteElementSpace fes(&mesh, &fec);

   GridFunction x(&fes), y_fa(&fes), y_pa(&fes);
   x.Randomize(0x100001b3);
   y_fa.Randomize(0x9e3779b9);
   y_pa = y_fa;

   const auto &fe = *fes.GetTypicalFE();
   const auto &Tr = *mesh.GetTypicalElementTransformation();
   const int order = 2 * fe.GetOrder() + Tr.OrderW();
   const IntegrationRule *ir = &StroudIntRules.Get(fe.GetGeomType(), order, false);

   ConstantCoefficient const_coeff(M_2_SQRTPI);
   FunctionCoefficient funct_coeff([](const Vector &x)
   { return M_1_PI + x[0] * x[0]; });

   BilinearForm fa(&fes), pa(&fes);
   fa.AddDomainIntegrator(new MassIntegrator(ir));
   fa.AddDomainIntegrator(new MassIntegrator(const_coeff, ir));
   fa.AddDomainIntegrator(new MassIntegrator(funct_coeff, ir));
   fa.AddDomainIntegrator(new DiffusionIntegrator(ir));
   fa.AddDomainIntegrator(new DiffusionIntegrator(const_coeff, ir));
   fa.AddDomainIntegrator(new DiffusionIntegrator(funct_coeff, ir));
   fa.Assemble();
   fa.Finalize();

   pa.AddDomainIntegrator(new MassIntegrator(ir));
   pa.AddDomainIntegrator(new MassIntegrator(const_coeff, ir));
   pa.AddDomainIntegrator(new MassIntegrator(funct_coeff, ir));
   pa.AddDomainIntegrator(new DiffusionIntegrator(ir));
   pa.AddDomainIntegrator(new DiffusionIntegrator(const_coeff, ir));
   pa.AddDomainIntegrator(new DiffusionIntegrator(funct_coeff, ir));
   pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   pa.Assemble();

   fa.Mult(x, y_fa);
   pa.Mult(x, y_pa);
   y_fa -= y_pa;
   REQUIRE(y_fa.Norml2() == MFEM_Approx(0.0));
}

TEST_CASE("PA Simplices", "[PartialAssembly][GPU][Simplices]")
{
   const bool all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? GENERATE(1, 2) : GENERATE(1, 2, 3, 4);

   SECTION("2D")
   {
      const auto filename2d = GENERATE("../../data/beam-tri.mesh",
                                       "../../data/inline-tri.mesh",
                                       "../../data/rt-2d-p4-tri.mesh",
                                       "../../data/ref-triangle.mesh");
      test_pa_simplices<2>(filename2d, p);
   }

   SECTION("3D")
   {
      const auto filename3d = GENERATE("../../data/beam-tet.mesh",
                                       "../../data/inline-tet.mesh",
                                       "../../data/ref-tetrahedron.mesh");
      test_pa_simplices<3>(filename3d, p);
   }
}

} // namespace pa_kernels
