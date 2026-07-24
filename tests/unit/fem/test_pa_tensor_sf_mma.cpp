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
#include "fem/integ/bilininteg_pa_tensor_sf_mma.hpp"

using namespace mfem;

namespace pa_tensor_sf_mma
{

void test_pa_tensor_sf_mma(int dim, int p)
{
   CAPTURE(dim, p);

   Mesh mesh = (dim == 2)
               ? Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL)
               : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);

   H1_FECollection fec(p, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);

   ForceTensorMmaPA(true);
   if (!CanUseTensorMmaPA(fes))
   {
      ForceTensorMmaPA(false);
      return;
   }

   GridFunction x(&fes), y_fa(&fes), y_pa(&fes);
   x.Randomize(0x100001b3);
   y_fa.Randomize(0x9e3779b9);
   y_pa = y_fa;

   const auto &fe = *fes.GetTypicalFE();
   // Match specialized (D1D,Q1D)=(p+1,p+2) pairs used by SF-MMA kernels.
   const IntegrationRule *ir = &IntRules.Get(fe.GetGeomType(), 2 * p + 2);

   ConstantCoefficient const_coeff(M_2_SQRTPI);
   FunctionCoefficient funct_coeff([](const Vector &pt)
   { return M_1_PI + pt[0] * pt[0]; });

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
   REQUIRE(y_fa.Normlinf() == MFEM_Approx(0.0, 1e-9, 1e-9));

   ForceTensorMmaPA(false);
}

TEST_CASE("Tensor SF-MMA PA vs FA", "[TensorSfMMA][GPU]")
{
   const int dim = GENERATE(2, 3);
   const int p = GENERATE(2, 3, 4, 5, 6);
   test_pa_tensor_sf_mma(dim, p);
}

TEST_CASE("Tensor SF-MMA eligibility", "[TensorSfMMA]")
{
   Mesh mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
   H1_FECollection fec(3, 3, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);

   ForceTensorMmaPA(false);
   REQUIRE_FALSE(CanUseTensorMmaPA(fes));

   ForceTensorMmaPA(true);
   // Without CUDA this returns false; with CUDA it should be true.
   if (Device::Allows(Backend::CUDA_MASK))
   {
      REQUIRE(CanUseTensorMmaPA(fes));
   }
   ForceTensorMmaPA(false);

   // p=1 is intentionally unsupported
   H1_FECollection fec1(1, 3, BasisType::GaussLobatto);
   FiniteElementSpace fes1(&mesh, &fec1);
   ForceTensorMmaPA(true);
   REQUIRE_FALSE(CanUseTensorMmaPA(fes1));
   ForceTensorMmaPA(false);
}

} // namespace pa_tensor_sf_mma
