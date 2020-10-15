// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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

#include <iostream>

using namespace mfem;

TEST_CASE("Test order of boundary integrators",
          "[BilinearForm]")
{
   // Create a simple mesh
   int dim = 2, nx = 2, ny = 2, order = 2;
   Element::Type e_type = Element::QUADRILATERAL;
   Mesh mesh(nx, ny, e_type);

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);

   SECTION("Order of restricted boundary integrators")
   {
      ConstantCoefficient one(1.0);
      ConstantCoefficient two(2.0);
      ConstantCoefficient three(3.0);
      ConstantCoefficient four(4.0);

      Array<int> bdr1(4); bdr1 = 0; bdr1[0] = 1;
      Array<int> bdr2(4); bdr2 = 0; bdr2[1] = 1;
      Array<int> bdr3(4); bdr3 = 0; bdr3[2] = 1;
      Array<int> bdr4(4); bdr4 = 0; bdr4[3] = 1;

      BilinearForm a1234(&fes);
      a1234.AddBoundaryIntegrator(new MassIntegrator(one), bdr1);
      a1234.AddBoundaryIntegrator(new MassIntegrator(two), bdr2);
      a1234.AddBoundaryIntegrator(new MassIntegrator(three), bdr3);
      a1234.AddBoundaryIntegrator(new MassIntegrator(four), bdr4);
      a1234.Assemble(0);
      a1234.Finalize(0);

      BilinearForm a4321(&fes);
      a4321.AddBoundaryIntegrator(new MassIntegrator(four), bdr4);
      a4321.AddBoundaryIntegrator(new MassIntegrator(three), bdr3);
      a4321.AddBoundaryIntegrator(new MassIntegrator(two), bdr2);
      a4321.AddBoundaryIntegrator(new MassIntegrator(one), bdr1);
      a4321.Assemble(0);
      a4321.Finalize(0);

      const SparseMatrix &A1234 = a1234.SpMat();
      const SparseMatrix &A4321 = a4321.SpMat();

      SparseMatrix *D = Add(1.0, A1234, -1.0, A4321);

      REQUIRE(D->MaxNorm() == MFEM_Approx(0.0));

      delete D;
   }
}
