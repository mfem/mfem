// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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

TEST_CASE("OperatorChebyshevSmoother", "[Chebyshev symmetry]")
{
   for (int order = 2; order < 5; ++order)
   {
      const int cheb_order = 2;

      Mesh mesh = Mesh::MakeCartesian3D(4, 4, 4, Element::HEXAHEDRON);
      FiniteElementCollection *fec = new H1_FECollection(order, 3);
      FiniteElementSpace fespace(&mesh, fec);
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      Array<int> ess_tdof_list;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      BilinearForm aform(&fespace);
      aform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      aform.AddDomainIntegrator(new DiffusionIntegrator);
      aform.Assemble();
      OperatorPtr opr;
      opr.SetType(Operator::ANY_TYPE);
      aform.FormSystemMatrix(ess_tdof_list, opr);
      Vector diag(fespace.GetTrueVSize());
      aform.AssembleDiagonal(diag);

      Solver* smoother = new OperatorChebyshevSmoother(*opr, diag, ess_tdof_list,
                                                       cheb_order);

      int n = smoother->Width();
      Vector left(n);
      Vector right(n);
      int seed = (int) time(0);
      left.Randomize(seed);
      right.Randomize(seed + 2);

      // test that x^T S y = y^T S x
      Vector smooth(n);
      smooth = 0.0;
      smoother->Mult(right, smooth);
      double forward_val = left * smooth;
      smoother->Mult(left, smooth);
      double transpose_val = right * smooth;

      double error = fabs(forward_val - transpose_val) / fabs(forward_val);
      CAPTURE(order, error);
      REQUIRE(error < 1.e-13);

      delete smoother;
      delete fec;
   }
}
