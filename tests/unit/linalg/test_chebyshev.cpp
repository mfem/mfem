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

TEST_CASE("Chebyshev symmetry", "[OperatorChebyshevSmoother]")
{
   const int order = GENERATE(2, 3, 4);
   const int cheb_order = GENERATE(2, 3);

   Mesh mesh = Mesh::MakeCartesian3D(4, 4, 4, Element::HEXAHEDRON);
   H1_FECollection fec(order, 3);
   FiniteElementSpace fespace(&mesh, &fec);

   Array<int> ess_tdof_list;
   fespace.GetBoundaryTrueDofs(ess_tdof_list);

   BilinearForm aform(&fespace);
   aform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   aform.AddDomainIntegrator(new DiffusionIntegrator);
   aform.Assemble();

   OperatorPtr opr;
   opr.SetType(Operator::ANY_TYPE);
   aform.FormSystemMatrix(ess_tdof_list, opr);

   Vector diag(fespace.GetTrueVSize());
   aform.AssembleDiagonal(diag);

   OperatorChebyshevSmoother smoother(*opr, diag, ess_tdof_list, cheb_order);

   const int n = smoother.Width();
   Vector left(n);
   Vector right(n);
   left.Randomize(1);
   right.Randomize(2);

   // test that x^T S y = y^T S x
   Vector smooth(n);
   smoother.Mult(right, smooth);
   double forward_val = left * smooth;

   smoother.Mult(left, smooth);
   double transpose_val = right * smooth;

   double error = std::abs(forward_val - transpose_val) / std::abs(forward_val);
   CAPTURE(order, error);
   REQUIRE(error == MFEM_Approx(0.0));
}
