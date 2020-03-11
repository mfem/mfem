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

#include "catch.hpp"
#include "mfem.hpp"

using namespace mfem;

TEST_CASE("PAGradient", "PAGradient")
{
   const int order = 2;
   const int num_elements = 2;
   Mesh mesh(num_elements, num_elements, Element::QUADRILATERAL, true);
   int dim = mesh.Dimension();
   FiniteElementCollection *h1_fec = new H1_FECollection(order, dim);
   FiniteElementCollection *nd_fec = new ND_FECollection(order, dim);
   FiniteElementSpace h1_fespace(&mesh, h1_fec);
   FiniteElementSpace nd_fespace(&mesh, nd_fec);

   DiscreteLinearOperator assembled_grad(&h1_fespace, &nd_fespace);
   assembled_grad.AddDomainInterpolator(new GradientInterpolator);
   const int skip_zeros = 1;
   assembled_grad.Assemble(skip_zeros);
   assembled_grad.Finalize(skip_zeros);
   const SparseMatrix& assembled_grad_mat = assembled_grad.SpMat();

   DiscreteLinearOperator pa_grad(&h1_fespace, &nd_fespace);
   pa_grad.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   pa_grad.AddDomainInterpolator(new GradientInterpolator);
   pa_grad.Assemble();
   pa_grad.Finalize();

   // if (transpose)
   int insize = h1_fespace.GetVSize();
   int outsize = nd_fespace.GetVSize();
   Vector xv(insize);
   Vector assembled_y(outsize);
   Vector pa_y(outsize);

   xv.Randomize();
   assembled_grad_mat.Mult(xv, assembled_y);
   pa_grad.Mult(xv, pa_y);

   pa_y -= assembled_y;
   double error = pa_y.Norml2() / assembled_y.Norml2();
   std::cout << "error in PA gradient: " << error << std::endl;
   REQUIRE(error < 1.e-12);
}
