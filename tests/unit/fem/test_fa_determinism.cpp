// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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

TEST_CASE("FA Determinism", "[PartialAssembly][CUDA]")
{
   Vector b_reference;

   const int order = 3;
   const char *mesh_filename = "../../data/star-q3.mesh";

   Mesh mesh = Mesh::LoadFromFile(mesh_filename);

   const int dim = mesh.Dimension();

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);

   const int ntrials = 100;

   for (int i = 0; i < ntrials; ++i)
   {
      BilinearForm a(&fes);
      a.AddDomainIntegrator(new DiffusionIntegrator);
      a.SetAssemblyLevel(AssemblyLevel::FULL);
      a.Assemble();
      a.Finalize();

      SparseMatrix &A = a.SpMat();
      const int n = A.Height();

      Vector x(n), b(n);

      x.Randomize(1);
      b = 0.0;

      A.Mult(x, b);

      if (b_reference.Size() > 0)
      {
         Vector b2 = b_reference;
         b2 -= b;
         REQUIRE(b2.Normlinf() == 0.0);
      }
      else
      {
         b_reference = b;
      }
   }
}
