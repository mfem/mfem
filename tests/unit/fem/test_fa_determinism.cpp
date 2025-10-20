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

TEST_CASE("FA Determinism", "[PartialAssembly][GPU]")
{
   const int order = 3;
   const char *mesh_filename = "../../data/star-q3.mesh";

   Mesh mesh = Mesh::LoadFromFile(mesh_filename);

   const int dim = mesh.Dimension();

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);

   Vector x, b_reference;

   const int ntrials = 2;
   for (int i = 0; i < ntrials; ++i)
   {
      BilinearForm a(&fes);
      a.AddDomainIntegrator(new DiffusionIntegrator);
      a.SetAssemblyLevel(AssemblyLevel::FULL);
      a.EnableSparseMatrixSorting(Device::Allows(~Backend::CPU_MASK));
      a.Assemble();
      a.Finalize();

      SparseMatrix &A = a.SpMat();
      const int n = A.Height();

      if (x.Size() == 0)
      {
         x.SetSize(n);
         x.Randomize(1);
      }

      Vector b(n);
      A.Mult(x, b);

      if (b_reference.Size() == 0)
      {
         b_reference = b;
      }
      else
      {
         b -= b_reference;
         REQUIRE(b.Normlinf() == 0.0);
      }
   }
}
