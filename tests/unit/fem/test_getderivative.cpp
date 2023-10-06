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

#include "unit_tests.hpp"
#include "mfem.hpp"

using namespace mfem;

namespace test_GetDerivative_parallel
{

double func(const Vector &coord)
{
   if (coord.Size() == 1) { return std::sin(coord(0)); }
   if (coord.Size() == 2) { return std::sin(coord(0)*coord(1)); }
   return std::sin(coord(0)*coord(1)*coord(2));
}

#ifdef MFEM_USE_MPI

// Compares serial vs parallel result of GetDerivative.
TEST_CASE("GetDerivative", "[Parallel]")
{
   for (int dimension = 1; dimension <= 3; ++dimension)
   {
      int num_procs;
      MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
      int myid;
      MPI_Comm_rank(MPI_COMM_WORLD, &myid);

      Mesh mesh;
      if (dimension == 1)
      {
         mesh = Mesh::MakeCartesian1D(100, 1.0);
      }
      else if (dimension == 2)
      {
         mesh = Mesh::LoadFromFile("../../data/star-mixed-p2.mesh");
      }
      else
      {
         mesh = Mesh::LoadFromFile("../../data/fichera-mixed-p2.mesh");
      }
      for (int i = 0; i < 2; i++) { mesh.UniformRefinement(); }
      ParMesh pmesh(MPI_COMM_WORLD, mesh);

      FunctionCoefficient x_coeff(func);
      H1_FECollection fec(3, dimension);

      // Serial.
      FiniteElementSpace fes(&mesh, &fec);
      GridFunction gf(&fes), gf_grad(&fes);
      gf.ProjectCoefficient(x_coeff);

      // Parallel.
      ParFiniteElementSpace pfes(&pmesh, &fec);
      ParGridFunction pgf(&pfes), pgf_grad(&pfes);
      pgf.ProjectCoefficient(x_coeff);

      ConstantCoefficient zero(0.0);
      for (int d = 0; d < dimension; d++)
      {
         gf.GetDerivative(1, d, gf_grad);
         pgf.GetDerivative(1, d, pgf_grad);
         REQUIRE(gf_grad.ComputeL2Error(zero) -
                 pgf_grad.ComputeL2Error(zero) == MFEM_Approx(0.0));
      }
   }
}

#endif

}
