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

#include "unit_tests.hpp"
#include "mfem.hpp"

using namespace mfem;

double func_getgrad_test(const Vector &coord)
{
   double x = coord(0),
          y = (coord.Size() > 1) ? coord(1) : 0.0,
          z = (coord.Size() > 2) ? coord(2) : 0.0;
   return x * x + y * y + z * z;
}

#ifdef MFEM_USE_MPI

TEST_CASE("GetGradient Shared Faces", "[ParGridFunction][Parallel]")
{
   const int fe_order = 3;
   for (int dim = 1; dim <= 3; dim++)
   {
      int num_procs;
      MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
      int myid;
      MPI_Comm_rank(MPI_COMM_WORLD, &myid);

      Mesh mesh;
      if (dim == 1)
      {
         mesh = Mesh::MakeCartesian1D(100, 1.0);
      }
      else if (dim == 2)
      {
         mesh = Mesh::LoadFromFile("../../data/star-mixed.mesh");
      }
      else
      {
         mesh = Mesh::LoadFromFile("../../data/fichera-mixed.mesh");
      }
      for (int i = 0; i < 2; i++) { mesh.UniformRefinement(); }
      ParMesh pmesh(MPI_COMM_WORLD, mesh);

      FunctionCoefficient x_coeff(func_getgrad_test);
      H1_FECollection fec(fe_order, dim);
      ParFiniteElementSpace pfes(&pmesh, &fec);
      ParGridFunction pgf(&pfes);
      pgf.ProjectCoefficient(x_coeff);
      pgf.ExchangeFaceNbrData();

      double max_error = 0.0;
      Vector grad_1(dim), grad_2(dim);
      for (int f = 0; f < pmesh.GetNSharedFaces(); f++)
      {
         FaceElementTransformations *ft = pmesh.GetSharedFaceTransformations(f);
         ElementTransformation *t1 = &ft->GetElement1Transformation();
         ElementTransformation *t2 = &ft->GetElement2Transformation();
         const IntegrationRule &ir = IntRules.Get(ft->GetGeometryType(),
                                                  2*fe_order + 2);

         for (int q = 0; q < ir.GetNPoints(); q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            ft->SetAllIntPoints(&ip);

            // On both sides, the function should be represented exactly.
            pgf.GetGradient(*t1, grad_1);
            pgf.GetGradient(*t2, grad_2);

            Vector g(grad_1);
            grad_1 -= grad_2;
            max_error = fmax(max_error, grad_1.Norml2());
         }
      }

      REQUIRE(max_error == MFEM_Approx(0.0));
   }
}

#endif
