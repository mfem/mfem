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
#include "../mesh/mesh_test_utils.hpp"

#include <cmath>
#include <limits>
#include <memory>

using namespace mfem;

#ifdef MFEM_USE_MPI

TEST_CASE("ProjectBdrCoefficientTangent face DOF orientation",
          "[Parallel][ND][DofTransformation]")
{
   const int num_procs = Mpi::WorldSize();
   if (num_procs == 1) { return; }

   Mesh smesh = DividingPlaneMesh(true, true, true);
   const int int_bdr_attr = smesh.bdr_attributes.Max();
   const int order = 2;
   ND_FECollection fec(order, 3);

   FiniteElementSpace sfes(&smesh, &fec);
   GridFunction sgf(&sfes);
   sgf = 0.0;
   VectorFunctionCoefficient coeff(3, [](const Vector &x, Vector &v)
   {
      v[0] =  1.234 * x[1] - 2.357 * x[2];
      v[1] = -1.234 * x[0] + 3.572 * x[2];
      v[2] =  2.357 * x[0] - 3.572 * x[1];
   });
   Array<int> sbdr(smesh.bdr_attributes.Max());
   sbdr = 0;
   sbdr[int_bdr_attr - 1] = 1;
   sgf.ProjectBdrCoefficientTangent(coeff, sbdr);

   std::unique_ptr<int[]> partitioning(
      smesh.GeneratePartitioning(num_procs));
   ParMesh pmesh(MPI_COMM_WORLD, smesh, partitioning.get());
   ParFiniteElementSpace pfes(&pmesh, &fec);

   Array<int> pbdr(pmesh.bdr_attributes.Max());
   pbdr = 0;
   pbdr[int_bdr_attr - 1] = 1;
   ParGridFunction expected(&pmesh, &sgf, partitioning.get());

   Array<int> ltori, ldsize;
   const int local_face_nnz =
      pfes.GetSharedTriFaceDofOrientations(ltori, ldsize);
   int local_marked_face_dofs = 0;
   int local_exercised_face_pairs = 0;
   Array<int> ess_vdofs;
   pfes.GetEssentialVDofs(pbdr, ess_vdofs);
   const real_t tol = 100 * std::numeric_limits<real_t>::epsilon();
   for (int i = 0; i < ltori.Size(); i++)
   {
      if (ldsize[i] != 2) { continue; }
      local_marked_face_dofs += 2;
      if (ltori[i] != 0)
      {
         local_exercised_face_pairs +=
            (ess_vdofs[i] && ess_vdofs[i+1] &&
             (std::abs(expected(i)) > tol ||
              std::abs(expected(i+1)) > tol));
      }
      i++;
   }
   REQUIRE(local_face_nnz == 2 * local_marked_face_dofs);
   int global_exercised_face_pairs = 0;
   MPI_Allreduce(&local_exercised_face_pairs, &global_exercised_face_pairs, 1,
                 MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   REQUIRE(global_exercised_face_pairs > 0);

   ParGridFunction pgf(&pfes);
   pgf = 0.0;
   pgf.ProjectBdrCoefficientTangent(coeff, pbdr);

   Vector error(pgf);
   error -= expected;
   real_t local_error = error.Normlinf();
   real_t global_error = 0.0;
   MPI_Allreduce(&local_error, &global_error, 1, MFEM_MPI_REAL_T, MPI_MAX,
                 MPI_COMM_WORLD);

   REQUIRE(global_error == MFEM_Approx(0.0, tol, tol));
}

#endif
