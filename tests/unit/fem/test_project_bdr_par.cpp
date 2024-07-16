// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

#ifdef MFEM_USE_MPI

TEST_CASE("ProjectBdrCoefficient", "[Parallel]")
{
   int num_procs = Mpi::WorldSize();

   Mesh serial_mesh = Mesh::MakeCartesian3D(num_procs, num_procs, 1,
                                            Element::HEXAHEDRON,
                                            1.0, 1.0, 0.1, false);

   // Assign alternating element attributes to each element to create a
   // checkerboard pattern
   for (int i = 0; i < serial_mesh.GetNE(); i++)
   {
      int attr = (i + (1 + num_procs % 2) * (i / num_procs)) % 2 + 1;
      serial_mesh.SetAttribute(i, attr);
   }

   int bdr_max = serial_mesh.bdr_attributes.Max();

   // Label all interior faces as boundary elements
   Array<int> v(4);
   for (int i=0; i < serial_mesh.GetNumFaces(); i++)
   {
      if (serial_mesh.FaceIsInterior(i))
      {
         serial_mesh.GetFaceVertices(i, v);
         serial_mesh.AddBdrQuad(v, bdr_max + i + 1);
      }
   }
   serial_mesh.FinalizeMesh();
   serial_mesh.SetAttributes();

   // Create an intentionally bad partitioning
   Array<int> partitioning(num_procs * num_procs);
   for (int i = 0; i < num_procs * num_procs; i++)
   {
      // The following creates a shifting pattern where neighboring elements
      // are never owned by the same processor
      partitioning[i] = (2 * num_procs - 1 - (i % num_procs) -
                         i / num_procs) % num_procs;
   }

   ParMesh par_mesh(MPI_COMM_WORLD, serial_mesh, partitioning);

   H1_FECollection h1fec(2, par_mesh.Dimension());
   ParFiniteElementSpace h1fes(&par_mesh, &h1fec);

   ParGridFunction gf(&h1fes);
   gf = 0.0;

   int par_bdr_max = par_mesh.bdr_attributes.Max();

   Array<int> all_bdr(par_bdr_max);
   all_bdr = 1;

   ConstantCoefficient coeff(123.456);
   gf.ProjectBdrCoefficient(coeff, all_bdr);

   // We projected a value to all interior and exterior boundary elements.
   // An interior boundary element is only owned by one of the two sharing processors
   // and we expect the GridFunction on each of the two processors to be the same value
   // on that face. We test this by checking that each element sum is the same.
   real_t local_sum_expected = 0.0;
   real_t local_sum = 0.0;
   for (int e = 0; e < par_mesh.GetNE(); e++)
   {
      Vector dof_vals;
      gf.GetElementDofValues(e, dof_vals);

      real_t e_sum = dof_vals.Sum();

      if (e == 0) { local_sum_expected = e_sum * par_mesh.GetGlobalNE(); }

      local_sum += e_sum;
   }
   real_t global_sum = 0.0, global_sum_expected = 0.0;
   MPI_Allreduce(&local_sum, &global_sum, 1,
                 MFEM_MPI_REAL_T, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&local_sum_expected, &global_sum_expected, 1,
                 MFEM_MPI_REAL_T, MPI_MAX, MPI_COMM_WORLD);

   REQUIRE(global_sum == MFEM_Approx(global_sum_expected));
}

#endif
