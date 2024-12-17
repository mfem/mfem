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

TEST_CASE("ParGridFunction in Serial", "[ParGridFunction][Parallel]")
{
   const int num_procs = Mpi::WorldSize();
   const int my_rank = Mpi::WorldRank();

   const int order = 1;
   const int save_rank = 0;

   const int n = 2 * num_procs;
   Mesh mesh = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL);
   mesh.SetCurvature(2);

   // Define a parallel mesh by a partitioning of the serial mesh.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   H1_FECollection fec(order, mesh.Dimension());
   ParFiniteElementSpace pfespace(&pmesh, &fec);
   ParGridFunction px(&pfespace);
   px.Randomize(1);
   // Ensure that the L-DOFs are set consistently on all ranks
   px.SetTrueVector();
   px.SetFromTrueVector();

   ConstantCoefficient zero(0.0);
   const double l2_norm = px.ComputeL2Error(zero);

   // Get the ParMesh and ParGridFunction on 1 of the mpi ranks. Check the
   // L2 error on that rank and save gridfunction.
   Mesh par_to_ser_mesh = pmesh.GetSerialMesh(save_rank);
   GridFunction x_par_to_ser = px.GetSerialGridFunction(
                                  save_rank, par_to_ser_mesh);
   if (my_rank == save_rank)
   {
      const double par_to_ser_l2_norm = x_par_to_ser.ComputeL2Error(zero);
      REQUIRE(par_to_ser_l2_norm == MFEM_Approx(l2_norm));
      // Save to disk
      par_to_ser_mesh.Save("parallel_in_serial.mesh");
   }

   {
      FiniteElementSpace &fes = *x_par_to_ser.FESpace();
      GridFunction x_par_to_ser_2 = px.GetSerialGridFunction(save_rank, fes);
      x_par_to_ser_2 -= x_par_to_ser;
      REQUIRE(x_par_to_ser_2.Normlinf() == MFEM_Approx(0.0));
   }

   // Save the mesh and then load the saved mesh and gridfunction, and check
   // the L2 error on all ranks.
   px.SaveAsSerial("parallel_in_serial.gf", 16, save_rank);

   if (my_rank == save_rank)
   {
      Mesh par_to_ser_mesh_read = Mesh("parallel_in_serial.mesh");
      named_ifgzstream gfstream("parallel_in_serial.gf");
      GridFunction x_par_to_ser_read(&par_to_ser_mesh_read, gfstream);
      const double par_to_ser_l2_read_norm = x_par_to_ser_read.ComputeL2Error(zero);

      REQUIRE(par_to_ser_l2_read_norm == MFEM_Approx(l2_norm));
   }

   if (my_rank == save_rank)
   {
      // Clean up
      REQUIRE(std::remove("parallel_in_serial.mesh") == 0);
      REQUIRE(std::remove("parallel_in_serial.gf") == 0);
   }
}

#endif // MFEM_USE_MPI
