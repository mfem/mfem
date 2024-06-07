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

namespace pgridfunc_save_in_serial
{

double squared(const Vector &x)
{
   double sum = 0.0;
   for (int d = 0; d < x.Size(); d++)
   {
      sum += std::pow(x(d), 2.0);
   }
   return sum;
}


#ifdef MFEM_USE_MPI
#
TEST_CASE("ParGridFunction in Serial",
          "[ParGridFunction]"
          "[Parallel]")
{
   int num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   int my_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   int order = 1;
   int save_rank = 0;
   int n = 2 * num_procs;

   FunctionCoefficient squaredFC(squared);
   Mesh mesh = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL);
   mesh.SetCurvature(2);
   H1_FECollection fec(order, mesh.Dimension());

   double ser_l2_err = 0.0;
   FiniteElementSpace fespace(&mesh, &fec);
   GridFunction x(&fespace);
   x.ProjectCoefficient(squaredFC);
   ser_l2_err = x.ComputeL2Error(squaredFC);

   // Define a parallel mesh by a partitioning of the serial mesh.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   ParFiniteElementSpace pfespace(&pmesh, &fec);
   ParGridFunction px(&pfespace);
   px.ProjectCoefficient(squaredFC);

   // Get the ParMesh and ParGridFunction on 1 of the mpi ranks. Check the
   // L2 error on that rank and save gridfunction.
   Mesh par_to_ser_mesh = pmesh.GetSerialMesh(save_rank);
   GridFunction x_par_to_ser = px.GetSerialGridFunction(save_rank,
                                                        par_to_ser_mesh);
   double par_to_ser_l2_err = 0.0;
   if (my_rank == save_rank)
   {
      par_to_ser_l2_err = x_par_to_ser.ComputeL2Error(squaredFC);
      REQUIRE(std::fabs(par_to_ser_l2_err-ser_l2_err) == MFEM_Approx(0.0));

      // Save
      par_to_ser_mesh.Save("parallel_in_serial.mesh");
   }
   MPI_Barrier(MPI_COMM_WORLD);

   // Save the mesh and then load the saved mesh and gridfunction, and check
   // the L2 error on all ranks.
   px.SaveAsSerial("parallel_in_serial.gf", 16, save_rank);
   Mesh par_to_ser_mesh_read = Mesh("parallel_in_serial.mesh");
   named_ifgzstream gfstream("parallel_in_serial.gf");
   GridFunction x_par_to_ser_read = GridFunction(&par_to_ser_mesh_read,
                                                 gfstream);
   double par_to_ser_l2_read_err = x_par_to_ser_read.ComputeL2Error(squaredFC);
   REQUIRE(std::fabs(par_to_ser_l2_read_err-ser_l2_err) == MFEM_Approx(0.0));
}
#endif // MFEM_USE_MPI

} // namespace pgridfunc_save_in_serial
