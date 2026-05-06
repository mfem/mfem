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
//
//    ---------------------------------------------------------------------
//     Compute bounds of a random 1D grid function on a generated mesh
//    ---------------------------------------------------------------------
//
// This miniapp generates a 1D segment mesh with nx elements, builds a random
// discontinuous grid function, computes element-wise piecewise linear bounds,
// and visualizes the input field together with the lower and upper bounds.
//
// Compile with: make random-gridfunction-bounds
//
// Sample runs:
//   mpirun -np 4 random-gridfunction-bounds
//   mpirun -np 4 random-gridfunction-bounds -nx 64 -o 6 -ref 3 -d hip

#include "mfem.hpp"

using namespace mfem;
using namespace std;

void VisualizeField(ParMesh &pmesh, ParGridFunction &input,
                    char *title, int pos_x, int pos_y);

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   int nx = 16;
   int order = 4;
   int ref = 2;
   int seed = 12345;
   bool visualization = false;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&nx, "-nx", "--num-elements",
                  "Number of 1D mesh elements.");
   args.AddOption(&order, "-o", "--order",
                  "Polynomial degree of the random discontinuous field.");
   args.AddOption(&ref, "-ref", "--piecewise-linear-ref-factor",
                  "Scaling factor for the resolution of the piecewise linear "
                  "bounds. If less than 2, the resolution is picked "
                  "automatically.");
   args.AddOption(&seed, "-rs", "--random-seed",
                  "Random seed used to initialize the field.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.ParseCheck();

   MFEM_VERIFY(nx > 0, "nx must be positive.");
   MFEM_VERIFY(order >= 0, "order must be non-negative.");

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   Mesh mesh = Mesh::MakeCartesian1D(nx, 1.0);
   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   const int dim = pmesh.Dimension();
   L2_FECollection fec(order, dim, BasisType::GaussLobatto);
   ParFiniteElementSpace fes(&pmesh, &fec);
   ParGridFunction input(&fes);
   input.Randomize(seed + Mpi::WorldRank());

   L2_FECollection fec_pc(0, dim);
   ParFiniteElementSpace fes_pc(&pmesh, &fec_pc, 1, Ordering::byNODES);
   ParGridFunction lowerb(&fes_pc), upperb(&fes_pc);

   PLBound plb = input.GetElementBounds(lowerb, upperb, ref);

   real_t lower_min = lowerb.Min();
   real_t upper_max = upperb.Max();
   MPI_Allreduce(MPI_IN_PLACE, &lower_min, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MIN, pmesh.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &upper_max, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MAX, pmesh.GetComm());

   if (Mpi::Root())
   {
      cout << "nx: " << nx << '\n'
           << "order: " << order << '\n'
           << "PL bound control-point factor: " << ref << '\n'
           << "global lower bound minimum: " << lower_min << '\n'
           << "global upper bound maximum: " << upper_max << endl;
   }

   if (visualization)
   {
      char title1[] = "Random input gridfunction";
      char title2[] = "Element-wise lower bound";
      char title3[] = "Element-wise upper bound";
      VisualizeField(pmesh, input, title1, 0, 0);
      VisualizeField(pmesh, lowerb, title2, 450, 0);
      VisualizeField(pmesh, upperb, title3, 900, 0);
   }

   return 0;
}

void VisualizeField(ParMesh &pmesh, ParGridFunction &input,
                    char *title, int pos_x, int pos_y)
{
   socketstream sock;
   if (pmesh.GetMyRank() == 0)
   {
      sock.open("localhost", 19916);
      sock << "solution\n";
   }
   pmesh.PrintAsOne(sock);
   input.SaveAsOne(sock);
   if (pmesh.GetMyRank() == 0)
   {
      sock << "window_title '" << title << "'\n"
           << "window_geometry "
           << pos_x << " " << pos_y << " " << 400 << " " << 400 << "\n"
           << "keys jRmclApppppppppppp//]]]]]]]]" << endl;
   }
}
