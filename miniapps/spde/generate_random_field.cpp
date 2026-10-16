// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details
//
//     -------------------------------------------------------------------
//     Gaussian Random Fields of Matern Covariance for Imperfect Materials
//     -------------------------------------------------------------------
//
//  See README.md for detailed description.
//
// Compile with: make generate_random_field
//
//  Sample runs:
//     (Basic usage)
//     mpirun -np 4 generate_random_field
//
//     (Generate 5 particles with random imperfections)
//     mpirun -np 4 generate_random_field -o 1 -r 3 -rp 3 -nu 2 -l1 0.015 -l2 0.015 -l3 0.015 -s 0.01 -t 0.08 -n 5 -pl2 3 -top 0 -rs
//
//     (Generate an Octet-Truss with random imperfections)
//     mpirun -np 4 generate_random_field -o 1 -r 3 -rp 3 -nu 2 -l1 0.02 -l2 0.02 -l3 0.02 -s 0.01 -t 0.08 -top 1 -rs
//
//     (Generate an Octet-Truss with random imperfections following a uniform distribution)
//     mpirun -np 4 generate_random_field -o 1 -r 3 -rp 3 -nu 2 -l1 0.02 -l2 0.02 -l3 0.02 -umin 0.01 -umax 0.05 -t 0.08 -top 1 -urf -rs
//
//     (2D random field with anisotropy)
//     mpirun -np 4 generate_random_field -o 1 -r 3 -rp 3 -nu 4 -l1 0.09 -l2 0.03 -l3 0.05 -s 0.01 -t 0.08 -top 1 -no-rs -m ../../data/ref-square.mesh

#include <iostream>
#include <string>
#include "mfem.hpp"

#include "material_metrics.hpp"
#include "spde_solver.hpp"
#include "transformation.hpp"
#include "util.hpp"
#include "visualizer.hpp"

using namespace std;
using namespace mfem;

enum TopologicalSupport { kParticles, kOctetTruss };

int main(int argc, char *argv[])
{
#ifdef MFEM_USE_SINGLE
   cout << "This miniapp is not supported in single precision.\n\n";
   return MFEM_SKIP_RETURN_VALUE;
#endif

   // 0. Initialize MPI.
   Mpi::Init(argc, argv);
   Hypre::Init();

   // 1. Parse command-line options.
   const char *mesh_file = "../../data/ref-cube.mesh";
   int order = 1;
   int num_refs = 3;
   int num_parallel_refs = 3;
   int number_of_particles = 3;
   int topological_support = TopologicalSupport::kOctetTruss;
   real_t nu = 2.0;
   real_t tau = 0.08;
   real_t l1 = 0.02;
   real_t l2 = 0.02;
   real_t l3 = 0.02;
   real_t e1 = 0;
   real_t e2 = 0;
   real_t e3 = 0;
   real_t pl1 = 1.0;
   real_t pl2 = 1.0;
   real_t pl3 = 1.0;
   real_t uniform_min = 0.0;
   real_t uniform_max = 1.0;
   real_t offset = 0.0;
   real_t scale = 0.01;
   real_t level_set_threshold = 0.0;
   bool paraview_export = true;
   bool glvis_export = true;
   bool uniform_rf = false;
   bool random_seed = true;
   bool compute_boundary_integrals = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&num_refs, "-r", "--refs", "Number of uniform refinements");
   args.AddOption(&num_parallel_refs, "-rp", "--refs-parallel",
                  "Number of uniform refinements");
   args.AddOption(&topological_support, "-top", "--topology",
                  "Topological support. 0 particles, 1 octet-truss");
   args.AddOption(&nu, "-nu", "--nu", "Fractional exponent nu (smoothness)");
   args.AddOption(&tau, "-t", "--tau", "Parameter for topology generation");
   args.AddOption(&l1, "-l1", "--l1",
                  "First component of diagonal core of theta");
   args.AddOption(&l2, "-l2", "--l2",
                  "Second component of diagonal core of theta");
   args.AddOption(&l3, "-l3", "--l3",
                  "Third component of diagonal core of theta");
   args.AddOption(&e1, "-e1", "--e1", "First euler angle for rotation of theta");
   args.AddOption(&e2, "-e2", "--e2",
                  "Second euler angle for rotation of theta");
   args.AddOption(&e3, "-e3", "--e3", "Third euler angle for rotation of theta");
   args.AddOption(&pl1, "-pl1", "--pl1", "Length scale 1 of particles");
   args.AddOption(&pl2, "-pl2", "--pl2", "Length scale 2 of particles");
   args.AddOption(&pl3, "-pl3", "--pl3", "Length scale 3 of particles");
   args.AddOption(&uniform_min, "-umin", "--uniform-min",
                  "Minimum value of uniform distribution");
   args.AddOption(&uniform_max, "-umax", "--uniform-max",
                  "Maximum value of uniform distribution");
   args.AddOption(&offset, "-off", "--offset",
                  "Offset for random field u(x) -> u(x) + a");
   args.AddOption(&scale, "-s", "--scale",
                  "Scale for random field u(x) -> a * u(x)");
   args.AddOption(&level_set_threshold, "-lst", "--level-set-threshold",
                  "Level set threshold");
   args.AddOption(&number_of_particles, "-n", "--number-of-particles",
                  "Number of particles");
   args.AddOption(&paraview_export, "-pvis", "--paraview-visualization",
                  "-no-pvis", "--no-paraview-visualization",
                  "Enable or disable ParaView visualization.");
   args.AddOption(&glvis_export, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&uniform_rf, "-urf", "--uniform-rf", "-no-urf",
                  "--no-uniform-rf",
                  "Enable or disable the transformation of GRF to URF.");
   args.AddOption(&random_seed, "-rs", "--random-seed", "-no-rs",
                  "--no-random-seed", "Enable or disable random seed.");
   args.AddOption(&compute_boundary_integrals, "-cbi",
                  "--compute-boundary-integrals", "-no-cbi",
                  "--no-compute-boundary-integrals",
                  "Enable or disable computation of boundary integrals.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(cout);
   }

   // 2. Read the mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   bool is_3d = (dim == 3);

   // 3. Refine the mesh to increase the resolution.
   for (int i = 0; i < num_refs; i++)
   {
      mesh.UniformRefinement();
   }
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   for (int i = 0; i < num_parallel_refs; i++)
   {
      pmesh.UniformRefinement();
   }

   // 4. Define a finite element space on the mesh.
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec);
   HYPRE_BigInt size = fespace.GlobalTrueVSize();
   if (Mpi::Root())
   {
      const Array<int> boundary(pmesh.bdr_attributes);
      cout << "Number of finite element unknowns: " << size << "\n";
      cout << "Boundary attributes: ";
      boundary.Print(cout, 6);
   }

   // ========================================================================
   // II. Generate topological support
   // ========================================================================

   ParGridFunction v(&fespace);
   v = 0.0;
   MaterialTopology *mdm = nullptr;

   // II.1 Define the metric for the topological support.
   if (is_3d)
   {
      if (topological_support == TopologicalSupport::kOctetTruss)
      {
         mdm = new OctetTrussTopology();
      }
      else if (topological_support == TopologicalSupport::kParticles)
      {
         // Create the same random particles on all processors.
         std::vector<real_t> random_positions(3 * number_of_particles);
         std::vector<real_t> random_rotations(9 * number_of_particles);
         if (Mpi::Root())
         {
            // Generate random positions and rotations. We generate them on the root
            // process and then broadcast them to all processes because we need the
            // same random positions and rotations on all processes.
            FillWithRandomNumbers(random_positions, 0.2, 0.8);
            FillWithRandomRotations(random_rotations);
         }

         // Broadcast the random positions and rotations to all processes.
         MPI_Bcast(random_positions.data(), 3 * number_of_particles,
                   MPITypeMap<real_t>::mpi_type, 0, MPI_COMM_WORLD);
         MPI_Bcast(random_rotations.data(), 9 * number_of_particles,
                   MPITypeMap<real_t>::mpi_type, 0, MPI_COMM_WORLD);

         mdm = new ParticleTopology(pl1, pl2, pl3, random_positions,
                                    random_rotations);
      }
      else
      {
         if (Mpi::Root())
         {
            mfem::out << "Error: Selected topological support not valid."
                      << std::endl;
         }
         return 1;
      }

      // II.2 Define lambda to wrap the call to the distance metric.
      auto topo = [&mdm, &tau](const Vector &x)
      {
         return (tau - mdm->ComputeMetric(x));
      };

      // II.3 Create a GridFunction for the topological support.
      FunctionCoefficient topo_coeff(topo);
      v.ProjectCoefficient(topo_coeff);
   }

   // ========================================================================
   // III. Generate random imperfections via fractional PDE
   // ========================================================================

   /// III.1 Define the fractional PDE solution
   ParGridFunction u(&fespace);
   u = 0.0;

   // III.2 Define the boundary conditions.
   spde::Boundary bc;
   if (Mpi::Root())
   {
      bc.PrintInfo();
      bc.VerifyDefinedBoundaries(pmesh);
   }

   // III.3 Solve the SPDE problem
   spde::SPDESolver solver(nu, bc, &fespace, l1, l2, l3, e1, e2,
                           e3);
   const int seed = (random_seed) ? 0 :
                    std::numeric_limits<int>::max() - Mpi::WorldRank();
   solver.SetupRandomFieldGenerator(seed);
   solver.GenerateRandomField(u);

   /// III.4 Verify boundary conditions
   if (compute_boundary_integrals)
   {
      bc.ComputeBoundaryError(u);
   }

   // ========================================================================
   // III. Combine topological support and random field
   // ========================================================================

   if (uniform_rf)
   {
      /// Transform the random field to a uniform random field.
      spde::UniformGRFTransformer transformation(uniform_min, uniform_max);
      transformation.Transform(u);
   }
   if (scale != 1.0)
   {
      /// Scale the random field.
      spde::ScaleTransformer transformation(scale);
      transformation.Transform(u);
   }
   if (offset != 0.0)
   {
      /// Add an offset to the random field.
      spde::OffsetTransformer transformation(offset);
      transformation.Transform(u);
   }
   ParGridFunction w(&fespace);  // Noisy material field.
   w = 0.0;
   w += u;
   w += v;
   ParGridFunction level_set(w);  // Level set field.
   {
      spde::LevelSetTransformer transformation(level_set_threshold);
      transformation.Transform(level_set);
   }

   // ========================================================================
   // IV. Export visualization to ParaView and GLVis
   // ========================================================================

   spde::Visualizer vis(pmesh, order, u, v, w, level_set, is_3d);
   if (paraview_export)
   {
      vis.ExportToParaView();
   }
   if (glvis_export)
   {
      vis.SendToGLVis();
   }

   delete mdm;
   return 0;
}
