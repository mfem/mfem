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
//           -----------------------------------------------------
//           Particle-In-Cell (PIC) Simulation (2D/3D)
//           -----------------------------------------------------
//
// This miniapp performs a Particle-In-Cell simulation (supports 2D or 3D
// spatial dimensions) of multiple charged particles subject to electric
// field forces.
//
//                           dp/dt = q E
//
// The method used is explicit time integration with a leap-frog scheme.
//
// The electric field is computed from the particle charge distribution using
// a Poisson solver. The particle trajectories are computed within a periodic
// domain (2D or 3D).
//
// Solution process (per timestep, repeating steps 1-6):
//   (1) Deposit charge from particles to grid via Dirac delta function
//       to form the RHS of the Poisson equation
//   (2) Solve Poisson equation (-Δφ = ρ - ρ_0) to compute potential φ, where
//       ρ_0 is a constant neutralizing term that enforces global charge
//       neutrality.
//   (3) Compute electric field E = -∇φ from the potential
//   (4) Interpolate E-field to particle positions
//   (5) Push particles using leap-frog scheme (update momentum and position)
//   (6) Redistribute particles across processors
//
// Compile with: make electrostatic-pic
//
// Sample runs:
//
//   2D2V Linear Landau damping test case (Ricketson & Hu, 2025):
//      mpirun -n 4 ./electrostatic-pic -rdi 1 -npt 409600 -k 0.2855993321 -a 0.05 -nt 200 -nx 32 -ny 32 -O 1 -q 0.001181640625 -m 0.001181640625 -oci 1000 -dt 0.1
//   3D3V Linear Landau damping test case (Zheng et al., 2025):
//      mpirun -n 128 ./electrostatic-pic -dim 3 -rdi 1 -npt 40960000 -k 0.5 -a 0.01 -nt 100 -nx 32 -ny 32 -nz 32 -O 1 -q 0.00004844730731 -m 0.00004844730731 -oci 1000 -dt 0.02 -no-vis
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../../general/text.hpp"
#include "../common/fem_extras.hpp"
#include "../common/pfem_extras.hpp"
#include "FieldSolver.hpp"
#include "ParticleMover.hpp"
#include "mfem.hpp"

using namespace std;
using namespace mfem;
using namespace mfem::common;

struct PICContext
{
   int dim = 2;     ///< Spatial dimension.
   int order = 1;   ///< FE order for spatial discretization.
   int nx = 100;    ///< Number of grid cells in x-direction.
   int ny = 100;    ///< Number of grid cells in y-direction.
   int nz = 100;    ///< Number of grid cells in z-direction.
   real_t L = 1.0;  ///< Domain length.

   int ordering = 1;  ///< Ordering of particles.
   int npt = 1000;    ///< Number of particles.
   real_t q = 1.0;    ///< Particle charge.
   real_t m = 1.0;    ///< Particle mass.

   real_t k = 1.0;      ///< Wave number (Landau damping init).
   real_t alpha = 0.1;  ///< Perturbation amplitude (Landau damping init).

   real_t dt = 1e-2;  ///< Time step size.
   real_t diffusivity = 0.0;  ///< Diffusivity coefficient c for diffusion matrix.

   int nt = 1000;            ///< Number of time steps to run.
   int redist_interval = 5;  ///< Redistribution and update E_gf interval.
   int output_csv_interval = 1000;  ///< Interval for outputting CSV data files.

   bool visualization = true;  ///< Enable visualization.
   int visport = 19916;        ///< Port number for visualization server.
   bool reproduce = true;      ///< Enable reproducible results.
} ctx;

/// Prints the program's logo to the given output stream
void display_banner(ostream& os);

int main(int argc, char* argv[])
{
   Mpi::Init(argc, argv);
   int num_ranks = Mpi::WorldSize();
   int rank = Mpi::WorldRank();
   Hypre::Init();

   if (Mpi::Root()) { display_banner(cout); }

   OptionsParser args(argc, argv);
   args.AddOption(&ctx.dim, "-dim", "--dimension",
                  "Spatial dimension (2 or 3)");
   args.AddOption(&ctx.order, "-O", "--order",
                  "Finite element polynomial degree");
   args.AddOption(&ctx.nx, "-nx", "--num-x",
                  "Number of elements in the x direction.");
   args.AddOption(&ctx.ny, "-ny", "--num-y",
                  "Number of elements in the y direction.");
   args.AddOption(&ctx.nz, "-nz", "--num-z",
                  "Number of elements in the z direction.");
   args.AddOption(&ctx.q, "-q", "--charge", "Particle charge.");
   args.AddOption(&ctx.m, "-m", "--mass", "Particle mass.");
   args.AddOption(&ctx.dt, "-dt", "--time-step", "Time Step.");
   args.AddOption(&ctx.diffusivity, "-diff", "--diffusivity",
                  "Diffusivity coefficient c for diffusion matrix.");
   args.AddOption(&ctx.nt, "-nt", "--num-timesteps", "Number of timesteps.");
   args.AddOption(&ctx.npt, "-npt", "--num-particles",
                  "Total number of particles.");
   args.AddOption(&ctx.k, "-k", "--k", "Wave number for initial distribution.");
   args.AddOption(&ctx.alpha, "-a", "--alpha",
                  "Perturbation amplitude for initial distribution.");
   args.AddOption(&ctx.ordering, "-o", "--ordering",
                  "Ordering of particle data. 0 = byNODES, 1 = byVDIM.");
   args.AddOption(&ctx.redist_interval, "-rdi", "--redist-interval",
                  "Redistribution and update E_gf interval. Disabled if < 0.");
   args.AddOption(&ctx.output_csv_interval, "-oci", "--output-csv-interval",
                  "Output CSV interval. Disabled if < 0.");
   args.AddOption(&ctx.visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&ctx.visport, "-p", "--send-port", "Socket for GLVis.");
   args.AddOption(&ctx.reproduce, "-rep", "--reproduce", "-no-rep",
                  "--no-reproduce",
                  "Enable or disable reproducible random seed.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (Mpi::Root()) { args.PrintOptions(cout); }

   // Assert that dimension is 2 or 3
   MFEM_VERIFY(ctx.dim == 2 || ctx.dim == 3,
               "Dimension must be 2 or 3, got " << ctx.dim);
   MFEM_VERIFY(ctx.alpha >= -1.0 && ctx.alpha < 1.0,
               "Alpha should be in range [-1, 1).");
   MFEM_VERIFY(ctx.k > 0.0,
               "k must be nonzero for displacement initialization.");

   ctx.L = 2.0 * M_PI / ctx.k;

   // 1. make a Cartesian Mesh (2D or 3D)
   Mesh serial_mesh;
   std::vector<Vector> translations;

   if (ctx.dim == 2)
   {
      serial_mesh = Mesh(Mesh::MakeCartesian2D(
         ctx.nx, ctx.ny, Element::QUADRILATERAL, false, ctx.L, ctx.L));
      translations = {Vector({ctx.L, 0.0}), Vector({0.0, ctx.L})};
   }
   else  // ctx.dim == 3
   {
      serial_mesh = Mesh(Mesh::MakeCartesian3D(
         ctx.nx, ctx.ny, ctx.nz, Element::HEXAHEDRON, ctx.L, ctx.L, ctx.L));
      translations = {Vector({ctx.L, 0.0, 0.0}), Vector({0.0, ctx.L, 0.0}),
                      Vector({0.0, 0.0, ctx.L})};
   }

   Mesh periodic_mesh(Mesh::MakePeriodic(
      serial_mesh, serial_mesh.CreatePeriodicVertexMapping(translations)));
   // 2. Partition and distribute the mesh
   ParMesh mesh(MPI_COMM_WORLD, periodic_mesh);
   serial_mesh.Clear();    // the serial mesh is no longer needed
   periodic_mesh.Clear();  // the periodic mesh is no longer needed

   // 3. Build the interpolator of E field
   mesh.EnsureNodes();
   FindPointsGSLIB E_finder(mesh);

   // 4. Define finite element spaces on the parallel mesh
   H1_FECollection phi_fec(ctx.order, ctx.dim);
   ParFiniteElementSpace phi_fespace(&mesh, &phi_fec);
   ND_FECollection E_fec(ctx.order, ctx.dim);
   ParFiniteElementSpace E_fespace(&mesh, &E_fec);

   // 5. Initialize the grid functions for the electric field and potential
   ParGridFunction phi_gf(&phi_fespace);
   ParGridFunction E_gf(&E_fespace);
   phi_gf = 0.0;  // Initialize phi_gf to zero
   E_gf = 0.0;    // Initialize E_gf to zero

   // 6. Construct the field solver
   FieldSolver field_solver(&phi_fespace, &E_fespace, E_finder, ctx.diffusivity,
                            true);

   // 7. Initialize ParticleMover
   Ordering::Type ordering_type =
      ctx.ordering == 0 ? Ordering::byNODES : Ordering::byVDIM;
   int num_particles =
      ctx.npt / num_ranks + (rank < (ctx.npt % num_ranks) ? 1 : 0);
   ParticleMover particle_mover(MPI_COMM_WORLD, &E_gf, E_finder, num_particles,
                                ordering_type);
   particle_mover.InitializeChargedParticles(ctx.k, ctx.alpha, ctx.m, ctx.q,
                                             ctx.L, ctx.reproduce);

   // 8. Start the main loop
   real_t t = 0;
   real_t dt = ctx.dt;

   mfem::StopWatch sw;
   sw.Start();
   for (int step = 1; step <= ctx.nt; step++)
   {
      // Step the FieldSolver
      if (ctx.redist_interval > 0 &&
          (step % ctx.redist_interval == 0 || step == 1) &&
          particle_mover.GetParticles().GetGlobalNParticles() > 0)
      {
         // Redistribute
         particle_mover.Redistribute();

         // Update phi_gf from particles
         field_solver.UpdatePhiGridFunction(particle_mover.GetParticles(),
                                            phi_gf);
         // Update E_gf from phi_gf
         field_solver.UpdateEGridFunction(phi_gf, E_gf);

         // Visualize fields if requested
         if (ctx.visualization)
         {
            static socketstream vis_e, vis_phi;
            common::VisualizeField(vis_e, "localhost", ctx.visport, E_gf,
                                   "E_field", 0, 0, 500, 500);
            common::VisualizeField(vis_phi, "localhost", ctx.visport, phi_gf,
                                   "Potential", 500, 0, 500, 500);
         }

         // Compute energies
         real_t kinetic_energy = particle_mover.ComputeKineticEnergy();
         real_t field_energy = field_solver.ComputeFieldEnergy(E_gf);

         // Output energies
         if (Mpi::Root())
         {
            cout << "Kinetic energy: " << kinetic_energy << "\t";
            cout << "Field energy: " << field_energy << "\t";
            cout << "Total energy: " << kinetic_energy + field_energy << endl;
         }
         // Write energies to a CSV file
         if (Mpi::Root())
         {
            std::ofstream energy_file("energy.csv", std::ios::app);
            energy_file << setprecision(10) << kinetic_energy << ","
                        << field_energy << "," << kinetic_energy + field_energy
                        << "\n";
         }
      }

      // Step the ParticleMover
      particle_mover.Step(t, dt, ctx.L, step == 1);
      if (Mpi::Root())
      {
         mfem::out << "Step: " << step << " | Time: " << t;
         mfem::out << " | Time per step: " << sw.RealTime() / step;
         mfem::out << endl;
      }
      // Output particle data to CSV
      if (ctx.output_csv_interval > 0 &&
          (step % ctx.output_csv_interval == 0 || step == 1))
      {
         std::string csv_prefix = "PIC_Part_";
         Array<int> field_idx{2}, tag_idx;
         std::string file_name =
            csv_prefix + mfem::to_padded_string(step, 6) + ".csv";
         particle_mover.GetParticles().PrintCSV(file_name.c_str(), field_idx,
                                                tag_idx);
      }
   }
}

void display_banner(ostream& os)
{
   os << R"(
      ██████╗░██╗░█████╗░
      ██╔══██╗██║██╔══██╗
      ██████╔╝██║██║░░╚═╝
      ██╔═══╝░██║██║░░██╗
      ██║░░░░░██║╚█████╔╝
      ╚═╝░░░░░╚═╝░╚════╝░
         )"
      << endl
      << flush;
}
