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
//      mpirun -n 4 ./electrostatic-pic -case 0 -rdi 1 -npt 409600 -k 0.2855993321 -a 0.05 -nt 200 -nx 32 -ny 32 -O 1 -oci 1000 -dt 0.1 -diff 10 -eoi 10
//   2D2V Landau with x-only cos(kx) density via inverse CDF (quiet start):
//      mpirun -n 4 ./electrostatic-pic -case 0 -landau1d -use-its -rdi 1 -npt 409600 -k 0.2855993321 -a 0.05 -nt 200 -nx 32 -ny 32 -O 1 -oci 1000 -dt 0.1 -no-vis
//   2D2V Two-stream instability (warm beams):
//      mpirun -n 4 ./electrostatic-pic -case 1 -rdi 1 -npt 409600 -k 0.2855993321  -v0 0.5 -vvar 0.01 -nt 200 -nx 32 -ny 32 -O 1 -oci 1000 -dt 0.1 -no-vis
//   2D2V Bump-on-tail:
//      mpirun -n 4 ./electrostatic-pic -case 2 -rdi 1 -npt 409600 -k 0.2855993321 -bf 0.1 -vb 4.5 -vth 1.0 -vtb 0.5 -nt 200 -nx 32 -ny 32 -O 1 -oci 1000 -dt 0.1 -no-vis
//   3D3V Linear Landau damping test case (Zheng et al., 2025):
//      mpirun -n 128 ./electrostatic-pic -dim 3 -rdi 1 -npt 40960000 -k 0.5 -a 0.01 -nt 100 -nx 32 -ny 32 -nz 32 -O 1 -oci 1000 -dt 0.02 -no-vis
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../../general/text.hpp"
#include "../common/pfem_extras.hpp"
#include "FieldSolver.hpp"
#include "ParticleMover.hpp"

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
   real_t q = -1.0;    ///< Particle charge. set to -1: auto-normalized
   real_t m = -1.0;    ///< Particle mass. set to -1: auto-normalized

   real_t k = 1.0;      ///< Wave number for initial distribution.
   real_t alpha = 0.1;  ///< Density perturbation amplitude.
   bool landau_x =
      false;  ///< Case 0: perturb density along x only (not all axes).
   bool use_its =
      false;  ///< Case 0: sample x from n~[1+alpha cos(kx)]/L via inverse CDF.

   int init_case = 0;  ///< 0 = Landau, 1 = two-stream, 2 = bump-on-tail.
   real_t v0 = 0.5;    ///< Counter-streaming beam speed (case 1).
   real_t beam_variance =
      0.0;  ///< Variance of each counter-streaming beam (case 1).
   real_t bump_fraction = 0.1;  ///< Bump weight in f0 (case 2).
   real_t vb = 4.5;             ///< Bump beam speed (case 2).
   real_t vth = 1.0;            ///< Bulk thermal speed (case 2).
   real_t vtb = 0.5;            ///< Bump thermal speed (case 2).

   real_t dt = 1e-2;  ///< Time step size.
   real_t diffusivity =
      0.0;  ///< Diffusivity coefficient c for diffusion matrix.
   int efield_output_interval =
      1000;  ///< E-field sampling CSV interval. Disabled if < 0.
   int phi_output_interval =
      1000;  ///< Phi sampling CSV interval. Disabled if < 0.
   int rho_output_interval =
      1000;  ///< Rho sampling CSV interval. Disabled if < 0.
   int field_sample_resolution =
      512;  ///< Sample grid resolution N for all field outputs.

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
   args.AddOption(&ctx.q, "-q", "--charge",
                  "Particle charge. If < 0, set so npt*q/L^dim = 1.");
   args.AddOption(&ctx.m, "-m", "--mass",
                  "Particle mass. If < 0, set so npt*m/L^dim = 1.");
   args.AddOption(&ctx.dt, "-dt", "--time-step", "Time Step.");
   args.AddOption(&ctx.diffusivity, "-diff", "--diffusivity",
                  "Diffusivity coefficient c for diffusion matrix.");
   args.AddOption(&ctx.efield_output_interval, "-eoi",
                  "--efield-output-interval",
                  "E-field sample CSV output interval. Disabled if < 0. "
                  "Use 0 to output every field update.");
   args.AddOption(&ctx.phi_output_interval, "-poi", "--phi-output-interval",
                  "Phi sample CSV output interval. Disabled if < 0. "
                  "Use 0 to output every field update.");
   args.AddOption(&ctx.rho_output_interval, "-roi", "--rho-output-interval",
                  "Rho sample CSV output interval. Disabled if < 0. "
                  "Use 0 to output every field update.");
   args.AddOption(&ctx.field_sample_resolution, "-fsr",
                  "--field-sample-resolution",
                  "Sample resolution N for an N x N output grid for E, phi, "
                  "and rho.");
   args.AddOption(&ctx.nt, "-nt", "--num-timesteps", "Number of timesteps.");
   args.AddOption(&ctx.npt, "-npt", "--num-particles",
                  "Total number of particles.");
   args.AddOption(&ctx.k, "-k", "--k", "Wave number for initial distribution.");
   args.AddOption(&ctx.init_case, "-case", "--case",
                  "Initial distribution: 0 = Landau, 1 = two-stream, "
                  "2 = bump-on-tail.");
   args.AddOption(&ctx.alpha, "-a", "--alpha",
                  "Perturbation amplitude for initial distribution "
                  "(case 0 only).");
   args.AddOption(&ctx.landau_x, "-landau1d", "--landau-1d",
                  "-no-landau1d", "--no-landau-1d",
                  "apply sin(kx) density perturbation along x only (case 0 only).");
   args.AddOption(&ctx.use_its, "-use-its", "--use-its", "-no-use-its",
                  "--no-use-its",
                  "sample x from n~[1+alpha cos(kx)]/L via inverse CDF "
                  "(case 0 only).");
   args.AddOption(&ctx.v0, "-v0", "--v0",
                  "Counter-streaming beam speed (case 1 only).");
   args.AddOption(&ctx.beam_variance, "-vvar", "--beam-variance",
                  "Variance of each counter-streaming beam (case 1 only).");
   args.AddOption(&ctx.bump_fraction, "-bf", "--bump-fraction",
                  "Bump weight in f0 (case 2 only).");
   args.AddOption(&ctx.vb, "-vb", "--vb",
                  "Bump beam speed v_b (case 2 only).");
   args.AddOption(&ctx.vth, "-vth", "--vth",
                  "Bulk thermal speed v_th (case 2 only).");
   args.AddOption(&ctx.vtb, "-vtb", "--vtb",
                  "Bump thermal speed v_tb (case 2 only).");
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

   // Assert that dimension is 2 or 3
   MFEM_VERIFY(ctx.dim == 2 || ctx.dim == 3,
               "Dimension must be 2 or 3, got " << ctx.dim);
   MFEM_VERIFY(ctx.npt > 0, "num-particles must be positive.");
   MFEM_VERIFY(ctx.alpha >= -1.0 && ctx.alpha < 1.0,
               "Alpha should be in range [-1, 1).");
   MFEM_VERIFY(ctx.k > 0.0,
               "k must be nonzero for displacement initialization.");
   MFEM_VERIFY(ctx.init_case == 0 || ctx.init_case == 1 || ctx.init_case == 2,
               "case must be 0 (Landau), 1 (two-stream), or 2 (bump-on-tail).");
   MFEM_VERIFY(!ctx.use_its || ctx.init_case == 0,
               "-use-its is only valid for case 0 (Landau).");
   MFEM_VERIFY(ctx.beam_variance >= 0.0,
               "beam-variance must be non-negative.");
   MFEM_VERIFY(ctx.bump_fraction >= 0.0 && ctx.bump_fraction <= 1.0,
               "bump-fraction must be in [0, 1].");
   MFEM_VERIFY(ctx.vth > 0.0, "vth must be positive.");
   MFEM_VERIFY(ctx.vtb > 0.0, "vtb must be positive.");

   ctx.L = 2.0 * M_PI / ctx.k;
   // Negative q/m means auto-normalize total density: npt*(q|m)/L^dim = 1.
   real_t vol = ctx.L * ctx.L;
   if (ctx.dim == 3) { vol *= ctx.L; }
   if (ctx.q < 0.0) { ctx.q = vol / ctx.npt; }
   if (ctx.m < 0.0) { ctx.m = vol / ctx.npt; }

   if (Mpi::Root()) { args.PrintOptions(cout); }

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
   ParGridFunction rho_gf(&phi_fespace);
   ParGridFunction E_gf(&E_fespace);
   phi_gf = 0.0;  // Initialize phi_gf to zero
   rho_gf = 0.0;  // Initialize rho_gf to zero
   E_gf = 0.0;    // Initialize E_gf to zero

   // 6. Construct the field solver
   FieldSolver field_solver(&phi_fespace, &E_fespace, E_finder, ctx.diffusivity,
                            true, ctx.efield_output_interval,
                            ctx.phi_output_interval, ctx.rho_output_interval,
                            ctx.field_sample_resolution);

   // 7. Initialize ParticleMover
   Ordering::Type ordering_type =
      ctx.ordering == 0 ? Ordering::byNODES : Ordering::byVDIM;
   int num_particles =
      ctx.npt / num_ranks + (rank < (ctx.npt % num_ranks) ? 1 : 0);
   ParticleMover particle_mover(MPI_COMM_WORLD, &E_gf, &phi_gf, &rho_gf,
                                E_finder, num_particles, ordering_type);
   particle_mover.InitializeChargedParticles(
      ctx.k, ctx.alpha, ctx.m, ctx.q, ctx.L, ctx.init_case, ctx.v0,
      ctx.beam_variance, ctx.bump_fraction, ctx.vb, ctx.vth, ctx.vtb,
      ctx.landau_x, ctx.use_its, ctx.reproduce);

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
                                            phi_gf, rho_gf);
         // Update E_gf from phi_gf
         field_solver.UpdateEGridFunction(phi_gf, E_gf);

         // Visualize fields if requested
         if (ctx.visualization)
         {
            static socketstream vis_e, vis_phi, vis_rho;
            common::VisualizeField(vis_e, "localhost", ctx.visport, E_gf,
                                   "E_field", 0, 0, 500, 500);
            common::VisualizeField(vis_phi, "localhost", ctx.visport, phi_gf,
                                   "Potential", 500, 0, 500, 500);
            common::VisualizeField(vis_rho, "localhost", ctx.visport, rho_gf,
                                   "Charge density", 1000, 0, 500, 500);

            // // Fix color scale to [emin, emax]
            // vis_phi << "autoscale off\n"
            //         << "valuerange " << -1 << " " << 1 << "\n"
            //         << flush;
         }
         field_solver.SaveFieldSamples(E_gf, phi_gf, rho_gf, step);

         // Compute energies
         real_t kinetic_energy = particle_mover.ComputeKineticEnergy();
         real_t field_energy = field_solver.ComputeFieldEnergy(E_gf);

         {
            ParLinearForm b(phi_gf.ParFESpace());
            GridFunctionCoefficient phi_coeff(&phi_gf);
            b.AddDomainIntegrator(new DomainLFIntegrator(phi_coeff));
            b.Assemble();

            field_solver.DiffuseRHS(b, phi_gf);
         }
         // Update E_gf from phi_gf
         field_solver.UpdateEGridFunction(phi_gf, E_gf);

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
         particle_mover.UpdateParticleOutputFields();
         Array<int> field_idx{ParticleMover::MOM, ParticleMover::PHI,
                              ParticleMover::RHO},
            tag_idx;
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
