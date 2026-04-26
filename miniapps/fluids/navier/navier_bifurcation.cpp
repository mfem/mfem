// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
//   ---------------------------------------------------------------------
//   Navier Bifurcation: Tracer Particles in a 2D Bifurcating Channel Flow
//   ---------------------------------------------------------------------
//
// Note: MFEM must be compiled with GSLIB for this miniapp to include
//       particles - otherwise, it will just compute the fluid flow.
//
// This miniapp demonstrates the usage of tracer particles (i.e. one-way
// coupled) in fluid flow. The fluid flow is computed with NavierSolver and
// the particles are advected with NavierParticles. Particles are injected
// periodically at random locations along the inlet boundary. Reflection
// boundary conditions are used on the channel walls.
//
// Sample run:
// * mpirun -np 10 navier_bifurcation -rs 3 -npt 100 -nt 4e5 -traj 10


#include "navier_solver.hpp"
#include "navier_particles.hpp"
#include "../../common/pfem_extras.hpp"
#include "../../common/particles_extras.hpp"
#include "../../../general/text.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace navier;
using namespace mfem::common;

struct flow_context
{
   // common
   real_t dt = 1e-3;
   real_t nt = 10000;

   // fluid
   int rs_levels = 3;
   int order = 4;
   real_t Re = 1000;             // Reynolds number
   int paraview_freq = 500;      // frequency of ParaView output

   // particle
   int add_particles_freq = 300; // frequency of particle injection
   int num_add_particles = 100;  // total particles added each injection
   real_t kappa_min = 1.0;       // drag property min
   real_t kappa_max = 10.0;      // drag property max
   real_t gamma = 0.0;           // gravity property
   real_t zeta = 0.19;           // lift property
   int print_csv_freq = 500;     // frequency of particle CSV outputting

   // GLVis visualization for solution and (optionally) particles
   bool visualization = true;
   // Particle trajectory visualization [visualization must be set to true]
   // traj_len_update_freq = 0 means no trajectory visualization.
   // traj_len_update_freq > 0 means update trajectory every
   // traj_len_update_freq time steps. This is useful for long simulations
   // with small time-steps.
   int traj_len_update_freq = 0;
} ctx;

#ifdef MFEM_USE_GSLIB
// Set properties for injected particles. The location is set randomly to be
// on the inlet boundary (x=0), velocity is initialized to 0, and
// kappa (drag property) is set randomly in [kappa_min, kappa_max]
// using kappa_seed.
void SetInjectedParticles(NavierParticles &particle_solver,
                          const Array<int> &p_idxs,
                          real_t kappa_min, real_t kappa_max, int kappa_seed,
                          real_t zeta, real_t gamma, int step);
#endif

// Dirichlet conditions for velocity
void vel_dbc(const Vector &x, real_t t, Vector &u);

int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int rank = Mpi::WorldRank();
   Hypre::Init();

   // Parse command line arguments
   OptionsParser args(argc, argv);
   args.AddOption(&ctx.dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&ctx.nt, "-nt", "--num-timesteps", "Number of time steps.");
   args.AddOption(&ctx.rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh in serial.");
   args.AddOption(&ctx.order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ctx.Re, "-Re", "--reynolds-number", "Reynolds number.");
   args.AddOption(&ctx.paraview_freq, "-pv", "--paraview-freq",
                  "ParaView data collection write frequency. 0 to disable.");
   args.AddOption(&ctx.visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&ctx.add_particles_freq, "-ipf", "--inject-particles-freq",
                  "Frequency of particle injection at domain inlet.");
   args.AddOption(&ctx.num_add_particles, "-npt", "--num-particles-inject",
                  "Number of particles to add each injection.");
   args.AddOption(&ctx.kappa_min, "-kmin", "--kappa-min",
                  "Kappa constant minimum.");
   args.AddOption(&ctx.kappa_max, "-kmax", "--kappa-max",
                  "Kappa constant maximum.");
   args.AddOption(&ctx.gamma, "-g", "--gamma", "Gamma constant.");
   args.AddOption(&ctx.zeta, "-z", "--zeta", "Zeta constant.");
   args.AddOption(&ctx.print_csv_freq, "-csv", "--csv-freq",
                  "Frequency of particle CSV outputting. 0 to disable.");
   args.AddOption(&ctx.traj_len_update_freq, "-traj", "--traj-freq",
                  "Frequency of particle trajectory update. 0 to disable.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (rank == 0)
   {
      args.PrintOptions(cout);
   }

   // Load mesh + complete any serial refinements
   Mesh mesh("../../../data/channel-bifurcation-2d.mesh");
   for (int lev = 0; lev < ctx.rs_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // Parallel decompose mesh
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   pmesh.EnsureNodes();
   mesh.Clear();

   // Create the flow solver
   NavierSolver flow_solver(&pmesh, ctx.order, 1.0/ctx.Re);
   flow_solver.EnablePA(true);

   real_t time = 0.0;

   // Initialize fluid IC
   VectorFunctionCoefficient u_excoeff(2, vel_dbc);
   ParGridFunction &u_gf = *flow_solver.GetCurrentVelocity();
   u_excoeff.SetTime(time);

   // Set fluid BCs
   Array<int> attr(pmesh.bdr_attributes.Max());
   attr = 0;
   // Inlet is attribute 1.
   attr[0] = 1;
   // Walls is attribute 2.
   attr[1] = 1;
   flow_solver.AddVelDirichletBC(vel_dbc, attr);

#ifdef MFEM_USE_GSLIB
   ParGridFunction &w_gf = *flow_solver.GetCurrentVorticity();
   // Create the particle solver
   NavierParticles particle_solver(MPI_COMM_WORLD, 0, pmesh);
   int nparticles = (ctx.nt/ctx.add_particles_freq) *
                    ctx.num_add_particles / Mpi::WorldSize();
   particle_solver.GetParticles().Reserve(nparticles);

   // Set particle BCs - left normal for line connecting start to end must
   // point into the domain. If not, we set invert_normal to true.
   particle_solver.Add2DReflectionBC(Vector({0.0, 1.0}), Vector({8.0, 1.0}),
                                     1.0, true);
   particle_solver.Add2DReflectionBC(Vector({8.0, 1.0}), Vector({8.0, 9.0}),
                                     1.0, true);
   particle_solver.Add2DReflectionBC(Vector({9.0, 9.0}), Vector({9.0, 1.0}),
                                     1.0, true);
   particle_solver.Add2DReflectionBC(Vector({9.0, 1.0}), Vector({17.0, 1.0}),
                                     1.0, true);
   particle_solver.Add2DReflectionBC(Vector({0.0, 0.0}), Vector({17.0, 0.0}),
                                     1.0, false);
#endif

   // Set up solution and particle visualization
   char vishost[] = "localhost";
   int visport = 19916;
   socketstream vis_sol;
   int Ww = 500, Wh = 500; // window size
   int Wx = 10, Wy = 0; // window position
   char keys[] = "mAcRjlmm]]]]]]]]]";
   std::unique_ptr<ParticleTrajectories> traj_vis;
   // Extract boundary mesh for particle visualization
   int nattr = pmesh.bdr_attributes.Max();
   Array<int> subdomain_attributes(nattr);
   for (int i = 0; i < nattr; i++)
   {
      subdomain_attributes[i] = i+1;
   }
   auto psubmesh = std::unique_ptr<ParMesh>(new ParMesh(
                                               ParSubMesh::CreateFromBoundary(pmesh, subdomain_attributes)));

   if (ctx.visualization)
   {
      VisualizeField(vis_sol, vishost, visport, u_gf, "Velocity",
                     Wx, Wy, Ww, Wh, keys);
#ifdef MFEM_USE_GSLIB
      if (ctx.traj_len_update_freq > 0)
      {
         int traj_length = 4; // length of trajectory in number of segments
         traj_vis = std::make_unique<ParticleTrajectories>(
                       particle_solver.GetParticles(),
                       traj_length, vishost, visport,
                       "Particle Trajectories",
                       Ww+Wx, Wy, Ww, Wh, "bbm");
         traj_vis->AddMeshForVisualization(psubmesh.get());
         traj_vis->Visualize();
      }
#endif
   }

   // Initialize ParaView DC (if freq != 0)
   std::unique_ptr<ParaViewDataCollection> pvdc;
   if (ctx.paraview_freq > 0)
   {
      pvdc = std::make_unique<ParaViewDataCollection>("Bifurcation", &pmesh);
      pvdc->SetPrefixPath("ParaView");
      pvdc->SetLevelsOfDetail(ctx.order);
      pvdc->SetDataFormat(VTKFormat::BINARY);
      pvdc->SetHighOrderOutput(true);
      pvdc->RegisterField("Velocity",flow_solver.GetCurrentVelocity());
      pvdc->RegisterField("Vorticity",flow_solver.GetCurrentVorticity());
      pvdc->SetTime(time);
      pvdc->SetCycle(0);
      pvdc->Save();
   }

#ifdef MFEM_USE_GSLIB
   std::string csv_prefix = "Navier_Bifurcation_";
   // Setup arrays to indicate what to print.
   // Particle id, rank, and coordinates are included by default. We also
   // include the first field (i.e. \kappa with index 0 inside the ParticleSet)
   Array<int> print_field_idxs({0});
   // Print the first tag for each particle.
   Array<int> print_tag_idxs({0});
   if (ctx.print_csv_freq > 0)
   {
      std::string file_name = csv_prefix +
                              mfem::to_padded_string(0, 6) + ".csv";
      particle_solver.GetParticles().PrintCSV(file_name.c_str(),
                                              print_field_idxs,
                                              print_tag_idxs);
   }
   particle_solver.Setup(ctx.dt);
#endif

   flow_solver.Setup(ctx.dt);

   u_gf.ProjectCoefficient(u_excoeff);
   int vis_count = 1;

   Array<int> add_particle_idxs;
   real_t cfl;
   for (int step = 1; step <= ctx.nt; step++)
   {
      flow_solver.Step(time, ctx.dt, step-1);

#ifdef MFEM_USE_GSLIB
      // Inject particles at inlet and initialize their properties
      if (step % ctx.add_particles_freq == 0)
      {
         int size = Mpi::WorldSize();
         int rank_num_particles = ctx.num_add_particles/size +
                                  (rank < (ctx.num_add_particles % size) ?
                                   1 : 0);
         // Add particles to the ParticleSet
         particle_solver.GetParticles().AddParticles(rank_num_particles,
                                                     &add_particle_idxs);
         // Initialize properties of the new particles
         SetInjectedParticles(particle_solver, add_particle_idxs,
                              ctx.kappa_min, ctx.kappa_max,
                              (rank+1)*step, ctx.zeta, ctx.gamma, step);
      }
      // Step the particles
      particle_solver.Step(ctx.dt, u_gf, w_gf);

      // Output particle data to csv
      if (ctx.print_csv_freq > 0 && step % ctx.print_csv_freq == 0)
      {
         vis_count++;
         std::string file_name = csv_prefix +
                                 mfem::to_padded_string(step, 6) + ".csv";
         particle_solver.GetParticles().PrintCSV(file_name.c_str(),
                                                 print_field_idxs,
                                                 print_tag_idxs);
      }
#endif

      // Output flow data for ParaView
      if (ctx.paraview_freq > 0 && step % ctx.paraview_freq == 0)
      {
         pvdc->SetTime(vis_count);
         pvdc->SetCycle(step);
         pvdc->Save();
      }

      // GLVis visualization
      if (ctx.visualization)
      {
         VisualizeField(vis_sol, vishost, visport, u_gf,
                        "Velocity", Wx, Wy, Ww, Wh, keys);
#ifdef MFEM_USE_GSLIB
         if (ctx.traj_len_update_freq > 0 &&
             step % ctx.traj_len_update_freq == 0)
         {
            traj_vis->Visualize();
         }
#endif
      }

      cfl = flow_solver.ComputeCFL(u_gf, ctx.dt);
#ifdef MFEM_USE_GSLIB
      auto global_np =  particle_solver.GetParticles().GetGlobalNParticles();
      auto inactive_global_np =
         particle_solver.GetInactiveParticles().GetGlobalNParticles();
#endif
      if (rank == 0)
      {
         printf("\n%-11s %-11s %-11s %-11s\n", "Step", "Time", "dt", "CFL");
         printf("%-11i %-11.5E %-11.5E %-11.5E\n", step, time, ctx.dt, cfl);
#ifdef MFEM_USE_GSLIB
         printf("\n%16s: %-9llu\n", "Active Particles", global_np);
         printf("%16s: %-9llu\n", "Lost Particles", inactive_global_np);
#endif
         printf("-----------------------------------------------\n");
         fflush(stdout);
      }
   }

   flow_solver.PrintTimingData();

   return 0;
}

#ifdef MFEM_USE_GSLIB
void SetInjectedParticles(NavierParticles &particle_solver,
                          const Array<int> &p_idxs, real_t kappa_min,
                          real_t kappa_max, int kappa_seed,
                          real_t zeta, real_t gamma, int step)
{
   // Set initial conditions for new particles.
   MPI_Comm comm = particle_solver.GetParticles().GetComm();
   int my_rank;
   MPI_Comm_rank(comm, &my_rank);
   Vector rand_init_yloc(p_idxs.Size());
   rand_init_yloc.Randomize(my_rank + step);
   for (int i = 0; i < p_idxs.Size(); i++)
   {
      int idx = p_idxs[i];

      for (int j = 0; j < 4; j++)
      {
         if (j == 0)
         {
            // Set position randomly along inlet
            real_t yval = rand_init_yloc(i);
            particle_solver.X().SetValues(idx, Vector({0.0, yval}));
         }
         else
         {
            // Zero-out position history
            particle_solver.X(j).SetValues(idx, Vector({0.0,0.0}));
         }

         // Zero-out particle velocities, fluid velocities, and fluid vorticities
         particle_solver.V(j).SetValues(idx, Vector({0.0,0.0}));
         particle_solver.U(j).SetValues(idx, Vector({0.0,0.0}));
         particle_solver.W(j).SetValues(idx, Vector({0.0,0.0}));

         // Set Kappa, Zeta, Gamma
         std::mt19937 gen(kappa_seed);
         std::uniform_real_distribution<> real_dist(0.0,1.0);
         particle_solver.Kappa()[idx] = kappa_min + real_dist(gen)*
                                        (kappa_max - kappa_min);
         particle_solver.Zeta()[idx] = zeta;
         particle_solver.Gamma()[idx] = gamma;
      }

      // Set order to 0
      particle_solver.Order()[idx] = 0;
   }
}
#endif

// Dirichlet conditions for velocity
void vel_dbc(const Vector &x, real_t t, Vector &u)
{
   real_t yi = x(1);
   real_t height = 1.0;
   u(0) = 0.;
   u(1) = 0.;
   if (std::fabs(yi)<1.0) { u(0) = 6.0*yi*(height-yi)/(height*height); }
}
