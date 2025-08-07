// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// Sample run:
//   mpirun -np 4 navier_bifurcation -rs 2

#include "navier_solver.hpp"
#include "navier_particles.hpp"
#include "../common/pfem_extras.hpp"
#include "../../general/text.hpp"
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
   int rs_levels = 0;
   int order = 4;
   real_t Re = 1000;
   int paraview_freq = 500;
   bool visualization = true;
   int visport = 19916;

   // particle
   real_t t0p = 5.0;
   int num_particles = 1000;
   real_t kappa = 10.0;
   real_t gamma = 0.0; //0.2; // should be 6
   real_t zeta = 0.19;
   int print_csv_freq = 500;
} ctx;

// Dirichlet conditions for velocity
void vel_dbc(const Vector &x, real_t t, Vector &u);

int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int size = Mpi::WorldSize();
   int rank = Mpi::WorldRank();
   Hypre::Init();

   // Parse command line arguments
   OptionsParser args(argc, argv);
   args.AddOption(&ctx.dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&ctx.nt, "-nt", "--num-timesteps", "Number of time steps.");
   args.AddOption(&ctx.rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh in serial.");
   args.AddOption(&ctx.order, "-o", "--order", "Order (degree) of the finite elements.");
   args.AddOption(&ctx.Re, "-Re", "--reynolds-number", "Reynolds number.");
   args.AddOption(&ctx.paraview_freq, "-pv", "--paraview-freq", "ParaView data collection write frequency. 0 to disable.");
   args.AddOption(&ctx.visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&ctx.visport, "-p", "--send-port", "Socket for GLVis.");
   args.AddOption(&ctx.t0p, "-t0p", "--particle-time", "Time to begin integrating particles.");
   args.AddOption(&ctx.num_particles, "-np", "--num-particles", "Number of particles to initialize on the domain.");
   args.AddOption(&ctx.kappa, "-k", "--kappa", "Kappa constant.");
   args.AddOption(&ctx.gamma, "-g", "--gamma", "Gamma constant.");
   args.AddOption(&ctx.zeta, "-z", "--zeta", "Zeta constant.");
   args.AddOption(&ctx.print_csv_freq, "-csv", "--csv-freq", "Frequency of particle CSV outputting. 0 to disable.");
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
   Mesh mesh("../../data/channel2.mesh");
   int dim = mesh.Dimension();
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
   
   // Create the particle solver
   NavierParticles particle_solver(MPI_COMM_WORLD, ctx.kappa, ctx.gamma, ctx.zeta, ctx.num_particles, pmesh);

   // Initialize particles at the channel inlet randomly
   Vector pos_min(2), pos_max(2);
   pos_min[0] = 0.0;
   pos_max[0] = 4.0;
   pos_min[1] = 0.05;
   pos_max[1] = 0.95;
   std::uniform_real_distribution<> real_dist(0.0,1.0);
   for (int i = 0; i < particle_solver.GetParticles().GetNP(); i++)
   {
      std::mt19937 gen(particle_solver.GetParticles().GetIDs()[i]);
      
      for (int d = 0; d < 2; d++)
      {
         particle_solver.X()(i, d) = pos_min[d] + real_dist(gen)*(pos_max[d] - pos_min[d]);
      }
   }

   real_t time = 0.0;

   // Initialize fluid IC
   VectorFunctionCoefficient u_excoeff(2, vel_dbc);
   ParGridFunction &u_gf = *flow_solver.GetCurrentVelocity();
   ParGridFunction &w_gf = *flow_solver.GetCurrentVorticity();
   u_excoeff.SetTime(time);

   // Set fluid BCs
   Array<int> attr(pmesh.bdr_attributes.Max());
   attr = 0;
   // Inlet is attribute 1.
   attr[0] = 1;
   // Walls is attribute 2.
   attr[1] = 1;
   flow_solver.AddVelDirichletBC(vel_dbc, attr);

   // Set up solution visualization
   char vishost[] = "localhost";
   socketstream vis_sol;
   int Ww = 350, Wh = 350; // window size
   int Wx = 10, Wy = 0; // window position
   char keys[] = "mAcRjlmm]]]]]]]]]";
   if (ctx.visualization)
   {
      VisualizeField(vis_sol, vishost, ctx.visport, u_gf, "Velocity", Wx, Wy, Ww, Wh, keys);
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

   std::string csv_prefix = "Navier_Bifurcation_";
   if (ctx.print_csv_freq > 0)
   {
      std::string file_name = csv_prefix + mfem::to_padded_string(0, 9) + ".csv";
      particle_solver.GetParticles().PrintCSV(file_name.c_str());
   }

   flow_solver.Setup(ctx.dt);
   u_gf.ProjectCoefficient(u_excoeff);
   int pstep = 0;
   for (int step = 1; step <= ctx.nt; step++)
   {
      real_t cfl;
      flow_solver.Step(time, ctx.dt, step-1);

      // Step particles after t0p
      if (time >= ctx.t0p)
      {
         particle_solver.Step(ctx.dt, pstep, u_gf, w_gf);
         pstep++;
         // particle_solver.Apply2DReflectionBC(Vector({0.0, 1.0}), Vector({8.0, 1.0}), 1.0, true);
         // particle_solver.Apply2DReflectionBC(Vector({8.0, 1.0}), Vector({8.0, 9.0}), 1.0, true);
         // particle_solver.Apply2DReflectionBC(Vector({9.0, 9.0}), Vector({9.0, 1.0}), 1.0, true);
         // particle_solver.Apply2DReflectionBC(Vector({9.0, 1.0}), Vector({17.0, 1.0}), 1.0, true);
         // particle_solver.Apply2DReflectionBC(Vector({0.0, 0.0}), Vector({17.0, 0.0}), 1.0, false);
      }

      if (ctx.print_csv_freq > 0 && step % ctx.print_csv_freq == 0)
      {
         // Output the particles
         std::string file_name = csv_prefix + mfem::to_padded_string(step, 9) + ".csv";
         particle_solver.GetParticles().PrintCSV(file_name.c_str());
      }

      if (ctx.paraview_freq > 0 && step % ctx.paraview_freq == 0)
      {
         pvdc->SetTime(time);
         pvdc->SetCycle(step);
         pvdc->Save();
      }

      if (ctx.visualization)
      {
         VisualizeField(vis_sol, vishost, ctx.visport, u_gf,
                        "Velocity", Wx, Wy, Ww, Wh, keys);
      }

      cfl = flow_solver.ComputeCFL(u_gf, ctx.dt);
      if (rank == 0)
      {
         printf("\n%-11s %-11s %-11s %-11s\n", "Step", "Time", "dt", "CFL");
         printf("%-11i %-11.5E %-11.5E %-11.5E\n", step, time, ctx.dt, cfl);
         printf("-----------------------------------------------\n");
         fflush(stdout);
      }
   }

   flow_solver.PrintTimingData();

   return 0;
}

// Dirichlet conditions for velocity
void vel_dbc(const Vector &x, real_t t, Vector &u)
{
   real_t xi = x(0);
   real_t yi = x(1);
   real_t height = 1.0;

   u(0) = 0.;
   u(1) = 0.;
   if (std::fabs(yi)<1.0) { u(0) = 6.0*yi*(height-yi)/(height*height); }
}