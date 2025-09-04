// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// Sample run: mpirun -np 12 ./navier_junction

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
   int rs_levels = 3;
   int order = 3;
   real_t Re = 100;
   int paraview_freq = 10;
   bool visualization = true;
   int visport = 19916;

   // particle
   int pnt_0 = round(0.1/dt);
   int add_particles_freq = 100;
   int num_add_particles = 50;
   real_t kappa_bottom = 10.0;
   real_t kappa_top = 20.0;
   real_t gamma = 0.0;
   real_t zeta = 0.19;
   int print_csv_freq = 10;
} ctx;

void SetInjectedParticles(NavierParticles &particle_solver, const Array<int> &p_idxs, const Vector &inlet_start, const Vector &inlet_end, real_t kappa, real_t zeta, real_t gamma, int step);

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
   args.AddOption(&ctx.pnt_0, "-pt", "--particle-timestep", "Timestep to begin integrating particles.");
   args.AddOption(&ctx.add_particles_freq, "-ipf", "--inject-particles-freq", "Frequency of particle injection at domain inlet.");
   args.AddOption(&ctx.num_add_particles, "-npt", "--num-particles-inject", "Number of particles to add each injection.");
   args.AddOption(&ctx.kappa_bottom, "-kbot", "--kappa-bottom", "Kappa constant for bottom.");
   args.AddOption(&ctx.kappa_top, "-ktop", "--kappa-top", "Kappa constant for top.");
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
   Mesh mesh("../../data/T_junction.mesh");
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
   NavierParticles particle_solver(MPI_COMM_WORLD, 0, pmesh);
   particle_solver.GetParticles().Reserve( ((ctx.nt - ctx.pnt_0)/ctx.add_particles_freq ) * ctx.num_add_particles / size);

   real_t time = 0.0;

   // Initialize fluid IC
   VectorFunctionCoefficient u_excoeff(2, vel_dbc);
   ParGridFunction &u_gf = *flow_solver.GetCurrentVelocity();
   ParGridFunction &w_gf = *flow_solver.GetCurrentVorticity();
   u_excoeff.SetTime(time);

   // Set fluid BCs
   Array<int> attr(pmesh.bdr_attributes.Max());
   attr[0] = 1; // Top inlet
   attr[1] = 1; // Bottom inlet
   attr[2] = 0; // Outlet
   attr[3] = 1; // Walls
   flow_solver.AddVelDirichletBC(vel_dbc, attr);


   // Set particle BCs
   particle_solver.Add2DReflectionBC(Vector({0.0, 0.0}), Vector({0.0, 1.0}), 1.0, true);
   particle_solver.Add2DReflectionBC(Vector({0.25, 0.0}), Vector({0.25, 0.375}), 1.0, false);
   particle_solver.Add2DReflectionBC(Vector({0.25, 0.375}), Vector({1.25, 0.375}), 1.0, false);
   particle_solver.Add2DReflectionBC(Vector({0.25, 0.625}), Vector({1.25, 0.625}), 1.0, true);
   particle_solver.Add2DReflectionBC(Vector({0.25, 0.625}), Vector({0.25, 1.0}), 1.0, false);

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
      pvdc = std::make_unique<ParaViewDataCollection>("Junction", &pmesh);
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

   std::string csv_prefix = "Navier_Junction_";
   if (ctx.print_csv_freq > 0)
   {
      std::string file_name = csv_prefix + mfem::to_padded_string(0, 6) + ".csv";
      Array<int> field_idx, tag_idx;
      particle_solver.GetParticles().PrintCSV(file_name.c_str(), field_idx, tag_idx);
   }
   int vis_count = 0;

   flow_solver.Setup(ctx.dt);
   particle_solver.Setup(ctx.dt);

   u_gf = 0.0;
   int pstep = 0;
   Array<int> add_particle_idxs;
   for (int step = 1; step <= ctx.nt; step++)
   {
      real_t cfl;
      flow_solver.Step(time, ctx.dt, step-1);

      // Step particles after pnt_0
      if (step >= ctx.pnt_0)
      {
         // Inject particles
         if (step % ctx.add_particles_freq == 0)
         {
            int rank_num_particles = ctx.num_add_particles/size + (rank < (ctx.num_add_particles % size) ? 1 : 0);
            particle_solver.GetParticles().AddParticles(rank_num_particles, &add_particle_idxs);
            SetInjectedParticles(particle_solver, add_particle_idxs, Vector({0.0, 0.0}), Vector({0.25, 0.0}), ctx.kappa_bottom, ctx.zeta, ctx.gamma, step*rank);
            particle_solver.GetParticles().AddParticles(rank_num_particles, &add_particle_idxs);
            SetInjectedParticles(particle_solver, add_particle_idxs, Vector({0.0, 1.0}), Vector({0.25, 1.0}), ctx.kappa_top, ctx.zeta, ctx.gamma, step*rank);
         }

         particle_solver.Step(ctx.dt, u_gf, w_gf);
         pstep++;
      }

      if (ctx.print_csv_freq > 0 && step % ctx.print_csv_freq == 0)
      {
         // Output the particles
         std::string file_name = csv_prefix + mfem::to_padded_string(step, 6) + ".csv";
         Array<int> field_idx({0}), tag_idx;
         particle_solver.GetParticles().PrintCSV(file_name.c_str(), field_idx, tag_idx);
         // particle_solver.GetParticles().PrintCSV(file_name.c_str());
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
      int global_np = particle_solver.GetParticles().GetGlobalNP();
      int inactive_global_np = particle_solver.GetInactiveParticles().GetGlobalNP();
      if (rank == 0)
      {
         printf("\n%-11s %-11s %-11s %-11s\n", "Step", "Time", "dt", "CFL");
         printf("%-11i %-11.5E %-11.5E %-11.5E\n", step, time, ctx.dt, cfl);
         printf("\n%16s: %-9i\n", "Active Particles", global_np);
         printf("%16s: %-9i\n", "Lost Particles", inactive_global_np);
         printf("-----------------------------------------------\n");
         fflush(stdout);
      }
   }

   flow_solver.PrintTimingData();

   return 0;
}

void SetInjectedParticles(NavierParticles &particle_solver, const Array<int> &p_idxs, const Vector &inlet_start, const Vector &inlet_end, real_t kappa, real_t zeta, real_t gamma, int seed)
{
    // Inject randomly along inlet
   real_t length = inlet_start.DistanceTo(inlet_end);

   Vector tan(2);
   tan = inlet_end;
   tan -= inlet_start;
   tan /= length;

   MPI_Comm comm = particle_solver.GetParticles().GetComm();

   int rank_num_add = p_idxs.Size();
   int global_num_particles = 0;
   MPI_Allreduce(&rank_num_add, &global_num_particles, 1, MPI_INT, MPI_SUM, comm);

   std::uniform_real_distribution<> real_dist(0.0, length);
   std::mt19937 gen(seed);

   Vector pos(2);
   for (int i = 0; i < p_idxs.Size(); i++)
   {
      int idx = p_idxs[i];

      for (int j = 0; j < 4; j++)
      {
         if (j == 0)
         {
            // Set position
            pos = 0.0;
            add(inlet_start, real_dist(gen), tan, pos);
            particle_solver.X().SetVectorValues(idx, pos);
         }
         else
         {
            // Zero-out position history
            particle_solver.X(j).SetVectorValues(idx, Vector({0.0,0.0}));
         }

         // Zero-out particle velocities, fluid velocities, and fluid vorticities
         particle_solver.V(j).SetVectorValues(idx, Vector({0.0,0.0}));
         particle_solver.U(j).SetVectorValues(idx, Vector({0.0,0.0}));
         particle_solver.W(j).SetVectorValues(idx, Vector({0.0,0.0}));

         // Set Kappa, Zeta, Gamma
         particle_solver.Kappa()[idx] = kappa;
         particle_solver.Zeta()[idx] = zeta;
         particle_solver.Gamma()[idx] = gamma;
      }

      // Set order to 0
      particle_solver.Order()[idx] = 0;
   }

}

// Dirichlet conditions for velocity
void vel_dbc(const Vector &x, real_t t, Vector &u)
{
   real_t xi = x(0);
   real_t yi = x(1);
   real_t length = 0.25;

   u(0) = 0.;
   u(1) = 0.;
   if (std::fabs(yi)<1e-5)
   {
      u(1) = 6.0*xi*(length-xi)/(length*length);
   }
   else if (std::fabs(yi)>0.99999)
   {
      u(1) = -6.0*xi*(length-xi)/(length*length);
   }
}