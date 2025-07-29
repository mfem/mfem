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


#include "navier_solver.hpp"
#include "navier_particles.hpp"
#include "../../general/text.hpp"

#include <cmath>
#include <random>

using namespace std;
using namespace mfem;
using namespace navier;

struct s_Context
{
   int order = 3;
   real_t dt = 0.01;
   int num_steps = 1000;
   int num_particles = 1000;
   real_t kappa = 1.0;
   real_t gamma = 1.0;
   real_t zeta = 4.0;
   Vector x_min{-1.0,-1.0};
   Vector x_max{1.0,1.0};
   Vector v_min{-1.0,-1.0};
   Vector v_max{1.0,1.0};
   int test = -1;
   int paraview_freq = 0;
   int print_csv_freq = 0;
} ctx;

void couetteFlow(const Vector &x, real_t t, Vector &u)
{
   u.SetSize(2);
   u = 0.0;
   u[0] = (1.0+x[1])/2.0;
}

struct AnalyticalParticles
{
   const real_t kappa, gamma, zeta;
   int test;
   ParticleSet particles;

   AnalyticalParticles(MPI_Comm comm, const real_t kappa_, const real_t gamma_, const real_t zeta_, int NP, int testID);
   void OnlyLift(const Vector &x0, const Vector &v0, double t, Vector &x, Vector &v);
   void NoLift(const Vector &x0, const Vector &v0, double t, Vector &x, Vector &v);
   void SetTime(real_t time);

   ParticleVector&  X() { return particles.Coords(); }
   ParticleVector& X0() { return particles.Field(0); }
   ParticleVector& V0() { return particles.Field(1); }
   ParticleVector&  V() { return particles.Field(2); }
};

int main (int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int size = Mpi::WorldSize();
   int rank = Mpi::WorldRank();
   Hypre::Init();

   OptionsParser args(argc, argv);
   args.AddOption(&ctx.order, "-o", "--order", "Order (degree) of the finite elements.");
   args.AddOption(&ctx.dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&ctx.num_steps, "-nt", "--num-timesteps", "Number of time steps to take.");
   args.AddOption(&ctx.num_particles, "-np", "--num-particles", "Number of particles to initialize on the domain.");
   args.AddOption(&ctx.kappa, "-k", "--kappa", "Kappa constant.");
   args.AddOption(&ctx.gamma, "-g", "--gamma", "Gamma constant.");
   args.AddOption(&ctx.zeta, "-z", "--zeta", "Zeta constant.");
   args.AddOption(&ctx.x_min, "-xmin", "--x-min", "Minimum initial particle location.");
   args.AddOption(&ctx.x_max, "-xmax", "--x-max", "Maximum initial particle location.");
   args.AddOption(&ctx.v_min, "-vmin", "--v-min", "Minimum initial particle velocity.");
   args.AddOption(&ctx.v_max, "-vmax", "--v-max", "Maximum initial particle velocity.");
   args.AddOption(&ctx.test, "-t", "--test", "Run test against analytical w/ exact solution outputting. -1 to disable. 0 for only-lift test. 1 for no-lift (drag + gravity) test.");
   args.AddOption(&ctx.paraview_freq, "-pv", "--paraview-freq", "Frequency of ParaView flow output. 0 to disable.");
   args.AddOption(&ctx.print_csv_freq, "-csv", "--csv-freq", "Frequency of particle CSV outputting. 0 to disable.");
   args.Parse();
   if (!args.Good())
   {
      if (rank == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (rank == 0)
   {
      args.PrintOptions(cout);
   }

   if (ctx.test == 0)
   {
      ctx.kappa = 0.0;
      ctx.gamma = 0.0;
      if (ctx.zeta <= 1.0)
      {
         ctx.zeta = 4.0;
      }
   }
   else if (ctx.test == 1)
   {
      ctx.zeta = 0.0;
      if (ctx.kappa <= 0.0)
      {
         ctx.kappa = 1.0;
      }
      if (ctx.gamma <= 0.0)
      {
         ctx.gamma = 1.0;
      }
   }

   // Initialize a simple straight-edged 2D domain [-10,10] x [-10,10]
   Mesh mesh = Mesh::MakeCartesian2D(20, 20, Element::Type::QUADRILATERAL, true, 20, 20);
   Vector transl(mesh.GetNV()*2);
   transl = -10.0; // translate down + left 10
   mesh.MoveNodes(transl);

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   pmesh.EnsureNodes();

   // Initialize the Navier solver
   NavierSolver flow_solver(&pmesh, ctx.order, 1.0);

   // Initialize NavierParticles
   NavierParticles particle_solver(MPI_COMM_WORLD, ctx.kappa, ctx.gamma, ctx.zeta, ctx.num_particles, pmesh);

   // If running test, initialize analytical particles
   std::unique_ptr<AnalyticalParticles> particles_exact;
   if (ctx.test != -1)
   {
      particles_exact = std::make_unique<AnalyticalParticles>(MPI_COMM_WORLD, ctx.kappa, ctx.gamma, ctx.zeta, ctx.num_particles, ctx.test);
   }

   // Initialize particles
   std::mt19937 gen(rank);
   std::uniform_real_distribution<> real_dist(0.0,1.0);
   for (int i = 0; i < particle_solver.GetParticles().GetNP(); i++)
   {
      for (int d = 0; d < 2; d++)
      {
         particle_solver.X().ParticleValue(i, d) = ctx.x_min[d] + real_dist(gen)*(ctx.x_max[d] - ctx.x_min[d]);
         particle_solver.V().ParticleValue(i, d) = ctx.v_min[d] + real_dist(gen)*(ctx.v_max[d] - ctx.v_min[d]);
      }
   }
   if (ctx.test != -1)
   {
      particles_exact->X() = particle_solver.X().GetData();
      particles_exact->X0() = particle_solver.X().GetData(); // x0
      particles_exact->V0() = particle_solver.V().GetData(); // v0
      particles_exact->V() = particle_solver.V().GetData(); // vn
   }

   real_t time = 0.0;

   // Initialize fluid IC
   VectorFunctionCoefficient u_excoeff(2, couetteFlow);
   ParGridFunction &u_gf = *flow_solver.GetCurrentVelocity();
   ParGridFunction &w_gf = *flow_solver.GetCurrentVorticity();
   u_excoeff.SetTime(time);
   u_gf.ProjectCoefficient(u_excoeff);
   w_gf = -0.5;

   // Set initial particle-interpolated fluid velocity and vorticity
   particle_solver.InterpolateUW(u_gf, w_gf, particle_solver.X(), particle_solver.U(), particle_solver.W());

   // Set fluid BCs
   Array<int> bdr_attr(pmesh.bdr_attributes.Max());
   bdr_attr = 1;
   flow_solver.AddVelDirichletBC(couetteFlow, bdr_attr);

   // Initialize ParaView DC (if freq != 0)
   std::unique_ptr<ParaViewDataCollection> pvdc;
   if (ctx.paraview_freq > 0)
   {
      pvdc = std::make_unique<ParaViewDataCollection>("Couette", &pmesh);
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
   
   std::string csv_prefix = "Navier_Couette_";
   if (ctx.print_csv_freq > 0)
   {
      std::string file_name = csv_prefix + mfem::to_padded_string(0, 9) + ".csv";
      particle_solver.GetParticles().PrintCSV(file_name.c_str());
      if (ctx.test != -1)
      {
         particles_exact->SetTime(time);
         std::string file_name_exact = csv_prefix + "Exact_" + mfem::to_padded_string(0, 9) + ".csv";
         particles_exact->particles.PrintCSV(file_name_exact.c_str());
      }
   }

   flow_solver.Setup(ctx.dt);
   for (int step = 1; step <= ctx.num_steps; step++)
   {
      // Step Navier
      flow_solver.Step(time, ctx.dt, step-1);

      // Step NavierParticles
      particle_solver.Step(ctx.dt, step-1, u_gf, w_gf);

      if (ctx.print_csv_freq > 0 && step % ctx.print_csv_freq == 0)
      {
         // Output the particles
         std::string file_name = csv_prefix + mfem::to_padded_string(step, 9) + ".csv";
         particle_solver.GetParticles().PrintCSV(file_name.c_str());

         // Set + output the exact
         if (ctx.test != -1)
         {
            particles_exact->SetTime(time);
            std::string file_name_exact = csv_prefix + "Exact_" + mfem::to_padded_string(step, 9) + ".csv";
            particles_exact->particles.PrintCSV(file_name_exact.c_str());
         }
      }
      if (ctx.paraview_freq > 0 && step % ctx.paraview_freq == 0)
      {
         pvdc->SetCycle(step);
         pvdc->SetTime(time);
         pvdc->Save();
      }
   }   
}


AnalyticalParticles::AnalyticalParticles(MPI_Comm comm, const real_t kappa_, const real_t gamma_, const real_t zeta_, int NP, int testID)
: kappa(kappa_), gamma(gamma_), zeta(zeta_),
  test(testID), particles(comm, NP, 2, Array<int>({2,2,2})) // Only need to store x0, v0, vn for exact
{

}

void AnalyticalParticles::OnlyLift(const Vector &x0, const Vector &v0, double t, Vector &x, Vector &v)
{
   MFEM_ASSERT(zeta > 0.0, "Zeta must be greater than one.");

   x.SetSize(2); v.SetSize(2);

   // Lift-only contribution
   const real_t C1_l = v0[1];
   const real_t C2_l = (sqrt(zeta)/2.0) * (x0[1] + 1 - 2*v0[0])/(sqrt(zeta-1));
   const real_t C3_l = x0[1] + 2.0*C2_l/(sqrt(zeta*(zeta-1)));
   const real_t C4_l = 0.5*(C3_l+1);
   const real_t C5_l = x0[0] + (2.0/(zeta-1))*v0[1];
   const real_t lam = zeta*(zeta-1)/4.0;

   x[0] = (zeta/(2.0*lam)) * (-C1_l*cos(sqrt(lam)*t) - C2_l*sin(sqrt(lam)*t)) + C4_l*t + C5_l;
   x[1] = (1.0/sqrt(lam)) * (C1_l*sin(sqrt(lam)*t) - C2_l*cos(sqrt(lam)*t)) + C3_l;
   v[0] = (zeta/(2.0*sqrt(lam))) * ( C1_l*sin(sqrt(lam)*t) - C2_l*cos(sqrt(lam)*t)) + C4_l;
   v[1] = C1_l*cos(sqrt(lam)*t) + C2_l*sin(sqrt(lam)*t);
}

void AnalyticalParticles::NoLift(const Vector &x0, const Vector &v0, double t, Vector &x, Vector &v)
{
   MFEM_ASSERT(kappa > 0.0, "Kappa must be greater than zero.");
   MFEM_ASSERT(gamma > 0.0, "Gamma must be greater than zero.");

   x.SetSize(2); v.SetSize(2);
   
   // Drag- + gravity-only contribution
   const real_t C1_d = v0[1] + gamma/kappa;
   const real_t C2_d = x0[1] + C1_d/kappa;
   const real_t C3_d = v0[0] - ( C2_d + 1 + gamma/pow(kappa,2) )/2.0;
   const real_t C4_d = x0[0] - C1_d/( 2*pow(kappa,2) ) + C3_d/kappa;

   x[0] = -(C3_d/kappa)*exp(-kappa*t) + (C1_d/2.0)*( (kappa*t+1) / (pow(kappa,2)) ) * exp(-kappa*t) - ( gamma/(4*kappa) )*pow(t,2) + ( ( C2_d + 1 + gamma/pow(kappa,2) )/2.0 )*t + C4_d;
   x[1] = -(1.0/kappa)*(C1_d*exp(-kappa*t) + gamma*t) + C2_d;
   v[0] = C3_d*exp(-kappa*t) - (C1_d/2.0)*t*exp(-kappa*t) - (gamma/(2*kappa))*t + ( C2_d + 1 + gamma/pow(kappa,2) )/2.0;
   v[1] = C1_d*exp(-kappa*t)-gamma/kappa;
}

void AnalyticalParticles::SetTime(real_t time)
{
   for (int i = 0; i < particles.GetNP(); i++)
   {
      Particle p_exact = particles.GetParticleRef(i);
      if (ctx.test == 0)
      {
         OnlyLift(p_exact.Field(0), p_exact.Field(1), time, p_exact.Coords(), p_exact.Field(2));
      }
      else if (ctx.test == 1)
      {
         NoLift(p_exact.Field(0), p_exact.Field(1), time, p_exact.Coords(), p_exact.Field(2));
      }
   }
}