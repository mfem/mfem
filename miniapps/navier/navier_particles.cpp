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


#include "navier_solver.hpp"
#include "../common/particles_extras.hpp"
#include "../../general/text.hpp"

#include <cmath>

using namespace std;
using namespace mfem;
using namespace mfem::common;
using namespace navier;

struct s_Context
{
   int order = 3;
   real_t dt = 0.25e-4;
   int num_steps = 1000;
   int num_particles = 1000;
   int p_ordering = Ordering::byNODES;
   real_t kappa = 1.0;
   real_t zeta = 4.0;
   real_t gamma = 1.0;
   Vector f = Vector({0.0,-1.0});
   int test = -1;
   int paraview_freq = 0;
   int print_csv_freq = 0;
   
} ctx;

// Particle State Variables:
enum State : int
{
   U_NP3,
   U_NP2, 
   U_NP1,
   U_N,

   V_NP3,
   V_NP2,
   V_NP1,
   V_N, 

   W_NP3,
   W_NP2, 
   W_NP1,
   W_N,

   X_NP3,
   X_NP2,
   X_NP1,

   SIZE  
};

void couetteFlow(const Vector &x, real_t t, Vector &u)
{
   u.SetSize(2);
   u = 0.0;
   u[0] = (1.0+x[1])/2.0;
}

// Test 0
// Only lift -- no drag nor body force
void analyticalTest0(const real_t zeta, const Vector &x0, const Vector &v0, double t, Vector &x, Vector &v)
{
   x.SetSize(2); v.SetSize(2);

   const real_t C1 = v0[1];
   const real_t C2 = (sqrt(zeta)/2.0) * (x0[1] + 1 - 2*v0[0])/(sqrt(zeta-1));
   const real_t C3 = x0[1] + 2.0*C2/(sqrt(zeta*(zeta-1)));
   const real_t C4 = 0.5*(C3+1);
   const real_t C5 = x0[0] + (2.0/(zeta-1))*v0[1];

   const real_t lam = zeta*(zeta-1)/4.0;

   x[0] = (zeta/(2.0*lam)) * (-C1*cos(sqrt(lam)*t) - C2*sin(sqrt(lam)*t)) + C4*t + C5;
   x[1] = (1.0/sqrt(lam)) * (C1*sin(sqrt(lam)*t) - C2*cos(sqrt(lam)*t)) + C3;
   v[0] = (zeta/(2.0*sqrt(lam))) * ( C1*sin(sqrt(lam)*t) - C2*cos(sqrt(lam)*t)) + C4;
   v[1] = C1*cos(sqrt(lam)*t) + C2*sin(sqrt(lam)*t);
}

class ParticleIntegrator
{
private:
   const int dim;
   const real_t zeta;
   FindPointsGSLIB finder;

   void ParticleStep2D(const real_t &dt, const Array<real_t> &beta, const Array<real_t> &alpha, Particle &p)
   {
      real_t w_n_ext = 0.0;

      // Extrapolate particle vorticity using EXTk (w_n is new vorticity at old particle loc)
      // w_n_ext = alpha1*w_np1 + alpha2*w_np2 + alpha3*w_np3
      for (int j = 0; j < 3; j++)
      {
         w_n_ext += alpha[j]*(p.GetStateVar(W_N-(j+1))[0]);
      }

      // Assemble the 2D matrix B
      DenseMatrix B({{beta[0], zeta*dt*w_n_ext},
                     {-zeta*dt*w_n_ext, beta[0]}});
      

      // Assemble the RHS
      Vector r(2);
      r = 0.0;
      for (int j = 1; j <= 3; j++)
      {
         // Add particle velocity component
         add(r, -beta[j], p.GetStateVar(V_N-j), r);
         
         // Add C
         add(r, dt*alpha[j-1]*w_n_ext, Vector({ zeta*p.GetStateVar(U_N-j)[1], 
                                               -zeta*p.GetStateVar(U_N-j)[0]}), r);
      }

      // Solve for particle velocity
      DenseMatrixInverse B_inv(B);
      B_inv.Mult(r, p.GetStateVar(V_N));

      // Compute updated particle position
      p.GetCoords() = 0.0;
      for (int j = 1; j <= 3; j++)
      {
         add(p.GetCoords(), -beta[j], p.GetStateVar((X_NP1+1)-j), p.GetCoords());
      }
      add(p.GetCoords(), dt, p.GetStateVar(V_N), p.GetCoords());
      p.GetCoords() *= 1.0/beta[0];

   }

public:
   ParticleIntegrator(MPI_Comm comm, const int dim_, const real_t zeta_, Mesh &m)
   : dim(dim_),
     zeta(zeta_),
     finder(comm)
   {
      finder.Setup(m);
   }

   void Step(const real_t &dt, const Array<real_t> &beta, const Array<real_t> &alpha, const ParGridFunction &u_gf, const ParGridFunction &w_gf, ParticleSet &particles)
   {
      // Shift fluid velocity, fluid vorticity, particle velocity, and particle position
      for (int i = 0; i < particles.GetNP()*dim; i++)
      {
         for (int j = 3; j > 0; j--)
         {
            particles.GetAllStateVar(U_N-j)[i] = particles.GetAllStateVar(U_N-j+1)[i];
            particles.GetAllStateVar(V_N-j)[i] = particles.GetAllStateVar(V_N-j+1)[i];
            particles.GetAllStateVar(W_N-j)[i] = particles.GetAllStateVar(W_N-j+1)[i];
            if (j > 1)
            {
               particles.GetAllStateVar((X_NP1+1)-j)[i] = particles.GetAllStateVar((X_NP1+1)-(j-1))[i];
            }
            else
            {
               particles.GetAllStateVar(X_NP1)[i] = particles.GetAllCoords()[i];
            }
         }
      }

      if (dim == 2)
      {
         for (int i = 0; i < particles.GetNP(); i++)
         {
            Particle p(particles.GetMeta());
            particles.GetParticle(i, p);
            ParticleStep2D(dt, beta, alpha, p); // Compute particle velocity + new position
            particles.SetParticle(i, p);
         }
      }

      // Re-interpolate fluid velocity + vorticity onto particles' new location
      finder.FindPoints(particles.GetAllCoords());
      Vector &p_wn = particles.GetAllStateVar(W_N);
      finder.Interpolate(w_gf, p_wn);
      Ordering::Reorder(p_wn, dim, w_gf.ParFESpace()->GetOrdering(), particles.GetOrdering());

      Vector &p_un = particles.GetAllStateVar(U_N);
      finder.Interpolate(u_gf, p_un);
      Ordering::Reorder(p_un, dim, u_gf.ParFESpace()->GetOrdering(), particles.GetOrdering());
   }
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
   args.AddOption(&ctx.num_steps, "-ns", "--num-steps", "Number of time steps to take.");
   args.AddOption(&ctx.num_particles, "-np", "--num-particles", "Number of particles to initialize on the domain.");
   args.AddOption(&ctx.p_ordering, "-ord", "--particle-ordering", "Ordering of Particle vector data. 0 for byNODES, 1 for byVDIM.");
   args.AddOption(&ctx.kappa, "-k", "--kappa", "Kappa constant.");
   args.AddOption(&ctx.zeta, "-z", "--zeta", "Zeta constant.");
   args.AddOption(&ctx.gamma, "-g", "--gamma", "Gamma constant.");
   args.AddOption(&ctx.test, "-t", "--test", "Override settings + run test. -1 to disable. 0 for only-lift, no drag nor body force test. 1 for no-lift terminal velocity test.");
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
      ctx.dt = 0.01;
      ctx.num_steps = 1000;
      ctx.num_particles = 100;
      ctx.kappa = 0.0;
      ctx.gamma = 0.0;
   }
   else if (ctx.test == 1)
   {

   }


   // Initialize a simple straight-edged 2D domain [0,20] x [-5,5]
   Mesh mesh = Mesh::MakeCartesian2D(10, 10, Element::Type::QUADRILATERAL, true, 20.0, 10.0);
   Vector transl(mesh.GetNV()*2);
   // Mesh vertex ordering is byNODES
   for (int i = 0; i < transl.Size()/2; i++)
   {
      transl[mesh.GetNV() + i] = -5.0; // translate down -1
   }
   mesh.MoveNodes(transl);


   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   pmesh.EnsureNodes();

   // Initialize the Navier solver
   const real_t Re = 1.0; // TODO. Unimportant right now.
   if (rank == 0)
   {
      cout << "Reynolds number: " << Re << endl;
   }
   NavierSolver flowsolver(&pmesh, ctx.order, 1.0/Re);

   // Initialize two particle sets - one numerical, one analytical
   Array<int> vdims(State::SIZE);
   vdims = 2;
   ParticleMeta pmeta(2, 2, vdims);
   ParticleSet particles(MPI_COMM_WORLD, pmeta, ctx.p_ordering == 0 ? Ordering::byNODES : Ordering::byVDIM);

   std::unique_ptr<ParticleMeta> pmeta_exact;
   std::unique_ptr<ParticleSet> particles_exact;
   if (ctx.test != -1)
   {
      // Only need to store x0, v0, vn for exact
      pmeta_exact = std::make_unique<ParticleMeta>(2, 0, Array<int>({2,2,2}));
      particles_exact = std::make_unique<ParticleSet>(MPI_COMM_WORLD, *pmeta_exact, ctx.p_ordering == 0 ? Ordering::byNODES : Ordering::byVDIM);
   }

   // Initialize both particle sets the same way
   Vector pos_min(2), pos_max(2);
   //mesh.GetBoundingBox(pos_min, pos_max);
   pos_min[0] = 3.0;
   pos_max[0] = 4.0;
   pos_min[1] = -0.8;
   pos_max[1] = 0.2;
   int seed = rank;

   for (int p = 0; p < ctx.num_particles; p++)
   {
      Particle part(pmeta);

      InitializeRandom(part, seed, pos_min, pos_max);
      //part.GetStateVar(V_N) = 0.0; // v0 = 0
      particles.AddParticle(part);

      if (ctx.test != -1)
      {
         Particle p_exact(*pmeta_exact);
         p_exact.GetCoords() = part.GetCoords();
         p_exact.GetStateVar(2) = part.GetStateVar(V_N);
         //p_exact.GetStateVar(2) = 0.0; // vn
         p_exact.GetStateVar(0) = p_exact.GetCoords(); // Set x0
         p_exact.GetStateVar(1) = p_exact.GetStateVar(2); // v0
         particles_exact->AddParticle(p_exact);
      }

      seed += size;
   }


   real_t time = 0.0;

   // Initialize fluid IC
   VectorFunctionCoefficient u_excoeff(2, couetteFlow);
   ParGridFunction &u_gf = *flowsolver.GetCurrentVelocity();
   ParGridFunction &w_gf = *flowsolver.GetCurrentVorticity();
   u_excoeff.SetTime(time);
   u_gf.ProjectCoefficient(u_excoeff);
   flowsolver.ComputeCurl2D(u_gf, w_gf);

   // Set fluid BCs
   Array<int> bdr_attr(pmesh.bdr_attributes.Max());
   bdr_attr[0] = 1; bdr_attr[2] = 1;
   flowsolver.AddVelDirichletBC(couetteFlow, bdr_attr);

   // Initialize particle fluid-dependent IC
   FindPointsGSLIB finder(MPI_COMM_WORLD);
   finder.Setup(pmesh);
   finder.FindPoints(particles.GetAllCoords());
   finder.Interpolate(u_gf, particles.GetAllStateVar(U_N));
   finder.Interpolate(w_gf, particles.GetAllStateVar(W_N));

   // Initialize particle integrator
   ParticleIntegrator pint(MPI_COMM_WORLD, 2, ctx.zeta, pmesh);

   // Initialize ParaView DC (if freq != 0)
   std::unique_ptr<ParaViewDataCollection> pvdc;
   if (ctx.paraview_freq > 0)
   {
      pvdc = std::make_unique<ParaViewDataCollection>("Couette", &pmesh);
      pvdc->SetPrefixPath("ParaView");
      pvdc->SetLevelsOfDetail(ctx.order);
      pvdc->SetDataFormat(VTKFormat::BINARY);
      pvdc->SetHighOrderOutput(true);
      pvdc->RegisterField("Velocity",flowsolver.GetCurrentVelocity());
      pvdc->RegisterField("Vorticity",flowsolver.GetCurrentVorticity());
      pvdc->SetTime(time);
      pvdc->SetCycle(0);
      pvdc->Save();
   }
   
   std::string csv_prefix = "Navier_Particles_";
   if (ctx.print_csv_freq > 0)
   {
      std::string file_name = csv_prefix + mfem::to_padded_string(0, 9) + ".csv";
      particles.PrintCSV(file_name.c_str());
      if (ctx.test != -1)
      {
         std::string file_name_exact = csv_prefix + "Exact_" + mfem::to_padded_string(0, 9) + ".csv";
         particles_exact->PrintCSV(file_name_exact.c_str());
      }
   }
   
   Array<real_t> beta, alpha; // EXTk/BDF coefficients
   flowsolver.Setup(ctx.dt);
   for (int step = 1; step <= ctx.num_steps; step++)
   {
      // Step Navier
      flowsolver.Step(time, ctx.dt, step-1);
      if (Mpi::Root())
      {
         cout << "Time: " << time << endl;
      }

      // Get the time-integration coefficients
      flowsolver.GetTimeIntegrationCoefficients(beta, alpha);
      
      // Step particles
      pint.Step(ctx.dt, beta, alpha, u_gf, w_gf, particles);

      if (ctx.print_csv_freq > 0 && step % ctx.print_csv_freq == 0)
      {
         // Output the particles
         std::string file_name = csv_prefix + mfem::to_padded_string(step, 9) + ".csv";
         particles.PrintCSV(file_name.c_str());

         // Set + output the exact
         if (ctx.test != -1)
         {
            for (int i = 0; i < particles_exact->GetNP(); i++)
            {
               Particle p_exact(*pmeta_exact);
               particles_exact->GetParticle(i, p_exact);
               analyticalTest0(ctx.zeta, p_exact.GetStateVar(0), p_exact.GetStateVar(1), time, p_exact.GetCoords(), p_exact.GetStateVar(2));
               particles_exact->SetParticle(i, p_exact);
            }
            std::string file_name_exact = csv_prefix + "Exact_" + mfem::to_padded_string(step, 9) + ".csv";
            particles_exact->PrintCSV(file_name_exact.c_str());
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