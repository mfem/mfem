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

#include <cmath>

using namespace std;
using namespace mfem;
using namespace mfem::common;
using namespace navier;

struct s_NavierContext
{
   int order = 5;
   real_t dt = 0.25e-4;
   int num_steps = 1000;
   int num_particles = 1000;
   int p_ordering = Ordering::byNODES;
   real_t zeta = 1.0;
   bool visualization = false;
   int visport = 19916;
   
} ctx;

// Particle Properties: Density (rho_s) and diameter (d)
enum Prop : int
{
   DIAMETER,
   DENSITY,
   SIZE
};

// Particle State Variables: fluid velocity u, particle velocity v, vorticity w, at three timesteps
enum State : int
{
   U_NP2, // VDim = 2
   V_NP2, // VDim = 2
   W_NP2, // VDim = 1
   U_NP1, // VDim = 2
   V_NP1, // VDim = 2
   W_NP1, // VDim = 1
   U_N,   // VDim = 2
   V_N,   // VDim = 2
   W_N,   // VDim = 1
   SIZE  
};

void couetteFlow(const Vector &x, Vector &u)
{
   u = 0.0;
   u[0] = (1.0+x[1])/2.0;
}

void analyticalNoDragCouette(const real_t zeta, const Vector &x0, const Vector &v0, double t, Vector &x, Vector &v)
{
   x.SetSize(2); v.SetSize(2);

   const real_t C1 = v0[1];
   const real_t C2 = (sqrt(zeta)/2.0) * (x0[1] + 1 - 2*v0[0])/(sqrt(zeta-1));
   const real_t C3 = x0[1] + 2.0*C2/(sqrt(zeta*(zeta-1)));
   const real_t C4 = 0.5*(C3+1);
   const real_t C5 = x0[0] + (2.0/(zeta-1))*v0[1];

   const real_t lam = zeta*(zeta-1)/4.0;

   x[0] = (zeta/(2.0*lam)) * (-C1*cos(sqrt(lam)*t) - C2*sin(sqrt(lam)*t)) + C4*t + C5;
   x[1] = (1.0/sqrt(zeta)) * (C1*sin(sqrt(lam)*t) - C2*cos(sqrt(lam)*t)) + C3;
   v[0] = (zeta/(2.0*sqrt(lam))) * ( C1*sin(sqrt(lam)*t) - C2*cos(sqrt(lam)*t)) + C4;
   v[1] = C1*cos(sqrt(lam)*t) + C2*sin(sqrt(lam)*t);
}

class ParticleIntegrator
{
private:
   const real_t zeta;
   FindPointsGSLIB finder;

public:
   ParticleIntegrator(MPI_Comm comm, Mesh &m, const real_t zeta_)
   : zeta(zeta_)
     finder(comm)
   {
      finder.Setup(m);
   }


   void ParticleStep(const real_t &dt, const Array<real_t> &beta, const Array<real_t> &alpha, Particle &p)
   {

      real_t w_n = p.GetStateVar(W_N)[0];

      // Assemble the 2D matrix B
      DenseMatrix B({{beta[0], zeta*dt*w_n},
                     {-zeta*dt*w_n, beta[0]}});
      
      // Assemble the RHS
      Vector r(2);
      r = 0.0;
      for (int j = 1; j <= k; j++)
      {
         // add particle velocity component
         add(r, beta[j], p.GetStateVar(1+3*k), r);

         // Assemble 
      }

   
   }

   void Step(const real_t &dt, const Array<real_t> &beta, const Array<real_t> &alpha, ParticleSet &particles, ParGridFunction &u_gf)
   {
      for (int i = 0; i < particles.GetNP(); i++)
      {
         Particle p;
         particles.GetParticle(i);
         ParticleStep(dt, beta, alpha, p);
         particles.SetParticle(i);
      }
   }
}


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
   args.AddOption(&ctx.rho_s, "-z", "--zeta", "Zeta constant.");
   args.AddOption(&ctx.rho_f, "-g", "--gamma", "Gamma constant.");

   args.AddOption(&ctx.visualization, "-vis", "--visualization", "-no-vis", "--no-visualization", "Enable or disable GLVis visualization.");
   args.AddOption(&ctx.visport, "-p", "--send-port", "Socket for GLVis.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }


   // Initialize a simple straight-edged 2D domain [0,12] x [-1,1]
   Mesh mesh = Mesh::MakeCartesian2D(50, 50, Element::Type::HEXAHEDRON, true, 12.0, 2.0);
   Vector transl(mesh.GetNV()*2);
   // Mesh vertex ordering is byNODES
   for (int i = 0; i < transl.Size()/2; i++)
   {
      transl[mesh.GetNV() + i] = -1.0;
   }
   mesh.MoveNodes(transl);


   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   
   // Initialize the Navier solver
   const real_t Re = 1.0; // TODO. Unimportant right now.
   if (rank == 0)
   {
      cout << "Reynolds number: " << Re << endl;
   }
   NavierSolver flowsolver(pmesh, ctx.order, 1.0/Re);


   // Prescribe the velocity condition for now
   ParGridFunction &u_gf = *flowsolver.GetCurrentVelocity();
   VectorFunctionCoefficient u_excoeff(2, couetteFlow);
   u_gf.ProjectCoefficient(u_excoeff);


   // Initialize two particle sets - one numerical, one analytical
   ParticleMeta pmeta(2, 2, {2,2,1,2,2,1,2,2,1});
   ParticleSet particles(pmeta, ctx.p_ordering);

   // Only need to store velocity
   ParticleMeta pmeta_exact(2, 0, {2});
   ParticleSet particles_exact(pmeta, ctx.p_ordering)

   // Initialize both particle sets the same
   Vector pos_min, pos_max;
   mesh.GetBoundingBox(pos_min, pos_max);

   int seed = rank;
   for (int p = 0 p < ctx.num_particles; p++)
   {
      Particle p(pmeta), p_exact(pmeta_exact);
      InitializeRandom(p, seed, pos_min, pos_max);
      InitializeRandom(p_exact, seed, pos_min, pos_max);
      particles.AddParticle(p);
      particles_exact.AddParticle(p_exact);
      seed += size;
   }

   // Set particle diameter and density
   particles.GetAllProperty(Prop::DIAMETER) = ctx.d;
   particles.GetAllProperty(Prop::DENSITY) = ctx.rho_s;

   // Ensure that entire state is zeroed out as IC
   for (int s = 0; s < State::SIZE; s++)
   {
      particles.GetAllStateVar(s) = 0.0;
      particles_exact.GetAllStateVar(s) = 0.0;
   }
   // Particles are now initialized across entire domain w/ zero IC



   




}