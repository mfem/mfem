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
   real_t d = 1e-3; // dimensional
   real_t rho_s = 1; // dimensional 
   real_t rho_f = 1; // dimensional 
   real_t U_m = 1.0; // dimensional
   real_t L = 1.0; // dimensional 
   real_t mu = 1.0; // dimensional
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

void prescibedFluidVelocity(const Vector &x, Vector &u)
{
   // For now, just assume uniform flowfield, nondimensionalized wrt U_m
   u = 1.0
}

// TODO: Author name?
class ParticleIntegrator
{
private:
   const real_t Fr; // Froude number
   const real_t CR; // C^R

   // TODO Discuss: - Defined St depends on v_r (directly and in C_D). not a constant??
   //               - C_L depends on local vorticity. not a constant??


   FindPointsGSLIB finder;

public:
   ParticleIntegrator(MPI_Comm comm, Mesh &m, real_t Fr_, real_t CR_)
   : finder(comm),
     Fr(Fr_),
     CR(CR_)
   {
      finder.Setup(m);
   }

   void Step(ParticleSet &particles, ParGridFunction &u_gf)
   {
      Vector 
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
   args.AddOption(&ctx.d, "-pd", "--particle-diameter", "Particle diameter, dimensional.");
   args.AddOption(&ctx.rho_s, "-rs", "--particle-density", "Particle density, dimensional.");
   args.AddOption(&ctx.rho_f, "-rf", "--fluid-density", "Fluid density, dimensional.");
   args.AddOption(&ctx.U_m, "-um", "--mean-vel", "Fluid velocity scale.");
   args.AddOption(&ctx.mu, "-mu", "--dynamic-visc", "Dynamic viscosity, dimensional.");
   args.AddOption(&ctx.L, "-L", "--length-scale", "Length scale, dimensional.");

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


   // Initialize a simple 2D domain [0,1] x [0,1]
   Mesh mesh = Mesh::MakeCartesian2D(50, 50, Element::Type::HEXAHEDRON);
   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   
   // Initialize the Navier solver
   const real_t Re = ctx.rho_f*ctx.U_m*ctx.L / ctx.mu;
   if (rank == 0)
   {
      cout << "Reynolds number: " << Re << endl;
   }
   NavierSolver flowsolver(pmesh, ctx.order, 1.0/Re);


   // Prescribe a velocity condition for now
   ParGridFunction &u_gf = *flowsolver.GetCurrentVelocity();
   VectorFunctionCoefficient u_excoeff(pmesh.Dimension(), prescibedFluidVelocity);
   u_gf.ProjectCoefficient(u_excoeff);


   // Initialize particles
   ParticleMeta pmeta(2, 2, {2,2,1,2,2,1,2,2,1});
   ParticleSet particles(pmeta, ctx.p_ordering);

   Vector pos_min, pos_max;
   mesh.GetBoundingBox(pos_min, pos_max);

   int seed = rank;
   for (int p = 0 p < ctx.num_particles; p++)
   {
      Particle p(pmeta);
      InitializeRandom(p, seed, pos_min, pos_max); // Initialize with random position, properties, + state
      particles.AddParticle(p);
      seed += size;
   }

   // Set particle diameter and density
   particles.GetAllProperty(Prop::DIAMETER) = ctx.d;
   particles.GetAllProperty(Prop::DENSITY) = ctx.rho_s;

   // Ensure that entire state is zeroed out as IC
   for (int s = 0; s < State::SIZE; s++)
   {
      particles.GetAllStateVar(s) = 0.0;
   }

   




}