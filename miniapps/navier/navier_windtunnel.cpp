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

// Navier Wind Tunnel
//
// This example constructs a numerical wind tunnel with flow in the +x direction.
// The inlet profile is specified by the user using the -profile option. Note that
// some of the profiles may require additional parameters to be set.
//
// Domain: [0, 3] × [0, 1] × [0, 0.5] (Length × Width × Height)
//
// Boundary conditions:
// - Left (attr 5, x=0): Inlet - Prescribed velocity Dirichlet BC
// - Bottom (attr 1, z=0): Ground - No-slip BC (all velocity components = 0)
// - Front (attr 2, y=0): Front wall - No-penetration BC (zero y-velocity component)
// - Right (attr 3, x=3): Outlet - Do nothing (no BC)
// - Back (attr 4, y=1): Back wall - No-penetration BC (zero y-velocity component)
// - Top (attr 6, z=0.5): Top wall - No-penetration BC (zero z-velocity component)

#include "navier_solver.hpp"
#include <fstream>

using namespace mfem;
using namespace navier;

constexpr real_t DOMAIN_LENGTH = 3.0;
constexpr real_t DOMAIN_WIDTH = 1.0;
constexpr real_t DOMAIN_HEIGHT = 0.5;
constexpr int INITIAL_NX = 6;
constexpr int INITIAL_NY = 2;
constexpr int INITIAL_NZ = 2;

struct s_NavierContext
{
   int ser_ref_levels = 1;
   int order = 2;
   real_t kinvis = 1.0 / 10.0;
   real_t t_final = 2.0;
   real_t dt = 0.01;

   bool pa = true;
   bool ni = false;
   bool visualization = false;

   // Inlet profile selection
   int inlet_profile_type = 0;

   // Common parameters
   real_t inlet_velocity = 1.0;
   real_t ref_height = 0.4;

   // Power law parameters
   real_t power_alpha = 0.15;    // Power law exponent

   // Logarithmic parameters
   real_t u_star = 0.1;          // Friction velocity
   real_t z0 = 0.001;            // Roughness length
   real_t kappa = 0.4;           // von Karman constant
   real_t f = 0.0001;            // Coriolis parameter
} ctx;

namespace InletProfile
{
/// Constant velocity profile
/// Set -profile 0
void constant(const Vector &x, real_t t, Vector &u)
{
   u(0) = ctx.inlet_velocity;
   u(1) = 0.0;
   u(2) = 0.0;
}

/// Power law profile
/// Set -profile 1
void power_law(const Vector &x, real_t t, Vector &u)
{
   real_t z = x(2);
   real_t u_z = ctx.inlet_velocity * pow(z / ctx.ref_height, ctx.power_alpha);

   u(0) = u_z > 0.0 ? u_z : 0.0;
   u(1) = 0.0;
   u(2) = 0.0;
}

/// Uniform logarithmic wind profile
/// Set -profile 2
void logarithmic(const Vector &x, real_t t, Vector &u)
{
   real_t z = x(2);

   // NOTE: To avoid log(0) = NaN, we enforce x(2) = z > ctx.z0
   real_t safe_z = std::max(z, ctx.z0 * real_t(1.01));

   real_t u_z = (ctx.u_star / ctx.kappa) * (log(safe_z/ctx.z0) + 34.5 * ctx.f *
                                            safe_z);

   u(0) = u_z > 0.0 ? u_z : 0.0;
   u(1) = 0.0;
   u(2) = 0.0;
}
} // END OF INLET PROFILE NAMESPACE

// Profile selection function
void (*get_inlet_profile())(const Vector &, real_t, Vector &)
{
   switch (ctx.inlet_profile_type)
   {
      case 0: return InletProfile::constant;
      case 1: return InletProfile::power_law;
      case 2: return InletProfile::logarithmic;
      default: return InletProfile::constant;
   }
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   int visport = 19916;

   OptionsParser args(argc, argv);
   args.AddOption(&ctx.ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&ctx.order,
                  "-o",
                  "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ctx.dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&ctx.t_final, "-tf", "--final-time", "Final time.");
   args.AddOption(&ctx.inlet_velocity, "-u0", "--inlet-velocity",
                  "Inlet velocity magnitude.");
   args.AddOption(&ctx.pa,
                  "-pa",
                  "--enable-pa",
                  "-no-pa",
                  "--disable-pa",
                  "Enable partial assembly.");
   args.AddOption(&ctx.ni,
                  "-ni",
                  "--enable-ni",
                  "-no-ni",
                  "--disable-ni",
                  "Enable numerical integration rules.");
   args.AddOption(&ctx.visualization,
                  "-vis",
                  "--visualization",
                  "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");
   args.AddOption(&ctx.inlet_profile_type, "-profile", "--inlet-profile",
                  "Inlet profile: 0=constant, 1=power, 2=logarithmic");
   args.AddOption(&ctx.ref_height, "-href", "--reference-height",
                  "Reference height for wind profile");
   args.AddOption(&ctx.power_alpha, "-alpha", "--power-alpha",
                  "Power law exponent (0.1-0.4)");
   args.AddOption(&ctx.z0, "-z0", "--roughness-length",
                  "Surface roughness length");
   args.AddOption(&ctx.u_star, "-ustar", "--friction-velocity",
                  "Friction velocity for log profile");
   args.Parse();

   // Specific comparison checks on parameters
   if (ctx.inlet_profile_type < 0 || ctx.inlet_profile_type > 2) {
      MFEM_WARNING("Inlet profile should be 0, 1, or 2. Using default profile (Constant).");
   }
   if (ctx.power_alpha < 0.1 || ctx.power_alpha > 0.4) {
      MFEM_WARNING("Power law exponent outside of (0.1, 0.4).");
   }



   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(mfem::out);
      }
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(mfem::out);
   }

   Mesh mesh = Mesh::MakeCartesian3D(INITIAL_NX, INITIAL_NY, INITIAL_NZ, Element::HEXAHEDRON,
                                     DOMAIN_LENGTH, DOMAIN_WIDTH, DOMAIN_HEIGHT);

   for (int i = 0; i < ctx.ser_ref_levels; ++i)
   {
      mesh.UniformRefinement();
   }

   if (Mpi::Root())
   {
      std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
      std::cout << "Mesh dimension: " << mesh.Dimension() << std::endl;
      std::cout << "Number of boundary attributes: " << mesh.bdr_attributes.Max() <<
                std::endl;

      // Print boundary attribute assignments for verification
      std::cout << "\nBoundary attribute assignments:" << std::endl;
      std::cout << "  Bottom (z=0): attr 1" << std::endl;
      std::cout << "  Front (y=0):  attr 2" << std::endl;
      std::cout << "  Right (x=max): attr 3" << std::endl;
      std::cout << "  Back (y=max):  attr 4" << std::endl;
      std::cout << "  Left (x=0):   attr 5" << std::endl;
      std::cout << "  Top (z=max):  attr 6" << std::endl;
   }

   auto *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Create the flow solver.
   NavierSolver flowsolver(pmesh, ctx.order, ctx.kinvis);
   flowsolver.EnablePA(ctx.pa);
   flowsolver.EnableNI(ctx.ni);

   // Set the initial condition (quiescent flow)
   ParGridFunction *u_ic = flowsolver.GetCurrentVelocity();
   VectorConstantCoefficient u_init(Vector({0.0, 0.0, 0.0}));
   u_ic->ProjectCoefficient(u_init);

   // BOUNDARY CONDITIONS

   // 1. GROUND: No-slip
   Array<int> attr_ground(pmesh->bdr_attributes.Max());
   attr_ground = 0;
   attr_ground[0] = 1;  // attr 1
   flowsolver.AddVelDirichletBC(new VectorConstantCoefficient(Vector({0.0, 0.0, 0.0})),
                                attr_ground);

   // 2. "FRONT" WALL: No-penetration
   Array<int> attr_front(pmesh->bdr_attributes.Max());
   attr_front = 0;
   attr_front[1] = 1;  // attr 2
   flowsolver.AddVelDirichletBC(new ConstantCoefficient(0.0), attr_front, 1);

   // 3. "BACK" WALL: No-penetration
   Array<int> attr_back(pmesh->bdr_attributes.Max());
   attr_back = 0;
   attr_back[3] = 1;  // attr 4
   flowsolver.AddVelDirichletBC(new ConstantCoefficient(0.0), attr_back, 1);

   // 4. "TOP" WALL: No-penetration
   Array<int> attr_top(pmesh->bdr_attributes.Max());
   attr_top = 0;
   attr_top[5] = 1;  // attr 6
   flowsolver.AddVelDirichletBC(new ConstantCoefficient(0.0), attr_top, 2);

   // 5. INLET: Prescribed velocity
   Array<int> attr_inlet(pmesh->bdr_attributes.Max());
   attr_inlet = 0;
   attr_inlet[4] = 1;  // attr 5
   flowsolver.AddVelDirichletBC(get_inlet_profile(), attr_inlet);

   // 6. OUTLET: Do nothing

   real_t t = 0.0;
   real_t dt = ctx.dt;
   real_t t_final = ctx.t_final;
   bool last_step = false;

   flowsolver.Setup(dt);

   ParGridFunction *u_gf = nullptr;
   ParGridFunction *p_gf = nullptr;

   if (Mpi::Root())
   {
      printf("%5s %8s %8s %8s %11s\n",
             "Step", "Time", "dt", "CFL", "||u||_max");
   }

   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      flowsolver.Step(t, dt, step);

      u_gf = flowsolver.GetCurrentVelocity();
      p_gf = flowsolver.GetCurrentPressure();

      ParaViewDataCollection pvdc("navier_windtunnel_output", pmesh);
      pvdc.SetDataFormat(VTKFormat::BINARY32);
      pvdc.SetHighOrderOutput(true);
      pvdc.SetLevelsOfDetail(ctx.order);
      pvdc.SetCycle(step);
      pvdc.SetTime(t);
      pvdc.RegisterField("velocity", u_gf);
      pvdc.RegisterField("pressure", p_gf);
      pvdc.Save();

      real_t cfl = flowsolver.ComputeCFL(*u_gf, dt);
      real_t max_vel = u_gf->Max();

      if (Mpi::Root() && step % 10 == 0)
      {
         printf("%5d %8.3f %8.2E %8.2E %11.4E\n",
                step, t, dt, cfl, max_vel);
         fflush(stdout);
      }
   }

   if (ctx.visualization)
   {
      char vishost[] = "localhost";
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "parallel " << Mpi::WorldSize() << " "
               << Mpi::WorldRank() << "\n";
      sol_sock << "solution\n" << *pmesh << *u_gf << std::endl;
   }

   flowsolver.PrintTimingData();

   delete pmesh;

   return 0;
}
