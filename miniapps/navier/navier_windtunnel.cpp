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

// Navier Wind Tunnel
//
// The problem domain is set up as follows:
//
//            Top (no-penetration)
//               +-------+
//              /|      /|
//             / |     / |
//    Left    /  |    /  |  Right
// (no-pen)  +-------+   | (no-pen)
//           |   |   |   |
//           |   +---|---+
//           |  /    |  /
//           | /     | /
//           |/      |/
//           +-------+
//          Ground (no-slip)
//
// Boundary conditions:
// - Left/Right (attr 2,4): Kovasznay velocity Dirichlet BC
// - Top/Bottom walls (attr 1,3): No-penetration BC (zero normal velocity)
//
// The problem, although steady state, is time integrated up to the
// final time and the solution is compared with the known exact solution.

#include "navier_solver.hpp"
#include <fstream>

using namespace mfem;
using namespace navier;

struct s_NavierContext
{
   int ser_ref_levels = 1;
   int order = 2; // TODO: Left as 2 for testing, but should be raised.
   real_t kinvis = 1.0 / 100.0; // TODO: Re = 100
   real_t t_final = 5.0; // TODO: Was 10 * 0.001, but want longer time
   real_t dt = 0.01; // TODO: Increased time step by factor of 10
   real_t inlet_velocity = 1.0;
   bool pa = true;
   bool ni = false;
   bool visualization = false;
   bool checkres = false;
} ctx;

void vel_inlet(const Vector &x, real_t t, Vector &u)
{
    u(0) = ctx.inlet_velocity;
    u(1) = 0.0;
    u(2) = 0.0;
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
   args.Parse();

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

   Mesh mesh = Mesh::MakeCartesian3D(6, 2, 4, Element::HEXAHEDRON, 3.0, 1.0, 2.0);

   for (int i = 0; i < ctx.ser_ref_levels; i++)
   {
      mesh.UniformRefinement();
   }

   if (Mpi::Root())
   {
      mfem::out << "Number of elements: " << mesh.GetNE() << std::endl;
      mfem::out << "Mesh dimension: " << mesh.Dimension() << std::endl;
      mfem::out << "Number of boundary attributes: " << mesh.bdr_attributes.Max() << std::endl;
   }

   auto *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   NavierSolver flowsolver(pmesh, ctx.order, ctx.kinvis);
   flowsolver.EnablePA(ctx.pa);
   flowsolver.EnableNI(ctx.ni);

   // Set up the boundary conditions

   // 1. INLET (attr 5): Prescribed velocity  TODO: Move to log-law
   Array<int> attr_inlet(pmesh->bdr_attributes.Max());
   attr_inlet = 0;
   attr_inlet[4] = 1;  // attr 5
   flowsolver.AddVelDirichletBC(vel_inlet, attr_inlet);

   // 2. GROUND (attr 3): No-slip
   Array<int> attr_ground(pmesh->bdr_attributes.Max());
   attr_ground = 0;
   attr_ground[2] = 1;  // attr 3
   VectorConstantCoefficient zero_vel(Vector({0.0, 0.0, 0.0}));
   flowsolver.AddVelDirichletBC(&zero_vel, attr_ground);

   // 3. LEFT WALL (attr 1): No-penetration
   Array<int> attr_left(pmesh->bdr_attributes.Max());
   attr_left = 0;
   attr_left[0] = 1;  // attr 1
   flowsolver.AddVelDirichletBC(new ConstantCoefficient(0.0), attr_left, 0);

   // 4. RIGHT WALL (attr 2): No-penetration
   Array<int> attr_right(pmesh->bdr_attributes.Max());
   attr_right = 0;
   attr_right[1] = 1;  // attr 2
   flowsolver.AddVelDirichletBC(new ConstantCoefficient(0.0), attr_right, 0);

   // 5. TOP WALL (attr 4): No-penetration
   Array<int> attr_top(pmesh->bdr_attributes.Max());
   attr_top = 0;
   attr_top[3] = 1;  // attr 4
   flowsolver.AddVelDirichletBC(new ConstantCoefficient(0.0), attr_top, 1);

   // 6. OUTLET (attr 6): Do nothing
   // Nothing to do here...

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

}
