// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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
// Incompressible Navier-Stokes stagnation flat plate example
//
// Solve for steady flow defined by:
// U = 3.0 m/s
// nu = 1.5x10^-3 m^2/s
// Re_x = 200 (at the end of the plate)
//
// The 2D problem domain is set up like this:
// 
// Uniform flow from the left passes over a flat plate of 0.1m in length with its
// leading edge located at 0.01m into the domain. The incompressible
// Navier-Stokes equations are modeled throughout the entire domain as
// illustrated below:
//
//                                 Outflow
//                             (mesh attr = 3)
//                  ________________________________________
//                 |                                        |
//                 |             FLUID DOMAIN               |
//                 |                                        |
//   --> inflow    |                                        | -->  outflow
// (mesh attr = 4) |                                        |  (mesh attr = 2)
//                 |_____-----------------------------------|
//	        Atm. sym. bc.	     flat plate
//             (mesh attr = 1)      (mesh attr = 1)    
//
// Dirichlet velocity conditions are imposed at the left boundary (inflow) and
// the bottom boundary (symmetry and plate). Numann Pressure conditions are imposed
// on the right and top boundaries (outflow).

#include "mfem.hpp"
#include "navier_solver.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace navier;

// Context structure for our Navier Solver
struct s_NavierContext
{
   int ser_ref_levels = 0; //Serial Refinement Levels
   int order = 1; //Finite Element function space order
   double kinvis = 0.0015; //Kinematic viscocity
   double dt = 1e-4; //Time step size
   double steps = 2000; //Number of time steps
   double t_final = dt*steps; //Total run time
   bool pa = true;
   bool ni = false;
   bool slip_top = false; //option to have a slip wall on the top of boundary (attr 3)
   bool paraview = true;
   const char *mesh = "flat-plate-coarse.msh";
   const char *sol_dir = "flat_plate_coarse";
} ctx; 

// Dirichlet conditions for velocity
void vel_ic(const Vector &x, double t, Vector &u);
void vel_dbc(const Vector &x, double t, Vector &u);

int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   // parse command-line options
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
   args.AddOption(&ctx.paraview,
                  "-paraview", 
		  "--paraview-datafiles", 
		  "-no-paraview",
        	  "--no-paraview-datafiles",
        	  "Save data files for ParaView (paraview.org) visualization.");
   args.AddOption(&ctx.mesh, 
                  "-m", 
                  "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ctx.sol_dir, 
                  "-sd", 
                  "--sol_dir",
                  "Paraview output directory to create.");
   args.AddOption(&ctx.slip_top, 
                  "-st", 
                  "--enable-slip_top",
		  "-no-slip_top",
		  "--disable-slip_top",
                  "Turn slip top on or off.");
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

   Mesh mesh = Mesh(ctx.mesh, 1, 1); 

   //Mesh refinement
   for (int i = 0; i < ctx.ser_ref_levels; ++i)
   {
      mesh.UniformRefinement();
   }

   if (Mpi::Root())
   {
      std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
   }

   auto *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();


   // Create the flow solver.
   NavierSolver flowsolver(pmesh, ctx.order, ctx.kinvis);
   flowsolver.EnablePA(ctx.pa); //Auto on - Change with command line options
   flowsolver.EnableNI(ctx.ni); //Auto off - Change with command line options

   // Set the initial conditions.
   ParGridFunction *u_ic = flowsolver.GetCurrentVelocity(); 
   VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel_ic);
   u_ic->ProjectCoefficient(u_excoeff);


   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the mesh.
   Array<int> attr(pmesh->bdr_attributes.Max());
   // Bottom is attribute 1.
   attr[0] = 1;
   // Outlet is attribute 2.
   attr[1] = 0; 
   // Top is attribute 3.
   if(ctx.slip_top)
   {
      attr[2] = 1;
   }
   else
   {
      attr[2] = 0;
   }
   // Inlet is attribute 4
   attr[3] = 1;
   flowsolver.AddVelDirichletBC(vel_dbc, attr);
   // ===============================================================

   double t = 0.0; //Start time
   double dt = ctx.dt; //Time step size
   double t_final = ctx.t_final; //Final time
   bool last_step = false;

   flowsolver.Setup(dt);

   //Solution vectors for velocity (u_gf) and pressure (p_gf)
   ParGridFunction *u_gf = flowsolver.GetCurrentVelocity();
   ParGridFunction *p_gf = flowsolver.GetCurrentPressure();

   //Paraview Data collection - Initializing based on command line flags - Auto on  
   ParaViewDataCollection *pvdc = NULL;
   if(ctx.paraview)
   {
      pvdc = new ParaViewDataCollection(ctx.sol_dir, pmesh);
      pvdc->SetDataFormat(VTKFormat::BINARY32);
      pvdc->SetHighOrderOutput(true);
      pvdc->SetLevelsOfDetail(ctx.order);
      pvdc->SetCycle(0);
      pvdc->SetTime(t);
      pvdc->RegisterField("velocity", u_gf);
      pvdc->RegisterField("pressure", p_gf);
      pvdc->Save();
   }
   //========================================================================
   
   //Solving over time steps
   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      flowsolver.Step(t, dt, step);

      if (ctx.paraview && (step % 10 == 0)) //Storing every 10th time step for paraview visualization
      {
         pvdc->SetCycle(step);
         pvdc->SetTime(t);
         pvdc->Save();
      }

      if (Mpi::Root())
      {
         printf("%11s %11s\n", "Time", "dt");
         printf("%.5E %.5E\n", t, dt);
         fflush(stdout);
      }
   }

   flowsolver.PrintTimingData();

   delete pmesh;
   return 0;
}

// VELOCITY FUNCTIONS
// Initial conditions for velocity everywhere in domain
// Dirichlet conditions for uniform flow velocity in the inlet
void vel_ic(const Vector &x, double t, Vector &u)
{
   double u_ic = 1e-5; //Small initial velocity to avoid divide by 0. 
   u(0) = u_ic;
   u(1) = 0.0;
}

void vel_dbc(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);
   
   double U = 3.0; //Freestream velocity
   double tol = 1e-9; //must be smaller than smallest mesh element
   
   // Bottom boundary
   if(yi <= tol)
   {
      if(xi < 0.01) //Symmetry condition before the plate
      {
         u(1) = 0.0;
      }
      
      else // No slip plate 
      { 
         u(0) = 0.0;
         u(1) = 0.0;
      }
   }
   //Top boundary
   if (ctx.slip_top && yi >= 0.05-tol)
   {
      u(1) = 0.0 //slip wall on top boundary if specified.;
   }
   
   //Uniform velocity at inlet on left boundary
   if(xi <= tol)
   {	
      u(0) = U;
      u(1) = 0.0;
   }
}
