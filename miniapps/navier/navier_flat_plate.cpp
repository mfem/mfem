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
// P = 101325Pa (sea level condition)
// u = 68.058 m/s (Mach 0.2)
// mu = 1.7894x10^-3 Kg /m * s (sea level condition scaled by 100 to ensure
//     the flow remains laminar)
// Re_x = 4.66x10^4 (at the end of the plate)
// rho = 1.225 km/m^3 (sea level condition)
//
// The 2D problem domain is set up like this:
// 
// Uniform flow from the left passes over a flat plate of 1m in length with its
// leading edge located at 0.1m into the domain. The incompressible
// Navier-Stokes equations are modeled throughout the entire domain as
// illustrated below:
//
//                 Atmosphere - symmetry boundary condition
//                 		(attr=3)
//                 ________________________________________
//                |                                        |
//                |             FLUID DOMAIN               |
//                |                                        |
//   --> inflow   |                                        | --> outflow
//    (attr=1)    |                                        |  (attr=2)
//                |_____-----------------------------------|
//	       Atm. sym. bc.	     flat plate
//               (attr=4)             (attr=5)    
//
// Uniform Dirichlet velocity conditions are imposed at inflow (attr=1) and
// homogeneous Dirichlet conditions are imposed on all surface (attr=3) except
// the outflow (attr=2) which has Neumann boundary conditions for velocity.

#include "mfem.hpp"
#include "navier_solver.hpp"
#include <fstream>
#include <iostream>

using namspace std;
using namespace mfem;
using namespace navier;

// Context structure for our Navier Solver
struct s_NavierContext
{
   int ser_ref_levels = 1; //Serial Refinement Levels
   int order = 6; // Finite Element function space order
   double kinvis = 0.0014607; //Kinematic viscocity - SET THIS TO APPROPRIATE VALUE
   double dt = 0.001; //Time-step size
   double t_final = 25 * dt; //Final time of simulation
   double reference_pressure = 0.0; //Reference Pressure
   bool pa = true;
   bool ni = false;
   bool visualization = false;
   bool paraview = true;
   bool checkres = false;
} ctx; 

// Dirichlet conditions for velocity
void vel_inlet(const Vector &x, double t, Vector &u);
void vel_ic(const Vector &x, double t, Vector &u);
void vel_wall(const Vector &x, double t, Vector &u);

// Visualizion of the solution
void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x = 0, int y = 0, int w = 400, int h = 400,
                    bool vec = false);

int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   int precision = 8;

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
   args.AddOption(&ctx.visualization,
                  "-vis",
                  "--visualization",
                  "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&ctx.paraview, "-paraview", 
		  "--paraview-datafiles", 
		  "-no-paraview",
        	  "--no-paraview-datafiles",
        	  "Save data files for ParaView (paraview.org) visualization.");
   args.AddOption(
      &ctx.checkres,
      "-cr",
      "--checkresult",
      "-no-cr",
      "--no-checkresult",
      "Enable or disable checking of the result. Returns -1 on failure.");
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

   Mesh *mesh = new Mesh("flat_plate.msh"); // Need to name our mesh flat-plate.mesh 
   int dim = mesh->Dimension();

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

   //FunctionCoefficient p_excoeff(pres_kovasznay);

   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the mesh.
   Array<int> attr(pmesh->bdr_attributes.Max());
   // Inlet is attribute 1.
   attr[0] = 1;
   // Outlet is arttribute 2.
   attr[1] = 0; 
   // Top symmetry is attribute 3.
   attr[2] = 1;
   // Bottom symmetry is attribute 4.
   attr[3] = 1;
   // Plate is attribute 5.
   attr[4] = 1;
   flowsolver.AddVelDirichletBC(vel_inlet, attr[0]);
   flowsolver.AddVelDirichletBC(vel_slip, attr[2]);
   flowsolver.AddVelDirichletBC(vel_slip, attr[3]);
   flowsolver.AddVelDirichletBC(vel_plate, attr[4]);
   // ===============================================================

   double t = 0.0; //Start time
   double dt = ctx.dt; //Time step size
   double t_final = ctx.t_final; //Final time
   bool last_step = false;

   flowsolver.Setup(dt);

   //Solution vectors for velocity (u_gf) and pressure (p_gf)
   ParGridFunction *u_gf = flowsolver.GetCurrentVelocity();
   ParGridFunction *p_gf = flowsolver.GetCurrentPressure();

   // INITIALIZING THE VISUALIZATION ASPECTS - GLVIS AND PARAVIEW IF ENEABLED
   // =======================================================================
   // Visualize the solution with GLVis - Initializing the viewer - Auto off
   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sout.open(vishost, visport);
      if (!sout)
      {
         if (Mpi::Root())
            cout << "Unable to connect to GLVis server at "
                 << vishost << ':' << visport << endl;
         visualization = false;
         if (Mpi::Root())
         {
            cout << "GLVis visualization disabled.\n";
         }
      }
      else
      {
         sout << "parallel " << num_procs << " " << myid << "\n";
         sout.precision(precision);
         sout << "Velocity solution\n" << *pmesh << *u_gf;
         sout << "pause\n";
         sout << flush;
         if (Mpi::Root())
            cout << "GLVis visualization paused."
                 << " Press space (in the GLVis window) to resume it.\n";
      }
   }


   //Paraview Data collection - Initializing based on command line flags - Auto on  
   ParaViewDataCollection pvdc = NULL;
   if(ctx.paraview)
   {
      pvdc = ParaViewDataCollection pvdc("flat_plate", pmesh);
      pvdc.SetDataFormat(VTKFormat::BINARY32);
      pvdc.SetHighOrderOutput(true);
      pvdc.SetLevelsOfDetail(ctx.order);
      pvdc.SetCycle(0);
      pvdc.SetTime(t);
      pvdc.RegisterField("velocity", u_gf);
      pvdc.RegisterField("pressure", p_gf);
      pvdc.Save();
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

      if(visualization)
      {
	 sout << "parallel " << num_procs << " " << myid << "\n";
         sout << "Velocity solution\n" << *pmesh << *u_gf << flush;
      }
      
      if (paraview && step % 10 == 0) //Storing every 10th time step for paraview visualization
      {
         pvdc.SetCycle(step);
         pvdc.SetTime(t);
         pvdc.Save();
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
// Fluid data
// Dirichlet conditions for uniform flow velocity in the inlet
void vel_inlet(const Vector &x, double t, Vector &u)
{
   double U = 68.058;
   u(0) = U;
   u(1) = 0.;
}

// Symmetry condition for top and bottom walls of domain
void vel_slip(const Vector &x, double t, Vector &u)
{
   u(1) = 0.;
}
// Initial conditions for velocity everywhere in domain
void vel_ic(const Vector &x, double t, Vector &u)
{
   double u_ic = 0.0001; //Small initial velocity to not divide by 0 anywhere. 
   u(0) = u_ic;
   u(1) = 0.;
}

// Direchlet 0 m/s at plate.
void vel_plate(const Vector &x, double t, Vector &u)
{
   u(0) = 0.;
   u(1) = 0.;
}

// To represent the outflow boundary condition the zero-stress boundary condition
// is used. The zero-stress boundary condition is automatically applied if there
// is no other boundary condition applied to a certain attribute.
