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

// 3D flow over a cylinder benchmark example

#include "navier_solver.hpp"
#include "ceed.h"
#include <fstream>

using namespace mfem;
using namespace navier;

struct s_NavierContext
{
   int order = 4;
   double kin_vis = 1.0 / 700.0;
   int nsteps = 5000;
   double dt = 1e-8;
   double dt_max = 5e-3;
   double cfl_target = 0.9;
   double t_final = 50.0;
   double Uavg = 1.0;
   double H = 1.0;
   double Lx = 9.0*H;
   double Ly = 3.035*H;
   double beta_x = 2.0;
   double beta_y = 2.4;
   double W = 1.929;
} ctx;

// See Almeida et al. 1993 and
// https://turbmodels.larc.nasa.gov/Other_LES_Data/2Dhill_periodic/hill-geometry.dat

inline double hill_step(double x, double w, double h)
{
   double y, xs = x / w;
   if (xs <= 0.0)
   {
      y = h;
   }
   else if (xs > 0.0 && xs <= 9.0/54.0)
   {
      y = h*std::min(1.0,1.0+7.05575248e-1*pow(xs,2.0)-1.1947737203e1*pow(xs,3.0));
   }
   else if (xs > 9.0/54.0 && xs <= 14.0/54.0)
   {
      y = h*(0.895484248+1.881283544*xs-10.582126017*pow(xs,2.0)
             +10.627665327*pow(xs,3.0));
   }
   else if ((xs>14.0/54.0) && (xs<=20.0/54.0))
   {
      y = h*(0.92128609+1.582719366*xs-9.430521329*pow(xs,2.0)
             +9.147030728*pow(xs,3.0));
   }
   else if ((xs>20.0/54.0) && (xs<=30.0/54.0))
   {
      y = h*(1.445155365-2.660621763*xs+2.026499719*pow(xs,2.0)
             -1.164288215*pow(xs,3.0));
   }
   else if ((xs>30.0/54.0) && (xs<=40.0/54.0))
   {
      y = h*(0.640164762+1.6863274926*xs-5.798008941*pow(xs,2.0)
             +3.530416981*pow(xs,3.0));
   }
   else if ((xs>40.0/54.0) && (xs<=1.0))
   {
      y = h*(2.013932568-3.877432121*xs+1.713066537*pow(xs,2.0)
             +0.150433015*pow(xs,3.0));
   }
   else
   {
      y = 0.0;
   }
   return y;
}
inline double hill_height(double x, double Lx, double w, double h)
{
   double xx = 0.0;
   if (x < 0.0)
   {
      xx = Lx + std::fmod(x, Lx);
   }
   else if (x > Lx)
   {
      xx = std::fmod(x, Lx);
   }
   else
   {
      xx = x;
   }

   return hill_step(xx,w,h) + hill_step(Lx-xx,w,h);
}

void vel_inlet(const Vector &c, double t, Vector &u)
{
   double x = c(0);
   double y = c(1);

   u(0) = ctx.Uavg;
   u(1) = 0.0;
}

void vel_ic(const Vector &c, Vector &u)
{
   double x = c(0);
   double y = c(1);
   double z = 0.0;

   double amp = 0.2, ran = 0.0;

   ran = 3.e4*(x*sin(y)+z*cos(y));
   ran = 6.e3*sin(ran);
   ran = 3.e3*sin(ran);
   ran = cos(ran);
   u(0) = 1. + ran*amp;

   ran = (2+ran)*1.e4*(y*sin(z)+x*cos(z));
   ran = 2.e3*sin(ran);
   ran = 7.e3*sin(ran);
   ran = cos(ran);
   u(1) = ran*amp;
}

void velocity_wall(const Vector &c, const double t, Vector &u)
{
   u(0) = 0.0;
   u(1) = 0.0;
}

bool in_between(const double x, const double x0, const double x1)
{
   double tol = 1e-8;
   return (fabs(x - x0) >= tol) && (fabs(x - x1) >= tol);
}

void flowrate(const Vector &, double t, Vector &u)
{
   u(0) = 0.0;
   u(1) = 0.0;
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *device_config = "cpu";
   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
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
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   int serial_refinements = 1;
   int nx = 11;
   int ny = 8;

   Mesh mesh = Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL);

   mesh.EnsureNodes();
   H1_FECollection mesh_fec(ctx.order, mesh.Dimension());
   FiniteElementSpace mesh_fes(&mesh, &mesh_fec, mesh.Dimension());
   mesh.SetNodalFESpace(&mesh_fes);

   // See Nek5000 example. Decrease resolution in the high velocity regions
   // (increase CFL) and increase resolution near the wall.
   mesh.Transform([](const Vector &coords, Vector &u)
   {
      const double x = coords(0);
      const double y = coords(1);

      u(0) = 0.5*(sinh(ctx.beta_x*(x-0.5))/sinh(ctx.beta_x*0.5) + 1.0);
      u(1) = 0.5*(tanh(ctx.beta_y*(2.0*y-1.0))/tanh(ctx.beta_y) + 1.0);
   });

   // Rescale to [0,Lx]x[0,Ly]
   mesh.Transform([](const Vector &coords, Vector &u)
   {
      const double x = coords(0);
      const double y = coords(1);

      double xscale = ctx.Lx;
      double yscale = ctx.Ly;

      u(0) = xscale * x;
      u(1) = yscale * y;
   });

   // Shift points in x
   mesh.Transform([](const Vector &coords, Vector &u)
   {
      const double x = coords(0);
      const double y = coords(1);

      double amp = 0.25;
      double xfac = 0.0;
      double yfac = pow((1.0-y/ctx.Ly), 3.0);

      if (x <= ctx.W/2.0)
      {
         xfac = -2.0/ctx.W * x;
      }
      else if ((x > ctx.W/2.0) && (x <= ctx.Lx-ctx.W/2.0))
      {
         xfac = 2.0/(ctx.Lx-ctx.W) * x - 1.0 - ctx.W/(ctx.Lx-ctx.W);
      }
      else if (x > (ctx.Lx-ctx.W/2.0))
      {
         xfac = -2.0/ctx.W * x + 2.0*ctx.Lx/ctx.W;
      }

      double shift = xfac*yfac;

      u(0) = x + amp * shift;
      u(1) = y;
   });

   mesh.Transform([](const Vector &coords, Vector &u)
   {
      const double x = coords(0);
      const double y = coords(1);

      const double yh = hill_height(x, ctx.Lx, ctx.W, ctx.H);

      u(0) = x;
      u(1) = yh + y * (1.0 - yh / ctx.Ly);
   });


   for (int i = 0; i < serial_refinements; ++i)
   {
      mesh.UniformRefinement();
   }

   Vector x_translation({ctx.Lx, 0.0});
   std::vector<Vector> translations = {x_translation};
   Mesh periodic_mesh = Mesh::MakePeriodic(mesh,
                                           mesh.CreatePeriodicVertexMapping(translations));

   if (Mpi::Root())
   {
      std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
   }

   // char buffer[50];
   // sprintf(buffer,"phill_%d_%d_%d.mesh", nx, ny, serial_refinements);
   // mesh.Save(buffer);

   auto *pmesh = new ParMesh(MPI_COMM_WORLD, periodic_mesh);

   // Create the flow solver.
   NavierSolver flowsolver(pmesh, ctx.order, ctx.kin_vis);

   auto kv_gf = flowsolver.GetVariableViscosity();
   ConstantCoefficient kv_coeff(ctx.kin_vis);
   kv_gf->ProjectCoefficient(kv_coeff);

   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the mesh.
   Array<int> inlet_attr(pmesh->bdr_attributes.Max());
   inlet_attr = 0;
   inlet_attr[3] = 1;

   Array<int> wall_attr(pmesh->bdr_attributes.Max());
   wall_attr = 0;
   wall_attr[0] = 1;
   wall_attr[2] = 1;

   Array<int> outlet_attr(pmesh->bdr_attributes.Max());
   outlet_attr = 0;
   outlet_attr[1] = 1;

   // Set the initial condition.
   ParGridFunction *u_ic = flowsolver.GetCurrentVelocity();
   VectorFunctionCoefficient u_ic_coeff(pmesh->Dimension(), vel_ic);
   u_ic->ProjectCoefficient(u_ic_coeff);

   // Inlet
   // VectorFunctionCoefficient u_inlet_coeff(pmesh->Dimension(), vel_inlet);
   // u_ic->ProjectBdrCoefficient(u_inlet_coeff, inlet_attr);

   // Walls
   VectorFunctionCoefficient velocity_wall_coeff(pmesh->Dimension(),
                                                 velocity_wall);
   u_ic->ProjectBdrCoefficient(velocity_wall_coeff, wall_attr);

   // flowsolver.AddVelDirichletBC(vel_inlet, inlet_attr);
   flowsolver.AddVelDirichletBC(velocity_wall, wall_attr);

   // ConstantCoefficient pres_outlet(0.0);
   // flowsolver.AddPresDirichletBC(&pres_outlet, outlet_attr);

   Array<int> domain_attr(pmesh->attributes.Max());
   domain_attr = 1;
   flowsolver.AddAccelTerm(flowrate, domain_attr);

   double t = 0.0;
   double dt = ctx.dt;
   double t_final = ctx.t_final;
   bool last_step = false;

   flowsolver.SetFilterAlpha(0.1);

   flowsolver.Setup(dt);

   ParGridFunction *u_next_gf = nullptr;
   ParGridFunction *u_gf = flowsolver.GetCurrentVelocity();
   ParGridFunction *p_gf = flowsolver.GetCurrentPressure();

   double cfl_max = 2.0;
   double cfl_tol = 1.0;

   ParaViewDataCollection pvdc("output/phill", pmesh);
   pvdc.SetDataFormat(VTKFormat::BINARY32);
   pvdc.SetHighOrderOutput(true);
   pvdc.SetLevelsOfDetail(ctx.order);
   pvdc.SetCycle(0);
   pvdc.SetTime(t);
   pvdc.RegisterField("velocity", u_gf);
   pvdc.RegisterField("pressure", p_gf);
   pvdc.Save();

   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      flowsolver.Step(t, dt, step, true);
      // Get a prediction for a stable timestep
      int ok = flowsolver.PredictTimestep(1e-8, ctx.dt_max, ctx.cfl_target, dt);
      if (ok < 0)
      {
         // Reject the time step
         if (Mpi::Root())
         {
            std::cout
                  << "Step reached maximum CFL or predicted CFL, retrying with smaller step size."
                  << std::endl;
         }
      }
      else
      {
         // Queue new time step in the history array
         flowsolver.UpdateTimestepHistory(dt);
         // Accept the time step
         t += dt;
      }

      // Compute the CFL
      double cfl = flowsolver.ComputeCFL(*u_gf, dt);

      if (step % 250 == 0)
      {
         pvdc.SetCycle(step);
         pvdc.SetTime(t);
         pvdc.Save();
      }

      if (Mpi::Root())
      {
         printf("%11s %11s %11s %11s\n", "Step", "Time", "dt", "CFL");
         printf("%11d %.5E %.5E %.5E\n", step, t, dt, cfl);
         fflush(stdout);
      }
   }

   flowsolver.PrintTimingData();

   delete pmesh;

   return 0;
}
