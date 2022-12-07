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
#include <fstream>

using namespace mfem;
using namespace navier;

struct s_NavierContext
{
   int order = 4;
   double kin_vis = 1.0 / 36000.0;
   int n_steps = 1000;
   double dt = 1e-6;
   double dt_min = 5e-3;
   double t_final = 200.0;
   double xmin = -130.0;
   double Uavg = 1.0;
   double h = 1.0;
} ctx;

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

   u(0) = ctx.Uavg;
   u(1) = 0.0;
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

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   int serial_refinements = 0;

   Mesh mesh("tests/bfs2d.mesh");
   mesh.EnsureNodes();

   for (int i = 0; i < serial_refinements; ++i)
   {
      mesh.UniformRefinement();
   }

   if (Mpi::Root())
   {
      std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
   }

   auto *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);

   // Create the flow solver.
   NavierSolver flowsolver(pmesh, ctx.order, ctx.kin_vis);

   auto kv_gf = flowsolver.GetVariableViscosity();
   ConstantCoefficient kv_coeff(ctx.kin_vis);
   kv_gf->ProjectCoefficient(kv_coeff);

   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the mesh.
   Array<int> inlet_attr(pmesh->bdr_attributes.Max());
   inlet_attr = 0;
   inlet_attr[0] = 1;
   inlet_attr[2] = 1;

   Array<int> wall_attr(pmesh->bdr_attributes.Max());
   wall_attr = 0;
   wall_attr[3] = 1;
   wall_attr[4] = 1;

   Array<int> outlet_attr(pmesh->bdr_attributes.Max());
   outlet_attr = 0;
   outlet_attr[1] = 1;

   // Set the initial condition.
   ParGridFunction *u_ic = flowsolver.GetCurrentVelocity();
   VectorFunctionCoefficient u_ic_coeff(pmesh->Dimension(), vel_ic);
   u_ic->ProjectCoefficient(u_ic_coeff);

   // Inlet
   VectorFunctionCoefficient u_inlet_coeff(pmesh->Dimension(), vel_inlet);
   u_ic->ProjectBdrCoefficient(u_inlet_coeff, inlet_attr);

   // Walls
   VectorFunctionCoefficient velocity_wall_coeff(pmesh->Dimension(),
                                                 velocity_wall);
   u_ic->ProjectBdrCoefficient(velocity_wall_coeff, wall_attr);

   flowsolver.AddVelDirichletBC(vel_inlet, inlet_attr);
   flowsolver.AddVelDirichletBC(velocity_wall, wall_attr);

   // ConstantCoefficient pres_outlet(0.0);
   // flowsolver.AddPresDirichletBC(&pres_outlet, outlet_attr);

   double t = 0.0;
   double dt = ctx.dt;
   double t_final = ctx.t_final;
   bool last_step = false;

   flowsolver.Setup(dt);

   ParGridFunction *u_next_gf = nullptr;
   ParGridFunction *u_gf = flowsolver.GetCurrentVelocity();
   ParGridFunction *p_gf = flowsolver.GetCurrentPressure();

   double cfl_max = 0.5;
   double cfl_tol = 1e-4;

   // ParaViewDataCollection pvdc("3dfoc", pmesh);
   // pvdc.SetDataFormat(VTKFormat::BINARY32);
   // pvdc.SetHighOrderOutput(true);
   // pvdc.SetLevelsOfDetail(ctx.order);
   // pvdc.SetCycle(0);
   // pvdc.SetTime(t);
   // pvdc.RegisterField("velocity", u_gf);
   // pvdc.RegisterField("pressure", p_gf);
   // pvdc.Save();

   char vishost[] = "128.15.198.77";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock << "parallel " << num_procs << " " << myid << "\n";
   sol_sock.precision(8);
   sol_sock << "solution\n" << *pmesh << *u_gf << "\n"
            << "keys rRjl\n"
            << "pause\n" << std::flush;

   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      flowsolver.Step(t, dt, step, true);

      // Retrieve the computed provisional velocity
      u_next_gf = flowsolver.GetProvisionalVelocity();

      // Compute the CFL based on the provisional velocity
      double cfl = flowsolver.ComputeCFL(*u_next_gf, dt);
      double error_est = cfl / (cfl_max + cfl_tol);
      if (error_est >= 1.0)
      {
         // Reject the time step
         if (Mpi::Root())
         {
            std::cout
                  << "Step reached maximum CFL, retrying with smaller step size..."
                  << std::endl;
         }
         dt *= 0.5;
         step -= 1;
      }
      else
      {
         // Accept the time step
         t += dt;

         // Predict new step size
         double fac_safety = 2.0;
         double eta = pow(1.0 / (fac_safety * error_est), 1.0 / (1.0 + 3.0));
         double fac_min = 0.1;
         double fac_max = 1.4;
         dt = dt * std::min(fac_max, std::max(fac_min, eta));

         if (dt < 1e-8)
         {
            MFEM_ABORT("Minimum timestep reached, likely unstable.");
         }

         dt = std::min(dt, ctx.dt_min);

         // Queue new time step in the history array
         flowsolver.UpdateTimestepHistory(dt);
      }

      if (step % 100 == 0)
      {
         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock.precision(8);
         sol_sock << "solution\n" << *pmesh << *u_gf << std::flush;
      }

      if (Mpi::Root())
      {
         printf("%11s %11s %11s\n", "Time", "dt", "CFL");
         printf("%.5E %.5E %.5E\n", t, dt, cfl);
         fflush(stdout);
      }
   }

   flowsolver.PrintTimingData();

   delete pmesh;

   return 0;
}
