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

#include "navier_solver.hpp"
#include <fstream>

using namespace mfem;
using namespace navier;

struct s_NavierContext
{
   int ser_ref_levels = 1;
   int order = 6;
   double kinvis = 1.0 / 40.0;
   double t_final = 10 * 0.001;
   double dt = 1.0e-3;
   double reference_pressure = 0.0;
   bool visualization = false;
   bool checkres = false;
} ctx;

double nu_tg(const Vector &c)
{
   double x = c(0);
   double y = c(1);
   return (2 + sin(x) * cos(y)) / 40.0;
   // return ctx.kinvis;
}

double F_tg(const Vector &x, const double t)
{
   return exp(-2.0 * nu_tg(x) * t);
}

void vel_tg(const Vector &c, double t, Vector &u)
{
   double x = c(0);
   double y = c(1);
   u(0) = cos(x) * sin(y) * F_tg(c, t);
   u(1) = -sin(x) * cos(y) * F_tg(c, t);
}

double pres_tg(const Vector &c, double t)
{
   double x = c(0);
   double y = c(1);
   return -0.25 * (cos(2.0*x) + cos(2.0*y)) * pow(F_tg(c, t),
                                                  2.0) + ctx.reference_pressure;
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

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
   args.AddOption(&ctx.visualization,
                  "-vis",
                  "--visualization",
                  "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
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

   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, false, 2*M_PI,
                                     2.0*M_PI);

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

   auto kv_gf = flowsolver.GetVariableViscosity();
   FunctionCoefficient kv_coeff(nu_tg);
   kv_gf->ProjectCoefficient(kv_coeff);

   // Set the initial condition.
   ParGridFunction *u_ic = flowsolver.GetCurrentVelocity();
   VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel_tg);
   u_ic->ProjectCoefficient(u_excoeff);

   FunctionCoefficient p_excoeff(pres_tg);

   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the mesh.
   Array<int> attr(pmesh->bdr_attributes.Max());
   attr = 1;
   flowsolver.AddVelDirichletBC(vel_tg, attr);

   double t = 0.0;
   double dt = ctx.dt;
   double t_final = ctx.t_final;
   bool last_step = false;

   flowsolver.Setup(dt);

   double err_u = 0.0;
   double err_p = 0.0;
   ParGridFunction *u_gf = nullptr;
   ParGridFunction *p_gf = nullptr;

   ParGridFunction u_ex_gf(flowsolver.GetCurrentVelocity()->ParFESpace());
   ParGridFunction p_ex_gf(flowsolver.GetCurrentPressure()->ParFESpace());
   GridFunctionCoefficient p_ex_gf_coeff(&p_ex_gf);

   char vishost[] = "128.15.198.77";
   int visport = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);

   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      flowsolver.Step(t, dt, step);

      // Compare against exact solution of velocity and pressure.
      u_gf = flowsolver.GetCurrentVelocity();
      p_gf = flowsolver.GetCurrentPressure();

      u_excoeff.SetTime(t);
      p_excoeff.SetTime(t);

      // Remove mean value from exact pressure solution.
      p_ex_gf.ProjectCoefficient(p_excoeff);
      flowsolver.MeanZero(p_ex_gf);

      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(flowsolver.gll_rules.Get(i, 2 * ctx.order - 1));
      }

      err_u = u_gf->ComputeL2Error(u_excoeff, irs);
      err_u = sqrt((err_u*err_u) / flowsolver.volume);

      err_p = p_gf->ComputeL2Error(p_ex_gf_coeff);

      double cfl = flowsolver.ComputeCFL(*u_gf, dt);

      if (Mpi::Root())
      {
         printf("%5s %8s %8s %8s %11s %11s\n",
                "Order",
                "CFL",
                "Time",
                "dt",
                "err_u",
                "err_p");
         printf("%5.2d %8.2E %.2E %.2E %.5E %.5E err\n",
                ctx.order,
                cfl,
                t,
                dt,
                err_u,
                err_p);
         fflush(stdout);
      }

      {
         sol_sock << "parallel " << Mpi::WorldSize() << " "
                  << Mpi::WorldRank() << "\n";
         sol_sock << "solution\n" << *pmesh << *u_gf << std::flush;
      }
   }

   flowsolver.PrintTimingData();

   // Test if the result for the test run is as expected.
   if (ctx.checkres)
   {
      double tol_u = 1e-6;
      double tol_p = 1e-5;
      if (err_u > tol_u || err_p > tol_p)
      {
         if (Mpi::Root())
         {
            mfem::out << "Result has a larger error than expected."
                      << std::endl;
         }
         return -1;
      }
   }

   delete pmesh;

   return 0;
}
