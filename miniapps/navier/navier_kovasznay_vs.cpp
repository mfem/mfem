// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
// Navier Kovasznay example with variable time step
//
// Solve for the steady Kovasznay flow at Re = 40 defined by
//
// u = [1 - exp(L * x) * cos(2 * pi * y),
//      L / (2 * pi) * exp(L * x) * sin(2 * pi * y)],
//
// p = 1/2 * (1 - exp(2 * L * x)),
//
// with L = Re/2 - sqrt(Re^2/4 + 4 * pi^2).
//
// The problem domain is set up like this
//
//            +-------------+
//            |             |
//            |             |
//            |             |
//            |             |
//  Inflow -> |             | -> Outflow
//            |             |
//            |             |
//            |             |
//            |             |
//            |             |
//            +-------------+
//
// and Dirichlet boundary conditions are applied for the velocity on every
// boundary. The problem, although steady state, is time integrated up to the
// final time and the solution is compared with the known exact solution.
//
// Additionally, this example shows the usage of variable time steps with
// Navier. A basic sample algorithm for determining the next time step is
// provided based on a CFL restriction on the velocity.

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
   double dt = 0.001;
   double reference_pressure = 0.0;
   double reynolds = 1.0 / kinvis;
   double lam = 0.5 * reynolds
                - sqrt(0.25 * reynolds * reynolds + 4.0 * M_PI * M_PI);
   bool pa = true;
   bool ni = false;
   bool visualization = false;
   bool checkres = false;
} ctx;

void vel_kovasznay(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   u(0) = 1.0 - exp(ctx.lam * xi) * cos(2.0 * M_PI * yi);
   u(1) = ctx.lam / (2.0 * M_PI) * exp(ctx.lam * xi) * sin(2.0 * M_PI * yi);
}

double pres_kovasznay(const Vector &x, double t)
{
   double xi = x(0);

   return 0.5 * (1.0 - exp(2.0 * ctx.lam * xi)) + ctx.reference_pressure;
}

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

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
      if (mpi.Root())
      {
         args.PrintUsage(mfem::out);
      }
      return 1;
   }
   if (mpi.Root())
   {
      args.PrintOptions(mfem::out);
   }

   Mesh *mesh = new Mesh(2, 4, Element::QUADRILATERAL, false, 1.5, 2.0);

   mesh->EnsureNodes();
   GridFunction *nodes = mesh->GetNodes();
   *nodes -= 0.5;

   for (int i = 0; i < ctx.ser_ref_levels; ++i)
   {
      mesh->UniformRefinement();
   }

   if (mpi.Root())
   {
      std::cout << "Number of elements: " << mesh->GetNE() << std::endl;
   }

   auto *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // Create the flow solver.
   NavierSolver flowsolver(pmesh, ctx.order, ctx.kinvis);
   flowsolver.EnablePA(ctx.pa);
   flowsolver.EnableNI(ctx.ni);

   // Set the initial condition.
   ParGridFunction *u_ic = flowsolver.GetCurrentVelocity();
   VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel_kovasznay);
   u_ic->ProjectCoefficient(u_excoeff);

   FunctionCoefficient p_excoeff(pres_kovasznay);

   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the mesh.
   Array<int> attr(pmesh->bdr_attributes.Max());
   attr = 1;
   flowsolver.AddVelDirichletBC(vel_kovasznay, attr);

   double t = 0.0;
   double dt = ctx.dt;
   double t_final = ctx.t_final;
   bool last_step = false;

   flowsolver.Setup(dt);

   double err_u = 0.0;
   double err_p = 0.0;
   ParGridFunction *u_next_gf = nullptr;
   ParGridFunction *u_gf = nullptr;
   ParGridFunction *p_gf = nullptr;

   ParGridFunction p_ex_gf(flowsolver.GetCurrentPressure()->ParFESpace());
   GridFunctionCoefficient p_ex_gf_coeff(&p_ex_gf);

   double cfl_max = 0.8;
   double cfl_tol = 1e-4;

   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      // Take a provisional step
      flowsolver.Step(t, dt, step, true);

      // Retrieve the computed provisional velocity
      u_next_gf = flowsolver.GetProvisionalVelocity();

      // Compute the CFL based on the provisional velocity
      double cfl = flowsolver.ComputeCFL(*u_next_gf, dt);

      double error_est = cfl / (cfl_max + cfl_tol);
      if (error_est >= 1.0)
      {
         // Reject the time step
         if (mpi.Root())
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

         // Queue new time step in the history array
         flowsolver.UpdateTimestepHistory(dt);
      }

      u_gf = flowsolver.GetCurrentVelocity();
      p_gf = flowsolver.GetCurrentPressure();

      u_excoeff.SetTime(t);
      p_excoeff.SetTime(t);

      // Remove mean value from exact pressure solution.
      p_ex_gf.ProjectCoefficient(p_excoeff);
      flowsolver.MeanZero(p_ex_gf);

      err_u = u_gf->ComputeL2Error(u_excoeff);
      err_p = p_gf->ComputeL2Error(p_ex_gf_coeff);

      if (mpi.Root())
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
   }

   if (ctx.visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "parallel " << mpi.WorldSize() << " " << mpi.WorldRank()
               << "\n";
      sol_sock << "solution\n" << *pmesh << *u_ic << std::flush;
   }

   flowsolver.PrintTimingData();

   // Test if the result for the test run is as expected.
   if (ctx.checkres)
   {
      double tol_u = 1e-6;
      double tol_p = 1e-5;
      if (err_u > tol_u || err_p > tol_p)
      {
         if (mpi.Root())
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
