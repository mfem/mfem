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
// Navier MMS example
//
// A manufactured solution is defined as
//
// u = [pi * sin(t) * sin(pi * x)^2 * sin(2 * pi * y),
//      -(pi * sin(t) * sin(2 * pi * x)) * sin(pi * y)^2].
//
// p = cos(pi * x) * sin(t) * sin(pi * y)
//
// The solution is used to compute the symbolic forcing term (right hand side),
// of the equation. Then the numerical solution is computed and compared to the
// exact manufactured solution to determine the error.

#include "navier_solver.hpp"
#include <fstream>

using namespace mfem;
using namespace navier;

struct s_NavierContext
{
   int ser_ref_levels = 1;
   int order = 6;
   double kinvis = 0.025;
   double t_final = 100 * 0.25e-4;
   double dt = 0.25e-4;
   bool visualization = false;
   bool checkres = false;
} ctx;

double kinvis_mms(const Vector &c)
{
   double x = c(0);
   double y = c(1);

   return 0.025*(2.0+cos(x)*sin(y));
   // return 0.025;
}

Vector dkinvis_mmsdx(const Vector &c)
{
   double x = c(0);
   double y = c(1);
   Vector dkvdx(2);
   dkvdx(0) = -(sin(x)*sin(y))*0.025;
   dkvdx(1) = cos(x)*cos(y)*0.025;
   // dkvdx(0) = 0.0;
   // dkvdx(1) = 0.0;
   return dkvdx;
}

void vel(const Vector &c, double t, Vector &u)
{
   double x = c(0);
   double y = c(1);

   u(0) = cos(x)*sin(t)*sin(y);
   u(1) = -(cos(y)*sin(t)*sin(x));
}

double p(const Vector &c, double t)
{
   double x = c(0);
   double y = c(1);

   return cos(x)*sin(t)*sin(y);
}

void accel(const Vector &c, double t, Vector &u)
{
   double x = c(0);
   double y = c(1);

   u(0) = cos(x)*pow(cos(y),2)*pow(sin(t),
                                   2)*sin(x) + cos(t)*cos(x)*sin(y) - sin(t)*sin(x)*sin(y) - cos(x)*pow(sin(t),
                                         2)*sin(x)*pow(sin(y),2) + 2*cos(x)*sin(t)*sin(y)*kinvis_mms(c) + 2*sin(t)*sin(
             x)*sin(y)*dkinvis_mmsdx(c)[0];

   u(1) = cos(x)*cos(y)*sin(t) - cos(t)*cos(y)*sin(x) + pow(cos(x),
                                                            2)*cos(y)*pow(sin(t),2)*sin(y) - cos(y)*pow(sin(t),2)*pow(sin(x),
                                                                  2)*sin(y) - 2*cos(y)*sin(t)*sin(x)*kinvis_mms(c) - 2*sin(t)*sin(x)*sin(
             y)*dkinvis_mmsdx(c)[1];
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

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);

   // Create the flow solver.
   NavierSolver naviersolver(pmesh, ctx.order, ctx.kinvis);

   auto kv_gf = naviersolver.GetVariableViscosity();
   FunctionCoefficient kv_coeff(kinvis_mms);
   kv_gf->ProjectCoefficient(kv_coeff);

   // Set the initial condition.
   ParGridFunction *u_ic = naviersolver.GetCurrentVelocity();
   VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel);
   u_ic->ProjectCoefficient(u_excoeff);

   FunctionCoefficient p_excoeff(p);

   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the mesh.
   Array<int> attr(pmesh->bdr_attributes.Max());
   attr = 1;
   naviersolver.AddVelDirichletBC(vel, attr);

   Array<int> domain_attr(pmesh->attributes.Max());
   domain_attr = 1;
   naviersolver.AddAccelTerm(accel, domain_attr);

   double t = 0.0;
   double dt = ctx.dt;
   double t_final = ctx.t_final;
   bool last_step = false;

   naviersolver.Setup(dt);

   double err_u = 0.0;
   double err_p = 0.0;
   ParGridFunction *u_gf = nullptr;
   ParGridFunction *p_gf = nullptr;
   u_gf = naviersolver.GetCurrentVelocity();
   p_gf = naviersolver.GetCurrentPressure();

   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      naviersolver.Step(t, dt, step);

      // Compare against exact solution of velocity and pressure.
      u_excoeff.SetTime(t);
      p_excoeff.SetTime(t);
      err_u = u_gf->ComputeL2Error(u_excoeff);
      err_p = p_gf->ComputeL2Error(p_excoeff);

      if (Mpi::Root())
      {
         printf("%11s %11s %11s %11s\n", "Time", "dt", "err_u", "err_p");
         printf("%.5E %.5E %.5E %.5E err\n", t, dt, err_u, err_p);
         fflush(stdout);
      }
   }

   if (ctx.visualization)
   {
      char vishost[] = "128.15.198.77";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << Mpi::WorldSize() << " "
               << Mpi::WorldRank() << "\n";
      sol_sock << "solution\n" << *pmesh << *kv_gf << std::flush;
   }

   naviersolver.PrintTimingData();

   // Test if the result for the test run is as expected.
   if (ctx.checkres)
   {
      double tol = 1e-3;
      if (err_u > tol || err_p > tol)
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
