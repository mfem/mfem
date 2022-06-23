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
   int order = 4;
   double kinvis = 1.0 / 40.0;
   double t_final = 1.0;
   double dt = 1e-2;
   bool pa = true;
   bool ni = false;
   bool visualization = false;
   bool checkres = false;
   double L = 1.0;
   double A = 0.08;
   double tg = t_final / 10.0;
} ctx;

void vel(const Vector &coords, double t, Vector &u)
{
   double x = coords(0);
   double y = coords(1);
   double e = exp(-4.0 * ctx.kinvis * M_PI * M_PI * t);

   u(0) = -sin(2.0 * M_PI * y) * e;
   u(1) = sin(2.0 * M_PI * x) * e;
}

double p(const Vector &coords, double t)
{
   double x = coords(0);
   double y = coords(1);
   double e = exp(-8.0 * ctx.kinvis * M_PI * M_PI * t);

   return -cos(2.0 * M_PI * x) * cos(2.0 * M_PI * y) * e;
}

void accel(const Vector &x, double t, Vector &u)
{
   u(0) = 0.0;
   u(1) = 0.0;
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

   // Mesh *mesh = new Mesh("../../data/periodic-square.mesh");
   Mesh *mesh = new Mesh("../../data/inline-quad.mesh");
   mesh->SetCurvature(ctx.order);
   GridFunction *nodes = mesh->GetNodes();
   *nodes -= ctx.L / 2.0;

   for (int i = 0; i < ctx.ser_ref_levels; ++i)
   {
      mesh->UniformRefinement();
   }

   if (mpi.Root())
   {
      std::cout << "Number of elements: " << mesh->GetNE() << std::endl;
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // Create the flow solver.
   NavierSolver naviersolver(pmesh, ctx.order, ctx.kinvis);
   naviersolver.EnablePA(ctx.pa);
   naviersolver.EnableNI(ctx.ni);

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

   // naviersolver.SetMaxBDFOrder(1);

   naviersolver.Setup(dt);

   double errl2_u = 0.0;
   double errlinf_u = 0.0;
   double errl2_p = 0.0;
   double errlinf_p = 0.0;
   ParGridFunction *u_gf = nullptr;
   ParGridFunction *p_gf = nullptr;
   ParGridFunction *u_next_gf = nullptr;
   GridFunction *wg_gf = nullptr;
   u_next_gf = naviersolver.GetProvisionalVelocity();
   u_next_gf->ProjectCoefficient(u_excoeff);
   u_gf = naviersolver.GetCurrentVelocity();
   p_gf = naviersolver.GetCurrentPressure();
   wg_gf = naviersolver.GetCurrentMeshVelocity();

   // This is just to initialize some datastructres in navier
   p_gf->ProjectCoefficient(p_excoeff);
   naviersolver.MeanZero(*p_gf);
   *p_gf = 0.0;

   ParGridFunction u_ex(*u_gf);
   ParGridFunction p_ex(*p_gf);

   auto userchk = [&](int step)
   {
      u_ex.ProjectCoefficient(u_excoeff);
      for (int i = 0; i < u_ex.Size(); ++i)
      {
         u_ex[i] = (*u_next_gf)[i] - u_ex[i];
      }

      p_ex.ProjectCoefficient(p_excoeff);
      for (int i = 0; i < p_ex.Size(); ++i)
      {
         p_ex[i] = (*p_gf)[i] - p_ex[i];
      }

      // Compare against exact solution of velocity and pressure.
      errl2_u = naviersolver.NekNorm(u_ex, 0, true);
      errl2_p = naviersolver.NekNorm(p_ex, 0, false);
      errlinf_u = naviersolver.NekNorm(u_ex, 1, true);
      errlinf_p = naviersolver.NekNorm(p_ex, 1, false);

      if (mpi.Root())
      {
         if (step == -1)
         {
            printf("%18s %18s %18s %18s %18s %18s errlog\n",
                   "time",
                   "dt",
                   "errl2_u",
                   "errl2_p",
                   "errlinf_u",
                   "errlinf_p");
         }
         printf("%18.12E %18.12E %18.12E %18.12E %18.12E %18.12E errlog\n",
                t,
                dt,
                errl2_u,
                errl2_p,
                errlinf_u,
                errlinf_p);
         fflush(stdout);
      }
   };

   userchk(-1);

   auto xi_0 = new GridFunction(*pmesh->GetNodes());

   VectorFunctionCoefficient
   mesh_nodes(2, [&](const Vector &cin, double t, Vector &cout)
   {
      double x = cin(0);
      double y = cin(1);
      cout(0) = x
                + ctx.A * sin((2.0 * M_PI * t) / ctx.tg)
                * sin((2.0 * M_PI * (ctx.L / 2. + y)) / ctx.L);
      cout(1) = y
                + ctx.A * sin((2.0 * M_PI * t) / ctx.tg)
                * sin((2.0 * M_PI * (ctx.L / 2. + x)) / ctx.L);
   });

   VectorFunctionCoefficient
   mesh_nodes_velocity(2, [&](const Vector &cin, double t, Vector &cout)
   {
      double x = cin(0);
      double y = cin(1);
      cout(0) = (2.0 * ctx.A * M_PI * cos((2.0 * M_PI * t) / ctx.tg)
                 * sin((2.0 * M_PI * (ctx.L / 2. + y)) / ctx.L))
                / ctx.tg;
      cout(1) = (2.0 * ctx.A * M_PI * cos((2.0 * M_PI * t) / ctx.tg)
                 * sin((2.0 * M_PI * (ctx.L / 2. + x)) / ctx.L))
                / ctx.tg;
   });

   ParaViewDataCollection paraview_dc("navier_ale_mms", pmesh);
   paraview_dc.SetLevelsOfDetail(ctx.order);
   paraview_dc.SetCycle(0);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetTime(t);
   paraview_dc.RegisterField("velocity", u_gf);
   paraview_dc.RegisterField("pressure", p_gf);
   paraview_dc.RegisterField("velocity_error", &u_ex);
   paraview_dc.RegisterField("pressure_error", &p_ex);

   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      *pmesh->GetNodes() = *xi_0;

      mesh_nodes_velocity.SetTime(t + dt);
      wg_gf->ProjectCoefficient(mesh_nodes_velocity);

      mesh_nodes.SetTime(t + dt);
      naviersolver.TransformMesh(mesh_nodes);

      naviersolver.Step(t, dt, step, true);
      t += dt;

      u_excoeff.SetTime(t);
      p_excoeff.SetTime(t);

      if (step < 5)
      {
         u_next_gf->ProjectCoefficient(u_excoeff);
         p_gf->ProjectCoefficient(p_excoeff);
      }

      userchk(step);

      naviersolver.UpdateTimestepHistory(dt);

      paraview_dc.SetCycle(step + 1);
      paraview_dc.SetTime(t);
      paraview_dc.Save();
   }

   if (ctx.visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream u_sock(vishost, visport);
      u_sock << "parallel " << mpi.WorldSize() << " " << mpi.WorldRank()
             << "\n";
      u_sock << "solution\n" << *pmesh << u_ex << std::flush;

      socketstream p_sock(vishost, visport);
      p_sock << "parallel " << mpi.WorldSize() << " " << mpi.WorldRank()
             << "\n";
      p_sock << "solution\n" << *pmesh << p_ex << std::flush;
   }

   naviersolver.PrintTimingData();

   delete pmesh;

   return 0;
}
