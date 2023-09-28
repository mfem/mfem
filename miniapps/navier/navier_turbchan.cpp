// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// DNS of a channel flow at Re_tau = 180 (variable). A detailed description of
// the test case can be found at [1]. Like described in the reference, the
// initial condition is based on the Reichardt function.
//
// [1] https://how5.cenaero.be/content/ws2-les-plane-channel-ret550

#include "navier_solver.hpp"
#include <fstream>
#include <cmath>

using namespace mfem;
using namespace navier;

struct s_NavierContext
{
   int order = 5;
   double Re_tau = 180.0;
   double kin_vis = 1.0 / Re_tau;
   double t_final = 50.0;
   double dt = -1.0;
} ctx;

double mesh_stretching_func(const double y)
{
   double C = 1.8;
   double delta = 1.0;

   return delta * tanh(C * (2.0 * y - 1.0)) / tanh(C);
}

void accel(const Vector &x, double t, Vector &f)
{
   f(0) = 1.0;
   f(1) = 0.0;
   f(2) = 0.0;
}

void vel_ic_reichardt(const Vector &coords, double t, Vector &u)
{
   double yp;
   double x = coords(0);
   double y = coords(1);
   double z = coords(2);

   double C = 5.17;
   double k = 0.4;
   double eps = 1e-2;

   if (y < 0)
   {
      yp = (1.0 + y) * ctx.Re_tau;
   }
   else
   {
      yp = (1.0 - y) * ctx.Re_tau;
   }

   u(0) = 1.0 / k * log(1.0 + k * yp) + (C - (1.0 / k) * log(k)) * (1 - exp(
                                                                       -yp / 11.0) - yp / 11.0 * exp(-yp / 3.0));

   double kx = 23.0;
   double kz = 13.0;

   double alpha = kx * 2.0 * M_PI / 2.0 * M_PI;
   double beta = kz * 2.0 * M_PI / M_PI;

   u(0) += eps * beta * sin(alpha * x) * cos(beta * z);
   u(1) = eps * sin(alpha * x) * sin(beta * z);
   u(2) = -eps * alpha * cos(alpha * x) * sin(beta * z);
}

void vel_wall(const Vector &x, double t, Vector &u)
{
   u(0) = 0.0;
   u(1) = 0.0;
   u(2) = 0.0;
}

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   double Lx = 2.0 * M_PI;
   double Ly = 1.0;
   double Lz = M_PI;

   int N = ctx.order + 1;
   int NL = static_cast<int>(std::round(64.0 / N)); // Coarse
   // int NL = std::round(96.0 / N); // Baseline
   // int NL = std::round(128.0 / N); // Fine
   double LC = M_PI / NL;
   int NX = 2 * NL;
   int NY = 2 * static_cast<int>(std::round(48.0 / N));
   int NZ = NL;

   Mesh mesh = Mesh::MakeCartesian3D(NX, NY, NZ, Element::HEXAHEDRON, Lx, Ly, Lz);

   for (int i = 0; i < mesh.GetNV(); ++i)
   {
      double *v = mesh.GetVertex(i);
      v[1] = mesh_stretching_func(v[1]);
   }

   // Create translation vectors defining the periodicity
   Vector x_translation({Lx, 0.0, 0.0});
   Vector z_translation({0.0, 0.0, Lz});
   std::vector<Vector> translations = {x_translation, z_translation};

   // Create the periodic mesh using the vertex mapping defined by the translation vectors
   Mesh periodic_mesh = Mesh::MakePeriodic(mesh,
                                           mesh.CreatePeriodicVertexMapping(translations));

   if (Mpi::Root())
   {
      printf("NL=%d NX=%d NY=%d NZ=%d dx+=%f\n", NL, NX, NY, NZ, LC * ctx.Re_tau);
      std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
   }

   double hmin, hmax, kappa_min, kappa_max;
   periodic_mesh.GetCharacteristics(hmin, hmax, kappa_min, kappa_max);

   double umax = 22.0;
   ctx.dt = 1.0 / pow(ctx.order, 1.5) * hmin / umax;

   auto *pmesh = new ParMesh(MPI_COMM_WORLD, periodic_mesh);

   // Create the flow solver.
   NavierSolver flowsolver(pmesh, ctx.order, ctx.kin_vis);
   flowsolver.EnablePA(true);

   // Set the initial condition.
   ParGridFunction *u_gf = flowsolver.GetCurrentVelocity();
   ParGridFunction *p_gf = flowsolver.GetCurrentPressure();

   VectorFunctionCoefficient u_ic_coef(pmesh->Dimension(), vel_ic_reichardt);
   u_gf->ProjectCoefficient(u_ic_coef);

   Array<int> domain_attr(pmesh->attributes);
   domain_attr = 1;
   flowsolver.AddAccelTerm(accel, domain_attr);

   Array<int> attr(pmesh->bdr_attributes.Max());
   attr[1] = 1;
   attr[3] = 1;
   flowsolver.AddVelDirichletBC(vel_wall, attr);

   double t = 0.0;
   double dt = ctx.dt;
   double t_final = ctx.t_final;
   bool last_step = false;

   flowsolver.Setup(dt);

   ParaViewDataCollection pvdc("turbchan", pmesh);
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

      flowsolver.Step(t, dt, step);

      if (step % 1000 == 0)
      {
         pvdc.SetCycle(step);
         pvdc.SetTime(t);
         pvdc.Save();
      }

      if (t > 5.0)
      {
         dt = 1e-2;
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
