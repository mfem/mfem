// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

#include "incompressible_navier_solver.hpp"
#include <fstream>

using namespace mfem;
using namespace incompressible_navier;

void vel(const Vector &x, real_t t, Vector &u)
{
   real_t xi = x(0);
   real_t yi = x(1);

   u = 0.0;
}

void vel_inlet(const Vector &x, real_t t, Vector &u)
{
   u = 0.0;
   if (x(0) < 0.001)
   {

      u(0) = -0.01 * (std::pow(x(1) - 0.5, 2.0) - 0.25);
   }
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   int serial_refinements = 1;
   int vOrder = 2;
   int pOrder = 1;
   int tOrder = 1;
   real_t kin_vis = 1.0;
   real_t dt = 1e-3;
   real_t t = 0.0;
   real_t t_final = 0.1;
   bool last_step = false;

   //Mesh *mesh = new Mesh("box-cylinder.mesh");
   Mesh mesh = Mesh::MakeCartesian2D(90, 30, mfem::Element::QUADRILATERAL, true, 3,
                                     1);

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
   IncompressibleNavierSolver flowsolver(pmesh, vOrder, pOrder, tOrder, kin_vis);
   flowsolver.EnablePA(false);

   // // Set the initial condition.
   // ParGridFunction *u_ic = flowsolver.GetCurrentVelocity();
   // VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel);
   // u_ic->ProjectCoefficient(u_excoeff);

   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the mesh.
   Array<int> attr(pmesh->bdr_attributes.Max());           attr = 0;
   Array<int> attr_inlet(pmesh->bdr_attributes.Max());     attr_inlet = 0;
   // Inlet is attribute 1.
   attr[0] = 1;
   // Walls is attribute 3.
   attr[2] = 1;
   flowsolver.AddVelDirichletBC(vel, attr);

   attr_inlet[3] = 1;
   flowsolver.AddVelDirichletBC(vel_inlet, attr_inlet);

   flowsolver.Setup(dt);

   ParGridFunction *u_gf = flowsolver.GetCurrentVelocity();
   ParGridFunction *p_gf = flowsolver.GetCurrentPressure();
   ParGridFunction *psi_gf = flowsolver.GetCurrentPsi();

   ParaViewDataCollection pvdc("3dfoc", pmesh);
   pvdc.SetDataFormat(VTKFormat::BINARY32);
   //pvdc.SetHighOrderOutput(true);
   pvdc.SetCycle(0);
   pvdc.SetTime(t);
   pvdc.RegisterField("velocity", u_gf);
   pvdc.RegisterField("pressure", p_gf);
   pvdc.RegisterField("psi", psi_gf);
   pvdc.Save();

   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      flowsolver.Step(t, dt, step);

      if (step % 1 == 0)
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

   // flowsolver.PrintTimingData();

   delete pmesh;

   return 0;
}
