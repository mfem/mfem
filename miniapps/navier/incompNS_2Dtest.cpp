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

#define NVTX_COLOR ::gpu::nvtx::kLawnGreen
#include "incompressible_navier_nvtx.hpp"

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
   if (x(0) < 0.001) { u(0) = -0.001 * (std::pow(x(1) - 0.5, 2.0) - 0.25); }
}

MFEM_EXPORT int navier(int argc, char *argv[], double &u, double &p, double &Ψ)
{
   NVTX();
   static mfem::MPI_Session mpi(argc, argv);
   const int myid = mpi.WorldRank();
   Hypre::Init();

   const char *device_config = "cpu";
   int serial_refinements = 1;
   int nx = 90, ny = 30;
   int v_order = 2;
   int p_order = 1;
   int t_order = 1;
   real_t kin_vis = 20.0;
   real_t dt = 1e-2;
   real_t t = 0.0;
   real_t t_final = 1.0;
   bool last_step = false;
   bool visualization = true;
   bool use_paraview = false;
   bool pa = false;
   int vis_steps = 100;
   int max_tsteps = -1;

   constexpr int precision = 8;
   std::cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&serial_refinements, "-sr", "--serial-refinements",
                  "Number serial refinements.");
   args.AddOption(&nx, "-nx", "--nx", "Number of elements in X.");
   args.AddOption(&ny, "-ny", "--ny", "Number of elements in Y.");
   args.AddOption(&v_order, "-vo", "--vo", "Order.");
   args.AddOption(&p_order, "-po", "--po", "Order.");
   args.AddOption(&t_order, "-to", "--to", "Order.");
   args.AddOption(&kin_vis, "-kv", "--kin-vis",
                  "Kineic viscosity coefficient.");
   args.AddOption(&dt, "-dt", "--time-step", "Initial time step size.");
   args.AddOption(&t_final, "-tf", "--t-final", "Final time; start time is 0.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly",
                  "Enable or disable partial assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&use_paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Use ParaView.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&max_tsteps, "-ms", "--max-steps",
                  "Maximum number of steps (negative means no restriction).");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(mfem::out); }
      return EXIT_FAILURE;
   }
   if (myid == 0) { args.PrintOptions(mfem::out); }

   // Mesh *mesh = new Mesh("box-cylinder.mesh");
   const real_t sx = 3.0, sy = 1.0;
   const bool generate_edges = true;
   const auto QUAD = Element::QUADRILATERAL;
   Mesh mesh = Mesh::MakeCartesian2D(nx, ny, QUAD, generate_edges, sx, sy);

   for (int i = 0; i < serial_refinements; ++i) { mesh.UniformRefinement(); }

   if (Mpi::Root())
   {
      std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
   }

   auto *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);

   // Create the flow solver.
   IncompressibleNavierSolver flowsolver(pmesh, v_order, p_order, t_order,
                                         kin_vis);
   flowsolver.EnablePA(pa);

   // // Set the initial condition.
   // ParGridFunction *u_ic = flowsolver.GetCurrentVelocity();
   // VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel);
   // u_ic->ProjectCoefficient(u_excoeff);

   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the mesh.
   Array<int> attr(pmesh->bdr_attributes.Max());
   attr = 0;
   Array<int> attr_inlet(pmesh->bdr_attributes.Max());
   attr_inlet = 0;
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
   if (use_paraview)
   {
      pvdc.SetDataFormat(VTKFormat::BINARY32);
      // pvdc.SetHighOrderOutput(true);
      pvdc.SetCycle(0);
      pvdc.SetTime(t);
      pvdc.RegisterField("velocity", u_gf);
      pvdc.RegisterField("pressure", p_gf);
      pvdc.RegisterField("psi", psi_gf);
      pvdc.Save();
   }

   for (int step = 0; !last_step; ++step)
   {
      if (step == max_tsteps) { last_step = true; }
      if (t + dt >= t_final - dt / 2) { last_step = true; }
      const bool vis_step = last_step || (step % vis_steps) == 0;

      flowsolver.Step(t, dt, step, vis_step);

      if (vis_step)
      {
         if (Mpi::Root() && vis_steps)
         {
            printf("%11s %11s\n", "Time", "dt");
            printf("%.5E %.5E\n", t, dt);
         }
         if (use_paraview)
         {
            pvdc.SetCycle(step);
            pvdc.SetTime(t);
            pvdc.Save();
         }
      }
   }

   // flowsolver.PrintTimingData();

   auto reduce = [](ParGridFunction *gf) -> real_t { return (*gf) * (*gf); };
   // auto reduce = [](ParGridFunction *gf) -> real_t { return gf->Norml2(); };
   u = reduce(u_gf), p = reduce(p_gf), Ψ = reduce(psi_gf);

   fflush(stdout);
   delete pmesh;

   return EXIT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
#ifndef MFEM_USE_CMAKE_TESTS
int main(int argc, char *argv[])
try
{
   double u, double p, double Ψ; // unused
   return navier(argc, argv, u, p, Ψ);
}
catch (std::exception &e)
{
   std::cerr << "\033[31m..xxxXXX[ERROR]XXXxxx.." << std::endl;
   std::cerr << "\033[31m{}" << e.what() << std::endl;
   return EXIT_FAILURE;
}
#endif // MFEM_USE_CMAKE_TESTS
