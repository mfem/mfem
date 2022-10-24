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
   double kin_vis = 1.0;
} ctx;

void vel_ic(const Vector &c, Vector &u)
{
   double x = c(0);
   double y = c(1);
   double z = 0.0;

   double amp = 0.2, ran = 0.0;

   // ran = 3.e4*(x*sin(y)+z*cos(y));
   // ran = 6.e3*sin(ran);
   // ran = 3.e3*sin(ran);
   // ran = cos(ran);
   // u(0) = 1. + ran*amp;

   // ran = (2+ran)*1.e4*(y*sin(z)+x*cos(z));
   // ran = 2.e3*sin(ran);
   // ran = 7.e3*sin(ran);
   // ran = cos(ran);
   // u(1) = ran*amp;

   u(0) = amp * sin(y);
   u(1) = amp * cos(x);
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

   int serial_refinements = 0;
   int nx = 1;
   int ny = 1;

   Mesh mesh = Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL);

   mesh.EnsureNodes();
   H1_FECollection mesh_fec(ctx.order, mesh.Dimension());
   FiniteElementSpace mesh_fes(&mesh, &mesh_fec, mesh.Dimension());
   mesh.SetNodalFESpace(&mesh_fes);

   auto *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);

   // Create the flow solver.
   NavierSolver navier(pmesh, ctx.order, ctx.kin_vis);

   // Set the initial condition.
   ParGridFunction *u_gf = navier.GetCurrentVelocity();
   VectorFunctionCoefficient u_ic_coeff(pmesh->Dimension(), vel_ic);
   u_gf->ProjectCoefficient(u_ic_coeff);

   navier.SetFilterAlpha(0.1);
   navier.Setup(1.0);

   auto hpfrt_gf = navier.BuildHPFForcing(*u_gf);

   ParaViewDataCollection pvdc("output/filter", pmesh);
   pvdc.SetDataFormat(VTKFormat::BINARY32);
   pvdc.SetHighOrderOutput(true);
   pvdc.SetLevelsOfDetail(ctx.order);
   pvdc.SetCycle(0);
   pvdc.SetTime(0);
   pvdc.RegisterField("velocity", u_gf);
   pvdc.RegisterField("filtered", hpfrt_gf);
   pvdc.Save();

   delete pmesh;

   return 0;
}
