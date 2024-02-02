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
//
// Navier double shear layer example
//
// Solve the double shear problem in the following configuration.
//
//       +-------------------+
//       |                   |
//       |      u0 = ua      |
//       |                   |
//  -------------------------------- y = 0.5
//       |                   |
//       |      u0 = ub      |
//       |                   |
//       +-------------------+
//
// The initial condition u0 is chosen to be a varying velocity in the y
// direction. It includes a perturbation at x = 0.5 which leads to an
// instability and the dynamics of the flow. The boundary conditions are fully
// periodic.

#include "lib/navier_solver.hpp"
#include <fstream>

using namespace mfem;
using namespace navier;

struct s_NavierContext
{
   int order = 6;
   double kinvis = 1.0 / 100000.0;
   double t_final = 1.0;
   double dt = 1e-3;
   double max_elem_error = 5.0e-3;
   double hysteresis = 0.15; // derefinement safety coefficient
   int nc_limit = 3;
} ctx;

void vel_shear_ic(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   double rho = 30.0;
   double delta = 0.05;

   if (yi <= 0.5)
   {
      u(0) = tanh(rho * (yi - 0.25));
   }
   else
   {
      u(0) = tanh(rho * (0.75 - yi));
   }

   u(1) = delta * sin(2.0 * M_PI * xi);
}

class MagnitudeCoefficient : public Coefficient
{
public:
   MagnitudeCoefficient(const ParGridFunction &u) : u(u) {}

   double Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector val;
      u.GetVectorValue(T, ip, val);
      return val.Norml2();
   }

private:
   const ParGridFunction &u;
};

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   int serial_refinements = 2;

   OptionsParser args(argc, argv);
   args.AddOption(&ctx.order,
                  "-o",
                  "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ctx.dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&ctx.t_final, "-tf", "--final-time", "Final time.");

   args.AddOption(&ctx.max_elem_error, "-e", "--max-err",
                  "Maximum element error");
   args.AddOption(&ctx.hysteresis, "-y", "--hysteresis",
                  "Derefinement safety coefficient.");
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

   Mesh *mesh = new Mesh("../../data/periodic-square.mesh");
   mesh->EnsureNodes();
   mesh->EnsureNCMesh();
   GridFunction *nodes = mesh->GetNodes();
   *nodes -= -1.0;
   *nodes /= 2.0;

   for (int i = 0; i < serial_refinements; ++i)
   {
      mesh->UniformRefinement();
   }

   // Vertex target(0.0, 0.0, 0.0);
   // for (int l = 0; l < 1; l++)
   // {
   //    // mesh->RefineAtVertex(target);
   //    mesh->RandomRefinement(0.5, false, 1, 1);
   // }

   if (Mpi::Root())
   {
      std::cout << "Number of elements: " << mesh->GetNE() << std::endl;
   }

   auto *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // Create the flow solver.
   NavierSolver flowsolver(pmesh, ctx.order, ctx.kinvis);

   // Set the initial condition.
   ParGridFunction *u_ic = flowsolver.GetCurrentVelocity();
   VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel_shear_ic);
   u_ic->ProjectCoefficient(u_excoeff);

   double t = 0.0;
   double dt = ctx.dt;
   double t_final = ctx.t_final;
   bool last_step = false;

   flowsolver.SetMaxBDFOrder(2);
   flowsolver.Setup(dt);

   ParGridFunction *u_gf = flowsolver.GetCurrentVelocity();
   ParGridFunction *p_gf = flowsolver.GetCurrentPressure();

   MagnitudeCoefficient mag_coeff(*u_gf);
   ParGridFunction vel_mag_gf(*p_gf);

   auto estimator_integ = new DiffusionIntegrator();

   L2_FECollection flux_fec(ctx.order, 2);
   // auto flux_fes = new ParFiniteElementSpace(pmesh, &flux_fec, 2);
   auto flux_fes = new ParFiniteElementSpace(pmesh, p_gf->ParFESpace()->FEColl(),
                                             2);
   auto estimator = new ZienkiewiczZhuEstimator(
      *estimator_integ, vel_mag_gf,
      flux_fes);

   ThresholdRefiner refiner(*estimator);
   refiner.SetMaxElements(1000);
   refiner.SetTotalErrorFraction(0.0); // use purely local threshold
   refiner.SetLocalErrorGoal(ctx.max_elem_error);
   refiner.PreferConformingRefinement();
   refiner.SetNCLimit(ctx.nc_limit);

   ThresholdDerefiner derefiner(*estimator);
   derefiner.SetThreshold(ctx.hysteresis * ctx.max_elem_error);
   derefiner.SetNCLimit(ctx.nc_limit);

   // ParGridFunction w_gf(*u_gf);
   // flowsolver.ComputeCurl2D(*u_gf, w_gf);

   ParaViewDataCollection pvdc("shear_output", pmesh);
   pvdc.SetDataFormat(VTKFormat::BINARY32);
   pvdc.SetHighOrderOutput(true);
   pvdc.SetLevelsOfDetail(ctx.order);
   pvdc.SetCycle(0);
   pvdc.SetTime(t);
   pvdc.RegisterField("velocity", u_gf);
   pvdc.RegisterField("pressure", p_gf);
   // pvdc.RegisterField("vorticity", &w_gf);
   pvdc.Save();

   // char vishost[] = "128.15.198.77";
   // int  visport   = 19916;
   // socketstream sol_sock(vishost, visport);
   // sol_sock << "parallel " << num_procs << " " << myid << "\n";
   // sol_sock.precision(8);
   // sol_sock << "solution\n" << *pmesh << *u_gf << "\n" << "pause\n" << std::flush;

   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      refiner.Reset();
      derefiner.Reset();

      if (step % 10 == 0)
      {
         vel_mag_gf.ProjectCoefficient(mag_coeff);
         refiner.Apply(*pmesh);
         printf("refined to #el: %lld\n", pmesh->GetGlobalNE());
         flowsolver.UpdateSpaces();
         flowsolver.UpdateForms();
         flowsolver.UpdateSolvers();
         vel_mag_gf.Update();
      }

      flowsolver.Step(t, dt, step);

      if (step % 10 == 0)
      {
         // flowsolver.ComputeCurl2D(*u_gf, w_gf);
         pvdc.SetCycle(step);
         pvdc.SetTime(t);
         pvdc.Save();

         // sol_sock << "parallel " << num_procs << " " << myid << "\n";
         // sol_sock.precision(8);
         // sol_sock << "solution\n" << *pmesh << *u_gf << std::flush;
      }

      if (step % 100 == 0)
      {
         derefiner.Apply(*pmesh);
         printf("derefined to #el: %lld\n", pmesh->GetGlobalNE());
         flowsolver.UpdateSpaces();
         flowsolver.UpdateForms();
         flowsolver.UpdateSolvers();
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
