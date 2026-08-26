// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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
//             -----------------------------------------
//                dFEM Navier-Stokes: Kovasznay flow
//             -----------------------------------------
//
// Compile with: make dfem-navier-kovasznay
//
// Sample runs:
//
// - Reynolds 40, Q2-Q1, transient from rest until the flow reaches steady state:
//   mpirun -np 4 ./dfem-navier-kovasznay -o 1 -r 2 -re 40 -dt 2e-2 -tf 20 -st 1e-8
//
// - Same transient at Q3-Q2 on a finer mesh, with ParaView output every 5 steps:
//   mpirun -np 4 ./dfem-navier-kovasznay -o 2 -r 3 -re 40 -dt 2e-2 -tf 20 -st 1e-8 -vs 5 -pv
//
// - Steady-state initialized test (follows miniapps/fluids/navier/navier_kovasznay.cpp).
//   No real transient but useful to check the solver preserves the steady state
//   (cr checks results against analytical sol, eic starts the run from the exact solution,
//   tu/tp set the tolerances for the check):
//   mpirun -np 4 ./dfem-navier-kovasznay -o 4 -r 2 -re 40 -dt 1e-3 \
//      -tf 1e-2 -eic -st 0 -cr -tu 1e-7 -tp 1e-7
//
// Description:
//   This miniapp solves the incompressible Navier-Stokes equations
//   using a mixed system defined by a dFEM q-function. This formulation can be
//   extended to more complex problems, including non-Newtonian rheology and
//   stabilization terms such as SUPG, GLS, and PSPG (WIP).
//
//   This example solves the Kovasznay flow problem,
//
//     u = [1 - exp(L x) cos(2 pi y), L / (2 pi) exp(L x) sin(2 pi y)],
//     p = -1/2 exp(2 L x),   L = Re/2 - sqrt(Re^2/4 + 4 pi^2),
//
//   an analytic steady solution of the incompressible Navier-Stokes equations.
//   Dirichlet data for the velocity is applied on every boundary. By default
//   the interior starts at rest, so the run is a genuine transient that has to
//   develop the Kovasznay solution.
//
//   With -eic the exact solution is instead used as the initial condition, as
//   in miniapps/fluids/navier/navier_kovasznay.cpp. Useful to check that the
//   integrator preserves the steady state, whereas without -eic the run is a
//   true transient evolution.
//

#include "mfem.hpp"
#include "lib/navier_solver.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <memory>

using namespace mfem;
using namespace mfem::dfem_navier;

namespace
{
constexpr int dim = 2;
real_t lambda = 0.0;

void ExactVelocity(const Vector &x, Vector &velocity)
{
   const real_t exponential = std::exp(lambda * x(0));
   velocity(0) = 1.0 - exponential * std::cos(2.0 * M_PI * x(1));
   velocity(1) = lambda / (2.0 * M_PI) * exponential
                 * std::sin(2.0 * M_PI * x(1));
}

real_t ExactPressure(const Vector &x)
{
   return -0.5 * std::exp(2.0 * lambda * x(0));
}
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   int order = 1;
   int refinements = 1;
   int ode_solver_type = 22;
   int visualization_steps = 1;
   real_t viscosity = -1.0;
   real_t reynolds = 40.0;
   real_t dt = 2.0e-2;
   real_t t_final = 20.0;
   real_t steady_tolerance = 1.0e-8;
   real_t tolerance_velocity = 1.0e-2;
   real_t tolerance_pressure = 1.0e-2;
   const char *device_config = "cpu";
   const char *outfolder = "./Output";
   bool paraview = false;
   bool check_result = false;
   bool exact_initial_condition = false;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Pressure order; velocity uses order + 1.");
   args.AddOption(&refinements, "-r", "--refinements",
                  "Number of serial uniform refinements.");
   args.AddOption(&reynolds, "-re", "--reynolds", "Reynolds number.");
   args.AddOption(&dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&t_final, "-tf", "--t-final", "Final time.");
   args.AddOption(&steady_tolerance, "-st", "--steady-tolerance",
                  "Relative velocity-change rate for early termination; "
                  "use zero (the default) to always run to t_final.");
   args.AddOption(&ode_solver_type, "-ode", "--ode-solver",
                  "Implicit ODE solver type.");
   args.AddOption(&visualization_steps, "-vs", "--visualization-steps",
                  "Save ParaView output every n steps.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv",
                  "--no-paraview", "Enable or disable ParaView output.");
   args.AddOption(&outfolder, "-of", "--output-folder",
                  "Output folder for ParaView DataCollection files.");
   args.AddOption(&exact_initial_condition, "-eic", "--exact-initial-condition",
                  "-rest", "--rest-initial-condition",
                  "Start from the exact solution (no transient, tests that the "
                  "integrator preserves it) or from rest (real transient).");
   args.AddOption(&check_result, "-cr", "--checkresult",
                  "-no-cr", "--no-checkresult",
                  "Enable or disable checking of the result. "
                  "Returns -1 on failure.");
   args.AddOption(&tolerance_velocity, "-tu", "--tolerance-velocity",
                  "Velocity L2 error accepted by --checkresult.");
   args.AddOption(&tolerance_pressure, "-tp", "--tolerance-pressure",
                  "Pressure L2 error accepted by --checkresult.");
   args.ParseCheck();

   MFEM_VERIFY(order >= 1, "pressure order must be at least one");
   MFEM_VERIFY(reynolds > 0.0, "Reynolds number must be positive");
   MFEM_VERIFY(visualization_steps >= 1,
               "visualization steps must be positive");
   MFEM_VERIFY(dt > 0.0 && t_final >= 0.0,
               "time interval must be valid");
   MFEM_VERIFY(steady_tolerance >= 0.0,
               "steady tolerance must be nonnegative");

   viscosity = 1.0 / reynolds;
   lambda = 0.5 * reynolds
            - std::sqrt(0.25 * reynolds * reynolds
                        + 4.0 * M_PI * M_PI);

   Device device(device_config);
   if (Mpi::Root())
   {
      device.Print();
      args.PrintOptions(mfem::out);
   }

   Mesh mesh = Mesh::MakeCartesian2D(2, 4, Element::QUADRILATERAL,
                                     false, 1.5, 2.0);
   mesh.EnsureNodes();
   *mesh.GetNodes() -= 0.5;
   for (int level = 0; level < refinements; level++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   pmesh.EnsureNodes();

   H1_FECollection velocity_fec(order + 1, dim);
   H1_FECollection pressure_fec(order, dim);
   ParFiniteElementSpace velocity_fes(&pmesh, &velocity_fec, dim);
   ParFiniteElementSpace pressure_fes(&pmesh, &pressure_fec);

   HYPRE_BigInt global_velocity_dofs = velocity_fes.GlobalTrueVSize();
   HYPRE_BigInt global_pressure_dofs = pressure_fes.GlobalTrueVSize();

   if (Mpi::Root())
   {
      mfem::out << "Number of velocity unknowns: " << global_velocity_dofs
                << "\nNumber of pressure unknowns: " << global_pressure_dofs
                << "\nReynolds number: " << reynolds
                << "\nKinematic viscosity: " << viscosity
                << "\nKovasznay lambda: " << lambda
                << std::endl;
   }

   Array<int> offsets({0, velocity_fes.GetTrueVSize(),
                       pressure_fes.GetTrueVSize()});
   offsets.PartialSum();
   BlockVector state(offsets);
   state = 0.0;

   Array<int> boundary(pmesh.bdr_attributes.Max());
   boundary = 1;
   const IntegrationRule &integration_rule = IntRules.Get(
                                                pmesh.GetTypicalElementGeometry(), 2 * (order + 1) + 2);
   NavierStokesOperator<dim> navier_operator(
      velocity_fes, pressure_fes, integration_rule, viscosity);
   navier_operator.SetEssentialVelocityAttributes(boundary);

   VectorFunctionCoefficient exact_velocity(dim, ExactVelocity);
   FunctionCoefficient exact_pressure(ExactPressure);
   ParGridFunction velocity(&velocity_fes);
   ParGridFunction pressure(&pressure_fes);
   ParGridFunction pressure_reference(&pressure_fes);
   velocity.ProjectCoefficient(exact_velocity);
   velocity.GetTrueDofs(state.GetBlock(U));

   // For a true transient, the initial velocity is zero and the boundary data is applied
   if (!exact_initial_condition)
   {
      const Array<int> &ess_tdofs =
         navier_operator.GetEssentialVelocityTrueDofs();
      Vector boundary_data(ess_tdofs.Size());
      state.GetBlock(U).GetSubVector(ess_tdofs, boundary_data);
      state.GetBlock(U) = 0.0;
      state.GetBlock(U).SetSubVector(ess_tdofs, boundary_data);
   }

   NavierStokesEvolution evolution(
      velocity_fes, pressure_fes, navier_operator, state);
   NavierStokesSolver solver(
      ODESolver::SelectImplicit(ode_solver_type), evolution);

   if (!exact_initial_condition)
   {
      evolution.ProjectDivergenceFree(state.GetBlock(U));
      evolution.RecoverPressure(state.GetBlock(U));
      state.GetBlock(P) = evolution.GetPressure();
   }

   std::unique_ptr<ParaViewDataCollection> pvdc;
   if (paraview)
   {
      pvdc = std::make_unique<ParaViewDataCollection>(
                "dfem-navier-kovasznay-output", &pmesh);
      pvdc->SetPrefixPath(outfolder);
      pvdc->SetDataFormat(VTKFormat::BINARY);
      pvdc->SetHighOrderOutput(true);
      pvdc->SetLevelsOfDetail(order + 1);
      pvdc->RegisterField("velocity", &velocity);
      pvdc->RegisterField("pressure", &pressure);
      velocity.Distribute(state.GetBlock(U));
      pressure.Distribute(state.GetBlock(P));
      pvdc->SetCycle(0);
      pvdc->SetTime(0.0);
      pvdc->Save();
   }

   // The exact pressure is only defined up to a constant, while the discrete
   // pressure is constrained to have zero mean. We project the exact field once
   // and remove its mean, so it can be compared directly at every step
   // (Kovasznay solution is steady, so the reference doesn't change in time).
   pressure_reference.ProjectCoefficient(exact_pressure);
   Vector pressure_reference_true_dofs;
   pressure_reference.GetTrueDofs(pressure_reference_true_dofs);
   evolution.MeanZero(pressure_reference_true_dofs);
   pressure_reference.Distribute(pressure_reference_true_dofs);
   GridFunctionCoefficient mean_zero_pressure(&pressure_reference);

   real_t time = 0.0;
   int step = 0;
   real_t velocity_error = 0.0;
   real_t pressure_error = 0.0;
   Vector previous_velocity(state.GetBlock(U));
   Vector velocity_change(state.GetBlock(U).Size());
   if (Mpi::Root())
   {
      mfem::out << '\n' << std::right
                << std::setw(6) << "step"
                << std::setw(14) << "time"
                << std::setw(14) << "dt"
                << std::setw(14) << "err_u"
                << std::setw(14) << "err_p"
                << std::setw(20) << "||du||/(dt ||u||)" << '\n'
                << std::string(82, '-') << '\n';
   }
   while (time < t_final - 1.0e-8 * dt)
   {
      // Solve current time step
      real_t step_dt = std::min(dt, t_final - time);
      previous_velocity = state.GetBlock(U);
      solver.Step(state, time, step_dt);
      step++;

      // Compute the relative change rate of the velocity field, we use it to
      // check if the solution has reached the steady Kovasznay state, and
      // terminate early.
      velocity_change = state.GetBlock(U);
      velocity_change -= previous_velocity;
      const real_t change_norm =
         std::sqrt(InnerProduct(MPI_COMM_WORLD, velocity_change,
                                velocity_change));
      const real_t velocity_norm =
         std::sqrt(InnerProduct(MPI_COMM_WORLD, state.GetBlock(U),
                                state.GetBlock(U)));
      const real_t relative_change_rate =
         change_norm / (step_dt * std::max(velocity_norm, 1.0e-16));
      const bool converged = steady_tolerance > 0.0 &&
                             relative_change_rate <= steady_tolerance;

      velocity.Distribute(state.GetBlock(U));
      pressure.Distribute(state.GetBlock(P));
      velocity_error = velocity.ComputeL2Error(exact_velocity);
      pressure_error = pressure.ComputeL2Error(mean_zero_pressure);

      if (Mpi::Root())
      {
         mfem::out << std::right << std::setw(6) << step
                   << std::scientific << std::setprecision(4)
                   << std::setw(14) << time
                   << std::setw(14) << step_dt
                   << std::setw(14) << velocity_error
                   << std::setw(14) << pressure_error
                   << std::setw(20) << relative_change_rate
                   << (converged ? " (steady)" : "") << '\n';
      }

      if (paraview &&
          (step % visualization_steps == 0 || converged ||
           time >= t_final - 1.0e-8 * dt))
      {
         pvdc->SetCycle(step);
         pvdc->SetTime(time);
         pvdc->Save();
      }

      if (converged) { break; }
   }

   if (Mpi::Root())
   {
      mfem::out << '\n' << "Kovasznay flow at t = " << time
                << " after " << step << " steps\n"
                << "  velocity L2 error: " << velocity_error << '\n'
                << "  pressure L2 error: " << pressure_error << '\n';
   }

   // Test if the result for the test run is as expected. The thresholds are
   // options as they change with the spatial discretization
   if (check_result)
   {
      if (velocity_error > tolerance_velocity ||
          pressure_error > tolerance_pressure)
      {
         if (Mpi::Root())
         {
            mfem::out << "Result has a larger error than expected: "
                      << "tolerances are " << tolerance_velocity
                      << " (velocity) and " << tolerance_pressure
                      << " (pressure)." << std::endl;
         }
         return -1;
      }
   }

   return 0;
}