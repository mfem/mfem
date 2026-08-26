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
//             ------------------------------------------
//                dFEM Navier-Stokes: convergence test
//             ------------------------------------------
//
// Compile with: make dfem-navier-convergence
//
// Sample runs:
//
// - Single Q2-Q1 Taylor-Green vortex run
//   (nu = 0.01, f = 1, dt = 0.1, t_final = 1.0):
//   mpirun -np 4 ./dfem-navier-convergence -o 1 -r 2 -nu 0.01 -f 1 -dt 0.1 -tf 1.0
//
// - Spatial convergence study:
//   mpirun -np 4 ./dfem-navier-convergence -o 1 -r 0 -sc -cl 4
//
// - Temporal convergence study (SDIRK23Solver)
//   mpirun -np 4 ./dfem-navier-convergence -o 1 -r 1 -tc -cl 3 -ode 22
//
// Description:
//   This miniapp solves the incompressible Navier-Stokes equations
//   using a mixed system defined by a dFEM q-function. This formulation can be
//   extended to more complex problems, including non-Newtonian rheology and
//   stabilization terms such as SUPG, GLS, and PSPG (WIP).
//
//   This example solves the Taylor-Green vortex problem on the unit square. It
//   can run either a single simulation or a spatial or temporal convergence
//   study. The specified order k is the pressure-space order; the velocity and
//   pressure use Q_{k+1}-Q_k Taylor-Hood elements, with k >= 1.
//

#include "mfem.hpp"
#include "lib/navier_solver.hpp"

#include <memory>
#include <iomanip>

using namespace mfem;
using namespace mfem::dfem_navier;

constexpr int dim = 2;

// ----------------------------------------------------------------------------
// Taylor-Green vortex on the unit square, an exact solution of the
// incompressible Navier-Stokes equations. With wavenumber a = f pi,
//
//   u(x,t) = exp(-2 nu a^2 t) * ( sin(a x) cos(a y), -cos(a x) sin(a y) )
//   p(x,t) = exp(-4 nu a^2 t) * (cos(2 a x) + cos(2 a y)) / 4
//
// The frequency f is the number of vortices per direction, and must be a
// positive INTEGER: on the walls x = 1 and y = 1 the normal velocity is
// proportional to sin(f pi), so a non-integer frequency drives flow through
// them and the free-slip conditions stop describing this exact solution.
// ----------------------------------------------------------------------------

void ExactVelocity(const Vector &x, real_t t, Vector &u, real_t viscosity,
                   int frequency)
{
   const real_t a = frequency * M_PI;
   const real_t decay = exp(-2.0 * viscosity * a * a * t);
   u.SetSize(dim);
   u(0) = decay * sin(a * x(0)) * cos(a * x(1));
   u(1) = -decay * cos(a * x(0)) * sin(a * x(1));
}

real_t ExactPressure(const Vector &x, real_t t, real_t viscosity,
                     int frequency)
{
   const real_t a = frequency * M_PI;
   const real_t decay = exp(-4.0 * viscosity * a * a * t);
   return 0.25 * decay * (cos(2.0 * a * x(0)) + cos(2.0 * a * x(1)));
}

// ----------------------------------------------------------------------------
//
// Main driver routines. For convenience we keep them here to allow single run
// and convergence studies in the same test.
// The main is just for parsing the command line and calling the appropriate study
//
// ----------------------------------------------------------------------------

struct RunConfiguration
{
   int order;
   int refinements;
   int elements_per_direction;
   int ode_solver_type;
   real_t viscosity;
   int frequency;
   real_t dt;
   real_t t_final;
   const char *outfolder;
   bool paraview;
   bool verbose;
};

struct RunResult
{
   HYPRE_BigInt velocity_dofs;
   HYPRE_BigInt pressure_dofs;
   int steps;
   real_t dt;
   real_t velocity_error;
   real_t pressure_error;
   real_t elapsed_time;
   Vector velocity_true_dofs;
   Vector pressure_true_dofs;
};


RunResult Run(const RunConfiguration &config,
              const RunResult *temporal_reference = nullptr)
{
   MPI_Barrier(MPI_COMM_WORLD);
   StopWatch timer;
   timer.Start();

   Mesh mesh = Mesh::MakeCartesian2D(config.elements_per_direction,
                                     config.elements_per_direction,
                                     Element::QUADRILATERAL);
   for (int level = 0; level < config.refinements; level++)
   {
      mesh.UniformRefinement();
   }
   mesh.SetCurvature(config.order + 1);

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   pmesh.EnsureNodes();

   H1_FECollection velocity_fec(config.order + 1, dim);
   H1_FECollection pressure_fec(config.order, dim);
   ParFiniteElementSpace ufes(&pmesh, &velocity_fec, dim);
   ParFiniteElementSpace pfes(&pmesh, &pressure_fec);

   const HYPRE_BigInt global_velocity_dofs = ufes.GlobalTrueVSize();
   const HYPRE_BigInt global_pressure_dofs = pfes.GlobalTrueVSize();
   if (config.verbose && Mpi::Root())
   {
      mfem::out << "\nTaylor-Green vortex\n"
                << "  nu = " << std::scientific << std::setprecision(3)
                << config.viscosity << ", frequency = " << config.frequency << '\n'
                << "  spaces: velocity H1_" << config.order + 1 << "^" << dim
                << ", pressure H1_" << config.order << '\n'
                << "  global dofs: velocity " << global_velocity_dofs
                << ", pressure " << global_pressure_dofs << "\n\n"
                << std::right
                << std::setw(6) << "step"
                << std::setw(12) << "time"
                << std::setw(16) << "||u - u_ex||_L2" << '\n'
                << std::string(34, '-') << '\n';
   }

   Array<int> block_offsets(3);
   block_offsets[0] = 0;
   block_offsets[1] = ufes.GetTrueVSize();
   block_offsets[2] = pfes.GetTrueVSize();
   block_offsets.PartialSum();

   BlockVector state(block_offsets);
   state = 0.0;

   // Free-slip walls: constrain the normal velocity component only. For the
   // Cartesian mesh built above the boundary attributes are 1 = bottom,
   // 2 = right, 3 = top, 4 = left, so the horizontal walls constrain u_y and
   // the vertical walls constrain u_x.
   Array<int> horizontal_walls(pmesh.bdr_attributes.Max());
   Array<int> vertical_walls(pmesh.bdr_attributes.Max());
   horizontal_walls = 0;
   vertical_walls = 0;
   horizontal_walls[0] = 1;
   horizontal_walls[2] = 1;
   vertical_walls[1] = 1;
   vertical_walls[3] = 1;

   Array<int> normal_tdofs, tdofs_y, tdofs_x;
   ufes.GetEssentialTrueDofs(horizontal_walls, tdofs_y, 1);
   ufes.GetEssentialTrueDofs(vertical_walls, tdofs_x, 0);
   normal_tdofs.Append(tdofs_y);
   normal_tdofs.Append(tdofs_x);
   normal_tdofs.Sort();
   normal_tdofs.Unique();

   const IntegrationRule &ir = IntRules.Get(
                                  pmesh.GetTypicalElementGeometry(),
                                  2 * (config.order + 1) + 2);

   NavierStokesOperator<dim> ns_operator(ufes, pfes, ir, config.viscosity);
   ns_operator.SetEssentialVelocityTrueDofs(normal_tdofs);
   NavierStokesEvolution evolution(ufes, pfes, ns_operator, state);

   VectorFunctionCoefficient exact_velocity(
      dim, [&config](const Vector &x, real_t t, Vector &u)
   {
      ExactVelocity(x, t, u, config.viscosity, config.frequency);
   });
   FunctionCoefficient exact_pressure(
      [&config](const Vector &x, real_t t)
   {
      return ExactPressure(x, t, config.viscosity, config.frequency);
   });

   ParGridFunction velocity(&ufes);
   ParGridFunction pressure(&pfes);
   exact_velocity.SetTime(0.0);
   velocity.ProjectCoefficient(exact_velocity);
   velocity.GetTrueDofs(state.GetBlock(U));
   evolution.ProjectDivergenceFree(state.GetBlock(U));

   NavierStokesSolver solver(
      ODESolver::SelectImplicit(config.ode_solver_type), evolution);

   velocity.Distribute(state.GetBlock(U));
   pressure = 0.0;

   std::unique_ptr<ParaViewDataCollection> pd;
   if (config.paraview)
   {
      pd = std::make_unique<ParaViewDataCollection>(
              "dfem-navier-stokes-output", &pmesh);
      pd->SetPrefixPath(config.outfolder);
      pd->RegisterField("velocity", &velocity);
      pd->RegisterField("pressure", &pressure);
      pd->SetDataFormat(VTKFormat::BINARY);
      pd->SetHighOrderOutput(true);
      pd->SetLevelsOfDetail(config.order + 1);
      pd->SetCycle(0);
      pd->SetTime(0.0);
      pd->Save();
   }

   real_t t = 0.0;
   int step = 0;
   bool done = false;
   while (!done)
   {
      real_t step_dt = std::min(config.dt, config.t_final - t);
      solver.Step(state, t, step_dt);
      step++;
      done = (t >= config.t_final - 1e-8 * config.dt);

      if (config.verbose || config.paraview)
      {
         velocity.Distribute(state.GetBlock(U));
         pressure.Distribute(state.GetBlock(P));

         if (config.verbose)
         {
            exact_velocity.SetTime(t);
            const real_t velocity_error = velocity.ComputeL2Error(exact_velocity);
            if (Mpi::Root())
            {
               mfem::out << std::right
                         << std::setw(6) << step
                         << std::scientific << std::setprecision(3)
                         << std::setw(12) << t
                         << std::setw(16) << velocity_error << '\n';
            }
         }
      }

      if (config.paraview)
      {
         pd->SetCycle(step);
         pd->SetTime(t);
         pd->Save();
      }
   }

   velocity.Distribute(state.GetBlock(U));
   pressure.Distribute(state.GetBlock(P));
   exact_velocity.SetTime(t);
   exact_pressure.SetTime(t);
   real_t velocity_error = velocity.ComputeL2Error(exact_velocity);
   real_t pressure_error = pressure.ComputeL2Error(exact_pressure);
   if (temporal_reference)
   {
      ParGridFunction reference_velocity(&ufes);
      ParGridFunction reference_pressure(&pfes);
      reference_velocity.Distribute(temporal_reference->velocity_true_dofs);
      reference_pressure.Distribute(temporal_reference->pressure_true_dofs);
      VectorGridFunctionCoefficient velocity_coefficient(&reference_velocity);
      GridFunctionCoefficient pressure_coefficient(&reference_pressure);
      velocity_error = velocity.ComputeL2Error(velocity_coefficient);
      pressure_error = pressure.ComputeL2Error(pressure_coefficient);
   }
   timer.Stop();
   real_t elapsed_time = timer.RealTime();
   MPI_Allreduce(MPI_IN_PLACE, &elapsed_time, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_MAX, MPI_COMM_WORLD);

   return {global_velocity_dofs, global_pressure_dofs, step, config.dt,
           velocity_error, pressure_error, elapsed_time,
           state.GetBlock(U), state.GetBlock(P)};
}

void PrintConvergenceRow(int level, const RunResult &result,
                         const RunResult *previous, bool temporal)
{
   if (!Mpi::Root()) { return; }

   real_t velocity_rate = 0.0;
   real_t pressure_rate = 0.0;
   if (previous)
   {
      const real_t refinement_ratio = temporal
                                      ? previous->dt / result.dt
                                      : 2.0;
      velocity_rate = log(previous->velocity_error / result.velocity_error) /
                      log(refinement_ratio);
      pressure_rate = log(previous->pressure_error / result.pressure_error) /
                      log(refinement_ratio);
   }

   mfem::out << std::right
             << std::setw(2) << level
             << std::setw(8) << result.velocity_dofs
             << std::setw(8) << result.pressure_dofs
             << std::setw(5) << result.steps
             << std::scientific << std::setprecision(3)
             << std::setw(10) << result.dt
             << std::setw(11) << result.velocity_error;
   if (previous)
   {
      mfem::out << std::fixed << std::setprecision(2)
                << std::setw(6) << velocity_rate;
   }
   else { mfem::out << std::setw(6) << '-'; }
   mfem::out << std::scientific << std::setprecision(3)
             << std::setw(11) << result.pressure_error;
   if (previous)
   {
      mfem::out << std::fixed << std::setprecision(2)
                << std::setw(6) << pressure_rate;
   }
   else { mfem::out << std::setw(6) << '-'; }
   mfem::out << std::fixed << std::setprecision(2)
             << std::setw(7) << result.elapsed_time << '\n';
}

// ----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   int order = 1;
   int refinements = 1;
   int elements_per_direction = 4;
   int ode_solver_type =
      21;         // 21 = backward Euler, 22 = SDIRK23, 23 = SDIRK33, ...
   real_t viscosity = 0.01;
   int frequency = 1;
   real_t dt = 1.0e-2;
   real_t t_final = 0.1;
   const char *device_config = "cpu";
   const char *outfolder = "./Output";
   bool paraview = false;
   bool temporal_convergence = false;
   bool spatial_convergence = false;
   int convergence_levels = 3;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "P order; velocity uses order + 1 (Taylor-Hood).");
   args.AddOption(&refinements, "-r", "--refinements",
                  "Number of serial uniform refinements.");
   args.AddOption(&elements_per_direction, "-n", "--elements",
                  "Initial Cartesian elements per direction.");
   args.AddOption(&viscosity, "-nu", "--viscosity", "Kinematic viscosity.");
   args.AddOption(&frequency, "-f", "--frequency",
                  "Taylor-Green frequency: vortices per direction. Must "
                  "be a positive integer, otherwise the free-slip walls "
                  "no longer match the exact solution.");
   args.AddOption(&dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&t_final, "-tf", "--t-final", "Final time.");
   args.AddOption(&ode_solver_type, "-ode", "--ode-solver",
                  "Implicit ODE solver, see ODESolver::SelectImplicit(): "
                  "21 = backward Euler, 22 = SDIRK23 (L-stable), "
                  "23 = SDIRK33, 32 = implicit midpoint.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv",
                  "--no-paraview", "Enable or disable ParaView output.");
   args.AddOption(&outfolder, "-of", "--output-folder",
                  "Output folder for ParaView DataCollection files.");
   args.AddOption(&temporal_convergence, "-tc", "--temporal-convergence",
                  "-no-tc", "--no-temporal-convergence",
                  "Run a time-step refinement study.");
   args.AddOption(&spatial_convergence, "-sc", "--spatial-convergence",
                  "-no-sc", "--no-spatial-convergence",
                  "Run a mesh refinement study.");
   args.AddOption(&convergence_levels, "-cl", "--convergence-levels",
                  "Number of runs in a convergence study.");
   args.ParseCheck();

   MFEM_VERIFY(order >= 1, "Taylor-Hood pressure order must be at least one");
   MFEM_VERIFY(elements_per_direction > 0,
               "elements per direction must be positive");
   MFEM_VERIFY(dt > 0.0, "time step must be positive");
   MFEM_VERIFY(refinements >= 0, "number of refinements must be nonnegative");
   MFEM_VERIFY(!(temporal_convergence && spatial_convergence),
               "select either temporal or spatial convergence, not both");
   MFEM_VERIFY(!temporal_convergence && !spatial_convergence ||
               convergence_levels > 1,
               "convergence studies require at least two levels");
   MFEM_VERIFY(frequency >= 1,
               "Taylor-Green frequency must be a positive integer; a "
               "non-integer wavenumber drives flow through the free-slip "
               "walls and is not this exact solution");

   Device device(device_config);
   if (Mpi::Root())
   {
      device.Print();
   }

   RunConfiguration config{order, refinements, elements_per_direction,
                           ode_solver_type, viscosity, frequency, dt, t_final,
                           outfolder, paraview, true};

   // Single run, no convergence study requested
   if (!temporal_convergence && !spatial_convergence)
   {
      Run(config);
      return 0;
   }

   // Print specifics of the Taylor-Green vortex problem
   if (Mpi::Root())
   {
      mfem::out << "Taylor-Green vortex, nu = " << config.viscosity
                << ", frequency = " << config.frequency << '\n'
                << "Taylor-Hood spaces: velocity H1_" << config.order + 1
                << "^" << dim << ", pressure H1_" << config.order << '\n';
   }

   // If either convergence study is requested, we disable ParaView output and verbose printing for the individual runs
   config.paraview = false;
   config.verbose = false;
   if (Mpi::Root())
   {
      mfem::out << '\n' << (temporal_convergence ? "Temporal" : "Spatial")
                << " convergence study\n";
      if (temporal_convergence)
      {
         const int temporal_order =
            ode_solver_type == 21 ? 1 : ode_solver_type == 22 ? 2 :
            ode_solver_type == 23 ? 3 : ode_solver_type == 32 ? 2 : 0;
         mfem::out << "Expected temporal rate: "
                   << (temporal_order ? std::to_string(temporal_order)
                       : "method-dependent")
                   << " for both velocity and pressure.\n";
         mfem::out << "Errors are relative to a same-mesh reference solution "
                   << "with dt / " << (1 << (convergence_levels + 2))
                   << ".\n";
      }
      else
      {
         mfem::out << "Expected spatial L2 rates: " << order + 2
                   << " for velocity (Q" << order + 1 << "), "
                   << order + 1 << " for pressure (Q" << order << ").\n"
                   << "(NOTE: rates might be higher for this smooth solution on a Cartesian mesh.)\n";
      }
      mfem::out << '\n'
                << std::right
                << std::setw(2) << "L"
                << std::setw(8) << "u-dofs"
                << std::setw(8) << "p-dofs"
                << std::setw(5) << "N"
                << std::setw(10) << "dt"
                << std::setw(11) << "||e_u||"
                << std::setw(6) << "rate"
                << std::setw(11) << "||e_p||"
                << std::setw(6) << "rate"
                << std::setw(7) << "sec" << '\n'
                << std::string(74, '-') << '\n';
   }

   RunResult temporal_reference;
   if (temporal_convergence)
   {
      /// For the temporal convergence study, we compute a reference solution with a much smaller time step (2 levels of refinement beyond the finest level)
      RunConfiguration reference_config = config;
      reference_config.dt = dt / (1 << (convergence_levels + 2));
      if (Mpi::Root())
      {
         mfem::out << "Computing reference solution (dt = "
                   << std::scientific << std::setprecision(3)
                   << reference_config.dt << ")..." << std::flush;
      }
      temporal_reference = Run(reference_config);
      if (Mpi::Root())
      {
         mfem::out << " done (" << std::fixed << std::setprecision(2)
                   << temporal_reference.elapsed_time << " s).\n";
      }
   }

   RunResult previous{};
   for (int level = 0; level < convergence_levels; level++)
   {
      RunConfiguration level_config = config;
      if (temporal_convergence)
      {
         level_config.dt = dt / pow(2.0, level);
      }
      else
      {
         level_config.refinements = refinements + level;
      }

      const RunResult result = Run(
                                  level_config,
                                  temporal_convergence ? &temporal_reference : nullptr);
      PrintConvergenceRow(level, result, level ? &previous : nullptr,
                          temporal_convergence);
      previous = result;
   }

   return 0;
}
