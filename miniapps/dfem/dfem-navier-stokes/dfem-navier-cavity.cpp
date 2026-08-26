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
//             ------------------------------------------------
//                dFEM Navier-Stokes: Lid-driven cavity flow
//             ------------------------------------------------
//
// Compile with: make dfem-navier-cavity
//
// Sample runs:
//
// - Reynolds 100, Q3-Q2, dt = 0.05, t_final = 20 (but should reach steady state at t ~ 10):
//   mpirun -np 4 ./dfem-navier-cavity -o 2 -r 1 -re 100 -dt 0.05 -tf 20 -st 1e-6 -pv
//
// - Reynolds 1000, Q3-Q2, dt = 0.01, t_final = 20:
//  mpirun -np 4 ./dfem-navier-cavity -o 2 -r 1 -re 1000 -dt 0.01 -tf 20 -st 1e-6 -pv
//
// To run in 3D, replace dim = 2 with dim = 3 below and recompile. For a less
// expensive center-plane study, use fewer elements in the spanwise direction:
//   mpirun -np 4 ./dfem-navier-cavity -o 2 -e 16 -re 100 \
//      -dt 0.025 -tf 15 -st 1e-5 -pv -vs 20
//
// Description:
//   This miniapp solves the incompressible Navier-Stokes equations
//   using a mixed system defined by a dFEM q-function. This formulation can be
//   extended to more complex problems, including non-Newtonian rheology and
//   stabilization terms such as SUPG, GLS, and PSPG (WIP).
//
//   This example solves the lid-driven cavity problem on the unit square or
//   cube. All walls carry an essential (Dirichlet) velocity condition: the top
//   lid is driven with a smooth profile that tapers to zero at the corners,
//   and the three remaining walls are no-slip. The lid profile is regularized
//   rather than the discontinuous u = 1 of the textbook problem, since the
//   corner singularity of the latter destroys the high-order convergence rate.
//   The test runs until the final time t_final, or until reaching a steady state
//

#include "mfem.hpp"
#include "lib/navier_solver.hpp"

#include <algorithm>
#include <iomanip>
#include <memory>

using namespace mfem;
using namespace mfem::dfem_navier;

namespace
{

constexpr int dim = 3;

/// Velocity of the cavity walls: zero on the three stationary walls and
/// u = (g(x), 0) on the top lid. Supplies the essential boundary data, which
/// stays fixed for the whole run since the residual is zeroed on those dofs.
// Compared to traditional u = 1 on the top lid, this profile is regularized
// to taper to zero at the corners, which avoids the pressure singularity.
// The g(x satisfies g(0) = g(1) = 0, g'(0) = g'(1) = 0, and g(0.5) = 1.
void WallVelocity(const Vector &x, Vector &velocity)
{
   const real_t x0 = x(0);
   const real_t x1 = x(1);
   const real_t g = 16.0 * x0 * x0 * (1.0 - x0) * (1.0 - x0);
   const real_t dg_dx = 32.0 * x0 * (1.0 - x0) * (1.0 - 2.0 * x0);
   const real_t h = x1 * x1 * (x1 - 1.0);
   const real_t dh_dy = x1 * (3.0 * x1 - 2.0);
   const real_t z_profile = x.Size() == 3
                            ? 16.0 * x(2) * x(2) * (1.0 - x(2)) * (1.0 - x(2))
                            : 1.0;

   // The stream function psi(x,y) = g(x) h(y) defines
   // u = (d psi/dy, -d psi/dx). Thus div(u) = 0, the velocity vanishes
   // on the stationary walls, and u = (g(x), 0) on the top lid. In 3D,
   // multiplying by z_profile also makes the velocity vanish on z = 0 and 1.
   velocity(0) = g * dh_dy * z_profile;
   velocity(1) = -dg_dx * h * z_profile;
   if (velocity.Size() == 3) { velocity(2) = 0.0; }
}
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   int order = 1;
   int elements = 8;
   int refinements = 0;
   int ode_solver_type = 22;
   int visualization_steps = 1;
   real_t viscosity = -1.0;
   real_t reynolds = 100;
   real_t dt = 1.0e-2;
   real_t t_final = 1.0;
   real_t steady_tolerance = 1.0e-6;
   const char *device_config = "cpu";
   bool paraview = true;
   const char *outfolder = "./Output";

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Pressure order; velocity uses order + 1.");
   args.AddOption(&elements, "-e", "--elements",
                  "Initial elements in x and y; 3D uses half as many in z.");
   args.AddOption(&refinements, "-r", "--refinements",
                  "Number of serial uniform refinements.");
   args.AddOption(&reynolds, "-re", "--reynolds", "Reynolds number.");
   args.AddOption(&dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&t_final, "-tf", "--t-final", "Final time.");
   args.AddOption(&steady_tolerance, "-st", "--steady-tolerance",
                  "Relative velocity-change rate for early termination; "
                  "use zero to disable.");
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
   args.ParseCheck();

   MFEM_VERIFY(order >= 1, "pressure order must be at least one");
   MFEM_VERIFY(elements >= 1, "elements must be positive");
   MFEM_VERIFY(visualization_steps >= 1,
               "visualization steps must be positive");
   MFEM_VERIFY(dt > 0.0 && t_final >= 0.0,
               "time interval must be valid");
   MFEM_VERIFY(steady_tolerance >= 0.0,
               "steady tolerance must be nonnegative");
   MFEM_VERIFY(reynolds > 0.0, "Reynolds number must be positive");

   viscosity = 1.0 / reynolds;

   Device device(device_config);
   if (Mpi::Root())
   {
      device.Print();
      args.PrintOptions(mfem::out);
   }

   Mesh mesh;
   if constexpr (dim == 2)
   {
      mesh = Mesh::MakeCartesian2D(elements, elements,
                                   Element::QUADRILATERAL);
   }
   else
   {
      mesh = Mesh::MakeCartesian3D(elements, elements,
                                   std::max(1, elements / 2),
                                   Element::HEXAHEDRON);
   }
   for (int level = 0; level < refinements; level++)
   {
      mesh.UniformRefinement();
   }
   mesh.SetCurvature(order + 1);

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
                << std::endl;
   }

   Array<int> offsets({0, velocity_fes.GetTrueVSize(),
                       pressure_fes.GetTrueVSize()});
   offsets.PartialSum();
   BlockVector state(offsets);
   state = 0.0;

   Array<int> walls(pmesh.bdr_attributes.Max());
   walls = 1;
   const IntegrationRule &integration_rule = IntRules.Get(
                                                pmesh.GetTypicalElementGeometry(), 2 * (order + 1) + 2);
   NavierStokesOperator<dim> navier_operator(
      velocity_fes, pressure_fes, integration_rule, viscosity);
   navier_operator.SetEssentialVelocityAttributes(walls);

   ParGridFunction velocity(&velocity_fes);
   ParGridFunction pressure(&pressure_fes);
   VectorFunctionCoefficient wall_velocity(dim, WallVelocity);
   velocity.ProjectCoefficient(wall_velocity);
   velocity.GetTrueDofs(state.GetBlock(U));

   // The projection above filled the whole domain, but only the essential dofs
   // are boundary data. Keep those and zero the interior so the fluid starts at
   // rest, as in the classical benchmark.
   {
      const Array<int> &ess_tdofs =
         navier_operator.GetEssentialVelocityTrueDofs();
      Vector wall_data(ess_tdofs.Size());
      state.GetBlock(U).GetSubVector(ess_tdofs, wall_data);
      state.GetBlock(U) = 0.0;
      state.GetBlock(U).SetSubVector(ess_tdofs, wall_data);
   }

   NavierStokesEvolution evolution(
      velocity_fes, pressure_fes, navier_operator, state);
   NavierStokesSolver solver(
      ODESolver::SelectImplicit(ode_solver_type), evolution);

   // A discrete rest state is not divergence-free next to the moving lid, which
   // would leave the first pressure solve fighting the constraint. Project onto
   // the discretely divergence-free subspace; this leaves the essential dofs
   // untouched, so the lid data survives.
   evolution.ProjectDivergenceFree(state.GetBlock(U));
   evolution.RecoverPressure(state.GetBlock(U));
   state.GetBlock(P) = evolution.GetPressure();

   velocity.Distribute(state.GetBlock(U));
   pressure.Distribute(state.GetBlock(P));

   std::unique_ptr<ParaViewDataCollection> pvdc;
   if (paraview)
   {
      pvdc = std::make_unique<ParaViewDataCollection>(
                               "dfem-navier-cavity", &pmesh);
      pvdc->SetPrefixPath(outfolder);
      pvdc->SetDataFormat(VTKFormat::BINARY);
      pvdc->SetHighOrderOutput(true);
      pvdc->SetLevelsOfDetail(order + 1);
      pvdc->RegisterField("velocity", &velocity);
      pvdc->RegisterField("pressure", &pressure);
      pvdc->SetCycle(0);
      pvdc->SetTime(0.0);
      pvdc->Save();
   }

   real_t time = 0.0;
   int step = 0;
   Vector previous_velocity(state.GetBlock(U));
   Vector velocity_change(state.GetBlock(U).Size());
   if (Mpi::Root())
   {
      mfem::out << '\n' << std::right
                << std::setw(8) << "step"
                << std::setw(16) << "time"
                << std::setw(24) << "||du|| / (dt ||u||)" << '\n'
                << std::string(48, '-') << '\n';
   }
   while (time < t_final - 1.0e-8 * dt)
   {
      // Solve current time step
      real_t step_dt = std::min(dt, t_final - time);
      previous_velocity = state.GetBlock(U);
      solver.Step(state, time, step_dt);
      step++;

      // Compute the relative change rate of the velocity field, 
      // we use it to check if the solution has reached a steady state,
      // and terminate it early.
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

      if (Mpi::Root())
      {
         mfem::out << std::right << std::setw(8) << step
                   << std::scientific << std::setprecision(6)
                   << std::setw(16) << time
                   << std::setw(24) << relative_change_rate
                   << (converged ? " (steady)" : "") << '\n';
      }

      if (paraview &&
          (step % visualization_steps == 0 || converged ||
           time >= t_final - 1.0e-8 * dt))
      {
         velocity.Distribute(state.GetBlock(U));
         pressure.Distribute(state.GetBlock(P));
         pvdc->SetCycle(step);
         pvdc->SetTime(time);
         pvdc->Save();
      }

      if (converged) { break; }
   }

   return 0;
}