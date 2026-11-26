/** @file stokes_adjoint_example.cpp
    @brief Force-amplitude sensitivity of a quadratic Stokes velocity objective.

    Solve steady Stokes flow on the unit square with constant viscosity mu=1,
    homogeneous velocity Dirichlet data, and body force f(alpha)=alpha*f0. The
    forward discretization is Taylor-Hood Q2/Q1 on a 3-by-3 quadrilateral mesh;
    the pressure gauge is selected by the forward solver. The objective is

        Q(alpha) = (1/2) integral_Omega |u_h(alpha)|^2 dx
                 = (1/2) u^T M u.

    Here u contains uniquely owned primal velocity true DOFs and M is the
    assembled velocity mass matrix. For R(z, alpha)=A*z-b(alpha), form the
    coordinate covector g=dQ/dz=(M*u, 0, 0), solve A^T*lambda=g, and compute
    dQ/dalpha=-lambda^T*(partial_R/partial_alpha). The explicit objective
    derivative at fixed z is zero because neither Q nor M depends on alpha.

    Only the load changes with alpha, so z(alpha)=alpha*z(1) and the independent
    analytic check is dQ/dalpha=2*Q/alpha. This is a scaling check, not a Taylor
    remainder test; the latter are in stokes_adjoint_tests.cpp.

    Adjoint outputs remain algebraic Vectors; no H1 field interpretation or
    grid-function conversion is applied. Grid functions used below belong to
    the primal state only. A second adjoint solve validates copied primal fields
    in the same FE spaces and reuses the existing adjoint setup without running
    another forward solve.

    From the workspace containing mfem/, build/, and set_up_local.sh:
    @code{.sh}
    set +u
    source ./set_up_local.sh
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    cmake --build build --target stokes_adjoint_example --parallel 8
    "$PETSC_DIR/bin/mpiexec" -launcher fork -rmk user -n 4 \
      build/miniapps/stokes/stokes_adjoint_example --solver fgmres
    @endcode

    The --solver option accepts fgmres (default), gmres, or mumps and selects
    both forward and adjoint backends. MUMPS requires MFEM_USE_MUMPS. All ranks
    must use the same options; solver calls and global contractions are
    collective. See ADJOINT.md for the general derivative and ownership contracts. */

#include "stokes.hpp"
#include "examples/support/manufactured.hpp"
#include <cmath>
#include <string>

/** @brief Run stored-state and supplied-state adjoints and check the sensitivity.

    Prints the objective, computed/expected derivative, algebraic adjoint norm,
    supplied-state convergence, and setup reuse on rank zero.

    @return Zero when all solves converge, setup is reused, and the derivative
    matches 2*Q/alpha within 1e-7*(1+|2*Q/alpha|); one for failed numerical checks.
    Invalid configuration can also terminate through MFEM's error handling. */
int main(int argc, char **argv)
{
   using namespace mfem;
   using namespace mfem::flow;
   Mpi::Init(argc, argv);
   Hypre::Init();
   Device device("cpu");
   const char *backend_name = "fgmres";
   OptionsParser arguments(argc, argv);
   arguments.AddOption(&backend_name, "-s", "--solver", "mumps, gmres, or fgmres.");
   arguments.ParseCheck();
   const std::string backend(backend_name);
   MFEM_VERIFY(backend == "mumps" || backend == "gmres" || backend == "fgmres", "Unknown adjoint example backend");
   const auto linear_solver = backend == "mumps" ? LinearSolverType::MUMPS :
                              backend == "gmres" ? LinearSolverType::GMRES : LinearSolverType::FGMRES;
   // FE spaces describe the primal discretization, not an assumed adjoint field.
   Mesh serial = MakeStokesMesh("quad", 3);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   H1_FECollection velocity_collection(2, 2), pressure_collection(1, 2);
   ParFiniteElementSpace velocity_space(&mesh, &velocity_collection, 2, Ordering::byVDIM);
   ParFiniteElementSpace pressure_space(&mesh, &pressure_collection);
   // Use the manufactured force as f0; differentiate only its scalar amplitude.
   ManufacturedSolution exact(2);
   const real_t amplitude = 1.2;
   VectorFunctionCoefficient force(2, [&](const Vector &point, Vector &value)
   {
      exact.Force(point, value);
      value *= amplitude;
   });
   FlowSolverOptions forward_options;
   forward_options.linear.relative_tolerance = 1e-12;
   forward_options.linear.absolute_tolerance = 1e-14;
   if (linear_solver == LinearSolverType::MUMPS) { forward_options.linear.backend = MumpsOptions{}; }
   else
   {
      IterativeOptions iterative; iterative.method = linear_solver;
      forward_options.linear.backend = iterative;
   }
   FlowProblem problem(velocity_space, pressure_space, 1.0);
   problem.SetPressureGauge(PressureGauge::Auto());
   problem.SetBodyForce(1, force);
   BoundaryConditions boundary;
   for (int attribute : mesh.bdr_attributes)
   {
      boundary.SetVelocity(attribute, Direction::X, 0.0);
      boundary.SetVelocity(attribute, Direction::Y, 0.0);
   }
   problem.SetBoundaryConditions(boundary);
   FlowSolver forward(problem, forward_options);
   ParGridFunction velocity(&velocity_space), pressure(&pressure_space);
   const auto forward_info = forward.Solve(velocity, pressure);
   if (!forward_info.converged) { return 1; }
   AdjointOptions adjoint_options;
   adjoint_options.linear = forward_options.linear;
   FlowAdjointSolver adjoint(problem, forward_options, adjoint_options);
   FlowState primal{velocity, pressure, forward_info.gauge_multiplier};
   AdjointState multipliers;
   adjoint.Prepare(primal);
   const Array<int> offsets(adjoint.GetLayout().GetOffsets());
   Vector objective_gradient(offsets.Last()), residual_direction(offsets.Last()), velocity_dofs;
   objective_gradient = 0.0;
   residual_direction = 0.0;
   BlockVector objective_blocks(objective_gradient, offsets), derivative_blocks(residual_direction, offsets);
   velocity.GetTrueDofs(velocity_dofs);
   ParBilinearForm mass_form(&velocity_space);
   mass_form.AddDomainIntegrator(new VectorMassIntegrator());
   mass_form.Assemble();
   mass_form.Finalize();
   std::unique_ptr<HypreParMatrix> mass(mass_form.ParallelAssemble());
   mass->Mult(velocity_dofs, objective_blocks.GetBlock(0));
   const real_t objective = 0.5 * InnerProduct(MPI_COMM_WORLD, velocity_dofs, objective_blocks.GetBlock(0));
   // At fixed z, partial_R/partial_alpha=-b0 because A is amplitude-independent.
   // Assemble the unscaled f0 load using the forward quadrature. Taylor-Hood has
   // no pressure-row forcing; stabilized methods may need additional load terms.
   ParLinearForm load_derivative(&velocity_space);
   load_derivative.AddDomainIntegrator(new RuleLinearIntegrator<VectorDomainLFIntegrator>(
                                         forward_info.integration_order, exact.force));
   load_derivative.Assemble();
   load_derivative.ParallelAssemble(derivative_blocks.GetBlock(0));
   derivative_blocks.GetBlock(0) *= -1.0;
   // Essential residual equations prescribe zero velocity independently of alpha,
   // so their parameter derivatives vanish. Zero only these residual entries,
   // not the objective covector or the resulting algebraic adjoint entries.
   Array<int> marker(mesh.bdr_attributes.Max()), essential;
   marker = 1;
   velocity_space.GetEssentialTrueDofs(marker, essential);
   for (int index = 0; index < essential.Size(); ++index)
   {
      derivative_blocks.GetBlock(0)(essential[index]) = 0.0;
   }
   // Validate the stored primal state and solve the exact constrained transpose.
   const auto first = adjoint.Solve(primal, objective_gradient, multipliers);
   if (!first.converged) { return 1; }
   // The packed Vector includes the gauge once, without shared/ghost duplication.
   // Its Euclidean norm is an algebraic diagnostic, not an H1 or L2 field norm.
   // This reference is not a snapshot: the next Solve() overwrites its contents.
   const Vector &adjoint_vector = multipliers.GetMultipliers();
   const real_t adjoint_norm = std::sqrt(InnerProduct(MPI_COMM_WORLD, adjoint_vector, adjoint_vector));
   // No explicit alpha dependence in Q: return 0-lambda^T*(partial_R/partial_alpha).
   const real_t gradient = adjoint.ComputeDirectionalSensitivity(0.0, residual_direction);
   // These grid functions copy primal fields, not adjoints. Supplied states must
   // use the same FE-space objects and include the replicated primal multiplier
   // exactly when the pressure gauge is active.
   ParGridFunction saved_velocity(velocity), saved_pressure(pressure);
   const FlowState supplied{saved_velocity, saved_pressure,
                                    forward_info.gauge_multiplier};
   // Only the state input path changes: no source setter or reassembly occurs,
   // so the same adjoint factors/preconditioner should be reused.
   const auto second = adjoint.Solve(supplied, objective_gradient, multipliers);
   if (Mpi::Root())
   {
      out << "Objective=" << objective << " force-amplitude gradient=" << gradient
          << " algebraic-adjoint norm=" << adjoint_norm
          << " expected=" << 2.0 * objective / amplitude
          << " supplied-state converged=" << second.converged
          << " setup-reused=" << second.setup_reused << '\n';
   }
   // With zero boundary data and fixed A, Q scales as alpha^2 on this discrete
   // mesh too. This check needs no exact continuum solution or finite differences.
   const real_t expected = 2.0 * objective / amplitude;
   return second.converged && second.setup_reused &&
          std::abs(gradient - expected) < 1e-7 * (1.0 + std::abs(expected)) ? 0 : 1;
}
