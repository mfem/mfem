// Regression for the common RK4-stage objective and its three adjoint paths.

#include "mfem.hpp"
#include "ElastodynamicsSolver.hpp"
#include "../../pde_filter.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

using namespace mfem;

namespace
{

real_t Norm(MPI_Comm comm, const Vector &value)
{
   return std::sqrt(std::max(InnerProduct(comm, value, value), real_t(0.0)));
}

real_t RelativeError(MPI_Comm comm,
                     const Vector &value,
                     const Vector &reference)
{
   Vector difference(value);
   difference -= reference;
   return Norm(comm, difference) /
          std::max(Norm(comm, reference), real_t(1e-30));
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   const MPI_Comm comm = MPI_COMM_WORLD;
   Device device("cpu");

   {
      constexpr int dimension = 2;
      constexpr int nx = 6;
      constexpr int ny = 3;
      constexpr int steps = 24;
      constexpr real_t final_time = 0.30;
      constexpr real_t time_step = final_time / steps;

      Mesh serial_mesh = Mesh::MakeCartesian2D(
         nx, ny, Element::QUADRILATERAL, true, 1.0, 0.5);
      ParMesh mesh(comm, serial_mesh);
      H1_FECollection state_collection(/*order=*/1, dimension);
      H1_FECollection filter_collection(/*order=*/1, dimension);
      L2_FECollection control_collection(
         /*order=*/0, dimension, BasisType::GaussLobatto);
      ParFiniteElementSpace state_fes(
         &mesh, &state_collection, dimension);
      ParFiniteElementSpace filter_fes(&mesh, &filter_collection);
      ParFiniteElementSpace control_fes(&mesh, &control_collection);

      ParGridFunction rho(&control_fes);
      ParGridFunction rho_tilde(&filter_fes);
      rho = 0.60;
      toopt::PDEFilterOptions filter_options;
      filter_options.filter_radius = 0.06;
      filter_options.solver_rtol = 1e-12;
      filter_options.solver_atol = 1e-14;
      filter_options.solver_maxiter = 200;
      toopt::PDEFilter filter(filter_fes, control_fes, filter_options);
      filter.Assemble();
      filter.Mult(rho, rho_tilde);

      MaterialParams material;
      material.rho0 = 1.0;
      material.lambda0 = 0.02;
      material.mu0 = 0.01;
      material.r_min = 0.10;
      material.r_max = 1.0;
      material.simp_p = 3.0;
      ConstantCoefficient rho0(material.rho0);
      ConstantCoefficient lambda0(material.lambda0);
      ConstantCoefficient mu0(material.mu0);
      SIMPCoefficient simp_mass(
         &rho_tilde, material.r_min, material.r_max, material.simp_p);
      SIMPCoefficient simp_stiffness(
         &rho_tilde, material.r_min, material.r_max, material.simp_p);
      ProductCoefficient mass(simp_mass, rho0);
      ProductCoefficient lambda(simp_stiffness, lambda0);
      ProductCoefficient mu(simp_stiffness, mu0);

      BoundaryLoadSpec load;
      load.domain_load = true;
      load.amplitude = 1.0;
      load.duration = final_time;
      load.time_profile = LoadTimeProfile::HARMONIC;
      load.frequency = 0.8;
      load.phase = 0.15;
      load.direction.SetSize(dimension);
      load.direction = 0.0;
      load.direction[1] = -1.0;
      DirectionalBoundaryLoadCoefficient load_coefficient(load.direction);

      ConstantCoefficient damping(0.03);
      Array<int> exterior_boundary(mesh.bdr_attributes.Max());
      exterior_boundary = 0;
      Array<int> essential_boundary(mesh.bdr_attributes.Max());
      essential_boundary = 0;
      MFEM_VERIFY(essential_boundary.Size() >= 4,
                  "Tiny Cartesian mesh has unexpected boundary attributes.");
      essential_boundary[3] = 1; // Clamp x=0.

      Vector target(dimension);
      target = 0.0;
      target[0] = 0.7;
      target[1] = -0.2;
      HarmonicDisplacementTrackingObjective objective(
         &state_fes, std::make_unique<ConstantCoefficient>(1.0),
         std::make_unique<VectorConstantCoefficient>(target),
         /*amplitude=*/0.25, /*frequency=*/0.55,
         /*phase=*/0.10, comm);

      const auto make_operator = [&]()
      {
         return std::make_unique<ElastodynamicsOperator>(
            state_fes, mass, lambda, mu,
            load.amplitude, load.duration, load.time_profile,
            load.phase, load.frequency, load.bdr_attributes,
            load_coefficient, load.domain_load, &damping,
            /*impedance=*/0.0, exterior_boundary, essential_boundary,
            MassSolverType::LUMPED, /*print_banner=*/false);
      };

      std::unique_ptr<ElastodynamicsOperator> oper = make_operator();
      ValidateLumpedRK4TimeStep(
         *oper, time_step, /*print_report=*/false);
      Vector initial_state(oper->Width());
      initial_state = 0.0;
      std::vector<Vector> states;
      const real_t objective_value =
         RK4StageObjectiveForwardSweepFullStorage(
            *oper, state_fes, objective, initial_state, steps,
            /*start_time=*/0.0, time_step, states);

      // The custom stage-objective rollout must reproduce MFEM's accepted RK4
      // endpoint exactly enough for REVOLVE's ordinary RK4 replay to be valid.
      RK4Solver reference_solver;
      reference_solver.Init(*oper);
      Vector reference_state(initial_state);
      real_t reference_time = 0.0;
      for (int step = 0; step < steps; step++)
      {
         real_t accepted_step = time_step;
         reference_solver.Step(
            reference_state, reference_time, accepted_step);
      }
      const real_t forward_endpoint_error =
         RelativeError(comm, reference_state, states.back());
      MFEM_VERIFY(forward_endpoint_error < 5e-14,
                  "Custom stage-objective rollout disagrees with MFEM RK4.");

      Vector p_do, p_modified, p_naive;
      Vector g_do, g_modified, g_naive;
      RK4StageObjectiveAdjointDesignSweepFullStorage(
         *oper, state_fes, filter_fes, rho_tilde, material, objective,
         states, steps, /*start_time=*/0.0, time_step,
         RK4StageAdjointForm::DISCRETE_REVERSE_AD, p_do, g_do);
      RK4StageObjectiveAdjointDesignSweepFullStorage(
         *oper, state_fes, filter_fes, rho_tilde, material, objective,
         states, steps, /*start_time=*/0.0, time_step,
         RK4StageAdjointForm::TRANSFORMED_PARTITIONED,
         p_modified, g_modified);

      const NestedTimeGrid same_grid = NestedTimeGrid::Create(
         final_time, steps, steps);
      Vector terminal(initial_state.Size());
      terminal = 0.0;
      ContinuousAdjointDesignSweepFullStorage(
         *oper, state_fes, filter_fes, rho_tilde, material, objective,
         states, same_grid, terminal, p_naive, g_naive);

      Vector raw_do, raw_modified, raw_naive;
      filter.MultTranspose(g_do, raw_do);
      filter.MultTranspose(g_modified, raw_modified);
      filter.MultTranspose(g_naive, raw_naive);
      const real_t p_modified_error =
         RelativeError(comm, p_modified, p_do);
      const real_t g_modified_error =
         RelativeError(comm, raw_modified, raw_do);
      const real_t g_naive_error =
         RelativeError(comm, raw_naive, raw_do);
      MFEM_VERIFY(p_modified_error < 5e-13 &&
                  g_modified_error < 5e-13,
                  "Transformed RK4 adjoint does not match reverse AD.");
      MFEM_VERIFY(std::isfinite(g_naive_error) && g_naive_error > 1e-10,
                  "Naive Hermite adjoint did not exercise the expected mismatch.");

      Vector direction(raw_do);
      const real_t direction_norm = Norm(comm, direction);
      MFEM_VERIFY(direction_norm > 0.0,
                  "RK4-stage raw design gradient is unexpectedly zero.");
      direction /= direction_norm;
      const real_t projected = InnerProduct(comm, raw_do, direction);
      Vector base;
      rho.GetTrueDofs(base);

      const auto evaluate = [&](const Vector &candidate)
      {
         rho.SetFromTrueDofs(candidate);
         filter.Mult(rho, rho_tilde);
         std::unique_ptr<ElastodynamicsOperator> perturbed = make_operator();
         std::vector<Vector> perturbed_states;
         return RK4StageObjectiveForwardSweepFullStorage(
            *perturbed, state_fes, objective, initial_state, steps,
            /*start_time=*/0.0, time_step, perturbed_states);
      };

      real_t errors[2] = {0.0, 0.0};
      const real_t epsilons[2] = {1e-2, 5e-3};
      for (int level = 0; level < 2; level++)
      {
         Vector plus(base), minus(base);
         plus.Add(epsilons[level], direction);
         minus.Add(-epsilons[level], direction);
         const real_t centered =
            (evaluate(plus) - evaluate(minus)) /
            (2.0 * epsilons[level]);
         errors[level] = std::abs(centered - projected) /
            std::max({std::abs(centered), std::abs(projected), real_t(1e-30)});
      }
      rho.SetFromTrueDofs(base);
      filter.Mult(rho, rho_tilde);
      MFEM_VERIFY(errors[1] < 5e-4 && errors[0] > 2.0 * errors[1],
                  "RK4-stage DO gradient failed the centered design FD test.");

      if (Mpi::Root())
      {
         mfem::out << "RK4-stage adjoint regression passed\n"
                   << "  J_h: " << std::scientific << std::setprecision(12)
                   << objective_value << '\n'
                   << "  custom/MFEM RK4 endpoint relative error: "
                   << forward_endpoint_error << '\n'
                   << "  OD_modified p relative error: "
                   << p_modified_error << '\n'
                   << "  OD_modified raw-gradient relative error: "
                   << g_modified_error << '\n'
                   << "  OD_naive_Hermite raw-gradient relative error: "
                   << g_naive_error << '\n'
                   << "  centered-FD relative errors: "
                   << errors[0] << ", " << errors[1] << '\n';
      }
   }

   return 0;
}
