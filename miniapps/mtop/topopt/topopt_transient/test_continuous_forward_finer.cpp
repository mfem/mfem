// =============================================================================
// Fine-forward / coarse-continuous-adjoint temporal verification
// =============================================================================
//
// This regression fixes the nested-grid ratio N_f/N_a=3.  The odd ratio is
// intentional: every coarse-adjoint RK4 midpoint lies halfway through a fine
// forward interval, so the test exercises both forward and adjoint cubic-
// Hermite dense output.  It verifies
//
//   1. joint fourth-order refinement of the objective, initial adjoint, and
//      filtered/raw design gradients;
//   2. filtered and raw directional finite differences against the exact
//      fine-forward Simpson objective used by the run API.
//
// An opt-in study repeats the directional finite differences with the
// constrained ITERATIVE (consistent) mass matrix.  It is disabled in the
// default regression because each of its 16 objective perturbations rebuilds
// and solves with a design-dependent consistent mass matrix.
//
// The spatial mesh is fixed and deliberately tiny, so the reported rates are
// temporal rates for one semidiscrete problem.
// =============================================================================

#include "mfem.hpp"
#include "ElastodynamicsSolver.hpp"
#include "../../pde_filter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace mfem;

namespace
{

constexpr int forward_steps_per_adjoint_step = 3;

real_t GlobalDot(MPI_Comm comm, const Vector &a, const Vector &b)
{
   return InnerProduct(comm, a, b);
}

real_t GlobalNorm(MPI_Comm comm, const Vector &x)
{
   return std::sqrt(std::max(GlobalDot(comm, x, x), real_t(0.0)));
}

void Normalize(MPI_Comm comm, Vector &x)
{
   const real_t norm = GlobalNorm(comm, x);
   MFEM_VERIFY(std::isfinite(norm) && norm > 0.0,
               "Cannot normalize a zero or non-finite direction.");
   x /= norm;
}

real_t RelativeScalarError(real_t value, real_t reference)
{
   const real_t scale =
      std::max(std::abs(reference), real_t(100.0) *
               std::numeric_limits<real_t>::epsilon());
   return std::abs(value - reference) / scale;
}

real_t RelativeVectorError(MPI_Comm comm,
                           const Vector &value,
                           const Vector &reference)
{
   Vector difference(value);
   difference -= reference;
   const real_t scale =
      std::max(GlobalNorm(comm, reference), real_t(100.0) *
               std::numeric_limits<real_t>::epsilon());
   return GlobalNorm(comm, difference) / scale;
}

real_t GlobalAdmissibleStep(MPI_Comm comm,
                            const Vector &base,
                            const Vector &direction,
                            real_t lower,
                            real_t upper)
{
   MFEM_VERIFY(base.Size() == direction.Size() && lower < upper,
               "Invalid density perturbation data.");
   real_t local_step = std::numeric_limits<real_t>::infinity();
   for (int i = 0; i < base.Size(); i++)
   {
      if (direction[i] > 0.0)
      {
         local_step =
            std::min(local_step, (upper - base[i]) / direction[i]);
      }
      else if (direction[i] < 0.0)
      {
         local_step =
            std::min(local_step, (base[i] - lower) / (-direction[i]));
      }
   }
   real_t global_step = local_step;
   MPI_Allreduce(&local_step, &global_step, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_MIN, comm);
   MFEM_VERIFY(std::isfinite(global_step) && global_step > 0.0,
               "No admissible density perturbation is available.");
   return global_step;
}

real_t MinimumFinalOrder(const std::vector<real_t> &errors,
                         const char *quantity)
{
   MFEM_VERIFY(errors.size() >= 4,
               "Joint refinement needs at least four temporal levels.");
   std::vector<real_t> orders;
   for (std::size_t level = 0; level + 1 < errors.size(); level++)
   {
      MFEM_VERIFY(std::isfinite(errors[level]) &&
                  std::isfinite(errors[level + 1]) &&
                  errors[level] > 0.0 && errors[level + 1] > 0.0,
                  "Joint-refinement errors are zero or non-finite.");
      if (errors[level] >
          real_t(1000.0) * std::numeric_limits<real_t>::epsilon())
      {
         orders.push_back(
            std::log(errors[level] / errors[level + 1]) /
            std::log(real_t(2.0)));
      }
   }
   MFEM_VERIFY(orders.size() >= 2,
               "Joint-refinement errors reached roundoff too early.");
   const real_t result =
      std::min(orders[orders.size() - 2], orders.back());
   MFEM_VERIFY(std::isfinite(result),
               "Joint-refinement order is non-finite.");
   (void)quantity;
   return result;
}

struct TemporalSample
{
   int forward_steps = 0;
   int adjoint_steps = 0;
   ContinuousDesignRunResult run;
   Vector raw_gradient;
   real_t objective_error = 0.0;
   real_t initial_adjoint_error = 0.0;
   real_t filtered_gradient_error = 0.0;
   real_t raw_gradient_error = 0.0;
};

struct DirectionalFD
{
   std::string label;
   real_t projected_gradient = 0.0;
   real_t richardson_fd = 0.0;
   real_t relative_error = 0.0;
   real_t richardson_change = 0.0;
   std::array<real_t, 4> epsilon{};
   std::array<real_t, 4> centered_fd{};
};

struct CheckpointComparison
{
   std::string mass;
   int ratio = 0;
   int checkpoints = 0;
   real_t objective_error = 0.0;
   real_t initial_adjoint_error = 0.0;
   real_t filtered_gradient_error = 0.0;
   real_t raw_gradient_error = 0.0;
   real_t replay_audit_error = 0.0;
   long long controller_blocks = 0;
   long long controller_intervals = 0;
   long long local_blocks = 0;
   long long local_intervals = 0;
};

std::vector<Vector> BuildIndependentRK4Trajectory(
   ElastodynamicsOperator &oper,
   const Vector &initial_state,
   const NestedTimeGrid &grid)
{
   std::vector<Vector> states(grid.forward_steps + 1);
   states[0] = initial_state;
   Vector state(initial_state);
   RK4Solver solver;
   solver.Init(oper);
   real_t time = 0.0;
   for (int step = 0; step < grid.forward_steps; step++)
   {
      real_t accepted_step = grid.dt_forward;
      solver.Step(state, time, accepted_step);
      time = (step + 1) * grid.dt_forward;
      states[step + 1] = state;
   }
   return states;
}

real_t CheckPoisonedForwardBlockReplay(
   ElastodynamicsOperator &oper,
   MPI_Comm comm,
   const NestedTimeGrid &grid,
   const std::vector<Vector> &expected_states)
{
   MFEM_VERIFY(grid.relation == NestedTimeGridRelation::FORWARD_FINER &&
               expected_states.size() ==
               static_cast<std::size_t>(grid.forward_steps + 1),
               "Poisoned block replay received an invalid trajectory.");
   const int block = std::min(1, grid.adjoint_steps - 1);
   const int first_forward_step = block * grid.integer_ratio;
   Vector state_left(expected_states[first_forward_step]);
   const Vector saved_left(state_left);
   const real_t poison = std::numeric_limits<real_t>::quiet_NaN();
   const auto poison_vector = [&](Vector &vector)
   {
      vector.SetSize(oper.Width());
      vector = poison;
   };

   ForwardIntervalReplayWorkspace workspace;
   poison_vector(workspace.k2);
   poison_vector(workspace.k3);
   poison_vector(workspace.k4);
   poison_vector(workspace.y1);
   poison_vector(workspace.y2);
   poison_vector(workspace.y3);

   ForwardBlockData block_data;
   block_data.states.resize(grid.integer_ratio + 1);
   block_data.derivatives.resize(grid.integer_ratio + 1);
   for (Vector &state : block_data.states) { poison_vector(state); }
   for (Vector &slope : block_data.derivatives) { poison_vector(slope); }

   ReplayForwardBlock(
      oper, block, block * grid.dt_adjoint,
      grid.dt_forward, grid.integer_ratio,
      state_left, workspace, block_data);

   MFEM_VERIFY(block_data.index == block &&
               block_data.states.size() ==
               static_cast<std::size_t>(grid.integer_ratio + 1) &&
               block_data.derivatives.size() == block_data.states.size(),
               "Block replay escaped its q+1-state local workspace.");

   real_t maximum_error =
      RelativeVectorError(comm, state_left, saved_left);
   for (int local_node = 0;
        local_node <= grid.integer_ratio; local_node++)
   {
      const int global_node = first_forward_step + local_node;
      const Vector &state = block_data.states[local_node];
      const Vector &slope = block_data.derivatives[local_node];
      for (int i = 0; i < state.Size(); i++)
      {
         MFEM_VERIFY(std::isfinite(state[i]) && std::isfinite(slope[i]),
                     "Poison survived in replayed block state or slope.");
      }
      maximum_error = std::max(
         maximum_error,
         RelativeVectorError(comm, state, expected_states[global_node]));
      Vector expected_slope;
      EvalRHS(
         oper, expected_states[global_node],
         global_node * grid.dt_forward, expected_slope);
      maximum_error = std::max(
         maximum_error,
         RelativeVectorError(comm, slope, expected_slope));
   }
   return maximum_error;
}

template <typename Evaluate>
DirectionalFD CheckDirectionalFD(const std::string &label,
                                 MPI_Comm comm,
                                 const Vector &base,
                                 const Vector &direction,
                                 const Vector &gradient,
                                 Evaluate &&evaluate)
{
   DirectionalFD result;
   result.label = label;
   result.projected_gradient = GlobalDot(comm, gradient, direction);
   MFEM_VERIFY(std::isfinite(result.projected_gradient) &&
               std::abs(result.projected_gradient) > 1e-12,
               "Directional derivative is negligible or non-finite.");

   const real_t admissible =
      GlobalAdmissibleStep(comm, base, direction, 0.05, 0.95);
   const real_t initial_epsilon =
      std::min(real_t(0.04), real_t(0.20) * admissible);
   MFEM_VERIFY(initial_epsilon > 1e-5,
               "Admissible finite-difference perturbation is too small.");

   Vector candidate(base.Size());
   for (int level = 0; level < 4; level++)
   {
      result.epsilon[level] =
         initial_epsilon / std::pow(real_t(2.0), level);
      candidate = base;
      candidate.Add(result.epsilon[level], direction);
      const real_t plus = evaluate(candidate);
      candidate = base;
      candidate.Add(-result.epsilon[level], direction);
      const real_t minus = evaluate(candidate);
      result.centered_fd[level] =
         (plus - minus) / (2.0 * result.epsilon[level]);
      MFEM_VERIFY(std::isfinite(result.centered_fd[level]),
                  "Directional finite difference is non-finite.");
   }

   std::array<real_t, 3> richardson{};
   for (int level = 0; level < 3; level++)
   {
      richardson[level] =
         (4.0 * result.centered_fd[level + 1] -
          result.centered_fd[level]) / 3.0;
   }
   result.richardson_fd = richardson.back();
   const real_t scale =
      std::max({std::abs(result.projected_gradient),
                std::abs(result.richardson_fd), real_t(1e-14)});
   result.relative_error =
      std::abs(result.richardson_fd - result.projected_gradient) / scale;
   result.richardson_change =
      std::abs(richardson[2] - richardson[1]) / scale;
   return result;
}

void PrintDirectionalFD(const DirectionalFD &result)
{
   if (!Mpi::Root()) { return; }
   mfem::out << "\n" << result.label << " directional finite difference\n"
             << "  projected gradient: " << std::scientific
             << std::setprecision(12) << result.projected_gradient << '\n'
             << "  Richardson FD:      " << result.richardson_fd << '\n'
             << "  relative error:     " << result.relative_error << '\n'
             << "  Richardson change:  " << result.richardson_change << '\n'
             << "  epsilon          centered FD\n";
   for (int level = 0; level < 4; level++)
   {
      mfem::out << "  " << std::setw(12) << result.epsilon[level]
                << "  " << std::setw(19) << result.centered_fd[level]
                << '\n';
   }
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();
   MPI_Comm comm = MPI_COMM_WORLD;

   int nx = 4;
   int ny = 2;
   int minimum_adjoint_steps = 8;
   int reference_factor = 4;
   int consistent_fd_adjoint_steps = 32;
   real_t final_time = 1.0;
   bool run_consistent_fd = false;

   OptionsParser args(argc, argv);
   args.AddOption(&nx, "-nx", "--elements-x",
                  "Tiny-mesh elements in the x direction");
   args.AddOption(&ny, "-ny", "--elements-y",
                  "Tiny-mesh elements in the y direction");
   args.AddOption(&minimum_adjoint_steps, "-na0", "--minimum-adjoint-steps",
                  "Minimum coarsest-grid adjoint interval count");
   args.AddOption(&reference_factor, "-rf", "--reference-factor",
                  "Reference refinement beyond the finest tested grid");
   args.AddOption(&run_consistent_fd,
                  "-cfd", "--consistent-mass-fd",
                  "-no-cfd", "--no-consistent-mass-fd",
                  "Run the opt-in consistent-mass directional-FD study");
   args.AddOption(&consistent_fd_adjoint_steps,
                  "-cfd-na", "--consistent-fd-adjoint-steps",
                  "Coarse-adjoint intervals in the q=3 consistent-mass FD grid");
   args.AddOption(&final_time, "-tf", "--final-time",
                  "Fixed final time");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(std::cout); }
      return 1;
   }
   MFEM_VERIFY(nx >= 2 && ny >= 2 && minimum_adjoint_steps >= 4 &&
               reference_factor >= 4 && final_time > 0.0,
               "Invalid forward-finer test controls.");
   MFEM_VERIFY(
      !run_consistent_fd ||
      (consistent_fd_adjoint_steps >= 16 &&
       consistent_fd_adjoint_steps <=
          std::numeric_limits<int>::max() /
             forward_steps_per_adjoint_step),
      "The consistent-mass FD study needs at least 16 adjoint intervals "
      "and a non-overflowing q=3 forward grid.");

   Device device("cpu");
   Mesh serial_mesh = Mesh::MakeCartesian2D(
      nx, ny, Element::TRIANGLE, true, 1.0, 0.5);
   ParMesh mesh(comm, serial_mesh);

   constexpr int dimension = 2;
   H1_FECollection state_collection(/*order=*/2, dimension);
   H1_FECollection filter_collection(/*order=*/1, dimension);
   L2_FECollection control_collection(
      /*order=*/0, dimension, BasisType::GaussLobatto);
   ParFiniteElementSpace state_fes(
      &mesh, &state_collection, dimension);
   ParFiniteElementSpace filter_fes(&mesh, &filter_collection);
   ParFiniteElementSpace control_fes(&mesh, &control_collection);

   ParGridFunction rho(&control_fes);
   ParGridFunction rho_tilde(&filter_fes);
   rho = 0.65;

   toopt::PDEFilterOptions filter_options;
   filter_options.filter_radius = 0.08;
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

   ConstantCoefficient rho0_coefficient(material.rho0);
   ConstantCoefficient lambda0_coefficient(material.lambda0);
   ConstantCoefficient mu0_coefficient(material.mu0);
   SIMPCoefficient simp_mass(
      &rho_tilde, material.r_min, material.r_max, material.simp_p);
   SIMPCoefficient simp_stiffness(
      &rho_tilde, material.r_min, material.r_max, material.simp_p);
   ProductCoefficient mass_coefficient(simp_mass, rho0_coefficient);
   ProductCoefficient lambda_coefficient(
      simp_stiffness, lambda0_coefficient);
   ProductCoefficient mu_coefficient(simp_stiffness, mu0_coefficient);

   BoundaryLoadSpec load;
   load.domain_load = true;
   load.amplitude = 1.0;
   load.duration = final_time;
   load.time_profile = LoadTimeProfile::HARMONIC;
   load.frequency = 0.75;
   load.phase = 0.2;
   load.direction.SetSize(dimension);
   load.direction = 0.0;
   load.direction[1] = -1.0;
   DirectionalBoundaryLoadCoefficient load_coefficient(load.direction);

   ConstantCoefficient damping_coefficient(0.04);
   Array<int> exterior_boundary(mesh.bdr_attributes.Max());
   exterior_boundary = 0;
   Array<int> essential_boundary(mesh.bdr_attributes.Max());
   essential_boundary = 0;
   MFEM_VERIFY(essential_boundary.Size() >= 4,
               "Tiny Cartesian mesh has unexpected boundary attributes.");
   essential_boundary[3] = 1;

   ElastodynamicsOperator oper(
      state_fes, mass_coefficient, lambda_coefficient, mu_coefficient,
      load.amplitude, load.duration, load.time_profile,
      load.phase, load.frequency, load.bdr_attributes,
      load_coefficient, load.domain_load, &damping_coefficient,
      /*impedance=*/0.0, exterior_boundary, essential_boundary,
      MassSolverType::LUMPED, /*print_banner=*/false);

   Vector target_value(dimension);
   target_value = 0.0;
   target_value[0] = 1.0;
   target_value[1] = 0.25;
   auto tracking_region = std::make_unique<ConstantCoefficient>(1.0);
   auto tracking_mode =
      std::make_unique<VectorConstantCoefficient>(target_value);
   HarmonicDisplacementTrackingObjective objective(
      &state_fes, std::move(tracking_region), std::move(tracking_mode),
      /*amplitude=*/0.35, /*frequency=*/0.40, /*phase=*/0.1, comm);

   Vector initial_state(oper.Width());
   initial_state = 0.0;

   const real_t wave_limit = oper.EstimateLumpedRK4TimeStep();
   const real_t damping_limit = oper.EstimateLumpedRK4DampingTimeStep();
   const real_t stability_limit = std::min(wave_limit, damping_limit);
   MFEM_VERIFY(std::isfinite(stability_limit) && stability_limit > 0.0,
               "Tiny manufactured operator has no finite CFL estimate.");
   const int cfl_adjoint_steps = static_cast<int>(
      std::ceil(final_time / (0.30 * stability_limit)));
   const int base_adjoint_steps =
      std::max(minimum_adjoint_steps, cfl_adjoint_steps);
   ValidateLumpedRK4TimeStep(
      oper, final_time / base_adjoint_steps, /*print_report=*/false);

   constexpr int tested_levels = 4;
   std::vector<TemporalSample> samples(tested_levels);
   for (int level = 0; level < tested_levels; level++)
   {
      TemporalSample &sample = samples[level];
      sample.adjoint_steps = base_adjoint_steps * (1 << level);
      sample.forward_steps =
         forward_steps_per_adjoint_step * sample.adjoint_steps;
      const NestedTimeGrid grid = NestedTimeGrid::Create(
         final_time, sample.forward_steps, sample.adjoint_steps);
      MFEM_VERIFY(grid.relation == NestedTimeGridRelation::FORWARD_FINER &&
                  grid.integer_ratio == forward_steps_per_adjoint_step,
                  "Test did not construct the requested odd forward-finer grid.");
      sample.run = RunContinuousDesignFullStorage(
         oper, state_fes, filter_fes, rho_tilde, material, objective,
         initial_state, grid);
      filter.MultTranspose(sample.run.gradient_tilde, sample.raw_gradient);
   }

   // Full-storage/REVOLVE equivalence is checked on the coarsest temporal
   // fixture.  That is the most sensitive level numerically and keeps this
   // structural test cheap enough to run for two checkpoint budgets and both
   // odd/even forward-finer ratios.
   const NestedTimeGrid checkpoint_grid_q3 = NestedTimeGrid::Create(
      final_time, forward_steps_per_adjoint_step * base_adjoint_steps,
      base_adjoint_steps);
   const std::vector<Vector> expected_q3 =
      BuildIndependentRK4Trajectory(
         oper, initial_state, checkpoint_grid_q3);
   const real_t poisoned_block_error =
      CheckPoisonedForwardBlockReplay(
         oper, comm, checkpoint_grid_q3, expected_q3);

   constexpr int even_ratio = 2;
   const NestedTimeGrid checkpoint_grid_q2 = NestedTimeGrid::Create(
      final_time, even_ratio * base_adjoint_steps, base_adjoint_steps);
   ContinuousDesignRunResult full_q2 = RunContinuousDesignFullStorage(
      oper, state_fes, filter_fes, rho_tilde, material, objective,
      initial_state, checkpoint_grid_q2);
   Vector full_q2_raw(control_fes.GetTrueVSize());
   filter.MultTranspose(full_q2.gradient_tilde, full_q2_raw);
   const std::vector<Vector> expected_q2 =
      BuildIndependentRK4Trajectory(
         oper, initial_state, checkpoint_grid_q2);

   std::vector<CheckpointComparison> checkpoint_comparisons;
   const auto compare_checkpointed =
      [&](ElastodynamicsOperator &checked_oper,
          const std::string &mass_label,
          const NestedTimeGrid &grid,
          const ContinuousDesignRunResult &full,
          const Vector &full_raw,
          const std::vector<Vector> &expected,
          int checkpoint_count)
      {
         std::vector<int> replay_visits(grid.forward_steps, 0);
         real_t replay_audit_error = 0.0;
         const auto replay_audit =
            [&](int step,
                const Vector &state_left,
                const ForwardIntervalData &interval)
            {
               MFEM_VERIFY(step >= 0 && step < grid.forward_steps &&
                           interval.index == step,
                           "Checkpoint replay audit received a bad fine index.");
               const real_t expected_time = step * grid.dt_forward;
               const real_t time_scale =
                  std::max(real_t(1.0), std::abs(expected_time));
               MFEM_VERIFY(
                  std::abs(interval.t_left - expected_time) <=
                  128.0 * std::numeric_limits<real_t>::epsilon() * time_scale,
                  "Checkpoint replay audit received a bad physical time.");
               replay_visits[step]++;
               replay_audit_error = std::max(
                  replay_audit_error,
                  RelativeVectorError(
                     comm, state_left, expected[step]));
               replay_audit_error = std::max(
                  replay_audit_error,
                  RelativeVectorError(
                     comm, interval.x_left, expected[step]));
               replay_audit_error = std::max(
                  replay_audit_error,
                  RelativeVectorError(
                     comm, interval.x_right, expected[step + 1]));

               Vector expected_left_slope, expected_right_slope;
               EvalRHS(
                  checked_oper, expected[step], expected_time,
                  expected_left_slope);
               EvalRHS(
                  checked_oper, expected[step + 1],
                  expected_time + grid.dt_forward,
                  expected_right_slope);
               replay_audit_error = std::max(
                  replay_audit_error,
                  RelativeVectorError(
                     comm, interval.f_left, expected_left_slope));
               replay_audit_error = std::max(
                  replay_audit_error,
                  RelativeVectorError(
                     comm, interval.f_right, expected_right_slope));
            };

         ContinuousDesignRunResult replay =
            RunContinuousDesignCheckpointed(
               checked_oper, state_fes, filter_fes, rho_tilde, material,
               objective,
               initial_state, grid, checkpoint_count, replay_audit);
         Vector replay_raw(control_fes.GetTrueVSize());
         filter.MultTranspose(replay.gradient_tilde, replay_raw);

         for (int visits : replay_visits)
         {
            MFEM_VERIFY(visits == 1,
                        "Local block replay did not audit each fine interval "
                        "exactly once.");
         }
         MFEM_VERIFY(
            replay.controller_recomputed_intervals ==
            grid.integer_ratio * replay.controller_recomputed_blocks,
            "Controller block/fine-interval replay counts disagree.");
         MFEM_VERIFY(
            replay.locally_replayed_blocks == grid.adjoint_steps &&
            replay.locally_replayed_intervals == grid.forward_steps,
            "Local replay counts do not match N_a blocks and N_f intervals.");

         CheckpointComparison comparison;
         comparison.mass = mass_label;
         comparison.ratio = grid.integer_ratio;
         comparison.checkpoints = checkpoint_count;
         comparison.objective_error =
            RelativeScalarError(replay.objective, full.objective);
         comparison.initial_adjoint_error =
            RelativeVectorError(
               comm, replay.initial_adjoint, full.initial_adjoint);
         comparison.filtered_gradient_error =
            RelativeVectorError(
               comm, replay.gradient_tilde, full.gradient_tilde);
         comparison.raw_gradient_error =
            RelativeVectorError(comm, replay_raw, full_raw);
         comparison.replay_audit_error = replay_audit_error;
         comparison.controller_blocks = replay.controller_recomputed_blocks;
         comparison.controller_intervals =
            replay.controller_recomputed_intervals;
         comparison.local_blocks = replay.locally_replayed_blocks;
         comparison.local_intervals = replay.locally_replayed_intervals;
         checkpoint_comparisons.push_back(comparison);
      };

   compare_checkpointed(
      oper, "lumped", checkpoint_grid_q3,
      samples.front().run, samples.front().raw_gradient,
      expected_q3, /*checkpoint_count=*/1);
   compare_checkpointed(
      oper, "lumped", checkpoint_grid_q3,
      samples.front().run, samples.front().raw_gradient,
      expected_q3, /*checkpoint_count=*/2);
   compare_checkpointed(
      oper, "lumped", checkpoint_grid_q2, full_q2, full_q2_raw,
      expected_q2, /*checkpoint_count=*/1);
   compare_checkpointed(
      oper, "lumped", checkpoint_grid_q2, full_q2, full_q2_raw,
      expected_q2, /*checkpoint_count=*/2);

   // Consistent-mass continuous replay uses the same semidiscrete mass matrix
   // in the forward RHS, transpose action, and design contraction.  This
   // focused check keeps the q=3 coarse fixture fixed and asks whether storage
   // policy changes any result.  The opt-in -cfd study below supplies the more
   // expensive refined consistent-mass directional-FD coverage.
   ElastodynamicsOperator consistent_oper(
      state_fes, mass_coefficient, lambda_coefficient, mu_coefficient,
      load.amplitude, load.duration, load.time_profile,
      load.phase, load.frequency, load.bdr_attributes,
      load_coefficient, load.domain_load, &damping_coefficient,
      /*impedance=*/0.0, exterior_boundary, essential_boundary,
      MassSolverType::ITERATIVE, /*print_banner=*/false);
   const int local_essential_dofs =
      consistent_oper.GetEssentialTrueDofs().Size();
   int global_essential_dofs = 0;
   MPI_Allreduce(&local_essential_dofs, &global_essential_dofs, 1,
                 MPI_INT, MPI_SUM, comm);
   MFEM_VERIFY(global_essential_dofs > 0,
               "Consistent-mass checks require a nonempty essential boundary.");
   ContinuousDesignRunResult consistent_full =
      RunContinuousDesignFullStorage(
         consistent_oper, state_fes, filter_fes, rho_tilde, material,
         objective, initial_state, checkpoint_grid_q3);
   Vector consistent_full_raw(control_fes.GetTrueVSize());
   filter.MultTranspose(
      consistent_full.gradient_tilde, consistent_full_raw);
   const std::vector<Vector> expected_consistent_q3 =
      BuildIndependentRK4Trajectory(
         consistent_oper, initial_state, checkpoint_grid_q3);
   compare_checkpointed(
      consistent_oper, "consistent", checkpoint_grid_q3,
      consistent_full, consistent_full_raw,
      expected_consistent_q3, /*checkpoint_count=*/1);
   compare_checkpointed(
      consistent_oper, "consistent", checkpoint_grid_q3,
      consistent_full, consistent_full_raw,
      expected_consistent_q3, /*checkpoint_count=*/2);

   MFEM_VERIFY(samples.back().adjoint_steps <=
               std::numeric_limits<int>::max() / reference_factor,
               "Reference adjoint interval count overflows int.");
   const int reference_adjoint_steps =
      reference_factor * samples.back().adjoint_steps;
   const int reference_forward_steps =
      forward_steps_per_adjoint_step * reference_adjoint_steps;
   const NestedTimeGrid reference_grid = NestedTimeGrid::Create(
      final_time, reference_forward_steps, reference_adjoint_steps);
   ContinuousDesignRunResult reference =
      RunContinuousDesignFullStorage(
         oper, state_fes, filter_fes, rho_tilde, material, objective,
         initial_state, reference_grid);
   Vector reference_raw_gradient(control_fes.GetTrueVSize());
   filter.MultTranspose(reference.gradient_tilde, reference_raw_gradient);

   std::vector<real_t> objective_errors;
   std::vector<real_t> initial_adjoint_errors;
   std::vector<real_t> filtered_gradient_errors;
   std::vector<real_t> raw_gradient_errors;
   for (TemporalSample &sample : samples)
   {
      sample.objective_error =
         RelativeScalarError(sample.run.objective, reference.objective);
      sample.initial_adjoint_error =
         RelativeVectorError(
            comm, sample.run.initial_adjoint, reference.initial_adjoint);
      sample.filtered_gradient_error =
         RelativeVectorError(
            comm, sample.run.gradient_tilde, reference.gradient_tilde);
      sample.raw_gradient_error =
         RelativeVectorError(
            comm, sample.raw_gradient, reference_raw_gradient);
      objective_errors.push_back(sample.objective_error);
      initial_adjoint_errors.push_back(sample.initial_adjoint_error);
      filtered_gradient_errors.push_back(sample.filtered_gradient_error);
      raw_gradient_errors.push_back(sample.raw_gradient_error);
   }

   const real_t objective_order =
      MinimumFinalOrder(objective_errors, "objective");
   const real_t initial_adjoint_order =
      MinimumFinalOrder(initial_adjoint_errors, "initial adjoint");
   const real_t filtered_gradient_order =
      MinimumFinalOrder(filtered_gradient_errors, "filtered gradient");
   const real_t raw_gradient_order =
      MinimumFinalOrder(raw_gradient_errors, "raw gradient");

   Vector filtered_base, raw_base;
   rho_tilde.GetTrueDofs(filtered_base);
   rho.GetTrueDofs(raw_base);
   Vector filtered_direction(reference.gradient_tilde);
   Vector raw_direction(reference_raw_gradient);
   Normalize(comm, filtered_direction);
   Normalize(comm, raw_direction);

   // Rebuild the design-dependent operator for every perturbation.  The
   // objective uses one Simpson panel on every fine forward RK4 interval,
   // exactly as RunContinuousDesignFullStorage(grid) does for FORWARD_FINER.
   const auto evaluate_fine_forward_objective = [&]()
   {
      ConstantCoefficient perturbed_rho0(material.rho0);
      ConstantCoefficient perturbed_lambda0(material.lambda0);
      ConstantCoefficient perturbed_mu0(material.mu0);
      SIMPCoefficient perturbed_simp_mass(
         &rho_tilde, material.r_min, material.r_max, material.simp_p);
      SIMPCoefficient perturbed_simp_stiffness(
         &rho_tilde, material.r_min, material.r_max, material.simp_p);
      ProductCoefficient perturbed_mass(
         perturbed_simp_mass, perturbed_rho0);
      ProductCoefficient perturbed_lambda(
         perturbed_simp_stiffness, perturbed_lambda0);
      ProductCoefficient perturbed_mu(
         perturbed_simp_stiffness, perturbed_mu0);
      ElastodynamicsOperator perturbed_oper(
         state_fes, perturbed_mass, perturbed_lambda, perturbed_mu,
         load.amplitude, load.duration, load.time_profile,
         load.phase, load.frequency, load.bdr_attributes,
         load_coefficient, load.domain_load, &damping_coefficient,
         /*impedance=*/0.0, exterior_boundary, essential_boundary,
         MassSolverType::LUMPED, /*print_banner=*/false);
      std::vector<Vector> states;
      return ContinuousForwardSweepFullStorage(
         perturbed_oper, state_fes, objective, initial_state,
         reference_forward_steps, /*start_time=*/0.0,
         reference_grid.dt_forward, /*fine_steps_per_interval=*/1,
         states);
   };

   const auto evaluate_filtered = [&](const Vector &candidate)
   {
      rho_tilde.SetFromTrueDofs(candidate);
      return evaluate_fine_forward_objective();
   };
   const auto evaluate_raw = [&](const Vector &candidate)
   {
      rho.SetFromTrueDofs(candidate);
      filter.Mult(rho, rho_tilde);
      return evaluate_fine_forward_objective();
   };

   const DirectionalFD filtered_fd = CheckDirectionalFD(
      "Filtered design", comm, filtered_base, filtered_direction,
      reference.gradient_tilde, evaluate_filtered);
   rho.SetFromTrueDofs(raw_base);
   rho_tilde.SetFromTrueDofs(filtered_base);

   const DirectionalFD raw_fd = CheckDirectionalFD(
      "Raw design", comm, raw_base, raw_direction,
      reference_raw_gradient, evaluate_raw);

   rho.SetFromTrueDofs(raw_base);
   rho_tilde.SetFromTrueDofs(filtered_base);

   DirectionalFD consistent_filtered_fd;
   DirectionalFD consistent_raw_fd;
   real_t consistent_fd_wall_seconds = 0.0;
   if (run_consistent_fd)
   {
      // Keep q=3 so the consistent-mass study exercises the same genuinely
      // nested dense-output path as the primary regression.  The adjoint step
      // is the larger step, so validating it is sufficient for stability;
      // validate the fine forward step too to make that contract explicit.
      const NestedTimeGrid consistent_fd_grid = NestedTimeGrid::Create(
         final_time,
         forward_steps_per_adjoint_step * consistent_fd_adjoint_steps,
         consistent_fd_adjoint_steps);
      MFEM_VERIFY(
         consistent_fd_grid.relation ==
            NestedTimeGridRelation::FORWARD_FINER &&
         consistent_fd_grid.integer_ratio == forward_steps_per_adjoint_step,
         "Consistent-mass FD study did not construct its q=3 grid.");
      ValidateRK4TimeStep(
         consistent_oper, consistent_fd_grid.dt_forward,
         /*print_report=*/false);
      ValidateRK4TimeStep(
         consistent_oper, consistent_fd_grid.dt_adjoint,
         /*print_report=*/false);

      rho.SetFromTrueDofs(raw_base);
      rho_tilde.SetFromTrueDofs(filtered_base);
      MPI_Barrier(comm);
      const real_t local_start = MPI_Wtime();

      ContinuousDesignRunResult consistent_fd_run =
         RunContinuousDesignFullStorage(
            consistent_oper, state_fes, filter_fes, rho_tilde, material,
            objective, initial_state, consistent_fd_grid);
      Vector consistent_fd_raw_gradient(control_fes.GetTrueVSize());
      filter.MultTranspose(
         consistent_fd_run.gradient_tilde, consistent_fd_raw_gradient);

      Vector consistent_filtered_direction(
         consistent_fd_run.gradient_tilde);
      Vector consistent_raw_direction(consistent_fd_raw_gradient);
      Normalize(comm, consistent_filtered_direction);
      Normalize(comm, consistent_raw_direction);

      // Reassembly is essential here: changing rho_tilde changes the
      // consistent mass matrix itself, not just a diagonal inverse cache.
      const auto evaluate_consistent_fine_forward_objective = [&]()
      {
         ConstantCoefficient perturbed_rho0(material.rho0);
         ConstantCoefficient perturbed_lambda0(material.lambda0);
         ConstantCoefficient perturbed_mu0(material.mu0);
         SIMPCoefficient perturbed_simp_mass(
            &rho_tilde, material.r_min, material.r_max, material.simp_p);
         SIMPCoefficient perturbed_simp_stiffness(
            &rho_tilde, material.r_min, material.r_max, material.simp_p);
         ProductCoefficient perturbed_mass(
            perturbed_simp_mass, perturbed_rho0);
         ProductCoefficient perturbed_lambda(
            perturbed_simp_stiffness, perturbed_lambda0);
         ProductCoefficient perturbed_mu(
            perturbed_simp_stiffness, perturbed_mu0);
         ElastodynamicsOperator perturbed_oper(
            state_fes, perturbed_mass, perturbed_lambda, perturbed_mu,
            load.amplitude, load.duration, load.time_profile,
            load.phase, load.frequency, load.bdr_attributes,
            load_coefficient, load.domain_load, &damping_coefficient,
            /*impedance=*/0.0, exterior_boundary, essential_boundary,
            MassSolverType::ITERATIVE, /*print_banner=*/false);
         std::vector<Vector> states;
         return ContinuousForwardSweepFullStorage(
            perturbed_oper, state_fes, objective, initial_state,
            consistent_fd_grid.forward_steps, /*start_time=*/0.0,
            consistent_fd_grid.dt_forward,
            /*fine_steps_per_interval=*/1, states);
      };

      const auto evaluate_consistent_filtered = [&](const Vector &candidate)
      {
         rho_tilde.SetFromTrueDofs(candidate);
         return evaluate_consistent_fine_forward_objective();
      };
      const auto evaluate_consistent_raw = [&](const Vector &candidate)
      {
         rho.SetFromTrueDofs(candidate);
         filter.Mult(rho, rho_tilde);
         return evaluate_consistent_fine_forward_objective();
      };

      consistent_filtered_fd = CheckDirectionalFD(
         "Consistent mass, filtered design", comm, filtered_base,
         consistent_filtered_direction, consistent_fd_run.gradient_tilde,
         evaluate_consistent_filtered);
      rho.SetFromTrueDofs(raw_base);
      rho_tilde.SetFromTrueDofs(filtered_base);

      consistent_raw_fd = CheckDirectionalFD(
         "Consistent mass, raw design", comm, raw_base,
         consistent_raw_direction, consistent_fd_raw_gradient,
         evaluate_consistent_raw);

      rho.SetFromTrueDofs(raw_base);
      rho_tilde.SetFromTrueDofs(filtered_base);
      MPI_Barrier(comm);
      const real_t local_elapsed = MPI_Wtime() - local_start;
      MPI_Allreduce(
         &local_elapsed, &consistent_fd_wall_seconds, 1,
         MPITypeMap<real_t>::mpi_type, MPI_MAX, comm);
   }

   if (Mpi::Root())
   {
      mfem::out
         << "\n=== Fine-Forward / Coarse-Adjoint Continuous Gradient ===\n"
         << "Spatial fixture: " << nx << " x " << ny
         << " triangles, state order 2, design order 1/0\n"
         << "Fixed odd ratio q=N_f/N_a="
         << forward_steps_per_adjoint_step << '\n'
         << "T=" << std::scientific << std::setprecision(6) << final_time
         << ", CFL limit=" << stability_limit
         << ", base N_a=" << base_adjoint_steps
         << ", reference (N_f,N_a)=(" << reference_forward_steps
         << ',' << reference_adjoint_steps << ")\n\n"
         << " N_f   N_a          dt_f           dt_a           rel J"
            "          rel p(0)       rel g_tilde       rel g_raw\n";
      for (const TemporalSample &sample : samples)
      {
         mfem::out << std::setw(4) << sample.forward_steps
                   << "  " << std::setw(4) << sample.adjoint_steps
                   << "  " << std::setw(13)
                   << final_time / sample.forward_steps
                   << "  " << std::setw(13)
                   << final_time / sample.adjoint_steps
                   << "  " << std::setw(13) << sample.objective_error
                   << "  " << std::setw(13) << sample.initial_adjoint_error
                   << "  " << std::setw(15)
                   << sample.filtered_gradient_error
                   << "  " << std::setw(13) << sample.raw_gradient_error
                   << '\n';
      }
      mfem::out
         << "\nMinimum of final two joint-refinement orders:\n"
         << "  objective:         " << objective_order << '\n'
         << "  initial adjoint:   " << initial_adjoint_order << '\n'
         << "  filtered gradient: " << filtered_gradient_order << '\n'
         << "  raw gradient:      " << raw_gradient_order << '\n'
         << "\nFull-storage / REVOLVE equivalence:\n"
         << " mass         q   C       rel J       rel p0    rel g_tilde"
            "      rel g_raw      audit    controller(block/fine)"
            "   local(block/fine)\n";
      for (const CheckpointComparison &comparison : checkpoint_comparisons)
      {
         mfem::out
            << std::setw(10) << comparison.mass
            << "  " << std::setw(2) << comparison.ratio
            << "  " << std::setw(2) << comparison.checkpoints
            << "  " << std::setw(10) << comparison.objective_error
            << "  " << std::setw(10) << comparison.initial_adjoint_error
            << "  " << std::setw(12) << comparison.filtered_gradient_error
            << "  " << std::setw(12) << comparison.raw_gradient_error
            << "  " << std::setw(9) << comparison.replay_audit_error
            << "      " << comparison.controller_blocks << '/'
            << comparison.controller_intervals
            << "                 " << comparison.local_blocks << '/'
            << comparison.local_intervals << '\n';
      }
      mfem::out << "Poisoned q=3 block replay error: "
                << poisoned_block_error << '\n'
                << "Consistent-mass storage/replay equivalence is always "
                   "checked; use -cfd for its refined directional-FD study.\n";
   }
   PrintDirectionalFD(filtered_fd);
   PrintDirectionalFD(raw_fd);
   if (run_consistent_fd)
   {
      if (Mpi::Root())
      {
         mfem::out
            << "\nOpt-in constrained consistent-mass FD study: q=3, "
            << "(N_f,N_a)=("
            << forward_steps_per_adjoint_step *
                  consistent_fd_adjoint_steps
            << ',' << consistent_fd_adjoint_steps << "), wall time="
            << std::fixed << std::setprecision(3)
            << consistent_fd_wall_seconds << " s\n";
      }
      PrintDirectionalFD(consistent_filtered_fd);
      PrintDirectionalFD(consistent_raw_fd);
   }

   MFEM_VERIFY(objective_order > 3.45,
               "Fine-forward objective did not converge at fourth order.");
   MFEM_VERIFY(initial_adjoint_order > 3.25,
               "Coarse continuous adjoint did not converge at fourth order.");
   MFEM_VERIFY(filtered_gradient_order > 3.15,
               "Filtered forward-finer gradient did not converge at fourth order.");
   MFEM_VERIFY(raw_gradient_order > 3.15,
               "Raw forward-finer gradient did not converge at fourth order.");
   MFEM_VERIFY(samples.back().objective_error < 3e-5 &&
               samples.back().initial_adjoint_error < 5e-5 &&
               samples.back().filtered_gradient_error < 8e-5 &&
               samples.back().raw_gradient_error < 8e-5,
               "Finest forward-finer grid is not close to its reference.");

   MFEM_VERIFY(poisoned_block_error < 1e-13,
               "Poisoned interval-local forward replay is not exact.");
   for (const CheckpointComparison &comparison : checkpoint_comparisons)
   {
      MFEM_VERIFY(comparison.objective_error < 1e-13 &&
                  comparison.initial_adjoint_error < 1e-12 &&
                  comparison.filtered_gradient_error < 1e-12 &&
                  comparison.raw_gradient_error < 1e-12 &&
                  comparison.replay_audit_error < 1e-13,
                  "Forward-finer full-storage/REVOLVE equivalence failed.");
   }

   const auto verify_fd = [](const DirectionalFD &result,
                             real_t tolerance,
                             real_t richardson_change_tolerance)
   {
      MFEM_VERIFY(result.relative_error < tolerance,
                  "Forward-finer gradient failed its directional FD check.");
      MFEM_VERIFY(result.richardson_change < richardson_change_tolerance,
                  "Forward-finer finite difference is not converged.");
   };
   verify_fd(filtered_fd, 8e-5, 3e-5);
   verify_fd(raw_fd, 1.2e-4, 3e-5);
   if (run_consistent_fd)
   {
      // At the default reduced grid (N_f,N_a)=(96,32), the measured relative
      // errors are 6.7e-8 and 7.1e-8 and the Richardson changes are below
      // 7.2e-8.  The 2e-6 limits leave margin for MPI reduction and iterative-
      // solver variation while still detecting a broken mass sensitivity.
      verify_fd(consistent_filtered_fd, 2e-6, 2e-6);
      verify_fd(consistent_raw_fd, 2e-6, 2e-6);
   }

   if (Mpi::Root())
   {
      mfem::out
         << "\nAll fine-forward/coarse-adjoint joint-refinement, "
            "checkpoint-replay, and directional-FD checks passed.\n";
   }
   return 0;
}
