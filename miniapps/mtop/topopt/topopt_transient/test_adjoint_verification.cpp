// =============================================================================
// Adjoint verification for the transient elastodynamics operator
// =============================================================================
//
// These checks intentionally stop before design sensitivities and MMA:
//   1. <J(x) v, w> = <v, J(x)^T w>
//   2. <D Phi_h(x) v, w> = <v, D Phi_h(x)^T w> for one RK4 step
//   3. the same identity for an n-step RK4 map
//
// MFEM in this checkout does not expose RK4 AdjointStep, so the RK4 transpose
// used here is a local reverse-mode transcription of MFEM's RK4Solver::Step.
//
// =============================================================================

#include "mfem.hpp"
#include "ElastodynamicsSolver.hpp"
#include "../../pde_filter.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

using namespace mfem;
using namespace std;

namespace
{

double GlobalDot(MPI_Comm comm, const Vector &a, const Vector &b)
{
   return InnerProduct(comm, a, b);
}

double RelativeError(double lhs, double rhs)
{
   double scale = 1.0;
   scale = max(scale, fabs(lhs));
   scale = max(scale, fabs(rhs));
   return fabs(lhs - rhs) / scale;
}

void RandomState(Vector &x, int seed)
{
   x.Randomize(seed);
   x *= 0.1;
}

void Normalize(MPI_Comm comm, Vector &x)
{
   const real_t norm = std::sqrt(GlobalDot(comm, x, x));
   MFEM_VERIFY(norm > 0.0, "Cannot normalize a zero vector.");
   x /= norm;
}

class ZeroInstantaneousObjective : public TimeIntegratedObjective
{
protected:
   void AssembleStateGradientScaled(
      const ParGridFunction &, real_t, real_t,
      ParLinearForm &gradient) override
   {
      gradient = 0.0;
   }

public:
   ZeroInstantaneousObjective(
      ParFiniteElementSpace *fespace, MPI_Comm comm)
      : TimeIntegratedObjective(fespace, comm) {}

   real_t EvaluateInstantaneous(
      const ParGridFunction &, real_t) override
   {
      return 0.0;
   }
};

class RecordingForwardStateProvider : public ForwardStateProvider
{
private:
   const ForwardStateProvider &provider_;

public:
   mutable std::vector<real_t> requested_times;

   explicit RecordingForwardStateProvider(
      const ForwardStateProvider &provider)
      : provider_(provider) {}

   void Evaluate(real_t physical_time, Vector &state) const override
   {
      requested_times.push_back(physical_time);
      provider_.Evaluate(physical_time, state);
   }
};

void CheckNestedTimeGridContract()
{
   const NestedTimeGrid same = NestedTimeGrid::Create(1.0, 8, 8);
   MFEM_VERIFY(same.relation == NestedTimeGridRelation::SAME &&
               same.integer_ratio == 1,
               "Same-grid nested-time contract is incorrect.");

   const NestedTimeGrid forward_finer = NestedTimeGrid::Create(1.0, 16, 4);
   MFEM_VERIFY(forward_finer.relation ==
               NestedTimeGridRelation::FORWARD_FINER &&
               forward_finer.integer_ratio == 4,
               "Forward-finer nested-time contract is incorrect.");

   const NestedTimeGrid adjoint_finer = NestedTimeGrid::Create(1.0, 4, 16);
   MFEM_VERIFY(adjoint_finer.relation ==
               NestedTimeGridRelation::ADJOINT_FINER &&
               adjoint_finer.integer_ratio == 4,
               "Adjoint-finer nested-time contract is incorrect.");

   std::vector<Vector> states(5);
   for (int i = 0; i < 5; i++)
   {
      states[i].SetSize(2);
      states[i][0] = i;
      states[i][1] = -2.0 * i;
   }
   ExactStoredForwardStateProvider provider(states, 0.0, 0.25);
   Vector state;
   provider.Evaluate(0.5, state);
   MFEM_VERIFY(state.Size() == 2 && state[0] == 2.0 && state[1] == -4.0,
               "Exact stored forward-state lookup returned the wrong node.");
}

struct HermiteProviderCheck
{
   double cubic_max_error;
   double oscillator_finest_error;
   double oscillator_minimum_order;
};

HermiteProviderCheck CheckCubicHermiteForwardStateProvider()
{
   const real_t start_time = 0.37;
   const real_t cubic_step = 0.2;
   const int cubic_intervals = 4;
   std::vector<Vector> cubic_states(cubic_intervals + 1);
   std::vector<Vector> cubic_derivatives(cubic_intervals + 1);

   const auto cubic_value = [](real_t time, Vector &value)
   {
      value.SetSize(2);
      value[0] = 1.0 + 2.0*time - 3.0*time*time
                 + 0.5*time*time*time;
      value[1] = -0.5 + time + 2.0*time*time
                 - time*time*time;
   };
   const auto cubic_derivative = [](real_t time, Vector &value)
   {
      value.SetSize(2);
      value[0] = 2.0 - 6.0*time + 1.5*time*time;
      value[1] = 1.0 + 4.0*time - 3.0*time*time;
   };

   for (int node = 0; node <= cubic_intervals; node++)
   {
      const real_t time = start_time + node * cubic_step;
      cubic_value(time, cubic_states[node]);
      cubic_derivative(time, cubic_derivatives[node]);
   }
   CubicHermiteForwardStateProvider cubic_provider(
      cubic_states, cubic_derivatives, start_time, cubic_step);

   const real_t query_coordinates[] =
      {3.75, 0.25, 2.5, 1.125, 3.0, 0.5, 2.875};
   double cubic_max_error = 0.0;
   Vector reconstructed, exact;
   const auto record_cubic_error = [&](double error)
   {
      MFEM_VERIFY(std::isfinite(error),
                  "Cubic Hermite provider produced a non-finite error.");
      cubic_max_error = std::max(cubic_max_error, error);
   };
   for (real_t coordinate : query_coordinates)
   {
      const real_t time = start_time + coordinate * cubic_step;
      cubic_provider.Evaluate(time, reconstructed);
      cubic_value(time, exact);
      reconstructed -= exact;
      record_cubic_error(
         static_cast<double>(reconstructed.Normlinf()));
   }

   // Exercise roundoff-level clamping at both ends and exact interior nodes.
   const real_t boundary_perturbation =
      64.0 * std::numeric_limits<real_t>::epsilon() * cubic_step;
   cubic_provider.Evaluate(
      start_time - boundary_perturbation, reconstructed);
   reconstructed -= cubic_states.front();
   record_cubic_error(
      static_cast<double>(reconstructed.Normlinf()));
   cubic_provider.Evaluate(
      start_time + cubic_intervals * cubic_step + boundary_perturbation,
      reconstructed);
   reconstructed -= cubic_states.back();
   record_cubic_error(
      static_cast<double>(reconstructed.Normlinf()));
   for (int node = 0; node <= cubic_intervals; node++)
   {
      cubic_provider.Evaluate(
         start_time + node * cubic_step, reconstructed);
      reconstructed -= cubic_states[node];
      record_cubic_error(
         static_cast<double>(reconstructed.Normlinf()));
   }
   MFEM_VERIFY(cubic_max_error < 5e-13,
               "Cubic Hermite provider does not reproduce a cubic exactly.");

   const auto oscillator_rhs = [](const Vector &state, Vector &slope)
   {
      slope.SetSize(2);
      slope[0] = state[1];
      slope[1] = -state[0];
   };
   const auto oscillator_error = [&](int intervals)
   {
      const real_t final_time = start_time + 1.0;
      const real_t step = (final_time - start_time) / intervals;
      std::vector<Vector> states(intervals + 1);
      std::vector<Vector> derivatives(intervals + 1);
      Vector state(2), k1(2), k2(2), k3(2), k4(2), stage(2);
      state[0] = std::cos(start_time);
      state[1] = -std::sin(start_time);

      for (int node = 0; node <= intervals; node++)
      {
         states[node] = state;
         oscillator_rhs(state, derivatives[node]);
         if (node == intervals) { break; }

         oscillator_rhs(state, k1);
         stage = state;
         stage.Add(0.5 * step, k1);
         oscillator_rhs(stage, k2);
         stage = state;
         stage.Add(0.5 * step, k2);
         oscillator_rhs(stage, k3);
         stage = state;
         stage.Add(step, k3);
         oscillator_rhs(stage, k4);

         state.Add(step / 6.0, k1);
         state.Add(step / 3.0, k2);
         state.Add(step / 3.0, k3);
         state.Add(step / 6.0, k4);
      }

      CubicHermiteForwardStateProvider provider(
         states, derivatives, start_time, step);
      double maximum_error = 0.0;
      const real_t stage_fractions[] = {0.75, 0.25, 0.5};
      // Deliberately query intervals in reverse order to ensure the provider
      // carries no monotone-query assumption.
      for (int interval = intervals - 1; interval >= 0; interval--)
      {
         for (real_t fraction : stage_fractions)
         {
            const real_t time =
               start_time + (interval + fraction) * step;
            provider.Evaluate(time, reconstructed);
            const real_t error_u =
               reconstructed[0] - std::cos(time);
            const real_t error_v =
               reconstructed[1] + std::sin(time);
            const double sample_error =
               static_cast<double>(
                  std::sqrt(error_u*error_u + error_v*error_v));
            MFEM_VERIFY(std::isfinite(sample_error),
                        "Oscillator reconstruction produced a non-finite error.");
            maximum_error = std::max(maximum_error, sample_error);
         }
      }
      return maximum_error;
   };

   const int interval_counts[] = {8, 16, 32, 64};
   double errors[4];
   for (int level = 0; level < 4; level++)
   {
      errors[level] = oscillator_error(interval_counts[level]);
      MFEM_VERIFY(std::isfinite(errors[level]) && errors[level] > 0.0,
                  "Oscillator reconstruction error must be finite and positive.");
      if (level > 0)
      {
         MFEM_VERIFY(errors[level] < errors[level - 1],
                     "Oscillator reconstruction errors are not decreasing.");
      }
   }
   double minimum_order = std::numeric_limits<double>::infinity();
   for (int level = 1; level < 3; level++)
   {
      const double order =
         std::log(errors[level] / errors[level + 1]) / std::log(2.0);
      MFEM_VERIFY(std::isfinite(order),
                  "Oscillator reconstruction order is non-finite.");
      minimum_order = std::min(minimum_order, order);
   }
   MFEM_VERIFY(std::isfinite(minimum_order) && minimum_order > 3.8,
               "Cubic Hermite reconstruction did not converge at fourth order.");

   HermiteProviderCheck result;
   result.cubic_max_error = cubic_max_error;
   result.oscillator_finest_error = errors[3];
   result.oscillator_minimum_order = minimum_order;
   return result;
}

void CheckLateIntervalHermiteStageTimes()
{
   constexpr int forward_steps = 9000;
   constexpr int adjoint_refinement = 3;
   constexpr real_t final_time = 6.0;
   const real_t forward_step = final_time / forward_steps;
   const real_t adjoint_step = forward_step / adjoint_refinement;

   ForwardIntervalData interval;
   interval.dt = forward_step;
   interval.x_left.SetSize(2);
   interval.x_right.SetSize(2);
   interval.f_left.SetSize(2);
   interval.f_right.SetSize(2);
   interval.x_left[0] = 1.0;
   interval.x_left[1] = -2.0;
   interval.x_right[0] = 3.0;
   interval.x_right[1] = 4.0;
   interval.f_left = 0.0;
   interval.f_right = 0.0;

   Vector reconstructed;
   for (int step = 0; step < forward_steps; step++)
   {
      interval.index = step;
      interval.t_left = step * forward_step;
      CubicHermiteForwardIntervalProvider provider(interval);

      // Enumerate the exact physical-time expressions used by every reverse
      // RK4 stage. At late intervals, subtraction can put a mathematically
      // exact endpoint a few normalized-coordinate ulps outside [0,1].
      for (int substep = 0;
           substep < adjoint_refinement; substep++)
      {
         const real_t t_right =
            interval.t_left + (substep + 1) * adjoint_step;
         const real_t stage_times[] =
         {
            t_right,
            t_right - 0.5 * adjoint_step,
            t_right - 0.5 * adjoint_step,
            t_right - adjoint_step
         };
         for (real_t stage_time : stage_times)
         {
            provider.Evaluate(stage_time, reconstructed);
            MFEM_VERIFY(
               reconstructed.Size() == 2 &&
               std::isfinite(reconstructed[0]) &&
               std::isfinite(reconstructed[1]),
               "Late-time Hermite stage reconstruction is non-finite.");
         }
      }

      provider.Evaluate(interval.t_left, reconstructed);
      MFEM_VERIFY(reconstructed[0] == interval.x_left[0] &&
                  reconstructed[1] == interval.x_left[1],
                  "Hermite provider did not snap its physical left endpoint.");
      provider.Evaluate(interval.t_left + interval.dt, reconstructed);
      MFEM_VERIFY(reconstructed[0] == interval.x_right[0] &&
                  reconstructed[1] == interval.x_right[1],
                  "Hermite provider did not snap its physical right endpoint.");

      const real_t first_stage_left =
         interval.t_left + adjoint_step - adjoint_step;
      provider.Evaluate(first_stage_left, reconstructed);
      MFEM_VERIFY(reconstructed[0] == interval.x_left[0] &&
                  reconstructed[1] == interval.x_left[1],
                  "Roundoff-shifted reverse stage did not snap left.");
   }

   // The forward-finer REVOLVE path uses a q+1-node provider whose origin is
   // the late physical block time, rather than zero.  Exercise the exact
   // coarse-adjoint RK stages and fine-interval Simpson stages used in
   // production.  This catches cancellation in (t - t_block)/dt_f that grows
   // like epsilon*N_f even though the provider contains only q intervals.
   constexpr int block_forward_steps = 9000;
   constexpr real_t block_final_time = 3.5;
   const real_t block_forward_step =
      block_final_time / block_forward_steps;
   for (int ratio : {2, 3})
   {
      const int adjoint_steps = block_forward_steps / ratio;
      const real_t block_size = ratio * block_forward_step;
      std::vector<Vector> states(ratio + 1), derivatives(ratio + 1);
      for (int node = 0; node <= ratio; node++)
      {
         states[node].SetSize(2);
         derivatives[node].SetSize(2);
         states[node][0] = node;
         states[node][1] = -node;
         derivatives[node] = 0.0;
      }

      for (int block = 0; block < adjoint_steps; block++)
      {
         const real_t block_left = block * block_size;
         CubicHermiteForwardStateProvider block_provider(
            states, derivatives, block_left, block_forward_step);
         const real_t block_right = block_left + block_size;
         const real_t adjoint_stage_times[] =
         {
            block_right,
            block_right - 0.5 * block_size,
            block_right - 0.5 * block_size,
            block_right - block_size
         };
         for (real_t stage_time : adjoint_stage_times)
         {
            block_provider.Evaluate(stage_time, reconstructed);
         }

         for (int local_step = 0; local_step < ratio; local_step++)
         {
            const real_t fine_left =
               block_left + local_step * block_forward_step;
            const real_t design_stage_times[] =
            {
               fine_left,
               fine_left + 0.5 * block_forward_step,
               fine_left + block_forward_step
            };
            for (real_t stage_time : design_stage_times)
            {
               block_provider.Evaluate(stage_time, reconstructed);
               MFEM_VERIFY(
                  reconstructed.Size() == 2 &&
                  std::isfinite(reconstructed[0]) &&
                  std::isfinite(reconstructed[1]),
                  "Late-time block-local Hermite stage is non-finite.");
            }
         }
      }
   }
}

double CheckInstantaneousObjectiveGradient(
   ParFiniteElementSpace &state_fes,
   const Array<int> &offsets,
   TimeIntegratedObjective &objective,
   MPI_Comm comm,
   int state_size,
   int seed,
   real_t physical_time)
{
   Vector state(state_size), direction(state_size);
   Vector plus(state_size), minus(state_size), gradient;
   RandomState(state, seed);
   RandomState(direction, seed + 1);
   Normalize(comm, direction);

   const real_t epsilon = 1e-6;
   plus = state;
   minus = state;
   plus.Add(epsilon, direction);
   minus.Add(-epsilon, direction);

   const auto evaluate = [&](Vector &candidate)
   {
      BlockVector block(candidate, offsets);
      ParGridFunction displacement(&state_fes);
      displacement.SetFromTrueDofs(block.GetBlock(0));
      return objective.EvaluateInstantaneous(displacement, physical_time);
   };

   const real_t value_plus = evaluate(plus);
   const real_t value_minus = evaluate(minus);
   const real_t finite_difference =
      (value_plus - value_minus) / (2.0 * epsilon);

   InstantaneousObjectiveGradientAtState(
      state_fes, offsets, objective, state, physical_time, gradient);
   const real_t projected_gradient = GlobalDot(comm, gradient, direction);
   const double scale =
      std::max({std::abs(static_cast<double>(finite_difference)),
                std::abs(static_cast<double>(projected_gradient)), 1e-30});
   const double relative_error =
      std::abs(static_cast<double>(finite_difference - projected_gradient))
      / scale;

   MFEM_VERIFY(relative_error < 1e-7,
               "Instantaneous objective-gradient finite-difference check "
               "failed.");
   return relative_error;
}

real_t EvaluateSimpsonObjective(
   ParFiniteElementSpace &state_fes,
   const Array<int> &offsets,
   TimeIntegratedObjective &objective,
   const std::vector<Vector> &states,
   real_t time_step)
{
   const int intervals = static_cast<int>(states.size()) - 1;
   MFEM_VERIFY(intervals > 0 && intervals % 2 == 0,
               "Composite Simpson evaluation requires an even interval count.");

   real_t weighted_sum = 0.0;
   for (int node = 0; node <= intervals; node++)
   {
      BlockVector state_block(const_cast<Vector&>(states[node]), offsets);
      ParGridFunction displacement(&state_fes);
      displacement.SetFromTrueDofs(state_block.GetBlock(0));
      const real_t weight =
         (node == 0 || node == intervals) ? 1.0 :
         ((node % 2 == 0) ? 2.0 : 4.0);
      weighted_sum +=
         weight * objective.EvaluateInstantaneous(
                     displacement, node * time_step);
   }
   return (time_step / 3.0) * weighted_sum;
}

double CheckContinuousAdjointDirectionalDerivative(
   ElastodynamicsOperator &oper,
   ParFiniteElementSpace &state_fes,
   const Array<int> &offsets,
   TimeIntegratedObjective &objective,
   MPI_Comm comm,
   int state_size,
   int requested_adjoint_steps,
   real_t requested_adjoint_step)
{
   const int adjoint_steps = std::max(1, requested_adjoint_steps);
   const int forward_steps = 2 * adjoint_steps;
   const real_t adjoint_step =
      std::min(requested_adjoint_step, real_t(5e-5));
   const real_t forward_step = 0.5 * adjoint_step;
   const real_t final_time = adjoint_steps * adjoint_step;

   Vector initial_state(state_size), direction(state_size);
   RandomState(initial_state, 640);
   RandomState(direction, 641);
   Normalize(comm, direction);

   std::vector<Vector> base_states;
   std::vector<real_t> base_times;
   RolloutObjective(
      oper, state_fes, offsets, objective, initial_state,
      forward_steps, 0.0, forward_step, &base_states, &base_times);

   ExactStoredForwardStateProvider provider(
      base_states, 0.0, forward_step);
   const NestedTimeGrid grid =
      NestedTimeGrid::Create(final_time, forward_steps, adjoint_steps);
   Vector terminal_adjoint(state_size), initial_adjoint;
   terminal_adjoint = 0.0;
   ContinuousAdjointSweepFullStorage(
      oper, state_fes, objective, provider, grid,
      terminal_adjoint, initial_adjoint);
   const real_t projected_adjoint =
      GlobalDot(comm, initial_adjoint, direction);

   const real_t epsilon = 1e-5;
   Vector plus(initial_state), minus(initial_state);
   plus.Add(epsilon, direction);
   minus.Add(-epsilon, direction);
   std::vector<Vector> plus_states, minus_states;
   std::vector<real_t> times;
   RolloutObjective(
      oper, state_fes, offsets, objective, plus,
      forward_steps, 0.0, forward_step, &plus_states, &times);
   RolloutObjective(
      oper, state_fes, offsets, objective, minus,
      forward_steps, 0.0, forward_step, &minus_states, &times);

   const real_t objective_plus = EvaluateSimpsonObjective(
      state_fes, offsets, objective, plus_states, forward_step);
   const real_t objective_minus = EvaluateSimpsonObjective(
      state_fes, offsets, objective, minus_states, forward_step);
   const real_t finite_difference =
      (objective_plus - objective_minus) / (2.0 * epsilon);

   const double scale =
      std::max({std::abs(static_cast<double>(finite_difference)),
                std::abs(static_cast<double>(projected_adjoint)), 1e-30});
   const double relative_error =
      std::abs(static_cast<double>(finite_difference - projected_adjoint))
      / scale;
   MFEM_VERIFY(relative_error < 1e-5,
               "Continuous-adjoint directional derivative check failed.");
   return relative_error;
}

struct ContinuousDesignReplayCheck
{
   double objective_relative_error;
   double initial_adjoint_relative_error;
   double filtered_gradient_relative_error;
   double raw_gradient_relative_error;
   double replayed_endpoint_relative_error;
   double filtered_gradient_fd_relative_error;
   double raw_gradient_fd_relative_error;
};

ContinuousDesignReplayCheck CheckContinuousDesignReplayEquivalence(
   ElastodynamicsOperator &oper,
   ParFiniteElementSpace &state_fes,
   ParFiniteElementSpace &filter_fes,
   ParFiniteElementSpace &control_fes,
   ParGridFunction &rho,
   ParGridFunction &rho_tilde,
   toopt::PDEFilter &filter,
   Coefficient &gamma_coef,
   Array<int> &exterior_bdr_attr,
   Array<int> &empty_bdr_attr,
   const MaterialParams &material,
   const BoundaryLoadSpec &load_spec,
   VectorCoefficient &load_coef,
   real_t impedance,
   TimeIntegratedObjective &objective,
   MPI_Comm comm,
   int requested_forward_steps,
   real_t requested_forward_step)
{
   const int forward_steps =
      std::max(3, std::min(requested_forward_steps, 5));
   const real_t forward_step =
      std::min(requested_forward_step, real_t(5e-5));
   // An odd ratio exercises Hermite requests at non-quarter fractions and
   // proves REVOLVE is scheduling coarse intervals rather than fine steps.
   const int adjoint_refinement = 3;

   Vector initial_state(oper.Width());
   RandomState(initial_state, 680);
   initial_state *= 10.0;

   ContinuousDesignRunResult full =
      RunContinuousDesignFullStorage(
         oper, state_fes, filter_fes, rho_tilde, material, objective,
         initial_state, forward_steps, forward_step, adjoint_refinement);

   std::vector<Vector> expected_forward_states(forward_steps + 1);
   expected_forward_states[0] = initial_state;
   Vector expected_state(initial_state);
   RK4Solver expected_solver;
   expected_solver.Init(oper);
   real_t expected_time = 0.0;
   for (int step = 0; step < forward_steps; step++)
   {
      real_t accepted_step = forward_step;
      expected_solver.Step(expected_state, expected_time, accepted_step);
      expected_time = (step + 1) * forward_step;
      expected_forward_states[step + 1] = expected_state;
   }
   int audited_replay_intervals = 0;
   double replayed_endpoint_relative_error = 0.0;
   const auto replay_audit =
      [&](int step,
          const Vector &revolve_state_left,
          const ForwardIntervalData &interval)
      {
         MFEM_VERIFY(
            step == forward_steps - 1 - audited_replay_intervals,
            "Continuous replay audit observed an out-of-order interval.");
         const auto record_error =
            [&](const Vector &computed, const Vector &expected)
            {
               Vector difference(computed);
               difference -= expected;
               const real_t denominator =
                  std::max(GlobalVectorNorm(comm, expected), real_t(1e-30));
               replayed_endpoint_relative_error =
                  std::max(
                     replayed_endpoint_relative_error,
                     static_cast<double>(
                        GlobalVectorNorm(comm, difference) / denominator));
            };
         record_error(
            revolve_state_left, expected_forward_states[step]);
         record_error(interval.x_left, expected_forward_states[step]);
         record_error(interval.x_right, expected_forward_states[step + 1]);
         audited_replay_intervals++;
      };
   ContinuousDesignRunResult replay_one =
      RunContinuousDesignCheckpointed(
         oper, state_fes, filter_fes, rho_tilde, material, objective,
         initial_state, forward_steps, forward_step, adjoint_refinement,
         /*num_checkpoints=*/1, replay_audit);
   ContinuousDesignRunResult replay_two =
      RunContinuousDesignCheckpointed(
         oper, state_fes, filter_fes, rho_tilde, material, objective,
         initial_state, forward_steps, forward_step, adjoint_refinement,
         /*num_checkpoints=*/2);
   ContinuousDesignRunResult full_same_grid =
      RunContinuousDesignFullStorage(
         oper, state_fes, filter_fes, rho_tilde, material, objective,
         initial_state, forward_steps, forward_step,
         /*adjoint_refinement=*/1);
   ContinuousDesignRunResult replay_same_grid =
      RunContinuousDesignCheckpointed(
         oper, state_fes, filter_fes, rho_tilde, material, objective,
         initial_state, forward_steps, forward_step,
         /*adjoint_refinement=*/1, /*num_checkpoints=*/1);

   const auto relative_scalar_error = [](real_t left, real_t right)
   {
      const real_t scale =
         std::max({std::abs(left), std::abs(right), real_t(1e-30)});
      return static_cast<double>(std::abs(left - right) / scale);
   };
   const auto relative_vector_error =
      [&](const Vector &left, const Vector &right)
      {
         Vector difference(left);
         difference -= right;
         const real_t denominator =
            std::max(GlobalVectorNorm(comm, right), real_t(1e-30));
         return static_cast<double>(
            GlobalVectorNorm(comm, difference) / denominator);
      };

   Vector raw_full(control_fes.GetTrueVSize());
   Vector raw_one(control_fes.GetTrueVSize());
   Vector raw_two(control_fes.GetTrueVSize());
   Vector raw_full_same(control_fes.GetTrueVSize());
   Vector raw_replay_same(control_fes.GetTrueVSize());
   filter.MultTranspose(full.gradient_tilde, raw_full);
   filter.MultTranspose(replay_one.gradient_tilde, raw_one);
   filter.MultTranspose(replay_two.gradient_tilde, raw_two);
   filter.MultTranspose(full_same_grid.gradient_tilde, raw_full_same);
   filter.MultTranspose(replay_same_grid.gradient_tilde, raw_replay_same);

   // Rebuild the spatial operator after every density perturbation.  The
   // objective below is exactly the fourth-order Simpson/Hermite objective used
   // by RunContinuousDesignFullStorage, but needs only the cheap forward sweep.
   const auto evaluate_current_filtered_design = [&]()
   {
      ConstantCoefficient rho_0_coef(material.rho0);
      ConstantCoefficient lambda_0_coef(material.lambda0);
      ConstantCoefficient mu_0_coef(material.mu0);
      SIMPCoefficient simp_mass(
         &rho_tilde, material.r_min, material.r_max, material.simp_p);
      SIMPCoefficient simp_stiff(
         &rho_tilde, material.r_min, material.r_max, material.simp_p);
      ProductCoefficient mass_coef(simp_mass, rho_0_coef);
      ProductCoefficient lambda_coef(simp_stiff, lambda_0_coef);
      ProductCoefficient mu_coef(simp_stiff, mu_0_coef);
      ElastodynamicsOperator perturbed_oper(
         state_fes, mass_coef, lambda_coef, mu_coef,
         load_spec.amplitude, load_spec.duration, load_spec.time_profile,
         load_spec.phase, load_spec.frequency, load_spec.bdr_attributes,
         load_coef, load_spec.domain_load, &gamma_coef, impedance,
         exterior_bdr_attr, empty_bdr_attr, MassSolverType::LUMPED,
         /*print_banner=*/false);

      std::vector<Vector> states;
      return ContinuousForwardSweepFullStorage(
         perturbed_oper, state_fes, objective, initial_state,
         forward_steps, /*start_time=*/0.0, forward_step,
         adjoint_refinement, states);
   };

   Vector filtered_base;
   Vector raw_base;
   rho_tilde.GetTrueDofs(filtered_base);
   rho.GetTrueDofs(raw_base);

   Vector filtered_direction(full.gradient_tilde);
   Vector raw_direction(raw_full);
   Normalize(comm, filtered_direction);
   Normalize(comm, raw_direction);
   const real_t projected_filtered =
      GlobalDot(comm, full.gradient_tilde, filtered_direction);
   const real_t projected_raw =
      GlobalDot(comm, raw_full, raw_direction);

   const auto directional_relative_error =
      [](real_t finite_difference, real_t projected_gradient)
      {
         const real_t scale =
            std::max({std::abs(finite_difference),
                      std::abs(projected_gradient), real_t(1e-30)});
         return static_cast<double>(
            std::abs(finite_difference - projected_gradient) / scale);
      };

   double best_filtered_fd_error =
      std::numeric_limits<double>::infinity();
   double best_raw_fd_error =
      std::numeric_limits<double>::infinity();
   real_t perturbation = 0.1;
   for (int scale_index = 0; scale_index < 4; scale_index++)
   {
      Vector plus(filtered_base);
      Vector minus(filtered_base);
      plus.Add(perturbation, filtered_direction);
      minus.Add(-perturbation, filtered_direction);
      rho_tilde.SetFromTrueDofs(plus);
      const real_t filtered_plus = evaluate_current_filtered_design();
      rho_tilde.SetFromTrueDofs(minus);
      const real_t filtered_minus = evaluate_current_filtered_design();
      const real_t filtered_fd =
         (filtered_plus - filtered_minus) / (2.0 * perturbation);
      best_filtered_fd_error =
         std::min(best_filtered_fd_error,
                  directional_relative_error(
                     filtered_fd, projected_filtered));

      plus = raw_base;
      minus = raw_base;
      plus.Add(perturbation, raw_direction);
      minus.Add(-perturbation, raw_direction);
      rho.SetFromTrueDofs(plus);
      filter.Mult(rho, rho_tilde);
      const real_t raw_plus = evaluate_current_filtered_design();
      rho.SetFromTrueDofs(minus);
      filter.Mult(rho, rho_tilde);
      const real_t raw_minus = evaluate_current_filtered_design();
      const real_t raw_fd =
         (raw_plus - raw_minus) / (2.0 * perturbation);
      best_raw_fd_error =
         std::min(best_raw_fd_error,
                  directional_relative_error(raw_fd, projected_raw));

      perturbation *= 0.25;
   }

   // Leave the shared coefficient fields at the unperturbed design for all
   // verification checks that follow this one.
   rho.SetFromTrueDofs(raw_base);
   rho_tilde.SetFromTrueDofs(filtered_base);

   ContinuousDesignReplayCheck result;
   result.objective_relative_error =
      std::max(
         std::max(
            relative_scalar_error(full.objective, replay_one.objective),
            relative_scalar_error(full.objective, replay_two.objective)),
         relative_scalar_error(
            full_same_grid.objective, replay_same_grid.objective));
   result.initial_adjoint_relative_error =
      std::max(
         std::max(
            relative_vector_error(
               replay_one.initial_adjoint, full.initial_adjoint),
            relative_vector_error(
               replay_two.initial_adjoint, full.initial_adjoint)),
         relative_vector_error(
            replay_same_grid.initial_adjoint,
            full_same_grid.initial_adjoint));
   result.filtered_gradient_relative_error =
      std::max(
         std::max(
            relative_vector_error(
               replay_one.gradient_tilde, full.gradient_tilde),
            relative_vector_error(
               replay_two.gradient_tilde, full.gradient_tilde)),
         relative_vector_error(
            replay_same_grid.gradient_tilde,
            full_same_grid.gradient_tilde));
   result.raw_gradient_relative_error =
      std::max(
         std::max(
            relative_vector_error(raw_one, raw_full),
            relative_vector_error(raw_two, raw_full)),
         relative_vector_error(raw_replay_same, raw_full_same));
   result.replayed_endpoint_relative_error =
      replayed_endpoint_relative_error;
   result.filtered_gradient_fd_relative_error = best_filtered_fd_error;
   result.raw_gradient_fd_relative_error = best_raw_fd_error;

   MFEM_VERIFY(result.objective_relative_error < 1e-13,
               "Full/replayed continuous objectives disagree.");
   MFEM_VERIFY(result.initial_adjoint_relative_error < 1e-12,
               "Full/replayed continuous initial adjoints disagree.");
   MFEM_VERIFY(result.filtered_gradient_relative_error < 1e-12,
               "Full/replayed filtered gradients disagree.");
   MFEM_VERIFY(result.raw_gradient_relative_error < 1e-12,
               "Full/replayed raw gradients disagree.");
   MFEM_VERIFY(
      audited_replay_intervals == forward_steps &&
      result.replayed_endpoint_relative_error < 1e-13,
      "Privately replayed coarse endpoints differ from the accepted "
      "forward trajectory.");
   MFEM_VERIFY(result.filtered_gradient_fd_relative_error < 5e-4,
               "Continuous filtered gradient failed its finite-difference "
               "check against the fourth-order objective.");
   MFEM_VERIFY(result.raw_gradient_fd_relative_error < 5e-4,
               "Continuous raw gradient failed its finite-difference check "
               "against the fourth-order objective.");
   MFEM_VERIFY(
      replay_one.locally_replayed_intervals == forward_steps &&
      replay_two.locally_replayed_intervals == forward_steps &&
      replay_same_grid.locally_replayed_intervals == forward_steps,
      "Continuous REVOLVE did not consume exactly N_f intervals.");
   MFEM_VERIFY(
      replay_same_grid.controller_recomputed_intervals ==
      replay_one.controller_recomputed_intervals,
      "REVOLVE controller replay count must depend on N_f and C, not on "
      "the adjoint refinement.");
   return result;
}

// DirectionalBoundaryLoadCoefficient: shared from BoundaryLoadSpec.hpp
// (pulled in via ElastodynamicsSolver.hpp).

// EvalRHS / EvalJacobianTranspose: promoted to ElastodynamicsSolver.hpp

void RK4OneStep(ElastodynamicsOperator &oper,
                const Vector &x0, real_t t0, real_t h, Vector &x1)
{
   x1 = x0;
   real_t t = t0;
   real_t dt = h;
   RK4Solver solver;
   solver.Init(oper);
   solver.Step(x1, t, dt);
}

struct ForwardReplayIntegrityCheck
{
   double endpoint_relative_error;
   double left_slope_relative_error;
   double right_slope_relative_error;
};

ForwardReplayIntegrityCheck CheckForwardIntervalReplayIntegrity(
   ElastodynamicsOperator &oper, MPI_Comm comm)
{
   constexpr real_t t_left = 0.137;
   constexpr real_t step_size = 5e-5;
   const real_t poison = std::numeric_limits<real_t>::quiet_NaN();

   Vector state_left(oper.Width());
   RandomState(state_left, 710);
   state_left *= 10.0;
   const Vector saved_left(state_left);

   ForwardIntervalReplayWorkspace workspace;
   Vector *workspace_vectors[] =
   {
      &workspace.k2, &workspace.k3, &workspace.k4,
      &workspace.y1, &workspace.y2, &workspace.y3
   };
   for (Vector *entry : workspace_vectors)
   {
      entry->SetSize(oper.Width());
      *entry = poison;
   }

   ForwardIntervalData interval;
   interval.index = -777;
   interval.t_left = poison;
   interval.dt = poison;
   Vector *interval_vectors[] =
   {
      &interval.x_left, &interval.x_right,
      &interval.f_left, &interval.f_right
   };
   for (Vector *entry : interval_vectors)
   {
      entry->SetSize(oper.Width());
      *entry = poison;
   }

   ReplayForwardInterval(
      oper, /*interval_index=*/7, t_left, step_size,
      state_left, workspace, interval);

   Vector accepted_endpoint;
   RK4OneStep(oper, saved_left, t_left, step_size, accepted_endpoint);
   Vector expected_left_slope, expected_right_slope;
   EvalRHS(oper, saved_left, t_left, expected_left_slope);
   EvalRHS(
      oper, accepted_endpoint, t_left + step_size, expected_right_slope);

   const auto relative_vector_error =
      [&](const Vector &computed, const Vector &expected)
      {
         Vector difference(computed);
         difference -= expected;
         return static_cast<double>(
            GlobalVectorNorm(comm, difference) /
            std::max(GlobalVectorNorm(comm, expected), real_t(1e-30)));
      };

   const double input_error = relative_vector_error(state_left, saved_left);
   ForwardReplayIntegrityCheck result;
   result.endpoint_relative_error =
      relative_vector_error(interval.x_right, accepted_endpoint);
   result.left_slope_relative_error =
      relative_vector_error(interval.f_left, expected_left_slope);
   result.right_slope_relative_error =
      relative_vector_error(interval.f_right, expected_right_slope);

   Vector endpoint_stage_difference(interval.f_right);
   endpoint_stage_difference -= workspace.k4;
   const real_t endpoint_stage_difference_norm =
      GlobalVectorNorm(comm, endpoint_stage_difference);
   const real_t endpoint_slope_scale =
      std::max(GlobalVectorNorm(comm, interval.f_right), real_t(1e-30));

   MFEM_VERIFY(input_error < 1e-15,
               "Forward interval replay mutated REVOLVE's left state.");
   MFEM_VERIFY(result.endpoint_relative_error < 1e-13,
               "Replayed RK4 endpoint differs from an accepted RK4 step.");
   MFEM_VERIFY(result.left_slope_relative_error < 1e-13 &&
               result.right_slope_relative_error < 1e-13,
               "Replayed physical endpoint slope is incorrect.");
   MFEM_VERIFY(
      endpoint_stage_difference_norm >
      100.0 * std::numeric_limits<real_t>::epsilon() *
      endpoint_slope_scale,
      "Replay-integrity fixture cannot distinguish physical f_right from k4.");
   return result;
}

double CheckContinuousTerminalFunctionalSign(
   ElastodynamicsOperator &oper,
   ParFiniteElementSpace &state_fes,
   MPI_Comm comm)
{
   constexpr int forward_steps = 4;
   constexpr int adjoint_refinement = 4;
   constexpr real_t forward_step = 1e-6;
   constexpr real_t perturbation = 1e-4;
   const real_t final_time = forward_steps * forward_step;

   Vector initial_state(oper.Width()), direction(oper.Width());
   RandomState(initial_state, 720);
   RandomState(direction, 721);
   Normalize(comm, direction);

   Vector plus(initial_state), minus(initial_state);
   plus.Add(perturbation, direction);
   minus.Add(-perturbation, direction);
   Vector plus_final, minus_final;
   const auto rollout_final =
      [&](const Vector &initial, Vector &final_state)
      {
         final_state = initial;
         real_t time = 0.0;
         RK4Solver solver;
         solver.Init(oper);
         for (int step = 0; step < forward_steps; step++)
         {
            real_t accepted_step = forward_step;
            solver.Step(final_state, time, accepted_step);
            time = (step + 1) * forward_step;
         }
      };
   rollout_final(plus, plus_final);
   rollout_final(minus, minus_final);

   Vector terminal_direction(plus_final);
   terminal_direction -= minus_final;
   terminal_direction /= (2.0 * perturbation);
   Normalize(comm, terminal_direction);
   Vector final_difference(plus_final);
   final_difference -= minus_final;
   const real_t terminal_finite_difference =
      GlobalDot(comm, terminal_direction, final_difference) /
      (2.0 * perturbation);
   MFEM_VERIFY(terminal_finite_difference > 0.0,
               "Manufactured terminal derivative must be positive.");

   ZeroInstantaneousObjective zero_objective(&state_fes, comm);
   std::vector<Vector> forward_states;
   ContinuousForwardSweepFullStorage(
      oper, state_fes, zero_objective, initial_state,
      forward_steps, /*start_time=*/0.0, forward_step,
      adjoint_refinement, forward_states);
   std::vector<Vector> forward_derivatives;
   BuildForwardStateTimeDerivatives(
      oper, forward_states, /*start_time=*/0.0,
      forward_step, forward_derivatives);
   CubicHermiteForwardStateProvider hermite_provider(
      forward_states, forward_derivatives,
      /*start_time=*/0.0, forward_step);
   RecordingForwardStateProvider recording_provider(hermite_provider);

   const NestedTimeGrid grid = NestedTimeGrid::Create(
      final_time, forward_steps, forward_steps * adjoint_refinement);
   Vector initial_adjoint;
   ContinuousAdjointSweepFullStorage(
      oper, state_fes, zero_objective, recording_provider, grid,
      terminal_direction, initial_adjoint);
   const real_t projected_initial_adjoint =
      GlobalDot(comm, initial_adjoint, direction);

   MFEM_VERIFY(
      recording_provider.requested_times.size() ==
      static_cast<std::size_t>(4 * grid.adjoint_steps),
      "Continuous terminal test observed the wrong number of RK4 stages.");
   std::size_t request = 0;
   for (int step = grid.adjoint_steps - 1; step >= 0; step--)
   {
      const real_t t_right = (step + 1) * grid.dt_adjoint;
      const real_t expected_times[] =
      {
         t_right,
         t_right - 0.5 * grid.dt_adjoint,
         t_right - 0.5 * grid.dt_adjoint,
         t_right - grid.dt_adjoint
      };
      for (real_t expected_time : expected_times)
      {
         const real_t observed_time =
            recording_provider.requested_times[request++];
         const real_t scale =
            std::max({real_t(1.0), std::abs(expected_time),
                      std::abs(observed_time)});
         MFEM_VERIFY(
            std::abs(observed_time - expected_time) <=
            64.0 * std::numeric_limits<real_t>::epsilon() * scale &&
            observed_time >= -64.0 *
               std::numeric_limits<real_t>::epsilon() &&
            observed_time <= final_time + 64.0 *
               std::numeric_limits<real_t>::epsilon(),
            "Continuous terminal test used an incorrect physical stage time.");
      }
   }

   const real_t derivative_scale =
      std::max({std::abs(terminal_finite_difference),
                std::abs(projected_initial_adjoint), real_t(1e-30)});
   const double relative_error =
      static_cast<double>(
         std::abs(projected_initial_adjoint -
                  terminal_finite_difference) / derivative_scale);
   MFEM_VERIFY(
      projected_initial_adjoint > 0.0 && relative_error < 1e-7,
      "Manufactured terminal-functional continuous adjoint has the wrong "
      "sign or magnitude.");
   return relative_error;
}

// RK4Stages / RK4AdjointOneStep: promoted to ElastodynamicsSolver.hpp

void RolloutRK4(ElastodynamicsOperator &oper,
                const Vector &x_init, int nsteps,
                real_t t_init, real_t h,
                Vector &x_final,
                vector<Vector> *states,
                vector<real_t> *times)
{
   const int n = x_init.Size();
   x_final = x_init;
   real_t t = t_init;

   RK4Solver solver;
   solver.Init(oper);

   if (states)
   {
      states->resize(nsteps + 1);
      for (int i = 0; i <= nsteps; i++) { (*states)[i].SetSize(n); }
      (*states)[0] = x_final;
   }
   if (times)
   {
      times->assign(nsteps + 1, 0.0);
      (*times)[0] = t;
   }

   for (int i = 0; i < nsteps; i++)
   {
      real_t dt = h;
      solver.Step(x_final, t, dt);

      if (states) { (*states)[i + 1] = x_final; }
      if (times)  { (*times)[i + 1] = t; }
   }
}

struct ConsistentMassConstraintCheck
{
   double free_residual_relative_error = 0.0;
   double inverse_symmetry_relative_error = 0.0;
   double constrained_value_maximum = 0.0;
   double wave_time_step_limit = 0.0;
   double damping_time_step_limit = 0.0;
};

ConsistentMassConstraintCheck CheckConstrainedConsistentMassInverse(
   ElastodynamicsOperator &oper, MPI_Comm comm)
{
   MFEM_VERIFY(oper.GetMassSolverType() == MassSolverType::ITERATIVE,
               "Constrained consistent-mass check needs the iterative mass.");
   const Array<int> &essential = oper.GetEssentialTrueDofs();
   MFEM_VERIFY(essential.Size() > 0,
               "Constrained consistent-mass check needs essential dofs.");

   const int n = oper.GetMassMatrix()->Height();
   Vector rhs(n), projected_rhs(n), acceleration(n), mass_acceleration(n);
   RandomState(rhs, 875);
   projected_rhs = rhs;
   oper.ProjectEssentialField(projected_rhs);
   oper.MultInvMass(rhs, acceleration);

   oper.GetMassMatrix()->Mult(acceleration, mass_acceleration);
   oper.ProjectEssentialField(mass_acceleration);
   mass_acceleration -= projected_rhs;
   const double residual_norm =
      std::sqrt(GlobalDot(comm, mass_acceleration, mass_acceleration));
   const double rhs_norm =
      std::sqrt(GlobalDot(comm, projected_rhs, projected_rhs));

   double local_constrained_maximum = 0.0;
   for (int i = 0; i < essential.Size(); i++)
   {
      local_constrained_maximum =
         std::max(local_constrained_maximum,
                  std::abs(static_cast<double>(acceleration[essential[i]])));
   }
   double global_constrained_maximum = 0.0;
   MPI_Allreduce(&local_constrained_maximum, &global_constrained_maximum,
                 1, MPI_DOUBLE, MPI_MAX, comm);

   Vector x(n), y(n), inverse_x(n), inverse_y(n);
   RandomState(x, 876);
   RandomState(y, 877);
   oper.MultInvMass(x, inverse_x);
   oper.MultInvMass(y, inverse_y);
   const double lhs = GlobalDot(comm, x, inverse_y);
   const double rhs_transpose = GlobalDot(comm, inverse_x, y);
   const double symmetry_scale =
      std::max({std::abs(lhs), std::abs(rhs_transpose), 1e-30});

   ConsistentMassConstraintCheck check;
   check.free_residual_relative_error =
      residual_norm / std::max(rhs_norm, 1e-30);
   check.inverse_symmetry_relative_error =
      std::abs(lhs - rhs_transpose) / symmetry_scale;
   check.constrained_value_maximum = global_constrained_maximum;
   check.wave_time_step_limit = oper.EstimateConsistentRK4TimeStep(12);
   check.damping_time_step_limit =
      oper.EstimateConsistentRK4DampingTimeStep(12);

   MFEM_VERIFY(check.free_residual_relative_error < 1e-9,
               "Constrained consistent-mass solve has a large free residual.");
   MFEM_VERIFY(check.inverse_symmetry_relative_error < 1e-9,
               "Constrained consistent-mass inverse is not symmetric.");
   MFEM_VERIFY(check.constrained_value_maximum == 0.0,
               "Constrained consistent-mass solve returned a nonzero "
               "essential value.");
   MFEM_VERIFY(check.wave_time_step_limit > 0.0 &&
               std::isfinite(check.wave_time_step_limit),
               "Consistent-mass wave CFL estimate is invalid.");
   MFEM_VERIFY(check.damping_time_step_limit > 0.0,
               "Consistent-mass damping CFL estimate is invalid.");

   // Exercise the compatibility validator too: it must no longer silently
   // skip ITERATIVE mass. A conservative fraction avoids sensitivity to the
   // shorter power iteration used for the metrics above.
   const double estimated_limit =
      std::min(check.wave_time_step_limit,
               check.damping_time_step_limit);
   ValidateLumpedRK4TimeStep(
      oper, 0.1 * estimated_limit, /*print_report=*/false);
   return check;
}

double CheckJacobianTranspose(ElastodynamicsOperator &oper,
                              MPI_Comm comm, int n,
                              int ntrials, real_t eps,
                              real_t tolerance)
{
   double worst = 0.0;
   const real_t t0 = 0.0137;

   Vector x(n), v(n), w(n), xp(n), xm(n), fp(n), fm(n), jv(n), jtw(n);

   for (int trial = 0; trial < ntrials; trial++)
   {
      RandomState(x, 100 + 3*trial);
      RandomState(v, 101 + 3*trial);
      RandomState(w, 102 + 3*trial);

      xp = x;
      xm = x;
      xp.Add(eps, v);
      xm.Add(-eps, v);

      EvalRHS(oper, xp, t0, fp);
      EvalRHS(oper, xm, t0, fm);

      jv = fp;
      jv -= fm;
      jv *= 0.5 / eps;

      EvalJacobianTranspose(oper, x, t0, w, jtw);

      const double lhs = GlobalDot(comm, jv, w);
      const double rhs = GlobalDot(comm, v, jtw);
      const double rel = RelativeError(lhs, rhs);
      worst = max(worst, rel);

      if (Mpi::Root())
      {
         mfem::out << "Jacobian transpose trial " << trial
                   << ": lhs=" << setprecision(16) << lhs
                   << ", rhs=" << rhs
                   << ", rel_err=" << rel << '\n';
      }
   }

   MFEM_VERIFY(worst < tolerance, "Jacobian transpose verification failed.");
   return worst;
}

double CheckRK4OneStepTranspose(ElastodynamicsOperator &oper,
                                MPI_Comm comm, int n,
                                int ntrials, real_t h, real_t eps,
                                real_t tolerance)
{
   double worst = 0.0;
   const real_t t0 = 0.0137;

   Vector x0(n), v(n), w(n), xp(n), xm(n), x_plus(n), x_minus(n);
   Vector jv(n), lambda_prev(n);

   for (int trial = 0; trial < ntrials; trial++)
   {
      RandomState(x0, 200 + 3*trial);
      RandomState(v, 201 + 3*trial);
      RandomState(w, 202 + 3*trial);

      xp = x0;
      xm = x0;
      xp.Add(eps, v);
      xm.Add(-eps, v);

      RK4OneStep(oper, xp, t0, h, x_plus);
      RK4OneStep(oper, xm, t0, h, x_minus);

      jv = x_plus;
      jv -= x_minus;
      jv *= 0.5 / eps;

      RK4AdjointOneStep(oper, x0, t0, h, w, lambda_prev);

      const double lhs = GlobalDot(comm, jv, w);
      const double rhs = GlobalDot(comm, v, lambda_prev);
      const double rel = RelativeError(lhs, rhs);
      worst = max(worst, rel);

      if (Mpi::Root())
      {
         mfem::out << "RK4 one-step transpose trial " << trial
                   << ": lhs=" << setprecision(16) << lhs
                   << ", rhs=" << rhs
                   << ", rel_err=" << rel << '\n';
      }
   }

   MFEM_VERIFY(worst < tolerance, "RK4 one-step transpose verification failed.");
   return worst;
}

double CheckRK4NStepTranspose(ElastodynamicsOperator &oper,
                              MPI_Comm comm, int n,
                              int nsteps, int ntrials,
                              real_t h, real_t eps,
                              real_t tolerance)
{
   double worst = 0.0;
   const real_t t0 = 0.0;

   Vector x0(n), v(n), w(n), xp(n), xm(n), x_plus(n), x_minus(n);
   Vector jv(n), lambda(n), lambda_prev(n);

   for (int trial = 0; trial < ntrials; trial++)
   {
      RandomState(x0, 300 + 3*trial);
      RandomState(v, 301 + 3*trial);
      RandomState(w, 302 + 3*trial);

      vector<Vector> states;
      vector<real_t> times;
      Vector x_base(n);
      RolloutRK4(oper, x0, nsteps, t0, h, x_base, &states, &times);

      xp = x0;
      xm = x0;
      xp.Add(eps, v);
      xm.Add(-eps, v);

      RolloutRK4(oper, xp, nsteps, t0, h, x_plus, nullptr, nullptr);
      RolloutRK4(oper, xm, nsteps, t0, h, x_minus, nullptr, nullptr);

      jv = x_plus;
      jv -= x_minus;
      jv *= 0.5 / eps;

      lambda = w;
      for (int i = nsteps - 1; i >= 0; i--)
      {
         const real_t hi = times[i + 1] - times[i];
         RK4AdjointOneStep(oper, states[i], times[i], hi, lambda, lambda_prev);
         lambda = lambda_prev;
      }

      const double lhs = GlobalDot(comm, jv, w);
      const double rhs = GlobalDot(comm, v, lambda);
      const double rel = RelativeError(lhs, rhs);
      worst = max(worst, rel);

      if (Mpi::Root())
      {
         mfem::out << "RK4 " << nsteps << "-step transpose trial " << trial
                   << ": lhs=" << setprecision(16) << lhs
                   << ", rhs=" << rhs
                   << ", rel_err=" << rel << '\n';
      }
   }

   MFEM_VERIFY(worst < tolerance, "RK4 n-step transpose verification failed.");
   return worst;
}

// AddObjectiveContribution / ObjectiveGradientAtState / RolloutObjective:
// promoted to ElastodynamicsSolver.hpp

real_t ObjectiveAdjointGradient(ElastodynamicsOperator &oper,
                                ParFiniteElementSpace &state_fes,
                                const Array<int> &offsets,
                                TimeIntegratedObjective &objective,
                                const Vector &x0,
                                int nsteps, real_t t_init, real_t h,
                                Vector &gradient)
{
   vector<Vector> states;
   vector<real_t> times;
   const real_t J = RolloutObjective(oper, state_fes, offsets, objective,
                                     x0, nsteps, t_init, h,
                                     &states, &times);

   const int n = x0.Size();
   const int total_steps = nsteps + 1;
   Vector q(n), lambda(n), lambda_prev(n);

   ObjectiveGradientAtStateAndTime(
      state_fes, offsets, objective, states[nsteps], times[nsteps],
      h, nsteps, total_steps, lambda);

   for (int i = nsteps - 1; i >= 0; i--)
   {
      const real_t hi = times[i + 1] - times[i];
      RK4AdjointOneStep(oper, states[i], times[i], hi, lambda, lambda_prev);

      ObjectiveGradientAtStateAndTime(
         state_fes, offsets, objective, states[i], times[i],
         h, i, total_steps, q);
      lambda = lambda_prev;
      lambda += q;
   }

   gradient = lambda;
   return J;
}

double CheckObjectiveTaylor(ElastodynamicsOperator &oper,
                            ParFiniteElementSpace &state_fes,
                            const Array<int> &offsets,
                            TimeIntegratedObjective &objective,
                            MPI_Comm comm, int n,
                            int nsteps, int ntrials,
                            real_t h, int nscales,
                            real_t initial_scale,
                            real_t tolerance)
{
   double worst_best_fd_rel = 0.0;
   const real_t t0 = 0.0;

   Vector x0(n), direction(n), gradient(n), xp(n), xm(n);

   for (int trial = 0; trial < ntrials; trial++)
   {
      RandomState(x0, 400 + 2*trial);
      RandomState(direction, 401 + 2*trial);
      Normalize(comm, direction);

      const real_t J0 = ObjectiveAdjointGradient(oper, state_fes, offsets,
                                                 objective, x0, nsteps,
                                                 t0, h, gradient);
      const real_t projected_grad = GlobalDot(comm, gradient, direction);

      if (Mpi::Root())
      {
         mfem::out << "\nObjective Taylor trial " << trial
                   << ": J0=" << setprecision(16) << J0
                   << ", <grad,p>=" << projected_grad << '\n';
      }

      real_t scale = initial_scale;
      double previous_remainder = -1.0;
      double trial_best_fd_rel = numeric_limits<double>::infinity();
      for (int s = 0; s < nscales; s++)
      {
         xp = x0;
         xm = x0;
         xp.Add(scale, direction);
         xm.Add(-scale, direction);

         const real_t Jp = RolloutObjective(oper, state_fes, offsets, objective,
                                            xp, nsteps, t0, h, nullptr, nullptr);
         const real_t Jm = RolloutObjective(oper, state_fes, offsets, objective,
                                            xm, nsteps, t0, h, nullptr, nullptr);

         const real_t fd = (Jp - Jm) / (2.0 * scale);
         const double derivative_scale =
            max(max(fabs(static_cast<double>(fd)),
                    fabs(static_cast<double>(projected_grad))), 1e-30);
         const double fd_rel = fabs(static_cast<double>(fd - projected_grad))
                               / derivative_scale;
         trial_best_fd_rel = min(trial_best_fd_rel, fd_rel);

         const real_t first_order_remainder =
            fabs(Jp - J0 - scale * projected_grad);
         const double remainder_ratio =
            (previous_remainder > 0.0) ?
            previous_remainder / first_order_remainder : 0.0;

         if (Mpi::Root())
         {
            mfem::out << "  scale=" << scientific << setprecision(3) << scale
                      << "  FD=" << setprecision(12) << fd
                      << "  rel_err=" << fd_rel
                      << "  first_order_rem=" << first_order_remainder;
            if (previous_remainder > 0.0)
            {
               mfem::out << "  rem_ratio=" << remainder_ratio;
            }
            mfem::out << '\n';
         }

         previous_remainder = first_order_remainder;
         scale *= 0.1;
      }

      worst_best_fd_rel = max(worst_best_fd_rel, trial_best_fd_rel);
   }

   MFEM_VERIFY(worst_best_fd_rel < tolerance,
               "Objective adjoint finite-difference Taylor check failed.");
   return worst_best_fd_rel;
}

// MaterialParams / SimpDerivative / StageMassDesignLFIntegrator /
// StageStiffnessDesignLFIntegrator: promoted to ElastodynamicsSolver.hpp

double CheckDesignTaylor(ParFiniteElementSpace &state_fes,
                         ParFiniteElementSpace &filter_fes,
                         ParFiniteElementSpace &control_fes,
                         ParGridFunction &rho,
                         ParGridFunction &rho_tilde,
                         toopt::PDEFilter &filter,
                         SpatialDampingCoefficient &gamma_coef,
                         Array<int> &exterior_bdr_attr,
                         Array<int> &empty_bdr_attr,
                         TimeIntegratedObjective &objective,
                         const MaterialParams &mat,
                         const BoundaryLoadSpec &load_spec,
                         VectorCoefficient &load_coef,
                         real_t impedance,
                         int nsteps,
                         real_t h,
                         int ntrials,
                         int nscales,
                         real_t initial_scale,
                         real_t state_scale,
                         real_t tolerance,
                         MassSolverType mass_type)
{
   MPI_Comm comm = state_fes.GetComm();
   double worst_best_fd_rel = 0.0;

   const char *mass_label =
      (mass_type == MassSolverType::LUMPED) ? "LUMPED" : "CONSISTENT";
   const bool clamped = (empty_bdr_attr.Size() > 0 && empty_bdr_attr.Max() > 0);
   if (Mpi::Root())
   {
      mfem::out << "\n--- Design Taylor check (" << mass_label << " mass"
                << (clamped ? ", clamped Dirichlet BC" : "") << ") ---\n";
   }

   Vector rho0;
   rho.GetTrueDofs(rho0);

   const int state_size = 2 * state_fes.GetTrueVSize();
   const int design_size = control_fes.GetTrueVSize();

   Vector x0(state_size), direction(design_size);
   Vector grad(design_size), rho_plus(design_size), rho_minus(design_size);

   for (int trial = 0; trial < ntrials; trial++)
   {
      RandomState(x0, 500 + 2*trial);
      x0 *= state_scale;
      RandomState(direction, 501 + 2*trial);
      Normalize(comm, direction);

      // The design sensitivity integrators differentiate whichever mass matrix
      // (consistent or row-lumped) drives the forward solve, so J and dJ/drho are
      // self-consistent for both mass_type choices (see StageMassDesignLFIntegrator).
      const real_t J0 = DesignObjectiveAdjointGradient(
         rho0, x0, state_fes, filter_fes, control_fes, mass_type,
         rho, rho_tilde, filter,
         gamma_coef, exterior_bdr_attr, empty_bdr_attr, objective,
         mat, load_spec, load_coef, impedance, nsteps, h, grad);

      const real_t projected_grad = GlobalDot(comm, grad, direction);

      if (Mpi::Root())
      {
         mfem::out << "\nDesign Taylor trial " << trial
                   << ": J0=" << setprecision(16) << J0
                   << ", <dJ/drho,p>=" << projected_grad << '\n';
      }

      real_t scale = initial_scale;
      double previous_remainder = -1.0;
      double trial_best_fd_rel = numeric_limits<double>::infinity();
      bool trial_has_quadratic_drop = false;
      for (int s = 0; s < nscales; s++)
      {
         rho_plus = rho0;
         rho_minus = rho0;
         rho_plus.Add(scale, direction);
         rho_minus.Add(-scale, direction);

         const real_t Jp = EvaluateDesignObjective(
            rho_plus, x0, state_fes, control_fes, rho, rho_tilde, filter,
            gamma_coef, exterior_bdr_attr, empty_bdr_attr,
            objective, mat, load_spec, load_coef,
            impedance, nsteps, h, mass_type);

         const real_t Jm = EvaluateDesignObjective(
            rho_minus, x0, state_fes, control_fes, rho, rho_tilde, filter,
            gamma_coef, exterior_bdr_attr, empty_bdr_attr,
            objective, mat, load_spec, load_coef,
            impedance, nsteps, h, mass_type);

         const real_t fd = (Jp - Jm) / (2.0 * scale);
         const double derivative_scale =
            max(max(fabs(static_cast<double>(fd)),
                    fabs(static_cast<double>(projected_grad))), 1e-30);
         const double fd_rel = fabs(static_cast<double>(fd - projected_grad))
                               / derivative_scale;
         trial_best_fd_rel = min(trial_best_fd_rel, fd_rel);

         const real_t first_order_remainder =
            fabs(Jp - J0 - scale * projected_grad);
         const double remainder_ratio =
            (previous_remainder > 0.0) ?
            previous_remainder / first_order_remainder : 0.0;

         if (Mpi::Root())
         {
            mfem::out << "  scale=" << scientific << setprecision(3) << scale
                      << "  FD=" << setprecision(12) << fd
                      << "  rel_err=" << fd_rel
                      << "  first_order_rem=" << first_order_remainder;
            if (previous_remainder > 0.0)
            {
               mfem::out << "  rem_ratio=" << remainder_ratio;
            }
            mfem::out << '\n';
         }

         if (previous_remainder > 0.0 && remainder_ratio > 50.0)
         {
            trial_has_quadratic_drop = true;
         }

         previous_remainder = first_order_remainder;
         scale *= 0.1;
      }

      worst_best_fd_rel = max(worst_best_fd_rel, trial_best_fd_rel);
      MFEM_VERIFY(trial_best_fd_rel < tolerance,
                  "Raw design Taylor check did not find an accurate scale.");
      MFEM_VERIFY(trial_has_quadratic_drop,
                  "Raw design Taylor check did not show quadratic remainder decay.");
   }

   return worst_best_fd_rel;
}

void CheckProductionRevolveSchedules()
{
   // Regression values from the production spherical configurations.  These
   // are controller-only checks with a two-entry dummy state, so they remain
   // cheap while guarding against pathological or non-terminating schedules.
   TrajectoryCheckpointing<> fine_dt(/*num_steps=*/9000,
                                     /*num_checkpoints=*/200,
                                     /*state_size=*/2);
   TrajectoryCheckpointing<> coarse_dt(/*num_steps=*/1800,
                                       /*num_checkpoints=*/200,
                                       /*state_size=*/2);

   MFEM_VERIFY(fine_dt.EstimateRecomputations() == 8799,
               "Unexpected REVOLVE recomputation count for N=9000, C=200.");
   MFEM_VERIFY(coarse_dt.EstimateRecomputations() == 1599,
               "Unexpected REVOLVE recomputation count for N=1800, C=200.");
}

void CheckCheckpointPhysicalMetadataReplay()
{
   constexpr real_t start_time = 1.25;
   constexpr real_t time_step = 0.037;
   constexpr int state_size = 4;
   const int step_counts[] = {1, 2, 3, 9};

   for (int num_steps : step_counts)
   {
      std::vector<int> checkpoint_counts = {1, std::min(2, num_steps),
                                            num_steps};
      std::sort(checkpoint_counts.begin(), checkpoint_counts.end());
      checkpoint_counts.erase(
         std::unique(checkpoint_counts.begin(), checkpoint_counts.end()),
         checkpoint_counts.end());

      for (int num_checkpoints : checkpoint_counts)
      {
         TrajectoryCheckpointing<> checkpoint(
            num_steps, num_checkpoints, state_size,
            start_time, time_step);
         MFEM_VERIFY(
            checkpoint.MemoryFootprintBytes() ==
            static_cast<size_t>(num_checkpoints) *
            RK4Snapshot::ByteSize(state_size),
            "REVOLVE memory footprint does not equal C snapshots.");

         for (int generation = 1; generation <= 2; generation++)
         {
            checkpoint.Reset();
            Vector state(state_size);
            const auto encode_state =
               [&](int index, Vector &encoded)
               {
                  encoded[0] = index;
                  encoded[1] = start_time + index * time_step;
                  encoded[2] = generation;
                  encoded[3] = 1000.0 * generation + 17.0 * index;
               };
            const auto verify_encoded_state =
               [&](int index, const Vector &encoded)
               {
                  MFEM_VERIFY(
                     encoded.Size() == state_size &&
                     std::abs(encoded[0] - index) < 1e-14 &&
                     std::abs(encoded[1] -
                              (start_time + index * time_step)) < 1e-14 &&
                     std::abs(encoded[2] - generation) < 1e-14 &&
                     std::abs(encoded[3] -
                              (1000.0 * generation +
                               17.0 * index)) < 1e-14,
                     "REVOLVE restored stale state or inconsistent physical "
                     "metadata.");
               };
            encode_state(/*index=*/0, state);
            auto advance_state = [&](int index, Vector &encoded)
            {
               verify_encoded_state(index, encoded);
               encode_state(index + 1, encoded);
            };
            for (int step = 0; step < num_steps; step++)
            {
               checkpoint.ForwardStep(
                  step, state, start_time + step * time_step,
                  advance_state);
            }
            verify_encoded_state(num_steps, state);

            Vector adjoint(state_size);
            adjoint = 0.0;
            Vector forward_work(state);
            int expected_reverse_index = num_steps - 1;
            long long actual_controller_replays = 0;
            auto replay_forward_step = [&](int index, Vector &encoded)
            {
               actual_controller_replays++;
               advance_state(index, encoded);
            };
            auto consume_interval =
               [&](int index, const Vector &encoded, Vector &)
               {
                  MFEM_VERIFY(
                     index == expected_reverse_index,
                     "REVOLVE consumed intervals out of reverse order.");
                  verify_encoded_state(index, encoded);
                  expected_reverse_index--;
               };
            for (int step = num_steps - 1; step >= 0; step--)
            {
               checkpoint.BackwardInterval(
                  step, adjoint, forward_work,
                  replay_forward_step, consume_interval);
            }

            MFEM_VERIFY(expected_reverse_index == -1,
                        "REVOLVE did not consume every coarse interval.");
            MFEM_VERIFY(
               actual_controller_replays ==
               checkpoint.EstimateRecomputations(),
               "Actual synthetic replay count differs from the REVOLVE "
               "estimate.");
         }
      }
   }
}

void CheckExperimentLoadDurations()
{
   TransientTopOptConfig band_cfg;
   band_cfg.t_final = 6.0;
   BandWaveguideProblem band_default(band_cfg);
   MFEM_VERIFY(std::abs(band_default.GetBoundaryLoad().duration -
                        band_cfg.t_final) < 1e-14,
               "Band-waveguide default load must span t_final.");

   TransientTopOptConfig sphere_cfg;
   sphere_cfg.t_final = 9.0;
   SphericalBandGapProblem sphere_default(sphere_cfg);
   MFEM_VERIFY(std::abs(sphere_default.GetBoundaryLoad().duration -
                        sphere_cfg.t_final) < 1e-14,
               "Spherical default load must span t_final.");

   band_cfg.load_duration_is_user = true;
   band_cfg.boundary_load.duration = 0.8;
   BandWaveguideProblem band_override(band_cfg);
   MFEM_VERIFY(std::abs(band_override.GetBoundaryLoad().duration - 0.8) < 1e-14,
               "Band-waveguide must preserve an explicit -dur override.");

   sphere_cfg.load_duration_is_user = true;
   sphere_cfg.boundary_load.duration = 3.0;
   SphericalBandGapProblem sphere_override(sphere_cfg);
   MFEM_VERIFY(std::abs(sphere_override.GetBoundaryLoad().duration - 3.0) < 1e-14,
               "Spherical problem must preserve an explicit -dur override.");

   const real_t endpoint_envelope = std::exp(-2.0);
   const real_t sphere_start = EvaluateLoadTimeFactor(
      LoadTimeProfile::MODULATED_GAUSSIAN, 0.0, 9.0, 1.0, 0.0);
   const real_t sphere_mid = EvaluateLoadTimeFactor(
      LoadTimeProfile::MODULATED_GAUSSIAN, 4.5, 9.0, 1.0, 0.0);
   const real_t sphere_end = EvaluateLoadTimeFactor(
      LoadTimeProfile::MODULATED_GAUSSIAN, 9.0, 9.0, 1.0, 0.0);
   MFEM_VERIFY(std::abs(std::abs(sphere_start) - endpoint_envelope) < 1e-14 &&
               std::abs(sphere_mid - 1.0) < 1e-14 &&
               std::abs(std::abs(sphere_end) - endpoint_envelope) < 1e-14,
               "Full-window spherical Gaussian load has unexpected values.");

   const real_t harmonic_quarter_cycle = EvaluateLoadTimeFactor(
      LoadTimeProfile::HARMONIC, 0.25, 1.0, 1.0, 0.0);
   MFEM_VERIFY(std::abs(harmonic_quarter_cycle - 1.0) < 1e-14,
               "Harmonic load frequency must be interpreted as cycles/time.");
}

void CheckBandModeConverterSpecification()
{
   TransientTopOptConfig default_cfg;
   BandModeConverterProblem default_problem(default_cfg);
   MFEM_VERIFY(default_problem.GetTargetMode() == 8,
               "Resolved band mode-converter default must use mode n=8.");

   TransientTopOptConfig cfg;
   cfg.t_final = 3.5;
   cfg.dt = 5e-4;
   cfg.vol_frac = 0.43;
   cfg.filter_radius = 0.07;
   cfg.max_it = 13;
   cfg.move = 0.11;
   cfg.mode_converter_target_mode = 4;
   cfg.mode_converter_target_amplitude = 0.8;

   BandModeConverterProblem normal(cfg);
   BandModeConverterProblem reverse(cfg, /*reverse_spectral_roles=*/true);
   const TransientTopOptConfig &effective = normal.GetConfig();

   MFEM_VERIFY(normal.UsesPeriodicYBoundary(),
               "Band mode converter must request periodic y boundaries.");
   MFEM_VERIFY(!normal.ReversesSpectralRoles() &&
               reverse.ReversesSpectralRoles(),
               "Band mode-converter spectral roles are wired incorrectly.");
   MFEM_VERIFY(normal.GetTargetMode() == 4 &&
               std::abs(normal.GetTargetAmplitude() - 0.8) < 1e-14,
               "Band mode-converter target controls were not preserved.");
   MFEM_VERIFY(std::abs(effective.t_final - cfg.t_final) < 1e-14 &&
               std::abs(effective.dt - cfg.dt) < 1e-14 &&
               std::abs(effective.vol_frac - cfg.vol_frac) < 1e-14 &&
               std::abs(effective.filter_radius - cfg.filter_radius) < 1e-14 &&
               effective.max_it == cfg.max_it &&
               std::abs(effective.move - cfg.move) < 1e-14,
               "Band mode converter changed unrelated optimization controls.");
   MFEM_VERIFY(effective.boundary_load.domain_load &&
               effective.boundary_load.time_profile ==
                  LoadTimeProfile::MODULATED_GAUSSIAN &&
               effective.boundary_load.direction.Size() == 2 &&
               effective.boundary_load.direction[0] == 1.0 &&
               effective.boundary_load.direction[1] == 0.0 &&
               std::abs(effective.boundary_load.duration - 1.0) < 1e-14 &&
               std::abs(effective.boundary_load.frequency - 5.0) < 1e-14,
               "Band mode-converter default load is inconsistent.");
   MFEM_VERIFY(std::abs(normal.GetPassiveDensity() - 1.0) < 1e-14 &&
               std::abs(reverse.GetPassiveDensity() - 1.0) < 1e-14,
               "Band mode-converter collars must be passive solid material.");

   Array<int> essential, absorbing;
   normal.GetEssentialBoundaryAttributes(essential);
   normal.GetAbsorbingBoundaryAttributes(absorbing);
   MFEM_VERIFY(essential.Size() == 1 && essential[0] == 4 &&
               absorbing.Size() == 1 && absorbing[0] == 2,
               "Band mode-converter boundary attributes are inconsistent.");
   const DampingParameters damping = normal.GetDampingParameters();
   MFEM_VERIFY(std::abs(damping.thickness - 0.75) < 1e-14 &&
               damping.damp_left && damping.damp_right &&
               !damping.damp_bottom && !damping.damp_top,
               "Band mode-converter damping must be confined to both ends.");

   Mesh mesh = normal.CreateMesh();
   MFEM_VERIFY(mesh.Dimension() == 2 && mesh.GetNE() == 384 * 64,
               "Band mode-converter generated mesh has unexpected dimensions.");
   real_t x_max = 0.0, y_max = 0.0;
   normal.GetReferenceDomainExtents(x_max, y_max);
   MFEM_VERIFY(std::abs(x_max - 12.0) < 1e-14 &&
               std::abs(y_max - 1.0) < 1e-14,
               "Band mode-converter domain extents are inconsistent.");

   std::unique_ptr<Coefficient> passive =
      normal.CreatePassiveRegionCoefficient();
   std::unique_ptr<VectorCoefficient> low_source =
      normal.CreateBoundaryLoadCoefficient();
   std::unique_ptr<VectorCoefficient> high_source =
      reverse.CreateBoundaryLoadCoefficient();
   std::unique_ptr<VectorCoefficient> normal_launched_probe =
      normal.CreateForwardModalProbe();
   std::unique_ptr<VectorCoefficient> normal_target_probe =
      normal.CreateTargetModalProbe();
   std::unique_ptr<VectorCoefficient> reverse_launched_probe =
      reverse.CreateForwardModalProbe();
   std::unique_ptr<VectorCoefficient> reverse_target_probe =
      reverse.CreateTargetModalProbe();
   MFEM_VERIFY(normal_launched_probe && normal_target_probe &&
               reverse_launched_probe && reverse_target_probe,
               "Band mode-converter launched/target probes must be defined "
               "for both spectral-role directions.");
   real_t passive_measure = 0.0;
   real_t low_norm_sq = 0.0, high_norm_sq = 0.0;
   real_t low_high_inner_product = 0.0;
   real_t low_active_leak = 0.0, high_active_leak = 0.0;
   real_t normal_launched_norm_sq = 0.0, normal_target_norm_sq = 0.0;
   real_t reverse_launched_norm_sq = 0.0, reverse_target_norm_sq = 0.0;
   real_t normal_probe_inner_product = 0.0;
   real_t reverse_probe_inner_product = 0.0;
   real_t launched_role_swap_error_sq = 0.0;
   real_t target_role_swap_error_sq = 0.0;
   Vector low_value, high_value;
   Vector normal_launched_value, normal_target_value;
   Vector reverse_launched_value, reverse_target_value;

   for (int e = 0; e < mesh.GetNE(); e++)
   {
      ElementTransformation *T = mesh.GetElementTransformation(e);
      const Geometry::Type geom = mesh.GetElementBaseGeometry(e);
      const IntegrationRule &ir = IntRules.Get(geom, 10);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T->SetIntPoint(&ip);
         const real_t weight = ip.weight * T->Weight();
         const real_t passive_value = passive->Eval(*T, ip);
         low_source->Eval(low_value, *T, ip);
         high_source->Eval(high_value, *T, ip);
         normal_launched_probe->Eval(normal_launched_value, *T, ip);
         normal_target_probe->Eval(normal_target_value, *T, ip);
         reverse_launched_probe->Eval(reverse_launched_value, *T, ip);
         reverse_target_probe->Eval(reverse_target_value, *T, ip);
         passive_measure += weight * passive_value;
         low_norm_sq += weight * (low_value * low_value);
         high_norm_sq += weight * (high_value * high_value);
         low_high_inner_product += weight * (low_value * high_value);
         low_active_leak += weight * (1.0 - passive_value) *
                            (low_value * low_value);
         high_active_leak += weight * (1.0 - passive_value) *
                             (high_value * high_value);
         normal_launched_norm_sq +=
            weight * (normal_launched_value * normal_launched_value);
         normal_target_norm_sq +=
            weight * (normal_target_value * normal_target_value);
         reverse_launched_norm_sq +=
            weight * (reverse_launched_value * reverse_launched_value);
         reverse_target_norm_sq +=
            weight * (reverse_target_value * reverse_target_value);
         normal_probe_inner_product +=
            weight * (normal_launched_value * normal_target_value);
         reverse_probe_inner_product +=
            weight * (reverse_launched_value * reverse_target_value);
         normal_launched_value -= reverse_target_value;
         reverse_launched_value -= normal_target_value;
         launched_role_swap_error_sq +=
            weight * (normal_launched_value * normal_launched_value);
         target_role_swap_error_sq +=
            weight * (reverse_launched_value * reverse_launched_value);
      }
   }

   MFEM_VERIFY(std::abs(passive_measure - 8.25) < 1e-10,
               "Band mode-converter passive mask has the wrong measure: "
               << passive_measure);
   MFEM_VERIFY(std::abs(low_norm_sq - 1.0) < 1e-11 &&
               std::abs(high_norm_sq - 1.0) < 1e-11,
               "Band mode-converter source modes are not L2-normalized.");
   MFEM_VERIFY(std::abs(low_high_inner_product) < 1e-11,
               "Band mode-converter low/high source modes are not orthogonal.");
   MFEM_VERIFY(std::abs(low_active_leak) < 1e-14 &&
               std::abs(high_active_leak) < 1e-14,
               "Band mode-converter source collar overlaps the active design.");
   MFEM_VERIFY(std::abs(normal_launched_norm_sq - 1.0) < 1e-11 &&
               std::abs(normal_target_norm_sq - 1.0) < 1e-11 &&
               std::abs(reverse_launched_norm_sq - 1.0) < 1e-11 &&
               std::abs(reverse_target_norm_sq - 1.0) < 1e-11,
               "Band mode-converter receiver probes are not L2-normalized.");
   MFEM_VERIFY(std::abs(normal_probe_inner_product) < 1e-11 &&
               std::abs(reverse_probe_inner_product) < 1e-11,
               "Band mode-converter launched/target receiver probes are not "
               "orthogonal.");
   MFEM_VERIFY(std::abs(launched_role_swap_error_sq) < 1e-14 &&
               std::abs(target_role_swap_error_sq) < 1e-14,
               "Band mode-converter receiver probes do not exchange modes "
               "when spectral roles are reversed.");

   TransientTopOptConfig override_cfg = cfg;
   override_cfg.load_frequency_is_user = true;
   override_cfg.boundary_load.frequency = 6.25;
   override_cfg.load_duration_is_user = true;
   override_cfg.boundary_load.duration = 0.9;
   BandModeConverterProblem overridden(override_cfg);
   MFEM_VERIFY(std::abs(overridden.GetBoundaryLoad().frequency - 6.25) < 1e-14 &&
               std::abs(overridden.GetBoundaryLoad().duration - 0.9) < 1e-14,
               "Band mode converter must preserve -freq/-dur overrides.");

   TransientTopOptConfig invalid_cfg = cfg;
   invalid_cfg.mode_converter_target_mode = 3;
   BandModeConverterProblem invalid(invalid_cfg);
   std::ostringstream validation;
   MFEM_VERIFY(!invalid.Validate(validation),
               "Periodic-y mode converter must reject odd cos(n*pi*y/H) modes.");

   TransientTopOptConfig invalid_amplitude_cfg = cfg;
   invalid_amplitude_cfg.mode_converter_target_amplitude =
      std::numeric_limits<real_t>::quiet_NaN();
   BandModeConverterProblem invalid_amplitude(invalid_amplitude_cfg);
   std::ostringstream amplitude_validation;
   MFEM_VERIFY(!invalid_amplitude.Validate(amplitude_validation),
               "Band mode converter must reject a non-finite target amplitude.");

   TransientTopOptConfig invalid_frequency_cfg = cfg;
   invalid_frequency_cfg.load_frequency_is_user = true;
   invalid_frequency_cfg.boundary_load.frequency =
      std::numeric_limits<real_t>::infinity();
   BandModeConverterProblem invalid_frequency(invalid_frequency_cfg);
   std::ostringstream frequency_validation;
   MFEM_VERIFY(!invalid_frequency.Validate(frequency_validation),
               "Band mode converter must reject a non-finite carrier frequency.");
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   MPI_Comm comm = MPI_COMM_WORLD;
   const int myid = Mpi::WorldRank();

   int ref_levels = 0;
   int order = 1;
   int design_order = -1;
   int ntrials = 3;
   int nsteps = 8;
   int taylor_scales = 6;
   real_t dt = 5e-5;
   real_t eps = 1.0;
   real_t tolerance = 1e-7;
   real_t taylor_initial_scale = 1e-1;
   real_t taylor_tolerance = 1e-5;
   real_t design_initial_scale = 1e-1;
   real_t design_state_scale = 100.0;
   real_t design_tolerance = 1e-4;
   real_t vol_frac = 0.5;
   real_t filter_radius = 0.05;
   real_t protected_radius = 0.2;
   const char *mesh_file = "lamb-problem-damping-mesh-triangs.msh";

   OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--refine", "Refinement level");
   args.AddOption(&order, "-o", "--order", "H1 finite element order");
   args.AddOption(&design_order, "-do", "--design-order",
                  "H1 order of rho_tilde; rho uses paired L2 order "
                  "max(0, design-order-1). Default: --order.");
   args.AddOption(&ntrials, "-nt", "--num-trials", "Random trials per check");
   args.AddOption(&nsteps, "-ns", "--num-steps", "RK4 steps for n-step check");
   args.AddOption(&taylor_scales, "-ts", "--taylor-scales",
                  "Number of Taylor finite-difference scales");
   args.AddOption(&taylor_initial_scale, "-t0", "--taylor-initial-scale",
                  "Initial Taylor finite-difference scale");
   args.AddOption(&dt, "-dt", "--time-step", "RK4 time step");
   args.AddOption(&eps, "-eps", "--epsilon", "Centered finite-difference step");
   args.AddOption(&tolerance, "-tol", "--tolerance", "Relative error tolerance");
   args.AddOption(&taylor_tolerance, "-ttol", "--taylor-tolerance",
                  "Objective Taylor relative derivative tolerance");
   args.AddOption(&design_initial_scale, "-d0", "--design-initial-scale",
                  "Initial raw-design Taylor finite-difference scale");
   args.AddOption(&design_state_scale, "-ds", "--design-state-scale",
                  "Initial-state amplification used only by design Taylor test");
   args.AddOption(&design_tolerance, "-dtol", "--design-tolerance",
                  "Raw-design Taylor relative derivative tolerance");
   args.AddOption(&vol_frac, "-vf", "--vol-frac", "Uniform control density");
   args.AddOption(&filter_radius, "-fr", "--filter-radius",
                  "Helmholtz filter radius");
   args.AddOption(&protected_radius, "-pr", "--protected-radius",
                  "Circular protected-zone radius for objective");
   args.AddOption(&mesh_file, "-mesh", "--mesh-file", "Mesh file");
   args.Parse();

   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (design_order < 0) { design_order = order; }
   if (design_order < 1)
   {
      if (myid == 0)
      {
         cerr << "Error: -do/--design-order must be at least 1.\n";
      }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   Device device("cpu");
   CheckProductionRevolveSchedules();
   CheckCheckpointPhysicalMetadataReplay();
   CheckExperimentLoadDurations();
   CheckBandModeConverterSpecification();
   CheckNestedTimeGridContract();
   CheckLateIntervalHermiteStageTimes();
   const HermiteProviderCheck hermite_provider_check =
      CheckCubicHermiteForwardStateProvider();

   ifstream imesh(mesh_file);
   if (!imesh)
   {
      if (myid == 0)
      {
         cerr << "Error: cannot open mesh file '" << mesh_file << "'.\n";
      }
      return 1;
   }

   Mesh mesh(imesh, 1, 1);
   imesh.close();
   const int dim = mesh.Dimension();

   for (int l = 0; l < ref_levels; l++) { mesh.UniformRefinement(); }

   ParMesh pmesh(comm, mesh);
   mesh.Clear();

   H1_FECollection state_fec(order, dim);
   H1_FECollection filter_fec(design_order, dim);
   L2_FECollection control_fec(max(0, design_order - 1), dim,
                               BasisType::GaussLobatto);

   ParFiniteElementSpace state_fes(&pmesh, &state_fec, dim);
   ParFiniteElementSpace filter_fes(&pmesh, &filter_fec);
   ParFiniteElementSpace control_fes(&pmesh, &control_fec);

   ParGridFunction rho(&control_fes);
   ParGridFunction rho_tilde(&filter_fes);
   rho = vol_frac;

   toopt::PDEFilterOptions filter_opts;
   filter_opts.filter_radius = filter_radius;
   toopt::PDEFilter filter(filter_fes, control_fes, filter_opts);
   filter.Assemble();
   filter.Mult(rho, rho_tilde);

   MaterialParams mat;

   ConstantCoefficient rho_0_coef(mat.rho0);
   ConstantCoefficient lambda_0_coef(mat.lambda0);
   ConstantCoefficient mu_0_coef(mat.mu0);

   SIMPCoefficient simp_mass(&rho_tilde, mat.r_min, mat.r_max, mat.simp_p);
   SIMPCoefficient simp_stiff(&rho_tilde, mat.r_min, mat.r_max, mat.simp_p);

   ProductCoefficient mass_coef(simp_mass, rho_0_coef);
   ProductCoefficient lambda_coef(simp_stiff, lambda_0_coef);
   ProductCoefficient mu_coef(simp_stiff, mu_0_coef);

   const real_t x_max = 1.5;
   const real_t y_max = 0.75;
   const real_t c_p = sqrt((mat.lambda0 + 2.0*mat.mu0) / mat.rho0);
   const real_t damping_thickness = 0.25;
   DampingProfile phi_profile(damping_thickness, x_max, y_max);
   const real_t gamma_max = (2.0 * c_p / 0.2136) * log(1.0 / 1e-4);
   SpatialDampingCoefficient gamma_coef(&phi_profile, gamma_max,
                                        mat.rho0, 2.0, 2);

   BoundaryLoadSpec load_spec;
   DirectionalBoundaryLoadCoefficient load_coef(load_spec.direction);
   const real_t impedance = mat.rho0 * c_p;

   Array<int> exterior_bdr_attr(pmesh.bdr_attributes.Max());
   exterior_bdr_attr = 0;
   if (pmesh.bdr_attributes.Max() >= 10) { exterior_bdr_attr[9] = 1; }
   if (pmesh.bdr_attributes.Max() >= 11) { exterior_bdr_attr[10] = 1; }
   if (pmesh.bdr_attributes.Max() >= 12) { exterior_bdr_attr[11] = 1; }

   Array<int> empty_bdr_attr(pmesh.bdr_attributes.Max());
   empty_bdr_attr = 0;

   // A non-empty essential marker to exercise clamped Dirichlet BC enforcement in
   // the design-gradient check (physical meaning irrelevant here; we only need
   // some essential dofs present to verify the adjoint stays transpose-consistent).
   Array<int> clamped_bdr_attr(pmesh.bdr_attributes.Max());
   clamped_bdr_attr = 0;
   if (pmesh.bdr_attributes.Max() >= 21) { clamped_bdr_attr[20] = 1; }

   ElastodynamicsOperator oper(
      state_fes, mass_coef, lambda_coef, mu_coef,
      load_spec.amplitude, load_spec.duration, load_spec.time_profile,
      load_spec.phase, load_spec.frequency, load_spec.bdr_attributes, load_coef,
      load_spec.domain_load,
      &gamma_coef, impedance, exterior_bdr_attr, empty_bdr_attr,
      MassSolverType::LUMPED);

   ConsistentMassConstraintCheck consistent_mass_constraint_check;
   {
      ElastodynamicsOperator consistent_clamped_oper(
         state_fes, mass_coef, lambda_coef, mu_coef,
         load_spec.amplitude, load_spec.duration, load_spec.time_profile,
         load_spec.phase, load_spec.frequency, load_spec.bdr_attributes,
         load_coef, load_spec.domain_load,
         &gamma_coef, impedance, exterior_bdr_attr, clamped_bdr_attr,
         MassSolverType::ITERATIVE, /*print_banner=*/false);
      consistent_mass_constraint_check =
         CheckConstrainedConsistentMassInverse(
            consistent_clamped_oper, comm);
   }

   const ForwardReplayIntegrityCheck replay_integrity_check =
      CheckForwardIntervalReplayIntegrity(oper, comm);
   const double terminal_functional_error =
      CheckContinuousTerminalFunctionalSign(oper, state_fes, comm);

   SubdomainIndicator subdomain_indicator(x_max/2.0, y_max/2.0,
                                          protected_radius);
   DisplacementL2Objective objective(&state_fes, subdomain_indicator, comm);

   const int state_size = oper.Height();

   const double instantaneous_l2_error =
      CheckInstantaneousObjectiveGradient(
         state_fes, oper.GetBlockOffsets(), objective, comm,
         state_size, /*seed=*/610, /*physical_time=*/0.137);

   Vector target_value(dim);
   target_value = 0.0;
   target_value[0] = 1.0;
   auto tracking_region = std::make_unique<ConstantCoefficient>(1.0);
   auto tracking_mode =
      std::make_unique<VectorConstantCoefficient>(target_value);
   HarmonicDisplacementTrackingObjective tracking_objective(
      &state_fes, std::move(tracking_region), std::move(tracking_mode),
      /*amplitude=*/0.7, /*frequency=*/1.3, /*phase=*/0.2, comm);
   const double instantaneous_tracking_error =
      CheckInstantaneousObjectiveGradient(
         state_fes, oper.GetBlockOffsets(), tracking_objective, comm,
         state_size, /*seed=*/620, /*physical_time=*/0.137);

   auto correlation_mode =
      std::make_unique<VectorConstantCoefficient>(target_value);
   HarmonicModalCorrelationObjective correlation_objective(
      &state_fes, std::move(correlation_mode),
      /*amplitude=*/0.7, /*frequency=*/1.3, /*phase=*/0.2, comm);
   const double instantaneous_correlation_error =
      CheckInstantaneousObjectiveGradient(
         state_fes, oper.GetBlockOffsets(), correlation_objective, comm,
         state_size, /*seed=*/625, /*physical_time=*/0.137);

   Vector instantaneous_load_direction(dim);
   instantaneous_load_direction = 0.0;
   instantaneous_load_direction[dim - 1] = -1.0;
   DirectionalBoundaryLoadCoefficient instantaneous_load(
      instantaneous_load_direction);
   ComplianceObjective instantaneous_compliance(
      &state_fes, instantaneous_load, comm);
   const double instantaneous_compliance_error =
      CheckInstantaneousObjectiveGradient(
         state_fes, oper.GetBlockOffsets(), instantaneous_compliance, comm,
         state_size, /*seed=*/630, /*physical_time=*/0.137);
   const double continuous_adjoint_error =
      CheckContinuousAdjointDirectionalDerivative(
         oper, state_fes, oper.GetBlockOffsets(), objective, comm,
         state_size, nsteps, dt);
   const ContinuousDesignReplayCheck continuous_replay_check =
      CheckContinuousDesignReplayEquivalence(
         oper, state_fes, filter_fes, control_fes, rho, rho_tilde, filter,
         gamma_coef, exterior_bdr_attr, empty_bdr_attr, mat, load_spec,
         load_coef, impedance, objective, comm, nsteps, dt);

   if (myid == 0)
   {
      mfem::out << "\n=== Adjoint Verification ===\n"
                << "State size: " << state_size << '\n'
                << "Trials: " << ntrials << '\n'
                << "RK4 n-step count: " << nsteps << '\n'
                << "dt: " << dt << '\n'
                << "eps: " << eps << '\n'
                << "transpose tolerance: " << tolerance << '\n'
                << "state Taylor tolerance: " << taylor_tolerance << '\n'
                << "design Taylor tolerance: " << design_tolerance << "\n\n";
   }

   const double jac_err =
      CheckJacobianTranspose(oper, comm, state_size, ntrials, eps, tolerance);
   const double one_step_err =
      CheckRK4OneStepTranspose(oper, comm, state_size, ntrials, dt, eps,
                               tolerance);
   const double n_step_err =
      CheckRK4NStepTranspose(oper, comm, state_size, nsteps, ntrials, dt, eps,
                             tolerance);
   const double taylor_err =
      CheckObjectiveTaylor(oper, state_fes, oper.GetBlockOffsets(), objective,
                           comm, state_size, nsteps, ntrials, dt,
                           taylor_scales, taylor_initial_scale,
                           taylor_tolerance);
   // Verify the design gradient for BOTH mass discretizations: the sensitivity
   // must be self-consistent with whichever mass solver drives the forward solve.
   const double design_taylor_err_cg =
      CheckDesignTaylor(state_fes, filter_fes, control_fes, rho, rho_tilde,
                        filter, gamma_coef, exterior_bdr_attr, empty_bdr_attr,
                        objective, mat, load_spec, load_coef,
                        impedance, nsteps, dt, ntrials,
                        taylor_scales, design_initial_scale,
                        design_state_scale, design_tolerance,
                        MassSolverType::ITERATIVE);

   const double design_taylor_err_lumped =
      CheckDesignTaylor(state_fes, filter_fes, control_fes, rho, rho_tilde,
                        filter, gamma_coef, exterior_bdr_attr, empty_bdr_attr,
                        objective, mat, load_spec, load_coef,
                        impedance, nsteps, dt, ntrials,
                        taylor_scales, design_initial_scale,
                        design_state_scale, design_tolerance,
                        MassSolverType::LUMPED);
   // Same gradient check but with a clamped Dirichlet BC active, to verify the
   // essential-dof projection is applied consistently in forward + adjoint.
   const double design_taylor_err_clamped =
      CheckDesignTaylor(state_fes, filter_fes, control_fes, rho, rho_tilde,
                        filter, gamma_coef, exterior_bdr_attr, clamped_bdr_attr,
                        objective, mat, load_spec, load_coef,
                        impedance, nsteps, dt, ntrials,
                        taylor_scales, design_initial_scale,
                        design_state_scale, design_tolerance,
                        MassSolverType::LUMPED);
   const double design_taylor_err_consistent_clamped =
      CheckDesignTaylor(state_fes, filter_fes, control_fes, rho, rho_tilde,
                        filter, gamma_coef, exterior_bdr_attr, clamped_bdr_attr,
                        objective, mat, load_spec, load_coef,
                        impedance, nsteps, dt, ntrials,
                        taylor_scales, design_initial_scale,
                        design_state_scale, design_tolerance,
                        MassSolverType::ITERATIVE);

   // Verify the ComplianceObjective (J = int f.u) gradient path, used by the
   // cantilever-compliance problem.  BoundaryLoadSpec's default direction is
   // two-dimensional, while this test also accepts 3D meshes; build a
   // dimension-matched coefficient so the compliance check remains meaningful
   // for the spherical p-state/design-order regression.
   Vector compliance_direction(dim);
   compliance_direction = 0.0;
   compliance_direction[dim - 1] = -1.0;
   DirectionalBoundaryLoadCoefficient compliance_load_coef(
      compliance_direction);
   ComplianceObjective compliance_obj(&state_fes, compliance_load_coef, comm);
   const double design_taylor_err_compliance =
      CheckDesignTaylor(state_fes, filter_fes, control_fes, rho, rho_tilde,
                        filter, gamma_coef, exterior_bdr_attr, empty_bdr_attr,
                        compliance_obj, mat, load_spec, compliance_load_coef,
                        impedance, nsteps, dt, ntrials,
                        taylor_scales, design_initial_scale,
                        design_state_scale, design_tolerance,
                        MassSolverType::LUMPED);

   if (myid == 0)
   {
      mfem::out << "\nAll adjoint and objective Taylor checks passed.\n"
                << "Worst Jacobian transpose error: "
                << scientific << setprecision(6) << jac_err << '\n'
                << "Worst RK4 one-step transpose error: "
                << one_step_err << '\n'
                << "Worst RK4 n-step transpose error: "
                << n_step_err << '\n'
                << "Instantaneous L2 objective-gradient error: "
                << instantaneous_l2_error << '\n'
                << "Instantaneous tracking objective-gradient error: "
                << instantaneous_tracking_error << '\n'
                << "Instantaneous correlation objective-gradient error: "
                << instantaneous_correlation_error << '\n'
                << "Instantaneous compliance objective-gradient error: "
                << instantaneous_compliance_error << '\n'
                << "Hermite exact-cubic maximum error: "
                << hermite_provider_check.cubic_max_error << '\n'
                << "Hermite oscillator finest-grid error: "
                << hermite_provider_check.oscillator_finest_error << '\n'
                << "Hermite oscillator minimum observed order: "
                << hermite_provider_check.oscillator_minimum_order << '\n'
                << "Poisoned replay endpoint error: "
                << replay_integrity_check.endpoint_relative_error << '\n'
                << "Poisoned replay left-slope error: "
                << replay_integrity_check.left_slope_relative_error << '\n'
                << "Poisoned replay right-slope error: "
                << replay_integrity_check.right_slope_relative_error << '\n'
                << "Continuous terminal-functional/sign error: "
                << terminal_functional_error << '\n'
                << "Continuous-adjoint directional-derivative error: "
                << continuous_adjoint_error << '\n'
                << "Continuous full/REVOLVE objective error: "
                << continuous_replay_check.objective_relative_error << '\n'
                << "Continuous full/REVOLVE initial-adjoint error: "
                << continuous_replay_check.initial_adjoint_relative_error
                << '\n'
                << "Continuous full/REVOLVE filtered-gradient error: "
                << continuous_replay_check.filtered_gradient_relative_error
                << '\n'
                << "Continuous full/REVOLVE raw-gradient error: "
                << continuous_replay_check.raw_gradient_relative_error << '\n'
                << "Continuous replayed-endpoint error: "
                << continuous_replay_check.replayed_endpoint_relative_error
                << '\n'
                << "Continuous filtered-gradient FD error: "
                << continuous_replay_check.filtered_gradient_fd_relative_error
                << '\n'
                << "Continuous raw-gradient FD error: "
                << continuous_replay_check.raw_gradient_fd_relative_error
                << '\n'
                << "Worst objective Taylor FD error: "
                << taylor_err << '\n'
                << "Worst raw-design Taylor FD error (consistent mass): "
                << design_taylor_err_cg << '\n'
                << "Worst raw-design Taylor FD error (lumped mass): "
                << design_taylor_err_lumped << '\n'
                << "Worst raw-design Taylor FD error (lumped, clamped BC): "
                << design_taylor_err_clamped << '\n'
                << "Worst raw-design Taylor FD error (consistent, clamped BC): "
                << design_taylor_err_consistent_clamped << '\n'
                << "Consistent constrained-mass free residual: "
                << consistent_mass_constraint_check.free_residual_relative_error
                << '\n'
                << "Consistent constrained-mass inverse symmetry error: "
                << consistent_mass_constraint_check.inverse_symmetry_relative_error
                << '\n'
                << "Consistent constrained-mass essential maximum: "
                << consistent_mass_constraint_check.constrained_value_maximum
                << '\n'
                << "Consistent constrained-mass wave dt limit: "
                << consistent_mass_constraint_check.wave_time_step_limit
                << '\n'
                << "Consistent constrained-mass damping dt limit: "
                << consistent_mass_constraint_check.damping_time_step_limit
                << '\n'
                << "Worst raw-design Taylor FD error (compliance objective): "
                << design_taylor_err_compliance << '\n';
   }

   return 0;
}
