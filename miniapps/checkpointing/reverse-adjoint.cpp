// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.

/** @file
    Demonstrates bounded reverse reconstruction and discrete adjoints.

    The scalar equation u' = p u - u^3 is advanced with the selected MFEM
    one-step solver. StoreEverything, Revolve, and online WMI all drive the
    same CheckpointController and application-owned reverse callback.

    The objective is J = 0.5 u_N^2. The program first computes a reference
    gradient from the full trajectory, then repeats the forward solve under a
    checkpoint schedule and reconstructs the required primal states during the
    reverse sweep. Both paths differentiate the discrete solver update rather
    than the continuous ODE. */

#include "mfem.hpp"
#include "checkpoint_demo.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace mfem;
using namespace mfem::checkpoint_demo;
using namespace std;

namespace
{

/// Supported one-step discretizations of the scalar model.
enum class SolverKind { ForwardEuler, BackwardEuler };

/// Miniapp-specific signature and version for complete primal snapshots.
constexpr std::uint64_t snapshot_magic = UINT64_C(0x324a444154504843);
constexpr std::uint64_t snapshot_version = 1;

/// Scalar operator f(u,p) = p u - u^3 used by both time integrators.
class CubicOperator : public TimeDependentOperator
{
private:
   real_t parameter; ///< Constant model parameter p.

public:
   explicit CubicOperator(real_t parameter_)
      : TimeDependentOperator(1), parameter(parameter_) { }

   /// Return the model parameter differentiated by the adjoint.
   real_t Parameter() const { return parameter; }

   /// Evaluate the explicit right-hand side f(u,p).
   void Mult(const Vector &state, Vector &rate) const override
   {
      rate.SetSize(1);
      rate[0] = parameter * state[0] - state[0] * state[0] * state[0];
   }

   void ImplicitSolve(real_t dt, const Vector &state,
                      Vector &rate) override
   {
      // BackwardEulerSolver requests k satisfying
      //   next = state + dt*k,  k = f(next,p).
      // Use the predecessor as the deterministic Newton initial guess. These
      // iteration and convergence rules are replayed after every restore.
      real_t next = state[0];
      for (int iteration = 0; iteration < 20; iteration++)
      {
         const real_t residual = next - state[0] -
            dt * (parameter * next - next * next * next);
         const real_t derivative = 1.0 -
            dt * (parameter - 3.0 * next * next);
         const real_t update = residual / derivative;
         next -= update;
         if (std::abs(update) <= 16.0 *
             std::numeric_limits<real_t>::epsilon() *
             std::max(real_t(1.0), std::abs(next)))
         {
            rate.SetSize(1);
            rate[0] = (next - state[0]) / dt;
            return;
         }
      }
      throw InvalidCheckpointState(
         "Backward Euler nonlinear solve did not converge");
   }
};

/// Capture and restore all state needed for bitwise-identical ODE replay.
/** In addition to the scalar solution, a snapshot records the logical state,
    physical time, step size, parameter, solver choice, format version, and
    optional persistent checkpoint identity. Restore also resets the operator
    time and reinitializes the selected solver, so no hidden solver state is
    inherited from a later point in the trajectory. */
class CubicStateAdapter : public CheckpointStateAdapter
{
private:
   ODESolver &solver;
   CubicOperator &oper;
   Vector &state;
   TimePoint &time;
   real_t &dt;
   SolverKind kind;

public:
   /// Bind externally owned forward-solve state for capture and restoration.
   CubicStateAdapter(ODESolver &solver_, CubicOperator &oper_, Vector &state_,
                     TimePoint &time_, real_t &dt_, SolverKind kind_)
      : solver(solver_), oper(oper_), state(state_), time(time_), dt(dt_),
        kind(kind_) { }

   /// Serialize one complete, self-validating primal state.
   Snapshot Capture(
      StateId id,
      std::optional<CheckpointId> checkpoint = std::nullopt) const override
   {
      if (id != time.step || state.Size() != 1 || !(dt > 0.0))
      {
         throw InvalidCheckpointState("invalid cubic ODE state for capture");
      }
      SnapshotWriter writer;
      writer.WriteUInt64(snapshot_magic);
      writer.WriteUInt64(snapshot_version);
      writer.WriteStateId(id);
      writer.WriteUInt64(checkpoint ? 1 : 0);
      writer.WriteUInt64(checkpoint.value_or(0));
      writer.WriteUInt64(kind == SolverKind::ForwardEuler ? 0 : 1);
      writer.WriteDouble(static_cast<double>(oper.Parameter()));
      writer.WriteDouble(static_cast<double>(time.time));
      writer.WriteDouble(static_cast<double>(dt));
      writer.WriteDouble(static_cast<double>(state[0]));
      return writer.Finish();
   }

   /// Validate and restore one complete primal state.
   void Restore(
      StateId id, const Snapshot &snapshot,
      std::optional<CheckpointId> checkpoint = std::nullopt) override
   {
      SnapshotReader reader(snapshot);
      const std::uint64_t magic = reader.ReadUInt64();
      const std::uint64_t version = reader.ReadUInt64();
      const StateId restored_id = reader.ReadStateId();
      const bool has_checkpoint = reader.ReadUInt64() != 0;
      const CheckpointId restored_checkpoint = reader.ReadUInt64();
      const std::uint64_t restored_kind = reader.ReadUInt64();
      const real_t restored_parameter =
         static_cast<real_t>(reader.ReadDouble());
      const real_t restored_time = static_cast<real_t>(reader.ReadDouble());
      const real_t restored_dt = static_cast<real_t>(reader.ReadDouble());
      const real_t restored_state = static_cast<real_t>(reader.ReadDouble());
      reader.RequireEnd();
      if (magic != snapshot_magic || version != snapshot_version ||
          restored_id != id || has_checkpoint != checkpoint.has_value() ||
          (checkpoint && restored_checkpoint != *checkpoint) ||
          (!checkpoint && restored_checkpoint != 0) ||
          restored_kind != (kind == SolverKind::ForwardEuler ? 0 : 1))
      {
         throw InvalidCheckpointFormat("incompatible cubic ODE snapshot");
      }
      if (restored_parameter != oper.Parameter() || !(restored_dt > 0.0) ||
          !std::isfinite(restored_time) || !std::isfinite(restored_state))
      {
         throw InvalidCheckpointState("invalid cubic ODE restart state");
      }
      state[0] = restored_state;
      time = TimePoint{restored_id, restored_time};
      dt = restored_dt;
      oper.SetTime(time.time);
      solver.Init(oper);
   }
};

/// Terminal state and parameter derivative of the terminal objective.
struct ReferenceResult
{
   real_t terminal;
   real_t gradient;
};

/// Construct the selected MFEM one-step integrator.
unique_ptr<ODESolver> MakeSolver(SolverKind kind)
{
   if (kind == SolverKind::ForwardEuler)
   {
      return std::make_unique<ForwardEulerSolver>();
   }
   return std::make_unique<BackwardEulerSolver>();
}

/// Return d Phi_n / d u_n for the discrete update u_(n+1) = Phi_n.
/** Forward Euler differentiates its explicit map at @a predecessor. Backward
    Euler differentiates the accepted implicit residual at @a successor. */
real_t StateDerivative(SolverKind kind, real_t predecessor,
                       real_t successor, real_t parameter, real_t dt)
{
   if (kind == SolverKind::ForwardEuler)
   {
      return 1.0 + dt * (parameter - 3.0 * predecessor * predecessor);
   }
   const real_t denominator =
      1.0 - dt * (parameter - 3.0 * successor * successor);
   return 1.0 / denominator;
}

/// Return d Phi_n / d p for the selected discrete update.
/** For Backward Euler, implicit differentiation gives
    h*u_(n+1)/(1-h*(p-3*u_(n+1)^2)). */
real_t ParameterDerivative(SolverKind kind, real_t predecessor,
                           real_t successor, real_t parameter, real_t dt)
{
   if (kind == SolverKind::ForwardEuler) { return dt * predecessor; }
   const real_t denominator =
      1.0 - dt * (parameter - 3.0 * successor * successor);
   return dt * successor / denominator;
}

/// Compute the terminal value and discrete gradient from a full trajectory.
/** This deliberately stores every scalar state independently of the
    checkpoint runtime and therefore serves as the numerical reference. */
ReferenceResult ComputeReference(SolverKind kind, StateId steps,
                                 real_t parameter, real_t dt, real_t initial)
{
   CubicOperator oper(parameter);
   unique_ptr<ODESolver> solver = MakeSolver(kind);
   solver->Init(oper);
   Vector state(1);
   state[0] = initial;
   real_t time = 0.0;
   real_t step_size = dt;
   vector<real_t> trajectory(1, initial);
   for (StateId step = 0; step < steps; ++step)
   {
      solver->Step(state, time, step_size);
      trajectory.push_back(state[0]);
   }
   // For J = 0.5*u_N^2, the terminal adjoint is dJ/du_N = u_N.
   real_t lambda = trajectory.back();
   real_t gradient = 0.0;
   for (StateId from = steps; from > 0; --from)
   {
      const real_t predecessor = trajectory[static_cast<size_t>(from - 1)];
      const real_t successor = trajectory[static_cast<size_t>(from)];
      gradient += lambda * ParameterDerivative(
                     kind, predecessor, successor, parameter, dt);
      lambda *= StateDerivative(kind, predecessor, successor, parameter, dt);
   }
   return {trajectory.back(), gradient};
}

/// Application-owned discrete-adjoint update for one reconstructed transition.
/** The controller keeps @a predecessor synchronized to the live Vector. This
    handler separately retains the successor value needed by the derivative
    formulas and owns both adjoint variables (`lambda` and `gradient`). */
class DiscreteAdjointHandler : public ReverseStateHandler
{
private:
   const Vector &predecessor; ///< Live exact predecessor reconstructed by core.
   SolverKind kind;           ///< Discrete map being differentiated.
   real_t parameter;          ///< Parameter p.
   real_t dt;                 ///< Fixed step size h.
   real_t successor;          ///< Exact successor retained across callbacks.

public:
   real_t lambda;         ///< Current discrete adjoint.
   real_t gradient = 0.0; ///< Accumulated derivative dJ/dp.

   /// Initialize the terminal adjoint for J = 0.5*u_N^2.
   DiscreteAdjointHandler(const Vector &state, SolverKind kind_,
                          real_t parameter_, real_t dt_, real_t terminal)
      : predecessor(state), kind(kind_), parameter(parameter_), dt(dt_),
        successor(terminal), lambda(terminal) { }

   /// Apply the reverse recurrence for predecessor_id -> successor_id.
   void Apply(StateId predecessor_id, StateId successor_id) override
   {
      if (successor_id != predecessor_id + 1)
      {
         throw InvalidReverseState("non-adjacent discrete adjoint step");
      }
      const real_t current = predecessor[0];
      // Use lambda_(n+1) in the parameter contribution before updating it to
      // lambda_n. This example has no per-step objective contribution.
      gradient += lambda * ParameterDerivative(
                     kind, current, successor, parameter, dt);
      lambda *= StateDerivative(kind, current, successor, parameter, dt);
      successor = current;
   }
};

/// Run forward checkpointing followed by generic reverse reconstruction.
/** All three schedules use the same adapter, propagator, controller, and
    reverse handler. StoreEverything stores S0 through SN; Revolve and WMI use
    @a checkpoints as the maximum number of simultaneously stored primal
    checkpoints. The active state, retained successor, and moving window do
    not consume scheduler slots. */
ReferenceResult RunCheckpointed(SolverKind kind, const string &schedule_name,
                                StateId steps, size_t checkpoints,
                                real_t parameter, real_t dt, real_t initial)
{
   CubicOperator oper(parameter);
   unique_ptr<ODESolver> solver = MakeSolver(kind);
   solver->Init(oper);
   Vector state(1);
   state[0] = initial;
   TimePoint time{0, 0.0};
   real_t step_size = dt;
   CubicStateAdapter adapter(*solver, oper, state, time, step_size, kind);
   ODEStatePropagator propagator(*solver, state, time, step_size);
   MemoryCheckpointStorage storage;
   ExactCheckpointWindow window(2);
   CheckpointController controller(adapter, propagator, storage, window);
   unique_ptr<CheckpointSchedule> schedule;

   controller.Initialize();
   if (schedule_name == "store-all")
   {
      StoreEverythingSchedule *selected = new StoreEverythingSchedule;
      selected->Configure(steps, static_cast<size_t>(steps) + 1);
      schedule.reset(selected);
      controller.ExecuteForward(*schedule, steps);
   }
   else if (schedule_name == "revolve")
   {
      RevolveSchedule *selected = new RevolveSchedule;
      selected->Configure(steps, checkpoints);
      schedule.reset(selected);
      controller.ExecuteForward(*schedule, steps);
   }
   else
   {
      // WMI makes placement decisions one state at a time and learns the
      // terminal horizon only when ExecuteOnlineForward completes.
      WangMoinIaccarinoSchedule *selected =
         new WangMoinIaccarinoSchedule;
      selected->Configure(checkpoints);
      schedule.reset(selected);
      controller.ExecuteOnlineForward(*selected, steps);
   }

   // Preserve u_N before the schedule is allowed to restore an earlier state.
   // The handler initializes lambda_N from the same exact terminal value.
   const real_t terminal = state[0];
   DiscreteAdjointHandler handler(state, kind, parameter, dt, terminal);
   controller.BeginReverse();
   controller.ExecuteReverse(*schedule, handler);
   return {terminal, handler.gradient};
}

} // namespace

int main(int argc, char *argv[])
{
   const char *solver_option = "forward-euler";
   const char *schedule_option = "revolve";
   int num_steps = 20;
   int checkpoints = 3;
   bool visualization = false;
   OptionsParser args(argc, argv);
   args.AddOption(&solver_option, "-s", "--solver",
                  "Solver: forward-euler or backward-euler.");
   args.AddOption(&schedule_option, "-c", "--schedule",
                  "Schedule: store-all, revolve, or wmi.");
   args.AddOption(&num_steps, "-n", "--num-steps",
                  "Number of forward state transitions.");
   args.AddOption(&checkpoints, "-m", "--checkpoints",
                  "Maximum simultaneously stored checkpoints.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Accepted for test compatibility.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   const bool forward = std::strcmp(solver_option, "forward-euler") == 0;
   const bool backward = std::strcmp(solver_option, "backward-euler") == 0;
   const string schedule_name(schedule_option);
   if ((!forward && !backward) ||
       (schedule_name != "store-all" && schedule_name != "revolve" &&
        schedule_name != "wmi") || num_steps < 0 || checkpoints < 1)
   {
      cerr << "Invalid solver, schedule, step count, or checkpoint budget.\n";
      return 2;
   }

   try
   {
      const SolverKind kind = forward ? SolverKind::ForwardEuler :
                              SolverKind::BackwardEuler;
      const real_t parameter = 0.7;
      const real_t dt = 0.01;
      const real_t initial = 0.4;
      const ReferenceResult reference = ComputeReference(
                                           kind, num_steps, parameter, dt,
                                           initial);
      const ReferenceResult checkpointed = RunCheckpointed(
                                               kind, schedule_name, num_steps,
                                               static_cast<size_t>(checkpoints),
                                               parameter, dt, initial);
      const real_t terminal_error =
         std::abs(reference.terminal - checkpointed.terminal);
      const real_t gradient_error =
         std::abs(reference.gradient - checkpointed.gradient);
      cout << setprecision(numeric_limits<real_t>::max_digits10)
           << "terminal replay error = " << terminal_error << '\n'
           << "discrete gradient error = " << gradient_error << '\n';
      return terminal_error == 0.0 && gradient_error == 0.0 ? 0 : 3;
   }
   catch (const std::exception &error)
   {
      cerr << "Reverse/adjoint checkpoint miniapp failed: "
           << error.what() << '\n';
      return 4;
   }
}
