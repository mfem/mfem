// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

//                  MFEM Checkpointing Miniapp: Forward Euler
//
// Compile with: make checkpoint-forward-euler
//
// Sample runs:  checkpoint-forward-euler
//               checkpoint-forward-euler -s 40
//
// Description: This example demonstrates exact checkpoint/replay for an
//              ODESolver-based integration. It advances the scalar equation
//
//                 du/dt = alpha u - u^3
//
//              with Forward Euler, first normally and then through a
//              CheckpointController using StoreEverything scheduling.
//
//              To exercise replay rather than a direct restore, the example
//              discards every persistent checkpoint except the initial one,
//              clears the transient moving window, and reconstructs the
//              terminal step from checkpoint 1. It succeeds only when the
//              replayed and reference terminal states are exactly equal.

#include "mfem.hpp"

#include <cmath>
#include <iostream>

using namespace mfem;
using namespace std;

namespace
{

/// Scalar autonomous ODE du/dt = alpha u - u^3.
/** For alpha > 0 the solution approaches the stable equilibrium sqrt(alpha).
    The cubic term makes Forward Euler updates nonlinear, so an exact replay
    match is a meaningful check of deterministic propagation. The right-hand
    side does not depend on time. */
class CubicOperator : public TimeDependentOperator
{
private:
   real_t parameter; ///< Linear growth rate alpha.

public:
   /// Construct the one-component operator with growth rate @a parameter_.
   explicit CubicOperator(real_t parameter_)
      : TimeDependentOperator(1), parameter(parameter_) { }

   /// Evaluate the slope: rate = alpha u - u^3.
   void Mult(const Vector &state, Vector &rate) const override
   {
      rate.SetSize(1);
      rate[0] = parameter * state[0] - state[0] * state[0] * state[0];
   }
};

/// Exact state adapter for MFEM's fixed-step ForwardEulerSolver.
/** The library adapter captures and restores the Vector, TimePoint, and step
    size; this class adds the solver-side reinitialization that Forward Euler
    needs. The solver keeps no stage history, so no restart bytes are used. */
class ForwardEulerCheckpointAdapter : public ODEVectorCheckpointAdapter
{
private:
   ForwardEulerSolver &solver;   ///< Borrowed solver to reinitialize.
   TimeDependentOperator &oper;  ///< Borrowed time-dependent operator.

protected:
   /// Reinitialize the borrowed solver against the restored physical time.
   void OnRestored() override
   {
      oper.SetTime(time.time);
      solver.Init(oper);
   }

public:
   /// Borrow ODE state, solver, and operator for the adapter lifetime.
   ForwardEulerCheckpointAdapter(ForwardEulerSolver &solver_,
                                 TimeDependentOperator &oper_, Vector &state_,
                                 TimePoint &time_, real_t &dt_)
      : ODEVectorCheckpointAdapter(state_, time_, dt_), solver(solver_),
        oper(oper_) { }
};

} // namespace

/** Integrate a reference trajectory, run the same integration through a
    CheckpointController, and rebuild the terminal state by replaying every
    step from the initial checkpoint. The replay must match the reference
    exactly.

    Exit codes: 0 on success, 1 for unparsable options, 2 for a non-positive
    step count, 3 when the replayed state differs from the reference, and 4
    when a checkpoint operation throws. */
int main(int argc, char *argv[])
{
   // 1. Parse command-line options. The visualization option is accepted for
   //    compatibility with MFEM's example test harness; this scalar example
   //    produces terminal output only.
   int steps = 20;
   bool visualization = false;
   OptionsParser args(argc, argv);
   args.AddOption(&steps, "-s", "--steps", "Number of Forward Euler steps.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Accepted for consistency; this example has no "
                  "visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   if (steps < 1)
   {
      cerr << "The number of steps must be positive.\n";
      return 2;
   }

   try
   {
      // 2. Define the scalar problem, fixed step size, and initial condition:
      //    alpha = 0.7, dt = 0.01, and u(0) = 0.4.
      const real_t parameter = 0.7;
      const real_t dt = 0.01;
      Vector initial(1);
      initial[0] = 0.4;

      // 3. Compute an ordinary Forward Euler trajectory as the exact reference.
      //    It shares no objects with the checkpointed run.
      CubicOperator reference_operator(parameter);
      ForwardEulerSolver reference_solver;
      reference_solver.Init(reference_operator);
      Vector reference_state(initial);
      real_t reference_time = 0.0;
      real_t reference_dt = dt;
      for (int step = 0; step < steps; step++)
      {
         reference_solver.Step(reference_state, reference_time, reference_dt);
      }

      // 4. Assemble the checkpoint/replay services. The ODE adapter binds the
      //    generic state-centric core to this externally owned continuation.
      //    The adapter and propagator borrow the same solver, solution, time,
      //    and step size. Storage keeps persistent checkpoints in memory; the
      //    window caches the two most recent states.
      CubicOperator checkpoint_operator(parameter);
      ForwardEulerSolver checkpoint_solver;
      Vector checkpoint_state(initial);
      TimePoint checkpoint_time{0, 0.0};
      real_t checkpoint_dt = dt;
      ForwardEulerCheckpointAdapter adapter(checkpoint_solver,
                                            checkpoint_operator,
                                            checkpoint_state,
                                            checkpoint_time, checkpoint_dt);
      ODEStatePropagator propagator(checkpoint_solver, checkpoint_state,
                                    checkpoint_time, checkpoint_dt);
      MemoryCheckpointStorage storage;
      ExactCheckpointWindow window(2);
      CheckpointController controller(adapter, propagator, storage, window);
      // 5. StoreEverything assigns checkpoint ID step + 1 to every state from
      //    the initial state through the terminal state, so it needs
      //    steps + 1 slots. Initialize() captures state 0; ExecuteForward()
      //    then advances one step at a time and stores every state.
      StoreEverythingSchedule schedule;
      schedule.Configure(steps, static_cast<size_t>(steps) + 1);

      controller.Initialize();
      controller.ExecuteForward(schedule, steps);

      // 6. Retain only the initial persistent checkpoint and clear transient
      //    replay state, forcing RestoreStep() to replay the full trajectory.
      //    Restore(1) loads state 0 into the application; RestoreStep() then
      //    replays all steps from it.
      const CheckpointId last_id = static_cast<CheckpointId>(steps) + 1;
      for (CheckpointId id = 2; id <= last_id; id++)
      {
         controller.Discard(id);
      }
      window.Clear();
      controller.Restore(1);
      controller.RestoreStep(steps);

      // 7. Exact deterministic replay must reproduce the reference exactly:
      //    any nonzero difference fails.
      const real_t replay_error =
         std::abs(checkpoint_state[0] - reference_state[0]);
      cout << "terminal replay error: " << replay_error << '\n';

      return replay_error == 0.0 ? 0 : 3;
   }
   catch (const std::exception &error)
   {
      // Checkpoint failures raise CheckpointError subclasses; report any
      // exception and exit without a verdict.
      cerr << "Checkpoint failure: " << error.what() << '\n';
      return 4;
   }
}
