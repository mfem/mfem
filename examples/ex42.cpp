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

//                                MFEM Example 42
//
// Compile with: make ex42
//
// Sample runs:  ex42
//               ex42 -s 40
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
//              replayed and reference terminal states are bitwise identical.

#include "mfem.hpp"

#include <cmath>
#include <iostream>

using namespace mfem;
using namespace std;

namespace
{

class CubicOperator : public TimeDependentOperator
{
private:
   real_t parameter;

public:
   explicit CubicOperator(real_t parameter_)
      : TimeDependentOperator(1), parameter(parameter_) { }

   void Mult(const Vector &state, Vector &rate) const override
   {
      rate.SetSize(1);
      rate[0] = parameter * state[0] - state[0] * state[0] * state[0];
   }
};

} // namespace

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

   // 2. Define the scalar problem, fixed step size, and initial condition.
   const real_t parameter = 0.7;
   const real_t dt = 0.01;
   Vector initial(1);
   initial[0] = 0.4;

   // 3. Compute an ordinary Forward Euler trajectory as the exact reference.
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
   //    the initial state through the terminal state.
   StoreEverythingSchedule schedule;
   schedule.Configure(steps, static_cast<size_t>(steps) + 1);

   controller.Initialize();
   controller.ExecuteForward(schedule, steps);

   // 6. Retain only the initial persistent checkpoint and clear transient
   //    replay state, forcing RestoreStep() to replay the complete trajectory.
   for (CheckpointId id = 2; id <= static_cast<CheckpointId>(steps) + 1; id++)
   {
      controller.Discard(id);
   }
   window.Clear();
   controller.Restore(1);
   controller.RestoreStep(steps);

   // 7. Exact deterministic replay must reproduce the reference bit for bit.
   const real_t replay_error =
      std::abs(checkpoint_state[0] - reference_state[0]);
   cout << "terminal replay error: " << replay_error << '\n';

   return replay_error == 0.0 ? 0 : 3;
}
