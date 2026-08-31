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
// Demonstrates exact checkpoint/replay of Forward Euler. A terminal state
// reconstructed from the initial checkpoint is compared with a normal solve.

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

   const real_t parameter = 0.7;
   const real_t dt = 0.01;
   Vector initial(1);
   initial[0] = 0.4;

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

   CubicOperator checkpoint_operator(parameter);
   ForwardEulerSolver checkpoint_solver;
   ForwardEulerCheckpointAdapter adapter(checkpoint_solver,
                                          checkpoint_operator);
   ODECheckpointPropagator propagator(checkpoint_solver);
   MemoryCheckpointStorage storage;
   ExactCheckpointWindow window(2);
   CheckpointController controller(adapter, propagator, storage, window);
   StoreEverythingSchedule schedule;
   schedule.Configure(steps, static_cast<size_t>(steps) + 1);

   controller.Initialize(initial, 0.0, dt);
   controller.ExecuteForward(schedule, steps);

   // Retain only the initial persistent checkpoint and clear transient replay
   // state, forcing RestoreStep() to replay the complete trajectory.
   for (CheckpointId id = 2; id <= static_cast<CheckpointId>(steps) + 1; id++)
   {
      controller.Discard(id);
   }
   window.Clear();
   controller.Restore(1);
   controller.RestoreStep(steps);

   const real_t replay_error =
      std::abs(controller.ActiveState().state[0] - reference_state[0]);
   cout << "terminal replay error: " << replay_error << '\n';

   return replay_error == 0.0 ? 0 : 3;
}
