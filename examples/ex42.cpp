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
//               ex42 -s 40 -c 4
//
// Demonstrates exact checkpoint/replay of Forward Euler and a discrete reverse
// update driven by the Revolve schedule. The full-storage and checkpointed
// gradients are compared at the end.

#include "mfem.hpp"

#include <cmath>
#include <iostream>
#include <vector>

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

class CubicReverseStep : public ReverseStepHandler
{
private:
   real_t parameter;
   real_t dt;

public:
   real_t adjoint;
   real_t gradient = 0.0;

   CubicReverseStep(real_t parameter_, real_t dt_, real_t terminal)
      : parameter(parameter_), dt(dt_), adjoint(terminal) { }

   void ReverseStep(StepId, StepId, const Vector &predecessor,
                    const Vector &) override
   {
      const real_t state = predecessor[0];
      gradient += dt * state * adjoint;
      adjoint *= 1.0 + dt * (parameter - 3.0 * state * state);
   }
};

} // namespace

int main(int argc, char *argv[])
{
   int steps = 20;
   int checkpoint_count = 3;
   OptionsParser args(argc, argv);
   args.AddOption(&steps, "-s", "--steps", "Number of Forward Euler steps.");
   args.AddOption(&checkpoint_count, "-c", "--checkpoints",
                  "Maximum number of stored primal checkpoints.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   if (steps < 1 || checkpoint_count < 1)
   {
      cerr << "Steps and checkpoints must both be positive.\n";
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
   vector<real_t> trajectory(static_cast<size_t>(steps) + 1);
   trajectory[0] = initial[0];
   for (int step = 0; step < steps; step++)
   {
      reference_solver.Step(reference_state, reference_time, reference_dt);
      trajectory[static_cast<size_t>(step) + 1] = reference_state[0];
   }
   real_t reference_adjoint = reference_state[0];
   real_t reference_gradient = 0.0;
   for (int step = steps - 1; step >= 0; step--)
   {
      const real_t state = trajectory[static_cast<size_t>(step)];
      reference_gradient += dt * state * reference_adjoint;
      reference_adjoint *=
         1.0 + dt * (parameter - 3.0 * state * state);
   }

   CubicOperator checkpoint_operator(parameter);
   ForwardEulerSolver checkpoint_solver;
   ForwardEulerCheckpointAdapter adapter(checkpoint_solver,
                                          checkpoint_operator);
   ODECheckpointPropagator propagator(checkpoint_solver);
   MemoryCheckpointStorage storage;
   ExactCheckpointWindow window(2);
   CheckpointController controller(adapter, propagator, storage, window);
   RevolveSchedule schedule;
   schedule.Configure(steps, static_cast<size_t>(checkpoint_count));

   controller.Initialize(initial, 0.0, dt);
   controller.ExecuteForward(schedule, steps);
   const real_t checkpoint_terminal = controller.ActiveState().state[0];
   controller.BeginReverse();
   CubicReverseStep reverse(parameter, dt, checkpoint_terminal);
   controller.ExecuteReverse(schedule, reverse);

   const real_t state_error = std::abs(checkpoint_terminal - reference_state[0]);
   const real_t gradient_error =
      std::abs(reverse.gradient - reference_gradient);
   cout << "terminal state error: " << state_error << '\n'
        << "discrete gradient error: " << gradient_error << '\n';

   return state_error == 0.0 && gradient_error == 0.0 ? 0 : 3;
}
