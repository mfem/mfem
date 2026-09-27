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

#ifndef MFEM_CHECKPOINT_DEMO_FORWARD_EULER
#define MFEM_CHECKPOINT_DEMO_FORWARD_EULER

#include "mfem.hpp"

namespace mfem
{
namespace checkpoint_demo
{

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

} // namespace checkpoint_demo
} // namespace mfem

#endif // MFEM_CHECKPOINT_DEMO_FORWARD_EULER
