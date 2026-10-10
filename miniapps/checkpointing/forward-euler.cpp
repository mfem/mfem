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

/** @file
    Integrate u' = 0.7 u - u^3 with ForwardEulerSolver from u(0) = 0.4.
    Save complete state through direct CheckpointSchedule dispatch, erase all
    records except a selected origin, invalidate the live state, and replay.
    Compare solution, logical step, time, and timestep with an independent
    reference using exact equality. Select memory/file blocks or separate
    memory/file records with --storage. File modes also close and reopen
    against fresh application objects.

    Example: checkpoint-forward-euler --storage file-block -s 20 */

#include "euler_checkpoint.hpp"

int main(int argc, char *argv[])
{
   return mfem::checkpoint_demo::RunEulerExample(argc, argv, false);
}
