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
    Integrate the stiff diagonal system u_i' = -lambda_i u_i, with rates 1 and
    50 and initial values (1, 1), using BackwardEulerSolver and a closed-form
    implicit solve. Restore an interior checkpoint and replay to the terminal
    step through direct schedule dispatch. Compare every continuation field
    exactly with an independent reference. Operator rates are saved and checked
    before solver reinitialization. All four storage modes are available; file
    modes also demonstrate clean close and replay into fresh objects.

    Example: checkpoint-backward-euler --storage file-snapshots -s 12 -r 4 */

#include "euler_checkpoint.hpp"

int main(int argc, char *argv[])
{
   return mfem::checkpoint_demo::RunEulerExample(argc, argv, true);
}
