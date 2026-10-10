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

#include "checkpoint.hpp"
#include "../general/error.hpp"

#include <limits>

namespace mfem
{

void StoreEverythingSchedule::Configure(StateId num_steps,
                                        std::size_t num_checkpoints)
{
   MFEM_VERIFY(num_steps >= 0, "Checkpoint horizon must be non-negative.");
   const std::uintmax_t required = static_cast<std::uintmax_t>(num_steps) + 1;
   MFEM_VERIFY(required <= std::numeric_limits<std::size_t>::max(),
               "Checkpoint count exceeds platform capacity.");
   MFEM_VERIFY(num_checkpoints >= static_cast<std::size_t>(required),
               "StoreEverythingSchedule requires " << required
               << " checkpoints, but only " << num_checkpoints
               << " were provided.");
   terminal = num_steps;
   configured = true;
   Reset();
}

CheckpointCommand StoreEverythingSchedule::Next()
{
   MFEM_VERIFY(configured, "Configure the checkpoint schedule before Next().");
   if (store_pending)
   {
      store_pending = false;
      return {CheckpointAction::Store, current, current,
              static_cast<CheckpointId>(current) + CheckpointId{1}};
   }
   if (current < terminal)
   {
      const StateId from = current++;
      store_pending = true;
      return {CheckpointAction::Advance, from, current, std::nullopt};
   }
   return {CheckpointAction::Finished, terminal, terminal, std::nullopt};
}

void StoreEverythingSchedule::Reset()
{
   current = 0;
   store_pending = true;
}

IntervalSchedule::IntervalSchedule(StateId terminal_, StateId interval_)
   : terminal(terminal_), interval(interval_)
{
   MFEM_VERIFY(terminal > 0 && interval > 0,
               "Checkpoint horizon and interval must be positive.");
}

CheckpointCommand IntervalSchedule::Next()
{
   if (store_pending)
   {
      store_pending = false;
      return {CheckpointAction::Store, current, current,
              static_cast<CheckpointId>(current) + CheckpointId{1}};
   }
   if (current < terminal)
   {
      const StateId from = current++;
      store_pending = current < terminal && current % interval == 0;
      return {CheckpointAction::Advance, from, current, std::nullopt};
   }
   return {CheckpointAction::Finished, terminal, terminal, std::nullopt};
}

void IntervalSchedule::Reset()
{
   current = 0;
   store_pending = true;
}

} // namespace mfem
