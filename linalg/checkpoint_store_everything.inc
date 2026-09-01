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

namespace checkpoint_detail
{

/// Reference offline schedule that stores every forward state.
///
/// Requires num_steps + 1 logical slots. It owns no snapshots and is not
/// internally synchronized.
class StoreEverythingScheduleImpl final : public OfflineCheckpointSchedule
{
public:
   /// Builds the deterministic store-everything trace.
   /// @throws std::invalid_argument for negative steps or insufficient slots.
   /// @throws std::length_error when size calculations exceed platform limits.
   void Configure(StateId num_steps, std::size_t num_checkpoints) override
   {
      const std::size_t step_count = checkedStepCount(num_steps);
      if (step_count == std::numeric_limits<std::size_t>::max())
      {
         throw std::length_error(
            "store-everything checkpoint-count calculation overflow");
      }
      const std::size_t required_checkpoints = step_count + 1;
      if (num_checkpoints < required_checkpoints)
      {
         throw std::invalid_argument(
            "store-everything schedule requires "
            + std::to_string(required_checkpoints) + " checkpoints for "
            + std::to_string(num_steps) + " steps, but only "
            + std::to_string(num_checkpoints) + " were provided");
      }
      if (step_count > (std::numeric_limits<std::size_t>::max() - 2) / 2)
      {
         throw std::length_error(
            "store-everything command-count calculation overflow");
      }

      const std::size_t command_count = 2 * step_count + 2;
      std::vector<CheckpointCommand> commands;
      if (command_count > commands.max_size())
      {
         throw std::length_error(
            "store-everything command sequence exceeds vector capacity");
      }
      commands.reserve(command_count);

      commands.push_back(storageCommand(CheckpointAction::Store, 0));
      for (StateId step = 0; step < num_steps; ++step)
      {
         commands.push_back(CheckpointCommand{CheckpointAction::Advance, step,
                                              step + 1, std::nullopt});
         commands.push_back(storageCommand(CheckpointAction::Store, step + 1));
      }

      commands.push_back(CheckpointCommand{CheckpointAction::Finished,
                                           num_steps, num_steps, std::nullopt});

      commands_ = std::move(commands);
      next_command_ = 0;
      configured_ = true;
   }

   /// Returns the next command, repeating Finished after exhaustion.
   /// @throws std::logic_error if configure() has not been called.
   CheckpointCommand Next() override
   {
      if (!configured_)
      {
         throw std::logic_error(
            "store-everything schedule must be configured before next()");
      }
      if (next_command_ < commands_.size())
      {
         return commands_[next_command_++];
      }
      return commands_.back();
   }

   /// Rewinds the configured trace.
   void Reset() noexcept override
   {
      if (configured_)
      {
         next_command_ = 0;
      }
   }

private:
   static std::size_t checkedStepCount(StateId num_steps)
   {
      if (num_steps < 0)
      {
         throw std::invalid_argument(
            "store-everything schedule requires a non-negative step count");
      }

      const std::uintmax_t unsigned_steps =
         static_cast<std::uintmax_t>(num_steps);
      if (unsigned_steps > static_cast<std::uintmax_t>(
             std::numeric_limits<std::size_t>::max()))
      {
         throw std::length_error(
            "store-everything step count exceeds platform capacity");
      }
      return static_cast<std::size_t>(unsigned_steps);
   }

   static CheckpointId checkpointId(StateId step) noexcept
   {
      return static_cast<CheckpointId>(step) + CheckpointId{1};
   }

   static CheckpointCommand storageCommand(CheckpointAction action,
                                           StateId step)
   {
      return CheckpointCommand{action, step, step, checkpointId(step)};
   }

   bool configured_{false};
   std::size_t next_command_{0};
   std::vector<CheckpointCommand> commands_;
};

} // namespace checkpoint_detail
