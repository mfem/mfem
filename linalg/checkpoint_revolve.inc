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

/// Canonical binomial offline Revolve schedule.
///
/// The checkpoint budget is the number of simultaneously stored primal
/// snapshots: stored u_0 consumes one slot and active u_N consumes none. Before
/// Reverse(n+1,n), the emitted trace makes u_n active. The object owns only
/// logical schedule state and is not internally synchronized.
class RevolveScheduleImpl final : public OfflineCheckpointSchedule
{
public:
   /// Configures a deterministic trace for a known horizon.
   /// @throws std::invalid_argument for a negative horizon or zero positive-
   /// horizon budget.
   /// @throws std::length_error for unrepresentable sizes.
   void Configure(StepId num_steps, std::size_t num_checkpoints) override
   {
      if (num_steps < 0)
      {
         throw std::invalid_argument(
            "Revolve schedule requires a non-negative step count");
      }
      if (num_steps > 0 && num_checkpoints == 0)
      {
         throw std::invalid_argument("Revolve schedule requires at least one "
                                     "checkpoint for a positive step count");
      }

      const std::uintmax_t unsigned_steps =
         static_cast<std::uintmax_t>(num_steps);
      if (unsigned_steps > static_cast<std::uintmax_t>(
             std::numeric_limits<std::size_t>::max()))
      {
         throw std::length_error(
            "Revolve step count exceeds platform capacity");
      }

      num_steps_ = num_steps;
      checkpoint_budget_ =
         num_steps == 0 ? std::size_t{0}
                        : std::min(num_checkpoints,
                                   static_cast<std::size_t>(unsigned_steps));
      configured_ = true;
      initializeState();
   }

   /// Advances the schedule state and returns its next command.
   /// @throws std::logic_error if unconfigured or an invariant is violated.
   CheckpointCommand Next() override
   {
      if (!configured_)
      {
         throw std::logic_error(
            "Revolve schedule must be configured before next()");
      }
      if (finished_)
      {
         return finishedCommand();
      }
      if (num_steps_ == 0)
      {
         finished_ = true;
         return finishedCommand();
      }

      if (!started_)
      {
         started_ = true;
         base_checkpoint_ = allocateCheckpoint(0);
         frames_.push_back(Frame{Mode::Initial, Stage::Enter, 0, num_steps_,
                                 checkpoint_budget_, base_checkpoint_});
         return storeCommand(0, base_checkpoint_);
      }

      while (!frames_.empty())
      {
         const std::optional<CheckpointCommand> command = advanceFrame();
         if (command.has_value())
         {
            return *command;
         }
      }

      if (!base_discarded_)
      {
         base_discarded_ = true;
         return discardCommand(base_checkpoint_, 0);
      }

      if (!live_checkpoints_.empty())
      {
         throw std::logic_error(
            "Revolve schedule completed with live checkpoints");
      }
      finished_ = true;
      return finishedCommand();
   }

   /// Rewinds the configured schedule to its initial state.
   void Reset() override
   {
      if (configured_)
      {
         initializeState();
      }
   }

private:
   enum class Mode
   {
      Initial,
      Replay
   };

   enum class Stage
   {
      Enter,
      LinearRestore,
      LinearAdvance,
      LinearReverse,
      ReplayAdvance,
      StoreSplit,
      LaunchSuffix,
      DiscardSplit,
      LaunchPrefix,
      Done
   };

   struct Frame
   {
      Mode mode{Mode::Initial};
      Stage stage{Stage::Enter};
      StepId start{0};
      StepId length{0};
      std::size_t checkpoints{0};
      CheckpointId base{0};
      StepId split_offset{0};
      StepId target{0};
      CheckpointId split_checkpoint{0};
   };

   struct LiveCheckpoint
   {
      CheckpointId id{0};
      StepId step{0};
   };

   void initializeState()
   {
      started_ = false;
      base_discarded_ = false;
      finished_ = false;
      current_step_ = 0;
      next_checkpoint_id_ = CheckpointId{1};
      base_checkpoint_ = CheckpointId{0};
      frames_.clear();
      live_checkpoints_.clear();
   }

   std::optional<CheckpointCommand> advanceFrame()
   {
      Frame &frame = frames_.back();
      switch (frame.stage)
      {
         case Stage::Enter:
            return enterFrame(frame);

         case Stage::LinearRestore:
            frame.stage = Stage::LinearAdvance;
            return restoreCommand(frame.base, frame.start);

         case Stage::LinearAdvance:
            frame.stage = Stage::LinearReverse;
            if (frame.target > frame.start)
            {
               return advanceCommand(frame.start, frame.target);
            }
            return std::nullopt;

         case Stage::LinearReverse:
         {
            const StepId target = frame.target;
            if (target == frame.start)
            {
               frame.stage = Stage::Done;
            }
            else
            {
               --frame.target;
               frame.stage = Stage::LinearRestore;
            }
            return reverseCommand(target + StepId{1}, target);
         }

         case Stage::ReplayAdvance:
            frame.stage = Stage::StoreSplit;
            return advanceCommand(frame.start,
                                  frame.start + frame.split_offset);

         case Stage::StoreSplit:
         {
            const StepId split_step = frame.start + frame.split_offset;
            frame.split_checkpoint = allocateCheckpoint(split_step);
            frame.stage = Stage::LaunchSuffix;
            return storeCommand(split_step, frame.split_checkpoint);
         }

         case Stage::LaunchSuffix:
         {
            const Frame child{frame.mode,
                              Stage::Enter,
                              frame.start + frame.split_offset,
                              frame.length - frame.split_offset,
                              frame.checkpoints - 1,
                              frame.split_checkpoint};
            frame.stage = Stage::DiscardSplit;
            frames_.push_back(child);
            return std::nullopt;
         }

         case Stage::DiscardSplit:
         {
            const StepId split_step = frame.start + frame.split_offset;
            frame.stage = Stage::LaunchPrefix;
            return discardCommand(frame.split_checkpoint, split_step);
         }

         case Stage::LaunchPrefix:
         {
            const Frame child{Mode::Replay,      Stage::Enter,
                              frame.start,       frame.split_offset,
                              frame.checkpoints, frame.base};
            frame.stage = Stage::Done;
            frames_.push_back(child);
            return std::nullopt;
         }

         case Stage::Done:
            frames_.pop_back();
            return std::nullopt;
      }

      throw std::logic_error("invalid Revolve frame state");
   }

   std::optional<CheckpointCommand> enterFrame(Frame &frame)
   {
      if (frame.length <= 0 || frame.checkpoints == 0)
      {
         throw std::logic_error("invalid Revolve interval state");
      }

      if (frame.length == 1 || frame.checkpoints == 1)
      {
         frame.target = frame.start + frame.length - StepId{1};
         frame.stage = Stage::LinearRestore;
         if (frame.mode == Mode::Initial)
         {
            return advanceCommand(frame.start, frame.start + frame.length);
         }
         return std::nullopt;
      }

      const std::uint64_t length = static_cast<std::uint64_t>(frame.length);
      const std::uint64_t checkpoints =
         static_cast<std::uint64_t>(frame.checkpoints);
      const std::uint64_t split = frame.mode == Mode::Initial
                                     ? initialSplit(length, checkpoints)
                                     : replaySplit(length, checkpoints);
      frame.split_offset = static_cast<StepId>(split);

      if (frame.mode == Mode::Initial)
      {
         frame.stage = Stage::StoreSplit;
         return advanceCommand(frame.start, frame.start + frame.split_offset);
      }

      frame.stage = Stage::ReplayAdvance;
      return restoreCommand(frame.base, frame.start);
   }

   static std::uint64_t initialSplit(std::uint64_t length,
                                     std::uint64_t checkpoints)
   {
      // The discrete slope of the canonical replay cost is its binomial
      // repetition level.  The first crossing below chooses the earliest
      // optimal split and makes tie-breaking deterministic.
      std::uint64_t low = 1;
      std::uint64_t high = length - 1;
      while (low < high)
      {
         const std::uint64_t middle = low + (high - low) / 2;
         const std::uint64_t right_slope = replayLevel(middle + 1, checkpoints);
         const std::uint64_t left_slope =
            initialLevel(length - middle, checkpoints - 1);
         if (right_slope >= left_slope)
         {
            high = middle;
         }
         else
         {
            low = middle + 1;
         }
      }
      return low;
   }

   static std::uint64_t replaySplit(std::uint64_t length,
                                    std::uint64_t checkpoints)
   {
      std::uint64_t low = 1;
      std::uint64_t high = length - 1;
      while (low < high)
      {
         const std::uint64_t middle = low + (high - low) / 2;
         const std::uint64_t right_slope = replayLevel(middle + 1, checkpoints);
         const std::uint64_t left_slope =
            replayLevel(length - middle, checkpoints - 1);
         if (right_slope == std::numeric_limits<std::uint64_t>::max()
             || right_slope + std::uint64_t{1} >= left_slope)
         {
            high = middle;
         }
         else
         {
            low = middle + 1;
         }
      }
      return low;
   }

   static std::uint64_t replayLevel(std::uint64_t length,
                                    std::uint64_t checkpoints)
   {
      if (length <= 1)
      {
         return 0;
      }
      return minimumLevel(length, checkpoints, false);
   }

   static std::uint64_t initialLevel(std::uint64_t length,
                                     std::uint64_t checkpoints)
   {
      if (length <= checkpoints)
      {
         return 0;
      }
      return minimumLevel(length + std::uint64_t{1}, checkpoints, true);
   }

   static std::uint64_t minimumLevel(std::uint64_t target,
                                     std::uint64_t checkpoints, bool initial)
   {
      if (checkpoints == 0)
      {
         throw std::logic_error("invalid zero-checkpoint binomial state");
      }

      std::uint64_t low = 0;
      std::uint64_t high = target - 1;
      while (low < high)
      {
         const std::uint64_t middle = low + (high - low) / 2;
         std::uint64_t total = checkedAdd(checkpoints, middle);
         if (initial)
         {
            total = checkedAdd(total, std::uint64_t{1});
         }
         if (binomialAtLeast(total, checkpoints, target))
         {
            high = middle;
         }
         else
         {
            low = middle + 1;
         }
      }
      return low;
   }

   static std::uint64_t checkedAdd(std::uint64_t left, std::uint64_t right)
   {
      if (right > std::numeric_limits<std::uint64_t>::max() - left)
      {
         throw std::length_error("Revolve binomial-level calculation overflow");
      }
      return left + right;
   }

   static bool binomialAtLeast(std::uint64_t total, std::uint64_t choose,
                               std::uint64_t target)
   {
      if (choose > total)
      {
         return false;
      }
      choose = std::min(choose, total - choose);
      std::uint64_t value = 1;
      for (std::uint64_t index = 1; index <= choose; ++index)
      {
         std::uint64_t numerator = total - choose + index;
         std::uint64_t denominator = index;

         const std::uint64_t first_gcd = std::gcd(numerator, denominator);
         numerator /= first_gcd;
         denominator /= first_gcd;

         const std::uint64_t second_gcd = std::gcd(value, denominator);
         value /= second_gcd;
         denominator /= second_gcd;
         if (denominator != 1)
         {
            throw std::logic_error("non-integral Revolve binomial recurrence");
         }

         const std::uint64_t quotient = target / numerator;
         const std::uint64_t remainder = target % numerator;
         const std::uint64_t threshold = quotient + (remainder == 0 ? 0 : 1);
         if (value >= threshold)
         {
            return true;
         }
         value *= numerator;
      }
      return value >= target;
   }

   CheckpointId allocateCheckpoint(StepId step)
   {
      if (live_checkpoints_.size() >= checkpoint_budget_)
      {
         throw std::logic_error("Revolve checkpoint budget exceeded at step "
                                + std::to_string(step));
      }
      if (next_checkpoint_id_ == CheckpointId{0})
      {
         throw std::length_error("Revolve checkpoint-ID overflow");
      }
      const CheckpointId id = next_checkpoint_id_;
      if (next_checkpoint_id_ == std::numeric_limits<CheckpointId>::max())
      {
         next_checkpoint_id_ = CheckpointId{0};
      }
      else
      {
         ++next_checkpoint_id_;
      }
      live_checkpoints_.push_back(LiveCheckpoint{id, step});
      return id;
   }

   std::vector<LiveCheckpoint>::iterator findCheckpoint(CheckpointId id)
   {
      return std::find_if(live_checkpoints_.begin(), live_checkpoints_.end(),
                          [id](const LiveCheckpoint &checkpoint)
                          { return checkpoint.id == id; });
   }

   CheckpointCommand advanceCommand(StepId from, StepId to)
   {
      if (current_step_ != from || to <= from || to > num_steps_)
      {
         throw std::logic_error("invalid Revolve Advance state");
      }
      current_step_ = to;
      return CheckpointCommand{CheckpointAction::Advance, from, to,
                               std::nullopt};
   }

   CheckpointCommand storeCommand(StepId step, CheckpointId id) const
   {
      if (current_step_ != step)
      {
         throw std::logic_error("invalid Revolve Store state");
      }
      return CheckpointCommand{CheckpointAction::Store, step, step, id};
   }

   CheckpointCommand restoreCommand(CheckpointId id, StepId step)
   {
      const auto found = findCheckpoint(id);
      if (found == live_checkpoints_.end() || found->step != step)
      {
         throw std::logic_error(
            "Revolve attempted to restore a stale checkpoint ID");
      }
      current_step_ = step;
      return CheckpointCommand{CheckpointAction::Restore, step, step, id};
   }

   CheckpointCommand reverseCommand(StepId from, StepId to) const
   {
      if (from != to + StepId{1} || current_step_ != to)
      {
         throw std::logic_error("invalid Revolve Reverse state");
      }
      return CheckpointCommand{CheckpointAction::Reverse, from, to,
                               std::nullopt};
   }

   CheckpointCommand discardCommand(CheckpointId id, StepId step)
   {
      const auto found = findCheckpoint(id);
      if (found == live_checkpoints_.end() || found->step != step)
      {
         throw std::logic_error(
            "Revolve attempted to discard a stale checkpoint ID");
      }
      live_checkpoints_.erase(found);
      return CheckpointCommand{CheckpointAction::Discard, step, step, id};
   }

   CheckpointCommand finishedCommand() const
   {
      return CheckpointCommand{CheckpointAction::Finished, num_steps_,
                               num_steps_, std::nullopt};
   }

   StepId num_steps_{0};
   std::size_t checkpoint_budget_{0};
   bool configured_{false};
   bool started_{false};
   bool base_discarded_{false};
   bool finished_{false};
   StepId current_step_{0};
   CheckpointId next_checkpoint_id_{1};
   CheckpointId base_checkpoint_{0};
   std::vector<Frame> frames_;
   std::vector<LiveCheckpoint> live_checkpoints_;
};

} // namespace checkpoint_detail
