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

/// Canonical lazy binomial Revolve schedule implementation.
class RevolveScheduleImpl final : public OfflineCheckpointSchedule
{
public:
   void Configure(StateId num_steps, std::size_t num_checkpoints) override
   {
      if (num_steps < 0)
      {
         throw std::invalid_argument(
            "Revolve schedule requires a non-negative step count");
      }
      if (num_steps > 0 && num_checkpoints == 0)
      {
         throw std::invalid_argument(
            "Revolve schedule requires at least one checkpoint");
      }
      const std::uintmax_t unsigned_steps =
         static_cast<std::uintmax_t>(num_steps);
      if (unsigned_steps > static_cast<std::uintmax_t>(
             std::numeric_limits<std::size_t>::max()))
      {
         throw std::length_error("Revolve step count exceeds platform capacity");
      }
      num_steps_ = num_steps;
      checkpoint_budget_ = num_steps == 0 ? std::size_t{0} :
                           std::min(num_checkpoints,
                                    static_cast<std::size_t>(unsigned_steps));
      configured_ = true;
      InitializeState();
   }

   CheckpointCommand Next() override
   {
      if (!configured_)
      {
         throw std::logic_error(
            "Revolve schedule must be configured before Next()");
      }
      if (finished_) { return FinishedCommand(); }
      if (num_steps_ == 0)
      {
         finished_ = true;
         return FinishedCommand();
      }
      if (!started_)
      {
         started_ = true;
         base_checkpoint_ = AllocateCheckpoint(0);
         frames_.push_back(Frame{Mode::Initial, Stage::Enter, 0, num_steps_,
                                 checkpoint_budget_, base_checkpoint_, 0, 0,
                                 CheckpointId{0}});
         return StoreCommand(0, base_checkpoint_);
      }
      while (!frames_.empty())
      {
         const std::optional<CheckpointCommand> command = AdvanceFrame();
         if (command) { return *command; }
      }
      if (!base_discarded_)
      {
         base_discarded_ = true;
         return DiscardCommand(base_checkpoint_, 0);
      }
      if (!live_checkpoints_.empty())
      {
         throw std::logic_error(
            "Revolve schedule completed with live checkpoints");
      }
      finished_ = true;
      return FinishedCommand();
   }

   void Reset() override
   {
      if (configured_) { InitializeState(); }
   }

private:
   enum class Mode { Initial, Replay };
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
      Mode mode;
      Stage stage;
      StateId start;
      StateId length;
      std::size_t checkpoints;
      CheckpointId base;
      StateId split_offset;
      StateId target;
      CheckpointId split_checkpoint;
   };

   struct LiveCheckpoint
   {
      CheckpointId id;
      StateId state;
   };

   void InitializeState()
   {
      started_ = false;
      base_discarded_ = false;
      finished_ = false;
      current_state_ = 0;
      next_checkpoint_id_ = CheckpointId{1};
      base_checkpoint_ = CheckpointId{0};
      frames_.clear();
      live_checkpoints_.clear();
   }

   std::optional<CheckpointCommand> AdvanceFrame()
   {
      Frame &frame = frames_.back();
      switch (frame.stage)
      {
         case Stage::Enter:
            return EnterFrame(frame);
         case Stage::LinearRestore:
            frame.stage = Stage::LinearAdvance;
            return RestoreCommand(frame.base, frame.start);
         case Stage::LinearAdvance:
            frame.stage = Stage::LinearReverse;
            if (frame.target > frame.start)
            {
               return AdvanceCommand(frame.start, frame.target);
            }
            return std::nullopt;
         case Stage::LinearReverse:
         {
            const StateId target = frame.target;
            if (target == frame.start) { frame.stage = Stage::Done; }
            else
            {
               --frame.target;
               frame.stage = Stage::LinearRestore;
            }
            return ReverseCommand(target + StateId{1}, target);
         }
         case Stage::ReplayAdvance:
            frame.stage = Stage::StoreSplit;
            return AdvanceCommand(frame.start,
                                  frame.start + frame.split_offset);
         case Stage::StoreSplit:
         {
            const StateId split_state = frame.start + frame.split_offset;
            frame.split_checkpoint = AllocateCheckpoint(split_state);
            frame.stage = Stage::LaunchSuffix;
            return StoreCommand(split_state, frame.split_checkpoint);
         }
         case Stage::LaunchSuffix:
         {
            const Frame child{frame.mode, Stage::Enter,
                              frame.start + frame.split_offset,
                              frame.length - frame.split_offset,
                              frame.checkpoints - 1, frame.split_checkpoint,
                              0, 0, CheckpointId{0}};
            frame.stage = Stage::DiscardSplit;
            frames_.push_back(child);
            return std::nullopt;
         }
         case Stage::DiscardSplit:
         {
            const StateId split_state = frame.start + frame.split_offset;
            frame.stage = Stage::LaunchPrefix;
            return DiscardCommand(frame.split_checkpoint, split_state);
         }
         case Stage::LaunchPrefix:
         {
            const Frame child{Mode::Replay, Stage::Enter, frame.start,
                              frame.split_offset, frame.checkpoints, frame.base,
                              0, 0, CheckpointId{0}};
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

   std::optional<CheckpointCommand> EnterFrame(Frame &frame)
   {
      if (frame.length <= 0 || frame.checkpoints == 0)
      {
         throw std::logic_error("invalid Revolve interval state");
      }
      if (frame.length == 1 || frame.checkpoints == 1)
      {
         frame.target = frame.start + frame.length - StateId{1};
         frame.stage = Stage::LinearRestore;
         if (frame.mode == Mode::Initial)
         {
            return AdvanceCommand(frame.start, frame.start + frame.length);
         }
         return std::nullopt;
      }

      const std::uint64_t length = static_cast<std::uint64_t>(frame.length);
      const std::uint64_t checkpoints =
         static_cast<std::uint64_t>(frame.checkpoints);
      const std::uint64_t split = frame.mode == Mode::Initial ?
                                  InitialSplit(length, checkpoints) :
                                  ReplaySplit(length, checkpoints);
      frame.split_offset = static_cast<StateId>(split);
      if (frame.mode == Mode::Initial)
      {
         frame.stage = Stage::StoreSplit;
         return AdvanceCommand(frame.start,
                               frame.start + frame.split_offset);
      }
      frame.stage = Stage::ReplayAdvance;
      return RestoreCommand(frame.base, frame.start);
   }

   static std::uint64_t InitialSplit(std::uint64_t length,
                                     std::uint64_t checkpoints)
   {
      std::uint64_t low = 1;
      std::uint64_t high = length - 1;
      while (low < high)
      {
         const std::uint64_t middle = low + (high - low) / 2;
         const std::uint64_t right = ReplayLevel(middle + 1, checkpoints);
         const std::uint64_t left =
            InitialLevel(length - middle, checkpoints - 1);
         if (right >= left) { high = middle; }
         else { low = middle + 1; }
      }
      return low;
   }

   static std::uint64_t ReplaySplit(std::uint64_t length,
                                    std::uint64_t checkpoints)
   {
      std::uint64_t low = 1;
      std::uint64_t high = length - 1;
      while (low < high)
      {
         const std::uint64_t middle = low + (high - low) / 2;
         const std::uint64_t right = ReplayLevel(middle + 1, checkpoints);
         const std::uint64_t left =
            ReplayLevel(length - middle, checkpoints - 1);
         if (right == std::numeric_limits<std::uint64_t>::max() ||
             right + std::uint64_t{1} >= left)
         {
            high = middle;
         }
         else { low = middle + 1; }
      }
      return low;
   }

   static std::uint64_t ReplayLevel(std::uint64_t length,
                                    std::uint64_t checkpoints)
   {
      return length <= 1 ? 0 : MinimumLevel(length, checkpoints, false);
   }

   static std::uint64_t InitialLevel(std::uint64_t length,
                                     std::uint64_t checkpoints)
   {
      return length <= checkpoints ? 0 :
             MinimumLevel(length + std::uint64_t{1}, checkpoints, true);
   }

   static std::uint64_t MinimumLevel(std::uint64_t target,
                                     std::uint64_t checkpoints,
                                     bool initial)
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
         std::uint64_t total = CheckedAdd(checkpoints, middle);
         if (initial) { total = CheckedAdd(total, std::uint64_t{1}); }
         if (BinomialAtLeast(total, checkpoints, target)) { high = middle; }
         else { low = middle + 1; }
      }
      return low;
   }

   static std::uint64_t CheckedAdd(std::uint64_t left,
                                   std::uint64_t right)
   {
      if (right > std::numeric_limits<std::uint64_t>::max() - left)
      {
         throw std::length_error(
            "Revolve binomial-level calculation overflow");
      }
      return left + right;
   }

   static bool BinomialAtLeast(std::uint64_t total,
                               std::uint64_t choose,
                               std::uint64_t target)
   {
      if (choose > total) { return false; }
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
            throw std::logic_error(
               "non-integral Revolve binomial recurrence");
         }
         const std::uint64_t quotient = target / numerator;
         const std::uint64_t remainder = target % numerator;
         const std::uint64_t threshold = quotient + (remainder == 0 ? 0 : 1);
         if (value >= threshold) { return true; }
         value *= numerator;
      }
      return value >= target;
   }

   CheckpointId AllocateCheckpoint(StateId state)
   {
      if (live_checkpoints_.size() >= checkpoint_budget_)
      {
         throw std::logic_error("Revolve checkpoint budget exceeded at state " +
                                std::to_string(state));
      }
      if (next_checkpoint_id_ == CheckpointId{0})
      {
         throw std::length_error("Revolve checkpoint ID overflow");
      }
      const CheckpointId id = next_checkpoint_id_;
      if (next_checkpoint_id_ == std::numeric_limits<CheckpointId>::max())
      {
         next_checkpoint_id_ = CheckpointId{0};
      }
      else { ++next_checkpoint_id_; }
      live_checkpoints_.push_back(LiveCheckpoint{id, state});
      return id;
   }

   std::vector<LiveCheckpoint>::iterator FindCheckpoint(CheckpointId id)
   {
      return std::find_if(live_checkpoints_.begin(), live_checkpoints_.end(),
                          [id](const LiveCheckpoint &checkpoint)
      {
         return checkpoint.id == id;
      });
   }

   CheckpointCommand AdvanceCommand(StateId from, StateId to)
   {
      if (current_state_ != from || to <= from || to > num_steps_)
      {
         throw std::logic_error("invalid Revolve Advance state");
      }
      current_state_ = to;
      return {CheckpointAction::Advance, from, to, std::nullopt};
   }

   CheckpointCommand StoreCommand(StateId state, CheckpointId id) const
   {
      if (current_state_ != state)
      {
         throw std::logic_error("invalid Revolve Store state");
      }
      return {CheckpointAction::Store, state, state, id};
   }

   CheckpointCommand RestoreCommand(CheckpointId id, StateId state)
   {
      const auto found = FindCheckpoint(id);
      if (found == live_checkpoints_.end() || found->state != state)
      {
         throw std::logic_error(
            "Revolve attempted to restore a stale checkpoint ID");
      }
      current_state_ = state;
      return {CheckpointAction::Restore, state, state, id};
   }

   CheckpointCommand ReverseCommand(StateId from, StateId to) const
   {
      if (from != to + StateId{1} || current_state_ != to)
      {
         throw std::logic_error("invalid Revolve Reverse state");
      }
      return {CheckpointAction::Reverse, from, to, std::nullopt};
   }

   CheckpointCommand DiscardCommand(CheckpointId id, StateId state)
   {
      const auto found = FindCheckpoint(id);
      if (found == live_checkpoints_.end() || found->state != state)
      {
         throw std::logic_error(
            "Revolve attempted to discard a stale checkpoint ID");
      }
      live_checkpoints_.erase(found);
      return {CheckpointAction::Discard, state, state, id};
   }

   CheckpointCommand FinishedCommand() const
   {
      return {CheckpointAction::Finished, num_steps_, num_steps_,
              std::nullopt};
   }

   StateId num_steps_ = 0;
   std::size_t checkpoint_budget_ = 0;
   bool configured_ = false;
   bool started_ = false;
   bool base_discarded_ = false;
   bool finished_ = false;
   StateId current_state_ = 0;
   CheckpointId next_checkpoint_id_ = 1;
   CheckpointId base_checkpoint_ = 0;
   std::vector<Frame> frames_;
   std::vector<LiveCheckpoint> live_checkpoints_;
};

} // namespace checkpoint_detail
