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

/// Online minimal-repetition Wang--Moin--Iaccarino implementation.
class WangMoinIaccarinoScheduleImpl final : public OnlineCheckpointSchedule
{
public:
   void Configure(std::size_t num_checkpoints) override
   {
      if (num_checkpoints == 0)
      {
         throw std::invalid_argument(
            "Wang-Moin-Iaccarino schedule requires at least one checkpoint");
      }
      checkpoint_budget_ = num_checkpoints;
      configured_ = true;
      InitializeState();
   }

   std::vector<CheckpointCommand> BeforeForwardState(StateId state) override
   {
      RequirePhase(Phase::OriginalForward, "BeforeForwardState");
      if (state != expected_forward_state_)
      {
         throw std::invalid_argument(
            "Wang-Moin-Iaccarino received an unexpected forward state");
      }
      if (state == std::numeric_limits<StateId>::max())
      {
         throw std::length_error(
            "Wang-Moin-Iaccarino forward state increment overflow");
      }
      active_state_ = state;
      std::vector<CheckpointCommand> commands = ProcessForwardState(state);
      ++expected_forward_state_;
      return commands;
   }

   std::vector<CheckpointCommand> ForwardComplete(
      StateId terminal_state) override
   {
      RequirePhase(Phase::OriginalForward, "ForwardComplete");
      if (terminal_state < 0 || terminal_state != expected_forward_state_)
      {
         throw std::invalid_argument(
            "Wang-Moin-Iaccarino terminal state is inconsistent");
      }
      if (checkpoints_.empty() ||
          checkpoints_.back().state != terminal_state ||
          checkpoints_.back().snapshot)
      {
         throw std::logic_error(
            "Wang-Moin-Iaccarino terminal placeholder is inconsistent");
      }
      active_state_ = terminal_state;
      terminal_state_ = terminal_state;
      reverse_from_ = terminal_state;
      checkpoints_.pop_back();
      reverse_stage_ = ReverseStage::BeginState;
      phase_ = Phase::Reverse;
      return {};
   }

   CheckpointCommand Next() override
   {
      if (!configured_)
      {
         throw std::logic_error(
            "Wang-Moin-Iaccarino schedule must be configured before Next()");
      }
      if (phase_ == Phase::OriginalForward)
      {
         throw std::logic_error(
            "Wang-Moin-Iaccarino reverse schedule is not finalized");
      }
      if (phase_ == Phase::Finished) { return FinishedCommand(); }
      if (phase_ != Phase::Reverse)
      {
         throw std::logic_error(
            "Wang-Moin-Iaccarino schedule is not ready for reverse execution");
      }

      while (true)
      {
         if (pending_command_ < pending_commands_.size())
         {
            return pending_commands_[pending_command_++];
         }
         pending_commands_.clear();
         pending_command_ = 0;
         switch (reverse_stage_)
         {
            case ReverseStage::BeginState:
               RemoveCurrentPlaceholder();
               if (reverse_from_ == 0)
               {
                  if (!checkpoints_.empty() || live_snapshot_count_ != 0)
                  {
                     throw std::logic_error(
                        "Wang-Moin-Iaccarino completed with live checkpoints");
                  }
                  phase_ = Phase::Finished;
                  return FinishedCommand();
               }
               BeginPrimalReconstruction();
               break;
            case ReverseStage::DiscardRestored:
               DiscardRestoredCheckpoint();
               break;
            case ReverseStage::PrepareReplay:
               PrepareReplayState();
               break;
            case ReverseStage::AdvanceReplay:
               reverse_stage_ = ReverseStage::PrepareReplay;
               return AdvanceCommand(active_state_, active_state_ + StateId{1});
            case ReverseStage::EmitReverse:
            {
               const StateId target = reverse_from_ - StateId{1};
               reverse_stage_ = ReverseStage::BeginState;
               reverse_from_ = target;
               return ReverseCommand(target + StateId{1}, target);
            }
         }
      }
   }

   void Reset() override
   {
      if (configured_) { InitializeState(); }
   }

private:
   enum class Phase { Unconfigured, OriginalForward, Reverse, Finished };
   enum class ReverseStage
   {
      BeginState,
      DiscardRestored,
      PrepareReplay,
      AdvanceReplay,
      EmitReverse
   };

   struct AlgorithmicCheckpoint
   {
      StateId state;
      std::uint64_t level;
      bool infinite_level;
      std::optional<CheckpointId> snapshot;
   };

   void InitializeState()
   {
      phase_ = Phase::OriginalForward;
      reverse_stage_ = ReverseStage::BeginState;
      expected_forward_state_ = 0;
      terminal_state_ = 0;
      reverse_from_ = 0;
      active_state_ = 0;
      next_checkpoint_id_ = CheckpointId{1};
      live_snapshot_count_ = 0;
      restored_checkpoint_id_.reset();
      pending_commands_.clear();
      pending_command_ = 0;
      checkpoints_.clear();
      checkpoints_.push_back(
         AlgorithmicCheckpoint{0, 0, true, std::nullopt});
   }

   void RequirePhase(Phase required, const char *operation) const
   {
      if (!configured_)
      {
         throw std::logic_error(std::string(
            "Wang-Moin-Iaccarino schedule must be configured before ") +
            operation + "()");
      }
      if (phase_ != required)
      {
         throw std::logic_error(std::string("Wang-Moin-Iaccarino ") +
                                operation + "() called in the wrong phase");
      }
   }

   std::vector<CheckpointCommand> ProcessForwardState(StateId state)
   {
      if (checkpoints_.empty() || checkpoints_.back().state != state ||
          checkpoints_.back().snapshot)
      {
         throw std::logic_error(
            "Wang-Moin-Iaccarino forward placeholder is inconsistent");
      }
      std::vector<CheckpointCommand> commands;
      if (checkpoints_.size() <= checkpoint_budget_)
      {
         checkpoints_.push_back(
            AlgorithmicCheckpoint{state + StateId{1}, 0, false, std::nullopt});
      }
      else
      {
         const std::optional<std::size_t> dispensable =
            LargestDispensableCheckpoint();
         if (dispensable)
         {
            commands.push_back(DiscardStoredCheckpoint(*dispensable));
            checkpoints_.erase(checkpoints_.begin() +
                               static_cast<std::ptrdiff_t>(*dispensable));
            checkpoints_.push_back(AlgorithmicCheckpoint{
               state + StateId{1}, 0, false, std::nullopt});
         }
         else
         {
            const std::uint64_t level = checkpoints_.back().level;
            if (level == std::numeric_limits<std::uint64_t>::max())
            {
               throw std::length_error(
                  "Wang-Moin-Iaccarino checkpoint level overflow");
            }
            checkpoints_.pop_back();
            checkpoints_.push_back(AlgorithmicCheckpoint{
               state + StateId{1}, level + std::uint64_t{1}, false,
               std::nullopt});
         }
      }

      const auto current = std::find_if(
                              checkpoints_.begin(), checkpoints_.end(),
                              [state](const AlgorithmicCheckpoint &checkpoint)
      {
         return checkpoint.state == state;
      });
      if (current != checkpoints_.end())
      {
         if (current->snapshot)
         {
            throw std::logic_error(
               "Wang-Moin-Iaccarino checkpoint stored twice");
         }
         const CheckpointId id = AllocateCheckpointId();
         current->snapshot = id;
         ++live_snapshot_count_;
         if (live_snapshot_count_ > checkpoint_budget_)
         {
            throw std::logic_error(
               "Wang-Moin-Iaccarino checkpoint budget exceeded");
         }
         commands.push_back(StorageCommand(CheckpointAction::Store, state, id));
      }
      return commands;
   }

   std::optional<std::size_t> LargestDispensableCheckpoint() const
   {
      bool have_higher_to_right = false;
      std::uint64_t highest_level_to_right = 0;
      bool infinite_to_right = false;
      for (std::size_t reverse_index = checkpoints_.size();
           reverse_index > 0; --reverse_index)
      {
         const std::size_t index = reverse_index - 1;
         const AlgorithmicCheckpoint &checkpoint = checkpoints_[index];
         const bool dispensable = infinite_to_right ||
            (have_higher_to_right && !checkpoint.infinite_level &&
             highest_level_to_right > checkpoint.level);
         if (dispensable) { return index; }
         if (checkpoint.infinite_level) { infinite_to_right = true; }
         else if (!have_higher_to_right ||
                  checkpoint.level > highest_level_to_right)
         {
            highest_level_to_right = checkpoint.level;
            have_higher_to_right = true;
         }
      }
      return std::nullopt;
   }

   CheckpointCommand DiscardStoredCheckpoint(std::size_t index)
   {
      AlgorithmicCheckpoint &checkpoint = checkpoints_[index];
      if (!checkpoint.snapshot || live_snapshot_count_ == 0)
      {
         throw std::logic_error(
            "Wang-Moin-Iaccarino attempted to discard a placeholder");
      }
      const CheckpointId id = *checkpoint.snapshot;
      checkpoint.snapshot.reset();
      --live_snapshot_count_;
      return StorageCommand(CheckpointAction::Discard, checkpoint.state, id);
   }

   void RemoveCurrentPlaceholder()
   {
      if (!checkpoints_.empty() &&
          checkpoints_.back().state == reverse_from_ &&
          !checkpoints_.back().snapshot)
      {
         checkpoints_.pop_back();
      }
   }

   void BeginPrimalReconstruction()
   {
      if (checkpoints_.empty() || !checkpoints_.back().snapshot)
      {
         throw std::logic_error(
            "Wang-Moin-Iaccarino has no stored reconstruction anchor");
      }
      AlgorithmicCheckpoint &checkpoint = checkpoints_.back();
      const StateId target = reverse_from_ - StateId{1};
      if (checkpoint.state > target)
      {
         throw std::logic_error(
            "Wang-Moin-Iaccarino reconstruction anchor follows its target");
      }
      restored_checkpoint_id_ = checkpoint.snapshot;
      active_state_ = checkpoint.state;
      reverse_stage_ = ReverseStage::DiscardRestored;
      pending_commands_.push_back(StorageCommand(
                                     CheckpointAction::Restore,
                                     checkpoint.state,
                                     *checkpoint.snapshot));
   }

   void DiscardRestoredCheckpoint()
   {
      if (checkpoints_.empty() || !restored_checkpoint_id_ ||
          checkpoints_.back().snapshot != restored_checkpoint_id_)
      {
         throw std::logic_error(
            "Wang-Moin-Iaccarino restored checkpoint lifetime is inconsistent");
      }
      pending_commands_.push_back(
         DiscardStoredCheckpoint(checkpoints_.size() - 1));
      restored_checkpoint_id_.reset();
      reverse_stage_ = active_state_ == reverse_from_ - StateId{1} ?
                       ReverseStage::EmitReverse : ReverseStage::PrepareReplay;
   }

   void PrepareReplayState()
   {
      const StateId target = reverse_from_ - StateId{1};
      if (active_state_ >= target)
      {
         reverse_stage_ = ReverseStage::EmitReverse;
         return;
      }
      pending_commands_ = ProcessForwardState(active_state_);
      pending_command_ = 0;
      reverse_stage_ = ReverseStage::AdvanceReplay;
   }

   CheckpointId AllocateCheckpointId()
   {
      if (next_checkpoint_id_ == CheckpointId{0})
      {
         throw std::length_error(
            "Wang-Moin-Iaccarino checkpoint ID overflow");
      }
      const CheckpointId id = next_checkpoint_id_;
      if (next_checkpoint_id_ == std::numeric_limits<CheckpointId>::max())
      {
         next_checkpoint_id_ = CheckpointId{0};
      }
      else { ++next_checkpoint_id_; }
      return id;
   }

   CheckpointCommand AdvanceCommand(StateId from, StateId to)
   {
      if (active_state_ != from || from < 0 || to != from + StateId{1} ||
          to > terminal_state_)
      {
         throw std::logic_error(
            "invalid Wang-Moin-Iaccarino Advance state");
      }
      active_state_ = to;
      return {CheckpointAction::Advance, from, to, std::nullopt};
   }

   CheckpointCommand ReverseCommand(StateId from, StateId to) const
   {
      if (from != to + StateId{1} || active_state_ != to)
      {
         throw std::logic_error(
            "invalid Wang-Moin-Iaccarino Reverse state");
      }
      return {CheckpointAction::Reverse, from, to, std::nullopt};
   }

   static CheckpointCommand StorageCommand(CheckpointAction action,
                                            StateId state,
                                            CheckpointId id)
   {
      return {action, state, state, id};
   }

   CheckpointCommand FinishedCommand() const
   {
      return {CheckpointAction::Finished, terminal_state_, terminal_state_,
              std::nullopt};
   }

   std::size_t checkpoint_budget_ = 0;
   bool configured_ = false;
   Phase phase_ = Phase::Unconfigured;
   ReverseStage reverse_stage_ = ReverseStage::BeginState;
   StateId expected_forward_state_ = 0;
   StateId terminal_state_ = 0;
   StateId reverse_from_ = 0;
   StateId active_state_ = 0;
   CheckpointId next_checkpoint_id_ = 1;
   std::size_t live_snapshot_count_ = 0;
   std::optional<CheckpointId> restored_checkpoint_id_;
   std::vector<AlgorithmicCheckpoint> checkpoints_;
   std::vector<CheckpointCommand> pending_commands_;
   std::size_t pending_command_ = 0;
};

} // namespace checkpoint_detail
