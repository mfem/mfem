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

/// Online minimal-repetition Wang-Moin-Iaccarino scheduler.
///
/// Implements Algorithms 3 and 4 of Wang, Moin, and Iaccarino (SIAM J. Sci.
/// Comput. 31(4), 2009). Algorithmic placeholders carry no snapshot bytes and
/// do not count against the physical checkpoint budget. Store is legal during
/// reverse replay. The object is stateful and not internally synchronized.
class WangMoinIaccarinoScheduleImpl final : public OnlineCheckpointSchedule
{
public:
   /// Configures the maximum number of populated checkpoint snapshots.
   /// @throws std::invalid_argument if @p num_checkpoints is zero.
   void Configure(std::size_t num_checkpoints) override
   {
      if (num_checkpoints == 0)
      {
         throw std::invalid_argument(
            "Wang-Moin-Iaccarino schedule requires at least one checkpoint");
      }
      checkpoint_budget_ = num_checkpoints;
      configured_ = true;
      initializeState();
   }

   /// Runs Algorithm 3 while u_step is active, before calculating u_(step+1).
   /// @throws std::logic_error outside original-forward phase.
   /// @throws std::invalid_argument for an unexpected step.
   std::vector<CheckpointCommand> BeforeForwardStep(StepId step) override
   {
      requirePhase(Phase::OriginalForward, "BeforeForwardStep");
      if (step != expected_forward_step_)
      {
         throw std::invalid_argument(
            "Wang-Moin-Iaccarino expected forward step "
            + std::to_string(expected_forward_step_) + ", but received "
            + std::to_string(step));
      }
      if (step == std::numeric_limits<StepId>::max())
      {
         throw std::length_error(
            "Wang-Moin-Iaccarino forward step increment overflow");
      }

      active_step_ = step;
      std::vector<CheckpointCommand> commands = processForwardStep(step);
      ++expected_forward_step_;
      return commands;
   }

   /// Replaces a terminal callback and transitions to reverse scheduling.
   std::vector<CheckpointCommand>
   ForwardIntegrationCompleted(StepId final_step) override
   {
      requirePhase(Phase::OriginalForward, "ForwardIntegrationCompleted");
      if (final_step < 0)
      {
         throw std::invalid_argument(
            "Wang-Moin-Iaccarino final step must be non-negative");
      }
      if (final_step != expected_forward_step_)
      {
         throw std::invalid_argument(
            "Wang-Moin-Iaccarino final step must equal the number of prepared "
            "forward steps");
      }
      if (checkpoints_.empty() || checkpoints_.back().step != final_step
          || checkpoints_.back().snapshot.has_value())
      {
         throw std::logic_error(
            "Wang-Moin-Iaccarino terminal placeholder is inconsistent");
      }

      active_step_ = final_step;
      final_step_ = final_step;
      reverse_from_ = final_step;
      checkpoints_.pop_back();
      reverse_stage_ = ReverseStage::BeginStep;
      phase_ = Phase::Reverse;
      return {};
   }

   /// Returns the next reverse/replay command after forward completion.
   /// @throws std::logic_error when called in an invalid phase.
   CheckpointCommand Next() override
   {
      if (!configured_)
      {
         throw std::logic_error(
            "Wang-Moin-Iaccarino schedule must be configured before next()");
      }
      if (phase_ == Phase::OriginalForward)
      {
         throw std::logic_error(
            "Wang-Moin-Iaccarino reverse schedule is not finalized");
      }
      if (phase_ == Phase::Finished)
      {
         return finishedCommand();
      }
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
            case ReverseStage::BeginStep:
               removeCurrentPlaceholder();
               if (reverse_from_ == 0)
               {
                  if (!checkpoints_.empty() || live_snapshot_count_ != 0)
                  {
                     throw std::logic_error(
                        "Wang-Moin-Iaccarino completed with live checkpoints");
                  }
                  phase_ = Phase::Finished;
                  return finishedCommand();
               }
               beginPrimalReconstruction();
               break;

            case ReverseStage::DiscardRestored:
               discardRestoredCheckpoint();
               break;

            case ReverseStage::PrepareReplay:
               prepareReplayStep();
               break;

            case ReverseStage::AdvanceReplay:
               reverse_stage_ = ReverseStage::PrepareReplay;
               return advanceCommand(active_step_, active_step_ + StepId{1});

            case ReverseStage::EmitReverse:
            {
               const StepId target = reverse_from_ - StepId{1};
               reverse_stage_ = ReverseStage::BeginStep;
               reverse_from_ = target;
               return reverseCommand(target + StepId{1}, target);
            }
         }
      }
   }

   /// Rewinds the configured online state machine.
   void Reset() override
   {
      if (configured_)
      {
         initializeState();
      }
   }

private:
   enum class Phase
   {
      Unconfigured,
      OriginalForward,
      Reverse,
      Finished
   };

   enum class ReverseStage
   {
      BeginStep,
      DiscardRestored,
      PrepareReplay,
      AdvanceReplay,
      EmitReverse
   };

   struct AlgorithmicCheckpoint
   {
      StepId step{0};
      std::uint64_t level{0};
      bool infinite_level{false};
      std::optional<CheckpointId> snapshot;
   };

   void initializeState()
   {
      phase_ = Phase::OriginalForward;
      reverse_stage_ = ReverseStage::BeginStep;
      expected_forward_step_ = 0;
      final_step_ = 0;
      reverse_from_ = 0;
      active_step_ = 0;
      next_checkpoint_id_ = CheckpointId{1};
      live_snapshot_count_ = 0;
      restored_checkpoint_id_.reset();
      pending_commands_.clear();
      pending_command_ = 0;
      checkpoints_.clear();
      checkpoints_.push_back(AlgorithmicCheckpoint{0, 0, true, std::nullopt});
   }

   void requirePhase(Phase required, const char *operation) const
   {
      if (!configured_)
      {
         throw std::logic_error(
            std::string(
               "Wang-Moin-Iaccarino schedule must be configured before ")
            + operation + "()");
      }
      if (phase_ != required)
      {
         throw std::logic_error(std::string("Wang-Moin-Iaccarino ") + operation
                                + "() called in the wrong phase");
      }
   }

   std::vector<CheckpointCommand> processForwardStep(StepId step)
   {
      if (checkpoints_.empty() || checkpoints_.back().step != step
          || checkpoints_.back().snapshot.has_value())
      {
         throw std::logic_error(
            "Wang-Moin-Iaccarino forward placeholder is inconsistent");
      }
      if (step == std::numeric_limits<StepId>::max())
      {
         throw std::length_error(
            "Wang-Moin-Iaccarino placeholder step overflow");
      }

      std::vector<CheckpointCommand> commands;
      if (checkpoints_.size() <= checkpoint_budget_)
      {
         checkpoints_.push_back(
            AlgorithmicCheckpoint{step + StepId{1}, 0, false, std::nullopt});
      }
      else
      {
         const std::optional<std::size_t> dispensable =
            largestDispensableCheckpoint();
         if (dispensable.has_value())
         {
            commands.push_back(discardStoredCheckpoint(*dispensable));
            checkpoints_.erase(checkpoints_.begin()
                               + static_cast<std::ptrdiff_t>(*dispensable));
            checkpoints_.push_back(
               AlgorithmicCheckpoint{step + StepId{1}, 0, false, std::nullopt});
         }
         else
         {
            const std::uint64_t level = checkpoints_.back().level;
            if (level == std::numeric_limits<std::uint64_t>::max())
            {
               throw std::length_error(
                  "Wang-Moin-Iaccarino checkpoint-level overflow");
            }
            checkpoints_.pop_back();
            checkpoints_.push_back(
               AlgorithmicCheckpoint{step + StepId{1}, level + std::uint64_t{1},
                                     false, std::nullopt});
         }
      }

      const auto current =
         std::find_if(checkpoints_.begin(), checkpoints_.end(),
                      [step](const AlgorithmicCheckpoint &checkpoint)
                      { return checkpoint.step == step; });
      if (current != checkpoints_.end())
      {
         if (current->snapshot.has_value())
         {
            throw std::logic_error(
               "Wang-Moin-Iaccarino checkpoint stored twice");
         }
         const CheckpointId id = allocateCheckpointId();
         current->snapshot = id;
         ++live_snapshot_count_;
         if (live_snapshot_count_ > checkpoint_budget_)
         {
            throw std::logic_error(
               "Wang-Moin-Iaccarino checkpoint budget exceeded");
         }
         commands.push_back(storageCommand(CheckpointAction::Store, step, id));
      }
      return commands;
   }

   std::optional<std::size_t> largestDispensableCheckpoint() const
   {
      bool have_higher_to_right = false;
      std::uint64_t highest_level_to_right = 0;
      bool infinite_to_right = false;

      for (std::size_t reverse_index = checkpoints_.size(); reverse_index > 0;
           --reverse_index)
      {
         const std::size_t index = reverse_index - 1;
         const AlgorithmicCheckpoint &checkpoint = checkpoints_[index];
         const bool dispensable =
            infinite_to_right
            || (have_higher_to_right && !checkpoint.infinite_level
                && highest_level_to_right > checkpoint.level);
         if (dispensable)
         {
            return index;
         }

         if (checkpoint.infinite_level)
         {
            infinite_to_right = true;
         }
         else if (!have_higher_to_right
                  || checkpoint.level > highest_level_to_right)
         {
            highest_level_to_right = checkpoint.level;
            have_higher_to_right = true;
         }
      }
      return std::nullopt;
   }

   CheckpointCommand discardStoredCheckpoint(std::size_t index)
   {
      AlgorithmicCheckpoint &checkpoint = checkpoints_[index];
      if (!checkpoint.snapshot.has_value() || live_snapshot_count_ == 0)
      {
         throw std::logic_error(
            "Wang-Moin-Iaccarino attempted to discard a placeholder");
      }
      const CheckpointId id = *checkpoint.snapshot;
      checkpoint.snapshot.reset();
      --live_snapshot_count_;
      return storageCommand(CheckpointAction::Discard, checkpoint.step, id);
   }

   void removeCurrentPlaceholder()
   {
      if (!checkpoints_.empty() && checkpoints_.back().step == reverse_from_
          && !checkpoints_.back().snapshot.has_value())
      {
         checkpoints_.pop_back();
      }
   }

   void beginPrimalReconstruction()
   {
      if (checkpoints_.empty())
      {
         throw std::logic_error(
            "Wang-Moin-Iaccarino has no checkpoint for reverse reconstruction");
      }
      AlgorithmicCheckpoint &checkpoint = checkpoints_.back();
      if (!checkpoint.snapshot.has_value())
      {
         throw std::logic_error(
            "Wang-Moin-Iaccarino reconstruction anchor is not stored");
      }
      const StepId target = reverse_from_ - StepId{1};
      if (checkpoint.step > target)
      {
         throw std::logic_error(
            "Wang-Moin-Iaccarino reconstruction anchor is after its target");
      }

      restored_checkpoint_id_ = checkpoint.snapshot;
      active_step_ = checkpoint.step;
      reverse_stage_ = ReverseStage::DiscardRestored;
      pending_commands_.push_back(storageCommand(
         CheckpointAction::Restore, checkpoint.step, *checkpoint.snapshot));
   }

   void discardRestoredCheckpoint()
   {
      if (checkpoints_.empty() || !restored_checkpoint_id_.has_value()
          || checkpoints_.back().snapshot != restored_checkpoint_id_)
      {
         throw std::logic_error(
            "Wang-Moin-Iaccarino restored checkpoint lifetime is inconsistent");
      }
      pending_commands_.push_back(
         discardStoredCheckpoint(checkpoints_.size() - 1));
      restored_checkpoint_id_.reset();
      reverse_stage_ = active_step_ == reverse_from_ - StepId{1}
                          ? ReverseStage::EmitReverse
                          : ReverseStage::PrepareReplay;
   }

   void prepareReplayStep()
   {
      const StepId target = reverse_from_ - StepId{1};
      if (active_step_ >= target)
      {
         reverse_stage_ = ReverseStage::EmitReverse;
         return;
      }
      pending_commands_ = processForwardStep(active_step_);
      pending_command_ = 0;
      reverse_stage_ = ReverseStage::AdvanceReplay;
   }

   CheckpointId allocateCheckpointId()
   {
      if (next_checkpoint_id_ == CheckpointId{0})
      {
         throw std::length_error("Wang-Moin-Iaccarino checkpoint-ID overflow");
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
      return id;
   }

   CheckpointCommand advanceCommand(StepId from, StepId to)
   {
      if (active_step_ != from || from < 0 || to != from + StepId{1}
          || to > final_step_)
      {
         throw std::logic_error("invalid Wang-Moin-Iaccarino Advance state");
      }
      active_step_ = to;
      return CheckpointCommand{CheckpointAction::Advance, from, to,
                               std::nullopt};
   }

   CheckpointCommand reverseCommand(StepId from, StepId to) const
   {
      if (from != to + StepId{1} || active_step_ != to)
      {
         throw std::logic_error("invalid Wang-Moin-Iaccarino Reverse state");
      }
      return CheckpointCommand{CheckpointAction::Reverse, from, to,
                               std::nullopt};
   }

   static CheckpointCommand storageCommand(CheckpointAction action, StepId step,
                                           CheckpointId id)
   {
      return CheckpointCommand{action, step, step, id};
   }

   CheckpointCommand finishedCommand() const
   {
      return CheckpointCommand{CheckpointAction::Finished, final_step_,
                               final_step_, std::nullopt};
   }

   std::size_t checkpoint_budget_{0};
   bool configured_{false};
   Phase phase_{Phase::Unconfigured};
   ReverseStage reverse_stage_{ReverseStage::BeginStep};
   StepId expected_forward_step_{0};
   StepId final_step_{0};
   StepId reverse_from_{0};
   StepId active_step_{0};
   CheckpointId next_checkpoint_id_{1};
   std::size_t live_snapshot_count_{0};
   std::optional<CheckpointId> restored_checkpoint_id_;
   std::vector<AlgorithmicCheckpoint> checkpoints_;
   std::vector<CheckpointCommand> pending_commands_;
   std::size_t pending_command_{0};
};

} // namespace checkpoint_detail
