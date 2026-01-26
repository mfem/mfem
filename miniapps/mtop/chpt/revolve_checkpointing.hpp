#ifndef MFEM_REVOLVE_CHECKPOINTING_HPP
#define MFEM_REVOLVE_CHECKPOINTING_HPP

#include "mfem.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

using mfem::out;

// -------------------------------
// Fixed-slot storage: Memory
// -------------------------------
class FixedSlotMemoryStorage
{
private:
   int max_slots_ = 0;
   size_t slot_bytes_ = 0;
   std::vector<uint8_t> data_;

public:
   FixedSlotMemoryStorage(int max_slots, size_t slot_bytes)
      : max_slots_(max_slots), slot_bytes_(slot_bytes),
        data_(size_t(max_slots)*slot_bytes, 0)
   {
      MFEM_VERIFY(max_slots_ > 0, "max_slots must be > 0");
      MFEM_VERIFY(slot_bytes_ > 0, "slot_bytes must be > 0");
   }

   int MaxSlots() const { return max_slots_; }
   size_t SlotBytes() const { return slot_bytes_; }

   void Save(int slot, const uint8_t *src, size_t bytes)
   {
      MFEM_VERIFY(0 <= slot && slot < max_slots_, "Save: slot out of range");
      MFEM_VERIFY(bytes == slot_bytes_, "Save: bytes mismatch");
      MFEM_VERIFY(src != nullptr, "Save: src is null");
      std::memcpy(data_.data() + size_t(slot)*slot_bytes_, src, slot_bytes_);
   }

   void Load(int slot, uint8_t *dst, size_t bytes) const
   {
      MFEM_VERIFY(0 <= slot && slot < max_slots_, "Load: slot out of range");
      MFEM_VERIFY(bytes == slot_bytes_, "Load: bytes mismatch");
      MFEM_VERIFY(dst != nullptr, "Load: dst is null");
      std::memcpy(dst, data_.data() + size_t(slot)*slot_bytes_, slot_bytes_);
   }
};

// -------------------------------
// Fixed-slot storage: Single file
// -------------------------------
class FixedSlotFileStorage
{
private:
   int max_slots_ = 0;
   size_t slot_bytes_ = 0;
   std::string filename_;
   mutable std::fstream file_;

   std::streamoff Offset(int slot) const
   {
      return std::streamoff(size_t(slot) * slot_bytes_);
   }

public:
   FixedSlotFileStorage(const std::string &filename,
                        int max_slots,
                        size_t slot_bytes)
      : max_slots_(max_slots), slot_bytes_(slot_bytes), filename_(filename)
   {
      MFEM_VERIFY(max_slots_ > 0, "max_slots must be > 0");
      MFEM_VERIFY(slot_bytes_ > 0, "slot_bytes must be > 0");
      MFEM_VERIFY(!filename_.empty(), "filename must not be empty");

      // Create/truncate file and size it.
      {
         std::ofstream ofs(filename_, std::ios::binary | std::ios::trunc);
         MFEM_VERIFY(ofs.good(), "Failed to create checkpoint file");
         const size_t total = size_t(max_slots_) * slot_bytes_;
         if (total > 0)
         {
            ofs.seekp(std::streamoff(total - 1));
            char zero = 0;
            ofs.write(&zero, 1);
         }
      }

      file_.open(filename_, std::ios::binary | std::ios::in | std::ios::out);
      MFEM_VERIFY(file_.good(), "Failed to open checkpoint file");
   }

   ~FixedSlotFileStorage()
   {
      if (file_.is_open()) { file_.close(); }
   }

   int MaxSlots() const { return max_slots_; }
   size_t SlotBytes() const { return slot_bytes_; }

   void Save(int slot, const uint8_t *src, size_t bytes)
   {
      MFEM_VERIFY(0 <= slot && slot < max_slots_, "Save: slot out of range");
      MFEM_VERIFY(bytes == slot_bytes_, "Save: bytes mismatch");
      MFEM_VERIFY(src != nullptr, "Save: src is null");

      file_.seekp(Offset(slot));
      MFEM_VERIFY(file_.good(), "Save: seekp failed");
      file_.write(reinterpret_cast<const char*>(src), std::streamsize(slot_bytes_));
      MFEM_VERIFY(file_.good(), "Save: write failed");
      file_.flush();
   }

   void Load(int slot, uint8_t *dst, size_t bytes) const
   {
      MFEM_VERIFY(0 <= slot && slot < max_slots_, "Load: slot out of range");
      MFEM_VERIFY(bytes == slot_bytes_, "Load: bytes mismatch");
      MFEM_VERIFY(dst != nullptr, "Load: dst is null");

      file_.seekg(Offset(slot));
      MFEM_VERIFY(file_.good(), "Load: seekg failed");
      file_.read(reinterpret_cast<char*>(dst), std::streamsize(slot_bytes_));
      MFEM_VERIFY(file_.good(), "Load: read failed");
   }
};

// -------------------------------
// REVOLVE controller (actions)
// (Transcribed from revolve.c used in ADOL-C; Algorithm 799.)
// -------------------------------
enum class RevolveAction
{
   advance,
   takeshot,
   restore,
   firsturn,
   youturn,
   terminate
};

class RevolveController
{
private:
   int snaps_ = 0;
   int check_ = -1;
   int capo_  = 0;
   int fine_  = 0;

   int turn_ = 0;
   int oldfine_ = 0;
   int oldsnaps_ = 0;

   // Stack of checkpoint times, indexed by 'check_'.
   std::vector<int> ch_;

public:
   RevolveController() = default;

   RevolveController(int snaps, int capo0, int fine0)
      : snaps_(snaps), check_(-1), capo_(capo0), fine_(fine0),
        turn_(0), oldfine_(fine0), oldsnaps_(snaps), ch_(snaps, 0)
   {
      MFEM_VERIFY(snaps_ > 0, "REVOLVE snaps must be > 0");
      MFEM_VERIFY(capo_ <= fine_, "REVOLVE: capo must be <= fine");

      // Match revolve.c initialization behavior.
      if (check_ == -1 && capo_ < fine_)
      {
         turn_ = 0;
         ch_[0] = capo_ - 1;
      }
   }

   int Snaps() const { return snaps_; }
   int Check() const { return check_; }
   int Capo() const  { return capo_; }
   int Fine() const  { return fine_; }
   const std::vector<int>& CheckpointTimes() const { return ch_; }

   RevolveAction Next()
   {
      MFEM_VERIFY(!(check_ < -1), "REVOLVE: check < -1");
      MFEM_VERIFY(!(capo_ > fine_), "REVOLVE: capo > fine");

      if ((check_ == -1) && (capo_ < fine_))
      {
         turn_ = 0;
         ch_[0] = capo_ - 1;
      }

      const int diff = fine_ - capo_;
      switch (diff)
      {
         case 0:
         {
            // Terminate or restore to next checkpoint on the stack.
            if (check_ == -1 || capo_ == ch_[0])
            {
               check_ -= 1; // mirror revolve.c behavior
               return RevolveAction::terminate;
            }
            else
            {
               capo_ = ch_[check_];
               oldfine_ = fine_;
               return RevolveAction::restore;
            }
         }
         case 1:
         {
            // One adjoint step available.
            fine_ -= 1;
            if (check_ >= 0 && ch_[check_] == capo_) { check_ -= 1; }

            if (turn_ == 0)
            {
               turn_ = 1;
               oldfine_ = fine_;
               return RevolveAction::firsturn;
            }
            else
            {
               oldfine_ = fine_;
               return RevolveAction::youturn;
            }
         }
         default:
         {
            // diff > 1
            if (check_ == -1 || ch_[check_] != capo_)
            {
               // Take a new checkpoint at current capo.
               check_ += 1;
               MFEM_VERIFY(check_ + 1 <= snaps_, "REVOLVE: exceeded snaps");
               ch_[check_] = capo_;
               oldfine_ = fine_;
               return RevolveAction::takeshot;
            }
            else
            {
               // Advance capo forward within (capo, fine).
               // This follows the binomial logic in revolve.c.
               MFEM_VERIFY(!((oldfine_ < fine_) && (snaps_ == check_ + 1)),
                           "REVOLVE: fine increased unexpectedly with full stack");

               const int oldcapo = capo_;
               const int ds = snaps_ - check_;
               MFEM_VERIFY(ds >= 1, "REVOLVE: ds < 1");

               int reps = 0;
               long long range = 1;
               while (range < (fine_ - capo_))
               {
                  reps += 1;
                  // range = range*(reps+ds)/reps  (integer arithmetic)
                  range = range * (reps + ds) / reps;
               }
               MFEM_VERIFY(reps >= 1, "REVOLVE: reps < 1");

               // Binomial helper values (integer)
               const long long bino1 = range * reps / (ds + reps);
               const long long bino2 = (ds > 1) ? (bino1 * ds / (ds + reps - 1)) : 1;
               const long long bino3 =
                  (ds == 1) ? 0 :
                  (ds > 2)  ? (bino2 * (ds - 1) / (ds + reps - 2)) : 1;
               const long long bino4 = bino2 * (reps - 1) / ds;
               const long long bino5 =
                  (ds < 3) ? 0 :
                  (ds > 3) ? (bino3 * (ds - 2) / reps) : 1;

               // Kowarz "new version": keep l^ as small as possible
               const long long bino6 = bino1 * ds / reps;

               const long long gap = fine_ - capo_;
               if (gap <= bino1 + bino3)
               {
                  capo_ += int(bino4);
               }
               else if (gap < bino1 + bino2)
               {
                  capo_ = fine_ - int(bino2 + bino3);
               }
               else if (gap <= bino1 + bino2 + bino5)
               {
                  capo_ += int(bino1 - bino3);
               }
               else
               {
                  capo_ = fine_ - int(bino6);
               }

               if (capo_ == oldcapo) { capo_ = oldcapo + 1; }

               oldfine_ = fine_;
               return RevolveAction::advance;
            }
         }
      }
   }
};

// -------------------------------
// Fixed-step REVOLVE checkpointing
// -------------------------------
template <typename StorageT>
class FixedStepRevolveCheckpointing
{
public:
   struct Shot
   {
      int time = 0;   // state index
      int slot = 0;   // checkpoint slot index
   };

private:
   int num_steps_ = 0;        // total number of primal steps (0..num_steps)
   int num_checkpoints_ = 0;  // REVOLVE "snaps"
   size_t snapshot_bytes_ = 0;

   StorageT *storage_ = nullptr;

   std::vector<Shot> forward_shots_;
   int forward_shot_cursor_ = 0;

   // Controller state at the beginning of the reverse sweep (pre-firsturn).
   RevolveController ctrl_init_;
   RevolveController ctrl_;

   // Two scratch buffers for (de)serialization.
   std::vector<uint8_t> io_buf_;
   std::vector<uint8_t> prefinal_buf_;
   bool prefinal_valid_ = false;

   // Reverse sweep bookkeeping.
   bool reverse_started_ = false;
   int u_work_time_ = -1;

public:
   FixedStepRevolveCheckpointing(int num_steps,
                                int num_checkpoints,
                                size_t snapshot_bytes,
                                StorageT &storage)
      : num_steps_(num_steps),
        num_checkpoints_(num_checkpoints),
        snapshot_bytes_(snapshot_bytes),
        storage_(&storage),
        io_buf_(snapshot_bytes, 0),
        prefinal_buf_(snapshot_bytes, 0)
   {
      MFEM_VERIFY(num_steps_ >= 0, "num_steps must be >= 0");
      MFEM_VERIFY(num_checkpoints_ > 0, "num_checkpoints must be > 0");
      MFEM_VERIFY(snapshot_bytes_ > 0, "snapshot_bytes must be > 0");

      MFEM_VERIFY(storage_->MaxSlots() == num_checkpoints_,
                  "Storage MaxSlots() must match num_checkpoints");
      MFEM_VERIFY(storage_->SlotBytes() == snapshot_bytes_,
                  "Storage SlotBytes() must match snapshot_bytes");

      BuildForwardPlanAndInitialControllerState();
      Reset();
   }

   void Reset()
   {
      forward_shot_cursor_ = 0;
      prefinal_valid_ = false;
      reverse_started_ = false;
      u_work_time_ = -1;
      ctrl_ = ctrl_init_;
   }

   int NumSteps() const { return num_steps_; }
   int NumCheckpoints() const { return num_checkpoints_; }
   size_t SnapshotBytes() const { return snapshot_bytes_; }

   // ForwardStep: called for i=0..num_steps-1
   template <typename State, typename PrimalStep, typename MakeSnapshot>
   void ForwardStep(int i,
                    State &u,
                    PrimalStep &&primal_step,
                    MakeSnapshot &&make_snapshot)
   {
      MFEM_VERIFY(0 <= i && i < num_steps_, "ForwardStep: i out of range");

      // Take any planned shots at time i (before advancing).
      while (forward_shot_cursor_ < (int)forward_shots_.size() &&
             forward_shots_[forward_shot_cursor_].time == i)
      {
         const int slot = forward_shots_[forward_shot_cursor_].slot;
         make_snapshot(u, io_buf_.data(), snapshot_bytes_);
         storage_->Save(slot, io_buf_.data(), snapshot_bytes_);
         forward_shot_cursor_++;
      }

      // Cache u_{num_steps-1} so reverse can start there (REVOLVE expects capo=num_steps-1).
      if (i == num_steps_ - 1)
      {
         make_snapshot(u, prefinal_buf_.data(), snapshot_bytes_);
         prefinal_valid_ = true;
      }

      // Advance one step.
      primal_step(i, u);
   }

   // BackwardStep: called for i=num_steps-1..0
   template <typename State, typename AdjointState,
             typename PrimalStep, typename AdjointStep,
             typename MakeSnapshot, typename RestoreSnapshot>
   void BackwardStep(int i,
                     AdjointState &lambda,
                     State &u_work,
                     PrimalStep &&primal_step,
                     AdjointStep &&adjoint_step,
                     MakeSnapshot &&make_snapshot,
                     RestoreSnapshot &&restore_snapshot)
   {
      MFEM_VERIFY(0 <= i && i < num_steps_, "BackwardStep: i out of range");

      if (!reverse_started_)
      {
         MFEM_VERIFY(prefinal_valid_ || num_steps_ == 0,
                     "Reverse started but prefinal state was not captured. "
                     "Did you run the forward loop through i=num_steps-1?");

         ctrl_ = ctrl_init_;
         reverse_started_ = true;

         if (num_steps_ > 0)
         {
            // Restore u_{num_steps-1} into u_work.
            restore_snapshot(u_work, prefinal_buf_.data(), snapshot_bytes_);
            u_work_time_ = num_steps_ - 1;
         }
      }

      // Execute controller actions until we perform exactly one adjoint step.
      for (;;)
      {
         RevolveAction act = ctrl_.Next();

         switch (act)
         {
            case RevolveAction::takeshot:
            {
               const int slot = ctrl_.Check();
               MFEM_VERIFY(u_work_time_ == ctrl_.Capo(),
                           "takeshot: u_work_time must equal capo");
               make_snapshot(u_work, io_buf_.data(), snapshot_bytes_);
               storage_->Save(slot, io_buf_.data(), snapshot_bytes_);
               break;
            }
            case RevolveAction::restore:
            {
               const int slot = ctrl_.Check();
               storage_->Load(slot, io_buf_.data(), snapshot_bytes_);
               restore_snapshot(u_work, io_buf_.data(), snapshot_bytes_);
               u_work_time_ = ctrl_.Capo();
               break;
            }
            case RevolveAction::advance:
            {
               const int target = ctrl_.Capo();
               MFEM_VERIFY(u_work_time_ >= 0, "advance: u_work_time not initialized");
               MFEM_VERIFY(target >= u_work_time_, "advance: target < current time");

               for (int t = u_work_time_; t < target; ++t)
               {
                  primal_step(t, u_work);
               }
               u_work_time_ = target;
               break;
            }
            case RevolveAction::firsturn:
            case RevolveAction::youturn:
            {
               // After firsturn/youturn, ctrl_.Fine() has been decremented and equals ctrl_.Capo().
               const int step = ctrl_.Fine();
               MFEM_VERIFY(step == ctrl_.Capo(), "youturn: fine != capo");
               MFEM_VERIFY(step == u_work_time_, "youturn: u_work_time != step");
               MFEM_VERIFY(step == i, "BackwardStep called with i that doesn't match REVOLVE schedule");

               // One adjoint step.
               adjoint_step(step, u_work, lambda);
               return;
            }
            case RevolveAction::terminate:
            {
               MFEM_ABORT("REVOLVE terminated early: BackwardStep called after completion?");
               break;
            }
         }
      }
   }

private:
   void BuildForwardPlanAndInitialControllerState()
   {
      forward_shots_.clear();

      RevolveController sim(num_checkpoints_, /*capo=*/0, /*fine=*/num_steps_);

      // Forward-plan phase: only TAKESHOT/ADVANCE should occur while fine-capo>1.
      while (sim.Fine() - sim.Capo() > 1)
      {
         RevolveAction a = sim.Next();
         if (a == RevolveAction::takeshot)
         {
            forward_shots_.push_back({sim.Capo(), sim.Check()});
         }
         else if (a == RevolveAction::advance)
         {
            // nothing to record; capo moved forward inside sim
         }
         else
         {
            MFEM_ABORT("Unexpected REVOLVE action during forward planning phase");
         }
      }

      // At this point, sim is in the pre-firsturn state (capo = num_steps-1, fine = num_steps)
      // for num_steps>=1. For num_steps<=1, fine-capo<=1 from the start.
      ctrl_init_ = sim;
   }
};





#endif //MFEM_REVOLVE_CHECKPOINTING_HPP