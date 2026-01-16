#ifndef MFEM_DYNAMIC_CHECKPOINTING_HPP
#define MFEM_DYNAMIC_CHECKPOINTING_HPP

#include "mfem.hpp"

#include <map>
#include <memory>
#include <limits>
#include <vector>
#include <utility>

namespace mfem
{

template <typename Snapshot>
class InMemoryCheckpointStorage
{
public:
   using Handle = int;

   Handle InvalidHandle() const { return -1; }
   bool IsValid(const Handle &h) const { return h >= 0; }

   Handle Store(Snapshot &&snap)
   {
      Handle id = InvalidHandle();
      if (!free_.empty())
      {
         id = free_.back();
         free_.pop_back();
         slots_[id].reset(new Snapshot(std::move(snap)));
      }
      else
      {
         id = (Handle) slots_.size();
         slots_.push_back(std::unique_ptr<Snapshot>(new Snapshot(std::move(snap))));
      }
      return id;
   }

   template <typename Func>
   void Read(const Handle &h, Func &&f) const
   {
      MFEM_VERIFY(IsValid(h), "InMemoryCheckpointStorage: invalid handle.");
      MFEM_VERIFY(h < (Handle) slots_.size(), "InMemoryCheckpointStorage: handle out of range.");
      MFEM_VERIFY(slots_[h].get() != nullptr, "InMemoryCheckpointStorage: empty slot.");
      f(*slots_[h]);
   }

   void Erase(Handle &h)
   {
      if (!IsValid(h)) { h = InvalidHandle(); return; }
      MFEM_VERIFY(h < (Handle) slots_.size(), "InMemoryCheckpointStorage: handle out of range.");
      slots_[h].reset();
      free_.push_back(h);
      h = InvalidHandle();
   }

private:
   std::vector<std::unique_ptr<Snapshot>> slots_;
   std::vector<Handle> free_;
};


/**
 * Dynamic checkpointing manager (Wang–Moin–Iaccarino 2009), with pluggable storage.
 *
 * Snapshot: checkpointed object (often the primal State itself).
 * Storage : must provide:
 *   using Handle = ...
 *   Handle InvalidHandle() const;
 *   bool   IsValid(const Handle&) const;
 *   Handle Store(Snapshot&&);
 *   template<class F> void Read(const Handle&, F&&) const; // calls f(const Snapshot&)
 *   void Erase(Handle&);
 */
template <typename Snapshot,
          typename Storage = InMemoryCheckpointStorage<Snapshot>>
class DynamicCheckpointing
{
public:
   using Step = long long;
   using Handle = typename Storage::Handle;

   struct Checkpoint
   {
      int level = 0;
      Handle h; // InvalidHandle => placeholder
      Checkpoint() = default;
      Checkpoint(int lvl, const Handle &inv) : level(lvl), h(inv) {}
   };

   explicit DynamicCheckpointing(int s)
      : s_(s),
        owned_storage_(new Storage()),
        storage_(owned_storage_.get())
   {
      MFEM_VERIFY(s_ > 0, "DynamicCheckpointing: s must be > 0.");
      Reset();
   }

   DynamicCheckpointing(int s, Storage &external_storage)
      : s_(s),
        owned_storage_(nullptr),
        storage_(&external_storage)
   {
      MFEM_VERIFY(s_ > 0, "DynamicCheckpointing: s must be > 0.");
      Reset();
   }

   ~DynamicCheckpointing() { ReleaseAll(); }

   void Reset()
   {
      ReleaseAll();
      cps_.emplace(Step(0), Checkpoint(InfLevel(), storage_->InvalidHandle()));
   }

   struct CheckpointInfo
   {
      Step step;
      int level;
      bool stored;
   };

   std::vector<CheckpointInfo> GetCheckpointInfo() const
   {
      std::vector<CheckpointInfo> out;
      out.reserve(cps_.size());
      for (const auto &kv : cps_)
      {
         out.push_back({kv.first, kv.second.level, storage_->IsValid(kv.second.h)});
      }
      return out;
   }

   Step GetMaxStep() const
   {
      MFEM_VERIFY(!cps_.empty(), "DynamicCheckpointing: checkpoint map is empty.");
      return cps_.rbegin()->first;
   }

   template <typename State, typename PrimalStepFn, typename MakeSnapshotFn>
   void ForwardStep(const Step i,
                    State &u_i_inout,
                    PrimalStepFn &&primal_step,
                    MakeSnapshotFn &&make_snapshot)
   {
      AllocateCheckpointForNextStep(i);

      auto it = cps_.find(i);
      if (it != cps_.end())
      {
         storage_->Erase(it->second.h);
         Snapshot snap = make_snapshot(u_i_inout);
         it->second.h = storage_->Store(std::move(snap));
      }

      primal_step(u_i_inout, i);
   }

   template <typename State,
             typename AdjState,
             typename PrimalStepFn,
             typename AdjointStepFn,
             typename MakeSnapshotFn,
             typename RestoreSnapshotFn>
   void BackwardStep(const Step i,
                     AdjState &q_ip1_inout,
                     State &u_work_inout,
                     PrimalStepFn &&primal_step,
                     AdjointStepFn &&adjoint_step,
                     MakeSnapshotFn &&make_snapshot,
                     RestoreSnapshotFn &&restore_snapshot)
   {
      // remove placeholder at i+1
      const Step ph = i + 1;
      auto it_ph = cps_.find(ph);
      MFEM_VERIFY(it_ph != cps_.end(),
                  "DynamicCheckpointing: expected checkpoint at i+1 before BackwardStep.");
      storage_->Erase(it_ph->second.h);
      cps_.erase(it_ph);

      MFEM_ASSERT(GetMaxStep() <= i,
                  "DynamicCheckpointing: found a checkpoint beyond current adjoint step.");

      auto restore_from_handle = [&](Handle &h)
      {
         storage_->Read(h, [&](const Snapshot &snap)
         {
            restore_snapshot(snap, u_work_inout);
         });
         storage_->Erase(h); // retrieved => placeholder (Algorithm 4 semantics)
      };

      if (GetMaxStep() == i)
      {
         Handle h = TakeHandleMakePlaceholder(i);
         restore_from_handle(h);
      }
      else
      {
         const Step k = GetMaxStep();
         Handle hk = TakeHandleMakePlaceholder(k);
         restore_from_handle(hk);

         for (Step t = k; t < i; ++t)
         {
            ForwardStep(t, u_work_inout, primal_step, make_snapshot);
         }
      }

      adjoint_step(q_ip1_inout, u_work_inout, i);
   }

private:
   int s_ = 0;
   std::unique_ptr<Storage> owned_storage_;
   Storage *storage_ = nullptr;
   std::map<Step, Checkpoint> cps_;

   static int InfLevel() { return std::numeric_limits<int>::max(); }

   void ReleaseAll()
   {
      if (!storage_) { cps_.clear(); return; }
      for (auto &kv : cps_) { storage_->Erase(kv.second.h); }
      cps_.clear();
   }

   bool FindDispensableLargestStep(Step &out_step) const
   {
      int max_level_seen = std::numeric_limits<int>::min();
      for (auto it = cps_.rbegin(); it != cps_.rend(); ++it)
      {
         const Step step = it->first;
         const int lvl = it->second.level;
         if (max_level_seen > lvl) { out_step = step; return true; }
         max_level_seen = (lvl > max_level_seen) ? lvl : max_level_seen;
      }
      return false;
   }

   void AllocateCheckpointForNextStep(const Step i)
   {
      const Step new_step = i + 1;
      MFEM_VERIFY(cps_.find(new_step) == cps_.end(),
                  "DynamicCheckpointing: checkpoint at i+1 already exists.");

      const Handle inv = storage_->InvalidHandle();

      // allow growth to s+1 entries (incl. placeholder)
      if ((int)cps_.size() <= s_)
      {
         cps_.emplace(new_step, Checkpoint(0, inv));
         return;
      }

      Step disp = -1;
      if (FindDispensableLargestStep(disp))
      {
         auto it = cps_.find(disp);
         MFEM_ASSERT(it != cps_.end(), "Internal error: dispensable checkpoint not found.");
         storage_->Erase(it->second.h);
         cps_.erase(it);
         cps_.emplace(new_step, Checkpoint(0, inv));
         return;
      }

      auto it_i = cps_.find(i);
      MFEM_VERIFY(it_i != cps_.end(),
                  "DynamicCheckpointing: promotion expected checkpoint at step i but none found.");
      MFEM_VERIFY(i != 0, "DynamicCheckpointing: attempted to remove step 0 checkpoint.");

      const int l = it_i->second.level;
      storage_->Erase(it_i->second.h);
      cps_.erase(it_i);
      cps_.emplace(new_step, Checkpoint(l + 1, inv));
   }

   Handle TakeHandleMakePlaceholder(const Step i)
   {
      auto it = cps_.find(i);
      MFEM_VERIFY(it != cps_.end(),
                  "DynamicCheckpointing: TakeHandle requested a non-existent checkpoint.");
      MFEM_VERIFY(storage_->IsValid(it->second.h),
                  "DynamicCheckpointing: TakeHandle requested a checkpoint with no snapshot.");

      Handle h = std::move(it->second.h);
      it->second.h = storage_->InvalidHandle();
      return h;
   }
};

} // namespace mfem

#endif // MFEM_DYNAMIC_CHECKPOINTING_HPP

