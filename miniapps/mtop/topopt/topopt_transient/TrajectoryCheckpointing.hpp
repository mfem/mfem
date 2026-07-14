// =============================================================================
// Trajectory Checkpointing for Transient Topology Optimization
// =============================================================================
//
// Wraps REVOLVE checkpointing algorithm for RK4 forward/adjoint sweeps.
//
// MEMORY SAVINGS:
//   Before: N timesteps × state_size × 8 bytes (full storage)
//   After:  C checkpoints × state_size × 8 bytes (C << N, typically 10-20)
//
// COST:
//   Forward re-evaluations during adjoint: ~2-3× original forward cost
//   (REVOLVE minimizes this for fixed checkpoint count)
//
// USAGE:
//   1. Forward sweep:
//      for (int i = 0; i < num_steps; i++)
//          checkpoint.ForwardStep(i, x, [&](int i, Vector& x) { RK4_step(x); });
//
//   2. Backward sweep:
//      for (int i = num_steps-1; i >= 0; i--)
//          checkpoint.BackwardStep(i, lambda, work,
//              [&](int i, Vector& x) { RK4_step(x); },
//              [&](int i, const Vector& x, Vector& lambda) { adjoint_step(x, lambda); });
//
// =============================================================================

#ifndef TRAJECTORY_CHECKPOINTING_HPP
#define TRAJECTORY_CHECKPOINTING_HPP

#include "mfem.hpp"
#include "mtop-chkpt/chpt/revolve_checkpointing.hpp"
#include <vector>
#include <cstring>

namespace mfem
{

// =============================================================================
// RK4 SNAPSHOT STRUCTURE
// =============================================================================
// What to checkpoint: displacement, velocity, time, step index
struct RK4Snapshot
{
   Vector u;          // displacement (dim × num_nodes)
   Vector v;          // velocity (dim × num_nodes)
   real_t time;       // current time t_i
   int step_index;    // step i (for validation)

   RK4Snapshot() : time(0.0), step_index(-1) {}

   RK4Snapshot(int state_size) : u(state_size/2), v(state_size/2),
                                 time(0.0), step_index(-1) {}

   void SetSize(int state_size)
   {
      u.SetSize(state_size / 2);
      v.SetSize(state_size / 2);
   }

   // Pack/unpack for REVOLVE binary storage
   static size_t ByteSize(int state_size)
   {
      return state_size * sizeof(real_t) +  // u + v
             sizeof(real_t) +                // time
             sizeof(int);                    // step_index
   }

   void Serialize(uint8_t *buffer, size_t buffer_size) const
   {
      const int half_size = u.Size();
      const size_t expected_size = ByteSize(2 * half_size);
      MFEM_VERIFY(buffer_size >= expected_size,
                  "RK4Snapshot::Serialize: buffer too small");

      uint8_t *ptr = buffer;

      // Pack u
      std::memcpy(ptr, u.GetData(), half_size * sizeof(real_t));
      ptr += half_size * sizeof(real_t);

      // Pack v
      std::memcpy(ptr, v.GetData(), half_size * sizeof(real_t));
      ptr += half_size * sizeof(real_t);

      // Pack time
      std::memcpy(ptr, &time, sizeof(real_t));
      ptr += sizeof(real_t);

      // Pack step_index
      std::memcpy(ptr, &step_index, sizeof(int));
   }

   void Deserialize(const uint8_t *buffer, size_t buffer_size)
   {
      const int half_size = u.Size();
      const size_t expected_size = ByteSize(2 * half_size);
      MFEM_VERIFY(buffer_size >= expected_size,
                  "RK4Snapshot::Deserialize: buffer too small");

      const uint8_t *ptr = buffer;

      // Unpack u
      std::memcpy(u.GetData(), ptr, half_size * sizeof(real_t));
      ptr += half_size * sizeof(real_t);

      // Unpack v
      std::memcpy(v.GetData(), ptr, half_size * sizeof(real_t));
      ptr += half_size * sizeof(real_t);

      // Unpack time
      std::memcpy(&time, ptr, sizeof(real_t));
      ptr += sizeof(real_t);

      // Unpack step_index
      std::memcpy(&step_index, ptr, sizeof(int));
   }
};

// =============================================================================
// TRAJECTORY CHECKPOINTING WRAPPER
// =============================================================================
// Template parameter: StorageT = FixedSlotMemoryStorage (default)
//                               or FixedSlotFileStorage (for extreme scale)
template <typename StorageT = FixedSlotMemoryStorage>
class TrajectoryCheckpointing
{
private:
   int num_steps_;             // Total RK4 steps
   int num_checkpoints_;       // REVOLVE "snaps" (user-specified)
   int state_size_;            // Size of [u, v] concatenated
   size_t snapshot_bytes_;     // Bytes per checkpoint

   StorageT storage_;
   std::unique_ptr<FixedStepRevolveCheckpointing<StorageT>> revolve_;

   RK4Snapshot work_snapshot_; // Scratch space for serialization

public:
   TrajectoryCheckpointing(int num_steps, int num_checkpoints, int state_size)
      : num_steps_(num_steps),
        num_checkpoints_(num_checkpoints),
        state_size_(state_size),
        snapshot_bytes_(RK4Snapshot::ByteSize(state_size)),
        storage_(num_checkpoints, snapshot_bytes_),
        work_snapshot_(state_size)
   {
      MFEM_VERIFY(num_steps > 0, "TrajectoryCheckpointing: num_steps must be > 0");
      MFEM_VERIFY(num_checkpoints > 0,
                  "TrajectoryCheckpointing: num_checkpoints must be > 0");
      MFEM_VERIFY(state_size > 0, "TrajectoryCheckpointing: state_size must be > 0");
      MFEM_VERIFY(state_size % 2 == 0,
                  "TrajectoryCheckpointing: state_size must be even (u+v)");

      revolve_ = std::make_unique<FixedStepRevolveCheckpointing<StorageT>>(
         num_steps_, num_checkpoints_, snapshot_bytes_, storage_);
   }

   int NumSteps() const { return num_steps_; }
   int NumCheckpoints() const { return num_checkpoints_; }
   size_t SnapshotBytes() const { return snapshot_bytes_; }

   // Reset for next MMA iteration (clears REVOLVE state)
   void Reset()
   {
      revolve_->Reset();
   }

   // =========================================================================
   // FORWARD STEP
   // =========================================================================
   // Call this for i = 0..num_steps-1 during forward sweep.
   //
   // Arguments:
   //   i             : current step index
   //   x             : [u_i, v_i] state vector (modified in-place to become [u_{i+1}, v_{i+1}])
   //   t             : current time t_i
   //   primal_step   : lambda(int i, Vector& x) that advances x one RK4 step
   //
   // Example:
   //   auto primal = [&](int i, Vector& x) {
   //       real_t dt = h_;
   //       real_t t = i * h_;
   //       solver.Step(x, t, dt);  // MFEM RK4Solver::Step
   //   };
   //   checkpoint.ForwardStep(i, x, t, primal);
   //
   template <typename PrimalStep>
   void ForwardStep(int i, Vector &x, real_t t, PrimalStep &&primal_step)
   {
      MFEM_VERIFY(x.Size() == state_size_,
                  "ForwardStep: state size mismatch");

      // Lambda to serialize state into buffer
      auto make_snapshot = [&](Vector &state, uint8_t *buffer, size_t buffer_size)
      {
         work_snapshot_.SetSize(state_size_);
         const int half = state_size_ / 2;
         work_snapshot_.u.SetDataAndSize(state.GetData(), half);
         work_snapshot_.v.SetDataAndSize(state.GetData() + half, half);
         work_snapshot_.time = t;
         work_snapshot_.step_index = i;
         work_snapshot_.Serialize(buffer, buffer_size);
      };

      // REVOLVE decides when to checkpoint
      revolve_->ForwardStep(i, x, std::forward<PrimalStep>(primal_step),
                            make_snapshot);
   }

   // =========================================================================
   // BACKWARD STEP
   // =========================================================================
   // Call this for i = num_steps-1..0 during adjoint sweep.
   //
   // Arguments:
   //   i             : current step index (decreasing)
   //   lambda        : adjoint state (modified in-place: lambda^{i+1} → lambda^{i})
   //   u_work        : scratch state for forward re-evaluation
   //   primal_step   : lambda(int i, Vector& x) that advances x one RK4 step
   //   adjoint_step  : lambda(int i, const Vector& x, Vector& lambda) that does adjoint step
   //
   // Example:
   //   auto primal = [&](int i, Vector& x) {
   //       real_t dt = h_;
   //       real_t t = i * h_;
   //       solver.Step(x, t, dt);
   //   };
   //   auto adjoint = [&](int i, const Vector& x, Vector& lambda) {
   //       real_t t = i * h_;
   //       RK4AdjointOneStep(oper, x, t, h_, lambda, lambda_prev);
   //       lambda = lambda_prev;
   //   };
   //   checkpoint.BackwardStep(i, lambda, work, primal, adjoint);
   //
   template <typename PrimalStep, typename AdjointStep>
   void BackwardStep(int i, Vector &lambda, Vector &u_work,
                     PrimalStep &&primal_step,
                     AdjointStep &&adjoint_step)
   {
      MFEM_VERIFY(lambda.Size() == state_size_,
                  "BackwardStep: adjoint size mismatch");
      MFEM_VERIFY(u_work.Size() == state_size_,
                  "BackwardStep: work state size mismatch");

      // Lambda to serialize state into buffer
      auto make_snapshot = [&](Vector &state, uint8_t *buffer, size_t buffer_size)
      {
         work_snapshot_.SetSize(state_size_);
         const int half = state_size_ / 2;
         work_snapshot_.u.SetDataAndSize(state.GetData(), half);
         work_snapshot_.v.SetDataAndSize(state.GetData() + half, half);
         work_snapshot_.time = i * (1.0 / num_steps_); // placeholder (REVOLVE doesn't use time in backward)
         work_snapshot_.step_index = i;
         work_snapshot_.Serialize(buffer, buffer_size);
      };

      // Lambda to deserialize buffer into state
      auto restore_snapshot = [&](Vector &state, const uint8_t *buffer,
                                  size_t buffer_size)
      {
         work_snapshot_.SetSize(state_size_);
         work_snapshot_.Deserialize(buffer, buffer_size);
         const int half = state_size_ / 2;
         std::memcpy(state.GetData(), work_snapshot_.u.GetData(),
                     half * sizeof(real_t));
         std::memcpy(state.GetData() + half, work_snapshot_.v.GetData(),
                     half * sizeof(real_t));
      };

      // REVOLVE orchestrates restore/recompute/adjoint
      revolve_->BackwardStep(i, lambda, u_work,
                             std::forward<PrimalStep>(primal_step),
                             std::forward<AdjointStep>(adjoint_step),
                             make_snapshot, restore_snapshot);
   }

   // =========================================================================
   // MEMORY FOOTPRINT ESTIMATE
   // =========================================================================
   size_t MemoryFootprintBytes() const
   {
      return num_checkpoints_ * snapshot_bytes_;
   }

   real_t MemoryFootprintMB() const
   {
      return static_cast<real_t>(MemoryFootprintBytes()) / (1024.0 * 1024.0);
   }

   // =========================================================================
   // REVOLVE STATISTICS
   // =========================================================================
   // Estimate forward re-evaluations during adjoint sweep
   // (REVOLVE minimizes this, typically ~2-3× forward cost)
   int EstimateRecomputations() const
   {
      // Rough estimate: num_steps * log(num_steps / num_checkpoints)
      // Exact value depends on REVOLVE schedule (binomial recursion)
      if (num_checkpoints_ >= num_steps_) { return 0; }
      const real_t ratio = static_cast<real_t>(num_steps_) / num_checkpoints_;
      return static_cast<int>(num_steps_ * std::log(ratio));
   }

   void PrintInfo(MPI_Comm comm = MPI_COMM_WORLD) const
   {
      if (Mpi::Root())
      {
         mfem::out << "\n╔═══════════════════════════════════════════════════════════╗\n"
                   << "║  TRAJECTORY CHECKPOINTING (REVOLVE)                   ║\n"
                   << "╠═══════════════════════════════════════════════════════════╣\n"
                   << "║  Timesteps:          " << std::setw(10) << num_steps_
                   << "                              ║\n"
                   << "║  Checkpoints:        " << std::setw(10) << num_checkpoints_
                   << "                              ║\n"
                   << "║  State size:         " << std::setw(10) << state_size_
                   << " DOFs                        ║\n"
                   << "║  Snapshot size:      " << std::setw(10)
                   << std::fixed << std::setprecision(2) << MemoryFootprintMB()
                   << " MB                          ║\n"
                   << "║  Est. recompute:     " << std::setw(10)
                   << EstimateRecomputations()
                   << " steps (~"
                   << std::fixed << std::setprecision(1)
                   << (num_checkpoints_ > 0 ?
                       static_cast<real_t>(EstimateRecomputations()) / num_steps_ :
                       0.0)
                   << "× forward)        ║\n"
                   << "╚═══════════════════════════════════════════════════════════╝\n";
      }
   }
};

// =============================================================================
// HELPER: AUTO-SIZE CHECKPOINT COUNT
// =============================================================================
// Heuristic: C = max(10, min(50, N / 100))
// User can override via command-line flag.
inline int AutoCheckpointCount(int num_steps, real_t memory_budget_mb = 0.0,
                               int state_size = 0)
{
   int auto_count = std::max(10, std::min(50, num_steps / 100));

   // If memory budget provided, respect it
   if (memory_budget_mb > 0.0 && state_size > 0)
   {
      const size_t snapshot_bytes = RK4Snapshot::ByteSize(state_size);
      const int budget_count =
         static_cast<int>((memory_budget_mb * 1024.0 * 1024.0) / snapshot_bytes);
      auto_count = std::min(auto_count, budget_count);
   }

   return std::max(1, auto_count);  // At least 1 checkpoint
}

} // namespace mfem

#endif // TRAJECTORY_CHECKPOINTING_HPP
