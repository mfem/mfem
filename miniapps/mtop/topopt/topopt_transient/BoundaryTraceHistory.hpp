// =============================================================================
// Boundary trace history for time-domain inverse problems
// =============================================================================

#ifndef BOUNDARY_TRACE_HISTORY_HPP
#define BOUNDARY_TRACE_HISTORY_HPP

#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace mfem
{

/**
 * Read-mostly storage for displacement observations on a selected exterior
 * boundary.  Samples live at the half-step times t_j = j * coarse_dt / 2,
 * j=0,...,2N.  Only local vector DOFs touched by the selected boundary are
 * retained; GetSample() reconstructs a zero-interior ParGridFunction for
 * quadrature evaluation.
 *
 * The history intentionally uses the reconstruction finite element space.
 * A higher-order reference generator must project/restrict its displacement to
 * that space before calling StoreSample().  This makes it impossible to
 * silently compare coefficient vectors from incompatible spaces.
 */
class BoundaryTraceHistory
{
private:
   ParFiniteElementSpace *fespace_;
   MPI_Comm comm_;
   real_t coarse_dt_;
   int coarse_steps_;
   Array<int> observation_marker_;
   Array<int> trace_vdofs_;
   std::vector<Vector> samples_;
   std::vector<unsigned char> sample_is_set_;
   mutable ParGridFunction reconstructed_sample_;
   mutable int reconstructed_index_;
   long fespace_sequence_;
   double local_memory_bytes_;
   double global_memory_bytes_;

   static int UnsignedVDof(int vdof)
   {
      return vdof >= 0 ? vdof : -1 - vdof;
   }

   void CheckIndex(int half_step_index) const
   {
      MFEM_VERIFY(half_step_index >= 0 &&
                  half_step_index < static_cast<int>(samples_.size()),
                  "BoundaryTraceHistory half-step index " << half_step_index
                  << " is outside [0," << samples_.size() - 1 << "].");
   }

   void CheckSpaceUnchanged() const
   {
      MFEM_VERIFY(fespace_->GetSequence() == fespace_sequence_,
                  "BoundaryTraceHistory finite element space changed after "
                  "the trace layout was constructed.");
   }

public:
   BoundaryTraceHistory(ParFiniteElementSpace *fespace,
                        const Array<int> &observation_marker,
                        real_t coarse_dt, int coarse_steps)
      : fespace_(fespace),
        comm_(fespace ? fespace->GetComm() : MPI_COMM_NULL),
        coarse_dt_(coarse_dt),
        coarse_steps_(coarse_steps),
        reconstructed_sample_(fespace),
        reconstructed_index_(-1),
        fespace_sequence_(fespace ? fespace->GetSequence() : -1),
        local_memory_bytes_(0.0),
        global_memory_bytes_(0.0)
   {
      MFEM_VERIFY(fespace_,
                  "BoundaryTraceHistory requires a finite element space.");
      MFEM_VERIFY(std::isfinite(coarse_dt_) && coarse_dt_ > 0.0,
                  "BoundaryTraceHistory requires a positive coarse time step.");
      MFEM_VERIFY(coarse_steps_ > 0 &&
                  coarse_steps_ <= (std::numeric_limits<int>::max() - 1) / 2,
                  "BoundaryTraceHistory has an invalid coarse step count.");

      ParMesh *pmesh = fespace_->GetParMesh();
      MFEM_VERIFY(pmesh,
                  "BoundaryTraceHistory requires a parallel mesh.");
      const int max_bdr_attr = pmesh->bdr_attributes.Size() ?
                               pmesh->bdr_attributes.Max() : 0;
      MFEM_VERIFY(max_bdr_attr > 0,
                  "BoundaryTraceHistory mesh has no boundary attributes.");
      MFEM_VERIFY(observation_marker.Size() == max_bdr_attr,
                  "BoundaryTraceHistory observation marker has size "
                  << observation_marker.Size() << "; expected "
                  << max_bdr_attr << ".");

      observation_marker_ = observation_marker;
      int local_marked_attributes = 0;
      for (int i = 0; i < observation_marker_.Size(); i++)
      {
         MFEM_VERIFY(observation_marker_[i] == 0 ||
                     observation_marker_[i] == 1,
                     "BoundaryTraceHistory marker entries must be zero or one.");
         local_marked_attributes += observation_marker_[i];
      }
      MFEM_VERIFY(local_marked_attributes > 0,
                  "BoundaryTraceHistory observation marker is empty.");

      Array<int> is_trace_vdof(fespace_->GetVSize());
      is_trace_vdof = 0;
      int local_observation_elements = 0;
      Array<int> vdofs;
      for (int be = 0; be < pmesh->GetNBE(); be++)
      {
         const int attribute = pmesh->GetBdrAttribute(be);
         if (observation_marker_[attribute - 1] == 0) { continue; }
         local_observation_elements++;
         fespace_->GetBdrElementVDofs(be, vdofs);
         for (int i = 0; i < vdofs.Size(); i++)
         {
            const int vdof = UnsignedVDof(vdofs[i]);
            MFEM_ASSERT(vdof >= 0 && vdof < is_trace_vdof.Size(),
                        "Invalid boundary vector DOF.");
            is_trace_vdof[vdof] = 1;
         }
      }

      int global_observation_elements = 0;
      MPI_Allreduce(&local_observation_elements, &global_observation_elements,
                    1, MPI_INT, MPI_SUM, comm_);
      MFEM_VERIFY(global_observation_elements > 0,
                  "BoundaryTraceHistory marker selects no boundary elements.");

      for (int vdof = 0; vdof < is_trace_vdof.Size(); vdof++)
      {
         if (is_trace_vdof[vdof]) { trace_vdofs_.Append(vdof); }
      }

      const int sample_count = 2 * coarse_steps_ + 1;
      samples_.resize(sample_count);
      sample_is_set_.assign(sample_count, 0);
      for (Vector &sample : samples_)
      {
         sample.SetSize(trace_vdofs_.Size());
      }
      reconstructed_sample_ = 0.0;

      local_memory_bytes_ = static_cast<double>(sample_count) *
                            static_cast<double>(trace_vdofs_.Size()) *
                            static_cast<double>(sizeof(real_t));
      MPI_Allreduce(&local_memory_bytes_, &global_memory_bytes_, 1,
                    MPI_DOUBLE, MPI_SUM, comm_);
   }

   ParFiniteElementSpace *FESpace() const { return fespace_; }
   const Array<int> &ObservationMarker() const { return observation_marker_; }
   real_t CoarseTimeStep() const { return coarse_dt_; }
   int CoarseStepCount() const { return coarse_steps_; }
   int SampleCount() const { return static_cast<int>(samples_.size()); }
   int LocalTraceVDofCount() const { return trace_vdofs_.Size(); }
   double EstimatedLocalMemoryBytes() const { return local_memory_bytes_; }
   double EstimatedGlobalMemoryBytes() const { return global_memory_bytes_; }

   int StoredSampleCount() const
   {
      int count = 0;
      for (unsigned char is_set : sample_is_set_) { count += is_set != 0; }
      return count;
   }

   int HalfStepIndex(real_t time) const
   {
      MFEM_VERIFY(std::isfinite(time),
                  "BoundaryTraceHistory received a non-finite time.");
      const real_t scaled_time = 2.0 * time / coarse_dt_;
      const real_t tolerance =
         512.0 * std::numeric_limits<real_t>::epsilon() *
         std::max(real_t(1.0), std::abs(scaled_time));
      MFEM_VERIFY(scaled_time >= -tolerance &&
                  scaled_time <= static_cast<real_t>(samples_.size() - 1) +
                                 tolerance,
                  "BoundaryTraceHistory time " << time
                  << " is outside the stored interval [0,"
                  << coarse_steps_ * coarse_dt_ << "].");
      const long long rounded = std::llround(scaled_time);
      MFEM_VERIFY(std::abs(scaled_time - static_cast<real_t>(rounded)) <=
                  tolerance,
                  "BoundaryTraceHistory time " << time
                  << " is not on the coarse half-step grid (dt="
                  << coarse_dt_ << ").");
      MFEM_VERIFY(rounded >= 0 &&
                  rounded < static_cast<long long>(samples_.size()),
                  "BoundaryTraceHistory time " << time
                  << " is outside the stored interval [0,"
                  << coarse_steps_ * coarse_dt_ << "].");
      return static_cast<int>(rounded);
   }

   void StoreSample(int half_step_index, const ParGridFunction &trace)
   {
      CheckIndex(half_step_index);
      CheckSpaceUnchanged();
      MFEM_VERIFY(trace.ParFESpace() == fespace_,
                  "BoundaryTraceHistory samples must be projected to the "
                  "reconstruction state space before storage.");
      MFEM_VERIFY(!sample_is_set_[half_step_index],
                  "BoundaryTraceHistory sample " << half_step_index
                  << " was stored more than once.");

      Vector &sample = samples_[half_step_index];
      int local_nonfinite = 0;
      for (int i = 0; i < trace_vdofs_.Size(); i++)
      {
         sample[i] = trace[trace_vdofs_[i]];
         if (!std::isfinite(sample[i])) { local_nonfinite = 1; }
      }
      int global_nonfinite = 0;
      MPI_Allreduce(&local_nonfinite, &global_nonfinite, 1, MPI_INT, MPI_MAX,
                    comm_);
      MFEM_VERIFY(global_nonfinite == 0,
                  "BoundaryTraceHistory refuses a non-finite trace sample.");
      sample_is_set_[half_step_index] = 1;
   }

   void StoreSampleAtTime(real_t time, const ParGridFunction &trace)
   {
      StoreSample(HalfStepIndex(time), trace);
   }

   const ParGridFunction &GetSample(int half_step_index) const
   {
      CheckIndex(half_step_index);
      CheckSpaceUnchanged();
      MFEM_VERIFY(sample_is_set_[half_step_index],
                  "BoundaryTraceHistory sample " << half_step_index
                  << " is missing.");

      if (reconstructed_index_ != half_step_index)
      {
         reconstructed_sample_ = 0.0;
         const Vector &sample = samples_[half_step_index];
         for (int i = 0; i < trace_vdofs_.Size(); i++)
         {
            reconstructed_sample_[trace_vdofs_[i]] = sample[i];
         }
         reconstructed_index_ = half_step_index;
      }
      return reconstructed_sample_;
   }

   const ParGridFunction &GetSampleAtTime(real_t time) const
   {
      return GetSample(HalfStepIndex(time));
   }

   void ValidateComplete() const
   {
      MFEM_VERIFY(StoredSampleCount() == SampleCount(),
                  "BoundaryTraceHistory is incomplete: stored "
                  << StoredSampleCount() << " of " << SampleCount()
                  << " samples.");
   }
};

} // namespace mfem

#endif // BOUNDARY_TRACE_HISTORY_HPP
