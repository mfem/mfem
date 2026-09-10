// =============================================================================
// High-order synthetic boundary data for elastic inclusion identification
// =============================================================================

#ifndef REFERENCE_BOUNDARY_DATA_HPP
#define REFERENCE_BOUNDARY_DATA_HPP

#include "ElastodynamicsSolver.hpp"
#include "BoundaryTraceHistory.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <sys/stat.h>

namespace mfem
{

struct ReferenceBoundaryDataMetadata
{
   int state_order = 0;
   int reference_steps = 0;
   int reference_steps_per_half_step = 0;
   real_t requested_time_step = 0.0;
   real_t effective_time_step = 0.0;
   MassSolverType mass_solver_type = MassSolverType::ITERATIVE;
   bool matrix_free_symplectic_euler = false;
   bool damping_enabled = true;
   HYPRE_BigInt global_state_true_dofs = 0;
   double local_trace_memory_bytes = 0.0;
   double maximum_trace_memory_bytes_per_rank = 0.0;
   double global_trace_memory_bytes = 0.0;
   double forward_seconds = 0.0;
   bool loaded_from_cache = false;
};

/// Invariant configuration of a reusable synthetic-observation cache.  The
/// cache is valid only for exactly the same distributed reconstruction space,
/// acquisition count, enriched time grid, mass/integrator route, and damping.
struct ReferenceTraceCacheSignature
{
   int format_version = 1;
   int n_mpi_ranks = 1;
   int number_of_sources = 1;
   int reconstruction_state_order = 1;
   long long reconstruction_state_true_dofs = 0;
   int coarse_steps = 0;
   real_t coarse_dt = 0.0;
   int reference_state_order = 0;
   int reference_steps = 0;
   int reference_steps_per_half_step = 0;
   real_t requested_reference_dt = 0.0;
   real_t effective_reference_dt = 0.0;
   int mass_solver_type = 0;
   bool matrix_free_symplectic_euler = false;
   bool damping_enabled = true;
   long long reference_state_true_dofs = 0;
};

enum class ReferenceTraceCacheLoadResult
{
   MISSING,
   LOADED,
   INCOMPATIBLE,
   IO_ERROR
};

/**
 * Immutable, rank-local persistence for the projected receiver histories.
 *
 * A trace cache avoids repeating the expensive enriched synthetic forward
 * solves when a checkpointed inverse optimization is restarted.  We retain one
 * compact binary payload per rank (all shots in that file), and commit its
 * human-readable metadata only after every rank has atomically replaced its
 * payload.  This is deliberately analogous to OptimizationCheckpoint, but it
 * stores observations rather than design variables and is never modified by
 * an MMA update.
 */
class ReferenceTraceCache
{
private:
   static constexpr int32_t cache_magic_ = 0x52544331; // "RTC1"
   static constexpr int32_t cache_format_version_ = 1;

   std::string directory_;
   MPI_Comm comm_;
   int rank_ = 0;
   int ranks_ = 1;

   std::string MetadataPath() const { return directory_ + "/metadata.txt"; }

   std::string PayloadPath(int rank) const
   {
      std::ostringstream name;
      name << directory_ << "/trace_rank" << std::setfill('0')
           << std::setw(6) << rank << ".bin";
      return name.str();
   }

   bool AllOk(bool local_ok) const
   {
      int local = local_ok ? 1 : 0;
      int global = 0;
      MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, comm_);
      return global == 1;
   }

   static bool NearlyEqual(real_t first, real_t second)
   {
      const real_t scale =
         std::max({real_t(1.0), std::abs(first), std::abs(second)});
      return std::abs(first - second) <=
             real_t(2048.0) * std::numeric_limits<real_t>::epsilon() * scale;
   }

   bool CreateDirectoryIfNeeded() const
   {
      bool ok = true;
      if (rank_ == 0)
      {
         struct stat st;
         if (stat(directory_.c_str(), &st) != 0)
         {
            ok = mkdir(directory_.c_str(), 0755) == 0;
            if (!ok)
            {
               std::cerr << "Reference trace cache: failed to create "
                         << directory_ << std::endl;
            }
         }
         else
         {
            ok = S_ISDIR(st.st_mode);
         }
      }
      MPI_Bcast(&ok, 1, MPI_C_BOOL, 0, comm_);
      return ok;
   }

   bool ReadMetadata(ReferenceTraceCacheSignature &signature) const
   {
      bool read_ok = true;
      if (rank_ == 0)
      {
         std::ifstream stream(MetadataPath());
         read_ok = stream.good();
         std::string key;
         while (read_ok && stream >> key)
         {
            if (key == "format_version") { stream >> signature.format_version; }
            else if (key == "n_mpi_ranks") { stream >> signature.n_mpi_ranks; }
            else if (key == "number_of_sources")
            {
               stream >> signature.number_of_sources;
            }
            else if (key == "reconstruction_state_order")
            {
               stream >> signature.reconstruction_state_order;
            }
            else if (key == "reconstruction_state_true_dofs")
            {
               stream >> signature.reconstruction_state_true_dofs;
            }
            else if (key == "coarse_steps") { stream >> signature.coarse_steps; }
            else if (key == "coarse_dt") { stream >> signature.coarse_dt; }
            else if (key == "reference_state_order")
            {
               stream >> signature.reference_state_order;
            }
            else if (key == "reference_steps")
            {
               stream >> signature.reference_steps;
            }
            else if (key == "reference_steps_per_half_step")
            {
               stream >> signature.reference_steps_per_half_step;
            }
            else if (key == "requested_reference_dt")
            {
               stream >> signature.requested_reference_dt;
            }
            else if (key == "effective_reference_dt")
            {
               stream >> signature.effective_reference_dt;
            }
            else if (key == "mass_solver_type")
            {
               stream >> signature.mass_solver_type;
            }
            else if (key == "matrix_free_symplectic_euler")
            {
               int value = 0;
               stream >> value;
               signature.matrix_free_symplectic_euler = value != 0;
            }
            else if (key == "damping_enabled")
            {
               int value = 0;
               stream >> value;
               signature.damping_enabled = value != 0;
            }
            else if (key == "reference_state_true_dofs")
            {
               stream >> signature.reference_state_true_dofs;
            }
            else
            {
               std::string ignored;
               std::getline(stream, ignored);
            }
            read_ok = !stream.fail();
         }
         read_ok = read_ok && signature.format_version == cache_format_version_ &&
                   signature.n_mpi_ranks > 0 &&
                   signature.number_of_sources > 0 &&
                   signature.coarse_steps > 0 &&
                   signature.reference_state_order > 0 &&
                   signature.reference_steps > 0 &&
                   signature.reference_steps_per_half_step > 0 &&
                   std::isfinite(signature.coarse_dt) &&
                   std::isfinite(signature.requested_reference_dt) &&
                   std::isfinite(signature.effective_reference_dt);
      }
      MPI_Bcast(&read_ok, 1, MPI_C_BOOL, 0, comm_);
      if (!read_ok) { return false; }

      MPI_Bcast(&signature.format_version, 1, MPI_INT, 0, comm_);
      MPI_Bcast(&signature.n_mpi_ranks, 1, MPI_INT, 0, comm_);
      MPI_Bcast(&signature.number_of_sources, 1, MPI_INT, 0, comm_);
      MPI_Bcast(&signature.reconstruction_state_order, 1, MPI_INT, 0, comm_);
      MPI_Bcast(&signature.reconstruction_state_true_dofs, 1,
                MPI_LONG_LONG, 0, comm_);
      MPI_Bcast(&signature.coarse_steps, 1, MPI_INT, 0, comm_);
      MPI_Bcast(&signature.coarse_dt, 1, MPITypeMap<real_t>::mpi_type, 0,
                comm_);
      MPI_Bcast(&signature.reference_state_order, 1, MPI_INT, 0, comm_);
      MPI_Bcast(&signature.reference_steps, 1, MPI_INT, 0, comm_);
      MPI_Bcast(&signature.reference_steps_per_half_step, 1, MPI_INT, 0,
                comm_);
      MPI_Bcast(&signature.requested_reference_dt, 1,
                MPITypeMap<real_t>::mpi_type, 0, comm_);
      MPI_Bcast(&signature.effective_reference_dt, 1,
                MPITypeMap<real_t>::mpi_type, 0, comm_);
      MPI_Bcast(&signature.mass_solver_type, 1, MPI_INT, 0, comm_);
      MPI_Bcast(&signature.matrix_free_symplectic_euler, 1, MPI_C_BOOL, 0,
                comm_);
      MPI_Bcast(&signature.damping_enabled, 1, MPI_C_BOOL, 0, comm_);
      MPI_Bcast(&signature.reference_state_true_dofs, 1, MPI_LONG_LONG, 0,
                comm_);
      return true;
   }

   bool WriteMetadata(const ReferenceTraceCacheSignature &signature) const
   {
      bool ok = true;
      if (rank_ == 0)
      {
         const std::string temporary = MetadataPath() + ".tmp";
         std::ofstream stream(temporary, std::ios::trunc);
         ok = stream.good();
         if (ok)
         {
            stream << "format_version " << cache_format_version_ << "\n"
                   << "n_mpi_ranks " << signature.n_mpi_ranks << "\n"
                   << "number_of_sources " << signature.number_of_sources << "\n"
                   << "reconstruction_state_order "
                   << signature.reconstruction_state_order << "\n"
                   << "reconstruction_state_true_dofs "
                   << signature.reconstruction_state_true_dofs << "\n"
                   << "coarse_steps " << signature.coarse_steps << "\n"
                   << std::setprecision(17)
                   << "coarse_dt " << signature.coarse_dt << "\n"
                   << "reference_state_order "
                   << signature.reference_state_order << "\n"
                   << "reference_steps " << signature.reference_steps << "\n"
                   << "reference_steps_per_half_step "
                   << signature.reference_steps_per_half_step << "\n"
                   << "requested_reference_dt "
                   << signature.requested_reference_dt << "\n"
                   << "effective_reference_dt "
                   << signature.effective_reference_dt << "\n"
                   << "mass_solver_type " << signature.mass_solver_type << "\n"
                   << "matrix_free_symplectic_euler "
                   << (signature.matrix_free_symplectic_euler ? 1 : 0) << "\n"
                   << "damping_enabled "
                   << (signature.damping_enabled ? 1 : 0) << "\n"
                   << "reference_state_true_dofs "
                   << signature.reference_state_true_dofs << "\n";
            stream.close();
            ok = stream.good() &&
                 std::rename(temporary.c_str(), MetadataPath().c_str()) == 0;
         }
      }
      MPI_Bcast(&ok, 1, MPI_C_BOOL, 0, comm_);
      return ok;
   }

   static bool Compatible(const ReferenceTraceCacheSignature &cached,
                          const ReferenceTraceCacheSignature &expected)
   {
      return cached.n_mpi_ranks == expected.n_mpi_ranks &&
             cached.number_of_sources == expected.number_of_sources &&
             cached.reconstruction_state_order ==
                expected.reconstruction_state_order &&
             cached.reconstruction_state_true_dofs ==
                expected.reconstruction_state_true_dofs &&
             cached.coarse_steps == expected.coarse_steps &&
             NearlyEqual(cached.coarse_dt, expected.coarse_dt) &&
             cached.reference_state_order == expected.reference_state_order &&
             cached.reference_steps == expected.reference_steps &&
             cached.reference_steps_per_half_step ==
                expected.reference_steps_per_half_step &&
             NearlyEqual(cached.requested_reference_dt,
                         expected.requested_reference_dt) &&
             NearlyEqual(cached.effective_reference_dt,
                         expected.effective_reference_dt) &&
             cached.mass_solver_type == expected.mass_solver_type &&
             cached.matrix_free_symplectic_euler ==
                expected.matrix_free_symplectic_euler &&
             cached.damping_enabled == expected.damping_enabled;
   }

public:
   ReferenceTraceCache(const std::string &directory, MPI_Comm comm)
      : directory_(directory), comm_(comm)
   {
      MPI_Comm_rank(comm_, &rank_);
      MPI_Comm_size(comm_, &ranks_);
   }

   ReferenceTraceCacheLoadResult Load(
      const ReferenceTraceCacheSignature &expected,
      ParFiniteElementSpace &state_fes,
      const std::vector<Array<int>> &observation_markers,
      std::vector<std::shared_ptr<const BoundaryTraceHistory>> &histories,
      std::vector<ReferenceBoundaryDataMetadata> &metadata) const
   {
      bool exists = false;
      if (rank_ == 0)
      {
         std::ifstream test(MetadataPath());
         exists = test.good();
      }
      MPI_Bcast(&exists, 1, MPI_C_BOOL, 0, comm_);
      if (!exists) { return ReferenceTraceCacheLoadResult::MISSING; }

      ReferenceTraceCacheSignature cached;
      if (!ReadMetadata(cached))
      {
         if (rank_ == 0)
         {
            std::cerr << "Reference trace cache: cannot parse "
                      << MetadataPath() << std::endl;
         }
         return ReferenceTraceCacheLoadResult::IO_ERROR;
      }
      if (!Compatible(cached, expected) ||
          static_cast<int>(observation_markers.size()) !=
             expected.number_of_sources)
      {
         if (rank_ == 0)
         {
            std::cerr << "Reference trace cache is incompatible with the "
                      << "current inverse acquisition; refusing reuse."
                      << std::endl;
         }
         return ReferenceTraceCacheLoadResult::INCOMPATIBLE;
      }

      std::ifstream stream(PayloadPath(rank_), std::ios::binary);
      int32_t magic = 0, format = 0, stored_rank = -1, source_count = -1;
      bool header_ok = stream.good();
      if (header_ok)
      {
         stream.read(reinterpret_cast<char *>(&magic), sizeof(magic));
         stream.read(reinterpret_cast<char *>(&format), sizeof(format));
         stream.read(reinterpret_cast<char *>(&stored_rank), sizeof(stored_rank));
         stream.read(reinterpret_cast<char *>(&source_count), sizeof(source_count));
         header_ok = stream.good() && magic == cache_magic_ &&
                     format == cache_format_version_ && stored_rank == rank_ &&
                     source_count == expected.number_of_sources;
      }
      if (!AllOk(header_ok))
      {
         if (rank_ == 0)
         {
            std::cerr << "Reference trace cache: missing or invalid rank-local "
                      << "payload(s)." << std::endl;
         }
         return ReferenceTraceCacheLoadResult::IO_ERROR;
      }

      histories.assign(expected.number_of_sources, nullptr);
      metadata.assign(expected.number_of_sources,
                      ReferenceBoundaryDataMetadata{});
      for (int source = 0; source < expected.number_of_sources; source++)
      {
         auto history = std::make_shared<BoundaryTraceHistory>(
            &state_fes, observation_markers[source], expected.coarse_dt,
            expected.coarse_steps);
         const bool payload_ok = history->ReadBinary(stream);
         if (!AllOk(payload_ok))
         {
            if (rank_ == 0)
            {
               std::cerr << "Reference trace cache: invalid data for source "
                         << source << "." << std::endl;
            }
            return ReferenceTraceCacheLoadResult::IO_ERROR;
         }

         ReferenceBoundaryDataMetadata source_metadata;
         source_metadata.state_order = expected.reference_state_order;
         source_metadata.reference_steps = expected.reference_steps;
         source_metadata.reference_steps_per_half_step =
            expected.reference_steps_per_half_step;
         source_metadata.requested_time_step = expected.requested_reference_dt;
         source_metadata.effective_time_step = expected.effective_reference_dt;
         source_metadata.mass_solver_type = static_cast<MassSolverType>(
            expected.mass_solver_type);
         source_metadata.matrix_free_symplectic_euler =
            expected.matrix_free_symplectic_euler;
         source_metadata.damping_enabled = expected.damping_enabled;
         source_metadata.global_state_true_dofs =
            static_cast<HYPRE_BigInt>(cached.reference_state_true_dofs);
         source_metadata.local_trace_memory_bytes =
            history->EstimatedLocalMemoryBytes();
         source_metadata.global_trace_memory_bytes =
            history->EstimatedGlobalMemoryBytes();
         MPI_Allreduce(&source_metadata.local_trace_memory_bytes,
                       &source_metadata.maximum_trace_memory_bytes_per_rank,
                       1, MPI_DOUBLE, MPI_MAX, comm_);
         source_metadata.forward_seconds = 0.0;
         source_metadata.loaded_from_cache = true;
         histories[source] = history;
         metadata[source] = source_metadata;
      }
      return ReferenceTraceCacheLoadResult::LOADED;
   }

   bool Save(const ReferenceTraceCacheSignature &signature,
             const std::vector<std::shared_ptr<const BoundaryTraceHistory>>
                &histories) const
   {
      const bool structure_ok =
         signature.format_version == cache_format_version_ &&
         signature.n_mpi_ranks == ranks_ &&
         signature.number_of_sources > 0 &&
         static_cast<int>(histories.size()) == signature.number_of_sources;
      if (!AllOk(structure_ok)) { return false; }
      if (!CreateDirectoryIfNeeded()) { return false; }

      bool payload_ok = true;
      const std::string payload = PayloadPath(rank_);
      const std::string temporary = payload + ".tmp";
      std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
      payload_ok = stream.good();
      const int32_t source_count = signature.number_of_sources;
      if (payload_ok)
      {
         const int32_t rank_value = rank_;
         const int32_t magic = cache_magic_;
         const int32_t format = cache_format_version_;
         stream.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
         stream.write(reinterpret_cast<const char *>(&format), sizeof(format));
         stream.write(reinterpret_cast<const char *>(&rank_value),
                      sizeof(rank_value));
         stream.write(reinterpret_cast<const char *>(&source_count),
                      sizeof(source_count));
         for (const auto &history : histories)
         {
            payload_ok = history && history->WriteBinary(stream);
            if (!payload_ok) { break; }
         }
         stream.close();
         payload_ok = payload_ok && stream.good() &&
                      std::rename(temporary.c_str(), payload.c_str()) == 0;
      }
      if (!AllOk(payload_ok))
      {
         if (rank_ == 0)
         {
            std::cerr << "Reference trace cache: payload write failed; "
                      << "metadata was not committed." << std::endl;
         }
         return false;
      }
      return WriteMetadata(signature);
   }
};

/// Space-time receiver-norm comparison of two histories on the same
/// reconstruction space and coarse half-step grid.
struct BoundaryTraceComparisonMetrics
{
   real_t difference_norm = 0.0;
   real_t reference_norm = 0.0;
   real_t relative_error = 0.0;
};

/**
 * Compare two complete boundary histories in the measurement norm
 *
 *   ||v||^2 = int_0^T int_Gamma_obs |v(x,t)|^2 ds dt.
 *
 * Both histories must use the same reconstruction finite-element space,
 * observation marker, and coarse half-step grid.  Space is integrated with
 * the same 2*p+2 boundary quadrature used by
 * BoundaryDisplacementTrackingObjective.  Time is integrated by composite
 * Simpson quadrature on the stored half steps.  Equivalently, each coarse
 * interval has the RK4-stage weights dt/6*(1,4,1), after coalescing the two
 * midpoint stages that request the same trace sample.
 *
 * The second history is the denominator/reference.  A zero reference trace
 * is rejected because it cannot define a meaningful relative convergence
 * error.
 */
inline BoundaryTraceComparisonMetrics CompareBoundaryTraceHistories(
   const BoundaryTraceHistory &candidate,
   const BoundaryTraceHistory &reference)
{
   candidate.ValidateComplete();
   reference.ValidateComplete();
   MFEM_VERIFY(candidate.FESpace() == reference.FESpace(),
               "Boundary trace comparison requires the same reconstruction "
               "finite-element space object.");
   MFEM_VERIFY(candidate.CoarseStepCount() == reference.CoarseStepCount() &&
               candidate.SampleCount() == reference.SampleCount(),
               "Boundary trace comparison requires identical time grids.");

   const real_t coarse_dt = candidate.CoarseTimeStep();
   const real_t reference_dt = reference.CoarseTimeStep();
   const real_t dt_scale =
      std::max({real_t(1.0), std::abs(coarse_dt), std::abs(reference_dt)});
   MFEM_VERIFY(std::abs(coarse_dt - reference_dt) <=
               real_t(2048.0) * std::numeric_limits<real_t>::epsilon() *
               dt_scale,
               "Boundary trace comparison requires identical coarse "
               "timesteps.");

   const Array<int> &candidate_marker = candidate.ObservationMarker();
   const Array<int> &reference_marker = reference.ObservationMarker();
   MFEM_VERIFY(candidate_marker.Size() == reference_marker.Size(),
               "Boundary trace comparison received different observation "
               "marker sizes.");
   for (int i = 0; i < candidate_marker.Size(); i++)
   {
      MFEM_VERIFY(candidate_marker[i] == reference_marker[i],
                  "Boundary trace comparison received different observation "
                  "markers.");
   }

   ParFiniteElementSpace *fespace = candidate.FESpace();
   MFEM_VERIFY(fespace,
               "Boundary trace comparison requires a finite-element space.");
   ParMesh *mesh = fespace->GetParMesh();
   MFEM_VERIFY(mesh,
               "Boundary trace comparison requires a parallel mesh.");

   real_t local_difference_squared = 0.0;
   real_t local_reference_squared = 0.0;
   Vector candidate_value, reference_value;
   const int final_sample = candidate.SampleCount() - 1;
   for (int sample = 0; sample <= final_sample; sample++)
   {
      const int simpson_coefficient =
         (sample == 0 || sample == final_sample) ? 1 :
         (sample % 2 == 1 ? 4 : 2);
      const real_t time_weight =
         coarse_dt * static_cast<real_t>(simpson_coefficient) / 6.0;
      const ParGridFunction &candidate_trace = candidate.GetSample(sample);
      const ParGridFunction &reference_trace = reference.GetSample(sample);

      for (int be = 0; be < mesh->GetNBE(); be++)
      {
         const int attribute = mesh->GetBdrAttribute(be);
         if (candidate_marker[attribute - 1] == 0) { continue; }

         const FiniteElement *element = fespace->GetBE(be);
         ElementTransformation *transformation =
            fespace->GetBdrElementTransformation(be);
         const IntegrationRule &rule = IntRules.Get(
            element->GetGeomType(), 2 * element->GetOrder() + 2);
         for (int q = 0; q < rule.GetNPoints(); q++)
         {
            const IntegrationPoint &point = rule.IntPoint(q);
            transformation->SetIntPoint(&point);
            candidate_trace.GetVectorValue(
               *transformation, point, candidate_value);
            reference_trace.GetVectorValue(
               *transformation, point, reference_value);
            candidate_value -= reference_value;
            const real_t weight =
               time_weight * point.weight * transformation->Weight();
            local_difference_squared +=
               weight * (candidate_value * candidate_value);
            local_reference_squared +=
               weight * (reference_value * reference_value);
         }
      }
   }

   real_t local_values[2] = {
      local_difference_squared, local_reference_squared
   };
   real_t global_values[2] = {0.0, 0.0};
   MPI_Allreduce(local_values, global_values, 2,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 fespace->GetComm());
   MFEM_VERIFY(std::isfinite(global_values[0]) &&
               std::isfinite(global_values[1]) &&
               global_values[0] >= 0.0 && global_values[1] > 0.0,
               "Boundary trace comparison produced a non-finite or zero "
               "reference norm.");

   BoundaryTraceComparisonMetrics metrics;
   metrics.difference_norm = std::sqrt(global_values[0]);
   metrics.reference_norm = std::sqrt(global_values[1]);
   metrics.relative_error =
      metrics.difference_norm / metrics.reference_norm;
   return metrics;
}

/**
 * Generate synthetic displacement observations with a higher-order state
 * discretization on the reconstruction ParMesh.
 *
 * The raw truth and its Helmholtz filtering are deliberately outside this
 * class: the caller needs those reconstruction-space fields first to compute
 * the exact active-control volume and to write them as named output fields.
 * This generator consumes the resulting fixed filtered truth, constructs the
 * corresponding SIMP material, and performs one high-order RK4 forward solve.
 *
 * The returned history retains no reference-space object.  At every requested
 * half step the high-order displacement is interpolated at the reconstruction
 * boundary nodes with ParGridFunction::ProjectBdrCoefficient, then packed by
 * BoundaryTraceHistory.  Consequently the returned object only depends on the
 * reconstruction state space, which must outlive it.
 */
class ReferenceBoundaryDataGenerator
{
private:
   const TransientTopOptProblem &problem_;
   ParFiniteElementSpace &reconstruction_state_fes_;
   ParGridFunction &truth_filtered_;
   Array<int> observation_marker_;
   int coarse_steps_;
   real_t coarse_dt_;
   int reference_order_;
   real_t requested_reference_dt_;
   bool damping_enabled_;
   MassSolverType mass_solver_type_;
   int source_index_;
   bool matrix_free_symplectic_euler_;
   bool generated_;
   ReferenceBoundaryDataMetadata metadata_;

   static Array<int> MakeBoundaryMarker(const ParMesh &mesh,
                                        const Array<int> &attributes,
                                        const char *description)
   {
      MFEM_VERIFY(mesh.bdr_attributes.Size() > 0,
                  "Reference boundary data mesh has no boundary attributes.");
      const int maximum_attribute = mesh.bdr_attributes.Max();
      Array<int> marker(maximum_attribute);
      marker = 0;
      for (int i = 0; i < attributes.Size(); i++)
      {
         const int attribute = attributes[i];
         MFEM_VERIFY(attribute >= 1 && attribute <= maximum_attribute,
                     "Reference boundary data " << description
                     << " attribute " << attribute << " is outside [1,"
                     << maximum_attribute << "].");
         marker[attribute - 1] = 1;
      }
      return marker;
   }

   static bool IsFinite(const Vector &value, MPI_Comm comm)
   {
      int local_nonfinite = 0;
      for (int i = 0; i < value.Size(); i++)
      {
         if (!std::isfinite(value[i]))
         {
            local_nonfinite = 1;
            break;
         }
      }
      int global_nonfinite = 0;
      MPI_Allreduce(&local_nonfinite, &global_nonfinite, 1, MPI_INT, MPI_MAX,
                    comm);
      return global_nonfinite == 0;
   }

   static bool NearlyEqual(real_t first, real_t second)
   {
      const real_t scale =
         std::max({real_t(1.0), std::abs(first), std::abs(second)});
      return std::abs(first - second) <=
             real_t(2048.0) * std::numeric_limits<real_t>::epsilon() * scale;
   }

public:
   ReferenceBoundaryDataGenerator(
      const TransientTopOptProblem &problem,
      ParFiniteElementSpace &reconstruction_state_fes,
      ParGridFunction &truth_filtered,
      const Array<int> &observation_marker,
      int coarse_steps, real_t coarse_dt,
      int reference_order, real_t reference_dt,
      bool damping_enabled,
      MassSolverType mass_solver_type,
      int source_index = 0,
      bool matrix_free_symplectic_euler = false)
      : problem_(problem),
        reconstruction_state_fes_(reconstruction_state_fes),
        truth_filtered_(truth_filtered),
        observation_marker_(observation_marker),
        coarse_steps_(coarse_steps),
        coarse_dt_(coarse_dt),
        reference_order_(reference_order),
        requested_reference_dt_(reference_dt),
        damping_enabled_(damping_enabled),
        mass_solver_type_(mass_solver_type),
        source_index_(source_index),
        matrix_free_symplectic_euler_(matrix_free_symplectic_euler),
        generated_(false)
   {
      MFEM_VERIFY(source_index_ >= 0 &&
                  source_index_ < problem_.GetNumberOfSources(),
                  "Reference generator received an invalid source index.");
      MFEM_VERIFY(!matrix_free_symplectic_euler_ ||
                  mass_solver_type_ == MassSolverType::LUMPED,
                  "A matrix-free symplectic-Euler reference needs lumped mass.");
   }

   const ReferenceBoundaryDataMetadata &Metadata() const
   {
      return metadata_;
   }

   std::shared_ptr<const BoundaryTraceHistory> Generate()
   {
      MFEM_VERIFY(!generated_,
                  "ReferenceBoundaryDataGenerator is a one-shot generator.");

      ParMesh *mesh = reconstruction_state_fes_.GetParMesh();
      MFEM_VERIFY(mesh,
                  "Reference boundary data requires a parallel mesh.");
      const MPI_Comm comm = reconstruction_state_fes_.GetComm();
      int rank = 0;
      MPI_Comm_rank(comm, &rank);

      const int dimension = mesh->Dimension();
      MFEM_VERIFY(dimension == 2 || dimension == 3,
                  "Elastic inclusion reference data requires a 2D or 3D mesh.");
      MFEM_VERIFY(reconstruction_state_fes_.GetVDim() == dimension,
                  "Reconstruction state space must be a displacement H1 space.");
      MFEM_VERIFY(
         reconstruction_state_fes_.FEColl()->GetContType() ==
         FiniteElementCollection::CONTINUOUS,
         "Reconstruction state space must be continuous.");
      MFEM_VERIFY(truth_filtered_.ParFESpace(),
                  "Reference boundary data requires a parallel filtered truth.");
      MFEM_VERIFY(truth_filtered_.ParFESpace()->GetParMesh() == mesh,
                  "The fixed filtered truth and reconstruction state must use "
                  "the exact same ParMesh object.");
      MFEM_VERIFY(truth_filtered_.VectorDim() == 1,
                  "The fixed filtered truth must be scalar.");

      MFEM_VERIFY(coarse_steps_ > 0 &&
                  coarse_steps_ <= (std::numeric_limits<int>::max() - 1) / 2,
                  "Reference boundary data has an invalid coarse step count.");
      MFEM_VERIFY(std::isfinite(coarse_dt_) && coarse_dt_ > 0.0,
                  "Reference boundary data requires a positive coarse timestep.");
      MFEM_VERIFY(
         NearlyEqual(coarse_steps_ * coarse_dt_, problem_.GetFinalTime()),
         "The coarse endpoint grid does not end at the problem final time.");

      const int reconstruction_order =
         reconstruction_state_fes_.GetMaxElementOrder();
      MFEM_VERIFY(reference_order_ >= reconstruction_order + 1,
                  "Reference state order must be at least reconstruction "
                  "order + 1 (received p=" << reconstruction_order
                  << ", p_dagger=" << reference_order_ << ").");
      MFEM_VERIFY(std::isfinite(requested_reference_dt_) &&
                  requested_reference_dt_ > 0.0,
                  "Reference timestep must be finite and positive.");

      const real_t steps_per_half_real =
         coarse_dt_ / (2.0 * requested_reference_dt_);
      MFEM_VERIFY(std::isfinite(steps_per_half_real) &&
                  steps_per_half_real >= 1.0 &&
                  steps_per_half_real <=
                  static_cast<real_t>(std::numeric_limits<int>::max()),
                  "The coarse/reference timestep ratio is unsupported.");
      const long long rounded_steps_per_half =
         std::llround(steps_per_half_real);
      const real_t nesting_tolerance =
         real_t(2048.0) * std::numeric_limits<real_t>::epsilon() *
         std::max(real_t(1.0), std::abs(steps_per_half_real));
      MFEM_VERIFY(
         std::abs(steps_per_half_real -
                  static_cast<real_t>(rounded_steps_per_half)) <=
         nesting_tolerance,
         "Reference timestep must divide one coarse half step exactly: "
         "coarse_dt/(2*reference_dt)=" << steps_per_half_real << ".");
      MFEM_VERIFY(rounded_steps_per_half >= 2,
                  "Reference timestep must be no larger than coarse_dt/4.");

      const long long reference_steps_long =
         2LL * coarse_steps_ * rounded_steps_per_half;
      MFEM_VERIFY(reference_steps_long > 0 &&
                  reference_steps_long <= std::numeric_limits<int>::max(),
                  "Reference time grid exceeds the supported step count.");
      const int reference_steps = static_cast<int>(reference_steps_long);
      const int steps_per_half =
         static_cast<int>(rounded_steps_per_half);
      // Snap to the exactly nested value after validating the user's value.
      const real_t reference_dt =
         coarse_dt_ / (2.0 * static_cast<real_t>(steps_per_half));

      metadata_ = ReferenceBoundaryDataMetadata{};
      metadata_.state_order = reference_order_;
      metadata_.reference_steps = reference_steps;
      metadata_.reference_steps_per_half_step = steps_per_half;
      metadata_.requested_time_step = requested_reference_dt_;
      metadata_.effective_time_step = reference_dt;
      metadata_.mass_solver_type = mass_solver_type_;
      metadata_.matrix_free_symplectic_euler =
         matrix_free_symplectic_euler_;
      metadata_.damping_enabled = damping_enabled_;

      auto history = std::make_shared<BoundaryTraceHistory>(
         &reconstruction_state_fes_, observation_marker_,
         coarse_dt_, coarse_steps_);
      metadata_.local_trace_memory_bytes =
         history->EstimatedLocalMemoryBytes();
      metadata_.global_trace_memory_bytes =
         history->EstimatedGlobalMemoryBytes();
      MPI_Allreduce(&metadata_.local_trace_memory_bytes,
                    &metadata_.maximum_trace_memory_bytes_per_rank,
                    1, MPI_DOUBLE, MPI_MAX, comm);

      if (rank == 0)
      {
         constexpr double bytes_per_megabyte = 1024.0 * 1024.0;
         mfem::out << "\n=== Reference Boundary Data ===\n"
                   << "Reference state order: " << reference_order_ << "\n"
                   << "Reference time grid: N=" << reference_steps
                   << ", requested dt=" << std::scientific
                   << std::setprecision(8) << requested_reference_dt_
                   << ", effective dt=" << reference_dt
                   << ", steps/coarse-half-step=" << steps_per_half << "\n"
                   << "Trace samples: " << history->SampleCount()
                   << ", observed local vector DOFs on rank 0="
                   << history->LocalTraceVDofCount() << "\n"
                   << "Estimated trace-history memory before solve: global="
                   << metadata_.global_trace_memory_bytes /
                      bytes_per_megabyte
                   << " MB, maximum/rank="
                   << metadata_.maximum_trace_memory_bytes_per_rank /
                      bytes_per_megabyte << " MB\n";
      }

      H1_FECollection reference_collection(reference_order_, dimension);
      ParFiniteElementSpace reference_state_fes(
         mesh, &reference_collection, dimension);
      metadata_.global_state_true_dofs =
         reference_state_fes.GlobalTrueVSize();

      const MaterialParams &material = problem_.GetMaterialParams();
      ConstantCoefficient rho0(material.rho0);
      ConstantCoefficient lambda0(material.lambda0);
      ConstantCoefficient mu0(material.mu0);
      SIMPCoefficient simp_scale(
         &truth_filtered_, material.r_min, material.r_max, material.simp_p);
      ProductCoefficient mass_coefficient(simp_scale, rho0);
      ProductCoefficient lambda_coefficient(simp_scale, lambda0);
      ProductCoefficient mu_coefficient(simp_scale, mu0);

      std::unique_ptr<DampingFieldBase> damping_field =
         problem_.CreateDampingField(damping_enabled_);
      MFEM_VERIFY(damping_field,
                  "Problem failed to create its reference damping field.");
      Coefficient &gamma = damping_field->GetCoefficient();

      Array<int> absorbing_attributes;
      problem_.GetAbsorbingBoundaryAttributes(absorbing_attributes);
      Array<int> absorbing_marker = MakeBoundaryMarker(
         *mesh, absorbing_attributes, "absorbing-boundary");
      Array<int> essential_attributes;
      problem_.GetEssentialBoundaryAttributes(essential_attributes);
      Array<int> essential_marker = MakeBoundaryMarker(
         *mesh, essential_attributes, "essential-boundary");

      std::unique_ptr<VectorCoefficient> load_coefficient =
         problem_.CreateBoundaryLoadCoefficient(source_index_);
      MFEM_VERIFY(load_coefficient,
                  "Problem failed to create its reference load coefficient.");
      const BoundaryLoadSpec &load = problem_.GetBoundaryLoad(source_index_);
      MFEM_VERIFY(load.direction.Size() == dimension &&
                  load_coefficient->GetVDim() == dimension,
                  "Reference load dimension does not match the mesh.");

      ElastodynamicsOperator reference_operator(
         reference_state_fes,
         mass_coefficient, lambda_coefficient, mu_coefficient,
         load.amplitude, load.duration, load.time_profile,
         load.phase, load.frequency, load.bdr_attributes,
         *load_coefficient, load.domain_load, &gamma,
         damping_field->GetImpedance(), absorbing_marker, essential_marker,
         mass_solver_type_, /*print_banner=*/true, load.frequencies,
         matrix_free_symplectic_euler_ ? SpatialOperatorMode::PARTIAL_ASSEMBLY :
                                         SpatialOperatorMode::FULL);
      if (matrix_free_symplectic_euler_)
      {
         ValidateKickDriftEulerTimeStep(reference_operator, reference_dt,
                                        /*print_report=*/true);
      }
      else
      {
         ValidateRK4TimeStep(reference_operator, reference_dt,
                             /*print_report=*/true);
      }

      Vector reference_state(reference_operator.Width());
      reference_state = 0.0;
      ParGridFunction reference_displacement(&reference_state_fes);
      ParGridFunction projected_trace(&reconstruction_state_fes_);
      VectorGridFunctionCoefficient reference_displacement_coefficient(
         &reference_displacement);
      const Array<int> &reference_offsets =
         reference_operator.GetBlockOffsets();

      const auto store_sample = [&](int half_step_index)
      {
         MFEM_VERIFY(IsFinite(reference_state, comm),
                     "Reference forward solve produced a non-finite state.");
         BlockVector state_blocks(reference_state, reference_offsets);
         reference_displacement.SetFromTrueDofs(state_blocks.GetBlock(0));
         projected_trace = 0.0;
         projected_trace.ProjectBdrCoefficient(
            reference_displacement_coefficient, observation_marker_);
         history->StoreSample(half_step_index, projected_trace);
      };

      RK4Solver solver;
      if (!matrix_free_symplectic_euler_)
      {
         solver.Init(reference_operator);
      }
      Vector kick_drift_rhs;
      if (matrix_free_symplectic_euler_)
      {
         kick_drift_rhs.SetSize(reference_state.Size());
      }
      real_t time = 0.0;
      store_sample(/*half_step_index=*/0);
      const double forward_start = MPI_Wtime();
      const int report_every = std::max(1, reference_steps / 10);
      for (int step = 1; step <= reference_steps; step++)
      {
         // Derive every accepted interval from its integer index so long runs
         // do not accumulate a floating-point time drift.
         time = (step - 1) * reference_dt;
         if (matrix_free_symplectic_euler_)
         {
            KickDriftEulerStep(reference_operator, reference_state, time,
                               reference_dt, kick_drift_rhs);
         }
         else
         {
            real_t step_size = reference_dt;
            solver.Step(reference_state, time, step_size);
         }
         time = step * reference_dt;

         if (step % steps_per_half == 0)
         {
            store_sample(step / steps_per_half);
         }
         if (rank == 0 &&
             (step % report_every == 0 || step == reference_steps))
         {
            mfem::out << "      reference " << std::setw(7) << step << '/'
                      << reference_steps << "  (" << std::setw(3)
                      << 100 * step / reference_steps << "%)\n";
         }
      }
      metadata_.forward_seconds = MPI_Wtime() - forward_start;

      MFEM_VERIFY(NearlyEqual(time, coarse_steps_ * coarse_dt_),
                  "Reference solve ended at the wrong physical time.");
      history->ValidateComplete();
      generated_ = true;

      if (rank == 0)
      {
         mfem::out << "Reference boundary data complete: "
                   << history->StoredSampleCount() << " samples, "
                   << metadata_.global_state_true_dofs
                   << " global state true DOFs, " << std::fixed
                   << std::setprecision(3) << metadata_.forward_seconds
                   << " s\n" << std::defaultfloat
                   << "===================================\n";
      }

      std::shared_ptr<const BoundaryTraceHistory> read_only_history = history;
      return read_only_history;
   }
};

} // namespace mfem

#endif // REFERENCE_BOUNDARY_DATA_HPP
