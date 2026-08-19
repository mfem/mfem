// =============================================================================
// High-order synthetic boundary data for elastic inclusion identification
// =============================================================================

#ifndef REFERENCE_BOUNDARY_DATA_HPP
#define REFERENCE_BOUNDARY_DATA_HPP

#include "ElastodynamicsSolver.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>

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
   bool damping_enabled = true;
   HYPRE_BigInt global_state_true_dofs = 0;
   double local_trace_memory_bytes = 0.0;
   double maximum_trace_memory_bytes_per_rank = 0.0;
   double global_trace_memory_bytes = 0.0;
   double forward_seconds = 0.0;
};

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
      MassSolverType mass_solver_type)
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
        generated_(false)
   {
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
      MFEM_VERIFY(dimension == 2,
                  "Elastic inclusion reference data is currently two-dimensional.");
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
         problem_.CreateBoundaryLoadCoefficient();
      MFEM_VERIFY(load_coefficient,
                  "Problem failed to create its reference load coefficient.");
      const BoundaryLoadSpec &load = problem_.GetBoundaryLoad();
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
         mass_solver_type_, /*print_banner=*/true);
      ValidateRK4TimeStep(reference_operator, reference_dt,
                          /*print_report=*/true);

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
      solver.Init(reference_operator);
      real_t time = 0.0;
      store_sample(/*half_step_index=*/0);
      const double forward_start = MPI_Wtime();
      const int report_every = std::max(1, reference_steps / 10);
      for (int step = 1; step <= reference_steps; step++)
      {
         // Derive every accepted interval from its integer index so long runs
         // do not accumulate a floating-point time drift.
         time = (step - 1) * reference_dt;
         real_t step_size = reference_dt;
         solver.Step(reference_state, time, step_size);
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
