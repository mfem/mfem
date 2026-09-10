// Focused regression for the elastic-inclusion reference-data convergence
// audit.  The physical mesh, Q0 -> Helmholtz-Q1 truth, passive material
// prescription, Q2 reconstruction trace space, and high-order reference
// generator are the same code paths used by TopOptTransient.  The simulated
// interval is shortened so this remains a practical MPI regression.

#include "mfem.hpp"
#include "ProblemSpecification.hpp"
#include "ReferenceBoundaryData.hpp"
#include "../../pde_filter.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>

using namespace mfem;

namespace
{

Array<int> IdentifyPassiveTrueDofs(ParFiniteElementSpace &fespace,
                                   Coefficient &passive_region)
{
   Array<int> local_is_passive(fespace.GetVSize());
   local_is_passive = 0;
   ParMesh *mesh = fespace.GetParMesh();
   MFEM_VERIFY(mesh,
               "Reference convergence regression requires a parallel mesh.");

   for (int element_index = 0; element_index < mesh->GetNE(); element_index++)
   {
      const FiniteElement *element = fespace.GetFE(element_index);
      ElementTransformation *transformation =
         fespace.GetElementTransformation(element_index);
      const IntegrationRule &rule = IntRules.Get(
         element->GetGeomType(), 2 * element->GetOrder());
      bool is_passive = false;
      for (int q = 0; q < rule.GetNPoints(); q++)
      {
         const IntegrationPoint &point = rule.IntPoint(q);
         transformation->SetIntPoint(&point);
         if (passive_region.Eval(*transformation, point) > 0.5)
         {
            is_passive = true;
            break;
         }
      }
      if (!is_passive) { continue; }

      Array<int> dofs;
      fespace.GetElementVDofs(element_index, dofs);
      for (int i = 0; i < dofs.Size(); i++)
      {
         const int dof = dofs[i] >= 0 ? dofs[i] : -1 - dofs[i];
         local_is_passive[dof] = 1;
      }
   }

   // Include passive elements seen only through a shared interface copy.
   fespace.Synchronize(local_is_passive);
   Array<int> true_is_passive(fespace.GetTrueVSize());
   true_is_passive = 0;
   const SparseMatrix *restriction = fespace.GetRestrictionMatrix();
   if (restriction)
   {
      restriction->BooleanMult(local_is_passive, true_is_passive);
   }
   else
   {
      for (int i = 0; i < true_is_passive.Size(); i++)
      {
         true_is_passive[i] = local_is_passive[i];
      }
   }

   Array<int> passive_true_dofs;
   for (int i = 0; i < true_is_passive.Size(); i++)
   {
      if (true_is_passive[i]) { passive_true_dofs.Append(i); }
   }
   return passive_true_dofs;
}

Array<int> MakeObservationMarker(
   const ElasticInclusionIdentificationProblem &problem,
   const ParMesh &mesh)
{
   Array<int> attributes;
   problem.GetObservationBoundaryAttributes(attributes);
   Array<int> marker(mesh.bdr_attributes.Max());
   marker = 0;
   for (int i = 0; i < attributes.Size(); i++)
   {
      MFEM_VERIFY(attributes[i] >= 1 && attributes[i] <= marker.Size(),
                  "Observation boundary attribute is out of range.");
      marker[attributes[i] - 1] = 1;
   }
   return marker;
}

class AnalyticTraceCoefficient final : public VectorCoefficient
{
private:
   bool candidate_;
   real_t time_ = 0.0;

public:
   explicit AnalyticTraceCoefficient(bool candidate)
      : VectorCoefficient(2), candidate_(candidate) { }

   void SetTime(real_t time) { time_ = time; }

   void Eval(Vector &value, ElementTransformation &,
             const IntegrationPoint &) override
   {
      value.SetSize(2);
      value[0] = 2.0 + (candidate_ ? time_ : 0.0);
      value[1] = -1.0;
   }
};

void VerifyAnalyticTraceComparison(ParFiniteElementSpace &state_fes,
                                   const Array<int> &observation_marker)
{
   constexpr real_t coarse_dt = 0.1;
   constexpr int coarse_steps = 2;
   auto candidate = std::make_shared<BoundaryTraceHistory>(
      &state_fes, observation_marker, coarse_dt, coarse_steps);
   auto reference = std::make_shared<BoundaryTraceHistory>(
      &state_fes, observation_marker, coarse_dt, coarse_steps);
   AnalyticTraceCoefficient candidate_coefficient(/*candidate=*/true);
   AnalyticTraceCoefficient reference_coefficient(/*candidate=*/false);
   ParGridFunction candidate_sample(&state_fes);
   ParGridFunction reference_sample(&state_fes);
   for (int sample = 0; sample < candidate->SampleCount(); sample++)
   {
      const real_t time = 0.5 * coarse_dt * sample;
      candidate_coefficient.SetTime(time);
      reference_coefficient.SetTime(time);
      candidate_sample = 0.0;
      reference_sample = 0.0;
      candidate_sample.ProjectBdrCoefficient(
         candidate_coefficient, observation_marker);
      reference_sample.ProjectBdrCoefficient(
         reference_coefficient, observation_marker);
      candidate->StoreSample(sample, candidate_sample);
      reference->StoreSample(sample, reference_sample);
   }

   const BoundaryTraceComparisonMetrics metrics =
      CompareBoundaryTraceHistories(*candidate, *reference);
   // Gamma_obs has measure 0.8.  The reference is (2,-1), while the
   // difference is (t,0), on 0 <= t <= 0.2.  Composite Simpson is exact for
   // both squared norms.
   constexpr real_t expected_reference_squared = 0.8;
   constexpr real_t expected_difference_squared = 0.8 * 0.008 / 3.0;
   const real_t tolerance =
      2.0e-12 * std::sqrt(expected_reference_squared);
   MFEM_VERIFY(std::abs(metrics.reference_norm -
                        std::sqrt(expected_reference_squared)) < tolerance &&
               std::abs(metrics.difference_norm -
                        std::sqrt(expected_difference_squared)) < tolerance,
               "Analytic space-time receiver norm regression failed.");
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   const MPI_Comm comm = MPI_COMM_WORLD;
   Device device("cpu");

   // Retain the production mesh and source/receiver geometry.  A 0.12-time
   // pulse on a 0.12 interval gives a nonzero receiver signal while keeping
   // this three-forward-solve regression inexpensive.
   TransientTopOptConfig base;
   base.order = 2;
   base.order_is_user = true;
   base.t_final = 0.12;
   base.t_final_is_user = true;
   base.dt = 0.002;
   base.time_step_is_user = true;
   base.boundary_load.duration = 0.12;
   base.load_duration_is_user = true;
   ElasticInclusionIdentificationProblem problem(
      base, ElasticInclusionTruthPreset::SINGLE_DISK);
   std::ostringstream validation_error;
   MFEM_VERIFY(problem.Validate(validation_error),
               "Reference convergence regression configuration is invalid: "
               << validation_error.str());

   Mesh serial_mesh = problem.CreateMesh();
   ParMesh mesh(comm, serial_mesh);
   const int dimension = mesh.Dimension();
   H1_FECollection state_collection(/*order=*/2, dimension);
   H1_FECollection filter_collection(/*order=*/1, dimension);
   L2_FECollection control_collection(
      /*order=*/0, dimension, BasisType::GaussLobatto);
   ParFiniteElementSpace state_fes(
      &mesh, &state_collection, /*vector_dimension=*/dimension);
   ParFiniteElementSpace filter_fes(&mesh, &filter_collection);
   ParFiniteElementSpace control_fes(&mesh, &control_collection);

   std::unique_ptr<Coefficient> passive_region =
      problem.CreatePassiveRegionCoefficient();
   MFEM_VERIFY(passive_region,
               "Elastic inclusion regression lost its passive region.");
   const Array<int> passive_filter_true_dofs =
      IdentifyPassiveTrueDofs(filter_fes, *passive_region);
   HYPRE_BigInt local_passive = passive_filter_true_dofs.Size();
   HYPRE_BigInt global_passive = 0;
   MPI_Allreduce(&local_passive, &global_passive, 1,
                 HYPRE_MPI_BIG_INT, MPI_SUM, comm);
   MFEM_VERIFY(global_passive > 0,
               "Elastic inclusion filtered passive mask is empty.");

   toopt::PDEFilterOptions filter_options;
   filter_options.filter_radius = problem.GetFilterRadius();
   toopt::PDEFilter filter(filter_fes, control_fes, filter_options);
   filter.SetPrescribedOutputDofs(
      passive_filter_true_dofs, problem.GetPassiveDensity());
   filter.Assemble();

   std::unique_ptr<Coefficient> truth_coefficient =
      problem.CreateTruthDensityCoefficient();
   MFEM_VERIFY(truth_coefficient,
               "Elastic inclusion regression failed to create its truth.");
   ParGridFunction truth_raw(&control_fes);
   ParGridFunction truth_filtered(&filter_fes);
   truth_raw.ProjectCoefficient(*truth_coefficient);
   filter.Mult(truth_raw, truth_filtered);

   const Array<int> observation_marker =
      MakeObservationMarker(problem, mesh);
   VerifyAnalyticTraceComparison(state_fes, observation_marker);
   constexpr int coarse_steps = 60;
   constexpr real_t coarse_dt = 0.002;
   constexpr real_t reference_dt_over_4 = 0.0005;
   constexpr real_t reference_dt_over_8 = 0.00025;

   const auto generate = [&](int order, real_t time_step,
                             ReferenceBoundaryDataMetadata &metadata)
   {
      ReferenceBoundaryDataGenerator generator(
         problem, state_fes, truth_filtered, observation_marker,
         coarse_steps, coarse_dt, order, time_step,
         /*damping_enabled=*/true, MassSolverType::ITERATIVE);
      std::shared_ptr<const BoundaryTraceHistory> history =
         generator.Generate();
      metadata = generator.Metadata();
      return history;
   };

   ReferenceBoundaryDataMetadata q3_dt4_metadata;
   ReferenceBoundaryDataMetadata q3_dt8_metadata;
   ReferenceBoundaryDataMetadata q4_dt8_metadata;
   const std::shared_ptr<const BoundaryTraceHistory> q3_dt4 =
      generate(/*order=*/3, reference_dt_over_4, q3_dt4_metadata);
   const std::shared_ptr<const BoundaryTraceHistory> q3_dt8 =
      generate(/*order=*/3, reference_dt_over_8, q3_dt8_metadata);
   const std::shared_ptr<const BoundaryTraceHistory> q4_dt8 =
      generate(/*order=*/4, reference_dt_over_8, q4_dt8_metadata);

   MFEM_VERIFY(q3_dt4_metadata.reference_steps == 240 &&
               q3_dt8_metadata.reference_steps == 480 &&
               q4_dt8_metadata.reference_steps == 480,
               "Reference convergence regression used an unexpected nested "
               "time grid.");

   const BoundaryTraceComparisonMetrics temporal =
      CompareBoundaryTraceHistories(*q3_dt4, *q3_dt8);
   const BoundaryTraceComparisonMetrics spatial =
      CompareBoundaryTraceHistories(*q3_dt8, *q4_dt8);
   const BoundaryTraceComparisonMetrics total =
      CompareBoundaryTraceHistories(*q3_dt4, *q4_dt8);

   // Normalize every audit component by the common finest Q4(dt/8) signal,
   // so pairwise errors are directly comparable and sum to a useful error
   // budget.  The helper's pairwise relative error remains available above.
   const real_t temporal_finest_relative =
      temporal.difference_norm / total.reference_norm;
   const real_t spatial_finest_relative =
      spatial.difference_norm / total.reference_norm;

   constexpr real_t maximum_pairwise_relative_error = 5.0e-2;
   MFEM_VERIFY(std::isfinite(temporal.relative_error) &&
               std::isfinite(spatial.relative_error) &&
               std::isfinite(total.relative_error),
               "Reference convergence regression produced a non-finite "
               "relative error.");
   MFEM_VERIFY(temporal_finest_relative < maximum_pairwise_relative_error &&
               spatial_finest_relative < maximum_pairwise_relative_error &&
               total.relative_error < maximum_pairwise_relative_error,
               "Required Q3(dt/4) boundary data are not within 5% of the "
               "Q3(dt/8) and Q4(dt/8) references.");
   const real_t triangle_tolerance =
      256.0 * std::numeric_limits<real_t>::epsilon();
   MFEM_VERIFY(total.difference_norm <=
               temporal.difference_norm + spatial.difference_norm +
               triangle_tolerance * q4_dt8_metadata.reference_steps *
               total.reference_norm,
               "Boundary trace comparison violates the triangle inequality.");

   if (Mpi::Root())
   {
      mfem::out << std::scientific << std::setprecision(8)
                << "Elastic-inclusion reference convergence regression passed\n"
                << "  temporal Q3(dt/4) vs Q3(dt/8): finest-relative="
                << temporal_finest_relative << ", pairwise-relative="
                << temporal.relative_error << ", absolute="
                << temporal.difference_norm << '\n'
                << "  spatial  Q3(dt/8) vs Q4(dt/8): finest-relative="
                << spatial_finest_relative << ", absolute="
                << spatial.difference_norm << '\n'
                << "  total    Q3(dt/4) vs Q4(dt/8): relative="
                << total.relative_error << ", absolute="
                << total.difference_norm << '\n'
                << "  Q4(dt/8) receiver signal norm: "
                << total.reference_norm << '\n';
   }
   return 0;
}
