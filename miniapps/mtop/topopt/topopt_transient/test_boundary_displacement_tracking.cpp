// Unit regression for half-step boundary traces and their tracking objective.

#include "mfem.hpp"
#include "ObjectiveFunctional.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <memory>

using namespace mfem;

namespace
{

enum class ManufacturedField
{
   TARGET,
   STATE,
   DIRECTION
};

class ManufacturedVectorCoefficient : public VectorCoefficient
{
private:
   ManufacturedField field_;
   real_t time_;

public:
   explicit ManufacturedVectorCoefficient(ManufacturedField field)
      : VectorCoefficient(2), field_(field), time_(0.0) {}

   void SetTime(real_t time) { time_ = time; }

   void Eval(Vector &value, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      Vector x(2);
      T.Transform(ip, x);
      value.SetSize(2);

      const real_t target_0 =
         1.0 + 0.4 * time_ + x[0] + 0.2 * x[1];
      const real_t target_1 =
         -0.3 + 0.1 * time_ - 0.25 * x[0] + 0.5 * x[1];
      const real_t perturbation_0 = 0.3 + 0.2 * x[0] - 0.1 * x[1];
      const real_t perturbation_1 = -0.2 + 0.1 * x[0] + 0.25 * x[1];

      if (field_ == ManufacturedField::TARGET)
      {
         value[0] = target_0;
         value[1] = target_1;
      }
      else if (field_ == ManufacturedField::STATE)
      {
         value[0] = target_0 + perturbation_0;
         value[1] = target_1 + perturbation_1;
      }
      else
      {
         value[0] = -0.4 + 0.3 * x[0] + 0.2 * x[1];
         value[1] = 0.15 - 0.05 * x[0] + 0.1 * x[1];
      }
   }
};

real_t GlobalAction(MPI_Comm comm, const Vector &linear_form,
                    const Vector &direction)
{
   const real_t local_action = linear_form * direction;
   real_t global_action = 0.0;
   MPI_Allreduce(&local_action, &global_action, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
   return global_action;
}

real_t GlobalNorm(MPI_Comm comm, const Vector &value)
{
   const real_t local_norm_squared = value * value;
   real_t global_norm_squared = 0.0;
   MPI_Allreduce(&local_norm_squared, &global_norm_squared, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
   return std::sqrt(std::max(global_norm_squared, real_t(0.0)));
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   const MPI_Comm comm = MPI_COMM_WORLD;
   Device device("cpu");

   {
      int ranks = 1;
      MPI_Comm_size(comm, &ranks);
      Mesh serial_mesh = Mesh::MakeCartesian2D(
         2 * ranks, 2, Element::QUADRILATERAL,
         /*generate_edges=*/true, 1.0, 1.0);
      ParMesh mesh(comm, serial_mesh);
      H1_FECollection collection(/*order=*/2, /*dimension=*/2);
      ParFiniteElementSpace state_fes(
         &mesh, &collection, /*vector_dimension=*/2);
      H1_FECollection reference_collection(/*order=*/3, /*dimension=*/2);
      ParFiniteElementSpace reference_state_fes(
         &mesh, &reference_collection, /*vector_dimension=*/2);

      // Cartesian boundary attributes: bottom=1, right=2, top=3, left=4.
      // Observe top and right while leaving the other pieces unobserved.
      Array<int> observation_marker(mesh.bdr_attributes.Max());
      observation_marker = 0;
      observation_marker[1] = 1;
      observation_marker[2] = 1;

      constexpr real_t coarse_dt = 0.2;
      constexpr int coarse_steps = 2;
      auto history = std::make_shared<BoundaryTraceHistory>(
         &state_fes, observation_marker, coarse_dt, coarse_steps);

      MFEM_VERIFY(history->SampleCount() == 2 * coarse_steps + 1,
                  "Boundary trace history has the wrong sample count.");
      MFEM_VERIFY(history->LocalTraceVDofCount() < state_fes.GetVSize(),
                  "Boundary trace history retained interior state DOFs.");
      MFEM_VERIFY(history->EstimatedLocalMemoryBytes() >= 0.0 &&
                  history->EstimatedGlobalMemoryBytes() > 0.0,
                  "Boundary trace memory estimate is invalid.");

      ManufacturedVectorCoefficient target_coefficient(
         ManufacturedField::TARGET);
      ParGridFunction reference_target(&reference_state_fes);
      VectorGridFunctionCoefficient reference_target_coefficient(
         &reference_target);
      ParGridFunction target_sample(&state_fes);
      ParGridFunction direct_target_sample(&state_fes);
      for (int sample = 0; sample < history->SampleCount(); sample++)
      {
         const real_t time = 0.5 * coarse_dt * sample;
         target_coefficient.SetTime(time);
         reference_target.ProjectCoefficient(target_coefficient);
         target_sample = 0.0;
         target_sample.ProjectBdrCoefficient(
            reference_target_coefficient, observation_marker);

         // The manufactured field is linear, so nodal Q3 -> Q2 trace
         // interpolation on the common mesh must match direct Q2 boundary
         // projection to roundoff.  This guards the order-mismatched transfer
         // used by ReferenceBoundaryDataGenerator without comparing raw
         // coefficient vectors from incompatible spaces.
         direct_target_sample = 0.0;
         direct_target_sample.ProjectBdrCoefficient(
            target_coefficient, observation_marker);
         direct_target_sample -= target_sample;
         MFEM_VERIFY(GlobalNorm(comm, direct_target_sample) < 2e-13,
                     "Manufactured Q3-to-Q2 boundary trace transfer failed.");
         history->StoreSampleAtTime(time, target_sample);
      }
      history->ValidateComplete();

      // Exercise reverse-time lookup before constructing the objective.  The
      // returned object is scratch storage, so copy each value before the next
      // access.
      Vector late_sample(history->GetSampleAtTime(2.0 * coarse_dt));
      Vector early_sample(history->GetSampleAtTime(0.5 * coarse_dt));
      late_sample -= early_sample;
      MFEM_VERIFY(GlobalNorm(comm, late_sample) > 0.0,
                  "Boundary trace lookup ignored physical time.");
      MFEM_VERIFY(history->HalfStepIndex(0.5 * coarse_dt) == 1,
                  "Boundary trace half-step lookup returned the wrong index.");

      std::shared_ptr<const BoundaryTraceHistory> read_only_history = history;
      BoundaryDisplacementTrackingObjective objective(
         &state_fes, read_only_history, comm);
      MFEM_VERIFY(std::abs(objective.ObservedBoundaryMeasure() - 2.0) < 1e-13,
                  "Observed boundary measure is incorrect.");

      constexpr real_t evaluation_time = 0.5 * coarse_dt;
      ManufacturedVectorCoefficient state_coefficient(
         ManufacturedField::STATE);
      state_coefficient.SetTime(evaluation_time);
      ParGridFunction state(&state_fes);
      state.ProjectCoefficient(state_coefficient);

      ManufacturedVectorCoefficient direction_coefficient(
         ManufacturedField::DIRECTION);
      ParGridFunction direction(&state_fes);
      direction.ProjectCoefficient(direction_coefficient);

      const real_t objective_value =
         objective.EvaluateInstantaneous(state, evaluation_time);
      // The manufactured state-target difference integrates analytically to
      // 47/300 over the selected unit-length top and right boundaries.
      constexpr real_t exact_objective = 47.0 / 300.0;
      MFEM_VERIFY(std::abs(objective_value - exact_objective) < 2e-13,
                  "Boundary tracking objective value is incorrect.");

      ParLinearForm gradient(&state_fes);
      objective.AssembleInstantaneousStateGradient(
         state, evaluation_time, gradient);
      const real_t adjoint_directional_derivative =
         GlobalAction(comm, gradient, direction);

      constexpr real_t epsilon = 1e-4;
      ParGridFunction state_plus(state);
      ParGridFunction state_minus(state);
      state_plus.Add(epsilon, direction);
      state_minus.Add(-epsilon, direction);
      const real_t finite_difference =
         (objective.EvaluateInstantaneous(state_plus, evaluation_time) -
          objective.EvaluateInstantaneous(state_minus, evaluation_time)) /
         (2.0 * epsilon);
      const real_t derivative_error =
         std::abs(finite_difference - adjoint_directional_derivative) /
         std::max({std::abs(finite_difference),
                   std::abs(adjoint_directional_derivative), real_t(1e-30)});
      if (Mpi::Root() && derivative_error >= 2e-10)
      {
         mfem::err << "Boundary derivative mismatch: assembled="
                   << std::setprecision(16)
                   << adjoint_directional_derivative
                   << ", finite_difference=" << finite_difference
                   << ", relative_error=" << derivative_error << '\n';
      }
      MFEM_VERIFY(derivative_error < 2e-10,
                  "Boundary tracking state derivative failed its centered "
                  "finite-difference check.");

      if (Mpi::Root())
      {
         mfem::out << "Boundary displacement tracking regression passed\n"
                   << "  observed measure: "
                   << objective.ObservedBoundaryMeasure() << '\n'
                   << "  objective: " << std::scientific
                   << std::setprecision(12) << objective_value << '\n'
                   << "  adjoint directional derivative: "
                   << adjoint_directional_derivative << '\n'
                   << "  centered FD derivative: " << finite_difference << '\n'
                   << "  relative derivative error: " << derivative_error
                   << '\n'
                   << "  trace memory (global bytes): "
                   << history->EstimatedGlobalMemoryBytes() << '\n';
      }
   }

   return 0;
}
