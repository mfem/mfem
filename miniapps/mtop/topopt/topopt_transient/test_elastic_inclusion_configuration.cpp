// Regression for the single-disk elastic-inclusion experiment configuration.

#include "mfem.hpp"
#include "ProblemSpecification.hpp"

#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>

using namespace mfem;

namespace
{

real_t BoundaryMeasure(const Mesh &mesh, int attribute)
{
   real_t measure = 0.0;
   for (int be = 0; be < mesh.GetNBE(); be++)
   {
      const Element *element = mesh.GetBdrElement(be);
      if (element->GetAttribute() != attribute) { continue; }

      Array<int> vertices;
      element->GetVertices(vertices);
      MFEM_VERIFY(vertices.Size() == 2,
                  "Elastic-inclusion boundary elements must be segments.");
      const real_t *x_0 = mesh.GetVertex(vertices[0]);
      const real_t *x_1 = mesh.GetVertex(vertices[1]);
      real_t length_squared = 0.0;
      for (int d = 0; d < mesh.SpaceDimension(); d++)
      {
         const real_t difference = x_1[d] - x_0[d];
         length_squared += difference * difference;
      }
      measure += std::sqrt(length_squared);
   }
   return measure;
}

void VerifyTopWindow(const Mesh &mesh, int attribute,
                     real_t left_min, real_t left_max,
                     real_t right_min, real_t right_max)
{
   constexpr real_t top = 0.75;
   constexpr real_t tolerance = 2.0e-14;
   for (int be = 0; be < mesh.GetNBE(); be++)
   {
      const Element *element = mesh.GetBdrElement(be);
      if (element->GetAttribute() != attribute) { continue; }

      Array<int> vertices;
      element->GetVertices(vertices);
      const real_t *x_0 = mesh.GetVertex(vertices[0]);
      const real_t *x_1 = mesh.GetVertex(vertices[1]);
      const real_t midpoint = 0.5 * (x_0[0] + x_1[0]);
      const bool in_left = midpoint >= left_min - tolerance &&
                           midpoint <= left_max + tolerance;
      const bool in_right = midpoint >= right_min - tolerance &&
                            midpoint <= right_max + tolerance;
      MFEM_VERIFY(std::abs(x_0[1] - top) <= tolerance &&
                  std::abs(x_1[1] - top) <= tolerance &&
                  (in_left || in_right),
                  "An observation element lies outside the accessible top "
                  "receiver windows.");
   }
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   const MPI_Comm comm = MPI_COMM_WORLD;
   Device device("cpu");

   TransientTopOptConfig base;
   ElasticInclusionIdentificationProblem problem(
      base, ElasticInclusionTruthPreset::SINGLE_DISK);
   const TransientTopOptConfig &cfg = problem.GetConfig();

   std::ostringstream validation_error;
   MFEM_VERIFY(problem.Validate(validation_error),
               "Default single-disk configuration is invalid: "
               << validation_error.str());

   MFEM_VERIFY(problem.GetTruthPreset() ==
               ElasticInclusionTruthPreset::SINGLE_DISK,
               "The elastic-inclusion default is not the single disk.");
   MFEM_VERIFY(std::abs(cfg.material.r_min - 0.1) < 1e-15 &&
               std::abs(cfg.material.r_max - 1.0) < 1e-15 &&
               std::abs(cfg.material.simp_p - 3.0) < 1e-15,
               "The single-disk SIMP defaults changed unexpectedly.");
   MFEM_VERIFY(std::abs(cfg.boundary_load.frequency - 4.0) < 1e-15 &&
               std::abs(cfg.boundary_load.duration - 0.75) < 1e-15,
               "The single-disk source defaults changed unexpectedly.");

   // Validate the final problem-owned material parameters, after its defaults
   // have been resolved.  This catches, for example, an explicit r_max below
   // the inclusion problem's default r_min=0.1.
   TransientTopOptConfig invalid_material = base;
   invalid_material.material.r_max = 0.05;
   invalid_material.simp_r_max_is_user = true;
   ElasticInclusionIdentificationProblem invalid_problem(
      invalid_material, ElasticInclusionTruthPreset::THREE_SHAPE);
   std::ostringstream invalid_error;
   MFEM_VERIFY(!invalid_problem.Validate(invalid_error) &&
               invalid_error.str().find("0 < r_min < r_max") !=
                  std::string::npos,
               "Post-default elastic-inclusion SIMP validation regressed.");

   using Disk = ElasticInclusionSingleDiskTruthCoefficient;
   const real_t disk_density = Disk::DiskDensity(cfg.material);
   const real_t disk_scale =
      cfg.material.r_min +
      (cfg.material.r_max - cfg.material.r_min) *
      std::pow(disk_density, cfg.material.simp_p);
   const real_t background_scale =
      cfg.material.r_min +
      (cfg.material.r_max - cfg.material.r_min) *
      std::pow(Disk::BackgroundDensity(), cfg.material.simp_p);
   MFEM_VERIFY(std::abs(disk_scale - Disk::DiskTargetMaterialScale()) < 1e-14,
               "The raw disk density no longer maps to SIMP scale 0.125.");
   MFEM_VERIFY(std::abs(background_scale - 1.0) < 1e-15,
               "The raw background density no longer maps to solid material.");

   std::unique_ptr<Coefficient> truth =
      problem.CreateTruthDensityCoefficient();
   const auto *disk_truth =
      dynamic_cast<const ElasticInclusionSingleDiskTruthCoefficient *>(
         truth.get());
   MFEM_VERIFY(disk_truth != nullptr &&
               std::abs(disk_truth->GetDiskDensity() - disk_density) < 1e-15,
               "The single-disk preset created the wrong truth coefficient.");

   constexpr real_t pi = 3.1415926535897932384626433832795;
   constexpr real_t active_area = (1.25 - 0.25) * (0.65 - 0.25);
   const real_t expected_volume_fraction =
      1.0 - (1.0 - disk_density) * pi * 0.1 * 0.1 / active_area;
   MFEM_VERIFY(std::abs(problem.GetAnalyticTruthVolumeFraction() -
                        expected_volume_fraction) < 1e-14,
               "The analytic single-disk raw volume fraction is incorrect.");

   Mesh serial_mesh = problem.CreateMesh();
   const int source_attribute = problem.GetSourceBoundaryAttribute();
   const int receiver_attribute = problem.GetReceiverBoundaryAttribute();
   MFEM_VERIFY(source_attribute == 5 && receiver_attribute == 6,
               "Elastic-inclusion source/receiver attributes changed.");
   MFEM_VERIFY(std::abs(BoundaryMeasure(serial_mesh, source_attribute) - 0.2)
               < 1e-13,
               "The top source window does not have measure 0.2.");
   MFEM_VERIFY(std::abs(BoundaryMeasure(serial_mesh, receiver_attribute) - 0.8)
               < 1e-13,
               "The accessible top receiver windows do not have measure 0.8.");
   VerifyTopWindow(serial_mesh, receiver_attribute,
                   0.25, 0.65, 0.85, 1.25);

   Array<int> observation_attributes;
   problem.GetObservationBoundaryAttributes(observation_attributes);
   MFEM_VERIFY(observation_attributes.Size() == 1 &&
               observation_attributes[0] == receiver_attribute,
               "Observations must use only the top receiver attribute.");

   // Exercise the actual boundary objective's parallel measure calculation,
   // not only the serial mesh geometry above.
   ParMesh mesh(comm, serial_mesh);
   H1_FECollection state_collection(cfg.order, mesh.Dimension());
   ParFiniteElementSpace state_fes(
      &mesh, &state_collection, /*vector_dimension=*/mesh.Dimension());
   Array<int> observation_marker(mesh.bdr_attributes.Max());
   observation_marker = 0;
   observation_marker[receiver_attribute - 1] = 1;
   constexpr int coarse_steps = 1;
   auto history = std::make_shared<BoundaryTraceHistory>(
      &state_fes, observation_marker, cfg.dt, coarse_steps);
   ParGridFunction zero_state(&state_fes);
   zero_state = 0.0;
   for (int sample = 0; sample < history->SampleCount(); sample++)
   {
      history->StoreSampleAtTime(0.5 * cfg.dt * sample, zero_state);
   }
   history->ValidateComplete();
   std::shared_ptr<const BoundaryTraceHistory> read_only_history = history;
   BoundaryDisplacementTrackingObjective objective(
      &state_fes, read_only_history, comm);
   MFEM_VERIFY(std::abs(objective.ObservedBoundaryMeasure() - 0.8) < 1e-13,
               "The inverse objective observes more than the accessible "
               "top boundary.");

   if (Mpi::Root())
   {
      mfem::out << "Elastic single-disk configuration regression passed\n"
                << "  receiver attribute/measure: " << receiver_attribute
                << " / " << objective.ObservedBoundaryMeasure() << '\n'
                << "  source attribute/measure: " << source_attribute
                << " / " << BoundaryMeasure(serial_mesh, source_attribute)
                << '\n'
                << "  SIMP law: s(rho)=0.1+0.9*rho^3\n"
                << "  disk raw density: " << std::setprecision(15)
                << disk_density << '\n'
                << "  disk material scale: " << disk_scale << '\n'
                << "  analytic raw volume fraction: "
                << expected_volume_fraction << '\n'
                << "  source frequency/duration: "
                << cfg.boundary_load.frequency << " / "
                << cfg.boundary_load.duration << '\n';
   }

   return 0;
}
