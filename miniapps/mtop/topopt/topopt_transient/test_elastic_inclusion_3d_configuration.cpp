// Regression for the single-ball and three-shot 3D elastic-inclusion
// configurations. It verifies the top-only geometry, the ball's SIMP
// contrast/volume target, and the source-specific receiver areas.

#include "mfem.hpp"
#include "ProblemSpecification.hpp"

#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>

using namespace mfem;

namespace
{

real_t TriangleArea(const real_t *a, const real_t *b, const real_t *c)
{
   const real_t ux = b[0] - a[0];
   const real_t uy = b[1] - a[1];
   const real_t uz = b[2] - a[2];
   const real_t vx = c[0] - a[0];
   const real_t vy = c[1] - a[1];
   const real_t vz = c[2] - a[2];
   const real_t cx = uy * vz - uz * vy;
   const real_t cy = uz * vx - ux * vz;
   const real_t cz = ux * vy - uy * vx;
   return 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
}

real_t BoundaryArea(const Mesh &mesh, int attribute)
{
   real_t area = 0.0;
   for (int be = 0; be < mesh.GetNBE(); be++)
   {
      const Element *element = mesh.GetBdrElement(be);
      if (element->GetAttribute() != attribute) { continue; }
      Array<int> vertices;
      element->GetVertices(vertices);
      MFEM_VERIFY(vertices.Size() == 4,
                  "Elastic-inclusion 3D boundary elements must be quads.");
      const real_t *x0 = mesh.GetVertex(vertices[0]);
      const real_t *x1 = mesh.GetVertex(vertices[1]);
      const real_t *x2 = mesh.GetVertex(vertices[2]);
      const real_t *x3 = mesh.GetVertex(vertices[3]);
      area += TriangleArea(x0, x1, x2) + TriangleArea(x0, x2, x3);
   }
   return area;
}

bool ContainsAttribute(const Array<int> &attributes, int attribute)
{
   for (int i = 0; i < attributes.Size(); i++)
   {
      if (attributes[i] == attribute) { return true; }
   }
   return false;
}

void VerifyTopReceiverSurface(const Mesh &mesh, int attribute,
                              real_t active_x_min, real_t active_x_max,
                              real_t active_z_min, real_t active_z_max,
                              real_t source_center_x,
                              real_t source_center_z, real_t source_radius)
{
   constexpr real_t top = 0.75;
   constexpr real_t tolerance = 2.0e-14;
   for (int be = 0; be < mesh.GetNBE(); be++)
   {
      const Element *element = mesh.GetBdrElement(be);
      if (element->GetAttribute() != attribute) { continue; }
      Array<int> vertices;
      element->GetVertices(vertices);
      MFEM_VERIFY(vertices.Size() == 4,
                  "Elastic-inclusion 3D receiver element must be a quad.");
      real_t x_mid = 0.0;
      for (int i = 0; i < vertices.Size(); i++)
      {
         const real_t *x = mesh.GetVertex(vertices[i]);
         MFEM_VERIFY(std::abs(x[1] - top) <= tolerance,
                     "An observation face does not lie on the top surface.");
         x_mid += x[0];
      }
      x_mid /= vertices.Size();
      MFEM_VERIFY(x_mid >= active_x_min - tolerance &&
                  x_mid <= active_x_max + tolerance,
                  "An observation face lies outside the active top footprint.");
      real_t z_mid = 0.0;
      for (int i = 0; i < vertices.Size(); i++)
      {
         z_mid += mesh.GetVertex(vertices[i])[2];
      }
      z_mid /= vertices.Size();
      MFEM_VERIFY(z_mid >= active_z_min - tolerance &&
                  z_mid <= active_z_max + tolerance,
                  "An observation face lies above a front/back sponge collar.");
      const real_t dx = x_mid - source_center_x;
      const real_t dz = z_mid - source_center_z;
      MFEM_VERIFY(dx * dx + dz * dz > source_radius * source_radius,
                  "An observation face overlaps the circular source patch.");
   }
}

void VerifyTopCircularPatch(const Mesh &mesh, int attribute,
                            real_t center_x, real_t center_z,
                            real_t radius)
{
   constexpr real_t top = 0.75;
   constexpr real_t tolerance = 2.0e-14;
   int source_faces = 0;
   for (int be = 0; be < mesh.GetNBE(); be++)
   {
      const Element *element = mesh.GetBdrElement(be);
      if (element->GetAttribute() != attribute) { continue; }
      Array<int> vertices;
      element->GetVertices(vertices);
      MFEM_VERIFY(vertices.Size() == 4,
                  "Elastic-inclusion 3D source element must be a quad.");
      real_t x_mid = 0.0;
      real_t z_mid = 0.0;
      for (int i = 0; i < vertices.Size(); i++)
      {
         const real_t *x = mesh.GetVertex(vertices[i]);
         MFEM_VERIFY(std::abs(x[1] - top) <= tolerance,
                     "A source face does not lie on the top surface.");
         x_mid += x[0];
         z_mid += x[2];
      }
      x_mid /= vertices.Size();
      z_mid /= vertices.Size();
      const real_t dx = x_mid - center_x;
      const real_t dz = z_mid - center_z;
      MFEM_VERIFY(dx * dx + dz * dz <= radius * radius + tolerance,
                  "A source face lies outside the circular top patch.");
      source_faces++;
   }
   MFEM_VERIFY(source_faces > 0, "The circular source patch is empty.");
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   const MPI_Comm comm = MPI_COMM_WORLD;
   Device device("cpu");

   TransientTopOptConfig base;
   ElasticInclusionIdentification3DProblem problem(base);
   const TransientTopOptConfig &cfg = problem.GetConfig();

   std::ostringstream validation_error;
   MFEM_VERIFY(problem.Validate(validation_error),
               "Default single-ball configuration is invalid: "
               << validation_error.str());
   MFEM_VERIFY(std::abs(cfg.material.r_min - 0.1) < 1e-15 &&
               std::abs(cfg.material.r_max - 1.0) < 1e-15 &&
               std::abs(cfg.material.simp_p - 3.0) < 1e-15,
               "The single-ball SIMP defaults changed unexpectedly.");
   MFEM_VERIFY(std::abs(cfg.boundary_load.frequency - 4.0) < 1e-15 &&
               std::abs(cfg.boundary_load.duration - 0.75) < 1e-15,
               "The single-ball source defaults changed unexpectedly.");
   MFEM_VERIFY(cfg.boundary_load.direction.Size() == 3 &&
               std::abs(cfg.boundary_load.direction[2]) < 1e-15,
               "The 3D source must retain the x/y oblique 2D loading plane.");

   using Ball = ElasticInclusionSingleBallTruthCoefficient;
   const real_t ball_density = Ball::BallDensity(cfg.material);
   const real_t ball_scale = cfg.material.r_min +
      (cfg.material.r_max - cfg.material.r_min) *
      std::pow(ball_density, cfg.material.simp_p);
   MFEM_VERIFY(std::abs(ball_scale - Ball::BallTargetMaterialScale()) < 1e-14,
               "The raw ball density no longer maps to SIMP scale 0.125.");

   std::unique_ptr<Coefficient> truth = problem.CreateTruthDensityCoefficient();
   const auto *ball_truth =
      dynamic_cast<const ElasticInclusionSingleBallTruthCoefficient *>(
         truth.get());
   MFEM_VERIFY(ball_truth != nullptr &&
               std::abs(ball_truth->GetBallDensity() - ball_density) < 1e-15,
               "The single-ball problem created the wrong truth coefficient.");

   constexpr real_t active_volume = (1.25 - 0.25) * (0.65 - 0.25) *
                                    (0.75 - 0.25);
   const real_t expected_volume_fraction =
      1.0 - (1.0 - ball_density) * Ball::BallVolume() / active_volume;
   MFEM_VERIFY(std::abs(problem.GetAnalyticTruthVolumeFraction() -
                        expected_volume_fraction) < 1e-14,
               "The analytic single-ball raw volume fraction is incorrect.");

   Mesh serial_mesh = problem.CreateMesh();
   const int source_attribute = problem.GetSourceBoundaryAttribute();
   const int receiver_attribute = problem.GetReceiverBoundaryAttribute();
   MFEM_VERIFY(source_attribute == 7 && receiver_attribute == 8,
               "Elastic-inclusion 3D source/receiver attributes changed.");
   // At h=0.025, 52 top quads have their centroids in the radius-0.10 disk.
   // The applied source area is therefore 52*h^2=0.0325, an explicit centroid
   // approximation to the physical pi*0.1^2 patch.
   MFEM_VERIFY(std::abs(BoundaryArea(serial_mesh, source_attribute) - 0.0325) <
               1e-13,
               "The top 3D circular source patch has the wrong mesh area.");
   // The accessible top footprint has area 1.0*0.5=0.5. Removing the 52
   // source quads gives a receiver area of 0.4675.
   MFEM_VERIFY(std::abs(BoundaryArea(serial_mesh, receiver_attribute) - 0.4675) <
               1e-13,
               "The top 3D receiver surface has the wrong mesh area.");
   VerifyTopReceiverSurface(serial_mesh, receiver_attribute,
                            problem.GetActiveXMin(), problem.GetActiveXMax(),
                            problem.GetActiveZMin(), problem.GetActiveZMax(),
                            problem.GetSourceCenterX(),
                            problem.GetSourceCenterZ(),
                            problem.GetSourceRadius());
   VerifyTopCircularPatch(serial_mesh, source_attribute,
                          problem.GetSourceCenterX(),
                          problem.GetSourceCenterZ(),
                          problem.GetSourceRadius());

   Array<int> observation_attributes;
   problem.GetObservationBoundaryAttributes(observation_attributes);
   MFEM_VERIFY(observation_attributes.Size() == 1 &&
               observation_attributes[0] == receiver_attribute,
               "Observations must use the top 3D receiver surface.");

   // The three-shot acquisition shares this domain/truth but removes both
   // regularizers. In shot s the active traction disk is the only inaccessible
   // part of the otherwise measured top footprint; the other two pads are
   // load-free receiver faces.
   TransientTopOptConfig multi_base;
   ElasticInclusionIdentification3DProblem multi_problem(
      multi_base, /*multi_source=*/true);
   const TransientTopOptConfig &multi_cfg = multi_problem.GetConfig();
   std::ostringstream multi_validation_error;
   MFEM_VERIFY(multi_problem.Validate(multi_validation_error),
               "Default multi-source configuration is invalid: "
               << multi_validation_error.str());
   MFEM_VERIFY(multi_problem.GetNumberOfSources() == 3 &&
               !multi_cfg.helmholtz_filter_enabled &&
               !multi_cfg.volume_constraint_enabled &&
               std::abs(multi_cfg.material.simp_p - 1.0) < 1e-15,
               "The multi-source inverse must use no filter, no volume "
               "constraint, and linear SIMP.");
   MFEM_VERIFY(!multi_problem.DefaultReferenceConvergenceAudit(),
               "The multi-source baseline must not silently request a "
               "three-run convergence audit per shot.");

   // The square pyramid has the same centroid and material volume as the
   // ball, but points upward toward the central source.  It is specifically
   // available to the three-shot matrix-free inverse experiment.
   TransientTopOptConfig pyramid_base;
   ElasticInclusionIdentification3DProblem pyramid_problem(
      pyramid_base, /*multi_source=*/true,
      ElasticInclusion3DTruthPreset::SQUARE_PYRAMID);
   const TransientTopOptConfig &pyramid_cfg = pyramid_problem.GetConfig();
   std::ostringstream pyramid_validation_error;
   MFEM_VERIFY(pyramid_problem.Validate(pyramid_validation_error),
               "Square-pyramid multi-source configuration is invalid: "
               << pyramid_validation_error.str());
   MFEM_VERIFY(pyramid_problem.GetTruthPreset() ==
               ElasticInclusion3DTruthPreset::SQUARE_PYRAMID &&
               std::string(pyramid_problem.GetTruthPresetName()) ==
               "square-pyramid",
               "The 3D pyramid truth preset was not retained.");
   using Pyramid = ElasticInclusionSquarePyramidTruthCoefficient;
   MFEM_VERIFY(std::abs(Pyramid::PyramidVolume() - Ball::BallVolume()) <
               1e-15 &&
               std::abs(Pyramid::CentroidY() - Ball::BallCenterY()) < 1e-15 &&
               std::abs(Pyramid::CenterX() - Ball::BallCenterX()) < 1e-15 &&
               std::abs(Pyramid::CenterZ() - Ball::BallCenterZ()) < 1e-15 &&
               std::abs(Pyramid::ApexY() - 0.60) < 1e-15,
               "The square pyramid no longer matches the ball's volume and "
               "centroid or point toward the top source.");
   MFEM_VERIFY(std::abs(Pyramid::BaseY() - pyramid_problem.GetActiveYMin()) >
               1e-12 &&
               Pyramid::BaseY() > pyramid_problem.GetActiveYMin() &&
               Pyramid::ApexY() < pyramid_problem.GetActiveYMax(),
               "The square pyramid must lie strictly inside the active box.");
   const real_t pyramid_density = Pyramid::PyramidDensity(pyramid_cfg.material);
   MFEM_VERIFY(std::abs(pyramid_density -
                        Ball::BallDensity(pyramid_cfg.material)) < 1e-15,
               "The pyramid and ball must use the same material contrast.");
   std::unique_ptr<Coefficient> pyramid_truth =
      pyramid_problem.CreateTruthDensityCoefficient();
   auto *pyramid_coefficient =
      dynamic_cast<ElasticInclusionSquarePyramidTruthCoefficient *>(
         pyramid_truth.get());
   MFEM_VERIFY(pyramid_coefficient != nullptr &&
               std::abs(pyramid_coefficient->GetPyramidDensity() -
                        pyramid_density) < 1e-15,
               "The pyramid problem created the wrong truth coefficient.");
   Mesh truth_mesh = Mesh::MakeCartesian3D(
      1, 1, 1, Element::HEXAHEDRON, 1.5, 0.75, 1.0);
   ElementTransformation *truth_transformation =
      truth_mesh.GetElementTransformation(0);
   const auto evaluate_pyramid = [&](real_t x, real_t y, real_t z)
   {
      IntegrationPoint ip;
      ip.Set3(x / 1.5, y / 0.75, z);
      return pyramid_coefficient->Eval(*truth_transformation, ip);
   };
   MFEM_VERIFY(std::abs(evaluate_pyramid(0.75, 0.40, 0.50) -
                        pyramid_density) < 1e-15 &&
               std::abs(evaluate_pyramid(0.75, 0.59, 0.50) -
                        pyramid_density) < 1e-15 &&
               std::abs(evaluate_pyramid(0.79, 0.55, 0.50) - 1.0) < 1e-15 &&
               std::abs(evaluate_pyramid(0.75, 0.61, 0.50) - 1.0) < 1e-15,
               "The pyramid truth does not have the expected square base, "
               "source-facing apex, and exterior.");
   const real_t expected_pyramid_volume_fraction =
      1.0 - (1.0 - pyramid_density) * Pyramid::PyramidVolume() /
      active_volume;
   MFEM_VERIFY(std::abs(pyramid_problem.GetAnalyticTruthVolumeFraction() -
                        expected_pyramid_volume_fraction) < 1e-14,
               "The analytic square-pyramid raw volume fraction is incorrect.");

   Mesh multi_mesh = multi_problem.CreateMesh();
   const int multi_receiver_attribute =
      multi_problem.GetReceiverBoundaryAttribute();
   real_t source_area_sum = 0.0;
   for (int source = 0; source < multi_problem.GetNumberOfSources(); source++)
   {
      const int source_attribute =
         multi_problem.GetSourceBoundaryAttribute(source);
      const real_t source_area = BoundaryArea(multi_mesh, source_attribute);
      source_area_sum += source_area;
      MFEM_VERIFY(std::abs(source_area - 0.0325) < 1e-13,
                  "A multi-source top disk has the wrong centroidal area.");
      VerifyTopCircularPatch(multi_mesh, source_attribute,
                             multi_problem.GetSourceCenterX(source),
                             multi_problem.GetSourceCenterZ(source),
                             multi_problem.GetSourceRadius());

      Array<int> shot_observation_attributes;
      multi_problem.GetObservationBoundaryAttributes(
         source, shot_observation_attributes);
      MFEM_VERIFY(shot_observation_attributes.Size() == 3 &&
                  ContainsAttribute(shot_observation_attributes,
                                    multi_receiver_attribute) &&
                  !ContainsAttribute(shot_observation_attributes,
                                     source_attribute),
                  "A shot must observe the receiver and both load-free pads, "
                  "but not its active source pad.");
      real_t shot_observed_area = 0.0;
      for (int i = 0; i < shot_observation_attributes.Size(); i++)
      {
         shot_observed_area += BoundaryArea(
            multi_mesh, shot_observation_attributes[i]);
      }
      MFEM_VERIFY(std::abs(shot_observed_area - 0.4675) < 1e-13,
                  "A multi-source shot does not observe the full active top "
                  "footprint minus exactly its own source disk.");
   }
   MFEM_VERIFY(std::abs(BoundaryArea(multi_mesh, multi_receiver_attribute) -
                        (0.5 - source_area_sum)) < 1e-13,
               "The multi-source receiver attribute has the wrong residual "
               "top-surface area.");

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
   MFEM_VERIFY(std::abs(objective.ObservedBoundaryMeasure() - 0.4675) < 1e-13,
               "The 3D inverse objective does not observe the full accessible "
               "top footprint minus the source.");

   if (Mpi::Root())
   {
      mfem::out << "Elastic single-ball 3D configuration regression passed\n"
                << "  receiver attribute/area: " << receiver_attribute
                << " / " << objective.ObservedBoundaryMeasure() << '\n'
                << "  source attribute/area: " << source_attribute
                << " / " << BoundaryArea(serial_mesh, source_attribute) << '\n'
                << "  SIMP law: s(rho)=0.1+0.9*rho^3\n"
                << "  ball raw density: " << std::setprecision(15)
                << ball_density << '\n'
                << "  ball material scale: " << ball_scale << '\n'
                << "  analytic raw volume fraction: "
                << expected_volume_fraction << '\n'
                << "  pyramid raw density: " << pyramid_density << '\n'
                << "  pyramid analytic raw volume fraction: "
                << expected_pyramid_volume_fraction << '\n'
                << "  multi-shot receiver area per shot: 0.4675\n";
   }
   return 0;
}
