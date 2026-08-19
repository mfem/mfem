// =============================================================================
// Coupled temporal-refinement verification for the continuous design adjoint
// =============================================================================
//
// This regression deliberately uses a tiny, fixed spatial discretization.  It
// therefore measures temporal convergence of one semi-discrete problem; no
// changing spatial mesh can hide or contaminate the expected RK4/Hermite order.
//
// The test checks
//   1. fixed-T refinement with dt_a = dt_f/3,
//   2. objective, initial-adjoint, filtered-gradient, and raw-gradient
//      convergence to an independently finer temporal reference,
//   3. filtered/raw Richardson-extrapolated directional finite differences,
//   4. second-order first-order Taylor remainders.
//
// =============================================================================

#include "mfem.hpp"
#include "ElastodynamicsSolver.hpp"
#include "../../pde_filter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace mfem;

namespace
{

real_t GlobalDot(MPI_Comm comm, const Vector &a, const Vector &b)
{
   return InnerProduct(comm, a, b);
}

real_t GlobalNorm(MPI_Comm comm, const Vector &x)
{
   return std::sqrt(std::max(GlobalDot(comm, x, x), real_t(0.0)));
}

void Normalize(MPI_Comm comm, Vector &x)
{
   const real_t norm = GlobalNorm(comm, x);
   MFEM_VERIFY(std::isfinite(norm) && norm > 0.0,
               "Cannot normalize a zero or non-finite direction.");
   x /= norm;
}

real_t RelativeScalarError(real_t value, real_t reference)
{
   const real_t scale =
      std::max(std::abs(reference), real_t(100.0) *
               std::numeric_limits<real_t>::epsilon());
   return std::abs(value - reference) / scale;
}

real_t RelativeVectorError(MPI_Comm comm,
                           const Vector &value,
                           const Vector &reference)
{
   Vector difference(value);
   difference -= reference;
   const real_t scale =
      std::max(GlobalNorm(comm, reference), real_t(100.0) *
               std::numeric_limits<real_t>::epsilon());
   return GlobalNorm(comm, difference) / scale;
}

real_t GlobalAdmissibleStep(MPI_Comm comm,
                            const Vector &base,
                            const Vector &direction,
                            real_t lower,
                            real_t upper)
{
   MFEM_VERIFY(base.Size() == direction.Size() && lower < upper,
               "Invalid density perturbation data.");
   real_t local_step = std::numeric_limits<real_t>::infinity();
   for (int i = 0; i < base.Size(); i++)
   {
      const real_t d = direction[i];
      if (d > 0.0)
      {
         local_step = std::min(local_step, (upper - base[i]) / d);
      }
      else if (d < 0.0)
      {
         local_step = std::min(local_step, (base[i] - lower) / (-d));
      }
   }
   real_t global_step = local_step;
   MPI_Allreduce(&local_step, &global_step, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_MIN, comm);
   MFEM_VERIFY(std::isfinite(global_step) && global_step > 0.0,
               "No admissible density perturbation is available.");
   return global_step;
}

struct TemporalSample
{
   int forward_steps = 0;
   real_t forward_step = 0.0;
   ContinuousDesignRunResult run;
   Vector raw_gradient;
   real_t objective_error = 0.0;
   real_t initial_adjoint_error = 0.0;
   real_t filtered_gradient_error = 0.0;
   real_t raw_gradient_error = 0.0;
};

struct DirectionalCheck
{
   std::string label;
   real_t projected_gradient = 0.0;
   real_t extrapolated_fd = 0.0;
   real_t fd_relative_error = 0.0;
   real_t richardson_relative_change = 0.0;
   std::array<real_t, 4> perturbations{};
   std::array<real_t, 4> objective_plus{};
   std::array<real_t, 4> objective_minus{};
   std::array<real_t, 4> centered_fd{};
   std::array<real_t, 4> taylor_remainder{};
   std::array<real_t, 3> taylor_order{};
};

template <typename EvaluatePerturbation>
DirectionalCheck CheckDirection(
   const std::string &label,
   MPI_Comm comm,
   const Vector &base,
   const Vector &direction,
   const Vector &gradient,
   real_t base_objective,
   EvaluatePerturbation &&evaluate_perturbation)
{
   DirectionalCheck check;
   check.label = label;
   check.projected_gradient = GlobalDot(comm, gradient, direction);
   MFEM_VERIFY(std::isfinite(check.projected_gradient) &&
               std::abs(check.projected_gradient) > 1e-12,
               "Temporal Taylor direction has a negligible derivative.");

   const real_t admissible =
      GlobalAdmissibleStep(comm, base, direction, 0.05, 0.95);
   const real_t initial_step =
      std::min(real_t(0.04), real_t(0.20) * admissible);
   MFEM_VERIFY(initial_step > 1e-5,
               "Admissible Taylor perturbation is unexpectedly small.");

   Vector candidate(base.Size());
   for (int level = 0; level < 4; level++)
   {
      const real_t epsilon = initial_step / std::pow(real_t(2.0), level);
      check.perturbations[level] = epsilon;

      candidate = base;
      candidate.Add(epsilon, direction);
      check.objective_plus[level] = evaluate_perturbation(candidate);

      candidate = base;
      candidate.Add(-epsilon, direction);
      check.objective_minus[level] = evaluate_perturbation(candidate);

      check.centered_fd[level] =
         (check.objective_plus[level] - check.objective_minus[level]) /
         (2.0 * epsilon);
      check.taylor_remainder[level] =
         std::abs(check.objective_plus[level] - base_objective -
                  epsilon * check.projected_gradient);
      MFEM_VERIFY(std::isfinite(check.centered_fd[level]) &&
                  std::isfinite(check.taylor_remainder[level]) &&
                  check.taylor_remainder[level] > 0.0,
                  "Directional temporal verification produced invalid data.");
   }

   std::array<real_t, 3> richardson{};
   for (int level = 0; level < 3; level++)
   {
      richardson[level] =
         (4.0 * check.centered_fd[level + 1] -
          check.centered_fd[level]) / 3.0;
      check.taylor_order[level] =
         std::log(check.taylor_remainder[level] /
                  check.taylor_remainder[level + 1]) / std::log(real_t(2.0));
   }
   check.extrapolated_fd = richardson[2];
   const real_t derivative_scale =
      std::max({std::abs(check.projected_gradient),
                std::abs(check.extrapolated_fd), real_t(1e-14)});
   check.fd_relative_error =
      std::abs(check.extrapolated_fd - check.projected_gradient) /
      derivative_scale;
   check.richardson_relative_change =
      std::abs(richardson[2] - richardson[1]) / derivative_scale;
   return check;
}

real_t MinimumReliableOrder(const std::vector<real_t> &errors,
                            const char *label)
{
   MFEM_VERIFY(errors.size() >= 4,
               "Temporal convergence needs four refinement levels.");
   std::vector<real_t> orders;
   for (std::size_t level = 0; level + 1 < errors.size(); level++)
   {
      MFEM_VERIFY(std::isfinite(errors[level]) &&
                  std::isfinite(errors[level + 1]) &&
                  errors[level] > 0.0 && errors[level + 1] > 0.0,
                  "Temporal convergence contains a zero or invalid error.");
      if (errors[level] >
          1e3 * std::numeric_limits<real_t>::epsilon())
      {
         const real_t order =
            std::log(errors[level] / errors[level + 1]) /
            std::log(real_t(2.0));
         if (std::isfinite(order)) { orders.push_back(order); }
      }
   }
   MFEM_VERIFY(orders.size() >= 2,
               "Temporal errors reached roundoff before an order was visible.");

   // The coarsest pair can be pre-asymptotic.  The final two reliable orders
   // are the meaningful RK4/Hermite convergence gate.
   const real_t minimum =
      std::min(orders[orders.size() - 1], orders[orders.size() - 2]);
   MFEM_VERIFY(std::isfinite(minimum),
               "Temporal convergence order is non-finite.");
   (void)label;
   return minimum;
}

void PrintDirectionalCheck(const DirectionalCheck &check)
{
   if (!Mpi::Root()) { return; }
   mfem::out << "\n" << check.label << " directional verification\n"
             << "  projected gradient: " << std::scientific
             << std::setprecision(12) << check.projected_gradient << '\n'
             << "  Richardson FD:      " << check.extrapolated_fd << '\n'
             << "  relative FD error:  " << check.fd_relative_error << '\n'
             << "  Richardson change:  "
             << check.richardson_relative_change << "\n"
             << "  epsilon          centered FD          first-order remainder"
                "    Taylor order\n";
   for (int level = 0; level < 4; level++)
   {
      mfem::out << "  " << std::setw(12) << check.perturbations[level]
                << "  " << std::setw(19) << check.centered_fd[level]
                << "  " << std::setw(24) << check.taylor_remainder[level];
      if (level < 3)
      {
         mfem::out << "  " << std::setw(12)
                   << check.taylor_order[level];
      }
      mfem::out << '\n';
   }
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();
   MPI_Comm comm = MPI_COMM_WORLD;

   int nx = 4;
   int ny = 2;
   int minimum_coarse_steps = 8;
   int reference_factor = 4;
   real_t final_time = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&nx, "-nx", "--elements-x",
                  "Tiny-mesh elements in the x direction");
   args.AddOption(&ny, "-ny", "--elements-y",
                  "Tiny-mesh elements in the y direction");
   args.AddOption(&minimum_coarse_steps, "-n0", "--minimum-coarse-steps",
                  "Minimum number of coarsest forward steps");
   args.AddOption(&reference_factor, "-rf", "--reference-factor",
                  "Reference refinement beyond the finest tested grid");
   args.AddOption(&final_time, "-tf", "--final-time",
                  "Fixed final time");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(std::cout); }
      return 1;
   }
   MFEM_VERIFY(nx >= 2 && ny >= 2 && minimum_coarse_steps >= 4 &&
               reference_factor >= 4 && final_time > 0.0,
               "Invalid temporal-refinement test controls.");

   Device device("cpu");

   Mesh serial_mesh = Mesh::MakeCartesian2D(
      nx, ny, Element::TRIANGLE, true, 1.0, 0.5);
   ParMesh mesh(comm, serial_mesh);

   constexpr int dimension = 2;
   H1_FECollection state_collection(/*order=*/2, dimension);
   H1_FECollection filter_collection(/*order=*/1, dimension);
   L2_FECollection control_collection(
      /*order=*/0, dimension, BasisType::GaussLobatto);
   ParFiniteElementSpace state_fes(
      &mesh, &state_collection, dimension);
   ParFiniteElementSpace filter_fes(&mesh, &filter_collection);
   ParFiniteElementSpace control_fes(&mesh, &control_collection);

   ParGridFunction rho(&control_fes);
   ParGridFunction rho_tilde(&filter_fes);
   rho = 0.65;

   toopt::PDEFilterOptions filter_options;
   filter_options.filter_radius = 0.08;
   filter_options.solver_rtol = 1e-12;
   filter_options.solver_atol = 1e-14;
   filter_options.solver_maxiter = 200;
   toopt::PDEFilter filter(filter_fes, control_fes, filter_options);
   filter.Assemble();
   filter.Mult(rho, rho_tilde);

   MaterialParams material;
   // Slow the tiny manufactured solid so T=1 contains useful smooth dynamics
   // without requiring hundreds of steps merely for CFL stability.
   material.rho0 = 1.0;
   material.lambda0 = 0.02;
   material.mu0 = 0.01;

   ConstantCoefficient rho0_coefficient(material.rho0);
   ConstantCoefficient lambda0_coefficient(material.lambda0);
   ConstantCoefficient mu0_coefficient(material.mu0);
   SIMPCoefficient simp_mass(
      &rho_tilde, material.r_min, material.r_max, material.simp_p);
   SIMPCoefficient simp_stiffness(
      &rho_tilde, material.r_min, material.r_max, material.simp_p);
   ProductCoefficient mass_coefficient(simp_mass, rho0_coefficient);
   ProductCoefficient lambda_coefficient(
      simp_stiffness, lambda0_coefficient);
   ProductCoefficient mu_coefficient(simp_stiffness, mu0_coefficient);

   BoundaryLoadSpec load;
   load.domain_load = true;
   load.amplitude = 1.0;
   load.duration = final_time;
   load.time_profile = LoadTimeProfile::HARMONIC;
   load.frequency = 0.75;
   load.phase = 0.2;
   load.direction.SetSize(dimension);
   load.direction = 0.0;
   load.direction[1] = -1.0;
   DirectionalBoundaryLoadCoefficient load_coefficient(load.direction);

   ConstantCoefficient damping_coefficient(0.04);
   Array<int> exterior_boundary(mesh.bdr_attributes.Max());
   exterior_boundary = 0;
   Array<int> essential_boundary(mesh.bdr_attributes.Max());
   essential_boundary = 0;
   // Cartesian MFEM meshes label the left boundary with attribute 4.
   MFEM_VERIFY(essential_boundary.Size() >= 4,
               "Tiny Cartesian mesh has unexpected boundary attributes.");
   essential_boundary[3] = 1;

   ElastodynamicsOperator oper(
      state_fes, mass_coefficient, lambda_coefficient, mu_coefficient,
      load.amplitude, load.duration, load.time_profile,
      load.phase, load.frequency, load.bdr_attributes,
      load_coefficient, load.domain_load, &damping_coefficient,
      /*impedance=*/0.0, exterior_boundary, essential_boundary,
      MassSolverType::LUMPED, /*print_banner=*/false);

   Vector target_value(dimension);
   target_value = 0.0;
   target_value[0] = 1.0;
   target_value[1] = 0.25;
   auto tracking_region = std::make_unique<ConstantCoefficient>(1.0);
   auto tracking_mode =
      std::make_unique<VectorConstantCoefficient>(target_value);
   HarmonicDisplacementTrackingObjective objective(
      &state_fes, std::move(tracking_region), std::move(tracking_mode),
      /*amplitude=*/0.35, /*frequency=*/0.40, /*phase=*/0.1, comm);

   Vector initial_state(oper.Width());
   initial_state = 0.0;

   const real_t wave_limit = oper.EstimateLumpedRK4TimeStep();
   const real_t damping_limit = oper.EstimateLumpedRK4DampingTimeStep();
   const real_t stability_limit = std::min(wave_limit, damping_limit);
   MFEM_VERIFY(std::isfinite(stability_limit) && stability_limit > 0.0,
               "Tiny manufactured operator has no finite CFL estimate.");
   const int cfl_steps = static_cast<int>(
      std::ceil(final_time / (0.30 * stability_limit)));
   const int coarse_steps = std::max(minimum_coarse_steps, cfl_steps);
   const real_t coarse_step = final_time / coarse_steps;
   ValidateLumpedRK4TimeStep(
      oper, coarse_step, /*print_report=*/false);

   constexpr int adjoint_refinement = 3;
   constexpr int tested_levels = 5;
   std::vector<TemporalSample> samples(tested_levels);
   for (int level = 0; level < tested_levels; level++)
   {
      TemporalSample &sample = samples[level];
      sample.forward_steps = coarse_steps * (1 << level);
      sample.forward_step = final_time / sample.forward_steps;
      sample.run = RunContinuousDesignFullStorage(
         oper, state_fes, filter_fes, rho_tilde, material, objective,
         initial_state, sample.forward_steps, sample.forward_step,
         adjoint_refinement);
      filter.MultTranspose(sample.run.gradient_tilde, sample.raw_gradient);
   }

   const int finest_steps = samples.back().forward_steps;
   MFEM_VERIFY(
      finest_steps <= std::numeric_limits<int>::max() / reference_factor,
      "Temporal reference step count overflows int.");
   const int reference_steps = reference_factor * finest_steps;
   constexpr int reference_adjoint_refinement = 6;
   ContinuousDesignRunResult reference =
      RunContinuousDesignFullStorage(
         oper, state_fes, filter_fes, rho_tilde, material, objective,
         initial_state, reference_steps, final_time / reference_steps,
         reference_adjoint_refinement);
   Vector reference_raw_gradient(control_fes.GetTrueVSize());
   filter.MultTranspose(reference.gradient_tilde, reference_raw_gradient);

   std::vector<real_t> objective_errors;
   std::vector<real_t> initial_adjoint_errors;
   std::vector<real_t> filtered_gradient_errors;
   std::vector<real_t> raw_gradient_errors;
   for (TemporalSample &sample : samples)
   {
      sample.objective_error =
         RelativeScalarError(sample.run.objective, reference.objective);
      sample.initial_adjoint_error =
         RelativeVectorError(
            comm, sample.run.initial_adjoint, reference.initial_adjoint);
      sample.filtered_gradient_error =
         RelativeVectorError(
            comm, sample.run.gradient_tilde, reference.gradient_tilde);
      sample.raw_gradient_error =
         RelativeVectorError(
            comm, sample.raw_gradient, reference_raw_gradient);
      objective_errors.push_back(sample.objective_error);
      initial_adjoint_errors.push_back(sample.initial_adjoint_error);
      filtered_gradient_errors.push_back(sample.filtered_gradient_error);
      raw_gradient_errors.push_back(sample.raw_gradient_error);
   }

   const real_t objective_order =
      MinimumReliableOrder(objective_errors, "objective");
   const real_t initial_adjoint_order =
      MinimumReliableOrder(initial_adjoint_errors, "initial adjoint");
   const real_t filtered_gradient_order =
      MinimumReliableOrder(filtered_gradient_errors, "filtered gradient");
   const real_t raw_gradient_order =
      MinimumReliableOrder(raw_gradient_errors, "raw gradient");

   Vector filtered_base, raw_base;
   rho_tilde.GetTrueDofs(filtered_base);
   rho.GetTrueDofs(raw_base);
   Vector filtered_direction(reference.gradient_tilde);
   Vector raw_direction(reference_raw_gradient);
   Normalize(comm, filtered_direction);
   Normalize(comm, raw_direction);

   const auto evaluate_current_filtered_design = [&]()
   {
      ConstantCoefficient perturbed_rho0(material.rho0);
      ConstantCoefficient perturbed_lambda0(material.lambda0);
      ConstantCoefficient perturbed_mu0(material.mu0);
      SIMPCoefficient perturbed_simp_mass(
         &rho_tilde, material.r_min, material.r_max, material.simp_p);
      SIMPCoefficient perturbed_simp_stiffness(
         &rho_tilde, material.r_min, material.r_max, material.simp_p);
      ProductCoefficient perturbed_mass(
         perturbed_simp_mass, perturbed_rho0);
      ProductCoefficient perturbed_lambda(
         perturbed_simp_stiffness, perturbed_lambda0);
      ProductCoefficient perturbed_mu(
         perturbed_simp_stiffness, perturbed_mu0);
      ElastodynamicsOperator perturbed_oper(
         state_fes, perturbed_mass, perturbed_lambda, perturbed_mu,
         load.amplitude, load.duration, load.time_profile,
         load.phase, load.frequency, load.bdr_attributes,
         load_coefficient, load.domain_load, &damping_coefficient,
         /*impedance=*/0.0, exterior_boundary, essential_boundary,
         MassSolverType::LUMPED, /*print_banner=*/false);
      std::vector<Vector> states;
      return ContinuousForwardSweepFullStorage(
         perturbed_oper, state_fes, objective, initial_state,
         reference_steps, /*start_time=*/0.0,
         final_time / reference_steps, reference_adjoint_refinement,
         states);
   };

   const auto evaluate_filtered = [&](const Vector &candidate)
   {
      rho_tilde.SetFromTrueDofs(candidate);
      return evaluate_current_filtered_design();
   };
   const auto evaluate_raw = [&](const Vector &candidate)
   {
      rho.SetFromTrueDofs(candidate);
      filter.Mult(rho, rho_tilde);
      return evaluate_current_filtered_design();
   };

   const DirectionalCheck filtered_check =
      CheckDirection(
         "Filtered design", comm, filtered_base, filtered_direction,
         reference.gradient_tilde, reference.objective, evaluate_filtered);
   rho_tilde.SetFromTrueDofs(filtered_base);
   rho.SetFromTrueDofs(raw_base);

   const DirectionalCheck raw_check =
      CheckDirection(
         "Raw design", comm, raw_base, raw_direction,
         reference_raw_gradient, reference.objective, evaluate_raw);

   // Restore both shared coefficient fields exactly before any reporting or
   // verification failure can leave this reusable test fixture perturbed.
   rho.SetFromTrueDofs(raw_base);
   rho_tilde.SetFromTrueDofs(filtered_base);

   if (Mpi::Root())
   {
      mfem::out
         << "\n=== Continuous Design Temporal Refinement ===\n"
         << "Spatial fixture: " << nx << " x " << ny
         << " triangles, state order 2, design order 1/0\n"
         << "State true size: " << oper.Width()
         << ", filtered/raw design sizes: "
         << filter_fes.GlobalTrueVSize() << "/"
         << control_fes.GlobalTrueVSize() << '\n'
         << "T=" << std::scientific << std::setprecision(6) << final_time
         << ", CFL limit=" << stability_limit
         << ", N0=" << coarse_steps
         << ", m=" << adjoint_refinement
         << ", reference N=" << reference_steps
         << ", reference m=" << reference_adjoint_refinement << "\n\n"
         << " N_f          dt_f           rel J          rel p(0)"
            "       rel g_tilde       rel g_raw\n";
      for (const TemporalSample &sample : samples)
      {
         mfem::out << std::setw(4) << sample.forward_steps
                   << "  " << std::setw(13) << sample.forward_step
                   << "  " << std::setw(13) << sample.objective_error
                   << "  " << std::setw(13) << sample.initial_adjoint_error
                   << "  " << std::setw(15)
                   << sample.filtered_gradient_error
                   << "  " << std::setw(13) << sample.raw_gradient_error
                   << '\n';
      }
      mfem::out
         << "\nMinimum of final two reliable orders:\n"
         << "  objective:         " << objective_order << '\n'
         << "  initial adjoint:   " << initial_adjoint_order << '\n'
         << "  filtered gradient: " << filtered_gradient_order << '\n'
         << "  raw gradient:      " << raw_gradient_order << '\n'
         << "Reference norms: ||p0||="
         << GlobalNorm(comm, reference.initial_adjoint)
         << ", ||g_tilde||="
         << GlobalNorm(comm, reference.gradient_tilde)
         << ", ||g_raw||="
         << GlobalNorm(comm, reference_raw_gradient) << '\n';
   }
   PrintDirectionalCheck(filtered_check);
   PrintDirectionalCheck(raw_check);

   // These are deliberately accuracy claims, not loose smoke-test bounds.
   // The tiny smooth fixture remains far above roundoff at the finest tested
   // grid, while its independent reference is four times finer again.
   MFEM_VERIFY(objective_order > 3.55,
               "Continuous objective did not converge at fourth order.");
   MFEM_VERIFY(initial_adjoint_order > 3.45,
               "Continuous initial adjoint did not converge at fourth order.");
   MFEM_VERIFY(filtered_gradient_order > 3.35,
               "Continuous filtered gradient did not converge at fourth order.");
   MFEM_VERIFY(raw_gradient_order > 3.35,
               "Continuous raw gradient did not converge at fourth order.");
   MFEM_VERIFY(samples.back().objective_error < 2e-5 &&
               samples.back().initial_adjoint_error < 2e-5 &&
               samples.back().filtered_gradient_error < 5e-5 &&
               samples.back().raw_gradient_error < 5e-5,
               "Finest tested temporal grid is not close to the reference.");

   const auto verify_direction = [](const DirectionalCheck &check,
                                    real_t fd_tolerance)
   {
      MFEM_VERIFY(check.fd_relative_error < fd_tolerance,
                  "Refined continuous gradient failed its directional FD check.");
      MFEM_VERIFY(check.richardson_relative_change < 2e-5,
                  "Directional finite difference is not Richardson-converged.");
      int quadratic_intervals = 0;
      for (real_t order : check.taylor_order)
      {
         if (std::isfinite(order) && order > 1.75 && order < 2.25)
         {
            quadratic_intervals++;
         }
      }
      MFEM_VERIFY(quadratic_intervals >= 2,
                  "First-order Taylor remainder has no second-order region.");
   };
   verify_direction(filtered_check, 2e-5);
   verify_direction(raw_check, 5e-5);

   if (Mpi::Root())
   {
      mfem::out
         << "\nAll coupled temporal-refinement, directional-FD, and Taylor "
            "checks passed.\n";
   }
   return 0;
}
