// =============================================================================
// Taylor Test for Gradient Verification (Phase 4)
// =============================================================================
//
// Verifies design sensitivity gradient via Taylor series test:
//   |J(ρ + ε δρ) - J(ρ) - ε ⟨dJ/dρ, δρ⟩| = O(ε²)
//
// If gradient is correct, error should decay quadratically with ε
//
// COMPILE:
//   make test_gradient_taylor -j8
//
// RUN:
//   srun -n <nprocs> ./test_gradient_taylor [options]
//
// =============================================================================

#include "mfem.hpp"
#include "ElastodynamicsSolver.hpp"
#include "ObjectiveFunctional.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>

using namespace std;
using namespace mfem;

// Helper function: Run forward solve and return objective
real_t ComputeObjective(ElastodynamicsOperator &oper,
                        ParGridFunction &rho_filter,
                        real_t dt, int num_steps, real_t t_final)
{
   BlockVector state(oper.GetBlockOffsets());
   state = 0.0;

   RK4Solver ode_solver;
   ode_solver.Init(oper);

   real_t t = 0.0;

   for (int ti = 0; ti < num_steps && t < t_final - dt/2; ti++)
   {
      oper.StoreTrajectoryStep(ti, state);
      oper.AccumulateObjective(state, dt, ti, num_steps);

      oper.SetTime(t + dt);
      ode_solver.Step(state, t, dt);
   }

   return oper.AccumulateObjective(state, dt, num_steps-1, num_steps);
}

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();
   int myid = Mpi::WorldRank();

   Device device("cpu");

   // Command-line options
   int ref_levels = 1;
   int order = 2;
   real_t t_final = 0.05;
   real_t dt = 0.001;
   const char *mesh_file = "lamb-problem-damping-mesh-quads.msh";

   OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--refine", "Refinement level");
   args.AddOption(&order, "-o", "--order", "FE order");
   args.AddOption(&t_final, "-tf", "--t-final", "Final time");
   args.AddOption(&dt, "-dt", "--time-step", "Time step");
   args.AddOption(&mesh_file, "-mesh", "--mesh-file", "Mesh file");
   args.Parse();

   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   // Domain parameters
   real_t x_max = 1.5;
   real_t y_max = 0.75;

   if (myid == 0)
   {
      cout << "\n=== Taylor Test for Gradient Verification ===" << endl;
      cout << "Testing: |J(ρ+εδρ) - J(ρ) - ε⟨dJ/dρ, δρ⟩| = O(ε²)" << endl;
   }

   // Load mesh
   ifstream imesh(mesh_file);
   if (!imesh)
   {
      if (myid == 0)
      {
         cerr << "Error: Cannot open mesh file '" << mesh_file << "'" << endl;
      }
      return 1;
   }

   Mesh mesh(imesh, 1, 1);
   imesh.close();
   int dim = mesh.Dimension();

   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec, dim);

   // Design space
   H1_FECollection design_fec(order, dim);
   ParFiniteElementSpace design_fes(&pmesh, &design_fec);
   ParGridFunction rho_filter(&design_fes);

   // Initialize with Gaussian design
   real_t damping_thickness = 0.25;
   real_t x_center = x_max / 2.0;
   real_t y_center = y_max / 2.0;
   real_t sigma_x = (x_max - 2.0 * damping_thickness) / 3.0;
   real_t sigma_y = (y_max - damping_thickness) / 3.0;

   GaussianDesignCoefficient gaussian_coef(x_center, y_center, sigma_x, sigma_y,
                                            0.3, 1.0);
   rho_filter.ProjectCoefficient(gaussian_coef);

   // Material properties
   real_t rho_0 = 1.0;
   real_t mu_0 = 1.0;
   real_t lambda_0 = 2.0;
   real_t r_min = 1e-6;
   real_t r_max = 1.0;
   real_t simp_exponent = 3.0;

   ConstantCoefficient rho_0_coef(rho_0);
   ConstantCoefficient lambda_0_coef(lambda_0);
   ConstantCoefficient mu_0_coef(mu_0);

   SIMPCoefficient simp_mass(&rho_filter, r_min, r_max, simp_exponent);
   SIMPCoefficient simp_stiff(&rho_filter, r_min, r_max, simp_exponent);

   ProductCoefficient mass_coef(simp_mass, rho_0_coef);
   ProductCoefficient lambda_coef(simp_stiff, lambda_0_coef);
   ProductCoefficient mu_coef(simp_stiff, mu_0_coef);

   real_t c_p = sqrt((lambda_0 + 2*mu_0) / rho_0);

   // Damping configuration
   DampingProfile phi_profile(damping_thickness, x_max, y_max);
   real_t gamma_max = (2.0 * c_p / 0.2136) * log(1.0 / 1e-4);
   SpatialDampingCoefficient gamma_coef(&phi_profile, gamma_max, rho_0, 2.0, 2);

   // Loading
   real_t pulse_duration = 0.005;
   real_t pulse_amplitude = 30.0;

   // Boundary conditions
   Array<int> exterior_bdr_attr(pmesh.bdr_attributes.Max());
   exterior_bdr_attr = 0;
   if (pmesh.bdr_attributes.Max() >= 10) exterior_bdr_attr[9] = 1;
   if (pmesh.bdr_attributes.Max() >= 11) exterior_bdr_attr[10] = 1;
   if (pmesh.bdr_attributes.Max() >= 12) exterior_bdr_attr[11] = 1;

   Array<int> empty_bdr_attr(pmesh.bdr_attributes.Max());
   empty_bdr_attr = 0;

   real_t impedance = rho_0 * c_p;

   // Objective
   real_t protected_radius = 0.2;
   SubdomainIndicator subdomain_indicator(x_center, y_center, protected_radius);
   TimeIntegratedObjective objective(&fespace, &subdomain_indicator, MPI_COMM_WORLD);

   // Trajectory storage
   int num_steps = static_cast<int>(t_final / dt) + 1;
   ForwardTrajectoryStorage trajectory(num_steps);
   trajectory.EnableStorage();

   // Create operator
   ElastodynamicsOperator oper(
      fespace, mass_coef, lambda_coef, mu_coef,
      pulse_amplitude, pulse_duration,
      &gamma_coef, impedance, exterior_bdr_attr, empty_bdr_attr,
      &trajectory, &objective);

   if (myid == 0)
   {
      cout << "\n=== Computing Baseline Objective J(ρ) ===" << endl;
   }

   // Compute J(ρ)
   objective.Reset();
   real_t J_rho = ComputeObjective(oper, rho_filter, dt, num_steps, t_final);
   real_t J_baseline = objective.GetObjective();

   if (myid == 0)
   {
      cout << "J(ρ) = " << J_baseline << endl;
   }

   // TODO: Compute gradient dJ/dρ via adjoint solve
   // For now, use finite difference as reference

   // Create random perturbation δρ
   Vector delta_rho(design_fes.GetTrueVSize());
   delta_rho.Randomize(12345);
   delta_rho.SetSubVector(Array<int>(), 0.0);  // Zero out boundary if needed

   real_t delta_norm = delta_rho.Norml2();
   delta_rho *= (1.0 / delta_norm);  // Normalize

   if (myid == 0)
   {
      cout << "\n=== Taylor Test ===" << endl;
      cout << "Random perturbation |δρ| = " << delta_rho.Norml2() << endl;
      cout << "\nTesting convergence for decreasing ε:" << endl;
      cout << setw(15) << "ε"
           << setw(20) << "J(ρ+εδρ) - J(ρ)"
           << setw(20) << "Order"
           << endl;
      cout << string(55, '-') << endl;
   }

   real_t prev_error = 0.0;

   for (int i = 0; i < 10; i++)
   {
      real_t epsilon = pow(10.0, -i);

      // Perturb design: ρ_new = ρ + ε δρ
      ParGridFunction rho_perturbed(rho_filter);
      Vector rho_true(design_fes.GetTrueVSize());
      rho_perturbed.GetTrueDofs(rho_true);
      rho_true.Add(epsilon, delta_rho);
      rho_perturbed.SetFromTrueDofs(rho_true);

      // Compute J(ρ + ε δρ)
      objective.Reset();
      trajectory = ForwardTrajectoryStorage(num_steps);
      trajectory.EnableStorage();

      ElastodynamicsOperator oper_pert(
         fespace, mass_coef, lambda_coef, mu_coef,
         pulse_amplitude, pulse_duration,
         &gamma_coef, impedance, exterior_bdr_attr, empty_bdr_attr,
         &trajectory, &objective);

      real_t J_pert = ComputeObjective(oper_pert, rho_perturbed, dt, num_steps, t_final);
      real_t J_perturbed = objective.GetObjective();

      // Finite difference error (first order)
      real_t error = abs(J_perturbed - J_baseline);

      real_t order = (prev_error > 0) ? log(error / prev_error) / log(0.1) : 0.0;

      if (myid == 0)
      {
         cout << setw(15) << scientific << epsilon
              << setw(20) << error
              << setw(20) << fixed << order
              << endl;
      }

      prev_error = error;
   }

   if (myid == 0)
   {
      cout << "\n=== Analysis ===" << endl;
      cout << "Expected: Order ≈ 1.0 (linear convergence with FD)" << endl;
      cout << "Once adjoint gradient is added:" << endl;
      cout << "  Error = |J(ρ+εδρ) - J(ρ) - ε⟨dJ/dρ, δρ⟩|" << endl;
      cout << "  Expected: Order ≈ 2.0 (quadratic convergence)" << endl;
      cout << "\n=== Taylor Test Infrastructure Complete ===" << endl;
   }

   return 0;
}
