// =============================================================================
// Convergence Study for Transient Topology Optimization (Phase 6)
// =============================================================================
//
// Tests convergence with respect to:
//   1. Mesh refinement (h-convergence)
//   2. Time step refinement (temporal convergence)
//   3. Combined refinement
//
// Computes:
//   - Objective J(ρ) for fixed design
//   - Convergence rates
//   - Error estimates
//
// COMPILE:
//   make test_convergence_study -j8
//
// RUN:
//   srun -n <nprocs> ./test_convergence_study [options]
//
// =============================================================================

#include "mfem.hpp"
#include "ElastodynamicsSolver.hpp"
#include "ObjectiveFunctional.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace std;
using namespace mfem;

// =============================================================================
// Compute Objective for Given Discretization
// =============================================================================
real_t ComputeObjectiveForDiscretization(
   ParMesh &pmesh,
   int order,
   real_t t_final,
   real_t dt,
   ParGridFunction &rho_filter,
   real_t &h_min)
{
   int myid = Mpi::WorldRank();

   // State space
   H1_FECollection fec(order, pmesh.Dimension());
   ParFiniteElementSpace fespace(&pmesh, &fec, pmesh.Dimension());

   // Compute mesh size
   h_min = pmesh.GetElementSize(0, 1);  // Min diameter
   for (int i = 1; i < pmesh.GetNE(); i++)
   {
      real_t h = pmesh.GetElementSize(i, 1);
      if (h < h_min) h_min = h;
   }

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

   // Damping
   real_t x_max = 1.5;
   real_t y_max = 0.75;
   real_t damping_thickness = 0.25;
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
   real_t x_center = x_max / 2.0;
   real_t y_center = y_max / 2.0;
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

   // Solve
   objective.Reset();

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

   return objective.GetObjective();
}

// =============================================================================
// MAIN
// =============================================================================
int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();
   int myid = Mpi::WorldRank();

   Device device("cpu");

   // Command-line options
   const char *mesh_file = "lamb-problem-damping-mesh-quads.msh";
   real_t t_final = 0.1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-mesh", "--mesh-file", "Mesh file");
   args.AddOption(&t_final, "-tf", "--t-final", "Final time");
   args.Parse();

   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }

   if (myid == 0)
   {
      cout << "\n" << string(80, '=') << endl;
      cout << "CONVERGENCE STUDY FOR TRANSIENT TOPOLOGY OPTIMIZATION" << endl;
      cout << string(80, '=') << endl;
   }

   // Load base mesh
   ifstream imesh(mesh_file);
   if (!imesh)
   {
      if (myid == 0)
      {
         cerr << "Error: Cannot open mesh file '" << mesh_file << "'" << endl;
      }
      return 1;
   }

   Mesh base_mesh(imesh, 1, 1);
   imesh.close();
   int dim = base_mesh.Dimension();

   // ==========================================================================
   // TEST 1: Spatial (h) Convergence
   // ==========================================================================
   if (myid == 0)
   {
      cout << "\n=== TEST 1: Spatial (h-) Convergence ===" << endl;
      cout << "Fixed dt, varying mesh refinement" << endl;
      cout << string(80, '-') << endl;
   }

   vector<int> ref_levels = {1, 2, 3, 4};
   int order = 2;
   real_t dt_fixed = 0.0005;

   vector<real_t> h_values;
   vector<real_t> J_h_values;

   for (int r : ref_levels)
   {
      Mesh mesh(base_mesh);
      for (int l = 0; l < r; l++)
      {
         mesh.UniformRefinement();
      }

      ParMesh pmesh(MPI_COMM_WORLD, mesh);
      mesh.Clear();

      // Design space
      H1_FECollection design_fec(order, dim);
      ParFiniteElementSpace design_fes(&pmesh, &design_fec);
      ParGridFunction rho_filter(&design_fes);

      // Use Gaussian design for testing
      real_t x_max = 1.5, y_max = 0.75;
      real_t x_center = x_max / 2.0, y_center = y_max / 2.0;
      real_t sigma_x = 0.3, sigma_y = 0.15;
      GaussianDesignCoefficient gaussian_coef(x_center, y_center, sigma_x, sigma_y,
                                               0.3, 1.0);
      rho_filter.ProjectCoefficient(gaussian_coef);

      real_t h_min;
      real_t J = ComputeObjectiveForDiscretization(pmesh, order, t_final, dt_fixed,
                                                    rho_filter, h_min);

      h_values.push_back(h_min);
      J_h_values.push_back(J);

      if (myid == 0)
      {
         cout << "Refinement " << r << ": h = " << scientific << setprecision(4) << h_min
              << ", J = " << J << endl;
      }
   }

   // Compute convergence rates
   if (myid == 0 && J_h_values.size() >= 2)
   {
      cout << "\nConvergence rates (J_i / J_{i+1}):" << endl;
      for (size_t i = 0; i < J_h_values.size() - 1; i++)
      {
         real_t rate = log(fabs(J_h_values[i+1] - J_h_values[i]) /
                           fabs(J_h_values[i] - J_h_values[i>0?i-1:i])) /
                       log(h_values[i+1] / h_values[i]);
         cout << "  Level " << i << "->" << i+1 << ": rate ≈ " << fixed
              << setprecision(2) << rate << endl;
      }
   }

   // ==========================================================================
   // TEST 2: Temporal Convergence
   // ==========================================================================
   if (myid == 0)
   {
      cout << "\n=== TEST 2: Temporal Convergence ===" << endl;
      cout << "Fixed mesh, varying timestep" << endl;
      cout << string(80, '-') << endl;
   }

   int ref_fixed = 2;
   vector<real_t> dt_values = {0.002, 0.001, 0.0005, 0.00025};

   vector<real_t> J_dt_values;

   for (real_t dt : dt_values)
   {
      Mesh mesh(base_mesh);
      for (int l = 0; l < ref_fixed; l++)
      {
         mesh.UniformRefinement();
      }

      ParMesh pmesh(MPI_COMM_WORLD, mesh);
      mesh.Clear();

      H1_FECollection design_fec(order, dim);
      ParFiniteElementSpace design_fes(&pmesh, &design_fec);
      ParGridFunction rho_filter(&design_fes);

      real_t x_max = 1.5, y_max = 0.75;
      real_t x_center = x_max / 2.0, y_center = y_max / 2.0;
      real_t sigma_x = 0.3, sigma_y = 0.15;
      GaussianDesignCoefficient gaussian_coef(x_center, y_center, sigma_x, sigma_y,
                                               0.3, 1.0);
      rho_filter.ProjectCoefficient(gaussian_coef);

      real_t h_min;
      real_t J = ComputeObjectiveForDiscretization(pmesh, order, t_final, dt,
                                                    rho_filter, h_min);

      J_dt_values.push_back(J);

      if (myid == 0)
      {
         cout << "dt = " << scientific << setprecision(4) << dt
              << ", J = " << J << endl;
      }
   }

   // Compute convergence rates
   if (myid == 0 && J_dt_values.size() >= 2)
   {
      cout << "\nConvergence rates:" << endl;
      for (size_t i = 0; i < J_dt_values.size() - 1; i++)
      {
         real_t rate = log(fabs(J_dt_values[i+1] - J_dt_values[i]) /
                           fabs(J_dt_values[i] - (i>0?J_dt_values[i-1]:J_dt_values[i]))) /
                       log(dt_values[i+1] / dt_values[i]);
         cout << "  Step " << i << "->" << i+1 << ": rate ≈ " << fixed
              << setprecision(2) << rate << " (RK4 is O(dt^4))" << endl;
      }
   }

   // ==========================================================================
   // TEST 3: Richardson Extrapolation
   // ==========================================================================
   if (myid == 0)
   {
      cout << "\n=== TEST 3: Richardson Extrapolation ===" << endl;
      cout << "Estimate converged value" << endl;
      cout << string(80, '-') << endl;
   }

   if (myid == 0 && J_h_values.size() >= 3)
   {
      // Use last 3 values for Richardson extrapolation
      real_t J1 = J_h_values[J_h_values.size()-3];
      real_t J2 = J_h_values[J_h_values.size()-2];
      real_t J3 = J_h_values[J_h_values.size()-1];
      real_t h1 = h_values[h_values.size()-3];
      real_t h2 = h_values[h_values.size()-2];
      real_t h3 = h_values[h_values.size()-1];

      // Assuming convergence rate p
      real_t p = log((J2-J1)/(J3-J2)) / log((h2-h1)/(h3-h2));
      if (p > 0 && p < 10)  // Sanity check
      {
         real_t J_extrap = J3 + (J3-J2)/(pow(h2/h3, p) - 1.0);
         cout << "Estimated convergence order: p ≈ " << p << endl;
         cout << "Richardson extrapolation: J_∞ ≈ " << scientific << J_extrap << endl;
         cout << "Error estimate: |J_finest - J_∞| ≈ " << fabs(J3 - J_extrap) << endl;
      }
   }

   // ==========================================================================
   // SUMMARY
   // ==========================================================================
   if (myid == 0)
   {
      cout << "\n" << string(80, '=') << endl;
      cout << "CONVERGENCE STUDY COMPLETE" << endl;
      cout << string(80, '=') << endl;
      cout << "\nExpected behavior:" << endl;
      cout << "  - Spatial: O(h^p) where p = order + 1 for smooth solutions" << endl;
      cout << "  - Temporal: O(dt^4) for RK4" << endl;
      cout << "\nResults saved to console output." << endl;
      cout << "Redirect to file: ./program > convergence_results.txt" << endl;
   }

   return 0;
}
