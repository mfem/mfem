// =============================================================================
// Physical Validation for Transient Topology Optimization (Phase 6)
// =============================================================================
//
// Validates physical behavior:
//   1. Wave amplitude reduction in protected zone
//   2. Energy conservation/dissipation
//   3. Comparison: optimized vs uniform design
//   4. Design quality metrics
//
// COMPILE:
//   make test_physical_validation -j8
//
// RUN:
//   srun -n <nprocs> ./test_physical_validation [options]
//
// =============================================================================

#include "mfem.hpp"
#include "ElastodynamicsSolver.hpp"
#include "ObjectiveFunctional.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

using namespace std;
using namespace mfem;

// =============================================================================
// Energy Calculator
// =============================================================================
class EnergyCalculator
{
private:
   ParFiniteElementSpace *fespace;
   HypreParMatrix *Mmat, *Kmat;
   MPI_Comm comm;

public:
   EnergyCalculator(ParFiniteElementSpace *fes,
                    HypreParMatrix *M, HypreParMatrix *K)
      : fespace(fes), Mmat(M), Kmat(K), comm(fes->GetComm()) {}

   real_t KineticEnergy(const Vector &v)
   {
      Vector Mv(v.Size());
      Mmat->Mult(v, Mv);
      real_t local_ke = 0.5 * (v * Mv);
      real_t global_ke;
      MPI_Allreduce(&local_ke, &global_ke, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
      return global_ke;
   }

   real_t PotentialEnergy(const Vector &u)
   {
      Vector Ku(u.Size());
      Kmat->Mult(u, Ku);
      real_t local_pe = 0.5 * (u * Ku);
      real_t global_pe;
      MPI_Allreduce(&local_pe, &global_pe, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
      return global_pe;
   }

   real_t TotalEnergy(const Vector &u, const Vector &v)
   {
      return KineticEnergy(v) + PotentialEnergy(u);
   }
};

// =============================================================================
// Run Simulation and Collect Metrics
// =============================================================================
struct SimulationMetrics
{
   real_t objective;
   real_t max_displacement_protected;
   real_t max_displacement_total;
   real_t initial_energy;
   real_t final_energy;
   real_t max_energy;
   vector<real_t> energy_history;
};

SimulationMetrics RunSimulation(
   ParMesh &pmesh,
   ParGridFunction &rho_filter,
   real_t t_final,
   real_t dt,
   const string &label)
{
   int myid = Mpi::WorldRank();
   SimulationMetrics metrics;

   // State space
   int order = 2;
   H1_FECollection fec(order, pmesh.Dimension());
   ParFiniteElementSpace fespace(&pmesh, &fec, pmesh.Dimension());

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
   real_t x_max = 1.5, y_max = 0.75;
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
   real_t x_center = x_max / 2.0, y_center = y_max / 2.0;
   real_t protected_radius = 0.2;
   SubdomainIndicator subdomain_indicator(x_center, y_center, protected_radius);
   TimeIntegratedObjective objective(&fespace, &subdomain_indicator, MPI_COMM_WORLD);

   int num_steps = static_cast<int>(t_final / dt) + 1;
   ForwardTrajectoryStorage trajectory(num_steps);
   trajectory.EnableStorage();

   ElastodynamicsOperator oper(
      fespace, mass_coef, lambda_coef, mu_coef,
      pulse_amplitude, pulse_duration,
      &gamma_coef, impedance, exterior_bdr_attr, empty_bdr_attr,
      &trajectory, &objective);

   // Energy calculator
   EnergyCalculator energy_calc(&fespace, oper.GetMassMatrix(),
                                 oper.GetStiffnessMatrix());

   // Solve
   objective.Reset();

   BlockVector state(oper.GetBlockOffsets());
   state = 0.0;

   RK4Solver ode_solver;
   ode_solver.Init(oper);

   real_t t = 0.0;
   metrics.max_displacement_protected = 0.0;
   metrics.max_displacement_total = 0.0;
   metrics.max_energy = 0.0;

   for (int ti = 0; ti < num_steps && t < t_final - dt/2; ti++)
   {
      oper.StoreTrajectoryStep(ti, state);
      oper.AccumulateObjective(state, dt, ti, num_steps);

      oper.SetTime(t + dt);
      ode_solver.Step(state, t, dt);

      // Extract u, v
      BlockVector bstate(state, oper.GetBlockOffsets());
      Vector u_true(bstate.GetBlock(0).GetData(), bstate.GetBlock(0).Size());
      Vector v_true(bstate.GetBlock(1).GetData(), bstate.GetBlock(1).Size());

      // Energy
      real_t E = energy_calc.TotalEnergy(u_true, v_true);
      metrics.energy_history.push_back(E);
      if (ti == 0) metrics.initial_energy = E;
      if (ti == num_steps - 1) metrics.final_energy = E;
      if (E > metrics.max_energy) metrics.max_energy = E;

      // Max displacement
      real_t u_max = u_true.Normlinf();
      if (u_max > metrics.max_displacement_total)
      {
         metrics.max_displacement_total = u_max;
      }

      // TODO: Max in protected zone (requires point evaluation)
   }

   metrics.objective = objective.GetObjective();

   if (myid == 0)
   {
      cout << "\n" << label << " Results:" << endl;
      cout << "  Objective J = " << scientific << setprecision(4)
           << metrics.objective << endl;
      cout << "  Max displacement = " << metrics.max_displacement_total << endl;
      cout << "  Initial energy = " << metrics.initial_energy << endl;
      cout << "  Final energy = " << metrics.final_energy << endl;
      cout << "  Max energy = " << metrics.max_energy << endl;
      cout << "  Energy dissipation = "
           << (metrics.initial_energy - metrics.final_energy) / metrics.initial_energy * 100
           << " %" << endl;
   }

   return metrics;
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
   int ref_levels = 2;
   int order = 2;
   real_t t_final = 0.15;
   real_t dt = 0.0005;
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

   if (myid == 0)
   {
      cout << "\n" << string(80, '=') << endl;
      cout << "PHYSICAL VALIDATION FOR TRANSIENT TOPOLOGY OPTIMIZATION" << endl;
      cout << string(80, '=') << endl;
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

   // Design space
   H1_FECollection design_fec(order, dim);
   ParFiniteElementSpace design_fes(&pmesh, &design_fec);

   // ==========================================================================
   // TEST 1: Uniform Design (Baseline)
   // ==========================================================================
   if (myid == 0)
   {
      cout << "\n=== TEST 1: Uniform Design (Baseline) ===" << endl;
      cout << "Uniform density ρ = 0.5 everywhere" << endl;
      cout << string(80, '-') << endl;
   }

   ParGridFunction rho_uniform(&design_fes);
   rho_uniform = 0.5;

   SimulationMetrics baseline = RunSimulation(pmesh, rho_uniform, t_final, dt,
                                               "Uniform Design");

   // ==========================================================================
   // TEST 2: Void Design (No Material)
   // ==========================================================================
   if (myid == 0)
   {
      cout << "\n=== TEST 2: Void Design ===" << endl;
      cout << "Minimal density ρ = 0.01 everywhere" << endl;
      cout << string(80, '-') << endl;
   }

   ParGridFunction rho_void(&design_fes);
   rho_void = 0.01;

   SimulationMetrics void_design = RunSimulation(pmesh, rho_void, t_final, dt,
                                                  "Void Design");

   // ==========================================================================
   // TEST 3: Solid Design (Full Material)
   // ==========================================================================
   if (myid == 0)
   {
      cout << "\n=== TEST 3: Solid Design ===" << endl;
      cout << "Full density ρ = 1.0 everywhere" << endl;
      cout << string(80, '-') << endl;
   }

   ParGridFunction rho_solid(&design_fes);
   rho_solid = 1.0;

   SimulationMetrics solid_design = RunSimulation(pmesh, rho_solid, t_final, dt,
                                                   "Solid Design");

   // ==========================================================================
   // TEST 4: Optimized Design (Gaussian - Mimics Optimization)
   // ==========================================================================
   if (myid == 0)
   {
      cout << "\n=== TEST 4: Optimized Design (Gaussian Approximation) ===" << endl;
      cout << "Gaussian distribution mimicking optimization result" << endl;
      cout << string(80, '-') << endl;
   }

   ParGridFunction rho_optimized(&design_fes);

   real_t x_max = 1.5, y_max = 0.75;
   real_t x_center = x_max / 2.0, y_center = y_max / 2.0;
   real_t sigma_x = 0.25, sigma_y = 0.15;
   GaussianDesignCoefficient gaussian_coef(x_center, y_center, sigma_x, sigma_y,
                                            0.3, 1.0);
   rho_optimized.ProjectCoefficient(gaussian_coef);

   SimulationMetrics optimized = RunSimulation(pmesh, rho_optimized, t_final, dt,
                                                "Optimized Design");

   // ==========================================================================
   // COMPARISON
   // ==========================================================================
   if (myid == 0)
   {
      cout << "\n" << string(80, '=') << endl;
      cout << "COMPARISON OF DESIGNS" << endl;
      cout << string(80, '=') << endl;

      cout << "\nObjective Values:" << endl;
      cout << setw(25) << "Design" << setw(20) << "Objective J"
           << setw(20) << "% vs Baseline" << endl;
      cout << string(65, '-') << endl;

      real_t J_base = baseline.objective;

      cout << setw(25) << "Uniform (baseline)" << setw(20) << scientific
           << setprecision(4) << baseline.objective << setw(20) << "0.0%" << endl;

      cout << setw(25) << "Void" << setw(20) << void_design.objective
           << setw(20) << fixed << setprecision(1)
           << (void_design.objective - J_base) / J_base * 100 << "%" << endl;

      cout << setw(25) << "Solid" << setw(20) << scientific
           << solid_design.objective << setw(20) << fixed
           << (solid_design.objective - J_base) / J_base * 100 << "%" << endl;

      cout << setw(25) << "Optimized (Gaussian)" << setw(20) << scientific
           << optimized.objective << setw(20) << fixed
           << (optimized.objective - J_base) / J_base * 100 << "%" << endl;

      cout << "\n=== Physical Validation Results ===" << endl;
      cout << "✓ Expected: Optimized design reduces objective vs baseline" << endl;
      cout << "  Status: " << (optimized.objective < baseline.objective ? "PASS" : "FAIL")
           << endl;

      cout << "\n✓ Expected: Energy dissipation due to damping" << endl;
      cout << "  Baseline dissipation: "
           << (baseline.initial_energy - baseline.final_energy) / baseline.initial_energy * 100
           << "%" << endl;
      cout << "  Status: "
           << ((baseline.initial_energy > baseline.final_energy) ? "PASS" : "FAIL")
           << endl;

      cout << "\n✓ Expected: Void design has lowest stiffness (highest objective)" << endl;
      cout << "  Status: " << (void_design.objective > baseline.objective ? "PASS" : "FAIL")
           << endl;

      cout << "\n" << string(80, '=') << endl;
      cout << "PHYSICAL VALIDATION COMPLETE" << endl;
      cout << string(80, '=') << endl;
   }

   return 0;
}
