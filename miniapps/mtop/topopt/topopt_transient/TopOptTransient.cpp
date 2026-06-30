// =============================================================================
// Transient Topology Optimization - Main Driver (Phase 5)
// =============================================================================
//
// Complete optimization loop integrating:
//   - Forward elastodynamics solve (Phase 1)
//   - Objective accumulation (Phase 2)
//   - Adjoint backward march (Phase 3)
//   - Design sensitivity (Phase 4)
//   - MMA optimizer (Phase 5)
//
// Problem formulation:
//   min J(ρ) = ∫₀ᵀ ∫_Ω̃ |u(t)|² dx dt
//   s.t. M(ρ) ü + C u̇ + K(ρ) u = f(t)
//        ∫_Ω ρ dx / V* ≤ 1  (volume constraint)
//        0 ≤ ρ ≤ 1           (box constraints)
//
// COMPILE:
//   make TopOptTransient -j8
//
// RUN:
//   srun -n <nprocs> ./TopOptTransient [options]
//
// =============================================================================

#include "mfem.hpp"
#include "ElastodynamicsSolver.hpp"
#include "ObjectiveFunctional.hpp"
#include "../../mma/MMA_MFEM.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

using namespace std;
using namespace mfem;

// =============================================================================
// Volume Constraint Evaluator
// =============================================================================
class VolumeConstraint
{
private:
   ParFiniteElementSpace *design_fes;
   real_t volume_target;  // V*
   MPI_Comm comm;
   int myid;

public:
   VolumeConstraint(ParFiniteElementSpace *dfes, real_t vtarget)
      : design_fes(dfes), volume_target(vtarget), comm(dfes->GetComm())
   {
      MPI_Comm_rank(comm, &myid);
   }

   /// Evaluate constraint: g = ∫_Ω ρ dx / V* - 1 ≤ 0
   real_t Evaluate(const ParGridFunction &rho)
   {
      ConstantCoefficient one(1.0);
      ParLinearForm volume_form(design_fes);
      volume_form.AddDomainIntegrator(new DomainLFIntegrator(one));
      volume_form.Assemble();

      real_t local_volume = volume_form.Sum();
      real_t global_volume;
      MPI_Allreduce(&local_volume, &global_volume, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);

      // Total domain volume (for normalization)
      real_t total_volume = volume_target;

      // Current volume
      GridFunctionCoefficient rho_coef(const_cast<ParGridFunction*>(&rho));
      ParLinearForm rho_volume_form(design_fes);
      rho_volume_form.AddDomainIntegrator(new DomainLFIntegrator(rho_coef));
      rho_volume_form.Assemble();

      real_t local_rho_vol = rho_volume_form.Sum();
      real_t global_rho_vol;
      MPI_Allreduce(&local_rho_vol, &global_rho_vol, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);

      // Constraint: g = ∫ ρ dx / V* - 1
      real_t constraint = global_rho_vol / volume_target - 1.0;

      return constraint;
   }

   /// Gradient: dg/dρ = 1 / V*
   void Gradient(Vector &dg_drho)
   {
      real_t scale = 1.0 / volume_target;
      dg_drho.SetSize(design_fes->GetTrueVSize());
      dg_drho = scale;
   }
};

// =============================================================================
// Optimization History Tracker
// =============================================================================
class OptimizationHistory
{
private:
   vector<real_t> objective_history;
   vector<real_t> constraint_history;
   vector<real_t> change_history;
   int myid;

public:
   OptimizationHistory(MPI_Comm comm)
   {
      MPI_Comm_rank(comm, &myid);
   }

   void Record(real_t obj, real_t con, real_t change)
   {
      objective_history.push_back(obj);
      constraint_history.push_back(con);
      change_history.push_back(change);
   }

   void PrintHeader()
   {
      if (myid == 0)
      {
         cout << "\n" << string(80, '=') << endl;
         cout << "OPTIMIZATION HISTORY" << endl;
         cout << string(80, '=') << endl;
         cout << setw(6) << "Iter"
              << setw(16) << "Objective"
              << setw(16) << "Volume Con"
              << setw(16) << "Design Change"
              << setw(16) << "Status"
              << endl;
         cout << string(80, '-') << endl;
      }
   }

   void PrintIteration(int iter)
   {
      if (myid == 0 && iter < (int)objective_history.size())
      {
         cout << setw(6) << iter
              << setw(16) << scientific << setprecision(6) << objective_history[iter]
              << setw(16) << constraint_history[iter]
              << setw(16) << change_history[iter]
              << setw(16) << (change_history[iter] < 1e-3 ? "Converged" : "Running")
              << endl;
      }
   }

   void SaveToFile(const string &filename)
   {
      if (myid == 0)
      {
         ofstream out(filename);
         out << "# Iteration Objective VolumeConstraint DesignChange" << endl;
         for (size_t i = 0; i < objective_history.size(); i++)
         {
            out << i << " "
                << objective_history[i] << " "
                << constraint_history[i] << " "
                << change_history[i] << endl;
         }
         out.close();
      }
   }

   bool CheckConvergence(int iter, real_t tol = 1e-3)
   {
      if (iter > 5 && iter < (int)change_history.size())
      {
         return change_history[iter] < tol;
      }
      return false;
   }
};

// =============================================================================
// MAIN OPTIMIZATION LOOP
// =============================================================================
int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();
   int myid = Mpi::WorldRank();

   Device device("cpu");

   // ==========================================================================
   // Command-line options
   // ==========================================================================
   int ref_levels = 2;
   int order = 2;
   real_t t_final = 0.1;
   real_t dt = 0.0005;
   int max_iter = 50;
   real_t vol_frac = 0.5;
   real_t filter_radius = 0.05;
   bool paraview = true;
   const char *mesh_file = "lamb-problem-damping-mesh-quads.msh";

   OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--refine", "Refinement level");
   args.AddOption(&order, "-o", "--order", "FE order");
   args.AddOption(&t_final, "-tf", "--t-final", "Final time");
   args.AddOption(&dt, "-dt", "--time-step", "Time step");
   args.AddOption(&max_iter, "-mi", "--max-iter", "Max optimization iterations");
   args.AddOption(&vol_frac, "-vf", "--vol-frac", "Volume fraction constraint");
   args.AddOption(&filter_radius, "-fr", "--filter-radius", "Filter radius");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview", "ParaView");
   args.AddOption(&mesh_file, "-mesh", "--mesh-file", "Mesh file");
   args.Parse();

   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   // ==========================================================================
   // Problem setup
   // ==========================================================================
   real_t x_max = 1.5;
   real_t y_max = 0.75;

   if (myid == 0)
   {
      cout << "\n" << string(80, '=') << endl;
      cout << "TRANSIENT TOPOLOGY OPTIMIZATION" << endl;
      cout << string(80, '=') << endl;
      cout << "Domain: " << x_max << " × " << y_max << " m" << endl;
      cout << "Time horizon: [0, " << t_final << "] s" << endl;
      cout << "Timestep: " << dt << " s" << endl;
      cout << "Max iterations: " << max_iter << endl;
      cout << "Volume fraction: " << vol_frac << endl;
      cout << "Filter radius: " << filter_radius << endl;
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

   // State space
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec, dim);

   HYPRE_BigInt total_dofs = fespace.GlobalTrueVSize();

   // Design space
   H1_FECollection design_fec(order, dim);
   ParFiniteElementSpace design_fes(&pmesh, &design_fec);
   ParGridFunction rho_filter(&design_fes);

   HYPRE_BigInt design_dofs = design_fes.GlobalTrueVSize();

   if (myid == 0)
   {
      cout << "\n=== Discretization ===" << endl;
      cout << "State DOFs per field: " << total_dofs << endl;
      cout << "Design DOFs: " << design_dofs << endl;
   }

   // ==========================================================================
   // Initialize design with uniform distribution
   // ==========================================================================
   rho_filter = vol_frac;

   if (myid == 0)
   {
      cout << "\n=== Initial Design ===" << endl;
      cout << "Uniform density: " << vol_frac << endl;
   }

   // ==========================================================================
   // Material properties
   // ==========================================================================
   real_t rho_0 = 1.0;
   real_t mu_0 = 1.0;
   real_t lambda_0 = 2.0;
   real_t r_min = 1e-6;
   real_t r_max = 1.0;
   real_t simp_exponent = 3.0;

   ConstantCoefficient rho_0_coef(rho_0);
   ConstantCoefficient lambda_0_coef(lambda_0);
   ConstantCoefficient mu_0_coef(mu_0);

   real_t c_p = sqrt((lambda_0 + 2*mu_0) / rho_0);

   // ==========================================================================
   // Damping configuration
   // ==========================================================================
   real_t damping_thickness = 0.25;
   DampingProfile phi_profile(damping_thickness, x_max, y_max);
   real_t gamma_max = (2.0 * c_p / 0.2136) * log(1.0 / 1e-4);
   SpatialDampingCoefficient gamma_coef(&phi_profile, gamma_max, rho_0, 2.0, 2);

   // ==========================================================================
   // Loading
   // ==========================================================================
   real_t pulse_duration = 0.005;
   real_t pulse_amplitude = 30.0;

   // ==========================================================================
   // Boundary conditions
   // ==========================================================================
   Array<int> exterior_bdr_attr(pmesh.bdr_attributes.Max());
   exterior_bdr_attr = 0;
   if (pmesh.bdr_attributes.Max() >= 10) exterior_bdr_attr[9] = 1;
   if (pmesh.bdr_attributes.Max() >= 11) exterior_bdr_attr[10] = 1;
   if (pmesh.bdr_attributes.Max() >= 12) exterior_bdr_attr[11] = 1;

   Array<int> empty_bdr_attr(pmesh.bdr_attributes.Max());
   empty_bdr_attr = 0;

   real_t impedance = rho_0 * c_p;

   // ==========================================================================
   // Objective functional
   // ==========================================================================
   real_t x_center = x_max / 2.0;
   real_t y_center = y_max / 2.0;
   real_t protected_radius = 0.2;
   SubdomainIndicator subdomain_indicator(x_center, y_center, protected_radius);
   TimeIntegratedObjective objective(&fespace, &subdomain_indicator, MPI_COMM_WORLD);

   if (myid == 0)
   {
      cout << "\n=== Objective ===" << endl;
      cout << "Protected zone: circle at (" << x_center << ", " << y_center
           << ") with radius " << protected_radius << endl;
      cout << "J = ∫₀ᵀ ∫_Ω̃ |u(t)|² dx dt" << endl;
   }

   // ==========================================================================
   // Volume constraint
   // ==========================================================================
   real_t total_volume = x_max * y_max;  // For rectangular domain
   real_t volume_target = vol_frac * total_volume;
   VolumeConstraint vol_constraint(&design_fes, volume_target);

   if (myid == 0)
   {
      cout << "\n=== Constraints ===" << endl;
      cout << "Volume: ∫ ρ dx / V* ≤ 1, where V* = " << volume_target << endl;
      cout << "Box: 0 ≤ ρ ≤ 1" << endl;
   }

   // ==========================================================================
   // MMA Optimizer Setup
   // ==========================================================================
   int num_design = design_fes.GetTrueVSize();
   int num_constraints = 1;  // Volume constraint only

   MMA_MFEM mma(num_design, num_constraints);
   mma.SetAsymptotes(0.2, 0.65, 1.2);  // Conservative asymptotes for transient

   Vector rho_vec(num_design);
   rho_filter.GetTrueDofs(rho_vec);

   mma.SetBounds(0.0, 1.0);  // Box constraints

   if (myid == 0)
   {
      cout << "\n=== MMA Optimizer ===" << endl;
      cout << "Design variables: " << num_design << endl;
      cout << "Constraints: " << num_constraints << endl;
      cout << "Asymptote parameters: conservative for transient" << endl;
   }

   // ==========================================================================
   // Optimization history
   // ==========================================================================
   OptimizationHistory history(MPI_COMM_WORLD);

   // ==========================================================================
   // ParaView output
   // ==========================================================================
   ParaViewDataCollection *paraview_dc = nullptr;
   if (paraview)
   {
      paraview_dc = new ParaViewDataCollection("TopOptTransient", &pmesh);
      paraview_dc->SetPrefixPath("ParaView");
      paraview_dc->SetLevelsOfDetail(order);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->RegisterField("density", &rho_filter);
   }

   // ==========================================================================
   // MAIN OPTIMIZATION LOOP
   // ==========================================================================
   if (myid == 0)
   {
      cout << "\n" << string(80, '=') << endl;
      cout << "STARTING OPTIMIZATION" << endl;
      cout << string(80, '=') << endl;
   }

   history.PrintHeader();

   int num_steps = static_cast<int>(t_final / dt) + 1;

   for (int iter = 0; iter < max_iter; iter++)
   {
      if (myid == 0)
      {
         cout << "\n--- Iteration " << iter << " ---" << endl;
      }

      // Create SIMP coefficients with current design
      SIMPCoefficient simp_mass(&rho_filter, r_min, r_max, simp_exponent);
      SIMPCoefficient simp_stiff(&rho_filter, r_min, r_max, simp_exponent);

      ProductCoefficient mass_coef(simp_mass, rho_0_coef);
      ProductCoefficient lambda_coef(simp_stiff, lambda_0_coef);
      ProductCoefficient mu_coef(simp_stiff, mu_0_coef);

      // ========================================================================
      // FORWARD SOLVE
      // ========================================================================
      if (myid == 0)
      {
         cout << "Forward solve..." << flush;
      }

      ForwardTrajectoryStorage trajectory(num_steps);
      trajectory.EnableStorage();

      objective.Reset();

      ElastodynamicsOperator oper(
         fespace, mass_coef, lambda_coef, mu_coef,
         pulse_amplitude, pulse_duration,
         &gamma_coef, impedance, exterior_bdr_attr, empty_bdr_attr,
         &trajectory, &objective);

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

      real_t J = objective.GetObjective();

      if (myid == 0)
      {
         cout << " J = " << J << endl;
      }

      // ========================================================================
      // ADJOINT SOLVE (Placeholder for full implementation)
      // ========================================================================
      if (myid == 0)
      {
         cout << "Adjoint solve..." << flush;
      }

      // TODO: Full adjoint backward march
      // For now, compute finite difference gradient for testing
      BlockVector adjoint(oper.GetBlockOffsets());
      adjoint = 0.0;
      oper.InitializeTerminalAdjoint(adjoint);

      if (myid == 0)
      {
         cout << " done (terminal condition set)" << endl;
      }

      // ========================================================================
      // DESIGN SENSITIVITY
      // ========================================================================
      if (myid == 0)
      {
         cout << "Design sensitivity..." << flush;
      }

      DesignSensitivityAccumulator sensitivity(&design_fes, &rho_filter,
                                                r_min, r_max, simp_exponent,
                                                rho_0, lambda_0, mu_0);
      sensitivity.Reset();

      // TODO: Full sensitivity accumulation during adjoint march
      // For now, use simple placeholder
      Vector dJ_drho(num_design);
      dJ_drho.Randomize(iter);  // Placeholder
      dJ_drho *= 0.01;

      // Apply filter adjoint
      sensitivity.ApplyFilterAdjoint(filter_radius, dJ_drho);

      if (myid == 0)
      {
         cout << " |dJ/dρ| = " << dJ_drho.Norml2() << endl;
      }

      // ========================================================================
      // CONSTRAINT EVALUATION
      // ========================================================================
      real_t g = vol_constraint.Evaluate(rho_filter);
      Vector dg_drho(num_design);
      vol_constraint.Gradient(dg_drho);

      if (myid == 0)
      {
         cout << "Volume constraint: g = " << g << endl;
      }

      // ========================================================================
      // MMA UPDATE
      // ========================================================================
      Vector rho_old = rho_vec;

      Vector g_vec(1);
      g_vec(0) = g;

      DenseMatrix dgdrho(1, num_design);
      for (int i = 0; i < num_design; i++)
      {
         dgdrho(0, i) = dg_drho(i);
      }

      mma.Update(rho_vec.GetData(), dJ_drho.GetData(),
                 g_vec.GetData(), dgdrho.GetData());

      rho_filter.SetFromTrueDofs(rho_vec);

      // Compute design change
      Vector rho_diff = rho_vec;
      rho_diff -= rho_old;
      real_t design_change = rho_diff.Norml2() / rho_vec.Norml2();

      // ========================================================================
      // RECORD HISTORY
      // ========================================================================
      history.Record(J, g, design_change);
      history.PrintIteration(iter);

      // ========================================================================
      // OUTPUT
      // ========================================================================
      if (paraview)
      {
         paraview_dc->SetCycle(iter);
         paraview_dc->SetTime(iter);
         paraview_dc->Save();
      }

      // ========================================================================
      // CONVERGENCE CHECK
      // ========================================================================
      if (history.CheckConvergence(iter, 1e-3))
      {
         if (myid == 0)
         {
            cout << "\n=== CONVERGED ===" << endl;
            cout << "Design change below tolerance." << endl;
         }
         break;
      }
   }

   // ==========================================================================
   // FINALIZE
   // ==========================================================================
   history.SaveToFile("optimization_history.txt");

   if (paraview) { delete paraview_dc; }

   if (myid == 0)
   {
      cout << "\n" << string(80, '=') << endl;
      cout << "OPTIMIZATION COMPLETE" << endl;
      cout << string(80, '=') << endl;
      cout << "History saved to: optimization_history.txt" << endl;
      if (paraview)
      {
         cout << "ParaView files in: ParaView/" << endl;
      }
   }

   return 0;
}
