// =============================================================================
// Transient Topology Optimization - Main Driver (SKETCH)
// =============================================================================
//
// Minimize wave amplitude in protected subdomain:
//   min J(ρ) = ∫₀ᵀ ∫_Ω̃ |u(t)|² dx dt
//   s.t. M(ρ) ü + C u̇ + K(ρ) u = f(t)
//        ∫_Ω ρ dx / V* ≤ 1
//        0 ≤ ρ ≤ 1
//
// Based on:
//   - Paper: topopt.tex Section 5 (Transient elasticity)
//   - Forward: ForwardElastodynamics.cpp
//   - Static: ElastTopOpt_transient.cpp
//
// =============================================================================

#include "mfem.hpp"
#include "ObjectiveFunctional.hpp"
#include "AdjointSolver.hpp"
#include "../../mma/MMA_MFEM.hpp"

using namespace std;
using namespace mfem;

// Forward elastodynamics operator (same as ForwardElastodynamics.cpp)
class ElastodynamicsOperator : public TimeDependentOperator
{
   // ... (copy from ForwardElastodynamics.cpp)
   // Key: Must store K(ρ, u^n) at each timestep for adjoint!
};

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();
   int myid = Mpi::WorldRank();

   // ========================================================================
   // PARAMETERS
   // ========================================================================
   int ref_levels = 1;
   int order = 4;
   real_t t_final = 0.5;
   real_t dt = 0.0001;
   real_t filter_r = 0.05;
   real_t vol_fraction = 0.5;
   int max_opt_iter = 100;

   // Protected subdomain Ω̃ (circle in center)
   real_t x_max = 1.5, y_max = 0.75;
   real_t protect_x = x_max / 2.0;
   real_t protect_y = y_max / 2.0;
   real_t protect_r = 0.15;  // 15 cm radius protection zone

   // SIMP parameters
   real_t E_min = 1e-6, E_max = 1.0, exponent = 3.0;

   // Parse options...
   OptionsParser args(argc, argv);
   // ... (add options)
   args.Parse();

   // ========================================================================
   // MESH & FE SPACES
   // ========================================================================
   Mesh mesh = Mesh::LoadFromFile("lamb-problem-damping-mesh-quads.msh");
   for (int l = 0; l < ref_levels; l++)
      mesh.UniformRefinement();

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   H1_FECollection state_fec(order, 2);
   H1_FECollection filter_fec(order, 2);
   L2_FECollection control_fec(order - 1, 2);

   ParFiniteElementSpace state_fes(&pmesh, &state_fec, 2);   // u, v
   ParFiniteElementSpace filter_fes(&pmesh, &filter_fec);    // ρ̃
   ParFiniteElementSpace control_fes(&pmesh, &control_fec);  // ρ

   if (myid == 0)
   {
      cout << "State DOFs:  " << state_fes.GlobalTrueVSize() << endl;
      cout << "Design DOFs: " << control_fes.GlobalTrueVSize() << endl;
   }

   // ========================================================================
   // DESIGN VARIABLES
   // ========================================================================
   ParGridFunction rho(&control_fes);
   ParGridFunction rho_filter(&filter_fes);
   rho = vol_fraction;  // Initial guess

   // ========================================================================
   // OBJECTIVE FUNCTION
   // ========================================================================
   SubdomainIndicator protect_indicator(protect_x, protect_y, protect_r);
   TimeIntegratedObjective objective(&state_fes, &protect_indicator,
                                      MPI_COMM_WORLD);

   if (myid == 0)
   {
      cout << "\nObjective: Minimize displacement in protected zone" << endl;
      cout << "  Center: (" << protect_x << ", " << protect_y << ")" << endl;
      cout << "  Radius: " << protect_r << endl;
   }

   // ========================================================================
   // HELMHOLTZ FILTER
   // ========================================================================
   // (r² K + M) ρ̃ = M ρ
   // Same as ElastTopOpt_transient.cpp lines 180-190

   // ========================================================================
   // VOLUME CONSTRAINT
   // ========================================================================
   ParLinearForm vol_form(&control_fes);
   ConstantCoefficient one(1.0);
   vol_form.AddDomainIntegrator(new DomainLFIntegrator(one));
   vol_form.Assemble();
   HypreParVector *vol_w = vol_form.ParallelAssemble();

   real_t domain_volume;
   real_t loc = vol_w->Sum();
   MPI_Allreduce(&loc, &domain_volume, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM, MPI_COMM_WORLD);
   const real_t Vstar = vol_fraction * domain_volume;

   // ========================================================================
   // MMA OPTIMIZER SETUP
   // ========================================================================
   int n = control_fes.GetTrueVSize();
   Vector rho_tv(n), dcdrho(n);
   rho.GetTrueDofs(rho_tv);

   Vector dvol(n);
   dvol = *vol_w;
   dvol /= Vstar;

   mfem_mma::MMAOptimizerParallel mma(MPI_COMM_WORLD, n, 1, rho_tv);
   mma.SetAsymptotes(0.5, 0.7, 1.2);

   Vector rho_min(n), rho_max(n);
   real_t move = 0.2;

   // ========================================================================
   // OPTIMIZATION LOOP
   // ========================================================================
   for (int opt_iter = 0; opt_iter < max_opt_iter; opt_iter++)
   {
      if (myid == 0)
         cout << "\n=== Optimization Iteration " << opt_iter + 1 << " ===" << endl;

      // ---------------------------------------------------------------------
      // 1. FORWARD SOLVE: Compute trajectory {u^n, v^n, K^n}
      // ---------------------------------------------------------------------
      if (myid == 0) cout << "Forward solve..." << endl;

      // Apply filter: ρ → ρ̃
      // ... (filter solve)

      // Setup forward operator with current design
      // ... (create ElastodynamicsOperator with M(ρ̃), K(ρ̃))

      // Storage for trajectory
      int num_steps = static_cast<int>(t_final / dt);
      ForwardTrajectoryStorage trajectory(num_steps);

      objective.Reset();

      // Time integration (RK4)
      RK4Solver ode_solver;
      // ... (initialize with ElastodynamicsOperator)

      BlockVector state(/* offsets */);
      state = 0.0;  // Initial condition

      real_t t = 0.0;
      for (int ti = 0; ti < num_steps; ti++)
      {
         // Step forward
         ode_solver.Step(state, t, dt);

         // Extract u^n, v^n
         Vector u_n(/* ... */);
         Vector v_n(/* ... */);

         // Get tangent stiffness K^n (from operator)
         HypreParMatrix *K_n = /* ... */;

         // Store for adjoint
         trajectory.Store(ti, u_n, v_n, K_n);

         // Accumulate objective
         ParGridFunction u_gf(&state_fes);
         u_gf.SetFromTrueDofs(u_n);
         objective.AccumulateTimestep(u_gf, dt, ti, num_steps);

         if (myid == 0 && ti % 100 == 0)
            cout << "  step " << ti << ", t=" << t << endl;
      }

      real_t J = objective.GetObjective();
      if (myid == 0)
         cout << "Objective J = " << J << endl;

      // ---------------------------------------------------------------------
      // 2. ADJOINT SOLVE: Backward march for {λ^n, μ^n}
      // ---------------------------------------------------------------------
      if (myid == 0) cout << "Adjoint solve..." << endl;

      AdjointElastodynamicsOperator adj_oper(state_fes, &trajectory, &objective,
                                              /* M, C_vol, C_abs, */ dt);

      BlockVector eta(/* offsets */);
      adj_oper.InitializeTerminal(num_steps - 1, eta);

      // Backward time integration (use -dt with forward RK4)
      // Paper eq. 1154-1170: Reverse RK4 stages
      RK4Solver adj_solver;
      // ... (initialize with adjoint operator)

      real_t t_adj = t_final;
      for (int ti = num_steps - 1; ti >= 0; ti--)
      {
         adj_oper.SetCurrentStep(ti);

         // Step backward (use -dt)
         adj_solver.Step(eta, t_adj, -dt);

         // Extract λ^n, μ^n
         // ... (for design sensitivity accumulation)

         if (myid == 0 && ti % 100 == 0)
            cout << "  adjoint step " << ti << ", t=" << t_adj << endl;
      }

      // ---------------------------------------------------------------------
      // 3. DESIGN SENSITIVITY: dJ/dρ̃ via adjoint
      // ---------------------------------------------------------------------
      if (myid == 0) cout << "Computing gradient..." << endl;

      // From paper eq. 912-921, 1081-1091:
      // dJ/dρ = ∂J/∂ρ - Σₙ (∂F^n/∂ρ)ᵀ ηⁿ⁺¹
      //
      // For each timestep, accumulate:
      //   ∂F^n/∂ρ = [ 0                                    ]
      //             [ M'[δρ](v^{n+1} - v^n) + Δt K'[δρ]u^{n+1} ]

      Vector dcdrho_tilde(filter_fes.GetTrueVSize());
      dcdrho_tilde = 0.0;

      for (int ti = 0; ti < num_steps; ti++)
      {
         // Get stored forward state
         Vector *u_n = trajectory.u_traj[ti];
         Vector *v_n = trajectory.v_traj[ti];

         // Get adjoint state (stored during backward march)
         // Vector *lambda_n = ...;

         // Compute: ∫ (∂P/∂ρ̃)[δρ̃] : ∇λ dx
         // Where P = first Piola-Kirchhoff = ∂Ψ/∂F
         // ... (similar to ElastTopOpt_transient.cpp lines 262-271)
      }

      // ---------------------------------------------------------------------
      // 4. FILTER ADJOINT: dJ/dρ̃ → dJ/dρ
      // ---------------------------------------------------------------------
      // (r² K + M) w̃ = -dJ/dρ̃
      // dJ/dρ = (w̃, ·)_L2
      // ... (same as static case, ElastTopOpt_transient.cpp lines 262-271)

      // ---------------------------------------------------------------------
      // 5. MMA UPDATE
      // ---------------------------------------------------------------------
      real_t vol = InnerProduct(MPI_COMM_WORLD, *vol_w, rho_tv) / domain_volume;

      Vector fival(1);
      fival(0) = InnerProduct(MPI_COMM_WORLD, *vol_w, rho_tv) / Vstar - 1.0;

      Vector *dfidx[1];
      dfidx[0] = &dvol;

      // Box constraints with move limits
      for (int i = 0; i < n; i++)
      {
         rho_min[i] = max(0.0, rho_tv[i] - move);
         rho_max[i] = min(1.0, rho_tv[i] + move);
      }

      mma.Update(rho_tv, dcdrho, J, fival, dfidx, rho_min, rho_max);
      rho.SetFromTrueDofs(rho_tv);

      if (myid == 0)
      {
         cout << "  Objective: " << J << endl;
         cout << "  Volume: " << vol << endl;
      }

      // Save visualization
      // ... (ParaView output)
   }

   if (myid == 0)
      cout << "\nOptimization completed!" << endl;

   return 0;
}
