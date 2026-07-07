// =============================================================================
// Elastodynamics Solver for Transient Topology Optimization
// =============================================================================
//
// Unified solver for forward and adjoint elastodynamics with design dependence.
// Follows the mtop-chkpt framework pattern with both Mult() and
// JacobianMultTranspose() in a single class.
//
// STATE VECTOR: x = [u, v] where u = displacement, v = velocity
// ADJOINT VECTOR: η = [μ, λ] (same structure)
//
// DESIGN DEPENDENCE:
//   - Mass: M(ρ) via SIMP interpolation
//   - Stiffness: K(ρ) via SIMP interpolation
//
// REFERENCE:
//   - Theory: topopt.tex Section 5 (transient topology optimization)
//   - Pattern: mtop-chkpt/mtop_solvers.hpp
//
// =============================================================================

#ifndef ELASTODYNAMICS_SOLVER_HPP
#define ELASTODYNAMICS_SOLVER_HPP

#include "mfem.hpp"
#include "ObjectiveFunctional.hpp"     // TimeIntegratedObjective (J, dJ/du)
#include "ProblemSpecification.hpp"    // MaterialParams, BoundaryLoadSpec, damping
#include "../../pde_filter.hpp"
#include <memory>
#include <vector>
#include <iomanip>
#include <iostream>

namespace mfem
{

// =============================================================================
// SIMP MATERIAL INTERPOLATION
// =============================================================================
// Computes r(ρ̃) = r_min + ρ̃^p (r_max - r_min)
class SIMPCoefficient : public Coefficient
{
private:
   GridFunction *rho_filter;  // Filtered density ρ̃
   real_t r_min, r_max;
   real_t exponent;

public:
   SIMPCoefficient(GridFunction *rho_filt, real_t rmin, real_t rmax, real_t p)
      : rho_filter(rho_filt), r_min(rmin), r_max(rmax), exponent(p) {}

   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      real_t rho_val = rho_filter->GetValue(T, ip);
      rho_val = std::min(std::max(rho_val, 0.0), 1.0);  // Clamp to [0,1]
      real_t rho_pow = std::pow(rho_val, exponent);
      return r_min + rho_pow * (r_max - r_min);
   }
};

// SIMP derivative: r'(ρ̃) = p ρ̃^(p-1) (r_max - r_min)
class SIMPGradCoefficient : public Coefficient
{
private:
   GridFunction *rho_filter;
   real_t r_min, r_max;
   real_t exponent;

public:
   SIMPGradCoefficient(GridFunction *rho_filt, real_t rmin, real_t rmax, real_t p)
      : rho_filter(rho_filt), r_min(rmin), r_max(rmax), exponent(p) {}

   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      real_t rho_val = rho_filter->GetValue(T, ip);
      rho_val = std::min(std::max(rho_val, 0.0), 1.0);
      if (rho_val < 1e-12) return 0.0;  // Avoid singularity at ρ=0
      real_t rho_pow = std::pow(rho_val, exponent - 1.0);
      return exponent * rho_pow * (r_max - r_min);
   }
};

// =============================================================================
// GAUSSIAN DESIGN DISTRIBUTION
// =============================================================================
// Creates a 2D Gaussian design field: ρ̃(x,y) = ρ_min + (1-ρ_min) * exp(-r²/(2σ²))
class GaussianDesignCoefficient : public Coefficient
{
private:
   real_t x_center, y_center;
   real_t sigma_x, sigma_y;
   real_t rho_min, rho_max;

public:
   GaussianDesignCoefficient(real_t xc, real_t yc, real_t sx, real_t sy,
                             real_t rmin = 0.3, real_t rmax = 1.0)
      : x_center(xc), y_center(yc), sigma_x(sx), sigma_y(sy),
        rho_min(rmin), rho_max(rmax) {}

   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      Vector x(2);
      T.Transform(ip, x);

      real_t dx = (x(0) - x_center) / sigma_x;
      real_t dy = (x(1) - y_center) / sigma_y;
      real_t r_squared = dx * dx + dy * dy;

      real_t gaussian = std::exp(-0.5 * r_squared);
      real_t rho = rho_min + (rho_max - rho_min) * gaussian;

      return std::min(std::max(rho, 0.0), 1.0);
   }
};

// DampingProfile / SpatialDampingCoefficient: promoted to DampingSpec.hpp
// (shared with the problem layer, which assembles them via DampingField).

// =============================================================================
// FORWARD TRAJECTORY STORAGE
// =============================================================================
// Storage for forward state needed by adjoint solver
struct ForwardTrajectoryStorage
{
   Array<Vector*> u_traj;       // Displacement at each timestep
   Array<Vector*> v_traj;       // Velocity at each timestep
   Array<HypreParMatrix*> K_traj;  // Tangent stiffness at each timestep

   int num_steps;
   bool storage_enabled;

   ForwardTrajectoryStorage(int n) : num_steps(n), storage_enabled(false)
   {
      u_traj.SetSize(n);
      v_traj.SetSize(n);
      K_traj.SetSize(n);

      for (int i = 0; i < n; i++)
      {
         u_traj[i] = nullptr;
         v_traj[i] = nullptr;
         K_traj[i] = nullptr;
      }
   }

   void EnableStorage() { storage_enabled = true; }

   void Store(int step, const Vector &u, const Vector &v, HypreParMatrix *K)
   {
      if (!storage_enabled) return;

      if (step >= num_steps) return;

      if (u_traj[step]) delete u_traj[step];
      if (v_traj[step]) delete v_traj[step];

      u_traj[step] = new Vector(u);
      v_traj[step] = new Vector(v);
      K_traj[step] = K;  // Store pointer only (managed by operator)
   }

   ~ForwardTrajectoryStorage()
   {
      for (int i = 0; i < num_steps; i++)
      {
         delete u_traj[i];
         delete v_traj[i];
         // Don't delete K_traj[i] - managed by operator
      }
   }
};

// =============================================================================
// MASS SOLVER STRATEGIES
// =============================================================================
enum class MassSolverType
{
   LUMPED,     // Diagonal (row-sum) mass matrix, optimal for explicit RK4
   ITERATIVE   // CG + AMG, for verification/comparison
};

// =============================================================================
// ELASTODYNAMICS OPERATOR (Forward and Adjoint)
// =============================================================================
// Implements both forward and adjoint elastodynamics operators.
//
// FORWARD: [u̇]   [                        v                          ]
//          [v̇] = [ M^{-1}(-K u - C_vol v - C_abs v + f(t)) ]
//
// ADJOINT: Uses JacobianMultTranspose for discrete adjoint via RK4 transpose
//
class ElastodynamicsOperator : public TimeDependentOperator
{
private:
   ParFiniteElementSpace &fespace;
   ParBilinearForm *M, *K, *C_vol, *C_abs;
   HypreParMatrix *Mmat, *Kmat, *Cvol_mat, *Cabs_mat;

   // Mass solver strategy
   MassSolverType mass_solver_type;

   // Lumped mass inverse (diagonal vector)
   Vector M_lumped_inv;

   // Iterative solver (CG + AMG) - only used if mass_solver_type == ITERATIVE
   HypreBoomerAMG *M_prec;
   CGSolver *M_solver;

   mutable ParGridFunction u_gf;  // For visualization
   mutable ParGridFunction v_gf;  // For visualization

   Array<int> ess_tdof_list;
   Array<int> block_true_offsets;
   int true_size;

   mutable Vector res;
   mutable Vector tmp;

   // Precomputed load vector (optimization B)
   Vector load_base_vector;
   real_t load_duration, load_amplitude, load_phase, load_frequency;
   LoadTimeProfile load_time_profile;
   bool load_on_domain;   // true: body force over Omega; false: boundary traction
   Array<int> load_bdr_markers;

   // Trajectory storage for adjoint
   ForwardTrajectoryStorage *trajectory;
   TimeIntegratedObjective *objective;

   // Current timestep for adjoint evaluation (mutable for use in const methods)
   mutable int current_adjoint_step;

public:
   ElastodynamicsOperator(
      ParFiniteElementSpace &f,
      Coefficient &mass_coef,
      Coefficient &lambda_coef,
      Coefficient &mu_coef,
      real_t amplitude, real_t duration,
      LoadTimeProfile load_profile,
      real_t load_phase,
      real_t load_frequency,
      const Array<int> &load_bdr_attrs,
      VectorCoefficient &load_coef,
      bool domain_load,
      Coefficient *gamma_coef,   // any damping field gamma(x): sponge, uniform, ...
      real_t impedance,
      Array<int> &exterior_bdr_attr,
      Array<int> &ess_bdr_attr,
      MassSolverType mass_type = MassSolverType::LUMPED,
      bool print_banner = true,
      ForwardTrajectoryStorage *traj = nullptr,
      TimeIntegratedObjective *obj = nullptr);

   void SetTime(real_t t) override { TimeDependentOperator::SetTime(t); }

   /// Forward RHS evaluation: dy/dt = f(y,t)
   virtual void Mult(const Vector &x, Vector &y) const override;

   /// Adjoint RHS evaluation: dη/dt = -f_y(y,t)^T η + q(y,t)
   /// Following mtop-chkpt pattern with JacobianMultTranspose
   // Returns the plain transpose action y = (df/dx)^T eta.
   virtual void JacobianMultTranspose(const Vector &x,
                                       const Vector &eta,
                                       Vector &eta_rhs) const;

   const Array<int>& GetEssentialTrueDofs() const { return ess_tdof_list; }

   ParGridFunction& GetDisplacement() { return u_gf; }
   ParGridFunction& GetVelocity() { return v_gf; }

   Array<int>& GetBlockOffsets() { return block_true_offsets; }

   MassSolverType GetMassSolverType() const { return mass_solver_type; }

   HypreParMatrix* GetMassMatrix() const { return Mmat; }
   HypreParMatrix* GetStiffnessMatrix() const { return Kmat; }
   HypreParMatrix* GetVolDampingMatrix() const { return Cvol_mat; }
   HypreParMatrix* GetAbsDampingMatrix() const { return Cabs_mat; }

   void MultInvMass(const Vector &rhs, Vector &sol) const
   {
      sol.SetSize(true_size);
      if (mass_solver_type == MassSolverType::LUMPED)
      {
         // Diagonal solve: sol[i] = M_lumped_inv[i] * rhs[i]
         for (int i = 0; i < true_size; i++)
         {
            sol[i] = M_lumped_inv[i] * rhs[i];
         }
      }
      else // ITERATIVE
      {
         sol = 0.0;
         M_solver->Mult(rhs, sol);
      }
   }

   // Homogeneous Dirichlet (clamped) enforcement. u = v = 0 (hence u_dot = v_dot
   // = 0) on the essential dofs for all time, so we zero the essential entries of
   // both blocks of a full state / state-derivative vector z = [u, v]. Applied to
   // the forward RHS (Mult) and, symmetrically, to the adjoint input
   // (JacobianMultTranspose); the projection is symmetric, keeping the discrete
   // adjoint an exact transpose. Exact for lumped mass (diagonal, no reaction
   // coupling); the consistent-mass clamped solve would additionally need the
   // essential rows/cols eliminated from M.
   void ProjectEssentialBC(Vector &z) const
   {
      if (ess_tdof_list.Size() == 0) { return; }
      BlockVector bz(z, block_true_offsets);
      Vector &u = bz.GetBlock(0);
      Vector &v = bz.GetBlock(1);
      for (int i = 0; i < ess_tdof_list.Size(); i++)
      {
         const int d = ess_tdof_list[i];
         u(d) = 0.0;
         v(d) = 0.0;
      }
   }

   void SetTrajectory(ForwardTrajectoryStorage *traj) { trajectory = traj; }
   void SetObjective(TimeIntegratedObjective *obj) { objective = obj; }

   void SetAdjointTimestep(int step) { current_adjoint_step = step; }

   /// Store current state in trajectory (call during forward march)
   void StoreTrajectoryStep(int step, const Vector &state) const
   {
      if (!trajectory) return;

      BlockVector bstate(const_cast<Vector&>(state), block_true_offsets);
      Vector u_true(bstate.GetBlock(0).GetData(), true_size);
      Vector v_true(bstate.GetBlock(1).GetData(), true_size);

      trajectory->Store(step, u_true, v_true, Kmat);
   }

   /// Accumulate objective functional during forward march
   real_t AccumulateObjective(const Vector &state, real_t dt, int step, int total_steps) const
   {
      if (!objective) return 0.0;

      BlockVector bstate(const_cast<Vector&>(state), block_true_offsets);
      Vector u_true(bstate.GetBlock(0).GetData(), true_size);

      u_gf.SetFromTrueDofs(u_true);
      return objective->AccumulateTimestep(u_gf, dt, step, total_steps);
   }

   /// Initialize terminal condition for adjoint solve
   /// From paper eq. 996-997: (A_N^+)^T η^N = q^N
   /// For J_T = 0 (no terminal objective), η^N = 0
   void InitializeTerminalAdjoint(Vector &eta_final) const
   {
      eta_final = 0.0;

      // If terminal objective exists: η^N = ∂J_T/∂z^N
      // For our case J_T = 0, so terminal adjoint is zero
      // This would be modified if we have a terminal cost
   }

   /// Compute objective gradient at current timestep
   /// Returns q^n = [∂J_Ω/∂u^n, ∂J_Ω/∂v^n]
   void ComputeObjectiveGradient(int step, int total_steps, real_t dt,
                                  Vector &q_u, Vector &q_v) const
   {
      q_u = 0.0;
      q_v = 0.0;

      if (!objective || !trajectory) return;
      if (step >= trajectory->num_steps) return;

      // Get forward state at this step
      Vector *u_n = trajectory->u_traj[step];
      if (!u_n) return;

      // Set grid function from stored state
      u_gf.SetFromTrueDofs(*u_n);

      // Compute ∂J_Ω/∂u = 2 χ_Ω̃ u (from ObjectiveFunctional)
      ParLinearForm grad_form(&fespace);
      objective->ComputeObjectiveGradient(u_gf, dt, step, total_steps, grad_form);
      grad_form.ParallelAssemble(q_u);

      // For J = ∫∫ |u|² dx dt, we have ∂J_Ω/∂v = 0
      // q_v already set to zero
   }

   virtual ~ElastodynamicsOperator();
};

ElastodynamicsOperator::ElastodynamicsOperator(
   ParFiniteElementSpace &f,
   Coefficient &mass_coef,
   Coefficient &lambda_coef,
   Coefficient &mu_coef,
   real_t amplitude, real_t duration,
   LoadTimeProfile load_profile,
   real_t phase,
   real_t frequency,
   const Array<int> &load_bdr_attrs,
   VectorCoefficient &load_coef,
   bool domain_load,
   Coefficient *gamma_coef,
   real_t impedance,
   Array<int> &exterior_bdr_attr,
   Array<int> &ess_bdr_attr,
   MassSolverType mass_type,
   bool print_banner,
   ForwardTrajectoryStorage *traj,
   TimeIntegratedObjective *obj)
   : TimeDependentOperator(2 * f.GetTrueVSize(), 0.0),
     fespace(f),
     mass_solver_type(mass_type),
     M_prec(nullptr),
     M_solver(nullptr),
     u_gf(&fespace),
     v_gf(&fespace),
     true_size(f.GetTrueVSize()),
     res(true_size),
     tmp(true_size),
     load_base_vector(true_size),
     load_duration(duration),
     load_amplitude(amplitude),
     load_phase(phase),
     load_frequency(frequency),
     load_time_profile(load_profile),
     load_on_domain(domain_load),
     trajectory(traj),
     objective(obj),
     current_adjoint_step(-1)
{
   int myid = Mpi::WorldRank();

   // Block structure: [displacement, velocity]
   block_true_offsets.SetSize(3);
   block_true_offsets[0] = 0;
   block_true_offsets[1] = true_size;
   block_true_offsets[2] = 2 * true_size;

   u_gf = 0.0;
   v_gf = 0.0;
   res = 0.0;
   tmp = 0.0;
   load_base_vector = 0.0;

   fespace.GetEssentialTrueDofs(ess_bdr_attr, ess_tdof_list);

   // Report the GLOBAL essential-dof count (the local count on rank 0 is
   // partition-dependent and misleading - it can read 0 while the constraint is
   // active on other ranks).
   long long local_ess = ess_tdof_list.Size(), global_ess = 0;
   MPI_Allreduce(&local_ess, &global_ess, 1, MPI_LONG_LONG, MPI_SUM,
                 fespace.GetComm());

   if (myid == 0 && print_banner)
   {
      std::cout << "\n=== Elastodynamics Operator ===" << std::endl;
      std::cout << "DOFs per field: " << true_size << std::endl;
      std::cout << "Essential DOFs: " << global_ess << std::endl;
      std::cout << "Mass solver: "
                << (mass_solver_type == MassSolverType::LUMPED ? "LUMPED" : "ITERATIVE")
                << std::endl;
   }

   // Assemble design-dependent mass matrix: M(ρ)
   M = new ParBilinearForm(&fespace);
   M->AddDomainIntegrator(new VectorMassIntegrator(mass_coef));
   M->Assemble();
   M->Finalize();
   Mmat = M->ParallelAssemble();

   // Assemble design-dependent stiffness matrix: K(ρ)
   K = new ParBilinearForm(&fespace);
   K->AddDomainIntegrator(new ElasticityIntegrator(lambda_coef, mu_coef));
   K->Assemble();
   K->Finalize();
   Kmat = K->ParallelAssemble();

   // Assemble volumetric damping matrix
   C_vol = new ParBilinearForm(&fespace);
   C_vol->AddDomainIntegrator(new VectorMassIntegrator(*gamma_coef));
   C_vol->Assemble();
   C_vol->Finalize();
   Cvol_mat = C_vol->ParallelAssemble();

   // Assemble absorbing boundary condition matrix
   C_abs = new ParBilinearForm(&fespace);
   ConstantCoefficient impedance_coef(impedance);
   C_abs->AddBoundaryIntegrator(new VectorMassIntegrator(impedance_coef), exterior_bdr_attr);
   C_abs->Assemble();
   C_abs->Finalize();
   Cabs_mat = C_abs->ParallelAssemble();

   HYPRE_BigInt mass_nnz = Mmat->NNZ();
   HYPRE_BigInt stiff_nnz = Kmat->NNZ();
   HYPRE_BigInt cvol_nnz = Cvol_mat->NNZ();
   HYPRE_BigInt cabs_nnz = Cabs_mat->NNZ();

   if (myid == 0)
   {
      std::cout << "Matrix assembly complete:" << std::endl;
      std::cout << "  Mass NNZ:     " << mass_nnz << std::endl;
      std::cout << "  Stiffness NNZ: " << stiff_nnz << std::endl;
      std::cout << "  Damping NNZ:   " << cvol_nnz << std::endl;
      std::cout << "  ABC NNZ:       " << cabs_nnz << std::endl;
   }

   // Set up mass matrix solver based on selected strategy
   if (mass_solver_type == MassSolverType::LUMPED)
   {
      // Compute lumped (row-sum) mass matrix: M_lumped = M * ones
      M_lumped_inv.SetSize(true_size);
      Vector ones(true_size);
      ones = 1.0;
      Mmat->Mult(ones, M_lumped_inv);

      // Check for near-zero entries (shouldn't happen for proper mass matrix)
      real_t min_mass = M_lumped_inv.Min();
      if (min_mass < 1e-14)
      {
         if (myid == 0)
         {
            std::cerr << "Warning: lumped mass has near-zero entries (min="
                      << min_mass << "). Using max(m, 1e-14)." << std::endl;
         }
         for (int i = 0; i < true_size; i++)
         {
            M_lumped_inv[i] = std::max(M_lumped_inv[i], 1e-14);
         }
      }

      // Invert: M_lumped_inv[i] = 1 / M_lumped[i]
      for (int i = 0; i < true_size; i++)
      {
         M_lumped_inv[i] = 1.0 / M_lumped_inv[i];
      }

      if (myid == 0)
      {
         std::cout << "Inverse lumped mass range: ["
                   << M_lumped_inv.Min() << ", "
                   << M_lumped_inv.Max() << "]" << std::endl;
      }
   }
   else // ITERATIVE
   {
      M_prec = new HypreBoomerAMG(*Mmat);
      M_prec->SetPrintLevel(0);

      M_solver = new CGSolver(fespace.GetComm());
      M_solver->SetPreconditioner(*M_prec);
      M_solver->SetOperator(*Mmat);
      M_solver->SetRelTol(1e-12);
      M_solver->SetAbsTol(0.0);
      M_solver->SetMaxIter(100);
      M_solver->SetPrintLevel(0);
   }

   // Set up boundary markers for loading
   ParMesh *pmesh = fespace.GetParMesh();
   int max_bdr_attr = pmesh->bdr_attributes.Max();
   load_bdr_markers.SetSize(max_bdr_attr);
   load_bdr_markers = 0;

   // Mark load boundaries supplied by the problem/configuration layer.
   for (int i = 0; i < load_bdr_attrs.Size(); i++)
   {
      const int attr = load_bdr_attrs[i];
      if (attr >= 1 && attr <= max_bdr_attr)
      {
         load_bdr_markers[attr - 1] = 1;
      }
   }

   // Precompute base load vector (optimization B): load(t) = load_base_vector
   // * time_factor(t), assembled once so the inner loop never re-assembles. The
   // load is either a boundary traction on load_bdr_markers or a body force over
   // the whole domain (e.g. a concentrated tip load), per the problem.
   ParLinearForm load_form(&fespace);
   if (load_on_domain)
   {
      load_form.AddDomainIntegrator(new VectorDomainLFIntegrator(load_coef));
   }
   else
   {
      load_form.AddBoundaryIntegrator(
         new VectorBoundaryLFIntegrator(load_coef), load_bdr_markers);
   }
   load_form.Assemble();
   load_form.ParallelAssemble(load_base_vector);

   if (myid == 0 && print_banner)
   {
      std::cout << "\nTime-dependent loading:" << std::endl;
      std::cout << "  Support: "
                << (load_on_domain ? "body force (domain)" : "traction (boundary)")
                << std::endl;
      std::cout << "  Time profile: " << LoadTimeProfileName(load_time_profile)
                << std::endl;
      std::cout << "  Amplitude: " << amplitude << std::endl;
      if (load_time_profile == LoadTimeProfile::GAUSSIAN ||
          load_time_profile == LoadTimeProfile::MODULATED_GAUSSIAN)
      {
         std::cout << "  Pulse duration: " << duration << " s" << std::endl;
         if (load_time_profile == LoadTimeProfile::MODULATED_GAUSSIAN)
         {
            std::cout << "  Carrier frequency: " << load_frequency
                      << ",  phase: " << load_phase << std::endl;
         }
      }
      else if (load_time_profile == LoadTimeProfile::HARMONIC)
      {
         std::cout << "  Frequency: " << load_frequency
                   << ",  phase: " << load_phase << std::endl;
      }
      std::cout << "  Base load norm: " << load_base_vector.Norml2() << std::endl;
      std::cout << "====================================\n" << std::endl;
   }
}

void ElastodynamicsOperator::Mult(const Vector &x, Vector &y) const
{
   real_t time = this->GetTime();

   y = 0.0;

   // Extract state blocks: x = [u, v], y = [u̇, v̇]
   BlockVector bx(const_cast<Vector&>(x), block_true_offsets);
   BlockVector by(y, block_true_offsets);

   Vector u_true(bx.GetBlock(0).GetData(), true_size);
   Vector v_true(bx.GetBlock(1).GetData(), true_size);

   // First equation: u̇ = v
   by.GetBlock(0) = v_true;

   // Second equation: v̇ = M^{-1}(-K u - C_vol v - C_abs v + f(t))
   res = 0.0;

   // Elastic restoring force: -K u
   Kmat->Mult(u_true, tmp);
   res.Add(-1.0, tmp);

   // Volumetric damping: -C_vol v
   Cvol_mat->Mult(v_true, tmp);
   res.Add(-1.0, tmp);

   // Absorbing boundary damping: -C_abs v
   Cabs_mat->Mult(v_true, tmp);
   res.Add(-1.0, tmp);

   // Time-dependent applied load (optimization B: precomputed base vector)
   real_t time_factor = 1.0;
   if (load_time_profile == LoadTimeProfile::GAUSSIAN)
   {
      const real_t t_center = load_duration / 2.0;
      const real_t sigma = load_duration / 4.0;
      const real_t t_diff = time - t_center;
      time_factor = exp(-t_diff * t_diff / (2.0 * sigma * sigma));
   }
   else if (load_time_profile == LoadTimeProfile::MODULATED_GAUSSIAN)
   {
      const real_t pi = 3.1415926535897932384626433832795;
      const real_t t_center = load_duration / 2.0;
      const real_t sigma = load_duration / 4.0;
      const real_t t_diff = time - t_center;
      const real_t envelope = exp(-t_diff * t_diff / (2.0 * sigma * sigma));
      const real_t carrier = cos(2.0 * pi * load_frequency * t_diff
                                 + load_phase);
      time_factor = envelope * carrier;
   }
   else if (load_time_profile == LoadTimeProfile::HARMONIC)
   {
      time_factor = sin(load_frequency * time + load_phase);
   }
   const real_t current_amplitude = load_amplitude * time_factor;

   // Scale precomputed load: res += current_amplitude * load_base_vector
   res.Add(current_amplitude, load_base_vector);

   // Solve M v̇ = res (optimization C: removed allreduce guard)
   MultInvMass(res, by.GetBlock(1));

   // Clamped dofs stay at rest: u_dot = v_dot = 0 there.
   ProjectEssentialBC(y);
}

void ElastodynamicsOperator::JacobianMultTranspose(const Vector &x,
                                                    const Vector &eta,
                                                    Vector &eta_rhs) const
{
   // Plain transpose of the forward RHS Jacobian:
   // F([u,v]) = [v, M^{-1}(-K u - C v + f(t))].
   // Therefore (dF/dx)^T [mu,lambda] =
   // [-K^T M^{-T} lambda, mu - C^T M^{-T} lambda].
   // The applied load is independent of the state, so it contributes zero.
   (void)x;

   eta_rhs = 0.0;

   // Transpose of the essential-BC projection applied in Mult (P F): since P is
   // symmetric, the transpose applies P to the incoming adjoint before the
   // unconstrained transpose. Project a local copy so the caller's vector is
   // untouched.
   Vector eta_p(eta);
   ProjectEssentialBC(eta_p);

   BlockVector b_eta_new(eta_p, block_true_offsets);
   BlockVector b_eta_rhs_new(eta_rhs, block_true_offsets);

   Vector mu_new(b_eta_new.GetBlock(0).GetData(), true_size);
   Vector lambda_new(b_eta_new.GetBlock(1).GetData(), true_size);

   Vector m_inv_lambda(true_size);
   m_inv_lambda = 0.0;
   MultInvMass(lambda_new, m_inv_lambda);

   Kmat->MultTranspose(m_inv_lambda, tmp);
   b_eta_rhs_new.GetBlock(0).Add(-1.0, tmp);

   b_eta_rhs_new.GetBlock(1) = mu_new;

   Cvol_mat->MultTranspose(m_inv_lambda, tmp);
   b_eta_rhs_new.GetBlock(1).Add(-1.0, tmp);

   Cabs_mat->MultTranspose(m_inv_lambda, tmp);
   b_eta_rhs_new.GetBlock(1).Add(-1.0, tmp);

   return;

   // Adjoint RHS evaluation for discrete adjoint (DO) via RK4 transpose
   // Following the pattern from tst_rk4_adj.cpp (lines 108-121)
   //
   // Forward system:
   //   u̇ = v
   //   v̇ = M^{-1}(-K u - C_vol v - C_abs v)
   //
   // Let F(z) = [v, M^{-1}(-K u - C v)] where z = [u, v]
   //
   // Jacobian transpose:
   //   ∂F/∂z = [ 0           I           ]
   //           [ -M^{-1}K   -M^{-1}C     ]
   //
   //   (∂F/∂z)^T = [ 0          -K^T M^{-T}      ]
   //                [ I          -C^T M^{-T}     ]
   //
   // Adjoint equation: η̇ = -(∂F/∂z)^T η + q
   // where η = [μ, λ], q = objective gradient

   eta_rhs = 0.0;

   BlockVector b_eta(const_cast<Vector&>(eta), block_true_offsets);
   BlockVector b_eta_rhs(eta_rhs, block_true_offsets);

   Vector mu(b_eta.GetBlock(0).GetData(), true_size);
   Vector lambda(b_eta.GetBlock(1).GetData(), true_size);

   // First adjoint equation: μ̇ = -0 * μ - I * λ + q_u
   //                             = -λ + q_u
   b_eta_rhs.GetBlock(0).Set(-1.0, lambda);

   // Second adjoint equation: λ̇ = K^T M^{-T} μ + C^T M^{-T} λ + q_v
   //
   // Step 1: Compute rhs = K^T μ + C^T λ
   Vector rhs(true_size);
   rhs = 0.0;

   // Add K^T μ
   Kmat->MultTranspose(mu, tmp);
   rhs.Add(1.0, tmp);

   // Add C_vol^T λ
   Cvol_mat->MultTranspose(lambda, tmp);
   rhs.Add(1.0, tmp);

   // Add C_abs^T λ
   Cabs_mat->MultTranspose(lambda, tmp);
   rhs.Add(1.0, tmp);

   // Step 2: Solve M^T λ̇ = rhs  (since M is symmetric: M^T = M)
   b_eta_rhs.GetBlock(1) = 0.0;
   M_solver->Mult(rhs, b_eta_rhs.GetBlock(1));

   // Note: Objective gradient q = [q_u, q_v] will be added separately
   // during the adjoint RK4 march. The RK4 transpose will call this
   // function at intermediate stages, and we'll add q at the full timestep.
   // For J = ∫∫ |u|² dx dt, we have ∂J/∂v = 0, so q_v = 0
}

ElastodynamicsOperator::~ElastodynamicsOperator()
{
   delete M_solver;
   delete M_prec;
   delete Cabs_mat;
   delete Cvol_mat;
   delete Kmat;
   delete Mmat;
   delete C_abs;
   delete C_vol;
   delete K;
   delete M;
}

// =============================================================================
// REUSABLE ADJOINT + DESIGN SENSITIVITY
// =============================================================================
// Verified to machine precision / expected Taylor orders in
// test_adjoint_verification.cpp. These replaced the earlier (incorrect)
// DesignSensitivityAccumulator / AdjointBackwardMarch stubs (now removed).
//
// The per-step adjoint (RK4AdjointOneStep / RK4AdjointOneStepWithDesign)
// consumes a SINGLE forward state, matching the adjoint_step(adj, fwd_state, i)
// callback of mtop-chkpt's DynamicCheckpointing. Today the forward states come
// from full storage; adding checkpointing later only changes how they are
// supplied, not the math here.

inline real_t SimpDerivative(const ParGridFunction &rho_tilde,
                             ElementTransformation &T,
                             const IntegrationPoint &ip,
                             const MaterialParams &mat)
{
   real_t rho = rho_tilde.GetValue(T, ip);
   rho = std::min(std::max(rho, real_t(0.0)), real_t(1.0));
   if (rho <= 0.0) { return 0.0; }
   return mat.simp_p * std::pow(rho, mat.simp_p - 1.0)
          * (mat.r_max - mat.r_min);
}

// Mass-matrix design sensitivity: assembles the filter-space linear form
//   elvect(k) += integral[ -rho0 * SIMP'(rho_tilde) * (a . z) * phi_k ] dx,
// which is d/d(rho_tilde) of -z^T M(rho_tilde) a, i.e. the mass contribution to
// dJ/d(rho_tilde) in the discrete adjoint (a = stage acceleration, z = M^{-1} of
// the adjoint velocity seed).
//
// The forward solve can use either the *consistent* mass M or the row-lumped
// mass M_lumped = diag(integral SIMP*rho0*phi_i) (partition of unity). The design
// sensitivity MUST differentiate whichever mass drives the forward solve:
//   - CONSISTENT: (a . z) is the L2 product of the interpolated fields at the
//     quadrature point, (sum_i a_i phi_i) . (sum_j z_j phi_j).
//   - LUMPED: the diagonal lump collapses the product to the nodal contraction
//     g(x) = sum_i (a_i . z_i) phi_i(x), the interpolant of the per-node dot
//     products. Using the consistent product with a lumped forward solve yields
//     an inconsistent (wrong) gradient.
class StageMassDesignLFIntegrator : public LinearFormIntegrator
{
private:
   ParGridFunction &rho_tilde;
   ParGridFunction &accel;
   ParGridFunction &z;
   MaterialParams mat;
   bool lumped;
   Vector shape, accel_val, z_val;

public:
   StageMassDesignLFIntegrator(ParGridFunction &rho_tilde_,
                               ParGridFunction &accel_,
                               ParGridFunction &z_,
                               const MaterialParams &mat_,
                               bool lumped_ = false)
      : rho_tilde(rho_tilde_), accel(accel_), z(z_), mat(mat_),
        lumped(lumped_) {}

   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &T,
                               Vector &elvect) override
   {
      const int dof = el.GetDof();
      shape.SetSize(dof);
      elvect.SetSize(dof);
      elvect = 0.0;

      // For the lumped mass, precompute the per-node dot products
      // g_i = a_i . z_i from the element-local nodal (vector) dofs.
      Vector g_nodal;
      if (lumped)
      {
         const FiniteElementSpace *afes = accel.FESpace();
         const int vdim = afes->GetVDim();
         const int ordering = afes->GetOrdering();
         Array<int> vdofs;
         afes->GetElementVDofs(T.ElementNo, vdofs);
         Vector a_edof, z_edof;
         accel.GetSubVector(vdofs, a_edof);
         z.GetSubVector(vdofs, z_edof);

         g_nodal.SetSize(dof);
         for (int i = 0; i < dof; i++)
         {
            real_t s = 0.0;
            for (int c = 0; c < vdim; c++)
            {
               const int idx = (ordering == Ordering::byNODES)
                               ? (c * dof + i) : (i * vdim + c);
               s += a_edof(idx) * z_edof(idx);
            }
            g_nodal(i) = s;
         }
      }

      const int int_order = 2 * el.GetOrder() + T.OrderW();
      const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), int_order);

      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T.SetIntPoint(&ip);
         el.CalcPhysShape(T, shape);

         real_t az;
         if (lumped)
         {
            az = g_nodal * shape;   // g(x) = sum_i (a_i . z_i) phi_i(x)
         }
         else
         {
            accel.GetVectorValue(T, ip, accel_val);
            z.GetVectorValue(T, ip, z_val);
            az = accel_val * z_val;
         }

         const real_t rp = SimpDerivative(rho_tilde, T, ip, mat);
         const real_t density = -mat.rho0 * rp * az;
         const real_t weight = ip.weight * T.Weight() * density;

         for (int i = 0; i < dof; i++)
         {
            elvect(i) += weight * shape(i);
         }
      }
   }

   using LinearFormIntegrator::AssembleRHSElementVect;
};

class StageStiffnessDesignLFIntegrator : public LinearFormIntegrator
{
private:
   ParGridFunction &rho_tilde;
   ParGridFunction &u;
   ParGridFunction &z;
   MaterialParams mat;
   Vector shape;
   DenseMatrix grad_u, grad_z;

public:
   StageStiffnessDesignLFIntegrator(ParGridFunction &rho_tilde_,
                                    ParGridFunction &u_,
                                    ParGridFunction &z_,
                                    const MaterialParams &mat_)
      : rho_tilde(rho_tilde_), u(u_), z(z_), mat(mat_) {}

   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &T,
                               Vector &elvect) override
   {
      const int dof = el.GetDof();
      shape.SetSize(dof);
      elvect.SetSize(dof);
      elvect = 0.0;

      const int int_order = 2 * T.OrderGrad(&el);
      const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), int_order);

      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T.SetIntPoint(&ip);
         el.CalcPhysShape(T, shape);

         u.GetVectorGradient(T, grad_u);
         z.GetVectorGradient(T, grad_z);

         const int dim = T.GetSpaceDim();
         const real_t div_u = grad_u.Trace();
         const real_t div_z = grad_z.Trace();

         real_t elastic_density = mat.lambda0 * div_u * div_z;
         for (int i = 0; i < dim; i++)
         {
            for (int j = 0; j < dim; j++)
            {
               elastic_density += mat.mu0 * grad_z(i, j)
                                  * (grad_u(i, j) + grad_u(j, i));
            }
         }

         const real_t rp = SimpDerivative(rho_tilde, T, ip, mat);
         const real_t density = -rp * elastic_density;
         const real_t weight = ip.weight * T.Weight() * density;

         for (int i = 0; i < dof; i++)
         {
            elvect(i) += weight * shape(i);
         }
      }
   }

   using LinearFormIntegrator::AssembleRHSElementVect;
};

inline void EvalRHS(ElastodynamicsOperator &oper,
                    const Vector &x, real_t t, Vector &y)
{
   y.SetSize(x.Size());
   oper.SetTime(t);
   oper.Mult(x, y);
}

inline void EvalJacobianTranspose(ElastodynamicsOperator &oper,
                                  const Vector &x, real_t t,
                                  const Vector &eta, Vector &jt_eta)
{
   jt_eta.SetSize(x.Size());
   oper.SetTime(t);
   oper.JacobianMultTranspose(x, eta, jt_eta);
}

inline void RK4Stages(ElastodynamicsOperator &oper,
                      const Vector &x0, real_t t0, real_t h,
                      Vector &k1, Vector &k2, Vector &k3, Vector &k4,
                      Vector &y1, Vector &y2, Vector &y3)
{
   EvalRHS(oper, x0, t0, k1);

   y1 = x0;
   y1.Add(0.5*h, k1);
   EvalRHS(oper, y1, t0 + 0.5*h, k2);

   y2 = x0;
   y2.Add(0.5*h, k2);
   EvalRHS(oper, y2, t0 + 0.5*h, k3);

   y3 = x0;
   y3.Add(h, k3);
   EvalRHS(oper, y3, t0 + h, k4);
}

inline void RK4AdjointOneStep(ElastodynamicsOperator &oper,
                              const Vector &x0, real_t t0, real_t h,
                              const Vector &lambda_next,
                              Vector &lambda_prev)
{
   const int n = x0.Size();
   Vector k1(n), k2(n), k3(n), k4(n);
   Vector y1(n), y2(n), y3(n);
   RK4Stages(oper, x0, t0, h, k1, k2, k3, k4, y1, y2, y3);

   Vector adj_x0(lambda_next);
   Vector adj_k1(n), adj_k2(n), adj_k3(n), adj_k4(n);
   Vector adj_y(n), jt(n);

   adj_k1.Set(h/6.0, lambda_next);
   adj_k2.Set(h/3.0, lambda_next);
   adj_k3.Set(h/3.0, lambda_next);
   adj_k4.Set(h/6.0, lambda_next);

   EvalJacobianTranspose(oper, y3, t0 + h, adj_k4, adj_y);
   adj_x0.Add(1.0, adj_y);
   adj_k3.Add(h, adj_y);

   EvalJacobianTranspose(oper, y2, t0 + 0.5*h, adj_k3, adj_y);
   adj_x0.Add(1.0, adj_y);
   adj_k2.Add(0.5*h, adj_y);

   EvalJacobianTranspose(oper, y1, t0 + 0.5*h, adj_k2, adj_y);
   adj_x0.Add(1.0, adj_y);
   adj_k1.Add(0.5*h, adj_y);

   EvalJacobianTranspose(oper, x0, t0, adj_k1, jt);
   adj_x0.Add(1.0, jt);

   lambda_prev = adj_x0;
}

inline void AddStageDesignGradientTilde(ElastodynamicsOperator &oper,
                                        ParFiniteElementSpace &state_fes,
                                        ParFiniteElementSpace &filter_fes,
                                        ParGridFunction &rho_tilde,
                                        const MaterialParams &mat,
                                        const Vector &stage_state,
                                        const Vector &stage_rhs,
                                        const Vector &stage_seed,
                                        Vector &dJ_drho_tilde)
{
   const Array<int> &offsets = oper.GetBlockOffsets();
   BlockVector state_blocks(const_cast<Vector&>(stage_state), offsets);
   BlockVector rhs_blocks(const_cast<Vector&>(stage_rhs), offsets);
   BlockVector seed_blocks(const_cast<Vector&>(stage_seed), offsets);

   Vector z_true;
   oper.MultInvMass(seed_blocks.GetBlock(1), z_true);

   ParGridFunction u_gf(&state_fes);
   ParGridFunction accel_gf(&state_fes);
   ParGridFunction z_gf(&state_fes);

   u_gf.SetFromTrueDofs(state_blocks.GetBlock(0));
   accel_gf.SetFromTrueDofs(rhs_blocks.GetBlock(1));
   z_gf.SetFromTrueDofs(z_true);

   const bool lumped = (oper.GetMassSolverType() == MassSolverType::LUMPED);
   ParLinearForm mass_lf(&filter_fes);
   mass_lf.AddDomainIntegrator(
      new StageMassDesignLFIntegrator(rho_tilde, accel_gf, z_gf, mat, lumped));
   mass_lf.Assemble();
   std::unique_ptr<HypreParVector> mass_vec(mass_lf.ParallelAssemble());

   ParLinearForm stiffness_lf(&filter_fes);
   stiffness_lf.AddDomainIntegrator(
      new StageStiffnessDesignLFIntegrator(rho_tilde, u_gf, z_gf, mat));
   stiffness_lf.Assemble();
   std::unique_ptr<HypreParVector> stiffness_vec(stiffness_lf.ParallelAssemble());

   dJ_drho_tilde.Add(1.0, *mass_vec);
   dJ_drho_tilde.Add(1.0, *stiffness_vec);
}

inline void RK4AdjointOneStepWithDesign(ElastodynamicsOperator &oper,
                                        ParFiniteElementSpace &state_fes,
                                        ParFiniteElementSpace &filter_fes,
                                        ParGridFunction &rho_tilde,
                                        const MaterialParams &mat,
                                        const Vector &x0, real_t t0, real_t h,
                                        const Vector &lambda_next,
                                        Vector &lambda_prev,
                                        Vector &dJ_drho_tilde)
{
   const int n = x0.Size();
   Vector k1(n), k2(n), k3(n), k4(n);
   Vector y1(n), y2(n), y3(n);
   RK4Stages(oper, x0, t0, h, k1, k2, k3, k4, y1, y2, y3);

   Vector adj_x0(lambda_next);
   Vector adj_k1(n), adj_k2(n), adj_k3(n), adj_k4(n);
   Vector adj_y(n), jt(n);

   adj_k1.Set(h/6.0, lambda_next);
   adj_k2.Set(h/3.0, lambda_next);
   adj_k3.Set(h/3.0, lambda_next);
   adj_k4.Set(h/6.0, lambda_next);

   AddStageDesignGradientTilde(oper, state_fes, filter_fes, rho_tilde, mat,
                               y3, k4, adj_k4, dJ_drho_tilde);
   EvalJacobianTranspose(oper, y3, t0 + h, adj_k4, adj_y);
   adj_x0.Add(1.0, adj_y);
   adj_k3.Add(h, adj_y);

   AddStageDesignGradientTilde(oper, state_fes, filter_fes, rho_tilde, mat,
                               y2, k3, adj_k3, dJ_drho_tilde);
   EvalJacobianTranspose(oper, y2, t0 + 0.5*h, adj_k3, adj_y);
   adj_x0.Add(1.0, adj_y);
   adj_k2.Add(0.5*h, adj_y);

   AddStageDesignGradientTilde(oper, state_fes, filter_fes, rho_tilde, mat,
                               y1, k2, adj_k2, dJ_drho_tilde);
   EvalJacobianTranspose(oper, y1, t0 + 0.5*h, adj_k2, adj_y);
   adj_x0.Add(1.0, adj_y);
   adj_k1.Add(0.5*h, adj_y);

   AddStageDesignGradientTilde(oper, state_fes, filter_fes, rho_tilde, mat,
                               x0, k1, adj_k1, dJ_drho_tilde);
   EvalJacobianTranspose(oper, x0, t0, adj_k1, jt);
   adj_x0.Add(1.0, jt);

   lambda_prev = adj_x0;
}

inline real_t AddObjectiveContribution(ParFiniteElementSpace &state_fes,
                                       const Array<int> &offsets,
                                       TimeIntegratedObjective &objective,
                                       const Vector &state,
                                       real_t dt, int step, int total_steps)
{
   BlockVector bstate(const_cast<Vector&>(state), offsets);
   ParGridFunction u_gf(&state_fes);
   u_gf.SetFromTrueDofs(bstate.GetBlock(0));
   return objective.AccumulateTimestep(u_gf, dt, step, total_steps);
}

inline void ObjectiveGradientAtState(ParFiniteElementSpace &state_fes,
                                     const Array<int> &offsets,
                                     TimeIntegratedObjective &objective,
                                     const Vector &state,
                                     real_t dt, int step, int total_steps,
                                     Vector &q_state)
{
   q_state.SetSize(state.Size());
   q_state = 0.0;

   BlockVector bstate(const_cast<Vector&>(state), offsets);
   BlockVector bq(q_state, offsets);

   ParGridFunction u_gf(&state_fes);
   u_gf.SetFromTrueDofs(bstate.GetBlock(0));

   ParLinearForm grad_form(&state_fes);
   objective.ComputeObjectiveGradient(u_gf, dt, step, total_steps, grad_form);

   std::unique_ptr<HypreParVector> q_u(grad_form.ParallelAssemble());
   bq.GetBlock(0) = *q_u;
   bq.GetBlock(1) = 0.0;
}

inline real_t RolloutObjective(ElastodynamicsOperator &oper,
                               ParFiniteElementSpace &state_fes,
                               const Array<int> &offsets,
                               TimeIntegratedObjective &objective,
                               const Vector &x_init,
                               int nsteps, real_t t_init, real_t h,
                               std::vector<Vector> *states,
                               std::vector<real_t> *times,
                               const char *progress_label = nullptr)
{
   const int n = x_init.Size();
   Vector x(x_init);
   real_t t = t_init;
   const int total_steps = nsteps + 1;

   // Progress monitoring (rank 0 only, throttled to ~10 lines per sweep).
   const bool report = (progress_label != nullptr) && (Mpi::WorldRank() == 0);
   const double phase_t0 = MPI_Wtime();
   const int report_every = std::max(1, nsteps / 10);

   objective.Reset();

   RK4Solver solver;
   solver.Init(oper);

   if (states)
   {
      states->resize(nsteps + 1);
      for (int i = 0; i <= nsteps; i++) { (*states)[i].SetSize(n); }
      (*states)[0] = x;
   }
   if (times)
   {
      times->assign(nsteps + 1, 0.0);
      (*times)[0] = t;
   }

   AddObjectiveContribution(state_fes, offsets, objective, x, h, 0,
                            total_steps);

   for (int i = 0; i < nsteps; i++)
   {
      real_t dt = h;
      solver.Step(x, t, dt);

      AddObjectiveContribution(state_fes, offsets, objective, x, h, i + 1,
                               total_steps);

      if (states) { (*states)[i + 1] = x; }
      if (times)  { (*times)[i + 1] = t; }

      if (report && ((i + 1) % report_every == 0 || i + 1 == nsteps))
      {
         std::cout << "      " << progress_label << ' '
                   << std::setw(6) << (i + 1) << '/' << nsteps
                   << "  (" << std::setw(3) << (100 * (i + 1) / nsteps) << "%)"
                   << "   " << std::fixed << std::setprecision(2)
                   << (MPI_Wtime() - phase_t0) << " s\n";
      }
   }

   return objective.GetObjective();
}

inline real_t EvaluateDesignObjective(const Vector &rho_tv,
                                      const Vector &x0,
                                      ParFiniteElementSpace &state_fes,
                                      ParFiniteElementSpace &control_fes,
                                      ParGridFunction &rho,
                                      ParGridFunction &rho_tilde,
                                      toopt::PDEFilter &filter,
                                      Coefficient &gamma_coef,
                                      Array<int> &exterior_bdr_attr,
                                      Array<int> &empty_bdr_attr,
                                      TimeIntegratedObjective &objective,
                                      const MaterialParams &mat,
                                      const BoundaryLoadSpec &load_spec,
                                      VectorCoefficient &load_coef,
                                      real_t impedance,
                                      int nsteps,
                                      real_t h,
                                      MassSolverType mass_type = MassSolverType::LUMPED)
{
   rho.SetFromTrueDofs(rho_tv);
   filter.Mult(rho, rho_tilde);

   ConstantCoefficient rho_0_coef(mat.rho0);
   ConstantCoefficient lambda_0_coef(mat.lambda0);
   ConstantCoefficient mu_0_coef(mat.mu0);

   SIMPCoefficient simp_mass(&rho_tilde, mat.r_min, mat.r_max, mat.simp_p);
   SIMPCoefficient simp_stiff(&rho_tilde, mat.r_min, mat.r_max, mat.simp_p);

   ProductCoefficient mass_coef(simp_mass, rho_0_coef);
   ProductCoefficient lambda_coef(simp_stiff, lambda_0_coef);
   ProductCoefficient mu_coef(simp_stiff, mu_0_coef);

   ElastodynamicsOperator oper(
      state_fes, mass_coef, lambda_coef, mu_coef,
      load_spec.amplitude, load_spec.duration, load_spec.time_profile,
      load_spec.phase, load_spec.frequency, load_spec.bdr_attributes, load_coef,
      load_spec.domain_load,
      &gamma_coef, impedance, exterior_bdr_attr, empty_bdr_attr,
      mass_type);

   (void)control_fes;
   return RolloutObjective(oper, state_fes, oper.GetBlockOffsets(), objective,
                           x0, nsteps, 0.0, h, nullptr, nullptr);
}

// Backward discrete-adjoint sweep (the "adjoint physics solve"). Given the stored
// forward trajectory, marches the RK4 adjoint from step nsteps down to 0,
// accumulating the design gradient dJ/d(rho_tilde) in filter space (via
// RK4AdjointOneStepWithDesign) plus the per-step objective seed. Companion to the
// forward RolloutObjective; both are the building blocks of the gradient below and
// of TransientDesignSolver's PhysicsFSolve / PhysicsASolve.
inline void AdjointDesignSweep(ElastodynamicsOperator &oper,
                               ParFiniteElementSpace &state_fes,
                               ParFiniteElementSpace &filter_fes,
                               ParGridFunction &rho_tilde,
                               const MaterialParams &mat,
                               TimeIntegratedObjective &objective,
                               const std::vector<Vector> &states,
                               const std::vector<real_t> &times,
                               int nsteps, real_t h,
                               Vector &dJ_drho_tilde,
                               int outer_it = -1)
{
   const int myid = Mpi::WorldRank();
   const int n = states[0].Size();
   const int total_steps = nsteps + 1;

   dJ_drho_tilde.SetSize(filter_fes.GetTrueVSize());
   dJ_drho_tilde = 0.0;

   Vector q(n), lambda(n), lambda_prev(n);
   ObjectiveGradientAtState(state_fes, oper.GetBlockOffsets(), objective,
                            states[nsteps], h, nsteps, total_steps, lambda);

   if (myid == 0)
   {
      std::cout << "    [it " << outer_it + 1 << "] adjoint sweep ("
                << nsteps << " steps)\n";
   }
   const double adj_t0 = MPI_Wtime();
   const int adj_report_every = std::max(1, nsteps / 10);

   for (int i = nsteps - 1; i >= 0; i--)
   {
      const real_t hi = times[i + 1] - times[i];
      RK4AdjointOneStepWithDesign(oper, state_fes, filter_fes, rho_tilde,
                                  mat, states[i], times[i], hi,
                                  lambda, lambda_prev, dJ_drho_tilde);

      ObjectiveGradientAtState(state_fes, oper.GetBlockOffsets(), objective,
                               states[i], h, i, total_steps, q);
      lambda = lambda_prev;
      lambda += q;

      const int done = nsteps - i;
      if (myid == 0 && (done % adj_report_every == 0 || done == nsteps))
      {
         std::cout << "      adjoint " << std::setw(6) << done << '/' << nsteps
                   << "  (" << std::setw(3) << (100 * done / nsteps) << "%)"
                   << "   " << std::fixed << std::setprecision(2)
                   << (MPI_Wtime() - adj_t0) << " s\n";
      }
   }
}

inline real_t DesignObjectiveAdjointGradient(const Vector &rho_tv,
                                             const Vector &x0,
                                             ParFiniteElementSpace &state_fes,
                                             ParFiniteElementSpace &filter_fes,
                                             ParFiniteElementSpace &control_fes,
                                             MassSolverType mass_type,
                                             ParGridFunction &rho,
                                             ParGridFunction &rho_tilde,
                                             toopt::PDEFilter &filter,
                                             Coefficient &gamma_coef,
                                             Array<int> &exterior_bdr_attr,
                                             Array<int> &empty_bdr_attr,
                                             TimeIntegratedObjective &objective,
                                             const MaterialParams &mat,
                                             const BoundaryLoadSpec &load_spec,
                                             VectorCoefficient &load_coef,
                                             real_t impedance,
                                             int nsteps,
                                             real_t h,
                                             Vector &dJ_drho,
                                             int outer_it = -1)
{
   const int myid = Mpi::WorldRank();

   rho.SetFromTrueDofs(rho_tv);
   filter.Mult(rho, rho_tilde);

   ConstantCoefficient rho_0_coef(mat.rho0);
   ConstantCoefficient lambda_0_coef(mat.lambda0);
   ConstantCoefficient mu_0_coef(mat.mu0);

   SIMPCoefficient simp_mass(&rho_tilde, mat.r_min, mat.r_max, mat.simp_p);
   SIMPCoefficient simp_stiff(&rho_tilde, mat.r_min, mat.r_max, mat.simp_p);

   ProductCoefficient mass_coef(simp_mass, rho_0_coef);
   ProductCoefficient lambda_coef(simp_stiff, lambda_0_coef);
   ProductCoefficient mu_coef(simp_stiff, mu_0_coef);

   ElastodynamicsOperator oper(
      state_fes, mass_coef, lambda_coef, mu_coef,
      load_spec.amplitude, load_spec.duration, load_spec.time_profile,
      load_spec.phase, load_spec.frequency, load_spec.bdr_attributes, load_coef,
      load_spec.domain_load,
      &gamma_coef, impedance, exterior_bdr_attr, empty_bdr_attr,
      mass_type);

   if (myid == 0)
   {
      std::cout << "    [it " << outer_it + 1 << "] forward sweep ("
                << nsteps << " steps)\n";
   }

   std::vector<Vector> states;
   std::vector<real_t> times;
   const real_t J = RolloutObjective(oper, state_fes, oper.GetBlockOffsets(),
                                     objective, x0, nsteps, 0.0, h,
                                     &states, &times, "forward");

   Vector dJ_drho_tilde;
   AdjointDesignSweep(oper, state_fes, filter_fes, rho_tilde, mat, objective,
                      states, times, nsteps, h, dJ_drho_tilde, outer_it);

   filter.MultTranspose(dJ_drho_tilde, dJ_drho);
   MFEM_VERIFY(dJ_drho.Size() == control_fes.GetTrueVSize(),
               "Raw design gradient has unexpected size.");

   return J;
}

// =============================================================================
// TRANSIENT DESIGN SOLVER
// =============================================================================
// Bundles the invariant per-run setup (spaces, filter, damping, BC markers,
// load, objective, material, mass solver, rest initial state) and exposes the
// four canonical topology-optimization steps as arg-free calls, so the optimizer
// loop reads like the textbook template regardless of physics/filter:
//
//   FilterFSolve  : forward filter,  rho -> rho_tilde        (Helmholtz solve)
//   PhysicsFSolve : forward physics, RK4 sweep -> J          (stores trajectory)
//   PhysicsASolve : adjoint physics, backward sweep -> dJ/d(rho_tilde)
//   FilterASolve  : adjoint filter,  dJ/d(rho_tilde) -> dJ/d(rho)  (filter^T)
//
// Stateful by design: PhysicsFSolve builds the operator and stores the forward
// trajectory that PhysicsASolve consumes (call order Filter/PhysicsFSolve before
// Physics/FilterASolve). The per-step logic delegates to the verified
// RolloutObjective / AdjointDesignSweep primitives that the Taylor tests exercise.
class TransientDesignSolver
{
private:
   ParFiniteElementSpace &state_fes_;
   ParFiniteElementSpace &filter_fes_;
   ParFiniteElementSpace &control_fes_;
   toopt::PDEFilter &filter_;
   Coefficient &gamma_coef_;
   Array<int> &exterior_bdr_attr_;
   Array<int> &ess_bdr_attr_;
   TimeIntegratedObjective &objective_;
   const MaterialParams &mat_;
   const BoundaryLoadSpec &load_spec_;
   VectorCoefficient &load_coef_;
   real_t impedance_;
   int nsteps_;
   real_t h_;
   MassSolverType mass_type_;
   ParGridFunction &rho_;         // working density (also the driver's ParaView field)
   ParGridFunction &rho_tilde_;   // filtered density
   Vector x0_;                    // rest initial state [u, v] = 0

   // Design-dependent SIMP material coefficients; built once, they evaluate the
   // live rho_tilde_, so the operator re-assembled each PhysicsFSolve picks up the
   // current design. (Declared after rho_tilde_ / mat_ that they reference.)
   ConstantCoefficient rho0_coef_, lambda0_coef_, mu0_coef_;
   SIMPCoefficient simp_mass_, simp_stiff_;
   ProductCoefficient mass_coef_, lambda_coef_, mu_coef_;

   // Per-iteration forward state produced by PhysicsFSolve, consumed by the
   // adjoint steps. (Declared after the coefficients it references.)
   std::unique_ptr<ElastodynamicsOperator> oper_;
   std::vector<Vector> states_;
   std::vector<real_t> times_;
   Vector dJ_drho_tilde_;
   int outer_it_ = -1;
   bool banner_printed_ = false;   // operator banner prints once, not every iter

public:
   TransientDesignSolver(ParFiniteElementSpace &state_fes,
                         ParFiniteElementSpace &filter_fes,
                         ParFiniteElementSpace &control_fes,
                         toopt::PDEFilter &filter,
                         Coefficient &gamma_coef,
                         Array<int> &exterior_bdr_attr,
                         Array<int> &ess_bdr_attr,
                         TimeIntegratedObjective &objective,
                         const MaterialParams &mat,
                         const BoundaryLoadSpec &load_spec,
                         VectorCoefficient &load_coef,
                         real_t impedance,
                         int nsteps, real_t h,
                         MassSolverType mass_type,
                         ParGridFunction &rho,
                         ParGridFunction &rho_tilde)
      : state_fes_(state_fes), filter_fes_(filter_fes), control_fes_(control_fes),
        filter_(filter), gamma_coef_(gamma_coef),
        exterior_bdr_attr_(exterior_bdr_attr), ess_bdr_attr_(ess_bdr_attr),
        objective_(objective), mat_(mat), load_spec_(load_spec),
        load_coef_(load_coef), impedance_(impedance),
        nsteps_(nsteps), h_(h), mass_type_(mass_type),
        rho_(rho), rho_tilde_(rho_tilde),
        x0_(2 * state_fes.GetTrueVSize()),
        rho0_coef_(mat.rho0), lambda0_coef_(mat.lambda0), mu0_coef_(mat.mu0),
        simp_mass_(&rho_tilde_, mat.r_min, mat.r_max, mat.simp_p),
        simp_stiff_(&rho_tilde_, mat.r_min, mat.r_max, mat.simp_p),
        mass_coef_(simp_mass_, rho0_coef_),
        lambda_coef_(simp_stiff_, lambda0_coef_),
        mu_coef_(simp_stiff_, mu0_coef_)
   {
      x0_ = 0.0;
   }

   int NumSteps() const { return nsteps_; }
   real_t TimeStep() const { return h_; }

   // 1. Forward filter: raw control density -> filtered density (Helmholtz solve).
   void FilterFSolve(const Vector &rho_tv)
   {
      rho_.SetFromTrueDofs(rho_tv);
      filter_.Mult(rho_, rho_tilde_);
   }

   // 2. Forward physics: (re)assemble the operator for the current rho_tilde_, run
   //    the RK4 forward sweep, store the trajectory, return J.
   real_t PhysicsFSolve(int outer_it = -1)
   {
      outer_it_ = outer_it;
      oper_ = std::make_unique<ElastodynamicsOperator>(
                 state_fes_, mass_coef_, lambda_coef_, mu_coef_,
                 load_spec_.amplitude, load_spec_.duration, load_spec_.time_profile,
                 load_spec_.phase, load_spec_.frequency, load_spec_.bdr_attributes,
                 load_coef_, load_spec_.domain_load, &gamma_coef_, impedance_,
                 exterior_bdr_attr_, ess_bdr_attr_, mass_type_,
                 /*print_banner=*/!banner_printed_);
      banner_printed_ = true;

      if (Mpi::Root())
      {
         std::cout << "    [it " << outer_it_ + 1 << "] forward sweep ("
                   << nsteps_ << " steps)\n";
      }
      return RolloutObjective(*oper_, state_fes_, oper_->GetBlockOffsets(),
                              objective_, x0_, nsteps_, 0.0, h_,
                              &states_, &times_, "forward");
   }

   // 3. Adjoint physics: backward discrete-adjoint sweep -> dJ/d(rho_tilde).
   void PhysicsASolve()
   {
      MFEM_VERIFY(oper_, "PhysicsASolve() requires a preceding PhysicsFSolve().");
      AdjointDesignSweep(*oper_, state_fes_, filter_fes_, rho_tilde_, mat_,
                         objective_, states_, times_, nsteps_, h_,
                         dJ_drho_tilde_, outer_it_);
   }

   // 4. Adjoint filter: transpose the filter, dJ/d(rho_tilde) -> dJ/d(rho).
   void FilterASolve(Vector &dJ_drho)
   {
      filter_.MultTranspose(dJ_drho_tilde_, dJ_drho);
      MFEM_VERIFY(dJ_drho.Size() == control_fes_.GetTrueVSize(),
                  "Raw design gradient has unexpected size.");
   }

   // Convenience: the four steps in sequence (forward filter + physics, adjoint
   // physics + filter). Returns J and fills dJ_drho.
   real_t ObjectiveAndGradient(const Vector &rho_tv, Vector &dJ_drho,
                               int outer_it = -1)
   {
      FilterFSolve(rho_tv);
      const real_t J = PhysicsFSolve(outer_it);
      PhysicsASolve();
      FilterASolve(dJ_drho);
      return J;
   }

   // Forward-only objective J(rho) (no gradient / no stored trajectory).
   real_t Objective(const Vector &rho_tv)
   {
      return EvaluateDesignObjective(
                rho_tv, x0_, state_fes_, control_fes_, rho_, rho_tilde_, filter_,
                gamma_coef_, exterior_bdr_attr_, ess_bdr_attr_, objective_, mat_,
                load_spec_, load_coef_, impedance_, nsteps_, h_, mass_type_);
   }
};

} // namespace mfem

#endif // ELASTODYNAMICS_SOLVER_HPP
