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
#include "ObjectiveFunctional.hpp"
#include "../../pde_filter.hpp"
#include <memory>
#include <vector>

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

// =============================================================================
// DAMPING PROFILE FOR SPONGE LAYERS
// =============================================================================
class DampingProfile : public Coefficient
{
private:
   real_t thickness;
   real_t x_max, y_max;
   real_t phi_max;

public:
   DampingProfile(real_t thick, real_t xmax, real_t ymax)
      : thickness(thick), x_max(xmax), y_max(ymax)
   {
      phi_max = thickness * thickness / 2.0;
   }

   real_t GetPhiMax() const { return phi_max; }

   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      Vector x(2);
      T.Transform(ip, x);

      real_t phi = 0.0;
      real_t s = 0.0;

      // Left boundary layer
      if (x(0) < thickness)
      {
         s = thickness - x(0);
         real_t phi_local = thickness * s - 0.5 * s * s;
         phi = std::max(phi, phi_local);
      }

      // Right boundary layer
      if (x(0) > x_max - thickness)
      {
         s = x(0) - (x_max - thickness);
         real_t phi_local = thickness * s - 0.5 * s * s;
         phi = std::max(phi, phi_local);
      }

      // Bottom boundary layer
      if (x(1) < thickness)
      {
         s = thickness - x(1);
         real_t phi_local = thickness * s - 0.5 * s * s;
         phi = std::max(phi, phi_local);
      }

      return phi;
   }
};

// =============================================================================
// SPATIALLY-VARYING DAMPING COEFFICIENT
// =============================================================================
class SpatialDampingCoefficient : public Coefficient
{
private:
   DampingProfile *phi_coef;
   real_t phi_max;
   real_t gamma_max;
   real_t rho;
   real_t beta;
   int m;

public:
   SpatialDampingCoefficient(DampingProfile *phi, real_t gmax,
                              real_t density, real_t b = 2.0, int mp = 2)
      : phi_coef(phi), gamma_max(gmax), rho(density), beta(b), m(mp)
   {
      phi_max = phi_coef->GetPhiMax();
   }

   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      real_t phi_val = phi_coef->Eval(T, ip);

      if (phi_max < 1e-12) return 0.0;

      real_t eta = phi_val / phi_max;
      eta = std::min(std::max(eta, 0.0), 1.0);

      real_t eta_pow = std::pow(eta, m);
      real_t F_eta = (std::exp(beta * eta_pow) - 1.0) / (std::exp(beta) - 1.0);

      return rho * gamma_max * F_eta;
   }
};

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
   HypreBoomerAMG *M_prec;
   CGSolver *M_solver;

   mutable ParGridFunction u_gf;  // For visualization
   mutable ParGridFunction v_gf;  // For visualization

   Array<int> ess_tdof_list;
   Array<int> block_true_offsets;
   int true_size;

   mutable Vector res;
   mutable Vector tmp;

   mutable Vector load_params;    // [duration, amplitude]
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
      SpatialDampingCoefficient *gamma_coef,
      real_t impedance,
      Array<int> &exterior_bdr_attr,
      Array<int> &ess_bdr_attr,
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

   HypreParMatrix* GetMassMatrix() const { return Mmat; }
   HypreParMatrix* GetStiffnessMatrix() const { return Kmat; }
   HypreParMatrix* GetVolDampingMatrix() const { return Cvol_mat; }
   HypreParMatrix* GetAbsDampingMatrix() const { return Cabs_mat; }

   void MultInvMass(const Vector &rhs, Vector &sol) const
   {
      sol.SetSize(true_size);
      sol = 0.0;
      M_solver->Mult(rhs, sol);
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
   SpatialDampingCoefficient *gamma_coef,
   real_t impedance,
   Array<int> &exterior_bdr_attr,
   Array<int> &ess_bdr_attr,
   ForwardTrajectoryStorage *traj,
   TimeIntegratedObjective *obj)
   : TimeDependentOperator(2 * f.GetTrueVSize(), 0.0),
     fespace(f),
     u_gf(&fespace),
     v_gf(&fespace),
     true_size(f.GetTrueVSize()),
     res(true_size),
     tmp(true_size),
     load_params(2),
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

   load_params(0) = duration;
   load_params(1) = amplitude;

   fespace.GetEssentialTrueDofs(ess_bdr_attr, ess_tdof_list);

   if (myid == 0)
   {
      std::cout << "\n=== Elastodynamics Operator ===" << std::endl;
      std::cout << "DOFs per field: " << true_size << std::endl;
      std::cout << "Essential DOFs: " << ess_tdof_list.Size() << std::endl;
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

   // Set up mass matrix solver
   M_prec = new HypreBoomerAMG(*Mmat);
   M_prec->SetPrintLevel(0);

   M_solver = new CGSolver(fespace.GetComm());
   M_solver->SetPreconditioner(*M_prec);
   M_solver->SetOperator(*Mmat);
   M_solver->SetRelTol(1e-12);
   M_solver->SetAbsTol(0.0);
   M_solver->SetMaxIter(100);
   M_solver->SetPrintLevel(0);

   // Set up boundary markers for loading
   ParMesh *pmesh = fespace.GetParMesh();
   int max_bdr_attr = pmesh->bdr_attributes.Max();
   load_bdr_markers.SetSize(max_bdr_attr);
   load_bdr_markers = 0;

   // Mark load boundaries (attributes 21-26 for Gmsh mesh)
   for (int attr = 21; attr <= 26; attr++)
   {
      if (attr <= max_bdr_attr)
      {
         load_bdr_markers[attr - 1] = 1;
      }
   }

   if (myid == 0)
   {
      std::cout << "\nTime-dependent loading:" << std::endl;
      std::cout << "  Type: Smooth Gaussian pulse" << std::endl;
      std::cout << "  Peak amplitude: " << amplitude << std::endl;
      std::cout << "  Duration: " << duration << " s" << std::endl;
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

   // Time-dependent applied load
   real_t duration = load_params(0);
   real_t amplitude = load_params(1);

   real_t t_center = duration / 2.0;
   real_t sigma = duration / 4.0;
   real_t t_diff = time - t_center;
   real_t gauss_factor = exp(-t_diff * t_diff / (2.0 * sigma * sigma));
   real_t current_amplitude = amplitude * gauss_factor;

   class GaussianLoad : public VectorCoefficient
   {
   private:
      real_t amp;
   public:
      GaussianLoad(int dim, real_t a) : VectorCoefficient(dim), amp(a) {}
      void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip) override
      {
         V.SetSize(vdim);
         V = 0.0;
         V(1) = -amp;
      }
   };

   GaussianLoad load_coef(fespace.GetParMesh()->SpaceDimension(), current_amplitude);

   ParLinearForm load_form(&fespace);
   load_form.AddBoundaryIntegrator(
      new VectorBoundaryLFIntegrator(load_coef),
      const_cast<Array<int>&>(load_bdr_markers));
   load_form.Assemble();

   Vector load_vec(true_size);
   load_form.ParallelAssemble(load_vec);

   res.Add(1.0, load_vec);

   // Solve M v̇ = res
   real_t local_res_norm_sq = res * res;
   real_t global_res_norm_sq;
   MPI_Allreduce(&local_res_norm_sq, &global_res_norm_sq, 1,
                 MPI_DOUBLE, MPI_SUM, fespace.GetComm());
   real_t global_res_norm = sqrt(global_res_norm_sq);

   if (global_res_norm < 1e-14)
   {
      by.GetBlock(1) = 0.0;
   }
   else
   {
      by.GetBlock(1) = 0.0;
      M_solver->Mult(res, by.GetBlock(1));
   }
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

   BlockVector b_eta_new(const_cast<Vector&>(eta), block_true_offsets);
   BlockVector b_eta_rhs_new(eta_rhs, block_true_offsets);

   Vector mu_new(b_eta_new.GetBlock(0).GetData(), true_size);
   Vector lambda_new(b_eta_new.GetBlock(1).GetData(), true_size);

   Vector m_inv_lambda(true_size);
   m_inv_lambda = 0.0;
   M_solver->Mult(lambda_new, m_inv_lambda);

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

struct MaterialParams
{
   real_t rho0 = 1.0;
   real_t lambda0 = 2.0;
   real_t mu0 = 1.0;
   real_t r_min = 1e-6;
   real_t r_max = 1.0;
   real_t simp_p = 3.0;
};

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

class StageMassDesignLFIntegrator : public LinearFormIntegrator
{
private:
   ParGridFunction &rho_tilde;
   ParGridFunction &accel;
   ParGridFunction &z;
   MaterialParams mat;
   Vector shape, accel_val, z_val;

public:
   StageMassDesignLFIntegrator(ParGridFunction &rho_tilde_,
                               ParGridFunction &accel_,
                               ParGridFunction &z_,
                               const MaterialParams &mat_)
      : rho_tilde(rho_tilde_), accel(accel_), z(z_), mat(mat_) {}

   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &T,
                               Vector &elvect) override
   {
      const int dof = el.GetDof();
      shape.SetSize(dof);
      elvect.SetSize(dof);
      elvect = 0.0;

      const int int_order = 2 * el.GetOrder() + T.OrderW();
      const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), int_order);

      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T.SetIntPoint(&ip);
         el.CalcPhysShape(T, shape);

         accel.GetVectorValue(T, ip, accel_val);
         z.GetVectorValue(T, ip, z_val);

         const real_t rp = SimpDerivative(rho_tilde, T, ip, mat);
         const real_t density = -mat.rho0 * rp * (accel_val * z_val);
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

   ParLinearForm mass_lf(&filter_fes);
   mass_lf.AddDomainIntegrator(
      new StageMassDesignLFIntegrator(rho_tilde, accel_gf, z_gf, mat));
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
                               std::vector<real_t> *times)
{
   const int n = x_init.Size();
   Vector x(x_init);
   real_t t = t_init;
   const int total_steps = nsteps + 1;

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
                                      SpatialDampingCoefficient &gamma_coef,
                                      Array<int> &exterior_bdr_attr,
                                      Array<int> &empty_bdr_attr,
                                      Coefficient &subdomain_indicator,
                                      const MaterialParams &mat,
                                      real_t pulse_amplitude,
                                      real_t pulse_duration,
                                      real_t impedance,
                                      int nsteps,
                                      real_t h)
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
      pulse_amplitude, pulse_duration,
      &gamma_coef, impedance, exterior_bdr_attr, empty_bdr_attr);

   TimeIntegratedObjective objective(&state_fes, &subdomain_indicator,
                                     state_fes.GetComm());

   (void)control_fes;
   return RolloutObjective(oper, state_fes, oper.GetBlockOffsets(), objective,
                           x0, nsteps, 0.0, h, nullptr, nullptr);
}

inline real_t DesignObjectiveAdjointGradient(const Vector &rho_tv,
                                             const Vector &x0,
                                             ParFiniteElementSpace &state_fes,
                                             ParFiniteElementSpace &filter_fes,
                                             ParFiniteElementSpace &control_fes,
                                             ParGridFunction &rho,
                                             ParGridFunction &rho_tilde,
                                             toopt::PDEFilter &filter,
                                             SpatialDampingCoefficient &gamma_coef,
                                             Array<int> &exterior_bdr_attr,
                                             Array<int> &empty_bdr_attr,
                                             Coefficient &subdomain_indicator,
                                             const MaterialParams &mat,
                                             real_t pulse_amplitude,
                                             real_t pulse_duration,
                                             real_t impedance,
                                             int nsteps,
                                             real_t h,
                                             Vector &dJ_drho)
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
      pulse_amplitude, pulse_duration,
      &gamma_coef, impedance, exterior_bdr_attr, empty_bdr_attr);

   TimeIntegratedObjective objective(&state_fes, &subdomain_indicator,
                                     state_fes.GetComm());

   std::vector<Vector> states;
   std::vector<real_t> times;
   const real_t J = RolloutObjective(oper, state_fes, oper.GetBlockOffsets(),
                                     objective, x0, nsteps, 0.0, h,
                                     &states, &times);

   const int n = x0.Size();
   const int total_steps = nsteps + 1;

   Vector dJ_drho_tilde(filter_fes.GetTrueVSize());
   dJ_drho_tilde = 0.0;

   Vector q(n), lambda(n), lambda_prev(n);
   ObjectiveGradientAtState(state_fes, oper.GetBlockOffsets(), objective,
                            states[nsteps], h, nsteps, total_steps, lambda);

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
   }

   filter.MultTranspose(dJ_drho_tilde, dJ_drho);
   MFEM_VERIFY(dJ_drho.Size() == control_fes.GetTrueVSize(),
               "Raw design gradient has unexpected size.");

   return J;
}

} // namespace mfem

#endif // ELASTODYNAMICS_SOLVER_HPP
