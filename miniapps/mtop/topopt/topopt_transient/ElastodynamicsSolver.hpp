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
#include "TrajectoryCheckpointing.hpp" // REVOLVE checkpointing
#include <memory>
#include <vector>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>

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

// DampingProfile / SpatialDampingCoefficient: promoted to ProblemSpecification.hpp
// (shared with the problem layer, which assembles them via DampingField).

// =============================================================================
// MASS SOLVER STRATEGIES
// =============================================================================
enum class MassSolverType
{
   // p=1: standard row-sum lumping. p>1 simplex: positive scaled-diagonal
   // lumping (standard nodal row sums can be zero or negative).
   LUMPED,
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
   bool use_scaled_diagonal_lumping;
   real_t lumped_diagonal_scale;

   // Iterative solver (CG + AMG) - only used if mass_solver_type == ITERATIVE
   HypreParMatrix *M_free_mat;
   HypreBoomerAMG *M_prec;
   CGSolver *M_solver;

   Array<int> ess_tdof_list;
   Array<int> block_true_offsets;
   int true_size;

   mutable Vector res;
   mutable Vector tmp;
   mutable Vector mass_rhs;

   // Precomputed base load vector: load(t) = load_base_vector * amplitude *
   // time_profile(t), assembled once so the inner loop never re-assembles.
   Vector load_base_vector;
   real_t load_duration, load_amplitude, load_phase, load_frequency;
   LoadTimeProfile load_time_profile;
   bool load_on_domain;   // true: body force over Omega; false: boundary traction
   Array<int> load_bdr_markers;

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
      bool print_banner = true);

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

   Array<int>& GetBlockOffsets() { return block_true_offsets; }

   MassSolverType GetMassSolverType() const { return mass_solver_type; }
   bool UsesScaledDiagonalLumping() const
   {
      return mass_solver_type == MassSolverType::LUMPED &&
             use_scaled_diagonal_lumping;
   }
   real_t GetLumpedDiagonalScale() const { return lumped_diagonal_scale; }

   HypreParMatrix* GetMassMatrix() const { return Mmat; }
   HypreParMatrix* GetStiffnessMatrix() const { return Kmat; }
   HypreParMatrix* GetVolDampingMatrix() const { return Cvol_mat; }
   HypreParMatrix* GetAbsDampingMatrix() const { return Cabs_mat; }
   const Vector &GetLoadBaseVector() const { return load_base_vector; }

   void MultInvMass(const Vector &rhs, Vector &sol) const
   {
      MFEM_VERIFY(rhs.Size() == true_size,
                  "Mass solve received an unexpected right-hand-side size.");
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
         // The consistent constrained inverse is
         //   P (P M P + I-P)^{-1} P,
         // where P zeros the essential true dofs.  M_free_mat has the
         // corresponding eliminated rows/columns; projecting both sides keeps
         // this routine correct even when it is called outside Mult().
         mass_rhs = rhs;
         ProjectEssentialField(mass_rhs);
         sol = 0.0;
         M_solver->Mult(mass_rhs, sol);
         MFEM_VERIFY(
            M_solver->GetConverged(),
            "Consistent mass CG failed to converge in "
            << M_solver->GetNumIterations() << " iterations (final norm "
            << M_solver->GetFinalNorm() << ").");
      }
      ProjectEssentialField(sol);
   }

   /// Apply the mass operator used by MultInvMass.  For lumped dynamics this is
   /// the diagonal M_L, not the assembled consistent matrix.
   void MultMass(const Vector &x, Vector &y) const
   {
      y.SetSize(true_size);
      if (mass_solver_type == MassSolverType::LUMPED)
      {
         for (int i = 0; i < true_size; i++)
         {
            y[i] = x[i] / M_lumped_inv[i];
         }
      }
      else
      {
         Mmat->Mult(x, y);
      }
   }

   /// Project a single displacement/force field onto the free true DOFs.
   void ProjectEssentialField(Vector &x) const
   {
      MFEM_VERIFY(x.Size() == true_size,
                  "Essential-field projection received an unexpected size.");
      for (int i = 0; i < ess_tdof_list.Size(); i++)
      {
         x[ess_tdof_list[i]] = 0.0;
      }
   }

   /// Estimate the undamped RK4 wave CFL limit for the current lumped operator.
   /// Power iteration is applied to the symmetric matrix
   /// M_L^{-1/2} K M_L^{-1/2}; dt*omega_max <= 2*sqrt(2) is the imaginary-axis
   /// stability condition for classical RK4.
   real_t EstimateLumpedRK4TimeStep(int power_iterations = 30) const
   {
      MFEM_VERIFY(mass_solver_type == MassSolverType::LUMPED,
                  "Lumped CFL estimate requires a diagonal mass.");
      MFEM_VERIFY(power_iterations > 0,
                  "CFL power iteration count must be positive.");

      const MPI_Comm comm = fespace.GetComm();
      int rank = 0;
      MPI_Comm_rank(comm, &rank);

      Vector v(true_size), scaled(true_size), w(true_size);
      for (int i = 0; i < true_size; i++)
      {
         // Deterministic non-constant seed with high-frequency content.
         v(i) = std::sin(real_t(0.61803398875) *
                         real_t(i + 1 + 104729 * rank));
      }
      for (int j = 0; j < ess_tdof_list.Size(); j++)
      {
         v(ess_tdof_list[j]) = 0.0;
      }

      const auto global_dot = [&](const Vector &a, const Vector &b)
      {
         const real_t local = a * b;
         real_t global = 0.0;
         MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type,
                       MPI_SUM, comm);
         return global;
      };

      real_t norm = std::sqrt(global_dot(v, v));
      MFEM_VERIFY(norm > 0.0, "CFL power iteration has a zero initial vector.");
      v /= norm;

      real_t lambda_max = 0.0;
      for (int it = 0; it < power_iterations; it++)
      {
         for (int i = 0; i < true_size; i++)
         {
            scaled(i) = std::sqrt(M_lumped_inv(i)) * v(i);
         }
         Kmat->Mult(scaled, w);
         for (int i = 0; i < true_size; i++)
         {
            w(i) *= std::sqrt(M_lumped_inv(i));
         }
         for (int j = 0; j < ess_tdof_list.Size(); j++)
         {
            w(ess_tdof_list[j]) = 0.0;
         }

         lambda_max = global_dot(v, w);
         norm = std::sqrt(global_dot(w, w));
         MFEM_VERIFY(norm > 0.0 && std::isfinite(norm),
                     "CFL power iteration failed.");
         v.Set(1.0 / norm, w);
      }

      // One final Rayleigh quotient at the normalized iterate.
      for (int i = 0; i < true_size; i++)
      {
         scaled(i) = std::sqrt(M_lumped_inv(i)) * v(i);
      }
      Kmat->Mult(scaled, w);
      for (int i = 0; i < true_size; i++)
      {
         w(i) *= std::sqrt(M_lumped_inv(i));
      }
      for (int j = 0; j < ess_tdof_list.Size(); j++)
      {
         w(ess_tdof_list[j]) = 0.0;
      }
      lambda_max = global_dot(v, w) / global_dot(v, v);
      MFEM_VERIFY(lambda_max > 0.0 && std::isfinite(lambda_max),
                  "CFL estimate produced an invalid maximum eigenvalue.");

      const real_t omega_max = std::sqrt(lambda_max);
      return 2.0 * std::sqrt(real_t(2.0)) / omega_max;
   }

   /// Estimate the negative-real-axis RK4 limit associated with the explicit
   /// volumetric and absorbing-boundary damping.  Power iteration is applied to
   /// M_L^{-1/2} (C_vol + C_abs) M_L^{-1/2}.  This is a separate necessary
   /// component of the timestep check: the undamped M^{-1}K wave estimate alone
   /// can be badly non-conservative for a strongly damped high-order boundary.
   real_t EstimateLumpedRK4DampingTimeStep(int power_iterations = 30) const
   {
      MFEM_VERIFY(mass_solver_type == MassSolverType::LUMPED,
                  "Lumped damping estimate requires a diagonal mass.");
      MFEM_VERIFY(power_iterations > 0,
                  "Damping power iteration count must be positive.");

      const MPI_Comm comm = fespace.GetComm();
      int rank = 0;
      MPI_Comm_rank(comm, &rank);

      Vector v(true_size), scaled(true_size), w(true_size), work(true_size);
      for (int i = 0; i < true_size; i++)
      {
         v(i) = std::sin(real_t(0.75487766625) *
                         real_t(i + 1 + 130363 * rank));
      }
      for (int j = 0; j < ess_tdof_list.Size(); j++)
      {
         v(ess_tdof_list[j]) = 0.0;
      }

      const auto global_dot = [&](const Vector &a, const Vector &b)
      {
         const real_t local = a * b;
         real_t global = 0.0;
         MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type,
                       MPI_SUM, comm);
         return global;
      };

      const auto apply_scaled_damping = [&](const Vector &input,
                                             Vector &output)
      {
         for (int i = 0; i < true_size; i++)
         {
            scaled(i) = std::sqrt(M_lumped_inv(i)) * input(i);
         }
         Cvol_mat->Mult(scaled, output);
         Cabs_mat->Mult(scaled, work);
         output += work;
         for (int i = 0; i < true_size; i++)
         {
            output(i) *= std::sqrt(M_lumped_inv(i));
         }
         for (int j = 0; j < ess_tdof_list.Size(); j++)
         {
            output(ess_tdof_list[j]) = 0.0;
         }
      };

      real_t norm = std::sqrt(global_dot(v, v));
      MFEM_VERIFY(norm > 0.0,
                  "Damping power iteration has a zero initial vector.");
      v /= norm;

      real_t damping_rate_max = 0.0;
      for (int it = 0; it < power_iterations; it++)
      {
         apply_scaled_damping(v, w);
         norm = std::sqrt(global_dot(w, w));
         if (norm <= std::numeric_limits<real_t>::epsilon())
         {
            return std::numeric_limits<real_t>::infinity();
         }
         MFEM_VERIFY(std::isfinite(norm),
                     "Damping power iteration failed.");
         damping_rate_max = global_dot(v, w);
         v.Set(1.0 / norm, w);
      }

      apply_scaled_damping(v, w);
      damping_rate_max = global_dot(v, w) / global_dot(v, v);
      if (damping_rate_max <=
          std::numeric_limits<real_t>::epsilon())
      {
         return std::numeric_limits<real_t>::infinity();
      }
      MFEM_VERIFY(std::isfinite(damping_rate_max),
                  "Damping estimate produced an invalid maximum rate.");

      // Extent of the classical RK4 stability interval on the negative real
      // axis (root of |R(-x)| = 1).
      constexpr real_t rk4_negative_real_extent = 2.7852935634052816;
      return rk4_negative_real_extent / damping_rate_max;
   }

   /// Estimate the undamped RK4 wave limit for the constrained consistent
   /// mass operator. Power iteration is applied to M_ff^{-1} K_ff and vectors
   /// are normalized in the M inner product. This is equivalent to iteration
   /// on the symmetric operator M_ff^{-1/2} K_ff M_ff^{-1/2} without forming a
   /// mass square root.
   real_t EstimateConsistentRK4TimeStep(int power_iterations = 30) const
   {
      MFEM_VERIFY(mass_solver_type == MassSolverType::ITERATIVE,
                  "Consistent CFL estimate requires the iterative mass.");
      MFEM_VERIFY(power_iterations > 0,
                  "CFL power iteration count must be positive.");

      const MPI_Comm comm = fespace.GetComm();
      int rank = 0;
      MPI_Comm_rank(comm, &rank);
      const auto global_dot = [&](const Vector &a, const Vector &b)
      {
         const real_t local = a * b;
         real_t global = 0.0;
         MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type,
                       MPI_SUM, comm);
         return global;
      };

      Vector v(true_size), stiffness_v(true_size), w(true_size),
             mass_work(true_size);
      for (int i = 0; i < true_size; i++)
      {
         v(i) = std::sin(real_t(0.61803398875) *
                         real_t(i + 1 + 104729 * rank));
      }
      ProjectEssentialField(v);

      const auto mass_norm = [&](const Vector &x)
      {
         MultMass(x, mass_work);
         const real_t norm_sq = global_dot(x, mass_work);
         MFEM_VERIFY(norm_sq > 0.0 && std::isfinite(norm_sq),
                     "Consistent CFL iteration has an invalid M norm.");
         return std::sqrt(norm_sq);
      };

      v /= mass_norm(v);
      for (int it = 0; it < power_iterations; it++)
      {
         Kmat->Mult(v, stiffness_v);
         ProjectEssentialField(stiffness_v);
         MultInvMass(stiffness_v, w);
         const real_t norm = mass_norm(w);
         v.Set(1.0 / norm, w);
      }

      Kmat->Mult(v, stiffness_v);
      MultMass(v, mass_work);
      const real_t denominator = global_dot(v, mass_work);
      const real_t lambda_max = global_dot(v, stiffness_v) / denominator;
      MFEM_VERIFY(lambda_max > 0.0 && std::isfinite(lambda_max),
                  "Consistent CFL estimate produced an invalid eigenvalue.");

      return 2.0 * std::sqrt(real_t(2.0)) / std::sqrt(lambda_max);
   }

   /// Estimate the negative-real-axis RK4 damping limit for the constrained
   /// consistent mass. Power iteration treats C_vol+C_abs as the symmetric
   /// operator in the generalized eigenproblem C phi = gamma M_ff phi.
   real_t EstimateConsistentRK4DampingTimeStep(
      int power_iterations = 30) const
   {
      MFEM_VERIFY(mass_solver_type == MassSolverType::ITERATIVE,
                  "Consistent damping estimate requires the iterative mass.");
      MFEM_VERIFY(power_iterations > 0,
                  "Damping power iteration count must be positive.");

      const MPI_Comm comm = fespace.GetComm();
      int rank = 0;
      MPI_Comm_rank(comm, &rank);
      const auto global_dot = [&](const Vector &a, const Vector &b)
      {
         const real_t local = a * b;
         real_t global = 0.0;
         MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type,
                       MPI_SUM, comm);
         return global;
      };

      Vector v(true_size), damping_v(true_size), damping_work(true_size),
             w(true_size), mass_work(true_size);
      for (int i = 0; i < true_size; i++)
      {
         v(i) = std::sin(real_t(0.75487766625) *
                         real_t(i + 1 + 130363 * rank));
      }
      ProjectEssentialField(v);

      const auto mass_norm = [&](const Vector &x)
      {
         MultMass(x, mass_work);
         const real_t norm_sq = global_dot(x, mass_work);
         MFEM_VERIFY(norm_sq > 0.0 && std::isfinite(norm_sq),
                     "Consistent damping iteration has an invalid M norm.");
         return std::sqrt(norm_sq);
      };
      const auto apply_damping = [&](const Vector &x, Vector &result)
      {
         Cvol_mat->Mult(x, result);
         Cabs_mat->Mult(x, damping_work);
         result += damping_work;
         ProjectEssentialField(result);
      };

      v /= mass_norm(v);
      for (int it = 0; it < power_iterations; it++)
      {
         apply_damping(v, damping_v);
         MultInvMass(damping_v, w);
         MultMass(w, mass_work);
         const real_t norm_sq = global_dot(w, mass_work);
         if (norm_sq <= std::numeric_limits<real_t>::epsilon())
         {
            return std::numeric_limits<real_t>::infinity();
         }
         MFEM_VERIFY(std::isfinite(norm_sq),
                     "Consistent damping power iteration failed.");
         v.Set(1.0 / std::sqrt(norm_sq), w);
      }

      apply_damping(v, damping_v);
      MultMass(v, mass_work);
      const real_t denominator = global_dot(v, mass_work);
      const real_t damping_rate_max = global_dot(v, damping_v) / denominator;
      if (damping_rate_max <= std::numeric_limits<real_t>::epsilon())
      {
         return std::numeric_limits<real_t>::infinity();
      }
      MFEM_VERIFY(std::isfinite(damping_rate_max),
                  "Consistent damping estimate produced an invalid rate.");

      constexpr real_t rk4_negative_real_extent = 2.7852935634052816;
      return rk4_negative_real_extent / damping_rate_max;
   }

   // Homogeneous Dirichlet (clamped) enforcement. u = v = 0 (hence u_dot = v_dot
   // = 0) on the essential dofs for all time, so we zero the essential entries of
   // both blocks of a full state / state-derivative vector z = [u, v]. Applied to
   // the forward RHS (Mult) and, symmetrically, to the adjoint input
   // (JacobianMultTranspose); the projection is symmetric, keeping the discrete
   // adjoint an exact transpose. The consistent path additionally uses the
   // row/column-eliminated M_free_mat in MultInvMass().
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
   bool print_banner)
   : TimeDependentOperator(2 * f.GetTrueVSize(), 0.0),
     fespace(f),
     mass_solver_type(mass_type),
     use_scaled_diagonal_lumping(false),
     lumped_diagonal_scale(1.0),
     M_free_mat(nullptr),
     M_prec(nullptr),
     M_solver(nullptr),
     true_size(f.GetTrueVSize()),
     res(true_size),
     tmp(true_size),
     mass_rhs(true_size),
     load_base_vector(true_size),
     load_duration(duration),
     load_amplitude(amplitude),
     load_phase(phase),
     load_frequency(frequency),
     load_time_profile(load_profile),
     load_on_domain(domain_load)
{
   int myid = Mpi::WorldRank();

   // Block structure: [displacement, velocity]
   block_true_offsets.SetSize(3);
   block_true_offsets[0] = 0;
   block_true_offsets[1] = true_size;
   block_true_offsets[2] = 2 * true_size;

   res = 0.0;
   tmp = 0.0;
   mass_rhs = 0.0;
   load_base_vector = 0.0;

   fespace.GetEssentialTrueDofs(ess_bdr_attr, ess_tdof_list);

   // Tensor-product H1/Gauss-Lobatto elements have positive nodal row-sum
   // quadrature weights at high order. Standard high-order simplex nodal
   // elements generally do not (e.g. p3 tetrahedra have zero edge weights),
   // so they use the positive scaled-diagonal construction below.
   int local_scaled_diagonal = 0;
   if (mass_solver_type == MassSolverType::LUMPED &&
       fespace.GetMaxElementOrder() > 1)
   {
      for (int e = 0; e < fespace.GetNE(); e++)
      {
         const Geometry::Type geom = fespace.GetFE(e)->GetGeomType();
         if (geom == Geometry::TRIANGLE || geom == Geometry::TETRAHEDRON)
         {
            local_scaled_diagonal = 1;
            break;
         }
      }
   }
   int global_scaled_diagonal = 0;
   MPI_Allreduce(&local_scaled_diagonal, &global_scaled_diagonal, 1,
                 MPI_INT, MPI_MAX, fespace.GetComm());
   use_scaled_diagonal_lumping = (global_scaled_diagonal != 0);

   // Report the GLOBAL essential-dof count (the local count on rank 0 is
   // partition-dependent and misleading - it can read 0 while the constraint is
   // active on other ranks).
   long long local_ess = ess_tdof_list.Size(), global_ess = 0;
   MPI_Allreduce(&local_ess, &global_ess, 1, MPI_LONG_LONG, MPI_SUM,
                 fespace.GetComm());

   if (myid == 0 && print_banner)
   {
      std::cout << "\n=== Elastodynamics Operator ===" << std::endl;
      std::cout << "DOFs per field: " << fespace.GlobalTrueVSize()
                << " global (" << true_size << " on rank 0)" << std::endl;
      std::cout << "Essential DOFs: " << global_ess << std::endl;
      std::cout << "Mass solver: ";
      if (mass_solver_type == MassSolverType::LUMPED)
      {
         std::cout << (use_scaled_diagonal_lumping ?
                      "LUMPED (positive scaled diagonal)" :
                      "LUMPED (row sum)");
      }
      else { std::cout << "ITERATIVE"; }
      std::cout << std::endl;
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
      M_lumped_inv.SetSize(true_size);

      if (use_scaled_diagonal_lumping)
      {
         // Standard nodal row sums on high-order triangles/tetrahedra can be
         // zero or negative. Use a positive scaled diagonal instead:
         //   M_L = s_p diag(M),
         // where s_p makes the total mass exact for a constant coefficient on
         // the reference element. The scale depends only on the state FE, not
         // on rho, so its design derivative remains local and exact.
         const int ne = fespace.GetNE();
         int local_bad = 0;
         real_t local_scale = 0.0;
         int local_has_scale = 0;
         if (ne > 0)
         {
            const FiniteElement *ref_el = fespace.GetFE(0);
            const Geometry::Type geom = ref_el->GetGeomType();
            const int order = ref_el->GetOrder();
            for (int e = 1; e < ne; e++)
            {
               const FiniteElement *el = fespace.GetFE(e);
               if (el->GetGeomType() != geom || el->GetOrder() != order)
               {
                  local_bad = 1;
               }
            }

            const IntegrationRule &ref_ir =
               IntRules.Get(geom, 2 * order);
            Vector ref_shape(ref_el->GetDof());
            real_t ref_volume = 0.0;
            real_t diagonal_mass = 0.0;
            for (int q = 0; q < ref_ir.GetNPoints(); q++)
            {
               const IntegrationPoint &ip = ref_ir.IntPoint(q);
               ref_el->CalcShape(ip, ref_shape);
               ref_volume += ip.weight;
               diagonal_mass += ip.weight * (ref_shape * ref_shape);
            }
            MFEM_VERIFY(ref_volume > 0.0 && diagonal_mass > 0.0,
                        "Invalid reference mass used for diagonal lumping.");
            local_scale = ref_volume / diagonal_mass;
            local_has_scale = 1;
         }

         int global_bad = 0, scale_count = 0;
         real_t scale_sum = 0.0;
         MPI_Allreduce(&local_bad, &global_bad, 1, MPI_INT, MPI_MAX,
                       fespace.GetComm());
         MPI_Allreduce(&local_has_scale, &scale_count, 1, MPI_INT, MPI_SUM,
                       fespace.GetComm());
         MPI_Allreduce(&local_scale, &scale_sum, 1,
                       MPITypeMap<real_t>::mpi_type, MPI_SUM,
                       fespace.GetComm());
         MFEM_VERIFY(global_bad == 0 && scale_count > 0,
                     "High-order lumping requires one uniform element geometry/order.");
         lumped_diagonal_scale = scale_sum / scale_count;

         const real_t local_scale_error = local_has_scale ?
            std::abs(local_scale - lumped_diagonal_scale) : 0.0;
         real_t global_scale_error = 0.0;
         MPI_Allreduce(&local_scale_error, &global_scale_error, 1,
                       MPITypeMap<real_t>::mpi_type, MPI_MAX,
                       fespace.GetComm());
         MFEM_VERIFY(global_scale_error <=
                     1e-12 * std::max(real_t(1.0), lumped_diagonal_scale),
                     "High-order lumping scale differs across MPI ranks.");

         Mmat->GetDiag(M_lumped_inv);
         M_lumped_inv *= lumped_diagonal_scale;
      }
      else
      {
         // Linear simplex elements retain standard row-sum lumping.
         Vector ones(true_size);
         ones = 1.0;
         Mmat->Mult(ones, M_lumped_inv);
      }

      real_t local_min_mass = M_lumped_inv.Min();
      real_t local_max_mass = M_lumped_inv.Max();
      real_t min_mass = 0.0, max_mass = 0.0;
      MPI_Allreduce(&local_min_mass, &min_mass, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_MIN,
                    fespace.GetComm());
      MPI_Allreduce(&local_max_mass, &max_mass, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_MAX,
                    fespace.GetComm());
      MFEM_VERIFY(min_mass > 0.0 && std::isfinite(min_mass) &&
                  std::isfinite(max_mass),
                  "Lumped mass must be finite and strictly positive.");

      // Invert: M_lumped_inv[i] = 1 / M_lumped[i]
      for (int i = 0; i < true_size; i++)
      {
         M_lumped_inv[i] = 1.0 / M_lumped_inv[i];
      }

      real_t local_min_inv = M_lumped_inv.Min();
      real_t local_max_inv = M_lumped_inv.Max();
      real_t min_inv = 0.0, max_inv = 0.0;
      MPI_Allreduce(&local_min_inv, &min_inv, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_MIN,
                    fespace.GetComm());
      MPI_Allreduce(&local_max_inv, &max_inv, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_MAX,
                    fespace.GetComm());

      if (myid == 0)
      {
         if (use_scaled_diagonal_lumping)
         {
            std::cout << "High-order lumping scale: "
                      << lumped_diagonal_scale << std::endl;
         }
         std::cout << "Inverse lumped mass range: ["
                   << min_inv << ", " << max_inv << "]" << std::endl;
      }
   }
   else // ITERATIVE
   {
      // Preserve Mmat as the full variational mass operator for inner products
      // and sensitivity audits.  When constraints are present, the solve
      // matrix is a deep copy with its essential rows/columns eliminated, so
      // its free block is exactly M_ff and its constrained diagonal is
      // design-independent.
      if (ess_tdof_list.Size() > 0)
      {
         M_free_mat = new HypreParMatrix(*Mmat);
         std::unique_ptr<HypreParMatrix> eliminated_entries(
            M_free_mat->EliminateRowsCols(ess_tdof_list));
      }
      else
      {
         // No constrained copy is needed for an unconstrained problem.
         M_free_mat = Mmat;
      }

      M_prec = new HypreBoomerAMG(*M_free_mat);
      M_prec->SetPrintLevel(0);

      M_solver = new CGSolver(fespace.GetComm());
      M_solver->SetPreconditioner(*M_prec);
      M_solver->SetOperator(*M_free_mat);
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

   const real_t local_load_norm_sq = load_base_vector * load_base_vector;
   real_t global_load_norm_sq = 0.0;
   MPI_Allreduce(&local_load_norm_sq, &global_load_norm_sq, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, fespace.GetComm());

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
      std::cout << "  Global base load norm: "
                << std::sqrt(global_load_norm_sq) << std::endl;
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
   const real_t time_factor =
      EvaluateLoadTimeFactor(load_time_profile, time, load_duration,
                             load_frequency, load_phase);
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
}

ElastodynamicsOperator::~ElastodynamicsOperator()
{
   delete M_solver;
   delete M_prec;
   if (M_free_mat != Mmat) { delete M_free_mat; }
   delete Cabs_mat;
   delete Cvol_mat;
   delete Kmat;
   delete Mmat;
   delete C_abs;
   delete C_vol;
   delete K;
   delete M;
}

struct RHSSpectralSummary
{
   real_t rhs_mass_inverse_norm = 0.0;
   real_t mean_lambda = 0.0;
   real_t lambda_05 = 0.0;
   real_t lambda_50 = 0.0;
   real_t lambda_95 = 0.0;
   Vector ritz_values;
   Vector spectral_weights;
};

inline real_t RHSSpectrumGlobalDot(MPI_Comm comm,
                                   const Vector &x, const Vector &y)
{
   const real_t local = x * y;
   real_t global = 0.0;
   MPI_Allreduce(&local, &global, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
   return global;
}

inline real_t MassInnerProduct(ElastodynamicsOperator &oper,
                               MPI_Comm comm,
                               const Vector &x, const Vector &y)
{
   Vector mass_y;
   oper.MultMass(y, mass_y);
   return RHSSpectrumGlobalDot(comm, x, mass_y);
}

// The configured MFEM library may not expose its LAPACK-backed dense
// eigensystem.  This Jacobi solve is ample for the <=O(50) symmetric Lanczos
// projection and keeps the diagnostic independent of that build option.
inline void SymmetricJacobiEigensystem(DenseMatrix matrix,
                                       Vector &eigenvalues,
                                       DenseMatrix &eigenvectors)
{
   const int n = matrix.Height();
   MFEM_VERIFY(matrix.Width() == n, "Jacobi eigensystem requires a square matrix.");
   eigenvectors.SetSize(n);
   eigenvectors = 0.0;
   for (int i = 0; i < n; i++) { eigenvectors(i, i) = 1.0; }

   real_t scale = 1.0;
   for (int i = 0; i < n; i++)
   {
      scale = std::max(scale, std::abs(matrix(i, i)));
   }
   const real_t tolerance = 1e-13 * scale;
   const int max_rotations = std::max(20, 100 * n * n);

   for (int rotation = 0; rotation < max_rotations; rotation++)
   {
      int p = 0, q = 0;
      real_t largest = 0.0;
      for (int i = 0; i < n; i++)
      {
         for (int j = i + 1; j < n; j++)
         {
            const real_t candidate = std::abs(matrix(i, j));
            if (candidate > largest)
            {
               largest = candidate;
               p = i;
               q = j;
            }
         }
      }
      if (largest <= tolerance) { break; }

      const real_t app = matrix(p, p);
      const real_t aqq = matrix(q, q);
      const real_t apq = matrix(p, q);
      const real_t tau = (aqq - app) / (2.0 * apq);
      const real_t t = (tau >= 0.0 ? 1.0 : -1.0) /
                       (std::abs(tau) + std::sqrt(1.0 + tau * tau));
      const real_t c = 1.0 / std::sqrt(1.0 + t * t);
      const real_t s = t * c;

      matrix(p, p) = app - t * apq;
      matrix(q, q) = aqq + t * apq;
      matrix(p, q) = matrix(q, p) = 0.0;

      for (int k = 0; k < n; k++)
      {
         if (k == p || k == q) { continue; }
         const real_t akp = matrix(k, p);
         const real_t akq = matrix(k, q);
         matrix(k, p) = matrix(p, k) = c * akp - s * akq;
         matrix(k, q) = matrix(q, k) = s * akp + c * akq;
      }

      for (int k = 0; k < n; k++)
      {
         const real_t vkp = eigenvectors(k, p);
         const real_t vkq = eigenvectors(k, q);
         eigenvectors(k, p) = c * vkp - s * vkq;
         eigenvectors(k, q) = s * vkp + c * vkq;
      }
   }

   eigenvalues.SetSize(n);
   for (int i = 0; i < n; i++) { eigenvalues[i] = matrix(i, i); }

   // Sort eigenpairs in ascending eigenvalue order.
   for (int i = 0; i < n; i++)
   {
      int smallest = i;
      for (int j = i + 1; j < n; j++)
      {
         if (eigenvalues[j] < eigenvalues[smallest]) { smallest = j; }
      }
      if (smallest != i)
      {
         std::swap(eigenvalues[i], eigenvalues[smallest]);
         for (int k = 0; k < n; k++)
         {
            std::swap(eigenvectors(k, i), eigenvectors(k, smallest));
         }
      }
   }
}

// Lanczos spectral measure of A = M^{-1}K seen from a right-hand side b.
// Starting with q_0 proportional to M^{-1}b means the Ritz weights approximate
// the energy |phi_j^T b|^2 / lambda-independent normalization in the
// M-orthonormal generalized eigenbasis K phi_j = lambda_j M phi_j.
inline RHSSpectralSummary AnalyzeRHSSpectrum(
   ElastodynamicsOperator &oper, MPI_Comm comm, const Vector &rhs,
   int requested_lanczos_steps, Vector *normalized_displacement = nullptr)
{
   MFEM_VERIFY(requested_lanczos_steps >= 2,
               "RHS spectral analysis needs at least two Lanczos steps.");

   Vector q;
   oper.MultInvMass(rhs, q);
   oper.ProjectEssentialField(q);

   const real_t norm_sq = MassInnerProduct(oper, comm, q, q);
   MFEM_VERIFY(norm_sq > 0.0 && std::isfinite(norm_sq),
               "Cannot analyze a zero or non-finite RHS.");
   const real_t norm = std::sqrt(norm_sq);
   q /= norm;
   if (normalized_displacement) { *normalized_displacement = q; }

   std::vector<Vector> basis;
   std::vector<real_t> diagonal;
   std::vector<real_t> off_diagonal;
   Vector stiffness_q(q.Size()), z(q.Size());

   for (int iteration = 0; iteration < requested_lanczos_steps; iteration++)
   {
      basis.push_back(q);
      oper.GetStiffnessMatrix()->Mult(q, stiffness_q);
      oper.MultInvMass(stiffness_q, z);
      oper.ProjectEssentialField(z);

      // Full M-orthogonalization keeps the small projected spectral measure
      // reliable even when the requested source is concentrated in a few modes.
      real_t alpha = 0.0;
      for (int j = 0; j < static_cast<int>(basis.size()); j++)
      {
         const real_t projection =
            MassInnerProduct(oper, comm, basis[j], z);
         if (j + 1 == static_cast<int>(basis.size()))
         {
            alpha = projection;
         }
         z.Add(-projection, basis[j]);
      }
      diagonal.push_back(alpha);

      const real_t beta_sq = MassInnerProduct(oper, comm, z, z);
      const real_t beta = std::sqrt(std::max(real_t(0.0), beta_sq));
      if (beta <= 1e-11 * std::max(real_t(1.0), std::abs(alpha)) ||
          iteration + 1 == requested_lanczos_steps)
      {
         break;
      }

      off_diagonal.push_back(beta);
      q = z;
      q /= beta;
   }

   const int n = static_cast<int>(diagonal.size());
   MFEM_VERIFY(n >= 1, "Lanczos spectral analysis produced no basis.");
   DenseMatrix projected(n);
   projected = 0.0;
   for (int i = 0; i < n; i++)
   {
      projected(i, i) = diagonal[i];
      if (i + 1 < n)
      {
         projected(i, i + 1) = off_diagonal[i];
         projected(i + 1, i) = off_diagonal[i];
      }
   }

   Vector eigenvalues;
   DenseMatrix eigenvectors;
   SymmetricJacobiEigensystem(projected, eigenvalues, eigenvectors);

   RHSSpectralSummary summary;
   summary.rhs_mass_inverse_norm = norm;
   summary.ritz_values = eigenvalues;
   summary.spectral_weights.SetSize(n);

   real_t weight_sum = 0.0;
   for (int i = 0; i < n; i++)
   {
      const real_t weight = eigenvectors(0, i) * eigenvectors(0, i);
      summary.spectral_weights[i] = weight;
      weight_sum += weight;
   }
   MFEM_VERIFY(weight_sum > 0.0, "Invalid Lanczos spectral weights.");
   summary.spectral_weights /= weight_sum;

   real_t cumulative = 0.0;
   bool set_05 = false, set_50 = false, set_95 = false;
   for (int i = 0; i < n; i++)
   {
      const real_t lambda = std::max(real_t(0.0), eigenvalues[i]);
      const real_t weight = summary.spectral_weights[i];
      summary.mean_lambda += weight * lambda;
      cumulative += weight;
      if (!set_05 && cumulative >= 0.05)
      {
         summary.lambda_05 = lambda;
         set_05 = true;
      }
      if (!set_50 && cumulative >= 0.50)
      {
         summary.lambda_50 = lambda;
         set_50 = true;
      }
      if (!set_95 && cumulative >= 0.95)
      {
         summary.lambda_95 = lambda;
         set_95 = true;
      }
   }
   return summary;
}

inline void ValidateRK4TimeStep(ElastodynamicsOperator &oper,
                                real_t requested_dt,
                                bool print_report = true)
{
   const bool lumped =
      oper.GetMassSolverType() == MassSolverType::LUMPED;
   const real_t dt_wave = lumped ?
      oper.EstimateLumpedRK4TimeStep() :
      oper.EstimateConsistentRK4TimeStep();
   const real_t dt_damping = lumped ?
      oper.EstimateLumpedRK4DampingTimeStep() :
      oper.EstimateConsistentRK4DampingTimeStep();
   const real_t omega_max = 2.0 * std::sqrt(real_t(2.0)) / dt_wave;
   const real_t damping_rate_max = std::isfinite(dt_damping) ?
      real_t(2.7852935634052816) / dt_damping : 0.0;
   const real_t dt_rk4 = std::min(dt_wave, dt_damping);
   const real_t recommended_dt = 0.8 * dt_rk4;
   const bool above_recommended = requested_dt > recommended_dt;

   if (Mpi::Root() && (print_report || above_recommended))
   {
      std::cout << "RK4 timestep spectral estimates ("
                << (lumped ? "lumped" : "consistent") << " mass):\n"
                << "  wave: omega_max = "
                << std::scientific << std::setprecision(6) << omega_max
                << ", raw dt_max = " << dt_wave << "\n"
                << "  damping: rate_max = " << damping_rate_max
                << ", raw dt_max = " << dt_damping << "\n"
                << "  componentwise recommended dt <= " << recommended_dt
                << " (80% safety), requested dt = " << requested_dt << "\n";
   }

   // The separate undamped-wave and pure-damping endpoints are not jointly
   // sufficient at their raw RK4 limits: a damped oscillator can lie outside
   // the two-dimensional RK4 stability region while satisfying both scalar
   // bounds.  Enforce the conservative margin instead of merely warning at
   // it.  This remains a componentwise spectral guard, but excludes the known
   // unstable corner where both effects are simultaneously near their limits.
   MFEM_VERIFY(requested_dt <= recommended_dt,
               "Requested timestep exceeds the enforced 80%-safe RK4 "
               "wave/damping estimate (raw component limits are not jointly "
               "sufficient for the damped first-order system).");
}

// Compatibility wrapper for existing call sites. Validation is no longer
// lumped-only: consistent mass uses the generalized free-DOF eigenproblems.
inline void ValidateLumpedRK4TimeStep(ElastodynamicsOperator &oper,
                                      real_t requested_dt,
                                      bool print_report = true)
{
   ValidateRK4TimeStep(oper, requested_dt, print_report);
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
// The forward solve can use the *consistent* mass M, linear/tensor-product
// row-sum lumping, or positive scaled-diagonal lumping for high-order simplex
// elements. The sensitivity MUST differentiate whichever mass drives forward:
//   - CONSISTENT: (a . z) is the L2 product of the interpolated fields at the
//     quadrature point, (sum_i a_i phi_i) . (sum_j z_j phi_j).
//   - ROW-SUM LUMPED: the diagonal lump collapses the product to the contraction
//     g(x) = sum_i (a_i . z_i) phi_i(x), the interpolant of the per-node dot
//     products.
//   - SCALED-DIAGONAL LUMPED: g(x) = s_p sum_i (a_i . z_i) phi_i(x)^2.
class StageMassDesignLFIntegrator : public LinearFormIntegrator
{
private:
   ParGridFunction &rho_tilde;
   ParGridFunction &accel;
   ParGridFunction &z;
   MaterialParams mat;
   bool lumped;
   bool scaled_diagonal;
   real_t diagonal_scale;
   Vector shape, state_shape, accel_val, z_val;

public:
   StageMassDesignLFIntegrator(ParGridFunction &rho_tilde_,
                               ParGridFunction &accel_,
                               ParGridFunction &z_,
                               const MaterialParams &mat_,
                               bool lumped_ = false,
                               bool scaled_diagonal_ = false,
                               real_t diagonal_scale_ = 1.0)
      : rho_tilde(rho_tilde_), accel(accel_), z(z_), mat(mat_),
        lumped(lumped_), scaled_diagonal(scaled_diagonal_),
        diagonal_scale(diagonal_scale_) {}

   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &T,
                               Vector &elvect) override
   {
      const int dof = el.GetDof();
      shape.SetSize(dof);
      elvect.SetSize(dof);
      elvect = 0.0;

      // For the lumped mass, precompute the per-node dot products
      // g_i = a_i . z_i from the STATE element's nodal vector dofs. The
      // resulting state-space interpolant is then integrated against the
      // filter basis. In particular, state and filter orders need not match.
      Vector g_nodal;
      const FiniteElementSpace *afes = accel.FESpace();
      const FiniteElement *state_el = afes->GetFE(T.ElementNo);
      const int state_dof = state_el->GetDof();
      if (lumped)
      {
         state_shape.SetSize(state_dof);
         const int vdim = afes->GetVDim();
         const int ordering = afes->GetOrdering();
         Array<int> vdofs;
         afes->GetElementVDofs(T.ElementNo, vdofs);
         Vector a_edof, z_edof;
         accel.GetSubVector(vdofs, a_edof);
         z.GetSubVector(vdofs, z_edof);

         g_nodal.SetSize(state_dof);
         for (int i = 0; i < state_dof; i++)
         {
            real_t s = 0.0;
            for (int c = 0; c < vdim; c++)
            {
               const int idx = (ordering == Ordering::byNODES)
                               ? (c * state_dof + i) : (i * vdim + c);
               s += a_edof(idx) * z_edof(idx);
            }
            g_nodal(i) = s;
         }
      }

      // Differentiate the quadrature rule that assembled the state mass
      // matrix. The old filter-only rule used p_design here, which no longer
      // matches the discrete forward operator when p_state > p_design.
      const int int_order = 2 * state_el->GetOrder() + T.OrderW();
      const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), int_order);

      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T.SetIntPoint(&ip);
         el.CalcPhysShape(T, shape);

         real_t az;
         if (lumped)
         {
            state_el->CalcPhysShape(T, state_shape);
            if (scaled_diagonal)
            {
               az = 0.0;
               for (int i = 0; i < state_dof; i++)
               {
                  az += g_nodal(i) * state_shape(i) * state_shape(i);
               }
               az *= diagonal_scale;
            }
            else
            {
               az = g_nodal * state_shape;
            }
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

      const FiniteElement *state_el = u.FESpace()->GetFE(T.ElementNo);
      // Match the quadrature used by the state-space ElasticityIntegrator,
      // rather than deriving it from the lower-order filter test space.
      const int int_order = 2 * T.OrderGrad(state_el);
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
                                        Vector &dJ_drho_tilde,
                                        real_t temporal_weight = 1.0)
{
   MFEM_VERIFY(std::isfinite(temporal_weight),
               "Design-gradient stage has a non-finite temporal weight.");
   const Array<int> &offsets = oper.GetBlockOffsets();
   BlockVector state_blocks(const_cast<Vector&>(stage_state), offsets);
   BlockVector rhs_blocks(const_cast<Vector&>(stage_rhs), offsets);
   BlockVector seed_blocks(const_cast<Vector&>(stage_seed), offsets);

   // The forward operator projects its acceleration block onto the free true
   // DOFs. The transpose contraction must therefore use M^{-T} P p_v, even
   // when a caller supplies an arbitrary (not already projected) stage seed.
   Vector projected_velocity_seed(seed_blocks.GetBlock(1));
   oper.ProjectEssentialField(projected_velocity_seed);
   Vector z_true;
   oper.MultInvMass(projected_velocity_seed, z_true);
   z_true *= temporal_weight;

   ParGridFunction u_gf(&state_fes);
   ParGridFunction accel_gf(&state_fes);
   ParGridFunction z_gf(&state_fes);

   u_gf.SetFromTrueDofs(state_blocks.GetBlock(0));
   accel_gf.SetFromTrueDofs(rhs_blocks.GetBlock(1));
   z_gf.SetFromTrueDofs(z_true);

   const bool lumped = (oper.GetMassSolverType() == MassSolverType::LUMPED);
   ParLinearForm mass_lf(&filter_fes);
   mass_lf.AddDomainIntegrator(
      new StageMassDesignLFIntegrator(
         rho_tilde, accel_gf, z_gf, mat, lumped,
         oper.UsesScaledDiagonalLumping(), oper.GetLumpedDiagonalScale()));
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

inline real_t AddObjectiveContributionAtTime(
   ParFiniteElementSpace &state_fes,
   const Array<int> &offsets,
   TimeIntegratedObjective &objective,
   const Vector &state,
   real_t physical_time,
   real_t dt, int step, int total_steps)
{
   BlockVector bstate(const_cast<Vector&>(state), offsets);
   ParGridFunction u_gf(&state_fes);
   u_gf.SetFromTrueDofs(bstate.GetBlock(0));
   const real_t contribution =
      objective.AccumulateTimestepAtTime(
         u_gf, physical_time, dt, step, total_steps);
   // The objective performs a global reduction, so every rank observes the same
   // value. Catch an unstable state here before REVOLVE replays it in the
   // adjoint or MMA consumes a NaN objective/gradient.
   MFEM_VERIFY(std::isfinite(contribution) &&
               std::isfinite(objective.GetObjective()),
               "Forward solve produced a non-finite objective contribution; "
               "the state is numerically unstable.");
   return contribution;
}

inline real_t AddObjectiveContribution(ParFiniteElementSpace &state_fes,
                                       const Array<int> &offsets,
                                       TimeIntegratedObjective &objective,
                                       const Vector &state,
                                       real_t dt, int step, int total_steps)
{
   return AddObjectiveContributionAtTime(
      state_fes, offsets, objective, state, step * dt,
      dt, step, total_steps);
}

inline void ObjectiveGradientAtStateAndTime(
   ParFiniteElementSpace &state_fes,
   const Array<int> &offsets,
   TimeIntegratedObjective &objective,
   const Vector &state,
   real_t physical_time,
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
   objective.ComputeObjectiveGradientAtTime(
      u_gf, physical_time, dt, step, total_steps, grad_form);

   std::unique_ptr<HypreParVector> q_u(grad_form.ParallelAssemble());
   bq.GetBlock(0) = *q_u;
   bq.GetBlock(1) = 0.0;
}

inline void ObjectiveGradientAtState(ParFiniteElementSpace &state_fes,
                                     const Array<int> &offsets,
                                     TimeIntegratedObjective &objective,
                                     const Vector &state,
                                     real_t dt, int step, int total_steps,
                                     Vector &q_state)
{
   ObjectiveGradientAtStateAndTime(
      state_fes, offsets, objective, state, step * dt,
      dt, step, total_steps, q_state);
}

inline void InstantaneousObjectiveGradientAtState(
   ParFiniteElementSpace &state_fes,
   const Array<int> &offsets,
   TimeIntegratedObjective &objective,
   const Vector &state,
   real_t physical_time,
   Vector &q_state)
{
   q_state.SetSize(state.Size());
   q_state = 0.0;

   BlockVector bstate(const_cast<Vector&>(state), offsets);
   BlockVector bq(q_state, offsets);

   ParGridFunction u_gf(&state_fes);
   u_gf.SetFromTrueDofs(bstate.GetBlock(0));

   ParLinearForm grad_form(&state_fes);
   objective.AssembleInstantaneousStateGradient(
      u_gf, physical_time, grad_form);

   std::unique_ptr<HypreParVector> q_u(grad_form.ParallelAssemble());
   bq.GetBlock(0) = *q_u;
   bq.GetBlock(1) = 0.0;
}

inline real_t InstantaneousObjectiveValueAtState(
   ParFiniteElementSpace &state_fes,
   const Array<int> &offsets,
   TimeIntegratedObjective &objective,
   const Vector &state,
   real_t physical_time)
{
   BlockVector state_blocks(const_cast<Vector&>(state), offsets);
   ParGridFunction displacement(&state_fes);
   displacement.SetFromTrueDofs(state_blocks.GetBlock(0));
   const real_t value =
      objective.EvaluateInstantaneous(displacement, physical_time);
   MFEM_VERIFY(std::isfinite(value),
               "Instantaneous objective evaluation is non-finite.");
   return value;
}

// One forward classical-RK4 step together with the matching four-stage
// quadrature of the running objective,
//
//   J_n = h sum_i b_i ell(Y_i,t_n+c_i h),
//
// where Y_1=x_n, Y_2 and Y_3 are the two distinct midpoint stages, and
// Y_4=x_n+h k_3.  In particular, Y_4 is not replaced by the accepted endpoint
// x_{n+1}; retaining that distinction is essential for an exact DO derivative.
inline real_t RK4StageObjectiveForwardOneStep(
   ElastodynamicsOperator &oper,
   ParFiniteElementSpace &state_fes,
   TimeIntegratedObjective &objective,
   const Vector &x0,
   real_t t0,
   real_t h,
   Vector &x_next)
{
   MFEM_VERIFY(std::isfinite(t0) && std::isfinite(h) && h > 0.0,
               "RK4 stage-objective step has an invalid time interval.");
   const int n = x0.Size();
   Vector k1(n), k2(n), k3(n), k4(n);
   Vector y2(n), y3(n), y4(n);
   RK4Stages(oper, x0, t0, h, k1, k2, k3, k4, y2, y3, y4);

   const Array<int> &offsets = oper.GetBlockOffsets();
   const real_t ell1 = InstantaneousObjectiveValueAtState(
      state_fes, offsets, objective, x0, t0);
   const real_t ell2 = InstantaneousObjectiveValueAtState(
      state_fes, offsets, objective, y2, t0 + 0.5 * h);
   const real_t ell3 = InstantaneousObjectiveValueAtState(
      state_fes, offsets, objective, y3, t0 + 0.5 * h);
   const real_t ell4 = InstantaneousObjectiveValueAtState(
      state_fes, offsets, objective, y4, t0 + h);
   const real_t contribution =
      h * (ell1 / 6.0 + ell2 / 3.0 + ell3 / 3.0 + ell4 / 6.0);
   MFEM_VERIFY(std::isfinite(contribution),
               "RK4 stage-objective contribution is non-finite.");

   // Match MFEM's RK4Solver accumulation order.
   x_next = x0;
   x_next.Add(h / 6.0, k1);
   x_next.Add(h / 3.0, k2);
   x_next.Add(h / 3.0, k3);
   x_next.Add(h / 6.0, k4);
   int local_nonfinite = 0;
   for (int i = 0; i < x_next.Size(); i++)
   {
      if (!std::isfinite(x_next[i]))
      {
         local_nonfinite = 1;
         break;
      }
   }
   int global_nonfinite = 0;
   MPI_Allreduce(&local_nonfinite, &global_nonfinite, 1, MPI_INT, MPI_MAX,
                 state_fes.GetComm());
   MFEM_VERIFY(global_nonfinite == 0,
               "RK4 stage-objective step produced a non-finite state.");
   return contribution;
}

inline real_t RK4StageObjectiveForwardSweepFullStorage(
   ElastodynamicsOperator &oper,
   ParFiniteElementSpace &state_fes,
   TimeIntegratedObjective &objective,
   const Vector &initial_state,
   int forward_steps,
   real_t start_time,
   real_t forward_step,
   std::vector<Vector> &states)
{
   MFEM_VERIFY(forward_steps > 0 && std::isfinite(start_time) &&
               std::isfinite(forward_step) && forward_step > 0.0,
               "RK4 stage-objective sweep has an invalid time grid.");
   states.resize(forward_steps + 1);
   states[0] = initial_state;
   real_t objective_value = 0.0;
   for (int step = 0; step < forward_steps; step++)
   {
      objective_value += RK4StageObjectiveForwardOneStep(
         oper, state_fes, objective, states[step],
         start_time + step * forward_step, forward_step,
         states[step + 1]);
   }
   MFEM_VERIFY(std::isfinite(objective_value),
               "RK4 stage-objective sweep produced a non-finite objective.");
   return objective_value;
}

inline real_t RK4StageObjectiveForwardSweepStreaming(
   ElastodynamicsOperator &oper,
   ParFiniteElementSpace &state_fes,
   TimeIntegratedObjective &objective,
   const Vector &initial_state,
   int forward_steps,
   real_t start_time,
   real_t forward_step,
   const std::function<void(int, real_t, const Vector&)> &endpoint = {})
{
   MFEM_VERIFY(forward_steps > 0 && std::isfinite(start_time) &&
               std::isfinite(forward_step) && forward_step > 0.0,
               "Streaming RK4 stage-objective sweep has an invalid time grid.");
   Vector state(initial_state), next(initial_state.Size());
   if (endpoint) { endpoint(0, start_time, state); }
   real_t objective_value = 0.0;
   for (int step = 0; step < forward_steps; step++)
   {
      const real_t time = start_time + step * forward_step;
      objective_value += RK4StageObjectiveForwardOneStep(
         oper, state_fes, objective, state, time, forward_step, next);
      state = next;
      if (endpoint)
      {
         endpoint(step + 1, time + forward_step, state);
      }
   }
   MFEM_VERIFY(std::isfinite(objective_value),
               "Streaming RK4 stage-objective is non-finite.");
   return objective_value;
}

// Exact reverse accumulation for the fully discrete RK4 step augmented by the
// four-stage running objective above.  This is the literal reverse AD (DO)
// form: objective seeds h*b_i*ell_z(Y_i) enter each stored stage before its
// dependencies are propagated to earlier stages.
inline void RK4StageObjectiveDOAdjointOneStepWithDesign(
   ElastodynamicsOperator &oper,
   ParFiniteElementSpace &state_fes,
   ParFiniteElementSpace &filter_fes,
   ParGridFunction &rho_tilde,
   const MaterialParams &mat,
   TimeIntegratedObjective &objective,
   const Vector &x0,
   real_t t0,
   real_t h,
   const Vector &lambda_next,
   Vector &lambda_prev,
   Vector &dJ_drho_tilde)
{
   const int n = x0.Size();
   Vector k1(n), k2(n), k3(n), k4(n);
   Vector y2(n), y3(n), y4(n);
   RK4Stages(oper, x0, t0, h, k1, k2, k3, k4, y2, y3, y4);

   Vector q1(n), q2(n), q3(n), q4(n);
   InstantaneousObjectiveGradientAtState(
      state_fes, oper.GetBlockOffsets(), objective, x0, t0, q1);
   InstantaneousObjectiveGradientAtState(
      state_fes, oper.GetBlockOffsets(), objective,
      y2, t0 + 0.5 * h, q2);
   InstantaneousObjectiveGradientAtState(
      state_fes, oper.GetBlockOffsets(), objective,
      y3, t0 + 0.5 * h, q3);
   InstantaneousObjectiveGradientAtState(
      state_fes, oper.GetBlockOffsets(), objective, y4, t0 + h, q4);

   Vector adj_x0(lambda_next);
   Vector adj_k1(n), adj_k2(n), adj_k3(n), adj_k4(n);
   Vector adj_y(n), jt(n);
   adj_k1.Set(h / 6.0, lambda_next);
   adj_k2.Set(h / 3.0, lambda_next);
   adj_k3.Set(h / 3.0, lambda_next);
   adj_k4.Set(h / 6.0, lambda_next);

   AddStageDesignGradientTilde(
      oper, state_fes, filter_fes, rho_tilde, mat,
      y4, k4, adj_k4, dJ_drho_tilde);
   EvalJacobianTranspose(oper, y4, t0 + h, adj_k4, adj_y);
   adj_y.Add(h / 6.0, q4);
   adj_x0 += adj_y;
   adj_k3.Add(h, adj_y);

   AddStageDesignGradientTilde(
      oper, state_fes, filter_fes, rho_tilde, mat,
      y3, k3, adj_k3, dJ_drho_tilde);
   EvalJacobianTranspose(oper, y3, t0 + 0.5 * h, adj_k3, adj_y);
   adj_y.Add(h / 3.0, q3);
   adj_x0 += adj_y;
   adj_k2.Add(0.5 * h, adj_y);

   AddStageDesignGradientTilde(
      oper, state_fes, filter_fes, rho_tilde, mat,
      y2, k2, adj_k2, dJ_drho_tilde);
   EvalJacobianTranspose(oper, y2, t0 + 0.5 * h, adj_k2, adj_y);
   adj_y.Add(h / 3.0, q2);
   adj_x0 += adj_y;
   adj_k1.Add(0.5 * h, adj_y);

   AddStageDesignGradientTilde(
      oper, state_fes, filter_fes, rho_tilde, mat,
      x0, k1, adj_k1, dJ_drho_tilde);
   EvalJacobianTranspose(oper, x0, t0, adj_k1, jt);
   jt.Add(h / 6.0, q1);
   adj_x0 += jt;
   lambda_prev = adj_x0;
}

// The transformed partitioned-RK4 adjoint written in normalized stage-adjoint
// variables P_i=bar{k}_i/(h b_i).  It is intentionally implemented separately
// from the reverse-AD kernel so the experiment can verify DO=OD_modified rather
// than assuming that identity in code.
inline void RK4StageObjectiveTransformedAdjointOneStepWithDesign(
   ElastodynamicsOperator &oper,
   ParFiniteElementSpace &state_fes,
   ParFiniteElementSpace &filter_fes,
   ParGridFunction &rho_tilde,
   const MaterialParams &mat,
   TimeIntegratedObjective &objective,
   const Vector &x0,
   real_t t0,
   real_t h,
   const Vector &lambda_next,
   Vector &lambda_prev,
   Vector &dJ_drho_tilde)
{
   const int n = x0.Size();
   Vector k1(n), k2(n), k3(n), k4(n);
   Vector y2(n), y3(n), y4(n);
   RK4Stages(oper, x0, t0, h, k1, k2, k3, k4, y2, y3, y4);

   Vector q1(n), q2(n), q3(n), q4(n);
   InstantaneousObjectiveGradientAtState(
      state_fes, oper.GetBlockOffsets(), objective, x0, t0, q1);
   InstantaneousObjectiveGradientAtState(
      state_fes, oper.GetBlockOffsets(), objective,
      y2, t0 + 0.5 * h, q2);
   InstantaneousObjectiveGradientAtState(
      state_fes, oper.GetBlockOffsets(), objective,
      y3, t0 + 0.5 * h, q3);
   InstantaneousObjectiveGradientAtState(
      state_fes, oper.GetBlockOffsets(), objective, y4, t0 + h, q4);

   Vector p1(n), p2(n), p3(n), p4(lambda_next);
   Vector g1(n), g2(n), g3(n), g4(n);

   EvalJacobianTranspose(oper, y4, t0 + h, p4, g4);
   g4 += q4;
   p3 = lambda_next;
   p3.Add(0.5 * h, g4);

   EvalJacobianTranspose(oper, y3, t0 + 0.5 * h, p3, g3);
   g3 += q3;
   p2 = lambda_next;
   p2.Add(0.5 * h, g3);

   EvalJacobianTranspose(oper, y2, t0 + 0.5 * h, p2, g2);
   g2 += q2;
   p1 = lambda_next;
   p1.Add(h, g2);

   EvalJacobianTranspose(oper, x0, t0, p1, g1);
   g1 += q1;

   AddStageDesignGradientTilde(
      oper, state_fes, filter_fes, rho_tilde, mat,
      x0, k1, p1, dJ_drho_tilde, h / 6.0);
   AddStageDesignGradientTilde(
      oper, state_fes, filter_fes, rho_tilde, mat,
      y2, k2, p2, dJ_drho_tilde, h / 3.0);
   AddStageDesignGradientTilde(
      oper, state_fes, filter_fes, rho_tilde, mat,
      y3, k3, p3, dJ_drho_tilde, h / 3.0);
   AddStageDesignGradientTilde(
      oper, state_fes, filter_fes, rho_tilde, mat,
      y4, k4, p4, dJ_drho_tilde, h / 6.0);

   lambda_prev = lambda_next;
   lambda_prev.Add(h / 6.0, g1);
   lambda_prev.Add(h / 3.0, g2);
   lambda_prev.Add(h / 3.0, g3);
   lambda_prev.Add(h / 6.0, g4);
}

enum class NestedTimeGridRelation
{
   SAME,
   FORWARD_FINER,
   ADJOINT_FINER
};

// The historical optimizer differentiates the fully discrete RK4/trapezoidal
// objective. The continuous mode is opt-in and uses an RK4/Hermite
// continuous-adjoint integral, with REVOLVE still scheduled on coarse forward
// intervals.
enum class TransientAdjointMode
{
   DISCRETE,
   CONTINUOUS
};

// Production trajectory ownership is independent of the continuous-adjoint
// time-grid relation. FULL retains every accepted forward endpoint, while
// REVOLVE retains only its scheduled block checkpoints and regenerates local
// interval data during the reverse sweep.
enum class TrajectoryStorageMode
{
   REVOLVE,
   FULL
};

inline const char *TransientAdjointModeName(TransientAdjointMode mode)
{
   switch (mode)
   {
      case TransientAdjointMode::DISCRETE: return "discrete";
      case TransientAdjointMode::CONTINUOUS: return "continuous";
   }
   return "unknown";
}

inline const char *TrajectoryStorageModeName(TrajectoryStorageMode mode)
{
   switch (mode)
   {
      case TrajectoryStorageMode::REVOLVE: return "revolve";
      case TrajectoryStorageMode::FULL: return "full";
   }
   return "unknown";
}

inline const char *NestedTimeGridRelationName(NestedTimeGridRelation relation)
{
   switch (relation)
   {
      case NestedTimeGridRelation::SAME: return "same";
      case NestedTimeGridRelation::FORWARD_FINER: return "forward-finer";
      case NestedTimeGridRelation::ADJOINT_FINER: return "adjoint-finer";
   }
   return "unknown";
}

// Describes two fixed, nested endpoint grids on [0,T].  The continuous-adjoint
// RK kernel is deliberately independent of this relation; a ForwardStateProvider
// decides how x(t) is supplied at its physical RK stage times.
struct NestedTimeGrid
{
   real_t t_final;
   int forward_steps;
   int adjoint_steps;
   real_t dt_forward;
   real_t dt_adjoint;
   int integer_ratio;
   NestedTimeGridRelation relation;

   static NestedTimeGrid Create(real_t final_time,
                                int num_forward_steps,
                                int num_adjoint_steps)
   {
      MFEM_VERIFY(final_time > 0.0,
                  "Nested time grids require positive final time.");
      MFEM_VERIFY(num_forward_steps > 0 && num_adjoint_steps > 0,
                  "Nested time grids require positive step counts.");

      NestedTimeGrid grid;
      grid.t_final = final_time;
      grid.forward_steps = num_forward_steps;
      grid.adjoint_steps = num_adjoint_steps;
      grid.dt_forward = final_time / num_forward_steps;
      grid.dt_adjoint = final_time / num_adjoint_steps;

      if (num_forward_steps == num_adjoint_steps)
      {
         grid.relation = NestedTimeGridRelation::SAME;
         grid.integer_ratio = 1;
      }
      else if (num_forward_steps > num_adjoint_steps)
      {
         MFEM_VERIFY(num_forward_steps % num_adjoint_steps == 0,
                     "Forward-finer grids require N_f to be divisible by N_a.");
         grid.relation = NestedTimeGridRelation::FORWARD_FINER;
         grid.integer_ratio = num_forward_steps / num_adjoint_steps;
      }
      else
      {
         MFEM_VERIFY(num_adjoint_steps % num_forward_steps == 0,
                     "Adjoint-finer grids require N_a to be divisible by N_f.");
         grid.relation = NestedTimeGridRelation::ADJOINT_FINER;
         grid.integer_ratio = num_adjoint_steps / num_forward_steps;
      }
      return grid;
   }
};

class ForwardStateProvider
{
public:
   virtual ~ForwardStateProvider() = default;
   virtual void Evaluate(real_t physical_time, Vector &state) const = 0;
};

// Exact lookup on a stored uniform forward grid.  This provider intentionally
// rejects off-node requests.  For an even forward-finer ratio, every classical
// RK4 adjoint endpoint/midpoint stage lands on one of these nodes.
class ExactStoredForwardStateProvider : public ForwardStateProvider
{
private:
   const std::vector<Vector> &states_;
   real_t start_time_;
   real_t time_step_;

public:
   ExactStoredForwardStateProvider(const std::vector<Vector> &states,
                                   real_t start_time,
                                   real_t time_step)
      : states_(states), start_time_(start_time), time_step_(time_step)
   {
      MFEM_VERIFY(states_.size() >= 2,
                  "Stored forward-state provider needs at least two nodes.");
      MFEM_VERIFY(time_step_ > 0.0,
                  "Stored forward-state provider needs a positive timestep.");
   }

   void Evaluate(real_t physical_time, Vector &state) const override
   {
      const real_t node_real = (physical_time - start_time_) / time_step_;
      const long long node = std::llround(node_real);
      const real_t tolerance =
         256.0 * std::numeric_limits<real_t>::epsilon()
         * std::max(real_t(1.0), std::abs(node_real));
      MFEM_VERIFY(std::abs(node_real - node) <= tolerance,
                  "Continuous adjoint requested a forward state away from a "
                  "stored node; use an interpolating ForwardStateProvider.");
      MFEM_VERIFY(node >= 0 &&
                  node < static_cast<long long>(states_.size()),
                  "Continuous adjoint requested a forward state outside [0,T].");
      state = states_[static_cast<std::size_t>(node)];
   }
};

// Build the physical endpoint slopes required by cubic Hermite interpolation:
// x'_i = f(x_i,t_i).  Full-storage verification keeps these beside the stored
// endpoint states.  A checkpointed implementation will build only the two
// slopes belonging to the replayed forward interval.
inline void BuildForwardStateTimeDerivatives(
   ElastodynamicsOperator &oper,
   const std::vector<Vector> &states,
   real_t start_time,
   real_t time_step,
   std::vector<Vector> &derivatives)
{
   MFEM_VERIFY(states.size() >= 2,
               "Forward-state derivative construction needs two or more nodes.");
   MFEM_VERIFY(std::isfinite(start_time) &&
               std::isfinite(time_step) && time_step > 0.0,
               "Forward-state derivative construction needs a positive step.");

   derivatives.resize(states.size());
   for (std::size_t node = 0; node < states.size(); node++)
   {
      MFEM_VERIFY(states[node].Size() == oper.Width(),
                  "Stored forward state has an unexpected size.");
      EvalRHS(oper, states[node],
              start_time + static_cast<real_t>(node) * time_step,
              derivatives[node]);
   }
}

inline void EvaluateCubicHermiteSegment(
   const Vector &x_left,
   const Vector &x_right,
   const Vector &f_left,
   const Vector &f_right,
   real_t interval_size,
   real_t theta,
   Vector &state)
{
   MFEM_VERIFY(x_left.Size() > 0 &&
               x_right.Size() == x_left.Size() &&
               f_left.Size() == x_left.Size() &&
               f_right.Size() == x_left.Size(),
               "Cubic Hermite segment data have inconsistent sizes.");
   MFEM_VERIFY(std::isfinite(interval_size) && interval_size > 0.0 &&
               std::isfinite(theta) && theta >= 0.0 && theta <= 1.0,
               "Cubic Hermite segment received invalid coordinates.");

   const real_t theta2 = theta * theta;
   const real_t theta3 = theta2 * theta;
   const real_t h00 = 2.0 * theta3 - 3.0 * theta2 + 1.0;
   const real_t h10 = theta3 - 2.0 * theta2 + theta;
   const real_t h01 = -2.0 * theta3 + 3.0 * theta2;
   const real_t h11 = theta3 - theta2;

   state.SetSize(x_left.Size());
   state.Set(h00, x_left);
   state.Add(interval_size * h10, f_left);
   state.Add(h01, x_right);
   state.Add(interval_size * h11, f_right);
}

// Piecewise cubic Hermite reconstruction of the full first-order state.
// Given endpoint values x_i, x_(i+1) and physical slopes f_i, f_(i+1), its
// pointwise state error is O(dt_f^4) for a smooth trajectory.  This provider
// is deliberately independent of the adjoint grid: same-grid and
// adjoint-finer RK4 stages can request any physical time in [0,T].
class CubicHermiteForwardStateProvider : public ForwardStateProvider
{
private:
   // Non-owning views. The stored states/slopes must outlive this provider.
   const std::vector<Vector> &states_;
   const std::vector<Vector> &derivatives_;
   real_t start_time_;
   real_t time_step_;

public:
   CubicHermiteForwardStateProvider(
      const std::vector<Vector> &states,
      const std::vector<Vector> &derivatives,
      real_t start_time,
      real_t time_step)
      : states_(states), derivatives_(derivatives),
        start_time_(start_time), time_step_(time_step)
   {
      MFEM_VERIFY(states_.size() >= 2,
                  "Cubic Hermite reconstruction needs at least two nodes.");
      MFEM_VERIFY(derivatives_.size() == states_.size(),
                  "Cubic Hermite states and derivatives must have equal size.");
      MFEM_VERIFY(std::isfinite(start_time_) &&
                  std::isfinite(time_step_) && time_step_ > 0.0,
                  "Cubic Hermite reconstruction needs finite time data and "
                  "a positive timestep.");

      const int state_size = states_.front().Size();
      MFEM_VERIFY(state_size > 0,
                  "Cubic Hermite reconstruction received empty states.");
      for (std::size_t node = 0; node < states_.size(); node++)
      {
         MFEM_VERIFY(states_[node].Size() == state_size &&
                     derivatives_[node].Size() == state_size,
                     "Cubic Hermite node data have inconsistent sizes.");
      }
   }

   void Evaluate(real_t physical_time, Vector &state) const override
   {
      MFEM_VERIFY(std::isfinite(physical_time),
                  "Cubic Hermite forward-state request time is non-finite.");
      const real_t interval_coordinate =
         (physical_time - start_time_) / time_step_;
      MFEM_VERIFY(std::isfinite(interval_coordinate),
                  "Cubic Hermite interval coordinate is non-finite.");
      const long long final_node =
         static_cast<long long>(states_.size()) - 1;
      // Subtracting a late block-local origin from a nearby physical stage
      // time loses absolute-time ulps before division by dt.  Scale the
      // normalized-coordinate tolerance by that cancellation as well as by
      // the local coordinate.  This matters for forward-finer REVOLVE, whose
      // provider is rebuilt with a nonzero origin for every q-interval block.
      const real_t cancellation_scale =
         std::max(std::abs(physical_time), std::abs(start_time_)) / time_step_;
      MFEM_VERIFY(std::isfinite(cancellation_scale),
                  "Cubic Hermite time origin is unresolved at this timestep.");
      const real_t tolerance =
         256.0 * std::numeric_limits<real_t>::epsilon()
         * std::max({real_t(1.0), std::abs(interval_coordinate),
                     cancellation_scale});

      MFEM_VERIFY(interval_coordinate >= -tolerance &&
                  interval_coordinate <= final_node + tolerance,
                  "Cubic Hermite forward-state request is outside [0,T].");

      const real_t clamped_coordinate =
         std::min(real_t(final_node),
                  std::max(real_t(0.0), interval_coordinate));
      const long long nearest_node = std::llround(clamped_coordinate);
      if (std::abs(clamped_coordinate - nearest_node) <= tolerance)
      {
         state = states_[static_cast<std::size_t>(nearest_node)];
         return;
      }

      const long long left_node =
         static_cast<long long>(std::floor(clamped_coordinate));
      MFEM_VERIFY(left_node >= 0 && left_node < final_node,
                  "Cubic Hermite interval lookup failed.");
      const real_t theta = clamped_coordinate - left_node;

      const std::size_t left = static_cast<std::size_t>(left_node);
      const std::size_t right = left + 1;
      EvaluateCubicHermiteSegment(
         states_[left], states_[right],
         derivatives_[left], derivatives_[right],
         time_step_, theta, state);
   }
};

// All data needed while the reverse sweep crosses one coarse forward
// interval. These vectors are scratch, not REVOLVE snapshot state.
struct ForwardIntervalData
{
   int index = -1;
   real_t t_left = 0.0;
   real_t dt = 0.0;
   Vector x_left;
   Vector x_right;
   Vector f_left;
   Vector f_right;
};

struct ForwardIntervalReplayWorkspace
{
   Vector k2, k3, k4;
   Vector y1, y2, y3;
};

inline void SetForwardIntervalFromEndpoints(
   ElastodynamicsOperator &oper,
   int interval_index,
   real_t t_left,
   real_t interval_size,
   const Vector &x_left,
   const Vector &x_right,
   ForwardIntervalData &interval)
{
   MFEM_VERIFY(interval_index >= 0 &&
               std::isfinite(t_left) &&
               std::isfinite(interval_size) && interval_size > 0.0,
               "Forward interval received invalid index/time data.");
   MFEM_VERIFY(x_left.Size() == oper.Width() &&
               x_right.Size() == oper.Width(),
               "Forward interval endpoint has an unexpected size.");

   interval.index = interval_index;
   interval.t_left = t_left;
   interval.dt = interval_size;
   interval.x_left = x_left;
   interval.x_right = x_right;
   EvalRHS(oper, interval.x_left, t_left, interval.f_left);
   EvalRHS(oper, interval.x_right, t_left + interval_size, interval.f_right);
}

// Reproduce one accepted classical-RK4 endpoint and build its physical
// endpoint slopes. In particular f_right is evaluated at the accepted state;
// RK4 k4 is only a stage slope and must never be substituted for it.
inline void ReplayForwardInterval(
   ElastodynamicsOperator &oper,
   int interval_index,
   real_t t_left,
   real_t interval_size,
   const Vector &x_left,
   ForwardIntervalReplayWorkspace &workspace,
   ForwardIntervalData &interval)
{
   MFEM_VERIFY(interval_index >= 0 &&
               std::isfinite(t_left) &&
               std::isfinite(interval_size) && interval_size > 0.0 &&
               std::isfinite(t_left + interval_size) &&
               t_left + interval_size > t_left,
               "Forward interval replay received invalid index/time data.");
   MFEM_VERIFY(x_left.Size() == oper.Width(),
               "Forward interval replay received an unexpected state size.");
   interval.index = interval_index;
   interval.t_left = t_left;
   interval.dt = interval_size;
   interval.x_left = x_left;

   RK4Stages(
      oper, interval.x_left, t_left, interval_size,
      interval.f_left, workspace.k2, workspace.k3, workspace.k4,
      workspace.y1, workspace.y2, workspace.y3);

   interval.x_right = interval.x_left;
   interval.x_right.Add(interval_size / 6.0, interval.f_left);
   interval.x_right.Add(interval_size / 3.0, workspace.k2);
   interval.x_right.Add(interval_size / 3.0, workspace.k3);
   interval.x_right.Add(interval_size / 6.0, workspace.k4);
   EvalRHS(
      oper, interval.x_right, t_left + interval_size, interval.f_right);
}

inline void AdvanceForwardInterval(
   ElastodynamicsOperator &oper,
   int interval_index,
   real_t t_left,
   real_t interval_size,
   Vector &state,
   ForwardIntervalReplayWorkspace &workspace,
   ForwardIntervalData &interval)
{
   ReplayForwardInterval(
      oper, interval_index, t_left, interval_size,
      state, workspace, interval);
   state = interval.x_right;
}

struct ForwardBlockData
{
   int index = -1;
   real_t t_left = 0.0;
   real_t dt_forward = 0.0;
   std::vector<Vector> states;
   std::vector<Vector> derivatives;
};

// Replay a block of accepted fine RK4 intervals from one checkpointed block
// endpoint.  Only this interval-local object retains the q+1 fine states; it is
// discarded immediately after the corresponding coarse adjoint step.
inline void ReplayForwardBlock(
   ElastodynamicsOperator &oper,
   int block_index,
   real_t t_left,
   real_t forward_step,
   int forward_steps_per_block,
   const Vector &x_left,
   ForwardIntervalReplayWorkspace &workspace,
   ForwardBlockData &block)
{
   MFEM_VERIFY(block_index >= 0 && forward_steps_per_block > 0 &&
               std::isfinite(t_left) &&
               std::isfinite(forward_step) && forward_step > 0.0 &&
               x_left.Size() == oper.Width(),
               "Forward block replay received invalid data.");
   block.index = block_index;
   block.t_left = t_left;
   block.dt_forward = forward_step;
   block.states.resize(forward_steps_per_block + 1);
   block.derivatives.resize(forward_steps_per_block + 1);
   block.states[0] = x_left;

   Vector state(x_left);
   ForwardIntervalData interval;
   for (int local_step = 0;
        local_step < forward_steps_per_block; local_step++)
   {
      const int global_step =
         block_index * forward_steps_per_block + local_step;
      const real_t interval_left = t_left + local_step * forward_step;
      AdvanceForwardInterval(
         oper, global_step, interval_left, forward_step,
         state, workspace, interval);
      if (local_step == 0) { block.derivatives[0] = interval.f_left; }
      block.states[local_step + 1] = state;
      block.derivatives[local_step + 1] = interval.f_right;
   }
}

inline void AdvanceForwardBlock(
   ElastodynamicsOperator &oper,
   int block_index,
   real_t t_left,
   real_t forward_step,
   int forward_steps_per_block,
   Vector &state,
   ForwardIntervalReplayWorkspace &workspace,
   ForwardIntervalData &scratch)
{
   MFEM_VERIFY(block_index >= 0 && forward_steps_per_block > 0,
               "Forward block advance received invalid data.");
   for (int local_step = 0;
        local_step < forward_steps_per_block; local_step++)
   {
      const int global_step =
         block_index * forward_steps_per_block + local_step;
      AdvanceForwardInterval(
         oper, global_step,
         t_left + local_step * forward_step, forward_step,
         state, workspace, scratch);
   }
}

// Non-owning provider valid only while one ForwardIntervalData object remains
// alive. This is the provider used by interval-local REVOLVE consumption.
class CubicHermiteForwardIntervalProvider : public ForwardStateProvider
{
private:
   const ForwardIntervalData &interval_;

public:
   explicit CubicHermiteForwardIntervalProvider(
      const ForwardIntervalData &interval)
      : interval_(interval)
   {
      const real_t t_right = interval_.t_left + interval_.dt;
      const real_t time_scale =
         std::max({real_t(1.0), std::abs(interval_.t_left),
                   std::abs(t_right)});
      const real_t time_resolution =
         256.0 * std::numeric_limits<real_t>::epsilon() * time_scale;
      MFEM_VERIFY(interval_.index >= 0 &&
                  std::isfinite(interval_.t_left) &&
                  std::isfinite(interval_.dt) &&
                  std::isfinite(t_right) &&
                  t_right > interval_.t_left &&
                  interval_.dt > 4.0 * time_resolution,
                  "Cubic Hermite interval provider received invalid data.");
   }

   void Evaluate(real_t physical_time, Vector &state) const override
   {
      MFEM_VERIFY(std::isfinite(physical_time),
                  "Cubic Hermite interval request time is non-finite.");
      const real_t t_right = interval_.t_left + interval_.dt;
      MFEM_VERIFY(std::isfinite(t_right) && t_right > interval_.t_left,
                  "Cubic Hermite interval has an unresolved right endpoint.");
      const real_t time_scale =
         std::max({real_t(1.0), std::abs(interval_.t_left),
                   std::abs(t_right), std::abs(physical_time)});
      const real_t tolerance =
         256.0 * std::numeric_limits<real_t>::epsilon() * time_scale;
      MFEM_VERIFY(physical_time >= interval_.t_left - tolerance &&
                  physical_time <= t_right + tolerance,
                  "Cubic Hermite request is outside its active interval.");

      if (std::abs(physical_time - interval_.t_left) <= tolerance)
      {
         state = interval_.x_left;
      }
      else if (std::abs(physical_time - t_right) <= tolerance)
      {
         state = interval_.x_right;
      }
      else
      {
         const real_t coordinate =
            (physical_time - interval_.t_left) / interval_.dt;
         MFEM_VERIFY(std::isfinite(coordinate),
                     "Cubic Hermite interval coordinate is non-finite.");
         const real_t theta =
            std::min(real_t(1.0), std::max(real_t(0.0), coordinate));
         EvaluateCubicHermiteSegment(
            interval_.x_left, interval_.x_right,
            interval_.f_left, interval_.f_right,
            interval_.dt, theta, state);
      }
   }
};

// Fourth-order Simpson/RK quadrature of the objective over one coarse forward
// interval. The subdivision matches the fine continuous-adjoint grid.
inline real_t EvaluateContinuousObjectiveInterval(
   ParFiniteElementSpace &state_fes,
   const Array<int> &offsets,
   TimeIntegratedObjective &objective,
   const ForwardStateProvider &forward_states,
   real_t t_left,
   real_t interval_size,
   int fine_steps_per_interval)
{
   MFEM_VERIFY(fine_steps_per_interval > 0 &&
               std::isfinite(t_left) &&
               std::isfinite(interval_size) && interval_size > 0.0,
               "Continuous objective interval has an invalid time grid.");

   const real_t fine_step = interval_size / fine_steps_per_interval;
   Vector x_left, x_mid, x_right;
   real_t value = 0.0;
   for (int substep = 0; substep < fine_steps_per_interval; substep++)
   {
      const real_t substep_left = t_left + substep * fine_step;
      const real_t substep_mid = substep_left + 0.5 * fine_step;
      const real_t substep_right = substep_left + fine_step;
      forward_states.Evaluate(substep_left, x_left);
      forward_states.Evaluate(substep_mid, x_mid);
      forward_states.Evaluate(substep_right, x_right);

      const real_t ell_left =
         InstantaneousObjectiveValueAtState(
            state_fes, offsets, objective, x_left, substep_left);
      const real_t ell_mid =
         InstantaneousObjectiveValueAtState(
            state_fes, offsets, objective, x_mid, substep_mid);
      const real_t ell_right =
         InstantaneousObjectiveValueAtState(
            state_fes, offsets, objective, x_right, substep_right);
      value += (fine_step / 6.0) *
               (ell_left + 4.0 * ell_mid + ell_right);
   }
   MFEM_VERIFY(std::isfinite(value),
               "Continuous objective quadrature is non-finite.");
   return value;
}

// Streaming fourth-order continuous objective. Endpoint states are exposed to
// an optional callback but are not retained here; full storage and forward-only
// visualization both build on this same accepted-RK4/Hermite sweep.
inline real_t ContinuousForwardSweepStreaming(
   ElastodynamicsOperator &oper,
   ParFiniteElementSpace &state_fes,
   TimeIntegratedObjective &objective,
   const Vector &initial_state,
   int forward_steps,
   real_t start_time,
   real_t forward_step,
   int fine_steps_per_interval,
   const char *progress_label = nullptr,
   const std::function<void(int, const Vector&)> &interval_complete = {})
{
   MFEM_VERIFY(forward_steps > 0 &&
               initial_state.Size() == oper.Width(),
               "Continuous streaming forward sweep has invalid inputs.");
   Vector state(initial_state);
   ForwardIntervalReplayWorkspace workspace;
   ForwardIntervalData interval;
   real_t objective_value = 0.0;
   const bool report = progress_label != nullptr;
   const double phase_t0 = MPI_Wtime();
   const int report_every = std::max(1, forward_steps / 10);

   for (int step = 0; step < forward_steps; step++)
   {
      const real_t t_left = start_time + step * forward_step;
      AdvanceForwardInterval(
         oper, step, t_left, forward_step, state, workspace, interval);
      CubicHermiteForwardIntervalProvider provider(interval);
      objective_value += EvaluateContinuousObjectiveInterval(
         state_fes, oper.GetBlockOffsets(), objective, provider,
         t_left, forward_step, fine_steps_per_interval);
      if (interval_complete) { interval_complete(step + 1, state); }

      if (report &&
          ((step + 1) % report_every == 0 || step + 1 == forward_steps))
      {
         const int n_disp = state.Size() / 2;
         real_t local_max_u = 0.0;
         int local_nonfinite = 0;
         for (int j = 0; j < state.Size(); j++)
         {
            if (!std::isfinite(state[j])) { local_nonfinite = 1; }
            if (j < n_disp)
            {
               local_max_u = std::max(local_max_u, std::abs(state[j]));
            }
         }
         real_t global_max_u = 0.0;
         int global_nonfinite = 0;
         MPI_Allreduce(&local_max_u, &global_max_u, 1,
                       MPITypeMap<real_t>::mpi_type, MPI_MAX,
                       state_fes.GetComm());
         MPI_Allreduce(&local_nonfinite, &global_nonfinite, 1, MPI_INT,
                       MPI_MAX, state_fes.GetComm());
         MFEM_VERIFY(global_nonfinite == 0,
                     "Continuous forward solve produced a non-finite state.");
         if (Mpi::Root())
         {
            std::cout << "      " << progress_label << ' '
                      << std::setw(6) << (step + 1) << '/' << forward_steps
                      << "  (" << std::setw(3)
                      << (100 * (step + 1) / forward_steps) << "%)   "
                      << std::fixed << std::setprecision(2)
                      << (MPI_Wtime() - phase_t0) << " s"
                      << "   max|u| = " << std::scientific
                      << std::setprecision(3) << global_max_u << "\n";
         }
      }
   }
   MFEM_VERIFY(std::isfinite(objective_value),
               "Continuous streaming forward objective is non-finite.");
   return objective_value;
}

// Forward sweep used by full-storage continuous-gradient verification. It uses
// the same explicit RK4 replay helper as the checkpointed path.
inline real_t ContinuousForwardSweepFullStorage(
   ElastodynamicsOperator &oper,
   ParFiniteElementSpace &state_fes,
   TimeIntegratedObjective &objective,
   const Vector &initial_state,
   int forward_steps,
   real_t start_time,
   real_t forward_step,
   int fine_steps_per_interval,
   std::vector<Vector> &states)
{
   states.resize(forward_steps + 1);
   states[0] = initial_state;
   const auto save_endpoint = [&](int endpoint, const Vector &state)
   {
      states[endpoint] = state;
   };
   return ContinuousForwardSweepStreaming(
      oper, state_fes, objective, initial_state,
      forward_steps, start_time, forward_step, fine_steps_per_interval,
      /*progress_label=*/nullptr, save_endpoint);
}

// Coarse-interval REVOLVE forward sweep. Only coarse endpoint states enter
// snapshots; Hermite slopes and objective midpoint states are interval scratch.
inline real_t ContinuousForwardSweepCheckpointed(
   ElastodynamicsOperator &oper,
   ParFiniteElementSpace &state_fes,
   TimeIntegratedObjective &objective,
   const Vector &initial_state,
   int forward_steps,
   real_t start_time,
   real_t forward_step,
   int fine_steps_per_interval,
   TrajectoryCheckpointing<> &checkpoint,
   Vector &final_state,
   const std::function<void(int, const Vector&)> &interval_complete = {})
{
   MFEM_VERIFY(checkpoint.NumSteps() == forward_steps &&
               initial_state.Size() == oper.Width(),
               "Continuous checkpointed forward sweep has invalid inputs.");
   checkpoint.Reset();
   final_state = initial_state;
   ForwardIntervalReplayWorkspace workspace;
   ForwardIntervalData interval;
   real_t objective_value = 0.0;

   for (int step = 0; step < forward_steps; step++)
   {
      const real_t t_left = start_time + step * forward_step;
      auto advance_interval = [&](int replay_step, Vector &state)
      {
         MFEM_VERIFY(replay_step == step,
                     "Initial checkpointed forward schedule changed step.");
         AdvanceForwardInterval(
            oper, replay_step, t_left, forward_step,
            state, workspace, interval);
      };
      checkpoint.ForwardStep(
         step, final_state, t_left, advance_interval);
      CubicHermiteForwardIntervalProvider provider(interval);
      objective_value += EvaluateContinuousObjectiveInterval(
         state_fes, oper.GetBlockOffsets(), objective, provider,
         t_left, forward_step, fine_steps_per_interval);
      if (interval_complete)
      {
         interval_complete(step + 1, final_state);
      }
   }
   MFEM_VERIFY(std::isfinite(objective_value),
               "Continuous checkpointed forward objective is non-finite.");
   return objective_value;
}

// Forward-finer checkpoint sweep. REVOLVE sees N_a coarse blocks, while each
// block callback advances q=N_f/N_a fine RK4 intervals and evaluates the
// objective on the fine forward grid.
inline real_t ContinuousForwardSweepBlockCheckpointed(
   ElastodynamicsOperator &oper,
   ParFiniteElementSpace &state_fes,
   TimeIntegratedObjective &objective,
   const Vector &initial_state,
   const NestedTimeGrid &grid,
   TrajectoryCheckpointing<> &checkpoint,
   Vector &final_state,
   const std::function<void(int, const Vector&)> &block_complete = {})
{
   MFEM_VERIFY(grid.relation == NestedTimeGridRelation::FORWARD_FINER &&
               checkpoint.NumSteps() == grid.adjoint_steps &&
               initial_state.Size() == oper.Width(),
               "Continuous block-checkpointed forward sweep has invalid data.");
   checkpoint.Reset();
   objective.Reset();
   final_state = initial_state;
   ForwardIntervalReplayWorkspace workspace;
   ForwardIntervalData interval;
   real_t objective_value = 0.0;

   for (int block = 0; block < grid.adjoint_steps; block++)
   {
      const real_t block_left = block * grid.dt_adjoint;
      auto advance_block = [&](int replay_block, Vector &state)
      {
         MFEM_VERIFY(replay_block == block,
                     "Initial checkpointed block schedule changed index.");
         for (int local_step = 0;
              local_step < grid.integer_ratio; local_step++)
         {
            const int global_step = block * grid.integer_ratio + local_step;
            const real_t interval_left =
               block_left + local_step * grid.dt_forward;
            AdvanceForwardInterval(
               oper, global_step, interval_left, grid.dt_forward,
               state, workspace, interval);
            CubicHermiteForwardIntervalProvider provider(interval);
            objective_value += EvaluateContinuousObjectiveInterval(
               state_fes, oper.GetBlockOffsets(), objective, provider,
               interval_left, grid.dt_forward,
               /*fine_steps_per_interval=*/1);
         }
      };
      checkpoint.ForwardStep(
         block, final_state, block_left, advance_block);
      if (block_complete) { block_complete(block + 1, final_state); }
   }
   MFEM_VERIFY(std::isfinite(objective_value),
               "Continuous block-checkpointed objective is non-finite.");
   return objective_value;
}

struct ForwardReconstructionAudit
{
   real_t state_relative_rms;
   real_t displacement_relative_rms;
   real_t velocity_relative_rms;
};

// Compare a supplied continuous reconstruction with a freshly integrated
// uniform fine-forward reference without storing the fine trajectory.  This
// audit measures the actual RK4-endpoint + Hermite reconstruction error, not
// merely the adjoint's convergence on a fixed reconstructed trajectory.
inline ForwardReconstructionAudit AuditForwardStateReconstruction(
   ElastodynamicsOperator &oper,
   const ForwardStateProvider &forward_states,
   const Vector &initial_state,
   real_t start_time,
   int reference_steps,
   real_t reference_step,
   MPI_Comm comm)
{
   MFEM_VERIFY(reference_steps > 0 && reference_step > 0.0,
               "Forward reconstruction audit needs a positive reference grid.");
   MFEM_VERIFY(std::isfinite(start_time) &&
               std::isfinite(reference_step),
               "Forward reconstruction audit needs finite time data.");
   MFEM_VERIFY(initial_state.Size() == oper.Width() &&
               initial_state.Size() % 2 == 0,
               "Forward reconstruction audit received an invalid state.");

   const int displacement_size = initial_state.Size() / 2;
   Vector reference(initial_state), reconstructed(initial_state.Size());
   real_t local_sums[4] = {0.0, 0.0, 0.0, 0.0};
   // [0] displacement error^2, [1] displacement reference^2,
   // [2] velocity error^2,     [3] velocity reference^2.
   const auto accumulate = [&](real_t physical_time)
   {
      forward_states.Evaluate(physical_time, reconstructed);
      for (int i = 0; i < displacement_size; i++)
      {
         const real_t error = reconstructed[i] - reference[i];
         local_sums[0] += error * error;
         local_sums[1] += reference[i] * reference[i];
      }
      for (int i = displacement_size; i < reference.Size(); i++)
      {
         const real_t error = reconstructed[i] - reference[i];
         local_sums[2] += error * error;
         local_sums[3] += reference[i] * reference[i];
      }
   };

   RK4Solver solver;
   solver.Init(oper);
   real_t time = start_time;
   accumulate(time);
   for (int step = 0; step < reference_steps; step++)
   {
      real_t dt = reference_step;
      solver.Step(reference, time, dt);
      // Avoid accumulating a long sequence of floating-point time additions;
      // provider interval lookup is defined by the uniform reference index.
      time = start_time + (step + 1) * reference_step;
      accumulate(time);
   }

   real_t global_sums[4] = {0.0, 0.0, 0.0, 0.0};
   MPI_Allreduce(local_sums, global_sums, 4,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
   const real_t total_reference_norm_sq =
      global_sums[1] + global_sums[3];
   MFEM_VERIFY(total_reference_norm_sq > 0.0,
               "Forward reconstruction audit has a zero reference state.");

   ForwardReconstructionAudit audit;
   audit.displacement_relative_rms =
      (global_sums[1] > 0.0) ?
      std::sqrt(global_sums[0] / global_sums[1]) :
      ((global_sums[0] == 0.0) ? 0.0 :
       std::numeric_limits<real_t>::infinity());
   audit.velocity_relative_rms =
      (global_sums[3] > 0.0) ?
      std::sqrt(global_sums[2] / global_sums[3]) :
      ((global_sums[2] == 0.0) ? 0.0 :
       std::numeric_limits<real_t>::infinity());
   audit.state_relative_rms =
      std::sqrt((global_sums[0] + global_sums[2]) /
                total_reference_norm_sq);
   MFEM_VERIFY(std::isfinite(audit.state_relative_rms) &&
               std::isfinite(audit.displacement_relative_rms) &&
               std::isfinite(audit.velocity_relative_rms),
               "Forward reconstruction audit produced a non-finite error.");
   return audit;
}

struct ContinuousDesignGradientData
{
   ParFiniteElementSpace &filter_fes;
   ParGridFunction &rho_tilde;
   const MaterialParams &material;
   Vector &gradient_tilde;
};

// Replay work that is outside the original forward sweep. Controller replay is
// selected by REVOLVE. Local replay is the one private coarse advance used to
// regenerate each interval's accepted right endpoint and Hermite slopes.
struct ContinuousReplayStatistics
{
   long long controller_replayed_blocks = 0;
   long long locally_replayed_blocks = 0;
   long long controller_replayed_intervals = 0;
   long long locally_replayed_intervals = 0;
};

inline void ReverseTimeContinuousAdjointRK4Step(
   ElastodynamicsOperator &oper,
   ParFiniteElementSpace &state_fes,
   TimeIntegratedObjective &objective,
   const ForwardStateProvider &forward_states,
   const Vector &p_right,
   real_t t_right,
   real_t step_size,
   Vector &p_left,
   ContinuousDesignGradientData *design_gradient = nullptr)
{
   MFEM_VERIFY(step_size > 0.0,
               "Continuous-adjoint RK4 step size must be positive.");
   MFEM_VERIFY(p_right.Size() == oper.Height(),
               "Continuous-adjoint RK4 step received an invalid adjoint size.");
   const int n = p_right.Size();

   Vector x_stage(n), q(n), jt(n), forward_rhs(n);
   Vector k1(n), k2(n), k3(n), k4(n), p_stage(n);

   const auto evaluate_rhs =
      [&](real_t physical_time, const Vector &p, Vector &rhs)
      {
         forward_states.Evaluate(physical_time, x_stage);
         EvalJacobianTranspose(oper, x_stage, physical_time, p, jt);
         InstantaneousObjectiveGradientAtState(
            state_fes, oper.GetBlockOffsets(), objective,
            x_stage, physical_time, q);
         rhs = jt;
         rhs += q;
      };
   const auto accumulate_design =
      [&](real_t physical_time,
          const Vector &p_stage_value,
          real_t temporal_weight)
      {
         if (!design_gradient) { return; }
         // The second block of the physical RHS is the acceleration required
         // by the mass-design contraction. Never differentiate Hermite data.
         EvalRHS(oper, x_stage, physical_time, forward_rhs);
         AddStageDesignGradientTilde(
            oper, state_fes, design_gradient->filter_fes,
            design_gradient->rho_tilde, design_gradient->material,
            x_stage, forward_rhs, p_stage_value,
            design_gradient->gradient_tilde, temporal_weight);
      };

   // March with reverse coordinate s=T-t:
   //   dp/ds = f_x(x,t)^T p + ell_x(x,t).
   // physical time therefore decreases across the positive RK4 step.
   evaluate_rhs(t_right, p_right, k1);
   accumulate_design(t_right, p_right, step_size / 6.0);

   p_stage = p_right;
   p_stage.Add(0.5 * step_size, k1);
   evaluate_rhs(t_right - 0.5 * step_size, p_stage, k2);
   accumulate_design(
      t_right - 0.5 * step_size, p_stage, step_size / 3.0);

   p_stage = p_right;
   p_stage.Add(0.5 * step_size, k2);
   evaluate_rhs(t_right - 0.5 * step_size, p_stage, k3);
   accumulate_design(
      t_right - 0.5 * step_size, p_stage, step_size / 3.0);

   p_stage = p_right;
   p_stage.Add(step_size, k3);
   evaluate_rhs(t_right - step_size, p_stage, k4);
   accumulate_design(
      t_right - step_size, p_stage, step_size / 6.0);

   p_left = p_right;
   p_left.Add(step_size / 6.0, k1);
   p_left.Add(step_size / 3.0, k2);
   p_left.Add(step_size / 3.0, k3);
   p_left.Add(step_size / 6.0, k4);
}

// Physical-time derivative of the continuous adjoint.  The reverse RK4
// kernel marches in s = T-t with dp/ds = f_x^T p + ell_x, hence
// dp/dt = -(f_x^T p + ell_x).  Endpoint values of this derivative provide
// fourth-order cubic-Hermite dense output for a coarser adjoint trajectory.
inline void EvaluateContinuousAdjointPhysicalTimeDerivative(
   ElastodynamicsOperator &oper,
   ParFiniteElementSpace &state_fes,
   TimeIntegratedObjective &objective,
   const ForwardStateProvider &forward_states,
   const Vector &adjoint,
   real_t physical_time,
   Vector &physical_derivative)
{
   MFEM_VERIFY(adjoint.Size() == oper.Height() &&
               std::isfinite(physical_time),
               "Continuous-adjoint derivative received invalid data.");
   Vector state, jacobian_transpose, objective_gradient;
   forward_states.Evaluate(physical_time, state);
   EvalJacobianTranspose(
      oper, state, physical_time, adjoint, jacobian_transpose);
   InstantaneousObjectiveGradientAtState(
      state_fes, oper.GetBlockOffsets(), objective,
      state, physical_time, objective_gradient);
   physical_derivative = jacobian_transpose;
   physical_derivative += objective_gradient;
   physical_derivative *= -1.0;
}

inline void SetContinuousAdjointDenseInterval(
   ElastodynamicsOperator &oper,
   ParFiniteElementSpace &state_fes,
   TimeIntegratedObjective &objective,
   const ForwardStateProvider &forward_states,
   int interval_index,
   real_t t_left,
   real_t interval_size,
   const Vector &p_left,
   const Vector &p_right,
   ForwardIntervalData &adjoint_interval)
{
   MFEM_VERIFY(interval_index >= 0 &&
               std::isfinite(t_left) &&
               std::isfinite(interval_size) && interval_size > 0.0 &&
               p_left.Size() == oper.Height() &&
               p_right.Size() == oper.Height(),
               "Continuous-adjoint dense interval received invalid data.");
   adjoint_interval.index = interval_index;
   adjoint_interval.t_left = t_left;
   adjoint_interval.dt = interval_size;
   adjoint_interval.x_left = p_left;
   adjoint_interval.x_right = p_right;
   EvaluateContinuousAdjointPhysicalTimeDerivative(
      oper, state_fes, objective, forward_states,
      p_left, t_left, adjoint_interval.f_left);
   EvaluateContinuousAdjointPhysicalTimeDerivative(
      oper, state_fes, objective, forward_states,
      p_right, t_left + interval_size, adjoint_interval.f_right);
}

// Fourth-order design quadrature on one interval of the finest state grid.
// This is required when the forward grid is finer than the adjoint grid: the
// adjoint is reconstructed from its accepted coarse endpoints, but the mass
// and stiffness contraction is still integrated where the forward trajectory
// is resolved.  A unique dense state at the midpoint gives Simpson weights.
inline void AccumulateContinuousDesignQuadratureInterval(
   ElastodynamicsOperator &oper,
   ParFiniteElementSpace &state_fes,
   const ForwardStateProvider &forward_states,
   const ForwardStateProvider &adjoint_states,
   real_t t_left,
   real_t interval_size,
   ContinuousDesignGradientData &design_gradient)
{
   MFEM_VERIFY(std::isfinite(t_left) &&
               std::isfinite(interval_size) && interval_size > 0.0,
               "Continuous design quadrature received an invalid interval.");
   Vector state, adjoint, forward_rhs;
   const real_t times[] =
   {
      t_left,
      t_left + 0.5 * interval_size,
      t_left + interval_size
   };
   const real_t weights[] =
   {
      interval_size / 6.0,
      2.0 * interval_size / 3.0,
      interval_size / 6.0
   };
   for (int stage = 0; stage < 3; stage++)
   {
      forward_states.Evaluate(times[stage], state);
      adjoint_states.Evaluate(times[stage], adjoint);
      EvalRHS(oper, state, times[stage], forward_rhs);
      AddStageDesignGradientTilde(
         oper, state_fes, design_gradient.filter_fes,
         design_gradient.rho_tilde, design_gradient.material,
         state, forward_rhs, adjoint,
         design_gradient.gradient_tilde, weights[stage]);
   }
}

// Shared forward-finer reverse block.  One coarse adjoint RK4 step spans a
// block of fine forward intervals.  Its accepted endpoints define a dense
// adjoint, after which the design integral is accumulated on every fine
// forward interval in the block.
inline void ReverseContinuousAdjointForwardFinerBlock(
   ElastodynamicsOperator &oper,
   ParFiniteElementSpace &state_fes,
   TimeIntegratedObjective &objective,
   const ForwardStateProvider &forward_states,
   int block_index,
   real_t t_left,
   real_t block_size,
   int forward_steps_per_block,
   const Vector &p_right,
   Vector &p_left,
   ContinuousDesignGradientData &design_gradient)
{
   MFEM_VERIFY(block_index >= 0 && forward_steps_per_block > 1 &&
               std::isfinite(t_left) &&
               std::isfinite(block_size) && block_size > 0.0,
               "Forward-finer reverse block received invalid data.");
   ReverseTimeContinuousAdjointRK4Step(
      oper, state_fes, objective, forward_states,
      p_right, t_left + block_size, block_size, p_left);

   ForwardIntervalData adjoint_interval;
   SetContinuousAdjointDenseInterval(
      oper, state_fes, objective, forward_states,
      block_index, t_left, block_size,
      p_left, p_right, adjoint_interval);
   CubicHermiteForwardIntervalProvider adjoint_provider(adjoint_interval);

   const real_t forward_step = block_size / forward_steps_per_block;
   for (int local_step = forward_steps_per_block - 1;
        local_step >= 0; local_step--)
   {
      AccumulateContinuousDesignQuadratureInterval(
         oper, state_fes, forward_states, adjoint_provider,
         t_left + local_step * forward_step, forward_step,
         design_gradient);
   }
}

// Shared reverse consumer for one coarse forward interval. Full storage and
// REVOLVE differ only in how they construct the supplied local provider.
inline void ReverseContinuousAdjointInterval(
   ElastodynamicsOperator &oper,
   ParFiniteElementSpace &state_fes,
   TimeIntegratedObjective &objective,
   const ForwardStateProvider &forward_states,
   real_t t_left,
   real_t forward_step,
   int fine_steps_per_interval,
   const Vector &p_right,
   Vector &p_left,
   ContinuousDesignGradientData *design_gradient = nullptr)
{
   MFEM_VERIFY(fine_steps_per_interval > 0 &&
               std::isfinite(t_left) &&
               std::isfinite(forward_step) && forward_step > 0.0,
               "Continuous reverse interval has an invalid time grid.");
   const real_t adjoint_step =
      forward_step / fine_steps_per_interval;
   Vector p(p_right), next(p_right.Size());
   for (int substep = fine_steps_per_interval - 1;
        substep >= 0; substep--)
   {
      const real_t t_right =
         t_left + (substep + 1) * adjoint_step;
      ReverseTimeContinuousAdjointRK4Step(
         oper, state_fes, objective, forward_states,
         p, t_right, adjoint_step, next, design_gradient);
      p = next;
   }
   p_left = p;
}

inline void ContinuousAdjointDesignSweepFullStorage(
   ElastodynamicsOperator &oper,
   ParFiniteElementSpace &state_fes,
   ParFiniteElementSpace &filter_fes,
   ParGridFunction &rho_tilde,
   const MaterialParams &material,
   TimeIntegratedObjective &objective,
   const std::vector<Vector> &forward_states,
   const NestedTimeGrid &grid,
   const Vector &terminal_adjoint,
   Vector &initial_adjoint,
   Vector &gradient_tilde)
{
   MFEM_VERIFY(
      forward_states.size() ==
      static_cast<std::size_t>(grid.forward_steps + 1),
      "Continuous design sweep received the wrong forward trajectory.");
   MFEM_VERIFY(terminal_adjoint.Size() == oper.Height(),
               "Continuous design sweep received an invalid terminal adjoint.");

   gradient_tilde.SetSize(filter_fes.GetTrueVSize());
   gradient_tilde = 0.0;
   ContinuousDesignGradientData design_data{
      filter_fes, rho_tilde, material, gradient_tilde};

   if (grid.relation == NestedTimeGridRelation::FORWARD_FINER)
   {
      std::vector<Vector> forward_derivatives;
      BuildForwardStateTimeDerivatives(
         oper, forward_states, /*start_time=*/0.0,
         grid.dt_forward, forward_derivatives);
      CubicHermiteForwardStateProvider forward_provider(
         forward_states, forward_derivatives,
         /*start_time=*/0.0, grid.dt_forward);

      Vector p(terminal_adjoint), p_left(terminal_adjoint.Size());
      for (int adjoint_step = grid.adjoint_steps - 1;
           adjoint_step >= 0; adjoint_step--)
      {
         const real_t t_left = adjoint_step * grid.dt_adjoint;
         ReverseContinuousAdjointForwardFinerBlock(
            oper, state_fes, objective, forward_provider,
            adjoint_step, t_left, grid.dt_adjoint,
            grid.integer_ratio, p, p_left, design_data);
         p = p_left;
      }
      initial_adjoint = p;
      return;
   }

   const int fine_steps_per_interval =
      grid.relation == NestedTimeGridRelation::SAME ?
      1 : grid.integer_ratio;
   Vector p(terminal_adjoint), p_left(terminal_adjoint.Size());
   ForwardIntervalData interval;
   for (int step = grid.forward_steps - 1; step >= 0; step--)
   {
      const real_t t_left = step * grid.dt_forward;
      SetForwardIntervalFromEndpoints(
         oper, step, t_left, grid.dt_forward,
         forward_states[step], forward_states[step + 1], interval);
      CubicHermiteForwardIntervalProvider provider(interval);
      ReverseContinuousAdjointInterval(
         oper, state_fes, objective, provider,
         t_left, grid.dt_forward, fine_steps_per_interval,
         p, p_left, &design_data);
      p = p_left;
   }
   initial_adjoint = p;
}

inline void ContinuousAdjointDesignSweepCheckpointed(
   ElastodynamicsOperator &oper,
   ParFiniteElementSpace &state_fes,
   ParFiniteElementSpace &filter_fes,
   ParGridFunction &rho_tilde,
   const MaterialParams &material,
   TimeIntegratedObjective &objective,
   const NestedTimeGrid &grid,
   TrajectoryCheckpointing<> &checkpoint,
   const Vector &forward_final_state,
   const Vector &terminal_adjoint,
   Vector &initial_adjoint,
   Vector &gradient_tilde,
   ContinuousReplayStatistics *replay_statistics = nullptr,
   const std::function<void(int)> &interval_complete = {},
   const std::function<void(
      int, const Vector&, const ForwardIntervalData&)> &replay_audit = {})
{
   const int scheduled_blocks =
      grid.relation == NestedTimeGridRelation::FORWARD_FINER ?
      grid.adjoint_steps : grid.forward_steps;
   MFEM_VERIFY(checkpoint.NumSteps() == scheduled_blocks &&
               forward_final_state.Size() == oper.Width() &&
               terminal_adjoint.Size() == oper.Height(),
               "Checkpointed continuous design sweep has invalid inputs.");

   const int fine_steps_per_interval =
      grid.relation == NestedTimeGridRelation::SAME ?
      1 : grid.integer_ratio;
   gradient_tilde.SetSize(filter_fes.GetTrueVSize());
   gradient_tilde = 0.0;
   ContinuousDesignGradientData design_data{
      filter_fes, rho_tilde, material, gradient_tilde};
   if (replay_statistics)
   {
      *replay_statistics = ContinuousReplayStatistics{};
   }

   if (grid.relation == NestedTimeGridRelation::FORWARD_FINER)
   {
      Vector p(terminal_adjoint), p_left(terminal_adjoint.Size());
      Vector forward_work(forward_final_state);
      ForwardIntervalReplayWorkspace controller_workspace;
      ForwardIntervalReplayWorkspace block_workspace;
      ForwardIntervalData controller_scratch;
      ForwardBlockData block_data;

      auto replay_forward_block = [&](int block, Vector &state)
      {
         if (replay_statistics)
         {
            replay_statistics->controller_replayed_blocks++;
            replay_statistics->controller_replayed_intervals +=
               grid.integer_ratio;
         }
         AdvanceForwardBlock(
            oper, block, block * grid.dt_adjoint,
            grid.dt_forward, grid.integer_ratio,
            state, controller_workspace, controller_scratch);
      };
      auto consume_reverse_block =
         [&](int block, const Vector &state_left, Vector &adjoint)
         {
            if (replay_statistics)
            {
               replay_statistics->locally_replayed_blocks++;
               replay_statistics->locally_replayed_intervals +=
                  grid.integer_ratio;
            }
            const real_t block_left = block * grid.dt_adjoint;
            ReplayForwardBlock(
               oper, block, block_left, grid.dt_forward,
               grid.integer_ratio, state_left,
               block_workspace, block_data);

            if (replay_audit)
            {
               ForwardIntervalData audit_interval;
               for (int local_step = 0;
                    local_step < grid.integer_ratio; local_step++)
               {
                  const int global_step =
                     block * grid.integer_ratio + local_step;
                  audit_interval.index = global_step;
                  audit_interval.t_left =
                     block_left + local_step * grid.dt_forward;
                  audit_interval.dt = grid.dt_forward;
                  audit_interval.x_left = block_data.states[local_step];
                  audit_interval.x_right = block_data.states[local_step + 1];
                  audit_interval.f_left =
                     block_data.derivatives[local_step];
                  audit_interval.f_right =
                     block_data.derivatives[local_step + 1];
                  replay_audit(global_step, audit_interval.x_left,
                               audit_interval);
               }
            }

            CubicHermiteForwardStateProvider forward_provider(
               block_data.states, block_data.derivatives,
               block_left, grid.dt_forward);
            ReverseContinuousAdjointForwardFinerBlock(
               oper, state_fes, objective, forward_provider,
               block, block_left, grid.dt_adjoint,
               grid.integer_ratio, adjoint, p_left, design_data);
            adjoint = p_left;
         };

      for (int block = grid.adjoint_steps - 1; block >= 0; block--)
      {
         checkpoint.BackwardInterval(
            block, p, forward_work,
            replay_forward_block, consume_reverse_block);
         if (interval_complete)
         {
            interval_complete(grid.adjoint_steps - block);
         }
      }
      initial_adjoint = p;
      return;
   }

   Vector p(terminal_adjoint), p_left(terminal_adjoint.Size());
   Vector forward_work(forward_final_state);
   ForwardIntervalReplayWorkspace controller_workspace;
   ForwardIntervalReplayWorkspace interval_workspace;
   ForwardIntervalData controller_scratch, interval;

   auto replay_coarse_step = [&](int step, Vector &state)
   {
      if (replay_statistics)
      {
         replay_statistics->controller_replayed_blocks++;
         replay_statistics->controller_replayed_intervals++;
      }
      const real_t t_left = step * grid.dt_forward;
      AdvanceForwardInterval(
         oper, step, t_left, grid.dt_forward,
         state, controller_workspace, controller_scratch);
   };
   auto consume_reverse_interval =
      [&](int step, const Vector &state_left, Vector &adjoint)
      {
         if (replay_statistics)
         {
            replay_statistics->locally_replayed_blocks++;
            replay_statistics->locally_replayed_intervals++;
         }
         const real_t t_left = step * grid.dt_forward;
         // Private replay: do not mutate REVOLVE's state_left or its tracked
         // work index. The interval data dies before this callback returns.
         ReplayForwardInterval(
            oper, step, t_left, grid.dt_forward,
            state_left, interval_workspace, interval);
         if (replay_audit)
         {
            replay_audit(step, state_left, interval);
         }
         CubicHermiteForwardIntervalProvider provider(interval);
         ReverseContinuousAdjointInterval(
            oper, state_fes, objective, provider,
            t_left, grid.dt_forward, fine_steps_per_interval,
            adjoint, p_left, &design_data);
         adjoint = p_left;
      };

   for (int step = grid.forward_steps - 1; step >= 0; step--)
   {
      checkpoint.BackwardInterval(
         step, p, forward_work,
         replay_coarse_step, consume_reverse_interval);
      if (interval_complete)
      {
         interval_complete(grid.forward_steps - step);
      }
   }
   initial_adjoint = p;
}

struct ContinuousDesignRunResult
{
   real_t objective = 0.0;
   Vector initial_adjoint;
   Vector gradient_tilde;
   long long controller_recomputed_blocks = 0;
   long long locally_replayed_blocks = 0;
   long long controller_recomputed_intervals = 0;
   long long locally_replayed_intervals = 0;
};

inline ContinuousDesignRunResult
RunContinuousDesignFullStorage(
   ElastodynamicsOperator &oper,
   ParFiniteElementSpace &state_fes,
   ParFiniteElementSpace &filter_fes,
   ParGridFunction &rho_tilde,
   const MaterialParams &material,
   TimeIntegratedObjective &objective,
   const Vector &initial_state,
   const NestedTimeGrid &grid)
{
   MFEM_VERIFY(grid.forward_steps > 0 && grid.adjoint_steps > 0 &&
               initial_state.Size() == oper.Width(),
               "Continuous design run has an invalid time grid or state.");
   const int objective_substeps =
      grid.relation == NestedTimeGridRelation::ADJOINT_FINER ?
      grid.integer_ratio : 1;

   std::vector<Vector> states;
   ContinuousDesignRunResult result;
   result.objective = ContinuousForwardSweepFullStorage(
      oper, state_fes, objective, initial_state,
      grid.forward_steps, /*start_time=*/0.0, grid.dt_forward,
      objective_substeps, states);

   Vector terminal(initial_state.Size());
   terminal = 0.0;
   ContinuousAdjointDesignSweepFullStorage(
      oper, state_fes, filter_fes, rho_tilde, material, objective,
      states, grid, terminal,
      result.initial_adjoint, result.gradient_tilde);
   return result;
}

inline ContinuousDesignRunResult
RunContinuousDesignFullStorage(
   ElastodynamicsOperator &oper,
   ParFiniteElementSpace &state_fes,
   ParFiniteElementSpace &filter_fes,
   ParGridFunction &rho_tilde,
   const MaterialParams &material,
   TimeIntegratedObjective &objective,
   const Vector &initial_state,
   int forward_steps,
   real_t forward_step,
   int adjoint_refinement)
{
   MFEM_VERIFY(forward_steps > 0 && forward_step > 0.0 &&
               adjoint_refinement > 0 &&
               initial_state.Size() == oper.Width(),
               "Continuous design run has invalid controls or initial state.");
   MFEM_VERIFY(
      forward_steps <=
      std::numeric_limits<int>::max() / adjoint_refinement,
      "Continuous design run has too many adjoint steps.");
   const NestedTimeGrid grid = NestedTimeGrid::Create(
      forward_steps * forward_step,
      forward_steps, forward_steps * adjoint_refinement);
   return RunContinuousDesignFullStorage(
      oper, state_fes, filter_fes, rho_tilde, material, objective,
      initial_state, grid);
}

inline ContinuousDesignRunResult
RunContinuousDesignCheckpointed(
   ElastodynamicsOperator &oper,
   ParFiniteElementSpace &state_fes,
   ParFiniteElementSpace &filter_fes,
   ParGridFunction &rho_tilde,
   const MaterialParams &material,
   TimeIntegratedObjective &objective,
   const Vector &initial_state,
   const NestedTimeGrid &grid,
   int num_checkpoints,
   const std::function<void(
      int, const Vector&, const ForwardIntervalData&)> &replay_audit = {})
{
   MFEM_VERIFY(grid.forward_steps > 0 && grid.adjoint_steps > 0 &&
               num_checkpoints > 0 &&
               initial_state.Size() == oper.Width(),
               "Checkpointed continuous design run has invalid inputs.");
   const bool forward_finer =
      grid.relation == NestedTimeGridRelation::FORWARD_FINER;
   const int scheduled_blocks =
      forward_finer ? grid.adjoint_steps : grid.forward_steps;
   const real_t scheduled_step =
      forward_finer ? grid.dt_adjoint : grid.dt_forward;
   TrajectoryCheckpointing<> checkpoint(
      scheduled_blocks, num_checkpoints, initial_state.Size(),
      /*start_time=*/0.0, scheduled_step);

   ContinuousDesignRunResult result;
   Vector final_state;
   if (forward_finer)
   {
      result.objective = ContinuousForwardSweepBlockCheckpointed(
         oper, state_fes, objective, initial_state,
         grid, checkpoint, final_state);
   }
   else
   {
      const int objective_substeps =
         grid.relation == NestedTimeGridRelation::ADJOINT_FINER ?
         grid.integer_ratio : 1;
      result.objective = ContinuousForwardSweepCheckpointed(
         oper, state_fes, objective, initial_state,
         grid.forward_steps, 0.0, grid.dt_forward, objective_substeps,
         checkpoint, final_state);
   }

   Vector terminal(initial_state.Size());
   terminal = 0.0;
   ContinuousReplayStatistics replay_statistics;
   ContinuousAdjointDesignSweepCheckpointed(
      oper, state_fes, filter_fes, rho_tilde, material, objective,
      grid, checkpoint, final_state, terminal,
      result.initial_adjoint, result.gradient_tilde, &replay_statistics,
      /*interval_complete=*/{}, replay_audit);
   result.controller_recomputed_blocks =
      replay_statistics.controller_replayed_blocks;
   result.locally_replayed_blocks =
      replay_statistics.locally_replayed_blocks;
   result.controller_recomputed_intervals =
      replay_statistics.controller_replayed_intervals;
   result.locally_replayed_intervals =
      replay_statistics.locally_replayed_intervals;
   MFEM_VERIFY(
      result.controller_recomputed_blocks ==
      checkpoint.EstimateRecomputations(),
      "Actual REVOLVE block replay count differs from the controller estimate.");
   MFEM_VERIFY(
      result.locally_replayed_blocks == scheduled_blocks &&
      result.locally_replayed_intervals == grid.forward_steps,
      "Continuous reverse sweep did not replay exactly one local copy of "
      "each scheduled block and fine forward interval.");
   return result;
}

inline ContinuousDesignRunResult
RunContinuousDesignCheckpointed(
   ElastodynamicsOperator &oper,
   ParFiniteElementSpace &state_fes,
   ParFiniteElementSpace &filter_fes,
   ParGridFunction &rho_tilde,
   const MaterialParams &material,
   TimeIntegratedObjective &objective,
   const Vector &initial_state,
   int forward_steps,
   real_t forward_step,
   int adjoint_refinement,
   int num_checkpoints,
   const std::function<void(
      int, const Vector&, const ForwardIntervalData&)> &replay_audit = {})
{
   MFEM_VERIFY(adjoint_refinement > 0 && num_checkpoints > 0,
               "Checkpointed continuous design run has invalid controls.");
   MFEM_VERIFY(
      forward_steps <=
      std::numeric_limits<int>::max() / adjoint_refinement,
      "Checkpointed continuous design run has too many adjoint steps.");
   const NestedTimeGrid grid = NestedTimeGrid::Create(
      forward_steps * forward_step,
      forward_steps, forward_steps * adjoint_refinement);
   return RunContinuousDesignCheckpointed(
      oper, state_fes, filter_fes, rho_tilde, material, objective,
      initial_state, grid, num_checkpoints, replay_audit);
}

inline void ContinuousAdjointSweepFullStorage(
   ElastodynamicsOperator &oper,
   ParFiniteElementSpace &state_fes,
   TimeIntegratedObjective &objective,
   const ForwardStateProvider &forward_states,
   const NestedTimeGrid &grid,
   const Vector &terminal_adjoint,
   Vector &initial_adjoint,
   std::vector<Vector> *adjoint_nodes = nullptr)
{
   MFEM_VERIFY(terminal_adjoint.Size() == oper.Height(),
               "Continuous-adjoint terminal seed has an unexpected size.");

   Vector p(terminal_adjoint), p_left(terminal_adjoint.Size());
   if (adjoint_nodes)
   {
      adjoint_nodes->resize(grid.adjoint_steps + 1);
      for (Vector &node : *adjoint_nodes)
      {
         node.SetSize(terminal_adjoint.Size());
      }
      (*adjoint_nodes)[grid.adjoint_steps] = p;
   }

   for (int step = grid.adjoint_steps - 1; step >= 0; step--)
   {
      const real_t t_right = (step + 1) * grid.dt_adjoint;
      ReverseTimeContinuousAdjointRK4Step(
         oper, state_fes, objective, forward_states,
         p, t_right, grid.dt_adjoint, p_left);
      p = p_left;
      if (adjoint_nodes) { (*adjoint_nodes)[step] = p; }
   }
   initial_adjoint = p;
}

inline real_t GlobalVectorNorm(MPI_Comm comm, const Vector &vector)
{
   const real_t local_norm_sq = vector * vector;
   real_t global_norm_sq = 0.0;
   MPI_Allreduce(&local_norm_sq, &global_norm_sq, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
   MFEM_VERIFY(std::isfinite(global_norm_sq) && global_norm_sq >= 0.0,
               "Global vector norm encountered a non-finite state.");
   return std::sqrt(global_norm_sq);
}

inline real_t GlobalVectorDot(MPI_Comm comm,
                              const Vector &left,
                              const Vector &right)
{
   MFEM_VERIFY(left.Size() == right.Size(),
               "Global vector dot product received mismatched sizes.");
   const real_t local_dot = left * right;
   real_t global_dot = 0.0;
   MPI_Allreduce(&local_dot, &global_dot, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
   MFEM_VERIFY(std::isfinite(global_dot),
               "Global vector dot product is non-finite.");
   return global_dot;
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

   // Progress monitoring, throttled to ~10 lines per sweep. NOTE: the report
   // branch below contains an MPI_Allreduce, so ALL ranks must take it
   // identically (guarding it with WorldRank()==0 deadlocks: rank 0 waits in
   // the Allreduce while the others run ahead into the next step's matvec).
   const bool report = (progress_label != nullptr);
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

   AddObjectiveContributionAtTime(
      state_fes, offsets, objective, x, t, h, 0, total_steps);

   for (int i = 0; i < nsteps; i++)
   {
      real_t dt = h;
      solver.Step(x, t, dt);

      AddObjectiveContributionAtTime(
         state_fes, offsets, objective, x, t, h, i + 1, total_steps);

      if (states) { (*states)[i + 1] = x; }
      if (times)  { (*times)[i + 1] = t; }

      if (report && ((i + 1) % report_every == 0 || i + 1 == nsteps))
      {
         // max|u| tracks pulse growth/decay and flags instability at a glance.
         // Collective on all ranks; only root prints.
         const int n_disp = x.Size() / 2;
         real_t local_max_u = 0.0;
         int local_nonfinite = 0;
         for (int j = 0; j < x.Size(); j++)
         {
            if (!std::isfinite(x[j])) { local_nonfinite = 1; }
            if (j < n_disp)
            {
               local_max_u = std::max(local_max_u, std::abs(x[j]));
            }
         }
         real_t global_max_u = 0.0;
         int global_nonfinite = 0;
         MPI_Allreduce(&local_max_u, &global_max_u, 1,
                       MPITypeMap<real_t>::mpi_type, MPI_MAX,
                       state_fes.GetComm());
         MPI_Allreduce(&local_nonfinite, &global_nonfinite, 1, MPI_INT, MPI_MAX,
                       state_fes.GetComm());

         MFEM_VERIFY(global_nonfinite == 0,
                     "Forward solve produced a non-finite state.");

         if (Mpi::Root())
         {
            std::cout << "      " << progress_label << ' '
                      << std::setw(6) << (i + 1) << '/' << nsteps
                      << "  (" << std::setw(3) << (100 * (i + 1) / nsteps)
                      << "%)   " << std::fixed << std::setprecision(2)
                      << (MPI_Wtime() - phase_t0) << " s"
                      << "   max|u| = " << std::scientific
                      << std::setprecision(3) << global_max_u << "\n";
         }
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
                                      MassSolverType mass_type = MassSolverType::LUMPED,
                                      const char *progress_label = nullptr,
                                      bool validate_cfl = false,
                                      int continuous_objective_substeps = 0,
                                      real_t stability_step = -1.0)
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

   if (validate_cfl)
   {
      ValidateLumpedRK4TimeStep(
         oper, stability_step > 0.0 ? stability_step : h);
   }

   (void)control_fes;
   if (continuous_objective_substeps > 0)
   {
      objective.Reset();
      return ContinuousForwardSweepStreaming(
         oper, state_fes, objective, x0, nsteps, /*start_time=*/0.0, h,
         continuous_objective_substeps, progress_label);
   }
   return RolloutObjective(oper, state_fes, oper.GetBlockOffsets(), objective,
                           x0, nsteps, 0.0, h, nullptr, nullptr, progress_label);
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
   ObjectiveGradientAtStateAndTime(
      state_fes, oper.GetBlockOffsets(), objective,
      states[nsteps], times[nsteps], h, nsteps, total_steps, lambda);

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

      ObjectiveGradientAtStateAndTime(
         state_fes, oper.GetBlockOffsets(), objective,
         states[i], times[i], h, i, total_steps, q);
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

enum class RK4StageAdjointForm
{
   DISCRETE_REVERSE_AD,
   TRANSFORMED_PARTITIONED
};

// Full-storage reverse sweep for the common RK4-stage objective.  The two
// implementations are algebraically identical but deliberately remain
// separate kernels so their agreement is a measured verification result.
inline void RK4StageObjectiveAdjointDesignSweepFullStorage(
   ElastodynamicsOperator &oper,
   ParFiniteElementSpace &state_fes,
   ParFiniteElementSpace &filter_fes,
   ParGridFunction &rho_tilde,
   const MaterialParams &material,
   TimeIntegratedObjective &objective,
   const std::vector<Vector> &forward_states,
   int forward_steps,
   real_t start_time,
   real_t forward_step,
   RK4StageAdjointForm form,
   Vector &initial_adjoint,
   Vector &gradient_tilde)
{
   MFEM_VERIFY(
      forward_steps > 0 && std::isfinite(start_time) &&
      std::isfinite(forward_step) && forward_step > 0.0 &&
      forward_states.size() ==
         static_cast<std::size_t>(forward_steps + 1),
      "RK4 stage-objective adjoint sweep has invalid trajectory data.");

   gradient_tilde.SetSize(filter_fes.GetTrueVSize());
   gradient_tilde = 0.0;
   Vector lambda(oper.Height()), lambda_prev(oper.Height());
   lambda = 0.0; // No terminal functional in this experiment.

   for (int step = forward_steps - 1; step >= 0; step--)
   {
      const real_t time = start_time + step * forward_step;
      if (form == RK4StageAdjointForm::DISCRETE_REVERSE_AD)
      {
         RK4StageObjectiveDOAdjointOneStepWithDesign(
            oper, state_fes, filter_fes, rho_tilde, material, objective,
            forward_states[step], time, forward_step,
            lambda, lambda_prev, gradient_tilde);
      }
      else
      {
         RK4StageObjectiveTransformedAdjointOneStepWithDesign(
            oper, state_fes, filter_fes, rho_tilde, material, objective,
            forward_states[step], time, forward_step,
            lambda, lambda_prev, gradient_tilde);
      }
      lambda = lambda_prev;
   }
   initial_adjoint = lambda;
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
// Physics/FilterASolve). The default path delegates to the verified exact
// discrete primitives; the opt-in continuous path delegates to the shared
// full-storage/REVOLVE interval kernels tested below.
struct ContinuousStorageTelemetry
{
   real_t forward_seconds = 0.0;
   real_t adjoint_seconds = 0.0;
   real_t trajectory_memory_mb = 0.0;
   long long controller_replayed_blocks = 0;
   long long locally_replayed_blocks = 0;
   long long controller_replayed_intervals = 0;
   long long locally_replayed_intervals = 0;
};

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
   TransientAdjointMode adjoint_mode_;
   TrajectoryStorageMode trajectory_storage_mode_;
   int adjoint_refinement_;
   int adjoint_coarsening_;
   bool rk4_stage_objective_;
   NestedTimeGrid time_grid_{};
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

   // Trajectory checkpointing (replaces full storage of states_/times_)
   std::unique_ptr<TrajectoryCheckpointing<>> checkpoint_;
   int num_checkpoints_;
   Vector checkpoint_state_;  // Persistent state vector for checkpointing (survives forward->adjoint)
   std::vector<Vector> full_forward_states_;

   Vector dJ_drho_tilde_;
   // Retain the adjoint at the initial time after every production adjoint
   // sweep.  Storage diagnostics need the actual p(0), not only the design
   // gradient that was accumulated from it.
   Vector initial_adjoint_;
   ContinuousStorageTelemetry continuous_telemetry_;
   int outer_it_ = -1;
   bool banner_printed_ = false;   // operator banner prints once, not every iter

   int ScheduledTrajectoryBlocks() const
   {
      return time_grid_.relation == NestedTimeGridRelation::FORWARD_FINER ?
             time_grid_.adjoint_steps : time_grid_.forward_steps;
   }

   real_t ScheduledTrajectoryStep() const
   {
      return time_grid_.relation == NestedTimeGridRelation::FORWARD_FINER ?
             time_grid_.dt_adjoint : time_grid_.dt_forward;
   }

   int ObjectiveSubstepsPerForwardInterval() const
   {
      return time_grid_.relation == NestedTimeGridRelation::ADJOINT_FINER ?
             time_grid_.integer_ratio : 1;
   }

   real_t FullTrajectoryMemoryMB() const
   {
      long long vector_count =
         time_grid_.relation == NestedTimeGridRelation::FORWARD_FINER ?
         2LL * (time_grid_.forward_steps + 1LL) :
         time_grid_.forward_steps + 1LL;
      if (adjoint_mode_ == TransientAdjointMode::CONTINUOUS)
      {
         // One active Hermite interval retains two endpoint values and two
         // physical endpoint slopes during the reverse sweep.
         vector_count += 4LL;
      }
      const long double bytes =
         static_cast<long double>(vector_count) *
         static_cast<long double>(x0_.Size()) * sizeof(real_t);
      return static_cast<real_t>(bytes / (1024.0L * 1024.0L));
   }

   long long LocalTrajectoryScratchVectors() const
   {
      if (adjoint_mode_ != TransientAdjointMode::CONTINUOUS) { return 0LL; }
      // Forward-finer replay retains q+1 local endpoint states and q+1
      // physical slopes, as well as the coarse adjoint's two endpoints and
      // two slopes. Other relations retain one forward interval's two
      // endpoints and two slopes. RK stage/adjoint work vectors are solver
      // workspace, not trajectory reconstruction storage.
      return time_grid_.relation == NestedTimeGridRelation::FORWARD_FINER ?
             2LL * (time_grid_.integer_ratio + 1LL) + 4LL : 4LL;
   }

   real_t RolloutRK4StageObjectiveCheckpointed()
   {
      MFEM_VERIFY(
         rk4_stage_objective_ &&
         time_grid_.relation == NestedTimeGridRelation::SAME && checkpoint_,
         "Checkpointed RK4-stage objective requires a same-grid checkpoint path.");
      checkpoint_state_ = x0_;
      real_t checkpoint_time = 0.0;
      real_t objective_value = 0.0;
      Vector next_state(x0_.Size());
      const double phase_start = MPI_Wtime();
      const int report_every = std::max(1, nsteps_ / 10);

      const auto primal_step = [&](int step, Vector &state)
      {
         objective_value += RK4StageObjectiveForwardOneStep(
            *oper_, state_fes_, objective_, state, step * h_, h_, next_state);
         state = next_state;
      };
      for (int step = 0; step < nsteps_; step++)
      {
         checkpoint_->ForwardStep(
            step, checkpoint_state_, checkpoint_time, primal_step);
         checkpoint_time = (step + 1) * h_;
         if (Mpi::Root() &&
             ((step + 1) % report_every == 0 || step + 1 == nsteps_))
         {
            std::cout << "      forward " << std::setw(6) << step + 1
                      << '/' << nsteps_ << "  (" << std::setw(3)
                      << (100 * (step + 1) / nsteps_) << "%)   "
                      << std::fixed << std::setprecision(2)
                      << (MPI_Wtime() - phase_start) << " s\n";
         }
      }
      MFEM_VERIFY(std::isfinite(objective_value),
                  "Checkpointed RK4-stage objective is non-finite.");
      return objective_value;
   }

   real_t RolloutRK4StageObjectiveFullStorage()
   {
      MFEM_VERIFY(
         rk4_stage_objective_ &&
         time_grid_.relation == NestedTimeGridRelation::SAME,
         "Full-storage RK4-stage objective requires a same time grid.");
      return RK4StageObjectiveForwardSweepFullStorage(
         *oper_, state_fes_, objective_, x0_, nsteps_,
         /*start_time=*/0.0, h_, full_forward_states_);
   }

   // Checkpointed forward sweep (returns J, checkpoints trajectory)
   real_t RolloutObjectiveCheckpointed()
   {
      checkpoint_state_ = x0_;  // Use persistent member variable
      real_t t = 0.0;
      const int total_steps = nsteps_ + 1;

      objective_.Reset();

      RK4Solver solver;
      solver.Init(*oper_);

      // Primal step lambda for REVOLVE
      auto primal_step = [&](int i, Vector &state)
      {
         real_t dt = h_;
         real_t ti = i * h_;
         solver.Step(state, ti, dt);
      };

      // Add initial objective contribution
      AddObjectiveContribution(state_fes_, oper_->GetBlockOffsets(),
                               objective_, checkpoint_state_, h_, 0, total_steps);

      // Progress reporting (throttled to ~10 lines; max|u| flags instability).
      const double phase_t0 = MPI_Wtime();
      const int report_every = std::max(1, nsteps_ / 10);

      // Forward loop with checkpointing
      for (int i = 0; i < nsteps_; i++)
      {
         checkpoint_->ForwardStep(i, checkpoint_state_, t, primal_step);
         t = (i + 1) * h_;

         AddObjectiveContribution(state_fes_, oper_->GetBlockOffsets(),
                                  objective_, checkpoint_state_, h_, i + 1, total_steps);

         if ((i + 1) % report_every == 0 || i + 1 == nsteps_)
         {
            const int n_disp = checkpoint_state_.Size() / 2;
            real_t local_max_u = 0.0;
            int local_nonfinite = 0;
            for (int j = 0; j < checkpoint_state_.Size(); j++)
            {
               if (!std::isfinite(checkpoint_state_[j]))
               {
                  local_nonfinite = 1;
               }
               if (j < n_disp)
               {
                  local_max_u = std::max(local_max_u,
                                         std::abs(checkpoint_state_[j]));
               }
            }
            real_t global_max_u = 0.0;
            int global_nonfinite = 0;
            MPI_Allreduce(&local_max_u, &global_max_u, 1,
                          MPITypeMap<real_t>::mpi_type, MPI_MAX,
                          state_fes_.GetComm());
            MPI_Allreduce(&local_nonfinite, &global_nonfinite, 1, MPI_INT,
                          MPI_MAX, state_fes_.GetComm());
            MFEM_VERIFY(global_nonfinite == 0,
                        "Checkpointed forward solve produced a non-finite state.");
            if (Mpi::Root())
            {
               std::cout << "      forward " << std::setw(6) << (i + 1)
                         << '/' << nsteps_
                         << "  (" << std::setw(3) << (100 * (i + 1) / nsteps_)
                         << "%)   " << std::fixed << std::setprecision(2)
                         << (MPI_Wtime() - phase_t0) << " s"
                         << "   max|u| = " << std::scientific
                         << std::setprecision(3) << global_max_u << "\n";
            }
         }
      }

      return objective_.GetObjective();
   }

   // Checkpointed adjoint sweep (accumulates dJ/drho_tilde)
   void AdjointDesignSweepCheckpointed()
   {
      const int n = x0_.Size();
      const int total_steps = nsteps_ + 1;

      dJ_drho_tilde_.SetSize(filter_fes_.GetTrueVSize());
      dJ_drho_tilde_ = 0.0;

      Vector q(n), lambda(n), lambda_prev(n), u_work(n);

      // Terminal adjoint seed from the final forward state. checkpoint_state_
      // still holds x(t_final) from the preceding RolloutObjectiveCheckpointed,
      // so no forward re-run is needed here.
      MFEM_VERIFY(checkpoint_state_.Size() == n,
                  "AdjointDesignSweepCheckpointed: missing forward final state; "
                  "call PhysicsFSolve first.");
      u_work = checkpoint_state_;
      if (rk4_stage_objective_)
      {
         // The running functional is attached entirely to internal RK stages.
         // This experiment has no terminal functional.
         lambda = 0.0;
      }
      else
      {
         ObjectiveGradientAtState(
            state_fes_, oper_->GetBlockOffsets(), objective_, u_work,
            h_, nsteps_, total_steps, lambda);
      }

      // Primal step lambda for REVOLVE (re-evaluation during adjoint).  Keep
      // one initialized solver for the entire reverse sweep: constructing an
      // RK4Solver inside this callback would repeatedly allocate all of its
      // stage vectors for every recomputed step (tens of thousands of large
      // allocations in production spherical runs).
      RK4Solver reeval_solver;
      reeval_solver.Init(*oper_);
      auto primal_step = [&](int i, Vector &state)
      {
         real_t dt = h_;
         real_t ti = i * h_;
         reeval_solver.Step(state, ti, dt);
      };

      // Adjoint step lambda for REVOLVE
      auto adjoint_step = [&](int i, const Vector &state_i, Vector &lambda_current)
      {
         real_t ti = i * h_;
         if (rk4_stage_objective_)
         {
            RK4StageObjectiveDOAdjointOneStepWithDesign(
               *oper_, state_fes_, filter_fes_, rho_tilde_, mat_, objective_,
               state_i, ti, h_, lambda_current, lambda_prev,
               dJ_drho_tilde_);
            lambda_current = lambda_prev;
         }
         else
         {
            RK4AdjointOneStepWithDesign(
               *oper_, state_fes_, filter_fes_, rho_tilde_, mat_, state_i,
               ti, h_, lambda_current, lambda_prev, dJ_drho_tilde_);
            ObjectiveGradientAtState(
               state_fes_, oper_->GetBlockOffsets(), objective_, state_i,
               h_, i, total_steps, q);
            lambda_current = lambda_prev;
            lambda_current += q;
         }
      };

      // Backward loop with checkpointing (throttled progress, ~10 lines)
      const double adj_t0 = MPI_Wtime();
      const int adj_report_every = std::max(1, nsteps_ / 10);
      for (int i = nsteps_ - 1; i >= 0; i--)
      {
         checkpoint_->BackwardStep(i, lambda, u_work, primal_step, adjoint_step);

         const int done = nsteps_ - i;
         if (Mpi::Root() && (done % adj_report_every == 0 || done == nsteps_))
         {
            std::cout << "      adjoint " << std::setw(6) << done
                      << '/' << nsteps_
                      << "  (" << std::setw(3) << (100 * done / nsteps_)
                      << "%)   " << std::fixed << std::setprecision(2)
                      << (MPI_Wtime() - adj_t0) << " s\n";
         }
      }
      initial_adjoint_ = lambda;
   }

   // Fourth-order continuous objective on the same fine-forward intervals
   // used by the continuous design integral. For forward-finer grids REVOLVE
   // schedules N_a blocks, each of which advances q fine forward intervals.
   real_t RolloutContinuousObjectiveCheckpointed()
   {
      objective_.Reset();
      const double phase_t0 = MPI_Wtime();
      const int scheduled_blocks = ScheduledTrajectoryBlocks();
      const int report_every = std::max(1, scheduled_blocks / 10);
      const auto report_progress =
         [&](int completed, const Vector &state)
         {
            if (completed % report_every != 0 &&
                completed != scheduled_blocks)
            {
               return;
            }

            const int n_disp = state.Size() / 2;
            real_t local_max_u = 0.0;
            int local_nonfinite = 0;
            for (int j = 0; j < state.Size(); j++)
            {
               if (!std::isfinite(state[j])) { local_nonfinite = 1; }
               if (j < n_disp)
               {
                  local_max_u =
                     std::max(local_max_u, std::abs(state[j]));
               }
            }

            real_t global_max_u = 0.0;
            int global_nonfinite = 0;
            MPI_Allreduce(&local_max_u, &global_max_u, 1,
                          MPITypeMap<real_t>::mpi_type, MPI_MAX,
                          state_fes_.GetComm());
            MPI_Allreduce(&local_nonfinite, &global_nonfinite, 1, MPI_INT,
                          MPI_MAX, state_fes_.GetComm());
            MFEM_VERIFY(
               global_nonfinite == 0,
               "Continuous checkpointed forward solve produced a "
               "non-finite state.");

            if (Mpi::Root())
            {
               const int completed_forward_intervals =
                  time_grid_.relation == NestedTimeGridRelation::FORWARD_FINER ?
                  completed * time_grid_.integer_ratio : completed;
               std::cout << "      forward " << std::setw(6)
                         << completed_forward_intervals
                         << '/' << time_grid_.forward_steps
                         << "  (" << std::setw(3)
                         << (100 * completed / scheduled_blocks) << "%)   "
                         << std::fixed << std::setprecision(2)
                         << (MPI_Wtime() - phase_t0) << " s"
                         << "   max|u| = " << std::scientific
                         << std::setprecision(3) << global_max_u << "\n";
            }
         };

      if (time_grid_.relation == NestedTimeGridRelation::FORWARD_FINER)
      {
         return ContinuousForwardSweepBlockCheckpointed(
            *oper_, state_fes_, objective_, x0_, time_grid_,
            *checkpoint_, checkpoint_state_, report_progress);
      }
      return ContinuousForwardSweepCheckpointed(
         *oper_, state_fes_, objective_, x0_,
         time_grid_.forward_steps, /*start_time=*/0.0,
         time_grid_.dt_forward, ObjectiveSubstepsPerForwardInterval(),
         *checkpoint_, checkpoint_state_, report_progress);
   }

   real_t RolloutContinuousObjectiveFullStorage()
   {
      objective_.Reset();
      return ContinuousForwardSweepFullStorage(
         *oper_, state_fes_, objective_, x0_,
         time_grid_.forward_steps, /*start_time=*/0.0,
         time_grid_.dt_forward, ObjectiveSubstepsPerForwardInterval(),
         full_forward_states_);
   }

   void RunContinuousAdjointDesignCheckpointed()
   {
      Vector terminal_adjoint(x0_.Size());
      terminal_adjoint = 0.0;
      ContinuousReplayStatistics replay_statistics;

      const double adj_t0 = MPI_Wtime();
      const int scheduled_blocks = ScheduledTrajectoryBlocks();
      const int report_every = std::max(1, scheduled_blocks / 10);
      const auto report_progress =
         [&](int completed)
         {
            if (Mpi::Root() &&
                (completed % report_every == 0 ||
                 completed == scheduled_blocks))
            {
               const int completed_adjoint_steps =
                  time_grid_.relation == NestedTimeGridRelation::ADJOINT_FINER ?
                  completed * time_grid_.integer_ratio : completed;
               const int completed_forward_intervals =
                  time_grid_.relation == NestedTimeGridRelation::FORWARD_FINER ?
                  completed * time_grid_.integer_ratio : completed;
               std::cout << "      adjoint " << std::setw(6)
                         << completed_adjoint_steps
                         << '/' << time_grid_.adjoint_steps
                         << " steps (" << std::setw(3)
                         << (100 * completed / scheduled_blocks) << "%), "
                         << completed_forward_intervals
                         << '/' << time_grid_.forward_steps
                         << " forward intervals   " << std::fixed
                         << std::setprecision(2)
                         << (MPI_Wtime() - adj_t0) << " s\n";
            }
         };

      mfem::ContinuousAdjointDesignSweepCheckpointed(
         *oper_, state_fes_, filter_fes_, rho_tilde_, mat_, objective_,
         time_grid_, *checkpoint_, checkpoint_state_, terminal_adjoint,
         initial_adjoint_, dJ_drho_tilde_, &replay_statistics,
         report_progress);

      MFEM_VERIFY(
         replay_statistics.controller_replayed_blocks ==
         checkpoint_->EstimateRecomputations(),
         "Actual continuous REVOLVE block replay count differs from its estimate.");
      MFEM_VERIFY(
         replay_statistics.locally_replayed_blocks == scheduled_blocks &&
         replay_statistics.locally_replayed_intervals ==
         time_grid_.forward_steps,
         "Continuous REVOLVE must build exactly one private copy of every "
         "scheduled block and fine forward interval.");
      continuous_telemetry_.controller_replayed_blocks =
         replay_statistics.controller_replayed_blocks;
      continuous_telemetry_.locally_replayed_blocks =
         replay_statistics.locally_replayed_blocks;
      continuous_telemetry_.controller_replayed_intervals =
         replay_statistics.controller_replayed_intervals;
      continuous_telemetry_.locally_replayed_intervals =
         replay_statistics.locally_replayed_intervals;
      if (Mpi::Root())
      {
         std::cout << "      REVOLVE replay: "
                   << replay_statistics.controller_replayed_blocks
                   << " controller blocks / "
                   << replay_statistics.controller_replayed_intervals
                   << " forward intervals; "
                   << replay_statistics.locally_replayed_blocks
                   << " local blocks / "
                   << replay_statistics.locally_replayed_intervals
                   << " local forward intervals\n";
      }
   }

   void RunContinuousAdjointDesignFullStorage()
   {
      Vector terminal_adjoint(x0_.Size());
      terminal_adjoint = 0.0;
      ContinuousAdjointDesignSweepFullStorage(
         *oper_, state_fes_, filter_fes_, rho_tilde_, mat_, objective_,
         full_forward_states_, time_grid_, terminal_adjoint,
         initial_adjoint_, dJ_drho_tilde_);
   }

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
                         ParGridFunction &rho_tilde,
                         int num_checkpoints = -1,  // -1 = auto-size
                         TransientAdjointMode adjoint_mode =
                            TransientAdjointMode::DISCRETE,
                         int adjoint_refinement = 1,
                         TrajectoryStorageMode trajectory_storage_mode =
                            TrajectoryStorageMode::REVOLVE,
                         int adjoint_coarsening = 1,
                         bool rk4_stage_objective = false)
      : state_fes_(state_fes), filter_fes_(filter_fes), control_fes_(control_fes),
        filter_(filter), gamma_coef_(gamma_coef),
        exterior_bdr_attr_(exterior_bdr_attr), ess_bdr_attr_(ess_bdr_attr),
        objective_(objective), mat_(mat), load_spec_(load_spec),
        load_coef_(load_coef), impedance_(impedance),
        nsteps_(nsteps), h_(h), mass_type_(mass_type),
        adjoint_mode_(adjoint_mode),
        trajectory_storage_mode_(trajectory_storage_mode),
        adjoint_refinement_(adjoint_refinement),
        adjoint_coarsening_(adjoint_coarsening),
        rk4_stage_objective_(rk4_stage_objective),
        rho_(rho), rho_tilde_(rho_tilde),
        x0_(2 * state_fes.GetTrueVSize()),
        rho0_coef_(mat.rho0), lambda0_coef_(mat.lambda0), mu0_coef_(mat.mu0),
        simp_mass_(&rho_tilde_, mat.r_min, mat.r_max, mat.simp_p),
        simp_stiff_(&rho_tilde_, mat.r_min, mat.r_max, mat.simp_p),
        mass_coef_(simp_mass_, rho0_coef_),
        lambda_coef_(simp_stiff_, lambda0_coef_),
        mu_coef_(simp_stiff_, mu0_coef_),
        num_checkpoints_(num_checkpoints)
   {
      MFEM_VERIFY(nsteps_ > 0 && std::isfinite(h_) && h_ > 0.0,
                  "Transient design solver has an invalid forward time grid.");
      MFEM_VERIFY(adjoint_refinement_ > 0,
                  "Adjoint refinement must be a positive integer.");
      MFEM_VERIFY(adjoint_coarsening_ > 0,
                  "Adjoint coarsening must be a positive integer.");
      MFEM_VERIFY(adjoint_refinement_ == 1 || adjoint_coarsening_ == 1,
                  "Adjoint refinement and coarsening are mutually exclusive.");
      MFEM_VERIFY(!rk4_stage_objective_ ||
                  (adjoint_refinement_ == 1 && adjoint_coarsening_ == 1),
                  "The RK4-stage objective experiment requires one common "
                  "same-grid forward/adjoint timestep.");

      int adjoint_steps = nsteps_;
      if (adjoint_mode_ == TransientAdjointMode::DISCRETE)
      {
         MFEM_VERIFY(
            adjoint_refinement_ == 1 && adjoint_coarsening_ == 1,
            "Adjoint refinement/coarsening is only meaningful in continuous mode.");
         MFEM_VERIFY(
            trajectory_storage_mode_ == TrajectoryStorageMode::REVOLVE,
            "Discrete production adjoints currently require REVOLVE storage.");
      }
      else
      {
         if (adjoint_refinement_ > 1)
         {
            MFEM_VERIFY(
               nsteps_ <=
               std::numeric_limits<int>::max() / adjoint_refinement_,
               "The refined adjoint time grid exceeds the supported step count.");
            adjoint_steps = nsteps_ * adjoint_refinement_;
         }
         else if (adjoint_coarsening_ > 1)
         {
            MFEM_VERIFY(
               nsteps_ % adjoint_coarsening_ == 0,
               "The forward step count must be divisible by adjoint coarsening.");
            adjoint_steps = nsteps_ / adjoint_coarsening_;
         }
      }

      const real_t final_time = nsteps_ * h_;
      MFEM_VERIFY(std::isfinite(final_time) && final_time > 0.0,
                  "Transient design solver has a non-finite final time.");
      time_grid_ = NestedTimeGrid::Create(
         final_time, nsteps_, adjoint_steps);

      if (trajectory_storage_mode_ == TrajectoryStorageMode::REVOLVE)
      {
         if (num_checkpoints_ < 0)
         {
            num_checkpoints_ = AutoCheckpointCount(
               ScheduledTrajectoryBlocks(), 0.0, x0_.Size());
         }
         MFEM_VERIFY(num_checkpoints_ > 0,
                     "REVOLVE storage needs at least one checkpoint.");
      }
      else
      {
         num_checkpoints_ = 0;
      }
      x0_ = 0.0;
   }

   int NumSteps() const { return nsteps_; }
   real_t TimeStep() const { return h_; }
   int AdjointRefinement() const { return adjoint_refinement_; }
   int AdjointCoarsening() const { return adjoint_coarsening_; }
   TransientAdjointMode AdjointMode() const { return adjoint_mode_; }
   TrajectoryStorageMode TrajectoryStorage() const
   {
      return trajectory_storage_mode_;
   }
   int NumTrajectoryCheckpoints() const { return num_checkpoints_; }
   real_t EstimatedTrajectoryMemoryMB() const
   {
      if (trajectory_storage_mode_ == TrajectoryStorageMode::FULL)
      {
         return FullTrajectoryMemoryMB();
      }
      const long double bytes =
         static_cast<long double>(num_checkpoints_) *
         static_cast<long double>(RK4Snapshot::ByteSize(x0_.Size())) +
         static_cast<long double>(LocalTrajectoryScratchVectors()) *
         static_cast<long double>(x0_.Size()) * sizeof(real_t);
      return static_cast<real_t>(bytes / (1024.0L * 1024.0L));
   }
   const NestedTimeGrid &TimeGrid() const { return time_grid_; }
   const ContinuousStorageTelemetry &StorageTelemetry() const
   {
      return continuous_telemetry_;
   }
   const Vector &FilteredDesignGradient() const
   {
      return dJ_drho_tilde_;
   }
   const Vector &InitialAdjoint() const
   {
      return initial_adjoint_;
   }

   // Same-grid paper experiment: one RK4-stage objective is evaluated once,
   // then differentiated by exact reverse AD (DO), an independently coded
   // transformed partitioned adjoint (OD_modified), and the deliberately
   // naive continuous-adjoint RK4 march over the accepted/Hermite trajectory.
   // The latter is intentionally named with its reconstruction because, for
   // classical RK4, feeding Y4,Y3,Y2,Y1 directly to a reverse RK4 step would
   // algebraically collapse onto the transformed/DO recurrence.
   void AnalyzeRK4AdjointComparison(
      const Vector &rho_tv,
      const Vector &control_volume_weights,
      const std::string &output_parent,
      const Array<int> *active_tdof_list = nullptr,
      int taylor_levels = 0,
      real_t initial_taylor_epsilon = 1e-2,
      bool taylor_linf_normalized = false)
   {
      MFEM_VERIFY(
         rho_tv.Size() == control_fes_.GetTrueVSize() &&
         control_volume_weights.Size() == rho_tv.Size(),
         "RK4 adjoint comparison received an invalid raw design or mass vector.");
      MFEM_VERIFY(
         taylor_levels >= 0 && std::isfinite(initial_taylor_epsilon) &&
         initial_taylor_epsilon > 0.0,
         "RK4 adjoint comparison has invalid Taylor controls.");

      const MPI_Comm comm = state_fes_.GetComm();
      FilterFSolve(rho_tv);
      ElastodynamicsOperator oper(
         state_fes_, mass_coef_, lambda_coef_, mu_coef_,
         load_spec_.amplitude, load_spec_.duration, load_spec_.time_profile,
         load_spec_.phase, load_spec_.frequency, load_spec_.bdr_attributes,
         load_coef_, load_spec_.domain_load, &gamma_coef_, impedance_,
         exterior_bdr_attr_, ess_bdr_attr_, mass_type_,
         /*print_banner=*/true);
      ValidateLumpedRK4TimeStep(oper, h_, /*print_report=*/true);

      const auto global_max = [&](real_t local_value)
      {
         real_t global_value = 0.0;
         MPI_Allreduce(&local_value, &global_value, 1,
                       MPITypeMap<real_t>::mpi_type, MPI_MAX, comm);
         return global_value;
      };
      const auto global_sum = [&](real_t local_value)
      {
         real_t global_value = 0.0;
         MPI_Allreduce(&local_value, &global_value, 1,
                       MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
         return global_value;
      };
      // Cache the active marker once; repeated linear scans would dominate
      // metric and Taylor-direction setup for large discontinuous control
      // spaces. The physical L2 metric is diagonal only for Q0; higher-order
      // L2 controls use the assembled control mass matrix below.
      Array<char> active_marker(rho_tv.Size());
      active_marker = active_tdof_list ? 0 : 1;
      if (active_tdof_list)
      {
         for (int i = 0; i < active_tdof_list->Size(); i++)
         {
            const int tdof = (*active_tdof_list)[i];
            MFEM_VERIFY(tdof >= 0 && tdof < rho_tv.Size(),
                        "Active raw-design true DOF is out of range.");
            active_marker[tdof] = 1;
         }
      }
      const bool diagonal_control_mass =
         control_fes_.GetFE(0)->GetOrder() == 0;
      std::unique_ptr<ParBilinearForm> control_mass_form;
      std::unique_ptr<HypreParMatrix> control_mass;
      std::unique_ptr<CGSolver> control_mass_solver;
      if (!diagonal_control_mass)
      {
         control_mass_form = std::make_unique<ParBilinearForm>(&control_fes_);
         control_mass_form->AddDomainIntegrator(new MassIntegrator());
         control_mass_form->Assemble();
         control_mass_form->Finalize();
         control_mass.reset(control_mass_form->ParallelAssemble());

         // The discontinuous control mass is element-block diagonal. CG is
         // therefore both inexpensive and an exact implementation of the
         // active L2 Riesz map to the requested tolerance.
         control_mass_solver = std::make_unique<CGSolver>(comm);
         control_mass_solver->SetOperator(*control_mass);
         control_mass_solver->SetRelTol(1.0e-13);
         control_mass_solver->SetAbsTol(0.0);
         control_mass_solver->SetMaxIter(32);
         control_mass_solver->SetPrintLevel(0);
      }

      const auto restrict_to_active = [&](const Vector &source, Vector &target)
      {
         MFEM_VERIFY(source.Size() == rho_tv.Size(),
                     "Active-control restriction received an invalid vector.");
         target = source;
         for (int i = 0; i < target.Size(); i++)
         {
            if (!active_marker[i]) { target[i] = 0.0; }
         }
      };
      const auto riesz_solve = [&](const Vector &covector, Vector &function)
      {
         Vector active_covector(covector.Size());
         restrict_to_active(covector, active_covector);
         function.SetSize(covector.Size());
         function = 0.0;
         if (diagonal_control_mass)
         {
            for (int i = 0; i < function.Size(); i++)
            {
               if (!active_marker[i]) { continue; }
               const real_t weight = control_volume_weights[i];
               MFEM_VERIFY(std::isfinite(weight) && weight > 0.0,
                           "Active Q0 design mass must be positive.");
               function[i] = active_covector[i] / weight;
            }
            return;
         }

         control_mass_solver->Mult(active_covector, function);
         MFEM_VERIFY(control_mass_solver->GetConverged(),
                     "Control-mass Riesz solve did not converge.");
         for (int i = 0; i < function.Size(); i++)
         {
            if (!active_marker[i]) { function[i] = 0.0; }
         }

         Vector residual(function.Size());
         control_mass->Mult(function, residual);
         residual -= active_covector;
         const real_t rhs_norm = GlobalVectorNorm(comm, active_covector);
         const real_t residual_norm = GlobalVectorNorm(comm, residual);
         MFEM_VERIFY(residual_norm <= 1.0e-11 * std::max(rhs_norm, real_t(1.0)),
                     "Control-mass Riesz solve residual is too large.");
      };
      const auto primal_inner_product =
         [&](const Vector &left, const Vector &right)
         {
            MFEM_VERIFY(left.Size() == rho_tv.Size() &&
                        right.Size() == rho_tv.Size(),
                        "Raw-design L2 metric received an invalid vector.");
            if (diagonal_control_mass)
            {
               real_t local_value = 0.0;
               for (int i = 0; i < left.Size(); i++)
               {
                  if (active_marker[i])
                  {
                     local_value += control_volume_weights[i] * left[i] * right[i];
                  }
               }
               return global_sum(local_value);
            }

            Vector active_right(right.Size()), mass_times_right(right.Size());
            restrict_to_active(right, active_right);
            control_mass->Mult(active_right, mass_times_right);
            real_t local_value = 0.0;
            for (int i = 0; i < left.Size(); i++)
            {
               if (active_marker[i])
               {
                  local_value += left[i] * mass_times_right[i];
               }
            }
            return global_sum(local_value);
         };
      const auto dual_inner_product =
         [&](const Vector &left, const Vector &right)
         {
            MFEM_VERIFY(left.Size() == rho_tv.Size() &&
                        right.Size() == rho_tv.Size(),
                        "Raw-gradient dual metric received an invalid vector.");
            Vector riesz_right(right.Size());
            riesz_solve(right, riesz_right);
            real_t local_value = 0.0;
            for (int i = 0; i < left.Size(); i++)
            {
               if (active_marker[i]) { local_value += left[i] * riesz_right[i]; }
            }
            return global_sum(local_value);
         };
      const auto dual_norm = [&](const Vector &value)
      {
         return std::sqrt(std::max(real_t(0.0),
                                   dual_inner_product(value, value)));
      };

      const long double trajectory_bytes =
         static_cast<long double>(nsteps_ + 1LL) *
         static_cast<long double>(x0_.Size()) * sizeof(real_t);
      const real_t trajectory_megabytes = global_max(static_cast<real_t>(
         trajectory_bytes / (1024.0L * 1024.0L)));
      if (Mpi::Root())
      {
         std::cout << "RK4 comparison uses full endpoint storage: estimated "
                   << std::fixed << std::setprecision(2)
                   << trajectory_megabytes << " MB per rank\n";
      }

      std::vector<Vector> forward_states;
      const double forward_start = MPI_Wtime();
      const real_t common_objective =
         RK4StageObjectiveForwardSweepFullStorage(
            oper, state_fes_, objective_, x0_, nsteps_,
            /*start_time=*/0.0, h_, forward_states);
      const real_t forward_seconds = global_max(
         static_cast<real_t>(MPI_Wtime() - forward_start));

      struct ComparisonResult
      {
         const char *name = "";
         Vector initial_adjoint;
         Vector filtered_gradient;
         Vector raw_gradient;
         real_t seconds = 0.0;
         real_t initial_adjoint_norm = 0.0;
         real_t initial_adjoint_relative_error_to_do = 0.0;
         real_t initial_adjoint_cosine_to_do = 1.0;
         real_t raw_dual_norm = 0.0;
         real_t relative_error_to_do = 0.0;
         real_t cosine_to_do = 1.0;
         real_t projected_gradient = 0.0;
      };

      const auto run_consistent =
         [&](const char *name, RK4StageAdjointForm form)
         {
            ComparisonResult result;
            result.name = name;
            const double start = MPI_Wtime();
            RK4StageObjectiveAdjointDesignSweepFullStorage(
               oper, state_fes_, filter_fes_, rho_tilde_, mat_, objective_,
               forward_states, nsteps_, /*start_time=*/0.0, h_, form,
               result.initial_adjoint, result.filtered_gradient);
            filter_.MultTranspose(result.filtered_gradient,
                                  result.raw_gradient);
            result.seconds = global_max(
               static_cast<real_t>(MPI_Wtime() - start));
            return result;
         };

      ComparisonResult do_result = run_consistent(
         "DO", RK4StageAdjointForm::DISCRETE_REVERSE_AD);
      ComparisonResult modified_result = run_consistent(
         "OD_modified", RK4StageAdjointForm::TRANSFORMED_PARTITIONED);

      ComparisonResult naive_result;
      naive_result.name = "OD_naive_Hermite";
      const double naive_start = MPI_Wtime();
      const NestedTimeGrid same_grid = NestedTimeGrid::Create(
         nsteps_ * h_, nsteps_, nsteps_);
      Vector terminal_adjoint(x0_.Size());
      terminal_adjoint = 0.0;
      ContinuousAdjointDesignSweepFullStorage(
         oper, state_fes_, filter_fes_, rho_tilde_, mat_, objective_,
         forward_states, same_grid, terminal_adjoint,
         naive_result.initial_adjoint, naive_result.filtered_gradient);
      filter_.MultTranspose(naive_result.filtered_gradient,
                            naive_result.raw_gradient);
      naive_result.seconds = global_max(
         static_cast<real_t>(MPI_Wtime() - naive_start));
      // All three reverse sweeps are complete. Release the potentially large
      // endpoint history before CSV processing and optional Taylor reruns.
      std::vector<Vector>().swap(forward_states);

      ComparisonResult *results[] =
      {
         &do_result, &modified_result, &naive_result
      };
      const real_t do_initial_adjoint_norm =
         GlobalVectorNorm(comm, do_result.initial_adjoint);
      const real_t do_raw_norm = dual_norm(do_result.raw_gradient);
      MFEM_VERIFY(do_initial_adjoint_norm > 0.0 &&
                  std::isfinite(do_initial_adjoint_norm),
                  "DO initial adjoint has zero or non-finite norm.");
      MFEM_VERIFY(do_raw_norm > 0.0 && std::isfinite(do_raw_norm),
                  "DO raw-design gradient has zero or non-finite dual norm.");
      for (ComparisonResult *result : results)
      {
         result->initial_adjoint_norm =
            GlobalVectorNorm(comm, result->initial_adjoint);
         Vector initial_adjoint_difference(result->initial_adjoint);
         initial_adjoint_difference -= do_result.initial_adjoint;
         result->initial_adjoint_relative_error_to_do =
            GlobalVectorNorm(comm, initial_adjoint_difference) /
            do_initial_adjoint_norm;
         if (result->initial_adjoint_norm > 0.0)
         {
            result->initial_adjoint_cosine_to_do = std::max(
               real_t(-1.0), std::min(real_t(1.0), GlobalVectorDot(
                  comm, result->initial_adjoint, do_result.initial_adjoint) /
                  (result->initial_adjoint_norm * do_initial_adjoint_norm)));
         }
         result->raw_dual_norm = dual_norm(result->raw_gradient);
         Vector difference(result->raw_gradient);
         difference -= do_result.raw_gradient;
         result->relative_error_to_do = dual_norm(difference) / do_raw_norm;
         if (result->raw_dual_norm > 0.0)
         {
            result->cosine_to_do = std::max(
               real_t(-1.0), std::min(real_t(1.0), dual_inner_product(
                  result->raw_gradient, do_result.raw_gradient) /
                  (result->raw_dual_norm * do_raw_norm)));
         }
      }

      // Construct the L2-Riesz DO direction, project it into the tangent of
      // the fixed-volume constraint, and normalize it in the active-design
      // L2 metric. For Q0 the Riesz solve is diagonal; for higher-order L2
      // controls it is the assembled control-mass solve above.
      Vector direction(rho_tv.Size());
      riesz_solve(do_result.raw_gradient, direction);
      real_t local_volume = 0.0;
      real_t local_weighted_direction_sum = 0.0;
      for (int i = 0; i < direction.Size(); i++)
      {
         if (!active_marker[i]) { continue; }
         const real_t weight = control_volume_weights[i];
         local_volume += weight;
         local_weighted_direction_sum += weight * direction[i];
      }
      const real_t active_volume = global_sum(local_volume);
      MFEM_VERIFY(active_volume > 0.0 && std::isfinite(active_volume),
                  "RK4 comparison has no finite positive active volume.");
      const real_t direction_mean =
         global_sum(local_weighted_direction_sum) / active_volume;
      for (int i = 0; i < direction.Size(); i++)
      {
         if (!active_marker[i]) { continue; }
         direction[i] -= direction_mean;
      }
      const real_t direction_norm = std::sqrt(std::max(
         real_t(0.0), primal_inner_product(direction, direction)));
      MFEM_VERIFY(direction_norm > 0.0 && std::isfinite(direction_norm),
                  "Volume-neutral DO Riesz direction is degenerate.");
      direction /= direction_norm;
      if (taylor_linf_normalized)
      {
         real_t local_max_abs_direction = 0.0;
         for (int i = 0; i < direction.Size(); i++)
         {
            if (active_marker[i])
            {
               local_max_abs_direction = std::max(
                  local_max_abs_direction, std::abs(direction[i]));
            }
         }
         const real_t max_abs_direction = global_max(local_max_abs_direction);
         MFEM_VERIFY(max_abs_direction > 0.0 &&
                     std::isfinite(max_abs_direction),
                     "Volume-neutral DO Riesz direction has zero L-infinity norm.");
         direction /= max_abs_direction;
      }
      for (ComparisonResult *result : results)
      {
         result->projected_gradient =
            GlobalVectorDot(comm, result->raw_gradient, direction);
      }

      if (Mpi::Root())
      {
         std::ofstream csv(output_parent + "/rk4_adjoint_comparison.csv");
         MFEM_VERIFY(csv.is_open(),
                     "Could not open the RK4 adjoint-comparison CSV.");
         csv << "method,objective,N,dt,state_order,filter_order,control_order,"
                "initial_adjoint_norm,initial_adjoint_relative_error_to_do,"
                "initial_adjoint_cosine_to_do,"
                "active_raw_gradient_dual_norm,"
                "active_raw_gradient_relative_error_to_do,"
                "active_raw_gradient_cosine_to_do,projected_gradient,"
                "forward_seconds,adjoint_seconds\n";
         const int state_order = state_fes_.GetFE(0)->GetOrder();
         const int filter_order = filter_fes_.GetFE(0)->GetOrder();
         const int control_order = control_fes_.GetFE(0)->GetOrder();
         for (const ComparisonResult *result : results)
         {
            csv << result->name << ',' << std::setprecision(16)
                << common_objective << ',' << nsteps_ << ',' << h_ << ','
                << state_order << ',' << filter_order << ',' << control_order
                << ',' << result->initial_adjoint_norm << ','
                << result->initial_adjoint_relative_error_to_do << ','
                << result->initial_adjoint_cosine_to_do << ','
                << result->raw_dual_norm << ','
                << result->relative_error_to_do << ','
                << result->cosine_to_do << ','
                << result->projected_gradient << ','
                << forward_seconds << ',' << result->seconds << '\n';
         }

         std::cout << "\n=== Same-grid RK4 adjoint comparison ===\n"
                   << "Common four-stage objective J_h = "
                   << std::scientific << std::setprecision(8)
                   << common_objective << "\n"
                   << "method                 ||g||_Q'       rel.to.DO"
                      "        cos.to.DO        adjoint_s\n";
         for (const ComparisonResult *result : results)
         {
            std::cout << std::left << std::setw(22) << result->name
                      << std::right << std::scientific
                      << std::setprecision(8) << std::setw(15)
                      << result->raw_dual_norm << std::setw(16)
                      << result->relative_error_to_do << std::setw(17)
                      << result->cosine_to_do << std::fixed
                      << std::setprecision(2) << std::setw(13)
                      << result->seconds << "\n";
         }
         std::cout << "Results: " << output_parent
                   << "/rk4_adjoint_comparison.csv\n";
      }

      if (taylor_levels > 0)
      {
         // Keep symmetric perturbations strictly inside [0,1]. Passive entries
         // have zero direction and therefore remain pinned at rho=1.
         real_t local_bound = std::numeric_limits<real_t>::infinity();
         for (int i = 0; i < rho_tv.Size(); i++)
         {
            if (!active_marker[i] || direction[i] == 0.0) { continue; }
            const real_t bound = std::min(
               rho_tv[i] / std::abs(direction[i]),
               (1.0 - rho_tv[i]) / std::abs(direction[i]));
            local_bound = std::min(local_bound, bound);
         }
         real_t global_bound = 0.0;
         MPI_Allreduce(&local_bound, &global_bound, 1,
                       MPITypeMap<real_t>::mpi_type, MPI_MIN, comm);
         MFEM_VERIFY(std::isfinite(global_bound) && global_bound > 0.0,
                     "Taylor direction has no admissible symmetric step; "
                     "use an interior initial design such as -init uniform.");
         // Use the requested magnitude whenever it is strictly feasible.  The
         // small 5% margin is only a guard against roundoff at an active bound;
         // the former factor 1/2 unnecessarily changed a requested 0.1
         // density perturbation into 0.075 for a bounded direction.
         real_t epsilon = std::min(initial_taylor_epsilon,
                                   0.95 * global_bound);

         std::ofstream taylor_csv;
         if (Mpi::Root())
         {
            taylor_csv.open(output_parent + "/rk4_adjoint_taylor.csv");
            MFEM_VERIFY(taylor_csv.is_open(),
                        "Could not open the RK4 Taylor-test CSV.");
            taylor_csv
               << "method,level,epsilon,J_plus,J_minus,centered_fd,"
                  "projected_gradient,fd_relative_error,"
                  "remainder_plus,remainder_minus\n";
         }

         const auto evaluate_perturbed_objective = [&](const Vector &design)
         {
            rho_.SetFromTrueDofs(design);
            filter_.Mult(rho_, rho_tilde_);
            ElastodynamicsOperator perturbed_oper(
               state_fes_, mass_coef_, lambda_coef_, mu_coef_,
               load_spec_.amplitude, load_spec_.duration,
               load_spec_.time_profile, load_spec_.phase,
               load_spec_.frequency, load_spec_.bdr_attributes,
               load_coef_, load_spec_.domain_load, &gamma_coef_, impedance_,
               exterior_bdr_attr_, ess_bdr_attr_, mass_type_,
               /*print_banner=*/false);
            return RK4StageObjectiveForwardSweepStreaming(
               perturbed_oper, state_fes_, objective_, x0_, nsteps_,
               /*start_time=*/0.0, h_);
         };

         Vector plus(rho_tv.Size()), minus(rho_tv.Size());
         for (int level = 0; level < taylor_levels; level++)
         {
            plus = rho_tv;
            minus = rho_tv;
            plus.Add(epsilon, direction);
            minus.Add(-epsilon, direction);
            const real_t j_plus = evaluate_perturbed_objective(plus);
            const real_t j_minus = evaluate_perturbed_objective(minus);
            const real_t centered_fd =
               (j_plus - j_minus) / (2.0 * epsilon);

            if (Mpi::Root())
            {
               for (const ComparisonResult *result : results)
               {
                  const real_t derivative_scale = std::max(
                     {std::abs(centered_fd),
                      std::abs(result->projected_gradient), real_t(1e-30)});
                  const real_t relative_error =
                     std::abs(centered_fd - result->projected_gradient) /
                     derivative_scale;
                  const real_t remainder_plus = std::abs(
                     j_plus - common_objective -
                     epsilon * result->projected_gradient);
                  const real_t remainder_minus = std::abs(
                     j_minus - common_objective +
                     epsilon * result->projected_gradient);
                  taylor_csv << result->name << ',' << level << ','
                             << std::setprecision(16) << epsilon << ','
                             << j_plus << ',' << j_minus << ','
                             << centered_fd << ','
                             << result->projected_gradient << ','
                             << relative_error << ',' << remainder_plus << ','
                             << remainder_minus << '\n';
               }
            }
            epsilon *= 0.5;
         }
         if (Mpi::Root())
         {
            std::cout << "Taylor results: " << output_parent
                      << "/rk4_adjoint_taylor.csv\n";
         }
      }

      // Taylor perturbations update the live coefficient field. Restore the
      // exact design supplied by the driver before returning.
      rho_.SetFromTrueDofs(rho_tv);
      filter_.Mult(rho_, rho_tilde_);
   }

   void AnalyzeRightHandSideSpectra(const Vector &rho_tv,
                                    const std::string &output_parent,
                                    bool save_paraview,
                                    int lanczos_steps = 30)
   {
      FilterFSolve(rho_tv);

      // Use the mass discretization selected for the production solve.  Then
      // A=M^{-1}K is self-adjoint in that same M inner product (up to the tight
      // consistent-mass CG tolerance), so the diagnostic does not silently
      // analyze a different marching operator.
      ElastodynamicsOperator spectral_oper(
         state_fes_, mass_coef_, lambda_coef_, mu_coef_,
         load_spec_.amplitude, load_spec_.duration, load_spec_.time_profile,
         load_spec_.phase, load_spec_.frequency, load_spec_.bdr_attributes,
         load_coef_, load_spec_.domain_load, &gamma_coef_, impedance_,
         exterior_bdr_attr_, ess_bdr_attr_, mass_type_,
         /*print_banner=*/true);

      const Vector forward_rhs(spectral_oper.GetLoadBaseVector());

      // Evaluate dJ/du at rest and t=0.  The tracking objective has a nonzero
      // prescribed target there, so this isolates its spatial adjoint source.
      Vector q_state;
      ObjectiveGradientAtState(state_fes_, spectral_oper.GetBlockOffsets(),
                               objective_, x0_, h_, 0, nsteps_ + 1, q_state);
      BlockVector q_blocks(q_state, spectral_oper.GetBlockOffsets());
      Vector adjoint_rhs(q_blocks.GetBlock(0));

      Vector forward_shape, adjoint_shape;
      RHSSpectralSummary forward = AnalyzeRHSSpectrum(
         spectral_oper, state_fes_.GetComm(), forward_rhs,
         lanczos_steps, &forward_shape);
      RHSSpectralSummary adjoint = AnalyzeRHSSpectrum(
         spectral_oper, state_fes_.GetComm(), adjoint_rhs,
         lanczos_steps, &adjoint_shape);

      constexpr real_t two_pi =
         2.0 * 3.1415926535897932384626433832795;
      const auto print_summary =
         [&](const char *name, const RHSSpectralSummary &s)
         {
            std::cout << "  " << name << ":\n"
                      << "    ||M^{-1}b||_M = " << std::scientific
                      << std::setprecision(6) << s.rhs_mass_inverse_norm << "\n"
                      << "    lambda mean   = " << s.mean_lambda << "\n"
                      << "    lambda 5/50/95% = " << s.lambda_05 << " / "
                      << s.lambda_50 << " / " << s.lambda_95 << "\n"
                      << "    frequency 5/50/95% = "
                      << std::sqrt(std::max(real_t(0.0), s.lambda_05)) / two_pi
                      << " / "
                      << std::sqrt(std::max(real_t(0.0), s.lambda_50)) / two_pi
                      << " / "
                      << std::sqrt(std::max(real_t(0.0), s.lambda_95)) / two_pi
                      << "\n";
         };

      if (Mpi::Root())
      {
         std::cout << "\n=== Forward/Adjoint RHS Spectral Measure ===\n"
                   << "Generalized spectrum: K phi = lambda "
                   << (mass_type_ == MassSolverType::LUMPED ? "M_L" : "M")
                   << " phi\n"
                   << "Lanczos steps requested: " << lanczos_steps << "\n";
         print_summary("forward F", forward);
         print_summary("adjoint dJ/du", adjoint);

         if (forward.lambda_95 > 0.0 && adjoint.lambda_95 > 0.0)
         {
            const real_t frequency_ratio =
               std::sqrt(adjoint.lambda_95 / forward.lambda_95);
            std::cout << "  95%-energy frequency ratio (adjoint/forward) = "
                      << frequency_ratio << "\n"
                      << "  Accuracy-step proxy dt_adjoint/dt_forward = "
                      << 1.0 / frequency_ratio << "\n";
         }
         std::cout << "NOTE: this ratio measures source-resolved accuracy; the "
                      "strict full-space explicit CFL limit is unchanged.\n"
                   << "==============================================\n";

         std::ofstream csv(output_parent + "/rhs_spectrum.csv");
         csv << "rhs,ritz_index,lambda,frequency,weight,cumulative_weight\n";
         const auto write_spectrum =
            [&](const char *name, const RHSSpectralSummary &s)
            {
               real_t cumulative = 0.0;
               for (int i = 0; i < s.ritz_values.Size(); i++)
               {
                  cumulative += s.spectral_weights[i];
                  const real_t lambda =
                     std::max(real_t(0.0), s.ritz_values[i]);
                  csv << name << ',' << i << ',' << std::setprecision(16)
                      << lambda << ',' << std::sqrt(lambda) / two_pi << ','
                      << s.spectral_weights[i] << ',' << cumulative << '\n';
               }
            };
         write_spectrum("forward", forward);
         write_spectrum("adjoint", adjoint);
      }

      if (save_paraview)
      {
         ParGridFunction forward_gf(&state_fes_);
         ParGridFunction adjoint_gf(&state_fes_);
         forward_gf.SetFromTrueDofs(forward_shape);
         adjoint_gf.SetFromTrueDofs(adjoint_shape);

         ParaViewDataCollection rhs_dc("rhs_spectrum",
                                       state_fes_.GetParMesh());
         rhs_dc.SetPrefixPath((output_parent + "/ParaView").c_str());
         rhs_dc.SetLevelsOfDetail(state_fes_.GetMaxElementOrder());
         rhs_dc.SetDataFormat(VTKFormat::BINARY);
         rhs_dc.SetHighOrderOutput(true);
         rhs_dc.RegisterField("M_inv_F", &forward_gf);
         rhs_dc.RegisterField("M_inv_dJdu", &adjoint_gf);
         rhs_dc.SetCycle(0);
         rhs_dc.SetTime(0.0);
         rhs_dc.Save();
      }
   }

   void AnalyzeContinuousAdjointCoarsening(
      const Vector &rho_tv,
      const std::string &output_parent,
      int maximum_coarsening = 16)
   {
      MFEM_VERIFY(maximum_coarsening >= 2,
                  "The interpolation-free RK4 coarsening diagnostic requires "
                  "a maximum adjoint coarsening of at least 2.");
      FilterFSolve(rho_tv);

      ElastodynamicsOperator oper(
         state_fes_, mass_coef_, lambda_coef_, mu_coef_,
         load_spec_.amplitude, load_spec_.duration, load_spec_.time_profile,
         load_spec_.phase, load_spec_.frequency, load_spec_.bdr_attributes,
         load_coef_, load_spec_.domain_load, &gamma_coef_, impedance_,
         exterior_bdr_attr_, ess_bdr_attr_, mass_type_,
         /*print_banner=*/true);

      ValidateLumpedRK4TimeStep(oper, h_, /*print_report=*/true);

      std::vector<Vector> forward_states;
      std::vector<real_t> forward_times;
      const real_t objective_value = RolloutObjective(
         oper, state_fes_, oper.GetBlockOffsets(), objective_,
         x0_, nsteps_, 0.0, h_, &forward_states, &forward_times,
         "fine forward");
      ExactStoredForwardStateProvider state_provider(
         forward_states, 0.0, h_);

      std::vector<int> ratios;
      for (int ratio = 2;
           ratio <= maximum_coarsening && ratio <= nsteps_;
           ratio *= 2)
      {
         if (nsteps_ % ratio == 0) { ratios.push_back(ratio); }
         if (ratio > std::numeric_limits<int>::max() / 2) { break; }
      }
      MFEM_VERIFY(!ratios.empty(),
                  "No usable even coarsening ratio divides the number of "
                  "forward steps. Use an even N_f and a maximum ratio >= 2.");

      const MPI_Comm comm = state_fes_.GetComm();
      Vector terminal_adjoint(x0_.Size());
      terminal_adjoint = 0.0;
      Vector reference_initial, initial_adjoint, difference;
      real_t reference_norm = 0.0;

      std::ofstream csv;
      if (Mpi::Root())
      {
         csv.open(output_parent + "/continuous_adjoint_coarsening.csv");
         MFEM_VERIFY(csv.is_open(),
                     "Could not open the continuous-adjoint CSV output.");
         csv << "coarsening,N_forward,N_adjoint,dt_forward,dt_adjoint,"
                "initial_adjoint_norm,relative_vector_error,reference_cosine\n";

         std::cout << "\n=== Full-Storage Continuous-Adjoint Coarsening ===\n"
                   << "Forward objective (discrete trapezoid report only): "
                   << std::scientific << std::setprecision(10)
                   << objective_value << "\n"
                   << "Forward steps: " << nsteps_
                   << ", dt_f: " << h_ << "\n"
                   << "Forward trajectory storage: " << forward_states.size()
                   << " full states\n"
                   << "The continuous adjoint uses instantaneous objective "
                      "forcing and p(T)=0.\n"
                   << "q=2 is the finest interpolation-free RK4 reference: "
                      "its endpoint and midpoint stages are forward nodes.\n"
                   << "  q       N_a          dt_a          ||p(0)||"
                      "       rel.error          cosine\n";
         if (ratios.size() < 3)
         {
            std::cout << "NOTE: fewer than q={2,4,8} are available; this is "
                         "a wiring smoke test, not a convergence proof.\n";
         }
      }

      for (int ratio : ratios)
      {
         const int adjoint_steps = nsteps_ / ratio;
         const NestedTimeGrid grid =
            NestedTimeGrid::Create(nsteps_ * h_, nsteps_, adjoint_steps);
         MFEM_VERIFY(grid.relation == NestedTimeGridRelation::FORWARD_FINER,
                     "Coarsening diagnostic requires a finer forward grid.");
         MFEM_VERIFY(ratio % 2 == 0,
                     "Classical RK4 coarse-adjoint stages require an even "
                     "forward-step ratio for exact stored-node lookup.");

         ValidateLumpedRK4TimeStep(
            oper, grid.dt_adjoint, /*print_report=*/false);

         ContinuousAdjointSweepFullStorage(
            oper, state_fes_, objective_, state_provider, grid,
            terminal_adjoint, initial_adjoint);
         const real_t initial_norm = GlobalVectorNorm(comm, initial_adjoint);
         MFEM_VERIFY(initial_norm > 0.0,
                     "Continuous-adjoint initial state has zero norm.");

         real_t relative_error = 0.0;
         real_t cosine = 1.0;
         if (ratio == ratios.front())
         {
            reference_initial = initial_adjoint;
            reference_norm = initial_norm;
            MFEM_VERIFY(reference_norm > 0.0 &&
                        std::isfinite(reference_norm),
                        "Continuous-adjoint reference has an invalid norm.");
         }
         else
         {
            difference = initial_adjoint;
            difference -= reference_initial;
            relative_error =
               GlobalVectorNorm(comm, difference) / reference_norm;
            cosine = GlobalVectorDot(
                        comm, initial_adjoint, reference_initial)
                     / (initial_norm * reference_norm);
         }

         if (Mpi::Root())
         {
            std::cout << std::setw(3) << ratio
                      << std::setw(10) << adjoint_steps
                      << "  " << std::scientific << std::setprecision(6)
                      << std::setw(13) << grid.dt_adjoint
                      << "  " << std::setw(13) << initial_norm
                      << "  " << std::setw(13) << relative_error
                      << "  " << std::setw(13) << cosine << "\n";
            csv << ratio << ',' << nsteps_ << ',' << adjoint_steps << ','
                << std::setprecision(16) << grid.dt_forward << ','
                << grid.dt_adjoint << ',' << initial_norm << ','
                << relative_error << ',' << cosine << '\n';
         }
      }

      if (Mpi::Root())
      {
         std::cout << "Results: " << output_parent
                   << "/continuous_adjoint_coarsening.csv\n"
                   << "==================================================\n";
      }
   }

   void AnalyzeContinuousAdjointRefinement(
      const Vector &rho_tv,
      const std::string &output_parent,
      int minimum_refinement = 1,
      int maximum_refinement = 16,
      const Array<int> *active_tdof_list = nullptr)
   {
      MFEM_VERIFY(minimum_refinement >= 1 &&
                  (minimum_refinement & (minimum_refinement - 1)) == 0,
                  "The minimum adjoint refinement must be a positive "
                  "power of two.");
      MFEM_VERIFY(maximum_refinement >= 2,
                  "The cubic-Hermite adjoint-refinement diagnostic requires "
                  "a maximum refinement of at least 2.");
      MFEM_VERIFY(minimum_refinement <= maximum_refinement / 2,
                  "Adjoint refinement requires at least two consecutive "
                  "power-of-two candidates.");
      FilterFSolve(rho_tv);

      ElastodynamicsOperator oper(
         state_fes_, mass_coef_, lambda_coef_, mu_coef_,
         load_spec_.amplitude, load_spec_.duration, load_spec_.time_profile,
         load_spec_.phase, load_spec_.frequency, load_spec_.bdr_attributes,
         load_coef_, load_spec_.domain_load, &gamma_coef_, impedance_,
         exterior_bdr_attr_, ess_bdr_attr_, mass_type_,
         /*print_banner=*/true);

      ValidateLumpedRK4TimeStep(oper, h_, /*print_report=*/true);

      std::vector<Vector> forward_states;
      std::vector<real_t> forward_times;
      const real_t objective_value = RolloutObjective(
         oper, state_fes_, oper.GetBlockOffsets(), objective_,
         x0_, nsteps_, 0.0, h_, &forward_states, &forward_times,
         "coarse forward");

      std::vector<Vector> forward_derivatives;
      BuildForwardStateTimeDerivatives(
         oper, forward_states, 0.0, h_, forward_derivatives);
      CubicHermiteForwardStateProvider state_provider(
         forward_states, forward_derivatives, 0.0, h_);

      std::vector<int> ratios;
      for (int ratio = minimum_refinement;
           ratio <= maximum_refinement; ratio *= 2)
      {
         MFEM_VERIFY(nsteps_ <= std::numeric_limits<int>::max() / ratio,
                     "Adjoint refinement produces too many timesteps.");
         ratios.push_back(ratio);
         if (ratio > std::numeric_limits<int>::max() / 2) { break; }
      }
      MFEM_VERIFY(ratios.size() >= 2,
                  "Adjoint refinement needs at least two consecutive ratios.");

      const MPI_Comm comm = state_fes_.GetComm();
      if (Mpi::Root())
      {
         std::cout
            << "\n=== Fixed-Design Continuous-Adjoint/Gradient Refinement ===\n"
            << "Forward objective (discrete trapezoid report only): "
            << std::scientific << std::setprecision(10)
            << objective_value << "\n"
            << "Forward steps: " << nsteps_
            << ", dt_f: " << h_ << "\n"
            << "Forward storage: " << forward_states.size()
            << " endpoint states + " << forward_derivatives.size()
            << " physical endpoint slopes\n"
            << "Forward states at all adjoint RK stages are reconstructed "
               "with cubic Hermite polynomials. Each candidate accumulates "
               "the production continuous objective and design gradient.\n";
         if (minimum_refinement > 1)
         {
            std::cout << "Incremental refinement leg starts at m="
                      << minimum_refinement
                      << "; combine its adjacent difference with the "
                         "preceding leg to calculate an observed order.\n";
         }
         if (ratios.size() < 3)
         {
            std::cout << "NOTE: this is a two-grid comparison. At least three "
                         "consecutive ratios (possibly across incremental "
                         "legs) are needed for an observed convergence order.\n";
         }
      }

      MFEM_VERIFY(
         ratios.back() <= std::numeric_limits<int>::max() / 2 &&
         nsteps_ <=
         std::numeric_limits<int>::max() / (2 * ratios.back()),
         "Hermite reconstruction audit produces too many timesteps.");
      // Include the midpoint of every finest-adjoint RK4 step, not only its
      // endpoints.  These are exactly the most interpolation-sensitive stage
      // times queried by the reverse RK4 kernel.
      const int reconstruction_refinement = 2 * ratios.back();
      const int reconstruction_reference_steps =
         nsteps_ * reconstruction_refinement;
      if (Mpi::Root())
      {
         std::cout << "Auditing cubic Hermite states against an RK4 reference "
                   << "with dt_ref=" << std::scientific
                   << std::setprecision(6)
                   << h_ / reconstruction_refinement << " ..."
                   << std::flush;
      }
      const double audit_start = MPI_Wtime();
      const ForwardReconstructionAudit reconstruction_audit =
         AuditForwardStateReconstruction(
            oper, state_provider, x0_, 0.0,
            reconstruction_reference_steps,
            h_ / reconstruction_refinement, comm);
      if (Mpi::Root())
      {
         std::cout << " done in " << std::fixed << std::setprecision(2)
                   << (MPI_Wtime() - audit_start) << " s\n"
                   << "  relative trajectory RMS: state="
                   << std::scientific << std::setprecision(6)
                   << reconstruction_audit.state_relative_rms
                   << ", displacement="
                   << reconstruction_audit.displacement_relative_rms
                   << ", velocity="
                   << reconstruction_audit.velocity_relative_rms << "\n";

         std::ofstream reconstruction_csv(
            output_parent + "/cubic_hermite_reconstruction_audit.csv");
         MFEM_VERIFY(reconstruction_csv.is_open(),
                     "Could not open the Hermite reconstruction audit CSV.");
         reconstruction_csv
            << "N_forward,dt_forward,reference_refinement,N_reference,"
               "dt_reference,state_relative_rms,"
               "displacement_relative_rms,velocity_relative_rms\n"
            << nsteps_ << ',' << std::setprecision(16) << h_ << ','
            << reconstruction_refinement << ','
            << reconstruction_reference_steps << ','
            << h_ / reconstruction_refinement << ','
            << reconstruction_audit.state_relative_rms << ','
            << reconstruction_audit.displacement_relative_rms << ','
            << reconstruction_audit.velocity_relative_rms << '\n';
      }

      Vector terminal_adjoint(x0_.Size());
      terminal_adjoint = 0.0;
      std::vector<Vector> initial_adjoints(ratios.size());
      std::vector<Vector> filtered_gradients(ratios.size());
      std::vector<Vector> active_raw_gradients(ratios.size());
      std::vector<real_t> continuous_objectives(ratios.size(), 0.0);
      std::vector<real_t> initial_norms(ratios.size(), 0.0);
      std::vector<real_t> filtered_gradient_norms(ratios.size(), 0.0);
      std::vector<real_t> raw_gradient_norms(ratios.size(), 0.0);
      std::vector<real_t> candidate_seconds(ratios.size(), 0.0);

      for (std::size_t index = 0; index < ratios.size(); index++)
      {
         const int ratio = ratios[index];
         const int adjoint_steps = nsteps_ * ratio;
         const NestedTimeGrid grid =
            NestedTimeGrid::Create(nsteps_ * h_, nsteps_, adjoint_steps);
         MFEM_VERIFY(
            (ratio == 1 &&
             grid.relation == NestedTimeGridRelation::SAME) ||
            (ratio > 1 &&
             grid.relation == NestedTimeGridRelation::ADJOINT_FINER),
            "Refinement diagnostic requires a same or finer adjoint grid.");

         ValidateLumpedRK4TimeStep(
            oper, grid.dt_adjoint, /*print_report=*/false);
         if (Mpi::Root())
         {
            std::cout << "  solving m=" << ratio
                      << " (N_a=" << adjoint_steps
                      << ", dt_a=" << std::scientific
                      << std::setprecision(6) << grid.dt_adjoint << ") ..."
                      << std::flush;
         }
         const double sweep_start = MPI_Wtime();

         // Evaluate the same fourth-order continuous objective used by the
         // production continuous-gradient path, without rerunning the common
         // coarse forward trajectory for every candidate adjoint grid.
         for (int step = 0; step < nsteps_; step++)
         {
            continuous_objectives[index] +=
               EvaluateContinuousObjectiveInterval(
                  state_fes_, oper.GetBlockOffsets(), objective_,
                  state_provider, step * h_, h_, ratio);
         }

         // March p and accumulate dJ/d(rho_tilde) in one reverse sweep.  The
         // prebuilt global Hermite provider reuses the same endpoint slopes for
         // every ratio, so differences isolate the adjoint/design quadrature.
         filtered_gradients[index].SetSize(filter_fes_.GetTrueVSize());
         filtered_gradients[index] = 0.0;
         ContinuousDesignGradientData design_data{
            filter_fes_, rho_tilde_, mat_, filtered_gradients[index]};
         Vector p(terminal_adjoint), p_left(terminal_adjoint.Size());
         for (int step = nsteps_ - 1; step >= 0; step--)
         {
            ReverseContinuousAdjointInterval(
               oper, state_fes_, objective_, state_provider,
               step * h_, h_, ratio, p, p_left, &design_data);
            p = p_left;
         }
         initial_adjoints[index] = p;

         Vector raw_gradient;
         filter_.MultTranspose(filtered_gradients[index], raw_gradient);
         if (active_tdof_list)
         {
            active_raw_gradients[index].SetSize(active_tdof_list->Size());
            for (int i = 0; i < active_tdof_list->Size(); i++)
            {
               active_raw_gradients[index][i] =
                  raw_gradient[(*active_tdof_list)[i]];
            }
         }
         else
         {
            active_raw_gradients[index] = raw_gradient;
         }

         initial_norms[index] =
            GlobalVectorNorm(comm, initial_adjoints[index]);
         filtered_gradient_norms[index] =
            GlobalVectorNorm(comm, filtered_gradients[index]);
         raw_gradient_norms[index] =
            GlobalVectorNorm(comm, active_raw_gradients[index]);
         candidate_seconds[index] = MPI_Wtime() - sweep_start;
         MFEM_VERIFY(
            std::isfinite(continuous_objectives[index]) &&
            std::isfinite(initial_norms[index]) &&
            std::isfinite(filtered_gradient_norms[index]) &&
            std::isfinite(raw_gradient_norms[index]),
            "Continuous objective/adjoint/design diagnostic produced a "
            "non-finite result.");
         if (Mpi::Root())
         {
            std::cout << " done in " << std::fixed << std::setprecision(2)
                      << candidate_seconds[index] << " s"
                      << ", J=" << std::scientific << std::setprecision(6)
                      << continuous_objectives[index] << "\n";
         }
      }

      const real_t nan = std::numeric_limits<real_t>::quiet_NaN();
      const auto make_nan_vector = [&]()
      {
         return std::vector<real_t>(ratios.size(), nan);
      };

      std::vector<real_t> objective_finest_errors = make_nan_vector();
      std::vector<real_t> objective_adjacent_errors = make_nan_vector();
      std::vector<real_t> objective_orders = make_nan_vector();
      objective_finest_errors.back() = 0.0;
      const real_t reference_objective = continuous_objectives.back();
      for (std::size_t index = 0; index + 1 < ratios.size(); index++)
      {
         if (std::abs(reference_objective) > 0.0)
         {
            objective_finest_errors[index] =
               std::abs(continuous_objectives[index] - reference_objective) /
               std::abs(reference_objective);
         }
         if (std::abs(continuous_objectives[index + 1]) > 0.0)
         {
            objective_adjacent_errors[index] =
               std::abs(continuous_objectives[index] -
                        continuous_objectives[index + 1]) /
               std::abs(continuous_objectives[index + 1]);
         }
      }
      for (std::size_t index = 0; index + 2 < ratios.size(); index++)
      {
         if (objective_adjacent_errors[index] > 0.0 &&
             objective_adjacent_errors[index + 1] > 0.0)
         {
            objective_orders[index] =
               std::log(objective_adjacent_errors[index] /
                        objective_adjacent_errors[index + 1])
               / std::log(real_t(2.0));
         }
      }

      struct VectorConvergenceMetrics
      {
         std::vector<real_t> finest_errors;
         std::vector<real_t> adjacent_errors;
         std::vector<real_t> reference_cosines;
         std::vector<real_t> orders;
      };
      const auto compute_vector_metrics =
         [&](const std::vector<Vector> &values,
             const std::vector<real_t> &norms)
         {
            VectorConvergenceMetrics metrics{
               make_nan_vector(), make_nan_vector(),
               make_nan_vector(), make_nan_vector()};
            const Vector &reference = values.back();
            const real_t reference_norm = norms.back();
            metrics.finest_errors.back() = 0.0;
            if (reference_norm > 0.0)
            {
               metrics.reference_cosines.back() = 1.0;
            }

            Vector difference;
            for (std::size_t index = 0;
                 index + 1 < ratios.size(); index++)
            {
               difference = values[index];
               difference -= reference;
               if (reference_norm > 0.0)
               {
                  metrics.finest_errors[index] =
                     GlobalVectorNorm(comm, difference) / reference_norm;
               }
               if (norms[index] > 0.0 && reference_norm > 0.0)
               {
                  metrics.reference_cosines[index] =
                     GlobalVectorDot(comm, values[index], reference) /
                     (norms[index] * reference_norm);
               }

               difference = values[index];
               difference -= values[index + 1];
               if (norms[index + 1] > 0.0)
               {
                  metrics.adjacent_errors[index] =
                     GlobalVectorNorm(comm, difference) / norms[index + 1];
               }
            }
            for (std::size_t index = 0;
                 index + 2 < ratios.size(); index++)
            {
               if (metrics.adjacent_errors[index] > 0.0 &&
                   metrics.adjacent_errors[index + 1] > 0.0)
               {
                  metrics.orders[index] =
                     std::log(metrics.adjacent_errors[index] /
                              metrics.adjacent_errors[index + 1]) /
                     std::log(real_t(2.0));
               }
            }
            return metrics;
         };

      const VectorConvergenceMetrics initial_metrics =
         compute_vector_metrics(initial_adjoints, initial_norms);
      const VectorConvergenceMetrics filtered_metrics =
         compute_vector_metrics(filtered_gradients, filtered_gradient_norms);
      const VectorConvergenceMetrics raw_metrics =
         compute_vector_metrics(active_raw_gradients, raw_gradient_norms);

      if (Mpi::Root())
      {
         std::ofstream csv(
            output_parent + "/continuous_adjoint_refinement.csv");
         MFEM_VERIFY(csv.is_open(),
                     "Could not open the adjoint-refinement CSV output.");
         csv << "refinement,N_forward,N_adjoint,dt_forward,dt_adjoint,"
                "continuous_objective,objective_finest_relative_error,"
                "objective_adjacent_relative_difference,"
                "objective_adjacent_observed_order,"
                "initial_adjoint_norm,initial_adjoint_finest_relative_error,"
                "initial_adjoint_adjacent_relative_difference,"
                "initial_adjoint_reference_cosine,"
                "initial_adjoint_adjacent_observed_order,"
                "filtered_gradient_norm,"
                "filtered_gradient_finest_relative_error,"
                "filtered_gradient_adjacent_relative_difference,"
                "filtered_gradient_reference_cosine,"
                "filtered_gradient_adjacent_observed_order,"
                "active_raw_gradient_norm,"
                "active_raw_gradient_finest_relative_error,"
                "active_raw_gradient_adjacent_relative_difference,"
                "active_raw_gradient_reference_cosine,"
                "active_raw_gradient_adjacent_observed_order,"
                "candidate_seconds\n";

         std::cout
            << "Finest reference: m=" << ratios.back()
            << "\n  m       N_a          dt_a              J"
               "       p.rel       gf.rel       gr.rel       gr.cos"
               "       seconds\n";
         for (std::size_t index = 0; index < ratios.size(); index++)
         {
            const int ratio = ratios[index];
            const int adjoint_steps = nsteps_ * ratio;
            const real_t dt_adjoint = h_ / ratio;
            std::cout << std::setw(3) << ratio
                      << std::setw(10) << adjoint_steps
                      << "  " << std::scientific << std::setprecision(6)
                      << std::setw(13) << dt_adjoint
                      << "  " << std::setw(13)
                      << continuous_objectives[index]
                      << "  " << std::setw(11)
                      << initial_metrics.finest_errors[index]
                      << "  " << std::setw(11)
                      << filtered_metrics.finest_errors[index]
                      << "  " << std::setw(11)
                      << raw_metrics.finest_errors[index]
                      << "  " << std::setw(11)
                      << raw_metrics.reference_cosines[index]
                      << "  " << std::fixed << std::setprecision(2)
                      << std::setw(9) << candidate_seconds[index] << "\n";

            csv << ratio << ',' << nsteps_ << ',' << adjoint_steps << ','
                << std::setprecision(16) << h_ << ',' << dt_adjoint << ','
                << continuous_objectives[index] << ','
                << objective_finest_errors[index] << ','
                << objective_adjacent_errors[index] << ','
                << objective_orders[index] << ','
                << initial_norms[index] << ','
                << initial_metrics.finest_errors[index] << ','
                << initial_metrics.adjacent_errors[index] << ','
                << initial_metrics.reference_cosines[index] << ','
                << initial_metrics.orders[index] << ','
                << filtered_gradient_norms[index] << ','
                << filtered_metrics.finest_errors[index] << ','
                << filtered_metrics.adjacent_errors[index] << ','
                << filtered_metrics.reference_cosines[index] << ','
                << filtered_metrics.orders[index] << ','
                << raw_gradient_norms[index] << ','
                << raw_metrics.finest_errors[index] << ','
                << raw_metrics.adjacent_errors[index] << ','
                << raw_metrics.reference_cosines[index] << ','
                << raw_metrics.orders[index] << ','
                << candidate_seconds[index] << '\n';
         }
         std::cout << "Results: " << output_parent
                   << "/continuous_adjoint_refinement.csv\n"
                   << "==================================================\n";
      }
   }

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
      const bool first_operator = !banner_printed_;
      oper_ = std::make_unique<ElastodynamicsOperator>(
                 state_fes_, mass_coef_, lambda_coef_, mu_coef_,
                 load_spec_.amplitude, load_spec_.duration, load_spec_.time_profile,
                 load_spec_.phase, load_spec_.frequency, load_spec_.bdr_attributes,
                 load_coef_, load_spec_.domain_load, &gamma_coef_, impedance_,
                 exterior_bdr_attr_, ess_bdr_attr_, mass_type_,
                 /*print_banner=*/first_operator);

      // The design-dependent M/K pair changes after every MMA update.  Recheck
      // the assembled operator each iteration; only the first (or a noteworthy
      // near-limit value) needs a full report.
      const real_t stability_dt =
         (adjoint_mode_ == TransientAdjointMode::CONTINUOUS) ?
         std::max(time_grid_.dt_forward, time_grid_.dt_adjoint) :
         time_grid_.dt_forward;
      ValidateLumpedRK4TimeStep(
         *oper_, stability_dt, /*print_report=*/first_operator);
      banner_printed_ = true;

      checkpoint_.reset();
      checkpoint_state_.SetSize(0);
      full_forward_states_.clear();
      initial_adjoint_.SetSize(0);
      continuous_telemetry_ = ContinuousStorageTelemetry{};

      if (trajectory_storage_mode_ == TrajectoryStorageMode::REVOLVE)
      {
         checkpoint_ = std::make_unique<TrajectoryCheckpointing<>>(
            ScheduledTrajectoryBlocks(), num_checkpoints_, x0_.Size(),
            /*start_time=*/0.0, ScheduledTrajectoryStep());
         continuous_telemetry_.trajectory_memory_mb =
            EstimatedTrajectoryMemoryMB();
      }
      else
      {
         continuous_telemetry_.trajectory_memory_mb =
            FullTrajectoryMemoryMB();
      }

      if (Mpi::Root())
      {
         std::cout << "    [it " << outer_it_ + 1 << "] forward sweep ("
                   << "N_f=" << time_grid_.forward_steps
                   << ", N_a=" << time_grid_.adjoint_steps
                   << ", " << NestedTimeGridRelationName(time_grid_.relation)
                   << ", dt_f=" << std::scientific
                   << std::setprecision(6) << time_grid_.dt_forward
                   << ", dt_a=" << time_grid_.dt_adjoint
                   << ", " << TransientAdjointModeName(adjoint_mode_)
                   << ", storage="
                   << TrajectoryStorageModeName(trajectory_storage_mode_)
                   << ")\n";
         if (trajectory_storage_mode_ == TrajectoryStorageMode::REVOLVE)
         {
            checkpoint_->PrintInfo();
         }
         else
         {
            std::cout << "      full trajectory: "
                      << time_grid_.forward_steps + 1
                      << " endpoint states, estimated " << std::fixed
                      << std::setprecision(2)
                      << continuous_telemetry_.trajectory_memory_mb
                      << " MB per rank\n";
         }
      }

      const double forward_start = MPI_Wtime();
      real_t objective_value = 0.0;
      if (rk4_stage_objective_)
      {
         objective_value =
            trajectory_storage_mode_ == TrajectoryStorageMode::REVOLVE ?
            RolloutRK4StageObjectiveCheckpointed() :
            RolloutRK4StageObjectiveFullStorage();
      }
      else if (adjoint_mode_ == TransientAdjointMode::CONTINUOUS)
      {
         objective_value =
            trajectory_storage_mode_ == TrajectoryStorageMode::REVOLVE ?
            RolloutContinuousObjectiveCheckpointed() :
            RolloutContinuousObjectiveFullStorage();
      }
      else
      {
         objective_value = RolloutObjectiveCheckpointed();
      }
      continuous_telemetry_.forward_seconds = MPI_Wtime() - forward_start;
      return objective_value;
   }

   // 3. Adjoint physics: selected discrete or continuous full/checkpointed
   //    sweep -> dJ/d(rho_tilde).
   void PhysicsASolve()
   {
      MFEM_VERIFY(oper_, "PhysicsASolve() requires a preceding PhysicsFSolve().");
      if (trajectory_storage_mode_ == TrajectoryStorageMode::REVOLVE)
      {
         MFEM_VERIFY(checkpoint_,
                     "PhysicsASolve() requires initialized checkpointing.");
      }
      else
      {
         MFEM_VERIFY(
            full_forward_states_.size() ==
            static_cast<std::size_t>(time_grid_.forward_steps + 1),
            "PhysicsASolve() requires a complete full forward trajectory.");
      }

      if (Mpi::Root())
      {
         std::cout << "    [it " << outer_it_ + 1 << "] adjoint sweep ("
                   << time_grid_.adjoint_steps << " steps, "
                   << NestedTimeGridRelationName(time_grid_.relation)
                   << ", storage="
                   << TrajectoryStorageModeName(trajectory_storage_mode_)
                   << ")\n";
      }

      const double adjoint_start = MPI_Wtime();
      if (adjoint_mode_ == TransientAdjointMode::CONTINUOUS)
      {
         if (trajectory_storage_mode_ == TrajectoryStorageMode::REVOLVE)
         {
            RunContinuousAdjointDesignCheckpointed();
         }
         else
         {
            RunContinuousAdjointDesignFullStorage();
         }
      }
      else
      {
         AdjointDesignSweepCheckpointed();
      }
      continuous_telemetry_.adjoint_seconds = MPI_Wtime() - adjoint_start;

      if (Mpi::Root() &&
          adjoint_mode_ == TransientAdjointMode::CONTINUOUS)
      {
         std::cout << "      trajectory telemetry: forward=" << std::fixed
                   << std::setprecision(3)
                   << continuous_telemetry_.forward_seconds
                   << " s, adjoint="
                   << continuous_telemetry_.adjoint_seconds
                   << " s, storage="
                   << continuous_telemetry_.trajectory_memory_mb << " MB";
         if (trajectory_storage_mode_ == TrajectoryStorageMode::REVOLVE)
         {
            std::cout << ", controller blocks/intervals="
                      << continuous_telemetry_.controller_replayed_blocks
                      << '/'
                      << continuous_telemetry_.controller_replayed_intervals
                      << ", local blocks/intervals="
                      << continuous_telemetry_.locally_replayed_blocks
                      << '/'
                      << continuous_telemetry_.locally_replayed_intervals;
         }
         std::cout << "\n";
      }
   }

   // 4. Adjoint filter: transpose the filter, dJ/d(rho_tilde) -> dJ/d(rho).
   void FilterASolve(Vector &dJ_drho)
   {
      filter_.MultTranspose(dJ_drho_tilde_, dJ_drho);
      MFEM_VERIFY(dJ_drho.Size() == control_fes_.GetTrueVSize(),
                  "Raw design gradient has unexpected size.");
   }

   // Once FilterASolve has materialized the raw-design gradient, neither the
   // design-dependent physics matrices nor the REVOLVE snapshots are needed.
   // Release them before an optional visualization sweep constructs its own
   // operator; otherwise the two operators plus checkpoint storage overlap at
   // peak memory.
   void ReleasePhysicsIterationStorage()
   {
      checkpoint_.reset();
      oper_.reset();
      checkpoint_state_.SetSize(0);
      full_forward_states_.clear();
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
      ReleasePhysicsIterationStorage();
      return J;
   }

   // Fixed-design production-path audit.  This deliberately invokes the same
   // ObjectiveAndGradient() entry point used by MMA once with every forward
   // endpoint retained and once with REVOLVE checkpoint/replay.  Consequently
   // the comparison covers the objective, p(0), the continuous design
   // integral, and the filter transpose rather than only an interval kernel.
   void AnalyzeContinuousStorageEquivalence(
      const Vector &rho_tv,
      const std::string &output_parent,
      const Array<int> *active_tdof_list = nullptr)
   {
      MFEM_VERIFY(
         adjoint_mode_ == TransientAdjointMode::CONTINUOUS,
         "The storage-equivalence diagnostic requires a continuous adjoint.");
      MFEM_VERIFY(
         trajectory_storage_mode_ == TrajectoryStorageMode::REVOLVE &&
         num_checkpoints_ > 0,
         "The storage-equivalence diagnostic must be initialized with "
         "REVOLVE storage and at least one checkpoint.");

      struct StorageResult
      {
         real_t objective = 0.0;
         Vector initial_adjoint;
         Vector filtered_gradient;
         Vector active_raw_gradient;
         ContinuousStorageTelemetry telemetry;
         real_t total_seconds = 0.0;
      };

      const MPI_Comm comm = state_fes_.GetComm();
      const int revolve_checkpoints = num_checkpoints_;

      const auto global_max_real = [&](real_t value)
      {
         real_t maximum = 0.0;
         MPI_Allreduce(&value, &maximum, 1,
                       MPITypeMap<real_t>::mpi_type, MPI_MAX, comm);
         return maximum;
      };
      const auto global_max_count = [&](long long value)
      {
         long long maximum = 0;
         MPI_Allreduce(&value, &maximum, 1, MPI_LONG_LONG, MPI_MAX, comm);
         return maximum;
      };

      const auto run_production_path =
         [&](TrajectoryStorageMode mode)
         {
            ReleasePhysicsIterationStorage();
            trajectory_storage_mode_ = mode;
            num_checkpoints_ =
               mode == TrajectoryStorageMode::REVOLVE ?
               revolve_checkpoints : 0;

            StorageResult result;
            Vector raw_gradient;
            const double start = MPI_Wtime();
            result.objective =
               ObjectiveAndGradient(rho_tv, raw_gradient, /*outer_it=*/0);
            result.total_seconds =
               global_max_real(static_cast<real_t>(MPI_Wtime() - start));
            result.initial_adjoint = InitialAdjoint();
            result.filtered_gradient = FilteredDesignGradient();
            if (active_tdof_list)
            {
               result.active_raw_gradient.SetSize(active_tdof_list->Size());
               for (int i = 0; i < active_tdof_list->Size(); i++)
               {
                  result.active_raw_gradient[i] =
                     raw_gradient[(*active_tdof_list)[i]];
               }
            }
            else
            {
               result.active_raw_gradient = raw_gradient;
            }

            result.telemetry = StorageTelemetry();
            result.telemetry.forward_seconds =
               global_max_real(result.telemetry.forward_seconds);
            result.telemetry.adjoint_seconds =
               global_max_real(result.telemetry.adjoint_seconds);
            result.telemetry.trajectory_memory_mb =
               global_max_real(result.telemetry.trajectory_memory_mb);
            result.telemetry.controller_replayed_blocks =
               global_max_count(
                  result.telemetry.controller_replayed_blocks);
            result.telemetry.locally_replayed_blocks =
               global_max_count(result.telemetry.locally_replayed_blocks);
            result.telemetry.controller_replayed_intervals =
               global_max_count(
                  result.telemetry.controller_replayed_intervals);
            result.telemetry.locally_replayed_intervals =
               global_max_count(result.telemetry.locally_replayed_intervals);

            // ObjectiveAndGradient() already releases the operator, retained
            // trajectory, and checkpoints. Keep this explicit so future
            // changes cannot accidentally overlap the two large cases.
            ReleasePhysicsIterationStorage();
            return result;
         };

      if (Mpi::Root())
      {
         std::cout
            << "\n=== Continuous Full-Storage/REVOLVE Equivalence ===\n"
            << "Fixed design; both cases use the production "
               "ObjectiveAndGradient path.\n"
            << "Time grid: N_f=" << time_grid_.forward_steps
            << ", N_a=" << time_grid_.adjoint_steps
            << ", dt_f=" << std::scientific << std::setprecision(6)
            << time_grid_.dt_forward
            << ", dt_a=" << time_grid_.dt_adjoint
            << "; REVOLVE checkpoints=" << revolve_checkpoints << "\n"
            << "  running FULL ...\n";
      }
      StorageResult full =
         run_production_path(TrajectoryStorageMode::FULL);

      if (Mpi::Root())
      {
         std::cout << "  FULL storage released; running REVOLVE ...\n";
      }
      StorageResult revolve =
         run_production_path(TrajectoryStorageMode::REVOLVE);

      // Leave the solver in its original configuration even though the driver
      // exits immediately after this diagnostic.
      trajectory_storage_mode_ = TrajectoryStorageMode::REVOLVE;
      num_checkpoints_ = revolve_checkpoints;

      struct VectorMetrics
      {
         real_t full_norm = 0.0;
         real_t revolve_norm = 0.0;
         real_t relative_error = 0.0;
         real_t cosine = 1.0;
      };
      const auto compare_vectors =
         [&](const Vector &full_vector, const Vector &revolve_vector)
         {
            MFEM_VERIFY(full_vector.Size() == revolve_vector.Size(),
                        "Storage-equivalence vectors have different sizes.");
            VectorMetrics metrics;
            metrics.full_norm = GlobalVectorNorm(comm, full_vector);
            metrics.revolve_norm = GlobalVectorNorm(comm, revolve_vector);
            Vector difference(revolve_vector);
            difference -= full_vector;
            const real_t difference_norm =
               GlobalVectorNorm(comm, difference);
            if (metrics.full_norm > 0.0)
            {
               metrics.relative_error =
                  difference_norm / metrics.full_norm;
               metrics.cosine = metrics.revolve_norm > 0.0 ?
                  GlobalVectorDot(comm, full_vector, revolve_vector) /
                  (metrics.full_norm * metrics.revolve_norm) :
                  std::numeric_limits<real_t>::quiet_NaN();
            }
            else if (metrics.revolve_norm == 0.0)
            {
               metrics.relative_error = 0.0;
               metrics.cosine = 1.0;
            }
            else
            {
               metrics.relative_error =
                  std::numeric_limits<real_t>::infinity();
               metrics.cosine =
                  std::numeric_limits<real_t>::quiet_NaN();
            }
            return metrics;
         };

      const real_t objective_denominator = std::abs(full.objective);
      const real_t objective_difference =
         std::abs(revolve.objective - full.objective);
      const real_t objective_relative_error =
         objective_denominator > 0.0 ?
         objective_difference / objective_denominator :
         (objective_difference == 0.0 ? 0.0 :
          std::numeric_limits<real_t>::infinity());
      const VectorMetrics initial_metrics = compare_vectors(
         full.initial_adjoint, revolve.initial_adjoint);
      const VectorMetrics filtered_metrics = compare_vectors(
         full.filtered_gradient, revolve.filtered_gradient);
      const VectorMetrics raw_metrics = compare_vectors(
         full.active_raw_gradient, revolve.active_raw_gradient);

      if (Mpi::Root())
      {
         std::cout << "  quantity                    FULL        REVOLVE"
                      "      rel.error         cosine\n"
                   << std::scientific << std::setprecision(6)
                   << "  J                 " << std::setw(13)
                   << full.objective << "  " << std::setw(13)
                   << revolve.objective << "  " << std::setw(13)
                   << objective_relative_error << "            n/a\n"
                   << "  ||p(0)||          " << std::setw(13)
                   << initial_metrics.full_norm << "  " << std::setw(13)
                   << initial_metrics.revolve_norm << "  " << std::setw(13)
                   << initial_metrics.relative_error << "  " << std::setw(13)
                   << initial_metrics.cosine << "\n"
                   << "  ||g_filtered||    " << std::setw(13)
                   << filtered_metrics.full_norm << "  " << std::setw(13)
                   << filtered_metrics.revolve_norm << "  " << std::setw(13)
                   << filtered_metrics.relative_error << "  " << std::setw(13)
                   << filtered_metrics.cosine << "\n"
                   << "  ||g_active_raw||  " << std::setw(13)
                   << raw_metrics.full_norm << "  " << std::setw(13)
                   << raw_metrics.revolve_norm << "  " << std::setw(13)
                   << raw_metrics.relative_error << "  " << std::setw(13)
                   << raw_metrics.cosine << "\n"
                   << std::fixed << std::setprecision(3)
                   << "  max-rank seconds (forward/adjoint/total): FULL "
                   << full.telemetry.forward_seconds << '/'
                   << full.telemetry.adjoint_seconds << '/'
                   << full.total_seconds << ", REVOLVE "
                   << revolve.telemetry.forward_seconds << '/'
                   << revolve.telemetry.adjoint_seconds << '/'
                   << revolve.total_seconds << "\n"
                   << "  estimated max MB/rank: FULL "
                   << full.telemetry.trajectory_memory_mb
                   << ", REVOLVE "
                   << revolve.telemetry.trajectory_memory_mb << "\n"
                   << "  REVOLVE replay blocks/intervals: controller "
                   << revolve.telemetry.controller_replayed_blocks << '/'
                   << revolve.telemetry.controller_replayed_intervals
                   << ", local "
                   << revolve.telemetry.locally_replayed_blocks << '/'
                   << revolve.telemetry.locally_replayed_intervals << "\n";

         std::ofstream csv(
            output_parent + "/continuous_storage_equivalence.csv");
         MFEM_VERIFY(csv.is_open(),
                     "Could not open the storage-equivalence CSV output.");
         csv
            << "N_forward,N_adjoint,dt_forward,dt_adjoint,"
               "revolve_checkpoints,J_full,J_revolve,J_relative_error,"
               "p0_full_norm,p0_revolve_norm,p0_relative_vector_error,"
               "p0_cosine,filtered_gradient_full_norm,"
               "filtered_gradient_revolve_norm,"
               "filtered_gradient_relative_vector_error,"
               "filtered_gradient_cosine,active_raw_gradient_full_norm,"
               "active_raw_gradient_revolve_norm,"
               "active_raw_gradient_relative_vector_error,"
               "active_raw_gradient_cosine,full_forward_s,full_adjoint_s,"
               "full_total_s,full_max_trajectory_MB_per_rank,"
               "revolve_forward_s,revolve_adjoint_s,revolve_total_s,"
               "revolve_max_trajectory_MB_per_rank,"
               "revolve_controller_replayed_blocks,"
               "revolve_locally_replayed_blocks,"
               "revolve_controller_replayed_intervals,"
               "revolve_locally_replayed_intervals\n"
            << time_grid_.forward_steps << ','
            << time_grid_.adjoint_steps << ',' << std::setprecision(16)
            << time_grid_.dt_forward << ',' << time_grid_.dt_adjoint << ','
            << revolve_checkpoints << ',' << full.objective << ','
            << revolve.objective << ',' << objective_relative_error << ','
            << initial_metrics.full_norm << ','
            << initial_metrics.revolve_norm << ','
            << initial_metrics.relative_error << ','
            << initial_metrics.cosine << ','
            << filtered_metrics.full_norm << ','
            << filtered_metrics.revolve_norm << ','
            << filtered_metrics.relative_error << ','
            << filtered_metrics.cosine << ',' << raw_metrics.full_norm << ','
            << raw_metrics.revolve_norm << ','
            << raw_metrics.relative_error << ',' << raw_metrics.cosine << ','
            << full.telemetry.forward_seconds << ','
            << full.telemetry.adjoint_seconds << ',' << full.total_seconds << ','
            << full.telemetry.trajectory_memory_mb << ','
            << revolve.telemetry.forward_seconds << ','
            << revolve.telemetry.adjoint_seconds << ','
            << revolve.total_seconds << ','
            << revolve.telemetry.trajectory_memory_mb << ','
            << revolve.telemetry.controller_replayed_blocks << ','
            << revolve.telemetry.locally_replayed_blocks << ','
            << revolve.telemetry.controller_replayed_intervals << ','
            << revolve.telemetry.locally_replayed_intervals << '\n';
         std::cout << "Results: " << output_parent
                   << "/continuous_storage_equivalence.csv\n"
                   << "=====================================================\n";
      }
   }

   // Forward-only objective J(rho) (no gradient / no stored trajectory).
   real_t Objective(const Vector &rho_tv, const char *progress_label = nullptr)
   {
      if (rk4_stage_objective_)
      {
         FilterFSolve(rho_tv);
         const bool first_operator = !banner_printed_;
         ElastodynamicsOperator objective_oper(
            state_fes_, mass_coef_, lambda_coef_, mu_coef_,
            load_spec_.amplitude, load_spec_.duration,
            load_spec_.time_profile, load_spec_.phase, load_spec_.frequency,
            load_spec_.bdr_attributes, load_coef_, load_spec_.domain_load,
            &gamma_coef_, impedance_, exterior_bdr_attr_, ess_bdr_attr_,
            mass_type_, /*print_banner=*/first_operator);
         ValidateLumpedRK4TimeStep(
            objective_oper, h_, /*print_report=*/first_operator);
         banner_printed_ = true;
         (void)progress_label;
         return RK4StageObjectiveForwardSweepStreaming(
            objective_oper, state_fes_, objective_, x0_, nsteps_,
            /*start_time=*/0.0, h_);
      }
      const int continuous_substeps =
         adjoint_mode_ == TransientAdjointMode::CONTINUOUS ?
         ObjectiveSubstepsPerForwardInterval() : 0;
      const real_t stability_step =
         adjoint_mode_ == TransientAdjointMode::CONTINUOUS ?
         std::max(time_grid_.dt_forward, time_grid_.dt_adjoint) : h_;
      return EvaluateDesignObjective(
                rho_tv, x0_, state_fes_, control_fes_, rho_, rho_tilde_, filter_,
                gamma_coef_, exterior_bdr_attr_, ess_bdr_attr_, objective_, mat_,
                load_spec_, load_coef_, impedance_, nsteps_, h_, mass_type_,
                progress_label, /*validate_cfl=*/true,
                continuous_substeps, stability_step);
   }

   // Forward-only sweep with sampled-state callbacks. Unlike
   // ForwardVisualizationSweep(), this never retains every state in memory;
   // the caller can write only the frames it needs as the RK4 sweep advances.
   // The returned value is the same time-integrated objective as Objective().
   real_t ForwardVisualizationSweepStream(
      const Vector &rho_tv, int sample_every,
      const std::function<void(int, real_t, const Vector &)> &save_state,
      bool refilter_design = true)
   {
      MFEM_VERIFY(sample_every >= 1,
                  "Forward visualization sample interval must be positive.");

      // Forward-only callers need the filter here.  The optimization driver
      // may already have refreshed rho_tilde to save a post-MMA design
      // snapshot immediately before this sweep; let it avoid an identical
      // second Helmholtz solve.
      if (refilter_design) { FilterFSolve(rho_tv); }
      Vector x(x0_);

      const bool first_operator = !banner_printed_;
      ElastodynamicsOperator viz_oper(
         state_fes_, mass_coef_, lambda_coef_, mu_coef_,
         load_spec_.amplitude, load_spec_.duration, load_spec_.time_profile,
         load_spec_.phase, load_spec_.frequency, load_spec_.bdr_attributes,
         load_coef_, load_spec_.domain_load, &gamma_coef_, impedance_,
         exterior_bdr_attr_, ess_bdr_attr_, mass_type_,
         /*print_banner=*/first_operator);
      const real_t stability_step =
         adjoint_mode_ == TransientAdjointMode::CONTINUOUS ?
         std::max(time_grid_.dt_forward, time_grid_.dt_adjoint) : h_;
      ValidateLumpedRK4TimeStep(viz_oper, stability_step,
                                /*print_report=*/first_operator);
      banner_printed_ = true;

      objective_.Reset();
      save_state(0, 0.0, x);
      if (rk4_stage_objective_)
      {
         const auto save_stage_endpoint =
            [&](int completed, real_t time, const Vector &state)
            {
               if (completed > 0 &&
                   (completed == nsteps_ || completed % sample_every == 0))
               {
                  save_state(completed, time, state);
               }
            };
         return RK4StageObjectiveForwardSweepStreaming(
            viz_oper, state_fes_, objective_, x, nsteps_,
            /*start_time=*/0.0, h_, save_stage_endpoint);
      }
      if (adjoint_mode_ == TransientAdjointMode::CONTINUOUS)
      {
         const auto save_continuous_endpoint =
            [&](int completed, const Vector &state)
            {
               if (completed == nsteps_ || completed % sample_every == 0)
               {
                  save_state(completed, completed * h_, state);
               }
            };
         return ContinuousForwardSweepStreaming(
            viz_oper, state_fes_, objective_, x, nsteps_,
            /*start_time=*/0.0, h_, ObjectiveSubstepsPerForwardInterval(),
            /*progress_label=*/"forward", save_continuous_endpoint);
      }

      const int total_steps = nsteps_ + 1;
      AddObjectiveContribution(state_fes_, viz_oper.GetBlockOffsets(), objective_, x,
                               h_, 0, total_steps);

      RK4Solver solver;
      solver.Init(viz_oper);
      real_t t = 0.0;
      const double phase_t0 = MPI_Wtime();
      const int report_every = std::max(1, nsteps_ / 10);

      for (int step = 0; step < nsteps_; step++)
      {
         real_t dt = h_;
         solver.Step(x, t, dt);
         AddObjectiveContribution(state_fes_, viz_oper.GetBlockOffsets(), objective_, x,
                                  h_, step + 1, total_steps);

         const int completed = step + 1;
         if (completed == nsteps_ || completed % sample_every == 0)
         {
            save_state(completed, t, x);
         }

         if (completed % report_every == 0 || completed == nsteps_)
         {
            const int n_disp = x.Size() / 2;
            real_t local_max_u = 0.0;
            int local_nonfinite = 0;
            for (int j = 0; j < x.Size(); j++)
            {
               if (!std::isfinite(x[j])) { local_nonfinite = 1; }
               if (j < n_disp)
               {
                  local_max_u = std::max(local_max_u, std::abs(x[j]));
               }
            }
            real_t global_max_u = 0.0;
            int global_nonfinite = 0;
            MPI_Allreduce(&local_max_u, &global_max_u, 1,
                          MPITypeMap<real_t>::mpi_type, MPI_MAX,
                          state_fes_.GetComm());
            MPI_Allreduce(&local_nonfinite, &global_nonfinite, 1, MPI_INT, MPI_MAX,
                          state_fes_.GetComm());
            MFEM_VERIFY(global_nonfinite == 0,
                        "Forward visualization produced a non-finite state.");

            if (Mpi::Root())
            {
               std::cout << "      forward " << std::setw(6) << completed
                         << '/' << nsteps_ << "  (" << std::setw(3)
                         << (100 * completed / nsteps_) << "%)   " << std::fixed
                         << std::setprecision(2) << (MPI_Wtime() - phase_t0) << " s"
                         << "   max|u| = " << std::scientific
                         << std::setprecision(3) << global_max_u << "\n";
            }
         }
      }

      return objective_.GetObjective();
   }

   // NOTE: GetFinalState(), GetState(t), GetNumStates() removed with checkpointing.
   // With REVOLVE, states are not stored - they're checkpointed and recomputed as needed.
   // If visualization of intermediate states is needed, run a forward-only sweep separately.

   // Forward-only sweep that stores ALL states (for visualization only).
   // WARNING: This defeats the memory savings from checkpointing!
   // Only use for first/last iteration visualization.
   void ForwardVisualizationSweep(const Vector &rho_tv,
                                   std::vector<Vector> &states_out,
                                   std::vector<real_t> &times_out)
   {
      states_out.clear();
      times_out.clear();
      states_out.reserve(nsteps_ + 1);
      times_out.reserve(nsteps_ + 1);

      const int state_size = x0_.Size();
      Vector x(x0_);  // Initial condition

      // Store initial state
      states_out.push_back(x);
      times_out.push_back(0.0);

      // Filter design (rho_tv already set by caller, rho_tilde_ and SIMP
      // coefficients are already configured by preceding FilterFSolve)
      // No need to filter again - SIMP coefficients already reference current rho_tilde_

      // Create physics operator (same as PhysicsFSolve, but don't store in oper_)
      ElastodynamicsOperator viz_oper(
         state_fes_, mass_coef_, lambda_coef_, mu_coef_,
         load_spec_.amplitude, load_spec_.duration, load_spec_.time_profile,
         load_spec_.phase, load_spec_.frequency, load_spec_.bdr_attributes,
         load_coef_, load_spec_.domain_load, &gamma_coef_, impedance_,
         exterior_bdr_attr_, ess_bdr_attr_, mass_type_,
         /*print_banner=*/false);

      // RK4 time integration
      RK4Solver solver;
      solver.Init(viz_oper);
      real_t t = 0.0;

      for (int i = 0; i < nsteps_; i++)
      {
         solver.Step(x, t, h_);
         states_out.push_back(x);
         times_out.push_back(t);
      }
   }

   // Get timestep size
   real_t GetTimeStep() const { return h_; }

   // Get number of timesteps
   int GetNumSteps() const { return nsteps_; }
};

} // namespace mfem

#endif // ELASTODYNAMICS_SOLVER_HPP
