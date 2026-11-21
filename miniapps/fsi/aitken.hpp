#ifndef MFEM_AITKEN_HPP
#define MFEM_AITKEN_HPP

#include <mfem.hpp>

namespace mfem
{
namespace fsi
{

/// Aitken relaxation method for accelerating fixed-point iterations
/// Computes relaxation parameter: ω_{i+1} = -ω_i * (r_{i}^T (r_{i+1} - r_{i})) / ||r_{i+1} - r_{i}||² 
/// see (44) in
/// "Küttler, U., Wall, W.A. Fixed-point fluid–structure interaction solvers with dynamic relaxation. Comput Mech 43, 61–72 (2008). https://doi.org/10.1007/s00466-008-0255-5"
class AitkenRelaxation
{
private:
   real_t last_omega;             ///< Current relaxation parameter
   ODEStateDataVector *residuals; ///< Circular buffer for residuals
   real_t omega_init;             ///< Initial relaxation parameter
   real_t omega_min;              ///< Minimum allowed relaxation parameter
   real_t omega_max;              ///< Maximum allowed relaxation parameter
   real_t safety_factor = 1.15;   ///< Safety factor for relaxation parameter
   int vsize;                     ///< Size of the solution vector (used for residuals buffer)
   int init_steps;                ///< Number of initial steps using constant relaxation
   int ss = 0;                        ///< Stage counter for residuals
   real_t last_r_i_norm_sq = 0; ///< Norm of the last residual (r_i) squared

   mutable Vector x_prev; ///< Previous solution vector (used for computing residuals)
   mutable Vector r;      ///< Current residual vector (r_i = x - x_prev)
   const int max_history = 2; ///< Maximum number of residuals to store
   const real_t TOL = 1e-10;  ///< Tolerance for numerical stability in Aitken's method
   const real_t REL_TOL = 1e-10; ///< Relative tolerance for convergence checks

   #ifdef MFEM_USE_MPI
   MPI_Comm comm; ///< MPI communicator for parallel operations
   #endif

   /// Aitken step w.r.t. the stored x_prev. Private: x_prev is overwritten here with the *unrelaxed* iterate, so
   /// calling this directly gives a wrong residual whenever w != 1 (use the two-argument overload).
   real_t ComputeRelaxation(const Vector& x);

public:
   /// Constructor
   /// @param omega_init Initial relaxation parameter (default: 0.1)
   /// @param omega_min Minimum relaxation parameter (default: 0.1)
   /// @param omega_max Maximum relaxation parameter (default: 1.5)
   /// @param init_steps Number of initial steps using constant relaxation (default: 2)
   AitkenRelaxation(real_t omega_init = 0.1, real_t omega_min = 0.1, real_t omega_max = 1.5, int init_steps_ = 2);

#ifdef MFEM_USE_MPI
   AitkenRelaxation(MPI_Comm comm, real_t omega_init = 0.1, real_t omega_min = 0.1, real_t omega_max = 1.5, int init_steps_ = 2);
#endif

   /// Destructor
   ~AitkenRelaxation();

   /// Set the size of the solution vector
   /// @param x0 Initial solution vector
   void SetSize(const Vector& x0);

   // Set the safety factor for the relaxation method
   /// @param safety_factor_ Safety factor for the relaxation method (default: 1.0)
   void SetSafetyFactor(real_t safety_factor_) { safety_factor = safety_factor_; }

   /// Aitken (Kuttler & Wall) relaxation with the fixed-point residual r_i = x_tilde - x_relaxed_prev,
   /// where x_tilde is the new unrelaxed iterate and x_relaxed_prev the previous relaxed iterate.
   real_t ComputeRelaxation(const Vector& x_tilde, const Vector& x_relaxed_prev)
   {
      x_prev = x_relaxed_prev;
      return ComputeRelaxation(x_tilde);
   }

   /// Get current relaxation parameter
   real_t GetOmega() const { return last_omega; }

   /// Reset the relaxation method (for new time step or problem)
   void Reset();

   /// Set bounds for relaxation parameter
   void SetBounds(real_t omega_min, real_t omega_max);
};

} // namespace fsi

} // namespace mfem

#endif

