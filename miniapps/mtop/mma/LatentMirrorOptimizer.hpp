/**
 * @file LatentMirrorOptimizer.hpp
 * @brief Latent-variable mirror-gradient optimizer for bound-constrained
 *        minimization with a general positive-definite inner product.
 *
 * Implements the algorithm described in:
 *   "A Latent-Variable Mirror-Gradient Algorithm for Bound-Constrained
 *    Minimization with a General Positive Definite Inner Product"
 *
 * The method solves
 * @code
 *   min   Φ(λ)   s.t.  lᵢ ≤ λᵢ ≤ uᵢ,   i = 1 … m
 * @endcode
 * by working entirely in the unconstrained latent space z ∈ ℝⁿ and mapping
 * back to the feasible primal λ = T(z) ∈ int(K) via invertible Legendre maps:
 *
 *   Unbounded  (lᵢ=-∞, uᵢ=+∞):  zᵢ = λᵢ                         (identity)
 *   LowerOnly  (lᵢ>-∞, uᵢ=+∞):  zᵢ = log(λᵢ − lᵢ)
 *   UpperOnly  (lᵢ=-∞, uᵢ<+∞):  zᵢ = −log(uᵢ − λᵢ)
 *   TwoSided   (lᵢ>-∞, uᵢ<+∞):  zᵢ = log((λᵢ−lᵢ)/(uᵢ−λᵢ))
 *
 * @section user_interface User interface
 * The user holds a latent vector z (not the primal λ).  The optimizer steps
 * z in-place; the caller reads the primal value on demand via LatentToPrimal.
 *
 * @code
 *   // --- serial, identity M ---
 *   mfem::Vector z(n);                       // latent variable (user-owned)
 *   PrimalToLatent(lam0, lo, hi, types, z);  // initialize from a primal point
 *
 *   LatentMirrorOptimizer opt(z, lo, hi);
 *
 *   for (int k = 0; k < max_iter; ++k) {
 *       mfem::Vector lam(n);
 *       LatentToPrimal(z, lo, hi, types, lam);       // λ = T(z)
 *       double phi = EvalObjective(lam);
 *       mfem::Vector d(n);
 *       EvalGradient(lam, d);                        // d = ∇_E Φ(λ)
 *       opt.Update(z, d, phi,
 *           [&](const mfem::Vector& z_trial, mfem::real_t& phi_out) {
 *               mfem::Vector lt(n);
 *               LatentToPrimal(z_trial, lo, hi, types, lt);
 *               phi_out = EvalObjective(lt);
 *           });
 *       if (opt.StationarityResidual() < tol) break;
 *   }
 * @endcode
 *
 * @section inner_product Inner product
 * The M-gradient is gᵏ = M⁻¹ dᵏ.  Pass an mfem::Operator for M and an
 * mfem::Solver for M⁻¹; null pointers fall back to the Euclidean identity.
 * Typical PDE-constrained choice: M = FEM mass matrix.
 *
 * @section parallel Parallel variant
 * LatentMirrorOptimizerParallel distributes z across MPI ranks.  The GBB
 * numerator/denominator and Armijo objective are assembled via MPI_Allreduce
 * so every rank takes identical steps.  Guarded by MFEM_USE_MPI.
 *
 * @section refs References
 *   - Bregman (1967); Nemirovsky & Yudin (1983); Beck & Teboulle (2003).
 *   - Barzilai & Borwein (1988) — two-point step-size estimate.
 *   - Kim, Lazarov, Surowiec & Keith (2025) — SiMPL / GBB strategy.
 *   - Schwedes et al. (2017); Petra et al. (2023) — Riesz-map gradient.
 */

#pragma once

#include <mfem.hpp>
#include <cmath>
#include <algorithm>
#include <functional>
#include <limits>
#include <vector>

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

namespace mfem_lmg {

// ============================================================
//  Bound-type tag
// ============================================================
enum class BoundType : uint8_t {
    Unbounded = 0,  ///< lᵢ = -∞, uᵢ = +∞
    LowerOnly = 1,  ///< lᵢ > -∞, uᵢ = +∞
    UpperOnly = 2,  ///< lᵢ = -∞, uᵢ < +∞
    TwoSided  = 3   ///< lᵢ > -∞, uᵢ < +∞
};

namespace detail {
/** Numerically stable sigmoid (eq. 42). */
MFEM_HOST_DEVICE inline mfem::real_t Sigmoid(mfem::real_t z)
{
    if (z >= mfem::real_t(0))
        return mfem::real_t(1) / (mfem::real_t(1) + std::exp(-z));
    const mfem::real_t ez = std::exp(z);
    return ez / (mfem::real_t(1) + ez);
}
} // namespace detail


// ============================================================
//           Primal ↔ Latent helpers (free functions)
// ============================================================

/**
 * @brief Classify each variable into one of the four BoundType categories.
 *
 * Use ±std::numeric_limits<real_t>::infinity() to denote absent bounds.
 */
void ClassifyBounds(const mfem::Vector& lo,
                    const mfem::Vector& hi,
                    std::vector<BoundType>& types);

/**
 * @brief λ → z  (eq. 9).  λ must be strictly inside K.
 *
 *   Unbounded : z = λ
 *   LowerOnly : z = log(λ − l)
 *   UpperOnly : z = −log(u − λ)
 *   TwoSided  : z = log((λ−l)/(u−λ))
 */
void PrimalToLatent(const mfem::Vector& lam,
                    const mfem::Vector& lo,
                    const mfem::Vector& hi,
                    const std::vector<BoundType>& types,
                    mfem::Vector& z);

/**
 * @brief z → λ = T(z) ∈ int(K)  (eqs. 10–12).
 *
 *   Unbounded : λ = z
 *   LowerOnly : λ = l + exp(z)
 *   UpperOnly : λ = u − exp(−z)
 *   TwoSided  : λ = l + (u−l) σ(z)
 *
 * Strict feasibility holds for every finite z.
 */
void LatentToPrimal(const mfem::Vector& z,
                    const mfem::Vector& lo,
                    const mfem::Vector& hi,
                    const std::vector<BoundType>& types,
                    mfem::Vector& lam);

/**
 * @brief Diagonal of JT(z) = dλ/dz (eqs. 43–44 support).
 *
 *   Unbounded : 1
 *   LowerOnly : exp(z)
 *   UpperOnly : exp(−z)
 *   TwoSided  : (u−l) σ(z)(1−σ(z))
 *
 * All entries are strictly positive.
 */
void LatentJacobianDiag(const mfem::Vector& z,
                        const mfem::Vector& lo,
                        const mfem::Vector& hi,
                        const std::vector<BoundType>& types,
                        mfem::Vector& diag);

/**
 * @brief Default strictly-feasible primal initialization (eq. 35).
 *
 *   Unbounded : λ = 0
 *   LowerOnly : λ = l + 1
 *   UpperOnly : λ = u − 1
 *   TwoSided  : λ = (l + u) / 2
 */
void DefaultPrimalInit(const mfem::Vector& lo,
                       const mfem::Vector& hi,
                       const std::vector<BoundType>& types,
                       mfem::Vector& lam);


// ============================================================
/**
 * @class LatentMirrorOptimizer
 * @brief Serial latent-variable mirror-gradient optimizer.
 *
 * The user holds the latent vector z.  Each call to Update() advances z
 * by one GBB+Armijo mirror-gradient step.  The primal variable λ = T(z)
 * is read on demand by calling LatentToPrimal().
 *
 * ### Algorithm per iteration (§8 of the paper)
 *  1. gᵏ = M⁻¹ dᵏ  (M-gradient, eq. 3 / 38).
 *  2. Trial step αₜ from GBB estimate (eq. 18) or eq. (22) on first step.
 *  3. Backtracking Armijo line search (eqs. 23–26):
 *       repeat: z⁺ = zᵏ − α gᵏ;  λ⁺ = T(z⁺)
 *               check Φ(λ⁺) ≤ Φ(λᵏ) + c₁ (dᵏ)ᵀ(λ⁺−λᵏ)
 *               if not: α ← β α
 *  4. Accept: zᵏ⁺¹ = z⁺.
 *
 * The Armijo objective evaluation is provided by the caller as a callback
 * (type EvalPhi) so the optimizer remains objective-agnostic.
 *
 * ### Typical usage
 * @code
 *   std::vector<BoundType> types;
 *   ClassifyBounds(lo, hi, types);
 *
 *   mfem::Vector z(n);
 *   PrimalToLatent(lam0, lo, hi, types, z);   // initialize z from lam0
 *
 *   LatentMirrorOptimizer opt(z, lo, hi);
 *
 *   mfem::Vector lam(n), d(n);
 *   for (int k = 0; k < max_iter; ++k) {
 *       LatentToPrimal(z, lo, hi, types, lam);
 *       double phi = EvalObjective(lam);
 *       EvalGradient(lam, d);
 *
 *       opt.Update(z, d, real_t(phi),
 *           [&](const mfem::Vector& z_trial, mfem::real_t& phi_out) {
 *               mfem::Vector lt(n);
 *               LatentToPrimal(z_trial, lo, hi, types, lt);
 *               phi_out = real_t(EvalObjective(lt));
 *           });
 *       if (opt.StationarityResidual() < tol) break;
 *   }
 * @endcode
 */
// ============================================================
class LatentMirrorOptimizer {
public:

    /**
     * @brief Callback type for Armijo objective evaluation.
     *
     * Called inside the backtracking loop with the trial latent vector.
     * The implementation must set phi_out = Φ(T(z_trial)).
     *
     * @param z_trial  Trial latent vector (do not modify).
     * @param phi_out  Output: objective value at the trial point.
     */
    using EvalPhi = std::function<void(const mfem::Vector& z_trial,
                                       mfem::real_t&       phi_out)>;

    // ── Constructors ──────────────────────────────────────────────────────

    /**
     * @brief Construct with identity M.
     *
     * @param z0  Initial latent vector (user-owned; copied into internal
     *            state — further modifications of z0 outside this class
     *            have no effect until the user passes it to Update()).
     * @param lo  Lower bounds (±∞ for absent bound).
     * @param hi  Upper bounds.
     */
    LatentMirrorOptimizer(const mfem::Vector& z0,
                          const mfem::Vector& lo,
                          const mfem::Vector& hi);

    /**
     * @brief Construct with a user-supplied inner product M and inverse M⁻¹.
     *
     * @param z0        Initial latent vector.
     * @param lo        Lower bounds.
     * @param hi        Upper bounds.
     * @param M_op      SPD operator M (null → identity).  Used for pairings
     *                  ⟨·,·⟩_M = (·)ᵀ M (·).
     * @param M_solver  Solver for Mx = b (null → identity).  Used to compute
     *                  gᵏ = M⁻¹ dᵏ.
     *
     * The caller retains ownership of M_op and M_solver.
     */
    LatentMirrorOptimizer(const mfem::Vector& z0,
                          const mfem::Vector& lo,
                          const mfem::Vector& hi,
                          mfem::Operator* M_op,
                          mfem::Solver*   M_solver);

    ~LatentMirrorOptimizer() = default;

    // ── Configuration ─────────────────────────────────────────────────────

    /**
     * @brief Set Armijo line-search parameters.
     * @param c1     Sufficient-decrease constant ∈ (0,1).  Default 1e-4.
     * @param beta   Step-reduction factor ∈ (0,1).        Default 0.5.
     * @param max_ls Maximum backtracking steps.            Default 50.
     */
    void SetLineSearchParams(mfem::real_t c1   = mfem::real_t(1e-4),
                             mfem::real_t beta = mfem::real_t(0.5),
                             int          max_ls = 50);

    /**
     * @brief Set GBB step-size safeguards.
     * @param alpha_min  Lower clamp on accepted step.  Default 1e-12.
     * @param alpha_max  Upper clamp on accepted step.  Default 1e4.
     * @param eps_bb     Skip GBB if |numerator| or |denominator| < eps_bb.
     *                   Default 1e-14.
     */
    void SetStepSizeSafeguards(mfem::real_t alpha_min = mfem::real_t(1e-12),
                                mfem::real_t alpha_max = mfem::real_t(1e4),
                                mfem::real_t eps_bb   = mfem::real_t(1e-14));

    /**
     * @brief Clip latent variables to [−zmax, zmax] after each step.
     *
     * Prevents sigmoid saturation.  Default 40 (σ(40) ≈ 1 − 4×10⁻¹⁸).
     */
    void SetLatentClipping(mfem::real_t zmax = mfem::real_t(40));

    /** @brief Stationarity tolerance used by HasConverged().  Default 1e-6. */
    void SetStationarityTol(mfem::real_t tol = mfem::real_t(1e-6));

    // ── Outer iteration ───────────────────────────────────────────────────

    /**
     * @brief Perform one GBB + Armijo mirror-gradient step.
     *
     * @param[in,out] z       Latent variable (user-owned; updated in-place).
     * @param[in]     d       Euclidean derivative dᵏ = ∇_E Φ(T(z)).
     * @param[in]     phi_k   Objective value Φ(T(z)) at the current z.
     *                        Must match z on entry (not the updated value).
     * @param[in]     eval_phi  Callback for Armijo objective evaluation.
     *                          Called at each trial step inside the backtrack.
     *                          If null, the Armijo check is skipped and the
     *                          initial GBB trial step is accepted directly
     *                          (useful when the objective is cheap enough to
     *                          not need a line search, or for testing).
     * @return Accepted step size αᵏ.
     */
    mfem::real_t Update(mfem::Vector&      z,
                        const mfem::Vector& d,
                        mfem::real_t        phi_k,
                        EvalPhi             eval_phi = nullptr);

    // ── Convergence diagnostics ───────────────────────────────────────────

    /**
     * @brief Euclidean projected-gradient stationarity residual (eq. 33).
     *
     * Computes  ‖λ − Π_K(λ − d)‖₂ / max(1, ‖λ‖₂)  from the given
     * latent z (converted internally).  Requires the Euclidean derivative d.
     */
    mfem::real_t StationarityResidual(const mfem::Vector& z,
                                      const mfem::Vector& d) const;

    /**
     * @brief Cached residual from the last Update() call.
     * Returns −1 before the first call.
     */
    mfem::real_t StationarityResidual() const { return last_stat_res_; }

    /** @brief True if cached residual < stationarity tolerance. */
    bool HasConverged() const {
        return last_stat_res_ >= mfem::real_t(0) &&
               last_stat_res_ < stat_tol_;
    }

    /**
     * @brief Relative M-norm step size ‖λᵏ⁺¹−λᵏ‖_M / max(1,‖λᵏ‖_M) (eq. 34).
     * Returns −1 before the first call.
     */
    mfem::real_t RelativeStepSize() const { return last_rel_step_; }

    /** @brief Number of backtracking reductions in the last step. */
    int LastLineSearchSteps() const { return last_ls_steps_; }

    /** @brief Total completed Update() calls. */
    int NumIterations() const { return iter_; }

    /** @brief Bound-type classification built at construction. */
    const std::vector<BoundType>& GetBoundTypes() const { return types_; }

private:
    int n_;
    mfem::Vector lo_, hi_;
    std::vector<BoundType> types_;

    mfem::Operator* M_op_;
    mfem::Solver*   M_solver_;

    // History for GBB (stored as latent vectors).
    mfem::Vector z_prev_;    ///< zᵏ⁻¹
    mfem::Vector lam_prev_;  ///< T(zᵏ⁻¹)
    mfem::Vector g_prev_;    ///< gᵏ⁻¹

    // Work vectors (avoid repeated allocation).
    mfem::Vector g_;          ///< current M-gradient
    mfem::Vector Mg_;         ///< M gᵏ = dᵏ  (for Armijo pairing)
    mfem::Vector z_trial_;    ///< trial latent vector
    mfem::Vector lam_cur_;    ///< T(zᵏ)    — current primal
    mfem::Vector lam_trial_;  ///< T(z_trial_) — trial primal

    mfem::real_t alpha_prev_;
    mfem::real_t alpha_min_, alpha_max_, eps_bb_;
    mfem::real_t c1_, beta_;
    int          max_ls_;
    mfem::real_t zmax_;
    mfem::real_t stat_tol_;

    int          iter_;
    int          last_ls_steps_;
    mfem::real_t last_stat_res_;
    mfem::real_t last_rel_step_;

    void         ApplyM   (const mfem::Vector& x, mfem::Vector& Mx)    const;
    void         ApplyMinv(const mfem::Vector& x, mfem::Vector& Minvx) const;
    mfem::real_t InnerProductM(const mfem::Vector& x,
                                const mfem::Vector& y) const;
    mfem::real_t NormM(const mfem::Vector& x) const {
        return std::sqrt(InnerProductM(x, x));
    }
    mfem::real_t ComputeGBBStepSize(const mfem::Vector& dz,
                                     const mfem::Vector& dlam,
                                     const mfem::Vector& dg,
                                     mfem::real_t fallback) const;
    void ClipLatent(mfem::Vector& z) const;
};


// ============================================================
#ifdef MFEM_USE_MPI
/**
 * @class LatentMirrorOptimizerParallel
 * @brief MPI-parallel latent-variable mirror-gradient optimizer.
 *
 * Distributes the n latent variables z across MPI ranks.  Each rank owns a
 * contiguous local chunk.  Global scalars (GBB numerator/denominator, Armijo
 * objective) are assembled via MPI_Allreduce so every rank takes identical
 * steps.
 *
 * The Armijo objective callback must return the GLOBAL objective value
 * (same on all ranks).  The caller is responsible for any reductions needed
 * to compute that global value from local contributions.
 *
 * ### Typical parallel usage
 * @code
 *   std::vector<BoundType> types;
 *   ClassifyBounds(lo_loc, hi_loc, types);
 *
 *   mfem::Vector z_loc(n_local);
 *   PrimalToLatent(lam0_loc, lo_loc, hi_loc, types, z_loc);
 *
 *   LatentMirrorOptimizerParallel opt(MPI_COMM_WORLD, z_loc, lo_loc, hi_loc);
 *
 *   mfem::Vector lam_loc(n_local), d_loc(n_local);
 *   for (int k = 0; k < max_iter; ++k) {
 *       LatentToPrimal(z_loc, lo_loc, hi_loc, types, lam_loc);
 *       double phi = EvalGlobalObjective(lam_loc);   // MPI reduction inside
 *       EvalLocalGradient(lam_loc, d_loc);
 *
 *       opt.Update(z_loc, d_loc, real_t(phi),
 *           [&](const mfem::Vector& zt, mfem::real_t& phi_out) {
 *               mfem::Vector lt(n_local);
 *               LatentToPrimal(zt, lo_loc, hi_loc, types, lt);
 *               phi_out = real_t(EvalGlobalObjective(lt)); // all ranks call this
 *           });
 *       if (opt.StationarityResidual() < tol) break;
 *   }
 * @endcode
 *
 * @note Compiled only when MFEM_USE_MPI is defined.
 */
// ============================================================
class LatentMirrorOptimizerParallel {
public:
    /** @copydoc LatentMirrorOptimizer::EvalPhi */
    using EvalPhi = std::function<void(const mfem::Vector& z_trial_local,
                                       mfem::real_t&       phi_out)>;

    // ── Constructors ──────────────────────────────────────────────────────

    /**
     * @brief Construct with identity M.
     * @param comm      MPI communicator.
     * @param z0_local  Initial local latent chunk.
     * @param lo_local  Local lower bounds.
     * @param hi_local  Local upper bounds.
     */
    LatentMirrorOptimizerParallel(MPI_Comm            comm,
                                   const mfem::Vector& z0_local,
                                   const mfem::Vector& lo_local,
                                   const mfem::Vector& hi_local);

    /**
     * @brief Construct with user-supplied local M operator and solver.
     * @param comm           MPI communicator.
     * @param z0_local       Initial local latent chunk.
     * @param lo_local       Local lower bounds.
     * @param hi_local       Local upper bounds.
     * @param M_op_local     Local block of M (null → identity).
     * @param M_solver_local Local solver for the M-block (null → identity).
     *
     * Global inner products ⟨x,y⟩_M are assembled as
     *   Σ_rank  xlocᵀ Mlocal ylocal  via MPI_Allreduce.
     */
    LatentMirrorOptimizerParallel(MPI_Comm            comm,
                                   const mfem::Vector& z0_local,
                                   const mfem::Vector& lo_local,
                                   const mfem::Vector& hi_local,
                                   mfem::Operator*     M_op_local,
                                   mfem::Solver*       M_solver_local);

    ~LatentMirrorOptimizerParallel() = default;

    // ── Configuration ─────────────────────────────────────────────────────

    /** @copydoc LatentMirrorOptimizer::SetLineSearchParams */
    void SetLineSearchParams(mfem::real_t c1   = mfem::real_t(1e-4),
                             mfem::real_t beta = mfem::real_t(0.5),
                             int          max_ls = 50);

    /** @copydoc LatentMirrorOptimizer::SetStepSizeSafeguards */
    void SetStepSizeSafeguards(mfem::real_t alpha_min = mfem::real_t(1e-12),
                                mfem::real_t alpha_max = mfem::real_t(1e4),
                                mfem::real_t eps_bb   = mfem::real_t(1e-14));

    /** @copydoc LatentMirrorOptimizer::SetLatentClipping */
    void SetLatentClipping(mfem::real_t zmax = mfem::real_t(40));

    /** @copydoc LatentMirrorOptimizer::SetStationarityTol */
    void SetStationarityTol(mfem::real_t tol = mfem::real_t(1e-6));

    // ── Outer iteration ───────────────────────────────────────────────────

    /**
     * @brief Perform one parallel GBB + Armijo mirror-gradient step.
     *
     * @param[in,out] z_local   Local latent chunk (updated in-place).
     * @param[in]     d_local   Local Euclidean derivative ∇_E Φ at T(z_local).
     * @param[in]     phi_k     Global objective value Φ(T(z)) — same on all
     *                          ranks.
     * @param[in]     eval_phi  Callback returning the GLOBAL objective at a
     *                          trial z_local.  Called on ALL ranks.  Null
     *                          disables the Armijo check.
     * @return Accepted step size αᵏ (same on all ranks).
     */
    mfem::real_t Update(mfem::Vector&      z_local,
                        const mfem::Vector& d_local,
                        mfem::real_t        phi_k,
                        EvalPhi             eval_phi = nullptr);

    // ── Convergence diagnostics ───────────────────────────────────────────

    /**
     * @brief Global projected-gradient stationarity residual (eq. 33).
     *
     * Global 2-norms assembled via MPI_Allreduce; identical on all ranks.
     */
    mfem::real_t StationarityResidual(const mfem::Vector& z_local,
                                       const mfem::Vector& d_local) const;

    /** @copydoc LatentMirrorOptimizer::StationarityResidual() const */
    mfem::real_t StationarityResidual() const { return last_stat_res_; }

    /** @copydoc LatentMirrorOptimizer::HasConverged */
    bool HasConverged() const {
        return last_stat_res_ >= mfem::real_t(0) &&
               last_stat_res_ < stat_tol_;
    }

    /** @copydoc LatentMirrorOptimizer::RelativeStepSize */
    mfem::real_t RelativeStepSize()     const { return last_rel_step_; }

    /** @brief Number of backtracking reductions in the last step. */
    int LastLineSearchSteps()           const { return last_ls_steps_; }

    /** @brief Total completed Update() calls. */
    int NumIterations()                 const { return iter_; }

    /** @brief Bound-type classification for this rank's local chunk. */
    const std::vector<BoundType>& GetBoundTypes() const { return types_; }

private:
    MPI_Comm     comm_;
    int          n_local_;

    mfem::Vector lo_local_, hi_local_;
    std::vector<BoundType> types_;

    mfem::Operator* M_op_;
    mfem::Solver*   M_solver_;

    mfem::Vector z_prev_local_;
    mfem::Vector lam_prev_local_;
    mfem::Vector g_prev_local_;

    mfem::Vector g_local_;
    mfem::Vector Mg_local_;
    mfem::Vector z_trial_local_;
    mfem::Vector lam_cur_local_;
    mfem::Vector lam_trial_local_;

    mfem::real_t alpha_prev_;
    mfem::real_t alpha_min_, alpha_max_, eps_bb_;
    mfem::real_t c1_, beta_;
    int          max_ls_;
    mfem::real_t zmax_;
    mfem::real_t stat_tol_;

    int          iter_;
    int          last_ls_steps_;
    mfem::real_t last_stat_res_;
    mfem::real_t last_rel_step_;

    void         ApplyM       (const mfem::Vector& x, mfem::Vector& Mx)    const;
    void         ApplyMinv    (const mfem::Vector& x, mfem::Vector& Minvx) const;
    mfem::real_t InnerProductM(const mfem::Vector& x,
                                const mfem::Vector& y) const; // global via Allreduce
    mfem::real_t NormM        (const mfem::Vector& x) const {
        return std::sqrt(InnerProductM(x, x));
    }
    mfem::real_t ComputeGBBStepSize(const mfem::Vector& dz,
                                     const mfem::Vector& dlam,
                                     const mfem::Vector& dg,
                                     mfem::real_t fallback) const;
    void ClipLatent(mfem::Vector& z) const;
};
#endif // MFEM_USE_MPI

} // namespace mfem_lmg
