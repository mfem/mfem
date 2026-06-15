/**
 * @file LatentMirrorOptimizer.hpp
 * @brief Device-aware latent-variable mirror-gradient optimizer for
 *        bound-constrained minimization with a general positive-definite
 *        inner product.
 *
 * Implements the algorithm described in:
 *   "A Latent-Variable Mirror-Gradient Algorithm for Bound-Constrained
 *    Minimization with a General Positive Definite Inner Product"
 *
 * The method solves
 * @code
 *   min   Φ(λ)   s.t.  lᵢ ≤ λᵢ ≤ uᵢ,   i = 1 … n
 * @endcode
 * by working in the unconstrained latent space z ∈ ℝⁿ and recovering
 * the feasible primal via λ = T(z) ∈ int(K):
 *
 *   Unbounded  (lᵢ=-∞, uᵢ=+∞):  zᵢ = λᵢ
 *   LowerOnly  (lᵢ>-∞, uᵢ=+∞):  zᵢ = log(λᵢ − lᵢ)
 *   UpperOnly  (lᵢ=-∞, uᵢ<+∞):  zᵢ = −log(uᵢ − λᵢ)
 *   TwoSided   (lᵢ>-∞, uᵢ<+∞):  zᵢ = log((λᵢ−lᵢ)/(uᵢ−λᵢ))
 *
 * @section device Device execution
 * Device awareness follows the MMA_MFEM convention exactly:
 *   - The device flag is read from the user's latent vector z via
 *     z.UseDevice() at each Update() call and propagated to all internal
 *     vectors.  Moving z to the GPU is sufficient; no other API change is
 *     needed.
 *   - All O(n) loops use mfem::forall_switch(use_dev, n, lambda).
 *   - Device memory is accessed only through Read()/Write()/ReadWrite().
 *
 * @section branching GPU-optimal loop structure (no warp divergence)
 * A naive single kernel with switch(BoundType) causes warp divergence:
 * threads in the same warp execute different branches serializing execution.
 *
 * The solution used here is **four separate branch-free kernels**, one per
 * bound type, each launched over a compact index array for that type:
 *
 * @code
 *   // Instead of:
 *   forall(n, [=](int i){ switch(type[i]) { ... } });  // divergent
 *
 *   // We use:
 *   forall(n_unbounded, [=](int k){ int i = idx_unbounded[k]; ... });
 *   forall(n_lower,     [=](int k){ int i = idx_lower[k];     ... });
 *   forall(n_upper,     [=](int k){ int i = idx_upper[k];     ... });
 *   forall(n_twosided,  [=](int k){ int i = idx_twosided[k];  ... });
 * @endcode
 *
 * Each kernel contains only the arithmetic for its bound type: no branches,
 * no divergence.  The index arrays are built once at construction and stored
 * on device.  This strategy also allows the compiler to specialize exp/log/σ
 * calls and vectorize independently for each bound type.
 *
 * The compact index arrays are managed by the BoundPartition helper class.
 *
 * @section usage Typical usage
 * @code
 *   std::vector<BoundType> types;
 *   ClassifyBounds(lo, hi, types);
 *
 *   mfem::Vector z(n);
 *   z.UseDevice(true);                // optional: run on GPU
 *   PrimalToLatent(lam0, lo, hi, types, z);
 *
 *   LatentMirrorOptimizer opt(z, lo, hi);
 *
 *   mfem::Vector lam(n), d(n);
 *   for (int k = 0; k < max_iter; ++k) {
 *       LatentToPrimal(z, lo, hi, types, lam);
 *       real_t phi = EvalObjective(lam);
 *       EvalGradient(lam, d);
 *       opt.Update(z, d, phi,
 *           [&](const mfem::Vector& zt, real_t& phi_out) {
 *               mfem::Vector lt(n);
 *               lt.UseDevice(z.UseDevice());
 *               LatentToPrimal(zt, lo, hi, types, lt);
 *               phi_out = EvalObjective(lt);
 *           });
 *       if (opt.StationarityResidual() < tol) break;
 *   }
 * @endcode
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
/** Numerically stable sigmoid (eq. 42): no overflow for |z| up to ~700. */
MFEM_HOST_DEVICE inline mfem::real_t Sigmoid(mfem::real_t z)
{
    if (z >= mfem::real_t(0))
        return mfem::real_t(1) / (mfem::real_t(1) + std::exp(-z));
    const mfem::real_t ez = std::exp(z);
    return ez / (mfem::real_t(1) + ez);
}
} // namespace detail


// ============================================================
/**
 * @class BoundPartition
 * @brief Compact per-type index arrays for branch-free GPU kernels.
 *
 * Built once from the bound vectors at construction time.  The four index
 * arrays (idx_unbounded, idx_lower, idx_upper, idx_twosided) are stored
 * as mfem::Arrays, which live on the same device as the user's z vector.
 * Each array is densely packed so that a forall over its length visits
 * every variable of that type exactly once without branching.
 *
 * Alongside each index array the corresponding lower/upper bound values
 * for those indices are stored in compact, aligned device vectors so that
 * the GPU kernel reads contiguous memory rather than scatter-loading from
 * the full-length lo/hi vectors.
 *
 * Layout principle:
 * @code
 *   // Branch-free kernel for TwoSided variables:
 *   const int* idx = partition.IdxTwoSided();
 *   const real_t* li = partition.LoTwoSided();
 *   const real_t* ui = partition.HiTwoSided();
 *   forall_switch(use_dev, partition.NumTwoSided(), [=](int k) {
 *       const int i = idx[k];
 *       z[i] = log((lam[i]-li[k]) / (ui[k]-lam[i]));  // no branch
 *   });
 * @endcode
 */
// ============================================================
class BoundPartition {
public:
    /**
     * @brief Classify bounds and build compact index + bound arrays.
     *
     * @param lo  Full lower-bound vector (n entries; ±∞ for absent bound).
     * @param hi  Full upper-bound vector (n entries).
     * @param ref Reference vector whose device flag is inherited.
     */
    BoundPartition(const mfem::Vector& lo,
                   const mfem::Vector& hi,
                   const mfem::Vector& ref);

    // ── Counts ────────────────────────────────────────────────────────────
    int NumUnbounded() const { return n_unbounded_; }
    int NumLower()     const { return n_lower_;     }
    int NumUpper()     const { return n_upper_;     }
    int NumTwoSided()  const { return n_twosided_;  }
    int Total()        const { return n_unbounded_ + n_lower_ +
                                      n_upper_ + n_twosided_; }

    // ── Raw device pointers (valid inside forall kernels) ─────────────────

    /// Global index of the k-th Unbounded variable.
    const int* IdxUnbounded() const { return idx_unbounded_.Read(); }
    /// Global index of the k-th LowerOnly variable.
    const int* IdxLower()     const { return idx_lower_.Read();     }
    /// Global index of the k-th UpperOnly variable.
    const int* IdxUpper()     const { return idx_upper_.Read();     }
    /// Global index of the k-th TwoSided variable.
    const int* IdxTwoSided()  const { return idx_twosided_.Read();  }

    /// Compact lower bounds for LowerOnly variables (length NumLower()).
    const mfem::real_t* LoLower()    const { return lo_lower_.Read();    }
    /// Compact upper bounds for UpperOnly variables (length NumUpper()).
    const mfem::real_t* HiUpper()    const { return hi_upper_.Read();    }
    /// Compact lower bounds for TwoSided variables (length NumTwoSided()).
    const mfem::real_t* LoTwoSided() const { return lo_twosided_.Read(); }
    /// Compact upper bounds for TwoSided variables (length NumTwoSided()).
    const mfem::real_t* HiTwoSided() const { return hi_twosided_.Read(); }

    /** @brief Build the std::vector<BoundType> classification for host use. */
    void GetTypes(std::vector<BoundType>& types) const;

private:
    int n_unbounded_, n_lower_, n_upper_, n_twosided_;

    mfem::Array<int> idx_unbounded_, idx_lower_, idx_upper_, idx_twosided_;

    // Compact bound values aligned with the index arrays.
    mfem::Vector lo_lower_;     ///< lᵢ for LowerOnly  (length n_lower_)
    mfem::Vector hi_upper_;     ///< uᵢ for UpperOnly  (length n_upper_)
    mfem::Vector lo_twosided_;  ///< lᵢ for TwoSided   (length n_twosided_)
    mfem::Vector hi_twosided_;  ///< uᵢ for TwoSided   (length n_twosided_)
};


// ============================================================
//           Primal ↔ Latent free functions
//           (device-aware; use the BoundPartition overloads for GPU)
// ============================================================

/**
 * @brief Classify each variable into one of the four BoundType categories.
 * ±std::numeric_limits<real_t>::infinity() denotes absent bounds.
 */
void ClassifyBounds(const mfem::Vector& lo,
                    const mfem::Vector& hi,
                    std::vector<BoundType>& types);

/**
 * @brief λ → z  (eq. 9).  Device-aware, branch-free, boundary-safe.
 *
 * Launches four separate forall kernels, one per bound type.  No branching
 * inside any kernel.  The device flag is inherited from lam.
 *
 * @par Boundary safety
 * The argument of every log() is clamped to a minimum positive value before
 * evaluation so that a primal point at or beyond a bound produces a
 * large-but-finite latent value rather than -Inf or NaN.  The minimum gap is
 *
 *   δ = max(kPrimalMinGap, kPrimalRelGap × (u − l))
 *
 * where kPrimalMinGap = exp(-kLatentClipDefault) ≈ 4×10⁻¹⁸ (double) and
 * kPrimalRelGap = machine epsilon.  This keeps z within the clipping range
 * [-kLatentClipDefault, +kLatentClipDefault] even when λ is exactly at a
 * bound, consistent with the latent clipping applied after each step.
 *
 * The clamp is a branchless fmax/fmin with no GPU divergence.
 *
 * @note Use BoundSafetyCheck() to count bound violations on the host before
 *       calling this function.
 */
void PrimalToLatent(const mfem::Vector& lam,
                    const BoundPartition& part,
                    mfem::Vector& z);

/**
 * @brief Host-side diagnostic: count primal entries at or beyond their bounds.
 *
 * Reads lam on the host and counts entries where
 *   λᵢ ≤ lᵢ + tol  (LowerOnly or TwoSided lower side)
 *   λᵢ ≥ uᵢ − tol  (UpperOnly or TwoSided upper side)
 *
 * Returns the total violation count (0 = all strictly feasible).
 * Does not modify any vector.
 *
 * @param lam  Primal variable to check (transferred to host if on device).
 * @param part BoundPartition built from the same lo/hi.
 * @param lo   Full lower-bound vector.
 * @param hi   Full upper-bound vector.
 * @param tol  Proximity threshold (default 0 = exact boundary only).
 */
int BoundSafetyCheck(const mfem::Vector& lam,
                     const BoundPartition& part,
                     const mfem::Vector& lo,
                     const mfem::Vector& hi,
                     mfem::real_t tol = mfem::real_t(0));

/**
 * @brief z → λ = T(z) ∈ int(K)  (eqs. 10–12).  Device-aware, branch-free.
 *
 * Launches four separate forall kernels.  Strict feasibility holds for
 * every finite z.
 */
void LatentToPrimal(const mfem::Vector& z,
                    const BoundPartition& part,
                    mfem::Vector& lam);

/**
 * @brief Diagonal of JT(z) = dλ/dz.  Device-aware, branch-free.
 *
 * All entries are strictly positive.
 */
void LatentJacobianDiag(const mfem::Vector& z,
                        const BoundPartition& part,
                        mfem::Vector& diag);

/**
 * @brief Default strictly-feasible primal initialization (eq. 35).
 * Device-aware, branch-free.
 */
void DefaultPrimalInit(const BoundPartition& part,
                       const mfem::Vector& lo,
                       const mfem::Vector& hi,
                       mfem::Vector& lam);

// ── Convenience overloads (host-only, builds a temporary BoundPartition) ──

/**
 * @brief Classify bounds (host-only helper; result stored in types vector).
 */
void PrimalToLatent(const mfem::Vector& lam,
                    const mfem::Vector& lo,
                    const mfem::Vector& hi,
                    const std::vector<BoundType>& types,
                    mfem::Vector& z);

void LatentToPrimal(const mfem::Vector& z,
                    const mfem::Vector& lo,
                    const mfem::Vector& hi,
                    const std::vector<BoundType>& types,
                    mfem::Vector& lam);

void DefaultPrimalInit(const mfem::Vector& lo,
                       const mfem::Vector& hi,
                       const std::vector<BoundType>& types,
                       mfem::Vector& lam);


// ============================================================
/**
 * @class LatentMirrorOptimizer
 * @brief Serial device-aware latent-variable mirror-gradient optimizer.
 *
 * The device is determined by the user's latent vector z passed to Update():
 * @code
 *   z.UseDevice(true);   // all O(n) kernels run on GPU
 *   z.UseDevice(false);  // all O(n) kernels run on CPU (default)
 * @endcode
 *
 * All internal vectors inherit the device flag of z at the first Update()
 * call and stay on the same device for the lifetime of the optimizer.
 *
 * ### Algorithm per iteration (§8 of the paper)
 *  1. gᵏ = M⁻¹ dᵏ.
 *  2. αₜ from GBB (eq. 18) or eq. (22) on first step.
 *  3. Backtracking Armijo line search (eqs. 23–26):
 *       z⁺ = zᵏ − α gᵏ;  λ⁺ = T(z⁺)
 *       check  Φ(λ⁺) ≤ Φ(λᵏ) + c₁ (dᵏ)ᵀ(λ⁺−λᵏ)
 *  4. Accept zᵏ⁺¹ = z⁺.
 */
// ============================================================
class LatentMirrorOptimizer {
public:
    /**
     * Callback for Armijo objective evaluation.
     * Called with z_trial (on the same device as z); must set phi_out to
     * the GLOBAL objective Φ(T(z_trial)).
     */
    using EvalPhi = std::function<void(const mfem::Vector& z_trial,
                                       mfem::real_t& phi_out)>;

    // ── Constructors ──────────────────────────────────────────────────────

    /** @brief Construct with identity M.
     *  @param z0  Initial latent vector.  Its device flag is inherited.
     *  @param lo  Lower bounds (use ±∞ for absent).
     *  @param hi  Upper bounds.
     */
    LatentMirrorOptimizer(const mfem::Vector& z0,
                          const mfem::Vector& lo,
                          const mfem::Vector& hi);

    /** @brief Construct with user-supplied inner product M and inverse M⁻¹.
     *  @param M_op     SPD operator M (null → identity).
     *  @param M_solver Solver for Mx=b  (null → identity).
     */
    LatentMirrorOptimizer(const mfem::Vector& z0,
                          const mfem::Vector& lo,
                          const mfem::Vector& hi,
                          mfem::Operator* M_op,
                          mfem::Solver*   M_solver);

    ~LatentMirrorOptimizer() = default;

    // ── Configuration ─────────────────────────────────────────────────────

    /** @brief Armijo line-search parameters.
     *  @param c1     Sufficient-decrease constant ∈ (0,1).  Default 1e-4.
     *  @param beta   Step-reduction factor ∈ (0,1).        Default 0.5.
     *  @param max_ls Maximum backtracking steps.            Default 50.
     */
    void SetLineSearchParams(mfem::real_t c1   = mfem::real_t(1e-4),
                             mfem::real_t beta = mfem::real_t(0.5),
                             int          max_ls = 50);

    /** @brief GBB safeguards.
     *  @param alpha_min  Lower clamp.   Default 1e-12.
     *  @param alpha_max  Upper clamp.   Default 1e4.
     *  @param eps_bb     Threshold below which GBB is skipped. Default 1e-14.
     */
    void SetStepSizeSafeguards(mfem::real_t alpha_min = mfem::real_t(1e-12),
                                mfem::real_t alpha_max = mfem::real_t(1e4),
                                mfem::real_t eps_bb   = mfem::real_t(1e-14));

    /** @brief Clip latent variables to [−zmax, zmax] after each step.
     *  Default 40.
     */
    void SetLatentClipping(mfem::real_t zmax = mfem::real_t(40));

    /** @brief Stationarity tolerance for HasConverged().  Default 1e-6. */
    void SetStationarityTol(mfem::real_t tol = mfem::real_t(1e-6));

    // ── Outer iteration ───────────────────────────────────────────────────

    /**
     * @brief One GBB + Armijo mirror-gradient step.
     *
     * @param[in,out] z        Latent variable (updated in-place; device
     *                         determined by z.UseDevice()).
     * @param[in]     d        Euclidean derivative dᵏ = ∇_E Φ(T(z)).
     * @param[in]     phi_k    Objective Φ(T(z)) at current z.
     * @param[in]     eval_phi Armijo callback (null → skip check).
     * @return Accepted step size αᵏ.
     */
    mfem::real_t Update(mfem::Vector&       z,
                        const mfem::Vector& d,
                        mfem::real_t        phi_k,
                        EvalPhi             eval_phi = nullptr);

    // ── Convergence ───────────────────────────────────────────────────────

    /** Euclidean projected-gradient residual (eq. 33) from z and d. */
    mfem::real_t StationarityResidual(const mfem::Vector& z,
                                      const mfem::Vector& d) const;

    /** Cached residual from the last Update(). Returns −1 before first call. */
    mfem::real_t StationarityResidual() const { return last_stat_res_; }

    bool HasConverged() const {
        return last_stat_res_ >= mfem::real_t(0) &&
               last_stat_res_ < stat_tol_;
    }

    /** Relative M-norm step ‖λᵏ⁺¹−λᵏ‖_M / max(1,‖λᵏ‖_M) (eq. 34). */
    mfem::real_t RelativeStepSize()    const { return last_rel_step_;  }
    int          LastLineSearchSteps() const { return last_ls_steps_;  }
    int          NumIterations()       const { return iter_;           }

    /** BoundPartition built at construction (useful for LatentToPrimal). */
    const BoundPartition& GetPartition() const { return part_; }

private:
    int            n_;
    BoundPartition part_;
    mfem::Vector   lo_full_, hi_full_;   ///< full-length lo/hi on device

    mfem::Operator* M_op_;
    mfem::Solver*   M_solver_;

    // GBB history (live on same device as z).
    mfem::Vector z_prev_, lam_prev_, g_prev_;

    // Work vectors.
    mfem::Vector g_, Mg_, z_trial_, lam_cur_, lam_trial_;

    bool initialized_;   ///< false until first Update(); deferred device init

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

    /** Resize and set UseDevice on all internal vectors to match ref. */
    void InitDeviceVectors(const mfem::Vector& ref);

    void         ApplyM   (const mfem::Vector& x, mfem::Vector& Mx)    const;
    void         ApplyMinv(const mfem::Vector& x, mfem::Vector& Minvx) const;
    mfem::real_t InnerProductM(const mfem::Vector& x,
                                const mfem::Vector& y) const;
    mfem::real_t NormM(const mfem::Vector& x) const {
        return std::sqrt(std::max(mfem::real_t(0), InnerProductM(x, x)));
    }
    mfem::real_t ComputeGBBStepSize(const mfem::Vector& dz,
                                     const mfem::Vector& dlam,
                                     const mfem::Vector& dg,
                                     mfem::real_t fallback) const;
    void ClipLatent(mfem::Vector& z) const;

    /** Compute projected-gradient residual on device; returns scalar. */
    mfem::real_t ProjectedGradResidual(const mfem::Vector& lam,
                                        const mfem::Vector& d) const;
};


// ============================================================
#ifdef MFEM_USE_MPI
/**
 * @class LatentMirrorOptimizerParallel
 * @brief MPI-parallel device-aware latent-variable mirror-gradient optimizer.
 *
 * Distributes z across MPI ranks.  Each rank owns a local chunk.  Global
 * inner products (GBB, Armijo pairing) are assembled via MPI_Allreduce.
 * The device flag is read from the local z passed to Update().
 *
 * The Armijo eval_phi callback must return the GLOBAL objective value
 * (same on all ranks; the callback is responsible for any MPI reduction).
 */
// ============================================================
class LatentMirrorOptimizerParallel {
public:
    using EvalPhi = std::function<void(const mfem::Vector& z_trial_local,
                                       mfem::real_t& phi_out)>;

    // ── Constructors ──────────────────────────────────────────────────────

    LatentMirrorOptimizerParallel(MPI_Comm            comm,
                                   const mfem::Vector& z0_local,
                                   const mfem::Vector& lo_local,
                                   const mfem::Vector& hi_local);

    LatentMirrorOptimizerParallel(MPI_Comm            comm,
                                   const mfem::Vector& z0_local,
                                   const mfem::Vector& lo_local,
                                   const mfem::Vector& hi_local,
                                   mfem::Operator*     M_op_local,
                                   mfem::Solver*       M_solver_local);

    ~LatentMirrorOptimizerParallel() = default;

    // ── Configuration ─────────────────────────────────────────────────────

    void SetLineSearchParams(mfem::real_t c1   = mfem::real_t(1e-4),
                             mfem::real_t beta = mfem::real_t(0.5),
                             int          max_ls = 50);
    void SetStepSizeSafeguards(mfem::real_t alpha_min = mfem::real_t(1e-12),
                                mfem::real_t alpha_max = mfem::real_t(1e4),
                                mfem::real_t eps_bb   = mfem::real_t(1e-14));
    void SetLatentClipping(mfem::real_t zmax = mfem::real_t(40));
    void SetStationarityTol(mfem::real_t tol = mfem::real_t(1e-6));

    // ── Outer iteration ───────────────────────────────────────────────────

    /**
     * @param[in,out] z_local   Local latent chunk (device flag inherited).
     * @param[in]     d_local   Local Euclidean derivative.
     * @param[in]     phi_k     Global objective (same on all ranks).
     * @param[in]     eval_phi  Callback returning global Φ at a trial z_local.
     *                          Called on ALL ranks.  Null disables Armijo.
     */
    mfem::real_t Update(mfem::Vector&       z_local,
                        const mfem::Vector& d_local,
                        mfem::real_t        phi_k,
                        EvalPhi             eval_phi = nullptr);

    // ── Convergence ───────────────────────────────────────────────────────

    mfem::real_t StationarityResidual(const mfem::Vector& z_local,
                                       const mfem::Vector& d_local) const;
    mfem::real_t StationarityResidual() const { return last_stat_res_; }
    bool HasConverged() const {
        return last_stat_res_ >= mfem::real_t(0) &&
               last_stat_res_ < stat_tol_;
    }
    mfem::real_t RelativeStepSize()    const { return last_rel_step_;  }
    int          LastLineSearchSteps() const { return last_ls_steps_;  }
    int          NumIterations()       const { return iter_;           }

    const BoundPartition& GetPartition() const { return part_; }

private:
    MPI_Comm       comm_;
    int            n_local_;
    BoundPartition part_;
    mfem::Vector   lo_full_local_, hi_full_local_;

    mfem::Operator* M_op_;
    mfem::Solver*   M_solver_;

    mfem::Vector z_prev_local_, lam_prev_local_, g_prev_local_;
    mfem::Vector g_local_, Mg_local_, z_trial_local_;
    mfem::Vector lam_cur_local_, lam_trial_local_;

    bool initialized_;

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

    void InitDeviceVectors(const mfem::Vector& ref);

    void         ApplyM   (const mfem::Vector& x, mfem::Vector& Mx)    const;
    void         ApplyMinv(const mfem::Vector& x, mfem::Vector& Minvx) const;
    mfem::real_t InnerProductM(const mfem::Vector& x,
                                const mfem::Vector& y) const; // global via Allreduce
    mfem::real_t NormM(const mfem::Vector& x) const {
        return std::sqrt(std::max(mfem::real_t(0), InnerProductM(x, x)));
    }
    mfem::real_t ComputeGBBStepSize(const mfem::Vector& dz,
                                     const mfem::Vector& dlam,
                                     const mfem::Vector& dg,
                                     mfem::real_t fallback) const;
    void ClipLatent(mfem::Vector& z) const;

    mfem::real_t ProjectedGradResidualGlobal(const mfem::Vector& lam_local,
                                              const mfem::Vector& d_local) const;
};
#endif // MFEM_USE_MPI

} // namespace mfem_lmg
