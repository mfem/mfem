/**
 * @file LatentMirrorOptimizer.hpp
 * @brief Device-aware latent-variable mirror-gradient optimizer for
 *        bound-constrained minimization with a general positive-definite
 *        inner product.
 *
 * @section problem Problem statement
 * Solves the bound-constrained minimization problem
 * @code
 *   min   Φ(λ)   s.t.  lᵢ ≤ λᵢ ≤ uᵢ,   i = 1 … n
 * @endcode
 * where either bound may be ±∞.  The method works entirely in an
 * unconstrained latent space z ∈ ℝⁿ and recovers the feasible primal
 * via the invertible Legendre maps T : ℝⁿ → int(K)  (eqs. 10–12):
 *
 *   Unbounded  (lᵢ=-∞, uᵢ=+∞):  zᵢ = λᵢ,                λᵢ = zᵢ
 *   LowerOnly  (lᵢ>-∞, uᵢ=+∞):  zᵢ = log(λᵢ−lᵢ),       λᵢ = lᵢ + exp(zᵢ)
 *   UpperOnly  (lᵢ=-∞, uᵢ<+∞):  zᵢ = −log(uᵢ−λᵢ),      λᵢ = uᵢ − exp(−zᵢ)
 *   TwoSided   (lᵢ>-∞, uᵢ<+∞):  zᵢ = log((λᵢ−lᵢ)/(uᵢ−λᵢ)),
 *                                  λᵢ = lᵢ + (uᵢ−lᵢ) σ(zᵢ)
 *
 * where σ(z) = 1/(1+exp(−z)) is the numerically-stable sigmoid (eq. 42).
 * The maps guarantee strict feasibility: T(z) ∈ int(K) for every finite z.
 *
 * @section algorithm Algorithm (§8 of the paper)
 * At each iteration:
 *  1. Compute the M-gradient  gᵏ = M⁻¹ dᵏ  (eq. 3), where dᵏ = ∇_E Φ(λᵏ)
 *     is the Euclidean derivative supplied by the caller.
 *  2. Choose a trial step size αₜ via the generalized Barzilai–Borwein (GBB)
 *     estimate (eq. 18) or, on the first step, via eq. (22).
 *  3. Run a backtracking Armijo line search (eqs. 23–26) in latent space:
 *       z⁺ = zᵏ − α gᵏ;   λ⁺ = T(z⁺)
 *       accept if Φ(λ⁺) ≤ Φ(λᵏ) + c₁ (dᵏ)ᵀ(λ⁺ − λᵏ)
 *  4. Accept:  zᵏ⁺¹ = z⁺,  λᵏ⁺¹ = λ⁺.
 *
 * Convergence is measured by the Euclidean projected-gradient stationarity
 * residual (eq. 33) and the relative M-norm step size (eq. 34).
 *
 * References:
 *   - Bregman (1967) — Bregman divergence.
 *   - Nemirovsky & Yudin (1983); Beck & Teboulle (2003) — mirror descent.
 *   - Barzilai & Borwein (1988) — two-point step-size estimate.
 *   - Kim, Lazarov, Surowiec & Keith (2025, SiMPL) — GBB step (eq. 18),
 *     geometric-mean guess (eq. 21), sigmoidal latent map.
 *   - Schwedes et al. (2017); Petra et al. (2023) — Riesz-map / mass-matrix
 *     gradient for mesh-independent PDE-constrained optimization.
 *
 * @section inner_product Inner product and M-gradient
 * The caller supplies an optional symmetric positive-definite operator M
 * (mfem::Operator) and its inverse M⁻¹ (mfem::Solver).  The M-gradient is
 * gᵏ = M⁻¹ dᵏ, and inner products ⟨x, y⟩_M = xᵀ M y are used in the GBB
 * numerator/denominator and the stationarity residual.
 * When both are null, M = I (Euclidean geometry).
 * A typical choice in mesh-based optimization is M = mass matrix, which
 * removes mesh-size dependence from the gradient direction.
 *
 * @section device Device execution
 * Device awareness follows the MMA_MFEM.cpp convention:
 *  - The device flag is read from the user's latent vector z at each
 *    Update() call.  Setting z.UseDevice(true) is sufficient to run on GPU.
 *  - All O(n) loops use mfem::forall_switch(use_dev, n, lambda).
 *  - Device memory is accessed only through Read() / Write() / ReadWrite().
 *  - Scalar reductions (InnerProduct, Normlinf, Sum) dispatch to cuBLAS/
 *    rocBLAS automatically on GPU.
 *  - Internal work vectors are allocated lazily on the first Update() call
 *    so the device flag is always inherited from the user's z.
 *
 * @section branching GPU-optimal loop structure: no warp divergence
 * A naive single kernel with switch(BoundType[i]) causes warp divergence:
 * threads in the same CUDA warp execute different branches, serializing
 * execution and halving (or quartering) throughput for four bound types.
 *
 * The solution used here is four separate branch-free kernels, one per bound
 * type, each launched over a compact index array built at construction:
 * @code
 *   // divergent (do NOT use):
 *   forall(n, [=](int i) { switch(type[i]) { case LowerOnly: ...; ... } });
 *
 *   // branch-free (used here):
 *   forall(n_lower,    [=](int k){ int i=idx_lower[k];    z[i]=log(lam[i]-lo[k]); });
 *   forall(n_upper,    [=](int k){ int i=idx_upper[k];    z[i]=-log(hi[k]-lam[i]); });
 *   forall(n_twosided, [=](int k){ int i=idx_twosided[k]; z[i]=log(...); });
 *   forall(n_unbounded,[=](int k){ int i=idx_unbounded[k];z[i]=lam[i]; });
 * @endcode
 * Each kernel contains exactly one arithmetic form: no branches, full
 * vectorization, and contiguous coalesced memory access via the compact
 * bound-value arrays stored alongside each index array.
 * Index and bound arrays are built once on the host and uploaded to device
 * in the BoundPartition constructor.
 *
 * @section user_interface User interface
 * The caller holds the latent vector z (not the primal λ).  The primal is
 * read on demand via LatentToPrimal().
 * @code
 *   // --- serial, identity M ---
 *   std::vector<BoundType> types;
 *   ClassifyBounds(lo, hi, types);
 *
 *   mfem::Vector z(n);
 *   z.UseDevice(true);                        // optional: GPU
 *   PrimalToLatent(lam0, lo, hi, types, z);   // initialize from primal
 *
 *   LatentMirrorOptimizer opt(z, lo, hi);
 *
 *   mfem::Vector lam(n), d(n);
 *   for (int k = 0; k < max_iter; ++k) {
 *       LatentToPrimal(z, lo, hi, types, lam);
 *       real_t phi = EvalObjective(lam);
 *       EvalGradient(lam, d);                 // d = ∇_E Φ(λ)
 *       opt.Update(z, d, phi,
 *           [&](const mfem::Vector& z_trial, real_t& phi_out) {
 *               mfem::Vector lt(n);
 *               lt.UseDevice(z.UseDevice());
 *               LatentToPrimal(z_trial, lo, hi, types, lt);
 *               phi_out = EvalObjective(lt);  // Armijo objective
 *           });
 *       if (opt.StationarityResidual() < tol) break;
 *   }
 * @endcode
 *
 * @section parallel Parallel variant
 * LatentMirrorOptimizerParallel distributes z across MPI ranks.  Each rank
 * owns a contiguous local chunk.  GBB inner products and the Armijo pairing
 * are assembled via MPI_Allreduce so all ranks take identical steps.
 * The eval_phi callback must return the GLOBAL objective (caller responsible
 * for any needed MPI reduction).  Guarded by MFEM_USE_MPI.
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
/// @brief Bound classification tag — one per design variable.
///
/// Assigned by ClassifyBounds() and used by BoundPartition to route each
/// variable to the correct latent map.  The integer values must not be
/// changed; they are used as array indices in some diagnostic code.
// ============================================================
enum class BoundType : uint8_t {
    Unbounded = 0,  ///< lᵢ = −∞, uᵢ = +∞  →  z = λ  (identity)
    LowerOnly = 1,  ///< lᵢ > −∞, uᵢ = +∞  →  z = log(λ − l)
    UpperOnly = 2,  ///< lᵢ = −∞, uᵢ < +∞  →  z = −log(u − λ)
    TwoSided  = 3   ///< lᵢ > −∞, uᵢ < +∞  →  z = log((λ−l)/(u−λ))
};

/// @brief Internal helpers (not part of the public API).
namespace detail {
/**
 * @brief Numerically stable sigmoid σ(z) = 1/(1 + exp(−z))  (eq. 42).
 *
 * Uses the log-sum-exp trick to avoid floating-point overflow for large |z|:
 *   z ≥ 0:  σ = 1/(1 + exp(−z))    (denominator ≥ 1, never overflows)
 *   z < 0:  σ = exp(z)/(1 + exp(z)) (numerator ≤ 1, never overflows)
 *
 * Declared MFEM_HOST_DEVICE so it can be called from forall GPU kernels.
 */
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
 * @brief Compact per-type index arrays enabling branch-free GPU kernels.
 *
 * @par Motivation
 * A single kernel iterating over all variables and branching on
 * BoundType[i] causes warp divergence on GPU: threads in the same 32-wide
 * warp take different paths, serializing execution.  BoundPartition avoids
 * this by partitioning the n variables into four disjoint sets at
 * construction time and storing compact index arrays for each set.  Each
 * per-type kernel then runs without any branch, allowing full vectorization
 * and coalesced memory access.
 *
 * @par Layout
 * For each bound type T ∈ {Unbounded, LowerOnly, UpperOnly, TwoSided}:
 *  - `idx_T[k]`  = global index i of the k-th variable of type T
 *    (k = 0 … numT−1).
 *  - For bounded types, compact bound-value vectors store lᵢ and/or uᵢ
 *    aligned with the index arrays so the kernel reads contiguous memory.
 *
 * @par Device storage
 * All arrays are uploaded to the device specified by the reference vector
 * `ref` passed to the constructor.  The raw device pointers returned by
 * `IdxXxx()`, `LoXxx()`, `HiXxx()` are valid inside forall kernels.
 *
 * @par Typical usage inside a kernel
 * @code
 *   // Branch-free TwoSided latent update:
 *   const int*    idx = part.IdxTwoSided();
 *   const real_t* li  = part.LoTwoSided();
 *   const real_t* ui  = part.HiTwoSided();
 *   forall_switch(use_dev, part.NumTwoSided(), [=](int k) MFEM_HOST_DEVICE {
 *       const int i = idx[k];
 *       z[i] = std::log((lam[i] - li[k]) / (ui[k] - lam[i]));
 *   });
 * @endcode
 *
 * @note BoundPartition is built once at optimizer construction time.
 *       It is immutable after construction.
 */
// ============================================================
class BoundPartition {
public:
    /**
     * @brief Classify bounds and build compact index + bound arrays.
     *
     * Reads `lo` and `hi` on the host, classifies each entry, packs the
     * per-type index and bound arrays, then uploads them to the device
     * specified by `ref.UseDevice()`.
     *
     * @param lo   Full lower-bound vector (length n).
     *             Use −std::numeric_limits<real_t>::infinity() for absent lower bound.
     * @param hi   Full upper-bound vector (length n).
     *             Use +std::numeric_limits<real_t>::infinity() for absent upper bound.
     * @param ref  Reference vector whose device flag (UseDevice()) is inherited
     *             by all index and bound arrays.
     *
     * @pre lo.Size() == hi.Size() > 0.
     * @pre For every i with a finite bound: lo[i] < hi[i].
     */
    BoundPartition(const mfem::Vector& lo,
                   const mfem::Vector& hi,
                   const mfem::Vector& ref);

    /// @name Variable counts per bound type
    /// @{
    int NumUnbounded() const { return n_unbounded_; } ///< Number of Unbounded variables.
    int NumLower()     const { return n_lower_;     } ///< Number of LowerOnly variables.
    int NumUpper()     const { return n_upper_;     } ///< Number of UpperOnly variables.
    int NumTwoSided()  const { return n_twosided_;  } ///< Number of TwoSided variables.
    /// Total variable count (= sum of the four counts = n).
    int Total()        const { return n_unbounded_ + n_lower_ + n_upper_ + n_twosided_; }
    /// @}

    /// @name Raw device pointers to index arrays
    /// Suitable for capture by forall device lambdas.  Valid until the
    /// BoundPartition object is destroyed.
    /// @{

    /// Device pointer to the global indices of Unbounded variables (length NumUnbounded()).
    const int* IdxUnbounded() const { return idx_unbounded_.Read(); }
    /// Device pointer to the global indices of LowerOnly variables (length NumLower()).
    const int* IdxLower()     const { return idx_lower_.Read();     }
    /// Device pointer to the global indices of UpperOnly variables (length NumUpper()).
    const int* IdxUpper()     const { return idx_upper_.Read();     }
    /// Device pointer to the global indices of TwoSided variables (length NumTwoSided()).
    const int* IdxTwoSided()  const { return idx_twosided_.Read();  }
    /// @}

    /// @name Raw device pointers to compact bound-value arrays
    /// Each array is aligned with its corresponding index array: the k-th
    /// entry corresponds to the variable at index `IdxXxx()[k]`.
    /// @{

    /// Lower bounds lᵢ for LowerOnly variables (length NumLower()).
    const mfem::real_t* LoLower()    const { return lo_lower_.Read();    }
    /// Upper bounds uᵢ for UpperOnly variables (length NumUpper()).
    const mfem::real_t* HiUpper()    const { return hi_upper_.Read();    }
    /// Lower bounds lᵢ for TwoSided variables (length NumTwoSided()).
    const mfem::real_t* LoTwoSided() const { return lo_twosided_.Read(); }
    /// Upper bounds uᵢ for TwoSided variables (length NumTwoSided()).
    const mfem::real_t* HiTwoSided() const { return hi_twosided_.Read(); }
    /// @}

    /**
     * @brief Reconstruct the full-length BoundType classification vector.
     *
     * Iterates over the packed index arrays on the host and fills `types`
     * with the BoundType of each of the n variables.  Useful for host-side
     * diagnostic code that needs the classification without storing it
     * separately.
     *
     * @param[out] types  Resized to Total() and filled on return.
     */
    void GetTypes(std::vector<BoundType>& types) const;

private:
    int n_unbounded_;  ///< Count of Unbounded variables.
    int n_lower_;      ///< Count of LowerOnly variables.
    int n_upper_;      ///< Count of UpperOnly variables.
    int n_twosided_;   ///< Count of TwoSided variables.

    /// Packed global indices per bound type (live on device when UseDevice=true).
    mfem::Array<int> idx_unbounded_, idx_lower_, idx_upper_, idx_twosided_;

    /// Compact bound values, aligned with the corresponding index arrays.
    mfem::Vector lo_lower_;     ///< lᵢ for LowerOnly  (length n_lower_)
    mfem::Vector hi_upper_;     ///< uᵢ for UpperOnly  (length n_upper_)
    mfem::Vector lo_twosided_;  ///< lᵢ for TwoSided   (length n_twosided_)
    mfem::Vector hi_twosided_;  ///< uᵢ for TwoSided   (length n_twosided_)
};


// ============================================================
/// @defgroup latent_helpers Primal ↔ Latent conversion helpers
///
/// Free functions that convert between the primal variable λ ∈ K and the
/// unconstrained latent variable z ∈ ℝⁿ.  Two overload families are
/// provided:
///  - **BoundPartition overloads** — device-aware, branch-free, use
///    pre-built index arrays.  Preferred for performance.
///  - **Convenience overloads** — host-only, accept a
///    `std::vector<BoundType>` built by ClassifyBounds().  Useful for
///    initialization or one-off conversions where a BoundPartition has not
///    been constructed yet.
///
/// @note All device-aware overloads inherit the device flag from their
///       primary input vector (lam or z).  Set UseDevice(true) on that
///       vector before calling to run on GPU.
// ============================================================

/// @{

/**
 * @brief Classify each variable into one of the four BoundType categories.
 *
 * Reads `lo` and `hi` on the host.  Absent bounds are represented by
 * ±std::numeric_limits<real_t>::infinity().  The result can be passed to
 * the convenience overloads of PrimalToLatent / LatentToPrimal, or used
 * to construct a BoundPartition.
 *
 * @param lo     Lower-bound vector (length n).
 * @param hi     Upper-bound vector (length n).
 * @param types  Output: resized to n and filled with the BoundType of each
 *               variable.
 */
void ClassifyBounds(const mfem::Vector& lo,
                    const mfem::Vector& hi,
                    std::vector<BoundType>& types);

/**
 * @brief λ → z  (eq. 9).  Device-aware, branch-free, boundary-safe.
 *
 * Applies the forward Legendre maps componentwise using four separate
 * forall kernels (one per bound type), each free of branches.  The device
 * flag is inherited from @p lam.
 *
 * @par Boundary safety
 * When λᵢ equals or exceeds a finite bound the log argument would be ≤ 0,
 * producing −Inf or NaN.  Each argument is clamped to a positive minimum δ
 * before the log call:
 *   - LowerOnly / UpperOnly:  δ = kPrimalMinGap
 *   - TwoSided:               δ = max(kPrimalMinGap, ε_machine × (uᵢ − lᵢ))
 *
 * where kPrimalMinGap = exp(−kLatentClipDefault) is stored as an exact
 * IEEE 754 hex float literal so that log(kPrimalMinGap) equals
 * −kLatentClipDefault exactly in floating point.  This guarantees
 * |zᵢ| ≤ kLatentClipDefault (with equality when λᵢ is exactly at a bound),
 * consistent with the latent clipping applied after each optimizer step.
 * The clamps are branchless fmax operations with no GPU divergence.
 *
 * Use BoundSafetyCheck() to diagnose bound violations before calling.
 *
 * @param lam   Primal variable λ (device flag inherited).
 * @param part  BoundPartition built from the same lo/hi.
 * @param z     Output latent vector (resized and device-flagged to match lam).
 *
 * @pre lam.Size() == part.Total().
 * @pre For every finite-bound variable: lᵢ ≤ λᵢ ≤ uᵢ  (soft: out-of-bounds
 *      inputs are clamped rather than rejected, but the latent value will be
 *      at ±kLatentClipDefault).
 */
void PrimalToLatent(const mfem::Vector& lam,
                    const BoundPartition& part,
                    mfem::Vector& z);

/**
 * @brief Host-side diagnostic: count primal entries at or beyond their bounds.
 *
 * Transfers lam to the host if needed and counts the number of entries
 * where:
 *   - λᵢ ≤ lᵢ + tol  (LowerOnly lower side, or TwoSided lower side)
 *   - λᵢ ≥ uᵢ − tol  (UpperOnly upper side, or TwoSided upper side)
 *
 * Unbounded variables are never counted.
 *
 * Returns the total violation count (0 = all strictly feasible within tol).
 * Does not modify any vector and does not affect optimizer state.
 *
 * @param lam   Primal variable to check (may be on device; read on host).
 * @param part  BoundPartition built from the same lo/hi.
 * @param lo    Full lower-bound vector.
 * @param hi    Full upper-bound vector.
 * @param tol   Proximity threshold.  Default 0 = exact boundary only.
 *              Use a small positive value to detect near-boundary points.
 * @return Number of violated entries.
 */
int BoundSafetyCheck(const mfem::Vector& lam,
                     const BoundPartition& part,
                     const mfem::Vector& lo,
                     const mfem::Vector& hi,
                     mfem::real_t tol = mfem::real_t(0));

/**
 * @brief z → λ = T(z) ∈ int(K)  (eqs. 10–12).  Device-aware, branch-free.
 *
 * Applies the inverse Legendre maps (eqs. 10–12) componentwise:
 *   Unbounded:  λᵢ = zᵢ
 *   LowerOnly:  λᵢ = lᵢ + exp(zᵢ)
 *   UpperOnly:  λᵢ = uᵢ − exp(−zᵢ)
 *   TwoSided:   λᵢ = lᵢ + (uᵢ−lᵢ) σ(zᵢ)
 *
 * Strict feasibility lᵢ < λᵢ < uᵢ is guaranteed for every finite zᵢ.
 * Four separate branch-free forall kernels are launched.  Device flag
 * inherited from @p z.
 *
 * @param z     Latent variable (unrestricted; device flag inherited).
 * @param part  BoundPartition built from the same lo/hi.
 * @param lam   Output primal variable (resized and device-flagged to match z).
 *
 * @pre z.Size() == part.Total().
 */
void LatentToPrimal(const mfem::Vector& z,
                    const BoundPartition& part,
                    mfem::Vector& lam);

/**
 * @brief Diagonal of the Jacobian JT(z) = dλ/dz.  Device-aware, branch-free.
 *
 * Computes the diagonal entries of the (diagonal) Jacobian of the inverse
 * map T, derived from eqs. (43)–(44):
 *   Unbounded:  JT[i] = 1
 *   LowerOnly:  JT[i] = exp(zᵢ)
 *   UpperOnly:  JT[i] = exp(−zᵢ)
 *   TwoSided:   JT[i] = (uᵢ−lᵢ) σ(zᵢ)(1 − σ(zᵢ))
 *
 * All entries are strictly positive for any finite z.  Useful for
 * debugging descent direction quality (eq. 44) and for preconditioning.
 *
 * @param z     Latent variable.
 * @param part  BoundPartition.
 * @param diag  Output diagonal (resized and device-flagged to match z).
 */
void LatentJacobianDiag(const mfem::Vector& z,
                        const BoundPartition& part,
                        mfem::Vector& diag);

/**
 * @brief Default strictly-feasible primal initialization (eq. 35).
 *
 * Sets λᵢ to a canonical interior point for each bound type:
 *   Unbounded:  λᵢ = 0
 *   LowerOnly:  λᵢ = lᵢ + 1
 *   UpperOnly:  λᵢ = uᵢ − 1
 *   TwoSided:   λᵢ = (lᵢ + uᵢ) / 2
 *
 * Device-aware and branch-free.  The result satisfies strict feasibility
 * provided the bounds are non-degenerate (lᵢ + 1 < uᵢ − 1 for TwoSided
 * bounds with width > 2).  For narrow intervals initialize manually.
 *
 * @param part  BoundPartition (provides per-type index arrays and bounds).
 * @param lo    Full lower-bound vector.
 * @param hi    Full upper-bound vector.
 * @param lam   Output primal vector (resized and device-flagged to match lo).
 */
void DefaultPrimalInit(const BoundPartition& part,
                       const mfem::Vector& lo,
                       const mfem::Vector& hi,
                       mfem::Vector& lam);

// ── Host-only convenience overloads ──────────────────────────────────────
// These accept a std::vector<BoundType> produced by ClassifyBounds() and
// run entirely on the host.  They implement the same clamping strategy as
// PrimalToLatent(BoundPartition) and are suitable for initialization or
// debugging without constructing a BoundPartition.

/**
 * @brief λ → z  (host-only convenience overload).
 *
 * Identical arithmetic to the BoundPartition overload, including the
 * boundary-safety clamping.  Runs on the host regardless of the device
 * flag of the inputs.
 *
 * @param lam    Primal variable (read on host).
 * @param lo     Lower bounds.
 * @param hi     Upper bounds.
 * @param types  Per-variable bound classification from ClassifyBounds().
 * @param z      Output latent variable (written on host, resized to n).
 */
void PrimalToLatent(const mfem::Vector& lam,
                    const mfem::Vector& lo,
                    const mfem::Vector& hi,
                    const std::vector<BoundType>& types,
                    mfem::Vector& z);

/**
 * @brief z → λ  (host-only convenience overload).
 *
 * @param z      Latent variable (read on host).
 * @param lo     Lower bounds.
 * @param hi     Upper bounds.
 * @param types  Per-variable bound classification.
 * @param lam    Output primal variable (written on host, resized to n).
 */
void LatentToPrimal(const mfem::Vector& z,
                    const mfem::Vector& lo,
                    const mfem::Vector& hi,
                    const std::vector<BoundType>& types,
                    mfem::Vector& lam);

/**
 * @brief Default strictly-feasible primal initialization (host-only).
 *
 * @param lo     Lower bounds.
 * @param hi     Upper bounds.
 * @param types  Per-variable bound classification.
 * @param lam    Output primal variable (written on host, resized to n).
 */
void DefaultPrimalInit(const mfem::Vector& lo,
                       const mfem::Vector& hi,
                       const std::vector<BoundType>& types,
                       mfem::Vector& lam);

/// @}  (end of latent_helpers group)


// ============================================================
/**
 * @class LatentMirrorOptimizer
 * @brief Serial device-aware latent-variable mirror-gradient optimizer.
 *
 * Solves  min Φ(λ)  s.t. lᵢ ≤ λᵢ ≤ uᵢ  by performing gradient steps in
 * the unconstrained latent space z.  The user holds z and calls
 * LatentToPrimal() to read the primal λ = T(z) when needed.
 *
 * @par Device selection
 * The device is determined by z.UseDevice() at each Update() call.  All
 * internal vectors are allocated on the same device at the first call.
 * Moving z between devices between calls is not supported.
 *
 * @par Inner product
 * Passing M_op and M_solver selects the M-geometry; both null gives
 * Euclidean M = I.  The caller retains ownership of both objects.
 *
 * @par Armijo callback
 * Update() accepts an optional EvalPhi callback for the backtracking line
 * search.  The callback receives the trial latent vector z_trial and must
 * set phi_out = Φ(T(z_trial)).  Passing null skips the line-search check
 * and accepts the GBB trial step immediately.
 *
 * @par Typical usage
 * @code
 *   LatentMirrorOptimizer opt(z0, lo, hi);   // identity M
 *   // or: LatentMirrorOptimizer opt(z0, lo, hi, &mass_op, &mass_solver);
 *
 *   for (int k = 0; k < max_iter; ++k) {
 *       LatentToPrimal(z, part, lam);
 *       real_t phi = EvalObjective(lam);
 *       EvalGradient(lam, d);
 *       opt.Update(z, d, phi,
 *           [&](const Vector& zt, real_t& po){
 *               LatentToPrimal(zt, part, lt); po = EvalObjective(lt);
 *           });
 *       if (opt.StationarityResidual() < tol) break;
 *   }
 * @endcode
 */
// ============================================================
class LatentMirrorOptimizer {
public:
    /**
     * @brief Armijo line-search callback type.
     *
     * Called inside the backtracking loop with a trial latent vector.
     * Must set `phi_out` = Φ(T(z_trial)).  The trial vector lives on the
     * same device as z; call LatentToPrimal() to recover the primal.
     *
     * @param z_trial  Trial latent vector (do not modify).
     * @param phi_out  Output: objective at the trial point.
     */
    using EvalPhi = std::function<void(const mfem::Vector& z_trial,
                                       mfem::real_t& phi_out)>;

    // ── Constructors ──────────────────────────────────────────────────────

    /**
     * @brief Construct with identity inner product M = I.
     *
     * @param z0  Initial latent vector (user-owned; copied internally).
     *            Its device flag is inherited by all internal vectors.
     * @param lo  Lower bounds (length n; use −∞ for absent).
     * @param hi  Upper bounds (length n; use +∞ for absent).
     *
     * @pre z0.Size() == lo.Size() == hi.Size() > 0.
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
     * @param M_op      Symmetric positive-definite operator M (null → I).
     *                  Used to evaluate pairings ⟨x, y⟩_M = xᵀ M y.
     * @param M_solver  Solver for M x = b (null → identity).
     *                  Used to compute gᵏ = M⁻¹ dᵏ.
     *
     * The caller retains ownership of M_op and M_solver.  Both may be null
     * independently; if M_op is non-null but M_solver is null, the pairings
     * use M but the gradient step uses the Euclidean direction.
     */
    LatentMirrorOptimizer(const mfem::Vector& z0,
                          const mfem::Vector& lo,
                          const mfem::Vector& hi,
                          mfem::Operator* M_op,
                          mfem::Solver*   M_solver);

    ~LatentMirrorOptimizer() = default;

    // ── Configuration ─────────────────────────────────────────────────────

    /**
     * @brief Set Armijo sufficient-decrease line-search parameters.
     *
     * @param c1     Armijo constant: step accepted if Φ(λ⁺) ≤ Φ(λ) + c₁ ⟨g,Δλ⟩_M.
     *               Must be ∈ (0, 1).  Default 1e-4.
     * @param beta   Step-reduction factor per backtrack.  Must be ∈ (0, 1).
     *               Default 0.5.
     * @param max_ls Maximum number of backtracking halvings before accepting.
     *               Default 50.
     */
    void SetLineSearchParams(mfem::real_t c1   = mfem::real_t(1e-4),
                             mfem::real_t beta = mfem::real_t(0.5),
                             int          max_ls = 50);

    /**
     * @brief Set safeguard bounds on the GBB step-size estimate.
     *
     * The GBB step (eq. 18) is clamped to [alpha_min, alpha_max] and the
     * geometric mean (eq. 21) then initializes the backtracking.  The GBB
     * formula is skipped and a fallback used when either its numerator or
     * denominator is below eps_bb in magnitude.
     *
     * @param alpha_min  Lower clamp on step size.  Default 1e-12.
     * @param alpha_max  Upper clamp on step size.  Default 1e4.
     * @param eps_bb     GBB skip threshold.        Default 1e-14.
     */
    void SetStepSizeSafeguards(mfem::real_t alpha_min = mfem::real_t(1e-12),
                                mfem::real_t alpha_max = mfem::real_t(1e4),
                                mfem::real_t eps_bb   = mfem::real_t(1e-14));

    /**
     * @brief Set the latent-variable clipping magnitude.
     *
     * After each gradient step the latent vector is clipped to [−zmax, zmax]
     * element-wise.  This prevents sigmoid saturation and floating-point
     * overflow while preserving feasibility: T(z) ∈ int(K) for any z.
     * Default 40  (σ(±40) ≈ 1 − 4×10⁻¹⁸ in double).
     *
     * @param zmax  Clipping magnitude > 0.  Must match kLatentClipDefault if
     *              the default PrimalToLatent clamping is to be consistent.
     */
    void SetLatentClipping(mfem::real_t zmax = mfem::real_t(40));

    /**
     * @brief Set the relative stationarity tolerance used by HasConverged().
     *
     * @param tol  Convergence threshold > 0.  Default 1e-6.
     */
    void SetStationarityTol(mfem::real_t tol = mfem::real_t(1e-6));

    // ── Outer iteration ───────────────────────────────────────────────────

    /**
     * @brief Perform one GBB + Armijo mirror-gradient step.
     *
     * Executes steps 1–4 of the algorithm (§8 of the paper):
     *  1. gᵏ = M⁻¹ dᵏ.
     *  2. αₜ from GBB (eq. 18) or default (eq. 22) on k = 0.
     *  3. Backtracking Armijo:
     *       z⁺ = zᵏ − α gᵏ → clip → λ⁺ = T(z⁺)
     *       accept if Φ(λ⁺) ≤ Φ(λᵏ) + c₁ (dᵏ)ᵀ(λ⁺ − λᵏ)
     *  4. Accept: z ← z⁺.
     *
     * On return, z holds zᵏ⁺¹ and the cached diagnostics are updated.
     *
     * @param[in,out] z        User-owned latent variable (updated in-place).
     *                         Device flag determines where kernels run.
     * @param[in]     d        Euclidean derivative dᵏ = ∇_E Φ(T(z)) at the
     *                         current z.  Same device as z.
     * @param[in]     phi_k    Objective value Φ(T(z)) at the current z.
     *                         Must match z on entry (not the updated value).
     * @param[in]     eval_phi Callback for Armijo objective evaluation
     *                         (see EvalPhi).  Null → accept GBB step directly.
     * @return Accepted step size αᵏ.
     */
    mfem::real_t Update(mfem::Vector&       z,
                        const mfem::Vector& d,
                        mfem::real_t        phi_k,
                        EvalPhi             eval_phi = nullptr);

    // ── Convergence diagnostics ───────────────────────────────────────────

    /**
     * @brief Euclidean projected-gradient stationarity residual (eq. 33).
     *
     * Computes  ‖λ − Π_K(λ − d)‖₂ / max(1, ‖λ‖₂)
     * where Π_K clips λ − d componentwise into [l, u].  This checks the KKT
     * sign conditions (eq. 28) and is cheap (one device kernel + Sum).
     *
     * The primal λ = T(z) is computed internally.
     *
     * @param z  Current latent variable.
     * @param d  Euclidean derivative at T(z).
     * @return Relative stationarity residual ≥ 0.
     */
    mfem::real_t StationarityResidual(const mfem::Vector& z,
                                      const mfem::Vector& d) const;

    /**
     * @brief Stationarity residual cached from the last Update() call.
     *
     * Computed at the **updated** iterate zᵏ⁺¹ using the gradient dᵏ that
     * was passed to Update().  Returns −1 before the first Update() call.
     */
    mfem::real_t StationarityResidual() const { return last_stat_res_; }

    /**
     * @brief True if the cached stationarity residual < stationarity tolerance.
     *
     * Equivalent to StationarityResidual() < stat_tol (set by SetStationarityTol).
     * Returns false before the first Update() call.
     */
    bool HasConverged() const {
        return last_stat_res_ >= mfem::real_t(0) &&
               last_stat_res_ < stat_tol_;
    }

    /**
     * @brief Relative M-norm step size from the last Update() (eq. 34).
     *
     * Returns ‖λᵏ⁺¹ − λᵏ‖_M / max(1, ‖λᵏ‖_M).  A small value indicates
     * the iterate is no longer moving significantly.
     * Returns −1 before the first Update() call.
     */
    mfem::real_t RelativeStepSize()    const { return last_rel_step_;  }

    /**
     * @brief Number of Armijo step-halvings in the last Update() call.
     *
     * Returns 0 if the initial GBB trial step was accepted without reduction.
     */
    int LastLineSearchSteps() const { return last_ls_steps_;  }

    /** @brief Total number of completed Update() calls. */
    int NumIterations()       const { return iter_;           }

    /**
     * @brief The BoundPartition built at construction.
     *
     * Provides access to type counts, index arrays, and bound arrays for
     * use in LatentToPrimal(), PrimalToLatent(), etc.
     */
    const BoundPartition& GetPartition() const { return part_; }

private:
    int            n_;             ///< Number of design variables.
    BoundPartition part_;          ///< Per-type index and bound arrays.
    mfem::Vector   lo_full_;       ///< Full lower-bound vector (on device).
    mfem::Vector   hi_full_;       ///< Full upper-bound vector (on device).

    mfem::Operator* M_op_;         ///< SPD operator M (null = identity).
    mfem::Solver*   M_solver_;     ///< Solver for M x = b (null = identity).

    /// GBB history vectors (live on the same device as z).
    mfem::Vector z_prev_;          ///< Latent variable at previous step  zᵏ⁻¹.
    mfem::Vector lam_prev_;        ///< Primal at previous step  T(zᵏ⁻¹).
    mfem::Vector g_prev_;          ///< M-gradient at previous step  gᵏ⁻¹.

    /// Per-iteration work vectors (avoid repeated allocation).
    mfem::Vector g_;               ///< Current M-gradient  gᵏ = M⁻¹ dᵏ.
    mfem::Vector Mg_;              ///< M gᵏ = dᵏ  (for Armijo pairing).
    mfem::Vector z_trial_;         ///< Trial latent vector  z⁺ = zᵏ − α gᵏ.
    mfem::Vector lam_cur_;         ///< Current primal  λᵏ = T(zᵏ).
    mfem::Vector lam_trial_;       ///< Trial primal  λ⁺ = T(z⁺).

    bool initialized_;  ///< False until first Update(); deferred device init.

    mfem::real_t alpha_prev_;   ///< Accepted step size from previous iteration.
    mfem::real_t alpha_min_;    ///< GBB lower safeguard.
    mfem::real_t alpha_max_;    ///< GBB upper safeguard.
    mfem::real_t eps_bb_;       ///< GBB skip threshold.
    mfem::real_t c1_;           ///< Armijo sufficient-decrease constant.
    mfem::real_t beta_;         ///< Armijo step-reduction factor.
    int          max_ls_;       ///< Maximum Armijo backtracking steps.
    mfem::real_t zmax_;         ///< Latent clipping magnitude.
    mfem::real_t stat_tol_;     ///< HasConverged() threshold.

    int          iter_;             ///< Iteration counter.
    int          last_ls_steps_;    ///< Backtracking steps in last Update().
    mfem::real_t last_stat_res_;    ///< Cached stationarity residual.
    mfem::real_t last_rel_step_;    ///< Cached relative M-norm step size.

    /** @brief Allocate and device-flag all internal work vectors on first Update(). */
    void InitDeviceVectors(const mfem::Vector& ref);

    /** @brief Apply M: Mx = M_op * x, or Mx = x when M_op is null. */
    void ApplyM(const mfem::Vector& x, mfem::Vector& Mx) const;
    /** @brief Apply M⁻¹: Minvx = M_solver * x, or Minvx = x when null. */
    void ApplyMinv(const mfem::Vector& x, mfem::Vector& Minvx) const;
    /** @brief Compute ⟨x, y⟩_M = xᵀ M y (or xᵀ y when M_op is null). */
    mfem::real_t InnerProductM(const mfem::Vector& x,
                                const mfem::Vector& y) const;
    /** @brief Compute ‖x‖_M = √⟨x,x⟩_M. */
    mfem::real_t NormM(const mfem::Vector& x) const {
        return std::sqrt(std::max(mfem::real_t(0), InnerProductM(x, x)));
    }
    /**
     * @brief Compute the GBB trial step size  (eq. 18 + eq. 21).
     *
     * αGBB = ⟨Δz, Δλ⟩_M / |⟨Δg, Δλ⟩_M|, clamped and combined with
     * alpha_prev_ via geometric mean (eq. 21).  Falls back to @p fallback
     * when the GBB estimate is numerically unreliable.
     */
    mfem::real_t ComputeGBBStepSize(const mfem::Vector& dz,
                                     const mfem::Vector& dlam,
                                     const mfem::Vector& dg,
                                     mfem::real_t fallback) const;
    /** @brief Clip z to [−zmax_, zmax_] element-wise using a device kernel. */
    void ClipLatent(mfem::Vector& z) const;
    /**
     * @brief Compute the projected-gradient stationarity residual on device.
     *
     * Launches a forall kernel to fill rᵢ² and λᵢ² arrays, then uses
     * Vector::Sum() (cuBLAS on GPU) to assemble the scalar result.
     */
    mfem::real_t ProjectedGradResidual(const mfem::Vector& lam,
                                        const mfem::Vector& d) const;
};


// ============================================================
#ifdef MFEM_USE_MPI
/**
 * @class LatentMirrorOptimizerParallel
 * @brief MPI-parallel device-aware latent-variable mirror-gradient optimizer.
 *
 * Distributes z across MPI ranks.  Each rank owns a contiguous local chunk
 * z_local of the global latent vector.  The per-rank BoundPartition is built
 * from the local lower/upper bounds; all bounds within a rank's chunk may
 * differ independently.
 *
 * @par Global consensus
 * All collective scalars — GBB numerator ⟨Δz,Δλ⟩_M, denominator ⟨Δg,Δλ⟩_M,
 * initial step size ‖g⁰‖∞, Armijo pairing (dᵏ)ᵀ(λ⁺−λᵏ), and the
 * stationarity residual — are assembled via MPI_Allreduce.  Every rank
 * evaluates the same GBB step size and the same Armijo test, so all ranks
 * accept or reject the trial step identically without additional
 * communication.
 *
 * @par Armijo callback
 * The eval_phi callback receives the LOCAL z_trial slice.  It must return
 * the GLOBAL objective value (the same double on every rank).  The callback
 * is responsible for any MPI reduction needed (e.g., MPI_Allreduce of local
 * objective contributions).
 *
 * @par Device selection
 * The device flag is read from z_local at each Update() call.
 * MPI communication always uses host memory; device vectors are copied to
 * host automatically by MFEM's memory manager when needed for MPI sends.
 *
 * @par Typical usage
 * @code
 *   LatentMirrorOptimizerParallel opt(MPI_COMM_WORLD, z_loc, lo_loc, hi_loc);
 *
 *   for (int k = 0; k < max_iter; ++k) {
 *       LatentToPrimal(z_loc, part_loc, lam_loc);
 *       // phi = global objective; compute locally + MPI_Allreduce
 *       double phi = LocalPhi(lam_loc);
 *       MPI_Allreduce(&phi, &phi_global, 1, MPI_DOUBLE, MPI_SUM, comm);
 *       LocalGrad(lam_loc, d_loc);
 *       opt.Update(z_loc, d_loc, real_t(phi_global),
 *           [&](const Vector& zt, real_t& po) {
 *               LatentToPrimal(zt, part_loc, lt);
 *               double lp = LocalPhi(lt);
 *               MPI_Allreduce(&lp, &po, 1, MPI_DOUBLE, MPI_SUM, comm);
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
    /**
     * @brief Armijo callback type for the parallel optimizer.
     *
     * Receives the LOCAL trial latent slice.  Must set phi_out to the
     * GLOBAL objective — the same value on every rank.  The callback is
     * called on ALL ranks simultaneously.
     *
     * @param z_trial_local  Local trial latent slice (do not modify).
     * @param phi_out        Output: global objective at the trial point.
     */
    using EvalPhi = std::function<void(const mfem::Vector& z_trial_local,
                                       mfem::real_t& phi_out)>;

    // ── Constructors ──────────────────────────────────────────────────────

    /**
     * @brief Construct with identity M.
     *
     * @param comm      MPI communicator (e.g., MPI_COMM_WORLD).
     * @param z0_local  Initial local latent chunk (user-owned; copied).
     * @param lo_local  Local lower bounds (same length as z0_local).
     * @param hi_local  Local upper bounds.
     */
    LatentMirrorOptimizerParallel(MPI_Comm            comm,
                                   const mfem::Vector& z0_local,
                                   const mfem::Vector& lo_local,
                                   const mfem::Vector& hi_local);

    /**
     * @brief Construct with a local M operator and solver.
     *
     * @param comm           MPI communicator.
     * @param z0_local       Initial local latent chunk.
     * @param lo_local       Local lower bounds.
     * @param hi_local       Local upper bounds.
     * @param M_op_local     Local block of M (null → identity).  Applied to
     *                       local vectors; the global inner product is assembled
     *                       by MPI_Allreduce of local dot products.
     * @param M_solver_local Local solver for the M-block (null → identity).
     *
     * The global M-inner product is computed as:
     *   ⟨x, y⟩_M = Σ_ranks  xlocᵀ M_local y_local
     */
    LatentMirrorOptimizerParallel(MPI_Comm            comm,
                                   const mfem::Vector& z0_local,
                                   const mfem::Vector& lo_local,
                                   const mfem::Vector& hi_local,
                                   mfem::Operator*     M_op_local,
                                   mfem::Solver*       M_solver_local);

    ~LatentMirrorOptimizerParallel() = default;

    // ── Configuration ─────────────────────────────────────────────────────

    /// @copydoc LatentMirrorOptimizer::SetLineSearchParams
    void SetLineSearchParams(mfem::real_t c1   = mfem::real_t(1e-4),
                             mfem::real_t beta = mfem::real_t(0.5),
                             int          max_ls = 50);
    /// @copydoc LatentMirrorOptimizer::SetStepSizeSafeguards
    void SetStepSizeSafeguards(mfem::real_t alpha_min = mfem::real_t(1e-12),
                                mfem::real_t alpha_max = mfem::real_t(1e4),
                                mfem::real_t eps_bb   = mfem::real_t(1e-14));
    /// @copydoc LatentMirrorOptimizer::SetLatentClipping
    void SetLatentClipping(mfem::real_t zmax = mfem::real_t(40));
    /// @copydoc LatentMirrorOptimizer::SetStationarityTol
    void SetStationarityTol(mfem::real_t tol = mfem::real_t(1e-6));

    // ── Outer iteration ───────────────────────────────────────────────────

    /**
     * @brief Perform one parallel GBB + Armijo mirror-gradient step.
     *
     * Identical algorithm to LatentMirrorOptimizer::Update() except that
     * all global scalars are assembled via MPI_Allreduce so every rank
     * computes the same step size and accepts the same trial point.
     *
     * @param[in,out] z_local   Local latent chunk (updated in-place).
     * @param[in]     d_local   Local Euclidean derivative ∇_E Φ at T(z_local).
     * @param[in]     phi_k     Global objective Φ(T(z)) — same on all ranks.
     * @param[in]     eval_phi  Armijo callback returning GLOBAL Φ at trial
     *                          z_local.  Called on ALL ranks.  Null skips check.
     * @return Accepted step size αᵏ (same on all ranks).
     */
    mfem::real_t Update(mfem::Vector&       z_local,
                        const mfem::Vector& d_local,
                        mfem::real_t        phi_k,
                        EvalPhi             eval_phi = nullptr);

    // ── Convergence diagnostics ───────────────────────────────────────────

    /**
     * @brief Global projected-gradient stationarity residual (eq. 33).
     *
     * Global 2-norms assembled via MPI_Allreduce; identical on all ranks.
     *
     * @param z_local  Local latent chunk.
     * @param d_local  Local Euclidean derivative.
     * @return Global relative stationarity residual ≥ 0.
     */
    mfem::real_t StationarityResidual(const mfem::Vector& z_local,
                                       const mfem::Vector& d_local) const;

    /// @copydoc LatentMirrorOptimizer::StationarityResidual() const
    mfem::real_t StationarityResidual() const { return last_stat_res_; }

    /// @copydoc LatentMirrorOptimizer::HasConverged
    bool HasConverged() const {
        return last_stat_res_ >= mfem::real_t(0) &&
               last_stat_res_ < stat_tol_;
    }

    /// @copydoc LatentMirrorOptimizer::RelativeStepSize
    mfem::real_t RelativeStepSize()    const { return last_rel_step_;  }
    /// @copydoc LatentMirrorOptimizer::LastLineSearchSteps
    int          LastLineSearchSteps() const { return last_ls_steps_;  }
    /// @copydoc LatentMirrorOptimizer::NumIterations
    int          NumIterations()       const { return iter_;           }

    /// @copydoc LatentMirrorOptimizer::GetPartition
    const BoundPartition& GetPartition() const { return part_; }

private:
    MPI_Comm       comm_;         ///< MPI communicator.
    int            n_local_;      ///< Local number of design variables.
    BoundPartition part_;         ///< Local per-type index and bound arrays.
    mfem::Vector   lo_full_local_; ///< Local lower bounds (on device).
    mfem::Vector   hi_full_local_; ///< Local upper bounds (on device).

    mfem::Operator* M_op_;         ///< Local M block (null = identity).
    mfem::Solver*   M_solver_;     ///< Local M⁻¹ solver (null = identity).

    mfem::Vector z_prev_local_;    ///< Previous local latent  zᵏ⁻¹.
    mfem::Vector lam_prev_local_;  ///< Previous local primal  T(zᵏ⁻¹).
    mfem::Vector g_prev_local_;    ///< Previous local M-gradient  gᵏ⁻¹.

    mfem::Vector g_local_;         ///< Current local M-gradient.
    mfem::Vector Mg_local_;        ///< M gᵏ (for Armijo pairing).
    mfem::Vector z_trial_local_;   ///< Trial local latent z⁺.
    mfem::Vector lam_cur_local_;   ///< Current local primal T(zᵏ).
    mfem::Vector lam_trial_local_; ///< Trial local primal T(z⁺).

    bool initialized_;  ///< False until first Update(); deferred device init.

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

    /** @brief Allocate and device-flag all local work vectors on first Update(). */
    void InitDeviceVectors(const mfem::Vector& ref);
    /** @brief Apply local M block (or identity). */
    void ApplyM(const mfem::Vector& x, mfem::Vector& Mx) const;
    /** @brief Apply local M⁻¹ (or identity). */
    void ApplyMinv(const mfem::Vector& x, mfem::Vector& Minvx) const;
    /**
     * @brief Compute global M-inner product via local dot + MPI_Allreduce.
     *
     * ⟨x, y⟩_M = Σ_ranks  xlocᵀ M_local y_local, reduced via MPI_SUM.
     * The result is identical on all ranks.
     */
    mfem::real_t InnerProductM(const mfem::Vector& x,
                                const mfem::Vector& y) const;
    /** @brief Global M-norm ‖x‖_M = √⟨x,x⟩_M (via InnerProductM). */
    mfem::real_t NormM(const mfem::Vector& x) const {
        return std::sqrt(std::max(mfem::real_t(0), InnerProductM(x, x)));
    }
    /** @brief Compute GBB step size using global inner products. */
    mfem::real_t ComputeGBBStepSize(const mfem::Vector& dz,
                                     const mfem::Vector& dlam,
                                     const mfem::Vector& dg,
                                     mfem::real_t fallback) const;
    /** @brief Clip local z to [−zmax_, zmax_] on device. */
    void ClipLatent(mfem::Vector& z) const;
    /**
     * @brief Compute global projected-gradient residual.
     *
     * Evaluates local rᵢ² and λᵢ² via forall, then assembles global sums
     * via MPI_Allreduce(2 doubles, MPI_SUM).
     */
    mfem::real_t ProjectedGradResidualGlobal(const mfem::Vector& lam_local,
                                              const mfem::Vector& d_local) const;
};
#endif // MFEM_USE_MPI

} // namespace mfem_lmg
