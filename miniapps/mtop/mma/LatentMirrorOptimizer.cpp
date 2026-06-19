/**
 * @file LatentMirrorOptimizer.cpp
 * @brief Implementation of the latent-variable mirror-gradient optimizer.
 *
 * @section overview Implementation overview
 * This file implements the algorithm of the paper:
 *   "A Latent-Variable Mirror-Gradient Algorithm for Bound-Constrained
 *    Minimization with a General Positive Definite Inner Product"
 *
 * All equation numbers below refer to that paper.
 *
 * @section device_strategy Device strategy (following MMA_MFEM.cpp)
 *   1. use_dev = z.UseDevice()  — read at every Update() entry.
 *   2. All O(n) loops → mfem::forall_switch(use_dev, n, lambda).
 *   3. Device data accessed ONLY via Read() / Write() / ReadWrite().
 *   4. HostRead() / HostWrite() used only for scalar quantities and the
 *      one-time BoundPartition construction.
 *   5. Scalar reductions (InnerProduct, Normlinf, Vector::Sum) dispatch
 *      automatically to cuBLAS / rocBLAS on GPU.
 *   6. Internal vectors are allocated lazily at the first Update() call
 *      (InitDeviceVectors) so the device flag is always inherited from the
 *      user's latent vector z.
 *
 * @section branching_strategy GPU branching strategy — BoundPartition
 * A single forall kernel with switch(BoundType[i]) causes warp divergence
 * (threads in the same 32-wide warp serialize across branches).  Instead,
 * BoundPartition stores four compact index arrays built on the host and
 * uploaded once to the device.  Every primal↔latent conversion then
 * launches four separate branch-free kernels, one per bound type, each
 * reading contiguous coalesced memory.
 *
 * @section boundary_safety Boundary safety in PrimalToLatent
 * The log argument is clamped to a positive minimum δ before evaluation so
 * that λ at or beyond a finite bound produces a large-but-finite z rather
 * than −Inf or NaN:
 *   LowerOnly / UpperOnly:  δ = kPrimalMinGap
 *   TwoSided:               δ = max(kPrimalMinGap, ε_machine × (u − l))
 *
 * kPrimalMinGap = exp(−kLatentClipDefault) is stored as an exact IEEE 754
 * hex float literal, guaranteeing log(kPrimalMinGap) == −kLatentClipDefault
 * exactly in double precision.  The clamps use branchless fmax — no warp
 * divergence.
 */

#include "LatentMirrorOptimizer.hpp"

#include <cassert>
#include <cmath>
#include <algorithm>
#include <limits>

namespace mfem_lmg {

// ============================================================
//  Internal anonymous-namespace helpers
//  (not visible outside this translation unit)
// ============================================================
namespace {

/// Infinity sentinel for the current floating-point type.
constexpr mfem::real_t kInf = std::numeric_limits<mfem::real_t>::infinity();

/// True when v is ≥ +∞ (absent upper bound).
MFEM_HOST_DEVICE inline bool IsInf   (mfem::real_t v) { return  v >= kInf; }
/// True when v is ≤ −∞ (absent lower bound).
MFEM_HOST_DEVICE inline bool IsNegInf(mfem::real_t v) { return  v <= -kInf; }

// ── Boundary-safety constants ─────────────────────────────────────────────
//
// When λᵢ lies at or beyond a finite bound the log argument is ≤ 0,
// producing −Inf or NaN and corrupting the optimizer state.  We clamp every
// log argument to a positive minimum δ, chosen so that
//   log(δ) = −kLatentClipDefault
// exactly in IEEE 754 arithmetic.  This ensures |zᵢ| ≤ kLatentClipDefault
// even when λᵢ is exactly at a bound — consistent with the latent clipper
// applied after each gradient step.
//
// kLatentClipDefault  : default zmax, matches SetLatentClipping() default.
// kPrimalMinGap       : exact hex float literal for exp(−kLatentClipDefault).
//                       double: 0x1.39792499b1a24p-58 = exp(−40), exact.
//                       float:  0x1.4875ccp-22 = first float with
//                               log(·) ≥ −15 (the nearest-float rounds
//                               log to −15.0000000063 < −15, so we take
//                               the next ULP up).
// kPrimalRelGap       : machine epsilon; relative floor for TwoSided bounds
//                       so the gap is never smaller than the representational
//                       precision of the bound values themselves.

/// Default latent clipping range [−zmax, zmax].
constexpr mfem::real_t kLatentClipDefault =
    mfem::real_t(sizeof(mfem::real_t) == 4 ? 15.0 : 40.0);

/// Machine epsilon ε: kPrimalRelGap × (u−l) is the smallest representable gap.
constexpr mfem::real_t kPrimalRelGap =
    std::numeric_limits<mfem::real_t>::epsilon();

/// exp(−kLatentClipDefault) as an exact IEEE 754 hex float literal.
/// Guarantees log(kPrimalMinGap) == −kLatentClipDefault exactly.
constexpr mfem::real_t kPrimalMinGap =
    mfem::real_t(sizeof(mfem::real_t) == 4
                 ? 0x1.4875ccp-22          // float: first float with log ≥ −15
                 : 0x1.39792499b1a24p-58); // double: exp(−40), exact

/**
 * @brief Branchless lower clamp: returns max(x, floor).
 *
 * Compiles to a single FMAX instruction on GPU (no branch, no divergence).
 */
MFEM_HOST_DEVICE inline mfem::real_t ClampBelow(mfem::real_t x,
                                                  mfem::real_t floor)
{ return x < floor ? floor : x; }

/**
 * @brief Allocate v with length sz and the same device flag as ref.
 *
 * Zero-initializes the vector.  Called from InitDeviceVectors() to set up
 * all per-iteration work vectors on the correct device at first Update().
 */
static void DeviceInit(mfem::Vector& v, int sz, const mfem::Vector& ref)
{
    v.SetSize(sz);
    v.UseDevice(ref.UseDevice());
    v = mfem::real_t(0);
}

} // anonymous namespace


// ============================================================
//  BoundPartition
//
//  Two-phase construction:
//  Phase 1 (host): classify all n variables and fill per-type
//           std::vector<int> index lists and std::vector<real_t> bound lists.
//  Phase 2 (upload): copy to mfem::Array<int> / mfem::Vector and push to
//           device via HostWrite() + Read().
// ============================================================

BoundPartition::BoundPartition(const mfem::Vector& lo,
                                const mfem::Vector& hi,
                                const mfem::Vector& ref)
{
    const int n = lo.Size();
    assert(hi.Size() == n);

    // ── Phase 1: classify on host ─────────────────────────────────────────
    const mfem::real_t* ld = lo.HostRead();
    const mfem::real_t* ud = hi.HostRead();

    // Temporary host vectors — one list per bound type.
    std::vector<int> h_ub, h_lo, h_up, h_ts;           // index lists
    std::vector<mfem::real_t> h_lo_lo, h_up_hi,        // compact bounds
                               h_ts_lo, h_ts_hi;

    for (int i = 0; i < n; ++i) {
        const bool has_lo = !IsNegInf(ld[i]);
        const bool has_hi = !IsInf   (ud[i]);
        if      (!has_lo && !has_hi) {
            h_ub.push_back(i);
        } else if ( has_lo && !has_hi) {
            h_lo.push_back(i);
            h_lo_lo.push_back(ld[i]);           // store lᵢ aligned with index
        } else if (!has_lo &&  has_hi) {
            h_up.push_back(i);
            h_up_hi.push_back(ud[i]);           // store uᵢ aligned with index
        } else {
            h_ts.push_back(i);
            h_ts_lo.push_back(ld[i]);           // store lᵢ and uᵢ aligned
            h_ts_hi.push_back(ud[i]);
        }
    }

    n_unbounded_ = (int)h_ub.size();
    n_lower_     = (int)h_lo.size();
    n_upper_     = (int)h_up.size();
    n_twosided_  = (int)h_ts.size();

    // ── Phase 2: upload to device ─────────────────────────────────────────
    const bool use_dev = ref.UseDevice();

    // mfem::Array<int> has no UseDevice(bool) setter; access via GetMemory().
    auto UploadIdx = [&](mfem::Array<int>& arr,
                         const std::vector<int>& src) {
        arr.SetSize((int)src.size());
        arr.GetMemory().UseDevice(use_dev);
        if (!src.empty()) {
            int* h = arr.HostWrite();
            std::copy(src.begin(), src.end(), h);
            arr.Read();   // triggers H→D copy when use_dev == true
        }
    };

    auto UploadVec = [&](mfem::Vector& v,
                         const std::vector<mfem::real_t>& src) {
        v.SetSize((int)src.size());
        v.UseDevice(use_dev);
        if (!src.empty()) {
            mfem::real_t* h = v.HostWrite();
            std::copy(src.begin(), src.end(), h);
            v.Read();     // triggers H→D copy when use_dev == true
        }
    };

    UploadIdx(idx_unbounded_, h_ub);
    UploadIdx(idx_lower_,     h_lo);
    UploadIdx(idx_upper_,     h_up);
    UploadIdx(idx_twosided_,  h_ts);

    UploadVec(lo_lower_,    h_lo_lo);
    UploadVec(hi_upper_,    h_up_hi);
    UploadVec(lo_twosided_, h_ts_lo);
    UploadVec(hi_twosided_, h_ts_hi);
}

/**
 * @brief Reconstruct the full-length BoundType array from the packed data.
 *
 * Reads the four index arrays on the host and fills `types` so that
 * types[i] == BoundType of variable i.  Not called by the optimizer itself;
 * provided for external diagnostic use.
 */
void BoundPartition::GetTypes(std::vector<BoundType>& types) const
{
    const int n = Total();
    types.resize(n, BoundType::Unbounded);  // default: Unbounded

    auto Fill = [&](const mfem::Array<int>& arr, BoundType t) {
        const int* h = arr.HostRead();
        for (int k = 0; k < arr.Size(); ++k) types[h[k]] = t;
    };
    Fill(idx_lower_,    BoundType::LowerOnly);
    Fill(idx_upper_,    BoundType::UpperOnly);
    Fill(idx_twosided_, BoundType::TwoSided);
    // Unbounded entries remain at the default.
}


// ============================================================
//  ClassifyBounds (host-only utility)
//
//  Direct loop over lo/hi on the host; produces the same classification as
//  BoundPartition but without allocating device arrays.  Useful for
//  host-side initialization or for the convenience overloads.
// ============================================================
void ClassifyBounds(const mfem::Vector& lo,
                    const mfem::Vector& hi,
                    std::vector<BoundType>& types)
{
    const int n = lo.Size();
    types.resize(n);
    const mfem::real_t* ld = lo.HostRead();
    const mfem::real_t* ud = hi.HostRead();
    for (int i = 0; i < n; ++i) {
        const bool has_lo = !IsNegInf(ld[i]);
        const bool has_hi = !IsInf   (ud[i]);
        if      (!has_lo && !has_hi) types[i] = BoundType::Unbounded;
        else if ( has_lo && !has_hi) types[i] = BoundType::LowerOnly;
        else if (!has_lo &&  has_hi) types[i] = BoundType::UpperOnly;
        else                         types[i] = BoundType::TwoSided;
    }
}


// ============================================================
//  PrimalToLatent  (eq. 9) — four branch-free device kernels
//
//  Forward Legendre maps with boundary-safe argument clamping.
//  Safety strategy (see header for full explanation):
//
//  Unbounded : z = λ                        (no clamp needed)
//  LowerOnly : z = log(max(δ, λ−l))        δ = kPrimalMinGap
//  UpperOnly : z = −log(max(δ, u−λ))       δ = kPrimalMinGap
//  TwoSided  : z = log(gap_lo / gap_hi)
//                gap_lo = max(δ, λ−l)
//                gap_hi = max(δ, u−λ)
//                δ = max(kPrimalMinGap, kPrimalRelGap × (u−l))
//
//  For TwoSided the relative floor kPrimalRelGap×(u−l) ensures the gap
//  is never smaller than the floating-point noise in the bound values
//  themselves (relevant when (u−l) is large).
//
//  The clamps are branchless fmax — no warp divergence on GPU.
// ============================================================
void PrimalToLatent(const mfem::Vector& lam,
                    const BoundPartition& part,
                    mfem::Vector& z)
{
    const int  n       = lam.Size();
    const bool use_dev = lam.UseDevice();
    z.SetSize(n);
    z.UseDevice(use_dev);

    const mfem::real_t* lp = lam.Read();
    mfem::real_t*       zp = z.ReadWrite();  // ReadWrite: preserve entries
                                              // not touched by a kernel below

    // ── Kernel 1: Unbounded — z = λ ───────────────────────────────────────
    {
        const int  nc  = part.NumUnbounded();
        const int* idx = part.IdxUnbounded();
        mfem::forall_switch(use_dev, nc, [=] MFEM_HOST_DEVICE (int k) {
            zp[idx[k]] = lp[idx[k]];
        });
    }

    // ── Kernel 2: LowerOnly — z = log(max(δ, λ−l)) ───────────────────────
    {
        const int              nc  = part.NumLower();
        const int*             idx = part.IdxLower();
        const mfem::real_t*    li  = part.LoLower();
        mfem::forall_switch(use_dev, nc, [=] MFEM_HOST_DEVICE (int k) {
            const int        i   = idx[k];
            const mfem::real_t gap = ClampBelow(lp[i] - li[k], kPrimalMinGap);
            zp[i] = std::log(gap);
        });
    }

    // ── Kernel 3: UpperOnly — z = −log(max(δ, u−λ)) ──────────────────────
    {
        const int              nc  = part.NumUpper();
        const int*             idx = part.IdxUpper();
        const mfem::real_t*    ui  = part.HiUpper();
        mfem::forall_switch(use_dev, nc, [=] MFEM_HOST_DEVICE (int k) {
            const int        i   = idx[k];
            const mfem::real_t gap = ClampBelow(ui[k] - lp[i], kPrimalMinGap);
            zp[i] = -std::log(gap);
        });
    }

    // ── Kernel 4: TwoSided — z = log(gap_lo / gap_hi) ────────────────────
    // Both gap_lo and gap_hi are clamped independently so z stays finite
    // at either bound.  The relative floor kPrimalRelGap×width prevents
    // the clamp from activating spuriously for narrow intervals.
    {
        const int              nc  = part.NumTwoSided();
        const int*             idx = part.IdxTwoSided();
        const mfem::real_t*    li  = part.LoTwoSided();
        const mfem::real_t*    ui  = part.HiTwoSided();
        mfem::forall_switch(use_dev, nc, [=] MFEM_HOST_DEVICE (int k) {
            const int        i     = idx[k];
            const mfem::real_t width = ui[k] - li[k];
            const mfem::real_t delta = ClampBelow(kPrimalRelGap * width,
                                                   kPrimalMinGap);
            const mfem::real_t gap_lo = ClampBelow(lp[i] - li[k], delta);
            const mfem::real_t gap_hi = ClampBelow(ui[k] - lp[i], delta);
            zp[i] = std::log(gap_lo / gap_hi);
        });
    }
}


// ============================================================
//  LatentToPrimal  (eqs. 10–12) — four branch-free device kernels
//
//  Inverse Legendre maps T : ℝⁿ → int(K):
//  Unbounded : λ = z
//  LowerOnly : λ = l + exp(z)
//  UpperOnly : λ = u − exp(−z)
//  TwoSided  : λ = l + (u−l) σ(z)     σ = numerically-stable sigmoid
//
//  Strict feasibility lᵢ < λᵢ < uᵢ is guaranteed for every finite zᵢ.
// ============================================================
void LatentToPrimal(const mfem::Vector& z,
                    const BoundPartition& part,
                    mfem::Vector& lam)
{
    const int  n       = z.Size();
    const bool use_dev = z.UseDevice();
    lam.SetSize(n);
    lam.UseDevice(use_dev);

    const mfem::real_t* zp = z.Read();
    mfem::real_t*       lp = lam.ReadWrite();

    // ── Kernel 1: Unbounded — λ = z ───────────────────────────────────────
    {
        const int  nc  = part.NumUnbounded();
        const int* idx = part.IdxUnbounded();
        mfem::forall_switch(use_dev, nc, [=] MFEM_HOST_DEVICE (int k) {
            const int i = idx[k];
            lp[i] = zp[i];
        });
    }

    // ── Kernel 2: LowerOnly — λ = l + exp(z) ─────────────────────────────
    {
        const int              nc  = part.NumLower();
        const int*             idx = part.IdxLower();
        const mfem::real_t*    li  = part.LoLower();
        mfem::forall_switch(use_dev, nc, [=] MFEM_HOST_DEVICE (int k) {
            const int i = idx[k];
            lp[i] = li[k] + std::exp(zp[i]);
        });
    }

    // ── Kernel 3: UpperOnly — λ = u − exp(−z) ────────────────────────────
    {
        const int              nc  = part.NumUpper();
        const int*             idx = part.IdxUpper();
        const mfem::real_t*    ui  = part.HiUpper();
        mfem::forall_switch(use_dev, nc, [=] MFEM_HOST_DEVICE (int k) {
            const int i = idx[k];
            lp[i] = ui[k] - std::exp(-zp[i]);
        });
    }

    // ── Kernel 4: TwoSided — λ = l + (u−l) σ(z) ─────────────────────────
    // detail::Sigmoid() is numerically stable: no overflow for |z| ≤ 700.
    {
        const int              nc  = part.NumTwoSided();
        const int*             idx = part.IdxTwoSided();
        const mfem::real_t*    li  = part.LoTwoSided();
        const mfem::real_t*    ui  = part.HiTwoSided();
        mfem::forall_switch(use_dev, nc, [=] MFEM_HOST_DEVICE (int k) {
            const int i = idx[k];
            lp[i] = li[k] + (ui[k] - li[k]) * detail::Sigmoid(zp[i]);
        });
    }
}


// ============================================================
//  LatentJacobianDiag — four branch-free device kernels
//
//  Diagonal entries of JT(z) = dλ/dz derived from the chain rule applied
//  to eqs. (10)–(12).  Used in the descent-direction analysis (eqs. 43–44)
//  and optionally for preconditioning.  All entries are strictly positive.
//
//  Unbounded : JT = 1
//  LowerOnly : JT = exp(z)         = λ − l  (equals the forward-map gap)
//  UpperOnly : JT = exp(−z)        = u − λ
//  TwoSided  : JT = (u−l) σ(z)(1−σ(z))  (derivative of the sigmoid formula)
// ============================================================
void LatentJacobianDiag(const mfem::Vector& z,
                        const BoundPartition& part,
                        mfem::Vector& diag)
{
    const int  n       = z.Size();
    const bool use_dev = z.UseDevice();
    diag.SetSize(n);
    diag.UseDevice(use_dev);

    const mfem::real_t* zp = z.Read();
    mfem::real_t*       dp = diag.ReadWrite();

    // Unbounded: JT = 1
    {
        const int  nc  = part.NumUnbounded();
        const int* idx = part.IdxUnbounded();
        mfem::forall_switch(use_dev, nc, [=] MFEM_HOST_DEVICE (int k) {
            dp[idx[k]] = mfem::real_t(1);
        });
    }
    // LowerOnly: JT = exp(z)
    {
        const int  nc  = part.NumLower();
        const int* idx = part.IdxLower();
        mfem::forall_switch(use_dev, nc, [=] MFEM_HOST_DEVICE (int k) {
            const int i = idx[k];
            dp[i] = std::exp(zp[i]);
        });
    }
    // UpperOnly: JT = exp(−z)
    {
        const int  nc  = part.NumUpper();
        const int* idx = part.IdxUpper();
        mfem::forall_switch(use_dev, nc, [=] MFEM_HOST_DEVICE (int k) {
            const int i = idx[k];
            dp[i] = std::exp(-zp[i]);
        });
    }
    // TwoSided: JT = (u−l) σ(z)(1−σ(z))
    {
        const int              nc  = part.NumTwoSided();
        const int*             idx = part.IdxTwoSided();
        const mfem::real_t*    li  = part.LoTwoSided();
        const mfem::real_t*    ui  = part.HiTwoSided();
        mfem::forall_switch(use_dev, nc, [=] MFEM_HOST_DEVICE (int k) {
            const int i = idx[k];
            const mfem::real_t s = detail::Sigmoid(zp[i]);
            dp[i] = (ui[k] - li[k]) * s * (mfem::real_t(1) - s);
        });
    }
}


// ============================================================
//  DefaultPrimalInit (eq. 35) — four branch-free device kernels
//
//  Canonical interior starting points:
//  Unbounded : λ = 0
//  LowerOnly : λ = l + 1
//  UpperOnly : λ = u − 1
//  TwoSided  : λ = (l + u) / 2
//
//  The result is strictly feasible for any interval with width > 2 (for
//  LowerOnly: trivially, since u = +∞; for TwoSided: provided u−l > 2).
//  For narrower TwoSided intervals the caller should override with a manual
//  initialization.
// ============================================================
void DefaultPrimalInit(const BoundPartition& part,
                       const mfem::Vector& lo,
                       const mfem::Vector& hi,
                       mfem::Vector& lam)
{
    const int  n       = lo.Size();
    const bool use_dev = lo.UseDevice();
    lam.SetSize(n);
    lam.UseDevice(use_dev);

    mfem::real_t* lp = lam.ReadWrite();

    // Unbounded: λ = 0
    {
        const int  nc  = part.NumUnbounded();
        const int* idx = part.IdxUnbounded();
        mfem::forall_switch(use_dev, nc, [=] MFEM_HOST_DEVICE (int k) {
            lp[idx[k]] = mfem::real_t(0);
        });
    }
    // LowerOnly: λ = l + 1
    {
        const int              nc  = part.NumLower();
        const int*             idx = part.IdxLower();
        const mfem::real_t*    li  = part.LoLower();
        mfem::forall_switch(use_dev, nc, [=] MFEM_HOST_DEVICE (int k) {
            lp[idx[k]] = li[k] + mfem::real_t(1);
        });
    }
    // UpperOnly: λ = u − 1
    {
        const int              nc  = part.NumUpper();
        const int*             idx = part.IdxUpper();
        const mfem::real_t*    ui  = part.HiUpper();
        mfem::forall_switch(use_dev, nc, [=] MFEM_HOST_DEVICE (int k) {
            lp[idx[k]] = ui[k] - mfem::real_t(1);
        });
    }
    // TwoSided: λ = (l + u) / 2
    {
        const int              nc  = part.NumTwoSided();
        const int*             idx = part.IdxTwoSided();
        const mfem::real_t*    li  = part.LoTwoSided();
        const mfem::real_t*    ui  = part.HiTwoSided();
        mfem::forall_switch(use_dev, nc, [=] MFEM_HOST_DEVICE (int k) {
            lp[idx[k]] = mfem::real_t(0.5) * (li[k] + ui[k]);
        });
    }
}


// ============================================================
//  Convenience overloads — host-only, accept std::vector<BoundType>
//
//  Implement the same forward/inverse maps and clamping strategy as the
//  BoundPartition overloads but iterate on the host.  Suitable for
//  initialization, testing, and debugging where a BoundPartition has not
//  been constructed yet.
// ============================================================

/// @brief λ → z, host-only (same clamping as the device overload).
void PrimalToLatent(const mfem::Vector& lam,
                    const mfem::Vector& lo,
                    const mfem::Vector& hi,
                    const std::vector<BoundType>& types,
                    mfem::Vector& z)
{
    const int n = lam.Size();
    z.SetSize(n);
    const mfem::real_t* lp = lam.HostRead();
    const mfem::real_t* ld = lo .HostRead();
    const mfem::real_t* ud = hi .HostRead();
    mfem::real_t*       zp = z  .HostWrite();
    for (int i = 0; i < n; ++i) {
        switch (types[i]) {
        case BoundType::Unbounded:
            zp[i] = lp[i];
            break;
        case BoundType::LowerOnly: {
            const mfem::real_t gap = ClampBelow(lp[i] - ld[i], kPrimalMinGap);
            zp[i] = std::log(gap);
            break;
        }
        case BoundType::UpperOnly: {
            const mfem::real_t gap = ClampBelow(ud[i] - lp[i], kPrimalMinGap);
            zp[i] = -std::log(gap);
            break;
        }
        case BoundType::TwoSided: {
            const mfem::real_t width = ud[i] - ld[i];
            const mfem::real_t delta = ClampBelow(kPrimalRelGap * width,
                                                   kPrimalMinGap);
            const mfem::real_t gap_lo = ClampBelow(lp[i] - ld[i], delta);
            const mfem::real_t gap_hi = ClampBelow(ud[i] - lp[i], delta);
            zp[i] = std::log(gap_lo / gap_hi);
            break;
        }
        }
    }
}

/// @brief z → λ, host-only.
void LatentToPrimal(const mfem::Vector& z,
                    const mfem::Vector& lo,
                    const mfem::Vector& hi,
                    const std::vector<BoundType>& types,
                    mfem::Vector& lam)
{
    const int n = z.Size();
    lam.SetSize(n);
    const mfem::real_t* zp = z  .HostRead();
    const mfem::real_t* ld = lo .HostRead();
    const mfem::real_t* ud = hi .HostRead();
    mfem::real_t*       lp = lam.HostWrite();
    for (int i = 0; i < n; ++i) {
        switch (types[i]) {
        case BoundType::Unbounded: lp[i] = zp[i];                                         break;
        case BoundType::LowerOnly: lp[i] = ld[i]+std::exp(zp[i]);                         break;
        case BoundType::UpperOnly: lp[i] = ud[i]-std::exp(-zp[i]);                        break;
        case BoundType::TwoSided:  lp[i] = ld[i]+(ud[i]-ld[i])*detail::Sigmoid(zp[i]);   break;
        }
    }
}

/// @brief Default strictly-feasible initialization, host-only (eq. 35).
void DefaultPrimalInit(const mfem::Vector& lo,
                       const mfem::Vector& hi,
                       const std::vector<BoundType>& types,
                       mfem::Vector& lam)
{
    const int n = lo.Size();
    lam.SetSize(n);
    const mfem::real_t* ld = lo .HostRead();
    const mfem::real_t* ud = hi .HostRead();
    mfem::real_t*       lp = lam.HostWrite();
    for (int i = 0; i < n; ++i) {
        switch (types[i]) {
        case BoundType::Unbounded: lp[i] = mfem::real_t(0);                   break;
        case BoundType::LowerOnly: lp[i] = ld[i]+mfem::real_t(1);             break;
        case BoundType::UpperOnly: lp[i] = ud[i]-mfem::real_t(1);             break;
        case BoundType::TwoSided:  lp[i] = mfem::real_t(0.5)*(ld[i]+ud[i]);  break;
        }
    }
}


// ============================================================
//  BoundSafetyCheck (host-side diagnostic)
//
//  Reads lam on the host and counts entries where λᵢ ≤ lᵢ + tol or
//  λᵢ ≥ uᵢ − tol for the relevant bound types.  Returns the total count.
//  Does not modify any vector.  Unbounded variables are never counted.
// ============================================================
int BoundSafetyCheck(const mfem::Vector& lam,
                     const BoundPartition& part,
                     const mfem::Vector& lo,
                     const mfem::Vector& hi,
                     mfem::real_t tol)
{
    // Bring lam to host regardless of its current device location.
    const mfem::real_t* lp = lam.HostRead();
    const mfem::real_t* ld = lo .HostRead();
    const mfem::real_t* ud = hi .HostRead();

    int count = 0;

    // LowerOnly: flag if λᵢ ≤ lᵢ + tol
    {
        const int* idx = part.IdxLower();
        for (int k = 0; k < part.NumLower(); ++k) {
            const int i = idx[k];
            if (lp[i] <= ld[i] + tol) ++count;
        }
    }
    // UpperOnly: flag if λᵢ ≥ uᵢ − tol
    {
        const int* idx = part.IdxUpper();
        for (int k = 0; k < part.NumUpper(); ++k) {
            const int i = idx[k];
            if (lp[i] >= ud[i] - tol) ++count;
        }
    }
    // TwoSided: flag either side independently
    {
        const int* idx = part.IdxTwoSided();
        for (int k = 0; k < part.NumTwoSided(); ++k) {
            const int i = idx[k];
            if (lp[i] <= ld[i] + tol) ++count;
            if (lp[i] >= ud[i] - tol) ++count;
        }
    }
    return count;
}


// ============================================================
//  Internal device kernels  (namespace detail)
//
//  ClipLatentKernel       : element-wise clip z to [−zmax, zmax].
//  ProjGradResidualKernel : projected-gradient stationarity residual helper.
// ============================================================
namespace detail {

/**
 * @brief Element-wise latent clipping kernel.
 *
 * Clips all entries of z to [−zmax, zmax] in-place using a branchless
 * ternary (maps to FMIN/FMAX on GPU).  Launched via forall_switch so it
 * runs on the same device as z.
 *
 * This clipping is applied after every accepted gradient step to prevent
 * sigmoid saturation.  The default zmax = 40 ensures σ(±40) ≈ 1 − 4×10⁻¹⁸
 * in double precision, which is numerically indistinguishable from the bound.
 */
static void ClipLatentKernel(mfem::Vector& z, mfem::real_t zmax, bool use_dev)
{
    const int     n  = z.Size();
    mfem::real_t* zp = z.ReadWrite();
    mfem::forall_switch(use_dev, n, [=] MFEM_HOST_DEVICE (int i) {
        // Branchless: compiles to FMIN + FMAX on GPU, no divergence.
        zp[i] = zp[i] >  zmax ?  zmax :
                zp[i] < -zmax ? -zmax : zp[i];
    });
}

/**
 * @brief Projected-gradient stationarity residual kernel (eq. 33).
 *
 * For each variable i computes:
 *   yᵢ = Π_K(λᵢ − dᵢ)    (component-wise clipping to [lᵢ, uᵢ])
 *   rᵢ = λᵢ − yᵢ
 *
 * Fills two temporary device vectors r2[i] = rᵢ² and l2[i] = λᵢ²,
 * then computes their global sums via Vector::Sum() (dispatches to
 * cuBLAS cublasDasum on GPU).
 *
 * The stationarity residual is  ‖r‖₂ / max(1, ‖λ‖₂)  (eq. 33).
 *
 * @param n        Number of variables.
 * @param use_dev  True to run on device.
 * @param lam      Primal variable λ (device pointer).
 * @param d        Euclidean derivative dᵏ (device pointer).
 * @param lo       Lower bounds (device pointer; may be −∞).
 * @param hi       Upper bounds (device pointer; may be +∞).
 * @param sum_r2   Output: Σ rᵢ².
 * @param sum_lam2 Output: Σ λᵢ².
 */
static void ProjGradResidualKernel(
    int n, bool use_dev,
    const mfem::real_t* lam, const mfem::real_t* d,
    const mfem::real_t* lo,  const mfem::real_t* hi,
    double& sum_r2, double& sum_lam2)
{
    mfem::Vector r2(n), l2(n);
    r2.UseDevice(use_dev);
    l2.UseDevice(use_dev);
    mfem::real_t* r2p = r2.Write();
    mfem::real_t* l2p = l2.Write();
    mfem::forall_switch(use_dev, n, [=] MFEM_HOST_DEVICE (int i) {
        // Π_K(λ − d): clip λᵢ − dᵢ into [lᵢ, uᵢ] branchlessly.
        mfem::real_t y = lam[i] - d[i];
        if (!IsNegInf(lo[i])) y = y < lo[i] ? lo[i] : y;
        if (!IsInf   (hi[i])) y = y > hi[i] ? hi[i] : y;
        const mfem::real_t ri = lam[i] - y;
        r2p[i] = ri    * ri;
        l2p[i] = lam[i] * lam[i];
    });
    // Sum via cuBLAS on GPU, plain loop on CPU.
    sum_r2   = double(r2.Sum());
    sum_lam2 = double(l2.Sum());
}

} // namespace detail


// ============================================================
//                  LatentMirrorOptimizer (serial)
// ============================================================

// ── Private helper implementations ────────────────────────────────────────

/**
 * @brief Deferred device initialization — called once on first Update().
 *
 * Allocates and zero-initializes all internal work vectors (g_, Mg_,
 * z_trial_, etc.) on the same device as @p ref.  Also migrates the full
 * lo/hi vectors to device so the stationarity residual kernel can read them.
 */
void LatentMirrorOptimizer::InitDeviceVectors(const mfem::Vector& ref)
{
    DeviceInit(z_prev_,    n_, ref);
    DeviceInit(lam_prev_,  n_, ref);
    DeviceInit(g_prev_,    n_, ref);
    DeviceInit(g_,         n_, ref);
    DeviceInit(Mg_,        n_, ref);
    DeviceInit(z_trial_,   n_, ref);
    DeviceInit(lam_cur_,   n_, ref);
    DeviceInit(lam_trial_, n_, ref);

    // Migrate full lo/hi bounds to device for the residual kernel.
    lo_full_.UseDevice(ref.UseDevice());
    hi_full_.UseDevice(ref.UseDevice());
    lo_full_.Read();   // triggers H→D if use_dev
    hi_full_.Read();
}

/// Apply M: Mx = M_op * x, or Mx = x when M_op is null (identity).
void LatentMirrorOptimizer::ApplyM(const mfem::Vector& x,
                                    mfem::Vector& Mx) const
{
    if (M_op_) { Mx.SetSize(x.Size()); M_op_->Mult(x, Mx); }
    else        { Mx = x; }
}

/// Apply M⁻¹: Minvx = M_solver * x, or Minvx = x when null.
void LatentMirrorOptimizer::ApplyMinv(const mfem::Vector& x,
                                       mfem::Vector& Minvx) const
{
    if (M_solver_) { Minvx.SetSize(x.Size()); M_solver_->Mult(x, Minvx); }
    else            { Minvx = x; }
}

/**
 * @brief Compute ⟨x, y⟩_M = xᵀ M y.
 *
 * When M_op_ is non-null: allocates My, applies M, and calls
 * mfem::InnerProduct(x, My) which dispatches to cuBLAS on GPU.
 * When null: calls mfem::InnerProduct(x, y) directly.
 */
mfem::real_t LatentMirrorOptimizer::InnerProductM(
        const mfem::Vector& x, const mfem::Vector& y) const
{
    if (M_op_) {
        mfem::Vector My(y.Size());
        My.UseDevice(y.UseDevice());
        M_op_->Mult(y, My);
        return mfem::InnerProduct(x, My);
    }
    return mfem::InnerProduct(x, y);
}

/**
 * @brief Compute the projected-gradient residual (eq. 33).
 *
 * Delegates to detail::ProjGradResidualKernel, which runs on the same
 * device as lam.
 */
mfem::real_t LatentMirrorOptimizer::ProjectedGradResidual(
        const mfem::Vector& lam, const mfem::Vector& d) const
{
    const bool use_dev = lam.UseDevice();
    double sr2, sl2;
    detail::ProjGradResidualKernel(n_, use_dev,
        lam.Read(), d.Read(), lo_full_.Read(), hi_full_.Read(),
        sr2, sl2);
    return mfem::real_t(std::sqrt(sr2) / std::max(1.0, std::sqrt(sl2)));
}

/// Element-wise clip z to [−zmax_, zmax_] using detail::ClipLatentKernel.
void LatentMirrorOptimizer::ClipLatent(mfem::Vector& z) const
{
    detail::ClipLatentKernel(z, zmax_, z.UseDevice());
}

/**
 * @brief Compute the GBB trial step size (eqs. 18 + 21).
 *
 * @f[
 *   \alpha_{\text{GBB}} = \frac{\langle \Delta z, \Delta\lambda \rangle_M}
 *                               {|\langle \Delta g, \Delta\lambda \rangle_M|}
 * @f]
 * clamped to [alpha_min_, alpha_max_] and combined with alpha_prev_ via
 * geometric mean:
 * @f[
 *   \alpha_0 = \sqrt{\alpha_{\text{GBB}} \cdot \alpha_{\text{prev}}}
 * @f]
 * Falls back to @p fallback when the GBB numerator or denominator is below
 * eps_bb_ in magnitude (indicating no reliable curvature information).
 */
mfem::real_t LatentMirrorOptimizer::ComputeGBBStepSize(
        const mfem::Vector& dz,   const mfem::Vector& dlam,
        const mfem::Vector& dg,   mfem::real_t fallback) const
{
    const mfem::real_t ak = InnerProductM(dz,  dlam);  // numerator   (eq. 18)
    const mfem::real_t bk = InnerProductM(dg,  dlam);  // denominator (eq. 18)
    mfem::real_t ag = (ak > eps_bb_ && std::abs(bk) > eps_bb_)
                        ? ak / std::abs(bk) : fallback;
    ag = std::max(alpha_min_, std::min(alpha_max_, ag));  // safeguard clamp
    return std::sqrt(ag * alpha_prev_);                   // geometric mean (eq. 21)
}

// ── Constructors ──────────────────────────────────────────────────────────

/// Delegating constructor: identity M.
LatentMirrorOptimizer::LatentMirrorOptimizer(
        const mfem::Vector& z0,
        const mfem::Vector& lo,
        const mfem::Vector& hi)
    : LatentMirrorOptimizer(z0, lo, hi, nullptr, nullptr) {}

/**
 * @brief Primary constructor.
 *
 * Builds the BoundPartition from lo/hi, stores M_op and M_solver, and
 * sets all algorithm parameters to their defaults.  Internal work vectors
 * are NOT allocated here; they are set up lazily in InitDeviceVectors() at
 * the first Update() call so the device flag can be inherited from z.
 */
LatentMirrorOptimizer::LatentMirrorOptimizer(
        const mfem::Vector& z0,
        const mfem::Vector& lo,
        const mfem::Vector& hi,
        mfem::Operator* M_op,
        mfem::Solver*   M_solver)
    : n_(z0.Size()), part_(lo, hi, z0),
      lo_full_(lo), hi_full_(hi),
      M_op_(M_op), M_solver_(M_solver),
      z_prev_(n_), lam_prev_(n_), g_prev_(n_),
      g_(n_), Mg_(n_), z_trial_(n_), lam_cur_(n_), lam_trial_(n_),
      initialized_(false),
      alpha_prev_(mfem::real_t(1)),
      alpha_min_(mfem::real_t(1e-12)), alpha_max_(mfem::real_t(1e4)),
      eps_bb_(mfem::real_t(1e-14)),
      c1_(mfem::real_t(1e-4)), beta_(mfem::real_t(0.5)), max_ls_(50),
      zmax_(mfem::real_t(40)), stat_tol_(mfem::real_t(1e-6)),
      iter_(0), last_ls_steps_(0),
      last_stat_res_(mfem::real_t(-1)), last_rel_step_(mfem::real_t(-1))
{
    assert(lo.Size() == n_ && hi.Size() == n_);
}

// ── Configuration setters ─────────────────────────────────────────────────

void LatentMirrorOptimizer::SetLineSearchParams(
        mfem::real_t c1, mfem::real_t beta, int max_ls)
{ c1_ = c1; beta_ = beta; max_ls_ = max_ls; }

void LatentMirrorOptimizer::SetStepSizeSafeguards(
        mfem::real_t alpha_min, mfem::real_t alpha_max, mfem::real_t eps_bb)
{ alpha_min_ = alpha_min; alpha_max_ = alpha_max; eps_bb_ = eps_bb; }

void LatentMirrorOptimizer::SetLatentClipping(mfem::real_t zmax)
{ zmax_ = zmax; }

void LatentMirrorOptimizer::SetStationarityTol(mfem::real_t tol)
{ stat_tol_ = tol; }

// ── Update  (Algorithm §8) ────────────────────────────────────────────────
/**
 * @brief One complete mirror-gradient step.
 *
 * Steps follow §8 of the paper:
 *
 * Step 1: Recover current primal  λᵏ = T(zᵏ)  (four branch-free kernels).
 *
 * Step 2: Compute M-gradient  gᵏ = M⁻¹ dᵏ  and pre-compute
 *         Mg_ = M gᵏ = dᵏ  for the Armijo pairing.
 *         (When M = I: Mg_ = g_ = d with no extra work.)
 *
 * Step 4: Compute trial step size.
 *   k = 0: α₀ = 1 / max(1, ‖g⁰‖∞)  (eq. 22; g_.Normlinf on device).
 *   k > 0: αGBB via eq. (18), clamped and combined with α_{k-1} via
 *           geometric mean (eq. 21).
 *         History vectors (z_prev_, lam_prev_, g_prev_) are saved BEFORE
 *         the line search so they hold zᵏ, λᵏ, gᵏ for the next GBB call.
 *
 * Step 5: Armijo backtracking line search.
 *   For ls = 0, 1, ..., max_ls−1:
 *     z⁺ = zᵏ − α gᵏ  (device AXPY)
 *     clip z⁺ to [−zmax_, zmax_]
 *     λ⁺ = T(z⁺)       (four branch-free kernels)
 *     if eval_phi == null: accept immediately
 *     pairing = (dᵏ)ᵀ(λ⁺ − λᵏ)  (device dot via mfem::InnerProduct(Mg_, Δλ))
 *       Note: pairing ≤ 0 for a genuine descent direction (eq. 44).
 *     if Φ(λ⁺) ≤ Φ(λᵏ) + c₁ × pairing: accept
 *     else: α ← β α, retry
 *
 * Step 6: Accept.  α_prev_ ← α;  z ← z⁺  (user's latent vector updated).
 *
 * Post-step diagnostics:
 *   - Stationarity residual (eq. 33) at the new zᵏ⁺¹.
 *   - Relative M-norm step size (eq. 34).
 */
mfem::real_t LatentMirrorOptimizer::Update(
        mfem::Vector&       z,
        const mfem::Vector& d,
        mfem::real_t        phi_k,
        EvalPhi             eval_phi)
{
    const bool use_dev = z.UseDevice();

    // ── Deferred device initialization (first call only) ──────────────────
    if (!initialized_) {
        InitDeviceVectors(z);
        z_prev_ = z;                           // seed GBB history with z₀
        LatentToPrimal(z, part_, lam_prev_);   // seed λ history with T(z₀)
        g_prev_ = mfem::real_t(0);             // zero: GBB skipped on k=0
        initialized_ = true;
    }

    // ── Step 1: current primal ────────────────────────────────────────────
    LatentToPrimal(z, part_, lam_cur_);

    // ── Step 2: M-gradient and Armijo pre-computation ─────────────────────
    ApplyMinv(d, g_);    // gᵏ = M⁻¹ dᵏ
    ApplyM(g_, Mg_);     // Mg_ = M gᵏ = dᵏ (used in pairing: (dᵏ)ᵀΔλ)

    // ── Step 4: trial step size ───────────────────────────────────────────
    mfem::real_t alpha_0;
    if (iter_ == 0) {
        alpha_0 = mfem::real_t(1) /
                  std::max(mfem::real_t(1), g_.Normlinf());  // eq. 22
        alpha_prev_ = alpha_0;
    } else {
        // Latent and primal displacements for GBB (all on device).
        mfem::Vector dz(n_), dlam(n_), dg(n_);
        dz.UseDevice(use_dev);
        dlam.UseDevice(use_dev);
        dg.UseDevice(use_dev);
        subtract(z,        z_prev_,   dz);    // Δzᵏ   = zᵏ − zᵏ⁻¹
        subtract(lam_cur_, lam_prev_, dlam);  // Δλᵏ   = λᵏ − λᵏ⁻¹
        subtract(g_,       g_prev_,   dg);    // Δgᵏ   = gᵏ − gᵏ⁻¹
        alpha_0 = ComputeGBBStepSize(dz, dlam, dg, alpha_prev_);  // eqs. 18, 21
    }

    // Save history BEFORE the line search moves lam_cur_ and z.
    z_prev_   = z;
    lam_prev_ = lam_cur_;
    g_prev_   = g_;

    // ── Step 5: Armijo backtracking ───────────────────────────────────────
    // Armijo condition (eq. 24):
    //   Φ(λ⁺) ≤ Φ(λᵏ) + c₁ (dᵏ)ᵀ(λ⁺ − λᵏ)
    //          = Φ(λᵏ) + c₁ ⟨gᵏ, λ⁺ − λᵏ⟩_M  (since Mg_ = dᵏ)
    mfem::real_t alpha = alpha_0;
    int ls_steps = 0;

    mfem::Vector dlam(n_);
    dlam.UseDevice(use_dev);

    for (int ls = 0; ls < max_ls_; ++ls) {
        // z⁺ = zᵏ − α gᵏ  (eq. 23 / eq. 13; zᵏ is preserved in z_prev_)
        z_trial_ = z;                     // copy current zᵏ
        z_trial_.Add(-alpha, g_);         // latent gradient step
        ClipLatent(z_trial_);             // keep z in [−zmax_, zmax_]

        // λ⁺ = T(z⁺)  (eq. 14; four branch-free kernels)
        LatentToPrimal(z_trial_, part_, lam_trial_);

        if (!eval_phi) break;   // no Armijo check: accept trial step directly

        // Armijo descent pairing (dᵏ)ᵀ(λ⁺ − λᵏ).
        // Uses Mg_ = M gᵏ = dᵏ (pre-computed in Step 2).
        subtract(lam_trial_, lam_cur_, dlam);
        const mfem::real_t pairing = mfem::InnerProduct(Mg_, dlam);

        // Evaluate Φ(T(z⁺)) via caller-supplied callback.
        mfem::real_t phi_trial = mfem::real_t(0);
        eval_phi(z_trial_, phi_trial);

        if (phi_trial <= phi_k + c1_ * pairing) break;  // Armijo satisfied

        alpha *= beta_;   // reduce step and retry
        ++ls_steps;
    }
    last_ls_steps_ = ls_steps;

    // ── Step 6: accept  (eq. 26) ──────────────────────────────────────────
    alpha_prev_ = alpha;
    z = z_trial_;    // update user's latent vector in-place

    // ── Post-step diagnostics ─────────────────────────────────────────────

    // Stationarity residual at zᵏ⁺¹ (eq. 33).
    LatentToPrimal(z, part_, lam_cur_);       // λᵏ⁺¹ = T(zᵏ⁺¹)
    last_stat_res_ = ProjectedGradResidual(lam_cur_, d);

    // Relative M-norm step size (eq. 34): ‖λᵏ⁺¹ − λᵏ‖_M / max(1, ‖λᵏ‖_M).
    {
        mfem::Vector step(n_);
        step.UseDevice(use_dev);
        subtract(lam_cur_, lam_prev_, step);  // λᵏ⁺¹ − λᵏ
        last_rel_step_ = NormM(step) /
                         std::max(mfem::real_t(1), NormM(lam_prev_));
    }

    ++iter_;
    return alpha;
}

/**
 * @brief Standalone stationarity residual (not cached).
 *
 * Converts z to primal via T(z) and evaluates eq. (33).
 * More expensive than StationarityResidual() (requires a LatentToPrimal
 * call) but works at any point, not just immediately after Update().
 */
mfem::real_t LatentMirrorOptimizer::StationarityResidual(
        const mfem::Vector& z, const mfem::Vector& d) const
{
    mfem::Vector lam(n_);
    lam.UseDevice(z.UseDevice());
    LatentToPrimal(z, part_, lam);
    return ProjectedGradResidual(lam, d);
}


// ============================================================
#ifdef MFEM_USE_MPI
//           LatentMirrorOptimizerParallel
//
//  The parallel implementation is structurally identical to the serial one.
//  The key differences are:
//
//  1. InnerProductM assembles the global dot product via MPI_Allreduce(SUM)
//     over local dot products.  The result is identical on all ranks, so
//     the GBB step size and all Armijo tests are rank-consistent.
//
//  2. The initial step size uses MPI_Allreduce(MAX) over local ‖g⁰‖∞.
//
//  3. The Armijo pairing (dᵏ)ᵀ(λ⁺ − λᵏ) is assembled via
//     MPI_Allreduce(SUM) of local dot products.
//
//  4. ProjectedGradResidualGlobal reduces two scalars (Σrᵢ², Σλᵢ²) via a
//     single MPI_Allreduce(SUM, 2 doubles).
//
//  5. The eval_phi callback must return the GLOBAL objective on all ranks.
// ============================================================

// ── Private helper implementations ────────────────────────────────────────

/// Allocate local work vectors on device (same logic as serial).
void LatentMirrorOptimizerParallel::InitDeviceVectors(const mfem::Vector& ref)
{
    DeviceInit(z_prev_local_,    n_local_, ref);
    DeviceInit(lam_prev_local_,  n_local_, ref);
    DeviceInit(g_prev_local_,    n_local_, ref);
    DeviceInit(g_local_,         n_local_, ref);
    DeviceInit(Mg_local_,        n_local_, ref);
    DeviceInit(z_trial_local_,   n_local_, ref);
    DeviceInit(lam_cur_local_,   n_local_, ref);
    DeviceInit(lam_trial_local_, n_local_, ref);

    lo_full_local_.UseDevice(ref.UseDevice());
    hi_full_local_.UseDevice(ref.UseDevice());
    lo_full_local_.Read();
    hi_full_local_.Read();
}

void LatentMirrorOptimizerParallel::ApplyM(const mfem::Vector& x,
                                            mfem::Vector& Mx) const
{
    if (M_op_) { Mx.SetSize(x.Size()); M_op_->Mult(x, Mx); }
    else        { Mx = x; }
}

void LatentMirrorOptimizerParallel::ApplyMinv(const mfem::Vector& x,
                                               mfem::Vector& Minvx) const
{
    if (M_solver_) { Minvx.SetSize(x.Size()); M_solver_->Mult(x, Minvx); }
    else            { Minvx = x; }
}

/**
 * @brief Global M-inner product: local dot (on device) + MPI_Allreduce(SUM).
 *
 * Each rank contributes its local dot xlocᵀ M_local y_local.  The global
 * sum is assembled via MPI_Allreduce and returned identically on all ranks.
 * mfem::InnerProduct dispatches to cuBLAS on GPU.
 */
mfem::real_t LatentMirrorOptimizerParallel::InnerProductM(
        const mfem::Vector& x, const mfem::Vector& y) const
{
    double local_val;
    if (M_op_) {
        mfem::Vector My(y.Size());
        My.UseDevice(y.UseDevice());
        M_op_->Mult(y, My);
        local_val = double(mfem::InnerProduct(x, My));
    } else {
        local_val = double(mfem::InnerProduct(x, y));
    }
    double global_val = 0.0;
    MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_SUM, comm_);
    return mfem::real_t(global_val);
}

void LatentMirrorOptimizerParallel::ClipLatent(mfem::Vector& z) const
{
    detail::ClipLatentKernel(z, zmax_, z.UseDevice());
}

/// Same formula as serial (eqs. 18 + 21); inner products already global.
mfem::real_t LatentMirrorOptimizerParallel::ComputeGBBStepSize(
        const mfem::Vector& dz,   const mfem::Vector& dlam,
        const mfem::Vector& dg,   mfem::real_t fallback) const
{
    const mfem::real_t ak = InnerProductM(dz,  dlam);
    const mfem::real_t bk = InnerProductM(dg,  dlam);
    mfem::real_t ag = (ak > eps_bb_ && std::abs(bk) > eps_bb_)
                        ? ak / std::abs(bk) : fallback;
    ag = std::max(alpha_min_, std::min(alpha_max_, ag));
    return std::sqrt(ag * alpha_prev_);
}

/**
 * @brief Global projected-gradient residual (eq. 33, parallel).
 *
 * Runs ProjGradResidualKernel locally (on device), then reduces
 * {sum_r², sum_λ²} via a single MPI_Allreduce(SUM) over two doubles.
 * The result is identical on all ranks.
 */
mfem::real_t LatentMirrorOptimizerParallel::ProjectedGradResidualGlobal(
        const mfem::Vector& lam_local, const mfem::Vector& d_local) const
{
    const bool use_dev = lam_local.UseDevice();
    double sr2_loc, sl2_loc;
    detail::ProjGradResidualKernel(n_local_, use_dev,
        lam_local.Read(), d_local.Read(),
        lo_full_local_.Read(), hi_full_local_.Read(),
        sr2_loc, sl2_loc);
    double buf[2]  = {sr2_loc, sl2_loc};
    double gbuf[2] = {0.0, 0.0};
    MPI_Allreduce(buf, gbuf, 2, MPI_DOUBLE, MPI_SUM, comm_);
    return mfem::real_t(std::sqrt(gbuf[0]) / std::max(1.0, std::sqrt(gbuf[1])));
}

// ── Constructors ──────────────────────────────────────────────────────────

LatentMirrorOptimizerParallel::LatentMirrorOptimizerParallel(
        MPI_Comm comm, const mfem::Vector& z0_local,
        const mfem::Vector& lo_local, const mfem::Vector& hi_local)
    : LatentMirrorOptimizerParallel(comm, z0_local, lo_local, hi_local,
                                    nullptr, nullptr) {}

LatentMirrorOptimizerParallel::LatentMirrorOptimizerParallel(
        MPI_Comm comm, const mfem::Vector& z0_local,
        const mfem::Vector& lo_local, const mfem::Vector& hi_local,
        mfem::Operator* M_op_local, mfem::Solver* M_solver_local)
    : comm_(comm), n_local_(z0_local.Size()),
      part_(lo_local, hi_local, z0_local),
      lo_full_local_(lo_local), hi_full_local_(hi_local),
      M_op_(M_op_local), M_solver_(M_solver_local),
      z_prev_local_(n_local_), lam_prev_local_(n_local_),
      g_prev_local_(n_local_),
      g_local_(n_local_), Mg_local_(n_local_),
      z_trial_local_(n_local_), lam_cur_local_(n_local_),
      lam_trial_local_(n_local_),
      initialized_(false),
      alpha_prev_(mfem::real_t(1)),
      alpha_min_(mfem::real_t(1e-12)), alpha_max_(mfem::real_t(1e4)),
      eps_bb_(mfem::real_t(1e-14)),
      c1_(mfem::real_t(1e-4)), beta_(mfem::real_t(0.5)), max_ls_(50),
      zmax_(mfem::real_t(40)), stat_tol_(mfem::real_t(1e-6)),
      iter_(0), last_ls_steps_(0),
      last_stat_res_(mfem::real_t(-1)), last_rel_step_(mfem::real_t(-1))
{}

// ── Configuration setters ─────────────────────────────────────────────────

void LatentMirrorOptimizerParallel::SetLineSearchParams(
        mfem::real_t c1, mfem::real_t beta, int max_ls)
{ c1_ = c1; beta_ = beta; max_ls_ = max_ls; }

void LatentMirrorOptimizerParallel::SetStepSizeSafeguards(
        mfem::real_t alpha_min, mfem::real_t alpha_max, mfem::real_t eps_bb)
{ alpha_min_ = alpha_min; alpha_max_ = alpha_max; eps_bb_ = eps_bb; }

void LatentMirrorOptimizerParallel::SetLatentClipping(mfem::real_t zmax)
{ zmax_ = zmax; }

void LatentMirrorOptimizerParallel::SetStationarityTol(mfem::real_t tol)
{ stat_tol_ = tol; }

// ── Update ────────────────────────────────────────────────────────────────
/**
 * @brief One parallel mirror-gradient step.
 *
 * Algorithm is identical to LatentMirrorOptimizer::Update().  The
 * differences are:
 *  - Initial ‖g⁰‖∞ assembled via MPI_Allreduce(MAX).
 *  - GBB inner products assembled via InnerProductM (calls Allreduce).
 *    alpha_0 is therefore identical on all ranks.
 *  - Armijo pairing (dᵏ)ᵀ(λ⁺ − λᵏ) assembled via MPI_Allreduce(SUM)
 *    of local dots.  All ranks test the same Armijo condition and advance
 *    or reduce step identically — no extra synchronization needed.
 *  - eval_phi callback must return the GLOBAL objective.
 *  - Stationarity residual via ProjectedGradResidualGlobal (Allreduce).
 */
mfem::real_t LatentMirrorOptimizerParallel::Update(
        mfem::Vector&       z_local,
        const mfem::Vector& d_local,
        mfem::real_t        phi_k,
        EvalPhi             eval_phi)
{
    const bool use_dev = z_local.UseDevice();

    // Deferred device initialization.
    if (!initialized_) {
        InitDeviceVectors(z_local);
        z_prev_local_ = z_local;
        LatentToPrimal(z_local, part_, lam_prev_local_);
        g_prev_local_ = mfem::real_t(0);
        initialized_ = true;
    }

    // Step 1: current primal.
    LatentToPrimal(z_local, part_, lam_cur_local_);

    // Step 2: M-gradient.
    ApplyMinv(d_local, g_local_);
    ApplyM(g_local_, Mg_local_);

    // Step 4: trial step size.
    mfem::real_t alpha_0;
    if (iter_ == 0) {
        // Global ‖g⁰‖∞ via MPI_Allreduce(MAX).
        double local_gnorm  = double(g_local_.Normlinf());
        double global_gnorm = 0.0;
        MPI_Allreduce(&local_gnorm, &global_gnorm, 1, MPI_DOUBLE, MPI_MAX, comm_);
        alpha_0 = mfem::real_t(1) /
                  std::max(mfem::real_t(1), mfem::real_t(global_gnorm));
        alpha_prev_ = alpha_0;
    } else {
        mfem::Vector dz(n_local_), dlam(n_local_), dg(n_local_);
        dz.UseDevice(use_dev);
        dlam.UseDevice(use_dev);
        dg.UseDevice(use_dev);
        subtract(z_local,        z_prev_local_,   dz);
        subtract(lam_cur_local_, lam_prev_local_,  dlam);
        subtract(g_local_,       g_prev_local_,    dg);
        // InnerProductM calls Allreduce → alpha_0 identical on all ranks.
        alpha_0 = ComputeGBBStepSize(dz, dlam, dg, alpha_prev_);
    }

    // Save history.
    z_prev_local_   = z_local;
    lam_prev_local_ = lam_cur_local_;
    g_prev_local_   = g_local_;

    // Step 5: Armijo backtracking.
    // All ranks run identical iterations because alpha_0 is global, the
    // Armijo pairing is globally reduced, and eval_phi returns a global value.
    mfem::real_t alpha = alpha_0;
    int ls_steps = 0;

    mfem::Vector dlam(n_local_);
    dlam.UseDevice(use_dev);

    for (int ls = 0; ls < max_ls_; ++ls) {
        z_trial_local_ = z_local;
        z_trial_local_.Add(-alpha, g_local_);
        ClipLatent(z_trial_local_);

        LatentToPrimal(z_trial_local_, part_, lam_trial_local_);

        if (!eval_phi) break;

        // Global Armijo pairing: local dot on device + Allreduce(SUM).
        subtract(lam_trial_local_, lam_cur_local_, dlam);
        double local_pair  = double(mfem::InnerProduct(Mg_local_, dlam));
        double global_pair = 0.0;
        MPI_Allreduce(&local_pair, &global_pair, 1, MPI_DOUBLE, MPI_SUM, comm_);
        const mfem::real_t pairing = mfem::real_t(global_pair);

        // Callback must return the global Φ(T(z⁺)).
        mfem::real_t phi_trial = mfem::real_t(0);
        eval_phi(z_trial_local_, phi_trial);

        if (phi_trial <= phi_k + c1_ * pairing) break;

        alpha *= beta_;
        ++ls_steps;
    }
    last_ls_steps_ = ls_steps;

    // Step 6: accept.
    alpha_prev_ = alpha;
    z_local = z_trial_local_;

    // Post-step diagnostics.
    LatentToPrimal(z_local, part_, lam_cur_local_);
    last_stat_res_ = ProjectedGradResidualGlobal(lam_cur_local_, d_local);

    {
        mfem::Vector step(n_local_);
        step.UseDevice(use_dev);
        subtract(lam_cur_local_, lam_prev_local_, step);
        last_rel_step_ = NormM(step) /
                         std::max(mfem::real_t(1), NormM(lam_prev_local_));
    }

    ++iter_;
    return alpha;
}

/// @brief Standalone global stationarity residual (parallel, eq. 33).
mfem::real_t LatentMirrorOptimizerParallel::StationarityResidual(
        const mfem::Vector& z_local, const mfem::Vector& d_local) const
{
    mfem::Vector lam(n_local_);
    lam.UseDevice(z_local.UseDevice());
    LatentToPrimal(z_local, part_, lam);
    return ProjectedGradResidualGlobal(lam, d_local);
}

#endif // MFEM_USE_MPI

} // namespace mfem_lmg
