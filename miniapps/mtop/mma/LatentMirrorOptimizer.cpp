/**
 * @file LatentMirrorOptimizer.cpp
 * @brief Device-aware implementation of the latent-variable mirror-gradient
 *        optimizer.
 *
 * Device strategy (identical to MMA_MFEM.cpp convention):
 *   - use_dev = z.UseDevice()  read at every Update() call.
 *   - All O(n) loops → mfem::forall_switch(use_dev, n, lambda).
 *   - Device data accessed only via Read()/Write()/ReadWrite().
 *   - HostRead()/HostWrite() used only for scalars and the m×m dual system.
 *
 * GPU branching strategy — BoundPartition:
 *   Instead of a single kernel with switch(BoundType[i]) (which causes warp
 *   divergence), we build compact index arrays (one per bound type) at
 *   construction time and launch four separate branch-free kernels, each
 *   operating only on the variables of one type.  The index arrays live on
 *   device alongside compact bound-value arrays so each kernel reads fully
 *   coalesced, contiguous memory.
 *
 * All equations referenced below are from:
 *   "A Latent-Variable Mirror-Gradient Algorithm for Bound-Constrained
 *    Minimization with a General Positive Definite Inner Product"
 */

#include "LatentMirrorOptimizer.hpp"

#include <cassert>
#include <cmath>
#include <algorithm>
#include <limits>

namespace mfem_lmg {

// ============================================================
//  Internal helpers
// ============================================================
namespace {

constexpr mfem::real_t kInf = std::numeric_limits<mfem::real_t>::infinity();

MFEM_HOST_DEVICE inline bool IsInf   (mfem::real_t v) { return  v >= kInf; }
MFEM_HOST_DEVICE inline bool IsNegInf(mfem::real_t v) { return  v <= -kInf; }

// ── Boundary-safety constants for PrimalToLatent ─────────────────────────
//
// When λ sits at or beyond a finite bound the log argument is zero or
// negative, producing -Inf or NaN.  We clamp the argument to a positive
// minimum δ so that the resulting latent value is large-but-finite and
// consistent with the latent clipping range [-zmax, zmax] used after each
// step.
//
// kLatentClipDefault  – the default zmax passed to SetLatentClipping().
//                       exp(-40) ≈ 4.2e-18  (double) or exp(-15) for float.
// kPrimalMinGap       – absolute minimum gap: exp(-kLatentClipDefault).
//                       Guarantees |z| ≤ kLatentClipDefault even when λ = l
//                       or λ = u exactly.
// kPrimalRelGap       – relative floor scaled to interval width (u-l).
//                       Set to machine epsilon so the gap is never smaller
//                       than the representational precision of the bound.
//
// The effective clamp is:  gap = max(kPrimalMinGap, kPrimalRelGap * width)
// Applied as a branchless fmax — no GPU divergence.

constexpr mfem::real_t kLatentClipDefault =
    mfem::real_t(sizeof(mfem::real_t) == 4 ? 15.0 : 40.0);

constexpr mfem::real_t kPrimalRelGap =
    std::numeric_limits<mfem::real_t>::epsilon();

// exp(-kLatentClipDefault): absolute minimum gap.
// Stored as exact IEEE 754 hex float literals so that
//   log(kPrimalMinGap) == -kLatentClipDefault  exactly in floating point.
// This guarantees |z| <= kLatentClipDefault when λ is at a bound.
//   double: exp(-40) = 0x1.39792499b1a24p-58  (exact)
//   float:  nearest representable value above exp(-15) so that
//           log(kPrimalMinGap) >= -15  in single precision.
//           exp(-15) rounds to 0x1.4875cap-22 which gives log = -15.0000000063,
//           so we use the next float up: 0x1.4875ccp-22, log = -14.9999999133.
constexpr mfem::real_t kPrimalMinGap =
    mfem::real_t(sizeof(mfem::real_t) == 4
                 ? 0x1.4875ccp-22          // float: first float with log >= -15
                 : 0x1.39792499b1a24p-58); // double: exp(-40), exact

// Branchless clamp helper: returns max(x, floor).
MFEM_HOST_DEVICE inline mfem::real_t ClampBelow(mfem::real_t x,
                                                  mfem::real_t floor)
{ return x < floor ? floor : x; }

/** Allocate v with size sz and same device flag as ref, initialized to 0. */
static void DeviceInit(mfem::Vector& v, int sz, const mfem::Vector& ref)
{
    v.SetSize(sz);
    v.UseDevice(ref.UseDevice());
    v = mfem::real_t(0);
}

/** Set UseDevice + zero without reallocating (size must already be correct). */
static void DeviceReuse(mfem::Vector& v, const mfem::Vector& ref)
{
    v.UseDevice(ref.UseDevice());
    v = mfem::real_t(0);
}

} // anonymous namespace


// ============================================================
//  BoundPartition
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

    // Temporary host-side index lists.
    std::vector<int> h_ub, h_lo, h_up, h_ts;
    std::vector<mfem::real_t> h_lo_lo, h_up_hi, h_ts_lo, h_ts_hi;

    for (int i = 0; i < n; ++i) {
        const bool has_lo = !IsNegInf(ld[i]);
        const bool has_hi = !IsInf   (ud[i]);
        if      (!has_lo && !has_hi) {
            h_ub.push_back(i);
        } else if ( has_lo && !has_hi) {
            h_lo.push_back(i);
            h_lo_lo.push_back(ld[i]);
        } else if (!has_lo &&  has_hi) {
            h_up.push_back(i);
            h_up_hi.push_back(ud[i]);
        } else {
            h_ts.push_back(i);
            h_ts_lo.push_back(ld[i]);
            h_ts_hi.push_back(ud[i]);
        }
    }

    n_unbounded_ = (int)h_ub.size();
    n_lower_     = (int)h_lo.size();
    n_upper_     = (int)h_up.size();
    n_twosided_  = (int)h_ts.size();

    // ── Phase 2: upload index arrays to device ────────────────────────────
    const bool use_dev = ref.UseDevice();

    auto UploadIdx = [&](mfem::Array<int>& arr, const std::vector<int>& src) {
        arr.SetSize((int)src.size());
        arr.GetMemory().UseDevice(use_dev);  // Array has no UseDevice(bool); set via Memory
        if (!src.empty()) {
            int* h = arr.HostWrite();
            std::copy(src.begin(), src.end(), h);
            arr.Read();  // push to device if use_dev
        }
    };

    auto UploadVec = [&](mfem::Vector& v, const std::vector<mfem::real_t>& src) {
        v.SetSize((int)src.size());
        v.UseDevice(use_dev);
        if (!src.empty()) {
            mfem::real_t* h = v.HostWrite();
            std::copy(src.begin(), src.end(), h);
            v.Read();    // push to device if use_dev
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

void BoundPartition::GetTypes(std::vector<BoundType>& types) const
{
    const int n = Total();
    types.resize(n, BoundType::Unbounded);

    auto Fill = [&](const mfem::Array<int>& arr, BoundType t) {
        const int* h = arr.HostRead();
        for (int k = 0; k < arr.Size(); ++k) types[h[k]] = t;
    };
    Fill(idx_lower_,    BoundType::LowerOnly);
    Fill(idx_upper_,    BoundType::UpperOnly);
    Fill(idx_twosided_, BoundType::TwoSided);
}


// ============================================================
//  ClassifyBounds (host-only utility)
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
//  PrimalToLatent  (eq. 9) — four branch-free kernels, boundary-safe
//
//  Safety strategy per bound type:
//
//  LowerOnly  z = log(gap)          gap  = max(δ,  λ−l)
//  UpperOnly  z = −log(gap)         gap  = max(δ,  u−λ)
//  TwoSided   z = log(gap_lo/gap_hi) where
//                 gap_lo = max(δ, λ−l),  gap_hi = max(δ, u−λ)
//                 δ = max(kPrimalMinGap, kPrimalRelGap×(u−l))
//
//  All clamps are branchless fmax/fmin — no GPU divergence.
//  When the gap equals δ the latent value is ≈ ±log(1/δ) = ±kLatentClipDefault,
//  which is then handled by the latent clipper after each step.
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
    mfem::real_t*       zp = z.ReadWrite();

    // ── Unbounded: z = λ  (no bound, no clamp needed) ────────────────────
    {
        const int  nc  = part.NumUnbounded();
        const int* idx = part.IdxUnbounded();
        mfem::forall_switch(use_dev, nc, [=] MFEM_HOST_DEVICE (int k) {
            zp[idx[k]] = lp[idx[k]];
        });
    }

    // ── LowerOnly: z = log(max(δ, λ−l)) ──────────────────────────────────
    //   δ = kPrimalMinGap  (absolute; no width available for relative floor)
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

    // ── UpperOnly: z = −log(max(δ, u−λ)) ─────────────────────────────────
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

    // ── TwoSided: z = log(gap_lo / gap_hi) ───────────────────────────────
    //   δ = max(kPrimalMinGap, kPrimalRelGap × (u−l))
    //   Both gaps clamped independently so z stays finite at either bound.
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
//  LatentToPrimal  (eqs. 10–12) — four branch-free kernels
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

    // ── Unbounded: λ = z ──────────────────────────────────────────────────
    {
        const int  nc  = part.NumUnbounded();
        const int* idx = part.IdxUnbounded();
        mfem::forall_switch(use_dev, nc, [=] MFEM_HOST_DEVICE (int k) {
            const int i = idx[k];
            lp[i] = zp[i];
        });
    }

    // ── LowerOnly: λ = l + exp(z) ─────────────────────────────────────────
    {
        const int              nc  = part.NumLower();
        const int*             idx = part.IdxLower();
        const mfem::real_t*    li  = part.LoLower();
        mfem::forall_switch(use_dev, nc, [=] MFEM_HOST_DEVICE (int k) {
            const int i = idx[k];
            lp[i] = li[k] + std::exp(zp[i]);
        });
    }

    // ── UpperOnly: λ = u − exp(−z) ───────────────────────────────────────
    {
        const int              nc  = part.NumUpper();
        const int*             idx = part.IdxUpper();
        const mfem::real_t*    ui  = part.HiUpper();
        mfem::forall_switch(use_dev, nc, [=] MFEM_HOST_DEVICE (int k) {
            const int i = idx[k];
            lp[i] = ui[k] - std::exp(-zp[i]);
        });
    }

    // ── TwoSided: λ = l + (u−l) σ(z) ─────────────────────────────────────
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
//  LatentJacobianDiag — four branch-free kernels
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
//  DefaultPrimalInit (eq. 35) — four branch-free kernels
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
//  Convenience overloads (host-only, use std::vector<BoundType>)
// ============================================================
// ── Host-only convenience overload: same clamping strategy ───────────────
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
// ============================================================
int BoundSafetyCheck(const mfem::Vector& lam,
                     const BoundPartition& part,
                     const mfem::Vector& lo,
                     const mfem::Vector& hi,
                     mfem::real_t tol)
{
    // Bring lam to host regardless of where it currently lives.
    const mfem::real_t* lp = lam.HostRead();
    const mfem::real_t* ld = lo .HostRead();
    const mfem::real_t* ud = hi .HostRead();

    int count = 0;

    // LowerOnly: check λᵢ > lᵢ + tol
    {
        const int* idx = part.IdxLower();
        for (int k = 0; k < part.NumLower(); ++k) {
            const int i = idx[k];
            if (lp[i] <= ld[i] + tol) ++count;
        }
    }
    // UpperOnly: check λᵢ < uᵢ − tol
    {
        const int* idx = part.IdxUpper();
        for (int k = 0; k < part.NumUpper(); ++k) {
            const int i = idx[k];
            if (lp[i] >= ud[i] - tol) ++count;
        }
    }
    // TwoSided: check both sides
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
//          ClipLatent — device kernel
// ============================================================
namespace detail {
static void ClipLatentKernel(mfem::Vector& z, mfem::real_t zmax, bool use_dev)
{
    const int       n  = z.Size();
    mfem::real_t*   zp = z.ReadWrite();
    mfem::forall_switch(use_dev, n, [=] MFEM_HOST_DEVICE (int i) {
        zp[i] = zp[i] >  zmax ?  zmax :
                zp[i] < -zmax ? -zmax : zp[i];
    });
}

// Projected-gradient residual contribution: Σ rᵢ², Σ λᵢ²  (device sums).
// Returns {sum_r2_global, sum_lam2_global} where "global" means
// after MPI_Allreduce when called from the parallel variant.
static void ProjGradResidualKernel(
    int n, bool use_dev,
    const mfem::real_t* lam, const mfem::real_t* d,
    const mfem::real_t* lo,  const mfem::real_t* hi,
    double& sum_r2, double& sum_lam2)
{
    // We use two temporary vectors to accumulate rᵢ² and λᵢ² on device,
    // then sum them with Vector::Sum() which dispatches to cuBLAS on GPU.
    mfem::Vector r2(n), l2(n);
    r2.UseDevice(use_dev);
    l2.UseDevice(use_dev);
    mfem::real_t* r2p = r2.Write();
    mfem::real_t* l2p = l2.Write();
    mfem::forall_switch(use_dev, n, [=] MFEM_HOST_DEVICE (int i) {
        mfem::real_t y = lam[i] - d[i];
        if (!IsNegInf(lo[i])) y = y < lo[i] ? lo[i] : y;
        if (!IsInf   (hi[i])) y = y > hi[i] ? hi[i] : y;
        const mfem::real_t ri = lam[i] - y;
        r2p[i] = ri   * ri;
        l2p[i] = lam[i] * lam[i];
    });
    sum_r2   = double(r2.Sum());
    sum_lam2 = double(l2.Sum());
}
} // namespace detail


// ============================================================
//              LatentMirrorOptimizer (serial)
// ============================================================

// ── Private helpers ───────────────────────────────────────────────────────

void LatentMirrorOptimizer::InitDeviceVectors(const mfem::Vector& ref)
{
    // Resize and mirror device flag; zero-initialize.
    DeviceInit(z_prev_,    n_, ref);
    DeviceInit(lam_prev_,  n_, ref);
    DeviceInit(g_prev_,    n_, ref);
    DeviceInit(g_,         n_, ref);
    DeviceInit(Mg_,        n_, ref);
    DeviceInit(z_trial_,   n_, ref);
    DeviceInit(lam_cur_,   n_, ref);
    DeviceInit(lam_trial_, n_, ref);

    // Migrate full lo/hi to device.
    lo_full_.UseDevice(ref.UseDevice());
    hi_full_.UseDevice(ref.UseDevice());
    lo_full_.Read();
    hi_full_.Read();
}

void LatentMirrorOptimizer::ApplyM(const mfem::Vector& x,
                                    mfem::Vector& Mx) const
{
    if (M_op_) { Mx.SetSize(x.Size()); M_op_->Mult(x, Mx); }
    else        { Mx = x; }
}

void LatentMirrorOptimizer::ApplyMinv(const mfem::Vector& x,
                                       mfem::Vector& Minvx) const
{
    if (M_solver_) { Minvx.SetSize(x.Size()); M_solver_->Mult(x, Minvx); }
    else            { Minvx = x; }
}

mfem::real_t LatentMirrorOptimizer::InnerProductM(
        const mfem::Vector& x, const mfem::Vector& y) const
{
    // mfem::InnerProduct dispatches to cuBLAS on GPU.
    if (M_op_) {
        mfem::Vector My(y.Size());
        My.UseDevice(y.UseDevice());
        M_op_->Mult(y, My);
        return mfem::InnerProduct(x, My);
    }
    return mfem::InnerProduct(x, y);
}

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

void LatentMirrorOptimizer::ClipLatent(mfem::Vector& z) const
{
    detail::ClipLatentKernel(z, zmax_, z.UseDevice());
}

mfem::real_t LatentMirrorOptimizer::ComputeGBBStepSize(
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

// ── Constructors ──────────────────────────────────────────────────────────

LatentMirrorOptimizer::LatentMirrorOptimizer(
        const mfem::Vector& z0,
        const mfem::Vector& lo,
        const mfem::Vector& hi)
    : LatentMirrorOptimizer(z0, lo, hi, nullptr, nullptr) {}

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

// ── Configuration ─────────────────────────────────────────────────────────

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
mfem::real_t LatentMirrorOptimizer::Update(
        mfem::Vector&       z,
        const mfem::Vector& d,
        mfem::real_t        phi_k,
        EvalPhi             eval_phi)
{
    const bool use_dev = z.UseDevice();

    // ── Deferred device initialization (first call only) ─────────────────
    // We wait until Update() to know the device flag from the user's z.
    if (!initialized_) {
        InitDeviceVectors(z);
        // Seed history from initial z so GBB has a safe fallback.
        z_prev_ = z;
        LatentToPrimal(z, part_, lam_prev_);
        g_prev_ = mfem::real_t(0);
        initialized_ = true;
    }

    // ── Step 1: current primal  λᵏ = T(zᵏ) ──────────────────────────────
    LatentToPrimal(z, part_, lam_cur_);

    // ── Step 2: M-gradient  gᵏ = M⁻¹ dᵏ  (eq. 3 / 38) ──────────────────
    ApplyMinv(d, g_);
    // Mg_ = M gᵏ = dᵏ  (used in Armijo pairing; free when M=I)
    ApplyM(g_, Mg_);

    // ── Step 4: initial trial step size ──────────────────────────────────
    mfem::real_t alpha_0;
    if (iter_ == 0) {
        // eq. (22): α₀ = 1 / max(1, ‖g⁰‖∞).
        // g_.Normlinf() dispatches to device reduction.
        alpha_0 = mfem::real_t(1) /
                  std::max(mfem::real_t(1), g_.Normlinf());
        alpha_prev_ = alpha_0;
    } else {
        // GBB (eq. 18) with geometric mean (eq. 21).
        // All subtracts and InnerProductM are device operations.
        mfem::Vector dz(n_), dlam(n_), dg(n_);
        dz.UseDevice(use_dev);
        dlam.UseDevice(use_dev);
        dg.UseDevice(use_dev);
        subtract(z,        z_prev_,   dz);
        subtract(lam_cur_, lam_prev_, dlam);
        subtract(g_,       g_prev_,   dg);
        alpha_0 = ComputeGBBStepSize(dz, dlam, dg, alpha_prev_);
    }

    // ── Save history (before line search overwrites lam_cur_, z) ─────────
    z_prev_   = z;
    lam_prev_ = lam_cur_;
    g_prev_   = g_;

    // ── Step 5: Armijo backtracking line search  (eqs. 23–26) ────────────
    //
    // All O(n) operations (z_trial_ update, LatentToPrimal, dot product)
    // run on device.  Only the Armijo scalar comparison happens on host.
    //
    // Armijo: Φ(λ⁺) ≤ Φ(λᵏ) + c₁ (dᵏ)ᵀ(λ⁺−λᵏ)  (eq. 24)
    // where (dᵏ)ᵀ(λ⁺−λᵏ) = mfem::InnerProduct(Mg_, Δλ)  on device.
    mfem::real_t alpha = alpha_0;
    int ls_steps = 0;

    // Work vector for Δλ = λ⁺ − λᵏ, lives on device.
    mfem::Vector dlam(n_);
    dlam.UseDevice(use_dev);

    for (int ls = 0; ls < max_ls_; ++ls) {
        // z⁺ = zᵏ − α gᵏ  (device AXPY)
        z_trial_ = z;
        z_trial_.Add(-alpha, g_);
        ClipLatent(z_trial_);

        // λ⁺ = T(z⁺)  — four branch-free device kernels
        LatentToPrimal(z_trial_, part_, lam_trial_);

        if (!eval_phi) break;   // no Armijo check requested

        // Armijo pairing (dᵏ)ᵀ(λ⁺−λᵏ) via device dot product.
        subtract(lam_trial_, lam_cur_, dlam);
        const mfem::real_t pairing = mfem::InnerProduct(Mg_, dlam);

        mfem::real_t phi_trial = mfem::real_t(0);
        eval_phi(z_trial_, phi_trial);  // user evaluates Φ(T(z⁺))

        if (phi_trial <= phi_k + c1_ * pairing) break;

        alpha *= beta_;
        ++ls_steps;
    }
    last_ls_steps_ = ls_steps;

    // ── Step 6: accept  (eq. 26) ─────────────────────────────────────────
    alpha_prev_ = alpha;
    z = z_trial_;               // update user's latent vector in-place

    // ── Stationarity residual at zᵏ⁺¹ (device kernels + Sum) ────────────
    LatentToPrimal(z, part_, lam_cur_);
    last_stat_res_ = ProjectedGradResidual(lam_cur_, d);

    // ── Relative M-norm step (eq. 34) ─────────────────────────────────────
    {
        mfem::Vector step(n_);
        step.UseDevice(use_dev);
        subtract(lam_cur_, lam_prev_, step);
        last_rel_step_ = NormM(step) /
                         std::max(mfem::real_t(1), NormM(lam_prev_));
    }

    ++iter_;
    return alpha;
}

// ── StationarityResidual (standalone) ────────────────────────────────────
mfem::real_t LatentMirrorOptimizer::StationarityResidual(
        const mfem::Vector& z, const mfem::Vector& d) const
{
    // Allocate a temporary lam on the same device as z.
    mfem::Vector lam(n_);
    lam.UseDevice(z.UseDevice());
    LatentToPrimal(z, part_, lam);
    return ProjectedGradResidual(lam, d);
}


// ============================================================
#ifdef MFEM_USE_MPI
//           LatentMirrorOptimizerParallel
// ============================================================

// ── Private helpers ───────────────────────────────────────────────────────

void LatentMirrorOptimizerParallel::InitDeviceVectors(const mfem::Vector& ref)
{
    DeviceInit(z_prev_local_,   n_local_, ref);
    DeviceInit(lam_prev_local_, n_local_, ref);
    DeviceInit(g_prev_local_,   n_local_, ref);
    DeviceInit(g_local_,        n_local_, ref);
    DeviceInit(Mg_local_,       n_local_, ref);
    DeviceInit(z_trial_local_,  n_local_, ref);
    DeviceInit(lam_cur_local_,  n_local_, ref);
    DeviceInit(lam_trial_local_,n_local_, ref);

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

// Global M-inner product: local dot on device, then MPI_Allreduce.
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

// ── Configuration ─────────────────────────────────────────────────────────

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
mfem::real_t LatentMirrorOptimizerParallel::Update(
        mfem::Vector&       z_local,
        const mfem::Vector& d_local,
        mfem::real_t        phi_k,
        EvalPhi             eval_phi)
{
    const bool use_dev = z_local.UseDevice();

    if (!initialized_) {
        InitDeviceVectors(z_local);
        z_prev_local_ = z_local;
        LatentToPrimal(z_local, part_, lam_prev_local_);
        g_prev_local_ = mfem::real_t(0);
        initialized_ = true;
    }

    // ── Step 1 ────────────────────────────────────────────────────────────
    LatentToPrimal(z_local, part_, lam_cur_local_);

    // ── Step 2 ────────────────────────────────────────────────────────────
    ApplyMinv(d_local, g_local_);
    ApplyM(g_local_, Mg_local_);

    // ── Step 4 ────────────────────────────────────────────────────────────
    mfem::real_t alpha_0;
    if (iter_ == 0) {
        // Global ‖g⁰‖∞ via MPI_Allreduce.
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
        // ComputeGBBStepSize uses InnerProductM which calls Allreduce.
        // alpha_0 is therefore identical on all ranks.
        alpha_0 = ComputeGBBStepSize(dz, dlam, dg, alpha_prev_);
    }

    z_prev_local_   = z_local;
    lam_prev_local_ = lam_cur_local_;
    g_prev_local_   = g_local_;

    // ── Step 5: Armijo backtracking ───────────────────────────────────────
    // All O(n_local) ops on device.  Armijo pairing reduced via Allreduce
    // so every rank tests the same condition and takes the same path.
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

        subtract(lam_trial_local_, lam_cur_local_, dlam);
        // Global pairing: local dot on device + Allreduce.
        double local_pair  = double(mfem::InnerProduct(Mg_local_, dlam));
        double global_pair = 0.0;
        MPI_Allreduce(&local_pair, &global_pair, 1, MPI_DOUBLE, MPI_SUM, comm_);
        const mfem::real_t pairing = mfem::real_t(global_pair);

        mfem::real_t phi_trial = mfem::real_t(0);
        eval_phi(z_trial_local_, phi_trial);   // all ranks call; returns global phi

        if (phi_trial <= phi_k + c1_ * pairing) break;

        alpha *= beta_;
        ++ls_steps;
    }
    last_ls_steps_ = ls_steps;

    // ── Step 6: accept ────────────────────────────────────────────────────
    alpha_prev_ = alpha;
    z_local = z_trial_local_;

    // ── Stationarity residual (global) ───────────────────────────────────
    LatentToPrimal(z_local, part_, lam_cur_local_);
    last_stat_res_ = ProjectedGradResidualGlobal(lam_cur_local_, d_local);

    // ── Relative step (global M-norm) ─────────────────────────────────────
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

// ── StationarityResidual (standalone, parallel) ───────────────────────────
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
