/**
 * @file LatentMirrorOptimizer.cpp
 * @brief Implementation of the latent-variable mirror-gradient optimizer.
 *
 * All equation references are to:
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

inline bool IsInf   (mfem::real_t v) { return  v >= kInf; }
inline bool IsNegInf(mfem::real_t v) { return  v <= -kInf; }

/** Euclidean projected-gradient residual numerator² + λ²
 *  used for the stationarity check (eq. 33).
 *  Returns {sum of rᵢ², sum of λᵢ²} for n_loc entries.
 */
inline void ProjectedGradResidual(
    int n_loc,
    const mfem::real_t* lam, const mfem::real_t* d,
    const mfem::real_t* lo,  const mfem::real_t* hi,
    double& sum_r2, double& sum_lam2)
{
    sum_r2 = 0.0; sum_lam2 = 0.0;
    for (int i = 0; i < n_loc; ++i) {
        mfem::real_t y = lam[i] - d[i];
        if (!IsNegInf(lo[i])) y = std::max(y, lo[i]);
        if (!IsInf   (hi[i])) y = std::min(y, hi[i]);
        const double ri = double(lam[i] - y);
        sum_r2   += ri * ri;
        sum_lam2 += double(lam[i]) * double(lam[i]);
    }
}

} // anonymous namespace


// ============================================================
//  ClassifyBounds
// ============================================================
void ClassifyBounds(const mfem::Vector& lo,
                    const mfem::Vector& hi,
                    std::vector<BoundType>& types)
{
    const int n = lo.Size();
    assert(hi.Size() == n);
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
//  PrimalToLatent  (eq. 9)
// ============================================================
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
        case BoundType::Unbounded: zp[i] = lp[i];                                    break;
        case BoundType::LowerOnly: zp[i] = std::log(lp[i] - ld[i]);                  break;
        case BoundType::UpperOnly: zp[i] = -std::log(ud[i] - lp[i]);                 break;
        case BoundType::TwoSided:  zp[i] = std::log((lp[i]-ld[i])/(ud[i]-lp[i]));   break;
        }
    }
}


// ============================================================
//  LatentToPrimal  (eqs. 10–12)
// ============================================================
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
        case BoundType::Unbounded: lp[i] = zp[i];                                           break;
        case BoundType::LowerOnly: lp[i] = ld[i] + std::exp(zp[i]);                         break;
        case BoundType::UpperOnly: lp[i] = ud[i] - std::exp(-zp[i]);                        break;
        case BoundType::TwoSided:  lp[i] = ld[i] + (ud[i]-ld[i]) * detail::Sigmoid(zp[i]); break;
        }
    }
}


// ============================================================
//  LatentJacobianDiag  (eqs. 43–44 support)
// ============================================================
void LatentJacobianDiag(const mfem::Vector& z,
                        const mfem::Vector& lo,
                        const mfem::Vector& hi,
                        const std::vector<BoundType>& types,
                        mfem::Vector& diag)
{
    const int n = z.Size();
    diag.SetSize(n);
    const mfem::real_t* zp = z   .HostRead();
    const mfem::real_t* ld = lo  .HostRead();
    const mfem::real_t* ud = hi  .HostRead();
    mfem::real_t*       dp = diag.HostWrite();
    for (int i = 0; i < n; ++i) {
        switch (types[i]) {
        case BoundType::Unbounded: dp[i] = mfem::real_t(1);                                break;
        case BoundType::LowerOnly: dp[i] = std::exp(zp[i]);                                break;
        case BoundType::UpperOnly: dp[i] = std::exp(-zp[i]);                               break;
        case BoundType::TwoSided: {
            const mfem::real_t s = detail::Sigmoid(zp[i]);
            dp[i] = (ud[i]-ld[i]) * s * (mfem::real_t(1) - s);
            break;
        }
        }
    }
}


// ============================================================
//  DefaultPrimalInit  (eq. 35)
// ============================================================
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
        case BoundType::Unbounded: lp[i] = mfem::real_t(0);                    break;
        case BoundType::LowerOnly: lp[i] = ld[i] + mfem::real_t(1);            break;
        case BoundType::UpperOnly: lp[i] = ud[i] - mfem::real_t(1);            break;
        case BoundType::TwoSided:  lp[i] = mfem::real_t(0.5)*(ld[i]+ud[i]);   break;
        }
    }
}


// ============================================================
//              LatentMirrorOptimizer  (serial)
// ============================================================

// ── Private helpers ───────────────────────────────────────────────────────

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
    if (M_op_) {
        mfem::Vector My(y.Size());
        M_op_->Mult(y, My);
        return mfem::InnerProduct(x, My);
    }
    return mfem::InnerProduct(x, y);
}

void LatentMirrorOptimizer::ClipLatent(mfem::Vector& z) const
{
    mfem::real_t* zp = z.HostReadWrite();
    for (int i = 0; i < z.Size(); ++i)
        zp[i] = std::max(-zmax_, std::min(zmax_, zp[i]));
}

mfem::real_t LatentMirrorOptimizer::ComputeGBBStepSize(
        const mfem::Vector& dz, const mfem::Vector& dlam,
        const mfem::Vector& dg, mfem::real_t fallback) const
{
    // αGBB = ⟨Δz, Δλ⟩_M / |⟨Δg, Δλ⟩_M|   (eq. 18)
    const mfem::real_t ak = InnerProductM(dz,  dlam);
    const mfem::real_t bk = InnerProductM(dg,  dlam);
    mfem::real_t ag = (ak > eps_bb_ && std::abs(bk) > eps_bb_)
                        ? ak / std::abs(bk)
                        : fallback;
    ag = std::max(alpha_min_, std::min(alpha_max_, ag));
    return std::sqrt(ag * alpha_prev_);  // geometric mean (eq. 21)
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
    : n_(z0.Size()), lo_(lo), hi_(hi),
      M_op_(M_op), M_solver_(M_solver),
      z_prev_(n_), lam_prev_(n_), g_prev_(n_),
      g_(n_), Mg_(n_), z_trial_(n_), lam_cur_(n_), lam_trial_(n_),
      alpha_prev_(mfem::real_t(1)),
      alpha_min_(mfem::real_t(1e-12)), alpha_max_(mfem::real_t(1e4)),
      eps_bb_(mfem::real_t(1e-14)),
      c1_(mfem::real_t(1e-4)), beta_(mfem::real_t(0.5)), max_ls_(50),
      zmax_(mfem::real_t(40)), stat_tol_(mfem::real_t(1e-6)),
      iter_(0), last_ls_steps_(0),
      last_stat_res_(mfem::real_t(-1)), last_rel_step_(mfem::real_t(-1))
{
    assert(lo.Size() == n_ && hi.Size() == n_);
    ClassifyBounds(lo_, hi_, types_);
    // Warm-start internal prev to z0 so first GBB fallback is clean.
    z_prev_ = z0;
    LatentToPrimal(z0, lo_, hi_, types_, lam_prev_);
    g_prev_ = mfem::real_t(0);  // zero: GBB skipped on k=0 anyway
}

// ── Configuration ─────────────────────────────────────────────────────────

void LatentMirrorOptimizer::SetLineSearchParams(
        mfem::real_t c1, mfem::real_t beta, int max_ls)
{ c1_ = c1; beta_ = beta; max_ls_ = max_ls; }

void LatentMirrorOptimizer::SetStepSizeSafeguards(
        mfem::real_t alpha_min, mfem::real_t alpha_max, mfem::real_t eps_bb)
{ alpha_min_ = alpha_min; alpha_max_ = alpha_max; eps_bb_ = eps_bb; }

void LatentMirrorOptimizer::SetLatentClipping(mfem::real_t zmax) { zmax_ = zmax; }
void LatentMirrorOptimizer::SetStationarityTol(mfem::real_t tol) { stat_tol_ = tol; }

// ── Update  (Algorithm §8) ────────────────────────────────────────────────
//
// Interface: the user holds the latent vector z (not λ).
// On entry, z is the current zᵏ.  On exit, z = zᵏ⁺¹.
// The primal λᵏ = T(zᵏ) is computed internally as needed.
mfem::real_t LatentMirrorOptimizer::Update(
        mfem::Vector&       z,
        const mfem::Vector& d,
        mfem::real_t        phi_k,
        EvalPhi             eval_phi)
{
    assert(z.Size() == n_);

    // ── Step 1: recover current primal  λᵏ = T(zᵏ)
    LatentToPrimal(z, lo_, hi_, types_, lam_cur_);

    // ── Step 2: M-gradient  gᵏ = M⁻¹ dᵏ  (eq. 38 / eq. 3)
    ApplyMinv(d, g_);
    // Pre-compute M gᵏ = dᵏ so that the Armijo pairing
    //   ⟨gᵏ, λ⁺−λᵏ⟩_M = (dᵏ)ᵀ(λ⁺−λᵏ)  (eq. 24)
    // costs only one dot product inside the backtrack loop.
    ApplyM(g_, Mg_);  // When M=I: Mg_ = g_ = d.

    // ── Step 4: choose initial trial step size
    mfem::real_t alpha_0;
    if (iter_ == 0) {
        // eq. (22):  α₀ = 1 / max(1, ‖g⁰‖∞)
        alpha_0 = mfem::real_t(1) /
                  std::max(mfem::real_t(1), g_.Normlinf());
        alpha_prev_ = alpha_0;
    } else {
        // GBB (eq. 18) with geometric mean (eq. 21)
        mfem::Vector dz(n_), dlam(n_), dg(n_);
        subtract(z,        z_prev_,   dz);
        subtract(lam_cur_, lam_prev_, dlam);
        subtract(g_,       g_prev_,   dg);
        alpha_0 = ComputeGBBStepSize(dz, dlam, dg, alpha_prev_);
    }

    // ── Save current state for next iteration's GBB BEFORE the line search
    //    overwrites z.
    z_prev_   = z;
    lam_prev_ = lam_cur_;
    g_prev_   = g_;

    // ── Step 5: backtracking Armijo line search  (eqs. 23–26)
    //
    // Armijo condition:
    //   Φ(λ⁺) ≤ Φ(λᵏ) + c₁ ⟨gᵏ, λ⁺−λᵏ⟩_M
    //          = Φ(λᵏ) + c₁ (dᵏ)ᵀ(λ⁺−λᵏ)         (eq. 24)
    //
    // z⁺  = zᵏ − α gᵏ                              (eq. 23 / 13)
    // λ⁺  = T(z⁺)                                  (eq. 14)
    //
    // The descent pairing (dᵏ)ᵀ(λ⁺−λᵏ) is strictly negative for small α
    // because JT(zᵏ) is diagonal positive, giving
    //   DΦ(λᵏ)[λ⁺−λᵏ] = −α (gᵏ)ᵀ M JT(zᵏ) gᵏ + O(α²) < 0  (eq. 44)
    // so the condition is eventually satisfied.
    mfem::real_t alpha = alpha_0;
    int ls_steps = 0;

    for (int ls = 0; ls < max_ls_; ++ls) {
        // z⁺ = zᵏ − α gᵏ
        z_trial_ = z;                    // z here is still zᵏ (saved above)
        z_trial_.Add(-alpha, g_);
        ClipLatent(z_trial_);

        // λ⁺ = T(z⁺)
        LatentToPrimal(z_trial_, lo_, hi_, types_, lam_trial_);

        if (!eval_phi) break;  // no Armijo check requested — accept immediately

        // Armijo descent pairing:  c₁ (dᵏ)ᵀ (λ⁺ − λᵏ)
        mfem::Vector dlam(n_);
        subtract(lam_trial_, lam_cur_, dlam);
        const mfem::real_t pairing = mfem::InnerProduct(Mg_, dlam); // = dᵀ Δλ

        // Evaluate objective at the trial point.
        mfem::real_t phi_trial = mfem::real_t(0);
        eval_phi(z_trial_, phi_trial);

        if (phi_trial <= phi_k + c1_ * pairing) break;  // Armijo satisfied

        // Reduce step and retry.
        alpha *= beta_;
        ++ls_steps;
    }
    last_ls_steps_ = ls_steps;

    // ── Step 6: accept  (eq. 26)
    alpha_prev_ = alpha;
    z = z_trial_;   // update user's latent vector in-place

    // ── Stationarity residual (eq. 33):  rᵏ_E = λ − Π_K(λ − dᵏ)
    //    Computed at the NEW z = zᵏ⁺¹.
    {
        LatentToPrimal(z, lo_, hi_, types_, lam_cur_);  // λᵏ⁺¹
        double sr2, sl2;
        ProjectedGradResidual(n_,
                              lam_cur_.HostRead(), d.HostRead(),
                              lo_.HostRead(),      hi_.HostRead(),
                              sr2, sl2);
        last_stat_res_ = mfem::real_t(std::sqrt(sr2) /
                                      std::max(1.0, std::sqrt(sl2)));
    }

    // ── Relative M-norm step size (eq. 34):
    //    ‖λᵏ⁺¹ − λᵏ‖_M / max(1, ‖λᵏ‖_M)
    {
        mfem::Vector step(n_);
        subtract(lam_cur_, lam_prev_, step);
        const mfem::real_t sn = NormM(step);
        const mfem::real_t ln = NormM(lam_prev_);
        last_rel_step_ = sn / std::max(mfem::real_t(1), ln);
    }

    ++iter_;
    return alpha;
}

// ── StationarityResidual  (eq. 33, standalone) ───────────────────────────
mfem::real_t LatentMirrorOptimizer::StationarityResidual(
        const mfem::Vector& z, const mfem::Vector& d) const
{
    mfem::Vector lam(n_);
    LatentToPrimal(z, lo_, hi_, types_, lam);
    double sr2, sl2;
    ProjectedGradResidual(n_,
                          lam.HostRead(), d.HostRead(),
                          lo_.HostRead(), hi_.HostRead(),
                          sr2, sl2);
    return mfem::real_t(std::sqrt(sr2) / std::max(1.0, std::sqrt(sl2)));
}


// ============================================================
#ifdef MFEM_USE_MPI
//              LatentMirrorOptimizerParallel
// ============================================================

// ── Private helpers ───────────────────────────────────────────────────────

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

// Global M-inner product: Σ_rank xlocᵀ M_loc y_loc, reduced via Allreduce.
mfem::real_t LatentMirrorOptimizerParallel::InnerProductM(
        const mfem::Vector& x, const mfem::Vector& y) const
{
    double local_val;
    if (M_op_) {
        mfem::Vector My(y.Size());
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
    mfem::real_t* zp = z.HostReadWrite();
    for (int i = 0; i < z.Size(); ++i)
        zp[i] = std::max(-zmax_, std::min(zmax_, zp[i]));
}

mfem::real_t LatentMirrorOptimizerParallel::ComputeGBBStepSize(
        const mfem::Vector& dz,   const mfem::Vector& dlam,
        const mfem::Vector& dg,   mfem::real_t fallback) const
{
    const mfem::real_t ak = InnerProductM(dz, dlam);
    const mfem::real_t bk = InnerProductM(dg, dlam);
    mfem::real_t ag = (ak > eps_bb_ && std::abs(bk) > eps_bb_)
                        ? ak / std::abs(bk)
                        : fallback;
    ag = std::max(alpha_min_, std::min(alpha_max_, ag));
    return std::sqrt(ag * alpha_prev_);
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
      lo_local_(lo_local), hi_local_(hi_local),
      M_op_(M_op_local), M_solver_(M_solver_local),
      z_prev_local_(n_local_), lam_prev_local_(n_local_),
      g_prev_local_(n_local_),
      g_local_(n_local_), Mg_local_(n_local_),
      z_trial_local_(n_local_), lam_cur_local_(n_local_),
      lam_trial_local_(n_local_),
      alpha_prev_(mfem::real_t(1)),
      alpha_min_(mfem::real_t(1e-12)), alpha_max_(mfem::real_t(1e4)),
      eps_bb_(mfem::real_t(1e-14)),
      c1_(mfem::real_t(1e-4)), beta_(mfem::real_t(0.5)), max_ls_(50),
      zmax_(mfem::real_t(40)), stat_tol_(mfem::real_t(1e-6)),
      iter_(0), last_ls_steps_(0),
      last_stat_res_(mfem::real_t(-1)), last_rel_step_(mfem::real_t(-1))
{
    ClassifyBounds(lo_local_, hi_local_, types_);
    z_prev_local_ = z0_local;
    LatentToPrimal(z0_local, lo_local_, hi_local_, types_, lam_prev_local_);
    g_prev_local_ = mfem::real_t(0);
}

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
//
// Interface: user holds z_local (not λ_local).
// On entry z_local = zᵏ_local.  On exit z_local = zᵏ⁺¹_local.
mfem::real_t LatentMirrorOptimizerParallel::Update(
        mfem::Vector&       z_local,
        const mfem::Vector& d_local,
        mfem::real_t        phi_k,
        EvalPhi             eval_phi)
{
    // ── Step 1: current primal  λᵏ = T(zᵏ)
    LatentToPrimal(z_local, lo_local_, hi_local_, types_, lam_cur_local_);

    // ── Step 2: M-gradient  gᵏ = M⁻¹ dᵏ
    ApplyMinv(d_local, g_local_);
    ApplyM(g_local_, Mg_local_);   // Mg = dᵏ for Armijo pairing

    // ── Step 4: initial trial step size
    mfem::real_t alpha_0;
    if (iter_ == 0) {
        double local_gnorm = double(g_local_.Normlinf());
        double global_gnorm = 0.0;
        MPI_Allreduce(&local_gnorm, &global_gnorm, 1, MPI_DOUBLE, MPI_MAX, comm_);
        alpha_0 = mfem::real_t(1) /
                  std::max(mfem::real_t(1), mfem::real_t(global_gnorm));
        alpha_prev_ = alpha_0;
    } else {
        mfem::Vector dz(n_local_), dlam(n_local_), dg(n_local_);
        subtract(z_local,        z_prev_local_,   dz);
        subtract(lam_cur_local_, lam_prev_local_,  dlam);
        subtract(g_local_,       g_prev_local_,    dg);
        alpha_0 = ComputeGBBStepSize(dz, dlam, dg, alpha_prev_);
        // alpha_0 is already global (InnerProductM used Allreduce) so
        // every rank has the same value.
    }

    // ── Save history before line search modifies z_local
    z_prev_local_   = z_local;
    lam_prev_local_ = lam_cur_local_;
    g_prev_local_   = g_local_;

    // ── Step 5: backtracking Armijo line search
    //
    // All ranks run identical iterations because:
    //   – alpha_0 was computed from global inner products (same on all ranks)
    //   – eval_phi must return the GLOBAL objective (same on all ranks)
    //   – alpha is reduced by the same factor beta_ on all ranks
    mfem::real_t alpha = alpha_0;
    int ls_steps = 0;

    for (int ls = 0; ls < max_ls_; ++ls) {
        // z⁺ = zᵏ − α gᵏ
        z_trial_local_ = z_local;
        z_trial_local_.Add(-alpha, g_local_);
        ClipLatent(z_trial_local_);

        // λ⁺ = T(z⁺)
        LatentToPrimal(z_trial_local_, lo_local_, hi_local_, types_,
                       lam_trial_local_);

        if (!eval_phi) break;  // no check — accept directly

        // Armijo pairing: (dᵏ)ᵀ(λ⁺−λᵏ)  — local contribution; global is
        // the sum over ranks, but since phi_k and phi_trial are already
        // global, the sign of the local pairing suffices only if all ranks
        // agree.  We therefore compute the GLOBAL pairing via MPI_Allreduce.
        mfem::Vector dlam(n_local_);
        subtract(lam_trial_local_, lam_cur_local_, dlam);
        double local_pair = double(mfem::InnerProduct(Mg_local_, dlam));
        double global_pair = 0.0;
        MPI_Allreduce(&local_pair, &global_pair, 1, MPI_DOUBLE, MPI_SUM, comm_);
        const mfem::real_t pairing = mfem::real_t(global_pair);

        mfem::real_t phi_trial = mfem::real_t(0);
        eval_phi(z_trial_local_, phi_trial);  // all ranks call; must return global phi

        if (phi_trial <= phi_k + c1_ * pairing) break;  // Armijo satisfied

        alpha *= beta_;
        ++ls_steps;
    }
    last_ls_steps_ = ls_steps;

    // ── Step 6: accept
    alpha_prev_ = alpha;
    z_local = z_trial_local_;   // update user's local latent chunk in-place

    // ── Stationarity residual (global, eq. 33)
    {
        LatentToPrimal(z_local, lo_local_, hi_local_, types_, lam_cur_local_);
        double sr2_loc, sl2_loc;
        ProjectedGradResidual(n_local_,
                              lam_cur_local_.HostRead(), d_local.HostRead(),
                              lo_local_.HostRead(),      hi_local_.HostRead(),
                              sr2_loc, sl2_loc);
        double buf[2]  = {sr2_loc, sl2_loc};
        double gbuf[2] = {0.0, 0.0};
        MPI_Allreduce(buf, gbuf, 2, MPI_DOUBLE, MPI_SUM, comm_);
        last_stat_res_ = mfem::real_t(std::sqrt(gbuf[0]) /
                                      std::max(1.0, std::sqrt(gbuf[1])));
    }

    // ── Relative M-norm step size (global, eq. 34)
    {
        mfem::Vector step(n_local_);
        subtract(lam_cur_local_, lam_prev_local_, step);
        const mfem::real_t sn = NormM(step);
        const mfem::real_t ln = NormM(lam_prev_local_);
        last_rel_step_ = sn / std::max(mfem::real_t(1), ln);
    }

    ++iter_;
    return alpha;
}

// ── StationarityResidual (standalone, parallel) ───────────────────────────
mfem::real_t LatentMirrorOptimizerParallel::StationarityResidual(
        const mfem::Vector& z_local, const mfem::Vector& d_local) const
{
    mfem::Vector lam(n_local_);
    LatentToPrimal(z_local, lo_local_, hi_local_, types_, lam);
    double sr2_loc, sl2_loc;
    ProjectedGradResidual(n_local_,
                          lam.HostRead(),       d_local.HostRead(),
                          lo_local_.HostRead(), hi_local_.HostRead(),
                          sr2_loc, sl2_loc);
    double buf[2]  = {sr2_loc, sl2_loc};
    double gbuf[2] = {0.0, 0.0};
    MPI_Allreduce(buf, gbuf, 2, MPI_DOUBLE, MPI_SUM, comm_);
    return mfem::real_t(std::sqrt(gbuf[0]) / std::max(1.0, std::sqrt(gbuf[1])));
}

#endif // MFEM_USE_MPI

} // namespace mfem_lmg
