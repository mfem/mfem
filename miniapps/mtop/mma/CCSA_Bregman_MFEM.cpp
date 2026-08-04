/**
 * CCSA_Bregman_MFEM.cpp  -  Device-aware entropy-CCSA optimiser for MFEM
 *
 * See CCSA_Bregman_MFEM.hpp for the full design rationale.  Section numbers
 * quoted below refer to the technical note "A Fermi-Dirac Bregman Separable
 * Approximation for CCSA/MMA" ([Note]).
 *
 * ─── What this file implements ───────────────────────────────────────────
 *  • BoundsGeometry            – TwoSided / LowerOnly / UpperOnly latent maps
 *  • detail::SolveDualEntropy  – pluggable dual solver (ProjectedAscent /
 *                                 BarrierNewton) for the Bregman subproblem
 *  • CCSAOptimizer             – serial optimiser
 *  • CCSAOptimizerParallel     – MPI-parallel optimiser
 *
 * ─── Relationship to MMA_MFEM.hpp / MMA_MFEM.cpp ─────────────────────────
 *  This file lives in the SAME namespace (mfem_mma) as MMA_MFEM.hpp/.cpp and
 *  genuinely reuses, rather than duplicates, everything that is safe to
 *  reuse across a translation-unit boundary:
 *    - PackFival() / PackFivalRelaxed() / PackedDfidx   (inline, in the header)
 *    - detail::SolveDense()                              (external linkage,
 *      declared in MMA_MFEM.hpp, defined in MMA_MFEM.cpp; used here by the
 *      BarrierNewton dual solver without a local re-implementation)
 *  Anything declared `static` in MMA_MFEM.cpp (mma_Allreduce, MMA_SERIAL_COMM,
 *  VecToDouble, DefaultPenalty, ComputeNGlobal) has internal linkage and is
 *  therefore NOT reachable from this translation unit; this file carries its
 *  own small ccsa_-prefixed equivalents of those five helpers below, purely
 *  because MMA_MFEM.hpp/.cpp are being kept untouched and separate in this
 *  initial phase. If/when CCSAOptimizer is folded into MMA_MFEM.hpp/.cpp
 *  directly, these five local helpers are the only things that would be
 *  deleted in favour of the originals -- nothing else changes.
 *
 * ─── Build / link ─────────────────────────────────────────────────────────
 *  Compile and link this file together with MMA_MFEM.cpp (this file calls
 *  mfem_mma::detail::SolveDense(), which is defined there).
 *
 * ─── Device / GPU strategy ────────────────────────────────────────────────
 *  Identical to MMA_MFEM.cpp: O(n) work runs via mfem::forall_switch on
 *  whichever device the caller's Vectors live on; the (small) dual system is
 *  always assembled/solved on the host.
 */

#include "CCSA_Bregman_MFEM.hpp"
#include <numeric>
#include <string>

namespace mfem_mma {

// ─── Small internal helpers, local to this translation unit ─────────────────
// See the file-level comment above: these mirror MMA_MFEM.cpp's private
// helpers of (almost) the same name/purpose, but cannot literally be the
// same functions while MMA_MFEM.hpp/.cpp remain unmodified and separately
// compiled (those are declared `static`, i.e. internal linkage).

/// Sentinel communicator used by the serial CCSAOptimizer so that
/// detail::SolveDualEntropy() never needs an active MPI_Init(), exactly
/// mirroring MMA_MFEM.cpp's MMA_SERIAL_COMM trick.
static constexpr MPI_Comm CCSA_SERIAL_COMM = 0;

static inline void ccsa_Allreduce(const double* src, double* dst, int n, MPI_Comm comm)
{
#ifdef MFEM_USE_MPI
    if (comm == CCSA_SERIAL_COMM) std::copy(src, src+n, dst);
    else MPI_Allreduce(src, dst, n, MPI_DOUBLE, MPI_SUM, comm);
#else
    (void)comm;
    std::copy(src, src+n, dst);
#endif
}

/// Convert an mfem::Vector (real_t) to a host std::vector<double>. Used when
/// penalty parameters are supplied as mfem::Vector. Mirrors MMA_MFEM.cpp's
/// VecToDouble().
static std::vector<double> VecToDouble(const mfem::Vector& v)
{
    const mfem::real_t* h = v.HostRead();
    return std::vector<double>(h, h+v.Size());
}

/// Default CCSA penalty parameters (a=0, c=max(1000,10n), d=1), same
/// convention as MMA_MFEM.cpp's DefaultPenalty().
static void DefaultPenalty(int n_global, int m,
                            std::vector<double>& a, std::vector<double>& c, std::vector<double>& d)
{
    a.assign(m, 0.0);
    c.assign(m, std::max(1000.0, 10.0*n_global));
    d.assign(m, 1.0);
}

#ifdef MFEM_USE_MPI
/// Mirrors MMA_MFEM.cpp's ComputeNGlobal().
static int ComputeNGlobal(MPI_Comm comm, int n_local)
{
    int ng=0; MPI_Allreduce(&n_local, &ng, 1, MPI_INT, MPI_SUM, comm); return ng;
}
#endif


// ============================================================
// BoundsGeometry
// ============================================================

BoundsGeometry BoundsGeometry::TwoSided(const mfem::Vector& lo, const mfem::Vector& hi)
{
    BoundsGeometry g;
    g.n_ = lo.Size(); g.kind_ = BoundsKind::TwoSided; g.set_ = true;
    g.lo_ = lo; g.hi_ = hi;
    g.h_.SetSize(g.n_); g.h_.UseDevice(lo.UseDevice());
    {
        bool ud = lo.UseDevice();
        const auto* lo_r = g.lo_.Read(); const auto* hi_r = g.hi_.Read();
        auto* h_w = g.h_.Write();
        mfem::forall_switch(ud, g.n_, [=] MFEM_HOST_DEVICE (int j) {
            h_w[j] = hi_r[j] - lo_r[j];
        });
    }
    return g;
}

BoundsGeometry BoundsGeometry::TwoSided(int n, mfem::real_t lo, mfem::real_t hi, bool use_device)
{
    mfem::Vector lov(n), hiv(n);
    lov.UseDevice(use_device); hiv.UseDevice(use_device);
    lov = lo; hiv = hi;
    return TwoSided(lov, hiv);
}

BoundsGeometry BoundsGeometry::LowerOnly(const mfem::Vector& lo, const mfem::Vector& scale)
{
    BoundsGeometry g;
    g.n_ = lo.Size(); g.kind_ = BoundsKind::LowerOnly; g.set_ = true;
    g.lo_ = lo; g.h_ = scale;
    return g;
}

BoundsGeometry BoundsGeometry::LowerOnly(int n, mfem::real_t lo, mfem::real_t scale, bool use_device)
{
    mfem::Vector lov(n), sv(n);
    lov.UseDevice(use_device); sv.UseDevice(use_device);
    lov = lo; sv = scale;
    return LowerOnly(lov, sv);
}

BoundsGeometry BoundsGeometry::UpperOnly(const mfem::Vector& hi, const mfem::Vector& scale)
{
    BoundsGeometry g;
    g.n_ = hi.Size(); g.kind_ = BoundsKind::UpperOnly; g.set_ = true;
    g.hi_ = hi; g.h_ = scale;
    return g;
}

BoundsGeometry BoundsGeometry::UpperOnly(int n, mfem::real_t hi, mfem::real_t scale, bool use_device)
{
    mfem::Vector hiv(n), sv(n);
    hiv.UseDevice(use_device); sv.UseDevice(use_device);
    hiv = hi; sv = scale;
    return UpperOnly(hiv, sv);
}

// ToLatent: physical -> latent.  See [Note] Sec. 8.1 (TwoSided) and the
// one-sided generalisation described in the header (BoundsKind doc comment).
void BoundsGeometry::ToLatent(const mfem::Vector& x, mfem::Vector& eta) const
{
    if (!set_) throw std::runtime_error("CCSA_Bregman: BoundsGeometry is unset");
    bool ud = x.UseDevice();
    eta.SetSize(n_); eta.UseDevice(ud);
    const mfem::real_t eps = clip_eps_;
    const BoundsKind kind = kind_;
    const auto* xr = x.Read();
    const auto* hr = h_.Read();
    auto* er = eta.Write();
    if (kind == BoundsKind::TwoSided) {
        const auto* lr = lo_.Read();
        mfem::forall_switch(ud, n_, [=] MFEM_HOST_DEVICE (int j) {
            double p = (double(xr[j]) - double(lr[j])) / double(hr[j]);
            p = p < double(eps) ? double(eps) : (p > 1.0-double(eps) ? 1.0-double(eps) : p);
            er[j] = mfem::real_t(::log(p/(1.0-p)));
        });
    } else if (kind == BoundsKind::LowerOnly) {
        const auto* lr = lo_.Read();
        mfem::forall_switch(ud, n_, [=] MFEM_HOST_DEVICE (int j) {
            double p = (double(xr[j]) - double(lr[j])) / double(hr[j]);
            p = p < double(eps) ? double(eps) : p;
            er[j] = mfem::real_t(::log(p));
        });
    } else { // UpperOnly
        const auto* ur = hi_.Read();
        mfem::forall_switch(ud, n_, [=] MFEM_HOST_DEVICE (int j) {
            double p = (double(ur[j]) - double(xr[j])) / double(hr[j]);
            p = p < double(eps) ? double(eps) : p;
            er[j] = mfem::real_t(-::log(p));
        });
    }
}

// ToPhysical: latent -> physical.  Inverse of ToLatent(); always strictly
// interior by construction, no clipping needed.
void BoundsGeometry::ToPhysical(const mfem::Vector& eta, mfem::Vector& x) const
{
    if (!set_) throw std::runtime_error("CCSA_Bregman: BoundsGeometry is unset");
    bool ud = eta.UseDevice();
    x.SetSize(n_); x.UseDevice(ud);
    const BoundsKind kind = kind_;
    const auto* er = eta.Read();
    const auto* hr = h_.Read();
    auto* xr = x.Write();
    if (kind == BoundsKind::TwoSided) {
        const auto* lr = lo_.Read();
        mfem::forall_switch(ud, n_, [=] MFEM_HOST_DEVICE (int j) {
            double p = 1.0/(1.0+::exp(-double(er[j])));
            xr[j] = mfem::real_t(double(lr[j]) + double(hr[j])*p);
        });
    } else if (kind == BoundsKind::LowerOnly) {
        const auto* lr = lo_.Read();
        mfem::forall_switch(ud, n_, [=] MFEM_HOST_DEVICE (int j) {
            double p = ::exp(double(er[j]));
            xr[j] = mfem::real_t(double(lr[j]) + double(hr[j])*p);
        });
    } else { // UpperOnly
        const auto* ur = hi_.Read();
        mfem::forall_switch(ud, n_, [=] MFEM_HOST_DEVICE (int j) {
            double p = ::exp(-double(er[j]));
            xr[j] = mfem::real_t(double(ur[j]) - double(hr[j])*p);
        });
    }
}

namespace detail {

// ─────────────────────────────────────────────────────────────────────────────
// EntropyPrimalUpdate — the fixed-lambda closed-form latent update shared by
// TwoSided/LowerOnly/UpperOnly ([Note] eq. 48/91/98 and its one-sided
// generalisation, see the BoundsKind doc comment in the header).
//
//   eta_trial_j = eta_c_j - h_j * C_j / R
//   x_trial_j   = bounds.ToPhysical(eta_trial)_j
//   kappa_j     = local curvature factor for the barrier-Newton Hessian
//                 (p(1-p) TwoSided, p one-sided)
//   wterm_j     = per-coordinate Bregman divergence D_j(x_trial_j) ([Note]
//                 eq. 85 TwoSided; analogous unnormalised-KL form one-sided)
//
// kappa and wterm pointers may be null when not needed by the caller.
// ─────────────────────────────────────────────────────────────────────────────
static void EntropyPrimalUpdate(
    bool use_dev, int n_loc, BoundsKind kind,
    const mfem::real_t* lo, const mfem::real_t* hi, const mfem::real_t* h,
    const mfem::real_t* eta_c, const mfem::Vector& Cvec, double R,
    mfem::real_t* eta_trial, mfem::real_t* x_trial,
    mfem::real_t* kappa, mfem::real_t* wterm)
{
    const auto* Cr = Cvec.Read();
    mfem::forall_switch(use_dev, n_loc, [=] MFEM_HOST_DEVICE (int j) {
        double hj = double(h[j]);
        double ecj = double(eta_c[j]);
        double ej  = ecj - hj*double(Cr[j])/R;
        double xj, kap, w;
        if (kind == BoundsKind::TwoSided) {
            double p  = 1.0/(1.0+::exp(-ej));
            double A  = (ej>0.0 ? ej : 0.0) + ::log(1.0+::exp(-(ej>0.0?ej:-ej)));
            double Ac = (ecj>0.0? ecj: 0.0) + ::log(1.0+::exp(-(ecj>0.0?ecj:-ecj)));
            xj  = double(lo[j]) + hj*p;
            kap = p*(1.0-p);
            w   = p*(ej-ecj) - A + Ac;
        } else if (kind == BoundsKind::LowerOnly) {
            double t  = ::exp(ej);
            double t0 = ::exp(ecj);
            xj  = double(lo[j]) + hj*t;
            kap = t;
            w   = t*(ej-ecj) - t + t0;
        } else { // UpperOnly
            double s  = ::exp(-ej);
            double s0 = ::exp(-ecj);
            xj  = double(hi[j]) - hj*s;
            kap = s;
            w   = s*(ecj-ej) - s + s0;
        }
        eta_trial[j] = mfem::real_t(ej);
        x_trial[j]   = mfem::real_t(xj);
        if (kappa) kappa[j] = mfem::real_t(kap);
        if (wterm) wterm[j] = mfem::real_t(w);
    });
}

// BregmanTerm — compute the shared per-coordinate Bregman divergence term
// W_j(eta, eta_c) directly from two already-known latent vectors. This is
// the same formula used inside EntropyPrimalUpdate's wterm output, factored
// out here because the outer conservatism check in UpdateGCMMA() already
// has eta_trial in hand and does not need to re-derive it from C/R.
static void BregmanTerm(bool use_dev, int n, BoundsKind kind,
                         const mfem::Vector& eta, const mfem::Vector& eta_c,
                         mfem::Vector& wterm)
{
    const auto* etat = eta.Read(); const auto* etac = eta_c.Read();
    auto* w = wterm.Write();
    mfem::forall_switch(use_dev, n, [=] MFEM_HOST_DEVICE (int j) {
        double ej=double(etat[j]), ecj=double(etac[j]); double val;
        if (kind==BoundsKind::TwoSided) {
            double p=1.0/(1.0+::exp(-ej));
            double A =(ej>0.0?ej:0.0)+::log(1.0+::exp(-(ej>0.0?ej:-ej)));
            double Ac=(ecj>0.0?ecj:0.0)+::log(1.0+::exp(-(ecj>0.0?ecj:-ecj)));
            val = p*(ej-ecj)-A+Ac;
        } else if (kind==BoundsKind::LowerOnly) {
            double t=::exp(ej), t0=::exp(ecj); val = t*(ej-ecj)-t+t0;
        } else {
            double s=::exp(-ej), s0=::exp(-ecj); val = s*(ecj-ej)-s+s0;
        }
        w[j]=mfem::real_t(val);
    });
}

// ComputeKappa — the curvature/chain-rule factor kappa_j = (dx_j/deta_j)/h_j
// at a GIVEN latent point (no dual solve, no trial move -- just evaluated
// at the current eta). Used by KKTresidual()'s latent-space stationarity
// check: dx_j/deta_j = h_j*kappa_j exactly reproduces what the old
// physical-space projected-gradient check achieved by explicit clipping,
// since kappa_j -> 0 as x_j saturates a bound (see BoundsKind's doc
// comment in the header for the p(1-p)/p/p formulas by kind).
static void ComputeKappa(bool use_dev, int n, BoundsKind kind,
                          const mfem::real_t* eta, mfem::Vector& kappa_out)
{
    auto* kw = kappa_out.Write();
    if (kind == BoundsKind::TwoSided) {
        mfem::forall_switch(use_dev, n, [=] MFEM_HOST_DEVICE (int j){
            double e = double(eta[j]);
            double p = 1.0/(1.0+::exp(-e));
            kw[j] = mfem::real_t(p*(1.0-p));
        });
    } else if (kind == BoundsKind::LowerOnly) {
        mfem::forall_switch(use_dev, n, [=] MFEM_HOST_DEVICE (int j){
            kw[j] = mfem::real_t(::exp(double(eta[j])));
        });
    } else { // UpperOnly
        mfem::forall_switch(use_dev, n, [=] MFEM_HOST_DEVICE (int j){
            kw[j] = mfem::real_t(::exp(-double(eta[j])));
        });
    }
}

// Euclidean-ish projection onto the dual feasible set D ([Note] eq. 42):
//   lambda_i >= 0,  lambda_i <= c_i if d_i == 0,  a^T lambda <= a0 (a0 == 1).
// The linear constraint is handled by bisecting a scalar multiplier; when
// a_pen is all zero (the common default penalty, see DefaultPenalty()) the
// halfspace is trivially satisfied and this reduces to a plain box clip.
static void ProjectDual(std::vector<double>& lam,
                         const std::vector<double>& a_pen,
                         const std::vector<double>& c_pen,
                         const std::vector<double>& d_pen,
                         double a0)
{
    const int m = (int)lam.size();
    auto boxclip = [&](std::vector<double>& v) {
        for (int i=0;i<m;++i) {
            v[i] = std::max(v[i], 0.0);
            if (d_pen[i]==0.0) v[i] = std::min(v[i], c_pen[i]);
        }
    };
    boxclip(lam);
    double atl=0.0; for (int i=0;i<m;++i) atl += a_pen[i]*lam[i];
    if (atl <= a0) return;
    double lo=0.0, hi=1.0;
    auto feas = [&](double mu) {
        double s=0.0;
        for (int i=0;i<m;++i) {
            double v = lam[i]-mu*a_pen[i];
            if (d_pen[i]==0.0) v=std::min(v,c_pen[i]);
            v = std::max(v,0.0);
            s += a_pen[i]*v;
        }
        return s;
    };
    int guard=0; while (feas(hi) > a0 && guard<60) { hi*=2.0; ++guard; }
    for (int it=0; it<60; ++it) {
        double mid=0.5*(lo+hi);
        if (feas(mid) > a0) lo=mid; else hi=mid;
    }
    double mu=0.5*(lo+hi);
    for (int i=0;i<m;++i) {
        double v = lam[i]-mu*a_pen[i];
        if (d_pen[i]==0.0) v=std::min(v,c_pen[i]);
        lam[i]=std::max(v,0.0);
    }
}

// Recover the exact primal (z*, y*) from the converged model constraint
// values gamma_i = g_i(x_trial; rho_i) ([Note] Sec. 7, eq. 74-79; a0 == 1).
static void RecoverZY(const std::vector<double>& gamma,
                      const std::vector<double>& a_pen,
                      const std::vector<double>& c_pen,
                      const std::vector<double>& d_pen,
                      double& z_out, std::vector<double>& y_out)
{
    const int m=(int)gamma.size();
    const double a0 = 1.0;
    auto dphi = [&](double z) {
        double s=0.0;
        for (int i=0;i<m;++i) {
            double g = gamma[i]-a_pen[i]*z;
            if (g>0.0) s += a_pen[i]*(c_pen[i]+d_pen[i]*g);
        }
        return a0 - s;
    };
    if (dphi(0.0) >= 0.0) { z_out = 0.0; }
    else {
        double hi=1.0;
        for (int i=0;i<m;++i) if (a_pen[i]>0.0) hi=std::max(hi, std::max(0.0,gamma[i])/a_pen[i]);
        int guard=0; while (dphi(hi) < 0.0 && guard<60) { hi*=2.0; ++guard; }
        double lo=0.0;
        for (int it=0; it<60; ++it) {
            double mid=0.5*(lo+hi);
            if (dphi(mid) < 0.0) lo=mid; else hi=mid;
        }
        z_out = 0.5*(lo+hi);
    }
    y_out.resize(m);
    for (int i=0;i<m;++i) y_out[i] = std::max(0.0, gamma[i]-a_pen[i]*z_out);
}

// ─────────────────────────────────────────────────────────────────────────────
// SolveDualEntropy — the pluggable dual solver, dispatching on DualSolverKind.
// ─────────────────────────────────────────────────────────────────────────────
void SolveDualEntropy(
    DualSolverKind kind,
    MPI_Comm comm,
    int n_loc, int m, int n_eq,
    bool use_dev,
    const BoundsGeometry& bounds,
    const mfem::real_t* eta_c_loc,
    const mfem::Vector& b0_loc,
    const std::vector<mfem::Vector>& bi_loc,
    const std::vector<double>& F,
    const std::vector<double>& rho,
    const std::vector<double>& a_pen,
    const std::vector<double>& c_pen,
    const std::vector<double>& d_pen,
    std::vector<double>& lam,
    std::vector<double>& mu,
    std::vector<double>& y,
    double& z,
    double dual_tol,
    int dual_max_iter,
    mfem::real_t* eta_trial_loc,
    mfem::real_t* x_trial_loc)
{
    const BoundsKind bkind = bounds.Kind();
    const mfem::real_t* lo_r = bounds.Lo().Size() ? bounds.Lo().Read() : nullptr;
    const mfem::real_t* hi_r = bounds.Hi().Size() ? bounds.Hi().Read() : nullptr;
    const mfem::real_t* h_r  = bounds.Scale().Read();

    // ── m == 0: closed form, no dual iteration at all ─────────────────────
    if (m == 0) {
        mfem::Vector Czero = b0_loc; // C = b0 when there are no constraints
        EntropyPrimalUpdate(use_dev,n_loc,bkind,lo_r,hi_r,h_r,eta_c_loc,Czero,
                             std::max(rho[0],1e-300),
                             eta_trial_loc,x_trial_loc,nullptr,nullptr);
        z = 0.0;
        return;
    }

    // ── Scratch device buffers ────────────────────────────────────────────
    mfem::Vector Cvec(n_loc), wterm(n_loc), kappa(n_loc), dxvec(n_loc);
    Cvec.UseDevice(use_dev); wterm.UseDevice(use_dev); kappa.UseDevice(use_dev); dxvec.UseDevice(use_dev);
    mfem::Vector eta_c_v(n_loc); eta_c_v.UseDevice(use_dev);
    { auto* w=eta_c_v.Write(); mfem::forall_switch(use_dev,n_loc,[=] MFEM_HOST_DEVICE (int j){ w[j]=eta_c_loc[j]; }); }
    mfem::Vector x_c_v = bounds.ToPhysical(eta_c_v);

    mfem::Vector eta_trial_v(n_loc), x_trial_v(n_loc);
    eta_trial_v.UseDevice(use_dev); x_trial_v.UseDevice(use_dev);

    double n_global_d = 0.0;
    { double nl=double(n_loc); ccsa_Allreduce(&nl,&n_global_d,1,comm); }

    // BuildC: C_j(lambda) = b0_j + sum_i lambda_i * bi_j[i],  R = rho0 + sum lambda_i rho_i.
    auto BuildC = [&](const std::vector<double>& l, double& R) {
        {
            const auto* b0r = b0_loc.Read();
            auto* cw = Cvec.Write();
            mfem::forall_switch(use_dev,n_loc,[=] MFEM_HOST_DEVICE (int j){ cw[j]=b0r[j]; });
        }
        for (int i=0;i<m;++i) {
            double li=l[i];
            const auto* bir = bi_loc[i].Read();
            auto* cw = Cvec.ReadWrite();
            mfem::forall_switch(use_dev,n_loc,[=] MFEM_HOST_DEVICE (int j){ cw[j]+=li*bir[j]; });
        }
        R = rho[0];
        for (int i=0;i<m;++i) R += l[i]*rho[i+1];
        if (R < 1e-300) R = 1e-300;
    };

    // EvalResidual: at dual point l, compute the primal update, the shared
    // Bregman term W, the model constraint values gamma_i, the y^lambda
    // slacks, and the dual residual r_i = gamma_i - y_i^lambda ([Note] eq. 57).
    // kappa_out is populated for use by AssembleHessian (BarrierNewton only).
    auto EvalResidual = [&](const std::vector<double>& l,
                             std::vector<double>& gamma, std::vector<double>& ylam,
                             std::vector<double>& r, double& R_out) {
        double R; BuildC(l,R); R_out = R;
        EntropyPrimalUpdate(use_dev,n_loc,bkind,lo_r,hi_r,h_r,eta_c_loc,Cvec,R,
                            eta_trial_v.Write(),x_trial_v.Write(),
                            kappa.Write(),wterm.Write());
        double W = wterm.Sum();
        gamma.resize(m); ylam.resize(m); r.resize(m);
        // dx = x_trial - x_c (device), reused for every constraint's inner product
        {
            const auto* xt = x_trial_v.Read(); const auto* xc = x_c_v.Read();
            auto* dxw = dxvec.Write();
            mfem::forall_switch(use_dev,n_loc,[=] MFEM_HOST_DEVICE (int j){ dxw[j]=xt[j]-xc[j]; });
        }
        std::vector<double> loc(m);
        for (int i=0;i<m;++i) loc[i] = mfem::InnerProduct(bi_loc[i], dxvec);
        std::vector<double> dxi(m);
        ccsa_Allreduce(loc.data(), dxi.data(), m, comm);
        double Wglobal=0.0; ccsa_Allreduce(&W,&Wglobal,1,comm);
        for (int i=0;i<m;++i) {
            gamma[i] = F[i+1] + dxi[i] + rho[i+1]*Wglobal;
            ylam[i]  = (d_pen[i]>0.0) ? std::max(0.0,(l[i]-c_pen[i])/d_pen[i]) : 0.0;
            r[i]     = gamma[i] - ylam[i];
        }
    };

    // AssembleHessian: dense m x m Hessian of the (unbarriered) dual function
    // psi ([Note] eq. 62-65), reusing Cvec/kappa from the most recent
    // EvalResidual() call at the same lambda.
    auto AssembleHessian = [&](const std::vector<double>& l, double R,
                                std::vector<double>& hess) {
        hess.assign(m*m,0.0);
        // s_i = b_i - rho_i * C/R  (device), scaled_i = h^2 * kappa / R * s_i
        std::vector<mfem::Vector> s(m), scaled(m);
        for (int i=0;i<m;++i) {
            s[i].SetSize(n_loc); s[i].UseDevice(use_dev);
            scaled[i].SetSize(n_loc); scaled[i].UseDevice(use_dev);
            const auto* bir = bi_loc[i].Read();
            const auto* Cr  = Cvec.Read();
            const auto* hr  = h_r; const auto* kr = kappa.Read();
            double rho_i = rho[i+1];
            auto* sw = s[i].Write(); auto* scw = scaled[i].Write();
            mfem::forall_switch(use_dev,n_loc,[=] MFEM_HOST_DEVICE (int j){
                double sij = double(bir[j]) - rho_i*double(Cr[j])/R;
                sw[j] = mfem::real_t(sij);
                double hj = double(hr[j]);
                scw[j] = mfem::real_t(hj*hj*double(kr[j])/R * sij);
            });
        }
        std::vector<double> loc(m*m,0.0);
        for (int r=0;r<m;++r)
            for (int c=r;c<m;++c) {
                double v = mfem::InnerProduct(scaled[r], s[c]);
                loc[r*m+c] = v; if (r!=c) loc[c*m+r]=v;
            }
        ccsa_Allreduce(loc.data(), hess.data(), m*m, comm);
        for (int r=0;r<m;++r) for (int c=0;c<m;++c) hess[r*m+c] = -hess[r*m+c];
        for (int i=0;i<m;++i) if (d_pen[i]>0.0 && l[i]>c_pen[i]) hess[i*m+i] -= 1.0/d_pen[i];
    };

    // ── ProjectedAscent (Sec. 6.7): simple, no global-convergence guarantee
    //    for m>1 in general, but robust and cheap. Backtracks on residual
    //    norm decrease. This is the default / "start simple" solver. ──────
    auto SolveProjected = [&]() {
        double alpha = 1.0;
        std::vector<double> gamma,ylam,r; double R;
        EvalResidual(lam,gamma,ylam,r,R);
        double resid = 0.0; for (double v: r) resid = std::max(resid, std::abs(v));
        for (int it=0; it<dual_max_iter && resid>dual_tol; ++it) {
            std::vector<double> trial(m);
            for (int i=0;i<m;++i) trial[i] = lam[i] + alpha*r[i];
            ProjectDual(trial,a_pen,c_pen,d_pen,1.0);
            std::vector<double> g2,y2,r2; double R2;
            EvalResidual(trial,g2,y2,r2,R2);
            double resid2=0.0; for (double v: r2) resid2=std::max(resid2,std::abs(v));
            int bt=0;
            while (resid2 > resid && bt<20) {
                alpha *= 0.5;
                for (int i=0;i<m;++i) trial[i] = lam[i] + alpha*r[i];
                ProjectDual(trial,a_pen,c_pen,d_pen,1.0);
                EvalResidual(trial,g2,y2,r2,R2);
                resid2=0.0; for (double v: r2) resid2=std::max(resid2,std::abs(v));
                ++bt;
            }
            lam = trial; gamma=g2; ylam=y2; r=r2; resid=resid2; R=R2;
            alpha = std::min(alpha*1.2, 1e8);
        }
        RecoverZY(gamma,a_pen,c_pen,d_pen,z,y);
    };

    // ── BarrierNewton (Sec. 6.8): log-barrier Newton on the dual feasible
    //    set D, mu reduced geometrically like detail::SolveDualIP's epsi. ──
    auto SolveNewton = [&]() {
        for (int i=0;i<m;++i) lam[i] = std::max(lam[i], 1.0);
        double mu_bar = *std::max_element(lam.begin(),lam.end());
        const double a0 = 1.0;
        std::vector<double> gamma,ylam,r; double R;
        for (;;) {
            int inner=0; double err=1e300;
            while (err > 0.9*mu_bar && inner < 60) {
                ++inner;
                EvalResidual(lam,gamma,ylam,r,R);
                std::vector<double> hess; AssembleHessian(lam,R,hess);
                double atl=0.0; for (int i=0;i<m;++i) atl += a_pen[i]*lam[i];
                double denom = a0-atl; if (denom<1e-12) denom=1e-12;
                std::vector<double> grad(m);
                for (int i=0;i<m;++i) {
                    grad[i] = r[i] + mu_bar/lam[i] - mu_bar*a_pen[i]/denom;
                    if (d_pen[i]==0.0) {
                        double denc = c_pen[i]-lam[i]; if (denc<1e-12) denc=1e-12;
                        grad[i] -= mu_bar/denc;
                    }
                }
                std::vector<double> K = hess;
                for (int i=0;i<m;++i) K[i*m+i] -= mu_bar/(lam[i]*lam[i]);
                for (int rI=0;rI<m;++rI) for (int cI=0;cI<m;++cI)
                    K[rI*m+cI] -= mu_bar*a_pen[rI]*a_pen[cI]/(denom*denom);
                for (int i=0;i<m;++i) if (d_pen[i]==0.0) {
                    double denc = c_pen[i]-lam[i]; if (denc<1e-12) denc=1e-12;
                    K[i*m+i] -= mu_bar/(denc*denc);
                }
                // Solve K*delta = -grad (K is negative (semi)definite here).
                std::vector<double> rhs(m); for (int i=0;i<m;++i) rhs[i]=-grad[i];
                SolveDense(K,rhs,m);
                const std::vector<double>& delta = rhs;
                // Backtrack to stay strictly feasible for all barrier args.
                double theta = 1.0;
                for (int i=0;i<m;++i) {
                    if (delta[i] < 0.0 && -1.01*delta[i] > theta*lam[i])
                        theta = lam[i] / (-1.01*delta[i]);
                    if (d_pen[i]==0.0 && delta[i] > 0.0) {
                        double denc = c_pen[i]-lam[i];
                        if (1.01*delta[i] > theta*denc) theta = denc/(1.01*delta[i]);
                    }
                }
                double ad=0.0; for (int i=0;i<m;++i) ad += a_pen[i]*delta[i];
                if (ad > 0.0 && 1.01*ad > theta*denom) theta = denom/(1.01*ad);
                theta = std::min(theta, 1.0);
                for (int i=0;i<m;++i) lam[i] += theta*delta[i];
                EvalResidual(lam,gamma,ylam,r,R);
                err=0.0; for (double v: r) err=std::max(err,std::abs(v));
                if (inner%25==0) mu_bar*=0.1;
            }
            mu_bar *= 0.1;
            if (mu_bar <= dual_tol) break;
        }
        EvalResidual(lam,gamma,ylam,r,R);
        RecoverZY(gamma,a_pen,c_pen,d_pen,z,y);
        for (int i=0;i<m;++i) mu[i]=mu_bar;
    };

    if (kind == DualSolverKind::ProjectedAscent) SolveProjected();
    else SolveNewton();

    // Write the accepted eta_trial/x_trial (consistent with the FINAL lam)
    // back into the caller's device buffers.
    {
        double R; BuildC(lam,R);
        EntropyPrimalUpdate(use_dev,n_loc,bkind,lo_r,hi_r,h_r,eta_c_loc,Cvec,R,
                            eta_trial_loc,x_trial_loc,nullptr,nullptr);
    }
}

} // namespace detail


// ============================================================
// CCSAOptimizer  (serial)
// ============================================================

CCSAOptimizer::CCSAOptimizer(int n, int m,
                              const BoundsGeometry* bounds,
                              const double* a, const double* c, const double* d)
    : n_(n), m_(m)
    , lam_(m,1.0), mu_(m,1.0), y_(m,0.0), rho_(m+1,1e-5)
{
    if (a && c && d) { a_.assign(a,a+m); c_.assign(c,c+m); d_.assign(d,d+m); }
    else DefaultPenalty(n,m,a_,c_,d_);

    if (bounds) bounds_ = *bounds;
}

CCSAOptimizer::CCSAOptimizer(int n, int m)
    : CCSAOptimizer(n,m,nullptr,nullptr,nullptr,nullptr) {}

CCSAOptimizer::CCSAOptimizer(int n, int m,
                              const double* a, const double* c, const double* d)
    : CCSAOptimizer(n,m,nullptr,a,c,d) {}

CCSAOptimizer::CCSAOptimizer(int n, int m,
                              const mfem::Vector& a, const mfem::Vector& c, const mfem::Vector& d)
    : CCSAOptimizer(n,m,nullptr,
                    VecToDouble(a).data(), VecToDouble(c).data(), VecToDouble(d).data()) {}

CCSAOptimizer::CCSAOptimizer(int n, int m, const BoundsGeometry& bounds)
    : CCSAOptimizer(n,m,&bounds,nullptr,nullptr,nullptr) {}

CCSAOptimizer::CCSAOptimizer(int n, int m, const BoundsGeometry& bounds,
                              const double* a, const double* c, const double* d)
    : CCSAOptimizer(n,m,&bounds,a,c,d) {}

CCSAOptimizer::CCSAOptimizer(int n, int m, const BoundsGeometry& bounds,
                              const mfem::Vector& a, const mfem::Vector& c, const mfem::Vector& d)
    : CCSAOptimizer(n,m,&bounds,
                    VecToDouble(a).data(), VecToDouble(c).data(), VecToDouble(d).data()) {}

void CCSAOptimizer::SetBounds(const BoundsGeometry& bounds)
{ bounds_ = bounds; }

void CCSAOptimizer::SetBounds(const mfem::Vector& xmin, const mfem::Vector& xmax)
{ SetBounds(BoundsGeometry::TwoSided(xmin,xmax)); }

void CCSAOptimizer::RequireBounds_() const
{
    if (!bounds_.IsSet())
        throw std::runtime_error(
            "CCSAOptimizer: no BoundsGeometry set. Call SetBounds() or construct "
            "with an explicit BoundsGeometry before Update()/UpdateGCMMA()/KKTresidual().");
}

void CCSAOptimizer::SetRhoParams(double rho_min, double gamma_safe, double gamma_max, double theta_decrease)
{ rho_min_=rho_min; rho_safe_=gamma_safe; rho_max_growth_=gamma_max; theta_rho_=theta_decrease; }

CCSAOptimizer CCSAOptimizer::WithEqualities(int n, int n_ineq, int n_eq)
{
    CCSAOptimizer o(n, n_ineq+2*n_eq); o.n_eq_ = n_eq;
    for (int i=n_ineq;i<n_ineq+2*n_eq;++i) { o.c_[i]=1e30; o.lam_[i]=1e-3; o.mu_[i]=1e-3; }
    return o;
}
CCSAOptimizer CCSAOptimizer::WithEqualities(int n, int n_ineq, int n_eq,
                                             const BoundsGeometry& bounds)
{
    CCSAOptimizer o(n, n_ineq+2*n_eq, bounds); o.n_eq_ = n_eq;
    for (int i=n_ineq;i<n_ineq+2*n_eq;++i) { o.c_[i]=1e30; o.lam_[i]=1e-3; o.mu_[i]=1e-3; }
    return o;
}
CCSAOptimizer CCSAOptimizer::WithRelaxedEqualities(int n, int n_ineq, int n_eq)
{
    CCSAOptimizer o(n, n_ineq+2*n_eq); // n_eq_ stays 0: independent inequality slots
    for (int i=n_ineq;i<n_ineq+2*n_eq;++i) { o.c_[i]=1e4; o.lam_[i]=1e-3; o.mu_[i]=1e-3; }
    return o;
}
CCSAOptimizer CCSAOptimizer::WithRelaxedEqualities(int n, int n_ineq, int n_eq,
                                                    const BoundsGeometry& bounds)
{
    CCSAOptimizer o(n, n_ineq+2*n_eq, bounds);
    for (int i=n_ineq;i<n_ineq+2*n_eq;++i) { o.c_[i]=1e4; o.lam_[i]=1e-3; o.mu_[i]=1e-3; }
    return o;
}

/// @brief Default rho estimator: same gradient-magnitude heuristic as
/// MMAOptimizer::UpdateGCMMA() -- see the doc comment on the header
/// declaration for the planned SiMPL/BB extension point.
void CCSAOptimizer::ComputeInitialRho_(const mfem::Vector& df0dx, const mfem::Vector* dfidx,
                                        std::vector<double>& rho_out) const
{
    // Uses the frozen bounds geometry's own width (h_j), not xmax-xmin —
    // see the doc comment on the header declaration.
    bool ud = df0dx.UseDevice();
    rho_out.assign(m_+1, 0.0);
    mfem::Vector d_tmp(n_); d_tmp.UseDevice(ud);
    const auto* hr = bounds_.Scale().Read();
    {
        const auto* df0r=df0dx.Read();
        auto* dt=d_tmp.Write();
        mfem::forall_switch(ud,n_,[=] MFEM_HOST_DEVICE (int j){
            double v=double(df0r[j]); if(v<0)v=-v;
            dt[j]=v*double(hr[j]);
        });
        rho_out[0]=d_tmp.Sum();
    }
    for (int i=0;i<m_;++i) {
        const auto* dfir=dfidx[i].Read();
        auto* dt=d_tmp.Write();
        mfem::forall_switch(ud,n_,[=] MFEM_HOST_DEVICE (int j){
            double v=double(dfir[j]); if(v<0)v=-v;
            dt[j]=v*double(hr[j]);
        });
        rho_out[i+1]=d_tmp.Sum();
    }
    for (int k=0;k<=m_;++k) rho_out[k]=std::max(rho_min_, 0.5/(double)n_*rho_out[k]);
}

void CCSAOptimizer::DecayRho_()
{
    for (int k=0;k<=m_;++k) rho_[k] = std::max(rho_min_, theta_rho_*rho_[k]);
}

void CCSAOptimizer::Update(mfem::Vector& x, const mfem::Vector& df0dx, mfem::real_t f0val,
                            const mfem::Vector& fival, const mfem::Vector* dfidx)
{
    RequireBounds_();
    bool ud = x.UseDevice();

    std::vector<double> F(m_+1); F[0]=double(f0val);
    for (int i=0;i<m_;++i) F[i+1]=double(fival(i));
    // rho_ is seeded ONCE via the gradient-magnitude heuristic and then
    // PERSISTS across calls (grown by the callback conservatism loop in
    // UpdateGCMMA, decayed by DecayRho_()) -- it must NOT be re-derived
    // from scratch every call, or all of that accumulated
    // conservatism/decay information is discarded and rho_ never adapts.
    if (!have_rho_) { ComputeInitialRho_(df0dx, dfidx, rho_); have_rho_ = true; }

    std::vector<mfem::Vector> bi(m_);
    for (int i=0;i<m_;++i) bi[i] = dfidx[i];

    mfem::Vector eta_trial(n_), x_trial(n_);
    eta_trial.UseDevice(ud); x_trial.UseDevice(ud);

    // x IS the latent center eta^k on entry -- see @ref latent in the header.
    detail::SolveDualEntropy(dual_solver_, CCSA_SERIAL_COMM, n_, m_, n_eq_, ud,
        bounds_, x.Read(), df0dx, bi, F, rho_, a_, c_, d_,
        lam_, mu_, y_, z_, dual_tol_, dual_max_iter_,
        eta_trial.Write(), x_trial.Write());

    eta_prev_ = x; df0dx_prev_ = df0dx; have_prev_ = true;
    x = eta_trial;   // latent out; x_trial (physical) is discarded here
    ++iter_;
}

void CCSAOptimizer::UpdateGCMMA(mfem::Vector& x, const mfem::Vector& df0dx, mfem::real_t f0val,
                                 const mfem::Vector& fival, const mfem::Vector* dfidx, int* innerIter)
{
    RequireBounds_();
    if (!have_rho_) { ComputeInitialRho_(df0dx, dfidx, rho_); have_rho_ = true; }

    bool ud = x.UseDevice();
    std::vector<double> F(m_+1); F[0]=double(f0val);
    for (int i=0;i<m_;++i) F[i+1]=double(fival(i));
    std::vector<mfem::Vector> bi(m_);
    for (int i=0;i<m_;++i) bi[i] = dfidx[i];

    mfem::Vector eta_trial(n_), x_trial(n_);
    eta_trial.UseDevice(ud); x_trial.UseDevice(ud);

    detail::SolveDualEntropy(dual_solver_, CCSA_SERIAL_COMM, n_, m_, n_eq_, ud,
        bounds_, x.Read(), df0dx, bi, F, rho_, a_, c_, d_,
        lam_, mu_, y_, z_, dual_tol_, dual_max_iter_,
        eta_trial.Write(), x_trial.Write());

    if (innerIter) *innerIter = 1;
    eta_prev_ = x; df0dx_prev_ = df0dx; have_prev_ = true;
    x = eta_trial;
    DecayRho_();
    ++iter_;
}

void CCSAOptimizer::UpdateGCMMA(mfem::Vector& x, const mfem::Vector& df0dx, mfem::real_t f0val,
                                 const mfem::Vector& fival, const mfem::Vector* dfidx,
                                 EvalCallback eval_fi, int max_inner, int* innerIter)
{
    RequireBounds_();
    if (!have_rho_) { ComputeInitialRho_(df0dx, dfidx, rho_); have_rho_ = true; }

    bool ud = x.UseDevice();
    std::vector<double> F(m_+1); F[0]=double(f0val);
    for (int i=0;i<m_;++i) F[i+1]=double(fival(i));
    std::vector<mfem::Vector> bi(m_);
    for (int i=0;i<m_;++i) bi[i] = dfidx[i];

    const mfem::Vector eta_c = x; // fixed anchor for the whole rho-inner loop
    int nu=0;
    for (; nu<max_inner; ++nu) {
        mfem::Vector eta_trial(n_), x_trial(n_);
        eta_trial.UseDevice(ud); x_trial.UseDevice(ud);

        detail::SolveDualEntropy(dual_solver_, CCSA_SERIAL_COMM, n_, m_, n_eq_, ud,
            bounds_, eta_c.Read(), df0dx, bi, F, rho_, a_, c_, d_,
            lam_, mu_, y_, z_, dual_tol_, dual_max_iter_,
            eta_trial.Write(), x_trial.Write());

        mfem::Vector fi_hat(m_); mfem::real_t f0_hat=0;
        if (eval_fi) eval_fi(x_trial, fi_hat, f0_hat);   // physical trial point

        // Conservatism check ([Note] eq. 32-33): compare true f_i(x_trial)
        // against the model g_i(x_trial;rho_i) = F[i]+<b_i,dx>+rho_i*W.
        // We recompute the model value cheaply from the accepted (lam_,
        // dual-consistent) primal rather than re-deriving W explicitly: the
        // dual solve already guarantees gamma_i (stored implicitly via the
        // last EvalResidual call) equals g_i(x_trial;rho_i); to keep this
        // routine self-contained we recompute the divergence term directly.
        bool conservative = true;
        if (eval_fi) {
            mfem::Vector x_c = bounds_.ToPhysical(eta_c);
            mfem::Vector dx(n_); dx.UseDevice(ud);
            {
                const auto* xt=x_trial.Read(); const auto* xc2=x_c.Read();
                auto* dxw=dx.Write();
                mfem::forall_switch(ud,n_,[=] MFEM_HOST_DEVICE (int j){ dxw[j]=xt[j]-xc2[j]; });
            }
            mfem::Vector eta_diff(n_); eta_diff.UseDevice(ud);
            detail::BregmanTerm(ud, n_, bounds_.Kind(), eta_trial, eta_c, eta_diff);
            // W is a Bregman divergence and is provably >= 0 mathematically
            // (a convex function's divergence from itself). When the step
            // is very small (eta_trial ~ eta_c -- which happens precisely
            // when rho has already grown large), computing it as a
            // difference of nearly-equal quantities can round to a tiny
            // NEGATIVE value purely from floating-point cancellation, even
            // though the true value is ~0. Left unclamped, that negative W
            // multiplied by an already-large rho produces a wildly wrong
            // (hugely negative) model prediction, which falsely triggers
            // "not conservative" and, since a negative W also fails the
            // W>1e-300 guard below, an unconditional rho*=10 -- compounding
            // every time this artifact recurs and driving rho toward
            // astronomical values for no real reason. Clamp it.
            double W = std::max(0.0, eta_diff.Sum());
            for (int i=0;i<m_ && conservative;++i) {
                double model = F[i+1] + mfem::InnerProduct(bi[i],dx) + rho_[i+1]*W;
                if (double(fi_hat(i)) > model) conservative = false;
            }
            double model0 = F[0] + mfem::InnerProduct(df0dx,dx) + rho_[0]*W;
            if (double(f0_hat) > model0) conservative = false;

            if (!conservative) {
                // rho_min_ already floors rho from below; cap it from
                // above too. Without this, repeated retries with a
                // near-zero Bregman term W (W<=1e-300, or even W just
                // small enough that e/W is astronomically large) can
                // drive rho toward double overflow (+Inf) over enough
                // iterations. Once rho is Inf, rho*W with a genuinely
                // zero W gives Inf*0 = NaN, and NaN comparisons are always
                // false in IEEE754 -- so the very next conservatism check
                // would silently report "conservative" for a corrupted
                // value instead of catching it. 1e100 is astronomically
                // larger than any rho this algorithm should legitimately
                // need, while leaving 200+ orders of magnitude of safety
                // margin below the actual overflow point.
                static constexpr double kRhoCeiling = 1e100;
                for (int i=0;i<m_;++i) {
                    double e = double(fi_hat(i)) - (F[i+1] + mfem::InnerProduct(bi[i],dx) + rho_[i+1]*W);
                    if (e > 0.0 && W > 1e-300)
                        rho_[i+1] = std::min(kRhoCeiling, std::min(rho_max_growth_*rho_[i+1], rho_safe_*(rho_[i+1]+e/W)));
                    else if (e > 0.0)
                        rho_[i+1] = std::min(kRhoCeiling, rho_max_growth_*rho_[i+1]);
                }
                double e0 = double(f0_hat) - model0;
                if (e0 > 0.0 && W > 1e-300)
                    rho_[0] = std::min(kRhoCeiling, std::min(rho_max_growth_*rho_[0], rho_safe_*(rho_[0]+e0/W)));
                else if (e0 > 0.0)
                    rho_[0] = std::min(kRhoCeiling, rho_max_growth_*rho_[0]);
                continue;
            }
        }
        eta_prev_ = x; df0dx_prev_ = df0dx; have_prev_ = true;
        x = eta_trial;
        break;
    }
    if (innerIter) *innerIter = std::min(nu+1,max_inner);
    DecayRho_();
    ++iter_;
}

mfem::real_t CCSAOptimizer::KKTresidual(const mfem::Vector& x, const mfem::Vector& df0dx,
                                          mfem::real_t, const mfem::Vector& fival,
                                          const mfem::Vector* dfidx,
                                          double* lambda_out) const
{
    RequireBounds_();
    if (lambda_out) std::copy(lam_.begin(),lam_.end(),lambda_out);
    bool ud = x.UseDevice();

    // Curvature factor at the CURRENT latent point (no dual solve needed).
    mfem::Vector kappa(n_); kappa.UseDevice(ud);
    detail::ComputeKappa(ud, n_, bounds_.Kind(), x.Read(), kappa);

    mfem::Vector d_tmp(n_); d_tmp.UseDevice(ud);
    {
        const auto* df0r=df0dx.Read();
        auto* dt=d_tmp.Write();
        mfem::forall_switch(ud,n_,[=] MFEM_HOST_DEVICE (int j){ dt[j]=df0r[j]; });
        for (int i=0;i<m_-2*n_eq_;++i) {
            double li=lam_[i]; const auto* dfir=dfidx[i].Read();
            auto* dtr=d_tmp.ReadWrite();
            mfem::forall_switch(ud,n_,[=] MFEM_HOST_DEVICE (int j){ dtr[j]+=li*double(dfir[j]); });
        }
        for (int k=0;k<n_eq_;++k) {
            const int ni=m_-2*n_eq_;
            double lnet=lam_[ni+k]-lam_[ni+n_eq_+k];
            const auto* dfir=dfidx[ni+k].Read();
            auto* dtr=d_tmp.ReadWrite();
            mfem::forall_switch(ud,n_,[=] MFEM_HOST_DEVICE (int j){ dtr[j]+=lnet*double(dfir[j]); });
        }
        // Latent-space stationarity: dL/deta_j = (dL/dx_j) * h_j * kappa_j.
        // No projection needed -- eta ranges over all of R^n by
        // construction, so there is no boundary to project against; the
        // h_j*kappa_j factor naturally -> 0 as x_j saturates a bound,
        // exactly reproducing what the old physical-space projection did
        // by explicit clipping (see @ref latent in the header).
        const auto* hr = bounds_.Scale().Read();
        const auto* kr = kappa.Read();
        mfem::forall_switch(ud,n_,[=] MFEM_HOST_DEVICE (int j){
            double g_eta = double(dt[j]) * double(hr[j]) * double(kr[j]);
            dt[j] = mfem::real_t(g_eta*g_eta);
        });
    }
    double primal = d_tmp.Sum();
    double dual = 0.0;
    for (int i=0;i<m_-2*n_eq_;++i) { double cs=lam_[i]*double(fival(i)); dual+=cs*cs; }
    for (int k=0;k<n_eq_;++k) {
        const int ni=m_-2*n_eq_;
        double vp=double(fival(ni+k)), vn=double(fival(ni+n_eq_+k));
        if (vp>0) dual+=vp*vp;
        if (vn>0) dual+=vn*vn;
    }
    return (primal+dual)/(double)n_;
}


#ifdef MFEM_USE_MPI
// ============================================================
// CCSAOptimizerParallel
// ============================================================

CCSAOptimizerParallel::CCSAOptimizerParallel(MPI_Comm comm, int n_local, int m,
                                              const BoundsGeometry* bounds_local,
                                              const double* a, const double* c, const double* d)
    : comm_(comm), n_local_(n_local), m_(m)
    , lam_(m,1.0), mu_(m,1.0), y_(m,0.0), rho_(m+1,1e-5)
{
    n_global_ = ComputeNGlobal(comm_, n_local_);
    if (a && c && d) { a_.assign(a,a+m); c_.assign(c,c+m); d_.assign(d,d+m); }
    else DefaultPenalty((int)n_global_, m, a_, c_, d_);

    if (bounds_local) bounds_ = *bounds_local;
}

CCSAOptimizerParallel::CCSAOptimizerParallel(MPI_Comm comm, int n_local, int m)
    : CCSAOptimizerParallel(comm,n_local,m,nullptr,nullptr,nullptr,nullptr) {}

CCSAOptimizerParallel::CCSAOptimizerParallel(MPI_Comm comm, int n_local, int m,
                                              const double* a, const double* c, const double* d)
    : CCSAOptimizerParallel(comm,n_local,m,nullptr,a,c,d) {}

CCSAOptimizerParallel::CCSAOptimizerParallel(MPI_Comm comm, int n_local, int m,
                                              const mfem::Vector& a, const mfem::Vector& c, const mfem::Vector& d)
    : CCSAOptimizerParallel(comm,n_local,m,nullptr,
                            VecToDouble(a).data(),VecToDouble(c).data(),VecToDouble(d).data()) {}

CCSAOptimizerParallel::CCSAOptimizerParallel(MPI_Comm comm, int n_local, int m,
                                              const BoundsGeometry& bounds_local)
    : CCSAOptimizerParallel(comm,n_local,m,&bounds_local,nullptr,nullptr,nullptr) {}

CCSAOptimizerParallel::CCSAOptimizerParallel(MPI_Comm comm, int n_local, int m,
                                              const BoundsGeometry& bounds_local,
                                              const double* a, const double* c, const double* d)
    : CCSAOptimizerParallel(comm,n_local,m,&bounds_local,a,c,d) {}

CCSAOptimizerParallel::CCSAOptimizerParallel(MPI_Comm comm, int n_local, int m,
                                              const BoundsGeometry& bounds_local,
                                              const mfem::Vector& a, const mfem::Vector& c, const mfem::Vector& d)
    : CCSAOptimizerParallel(comm,n_local,m,&bounds_local,
                            VecToDouble(a).data(),VecToDouble(c).data(),VecToDouble(d).data()) {}

void CCSAOptimizerParallel::SetBounds(const BoundsGeometry& bounds_local)
{ bounds_ = bounds_local; }

void CCSAOptimizerParallel::SetBounds(const mfem::Vector& xmin_local, const mfem::Vector& xmax_local)
{ SetBounds(BoundsGeometry::TwoSided(xmin_local,xmax_local)); }

void CCSAOptimizerParallel::RequireBounds_() const
{
    if (!bounds_.IsSet())
        throw std::runtime_error(
            "CCSAOptimizerParallel: no BoundsGeometry set. Call SetBounds() or construct "
            "with an explicit BoundsGeometry before Update()/UpdateGCMMA()/KKTresidual().");
}

void CCSAOptimizerParallel::SetRhoParams(double rho_min, double gamma_safe, double gamma_max, double theta_decrease)
{ rho_min_=rho_min; rho_safe_=gamma_safe; rho_max_growth_=gamma_max; theta_rho_=theta_decrease; }

CCSAOptimizerParallel CCSAOptimizerParallel::WithEqualities(MPI_Comm comm, int n_local,
                                                             int n_ineq, int n_eq)
{
    CCSAOptimizerParallel o(comm,n_local,n_ineq+2*n_eq); o.n_eq_=n_eq;
    for (int i=n_ineq;i<n_ineq+2*n_eq;++i) { o.c_[i]=1e30; o.lam_[i]=1e-3; o.mu_[i]=1e-3; }
    return o;
}
CCSAOptimizerParallel CCSAOptimizerParallel::WithEqualities(MPI_Comm comm, int n_local,
                                                             int n_ineq, int n_eq,
                                                             const BoundsGeometry& bounds_local)
{
    CCSAOptimizerParallel o(comm,n_local,n_ineq+2*n_eq,bounds_local); o.n_eq_=n_eq;
    for (int i=n_ineq;i<n_ineq+2*n_eq;++i) { o.c_[i]=1e30; o.lam_[i]=1e-3; o.mu_[i]=1e-3; }
    return o;
}
CCSAOptimizerParallel CCSAOptimizerParallel::WithRelaxedEqualities(MPI_Comm comm, int n_local,
                                                                    int n_ineq, int n_eq)
{
    CCSAOptimizerParallel o(comm,n_local,n_ineq+2*n_eq);
    for (int i=n_ineq;i<n_ineq+2*n_eq;++i) { o.c_[i]=1e4; o.lam_[i]=1e-3; o.mu_[i]=1e-3; }
    return o;
}
CCSAOptimizerParallel CCSAOptimizerParallel::WithRelaxedEqualities(MPI_Comm comm, int n_local,
                                                                    int n_ineq, int n_eq,
                                                                    const BoundsGeometry& bounds_local)
{
    CCSAOptimizerParallel o(comm,n_local,n_ineq+2*n_eq,bounds_local);
    for (int i=n_ineq;i<n_ineq+2*n_eq;++i) { o.c_[i]=1e4; o.lam_[i]=1e-3; o.mu_[i]=1e-3; }
    return o;
}

void CCSAOptimizerParallel::ComputeInitialRho_(const mfem::Vector& df0dx_local, const mfem::Vector* dfidx_local,
                                                std::vector<double>& rho_out) const
{
    // Uses the frozen (local-chunk) bounds geometry's own width (h_j), not
    // xmax-xmin — see the doc comment on the header declaration.
    bool ud = df0dx_local.UseDevice();
    rho_out.assign(m_+1, 0.0);
    mfem::Vector d_tmp(n_local_); d_tmp.UseDevice(ud);
    const auto* hr = bounds_.Scale().Read();
    std::vector<double> loc(m_+1,0.0);
    {
        const auto* df0r=df0dx_local.Read();
        auto* dt=d_tmp.Write();
        mfem::forall_switch(ud,n_local_,[=] MFEM_HOST_DEVICE (int j){
            double v=double(df0r[j]); if(v<0)v=-v;
            dt[j]=v*double(hr[j]);
        });
        loc[0]=d_tmp.Sum();
    }
    for (int i=0;i<m_;++i) {
        const auto* dfir=dfidx_local[i].Read();
        auto* dt=d_tmp.Write();
        mfem::forall_switch(ud,n_local_,[=] MFEM_HOST_DEVICE (int j){
            double v=double(dfir[j]); if(v<0)v=-v;
            dt[j]=v*double(hr[j]);
        });
        loc[i+1]=d_tmp.Sum();
    }
    ccsa_Allreduce(loc.data(), rho_out.data(), m_+1, comm_);
    for (int k=0;k<=m_;++k) rho_out[k]=std::max(rho_min_, 0.5/(double)n_global_*rho_out[k]);
}

void CCSAOptimizerParallel::DecayRho_()
{ for (int k=0;k<=m_;++k) rho_[k] = std::max(rho_min_, theta_rho_*rho_[k]); }

void CCSAOptimizerParallel::Update(mfem::Vector& x_local, const mfem::Vector& df0dx_local, mfem::real_t f0val,
                                    const mfem::Vector& fival, const mfem::Vector* dfidx_local)
{
    RequireBounds_();
    bool ud = x_local.UseDevice();
    std::vector<double> F(m_+1); F[0]=double(f0val);
    for (int i=0;i<m_;++i) F[i+1]=double(fival(i));
    // See the serial Update()'s comment: rho_ is seeded ONCE and persists
    // across calls, it must NOT be re-derived from scratch every call.
    if (!have_rho_) { ComputeInitialRho_(df0dx_local, dfidx_local, rho_); have_rho_ = true; }

    std::vector<mfem::Vector> bi(m_);
    for (int i=0;i<m_;++i) bi[i] = dfidx_local[i];

    mfem::Vector eta_trial(n_local_), x_trial(n_local_);
    eta_trial.UseDevice(ud); x_trial.UseDevice(ud);

    detail::SolveDualEntropy(dual_solver_, comm_, n_local_, m_, n_eq_, ud,
        bounds_, x_local.Read(), df0dx_local, bi, F, rho_, a_, c_, d_,
        lam_, mu_, y_, z_, dual_tol_, dual_max_iter_,
        eta_trial.Write(), x_trial.Write());

    eta_prev_=x_local; df0dx_prev_=df0dx_local; have_prev_=true;
    x_local = eta_trial;
    ++iter_;
}

void CCSAOptimizerParallel::UpdateGCMMA(mfem::Vector& x_local, const mfem::Vector& df0dx_local, mfem::real_t f0val,
                                         const mfem::Vector& fival, const mfem::Vector* dfidx_local,
                                         int* innerIter)
{
    RequireBounds_();
    if (!have_rho_) { ComputeInitialRho_(df0dx_local, dfidx_local, rho_); have_rho_ = true; }

    bool ud = x_local.UseDevice();
    std::vector<double> F(m_+1); F[0]=double(f0val);
    for (int i=0;i<m_;++i) F[i+1]=double(fival(i));
    std::vector<mfem::Vector> bi(m_);
    for (int i=0;i<m_;++i) bi[i] = dfidx_local[i];

    mfem::Vector eta_trial(n_local_), x_trial(n_local_);
    eta_trial.UseDevice(ud); x_trial.UseDevice(ud);

    detail::SolveDualEntropy(dual_solver_, comm_, n_local_, m_, n_eq_, ud,
        bounds_, x_local.Read(), df0dx_local, bi, F, rho_, a_, c_, d_,
        lam_, mu_, y_, z_, dual_tol_, dual_max_iter_,
        eta_trial.Write(), x_trial.Write());

    if (innerIter) *innerIter=1;
    eta_prev_=x_local; df0dx_prev_=df0dx_local; have_prev_=true;
    x_local = eta_trial;
    DecayRho_();
    ++iter_;
}

void CCSAOptimizerParallel::UpdateGCMMA(mfem::Vector& x_local, const mfem::Vector& df0dx_local, mfem::real_t f0val,
                                         const mfem::Vector& fival, const mfem::Vector* dfidx_local,
                                         EvalCallback eval_fi, int max_inner, int* innerIter)
{
    RequireBounds_();
    if (!have_rho_) { ComputeInitialRho_(df0dx_local, dfidx_local, rho_); have_rho_ = true; }

    bool ud = x_local.UseDevice();
    std::vector<double> F(m_+1); F[0]=double(f0val);
    for (int i=0;i<m_;++i) F[i+1]=double(fival(i));
    std::vector<mfem::Vector> bi(m_);
    for (int i=0;i<m_;++i) bi[i] = dfidx_local[i];

    const mfem::Vector eta_c = x_local;
    int nu=0;
    for (; nu<max_inner; ++nu) {
        mfem::Vector eta_trial(n_local_), x_trial(n_local_);
        eta_trial.UseDevice(ud); x_trial.UseDevice(ud);

        detail::SolveDualEntropy(dual_solver_, comm_, n_local_, m_, n_eq_, ud,
            bounds_, eta_c.Read(), df0dx_local, bi, F, rho_, a_, c_, d_,
            lam_, mu_, y_, z_, dual_tol_, dual_max_iter_,
            eta_trial.Write(), x_trial.Write());

        // eval_fi is called with the LOCAL trial chunk (physical); the
        // caller (as with MMAOptimizerParallel) is responsible for
        // producing globally consistent fi_hat/f0_hat (e.g. via its own
        // MPI_Allreduce) exactly as documented for
        // MMAOptimizerParallel::UpdateGCMMA().
        mfem::Vector fi_hat(m_); mfem::real_t f0_hat=0;
        if (eval_fi) eval_fi(x_trial, fi_hat, f0_hat);

        bool conservative = true;
        if (eval_fi) {
            mfem::Vector x_c = bounds_.ToPhysical(eta_c);
            mfem::Vector dx(n_local_); dx.UseDevice(ud);
            {
                const auto* xt=x_trial.Read(); const auto* xc2=x_c.Read();
                auto* dxw=dx.Write();
                mfem::forall_switch(ud,n_local_,[=] MFEM_HOST_DEVICE (int j){ dxw[j]=xt[j]-xc2[j]; });
            }
            mfem::Vector eta_diff(n_local_); eta_diff.UseDevice(ud);
            detail::BregmanTerm(ud, n_local_, bounds_.Kind(), eta_trial, eta_c, eta_diff);
            // See the identical comment in the serial class's UpdateGCMMA():
            // W is a Bregman divergence, provably >= 0; a small/very-small
            // step can make its per-rank partial sum round to a tiny
            // negative value from pure floating-point cancellation. Clamp
            // both the per-rank contribution and the reduced total.
            double Wloc = std::max(0.0, eta_diff.Sum()); double W=0.0; ccsa_Allreduce(&Wloc,&W,1,comm_);
            W = std::max(0.0, W);
            std::vector<double> dxi_loc(m_), dxi(m_);
            for (int i=0;i<m_;++i) dxi_loc[i]=mfem::InnerProduct(bi[i],dx);
            // Zero-count MPI_Allreduce (m_==0, the unconstrained case) is
            // a no-op mathematically, but some MPI implementations handle
            // zero-count collectives across >1 real ranks poorly (this
            // exact call has been observed to hang a >1-rank run when
            // m_==0, while never being an issue for m_==0 in a 1-rank run,
            // where MPI_Allreduce degenerates to a trivial single-process
            // case regardless of count). Skip it entirely when there is
            // nothing to reduce.
            if (m_ > 0) ccsa_Allreduce(dxi_loc.data(),dxi.data(),m_,comm_);
            double dx0_loc = mfem::InnerProduct(df0dx_local,dx), dx0;
            ccsa_Allreduce(&dx0_loc,&dx0,1,comm_);

            for (int i=0;i<m_ && conservative;++i) {
                double model = F[i+1] + dxi[i] + rho_[i+1]*W;
                if (double(fi_hat(i)) > model) conservative=false;
            }
            double model0 = F[0] + dx0 + rho_[0]*W;
            if (double(f0_hat) > model0) conservative=false;

            if (!conservative) {
                // See the identical comment in the serial class's
                // UpdateGCMMA(): cap rho from above to prevent runaway
                // growth toward double overflow (and the resulting
                // Inf*0=NaN poisoning of the conservatism check).
                static constexpr double kRhoCeiling = 1e100;
                for (int i=0;i<m_;++i) {
                    double e = double(fi_hat(i)) - (F[i+1]+dxi[i]+rho_[i+1]*W);
                    if (e>0.0 && W>1e-300) rho_[i+1]=std::min(kRhoCeiling, std::min(rho_max_growth_*rho_[i+1], rho_safe_*(rho_[i+1]+e/W)));
                    else if (e>0.0) rho_[i+1]=std::min(kRhoCeiling, rho_max_growth_*rho_[i+1]);
                }
                double e0 = double(f0_hat)-model0;
                if (e0>0.0 && W>1e-300) rho_[0]=std::min(kRhoCeiling, std::min(rho_max_growth_*rho_[0], rho_safe_*(rho_[0]+e0/W)));
                else if (e0>0.0) rho_[0]=std::min(kRhoCeiling, rho_max_growth_*rho_[0]);
                continue;
            }
        }
        eta_prev_=x_local; df0dx_prev_=df0dx_local; have_prev_=true;
        x_local = eta_trial;
        break;
    }
    if (innerIter) *innerIter=std::min(nu+1,max_inner);
    DecayRho_();
    ++iter_;
}

mfem::real_t CCSAOptimizerParallel::KKTresidual(const mfem::Vector& x_local, const mfem::Vector& df0dx_local,
                                                  mfem::real_t, const mfem::Vector& fival,
                                                  const mfem::Vector* dfidx_local,
                                                  double* lambda_out) const
{
    RequireBounds_();
    if (lambda_out) std::copy(lam_.begin(),lam_.end(),lambda_out);
    bool ud = x_local.UseDevice();

    mfem::Vector kappa(n_local_); kappa.UseDevice(ud);
    detail::ComputeKappa(ud, n_local_, bounds_.Kind(), x_local.Read(), kappa);

    mfem::Vector d_tmp(n_local_); d_tmp.UseDevice(ud);
    {
        const auto* df0r=df0dx_local.Read();
        auto* dt=d_tmp.Write();
        mfem::forall_switch(ud,n_local_,[=] MFEM_HOST_DEVICE (int j){ dt[j]=df0r[j]; });
        for (int i=0;i<m_-2*n_eq_;++i) {
            double li=lam_[i]; const auto* dfir=dfidx_local[i].Read();
            auto* dtr=d_tmp.ReadWrite();
            mfem::forall_switch(ud,n_local_,[=] MFEM_HOST_DEVICE (int j){ dtr[j]+=li*double(dfir[j]); });
        }
        for (int k=0;k<n_eq_;++k) {
            const int ni=m_-2*n_eq_;
            double lnet=lam_[ni+k]-lam_[ni+n_eq_+k];
            const auto* dfir=dfidx_local[ni+k].Read();
            auto* dtr=d_tmp.ReadWrite();
            mfem::forall_switch(ud,n_local_,[=] MFEM_HOST_DEVICE (int j){ dtr[j]+=lnet*double(dfir[j]); });
        }
        // Latent-space stationarity, see the serial KKTresidual()'s comment.
        const auto* hr = bounds_.Scale().Read();
        const auto* kr = kappa.Read();
        mfem::forall_switch(ud,n_local_,[=] MFEM_HOST_DEVICE (int j){
            double g_eta = double(dt[j]) * double(hr[j]) * double(kr[j]);
            dt[j] = mfem::real_t(g_eta*g_eta);
        });
    }
    double primal_loc = d_tmp.Sum(); double primal=0.0;
    ccsa_Allreduce(&primal_loc,&primal,1,comm_);
    double dual=0.0;
    for (int i=0;i<m_-2*n_eq_;++i) { double cs=lam_[i]*double(fival(i)); dual+=cs*cs; }
    for (int k=0;k<n_eq_;++k) {
        const int ni=m_-2*n_eq_;
        double vp=double(fival(ni+k)), vn=double(fival(ni+n_eq_+k));
        if (vp>0) dual+=vp*vp;
        if (vn>0) dual+=vn*vn;
    }
    return mfem::real_t((primal+dual)/(double)n_global_);
}

#endif // MFEM_USE_MPI

} // namespace mfem_mma
