/**
 * MMA_MFEM.cpp  —  Device-aware MMA / SQ optimisers for MFEM
 *
 * ─── What this file implements ───────────────────────────────────────────
 *
 *  • detail::SolveDense        – m×m linear solve (LU + SVD fallback)
 *  • detail::SolveDualIP       – interior-point dual solver for MMA
 *  • detail::SolveDualSQ       – interior-point dual solver for SQ
 *  • UpdateAsymptotes          – adaptive MMA asymptote update rule
 *  • BuildCoeffs               – p/q coefficient construction for MMA/SQ
 *  • MMAOptimizer              – serial MMA/GCMMA optimiser
 *  • MMAOptimizerParallel      – MPI-parallel MMA/GCMMA optimiser
 *  • SQOptimizer               – serial SQ/GCMMA optimiser
 *  • SQOptimizerParallel       – MPI-parallel SQ/GCMMA optimiser
 *
 * ─── Device / GPU strategy ───────────────────────────────────────────────
 *
 *  The device flag is read from x.UseDevice() at each Update() call and
 *  propagated to all internal Vectors.  Nothing changes in the calling
 *  code; moving x to a GPU device is sufficient.
 *
 *  • All O(n) element-wise loops use mfem::forall_switch(use_dev, n, λ).
 *    On CPU this becomes a plain for-loop; on GPU it launches a kernel.
 *
 *  • Device memory is accessed exclusively through mfem::Vector::Read() /
 *    Write() / ReadWrite(), which drive MFEM's memory manager and avoid
 *    manual host↔device copies.
 *
 *  • The m×m dual Newton system (m = number of constraints, typically ≤ 100)
 *    is always assembled and solved on the host.  Reductions over the n
 *    design variables (inner products, sums) are performed on-device via
 *    mfem::InnerProduct / Vector::Sum(), bringing only m or m² scalars back
 *    to the host for the Newton solve.
 *
 *  • p0_, q0_, pij_[i], qij_[i] are mfem::Vector so they inherit the
 *    device flag of x and stay on the same memory space automatically.
 */

#include "MMA_MFEM.hpp"
#include <stdexcept>
#include <string>
#include <numeric>
#include <cmath>

// ─── LAPACK declarations ─────────────────────────────────────────────────────
//
// We use two LAPACK routines for the m×m Newton solve:
//
//   dgesv_   — standard LU factorisation (O(m³), fast).  Used as the first
//               attempt.  Fails (info != 0) when the matrix is singular or
//               near-singular, which happens for equality constraints whose
//               ±h Hessian is exactly rank-deficient.
//
//   dgelsd_  — minimum-norm least-squares via divide-and-conquer SVD.  Used
//               as a fallback when dgesv_ fails.  The rcond threshold is set
//               to sqrt(eps) ≈ 1.5 × 10⁻⁸ so that near-zero singular values
//               from rank-deficient equality Hessians are correctly treated
//               as zero rather than inverted (which would amplify noise by
//               factors of 10¹⁵).
//
extern "C" {
void dgesv_(int* n, int* nrhs, double* A, int* lda, int* ipiv,
            double* B, int* ldb, int* info);
void dgelsd_(int* m, int* n, int* nrhs,
             double* A, int* lda,
             double* B, int* ldb,
             double* s, double* rcond, int* rank,
             double* work, int* lwork, int* iwork, int* info);
}

namespace mfem_mma {

// ─── Small internal helpers ───────────────────────────────────────────────────

// Pick the right MPI type for mfem::real_t (double or float build).
static MPI_Datatype MpiTypeReal()
{ return (sizeof(mfem::real_t)==sizeof(double)) ? MPI_DOUBLE : MPI_FLOAT; }

// Convert an mfem::Vector (real_t) to a host std::vector<double>.
// Used when penalty parameters are supplied as mfem::Vector.
static std::vector<double> VecToDouble(const mfem::Vector& v)
{
    const mfem::real_t* h = v.HostRead();
    return std::vector<double>(h, h + v.Size());
}

// Set the standard MMA penalty parameters:
//   a = 0   (no linear objective term in the dual)
//   c = max(1000, 10·n)  (must exceed the expected Lagrange multipliers;
//                          for topology optimisation λ* ~ n / Vfrac²)
//   d = 1   (standard scaling)
static void DefaultPenalty(int n_global, int m,
                            std::vector<double>& a,
                            std::vector<double>& c,
                            std::vector<double>& d)
{
    a.assign(m, 0.0);
    c.assign(m, std::max(1000.0, 10.0*n_global));
    d.assign(m, 1.0);
}


/**
 * @brief Compute the global DOF count from the local count via MPI_Allreduce.
 * Called once in each MMAOptimizerParallel constructor.
 */
static int ComputeNGlobal(MPI_Comm comm, int n_local)
{
    int ng = 0;
    MPI_Allreduce(&n_local, &ng, 1, MPI_INT, MPI_SUM, comm);
    return ng;
}

// ── Allocate a device Vector matching another's memory type ──────────────
static mfem::Vector DeviceVector(int n, const mfem::Vector& ref)
{
    mfem::Vector v(n);
    v.UseDevice(ref.UseDevice());
    v = 0.0;
    return v;
}

namespace detail {

// ─────────────────────────────────────────────────────────────────────────────
// SolveDense  —  m×m linear solve, LU with SVD fallback
// ─────────────────────────────────────────────────────────────────────────────
//
// Solves K · s = rhs, overwriting rhs with the solution.
//
// Two-stage strategy:
//
//  1. Try dgesv_ (LU).  Fast and exact for well-conditioned systems.
//     Returns immediately on success (info == 0).
//
//  2. If LU fails (singular or near-singular K), fall back to dgelsd_
//     (minimum-norm least-squares via SVD).  This handles the case where K is
//     exactly rank-deficient, which occurs for equality-constraint Hessians:
//     each ±h pair contributes a 2×2 block [[a,−a],[−a,a]] with a zero
//     eigenvalue in direction [1,1].  The minimum-norm SVD solution correctly
//     projects out that null direction instead of inverting it.
//
//     rcond = sqrt(eps) ≈ 1.5×10⁻⁸ is the singular-value threshold.  The
//     default (rcond = 2.2×10⁻¹⁶ · m) would treat numerical noise (~10⁻¹⁵)
//     as a true singular value and invert it, producing steps of size 10¹⁵.
//
void SolveDense(std::vector<double>& K, std::vector<double>& rhs, int m)
{
    if (m == 0) return;
    if (m == 1) {
        if (std::abs(K[0]) < 1e-300) K[0] = 1e-300;
        rhs[0] /= K[0]; return;
    }
    // Fast path: LU
    {
        int nrhs=1, info=0;
        std::vector<int>    ipiv(m);
        std::vector<double> K_lu(K), rhs_lu(rhs);
        dgesv_(&m,&nrhs,K_lu.data(),&m,ipiv.data(),rhs_lu.data(),&m,&info);
        if (info==0) { rhs=rhs_lu; return; }
        if (info<0)
            throw std::runtime_error(
                "MMA_MFEM: dgesv bad arg (info="+std::to_string(info)+")");
    }
    // Fallback: minimum-norm least-squares via SVD (redundant constraints)
    {
        int nrhs=1, info=0, rank=0;
        // Use sqrt(eps)*max_singular_value as the threshold so that near-zero
        // singular values from rank-deficient equality constraint Hessians are
        // correctly treated as zero.  The default 2.2e-16*m is too tight and
        // causes catastrophic null-space amplification for the ±h encoding.
        double rcond = std::sqrt(2.2e-16);
        std::vector<double> svals(m), K_svd(K);
        // Work with a copy of rhs so the workspace-query call (lwork=-1) cannot
        // corrupt the input; copy the solution back afterwards.
        std::vector<double> rhs_svd(rhs);
        int nlvl   = std::max(0,(int)std::ceil(std::log2(double(m)/25.0+1.0))+1);
        int liwork = std::max(1, 3*m*nlvl+11*m);
        std::vector<int> iwork(liwork);
        int lwork = -1; double wq;
        dgelsd_(&m,&m,&nrhs,K_svd.data(),&m,rhs_svd.data(),&m,
                svals.data(),&rcond,&rank,&wq,&lwork,iwork.data(),&info);
        lwork = (info==0)?(int)wq:10*m*m;
        lwork = std::max(lwork,1);
        std::vector<double> work(lwork);
        K_svd = K;
        rhs_svd = rhs;
        dgelsd_(&m,&m,&nrhs,K_svd.data(),&m,rhs_svd.data(),&m,
                svals.data(),&rcond,&rank,work.data(),&lwork,iwork.data(),&info);
        if (info == 0 || info > 0) {
            // info==0: full convergence; info>0: partial convergence but best
            // available solution is in rhs_svd.  Accept either.
            rhs = rhs_svd;
        }
        if (info < 0)
            throw std::runtime_error(
                "MMA_MFEM: dgelsd bad arg (info="+std::to_string(info)+")");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SolveDualIP  —  interior-point dual solver for the MMA subproblem
// ─────────────────────────────────────────────────────────────────────────────
//
// Solves the dual of the MMA separable rational subproblem using Svanberg's
// barrier method (Svanberg 1998, Algorithm 2.1).  The dual variables are the
// Lagrange multipliers λᵢ for the m constraints.
//
// The primal optimum x*(λ) has the closed form
//
//   x_j*(λ) = clip( (√p_j · L_j + √q_j · U_j) / (√p_j + √q_j),  α_j, β_j )
//
// where p_j = p0_j + Σ λᵢ·pᵢⱼ,  q_j = q0_j + Σ λᵢ·qᵢⱼ  (weighted p/q sums).
//
// The dual function W(λ) = Σⱼ f(x_j*(λ)) − bᵀλ is maximised using a
// Newton–barrier method with decreasing barrier parameter ε:
//
//   while ε > tol:
//       while ‖residual‖ > 0.9·ε:
//           assemble gradient and Hessian of W on-device → reduce to host
//           solve m×m Newton system (host)
//           line-search to keep λ,μ > 0
//       ε ← 0.1·ε
//
// All O(n) work (p/q sums, inner products) runs on-device via
// mfem::forall_switch.  Only m or m² scalars are brought back to the host
// for the Newton solve.
//
// Parameters labelled "_loc" are local to this MPI rank.  MPI_Allreduce is
// called on dual gradients, Hessians, and the residual (all O(m) or O(m²)).
//
void SolveDualIP(
    MPI_Comm comm,
    int n_loc, int m, int n_eq,
    bool use_dev,
    const mfem::real_t* L_loc,
    const mfem::real_t* U_loc,
    const mfem::real_t* alpha_loc,
    const mfem::real_t* beta_loc,
    const double*  p0_loc,
    const double*  q0_loc,
    const std::vector<mfem::Vector>& pij_loc,
    const std::vector<mfem::Vector>& qij_loc,
    const std::vector<double>& b,
    const std::vector<double>& a_pen,
    const std::vector<double>& c_pen,
    const std::vector<double>& d_pen,
    std::vector<double>& lam,
    std::vector<double>& mu,
    std::vector<double>& y,
    double& z,
    mfem::real_t* x_loc)
{
    // Inequality multipliers must start >=1 for the IP barrier.
    // Equality slots (last 2*n_eq entries) may start at any value — do not clamp.
    for (int i=0;i<m-2*n_eq;++i){ lam[i]=std::max(lam[i],1.0); mu[i]=std::max(mu[i],1.0); }
    for (int i=m-2*n_eq;i<m;++i){ mu[i]=std::max(mu[i],1.0); }  // mu still needs >0

    // ── m=0: closed-form primal (no dual system) ──────────────────────────
    if (m == 0) {
        mfem::forall_switch(use_dev, n_loc, [=] MFEM_HOST_DEVICE (int j) {
            double pjl = p0_loc[j], qjl = q0_loc[j];
            double xj  = (::sqrt(pjl)*double(L_loc[j]) +
                          ::sqrt(qjl)*double(U_loc[j])) /
                         (::sqrt(pjl) + ::sqrt(qjl));
            xj = xj < double(alpha_loc[j]) ? double(alpha_loc[j]) : xj;
            xj = xj > double(beta_loc[j])  ? double(beta_loc[j])  : xj;
            x_loc[j] = mfem::real_t(xj);
        });
        return;
    }

    // epsi warm-start: use max over all multipliers (including equality slots)
    // so the barrier parameter reflects the scale of the warm-started lam values.
    // No floor — if lam was initialised to small values (e.g. 1e-3) the IP
    // correctly starts with small epsi and reduces it to tol in a few steps.
    double lam_max = *std::max_element(lam.begin(), lam.end());
    double epsi    = lam_max;
    // tol must be the same on every rank so the Newton loop exits
    // in lockstep.  MPI_Allreduce n_loc to get n_global.
    double _n_global = 0.0;
    { double _nl = double(n_loc);
      MPI_Allreduce(&_nl, &_n_global, 1, MPI_DOUBLE, MPI_SUM, comm); }
    const double tol = std::sqrt(_n_global + double(m)) * 1e-9;

    // ── Temporary device Vectors for computations ─────────────────────────
    // We need per-iteration scratch: df2 (n), PQ_i (n each), tmp (n)
    // Allocate once outside the loop.
    mfem::Vector d_df2(n_loc), d_tmp(n_loc);
    d_df2.UseDevice(use_dev); d_tmp.UseDevice(use_dev);
    std::vector<mfem::Vector> d_PQ(m);  // PQ[:,i] for each constraint
    for (int i=0;i<m;++i) { d_PQ[i].SetSize(n_loc); d_PQ[i].UseDevice(use_dev); }

    // PrimalFromDual: given the current dual variables λ, compute x*(λ) on device.
    //
    // Steps:
    //   1. Project λ ≥ 0 (inequality slots only); update the slack y and z.
    //   2. Build the weighted sums p_j(λ) = p0_j + Σᵢ λᵢ·pᵢⱼ (in d_tmp)
    //      and q_j(λ) = q0_j + Σᵢ λᵢ·qᵢⱼ (in d_df2).
    //   3. Evaluate the primal optimum x_j* = (√p·L + √q·U)/(√p+√q),
    //      then clip to [α_j, β_j].
    //
    auto PrimalFromDual = [&]() {
        double lamai=0.0;
        for (int i=0;i<m;++i){
            lam[i]=std::max(lam[i],0.0);
            y[i]=std::max(0.0,lam[i]-c_pen[i]);
            lamai+=a_pen[i]*lam[i];
        }
        z = std::max(0.0, 10.0*(lamai-1.0));

        // Accumulate p_j(λ) into d_tmp and q_j(λ) into d_df2 on-device,
        // starting from p0/q0 and adding each pij*lam[i] term.
        auto* d_x     = x_loc;
        auto* d_Ld    = L_loc;
        auto* d_Ud    = U_loc;
        auto* d_ad    = alpha_loc;
        auto* d_bd    = beta_loc;
        auto* d_p0    = p0_loc;
        auto* d_q0    = q0_loc;

        // Copy p0, q0 into d_tmp, d_df2 (device copy via forall)
        {
            auto* pt = d_tmp.Write();
            auto* qt = d_df2.Write();
            mfem::forall_switch(use_dev, n_loc, [=] MFEM_HOST_DEVICE (int j){
                pt[j] = d_p0[j]; qt[j] = d_q0[j];
            });
        }
        // Accumulate pij*lam, qij*lam
        for (int i=0;i<m;++i){
            double li = lam[i];
            const auto* pij_dev = pij_loc[i].Read();
            auto* pt = d_tmp.ReadWrite();
            mfem::forall_switch(use_dev, n_loc, [=] MFEM_HOST_DEVICE (int j){
                pt[j] += li * pij_dev[j];
            });
        }
        for (int i=0;i<m;++i){
            double li = lam[i];
            const auto* qij_dev = qij_loc[i].Read();
            auto* qt = d_df2.ReadWrite();
            mfem::forall_switch(use_dev, n_loc, [=] MFEM_HOST_DEVICE (int j){
                qt[j] += li * qij_dev[j];
            });
        }

        // Compute xj from pjlam (d_tmp), qjlam (d_df2)
        const auto* pjl_r = d_tmp.Read();
        const auto* qjl_r = d_df2.Read();
        mfem::forall_switch(use_dev, n_loc, [=] MFEM_HOST_DEVICE (int j){
            double pl = pjl_r[j], ql = qjl_r[j];
            double sp = ::sqrt(pl), sq = ::sqrt(ql);
            double xj = (sp*double(d_Ld[j]) + sq*double(d_Ud[j])) / (sp + sq);
            xj = xj < double(d_ad[j]) ? double(d_ad[j]) : xj;
            xj = xj > double(d_bd[j]) ? double(d_bd[j]) : xj;
            d_x[j] = mfem::real_t(xj);
        });
    };

    // DualGrad: compute ∂W/∂λᵢ = Σⱼ[pᵢⱼ/(U−x) + qᵢⱼ/(x−L)] − bᵢ − aᵢz − yᵢ
    //
    // The summand is evaluated per-element on device, summed via d_tmp.Sum(),
    // then reduced across MPI ranks.
    //
    auto DualGrad = [&](std::vector<double>& grad) {
        std::vector<double> loc(m,0.0);
        for (int i=0;i<m;++i){
            const auto* pij_r = pij_loc[i].Read();
            const auto* qij_r = qij_loc[i].Read();
            const auto* d_x   = x_loc;
            const auto* d_L   = L_loc;
            const auto* d_U   = U_loc;
            auto* d_t = d_tmp.Write();
            mfem::forall_switch(use_dev, n_loc, [=] MFEM_HOST_DEVICE (int j){
                double dUx = double(d_U[j])-double(d_x[j]);
                double dxL = double(d_x[j])-double(d_L[j]);
                d_t[j] = pij_r[j]/dUx + qij_r[j]/dxL;
            });
            loc[i] = d_tmp.Sum();  // device-aware sum (cuBLAS dot on GPU)
        }
        MPI_Allreduce(loc.data(), grad.data(), m, MPI_DOUBLE, MPI_SUM, comm);
        for (int i=0;i<m;++i)
            grad[i] -= b[i] + a_pen[i]*z + y[i];
    };

    // DualHess: compute H_rc = Σⱼ (∂²W/∂λᵣ∂λ_c) for the Newton step.
    //
    // Using the envelope theorem on x*(λ):
    //
    //   H_rc = Σⱼ PQ_rj · (−1/f″_j) · PQ_cj
    //
    // where  PQ_ij = pij/(U−x)² − qij/(x−L)²
    // and    f″_j  =  2·pjλ/(U−x)³ + 2·qjλ/(x−L)³  (curvature of inner obj)
    //
    // Elements at a bound (x = α or β) contribute zero (primal is clamped,
    // so the second derivative through x*(λ) vanishes).
    //
    // Implementation: build df2_j = −1/f″_j and PQ_i·df2 in device Vectors,
    // then compute each H_rc = InnerProduct(PQ_r·df2, PQ_c) — a single dot
    // product on device.  All m² entries are reduced via MPI_Allreduce.
    //
    auto DualHess = [&](std::vector<double>& hess) {
        // First build df2 vector on device (stored in d_df2)
        const auto* d_x = x_loc;
        const auto* d_L = L_loc, *d_U = U_loc, *d_a = alpha_loc, *d_b = beta_loc;
        const auto* d_p0 = p0_loc, *d_q0 = q0_loc;

        // Recompute pjlam, qjlam into d_tmp, d_df2
        {
            auto* pt = d_tmp.Write();
            auto* qt = d_df2.Write();
            mfem::forall_switch(use_dev, n_loc, [=] MFEM_HOST_DEVICE (int j){
                pt[j] = d_p0[j]; qt[j] = d_q0[j];
            });
        }
        for (int i=0;i<m;++i){
            double li = lam[i];
            const auto* pij_r = pij_loc[i].Read();
            auto* pt = d_tmp.ReadWrite();
            mfem::forall_switch(use_dev, n_loc, [=] MFEM_HOST_DEVICE (int j){
                pt[j] += li * pij_r[j];
            });
        }
        for (int i=0;i<m;++i){
            double li = lam[i];
            const auto* qij_r = qij_loc[i].Read();
            auto* qt = d_df2.ReadWrite();
            mfem::forall_switch(use_dev, n_loc, [=] MFEM_HOST_DEVICE (int j){
                qt[j] += li * qij_r[j];
            });
        }
        // d_tmp = pjlam, d_df2 = qjlam
        // Compute df2 in-place in d_df2 reusing a new kernel
        // We need dUx, dxL from x. Use d_tmp temporarily first.
        // Strategy: write df2 into d_df2 overwriting qjlam (we have pjlam in d_tmp)
        {
            const auto* pjl = d_tmp.Read();
            auto* df2w = d_df2.ReadWrite();  // currently qjlam
            mfem::forall_switch(use_dev, n_loc, [=] MFEM_HOST_DEVICE (int j){
                double dUx = double(d_U[j])-double(d_x[j]);
                double dxL = double(d_x[j])-double(d_L[j]);
                double pl  = pjl[j];   // pjlam
                double ql  = df2w[j];  // qjlam
                double denom = 2.0*pl/(dUx*dUx*dUx) + 2.0*ql/(dxL*dxL*dxL);
                // Check if primal at bound (xfree outside [alpha,beta])
                double sp = ::sqrt(pl < 0 ? 0.0 : pl);
                double sq = ::sqrt(ql < 0 ? 0.0 : ql);
                double xfree = (sp+sq > 0) ?
                    (sp*double(d_L[j])+sq*double(d_U[j]))/(sp+sq) : double(d_x[j]);
                bool at_bound = xfree < double(d_a[j]) || xfree > double(d_b[j]);
                df2w[j] = at_bound ? 0.0 : -1.0/denom;
            });
        }
        // d_df2 = df2_j.  Now build PQ columns on device and compute outer product.
        // For each constraint pair (r,c): H_rc = Σ_j PQ_rj * df2_j * PQ_cj
        // Use d_tmp as scratch for PQ_r * df2_r
        std::vector<double> loc(m*m, 0.0);
        // We'll build PQ columns in d_PQ[i], then do InnerProduct pairs.
        // Build PQ[:,i]*df2 into d_PQ[i] (scratch for dot products)
        for (int i=0;i<m;++i){
            const auto* pij_r = pij_loc[i].Read();
            const auto* qij_r = qij_loc[i].Read();
            auto* pq_w = d_PQ[i].Write();
            const auto* df2_r = d_df2.Read();
            double li = lam[i];
            mfem::forall_switch(use_dev, n_loc, [=] MFEM_HOST_DEVICE (int j){
                double dUx = double(d_U[j])-double(d_x[j]);
                double dxL = double(d_x[j])-double(d_L[j]);
                double g = pij_r[j]/(dUx*dUx) - qij_r[j]/(dxL*dxL);
                pq_w[j] = g * df2_r[j];
            });
        }
        // H_rc = Σ_j (PQ_rj*df2_j) * PQ_cj = dot(d_PQ[r]*df2, PQ_c_raw)
        // But we stored PQ*df2 in d_PQ[r]. We need raw PQ_c too.
        // We need a second buffer for raw PQ columns. Use d_tmp for raw PQ_c.
        for (int r=0;r<m;++r){
            for (int c=r;c<m;++c){
                {
                    const auto* pij_c_d = pij_loc[c].Read();
                    const auto* qij_c_d = qij_loc[c].Read();
                    auto* pq_raw_c = d_tmp.Write();
                    mfem::forall_switch(use_dev, n_loc, [=] MFEM_HOST_DEVICE (int j){
                        double dUx = double(d_U[j])-double(d_x[j]);
                        double dxL = double(d_x[j])-double(d_L[j]);
                        pq_raw_c[j] = pij_c_d[j]/(dUx*dUx) - qij_c_d[j]/(dxL*dxL);
                    });
                }
                // Now: d_PQ[r] = PQ_r * df2,  d_tmp = PQ_c_raw
                double v = mfem::InnerProduct(d_PQ[r], d_tmp);
                loc[r*m+c] = v;
                if (r!=c) loc[c*m+r] = v;
            }
        }
        MPI_Allreduce(loc.data(), hess.data(), m*m, MPI_DOUBLE, MPI_SUM, comm);

        // Diagonal corrections (host-only)
        double lamai=0.0;
        for (int i=0;i<m;++i) lamai+=lam[i]*a_pen[i];
        for (int i=0;i<m;++i){
            if (lam[i]>c_pen[i]) hess[i*m+i]-=1.0;
            if (i < m-2*n_eq)  // inequality only: IP barrier curvature
                hess[i*m+i] -= mu[i]/lam[i];
        }
        if (lamai>0.0)
            for (int r=0;r<m;++r)
                for (int c=0;c<m;++c)
                    hess[r*m+c] -= 10.0*a_pen[r]*a_pen[c];
        double trace=0.0;
        for (int i=0;i<m;++i) trace+=hess[i*m+i];
        double corr=1e-4*trace/m;
        if (-corr<1e-7) corr=-1e-7;
        for (int i=0;i<m;++i) hess[i*m+i]+=corr;
    };

    // DualResidual: return ‖KKT residual‖∞ at current (λ, μ, y, z).
    //
    // The barrier KKT conditions are:
    //   r1_i = ∂W/∂λᵢ − bᵢ − aᵢz − yᵢ + μᵢ  = 0   (stationarity in λ)
    //   r2_i = μᵢ·λᵢ − ε                       = 0   (barrier complementarity)
    //
    // Equality slots (last 2·n_eq entries) have no barrier, so r1 simplifies
    // and r2 is dropped for those indices.
    //
    auto DualResidual = [&]() -> double {
        std::vector<double> loc(m,0.0);
        for (int i=0;i<m;++i){
            const auto* pij_r = pij_loc[i].Read();
            const auto* qij_r = qij_loc[i].Read();
            const auto* d_x = x_loc;
            const auto* d_L = L_loc, *d_U = U_loc;
            auto* d_t = d_tmp.Write();
            mfem::forall_switch(use_dev, n_loc, [=] MFEM_HOST_DEVICE (int j){
                double dUx = double(d_U[j])-double(d_x[j]);
                double dxL = double(d_x[j])-double(d_L[j]);
                d_t[j] = pij_r[j]/dUx + qij_r[j]/dxL;
            });
            loc[i] = d_tmp.Sum();
        }
        std::vector<double> res(m);
        MPI_Allreduce(loc.data(), res.data(), m, MPI_DOUBLE, MPI_SUM, comm);
        double nrI=0.0;
        for (int i=0;i<m;++i){
            // Equality slots (last 2*n_eq): no barrier, only constraint stationarity
            double r1 = (i < m-2*n_eq) ?
                res[i]-b[i]-a_pen[i]*z-y[i]+mu[i] : res[i]-b[i];
            nrI=std::max(nrI,std::abs(r1));
            if (i < m-2*n_eq){
                double r2 = mu[i]*lam[i]-epsi;
                nrI=std::max(nrI,std::abs(r2));
            }
        }
        return nrI;
    };

    // LineSearch: find the largest step θ ≤ 1 such that λ and μ stay positive
    // for inequality slots.  The factor 1.005 gives a 0.5 % interior margin.
    // Equality slots have no positivity constraint on λ, so they are skipped.
    //
    auto LineSearch = [&](const std::vector<double>& s) {
        double theta=1.005;
        for (int i=0;i<m-2*n_eq;++i){
            // Keep λᵢ > 0 and μᵢ > 0 by shrinking θ if the Newton step overshoots.
            if (theta < -1.01*s[i]   /lam[i]) theta=-1.01*s[i]   /lam[i];
            if (theta < -1.01*s[m+i] /mu[i])  theta=-1.01*s[m+i] /mu[i];
        }
        theta=1.0/theta;
        for (int i=0;i<m;++i){ lam[i]+=theta*s[i]; mu[i]+=theta*s[m+i]; }
    };

    // Main Newton–barrier loop.
    //
    // Outer loop: reduce ε from lam_max down to tol (factor 10 per step).
    // Inner loop: Newton steps until ‖residual‖∞ < 0.9·ε or 500 steps.
    //   If the inner loop hits a multiple of 25 steps without converging,
    //   ε is reduced early to help escape near-convergent regions.
    //
    std::vector<double> grad(m), hess(m*m), s(2*m);

    while (epsi > tol) {
        double err=1.0; int loop=0;
        while (err > 0.9*epsi && loop < 500) {
            ++loop;
            PrimalFromDual();
            DualGrad(grad);
            for (int i=0;i<m;++i)
                grad[i] = -grad[i] - (i < m-2*n_eq ? epsi/lam[i] : 0.0);
            DualHess(hess);
            std::vector<double> K=hess;
            s.assign(2*m,0.0);
            std::copy(grad.begin(),grad.end(),s.begin());
            SolveDense(K,s,m);
            for (int i=0;i<m;++i)
                s[m+i] = (i < m-2*n_eq) ?
                    -mu[i]+epsi/lam[i]-s[i]*mu[i]/lam[i] : 0.0;
            LineSearch(s);
            PrimalFromDual();
            err=DualResidual();
            if (loop%25==0) epsi*=0.1;
        }
        epsi*=0.1;
    }
    PrimalFromDual();
}


// ─────────────────────────────────────────────────────────────────────────────
// SolveDualSQ  —  interior-point dual solver for the SQ subproblem
// ─────────────────────────────────────────────────────────────────────────────
//
// The SQ approximation (Svanberg 2007 §5.1) replaces f_i near x^k with a
// separable quadratic:
//
//   g_i(x) = f_i(x^k) + ∇f_i(x^k)ᵀ(x − x^k)  +  (ρᵢ/2) Σⱼ δⱼ²/σⱼ²
//
// where δⱼ = xⱼ − xⱼᵏ and σⱼ = U_j − x_j^k  (half the asymptote width).
//
// Primal optimum given multipliers λ:
//
//   D(λ)   = ρ₀ + Σᵢ λᵢ·ρᵢ            (total curvature denominator)
//   G̃_j(λ) = (p0_j−q0_j) + Σᵢ λᵢ·(pij_j−qij_j)   (weighted gradient sum)
//   x_j*(λ) = clip( x_j^k − σⱼ²·G̃_j / D,   α_j, β_j )
//
// Dual function value at x*(λ):
//
//   W_i(λ) = f_i(x^k) + ∇f_iᵀ·δ + (ρᵢ/2) Σⱼ δⱼ²/σⱼ²
//
// Dual gradient (via envelope theorem):
//
//   ∂W/∂λᵢ = Σⱼ [ dfi_j · δ_j  +  (ρᵢ/2)·δ_j²/σⱼ² ]  +  f_i(x^k)
//   where dfi_j = (pij_j − qij_j) / σⱼ²
//
// Dual Hessian:
//
//   H_rc = Σⱼ  dfi_r_j · (−σⱼ²/D) · dfi_c_j
//
// The solver uses the same Newton–barrier loop as SolveDualIP (ε-reduction,
// gradient/Hessian on-device, Newton solve on-host) with two key differences
// for equality constraints (the ±h encoding):
//
//  1. Gradient projection:  the null-space component [1,1] of the ±h
//     Hessian block is stripped from the gradient before the Newton solve,
//     preventing it from driving the step toward the singular direction.
//
//  2. Analytic solve for equality pairs:  the 2×2 block [[a,−a],[−a,a]]
//     has an exact minimum-norm solution s₁ = (g₁−g₂)/(4a), s₂ = −s₁.
//     This replaces the SVD-based SolveDense for those entries, bypassing
//     any floating-point null-space amplification entirely.
//
//  3. Residual projection:  the convergence check uses the range-space
//     residual |(res[+h] − res[−h])|/2 per pair so the symmetric
//     quadratic curvature term (which is identical for both halves) does
//     not inflate the norm and stall convergence.
//
//  4. Multiplier renorm after convergence:  the net multiplier for each
//     equality is ν = λ[+h] − λ[−h].  After the dual solve, the larger
//     slot is set to |ν| and the smaller to a small floor (1e−3), keeping
//     the warm start well-conditioned for the next outer iteration.
//

void SolveDualSQ(
    MPI_Comm comm,
    int n_loc, int m, int n_eq,
    bool use_dev,
    const mfem::real_t* L_loc,
    const mfem::real_t* U_loc,
    const mfem::real_t* alpha_loc,
    const mfem::real_t* beta_loc,
    const double* p0_loc,
    const double* q0_loc,
    const std::vector<mfem::Vector>& pij_loc,
    const std::vector<mfem::Vector>& qij_loc,
    const std::vector<double>& a_pen,
    const std::vector<double>& c_pen,
    std::vector<double>& lam,
    std::vector<double>& mu,
    std::vector<double>& y,
    double& z,
    mfem::real_t* x_loc,
    const mfem::real_t* xk_sq,
    const double* rho_sq,
    const double* fival_sq)
{
    for (int i=0;i<m-2*n_eq;++i){ lam[i]=std::max(lam[i],1.0); mu[i]=std::max(mu[i],1.0); }
    for (int i=m-2*n_eq;i<m;++i){ mu[i]=std::max(mu[i],1.0); }

    double _n_global=0.0;
    { double _nl=double(n_loc);
      MPI_Allreduce(&_nl,&_n_global,1,MPI_DOUBLE,MPI_SUM,comm); }
    const double tol=std::sqrt(_n_global+double(m))*1e-9;

    // epsi declared before lambdas so they can capture it
    // epsi warm-start: use max over all multipliers (including equality slots).
    // No floor — correct initialisation for both fresh (lam~1e-3) and warm
    // (lam~lam_opt) starts. The IP reduces epsi by 0.1 per outer step down to tol.
    double lam_max = *std::max_element(lam.begin(), lam.end());
    double epsi=lam_max;

    mfem::Vector d_tmp(n_loc), d_df2(n_loc), d_delta(n_loc);
    d_tmp.UseDevice(use_dev); d_df2.UseDevice(use_dev); d_delta.UseDevice(use_dev);
    std::vector<mfem::Vector> d_PQ(m);
    for (int i=0;i<m;++i){ d_PQ[i].SetSize(n_loc); d_PQ[i].UseDevice(use_dev); }

    // D(λ) = ρ₀ + Σᵢ λᵢ·ρᵢ — the curvature denominator that appears in the
    // primal formula x* = xk − σ²·G̃/D.  Clamped to 1e-30 to avoid division
    // by zero when all ρ and λ happen to be tiny.
    auto ComputeD = [&]() {
        double D=rho_sq[0];
        for (int i=0;i<m;++i) D+=lam[i]*rho_sq[i+1];
        return std::max(D,1e-30);
    };

    // PrimalFromDual (SQ version): compute x*(λ) = clip(xk − σ²·G̃/D, α, β).
    //
    // G̃_j = (p0_j−q0_j) + Σᵢ λᵢ·(pij_j−qij_j)  is accumulated on-device.
    // The displacement δ_j = x_j* − x_j^k is stored in d_delta for use by
    // DualGrad and DualResidual.
    //
    auto PrimalFromDual = [&]() {
        double lamai=0.0;
        for (int i=0;i<m;++i){
            if (i<m-2*n_eq) lam[i]=std::max(lam[i],0.0);  // keep inequality λ ≥ 0
            y[i]=(i>=m-2*n_eq)?0.0:std::max(0.0,lam[i]-c_pen[i]);
            lamai+=a_pen[i]*lam[i];
        }
        z=std::max(0.0,10.0*(lamai-1.0));
        double D=ComputeD(), invD=1.0/D;
        // Compute G̃_j = (p0-q0) + Σᵢ λᵢ·(pij−qij) into d_tmp on-device
        { auto* pt=d_tmp.Write();
          mfem::forall_switch(use_dev,n_loc,[=] MFEM_HOST_DEVICE (int j){
              pt[j]=p0_loc[j]-q0_loc[j]; }); }
        for (int i=0;i<m;++i){
            double li=lam[i];
            const auto* pij_r=pij_loc[i].Read(); const auto* qij_r=qij_loc[i].Read();
            auto* pt=d_tmp.ReadWrite();
            mfem::forall_switch(use_dev,n_loc,[=] MFEM_HOST_DEVICE (int j){
                pt[j]+=li*(pij_r[j]-qij_r[j]); });
        }
        const auto* G=d_tmp.Read(); const auto* xkr=xk_sq;
        auto* dx=x_loc; auto* dd=d_delta.Write();
        const auto* ad=alpha_loc; const auto* bd=beta_loc;
        mfem::forall_switch(use_dev,n_loc,[=] MFEM_HOST_DEVICE (int j){
            double xj=double(xkr[j])-G[j]*invD;
            xj=xj<double(ad[j])?double(ad[j]):xj;
            xj=xj>double(bd[j])?double(bd[j]):xj;
            dx[j]=mfem::real_t(xj); dd[j]=xj-double(xkr[j]); });
    };

    // DualGrad (SQ version): ∂W/∂λᵢ = Σⱼ[dfi_j·δ_j + (ρᵢ/2)·δ_j²/σⱼ²] + fi(xk)
    //
    // dfi_j = (pij_j − qij_j) / σⱼ²  (the linearised constraint gradient)
    // δ_j   = x_j*(λ) − x_j^k        (already in d_delta from PrimalFromDual)
    //
    auto DualGrad = [&](std::vector<double>& grad) {
        std::vector<double> loc(m,0.0);
        for (int i=0;i<m;++i){
            const auto* pij_r=pij_loc[i].Read(); const auto* qij_r=qij_loc[i].Read();
            const auto* dd=d_delta.Read(); const auto* xkr=xk_sq; const auto* Ur=U_loc;
            double rho_i=rho_sq[i+1];
            auto* d_t=d_tmp.Write();
            mfem::forall_switch(use_dev,n_loc,[=] MFEM_HOST_DEVICE (int j){
                double sig2=double(Ur[j])-double(xkr[j]); sig2=sig2*sig2;
                if(sig2<1e-20) sig2=1e-20;
                double dfi_j=(double(pij_r[j])-double(qij_r[j]))/sig2;
                double dj=dd[j];
                d_t[j]=dfi_j*dj+0.5*rho_i*dj*dj/sig2;
            });
            loc[i]=d_tmp.Sum();
            if (fival_sq) loc[i]+=fival_sq[i];
        }
        MPI_Allreduce(loc.data(),grad.data(),m,MPI_DOUBLE,MPI_SUM,comm);
        for (int i=0;i<m;++i) grad[i]-=a_pen[i]*z+y[i];
    };

    // DualHess (SQ version): H_rc = Σⱼ dfi_r_j · (−σⱼ²/D) · dfi_c_j
    //
    // For each constraint i, precompute the scaled column vector
    //   col_i_j = dfi_i_j · σⱼ² · (−1/D)  = (pij−qij)/σⱼ² · σⱼ² · (−1/D)
    //           = (pij_j − qij_j) · (−1/D)
    //
    // Then H_rc = InnerProduct(col_r, raw_dfi_c), a single device dot product.
    // The m×m result is accumulated in a local array and reduced across ranks.
    //
    auto DualHess = [&](std::vector<double>& hess) {
        double D=ComputeD();
        const auto* xkr=xk_sq; const auto* Ur=U_loc;
        for (int i=0;i<m;++i){
            const auto* pij_r=pij_loc[i].Read(); const auto* qij_r=qij_loc[i].Read();
            auto* pq_w=d_PQ[i].Write(); double inv_neg_D=-1.0/D;
            mfem::forall_switch(use_dev,n_loc,[=] MFEM_HOST_DEVICE (int j){
                double sig2=double(Ur[j])-double(xkr[j]); sig2=sig2*sig2;
                if(sig2<1e-20) sig2=1e-20;
                double dfi_j=(double(pij_r[j])-double(qij_r[j]))/sig2;
                pq_w[j]=dfi_j*sig2*inv_neg_D;
            });

        }
        std::vector<double> loc(m*m,0.0);
        for (int r=0;r<m;++r) for (int c=r;c<m;++c){
            const auto* pij_c=pij_loc[c].Read(); const auto* qij_c=qij_loc[c].Read();
            auto* raw=d_tmp.Write();
            mfem::forall_switch(use_dev,n_loc,[=] MFEM_HOST_DEVICE (int j){
                double sig2=double(Ur[j])-double(xkr[j]); sig2=sig2*sig2;
                if(sig2<1e-20) sig2=1e-20;
                raw[j]=(double(pij_c[j])-double(qij_c[j]))/sig2;
            });
            double v=mfem::InnerProduct(comm,d_PQ[r],d_tmp);
            loc[r*m+c]=v; if(r!=c) loc[c*m+r]=v;
        }
        MPI_Allreduce(loc.data(),hess.data(),m*m,MPI_DOUBLE,MPI_SUM,comm);
        double lamai=0.0;
        for (int i=0;i<m;++i) lamai+=lam[i]*a_pen[i];
        for (int i=0;i<m;++i){
            if (lam[i]>c_pen[i]) hess[i*m+i]-=1.0;
            if (i<m-2*n_eq) hess[i*m+i]-=mu[i]/lam[i];
        }
        if (lamai>0.0)
            for (int r=0;r<m;++r) for (int c=0;c<m;++c)
                hess[r*m+c]-=10.0*a_pen[r]*a_pen[c];
    };

    // DualResidual (SQ version): ‖KKT residual‖∞ at current (λ, μ).
    //
    // For inequality slots the residual is the usual barrier-KKT pair:
    //   r1_i = ∂W/∂λᵢ − aᵢz − yᵢ + μᵢ
    //   r2_i = μᵢ·λᵢ − ε
    //
    // For equality pairs (slots ni+k and ni+n_eq+k) we project the residual
    // onto the range space of the ±h Hessian before taking its norm.  Both
    // halves share the symmetric quadratic curvature term (ρᵢ/2)·‖δ/σ‖²,
    // which is identical for +h and −h.  Using the antisymmetric combination
    // (res[+h] − res[−h])/2 cancels that common term and gives a residual
    // that faithfully measures convergence of the net dual variable ν = λ₊−λ₋.
    //
    auto DualResidual = [&]() -> double {
        std::vector<double> loc(m,0.0);
        for (int i=0;i<m;++i){
            const auto* pij_r=pij_loc[i].Read(); const auto* qij_r=qij_loc[i].Read();
            const auto* dd=d_delta.Read(); const auto* xkr=xk_sq; const auto* Ur=U_loc;
            double rho_i=rho_sq[i+1];
            auto* d_t=d_tmp.Write();
            mfem::forall_switch(use_dev,n_loc,[=] MFEM_HOST_DEVICE (int j){
                double sig2=double(Ur[j])-double(xkr[j]); sig2=sig2*sig2;
                if(sig2<1e-20) sig2=1e-20;
                double dfi_j=(double(pij_r[j])-double(qij_r[j]))/sig2;
                double dj=dd[j];
                d_t[j]=dfi_j*dj+0.5*rho_i*dj*dj/sig2;
            });
            loc[i]=d_tmp.Sum();
            if (fival_sq) loc[i]+=fival_sq[i];
        }
        std::vector<double> res(m);
        MPI_Allreduce(loc.data(),res.data(),m,MPI_DOUBLE,MPI_SUM,comm);
        // For equality pairs: use the range-space residual |(res[+h]-res[-h])/2|
        // to avoid the symmetric quad term (rho/2*||delta||^2/sig^2) inflating
        // the residual and preventing inner-loop convergence.
        {
            const int ni=m-2*n_eq;
            for (int k=0;k<n_eq;++k) {
                const int i1=ni+k, i2=ni+n_eq+k;
                double half=(res[i1]-res[i2])*0.5;
                res[i1]= half;
                res[i2]=-half;
            }
        }
        double nrI=0.0;
        for (int i=0;i<m;++i){
            double r1=(i<m-2*n_eq)?res[i]-a_pen[i]*z-y[i]+mu[i]:res[i];
            nrI=std::max(nrI,std::abs(r1));
            if(i<m-2*n_eq){ double r2=mu[i]*lam[i]-epsi; nrI=std::max(nrI,std::abs(r2)); }
        }
        return nrI;
    };

    // LineSearch: same as SolveDualIP — shrink θ to keep λ,μ > 0 for
    // inequality slots.  Equality slots have no positivity constraint on λ.
    //
    auto LineSearch = [&](const std::vector<double>& s) {
        double theta=1.005;
        for (int i=0;i<m-2*n_eq;++i){
            if (theta < -1.01*s[i]   /lam[i]) theta=-1.01*s[i]   /lam[i];
            if (theta < -1.01*s[m+i] /mu[i])  theta=-1.01*s[m+i] /mu[i];
        }
        theta=1.0/theta;
        for (int i=0;i<m;++i) lam[i]+=theta*s[i];
        for (int i=0;i<m-2*n_eq;++i) mu[i]+=theta*s[m+i];
    };

    // Fast path for unconstrained problems (m = 0):
    //   x_j* = clip( x_j^k − G̃_j / ρ₀,  α_j, β_j )
    // No dual system needed; a single forall kernel suffices.
    //
    if (m==0){
        double D=rho_sq[0]; if(D<1e-30) D=1e-30;
        const auto* xkr=xk_sq;
        mfem::forall_switch(use_dev,n_loc,[=] MFEM_HOST_DEVICE (int j){
            double gj=p0_loc[j]-q0_loc[j];
            double xj=double(xkr[j])-gj/D;
            xj=xj<double(alpha_loc[j])?double(alpha_loc[j]):xj;
            xj=xj>double(beta_loc[j])?double(beta_loc[j]):xj;
            x_loc[j]=mfem::real_t(xj);
        });
        return;
    }

    // Main Newton–barrier loop (same structure as SolveDualIP).
    // The equality-specific logic (gradient projection, analytic pair solve,
    // residual projection) is applied inside the inner loop.
    //
    std::vector<double> grad(m),hess(m*m),s(2*m);
    while (epsi>tol){
        double err=1.0; int loop=0;
        while (err>0.9*epsi && loop<500){
            ++loop;
            PrimalFromDual();
            DualGrad(grad);
            for (int i=0;i<m;++i)
                grad[i]=-grad[i]-(i<m-2*n_eq?epsi/lam[i]:0.0);
            // Equality pairs: project the gradient onto the range space of the ±h
            // Hessian before solving.  The Hessian block for pair k is [[a,−a],[−a,a]],
            // which is rank-1 with null direction [1,1].  Without projection, the
            // symmetric component (g[i1]+g[i2])/2 would feed into the singular
            // direction and produce catastrophically large steps via dgelsd.
            // Replacing (g[i1], g[i2]) with ((g[i1]−g[i2])/2, (g[i2]−g[i1])/2)
            // zeros that component, leaving only the antisymmetric part.
            {
                const int ni=m-2*n_eq;
                for (int k=0;k<n_eq;++k) {
                    const int i1=ni+k, i2=ni+n_eq+k;
                    double half=(grad[i1]-grad[i2])*0.5;
                    grad[i1]= half;
                    grad[i2]=-half;
                }
            }
            DualHess(hess);
            std::vector<double> K=hess;
            s.assign(2*m,0.0);
            std::copy(grad.begin(),grad.end(),s.begin());
            SolveDense(K,s,m);
            // Analytic override for equality pairs — pure-equality systems only.
            //
            // Each +/-h pair has a rank-1 Hessian block [[-a,a],[a,-a]].  In a
            // PURE equality system (n_ineq == 0) the full m*m Hessian is block
            // diagonal over these pairs, so the 2x2 minimum-norm formula
            //   s1 = (g1-g2)/(4a),  s2 = -s1
            // gives the exact answer and avoids any null-space amplification by
            // dgelsd.
            //
            // In a MIXED system (n_ineq > 0) the inequality and equality blocks
            // are coupled through off-diagonal Hessian entries H[ineq,eq] != 0
            // whenever the constraint gradients are not orthogonal.  Overriding
            // only the equality slots would destroy those cross-term contributions.
            // SolveDense's SVD minimum-norm solution already handles the rank
            // deficiency correctly via the rcond threshold in that case.
            if (n_eq > 0 && (m - 2*n_eq) == 0) {
                // Pure equality (n_ineq==0): use exact analytic formula per pair.
                for (int k = 0; k < n_eq; ++k) {
                    const int i1 = k, i2 = n_eq+k;
                    double a = hess[i1*m+i1];
                    if (std::abs(a) > 1e-30) {
                        double step = (grad[i1] - grad[i2]) / (4.0*a);
                        s[i1] =  step;
                        s[i2] = -step;
                    } else {
                        s[i1] = s[i2] = 0.0;
                    }
                }
            }

            // Complementarity step: Δμᵢ = −μᵢ + ε/λᵢ − Δλᵢ·μᵢ/λᵢ  (inequality only)
            for (int i=0;i<m;++i)
                s[m+i]=(i<m-2*n_eq)?-mu[i]+epsi/lam[i]-s[i]*mu[i]/lam[i]:0.0;
            LineSearch(s);
            PrimalFromDual();
            err=DualResidual();
            if (loop%25==0) epsi*=0.1;
        }
        epsi*=0.1;
    }
    PrimalFromDual();
}
} // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
// UpdateAsymptotes  —  adaptive MMA asymptote update
// ─────────────────────────────────────────────────────────────────────────────
//
// Implements the standard MMA asymptote adaptation rule (Svanberg 1987):
//
//  Iteration 0 or 1 (no history):
//    L_j = x_j − asyinit · (xmax_j − xmin_j)
//    U_j = x_j + asyinit · (xmax_j − xmin_j)
//
//  Later iterations:
//    if sign(x_j − x_j¹)(x_j¹ − x_j²) < 0  →  oscillation  →  γ = asydec
//    if sign(x_j − x_j¹)(x_j¹ − x_j²) > 0  →  monotone     →  γ = asyinc
//    else                                                         γ = 1
//    L_j = x_j − γ · (x_j¹ − L_j¹)   (shrink/widen from previous L)
//    U_j = x_j + γ · (U_j¹ − x_j¹)
//    Both asymptotes are clamped to stay within [x ± 100·range, x ± 1e-4·range].
//
//  Move limits (used as inner bounds for x):
//    α_j = max(xmin_j,  L_j + 0.1·(x_j − L_j),  x_j − 0.5·range_j)
//    β_j = min(xmax_j,  U_j − 0.1·(U_j − x_j),  x_j + 0.5·range_j)
//
static void UpdateAsymptotes(
    int n, int iter,
    double asyinit, double asydec, double asyinc,
    const mfem::real_t* x,
    const mfem::real_t* xo1,
    const mfem::real_t* xo2,
    const mfem::real_t* xmin,
    const mfem::real_t* xmax,
    mfem::real_t* L, mfem::real_t* U,
    mfem::real_t* alpha, mfem::real_t* beta,
    bool use_dev)
{
    mfem::forall_switch(use_dev, n, [=] MFEM_HOST_DEVICE (int j) {
        double rng = double(xmax[j])-double(xmin[j]);
        double Lj, Uj;
        if (iter < 2) {
            Lj = double(x[j]) - asyinit*rng;
            Uj = double(x[j]) + asyinit*rng;
        } else {
            double prod = (double(x[j])-double(xo1[j]))*(double(xo1[j])-double(xo2[j]));
            double gamma = (prod < 0.0) ? asydec : (prod > 0.0) ? asyinc : 1.0;
            Lj = double(x[j]) - gamma*(double(xo1[j])-double(L[j]));
            Uj = double(x[j]) + gamma*(double(U[j])-double(xo1[j]));
            double xmi = rng > 1e-5 ? rng : 1e-5;
            Lj = Lj < double(x[j])-100.0*xmi ? double(x[j])-100.0*xmi : Lj;
            Lj = Lj > double(x[j])-1e-4*xmi  ? double(x[j])-1e-4*xmi  : Lj;
            Uj = Uj < double(x[j])+1e-4*xmi  ? double(x[j])+1e-4*xmi  : Uj;
            Uj = Uj > double(x[j])+100.0*xmi ? double(x[j])+100.0*xmi : Uj;
        }
        L[j] = mfem::real_t(Lj); U[j] = mfem::real_t(Uj);
        double al = double(xmin[j]);
        double a1 = Lj+0.1*(double(x[j])-Lj);
        double a2 = double(x[j])-0.5*rng;
        al = al>a1?al:a1; al = al>a2?al:a2;
        al = al<double(xmax[j])?al:double(xmax[j]);
        alpha[j] = mfem::real_t(al);
        double be = double(xmax[j]);
        double b1 = Uj-0.1*(Uj-double(x[j]));
        double b2 = double(x[j])+0.5*rng;
        be = be<b1?be:b1; be = be<b2?be:b2;
        be = be>double(xmin[j])?be:double(xmin[j]);
        beta[j] = mfem::real_t(be);
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// BuildCoeffs  —  construct the MMA subproblem p/q coefficients
// ─────────────────────────────────────────────────────────────────────────────
//
// The MMA subproblem approximates each function fᵢ(x) by the separable
// rational model (Svanberg 1987, eq. 3.3):
//
//   g_i(x) = Σⱼ [ pᵢⱼ/(U_j−x_j) + qᵢⱼ/(x_j−L_j) ]  −  const
//
// where pᵢⱼ ≥ 0 and qᵢⱼ ≥ 0 are built so that the model matches the value
// and gradient of fᵢ at x^k and is globally convex on (L, U).
//
// For the objective (i = −1, stored in p0/q0):
//   p0_j = (U_j−x_j)² · ( max(∂f₀/∂xⱼ, 0)  +  rh_j )
//   q0_j = (x_j−L_j)² · ( max(−∂f₀/∂xⱼ, 0) +  rh_j )
//   rh_j  = 0.001·|∂f₀/∂xⱼ| + (ρ₀ + ρfloor) / (xmax_j−xmin_j)
//
// For each constraint i:
//   pᵢⱼ = (U_j−x_j)² · ( max(∂fᵢ/∂xⱼ, 0)  +  rhc_j )
//   qᵢⱼ = (x_j−L_j)² · ( max(−∂fᵢ/∂xⱼ, 0) +  rhc_j )
//   rhc_j = 0.001·|∂fᵢ/∂xⱼ| + (ρᵢ + ρfloor) / (xmax_j−xmin_j)
//
// The regularisation terms rh/rhc (proportional to ρ) are the GCMMA
// curvature parameters; rho = nullptr means ρ = 0 (plain MMA, no curvature).
//
// The b vector, b_i = Σⱼ[pᵢⱼ/(U−x) + qᵢⱼ/(x−L)] − fᵢ(x^k), is the
// constant term in the dual.  It is computed here (device sum) and reduced
// across MPI ranks by the caller.
//
// All work is device-aware: kernel launches via forall_switch, sums via
// Vector::Sum().  The output Vectors (p0, q0, pij, qij) inherit the device
// flag from the caller.
//
static void BuildCoeffs(
    int n_loc, int m, bool use_dev,
    const mfem::real_t* x,
    const mfem::real_t* L, const mfem::real_t* U,
    const mfem::real_t* xmin, const mfem::real_t* xmax,
    const mfem::real_t* df0,
    std::vector<mfem::Vector>& pij_v,  // [m] device Vectors
    std::vector<mfem::Vector>& qij_v,
    mfem::Vector& p0_v, mfem::Vector& q0_v,
    const mfem::real_t* const* dfi,           // m device ptrs
    const double* rho,                        // m+1, host (rho[0]=obj, rho[i+1]=cstr)
    std::vector<double>& b_loc)
{
    const double rho0 = 1e-5;
    double rho_obj = rho ? rho[0] : 0.0;
    // Build p0, q0
    {
        auto* p0w = p0_v.Write();
        auto* q0w = q0_v.Write();
        mfem::forall_switch(use_dev, n_loc, [=] MFEM_HOST_DEVICE (int j){
            double dUx = double(U[j])-double(x[j]);
            double dxL = double(x[j])-double(L[j]);
            double xmi = double(xmax[j])-double(xmin[j]);
            xmi = xmi > 1e-5 ? xmi : 1e-5;
            double df = double(df0[j]);
            double pp = df > 0 ? df : 0.0;
            double pm = df < 0 ?-df : 0.0;
            double rh = 0.001*(df>0?df:-df) + (rho_obj+rho0)/xmi;
            p0w[j] = dUx*dUx*(pp+rh);
            q0w[j] = dxL*dxL*(pm+rh);
        });
    }
    // Build pij, qij for each constraint
    for (int i=0;i<m;++i){
        double rho_i = rho ? rho[i+1] : 0.0;
        const auto* dfi_i = dfi[i];
        auto* piw = pij_v[i].Write();
        auto* qiw = qij_v[i].Write();
        mfem::forall_switch(use_dev, n_loc, [=] MFEM_HOST_DEVICE (int j){
            double dUx = double(U[j])-double(x[j]);
            double dxL = double(x[j])-double(L[j]);
            double xmi = double(xmax[j])-double(xmin[j]);
            xmi = xmi > 1e-5 ? xmi : 1e-5;
            double dg = double(dfi_i[j]);
            double dp = dg > 0 ? dg : 0.0;
            double dm = dg < 0 ?-dg : 0.0;
            double rhc = 0.001*(dg>0?dg:-dg) + (rho_i+rho0)/xmi;
            piw[j] = dUx*dUx*(dp+rhc);
            qiw[j] = dxL*dxL*(dm+rhc);
        });
    }
    // Build b_loc[i] = sum_j[pij/(U-x)+qij/(x-L)]  (device sum → host)
    // Use a temporary device Vector for the summand
    mfem::Vector d_tmp(n_loc); d_tmp.UseDevice(use_dev);
    for (int i=0;i<m;++i){
        const auto* pi_r = pij_v[i].Read();
        const auto* qi_r = qij_v[i].Read();
        auto* dt = d_tmp.Write();
        mfem::forall_switch(use_dev, n_loc, [=] MFEM_HOST_DEVICE (int j){
            double dUx=double(U[j])-double(x[j]);
            double dxL=double(x[j])-double(L[j]);
            dt[j] = pi_r[j]/dUx + qi_r[j]/dxL;
        });
        b_loc[i] = d_tmp.Sum();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MMAOptimizer  —  serial MMA/GCMMA implementation
// ─────────────────────────────────────────────────────────────────────────────

static void InitDeviceVectors(
    int n, int m, const mfem::Vector& ref,
    mfem::Vector& p0, mfem::Vector& q0,
    std::vector<mfem::Vector>& pij,
    std::vector<mfem::Vector>& qij,
    mfem::Vector& L, mfem::Vector& U,
    mfem::Vector& alpha, mfem::Vector& beta,
    mfem::Vector& xo1, mfem::Vector& xo2)
{
    auto Init = [&](mfem::Vector& v, int sz) {
        v.SetSize(sz); v.UseDevice(ref.UseDevice()); v=0.0;
    };
    Init(p0,n); Init(q0,n); Init(L,n); Init(U,n); Init(alpha,n); Init(beta,n);
    xo1=ref; xo2=ref;  // copies data AND device flag
    pij.resize(m); qij.resize(m);
    for (int i=0;i<m;++i){ Init(pij[i],n); Init(qij[i],n); }
}

MMAOptimizer::MMAOptimizer(int n, int m, const mfem::Vector& x)
    : n_(n), m_(m), iter_(0)
    , asyminit_(0.5), asymdec_(0.7), asyminc_(1.2)
    , z_(1.0)
    , b_(m,0.0), rho_(m+1,1e-5)
    , lam_(m,1.0), mu_(m,1.0), y_(m,0.0)
{
    DefaultPenalty(n,m,a_,c_,d_);
    InitDeviceVectors(n,m,x,p0_,q0_,pij_,qij_,L_,U_,alpha_,beta_,xo1_,xo2_);
}

MMAOptimizer::MMAOptimizer(int n, int m, const mfem::Vector& x,
                            const double* a, const double* c, const double* d)
    : n_(n), m_(m), iter_(0)
    , asyminit_(0.5), asymdec_(0.7), asyminc_(1.2)
    , z_(1.0)
    , b_(m,0.0), rho_(m+1,1e-5)
    , lam_(m,1.0), mu_(m,1.0), y_(m,0.0)
{
    a_.assign(a,a+m); c_.assign(c,c+m); d_.assign(d,d+m);
    InitDeviceVectors(n,m,x,p0_,q0_,pij_,qij_,L_,U_,alpha_,beta_,xo1_,xo2_);
}

MMAOptimizer::MMAOptimizer(int n, int m, const mfem::Vector& x,
                            const mfem::Vector& a, const mfem::Vector& c,
                            const mfem::Vector& d)
    : MMAOptimizer(n, m, x,
                   VecToDouble(a).data(),
                   VecToDouble(c).data(),
                   VecToDouble(d).data())
{}

/// @brief (Serial) Store asymptote adaptation speeds.
void MMAOptimizer::SetAsymptotes(mfem::real_t i, mfem::real_t d, mfem::real_t inc)
{ asyminit_=i; asymdec_=d; asyminc_=inc; }

/// @brief (Serial) Update asymptotes and build plain-MMA sub-problem (rho=0).
void MMAOptimizer::BuildSubproblem_(
    const mfem::Vector& x, const mfem::Vector& df0dx,
    const mfem::Vector& fival, const mfem::Vector* dfidx,
    const mfem::Vector& xmin, const mfem::Vector& xmax)
{
    bool ud = x.UseDevice();
    UpdateAsymptotes(n_,iter_,asyminit_,asymdec_,asyminc_,
        x.Read(),xo1_.Read(),xo2_.Read(),xmin.Read(),xmax.Read(),
        L_.Write(),U_.Write(),alpha_.Write(),beta_.Write(),ud);

    std::vector<const mfem::real_t*> dfi(m_);
    for (int i=0;i<m_;++i) dfi[i]=dfidx[i].Read();

    // null ptr sentinel for rho (plain MMA uses rho=0)
    static const double zero_rho[1]={0.0};
    std::vector<double> rho_zero(m_+1,0.0);
    BuildCoeffs(n_,m_,ud,
        x.Read(),L_.Read(),U_.Read(),xmin.Read(),xmax.Read(),
        df0dx.Read(),pij_,qij_,p0_,q0_,
        dfi.data(),rho_zero.data(),b_);
    for (int i=0;i<m_;++i) b_[i]=b_[i]-double(fival(i));
}

/// @brief (Serial) Build GCMMA sub-problem with explicit conservatism rho.
void MMAOptimizer::BuildSubproblemRho_(
    const mfem::Vector& x, const mfem::Vector& df0dx,
    const mfem::Vector& fival, const mfem::Vector* dfidx,
    const mfem::Vector& xmin, const mfem::Vector& xmax,
    const std::vector<double>& rho)
{
    bool ud = x.UseDevice();
    std::vector<const mfem::real_t*> dfi(m_);
    for (int i=0;i<m_;++i) dfi[i]=dfidx[i].Read();
    BuildCoeffs(n_,m_,ud,
        x.Read(),L_.Read(),U_.Read(),xmin.Read(),xmax.Read(),
        df0dx.Read(),pij_,qij_,p0_,q0_,
        dfi.data(),rho.data(),b_);
    for (int i=0;i<m_;++i) b_[i]=b_[i]-double(fival(i));
}

/**
 * @brief (Serial) One MMA outer iteration.
 *
 * Reads use_dev from x, calls BuildSubproblem_(), advances xo1_/xo2_
 * history, then calls SolveDualIP() with MPI_COMM_SELF.
 * p0_ and q0_ are passed to SolveDualIP via HostRead() since the dual
 * solver accumulates them into std::vector<double> on the host.
 */
void MMAOptimizer::Update(
    mfem::Vector& x, const mfem::Vector& df0dx, mfem::real_t f0val,
    const mfem::Vector& fival, const mfem::Vector* dfidx,
    const mfem::Vector& xmin, const mfem::Vector& xmax)
{
    bool ud = x.UseDevice();
    BuildSubproblem_(x,df0dx,fival,dfidx,xmin,xmax);
    xo2_=xo1_; xo1_=x;
    detail::SolveDualIP(MPI_COMM_SELF,n_,m_,n_eq_,ud,
        L_.Read(),U_.Read(),alpha_.Read(),beta_.Read(),
        p0_.HostRead(),q0_.HostRead(),pij_,qij_,
        b_,a_,c_,d_,lam_,mu_,y_,z_,x.ReadWrite());
    ++iter_;
}

/**
 * @brief (Serial) One GCMMA outer iteration.
 *
 * Computes rho on-device via forall_switch (sum|df/dx|*range),
 * then calls BuildSubproblemRho_() and SolveDualIP() once.
 * History is advanced once after the solve.
 */
void MMAOptimizer::UpdateGCMMA(
    mfem::Vector& x, const mfem::Vector& df0dx, mfem::real_t f0val,
    const mfem::Vector& fival, const mfem::Vector* dfidx,
    const mfem::Vector& xmin, const mfem::Vector& xmax, int* innerIter)
{
    bool ud = x.UseDevice();
    UpdateAsymptotes(n_,iter_,asyminit_,asymdec_,asyminc_,
        x.Read(),xo1_.Read(),xo2_.Read(),xmin.Read(),xmax.Read(),
        L_.Write(),U_.Write(),alpha_.Write(),beta_.Write(),ud);

    rho_.assign(m_+1,0.0);
    // Compute rho sum on device — InnerProduct of |df0|*(xmax-xmin)
    // Use a temporary vector for the product
    mfem::Vector d_tmp(n_); d_tmp.UseDevice(ud);
    {
        const auto* df0r = df0dx.Read();
        const auto* xmnr = xmin.Read();
        const auto* xmxr = xmax.Read();
        auto* dt = d_tmp.Write();
        mfem::forall_switch(ud, n_, [=] MFEM_HOST_DEVICE (int j){
            double v = double(df0r[j]); if(v<0)v=-v;
            dt[j] = v*(double(xmxr[j])-double(xmnr[j]));
        });
        rho_[0] = d_tmp.Sum();
    }
    for (int i=0;i<m_;++i){
        const auto* dfir = dfidx[i].Read();
        const auto* xmnr = xmin.Read();
        const auto* xmxr = xmax.Read();
        auto* dt = d_tmp.Write();
        mfem::forall_switch(ud, n_, [=] MFEM_HOST_DEVICE (int j){
            double v = double(dfir[j]); if(v<0)v=-v;
            dt[j] = v*(double(xmxr[j])-double(xmnr[j]));
        });
        rho_[i+1] = d_tmp.Sum();
    }
    for (int k=0;k<=m_;++k)
        rho_[k]=std::max(1e-6, 0.5/(double)n_*rho_[k]);

    mfem::Vector xk(x);  // device copy
    BuildSubproblemRho_(xk,df0dx,fival,dfidx,xmin,xmax,rho_);
    detail::SolveDualIP(MPI_COMM_SELF,n_,m_,n_eq_,ud,
        L_.Read(),U_.Read(),alpha_.Read(),beta_.Read(),
        p0_.HostRead(),q0_.HostRead(),pij_,qij_,
        b_,a_,c_,d_,lam_,mu_,y_,z_,x.ReadWrite());
    if (innerIter) *innerIter=1;
    xo2_=xo1_; xo1_=xk; ++iter_;
}

/**
 * @brief (Serial) Projected-gradient KKT residual.
 *
 * Assembles the Lagrangian gradient on the device, projects at bounds
 * via forall_switch, squares, and sums via Vector::Sum().
 * Adds the dual complementarity term sum(lambda_i*fi_i)^2 on the host.
 * Divides by n_.
 */
mfem::real_t MMAOptimizer::KKTresidual(
    const mfem::Vector& x, const mfem::Vector& df0dx, double,
    const mfem::Vector& fival, const mfem::Vector* dfidx,
    const mfem::Vector& xmin, const mfem::Vector& xmax,
    double* lambda_out) const
{
    if (lambda_out) std::copy(lam_.begin(),lam_.end(),lambda_out);
    bool ud = x.UseDevice();
    // Compute primal+dual KKT on device, sum to host
    mfem::Vector d_tmp(n_); d_tmp.UseDevice(ud);
    // Lagrangian gradient: gradL_j = df0_j + sum_i lam_i * dfi_j
    // Projected gradient squared: pg_j^2
    {
        const auto* xr   = x.Read();
        const auto* df0r = df0dx.Read();
        const auto* xmnr = xmin.Read();
        const auto* xmxr = xmax.Read();
        auto* dt = d_tmp.Write();
        const double tol = 1e-3;
        mfem::forall_switch(ud, n_, [=] MFEM_HOST_DEVICE (int j){
            double g = double(df0r[j]);
            dt[j] = g; // accumulate lam*dfi below
        });
        for (int i=0;i<m_-2*n_eq_;++i){
            double li=lam_[i];
            const auto* dfir=dfidx[i].Read();
            auto* dtr=d_tmp.ReadWrite();
            mfem::forall_switch(ud, n_, [=] MFEM_HOST_DEVICE (int j){
                dtr[j] += li*double(dfir[j]);
            });
        }
        for (int k=0;k<n_eq_;++k){
            const int ni2=m_-2*n_eq_;
            double lnet=lam_[ni2+k]-lam_[ni2+n_eq_+k];
            const auto* dfir=dfidx[ni2+k].Read();
            auto* dtr=d_tmp.ReadWrite();
            mfem::forall_switch(ud, n_, [=] MFEM_HOST_DEVICE (int j){
                dtr[j] += lnet*double(dfir[j]);
            });
        }
        // Project and square
        mfem::forall_switch(ud, n_, [=] MFEM_HOST_DEVICE (int j){
            double rng = double(xmxr[j])-double(xmnr[j]);
            double t   = tol*rng;
            double g   = double(dt[j]);
            double pg  = g;
            if (double(xr[j])<=double(xmnr[j])+t) pg = g<0?g:0.0;
            if (double(xr[j])>=double(xmxr[j])-t) pg = g>0?g:0.0;
            dt[j] = pg*pg;
        });
    }
    double primal = d_tmp.Sum();
    double dual = 0.0;
    for (int i=0;i<m_-2*n_eq_;++i){ double cs=lam_[i]*double(fival(i)); dual+=cs*cs; }
    for (int k=0;k<n_eq_;++k){ double hk=double(fival(m_-2*n_eq_+k)); dual+=hk*hk; }
    return (primal+dual)/(double)n_;
}

// ============================================================
// MMAOptimizerParallel
// ============================================================


/**
 * @brief (Serial) GCMMA with full inner conservatism loop.
 *
 * Implements Svanberg (2007) §4 inner loop with user-supplied evaluator.
 * After each sub-problem solve, eval_fi is called at the candidate x̂ to
 * check whether the MMA approximation is conservative.  If not, ρ is
 * increased and the sub-problem is re-solved.
 */
void MMAOptimizer::UpdateGCMMA(
    mfem::Vector& x,
    const mfem::Vector& df0dx,
    mfem::real_t f0val,
    const mfem::Vector& fival,
    const mfem::Vector* dfidx,
    const mfem::Vector& xmin,
    const mfem::Vector& xmax,
    EvalCallback eval_fi,
    int  max_inner,
    int* innerIter)
{
    bool ud = x.UseDevice();

    UpdateAsymptotes(n_,iter_,asyminit_,asymdec_,asyminc_,
        x.Read(),xo1_.Read(),xo2_.Read(),xmin.Read(),xmax.Read(),
        L_.Write(),U_.Write(),alpha_.Write(),beta_.Write(),ud);

    rho_.assign(m_+1,0.0);
    mfem::Vector d_tmp(n_); d_tmp.UseDevice(ud);
    {
        const auto* df0r=df0dx.Read();
        const auto* xmnr=xmin.Read(), *xmxr=xmax.Read();
        auto* dt=d_tmp.Write();
        mfem::forall_switch(ud,n_,[=] MFEM_HOST_DEVICE (int j){
            double v=double(df0r[j]); if(v<0)v=-v;
            dt[j]=v*(double(xmxr[j])-double(xmnr[j]));
        });
        rho_[0]=d_tmp.Sum();
    }
    for (int i=0;i<m_;++i){
        const auto* dfir=dfidx[i].Read();
        const auto* xmnr=xmin.Read(), *xmxr=xmax.Read();
        auto* dt=d_tmp.Write();
        mfem::forall_switch(ud,n_,[=] MFEM_HOST_DEVICE (int j){
            double v=double(dfir[j]); if(v<0)v=-v;
            dt[j]=v*(double(xmxr[j])-double(xmnr[j]));
        });
        rho_[i+1]=d_tmp.Sum();
    }
    for (int k=0;k<=m_;++k)
        rho_[k]=std::max(1e-6, 0.5/(double)n_*rho_[k]);

    mfem::Vector xk(x);
    int nu=0;

    while (nu < max_inner) {
        x = xk;
        BuildSubproblemRho_(xk,df0dx,fival,dfidx,xmin,xmax,rho_);
        detail::SolveDualIP(MPI_COMM_SELF,n_,m_,n_eq_,ud,
            L_.Read(),U_.Read(),alpha_.Read(),beta_.Read(),
            p0_.HostRead(),q0_.HostRead(),pij_,qij_,
            b_,a_,c_,d_,lam_,mu_,y_,z_,x.ReadWrite());

        mfem::Vector fi_hat(m_);
        mfem::real_t f0_hat=0;
        eval_fi(x, fi_hat, f0_hat);

        // d^k(x̂) denominator
        double dk=0.0;
        const auto* xr =x.HostRead();
        const auto* xkr=xk.HostRead();
        const auto* Lr =L_.HostRead();
        const auto* Ur =U_.HostRead();
        const auto* xmnr=xmin.HostRead();
        const auto* xmxr=xmax.HostRead();
        for (int j=0;j<n_;++j){
            double dUx=double(Ur[j])-double(xr[j]);
            double dxL=double(xr[j])-double(Lr[j]);
            double dxk=double(xr[j])-double(xkr[j]);
            double rng=std::max(double(xmxr[j])-double(xmnr[j]),1e-10);
            dk+=(double(Ur[j])-double(Lr[j]))*dxk*dxk/(dUx*dxL*rng+1e-30);
        }
        if (dk<1e-14) dk=1e-14;

        bool conservative=true;

        // Objective conservatism
        {
            double s_xhat=0, s_xk=0;
            for (int j=0;j<n_;++j){
                double dUx =double(Ur[j])-double(xr[j]);
                double dxL =double(xr[j])-double(Lr[j]);
                double dUxk=double(Ur[j])-double(xkr[j]);
                double dxLk=double(xkr[j])-double(Lr[j]);
                s_xhat+=p0_.HostRead()[j]/dUx +q0_.HostRead()[j]/dxL;
                s_xk  +=p0_.HostRead()[j]/dUxk+q0_.HostRead()[j]/dxLk;
            }
            double b_obj  = s_xk - double(f0val);
            double ftilde0 = s_xhat - b_obj;
            double delta   = (double(f0_hat) - ftilde0) / dk;
            if (delta>0.0){
                rho_[0]=std::min(1.1*(rho_[0]+delta), 10.0*rho_[0]);
                conservative=false;
            }
        }

        // Constraint conservatism
        for (int i=0;i<m_;++i){
            double s_xhat=0;
            for (int j=0;j<n_;++j){
                double dUx=double(Ur[j])-double(xr[j]);
                double dxL=double(xr[j])-double(Lr[j]);
                s_xhat+=pij_[i].HostRead()[j]/dUx
                       +qij_[i].HostRead()[j]/dxL;
            }
            double ftilde_i = s_xhat - b_[i];
            double delta    = (double(fi_hat(i)) - ftilde_i) / dk;
            if (delta>0.0){
                rho_[i+1]=std::min(1.1*(rho_[i+1]+delta), 10.0*rho_[i+1]);
                conservative=false;
            }
        }

        ++nu;
        if (conservative) break;
    }

    if (innerIter) *innerIter=nu;
    xo2_=xo1_; xo1_=xk; ++iter_;
}


MMAOptimizerParallel::MMAOptimizerParallel(
    MPI_Comm comm, int n_local, int m,
    const mfem::Vector& x_local)
    : comm_(comm), n_global_(ComputeNGlobal(comm,n_local)), n_local_(n_local), m_(m), iter_(0)
    , asyminit_(0.5), asymdec_(0.7), asyminc_(1.2)
    , z_(1.0)
    , b_(m,0.0), rho_(m+1,1e-5)
    , lam_(m,1.0), mu_(m,1.0), y_(m,0.0)
{
    DefaultPenalty(n_global_,m,a_,c_,d_);
    InitDeviceVectors(n_local,m,x_local,p0_,q0_,pij_,qij_,L_,U_,alpha_,beta_,xo1_,xo2_);
}

MMAOptimizerParallel::MMAOptimizerParallel(
    MPI_Comm comm, int n_local, int m,
    const mfem::Vector& x_local,
    const double* a, const double* c, const double* d)
    : comm_(comm), n_global_(ComputeNGlobal(comm,n_local)), n_local_(n_local), m_(m), iter_(0)
    , asyminit_(0.5), asymdec_(0.7), asyminc_(1.2)
    , z_(1.0)
    , b_(m,0.0), rho_(m+1,1e-5)
    , lam_(m,1.0), mu_(m,1.0), y_(m,0.0)
{
    a_.assign(a,a+m); c_.assign(c,c+m); d_.assign(d,d+m);
    InitDeviceVectors(n_local,m,x_local,p0_,q0_,pij_,qij_,L_,U_,alpha_,beta_,xo1_,xo2_);
}

MMAOptimizerParallel::MMAOptimizerParallel(
    MPI_Comm comm, int n_local, int m,
    const mfem::Vector& x_local,
    const mfem::Vector& a, const mfem::Vector& c, const mfem::Vector& d)
    : MMAOptimizerParallel(comm, n_local, m, x_local,
                           VecToDouble(a).data(),
                           VecToDouble(c).data(),
                           VecToDouble(d).data())
{}

/// @brief (Parallel) Store asymptote adaptation speeds.
void MMAOptimizerParallel::SetAsymptotes(mfem::real_t i, mfem::real_t d, mfem::real_t inc)
{ asyminit_=i; asymdec_=d; asyminc_=inc; }

/// @brief (Parallel) Update local asymptotes and build MMA sub-problem; Allreduces b.
void MMAOptimizerParallel::BuildSubproblem_(
    const mfem::Vector& x, const mfem::Vector& df0dx,
    const mfem::Vector& fival, const mfem::Vector* dfidx,
    const mfem::Vector& xmin, const mfem::Vector& xmax)
{
    bool ud = x.UseDevice();
    UpdateAsymptotes(n_local_,iter_,asyminit_,asymdec_,asyminc_,
        x.Read(),xo1_.Read(),xo2_.Read(),xmin.Read(),xmax.Read(),
        L_.Write(),U_.Write(),alpha_.Write(),beta_.Write(),ud);

    std::vector<const mfem::real_t*> dfi(m_);
    for (int i=0;i<m_;++i) dfi[i]=dfidx[i].Read();
    std::vector<double> rho_zero(m_+1,0.0);
    std::vector<double> b_loc(m_);
    BuildCoeffs(n_local_,m_,ud,
        x.Read(),L_.Read(),U_.Read(),xmin.Read(),xmax.Read(),
        df0dx.Read(),pij_,qij_,p0_,q0_,
        dfi.data(),rho_zero.data(),b_loc);
    MPI_Allreduce(b_loc.data(),b_.data(),m_,MPI_DOUBLE,MPI_SUM,comm_);
    for (int i=0;i<m_;++i) b_[i]=b_[i]-double(fival(i));
}

/// @brief (Parallel) Build GCMMA sub-problem with rho; Allreduces b.
void MMAOptimizerParallel::BuildSubproblemRho_(
    const mfem::Vector& x, const mfem::Vector& df0dx,
    const mfem::Vector& fival, const mfem::Vector* dfidx,
    const mfem::Vector& xmin, const mfem::Vector& xmax,
    const std::vector<double>& rho)
{
    bool ud = x.UseDevice();
    std::vector<const mfem::real_t*> dfi(m_);
    for (int i=0;i<m_;++i) dfi[i]=dfidx[i].Read();
    std::vector<double> b_loc(m_);
    BuildCoeffs(n_local_,m_,ud,
        x.Read(),L_.Read(),U_.Read(),xmin.Read(),xmax.Read(),
        df0dx.Read(),pij_,qij_,p0_,q0_,
        dfi.data(),rho.data(),b_loc);
    MPI_Allreduce(b_loc.data(),b_.data(),m_,MPI_DOUBLE,MPI_SUM,comm_);
    for (int i=0;i<m_;++i) b_[i]=b_[i]-double(fival(i));
}

/**
 * @brief (Parallel) One MMA outer iteration.
 *
 * Calls BuildSubproblem_() which contains MPI_Allreduce for b,
 * advances local history, then calls SolveDualIP() with comm_.
 * The Newton loop includes MPI_Allreduce for the dual gradient and
 * Hessian, so all ranks execute identical steps and hold identical
 * dual variables after each call.
 */
void MMAOptimizerParallel::Update(
    mfem::Vector& x, const mfem::Vector& df0dx, mfem::real_t f0val,
    const mfem::Vector& fival, const mfem::Vector* dfidx,
    const mfem::Vector& xmin, const mfem::Vector& xmax)
{
    bool ud = x.UseDevice();
    BuildSubproblem_(x,df0dx,fival,dfidx,xmin,xmax);
    xo2_=xo1_; xo1_=x;
    detail::SolveDualIP(comm_,n_local_,m_,n_eq_,ud,
        L_.Read(),U_.Read(),alpha_.Read(),beta_.Read(),
        p0_.HostRead(),q0_.HostRead(),pij_,qij_,
        b_,a_,c_,d_,lam_,mu_,y_,z_,x.ReadWrite());
    ++iter_;
}

/**
 * @brief (Parallel) One GCMMA outer iteration.
 *
 * rho initialisation performs MPI_Allreduce of local gradient-magnitude
 * sums so all ranks use the same rho values.  Then calls
 * BuildSubproblemRho_() (another Allreduce for b) and SolveDualIP().
 */
void MMAOptimizerParallel::UpdateGCMMA(
    mfem::Vector& x, const mfem::Vector& df0dx, mfem::real_t f0val,
    const mfem::Vector& fival, const mfem::Vector* dfidx,
    const mfem::Vector& xmin, const mfem::Vector& xmax, int* innerIter)
{
    bool ud = x.UseDevice();
    UpdateAsymptotes(n_local_,iter_,asyminit_,asymdec_,asyminc_,
        x.Read(),xo1_.Read(),xo2_.Read(),xmin.Read(),xmax.Read(),
        L_.Write(),U_.Write(),alpha_.Write(),beta_.Write(),ud);

    rho_.assign(m_+1,0.0);
    mfem::Vector d_tmp(n_local_); d_tmp.UseDevice(ud);
    std::vector<double> rho_loc(m_+1,0.0);
    {
        const auto* df0r=df0dx.Read();
        const auto* xmnr=xmin.Read(); const auto* xmxr=xmax.Read();
        auto* dt=d_tmp.Write();
        mfem::forall_switch(ud,n_local_,[=] MFEM_HOST_DEVICE (int j){
            double v=double(df0r[j]); if(v<0)v=-v;
            dt[j]=v*(double(xmxr[j])-double(xmnr[j]));
        });
        rho_loc[0]=d_tmp.Sum();
    }
    for (int i=0;i<m_;++i){
        const auto* dfir=dfidx[i].Read();
        const auto* xmnr=xmin.Read(); const auto* xmxr=xmax.Read();
        auto* dt=d_tmp.Write();
        mfem::forall_switch(ud,n_local_,[=] MFEM_HOST_DEVICE (int j){
            double v=double(dfir[j]); if(v<0)v=-v;
            dt[j]=v*(double(xmxr[j])-double(xmnr[j]));
        });
        rho_loc[i+1]=d_tmp.Sum();
    }
    MPI_Allreduce(rho_loc.data(),rho_.data(),m_+1,MPI_DOUBLE,MPI_SUM,comm_);
    for (int k=0;k<=m_;++k)
        rho_[k]=std::max(1e-6,0.5/(double)n_global_*rho_[k]);

    mfem::Vector xk(x);
    BuildSubproblemRho_(xk,df0dx,fival,dfidx,xmin,xmax,rho_);
    detail::SolveDualIP(comm_,n_local_,m_,n_eq_,ud,
        L_.Read(),U_.Read(),alpha_.Read(),beta_.Read(),
        p0_.HostRead(),q0_.HostRead(),pij_,qij_,
        b_,a_,c_,d_,lam_,mu_,y_,z_,x.ReadWrite());
    if (innerIter) *innerIter=1;
    xo2_=xo1_; xo1_=xk; ++iter_;
}

/**
 * @brief (Parallel) Global projected-gradient KKT residual.
 *
 * Assembles the local primal term on device, then reduces via
 * MPI_Allreduce(SUM) before dividing by n_global_.  The dual
 * complementarity term is computed on the host from replicated lambda.
 * Returns the same value on every rank.
 */
mfem::real_t MMAOptimizerParallel::KKTresidual(
    const mfem::Vector& x, const mfem::Vector& df0dx, double,
    const mfem::Vector& fival, const mfem::Vector* dfidx,
    const mfem::Vector& xmin, const mfem::Vector& xmax,
    double* lambda_out) const
{
    if (lambda_out) std::copy(lam_.begin(),lam_.end(),lambda_out);
    bool ud = x.UseDevice();
    mfem::Vector d_tmp(n_local_); d_tmp.UseDevice(ud);
    {
        const auto* xr=x.Read(); const auto* df0r=df0dx.Read();
        const auto* xmnr=xmin.Read(); const auto* xmxr=xmax.Read();
        auto* dt=d_tmp.Write();
        const double tol=1e-3;
        mfem::forall_switch(ud,n_local_,[=] MFEM_HOST_DEVICE (int j){
            dt[j]=double(df0r[j]);
        });
        for (int i=0;i<m_-2*n_eq_;++i){
            double li=lam_[i];
            const auto* dfir=dfidx[i].Read();
            auto* dtr=d_tmp.ReadWrite();
            mfem::forall_switch(ud,n_local_,[=] MFEM_HOST_DEVICE (int j){
                dtr[j]+=li*double(dfir[j]);
            });
        }
        for (int k=0;k<n_eq_;++k){
            const int ni2=m_-2*n_eq_;
            double lnet=lam_[ni2+k]-lam_[ni2+n_eq_+k];
            const auto* dfir=dfidx[ni2+k].Read();
            auto* dtr=d_tmp.ReadWrite();
            mfem::forall_switch(ud,n_local_,[=] MFEM_HOST_DEVICE (int j){
                dtr[j]+=lnet*double(dfir[j]);
            });
        }
        mfem::forall_switch(ud,n_local_,[=] MFEM_HOST_DEVICE (int j){
            double rng=double(xmxr[j])-double(xmnr[j]);
            double t=tol*rng;
            double g=double(dt[j]); double pg=g;
            if(double(xr[j])<=double(xmnr[j])+t) pg=g<0?g:0.0;
            if(double(xr[j])>=double(xmxr[j])-t) pg=g>0?g:0.0;
            dt[j]=pg*pg;
        });
    }
    double primal_loc = d_tmp.Sum();
    double primal_global = 0.0;
    MPI_Allreduce(&primal_loc,&primal_global,1,MPI_DOUBLE,MPI_SUM,comm_);
    double dual=0.0;
    for (int i=0;i<m_-2*n_eq_;++i){ double cs=lam_[i]*double(fival(i)); dual+=cs*cs; }
    for (int k=0;k<n_eq_;++k){ double hk=double(fival(m_-2*n_eq_+k)); dual+=hk*hk; }
    return (primal_global+dual)/(double)n_global_;
}


/**
 * @brief (Parallel) GCMMA with full inner conservatism loop.
 *
 * Parallel variant of the callback-based UpdateGCMMA.
 * d^k and f̃ computations use MPI_Allreduce for global sums.
 * eval_fi must be called on ALL ranks and return global values.
 */
void MMAOptimizerParallel::UpdateGCMMA(
    mfem::Vector& x_local,
    const mfem::Vector& df0dx_local,
    mfem::real_t f0val,
    const mfem::Vector& fival,
    const mfem::Vector* dfidx_local,
    const mfem::Vector& xmin_local,
    const mfem::Vector& xmax_local,
    EvalCallback eval_fi,
    int  max_inner,
    int* innerIter)
{
    bool ud=x_local.UseDevice();

    UpdateAsymptotes(n_local_,iter_,asyminit_,asymdec_,asyminc_,
        x_local.Read(),xo1_.Read(),xo2_.Read(),
        xmin_local.Read(),xmax_local.Read(),
        L_.Write(),U_.Write(),alpha_.Write(),beta_.Write(),ud);

    rho_.assign(m_+1,0.0);
    mfem::Vector d_tmp(n_local_); d_tmp.UseDevice(ud);
    std::vector<double> rho_loc(m_+1,0.0);
    {
        const auto* df0r=df0dx_local.Read();
        const auto* xmnr=xmin_local.Read(), *xmxr=xmax_local.Read();
        auto* dt=d_tmp.Write();
        mfem::forall_switch(ud,n_local_,[=] MFEM_HOST_DEVICE (int j){
            double v=double(df0r[j]); if(v<0)v=-v;
            dt[j]=v*(double(xmxr[j])-double(xmnr[j]));
        });
        rho_loc[0]=d_tmp.Sum();
    }
    for (int i=0;i<m_;++i){
        const auto* dfir=dfidx_local[i].Read();
        const auto* xmnr=xmin_local.Read(), *xmxr=xmax_local.Read();
        auto* dt=d_tmp.Write();
        mfem::forall_switch(ud,n_local_,[=] MFEM_HOST_DEVICE (int j){
            double v=double(dfir[j]); if(v<0)v=-v;
            dt[j]=v*(double(xmxr[j])-double(xmnr[j]));
        });
        rho_loc[i+1]=d_tmp.Sum();
    }
    MPI_Allreduce(rho_loc.data(),rho_.data(),m_+1,MPI_DOUBLE,MPI_SUM,comm_);
    for (int k=0;k<=m_;++k)
        rho_[k]=std::max(1e-6,0.5/(double)n_global_*rho_[k]);

    mfem::Vector xk(x_local);
    int nu=0;

    while (nu<max_inner){
        x_local=xk;
        BuildSubproblemRho_(xk,df0dx_local,fival,dfidx_local,
                            xmin_local,xmax_local,rho_);
        detail::SolveDualIP(comm_,n_local_,m_,n_eq_,ud,
            L_.Read(),U_.Read(),alpha_.Read(),beta_.Read(),
            p0_.HostRead(),q0_.HostRead(),pij_,qij_,
            b_,a_,c_,d_,lam_,mu_,y_,z_,x_local.ReadWrite());

        mfem::Vector fi_hat(m_);
        mfem::real_t f0_hat=0;
        eval_fi(x_local, fi_hat, f0_hat);   // all ranks

        // d^k: global reduce
        double dk_loc=0.0;
        const auto* xr  =x_local.HostRead();
        const auto* xkr =xk.HostRead();
        const auto* Lr  =L_.HostRead();
        const auto* Ur  =U_.HostRead();
        const auto* xmnr=xmin_local.HostRead();
        const auto* xmxr=xmax_local.HostRead();
        for (int j=0;j<n_local_;++j){
            double dUx=double(Ur[j])-double(xr[j]);
            double dxL=double(xr[j])-double(Lr[j]);
            double dxk=double(xr[j])-double(xkr[j]);
            double rng=std::max(double(xmxr[j])-double(xmnr[j]),1e-10);
            dk_loc+=(double(Ur[j])-double(Lr[j]))*dxk*dxk
                    /(dUx*dxL*rng+1e-30);
        }
        double dk=0.0;
        MPI_Allreduce(&dk_loc,&dk,1,MPI_DOUBLE,MPI_SUM,comm_);
        if (dk<1e-14) dk=1e-14;

        bool conservative=true;

        // Objective
        {
            double loc[2]={0,0};
            for (int j=0;j<n_local_;++j){
                double dUx =double(Ur[j])-double(xr[j]);
                double dxL =double(xr[j])-double(Lr[j]);
                double dUxk=double(Ur[j])-double(xkr[j]);
                double dxLk=double(xkr[j])-double(Lr[j]);
                loc[0]+=p0_.HostRead()[j]/dUx +q0_.HostRead()[j]/dxL;
                loc[1]+=p0_.HostRead()[j]/dUxk+q0_.HostRead()[j]/dxLk;
            }
            double g[2]; MPI_Allreduce(loc,g,2,MPI_DOUBLE,MPI_SUM,comm_);
            double b_obj   = g[1] - double(f0val);
            double ftilde0 = g[0] - b_obj;
            double delta   = (double(f0_hat) - ftilde0) / dk;
            if (delta>0.0){
                rho_[0]=std::min(1.1*(rho_[0]+delta),10.0*rho_[0]);
                conservative=false;
            }
        }

        // Constraints
        for (int i=0;i<m_;++i){
            double s_loc=0;
            for (int j=0;j<n_local_;++j){
                double dUx=double(Ur[j])-double(xr[j]);
                double dxL=double(xr[j])-double(Lr[j]);
                s_loc+=pij_[i].HostRead()[j]/dUx
                      +qij_[i].HostRead()[j]/dxL;
            }
            double sg=0; MPI_Allreduce(&s_loc,&sg,1,MPI_DOUBLE,MPI_SUM,comm_);
            double ftilde_i = sg - b_[i];
            double delta    = (double(fi_hat(i)) - ftilde_i) / dk;
            if (delta>0.0){
                rho_[i+1]=std::min(1.1*(rho_[i+1]+delta),10.0*rho_[i+1]);
                conservative=false;
            }
        }

        ++nu;
        if (conservative) break;
    }

    if (innerIter) *innerIter=nu;
    xo2_=xo1_; xo1_=xk; ++iter_;
}



// ============================================================
// SQOptimizer / SQOptimizerParallel implementations
// ============================================================

// ─────────────────────────────────────────────────────────────────────────────
// SQOptimizer / SQOptimizerParallel  —  implementation
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// Build the symmetric SQ asymptotes L = x−σ, U = x+σ and the move limits
//   α = max(xmin, x − 0.9σ),  β = min(xmax, x + 0.9σ).
//
// σ_j = sigma_scale · (xmax_j − xmin_j).  The 0.9 margin keeps x strictly
// inside (L, U) so that the rational MMA model remains well-defined.
//
static void BuildSQAsymptotes(
    int n_loc, double sigma_scale, bool use_dev,
    const mfem::real_t* x, const mfem::real_t* xmin, const mfem::real_t* xmax,
    mfem::real_t* L, mfem::real_t* U, mfem::real_t* alpha, mfem::real_t* beta)
{
    mfem::forall_switch(use_dev, n_loc, [=] MFEM_HOST_DEVICE (int j){
        double xj=double(x[j]), xn=double(xmin[j]), xx=double(xmax[j]);
        double sig=sigma_scale*(xx-xn); if(sig<1e-8) sig=1e-8;
        L[j]=mfem::real_t(xj-sig); U[j]=mfem::real_t(xj+sig);
        double a=xj-0.9*sig, b=xj+0.9*sig;
        alpha[j]=mfem::real_t(a<xn?xn:a); beta[j]=mfem::real_t(b>xx?xx:b);
    });
}
} // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// SQOptimizer  —  constructors
// ─────────────────────────────────────────────────────────────────────────────

SQOptimizer::SQOptimizer(int n, int m, const mfem::Vector& x)
    : n_(n), m_(m), z_(1.0)
    , b_(m,0.0), rho_(m+1,1e-5)
    , lam_(m,1.0), mu_(m,1.0), y_(m,0.0)
{
    DefaultPenalty(n, m, a_, c_, d_);
    bool ud=x.UseDevice();
    auto init=[&](mfem::Vector& v, int sz){ v.SetSize(sz); v.UseDevice(ud); };
    init(p0_,n); init(q0_,n); init(L_,n); init(U_,n);
    init(alpha_,n); init(beta_,n);
    pij_.resize(m); qij_.resize(m);
    for(int i=0;i<m;++i){ init(pij_[i],n); init(qij_[i],n); }
}

SQOptimizer::SQOptimizer(int n, int m, const mfem::Vector& x,
                          const double* a, const double* c, const double* d)
    : SQOptimizer(n, m, x)
{ a_.assign(a,a+m); c_.assign(c,c+m); d_.assign(d,d+m); }

// SQOptimizer::BuildSubproblem_
//
// One call per outer iteration (and per inner GCMMA trial).
// 1. Set symmetric asymptotes and move limits from the current x.
// 2. Call BuildCoeffs to compute p0, q0, pij, qij and the b vector.
//    rho_override passes the current ρ values (from SQRho) so that the
//    GCMMA curvature term is baked into the p/q coefficients.
//

void SQOptimizer::BuildSubproblem_(
    const mfem::Vector& x, const mfem::Vector& df0dx,
    const mfem::Vector& fival, const mfem::Vector* dfidx,
    const mfem::Vector& xmin, const mfem::Vector& xmax,
    const double* rho_override)
{
    bool ud=x.UseDevice();
    BuildSQAsymptotes(n_, sigma_scale_, ud,
        x.Read(), xmin.Read(), xmax.Read(),
        L_.Write(), U_.Write(), alpha_.Write(), beta_.Write());
    std::vector<const mfem::real_t*> dfi(m_);
    for(int i=0;i<m_;++i) dfi[i]=dfidx?dfidx[i].Read():nullptr;
    std::vector<double> rho_use(m_+1, 0.0);
    if(rho_override) for(int k=0;k<=m_;++k) rho_use[k]=rho_override[k];
    BuildCoeffs(n_,m_,ud, x.Read(),L_.Read(),U_.Read(),xmin.Read(),xmax.Read(),
        df0dx.Read(),pij_,qij_,p0_,q0_, dfi.data(),rho_use.data(),b_);
    for(int i=0;i<m_;++i) b_[i]-=double(fival(i));
}

// SQRho  —  compute the curvature parameter ρ for one gradient field.
//
// ρ must be large enough that the SQ model is a valid global upper bound on
// the true function.  Three estimates are taken and the maximum is used:
//
//   ρ_stable = max_j(0.5 · range_j²) / n          — stability floor
//   ρ_sum    = 0.5/n · Σⱼ |df_j| · range_j        — sum-based estimate
//   ρ_max    = (0.5/0.9) · max_j(|df_j| · range_j) — pointwise estimate
//
// The 0.5/0.9 factor in ρ_max ensures the move limit β−α = 0.9·σ satisfies
// the sufficient decrease condition for the quadratic model.
//
// For the parallel variant (SQRhoGlobal), all three quantities are reduced
// across MPI ranks before taking the max.
//
static double SQRho(bool ud, int n, mfem::Vector& tmp,
    const mfem::real_t* dfr, const mfem::real_t* xmnr, const mfem::real_t* xmxr)
{
    auto* dt=tmp.Write();
    mfem::forall_switch(ud,n,[=] MFEM_HOST_DEVICE (int j){
        double v=double(dfr[j]); if(v<0) v=-v;
        dt[j]=v*(double(xmxr[j])-double(xmnr[j]));
    });
    // Compute sum (for ρ_sum) and max (for ρ_max) of |df|·range on device.
    double s=tmp.Sum(), mx=tmp.Max();
    // reuse tmp for stability floor
    auto* dt2=tmp.Write();
    mfem::forall_switch(ud,n,[=] MFEM_HOST_DEVICE (int j){
        double r=double(xmxr[j])-double(xmnr[j]); dt2[j]=0.5*r*r;
    });
    double stable=tmp.Max()/(double)n;
    return std::max({1e-6, 0.5/(double)n*s, (0.5/0.9)*mx, stable});
}

// SQOptimizer::Update
//
// One plain SQ outer iteration:
//   1. Snapshot x^k (needed as the SQ linearisation point throughout the dual solve).
//   2. Compute ρ for each function from the gradient magnitudes (SQRho).
//   3. Build the subproblem (asymptotes, p/q coefficients, b vector).
//   4. Solve the dual via SolveDualSQ → x is updated in place.
//   5. Renormalise equality multipliers: extract the net ν = λ₊ − λ₋ and
//      reset the pair so that the larger slot carries |ν| and the smaller
//      holds a floor of 1e-3.  This keeps the warm start well-conditioned.
//

void SQOptimizer::Update(
    mfem::Vector& x, const mfem::Vector& df0dx, mfem::real_t,
    const mfem::Vector& fival, const mfem::Vector* dfidx,
    const mfem::Vector& xmin, const mfem::Vector& xmax)
{
    bool ud=x.UseDevice();
    mfem::Vector xk(x), tmp(n_); tmp.UseDevice(ud);
    rho_.assign(m_+1,0.0);
    rho_[0]=SQRho(ud,n_,tmp,df0dx.Read(),xmin.Read(),xmax.Read());
    for(int i=0;i<m_;++i) if(dfidx)
        rho_[i+1]=SQRho(ud,n_,tmp,dfidx[i].Read(),xmin.Read(),xmax.Read());
    BuildSubproblem_(x,df0dx,fival,dfidx,xmin,xmax,rho_.data());
    std::vector<double> fival_sq(m_);
    for(int i=0;i<m_;++i) fival_sq[i]=(m_>0)?double(fival(i)):0.0;
    detail::SolveDualSQ(MPI_COMM_SELF,n_,m_,n_eq_,ud,
        L_.Read(),U_.Read(),alpha_.Read(),beta_.Read(),
        p0_.HostRead(),q0_.HostRead(),pij_,qij_,
        a_,c_,lam_,mu_,y_,z_,x.ReadWrite(),
        xk.Read(),rho_.data(),fival_sq.data());

    if (n_eq_>0) {
        const int ni=m_-2*n_eq_;
        for (int k=0;k<n_eq_;++k) {
            // Extract net multiplier nu = lam[+h] - lam[-h].
            // Assign back without swapping: +h slot gets max(+nu, floor),
            // -h slot gets max(-nu, floor). This preserves sign information
            // across outer iterations and avoids the oscillation caused by
            // swapping the large value between slots when h changes sign.
            // mu is reset to match the new lam scale.
            double nu=lam_[ni+k]-lam_[ni+n_eq_+k];
            lam_[ni+k]       = std::max( nu, 1e-3);
            lam_[ni+n_eq_+k] = std::max(-nu, 1e-3);
            mu_[ni+k]        = lam_[ni+k];
            mu_[ni+n_eq_+k]  = lam_[ni+n_eq_+k];
        }
    }
    ++iter_;

}

// SQOptimizer::UpdateGCMMA
//
// GCMMA outer iteration: extends Update with an inner conservatism loop.
// After each candidate step x̃ is accepted from the dual solve, eval_fi is
// called to evaluate the true constraint values at x̃.  If any constraint is
// violated (fi(x̃) > g_i(x̃) + 1e-10), that constraint's ρ is doubled and
// the subproblem is rebuilt and re-solved.  Up to 10 inner iterations are
// allowed; after that the best available x̃ is accepted regardless.
//
// If eval_fi is nullptr the conservatism check is skipped (plain SQ).
//

void SQOptimizer::UpdateGCMMA(
    mfem::Vector& x, const mfem::Vector& df0dx, mfem::real_t,
    const mfem::Vector& fival, const mfem::Vector* dfidx,
    const mfem::Vector& xmin, const mfem::Vector& xmax,
    std::function<void(const mfem::Vector&, mfem::Vector&, mfem::Vector*)> eval_fi,
    int* innerIter)
{
    bool ud=x.UseDevice();
    mfem::Vector xk(x), tmp(n_); tmp.UseDevice(ud);
    rho_.assign(m_+1,0.0);
    rho_[0]=SQRho(ud,n_,tmp,df0dx.Read(),xmin.Read(),xmax.Read());
    for(int i=0;i<m_;++i) if(dfidx)
        rho_[i+1]=SQRho(ud,n_,tmp,dfidx[i].Read(),xmin.Read(),xmax.Read());
    std::vector<double> fival_sq(m_);
    for(int i=0;i<m_;++i) fival_sq[i]=(m_>0)?double(fival(i)):0.0;
    BuildSubproblem_(xk,df0dx,fival,dfidx,xmin,xmax,rho_.data());
    detail::SolveDualSQ(MPI_COMM_SELF,n_,m_,n_eq_,ud,
        L_.Read(),U_.Read(),alpha_.Read(),beta_.Read(),
        p0_.HostRead(),q0_.HostRead(),pij_,qij_,
        a_,c_,lam_,mu_,y_,z_,x.ReadWrite(),
        xk.Read(),rho_.data(),fival_sq.data());
    int nu=1;
    if(eval_fi){
        mfem::Vector fi_hat(m_>0?m_:1);
        for(int inner=0;inner<10;++inner){
            eval_fi(x,fi_hat,nullptr);
            bool ok=true;
            for(int i=0;i<m_;++i)
                if(double(fi_hat(i))>b_[i]+double(fival(i))+1e-10){ ok=false; rho_[i+1]*=2.0; }
            if(ok) break;
            for(int i=0;i<m_;++i) fival_sq[i]=(m_>0)?double(fival(i)):0.0;
            BuildSubproblem_(xk,df0dx,fival,dfidx,xmin,xmax,rho_.data());
            detail::SolveDualSQ(MPI_COMM_SELF,n_,m_,n_eq_,ud,
                L_.Read(),U_.Read(),alpha_.Read(),beta_.Read(),
                p0_.HostRead(),q0_.HostRead(),pij_,qij_,
                a_,c_,lam_,mu_,y_,z_,x.ReadWrite(),
                xk.Read(),rho_.data(),fival_sq.data());
            ++nu;
        }
    }
    if(innerIter) *innerIter=nu;
    if (n_eq_>0) {
        const int ni=m_-2*n_eq_;
        for (int k=0;k<n_eq_;++k) {
            // Extract net multiplier nu = lam[+h] - lam[-h].
            // Assign back without swapping: +h slot gets max(+nu, floor),
            // -h slot gets max(-nu, floor). This preserves sign information
            // across outer iterations and avoids the oscillation caused by
            // swapping the large value between slots when h changes sign.
            // mu is reset to match the new lam scale.
            double nu=lam_[ni+k]-lam_[ni+n_eq_+k];
            lam_[ni+k]       = std::max( nu, 1e-3);
            lam_[ni+n_eq_+k] = std::max(-nu, 1e-3);
            mu_[ni+k]        = lam_[ni+k];
            mu_[ni+n_eq_+k]  = lam_[ni+n_eq_+k];
        }
    }
    ++iter_;
}

// SQOptimizer::KKTresidual
//
// Evaluates the KKT conditions at (x, λ):
//
//   Primal stationarity:  ∂L/∂xⱼ = ∂f₀/∂xⱼ + Σᵢ λᵢ·∂fᵢ/∂xⱼ
//     → project to zero if x is at a bound (projected gradient)
//     → sum of squares, normalised by n
//
//   Dual complementarity:  λᵢ·fᵢ(x) = 0  (for inequalities)
//   Equality satisfaction:  hₖ(x) = 0     (for equalities)
//     → both added as squared terms, normalised by n
//
// Equality multipliers use the net value ν_k = λ[+h] − λ[−h] so that the
// sign-indeterminate ±h encoding is transparent to the caller.
//

mfem::real_t SQOptimizer::KKTresidual(
    const mfem::Vector& x, const mfem::Vector& df0dx, mfem::real_t,
    const mfem::Vector& fival, const mfem::Vector* dfidx,
    const mfem::Vector& xmin, const mfem::Vector& xmax) const
{
    bool ud=x.UseDevice();
    mfem::Vector d_tmp(n_); d_tmp.UseDevice(ud);
    { const auto* df0r=df0dx.Read(); auto* dt=d_tmp.Write();
      mfem::forall_switch(ud,n_,[=] MFEM_HOST_DEVICE (int j){ dt[j]=double(df0r[j]); }); }
    for(int i=0;i<m_-2*n_eq_;++i) if(dfidx){
        double li=lam_[i]; const auto* dfir=dfidx[i].Read(); auto* dtr=d_tmp.ReadWrite();
        mfem::forall_switch(ud,n_,[=] MFEM_HOST_DEVICE (int j){ dtr[j]+=li*double(dfir[j]); });
    }
    for(int k=0;k<n_eq_;++k) if(dfidx){
        const int ni2=m_-2*n_eq_;
        double lnet=lam_[ni2+k]-lam_[ni2+n_eq_+k];
        const auto* dfir=dfidx[ni2+k].Read(); auto* dtr=d_tmp.ReadWrite();
        mfem::forall_switch(ud,n_,[=] MFEM_HOST_DEVICE (int j){ dtr[j]+=lnet*double(dfir[j]); });
    }
    { const auto* xr=x.Read(); const auto* xmnr=xmin.Read(); const auto* xmxr=xmax.Read();
      auto* dt=d_tmp.ReadWrite(); const double tol=1e-3;
      mfem::forall_switch(ud,n_,[=] MFEM_HOST_DEVICE (int j){
          double g=dt[j], xj=double(xr[j]), xn=double(xmnr[j]), xx=double(xmxr[j]);
          double rng=xx-xn; if(rng<1e-10) rng=1e-10; double t=tol*rng;
          double pg=(xj<=xn+t&&g>0)?0.0:(xj>=xx-t&&g<0)?0.0:g; dt[j]=pg*pg; }); }
    double primal=d_tmp.Sum(), dual=0.0;
    for(int i=0;i<m_-2*n_eq_;++i){ double cs=lam_[i]*double(fival(i)); dual+=cs*cs; }
    for(int k=0;k<n_eq_;++k){ double hk=double(fival(m_-2*n_eq_+k)); dual+=hk*hk; }
    return (primal+dual)/(double)n_;
}

// ─────────────────────────────────────────────────────────────────────────────
// SQOptimizerParallel  —  MPI-parallel implementation
// ─────────────────────────────────────────────────────────────────────────────
//
// Mirrors SQOptimizer exactly.  The only differences are:
//
//  • n_global_ is determined at construction via MPI_Allreduce on n_local.
//  • BuildSubproblem_ reduces b_loc (per-rank partial sums of b) via
//    MPI_Allreduce before subtracting fi(xk).
//  • SQRhoGlobal reduces sum, max, and the stability term across all ranks
//    before computing ρ, ensuring every rank uses the same curvature.
//  • KKTresidual reduces the primal part of the squared norm across ranks.
//  • The dual solve (SolveDualSQ) is called with comm_, so all internal
//    MPI_Allreduce calls use the correct communicator.
//

SQOptimizerParallel::SQOptimizerParallel(
    MPI_Comm comm, int n_local, int m, const mfem::Vector& x_local)
    : comm_(comm), n_local_(n_local), m_(m), z_(1.0)
    , b_(m,0.0), rho_(m+1,1e-5), lam_(m,1.0), mu_(m,1.0), y_(m,0.0)
{
    long long nl=n_local;
    MPI_Allreduce(&nl,&n_global_,1,MPI_LONG_LONG,MPI_SUM,comm);
    DefaultPenalty((int)n_global_,m,a_,c_,d_);
    bool ud=x_local.UseDevice();
    auto init=[&](mfem::Vector& v){ v.SetSize(n_local); v.UseDevice(ud); };
    init(p0_); init(q0_); init(L_); init(U_); init(alpha_); init(beta_);
    pij_.resize(m); qij_.resize(m);
    for(int i=0;i<m;++i){ init(pij_[i]); init(qij_[i]); }
}

SQOptimizerParallel::SQOptimizerParallel(
    MPI_Comm comm, int n_local, int m, const mfem::Vector& x_local,
    const double* a, const double* c, const double* d)
    : SQOptimizerParallel(comm,n_local,m,x_local)
{ a_.assign(a,a+m); c_.assign(c,c+m); d_.assign(d,d+m); }

void SQOptimizerParallel::BuildSubproblem_(
    const mfem::Vector& x, const mfem::Vector& df0dx,
    const mfem::Vector& fival, const mfem::Vector* dfidx,
    const mfem::Vector& xmin, const mfem::Vector& xmax,
    const double* rho_override)
{
    bool ud=x.UseDevice();
    BuildSQAsymptotes(n_local_,sigma_scale_,ud,
        x.Read(),xmin.Read(),xmax.Read(),
        L_.Write(),U_.Write(),alpha_.Write(),beta_.Write());
    std::vector<const mfem::real_t*> dfi(m_);
    for(int i=0;i<m_;++i) dfi[i]=dfidx?dfidx[i].Read():nullptr;
    std::vector<double> rho_use(m_+1,0.0);
    if(rho_override) for(int k=0;k<=m_;++k) rho_use[k]=rho_override[k];
    std::vector<double> b_loc(m_,0.0);
    BuildCoeffs(n_local_,m_,ud, x.Read(),L_.Read(),U_.Read(),xmin.Read(),xmax.Read(),
        df0dx.Read(),pij_,qij_,p0_,q0_, dfi.data(),rho_use.data(),b_loc);
    MPI_Allreduce(b_loc.data(),b_.data(),m_,MPI_DOUBLE,MPI_SUM,comm_);
    for(int i=0;i<m_;++i) b_[i]-=double(fival(i));
}

// SQRhoGlobal: same formula as SQRho but all three quantities (sum, max,
// stability) are reduced across MPI ranks before taking the max.
// This guarantees that every rank uses the same ρ, which is required for the
// dual solve to produce consistent x updates across the partition.
//
static double SQRhoGlobal(MPI_Comm comm, bool ud, int n_local, long long n_global,
    mfem::Vector& tmp, const mfem::real_t* dfr,
    const mfem::real_t* xmnr, const mfem::real_t* xmxr)
{
    auto* dt=tmp.Write();
    mfem::forall_switch(ud,n_local,[=] MFEM_HOST_DEVICE (int j){
        double v=double(dfr[j]); if(v<0) v=-v;
        dt[j]=v*(double(xmxr[j])-double(xmnr[j]));
    });
    double loc_s=tmp.Sum(), loc_mx=tmp.Max(), glb_s=0.0, glb_mx=0.0;
    MPI_Allreduce(&loc_s, &glb_s, 1,MPI_DOUBLE,MPI_SUM,comm);
    MPI_Allreduce(&loc_mx,&glb_mx,1,MPI_DOUBLE,MPI_MAX,comm);
    auto* dt2=tmp.Write();
    mfem::forall_switch(ud,n_local,[=] MFEM_HOST_DEVICE (int j){
        double r=double(xmxr[j])-double(xmnr[j]); dt2[j]=0.5*r*r;
    });
    double loc_st=tmp.Max(), glb_st=0.0;
    MPI_Allreduce(&loc_st,&glb_st,1,MPI_DOUBLE,MPI_MAX,comm);
    glb_st/=(double)n_global;
    return std::max({1e-6, 0.5/(double)n_global*glb_s, (0.5/0.9)*glb_mx, glb_st});
}

void SQOptimizerParallel::Update(
    mfem::Vector& x, const mfem::Vector& df0dx, mfem::real_t,
    const mfem::Vector& fival, const mfem::Vector* dfidx,
    const mfem::Vector& xmin, const mfem::Vector& xmax)
{
    bool ud=x.UseDevice();
    mfem::Vector xk(x), tmp(n_local_); tmp.UseDevice(ud);
    rho_.assign(m_+1,0.0);
    rho_[0]=SQRhoGlobal(comm_,ud,n_local_,n_global_,tmp,
        df0dx.Read(),xmin.Read(),xmax.Read());
    for(int i=0;i<m_;++i) if(dfidx)
        rho_[i+1]=SQRhoGlobal(comm_,ud,n_local_,n_global_,tmp,
            dfidx[i].Read(),xmin.Read(),xmax.Read());
    BuildSubproblem_(x,df0dx,fival,dfidx,xmin,xmax,rho_.data());
    std::vector<double> fival_sq(m_);
    for(int i=0;i<m_;++i) fival_sq[i]=(m_>0)?double(fival(i)):0.0;
    detail::SolveDualSQ(comm_,n_local_,m_,n_eq_,ud,
        L_.Read(),U_.Read(),alpha_.Read(),beta_.Read(),
        p0_.HostRead(),q0_.HostRead(),pij_,qij_,
        a_,c_,lam_,mu_,y_,z_,x.ReadWrite(),
        xk.Read(),rho_.data(),fival_sq.data());

    if (n_eq_>0) {
        const int ni=m_-2*n_eq_;
        for (int k=0;k<n_eq_;++k) {
            // Extract net multiplier nu = lam[+h] - lam[-h].
            // Assign back without swapping: +h slot gets max(+nu, floor),
            // -h slot gets max(-nu, floor). This preserves sign information
            // across outer iterations and avoids the oscillation caused by
            // swapping the large value between slots when h changes sign.
            // mu is reset to match the new lam scale.
            double nu=lam_[ni+k]-lam_[ni+n_eq_+k];
            lam_[ni+k]       = std::max( nu, 1e-3);
            lam_[ni+n_eq_+k] = std::max(-nu, 1e-3);
            mu_[ni+k]        = lam_[ni+k];
            mu_[ni+n_eq_+k]  = lam_[ni+n_eq_+k];
        }
    }
    ++iter_;

}

void SQOptimizerParallel::UpdateGCMMA(
    mfem::Vector& x, const mfem::Vector& df0dx, mfem::real_t,
    const mfem::Vector& fival, const mfem::Vector* dfidx,
    const mfem::Vector& xmin, const mfem::Vector& xmax,
    std::function<void(const mfem::Vector&, mfem::Vector&, mfem::Vector*)> eval_fi,
    int* innerIter)
{
    bool ud=x.UseDevice();
    mfem::Vector xk(x), tmp(n_local_); tmp.UseDevice(ud);
    rho_.assign(m_+1,0.0);
    rho_[0]=SQRhoGlobal(comm_,ud,n_local_,n_global_,tmp,
        df0dx.Read(),xmin.Read(),xmax.Read());
    for(int i=0;i<m_;++i) if(dfidx)
        rho_[i+1]=SQRhoGlobal(comm_,ud,n_local_,n_global_,tmp,
            dfidx[i].Read(),xmin.Read(),xmax.Read());
    std::vector<double> fival_sq(m_);
    for(int i=0;i<m_;++i) fival_sq[i]=(m_>0)?double(fival(i)):0.0;
    BuildSubproblem_(xk,df0dx,fival,dfidx,xmin,xmax,rho_.data());
    detail::SolveDualSQ(comm_,n_local_,m_,n_eq_,ud,
        L_.Read(),U_.Read(),alpha_.Read(),beta_.Read(),
        p0_.HostRead(),q0_.HostRead(),pij_,qij_,
        a_,c_,lam_,mu_,y_,z_,x.ReadWrite(),
        xk.Read(),rho_.data(),fival_sq.data());
    int nu=1;
    if(eval_fi){
        mfem::Vector fi_hat(m_>0?m_:1);
        for(int inner=0;inner<10;++inner){
            eval_fi(x,fi_hat,nullptr);
            bool ok=true;
            for(int i=0;i<m_;++i)
                if(double(fi_hat(i))>b_[i]+double(fival(i))+1e-10){ ok=false; rho_[i+1]*=2.0; }
            if(ok) break;
            for(int i=0;i<m_;++i) fival_sq[i]=(m_>0)?double(fival(i)):0.0;
            BuildSubproblem_(xk,df0dx,fival,dfidx,xmin,xmax,rho_.data());
            detail::SolveDualSQ(comm_,n_local_,m_,n_eq_,ud,
                L_.Read(),U_.Read(),alpha_.Read(),beta_.Read(),
                p0_.HostRead(),q0_.HostRead(),pij_,qij_,
                a_,c_,lam_,mu_,y_,z_,x.ReadWrite(),
                xk.Read(),rho_.data(),fival_sq.data());
            ++nu;
        }
    }
    if(innerIter) *innerIter=nu;
    if (n_eq_>0) {
        const int ni=m_-2*n_eq_;
        for (int k=0;k<n_eq_;++k) {
            // Extract net multiplier nu = lam[+h] - lam[-h].
            // Assign back without swapping: +h slot gets max(+nu, floor),
            // -h slot gets max(-nu, floor). This preserves sign information
            // across outer iterations and avoids the oscillation caused by
            // swapping the large value between slots when h changes sign.
            // mu is reset to match the new lam scale.
            double nu=lam_[ni+k]-lam_[ni+n_eq_+k];
            lam_[ni+k]       = std::max( nu, 1e-3);
            lam_[ni+n_eq_+k] = std::max(-nu, 1e-3);
            mu_[ni+k]        = lam_[ni+k];
            mu_[ni+n_eq_+k]  = lam_[ni+n_eq_+k];
        }
    }
    ++iter_;
}

mfem::real_t SQOptimizerParallel::KKTresidual(
    const mfem::Vector& x, const mfem::Vector& df0dx, mfem::real_t,
    const mfem::Vector& fival, const mfem::Vector* dfidx,
    const mfem::Vector& xmin, const mfem::Vector& xmax) const
{
    bool ud=x.UseDevice();
    mfem::Vector d_tmp(n_local_); d_tmp.UseDevice(ud);
    { const auto* df0r=df0dx.Read(); auto* dt=d_tmp.Write();
      mfem::forall_switch(ud,n_local_,[=] MFEM_HOST_DEVICE (int j){ dt[j]=double(df0r[j]); }); }
    for(int i=0;i<m_-2*n_eq_;++i) if(dfidx){
        double li=lam_[i]; const auto* dfir=dfidx[i].Read(); auto* dtr=d_tmp.ReadWrite();
        mfem::forall_switch(ud,n_local_,[=] MFEM_HOST_DEVICE (int j){ dtr[j]+=li*double(dfir[j]); });
    }
    for(int k=0;k<n_eq_;++k) if(dfidx){
        const int ni2=m_-2*n_eq_;
        double lnet=lam_[ni2+k]-lam_[ni2+n_eq_+k];
        const auto* dfir=dfidx[ni2+k].Read(); auto* dtr=d_tmp.ReadWrite();
        mfem::forall_switch(ud,n_local_,[=] MFEM_HOST_DEVICE (int j){ dtr[j]+=lnet*double(dfir[j]); });
    }
    { const auto* xr=x.Read(); const auto* xmnr=xmin.Read(); const auto* xmxr=xmax.Read();
      auto* dt=d_tmp.ReadWrite(); const double tol=1e-3;
      mfem::forall_switch(ud,n_local_,[=] MFEM_HOST_DEVICE (int j){
          double g=dt[j], xj=double(xr[j]), xn=double(xmnr[j]), xx=double(xmxr[j]);
          double rng=xx-xn; if(rng<1e-10) rng=1e-10; double t=tol*rng;
          double pg=(xj<=xn+t&&g>0)?0.0:(xj>=xx-t&&g<0)?0.0:g; dt[j]=pg*pg; }); }
    double primal_loc=d_tmp.Sum(), primal_global=0.0;
    MPI_Allreduce(&primal_loc,&primal_global,1,MPI_DOUBLE,MPI_SUM,comm_);
    double dual=0.0;
    for(int i=0;i<m_-2*n_eq_;++i){ double cs=lam_[i]*double(fival(i)); dual+=cs*cs; }
    for(int k=0;k<n_eq_;++k){ double hk=double(fival(m_-2*n_eq_+k)); dual+=hk*hk; }
    return (primal_global+dual)/(double)n_global_;
}

} // namespace mfem_mma
