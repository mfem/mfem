/**
 * MMA_MFEM.cpp  —  Device-aware MMA for MFEM (CUDA / HIP / CPU)
 *
 * Device strategy
 * ───────────────
 * • use_dev is read from x.UseDevice() at each Update() call.
 * • All O(n) loops run via mfem::forall_switch(use_dev, n, lambda)
 *   so they execute on GPU when x lives on device, or CPU otherwise.
 * • Device pointers are obtained through mfem::Vector::Read() /
 *   Write() / ReadWrite() which trigger the MFEM memory manager.
 * • The m×m dual Newton system is always assembled and solved on the
 *   CPU. Reductions (dual grad/Hess sums over n) are done on-device
 *   via MFEM InnerProduct, bringing only m or m² scalars to the host.
 * • pij_[i] / qij_[i] / p0_ / q0_ are mfem::Vector so their memory
 *   follows the device flag of x automatically.
 */

#include "MMA_MFEM.hpp"
#include <stdexcept>
#include <string>
#include <numeric>
#include <cmath>

// ── LAPACK ────────────────────────────────────────────────────────────────
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

// ── MPI type helper ───────────────────────────────────────────────────────
static MPI_Datatype MpiTypeReal()
{ return (sizeof(mfem::real_t)==sizeof(double)) ? MPI_DOUBLE : MPI_FLOAT; }

// ── Default penalty ───────────────────────────────────────────────────────
// Convert mfem::Vector (real_t) to std::vector<double> for penalty arrays.
static std::vector<double> VecToDouble(const mfem::Vector& v)
{
    const mfem::real_t* h = v.HostRead();
    return std::vector<double>(h, h + v.Size());
}

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

// ============================================================
// SolveDense: LU with SVD fallback for rank-deficient systems
// ============================================================
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
        double rcond = 2.2e-16 * m;
        std::vector<double> svals(m), K_svd(K);
        int nlvl   = std::max(0,(int)std::ceil(std::log2(double(m)/25.0+1.0))+1);
        int liwork = std::max(1, 3*m*nlvl+11*m);
        std::vector<int> iwork(liwork);
        int lwork = -1; double wq;
        dgelsd_(&m,&m,&nrhs,K_svd.data(),&m,rhs.data(),&m,
                svals.data(),&rcond,&rank,&wq,&lwork,iwork.data(),&info);
        lwork = (info==0)?(int)wq:10*m*m;
        lwork = std::max(lwork,1);
        std::vector<double> work(lwork);
        K_svd = K;
        dgelsd_(&m,&m,&nrhs,K_svd.data(),&m,rhs.data(),&m,
                svals.data(),&rcond,&rank,work.data(),&lwork,iwork.data(),&info);
        if (info != 0) {
            // dgelsd did not fully converge (info > 0 means SVD iteration
            // stalled after info superdiagonals failed to converge).
            // LAPACK still provides the best available solution in rhs[].
            // We accept it rather than throwing, so the Newton loop can
            // continue: the step may be suboptimal but the outer ε-reduction
            // and line-search will compensate.
            // Only re-throw on bad-argument errors (info < 0).
            if (info < 0)
                throw std::runtime_error(
                    "MMA_MFEM: dgelsd bad arg (info="+std::to_string(info)+")");
            // info > 0: partial convergence — use best available solution
            // (rhs already contains the least-squares approximation)
        }
    }
}

// ============================================================
// SolveDualIP  —  device-aware interior-point dual solver
//
// n-loops run on device via forall_switch.
// m-reductions are done via device InnerProduct → host doubles.
// The m×m Newton system is solved on host by SolveDense.
// ============================================================
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

    double lam_max = (m > 2*n_eq) ?
        *std::max_element(lam.begin(), lam.begin()+(m-2*n_eq)) : 1.0;
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

    // ── PrimalFromDual: runs entirely on device ───────────────────────────
    auto PrimalFromDual = [&]() {
        double lamai=0.0;
        for (int i=0;i<m;++i){
            lam[i]=std::max(lam[i],0.0);
            y[i]=std::max(0.0,lam[i]-c_pen[i]);
            lamai+=a_pen[i]*lam[i];
        }
        z = std::max(0.0, 10.0*(lamai-1.0));

        // Build lam-weighted p/q sums on device, compute xj
        // We need p0+Σpij*lam and q0+Σqij*lam for each j.
        // Approach: start from p0/q0, add pij*lam[i] iteratively.
        // Use d_tmp as accumulator for pjlam, d_df2 for qjlam.
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

    // ── DualGrad: device sum → Allreduce → host m-vector ─────────────────
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

    // ── DualHess: device outer products → Allreduce → host m²-matrix ─────
    // We precompute:
    //   df2_j = -1 / (2*pjlam/dUx³ + 2*qjlam/dxL³)
    //   PQ_ij = pij_j/dUx² - qij_j/dxL²
    // Then H_rc = Σ_j PQ_rj * df2_j * PQ_cj = InnerProduct(PQ_r.*df2, PQ_c)
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

    // ── DualResidual: device sums → host m-vector ─────────────────────────
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

    // ── Line search ───────────────────────────────────────────────────────
    auto LineSearch = [&](const std::vector<double>& s) {
        double theta=1.005;
        for (int i=0;i<m-2*n_eq;++i){
            // inequality: lam and mu must stay positive
            if (theta < -1.01*s[i]   /lam[i]) theta=-1.01*s[i]   /lam[i];
            if (theta < -1.01*s[m+i] /mu[i])  theta=-1.01*s[m+i] /mu[i];
        }
        theta=1.0/theta;
        for (int i=0;i<m;++i){ lam[i]+=theta*s[i]; mu[i]+=theta*s[m+i]; }
    };

    // ── Main Newton loop (host logic, device data) ────────────────────────
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

} // namespace detail

// ============================================================
// Shared: device-aware asymptote update
// ============================================================
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

// ============================================================
// Shared: build subproblem coefficients (device-aware)
// p0, q0, pij, qij are mfem::Vector with device memory.
// b_loc is host (m×1).
// ============================================================
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

// ============================================================
// MMAOptimizer  (serial)
// ============================================================

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


} // namespace mfem_mma
