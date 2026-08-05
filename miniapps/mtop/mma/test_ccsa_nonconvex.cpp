/**
 * test_ccsa_nonconvex.cpp  —  Non-convex large-scale CCSA/GCMMA test suite
 *
 * Implements density-filtered SIMP topology optimisation on a 1D domain —
 * the exact convergence mechanism of real topology optimisation without
 * requiring a FEM linear solve. Identical problem set to test_mma_nonconvex.cpp
 * (the MMA suite), with mfem_mma::MMAOptimizerParallel replaced by
 * mfem_mma::CCSAOptimizerParallel, so iteration counts and objective drops
 * are directly comparable between the two optimizers.
 *
 * CCSAOptimizerParallel works directly on the LATENT variable eta:
 * Update()/UpdateGCMMA()/KKTresidual() take/return eta, not the physical
 * design, and take no xmin/xmax. RunFilteredSIMP() below builds a
 * BoundsGeometry up front, seeds eta = opt.ToLatent(x0), and refreshes the
 * physical view x = opt.ToPhysical(eta) every time the filter/objective
 * (EvalF/EvalFi, both physical-space) need to be evaluated.
 *
 * Density filtering
 * ─────────────────
 *   The filtered density x̂ⱼ = Σₖ Hⱼₖ xₖ / Σₖ Hⱼₖ  (normalised Gaussian)
 *
 *   H_{jk} = exp(−(j−k)²/(2r²))   for |j−k| ≤ 3r, else 0.
 *
 *   The objective uses x̂ instead of x:
 *     f(x) = (1/n) Σⱼ wⱼ / x̂ⱼᵖ
 *   Gradient via chain rule:
 *     ∂f/∂xₑ = Σⱼ (∂f/∂x̂ⱼ) Hⱼₑ / (Σₖ Hⱼₖ)
 *             = Σⱼ [−p wⱼ / (n x̂ⱼᵖ⁺¹)] · H̃ⱼₑ   where H̃=normalised H
 *
 *   Why this needs 100–500 iterations:
 *   • Each element xₑ influences all x̂ⱼ within radius r.
 *   • After each CCSA step, ALL filtered densities shift, changing gradients.
 *   • The filter creates spatially correlated oscillation modes that CCSA's
 *     fixed bounds geometry (no asymptotes — see CCSA_Bregman_MFEM.hpp) must
 *     dampen one spatial frequency at a time via the rho (conservatism)
 *     weights instead of moving L/U asymptotes.
 *   • Filter radius r=5: ~10 coupled neighbours → ~100–300 iterations.
 *   • Filter radius r=10: ~20 coupled neighbours → ~200–500 iterations.
 *   • This is EXACTLY the convergence behaviour of density-filtered 2D/3D
 *     SIMP topology optimisation (Lazarov & Sigmund 2016).
 *
 * Volume constraint
 * ─────────────────
 *   Encoded as equality via two inequalities:
 *     fi(0)= mean(x) − Vfrac ≤ 0  (upper)
 *     fi(1)= Vfrac − mean(x) ≤ 0  (lower)
 *
 *   This keeps mean(x) = Vfrac throughout — material is always being
 *   redistributed, never accumulating or depleting.
 *
 * Weight pattern
 * ──────────────
 *   Spatially varying load: wⱼ = 1 + sin(2π·q·j/n) for q spatial modes.
 *   This breaks the symmetry so the uniform design is NOT a KKT point,
 *   forcing the elements to spatially redistribute according to the load.
 *
 * Problems
 * ────────
 *   P1: r=5,  q=4 modes,  p=3 (fixed)           → ~100–200 iterations
 *   P2: r=5,  q=3 modes,  p=5 (fixed, GCMMA)    → globalization stress
 *   P3: r=10, q=3 modes,  p: 1→5 continuation → ~200–500 iterations
 *
 * Build:  cmake --build build   (links MMA_MFEM.cpp + CCSA_Bregman_MFEM.cpp)
 * Run:    ./build/test_ccsa_nonconvex
 *         mpirun -np 4 ./build/test_ccsa_nonconvex
 *         ./build/test_ccsa_nonconvex --large
 */

#include "CCSA_Bregman_MFEM.hpp"
#include <mfem.hpp>
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <string>
#include <chrono>

using namespace mfem;
using namespace mfem_mma;
using Clock = std::chrono::steady_clock;

static int  g_rank   = 0;
static int  g_nranks = 1;
static int  g_nfail  = 0;
static bool g_large  = false;

static void Check(bool cond, const char* msg)
{
    if(g_rank!=0) return;
    if(cond) printf("  [PASS] %s\n",msg);
    else    {printf("  [FAIL] %s\n",msg);++g_nfail;}
}
static double GSum(double v)
{double g;MPI_Allreduce(&v,&g,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);return g;}

static std::pair<int,int> Distribute(int n)
{int b=n/g_nranks,r=n%g_nranks;return{b+(g_rank<r?1:0),g_rank*b+std::min(g_rank,r)};}

// ── Pre-compute the Gaussian filter kernel (local stencil) ─────────────────
// For each local element e (global index g=off+e), store:
//   filter_idx(e)  : global indices of neighbours within 3r
//   filter_wgt(e)  : normalised weights (sum=1)
//
// We compute x̂ locally using ghost values from neighbours via MPI.
// For simplicity: gather the full x to all ranks (fine for n ≤ 1M
// since it's just one float per element and happens once per iteration).
struct Filter {
    int n_global, r;
    std::vector<std::vector<int>>    idx; // [n_local][stencil_size]
    std::vector<std::vector<double>> wgt; // normalised weights

    Filter(int ng, int nl, int off, int radius) : n_global(ng), r(radius)
    {
        idx.resize(nl); wgt.resize(nl);
        for(int e=0;e<nl;++e){
            int g=off+e;
            double wsum=0;
            std::vector<int>    nbr;
            std::vector<double> w;
            int lo=std::max(0,g-3*r), hi=std::min(ng-1,g+3*r);
            for(int k=lo;k<=hi;++k){
                double d=double(g-k);
                double wi=std::exp(-d*d/(2.0*r*r));
                nbr.push_back(k); w.push_back(wi); wsum+=wi;
            }
            for(auto& wi:w) wi/=wsum;
            idx[e]=nbr; wgt[e]=w;
        }
    }

    // Apply filter: x_hat = H x  (needs full x vector, gathered)
    void apply(const std::vector<double>& x_full, int nl, int /*off*/,
               std::vector<double>& x_hat) const
    {
        x_hat.resize(nl);
        for(int e=0;e<nl;++e){
            double s=0;
            for(int i=0;i<(int)idx[e].size();++i)
                s+=wgt[e][i]*x_full[idx[e][i]];
            x_hat[e]=s;
        }
    }

    // Accumulate this rank's rows of H^T*sens_hat into a global-indexed
    // vector.  The caller allreduces these contributions across ranks.
    // Row normalization makes H nonsymmetric near boundaries, so applying H
    // again would not be the correct adjoint.
    void adjoint_contribution(const std::vector<double>& sens_hat,
                              std::vector<double>& sens_full) const
    {
        sens_full.assign(n_global,0.0);
        for(int e=0;e<(int)idx.size();++e)
            for(int i=0;i<(int)idx[e].size();++i)
                sens_full[idx[e][i]] += sens_hat[e]*wgt[e][i];
    }
};

// Gather full vector to all ranks
static std::vector<double> GatherFull(int n_global, int nl, int off,
                                       const Vector& x, MPI_Comm comm)
{
    std::vector<double> full(n_global,0.0);
    for(int j=0;j<nl;++j) full[off+j]=double(x(j));
    MPI_Allreduce(MPI_IN_PLACE,full.data(),n_global,MPI_DOUBLE,MPI_SUM,comm);
    return full;
}

// ── Run one filtered SIMP test ─────────────────────────────────────────────
struct Result {
    int iters=0;
    double kkt_final=1.0, max_viol=1.0;
    double f0_init=0.0, f0_final=0.0;
    double grayness_initial=1.0, grayness=1.0;
    bool kkt_finite=true;
};

static Result RunFilteredSIMP(
    int n_global, int m, double Vfrac,
    double simp_p, bool continuation,
    int filter_r, int load_modes,
    bool gcmma, int max_iter)
{
    auto [nl,off] = Distribute(n_global);
    MPI_Comm comm = MPI_COMM_WORLD;

    // Load = checkerboard at period 2r (frustrates filter) 
    //      + slow sinusoidal envelope at period n/load_modes (breaks block symmetry)
    //      + small random noise (unique per element — prevents identical-block lockout)
    //
    // The slow envelope makes each pair of r-blocks slightly different from
    // its neighbours, so CCSA cannot treat all blocks identically.
    // The noise is tiny (±0.02) so it doesn't dominate the gradient.
    const int mode1 = load_modes > 0 ? load_modes : 3;
    const int mode2 = 2*mode1 + 1;
    std::vector<double> w_local(nl);
    // LCG for reproducible per-element noise
    uint64_t lcg_s = 314159265ULL;
    for(int gg=0;gg<off;++gg){lcg_s=lcg_s*6364136223846793005ULL+1442695040888963407ULL;}
    for(int e=0;e<nl;++e){
        int g=off+e;
        lcg_s=lcg_s*6364136223846793005ULL+1442695040888963407ULL;
        double noise = 0.04*(double(lcg_s>>33)/double(1ULL<<31) - 0.5); // ±0.02
        double sq1 = (g/filter_r)%2==0 ? 1.0 : -1.0;           // period 2r
        double sq2 = (g/(2*filter_r))%2==0 ? 1.0 : -1.0;        // period 4r
        // Slow envelope breaks block symmetry across the domain
        double env = 1.0 + 0.3*std::sin(2.0*M_PI*mode1*g/n_global)
                        + 0.15*std::sin(2.0*M_PI*mode2*g/n_global);
        w_local[e] = env * (1.0 + 0.40*sq1 + 0.15*sq2) + noise;
        if(w_local[e]<0.05) w_local[e]=0.05;
    }

    // Build filter
    Filter filt(n_global,nl,off,filter_r);

    // Start from a common feasible design.  With regional constraints, a
    // uniform x=Vfrac violates every even-numbered block's tighter
    // Vfrac-0.05 upper bound, making objective "drop" from that baseline
    // meaningless.  Initialize each block at its own alternating bound;
    // the even block counts used below preserve the global mean Vfrac.
    const double xmin_v=0.01;
    Vector xmin_v_(nl),xmax_v_(nl),df0(nl);
    xmin_v_=real_t(xmin_v); xmax_v_=1.0;

    // Constraint dg (unfiltered x, so volume constraint uses raw x)
    // m constraints:
    //   fi(0): mean(x) - Vfrac ≤ 0
    //   fi(1): Vfrac - mean(x) ≤ 0
    //   fi[2..m-1]: regional upper bounds
    std::vector<Vector> dg(m);
    for(int k=0;k<m;++k){dg[k].SetSize(nl);dg[k]=0.0;}
    for(int j=0;j<nl;++j){
        dg[0](j)=real_t(+1.0/n_global);
        if(m>1) dg[1](j)=real_t(-1.0/n_global);
    }
    // Regional blocks: split into m-2 equal parts
    if(m>2){
        int nb=m-2, bw=n_global/nb;
        for(int j=0;j<nl;++j){
            int g=off+j;
            for(int k=2;k<m;++k){
                int b=k-2, bs=b*bw, be=(b<nb-1)?(b+1)*bw:n_global;
                if(g>=bs&&g<be) dg[k](j)=real_t(1.0/(be-bs));
            }
        }
    }

    const double cv=std::max(1000.0,10.0*n_global);
    std::vector<double> av(m,0),cv_v(m,cv),dv(m,1);
    BoundsGeometry bounds = BoundsGeometry::TwoSided(xmin_v_, xmax_v_);
    CCSAOptimizerParallel opt(comm,nl,m,bounds,av.data(),cv_v.data(),dv.data());
    Vector x0(nl); x0=real_t(Vfrac);
    if(m>2){
        const int nb=m-2, bw=n_global/nb;
        for(int j=0;j<nl;++j){
            const int g=off+j;
            const int b=std::min(g/bw,nb-1);
            x0(j)=real_t(Vfrac+0.05*(b%2==0?-1.0:1.0));
        }
    }
    Vector eta = opt.ToLatent(x0);
    Vector x(nl); x = x0;

    auto EvalFi=[&](const Vector& xp)->mfem::Vector{
        double xloc=0; for(int j=0;j<nl;++j) xloc+=double(xp(j));
        double xm=GSum(xloc)/n_global;
        mfem::Vector fi(m);
        fi(0)=real_t(xm-Vfrac);
        if(m>1) fi(1)=real_t(Vfrac-xm);
        if(m>2){
            int nb=m-2, bw=n_global/nb;
            for(int k=2;k<m;++k){
                int b=k-2, bs=b*bw, be=(b<nb-1)?(b+1)*bw:n_global;
                double sl=0;
                for(int j=0;j<nl;++j){int g=off+j;if(g>=bs&&g<be)sl+=double(xp(j))/(be-bs);}
                fi(k)=real_t(GSum(sl)-(Vfrac+0.05*(b%2==0?-1.0:1.0)));
            }
        }
        return fi;
    };

    auto EvalF=[&](const Vector& xp, double p, bool compute_gradient)->double{
        // Gather full x
        auto xfull=GatherFull(n_global,nl,off,xp,comm);
        // Filter
        std::vector<double> xhat;
        filt.apply(xfull,nl,off,xhat);
        // f and sensitivity w.r.t. x_hat
        std::vector<double> sens_hat(nl);
        double f_loc=0;
        for(int e=0;e<nl;++e){
            double xhe=std::max(xhat[e],xmin_v);
            double xhp=std::pow(xhe,p);
            f_loc+=w_local[e]/xhp;
            sens_hat[e]=-p*w_local[e]/(xhp*xhe)/n_global; // df/d(x_hat_e)
        }
        double f=GSum(f_loc)/n_global;
        if(compute_gradient){
            std::vector<double> grad_full;
            filt.adjoint_contribution(sens_hat,grad_full);
            MPI_Allreduce(MPI_IN_PLACE,grad_full.data(),n_global,
                          MPI_DOUBLE,MPI_SUM,comm);
            for(int e=0;e<nl;++e) df0(e)=real_t(grad_full[off+e]);
        }
        return f;
    };

    Result res;
    // Use the untouched uniform design at the requested final SIMP exponent
    // as a common baseline.  Recording the post-update value made CCSA and
    // GCMMA objective-drop percentages incomparable because their first steps
    // differ.  Using simp_p here also gives continuation runs a same-scale
    // initial/final comparison.
    res.f0_init=EvalF(x,simp_p,false);
    res.f0_final=res.f0_init;
    double gray0_loc=0.0;
    for(int j=0;j<nl;++j){
        const double t=(double(x(j))-xmin_v)/(1.0-xmin_v);
        gray0_loc+=4.0*t*(1.0-t);
    }
    res.grayness_initial=GSum(gray0_loc)/n_global;
    auto t0=Clock::now();

    // Run all max_iter steps so fixed-p and continuation cases receive the
    // same optimization budget and expose late-iteration instability.
    for(int it=0;it<max_iter;++it){
        // ramp from 1 to simp_p over 200 iterations
        double p=continuation ? 1.0+(simp_p-1.0)*double(std::min(it,200))/200.0 : simp_p;
        x = opt.ToPhysical(eta);
        double f0=EvalF(x,p,true);
        auto fi=EvalFi(x);

        if(gcmma){
            opt.UpdateGCMMA(eta,df0,real_t(f0),fi,dg.data(),
                [&](const Vector& xc, Vector& fi_hat, real_t& f0_hat){
                    f0_hat=real_t(EvalF(xc,p,false));
                    fi_hat=EvalFi(xc);
                }, /*max_inner=*/15);
        }
        else
            opt.Update     (eta,df0,real_t(f0),fi,dg.data());

        x = opt.ToPhysical(eta);
        f0=EvalF(x,p,true);
        fi=EvalFi(x);
        double kkt=opt.KKTresidual(eta,df0,real_t(f0),fi,dg.data());
        res.kkt_final=kkt;
        res.kkt_finite = res.kkt_finite && std::isfinite(kkt);
        // Record the post-update objective once the continuation reaches its
        // final exponent; f0_init is the uniform-design value at that same p.
        bool at_final_p = (!continuation) || (p >= simp_p - 1e-9);
        if(at_final_p) res.f0_final=f0;
        res.iters=it+1;
        if(g_rank==0&&(it%50==0||it==max_iter-1)){
            double gmax=double(fi(0));
            for(auto v:fi) gmax=std::max(gmax,double(v));
            printf("  iter %4d: f0=%.4e  g_max=%+.3e  kkt=%.3e  p=%.2f\n",
                   it,f0,gmax,kkt,p);
        }
    }
    double ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();

    // Final constraint violation
    x = opt.ToPhysical(eta);
    {
        auto fi_fin=EvalFi(x);
        double viol=0;
        for(int k=0;k<m;++k) viol=std::max(viol,double(fi_fin(k)));
        res.max_viol=viol;
    }
    double gray_loc=0.0;
    for(int j=0;j<nl;++j){
        const double t=(double(x(j))-xmin_v)/(1.0-xmin_v);
        gray_loc+=4.0*t*(1.0-t);
    }
    res.grayness=GSum(gray_loc)/n_global;
    if(g_rank==0){
        double obj_drop = res.f0_final < res.f0_init ?
            100.0*(res.f0_init-res.f0_final)/std::max(res.f0_init,1e-30) : 0.0;
        printf("  Final: iters=%d  kkt=%.2e  max_viol=%.2e  gray=%.3f→%.3f"
               "  obj: %.4e→%.4e (drop=%.1f%%)  time=%.0fms (%.2fms/it)\n",
               res.iters,res.kkt_final,res.max_viol,
               res.grayness_initial,res.grayness,
               res.f0_init,res.f0_final,obj_drop,ms,ms/res.iters);
    }
    return res;
}

static void Test_FilteredSIMP(int n_global,int m,double Vfrac,
                               double p,bool cont,int r,int modes,
                               bool gcmma,int max_iter,
                               const char* label)
{
    const int effective_modes=modes>0?modes:3;
    const std::string p_label=cont
        ? std::string("1→")+std::to_string((int)p)
        : std::to_string((int)p);
    if(g_rank==0)
        printf("\n--- %-10s  n=%-7d  m=%-3d  Vfrac=%.2f  r=%-3d"
               "  modes=%-2d  p=%s  [%s] ---\n",
               label,n_global,m,Vfrac,r,effective_modes,p_label.c_str(),
               gcmma?"GCMMA":"CCSA");

    auto res=RunFilteredSIMP(n_global,m,Vfrac,p,cont,r,modes,gcmma,max_iter);

    std::string tag=std::string("[")+label+",n="+std::to_string(n_global)
                   +",r="+std::to_string(r)+","+(gcmma?"GCMMA":"CCSA")+"]";
    // A tight convergence tolerance is problem-size dependent, but the final
    // residual must remain finite and bounded; using the historical minimum
    // would allow late-iteration divergence to pass unnoticed.
    Check(res.kkt_finite && std::isfinite(res.kkt_final) && res.kkt_final < 1.0,
          (tag+" final KKT<1 and finite").c_str());
    // Objective at final p is not worse than 2× its initial value at that p.
    // A factor of 2 allows transient overshoots from large initial steps
    // while still catching genuine divergence.
    // Both values use the final requested exponent, including continuation.
    if(res.f0_init > 0)
        Check(res.f0_final < 2.0*res.f0_init,
              (tag+" objective bounded (f_final < 2*f_init_at_p)").c_str());
    // Volume constraint satisfied at the end
    Check(res.max_viol < 1e-3, (tag+" volume constraint satisfied").c_str());
    // Normalized grayness is 0 for a binary design and 1 at the midpoint.
    // At fixed mean, nontrivial redistribution should lower it from the
    // common feasible initial design's value.
    Check(std::isfinite(res.grayness) &&
          res.grayness < res.grayness_initial-1e-4,
          (tag+" design is nontrivially redistributed").c_str());
}

int main(int argc,char** argv)
{
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&g_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&g_nranks);
    for(int i=1;i<argc;++i)
        if(std::strcmp(argv[i],"--large")==0) g_large=true;

    if(g_rank==0)
        printf(
"╔══════════════════════════════════════════════════════════╗\n"
"║  Density-filtered SIMP test — CCSA  (%2d rank(s))%s║\n"
"╠══════════════════════════════════════════════════════════╣\n"
"║  x̂_j = H*x (Gaussian filter, radius r)                  ║\n"
"║  f = (1/n)Σ w_j/x̂_j^p   w_j = spatial load pattern     ║\n"
"║  Volume equality via two inequalities                    ║\n"
"║  Small r=5..10: filter couples neighbours, not global    ║\n"
"╚══════════════════════════════════════════════════════════╝\n",
g_nranks,g_large?"  [--large] ":"            ");

    // Filter radius r ≈ n/20 gives ~5% coupling per element → slow spatial modes
    // r too small → problem almost separable → converges fast
    // r too large → problem nearly uniform → converges fast  
    // r=10 (absolute, small): couples ~20 neighbours regardless of n.
    // Load modes at period n/8 >> r → filter cannot wash out the load pattern.
    // p=3: standard SIMP; binarisation pressure drives 80-200 iterations.

    // Checkerboard load at period 2r frustrates the filter at ALL scales of n.
    // P1: r=10, p=3
    if(g_rank==0)
        printf("\n═══ P1: r=10  p=3  (checkerboard load, period 2r=20) ═══\n");
    //              n       m   Vfrac  p  cont   r  modes gcmma  iters label
    Test_FilteredSIMP( 1000, 4, 0.4, 3.0,false, 10, 0, false, 500,"p3r10");
    Test_FilteredSIMP( 1000, 4, 0.4, 3.0,false, 10, 0, true,  500,"p3r10");
    Test_FilteredSIMP(10000, 4, 0.4, 3.0,false, 10, 0, false, 500,"p3r10");
    Test_FilteredSIMP(10000, 4, 0.4, 3.0,false, 10, 0, true,  500,"p3r10");
    if(g_large){
        Test_FilteredSIMP( 50000, 4,0.4,3.0,false,10,0,false,500,"p3r10");
        Test_FilteredSIMP( 50000, 6,0.4,3.0,false,10,0,true, 500,"p3r10");
        Test_FilteredSIMP(100000, 4,0.4,3.0,false,10,0,false,500,"p3r10");
        Test_FilteredSIMP(500000, 4,0.4,3.0,false,10,0,false,500,"p3r10");
        Test_FilteredSIMP(1000000,4,0.4,3.0,false,10,0,false,300,"p3r10");
    }

    // P2 is deliberately a cold-start, strongly non-convex problem.  A
    // single plain CCSA model is not globally convergent here (and is already
    // covered at p=3 in P1 and through safe continuation in P3).  This case
    // specifically verifies that GCMMA's conservatism retries stabilize the
    // p=5 problem.
    if(g_rank==0)
        printf("\n═══ P2: r=5  p=5  (cold-start GCMMA globalization) ═════\n");
    Test_FilteredSIMP( 1000, 4, 0.4, 5.0,false,  5, 0, true,  500,"p5r5");
    Test_FilteredSIMP(10000, 4, 0.4, 5.0,false,  5, 0, true,  500,"p5r5");
    if(g_large){
        Test_FilteredSIMP( 50000, 6,0.4,5.0,false, 5,0,true, 500,"p5r5");
    }

    // P3: r=10, p: 1→5 continuation
    if(g_rank==0)
        printf("\n═══ P3: r=10  p: 1→5 continuation ══════════════════════\n");
    Test_FilteredSIMP( 1000, 4, 0.4, 5.0, true, 10, 0, false, 500,"p5cont");
    Test_FilteredSIMP( 1000, 4, 0.4, 5.0, true, 10, 0, true,  500,"p5cont");
    Test_FilteredSIMP(10000, 4, 0.4, 5.0, true, 10, 0, false, 500,"p5cont");
    Test_FilteredSIMP(10000, 4, 0.4, 5.0, true, 10, 0, true,  500,"p5cont");
    if(g_large){
        Test_FilteredSIMP( 50000, 4,0.4,5.0,true,10,0,false,500,"p5cont");
        Test_FilteredSIMP(100000, 4,0.4,5.0,true,10,0,false,500,"p5cont");
        Test_FilteredSIMP(500000, 4,0.4,5.0,true,10,0,false,500,"p5cont");
    }

    if(g_rank==0){
        printf("\n╔══════════════════════════════════════════════════════════╗\n");
        if(g_nfail==0)
            printf("║  All filtered SIMP tests PASSED.                         ║\n");
        else
            printf("║  %d test(s) FAILED.%-38s║\n",g_nfail,"");
        printf("╚══════════════════════════════════════════════════════════╝\n");
    }
    MPI_Finalize();
    return g_nfail>0?1:0;
}
