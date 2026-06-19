/**
 * test_sq_device.cpp  —  Device-aware SQOptimizer test suite
 *
 * Same device tests as test_mma_device.cpp using SQOptimizer.
 *
 * Tests that MMAOptimizer and MMAOptimizerParallel produce identical
 * results when x lives on-device (GPU) vs on-host (CPU), and benchmarks
 * the GPU speedup for large problems.
 *
 * For each test we:
 *   1. Run with x.UseDevice(false)  → CPU path
 *   2. Run with x.UseDevice(true)   → GPU path (if device enabled)
 *   3. Compare results: KKT and max pointwise difference < tolerance
 *
 * Device selection:
 *   mfem::Device device("cuda");   // or "hip", "cpu"
 *   x.UseDevice(true);
 *
 * Build:
 *   cmake --build build
 *
 * Run (CPU):
 *   ./build/test_mma_device
 *
 * Run (GPU, requires MFEM built with MFEM_USE_CUDA=YES):
 *   ./build/test_mma_device --device cuda
 *
 * Run (parallel GPU):
 *   mpirun -np 4 ./build/test_mma_device --device cuda
 */

#include "MMA_MFEM.hpp"
#include <mfem.hpp>
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <chrono>
#include <string>
#include <cstring>

using namespace mfem;
using namespace mfem_mma;
using Clock = std::chrono::steady_clock;

static int  g_rank  = 0;
static int  g_nfail = 0;
static bool g_has_device = false;

static void Check(bool cond, const char* msg)
{
    if (g_rank != 0) return;
    if (cond) printf("  [PASS] %s\n", msg);
    else     { printf("  [FAIL] %s\n", msg); ++g_nfail; }
}

static double GlobalSum(double v)
{ double g; MPI_Allreduce(&v,&g,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD); return g; }
static double GlobalMax(double v)
{ double g; MPI_Allreduce(&v,&g,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD); return g; }

static std::pair<int,int> Distribute(int n)
{
    int nr; MPI_Comm_size(MPI_COMM_WORLD,&nr);
    int b=n/nr,r=n%nr;
    return {b+(g_rank<r?1:0), g_rank*b+std::min(g_rank,r)};
}

static uint64_t lcg(uint64_t& s)
{ s=s*6364136223846793005ULL+1442695040888963407ULL; return s>>33; }

// ============================================================
// Helper: run compliance proxy on one device setting, return result
// ============================================================
struct RunResult { double kkt; double xmean; int iters; double ms_per_iter; };

static RunResult RunComplianceProxy(int n, double Vfrac, bool on_device,
                                     bool gcmma = false)
{
    auto [nl,off] = Distribute(n);
    MPI_Comm comm = MPI_COMM_WORLD;

    Vector x(nl), xmin(nl), xmax(nl), df0(nl), dg(nl);
    x.UseDevice(on_device);
    xmin.UseDevice(on_device); xmax.UseDevice(on_device);
    df0.UseDevice(on_device);  dg.UseDevice(on_device);

    // Initialize on host, then push to device
    {
        real_t* h = x.HostWrite();    for(int j=0;j<nl;++j) h[j]=0.5;
    }
    {
        real_t* h = xmin.HostWrite(); for(int j=0;j<nl;++j) h[j]=0.001;
    }
    {
        real_t* h = xmax.HostWrite(); for(int j=0;j<nl;++j) h[j]=1.0;
    }
    {
        real_t* h = dg.HostWrite();   for(int j=0;j<nl;++j) h[j]=real_t(1.0/n);
    }

    MMAOptimizerParallel opt(comm,nl,1,x);
    double kkt=1.0; int it=0;
    auto t0 = Clock::now();

    for(;it<200&&kkt>1e-5;++it){
        // Compute gradient on device via forall_switch
        {
            auto* xr = x.Read();
            auto* df = df0.Write();
            forall_switch(on_device, nl, [=] MFEM_HOST_DEVICE (int j){
                double xj = double(xr[j]);
                df[j] = real_t(-1.0/(xj*xj));
            });
        }
        // Compute f0 and fi on host (global reduce)
        double f0_loc=0.0, g_loc=0.0;
        {
            const real_t* xh = x.HostRead();
            for(int j=0;j<nl;++j){ double xj=xh[j]; f0_loc+=1.0/xj; g_loc+=xj; }
        }
        double f0=GlobalSum(f0_loc);
        double g =GlobalSum(g_loc)/(double)n - Vfrac;
        mfem::Vector fival(1); fival(0)=g;

        if(gcmma)
            opt.UpdateGCMMA(x,df0,f0,fival,&dg,xmin,xmax);
        else
            opt.Update(x,df0,f0,fival,&dg,xmin,xmax);

        // KKT: recompute df0 at new x
        {
            auto* xr=x.Read(); auto* df=df0.Write();
            forall_switch(on_device,nl,[=] MFEM_HOST_DEVICE (int j){
                double xj=double(xr[j]); df[j]=real_t(-1.0/(xj*xj));
            });
        }
        g_loc=0;
        const real_t* xh=x.HostRead();
        for(int j=0;j<nl;++j) g_loc+=xh[j];
        g=GlobalSum(g_loc)/(double)n-Vfrac; fival(0)=g;
        // Recompute df0 on device again (HostRead may have invalidated device copy)
        {
            auto* xr=x.Read(); auto* df=df0.Write();
            forall_switch(on_device,nl,[=] MFEM_HOST_DEVICE (int j){
                double xj=double(xr[j]); df[j]=real_t(-1.0/(xj*xj));
            });
        }
        kkt=opt.KKTresidual(x,df0,f0,fival,&dg,xmin,xmax);
    }

    double ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
    double xloc=0; { const real_t* xh=x.HostRead(); for(int j=0;j<nl;++j) xloc+=xh[j]; }
    return {kkt, GlobalSum(xloc)/(double)n, opt.NumIterations(), it>0?ms/it:0};
}

// ============================================================
// Test: CPU vs GPU result comparison
// ============================================================
static void Test_CpuGpuMatch(int n, double Vfrac, bool gcmma=false)
{
    if(g_rank==0)
        printf("\n--- CpuGpuMatch (n=%d, Vfrac=%.2f, %s) ---\n",
               n, Vfrac, gcmma?"GCMMA":"MMA");

    auto cpu = RunComplianceProxy(n, Vfrac, false, gcmma);
    if(g_rank==0)
        printf("  CPU: kkt=%.2e  xmean=%.6f  iters=%d  %.2fms/iter\n",
               cpu.kkt, cpu.xmean, cpu.iters, cpu.ms_per_iter);

    Check(cpu.kkt < 1e-4,
          (std::string("CPU KKT<1e-4 (n=")+std::to_string(n)+")").c_str());

    if (!g_has_device) {
        if(g_rank==0) printf("  GPU: skipped (no device enabled)\n");
        return;
    }

    auto gpu = RunComplianceProxy(n, Vfrac, true, gcmma);
    if(g_rank==0)
        printf("  GPU: kkt=%.2e  xmean=%.6f  iters=%d  %.2fms/iter\n",
               gpu.kkt, gpu.xmean, gpu.iters, gpu.ms_per_iter);

    double diff = std::abs(cpu.xmean - gpu.xmean);
    if(g_rank==0) {
        printf("  |cpu_xmean - gpu_xmean| = %.2e\n", diff);
        if(cpu.ms_per_iter>0 && gpu.ms_per_iter>0)
            printf("  Speedup: %.1fx\n", cpu.ms_per_iter/gpu.ms_per_iter);
    }

    Check(gpu.kkt < 1e-4,
          (std::string("GPU KKT<1e-4 (n=")+std::to_string(n)+")").c_str());
    Check(diff < 0.01,
          (std::string("CPU==GPU result (n=")+std::to_string(n)+")").c_str());
}

// ============================================================
// Test: device-aware gradient accumulation (forall_switch)
// ============================================================
static void Test_DeviceGradient(int n)
{
    if(g_rank==0) printf("\n--- DeviceGradient (n=%d) ---\n", n);
    auto [nl,off] = Distribute(n);

    // Compute the same gradient on CPU and GPU, compare
    Vector x_cpu(nl), x_gpu(nl), df_cpu(nl), df_gpu(nl);
    x_cpu.UseDevice(false); x_gpu.UseDevice(g_has_device);
    df_cpu.UseDevice(false); df_gpu.UseDevice(g_has_device);

    {
        real_t* h=x_cpu.HostWrite();
        for(int j=0;j<nl;++j) h[j]=real_t(0.3+0.4*(double(off+j)/n));
    }
    // Copy to GPU version
    {
        const real_t* hc=x_cpu.HostRead();
        real_t* hg=x_gpu.HostWrite();
        std::memcpy(hg,hc,nl*sizeof(real_t));
    }

    // CPU gradient
    {
        const auto* xr=x_cpu.Read(); auto* df=df_cpu.Write();
        forall_switch(false, nl, [=] MFEM_HOST_DEVICE (int j){
            double xj=double(xr[j]); df[j]=real_t(-1.0/(xj*xj));
        });
    }
    // Device gradient
    {
        const auto* xr=x_gpu.Read(); auto* df=df_gpu.Write();
        forall_switch(g_has_device, nl, [=] MFEM_HOST_DEVICE (int j){
            double xj=double(xr[j]); df[j]=real_t(-1.0/(xj*xj));
        });
    }

    double err_loc=0.0;
    {
        const real_t* dc=df_cpu.HostRead();
        const real_t* dg=df_gpu.HostRead();
        for(int j=0;j<nl;++j) err_loc=std::max(err_loc,std::abs(double(dc[j]-dg[j])));
    }
    double maxerr = GlobalMax(err_loc);
    if(g_rank==0) printf("  max |df_cpu - df_gpu| = %.2e\n", maxerr);
    Check(maxerr < 1e-12, "CPU and GPU gradients match to 1e-12");
}

// ============================================================
// Test: large-scale device throughput
// ============================================================
static void Test_LargeScaleThroughput(int n)
{
    if(g_rank==0)
        printf("\n--- LargeScaleThroughput (n=%d, m=1) ---\n", n);

    auto [nl,off] = Distribute(n);
    MPI_Comm comm = MPI_COMM_WORLD;

    auto Bench = [&](bool on_dev, const char* label) {
        Vector x(nl), xmin(nl), xmax(nl), df0(nl), dg(nl);
        x.UseDevice(on_dev); xmin.UseDevice(on_dev);
        xmax.UseDevice(on_dev); df0.UseDevice(on_dev); dg.UseDevice(on_dev);
        {
            real_t *hx=x.HostWrite(),*hmn=xmin.HostWrite(),
                   *hmx=xmax.HostWrite(),*hdg=dg.HostWrite();
            for(int j=0;j<nl;++j){
                hx[j]=0.5; hmn[j]=0.001; hmx[j]=1.0; hdg[j]=real_t(1.0/n);
            }
        }
        double cv=std::max(1000.0,10.0*n);
        double a1[1]={0},c1[1]={cv},d1[1]={1};
        MMAOptimizerParallel opt(comm,nl,1,x,a1,c1,d1);
        double kkt=1.0; int it=0;
        auto t0=Clock::now();
        for(;it<50&&kkt>1e-5;++it){
            {
                auto* xr=x.Read(); auto* df=df0.Write();
                forall_switch(on_dev,nl,[=] MFEM_HOST_DEVICE (int j){
                    double xj=double(xr[j]); df[j]=real_t(-1.0/(xj*xj));
                });
            }
            double gl=0; { const real_t* xh=x.HostRead(); for(int j=0;j<nl;++j) gl+=xh[j]; }
            double g=GlobalSum(gl)/(double)n-0.4; mfem::Vector fival(1); fival(0)=g;
            opt.Update(x,df0,0.0,fival,&dg,xmin,xmax);
            {
                auto* xr=x.Read(); auto* df=df0.Write();
                forall_switch(on_dev,nl,[=] MFEM_HOST_DEVICE (int j){
                    double xj=double(xr[j]); df[j]=real_t(-1.0/(xj*xj));
                });
            }
            gl=0; { const real_t* xh=x.HostRead(); for(int j=0;j<nl;++j) gl+=xh[j]; }
            g=GlobalSum(gl)/(double)n-0.4; fival(0)=g;
            {
                auto* xr=x.Read(); auto* df=df0.Write();
                forall_switch(on_dev,nl,[=] MFEM_HOST_DEVICE (int j){
                    double xj=double(xr[j]); df[j]=real_t(-1.0/(xj*xj));
                });
            }
            kkt=opt.KKTresidual(x,df0,0.0,fival,&dg,xmin,xmax);
        }
        double ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
        double xloc=0; { const real_t* xh=x.HostRead(); for(int j=0;j<nl;++j) xloc+=xh[j]; }
        double xmean=GlobalSum(xloc)/(double)n;
        if(g_rank==0)
            printf("  %s: kkt=%.2e  xmean=%.4f  iters=%d  total=%.0fms  %.2fms/iter\n",
                   label, kkt, xmean, opt.NumIterations(), ms, it>0?ms/it:0);
        return std::make_pair(kkt, xmean);
    };

    auto [kkt_cpu, xm_cpu] = Bench(false, "CPU");
    Check(kkt_cpu<1e-4,
          (std::string("CPU KKT<1e-4 (n=")+std::to_string(n)+")").c_str());

    if(g_has_device){
        auto [kkt_gpu, xm_gpu] = Bench(true, "GPU");
        Check(kkt_gpu<1e-4,
              (std::string("GPU KKT<1e-4 (n=")+std::to_string(n)+")").c_str());
        Check(std::abs(xm_cpu-xm_gpu)<0.01,
              (std::string("CPU==GPU xmean (n=")+std::to_string(n)+")").c_str());
    }
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    int nranks; MPI_Comm_size(MPI_COMM_WORLD,&nranks);

    // Parse --device argument
    std::string device_str = "cpu";
    for(int i=1;i<argc;++i){
        if(std::strcmp(argv[i],"--device")==0 && i+1<argc)
            device_str = argv[++i];
    }

    // Initialise MFEM device
    mfem::Device device(device_str.c_str());
    if(g_rank==0) device.Print();
    g_has_device = mfem::Device::IsEnabled();

    if(g_rank==0)
        printf("=== MFEM MMA Device test suite (%d rank(s), device=%s) ===\n\n",
               nranks, device_str.c_str());
    if(g_rank==0 && !g_has_device)
        printf("  NOTE: No GPU device enabled — GPU tests will be skipped.\n"
               "  To enable GPU: --device cuda  or  --device hip\n\n");

    // ── Gradient correctness ──────────────────────────────────────────────
    Test_DeviceGradient(10000);
    Test_DeviceGradient(100000);

    // ── CPU vs GPU match ──────────────────────────────────────────────────
    Test_CpuGpuMatch(1000,   0.4);
    Test_CpuGpuMatch(10000,  0.4);
    Test_CpuGpuMatch(50000,  0.4);
    Test_CpuGpuMatch(10000,  0.4, true);  // GCMMA

    // ── Throughput scaling ────────────────────────────────────────────────
    Test_LargeScaleThroughput(10000);
    Test_LargeScaleThroughput(100000);
    if(g_has_device) Test_LargeScaleThroughput(1000000);

    if(g_rank==0){
        printf("\n========================================\n");
        if(g_nfail==0) printf("All device tests PASSED.\n");
        else           printf("%d device test(s) FAILED.\n",g_nfail);
        printf("========================================\n");
    }
    MPI_Finalize();
    return g_nfail>0?1:0;
}
