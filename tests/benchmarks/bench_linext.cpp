// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "bench.hpp"

#ifdef MFEM_USE_BENCHMARK

#include "tests/unit/fem/test_linearform_ext.hpp"

namespace mfem
{

namespace linearform_ext_tests
{

void LinearFormExtTest::Run() { MFEM_ABORT("Should use the virtuals!"); }
void LinearFormExtTest::Description() { MFEM_ABORT("Should use the virtuals!"); }

////////////////////////////////////////////////////////////////////////////////
/// TEST for LinearFormExtension
struct Test: public LinearFormExtTest
{
   Test(int N, int dim, int vdim, int ordering, bool gll, int problem, int order):
      LinearFormExtTest(N, dim, vdim, ordering, gll,
                        problem, order,
                        true) { }

   void Description() override { /* */ }

   void Run() override
   {
      AssembleBoth();
      mdofs += MDofs();
      MFEM_DEVICE_SYNC;
      const double tolerance = 1e-12;
      const double dtd = lf_full*lf_full;
      const double rtr = lf_legacy*lf_legacy;
      const bool almost_eq = almost_equal(dtd, rtr, tolerance);
      MFEM_VERIFY(almost_eq, "almost_equal test error: " << dtd << " " << rtr);
   }
};

////////////////////////////////////////////////////////////////////////////////
constexpr int GL = false; // Gauss-Legendre, q=p+2
constexpr int GLL = true; // Gauss-Legendre-Lobatto, q=p+1
constexpr int VDIM = 24;

/// Scalar Linear Form Extension Tests
#define LinExtTest(Problem,dim,vdim,gll)\
static void TEST_##Problem##dim##gll(bm::State &state){\
   const int p = state.range(0);\
   Test ker(4, dim,vdim,Ordering::byVDIM,gll,LinearFormExtTest::Problem,p);\
   while(state.KeepRunning()) { ker.Run(); }\
   state.counters["MDof/s"] = bm::Counter(ker.SumMdofs(),bm::Counter::kIsRate);}\
BENCHMARK(TEST_##Problem##dim##gll)->DenseRange(1,6)->Unit(bm::kMillisecond);

/// Scalar linear form tests & Gauss-Legendre-Lobatto, q=p+1
LinExtTest(DomainLF,2,1,GLL)
LinExtTest(DomainLF,3,1,GLL)

/// Vector linear form tests & Gauss-Legendre-Lobatto, q=p+1
LinExtTest(VectorDomainLF,2,VDIM,GLL)
LinExtTest(VectorDomainLF,3,VDIM,GLL)

/// Grad linear form tests & Gauss-Legendre-Lobatto, q=p+1
LinExtTest(DomainLFGrad,2,1,GLL)
LinExtTest(DomainLFGrad,3,1,GLL)

/// Vector Grad linear form tests & Gauss-Legendre-Lobatto, q=p+1
LinExtTest(VectorDomainLFGrad,2,VDIM,GLL)
LinExtTest(VectorDomainLFGrad,3,VDIM,GLL)

/// Scalar linear form tests & Gauss-Legendre, q=p+2
LinExtTest(DomainLF,2,1,GL)
LinExtTest(DomainLF,3,1,GL)

/// Vector linear form tests & Gauss-Legendre, q=p+2
LinExtTest(VectorDomainLF,2,VDIM,GL)
LinExtTest(VectorDomainLF,3,VDIM,GL)

/// Grad linear form tests & Gauss-Legendre, q=p+2
LinExtTest(DomainLFGrad,2,1,GL)
LinExtTest(DomainLFGrad,3,1,GL)

/// Vector Grad linear form tests & Gauss-Legendre, q=p+2
LinExtTest(VectorDomainLFGrad,2,VDIM,GL)
LinExtTest(VectorDomainLFGrad,3,VDIM,GL)


////////////////////////////////////////////////////////////////////////////////
/// BENCH for LinearFormExtension
template<enum AssemblyLevel LAL>
struct Bench: public LinearFormExtTest
{
   Bench(int dim, int vdim, int ordering, bool gll, int problem, int p):
      LinearFormExtTest(Device::IsEnabled()?24:4,
                        dim, vdim, ordering, gll,
                        problem, p,
                        false)
   { }

   void Description() override { /* */ }

   void Run() override
   {
      mdofs += MDofs();
      MFEM_DEVICE_SYNC;
      if (LAL==AssemblyLevel::FULL) { lf_full.Assemble(); }
      if (LAL==AssemblyLevel::LEGACY) { lf_legacy.Assemble(); }
   }
};

/// Linear Form Extension Scalar Benchs
#define LinExtBench(Problem,lal,dim,vdim,gll)\
static void BENCH_##lal##_##Problem##dim##gll(bm::State &state){\
   const int p = state.range(0);\
   Bench<AssemblyLevel::lal> ker(dim,vdim,Ordering::byVDIM,gll,LinearFormExtTest::Problem, p);\
   while (state.KeepRunning()) { ker.Run(); }\
   state.counters["MDof/s"] = bm::Counter(ker.SumMdofs(), bm::Counter::kIsRate);}\
BENCHMARK(BENCH_##lal##_##Problem##dim##gll)->DenseRange(1,6)->Unit(bm::kMicrosecond);

/// Scalar linear form bench & Gauss-Legendre-Lobatto, q=p+1
LinExtBench(DomainLF,LEGACY,2,1,GLL)
LinExtBench(DomainLF,  FULL,2,1,GLL)
LinExtBench(DomainLF,LEGACY,3,1,GLL)
LinExtBench(DomainLF,  FULL,3,1,GLL)

/// Vector linear form bench & Gauss-Legendre-Lobatto, q=p+1
LinExtBench(VectorDomainLF,LEGACY,2,VDIM,GLL)
LinExtBench(VectorDomainLF,  FULL,2,VDIM,GLL)
LinExtBench(VectorDomainLF,LEGACY,3,VDIM,GLL)
LinExtBench(VectorDomainLF,  FULL,3,VDIM,GLL)

/// Grad Scalar linear form bench & Gauss-Legendre-Lobatto, q=p+1
LinExtBench(DomainLFGrad,LEGACY,2,1,GLL)
LinExtBench(DomainLFGrad,  FULL,2,1,GLL)
LinExtBench(DomainLFGrad,LEGACY,3,1,GLL)
LinExtBench(DomainLFGrad,  FULL,3,1,GLL)

/// Vector Grad linear form bench & Gauss-Legendre-Lobatto, q=p+1
LinExtBench(VectorDomainLFGrad,LEGACY,2,VDIM,GLL)
LinExtBench(VectorDomainLFGrad,  FULL,2,VDIM,GLL)
LinExtBench(VectorDomainLFGrad,LEGACY,3,VDIM,GLL)
LinExtBench(VectorDomainLFGrad,  FULL,3,VDIM,GLL)

/// Scalar linear form bench & Gauss-Legendre, q=p+2
LinExtBench(DomainLF,LEGACY,2,1,GL)
LinExtBench(DomainLF,  FULL,2,1,GL)
LinExtBench(DomainLF,LEGACY,3,1,GL)
LinExtBench(DomainLF,  FULL,3,1,GL)

/// Vector linear form bench & Gauss-Legendre, q=p+2
LinExtBench(VectorDomainLF,LEGACY,2,VDIM,GL)
LinExtBench(VectorDomainLF,  FULL,2,VDIM,GL)
LinExtBench(VectorDomainLF,LEGACY,3,VDIM,GL)
LinExtBench(VectorDomainLF,  FULL,3,VDIM,GL)

/// Grad Scalar linear form bench & Gauss-Legendre, q=p+2
LinExtBench(DomainLFGrad,LEGACY,2,1,GL)
LinExtBench(DomainLFGrad,  FULL,2,1,GL)
LinExtBench(DomainLFGrad,LEGACY,3,1,GL)
LinExtBench(DomainLFGrad,  FULL,3,1,GL)

/// Vector Grad linear form bench & Gauss-Legendre, q=p+2
LinExtBench(VectorDomainLFGrad,LEGACY,2,VDIM,GL)
LinExtBench(VectorDomainLFGrad,  FULL,2,VDIM,GL)
LinExtBench(VectorDomainLFGrad,LEGACY,3,VDIM,GL)
LinExtBench(VectorDomainLFGrad,  FULL,3,VDIM,GL)

} // namespace linearform_ext_tests

} // namespace mfem

/** ****************************************************************************
 * @brief main entry point, some options are for example:
 * --benchmark_filter=TEST --benchmark_min_time=0.01
 * --benchmark_filter=BENCH_FULL --benchmark_min_time=0.1
 * --benchmark_context=device=cuda
 **************************************************************************** */
int main(int argc, char *argv[])
{
   bm::ConsoleReporter CR;
   bm::Initialize(&argc, argv);

   // Device setup, CPU by default
   std::string device_config = "cpu";
   if (bmi::global_context != nullptr)
   {
      const auto device = bmi::global_context->find("device");
      if (device != bmi::global_context->end())
      {
         mfem::out << device->first << " : " << device->second << std::endl;
         device_config = device->second;
      }
   }
   Device device(device_config.c_str());
   device.Print();

   if (bm::ReportUnrecognizedArguments(argc, argv)) { return 1; }
   bm::RunSpecifiedBenchmarks(&CR);
   return 0;
}

#else // MFEM_USE_BENCHMARK

int main(int, char *[]) { return 0; }

#endif // MFEM_USE_BENCHMARK
