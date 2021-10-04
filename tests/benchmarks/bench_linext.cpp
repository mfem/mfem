// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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

#define MFEM_DEBUG_COLOR 206
#include "../../general/debug.hpp"

#ifdef MFEM_USE_BENCHMARK

// double f(const double x) { return std::sin(M_PI*x*0.125); }

////////////////////////////////////////////////////////////////////////////////
/// Base class for the test and the bench for the LinearForm extension
struct LinExt
{
   const int N, p, q, dim = 3;
   Mesh mesh;
   H1_FECollection fec;
   FiniteElementSpace fes;
   const Geometry::Type geom_type;
   const IntegrationRule *ir;
   ConstantCoefficient one;
   LinearForm b;
   const int dofs;
   double mdofs;

   LinExt(int p, int vdim):
      N(Device::IsEnabled()?32:8),
      p(p),
      q(2*p + 3),
      mesh(Mesh::MakeCartesian3D(N,N,N,Element::HEXAHEDRON)),
      fec(p, dim, BasisType::GaussLobatto),
      fes(&mesh, &fec, vdim),
      geom_type(fes.GetFE(0)->GetGeomType()),
      ir(&IntRules.Get(geom_type, q)),
      one(1.0),
      b(&fes),
      dofs(fes.GetTrueVSize()),
      mdofs(0.0) { }

   virtual void benchmark() = 0;

   double SumMdofs() const { return mdofs; }

   double MDofs() const { return 1e-6 * dofs; }
};

////////////////////////////////////////////////////////////////////////////////
/// TEST for LinearFormExtension
template<int VDIM>
struct Test: public LinExt
{
   LinearForm c;
   ConstantCoefficient f;
   Test(int order, const double pi = M_PI): LinExt(order, VDIM), c(&fes), f(pi)
   {
      b.SetAssemblyLevel(LinearAssemblyLevel::FULL);
      c.SetAssemblyLevel(LinearAssemblyLevel::LEGACY);
      b.AddDomainIntegrator(new DomainLFIntegrator(f));
      c.AddDomainIntegrator(new DomainLFIntegrator(f));
      MFEM_DEVICE_SYNC;
   }

   void benchmark() override
   {
      b.Assemble();
      c.Assemble();
      const double btb = b*b;
      const double ctc = c*c;
      MFEM_VERIFY(almost_equal(btb, ctc), "almost_equal test error!");
      MFEM_DEVICE_SYNC;
      mdofs += MDofs();
   }
};

/// Linear Form Extension Tests
#define LinExtTest(VDIM)\
static void TEST_##VDIM##D(bm::State &state){\
   Test<VDIM> ker(state.range(0));\
   while (state.KeepRunning()) { ker.benchmark(); }\
   state.counters["MDof/s"] = bm::Counter(ker.SumMdofs(), bm::Counter::kIsRate);}\
BENCHMARK(TEST_##VDIM##D)->DenseRange(1,6)->Unit(bm::kMillisecond);

/// 1D scalar linear form tests
LinExtTest(1)

////////////////////////////////////////////////////////////////////////////////
/// BENCH for LinearFormExtension
template<int VDIM, enum LinearAssemblyLevel LINEAR_ASSEMBLY_LEVEL>
struct Bench: public LinExt
{
   Bench(int order): LinExt(order, VDIM)
   {
      b.SetAssemblyLevel(LINEAR_ASSEMBLY_LEVEL);
      b.AddDomainIntegrator(new DomainLFIntegrator(one));
      MFEM_DEVICE_SYNC;
   }

   void benchmark() override
   {
      b.Assemble();
      MFEM_DEVICE_SYNC;
      mdofs += MDofs();
   }
};

/// Linear Form Extension Benchs
#define LinExtBench(VDIM,LVL)\
static void BENCH_##LVL##_##VDIM##D(bm::State &state){\
   Bench<VDIM,LinearAssemblyLevel::LVL> ker(state.range(0));\
   while (state.KeepRunning()) { ker.benchmark(); }\
   state.counters["MDof/s"] = bm::Counter(ker.SumMdofs(), bm::Counter::kIsRate);}\
BENCHMARK(BENCH_##LVL##_##VDIM##D)->DenseRange(1,6)->Unit(bm::kMillisecond);

/// 1D scalar linear form bench
LinExtBench(1,FULL)
LinExtBench(1,LEGACY)

/// 2D vector linear form bench
//LinExtBench(2,FULL)
//LinExtBench(2,LEGACY)

/// 3D vector linear form bench
//LinExtBench(3,FULL)
//LinExtBench(3,LEGACY)


/** ****************************************************************************
 * @brief main entry point
 * --benchmark_filter=TB1
 * --benchmark_context=device=cuda
 */
int main(int argc, char *argv[])
{
   bm::ConsoleReporter CR;
   bm::Initialize(&argc, argv);

   // Device setup, cpu by default
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

#endif // MFEM_USE_BENCHMARK
