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
/// Base class for the LinearForm extension test and the bench
struct LinExt
{
   const int N, p, q, dim;
   Mesh mesh;
   H1_FECollection fec;
   FiniteElementSpace fes;
   const Geometry::Type geom_type;
   const IntegrationRule *ir;
   ConstantCoefficient one;
   LinearForm b;
   const int dofs;
   double mdofs;

   LinExt(int p, int dim, int vdim):
      N(Device::IsEnabled()?32:8),
      p(p),
      q(2*p + 3),
      dim(dim),
      mesh(dim==3 ? Mesh::MakeCartesian3D(N,N,N,Element::HEXAHEDRON):
           (assert(dim==2), Mesh::MakeCartesian2D(N,N,Element::QUADRILATERAL))),
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
template<typename LFI, int DIM, int VDIM>
struct Test: public LinExt
{
   LinearForm c;
   ConstantCoefficient f;
   Test(int order, const double pi = M_PI):
      LinExt(order,DIM,VDIM),
      c(&fes),
      f(pi)
   {
      b.SetAssemblyLevel(LinearAssemblyLevel::FULL);
      c.SetAssemblyLevel(LinearAssemblyLevel::LEGACY);
      b.AddDomainIntegrator(new LFI(f));
      c.AddDomainIntegrator(new LFI(f));
      MFEM_DEVICE_SYNC;
   }

   void benchmark() override
   {
      b.Assemble();
      c.Assemble();
      const double btb = b*b;
      const double ctc = c*c;
      MFEM_VERIFY(almost_equal(btb,ctc,10), "almost_equal test error!");
      MFEM_DEVICE_SYNC;
      mdofs += MDofs();
   }
};

/// Scalar Linear Form Extension Tests
#define LinExtTest(Kernel,DIM)\
static void TEST_##Kernel##_##DIM##D(bm::State &state){\
   Test<Kernel##Integrator,DIM,1> ker(state.range(0));\
   while (state.KeepRunning()) { ker.benchmark(); }\
   state.counters["MDof/s"] = bm::Counter(ker.SumMdofs(), bm::Counter::kIsRate);}\
BENCHMARK(TEST_##Kernel##_##DIM##D)->DenseRange(1,6)->Unit(bm::kMillisecond);

/// Scalar linear form tests, VDIM = DIM
LinExtTest(DomainLF,2)
LinExtTest(DomainLF,3)

////////////////////////////////////////////////////////////////////////////////
/// VectorTEST for LinearFormExtension
template<typename LFI, int DIM, int VDIM>
struct VectorTest: public LinExt
{
   Vector v;
   LinearForm c;
   VectorConstantCoefficient f;
   VectorTest(int order):
      LinExt(order, DIM, VDIM),
      v(VDIM),
      c(&fes),
      f((v.Randomize(),v))
   {
      b.SetAssemblyLevel(LinearAssemblyLevel::FULL);
      c.SetAssemblyLevel(LinearAssemblyLevel::LEGACY);
      b.AddDomainIntegrator(new LFI(f));
      c.AddDomainIntegrator(new LFI(f));
      MFEM_DEVICE_SYNC;
   }

   void benchmark() override
   {
      b.Assemble();
      c.Assemble();
      const double btb = b*b;
      const double ctc = c*c;
      MFEM_VERIFY(almost_equal(btb,ctc,10), "almost_equal test error!");
      MFEM_DEVICE_SYNC;
      mdofs += MDofs();
   }
};

/// Vector Linear Form Extension Tests
#define VectorLinExtTest(Kernel,DIM,VDIM)\
static void TEST_##Kernel##_##DIM##D(bm::State &state){\
   VectorTest<Kernel##Integrator,DIM,VDIM> ker(state.range(0));\
   while (state.KeepRunning()) { ker.benchmark(); }\
   state.counters["MDof/s"] = bm::Counter(ker.SumMdofs(), bm::Counter::kIsRate);}\
BENCHMARK(TEST_##Kernel##_##DIM##D)->DenseRange(1,6)->Unit(bm::kMillisecond);

/// Vector linear form tests, VDIM = DIM
VectorLinExtTest(VectorDomainLF,2,2)
VectorLinExtTest(VectorDomainLF,3,3)

////////////////////////////////////////////////////////////////////////////////
/// BENCH for LinearFormExtension
template<int DIM, int VDIM, enum LinearAssemblyLevel LINEAR_ASSEMBLY_LEVEL>
struct Bench: public LinExt
{
   Bench(int order): LinExt(order, DIM, VDIM)
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

/// Linear Form Extension Scalar Benchs
#define LinExtBench(LVL,DIM)\
static void BENCH_##LVL##_DomainLF_##DIM##D(bm::State &state){\
   Bench<DIM,1,LinearAssemblyLevel::LVL> ker(state.range(0));\
   while (state.KeepRunning()) { ker.benchmark(); }\
   state.counters["MDof/s"] = bm::Counter(ker.SumMdofs(), bm::Counter::kIsRate);}\
BENCHMARK(BENCH_##LVL##_DomainLF_##DIM##D)->DenseRange(1,6)->Unit(bm::kMicrosecond);

/// 2D scalar linear form bench
LinExtBench(LEGACY,2)
LinExtBench(FULL,2)

/// 3D scalar linear form bench
LinExtBench(LEGACY,3)
LinExtBench(FULL,3)

////////////////////////////////////////////////////////////////////////////////
/// Vector BENCH for LinearFormExtension
template<int DIM, int VDIM, enum LinearAssemblyLevel LINEAR_ASSEMBLY_LEVEL>
struct VectorBench: public LinExt
{
   Vector v;
   VectorConstantCoefficient f;
   VectorBench(int order): LinExt(order, DIM, VDIM),
      v(VDIM),
      f((v.Randomize(),v))
   {
      b.SetAssemblyLevel(LINEAR_ASSEMBLY_LEVEL);
      b.AddDomainIntegrator(new VectorDomainLFIntegrator(f));
      MFEM_DEVICE_SYNC;
   }

   void benchmark() override
   {
      b.Assemble();
      MFEM_DEVICE_SYNC;
      mdofs += MDofs();
   }
};

/// Linear Form Extension Vector Benchs
#define VectorLinExtBench(LVL,DIM)\
static void BENCH_##LVL##_VectorDomainLF##DIM##D(bm::State &state){\
   Bench<DIM,1,LinearAssemblyLevel::LVL> ker(state.range(0));\
   while (state.KeepRunning()) { ker.benchmark(); }\
   state.counters["MDof/s"] = bm::Counter(ker.SumMdofs(), bm::Counter::kIsRate);}\
BENCHMARK(BENCH_##LVL##_VectorDomainLF##DIM##D)->DenseRange(1,6)->Unit(bm::kMicrosecond);

/// 2D vector linear form bench
VectorLinExtBench(LEGACY,2)
VectorLinExtBench(FULL,2)

/// 3D vector linear form bench
VectorLinExtBench(LEGACY,3)
VectorLinExtBench(FULL,3)


/** ****************************************************************************
 * @brief main entry point
 * --benchmark_filter=TEST
 * --benchmark_filter=BENCH
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
