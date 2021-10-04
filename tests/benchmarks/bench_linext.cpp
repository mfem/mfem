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
      mesh(dim ==3 ? Mesh::MakeCartesian3D(N,N,N,Element::HEXAHEDRON):
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
template<int DIM, typename LFI>
struct Test: public LinExt
{
   Vector v;
   LinearForm c;
   ConstantCoefficient f;
   Test(int order, const double pi = M_PI): LinExt(order,DIM,1), c(&fes), f(pi)
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
      //dbg("%.21e %.21e", btb,ctc);
      MFEM_VERIFY(almost_equal(btb,ctc,10), "almost_equal test error!");
      MFEM_DEVICE_SYNC;
      mdofs += MDofs();
   }
};

/// Linear Form Extension Tests
#define LinExtTest(DIM,Kernel)\
static void TEST_##Kernel(bm::State &state){\
   Test<DIM,Kernel##Integrator> ker(state.range(0));\
   while (state.KeepRunning()) { ker.benchmark(); }\
   state.counters["MDof/s"] = bm::Counter(ker.SumMdofs(), bm::Counter::kIsRate);}\
BENCHMARK(TEST_##Kernel)->DenseRange(1,6)->Unit(bm::kMillisecond);

/// 1D scalar linear form tests
LinExtTest(3,DomainLF)

////////////////////////////////////////////////////////////////////////////////
/// VectorTEST for LinearFormExtension
template<typename LFI, int DIM, int VDIM>
struct VectorTest: public LinExt
{
   Vector v;
   LinearForm c;
   VectorConstantCoefficient f;
   VectorTest(int order, const double pi = M_PI):
      LinExt(order, DIM, VDIM),
      v(VDIM),
      c(&fes),
      f((v=pi,v))
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
      //dbg("%.21e %.21e", btb,ctc);
      MFEM_VERIFY(almost_equal(btb,ctc,10), "almost_equal test error!");
      MFEM_DEVICE_SYNC;
      mdofs += MDofs();
   }
};

/// Linear Form Extension Tests
#define VectorLinExtTest(Kernel,DIM,VDIM)\
static void VTEST_##Kernel(bm::State &state){\
   VectorTest<Kernel##Integrator,DIM,VDIM> ker(state.range(0));\
   while (state.KeepRunning()) { ker.benchmark(); }\
   state.counters["MDof/s"] = bm::Counter(ker.SumMdofs(), bm::Counter::kIsRate);}\
BENCHMARK(VTEST_##Kernel)->DenseRange(1,6)->Unit(bm::kMillisecond);

/// Vector linear form tests
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

/// Linear Form Extension Benchs
#define LinExtBench(DIM,VDIM,LVL)\
static void BENCH_##LVL##_##VDIM##D(bm::State &state){\
   Bench<DIM,VDIM,LinearAssemblyLevel::LVL> ker(state.range(0));\
   while (state.KeepRunning()) { ker.benchmark(); }\
   state.counters["MDof/s"] = bm::Counter(ker.SumMdofs(), bm::Counter::kIsRate);}\
BENCHMARK(BENCH_##LVL##_##VDIM##D)->DenseRange(1,6)->Unit(bm::kMillisecond);

/// 1D scalar linear form bench
//LinExtBench(1,FULL)
//LinExtBench(1,LEGACY)
LinExtBench(3,3,FULL)

/// 2D vector linear form bench
//LinExtBench(2,FULL)
//LinExtBench(2,LEGACY)

/// 3D vector linear form bench
//LinExtBench(3,FULL)
//LinExtBench(3,LEGACY)


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
