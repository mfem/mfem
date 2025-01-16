// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

#include "fem/tmop.hpp"
#include <cassert>
#include <memory>
#include <cmath>

struct TMOP
{
   const int N, p, q, dim = 3;
   Mesh mesh;
   TMOP_Metric_302 metric;
   TargetConstructor::TargetType target_t;
   TargetConstructor target_c;
   H1_FECollection fec;
   FiniteElementSpace fes;
   const Operator *R;
   const IntegrationRule *ir;
   TMOP_Integrator nlfi;
   const int dofs;
   GridFunction x;
   Vector de,xe,ye;
   double mdof;

   TMOP(int p, bool p_eq_q = false):
      N(Device::IsEnabled()?16:8),
      p(p), q(2*p + (p_eq_q ? 0 : 2)),
      mesh(Mesh::MakeCartesian3D(N,N,N,Element::HEXAHEDRON)),
      target_t(TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE),
      target_c(target_t),
      fec(p, dim),
      fes(&mesh, &fec, dim),
      R(fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC)),
      ir(&IntRules.Get(mesh.GetTypicalElementGeometry(), q)),
      nlfi(&metric, &target_c),
      dofs(fes.GetVSize()),
      x(&fes),
      de(R->Height(), Device::GetMemoryType()),
      xe(R->Height(), Device::GetMemoryType()),
      ye(R->Height(), Device::GetMemoryType()),
      mdof(0.0)
   {
      mesh.SetNodalGridFunction(&x);
      target_c.SetNodes(x);

      R->Mult(x, xe);
      ye = 0.0;

      nlfi.SetIntegrationRule(*ir);
      nlfi.AssemblePA(fes);
      nlfi.AssembleGradPA(xe,fes);

      tic_toc.Clear();
   }

   void AddMultPA()
   {
      tic_toc.Start();
      nlfi.AddMultPA(xe,ye);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * dofs;
   }

   void AddMultGradPA()
   {
      tic_toc.Start();
      nlfi.AddMultGradPA(xe,ye);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * dofs;
   }

   void GetLocalStateEnergyPA()
   {
      tic_toc.Start();
      const double energy = nlfi.GetLocalStateEnergyPA(xe);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      MFEM_CONTRACT_VAR(energy);
      mdof += 1e-6 * dofs;
   }

   void AssembleGradDiagonalPA()
   {
      tic_toc.Start();
      nlfi.AssembleGradDiagonalPA(de);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * dofs;
   }

   double Mdof() const { return mdof; }

   double Mdofs() const { return mdof / tic_toc.RealTime(); }
};

// The different orders the tests can run
#define P_ORDERS {1,2,3,4}

// P_EQ_Q selects the D1D & Q1D to use instantiated kernels
//  P_EQ_Q: 0x22, 0x33, 0x44, 0x55
// !P_EQ_Q: 0x23, 0x34, 0x45, 0x56
#define P_EQ_Q {false,true}

/**
 * @brief The Kernel bm::Fixture struct
 */
struct Kernel: public bm::Fixture
{
   std::unique_ptr<TMOP> ker;
   ~Kernel() { assert(ker == nullptr); }

   using bm::Fixture::SetUp;
   void SetUp(const bm::State& state) BENCHMARK_OVERRIDE
   { ker.reset(new TMOP(state.range(1), state.range(0))); }

   using bm::Fixture::TearDown;
   void TearDown(const bm::State &) BENCHMARK_OVERRIDE { ker.reset(); }
};

/**
  Fixture kernels definitions and registrations
*/
#define BENCHMARK_TMOP_F(Bench)\
BENCHMARK_DEFINE_F(Kernel,Bench)(bm::State &state){\
   assert(ker.get());\
   while (state.KeepRunning()) { ker->Bench(); }\
   state.counters["MDof"] = bm::Counter(ker->Mdof(), bm::Counter::kIsRate);\
   state.counters["MDof/s"] = bm::Counter(ker->Mdofs());}\
 BENCHMARK_REGISTER_F(Kernel,Bench)->ArgsProduct({P_EQ_Q,P_ORDERS})->Unit(bm::kMicrosecond);
/// creating/registering, not used
//BENCHMARK_TMOP_F(AddMultPA)
//BENCHMARK_TMOP_F(AddMultGradPA)
//BENCHMARK_TMOP_F(GetLocalStateEnergyPA)
//BENCHMARK_TMOP_F(AssembleGradDiagonalPA)

/**
  Kernels definitions and registrations
*/
#define BENCHMARK_TMOP(Bench)\
static void Bench(bm::State &state){\
   TMOP ker(state.range(1),state.range(0));\
   while (state.KeepRunning()) { ker.Bench(); }\
   state.counters["MDof"] = bm::Counter(ker.Mdof(), bm::Counter::kIsRate);\
   state.counters["MDof/s"] = bm::Counter(ker.Mdofs());}\
 BENCHMARK(Bench)->ArgsProduct({P_EQ_Q,P_ORDERS})->Unit(bm::kMicrosecond);
/// creating/registering
BENCHMARK_TMOP(AddMultPA)
BENCHMARK_TMOP(AddMultGradPA)
BENCHMARK_TMOP(GetLocalStateEnergyPA)
BENCHMARK_TMOP(AssembleGradDiagonalPA)

/**
 * @brief main entry point
 * --benchmark_filter=AddMultPA/4
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
