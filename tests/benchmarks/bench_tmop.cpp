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

#ifdef MFEM_USE_BENCHMARK

#include "fem/tmop.hpp"
#include <cassert>
#include <memory>
#include <string>
#include <cmath>

struct TMOP
{
   const int p, q, n, nx, ny, nz, dim = 3;
   const bool check_x, check_y, check_z, checked;
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

   TMOP(int p, int c, bool p_eq_q = false):
      p(p),
      q(2*p + (p_eq_q ? 0 : 2)),
      n((assert(c>=p),c/p)),
      nx(n + (p*(n+1)*p*n*p*n < c*c*c ?1:0)),
      ny(n + (p*(n+1)*p*(n+1)*p*n < c*c*c ?1:0)),
      nz(n),
      check_x(p*nx * p*ny * p*nz <= c*c*c),
      check_y(p*(nx+1) * p*(ny+1) * p*nz > c*c*c),
      check_z(p*(nx+1) * p*(ny+1) * p*(nz+1) > c*c*c),
      checked((assert(check_x && check_y && check_z), true)),
      mesh(Mesh::MakeCartesian3D(nx,ny,nz,Element::HEXAHEDRON)),
      target_t(TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE),
      target_c(target_t),
      fec(p, dim),
      fes(&mesh, &fec, dim),
      R(fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC)),
      ir(&IntRules.Get(fes.GetFE(0)->GetGeomType(), q)),
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
   }

   void AddMultPA()
   {
      nlfi.AddMultPA(xe,ye);
      MFEM_DEVICE_SYNC;
      mdof += 1e-6 * dofs;
   }

   void AddMultGradPA()
   {
      nlfi.AddMultGradPA(xe,ye);
      MFEM_DEVICE_SYNC;
      mdof += 1e-6 * dofs;
   }

   void GetLocalStateEnergyPA()
   {
      const double energy = nlfi.GetLocalStateEnergyPA(xe);
      MFEM_DEVICE_SYNC;
      MFEM_CONTRACT_VAR(energy);
      mdof += 1e-6 * dofs;
   }

   void AssembleGradDiagonalPA()
   {
      nlfi.AssembleGradDiagonalPA(de);
      MFEM_DEVICE_SYNC;
      mdof += 1e-6 * dofs;
   }

   double Mdof() const { return mdof; }
};

// The different orders the tests can run
#define P_ORDERS bm::CreateDenseRange(1,4,1)

// The different sides of the mesh
#define N_SIDES bm::CreateDenseRange(10,84,4)
#define MAX_NUMBER_OF_DOFS 2*1024*1024

// P_EQ_Q selects the D1D & Q1D to use instantiated kernels
//  P_EQ_Q: 0x22, 0x33, 0x44, 0x55
// !P_EQ_Q: 0x23, 0x34, 0x45, 0x56
#define P_EQ_Q {false,true}

/**
  Kernels definitions and registrations
*/
#define BENCHMARK_TMOP(Bench)\
static void Bench(bm::State &state){\
   const int p = state.range(0);\
   TMOP ker(p, state.range(1), false);\
   if (ker.dofs > MAX_NUMBER_OF_DOFS) { state.SkipWithError("MAX_NUMBER_OF_DOFS"); }\
   while (state.KeepRunning()) { ker.Bench(); }\
   state.counters["MDofs"] = bm::Counter(ker.Mdof(), bm::Counter::kIsRate);\
   state.counters["Dofs"] = bm::Counter(ker.dofs);\
   state.counters["p"] = bm::Counter(p);}\
 BENCHMARK(Bench)->ArgsProduct({P_ORDERS,N_SIDES})->Unit(bm::kMicrosecond);
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
