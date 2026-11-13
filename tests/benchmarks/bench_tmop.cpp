// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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
#include "fem/tmop_tools.hpp"
#include <cassert>
#include <memory>
#include <cmath>

static real_t LimitingCoefficient(const Vector &x) { return x(0) + 1.0; }

// Match the approximate per-rank element count in triple-pt ref2 profiles.
constexpr int TMOP_BENCH_N = 37;
constexpr int TMOP_BENCH_D1D = 3;
constexpr int TMOP_BENCH_Q1D = 4;

class BenchmarkTMOPIntegrator : public TMOP_Integrator
{
public:
   using TMOP_Integrator::TMOP_Integrator;

   void AssembleGradPA3DMetric(const Vector &x) const { AssembleGradPA_3D(x); }
   void AssembleGradPA3DLimiting(const Vector &x) const
   { AssembleGradPA_C0_3D(x); }

   void AddMultPA3DMetric(const Vector &x, Vector &y) const
   { AddMultPA_3D(x, y); }
   void AddMultPA3DLimiting(const Vector &x, Vector &y) const
   { AddMultPA_C0_3D(x, y); }

   void AddMultGradPA3DMetric(const Vector &x, Vector &y) const
   { AddMultGradPA_3D(x, y); }
   void AddMultGradPA3DLimiting(const Vector &x, Vector &y) const
   { AddMultGradPA_C0_3D(x, y); }

   real_t GetLocalStateEnergyPA3DMetric(const Vector &x) const
   {
      real_t energy = 0.0;
      GetLocalStateEnergyPA_3D(x, energy);
      return energy;
   }
   real_t GetLocalStateEnergyPA3DLimiting(const Vector &x) const
   { return GetLocalStateEnergyPA_C0_3D(x); }

   void AssembleGradDiagonalPA3DMetric(Vector &d) const
   { AssembleDiagonalPA_3D(d); }
   void AssembleGradDiagonalPA3DLimiting(Vector &d) const
   { AssembleDiagonalPA_C0_3D(d); }

   void ComputeAllElementTargetsPA(const Vector &x) const
   { ComputeAllElementTargets(x); }
};

struct TMOP
{
   const int N = TMOP_BENCH_N, p = TMOP_BENCH_D1D - 1,
             q1d = TMOP_BENCH_Q1D, dim = 3;
   const int ir_order = 2 * TMOP_BENCH_Q1D - 1;
   Mesh mesh;
   std::unique_ptr<TMOP_QualityMetric> metric;
   std::unique_ptr<TargetConstructor> target_c;
   H1_FECollection size_fec;
   FiniteElementSpace size_fes;
   GridFunction size;
   H1_FECollection fec;
   FiniteElementSpace fes;
   H1_FECollection dist_fec;
   FiniteElementSpace dist_fes;
   GridFunction dist;
   FunctionCoefficient lim_coeff;
   const Operator *R;
   const IntegrationRule *ir;
   BenchmarkTMOPIntegrator nlfi;
   const int dofs;
   GridFunction x;
   Vector de,xe,ye;
   double mdof;

   TMOP(bool metric_321 = true, bool limiting = false,
        bool discrete_size = false):
      mesh(Mesh::MakeCartesian3D(N,N,N,Element::HEXAHEDRON)),
      metric(metric_321 ?
             std::unique_ptr<TMOP_QualityMetric>(new TMOP_Metric_321) :
             std::unique_ptr<TMOP_QualityMetric>(new TMOP_Metric_302)),
      target_c(discrete_size ?
               std::unique_ptr<TargetConstructor>(new DiscreteAdaptTC(
                                                     TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE)) :
               std::unique_ptr<TargetConstructor>(new TargetConstructor(
                                                     TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE))),
      size_fec(1, dim),
      size_fes(&mesh, &size_fec),
      size(&size_fes),
      fec(p, dim),
      fes(&mesh, &fec, dim),
      dist_fec(p, dim),
      dist_fes(&mesh, &dist_fec),
      dist(&dist_fes),
      lim_coeff(LimitingCoefficient),
      R(fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC)),
      ir(&IntRules.Get(mesh.GetTypicalElementGeometry(), ir_order)),
      nlfi(metric.get(), target_c.get()),
      dofs(fes.GetVSize()),
      x(&fes),
      de(R->Height(), Device::GetMemoryType()),
      xe(R->Height(), Device::GetMemoryType()),
      ye(R->Height(), Device::GetMemoryType()),
      mdof(0.0)
   {
      mesh.SetNodalGridFunction(&x);

      if (discrete_size)
      {
         auto *discr_tc = dynamic_cast<DiscreteAdaptTC *>(target_c.get());
         MFEM_VERIFY(discr_tc, "");
         size = 1.0;
         discr_tc->SetAdaptivityEvaluator(new AdvectorCG(AssemblyLevel::PARTIAL));
         discr_tc->SetSerialDiscreteTargetSize(size);
         discr_tc->SetMinSizeForTargets(1.0);
      }

      target_c->SetNodes(x);

      if (limiting)
      {
         dist = 1.0;
         nlfi.EnableLimiting(x, dist, lim_coeff);
      }

      R->Mult(x, xe);
      ye = 0.0;

      nlfi.SetIntegrationRule(*ir);
      nlfi.AssemblePA(fes);
      nlfi.AssembleGradPA(xe,fes);

      tic_toc.Clear();
   }

   void AssembleGradPA()
   {
      tic_toc.Start();
      nlfi.AssembleGradPA(xe,fes);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * dofs;
   }

   void TMOPAssembleGradPA3D()
   {
      tic_toc.Start();
      nlfi.AssembleGradPA3DMetric(xe);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * dofs;
   }

   void TMOP_AssembleGradPA_C0_3D()
   {
      tic_toc.Start();
      nlfi.AssembleGradPA3DLimiting(xe);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * dofs;
   }

   void AddMultPA()
   {
      tic_toc.Start();
      nlfi.AddMultPA(xe,ye);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * dofs;
   }

   void TMOPAddMultPA3D()
   {
      tic_toc.Start();
      nlfi.AddMultPA3DMetric(xe,ye);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * dofs;
   }

   void TMOP_AddMultPA_C0_3D()
   {
      tic_toc.Start();
      nlfi.AddMultPA3DLimiting(xe,ye);
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

   void TMOP_AddMultGradPA_3D()
   {
      tic_toc.Start();
      nlfi.AddMultGradPA3DMetric(xe,ye);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * dofs;
   }

   void TMOP_AddMultGradPA_C0_3D()
   {
      tic_toc.Start();
      nlfi.AddMultGradPA3DLimiting(xe,ye);
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

   void TMOPEnergyPA3D()
   {
      tic_toc.Start();
      const double energy = nlfi.GetLocalStateEnergyPA3DMetric(xe);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      MFEM_CONTRACT_VAR(energy);
      mdof += 1e-6 * dofs;
   }

   void TMOP_EnergyPA_C0_3D()
   {
      tic_toc.Start();
      const double energy = nlfi.GetLocalStateEnergyPA3DLimiting(xe);
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

   void TMOP_AssembleDiagPA_3D()
   {
      tic_toc.Start();
      nlfi.AssembleGradDiagonalPA3DMetric(de);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * dofs;
   }

   void TMOP_AssembleDiagPA_C0_3D()
   {
      tic_toc.Start();
      nlfi.AssembleGradDiagonalPA3DLimiting(de);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * dofs;
   }

   void TMOP_DatcSize_3D()
   {
      tic_toc.Start();
      nlfi.ComputeAllElementTargetsPA(xe);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * dofs;
   }

   double Mdof() const { return mdof; }

   double Mdofs() const { return mdof / tic_toc.RealTime(); }
};

struct TMOPMinDetSolver : public TMOPNewtonSolver
{
   using TMOPNewtonSolver::TMOPNewtonSolver;

   real_t MinDet3D(const FiniteElementSpace &fes, GridFunction &nodes,
                   const Vector &d)
   {
      x_0.SetSpace(const_cast<FiniteElementSpace *>(nodes.FESpace()));
      x_0 = nodes;
      periodic = false;
      return MinDetJpr_3D(&fes, d);
   }
};

struct TMOPMinDet
{
   const int N = TMOP_BENCH_N, p = TMOP_BENCH_D1D - 1,
             q1d = TMOP_BENCH_Q1D, dim = 3;
   const int ir_order = 2 * TMOP_BENCH_Q1D - 1;
   Mesh mesh;
   H1_FECollection fec;
   FiniteElementSpace fes;
   const IntegrationRule *ir;
   GridFunction x;
   Vector d;
   TMOPMinDetSolver solver;
   double mdof;

   TMOPMinDet():
      mesh(Mesh::MakeCartesian3D(N,N,N,Element::HEXAHEDRON)),
      fec(p, dim),
      fes(&mesh, &fec, dim),
      ir(&IntRules.Get(mesh.GetTypicalElementGeometry(), ir_order)),
      x(&fes),
      d(fes.GetVSize(), Device::GetMemoryType()),
      solver(*ir),
      mdof(0.0)
   {
      mesh.SetNodalGridFunction(&x);
      d = 0.0;
      tic_toc.Clear();
   }

   void MinDetJpr()
   {
      tic_toc.Start();
      const real_t min_det = solver.MinDet3D(fes, x, d);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      MFEM_CONTRACT_VAR(min_det);
      mdof += 1e-6 * fes.GetVSize();
   }

   double Mdof() const { return mdof; }

   double Mdofs() const { return mdof / tic_toc.RealTime(); }
};

/**
  Kernels definitions and registrations
*/
#define BENCHMARK_TMOP(Bench, Method, Label, Limiting, DiscreteSize)\
static void Bench(bm::State &state){\
   TMOP ker(true, Limiting, DiscreteSize);\
   while (state.KeepRunning()) { ker.Method(); }\
   state.counters["MDof"] = bm::Counter(ker.Mdof(), bm::Counter::kIsRate);\
   state.counters["MDof/s"] = bm::Counter(ker.Mdofs());}\
 BENCHMARK(Bench)->Name(Label)->Unit(bm::kMicrosecond);

static void MinDetJpr_3x4(bm::State &state)
{
   TMOPMinDet ker;
   while (state.KeepRunning()) { ker.MinDetJpr(); }
   state.counters["MDof"] = bm::Counter(ker.Mdof(), bm::Counter::kIsRate);
   state.counters["MDof/s"] = bm::Counter(ker.Mdofs());
}

/// creating/registering
BENCHMARK_TMOP(AssembleGradPA_3x4, AssembleGradPA, "AssembleGradPA_3x4", false, false)
BENCHMARK_TMOP(AddMultGradPA_3x4, AddMultGradPA, "AddMultGradPA_3x4", false, false)
BENCHMARK_TMOP(AddMultPA_3x4, AddMultPA, "AddMultPA_3x4", false, false)
BENCHMARK_TMOP(GetLocalStateEnergyPA_3x4, GetLocalStateEnergyPA, "GetLocalStateEnergyPA_3x4", false, false)
BENCHMARK_TMOP(AssembleGradDiagonalPA_3x4, AssembleGradDiagonalPA, "AssembleGradDiagonalPA_3x4", false, false)

BENCHMARK_TMOP(AssembleGradPA_Limiting_3x4, AssembleGradPA, "AssembleGradPA_Limiting_3x4", true, false)
BENCHMARK_TMOP(AddMultGradPA_Limiting_3x4, AddMultGradPA, "AddMultGradPA_Limiting_3x4", true, false)
BENCHMARK_TMOP(AddMultPA_Limiting_3x4, AddMultPA, "AddMultPA_Limiting_3x4", true, false)
BENCHMARK_TMOP(GetLocalStateEnergyPA_Limiting_3x4, GetLocalStateEnergyPA, "GetLocalStateEnergyPA_Limiting_3x4", true, false)
BENCHMARK_TMOP(AssembleGradDiagonalPA_Limiting_3x4, AssembleGradDiagonalPA, "AssembleGradDiagonalPA_Limiting_3x4", true, false)

BENCHMARK_TMOP(AssembleGradPA_DiscreteSize_3x4, AssembleGradPA, "AssembleGradPA_DiscreteSize_3x4", false, true)
BENCHMARK_TMOP(AddMultPA_DiscreteSize_3x4, AddMultPA, "AddMultPA_DiscreteSize_3x4", false, true)

BENCHMARK_TMOP(TMOPAssembleGradPA3D_Metric321_3x4, TMOPAssembleGradPA3D, "TMOPAssembleGradPA3D::Mult<3,4,TMOP_PA_Metric_321,3,4>", false, false)
BENCHMARK_TMOP(TMOP_AssembleGradPA_C0_3D_3x4, TMOP_AssembleGradPA_C0_3D, "TMOP_AssembleGradPA_C0_3D<3,4,3,4>", true, false)
BENCHMARK_TMOP(TMOPAddMultPA3D_Metric321_3x4, TMOPAddMultPA3D, "TMOPAddMultPA3D::Mult<3,4,TMOP_PA_Metric_321,3,4>", false, false)
BENCHMARK_TMOP(TMOP_AddMultPA_C0_3D_3x4, TMOP_AddMultPA_C0_3D, "TMOP_AddMultPA_C0_3D<3,4,3,4>", true, false)
BENCHMARK_TMOP(TMOP_AddMultGradPA_3D_3x4, TMOP_AddMultGradPA_3D, "TMOP_AddMultGradPA_3D<3,4,3,4>", false, false)
BENCHMARK_TMOP(TMOP_AddMultGradPA_C0_3D_3x4, TMOP_AddMultGradPA_C0_3D, "TMOP_AddMultGradPA_C0_3D<3,4,3,4>", true, false)
BENCHMARK_TMOP(TMOPEnergyPA3D_Metric321_3x4, TMOPEnergyPA3D, "TMOPEnergyPA3D::Mult<3,4,TMOP_PA_Metric_321,3,4>", false, false)
BENCHMARK_TMOP(TMOP_EnergyPA_C0_3D_3x4, TMOP_EnergyPA_C0_3D, "TMOP_EnergyPA_C0_3D<3,4,3,4>", true, false)
BENCHMARK_TMOP(TMOP_AssembleDiagPA_3D_3x4, TMOP_AssembleDiagPA_3D, "TMOP_AssembleDiagPA_3D<3,4,3,4>", false, false)
BENCHMARK_TMOP(TMOP_AssembleDiagPA_C0_3D_3x4, TMOP_AssembleDiagPA_C0_3D, "TMOP_AssembleDiagPA_C0_3D<3,4,3,4>", true, false)
BENCHMARK_TMOP(TMOP_DatcSize_3D_3x4, TMOP_DatcSize_3D, "TMOP_DatcSize_3D<2,4,2,4>", false, true)

BENCHMARK(MinDetJpr_3x4)->Name("TMOP_MinDetJpr_3D<3,4,3,4>")
                         ->Unit(bm::kMicrosecond);

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
   auto global_context = bmi::GetGlobalContext();
   if (bmi::GetGlobalContext() != nullptr)
   {
      const auto device = global_context->find("device");
      if (device != global_context->end())
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
