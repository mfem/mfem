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

template<TargetConstructor::TargetType TC> struct AddMultPA_Kernel_3D
{
   const int N, order, quad_order, dim = 3;
   Mesh mesh;
   H1_FECollection fec;
   FiniteElementSpace fes, fes1;
   NonlinearForm nlf;
   const int dofs;
   Array<int> ess_tdof_list, ess_bdr;
   ConstantCoefficient one, lim;
   LinearForm b;
   GridFunction x, d;
   Operator *A;
   Vector B, X;
   double mdof;

   AddMultPA_Kernel_3D(int order):
      N(Device::IsEnabled()?16:8),
      order(order),
      quad_order(2*order),     // 0x22, 0x33, 0x44, 0x55, max 2.1926k MDOF/s
      //quad_order(2*order+2), // 0x23, 0x34, 0x45, 0x56, max 1.3016k MDOF/s
      mesh(Mesh::MakeCartesian3D(N,N,N,Element::HEXAHEDRON)),
      fec(order, dim),
      fes(&mesh, &fec, dim), // vector
      fes1(&mesh, &fec), // scalar
      nlf(&fes),
      dofs(fes.GetVSize()),
      ess_bdr(mesh.bdr_attributes.Max()),
      one((ess_bdr=1,fes.GetEssentialTrueDofs(ess_bdr,ess_tdof_list), 1.0)),
      lim(1./M_PI),
      b(&fes),
      x(&fes),
      d(&fes1),
      mdof(0.0)
   {
      b.AddDomainIntegrator(new DomainLFIntegrator(one));
      b.Assemble();

      mesh.SetNodalGridFunction(&x);

      TMOP_QualityMetric *metric = new TMOP_Metric_302;
      TargetConstructor::TargetType target_t = TC;
      TargetConstructor *target_c = new TargetConstructor(target_t);
      target_c->SetNodes(x);

      const int geom_type = fes.GetFE(0)->GetGeomType();
      //IntegrationRules *IntRulesLo =
      //        new IntegrationRules(0, Quadrature1D::GaussLobatto);
      //const IntegrationRule *ir = &IntRulesLo->Get(geom_type, quad_order);
      const IntegrationRule *ir = &IntRules.Get(geom_type, quad_order);

      TMOP_Integrator *he_nlf_integ = new TMOP_Integrator(metric, target_c);
      he_nlf_integ->SetIntegrationRule(*ir);
      //he_nlf_integ->EnableNormalization(x);
      //he_nlf_integ->EnableLimiting(x, d = 1.0, lim);

      nlf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      nlf.AddDomainIntegrator(he_nlf_integ);
      nlf.Setup();
      nlf.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      tic_toc.Clear();
   }

   void benchmark()
   {
      tic_toc.Start();
      nlf.Mult(B,X); // AddMultPA_Kernel_3D
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += (1e-6 * dofs);
   }

   double Mdof() const { return mdof; }

   double Mdofs() const { return mdof / tic_toc.RealTime(); }
};

/**
  Kernels
*/
#define BENCHMARK_TMOP_KERNEL(TC)\
static void TMOP(bm::State &state){\
   AddMultPA_Kernel_3D<TC> ker(state.range(0));\
   while (state.KeepRunning()) { ker.benchmark(); }\
   state.counters["MDof"] = bm::Counter(ker.Mdof(), bm::Counter::kIsRate);\
   state.counters["MDof/s"] = bm::Counter(ker.Mdofs());}\
BENCHMARK(TMOP)->DenseRange(1,4);

/**
  Launch all benchmarks: AddMultPA_Kernel_3D
  */
BENCHMARK_TMOP_KERNEL(TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE)

/**
 * @brief main entry point
 * --benchmark_filter=Mass/6
 * --benchmark_filter=Diffusion/6
 * --benchmark_context=device=cpu
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
