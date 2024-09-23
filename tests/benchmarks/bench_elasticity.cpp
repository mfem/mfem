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

#include <cmath>

using namespace mfem::internal; // for IsotropicIdentity

#include "miniapps/hooke/materials/neohookean.hpp"
#include "miniapps/hooke/kernels/elasticity_kernels.hpp"

struct ElasticityBench
{
   static constexpr int DIM = 3;
   using material_t = NeoHookeanMaterial<DIM, GradientType::InternalFwd>;

   const int p, c, q, n, nx, ny, nz;
   Mesh mesh;
   const int ne, dim, vdim;
   H1_FECollection fec;
   FiniteElementSpace fes;
   const IntegrationRule &ir;
   const GeometricFactors *geometric_factors;
   const DofToQuad *maps;
   const int d1d, q1d;
   const Operator *R;
   mutable Vector X, Y, S;
   material_t material;
   std::function<void(const int,
                      const Array<double>&,
                      const Array<double>&,
                      const Array<double>&,
                      const Vector&, const Vector&,
                      const Vector&, Vector&, const Vector&)> ApplyGradient3D;
   const bool use_cache;
   mutable bool recompute_cache = false;
   Vector dsigma_cache;
   const int dofs;
   double mdof;

   ElasticityBench(int p, int side, bool use_cache):
      p(p),
      c(side),
      q(2*p + 1),
      n((assert(c>=p),c/p)),
      nx(n + (p*(n+1)*p*n*p*n < c*c*c ?1:0)),
      ny(n + (p*(n+1)*p*(n+1)*p*n < c*c*c ?1:0)),
      nz(n),
      mesh(Mesh::MakeCartesian3D(nx,ny,nz, Element::HEXAHEDRON)),
      ne(mesh.GetNE()),
      dim(mesh.SpaceDimension()),
      vdim(mesh.SpaceDimension()),
      fec(p, dim),
      fes(&mesh, &fec, vdim, Ordering::byNODES),
      ir(IntRules.Get(Geometry::Type::CUBE, q)),
      geometric_factors(fes.GetMesh()->
                        GetGeometricFactors(ir, GeometricFactors::JACOBIANS |
                                            GeometricFactors::DETERMINANTS)),
      maps(&fes.GetTypicalFE()->GetDofToQuad(ir, DofToQuad::TENSOR)),
      d1d(maps->ndof),
      q1d(maps->nqpt),
      R(fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC)),
      X(R->Height()),
      Y(R->Height()),
      S(R->Height()),
      material(),
      ApplyGradient3D([=](const int NE,
                          const Array<double> &B,
                          const Array<double> &G,
                          const Array<double> &W,
                          const Vector &J,
                          const Vector &detJ,
                          const Vector &dU,
                          Vector &dF,
                          const Vector &U)
   {
      void (*ker)(const int ne,
                  const Array<double> &B_, const Array<double> &G_,
                  const Array<double> &W_, const Vector &Jacobian_,
                  const Vector &detJ_, const Vector &dU_, Vector &dF_,
                  const Vector &U_, const material_t &material,
                  const bool use_cache_, const bool recompute_cache_,
                  Vector &dsigma_cache_) = nullptr;
      const int id = (d1d << 4) | q1d;
      switch (id)
      {
         case 0x22: ker = ElasticityKernels::ApplyGradient3D<2,2,material_t>; break;
         case 0x33: ker = ElasticityKernels::ApplyGradient3D<3,3,material_t>; break;
         case 0x44: ker = ElasticityKernels::ApplyGradient3D<4,4,material_t>; break;
         default: MFEM_ABORT("Unknown kernel: 0x" << std::hex << id << std::dec);
      }
      ker(NE, B, G, W, J, detJ, dU, dF, U,
          material, use_cache, recompute_cache, dsigma_cache);
   }),
   use_cache(use_cache),
   dsigma_cache(use_cache ? ne*q1d*q1d*q1d*dim*dim*dim*dim:0),
   dofs(fes.GetVSize()),
   mdof(0.0)
   {
      X.UseDevice(true);
      Y.UseDevice(true);
      S.UseDevice(true);
      dsigma_cache.UseDevice(true);

      X = M_PI;
      Y = M_PI;
      S = M_PI;

      maps->B.Read();
      maps->G.Read();
      ir.GetWeights().Read();
      geometric_factors->J.Read();
      geometric_factors->detJ.Read();
      X.Read();
      Y.Read();
      S.Read();
      dsigma_cache.Read();
   }

   void Gradient3D()
   {
      MFEM_DEVICE_SYNC;
      ApplyGradient3D(ne,
                      maps->B,
                      maps->G,
                      ir.GetWeights(),
                      geometric_factors->J,
                      geometric_factors->detJ,
                      X, Y, S);
      mdof += 1e-6 * dofs;
   }
};

/// The different orders the tests can run
#define P_ORDERS bm::CreateDenseRange(1,3,1)

/// The different sides of the cartesian 3D mesh
#define N_SIDES bm::CreateDenseRange(10,120,10)
#define MAX_NDOFS 8*1024*1024

/// The different cache options
#define USE_CACHE {false, true}

/// Kernels definitions and registrations
#define Benchmark(Kernel)\
static void Kernel(bm::State &state){\
   const int p = state.range(0);\
   const int side = state.range(1);\
   const bool use_cache = state.range(2);\
   ElasticityBench nle(p, side, use_cache);\
   if (nle.dofs > MAX_NDOFS) { state.SkipWithError("MAX_NDOFS"); }\
   while (state.KeepRunning()) { nle.Kernel(); }\
   bm::Counter::Flags invrt_rate = bm::Counter::kIsIterationInvariantRate;\
   state.counters["MDof"] = bm::Counter(1e-6*nle.dofs, invrt_rate);\
   state.counters["dofs"] = bm::Counter(nle.dofs);\
   state.counters["p"] = bm::Counter(p);\
   state.counters["use_cache"] = bm::Counter(use_cache);\
}\
BENCHMARK(Kernel)\
            -> ArgsProduct({P_ORDERS, N_SIDES, USE_CACHE})\
            -> Unit(bm::kMillisecond);

/// Possible benchmarks
Benchmark(Gradient3D)

/**
 * @brief main entry point
 * --benchmark_filter=Gradient3D/3/30
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
