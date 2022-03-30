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

#include <cmath>

#define MFEM_NVTX_COLOR Lime
#include "general/nvtx.hpp"

#include "general/forall.hpp"
#include "linalg/kernels.hpp"
#include "linalg/tensor.hpp"

#include "miniapps/elasticity/materials/neohookean.hpp"
#include "miniapps/elasticity/kernels/elasticity_kernels.hpp"
//#include "miniapps/elasticity/kernels/kernel_helpers.hpp"

using namespace mfem::internal;

constexpr int DIM = 3;

/// Define the identity tensor in DIM dimensions
MFEM_DEVICE static constexpr auto I = IsotropicIdentity<DIM>();

template <typename T>
MFEM_HOST_DEVICE tensor<T, DIM, DIM>
stress(const tensor<T, DIM, DIM> &__restrict__ dudx)
{
   const double D1 = 100.0;
   const double C1 = 50.0;
   T J = det(I + dudx);
   T p = -2.0 * D1 * J * (J - 1);
   auto devB = dev(dudx + transpose(dudx) + dot(dudx, transpose(dudx)));
   auto sigma = -(p / J) * I + 2 * (C1 / pow(J, 5.0 / 3.0)) * devB;
   return sigma;
}

MFEM_HOST_DEVICE tensor<double, DIM, DIM>
material_action_of_gradient_dual(const tensor<double, DIM, DIM> &dudx,
                                 const tensor<double, DIM, DIM> &ddudx)
{
   auto sigma = stress(make_tensor<DIM, DIM>([&](int i, int j)
   {
      return dual<double, double> {dudx[i][j], ddudx[i][j]};
   }));
   return make_tensor<DIM, DIM>([&](int i, int j) { return sigma[i][j].gradient; });
}

////// ApplyGradient3D with static B & G
namespace WithStaticBG
{

template<int d1d, int q1d>
const tensor<double, q1d, d1d> CopyFromArray(const Array<double> &A)
{
   NVTX("CopyFromArray");
   return make_tensor<q1d, d1d>([&](int i, int j) { return A[i + q1d*j]; });
}

// Holds a single tensor of basis functions evaluated at quadrature points.
template<int d1d, int q1d>
const tensor<double, q1d, d1d> &AsTensorB(const Array<double> &A)
{
   NVTX("AsTensorB");
   static const tensor<double, q1d, d1d> B = CopyFromArray<d1d, q1d>(A);
   return B;
}

// Holds a single tensor of gradients of basis functions
// evaluated at quadrature points.
template<int d1d, int q1d>
const tensor<double, q1d, d1d> &AsTensorG(const Array<double> &A)
{
   NVTX("AsTensorG");
   static const tensor<double, q1d, d1d> G = CopyFromArray<d1d, q1d>(A);
   return G;
}

template <int d1d, int q1d> static inline
void ApplyGradient3D(const int ne,
                     const Array<double> &B_, const Array<double> &G_,
                     const Array<double> &W_, const Vector &Jacobian_,
                     const Vector &detJ_, const Vector &dU_, Vector &dF_,
                     const Vector &U_)
{
   NVTX("ApplyGradient3D");
   constexpr int dim = DIM;

   // Basis functions evaluated at quadrature points.
   const tensor<double, q1d, d1d> &B = AsTensorB<d1d, q1d>(B_);

   // Gradients of basis functions evaluated at quadrature points.
   const tensor<double, q1d, d1d> &G = AsTensorG<d1d, q1d>(G_);

   const auto qweights = Reshape(W_.Read(), q1d, q1d, q1d);
   // Jacobians of the element transformations at all quadrature points in
   // column-major layout q1d x q1d x q1d x sdim x dim x ne
   const auto J = Reshape(Jacobian_.Read(), q1d, q1d, q1d, dim, dim, ne);
   const auto detJ = Reshape(detJ_.Read(), q1d, q1d, q1d, ne);
   // Input vector
   // d1d x d1d x d1d x vdim x ne
   const auto dU = Reshape(dU_.Read(), d1d, d1d, d1d, dim, ne);
   // Output vector
   // d1d x d1d x d1d x vdim x ne
   auto force = Reshape(dF_.ReadWrite(), d1d, d1d, d1d, dim, ne);
   // Input vector
   // d1d x d1d x d1d x vdim x ne
   const auto U = Reshape(U_.Read(), d1d, d1d, d1d, dim, ne);

   MFEM_FORALL_3D(e, ne, q1d, q1d, q1d,
   {
      // shared memory placeholders for temporary contraction results
      MFEM_SHARED tensor<double, 2, 3, q1d, q1d, q1d> smem;
      // cauchy stress
      MFEM_SHARED tensor<double, q1d, q1d, q1d, dim, dim> invJ_dsigma_detJw;
      // du/dxi, ddu/dxi
      MFEM_SHARED_3D_BLOCK_TENSOR( dudxi, double, q1d, q1d, q1d, dim, dim);
      MFEM_SHARED_3D_BLOCK_TENSOR(ddudxi, double, q1d, q1d, q1d, dim, dim);

      const auto U_el = Reshape(&U(0, 0, 0, 0, e), d1d, d1d, d1d, dim);
      KernelHelpers::CalcGrad(B, G, smem, U_el, dudxi);

      const auto dU_el = Reshape(&dU(0, 0, 0, 0, e), d1d, d1d, d1d, dim);
      KernelHelpers::CalcGrad(B, G, smem, dU_el, ddudxi);

      MFEM_FOREACH_THREAD(qx, x, q1d)
      {
         MFEM_FOREACH_THREAD(qy, y, q1d)
         {
            MFEM_FOREACH_THREAD(qz, z, q1d)
            {
               auto invJqp = inv(make_tensor<dim, dim>(
               [&](int i, int j) { return J(qx, qy, qz, i, j, e); }));

               auto dudx = dudxi(qz, qy, qx) * invJqp;
               auto ddudx = ddudxi(qz, qy, qx) * invJqp;

               auto dsigma = material_action_of_gradient_dual(dudx, ddudx);

               invJ_dsigma_detJw(qx, qy, qz) =
                  invJqp * dsigma * detJ(qx, qy, qz, e) * qweights(qx, qy, qz);
            }
         }
      }
      MFEM_SYNC_THREAD;
      auto F = Reshape(&force(0, 0, 0, 0, e), d1d, d1d, d1d, dim);
      KernelHelpers::CalcGradTSum(B, G, smem, invJ_dsigma_detJw, F);
   }); // for each element
}

} // namespace WithStaticBG


struct ElasticityBench
{
   const Nvtx NLEBench_nvtx = {"ElasticityBench"};
   const int p, c, q, n, nx, ny, nz;
   const Nvtx mesh_nvtx = {"mesh"};
   Mesh mesh;
   const int ne, dim, vdim;
   H1_FECollection fec;
   const Nvtx fes_nvtx = {"fes"};
   FiniteElementSpace fes;
   const IntegrationRule &ir;
   const Nvtx gf_nvtx = {"gf"};
   const GeometricFactors *geometric_factors;
   const DofToQuad *maps;
   const int d1d, q1d;
   const Nvtx R_nvtx = {"R"};
   const Operator *R;
   mutable Vector X, Y, S;
   using material_type =
      NeoHookeanMaterial<DIM, GradientType::DualNumbers>;
   material_type material;
   std::function<void(const int,
                      const Array<double>&,
                      const Array<double>&,
                      const Array<double>&,
                      const Vector&, const Vector&,
                      const Vector&, Vector&, const Vector&)>
   ApplyGradient3D_wrapper_with_static_BG,
   ApplyGradient3D_wrapper;

   bool use_cache = false; // true ?
   mutable bool recompute_cache = false;
   Vector dsigma_cache;

   const int dofs;
   double mdof;
   const Nvtx Ready_nvtx = {"Ready"};

   ElasticityBench(int p, int side):
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
      maps(&fes.GetFE(0)->GetDofToQuad(ir, DofToQuad::TENSOR)),
      d1d(maps->ndof),
      q1d(maps->nqpt),
      R(fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC)),

      X(R->Height()),
      Y(R->Height()),
      S(R->Height()),
      material(),
      ApplyGradient3D_wrapper_with_static_BG([=](const int NE,
                                                 const Array<double> &B,
                                                 const Array<double> &G,
                                                 const Array<double> &W,
                                                 const Vector &J,
                                                 const Vector &detJ,
                                                 const Vector &dU,
                                                 Vector &dF,
                                                 const Vector &U)
   {
      const int id = (d1d << 4) | q1d;
      switch (id)
      {
         case 0x22: return WithStaticBG::ApplyGradient3D<2,2>(NE,B,G,W,J,detJ,dU,dF,U);
         case 0x33: return WithStaticBG::ApplyGradient3D<3,3>(NE,B,G,W,J,detJ,dU,dF,U);
         case 0x44: return WithStaticBG::ApplyGradient3D<4,4>(NE,B,G,W,J,detJ,dU,dF,U);
         default: MFEM_ABORT("Unknown kernel: 0x" << std::hex << id << std::dec);
      }
   }),
   ApplyGradient3D_wrapper([=](const int NE,
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
                  const Vector &U_, const material_type &material,
                  const bool use_cache_, const bool recompute_cache_,
                  Vector &dsigma_cache_) = nullptr;
      const int id = (d1d << 4) | q1d;
      switch (id)
      {
         case 0x22:
            ker = ElasticityKernels::ApplyGradient3D<2,2,material_type>; break;
         case 0x33:
            ker = ElasticityKernels::ApplyGradient3D<3,3,material_type>; break;
         case 0x44:
            ker = ElasticityKernels::ApplyGradient3D<4,4,material_type>; break;
         default: MFEM_ABORT("Unknown kernel: 0x" << std::hex << id << std::dec);
      }
      ker(NE,B,G,W,J,detJ,dU,dF,U,
          material,use_cache, recompute_cache, dsigma_cache);
   }),
   dofs(fes.GetVSize()),
   mdof(0.0)
   {
      X.UseDevice(true);
      Y.UseDevice(true);
      S.UseDevice(true);

      {
         NVTX("Set");
         { NVTX("Xpi"); X = M_PI; }
         { NVTX("Ypi"); Y = M_PI; }
         { NVTX("Spi"); S = M_PI; }
      }

      {
         NVTX("Reads");
         { NVTX("B"); maps->B.Read();}
         { NVTX("G"); maps->G.Read();}
         { NVTX("W"); ir.GetWeights().Read();}
         { NVTX("J"); geometric_factors->J.Read();}
         { NVTX("detJ"); geometric_factors->detJ.Read();}
         { NVTX("Xr"); X.Read();}
         { NVTX("Yr"); Y.Read();}
         { NVTX("Sr"); S.Read();}
      }

      tic_toc.Clear();
   }


   void ApplyGradient3D_with_static_BG()
   {
      MFEM_DEVICE_SYNC;
      tic_toc.Start();
      ApplyGradient3D_wrapper_with_static_BG(ne,
                                             maps->B,
                                             maps->G,
                                             ir.GetWeights(),
                                             geometric_factors->J,
                                             geometric_factors->detJ,
                                             X, Y, S);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * dofs;
   }

   void ApplyGradient3D()
   {
      MFEM_DEVICE_SYNC;
      tic_toc.Start();
      ApplyGradient3D_wrapper(ne,
                              maps->B,
                              maps->G,
                              ir.GetWeights(),
                              geometric_factors->J,
                              geometric_factors->detJ,
                              X, Y, S);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * dofs;
   }

   double Mdofs() const { return mdof / tic_toc.RealTime(); }
};

/// The different orders the tests can run
#define P_ORDERS bm::CreateDenseRange(1,3,1)

/// The different sides of the cartesian 3D mesh
#define N_SIDES bm::CreateDenseRange(10,120,10)
#define MAX_NDOFS 8*1024*1024

/// Kernels definitions and registrations
#define Benchmark(Name)\
static void Name(bm::State &state){\
   const int p = state.range(0);\
   const int side = state.range(1);\
   ElasticityBench nle(p, side);\
   if (nle.dofs > MAX_NDOFS) { state.SkipWithError("MAX_NDOFS"); }\
   while (state.KeepRunning()) { nle.Name(); }\
   bm::Counter::Flags flags = bm::Counter::kIsIterationInvariantRate;\
   state.counters["MDof"] = bm::Counter(1e-6*nle.dofs, flags);\
   state.counters["MDof/s"] = bm::Counter(nle.Mdofs());\
   state.counters["dofs"] = bm::Counter(nle.dofs);\
   state.counters["p"] = bm::Counter(p);\
}\
BENCHMARK(Name)\
            -> ArgsProduct({P_ORDERS, N_SIDES})\
            -> Unit(bm::kMillisecond);
// -> Iterations(500);

/// Possible benchmarks
Benchmark(ApplyGradient3D)
Benchmark(ApplyGradient3D_with_static_BG)

/**
 * @brief main entry point
 * --benchmark_filter=KerGrad/4/20
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
