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
#include "linalg/tensor_isotropic.hpp"

using namespace mfem::internal;

constexpr int DIM = 3;

/// Define the identity tensor in DIM dimensions
MFEM_DEVICE static constexpr auto I = IsotropicIdentity<DIM>();

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

// MFEM_SHARED_3D_BLOCK_TENSOR definition
// Should be moved in backends/cuda/hip header files.
#if defined(__CUDA_ARCH__)
#define MFEM_SHARED_3D_BLOCK_TENSOR(name,T,bx,by,bz,...)\
MFEM_SHARED tensor<T,bx,by,bz,__VA_ARGS__> name;\
name(threadIdx.x, threadIdx.y, threadIdx.z) = tensor<T,__VA_ARGS__> {};
#else
#define MFEM_SHARED_3D_BLOCK_TENSOR(name,...) tensor<__VA_ARGS__> name {};
#endif

template <int dim, int d1d, int q1d>
static inline MFEM_HOST_DEVICE void
CalcGrad(const tensor<double, q1d, d1d> &B, // q1d x d1d
         const tensor<double, q1d, d1d> &G, // q1d x d1d
         tensor<double,2,3,q1d,q1d,q1d> &smem,
         const DeviceTensor<4, const double> &U, // d1d x d1d x d1d x dim
         tensor<double, q1d, q1d, q1d, dim, dim> &dUdxi)
{
   for (int c = 0; c < dim; ++c)
   {
      MFEM_FOREACH_THREAD(dz,z,d1d)
      {
         MFEM_FOREACH_THREAD(dy,y,d1d)
         {
            MFEM_FOREACH_THREAD(dx,x,d1d)
            {
               smem(0,0,dx,dy,dz) = U(dx,dy,dz,c);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,d1d)
      {
         MFEM_FOREACH_THREAD(dy,y,d1d)
         {
            MFEM_FOREACH_THREAD(qx,x,q1d)
            {
               double u = 0.0;
               double v = 0.0;
               for (int dx = 0; dx < d1d; ++dx)
               {
                  const double input = smem(0,0,dx,dy,dz);
                  u += input * B(qx,dx);
                  v += input * G(qx,dx);
               }
               smem(0,1,dz,dy,qx) = u;
               smem(0,2,dz,dy,qx) = v;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,d1d)
      {
         MFEM_FOREACH_THREAD(qy,y,q1d)
         {
            MFEM_FOREACH_THREAD(qx,x,q1d)
            {
               double u = 0.0;
               double v = 0.0;
               double w = 0.0;
               for (int dy = 0; dy < d1d; ++dy)
               {
                  u += smem(0,2,dz,dy,qx) * B(qy,dy);
                  v += smem(0,1,dz,dy,qx) * G(qy,dy);
                  w += smem(0,1,dz,dy,qx) * B(qy,dy);
               }
               smem(1,0,dz,qy,qx) = u;
               smem(1,1,dz,qy,qx) = v;
               smem(1,2,dz,qy,qx) = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,q1d)
      {
         MFEM_FOREACH_THREAD(qy,y,q1d)
         {
            MFEM_FOREACH_THREAD(qx,x,q1d)
            {
               double u = 0.0;
               double v = 0.0;
               double w = 0.0;
               for (int dz = 0; dz < d1d; ++dz)
               {
                  u += smem(1,0,dz,qy,qx) * B(qz,dz);
                  v += smem(1,1,dz,qy,qx) * B(qz,dz);
                  w += smem(1,2,dz,qy,qx) * G(qz,dz);
               }
               dUdxi(qz,qy,qx,c,0) += u;
               dUdxi(qz,qy,qx,c,1) += v;
               dUdxi(qz,qy,qx,c,2) += w;
            }
         }
      }
      MFEM_SYNC_THREAD;
   } // vdim
}

template <int dim, int d1d, int q1d>
static inline MFEM_HOST_DEVICE void
CalcGradTSum(const tensor<double, q1d, d1d> &B,
             const tensor<double, q1d, d1d> &G,
             tensor<double, 2, 3, q1d, q1d, q1d> &smem,
             const tensor<double, q1d, q1d, q1d, dim, dim> &U, // q1d x q1d x q1d x dim
             DeviceTensor<4, double> &F)                       // d1d x d1d x d1d x dim
{
   for (int c = 0; c < dim; ++c)
   {
      MFEM_FOREACH_THREAD(qz, z, q1d)
      {
         MFEM_FOREACH_THREAD(qy, y, q1d)
         {
            MFEM_FOREACH_THREAD(dx, x, d1d)
            {
               double u = 0.0, v = 0.0, w = 0.0;
               for (int qx = 0; qx < q1d; ++qx)
               {
                  u += U(qx, qy, qz, 0, c) * G(qx, dx);
                  v += U(qx, qy, qz, 1, c) * B(qx, dx);
                  w += U(qx, qy, qz, 2, c) * B(qx, dx);
               }
               smem(0, 0, qz, qy, dx) = u;
               smem(0, 1, qz, qy, dx) = v;
               smem(0, 2, qz, qy, dx) = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz, z, q1d)
      {
         MFEM_FOREACH_THREAD(dy, y, d1d)
         {
            MFEM_FOREACH_THREAD(dx, x, d1d)
            {
               double u = 0.0, v = 0.0, w = 0.0;
               for (int qy = 0; qy < q1d; ++qy)
               {
                  u += smem(0, 0, qz, qy, dx) * B(qy, dy);
                  v += smem(0, 1, qz, qy, dx) * G(qy, dy);
                  w += smem(0, 2, qz, qy, dx) * B(qy, dy);
               }
               smem(1, 0, qz, dy, dx) = u;
               smem(1, 1, qz, dy, dx) = v;
               smem(1, 2, qz, dy, dx) = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz, z, d1d)
      {
         MFEM_FOREACH_THREAD(dy, y, d1d)
         {
            MFEM_FOREACH_THREAD(dx, x, d1d)
            {
               double u = 0.0, v = 0.0, w = 0.0;
               for (int qz = 0; qz < q1d; ++qz)
               {
                  u += smem(1, 0, qz, dy, dx) * B(qz, dz);
                  v += smem(1, 1, qz, dy, dx) * B(qz, dz);
                  w += smem(1, 2, qz, dy, dx) * G(qz, dz);
               }
               const double sum = u + v + w;
               F(dx, dy, dz, c) += sum;
            }
         }
      }
      MFEM_SYNC_THREAD;
   }
}

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
      CalcGrad(B, G, smem, U_el, dudxi);

      const auto dU_el = Reshape(&dU(0, 0, 0, 0, e), d1d, d1d, d1d, dim);
      CalcGrad(B, G, smem, dU_el, ddudxi);

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
      CalcGradTSum(B, G, smem, invJ_dsigma_detJw, F);
   }); // for each element
}

struct NLEBench
{
   const Nvtx NLEBench_nvtx = {"NLEBench"};
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
   std::function<void(const int,
                      const Array<double>&, const Array<double>&,
                      const Array<double>&,
                      const Vector&, const Vector&,
                      const Vector&, Vector&, const Vector&)>
   element_apply_gradient_kernel_wrapper;
   const int dofs;
   double mdof;
   const Nvtx Ready_nvtx = {"Ready"};

   NLEBench(int p, int side):
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
      element_apply_gradient_kernel_wrapper([=](const int NE,
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
         case 0x22: return ApplyGradient3D<2,2>(NE,B,G,W,J,detJ,dU,dF,U);
         case 0x33: return ApplyGradient3D<3,3>(NE,B,G,W,J,detJ,dU,dF,U);
         case 0x44: return ApplyGradient3D<4,4>(NE,B,G,W,J,detJ,dU,dF,U);
         default: MFEM_ABORT("Unknown kernel: 0x" << std::hex << id << std::dec);
      }
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

   void KerGrad()
   {
      MFEM_DEVICE_SYNC;
      tic_toc.Start();
      element_apply_gradient_kernel_wrapper(ne,
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
   NLEBench nle(p, side);\
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
Benchmark(KerGrad)

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
