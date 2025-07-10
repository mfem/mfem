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

#include "bench.hpp" // IWYU pragma: keep

#ifdef MFEM_USE_BENCHMARK

#include <cstdlib>
#include <memory>

#include <fem/qinterp/det.cpp>
#include <fem/qinterp/grad.hpp> // IWYU pragma: keep

#include "fem/dfem/doperator.hpp"
#include <linalg/tensor.hpp>
#include "tests/benchmarks/kernels.hpp"

#include "fem/kernels.hpp"
namespace ker = kernels::internal;

#if defined(__has_include) && __has_include("general/nvtx.hpp") && !defined(_WIN32)
#undef NVTX_COLOR
#define NVTX_COLOR ::nvtx::kNvidia
#include "general/nvtx.hpp"
#else
#define dbg(...)
#endif

using namespace mfem;

using mfem::future::tuple;
using mfem::future::tensor;

using future::DifferentiableOperator;
using future::UniformParameterSpace;
using future::ParameterFunction;
using future::FieldDescriptor;
using future::make_tensor;
using future::Gradient;
using future::Weight;
using future::Identity;

#ifdef MFEM_USE_ENZYME
using dscalar_t = real_t;
#else
using mfem::future::dual;
using dscalar_t = dual<real_t, real_t>;
#endif

#if ((defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)) ||       \
     (defined(MFEM_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)))

template<int N>
using MQZ = std::integral_constant<int, 0>;

#else

template<int N>
using MQZ = std::integral_constant<int, N>;

constexpr int SetMaxOf2(int n) { return mfem::kernels::internal::NextMultipleOf<2>(n); }
#endif

/// Max number of DOFs ////////////////////////////////////////////////////////
#if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
constexpr int MAX_NDOFS = 128 * 1024;
constexpr int NDOFS_INC = 25;
#else
constexpr int MAX_NDOFS = 10 * 1024 * 1024;
constexpr int NDOFS_INC = 25;
#endif

/// Benchmarks Arguments //////////////////////////////////////////////////////
static void OrderSideVersionArgs(bmi::Benchmark *b)
{
   const auto est = [](int c) { return (c + 1) * (c + 1) * (c + 1); };
   const auto versions = { 0, 1, 2, 3 };
   for (auto k : versions)
   {
      for (int p = 8; p >= 1; p -= 1)
      {
         for (int c = NDOFS_INC; est(c) <= MAX_NDOFS; c += NDOFS_INC)
         {
            b->Args({ k, p, c });
         }
      }
   }
}

/// Globals ///////////////////////////////////////////////////////////////////
Device *device_ptr = nullptr;
static int gD1D = 0, gQ1D = 0;
static bool use_new_kernels = false;
static bool use_kernels_specialization = true;

/// StiffnessIntegrator ///////////////////////////////////////////////////////
struct StiffnessIntegrator : public BilinearFormIntegrator
{
   const FiniteElementSpace *fes;
   const real_t *B, *G, *DX;
   int ne, d1d, q1d;
   Vector J0, dx;

public:
   StiffnessIntegrator()
   {
      dbg();
      NVTX();
      if (!use_kernels_specialization) { return; }
      dbg("Adding StiffnessKernels specializations");
      StiffnessKernels::Specialization<2, 3>::Add();
      StiffnessKernels::Specialization<3, 4>::Add();
      StiffnessKernels::Specialization<4, 5>::Add();
      StiffnessKernels::Specialization<5, 6>::Add();
      StiffnessKernels::Specialization<6, 7>::Add();
      StiffnessKernels::Specialization<7, 8>::Add();
      StiffnessKernels::Specialization<9, 10>::Add();
   }

   void AssemblePA(const FiniteElementSpace &fespace) override
   {
      NVTX();
      fes = &fespace;
      auto *mesh = fes->GetMesh();
      const int DIM = mesh->Dimension();
      ne = mesh->GetNE();
      const auto p = fes->GetFE(0)->GetOrder();
      const auto q = 2 * p + mesh->GetElementTransformation(0)->OrderW();
      const auto type = mesh->GetElementBaseGeometry(0);
      const IntegrationRule &ir = IntRules.Get(type, q);
      const int NQPT = ir.GetNPoints();
      d1d = p + 1;
      q1d = IntRules.Get(Geometry::SEGMENT, ir.GetOrder()).GetNPoints();
      MFEM_VERIFY(d1d == gD1D, "D1D mismatch: " << d1d << " != " << gD1D);
      MFEM_VERIFY(q1d == gQ1D, "Q1D mismatch: " << q1d << " != " << gQ1D);
      MFEM_VERIFY(NQPT == q1d * q1d * q1d, "");
      const DofToQuad *maps =
         &fes->GetFE(0)->GetDofToQuad(ir, DofToQuad::TENSOR);
      const GridFunction *nodes = (mesh->EnsureNodes(), mesh->GetNodes());
      const FiniteElementSpace *nfes = nodes->FESpace();
      const int nVDIM = nfes->GetVDim();
      dx.SetSize(nVDIM * DIM * NQPT * ne, Device::GetDeviceMemoryType());
      J0.SetSize(nVDIM * DIM * NQPT * ne, Device::GetDeviceMemoryType());
      dx.UseDevice(true), J0.UseDevice(true);
      B = maps->B.Read(), G = maps->G.Read(), DX = dx.Read();

      const Operator *NR =
         nfes->GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
      const QuadratureInterpolator *nqi = nfes->GetQuadratureInterpolator(ir);
      nqi->SetOutputLayout(QVectorLayout::byVDIM);
      const int nd = nfes->GetFE(0)->GetDof();
      Vector xe(nVDIM * nd * ne, Device::GetDeviceMemoryType());
      NR->Mult(*nodes, (xe.UseDevice(true), xe));
      nqi->Derivatives(xe, J0);

      const int Q1D = q1d;
      const auto w_r = ir.GetWeights().Read();
      const auto W = Reshape(w_r, q1d, q1d, q1d);
      const auto J = Reshape(J0.Read(), 3, 3, q1d, q1d, q1d, ne);
      auto DX_w = Reshape(dx.Write(), 3, 3, q1d, q1d, q1d, ne);

      mfem::forall_3D(ne, Q1D, Q1D, Q1D,[=] MFEM_HOST_DEVICE(int e)
      {
         MFEM_FOREACH_THREAD_DIRECT(qz, z, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  const real_t w = W(qx, qy, qz);
                  const real_t *Jtr = &J(0, 0, qx, qy, qz, e);
                  const real_t detJ = kernels::Det<3>(Jtr);
                  const real_t wd = w * detJ;
                  const real_t D[9] = { wd, 0.0, 0.0,
                                        0.0, wd, 0.0,
                                        0.0, 0.0, wd
                                      };
                  real_t Jrt[9], A[9];
                  kernels::CalcInverse<3>(Jtr, Jrt);
                  kernels::MultABt(3, 3, 3, D, Jrt, A);
                  kernels::Mult(3, 3, 3, A, Jrt, &DX_w(0, 0, qx, qy, qz, e));
               }
            }
         }
         MFEM_SYNC_THREAD;
      });
   }

   ///////////////////////////////////////////////////////////////////
   /// BP3/1/6/25 | 12.5ms | 12.5ms | 10 | 15.625k | 40.1397/s | 6 | 1
   template <int T_D1D = 0, int T_Q1D = 0>
   static void StiffnessMult(const int NE, const real_t *b, const real_t *g,
                             const real_t *dx, const real_t *xe, real_t *ye,
                             const int d1d, const int q1d)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      constexpr int DIM = 3, VDIM = 1;
      const auto XE = Reshape(xe, D1D, D1D, D1D, VDIM, NE);
      const auto DX = Reshape(dx, 3, 3, Q1D, Q1D, Q1D, NE);
      auto YE = Reshape(ye, D1D, D1D, D1D, VDIM, NE);

      mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
      {
         constexpr int MD1 = T_D1D > 0 ? T_D1D : 32;
         constexpr int MQ1 = T_Q1D > 0 ? T_Q1D : 32;

         MFEM_SHARED real_t smem[MQ1][MQ1];
         MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];
         ker::vd_regs3d_t<VDIM, DIM, MQ1> r0, r1;

         ker::LoadMatrix(D1D, Q1D, b, sB);
         ker::LoadMatrix(D1D, Q1D, g, sG);

         ker::LoadDofs3d(e, D1D, XE, r0);
         ker::Grad3d(D1D, Q1D, smem, sB, sG, r0, r1);

         for (int qz = 0; qz < Q1D; qz++)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  real_t v[3], u[3] = { r1[0][0][qz][qy][qx],
                                        r1[0][1][qz][qy][qx],
                                        r1[0][2][qz][qy][qx]
                                      };
                  const real_t *dx = &DX(0, 0, qx, qy, qz, e);
                  kernels::Mult(3, 3, dx, u, v);
                  r0[0][0][qz][qy][qx] = v[0];
                  r0[0][1][qz][qy][qx] = v[1];
                  r0[0][2][qz][qy][qx] = v[2];
               }
            }
         }
         ker::GradTranspose3d(D1D, Q1D, smem, sB, sG, r0, r1);
         ker::WriteDofs3d(e, D1D, r1, YE);
      });
   }

   ///////////////////////////////////////////////////////////////////
   // BP3/1/6/25 | 49.5 ms | 49.5 ms | 10 | 15.625k | 10.0983/s | 6 | 1
   template <int T_D1D = 0, int T_Q1D = 0>
   static void StiffnessMultVD(const int NE, const real_t *b, const real_t *g,
                               const real_t *dx, const real_t *xe, real_t *ye,
                               const int d1d, const int q1d)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      db1("StiffnessMultVD: D1D:{} Q1D:{}", D1D, Q1D);

      constexpr int DIM = 3, VDIM = 1;
      const auto XE = Reshape(xe, D1D, D1D, D1D, VDIM, NE);
      const auto DX = Reshape(dx, 3, 3, Q1D, Q1D, Q1D, NE);
      auto YE = Reshape(ye, D1D, D1D, D1D, VDIM, NE);

      mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
      {
         // constexpr int MD1 = T_D1D > 0 ? kernels::internal::SetMaxOf(T_D1D) : 32;
         // constexpr int MQ1 = T_Q1D > 0 ? kernels::internal::SetMaxOf(T_Q1D) : 32;
         constexpr int MD1 = T_D1D > 0 ? T_D1D : 32;
         constexpr int MQ1 = T_Q1D > 0 ? T_Q1D : 32;

         MFEM_SHARED real_t smem[MQ1][MQ1];
         MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];

         constexpr int MZ1 = MQZ<MQ1>::value;
         mfem::future::tensor<real_t, MQ1, MZ1, MZ1, VDIM, DIM> v0, v1;

         ker::LoadMatrix(D1D, Q1D, b, sB);
         ker::LoadMatrix(D1D, Q1D, g, sG);

         ker::vd::LoadDofs3d(e, D1D, XE, v0);
         ker::vd::Grad3d(D1D, Q1D, smem, sB, sG, v0, v1);

         for (int qz = 0; qz < Q1D; qz++)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  auto &vd_v = v0[qz][qy][qx][0];
                  const auto &vd_u = v1[qz][qy][qx][0];
                  const auto d = make_tensor<3, 3>([&](int i, int j) { return DX(i, j, qx, qy, qz, e); });
                  vd_v = d * vd_u;
               }
            }
         }
         ker::vd::GradTranspose3d(D1D, Q1D, smem, sB, sG, v0, v1);
         ker::vd::WriteDofs3d(e, D1D, v1, YE);
      });
   }

   using StiffnessKernelType = decltype(&StiffnessMult<>);
   MFEM_REGISTER_KERNELS(StiffnessKernels, StiffnessKernelType, (int, int));

   void AddMultPA(const Vector &x, Vector &y) const override
   {
      StiffnessKernels::Run(d1d, q1d,
                            ne, B, G, DX, x.Read(), y.ReadWrite(),
                            d1d, q1d);
   }
};
/*
--------------------------------------------------------------------------------------------------
Benchmark            Time             CPU   Iterations       Dofs     MDof/s          p    version
--------------------------------------------------------------------------------------------------
BP3/0/6/25        2.33 ms         2.29 ms          306    15.625k  218.454/s          6          0
BP3/0/6/50        2.95 ms         2.92 ms          232   132.055k 1.44567k/s          6          0
BP3/0/6/75        4.64 ms         4.61 ms          149   420.991k 2.92423k/s          6          0
BP3/0/6/100       8.84 ms         8.80 ms           78   1.02907M 3.74023k/s          6          0
BP3/0/6/125       15.5 ms         15.5 ms           45   1.95161M 4.02333k/s          6          0
BP3/0/6/150       26.5 ms         26.5 ms           26   3.44295M 4.16497k/s          6          0
BP3/0/6/175       40.9 ms         40.6 ms           17   5.35938M 4.22609k/s          6          0
BP3/0/6/200       61.3 ms         60.7 ms           12    8.1182M 4.27719k/s          6          0
--------------------------------------------------------------------------------------------------
StiffnessMultVD:
BP3/1/6/25        2.86 ms         2.80 ms          304    15.625k  178.514/s          6          1
BP3/1/6/50        3.39 ms         3.33 ms          210   132.055k 1.26724k/s          6          1
BP3/1/6/75        5.06 ms         4.99 ms          136   420.991k 2.69718k/s          6          1
BP3/1/6/100       8.71 ms         8.63 ms           80   1.02907M 3.81789k/s          6          1
BP3/1/6/125       15.5 ms         15.1 ms           46   1.95161M 4.13199k/s          6          1
BP3/1/6/150       24.4 ms         24.2 ms           29   3.44295M 4.54897k/s          6          1
BP3/1/6/175       37.4 ms         36.7 ms           19   5.35938M 4.67061k/s          6          1
BP3/1/6/200       56.1 ms         55.2 ms           13    8.1182M 4.70794k/s          6          1
--------------------------------------------------------------------------------------------------
StiffnessMult:
BP3/1/6/25        2.94 ms         2.86 ms          245    15.625k  174.602/s          6          1
BP3/1/6/50        3.44 ms         3.37 ms          206   132.055k 1.25531k/s          6          1
BP3/1/6/75        5.26 ms         5.15 ms          137   420.991k 2.61688k/s          6          1
BP3/1/6/100       9.15 ms         9.04 ms           77   1.02907M 3.64348k/s          6          1
BP3/1/6/125       15.7 ms         15.6 ms           45   1.95161M 4.01224k/s          6          1
BP3/1/6/150       25.1 ms         25.0 ms           28   3.44295M 4.40159k/s          6          1
BP3/1/6/175       37.7 ms         37.7 ms           18   5.35938M 4.54855k/s          6          1
BP3/1/6/200       56.2 ms         56.2 ms           12    8.1182M 4.62524k/s          6          1


[Darwin]
-------------------------------------------------------------------------------------------------
Benchmark           Time             CPU   Iterations       Dofs     MDof/s          p    version
-------------------------------------------------------------------------------------------------
StiffnessMult:
BP3/0/6/25       17.2 ms         17.2 ms           41    15.625k  29.0621/s          6          0
BP3/1/6/25       13.3 ms         13.1 ms           52    15.625k   38.234/s          6          1
BP3/2/6/25       39.9 ms         39.8 ms           18    15.625k  12.5509/s          6          2
BP3/3/6/25       24.4 ms         24.1 ms           29    15.625k  20.7329/s          6          3
-------------------------------------------------------------------------------------------------
StiffnessMultVD:
BP3/0/6/25       17.4 ms         17.4 ms           38    15.625k  28.8158/s          6          0
BP3/1/6/25       51.9 ms         51.9 ms           13    15.625k  9.63093/s          6          1
BP3/2/6/25       39.9 ms         39.9 ms           17    15.625k  12.5196/s          6          2
BP3/3/6/25       23.4 ms         23.4 ms           30    15.625k  21.3528/s          6          3
*/
template <int D1D, int Q1D>
StiffnessIntegrator::StiffnessKernelType
StiffnessIntegrator::StiffnessKernels::Kernel()
{
   return StiffnessMult<D1D, Q1D>;
   // return StiffnessMultVD<D1D, Q1D>;
}

StiffnessIntegrator::StiffnessKernelType
StiffnessIntegrator::StiffnessKernels::Fallback(int d1d, int q1d)
{
   // dbg("\x1b[33mFallback d1d:{} q1d:{}", d1d, q1d);
   return StiffnessMult<>;
}

/// BakeOff ///////////////////////////////////////////////////////////////////
template <int VDIM, bool GLL>
struct BakeOff
{
   static constexpr int DIM = 3;
   const int p, c, q, n, nx, ny, nz;
   const bool check_x, check_y, check_z, checked;
   Mesh smesh;
   ParMesh pmesh;
   H1_FECollection fec;
   ParFiniteElementSpace pfes;
   const Geometry::Type geom_type;
   IntegrationRules irs;
   const IntegrationRule *ir;
   ConstantCoefficient one;
   Vector uvec;
   VectorConstantCoefficient unit_vec;
   const int dofs;
   ParGridFunction *nodes;
   ParFiniteElementSpace& mfes;
   ParGridFunction x, y;
   ParBilinearForm a;
   std::unique_ptr<DifferentiableOperator> dop;
   const int elem_size, total_size, d1d, q1d;
   UniformParameterSpace qd_ps;
   ParameterFunction qdata;

   double mdofs{};

   BakeOff(int p, int side):
      p(p), c(side), q(2 * p + (GLL ? -1 : 3)), n((assert(c >= p), c / p)),
      nx(n + (p * (n + 1) * p * n * p * n < c * c * c ? 1 : 0)),
      ny(n + (p * (n + 1) * p * (n + 1) * p * n < c * c * c ? 1 : 0)), nz(n),
      check_x(p * nx * p * ny * p * nz <= c * c * c),
      check_y(p * (nx + 1) * p * (ny + 1) * p * nz > c * c * c),
      check_z(p * (nx + 1) * p * (ny + 1) * p * (nz + 1) > c * c * c),
      checked((assert(check_x &&check_y &&check_z), true)),
      smesh(Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON)),
      pmesh(MPI_COMM_WORLD, (smesh.EnsureNodes(), smesh)),
      fec(p, DIM, BasisType::GaussLobatto),
      pfes(&pmesh, &fec, VDIM),//, Ordering::byNODES),
      geom_type(pmesh.GetTypicalElementGeometry()),
      irs(0, GLL ? Quadrature1D::GaussLobatto : Quadrature1D::GaussLegendre),
      ir(&irs.Get(geom_type, q)), one(1.0), uvec(DIM),
      unit_vec((uvec = 1.0, uvec /= uvec.Norml2(), uvec)),
      dofs(pfes.GetTrueVSize()),
      nodes(static_cast<ParGridFunction*>(pmesh.GetNodes())),
      mfes(*nodes->ParFESpace()),
      x(&pfes),
      y(&pfes),
      a(&pfes),
      elem_size(DIM * DIM * ir->GetNPoints()),
      total_size(elem_size * pmesh.GetNE()),
      d1d(p + 1),
      q1d(IntRules.Get(Geometry::SEGMENT, ir->GetOrder()).GetNPoints()),
      qd_ps(pmesh, *ir, DIM*DIM),
      qdata(qd_ps)
   {
      NVTX_MARK_FUNCTION;
      // dbg("p:{} q:{}", p, q);
      // pmesh.SetCurvature(p);
      smesh.Clear();
      x = 0.0;

      gD1D = d1d, gQ1D = q1d;
      // dbg("D1D: {}, Q1D: {}", gD1D, gQ1D);
      qdata.UseDevice(true);
      MFEM_VERIFY(q1d*q1d*q1d == ir->GetNPoints(), "");
   }

   virtual void benchmark() = 0;

   double SumMdofs() const { return mdofs; }

   double MDofs() const { return 1e-6 * dofs; }
};

/// Diffusion /////////////////////////////////////////////////////////////////
template <int VDIM = 1, bool GLL = false>
struct Diffusion : public BakeOff<VDIM, GLL>
{
   static constexpr int DIM = 3;
   static constexpr int U = 0, Ξ = 1, Q = 2;

   const real_t rtol = 0.0;
   const int max_it = 32, print_lvl = -1;

   Array<int> ess_tdof_list, ess_bdr, all_domain_attr;
   ParLinearForm b;
   FieldDescriptor u_fd, Ξ_fd, q_fd;
   std::vector<FieldDescriptor> u_sol, q_param, Ξ_q_params;
   OperatorPtr A;
   Operator *A_ptr;
   Vector B, X;
   CGSolver cg;

   using BakeOff<VDIM, GLL>::a;
   using BakeOff<VDIM, GLL>::ir;
   using BakeOff<VDIM, GLL>::one;
   using BakeOff<VDIM, GLL>::pmesh;
   using BakeOff<VDIM, GLL>::pfes;
   using BakeOff<VDIM, GLL>::mfes;
   using BakeOff<VDIM, GLL>::x;
   using BakeOff<VDIM, GLL>::y;
   using BakeOff<VDIM, GLL>::mdofs;
   using BakeOff<VDIM, GLL>::dop;
   using BakeOff<VDIM, GLL>::nodes;
   using BakeOff<VDIM, GLL>::qdata;
   using BakeOff<VDIM, GLL>::qd_ps;
   using BakeOff<VDIM, GLL>::dofs;

   Diffusion(int version, int order, int side):
      BakeOff<VDIM, GLL>(order, side),
      ess_bdr(pmesh.bdr_attributes.Max()),
      all_domain_attr(pmesh.bdr_attributes.Max()),
      b(&pfes),
      u_fd{U, &pfes}, Ξ_fd{Ξ, &mfes}, q_fd{Q, &qd_ps},
      u_sol{u_fd},
      q_param {q_fd},
      Ξ_q_params {Ξ_fd, q_fd},
      cg(MPI_COMM_WORLD)
   {
      // dbg("pmesh.bdr_attributes.Max():{}",pmesh.bdr_attributes.Max());
      static_assert(VDIM == 1 && GLL == false);

      ess_bdr = 1;
      all_domain_attr = 1;
      pfes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      b.AddDomainIntegrator(new DomainLFIntegrator(this->one));
      b.UseFastAssembly(true);
      b.Assemble();

      if (version < 2) // standard, new PA regs
      {
         a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         if (version == 0) { a.AddDomainIntegrator(new DiffusionIntegrator(ir)); }
         if (version == 1) { a.AddDomainIntegrator(new StiffnessIntegrator()); }
         a.Assemble();
         a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
         if (version == 0)
         {
            BilinearFormIntegrator *bfi = a.GetDBFI()->operator[](0);
            auto *di = dynamic_cast<DiffusionIntegrator*>(bfi);
            assert(di);
            const int d1d = di->dofs1D, q1d = di->quad1D;
            // dbg("\x1b[33md1d: {} q1d: {}", d1d, q1d);
            MFEM_VERIFY(d1d == gD1D, "D1D mismatch: " << d1d << " != " << gD1D);
            MFEM_VERIFY(q1d == gQ1D, "Q1D mismatch: " << q1d << " != " << gQ1D);
         }
      }
      else if (version == 2) // 2: MF ∂fem
      {
         dbg("MF ∂fem");
         auto solutions = std::vector{FieldDescriptor{U, &pfes}};
         auto parameters = std::vector{FieldDescriptor{Ξ, &mfes}};
         dop = std::make_unique<DifferentiableOperator>(solutions, parameters, pmesh);
         dop->SetParameters({nodes});
         auto diffusion_mf_kernel =
            [] MFEM_HOST_DEVICE (const tensor<real_t, DIM>& Gu,
                                 const tensor<real_t, DIM, DIM>& J,
                                 const real_t& w)
         {
            auto invJ = inv(J);
            return tuple{((Gu * invJ)) * transpose(invJ) * det(J) * w};
         };
         dop->AddDomainIntegrator(diffusion_mf_kernel,
                                  tuple{Gradient<U>{}, Gradient<Ξ>{}, Weight{}},
                                  tuple{Gradient<U>{}},
                                  *ir, ess_bdr);
         dop->FormLinearSystem(ess_tdof_list, x, b, A_ptr, X, B);
         A.Reset(A_ptr);
      }
      else if (version == 3) // PA ∂fem
      {
         dbg("[PA ∂fem]");
         MFEM_ABORT("PA ∂fem is not implemented yet. ❌");
         /*auto W = Weight{};
         auto Iq = Identity<Q> {};
         auto Iu = Identity<U> {};
         auto Gu = Gradient<U> {};
         auto GΞ = Gradient<Ξ> {};
         tuple Gu_Iq = {Gu, Iq};
         tuple Iu_GΞ_W = {Iu, GΞ, W};

         dbg("[PA ∂fem] SETUP 🟣🟣🟣🟣");
         auto pa_setup_qf =
            [] MFEM_HOST_DEVICE(const real_t &u,
                                const tensor<real_t, DIM, DIM> &J,
                                const real_t &w)
         {
            return tuple{inv(J) * transpose(inv(J)) * det(J) * w};
         };
         DifferentiableOperator dSetup(u_sol, Ξ_q_params, pmesh);
         dSetup.AddDomainIntegrator(pa_setup_qf, Iu_GΞ_W, tuple{Iq}, *ir, ess_bdr);
         dSetup.SetParameters({nodes, &qdata});
         X.SetSize(pfes.GetTrueVSize());
         pfes.GetRestrictionMatrix()->Mult(x, X);
         dSetup.Mult(X, qdata);

         dbg("[PA ∂fem] APPLY 🟢🟢🟢🟢");
         auto pa_apply_qf =
            [] MFEM_HOST_DEVICE(const tensor<real_t, DIM> &Gu,
                                const tensor<real_t, DIM, DIM> &q)
         {
            return tuple{q * Gu};
         };
         dop = std::make_unique<DifferentiableOperator>(u_sol, q_param, pmesh);
         if (use_new_kernels) { dop->UseNewKernels(); }
         if (use_kernels_specialization) { dop->UseKernelsSpecialization(); }
         else { dbg("[PA ∂fem] NOT using kernels specialization"); }
         dop->AddDomainIntegrator(pa_apply_qf, Gu_Iq, tuple{Gu}, *ir, ess_bdr);
         dop->SetParameters({ &qdata });

         dop->FormLinearSystem(ess_tdof_list, x, b, A_ptr, X, B);
         A.Reset(A_ptr);*/
      }
      else if (version == 4) // Linearisation ∂fem
      {
         dbg("[Linearised ∂fem]");
         auto solutions = std::vector{FieldDescriptor{U, &pfes}};
         auto parameters = std::vector{FieldDescriptor{Ξ, &mfes}};
         auto derivatives = std::integer_sequence<size_t, U> {};
         dop = std::make_unique<DifferentiableOperator>(solutions, parameters, pmesh);
         const auto diffusion_mf_kernel =
            [] MFEM_HOST_DEVICE (const tensor<dscalar_t, DIM>& ∇u,
                                 const tensor<real_t, DIM, DIM>& J,
                                 const real_t& w)
         {
            const auto invJ = inv(J), TinJ = transpose(invJ);
            return tuple{((∇u * invJ)) * TinJ * det(J) * w};
         };
         dop->AddDomainIntegrator(diffusion_mf_kernel,
                                  tuple{Gradient<U>{}, Gradient<Ξ>{}, Weight{}},
                                  tuple{Gradient<U>{}},
                                  *ir, ess_bdr, derivatives);
         dop->SetParameters({nodes});
         auto dRdU = dop->GetDerivative(U, {&x}, {nodes});

         // constr_op.reset(new ConstrainedOperator(dRdU.get(), ess_bdr));
         // dop->FormLinearSystem(ess_tdof_list, x, b, A_ptr, X, B);
         // A.Reset(A_ptr);
         // A.Reset(new ConstrainedOperator(dRdU.get(), ess_bdr));
      }
      else { MFEM_ABORT("Invalid version"); }

      cg.SetOperator(*A);
      cg.iterative_mode = false;
      if (dofs < 128 * 1024) // check
      {
         cg.SetPrintLevel(3/*-1*/);
         cg.SetMaxIter(2000);
         cg.SetRelTol(1e-8);
         cg.SetAbsTol(0.0);
         cg.Mult(B, X);
         // MFEM_VERIFY(cg.GetConverged(), "❌ CG solver did not converge.");
         MFEM_DEVICE_SYNC;
         // mfem::out << "✅" << std::endl;
      }
      cg.SetAbsTol(0.0);
      cg.SetRelTol(rtol);
      cg.SetMaxIter(max_it);
      cg.SetPrintLevel(print_lvl);
      benchmark();
      mdofs = 0.0;
   }

   void benchmark() override
   {
      cg.Mult(B, X);
      MFEM_DEVICE_SYNC;
      mdofs += this->MDofs() * cg.GetNumIterations();
   }
};

///////////////////////////////////////////////////////////////////////////////
#define BakeOff_Problem(i, Problem)                                  \
   static void BP##i(bm::State &state)                               \
   {                                                                 \
      const auto version = static_cast<int>(state.range(0));         \
      const auto order = static_cast<int>(state.range(1));           \
      const auto side = static_cast<int>(state.range(2));            \
      Problem ker(version, order, side);                             \
      while (state.KeepRunning()) { ker.benchmark(); }               \
      bm::Counter::Flags flags = bm::Counter::kIsRate;               \
      state.counters["MDof/s"] = bm::Counter(ker.SumMdofs(), flags); \
      state.counters["Dofs"] = bm::Counter(ker.dofs);                \
      state.counters["p"] = bm::Counter(order);                      \
      state.counters["version"] = bm::Counter(version);              \
   }                                                                 \
   BENCHMARK(BP##i)                                                  \
      ->Apply(OrderSideVersionArgs)                                  \
      ->Unit(bm::kMillisecond)

BakeOff_Problem(3, Diffusion);

/// Basic Kernels Specializations /////////////////////////////////////////////
static void AddBasicKernelSpecializations()
{
   using Det = QuadratureInterpolator::DetKernels;
   Det::Specialization<3, 3, 2, 2>::Add();
   Det::Specialization<3, 3, 2, 3>::Add();
   Det::Specialization<3, 3, 2, 5>::Add();
   Det::Specialization<3, 3, 2, 6>::Add();
   Det::Specialization<3, 3, 2, 7>::Add();

   using Grad = QuadratureInterpolator::GradKernels;
   Grad::Specialization<3, QVectorLayout::byVDIM,  false, 3, 2, 3>::Add();
   Grad::Specialization<3, QVectorLayout::byVDIM,  false, 3, 2, 4>::Add();
   Grad::Specialization<3, QVectorLayout::byVDIM,  false, 3, 2, 5>::Add();
   Grad::Specialization<3, QVectorLayout::byVDIM,  false, 3, 2, 6>::Add();
   Grad::Specialization<3, QVectorLayout::byVDIM,  false, 3, 2, 7>::Add();
   Grad::Specialization<3, QVectorLayout::byVDIM,  false, 3, 2, 8>::Add();
   Grad::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 7>::Add();
   Grad::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 8>::Add();

   // using Diffusion = DiffusionIntegrator::ApplyPAKernels;
   // Diffusion::Specialization<3, 9, 10>::Add();
}

/// info //////////////////////////////////////////////////////////////////////
static void DumpVersionInfo()
{
   mfem::out << "\x1b[33m";
   mfem::out << "version 0: PA std" << std::endl;
   mfem::out << "version 1: PA new" << std::endl;
   mfem::out << "version 2: MF ∂fem" << std::endl;
   mfem::out << "version 3: PA ∂fem" << std::endl;
   mfem::out << "\x1b[m" << std::endl;
}

/// main //////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
   dbg();
   DumpVersionInfo();
   AddBasicKernelSpecializations();
   static mfem::MPI_Session mpi(argc, argv);

   bm::ConsoleReporter CR;
   bm::Initialize(&argc, argv);

   // Device setup, cpu by default
   std::string device_context = "cpu",
               kernels_context = "std",
               kernels_specialization = "yes";
   const auto global_context = bmi::GetGlobalContext();
   if (global_context != nullptr)
   {
      const auto device = global_context->find("device");
      if (device != global_context->end())
      {
         mfem::out << device->first << " : "
                   << device->second << std::endl;
         device_context = device->second;
      }

      const auto kernels = global_context->find("kernels");
      if (kernels != global_context->end())
      {
         mfem::out << kernels->first << " : "
                   << kernels->second << std::endl;
         kernels_context = kernels->second;
         MFEM_VERIFY(kernels_context == "std" || kernels_context == "new",
                     "Invalid kernels config: " << kernels_context);
         use_new_kernels = (kernels_context == "new");
      }

      const auto specialization = global_context->find("specialization");
      if (specialization != global_context->end())
      {
         mfem::out << specialization->first << " : "
                   << specialization->second << std::endl;
         kernels_specialization = specialization->second;
         MFEM_VERIFY(kernels_specialization == "yes" || kernels_specialization == "no",
                     "Invalid kernels specialization config: " << kernels_specialization);
         use_kernels_specialization = (kernels_specialization == "yes");
      }
   }
   dbg("device_config: {}", device_context);
   Device device(device_context.c_str());
   device_ptr = &device;
   device.Print();

   if (bm::ReportUnrecognizedArguments(argc, argv)) { return EXIT_FAILURE; }

   bm::RunSpecifiedBenchmarks(&CR);

   return EXIT_SUCCESS;
}

#endif // MFEM_USE_BENCHMARK
