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

#include <memory>

#include <fem/qinterp/det.cpp>
#include <fem/qinterp/grad.hpp> // IWYU pragma: keep

#include <fem/dfem/doperator.hpp>
#include <linalg/tensor.hpp>

#include "examples/dfem/kernels_pa.hpp"

#undef NVTX_COLOR
#define NVTX_COLOR nvtx::kAquamarine
#include "general/nvtx.hpp"

using namespace mfem;
using mfem::internal::tensor;

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
      for (int p = 4/*6*/; p >= 1; p -= 1)
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
static int D1D = 0, Q1D = 0;

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
      StiffnessKernels::Specialization<2, 3>::Add();
      StiffnessKernels::Specialization<3, 4>::Add();
      StiffnessKernels::Specialization<4, 5>::Add();
      StiffnessKernels::Specialization<5, 6>::Add();
      StiffnessKernels::Specialization<6, 7>::Add();
      StiffnessKernels::Specialization<7, 8>::Add();
   }

   void AssemblePA(const FiniteElementSpace &fespace) override
   {
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
      MFEM_VERIFY(d1d == D1D, "D1D mismatch: " << d1d << " != " << D1D);
      MFEM_VERIFY(q1d == Q1D, "Q1D mismatch: " << q1d << " != " << Q1D);
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

      const int D1D = d1d, Q1D = q1d;
      const auto w_r = ir.GetWeights().Read();
      const auto W = Reshape(w_r, Q1D, Q1D, Q1D);
      const auto J = Reshape(J0.Read(), 3, 3, Q1D, Q1D, Q1D, ne);
      auto DX_w = Reshape(dx.Write(), 3, 3, Q1D, Q1D, Q1D, ne);
      mfem::forall_3D(ne, Q1D, Q1D, Q1D,[=] MFEM_HOST_DEVICE(int e)
      {
         mfem::foreach_z_thread(Q1D,[&](int qz)
         {
            mfem::foreach_y_thread(Q1D,[&](int qy)
            {
               mfem::foreach_x_thread(Q1D,[&](int qx)
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
               });
            });
         });
         MFEM_SYNC_THREAD;
      });
   }

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
         constexpr int MQ1 = SetMaxOf(T_Q1D ? T_Q1D : 32);
         constexpr int MD1 = SetMaxOf(T_D1D ? T_D1D : 32);

         MFEM_SHARED real_t smem[MQ1][MQ1];
         MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];
         regs5d_t<VDIM, DIM, MQ1> r0, r1;

         LoadMatrix(D1D, Q1D, b, sB);
         LoadMatrix(D1D, Q1D, g, sG);

         LoadDofs3d<VDIM, DIM, MQ1>(e, D1D, XE, r0);
         regs_Grad3d<VDIM, DIM, MD1, MQ1>(D1D, Q1D, smem, sB, sG, r0, r1);

         for (int qz = 0; qz < Q1D; qz++)
         {
            mfem::foreach_y_thread(Q1D, [&](int qy)
            {
               mfem::foreach_x_thread(Q1D, [&](int qx)
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
               });
            });
         }
         regs_GradTranspose3d<VDIM, DIM, MD1, MQ1>(D1D, Q1D, smem, sB, sG,
                                                   r0, r1);
         WriteDofs3d<VDIM, DIM, MQ1>(e, D1D, r1, YE);
      });
   }

   using StiffnessKernelType = decltype(&StiffnessMult<>);
   MFEM_REGISTER_KERNELS(StiffnessKernels, StiffnessKernelType, (int, int));

   void AddMultPA(const Vector &x, Vector &y) const override
   {
      StiffnessKernels::Run(d1d, q1d, ne, B, G, DX, x.Read(), y.ReadWrite(),
                            d1d, q1d);
   }
};

template <int D1D, int Q1D>
StiffnessIntegrator::StiffnessKernelType
StiffnessIntegrator::StiffnessKernels::Kernel()
{
   return StiffnessMult<D1D, Q1D>;
}

StiffnessIntegrator::StiffnessKernelType
StiffnessIntegrator::StiffnessKernels::Fallback(int d1d, int q1d)
{
   dbg("\x1b[33mFallback d1d:{} q1d:{}", d1d, q1d);
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
   std::unique_ptr<DifferentiableOperator> ∂op;
   const int elem_size, total_size, d1d, q1d;
   ParametricSpace qdata_space;
   ParametricFunction qdata;

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
      pfes(&pmesh, &fec, VDIM, Ordering::byNODES),
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
      qdata_space(DIM, DIM * DIM, elem_size, total_size, d1d, q1d),
      qdata(qdata_space)
   {
      // dbg("p:{} q:{}", p, q);
      // pmesh.SetCurvature(p);
      smesh.Clear();
      x = 0.0;

      D1D = d1d, Q1D = q1d;
      dbg("D1D: {}, Q1D: {}", D1D, Q1D);
      qdata.UseDevice(true);
      assert(q1d*q1d*q1d == ir->GetNPoints());
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

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
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
   using BakeOff<VDIM, GLL>::∂op;
   using BakeOff<VDIM, GLL>::nodes;
   using BakeOff<VDIM, GLL>::qdata;
   using BakeOff<VDIM, GLL>::dofs;

   Diffusion(int version, int order, int side):
      BakeOff<VDIM, GLL>(order, side), ess_bdr(pmesh.bdr_attributes.Max()),
      b(&pfes),
      u_fd{U, &pfes}, Ξ_fd{Ξ, &mfes}, q_fd{Q, &qdata.space},
      u_sol{u_fd},
      q_param {q_fd},
      Ξ_q_params {Ξ_fd, q_fd},
      cg(MPI_COMM_WORLD)
   {
      static_assert(VDIM == 1 && GLL == false);

      /*{
         const int deviceId = 0;
         hipDeviceProp_t deviceProp;
         hipGetDeviceProperties(&deviceProp, deviceId);
         MFEM_GPU_CHECK(hipGetLastError());
         dbg("Device #{} {}", deviceId, deviceProp.name);
         dbg("Max smem per block: {} bytes", deviceProp.sharedMemPerBlock);
         dbg("Max smem per block: {} KiB", deviceProp.sharedMemPerBlock / 1024.0);
      }*/

      ess_bdr = 1;
      pfes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      b.AddDomainIntegrator(new DomainLFIntegrator(this->one));
      b.UseFastAssembly(true);
      b.Assemble();

      if (version < 2)
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
            dbg("\x1b[33md1d: {} q1d: {}", d1d, q1d);
            MFEM_VERIFY(d1d == D1D, "D1D mismatch: " << d1d << " != " << D1D);
            MFEM_VERIFY(q1d == Q1D, "Q1D mismatch: " << q1d << " != " << Q1D);
         }
      }
      else if (version == 2) // MF ∂fem
      {
         dbg("MF ∂fem");
         // MFEM_GPU_CHECK(hipGetLastError());
         auto solutions = std::vector{FieldDescriptor{U, &pfes}};
         auto parameters = std::vector{FieldDescriptor{Ξ, &mfes}};
         ∂op = std::make_unique<DifferentiableOperator>(solutions, parameters, pmesh);
         ∂op->SetParameters({nodes});
         auto diffusion_mf_kernel =
            [] MFEM_HOST_DEVICE (const tensor<real_t, DIM>& ∇u,
                                 const tensor<real_t, DIM, DIM>& J,
                                 const real_t& w)
         {
            auto invJ = inv(J);
            return mfem::tuple{((∇u * invJ)) * transpose(invJ) * det(J) * w};
         };
         ∂op->AddDomainIntegrator(diffusion_mf_kernel,
                                    mfem::tuple{Gradient<U>{}, Gradient<Ξ>{}, Weight{}},
                                    mfem::tuple{Gradient<U>{}},
                                    *ir);
         Operator *A_ptr;
         ∂op->FormLinearSystem(ess_tdof_list, x, b, A_ptr, X, B);
         A.Reset(A_ptr);
         // MFEM_GPU_CHECK(hipGetLastError());
      }
      else if (version == 3) // PA ∂fem
      {
         dbg("PA ∂fem");
         // MFEM_GPU_CHECK(hipGetLastError());

         auto w = Weight{};
         auto q = None<Q> {};
         auto u = None<U> {};
         auto ∇u = Gradient<U> {};
         auto ∇Ξ = Gradient<Ξ> {};
         mfem::tuple ∇u_q = {∇u, q};
         mfem::tuple u_J_w = {u, ∇Ξ, w};

         auto setup =
            [] MFEM_HOST_DEVICE(const real_t &u,
                                const tensor<real_t, DIM, DIM> &J,
                                const real_t &w)
         {
            return mfem::tuple{inv(J) * transpose(inv(J)) * det(J) * w};
         };
         DifferentiableOperator ∂Setup(u_sol, Ξ_q_params, pmesh);
         ∂Setup.SetParameters({nodes, &qdata});
         ∂Setup.AddDomainIntegrator(setup, u_J_w, mfem::tuple{q}, *ir);
         ∂Setup.Mult(x, qdata);

         auto apply =
            [] MFEM_HOST_DEVICE(const tensor<real_t, DIM> &∇u,
                                const tensor<real_t, DIM, DIM> &q)
         {
            return mfem::tuple{q * ∇u};
         };
         ∂op = std::make_unique<DifferentiableOperator>(u_sol, q_param, pmesh);
         ∂op->SetParameters({ &qdata });
         ∂op->AddDomainIntegrator(apply, ∇u_q, mfem::tuple{∇u}, *ir);

         ∂op->FormLinearSystem(ess_tdof_list, x, b, A_ptr, X, B);
         A.Reset(A_ptr);
      }
      else { MFEM_ABORT("Invalid version"); }

      cg.SetOperator(*A);
      cg.iterative_mode = false;
      if (dofs < 128 * 1024) // check
      {
         cg.SetPrintLevel(-1);
         cg.SetMaxIter(2000);
         cg.SetRelTol(1e-8);
         cg.SetAbsTol(0.0);
         cg.Mult(B, X);
         MFEM_VERIFY(cg.GetConverged(), "CG solver did not converge.");
         MFEM_DEVICE_SYNC;
         dbg("✅");
         // MFEM_GPU_CHECK(hipGetLastError());
      }
      cg.SetAbsTol(0.0);
      cg.SetRelTol(rtol);
      cg.SetMaxIter(max_it);
      cg.SetPrintLevel(print_lvl);
      MFEM_DEVICE_SYNC;
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
      ->Unit(bm::kMillisecond);
// ->Iterations(10);

BakeOff_Problem(3, Diffusion)

/// Specializations ///////////////////////////////////////////////////////////
void AddKernelSpecializations()
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
}

/// main //////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
   static mfem::MPI_Session mpi(argc, argv);

   bm::ConsoleReporter CR;
   bm::Initialize(&argc, argv);

   AddKernelSpecializations();

   // Device setup, cpu by default
   std::string device_config = "cpu";
   const auto global_context = bmi::GetGlobalContext();
   if (global_context != nullptr)
   {
      const auto device = global_context->find("device");
      if (device != global_context->end())
      {
         mfem::out << device->first << " : " << device->second << std::endl;
         device_config = device->second;
      }
   }
   dbg("device_config: {}", device_config);
   Device device(device_config.c_str());
   device_ptr = &device;
   device.Print();

   if (bm::ReportUnrecognizedArguments(argc, argv)) { return EXIT_FAILURE; }

   bm::RunSpecifiedBenchmarks(&CR);

   return EXIT_SUCCESS;
}

#endif // MFEM_USE_BENCHMARK
