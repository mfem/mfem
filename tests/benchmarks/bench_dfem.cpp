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
// #include <numeric>

#include "bench.hpp" // IWYU pragma: keep

#ifdef MFEM_USE_BENCHMARK

#include "mfem.hpp"
using namespace mfem;

// #include "general/forall.hpp"
// #include "linalg/kernels.hpp"
#include "fem/kernel_dispatch.hpp"

#undef NVTX_COLOR
#define NVTX_COLOR nvtx::kAquamarine
#include "general/nvtx.hpp"

namespace mfem
{

///////////////////////////////////////////////////////////////////////////////
/*inline static void StiffnessSetup(const int Q1D, const int NE, const real_t
*J0, const real_t *w, real_t *dx)
{
   constexpr int DIM = 3, DX0 = 3, DX1 = 3;
   const auto W = Reshape(w, Q1D, Q1D, Q1D);
   const auto J = Reshape(J0, DIM, DIM, Q1D, Q1D, Q1D, NE);
   auto DX = Reshape(dx, DX0, DX1, Q1D, Q1D, Q1D, NE);

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      mfem::foreach_z_thread(Q1D, [&](int qz)
      {
         mfem::foreach_y_thread(Q1D, [&](int qy)
         {
            mfem::foreach_x_thread(Q1D, [&](int qx)
            {
               const real_t w = W(qx, qy, qz);
               const real_t *Jtr = &J(0, 0, qx, qy, qz, e);
               const real_t detJ = kernels::Det<DIM>(Jtr);
               const real_t wd = w * detJ;
               real_t Jrt[DIM * DIM], A[DX0 * DX1],
                      D[DX0 * DX1] = { wd,  0.0, 0.0, 0.0, wd,
                                       0.0, 0.0, 0.0, wd
                                     };
               kernels::CalcInverse<DIM>(Jtr, Jrt);
               kernels::MultABt(DIM, DIM, DIM, D, Jrt, A);
               kernels::Mult(DIM, DIM, DIM, A, Jrt,
                             &DX(0, 0, qx, qy, qz, e));
            });
         });
      });
      MFEM_SYNC_THREAD;
   });
}*/

///////////////////////////////////////////////////////////////////////////////
/*template <int T_D1D = 0, int T_Q1D = 0>
static void StiffnessMult(const int ND, const int NE, const real_t *b,
                          const real_t *g, const int *map, const real_t *dx,
                          const real_t *XD, real_t *YD,
                          const int d1d, const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= Q1D, "");

   constexpr int DX0 = 3, DX1 = 3, DIM = 3, VDIM = 1;
   const auto DX = Reshape(dx, DX0, DX1, Q1D, Q1D, Q1D, NE);

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MQ1 = regs::SetMaxOf(T_Q1D ? T_Q1D : 32);
      constexpr int MD1 = regs::SetMaxOf(T_D1D ? T_D1D : 32);

      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];
      regs::regs5d_t<VDIM, DIM, MQ1> r0, r1;

      regs::LoadMatrix<MD1, MQ1>(D1D, Q1D, b, sB);
      regs::LoadMatrix<MD1, MQ1>(D1D, Q1D, g, sG);

      regs::ReadDofsOffset3dMap<VDIM, DIM, MQ1>(e, D1D, ND, map, XD, r0);
      regs::Grad3d<VDIM, DIM, MD1, MQ1>(D1D, Q1D, smem, sB, sG, r0, r1);

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
               kernels::Mult(DX0, DX1, dx, u, v);
               r0[0][0][qz][qy][qx] = v[0];
               r0[0][1][qz][qy][qx] = v[1];
               r0[0][2][qz][qy][qx] = v[2];
            });
         });
      }

      regs::GradTranspose3d<VDIM, DIM, MD1, MQ1>(D1D, Q1D, smem, sB, sG, r0,
r1); regs::WriteDofsOffset3d<VDIM, DIM, MQ1>(e, D1D, ND, map, r1, YD);
   });
}*/

///////////////////////////////////////////////////////////////////////////////
/*struct StiffnessIntegrator : public BilinearFormIntegrator
{
   static constexpr auto mode = DofToQuad::TENSOR;
   static constexpr auto e_ordering = ElementDofOrdering::LEXICOGRAPHIC;

   const FiniteElementSpace *fes;
   const ElementRestriction *ER;
   const DofToQuad *maps;
   int DIM, SDIM, VDIM, NDOFS, NE, NQPT, P1d, Q1d;
   int nVDIM;
   Vector J0, dx;
   const int *map;
   const real_t *B, *G, *DX;

   using StiffnessKernelType = decltype(&StiffnessMult<>);
   MFEM_REGISTER_KERNELS(StiffnessKernels, StiffnessKernelType,
                         (int, int));

public:
   StiffnessIntegrator() { action_type = ActionType::L2L; }

   ////////////////////////////////////////////////////////////////////////////
   void AddMultPA(const Vector &x, Vector &y) const override
   {
      StiffnessKernels::Run(P1d, Q1d, NDOFS, NE, B, G, map, DX, x.Read(),
                            y.ReadWrite(), P1d, Q1d);
   }

   ////////////////////////////////////////////////////////////////////////////
   void AssemblePA(const FiniteElementSpace &fespace) override
   {
      fes = &fespace;
      auto *mesh = fes->GetMesh();
      DIM = mesh->Dimension();
      SDIM = mesh->SpaceDimension();
      VDIM = fes->GetVDim();
      NDOFS = fes->GetNDofs();
      NE = mesh->GetNE();
      const auto p = fes->GetFE(0)->GetOrder();
      const auto q = 2 * p + mesh->GetElementTransformation(0)->OrderW();
      const auto type = mesh->GetElementBaseGeometry(0);
      const IntegrationRule &ir = IntRules.Get(type, q);
      NQPT = ir.GetNPoints();
      P1d = fes->GetFE(0)->GetOrder() + 1;
      Q1d = IntRules.Get(Geometry::SEGMENT, ir.GetOrder()).GetNPoints();
      maps = &fes->GetFE(0)->GetDofToQuad(ir, mode);
      assert(maps);
      ER = dynamic_cast<const ElementRestriction *>(
              fes->GetElementRestriction(e_ordering));
      assert(ER);
      map = ER->GatherMap().Read();
      const GridFunction *nodes = (mesh->EnsureNodes(), mesh->GetNodes());
      const FiniteElementSpace *nfes = nodes->FESpace();
      nVDIM = nfes->GetVDim();
      assert(nVDIM == 3);

      dx.SetSize(nVDIM * DIM * NQPT * NE, Device::GetDeviceMemoryType());
      J0.SetSize(nVDIM * DIM * NQPT * NE, Device::GetDeviceMemoryType());
      dx.UseDevice(true);
      J0.UseDevice(true);

      MFEM_VERIFY(NQPT == Q1d * Q1d * Q1d, "");

      const Operator *NR = nfes->GetElementRestriction(e_ordering);
      const QuadratureInterpolator *nqi = nfes->GetQuadratureInterpolator(ir);
      nqi->SetOutputLayout(QVectorLayout::byVDIM);
      const int nd = nfes->GetFE(0)->GetDof();

      Vector xe(nVDIM * nd * NE, Device::GetDeviceMemoryType());
      xe.UseDevice(true);
      NR->Mult(*nodes, xe);
      nqi->Derivatives(xe, J0);

      StiffnessSetup(Q1d, NE, J0.Read(), ir.GetWeights().Read(), dx.Write());

      B = maps->B.Read(), G = maps->G.Read(), DX = dx.Read();
   }
};*/

// template <int P1d, int Q1d>
// StiffnessIntegrator::StiffnessKernelType
// StiffnessIntegrator::StiffnessKernels::Kernel()
// {
//    return StiffnessMult<P1d, Q1d>;
// }

// StiffnessIntegrator::StiffnessKernelType
// StiffnessIntegrator::StiffnessKernels::Fallback(int, int)
// {
//    assert(false);
//    return StiffnessMult<>;
// }

} // namespace mfem

template <int VDIM, bool GLL>
struct BakeOff
{
   static constexpr int DIM = 3;
   const int p, c, q, n, nx, ny, nz;
   const bool check_x, check_y, check_z, checked;
   Mesh mesh;
   H1_FECollection fec;
   FiniteElementSpace fes;
   const Geometry::Type geom_type;
   IntegrationRules irs;
   const IntegrationRule *ir;
   ConstantCoefficient one;
   Vector uvec;
   VectorConstantCoefficient unit_vec;
   const int dofs;
   GridFunction x, y;
   BilinearForm a;
   double mdofs{};

   BakeOff(int p, int side):
      p(p), c(side), q(2 * p + (GLL ? -1 : 3)), n((assert(c >= p), c / p)),
      nx(n + (p * (n + 1) * p * n * p * n < c * c * c ? 1 : 0)),
      ny(n + (p * (n + 1) * p * (n + 1) * p * n < c * c * c ? 1 : 0)), nz(n),
      check_x(p * nx * p * ny * p * nz <= c * c * c),
      check_y(p * (nx + 1) * p * (ny + 1) * p * nz > c * c * c),
      check_z(p * (nx + 1) * p * (ny + 1) * p * (nz + 1) > c * c * c),
      checked((assert(check_x &&check_y &&check_z), true)),
      mesh(Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON)),
      fec(p, DIM, BasisType::GaussLobatto),
      fes(&mesh, &fec, VDIM, VDIM == 3 ? Ordering::byVDIM : Ordering::byNODES),
      geom_type(mesh.GetTypicalElementGeometry()),
      irs(0, GLL ? Quadrature1D::GaussLobatto : Quadrature1D::GaussLegendre),
      ir(&irs.Get(geom_type, q)), one(1.0), uvec(DIM),
      unit_vec((uvec = 1.0, uvec /= uvec.Norml2(), uvec)),
      dofs(fes.GetTrueVSize()), x(&fes), y(&fes), a(&fes)
   {
      x = 0.0;
   }

   virtual void benchmark() = 0;

   double SumMdofs() const { return mdofs; }

   double MDofs() const { return 1e-6 * dofs; }
};

/// Bake-off Problems (BPs) ///////////////////////////////////////////////////
template <typename BFI, int VDIM, bool GLL>
struct Problem : public BakeOff<VDIM, GLL>
{
   const real_t rtol = 0.0;
   const int max_it = 32, print_lvl = -1;

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   LinearForm b;
   OperatorPtr A;
   Vector B, X;
   CGSolver cg;

   using BakeOff<VDIM, GLL>::a;
   using BakeOff<VDIM, GLL>::ir;
   using BakeOff<VDIM, GLL>::one;
   using BakeOff<VDIM, GLL>::mesh;
   using BakeOff<VDIM, GLL>::fes;
   using BakeOff<VDIM, GLL>::x;
   using BakeOff<VDIM, GLL>::y;
   using BakeOff<VDIM, GLL>::mdofs;

   Problem(int order, int side):
      BakeOff<VDIM, GLL>(order, side), ess_bdr(mesh.bdr_attributes.Max()),
      b(&fes)
   {
      ess_bdr = 1;
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      if (VDIM == 1)
      {
         b.AddDomainIntegrator(new DomainLFIntegrator(this->one));
      }
      else
      {
         b.AddDomainIntegrator(new VectorDomainLFIntegrator(this->unit_vec));
      }
      b.UseFastAssembly(true);
      b.Assemble();

      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      a.AddDomainIntegrator(new BFI());
      a.Assemble();
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      cg.SetOperator(*A);
      cg.iterative_mode = false;
      if constexpr (true) // check
      {
         cg.SetPrintLevel(3);
         cg.SetMaxIter(1024);
         cg.SetRelTol(1e-8);
         cg.SetAbsTol(0.0);
         cg.Mult(B, X);
         MFEM_VERIFY(cg.GetConverged(), "CG solver did not converge.");
         MFEM_DEVICE_SYNC;
         mfem::out << "✅" << std::endl;
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

/// GenerateArgs
/// //////////////////////////////////////////////////////////////
#ifndef MFEM_USE_HIP
#define MAX_NDOFS 128 * 1024
#else
#define MAX_NDOFS 10 * 1024 * 1024
#endif

static void KerOrderSideArgs(bmi::Benchmark *b)
{
   const auto est = [](int c) { return (c + 1) * (c + 1) * (c + 1); };
   const auto versions = { 1 }; // only one version of the kernel yet
   for (const auto k : versions)
   {
      for (int p = 6; p >= 1; p -= 1)
      {
         for (int c = 25; est(c) <= MAX_NDOFS; c += 25)
         {
            b->Args({ k, p, c });
         }
      }
   }
}

///////////////////////////////////////////////////////////////////////////////
Device *device_ptr = nullptr;

///////////////////////////////////////////////////////////////////////////////
static void BP3Grad(bm::State &state)
{
   const int version = static_cast<int>(state.range(0));
   const auto order = static_cast<int>(state.range(1));
   const auto side = static_cast<int>(state.range(2));
   //    Problem<StiffnessIntegrator, 1, false> ker(order, side);
   Problem<DiffusionIntegrator, 1, false> ker(order, side);
   //    device_ptr->SetKernelsVersion(version);
   // if (k > 1) { device_ptr->EnableFastKernels(); }
   while (state.KeepRunning()) { ker.benchmark(); }
   bm::Counter::Flags flags = bm::Counter::kIsRate;
   state.counters["MDof/s"] = bm::Counter(ker.SumMdofs(), flags);
   state.counters["Dofs"] = bm::Counter(ker.dofs);
   state.counters["p"] = bm::Counter(order);
   state.counters["version"] = bm::Counter(version);
}
BENCHMARK(BP3Grad)
->Apply(KerOrderSideArgs)
   ->Unit(bm::kMillisecond)
   ->Iterations(10);

///////////////////////////////////////////////////////////////////////////////
#include "fem/qinterp/det.cpp"
#include "fem/qinterp/grad.hpp" // IWYU pragma: keep
void AddKernelSpecializations()
{
   dbg();
   using DET = QuadratureInterpolator::DetKernels;
   DET::Specialization<3, 3, 2, 2>::Add();
   DET::Specialization<3, 3, 2, 3>::Add();
   DET::Specialization<3, 3, 2, 5>::Add();
   DET::Specialization<3, 3, 2, 6>::Add();
   DET::Specialization<3, 3, 2, 7>::Add();
   // DET::Specialization<3,3, 2,8>::Add(); // local memory exceeds limit on AMD

   using GRAD = QuadratureInterpolator::GradKernels;
   GRAD::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 7>::Add();
   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 3, 2, 3>::Add();
   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 3, 2, 4>::Add();
   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 3, 2, 5>::Add();
   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 3, 2, 6>::Add();
   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 3, 2, 7>::Add();
   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 3, 2, 8>::Add();

   //    using Stiffness = StiffnessIntegrator::StiffnessKernels;
   //    Stiffness::Specialization<2, 3>::Add();
   //    Stiffness::Specialization<3, 4>::Add();
   //    Stiffness::Specialization<4, 5>::Add();
   //    Stiffness::Specialization<5, 6>::Add();
   //    Stiffness::Specialization<6, 7>::Add();
   //    Stiffness::Specialization<7, 8>::Add();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
   bm::ConsoleReporter CR;
   bm::Initialize(&argc, argv);

   AddKernelSpecializations();

   // Device setup, cpu by default
   std::string device_config = "cpu";
   auto global_context = bmi::GetGlobalContext();
   if (global_context != nullptr)
   {
      const auto device = global_context->find("device");
      if (device != global_context->end())
      {
         mfem::out << device->first << " : " << device->second << std::endl;
         device_config = device->second;
      }
   }
   Device device(device_config.c_str());
   device_ptr = &device;
   device.Print();

   if (bm::ReportUnrecognizedArguments(argc, argv)) { return 1; }
   bm::RunSpecifiedBenchmarks(&CR);
   return 0;
}

#endif // MFEM_USE_BENCHMARK
