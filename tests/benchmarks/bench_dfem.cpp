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

#undef MFEM_ADD_SPECIALIZATIONS

#include <memory>

#include "fem/qinterp/det.hpp" // IWYU pragma: keep
#include "fem/qinterp/grad.hpp" // IWYU pragma: keep
#include "fem/qinterp/grad_transpose.hpp" // IWYU pragma: keep
#include "fem/quadinterpolator.hpp" // IWYU pragma: keep
#include "fem/integ/lininteg_domain_kernels.hpp" // IWYU pragma: keep
#include "fem/integ/bilininteg_vecdiffusion_pa.hpp" // IWYU pragma: keep 

#include "fem/dfem/backends/global_qf/default/qf_global_prelude.hpp"
using global_default_backend = mfem::future::GlobalQFDefaultBackend;
#include "fem/dfem/backends/global_qf/devices/qf_global_devices.hpp"
using global_devices_backend = mfem::future::GlobalQFDevicesBackend;

#include "fem/dfem/backends/local_qf/default/qf_local_prelude.hpp"
using local_default_backend = mfem::future::LocalQFDefaultBackend;

#include "fem/dfem/backends/local_qf/devices/mono/qf_local_devices_mono_backend.hpp"
using local_devices_mono_backend = mfem::future::LocalQFDevicesMonoBackend;

#include "fem/dfem/backends/local_qf/devices/poly/qf_local_devices_poly_backend.hpp"
using local_devices_poly_backend = mfem::future::LocalQFDevicesPolyBackend;

#include "fem/dfem/doperator.hpp"
#include "linalg/tensor.hpp"
#include "linalg/tensor_arrays.hpp"

#include "fem/kernels.hpp"
namespace ker = kernels::internal;

#if defined(__HIP__)
#include "../usr/src/array/tensor_std_array.hpp"
#endif

using namespace mfem;

// using mfem::future::tuple;
using mfem::future::tensor;
using mfem::future::tensor_array;

using future::DifferentiableOperator;
using future::UniformParameterSpace;
using future::ParameterFunction;
using future::FieldDescriptor;
using future::Gradient;
using future::Value;
using future::Weight;
using future::Identity;

#undef NVTX_COLOR
#define NVTX_COLOR ::nvtx::kNvidia

/// info //////////////////////////////////////////////////////////////////////
void info()
{
   mfem::out << "\x1b[33m";
   mfem::out << "version 0: PA std" << std::endl;
   mfem::out << "version 1: PA new" << std::endl;
   // global QF versions
   mfem::out << "version 2: MF ∂FEM: global, default" << std::endl;
   // mfem::out << "version 3: PA ∂FEM: global, default" << std::endl;
   // mfem::out << "version 4: MF ∂FEM: global, devices" << std::endl;
   mfem::out << "version 5: PA ∂FEM: global, devices" << std::endl;
   // local QF mono/poly versions
   // ⚠️ PA local-default-multi does not support QuadratureFunction fields
   mfem::out << "version 6: MF ∂FEM: local, default & poly" << std::endl;
   mfem::out << "version 7: PA ∂FEM: local, devices & mono" << std::endl;
   mfem::out << "version 8: MF ∂FEM: local, devices & poly" << std::endl;
   mfem::out << "\x1b[m" << std::endl;
}

// Custom benchmark arguments generator ///////////////////////////////////////
static void CustomArguments(bm::Benchmark *b) noexcept
{
   constexpr int MAX_NDOFS = 8 * 1024 * (mfem_use_gpu ? 1024 : 8);

   const auto versions = { 0, 1, 2, /*3, 4,*/ 5, 6, 7, 8 };

   const auto orders = { 6, 5, 4, 3, 2, 1 };

   constexpr auto ndofs = [](int n) constexpr noexcept -> int
   {
      return (n + 1) * (n + 1) * (n + 1);
   };

   constexpr auto inc = [](int n) constexpr noexcept -> int
   {
      return n < 160 ?  4 : n < 240 ?  8 : n < 320 ? 16 : 32;
   };

   for (auto k : versions)
   {
      for (auto p : orders)
      {
         for (int n = 4; ndofs(n) <= MAX_NDOFS; n += inc(n))
         {
            b->Args({k, p, n});
         }
      }
   }
}

// Register kernel specializations used in the benchmarks /////////////////////
static void AddKernelSpecializations()
{
#ifdef MFEM_ADD_SPECIALIZATIONS
   QuadratureInterpolator::DetKernels::Specialization<3, 3, 2, 2>::Add();
   QuadratureInterpolator::DetKernels::Specialization<3, 3, 2, 3>::Add();
   QuadratureInterpolator::DetKernels::Specialization<3, 3, 2, 5>::Add();
   QuadratureInterpolator::DetKernels::Specialization<3, 3, 2, 6>::Add();
   QuadratureInterpolator::DetKernels::Specialization<3, 3, 5, 5>::Add();
   // Others use too much shared data
   // uadratureInterpolator::DetKernels::Add<3, 3, 2, 7>();

   using GRAD = QuadratureInterpolator::GradKernels;
   GRAD::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 2>::Add();
   GRAD::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 7>::Add();
   GRAD::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 8>::Add();
   GRAD::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 9>::Add();

   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 3, 2, 3>::Add();
   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 3, 2, 4>::Add();
   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 3, 2, 5>::Add();
   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 3, 2, 6>::Add();
   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 3, 2, 7>::Add();
   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 3, 2, 8>::Add();

   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 1, 2, 3>::Add();
   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 1, 4, 5>::Add();
   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 1, 5, 6>::Add();
   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 1, 6, 7>::Add();
   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 1, 7, 8>::Add();

   using GRAD_TRANSPOSE = QuadratureInterpolator::GradTransposeKernels;
   GRAD_TRANSPOSE::Specialization<3, QVectorLayout::byVDIM, false, 1,2,3>::Add();
   GRAD_TRANSPOSE::Specialization<3, QVectorLayout::byVDIM, false, 1,4,5>::Add();
   GRAD_TRANSPOSE::Specialization<3, QVectorLayout::byVDIM, false, 1,5,6>::Add();
   GRAD_TRANSPOSE::Specialization<3, QVectorLayout::byVDIM, false, 1,6,7>::Add();
   GRAD_TRANSPOSE::Specialization<3, QVectorLayout::byVDIM, false, 1,7,8>::Add();

   using LIN = DomainLFIntegrator::AssembleKernels;
   LIN::Specialization<3, 7, 7>::Add();
   LIN::Specialization<3, 6, 6>::Add();
   LIN::Specialization<3, 8, 8>::Add();

   using VDIFF = VectorDiffusionIntegrator::ApplyPAKernels;
   VDIFF::Specialization<3, 3, 3, 3>::Add();
   VDIFF::Specialization<3, 3, 4, 4>::Add();
   VDIFF::Specialization<3, 3, 5, 5>::Add();
   VDIFF::Specialization<3, 3, 6, 6>::Add();
   VDIFF::Specialization<3, 3, 7, 7>::Add();
   VDIFF::Specialization<3, 3, 8, 8>::Add();
#endif // MFEM_ADD_SPECIALIZATIONS
}

/// Globals ///////////////////////////////////////////////////////////////////
Device *device_ptr = nullptr;
static int gD1D = 0, gQ1D = 0;

/// StiffnessIntegrator ///////////////////////////////////////////////////////
struct StiffnessIntegrator : public BilinearFormIntegrator
{
   const FiniteElementSpace *fes;
   const real_t *B, *G, *DX;
   int ne, d1d, q1d;
   Vector J0, dx;
   Vector &qdata;

public:
   StiffnessIntegrator(Vector &qdata): qdata(qdata)
   {
#ifdef MFEM_ADD_SPECIALIZATIONS
      StiffnessKernels::Specialization<2, 3>::Add();
      StiffnessKernels::Specialization<3, 4>::Add();
      StiffnessKernels::Specialization<4, 5>::Add();
      StiffnessKernels::Specialization<5, 6>::Add();
      StiffnessKernels::Specialization<6, 7>::Add();
      StiffnessKernels::Specialization<7, 8>::Add();
      StiffnessKernels::Specialization<9, 10>::Add();
#endif // MFEM_ADD_SPECIALIZATIONS
   }

   void AssemblePA(const FiniteElementSpace &fespace) override
   {
      dbg();
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
      qdata = dx;
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

      mfem::forall_2D<T_Q1D*T_Q1D>(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
      {
         constexpr int MD1 = T_D1D > 0 ? kernels::internal::SetMaxOf(T_D1D) : 32;
         constexpr int MQ1 = T_Q1D > 0 ? kernels::internal::SetMaxOf(T_Q1D) : 32;

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

   using StiffnessKernelType = decltype(&StiffnessMult<0, 0>);
   MFEM_REGISTER_KERNELS(StiffnessKernels, StiffnessKernelType, (int, int));

   void AddMultPA(const Vector &x, Vector &y) const override
   {
      StiffnessKernels::Run(d1d, q1d,
                            ne, B, G, DX, x.Read(), y.ReadWrite(),
                            d1d, q1d);
   }
};

template <int D1D, int Q1D>
StiffnessIntegrator::StiffnessKernelType
StiffnessIntegrator::StiffnessKernels::Kernel()
{
   db1("D1D:{} Q1D:{}", D1D, Q1D);
   return StiffnessMult<D1D, Q1D>;
}

StiffnessIntegrator::StiffnessKernelType
StiffnessIntegrator::StiffnessKernels::Fallback(int d1d, int q1d)
{
   db1("\x1b[33mFallback d1d:{} q1d:{}", d1d, q1d);
   // MFEM_ABORT("No kernel for d1d=" << d1d << " q1d=" << q1d);
   // return nullptr;
   return StiffnessMult;
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
   ParGridFunction &nodes;
   ParFiniteElementSpace& mfes;
   ParGridFunction x, y;
   ParBilinearForm a;
   std::unique_ptr<DifferentiableOperator> dop;
   const int elem_size, total_size, d1d, q1d;
   QuadratureSpace qspace;
   QuadratureFunction qfct;
   UniformParameterSpace upspace;
   ParameterFunction pfct;

   double mdofs{};

   BakeOff(int p, int side):
      p(p), c(side), q(2 * p + (GLL ? -1 : 3)), n((assert(c >= p), c / p)),
      nx(n + (p * (n + 1) * p * n * p * n < c * c * c ? 1 : 0)),
      ny(n + (p * (n + 1) * p * (n + 1) * p * n < c * c * c ? 1 : 0)), nz(n),
      check_x(p * nx * p * ny * p * nz <= c * c * c),
      check_y(p * (nx + 1) * p * (ny + 1) * p * nz > c * c * c),
      check_z(p * (nx + 1) * p * (ny + 1) * p * (nz + 1) > c * c * c),
      checked((assert(check_x &&check_y && check_z), true)),
      smesh(Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON)),
      pmesh(MPI_COMM_WORLD, (smesh.EnsureNodes(), smesh)),
      fec(p, DIM, BasisType::GaussLobatto),
      pfes(&pmesh, &fec, VDIM),
      geom_type(pmesh.GetTypicalElementGeometry()),
      irs(0, GLL ? Quadrature1D::GaussLobatto : Quadrature1D::GaussLegendre),
      ir(&irs.Get(geom_type, q)), one(1.0), uvec(DIM),
      unit_vec((uvec = 1.0, uvec /= uvec.Norml2(), uvec)),
      dofs(pfes.GetTrueVSize()),
      nodes(*static_cast<ParGridFunction*>(pmesh.GetNodes())),
      mfes(*(nodes.ParFESpace())),
      x(&pfes),
      y(&pfes),
      a(&pfes),
      elem_size(DIM * DIM * ir->GetNPoints()),
      total_size(elem_size * pmesh.GetNE()),
      d1d(p + 1),
      q1d(IntRules.Get(Geometry::SEGMENT, ir->GetOrder()).GetNPoints()),
      qspace(pmesh, *ir),
      qfct(qspace, DIM*DIM),
      upspace(pmesh, *ir, DIM*DIM),
      pfct(upspace)
   {
      NVTX_MARK_FUNCTION;
      smesh.Clear();
      x = 0.0;

      gD1D = d1d, gQ1D = q1d;
      assert(q1d*q1d*q1d == ir->GetNPoints());
   }

   virtual void Benchmark() { MFEM_ABORT("Not implemented."); }

   [[nodiscard]] double SumMdofs() const noexcept { return mdofs; }

   [[nodiscard]] double MDofs() const noexcept { return 1e-6 * dofs; }
};

/// Q-Functions ///////////////////////////////////////////////////////////////
template<int DIM>
struct MFApply_global_qf
{
   void operator()(tensor_array<const real_t, DIM> &Gu,
                   tensor_array<const real_t, DIM, DIM> &J,
                   tensor_array<const real_t> &weight,
                   tensor_array<real_t, DIM> &Gv) const
   {
      NVTX_MARK_FUNCTION;
      mfem::forall(J.size(), [=] MFEM_HOST_DEVICE (int q)
      {
         const auto invJ = inv(J(q));
         const real_t detJ = det(J(q));
         Gv(q) = ((Gu(q) * invJ)) * transpose(invJ) * detJ * weight(q);
      });
   }
};

template<int DIM>
struct MFApply_local_qf
{
   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<real_t, DIM>& Gu,
                   const tensor<real_t, DIM, DIM>& J,
                   const real_t& w) const
   {
      auto invJ = inv(J);
      return std::tuple{((Gu * invJ)) * transpose(invJ) * det(J) * w};
   }
};

template<int DIM>
struct PASetup_global_qf
{
   void operator()(tensor_array<const real_t, DIM, DIM> &J,
                   tensor_array<const real_t> &weight,
                   tensor_array<real_t, DIM, DIM> &D) const
   {
      NVTX_MARK_FUNCTION;
      mfem::forall(J.size(), [=] MFEM_HOST_DEVICE (int q)
      {
         const auto invJ = inv(J(q));
         const real_t detJ = det(J(q));
         D(q) = invJ * transpose(invJ) * detJ * weight(q);
      });
   }
};

template<int DIM>
struct PAApply_global_qf
{
   void operator()(tensor_array<const real_t, DIM> &Gu,
                   tensor_array<const real_t, DIM, DIM> &D,
                   tensor_array<const real_t> &/*weight*/,
                   tensor_array<real_t, DIM> &Gv) const
   {
      NVTX_MARK_FUNCTION;
      mfem::forall(Gu.size(), [=] MFEM_HOST_DEVICE (int q) { Gv(q) = D(q) * Gu(q); });
   }
};

template<int DIM>
struct PASetup_local_qf
{
   MFEM_HOST_DEVICE inline
   auto operator()([[maybe_unused]] const real_t &u,
                   const tensor<real_t, DIM, DIM> &J,
                   const real_t &w)const
   {
      return std::tuple{inv(J) * transpose(inv(J)) * det(J) * w};
   }
};

template<int DIM>
struct PAApply_local_qf
{
   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<real_t, DIM> &Gu,
                   const tensor<real_t, DIM, DIM> &q) const
   {
      return std::tuple{q * Gu};
   };
};

template<int DIM>
struct MFApply_local_with_outputs_qf
{
   MFEM_HOST_DEVICE inline
   void operator()(const tensor<real_t, DIM> &Gu,
                   const tensor<real_t, DIM, DIM> &J,
                   const real_t &w,
                   tensor<real_t, DIM> &res) const
   {
      const auto invJ = inv(J);
      res = ((Gu * invJ)) * transpose(invJ) * det(J) * w;
   };
};

/// Diffusion /////////////////////////////////////////////////////////////////
template <int VDIM, bool GLL>
struct Diffusion : public BakeOff<VDIM, GLL>
{
   static constexpr int DIM = 3;
   static constexpr int U = 0, Ξ = 1, Q = 2;

   const real_t rtol = 0.0;
   const int max_it = 32, print_lvl = -1;

   const int version;
   Array<int> ess_tdof_list, ess_bdr, all_domain_attr;
   ParLinearForm b;
   FieldDescriptor u_fd, Ξ_fd, q_fd;
   std::vector<FieldDescriptor> u_sol, q_param;
   OperatorPtr A;
   Operator *A_ptr;
   Vector B, X;
   CGSolver cg;
   struct WrapOpArg1: public Operator
   {
      const std::unique_ptr<DifferentiableOperator> &dop;
      Vector &arg1;

      WrapOpArg1(const std::unique_ptr<DifferentiableOperator> &dop,
                 const int height, const int width, Vector &arg1):
         Operator(height, width), dop(dop), arg1(arg1) { }

      void Mult(const Vector &x, Vector &y) const override
      {
         MultiVector M{const_cast<Vector&>(x), arg1}, Y{y};
         dop->Mult(M, Y);
      }
   };
   std::unique_ptr<WrapOpArg1> wop;

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
   using BakeOff<VDIM, GLL>::qspace;
   using BakeOff<VDIM, GLL>::qfct;
   using BakeOff<VDIM, GLL>::upspace;
   using BakeOff<VDIM, GLL>::pfct;
   using BakeOff<VDIM, GLL>::dofs;

   Diffusion(int version, int order, int side):
      BakeOff<VDIM, GLL>(order, side),
      version(version),
      ess_bdr(pmesh.bdr_attributes.Max()),
      all_domain_attr(pmesh.bdr_attributes.Max()),
      b(&pfes),
      u_fd{U, &pfes}, Ξ_fd{Ξ, &mfes}, q_fd{Q, &upspace},
      u_sol{u_fd},
      q_param {q_fd},
      B(pfes.GetVSize()),
      X(x),
      cg(MPI_COMM_WORLD)
   {
      static_assert(VDIM == 1 && GLL == false);

      ess_bdr = 1;
      all_domain_attr = 1;
      pfes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      b.AddDomainIntegrator(new DomainLFIntegrator(this->one));
      b.UseFastAssembly(true);
      b.Assemble();

      // MF ∂FEM Global setup ///////////////////////////////////////
      const auto dMFGlobalOperatorSetup = [&] (auto backend)
      {
         using backend_t = decltype(backend);
         const auto ifs = std::vector<FieldDescriptor> {{U, &pfes}, {Ξ, &mfes}};
         const auto ofs = std::vector<FieldDescriptor> {{U, &pfes}};
         const int height = pfes.GetVSize(), width = pfes.GetVSize();
         dop = std::make_unique<DifferentiableOperator>(height, width, ifs, ofs, pmesh);
         MFApply_global_qf<DIM> mf_apply_gqf;
         dop->template AddDomainIntegrator<backend_t>(mf_apply_gqf,
                                                      std::tuple{Gradient<U>{}, Gradient<Ξ>{}, Weight{}},
                                                      std::tuple{Gradient<U>{}},
                                                      *ir, ess_bdr);
         dop->SetMultLevel(DifferentiableOperator::MultLevel::LVECTOR);
         wop = std::make_unique<WrapOpArg1>(dop, height, width, nodes);
         wop->FormLinearSystem(ess_tdof_list, x, b, A_ptr, X, B);
         A.Reset(A_ptr);
      };

      // PA ∂FEM Global setup ///////////////////////////////////////
      const auto dPAGlobalOperatorSetup = [&] (auto backend)
      {
         using backend_t = decltype(backend);
         static_assert(backend_t::is_poly, "Backend must be poly");
         const int height = pfes.GetVSize(), width = pfes.GetVSize();
         dbg("height: {} width: {}", height, width);
         dbg("\x1b[33m PA Setup operator");
         const auto i0 = std::vector<FieldDescriptor> { {Ξ, &mfes}};
         const auto o0 = std::vector<FieldDescriptor> { {Q, &qfct}};
         DifferentiableOperator dSetup(height, width, i0, o0, pmesh);
         PASetup_global_qf<DIM> pa_setup_gqf;
         dSetup.AddDomainIntegrator<backend_t>(pa_setup_gqf,
                                               std::tuple{Gradient<Ξ>{}, Weight{}},
                                               std::tuple{Identity<Q>{}},
                                               *ir, ess_bdr);
         dSetup.SetMultLevel(DifferentiableOperator::MultLevel::LVECTOR);
         MultiVector N{nodes}, D{qfct};
         dSetup.Mult(N, D);

         dbg("\x1b[33m PA Apply operator");
         const auto i1 = std::vector<FieldDescriptor> { {U, &pfes}, {Q, &qfct}};
         const auto o1 = std::vector<FieldDescriptor> { {U, &pfes}};
         dop = std::make_unique<DifferentiableOperator>(height, width, i1, o1, pmesh);
         PAApply_global_qf<DIM> pa_apply_gqf;
         dop->template AddDomainIntegrator<backend_t>(pa_apply_gqf,
                                                      std::tuple{Gradient<U>{}, Identity<Q>{}, Weight{}},
                                                      std::tuple{Gradient<U>{}},
                                                      *ir, ess_bdr);
         dop->SetMultLevel(DifferentiableOperator::MultLevel::LVECTOR);
         wop = std::make_unique<WrapOpArg1>(dop, height, width, qfct);
         wop->FormLinearSystem(ess_tdof_list, x, b, A_ptr, X, B);
         A.Reset(A_ptr);
      };

      // MF ∂FEM Local default backend setup ////////////////////////////////////////
      const auto dMFLocalDefaultOperatorSetup = [&] (auto backend)
      {
         dbg("[MF ∂fem] Local default");
         using backend_t = decltype(backend);
         static_assert(backend_t::is_poly, "Backend must be poly");
         const auto ifs = std::vector<FieldDescriptor> { {U, &pfes}, {Ξ, &mfes}};
         const auto ofs = std::vector<FieldDescriptor> { {U, &pfes}};
         const int height = pfes.GetVSize(), width = pfes.GetVSize();
         dop = std::make_unique<DifferentiableOperator>(height, width, ifs, ofs, pmesh);
         MFApply_local_with_outputs_qf<DIM> mf_apply_lqf;
         dop->template AddDomainIntegrator<backend_t>(mf_apply_lqf,
                                                      std::tuple{Gradient<U>{}, Gradient<Ξ>{}, Weight{}},
                                                      std::tuple{Gradient<U>{}},
                                                      *ir, ess_bdr);
         dop->SetMultLevel(DifferentiableOperator::MultLevel::LVECTOR);
         wop = std::make_unique<WrapOpArg1>(dop, height, width, nodes);
         wop->FormLinearSystem(ess_tdof_list, x, b, A_ptr, X, B);
         A.Reset(A_ptr);
      };

      // PA ∂FEM Local devices mono backend setup ////////////////////////////////////////
      const auto dPALocalDevicesMonoOperatorSetup = [&] (auto backend,
                                                         bool use_kernel_specializations)
      {
         using backend_t = decltype(backend);
         static_assert(backend_t::is_poly == false, "Backend must be mono");
         dbg("[PA ∂fem] Local Setup (borrowing PA setup)");
         {
            ParBilinearForm bf(&pfes);
            bf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
            bf.AddDomainIntegrator(new StiffnessIntegrator(qfct));
            bf.Assemble();
         }
         dbg("[PA ∂fem] Local Apply");
         auto Iq = Identity<Q> {};
         auto Gu = Gradient<U> {};
         std::tuple Gu_Iq = {Gu, Iq};
         PAApply_local_qf<DIM> pa_apply_qf;
         dop = std::make_unique<DifferentiableOperator>(u_sol, q_param, pmesh);
         dop->SetMultLevel(DifferentiableOperator::MultLevel::LVECTOR);
         if (use_kernel_specializations) { dop->UseKernelSpecializations(); }
         else { dbg("[PA ∂fem] NOT using kernels specialization"); }
         dop->template AddDomainIntegrator<backend_t>(pa_apply_qf, Gu_Iq, std::tuple{Gu},
                                                      *ir, ess_bdr);
         assert(qfct*qfct > 0.0);
         dop->SetParameters({ &qfct });
         dop->FormLinearSystem(ess_tdof_list, x, b, A_ptr, X, B);
         A.Reset(A_ptr);
         dbg("[PA ∂fem] done");
      };

      // MF ∂FEM Local devices poly backend setup ////////////////////////////////////////
      const auto dMFLocalDevicesPolyOperatorSetup = [&] (auto backend)
      {
         using backend_t = decltype(backend);
         static_assert(backend_t::is_poly, "Backend must be poly");
         const auto ifs = std::vector<FieldDescriptor> { {U, &pfes}, {Ξ, &mfes}};
         const auto ofs = std::vector<FieldDescriptor> { {U, &pfes}};
         const int height = pfes.GetVSize(), width = pfes.GetVSize();
         dop = std::make_unique<DifferentiableOperator>(height, width, ifs, ofs, pmesh);
         MFApply_local_with_outputs_qf<DIM> mf_apply_lqf;
         dop->template AddDomainIntegrator<backend_t>(mf_apply_lqf,
                                                      std::tuple{Gradient<U>{}, Gradient<Ξ>{}, Weight{}},
                                                      std::tuple{Gradient<U>{}},
                                                      *ir, ess_bdr);
         dop->SetMultLevel(DifferentiableOperator::MultLevel::LVECTOR);
         wop = std::make_unique<WrapOpArg1>(dop, height, width, nodes);
         wop->FormLinearSystem(ess_tdof_list, x, b, A_ptr, X, B);
         A.Reset(A_ptr);
      };

      if (version < 2) // standard, new PA regs
      {
         a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         if (version == 0) { a.AddDomainIntegrator(new DiffusionIntegrator(ir)); }
         if (version == 1) { a.AddDomainIntegrator(new StiffnessIntegrator(qfct)); }
         a.Assemble();
         a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
         if (version == 0)
         {
            BilinearFormIntegrator *bfi = a.GetDBFI()->operator[](0);
            auto *di = dynamic_cast<DiffusionIntegrator*>(bfi);
            assert(di);
            const int d1d = di->GetD1D(), q1d = di->GetQ1D();
            MFEM_VERIFY(d1d == gD1D, "D1D mismatch: " << d1d << " != " << gD1D);
            MFEM_VERIFY(q1d == gQ1D, "Q1D mismatch: " << q1d << " != " << gQ1D);
         }
      }
      else if (version == 2) // MF ∂fem-global 'default' backend
      {
         dbg("\x1b[33m MF ∂FEM-Global");
         dMFGlobalOperatorSetup(global_default_backend{});
      }
      else if (version == 3) // PA ∂fem-global 'default' backend
      {
         dbg("\x1b[33m PA ∂FEM global + default backend");
         dPAGlobalOperatorSetup(global_default_backend{});
      }
      else if (version == 4) // MF ∂fem-global 'devices' backend
      {
         dbg("\x1b[33m MF ∂FEM global + devices backend");
         dMFGlobalOperatorSetup(global_devices_backend{});
      }
      else if (version == 5) // PA ∂fem-global 'devices' backend
      {
         dbg("\x1b[33m PA ∂FEM global + devices backend");
         dPAGlobalOperatorSetup(global_devices_backend{});
      }
      else if (version == 6) // MF ∂fem-local 'default' backend
      {
         dbg("\x1b[33m MF ∂FEM Local + default backend");
         dMFLocalDefaultOperatorSetup(local_default_backend{});
      }
      else if (version == 7) // PA ∂FEM: local, devices & mono
      {
         dbg("\x1b[33m PA ∂FEM: local, devices & mono");
         dPALocalDevicesMonoOperatorSetup(local_devices_mono_backend{}, true);
      }
      else if (version == 8) // MF ∂FEM: local, devices & poly
      {
         dbg("\x1b[33m MF ∂FEM: local, devices & poly");
         dMFLocalDevicesPolyOperatorSetup(local_devices_poly_backend{});
      }
      else { MFEM_ABORT("Invalid version"); }

      cg.SetOperator(*A);
      cg.iterative_mode = false;
      cg.SetAbsTol(0.0);
      if (dofs < 128 * 1024)
      {
         cg.SetPrintLevel(3/*-1*/);
         cg.SetMaxIter(2000);
         cg.SetRelTol(1e-8);
         cg.Mult(B, X);
         MFEM_VERIFY(cg.GetConverged(), "❌ CG solver did not converge.");
         // mfem::out << "✅" << std::endl;
      }
      cg.SetRelTol(rtol);
      cg.SetMaxIter(max_it);
      cg.SetPrintLevel(print_lvl);
      Benchmark();
      mdofs = 0.0;
   }

   void Benchmark() override
   {
      cg.Mult(B, X);
      MFEM_DEVICE_SYNC;
      mdofs += this->MDofs() * cg.GetNumIterations();
   }
};

///////////////////////////////////////////////////////////////////////////////
#define BakeOff_Problem(i)                                           \
   static void BP##i(bm::State &state)                               \
   {                                                                 \
      const auto version = static_cast<int>(state.range(0));         \
      const auto order = static_cast<int>(state.range(1));           \
      const auto side = static_cast<int>(state.range(2));            \
      Diffusion<1,false> ker(version, order, side);                  \
      while (state.KeepRunning()) { ker.Benchmark(); }               \
      bm::Counter::Flags flags = bm::Counter::kIsRate;               \
      state.counters["MDof/s"] = bm::Counter(ker.SumMdofs(), flags); \
      state.counters["Dofs"] = bm::Counter(ker.dofs);                \
      state.counters["p"] = bm::Counter(order);                      \
      state.counters["version"] = bm::Counter(version);              \
   }                                                                 \
   BENCHMARK(BP##i)                                                  \
      ->Apply(CustomArguments)                                       \
      ->Unit(bm::kMillisecond)

BakeOff_Problem(3);

/// main //////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
   dbg();
   static mfem::MPI_Session mpi(argc, argv);

   bm::ConsoleReporter CR;
   bm::Initialize(&argc, argv);

   AddKernelSpecializations();
   info();

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

   bm::RunSpecifiedBenchmarks((bm::BenchmarkReporter*)&CR);

   return EXIT_SUCCESS;
}

#endif // MFEM_USE_BENCHMARK
