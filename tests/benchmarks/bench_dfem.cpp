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
#include <type_traits>

#include "fem/qinterp/det.hpp" // IWYU pragma: keep
#include "fem/qinterp/grad.hpp" // IWYU pragma: keep
#include "fem/qinterp/grad_transpose.hpp" // IWYU pragma: keep
#include "fem/quadinterpolator.hpp" // IWYU pragma: keep
#include "fem/integ/lininteg_domain_kernels.hpp" // IWYU pragma: keep
#include "fem/integ/bilininteg_vecdiffusion_pa.hpp" // IWYU pragma: keep 

#include "fem/dfem/backends/global_qf/prelude.hpp"
using global_backend = mfem::future::GlobalQFBackend;

#include "fem/dfem/backends/local_qf/prelude.hpp"
using local_backend = mfem::future::LocalQFBackend;

#include "fem/dfem/tuple.hpp"
using future::tuple;

#include "fem/dfem/doperator.hpp"
#include "linalg/tensor.hpp"
#include "linalg/tensor_arrays.hpp"

#include "fem/kernels.hpp"
namespace ker = kernels::internal;

#if defined(__HIP__)
#include "../usr/src/array/tensor_std_array.hpp"
#endif

using namespace mfem;

using future::tensor;
using future::tensor_array;

using future::DifferentiableOperator;
using future::UniformParameterSpace;
using future::ParameterFunction;
using future::FieldDescriptor;
using future::Gradient;
using future::Value;
using future::Weight;
using future::Identity;

/// info //////////////////////////////////////////////////////////////////////
void info()
{
   mfem::out << "\x1b[33m";
   // MFEM versions
   mfem::out << "version  0: 🟢 PA std" << std::endl;
   mfem::out << "version  1: 🟠 MF HO reg" << std::endl;
   mfem::out << "version  2: 🟢 PA HO reg" << std::endl;
   // dFEM global QF versions
   mfem::out << "version  3: 🟠 MF global default" << std::endl;
   mfem::out << "version  4: 🟢 PA global default" << std::endl;
   // dFEM local QF versions
   mfem::out << "version  5: 🟠 MF local default" << std::endl;
   mfem::out << "version  6: 🟢 PA local default" << std::endl;
   mfem::out << "\x1b[m" << std::endl;
}

// Version ////////////////////////////////////////////////////////////////////
enum class Version
{
   // MFEM versions
   PA_mfem_std,
   MF_mfem_ker,
   PA_mfem_ker,
   // dFEM global QF versions
   MF_dfem_global,
   PA_dfem_global,
   // dFEM local QF versions
   MF_dfem_local,
   PA_dfem_local,
};

constexpr int version_int(Version v) noexcept
{
   return static_cast<int>(static_cast<std::underlying_type_t<Version>>(v));
}

// Custom benchmark arguments generator ///////////////////////////////////////
static void CustomArguments(bm::Benchmark *b) noexcept
{
   constexpr int MAX_NDOFS = 8 * 1024 * (mfem_use_gpu ? 1024 : 8);

   const auto orders = { 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1 };

   constexpr auto ndofs = [](int n) constexpr noexcept -> int
   {
      return (n + 1) * (n + 1) * (n + 1);
   };

   constexpr auto inc = [](int n) constexpr noexcept -> int
   {
      return n < 160 ?  4 : n < 240 ?  8 : n < 320 ? 16 : 32;
   };

   for (auto p : orders)
   {
      for (int n = 4; ndofs(n) <= MAX_NDOFS; n += inc(n))
      {
         b->Args({p, n});
      }
   }
}

// Register kernel specializations used in the benchmarks /////////////////////
static void AddKernelSpecializations()
{
#ifndef MFEM_DEBUG
   QuadratureInterpolator::DetKernels::Specialization<3, 3, 2, 2>::Add();
   QuadratureInterpolator::DetKernels::Specialization<3, 3, 2, 3>::Add();
   QuadratureInterpolator::DetKernels::Specialization<3, 3, 2, 5>::Add();
   QuadratureInterpolator::DetKernels::Specialization<3, 3, 2, 6>::Add();
   QuadratureInterpolator::DetKernels::Specialization<3, 3, 5, 5>::Add();
   // Others use too much shared data

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
#endif // MFEM_DEBUG
}

/// Globals ///////////////////////////////////////////////////////////////////
Device *device_ptr = nullptr;

/// GLOBAL Mass Q-Functions ///////////////////////////////////////////////////
template<int DIM>
struct MF_Mass_global_qf
{
   void operator()(tensor_array<const real_t> &u,
                   tensor_array<const real_t, DIM, DIM> &J,
                   tensor_array<const real_t> &weight,
                   tensor_array<real_t> &v) const
   {
      mfem::forall(v.size(), [=] MFEM_HOST_DEVICE (int q)
      {
         v(q) = weight(q) * det(J(q)) * u(q);
      });
   }
};

template<int DIM>
struct PA_Mass_Setup_global_qf
{
   void operator()(tensor_array<const real_t, DIM, DIM> &J,
                   tensor_array<const real_t> &weight,
                   tensor_array<real_t> &D) const
   {
      mfem::forall(D.size(), [=] MFEM_HOST_DEVICE (int q)
      {
         D(q) = weight(q) * det(J(q));
      });
   }
};

template<int>
struct PA_Mass_Apply_global_qf
{
   void operator()(tensor_array<const real_t> &u,
                   tensor_array<const real_t> &D,
                   tensor_array<real_t> &v) const
   {
      mfem::forall(v.size(), [=] MFEM_HOST_DEVICE (int q)
      {
         v(q) = D(q) * u(q);
      });
   }
};

/// LOCAL Mass Q-Functions ////////////////////////////////////////////////////
template<int DIM>
struct MF_Mass_local_qf
{
   MFEM_HOST_DEVICE inline
   void operator()(const real_t &u,
                   const tensor<real_t, DIM, DIM> &J,
                   const real_t &weight,
                   real_t &v) const
   {
      v =  weight * det(J) * u;
   };
};

template<int DIM>
struct PA_Mass_Setup_local_qf
{
   MFEM_HOST_DEVICE inline
   void operator()(const tensor<real_t, DIM, DIM> &J,
                   const tensor<real_t> &weight,
                   real_t &D) const
   {
      D = weight * det(J);
   }
};

template<int>
struct PA_Mass_Apply_local_qf
{
   MFEM_HOST_DEVICE inline
   void operator()(const real_t &u,
                   const real_t &D,
                   real_t &v) const
   {
      v = D * u;
   };
};

template<typename qfunction_t, int DIM>
constexpr bool mass_qf =
   std::disjunction_v<
   std::is_same<qfunction_t, MF_Mass_global_qf<DIM>>,
   std::is_same<qfunction_t, PA_Mass_Setup_global_qf<DIM>>,
   std::is_same<qfunction_t, PA_Mass_Apply_global_qf<DIM>>,
   std::is_same<qfunction_t, MF_Mass_local_qf<DIM>>,
   std::is_same<qfunction_t, PA_Mass_Setup_local_qf<DIM>>,
   std::is_same<qfunction_t, PA_Mass_Apply_local_qf<DIM>>>;

template<typename qfunction_t, int DIM, int U>
constexpr auto GradOrValue() ->
std::conditional_t<mass_qf<qfunction_t, DIM>, Value<U>, Gradient<U>>
{
   if constexpr (mass_qf<qfunction_t, DIM>) { return Value<U> {}; }
   else { return Gradient<U> {}; }
};

/// Add dFEM local QFunction action specializations ///////////////////////////
template<typename backend_t, int DIM, typename QT, typename IT, typename OT>
void AddLocalQFActionSpecializations()
{
   if constexpr (std::is_same_v<backend_t, local_backend>)
   {
      mfem::future::AddAction<DIM, 6, QT, IT, OT>();
      mfem::future::AddAction<DIM, 8, QT, IT, OT>();
   }
}

/// GLOBAL Diffusion Q-Functions //////////////////////////////////////////////
template<int DIM>
struct MF_Diffusion_global_qf
{
   void operator()(tensor_array<const real_t, DIM> &Gu,
                   tensor_array<const real_t, DIM, DIM> &J,
                   tensor_array<const real_t> &weight,
                   tensor_array<real_t, DIM> &Gv) const
   {
      mfem::forall(J.size(), [=] MFEM_HOST_DEVICE (int q)
      {
         const auto invJ = inv(J(q));
         const real_t detJ = det(J(q));
         Gv(q) = weight(q) * detJ * (transpose(invJ) * (invJ * Gu(q)));
      });
   }
};

template<int DIM>
struct PA_Diffusion_Setup_global_qf
{
   void operator()(tensor_array<const real_t, DIM, DIM> &J,
                   tensor_array<const real_t> &weight,
                   tensor_array<real_t, DIM, DIM> &D) const
   {
      mfem::forall(J.size(), [=] MFEM_HOST_DEVICE (int q)
      {
         const auto invJ = inv(J(q));
         D(q) = weight(q) * det(J(q)) * (invJ * transpose(invJ));
      });
   }
};

template<int DIM>
struct PA_Diffusion_Apply_global_qf
{
   void operator()(tensor_array<const real_t, DIM> &Gu,
                   tensor_array<const real_t, DIM, DIM> &D,
                   tensor_array<real_t, DIM> &Gv) const
   {
      mfem::forall(Gu.size(), [=] MFEM_HOST_DEVICE (int q)
      {
         Gv(q) = D(q) * Gu(q);
      });
   }
};

/// LOCAL Diffusion Q-Functions ///////////////////////////////////////////////
template<int DIM>
struct MF_Diffusion_local_qf
{
   MFEM_HOST_DEVICE inline
   void operator()(const tensor<real_t, DIM> &Gu,
                   const tensor<real_t, DIM, DIM> &J,
                   const real_t &weight,
                   tensor<real_t, DIM> &Gv) const
   {
      const auto invJ = inv(J);
      Gv =  weight * det(J) * (transpose(invJ) * (invJ * Gu));
   };
};

template<int DIM>
struct PA_Diffusion_Setup_local_qf
{
   MFEM_HOST_DEVICE inline
   void operator()(const tensor<real_t, DIM, DIM> &J,
                   const tensor<real_t> &weight,
                   tensor<real_t, DIM, DIM> &D) const
   {
      const auto invJ = inv(J);
      D = weight * det(J) * (invJ * transpose(invJ));
   }
};

template<int DIM>
struct PA_Diffusion_Apply_local_qf
{
   MFEM_HOST_DEVICE inline
   void operator()(const tensor<real_t, DIM> &Gu,
                   const tensor<real_t, DIM, DIM> &D,
                   tensor<real_t, DIM> &Gv) const
   {
      Gv = D * Gu;
   };
};

/// MF StiffnessIntegrator ///////////////////////////////////////////////////////
struct MFStiffnessIntegrator : public BilinearFormIntegrator
{
   const FiniteElementSpace *fes;
   Vector NE;
   int ne, p, d1d, q, q1d, d1n;
   Geometry::Type geom_type;
   const IntegrationRule *ir;
   const DofToQuad *maps;
   const real_t *B, *G;
   const real_t *Bn, *Gn;

public:
   MFStiffnessIntegrator()
   {
#ifndef MFEM_DEBUG
      MFStiffnessKernels::Specialization<2, 3, 2>::Add();
      MFStiffnessKernels::Specialization<3, 4, 2>::Add();
      MFStiffnessKernels::Specialization<4, 5, 2>::Add();
      MFStiffnessKernels::Specialization<5, 6, 2>::Add();
      MFStiffnessKernels::Specialization<6, 7, 2>::Add();
      MFStiffnessKernels::Specialization<7, 8, 2>::Add();
      MFStiffnessKernels::Specialization<9, 10, 2>::Add();
#endif // MFEM_DEBUG
   }

   using BilinearFormIntegrator::AssemblePA;
   void AssemblePA(const FiniteElementSpace &fespace) override
   {
      fes = &fespace;
      auto *mesh = fes->GetMesh();
      ne = mesh->GetNE();
      p = fes->GetFE(0)->GetOrder();
      d1d = p + 1;
      q = 2 * p + mesh->GetElementTransformation(0)->OrderW();
      geom_type = mesh->GetElementBaseGeometry(0);
      ir = &IntRules.Get(geom_type, q);
      q1d = IntRules.Get(Geometry::SEGMENT, ir->GetOrder()).GetNPoints();
      maps = &fes->GetFE(0)->GetDofToQuad(*ir, DofToQuad::TENSOR);
      B = maps->B.Read(), G = maps->G.Read();

      const GridFunction *nodes = (mesh->EnsureNodes(), mesh->GetNodes());
      const auto nfes = nodes->FESpace();
      assert(nfes->GetVDim() == 3);
      constexpr auto LEX = ElementDofOrdering::LEXICOGRAPHIC;
      const Operator *nRop = nfes->GetElementRestriction(LEX);
      auto nR = dynamic_cast<const ElementRestriction*>(nRop);
      NE.SetSize(nR->Height());

      d1n = nfes->GetFE(0)->GetOrder() + 1;
      nR->Mult(*nodes, (NE.UseDevice(true), NE));
      MFEM_VERIFY(NE.Size() == ne * d1n * d1n * d1n * 3, "Invalid NE size");
      const auto nfe = nfes->GetTypicalFE();
      const auto &nmaps = nfe->GetDofToQuad(*ir, DofToQuad::TENSOR);
      Bn = nmaps.B.Read(), Gn = nmaps.G.Read();
   }

   template <int T_D1D = 0, int T_Q1D = 0, int T_D1N = 0>
   static void MFStiffnessMult(const int ne,
                               const real_t *nodes_e,
                               const IntegrationRule *ir,
                               const real_t *b, const real_t *g,
                               const real_t *bn, const real_t *gn,
                               const real_t *xe, real_t *ye,
                               const int d1d, const int q1d, const int d1n)
   {
      constexpr int DIM = 3;

      const auto w_r = ir->GetWeights().Read();
      const auto W = Reshape(w_r, q1d, q1d, q1d);

      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int D1N = T_D1N ? T_D1N : d1n;

      const auto XE = Reshape(xe, D1D, D1D, D1D, 1, ne);
      const auto NE = Reshape(nodes_e, D1N, D1N, D1N, 3, ne);
      auto YE = Reshape(ye, D1D, D1D, D1D, 1, ne);

      mfem::forall_2D<T_Q1D*T_Q1D>(ne, Q1D, Q1D,
                                   [=] MFEM_HOST_DEVICE(int e)
      {
         constexpr int MD1 = T_D1D > 0 ? T_D1D : 32;
         constexpr int MQ1 = T_Q1D > 0 ? T_Q1D : 32;

         MFEM_SHARED real_t smem[MQ1][MQ1];
         MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];

         ker::vd_regs3d_t<1, DIM, MQ1> r0, r1;
         ker::vd_regs3d_t<3, DIM, MQ1> g0, g1;

         ker::LoadMatrix(D1N, Q1D, bn, sB);
         ker::LoadMatrix(D1N, Q1D, gn, sG);
         ker::LoadDofs3d(e, D1N, NE, g0);
         ker::Grad3d(D1N, Q1D, smem, sB, sG, g0, g1); // g1 = grad(NE) = ∇Ξ

         ker::LoadMatrix(D1D, Q1D, b, sB);
         ker::LoadMatrix(D1D, Q1D, g, sG);
         ker::LoadDofs3d(e, D1D, XE, r0);
         ker::Grad3d(D1D, Q1D, smem, sB, sG, r0, r1); // r1 = grad(XE) = ∇u

         for (int qz = 0; qz < Q1D; qz++)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  real_t Ju[3] =
                  {
                     r1[0][0][qz][qy][qx],
                     r1[0][1][qz][qy][qx],
                     r1[0][2][qz][qy][qx]
                  };
                  const real_t Jn[9] =
                  {
                     g1(0, 0, qz, qy, qx), g1(1, 0, qz, qy, qx), g1(2, 0, qz, qy, qx),
                     g1(0, 1, qz, qy, qx), g1(1, 1, qz, qy, qx), g1(2, 1, qz, qy, qx),
                     g1(0, 2, qz, qy, qx), g1(1, 2, qz, qy, qx), g1(2, 2, qz, qy, qx)
                  };

                  real_t Jv[3], invJn[9];
                  kernels::CalcInverse<3>(Jn, invJn);
                  kernels::Mult(3, 3, invJn, Ju, Jv);
                  kernels::MultTranspose(3, 3, invJn, Jv, Ju);

                  const real_t w = W(qx, qy, qz);
                  const real_t detJn = kernels::Det<3>(Jn);
                  r0[0][0][qz][qy][qx] = w * detJn * Ju[0];
                  r0[0][1][qz][qy][qx] = w * detJn * Ju[1];
                  r0[0][2][qz][qy][qx] = w * detJn * Ju[2];
               }
            }
         }
         // re-use sB sG
         ker::GradTranspose3d(D1D, Q1D, smem, sB, sG, r0, r1);
         ker::WriteDofs3d(e, D1D, r1, YE);
      });
   }

   using MFStiffnessKernelType = decltype(&MFStiffnessMult<>);
   MFEM_REGISTER_KERNELS(MFStiffnessKernels, MFStiffnessKernelType, (int, int,
                                                                     int));

   void AddMultPA(const Vector &xe, Vector &ye) const override
   {
      MFStiffnessKernels::Run(d1d, q1d, d1n,
                              ne, NE.Read(),
                              ir, B, G, Bn, Gn,
                              xe.Read(), ye.ReadWrite(),
                              d1d, q1d, d1n);
   }
};

template <int D1D, int Q1D, int D1N>
MFStiffnessIntegrator::MFStiffnessKernelType
MFStiffnessIntegrator::MFStiffnessKernels::Kernel()
{
   return MFStiffnessMult<D1D, Q1D, D1N>;
}

MFStiffnessIntegrator::MFStiffnessKernelType
MFStiffnessIntegrator::MFStiffnessKernels::Fallback(int d1, int q1, int n1)
{
   MFEM_CONTRACT_VAR(d1);
   MFEM_CONTRACT_VAR(q1);
   MFEM_CONTRACT_VAR(n1);
   return MFStiffnessMult;
}

/// MF MassIntegrator ///////////////////////////////////////////////////////
struct MFMassIntegrator : public BilinearFormIntegrator
{
   const FiniteElementSpace *fes;
   Vector NE;
   int ne, p, d1d, q, q1d, d1n;
   Geometry::Type geom_type;
   const IntegrationRule *ir;
   const DofToQuad *maps;
   const real_t *B, *G;
   const real_t *Bn, *Gn;

public:
   MFMassIntegrator()
   {
#ifndef MFEM_DEBUG
      MFMassKernels::Specialization<2, 3, 2>::Add();
      MFMassKernels::Specialization<3, 4, 2>::Add();
      MFMassKernels::Specialization<4, 5, 2>::Add();
      MFMassKernels::Specialization<5, 6, 2>::Add();
      MFMassKernels::Specialization<6, 7, 2>::Add();
      MFMassKernels::Specialization<7, 8, 2>::Add();
      MFMassKernels::Specialization<9, 10, 2>::Add();
#endif // MFEM_DEBUG
   }

   using BilinearFormIntegrator::AssemblePA;
   void AssemblePA(const FiniteElementSpace &fespace) override
   {
      fes = &fespace;
      auto *mesh = fes->GetMesh();
      ne = mesh->GetNE();
      p = fes->GetFE(0)->GetOrder();
      d1d = p + 1;
      q = 2 * p + mesh->GetElementTransformation(0)->OrderW();
      geom_type = mesh->GetElementBaseGeometry(0);
      ir = &IntRules.Get(geom_type, q);
      q1d = IntRules.Get(Geometry::SEGMENT, ir->GetOrder()).GetNPoints();
      maps = &fes->GetFE(0)->GetDofToQuad(*ir, DofToQuad::TENSOR);
      B = maps->B.Read(), G = maps->G.Read();

      const GridFunction *nodes = (mesh->EnsureNodes(), mesh->GetNodes());
      const auto nfes = nodes->FESpace();
      assert(nfes->GetVDim() == 3);
      constexpr auto LEX = ElementDofOrdering::LEXICOGRAPHIC;
      const Operator *nRop = nfes->GetElementRestriction(LEX);
      auto nR = dynamic_cast<const ElementRestriction*>(nRop);
      NE.SetSize(nR->Height());

      d1n = nfes->GetFE(0)->GetOrder() + 1;
      nR->Mult(*nodes, (NE.UseDevice(true), NE));
      MFEM_VERIFY(NE.Size() == ne * d1n * d1n * d1n * 3, "Invalid NE size");
      const auto nfe = nfes->GetTypicalFE();
      const auto &nmaps = nfe->GetDofToQuad(*ir, DofToQuad::TENSOR);
      Bn = nmaps.B.Read(), Gn = nmaps.G.Read();
   }

   template <int T_D1D = 0, int T_Q1D = 0, int T_D1N = 0>
   static void MFMassMult(const int ne,
                          const real_t *nodes_e,
                          const IntegrationRule *ir,
                          const real_t *b,
                          const real_t *bn, const real_t *gn,
                          const real_t *xe, real_t *ye,
                          const int d1d, const int q1d, const int d1n)
   {
      constexpr int DIM = 3;

      const auto w_r = ir->GetWeights().Read();
      const auto W = Reshape(w_r, q1d, q1d, q1d);

      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int D1N = T_D1N ? T_D1N : d1n;

      const auto XE = Reshape(xe, D1D, D1D, D1D, 1, ne);
      const auto NE = Reshape(nodes_e, D1N, D1N, D1N, 3, ne);
      auto YE = Reshape(ye, D1D, D1D, D1D, 1, ne);

      mfem::forall_2D<T_Q1D*T_Q1D>(ne, Q1D, Q1D,
                                   [=] MFEM_HOST_DEVICE(int e)
      {
         constexpr int MD1 = T_D1D > 0 ? T_D1D : 32;
         constexpr int MQ1 = T_Q1D > 0 ? T_Q1D : 32;

         MFEM_SHARED real_t smem[MQ1][MQ1];
         MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];

         ker::v_regs3d_t<1, MQ1> r0, r1;
         ker::vd_regs3d_t<3, DIM, MQ1> g0, g1;

         ker::LoadMatrix(D1N, Q1D, bn, sB);
         ker::LoadMatrix(D1N, Q1D, gn, sG);
         ker::LoadDofs3d(e, D1N, NE, g0);
         ker::Grad3d(D1N, Q1D, smem, sB, sG, g0, g1); // g1 = grad(NE) = ∇Ξ

         ker::LoadMatrix(D1D, Q1D, b, sB);
         ker::LoadDofs3d(e, D1D, XE, r0);
         ker::Eval3d(D1D, Q1D, smem, sB, r0, r1); // r1 = Eval(XE) = u

         for (int qz = 0; qz < Q1D; qz++)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  const real_t u = r1[0][qz][qy][qx];
                  const real_t Jn[9] =
                  {
                     g1(0, 0, qz, qy, qx), g1(1, 0, qz, qy, qx), g1(2, 0, qz, qy, qx),
                     g1(0, 1, qz, qy, qx), g1(1, 1, qz, qy, qx), g1(2, 1, qz, qy, qx),
                     g1(0, 2, qz, qy, qx), g1(1, 2, qz, qy, qx), g1(2, 2, qz, qy, qx)
                  };

                  const real_t w = W(qx, qy, qz);
                  const real_t detJn = kernels::Det<3>(Jn);
                  r0[0][qz][qy][qx] = w * detJn * u;
               }
            }
         }
         // re-use sB
         ker::EvalTranspose3d(D1D, Q1D, smem, sB, r0, r1);
         ker::WriteDofs3d(e, D1D, r1, YE);
      });
   }

   using MFMassKernelType = decltype(&MFMassMult<>);
   MFEM_REGISTER_KERNELS(MFMassKernels, MFMassKernelType, (int, int,
                                                           int));

   void AddMultPA(const Vector &xe, Vector &ye) const override
   {
      MFMassKernels::Run(d1d, q1d, d1n,
                         ne, NE.Read(),
                         ir, B, Bn, Gn,
                         xe.Read(), ye.ReadWrite(),
                         d1d, q1d, d1n);
   }
};

template <int D1D, int Q1D, int D1N>
MFMassIntegrator::MFMassKernelType
MFMassIntegrator::MFMassKernels::Kernel()
{
   return MFMassMult<D1D, Q1D, D1N>;
}

MFMassIntegrator::MFMassKernelType
MFMassIntegrator::MFMassKernels::Fallback(int, int, int)
{
   return MFMassMult;
}

/// PA MassIntegrator ///////////////////////////////////////////////////////
struct PAStiffnessIntegrator : public BilinearFormIntegrator
{
   const FiniteElementSpace *fes;
   const real_t *B, *G, *DX;
   int ne, d1d, q1d;
   Vector J0, dx;
   Vector &qdata;

public:
   PAStiffnessIntegrator(Vector &qdata): qdata(qdata)
   {
#ifndef MFEM_DEBUG
      StiffnessKernels::Specialization<2, 3>::Add();
      StiffnessKernels::Specialization<3, 4>::Add();
      StiffnessKernels::Specialization<4, 5>::Add();
      StiffnessKernels::Specialization<5, 6>::Add();
      StiffnessKernels::Specialization<6, 7>::Add();
      StiffnessKernels::Specialization<7, 8>::Add();
      StiffnessKernels::Specialization<9, 10>::Add();
#endif // MFEM_DEBUG
   }

   using BilinearFormIntegrator::AssemblePA;
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
PAStiffnessIntegrator::StiffnessKernelType
PAStiffnessIntegrator::StiffnessKernels::Kernel()
{
   return StiffnessMult<D1D, Q1D>;
}

PAStiffnessIntegrator::StiffnessKernelType
PAStiffnessIntegrator::StiffnessKernels::Fallback(int d1d, int q1d)
{
   MFEM_CONTRACT_VAR(d1d);
   MFEM_CONTRACT_VAR(q1d);
   return StiffnessMult;
}

/// BakeOff ///////////////////////////////////////////////////////////////////
template <int BFI, Version VER, int VDIM, bool GLL>
struct BakeOff
{
   static constexpr Version version = VER;
   static constexpr int DIM = 3;
   const int p, c, q, n, nx, ny, nz;
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
   ParGridFunction x;
   ParBilinearForm a;

   Array<int> ess_tdof_list, ess_bdr;
   ParLinearForm b;
   Vector B, X;
   OperatorPtr A;

   static constexpr int U = 0, Ξ = 1, Q = 2;
   std::unique_ptr<DifferentiableOperator> dop;
   std::unique_ptr<DifferentiableOperator> qdata_setup_dop;
   QuadratureSpace qspace;
   VectorQuadratureSpace vqspace;
   QuadratureFunction qfct;

   struct WrapOpArg1: public Operator
   {
      const std::unique_ptr<DifferentiableOperator> &dop;
      Vector &arg1;

      WrapOpArg1(const std::unique_ptr<DifferentiableOperator> &dop,
                 const int height, const int width, Vector &arg1):
         Operator(height, width), dop(dop), arg1(arg1) { }

      void Mult(const Vector &xv, Vector &yv) const override
      {
         MultiVector MX{const_cast<Vector&>(xv), arg1}, MY{yv};
         dop->Mult(MX, MY);
      }
   };
   std::unique_ptr<WrapOpArg1> wop;

   double mdofs{};

   BakeOff(int p, int side):
      p(p), c(side), q(2 * p + (GLL ? -1 : 3)), n((assert(c >= p), c / p)),
      nx(n + (p * (n + 1) * p * n * p * n < c * c * c ? 1 : 0)),
      ny(n + (p * (n + 1) * p * (n + 1) * p * n < c * c * c ? 1 : 0)), nz(n),
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
      a(&pfes),
      ess_bdr(pmesh.bdr_attributes.Max()),
      b(&pfes),
      B(pfes.GetVSize()),
      X(x),
      qspace(pmesh, *ir),
      vqspace(qspace, DIM*DIM),
      qfct(vqspace)
   {
      smesh.Clear();
      x.Randomize(0x9e3779b9);
      const int q1d = IntRules.Get(Geometry::SEGMENT, ir->GetOrder()).GetNPoints();
      assert(q1d*q1d*q1d == ir->GetNPoints());

      ess_bdr = 1;
      pfes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      // LinearForm b
      if constexpr (VDIM == 1)
      {
         b.AddDomainIntegrator(new DomainLFIntegrator(one));
      }
      else
      {
         b.AddDomainIntegrator(new VectorDomainLFIntegrator(unit_vec));
      }
      b.UseFastAssembly(true);
      b.Assemble();

      // BilinearForm a
      const int height = pfes.GetVSize(), width = pfes.GetVSize();
      const auto formLinearSystem = [&] (Vector &arg1)
      {
         Operator *A_ptr = nullptr;
         wop = std::make_unique<WrapOpArg1>(dop, height, width, arg1);
         wop->FormLinearSystem(ess_tdof_list, x, b, A_ptr, X, B);
         A.Reset(A_ptr);
      };
      // PA MFEM Setup ////////////////////////////////////
      const auto mPASetup = [&] (auto integrator)
      {
         a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         a.AddDomainIntegrator(integrator);
         a.Assemble();
         a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
      };
      // MF ∂FEM setup ////////////////////////////////////
      const auto dMFSetup = [&] (auto backend, auto qfunction)
      {
         using backend_t = decltype(backend);
         using qfunction_t = decltype(qfunction);
         const auto ifd = std::vector<FieldDescriptor> {{U, &pfes}, {Ξ, &mfes}};
         const auto ofd = std::vector<FieldDescriptor> {{U, &pfes}};
         dop = std::make_unique<DifferentiableOperator>(ifd, ofd, pmesh);
         dop->SetMultLevel(DifferentiableOperator::MultLevel::LVECTOR);
         constexpr auto GradValU = GradOrValue<qfunction_t, DIM, U>();
         dop->template AddDomainIntegrator<backend_t>(qfunction,
                                                      tuple{GradValU, Gradient<Ξ>{}, Weight{}},
                                                      tuple{GradValU},
                                                      *ir, ess_bdr);
         using QT = decltype(qfunction);
         using IT = decltype(tuple{GradValU, Gradient<Ξ>{}, Weight{}});
         using OT = decltype(tuple{GradValU});
         AddLocalQFActionSpecializations<backend_t, DIM, QT, IT, OT>();
         formLinearSystem(nodes);
      };
      // PA ∂FEM setup ////////////////////////////////////
      const auto dPASetup = [&] (auto backend, auto setup_qf, auto apply_qf)
      {
         using backend_t = decltype(backend);
         const auto ifd0 = std::vector<FieldDescriptor> {{Ξ, &mfes}};
         const auto ofd0 = std::vector<FieldDescriptor> {{Q, &vqspace}};
         qdata_setup_dop = std::make_unique<DifferentiableOperator>(ifd0, ofd0, pmesh);
         qdata_setup_dop->SetMultLevel(DifferentiableOperator::MultLevel::LVECTOR);
         qdata_setup_dop->template AddDomainIntegrator<backend_t>(
            setup_qf,
            tuple{Gradient<Ξ>{}, Weight{}},
            tuple{Identity<Q>{}},
            *ir, ess_bdr);
         using SetupQT = decltype(setup_qf);
         using SetupIT = decltype(tuple{Gradient<Ξ>{}, Weight{}});
         using SetupOT = decltype(tuple{Identity<Q>{}});
         AddLocalQFActionSpecializations<backend_t, DIM, SetupQT, SetupIT, SetupOT>();
         MultiVector N{nodes}, D{qfct};
         qdata_setup_dop->Mult(N, D);

         const auto ifd1 = std::vector<FieldDescriptor> {{U, &pfes}, {Q, &vqspace}};
         const auto ofd1 = std::vector<FieldDescriptor> {{U, &pfes}};
         dop = std::make_unique<DifferentiableOperator>(ifd1, ofd1, pmesh);
         dop->SetMultLevel(DifferentiableOperator::MultLevel::LVECTOR);
         constexpr auto GradValU = GradOrValue<decltype(apply_qf), DIM, U>();
         dop->template AddDomainIntegrator<backend_t>(apply_qf,
                                                      tuple{GradValU, Identity<Q>{}},
                                                      tuple{GradValU},
                                                      *ir, ess_bdr);
         using ApplyQT = decltype(apply_qf);
         using ApplyIT = decltype(tuple{GradValU, Identity<Q>{}});
         using ApplyOT = decltype(tuple{GradValU});
         AddLocalQFActionSpecializations<backend_t, DIM, ApplyQT, ApplyIT, ApplyOT>();
         formLinearSystem(qfct);
      };

      if constexpr (BFI == 1)
      {
         if constexpr (VER == Version::PA_mfem_std)
         {
            mPASetup(new MassIntegrator(/*ir*/));
         }
         else if constexpr (VER == Version::MF_mfem_ker)
         {
            mPASetup(new MFMassIntegrator());
         }
         /// dFEM Global versions /////////////////////////////////////////////
         else if constexpr (VER == Version::MF_dfem_global)
         {
            dMFSetup(global_backend{}, MF_Mass_global_qf<DIM> {});
         }
         else if constexpr (VER == Version::PA_dfem_global)
         {
            dPASetup(global_backend{},
                     PA_Mass_Setup_global_qf<DIM> {},
                     PA_Mass_Apply_global_qf<DIM> {});
         }
         /// dFEM Local versions //////////////////////////////////////////////
         else if constexpr (VER == Version::MF_dfem_local)
         {
            dMFSetup(local_backend{}, MF_Mass_local_qf<DIM> {});
         }
         else if constexpr (VER == Version::PA_dfem_local)
         {
            dPASetup(local_backend{},
                     PA_Mass_Setup_local_qf<DIM> {},
                     PA_Mass_Apply_local_qf<DIM> {});
         }
         else { static_assert(false, "Invalid version"); }
      }
      else if constexpr (BFI == 2)
      {
         mPASetup(new VectorMassIntegrator(one, ir));
      }
      else if constexpr (BFI == 3 || BFI == 5)
      {
         /// MFEM PA versions /////////////////////////////////////////////////
         if constexpr (VER == Version::PA_mfem_std)
         {
            mPASetup(new DiffusionIntegrator(/*ir*/));
         }
         else if constexpr (VER == Version::MF_mfem_ker)
         {
            mPASetup(new MFStiffnessIntegrator());
         }
         else if constexpr (VER == Version::PA_mfem_ker)
         {
            mPASetup(new PAStiffnessIntegrator(qfct));
         }
         /// dFEM Global versions /////////////////////////////////////////////
         else if constexpr (VER == Version::MF_dfem_global)
         {
            dMFSetup(global_backend{}, MF_Diffusion_global_qf<DIM> {});
         }
         else if constexpr (VER == Version::PA_dfem_global)
         {
            dPASetup(global_backend{},
                     PA_Diffusion_Setup_global_qf<DIM> {},
                     PA_Diffusion_Apply_global_qf<DIM> {});
         }
         /// dFEM Local versions //////////////////////////////////////////////
         else if constexpr (VER == Version::MF_dfem_local)
         {
            dMFSetup(local_backend{}, MF_Diffusion_local_qf<DIM> {});
         }
         else if constexpr (VER == Version::PA_dfem_local)
         {
            dPASetup(local_backend{},
                     PA_Diffusion_Setup_local_qf<DIM> {},
                     PA_Diffusion_Apply_local_qf<DIM> {});
         }
         else { static_assert(false, "Invalid version"); }
      }
      else if constexpr (BFI == 4 || BFI == 6)
      {
         mPASetup(new VectorDiffusionIntegrator(one, ir));
      }
      else
      {
         static_assert(BFI >= 1 && BFI <= 6, "Invalid BilinearFormIntegrator");
      }
   }

   virtual void benchmark() = 0;

   [[nodiscard]] double SumMdofs() const noexcept { return mdofs; }

   [[nodiscard]] double MDofs() const noexcept { return 1e-6 * dofs; }

};

/// Bake-off Problems (BPs) ///////////////////////////////////////////////////
template <int BFI, Version VER, int VDIM=1, bool GLL=false>
struct BP : public BakeOff<BFI, VER, VDIM, GLL>
{
   const int max_it = 32, print_lvl = -1;

   CGSolver cg;

   using base = BakeOff<BFI, VER, VDIM, GLL>;
   using base::A;
   using base::B;
   using base::X;
   using base::dofs;
   using base::mdofs;

   BP(int p, int side) noexcept: base(p, side),
      cg(MPI_COMM_WORLD)
   {
      static_assert(VDIM == 1 && GLL == false);

      cg.SetOperator(*A);
      cg.SetAbsTol(0.0);
      cg.iterative_mode = false;
      if (dofs < 128 * 1024)
      {
         cg.SetPrintLevel(3/*-1*/);
         cg.SetMaxIter(200);
         cg.SetRelTol(1e-8);
         cg.Mult(B, X);
         MFEM_VERIFY(cg.GetConverged(), "❌ CG solver did not converge.");
         // mfem::out << (cg.GetConverged() ? "✅" : "❌") << std::endl;
      }
      cg.SetRelTol(0.0);
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

/// Benchmarks Registration ///////////////////////////////////////////////////
template <typename T>
static void Benchmark(bm::State& state) noexcept
{
   T run(state.range(0), state.range(1));
   while (state.KeepRunning()) { run.benchmark(); }
   state.counters["Dofs"] = bm::Counter(run.dofs);
   state.counters["MDof/s"] = bm::Counter(run.SumMdofs(), bm::Counter::kIsRate);
   state.counters["p"] = bm::Counter(state.range(0));
   state.counters["version"] = bm::Counter(version_int(T::version));
}
#define REGISTER(PK, BFI, VER) \
   BENCHMARK_TEMPLATE(Benchmark, PK<BFI, Version::VER>) \
   ->Name(#PK #BFI "_" #VER)->Apply(CustomArguments)->Unit(bm::kMillisecond)

/// BP1 /////////////////////////////////////////////////////////////////////
REGISTER(BP, 1, PA_mfem_std);

REGISTER(BP, 1, MF_dfem_global);
REGISTER(BP, 1, PA_dfem_global);

REGISTER(BP, 1, MF_dfem_local);
REGISTER(BP, 1, PA_dfem_local);

/// BP3 /////////////////////////////////////////////////////////////////////
REGISTER(BP, 3, PA_mfem_std);
REGISTER(BP, 3, MF_mfem_ker);
REGISTER(BP, 3, PA_mfem_ker);

REGISTER(BP, 3, MF_dfem_global);
REGISTER(BP, 3, PA_dfem_global);

REGISTER(BP, 3, MF_dfem_local);
REGISTER(BP, 3, PA_dfem_local);

/// main //////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
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
   Device device(device_config.c_str());
   device_ptr = &device;
   device.Print();

   if (bm::ReportUnrecognizedArguments(argc, argv)) { return EXIT_FAILURE; }

   bm::RunSpecifiedBenchmarks((bm::BenchmarkReporter*)&CR);

   return EXIT_SUCCESS;
}

#endif // MFEM_USE_BENCHMARK
