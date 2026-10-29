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

#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>

#include "fem/dfem/backends/global_qf/prelude.hpp"
#include "fem/dfem/backends/local_qf/prelude.hpp"
#include "fem/dfem/doperator.hpp"
#include "fem/dfem/tuple.hpp"
#include "fem/qinterp/grad.hpp" // IWYU pragma: keep
#include "fem/qinterp/eval.hpp" // IWYU pragma: keep
#include "fem/quadinterpolator.hpp" // IWYU pragma: keep
#include "linalg/tensor.hpp"
#include "linalg/tensor_arrays.hpp"

using namespace mfem;

using future::DifferentiableOperator;
using future::FieldDescriptor;
using future::GlobalQFBackend;
using future::Gradient;
using future::Identity;
using future::LocalQFBackend;
using future::Value;
using future::Weight;
using future::IdentityMatrix;
using future::tensor;
using future::tensor_array;
using future::tuple;

/// info //////////////////////////////////////////////////////////////////////
void info()
{
   mfem::out << "\x1b[33m";
   mfem::out << "BPL: Laghos QUpdate bake-off problem" << std::endl;
   mfem::out << "version 0: global L-vector Mult" << std::endl;
   mfem::out << "version 1: local L-vector Mult" << std::endl;
   mfem::out << "version 2: global L-vector derivative Mult" << std::endl;
   mfem::out << "version 3: local L-vector derivative Mult" << std::endl;
   mfem::out << "version 4: MFEM PA BP3 L-vector Mult" << std::endl;
   mfem::out << "\x1b[m" << std::endl;
}

// Version ////////////////////////////////////////////////////////////////////
enum class Version
{
   GlobalMult,
   LocalMult,
   GlobalDerivativeMult,
   LocalDerivativeMult,
   BP3PAMult,
};

constexpr int version_int(Version v) noexcept
{
   return static_cast<int>(static_cast<std::underlying_type_t<Version>>(v));
}

// Custom benchmark arguments generator ///////////////////////////////////////
static void CustomArguments(bm::Benchmark *b) noexcept
{
   constexpr int MAX_NDOFS = 8 * 1024 * (mfem_use_gpu ? 1024 : 8);
   const auto orders = { 8, 7, 6, 5, 4, 3, 2, 1 };

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

// Register kernel specializations used in the smoke-test and low-order BPL
// runs. Other cases can use the MFEM fallback kernels.
static void AddKernelSpecializations()
{
#ifndef MFEM_DEBUG
   using GRAD = QuadratureInterpolator::GradKernels;
   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 3, 2, 2>::Add();

   using EVAL = QuadratureInterpolator::TensorEvalKernels;
   EVAL::Specialization<3, QVectorLayout::byVDIM, 1, 1, 2>::Opt<1>::Add();

#endif // MFEM_DEBUG
}

// Laghos QUpdate qfunction support ///////////////////////////////////////////
MFEM_HOST_DEVICE inline real_t softstep7_centered(const real_t width,
                                                  const real_t x)
{
   return 0.5 + 0.5 * std::tanh(0.5 * x / width);
}

template<int DIM> MFEM_HOST_DEVICE inline
real_t CharacteristicLength(const real_t h0, const real_t detJ)
{
   if constexpr (DIM == 1) { return h0 * detJ; }
   else if constexpr (DIM == 2) { return h0 * std::sqrt(detJ); }
   else { return h0 * std::cbrt(detJ); }
}

template<int DIM> MFEM_HOST_DEVICE inline
real_t InverseWithDet(const tensor<real_t, DIM, DIM> &A,
                      tensor<real_t, DIM, DIM> &invA)
{
   if constexpr (DIM == 1)
   {
      const real_t detA = A(0, 0);
      invA(0, 0) = 1.0 / detA;
      return detA;
   }
   else if constexpr (DIM == 2)
   {
      const real_t detA = future::det(A);
      const real_t inv_detA = 1.0 / detA;
      invA(0, 0) =  A(1, 1) * inv_detA;
      invA(0, 1) = -A(0, 1) * inv_detA;
      invA(1, 0) = -A(1, 0) * inv_detA;
      invA(1, 1) =  A(0, 0) * inv_detA;
      return detA;
   }
   else
   {
      static_assert(DIM == 3, "unsupported dimension");
      const real_t c00 = A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1);
      const real_t c01 = A(0, 2) * A(2, 1) - A(0, 1) * A(2, 2);
      const real_t c02 = A(0, 1) * A(1, 2) - A(0, 2) * A(1, 1);
      const real_t c10 = A(1, 2) * A(2, 0) - A(1, 0) * A(2, 2);
      const real_t c11 = A(0, 0) * A(2, 2) - A(0, 2) * A(2, 0);
      const real_t c12 = A(0, 2) * A(1, 0) - A(0, 0) * A(1, 2);
      const real_t c20 = A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0);
      const real_t c21 = A(0, 1) * A(2, 0) - A(0, 0) * A(2, 1);
      const real_t c22 = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
      const real_t detA = A(0, 0) * c00 + A(0, 1) * c10 + A(0, 2) * c20;
      const real_t inv_detA = 1.0 / detA;
      invA(0, 0) = c00 * inv_detA;
      invA(0, 1) = c01 * inv_detA;
      invA(0, 2) = c02 * inv_detA;
      invA(1, 0) = c10 * inv_detA;
      invA(1, 1) = c11 * inv_detA;
      invA(1, 2) = c12 * inv_detA;
      invA(2, 0) = c20 * inv_detA;
      invA(2, 1) = c21 * inv_detA;
      invA(2, 2) = c22 * inv_detA;
      return detA;
   }
}

template<int DIM> MFEM_HOST_DEVICE static inline
void QUpdateBody1(const real_t &weight,
                  const tensor<real_t, DIM, DIM> &J,
                  const real_t &rho0DetJ0w,
                  const real_t &e_quad,
                  tensor<real_t, DIM, DIM> &Jinv_out,
                  real_t &detJ_out,
                  real_t &R_out,
                  real_t &E_out)
{
   const real_t inv_weight = 1.0 / weight;
   const real_t detJ = InverseWithDet<DIM>(J, Jinv_out);
   const real_t inv_detJ = 1.0 / detJ;
   detJ_out = detJ;
   R_out = inv_weight * rho0DetJ0w * inv_detJ;
   E_out = std::fmax(0.0, e_quad);
}

template<int DIM> MFEM_HOST_DEVICE static inline
void MaterialModel(const real_t &gamma,
                   const real_t R,
                   const real_t E,
                   real_t &P_out,
                   real_t &S_out)
{
   P_out = (gamma - 1.0) * R * E;
   S_out = std::sqrt(gamma * (gamma - 1.0) * E);
}

template<int DIM,
         bool USE_VISCOSITY,
         bool USE_VORTICITY,
         bool COMPUTE_TIMESTEP>
MFEM_HOST_DEVICE static inline
void QUpdateBody2(
   const real_t h0,
   const real_t h1order,
   const real_t cfl,
   const real_t &weight,
   const tensor<real_t, DIM, DIM> &Jinv,
   const tensor<real_t, DIM, DIM> &grad_v,
   const real_t P,
   const real_t R,
   const real_t S,
   const real_t detJ,
   tensor<real_t, DIM, DIM> &stressJiT,
   real_t &dt_est)
{
   if constexpr (!COMPUTE_TIMESTEP)
   {
      auto stress = (-P) * IdentityMatrix<DIM>();
      const real_t h = CharacteristicLength<DIM>(h0, detJ);
      if constexpr (USE_VISCOSITY)
      {
         const auto dvdx = grad_v * Jinv;
         const auto sgrad_v = future::sym(dvdx);

         real_t vorticity_coeff = 1.0;
         if constexpr (USE_VORTICITY)
         {
            const real_t grad_norm = future::norm(sgrad_v);
            const real_t div_v = std::fabs(future::tr(sgrad_v));
            if (grad_norm > 0.0)
            {
               const real_t inv_grad_norm = 1.0 / grad_norm;
               vorticity_coeff = div_v * inv_grad_norm;
            }
         }
         else
         {
            MFEM_CONTRACT_VAR(vorticity_coeff);
         }

         constexpr real_t visc_q2 = 2.0;
         constexpr real_t visc_q1 = 0.5;
         const real_t div_v = future::tr(dvdx);
         const real_t psi = softstep7_centered(2.0 * S, -div_v);
         const real_t abs_delta_v_over_h = std::sqrt(div_v * div_v + 4.0 * S * S);
         const real_t visc_coeff =
            R * h * (visc_q2 * h * abs_delta_v_over_h + psi * visc_q1 * S);
         stress = stress + visc_coeff * sgrad_v;
      }

      MFEM_CONTRACT_VAR(h1order);
      MFEM_CONTRACT_VAR(cfl);
      MFEM_CONTRACT_VAR(S);
      MFEM_CONTRACT_VAR(dt_est);
      stressJiT = (weight * detJ) * (stress * future::transpose(Jinv));
      return;
   }

   const real_t h = CharacteristicLength<DIM>(h0, detJ);
   real_t visc_coeff = 0.0;
   if constexpr (USE_VISCOSITY)
   {
      const auto dvdx = grad_v * Jinv;
      const auto sgrad_v = future::sym(dvdx);

      real_t vorticity_coeff = 1.0;
      if constexpr (USE_VORTICITY)
      {
         const real_t grad_norm = future::norm(sgrad_v);
         const real_t div_v = std::fabs(future::tr(sgrad_v));
         if (grad_norm > 0.0)
         {
            const real_t inv_grad_norm = 1.0 / grad_norm;
            vorticity_coeff = div_v * inv_grad_norm;
         }
      }
      else
      {
         MFEM_CONTRACT_VAR(vorticity_coeff);
      }

      constexpr real_t visc_q2 = 2.0;
      constexpr real_t visc_q1 = 0.5;
      const real_t div_v = future::tr(dvdx);
      const real_t psi = softstep7_centered(2.0 * S, -div_v);
      const real_t abs_delta_v_over_h = std::sqrt(div_v * div_v + 4.0 * S * S);
      visc_coeff = R * h * (visc_q2 * h * abs_delta_v_over_h + psi * visc_q1 * S);
      auto stress = (-P) * IdentityMatrix<DIM>();
      stress = stress + visc_coeff * sgrad_v;
      if constexpr (COMPUTE_TIMESTEP)
      {
         stressJiT = (weight * detJ) * future::dot_transpose(stress, Jinv);
      }
      else
      {
         stressJiT = (weight * detJ) * (stress * future::transpose(Jinv));
      }
   }
   else
   {
      if constexpr (COMPUTE_TIMESTEP)
      {
         stressJiT = future::scaled_transpose(-weight * detJ * P, Jinv);
      }
      else
      {
         stressJiT = (-weight * detJ * P) * future::transpose(Jinv);
      }
   }

   if constexpr (COMPUTE_TIMESTEP)
   {
      const real_t inv_h = 1.0 / h;
      const real_t ih_min = h1order * inv_h;
      const real_t inv_R = 1.0 / R;
      const real_t irho_ih_min_sq = ih_min * ih_min * inv_R;
      const real_t idt = S * ih_min + 2.5 * visc_coeff * irho_ih_min_sq;
      if (detJ < 0.0)
      {
         dt_est = 0.0;
      }
      else if (idt > 0.0)
      {
         const real_t inv_idt = 1.0 / idt;
         dt_est = std::fmin(dt_est, cfl * inv_idt);
      }
   }
   else
   {
      MFEM_CONTRACT_VAR(h1order);
      MFEM_CONTRACT_VAR(cfl);
      MFEM_CONTRACT_VAR(S);
      MFEM_CONTRACT_VAR(dt_est);
   }

}

template <int DIM, bool USE_VISCOSITY, bool USE_VORTICITY>
struct LaghosGlobalQF
{
   const real_t h0;

   LaghosGlobalQF() = delete;
   explicit LaghosGlobalQF(const real_t h0): h0(h0) { }

   inline MFEM_HOST_DEVICE
   void operator()(tensor_array<const real_t, DIM, DIM> &dvdxi,
                   tensor_array<const real_t, DIM, DIM> &J,
                   tensor_array<const real_t> &E,
                   tensor_array<const real_t> &gamma,
                   tensor_array<const real_t> &rhoDetJw,
                   tensor_array<const real_t> &weight,
                   tensor_array<real_t, DIM, DIM> &TsJiT) const
   {
      const int NQ = static_cast<int>(dvdxi.size());
      const real_t qf_h0 = h0;

      mfem::forall<UseEnzyme>(NQ, [=] MFEM_HOST_DEVICE (int q)
      {
         const auto J_loc = J(q);
         const auto dvdxi_loc = dvdxi(q);
         tensor<real_t, DIM, DIM> Jinv_loc, sJiT;
         real_t detJ_loc, R_loc, E_loc, P_loc, S_loc;
         real_t d_dt_est = std::numeric_limits<real_t>::infinity();

         QUpdateBody1<DIM>(weight(q), J_loc, rhoDetJw(q), E(q), Jinv_loc,
                           detJ_loc, R_loc, E_loc);
         MaterialModel<DIM>(gamma(q), R_loc, E_loc, P_loc, S_loc);
         QUpdateBody2<DIM, USE_VISCOSITY, USE_VORTICITY, false>(
            qf_h0, 0.0, 0.0,
            weight(q), Jinv_loc, dvdxi_loc,
            P_loc, R_loc, S_loc, detJ_loc, sJiT, d_dt_est);
         TsJiT(q) = sJiT;
         MFEM_CONTRACT_VAR(d_dt_est);
      });
   }
};

template <int DIM, bool USE_VISCOSITY, bool USE_VORTICITY>
struct LaghosLocalQF
{
   const real_t h0;

   LaghosLocalQF() = delete;
   explicit LaghosLocalQF(const real_t h0): h0(h0) { }

   inline MFEM_HOST_DEVICE
   void operator()(const tensor<real_t, DIM, DIM> &dvdxi,
                   const tensor<real_t, DIM, DIM> &J,
                   const real_t &E,
                   const real_t &gamma,
                   const real_t &rhoDetJw,
                   const real_t &weight,
                   tensor<real_t, DIM, DIM> &TsJiT) const
   {
      tensor<real_t, DIM, DIM> Jinv_loc;
      real_t detJ_loc, R_loc, E_loc;
      real_t P_loc, S_loc;
      real_t d_dt_est = std::numeric_limits<real_t>::infinity();

      QUpdateBody1<DIM>(weight, J, rhoDetJw, E, Jinv_loc,
                        detJ_loc, R_loc, E_loc);
      MaterialModel<DIM>(gamma, R_loc, E_loc, P_loc, S_loc);
      QUpdateBody2<DIM, USE_VISCOSITY, USE_VORTICITY, false>(
         h0, 0.0, 0.0,
         weight, Jinv_loc, dvdxi,
         P_loc, R_loc, S_loc, detJ_loc, TsJiT, d_dt_est);
      MFEM_CONTRACT_VAR(d_dt_est);
   }
};

// BPL driver /////////////////////////////////////////////////////////////////
template <Version VER>
struct BPL
{
   static constexpr Version version = VER;
   static constexpr int DIM = 3;

   static constexpr int Velocity = 0;
   static constexpr int Coordinates = 1;
   static constexpr int Energy = 2;
   static constexpr int Gamma = 3;
   static constexpr int Rho0DetJ0W = 5;
   static constexpr int StressTensor = 7;
   static constexpr bool UseViscosity = true;
   static constexpr bool UseVorticity = false;
   using GlobalQF = LaghosGlobalQF<DIM, UseViscosity, UseVorticity>;
   using LocalQF = LaghosLocalQF<DIM, UseViscosity, UseVorticity>;

   const int p, energy_order, side, n, nx, ny, nz, q_order;
   Mesh smesh;
   ParMesh pmesh;
   H1_FECollection h1_fec;
   L2_FECollection l2_fec;
   L2_FECollection l0_fec;
   ParFiniteElementSpace H1, L2, L0;
   Array<int> domain_attr;
   IntegrationRules irs;
   const IntegrationRule *ir;
   const int q1d;
   QuadratureSpace qspace;
   VectorQuadratureSpace scalar_qspace, dimsqr_qspace;
   ParGridFunction x, v, e, gamma;
   QuadratureFunction rho0DetJ0w, stress;
   Array<int> Xglobal_sizes, Yglobal_sizes, Xlocal_sizes, Ylocal_sizes;
   Array<int> Xglobal_offsets, Xlocal_offsets;
   BlockVector Xglobal_b, Yglobal_b, Xlocal_b, Ylocal_b;
   MultiVector Xglobal_mv, Yglobal_mv, Xlocal_mv, Ylocal_mv;
   GlobalQF global_qf;
   LocalQF local_qf;
   DifferentiableOperator global_dop;
   DifferentiableOperator local_dop;
   std::shared_ptr<future::DerivativeOperator> dglobal_dop;
   std::shared_ptr<future::DerivativeOperator> dlocal_dop;
   Vector de;
   const HYPRE_BigInt qpts;
   HYPRE_BigInt dofs = 0;
   double mdofs = 0.0;
   double mqpts = 0.0;

   BPL(int order, int side_):
      p(order), energy_order(std::max(0, order - 1)), side(side_),
      n(std::max(1, side / p)),
      nx(n + (p * (n + 1) * p * n * p * n < side * side * side ? 1 : 0)),
      ny(n + (p * (n + 1) * p * (n + 1) * p * n < side * side * side ? 1 : 0)),
      nz(n),
      q_order(2 * p + 2),
      smesh(Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON)),
      pmesh(MPI_COMM_WORLD, smesh),
      h1_fec(p, DIM),
      l2_fec(energy_order, DIM, BasisType::Positive),
      l0_fec(0, DIM, BasisType::Positive),
      H1(&pmesh, &h1_fec, DIM),
      L2(&pmesh, &l2_fec),
      L0(&pmesh, &l0_fec),
      irs(0, Quadrature1D::GaussLegendre),
      ir(&irs.Get(pmesh.GetTypicalElementGeometry(), q_order)),
      q1d(IntRules.Get(Geometry::SEGMENT, ir->GetOrder()).GetNPoints()),
      qspace(pmesh, *ir),
      scalar_qspace(qspace, 1),
      dimsqr_qspace(qspace, DIM*DIM),
      x(&H1), v(&H1), e(&L2), gamma(&L0),
      rho0DetJ0w(scalar_qspace), stress(dimsqr_qspace),
      global_qf(1.0 / std::max(1, n)),
      local_qf(1.0 / std::max(1, n)),
      global_dop({{Velocity, &H1},
      {Coordinates, &H1},
      {Energy, &L2},
      {Gamma, &L0},
      {Rho0DetJ0W, &scalar_qspace}},
   { {StressTensor, &dimsqr_qspace}},
   pmesh),
   local_dop({{Velocity, &H1},
      {Coordinates, &H1},
      {Energy, &L2},
      {Gamma, &L0},
      {Rho0DetJ0W, &scalar_qspace}},
   {{StressTensor, &dimsqr_qspace}},
   pmesh),
   qpts(qspace.GetSize())
   {
      smesh.Clear();
      SetDomainAttributes();
      InitializeFields();
      SetupQData();
      SetupBlockVectors();
      SetupOperators();
      SetupDerivativeState();
      VerifySmokeConsistency();
      Warmup();
   }

   void SetDomainAttributes()
   {
      if (pmesh.attributes.Size() > 0)
      {
         domain_attr.SetSize(pmesh.attributes.Max());
         domain_attr = 1;
      }
   }

   void InitializeFields()
   {
      const bool use_dev = Device::Allows(Backend::DEVICE_MASK);
      x.UseDevice(use_dev);
      v.UseDevice(use_dev);
      e.UseDevice(use_dev);
      gamma.UseDevice(use_dev);
      rho0DetJ0w.UseDevice(use_dev);
      stress.UseDevice(use_dev);

      VectorFunctionCoefficient x_coeff(DIM,
                                        [] (const Vector &X, Vector &Y)
      {
         const real_t sx = std::sin(2.0 * M_PI * X(0));
         const real_t sy = std::sin(2.0 * M_PI * X(1));
         const real_t sz = std::sin(2.0 * M_PI * X(2));
         Y.SetSize(DIM);
         Y(0) = X(0) + 0.035 * sx * sy;
         Y(1) = X(1) + 0.035 * sy * sz;
         Y(2) = X(2) + 0.035 * sz * sx;
      });
      x.ProjectCoefficient(x_coeff);

      VectorFunctionCoefficient v_coeff(DIM,
                                        [] (const Vector &X, Vector &Y)
      {
         Y.SetSize(DIM);
         Y(0) = std::sin(M_PI * X(0)) * std::cos(M_PI * X(1));
         Y(1) = std::sin(M_PI * X(1)) * std::cos(M_PI * X(2));
         Y(2) = std::sin(M_PI * X(2)) * std::cos(M_PI * X(0));
      });
      v.ProjectCoefficient(v_coeff);

      ConstantCoefficient one(1.0);
      ConstantCoefficient gamma_coeff(5.0 / 3.0);
      e.ProjectCoefficient(one);
      gamma.ProjectCoefficient(gamma_coeff);
      stress = 0.0;
   }

   void SetupQData()
   {
      const int nq = qspace.GetSize();
      real_t *rho = rho0DetJ0w.Write(Device::Allows(Backend::DEVICE_MASK));
      mfem::forall(nq, [=] MFEM_HOST_DEVICE (int q)
      {
         rho[q] = 1.0;
      });
      MFEM_DEVICE_SYNC;
   }

   void SetupBlockVectors()
   {
      Xglobal_sizes.SetSize(5);
      Xglobal_sizes[0] = H1.GetVSize();
      Xglobal_sizes[1] = H1.GetVSize();
      Xglobal_sizes[2] = L2.GetVSize();
      Xglobal_sizes[3] = L0.GetVSize();
      Xglobal_sizes[4] = rho0DetJ0w.Size();
      Xglobal_offsets.SetSize(Xglobal_sizes.Size() + 1);
      Xglobal_offsets[0] = 0;
      for (int i = 0; i < Xglobal_sizes.Size(); i++)
      {
         Xglobal_offsets[i + 1] = Xglobal_sizes[i];
      }
      Xglobal_offsets.PartialSum();
      Xglobal_b.Update(Xglobal_offsets, Device::GetMemoryType());
      Xglobal_mv.MakeRef(Xglobal_b, Xglobal_sizes);

      Xlocal_sizes.SetSize(5);
      Xlocal_sizes[0] = H1.GetVSize();
      Xlocal_sizes[1] = H1.GetVSize();
      Xlocal_sizes[2] = L2.GetVSize();
      Xlocal_sizes[3] = L0.GetVSize();
      Xlocal_sizes[4] = rho0DetJ0w.Size();
      Xlocal_offsets.SetSize(Xlocal_sizes.Size() + 1);
      Xlocal_offsets[0] = 0;
      for (int i = 0; i < Xlocal_sizes.Size(); i++)
      {
         Xlocal_offsets[i + 1] = Xlocal_sizes[i];
      }
      Xlocal_offsets.PartialSum();
      Xlocal_b.Update(Xlocal_offsets, Device::GetMemoryType());
      Xlocal_mv.MakeRef(Xlocal_b, Xlocal_sizes);

      Yglobal_sizes.SetSize(1);
      Yglobal_sizes[0] = stress.Size();
      Yglobal_mv.SetSizes(Yglobal_sizes, Device::GetMemoryType());

      Ylocal_sizes.SetSize(1);
      Ylocal_sizes[0] = stress.Size();
      Ylocal_mv.SetSizes(Ylocal_sizes, Device::GetMemoryType());

      CopyVector(v, Xglobal_b.GetBlock(0));
      CopyVector(x, Xglobal_b.GetBlock(1));
      CopyVector(e, Xglobal_b.GetBlock(2));
      CopyVector(gamma, Xglobal_b.GetBlock(3));
      CopyVector(rho0DetJ0w, Xglobal_b.GetBlock(4));
      for (int i = 0; i < Yglobal_mv.NumBlocks(); i++) { Yglobal_mv[i] = 0.0; }

      CopyVector(v, Xlocal_b.GetBlock(0));
      CopyVector(x, Xlocal_b.GetBlock(1));
      CopyVector(e, Xlocal_b.GetBlock(2));
      CopyVector(gamma, Xlocal_b.GetBlock(3));
      CopyVector(rho0DetJ0w, Xlocal_b.GetBlock(4));
      for (int i = 0; i < Ylocal_mv.NumBlocks(); i++) { Ylocal_mv[i] = 0.0; }

      dofs = UsesGlobalInput() ? Xglobal_b.Size() : Xlocal_b.Size();
   }

   static constexpr bool UsesGlobalInput() noexcept
   {
      return VER == Version::GlobalMult || VER == Version::GlobalDerivativeMult;
   }

   void CopyVector(const Vector &src, Vector &dst) const
   {
      MFEM_VERIFY(src.Size() == dst.Size(), "vector size mismatch");
      const int n = src.Size();
      const bool use_dev = Device::Allows(Backend::DEVICE_MASK);
      const real_t *s = src.Read(use_dev);
      real_t *d = dst.Write(use_dev);
      mfem::forall_switch(use_dev, n, [=] MFEM_HOST_DEVICE (int i)
      {
         d[i] = s[i];
      });
   }

   void SetupOperators()
   {
      global_dop.SetMultLevel(DifferentiableOperator::MultLevel::LVECTOR);
      local_dop.SetMultLevel(DifferentiableOperator::MultLevel::LVECTOR);

      global_dop.SetQLayouts({}, {{Identity<StressTensor>{}, {2,1,0}}});
      global_dop.AddDomainIntegrator<GlobalQFBackend>(
         global_qf,
         tuple{Gradient<Velocity>{},
               Gradient<Coordinates>{},
               Value<Energy>{},
               Value<Gamma>{},
               Identity<Rho0DetJ0W>{},
               Weight{}},
         tuple{Identity<StressTensor>{}},
         *ir, domain_attr, future::Derivatives<Energy> {});
      const auto local_inputs = tuple{Gradient<Velocity>{},
                                      Gradient<Coordinates>{},
                                      Value<Energy>{},
                                      Value<Gamma>{},
                                      Identity<Rho0DetJ0W>{},
                                      Weight{}};
      const auto local_outputs = tuple{Identity<StressTensor>{}};
      AddLaghosLocalSpecializations<2>(local_inputs, local_outputs);
      AddLaghosLocalSpecializations<3>(local_inputs, local_outputs);
      AddLaghosLocalSpecializations<4>(local_inputs, local_outputs);
      AddLaghosLocalSpecializations<5>(local_inputs, local_outputs);
      AddLaghosLocalSpecializations<6>(local_inputs, local_outputs);
      AddLaghosLocalSpecializations<7>(local_inputs, local_outputs);
      AddLaghosLocalSpecializations<8>(local_inputs, local_outputs);
      AddLaghosLocalSpecializations<9>(local_inputs, local_outputs);

      local_dop.SetQLayouts({}, {{Identity<StressTensor>{}, {2,1,0}}});
      local_dop.AddDomainIntegrator<LocalQFBackend>(
         local_qf, local_inputs, local_outputs, *ir, domain_attr,
         future::Derivatives<Energy> {});
   }

   template<int Q1D, typename IT, typename OT>
   void AddLaghosLocalSpecializations(const IT &, const OT &)
   {
      future::AddLocalSpecializations<DIM, Q1D, LocalQF, IT, OT,
             future::Derivatives<Energy>>();
   }

   void SetupDerivativeState()
   {
      de.SetSize(L2.GetVSize(), Device::GetMemoryType());
      de.UseDevice(Device::Allows(Backend::DEVICE_MASK));
      CopyVector(e, de);
      const real_t de_norm = de.Norml2();
      if (de_norm > 0.0) { de *= 1.0 / de_norm; }

      dglobal_dop = global_dop.GetDerivative(Energy, Xglobal_mv, false);
      dlocal_dop = local_dop.GetDerivative(Energy, Xlocal_mv, false);
   }

   real_t RelativeLinfError(const Vector &a, const Vector &b) const
   {
      MFEM_VERIFY(a.Size() == b.Size(), "vector size mismatch");
      Vector diff(a);
      diff.UseDevice(Device::Allows(Backend::DEVICE_MASK));
      diff -= b;
      const real_t diff_norm = diff.Normlinf();
      const real_t ref_norm = std::max(a.Normlinf(), b.Normlinf());
      return diff_norm / std::max(ref_norm, real_t(1.0e-30));
   }

   void VerifySmokeConsistency()
   {
      if (p != 1 || side != 4) { return; }

      global_dop.Mult(Xglobal_mv, Yglobal_mv);
      local_dop.Mult(Xlocal_mv, Ylocal_mv);
      MFEM_DEVICE_SYNC;

      const real_t forward_error = RelativeLinfError(Yglobal_mv[0], Ylocal_mv[0]);
      MFEM_VERIFY(forward_error < 1.0e-10,
                  "BPL smoke check failed: local/global T-vector mismatch, rel_linf="
                  << forward_error);

   }

   void Warmup()
   {
      benchmark();
      MFEM_DEVICE_SYNC;
      mdofs = 0.0;
      mqpts = 0.0;
   }

   void benchmark()
   {
      if constexpr (VER == Version::GlobalMult)
      {
         global_dop.Mult(Xglobal_mv, Yglobal_mv);
      }
      else if constexpr (VER == Version::LocalMult)
      {
         local_dop.Mult(Xlocal_mv, Ylocal_mv);
      }
      else if constexpr (VER == Version::GlobalDerivativeMult)
      {
         dglobal_dop->Mult(de, Yglobal_mv);
      }
      else if constexpr (VER == Version::LocalDerivativeMult)
      {
         dlocal_dop->Mult(de, Ylocal_mv);
      }
      MFEM_DEVICE_SYNC;
      mdofs += 1.0e-6 * static_cast<double>(dofs);
      mqpts += 1.0e-6 * static_cast<double>(qpts);
   }
};

// BP3-style MFEM PA direct operator apply ////////////////////////////////////
struct BP3PAMult
{
   static constexpr Version version = Version::BP3PAMult;
   static constexpr int DIM = 3;

   const int p, side, n, nx, ny, nz, q_order;
   Mesh smesh;
   ParMesh pmesh;
   H1_FECollection fec;
   ParFiniteElementSpace fes;
   IntegrationRules irs;
   const IntegrationRule *ir;
   const int q1d;
   QuadratureSpace qspace;
   ParBilinearForm a;
   Vector X, Y;
   HYPRE_BigInt dofs = 0;
   const HYPRE_BigInt qpts;
   double mdofs = 0.0;
   double mqpts = 0.0;

   BP3PAMult(int order, int side_):
      p(order), side(side_), n(std::max(1, side / p)),
      nx(n + (p * (n + 1) * p * n * p * n < side * side * side ? 1 : 0)),
      ny(n + (p * (n + 1) * p * (n + 1) * p * n < side * side * side ? 1 : 0)),
      nz(n),
      q_order(2 * p + 2),
      smesh(Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON)),
      pmesh(MPI_COMM_WORLD, (smesh.EnsureNodes(), smesh)),
      fec(p, DIM, BasisType::GaussLobatto),
      fes(&pmesh, &fec),
      irs(0, Quadrature1D::GaussLegendre),
      ir(&irs.Get(pmesh.GetTypicalElementGeometry(), q_order)),
      q1d(IntRules.Get(Geometry::SEGMENT, ir->GetOrder()).GetNPoints()),
      qspace(pmesh, *ir),
      a(&fes),
      qpts(qspace.GetSize())
   {
      smesh.Clear();
      SetupOperator();
      Warmup();
   }

   void SetupOperator()
   {
      const bool use_dev = Device::Allows(Backend::DEVICE_MASK);

      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      a.AddDomainIntegrator(new DiffusionIntegrator(ir));
      a.Assemble();

      X.SetSize(fes.GetVSize(), Device::GetMemoryType());
      X.UseDevice(use_dev);
      X.Randomize(0x243f6a88);
      Y.SetSize(fes.GetVSize(), Device::GetMemoryType());
      Y.UseDevice(use_dev);
      Y = 0.0;
      dofs = X.Size();
   }

   void Warmup()
   {
      benchmark();
      MFEM_DEVICE_SYNC;
      mdofs = 0.0;
      mqpts = 0.0;
   }

   void benchmark()
   {
      a.Mult(X, Y);
      MFEM_DEVICE_SYNC;
      mdofs += 1.0e-6 * static_cast<double>(dofs);
      mqpts += 1.0e-6 * static_cast<double>(qpts);
   }
};

/// Benchmarks Registration ///////////////////////////////////////////////////
template <typename T>
static void Benchmark(bm::State& state) noexcept
{
   T run(state.range(0), state.range(1));
   while (state.KeepRunning()) { run.benchmark(); }
   state.counters["Dofs"] = bm::Counter(static_cast<double>(run.dofs));
   state.counters["MDof/s"] = bm::Counter(run.mdofs, bm::Counter::kIsRate);
   state.counters["Qpts"] = bm::Counter(static_cast<double>(run.qpts));
   state.counters["MQpt/s"] = bm::Counter(run.mqpts, bm::Counter::kIsRate);
   state.counters["p"] = bm::Counter(state.range(0));
   state.counters["q1d"] = bm::Counter(run.q1d);
   state.counters["version"] = bm::Counter(version_int(T::version));
}

#define REGISTER(VER) \
   BENCHMARK_TEMPLATE(Benchmark, BPL<Version::VER>) \
   ->Name("BPL_" #VER)->Apply(CustomArguments)->Unit(bm::kMillisecond)

REGISTER(GlobalMult);
REGISTER(LocalMult);
REGISTER(GlobalDerivativeMult);
REGISTER(LocalDerivativeMult);

BENCHMARK_TEMPLATE(Benchmark, BP3PAMult)
->Name("BPL_BP3PAMult")->Apply(CustomArguments)->Unit(bm::kMillisecond);

/// main //////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
   static mfem::MPI_Session mpi(argc, argv);

   bm::ConsoleReporter CR;
   bm::Initialize(&argc, argv);
   AddKernelSpecializations();
   info();

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
   device.Print();

   if (bm::ReportUnrecognizedArguments(argc, argv)) { return EXIT_FAILURE; }

   bm::RunSpecifiedBenchmarks((bm::BenchmarkReporter*)&CR);

   return EXIT_SUCCESS;
}

#endif // MFEM_USE_BENCHMARK
