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
#pragma once

#include "fem/kernels.hpp"
namespace ker = mfem::kernels::internal;

#include "../../util.hpp" // for ThreadBlocks

#include "qf_local_util.hpp"
#include "qf_local_derivative_qp.hpp"

namespace mfem::future
{

///////////////////////////////////////////////////////////////////////////////
/// Register type for one HO q-function parameter
template<typename KerOps, typename T, int rank = qf_param_shape<T>::rank>
struct ho_qreg;

template<typename KerOps, typename T>
struct ho_qreg<KerOps, T, 0>
{
   using type = typename KerOps::template val_reg_t<1>;
};

template<typename KerOps, typename T>
struct ho_qreg<KerOps, T, 1>
{
   using type = typename KerOps::template del_reg_t<1, KerOps::DIM>;
};

template<typename KerOps, typename T>
struct ho_qreg<KerOps, T, 2>
{
   static constexpr int VDIM = qf_param_shape<T>::extents[0];
   static constexpr int SDIM = qf_param_shape<T>::extents[1];
   using type = typename KerOps::template del_reg_t<VDIM, SDIM>;
};

template<typename KerOps, typename T>
using ho_qreg_t = typename ho_qreg<KerOps, T>::type;

namespace ho_qp_detail
{

/// Load one quadrature-point value
template<int DIM, typename T, typename Reg>
MFEM_HOST_DEVICE inline auto load_at(Reg &reg, int qx, int qy, int qz)
{
   static_assert(DIM == 2 || DIM == 3);
   constexpr int RNK = qf_param_shape<T>::rank;
   if constexpr (DIM == 2)
   {
      MFEM_CONTRACT_VAR(qz);
      if constexpr (RNK == 0) { return T{reg(0, qy, qx)}; }
      else if constexpr (RNK == 1)
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         constexpr int n = (e0 < DIM) ? e0 : DIM;
         T t{};
         MFEM_UNROLL(n)
         for (int dd = 0; dd < n; ++dd) { t(dd) = reg(0, dd, qy, qx); }
         return t;
      }
      else
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         constexpr int e1 = qf_param_shape<T>::extents[1];
         T t;
         MFEM_UNROLL(e0)
         for (int i = 0; i < e0; ++i)
         {
            MFEM_UNROLL(e1)
            for (int j = 0; j < e1; ++j) { t(i, j) = reg(i, j, qy, qx); }
         }
         return t;
      }
   }
   else
   {
      if constexpr (RNK == 0) { return T{reg(0, qz, qy, qx)}; }
      else if constexpr (RNK == 1)
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         constexpr int n = (e0 < DIM) ? e0 : DIM;
         T t{};
         MFEM_UNROLL(n)
         for (int dd = 0; dd < n; ++dd) { t(dd) = reg(0, dd, qz, qy, qx); }
         return t;
      }
      else
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         constexpr int e1 = qf_param_shape<T>::extents[1];
         T t;
         MFEM_UNROLL(e0)
         for (int i = 0; i < e0; ++i)
         {
            MFEM_UNROLL(e1)
            for (int j = 0; j < e1; ++j) { t(i, j) = reg(i, j, qz, qy, qx); }
         }
         return t;
      }
   }
}

template<bool tangent, typename U>
MFEM_HOST_DEVICE inline auto qp_store(const U &v)
{
   if constexpr (tangent) { return qf_store_gradient(v); }
   else { return qf_store_value(v); }
}

/// Store primal value or dual tangent at one quadrature point
template<int DIM, typename T, typename Reg, bool tangent>
MFEM_HOST_DEVICE inline void store_at(Reg &reg, int qx, int qy, int qz,
                                      const T &out)
{
   static_assert(DIM == 2 || DIM == 3);
   constexpr int RNK = qf_param_shape<T>::rank;
   if constexpr (DIM == 2)
   {
      MFEM_CONTRACT_VAR(qz);
      if constexpr (RNK == 0) { reg(0, qy, qx) = qp_store<tangent>(out); }
      else if constexpr (RNK == 1)
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         constexpr int n = (e0 < DIM) ? e0 : DIM;
         MFEM_UNROLL(n)
         for (int dd = 0; dd < n; ++dd)
         {
            reg(0, dd, qy, qx) = qp_store<tangent>(out(dd));
         }
      }
      else
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         constexpr int e1 = qf_param_shape<T>::extents[1];
         MFEM_UNROLL(e0)
         for (int i = 0; i < e0; ++i)
         {
            MFEM_UNROLL(e1)
            for (int j = 0; j < e1; ++j)
            {
               reg(i, j, qy, qx) = qp_store<tangent>(out(i, j));
            }
         }
      }
   }
   else
   {
      if constexpr (RNK == 0) { reg(0, qz, qy, qx) = qp_store<tangent>(out); }
      else if constexpr (RNK == 1)
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         constexpr int n = (e0 < DIM) ? e0 : DIM;
         MFEM_UNROLL(n)
         for (int dd = 0; dd < n; ++dd)
         {
            reg(0, dd, qz, qy, qx) = qp_store<tangent>(out(dd));
         }
      }
      else
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         constexpr int e1 = qf_param_shape<T>::extents[1];
         MFEM_UNROLL(e0)
         for (int i = 0; i < e0; ++i)
         {
            MFEM_UNROLL(e1)
            for (int j = 0; j < e1; ++j)
            {
               reg(i, j, qz, qy, qx) = qp_store<tangent>(out(i, j));
            }
         }
      }
   }
}

/// Pull primal/tangent pair into a dual q-function argument (HO registers)
template<int DIM, typename T, typename Reg>
MFEM_HOST_DEVICE inline auto pull_directional(Reg &preg, Reg &sreg,
                                              int qx, int qy, int qz,
                                              bool dependent)
{
   if constexpr (!qf_param_uses_dual_v<T>)
   {
      return load_at<DIM, T>(preg, qx, qy, qz);
   }
   else
   {
      if (!dependent) { return load_at<DIM, T>(preg, qx, qy, qz); }
      constexpr int RNK = qf_param_shape<T>::rank;
      if constexpr (DIM == 2)
      {
         MFEM_CONTRACT_VAR(qz);
         if constexpr (RNK == 0)
         {
            return T{preg(0, qy, qx), sreg(0, qy, qx)};
         }
         else if constexpr (RNK == 1)
         {
            constexpr int e0 = qf_param_shape<T>::extents[0];
            constexpr int n = (e0 < DIM) ? e0 : DIM;
            T t{};
            MFEM_UNROLL(n)
            for (int dd = 0; dd < n; ++dd)
            {
               t(dd) = {preg(0, dd, qy, qx), sreg(0, dd, qy, qx)};
            }
            return t;
         }
         else
         {
            constexpr int e0 = qf_param_shape<T>::extents[0];
            constexpr int e1 = qf_param_shape<T>::extents[1];
            T t;
            MFEM_UNROLL(e0)
            for (int i = 0; i < e0; ++i)
            {
               MFEM_UNROLL(e1)
               for (int j = 0; j < e1; ++j)
               {
                  t(i, j) = {preg(i, j, qy, qx), sreg(i, j, qy, qx)};
               }
            }
            return t;
         }
      }
      else
      {
         if constexpr (RNK == 0)
         {
            return T{preg(0, qz, qy, qx), sreg(0, qz, qy, qx)};
         }
         else if constexpr (RNK == 1)
         {
            constexpr int e0 = qf_param_shape<T>::extents[0];
            constexpr int n = (e0 < DIM) ? e0 : DIM;
            T t{};
            MFEM_UNROLL(n)
            for (int dd = 0; dd < n; ++dd)
            {
               t(dd) = {preg(0, dd, qz, qy, qx), sreg(0, dd, qz, qy, qx)};
            }
            return t;
         }
         else
         {
            constexpr int e0 = qf_param_shape<T>::extents[0];
            constexpr int e1 = qf_param_shape<T>::extents[1];
            T t;
            MFEM_UNROLL(e0)
            for (int i = 0; i < e0; ++i)
            {
               MFEM_UNROLL(e1)
               for (int j = 0; j < e1; ++j)
               {
                  t(i, j) = {preg(i, j, qz, qy, qx), sreg(i, j, qz, qy, qx)};
               }
            }
            return t;
         }
      }
   }
}

} // namespace ho_qp_detail

/// HO tensor-product kernels
template<int T_DIM, int MQ1>
struct ho_ker_backend
{
   static constexpr int DIM = T_DIM;
   static_assert(DIM == 2 || DIM == 3);

   template<int VDIM>
   using val_reg_t = std::conditional_t<(DIM == 2),
         ker::v_regs2d_t<VDIM, MQ1>, ker::v_regs3d_t<VDIM, MQ1>>;

   template<int VDIM, int SDIM>
   using del_reg_t = std::conditional_t<(DIM == 2),
         ker::vd_regs2d_t<VDIM, SDIM, MQ1>, ker::vd_regs3d_t<VDIM, SDIM, MQ1>>;

   template<typename XE_t, typename Dofs>
   static MFEM_HOST_DEVICE void load_dofs(const int e, const int d,
                                          const XE_t &XE, Dofs &dofs)
   {
      if constexpr (DIM == 2) { ker::LoadDofs2d(e, d, XE, dofs); }
      else { ker::LoadDofs3d(e, d, XE, dofs); }
   }

   template<int VDIM, int SDIM, typename XE_t, typename Dofs>
   static MFEM_HOST_DEVICE void load_grad_dofs(const int e, const int d,
                                               const XE_t &XE, Dofs &dofs)
   {
      static_assert(SDIM == DIM, "gradient spatial dim must match kernel DIM");
      load_dofs(e, d, XE, dofs);
   }

   template<typename Smem, typename Dofs, typename ArgReg>
   static MFEM_HOST_DEVICE void eval_value(const int d, const int q,
                                           Smem &s, Dofs &dofs, ArgReg &rarg)
   {
      if constexpr (DIM == 2)
      {
         ker::Eval2d<1, MQ1>(d, q, s.M, s.B, dofs, rarg);
      }
      else
      {
         ker::Eval3d(d, q, s.M, s.B, dofs, rarg);
      }
   }

   template<int VDIM, int SDIM, typename Smem, typename Dofs, typename ArgReg>
   static MFEM_HOST_DEVICE void grad(const int d, const int q,
                                     Smem &s, Dofs &dofs, ArgReg &rarg)
   {
      static_assert(SDIM == DIM, "gradient spatial dim must match kernel DIM");
      if constexpr (DIM == 2)
      {
         ker::Grad2d<VDIM, SDIM, MQ1>(d, q, s.M, s.B, s.G, dofs, rarg);
      }
      else
      {
         ker::Grad3d(d, q, s.M, s.B, s.G, dofs, rarg);
      }
   }

   template<typename Smem, typename Dofs, typename ArgReg, typename YE_t>
   static MFEM_HOST_DEVICE void write_value(const int d, const int q,
                                            const int e, Smem &s,
                                            ArgReg &rarg, Dofs &dofs,
                                            YE_t &YE)
   {
      if constexpr (DIM == 2)
      {
         ker::EvalTranspose2d<1, MQ1>(d, q, s.M, s.B, rarg, dofs);
         ker::WriteDofs2d(e, d, dofs, YE);
      }
      else
      {
         ker::EvalTranspose3d(d, q, s.M, s.B, rarg, dofs);
         ker::WriteDofs3d(e, d, dofs, YE);
      }
   }

   template<int VDIM, int SDIM, typename Smem, typename Dofs, typename ArgReg,
            typename YE_t>
   static MFEM_HOST_DEVICE void write_gradient_2d(const int d, const int q,
                                                  const int e, Smem &s,
                                                  ArgReg &rarg, Dofs &dofs,
                                                  YE_t &YE)
   {
      ker::GradTranspose2d<VDIM, SDIM, MQ1>(d, q, s.M, s.B, s.G, rarg, dofs);
      ker::WriteDofs2d<VDIM, SDIM, MQ1>(e, d, dofs, YE);
   }

   template<typename Smem, typename Dofs, typename ArgReg, typename YE_t>
   static MFEM_HOST_DEVICE void write_gradient_3d(const int d, const int q,
                                                  const int e, Smem &s,
                                                  ArgReg &rarg, Dofs &dofs,
                                                  YE_t &YE)
   {
      ker::GradTranspose3d(d, q, s.M, s.B, s.G, rarg, dofs);
      ker::WriteDofs3d(e, d, dofs, YE);
   }

   template<int VDIM, int SDIM, typename Smem, typename Dofs, typename ArgReg,
            typename YE_t>
   static MFEM_HOST_DEVICE void write_gradient(const int d, const int q,
                                               const int e, Smem &s,
                                               ArgReg &rarg, Dofs &dofs,
                                               YE_t &YE)
   {
      static_assert(SDIM == DIM, "gradient spatial dim must match kernel DIM");
      if constexpr (DIM == 2)
      {
         write_gradient_2d<VDIM, SDIM>(d, q, e, s, rarg, dofs, YE);
      }
      else
      {
         write_gradient_3d(d, q, e, s, rarg, dofs, YE);
      }
   }
};

///////////////////////////////////////////////////////////////////////////////
template<int T_DIM>
struct LocalQFHOBackend
{
   //////////////////////////////////////////////////////////////////
   static constexpr int DIM = T_DIM, MQ1 = 20;
   static constexpr bool derivative_use_enzyme = false;

   //////////////////////////////////////////////////////////////////
   static inline ThreadBlocks thread_blocks(const int q1d)
   {
      MFEM_ASSERT(q1d <= MQ1, "q1d must be less than or equal to MQ1:" << MQ1);
      return {q1d, q1d, 1};
   }

   //////////////////////////////////////////////////////////////////
   template <int Q1D> static inline
   constexpr int MAX_THREADS_PER_BLOCK() { return Q1D * Q1D; }

   //////////////////////////////////////////////////////////////////
   template<int MQ1>
   struct Shared
   {
      real_t M[MQ1][MQ1], B[MQ1][MQ1], G[MQ1][MQ1];
   };

   //////////////////////////////////////////////////////////////////
   template<int MQ1>
   using backend_t = ho_ker_backend<DIM, MQ1>;

   //////////////////////////////////////////////////////////////////
   template<typename T, int MQ1>
   using QReg = ho_qreg_t<backend_t<MQ1>, T>;

   //////////////////////////////////////////////////////////////////
   template<int MQ1, typename ArgRegT, typename XE_T>
   static inline MFEM_HOST_DEVICE
   void LoadValue(Shared<MQ1> &s,
                  const int e, const int d, const int q, const int,
                  const real_t *B, const XE_T &XE, ArgRegT &rarg)
   {
      ker::LoadMatrix(d, q, B, s.B);
      typename backend_t<MQ1>::template val_reg_t<1> dofs;
      backend_t<MQ1>::load_dofs(e, d, XE, dofs);
      backend_t<MQ1>::eval_value(d, q, s, dofs, rarg);
   }

   //////////////////////////////////////////////////////////////////
   template<int RNK, int MQ1, typename ArgRegT, typename XE_T,
            typename FieldParamT = ArgRegT>
   static inline MFEM_HOST_DEVICE
   void LoadGradient(Shared<MQ1> &s,
                     const int e, const int d, const int q, const int,
                     const real_t *B, const real_t *G,
                     const XE_T &XE, ArgRegT &rarg)
   {
      ker::LoadMatrix(d, q, B, s.B);
      ker::LoadMatrix(d, q, G, s.G);
      static_assert(RNK == 1 || RNK == 2);
      static constexpr int VDIM =
         (RNK == 1) ? 1 : qf_param_shape<FieldParamT>::extents[0];
      static constexpr int SDIM =
         (RNK == 1) ? qf_param_shape<FieldParamT>::extents[0]
         : qf_param_shape<FieldParamT>::extents[1];
      if constexpr (SDIM == DIM)
      {
         typename backend_t<MQ1>::template del_reg_t<VDIM, SDIM> dofs;
         if constexpr (RNK == 1) { backend_t<MQ1>::load_dofs(e, d, XE, dofs); }
         else { backend_t<MQ1>::template load_grad_dofs<VDIM, SDIM>(e, d, XE, dofs); }
         backend_t<MQ1>::template grad<VDIM, SDIM>(d, q, s, dofs, rarg);
      }
   }

   //////////////////////////////////////////////////////////////////
   template<typename T, int MQ1>
   static MFEM_HOST_DEVICE inline
   auto qp_pull(QReg<T, MQ1> &reg, int qx, int qy, int qz)
   {
      return ho_qp_detail::load_at<DIM, T>(reg, qx, qy, qz);
   }

   //////////////////////////////////////////////////////////////////
   template<typename T, int MQ1>
   static MFEM_HOST_DEVICE inline
   void qp_push(QReg<T, MQ1> &reg, int qx, int qy, int qz, const T &out)
   {
      ho_qp_detail::store_at<DIM, T, decltype(reg), false>(reg, qx, qy, qz, out);
   }

   //////////////////////////////////////////////////////////////////
   template<typename T, int MQ1>
   static MFEM_HOST_DEVICE inline
   auto qp_pull_directional(QReg<T, MQ1> &preg, QReg<T, MQ1> &sreg,
                            int qx, int qy, int qz, bool dependent)
   {
      return ho_qp_detail::pull_directional<DIM, T>(preg, sreg, qx, qy, qz,
                                                    dependent);
   }

   //////////////////////////////////////////////////////////////////
   template<typename T, int MQ1>
   static MFEM_HOST_DEVICE inline
   void qp_push_tangent(QReg<T, MQ1> &reg, int qx, int qy, int qz, const T &out)
   {
      ho_qp_detail::store_at<DIM, T, decltype(reg), qf_param_uses_dual_v<T>>
                                                                          (reg, qx, qy, qz, out);
   }

   //////////////////////////////////////////////////////////////////
   template<typename DT, typename XE_T>
   static MFEM_HOST_DEVICE inline
   DT identity_qp_pull_dual(bool dependent,
                            const XE_T &XP, const XE_T &XD,
                            int qx, int qy, int qz, int e)
   {
      constexpr int RNK = qf_param_shape<DT>::rank;
      if constexpr (RNK == 0)
      {
         DT t{};
         t.value = XP(0, qx, qy, qz, e);
         t.gradient = dependent ? XD(0, qx, qy, qz, e) : 0.0;
         return t;
      }
      else if constexpr (RNK == 1)
      {
         constexpr int e0 = qf_param_shape<DT>::extents[0];
         DT t{};
         MFEM_UNROLL(e0)
         for (int dd = 0; dd < e0; ++dd)
         {
            t(dd).value = XP(dd, qx, qy, qz, e);
            t(dd).gradient = dependent ? XD(dd, qx, qy, qz, e) : 0.0;
         }
         return t;
      }
      else if constexpr (RNK == 2)
      {
         constexpr int e0 = qf_param_shape<DT>::extents[0];
         constexpr int e1 = qf_param_shape<DT>::extents[1];
         DT t{};
         MFEM_UNROLL(e0)
         for (int i = 0; i < e0; ++i)
         {
            MFEM_UNROLL(e1)
            for (int j = 0; j < e1; ++j)
            {
               t(i, j).value = XP(i + e0 * j, qx, qy, qz, e);
               t(i, j).gradient = dependent ? XD(i + e0 * j, qx, qy, qz, e) : 0.0;
            }
         }
         return t;
      }
      else
      {
         static_assert(false, "Unsupported");
      }
   }

   //////////////////////////////////////////////////////////////////
   template<typename DT, typename YE_T>
   static MFEM_HOST_DEVICE inline
   void identity_qp_write_tangent(YE_T &YE, int qx, int qy, int qz, int e,
                                  const DT &qout)
   {
      constexpr int RNK = qf_param_shape<DT>::rank;
      if constexpr (qf_param_uses_dual_v<DT>)
      {
         if constexpr (RNK == 0)
         {
            YE(0, qx, qy, qz, e) = qf_store_gradient(qout);
         }
         else if constexpr (RNK == 1)
         {
            constexpr int e0 = qf_param_shape<DT>::extents[0];
            MFEM_UNROLL(e0)
            for (int dd = 0; dd < e0; ++dd)
            {
               YE(dd, qx, qy, qz, e) = qf_store_gradient(qout(dd));
            }
         }
         else if constexpr (RNK == 2)
         {
            constexpr int e0 = qf_param_shape<DT>::extents[0];
            constexpr int e1 = qf_param_shape<DT>::extents[1];
            MFEM_UNROLL(e0)
            for (int i = 0; i < e0; ++i)
            {
               MFEM_UNROLL(e1)
               for (int j = 0; j < e1; ++j)
               {
                  YE(i + e0 * j, qx, qy, qz, e) = qf_store_gradient(qout(i, j));
               }
            }
         }
         else
         {
            static_assert(false, "Unsupported");
         }
      }
   }

   //////////////////////////////////////////////////////////////////
   template<int MQ1, typename ArgRegT, typename YE_T>
   static inline MFEM_HOST_DEVICE
   void WriteValue(Shared<MQ1> &s,
                   const int e, const int d, const int q, const int,
                   const real_t *B, YE_T &YE, ArgRegT &rarg)
   {
      ker::LoadMatrix(d, q, B, s.B);
      typename backend_t<MQ1>::template val_reg_t<1> dofs;
      backend_t<MQ1>::write_value(d, q, e, s, rarg, dofs, YE);
   }

   //////////////////////////////////////////////////////////////////
   template<int RNK, int MQ1, typename ArgRegT, typename YE_T,
            typename FieldParamT = ArgRegT>
   static inline MFEM_HOST_DEVICE
   void WriteGradient(Shared<MQ1> &s,
                      const int e, const int d, const int q, const int,
                      const real_t *B, const real_t *G,
                      YE_T &YE, ArgRegT &rarg)
   {
      ker::LoadMatrix(d, q, B, s.B);
      ker::LoadMatrix(d, q, G, s.G);
      static_assert(RNK == 1 || RNK == 2);
      static constexpr int VDIM =
         (RNK == 1) ? 1 : qf_param_shape<FieldParamT>::extents[0];
      static constexpr int SDIM =
         (RNK == 1) ? qf_param_shape<FieldParamT>::extents[0]
         : qf_param_shape<FieldParamT>::extents[1];
      if constexpr (SDIM == DIM)
      {
         typename backend_t<MQ1>::template del_reg_t<VDIM, SDIM> dofs;
         backend_t<MQ1>::template write_gradient<VDIM, SDIM>
         (d, q, e, s, rarg, dofs, YE);
      }
   }

   //////////////////////////////////////////////////////////////////
   template<
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t,
      int MQ1,
      typename RArgs,
      typename SArgs,
      typename InXE,
      typename InXEd,
      typename OutYE>
   static MFEM_HOST_DEVICE inline void DerivativeEvaluateAtQP(
      const qfunc_t &qfunc,
      RArgs &rargs,
      SArgs &sargs,
      const std::array<bool, tuple_size<inputs_t>::value> &input_dep,
      const int qx, const int qy, const int qz, const int e,
      const InXE &in_XE,
      const InXEd &in_XE_dir,
      OutYE &out_YE)
   {
      derivative_evaluate_at_qp<LocalQFHOBackend<DIM>, qfunc_t, inputs_t,
                                  outputs_t, MQ1>
      (qfunc, rargs, sargs, input_dep, qx, qy, qz, e, in_XE, in_XE_dir, out_YE);
   }
};

} // namespace mfem::future
