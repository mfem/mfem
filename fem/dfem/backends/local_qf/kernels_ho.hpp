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

#include "../../../kernels.hpp"
namespace ker = mfem::kernels::internal;

#include "../../util.hpp" // for ThreadBlocks
#include "util.hpp"

namespace mfem::future
{

// ────────────────────────────────────────────────────────────────────────────
inline constexpr int LocalQFHOBackendMQ1() { return 16; }

// ────────────────────────────────────────────────────────────────────────────
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

// ────────────────────────────────────────────────────────────────────────────
namespace hok
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
      if constexpr (RNK == 0) { return T{ reg(0, qy, qx) }; }
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
      if constexpr (RNK == 0) { return T{ reg(0, qz, qy, qx) }; }
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
   else
   {
      return qf_store_value(v);
   }
}

// Store primal value or dual tangent at one quadrature point
template<int DIM, typename T, typename Reg, bool tangent>
MFEM_HOST_DEVICE inline void
store_at(Reg &reg, int qx, int qy, int qz, const T &out)
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

// Pull primal/tangent pair into a dual q-function argument
template<int DIM, typename T, typename Reg>
MFEM_HOST_DEVICE inline auto
pull_directional(Reg &preg, Reg &sreg, int qx, int qy, int qz, bool dependent)
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
            return T{ preg(0, qy, qx), sreg(0, qy, qx) };
         }
         else if constexpr (RNK == 1)
         {
            constexpr int e0 = qf_param_shape<T>::extents[0];
            constexpr int n = (e0 < DIM) ? e0 : DIM;
            T t{};
            MFEM_UNROLL(n)
            for (int dd = 0; dd < n; ++dd)
            {
               t(dd) = { preg(0, dd, qy, qx), sreg(0, dd, qy, qx) };
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
                  t(i, j) = { preg(i, j, qy, qx), sreg(i, j, qy, qx) };
               }
            }
            return t;
         }
      }
      else
      {
         if constexpr (RNK == 0)
         {
            return T{ preg(0, qz, qy, qx), sreg(0, qz, qy, qx) };
         }
         else if constexpr (RNK == 1)
         {
            constexpr int e0 = qf_param_shape<T>::extents[0];
            constexpr int n = (e0 < DIM) ? e0 : DIM;
            T t{};
            MFEM_UNROLL(n)
            for (int dd = 0; dd < n; ++dd)
            {
               t(dd) = { preg(0, dd, qz, qy, qx), sreg(0, dd, qz, qy, qx) };
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
                  t(i, j) = { preg(i, j, qz, qy, qx), sreg(i, j, qz, qy, qx) };
               }
            }
            return t;
         }
      }
   }
}

} // namespace hok

// ────────────────────────────────────────────────────────────────────────────
/// HO tensor-product kernels
template<int T_DIM, int MQ1>
struct ho_ker_backend
{
   static constexpr int DIM = T_DIM;
   static_assert(DIM == 2 || DIM == 3);

   template<int VDIM>
   using val_reg_t = std::conditional_t<(DIM == 2),
         ker::v_regs2d_t<VDIM, MQ1>,
         ker::v_regs3d_t<VDIM, MQ1>>;

   template<int VDIM, int SDIM>
   using del_reg_t = std::conditional_t<(DIM == 2),
         ker::vd_regs2d_t<VDIM, SDIM, MQ1>,
         ker::vd_regs3d_t<VDIM, SDIM, MQ1>>;

   struct Shared
   {
      real_t M[MQ1][MQ1], B[MQ1][MQ1], G[MQ1][MQ1];
   };

   template<typename XE_t, typename Dofs>
   static MFEM_HOST_DEVICE void
   load_dofs(const int e, const int d, const XE_t &XE, Dofs &dofs)
   {
      if constexpr (DIM == 2) { ker::LoadDofs2d(e, d, XE, dofs); }
      else
      {
         ker::LoadDofs3d(e, d, XE, dofs);
      }
   }

   template<int VDIM, int SDIM, typename XE_t, typename Dofs>
   static MFEM_HOST_DEVICE void
   load_grad_dofs(const int e, const int d, const XE_t &XE, Dofs &dofs)
   {
      static_assert(SDIM == DIM, "gradient spatial dim must match kernel DIM");
      load_dofs(e, d, XE, dofs);
   }

   template<typename Smem, typename Dofs, typename ArgReg>
   static MFEM_HOST_DEVICE void
   eval_value(const int d, const int q, Smem &s, Dofs &dofs, ArgReg &rarg)
   {
      if constexpr (DIM == 2) { ker::Eval2d(d, q, s.M, s.B, dofs, rarg); }
      else
      {
         ker::Eval3d(d, q, s.M, s.B, dofs, rarg);
      }
   }

   template<int VDIM, int SDIM, typename Smem, typename Dofs, typename ArgReg>
   static MFEM_HOST_DEVICE void
   grad(const int d, const int q, Smem &s, Dofs &dofs, ArgReg &rarg)
   {
      static_assert(SDIM == DIM, "gradient spatial dim must match kernel DIM");
      if constexpr (DIM == 2) { ker::Grad2d(d, q, s.M, s.B, s.G, dofs, rarg); }
      else
      {
         ker::Grad3d(d, q, s.M, s.B, s.G, dofs, rarg);
      }
   }

   template<typename Smem, typename Dofs, typename ArgReg, typename YE_t>
   static MFEM_HOST_DEVICE void write_value(const int d,
                                            const int q,
                                            const int e,
                                            Smem &s,
                                            ArgReg &rarg,
                                            Dofs &dofs,
                                            YE_t &YE)
   {
      if constexpr (DIM == 2)
      {
         ker::EvalTranspose2d(d, q, s.M, s.B, rarg, dofs);
         ker::WriteDofs2d(e, d, dofs, YE);
      }
      else
      {
         ker::EvalTranspose3d(d, q, s.M, s.B, rarg, dofs);
         ker::WriteDofs3d(e, d, dofs, YE);
      }
   }

   template<typename Smem, typename Dofs, typename ArgReg, typename YE_t>
   static MFEM_HOST_DEVICE void write_gradient_2d(const int d,
                                                  const int q,
                                                  const int e,
                                                  Smem &s,
                                                  ArgReg &rarg,
                                                  Dofs &dofs,
                                                  YE_t &YE)
   {
      ker::GradTranspose2d(d, q, s.M, s.B, s.G, rarg, dofs);
      ker::WriteDofs2d(e, d, dofs, YE);
   }

   template<typename Smem, typename Dofs, typename ArgReg, typename YE_t>
   static MFEM_HOST_DEVICE void write_gradient_3d(const int d,
                                                  const int q,
                                                  const int e,
                                                  Smem &s,
                                                  ArgReg &rarg,
                                                  Dofs &dofs,
                                                  YE_t &YE)
   {
      ker::GradTranspose3d(d, q, s.M, s.B, s.G, rarg, dofs);
      ker::WriteDofs3d(e, d, dofs, YE);
   }

   template<int VDIM,
            int SDIM,
            typename Smem,
            typename Dofs,
            typename ArgReg,
            typename YE_t>
   static MFEM_HOST_DEVICE void write_gradient(const int d,
                                               const int q,
                                               const int e,
                                               Smem &s,
                                               ArgReg &rarg,
                                               Dofs &dofs,
                                               YE_t &YE)
   {
      static_assert(SDIM == DIM, "gradient spatial dim must match kernel DIM");
      if constexpr (DIM == 2) { write_gradient_2d(d, q, e, s, rarg, dofs, YE); }
      else
      {
         write_gradient_3d(d, q, e, s, rarg, dofs, YE);
      }
   }
};

// ────────────────────────────────────────────────────────────────────────────
template<int T_DIM, int T_Q1D = LocalQFHOBackendMQ1()>
struct LocalQFHOBackend
{
   // ─────────────────────────────────────────────────────
   static constexpr int DIM = T_DIM, MQ1 = T_Q1D, Q1D = T_Q1D;
   static_assert(DIM == 2 || DIM == 3);

   // ─────────────────────────────────────────────────────
   static inline ThreadBlocks thread_blocks(const int q1d)
   {
      MFEM_ASSERT(q1d <= Q1D, "q1d must be <= " << Q1D);
      return { q1d, q1d, 1 };
   }

   // ─────────────────────────────────────────────────────
   static inline constexpr int MAX_THREADS_PER_BLOCK() { return Q1D * Q1D; }

   // ─────────────────────────────────────────────────────
   using backend_t = ho_ker_backend<DIM, Q1D>;

   // ─────────────────────────────────────────────────────
   using Shared = typename backend_t::Shared;

   // ─────────────────────────────────────────────────────
   template<typename WT, typename WI, typename Cache, typename AddY>
   static MFEM_HOST_DEVICE inline void DiagContract(Shared &s,
                                                    const int num_dof_1d,
                                                    const int q1d,
                                                    const int nz_dof,
                                                    WT wt,
                                                    WI wi,
                                                    Cache cache,
                                                    AddY add_y)
   {
      MFEM_CONTRACT_VAR(nz_dof);
      const int nqz = (DIM == 3) ? q1d : 1;
      const int ndz = (DIM == 3) ? num_dof_1d : 1;

      ker::s_regs3d_t<MQ1> rz, ry;
      auto &smem = s.M;

      MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
         {
            for (int dz = 0; dz < ndz; dz++)
            {
               real_t u = 0.0;
               for (int qz = 0; qz < nqz; qz++)
               {
                  const int q = qx + (qy + qz * q1d) * q1d;
                  const real_t wz =
                     (DIM == 3) ? (wt(2, qz, dz) * wi(2, qz, dz)) : real_t(1);
                  u += wz * cache(q);
               }
               rz[dz][qy][qx] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;

      for (int dz = 0; dz < ndz; dz++)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
            { smem[qy][qx] = rz[dz][qy][qx]; }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD_DIRECT(dy, y, num_dof_1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
            {
               real_t u = 0.0;
               for (int qy = 0; qy < q1d; qy++)
               {
                  u += wt(1, qy, dy) * wi(1, qy, dy) * smem[qy][qx];
               }
               ry[dz][dy][qx] = u;
            }
         }
         MFEM_SYNC_THREAD;
      }

      for (int dz = 0; dz < ndz; dz++)
      {
         MFEM_FOREACH_THREAD_DIRECT(dy, y, num_dof_1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
            { smem[dy][qx] = ry[dz][dy][qx]; }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD_DIRECT(dy, y, num_dof_1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(dx, x, num_dof_1d)
            {
               real_t u = 0.0;
               for (int qx = 0; qx < q1d; qx++)
               {
                  u += wt(0, qx, dx) * wi(0, qx, dx) * smem[dy][qx];
               }
               add_y(dx, dy, dz, u);
            }
         }
         MFEM_SYNC_THREAD;
      }
   }

   // ─────────────────────────────────────────────────────
   template<typename T>
   using QReg = ho_qreg_t<backend_t, T>;

   // ─────────────────────────────────────────────────────
   template<typename ArgRegT, typename XE_T>
   static inline MFEM_HOST_DEVICE void LoadValue(Shared &s,
                                                 const int e,
                                                 const int d,
                                                 const int q,
                                                 const int,
                                                 const real_t *B,
                                                 const XE_T &XE,
                                                 ArgRegT &rarg)
   {
      ker::LoadMatrix(d, q, B, s.B);
      typename backend_t::template val_reg_t<1> dofs;
      backend_t::load_dofs(e, d, XE, dofs);
      backend_t::eval_value(d, q, s, dofs, rarg);
   }

   // ─────────────────────────────────────────────────────
   template<int RNK,
            typename ArgRegT,
            typename XE_T,
            typename FieldParamT = ArgRegT>
   static inline MFEM_HOST_DEVICE void LoadGradient(Shared &s,
                                                    const int e,
                                                    const int d,
                                                    const int q,
                                                    const int,
                                                    const real_t *B,
                                                    const real_t *G,
                                                    const XE_T &XE,
                                                    ArgRegT &rarg)
   {
      ker::LoadMatrix(d, q, B, s.B);
      ker::LoadMatrix(d, q, G, s.G);
      static_assert(RNK == 1 || RNK == 2);
      static constexpr int VDIM =
         (RNK == 1) ? 1 : qf_param_shape<FieldParamT>::extents[0];
      static constexpr int SDIM = (RNK == 1)
                                  ? qf_param_shape<FieldParamT>::extents[0]
                                  : qf_param_shape<FieldParamT>::extents[1];
      if constexpr (SDIM == DIM)
      {
         typename backend_t::template del_reg_t<VDIM, SDIM> dofs;
         if constexpr (RNK == 1) { backend_t::load_dofs(e, d, XE, dofs); }
         else
         {
            backend_t::template load_grad_dofs<VDIM, SDIM>(e, d, XE, dofs);
         }
         backend_t::template grad<VDIM, SDIM>(d, q, s, dofs, rarg);
      }
   }

   // ─────────────────────────────────────────────────────
   template<typename T>
   static MFEM_HOST_DEVICE inline auto
   qp_pull(QReg<T> &reg, int qx, int qy, int qz)
   { return hok::load_at<DIM, T>(reg, qx, qy, qz); }

   // ─────────────────────────────────────────────────────
   template<typename T>
   static MFEM_HOST_DEVICE inline auto qp_pull_directional(
      QReg<T> &preg, QReg<T> &sreg, int qx, int qy, int qz, bool dependent)
   { return hok::pull_directional<DIM, T>(preg, sreg, qx, qy, qz, dependent); }

   // ─────────────────────────────────────────────────────
   template<typename DT, typename XE_T>
   static MFEM_HOST_DEVICE inline DT identity_qp_pull_dual(bool dependent,
                                                           const XE_T &XP,
                                                           const XE_T &XD,
                                                           int qx,
                                                           int qy,
                                                           int qz,
                                                           int e)
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
               t(i, j).gradient =
                  dependent ? XD(i + e0 * j, qx, qy, qz, e) : 0.0;
            }
         }
         return t;
      }
      else
      {
         static_assert(false, "Unsupported");
      }
   }

   // ─────────────────────────────────────────────────────
   template<typename T>
   static MFEM_HOST_DEVICE inline void
   qp_push(QReg<T> &reg, int qx, int qy, int qz, const T &out)
   { hok::store_at<DIM, T, decltype(reg), false>(reg, qx, qy, qz, out); }

   // ─────────────────────────────────────────────────────
   template<typename T>
   static MFEM_HOST_DEVICE inline void
   qp_push_tangent(QReg<T> &reg, int qx, int qy, int qz, const T &out)
   {
      hok::store_at<DIM, T, decltype(reg), qf_param_uses_dual_v<T>>(
                                                                    reg, qx, qy, qz, out);
   }

   // ─────────────────────────────────────────────────────
   template<typename DT, typename YE_T>
   static MFEM_HOST_DEVICE inline void identity_qp_write_value(
      YE_T &YE, int qx, int qy, int qz, int e, const DT &qout)
   {
      constexpr int RNK = qf_param_shape<DT>::rank;
      if constexpr (qf_param_uses_dual_v<DT>)
      {
         if constexpr (RNK == 0)
         {
            YE(0, qx, qy, qz, e) = qf_store_value(qout);
         }
         else if constexpr (RNK == 1)
         {
            constexpr int e0 = qf_param_shape<DT>::extents[0];
            MFEM_UNROLL(e0)
            for (int dd = 0; dd < e0; ++dd)
            {
               YE(dd, qx, qy, qz, e) = qf_store_value(qout(dd));
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
                  YE(i + e0 * j, qx, qy, qz, e) = qf_store_value(qout(i, j));
               }
            }
         }
         else
         {
            static_assert(false, "Unsupported");
         }
      }
   }

   // ─────────────────────────────────────────────────────
   template<typename DT, typename YE_T>
   static MFEM_HOST_DEVICE inline void identity_qp_write_tangent(
      YE_T &YE, int qx, int qy, int qz, int e, const DT &qout)
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

   // ─────────────────────────────────────────────────────
   template<typename ArgRegT, typename YE_T>
   static inline MFEM_HOST_DEVICE void WriteValue(Shared &s,
                                                  const int e,
                                                  const int d,
                                                  const int q,
                                                  const int,
                                                  const real_t *B,
                                                  YE_T &YE,
                                                  ArgRegT &rarg)
   {
      ker::LoadMatrix(d, q, B, s.B);
      typename backend_t::template val_reg_t<1> dofs;
      backend_t::write_value(d, q, e, s, rarg, dofs, YE);
   }

   // ─────────────────────────────────────────────────────
   template<int RNK,
            typename ArgRegT,
            typename YE_T,
            typename FieldParamT = ArgRegT>
   static inline MFEM_HOST_DEVICE void WriteGradient(Shared &s,
                                                     const int e,
                                                     const int d,
                                                     const int q,
                                                     const int,
                                                     const real_t *B,
                                                     const real_t *G,
                                                     YE_T &YE,
                                                     ArgRegT &rarg)
   {
      ker::LoadMatrix(d, q, B, s.B);
      ker::LoadMatrix(d, q, G, s.G);
      static_assert(RNK == 1 || RNK == 2);
      static constexpr int VDIM =
         (RNK == 1) ? 1 : qf_param_shape<FieldParamT>::extents[0];
      static constexpr int SDIM = (RNK == 1)
                                  ? qf_param_shape<FieldParamT>::extents[0]
                                  : qf_param_shape<FieldParamT>::extents[1];
      if constexpr (SDIM == DIM)
      {
         typename backend_t::template del_reg_t<VDIM, SDIM> dofs;
         backend_t::template write_gradient<VDIM, SDIM>(
            d, q, e, s, rarg, dofs, YE);
      }
   }
};

/// @brief Dispatch to a compile-time HO kernel with MQ1 >= runtime @a q1d.
template <typename HOKernelTable, int DIM>
inline typename HOKernelTable::KernelSignature
DispatchHOKernelByQ1D(int q1d)
{
   constexpr int MQ1 = LocalQFHOBackendMQ1();
   MFEM_VERIFY(q1d >= 2 && q1d <= MQ1,
               "Unsupported HO quadrature order: " << q1d);
   return HOKernelTable::template Kernel<DIM, MQ1>();
}

} // namespace mfem::future
