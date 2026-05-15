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

namespace mfem::future
{

/// HO tensor-product kernels (2D/3D selected at compile time via @a DIM).
template<int DIM, int MQ1>
struct ho_ker_backend
{
   static constexpr int dimension = DIM;
   static_assert(DIM == 2 || DIM == 3, "ho_ker_backend: only 2D and 3D");

   template<int VDIM = 1>
   using value_reg_t =
      std::conditional_t<(DIM == 2), ker::v_regs2d_t<VDIM, MQ1>, ker::v_regs3d_t<VDIM, MQ1>>;

   template<int e0, int e1>
   using grad_reg_t = std::conditional_t<(DIM == 2),
         ker::vd_regs2d_t<e0, e1, MQ1>, ker::vd_regs3d_t<e0, e1, MQ1>>;

   template<typename XE_t, typename Dofs>
   static MFEM_HOST_DEVICE void load_value_dofs(const int e, const int d,
                                                const XE_t &XE, Dofs &dofs)
   {
      if constexpr (DIM == 2)
      {
         for (int c = 0; c < 1; ++c)
         {
            MFEM_FOREACH_THREAD_DIRECT(dy, y, d)
            {
               MFEM_FOREACH_THREAD_DIRECT(dx, x, d)
               {
                  dofs[c][dy][dx] = XE(dx, dy, 0, c, e);
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
      else { ker::LoadDofs3d(e, d, XE, dofs); }
   }

   template<typename XE_t, typename Dofs>
   static MFEM_HOST_DEVICE void load_grad_scalar_dofs(const int e, const int d,
                                                      const XE_t &XE, Dofs &dofs)
   {
      if constexpr (DIM == 2)
      {
         for (int c = 0; c < 1; ++c)
         {
            for (int dd = 0; dd < DIM; ++dd)
            {
               MFEM_FOREACH_THREAD_DIRECT(dy, y, d)
               {
                  MFEM_FOREACH_THREAD_DIRECT(dx, x, d)
                  {
                     dofs[c][dd][dy][dx] = XE(dx, dy, 0, c, e);
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
      else { ker::LoadDofs3d(e, d, XE, dofs); }
   }

   template<int e0, int e1, typename XE_t, typename Dofs>
   static MFEM_HOST_DEVICE void load_grad_dofs(const int e, const int d,
                                               const XE_t &XE, Dofs &dofs)
   {
      if constexpr (DIM == 2)
      {
         for (int c = 0; c < e0; ++c)
         {
            for (int dd = 0; dd < e1; ++dd)
            {
               MFEM_FOREACH_THREAD_DIRECT(dy, y, d)
               {
                  MFEM_FOREACH_THREAD_DIRECT(dx, x, d)
                  {
                     dofs[c][dd][dy][dx] = XE(dx, dy, 0, c, e);
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
      else { ker::LoadDofs3d(e, d, XE, dofs); }
   }

   template<typename Smem, typename Dofs, typename ArgReg>
   static MFEM_HOST_DEVICE void eval_value(const int d, const int q,
                                           Smem &s, Dofs &dofs, ArgReg &rarg)
   {
      if constexpr (DIM == 2) { ker::Eval2d<1, MQ1>(d, q, s.M, s.B, dofs, rarg); }
      else { ker::Eval3d(d, q, s.M, s.B, dofs, rarg); }
   }

   template<int e0, int e1, typename Smem, typename Dofs, typename ArgReg>
   static MFEM_HOST_DEVICE void grad(const int d, const int q,
                                     Smem &s, Dofs &dofs, ArgReg &rarg)
   {
      if constexpr (DIM == 2)
      {
         ker::Grad2d<e0, e1, MQ1>(d, q, s.M, s.B, s.G, dofs, rarg);
      }
      else { ker::Grad3d(d, q, s.M, s.B, s.G, dofs, rarg); }
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
         for (int c = 0; c < 1; ++c)
         {
            MFEM_FOREACH_THREAD_DIRECT(dy, y, d)
            {
               MFEM_FOREACH_THREAD_DIRECT(dx, x, d)
               {
                  YE(dx, dy, 0, c, e) += dofs[c][dy][dx];
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
      else
      {
         ker::EvalTranspose3d(d, q, s.M, s.B, rarg, dofs);
         ker::WriteDofs3d(e, d, dofs, YE);
      }
   }

   template<typename Smem, typename Dofs, typename ArgReg, typename YE_t>
   static MFEM_HOST_DEVICE void write_gradient(const int d, const int q,
                                               const int e, Smem &s,
                                               ArgReg &rarg, Dofs &dofs,
                                               YE_t &YE)
   {
      if constexpr (DIM == 2)
      {
         ker::GradTranspose2d(d, q, s.M, s.B, s.G, rarg, dofs);
         for (int c = 0; c < 1; ++c)
         {
            MFEM_FOREACH_THREAD_DIRECT(dy, y, d)
            {
               MFEM_FOREACH_THREAD_DIRECT(dx, x, d)
               {
                  real_t y = 0.0;
                  for (int dd = 0; dd < DIM; ++dd) { y += dofs[c][dd][dy][dx]; }
                  YE(dx, dy, 0, c, e) += y;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
      else
      {
         ker::GradTranspose3d(d, q, s.M, s.B, s.G, rarg, dofs);
         ker::WriteDofs3d(e, d, dofs, YE);
      }
   }
};

template<int DIM>
struct LocalQFHOBackend
{
   static constexpr int dimension = DIM;
   static constexpr int MQ1 = 32;

   template<int MQ1_ = MQ1>
   using K = ho_ker_backend<DIM, MQ1_>;

   static ThreadBlocks thread_blocks(const int q1d)
   {
      MFEM_VERIFY(q1d <= MQ1, "q1d must be less than or equal to MQ1:" << MQ1);
      return {q1d, q1d, 1};
   }

   template <int T_Q1D> static inline
   constexpr int MAX_THREADS_PER_BLOCK() { return T_Q1D * T_Q1D; }

   template<int MQ1_ = MQ1>
   struct Shared
   {
      real_t M[MQ1_][MQ1_], B[MQ1_][MQ1_], G[MQ1_][MQ1_];
   };

   template<typename T, int MQ1_ = MQ1>
   using QReg = ho_qreg_t<K<MQ1_>, T>;

   template<int MQ1_, typename ArgRegT, typename XE_T>
   static inline MFEM_HOST_DEVICE
   void LoadValue(Shared<MQ1_> &s,
                  const int e, const int d, const int q, const int,
                  const real_t *B, const XE_T &XE, ArgRegT &rarg)
   {
      ker::LoadMatrix(d, q, B, s.B);
      typename K<MQ1_>::template value_reg_t<1> dofs;
      K<MQ1_>::load_value_dofs(e, d, XE, dofs);
      K<MQ1_>::eval_value(d, q, s, dofs, rarg);
   }

   template<int RNK, int MQ1_, typename ArgRegT, typename XE_T,
            typename FieldParamT = ArgRegT>
   static inline MFEM_HOST_DEVICE
   void LoadGradient(Shared<MQ1_> &s,
                     const int e, const int d, const int q, const int,
                     const real_t *B, const real_t *G,
                     const XE_T &XE, ArgRegT &rarg)
   {
      ker::LoadMatrix(d, q, B, s.B);
      ker::LoadMatrix(d, q, G, s.G);
      static_assert(RNK == 1 || RNK == 2);
      if constexpr (RNK == 1)
      {
         typename K<MQ1_>::template grad_reg_t<1, K<MQ1_>::dimension> dofs;
         K<MQ1_>::load_grad_scalar_dofs(e, d, XE, dofs);
         K<MQ1_>::template grad<1, K<MQ1_>::dimension>(d, q, s, dofs, rarg);
      }
      else
      {
         static constexpr int e0 = qf_param_shape<FieldParamT>::extents[0];
         static constexpr int e1 = qf_param_shape<FieldParamT>::extents[1];
         typename K<MQ1_>::template grad_reg_t<e0, e1> dofs;
         K<MQ1_>::template load_grad_dofs<e0, e1>(e, d, XE, dofs);
         K<MQ1_>::template grad<e0, e1>(d, q, s, dofs, rarg);
      }
   }

   template<typename T, int MQ1_ = MQ1>
   static MFEM_HOST_DEVICE inline
   auto qp_pull(QReg<T, MQ1_> &reg, int qx, int qy, int qz)
   {
      return ho_qp_pull<DIM, T>(reg, qx, qy, qz);
   }

   template<typename T, int MQ1_ = MQ1>
   static MFEM_HOST_DEVICE inline
   void qp_push(QReg<T, MQ1_> &reg, int qx, int qy, int qz, const T &out)
   {
      ho_qp_push<DIM, T>(reg, qx, qy, qz, out);
   }

   template<int MQ1_, typename ArgRegT, typename YE_T>
   static inline MFEM_HOST_DEVICE
   void WriteValue(Shared<MQ1_> &s,
                   const int e, const int d, const int q, const int,
                   const real_t *B, YE_T &YE, ArgRegT &rarg)
   {
      ker::LoadMatrix(d, q, B, s.B);
      typename K<MQ1_>::template value_reg_t<1> dofs;
      K<MQ1_>::write_value(d, q, e, s, rarg, dofs, YE);
   }

   template<int MQ1_, typename ArgRegT, typename YE_T>
   static inline MFEM_HOST_DEVICE
   void WriteGradient(Shared<MQ1_> &s,
                      const int e, const int d, const int q, const int,
                      const real_t *B, const real_t *G,
                      YE_T &YE, ArgRegT &rarg)
   {
      ker::LoadMatrix(d, q, B, s.B);
      ker::LoadMatrix(d, q, G, s.G);
      typename K<MQ1_>::template grad_reg_t<1, K<MQ1_>::dimension> dofs;
      K<MQ1_>::write_gradient(d, q, e, s, rarg, dofs, YE);
   }
};

} // namespace mfem::future
