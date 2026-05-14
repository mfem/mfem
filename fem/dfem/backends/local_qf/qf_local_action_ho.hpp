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

struct LocalQFHOBackend
{
   //////////////////////////////////////////////////////////////////
   static constexpr int MQ1 = 16;

   //////////////////////////////////////////////////////////////////
   template <int DIM> static
   ThreadBlocks thread_blocks(const int q1d)
   {
      MFEM_VERIFY(q1d <= MQ1, "q1d must be less than or equal to MQ1:" << MQ1);
      return {q1d, (DIM >= 2) ? q1d : 1, 1};
   }

   template <int T_Q1D> static inline
   constexpr int MAX_THREADS_PER_BLOCK() { return T_Q1D*T_Q1D; }

   //////////////////////////////////////////////////////////////////
   template<int MQ1>
   struct Shared
   {
      real_t M[MQ1][MQ1], B[MQ1][MQ1], G[MQ1][MQ1];
   };

   //////////////////////////////////////////////////////////////////
   template<int DIM, int ext_sz, int MQ1, typename ArgRegT, typename XE_T>
   static inline MFEM_HOST_DEVICE
   void LoadValue(Shared<MQ1> &s,
                  const int e, const int d, const int q, const int,
                  const real_t *B, const XE_T &XE, ArgRegT &rarg)
   {
      static_assert(DIM == 3);
      static_assert(ext_sz == 1);
      ker::LoadMatrix(d, q, B, s.B);
      {
         ker::vd_regs3d_t<1, 3, MQ1> dofs;
         ker::LoadDofs3d(e, d, XE, dofs);
         ker::Eval3d(d, q, s.M, s.B, s.G, dofs, rarg);
      }
   }

   //////////////////////////////////////////////////////////////////
   template<int DIM, int RNK, int MQ1, typename ArgRegT, typename XE_T>
   static inline MFEM_HOST_DEVICE
   void LoadGradient(Shared<MQ1> &s,
                     const int e, const int d, const int q, const int,
                     const real_t *B, const real_t *G,
                     const XE_T &XE, ArgRegT &rarg)
   {
      static_assert(RNK == 1 || RNK == 2);
      ker::LoadMatrix(d, q, B, s.B);
      ker::LoadMatrix(d, q, G, s.G);
      if constexpr (RNK == 1)
      {
         ker::vd_regs3d_t<1, 3, MQ1> dofs;
         ker::LoadDofs3d(e, d, XE, dofs);
         ker::Grad3d(d, q, s.M, s.B, s.G, dofs, rarg);
      }
      if constexpr (RNK == 2)
      {
         ker::vd_regs3d_t<3, 3, MQ1> dofs;
         ker::LoadDofs3d(e, d, XE, dofs);
         ker::Grad3d(d, q, s.M, s.B, s.G, dofs, rarg);
      }
   }

   //////////////////////////////////////////////////////////////////
   /// High-order 3D quadrature register storage for a q-function parameter
   template <typename T, int MQ1, int RNK = qf_param_shape<T>::rank>
   struct high_order_qp_reg
   {
      static_assert(RNK >= 0 && RNK <= 2);
   };

   template <typename T, int MQ1>
   struct high_order_qp_reg<T, MQ1, 0>
   {
      using type = mfem::kernels::internal::v_regs3d_t<1, MQ1>;
   };

   template <typename T, int MQ1>
   struct high_order_qp_reg<T, MQ1, 1>
   {
      static constexpr int e0 = qf_param_shape<T>::extents[0];
      using type = mfem::kernels::internal::vd_regs3d_t<1, e0, MQ1>;
   };

   template <typename T, int MQ1>
   struct high_order_qp_reg<T, MQ1, 2>
   {
      static constexpr int e0 = qf_param_shape<T>::extents[0];
      static constexpr int e1 = qf_param_shape<T>::extents[1];
      using type = mfem::kernels::internal::vd_regs3d_t<e0, e1, MQ1>;
   };

   template <typename T, int MQ1>
   using QPReg = typename high_order_qp_reg<T, MQ1>::type;

   //////////////////////////////////////////////////////////////////
   template<typename T, int MQ1>
   static MFEM_HOST_DEVICE inline
   auto qp_load(QPReg<T, MQ1> &reg, int qz, int qy, int qx)
   {
      constexpr int R = qf_param_shape<T>::rank;
      if constexpr (R == 0)
      {
         if constexpr (std::is_same_v<T, real_t>)
         {
            return reg(0, qz, qy, qx);
         }
         else
         {
            return T{reg(0, qz, qy, qx)};
         }
      }
      else if constexpr (R == 1)
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         T t;
         MFEM_UNROLL(e0)
         for (int dd = 0; dd < e0; ++dd) { t(dd) = reg(0, dd, qz, qy, qx); }
         return t;
      }
      else // R == 2
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
   //////////////////////////////////////////////////////////////////
   template<typename T, int MQ1>
   static MFEM_HOST_DEVICE inline
   void qp_store(QPReg<T, MQ1> &reg, int qz, int qy, int qx,
                 const T &out)
   {
      constexpr int R = qf_param_shape<T>::rank;
      if constexpr (R == 0)
      {
         if constexpr (std::is_same_v<T, real_t>)
         {
            reg(0, qz, qy, qx) = out;
         }
         else
         {
            reg(0, qz, qy, qx) = out.scalar();
         }
      }
      else if constexpr (R == 1)
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         MFEM_UNROLL(e0)
         for (int dd = 0; dd < e0; ++dd) { reg(0, dd, qz, qy, qx) = out(dd); }
      }
      else if constexpr (R == 2)
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         constexpr int e1 = qf_param_shape<T>::extents[1];
         MFEM_UNROLL(e0)
         for (int i = 0; i < e0; ++i)
         {
            MFEM_UNROLL(e1)
            for (int j = 0; j < e1; ++j) { reg(i, j, qz, qy, qx) = out(i, j); }
         }
      }
      else
      {
         static_assert(false, "Unsupported");
      }
   }

   //////////////////////////////////////////////////////////////////
   template<int DIM, int ext_sz, int MQ1, typename ArgRegT, typename YE_T>
   static inline MFEM_HOST_DEVICE
   void WriteValue(Shared<MQ1>&,
                   const int, const int, const int, const int,
                   const real_t*, const YE_T &, ArgRegT &)
   {
      static_assert(sizeof(ArgRegT) == 0,
                    "LocalQFHOBackend::WriteValue is not implemented");
   }

   //////////////////////////////////////////////////////////////////
   template<int DIM, int ext_sz, int MQ1, typename ArgRegT, typename YE_T>
   static inline MFEM_HOST_DEVICE
   void WriteGradient(Shared<MQ1> &s,
                      const int e, const int d, const int q, const int,
                      const real_t *B, const real_t *G,
                      YE_T &YE, ArgRegT &rarg)
   {
      if constexpr (ext_sz == 1)
      {
         ker::LoadMatrix(d, q, B, s.B);
         ker::LoadMatrix(d, q, G, s.G);
         ker::vd_regs3d_t<1, 3, MQ1> dofs;
         ker::GradTranspose3d(d, q, s.M, s.B, s.G, rarg, dofs);
         ker::WriteDofs3d(e, d, dofs, YE);
      }
      else { static_assert(ext_sz == 1, "Unsupported gradient rank"); }
   }
};

} // namespace mfem::future