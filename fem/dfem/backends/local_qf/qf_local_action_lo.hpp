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

#include "../util.hpp" // for as_tensor

#include "qf_local_util.hpp"

namespace mfem::future
{

struct LocalQFLOBackend
{
   //////////////////////////////////////////////////////////////////
   static constexpr int DIM = 3, MQ1 = 8;

   //////////////////////////////////////////////////////////////////
   static inline ThreadBlocks thread_blocks(const int q1d)
   {
      MFEM_VERIFY(q1d <= MQ1, "q1d must be less than or equal to MQ1:" << MQ1);
      return {q1d, q1d, q1d};
   }

   //////////////////////////////////////////////////////////////////
   template <int Q1D> static inline
   constexpr int MAX_THREADS_PER_BLOCK() { return Q1D * Q1D * Q1D; }

   //////////////////////////////////////////////////////////////////
   template<int MQ1>
   struct Shared
   {
      real_t M[2][MQ1][MQ1][MQ1][3];
      real_t B[MQ1][MQ1], G[MQ1][MQ1];
   };

   //////////////////////////////////////////////////////////////////
   template<int MQ1, typename ArgRegT, typename XE_T>
   static inline MFEM_HOST_DEVICE
   void LoadValue(Shared<MQ1> &s,
                  const int e, const int d, const int q, const int,
                  const real_t *B, const XE_T &XE, ArgRegT &rarg)
   {
      ker::LoadMatrix(d, q, B, s.B);
      ker::LoadDofs3d(e, d, XE, s.M[0]);
      ker::Eval3d(d, q, s.B, s.M[0], s.M[1], rarg);
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
      if constexpr (RNK == 1)
      {
         ker::LoadDofs3d(e, d, XE, s.M[0]);
         ker::Grad3d(d, q, s.B, s.G, s.M[0], s.M[1], rarg);
      }
      if constexpr (RNK == 2)
      {
         static constexpr int VDIM = qf_param_shape<FieldParamT>::extents[0];
         static constexpr int SDIM = qf_param_shape<FieldParamT>::extents[1];
         static_assert(SDIM == DIM, "LO backend expects 3D spatial gradients");
         for (int c = 0; c < VDIM; c++)
         {
            ker::LoadDofs3d(e, d, c, XE, s.M[0]);
            ker::VectorGrad3d<VDIM, SDIM, MQ1>(d, q, c, s.B, s.G, s.M[0], s.M[1], rarg);
         }
      }
   }

   //////////////////////////////////////////////////////////////////
   /// Low-order 3D quadrature register storage for a decayed q-function parameter
   template <typename T, int MQ1, int RNK = qf_param_shape<T>::rank>
   struct low_order_qreg
   {
      static_assert(RNK >= 0 && RNK <= 2);
   };

   template <typename T, int MQ1>
   struct low_order_qreg<T, MQ1, 0>
   {
      using type = ker::regs3d_t<1, MQ1>;
   };

   template <typename T, int MQ1>
   struct low_order_qreg<T, MQ1, 1>
   {
      static constexpr int e0 = qf_param_shape<T>::extents[0];
      using type = ker::regs3d_t<e0, MQ1>;
   };

   template <typename T, int MQ1>
   struct low_order_qreg<T, MQ1, 2>
   {
      static constexpr int e0 = qf_param_shape<T>::extents[0];
      static constexpr int e1 = qf_param_shape<T>::extents[1];
      using type = ker::regs3d_vd_t<e0, e1, MQ1>;
   };

   //////////////////////////////////////////////////////////////////
   template<typename T, int MQ1>
   using QReg = typename low_order_qreg<T, MQ1>::type;

   //////////////////////////////////////////////////////////////////
   template<typename T, int MQ1>
   static MFEM_HOST_DEVICE inline
   auto qp_pull(QReg<T, MQ1> &reg, int qx, int qy, int qz)
   {
      constexpr int RNK = qf_param_shape<T>::rank;
      if constexpr (qf_param_uses_dual_v<T>)
      {
         if constexpr (RNK == 0)
         {
            return T{reg[qz][qy][qx][0]};
         }
         else if constexpr (RNK == 1)
         {
            constexpr int e0 = qf_param_shape<T>::extents[0];
            T t{};
            MFEM_UNROLL(e0)
            for (int dd = 0; dd < e0; ++dd) { t(dd) = reg[qz][qy][qx][dd]; }
            return t;
         }
         else if constexpr (RNK == 2)
         {
            constexpr int e0 = qf_param_shape<T>::extents[0];
            constexpr int e1 = qf_param_shape<T>::extents[1];
            T t;
            MFEM_UNROLL(e0)
            for (int i = 0; i < e0; ++i)
            {
               MFEM_UNROLL(e1)
               for (int j = 0; j < e1; ++j) { t(i, j) = reg[qz][qy][qx][i][j]; }
            }
            return t;
         }
         else
         {
            static_assert(false, "Unsupported");
         }
      }
      else if constexpr (RNK == 0)
      {
         return as_tensor<real_t>(&reg[qz][qy][qx][0]);
      }
      else if constexpr (RNK == 1)
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         return as_tensor<real_t, e0>(&reg[qz][qy][qx][0]);
      }
      else if constexpr (RNK == 2)
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         constexpr int e1 = qf_param_shape<T>::extents[1];
         return as_tensor<real_t, e0, e1>(&reg[qz][qy][qx][0][0]);
      }
      else
      {
         static_assert(false, "Unsupported");
      }
   }

   //////////////////////////////////////////////////////////////////
   template<typename T, int MQ1>
   static MFEM_HOST_DEVICE inline
   void qp_push(QReg<T, MQ1> &reg, int qx, int qy, int qz, const T &out)
   {
      constexpr int RNK = qf_param_shape<T>::rank;
      if constexpr (qf_param_uses_dual_v<T>)
      {
         if constexpr (RNK == 0)
         {
            reg[qz][qy][qx][0] = qf_store_value(out);
         }
         else if constexpr (RNK == 1)
         {
            constexpr int e0 = qf_param_shape<T>::extents[0];
            MFEM_UNROLL(e0)
            for (int dd = 0; dd < e0; ++dd)
            {
               reg[qz][qy][qx][dd] = qf_store_value(out(dd));
            }
         }
         else if constexpr (RNK == 2)
         {
            constexpr int e0 = qf_param_shape<T>::extents[0];
            constexpr int e1 = qf_param_shape<T>::extents[1];
            MFEM_UNROLL(e0)
            for (int i = 0; i < e0; ++i)
            {
               MFEM_UNROLL(e1)
               for (int j = 0; j < e1; ++j)
               {
                  reg[qz][qy][qx][i][j] = qf_store_value(out(i, j));
               }
            }
         }
         else
         {
            static_assert(false, "Unsupported");
         }
      }
      else if constexpr (RNK == 0)
      {
         as_tensor<real_t>(&reg[qz][qy][qx][0]) = out;
      }
      else if constexpr (RNK == 1)
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         as_tensor<real_t, e0>(&reg[qz][qy][qx][0]) = out;
      }
      else if constexpr (RNK == 2)
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         constexpr int e1 = qf_param_shape<T>::extents[1];
         as_tensor<real_t, e0, e1>(&reg[qz][qy][qx][0][0]) = out;
      }
      else
      {
         static_assert(false, "Unsupported");
      }
   }

   //////////////////////////////////////////////////////////////////
   template<int MQ1, typename ArgRegT, typename YE_T>
   static inline MFEM_HOST_DEVICE
   void WriteValue(Shared<MQ1> &s,
                   const int e, const int d, const int q, const int,
                   const real_t *B, const YE_T &YE, ArgRegT &rarg)
   {
      ker::LoadMatrix(d, q, B, s.B);
      ker::EvalTranspose3d(d, q, s.B, rarg, s.M[1], s.M[0]);
      ker::WriteEvalDofs3d(d, 0, e, rarg, YE);
   }

   //////////////////////////////////////////////////////////////////
   template<int MQ1, typename ArgRegT, typename YE_T>
   static inline MFEM_HOST_DEVICE
   void WriteGradient(Shared<MQ1> &s,
                      const int e, const int d, const int q, const int,
                      const real_t *B, const real_t *G,
                      YE_T &YE, ArgRegT &rarg)
   {
      ker::LoadMatrix(d, q, B, s.B);
      ker::LoadMatrix(d, q, G, s.G);
      ker::GradTranspose3d(d, q, s.B, s.G, rarg, s.M[1], s.M[0]);
      ker::WriteGradDofs3d(d, 0, e, rarg, YE);
   }
};

} // namespace mfem::future