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
#include "util.hpp"

namespace mfem::future
{

///////////////////////////////////////////////////////////////////////////////
/// Register type for one LO q-function parameter
template<typename KerOps, typename T, int rank = qf_param_shape<T>::rank>
struct lo_qreg;

template<typename KerOps, typename T>
struct lo_qreg<KerOps, T, 0>
{
   using type = typename KerOps::template qreg_t<1>;
};

template<typename KerOps, typename T>
struct lo_qreg<KerOps, T, 1>
{
   static constexpr int e0 = qf_param_shape<T>::extents[0];
   using type = typename KerOps::template qreg_t<e0>;
};

template<typename KerOps, typename T>
struct lo_qreg<KerOps, T, 2>
{
   static constexpr int e0 = qf_param_shape<T>::extents[0];
   static constexpr int e1 = qf_param_shape<T>::extents[1];
   using type = typename KerOps::template qreg_vd_t<e0, e1>;
};

template<typename KerOps, typename T>
using lo_qreg_t = typename lo_qreg<KerOps, T>::type;

///////////////////////////////////////////////////////////////////////////////
namespace lok
{

template<int DIM, typename Reg>
MFEM_HOST_DEVICE inline auto &at(Reg &reg, int qx, int qy, int qz)
{
   if constexpr (DIM == 2)
   {
      MFEM_CONTRACT_VAR(qz);
      return reg[qy][qx];
   }
   else
   {
      return reg[qz][qy][qx];
   }
}

template<int DIM, typename T, typename Reg>
MFEM_HOST_DEVICE inline auto load_at(Reg &reg, int qx, int qy, int qz)
{
   constexpr int RNK = qf_param_shape<T>::rank;
   auto &qp = at<DIM>(reg, qx, qy, qz);
   if constexpr (RNK == 0) { return T{qp[0]}; }
   else if constexpr (RNK == 1)
   {
      constexpr int e0 = qf_param_shape<T>::extents[0];
      T t{};
      MFEM_UNROLL(e0)
      for (int dd = 0; dd < e0; ++dd) { t(dd) = qp[dd]; }
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
         for (int j = 0; j < e1; ++j) { t(i, j) = qp[i][j]; }
      }
      return t;
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
   constexpr int RNK = qf_param_shape<T>::rank;
   auto &qp = at<DIM>(reg, qx, qy, qz);
   if constexpr (RNK == 0) { qp[0] = qp_store<tangent>(out); }
   else if constexpr (RNK == 1)
   {
      constexpr int e0 = qf_param_shape<T>::extents[0];
      MFEM_UNROLL(e0)
      for (int dd = 0; dd < e0; ++dd) { qp[dd] = qp_store<tangent>(out(dd)); }
   }
   else
   {
      constexpr int e0 = qf_param_shape<T>::extents[0];
      constexpr int e1 = qf_param_shape<T>::extents[1];
      MFEM_UNROLL(e0)
      for (int i = 0; i < e0; ++i)
      {
         MFEM_UNROLL(e1)
         for (int j = 0; j < e1; ++j) { qp[i][j] = qp_store<tangent>(out(i, j)); }
      }
   }
}

/// Pull primal/tangent pair into a dual q-function argument
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
      auto &pqp = at<DIM>(preg, qx, qy, qz);
      auto &sqp = at<DIM>(sreg, qx, qy, qz);
      if constexpr (RNK == 0) { return T{pqp[0], sqp[0]}; }
      else if constexpr (RNK == 1)
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         T t{};
         MFEM_UNROLL(e0)
         for (int dd = 0; dd < e0; ++dd) { t(dd) = {pqp[dd], sqp[dd]}; }
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
            for (int j = 0; j < e1; ++j) { t(i, j) = {pqp[i][j], sqp[i][j]}; }
         }
         return t;
      }
   }
}

} // namespace lok

///////////////////////////////////////////////////////////////////////////////
/// LO tensor-product kernels
template<int T_DIM, int MQ1>
struct lo_ker_backend
{
   static constexpr int DIM = T_DIM;
   static_assert(DIM == 2 || DIM == 3);

   template<int VDIM>
   using qreg_t = std::conditional_t<(DIM == 2),
         ker::regs2d_t<VDIM, MQ1>, ker::regs3d_t<VDIM, MQ1>>;

   template<int VDIM, int SDIM>
   using qreg_vd_t = std::conditional_t<(DIM == 2),
         ker::regs2d_vd_t<VDIM, SDIM, MQ1>, ker::regs3d_vd_t<VDIM, SDIM, MQ1>>;

   struct Shared2d
   {
      real_t M[2][MQ1][MQ1][DIM];
      real_t B[MQ1][MQ1], G[MQ1][MQ1];
   };

   struct Shared3d
   {
      real_t M[2][MQ1][MQ1][MQ1][DIM];
      real_t B[MQ1][MQ1], G[MQ1][MQ1];
   };

   using Shared = std::conditional_t<(DIM == 2), Shared2d, Shared3d>;

   template<typename ArgRegT, typename XE_T>
   static MFEM_HOST_DEVICE void load_value(Shared &s,
                                           const int e, const int d, const int q,
                                           const real_t *B, const XE_T &XE,
                                           ArgRegT &rarg)
   {
      ker::LoadMatrix(d, q, B, s.B);
      if constexpr (DIM == 2)
      {
         ker::LoadDofs2d(e, d, XE, s.M[0]);
         ker::Eval2d(d, q, s.B, s.M[0], s.M[1], rarg);
      }
      else
      {
         ker::LoadDofs3d(e, d, XE, s.M[0]);
         ker::Eval3d(d, q, s.B, s.M[0], s.M[1], rarg);
      }
   }

   template<int RNK, typename ArgRegT, typename XE_T,
            typename FieldParamT = ArgRegT>
   static MFEM_HOST_DEVICE void load_gradient(Shared &s,
                                              const int e, const int d, const int q,
                                              const real_t *B, const real_t *G,
                                              const XE_T &XE, ArgRegT &rarg)
   {
      ker::LoadMatrix(d, q, B, s.B);
      ker::LoadMatrix(d, q, G, s.G);
      static_assert(RNK == 1 || RNK == 2);
      if constexpr (RNK == 1)
      {
         static constexpr int SDIM = qf_param_shape<FieldParamT>::extents[0];
         if constexpr (SDIM == DIM)
         {
            if constexpr (DIM == 2)
            {
               ker::LoadDofs2d(e, d, XE, s.M[0]);
               ker::Grad2d(d, q, s.B, s.G, s.M[0], s.M[1], rarg);
            }
            else
            {
               ker::LoadDofs3d(e, d, XE, s.M[0]);
               ker::Grad3d(d, q, s.B, s.G, s.M[0], s.M[1], rarg);
            }
         }
      }
      if constexpr (RNK == 2)
      {
         static constexpr int VDIM = qf_param_shape<FieldParamT>::extents[0];
         static constexpr int SDIM = qf_param_shape<FieldParamT>::extents[1];
         if constexpr (SDIM == DIM)
         {
            for (int c = 0; c < VDIM; ++c)
            {
               if constexpr (DIM == 2)
               {
                  ker::LoadDofs2d(e, d, c, XE, s.M[0]);
                  ker::VectorGrad2d(d, q, c, s.B, s.G, s.M[0], s.M[1], rarg);
               }
               else
               {
                  ker::LoadDofs3d(e, d, c, XE, s.M[0]);
                  ker::VectorGrad3d(d, q, c, s.B, s.G, s.M[0], s.M[1], rarg);
               }
            }
         }
      }
   }

   template<typename ArgRegT, typename YE_T>
   static MFEM_HOST_DEVICE void write_value(Shared &s,
                                            const int e, const int d, const int q,
                                            const real_t *B, const YE_T &YE,
                                            ArgRegT &rarg)
   {
      ker::LoadMatrix(d, q, B, s.B);
      if constexpr (DIM == 2)
      {
         ker::EvalTranspose2d(d, q, s.B, rarg, s.M[1], s.M[0]);
         ker::WriteEvalDofs2d(d, 0, e, rarg, YE);
      }
      else
      {
         ker::EvalTranspose3d(d, q, s.B, rarg, s.M[1], s.M[0]);
         ker::WriteEvalDofs3d(d, 0, e, rarg, YE);
      }
   }

   template<int RNK, typename ArgRegT, typename YE_T,
            typename FieldParamT = ArgRegT>
   static MFEM_HOST_DEVICE void write_gradient(Shared &s,
                                               const int e, const int d, const int q,
                                               const real_t *B, const real_t *G,
                                               YE_T &YE, ArgRegT &rarg)
   {
      ker::LoadMatrix(d, q, B, s.B);
      ker::LoadMatrix(d, q, G, s.G);
      static_assert(RNK == 1 || RNK == 2);
      if constexpr (RNK == 1)
      {
         static constexpr int SDIM = qf_param_shape<FieldParamT>::extents[0];
         if constexpr (SDIM == DIM)
         {
            if constexpr (DIM == 2)
            {
               ker::GradTranspose2d(d, q, s.B, s.G, rarg, s.M[1], s.M[0]);
               ker::WriteGradDofs2d(d, 0, e, rarg, YE);
            }
            else
            {
               ker::GradTranspose3d(d, q, s.B, s.G, rarg, s.M[1], s.M[0]);
               ker::WriteGradDofs3d(d, 0, e, rarg, YE);
            }
         }
      }
      else if constexpr (RNK == 2)
      {
         static constexpr int VDIM = qf_param_shape<FieldParamT>::extents[0];
         static constexpr int SDIM = qf_param_shape<FieldParamT>::extents[1];
         if constexpr (SDIM == DIM)
         {
            for (int c = 0; c < VDIM; ++c)
            {
               if constexpr (DIM == 2)
               {
                  ker::VectorGradTranspose2d(d, q, c, s.B, s.G, s.M[0], s.M[1], rarg);
                  ker::WriteGradDofs2d(d, c, e, rarg, YE);
               }
               else
               {
                  ker::VectorGradTranspose3d(d, q, c, s.B, s.G, s.M[0], s.M[1], rarg);
                  ker::WriteGradDofs3d(d, c, e, rarg, YE);
               }
            }
         }
      }
      else { static_assert(false, "Unsupported"); }
   }
};

///////////////////////////////////////////////////////////////////////////////
template<int T_DIM>
struct LocalQFLOBackend
{
   //////////////////////////////////////////////////////////////////
   static constexpr int DIM = T_DIM, MQ1 = 8;
   static_assert(DIM == 2 || DIM == 3);

   //////////////////////////////////////////////////////////////////
   static inline ThreadBlocks thread_blocks(const int q1d)
   {
      MFEM_VERIFY(q1d <= MQ1, "q1d must be <= MQ1:" << MQ1);
      return {q1d, q1d, (DIM == 2) ? 1 : q1d};
   }

   //////////////////////////////////////////////////////////////////
   template <int Q1D> static inline
   constexpr int MAX_THREADS_PER_BLOCK()
   {
      return Q1D * Q1D * ((DIM == 2) ? 1 : Q1D);
   }

   //////////////////////////////////////////////////////////////////
   template<int MQ1>
   using backend_t = lo_ker_backend<DIM, MQ1>;

   //////////////////////////////////////////////////////////////////
   template<int MQ1>
   using Shared = typename backend_t<MQ1>::Shared;

   //////////////////////////////////////////////////////////////////
   template<typename T, int MQ1>
   using QReg = lo_qreg_t<backend_t<MQ1>, T>;

   //////////////////////////////////////////////////////////////////
   template<int MQ1, typename ArgRegT, typename XE_T>
   static inline MFEM_HOST_DEVICE
   void LoadValue(Shared<MQ1> &s,
                  const int e, const int d, const int q, const int,
                  const real_t *B, const XE_T &XE, ArgRegT &rarg)
   {
      backend_t<MQ1>::load_value(s, e, d, q, B, XE, rarg);
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
      backend_t<MQ1>::template load_gradient<RNK, ArgRegT, XE_T, FieldParamT>
      (s, e, d, q, B, G, XE, rarg);
   }

   //////////////////////////////////////////////////////////////////
   template<typename T, int MQ1>
   static MFEM_HOST_DEVICE inline
   auto qp_pull(QReg<T, MQ1> &reg, int qx, int qy, int qz)
   {
      if constexpr (qf_param_uses_dual_v<T>)
      {
         return lok::load_at<DIM, T>(reg, qx, qy, qz);
      }
      else
      {
         constexpr int RNK = qf_param_shape<T>::rank;
         if constexpr (RNK == 0)
         {
            return as_tensor<real_t>(&lok::at<DIM>(reg, qx, qy, qz)[0]);
         }
         else if constexpr (RNK == 1)
         {
            constexpr int e0 = qf_param_shape<T>::extents[0];
            return as_tensor<real_t, e0>(&lok::at<DIM>(reg, qx, qy, qz)[0]);
         }
         else if constexpr (RNK == 2)
         {
            constexpr int e0 = qf_param_shape<T>::extents[0];
            constexpr int e1 = qf_param_shape<T>::extents[1];
            return as_tensor<real_t, e0, e1>(
                      &lok::at<DIM>(reg, qx, qy, qz)[0][0]);
         }
         else { static_assert(false, "Unsupported"); }
      }
   }

   //////////////////////////////////////////////////////////////////
   template<typename T, int MQ1>
   static MFEM_HOST_DEVICE inline
   auto qp_pull_directional(QReg<T, MQ1> &preg, QReg<T, MQ1> &sreg,
                            int qx, int qy, int qz, bool dependent)
   {
      return lok::pull_directional<DIM, T>(preg, sreg, qx, qy, qz,
                                           dependent);
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
      else { static_assert(false, "Unsupported"); }
   }

   //////////////////////////////////////////////////////////////////
   template<typename T, int MQ1>
   static MFEM_HOST_DEVICE inline
   void qp_push_tangent(QReg<T, MQ1> &reg, int qx, int qy, int qz, const T &out)
   {
      if constexpr (!qf_param_uses_dual_v<T>)
      {
         qp_push<T, MQ1>(reg, qx, qy, qz, out);
      }
      else
      {
         lok::store_at<DIM, T, decltype(reg), true>(reg, qx, qy, qz, out);
      }
   }

   //////////////////////////////////////////////////////////////////
   template<typename DT, typename YE_T>
   static MFEM_HOST_DEVICE inline
   void identity_qp_write_value(YE_T &YE, int qx, int qy, int qz, int e,
                                const DT &qout)
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
         else { static_assert(false, "Unsupported"); }
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
         else { static_assert(false, "Unsupported"); }
      }
   }

   //////////////////////////////////////////////////////////////////
   template<typename T, int MQ1>
   static MFEM_HOST_DEVICE inline
   void qp_push(QReg<T, MQ1> &reg, int qx, int qy, int qz, const T &out)
   {
      if constexpr (qf_param_uses_dual_v<T>)
      {
         lok::store_at<DIM, T, decltype(reg), false>(reg, qx, qy, qz, out);
      }
      else
      {
         constexpr int RNK = qf_param_shape<T>::rank;
         if constexpr (RNK == 0)
         {
            as_tensor<real_t>(&lok::at<DIM>(reg, qx, qy, qz)[0]) = out;
         }
         else if constexpr (RNK == 1)
         {
            constexpr int e0 = qf_param_shape<T>::extents[0];
            as_tensor<real_t, e0>(&lok::at<DIM>(reg, qx, qy, qz)[0]) = out;
         }
         else if constexpr (RNK == 2)
         {
            constexpr int e0 = qf_param_shape<T>::extents[0];
            constexpr int e1 = qf_param_shape<T>::extents[1];
            as_tensor<real_t, e0, e1>(
               &lok::at<DIM>(reg, qx, qy, qz)[0][0]) = out;
         }
         else { static_assert(false, "Unsupported"); }
      }
   }

   //////////////////////////////////////////////////////////////////
   template<int MQ1, typename ArgRegT, typename YE_T>
   static inline MFEM_HOST_DEVICE
   void WriteValue(Shared<MQ1> &s,
                   const int e, const int d, const int q, const int,
                   const real_t *B, const YE_T &YE, ArgRegT &rarg)
   {
      backend_t<MQ1>::write_value(s, e, d, q, B, YE, rarg);
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
      backend_t<MQ1>::template write_gradient<RNK, ArgRegT, YE_T, FieldParamT>
      (s, e, d, q, B, G, YE, rarg);
   }
};

} // namespace mfem::future
