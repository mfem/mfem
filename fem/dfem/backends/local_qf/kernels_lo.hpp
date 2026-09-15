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
#include "../util.hpp"    // for as_tensor
#include "util.hpp"

namespace mfem::future
{

// ────────────────────────────────────────────────────────────────────────────
inline constexpr int LocalQFLOBackendMQ1() { return 8; }

// ────────────────────────────────────────────────────────────────────────────
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

template<typename T, int rank = qf_param_shape<T>::rank>
struct qf_value_vdim
{
   static constexpr int value = qf_param_shape<T>::extents[rank - 1];
};

template<typename T>
struct qf_value_vdim<T, 0>
{
   static constexpr int value = 1;
};

template<typename T>
inline constexpr int qf_value_vdim_v = qf_value_vdim<T>::value;

// ────────────────────────────────────────────────────────────────────────────
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
   if constexpr (RNK == 0) { return T{ qp[0] }; }
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
         for (int j = 0; j < e1; ++j)
         {
            qp[i][j] = qp_store<tangent>(out(i, j));
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
      auto &pqp = at<DIM>(preg, qx, qy, qz);
      auto &sqp = at<DIM>(sreg, qx, qy, qz);
      if constexpr (RNK == 0) { return T{ pqp[0], sqp[0] }; }
      else if constexpr (RNK == 1)
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         T t{};
         MFEM_UNROLL(e0)
         for (int dd = 0; dd < e0; ++dd) { t(dd) = { pqp[dd], sqp[dd] }; }
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
            for (int j = 0; j < e1; ++j) { t(i, j) = { pqp[i][j], sqp[i][j] }; }
         }
         return t;
      }
   }
}

} // namespace lok

// ────────────────────────────────────────────────────────────────────────────
/// LO tensor-product kernels
namespace loz
{

template<int VDIM, int DIM, int MQ1>
inline MFEM_HOST_DEVICE void EvalX(const int d1d, const int q1d,
                                   const real_t (*B)[MQ1],
                                   const real_t (&sm0)[MQ1][MQ1][MQ1][DIM],
                                   real_t (&sm1)[MQ1][MQ1][MQ1][DIM])
{
   static_assert(VDIM <= DIM, "shared value workspace must fit VDIM");
   MFEM_FOREACH_THREAD(dz, z, d1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(dy, y, d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
         {
            MFEM_UNROLL(VDIM)
            for (int c = 0; c < VDIM; ++c)
            {
               real_t u = 0.0;
               MFEM_UNROLL(MQ1)
               for (int dx = 0; dx < d1d; ++dx)
               {
                  u = std::fma(B[dx][qx], sm0[dz][dy][dx][c], u);
               }
               sm1[dz][dy][qx][c] = u;
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
}

template<int VDIM, int DIM, int MQ1>
inline MFEM_HOST_DEVICE void EvalY(const int d1d, const int q1d,
                                   const real_t (*B)[MQ1],
                                   const real_t (&sm1)[MQ1][MQ1][MQ1][DIM],
                                   real_t (&sm0)[MQ1][MQ1][MQ1][DIM])
{
   static_assert(VDIM <= DIM, "shared value workspace must fit VDIM");
   MFEM_FOREACH_THREAD(dz, z, d1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
         {
            MFEM_UNROLL(VDIM)
            for (int c = 0; c < VDIM; ++c)
            {
               real_t u = 0.0;
               MFEM_UNROLL(MQ1)
               for (int dy = 0; dy < d1d; ++dy)
               {
                  u = std::fma(B[dy][qy], sm1[dz][dy][qx][c], u);
               }
               sm0[dz][qy][qx][c] = u;
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
}

template<int VDIM, int DIM, int MQ1>
inline MFEM_HOST_DEVICE void EvalZ(const int d1d, const int q1d,
                                   const real_t (*B)[MQ1],
                                   const real_t (&sm0)[MQ1][MQ1][MQ1][DIM],
                                   ker::regs3d_t<VDIM, MQ1> &reg)
{
   static_assert(VDIM <= DIM, "shared value workspace must fit VDIM");
   MFEM_FOREACH_THREAD(qz, z, q1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
         {
            MFEM_UNROLL(VDIM)
            for (int c = 0; c < VDIM; ++c)
            {
               real_t u = 0.0;
               MFEM_UNROLL(MQ1)
               for (int dz = 0; dz < d1d; ++dz)
               {
                  u = std::fma(B[dz][qz], sm0[dz][qy][qx][c], u);
               }
               reg[qz][qy][qx][c] = u;
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
}

template<int VDIM, int DIM, int MQ1>
inline MFEM_HOST_DEVICE void Eval3d(const int d1d, const int q1d,
                                    const real_t (*B)[MQ1],
                                    real_t (&sm0)[MQ1][MQ1][MQ1][DIM],
                                    real_t (&sm1)[MQ1][MQ1][MQ1][DIM],
                                    ker::regs3d_t<VDIM, MQ1> &reg)
{
   loz::EvalX<VDIM, DIM, MQ1>(d1d, q1d, B, sm0, sm1);
   loz::EvalY<VDIM, DIM, MQ1>(d1d, q1d, B, sm1, sm0);
   loz::EvalZ<VDIM, DIM, MQ1>(d1d, q1d, B, sm0, reg);
}

template<int DIM, int MQ1>
inline MFEM_HOST_DEVICE void GradX(const int d1d, const int q1d,
                                   const real_t (*B)[MQ1],
                                   const real_t (*G)[MQ1],
                                   const real_t (&sm0)[MQ1][MQ1][MQ1][DIM],
                                   real_t (&sm1)[MQ1][MQ1][MQ1][DIM])
{
   MFEM_FOREACH_THREAD(dz, z, d1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(dy, y, d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
         {
            real_t u = 0.0, v = 0.0;
            MFEM_UNROLL(MQ1)
            for (int dx = 0; dx < d1d; ++dx)
            {
               const real_t x = sm0[dz][dy][dx][0];
               u = std::fma(B[dx][qx], x, u);
               v = std::fma(G[dx][qx], x, v);
            }
            sm1[dz][dy][qx][0] = u;
            sm1[dz][dy][qx][1] = v;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

template<int DIM, int MQ1>
inline MFEM_HOST_DEVICE void GradY(const int d1d, const int q1d,
                                   const real_t (*B)[MQ1],
                                   const real_t (*G)[MQ1],
                                   const real_t (&sm1)[MQ1][MQ1][MQ1][DIM],
                                   real_t (&sm0)[MQ1][MQ1][MQ1][DIM])
{
   MFEM_FOREACH_THREAD(dz, z, d1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
         {
            real_t u = 0.0, v = 0.0, w = 0.0;
            MFEM_UNROLL(MQ1)
            for (int dy = 0; dy < d1d; ++dy)
            {
               u = std::fma(sm1[dz][dy][qx][1], B[dy][qy], u);
               v = std::fma(sm1[dz][dy][qx][0], G[dy][qy], v);
               w = std::fma(sm1[dz][dy][qx][0], B[dy][qy], w);
            }
            sm0[dz][qy][qx][0] = u;
            sm0[dz][qy][qx][1] = v;
            sm0[dz][qy][qx][2] = w;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

template<int DIM, int MQ1>
inline MFEM_HOST_DEVICE void GradZ(const int d1d, const int q1d,
                                   const real_t (*B)[MQ1],
                                   const real_t (*G)[MQ1],
                                   const real_t (&sm0)[MQ1][MQ1][MQ1][DIM],
                                   ker::regs3d_t<DIM, MQ1> &reg)
{
   MFEM_FOREACH_THREAD(qz, z, q1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
         {
            real_t u[3] = {0.0, 0.0, 0.0};
            MFEM_UNROLL(MQ1)
            for (int dz = 0; dz < d1d; ++dz)
            {
               u[0] = std::fma(B[dz][qz], sm0[dz][qy][qx][0], u[0]);
               u[1] = std::fma(B[dz][qz], sm0[dz][qy][qx][1], u[1]);
               u[2] = std::fma(G[dz][qz], sm0[dz][qy][qx][2], u[2]);
            }
            reg[qz][qy][qx][0] = u[0];
            reg[qz][qy][qx][1] = u[1];
            reg[qz][qy][qx][2] = u[2];
         }
      }
   }
   MFEM_SYNC_THREAD;
}

template<int VDIM, int DIM, int MQ1>
inline MFEM_HOST_DEVICE void VectorGradZ(
   const int d1d, const int q1d, const int c,
   const real_t (*B)[MQ1], const real_t (*G)[MQ1],
   const real_t (&sm0)[MQ1][MQ1][MQ1][DIM],
   ker::regs3d_vd_t<VDIM, DIM, MQ1> &reg)
{
   MFEM_FOREACH_THREAD(qz, z, q1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
         {
            real_t u[3] = {0.0, 0.0, 0.0};
            MFEM_UNROLL(MQ1)
            for (int dz = 0; dz < d1d; ++dz)
            {
               u[0] = std::fma(B[dz][qz], sm0[dz][qy][qx][0], u[0]);
               u[1] = std::fma(B[dz][qz], sm0[dz][qy][qx][1], u[1]);
               u[2] = std::fma(G[dz][qz], sm0[dz][qy][qx][2], u[2]);
            }
            reg[qz][qy][qx][c][0] = u[0];
            reg[qz][qy][qx][c][1] = u[1];
            reg[qz][qy][qx][c][2] = u[2];
         }
      }
   }
   MFEM_SYNC_THREAD;
}

template<int DIM, int MQ1>
inline MFEM_HOST_DEVICE void Grad3d(const int d1d, const int q1d,
                                    const real_t (*B)[MQ1],
                                    const real_t (*G)[MQ1],
                                    real_t (&sm0)[MQ1][MQ1][MQ1][DIM],
                                    real_t (&sm1)[MQ1][MQ1][MQ1][DIM],
                                    ker::regs3d_t<DIM, MQ1> &reg)
{
   loz::GradX<DIM, MQ1>(d1d, q1d, B, G, sm0, sm1);
   loz::GradY<DIM, MQ1>(d1d, q1d, B, G, sm1, sm0);
   loz::GradZ<DIM, MQ1>(d1d, q1d, B, G, sm0, reg);
}

template<int VDIM, int DIM, int MQ1>
inline MFEM_HOST_DEVICE void VectorGrad3d(const int d1d, const int q1d,
                                          const int c,
                                          const real_t (*B)[MQ1],
                                          const real_t (*G)[MQ1],
                                          real_t (&sm0)[MQ1][MQ1][MQ1][DIM],
                                          real_t (&sm1)[MQ1][MQ1][MQ1][DIM],
                                          ker::regs3d_vd_t<VDIM, DIM, MQ1> &reg)
{
   loz::GradX<DIM, MQ1>(d1d, q1d, B, G, sm0, sm1);
   loz::GradY<DIM, MQ1>(d1d, q1d, B, G, sm1, sm0);
   loz::VectorGradZ<VDIM, DIM, MQ1>(d1d, q1d, c, B, G, sm0, reg);
}

} // namespace loz

template<int T_DIM, int MQ1, bool LOOP_Z = false>
struct lo_ker_backend
{
   static constexpr int DIM = T_DIM;
   static_assert(DIM == 2 || DIM == 3);

   template<int VDIM>
   using qreg_t = std::conditional_t<(DIM == 2),
         ker::regs2d_t<VDIM, MQ1>,
         ker::regs3d_t<VDIM, MQ1>>;

   template<int VDIM, int SDIM>
   using qreg_vd_t = std::conditional_t<(DIM == 2),
         ker::regs2d_vd_t<VDIM, SDIM, MQ1>,
         ker::regs3d_vd_t<VDIM, SDIM, MQ1>>;

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

   template<typename FieldParamT, typename ArgRegT, typename XE_T>
   static MFEM_HOST_DEVICE void load_value(Shared &s,
                                           const int e,
                                           const int d,
                                           const int q,
                                           const real_t *B,
                                           const XE_T &XE,
                                           ArgRegT &rarg)
   {
      ker::LoadMatrix(d, q, B, s.B);
      using field_t = std::remove_cv_t<std::remove_reference_t<FieldParamT>>;
      constexpr int VDIM = qf_value_vdim_v<field_t>;
      if constexpr (DIM == 2)
      {
         ker::LoadDofs2d<VDIM, DIM, MQ1>(e, d, XE, s.M[0]);
         ker::Eval2d<VDIM, DIM, MQ1>(d, q, s.B, s.M[0], s.M[1], rarg);
      }
      else
      {
         ker::LoadDofs3d<VDIM, DIM, MQ1>(e, d, XE, s.M[0]);
         if constexpr (LOOP_Z)
         {
            loz::Eval3d<VDIM, DIM, MQ1>(d, q, s.B, s.M[0], s.M[1], rarg);
         }
         else
         {
            ker::Eval3d<VDIM, DIM, MQ1>(d, q, s.B, s.M[0], s.M[1], rarg);
         }
      }
   }

   template<int RNK,
            typename ArgRegT,
            typename XE_T,
            typename FieldParamT = ArgRegT>
   static MFEM_HOST_DEVICE void load_gradient(Shared &s,
                                              const int e,
                                              const int d,
                                              const int q,
                                              const real_t *B,
                                              const real_t *G,
                                              const XE_T &XE,
                                              ArgRegT &rarg)
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
               ker::LoadDofs2d(e, d, 0, XE, s.M[0]);
               ker::Grad2d(d, q, s.B, s.G, s.M[0], s.M[1], rarg);
            }
            else
            {
               ker::LoadDofs3d(e, d, 0, XE, s.M[0]);
               if constexpr (LOOP_Z)
               {
                  loz::Grad3d<DIM, MQ1>(d, q, s.B, s.G, s.M[0], s.M[1], rarg);
               }
               else
               {
                  ker::Grad3d(d, q, s.B, s.G, s.M[0], s.M[1], rarg);
               }
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
                  if constexpr (LOOP_Z)
                  {
                     loz::VectorGrad3d<VDIM, DIM, MQ1>(
                        d, q, c, s.B, s.G, s.M[0], s.M[1], rarg);
                  }
                  else
                  {
                     ker::VectorGrad3d(d, q, c, s.B, s.G, s.M[0], s.M[1], rarg);
                  }
               }
            }
         }
      }
   }

   template<typename ArgRegT, typename YE_T>
   static MFEM_HOST_DEVICE void write_value(Shared &s,
                                            const int e,
                                            const int d,
                                            const int q,
                                            const real_t *B,
                                            const YE_T &YE,
                                            ArgRegT &rarg)
   {
      ker::LoadMatrix(d, q, B, s.B);
      using field_t = std::remove_cv_t<std::remove_reference_t<ArgRegT>>;
      constexpr int VDIM = qf_value_vdim_v<field_t>;
      if constexpr (DIM == 2)
      {
         ker::EvalTranspose2d<VDIM, DIM, MQ1>(d, q, s.B, rarg, s.M[1], s.M[0]);
         ker::WriteEvalDofs2d<VDIM, MQ1>(d, 0, e, rarg, YE);
      }
      else
      {
         ker::EvalTranspose3d<VDIM, DIM, MQ1>(d, q, s.B, rarg, s.M[1], s.M[0]);
         ker::WriteEvalDofs3d<VDIM, MQ1>(d, 0, e, rarg, YE);
      }
   }

   /// Interpolate a vector element at the quadrature points under field
   /// operator @a FOP.
   template<typename FOP, typename XE_T, typename ArgRegT>
   static MFEM_HOST_DEVICE void load_vector(
      Shared &s, const int e, const DofToQuadMap &m,
      const XE_T &XE, ArgRegT &rarg)
   {
      const int q1d = m.Q1D();
      const int nslots = vector_num_slots<FOP>(m.range_dim);
      MFEM_FOREACH_THREAD(qy, y, q1d)
      MFEM_FOREACH_THREAD(qx, x, q1d)
      for (int qz = 0; qz < ((DIM == 2) ? 1 : q1d); qz++)
         for (int c = 0; c < nslots; c++)
         {
            lok::at<DIM>(rarg, qx, qy, qz)[c] = 0.0;
         }
      MFEM_SYNC_THREAD;

      const int nterms = vector_num_terms<FOP>(m.range_dim);
      for (int c = 0; c < m.range_dim; c++)
      {
         for (int t = 0; t < nterms; t++)
         {
            contract_vector_component<FOP>(s, e, m, XE, c, t, rarg);
         }
      }
   }

   /// Integrate quadrature-point data against a vector element under field
   /// operator @a FOP and add the result into the element vector. The adjoint of
   /// load_vector, off the same vector_term() table.
   template<typename FOP, typename YE_T, typename ArgRegT>
   static MFEM_HOST_DEVICE void write_vector(
      Shared &s, const int e, const DofToQuadMap &m,
      const YE_T &YE, ArgRegT &rarg)
   {
      const int nterms = vector_num_terms<FOP>(m.range_dim);
      for (int c = 0; c < m.range_dim; c++)
      {
         for (int t = 0; t < nterms; t++)
         {
            // The scatter adds into YE, so several sweeps of one component
            // accumulate rather than overwrite.
            contract_vector_component_transpose<FOP>(s, e, m, c, t, rarg, YE);
         }
      }
   }
private:
   /// Accumulate one quadrature-point contribution into the slot named by
   /// @a vt. The sign is only ever non-unit for a Curl, so the other
   /// operators keep a bare add in the innermost loop.
   template<typename FOP, typename Reg>
   static MFEM_HOST_DEVICE void vector_accum(
      Reg &reg, int qx, int qy, int qz, const VecTerm &vt, real_t value)
   {
      if constexpr (is_curl_fop_v<FOP>)
      {
         lok::at<DIM>(reg, qx, qy, qz)[vt.slot] += vt.sgn * value;
      }
      else { lok::at<DIM>(reg, qx, qy, qz)[vt.slot] += value; }
   }

   /// Read back the slot named by @a vt, the mirror of vector_accum.
   template<typename FOP, typename Reg>
   static MFEM_HOST_DEVICE real_t vector_src(
      Reg &reg, int qx, int qy, int qz, const VecTerm &vt)
   {
      const real_t value = lok::at<DIM>(reg, qx, qy, qz)[vt.slot];
      if constexpr (is_curl_fop_v<FOP>) { return vt.sgn * value; }
      else { return value; }
   }

   /// Run sweep @a t of component block @a c under field operator @a FOP,
   /// accumulating into the q-function register bank @a rarg.
   /// Bx -> By -> Bz
   ///
   /// See the HO kernel for why the destination comes from vector_term()
   /// rather than the component index. Here the gather from XE is fused into
   /// the x-sweep instead of staging the dofs first.
   template<typename FOP, typename XE_T, typename ArgRegT>
   static MFEM_HOST_DEVICE void contract_vector_component(
      Shared &s, const int e, const DofToQuadMap &m,
      const XE_T &XE, const int c, const int t, ArgRegT &rarg)
   {
      const VecTerm vt = vector_term<FOP>(c, t);
      const int deriv_dir = vt.deriv_dir;
      const int q1d = m.Q1D();
      // The 1D factor for a given (component, axis, deriv) is fixed for the
      // whole sweep, so it is resolved once here instead of per multiply-add.
      const real_t *Bx = m.Basis(c, 0, deriv_dir == 0);
      const real_t *By = m.Basis(c, 1, deriv_dir == 1);
      const real_t *Bz = m.Basis(c, 2, deriv_dir == 2);
      MFEM_CONTRACT_VAR(Bz);
      auto sm = reinterpret_cast<real_t (*)[MQ1]>(
                   reinterpret_cast<real_t *>(&s.M[0]));
      // Get the extents of the element along each axis.
      const int ex = m.Extent(c, 0), ey = m.Extent(c, 1);
      const int ez = (DIM == 2) ? 1 : m.Extent(c, 2);
      const int off = m.Offset(c);
      for (int dz = 0; dz < ez; dz++)
      {
         // Sweep along the x-axis for the current z-slice.
         MFEM_FOREACH_THREAD(dy, y, ey)
         MFEM_FOREACH_THREAD(qx, x, q1d)
         {
            real_t value = 0.0;
            for (int dx = 0; dx < ex; dx++)
            {
               const int idx = off + dx + ex * (dy + ey * dz);
               value += Bx[qx + q1d * dx] *
                        XE(idx, 0, 0, 0, e);
            }
            sm[dy][qx] = value;
         }
         MFEM_SYNC_THREAD;
         // Sweep along the y-axis for the current z-slice.
         MFEM_FOREACH_THREAD(qy, y, q1d)
         MFEM_FOREACH_THREAD(qx, x, q1d)
         {
            real_t value = 0.0;
            for (int dy = 0; dy < ey; dy++)
            {
               value += By[qy + q1d * dy] * sm[dy][qx];
            }
            if constexpr (DIM == 2) { vector_accum<FOP>(rarg, qx, qy, 0, vt, value); }
            else
            {
               for (int qz = 0; qz < q1d; qz++)
               {
                  vector_accum<FOP>(rarg, qx, qy, qz, vt,
                                    Bz[qz + q1d * dz] * value);
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   }

   /// Integrate quadrature-point values against one component block,
   /// producing that block's degrees of freedom.
   /// Same assumptions as for the forward contraction above.
   /// Bzt -> Byt -> Bxt
   template<typename FOP, typename ArgRegT, typename YE_T>
   static MFEM_HOST_DEVICE void contract_vector_component_transpose(
      Shared &s, const int e, const DofToQuadMap &m, const int c,
      const int t, ArgRegT &rarg, const YE_T &YE)
   {
      const VecTerm vt = vector_term<FOP>(c, t);
      const int deriv_dir = vt.deriv_dir;
      const int q1d = m.Q1D();
      // The 1D factor for a given (component, axis, deriv) is fixed for the
      // whole sweep, so it is resolved once here instead of per multiply-add.
      const real_t *Bx = m.Basis(c, 0, deriv_dir == 0);
      const real_t *By = m.Basis(c, 1, deriv_dir == 1);
      const real_t *Bz = m.Basis(c, 2, deriv_dir == 2);
      MFEM_CONTRACT_VAR(Bz);
      real_t *base = reinterpret_cast<real_t *>(&s.M[0]);
      auto sm0 = reinterpret_cast<real_t (*)[MQ1]>(base);
      auto sm1 = reinterpret_cast<real_t (*)[MQ1]>(base + MQ1 * MQ1);
      // Get the extents of the element along each axis.
      const int ex = m.Extent(c, 0), ey = m.Extent(c, 1);
      const int ez = (DIM == 2) ? 1 : m.Extent(c, 2);
      const int off = m.Offset(c);
      for (int dz = 0; dz < ez; dz++)
      {
         // Sweep along the z-axis for the current element slice.
         MFEM_FOREACH_THREAD(qy, y, q1d)
         MFEM_FOREACH_THREAD(qx, x, q1d)
         {
            real_t value = 0.0;
            if constexpr (DIM == 2)
            { value = vector_src<FOP>(rarg, qx, qy, 0, vt); }
            else
            {
               for (int qz = 0; qz < q1d; qz++)
               {
                  value += Bz[qz + q1d * dz] *
                           vector_src<FOP>(rarg, qx, qy, qz, vt);
               }
            }
            sm0[qy][qx] = value;
         }
         MFEM_SYNC_THREAD;
         // Sweep along the y-axis for the current z-slice.
         MFEM_FOREACH_THREAD(dy, y, ey)
         MFEM_FOREACH_THREAD(qx, x, q1d)
         {
            real_t value = 0.0;
            for (int qy = 0; qy < q1d; qy++)
            {
               value += By[qy + q1d * dy] * sm0[qy][qx];
            }
            sm1[dy][qx] = value;
         }
         MFEM_SYNC_THREAD;
         // Sweep along the x-axis for the current y-slice.
         MFEM_FOREACH_THREAD(dy, y, ey)
         MFEM_FOREACH_THREAD(dx, x, ex)
         {
            real_t value = 0.0;
            for (int qx = 0; qx < q1d; qx++)
            {
               value += Bx[qx + q1d * dx] * sm1[dy][qx];
            }
            YE(off + dx + ex * (dy + ey * dz), 0, 0, 0, e) += value;
         }
         MFEM_SYNC_THREAD;
      }
   }

public:

   template<int RNK,
            typename ArgRegT,
            typename YE_T,
            typename FieldParamT = ArgRegT>
   static MFEM_HOST_DEVICE void write_gradient(Shared &s,
                                               const int e,
                                               const int d,
                                               const int q,
                                               const real_t *B,
                                               const real_t *G,
                                               YE_T &YE,
                                               ArgRegT &rarg)
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
                  ker::VectorGradTranspose2d(
                     d, q, c, s.B, s.G, s.M[0], s.M[1], rarg);
                  ker::WriteGradDofs2d(d, c, e, rarg, YE);
               }
               else
               {
                  ker::VectorGradTranspose3d(
                     d, q, c, s.B, s.G, s.M[0], s.M[1], rarg);
                  ker::WriteGradDofs3d(d, c, e, rarg, YE);
               }
            }
         }
      }
      else
      {
         static_assert(false, "Unsupported");
      }
   }
};

// ────────────────────────────────────────────────────────────────────────────
template<int T_DIM, int T_Q1D = LocalQFLOBackendMQ1(), int T_ZTHREADS = T_Q1D>
struct LocalQFLOBackend
{
   // ─────────────────────────────────────────────────────
   static constexpr int DIM = T_DIM, MQ1 = T_Q1D, Q1D = T_Q1D;
   static constexpr int ZTHREADS = T_ZTHREADS;
   static_assert(DIM == 2 || DIM == 3);
   static_assert(ZTHREADS > 0 && ZTHREADS <= Q1D);

   // ─────────────────────────────────────────────────────
   static inline ThreadBlocks thread_blocks(const int q1d)
   {
      MFEM_ASSERT(q1d <= Q1D, "q1d must be <= " << Q1D);
      return { q1d, q1d, (DIM == 2) ? 1 : std::min(q1d, ZTHREADS) };
   }

   // ─────────────────────────────────────────────────────
   static inline constexpr int MAX_THREADS_PER_BLOCK()
   { return Q1D * Q1D * ((DIM == 2) ? 1 : ZTHREADS); }

   // ─────────────────────────────────────────────────────
   using backend_t = lo_ker_backend<DIM, Q1D, (DIM == 3 && ZTHREADS < Q1D)>;

   // ─────────────────────────────────────────────────────
   using Shared = typename backend_t::Shared;

   // ─────────────────────────────────────────────────────
   template<typename WT, typename WI, typename Cache, typename AddY>
   static MFEM_HOST_DEVICE inline void DiagContract(Shared &s,
                                                    const int ndx,
                                                    const int ndy,
                                                    const int ndz_in,
                                                    const int q1d,
                                                    WT wt,
                                                    WI wi,
                                                    Cache cache,
                                                    AddY add_y)
   {
      MFEM_CONTRACT_VAR(ndz_in);
      real_t *base = reinterpret_cast<real_t *>(&s.M[0]);
      auto s0 = reinterpret_cast<real_t(*)[Q1D][Q1D]>(base);

      if constexpr (DIM == 3)
      {
         auto s1 =
            reinterpret_cast<real_t(*)[Q1D][Q1D]>(base + Q1D * Q1D * Q1D);

         // reduce qz → dz : s0[dz][qy][qx]
         MFEM_FOREACH_THREAD(dz, z, ndz_in)
         MFEM_FOREACH_THREAD(qy, y, q1d)
         MFEM_FOREACH_THREAD(qx, x, q1d)
         {
            real_t u = 0.0;
            for (int qz = 0; qz < q1d; qz++)
            {
               const int q = qx + (qy + qz * q1d) * q1d;
               u += wt(2, qz, dz) * wi(2, qz, dz) * cache(q);
            }
            s0[dz][qy][qx] = u;
         }
         MFEM_SYNC_THREAD;

         // reduce qy → dy : s1[dz][dy][qx]
         MFEM_FOREACH_THREAD(dz, z, ndz_in)
         MFEM_FOREACH_THREAD(dy, y, ndy)
         MFEM_FOREACH_THREAD(qx, x, q1d)
         {
            real_t u = 0.0;
            for (int qy = 0; qy < q1d; qy++)
            {
               u += wt(1, qy, dy) * wi(1, qy, dy) * s0[dz][qy][qx];
            }
            s1[dz][dy][qx] = u;
         }
         MFEM_SYNC_THREAD;

         // reduce qx → dx : Y(dx,dy,dz)
         MFEM_FOREACH_THREAD(dz, z, ndz_in)
         MFEM_FOREACH_THREAD(dy, y, ndy)
         MFEM_FOREACH_THREAD(dx, x, ndx)
         {
            real_t u = 0.0;
            for (int qx = 0; qx < q1d; qx++)
            {
               u += wt(0, qx, dx) * wi(0, qx, dx) * s1[dz][dy][qx];
            }
            add_y(dx, dy, dz, u);
         }
         MFEM_SYNC_THREAD;
      }
      else
      {
         // reduce qy → dy : s0[0][dy][qx]
         MFEM_FOREACH_THREAD(dy, y, ndy)
         MFEM_FOREACH_THREAD(qx, x, q1d)
         {
            real_t u = 0.0;
            for (int qy = 0; qy < q1d; qy++)
            {
               const int q = qx + qy * q1d;
               u += wt(1, qy, dy) * wi(1, qy, dy) * cache(q);
            }
            s0[0][dy][qx] = u;
         }
         MFEM_SYNC_THREAD;

         // reduce qx → dx : Y(dx,dy,0)
         MFEM_FOREACH_THREAD(dy, y, ndy)
         MFEM_FOREACH_THREAD(dx, x, ndx)
         {
            real_t u = 0.0;
            for (int qx = 0; qx < q1d; qx++)
            {
               u += wt(0, qx, dx) * wi(0, qx, dx) * s0[0][dy][qx];
            }
            add_y(dx, dy, 0, u);
         }
         MFEM_SYNC_THREAD;
      }
   }

   // ─────────────────────────────────────────────────────
   template<typename T>
   using QReg = lo_qreg_t<backend_t, T>;

   // ─────────────────────────────────────────────────────
   /// Interpolate a field to quadrature points.
   /// Since we dispatch based on the DofToQuadMap, we can handle both
   /// scalar and vector FE.
   template<typename ArgRegT, typename XE_T>
   static inline MFEM_HOST_DEVICE void LoadValue(Shared &s,
                                                 const int e,
                                                 const DofToQuadMap &dtq,
                                                 const XE_T &XE,
                                                 ArgRegT &rarg)
   {
      if (dtq.IsVectorFE())
      {
         backend_t::template load_vector<Value<>>(s, e, dtq, XE, rarg);
         return;
      }
      backend_t::template load_value<ArgRegT>(
         s, e, dtq.D1D(), dtq.Q1D(), dtq.B, XE, rarg);
   }

   // ─────────────────────────────────────────────────────
   /// Evaluate the reference divergence of a field at quadrature points.
   template<typename ArgRegT, typename XE_T>
   static inline MFEM_HOST_DEVICE void LoadDiv(Shared &s,
                                               const int e,
                                               const DofToQuadMap &dtq,
                                               const XE_T &XE,
                                               ArgRegT &rarg)
   {
      backend_t::template load_vector<Div<>>(s, e, dtq, XE, rarg);
   }

   // ─────────────────────────────────────────────────────
   template<int RNK,
            typename ArgRegT,
            typename XE_T,
            typename FieldParamT = ArgRegT>
   static inline MFEM_HOST_DEVICE void LoadGradient(Shared &s,
                                                    const int e,
                                                    const DofToQuadMap &dtq,
                                                    const XE_T &XE,
                                                    ArgRegT &rarg)
   {
      backend_t::template load_gradient<RNK, ArgRegT, XE_T, FieldParamT>(
         s, e, dtq.D1D(), dtq.Q1D(), dtq.B, dtq.G, XE, rarg);
   }

   // ─────────────────────────────────────────────────────
   template<typename T>
   static MFEM_HOST_DEVICE inline auto
   qp_pull(QReg<T> &reg, int qx, int qy, int qz)
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
         else
         {
            static_assert(false, "Unsupported");
         }
      }
   }

   // ─────────────────────────────────────────────────────
   template<typename T>
   static MFEM_HOST_DEVICE inline auto qp_pull_directional(
      QReg<T> &preg, QReg<T> &sreg, int qx, int qy, int qz, bool dependent)
   { return lok::pull_directional<DIM, T>(preg, sreg, qx, qy, qz, dependent); }

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
            as_tensor<real_t, e0, e1>(&lok::at<DIM>(reg, qx, qy, qz)[0][0]) =
               out;
         }
         else
         {
            static_assert(false, "Unsupported");
         }
      }
   }

   // ─────────────────────────────────────────────────────
   template<typename T>
   static MFEM_HOST_DEVICE inline void
   qp_push_tangent(QReg<T> &reg, int qx, int qy, int qz, const T &out)
   {
      if constexpr (!qf_param_uses_dual_v<T>)
      {
         qp_push<T>(reg, qx, qy, qz, out);
      }
      else
      {
         lok::store_at<DIM, T, decltype(reg), true>(reg, qx, qy, qz, out);
      }
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
   /// Integrate qp scalars against test basis functions.
   /// Since we dispatch based on the DofToQuadMap, we can deal with both
   /// scalar and vector FE.
   template<typename ArgRegT, typename YE_T>
   static inline MFEM_HOST_DEVICE void WriteValue(Shared &s,
                                                  const int e,
                                                  const DofToQuadMap &dtq,
                                                  const YE_T &YE,
                                                  ArgRegT &rarg)
   {
      if (dtq.IsVectorFE())
      {
         backend_t::template write_vector<Value<>>(s, e, dtq, YE, rarg);
         return;
      }
      backend_t::write_value(s, e, dtq.D1D(), dtq.Q1D(), dtq.B, YE, rarg);
   }

   // ─────────────────────────────────────────────────────
   /// Integrate qp scalars against divergence of test basis.
   template<typename ArgRegT, typename YE_T>
   static inline MFEM_HOST_DEVICE void WriteDiv(Shared &s,
                                                const int e,
                                                const DofToQuadMap &dtq,
                                                const YE_T &YE,
                                                ArgRegT &rarg)
   {
      backend_t::template write_vector<Div<>>(s, e, dtq, YE, rarg);
   }

   // ─────────────────────────────────────────────────────
   template<int RNK,
            typename ArgRegT,
            typename YE_T,
            typename FieldParamT = ArgRegT>
   static inline MFEM_HOST_DEVICE void WriteGradient(Shared &s,
                                                     const int e,
                                                     const DofToQuadMap &dtq,
                                                     YE_T &YE,
                                                     ArgRegT &rarg)
   {
      backend_t::template write_gradient<RNK, ArgRegT, YE_T, FieldParamT>(
         s, e, dtq.D1D(), dtq.Q1D(), dtq.B, dtq.G, YE, rarg);
   }
};

/// @brief Dispatch to a compile-time LO kernel matching runtime @a q1d.
template <typename LOKernelTable, int DIM, int MQ1 = LocalQFLOBackendMQ1()>
inline typename LOKernelTable::KernelSignature
DispatchLOKernelByQ1D(int q1d)
{
   MFEM_VERIFY(q1d >= 2 && q1d <= MQ1,
               "Unsupported LO quadrature order: " << q1d);
   switch (q1d)
   {
      case 2: return LOKernelTable::template Kernel<DIM, 2>();
      case 3:
         if constexpr (MQ1 >= 3)
         {
            return LOKernelTable::template Kernel<DIM, 3>();
         }
         break;
      case 4:
         if constexpr (MQ1 >= 4)
         {
            return LOKernelTable::template Kernel<DIM, 4>();
         }
         break;
      case 5:
         if constexpr (MQ1 >= 5)
         {
            return LOKernelTable::template Kernel<DIM, 5>();
         }
         break;
      case 6:
         if constexpr (MQ1 >= 6)
         {
            return LOKernelTable::template Kernel<DIM, 6>();
         }
         break;
      case 7:
         if constexpr (MQ1 >= 7)
         {
            return LOKernelTable::template Kernel<DIM, 7>();
         }
         break;
      case 8:
         if constexpr (MQ1 >= 8)
         {
            return LOKernelTable::template Kernel<DIM, 8>();
         }
         break;
      default: return nullptr;
   }
   return nullptr;
}

/// @brief Select the compile-time LO kernel for runtime @a dim.
///
/// QFDIM is deduced at compile-time from the q-function signature.
/// If it is not possible to deduce the dimension (QFDIM=0), we fallback
/// to the original runtime dispatching, which means we emit both 2D and 3D
/// branches.
template <typename LOKernelTable, int QFDIM, int MQ1 = LocalQFLOBackendMQ1()>
inline typename LOKernelTable::KernelSignature
DispatchLOKernelByDim(int dim, int q1d)
{
   if constexpr (QFDIM == 2 || QFDIM == 3)
   {
      MFEM_VERIFY(dim == QFDIM,
                  "mesh dimension " << dim << " does not match the " << QFDIM
                  << "D q-function signature this integrator was built from");
      return DispatchLOKernelByQ1D<LOKernelTable, QFDIM, MQ1>(q1d);
   }
   else
   {
      // Couldn't deduce the dimension from the q-function signature,
      // we fall back to original runtime dispatching.
      if (dim == 2)
      {
         return DispatchLOKernelByQ1D<LOKernelTable, 2, MQ1>(q1d);
      }
      if (dim == 3)
      {
         return DispatchLOKernelByQ1D<LOKernelTable, 3, MQ1>(q1d);
      }
      MFEM_ABORT("Unsupported dimension " << dim);
      return nullptr;
   }
}

} // namespace mfem::future
