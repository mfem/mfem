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

#include <cstddef>
#include <type_traits>

// Typed per-QP register layouts derived from decayed q-function parameter types,
// for both the LO and HO backends.

#include "fem/kernels.hpp"   // ker::v_regs3d_t, ker::vd_regs3d_t (HO layout)
#include "fem/kernels3d.hpp" // low::regs3d_t, low::regs3d_vd_t (LO layout)
#include "qf_local_data.hpp"

#include "../util.hpp" // for as_tensor

namespace mfem::future
{

///////////////////////////////////////////////////////////////////////////////
/// Low-order 3D quadrature register storage for a decayed q-function parameter

template <typename T, int MQ1, int Rank = qf_param_shape<T>::rank>
struct low_order_qp_reg_for_decay
{
   static_assert(Rank >= 0 && Rank <= 2);
};

template <typename T, int MQ1>
struct low_order_qp_reg_for_decay<T, MQ1, 0>
{
   using type = mfem::kernels::internal::low::regs3d_t<1, MQ1>;
};

template <typename T, int MQ1>
struct low_order_qp_reg_for_decay<T, MQ1, 1>
{
   static constexpr int e0 = qf_param_shape<T>::extents[0];
   using type = mfem::kernels::internal::low::regs3d_t<e0, MQ1>;
};

template <typename T, int MQ1>
struct low_order_qp_reg_for_decay<T, MQ1, 2>
{
   static constexpr int e0 = qf_param_shape<T>::extents[0];
   static constexpr int e1 = qf_param_shape<T>::extents[1];
   using type = mfem::kernels::internal::low::regs3d_vd_t<e0, e1, MQ1>;
};

template <typename T, int MQ1>
using lo_qp_reg_for_decay_t =
   typename low_order_qp_reg_for_decay<T, MQ1>::type;

template <typename T, int MQ1>
MFEM_HOST_DEVICE auto lo_input_qp_reg_as_arg_at(
   lo_qp_reg_for_decay_t<T, MQ1> &reg, int qx, int qy, int qz)
{
   constexpr int R = qf_param_shape<T>::rank;
   if constexpr (R == 0)
   {
      return as_tensor<real_t>(&reg[qz][qy][qx][0]);
   }
   else if constexpr (R == 1)
   {
      constexpr int e0 = qf_param_shape<T>::extents[0];
      return as_tensor<real_t, e0>(&reg[qz][qy][qx][0]);
   }
   else if constexpr (R == 2)
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

/// Write q-argument value `out` (`T` at QP) into the LO register block at (qz,qy,qx).
template <typename T, int MQ1>
MFEM_HOST_DEVICE void lo_output_qp_reg_assign_at(
   lo_qp_reg_for_decay_t<T, MQ1> &reg,
   int qx, int qy, int qz, const T &out)
{
   constexpr int R = qf_param_shape<T>::rank;
   if constexpr (R == 0)
   {
      as_tensor<real_t>(&reg[qz][qy][qx][0]) = out;
   }
   else if constexpr (R == 1)
   {
      constexpr int e0 = qf_param_shape<T>::extents[0];
      as_tensor<real_t, e0>(&reg[qz][qy][qx][0]) = out;
   }
   else if constexpr (R == 2)
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

///////////////////////////////////////////////////////////////////////////////
/// High-order 3D quadrature register storage for a q-function parameter

template <typename T, int MQ1, int Rank = qf_param_shape<T>::rank>
struct ho_qp_reg_for_decay
{
   static_assert(Rank >= 0 && Rank <= 2);
};

template <typename T, int MQ1>
struct ho_qp_reg_for_decay<T, MQ1, 0>
{
   using type = mfem::kernels::internal::v_regs3d_t<1, MQ1>;
};

template <typename T, int MQ1>
struct ho_qp_reg_for_decay<T, MQ1, 1>
{
   static constexpr int e0 = qf_param_shape<T>::extents[0];
   using type = mfem::kernels::internal::vd_regs3d_t<1, e0, MQ1>;
};

template <typename T, int MQ1>
struct ho_qp_reg_for_decay<T, MQ1, 2>
{
   static constexpr int e0 = qf_param_shape<T>::extents[0];
   static constexpr int e1 = qf_param_shape<T>::extents[1];
   using type = mfem::kernels::internal::vd_regs3d_t<e0, e1, MQ1>;
};

template <typename T, int MQ1>
using ho_qp_reg_for_decay_t = typename ho_qp_reg_for_decay<T, MQ1>::type;

template <typename T, int MQ1>
MFEM_HOST_DEVICE auto ho_input_qp_reg_as_arg_at(
   ho_qp_reg_for_decay_t<T, MQ1> &reg,
   int qz, int qy, int qx)
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

/// Scatter a QP value `out` of type `T` into slice `(qz, qy, qx)`
template <typename T, int MQ1>
MFEM_HOST_DEVICE void ho_output_qp_reg_assign_at(
   ho_qp_reg_for_decay_t<T, MQ1> &reg,
   int qz, int qy, int qx,
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

///////////////////////////////////////////////////////////////////////////////
template <typename backend_t, typename qfunc_t, typename inputs_t, typename outputs_t,
          int MQ1, std::size_t K, std::size_t N, typename... Acc>
struct build_args_reg_tuple_impl;

template <typename backend_t, typename qfunc_t, typename inputs_t, typename outputs_t,
          int MQ1, std::size_t N, typename... Acc>
struct build_args_reg_tuple_impl<backend_t, qfunc_t, inputs_t, outputs_t, MQ1, N, N, Acc...>
{
   using type = tuple<Acc...>;
   static_assert(sizeof...(Acc) == N);
   static_assert(sizeof...(Acc) <= 9);
};

template <typename backend_t, typename qfunc_t, typename inputs_t, typename outputs_t,
          int MQ1, std::size_t K, std::size_t N, typename... Acc>
struct build_args_reg_tuple_impl
{
   using R = typename backend_t::template QPReg<
                typename qf_param_slot<qfunc_t, K>::decay_t, MQ1>;
   using type = typename build_args_reg_tuple_impl<backend_t, qfunc_t, inputs_t,
         outputs_t, MQ1, K + 1, N, Acc..., R>::type;
};

template <typename backend_t, typename qfunc_t, typename inputs_t, typename outputs_t,
          int MQ1>
using args_reg_t = typename build_args_reg_tuple_impl<backend_t, qfunc_t,
      inputs_t, outputs_t, MQ1, 0,
      tuple_size<inputs_t>::value + tuple_size<outputs_t>::value>::type;

} // namespace mfem::future
