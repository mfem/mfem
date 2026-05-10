#pragma once

// Typed low-order PA register layouts derived from decayed q-function parameter types.

#include "fem/kernels3d.hpp"
#include "qf_local_data.hpp"

#include <cstddef>
#include <type_traits>

#include "../util.hpp" // for as_tensor

namespace mfem::future
{

/// Low-order 3D quadrature register storage for a decayed q-function parameter
template <typename DecayT, int MQ1, int Rank = qf_param_tensor_extents<DecayT>::rank>
struct low_order_qp_reg_for_decay
{
   static_assert(Rank >= 0 && Rank <= 2,
                 "low_order_qp_reg_for_decay: rank > 2 not supported for LO registers");
};

template <typename DecayT, int MQ1>
struct low_order_qp_reg_for_decay<DecayT, MQ1, 0>
{
   using type = mfem::kernels::internal::low::regs3d_t<1, MQ1>;
};

template <typename DecayT, int MQ1>
struct low_order_qp_reg_for_decay<DecayT, MQ1, 1>
{
private:
   static constexpr int e0 = qf_param_tensor_extents<DecayT>::extents[0];

public:
   using type = mfem::kernels::internal::low::regs3d_t<e0, MQ1>;
};

template <typename DecayT, int MQ1>
struct low_order_qp_reg_for_decay<DecayT, MQ1, 2>
{
private:
   static constexpr int e0 = qf_param_tensor_extents<DecayT>::extents[0];
   static constexpr int e1 = qf_param_tensor_extents<DecayT>::extents[1];

public:
   using type = mfem::kernels::internal::low::regs3d_vd_t<e0, e1, MQ1>;
};

template <typename DecayT, int MQ1>
using low_order_qp_reg_for_decay_t =
   typename low_order_qp_reg_for_decay<DecayT, MQ1>::type;

/// Quadrature-point value for q-argument type `DecayT`, read from the matching LO register block.
template <typename DecayT, int MQ1>
MFEM_HOST_DEVICE auto input_qp_reg_as_arg_at(
   low_order_qp_reg_for_decay_t<DecayT, MQ1> &reg, int qz, int qy, int qx)
{
   constexpr int R = qf_param_tensor_extents<DecayT>::rank;
   if constexpr (R == 0)
   {
      if constexpr (std::is_same_v<DecayT, real_t>)
      {
         return reg(qz, qy, qx, 0);
      }
      else
      {
         return DecayT {reg(qz, qy, qx, 0)};
      }
   }
   else if constexpr (R == 1)
   {
      constexpr int e0 = qf_param_tensor_extents<DecayT>::extents[0];
      return as_tensor<real_t, e0>(&reg(qz, qy, qx, 0));
   }
   else
   {
      constexpr int e0 = qf_param_tensor_extents<DecayT>::extents[0];
      constexpr int e1 = qf_param_tensor_extents<DecayT>::extents[1];
      return as_tensor<real_t, e0, e1>(&reg(qz, qy, qx, 0, 0));
   }
}

/// Write q-argument value `out` (`DecayT` at QP) into the LO register block at (qz,qy,qx).
template <typename DecayT, int MQ1>
MFEM_HOST_DEVICE void output_qp_reg_assign_at(
   low_order_qp_reg_for_decay_t<DecayT, MQ1> &reg,
   int qz, int qy, int qx, const DecayT &out)
{
   constexpr int R = qf_param_tensor_extents<DecayT>::rank;
   if constexpr (R == 0)
   {
      if constexpr (std::is_same_v<DecayT, real_t>)
      {
         reg(qz, qy, qx, 0) = out;
      }
      else
      {
         reg(qz, qy, qx, 0) = out.scalar();
      }
   }
   else if constexpr (R == 1)
   {
      constexpr int e0 = qf_param_tensor_extents<DecayT>::extents[0];
      as_tensor<real_t, e0>(&reg(qz, qy, qx, 0)) = out;
   }
   else
   {
      constexpr int e0 = qf_param_tensor_extents<DecayT>::extents[0];
      constexpr int e1 = qf_param_tensor_extents<DecayT>::extents[1];
      as_tensor<real_t, e0, e1>(&reg(qz, qy, qx, 0, 0)) = out;
   }
}

namespace kern_low = mfem::kernels::internal::low;

/// `tuple<R0, …, R{n-1}>` with `Rk = low_order_qp_reg_for_decay_t< qf_decay_param_t<k>, MQ1 >`
/// for full q-function arity **`n = n_inputs + n_outputs`** (parameter index **k** equals
/// input slot for `k < n_inputs`, else output slot `k - n_inputs`).
template <typename qfunc_t, typename inputs_t, typename outputs_t, int MQ1, std::size_t K,
          std::size_t N, typename... Acc>
struct build_args_reg_tuple_impl;

template <typename qfunc_t, typename inputs_t, typename outputs_t, int MQ1, std::size_t N,
          typename... Acc>
struct build_args_reg_tuple_impl<qfunc_t, inputs_t, outputs_t, MQ1, N, N, Acc...>
{
   using type = tuple<Acc...>;
   static_assert(sizeof...(Acc) == N);
   static_assert(sizeof...(Acc) <= 9,
                 "mfem::future::tuple supports at most 9 elements for this use case");
};

template <typename qfunc_t, typename inputs_t, typename outputs_t, int MQ1, std::size_t K,
          std::size_t N, typename... Acc>
struct build_args_reg_tuple_impl
{
   using R = low_order_qp_reg_for_decay_t<
             typename LocalQFArgMetadata<qfunc_t, inputs_t, outputs_t>::template
             qf_decay_param_t<K>, MQ1>;
   using type = typename build_args_reg_tuple_impl<qfunc_t, inputs_t,
         outputs_t, MQ1,
         K + 1, N, Acc..., R>::type;
};

template <typename qfunc_t, typename inputs_t, typename outputs_t, int MQ1>
using args_reg_t =
   typename build_args_reg_tuple_impl<qfunc_t, inputs_t, outputs_t, MQ1, 0,
   tuple_size<inputs_t>::value + tuple_size<outputs_t>::value>::type;

} // namespace mfem::future
