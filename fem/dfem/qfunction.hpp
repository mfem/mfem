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

#include "util.hpp"
#include "../../linalg/tensor.hpp"

namespace mfem::experimental
{

template <typename func_t, typename... arg_ts>
MFEM_HOST_DEVICE inline
auto qfunction_wrapper(const func_t &f, arg_ts &&...args)
{
   return f(args...);
}

template <typename T0, typename T1>
MFEM_HOST_DEVICE inline
void process_kf_arg(const T0 &, T1 &)
{
   static_assert(always_false<T0, T1>,
                 "process_kf_arg not implemented for arg type");
}

template <typename T>
MFEM_HOST_DEVICE inline
void process_kf_arg(
   const DeviceTensor<1, T> &u,
   T &arg)
{
   arg = u(0);
}

template <typename T>
MFEM_HOST_DEVICE inline
void process_kf_arg(
   const DeviceTensor<1, T> &u,
   internal::tensor<T> &arg)
{
   arg(0) = u(0);
}

template <typename T, int n>
MFEM_HOST_DEVICE inline
void process_kf_arg(
   const DeviceTensor<1> &u,
   internal::tensor<T, n> &arg)
{
   for (int i = 0; i < n; i++)
   {
      arg(i) = u(i);
   }
}

template <typename T, int n, int m>
MFEM_HOST_DEVICE inline
void process_kf_arg(
   const DeviceTensor<1> &u,
   internal::tensor<T, n, m> &arg)
{
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         arg(j, i) = u((i * m) + j);
      }
   }
}

template <typename arg_type>
MFEM_HOST_DEVICE inline
void process_kf_arg(const DeviceTensor<2> &u, arg_type &arg, int qp)
{
   const auto u_qp = Reshape(&u(0, qp), u.GetShape()[0]);
   process_kf_arg(u_qp, arg);
}

template <size_t num_fields, typename kf_args>
MFEM_HOST_DEVICE inline
void process_kf_args(
   const std::array<DeviceTensor<2>, num_fields> &u,
   kf_args &args,
   const int &qp)
{
   for_constexpr<tuple_size<kf_args>::value>([&](auto i)
   {
      process_kf_arg(u[i], get<i>(args), qp);
   });
}

template <typename T0, typename T1>
MFEM_HOST_DEVICE inline
Vector process_kf_result(T0, T1)
{
   static_assert(always_false<T0, T1>,
                 "process_kf_result not implemented for result type");
   return Vector{};
}

template <typename T>
MFEM_HOST_DEVICE inline
void process_kf_result(
   DeviceTensor<1, T> &r,
   const double &x)
{
   r(0) = x;
}

template <typename T>
MFEM_HOST_DEVICE inline
void process_kf_result(
   DeviceTensor<1, T> &r,
   const internal::tensor<T> &x)
{
   r(0) = x(0);
}

template <typename T, int n>
MFEM_HOST_DEVICE inline
void process_kf_result(
   DeviceTensor<1, T> &r,
   const internal::tensor<T, n> &x)
{
   for (size_t i = 0; i < n; i++)
   {
      r(i) = x(i);
   }
}

template <typename T, int n, int m>
MFEM_HOST_DEVICE inline
void process_kf_result(
   DeviceTensor<1, T> &r,
   const internal::tensor<T, n, m> &x)
{
   for (size_t i = 0; i < n; i++)
   {
      for (size_t j = 0; j < m; j++)
      {
         r(i + n * j) = x(i, j);
      }
   }
}

template <typename T>
MFEM_HOST_DEVICE inline
void process_kf_arg(
   const DeviceTensor<1> &u,
   const DeviceTensor<1> &v,
   double &arg)
{
   arg = u(0);
}

template <int n, int m>
MFEM_HOST_DEVICE inline
void process_kf_arg(
   const DeviceTensor<1> &u,
   const DeviceTensor<1> &v,
   internal::tensor<double, n, m> &arg)
{
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         arg(j, i) = u((i * m) + j);
      }
   }
}

template <typename qfunc_t, typename args_ts, size_t num_args>
MFEM_HOST_DEVICE inline
void apply_kernel(
   DeviceTensor<1, double> &f_qp,
   const qfunc_t &qfunc,
   args_ts &args,
   const std::array<DeviceTensor<2>, num_args> &u,
   int qp)
{
   process_kf_args(u, args, qp);
   process_kf_result(f_qp, get<0>(apply(qfunc, args)));
}

#ifdef MFEM_USE_ENZYME
// Version for active function arguments only
//
// This is an Enzyme regression and can be removed in later versions.
template <typename qfunc_t, typename arg_ts, std::size_t... Is,
          typename inactive_arg_ts>
MFEM_HOST_DEVICE inline
auto fwddiff_apply_enzyme_indexed(qfunc_t &qfunc, arg_ts &&args,
                                  arg_ts &&shadow_args,
                                  std::index_sequence<Is...>,
                                  inactive_arg_ts &&inactive_args,
                                  std::index_sequence<>)
{
   using qf_return_t = typename create_function_signature<
                       decltype(&qfunc_t::operator())>::type::return_t;
   return __enzyme_fwddiff<qf_return_t>(
             qfunction_wrapper<qfunc_t, decltype(get<Is>(args))...>, enzyme_const,
             (void *)&qfunc, enzyme_dup, &get<Is>(args)..., enzyme_interleave,
             &get<Is>(shadow_args)...);
}

// Interleave function arguments for enzyme
template <typename qfunc_t, typename arg_ts, std::size_t... Is,
          typename inactive_arg_ts, std::size_t... Js>
MFEM_HOST_DEVICE inline
auto fwddiff_apply_enzyme_indexed(qfunc_t &qfunc, arg_ts &&args,
                                  arg_ts &&shadow_args,
                                  std::index_sequence<Is...>,
                                  inactive_arg_ts &&inactive_args,
                                  std::index_sequence<Js...>)
{
   using qf_return_t = typename create_function_signature<
                       decltype(&qfunc_t::operator())>::type::return_t;
   return __enzyme_fwddiff<qf_return_t>(
             qfunction_wrapper<qfunc_t, decltype(get<Is>(args))...,
             decltype(get<Js>(inactive_args))...>,
             enzyme_const, (void *)&qfunc, enzyme_dup, &get<Is>(args)...,
             enzyme_const, &get<Js>(inactive_args)..., enzyme_interleave,
             &get<Is>(shadow_args)...);
}

template <typename qfunc_t, typename arg_ts, typename inactive_arg_ts>
MFEM_HOST_DEVICE inline
auto fwddiff_apply_enzyme(qfunc_t &qfunc, arg_ts &&args,
                          arg_ts &&shadow_args,
                          inactive_arg_ts &&inactive_args)
{
   auto arg_indices = std::make_index_sequence<
                      tuple_size<std::remove_reference_t<arg_ts>>::value> {};

   auto inactive_arg_indices = std::make_index_sequence<
                               tuple_size<std::remove_reference_t<inactive_arg_ts>>::value> {};

   return fwddiff_apply_enzyme_indexed(qfunc, args, shadow_args, arg_indices,
                                       inactive_args, inactive_arg_indices);
}

template <typename qfunc_t, typename arg_ts, size_t num_args>
MFEM_HOST_DEVICE inline
void apply_kernel_fwddiff_enzyme(
   DeviceTensor<1, double> &f_qp,
   qfunc_t &qfunc,
   arg_ts &args,
   arg_ts &shadow_args,
   const std::array<DeviceTensor<2>, num_args> &u,
   const std::array<DeviceTensor<2>, num_args> &v,
   int qp_idx)
{
   process_kf_args(u, args, qp_idx);
   process_kf_args(v, shadow_args, qp_idx);
   process_kf_result(f_qp,
                     get<0>(fwddiff_apply_enzyme(qfunc, args, shadow_args, tuple<> {})));
}
#endif // MFEM_USE_ENZYME

} // namespace mfem
