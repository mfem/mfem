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
#include "qfunction_transform.hpp"

namespace mfem::future
{

/// @brief Call a qfunction with the given parameters.
///
/// @param qfunc the qfunction to call.
/// @param input_shmem the input shared memory.
/// @param residual_shmem the residual shared memory.
/// @param rs_qp the size of the residual.
/// @param num_qp the number of quadrature points.
/// @param q1d the number of quadrature points in 1D.
/// @param dimension the spatial dimension.
/// @param use_sum_factorization whether to use sum factorization.
/// @tparam qf_param_ts the tuple type of the qfunction parameters.
template <
   typename qf_param_ts,
   typename qfunc_t,
   std::size_t num_fields>
MFEM_HOST_DEVICE inline
void call_qfunction(
   qfunc_t &qfunc,
   const std::array<DeviceTensor<2>, num_fields> &input_shmem,
   DeviceTensor<2> &residual_shmem,
   const int &rs_qp,
   const int &num_qp,
   const int &q1d,
   const int &dimension,
   const bool &use_sum_factorization)
{
   if (use_sum_factorization)
   {
      if (dimension == 1)
      {
         MFEM_FOREACH_THREAD_DIRECT(q, x, q1d)
         {
            auto qf_args = decay_tuple<qf_param_ts> {};
            auto r = Reshape(&residual_shmem(0, q), rs_qp);
            apply_kernel(r, qfunc, qf_args, input_shmem, q);
         }
      }
      else if (dimension == 2)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
            {
               const int q = qx + q1d * qy;
               auto qf_args = decay_tuple<qf_param_ts> {};
               auto r = Reshape(&residual_shmem(0, q), rs_qp);
               apply_kernel(r, qfunc, qf_args, input_shmem, q);
            }
         }
      }
      else if (dimension == 3)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
            {
               MFEM_FOREACH_THREAD_DIRECT(qz, z, q1d)
               {
                  const int q = qx + q1d * (qy + q1d * qz);
                  auto qf_args = decay_tuple<qf_param_ts> {};
                  auto r = Reshape(&residual_shmem(0, q), rs_qp);
                  apply_kernel(r, qfunc, qf_args, input_shmem, q);
               }
            }
         }
      }
      else
      {
#if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
         MFEM_ABORT("unsupported dimension for sum factorization");
#endif
      }
      MFEM_SYNC_THREAD;
   }
   else
   {
      MFEM_FOREACH_THREAD_DIRECT(q, x, num_qp)
      {
         auto qf_args = decay_tuple<qf_param_ts> {};
         auto r = Reshape(&residual_shmem(0, q), rs_qp);
         apply_kernel(r, qfunc, qf_args, input_shmem, q);
      }
   }
}

/// @brief Call a qfunction with the given parameters and
/// compute it's derivative action.
///
/// @param qfunc the qfunction to call.
/// @param input_shmem the input shared memory.
/// @param shadow_shmem the shadow shared memory.
/// @param residual_shmem the residual shared memory.
/// @param das_qp the size of the derivative action.
/// @param num_qp the number of quadrature points.
/// @param q1d the number of quadrature points in 1D.
/// @param dimension the spatial dimension.
/// @param use_sum_factorization whether to use sum factorization.
/// @tparam qf_param_ts the tuple type of the qfunction parameters.
template <
   typename qf_param_ts,
   typename qfunc_t,
   std::size_t num_fields>
MFEM_HOST_DEVICE inline
void call_qfunction_derivative_action(
   qfunc_t &qfunc,
   const std::array<DeviceTensor<2>, num_fields> &input_shmem,
   const std::array<DeviceTensor<2>, num_fields> &shadow_shmem,
   DeviceTensor<2> &residual_shmem,
   const int &das_qp,
   const int &num_qp,
   const int &q1d,
   const int &dimension,
   const bool &use_sum_factorization)
{
   if (use_sum_factorization)
   {
      if (dimension == 1)
      {
         MFEM_FOREACH_THREAD_DIRECT(q, x, q1d)
         {
            auto r = Reshape(&residual_shmem(0, q), das_qp);
            auto qf_args = decay_tuple<qf_param_ts> {};
#ifdef MFEM_USE_ENZYME
            auto qf_shadow_args = decay_tuple<qf_param_ts> {};
            apply_kernel_fwddiff_enzyme(r, qfunc, qf_args, qf_shadow_args, input_shmem,
                                        shadow_shmem, q);
#else
            apply_kernel_native_dual(r, qfunc, qf_args, input_shmem, shadow_shmem, q);
#endif
         }
      }
      else if (dimension == 2)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
            {
               const int q = qx + q1d * qy;
               auto r = Reshape(&residual_shmem(0, q), das_qp);
               auto qf_args = decay_tuple<qf_param_ts> {};
#ifdef MFEM_USE_ENZYME
               auto qf_shadow_args = decay_tuple<qf_param_ts> {};
               apply_kernel_fwddiff_enzyme(r, qfunc, qf_args, qf_shadow_args, input_shmem,
                                           shadow_shmem, q);
#else
               apply_kernel_native_dual(r, qfunc, qf_args, input_shmem, shadow_shmem, q);
#endif
            }
         }
      }
      else if (dimension == 3)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
            {
               MFEM_FOREACH_THREAD_DIRECT(qz, z, q1d)
               {
                  const int q = qx + q1d * (qy + q1d * qz);
                  auto r = Reshape(&residual_shmem(0, q), das_qp);
                  auto qf_args = decay_tuple<qf_param_ts> {};
#ifdef MFEM_USE_ENZYME
                  auto qf_shadow_args = decay_tuple<qf_param_ts> {};
                  apply_kernel_fwddiff_enzyme(r, qfunc, qf_args, qf_shadow_args, input_shmem,
                                              shadow_shmem, q);
#else
                  apply_kernel_native_dual(r, qfunc, qf_args, input_shmem, shadow_shmem, q);
#endif
               }
            }
         }
      }
      else
      {
         MFEM_ABORT_KERNEL("unsupported dimension");
      }
   }
   else
   {
      MFEM_FOREACH_THREAD_DIRECT(q, x, num_qp)
      {
         auto r = Reshape(&residual_shmem(0, q), das_qp);
         auto qf_args = decay_tuple<qf_param_ts> {};
#ifdef MFEM_USE_ENZYME
         auto qf_shadow_args = decay_tuple<qf_param_ts> {};
         apply_kernel_fwddiff_enzyme(r, qfunc, qf_args, qf_shadow_args, input_shmem,
                                     shadow_shmem, q);
#else
         apply_kernel_native_dual(r, qfunc, qf_args, input_shmem, shadow_shmem, q);
#endif
      }
   }
   MFEM_SYNC_THREAD;
}

namespace detail
{
template <
   typename qf_param_ts,
   typename qfunc_t,
   std::size_t num_fields>
MFEM_HOST_DEVICE inline
void call_qfunction_derivative(
   qfunc_t &qfunc,
   const std::array<DeviceTensor<2>, num_fields> &input_shmem,
   const std::array<DeviceTensor<2>, num_fields> &shadow_shmem,
   DeviceTensor<2> &residual_shmem,
   DeviceTensor<5> &qpdc,
   const DeviceTensor<1, const real_t> &itod,
   const int &das_qp,
   const int &q)
{
   const int test_vdim = qpdc.GetShape()[0];
   const int test_op_dim = qpdc.GetShape()[1];
   const int trial_vdim = qpdc.GetShape()[2];
   const int num_qp = qpdc.GetShape()[4];
   const size_t num_inputs = itod.GetShape()[0];

   for (int j = 0; j < trial_vdim; j++)
   {
      int m_offset = 0;
      for (size_t s = 0; s < num_inputs; s++)
      {
         const int trial_op_dim = static_cast<int>(itod(s));
         if (trial_op_dim == 0)
         {
            continue;
         }

         auto d_qp = Reshape(&(shadow_shmem[s])[0], trial_vdim, trial_op_dim, num_qp);
         for (int m = 0; m < trial_op_dim; m++)
         {
            d_qp(j, m, q) = 1.0;

            auto r = Reshape(&residual_shmem(0, q), das_qp);
            auto qf_args = decay_tuple<qf_param_ts> {};
#ifdef MFEM_USE_ENZYME
            auto qf_shadow_args = decay_tuple<qf_param_ts> {};
            apply_kernel_fwddiff_enzyme(r, qfunc, qf_args, qf_shadow_args, input_shmem,
                                        shadow_shmem, q);
#else
            apply_kernel_native_dual(r, qfunc, qf_args, input_shmem, shadow_shmem, q);
#endif
            d_qp(j, m, q) = 0.0;

            auto f = Reshape(&r(0), test_vdim, test_op_dim);
            for (int i = 0; i < test_vdim; i++)
            {
               for (int k = 0; k < test_op_dim; k++)
               {
                  qpdc(i, k, j, m + m_offset, q) = f(i, k);
               }
            }
         }
         m_offset += trial_op_dim;
      }
   }
}
}

/// @brief Call a qfunction with the given parameters and
/// compute it's derivative represented by the Jacobian on
/// each quadrature point.
///
/// @param qfunc the qfunction to call.
/// @param input_shmem the input shared memory.
/// @param shadow_shmem the shadow shared memory.
/// @param residual_shmem the residual shared memory.
/// @param qpdc the quadrature point data cache holding the resulting
/// Jacobians on each quadrature point.
/// @param itod inputs trial operator dimension.
/// If input is dependent the value corresponds to the spatial dimension, otherwise
/// a zero indicates non-dependence on the variable.
/// @param das_qp the size of the derivative action.
/// @param q1d the number of quadrature points in 1D.
/// @param dimension the spatial dimension.
/// @param use_sum_factorization whether to use sum factorization.
/// @tparam qf_param_ts the tuple type of the qfunction parameters.
template <
   typename qf_param_ts,
   typename qfunc_t,
   std::size_t num_fields>
MFEM_HOST_DEVICE inline
void call_qfunction_derivative(
   qfunc_t &qfunc,
   const std::array<DeviceTensor<2>, num_fields> &input_shmem,
   const std::array<DeviceTensor<2>, num_fields> &shadow_shmem,
   DeviceTensor<2> &residual_shmem,
   DeviceTensor<5> &qpdc,
   const DeviceTensor<1, const real_t> &itod,
   const int &das_qp,
   const int &q1d,
   const int &dimension,
   const bool &use_sum_factorization)
{
   if (use_sum_factorization)
   {
      if (dimension == 1)
      {
         MFEM_FOREACH_THREAD_DIRECT(q, x, q1d)
         {
            detail::call_qfunction_derivative<qf_param_ts>(
               qfunc, input_shmem, shadow_shmem, residual_shmem, qpdc, itod, das_qp, q);
         }
      }
      else if (dimension == 2)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
            {
               const int q = qx + q1d * qy;
               detail::call_qfunction_derivative<qf_param_ts>(
                  qfunc, input_shmem, shadow_shmem, residual_shmem, qpdc, itod, das_qp, q);
            }
         }
      }
      else if (dimension == 3)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
            {
               MFEM_FOREACH_THREAD_DIRECT(qz, z, q1d)
               {
                  const int q = qx + q1d * (qy + q1d * qz);
                  detail::call_qfunction_derivative<qf_param_ts>(
                     qfunc, input_shmem, shadow_shmem, residual_shmem, qpdc, itod, das_qp, q);
               }
            }
         }
      }
      else
      {
         MFEM_ABORT_KERNEL("unsupported dimension");
      }
   }
   else
   {
      const int num_qp = qpdc.GetShape()[4];
      MFEM_FOREACH_THREAD_DIRECT(q, x, num_qp)
      {
         detail::call_qfunction_derivative<qf_param_ts>(
            qfunc, input_shmem, shadow_shmem, residual_shmem, qpdc, itod, das_qp, q);
      }
   }
   MFEM_SYNC_THREAD;
}

namespace detail
{

/// @brief Apply the quadrature point data cache (qpdc) to a vector
/// (usually a direction) on quadrature point q.
///
/// The qpdc consists of compatible data to be used for integration with a test
/// operator, e.g. Jacobians of a linearization from a FE operation with a trial
/// function including integration weights and necessesary transformations.
///
/// @param fhat the qpdc applied to a vector in shadow_memory.
/// @param shadow_shmem the shadow shared memory.
/// @param qpdc the quadrature point data cache holding the resulting
/// Jacobians on each quadrature point.
/// @param itod inputs trial operator dimension.
/// If input is dependent the value corresponds to the spatial dimension, otherwise
/// a zero indicates non-dependence on the variable.
/// @param q the current quadrature point index.
template <size_t num_fields>
MFEM_HOST_DEVICE inline
void apply_qpdc(
   DeviceTensor<3> &fhat,
   const std::array<DeviceTensor<2>, num_fields> &shadow_shmem,
   const DeviceTensor<5, const real_t> &qpdc,
   const DeviceTensor<1, const real_t> &itod,
   const int &q)
{
   const int test_vdim = qpdc.GetShape()[0];
   const int test_op_dim = qpdc.GetShape()[1];
   const int trial_vdim = qpdc.GetShape()[2];
   const int num_qp = qpdc.GetShape()[4];
   const size_t num_inputs = itod.GetShape()[0];

   for (int i = 0; i < test_vdim; i++)
   {
      for (int k = 0; k < test_op_dim; k++)
      {
         real_t sum = 0.0;
         int m_offset = 0;
         for (size_t s = 0; s < num_inputs; s++)
         {
            const int trial_op_dim = static_cast<int>(itod(s));
            if (trial_op_dim == 0)
            {
               continue;
            }
            const auto d_qp =
               Reshape(&(shadow_shmem[s])[0], trial_vdim, trial_op_dim, num_qp);
            for (int j = 0; j < trial_vdim; j++)
            {
               for (int m = 0; m < trial_op_dim; m++)
               {
                  sum += qpdc(i, k, j, m + m_offset, q) * d_qp(j, m, q);
               }
            }
            m_offset += trial_op_dim;
         }
         fhat(i, k, q) = sum;
      }
   }
}
}

/// @brief Apply the quadrature point data cache (qpdc) to a vector
/// (usually a direction).
///
/// The qpdc consists of compatible data to be used for integration with a test
/// operator, e.g. Jacobians of a linearization from a FE operation with a trial
/// function including integration weights and necessesary transformations.
///
/// @param fhat the qpdc applied to a vector in shadow_memory.
/// @param shadow_shmem the shadow shared memory.
/// @param qpdc the quadrature point data cache holding the resulting
/// Jacobians on each quadrature point.
/// @param itod inputs trial operator dimension.
/// If input is dependent the value corresponds to the spatial dimension, otherwise
/// a zero indicates non-dependence on the variable.
/// @param q1d number of quadrature points in 1D.
/// @param dimension spatial dimension.
/// @param use_sum_factorization whether to use sum factorization.
template <size_t num_fields>
MFEM_HOST_DEVICE inline
void apply_qpdc(
   DeviceTensor<3> &fhat,
   const std::array<DeviceTensor<2>, num_fields> &shadow_shmem,
   const DeviceTensor<5, const real_t> &qpdc,
   const DeviceTensor<1, const real_t> &itod,
   const int &q1d,
   const int &dimension,
   const bool &use_sum_factorization)
{
   if (use_sum_factorization)
   {
      if (dimension == 1)
      {
         MFEM_FOREACH_THREAD_DIRECT(q, x, q1d)
         {
            detail::apply_qpdc(fhat, shadow_shmem, qpdc, itod, q);
         }
      }
      else if (dimension == 2)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
            {
               const int q = qx + q1d * qy;
               detail::apply_qpdc(fhat, shadow_shmem, qpdc, itod, q);
            }
         }
      }
      else if (dimension == 3)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
            {
               MFEM_FOREACH_THREAD_DIRECT(qz, z, q1d)
               {
                  const int q = qx + q1d * (qy + q1d * qz);
                  detail::apply_qpdc(fhat, shadow_shmem, qpdc, itod, q);
               }
            }
         }
      }
      else
      {
         MFEM_ABORT_KERNEL("unsupported dimension");
      }
   }
   else
   {
      const int num_qp = qpdc.GetShape()[4];
      MFEM_FOREACH_THREAD_DIRECT(q, x, num_qp)
      {
         detail::apply_qpdc(fhat, shadow_shmem, qpdc, itod, q);
      }
   }
}

template <typename qfunc_t, typename args_ts, size_t num_args>
MFEM_HOST_DEVICE inline
void apply_kernel(
   DeviceTensor<1, real_t> &f_qp,
   const qfunc_t &qfunc,
   args_ts &args,
   const std::array<DeviceTensor<2>, num_args> &u,
   int qp)
{
   process_qf_args(u, args, qp);
   process_qf_result(f_qp, get<0>(apply(qfunc, args)));
}

template <typename qfunc_t, typename arg_ts, size_t num_args>
MFEM_HOST_DEVICE inline
void apply_kernel_native_dual(
   DeviceTensor<1, real_t> &f_qp,
   const qfunc_t &qfunc,
   arg_ts &args,
   const std::array<DeviceTensor<2>, num_args> &u,
   const std::array<DeviceTensor<2>, num_args> &v,
   const int &qp_idx)
{
   process_qf_args(u, v, args, qp_idx);
   auto r = get<0>(apply(qfunc, args));
   process_derivative_from_native_dual(f_qp, r);
}

#ifdef MFEM_USE_ENZYME

template <typename func_t, typename... arg_ts>
MFEM_HOST_DEVICE inline
auto qfunction_wrapper(const func_t &f, arg_ts &&...args)
{
   return f(args...);
}

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
   DeviceTensor<1, real_t> &f_qp,
   qfunc_t &qfunc,
   arg_ts &args,
   arg_ts &shadow_args,
   const std::array<DeviceTensor<2>, num_args> &u,
   const std::array<DeviceTensor<2>, num_args> &v,
   int qp_idx)
{
   process_qf_args(u, args, qp_idx);
   process_qf_args(v, shadow_args, qp_idx);
   process_qf_result(f_qp,
                     get<0>(fwddiff_apply_enzyme(qfunc, args, shadow_args, tuple<> {})));
}
template <typename func_t, typename ret_t, typename... arg_ts>
MFEM_HOST_DEVICE inline
void qfunction_wrapper_out(const func_t &f, ret_t &out, arg_ts &&...args)
{
   out = f(std::forward<arg_ts>(args)...);
}

template <typename T>
MFEM_HOST_DEVICE inline
void process_qf_result_rev(const DeviceTensor<1, real_t> &r, T &x)
{
   x = r(0);
}

template <typename T>
MFEM_HOST_DEVICE inline
void process_qf_result_rev(const DeviceTensor<1, real_t> &r, dual<T, T> &x)
{
   x.value = r(0);
}

template <typename T>
MFEM_HOST_DEVICE inline
void process_qf_result_rev(const DeviceTensor<1, real_t> &r, tensor<T> &x)
{
   x(0) = r(0);
}

template <typename T, int n>
MFEM_HOST_DEVICE inline
void process_qf_result_rev(const DeviceTensor<1, real_t> &r, tensor<T, n> &x)
{
   for (size_t i = 0; i < n; i++)
   {
      x(i) = r(i);
   }
}

template <typename T, int n, int m>
MFEM_HOST_DEVICE inline
void process_qf_result_rev(const DeviceTensor<1, real_t> &r, tensor<T, n, m> &x)
{
   for (size_t i = 0; i < n; i++)
   {
      for (size_t j = 0; j < m; j++)
      {
         x(i, j) = r(i + n * j);
      }
   }
}

// In Reverse mode, enzyme expects interleaved dup arguments:
// __enzyme_autodiff(..., enzyme_dup, &arg1, &shadow_arg1, enzyme_dup, &arg2, &shadow_arg2, ...)

#define ENZYME_DUP_ARGS_1(A, SA) enzyme_dup, &(A), &(SA)
#define ENZYME_DUP_ARGS_2(A, SA, B, SB) enzyme_dup, &(A), &(SA), enzyme_dup, &(B), &(SB)
#define ENZYME_DUP_ARGS_3(A, SA, B, SB, C, SC) enzyme_dup, &(A), &(SA), enzyme_dup, &(B), &(SB), enzyme_dup, &(C), &(SC)
#define ENZYME_DUP_ARGS_4(A, SA, B, SB, C, SC, D, SD) enzyme_dup, &(A), &(SA), enzyme_dup, &(B), &(SB), enzyme_dup, &(C), &(SC), enzyme_dup, &(D), &(SD)
#define ENZYME_DUP_ARGS_5(A, SA, B, SB, C, SC, D, SD, E, SE) enzyme_dup, &(A), &(SA), enzyme_dup, &(B), &(SB), enzyme_dup, &(C), &(SC), enzyme_dup, &(D), &(SD), enzyme_dup, &(E), &(SE)

template <typename qfunc_t, typename arg_ts, typename inactive_arg_ts>
MFEM_HOST_DEVICE inline
void autodiff_apply_enzyme_indexed_1(qfunc_t &qfunc, void *out, void *dout, arg_ts &args, arg_ts &shadow_args, inactive_arg_ts &inactive_args)
{
   using qf_return_t = typename create_function_signature<decltype(&qfunc_t::operator())>::type::return_t;
   __enzyme_autodiff<void>((void*)&qfunction_wrapper_out<qfunc_t, qf_return_t, decltype(get<0>(args))>,
      enzyme_const, &qfunc,
      enzyme_dupnoneed, out, dout,
      ENZYME_DUP_ARGS_1(get<0>(args), get<0>(shadow_args))
   );
}

template <typename qfunc_t, typename arg_ts, typename inactive_arg_ts>
MFEM_HOST_DEVICE inline
void autodiff_apply_enzyme_indexed_2(qfunc_t &qfunc, void *out, void *dout, arg_ts &args, arg_ts &shadow_args, inactive_arg_ts &inactive_args)
{
   using qf_return_t = typename create_function_signature<decltype(&qfunc_t::operator())>::type::return_t;
   __enzyme_autodiff<void>((void*)&qfunction_wrapper_out<qfunc_t, qf_return_t, decltype(get<0>(args)), decltype(get<1>(args))>,
      enzyme_const, &qfunc,
      enzyme_dupnoneed, out, dout,
      ENZYME_DUP_ARGS_2(get<0>(args), get<0>(shadow_args), get<1>(args), get<1>(shadow_args))
   );
}

template <typename qfunc_t, typename arg_ts, typename inactive_arg_ts>
MFEM_HOST_DEVICE inline
void autodiff_apply_enzyme_indexed_3(qfunc_t &qfunc, void *out, void *dout, arg_ts &args, arg_ts &shadow_args, inactive_arg_ts &inactive_args)
{
   using qf_return_t = typename create_function_signature<decltype(&qfunc_t::operator())>::type::return_t;
   __enzyme_autodiff<void>((void*)&qfunction_wrapper_out<qfunc_t, qf_return_t, decltype(get<0>(args)), decltype(get<1>(args)), decltype(get<2>(args))>,
      enzyme_const, &qfunc,
      enzyme_dupnoneed, out, dout,
      ENZYME_DUP_ARGS_3(get<0>(args), get<0>(shadow_args), get<1>(args), get<1>(shadow_args), get<2>(args), get<2>(shadow_args))
   );
}

template <typename qfunc_t, typename arg_ts, typename inactive_arg_ts>
MFEM_HOST_DEVICE inline
void autodiff_apply_enzyme_indexed_4(qfunc_t &qfunc, void *out, void *dout, arg_ts &args, arg_ts &shadow_args, inactive_arg_ts &inactive_args)
{
   using qf_return_t = typename create_function_signature<decltype(&qfunc_t::operator())>::type::return_t;
   __enzyme_autodiff<void>((void*)&qfunction_wrapper_out<qfunc_t, qf_return_t, decltype(get<0>(args)), decltype(get<1>(args)), decltype(get<2>(args)), decltype(get<3>(args))>,
      enzyme_const, &qfunc,
      enzyme_dupnoneed, out, dout,
      ENZYME_DUP_ARGS_4(get<0>(args), get<0>(shadow_args), get<1>(args), get<1>(shadow_args), get<2>(args), get<2>(shadow_args), get<3>(args), get<3>(shadow_args))
   );
}

template <typename qfunc_t, typename arg_ts, typename inactive_arg_ts>
MFEM_HOST_DEVICE inline
void autodiff_apply_enzyme_indexed_5(qfunc_t &qfunc, void *out, void *dout, arg_ts &args, arg_ts &shadow_args, inactive_arg_ts &inactive_args)
{
   using qf_return_t = typename create_function_signature<decltype(&qfunc_t::operator())>::type::return_t;
   __enzyme_autodiff<void>((void*)&qfunction_wrapper_out<qfunc_t, qf_return_t, decltype(get<0>(args)), decltype(get<1>(args)), decltype(get<2>(args)), decltype(get<3>(args)), decltype(get<4>(args))>,
      enzyme_const, &qfunc,
      enzyme_dupnoneed, out, dout,
      ENZYME_DUP_ARGS_5(get<0>(args), get<0>(shadow_args), get<1>(args), get<1>(shadow_args), get<2>(args), get<2>(shadow_args), get<3>(args), get<3>(shadow_args), get<4>(args), get<4>(shadow_args))
   );
}

template <typename qfunc_t, typename arg_ts, typename inactive_arg_ts>
MFEM_HOST_DEVICE inline
void autodiff_apply_enzyme(qfunc_t &qfunc, void *out, void *dout, arg_ts &&args,
                          arg_ts &&shadow_args,
                          inactive_arg_ts &&inactive_args)
{
   constexpr size_t num_active_args = tuple_size<std::remove_reference_t<arg_ts>>::value;
   
   if constexpr (num_active_args == 1) {
       autodiff_apply_enzyme_indexed_1(qfunc, out, dout, args, shadow_args, inactive_args);
   } else if constexpr (num_active_args == 2) {
       autodiff_apply_enzyme_indexed_2(qfunc, out, dout, args, shadow_args, inactive_args);
   } else if constexpr (num_active_args == 3) {
       autodiff_apply_enzyme_indexed_3(qfunc, out, dout, args, shadow_args, inactive_args);
   } else if constexpr (num_active_args == 4) {
       autodiff_apply_enzyme_indexed_4(qfunc, out, dout, args, shadow_args, inactive_args);
   } else if constexpr (num_active_args == 5) {
       autodiff_apply_enzyme_indexed_5(qfunc, out, dout, args, shadow_args, inactive_args);
   } else {
       MFEM_ABORT("Unsupported number of arguments for reverse mode AD.");
   }
}

template <typename qfunc_t, typename arg_ts, size_t num_args>
MFEM_HOST_DEVICE inline
void apply_kernel_vjp_enzyme(
   const DeviceTensor<1, real_t> &df_qp,
   qfunc_t &qfunc,
   arg_ts &args,
   arg_ts &shadow_args,
   const std::array<DeviceTensor<2>, num_args> &u,
   const std::array<DeviceTensor<2>, num_args> &du,
   int qp_idx)
{
   process_qf_args(u, args, qp_idx);
   process_qf_args(du, shadow_args, qp_idx);
   
   using qf_return_t = typename create_function_signature<
                       decltype(&qfunc_t::operator())>::type::return_t;
   qf_return_t out{};
   qf_return_t dout{};

   process_qf_result_rev(df_qp, get<0>(dout));

   autodiff_apply_enzyme(qfunc, &out, &dout, std::forward<arg_ts>(args), std::forward<arg_ts>(shadow_args), tuple<> {});
}

template <
   typename qf_param_ts,
   typename qfunc_t,
   std::size_t num_fields>
MFEM_HOST_DEVICE inline
void call_qfunction_vjp(
   qfunc_t &qfunc,
   const std::array<DeviceTensor<2>, num_fields> &input_shmem,
   const std::array<DeviceTensor<2>, num_fields> &shadow_shmem,
   DeviceTensor<2> &residual_shmem_adj,
   const int &rs_qp,
   const int &num_qp,
   const int &q1d,
   const int &dimension,
   const bool &use_sum_factorization)
{
   if (use_sum_factorization)
   {
      if (dimension == 1)
      {
         MFEM_FOREACH_THREAD_DIRECT(q, x, q1d)
         {
            auto qf_args = decay_tuple<qf_param_ts> {};
            auto qf_shadow_args = decay_tuple<qf_param_ts> {};
            auto r_adj = Reshape(&residual_shmem_adj(0, q), rs_qp);
            apply_kernel_vjp_enzyme(r_adj, qfunc, qf_args, qf_shadow_args, input_shmem, shadow_shmem, q);
         }
      }
      else if (dimension == 2)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
            {
               const int q = qx + q1d * qy;
               auto qf_args = decay_tuple<qf_param_ts> {};
               auto qf_shadow_args = decay_tuple<qf_param_ts> {};
               auto r_adj = Reshape(&residual_shmem_adj(0, q), rs_qp);
               apply_kernel_vjp_enzyme(r_adj, qfunc, qf_args, qf_shadow_args, input_shmem, shadow_shmem, q);
            }
         }
      }
      else if (dimension == 3)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
            {
               MFEM_FOREACH_THREAD_DIRECT(qz, z, q1d)
               {
                  const int q = qx + q1d * (qy + q1d * qz);
                  auto qf_args = decay_tuple<qf_param_ts> {};
                  auto qf_shadow_args = decay_tuple<qf_param_ts> {};
                  auto r_adj = Reshape(&residual_shmem_adj(0, q), rs_qp);
                  apply_kernel_vjp_enzyme(r_adj, qfunc, qf_args, qf_shadow_args, input_shmem, shadow_shmem, q);
               }
            }
         }
      }
      else
      {
#if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
         MFEM_ABORT("unsupported dimension for sum factorization");
#endif
      }
   }
   else
   {
      MFEM_FOREACH_THREAD_DIRECT(q, x, num_qp)
      {
         auto qf_args = decay_tuple<qf_param_ts> {};
         auto qf_shadow_args = decay_tuple<qf_param_ts> {};
         auto r_adj = Reshape(&residual_shmem_adj(0, q), rs_qp);
         apply_kernel_vjp_enzyme(r_adj, qfunc, qf_args, qf_shadow_args, input_shmem, shadow_shmem, q);
      }
   }
   MFEM_SYNC_THREAD;
}
#endif // MFEM_USE_ENZYME

} // namespace mfem::future
