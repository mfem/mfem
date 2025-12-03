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
#endif // MFEM_USE_ENZYME

} // namespace mfem::future
