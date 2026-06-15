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

namespace mfem::future
{

template <typename T0, typename T1, typename T2>
MFEM_HOST_DEVICE
void process_qf_arg(const T0 &, const T1 &, T2 &)
{
   static_assert(dfem::always_false<T0, T1, T2>,
                 "process_qf_arg not implemented for arg type");
}

template <typename T>
MFEM_HOST_DEVICE
void process_qf_arg(
   const DeviceTensor<1, T> &u,
   const DeviceTensor<1, T> &v,
   T &arg)
{
   arg = u(0);
}

template <typename T, int n, int m>
MFEM_HOST_DEVICE inline
void process_qf_arg(
   const DeviceTensor<1> &u,
   tensor<dual<T, T>, n, m> &arg)
{
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         arg(j, i).value = u((i * n) + j);
      }
   }
}

template <typename T>
MFEM_HOST_DEVICE inline
void process_qf_arg(
   const DeviceTensor<1> &u,
   dual<T, T> &arg)
{
   arg.value = u(0);
}

template <typename T>
MFEM_HOST_DEVICE inline
void process_qf_arg(
   const DeviceTensor<1> &u,
   const DeviceTensor<1> &v,
   dual<T, T> &arg)
{
   arg.value = u(0);
   arg.gradient = v(0);
}

template <typename T, int n>
MFEM_HOST_DEVICE inline
void process_qf_arg(
   const DeviceTensor<1> &u,
   const DeviceTensor<1> &v,
   tensor<dual<T, T>, n> &arg)
{
   for (int i = 0; i < n; i++)
   {
      arg(i).value = u(i);
      arg(i).gradient = v(i);
   }
}

template <typename T, int n, int m>
MFEM_HOST_DEVICE inline
void process_qf_arg(
   const DeviceTensor<1> &u,
   const DeviceTensor<1> &v,
   tensor<dual<T, T>, n, m> &arg)
{
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         arg(j, i).value = u((i * n) + j);
         arg(j, i).gradient = v((i * n) + j);
      }
   }
}

template <typename T, int n>
MFEM_HOST_DEVICE inline
void process_qf_result(
   DeviceTensor<1, T> &r,
   const tensor<dual<T, T>, n> &x)
{
   for (size_t i = 0; i < n; i++)
   {
      r(i) = x(i).value;
   }
}

template <typename T, int n, int m>
MFEM_HOST_DEVICE inline
void process_qf_result(
   DeviceTensor<1, T> &r,
   const tensor<dual<T, T>, n, m> &x)
{
   for (size_t i = 0; i < n; i++)
   {
      for (size_t j = 0; j < m; j++)
      {
         r(i + n * j) = x(i, j).value;
      }
   }
}

template <typename arg_type>
MFEM_HOST_DEVICE inline
void process_qf_arg(
   const DeviceTensor<2> &u,
   const DeviceTensor<2> &v,
   arg_type &arg,
   const int &qp)
{
   const auto u_qp = Reshape(&u(0, qp), u.GetShape()[0]);
   const auto v_qp = Reshape(&v(0, qp), v.GetShape()[0]);
   process_qf_arg(u_qp, v_qp, arg);
}

template <size_t num_fields, typename qf_args>
MFEM_HOST_DEVICE inline
void process_qf_args(
   const std::array<DeviceTensor<2>, num_fields> &u,
   const std::array<DeviceTensor<2>, num_fields> &v,
   qf_args &args,
   const int &qp)
{
   for_constexpr<tuple_size<qf_args>::value>([&](auto i)
   {
      process_qf_arg(u[i], v[i], get<i>(args), qp);
   });
}

template <typename T, int n, int m>
MFEM_HOST_DEVICE inline
void process_derivative_from_native_dual(
   DeviceTensor<1, T> &r,
   const tensor<dual<T, T>, n, m> &x)
{
   for (size_t i = 0; i < n; i++)
   {
      for (size_t j = 0; j < m; j++)
      {
         r(i + n * j) = x(i, j).gradient;
      }
   }
}

template <typename T, int n>
MFEM_HOST_DEVICE inline
void process_derivative_from_native_dual(
   DeviceTensor<1, T> &r,
   const tensor<dual<T, T>, n> &x)
{
   for (size_t i = 0; i < n; i++)
   {
      r(i) = x(i).gradient;
   }
}

template <typename T>
MFEM_HOST_DEVICE inline
void process_derivative_from_native_dual(
   DeviceTensor<1, T> &r,
   const dual<T, T> &x)
{
   r(0) = x.gradient;
}

template <typename T0, typename T1>
MFEM_HOST_DEVICE inline
void process_qf_arg(const T0 &, T1 &)
{
   static_assert(dfem::always_false<T0, T1>,
                 "process_qf_arg not implemented for arg type");
}

template <typename T>
MFEM_HOST_DEVICE inline
void process_qf_arg(
   const DeviceTensor<1, T> &u,
   T &arg)
{
   arg = u(0);
}

template <typename T>
MFEM_HOST_DEVICE inline
void process_qf_arg(
   const DeviceTensor<1, T> &u,
   tensor<T> &arg)
{
   arg(0) = u(0);
}

template <typename T, int n>
MFEM_HOST_DEVICE inline
void process_qf_arg(
   const DeviceTensor<1> &u,
   tensor<T, n> &arg)
{
   for (int i = 0; i < n; i++)
   {
      arg(i) = u(i);
   }
}

template <typename T, int n, int m>
MFEM_HOST_DEVICE inline
void process_qf_arg(
   const DeviceTensor<1> &u,
   tensor<T, n, m> &arg)
{
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         arg(j, i) = u((i * n) + j);
      }
   }
}

template <typename arg_type>
MFEM_HOST_DEVICE inline
void process_qf_arg(const DeviceTensor<2> &u, arg_type &arg, int qp)
{
   const auto u_qp = Reshape(&u(0, qp), u.GetShape()[0]);
   process_qf_arg(u_qp, arg);
}

template <size_t num_fields, typename qf_args>
MFEM_HOST_DEVICE inline
void process_qf_args(
   const std::array<DeviceTensor<2>, num_fields> &u,
   qf_args &args,
   const int &qp)
{
   for_constexpr<tuple_size<qf_args>::value>([&](auto i)
   {
      process_qf_arg(u[i], get<i>(args), qp);
   });
}

template <typename T0, typename T1>
MFEM_HOST_DEVICE inline
Vector process_qf_result(T0, T1)
{
   static_assert(dfem::always_false<T0, T1>,
                 "process_qf_result not implemented for result type");
   return Vector{};
}

template <typename T>
MFEM_HOST_DEVICE inline
void process_qf_result(
   DeviceTensor<1, T> &r,
   const T &x)
{
   r(0) = x;
}

template <typename T>
MFEM_HOST_DEVICE inline
void process_qf_result(
   DeviceTensor<1> &r,
   const dual<T, T> &x)
{
   r(0) = x.value;
}

template <typename T>
MFEM_HOST_DEVICE inline
void process_qf_result(
   DeviceTensor<1, T> &r,
   const tensor<T> &x)
{
   r(0) = x(0);
}

template <typename T, int n>
MFEM_HOST_DEVICE inline
void process_qf_result(
   DeviceTensor<1, T> &r,
   const tensor<T, n> &x)
{
   for (size_t i = 0; i < n; i++)
   {
      r(i) = x(i);
   }
}

template <typename T, int n, int m>
MFEM_HOST_DEVICE inline
void process_qf_result(
   DeviceTensor<1, T> &r,
   const tensor<T, n, m> &x)
{
   for (size_t i = 0; i < n; i++)
   {
      for (size_t j = 0; j < m; j++)
      {
         r(i + n * j) = x(i, j);
      }
   }
}

template <typename T, int n, int m>
MFEM_HOST_DEVICE inline
void process_qf_arg(
   const DeviceTensor<1, T> &u,
   const DeviceTensor<1, T> &v,
   tensor<T, n, m> &arg)
{
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         arg(j, i) = u((i * n) + j);
      }
   }
}

} // namespace mfem::future
