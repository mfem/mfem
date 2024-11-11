#pragma once
#include "dfem_util.hpp"
#include "dfem_qfunction.hpp"

namespace mfem
{

MFEM_HOST_DEVICE
template <typename T0, typename T1, typename T2>
void process_kf_arg(const T0 &, const T1 &, T2 &)
{
   static_assert(always_false<T0, T1, T2>,
                 "process_kf_arg not implemented for arg type");
}

template <typename T>
MFEM_HOST_DEVICE
void process_kf_arg(
   const DeviceTensor<1, T> &u,
   const DeviceTensor<1, T> &v,
   T &arg)
{
   arg = u(0);
}

template <typename T, int n, int m>
MFEM_HOST_DEVICE inline
void process_kf_arg(
   const DeviceTensor<1> &u,
   internal::tensor<internal::dual<T, T>, n, m> &arg)
{
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         arg(j, i).value = u((i * m) + j);
      }
   }
}

template <typename T>
MFEM_HOST_DEVICE inline
void process_kf_arg(
   const DeviceTensor<1> &u,
   internal::dual<T, T> &arg)
{
   arg.value = u(0);
}

template <typename T>
MFEM_HOST_DEVICE inline
void process_kf_arg(
   const DeviceTensor<1> &u,
   const DeviceTensor<1> &v,
   internal::dual<T, T> &arg)
{
   arg.value = u(0);
   arg.gradient = v(0);
}

template <typename T, int n>
MFEM_HOST_DEVICE inline
void process_kf_arg(
   const DeviceTensor<1> &u,
   const DeviceTensor<1> &v,
   internal::tensor<internal::dual<T, T>, n> &arg)
{
   for (int i = 0; i < n; i++)
   {
      arg(i).value = u(i);
      arg(i).gradient = v(i);
   }
}

template <typename T, int n, int m>
MFEM_HOST_DEVICE inline
void process_kf_arg(
   const DeviceTensor<1> &u,
   const DeviceTensor<1> &v,
   internal::tensor<internal::dual<T, T>, n, m> &arg)
{
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         arg(j, i).value = u((i * m) + j);
         arg(j, i).gradient = v((i * m) + j);
      }
   }
}

template <typename T, int n>
MFEM_HOST_DEVICE inline
void process_kf_result(
   DeviceTensor<1, T> &r,
   const internal::tensor<internal::dual<T, T>, n> &x)
{
   for (size_t i = 0; i < n; i++)
   {
      r(i) = x(i).value;
   }
}

template <typename T, int n, int m>
MFEM_HOST_DEVICE inline
void process_kf_result(
   DeviceTensor<1, T> &r,
   const internal::tensor<internal::dual<T, T>, n, m> &x)
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
void process_kf_arg(
   const DeviceTensor<2> &u,
   const DeviceTensor<2> &v,
   arg_type &arg,
   const int &qp)
{
   const auto u_qp = Reshape(&u(0, qp), u.GetShape()[0]);
   const auto v_qp = Reshape(&v(0, qp), v.GetShape()[0]);
   process_kf_arg(u_qp, v_qp, arg);
}

template <size_t num_args, typename kf_args, std::size_t... Is>
MFEM_HOST_DEVICE inline
void process_kf_args(
   const std::array<DeviceTensor<2>, num_args> &u,
   const std::array<DeviceTensor<2>, num_args> &v,
   kf_args &args,
   const int &qp,
   std::index_sequence<Is...>)
{
   (process_kf_arg(u[Is], v[Is], mfem::get<Is>(args), qp), ...);
}

template <typename T, int n, int m>
MFEM_HOST_DEVICE inline
void process_derivative_from_native_dual(
   DeviceTensor<1, T> &r,
   const internal::tensor<internal::dual<T, T>, n, m> &x)
{
   for (size_t i = 0; i < n; i++)
   {
      for (size_t j = 0; j < m; j++)
      {
         r(i + n * j) = x(i, j).gradient;
      }
   }
   assert(false);
}

template <typename T, int n>
MFEM_HOST_DEVICE inline
void process_derivative_from_native_dual(
   DeviceTensor<1, T> &r,
   const internal::tensor<internal::dual<T, T>, n> &x)
{
   for (size_t i = 0; i < n; i++)
   {
      r(i) = x(i).gradient;
   }
}

template <typename kf_t, typename kernel_arg_ts, size_t num_args>
MFEM_HOST_DEVICE inline
void apply_kernel_native_dual(
   DeviceTensor<1, double> &f_qp,
   const kf_t &kf,
   kernel_arg_ts &args,
   const std::array<DeviceTensor<2>, num_args> &u,
   const std::array<DeviceTensor<2>, num_args> &v,
   const int &qp_idx)
{
   process_kf_args(u, v, args, qp_idx,
                   std::make_index_sequence<mfem::tuple_size<kernel_arg_ts>::value> {});
   auto r = mfem::get<0>(mfem::apply(kf, args));
   process_derivative_from_native_dual(f_qp, r);
}

} // namespace mfem
