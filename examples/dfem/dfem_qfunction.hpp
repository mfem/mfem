#pragma once
#include "dfem_util.hpp"

namespace mfem
{

MFEM_HOST_DEVICE inline
void process_kf_arg(
   const DeviceTensor<1> &u,
   double &arg)
{
   arg = u(0);
}

MFEM_HOST_DEVICE inline
void process_kf_arg(
   const DeviceTensor<1> &u,
   internal::tensor<double> &arg)
{
   arg(0) = u(0);
}

template <typename T, int n>
MFEM_HOST_DEVICE
void process_kf_arg(
   const DeviceTensor<1> &u,
   internal::tensor<T, n> &arg)
{
   for (int i = 0; i < n; i++)
   {
      arg(i) = u(i);
   }
}

template <int n, int m>
MFEM_HOST_DEVICE
void process_kf_arg(
   const DeviceTensor<1> &u,
   internal::tensor<double, n, m> &arg)
{
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         arg(j, i) = u((i * m) + j);
      }
   }
   // assuming col major layout. translating to row major.
   // i + N_i*j
   // arg(0, 0) = u(0);
   // arg(0, 1) = u(0 + 2 * 1);
   // arg(1, 0) = u(1 + 2 * 0);
   // arg(1, 1) = u(1 + 2 * 1);
}

template <typename arg_type>
MFEM_HOST_DEVICE
void process_kf_arg(const DeviceTensor<2> &u, arg_type &arg, int qp)
{
   // out << "qp: " << qp << "\n";
   // for (int i = 0; i < u.GetShape()[0] * u.GetShape()[1]; i++)
   // {
   //    out << (&u(0, 0))[i] << " ";
   // }
   // out << "\n";

   const auto u_qp = Reshape(&u(0, qp), u.GetShape()[0]);
   // for (int i = 0; i < u_qp.GetShape()[0]; i++)
   // {
   //    out << (&u_qp(0))[i] << " ";
   // }
   // out << "\n";

   process_kf_arg(u_qp, arg);
}

template <size_t num_fields, typename kf_args, std::size_t... i>
MFEM_HOST_DEVICE
void process_kf_args(const std::array<DeviceTensor<2>, num_fields> &u,
                     kf_args &args, int qp, std::index_sequence<i...>)
{
   (process_kf_arg(u[i], mfem::get<i>(args), qp), ...);
}

template <typename T0, typename T1> inline
Vector process_kf_result(T0, T1)
{
   static_assert(always_false<T0, T1>,
                 "process_kf_result not implemented for result type");
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
   // out << "x: " << x << "\n";
   for (size_t i = 0; i < n; i++)
   {
      for (size_t j = 0; j < m; j++)
      {
         r(i + n * j) = x(i, j);
      }
   }

   // out << "r: ";
   // for (int i = 0; i < r.GetShape()[0]; i++)
   // {
   //    out << r(i) << " ";
   // }
   // out << "\n\n";
}

template <typename T> inline
void process_kf_arg(const DeviceTensor<1> &u, const DeviceTensor<1> &v,
                    double &arg)
{
   arg = u(0);
}

template <int n, int m> inline
void process_kf_arg(const DeviceTensor<1> &u, const DeviceTensor<1> &v,
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

template <typename arg_type> inline
void process_kf_arg(const DeviceTensor<2> &u, const DeviceTensor<2> &v,
                    arg_type &arg, int qp)
{
   const auto u_qp = Reshape(&u(0, qp), u.GetShape()[0]);
   const auto v_qp = Reshape(&v(0, qp), v.GetShape()[0]);
   process_kf_arg(u_qp, v_qp, arg);
}

template <size_t num_fields, typename kf_args, std::size_t... i> inline
void process_kf_args(std::array<DeviceTensor<2>, num_fields> &u,
                     std::array<DeviceTensor<2>, num_fields> &v,
                     kf_args &args, int qp, std::index_sequence<i...>)
{
   (process_kf_arg(u[i], v[i], mfem::get<i>(args), qp), ...);
}

template <typename kernel_func_t, typename kernel_args_ts, size_t num_args>
MFEM_HOST_DEVICE inline
void apply_kernel(
   DeviceTensor<1, double> &f_qp,
   const kernel_func_t &kf,
   kernel_args_ts &args,
   const std::array<DeviceTensor<2>, num_args> &u,
   int qp)
{
   process_kf_args(u, args, qp,
                   std::make_index_sequence<mfem::tuple_size<kernel_args_ts>::value> {});

   process_kf_result(f_qp, mfem::get<0>(mfem::apply(kf, args)));
}

// Version for active function arguments only
//
// This is an Enzyme regression and can be removed in later versions.
template <typename kernel_t, typename arg_ts, std::size_t... Is,
          typename inactive_arg_ts>
inline auto fwddiff_apply_enzyme_indexed(kernel_t kernel, arg_ts &&args,
                                         arg_ts &&shadow_args,
                                         std::index_sequence<Is...>,
                                         inactive_arg_ts &&inactive_args,
                                         std::index_sequence<>)
{
   using kf_return_t = typename create_function_signature<
                       decltype(&kernel_t::operator())>::type::return_t;
   return __enzyme_fwddiff<kf_return_t>(
             +kernel, enzyme_dup, &mfem::get<Is>(args)..., enzyme_interleave,
             &mfem::get<Is>(shadow_args)...);
}

// Interleave function arguments for enzyme
template <typename kernel_t, typename arg_ts, std::size_t... Is,
          typename inactive_arg_ts, std::size_t... Js>
inline auto fwddiff_apply_enzyme_indexed(kernel_t kernel, arg_ts &&args,
                                         arg_ts &&shadow_args,
                                         std::index_sequence<Is...>,
                                         inactive_arg_ts &&inactive_args,
                                         std::index_sequence<Js...>)
{
   using kf_return_t = typename create_function_signature<
                       decltype(&kernel_t::operator())>::type::return_t;
   return __enzyme_fwddiff<kf_return_t>(
             +kernel, enzyme_dup, &std::get<Is>(args)..., enzyme_const,
             &mfem::get<Js>(inactive_args)..., enzyme_interleave,
             &mfem::get<Is>(shadow_args)...);
}

template <typename kernel_t, typename arg_ts, typename inactive_arg_ts>
inline auto fwddiff_apply_enzyme(kernel_t kernel, arg_ts &&args,
                                 arg_ts &&shadow_args,
                                 inactive_arg_ts &&inactive_args)
{
   auto arg_indices = std::make_index_sequence<
                      mfem::tuple_size<std::remove_reference_t<arg_ts>>::value> {};

   auto inactive_arg_indices = std::make_index_sequence<
                               mfem::tuple_size<std::remove_reference_t<inactive_arg_ts>>::value> {};

   return fwddiff_apply_enzyme_indexed(kernel, args, shadow_args, arg_indices,
                                       inactive_args, inactive_arg_indices);
}

template <typename kf_t, typename kernel_arg_ts, size_t num_args>
MFEM_HOST_DEVICE inline
void apply_kernel_fwddiff_enzyme(
   DeviceTensor<1, double> &f_qp,
   const kf_t &kf,
   kernel_arg_ts &args,
   const std::array<DeviceTensor<2>, num_args> &u,
   kernel_arg_ts &shadow_args,
   const std::array<DeviceTensor<2>, num_args> &v,
   int qp_idx)
{
   process_kf_args(u, args, qp_idx,
                   std::make_index_sequence<mfem::tuple_size<kernel_arg_ts>::value> {});

   process_kf_args(v, shadow_args, qp_idx,
                   std::make_index_sequence<mfem::tuple_size<kernel_arg_ts>::value> {});

   process_kf_result(f_qp,
                     mfem::get<0>(fwddiff_apply_enzyme(kf, args, shadow_args, mfem::tuple<> {})));
}

}
