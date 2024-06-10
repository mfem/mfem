#pragma once

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>
#include <type_traits>

#include "dfem_util.hpp"
#include <linalg/tensor.hpp>

#include <enzyme/enzyme>

using std::size_t;

namespace mfem
{

template <size_t num_fields>
typename std::array<FieldDescriptor, num_fields>::const_iterator find_name(
   const std::array<FieldDescriptor, num_fields> &fields,
   const std::string &input_name)
{
   auto it = std::find_if(fields.begin(),
                          fields.end(), [&](const FieldDescriptor &field)
   {
      return field.field_label == input_name;
   });

   return it;
}

template <size_t num_fields>
int find_name_idx(const std::array<FieldDescriptor, num_fields> &fields,
                  const std::string &input_name)
{
   typename std::array<FieldDescriptor, num_fields>::const_iterator it
      = find_name(fields, input_name);
   if (it == fields.end())
   {
      return -1;
   }
   return (it - fields.begin());
}

template <typename entity_t, size_t num_fields, typename field_operator_ts, std::size_t... idx>
std::array<int, std::tuple_size_v<field_operator_ts>>
                                                   create_descriptors_to_fields_map(
                                                      std::array<FieldDescriptor, num_fields> &fields,
                                                      field_operator_ts &fops,
                                                      std::index_sequence<idx...>)
{
   std::array<int, std::tuple_size_v<field_operator_ts>> map;

   auto f = [&](auto &fop, auto &map)
   {
      int i;

      if constexpr (std::is_same_v<decltype(fop), Weight&>)
      {
         fop.dim = 1;
         fop.vdim = 1;
         fop.size_on_qp = 1;
         map = -1;
      }
      else if ((i = find_name_idx(fields, fop.field_label)) != -1)
      {
         fop.dim = GetDimension<entity_t>(fields[i]);
         fop.vdim = GetVDim(fields[i]);
         fop.size_on_qp = GetSizeOnQP<entity_t>(fop, fields[i]);
         map = i;
      }
      else
      {
         MFEM_ABORT("can't find field for " << fop.field_label);
      }
   };

   (f(std::get<idx>(fops), map[idx]), ...);

   return map;
}

template <typename input_t, std::size_t... i>
std::array<DeviceTensor<2>, sizeof...(i)> map_inputs_to_memory(
   std::array<Vector, sizeof...(i)> &input_qp_mem, int num_qp,
   const input_t &inputs, std::index_sequence<i...>)
{
   return {DeviceTensor<2>(input_qp_mem[i].ReadWrite(), std::get<i>(inputs).size_on_qp, num_qp) ...};
}

template <typename input_t, std::size_t... i>
std::array<Vector, sizeof...(i)> create_input_qp_memory(
   int num_qp,
   input_t &inputs,
   std::index_sequence<i...>)
{
   return {Vector(std::get<i>(inputs).size_on_qp * num_qp)...};
}

struct DofToQuadOperator
{
   static constexpr int rank = 3;
   DeviceTensor<rank, const double> B;
   const int which_input = -1;

   MFEM_HOST_DEVICE inline
   const double& operator()(int qp, int d, int dof) const
   {
      return B(qp, d, dof);
   }

   MFEM_HOST_DEVICE inline
   std::array<int, rank> GetShape() const
   {
      return B.GetShape();
   }
};

template <typename field_operator_t>
void map_field_to_quadrature_data(
   DeviceTensor<2> field_qp,
   int element_idx,
   const DofToQuadOperator &B,
   const Vector &field_e,
   field_operator_t &input,
   DeviceTensor<1, const double> integration_weights)
{
   if constexpr (std::is_same_v<field_operator_t, Value>)
   {
      auto [num_qp, unused, num_dof] = B.GetShape();
      const int vdim = input.vdim;
      const int element_offset = element_idx * num_dof * vdim;
      const auto field = Reshape(field_e.Read() + element_offset, num_dof, vdim);

      for (int vd = 0; vd < vdim; vd++)
      {
         for (int qp = 0; qp < num_qp; qp++)
         {
            double acc = 0.0;
            for (int dof = 0; dof < num_dof; dof++)
            {
               acc += B(qp, 0, dof) * field(dof, vd);
            }
            field_qp(vd, qp) = acc;
         }
      }
   }
   else if constexpr (std::is_same_v<field_operator_t, Gradient>)
   {
      const auto [num_qp, dim, num_dof] = B.GetShape();
      const int vdim = input.vdim;
      const int element_offset = element_idx * num_dof * vdim;
      const auto field = Reshape(field_e.Read() + element_offset, num_dof, vdim);

      auto f = Reshape(&field_qp[0], vdim, dim, num_qp);
      for (int qp = 0; qp < num_qp; qp++)
      {
         for (int vd = 0; vd < vdim; vd++)
         {
            for (int d = 0; d < dim; d++)
            {
               double acc = 0.0;
               for (int dof = 0; dof < num_dof; dof++)
               {
                  acc += B(qp, d, dof) * field(dof, vd);
               }
               f(vd, d, qp) = acc;
            }
         }
      }
   }
   else if constexpr (std::is_same_v<field_operator_t, Curl>)
   {
      const auto [num_qp, cdim, num_dof] = B.GetShape();
      const int vdim = input.vdim;
      const int element_offset = element_idx * num_dof * vdim;
      const auto field = Reshape(field_e.Read() + element_offset, num_dof, vdim,
                                 cdim);

      auto f = Reshape(&field_qp[0], vdim, cdim, num_qp);
      for (int qp = 0; qp < num_qp; qp++)
      {
         for (int vd = 0; vd < vdim; vd++)
         {
            for (int cd = 0; cd < cdim; cd++)
            {
               double acc = 0.0;
               for (int dof = 0; dof < num_dof; dof++)
               {
                  acc += B(qp, cd, dof) * field(dof, vd, cd);
               }
               f(vd, cd, qp) = acc;
            }
         }
      }
   }
   else if constexpr (std::is_same_v<field_operator_t, FaceValueLeft>)
   {
      auto [num_qp, unused, num_dof] = B.GetShape();
      const int vdim = input.vdim;
      const int element_offset = element_idx * num_dof * vdim;
      const auto field = Reshape(field_e.Read() + element_offset, num_dof, vdim);

      // for (int vd = 0; vd < vdim; vd++)
      // {
      //    for (int qp = 0; qp < num_qp; qp++)
      //    {
      //       double acc = 0.0;
      //       for (int dof = 0; dof < num_dof; dof++)
      //       {
      //          acc += B(qp, 0, dof) * field(dof, vd);
      //       }
      //       field_qp(vd, qp) = acc;
      //    }
      // }
   }
   else if constexpr (std::is_same_v<field_operator_t, FaceValueRight>)
   {
      auto [num_qp, unused, num_dof] = B.GetShape();
      const int vdim = input.vdim;
      const int element_offset = element_idx * num_dof * vdim;
      const auto field = Reshape(field_e.Read() + element_offset, num_dof, vdim);

      // for (int vd = 0; vd < vdim; vd++)
      // {
      //    for (int qp = 0; qp < num_qp; qp++)
      //    {
      //       double acc = 0.0;
      //       for (int dof = 0; dof < num_dof; dof++)
      //       {
      //          acc += B(qp, 0, dof) * field(dof, vd);
      //       }
      //       field_qp(vd, qp) = acc;
      //    }
      // }
   }
   // TODO: Create separate function for clarity
   else if constexpr (std::is_same_v<field_operator_t, Weight>)
   {
      const int num_qp = integration_weights.GetShape()[0];
      auto f = Reshape(&field_qp[0], num_qp);
      for (int qp = 0; qp < num_qp; qp++)
      {
         f(qp) = integration_weights(qp);
      }
   }
   else if constexpr (std::is_same_v<field_operator_t, None>)
   {
      auto [num_qp, unused, num_dof] = B.GetShape();
      const int size_on_qp = input.size_on_qp;
      const int element_offset = element_idx * size_on_qp * num_qp;
      const auto field = Reshape(field_e.Read() + element_offset,
                                 size_on_qp * num_qp);
      auto f = Reshape(&field_qp[0], size_on_qp * num_qp);
      for (int i = 0; i < size_on_qp * num_qp; i++)
      {
         f(i) = field(i);
      }
   }
   else
   {
      static_assert(always_false<field_operator_t>,
                    "can't map field to quadrature data");
   }
}

template <size_t num_fields, size_t num_kinputs, typename field_operator_tuple_t, std::size_t... i>
void map_fields_to_quadrature_data(
   std::array<DeviceTensor<2>, num_kinputs> &fields_qp,
   int element_idx,
   const std::array<Vector, num_fields> fields_e,
   const std::array<int, num_kinputs> &kfinput_to_field,
   const std::vector<DofToQuadOperator> &dtqmaps,
   const DeviceTensor<1, const double> &integration_weights,
   field_operator_tuple_t fops,
   std::index_sequence<i...>)
{
   (map_field_to_quadrature_data(fields_qp[i], element_idx,
                                 dtqmaps[i], fields_e[kfinput_to_field[i]],
                                 std::get<i>(fops), integration_weights),
    ...);
}

template <typename input_type>
void map_field_to_quadrature_data_conditional(
   DeviceTensor<2> field_qp, int element_idx, DofToQuadOperator &dtqmaps,
   const Vector &field_e, input_type &input,
   DeviceTensor<1, const double> integration_weights,
   const bool condition)
{
   if (condition)
   {
      map_field_to_quadrature_data(field_qp, element_idx, dtqmaps, field_e, input,
                                   integration_weights);
   }
}

template <size_t num_fields, size_t num_kinputs, typename field_operator_tuple_t, std::size_t... i>
void map_fields_to_quadrature_data_conditional(
   std::array<DeviceTensor<2>, num_kinputs> &fields_qp,
   int element_idx,
   const std::array<Vector, num_fields> &fields_e,
   const int field_idx,
   std::vector<DofToQuadOperator> &dtqmaps,
   DeviceTensor<1, const double> integration_weights,
   std::array<bool, num_kinputs> conditions,
   field_operator_tuple_t fops,
   std::index_sequence<i...>)
{
   (map_field_to_quadrature_data_conditional(fields_qp[i], element_idx,
                                             dtqmaps[field_idx], fields_e[field_idx],
                                             std::get<i>(fops), integration_weights, conditions[i]),
    ...);
}

template <typename input_t, size_t num_fields, std::size_t... i>
int accumulate_sizes_on_qp(
   const input_t &inputs,
   std::array<bool, sizeof...(i)> &kinput_is_dependent,
   const std::array<int, sizeof...(i)> &kinput_to_field,
   const std::array<FieldDescriptor, num_fields> &fields,
   std::index_sequence<i...>)
{
   return (... + [](auto &input, auto is_dependent, auto field)
   {
      if (!is_dependent) { return 0; }
      return GetSizeOnQP(input, field);
   }(std::get<i>(inputs),
     std::get<i>(kinput_is_dependent),
     fields[kinput_to_field[i]]));
}

void prepare_kf_arg(const DeviceTensor<1> &u, double &arg) { arg = u(0); }

template <typename T, int length>
void prepare_kf_arg(const DeviceTensor<1> &u,
                    internal::tensor<T, length> &arg)
{
   for (int i = 0; i < u.GetShape()[0]; i++)
   {
      arg(i) = u(i);
   }
}

template <int dim, int vdim>
void prepare_kf_arg(const DeviceTensor<1> &u,
                    internal::tensor<double, dim, vdim> &arg)
{
   for (int i = 0; i < vdim; i++)
   {
      for (int j = 0; j < dim; j++)
      {
         arg(j, i) = u((i * vdim) + j);
      }
   }
}

template <typename arg_type>
void prepare_kf_arg(const DeviceTensor<2> &u, arg_type &arg, int qp)
{
   const auto u_qp = Reshape(&u(0, qp), u.GetShape()[0]);
   prepare_kf_arg(u_qp, arg);
}

template <size_t num_fields, typename kf_args, std::size_t... i>
void prepare_kf_args(std::array<DeviceTensor<2>, num_fields> &u,
                     kf_args &args, int qp, std::index_sequence<i...>)
{
   (prepare_kf_arg(u[i], std::get<i>(args), qp), ...);
}

inline
Vector prepare_kf_result(std::tuple<double> x)
{
   Vector r(1);
   r = std::get<0>(x);
   return r;
}

inline
Vector prepare_kf_result(std::tuple<Vector> x)
{
   return std::get<0>(x);
}

template <typename T, int length> inline
Vector prepare_kf_result(std::tuple<internal::tensor<double, length>> x)
{
   Vector r(length);
   for (size_t i = 0; i < length; i++)
   {
      r(i) = std::get<0>(x)(i);
   }
   return r;
}

template <typename T, int length> inline
Vector prepare_kf_result(std::tuple<internal::tensor<T, length>> x)
{
   Vector r(length);
   for (size_t i = 0; i < length; i++)
   {
      r(i) = std::get<0>(x)(i);
   }
   return r;
}

template <typename T, int length_1, int length_2> inline
Vector prepare_kf_result(std::tuple<internal::tensor<T, length_1, length_2>> x)
{
   Vector r(length_1 * length_2);
   for (size_t i = 0; i < length_1; i++)
   {
      for (size_t j = 0; j < length_2; j++)
      {
         r(j + length_2 * i) = std::get<0>(x)(i, j);
      }
   }
   return r;
}

template <typename T, int length> inline
Vector prepare_kf_result(
   std::tuple<internal::tensor<internal::dual<T, T>, length>> x)
{
   Vector r(length);
   for (size_t i = 0; i < length; i++)
   {
      r(i) = std::get<0>(x)(i).value;
   }
   return r;
}

template <typename T> inline
Vector prepare_kf_result(std::tuple<internal::tensor<T, 2, 2>> x)
{
   Vector r(4);
   for (size_t i = 0; i < 2; i++)
   {
      for (size_t j = 0; j < 2; j++)
      {
         // TODO: Careful with the indices here!
         r(j + (i * 2)) = std::get<0>(x)(j, i);
      }
   }
   return r;
}

template <typename T> inline
Vector prepare_kf_result(std::tuple<internal::dual<T, T>> x)
{
   Vector r(1);
   r(0) = std::get<0>(x).value;
   return r;
}

template <typename T> inline
Vector get_derivative_from_dual(std::tuple<internal::dual<T, T>> x)
{
   Vector r(1);
   r(0) = std::get<0>(x).gradient;
   return r;
}

template <typename T> inline
Vector get_derivative_from_dual(
   std::tuple<internal::tensor<internal::dual<T, T>, 2, 2>> x)
{
   Vector r(4);
   for (size_t i = 0; i < 2; i++)
   {
      for (size_t j = 0; j < 2; j++)
      {
         // TODO: Careful with the indices here!
         r(j + (i * 2)) = std::get<0>(x)(j, i).gradient;
      }
   }
   return r;
}

template <typename T, int length> inline
Vector get_derivative_from_dual(
   std::tuple<internal::tensor<internal::dual<T, T>, length>> x)
{
   Vector r(length);
   for (size_t i = 0; i < length; i++)
   {
      r(i) = std::get<0>(x)(i).gradient;
   }
   return r;
}

template <typename T> inline
Vector prepare_kf_result(T)
{
   static_assert(always_false<T>,
                 "prepare_kf_result not implemented for result type");
}

template <size_t num_fields, typename kernel_func_t, typename kernel_args>
inline
auto apply_kernel(const kernel_func_t &kf, kernel_args &args,
                  std::array<DeviceTensor<2>, num_fields> &u,
                  int qp)
{
   prepare_kf_args(u, args, qp,
                   std::make_index_sequence<std::tuple_size_v<kernel_args>> {});

   return prepare_kf_result(std::apply(kf, args));
}

template <typename arg_ts, std::size_t... Is> inline
auto create_enzyme_args(arg_ts &args,
                        arg_ts &shadow_args,
                        std::index_sequence<Is...>)
{
   // (out << ... << std::get<Is>(shadow_args));
   return std::tuple_cat(std::tie(enzyme_dup, std::get<Is>(args),
                                  std::get<Is>(shadow_args))...);
}

// template <typename arg_ts, std::size_t... Is>
// auto create_enzyme_args(arg_ts &args,
//                         arg_ts &shadow_args,
//                         std::index_sequence<Is...>)
// {
//    // return std::tuple_cat(std::make_tuple(
//    //                          enzyme::Duplicated<std::remove_cv_t<std::remove_reference_t<decltype(std::get<Is>(args))>>*> {&std::get<Is>(args), &std::get<Is>(shadow_args)})...);
//    return std::tuple<enzyme::Duplicated<decltype(std::get<Is>(args))>...>
//    {
//       { std::get<Is>(args), std::get<Is>(shadow_args) }...
//    };
// }

template <typename kernel_t, typename arg_ts> inline
auto fwddiff_apply_enzyme(kernel_t kernel, arg_ts &&args, arg_ts &&shadow_args)
{
   auto arg_indices =
      std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<arg_ts>>> {};

   auto enzyme_args = create_enzyme_args(args, shadow_args, arg_indices);

   using kf_return_t = typename create_function_signature<
                       decltype(&kernel_t::operator())>::type::return_t;

   // out << "args is " << get_type_name<decltype(args)>() << "\n\n";
   // out << "enzyme_args type is " << get_type_name<decltype(enzyme_args)>() <<
   //     "\n\n";
   // out << "return type is " << get_type_name<decltype(kf_return_t{})>() <<
   //     "\n\n";

   return std::apply([&](auto &&...args)
   {
      return __enzyme_fwddiff<kf_return_t>((void*)+kernel, &args...);
      // return enzyme::get<0>( enzyme::autodiff<enzyme::Forward> (+kernel, args...));
   }, enzyme_args);
}

template <typename kf_t, typename kernel_arg_ts, size_t num_args> inline
auto apply_kernel_fwddiff_enzyme(const kf_t &kf,
                                 kernel_arg_ts &args,
                                 std::array<DeviceTensor<2>, num_args> &u,
                                 kernel_arg_ts &shadow_args,
                                 std::array<DeviceTensor<2>, num_args> &v,
                                 int qp)
{
   prepare_kf_args(u, args, qp,
                   std::make_index_sequence<std::tuple_size_v<kernel_arg_ts>> {});

   prepare_kf_args(v, shadow_args, qp,
                   std::make_index_sequence<std::tuple_size_v<kernel_arg_ts>> {});

   return prepare_kf_result(fwddiff_apply_enzyme(kf, args, shadow_args));
}

template <typename T> inline
void prepare_kf_arg(const DeviceTensor<1> &u, const DeviceTensor<1> &v,
                    double &arg)
{
   arg = u(0);
}

template <typename T, int dim, int vdim> inline
void prepare_kf_arg(const DeviceTensor<1> &u,
                    internal::tensor<internal::dual<T, T>, dim, vdim> &arg)
{
   for (int i = 0; i < vdim; i++)
   {
      for (int j = 0; j < dim; j++)
      {
         arg(j, i).value = u((i * vdim) + j);
      }
   }
}

template <typename T> inline
void prepare_kf_arg(const DeviceTensor<1> &u,
                    internal::dual<T, T> &arg)
{
   arg.value = u(0);
}

template <typename T> inline
void prepare_kf_arg(const DeviceTensor<1> &u, const DeviceTensor<1> &v, T &arg)
{
   arg = u(0);
}

template <typename T> inline
void prepare_kf_arg(const DeviceTensor<1> &u, const DeviceTensor<1> &v,
                    internal::dual<T, T> &arg)
{
   arg.value = u(0);
   arg.gradient = v(0);
}

template <typename T, int length> inline
void prepare_kf_arg(const DeviceTensor<1> &u, const DeviceTensor<1> &v,
                    internal::tensor<internal::dual<T, T>, length> &arg)
{
   for (int i = 0; i < u.GetShape()[0]; i++)
   {
      arg(i).value = u(i);
      arg(i).gradient = v(i);
   }
}

template <int dim, int vdim> inline
void prepare_kf_arg(const DeviceTensor<1> &u, const DeviceTensor<1> &v,
                    internal::tensor<double, dim, vdim> &arg)
{
   for (int i = 0; i < vdim; i++)
   {
      for (int j = 0; j < dim; j++)
      {
         arg(j, i) = u((i * vdim) + j);
      }
   }
}

template <typename T, int dim, int vdim> inline
void prepare_kf_arg(const DeviceTensor<1> &u, const DeviceTensor<1> &v,
                    internal::tensor<internal::dual<T, T>, dim, vdim> &arg)
{
   for (int i = 0; i < vdim; i++)
   {
      for (int j = 0; j < dim; j++)
      {
         arg(j, i).value = u((i * vdim) + j);
         arg(j, i).gradient = v((i * vdim) + j);
      }
   }
}

template <typename arg_type> inline
void prepare_kf_arg(const DeviceTensor<2> &u, const DeviceTensor<2> &v,
                    arg_type &arg, int qp)
{
   const auto u_qp = Reshape(&u(0, qp), u.GetShape()[0]);
   const auto v_qp = Reshape(&v(0, qp), v.GetShape()[0]);
   prepare_kf_arg(u_qp, v_qp, arg);
}

template <size_t num_fields, typename kf_args, std::size_t... i> inline
void prepare_kf_args(std::array<DeviceTensor<2>, num_fields> &u,
                     std::array<DeviceTensor<2>, num_fields> &v,
                     kf_args &args, int qp, std::index_sequence<i...>)
{
   (prepare_kf_arg(u[i], v[i], std::get<i>(args), qp), ...);
}

template <typename kernel_t, typename arg_ts> inline
auto fwddiff_apply_dual(kernel_t kernel, arg_ts &&args)
{
   return std::apply([&](auto &&...args)
   {
      return kernel(args...);
   },
   args);
}

template <typename kf_t, typename kernel_arg_ts, size_t num_args> inline
auto apply_kernel_fwddiff_dual(const kf_t &kf,
                               kernel_arg_ts &args,
                               std::array<DeviceTensor<2>, num_args> &u,
                               std::array<DeviceTensor<2>, num_args> &v,
                               int qp)
{
   prepare_kf_args(u, v, args, qp,
                   std::make_index_sequence<std::tuple_size_v<kernel_arg_ts>> {});

   return get_derivative_from_dual(fwddiff_apply_dual(kf, args));
}

template <typename output_type>
void map_quadrature_data_to_fields(DeviceTensor<2, double> y,
                                   DeviceTensor<3, double> c,
                                   output_type output,
                                   DofToQuadOperator &B)
{
   // assuming the quadrature point residual has to "play nice with
   // the test function"
   if constexpr (std::is_same_v<decltype(output), Value>)
   {
      const auto [num_qp, cdim, num_dof] = B.GetShape();
      const int vdim = output.vdim > 0 ? output.vdim : cdim ;
      for (int dof = 0; dof < num_dof; dof++)
      {
         for (int vd = 0; vd < vdim; vd++)
         {
            double acc = 0.0;
            for (int qp = 0; qp < num_qp; qp++)
            {
               acc += B(qp, 0, dof) * c(vd, 0, qp);
            }
            y(dof, vd) += acc;
         }
      }
   }
   else if constexpr (std::is_same_v<decltype(output), Gradient>)
   {
      const auto [num_qp, dim, num_dof] = B.GetShape();
      const int vdim = output.vdim;
      for (int dof = 0; dof < num_dof; dof++)
      {
         for (int vd = 0; vd < vdim; vd++)
         {
            double acc = 0.0;
            for (int d = 0; d < dim; d++)
            {
               for (int qp = 0; qp < num_qp; qp++)
               {
                  acc += B(qp, d, dof) * c(vd, d, qp);
               }
            }
            y(dof, vd) += acc;
         }
      }
   }
   else if constexpr (std::is_same_v<decltype(output), One>)
   {
      // This is the "integral over all quadrature points type" applying
      // B = 1 s.t. B^T * C \in R^1.
      auto [__, _, num_qp] = c.GetShape();
      auto cc = Reshape(&c(0, 0, 0), num_qp);
      for (int i = 0; i < num_qp; i++)
      {
         y(0, 0) += cc(i);
      }
   }
   else
   {
      MFEM_ABORT("quadrature data mapping to field is not implemented for"
                 " this field descriptor");
   }
}

template <typename entity_t, typename field_operator_ts, size_t N, std::size_t... i>
std::vector<DofToQuadOperator> create_dtq_operators_conditional(
   field_operator_ts &fops,
   std::vector<const DofToQuad*> dtqmaps,
   const std::array<int, N> &to_field_map,
   std::array<bool, N> active,
   std::index_sequence<i...>)
{
   std::vector<DofToQuadOperator> ops;
   auto f = [&](auto fop, size_t idx)
   {
      if (active[idx])
      {
         if (to_field_map[idx] == -1)
         {
            ops.push_back({DeviceTensor<3, const double>(nullptr, 1, 1, 1), -1});
            return;
         }
         auto dtqmap = dtqmaps[to_field_map[idx]];
         if constexpr (std::is_same_v<decltype(fop), Value>)
         {
            const int d = dtqmap->FE->GetRangeDim() ? dtqmap->FE->GetRangeDim() : 1;
            ops.push_back({DeviceTensor<3, const double>(dtqmap->B.Read(), dtqmap->nqpt, d, dtqmap->ndof), static_cast<int>(idx)});
         }
         else if constexpr (std::is_same_v<decltype(fop), Gradient>)
         {
            MFEM_ASSERT(dtqmap->FE->GetMapType() == FiniteElement::MapType::VALUE,
                        "trying to compute gradient of non compatible FE type");
            const int d = dtqmap->FE->GetDim();
            ops.push_back({DeviceTensor<3, const double>(dtqmap->G.Read(), dtqmap->nqpt, d, dtqmap->ndof), static_cast<int>(idx)});
         }
         else if constexpr (std::is_same_v<decltype(fop), Curl>)
         {
            MFEM_ASSERT(dtqmap->FE->GetMapType() == FiniteElement::MapType::H_CURL,
                        "trying to compute gradient of non compatible FE type");
            const int d = dtqmap->FE->GetCurlDim();
            ops.push_back({DeviceTensor<3, const double>(dtqmap->G.Read(), dtqmap->nqpt, d, dtqmap->ndof), static_cast<int>(idx)});
         }
         else if constexpr (std::is_same_v<decltype(fop), Div>)
         {
            MFEM_ASSERT(dtqmap->FE->GetMapType() == FiniteElement::MapType::H_DIV,
                        "trying to compute gradient of non compatible FE type");
            ops.push_back({DeviceTensor<3, const double>(dtqmap->G.Read(), dtqmap->nqpt, 1, dtqmap->ndof), static_cast<int>(idx)});
         }
         else if constexpr(std::is_same_v<decltype(fop), One>)
         {
            ops.push_back({DeviceTensor<3, const double>(nullptr, 1, 1, 1), static_cast<int>(idx)});
         }
      }
   };
   (f(std::get<i>(fops), i), ...);
   return ops;
}

template <typename entity_t, typename field_operator_ts, size_t N>
std::vector<DofToQuadOperator> create_dtq_operators(
   field_operator_ts &fops,
   std::vector<const DofToQuad*> dtqmaps,
   const std::array<int, N> &to_field_map)
{
   std::array<bool, N> is_dependent;
   std::fill(is_dependent.begin(), is_dependent.end(), true);
   return create_dtq_operators_conditional<entity_t>(fops, dtqmaps,
                                                     to_field_map,
                                                     is_dependent,
                                                     std::make_index_sequence<std::tuple_size_v<field_operator_ts>> {});
}

template <
   typename kernels_tuple,
   size_t num_solutions,
   size_t num_parameters,
   size_t num_fields = num_solutions + num_parameters,
   size_t num_kernels = std::tuple_size_v<kernels_tuple>
   >
class DifferentiableOperator : public Operator
{
public:
   class Action : public Operator
   {
   public:
      template <typename kernel_t>
      void create_action_callback(kernel_t kernel, mult_func_t &func);

      template<std::size_t... idx>
      void materialize_callbacks(kernels_tuple &ks,
                                 std::array<mult_func_t, num_kernels>,
                                 std::index_sequence<idx...> const&)
      {
         (create_action_callback(std::get<idx>(ks), funcs[idx]), ...);
      }

      Action(DifferentiableOperator &op, kernels_tuple &ks) : op(op)
      {
         materialize_callbacks(ks, funcs,
                               std::make_index_sequence<std::tuple_size_v<kernels_tuple>>());
      }

      void Mult(const Vector &x, Vector &y) const
      {
         prolongation(op.solutions, x, solutions_l);

         residual_e = 0.0;
         for (const auto &f : funcs)
         {
            f(residual_e);
         }

         prolongation_transpose(residual_l, y);

         y.SetSubVector(op.ess_tdof_list, 0.0);
      }

      void SetParameters(std::vector<Vector *> p)
      {
         MFEM_ASSERT(num_parameters == p.size(),
                     "number of parameters doesn't match descriptors");
         for (int i = 0; i < num_parameters; i++)
         {
            // parameters_local[i].MakeRef(*(p[i]), 0, p[i]->Size());
            parameters_l[i] = *(p[i]);
         }
      }

      const Vector& GetResidualQpMemory() const
      {
         return residual_qp_mem;
      }

   protected:
      DifferentiableOperator &op;
      std::array<mult_func_t, num_kernels> funcs;

      std::function<void(Vector &, Vector &)> restriction_transpose;
      std::function<void(Vector &, Vector &)> prolongation_transpose;

      mutable std::array<Vector, num_solutions> solutions_l;
      mutable std::array<Vector, num_parameters> parameters_l;
      mutable Vector residual_l;

      mutable std::array<Vector, num_fields> fields_e;
      mutable Vector residual_e;
      Vector residual_qp_mem;
   };

   template <size_t derivative_idx>
   class Derivative : public Operator
   {
   public:
      template <typename kernel_t>
      void create_callback(kernel_t kernel, mult_func_t &func);

      template<std::size_t... idx>
      void materialize_callbacks(kernels_tuple &ks,
                                 std::array<mult_func_t, num_kernels>,
                                 std::index_sequence<idx...> const&)
      {
         (create_callback(std::get<idx>(ks), funcs[idx]), ...);
      }

      Derivative(
         DifferentiableOperator &op,
         std::array<Vector *, num_solutions> &solutions,
         std::array<Vector *, num_parameters> &parameters,
         kernels_tuple &ks) : op(op), ks(ks)
      {
         for (int i = 0; i < num_solutions; i++)
         {
            solutions_l[i] = *solutions[i];
         }

         for (int i = 0; i < num_parameters; i++)
         {
            parameters_l[i] = *parameters[i];
         }

         // G
         // if constexpr (std::is_same_v<OperatesOn, OperatesOnElement>)
         // {
         element_restriction(op.solutions, solutions_l, fields_e,
                             op.element_dof_ordering);
         element_restriction(op.parameters, parameters_l, fields_e,
                             op.element_dof_ordering,
                             op.solutions.size());
         // }
         // else
         // {
         //    MFEM_ABORT("restriction not implemented for OperatesOn");
         // }

         // TODO-multvar: doesn't work for multiple solution variables
         directions[0] = op.fields[derivative_idx];

         size_t derivative_action_l_size = 0;
         for (auto &s : op.solutions)
         {
            derivative_action_l_size += GetVSize(s);
            this->width += GetTrueVSize(s);
         }
         this->height = derivative_action_l_size;
         derivative_action_l.SetSize(derivative_action_l_size);

         materialize_callbacks(ks, funcs,
                               std::make_index_sequence<num_kernels>());
      }

      void Mult(const Vector &x, Vector &y) const override
      {
         current_directions_t = x;
         current_directions_t.SetSubVector(op.ess_tdof_list, 0.0);

         prolongation(directions, current_directions_t, directions_l);

         derivative_action_e = 0.0;
         for (const auto &f : funcs)
         {
            f(derivative_action_e);
         }

         prolongation_transpose(derivative_action_l, y);

         y.SetSubVector(op.ess_tdof_list, 0.0);
      }

      template <typename kernel_t>
      void assemble_vector_impl(kernel_t kernel, Vector &v);

      template<std::size_t... idx>
      void assemble_vector(
         kernels_tuple &ks,
         Vector &v,
         std::index_sequence<idx...> const&)
      {
         (assemble_vector_impl(std::get<idx>(ks), v), ...);
      }

      void Assemble(Vector &v)
      {
         assemble_vector(ks, v, std::make_index_sequence<num_kernels>());
      }

      template <typename kernel_t>
      void assemble_hypreparmatrix_impl(kernel_t kernel, HypreParMatrix &A);

      template<std::size_t... idx>
      void assemble_hypreparmatrix(
         kernels_tuple &ks,
         HypreParMatrix &A,
         std::index_sequence<idx...> const&)
      {
         (assemble_hypreparmatrix_impl(std::get<idx>(ks), A), ...);
      }

      void Assemble(HypreParMatrix &A)
      {
         assemble_hypreparmatrix(ks, A, std::make_index_sequence<num_kernels>());
      }

      void AssembleDiagonal(Vector &d) const override {}

      const Vector& GetResidualQpMemory() const
      {
         return da_qp_mem;
      }

   protected:
      DifferentiableOperator &op;
      kernels_tuple &ks;
      std::array<mult_func_t, num_kernels> funcs;

      std::function<void(Vector &, Vector &)> restriction_transpose;
      std::function<void(Vector &, Vector &)> prolongation_transpose;

      // TODO-multvar: doesn't work for multiple solution variables
      std::array<FieldDescriptor, 1> directions;

      std::array<Vector, num_solutions> solutions_l;
      std::array<Vector, num_parameters> parameters_l;
      mutable std::array<Vector, 1> directions_l;
      mutable Vector derivative_action_l;

      mutable std::array<Vector, num_fields> fields_e;
      mutable std::array<Vector, num_fields> directions_e;
      mutable Vector derivative_action_e;

      mutable Vector current_directions_t;

      Vector da_qp_mem;
   };

   DifferentiableOperator(std::array<FieldDescriptor, num_solutions> s,
                          std::array<FieldDescriptor, num_parameters> p,
                          kernels_tuple ks,
                          ParMesh &m,
                          const IntegrationRule &integration_rule) :
      kernels(ks),
      mesh(m),
      dim(mesh.Dimension()),
      integration_rule(integration_rule),
      solutions(s),
      parameters(p)
   {
      for (int i = 0; i < num_solutions; i++)
      {
         fields[i] = solutions[i];
      }

      for (int i = 0; i < num_parameters; i++)
      {
         fields[i + num_solutions] = parameters[i];
      }

      residual.reset(new Action(*this, kernels));
   }

   void SetParameters(std::vector<Vector *> p)
   {
      residual->SetParameters(p);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      residual->Mult(x, y);
   }

   template <int derivative_idx>
   std::shared_ptr<Derivative<derivative_idx>>
                                            GetDerivativeWrt(std::array<Vector *, num_solutions> solutions,
                                                             std::array<Vector *, num_parameters> parameters)
   {
      return std::shared_ptr<Derivative<derivative_idx>>(
                new Derivative<derivative_idx>(*this, solutions, parameters, kernels));
   }

   // This function returns a Vector holding the memory right after the kernel function
   // has been executed. This means the output transformation has not been applied yet,
   // which is useful for testing or intricate, advanced usage. It is not recommended to
   // use or rely on this function.
   const Vector& GetResidualQpMemory() const
   {
      return residual->GetResidualQpMemory();
   }

   void SetEssentialTrueDofs(const Array<int> &l) { l.Copy(ess_tdof_list); }

   kernels_tuple kernels;
   ParMesh &mesh;
   const int dim;
   const IntegrationRule &integration_rule;

   std::array<FieldDescriptor, num_solutions> solutions;
   std::array<FieldDescriptor, num_parameters> parameters;
   // solutions and parameters
   std::array<FieldDescriptor, num_fields> fields;

   int residual_lsize = 0;

   mutable std::array<Vector, num_solutions> current_state_l;
   mutable Vector direction_l;

   mutable Vector current_direction_t;

   Array<int> ess_tdof_list;

   static constexpr ElementDofOrdering element_dof_ordering =
      ElementDofOrdering::LEXICOGRAPHIC;

   static constexpr DofToQuad::Mode doftoquad_mode =
      DofToQuad::Mode::LEXICOGRAPHIC_FULL;

   std::unique_ptr<Action> residual;
};

#include "dfem_action_callback.icc"
#include "dfem_derivative_callback.icc"
#include "dfem_assemble_vector.icc"
#include "dfem_assemble_hypreparmatrix.icc"

}
