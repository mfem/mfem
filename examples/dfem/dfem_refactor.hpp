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
#include "general/error.hpp"
#include "linalg/densemat.hpp"
#include "linalg/hypre.hpp"
#include <linalg/tensor.hpp>

#include <enzyme/enzyme>

using std::size_t;

namespace mfem
{

template <size_t N, size_t M>
void prolongation(const std::array<FieldDescriptor, N> fields,
                  const Vector &x,
                  std::array<Vector, M> &fields_l)
{
   int data_offset = 0;
   for (int i = 0; i < N; i++)
   {
      const auto P = get_prolongation(fields[i]);
      if (P != nullptr)
      {
         const int width = P->Width();
         const Vector x_i(x.GetData() + data_offset, width);
         fields_l[i].SetSize(P->Height());

         P->Mult(x_i, fields_l[i]);
         data_offset += width;
      }
      else
      {
         const int width = GetTrueVSize(fields[i]);
         fields_l[i].SetSize(width);
         const Vector x_i(x.GetData() + data_offset, width);
         fields_l[i] = x_i;
         data_offset += width;
      }
   }
}

template <size_t N, size_t M>
void element_restriction(const std::array<FieldDescriptor, N> u,
                         const std::array<Vector, N> &u_l,
                         std::array<Vector, M> &fields_e,
                         ElementDofOrdering ordering,
                         const int offset = 0)
{
   for (int i = 0; i < N; i++)
   {
      const auto R = get_element_restriction(u[i], ordering);
      if (R != nullptr)
      {
         MFEM_ASSERT(R->Width() == u_l[i].Size(),
                     "element restriction not applicable to given data size");
         const int height = R->Height();
         fields_e[i + offset].SetSize(height);
         R->Mult(u_l[i], fields_e[i + offset]);
      }
      else
      {
         const int height = GetTrueVSize(u[i]);
         fields_e[i + offset].SetSize(height);
         fields_e[i + offset] = u_l[i];
      }
   }
}

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

template <size_t num_fields, typename field_operator_ts, std::size_t... idx>
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
         fop.dim = GetDimension(fields[i]);
         fop.vdim = GetVDim(fields[i]);
         fop.size_on_qp = GetSizeOnQP(fop, fields[i]);
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
                    mfem::internal::tensor<T, length> &arg)
{
   for (int i = 0; i < u.GetShape()[0]; i++)
   {
      arg(i) = u(i);
   }
}

template <typename T, int dim, int vdim>
void prepare_kf_arg(const DeviceTensor<1> &u,
                    mfem::internal::tensor<T, dim, vdim> &arg)
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
   // we have several options here
   // - reinterpret_cast
   // - memcpy (copy data of u -> arg with overloading operator= for example)
   // for (int j = 0; j < num_fields; j++)
   // {
   //    auto [M, N] = u[j].GetShape();
   //    for (int m = 0; m < M; m++)
   //    {
   //       for (int n = 0; n < N; n++)
   //       {
   //          out << u[j](m, n) << " ";
   //       }
   //    }
   //    out << "\n";
   // }
   // out << "\n";
   (prepare_kf_arg(u[i], std::get<i>(args), qp), ...);
}

Vector prepare_kf_result(std::tuple<double> x)
{
   Vector r(1);
   r = std::get<0>(x);
   return r;
}

Vector prepare_kf_result(std::tuple<Vector> x)
{
   return std::get<0>(x);
}

template <int length>
Vector prepare_kf_result(std::tuple<mfem::internal::tensor<double, length>> x)
{
   Vector r(length);
   for (size_t i = 0; i < length; i++)
   {
      r(i) = std::get<0>(x)(i);
   }
   return r;
}

Vector prepare_kf_result(std::tuple<mfem::internal::tensor<double, 2, 2>> x)
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

template <typename T>
Vector prepare_kf_result(T)
{
   static_assert(always_false<T>,
                 "prepare_kf_result not implemented for result type");
}

template <size_t num_fields, typename kernel_func_t, typename kernel_args>
auto apply_kernel(const kernel_func_t &kf, kernel_args &args,
                  std::array<DeviceTensor<2>, num_fields> &u,
                  int qp)
{
   prepare_kf_args(u, args, qp,
                   std::make_index_sequence<std::tuple_size_v<kernel_args>> {});

   return prepare_kf_result(std::apply(kf, args));
}

// template <typename arg_ts, std::size_t... Is>
// auto create_enzyme_args(arg_ts &args,
//                         arg_ts &shadow_args,
//                         std::index_sequence<Is...>)
// {
//    return std::tuple_cat(std::make_tuple(enzyme_dup, std::get<Is>(args),
//                                          std::get<Is>(shadow_args))...);
// }

template <typename arg_ts, std::size_t... Is>
auto create_enzyme_args(arg_ts &args,
                        arg_ts &shadow_args,
                        std::index_sequence<Is...>)
{
   // (out << ... << std::get<Is>(shadow_args));
   return std::tuple_cat(std::make_tuple(
                            enzyme::Duplicated<std::remove_cv_t<std::remove_reference_t<decltype(std::get<Is>(args))>>*> {&std::get<Is>(args), &std::get<Is>(shadow_args)})...);
}

template <typename kernel_t, typename arg_ts>
auto fwddiff_apply_enzyme(kernel_t kernel, arg_ts &&args, arg_ts &&shadow_args)
{
   auto arg_indices =
      std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<arg_ts>>> {};

   auto enzyme_args = create_enzyme_args(args, shadow_args, arg_indices);
   // out << "arg_ts is " << get_type_name<decltype(args)>() << "\n\n";
   // out << "return type is " << get_type_name<decltype(enzyme_args)>() << "\n\n";

   return std::apply([&](auto &&...args)
   {
      using kf_return_t = typename create_function_signature<
                          decltype(&kernel_t::operator())>::type::return_t;

#ifdef MFEM_USE_ENZYME
      // return __enzyme_fwddiff<kernel_return_t>((void *)+kernel, args...);
      return enzyme::get<0>(
                enzyme::autodiff<enzyme::Forward, enzyme::DuplicatedNoNeed<kf_return_t>>
                (+kernel, args...));
#else
      return 0;
#endif
   },
   enzyme_args);
}

template <typename kf_t, typename kernel_arg_ts, size_t num_args>
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
      auto [vdim, dim, num_qp] = c.GetShape();
      auto cc = Reshape(&c(0, 0, 0), vdim * dim * num_qp);
      for (int i = 0; i < vdim * dim * num_qp; i++)
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

template <typename field_operator_ts, size_t N, std::size_t... i>
std::vector<DofToQuadOperator> create_dtq_operators_conditional(
   field_operator_ts &fops,
   std::vector<const DofToQuad*> dtqmaps,
   const std::array<int, N> &to_field_map,
   std::array<bool, N> is_dependent,
   std::index_sequence<i...>)
{
   std::vector<DofToQuadOperator> ops;
   auto f = [&](auto fop, size_t idx)
   {
      if (is_dependent[idx])
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

template <typename field_operator_ts, size_t N>
std::vector<DofToQuadOperator> create_dtq_operators(
   field_operator_ts &fops,
   std::vector<const DofToQuad*> dtqmaps,
   const std::array<int, N> &to_field_map)
{
   std::array<bool, N> is_dependent;
   std::fill(is_dependent.begin(), is_dependent.end(), true);
   return create_dtq_operators_conditional(fops, dtqmaps,
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
      void create_callback(kernel_t kernel, mult_func_t &func)
      {
         auto kinput_to_field = create_descriptors_to_fields_map(op.fields,
                                                                 kernel.inputs, std::make_index_sequence<kernel.num_kinputs> {});

         auto koutput_to_field = create_descriptors_to_fields_map(op.fields,
                                                                  kernel.outputs, std::make_index_sequence<kernel.num_koutputs> {});

         constexpr int hardcoded_output_idx = 0;
         const int test_space_field_idx = koutput_to_field[hardcoded_output_idx];

         auto output_fop = std::get<0>(kernel.outputs);

         const int num_el = op.mesh.GetNE();
         const int num_qp = op.integration_rule.GetNPoints();

         // All solutions T-vector sizes make up the width of the operator, since
         // they are explicitly provided in Mult() for example.

         op.width = GetTrueVSize(op.fields[test_space_field_idx]);
         op.residual_lsize = GetVSize(op.fields[test_space_field_idx]);

         if constexpr (std::is_same_v<decltype(output_fop), One>)
         {
            op.height = 1;
         }
         else
         {
            op.height = op.residual_lsize;
         }

         residual_l.SetSize(op.residual_lsize);

         // assume only a single element type for now
         std::vector<const DofToQuad*> dtqmaps;
         for (const auto &field : op.fields)
         {
            dtqmaps.emplace_back(GetDofToQuad(field, op.integration_rule, doftoquad_mode));
         }

         if (residual_e.Size() > 0 &&
             (residual_e.Size() != GetVSize(op.fields[test_space_field_idx]) * num_el))
         {
            MFEM_ABORT("inconsistent kernels");
         }
         else
         {
            residual_e.SetSize(GetVSize(op.fields[test_space_field_idx]) * num_el);
         }

         const int residual_size_on_qp = GetSizeOnQP(std::get<0>(kernel.outputs),
                                                     op.fields[test_space_field_idx]);

         if (residual_qp_mem.Size() > 0 &&
             (residual_qp_mem.Size() != residual_size_on_qp * num_qp * num_el))
         {
            MFEM_ABORT("inconsistent kernels");
         }
         else
         {
            residual_qp_mem.SetSize(residual_size_on_qp * num_qp * num_el);
         }

         // Allocate memory for fields on quadrature points
         auto input_qp_mem = create_input_qp_memory(num_qp, kernel.inputs,
                                                    std::make_index_sequence<kernel.num_kinputs> {});

         func = [this, kernel, num_el, num_qp, dtqmaps, residual_size_on_qp,
                       input_qp_mem, kinput_to_field, koutput_to_field,
                       output_fop]
         (Vector &ye_mem) mutable
         {
            auto residual_qp = Reshape(residual_qp_mem.ReadWrite(),
                                       residual_size_on_qp, num_qp, num_el);

            auto kernel_args = decay_tuple<typename kernel_t::kf_param_ts> {};

            DeviceTensor<1, const double> integration_weights(
               this->op.integration_rule.GetWeights().Read(), num_qp);

            // Fields interpolated to the quadrature points in the order of
            // kernel function arguments
            auto input_qp = map_inputs_to_memory(input_qp_mem, num_qp,
                                                 kernel.inputs,
                                                 std::make_index_sequence<kernel.num_kinputs> {});

            auto input_dtq_ops = create_dtq_operators(kernel.inputs, dtqmaps, kinput_to_field);
            auto output_dtq_ops = create_dtq_operators(kernel.outputs, dtqmaps, koutput_to_field);

            constexpr int fixed_output_idx = 0;
            auto Bv = output_dtq_ops[fixed_output_idx];
            auto [num_test_qp, test_op_dim, num_test_dof] = Bv.GetShape();

            const int test_vdim = std::get<0>(kernel.outputs).vdim;

            DeviceTensor<3> ye = Reshape(ye_mem.ReadWrite(), num_test_dof, test_vdim, num_el);

            for (int e = 0; e < num_el; e++)
            {
               map_fields_to_quadrature_data(
                  input_qp, e, this->fields_e,
                  kinput_to_field, input_dtq_ops,
                  integration_weights, kernel.inputs,
                  std::make_index_sequence<kernel.num_kinputs> {});

               for (int q = 0; q < num_qp; q++)
               {
                  auto f_qp = apply_kernel(kernel.func, kernel_args, input_qp, q);

                  auto r_qp = Reshape(&residual_qp(0, q, e), residual_size_on_qp);
                  for (int i = 0; i < residual_size_on_qp; i++)
                  {
                     r_qp(i) = f_qp(i);
                  }
               }

               DeviceTensor<3> fhat = Reshape(&residual_qp(0, 0, e), test_vdim, test_op_dim,
                                              num_qp);
               DeviceTensor<2> y = Reshape(&ye(0, 0, e), num_test_dof, test_vdim);
               map_quadrature_data_to_fields(y, fhat,
                                             output_fop,
                                             output_dtq_ops[hardcoded_output_idx]);
            }
         };

         if constexpr (std::is_same_v<decltype(output_fop), One>)
         {
            element_restriction_transpose = [](Vector &r_e, Vector &y)
            {
               y = r_e;
            };

            prolongation_transpose = [&](Vector &r_local, Vector &y)
            {
               double local_sum = r_local.Sum();
               MPI_Allreduce(&local_sum, y.GetData(), 1, MPI_DOUBLE, MPI_SUM,
                             op.mesh.GetComm());
               MFEM_ASSERT(y.Size() == 1, "output size doesn't match kernel description");
            };
         }
         else
         {
            auto R = get_element_restriction(op.fields[test_space_field_idx],
                                             element_dof_ordering);
            element_restriction_transpose = [R](const Vector &r_e, Vector &y)
            {
               R->MultTranspose(r_e, y);
            };

            auto P = get_prolongation(op.fields[test_space_field_idx]);
            prolongation_transpose = [P](const Vector &r_local, Vector &y)
            {
               // out << "address before P^T: " << y.GetData() << "\n";
               P->MultTranspose(r_local, y);
               // out << "address after P^T: " << y.GetData() << "\n";
            };
         }
      }

      template<std::size_t... idx>
      void materialize_callbacks(kernels_tuple &ks,
                                 std::array<mult_func_t, num_kernels>,
                                 std::index_sequence<idx...> const&)
      {
         (create_callback(std::get<idx>(ks), funcs[idx]), ...);
      }

      Action(DifferentiableOperator &op, kernels_tuple &ks) : op(op)
      {
         materialize_callbacks(ks, funcs,
                               std::make_index_sequence<std::tuple_size_v<kernels_tuple>>());
      }

      void Mult(const Vector &x, Vector &y) const override
      {
         prolongation(op.solutions, x, solutions_l);

         element_restriction(op.solutions, solutions_l, fields_e,
                             op.element_dof_ordering);
         element_restriction(op.parameters, parameters_l, fields_e,
                             op.element_dof_ordering,
                             op.solutions.size());

         residual_e = 0.0;
         for (const auto &f : funcs)
         {
            f(residual_e);
         }

         element_restriction_transpose(residual_e, residual_l);

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

      std::function<void(Vector &, Vector &)> element_restriction_transpose;
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
      void create_callback(kernel_t kernel, mult_func_t &func)
      {
         auto kinput_to_field = create_descriptors_to_fields_map(op.fields,
                                                                 kernel.inputs, std::make_index_sequence<kernel.num_kinputs> {});

         auto koutput_to_field = create_descriptors_to_fields_map(op.fields,
                                                                  kernel.outputs, std::make_index_sequence<kernel.num_koutputs> {});

         constexpr int hardcoded_output_idx = 0;
         const int test_space_field_idx = koutput_to_field[hardcoded_output_idx];

         auto output_fop = std::get<0>(kernel.outputs);

         const int num_el = op.mesh.GetNE();
         const int num_qp = op.integration_rule.GetNPoints();

         // assume only a single element type for now
         std::vector<const DofToQuad*> dtqmaps;
         for (const auto &field : op.fields)
         {
            dtqmaps.emplace_back(GetDofToQuad(field, op.integration_rule, doftoquad_mode));
         }

         derivative_action_e.SetSize(GetVDim(op.fields[test_space_field_idx]) * num_el *
                                     num_qp);

         const int da_size_on_qp = GetSizeOnQP(std::get<0>(kernel.outputs),
                                               op.fields[test_space_field_idx]);

         da_qp_mem.SetSize(da_size_on_qp * num_qp * num_el);

         // Allocate memory for fields on quadrature points
         auto input_qp_mem = create_input_qp_memory(num_qp, kernel.inputs,
                                                    std::make_index_sequence<kernel.num_kinputs> {});

         auto directions_qp_mem = create_input_qp_memory(num_qp, kernel.inputs,
                                                         std::make_index_sequence<kernel.num_kinputs> {});

         for (auto &d_qp_mem : directions_qp_mem)
         {
            d_qp_mem = 0.0;
         }

         func = [this, kernel, num_el, num_qp, dtqmaps, da_size_on_qp, input_qp_mem,
                       directions_qp_mem,
                       kinput_to_field, koutput_to_field, output_fop]
         (Vector &ye_mem) mutable
         {
            // Check which qf inputs are dependent on the dependent variable
            std::array<bool, kernel.num_kinputs> kinput_is_dependent;
            bool no_qfinput_is_dependent = true;
            for (int i = 0; i < kinput_is_dependent.size(); i++)
            {
               if (kinput_to_field[i] == derivative_idx)
               {
                  no_qfinput_is_dependent = false;
                  kinput_is_dependent[i] = true;
                  // out << "function input " << i << " is dependent on "
                  //     << op.fields[kinput_to_field[i]].field_label << "\n";
               }
               else
               {
                  kinput_is_dependent[i] = false;
               }
            }

            if (no_qfinput_is_dependent)
            {
               return;
            }

            const auto da_qp = Reshape(da_qp_mem.ReadWrite(), da_size_on_qp, num_qp, num_el);

            auto kernel_args = decay_tuple<typename kernel_t::kf_param_ts> {};
            auto kernel_shadow_args = decay_tuple<typename kernel_t::kf_param_ts> {};

            DeviceTensor<1, const double> integration_weights(
               this->op.integration_rule.GetWeights().Read(), num_qp);

            // Fields interpolated to the quadrature points in the order of
            // kernel function arguments
            auto input_qp = map_inputs_to_memory(input_qp_mem, num_qp,
                                                 kernel.inputs,
                                                 std::make_index_sequence<kernel.num_kinputs> {});

            auto directions_qp = map_inputs_to_memory(directions_qp_mem, num_qp,
                                                      kernel.inputs,
                                                      std::make_index_sequence<kernel.num_kinputs> {});

            auto input_dtq_ops = create_dtq_operators(kernel.inputs, dtqmaps, kinput_to_field);
            auto output_dtq_ops = create_dtq_operators(kernel.outputs, dtqmaps, koutput_to_field);

            constexpr int fixed_output_idx = 0;
            auto Bv = output_dtq_ops[fixed_output_idx];
            auto [num_test_qp, test_op_dim, num_test_dof] = Bv.GetShape();

            const int test_vdim = std::get<0>(kernel.outputs).vdim;

            DeviceTensor<3> ye = Reshape(ye_mem.ReadWrite(), num_test_dof, test_vdim, num_el);

            Vector f_qp(da_size_on_qp);

            for (int e = 0; e < num_el; e++)
            {
               map_fields_to_quadrature_data(
                  input_qp, e, this->fields_e,
                  kinput_to_field, input_dtq_ops,
                  integration_weights, kernel.inputs,
                  std::make_index_sequence<kernel.num_kinputs> {});

               map_fields_to_quadrature_data_conditional(
                  directions_qp, e,
                  directions_e, derivative_idx,
                  input_dtq_ops,
                  integration_weights,
                  kinput_is_dependent,
                  kernel.inputs,
                  std::make_index_sequence<kernel.num_kinputs> {});

               for (int qp = 0; qp < num_qp; qp++)
               {
                  f_qp = apply_kernel_fwddiff_enzyme(
                            kernel.func,
                            kernel_args,
                            input_qp,
                            kernel_shadow_args,
                            directions_qp,
                            qp);

                  auto r_qp = Reshape(&da_qp(0, qp, e), da_size_on_qp);
                  for (int i = 0; i < da_size_on_qp; i++)
                  {
                     r_qp(i) = f_qp(i);
                  }
               }

               DeviceTensor<3> fhat = Reshape(&da_qp(0, 0, e), test_vdim, test_op_dim, num_qp);
               DeviceTensor<2> y = Reshape(&ye(0, 0, e), num_test_dof, test_vdim);
               map_quadrature_data_to_fields(y, fhat,
                                             output_fop,
                                             output_dtq_ops[hardcoded_output_idx]);
            }
         };

         if constexpr (std::is_same_v<decltype(output_fop), One>)
         {
            element_restriction_transpose = [](Vector &r_e, Vector &y)
            {
               y = r_e;
            };

            prolongation_transpose = [&](Vector &r_local, Vector &y)
            {
               double local_sum = r_local.Sum();
               MPI_Allreduce(&local_sum, y.GetData(), 1, MPI_DOUBLE, MPI_SUM,
                             op.mesh.GetComm());
               MFEM_ASSERT(y.Size() == 1, "output size doesn't match kernel description");
            };
         }
         else
         {
            auto R = get_element_restriction(op.fields[test_space_field_idx],
                                             element_dof_ordering);
            element_restriction_transpose = [R](Vector &r_e, Vector &y)
            {
               R->MultTranspose(r_e, y);
            };

            auto P = get_prolongation(op.fields[test_space_field_idx]);
            prolongation_transpose = [P](Vector &r_l, Vector &y)
            {
               P->MultTranspose(r_l, y);
            };
         }
      }

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
         element_restriction(op.solutions, solutions_l, fields_e,
                             op.element_dof_ordering);
         element_restriction(op.parameters, parameters_l, fields_e,
                             op.element_dof_ordering,
                             op.solutions.size());

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

         element_restriction(directions, directions_l, directions_e,
                             op.element_dof_ordering, derivative_idx);

         derivative_action_e = 0.0;
         for (const auto &f : funcs)
         {
            f(derivative_action_e);
         }

         element_restriction_transpose(derivative_action_e, derivative_action_l);

         prolongation_transpose(derivative_action_l, y);

         y.SetSubVector(op.ess_tdof_list, 0.0);
      }

      void Assemble(Vector &v) {}

      template <typename kernel_t>
      void assemble_hypreparmatrix_impl(kernel_t kernel, HypreParMatrix &A)
      {
         auto kinput_to_field = create_descriptors_to_fields_map(op.fields,
                                                                 kernel.inputs,
                                                                 std::make_index_sequence<kernel.num_kinputs> {});

         auto koutput_to_field = create_descriptors_to_fields_map(op.fields,
                                                                  kernel.outputs,
                                                                  std::make_index_sequence<kernel.num_koutputs> {});

         auto output_fop = std::get<0>(kernel.outputs);

         constexpr int hardcoded_output_idx = 0;

         int num_qp = op.integration_rule.GetNPoints();;
         int num_el = 0;
         int dimension = 0;
         if constexpr (std::is_same_v<typename kernel_t::OperatesOn, OperatesOnElement>)
         {
            num_el = op.mesh.GetNE();
            dimension = op.dim;
         }
         else if (std::is_same_v<typename kernel_t::OperatesOn, OperatesOnBoundary>)
         {
            num_el = op.mesh.GetNBE();
            dimension = op.dim - 1;
         }
         else
         {
            static_assert(always_false<typename kernel_t::OperatesOn>, "not implemented");
         }

         std::vector<const DofToQuad*> dtqmaps;
         for (const auto &field : op.fields)
         {
            dtqmaps.emplace_back(GetDofToQuad(field, op.integration_rule, doftoquad_mode));
         }

         // Allocate memory for fields on quadrature points
         auto input_qp_mem = create_input_qp_memory(num_qp, kernel.inputs,
                                                    std::make_index_sequence<kernel.num_kinputs> {});

         auto directions_qp_mem = create_input_qp_memory(num_qp, kernel.inputs,
                                                         std::make_index_sequence<kernel.num_kinputs> {});

         for (auto &d_qp_mem : directions_qp_mem)
         {
            d_qp_mem = 0.0;
         }

         std::array<bool, kernel.num_kinputs> kinput_is_dependent;
         bool no_kinput_is_dependent = true;
         for (int i = 0; i < kinput_is_dependent.size(); i++)
         {
            if (kinput_to_field[i] == derivative_idx)
            {
               no_kinput_is_dependent = false;
               kinput_is_dependent[i] = true;
               // out << "function input " << i << " is dependent on "
               //     << op.fields[kinput_to_field[i]].field_label << "\n";
            }
            else
            {
               kinput_is_dependent[i] = false;
            }
         }

         if (no_kinput_is_dependent)
         {
            return;
         }

         auto kernel_args = decay_tuple<typename kernel_t::kf_param_ts> {};
         auto kernel_shadow_args = decay_tuple<typename kernel_t::kf_param_ts> {};

         DeviceTensor<1, const double> integration_weights(
            this->op.integration_rule.GetWeights().Read(), num_qp);

         // fields interpolated to the quadrature points in the order of
         // kernel function arguments
         auto input_qp = map_inputs_to_memory(input_qp_mem, num_qp,
                                              kernel.inputs,
                                              std::make_index_sequence<kernel.num_kinputs> {});

         auto directions_qp = map_inputs_to_memory(directions_qp_mem, num_qp,
                                                   kernel.inputs,
                                                   std::make_index_sequence<kernel.num_kinputs> {});

         auto input_dtq_ops = create_dtq_operators(kernel.inputs, dtqmaps,
                                                   kinput_to_field);
         auto dependent_input_dtq_ops = create_dtq_operators_conditional(
                                           kernel.inputs,
                                           dtqmaps,
                                           kinput_to_field,
                                           kinput_is_dependent, std::make_index_sequence<kernel.num_kinputs> {});

         auto output_dtq_ops = create_dtq_operators(kernel.outputs, dtqmaps,
                                                    koutput_to_field);

         constexpr int fixed_output_idx = 0;
         auto Bv = output_dtq_ops[fixed_output_idx];
         auto [num_test_qp, test_op_dim, num_test_dof] = Bv.GetShape();
         const int test_vdim = std::get<0>(kernel.outputs).vdim;

         const int num_trial_dof = dependent_input_dtq_ops[0].GetShape()[2];
         int trial_vdim = 0;
         for (int i = 0; i < kinput_is_dependent.size(); i++)
         {
            if (kinput_is_dependent[i])
            {
               trial_vdim = GetVDim(op.fields[kinput_to_field[i]]);
               break;
            }
         }

         // All trial operators dimensions accumulated
         int total_trial_op_dim = 0;
         for (int s = 0; s < dependent_input_dtq_ops.size(); s++)
         {
            total_trial_op_dim += dependent_input_dtq_ops[s].GetShape()[1];
         }

         // // number of rows for the derivative of the kernel \partial D
         // const int nrows_a = GetSizeOnQP(std::get<0>(kernel.outputs),
         //                                 op.fields[test_space_field_idx]);

         // // number of columns for the derivative of the kernel \partial D
         // const int ncols_a = accumulate_sizes_on_qp(kernel.inputs,
         //                                            kinput_is_dependent,
         //                                            kinput_to_field,
         //                                            op.fields,
         //                                            std::make_index_sequence<kernel.num_kinputs> {});

         Vector a_qp_mem(test_vdim * test_op_dim * trial_vdim * total_trial_op_dim *
                         num_qp *
                         num_el);
         const auto a_qp = Reshape(a_qp_mem.ReadWrite(), test_vdim, test_op_dim,
                                   trial_vdim, total_trial_op_dim, num_qp,
                                   num_el);

         Vector Ae_mem(num_test_dof * test_vdim * num_trial_dof * trial_vdim * num_el);
         Ae_mem = 0.0;

         auto A_e = Reshape(Ae_mem.ReadWrite(), num_test_dof, test_vdim, num_trial_dof,
                            trial_vdim, num_el);

         // for (int s = 0; s < dependent_input_dtq_ops.size(); s++)
         // {
         //    auto [num_qp, trial_op_dim,
         //          num_trial_dof] = dependent_input_dtq_ops[s].GetShape();
         //    Vector budense(num_qp * num_trial_dof * trial_op_dim);
         //    auto bb = Reshape(&dependent_input_dtq_ops[s].B(0, 0, 0),
         //                      num_qp * trial_op_dim * num_trial_dof);
         //    for (int i = 0; i < num_qp * trial_op_dim * num_trial_dof; i++)
         //    {
         //       budense(i) = bb(i);
         //    }
         //    print_vector(budense);
         // }

         for (int e = 0; e < num_el; e++)
         {
            map_fields_to_quadrature_data(
               input_qp, e, this->fields_e,
               kinput_to_field, input_dtq_ops,
               integration_weights, kernel.inputs,
               std::make_index_sequence<kernel.num_kinputs> {});

            for (int q = 0; q < num_qp; q++)
            {
               for (int j = 0; j < trial_vdim; j++)
               {
                  size_t m_offset = 0;
                  for (int s = 0; s < dependent_input_dtq_ops.size(); s++)
                  {
                     auto Bu = dependent_input_dtq_ops[s];
                     auto [unused1, trial_op_dim, unused2] = Bu.GetShape();
                     auto d_qp = Reshape(&(directions_qp[Bu.which_input])[0], trial_vdim,
                                         trial_op_dim, num_qp);
                     for (int m = 0; m < trial_op_dim; m++)
                     {
                        d_qp(j, m, q) = 1.0;
                        Vector f_qp = apply_kernel_fwddiff_enzyme(
                                         kernel.func,
                                         kernel_args,
                                         input_qp,
                                         kernel_shadow_args,
                                         directions_qp,
                                         q);
                        d_qp(j, m, q) = 0.0;

                        auto f = Reshape(f_qp.Read(), test_vdim, test_op_dim);

                        for (int i = 0; i < test_vdim; i++)
                        {
                           for (int k = 0; k < test_op_dim; k++)
                           {
                              a_qp(i, k, j, m + m_offset, q, e) = f(i, k);
                           }
                        }
                     }
                     m_offset += trial_op_dim;
                  }
               }
            }

            Vector fhat_mem(test_op_dim * num_qp * dimension);
            auto fhat = Reshape(fhat_mem.ReadWrite(), test_vdim, test_op_dim, num_qp);
            for (int J = 0; J < num_trial_dof; J++)
            {
               for (int j = 0; j < trial_vdim; j++)
               {
                  fhat_mem = 0.0;
                  size_t m_offset = 0;
                  for (int s = 0; s < dependent_input_dtq_ops.size(); s++)
                  {
                     auto Bu = dependent_input_dtq_ops[s];
                     int trial_op_dim = dependent_input_dtq_ops[s].GetShape()[1];
                     for (int q = 0; q < num_qp; q++)
                     {
                        for (int i = 0; i < test_vdim; i++)
                        {
                           for (int k = 0; k < test_op_dim; k++)
                           {
                              for (int m = 0; m < trial_op_dim; m++)
                              {
                                 fhat(i, k, q) += a_qp(i, k, j, m + m_offset, q, e) * Bu(q, m, J);
                              }
                           }
                        }
                     }
                     m_offset += trial_op_dim;
                  }

                  auto bvtfhat = Reshape(&A_e(0, 0, J, j, e), num_test_dof, test_vdim);
                  map_quadrature_data_to_fields(bvtfhat, fhat, output_fop,
                                                output_dtq_ops[hardcoded_output_idx]);
               }
            }
         }

         bool same_test_and_trial = false;
         if (koutput_to_field[0] ==
             kinput_to_field[dependent_input_dtq_ops[0].which_input])
         {
            same_test_and_trial = true;
         }

         auto trial_fes = *std::get_if<const ParFiniteElementSpace *>
                          (&op.fields[kinput_to_field[dependent_input_dtq_ops[0].which_input]].data);

         auto test_fes = *std::get_if<const ParFiniteElementSpace *>
                         (&op.fields[koutput_to_field[0]].data);

         SparseMatrix mat(test_fes->GlobalVSize(), trial_fes->GlobalVSize());

         if (test_fes == nullptr)
         {
            MFEM_ABORT("error");
         }

         for (int e = 0; e < num_el; e++)
         {
            auto tmp = Reshape(Ae_mem.ReadWrite(), num_test_dof * test_vdim,
                               num_trial_dof * trial_vdim,
                               num_el);
            DenseMatrix A_e(&tmp(0, 0, e), num_test_dof * test_vdim,
                            num_trial_dof * trial_vdim);
            Array<int> test_vdofs, trial_vdofs;
            test_fes->GetElementVDofs(e, test_vdofs);
            GetElementVDofs(
               op.fields[kinput_to_field[dependent_input_dtq_ops[0].which_input]], e,
               trial_vdofs);
            mat.AddSubMatrix(test_vdofs, trial_vdofs, A_e, 1);
         }
         mat.Finalize();

         if (same_test_and_trial)
         {
            HypreParMatrix tmp(test_fes->GetComm(),
                               test_fes->GlobalVSize(),
                               test_fes->GetDofOffsets(),
                               &mat);

            A = *RAP(&tmp, test_fes->Dof_TrueDof_Matrix());
            A.EliminateBC(op.ess_tdof_list, DiagonalPolicy::DIAG_ONE);
         }
         else
         {
            HypreParMatrix tmp(test_fes->GetComm(),
                               test_fes->GlobalVSize(),
                               trial_fes->GlobalVSize(),
                               test_fes->GetDofOffsets(),
                               trial_fes->GetDofOffsets(),
                               &mat);

            A = *RAP(test_fes->Dof_TrueDof_Matrix(), &tmp, trial_fes->Dof_TrueDof_Matrix());
            // A.EliminateBC(op.ess_tdof_list, DiagonalPolicy::DIAG_ONE);
         }
      }

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

      std::function<void(Vector &, Vector &)> element_restriction_transpose;
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
      ElementDofOrdering::NATIVE;

   static constexpr DofToQuad::Mode doftoquad_mode = DofToQuad::Mode::FULL;

   std::unique_ptr<Action> residual;
};

}
