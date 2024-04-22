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

std::vector<DofToQuadMaps> map_dtqmaps(std::vector<const DofToQuad*>
                                       dtqmaps, int dim, int num_qp)
{
   std::vector<DofToQuadMaps> dtqmaps_tensor;
   for (auto m : dtqmaps)
   {
      if (m != nullptr)
      {
         dtqmaps_tensor.push_back(
         {
            Reshape(m->B.Read(), m->nqpt, m->ndof),
            Reshape(m->G.Read(), m->nqpt, dim, m->ndof)
         });
      }
      else
      {
         // If there is no DofToQuad map available, we assume that the "map"
         // is identity and therefore maps 1 to 1 from #qp to #qp.
         DeviceTensor<2, const double> B(nullptr, num_qp, num_qp);
         DeviceTensor<3, const double> G(nullptr, num_qp, dim, num_qp);
         dtqmaps_tensor.push_back({B, G});
      }
   }
   return dtqmaps_tensor;
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

template <typename field_operator_t>
void map_field_to_quadrature_data(
   DeviceTensor<2> field_qp,
   int element_idx,
   const DofToQuadMaps &dtqmaps,
   const Vector &field_e,
   field_operator_t &input,
   DeviceTensor<1, const double> integration_weights)
{
   if constexpr (std::is_same_v<field_operator_t, Value>)
   {
      const auto B(dtqmaps.B);
      auto [num_qp, num_dof] = B.GetShape();
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
               acc += B(qp, dof) * field(dof, vd);
            }
            field_qp(vd, qp) = acc;
         }
      }
   }
   else if constexpr (std::is_same_v<field_operator_t, Gradient>)
   {
      const auto G(dtqmaps.G);
      const auto [num_qp, dim, num_dof] = G.GetShape();
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
                  acc += G(qp, d, dof) * field(dof, vd);
               }
               f(vd, d, qp) = acc;
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
      const auto B(dtqmaps.B);
      auto [num_qp, num_dof] = B.GetShape();
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
   const std::vector<DofToQuadMaps> &dtqmaps,
   const DeviceTensor<1, const double> &integration_weights,
   field_operator_tuple_t fops,
   std::index_sequence<i...>)
{
   (map_field_to_quadrature_data(fields_qp[i], element_idx,
                                 dtqmaps[kfinput_to_field[i]], fields_e[kfinput_to_field[i]],
                                 std::get<i>(fops), integration_weights),
    ...);
}

template <typename input_type>
void map_field_to_quadrature_data_conditional(
   DeviceTensor<2> field_qp, int element_idx, DofToQuadMaps &dtqmaps,
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
   std::vector<DofToQuadMaps> &dtqmaps,
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

Vector prepare_kf_result(double x)
{
   Vector r(1);
   r = x;
   return r;
}

Vector prepare_kf_result(Vector x)
{
   return x;
}

template <int length>
Vector prepare_kf_result(mfem::internal::tensor<double, length> x)
{
   Vector r(length);
   for (size_t i = 0; i < length; i++)
   {
      r(i) = x(i);
   }
   return r;
}

Vector prepare_kf_result(mfem::internal::tensor<double, 2, 2> x)
{
   Vector r(4);
   for (size_t i = 0; i < 2; i++)
   {
      for (size_t j = 0; j < 2; j++)
      {
         // TODO: Careful with the indices here!
         r(j + (i * 2)) = x(j, i);
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
void map_quadrature_data_to_fields(Vector &y_e, int element_idx, int num_el,
                                   DeviceTensor<3, double> r_qp,
                                   output_type output,
                                   std::vector<DofToQuadMaps> &dtqmaps,
                                   int test_space_field_idx)
{
   // assuming the quadrature point residual has to "play nice with
   // the test function"
   if constexpr (std::is_same_v<decltype(output), Value>)
   {
      const auto B(dtqmaps[test_space_field_idx].B);
      const auto [num_qp, num_dof] = B.GetShape();
      const int vdim = output.vdim;
      auto C = Reshape(&r_qp(0, 0, element_idx), vdim, num_qp);
      auto y = Reshape(y_e.ReadWrite(), num_dof, vdim, num_el);
      for (int dof = 0; dof < num_dof; dof++)
      {
         for (int vd = 0; vd < vdim; vd++)
         {
            double acc = 0.0;
            for (int qp = 0; qp < num_qp; qp++)
            {
               // |JxW| is assumed to be in C
               acc += B(qp, dof) * C(vd, qp);
            }
            y(dof, vd, element_idx) += acc;
         }
      }
   }
   else if constexpr (std::is_same_v<decltype(output), Gradient>)
   {
      const auto G(dtqmaps[test_space_field_idx].G);
      const auto [num_qp, dim, num_dof] = G.GetShape();
      const int vdim = output.vdim;
      auto C = Reshape(&r_qp(0, 0, element_idx), vdim, dim, num_qp);
      auto y = Reshape(y_e.ReadWrite(), num_dof, vdim, num_el);
      for (int dof = 0; dof < num_dof; dof++)
      {
         for (int vd = 0; vd < vdim; vd++)
         {
            double acc = 0.0;
            for (int d = 0; d < dim; d++)
            {
               for (int qp = 0; qp < num_qp; qp++)
               {
                  acc += G(qp, d, dof) * C(vd, d, qp);
               }
            }
            y(dof, vd, element_idx) += acc;
         }
      }
   }
   else if constexpr (std::is_same_v<decltype(output), One>)
   {
      // This is the "integral over all quadrature points type" applying
      // B = 1 s.t. B^T * C \in R^1.
      auto [size_on_qp, num_qp, num_el] = r_qp.GetShape();
      auto C = Reshape(&r_qp(0, 0, element_idx), size_on_qp * num_qp);
      auto y = Reshape(y_e.ReadWrite(), num_el);
      for (int i = 0; i < size_on_qp * num_qp; i++)
      {
         y(element_idx) += C(i);
      }
   }
   else
   {
      MFEM_ABORT("quadrature data mapping to field is not implemented for"
                 " this field descriptor");
   }
}

struct DofToQuadOperator
{
   static constexpr int rank = 3;
   DeviceTensor<rank, const double> B;

   const double& operator()(int qp, int d, int dof) const
   {
      return B(qp, d, dof);
   }

   MFEM_HOST_DEVICE inline std::array<int, rank> GetShape() const
   {
      return B.GetShape();
   }
};

template <typename field_operator_ts, size_t N, std::size_t... i>
std::vector<DofToQuadOperator> create_dtq_operators(
   field_operator_ts &fops,
   std::vector<const DofToQuad*> dtqmaps,
   int dim,
   const std::array<int, N> &to_field_map,
   std::array<bool, N> is_dependent,
   std::index_sequence<i...>)
{
   std::vector<DofToQuadOperator> ops;
   auto f = [&](auto fop, size_t idx)
   {
      if (to_field_map[idx] == -1)
      {
         return;
      }
      auto dtqmap = dtqmaps[to_field_map[idx]];
      if (is_dependent[idx])
      {
         using field_operator_t = decltype(fop);
         if constexpr (std::is_same_v<field_operator_t, Value>)
         {
            ops.push_back({DeviceTensor<3, const double>(dtqmap->B.Read(), dtqmap->nqpt, 1, dtqmap->ndof)});
         }
         else if constexpr (std::is_same_v<field_operator_t, Gradient>)
         {
            ops.push_back({DeviceTensor<3, const double>(dtqmap->G.Read(), dtqmap->nqpt, dim, dtqmap->ndof)});
         }
      }
   };
   (f(std::get<i>(fops), i), ...);
   return ops;
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
         op.width = 0;
         op.residual_lsize = 0;
         for (auto &s : op.solutions)
         {
            op.width += GetTrueVSize(s);
            op.residual_lsize += GetVSize(s);
         }
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
                       input_qp_mem, kinput_to_field, test_space_field_idx, output_fop]
         (Vector &y_e) mutable
         {
            auto dtqmaps_tensor = map_dtqmaps(dtqmaps, this->op.dim, num_qp);

            const auto residual_qp = Reshape(residual_qp_mem.ReadWrite(),
                                             residual_size_on_qp, num_qp, num_el);

            auto kernel_args = decay_tuple<typename kernel_t::kf_param_ts> {};

            DeviceTensor<1, const double> integration_weights(
               this->op.integration_rule.GetWeights().Read(), num_qp);

            // Fields interpolated to the quadrature points in the order of
            // kernel function arguments
            auto input_qp = map_inputs_to_memory(input_qp_mem, num_qp,
                                                 kernel.inputs,
                                                 std::make_index_sequence<kernel.num_kinputs> {});

            for (int el = 0; el < num_el; el++)
            {
               map_fields_to_quadrature_data(
                  input_qp, el, this->fields_e,
                  kinput_to_field, dtqmaps_tensor,
                  integration_weights, kernel.inputs,
                  std::make_index_sequence<kernel.num_kinputs> {});

               for (int qp = 0; qp < num_qp; qp++)
               {
                  auto f_qp = apply_kernel(kernel.func, kernel_args, input_qp, qp);

                  auto r_qp = Reshape(&residual_qp(0, qp, el), residual_size_on_qp);
                  for (int i = 0; i < residual_size_on_qp; i++)
                  {
                     r_qp(i) = f_qp(i);
                  }
               }

               map_quadrature_data_to_fields(y_e, el, num_el, residual_qp,
                                             output_fop,
                                             dtqmaps_tensor,
                                             test_space_field_idx);
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
            prolongation_transpose = [P](Vector &r_local, Vector &y)
            {
               P->MultTranspose(r_local, y);
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
                       kinput_to_field, test_space_field_idx, output_fop]
         (Vector &y_e) mutable
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
                  out << "function input " << i << " is dependent on "
                      << op.fields[kinput_to_field[i]].field_label << "\n";
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

            auto dtqmaps_tensor = map_dtqmaps(dtqmaps, this->op.dim, num_qp);

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
            Vector f_qp(da_size_on_qp);
            for (int el = 0; el < num_el; el++)
            {
               map_fields_to_quadrature_data(
                  input_qp, el, this->fields_e,
                  kinput_to_field, dtqmaps_tensor,
                  integration_weights, kernel.inputs,
                  std::make_index_sequence<kernel.num_kinputs> {});

               map_fields_to_quadrature_data_conditional(
                  directions_qp, el,
                  directions_e, derivative_idx,
                  dtqmaps_tensor,
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

                  auto r_qp = Reshape(&da_qp(0, qp, el), da_size_on_qp);
                  for (int i = 0; i < da_size_on_qp; i++)
                  {
                     r_qp(i) = f_qp(i);
                  }
               }

               map_quadrature_data_to_fields(y_e, el, num_el, da_qp,
                                             output_fop,
                                             dtqmaps_tensor,
                                             test_space_field_idx);
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

      template <typename kernel_t>
      void assemble_vector_impl(kernel_t kernel, Vector &v)
      {
         auto output_fop = std::get<0>(kernel.outputs);

         if constexpr (!std::is_same_v<decltype(output_fop), One>)
         {
            MFEM_ASSERT(false,
                        "trying to get a derivative for a vector from a residual that is not scalar");
         }

         auto kinput_to_field = create_descriptors_to_fields_map(op.fields,
                                                                 kernel.inputs,
                                                                 std::make_index_sequence<kernel.num_kinputs> {});

         auto koutput_to_field = create_descriptors_to_fields_map(op.fields,
                                                                  kernel.outputs,
                                                                  std::make_index_sequence<kernel.num_koutputs> {});

         constexpr int hardcoded_output_idx = 0;
         const int test_space_field_idx = koutput_to_field[hardcoded_output_idx];

         // Create Value output FieldOperator
         auto output_fop_value = Value{std::get<0>(kernel.outputs).field_label};
         output_fop_value.vdim = GetVDim(op.fields[test_space_field_idx]);

         const int num_el = op.mesh.GetNE();
         const int num_qp = op.integration_rule.GetNPoints();

         v.SetSize(op.width);
         v = 0.0;

         // assume only a single element type for now
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
         bool no_qfinput_is_dependent = true;
         for (int i = 0; i < kinput_is_dependent.size(); i++)
         {
            if (kinput_to_field[i] == derivative_idx)
            {
               no_qfinput_is_dependent = false;
               kinput_is_dependent[i] = true;
               out << "function input " << i << " is dependent on "
                   << op.fields[kinput_to_field[i]].field_label << "\n";
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

         auto dtqmaps_tensor = map_dtqmaps(dtqmaps, this->op.dim, num_qp);

         auto kernel_args = decay_tuple<typename kernel_t::kf_param_ts> {};
         auto kernel_shadow_args = decay_tuple<typename kernel_t::kf_param_ts> {};

         DeviceTensor<1, const double> integration_weights(
            this->op.integration_rule.GetWeights().Read(), num_qp);

         for (int i = 0; i < num_fields; i++)
         {
            directions_e[i].SetSize(fields_e[i].Size());
            directions_e[i] = 1.0;
         }

         derivative_action_e = 0.0;

         // Fields interpolated to the quadrature points in the order of
         // kernel function arguments
         auto input_qp = map_inputs_to_memory(input_qp_mem, num_qp,
                                              kernel.inputs,
                                              std::make_index_sequence<kernel.num_kinputs> {});

         auto directions_qp = map_inputs_to_memory(directions_qp_mem, num_qp,
                                                   kernel.inputs,
                                                   std::make_index_sequence<kernel.num_kinputs> {});

         da_qp_mem = 0.0;
         const auto da_qp = Reshape(da_qp_mem.ReadWrite(), 1, num_qp, num_el);

         for (int el = 0; el < num_el; el++)
         {
            map_fields_to_quadrature_data(
               input_qp, el, this->fields_e,
               kinput_to_field, dtqmaps_tensor,
               integration_weights, kernel.inputs,
               std::make_index_sequence<kernel.num_kinputs> {});

            map_fields_to_quadrature_data_conditional(
               directions_qp, el,
               directions_e, derivative_idx,
               dtqmaps_tensor,
               integration_weights,
               kinput_is_dependent,
               kernel.inputs,
               std::make_index_sequence<kernel.num_kinputs> {});

            for (int qp = 0; qp < num_qp; qp++)
            {
               auto r_qp = Reshape(&da_qp(0, qp, el), 1);

               r_qp(0) = apply_kernel_fwddiff_enzyme(
                            kernel.func,
                            kernel_args,
                            input_qp,
                            kernel_shadow_args,
                            directions_qp,
                            qp)(0);
            }

            map_quadrature_data_to_fields(derivative_action_e, el, num_el, da_qp,
                                          output_fop_value,
                                          dtqmaps_tensor,
                                          test_space_field_idx);
         }

         auto R = get_element_restriction(op.fields[test_space_field_idx],
                                          element_dof_ordering);
         Vector v_l(R->Width());
         R->MultTranspose(derivative_action_e, v_l);

         auto P = get_prolongation(op.fields[test_space_field_idx]);
         P->MultTranspose(v_l, v);
      }

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
         const int test_space_field_idx = koutput_to_field[hardcoded_output_idx];

         int num_el = 0;
         int num_qp = 0;
         if constexpr (std::is_same_v<typename kernel_t::OperatesOn, OperatesOnElement>)
         {
            num_el = op.mesh.GetNE();
            num_qp = op.integration_rule.GetNPoints();
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
               out << "function input " << i << " is dependent on "
                   << op.fields[kinput_to_field[i]].field_label << "\n";
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

         auto dtqmaps_tensor = map_dtqmaps(dtqmaps, this->op.dim, num_qp);

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

         constexpr int fixed_output_idx = 0;
         std::array<bool, kernel.num_koutputs> out_is_dependent;
         out_is_dependent[fixed_output_idx] = true;
         auto output_dtq_ops = create_dtq_operators(kernel.outputs, dtqmaps, op.dim,
                                                    koutput_to_field,
                                                    out_is_dependent, std::make_index_sequence<kernel.num_koutputs> {});
         auto [unused0, test_op_dim, num_test_dof] =
            output_dtq_ops[fixed_output_idx].GetShape();

         auto dependent_input_dtq_ops = create_dtq_operators(kernel.inputs, dtqmaps,
                                                             op.dim,
                                                             kinput_to_field,
                                                             kinput_is_dependent, std::make_index_sequence<kernel.num_kinputs> {});
         const int num_trial_dof = dependent_input_dtq_ops[0].GetShape()[2];

         // number of rows for the derivative of the kernel \partial D
         const int nrows_a = GetSizeOnQP(std::get<0>(kernel.outputs),
                                         op.fields[test_space_field_idx]);

         // number of columns for the derivative of the kernel \partial D
         const int ncols_a = accumulate_sizes_on_qp(kernel.inputs,
                                                    kinput_is_dependent,
                                                    kinput_to_field,
                                                    op.fields,
                                                    std::make_index_sequence<kernel.num_kinputs> {});

         Vector a_qp_mem(nrows_a * ncols_a * num_qp * num_el);
         const auto a_qp = Reshape(a_qp_mem.ReadWrite(), nrows_a, ncols_a, num_qp,
                                   num_el);

         Vector A_l(num_test_dof * num_trial_dof * num_el);
         A_l = 0.0;

         auto A_e = Reshape(A_l.ReadWrite(), num_test_dof, num_trial_dof,
                            num_el);

         for (int el = 0; el < num_el; el++)
         {
            map_fields_to_quadrature_data(
               input_qp, el, this->fields_e,
               kinput_to_field, dtqmaps_tensor,
               integration_weights, kernel.inputs,
               std::make_index_sequence<kernel.num_kinputs> {});

            for (int qp = 0; qp < num_qp; qp++)
            {
               size_t colidx_a = 0;
               for (int d_idx = 0; d_idx < directions_qp.size(); d_idx++)
               {
                  if (!kinput_is_dependent[d_idx]) { continue; }
                  auto [dir_soq, unused] = directions_qp[d_idx].GetShape();
                  auto d_qp = Reshape(&(directions_qp[d_idx])[0], dir_soq, num_qp);
                  for (int j = 0; j < dir_soq; j++)
                  {
                     d_qp(j, qp) = 1.0;
                     Vector f_qp = apply_kernel_fwddiff_enzyme(
                                      kernel.func,
                                      kernel_args,
                                      input_qp,
                                      kernel_shadow_args,
                                      directions_qp,
                                      qp);
                     d_qp(j, qp) = 0.0;
                     for (int rowidx_a = 0; rowidx_a < nrows_a; rowidx_a++)
                     {
                        a_qp(rowidx_a, colidx_a, qp, el) = f_qp(rowidx_a);
                     }
                     colidx_a++;
                  }
               }
               DenseMatrix adense(&a_qp(0, 0, qp, el), nrows_a, ncols_a);
               print_matrix(adense);
            }

            // compute B_v^T A_e B_u with N = #dofs, Q = #qp, D = dimension
            //
            // B_u is (size of dependent fields on qp) x N
            // A_e is (size of test function on qp) x (size of dependent fields on qp)
            // B_v is (size of test function on qp) x N
            //
            // Example for (\partial u_i / \partial u_i phi_j, phi_i):
            // [Q x Ni]^T [Q x Q] [Q x Nj] = [Ni x Nj]
            //
            // Example for (\partial (\nabla u_i phi_j) / \partial u_i, \nabla phi_i):
            // [DQ x N]^T [DQ x DQ] [DQ x N] = [N x N]
            //
            // Example for (\partial |u_i * phi_j| (\nabla u_i phi_j) / \partial u_i, \nabla phi_i):
            // [DQ x N]^T [DQ x DQ+Q] [DQ+Q x N] = [N x N]
            //
            // [DQ x N]^T [DQ x DQ] [DQ x N] = [N x N]
            // [      ]   [Q x Q] [Q x N]

            auto Bv = output_dtq_ops[fixed_output_idx];
            auto [num_test_qp, test_op_dim, num_test_dof] = Bv.GetShape();

            // for (int trial_dof = 0; trial_dof < num_trial_dof; trial_dof++)
            // {
            //    int a_coloffset = 0;
            //    for (int d_idx = 0; d_idx < dependent_input_dtq_ops.size(); d_idx++)
            //    {
            //       auto Bu = dependent_input_dtq_ops[d_idx];
            //       auto [unused1, trial_op_dim, unused2] = Bu.GetShape();
            //       for (int test_dof = 0; test_dof < num_test_dof; test_dof++)
            //       {
            //          for (int qp = 0; qp < num_qp; qp++)
            //          {
            //             double acc = 0.0;
            //             for (int k = 0; k < test_op_dim; k++)
            //             {
            //                for (int l = 0; l < trial_op_dim; l++)
            //                {
            //                   acc += Bv(qp, k, test_dof) * a_qp(k, l + a_coloffset, qp, el) * Bu(qp, l,
            //                                                                                      trial_dof);
            //                }
            //             }
            //             A_e(test_dof, trial_dof, el) += acc;
            //          }
            //       }
            //       a_coloffset += trial_op_dim;
            //    }
            // }

            Vector fhat_mem(test_op_dim * num_qp * num_el);
            auto fhat = Reshape(fhat_mem.ReadWrite(), test_op_dim, num_qp, num_el);

            Vector bvtfhat_mem(num_test_dof * num_el);
            auto bvtfhat = Reshape(bvtfhat_mem.ReadWrite(), num_test_dof, num_el);

            for (int d_idx = 0; d_idx < dependent_input_dtq_ops.size(); d_idx++)
            {
               auto Bu = dependent_input_dtq_ops[d_idx];
               auto [num_trial_qp, trial_op_dim, num_trial_dof] = Bu.GetShape();
               for (int trial_dof = 0; trial_dof < num_trial_dof; trial_dof++)
               {
                  fhat_mem = 0.0;
                  for (int qp = 0; qp < num_qp; qp++)
                  {
                     for (int k = 0; k < test_op_dim; k++)
                     {
                        for (int m = 0; m < trial_op_dim; m++)
                        {
                           fhat(k, qp, el) += a_qp(k, m, qp, el) * Bu(qp, m, trial_dof);
                        }
                     }
                  }

                  bvtfhat_mem = 0.0;
                  map_quadrature_data_to_fields(bvtfhat_mem, el, num_el, fhat, output_fop,
                                                dtqmaps_tensor,
                                                test_space_field_idx);

                  for (int test_dof = 0; test_dof < num_test_dof; test_dof++)
                  {
                     A_e(test_dof, trial_dof, el) += bvtfhat(test_dof, el);
                  }
               }
            }
         }

         SparseMatrix mat(GetVSize(op.fields[koutput_to_field[0]]),
                          GetVSize(op.fields[kinput_to_field[0]]));

         auto test_fes = *std::get_if<const ParFiniteElementSpace *>
                         (&op.fields[koutput_to_field[0]].data);

         if (test_fes == nullptr)
         {
            MFEM_ABORT("error");
         }

         for (int el = 0; el < num_el; el++)
         {
            auto tmp = Reshape(A_l.ReadWrite(), num_test_dof, num_trial_dof,
                               num_el);
            DenseMatrix A_e(&tmp(0, 0, el), num_test_dof, num_trial_dof);
            Array<int> vdofs;
            test_fes->GetElementVDofs(el, vdofs);
            mat.AddSubMatrix(vdofs, vdofs, A_e, 1);
         }
         mat.Finalize();

         HypreParMatrix tmp(test_fes->GetComm(),
                            test_fes->GlobalVSize(),
                            test_fes->GetDofOffsets(),
                            &mat);

         if (A.Height() == 0)
         {
            A = *RAP(&tmp, test_fes->Dof_TrueDof_Matrix());
            A.EliminateBC(op.ess_tdof_list, DiagonalPolicy::DIAG_ONE);
         }
         else
         {
            auto A_part = RAP(&tmp, test_fes->Dof_TrueDof_Matrix());
            A += *A_part;
            A.EliminateBC(op.ess_tdof_list, DiagonalPolicy::DIAG_ONE);
            delete A_part;
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
      if (solutions.size() > 1)
      {
         MFEM_ABORT("only one trial space allowed at the moment");
      }

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

private:
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
