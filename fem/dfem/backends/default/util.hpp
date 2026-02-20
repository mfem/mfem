#pragma once

#include "../fem/quadinterpolator.hpp"
#include "../../util.hpp"
#include "general/enzyme.hpp"

namespace mfem::future
{

template <size_t N, size_t... Is>
constexpr std::array<bool, N> all_true_impl(std::index_sequence<Is...>)
{
   return {{((void)Is, true)...}};
}

template <size_t N>
constexpr std::array<bool, N> all_true()
{
   return all_true_impl<N>(std::make_index_sequence<N> {});
}

template <typename fops_t, size_t ninputs = tuple_size<fops_t>::value>
void interpolate(
   const fops_t &fops,
   const std::unordered_map<int, const QuadratureInterpolator *> &qis,
   const IntegrationRule &ir,
   const std::vector<Vector *> &xe,
   BlockVector &xq,
   const std::array<bool, ninputs> &conditional = all_true<ninputs>())
{
   constexpr_for<0, ninputs>([&](auto i)
   {
      if (!conditional.empty() && !conditional[i]) { return; }

      const auto input = get<i>(fops);
      using input_t = std::decay_t<decltype(input)>;

      if constexpr (is_weight_fop<input_t>::value)
      {
         Vector &w = xq.GetBlock(i);

         const int nqp = ir.GetNPoints();
         MFEM_ASSERT(w.Size() % nqp == 0, "weight block has unexpected size");

         const int ne = w.Size() / nqp;
         const real_t *wref = ir.GetWeights().Read();

         for (int e = 0; e < ne; e++)
         {
            std::memcpy(w.GetData() + e*nqp, wref, nqp*sizeof(real_t));
         }
         return;
      }

      auto search = qis.find(input.GetFieldId());
      MFEM_ASSERT(search != qis.end(),
                  "can't find QuadratureInterpolator for given ID " << input.GetFieldId());
      auto qi = search->second;

      qi->SetOutputLayout(QVectorLayout::byVDIM);

      if constexpr (is_value_fop<input_t>::value)
      {
         qi->Values(*xe[i], xq.GetBlock(i));
      }
      else if constexpr (is_gradient_fop<input_t>::value)
      {
         qi->Derivatives(*xe[i], xq.GetBlock(i));
      }
      else
      {
         MFEM_ABORT("default backend doesn't support " << get_type_name<input_t>());
      }
   });
}

// Create quadrature function output to output fields map
template <typename fops_t, size_t nfops>
void create_output_to_outfd(const fops_t &fops,
                            const std::vector<FieldDescriptor> &fields,
                            std::array<size_t, nfops> &output_to_outfd)
{
   constexpr_for<0, nfops>([&](auto i)
   {
      const auto output = get<i>(fops);
      output_to_outfd[i] = std::numeric_limits<size_t>::max();
      for (size_t j = 0; j < fields.size(); j++)
      {
         // TODO: output.GetFieldId() should probably store/return size_t
         if (static_cast<int>(fields[j].id) == output.GetFieldId())
         {
            output_to_outfd[i] = j;
         }
      }
      if (output_to_outfd[i] == std::numeric_limits<size_t>::max())
      {
         MFEM_ABORT("can't find field referenced in quadrature function output");
      }
   });
}

template <typename fops_t, size_t noutputs>
void integrate(
   const fops_t &fops,
   const std::array<size_t, noutputs> &output_to_outfd,
   const std::unordered_map<int, const QuadratureInterpolator *> &qis,
   const BlockVector &yq,
   std::vector<Vector *> &ye)
{
   std::cout << "integrate\n";
   for (auto v : ye) { *v = 0.0; }
   constexpr_for<0, noutputs>([&](auto i)
   {
      for (size_t l = 0; l < ye.size(); l++)
      {
         std::cout << "ye[" << l << "] = ";
         pretty_print(*ye[l]);
      }

      const auto output = get<i>(fops);
      using output_t = std::decay_t<decltype(output)>;

      // Weights are not outputs - they're only inputs
      if constexpr (is_weight_fop<output_t>::value)
      {
         MFEM_ABORT("Weight cannot be an output");
         return;
      }

      const size_t j = output_to_outfd[i];
      // Check that output vector is allocated
      MFEM_ASSERT(ye[j] != nullptr, "output vector ye[" << j << "] is null");

      std::cout << "filling ye[" << j << "]\n";

      auto search = qis.find(output.GetFieldId());
      MFEM_ASSERT(search != qis.end(),
                  "can't find QuadratureInterpolator for given ID " << output.GetFieldId());
      auto qi = search->second;
      qi->SetOutputLayout(QVectorLayout::byVDIM);

      // For unused arguments
      Vector empty;

      if constexpr (is_value_fop<output_t>::value)
      {
         // Integrate values: Q -> E
         qi->AddMultTranspose(QuadratureInterpolator::VALUES,
                              yq.GetBlock(i), empty, *ye[j]);
      }
      else if constexpr (is_gradient_fop<output_t>::value)
      {
         MFEM_ABORT("errrr");
         // Integrate gradients: Q -> E
         // qi->MultTranspose(QuadratureInterpolator::DERIVATIVES,
         //                   empty, yq.GetBlock(i), *ye[i]);
      }
      else
      {
         MFEM_ABORT("default backend doesn't support " << get_type_name<output_t>());
      }

      std::cout << "after AddMultTranspose\n";
      for (size_t l = 0; l < ye.size(); l++)
      {
         std::cout << "ye[" << l << "] = ";
         pretty_print(*ye[l]);
      }
   });
}


namespace detail
{

template <typename T>
struct is_tensor_array : std::false_type {};

template <typename scalar_t, int... Dims>
struct is_tensor_array<tensor_array<scalar_t, Dims...>> : std::true_type {};

template <typename T>
struct is_tensor_array_mut : std::false_type {};

template <typename scalar_t, int... Dims>
struct is_tensor_array_mut<tensor_array<scalar_t, Dims...>> :
                                                            std::bool_constant<!std::is_const_v<scalar_t>> {};


template <typename ndarray_t>
inline void set_layout(ndarray_t &a)
{
   if constexpr (ndarray_t::tensor_rank() == 0) { return; }

   constexpr std::size_t nd = ndarray_t::rank();
   constexpr std::size_t td = ndarray_t::tensor_rank();
   std::array<std::size_t, nd + td> perm{};

   for (std::size_t i = 0; i < td; i++) { perm[i] = nd + i; }
   for (std::size_t i = 0; i < nd; i++) { perm[td + i] = i; }

   a.set_layout(perm);
}

/// Primary template: intentionally undefined — gives a clear error for unsupported types.
template <typename T>
struct tensor_array_traits;

/// Matches tensor<scalar_t, sizes...>
template <typename scalar_t, int... sizes>
struct tensor_array_traits<tensor<scalar_t, sizes...>>
{
   using scalar_type = scalar_t;
   template <std::size_t ndims>
   using array_type = tensor_ndarray<scalar_t, ndims, sizes...>;
};

/// Matches tensor_ndarray<scalar_t, ndims, tensor_sizes...>
template <typename scalar_t, int ndims, int... tensor_sizes>
struct tensor_array_traits<tensor_ndarray<scalar_t, ndims, tensor_sizes...>>
{
   using scalar_type = scalar_t;
   template <std::size_t N>
   using array_type = tensor_ndarray<scalar_t, N, tensor_sizes...>;
};

/// Entry point: explicit tensor type T as template argument.
template <typename T, typename ptr_scalar_t, typename... dyn_sizes_t>
decltype(auto) make_tensor_array(ptr_scalar_t *ptr,
                                 dyn_sizes_t... dynamic_sizes)
{
   using traits = tensor_array_traits<T>;
   using array_t = typename traits::template array_type<sizeof...(dynamic_sizes)>;
   auto a = array_t(ptr, {std::size_t(dynamic_sizes)...});
   set_layout(a);
   return a;
}

template <typename qfunc_t, typename inputs_t, typename outputs_t>
struct supports_tensor_array_qfunc
{
   using qf_signature = typename get_function_signature<qfunc_t>::type;
   using qf_param_ts = typename qf_signature::parameter_ts;

   static constexpr int ninputs = tuple_size<inputs_t>::value;
   static constexpr int noutputs = tuple_size<outputs_t>::value;
   static constexpr int nparams = tuple_size<qf_param_ts>::value;

   template <std::size_t... Is>
   static constexpr bool InputsOk(std::index_sequence<Is...>)
   {
      return (is_tensor_array<std::remove_cv_t<std::remove_reference_t<
              typename tuple_element<Is, qf_param_ts>::type>>>::value && ...);
   }

   template <std::size_t... Is>
   static constexpr bool OutputsOk(std::index_sequence<Is...>)
   {
      return (is_tensor_array_mut<std::remove_cv_t<std::remove_reference_t<
              typename tuple_element<ninputs + Is, qf_param_ts>::type>>>::value && ...);
   }

   static constexpr bool value =
      (nparams == ninputs + noutputs) &&
      InputsOk(std::make_index_sequence<ninputs> {}) &&
      OutputsOk(std::make_index_sequence<noutputs> {});
};

template <typename qfunc_t, std::size_t... Is, std::size_t... Os>
inline void call_qfunc(
   const qfunc_t &qfunc,
   const BlockVector &xq,
   BlockVector &yq,
   int gnqp,
   std::index_sequence<Is...>,
   std::index_sequence<Os...>)
{
   constexpr std::size_t ninputs = sizeof...(Is);

   using qf_signature = typename get_function_signature<qfunc_t>::type;
   using qf_param_ts = typename qf_signature::parameter_ts;

   qfunc(
      make_tensor_array<std::remove_cv_t<std::remove_reference_t<
      typename tuple_element<Is, qf_param_ts>::type>>>(
         xq.GetBlock(Is).Read(), gnqp)...,
      make_tensor_array<std::remove_cv_t<std::remove_reference_t<
      typename tuple_element<ninputs + Os, qf_param_ts>::type>>>(
         yq.GetBlock(Os).ReadWrite(), gnqp)...);
}

template <typename func_t, typename... arg_ts>
MFEM_HOST_DEVICE inline
auto qfunction_wrapper(const func_t &f, arg_ts...args)
{
   return f(args...);
}

template <std::size_t derivative_id, std::size_t I, typename Tuple, std::size_t... Is>
constexpr std::array<bool, sizeof...(Is)>
make_activity_array(std::index_sequence<Is...>)
{
   return { (std::decay_t<typename tuple_element<Is, Tuple>::type>::GetFieldId() == derivative_id)... };
}

template <std::size_t derivative_id, typename inputs_t, std::size_t... Is>
constexpr auto make_activity_map_impl(std::index_sequence<Is...>)
{
   constexpr std::size_t N = sizeof...(Is);

   if constexpr (N == 0)
      return std::array<bool, 0> {};

   return make_activity_array<derivative_id, 0, inputs_t>
          (std::make_index_sequence<N> {});
}

template <std::size_t derivative_id, typename inputs_t>
constexpr auto make_activity_map(inputs_t)
{
   return make_activity_map_impl<derivative_id, inputs_t>(
             std::make_index_sequence<tuple_size<inputs_t>::value> {});
}

namespace enzyme_detail
{

template <auto wrapper_fn, typename qf_return_t, typename... AccArgs>
__attribute__((always_inline)) inline void
do_enzyme_call(AccArgs... acc)
{
   std::cout << "__enzyme_fwddiff args:\n";
   int i = 0;
   ((std::cout << "  [" << i++ << "] " << get_type_name<AccArgs>() << "\n"), ...);
   __enzyme_fwddiff<qf_return_t>(wrapper_fn, acc...);
}

template <auto wrapper_fn, typename qf_return_t,
          size_t CurO, size_t NO,
          typename primals_t, typename derivs_t,
          typename... AccArgs>
__attribute__((always_inline)) inline void
process_outputs(primals_t &primals, derivs_t &derivs, AccArgs... acc)
{
   if constexpr (CurO == NO)
   {
      do_enzyme_call<wrapper_fn, qf_return_t>(acc...);
   }
   else
   {
      process_outputs<wrapper_fn, qf_return_t, CurO + 1, NO>(
         primals, derivs,
         acc...,
         enzyme_dupnoneed,
         &std::get<CurO>(primals),
         &std::get<CurO>(derivs));
   }
}

template <auto wrapper_fn, typename qf_return_t,
          size_t CurI, size_t NI, bool... ActivityMap,
          typename inputs_t, typename shadows_t,
          typename primals_t, typename derivs_t,
          typename... AccArgs>
__attribute__((always_inline)) inline void
process_inputs(inputs_t &inputs, shadows_t &shadows,
               primals_t &primals, derivs_t &derivs,
               AccArgs... acc)
{
   if constexpr (CurI == NI)
   {
      constexpr size_t NO = std::tuple_size_v<primals_t>;
      process_outputs<wrapper_fn, qf_return_t, 0, NO>(
         primals, derivs, acc...);
   }
   else
   {
      constexpr bool active =
         std::array<bool, sizeof...(ActivityMap)> {ActivityMap...} [CurI];

      if constexpr (active)
      {
         process_inputs<wrapper_fn, qf_return_t, CurI + 1, NI, ActivityMap...>(
            inputs, shadows, primals, derivs,
            acc...,
            enzyme_dup,
            &std::get<CurI>(inputs),
            &std::get<CurI>(shadows));
      }
      else
      {
         process_inputs<wrapper_fn, qf_return_t, CurI + 1, NI, ActivityMap...>(
            inputs, shadows, primals, derivs,
            acc...,
            enzyme_const,
            &std::get<CurI>(inputs));
      }
   }
}

} // namespace enzyme_detail

template <size_t derivative_id, typename qfunc_t, typename inputs_t, typename outputs_t,
          std::size_t... Is, std::size_t... Os>
inline void enzyme_fwddiff(
   qfunc_t &qfunc,
   const BlockVector &xq,
   const BlockVector &shadow_xq,
   BlockVector &yq,
   const int &gnqp,
   std::index_sequence<Is...>,
   std::index_sequence<Os...>)
{
#ifdef MFEM_USE_ENZYME
   constexpr std::size_t ninputs  = sizeof...(Is);
   constexpr std::size_t noutputs = sizeof...(Os);

   using qf_signature = typename get_function_signature<qfunc_t>::type;
   using qf_param_ts  = typename qf_signature::parameter_ts;
   using qf_return_t  = typename qf_signature::return_t;

   constexpr auto activity_map = make_activity_map<derivative_id>(inputs_t{});
   static_assert(activity_map.size() == ninputs, "activity map size mismatch");

   auto inputs = std::make_tuple(
                    make_tensor_array<std::remove_cv_t<std::remove_reference_t<
                    typename tuple_element<Is, qf_param_ts>::type>>>(
                       xq.GetBlock(Is).Read(), gnqp)...);

   auto shadows = std::make_tuple(
                     make_tensor_array<std::remove_cv_t<std::remove_reference_t<
                     typename tuple_element<Is, qf_param_ts>::type>>>(
                        shadow_xq.GetBlock(Is).Read(), gnqp)...);

   std::array<Vector, noutputs> primal_storage;
   ((primal_storage[Os].SetSize(yq.GetBlock(Os).Size())), ...);

   auto primals_out = std::make_tuple(
                         make_tensor_array<std::remove_cv_t<std::remove_reference_t<
                         typename tuple_element<ninputs + Os, qf_param_ts>::type>>>(
                            primal_storage[Os].ReadWrite(), gnqp)...);

   auto derivs_out = std::make_tuple(
                        make_tensor_array<std::remove_cv_t<std::remove_reference_t<
                        typename tuple_element<ninputs + Os, qf_param_ts>::type>>>(
                           yq.GetBlock(Os).ReadWrite(), gnqp)...);

   using wrapper_fn_t = qf_return_t (*)(
                           const qfunc_t &,
                           std::remove_reference_t<decltype(std::get<Is>(inputs))>...,
                           std::remove_reference_t<decltype(std::get<Os>(primals_out))>...);

   constexpr wrapper_fn_t wrapper_fn =
      qfunction_wrapper<qfunc_t,
      std::remove_reference_t<decltype(std::get<Is>(inputs))>...,
      std::remove_reference_t<decltype(std::get<Os>(primals_out))>...>;

   // wrapper_fn travels as a non-type template parameter throughout without
   // being stored.
   enzyme_detail::process_inputs<
   wrapper_fn,
   qf_return_t,
   0,
   ninputs,
   activity_map[Is]...
   >(inputs, shadows,
     primals_out, derivs_out,
     enzyme_const, &qfunc // seed: qfunc is always inactive
    );

#else
   MFEM_ABORT("enzyme_fwddiff requires MFEM_USE_ENZYME");
#endif
}

} // namespace detail

}
