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

#include "../../quadinterpolator.hpp"
#include "../util.hpp"
#include "../../../general/enzyme.hpp"
#include "../../../linalg/dual.hpp"

#define DFEM_USE_OWN_TUPLE
#ifndef DFEM_USE_OWN_TUPLE
#include <tuple>
#else
#include "fem/dfem/tuple.hpp"
using mfem::future::tuple;
using mfem::future::tuple_size;
using mfem::future::tuple_element;
using mfem::future::tuple_element_t;
#endif

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

inline void InitBlockVector(BlockVector &v, Array<int> &offsets)
{
   v.Update(offsets, Device::GetMemoryType());
   v.UseDevice(true);
   v = 0.0;
   v.SyncToBlocks();
}

struct FieldBasis
{
   // E-vector -> Q-vector
   std::function<void(const Vector &, Vector &)> forward;

   // Q-vector -> E-vector
   std::function<void(const Vector &, Vector &)> transpose;
};

inline FieldBasis FromQI(const QuadratureInterpolator *qi,
                         QuadratureInterpolator::EvalFlags mode)
{
   return
   {
      [qi, mode](const Vector &xe, Vector &xq)
      {
         qi->SetOutputLayout(QVectorLayout::byVDIM);
         if (mode == QuadratureInterpolator::VALUES)
         {
            qi->Values(xe, xq);
         }
         else
         {
            qi->Derivatives(xe, xq);
         }
      },
      [qi, mode](const Vector &yq, Vector &ye)
      {
         Vector empty;
         qi->SetOutputLayout(QVectorLayout::byVDIM);
         if (mode == QuadratureInterpolator::VALUES)
         {
            qi->AddMultTranspose(QuadratureInterpolator::VALUES, yq, empty, ye);
         }
         else
         {
            qi->AddMultTranspose(QuadratureInterpolator::DERIVATIVES, empty, yq, ye);
         }
      }
   };
}

// VectorQuadratureSpace identity copy
inline FieldBasis FromQS()
{
   return
   {
      [](const Vector &xe, Vector &xq)
      {
         xq.NewMemoryAndSize(xe.GetMemory(), xe.Size(), false);
      },
      [](const Vector &yq, Vector &ye) { ye = yq; }
   };
}

// User-defined parameter space B
inline FieldBasis FromPS(const Operator *B, const Operator *Bt)
{
   return
   {
      [B](const Vector &xe, Vector &xq) { B->Mult(xe, xq); },
      [Bt](const Vector &yq, Vector &ye) { Bt->Mult(yq, ye); }
   };
}

inline FieldBasis FieldBasisFromWeight(const IntegrationRule &ir)
{
   return
   {
      [&ir](const Vector &, Vector &xq)
      {
         const int nqp = ir.GetNPoints();
         MFEM_ASSERT(xq.Size() % nqp == 0, "weight block has unexpected size");
         const int ne = xq.Size() / nqp;
         const auto wref = ir.GetWeights().Read();
         auto xq_w = Reshape(xq.Write(), nqp, ne);
         mfem::forall(ne * nqp, [=] MFEM_HOST_DEVICE(int eq)
         {
            const int q = eq % nqp, e = eq / nqp;
            xq_w(q,e) = wref[q];
         });
      },
      [](const Vector &, Vector &) {}
   };
}

inline const FieldBasis GetFieldBasis(const FieldDescriptor &f,
                                      const IntegrationRule &ir,
                                      QuadratureInterpolator::EvalFlags mode)
{
   return std::visit([&ir, &mode](auto && arg) -> FieldBasis
   {
      using T = std::decay_t<decltype(arg)>;

      if constexpr (std::is_same_v<T, const FiniteElementSpace *>)
      {
         return FromQI(arg->GetQuadratureInterpolator(ir), mode);
      }
      else if constexpr (std::is_same_v<T, const ParFiniteElementSpace *>)
      {
         return FromQI(arg->GetQuadratureInterpolator(ir), mode);
      }
      else if constexpr (std::is_same_v<T, const VectorQuadratureSpace *>)
      {
         return FromQS();
      }
      else if constexpr (std::is_same_v<T, const ParameterSpace *>)
      {
         return FromPS(arg->GetB(), arg->GetBt());
      }
      else if constexpr (std::is_same_v<T, const IntegrationRule *>)
      {
         return FieldBasis{};
      }
      else
      {
         static_assert(dfem::always_false<T>, "internal error");
      }
   }, f.data);
}

template <typename fops_t, size_t nfops>
inline void create_fieldbases(
   fops_t &fops,
   const std::array<size_t, nfops> &fop_to_fd,
   const std::vector<FieldDescriptor> &fds,
   const IntegrationRule &ir,
   std::array<FieldBasis, nfops> &bases)
{
   constexpr_for<0, nfops>([&](auto i)
   {
      const auto fop = get<i>(fops);
      using fop_t = std::decay_t<decltype(fop)>;

      const auto fd = fds[fop_to_fd[i]];

      constexpr QuadratureInterpolator::EvalFlags dummy_mode =
         QuadratureInterpolator::VALUES;
      if constexpr (is_identity_fop<fop_t>::value)
      {
         bases[i] = GetFieldBasis(fd, ir, dummy_mode);
      }
      else if constexpr (is_weight_fop<fop_t>::value)
      {
         bases[i] = FieldBasisFromWeight(ir);
      }
      else if constexpr (is_value_fop<fop_t>::value)
      {
         bases[i] = GetFieldBasis(fd, ir, QuadratureInterpolator::VALUES);
      }
      else if constexpr (is_gradient_fop<fop_t>::value)
      {
         bases[i] = GetFieldBasis(fd, ir, QuadratureInterpolator::DERIVATIVES);
      }
   });
}

template <typename fops_t, size_t nfops>
inline void check_consistency(
   fops_t &fops,
   const std::array<size_t, nfops> &fop_to_fd,
   const std::vector<FieldDescriptor> &fields)
{
   constexpr_for<0, nfops>([&](auto i)
   {
      const auto input = get<i>(fops);
      using input_t = std::decay_t<decltype(input)>;

      const auto fd = fields[fop_to_fd[i]];

      if constexpr (is_identity_fop<input_t>::value)
      {
         MFEM_ASSERT(std::holds_alternative<const VectorQuadratureSpace *>(fd.data),
                     "Identity FieldOperator requested on non "
                     "VectorQuadratureSpace");
      }
      else if constexpr (is_weight_fop<input_t>::value)
      {
         MFEM_CONTRACT_VAR(fd);
      }
      else if constexpr (is_value_fop<input_t>::value)
      {
         MFEM_ASSERT(std::holds_alternative<const FiniteElementSpace *>(fd.data) ||
                     std::holds_alternative<const ParFiniteElementSpace *>(fd.data) ||
                     std::holds_alternative<const ParameterSpace *>(fd.data),
                     "Value FieldOperator requested on non "
                     "VectorQuadratureSpace");
      }
      else if constexpr (is_gradient_fop<input_t>::value)
      {
         MFEM_ASSERT(std::holds_alternative<const FiniteElementSpace *>(fd.data) ||
                     std::holds_alternative<const ParFiniteElementSpace *>(fd.data),
                     "Value FieldOperator requested on non "
                     "VectorQuadratureSpace");
      }
   });
}

template <size_t ninputs>
inline void interpolate(
   const std::array<size_t, ninputs> &input_to_infd,
   const std::array<FieldBasis, ninputs> &input_bases,
   const std::vector<Vector *> &xe,
   BlockVector &xq,
   const std::array<bool, ninputs> &conditional = all_true<ninputs>())
{
   constexpr_for<0, ninputs>([&](auto i)
   {
      if (!conditional.empty() && !conditional[i]) { return; }

      input_bases[i].forward(*xe[input_to_infd[i]], xq.GetBlock(i));
   });
}

template <size_t noutputs>
inline void integrate(
   const std::array<size_t, noutputs> &output_to_outfd,
   const std::array<FieldBasis, noutputs> &output_bases,
   const BlockVector &yq,
   std::vector<Vector *> &ye)
{
   constexpr_for<0, noutputs>([&](auto i)
   {
      output_bases[i].transpose(yq.GetBlock(i), *ye[output_to_outfd[i]]);
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
inline void set_layout_default(ndarray_t &a)
{
   if constexpr (ndarray_t::tensor_rank() == 0) { return; }

   constexpr std::size_t nd = ndarray_t::rank();
   constexpr std::size_t td = ndarray_t::tensor_rank();
   std::array<std::size_t, nd + td> perm{};

   if constexpr (td > 0)
   {
      for (std::size_t i = 0; i < td; i++) { perm[i] = nd + i; }
   }

   if constexpr (nd > 0)
   {
      for (std::size_t i = 0; i < nd; i++) { perm[td + i] = i; }
   }

   a.set_layout(perm);
}

template <typename ndarray_t>
inline void set_layout(ndarray_t& a, const std::vector<int>& layout)
{
   if constexpr (ndarray_t::tensor_rank() == 0) { return; }

   constexpr std::size_t nd = ndarray_t::rank();
   constexpr std::size_t td = ndarray_t::tensor_rank();
   constexpr std::size_t N  = nd + td;

   // missing means default
   if (layout.empty()) { set_layout_default(a); return; }

   MFEM_VERIFY(layout.size() == N,
               "layout size mismatch: expected " << N << " got " << layout.size());

   // TODO: make a version of set_layout that takes `std::vector<int>`
   std::array<std::size_t, N> perm{};
   for (std::size_t i = 0; i < N; i++)
   {
      MFEM_VERIFY(layout[i] >= 0, "layout index must be >=0");
      perm[i] = static_cast<std::size_t>(layout[i]);
   }

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
                                 const std::vector<int>* layout,
                                 dyn_sizes_t... dynamic_sizes)
{
   using traits = tensor_array_traits<T>;
   using array_t = typename traits::template array_type<sizeof...(dynamic_sizes)>;
   auto a = array_t(ptr, {std::size_t(dynamic_sizes)...});
   if (layout) { set_layout(a, *layout); }
   else        { set_layout_default(a); }
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

template <typename T>
struct qp_type_uses_dual : std::false_type {};

template <typename V, typename G>
struct qp_type_uses_dual<dual<V, G>> : std::true_type {};

template <typename S, int... Is>
struct qp_type_uses_dual<tensor<S, Is...>> : qp_type_uses_dual<S> {};

template <typename S, int ndims, int... Is>
struct qp_type_uses_dual<tensor_ndarray<S, ndims, Is...>>
                                                          : qp_type_uses_dual<std::remove_cv_t<S>> {};

template <typename T>
inline constexpr bool qp_type_uses_dual_v =
   qp_type_uses_dual<std::remove_cv_t<std::remove_reference_t<T>>>::value;

inline void pack_dual_from_primal_shadow(
   const real_t *primal,
   const real_t *shadow,
   dual<real_t, real_t> *out,
   const int n,
   const bool active)
{
   for (int i = 0; i < n; i++)
   {
      out[i].value = primal[i];
      out[i].gradient = active ? shadow[i] : real_t(0);
   }
}

template <bool unpack_primal_values>
inline void unpack_dual_to_real(
   const dual<real_t, real_t> *dual_out,
   real_t *yq,
   const int n)
{
   for (int i = 0; i < n; i++)
   {
      yq[i] = unpack_primal_values ? dual_out[i].value : dual_out[i].gradient;
   }
}

template <typename decay_t, std::size_t I>
decltype(auto) make_native_dual_input(
   const BlockVector &xq,
   const BlockVector &shadow_xq,
   std::vector<dual<real_t, real_t>> &dual_storage,
   const std::vector<int> &layout,
   const int gnqp,
   const bool active)
{
   if constexpr (qp_type_uses_dual_v<decay_t>)
   {
      const int sz = xq.GetBlock(I).Size();
      dual_storage.resize(sz);
      pack_dual_from_primal_shadow(
         xq.GetBlock(I).Read(), shadow_xq.GetBlock(I).Read(),
         dual_storage.data(), sz, active);
      return make_tensor_array<decay_t>(dual_storage.data(), &layout, gnqp);
   }
   else
   {
      return make_tensor_array<decay_t>(xq.GetBlock(I).Read(), &layout, gnqp);
   }
}

template <typename decay_t, std::size_t O>
decltype(auto) make_native_dual_output(
   BlockVector &yq,
   std::vector<dual<real_t, real_t>> &dual_storage,
   const std::vector<int> &layout,
   const int gnqp)
{
   if constexpr (qp_type_uses_dual_v<decay_t>)
   {
      const int sz = yq.GetBlock(O).Size();
      dual_storage.resize(sz);
      return make_tensor_array<decay_t>(dual_storage.data(), &layout, gnqp);
   }
   else
   {
      return make_tensor_array<decay_t>(yq.GetBlock(O).ReadWrite(), &layout, gnqp);
   }
}

template <bool unpack_primal_values, typename decay_t, std::size_t O>
void finish_native_dual_output(
   BlockVector &yq,
   std::vector<dual<real_t, real_t>> &dual_storage)
{
   if constexpr (qp_type_uses_dual_v<decay_t>)
   {
      unpack_dual_to_real<unpack_primal_values>(
         dual_storage.data(), yq.GetBlock(O).HostWrite(), yq.GetBlock(O).Size());
   }
}

template <typename T>
using qf_param_decay_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename qf_param_ts, std::size_t... Is>
constexpr bool qf_signature_uses_dual_impl(std::index_sequence<Is...>)
{
   return (qp_type_uses_dual_v<qf_param_decay_t<
           typename tuple_element<Is, qf_param_ts>::type>> || ...);
}

template <typename qf_param_ts>
inline constexpr bool qf_signature_uses_dual_v =
   qf_signature_uses_dual_impl<qf_param_ts>(
      std::make_index_sequence<tuple_size<qf_param_ts>::value> {});

template <
   bool unpack_primal_values,
   typename qfunc_t,
   std::size_t... Is,
   std::size_t... Os>
inline void native_dual_evaluate_qfunc(
   const qfunc_t &qfunc,
   const BlockVector &xq,
   const BlockVector &shadow_xq,
   BlockVector &yq,
   int gnqp,
   const std::array<std::vector<int>, sizeof...(Is)> &in_layouts,
   const std::array<std::vector<int>, sizeof...(Os)> &out_layouts,
   const std::array<bool, sizeof...(Is)> *input_active,
   std::index_sequence<Is...>,
   std::index_sequence<Os...>)
{
   constexpr std::size_t ninputs = sizeof...(Is);
   constexpr std::size_t noutputs = sizeof...(Os);

   using qf_signature = typename get_function_signature<qfunc_t>::type;
   using qf_param_ts = typename qf_signature::parameter_ts;

   std::array<std::vector<dual<real_t, real_t>>, ninputs> dual_inputs {};
   std::array<std::vector<dual<real_t, real_t>>, noutputs> dual_outputs {};

   using input_decay_ts = std::tuple<qf_param_decay_t<
                          typename tuple_element<Is, qf_param_ts>::type>...>;
   using output_decay_ts = std::tuple<qf_param_decay_t<
                           typename tuple_element<ninputs + Os, qf_param_ts>::type>...>;

   auto inputs =
      std::make_tuple(
         make_native_dual_input<
         std::tuple_element_t<Is, input_decay_ts>, Is>(
            xq, shadow_xq, dual_inputs[Is], in_layouts[Is], gnqp,
            input_active ? (*input_active)[Is] : false)...);

   auto outputs =
      std::make_tuple(
         make_native_dual_output<
         std::tuple_element_t<Os, output_decay_ts>, Os>(
            yq, dual_outputs[Os], out_layouts[Os], gnqp)...);

   std::apply([&](auto &&...args) { qfunc(args...); },
   std::tuple_cat(inputs, outputs));

   constexpr_for<0, noutputs>([&](auto o)
   {
      finish_native_dual_output<
      unpack_primal_values,
      std::tuple_element_t<o, output_decay_ts>, o>(yq, dual_outputs[o]);
   });
}

template <typename qfunc_t, std::size_t... Is, std::size_t... Os>
inline void call_qfunc(
   const qfunc_t &qfunc,
   const BlockVector &xq,
   BlockVector &yq,
   int gnqp,
   const std::array<std::vector<int>, sizeof...(Is)> &in_layouts,
   const std::array<std::vector<int>, sizeof...(Os)> &out_layouts,
   std::index_sequence<Is...> is,
   std::index_sequence<Os...> os)
{
   using qf_signature = typename get_function_signature<qfunc_t>::type;
   using qf_param_ts = typename qf_signature::parameter_ts;

   if constexpr (qf_signature_uses_dual_v<qf_param_ts>)
   {
      native_dual_evaluate_qfunc<true, qfunc_t>(
         qfunc, xq, xq, yq, gnqp, in_layouts, out_layouts, nullptr, is, os);
   }
   else
   {
      constexpr std::size_t ninputs = sizeof...(Is);

      auto inputs =
         std::make_tuple(
            make_tensor_array<qf_param_decay_t<
            typename tuple_element<Is, qf_param_ts>::type>>(
               xq.GetBlock(Is).Read(), &in_layouts[Is], gnqp)...);

      auto outputs =
         std::make_tuple(
            make_tensor_array<qf_param_decay_t<
            typename tuple_element<ninputs + Os, qf_param_ts>::type>>(
               yq.GetBlock(Os).ReadWrite(), &out_layouts[Os], gnqp)...);

      std::apply([&](auto &&...args) { qfunc(args...); },
      std::tuple_cat(inputs, outputs));
   }
}

template <
   size_t derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t,
   std::size_t... Is,
   std::size_t... Os>
inline void native_dual_fwddiff(
   const qfunc_t &qfunc,
   const BlockVector &xq,
   const BlockVector &shadow_xq,
   BlockVector &yq,
   const int &gnqp,
   const std::array<std::vector<int>, sizeof...(Is)> &in_layouts,
   const std::array<std::vector<int>, sizeof...(Os)> &out_layouts,
   std::index_sequence<Is...> is,
   std::index_sequence<Os...> os)
{
   constexpr auto activity_map = make_activity_map<derivative_id>(inputs_t {});
   native_dual_evaluate_qfunc<false, qfunc_t>(
      qfunc, xq, shadow_xq, yq, gnqp, in_layouts, out_layouts, &activity_map, is, os);
}

#ifdef MFEM_USE_ENZYME
namespace enzyme_detail
{

template <auto wrapper_fn, typename qf_return_t, typename... AccArgs>
__attribute__((always_inline)) inline void
do_enzyme_call(AccArgs... acc)
{
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
         // Inactive inputs are passed as enzyme_dup with their (zeroed) shadow,
         // NOT as enzyme_const. Enzyme_const is unsafe here: when an inactive
         // tensor_array is read through its copying accessor (e.g. J(q) ->
         // get_tensor copies into a local tensor, then det(J)), Enzyme fails to
         // zero the shadow of that local copy and propagates a primal-valued
         // tangent (effectively dJ = J), injecting a spurious derivative term.
         process_inputs<wrapper_fn, qf_return_t, CurI + 1, NI, ActivityMap...>(
            inputs, shadows, primals, derivs,
            acc...,
            enzyme_dup,
            &std::get<CurI>(inputs),
            &std::get<CurI>(shadows));
      }
   }
}

} // namespace enzyme_detail

template <size_t derivative_id, typename qfunc_t, typename inputs_t, typename outputs_t,
          std::size_t... Is, std::size_t... Os>
inline void enzyme_fwddiff(
   const qfunc_t &qfunc,
   const BlockVector &xq,
   const BlockVector &shadow_xq,
   BlockVector &yq,
   const int &gnqp,
   const std::array<std::vector<int>, sizeof...(Is)>& in_layouts,
   const std::array<std::vector<int>, sizeof...(Os)>& out_layouts,
   std::index_sequence<Is...>,
   std::index_sequence<Os...>)
{
   constexpr std::size_t ninputs  = sizeof...(Is);
   constexpr std::size_t noutputs = sizeof...(Os);

   using qf_signature = typename get_function_signature<qfunc_t>::type;
   using qf_param_ts  = typename qf_signature::parameter_ts;
   using qf_return_t  = typename qf_signature::return_t;

   constexpr auto activity_map = make_activity_map<derivative_id>(inputs_t {});
   static_assert(activity_map.size() == ninputs, "activity map size mismatch");

   auto inputs = std::make_tuple(
                    make_tensor_array<std::remove_cv_t<std::remove_reference_t<
                    typename tuple_element<Is, qf_param_ts>::type>>>(
                       xq.GetBlock(Is).Read(), &in_layouts[Is], gnqp)...);

   auto shadows = std::make_tuple(
                     make_tensor_array<std::remove_cv_t<std::remove_reference_t<
                     typename tuple_element<Is, qf_param_ts>::type>>>(
                        shadow_xq.GetBlock(Is).Read(), &in_layouts[Is], gnqp)...);

   // forward action: evaluate the QF at the current quadrature data,
   // used to populate the primal_storage vector.
   call_qfunc(qfunc, xq, yq, gnqp, in_layouts, out_layouts,
              std::make_index_sequence<ninputs> {},
              std::make_index_sequence<noutputs> {});

   std::array<Vector, noutputs> primal_storage;
   ((primal_storage[Os].SetSize(yq.GetBlock(Os).Size())), ...);
   ((primal_storage[Os] = yq.GetBlock(Os)), ...);

   auto primals_out = std::make_tuple(
                         make_tensor_array<std::remove_cv_t<std::remove_reference_t<
                         typename tuple_element<ninputs + Os, qf_param_ts>::type>>>(
                            primal_storage[Os].ReadWrite(), &out_layouts[Os], gnqp)...);

   auto derivs_out = std::make_tuple(
                        make_tensor_array<std::remove_cv_t<std::remove_reference_t<
                        typename tuple_element<ninputs + Os, qf_param_ts>::type>>>(
                           yq.GetBlock(Os).ReadWrite(), &out_layouts[Os], gnqp)...);

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
}

#endif // MFEM_USE_ENZYME

template <
   size_t derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t,
   std::size_t... Is,
   std::size_t... Os>
inline void fwddiff(
   const qfunc_t &qfunc,
   const BlockVector &xq,
   const BlockVector &shadow_xq,
   BlockVector &yq,
   const int &gnqp,
   const std::array<std::vector<int>, sizeof...(Is)> &in_layouts,
   const std::array<std::vector<int>, sizeof...(Os)> &out_layouts,
   std::index_sequence<Is...> is,
   std::index_sequence<Os...> os)
{
#ifdef MFEM_USE_ENZYME
   enzyme_fwddiff<derivative_id, qfunc_t, inputs_t, outputs_t>(
      qfunc, xq, shadow_xq, yq, gnqp, in_layouts, out_layouts, is, os);
#else
   native_dual_fwddiff<derivative_id, qfunc_t, inputs_t, outputs_t>(
      qfunc, xq, shadow_xq, yq, gnqp, in_layouts, out_layouts, is, os);
#endif
}

} // namespace detail

// Create quadrature function fop to fields map
template <typename fops_t, size_t N = tuple_size<fops_t>::value, size_t M>
void create_fop_to_fd(const fops_t &fops,
                      const std::vector<FieldDescriptor> &fields,
                      std::array<size_t, M> &fop_to_fd)
{
   static_assert(N == M, "sizes must match");
   constexpr_for<0, N>([&](auto i)
   {
      const auto fop = get<i>(fops);
      fop_to_fd[i] = std::numeric_limits<size_t>::max();
      for (size_t j = 0; j < fields.size(); j++)
      {
         // TODO: output.GetFieldId() should probably store/return size_t
         if (static_cast<int>(fields[j].id) == fop.GetFieldId())
         {
            fop_to_fd[i] = j;
         }
      }
      // Handle Weight type. There is no FieldDescriptor for the weight.
      // TODO: Create weight descriptor for the weight for internal use?
      // TODO: this is a hack...
      if (is_weight_fop<std::remove_cv_t<decltype(fop)>>::value)
      {
         fop_to_fd[i] = 0;
      }
      else if (fop_to_fd[i] == std::numeric_limits<size_t>::max())
      {
         MFEM_ABORT("not found");
      }
   });
}

template <typename fops_t, size_t nfops>
void create_qlayouts(const fops_t &fops,
                     const std::unordered_map<std::type_index, std::vector<int>> &a,
                     std::array<std::vector<int>, nfops> &b)
{
   constexpr_for<0, nfops>([&](auto i)
   {
      using fop_t =
         std::remove_cv_t<std::remove_reference_t<decltype(get<i>(fops))>>;
      auto it = a.find(std::type_index(typeid(fop_t)));
      if (it != a.end()) { b[i] = it->second; }
      else               { b[i].clear(); }
   });
}

///////////////////////////////////////////////////////////////////////////////
// Zero-copy view of a contiguous block as a tensor<T>
template<typename T> inline
MFEM_HOST_DEVICE const tensor<T>& as_tensor(const T* ptr)
{
   return *std::launder(reinterpret_cast<const tensor<T>*>(ptr));
}

template<typename T> inline
MFEM_HOST_DEVICE tensor<T>& as_tensor(T* ptr)
{
   return *std::launder(reinterpret_cast<tensor<T>*>(ptr));
}

// Zero-copy view of a contiguous block as a tensor<T, n1>
template<typename T, int n1> inline
MFEM_HOST_DEVICE const tensor<T, n1>& as_tensor(const T* ptr)
{
   return *std::launder(reinterpret_cast<const tensor<T, n1>*>(ptr));
}

template<typename T, int n1> inline
MFEM_HOST_DEVICE tensor<T, n1>& as_tensor(T* ptr)
{
   return *std::launder(reinterpret_cast<tensor<T, n1>*>(ptr));
}

// Zero-copy view of a contiguous block as a tensor<T, n1, n2>
template<typename T, int n1, int n2> inline
MFEM_HOST_DEVICE const tensor<T, n1, n2>& as_tensor(const T* ptr)
{
   return *std::launder(reinterpret_cast<const tensor<T, n1, n2>*>(ptr));
}

template<typename T, int n1, int n2> inline
MFEM_HOST_DEVICE tensor<T, n1, n2>& as_tensor(T* ptr)
{
   return *std::launder(reinterpret_cast<tensor<T, n1, n2>*>(ptr));
}

// Zero-copy view of a contiguous block as a tensor<T, n1, n2, n3>
template<typename T, int n1, int n2, int n3> inline
MFEM_HOST_DEVICE const tensor<T, n1, n2, n3>& as_tensor(const T* ptr)
{
   return *std::launder(reinterpret_cast<const tensor<T, n1, n2, n3>*>(ptr));
}

template<typename T, int n1, int n2, int n3> inline
MFEM_HOST_DEVICE tensor<T, n1, n2, n3>& as_tensor(T* ptr)
{
   return *std::launder(reinterpret_cast<tensor<T, n1, n2, n3>*>(ptr));
}

// Zero-copy view of a contiguous block as a tensor<T, n1, n2, n3, n4>
template<typename T, int n1, int n2, int n3, int n4> inline
MFEM_HOST_DEVICE const tensor<T, n1, n2, n3, n4>& as_tensor(const T* ptr)
{
   return *std::launder(reinterpret_cast<const tensor<T, n1, n2, n3, n4>*>(ptr));
}

template<typename T, int n1, int n2, int n3, int n4> inline
MFEM_HOST_DEVICE tensor<T, n1, n2, n3, n4>& as_tensor(T* ptr)
{
   return *std::launder(reinterpret_cast<tensor<T, n1, n2, n3, n4>*>(ptr));
}

// Generic zero-copy view using an explicit tensor type (e.g. tensor<real_t, DIM, DIM>)
template <typename Tensor> inline
MFEM_HOST_DEVICE Tensor& as_tensor(real_t* ptr)
{
   return *std::launder(reinterpret_cast<Tensor*>(ptr));
}

template <typename Tensor> inline
MFEM_HOST_DEVICE const Tensor& as_tensor(const real_t* ptr)
{
   return *std::launder(reinterpret_cast<const Tensor*>(ptr));
}

///////////////////////////////////////////////////////////////////////////////
template<typename T>
struct is_std_tuple : std::false_type {};

template<typename... Ts>
struct is_std_tuple<std::tuple<Ts...>> : std::true_type {};

template<typename T>
inline constexpr bool is_std_tuple_v = is_std_tuple<T>::value;

///////////////////////////////////////////////////////////////////////////////
struct Unused
{
   MFEM_HOST_DEVICE int operator[](int) { return int{}; }
};
template<size_t N, size_t M, typename T>
using reg_array_t = std::conditional_t<N == 0, Unused, std::array<T, N + M>>;

///////////////////////////////////////////////////////////////////////////////
template <typename...> struct static_type;


///////////////////////////////////////////////////////////////////////////////
template <typename func_t, typename args_t, int... Is>
MFEM_HOST_DEVICE static void call_qfunc_no_move_impl(
   const func_t &func, args_t &args, std::integer_sequence<int, Is...>)
{
   (void)func(get<Is>(args)...);
}

template <typename func_t, typename args_t>
MFEM_HOST_DEVICE static void call_qfunc_no_move(const func_t &func,
                                                args_t &args)
{
   constexpr int nargs = static_cast<int>(tuple_size<args_t>::value);
   call_qfunc_no_move_impl(func, args, std::make_integer_sequence<int, nargs> {});
}

template <typename qfunc_t, typename args_t, int... Is>
MFEM_HOST_DEVICE static void call_enzyme_fwddiff_impl(
   const qfunc_t &qfunc,
   args_t &primal_args,
   args_t &shadow_args,
   std::integer_sequence<int, Is...>)
{
#ifdef MFEM_USE_ENZYME
   auto wrapper = [](const qfunc_t *qf, decltype(get<Is>(primal_args))&... args)
   {
      (*qf)(args...);
   };
   __enzyme_fwddiff<void>(
      (void (*)(const qfunc_t*, decltype(get<Is>(primal_args))&...))wrapper,
      enzyme_const, &qfunc,
      enzyme_dup, &get<Is>(primal_args)..., enzyme_interleave,
      &get<Is>(shadow_args)...);
#else
   MFEM_CONTRACT_VAR(qfunc);
   MFEM_CONTRACT_VAR(primal_args);
   MFEM_CONTRACT_VAR(shadow_args);
   MFEM_ABORT("Enzyme not available");
#endif
}

template <typename qfunc_t, typename args_t>
MFEM_HOST_DEVICE static void call_enzyme_fwddiff(
   const qfunc_t &qfunc,
   args_t &primal_args,
   args_t &shadow_args)
{
   constexpr int nargs = static_cast<int>(tuple_size<args_t>::value);
   call_enzyme_fwddiff_impl(qfunc, primal_args, shadow_args,
                            std::make_integer_sequence<int, nargs> {});
}

} // namespace mfem::future
