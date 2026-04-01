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

#include <tuple>
#include <utility>

#include "fem/dfem/integrator_ctx.hpp"
#include "fem/dfem/fieldoperator.hpp" // Identity, Value, Gradient
// #include "fem/dfem/tuple.hpp"

#include "fem/pfespace.hpp"
#include "fem/quadinterpolator.hpp"
#include "general/forall.hpp"
#include "linalg/dtensor.hpp"
#include "linalg/tensor_arrays.hpp"
#include "linalg/vector.hpp"

namespace mfem::future::device::detail
{

///////////////////////////////////////////////////////////////////////////////
struct FieldBasis
{
   // E-vector -> Q-vector
   std::function<void(const Vector &, Vector &)> forward;
   // Q-vector -> E-vector
   std::function<void(const Vector &, Vector &)> transpose;
};

///////////////////////////////////////////////////////////////////////////////
template <class F> struct FunctionSignature;

template <typename output_t, typename... input_ts>
struct FunctionSignature<output_t(input_ts...)>
{
   using return_t = output_t;
   using parameter_ts = std::tuple<input_ts...>;
};

template <class T> struct create_function_signature;

// Specialization for member functions (lambdas)
template <typename output_t, typename T, typename... input_ts>
struct create_function_signature<output_t (T::*)(input_ts...) const>
{
   using type = FunctionSignature<output_t(input_ts...)>;
};

// Specialization for function pointers
template <typename output_t, typename... input_ts>
struct create_function_signature<output_t (*)(input_ts...)>
{
   using type = FunctionSignature<output_t(input_ts...)>;
};

///////////////////////////////////////////////////////////////////////////////
template <typename...>
using void_t = void;

template <typename T, typename = void>
struct get_function_signature
{
   using type = typename create_function_signature<T>::type;
};

template <typename T>
struct get_function_signature<T, void_t<decltype(&T::operator())>>
{
   using type = typename create_function_signature<decltype(&T::operator())>::type;
};

///////////////////////////////////////////////////////////////////////////////
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

///////////////////////////////////////////////////////////////////////////////
template <typename ndarray_t>
inline void set_layout_default(ndarray_t &a)
{
   NVTX_MARK_FUNCTION;
   if constexpr (ndarray_t::tensor_rank() == 0) { return; }
   constexpr std::size_t nd = ndarray_t::rank();
   constexpr std::size_t td = ndarray_t::tensor_rank();
   std::array<std::size_t, nd + td> perm{};
   for (std::size_t i = 0; i < td; i++) { perm[i] = nd + i; }
   for (std::size_t i = 0; i < nd; i++) { perm[td + i] = i; }
   a.set_layout(perm);
}

template <typename ndarray_t>
inline void set_layout(ndarray_t& a, const std::vector<int>& layout)
{
   NVTX_MARK_FUNCTION;
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

///////////////////////////////////////////////////////////////////////////////
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
   NVTX_MARK_FUNCTION;
   using traits = tensor_array_traits<T>;
   using array_t = typename traits::template array_type<sizeof...(dynamic_sizes)>;
   auto a = array_t(ptr, {std::size_t(dynamic_sizes)...});
   if (layout) { set_layout(a, *layout); }
   else        { set_layout_default(a); }
   return a;
}

///////////////////////////////////////////////////////////////////////////////
template <auto start, auto end, auto inc = 1, typename F>
constexpr void constexpr_for(F&& f)
{
   if constexpr (start < end)
   {
      f(std::integral_constant<decltype(start), start>());
      constexpr_for<start + inc, end, inc>(f);
   }
}

///////////////////////////////////////////////////////////////////////////////
template <size_t noutputs>
inline void integrate(const std::array<size_t, noutputs> &output_to_outfd,
                      const std::array<FieldBasis, noutputs> &output_bases,
                      const BlockVector &yq,
                      std::vector<Vector *> &ye)
{
   NVTX_MARK_FUNCTION;
   for (auto v : ye) { *v = 0.0; }
   constexpr_for<0, noutputs>([&](auto i)
   {
      NVTX_MARK("out transpose block #{}", i.value);
      output_bases[i].transpose(yq.GetBlock(i), *ye[output_to_outfd[i]]);
   });
}

///////////////////////////////////////////////////////////////////////////////
template <typename qfunc_t, std::size_t... Is, std::size_t... Os>
inline void call_qfunc(const qfunc_t &qfunc,
                       const BlockVector &xq,
                       BlockVector &yq,
                       int gnqp,
                       const std::array<std::vector<int>, sizeof...(Is)>& in_layouts,
                       const std::array<std::vector<int>, sizeof...(Os)>& out_layouts,
                       std::index_sequence<Is...>,
                       std::index_sequence<Os...>)
{
   NVTX_MARK_FUNCTION;
   constexpr std::size_t ninputs = sizeof...(Is);

   using qf_signature = typename get_function_signature<qfunc_t>::type;
   using qf_param_ts = typename qf_signature::parameter_ts;

   NVTX_MARK_INI("inputs");
   auto inputs = std::make_tuple(
                    make_tensor_array<std::remove_cv_t<std::remove_reference_t<
                    std::tuple_element_t<Is, qf_param_ts>>>>(
                       xq.GetBlock(Is).Read(), &in_layouts[Is], gnqp)...);
   NVTX_MARK_END("inputs");

   NVTX_MARK_INI("outputs");
   auto outputs = std::make_tuple(
                     make_tensor_array<std::remove_cv_t<std::remove_reference_t<
                     std::tuple_element_t<ninputs + Os, qf_param_ts>>>>(
                        yq.GetBlock(Os).ReadWrite(), &out_layouts[Os], gnqp)...);
   NVTX_MARK_END("outputs");

   NVTX_MARK_INI("apply");
   std::apply([&](auto&&... args)
   {
      qfunc(args...);
   }, std::tuple_cat(inputs, outputs));
   NVTX_MARK_END("apply");
}

///////////////////////////////////////////////////////////////////////////////
template <typename T>
struct is_tensor_array : std::false_type {};

template <typename scalar_t, int... Dims>
struct is_tensor_array<tensor_array<scalar_t, Dims...>> : std::true_type {};

template <typename T>
struct is_tensor_array_mut : std::false_type {};

template <typename scalar_t, int... Dims>
struct is_tensor_array_mut<tensor_array<scalar_t, Dims...>>:
/*  */ std::bool_constant<!std::is_const_v<scalar_t>> {};

///////////////////////////////////////////////////////////////////////////////
template <typename qfunc_t, typename inputs_t, typename outputs_t>
struct supports_tensor_array_qfunc
{
   using qf_signature = typename get_function_signature<qfunc_t>::type;
   using qf_param_ts = typename qf_signature::parameter_ts;

   static constexpr int ninputs = std::tuple_size_v<inputs_t>;
   static constexpr int noutputs = std::tuple_size_v<outputs_t>;
   static constexpr int nparams = std::tuple_size_v<qf_param_ts>;

   template <std::size_t... Is>
   static constexpr bool InputsOk(std::index_sequence<Is...>)
   {
      return (is_tensor_array<std::remove_cv_t<std::remove_reference_t<
              std::tuple_element_t<Is, qf_param_ts>>>>::value && ...);
   }

   template <std::size_t... Is>
   static constexpr bool OutputsOk(std::index_sequence<Is...>)
   {
      return (is_tensor_array_mut<std::remove_cv_t<std::remove_reference_t<
              std::tuple_element_t<ninputs + Is, qf_param_ts>>>>::value && ...);
   }

   static constexpr bool value =
      (nparams == ninputs + noutputs) &&
      InputsOk(std::make_index_sequence<ninputs> {}) &&
      OutputsOk(std::make_index_sequence<noutputs> {});
};

///////////////////////////////////////////////////////////////////////////////
template <size_t ninputs>
inline void interpolate(const std::array<size_t, ninputs> &input_to_infd,
                        const std::array<FieldBasis, ninputs> &input_bases,
                        const std::vector<Vector *> &xe,
                        BlockVector &xq,
                        const std::array<bool, ninputs> &conditional = all_true<ninputs>())
{
   NVTX_MARK_FUNCTION;
   constexpr_for<0, ninputs>([&](auto i)
   {
      if (!conditional.empty() && !conditional[i]) { return; }
      NVTX_MARK("input forward block #{}", i.value);
      input_bases[i].forward(*xe[input_to_infd[i]], xq.GetBlock(i));
   });
}

///////////////////////////////////////////////////////////////////////////////
inline FieldBasis FieldBasisFromWeight(const IntegrationRule &ir)
{
   NVTX_MARK_FUNCTION;
   return
   {
      [&ir](const Vector &, Vector &xq)
      {
         NVTX("FieldBasisFromWeight");
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
      [](const Vector &, Vector &) { }
   };
}

inline FieldBasis FromQI(const QuadratureInterpolator *qi,
                         QuadratureInterpolator::EvalFlags mode)
{
   NVTX_MARK_FUNCTION;
   return
   {
      [qi, mode](const Vector &xe, Vector &xq)
      {
         qi->SetOutputLayout(QVectorLayout::byVDIM);
         if (mode == QuadratureInterpolator::VALUES)
         {
            NVTX("VALUES");
            qi->Values(xe, xq);
         }
         else
         {
            NVTX("DERIVATIVES");
            qi->Derivatives(xe, xq);
         }
      },
      [qi, mode](const Vector &yq, Vector &ye)
      {
         Vector empty;
         qi->SetOutputLayout(QVectorLayout::byVDIM);
         if (mode == QuadratureInterpolator::VALUES)
         {
            NVTX("Transposed VALUES");
            qi->AddMultTranspose(QuadratureInterpolator::VALUES, yq, empty, ye);
         }
         else
         {
            NVTX("Transposed DERIVATIVES");
            qi->AddMultTranspose(QuadratureInterpolator::DERIVATIVES, empty, yq, ye);
         }
      }
   };
}

// QuadratureFunction identity copy
inline FieldBasis FromQF()
{
   NVTX_MARK_FUNCTION;
   return
   {
      [](const Vector &xe, Vector &xq) { NVTX("FromQF(e->q)"); xq = xe; },
      [](const Vector &yq, Vector &ye) { NVTX("FromQF(q->e)"); ye = yq; }
   };
}

// User-defined parameter space B
inline FieldBasis FromPS(const Operator *B, const Operator *Bt)
{
   NVTX_MARK_FUNCTION;
   return
   {
      [B](const Vector &xe, Vector &xq) { NVTX("B->Mult(e->q)");  B->Mult(xe, xq); },
      [Bt](const Vector &yq, Vector &ye) { NVTX("Bt->Mult(q->e)"); Bt->Mult(yq, ye); }
   };
}

///////////////////////////////////////////////////////////////////////////////
inline const FieldBasis GetFieldBasis(const FieldDescriptor &f,
                                      const IntegrationRule &ir,
                                      QuadratureInterpolator::EvalFlags mode)
{
   NVTX_MARK_FUNCTION;
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
      else if constexpr (std::is_same_v<T, const QuadratureFunction *>)
      {
         return FromQF();
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
         static_assert(false, "internal error");
      }
   }, f.data);
}

///////////////////////////////////////////////////////////////////////////////
template <typename fops_t, size_t nfops>
inline void create_fieldbases(fops_t &fops,
                              const std::array<size_t, nfops> &fop_to_fd,
                              const std::vector<FieldDescriptor> &fds,
                              const IntegrationRule &ir,
                              std::array<FieldBasis, nfops> &bases)
{
   NVTX_MARK_FUNCTION;
   constexpr_for<0, nfops>([&](auto i)
   {
      const auto fop = get<i>(fops);
      using fop_t = std::decay_t<decltype(fop)>;

      const auto fd = fds[fop_to_fd[i]];

      constexpr QuadratureInterpolator::EvalFlags dummy_mode =
         QuadratureInterpolator::VALUES;
      if constexpr (is_identity_fop<fop_t>::value)
      {
         bases[i] = detail::GetFieldBasis(fd, ir, dummy_mode);
      }
      else if constexpr (is_weight_fop<fop_t>::value)
      {
         bases[i] = detail::FieldBasisFromWeight(ir);
      }
      else if constexpr (is_value_fop<fop_t>::value)
      {
         bases[i] = detail::GetFieldBasis(fd, ir, QuadratureInterpolator::VALUES);
      }
      else if constexpr (is_gradient_fop<fop_t>::value)
      {
         bases[i] = detail::GetFieldBasis(fd, ir, QuadratureInterpolator::DERIVATIVES);
      }
   });
}

///////////////////////////////////////////////////////////////////////////////
template <typename fops_t, size_t nfops>
inline void check_consistency(fops_t &fops,
                              const std::array<size_t, nfops> &fop_to_fd,
                              const std::vector<FieldDescriptor> &fields)
{
   NVTX_MARK_FUNCTION;
   constexpr_for<0, nfops>([&](auto i)
   {
      const auto input = get<i>(fops);
      using input_t = std::decay_t<decltype(input)>;

      const auto fd = fields[fop_to_fd[i]];

      if constexpr (is_identity_fop<input_t>::value)
      {
         MFEM_ASSERT(std::holds_alternative<const QuadratureFunction *>(fd.data),
                     "Identity FieldOperator requested on non "
                     "QuadratureFunction");
      }
      else if constexpr (is_weight_fop<input_t>::value)
      {
      }
      else if constexpr (is_value_fop<input_t>::value)
      {
         MFEM_ASSERT(std::holds_alternative<const FiniteElementSpace *>(fd.data) ||
                     std::holds_alternative<const ParFiniteElementSpace *>(fd.data) ||
                     std::holds_alternative<const ParameterSpace *>(fd.data),
                     "Value FieldOperator requested on non "
                     "QuadratureFunction");
      }
      else if constexpr (is_gradient_fop<input_t>::value)
      {
         MFEM_ASSERT(std::holds_alternative<const FiniteElementSpace *>(fd.data) ||
                     std::holds_alternative<const ParFiniteElementSpace *>(fd.data),
                     "Value FieldOperator requested on non "
                     "QuadratureFunction");
      }
   });
}

///////////////////////////////////////////////////////////////////////////////
// Create quadrature function fop to fields map
template <typename fops_t, size_t N = std::tuple_size_v<fops_t>, size_t M>
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

template<typename qfunc_t,
         typename inputs_t,
         typename outputs_t,
         size_t ninputs = std::tuple_size_v<inputs_t>,
         size_t noutputs = std::tuple_size_v<outputs_t>>
struct Action
{
   Action(IntegratorContext ctx,
          qfunc_t qfunc,
          inputs_t inputs,
          outputs_t outputs):
      ctx(ctx), qfunc(std::move(qfunc)), inputs(inputs), outputs(outputs)
   {
      NVTX_MARK_FUNCTION;
      // dbg("ninputs: {}, noutputs: {}", ninputs, noutputs);
      detail::create_fop_to_fd(inputs, ctx.infds, input_to_infd);
      detail::create_fop_to_fd(outputs, ctx.outfds, output_to_outfd);

      detail::check_consistency(inputs, input_to_infd, ctx.infds);
      detail::check_consistency(outputs, output_to_outfd, ctx.outfds);

      detail::create_fieldbases(inputs, input_to_infd, ctx.infds, ctx.ir,
                                input_bases);
      detail::create_fieldbases(outputs, output_to_outfd, ctx.outfds, ctx.ir,
                                output_bases);

      // Prepare inputs q-layouts maps for the qfunc call
      constexpr_for<0, ninputs>([&](auto i)
      {
         using fop_t =
            std::remove_cv_t<std::remove_reference_t<decltype(get<i>(inputs))>>;
         const auto it = ctx.in_qlayouts.find(std::type_index(typeid(fop_t)));
         if (it != ctx.in_qlayouts.end()) { input_qlayouts[i] = it->second; }
         else
         {
            input_qlayouts[i].clear();
         }
      });

      // Prepare outputs q-layouts maps for the qfunc call
      constexpr_for<0, noutputs>([&](auto i)
      {
         using fop_t =
            std::remove_cv_t<std::remove_reference_t<decltype(get<i>(outputs))>>;
         const auto it = ctx.out_qlayouts.find(std::type_index(typeid(fop_t)));
         if (it != ctx.out_qlayouts.end())
         {
            output_qlayouts[i] = it->second;
         }
         else
         {
            output_qlayouts[i].clear();
         }
      });

      const int nqp = ctx.ir.GetNPoints();
      gnqp = nqp * ctx.nentities;

      // prepare xq and yq BlockVectors
      xq_offsets.SetSize(ninputs + 1);
      xq_offsets[0] = 0;
      constexpr_for<0, ninputs>([&](auto i)
      {
         const auto input = get<i>(inputs);
         xq_offsets[i + 1] = nqp * input.size_on_qp * ctx.nentities;
      });
      xq_offsets.PartialSum();
      xq.Update(xq_offsets);

      yq_offsets.SetSize(noutputs + 1);
      yq_offsets[0] = 0;
      constexpr_for<0, noutputs>([&](auto i)
      {
         const auto output = get<i>(outputs);
         yq_offsets[i + 1] = nqp * output.size_on_qp * ctx.nentities;
      });
      yq_offsets.PartialSum();
      yq.Update(yq_offsets);
   }

   //////////////////////////////////////////////////////////////////
   void operator()(const std::vector<Vector *> &xe,
                   std::vector<Vector *> &ye) const
   {
      NVTX_MARK_FUNCTION;
      if (ctx.attr.Size() == 0) { return; }
      // E -> Q
      detail::interpolate(input_to_infd, input_bases, xe, xq);
      // Q -> Q
      static_assert(
         supports_tensor_array_qfunc<qfunc_t, inputs_t, outputs_t>::value,
         "qfunc signature not supported by default backend Action");
      detail::call_qfunc(qfunc,
                         xq,
                         yq,
                         gnqp,
                         input_qlayouts,
                         output_qlayouts,
                         std::make_index_sequence<ninputs> {},
                         std::make_index_sequence<noutputs> {});
      // Q -> E
      detail::integrate(output_to_outfd, output_bases, yq, ye);
   }

   IntegratorContext ctx;
   qfunc_t qfunc;
   inputs_t inputs;
   outputs_t outputs;

   std::array<size_t, ninputs> input_to_infd;
   std::array<size_t, noutputs> output_to_outfd;

   std::array<FieldBasis, ninputs> input_bases;
   std::array<FieldBasis, noutputs> output_bases;

   std::array<std::vector<int>, ninputs> input_qlayouts;
   std::array<std::vector<int>, noutputs> output_qlayouts;

   int gnqp = 0;
   Array<int> xq_offsets, yq_offsets;
   mutable BlockVector xq, yq;
};

} // namespace mfem::future::device::detail

namespace mfem::future
{

struct DeviceBackend
{
   template<typename qfunc_t, typename inputs_t, typename outputs_t>
   auto static MakeAction(const IntegratorContext &ctx,
                          qfunc_t qfunc,
                          inputs_t inputs,
                          outputs_t outputs)
   {
      NVTX_MARK_FUNCTION;
      return device::detail::Action(ctx, qfunc, inputs, outputs);
   }
};

} // namespace mfem::future
