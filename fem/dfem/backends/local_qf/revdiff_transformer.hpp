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

#include "../util.hpp"
#include "util.hpp"
#include "../../util.hpp"

namespace mfem::future
{


////////////////////////////////////////////////////
///---- Function traits for q-function types ----///
////////////////////////////////////////////////////

template <typename T> struct function_traits;

template <typename R, typename C, typename... Args>
struct function_traits<R (C::*)(Args...) const>
{
   using primal_return_type = R;
   using args_tuple = tuple<Args...>;
   static constexpr size_t arity = sizeof...(Args);
};

template <typename T>
struct qp_scalar_traits
{
   using view_type = T;
   using dual_type = dual<T, T>;
};

template <typename V, typename G>
struct qp_scalar_traits<dual<V, G>>
{
   using view_type = V;
   using dual_type = dual<V, G>;
};

using native_dual_t = dual<real_t, real_t>;
using nested_native_dual_t = dual<native_dual_t, native_dual_t>;


///////////////////////////////////////////////////////////////
///---- Utils for Nested dual numbers (2nd derivatives) ----///
///////////////////////////////////////////////////////////////

//<--- Types
/// TODO: Should we move the 'hyper' (nested) duals to the dual.hpp?
template <typename T>
struct make_nested_qp_type
{
   using type = T;
};

template <typename V, typename G>
struct make_nested_qp_type<dual<V, G>>
{
   using type = dual<dual<V, G>, dual<V, G>>;
};

template <typename S, int... Sizes>
struct make_nested_qp_type<tensor<S, Sizes...>>
{
   using type = tensor<typename make_nested_qp_type<S>::type, Sizes...>;
};

template <typename T>
using make_nested_qp_type_t = typename make_nested_qp_type<T>::type;


//<--- Lift of dual to nested dual
// Maps dual(a,b) --> dual(dual(a,0), dual(b,0)) for 2nd derivative computation

template <typename Dst, typename Src>
MFEM_HOST_DEVICE void lift_to_nested_arg(const Src &src, Dst &dst)
{
   using dst_t = std::decay_t<Dst>;
   constexpr bool dst_uses_dual = is_dual_number<dst_t>::value ||
                                  qf_param_uses_dual_v<dst_t> ||
                                  is_nested_dual_number<dst_t>::value ||
                                  qf_param_uses_nested_dual_v<dst_t>;
   if constexpr (dst_uses_dual)
   {
      constexpr int ncomp = qf_param_shape<dst_t>::rank == 0 ?
                            1 : []() constexpr
      {
         int n = 1;
         for (int extent : qf_param_shape<dst_t>::extents) { n *= extent; }
         return n;
      }();
      for (int component = 0; component < ncomp; component++)
      {
         qf_set_flat_value(dst, component,
                           qf_flat_value(src, component));
         qf_set_flat_gradient(dst, component,
                              qf_flat_gradient(src, component));
      }
   }
   else
   {
      // If the destination type does not use dual numbers, just copy the source to destination.
      dst = src;
   }
}

template <typename Tuple, size_t... Is>
auto nested_tuple_types_impl(std::index_sequence<Is...>)
   -> tuple<make_nested_qp_type_t<std::decay_t<tuple_element_t<Is, Tuple>>>...>;

template <typename Tuple>
using nested_tuple_t = decltype(nested_tuple_types_impl<Tuple>(
                                  std::make_index_sequence<tuple_size<Tuple>::value> {}));

template <typename SrcTuple, typename DstTuple, size_t... Is>
MFEM_HOST_DEVICE void lift_tuple_to_nested_impl(const SrcTuple &src, DstTuple &dst,
                                                std::index_sequence<Is...>)
{
   (lift_to_nested_arg(get<Is>(src), get<Is>(dst)), ...);
}

template <typename SrcTuple, typename DstTuple>
MFEM_HOST_DEVICE void lift_tuple_to_nested(const SrcTuple &src, DstTuple &dst)
{
   lift_tuple_to_nested_impl(src, dst,
                             std::make_index_sequence<tuple_size<SrcTuple>::value> {});
}


//<--- Qfunc rebinding to nested duals arguments

template <typename qfunc_t, typename nested_scalar_t, typename = void>
struct rebind_qfunc_scalar
{
   static constexpr bool supported = false;
};

template <template <typename> class qfunc_template_t,
          typename old_scalar_t,
          typename nested_scalar_t>
struct rebind_qfunc_scalar<qfunc_template_t<old_scalar_t>, nested_scalar_t>
{
   static constexpr bool supported = true;
   using type = qfunc_template_t<nested_scalar_t>;
};

template <template <typename, auto...> class qfunc_template_t,
          typename old_scalar_t,
          auto... Params,
          typename nested_scalar_t>
struct rebind_qfunc_scalar<qfunc_template_t<old_scalar_t, Params...>,
                           nested_scalar_t,
                           std::enable_if_t<(sizeof...(Params) > 0)>>
{
   static constexpr bool supported = true;
   using type = qfunc_template_t<nested_scalar_t, Params...>;
};

template <typename qfunc_t, typename nested_scalar_t>
using rebind_qfunc_scalar_t =
   typename rebind_qfunc_scalar<qfunc_t, nested_scalar_t>::type;

// Component count and writable counterpart of a per-point argument, which
// is either a tensor or a plain scalar.
template <typename Arg> struct qp_traits
{
   static_assert(std::is_arithmetic_v<Arg> || is_dual_number<Arg>::value,
                 "per-point arguments must be tensors or scalars");
   using scalar_type = std::remove_const_t<Arg>;
   using view_type = typename qp_scalar_traits<scalar_type>::view_type;
   using dual_type = typename qp_scalar_traits<scalar_type>::dual_type;
   static constexpr int components = 1;
};

template <typename T, int... Sizes> struct qp_traits<tensor<T, Sizes...>>
{
   using scalar_type = std::remove_const_t<T>;
   using scalar_traits = qp_traits<scalar_type>;
   using view_type = tensor<typename scalar_traits::view_type, Sizes...>;
   using dual_type = tensor<typename scalar_traits::dual_type, Sizes...>;
   static constexpr int components = (Sizes * ... * 1);
};

template <typename... T1s, typename... T2s>
constexpr tuple<T1s..., T2s...> concat_tuples(tuple<T1s...>, tuple<T2s...>);


// RevDiff: computes the full gradient of a pointwise qfunction at a single
// quadrature point using one Enzyme reverse-mode (autodiff) call.
//
// operator()(args...) takes all qfunction input primals followed by one
// writable gradient output per Active input (same shape as that input).
// The qfunction's own output is not passed; Enzyme writes it to stack
// scratch (enzyme_dupnoneed).
//
// A single __enzyme_autodiff call with the output adjoint seeded to 1
// yields all gradient blocks simultaneously — O(1) calls per point
// regardless of input size, vs O(ncomp) for forward mode.
//
// operator() is MFEM_HOST_DEVICE, stateless and allocation-free.

//<--- Determines whether the RevDiff transformer is used in a gradient action (Eval)
// or to compute second derivatives (Derivative).
// This only affects dual number mode, and allows to compute 2nd derivatives with hyper dual numbers. 
enum class RevDiffDualMode
{
   Eval,
   Derivative
};

template <typename Func, typename InputActivityTuple,
          typename OutputActivityTuple,
          RevDiffDualMode mode = RevDiffDualMode::Eval>
struct RevDiff
{
   using traits = function_traits<decltype(&Func::operator())>;
   using args_tuple = typename traits::args_tuple;
   using activity =
      decltype(concat_tuples(InputActivityTuple{}, OutputActivityTuple{}));
   static constexpr size_t arity = traits::arity;
   static constexpr size_t num_inputs = tuple_size<InputActivityTuple>::value;

   static_assert(std::is_void_v<typename traits::primal_return_type>,
                 "RevDiff only supports primal functions with void return type");
   static_assert(tuple_size<activity>::value == arity,
                 "Number of input and output activity tags must match function "
                 "arity");

   template <size_t I>
   static constexpr bool is_active = qf_param_is_active_v<activity, I>;

   // Number of Active inputs and their argument indices, in ascending order.
   // A qfunction may have several Active inputs at once: e.g. a field's value
   // u and its gradient dudx both feed the output and both must be
   // differentiated (the chain-rule contraction with the value/gradient shape
   // functions then happens at the FE-operator level). We produce one gradient
   // block, d(output)/d(input), per Active input — each computed with the other
   // Active inputs frozen, so they come out as isolated partials.
   template <size_t... Is>
   static constexpr size_t count_active_inputs(std::index_sequence<Is...>)
   {
      return ((Is < num_inputs && is_active<Is> ? size_t{1} : size_t{0}) + ...);
   }
   static constexpr size_t num_active_inputs =
      count_active_inputs(std::make_index_sequence<arity> {});

   template <size_t... Is>
   static constexpr std::array<size_t, num_active_inputs>
   collect_active_inputs(std::index_sequence<Is...>)
   {
      std::array<size_t, num_active_inputs> idx{};
      size_t j = 0;
      (((Is < num_inputs && is_active<Is>) ? (idx[j++] = Is) : size_t{0}), ...);
      return idx;
   }
   static constexpr auto active_inputs =
   collect_active_inputs(std::make_index_sequence<arity> {});

   // Slot index of argument I in the active_inputs array (compile-time).
   template <size_t I>
   static constexpr size_t slot_of = []() constexpr -> size_t
   {
      for (size_t s = 0; s < num_active_inputs; s++)
         if (active_inputs[s] == I) { return s; }
      return num_active_inputs;
   }();

   static constexpr size_t active_output =
      find_single_active_qparam<activity, num_inputs, arity>();
   static_assert(active_output < arity,
                 "gradient mode requires exactly one Active output");
   static_assert(num_active_inputs >= 1,
                 "gradient mode requires at least one Active input");
   static_assert(tuple_size<OutputActivityTuple>::value == 1,
                 "gradient mode requires exactly one (scalar) output");

   using output_type =
      std::decay_t<tuple_element_t<active_output, args_tuple>>;

   using output_view = typename qp_traits<output_type>::view_type;

   static_assert(qp_traits<output_type>::components == 1,
                 "gradient output requires a scalar output");

   template <size_t I>
   using primal_arg_t = tuple_element_t<I, args_tuple>;

   template <size_t I>
   using derivative_arg_t =
      std::conditional_t<mode == RevDiffDualMode::Derivative && is_active<I>,
                         typename qp_traits<std::decay_t<tuple_element_t<I, args_tuple>>>::dual_type,
                         primal_arg_t<I>>;

   template <size_t S>
   using active_arg_decay_t =
      std::decay_t<tuple_element_t<active_inputs[S], args_tuple>>;

   template <size_t S>
   static constexpr bool active_arg_uses_dual =
      is_dual_number<active_arg_decay_t<S>>::value ||
      qf_param_uses_dual_v<active_arg_decay_t<S>>;

   template <size_t S>
   using grad_arg_t =
      std::conditional_t<mode == RevDiffDualMode::Derivative || active_arg_uses_dual<S>,
                         typename qp_traits<active_arg_decay_t<S>>::dual_type,
                         typename qp_traits<active_arg_decay_t<S>>::view_type>
      &;

   template <typename qfunc_type>
   using qfunc_args_tuple_t = decay_tuple<typename function_traits<decltype(&qfunc_type::operator())>::args_tuple>;

   template <size_t... Is, size_t... Ss>
   static FunctionSignature<void(derivative_arg_t<Is>..., grad_arg_t<Ss>...)>
   signature_impl(std::index_sequence<Is...>, std::index_sequence<Ss...>);

   using signature = decltype(signature_impl(std::make_index_sequence<num_inputs> {},
                                             std::make_index_sequence<num_active_inputs> {}));

   // Plain function with the qfunction's exact (reference) signature so it can
   // be handed to Enzyme as a function pointer.
   template <size_t... Is>
   MFEM_HOST_DEVICE static __attribute__((always_inline)) void
   static_call(tuple_element_t<Is, args_tuple>... args)
   {
      Func{}(args...);
   }

   template <size_t... Is>
   static constexpr auto fn_ptr(std::index_sequence<Is...>)
   {
      return &static_call<Is...>;
   }
   static constexpr auto fn = fn_ptr(std::make_index_sequence<arity> {});

   // Load primal inputs from the pointer tuple into a local qargs copy.
   // All dual gradient parts are implicitly zero because qargs is value-initialized.
   template <typename QArgs, typename AllPtrs, size_t... Is>
   MFEM_HOST_DEVICE static __attribute__((always_inline)) void load_qargs(
      QArgs &qargs, AllPtrs &ptrs, std::index_sequence<Is...>)
   {
      ((mfem::future::get<int(Is)>(qargs) =
           *mfem::future::get<int(Is)>(ptrs)), ...);
   }

   template <typename QArgs, typename AllPtrs, size_t... Is>
   MFEM_HOST_DEVICE static __attribute__((always_inline)) void lift_qargs_to_nested_dual(
      QArgs &qargs, AllPtrs &ptrs, std::index_sequence<Is...>)
   {
      (lift_to_nested_arg(*mfem::future::get<int(Is)>(ptrs), get<Is>(qargs)), ...);
   }

   template <size_t S, typename AllPtrs>
   MFEM_HOST_DEVICE static __attribute__((always_inline)) void seed_active_input(
       AllPtrs &ptrs)
   {
      constexpr size_t input_idx = active_inputs[S];
      using active_arg_t = std::decay_t<tuple_element_t<input_idx, args_tuple>>;
      constexpr int ncomp = qp_traits<active_arg_t>::components;

      for (int component = 0; component < ncomp; component++)
      {
         // Use RevDiff for gradient action computation 
         if constexpr (mode == RevDiffDualMode::Eval)
         {
            // Fresh default-constructed qargs: primal values are loaded below,
            // all dual gradient parts start at zero (no explicit clear needed).
            qfunc_args_tuple_t<Func> qargs{};
            load_qargs(qargs, ptrs, std::make_index_sequence<num_inputs>{});

            auto &grad = *get<num_inputs + S>(ptrs);

            qf_set_flat_gradient(get<input_idx>(qargs), component, 1.0);

            call_qfunc_no_move(Func{}, qargs);

            auto &out = get<num_inputs>(qargs);

            qf_set_flat_value(grad, component, qf_flat_gradient(out, 0));
         }
         else // Use RevDiffDualMode::RevDiff for second derivative computation, we need nested dual numbers to avoid overwriting the first dual pair.
         {
            // Lift dFEM's dual input (a,b) to ((a,c),(b,d)). Here b is
            // the incoming Hessian-action direction, c is RevDiff's local
            // component seed. After evaluating E, the nested scalar output is
            // ((E, dE/dx_i), (E'[b], H_i[b])); after computing the qfunc,
            // we unpack the result for the return to dfem as (dE/dx_i, H_i[b]).

            static_assert(rebind_qfunc_scalar<Func, nested_native_dual_t>::supported,
                          "RevDiff native-dual derivative mode requires q-function "
                          "types of the form QFunc<scalar_t> so they can be "
                          "rebound to nested dual scalars");
            using nested_func_t = rebind_qfunc_scalar_t<Func, nested_native_dual_t>;
            qfunc_args_tuple_t<nested_func_t> nested_qargs{};

            lift_qargs_to_nested_dual(nested_qargs, ptrs, std::make_index_sequence<num_inputs>{});

            qf_set_flat_value_gradient(get<input_idx>(nested_qargs), component, 1.0);

            call_qfunc_no_move(nested_func_t{}, nested_qargs);

            // Unpack the nested dual output into the gradient output for the 2nd derivative.
            auto &out = get<active_output>(nested_qargs);
            auto &grad = *get<num_inputs + S>(ptrs);

            qf_set_flat_value(grad, component,
                              qf_flat_value_gradient(out, 0));
            qf_set_flat_gradient(grad, component,
                                 qf_flat_gradient_gradient(out, 0));
         }
      }
   }

   // Compute gradient of the qfunction with respect to all Active inputs at once, using  forward-mode seeding.
   // This is the dual-number fallback when Enzyme is not available.  
   template <typename AllPtrs>
   MFEM_HOST_DEVICE static __attribute__((always_inline)) void call_dual_rev(
      AllPtrs &ptrs)
   {
      for_constexpr<num_active_inputs>([&](auto s)
      {
         // Seed the s-th Active input w/ dual-number tangent 1.0, then call the qfunction to compute the corresponding gradient block.
         seed_active_input<decltype(s)::value>(ptrs);
      });
   }

#ifdef MFEM_USE_ENZYME
   // Recursively flatten the enzyme argument list for a single reverse-mode
   // call that differentiates ALL active inputs at once:
   //   active input:  enzyme_dup, &primal, &grad      — grad accumulates
   //   const input:   enzyme_const, &primal
   //   active output: enzyme_dupnoneed, &scratch, &adjoint_seed
   // ptrs holds all primal pointers in [0, num_inputs) and all gradient output
   // pointers in [num_inputs, num_inputs+num_active_inputs).
   template <size_t I = 0, typename AllPtrs, typename... Built>
   MFEM_HOST_DEVICE static __attribute__((always_inline)) void
   call_enzyme_rev(AllPtrs &ptrs, output_view &out_scratch,
                   output_view &out_adjoint, Built... built)
   {
      if constexpr (I == arity)
      {
         __enzyme_autodiff<void>(fn, built...);
      }
      else if constexpr (I == active_output)
      {
         call_enzyme_rev<I + 1>(ptrs, out_scratch, out_adjoint, built...,
                                enzyme_dupnoneed, &out_scratch, &out_adjoint);
      }
      else if constexpr (is_active<I>)
      {
         call_enzyme_rev<I + 1>(
            ptrs, out_scratch, out_adjoint, built..., enzyme_dup,
            mfem::future::get<int(I)>(ptrs),
            mfem::future::get<int(num_inputs + slot_of<I>)>(ptrs));
      }
      else
      {
         call_enzyme_rev<I + 1>(ptrs, out_scratch, out_adjoint, built...,
                                enzyme_const, mfem::future::get<int(I)>(ptrs));
      }
   }
#endif

   // Zero all gradient outputs before the enzyme call (Enzyme accumulates).
   template <typename AllPtrs, size_t... Ss>
   MFEM_HOST_DEVICE static __attribute__((always_inline)) void zero_grads(
      AllPtrs &ptrs,
      std::index_sequence<Ss...>)
   {
      ((*mfem::future::get<int(num_inputs + Ss)>(ptrs) =
           std::decay_t<decltype(*mfem::future::get<int(num_inputs + Ss)>(ptrs))> {}),
       ...);
   }

   // Called once per quadrature point. Arguments are, in order:
   //   * the primal value of every qfunction input (active and const), then
   //   * one gradient output per Active input (ascending index order), each
   //     shaped like its Active input.
   // The qfunction's own output slot is not passed; Enzyme writes it to stack
   // scratch (enzyme_dupnoneed). A single __enzyme_autodiff call yields all
   // gradient blocks simultaneously.
   template <typename... Args>
   MFEM_HOST_DEVICE __attribute__((always_inline)) void operator()(
      Args &&...args) const
   {
      static_assert(sizeof...(Args) == num_inputs + num_active_inputs,
                    "expected one primal per input plus one gradient output per "
                    "Active input");
      auto ptrs = mfem::future::make_tuple(&args...);
      zero_grads(ptrs, std::make_index_sequence<num_active_inputs>{});
      output_view out_scratch{};
      output_view out_adjoint{1.0}; // seed: d(output)/d(output) = 1
#ifdef MFEM_USE_ENZYME
      call_enzyme_rev(ptrs, out_scratch, out_adjoint);
#else
      call_dual_rev(ptrs);
#endif
   }

   static __attribute__((always_inline)) void print() { print_impl(std::make_index_sequence<arity> {}); }

   template <size_t... Is> static __attribute__((always_inline)) void print_impl(
      std::index_sequence<Is...>)
   {
      mfem::out << "__enzyme_autodiff<void>(fptr";
      (([&]
      {
         auto name = get_type_name<tuple_element_t<Is, args_tuple>>();
         if constexpr (Is == active_output)
            mfem::out << ", enzyme_dupnoneed, " << name << ", adjoint=1";
         else if constexpr (is_active<Is>)
            mfem::out << ", enzyme_dup, " << name << ", grad out";
         else
         {
            mfem::out << ", enzyme_const, " << name;
         }
      }()),
      ...);
      mfem::out << ")\n";
   }
};

template <typename Func, typename InputActivityTuple, typename OutputActivityTuple,
          RevDiffDualMode mode>
struct create_function_signature<RevDiff<Func, InputActivityTuple,
                                         OutputActivityTuple, mode>>
{
   using type = typename
                RevDiff<Func, InputActivityTuple, OutputActivityTuple, mode>::signature;
};

} // namespace mfem::future
