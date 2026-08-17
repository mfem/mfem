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

template <typename T> struct function_traits;

template <typename R, typename C, typename... Args>
struct function_traits<R (C::*)(Args...) const>
{
   using primal_return_type = R;
   using args_tuple = tuple<Args...>;
   static constexpr size_t arity = sizeof...(Args);
};

/// Scalar-level view of a per-point argument: the plain value type it stores
/// and the dual type used to carry a first-order tangent alongside it.
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

// Component count and writable counterpart of a per-point argument, which
// is either a tensor or a plain scalar. `view_type` keeps the argument's own
// scalar type; `dual_type` is the same shape with a dual scalar, used for the
// gradient blocks of the native dual-number backend.
template <typename Arg> struct qp_traits
{
   static_assert(std::is_arithmetic_v<Arg> || is_dual_number<Arg>::value,
                 "per-point arguments must be tensors or scalars");
   using scalar_type = std::remove_const_t<Arg>;
   using view_type = scalar_type;
   using dual_type = typename qp_scalar_traits<scalar_type>::dual_type;
   static constexpr int components = 1;
};

template <typename T, int... Sizes> struct qp_traits<tensor<T, Sizes...>>
{
   using scalar_type = std::remove_const_t<T>;
   using view_type = tensor<scalar_type, Sizes...>;
   using dual_type =
      tensor<typename qp_scalar_traits<scalar_type>::dual_type, Sizes...>;
   static constexpr int components = (Sizes * ... * 1);
};

template <typename... T1s, typename... T2s>
constexpr tuple<T1s..., T2s...> concat_tuples(tuple<T1s...>, tuple<T2s...>);

///////////////////////////////////////////////////////////////////////////////
/// Nested ("hyper") dual utilities, used for second derivatives on the native
/// dual-number backend.
///
/// A second derivative taken with plain duals would have to reuse the single
/// gradient slot that already carries the incoming direction. Lifting the
/// scalar to `dual<dual<V,G>, dual<V,G>>` adds a second, independent slot:
///
///   dual(a, b) -> ((a, c), (b, d))
///
/// `a`/`b` stay the incoming primal/direction, `c` is seeded per component and
/// `d` returns the second-order result.
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

using native_dual_t = typename qp_scalar_traits<real_t>::dual_type;
using nested_native_dual_t = make_nested_qp_type_t<native_dual_t>;

/// Rebinds a q-function's scalar template parameter so its arguments are
/// nested duals. Only the leading scalar parameter is rebound; any remaining
/// non-type parameters (e.g. `dim`) are carried through unchanged. This
/// requires q-functions of the form `QFunc<scalar_t>` or `QFunc<scalar_t,
/// Params...>`; `supported` reports whether that shape was matched, so callers
/// can fail with a readable static_assert.
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

/// Copies a q-function argument into its nested-dual counterpart, mapping
/// dual(a, b) -> ((a, 0), (b, 0)). The inner gradients stay zero; the caller
/// seeds one of them per component.
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
      constexpr int ncomp = qp_traits<dst_t>::components;
      for (int component = 0; component < ncomp; component++)
      {
         qf_set_flat_value(dst, component, qf_flat_value(src, component));
         qf_set_flat_gradient(dst, component, qf_flat_gradient(src, component));
      }
   }
   else
   {
      // Destination carries no derivative slots: a plain copy is the lift.
      dst = src;
   }
}

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
// operator() is MFEM_HOST_DEVICE and allocation-free.
//
// Without Enzyme the same interface is served by a forward-mode dual-number
// fallback (`call_dual_rev`), which seeds one component at a time. `mode`
// selects what that fallback is being asked for: `Eval` is the plain gradient,
// `Derivative` is a gradient taken inside an outer derivative, which lifts the
// q-function to nested duals so seeding does not clobber the outer direction.
// With Enzyme both modes use the single reverse-mode call and `mode` is inert.
enum class RevDiffDualMode
{
   Eval,
   Derivative
};

// Number of Active inputs and their argument indices, in ascending order.
// A qfunction may have several Active inputs at once: e.g. a field's value
// u and its gradient dudx both feed the output and both must be
// differentiated (the chain-rule contraction with the value/gradient shape
// functions then happens at the FE-operator level). We produce one gradient
// block, d(output)/d(input), per Active input — each computed with the other
// Active inputs frozen, so they come out as isolated partials.
template <typename activity_t, size_t num_inputs, size_t... Is>
constexpr size_t count_active_inputs(std::index_sequence<Is...>)
{
   return ((Is < num_inputs && qf_param_is_active_v<activity_t, Is>
            ? size_t{1} : size_t{0}) + ...);
}

template <typename activity_t, size_t num_inputs, size_t num_active,
          size_t... Is>
constexpr std::array<size_t, num_active>
collect_active_inputs(std::index_sequence<Is...>)
{
   std::array<size_t, num_active> idx{};
   size_t j = 0;
   (((Is < num_inputs && qf_param_is_active_v<activity_t, Is>)
     ? (idx[j++] = Is) : size_t{0}), ...);
   return idx;
}

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

   static constexpr size_t num_active_inputs =
      count_active_inputs<activity, num_inputs>(
         std::make_index_sequence<arity> {});

   static constexpr auto active_inputs =
      collect_active_inputs<activity, num_inputs, num_active_inputs>(
   std::make_index_sequence<arity> {});

   // Slot index of argument I in the active_inputs array (compile-time).
   template <size_t I>
   static constexpr size_t slot_of()
   {
      for (size_t s = 0; s < num_active_inputs; s++)
         if (active_inputs[s] == I) { return s; }
      return num_active_inputs;
   }

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

   // True when reverse mode is served by the dual-number fallback rather than
   // Enzyme. Everything below that widens a type to a dual is gated on this, so
   // an Enzyme build sees exactly the types it saw before nested duals existed.
#ifdef MFEM_USE_ENZYME
   static constexpr bool native_dual_backend = false;
#else
   static constexpr bool native_dual_backend = true;
#endif

   static constexpr bool use_native_dual_derivative =
      native_dual_backend && (mode == RevDiffDualMode::Derivative);

   // Under the native-dual second-derivative path the active primals arrive
   // carrying the outer direction, so they must be dual-typed.
   template <size_t I>
   using derivative_arg_t =
      std::conditional_t<use_native_dual_derivative &&
      qf_param_is_active_v<activity, I>,
      typename qp_traits<std::decay_t<tuple_element_t<I, args_tuple>>>::dual_type,
      primal_arg_t<I>>;

   template <size_t S>
   using active_arg_decay_t =
      std::decay_t<tuple_element_t<active_inputs[S], args_tuple>>;

   template <size_t S>
   static constexpr bool active_arg_uses_dual()
   {
      return native_dual_backend &&
             (is_dual_number<active_arg_decay_t<S>>::value ||
              qf_param_uses_dual_v<active_arg_decay_t<S>>);
   }

   // A gradient block mirrors its active input's shape. It needs a dual scalar
   // whenever the fallback has to return a value and a tangent through it.
   template <size_t S>
   using grad_arg_t =
      std::conditional_t<use_native_dual_derivative || active_arg_uses_dual<S>(),
      typename qp_traits<active_arg_decay_t<S>>::dual_type,
      typename qp_traits<active_arg_decay_t<S>>::view_type>
      &;

   template <typename qfunc_type>
   using qfunc_args_tuple_t =
      decay_tuple<typename function_traits<decltype(&qfunc_type::operator())>::args_tuple>;

   template <size_t... Is, size_t... Ss>
   static FunctionSignature<void(derivative_arg_t<Is>..., grad_arg_t<Ss>...)>
   signature_impl(std::index_sequence<Is...>, std::index_sequence<Ss...>);

   using signature = decltype(signature_impl(std::make_index_sequence<num_inputs> {},
                                             std::make_index_sequence<num_active_inputs> {}));

   Func func {};

   RevDiff() = default;
   MFEM_HOST_DEVICE explicit RevDiff(const Func &func_) : func(func_) { }

   // Plain function with the qfunction's exact (reference) signature, plus the
   // configured qfunction instance, so it can be handed to Enzyme as a function
   // pointer without default-constructing away runtime qfunction state.
   template <size_t... Is>
   MFEM_HOST_DEVICE static MFEM_FUTURE_ALWAYS_INLINE void
   static_call(Func *func, tuple_element_t<Is, args_tuple>... args)
   {
      (*func)(args...);
   }

   template <size_t... Is>
   static constexpr auto fn_ptr(std::index_sequence<Is...>)
   {
      return &static_call<Is...>;
   }
   static constexpr auto fn()
   {
      return fn_ptr(std::make_index_sequence<arity> {});
   }

   // Load primal inputs from the pointer tuple into a local qargs copy. Dual
   // gradient parts are implicitly zero because qargs is value-initialized.
   template <typename QArgs, typename AllPtrs, size_t... Is>
   MFEM_HOST_DEVICE static MFEM_FUTURE_ALWAYS_INLINE void load_qargs(
      QArgs &qargs, AllPtrs &ptrs, std::index_sequence<Is...>)
   {
      ((mfem::future::get<int(Is)>(qargs) =
           *mfem::future::get<int(Is)>(ptrs)), ...);
   }

   template <typename QArgs, typename AllPtrs, size_t... Is>
   MFEM_HOST_DEVICE static MFEM_FUTURE_ALWAYS_INLINE void
   lift_qargs_to_nested_dual(QArgs &qargs, AllPtrs &ptrs,
                             std::index_sequence<Is...>)
   {
      (lift_to_nested_arg(*mfem::future::get<int(Is)>(ptrs),
                          mfem::future::get<Is>(qargs)), ...);
   }

   // The nested-dual q-function is a *different* type — its scalar template
   // parameter is rebound — so a configured instance cannot simply be copied
   // over. Runtime q-function state must still survive, or the second
   // derivative would silently be taken of a differently-parameterised energy.
   //
   // Three cases, in order:
   //   * the rebound type converts from this one: use that conversion;
   //   * no state at all: nothing to carry;
   //   * same size and trivially copyable: none of the members depend on the
   //     rebound scalar, so the two are layout-identical and the state copies
   //     over bytewise. A member that *did* depend on the scalar would change
   //     the size and land in the static_assert below instead.
   template <typename nested_func_t>
   MFEM_HOST_DEVICE MFEM_FUTURE_ALWAYS_INLINE nested_func_t
   make_nested_func() const
   {
      if constexpr (std::is_constructible_v<nested_func_t, const Func &>)
      {
         return nested_func_t(func);
      }
      else if constexpr (std::is_empty_v<Func>)
      {
         return nested_func_t {};
      }
      else
      {
         static_assert(std::is_trivially_copyable_v<Func> &&
                       std::is_trivially_copyable_v<nested_func_t> &&
                       sizeof(Func) == sizeof(nested_func_t),
                       "second derivatives on the native dual backend rebind "
                       "the q-function's scalar type; a q-function whose state "
                       "depends on that scalar must be constructible from its "
                       "rebound form");
         nested_func_t nested {};
         const auto *src = reinterpret_cast<const unsigned char *>(&func);
         auto *dst = reinterpret_cast<unsigned char *>(&nested);
         for (size_t b = 0; b < sizeof(Func); b++) { dst[b] = src[b]; }
         return nested;
      }
   }

   // Seed the s-th Active input one component at a time and read the resulting
   // gradient block back out. This is the forward-mode dual-number stand-in for
   // one reverse-mode call: O(ncomp) evaluations instead of O(1).
   template <size_t S, typename AllPtrs>
   MFEM_HOST_DEVICE MFEM_FUTURE_ALWAYS_INLINE void seed_active_input(
      AllPtrs &ptrs) const
   {
      constexpr size_t input_idx = active_inputs[S];
      using active_arg_t = std::decay_t<tuple_element_t<input_idx, args_tuple>>;
      constexpr int ncomp = qp_traits<active_arg_t>::components;

      for (int component = 0; component < ncomp; component++)
      {
         if constexpr (mode == RevDiffDualMode::Eval)
         {
            // Fresh value-initialized qargs: primals loaded below, all dual
            // gradient parts start at zero, so no explicit clear is needed.
            qfunc_args_tuple_t<Func> qargs {};
            load_qargs(qargs, ptrs, std::make_index_sequence<num_inputs> {});

            auto &grad = *mfem::future::get<num_inputs + S>(ptrs);

            qf_set_flat_gradient(mfem::future::get<input_idx>(qargs), component,
                                 1.0);

            call_qfunc_no_move(func, qargs);

            auto &out = mfem::future::get<active_output>(qargs);

            qf_set_flat_value(grad, component, qf_flat_gradient(out, 0));
         }
         else
         {
            // Lift the incoming dual (a, b) to ((a, c), (b, d)): b is the outer
            // Hessian-action direction, c is this loop's component seed. After
            // evaluating E the nested output holds ((E, dE/dx_i), (E'[b],
            // H_i[b])), and we hand dfem back (dE/dx_i, H_i[b]).
            static_assert(rebind_qfunc_scalar<Func, nested_native_dual_t>::supported,
                          "RevDiff native-dual derivative mode requires "
                          "q-function types of the form QFunc<scalar_t> so they "
                          "can be rebound to nested dual scalars");
            using nested_func_t = rebind_qfunc_scalar_t<Func, nested_native_dual_t>;
            qfunc_args_tuple_t<nested_func_t> nested_qargs {};

            lift_qargs_to_nested_dual(nested_qargs, ptrs,
                                      std::make_index_sequence<num_inputs> {});

            qf_set_flat_value_gradient(
               mfem::future::get<input_idx>(nested_qargs), component, 1.0);

            call_qfunc_no_move(make_nested_func<nested_func_t>(), nested_qargs);

            auto &out = mfem::future::get<active_output>(nested_qargs);
            auto &grad = *mfem::future::get<num_inputs + S>(ptrs);

            qf_set_flat_value(grad, component, qf_flat_value_gradient(out, 0));
            qf_set_flat_gradient(grad, component,
                                 qf_flat_gradient_gradient(out, 0));
         }
      }
   }

   // Dual-number fallback for the whole reverse-mode call: one seeded sweep per
   // Active input.
   template <typename AllPtrs>
   MFEM_HOST_DEVICE MFEM_FUTURE_ALWAYS_INLINE void call_dual_rev(
      AllPtrs &ptrs) const
   {
      for_constexpr<num_active_inputs>([&](auto s)
      {
         seed_active_input<decltype(s)::value>(ptrs);
      });
   }

#ifdef MFEM_USE_ENZYME
   // Recursive builder of the per-argument reverse-mode enzyme call.
   template <size_t I = 0, typename AllPtrs, typename... Built>
   MFEM_HOST_DEVICE MFEM_FUTURE_ALWAYS_INLINE void
   call_enzyme_rev(AllPtrs &ptrs, output_view &scratch, output_view &adjoint,
                   Built... built) const
   {
      if constexpr (I == arity)
      {
         __enzyme_autodiff<void>(fn(), enzyme_const, const_cast<Func *>(&func),
                                 built...);
      }
      else if constexpr (I == active_output)
      {
         // Output: primal written to scratch (unused), adjoint seeded to 1.
         call_enzyme_rev<I + 1>(ptrs, scratch, adjoint, built...,
                                enzyme_dupnoneed, &scratch, &adjoint);
      }
      else if constexpr (qf_param_is_active_v<activity, I>)
      {
         // Active input: gradient accumulates into its grad-output slot.
         call_enzyme_rev<I + 1>(
            ptrs, scratch, adjoint, built..., enzyme_dup,
            mfem::future::get<int(I)>(ptrs),
            mfem::future::get<int(num_inputs + slot_of<I>())>(ptrs));
      }
      else
      {
         // Const input: primal only, no shadow.
         call_enzyme_rev<I + 1>(ptrs, scratch, adjoint, built...,
                                enzyme_const, mfem::future::get<int(I)>(ptrs));
      }
   }
#endif // MFEM_USE_ENZYME

   // Zero all gradient outputs before the enzyme call (Enzyme accumulates).
   template <typename AllPtrs, size_t... Ss>
   MFEM_HOST_DEVICE static
   MFEM_FUTURE_ALWAYS_INLINE void zero_grads(
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
   MFEM_HOST_DEVICE MFEM_FUTURE_ALWAYS_INLINE void operator()(
      Args &&...args) const
   {
      static_assert(sizeof...(Args) == num_inputs + num_active_inputs,
                    "expected one primal per input plus one gradient output per "
                    "Active input");
      auto ptrs = mfem::future::make_tuple(&args...);
      zero_grads(ptrs, std::make_index_sequence<num_active_inputs> {});
#ifdef MFEM_USE_ENZYME
      output_view out_scratch {};
      output_view out_adjoint{1.0}; // seed: d(output)/d(output) = 1
      call_enzyme_rev(ptrs, out_scratch, out_adjoint);
#else
      call_dual_rev(ptrs);
#endif
   }

   static MFEM_FUTURE_ALWAYS_INLINE void print() { print_impl(std::make_index_sequence<arity> {}); }

   template <size_t... Is> static MFEM_FUTURE_ALWAYS_INLINE void print_impl(
      std::index_sequence<Is...>)
   {
      mfem::out << "__enzyme_autodiff<void>(fptr";
      (([&]
      {
         auto name = get_type_name<tuple_element_t<Is, args_tuple>>();
         if constexpr (Is == active_output)
            mfem::out << ", enzyme_dupnoneed, " << name << ", adjoint=1";
         else if constexpr (qf_param_is_active_v<activity, Is>)
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

template <typename Func, typename InputActivityTuple,
          typename OutputActivityTuple, RevDiffDualMode mode>
struct create_function_signature<RevDiff<Func, InputActivityTuple,
          OutputActivityTuple, mode>>
{
   using type = typename
                RevDiff<Func, InputActivityTuple, OutputActivityTuple, mode>::signature;
};

/// Builds the reverse-mode transform of @a f, differentiating the inputs marked
/// Active in @a activity_t.
///
/// A factory rather than a plain declaration of a RevDiff variable for compatibility with MSVC.
template <typename activity_t, RevDiffDualMode mode = RevDiffDualMode::Eval,
          typename func_t>
auto make_revdiff(const func_t &f)
{
   return RevDiff<func_t, activity_t, tuple<Active>, mode>(f);
}

} // namespace mfem::future
