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

#ifdef MFEM_USE_ENZYME

namespace mfem::future
{

template <typename T> struct function_traits;

template <typename R, typename C, typename... Args>
struct function_traits<R (C::*)(Args...) const>
{
   using primal_return_type = R;
   using args_tuple = std::tuple<Args...>;
   static constexpr size_t arity = sizeof...(Args);
};

// Component count and writable counterpart of a per-point argument, which
// is either a tensor or a plain scalar.
template <typename Arg> struct qp_traits
{
   static_assert(std::is_arithmetic_v<Arg>,
                 "per-point arguments must be tensors or scalars");
   using view_type = Arg;
   static constexpr int components = 1;
};

template <typename T, int... Sizes> struct qp_traits<tensor<T, Sizes...>>
{
   using view_type = tensor<std::remove_const_t<T>, Sizes...>;
   static constexpr int components = (Sizes * ... * 1);
};

// Generic FwdDiff: computes the full gradient of a pointwise qfunction at a
// single quadrature point.
//
// active_input is the index of the argument to differentiate with respect
// to; active_output is the index of the (scalar) output argument whose
// derivative is taken.
//
// operator()(args...) takes the qfunction's arguments, except that the
// active output position receives the *gradient*, shaped like the active
// input (d(output)/d(input component)).
//
// Per input component d, one enzyme fwddiff call with the one-hot seed e_d
// in the input tangent yields gradient entry d, i.e. grad_components enzyme
// calls per point. The output tangent is not pre-zeroed, so the qfunction
// must fully write its output (tangent stores overwrite).
//
// operator() is MFEM_HOST_DEVICE, stateless and allocation-free, so it is
// callable inside a GPU kernel: seed, primal scratch and tangent all live
// on the stack.
template <typename Func, size_t active_input, size_t active_output>
struct FwdDiff
{
   using traits = function_traits<decltype(&Func::operator())>;
   using args_tuple = typename traits::args_tuple;
   static constexpr size_t arity = traits::arity;

   static_assert(std::is_void_v<typename traits::primal_return_type>,
                 "FwdDiff only supports primal functions with void return type");
   static_assert(active_input < arity && active_output < arity,
                 "active argument indices must be within the function arity");
   static_assert(active_input != active_output,
                 "active input and output must be different arguments");

   using input_type =
      std::decay_t<std::tuple_element_t<active_input, args_tuple>>;
   using output_type =
      std::decay_t<std::tuple_element_t<active_output, args_tuple>>;

   using grad_type = typename qp_traits<input_type>::view_type;
   using output_view = typename qp_traits<output_type>::view_type;

   static constexpr int grad_components = qp_traits<input_type>::components;

   static_assert(qp_traits<output_type>::components == 1,
                 "gradient output requires a scalar output");

   // Signature of the differentiated qfunction: the primal arguments, with
   // the Active output slot receiving the (writable) gradient instead.
   // Exposed through create_function_signature below so that
   // DifferentiableOperator can deduce the parameter types, which it cannot
   // do from the variadic operator().
   template <size_t I>
   using qf_arg_t = std::conditional_t<I == active_output, grad_type &,
         std::tuple_element_t<I, args_tuple>>;

   template <size_t... Is>
   static FunctionSignature<void(qf_arg_t<Is>...)>
   signature_impl(std::index_sequence<Is...>);

   using signature =
      decltype(signature_impl(std::make_index_sequence<arity> {}));

   // d-th scalar of a per-point argument in flat row-major order, regardless
   // of rank, built on the native operator[] (tensor has no flat-index
   // accessor; flatten() returns a copy, so it cannot be written through).
   MFEM_HOST_DEVICE static double &component(double &t, int) { return t; }

   template <typename T, int n0, int... n>
   MFEM_HOST_DEVICE static T &component(tensor<T, n0, n...> &t, int d)
   {
      if constexpr (sizeof...(n) == 0)
      {
         return t[d];
      }
      else
      {
         constexpr int stride = (n * ... * 1);
         return component(t[d / stride], d % stride);
      }
   }

   // Plain function with the qfunction's exact (reference) signature, so it
   // can be handed to Enzyme as a function pointer; references are pointers
   // to Enzyme, so primal arguments and shadows are passed by address below.
   template <size_t... Is>
   MFEM_HOST_DEVICE static void
   static_call(std::tuple_element_t<Is, args_tuple>... args)
   {
      Func{}(args...);
   }

   template <size_t... Is>
   static constexpr auto fn_ptr(std::index_sequence<Is...>)
   {
      return &static_call<Is...>;
   }
   static constexpr auto fn = fn_ptr(std::make_index_sequence<arity> {});

   // Writable, zero-initialized scratch with the shape of argument I, used
   // as its enzyme shadow.
   template <size_t I>
   using shadow_t = typename qp_traits<
                    std::decay_t<std::tuple_element_t<I, args_tuple>>>::view_type;

   template <size_t... Is>
   MFEM_HOST_DEVICE static auto make_shadows(std::index_sequence<Is...>)
   {
      return mfem::future::make_tuple(shadow_t<Is> {}...);
   }

   template <typename Shadows, size_t... Is>
   MFEM_HOST_DEVICE static auto make_shadow_ptrs(Shadows &shadows,
                                                 std::index_sequence<Is...>)
   {
      return mfem::future::make_tuple(&mfem::future::get<int(Is)>(shadows)...);
   }

   // The caller's argument pointers, except the active output slot, which
   // points to scalar scratch: the caller's slot holds the gradient, while
   // the primal function writes its scalar output there.
   template <size_t I, typename Ptrs>
   MFEM_HOST_DEVICE static auto primal_ptr(Ptrs &ptrs, output_view &primal)
   {
      if constexpr (I == active_output) { return &primal; }
      else { return mfem::future::get<int(I)>(ptrs); }
   }

   template <typename Ptrs, size_t... Is>
   MFEM_HOST_DEVICE static auto make_primal_ptrs(Ptrs &ptrs,
                                                 output_view &primal,
                                                 std::index_sequence<Is...>)
   {
      return mfem::future::make_tuple(primal_ptr<Is>(ptrs, primal)...);
   }

   // Single flat enzyme call. The activity markers must appear directly in
   // the __enzyme_fwddiff argument list — Enzyme cannot trace markers that
   // were forwarded through function parameters (e.g. at -O0, where nothing
   // is inlined). Every argument is therefore enzyme_dup'd in one sticky
   // group; Const arguments simply carry a zero tangent, which is equivalent
   // to marking them enzyme_const.
   //
   // always_inline is load-bearing: when FwdDiff is itself differentiated
   // (second derivatives, forward-over-forward), Enzyme only recognizes this
   // nested __enzyme_fwddiff call if it sits at most one call level below
   // the function handed to the outer __enzyme_fwddiff. Without inlining
   // (-O0) it sits two levels down (wrapper -> operator() -> call_enzyme)
   // and the outer pass treats it as a regular call: the activity marker
   // ints then receive undef shadows, which misaligns the argument pairing
   // (observed as "cannot compute with global variable that doesn't have
   // marked shadow global" at compile time or null-shadow segfaults at
   // runtime). The always-inliner runs even at -O0, hoisting this call into
   // operator() where the nested handling applies.
   template <typename PrimalPtrs, typename ShadowPtrs, size_t... Is>
   __attribute__((always_inline))
   MFEM_HOST_DEVICE static void call_enzyme(PrimalPtrs &primal_ptrs,
                                            ShadowPtrs &shadow_ptrs,
                                            std::index_sequence<Is...>)
   {
      __enzyme_fwddiff<void>(fn, enzyme_dup,
                             mfem::future::get<int(Is)>(primal_ptrs)...,
                             enzyme_interleave,
                             mfem::future::get<int(Is)>(shadow_ptrs)...,
                             enzyme_runtime_activity);
   }

   template <typename... Args>
   MFEM_HOST_DEVICE void operator()(Args &&...args) const
   {
      static_assert(sizeof...(Args) == arity, "Wrong number of arguments");
      auto ptrs = mfem::future::make_tuple(&args...);

      auto &grad = *mfem::future::get<int(active_output)>(ptrs);
      static_assert(std::is_same_v<std::decay_t<decltype(grad)>, grad_type>,
                    "gradient argument must be shaped like the Active input "
                    "(with writable scalars)");

      constexpr auto seq = std::make_index_sequence<arity> {};

      output_view primal{};
      auto primal_ptrs = make_primal_ptrs(ptrs, primal, seq);

      auto shadows = make_shadows(seq);
      auto shadow_ptrs = make_shadow_ptrs(shadows, seq);
      auto &seed = mfem::future::get<int(active_input)>(shadows);
      auto &tangent = mfem::future::get<int(active_output)>(shadows);

      // One enzyme call per input component d: seed e_d in the input tangent
      // and read gradient entry d off the output tangent.
      for (int d = 0; d < grad_components; d++)
      {
         component(seed, d) = 1.0;
         call_enzyme(primal_ptrs, shadow_ptrs, seq);
         component(grad, d) = component(tangent, 0);
         component(seed, d) = 0.0;
      }
   }

   static void print() { print_impl(std::make_index_sequence<arity> {}); }

   template <size_t... Is> static void print_impl(std::index_sequence<Is...>)
   {
      std::cout << "for d in [0, " << grad_components
                << "): __enzyme_fwddiff<void>(fptr, enzyme_dup";
      ((std::cout << ", "
        << get_type_name<std::tuple_element_t<Is, args_tuple>>()),
       ...);
      std::cout << ", enzyme_interleave";
      (([&]
      {
         if constexpr (Is == active_input) { std::cout << ", e_d seed"; }
         else if constexpr (Is == active_output) { std::cout << ", tangent out"; }
         else { std::cout << ", zero tangent"; }
      }()),
      ...);
      std::cout << ")\n";
   }
};

template <typename Func, size_t active_input, size_t active_output>
struct create_function_signature<FwdDiff<Func, active_input, active_output>>
{
   using type =
      typename FwdDiff<Func, active_input, active_output>::signature;
};

} // namespace mfem::future

#endif // MFEM_USE_ENZYME
