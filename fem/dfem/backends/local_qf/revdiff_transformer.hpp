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

#include "../../util.hpp"

#ifdef MFEM_USE_ENZYME

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
template <typename Func, typename InputActivityTuple,
          typename OutputActivityTuple>
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
   static constexpr bool is_active =
      std::is_same_v<tuple_element_t<I, activity>, Active>;

   // Index of the single Active tag in [Lo, Hi), or arity if not exactly one.
   template <size_t Lo, size_t Hi, size_t... Is>
   static constexpr size_t find_single_active(std::index_sequence<Is...>)
   {
      size_t idx = arity, count = 0;
      (((Is >= Lo && Is < Hi && is_active<Is>) ? (idx = Is, ++count) : size_t{0}),
       ...);
      return count == 1 ? idx : arity;
   }

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
      find_single_active<num_inputs, arity>(std::make_index_sequence<arity> {});
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

   template <size_t S>
   using grad_arg_t =
      typename qp_traits<std::decay_t<tuple_element_t<active_inputs[S], args_tuple>>>::view_type
      &;

   template <size_t... Is, size_t... Ss>
   static FunctionSignature<void(primal_arg_t<Is>..., grad_arg_t<Ss>...)>
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
      zero_grads(ptrs, std::make_index_sequence<num_active_inputs> {});
      output_view out_scratch{};
      output_view out_adjoint{1.0}; // seed: d(output)/d(output) = 1
      call_enzyme_rev(ptrs, out_scratch, out_adjoint);
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

template <typename Func, typename InputActivityTuple, typename OutputActivityTuple>
struct create_function_signature<RevDiff<Func, InputActivityTuple, OutputActivityTuple>>
{
   using type = typename
                RevDiff<Func, InputActivityTuple, OutputActivityTuple>::signature;
};

} // namespace mfem::future

#endif // MFEM_USE_ENZYME
