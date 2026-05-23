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
#include "qf_local_util.hpp"

namespace mfem::future
{

/// Evaluate a differentiable q-function at one quadrature point.
/// Primal registers live in @a rargs; input tangents in @a sargs (Enzyme shadow).
template<
   typename backend_t,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t,
   int MQ1,
   typename RArgs,
   typename SArgs,
   typename InXE,
   typename InXEd,
   typename OutYE>
MFEM_HOST_DEVICE inline void derivative_evaluate_at_qp(
   const qfunc_t &qfunc,
   RArgs &rargs,
   SArgs &sargs,
   const std::array<bool, tuple_size<inputs_t>::value> &input_dep,
   const int qx, const int qy, const int qz, const int e,
   const InXE &in_XE,
   const InXEd &in_XE_dir,
   OutYE &out_YE)
{
   using qf_signature = typename get_function_signature<qfunc_t>::type;
   using qf_param_ts = typename qf_signature::parameter_ts;
   using args_tuple_t = decay_tuple<qf_param_ts>;
   static constexpr std::size_t n_inputs = tuple_size<inputs_t>::value;
   static constexpr std::size_t n_outputs = tuple_size<outputs_t>::value;

   if constexpr (backend_t::derivative_use_enzyme)
   {
      args_tuple_t primal{}, enzyme_shadow{};

      for_constexpr<n_inputs>([&](auto ic)
      {
         constexpr size_t i = ic.value;
         auto &parg = get<i>(primal);
         auto &targ = get<i>(enzyme_shadow);
         const auto &XE = in_XE[i];
         const auto &XEd = in_XE_dir[i];
         using FOP = tuple_element_t<i, inputs_t>;
         using ARG = typename qf_param_slot<qfunc_t, i>::qf_reg_param_t;
         if constexpr (is_identity_fop_v<FOP>)
         {
            parg = as_tensor<ARG>(&XE(0, qx, qy, qz, e));
            if (input_dep[i])
            {
               targ = as_tensor<ARG>(&XEd(0, qx, qy, qz, e));
            }
            else { targ = ARG{}; }
         }
         else if constexpr (is_weight_fop_v<FOP>)
         {
            parg = XE(qx, qy, qz, 0, 0);
            targ = real_t(0.0);
         }
         else if constexpr (is_value_fop_v<FOP> || is_gradient_fop_v<FOP>)
         {
            parg = backend_t::template qp_pull<ARG, MQ1>
            (get<i>(rargs), qx, qy, qz);
            if (input_dep[i])
            {
               targ = backend_t::template qp_pull<ARG, MQ1>
               (get<i>(sargs), qx, qy, qz);
            }
            else { targ = ARG{}; }
         }
         else { static_assert(false, "Unsupported"); }
      });

      call_enzyme_fwddiff(qfunc, primal, enzyme_shadow);

      for_constexpr<n_outputs>([&](auto ic)
      {
         constexpr size_t i = ic.value, o = n_inputs + i;
         const auto &qout = get<o>(enzyme_shadow);
         auto &YE = out_YE[i];
         using FOP = tuple_element_t<i, outputs_t>;
         using ARG = typename qf_param_slot<qfunc_t, o>::qf_reg_param_t;
         if constexpr (is_identity_fop_v<FOP>)
         {
            as_tensor<ARG>(&YE(0, qx, qy, qz, e)) = qout;
         }
         else if constexpr (is_value_fop_v<FOP> || is_gradient_fop_v<FOP>)
         {
            auto &rarg = get<o>(rargs);
            backend_t::template qp_push_tangent<ARG, MQ1>
            (rarg, qx, qy, qz, qout);
         }
         else { static_assert(false, "Unsupported"); }
      });
   }
   else
   {
      args_tuple_t qargs;

      for_constexpr<n_inputs>([&](auto ic)
      {
         constexpr size_t i = ic.value;
         auto &qarg = get<i>(qargs);
         const auto &XE = in_XE[i];
         const auto &XEd = in_XE_dir[i];
         using FOP = tuple_element_t<i, inputs_t>;
         using ARG = typename qf_param_slot<qfunc_t, i>::qf_reg_param_t;
         if constexpr (is_identity_fop_v<FOP>)
         {
            using DT = typename qf_param_slot<qfunc_t, i>::qf_decay_param_t;
            if constexpr (qf_param_uses_dual_v<DT>)
            {
               qarg = backend_t::template identity_qp_pull_dual<DT>
               (input_dep[i], XE, XEd, qx, qy, qz, e);
            }
            else
            {
               qarg = as_tensor<ARG>(&XE(0, qx, qy, qz, e));
            }
         }
         else if constexpr (is_weight_fop_v<FOP>)
         {
            qarg = XE(qx, qy, qz, 0, 0);
         }
         else if constexpr (is_value_fop_v<FOP> || is_gradient_fop_v<FOP>)
         {
            qarg = backend_t::template qp_pull_directional<ARG, MQ1>
            (get<i>(rargs), get<i>(sargs), qx, qy, qz, input_dep[i]);
         }
         else { static_assert(false, "Unsupported"); }
      });

      call_qfunc_no_move(qfunc, qargs);

      for_constexpr<n_outputs>([&](auto ic)
      {
         constexpr size_t i = ic.value, o = n_inputs + i;
         const auto &qarg = get<o>(qargs);
         auto &YE = out_YE[i];
         using FOP = tuple_element_t<i, outputs_t>;
         using ARG = typename qf_param_slot<qfunc_t, o>::qf_reg_param_t;
         if constexpr (is_identity_fop_v<FOP>)
         {
            using DT = typename qf_param_slot<qfunc_t, o>::qf_decay_param_t;
            if constexpr (qf_param_uses_dual_v<DT>)
            {
               backend_t::identity_qp_write_tangent
               (YE, qx, qy, qz, e, qarg);
            }
            else
            {
               as_tensor<ARG>(&YE(0, qx, qy, qz, e)) = qarg;
            }
         }
         else if constexpr (is_value_fop_v<FOP> || is_gradient_fop_v<FOP>)
         {
            auto &rarg = get<o>(rargs);
            backend_t::template qp_push_tangent<ARG, MQ1>
            (rarg, qx, qy, qz, qarg);
         }
         else { static_assert(false, "Unsupported"); }
      });
   }
}

} // namespace mfem::future
