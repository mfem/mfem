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

#include "qf_local_action.hpp"
#include "qf_local_derivative_action.hpp"

namespace mfem::future
{

struct LocalQFKernelsBackend
{
   static constexpr bool has_cached_derivative = false;

   /**
    * @brief Make an action for a local kernels backend.
    *
    * @param ctx The integrator context.
    * @param args The arguments to the action.
    * @return The action.
    */
   template<typename... Args>
   static auto MakeAction(const IntegratorContext &ctx, Args... args)
   {
      return LocalQFKernelsImpl::Action<Args...>(ctx, args...);
   }


   /**
    * @brief Make a derivative action for a local kernels backend.
    *
    * @tparam derivative_id The id of the derivative.
    * @param ctx The integrator context.
    * @param args The arguments to the derivative action.
    * @return The derivative action.
    */
   template<int id, typename... Args>
   static auto MakeDerivativeAction(const IntegratorContext &ctx, Args... args)
   {
      return LocalQFKernelsImpl::DerivativeAction<id, Args...>(ctx, args...);
   }
};

} // namespace mfem::future
