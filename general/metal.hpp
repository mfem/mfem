// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

#include "metal.h"

#include "../config/config.hpp"
#include "mem_manager.hpp"

using mfem::real_t;

namespace mfem
{

namespace metal
{

/// ////////////////////////////////////////////////////////////////////////////
using setup_t =
   std::tuple<
   MTL::Device*,                 // device
   MTL::CommandBuffer*,          // commands
   MTL::ComputeCommandEncoder*,  // encoder
   NS::UInteger>;                // maxTotalThreadsPerThreadgroup

/// ////////////////////////////////////////////////////////////////////////////
std::string KernelOps(const char *name, const char *ops);
setup_t KernelSetup(const char* name, const char *src);

/// ////////////////////////////////////////////////////////////////////////////
template <typename F, typename D, typename E, typename... Args>
void KernelApply(F func, D &dev, E &enc, Args&&... args)
{
   (func(dev, enc, std::forward<Args>(args)), ...);
}

/// ////////////////////////////////////////////////////////////////////////////
template <typename... Args>
void Kernel_1D(const size_t N, const char* name, const char *ops,
               Args &&...args)
{
   const std::string src = KernelOps(name, ops);
   auto [device, commands, encoder, KerMaxTPG] = KernelSetup(name, src.c_str());

   // enqueue the argument buffers
   int k = 0;
   auto enqueue = [&k](auto &device, auto&encoder, auto &arg)
   {
      // dbg("\t#{} {}", k, typeid(arg).name());
      using arg_t = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<arg_t, real_t*> ||
                    std::is_same_v<arg_t, const real_t*>)
      {
         encoder->setBuffer(mfem::MemoryManager::GetDeviceBfr(arg), 0, k++);
      }
      else if constexpr (std::is_same_v<arg_t, float>)
      {
         // turn real_t into a buffer
         auto N_MTL = device->newBuffer(sizeof(float), MTL::ResourceStorageModeManaged);
         *static_cast<float *>(N_MTL->contents()) = arg;
         encoder->setBuffer(N_MTL, 0, k++);
      }
      else
      {
         MFEM_ABORT("unsupported type");
      }
   };

   KernelApply(enqueue, device, encoder, args...);

   const auto threadGroupSize = KerMaxTPG > N ? N : KerMaxTPG;
   const auto threadsPerThreadgroup = MTL::Size::Make(threadGroupSize, 1, 1);
   const auto threadsPerGrid = MTL::Size::Make(N, 1, 1);

   // Encode the compute command
   encoder->dispatchThreads(threadsPerGrid, threadsPerThreadgroup);
   encoder->endEncoding();

   commands->commit();
   commands->waitUntilCompleted();
}

} // namespace metal

} // namespace mfem
