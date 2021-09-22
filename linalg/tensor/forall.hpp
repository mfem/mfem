// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TENSOR_FORALL
#define MFEM_TENSOR_FORALL

#include "config.hpp"

namespace mfem
{

#define MFEM_FORALL_CONFIG(config,i,N,...)                       \
   const int quads = config.quads;                               \
   constexpr int BatchSize = get_config_batchsize<Config>;       \
   const int X = config_use_xthreads<Config> ? quads : 1;        \
   const int Y = config_use_ythreads<Config> ? quads : 1;        \
   const int Z = BatchSize * ( config_use_ythreads<Config> ? quads : 1); \
   ForallWrap<3>(true,N,                                         \
                 [=] MFEM_DEVICE (int i) mutable {__VA_ARGS__},  \
                 [&] MFEM_LAMBDA (int i) {__VA_ARGS__},          \
                 X,Y,Z)

template <typename Config,
          typename Lambda,
          std::enable_if_t<
             config_use_1dthreads<Config>,
             bool
          > = true >
void forall(const Config &config, int ne, Lambda &&kernel)
{
   const int quads = config.quads;
   constexpr int BatchSize = get_config_batchsize<Config>;
   ForallWrap<3>(true,ne,
                 [=] MFEM_DEVICE (int i) { kernel(i); },
                 [&] MFEM_LAMBDA (int i) { kernel(i); },
                 quads, 1, BatchSize);
}

template <typename Config,
          typename Lambda,
          std::enable_if_t<
             config_use_2dthreads<Config>,
             bool
          > = true >
void forall(const Config &config, int ne, Lambda &&kernel)
{
   const int quads = config.quads;
   constexpr int BatchSize = get_config_batchsize<Config>;
   ForallWrap<3>(true,ne,
                 [=] MFEM_DEVICE (int i) { kernel(i); },
                 [&] MFEM_LAMBDA (int i) { kernel(i); },
                 quads, quads, BatchSize);
}

template <typename Config,
          typename Lambda,
          std::enable_if_t<
             config_use_3dthreads<Config>,
             bool
          > = true >
void forall(const Config &config, int ne, Lambda &&kernel)
{
   const int quads = config.quads;
   constexpr int BatchSize = get_config_batchsize<Config>;
   ForallWrap<3>(true,ne,
                 [=] MFEM_DEVICE (int i) { kernel(i); },
                 [&] MFEM_LAMBDA (int i) { kernel(i); },
                 quads, quads, quads * BatchSize);
}

} // namespace mfem

#endif // MFEM_TENSOR_FORALL
