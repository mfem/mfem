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

/// Macro to launch kernels that uses the config to choose the appropriate
/// threading strategy.
#define MFEM_FORALL_CONFIG(config,i,N,...)                                     \
   const int threads = config.quads > config.dofs ?                            \
                       config.quads : config.dofs;                             \
   using Config = decltype(config);                                            \
   constexpr int Bsize = get_config_batchsize<Config>;                         \
   const int Xthreads = config_use_xthreads<Config> ? threads : 1;             \
   const int Ythreads = config_use_ythreads<Config> ? threads : 1;             \
   const int Zthreads = Bsize * ( config_use_zthreads<Config> ? threads : 1 ); \
   ForallWrap<3>(true,N,                                                       \
                 [=] MFEM_DEVICE (int i) mutable {__VA_ARGS__},                \
                 [&] MFEM_LAMBDA (int i) {__VA_ARGS__},                        \
                 Xthreads, Ythreads, Zthreads)

} // namespace mfem

#endif // MFEM_TENSOR_FORALL
