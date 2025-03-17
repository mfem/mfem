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

#include <cstddef>
#include <utility>
#include "../general/backends.hpp" // IWYU pragma: keep

#define TMOP_ARG_COUNT_N(_1, _2, _3, _4, _5, N, ...) N
#define TMOP_ARG_COUNT(...) TMOP_ARG_COUNT_N(__VA_ARGS__, 5, 4, 3, 2, 1, 0)

#define TMOP_FOREACH_THREAD_N(MFEM_TMOP_FOREACH_THREAD_, N) TMOP_FOREACH_THREAD_ ## N
#define TMOP_FOREACH_THREAD_(N)  TMOP_FOREACH_THREAD_N(TMOP_FOREACH_THREAD_, N)
#define TMOP_FOREACH_THREAD(...) TMOP_FOREACH_THREAD_(TMOP_ARG_COUNT(__VA_ARGS__))(__VA_ARGS__)

#define TMOP_FOREACH_ERROR \
   static_assert(false, "MFEM_TMOP_FOREACH_THREAD requires exactly 2 or 3 arguments");

#define TMOP_FOREACH_THREAD_0(...) MFEM_TMOP_FOREACH_ERROR
#define TMOP_FOREACH_THREAD_1(...) MFEM_TMOP_FOREACH_ERROR
#define TMOP_FOREACH_THREAD_4(...) MFEM_TMOP_FOREACH_ERROR
#define TMOP_FOREACH_THREAD_5(...) MFEM_TMOP_FOREACH_ERROR

#define TMOP_FOREACH_THREAD_2(i, k) int i = hipThreadIdx_ ##k; if (i<hipBlockDim_ ##k)

#define TMOP_FOREACH_THREAD_3(i, k, N) TMOP_FOREACH_THREAD(i, k, N)

namespace mfem
{

#if defined(MFEM_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)
template <int N, typename F> inline MFEM_HOST_DEVICE
void foreach_thread(F&& func)
{
   if (hipThreadIdx_ ##k < hipBlockDim_ ##k)
   {
      func(hipThreadIdx_ ##k);
   }
}
#else
template <int I, int N, typename F>
struct StaticFor
{
   static inline MFEM_HOST_DEVICE void apply(F&& func)
   {
      func(I);
      StaticFor<I + 1, N, F>::apply(std::forward<F>(func));
   }
};

template <int N, typename F>
struct StaticFor<N, N, F> { static inline MFEM_HOST_DEVICE void apply(F&&) {} };

template <int N, typename F> inline MFEM_HOST_DEVICE
void foreach_thread(F&& func)
{
   StaticFor<0, N, F>::apply(std::forward<F>(func));
}
#endif

template <int N, typename F> inline MFEM_HOST_DEVICE
void foreach_thread_x(F&& func) { foreach_thread<N>(func); }

template <int N, typename F> inline MFEM_HOST_DEVICE
void foreach_thread_y(F&& func) { foreach_thread<N>(func); }

template <int N, typename F> inline MFEM_HOST_DEVICE
void foreach_thread_z(F&& func) { foreach_thread<N>(func); }

} // namespace mfem
