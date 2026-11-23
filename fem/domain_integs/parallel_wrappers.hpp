#ifndef PARALLEL_WRAPPERS_HPP
#define PARALLEL_WRAPPERS_HPP

#include <mfem.hpp>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#ifdef MFEM_USE_OPENMP
#include <omp.h>
#endif

namespace mfem {

#if defined(__CUDACC__) || defined(__HIP__)
#define MFEM_NESTED_ALWAYS_INLINE __forceinline__
#else
#define MFEM_NESTED_ALWAYS_INLINE inline
#endif

#if (defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)) ||                    \
    (defined(MFEM_USE_HIP) && defined(__HIP_DEVICE_COMPILE__))
// contiguous in x
#define MFEM_FOREACH_THREAD_2D(ix, iy, k, SX, SY)                              \
  MFEM_FOREACH_THREAD(mfem_flat_idx, k, (SX) * (SY))                           \
  if (int ix = mfem_flat_idx % (SX), iy = mfem_flat_idx / (SX); true)
#define MFEM_FOREACH_THREAD_3D(ix, iy, iz, k, SX, SY, SZ)                      \
  MFEM_FOREACH_THREAD(mfem_flat_idx, k, (SX) * (SY) * (SZ))                    \
  if (int ix = mfem_flat_idx % (SX), iy = mfem_flat_idx / (SX),                \
      iz = iy / (SY);                                                          \
      (iy %= (SY)), true)
#define MFEM_FOREACH_THREAD_4D(ix, iy, iz, iw, k, SX, SY, SZ, SW)              \
  MFEM_FOREACH_THREAD(mfem_flat_idx, k, (SX) * (SY) * (SZ) * (SW))             \
  if (int ix = mfem_flat_idx % (SX), iy = mfem_flat_idx / (SX),                \
      iz = iy / (SY);                                                          \
      (iy %= (SY)), true)                                                      \
    if (int iw = iz / (SZ); (iz %= (SZ)), true)
#define MFEM_FOREACH_THREAD_5D(ix, iy, iz, iw, iv, k, SX, SY, SZ, SW, SV)      \
  MFEM_FOREACH_THREAD(mfem_flat_idx, k, (SX) * (SY) * (SZ) * (SW) * (SV))      \
  if (int ix = mfem_flat_idx % (SX), iy = mfem_flat_idx / (SX),                \
      iz = iy / (SY);                                                          \
      (iy %= (SY)), true)                                                      \
    if (int iw = iz / (SZ); (iz %= (SZ)), true)                                \
      if (int iv = iw / (SW); (iw %= (SW)), true)
#define MFEM_FOREACH_THREAD_6D(ix, iy, iz, iw, iv, iu, k, SX, SY, SZ, SW, SV,  \
                               SU)                                             \
  MFEM_FOREACH_THREAD(mfem_flat_idx, k,                                        \
                      (SX) * (SY) * (SZ) * (SW) * (SV) * (SU))                 \
  if (int ix = mfem_flat_idx % (SX), iy = mfem_flat_idx / (SX),                \
      iz = iy / (SY);                                                          \
      (iy %= (SY)), true)                                                      \
    if (int iw = iz / (SZ); (iz %= (SZ)), true)                                \
      if (int iv = iw / (SW); (iw %= (SW)), true)                              \
        if (int iu = iv / (SV); (iv %= (SV)), true)

// Flattened direct mappings for kernels which guarantee that the block has at
// least as many threads as loop entries.  Unlike MFEM_FOREACH_THREAD_<N>D,
// these deliberately assign at most one entry to each thread.
#define MFEM_FOREACH_THREAD_2D_DIRECT(ix, iy, k, SX, SY)                       \
  MFEM_FOREACH_THREAD_DIRECT(mfem_flat_idx, k, (SX) * (SY))                    \
  if (int ix = mfem_flat_idx % (SX), iy = mfem_flat_idx / (SX); true)
#define MFEM_FOREACH_THREAD_3D_DIRECT(ix, iy, iz, k, SX, SY, SZ)               \
  MFEM_FOREACH_THREAD_DIRECT(mfem_flat_idx, k, (SX) * (SY) * (SZ))             \
  if (int ix = mfem_flat_idx % (SX), iy = mfem_flat_idx / (SX),                \
      iz = iy / (SY);                                                          \
      (iy %= (SY)), true)
#define MFEM_FOREACH_THREAD_4D_DIRECT(ix, iy, iz, iw, k, SX, SY, SZ, SW)       \
  MFEM_FOREACH_THREAD_DIRECT(mfem_flat_idx, k, (SX) * (SY) * (SZ) * (SW))      \
  if (int ix = mfem_flat_idx % (SX), iy = mfem_flat_idx / (SX),                \
      iz = iy / (SY);                                                          \
      (iy %= (SY)), true)                                                      \
    if (int iw = iz / (SZ); (iz %= (SZ)), true)
#define MFEM_FOREACH_THREAD_5D_DIRECT(ix, iy, iz, iw, iv, k, SX, SY, SZ, SW,   \
                                      SV)                                      \
  MFEM_FOREACH_THREAD_DIRECT(mfem_flat_idx, k,                                 \
                             (SX) * (SY) * (SZ) * (SW) * (SV))                 \
  if (int ix = mfem_flat_idx % (SX), iy = mfem_flat_idx / (SX),                \
      iz = iy / (SY);                                                          \
      (iy %= (SY)), true)                                                      \
    if (int iw = iz / (SZ); (iz %= (SZ)), true)                                \
      if (int iv = iw / (SW); (iw %= (SW)), true)
#define MFEM_FOREACH_THREAD_6D_DIRECT(ix, iy, iz, iw, iv, iu, k, SX, SY, SZ,   \
                                      SW, SV, SU)                              \
  MFEM_FOREACH_THREAD_DIRECT(                                                  \
      mfem_flat_idx, k, (SX) * (SY) * (SZ) * (SW) * (SV) * (SU))              \
  if (int ix = mfem_flat_idx % (SX), iy = mfem_flat_idx / (SX),                \
      iz = iy / (SY);                                                          \
      (iy %= (SY)), true)                                                      \
    if (int iw = iz / (SZ); (iz %= (SZ)), true)                                \
      if (int iv = iw / (SW); (iw %= (SW)), true)                              \
        if (int iu = iv / (SV); (iv %= (SV)), true)

// Compile-time selection between direct and strided flattened mappings.  The
// direct case has a single provable iteration and avoids the loop back-edge.
#define MFEM_FOREACH_THREAD_ND_MODE(DIRECT, mfem_flat_idx, k, N)               \
  for (int mfem_flat_idx = MFEM_THREAD_ID(k),                                  \
           mfem_thread_step = (DIRECT) ? (N) : MFEM_THREAD_SIZE(k);            \
       mfem_flat_idx < (N); mfem_flat_idx += mfem_thread_step)
#define MFEM_FOREACH_THREAD_1D_MODE(DIRECT, ix, k, SX)                         \
  MFEM_FOREACH_THREAD_ND_MODE(DIRECT, ix, k, SX)
#define MFEM_FOREACH_THREAD_2D_MODE(DIRECT, ix, iy, k, SX, SY)                 \
  MFEM_FOREACH_THREAD_ND_MODE(DIRECT, mfem_flat_idx, k, (SX) * (SY))           \
  if (int ix = mfem_flat_idx % (SX), iy = mfem_flat_idx / (SX); true)
#define MFEM_FOREACH_THREAD_3D_MODE(DIRECT, ix, iy, iz, k, SX, SY, SZ)         \
  MFEM_FOREACH_THREAD_ND_MODE(DIRECT, mfem_flat_idx, k,                        \
                              (SX) * (SY) * (SZ))                              \
  if (int ix = mfem_flat_idx % (SX), iy = mfem_flat_idx / (SX),                \
      iz = iy / (SY);                                                          \
      (iy %= (SY)), true)
#define MFEM_FOREACH_THREAD_4D_MODE(DIRECT, ix, iy, iz, iw, k, SX, SY, SZ, SW) \
  MFEM_FOREACH_THREAD_ND_MODE(DIRECT, mfem_flat_idx, k,                        \
                              (SX) * (SY) * (SZ) * (SW))                       \
  if (int ix = mfem_flat_idx % (SX), iy = mfem_flat_idx / (SX),                \
      iz = iy / (SY);                                                          \
      (iy %= (SY)), true)                                                      \
    if (int iw = iz / (SZ); (iz %= (SZ)), true)
#else
#define MFEM_FOREACH_THREAD_2D(ix, iy, k, SX, SY)                            \
  for (int iy = 0; iy < (SY); ++iy)                                          \
    for (int ix = 0; ix < (SX); ++ix)
#define MFEM_FOREACH_THREAD_3D(ix, iy, iz, k, SX, SY, SZ)                    \
  for (int iz = 0; iz < (SZ); ++iz)                                          \
    for (int iy = 0; iy < (SY); ++iy)                                        \
      for (int ix = 0; ix < (SX); ++ix)
#define MFEM_FOREACH_THREAD_4D(ix, iy, iz, iw, k, SX, SY, SZ, SW)            \
  for (int iw = 0; iw < (SW); ++iw)                                          \
    for (int iz = 0; iz < (SZ); ++iz)                                        \
      for (int iy = 0; iy < (SY); ++iy)                                      \
        for (int ix = 0; ix < (SX); ++ix)
#define MFEM_FOREACH_THREAD_5D(ix, iy, iz, iw, iv, k, SX, SY, SZ, SW, SV)    \
  for (int iv = 0; iv < (SV); ++iv)                                          \
    for (int iw = 0; iw < (SW); ++iw)                                        \
      for (int iz = 0; iz < (SZ); ++iz)                                      \
        for (int iy = 0; iy < (SY); ++iy)                                    \
          for (int ix = 0; ix < (SX); ++ix)
#define MFEM_FOREACH_THREAD_6D(ix, iy, iz, iw, iv, iu, k, SX, SY, SZ, SW,    \
                               SV, SU)                                       \
  for (int iu = 0; iu < (SU); ++iu)                                          \
    for (int iv = 0; iv < (SV); ++iv)                                        \
      for (int iw = 0; iw < (SW); ++iw)                                      \
        for (int iz = 0; iz < (SZ); ++iz)                                    \
          for (int iy = 0; iy < (SY); ++iy)                                  \
            for (int ix = 0; ix < (SX); ++ix)

// CPU execution is serial within a nested kernel, so direct and strided inner
// loops intentionally have the same mapping.
#define MFEM_FOREACH_THREAD_2D_DIRECT MFEM_FOREACH_THREAD_2D
#define MFEM_FOREACH_THREAD_3D_DIRECT MFEM_FOREACH_THREAD_3D
#define MFEM_FOREACH_THREAD_4D_DIRECT MFEM_FOREACH_THREAD_4D
#define MFEM_FOREACH_THREAD_5D_DIRECT MFEM_FOREACH_THREAD_5D
#define MFEM_FOREACH_THREAD_6D_DIRECT MFEM_FOREACH_THREAD_6D
#define MFEM_FOREACH_THREAD_1D_MODE(DIRECT, ix, k, SX)                        \
  MFEM_FOREACH_THREAD(ix, k, SX)
#define MFEM_FOREACH_THREAD_2D_MODE(DIRECT, ix, iy, k, SX, SY)                \
  MFEM_FOREACH_THREAD_2D(ix, iy, k, SX, SY)
#define MFEM_FOREACH_THREAD_3D_MODE(DIRECT, ix, iy, iz, k, SX, SY, SZ)        \
  MFEM_FOREACH_THREAD_3D(ix, iy, iz, k, SX, SY, SZ)
#define MFEM_FOREACH_THREAD_4D_MODE(DIRECT, ix, iy, iz, iw, k, SX, SY, SZ,    \
                                    SW)                                       \
  MFEM_FOREACH_THREAD_4D(ix, iy, iz, iw, k, SX, SY, SZ, SW)
#endif

/// Portable baseline.  Kernels needing more can use the global-workspace
/// overload below instead of relying on a device-specific opt-in limit.
inline constexpr int forall_nested_shared_memory_limit = 48 * 1024;

inline int forall_nested_max_concurrent_blocks(bool allow_device = true) {
  if (allow_device) {
#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
    if (mfem::Device::Allows(mfem::Backend::CUDA | mfem::Backend::HIP |
                             mfem::Backend::RAJA_CUDA | mfem::Backend::RAJA_HIP)) {
#if defined(MFEM_USE_CUDA)
      constexpr int mp_sat = 8;
#else
      constexpr int mp_sat = 4;
#endif
      return mp_sat * mfem::Device::NumMultiprocessors(mfem::Device::GetId());
    }
#endif
#if defined(MFEM_USE_OPENMP) && defined(_OPENMP)
    if (mfem::Device::Allows(mfem::Backend::OMP | mfem::Backend::RAJA_OMP)) {
      return omp_get_max_threads();
    }
#endif
  }
  return 1;
}

constexpr int forall_nested_num_blocks(int N, int max_concurrent_blocks) {
  return N > 0 && max_concurrent_blocks > 0 ? std::min(N, max_concurrent_blocks)
                                            : 0;
}

constexpr std::size_t forall_nested_workspace_stride(int workspace_nbytes) {
  constexpr std::size_t alignment = alignof(std::max_align_t);
  return workspace_nbytes > 0
             ? (static_cast<std::size_t>(workspace_nbytes) + alignment - 1) /
                   alignment * alignment
             : 0;
}

/// Bytes callers must allocate when a kernel's per-block workspace is too
/// large for the portable shared-memory path.
constexpr std::size_t
forall_nested_global_workspace_size(int N, int max_concurrent_blocks,
                                    int workspace_nbytes) {
  return static_cast<std::size_t>(
             forall_nested_num_blocks(N, max_concurrent_blocks)) *
         forall_nested_workspace_stride(workspace_nbytes);
}

template <class F>
void serial_runner(F &&func, int N, int workspace_nbytes,
                   char *global_workspace = nullptr,
                   std::size_t global_stride = 0) {
  // assume N > 0 (should have been checked previously)
  const auto words = (forall_nested_workspace_stride(workspace_nbytes) +
                      sizeof(std::max_align_t) - 1) /
                     sizeof(std::max_align_t);
  std::vector<std::max_align_t> local(global_workspace ? 0 : words);
  auto *buffer = global_workspace ? global_workspace
                                  : reinterpret_cast<char *>(local.data());
  func(0, N, buffer);
}
#if defined(MFEM_USE_OPENMP) && defined(_OPENMP)
template <class F>
void openmp_runner(F func, int N, int max_concurrent_blocks,
                   int workspace_nbytes, char *global_workspace = nullptr,
                   std::size_t global_stride = 0) {
  // assume N > 0 (should have been checked previously)
  int max_threads =
      std::min(std::min(omp_get_max_threads(), max_concurrent_blocks), N);
  int Nper = (N + max_threads - 1) / max_threads;
#pragma omp parallel num_threads(max_threads)
  {
    int i = omp_get_thread_num();
    const auto words = (forall_nested_workspace_stride(workspace_nbytes) +
                        sizeof(std::max_align_t) - 1) /
                       sizeof(std::max_align_t);
    std::vector<std::max_align_t> local(global_workspace ? 0 : words);
    auto *buffer = global_workspace ? global_workspace + i * global_stride
                                    : reinterpret_cast<char *>(local.data());
    func(i * Nper, std::min((i + 1) * Nper, N), buffer);
  }
}
#endif

#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
template <int MAX_THREADS_PER_BLOCK, class F>
__global__ MFEM_LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK) void gpu_shared_runner(
    F func, int N, int Nmult) {
  extern __shared__ char buf[];
  int Nstart = MFEM_BLOCK_ID(x) * Nmult;
  int Nstop = min((MFEM_BLOCK_ID(x) + 1) * Nmult, N);
  func(Nstart, Nstop, buf);
}

template <int MAX_THREADS_PER_BLOCK, class F>
__global__ MFEM_LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK) void gpu_global_runner(
    F func, int N, int Nmult, char *global_workspace,
    std::size_t global_stride) {
  int Nstart = MFEM_BLOCK_ID(x) * Nmult;
  int Nstop = min((MFEM_BLOCK_ID(x) + 1) * Nmult, N);
  auto *workspace = global_workspace + MFEM_BLOCK_ID(x) * global_stride;
  func(Nstart, Nstop, workspace);
}

template <class F>
__global__ void gpu_shared_runner(F func, int N, int Nmult) {
  extern __shared__ char buf[];
  int Nstart = MFEM_BLOCK_ID(x) * Nmult;
  int Nstop = min((MFEM_BLOCK_ID(x) + 1) * Nmult, N);
  func(Nstart, Nstop, buf);
}

template <class F>
__global__ void gpu_global_runner(F func, int N, int Nmult,
                                  char *global_workspace,
                                  std::size_t global_stride) {
  int Nstart = MFEM_BLOCK_ID(x) * Nmult;
  int Nstop = min((MFEM_BLOCK_ID(x) + 1) * Nmult, N);
  auto *workspace = global_workspace + MFEM_BLOCK_ID(x) * global_stride;
  func(Nstart, Nstop, workspace);
}
#endif

#ifdef MFEM_USE_RAJA
template <class Policy, class F>
void raja_host_nested_runner(F func, int N, int num_blocks,
                            int workspace_nbytes, char *global_workspace,
                            std::size_t global_stride) {
  const int Nper = (N + num_blocks - 1) / num_blocks;
  RAJA::forall<Policy>(RAJA::RangeSegment(0, num_blocks), [=](int i) mutable {
    const auto words = (forall_nested_workspace_stride(workspace_nbytes) +
                        sizeof(std::max_align_t) - 1) /
                       sizeof(std::max_align_t);
    std::vector<std::max_align_t> local(global_workspace ? 0 : words);
    auto *buffer = global_workspace ? global_workspace + i * global_stride
                                    : reinterpret_cast<char *>(local.data());
    func(i * Nper, std::min((i + 1) * Nper, N), buffer);
  });
}
#endif

#if defined(MFEM_USE_RAJA) &&                                               \
    ((defined(RAJA_ENABLE_CUDA) && defined(__CUDACC__)) ||                   \
     (defined(RAJA_ENABLE_HIP) && defined(__HIP__)))
template <class LaunchPolicy, class F>
void raja_gpu_nested_runner(F func, int N, int bx, int by, int bz,
                           int max_concurrent_blocks, int smem_nbytes,
                           char *global_workspace,
                           std::size_t global_stride) {
  const int blocks = forall_nested_num_blocks(N, max_concurrent_blocks);
  const int Nmult = (N + blocks - 1) / blocks;
  RAJA::launch<LaunchPolicy>(
      mfem::Device::GetRajaResource(),
      RAJA::LaunchParams(RAJA::Teams(blocks), RAJA::Threads(bx, by, bz),
                         global_workspace ? 0 : smem_nbytes),
      [=] RAJA_DEVICE(RAJA::LaunchContext ctx) mutable {
        const int block = MFEM_BLOCK_ID(x);
        const int Nstart = block * Nmult;
        const int Nstop = min((block + 1) * Nmult, N);
        auto *workspace = global_workspace
                              ? global_workspace + block * global_stride
                              : ctx.getSharedMemory<char>(smem_nbytes);
        func(Nstart, Nstop, workspace);
      });
}
#endif

/// Hierarchical nested parallel for over N. The provided functor is given a
/// range [Nstart, Nstop) over which they are responsible for performing a local
/// serial loop over.
///
/// On CPU: Serial execution, or OpenMP with at most min(N,
/// max_concurrent_blocks, omp_get_max_threads()) threads. Each thread is given
/// [Nstart, Nstop) it is responsible for. On GPU:
/// Assigns a (bx, by, bz) group of threads to a parallel for over min(N,
/// max_concurrent_blocks). Each block is given [Nstart, Nstop) they are
/// responsible for.
///
/// Nstart >= 0 and Nstop <= N. Rounded-up chunk sizes can give trailing
/// threads/blocks Nstart >= Nstop; functors must tolerate these empty ranges.
///
/// Functor signatures:
/// {host/dev}_func(Nstart, Nstop, char *workspace) -> void
template <int LB, class FH, class FD>
void forall_nested_launch_bounds(FH &&host_func, FD &&dev_func, int N, int bx,
                                 int by, int bz, int max_concurrent_blocks,
                                 int smem_nbytes, bool allow_device = true,
                                 char *global_workspace = nullptr,
                                 std::size_t global_stride = 0) {
  if (N) {
    MFEM_VERIFY(max_concurrent_blocks > 0,
                "max_concurrent_blocks must be positive");
    if (allow_device) {
#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_CUDA) && defined(__CUDACC__)
      if (mfem::Device::Allows(mfem::Backend::RAJA_CUDA)) {
        raja_gpu_nested_runner<cuda_launch_bounds_policy<LB>>(
            std::forward<FD>(dev_func), N, bx, by, bz, max_concurrent_blocks,
            smem_nbytes, global_workspace, global_stride);
        return;
      }
#endif
#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_HIP) && defined(__HIP__)
      if (mfem::Device::Allows(mfem::Backend::RAJA_HIP)) {
        raja_gpu_nested_runner<hip_launch_bounds_policy<LB>>(
            std::forward<FD>(dev_func), N, bx, by, bz, max_concurrent_blocks,
            smem_nbytes, global_workspace, global_stride);
        return;
      }
#endif
#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
      if (mfem::Device::Allows(mfem::Backend::CUDA | mfem::Backend::HIP)) {
        max_concurrent_blocks = std::min(max_concurrent_blocks, N);
        // compute Nmult
        int Nmult = (N + max_concurrent_blocks - 1) / max_concurrent_blocks;
        if (global_workspace) {
          gpu_global_runner<LB>
              <<<max_concurrent_blocks, dim3(bx, by, bz), 0, 0>>>(
                  std::forward<FD>(dev_func), N, Nmult, global_workspace,
                  global_stride);
        } else {
          gpu_shared_runner<LB>
              <<<max_concurrent_blocks, dim3(bx, by, bz), smem_nbytes, 0>>>(
                  std::forward<FD>(dev_func), N, Nmult);
        }
        return;
      }
#endif
#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)
      if (mfem::Device::Allows(mfem::Backend::RAJA_OMP)) {
        const int blocks =
            std::min({N, max_concurrent_blocks, omp_get_max_threads()});
        raja_host_nested_runner<RAJA::omp_parallel_for_exec>(
            std::forward<FH>(host_func), N, blocks, smem_nbytes,
            global_workspace, global_stride);
        return;
      }
#endif
#if defined(MFEM_USE_OPENMP) && defined(_OPENMP)
      if (mfem::Device::Allows(mfem::Backend::OMP)) {
        openmp_runner(std::forward<FH>(host_func), N, max_concurrent_blocks,
                      smem_nbytes, global_workspace, global_stride);
        return;
      }
#endif
#ifdef MFEM_USE_RAJA
      if (mfem::Device::Allows(mfem::Backend::RAJA_CPU)) {
        raja_host_nested_runner<RAJA::seq_exec>(
            std::forward<FH>(host_func), N, 1, smem_nbytes,
            global_workspace, global_stride);
        return;
      }
#endif
    }
    serial_runner(std::forward<FH>(host_func), N, smem_nbytes, global_workspace,
                  global_stride);
  }
}

template <int LB, class F>
void forall_nested_launch_bounds(F &&func, int N, int bx, int by, int bz,
                                 int max_concurrent_blocks, int smem_nbytes,
                                 bool allow_device = true) {
  forall_nested_launch_bounds<LB>(
      std::forward<F>(func), std::forward<F>(func), N, bx, by, bz,
      max_concurrent_blocks, smem_nbytes, allow_device);
}

/// Equivalent to forall_nested_launch_bounds, without applying a compile-time
/// launch-bounds constraint to the GPU kernel.
template <class FH, class FD>
void forall_nested(FH &&host_func, FD &&dev_func, int N, int bx, int by, int bz,
                   int max_concurrent_blocks, int smem_nbytes,
                   bool allow_device = true, char *global_workspace = nullptr,
                   std::size_t global_stride = 0) {
  if (N) {
    MFEM_VERIFY(max_concurrent_blocks > 0,
                "max_concurrent_blocks must be positive");
    if (allow_device) {
#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_CUDA) && defined(__CUDACC__)
      if (mfem::Device::Allows(mfem::Backend::RAJA_CUDA)) {
        raja_gpu_nested_runner<cuda_launch_policy>(
            std::forward<FD>(dev_func), N, bx, by, bz, max_concurrent_blocks,
            smem_nbytes, global_workspace, global_stride);
        return;
      }
#endif
#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_HIP) && defined(__HIP__)
      if (mfem::Device::Allows(mfem::Backend::RAJA_HIP)) {
        raja_gpu_nested_runner<hip_launch_policy>(
            std::forward<FD>(dev_func), N, bx, by, bz, max_concurrent_blocks,
            smem_nbytes, global_workspace, global_stride);
        return;
      }
#endif
#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
      if (mfem::Device::Allows(mfem::Backend::CUDA | mfem::Backend::HIP)) {
        max_concurrent_blocks = std::min(max_concurrent_blocks, N);
        const int Nmult =
            (N + max_concurrent_blocks - 1) / max_concurrent_blocks;
        if (global_workspace) {
          gpu_global_runner<<<max_concurrent_blocks, dim3(bx, by, bz), 0, 0>>>(
              std::forward<FD>(dev_func), N, Nmult, global_workspace,
              global_stride);
        } else {
          gpu_shared_runner<<<max_concurrent_blocks, dim3(bx, by, bz),
                              smem_nbytes, 0>>>(std::forward<FD>(dev_func), N,
                                               Nmult);
        }
        return;
      }
#endif
#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)
      if (mfem::Device::Allows(mfem::Backend::RAJA_OMP)) {
        const int blocks =
            std::min({N, max_concurrent_blocks, omp_get_max_threads()});
        raja_host_nested_runner<RAJA::omp_parallel_for_exec>(
            std::forward<FH>(host_func), N, blocks, smem_nbytes,
            global_workspace, global_stride);
        return;
      }
#endif
#if defined(MFEM_USE_OPENMP) && defined(_OPENMP)
      if (mfem::Device::Allows(mfem::Backend::OMP)) {
        openmp_runner(std::forward<FH>(host_func), N, max_concurrent_blocks,
                      smem_nbytes, global_workspace, global_stride);
        return;
      }
#endif
#ifdef MFEM_USE_RAJA
      if (mfem::Device::Allows(mfem::Backend::RAJA_CPU)) {
        raja_host_nested_runner<RAJA::seq_exec>(
            std::forward<FH>(host_func), N, 1, smem_nbytes,
            global_workspace, global_stride);
        return;
      }
#endif
    }
    serial_runner(std::forward<FH>(host_func), N, smem_nbytes, global_workspace,
                  global_stride);
  }
}

template <class F>
void forall_nested(F &&func, int N, int bx, int by, int bz,
                   int max_concurrent_blocks, int smem_nbytes,
                   bool allow_device = true) {
  forall_nested(std::forward<F>(func), std::forward<F>(func), N, bx, by, bz,
                max_concurrent_blocks, smem_nbytes, allow_device);
}

/// Launch using shared memory when workspace_nbytes fits in the portable 48 KiB
/// budget, otherwise use one aligned slice of global_workspace per launched
/// block.  global_workspace may be null only on the shared-memory path.  Its
/// required size is returned by forall_nested_global_workspace_size(), and it
/// must remain valid until the launched device work has completed.
template <int LB, class FH, class FD>
void forall_nested_launch_bounds_workspace(
    FH &&host_func, FD &&dev_func, int N, int bx, int by, int bz,
    int max_concurrent_blocks, int workspace_nbytes, char *global_workspace,
    std::size_t global_workspace_nbytes, bool allow_device = true) {
  MFEM_VERIFY(workspace_nbytes >= 0, "workspace size must be nonnegative");
  const bool gpu =
#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
      allow_device &&
      mfem::Device::Allows(mfem::Backend::CUDA | mfem::Backend::HIP |
                           mfem::Backend::RAJA_CUDA | mfem::Backend::RAJA_HIP);
#else
      false;
#endif
  const bool use_global =
      gpu && N > 0 && workspace_nbytes > forall_nested_shared_memory_limit;
  const auto stride = forall_nested_workspace_stride(workspace_nbytes);
  const auto required = forall_nested_global_workspace_size(
      N, max_concurrent_blocks, workspace_nbytes);
  MFEM_VERIFY(!use_global || global_workspace,
              "global workspace is required above the 48 KiB shared-memory "
              "limit");
  MFEM_VERIFY(!use_global || global_workspace_nbytes >= required,
              "global workspace is too small");
  forall_nested_launch_bounds<LB>(
      std::forward<FH>(host_func), std::forward<FD>(dev_func), N, bx, by, bz,
      max_concurrent_blocks, use_global ? 0 : workspace_nbytes, allow_device,
      use_global ? global_workspace : nullptr, use_global ? stride : 0);
}

template <int LB, class F>
void forall_nested_launch_bounds_workspace(F &&func, int N, int bx, int by,
                                           int bz, int max_concurrent_blocks,
                                           int workspace_nbytes,
                                           char *global_workspace,
                                           std::size_t global_workspace_nbytes,
                                           bool allow_device = true) {
  forall_nested_launch_bounds_workspace<LB>(
      std::forward<F>(func), std::forward<F>(func), N, bx, by, bz,
      max_concurrent_blocks, workspace_nbytes, global_workspace,
      global_workspace_nbytes, allow_device);
}

/// Equivalent to forall_nested_launch_bounds_workspace, without applying a
/// compile-time launch-bounds constraint to the GPU kernel.
template <class FH, class FD>
void forall_nested_workspace(FH &&host_func, FD &&dev_func, int N, int bx,
                             int by, int bz, int max_concurrent_blocks,
                             int workspace_nbytes, char *global_workspace,
                             std::size_t global_workspace_nbytes,
                             bool allow_device = true) {
  MFEM_VERIFY(workspace_nbytes >= 0, "workspace size must be nonnegative");
  const bool gpu =
#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
      allow_device &&
      mfem::Device::Allows(mfem::Backend::CUDA | mfem::Backend::HIP |
                           mfem::Backend::RAJA_CUDA | mfem::Backend::RAJA_HIP);
#else
      false;
#endif
  const bool use_global =
      gpu && N > 0 && workspace_nbytes > forall_nested_shared_memory_limit;
  const auto stride = forall_nested_workspace_stride(workspace_nbytes);
  const auto required = forall_nested_global_workspace_size(
      N, max_concurrent_blocks, workspace_nbytes);
  MFEM_VERIFY(!use_global || global_workspace,
              "global workspace is required above the 48 KiB shared-memory "
              "limit");
  MFEM_VERIFY(!use_global || global_workspace_nbytes >= required,
              "global workspace is too small");
  forall_nested(std::forward<FH>(host_func), std::forward<FD>(dev_func), N, bx,
                by, bz, max_concurrent_blocks,
                use_global ? 0 : workspace_nbytes, allow_device,
                use_global ? global_workspace : nullptr,
                use_global ? stride : 0);
}

template <class F>
void forall_nested_workspace(F &&func, int N, int bx, int by, int bz,
                             int max_concurrent_blocks, int workspace_nbytes,
                             char *global_workspace,
                             std::size_t global_workspace_nbytes,
                             bool allow_device = true) {
  forall_nested_workspace(
      std::forward<F>(func), std::forward<F>(func), N, bx, by, bz,
      max_concurrent_blocks, workspace_nbytes, global_workspace,
      global_workspace_nbytes, allow_device);
}
} // namespace mfem

#endif
