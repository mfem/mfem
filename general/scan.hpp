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

#ifndef MFEM_SCAN_HPP
#define MFEM_SCAN_HPP

#include "backends.hpp"

#ifdef MFEM_USE_CUDA
#include <cub/device/device_scan.cuh>
#define MFEM_CUB_NAMESPACE cub
#elif defined(MFEM_USE_HIP)
#include <hipcub/device/device_scan.hpp>
#define MFEM_CUB_NAMESPACE hipcub
#endif

#include <functional>
#include <numeric>
#include <cstddef>

namespace mfem
{
/// Equivalent to InclusiveScan(use_dev, d_in, d_out, num_items, workspace,
/// std::plus<>{})
template <class InputIt, class OutputIt>
void InclusiveScan(bool use_dev, InputIt d_in, OutputIt d_out, size_t num_items)
{
   // forward to InclusiveSum for potentially faster kernels
#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
   if (use_dev && mfem::Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
   {
      static Array<std::byte> workspace;
      size_t bytes = workspace.Size();
      if (bytes)
      {
         auto err = MFEM_CUB_NAMESPACE::DeviceScan::InclusiveSum(
                       workspace.Write(), bytes, d_in, d_out, num_items);
#if defined(MFEM_USE_CUDA)
         if (err == cudaSuccess)
         {
            return;
         }
#elif defined(MFEM_USE_HIP)
         if (err == hipSuccess)
         {
            return;
         }
#endif
      }
      // try allocating a larger buffer
      bytes = 0;
      MFEM_GPU_CHECK(MFEM_CUB_NAMESPACE::DeviceScan::InclusiveSum(
                        nullptr, bytes, d_in, d_out, num_items));
      workspace.SetSize(bytes);
      MFEM_GPU_CHECK(MFEM_CUB_NAMESPACE::DeviceScan::InclusiveSum(
                        workspace.Write(), bytes, d_in, d_out, num_items));
      return;
   }
#endif
#if 0
   std::inclusive_scan(d_in, d_in + num_items, d_out);
#else
   // work-around to some compilers not fully supporting C++17
   if (num_items)
   {
      *d_out = *d_in;
      auto prev = d_out;
      ++d_in;
      ++d_out;
      for (size_t i = 1; i < num_items; ++i)
      {
         *d_out = (*prev) + (*d_in);
         prev = d_out;
         ++d_in;
         ++d_out;
      }
   }
#endif
}

/// @brief Performs an inclusive scan of [d_in, d_in+num_items) -> [d_out,
/// d_out+num_items). This call is potentially asynchronous on the device.
///
/// @a d_in input start.
/// @a d_out output start. Can perform in-place scans with d_out = d_in
/// @a scan_op binary scan functor. Must be associative. If only weakly
/// associative (i.e. floating point addition) results are not deterministic. On
/// device this must also be commutative.
template <class InputIt, class OutputIt, class ScanOp>
void InclusiveScan(bool use_dev, InputIt d_in, OutputIt d_out, size_t num_items,
                   ScanOp scan_op)
{
#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
   if (use_dev && mfem::Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
   {
      static Array<std::byte> workspace;
      size_t bytes = workspace.Size();
      if (bytes)
      {
         auto err = MFEM_CUB_NAMESPACE::DeviceScan::InclusiveScan(
                       workspace.Write(), bytes, d_in, d_out, scan_op, num_items);
#if defined(MFEM_USE_CUDA)
         if (err == cudaSuccess)
         {
            return;
         }
#elif defined(MFEM_USE_HIP)
         if (err == hipSuccess)
         {
            return;
         }
#endif
      }
      // try allocating a larger buffer
      bytes = 0;
      MFEM_GPU_CHECK(MFEM_CUB_NAMESPACE::DeviceScan::InclusiveScan(
                        nullptr, bytes, d_in, d_out, scan_op, num_items));
      workspace.SetSize(bytes);
      MFEM_GPU_CHECK(MFEM_CUB_NAMESPACE::DeviceScan::InclusiveScan(
                        workspace.Write(), bytes, d_in, d_out, scan_op, num_items));
      return;
   }
#endif
#if 0
   std::inclusive_scan(d_in, d_in + num_items, d_out, scan_op);
#else
   // work-around to some compilers not fully supporting C++17
   if (num_items)
   {
      *d_out = *d_in;
      auto prev = d_out;
      ++d_in;
      ++d_out;
      for (size_t i = 1; i < num_items; ++i)
      {
         *d_out = scan_op(*prev, *d_in);
         prev = d_out;
         ++d_in;
         ++d_out;
      }
   }
#endif
}

/// Performs an exclusive scan of [d_in, d_in+num_items) -> [d_out,
/// d_out+num_items). This call is potentially asynchronous on the device.
/// @a d_in input start.
/// @a d_out output start. Can perform in-place scans with d_out = d_in
/// @a scan_op binary scan functor. Must be associative. If only weakly
/// associative (i.e. floating point addition) results are not deterministic. On
/// device this must also be commutative.
template <class InputIt, class OutputIt, class T, class ScanOp>
void ExclusiveScan(bool use_dev, InputIt d_in, OutputIt d_out, size_t num_items,
                   T init_value, ScanOp scan_op)
{
#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
   if (use_dev && mfem::Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
   {
      static Array<std::byte> workspace;
      size_t bytes = workspace.Size();
      if (bytes)
      {
         auto err = MFEM_CUB_NAMESPACE::DeviceScan::ExclusiveScan(
                       workspace.Write(), bytes, d_in, d_out, scan_op, init_value,
                       num_items);
#if defined(MFEM_USE_CUDA)
         if (err == cudaSuccess)
         {
            return;
         }
#elif defined(MFEM_USE_HIP)
         if (err == hipSuccess)
         {
            return;
         }
#endif
      }
      // try allocating a larger buffer
      bytes = 0;
      MFEM_GPU_CHECK(MFEM_CUB_NAMESPACE::DeviceScan::ExclusiveScan(
                        nullptr, bytes, d_in, d_out, scan_op, init_value, num_items));
      workspace.SetSize(bytes);
      MFEM_GPU_CHECK(MFEM_CUB_NAMESPACE::DeviceScan::ExclusiveScan(
                        workspace.Write(), bytes, d_in, d_out, scan_op, init_value,
                        num_items));
      return;
   }
#endif
#if 0
   std::exclusive_scan(d_in, d_in + num_items, d_out, init_value, scan_op);
#else
   // work-around to some compilers not fully supporting C++17
   if (num_items)
   {
      for (size_t i = 0; i < num_items; ++i)
      {
         auto next = scan_op(init_value, *d_in);
         *d_out = init_value;
         init_value = next;
         ++d_out;
         ++d_in;
      }
   }
#endif
}

/// Equivalent to ExclusiveScan(use_dev, d_in, d_out, num_items, init_value,
/// workspace, std::plus<>{})
template <class InputIt, class OutputIt, class T>
void ExclusiveScan(bool use_dev, InputIt d_in, OutputIt d_out, size_t num_items,
                   T init_value)
{
   ExclusiveScan(use_dev, d_in, d_out, num_items, init_value, std::plus<> {});
}

} // namespace mfem

#undef MFEM_CUB_NAMESPACE

#endif
